"""Raw StarVLA LeRobot schema for official OpenPI PI0.5 RoboCasa training."""

import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform


class Pi05PandaOmronRoboCasa365DataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_eye_in_hand",
        "video.robot0_agentview_right",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.base_position",
        "state.base_rotation",
        "state.gripper_qpos",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]
    language_keys = ["annotation.human.task_description"]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=[0], modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=list(range(50)), modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        # OpenPI owns all state/action normalization and model preprocessing.
        return ComposedModalityTransform(transforms=[])

    def make_dataset_mixture(
        self,
        *,
        data_root_dir,
        mixture_spec,
        delete_pause_frame,
        data_cfg,
    ):
        """Build the official 300-task OpenPI sampling population.

        The downloaded RoboCasa365 data is stored as two aggregate LeRobot v3
        bundles. Official OpenPI instead creates one dataset per task, applies
        ``100_demos`` with seed 0, and weights the resulting task datasets by
        ``len(dataset) ** 0.4``. Logical views restore those semantics without
        copying parquet or video files.
        """
        if not data_cfg.get("official_task_sampling", False):
            return None

        expected_names = {
            "robocasa365-pretrain-atomic",
            "robocasa365-pretrain-composite",
        }
        actual_names = {dataset_name for dataset_name, _, _ in mixture_spec}
        if actual_names != expected_names:
            raise ValueError(
                "PI0.5 official_task_sampling requires exactly the pretrain "
                f"atomic/composite aggregates, got {sorted(actual_names)}"
            )

        registry_file = Path(data_cfg.official_registry_file)
        task_names = _load_pretrain300_tasks(registry_file)
        filter_key = str(data_cfg.get("official_filter_key", "100_demos"))
        filter_seed = int(data_cfg.get("official_filter_seed", 0))
        weight_alpha = float(data_cfg.get("official_weight_alpha", 0.4))

        base_datasets = {}
        episode_tables = []
        for dataset_name, _, _ in mixture_spec:
            dataset_path = Path(data_root_dir) / dataset_name
            base_dataset = LeRobotSingleDataset(
                dataset_path=dataset_path,
                modality_configs=self.modality_config(),
                transforms=self.transform(),
                embodiment_tag=self.embodiment_tag,
                video_backend=data_cfg.get("video_backend", "decord"),
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
            )
            base_datasets[dataset_name] = base_dataset

            files = sorted((dataset_path / "meta" / "episodes").glob("*/*.parquet"))
            if not files:
                raise FileNotFoundError(f"No LeRobot v3 episode metadata below {dataset_path}")
            table = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
            required = {"episode_index", "length", "source_prefix", "source_episode_index"}
            missing = required - set(table.columns)
            if missing:
                raise ValueError(f"{dataset_path} episode metadata lacks {sorted(missing)}")
            table = table[list(required)].copy()
            table["dataset_name"] = dataset_name
            table["task_name"] = table["source_prefix"].map(_task_from_source_prefix)
            episode_tables.append(table)

        episodes = pd.concat(episode_tables, ignore_index=True)
        smoke_manifest = data_cfg.get('pri4r_smoke_manifest')
        if smoke_manifest:
            import json
            entries = json.loads(Path(smoke_manifest).read_text())['episodes']
            if not 1 <= len(entries) <= 8:
                raise ValueError('Bounded Pri4R smoke supports 1..8 real episodes')
            keys = {(e['current_dataset_name'], int(e['current_episode_index'])) for e in entries}
            episodes = episodes[[ (r.dataset_name, int(r.episode_index)) in keys
                                  for r in episodes.itertuples() ]].copy()
            if len(episodes) != len(keys):raise ValueError('Smoke episodes missing from current dataset')
            task_names = sorted(episodes['task_name'].unique())
        available_tasks = set(episodes["task_name"].unique())
        missing_tasks = [task for task in task_names if task not in available_tasks]
        extra_tasks = sorted(available_tasks - set(task_names))
        if missing_tasks or extra_tasks:
            raise ValueError(
                "Aggregate RoboCasa365 tasks do not match official pretrain300: "
                f"missing={missing_tasks}, extra={extra_tasks}"
            )

        views = []
        for task_name in task_names:
            task_rows = episodes.loc[episodes["task_name"] == task_name].copy()
            dataset_names = task_rows["dataset_name"].unique().tolist()
            if len(dataset_names) != 1:
                raise ValueError(f"Task {task_name} maps to datasets {dataset_names}")
            selected = task_rows if smoke_manifest else _filter_task_episodes(task_rows, filter_key, filter_seed)
            view = _TaskDatasetView(
                base_dataset=base_datasets[dataset_names[0]],
                task_name=task_name,
                trajectory_ids=selected["episode_index"].to_numpy(dtype=np.int64),
                trajectory_lengths=selected["length"].to_numpy(dtype=np.int64),
            )
            views.append(view)

        raw_weights = np.asarray([len(view) ** weight_alpha for view in views], dtype=np.float64)
        # Match GrootOpenpiMultiDataset: one original weight must be exactly 1.0.
        raw_weights /= raw_weights[0]
        print(
            "[pi05 official sampling] "
            f"tasks={len(views)} demos={sum(len(v.trajectory_ids) for v in views)} "
            f"frames={sum(len(v) for v in views)} filter={filter_key} "
            f"filter_seed={filter_seed} weight_alpha={weight_alpha}"
        )
        return list(zip(views, raw_weights.tolist()))


class _TaskDatasetView:
    """A task-filtered, zero-copy view over an aggregate LeRobot dataset."""

    def __init__(self, *, base_dataset, task_name, trajectory_ids, trajectory_lengths):
        if len(trajectory_ids) == 0:
            raise ValueError(f"Official task view {task_name} is empty")
        self._base_dataset = base_dataset
        self._dataset_name = f"{base_dataset.dataset_name}:{task_name}"
        self._trajectory_ids = trajectory_ids
        self._trajectory_lengths = trajectory_lengths
        self._num_steps = int(trajectory_lengths.sum())

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def trajectory_ids(self):
        return self._trajectory_ids

    @property
    def trajectory_lengths(self):
        return self._trajectory_lengths

    def __len__(self):
        return self._num_steps

    def __getattr__(self, name):
        base_dataset = self.__dict__.get("_base_dataset")
        if base_dataset is None:
            raise AttributeError(name)
        return getattr(base_dataset, name)


def _load_pretrain300_tasks(registry_file: Path) -> list[str]:
    """Read PRETRAINING_TASKS['pretrain300'] without importing RoboCasa."""
    if not registry_file.is_file():
        raise FileNotFoundError(f"Official RoboCasa registry not found: {registry_file}")
    tree = ast.parse(registry_file.read_text(encoding="utf-8"), filename=str(registry_file))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "PRETRAINING_TASKS"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call):
            break
        for keyword in node.value.keywords:
            if keyword.arg == "pretrain300":
                tasks = ast.literal_eval(keyword.value)
                if len(tasks) != 300 or len(set(tasks)) != 300:
                    raise ValueError(
                        f"Expected 300 unique pretrain300 tasks, got {len(tasks)} / {len(set(tasks))}"
                    )
                return tasks
    raise ValueError(f"Could not find PRETRAINING_TASKS['pretrain300'] in {registry_file}")


def _task_from_source_prefix(source_prefix: str) -> str:
    parts = str(source_prefix).strip("/").split("/")
    if len(parts) < 3 or parts[0] != "pretrain":
        raise ValueError(f"Unexpected RoboCasa365 source_prefix: {source_prefix!r}")
    return parts[2]


def _filter_task_episodes(table: pd.DataFrame, filter_key: str, seed: int) -> pd.DataFrame:
    """Mirror RoboCasa get_subset_demos_filter_key for one task."""
    try:
        num_demos = int(filter_key.split("_", 1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid official filter key: {filter_key!r}") from exc
    if num_demos <= 0:
        raise ValueError(f"official_filter_key must select at least one demo: {filter_key!r}")

    ordered = table.sort_values("source_episode_index", kind="stable")
    source_ids = ordered["source_episode_index"].astype(int).tolist()
    shuffled = list(source_ids)
    random.Random(seed).shuffle(shuffled)
    if num_demos < len(shuffled):
        selected_source_ids = set(shuffled[:num_demos])
        ordered = ordered.loc[ordered["source_episode_index"].isin(selected_source_ids)]
    # Official filtering preserves episode-file order after choosing the subset.
    return ordered.sort_values("episode_index", kind="stable")


_ROBOT_TYPE = "pi05_panda_omron_robocasa365"

ROBOT_TYPE_CONFIG_MAP = {
    _ROBOT_TYPE: Pi05PandaOmronRoboCasa365DataConfig(),
}


def _entry(dataset_name: str):
    return (dataset_name, 1.0, _ROBOT_TYPE)


DATASET_NAMED_MIXTURES = {
    "pi05_robocasa365_pretrain_atomic": [_entry("robocasa365-pretrain-atomic")],
    "pi05_robocasa365_pretrain_composite": [_entry("robocasa365-pretrain-composite")],
    "pi05_robocasa365_pretrain_all": [
        _entry("robocasa365-pretrain-atomic"),
        _entry("robocasa365-pretrain-composite"),
    ],
    "pi05_robocasa365_target_all": [
        _entry("robocasa365-target-atomic"),
        _entry("robocasa365-target-composite-seen"),
        _entry("robocasa365-target-composite-unseen"),
    ],
}
