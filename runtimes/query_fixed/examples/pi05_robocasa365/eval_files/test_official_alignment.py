"""Static and numeric checks against the fixed official OpenPI submission."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import (
    ACTION_SLICES,
    STATE_KEYS,
    VIDEO_KEYS,
    convert_to_uint8,
    resize_with_pad,
)


OPENPI_COMMIT = "ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8"
EXPECTED_VIDEO_KEYS = {
    "observation/image": "video.robot0_agentview_left",
    "observation/wrist_image": "video.robot0_eye_in_hand",
    "observation/right_image": "video.robot0_agentview_right",
}
EXPECTED_STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)
EXPECTED_ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_in_order(text: str, snippets: tuple[str, ...]) -> None:
    positions = [text.index(snippet) for snippet in snippets]
    assert positions == sorted(positions), (snippets, positions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openpi-home",
        type=Path,
        default=Path("playground/Code/openpi_robocasa365"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "playground/Checkpoints/pi05_pretrain_human300/multitask_learning/75000"
        ),
    )
    args = parser.parse_args()

    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={args.openpi_home.resolve()}", "-C", args.openpi_home, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert commit == OPENPI_COMMIT, commit
    assert VIDEO_KEYS == EXPECTED_VIDEO_KEYS
    assert STATE_KEYS == EXPECTED_STATE_KEYS
    assert ACTION_SLICES == EXPECTED_ACTION_SLICES

    official_main = (args.openpi_home / "examples/robocasa/main.py").read_text()
    assert_in_order(official_main, tuple(f'obs["{key}"]' for key in EXPECTED_VIDEO_KEYS.values()))
    assert_in_order(official_main, tuple(f'obs["{key}"]' for key in EXPECTED_STATE_KEYS))
    for snippet in (
        "seed: int = 7",
        'split: str = "pretrain"',
        "replan_steps: int = 5",
        "get_task_horizon(env_name) * 1.5",
        "action_chunk[: replan_steps]",
        "action = convert_action(action)",
        'done = info["success"]',
    ):
        assert snippet in official_main, snippet

    official_policy = (args.openpi_home / "src/openpi/policies/robocasa_policy.py").read_text()
    for snippet in (
        'data["observation/right_image"]',
        '"right_wrist_0_rgb": right_image',
        'np.asarray(data["actions"][:, :12])',
    ):
        assert snippet in official_policy, snippet

    official_images = load_module(
        args.openpi_home / "packages/openpi-client/src/openpi_client/image_tools.py",
        "official_openpi_image_tools",
    )
    rng = np.random.default_rng(7)
    images = (
        rng.integers(0, 256, size=(121, 257, 3), dtype=np.uint8),
        rng.random((121, 257, 3), dtype=np.float32),
        rng.uniform(0, 255, size=(121, 257, 3)).astype(np.float32),
        rng.random((224, 224, 3), dtype=np.float32),
    )
    for image in images:
        expected = official_images.convert_to_uint8(
            official_images.resize_with_pad(image, 224, 224)
        )
        actual = convert_to_uint8(resize_with_pad(image, 224, 224))
        assert np.array_equal(actual, expected), (image.shape, image.dtype)

    norm_stats = args.checkpoint / "assets/norm_stats.json"
    assert norm_stats.is_file(), norm_stats
    stats = json.loads(norm_stats.read_text())
    assert "norm_stats" in stats or stats, "empty norm_stats.json"
    params = [path for path in (args.checkpoint / "params").rglob("*") if path.is_file()]
    assert len(params) == 18, len(params)

    print("Official OpenPI alignment checks passed.")
    print(f"commit={commit}")
    print("cameras=left agentview, wrist, right agentview")
    print("state_dim=16 action_chunk=50x12 replan=5 seed=7 horizon=1.5x")
    print(f"checkpoint_params={len(params)} norm_stats={norm_stats}")


if __name__ == "__main__":
    main()
