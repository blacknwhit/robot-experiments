"""Fast, process-safe reader for aggregate LeRobot v3 Parquet datasets.

LeRobot v3 aggregate files may contain many episodes in one Parquet row group.
Reading a whole file for every sampled episode therefore wastes most of the I/O.
This module materializes Hugging Face Datasets' Arrow cache once, memory-maps it
per worker process, and retrieves an episode by its global contiguous row range.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


_SHARED_DATASETS: dict[tuple[int, tuple[str, ...], str], Any] = {}
_SHARED_DATASETS_LOCK = threading.Lock()


def default_v3_arrow_cache_dir() -> Path:
    """Return the configurable persistent cache root used by v3 readers."""
    configured = os.environ.get("STARVLA_LEROBOT_V3_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "starvla" / "lerobot_v3_arrow"


def _import_arrow_cache_dependencies():
    try:
        from datasets import Dataset as HuggingFaceDataset
        from datasets.table import ConcatenationTable, MemoryMappedTable
        from filelock import FileLock
    except ImportError as exc:
        raise ImportError(
            "LeRobot v3 Arrow reading requires the `datasets` and `filelock` "
            "packages. Install the StarVLA PI0.5 environment or add those "
            "packages to the active environment."
        ) from exc
    return HuggingFaceDataset, MemoryMappedTable, ConcatenationTable, FileLock


class LeRobotV3ArrowReader:
    """Read v3 episodes from a shared memory-mapped Arrow cache.

    The reader intentionally opens the Arrow dataset lazily. This prevents a
    parent training process from handing live mmap/file objects to DataLoader
    workers and lets every rank/worker reopen the same persistent cache safely.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        *,
        data_files: Sequence[Path | str] | None = None,
        cache_dir: Path | str | None = None,
        expected_num_rows: int | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if data_files is None:
            data_files = sorted(self.dataset_path.glob("data/*/*.parquet"))
        self.data_files = tuple(Path(path).expanduser().resolve() for path in data_files)
        if not self.data_files:
            raise FileNotFoundError(f"No LeRobot v3 Parquet data below {self.dataset_path}")

        missing = [str(path) for path in self.data_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing LeRobot v3 Parquet files: {missing[:3]}")

        self.cache_dir = Path(cache_dir or default_v3_arrow_cache_dir()).expanduser().resolve()
        self.expected_num_rows = None if expected_num_rows is None else int(expected_num_rows)

        # Import the Arrow/Hugging Face modules while the training process is
        # still constructing the dataset, before wandb.init() installs its
        # post-import telemetry hooks. Accelerate may defer creation of the
        # underlying DataLoader iterator until the first next() call; without
        # this preload, forked workers perform the imports after W&B starts and
        # inherit a process-bound W&B asyncio manager. Importing modules here is
        # safe: mmap tables and Dataset objects remain lazy and are still opened
        # separately in each worker by _open_dataset().
        _import_arrow_cache_dependencies()

        self._runtime_dataset = None
        self._runtime_pid: int | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_runtime_dataset"] = None
        state["_runtime_pid"] = None
        return state

    @property
    def _cache_identity(self) -> str:
        source_paths = "\n".join(str(path) for path in self.data_files).encode("utf-8")
        return hashlib.sha256(source_paths).hexdigest()[:20]

    @property
    def _manifest_path(self) -> Path:
        return self.cache_dir / "manifests" / f"{self._cache_identity}.json"

    def _source_file_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in self.data_files
        ]

    def _validate_num_rows(self, dataset: Any) -> None:
        if self.expected_num_rows is not None and len(dataset) != self.expected_num_rows:
            raise ValueError(
                "LeRobot v3 Arrow row count mismatch for "
                f"{self.dataset_path}: expected {self.expected_num_rows}, got {len(dataset)}"
            )

    def _load_manifest_dataset(self):
        """Open prepared Arrow shards directly, without invoking HF's Parquet builder."""
        manifest_path = self._manifest_path
        if not manifest_path.is_file():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if manifest.get("format_version") != 1:
            return None
        if manifest.get("source_files") != self._source_file_metadata():
            return None

        arrow_files = [Path(path) for path in manifest.get("arrow_files", [])]
        if not arrow_files or any(not path.is_file() for path in arrow_files):
            return None

        (
            HuggingFaceDataset,
            MemoryMappedTable,
            ConcatenationTable,
            _,
        ) = _import_arrow_cache_dependencies()
        tables = [MemoryMappedTable.from_file(str(path)) for path in arrow_files]
        table = tables[0] if len(tables) == 1 else ConcatenationTable.from_tables(tables)
        # Dataset.from_file() still calls generate_fingerprint(), whose dill
        # traversal can recurse through Accelerate / DeepSpeed worker globals.
        # The manifest already contains a complete stable identity, so provide
        # its digest explicitly and bypass all automatic fingerprinting.
        fingerprint = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        dataset = HuggingFaceDataset(table, fingerprint=fingerprint)
        self._validate_num_rows(dataset)
        return dataset

    def _write_manifest(self, dataset: Any) -> None:
        arrow_files = [entry.get("filename") for entry in dataset.cache_files]
        if not arrow_files or any(not filename for filename in arrow_files):
            raise RuntimeError(
                f"Hugging Face did not expose Arrow cache files for {self.dataset_path}"
            )

        manifest = {
            "format_version": 1,
            "dataset_path": str(self.dataset_path),
            "source_files": self._source_file_metadata(),
            "arrow_files": arrow_files,
            "num_rows": len(dataset),
        }
        manifest_path = self._manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)

    @staticmethod
    def _in_dataloader_worker() -> bool:
        try:
            from torch.utils.data import get_worker_info
        except ImportError:
            return False
        return get_worker_info() is not None

    @property
    def _shared_key(self) -> tuple[int, tuple[str, ...], str]:
        return (
            os.getpid(),
            tuple(str(path) for path in self.data_files),
            str(self.cache_dir),
        )

    def _open_dataset(self):
        pid = os.getpid()
        if self._runtime_pid == pid and self._runtime_dataset is not None:
            return self._runtime_dataset

        key = self._shared_key
        with _SHARED_DATASETS_LOCK:
            dataset = _SHARED_DATASETS.get(key)
            if dataset is None:
                dataset = self._load_manifest_dataset()
            if dataset is None:
                if self._in_dataloader_worker():
                    raise RuntimeError(
                        "LeRobot v3 Arrow cache manifest is missing or stale for "
                        f"{self.dataset_path}. Run prepare_v3_arrow_cache before "
                        "starting DataLoader workers."
                    )

                HuggingFaceDataset, _, _, FileLock = _import_arrow_cache_dependencies()
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                lock_name = self._cache_identity + ".lock"
                lock_path = self.cache_dir / ".locks" / lock_name
                lock_path.parent.mkdir(parents=True, exist_ok=True)

                # Only the first process builds the Arrow cache. Other ranks and
                # workers wait for the build, then mmap the completed cache.
                with FileLock(str(lock_path)):
                    dataset = self._load_manifest_dataset()
                    if dataset is None:
                        dataset = HuggingFaceDataset.from_parquet(
                            [str(path) for path in self.data_files],
                            cache_dir=str(self.cache_dir),
                            keep_in_memory=False,
                        )
                        self._validate_num_rows(dataset)
                        self._write_manifest(dataset)

            _SHARED_DATASETS[key] = dataset

        self._runtime_dataset = dataset
        self._runtime_pid = pid
        return dataset

    def prepare(self) -> dict[str, Any]:
        """Build/open the cache and return concise cache metadata."""
        dataset = self._open_dataset()
        return {
            "dataset_path": str(self.dataset_path),
            "cache_dir": str(self.cache_dir),
            "num_files": len(self.data_files),
            "num_rows": len(dataset),
            "cache_files": [entry.get("filename") for entry in dataset.cache_files],
        }

    def read_episode(
        self,
        *,
        trajectory_id: int,
        dataset_from_index: int,
        dataset_to_index: int,
    ) -> pd.DataFrame:
        """Return one episode using its half-open global row range."""
        start = int(dataset_from_index)
        stop = int(dataset_to_index)
        if start < 0 or stop <= start:
            raise ValueError(
                f"Invalid row range [{start}, {stop}) for trajectory {trajectory_id}"
            )

        dataset = self._open_dataset()
        if stop > len(dataset):
            raise IndexError(
                f"Row range [{start}, {stop}) exceeds dataset length {len(dataset)} "
                f"for trajectory {trajectory_id}"
            )

        # Slice the underlying Arrow table directly. Dataset.select() derives a
        # new fingerprint and would re-enter dill inside the worker.
        episode = dataset.data.slice(start, stop - start).to_pandas()
        if len(episode) != stop - start:
            raise ValueError(
                f"Episode {trajectory_id} length mismatch: expected {stop - start}, got {len(episode)}"
            )
        if "episode_index" not in episode.columns:
            raise KeyError("LeRobot v3 data has no episode_index column")

        episode_ids = np.asarray(episode["episode_index"].unique())
        if len(episode_ids) != 1 or int(episode_ids[0]) != int(trajectory_id):
            raise ValueError(
                f"Global row range [{start}, {stop}) resolved to episode IDs "
                f"{episode_ids.tolist()}, expected {trajectory_id}"
            )
        return episode


def clear_v3_reader_process_cache() -> None:
    """Clear only this process' shared reader objects (primarily for tests)."""
    pid = os.getpid()
    with _SHARED_DATASETS_LOCK:
        for key in [key for key in _SHARED_DATASETS if key[0] == pid]:
            del _SHARED_DATASETS[key]
