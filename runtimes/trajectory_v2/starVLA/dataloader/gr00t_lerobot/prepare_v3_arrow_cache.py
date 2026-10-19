"""Prebuild the shared Arrow cache used by StarVLA's LeRobot v3 reader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from starVLA.dataloader.gr00t_lerobot.v3_reader import (
    LeRobotV3ArrowReader,
    default_v3_arrow_cache_dir,
)


def expected_num_rows(dataset_path: Path) -> int:
    metadata_files = sorted((dataset_path / "meta" / "episodes").glob("*/*.parquet"))
    if not metadata_files:
        raise FileNotFoundError(f"No LeRobot v3 episode metadata below {dataset_path}")

    maximum = 0
    total_episode_rows = 0
    for path in metadata_files:
        metadata = pd.read_parquet(
            path,
            columns=["dataset_from_index", "dataset_to_index", "length"],
        )
        lengths = metadata["dataset_to_index"] - metadata["dataset_from_index"]
        if not (lengths.to_numpy() == metadata["length"].to_numpy()).all():
            raise ValueError(f"Inconsistent episode row ranges in {path}")
        maximum = max(maximum, int(metadata["dataset_to_index"].max()))
        total_episode_rows += int(metadata["length"].sum())

    if maximum != total_episode_rows:
        raise ValueError(
            f"Episode ranges are not contiguous below {dataset_path}: "
            f"max endpoint={maximum}, summed lengths={total_episode_rows}"
        )
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        type=Path,
        help="One or more aggregate LeRobot v3 dataset directories.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_v3_arrow_cache_dir(),
        help="Persistent Arrow cache root shared by all ranks and workers.",
    )
    args = parser.parse_args()

    results = []
    for dataset_path in args.dataset_paths:
        dataset_path = dataset_path.expanduser().resolve()
        reader = LeRobotV3ArrowReader(
            dataset_path,
            cache_dir=args.cache_dir,
            expected_num_rows=expected_num_rows(dataset_path),
        )
        results.append(reader.prepare())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
