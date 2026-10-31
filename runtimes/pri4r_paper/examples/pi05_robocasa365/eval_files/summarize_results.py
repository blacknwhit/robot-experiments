"""Aggregate PI0.5 RoboCasa365 task results and compare with the publication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from robocasa.utils.dataset_registry import TARGET_TASKS


PUBLISHED_REFERENCE = {
    "atomic_seen": 39.6,
    "composite_seen": 7.1,
    "composite_unseen": 1.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--task-sets", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="*", required=True, help="Entries formatted as task_set:task")
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--protocol-comparable", type=int, choices=(0, 1), required=True)
    parser.add_argument("--robocasa-version", required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--n-action-steps", type=int, required=True)
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--horizon-multiplier", type=float, required=True)
    parser.add_argument("--ld-preload", default="")
    parser.add_argument("--gpu-ids", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-leaderboard-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {name: [] for name in args.task_sets}
    for item in args.tasks:
        task_set, separator, task = item.partition(":")
        if not separator or task_set not in selected or task not in TARGET_TASKS[task_set]:
            raise ValueError(f"Invalid task selector: {item!r}")
        selected[task_set].append(task)

    rows = []
    incomplete = []
    for task_set in args.task_sets:
        for task in selected[task_set]:
            path = args.results_dir / task_set / f"{task}.json"
            if not path.is_file():
                incomplete.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            successes = [bool(value) for value in payload.get("successes", [])]
            metadata_ok = (
                payload.get("profile") == args.profile
                and payload.get("checkpoint") == args.checkpoint
                and payload.get("robocasa_version") == args.robocasa_version
                and payload.get("n_action_steps") == args.n_action_steps
                and payload.get("n_envs") == args.n_envs
                and payload.get("horizon_multiplier") == args.horizon_multiplier
                and payload.get("ld_preload", "") == args.ld_preload
            )
            if len(successes) != args.expected_trials or not metadata_ok:
                incomplete.append(f"{path} (trial count or metadata mismatch)")
                continue
            rows.append(
                {
                    "task_set": task_set,
                    "task": task,
                    "num_successes": sum(successes),
                    "num_trials": len(successes),
                    "success_rate": fmean(successes),
                    "result_path": str(path),
                }
            )

    groups = {}
    for task_set in args.task_sets:
        group_rows = [row for row in rows if row["task_set"] == task_set]
        rate = fmean(row["success_rate"] for row in group_rows) if group_rows else None
        percentage = None if rate is None else round(100.0 * rate, 4)
        reference = PUBLISHED_REFERENCE.get(task_set)
        groups[task_set] = {
            "completed_tasks": len(group_rows),
            "expected_tasks": len(selected[task_set]),
            "macro_success_rate": rate,
            "success_percent": percentage,
            "published_pi05_percent": reference,
            "delta_from_published_points": (
                None if percentage is None or reference is None else round(percentage - reference, 4)
            ),
        }

    full_official_tasks = all(selected.get(name) == list(TARGET_TASKS[name]) for name in PUBLISHED_REFERENCE)
    directly_comparable = (
        bool(args.protocol_comparable)
        and args.profile == "published_v1_0_0"
        and args.robocasa_version == "1.0.0"
        and args.expected_trials == 50
        and set(args.task_sets) == set(PUBLISHED_REFERENCE)
        and full_official_tasks
    )
    protocol_issues = []
    if not directly_comparable:
        protocol_issues.append("This run is diagnostic and is not directly comparable to the published PI0.5 row.")
    if incomplete:
        protocol_issues.append("One or more selected tasks are missing or have incompatible metadata.")

    summary = {
        "profile": args.profile,
        "directly_comparable_to_published_pi05": directly_comparable,
        "protocol_issues": protocol_issues,
        "checkpoint": args.checkpoint,
        "openpi_commit": args.openpi_commit,
        "robocasa_version": args.robocasa_version,
        "base_seed": args.base_seed,
        "n_action_steps": args.n_action_steps,
        "n_envs": args.n_envs,
        "horizon_multiplier": args.horizon_multiplier,
        "ld_preload": args.ld_preload,
        "gpu_ids": args.gpu_ids,
        "expected_trials_per_task": args.expected_trials,
        "groups": groups,
        "missing_or_incomplete": incomplete,
        "tasks": rows,
    }

    leaderboard = {
        "directly_comparable_to_published_pi05": directly_comparable,
        "robocasa_version": args.robocasa_version,
        "atomic_seen_success": groups.get("atomic_seen", {}).get("success_percent"),
        "composite_seen_success": groups.get("composite_seen", {}).get("success_percent"),
        "composite_unseen_success": groups.get("composite_unseen", {}).get("success_percent"),
        "published_reference": PUBLISHED_REFERENCE,
    }
    summary["leaderboard_metrics"] = leaderboard

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_leaderboard_json.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_set", "task", "num_successes", "num_trials", "success_rate", "result_path"),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== PI0.5 RoboCasa365 summary ===")
    for task_set, payload in groups.items():
        print(
            f"{task_set:20s}: {payload['success_percent']}% "
            f"({payload['completed_tasks']}/{payload['expected_tasks']} tasks), "
            f"published={payload['published_pi05_percent']}%"
        )
    print(f"Directly comparable: {directly_comparable}")
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
