"""Protocol profiles for diagnosing RoboCasa365 evaluation differences."""

from __future__ import annotations

import argparse
import dataclasses
import math
import shlex
import subprocess


@dataclasses.dataclass(frozen=True)
class EvaluationProfile:
    name: str
    robocasa_version: str
    robocasa_checkout: str
    replan_steps: int
    horizon_multiplier: float
    seed_mode: str
    n_envs: int
    description: str


PROFILES = {
    "published_v1_0_0": EvaluationProfile(
        name="published_v1_0_0",
        robocasa_version="1.0.0",
        robocasa_checkout="robocasa365_v1.0.0",
        replan_steps=5,
        horizon_multiplier=1.5,
        seed_mode="fixed",
        n_envs=1,
        description="Published PI0.5 protocol on RoboCasa 1.0.0.",
    ),
    "published_protocol_v1_0_1": EvaluationProfile(
        name="published_protocol_v1_0_1",
        robocasa_version="1.0.1",
        robocasa_checkout="robocasa365",
        replan_steps=5,
        horizon_multiplier=1.5,
        seed_mode="fixed",
        n_envs=1,
        description="Published PI0.5 protocol with only RoboCasa changed to 1.0.1.",
    ),
    "starvla_current_v1_0_1": EvaluationProfile(
        name="starvla_current_v1_0_1",
        robocasa_version="1.0.1",
        robocasa_checkout="robocasa365",
        replan_steps=8,
        horizon_multiplier=1.0,
        seed_mode="task_checksum",
        n_envs=8,
        description="Current StarVLA RoboCasa365 harness behavior on RoboCasa 1.0.1.",
    ),
}


def get_profile(name: str) -> EvaluationProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown profile {name!r}; choose from {sorted(PROFILES)}") from exc


def scaled_horizon(registry_horizon: int, multiplier: float) -> int:
    return int(math.ceil(int(registry_horizon) * float(multiplier)))


def task_seed(profile: EvaluationProfile, task: str, base_seed: int = 7) -> int:
    if profile.seed_mode == "fixed":
        return int(base_seed)
    if profile.seed_mode == "task_checksum":
        # Match the POSIX `cksum` command used by the current StarVLA launcher.
        output = subprocess.check_output(["cksum"], input=task.encode("utf-8"), text=False)
        checksum = int(output.split()[0])
        return int(base_seed) + checksum % 1_000_000
    raise ValueError(f"Unsupported seed mode: {profile.seed_mode}")


def print_shell(profile: EvaluationProfile) -> None:
    values = {
        "PROFILE_EXPECTED_ROBOCASA_VERSION": profile.robocasa_version,
        "PROFILE_ROBOCASA_CHECKOUT": profile.robocasa_checkout,
        "PROFILE_REPLAN_STEPS": str(profile.replan_steps),
        "PROFILE_HORIZON_MULTIPLIER": str(profile.horizon_multiplier),
        "PROFILE_SEED_MODE": profile.seed_mode,
        "PROFILE_N_ENVS": str(profile.n_envs),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--registry-horizon", type=int)
    args = parser.parse_args()
    profile = get_profile(args.profile)
    if args.task is not None:
        print(task_seed(profile, args.task, args.base_seed))
    elif args.registry_horizon is not None:
        print(scaled_horizon(args.registry_horizon, profile.horizon_multiplier))
    elif args.shell:
        print_shell(profile)
    else:
        print(dataclasses.asdict(profile))


if __name__ == "__main__":
    main()
