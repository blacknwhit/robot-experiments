"""One-observation PI0.5 preflight; this never steps the simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import robocasa
import robocasa.wrappers.gym_wrapper  # noqa: F401
from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter

from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import PolicyWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--split", default="pretrain", choices=("pretrain", "target"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def add_batch_and_time_axes(observation: dict) -> dict:
    output = {}
    for key, value in observation.items():
        if key.startswith(("video.", "state.")):
            output[key] = np.asarray(value)[None, None]
        elif key.startswith("annotation."):
            output[key] = np.asarray([value])
    return output


def main() -> None:
    args = parse_args()
    env = gym.make("robocasa/OpenDrawer", enable_render=True, split=args.split, seed=args.seed)
    model = PolicyWrapper(host=args.host, port=args.port, n_action_steps=args.n_action_steps)
    try:
        observation, _ = env.reset()
        output = model.step(add_batch_and_time_axes(observation))["actions"]
        first_action = {
            key: np.asarray(value[0, 0], dtype=env.action_space[key].dtype)
            for key, value in output.items()
        }
        contains = bool(env.action_space.contains(first_action))
        robot_action = PandaOmronKeyConverter.unmap_action(first_action)
        unmap_finite = all(bool(np.all(np.isfinite(value))) for value in robot_action.values())
        shapes = {key: list(value.shape) for key, value in output.items()}
        finite = all(bool(np.all(np.isfinite(value))) for value in output.values())
        report = {
            "robocasa_version": robocasa.__version__,
            "env": "robocasa/OpenDrawer",
            "simulator_stepped": False,
            "action_shapes": shapes,
            "all_actions_finite": finite,
            "first_action_in_space": contains,
            "pandaomron_unmap_succeeded": unmap_finite,
        }
        if not finite:
            raise RuntimeError("Preflight actions contain NaN or Inf")
        if not unmap_finite:
            raise RuntimeError(f"PandaOmron action conversion produced non-finite values: {robot_action}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        model.close()
        env.close()


if __name__ == "__main__":
    main()
