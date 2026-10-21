"""Run one RoboCasa365 task against an OpenPI PI0.5 websocket server."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

import robocasa
import robocasa.wrappers.gym_wrapper  # noqa: F401 - registers robocasa/* envs

from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import (
    POLICY_RNG_MODE,
    POLICY_SEED_VERSION,
    STRICT_DETERMINISM_VERSION,
    PolicyWrapper,
)
from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
from examples.Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--env-split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--protocol-comparable", type=int, choices=(0, 1), required=True)
    parser.add_argument("--horizon-multiplier", type=float, required=True)
    parser.add_argument("--ld-preload", default="")
    parser.add_argument("--strict-determinism", type=int, choices=(0, 1), default=0)
    parser.add_argument("--trace-determinism", type=int, choices=(0, 1), default=0)
    parser.add_argument("--determinism-version", default=STRICT_DETERMINISM_VERSION)
    parser.add_argument("--renderer-backend", required=True)
    parser.add_argument("--result-out-path", type=Path, required=True)
    parser.add_argument("--video-out-path", type=Path)
    return parser.parse_args()


def create_single_env(args: argparse.Namespace, index: int) -> gym.Env:
    env = gym.make(
        args.env_name,
        enable_render=True,
        split=args.env_split,
        seed=args.seed + index,
    )
    if args.video_out_path is not None:
        recorder = VideoRecorder.create_h264(
            fps=20,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        env = VideoRecordingWrapper(
            env,
            recorder,
            video_dir=args.video_out_path,
            steps_per_render=2,
        )
    return MultiStepWrapper(
        env,
        video_delta_indices=np.asarray([0]),
        state_delta_indices=np.asarray([0]),
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )


def run_simulation(model: PolicyWrapper, args: argparse.Namespace) -> Tuple[str, List[bool]]:
    print(
        f"[pi05 robocasa365] {args.env_name}: trials={args.num_trials}, "
        f"n_envs={args.n_envs}, replan={args.n_action_steps}, horizon={args.max_episode_steps}"
    )
    env_fns = [partial(create_single_env, args, index) for index in range(args.n_envs)]
    env = (
        gym.vector.SyncVectorEnv(env_fns)
        if args.n_envs == 1
        else gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    )

    completed = 0
    successes: List[bool] = []
    active_success = [False] * args.n_envs
    observations, _ = env.reset()
    started = time.time()
    args._rollout_started_at = started
    try:
        while completed < args.num_trials:
            action_dict = model.step(observations)["actions"]
            next_observations, rewards, terminated, truncated, _ = env.step(action_dict)
            done = np.logical_or(terminated, truncated)
            for index in range(args.n_envs):
                active_success[index] |= bool(rewards[index])
                if done[index]:
                    if completed >= args.num_trials:
                        break
                    successes.append(active_success[index])
                    active_success[index] = False
                    completed += 1
            model.finish_step(done)
            observations = next_observations
    finally:
        args._rollout_finished_at = time.time()
        model.close()
        env.close()

    print(f"[pi05 robocasa365] completed {completed} trials in {time.time() - started:.1f}s")
    return args.env_name, successes


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info(json.dumps(vars(args), indent=2, default=str))
    model = PolicyWrapper(
        host=args.host,
        port=args.port,
        image_size=(224, 224),
        n_action_steps=args.n_action_steps,
        task_name=args.env_name,
        base_seed=args.seed,
        strict_determinism=bool(args.strict_determinism),
        trace_determinism=bool(args.trace_determinism),
        determinism_version=args.determinism_version,
    )
    env_name, successes = run_simulation(model, args)
    success_rate = float(np.mean(successes)) if successes else 0.0
    episode_trace_hashes = model.episode_trace_hashes
    if args.trace_determinism and len(episode_trace_hashes) < len(successes):
        raise RuntimeError(
            f"Missing episode trace hashes: {len(episode_trace_hashes)} for {len(successes)} trials"
        )
    payload: Dict = {
        "env": env_name,
        "env_split": args.env_split,
        "robocasa_version": robocasa.__version__,
        "checkpoint": args.checkpoint,
        "profile": args.profile,
        "protocol_comparable": bool(args.protocol_comparable),
        "seed": args.seed,
        "n_envs": args.n_envs,
        "max_episode_steps": args.max_episode_steps,
        "horizon_multiplier": args.horizon_multiplier,
        "n_action_steps": args.n_action_steps,
        "ld_preload": args.ld_preload,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "policy_rng_mode": POLICY_RNG_MODE,
        "policy_seed_version": POLICY_SEED_VERSION,
        "strict_determinism": bool(args.strict_determinism),
        "trace_determinism": bool(args.trace_determinism),
        "determinism_version": args.determinism_version,
        "renderer_backend": args.renderer_backend,
        "policy_server_metadata": model.server_metadata,
        "episode_trace_sha256": episode_trace_hashes[: len(successes)],
        "success_rate": success_rate,
        "num_successes": int(sum(successes)),
        "rollout_timing": {"started_at": args._rollout_started_at,
                           "finished_at": args._rollout_finished_at},
        "num_trials": len(successes),
        "successes": [bool(value) for value in successes],
    }
    args.result_out_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"{env_name}: {sum(successes)}/{len(successes)} "
        f"({100.0 * success_rate:.2f}%) -> {args.result_out_path}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
