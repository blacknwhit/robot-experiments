"""Shared initialization for the new, policy-independent fixed-scene protocol."""
import random
from types import SimpleNamespace

import numpy as np

PROTOCOL = 'fresh_environment_per_episode_v1'


def create_scene(task, horizon, scene_seed):
    # Reset before construction too: construction can sample objects / textures.
    # A previous policy's actions must never influence the next scene.
    from examples.pi05_robocasa365.eval_files.simulation_env import create_single_env
    random.seed(scene_seed)
    np.random.seed(scene_seed)
    args = SimpleNamespace(env_name=task, env_split='pretrain', seed=scene_seed,
                           video_out_path=None, n_action_steps=5, max_episode_steps=horizon)
    env = create_single_env(args, 0)
    try:
        observation, info = env.reset()
        return env, observation, info
    except BaseException:
        env.close()
        raise


def fingerprints(env, observation):
    from episode_plan import observation_hash, mujoco_state_hash
    return {'observation_sha256': observation_hash(observation),
            'state_sha256': mujoco_state_hash(env.unwrapped.env)}


def verify_scene(row, actual):
    expected = {'observation_sha256': row['initial_observation_sha256'],
                'state_sha256': row['initial_state_sha256']}
    if actual != expected:
        raise ValueError(f"Initial scene differs for episode {row['episode']}: "
                         f"expected={expected}, actual={actual}")


def batch_observation(observation):
    # Match SyncVectorEnv's leading batch dimension without its automatic reset.
    return {key: np.expand_dims(np.asarray(value), 0) for key, value in observation.items()}


def unbatch_action(actions):
    if any(np.asarray(value).shape[0] != 1 for value in actions.values()):
        raise ValueError('The fixed-scene worker requires batch size one')
    return {key: value[0] for key, value in actions.items()}
