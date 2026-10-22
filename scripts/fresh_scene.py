"""Shared initialization for the new, policy-independent fixed-scene protocol."""
import random
import hashlib
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np

PROTOCOL = 'fresh_environment_stable_regions_v1'
COUNTER_METHOD_SHA256 = '3a70cf8f230155ded44711da609746d97d1679426478b080671a166e14c6ea9e'


def _stable_region_method(original, expected_sha):
    source = textwrap.dedent(inspect.getsource(original))
    old = 'valid_geoms = list(set(valid_geoms))'
    if (hashlib.sha256(source.encode()).hexdigest() != expected_sha
            or source.count(old) != 1 or original.__closure__ is not None):
        raise ValueError('Unsupported Counter.get_reset_regions source; do not guess a patch')
    # XML Element hashes use object identity. Preserve the XML-derived order,
    # keeping the same unique candidate set and uniform selection probabilities.
    source = source.replace(old, 'valid_geoms = list(dict.fromkeys(valid_geoms))')
    namespace = dict(original.__globals__)
    exec(compile(source, original.__code__.co_filename + '#fixed-scene-stable-order', 'exec'), namespace)
    repaired = namespace[original.__name__]
    repaired.__qualname__ = original.__qualname__
    repaired._fixed_scene_original_sha256 = expected_sha
    return repaired


def install_stable_regions():
    from robocasa.models.fixtures.counter import Counter
    original = Counter.get_reset_regions
    if getattr(original, '_fixed_scene_original_sha256', None) == COUNTER_METHOD_SHA256:
        return
    Counter.get_reset_regions = _stable_region_method(original, COUNTER_METHOD_SHA256)


def create_scene(task, horizon, scene_seed):
    # Reset before construction too: construction can sample objects / textures.
    # A previous policy's actions must never influence the next scene.
    from examples.pi05_robocasa365.eval_files.simulation_env import create_single_env
    install_stable_regions()
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
