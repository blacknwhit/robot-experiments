"""Contracts for fixed episode seeds without prerecorded scene fingerprints."""
import math
from pathlib import Path

import numpy as np
from common import ROOT
from episode_plan import file_sha
from scene_suite import DEFINITION, definition

PROTOCOL = 'fixed_episode_seeds_v1'
EXECUTION_VERSION = 'single_process_envs_async_policy_v1'
HEALTH_VERSION = 'rgb_shape_range_nonblack_and_finite_state_v1'
VIDEO_KEYS = ('video.robot0_agentview_left', 'video.robot0_eye_in_hand', 'video.robot0_agentview_right')


def seed_suite():
    data = definition()
    return {'protocol': PROTOCOL, 'scene_seed_definition_sha256': file_sha(DEFINITION),
            'task_registry_sha256': file_sha(ROOT / 'assets/eval_tasks.json'),
            'tasks': 50, 'scored_episodes': 2500, 'reference_plan_required': False}


def task_seeds(task):
    data = definition()
    name = task.removeprefix('robocasa/')
    spec = next((row for row in data['tasks'] if row['task'] == name), None)
    if spec is None:
        raise ValueError('Unknown task: ' + name)
    return spec, data['scene_seeds'][name][:50]


def selected_episodes(text=None):
    if text is None:
        return list(range(50))
    try:
        ids = [int(part.strip()) for part in text.split(',')]
    except ValueError as error:
        raise ValueError('Episode IDs must be comma-separated integers') from error
    if not ids or ids != sorted(set(ids)) or any(i < 0 or i >= 50 for i in ids):
        raise ValueError('Episode IDs must be unique, increasing and within 0..49')
    return ids


def partitions(ids, n_envs):
    if type(n_envs) is not int or not 1 <= n_envs <= min(50, len(ids)):
        raise ValueError('n-envs must be 1..min(50, selected episodes)')
    return [ids[i::n_envs] for i in range(n_envs)]


def check_observation(observation):
    """Reject unusable observations, without comparing pixels with any reference.

    No perceptual-distance threshold is imposed. A completely zero RGB camera
    frame, invalid image shape/range, or non-finite state is an execution error.
    """
    images = {}
    for key in VIDEO_KEYS:
        if key not in observation:
            raise ValueError('Missing camera: ' + key)
        image = np.asarray(observation[key])
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
            raise ValueError('Invalid RGB shape: ' + key + ' ' + str(image.shape))
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError('Invalid RGB values: ' + key)
        low, high = float(image.min()), float(image.max())
        if low < 0 or high > 255:
            raise ValueError('RGB values outside 0..255: ' + key)
        if high == 0:
            raise ValueError('All-zero RGB camera frame: ' + key)
        images[key] = {'shape': list(image.shape), 'dtype': str(image.dtype), 'min': low, 'max': high}
    states = [np.asarray(value) for key, value in observation.items() if key.startswith('state.')]
    if not states or any(not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all() for x in states):
        raise ValueError('Missing or non-finite robot state')
    return {'version': HEALTH_VERSION, 'images': images, 'finite_state': True}


def set_policy_episode(model, episode):
    # This path does not enable reference trajectory hashing. Do not call the
    # legacy select_episode helper, which assumes trace hashers were allocated.
    model._ensure_counters(1)
    model._episode_indices[0] = episode
    model._replan_indices[0] = 0


def validate_result(data, task, ids, n_envs, identity, diagnostic=False, max_replans=None):
    from simulator_renderer import VERSION as RGB_ADAPTER_VERSION
    if data.get('simulator_versions', {}).get('rgb_adapter') != RGB_ADAPTER_VERSION:
        raise ValueError('Missing native RGB adapter identity')
    spec, seeds = task_seeds(task)
    expected = {'env': 'robocasa/' + spec['task'], 'episode_protocol': PROTOCOL,
                'simulator_execution': EXECUTION_VERSION,
                'scene_seed_definition_sha256': identity['scene_seed_definition_sha256'],
                'episode_ids': ids, 'scene_seeds': [seeds[i] for i in ids], 'n_envs': n_envs,
                'num_trials': len(ids), 'logical_environment_index': 0, 'seed': 7,
                'max_episode_steps': spec['horizon'], 'n_action_steps': 5,
                'evaluation_kind': 'diagnostic' if diagnostic else 'formal',
                'diagnostic_max_replans': max_replans, 'reference_plan_required': False,
                'reference_observation_comparison': False, 'observation_health_version': HEALTH_VERSION}
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError('Seeded result contract mismatch: ' + key)
    rows = data['episodes']
    if [row['episode'] for row in rows] != ids:
        raise ValueError('Missing, duplicated or reordered episodes')
    for episode, row in zip(ids, rows):
        renderer = row.get('renderer', {})
        if (renderer.get('adapter') != RGB_ADAPTER_VERSION or renderer.get('backend') != 'egl'
                or renderer.get('engine') != 'mujoco' or renderer.get('msaa') != 4
                or renderer.get('max_geom') != 10000
                or renderer.get('gl_context_type') != 'mujoco.egl.GLContext'):
            raise ValueError('Unexpected RGB rendering configuration')
        if row['scene_seed'] != seeds[episode] or type(row['success']) is not bool:
            raise ValueError('Wrong episode seed or raw success')
        requests = row['requests']
        if type(requests) is not int or not 1 <= requests <= math.ceil(spec['horizon'] / 5):
            raise ValueError('Invalid request count')
        if row['health_checks'] != requests + 1 or row['initial_health']['version'] != HEALTH_VERSION:
            raise ValueError('Missing observation health evidence')
        if not row['initial_health']['finite_state'] or set(row['initial_health']['images']) != set(VIDEO_KEYS):
            raise ValueError('Incomplete initial observation evidence')
        for camera in row['initial_health']['images'].values():
            shape = camera.get('shape', [])
            if (len(shape) != 3 or shape[-1] != 3 or any(type(n) is not int or n < 1 for n in shape)
                    or not 0 <= camera['min'] <= camera['max'] <= 255 or camera['max'] == 0):
                raise ValueError('Invalid initial RGB health evidence')
        if type(row['episode_complete']) is not bool:
            raise ValueError('Missing terminal status')
        if not row['episode_complete'] and (max_replans is None or requests != max_replans):
            raise ValueError('Unexplained incomplete episode')
    completed = all(row['episode_complete'] for row in rows)
    if data.get('complete') is not completed:
        raise ValueError('Wrong completion marker')
    if not diagnostic and (not completed or ids != list(range(50)) or max_replans is not None):
        raise ValueError('Formal results require all 50 complete episodes')
    expected_successes = [row['success'] for row in rows] if completed else None
    if data.get('successes') != expected_successes or data.get('num_successes') != (sum(expected_successes) if completed else None):
        raise ValueError('Invalid successes or scoring of incomplete diagnostic')
    if data.get('requests') != sum(row['requests'] for row in rows):
        raise ValueError('Invalid total request count')
    metadata = data['policy_server_metadata']
    for key in ('variant', 'checkpoint_sha256', 'config_sha256', 'inference_backend', 'inference_backend_version'):
        if metadata.get(key) != identity[key]:
            raise ValueError('Policy identity mismatch: ' + key)
    if metadata.get('strict_full_state', {}).get('all_loaded_values_exact_at_model_dtype') is not True:
        raise ValueError('Strict full-state load missing')
    if metadata.get('policy_rng_mode') != 'stateless_request_v1' or metadata.get('policy_seed_version') != 'sha256_u32_v1':
        raise ValueError('Policy seed contract changed')
    return data
