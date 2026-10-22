"""Immutable, policy-independent scene definitions for all future evaluations.

Generation only resets environments; it does not load a policy or score episodes.
Prepared plans are specific to the verified simulator installation. Keep one plan
directory for every model in a comparison; never regenerate it per checkpoint.
"""
import argparse
import concurrent.futures
import copy
import json
import os
import random
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from common import ROOT, activate, environment, runtime
from episode_plan import SEEDED_FORMAT, atomic_json, build_plan_set, file_sha, load_plan, restore_rng

DEFINITION = ROOT / 'assets/eval_scene_seeds_v1.json'


def definition():
    data = json.loads(DEFINITION.read_text())
    tasks = json.loads((ROOT / 'assets/eval_tasks.json').read_text())
    if data['format'] != 'robocasa_scene_seeds_v1' or data['tasks'] != tasks:
        raise ValueError('Scene definition task registry changed')
    if data['num_trials'] != 50 or data['policy_seed'] != 7:
        raise ValueError('Scene definition protocol changed')
    seeds = data['scene_seeds']
    if set(seeds) != {t['task'] for t in tasks}:
        raise ValueError('Scene definition coverage changed')
    values = [s for task in tasks for s in seeds[task['task']]]
    if any(len(seeds[t['task']]) != 51 for t in tasks) or len(set(values)) != 2550:
        raise ValueError('Expected 50 scored scenes plus one reset sentinel per task')
    if any(type(s) is not int or not 0 <= s < 2**32 for s in values):
        raise ValueError('Invalid scene seed')
    return data


def seeded_rng(seed):
    """Create states without modifying the process-global random streams."""
    n = np.random.RandomState(seed).get_state()
    p = random.Random(seed).getstate()
    return {'env': copy.deepcopy(np.random.PCG64(seed).state),
            'numpy': [n[0], n[1].tolist(), *n[2:]], 'python': [p[0], list(p[1]), p[2]]}


def validate_seeded_plan(plan):
    data = definition()
    task = plan['env'].removeprefix('robocasa/')
    spec = next((t for t in data['tasks'] if t['task'] == task), None)
    if spec is None or plan.get('suite_definition_sha256') != file_sha(DEFINITION):
        raise ValueError('Seeded plan belongs to another scene definition')
    if plan['num_trials'] != 50 or plan['max_episode_steps'] != spec['horizon']:
        raise ValueError('Seeded plan episode count or horizon changed')
    if plan.get('generation_protocol') != 'independent_rng_reset_only_v1':
        raise ValueError('Unknown scene generation protocol')
    for row, seed in zip(plan['resets'], data['scene_seeds'][task]):
        if row.get('scene_seed') != seed or 'pre_reset_rng' in row:
            raise ValueError('Seeded plan RNG differs from the published definition')


CHUNK_SIZE = 8


def chunk_ranges():
    return [(start, min(CHUNK_SIZE, 51 - start)) for start in range(0, 51, CHUNK_SIZE)]


def validate_chunk(chunk, task, freeze_sha, start, count):
    spec = next(t for t in definition()['tasks'] if t['task'] == task)
    expected = {'format': 'robocasa_scene_chunk_v1', 'complete': True, 'env': 'robocasa/' + task,
                'chunk_start': start, 'chunk_count': count, 'max_episode_steps': spec['horizon'],
                'suite_definition_sha256': file_sha(DEFINITION), 'sim_freeze_sha256': freeze_sha,
                'generator_sha256': file_sha(Path(__file__)), 'generation_protocol': 'independent_rng_reset_only_v1'}
    for key, value in expected.items():
        if chunk.get(key) != value:
            raise ValueError('Scene chunk identity mismatch: ' + key)
    rows = chunk['resets']
    if [row['episode'] for row in rows] != list(range(start, start + count)):
        raise ValueError('Scene chunk coverage mismatch')
    for row in rows:
        if row['scene_seed'] != definition()['scene_seeds'][task][row['episode']] or 'pre_reset_rng' in row:
            raise ValueError('Scene chunk seed mismatch')
        if any(len(row[key]) != 64 for key in ('initial_observation_sha256', 'initial_state_sha256')):
            raise ValueError('Invalid scene chunk hash')
    return chunk


def generate_chunk(task, output, freeze_sha, start):
    """Bound native renderer lifetime; a fresh process owns at most eight resets."""
    data = definition()
    spec = next(t for t in data['tasks'] if t['task'] == task)
    count = dict(chunk_ranges())[start]
    target = output / '.chunks' / task / f'{start:03d}.json'
    activate('baseline', shim=False, overlay=False)
    import robocasa
    import robosuite
    import resource
    from examples.pi05_robocasa365.eval_files.simulation_env import create_single_env
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
    from episode_plan import observation_hash, mujoco_state_hash
    random.seed(7)
    np.random.seed(7)
    args = SimpleNamespace(env_name='robocasa/' + task, env_split='pretrain', seed=7,
                           video_out_path=None, n_action_steps=5, max_episode_steps=spec['horizon'])
    rows = []
    original = RoboCasaGymEnv.reset

    def reset(obj, *a, **kw):
        episode = start + len(rows)
        seed = data['scene_seeds'][task][episode]
        state = seeded_rng(seed)
        restore_rng(obj.env, state)
        obs, info = original(obj, *a, **kw)
        rows.append({'episode': episode, 'scene_seed': seed,
                     'initial_observation_sha256': observation_hash(obs),
                     'initial_state_sha256': mujoco_state_hash(obj.env)})
        return obs, info

    RoboCasaGymEnv.reset = reset
    env = None
    try:
        env = create_single_env(args, 0)
        for _ in range(count):
            env.reset()
            atomic_json(target.with_suffix('.inprogress.json'), {'complete': False, 'resets': rows,
                        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'pid': os.getpid()})
    finally:
        RoboCasaGymEnv.reset = original
        if env is not None:
            env.close()
    plan = {'format': 'robocasa_scene_chunk_v1', 'complete': True, 'env': args.env_name, 'env_split': 'pretrain',
            'seed': 7, 'n_action_steps': 5, 'max_episode_steps': spec['horizon'], 'num_trials': 50,
            'robocasa_version': robocasa.__version__, 'robosuite_version': robosuite.__version__,
            'sim_freeze_sha256': freeze_sha, 'suite_definition_sha256': file_sha(DEFINITION),
            'generation_protocol': 'independent_rng_reset_only_v1', 'resets': rows,
            'python_version': platform.python_version(), 'rng_algorithms': ['PCG64', 'MT19937', 'python_random_v3'],
            'chunk_start': start, 'chunk_count': count, 'generator_sha256': file_sha(Path(__file__))}
    validate_chunk(plan, task, freeze_sha, start, count)
    atomic_json(target, plan)


def generate_task(task, output, freeze_sha, home, egl):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    target = output / (task + '.json')
    with (output / (task + '.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            old = load_plan(target)
            if old['format'] != SEEDED_FORMAT or old['sim_freeze_sha256'] != freeze_sha:
                raise ValueError('Existing plan identity differs; do not overwrite it')
            return
        chunks = []
        for start, count in chunk_ranges():
            path = output / '.chunks' / task / f'{start:03d}.json'
            if not path.exists():
                cmd = [sys.executable, str(Path(__file__).resolve()), '--home', str(home), '--output', str(output),
                       '--egl-device-id', egl, '--task', task, '--chunk-start', str(start)]
                began = time.time()
                try:
                    result = subprocess.run(cmd, check=False, timeout=1800)
                except subprocess.TimeoutExpired:
                    atomic_json(path.with_suffix('.exit.json'), {'started': began, 'ended': time.time(),
                                'returncode': None, 'timeout': True, 'chunk_start': start, 'chunk_count': count})
                    raise
                atomic_json(path.with_suffix('.exit.json'), {'started': began, 'ended': time.time(),
                            'returncode': result.returncode, 'chunk_start': start, 'chunk_count': count})
                if result.returncode:
                    raise RuntimeError(f'{task} scene chunk {start} exited with {result.returncode}; see .chunks exit receipt')
            chunk = validate_chunk(json.loads(path.read_text()), task, freeze_sha, start, count)
            chunks.append(chunk)
        plan = dict(chunks[0])
        for key in ('chunk_start', 'chunk_count'):plan.pop(key)
        plan.update(format=SEEDED_FORMAT, resets=[row for chunk in chunks for row in chunk['resets']])
        for chunk in chunks:
            for key in ('robocasa_version', 'robosuite_version', 'python_version', 'rng_algorithms'):
                if chunk[key] != plan[key]:raise ValueError('Mixed scene chunk runtime: ' + key)
        validate_seeded_plan(plan)
        atomic_json(target, plan)
        load_plan(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--egl-device-id', default='0')
    p.add_argument('--workers', type=int, choices=(1, 2, 4, 8), default=4)
    p.add_argument('--task', choices=[t['task'] for t in definition()['tasks']], help=argparse.SUPPRESS)
    p.add_argument('--chunk-start', type=int, choices=[s for s, _ in chunk_ranges()], help=argparse.SUPPRESS)
    a = p.parse_args()
    a.output = a.output.resolve()
    freeze = a.home.resolve() / 'receipts/sim.freeze.txt'
    if not freeze.is_file():
        p.error('Prepare the simulator installation and its real sim.freeze.txt first')
    if not a.egl_device_id.isdecimal():
        p.error('EGL device ID must be an index into the actual EGL device list')
    freeze_sha = file_sha(freeze)
    if a.task:
        if os.environ.get('PYTHONHASHSEED') != '7':p.error('Internal scene workers require PYTHONHASHSEED=7; use the public driver')
        os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=a.egl_device_id)
        if a.chunk_start is not None:generate_chunk(a.task, a.output, freeze_sha, a.chunk_start)
        else:generate_task(a.task, a.output, freeze_sha, a.home.resolve(), a.egl_device_id)
        return
    if a.chunk_start is not None:p.error('--chunk-start is internal and requires --task')
    a.output.mkdir(parents=True, exist_ok=True)
    env = environment('baseline', shim=False, overlay=False)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT / 'scripts'), str(runtime('baseline')))),
               PYTHONHASHSEED='7', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=a.egl_device_id)

    failed = threading.Event()
    def generate(task):
        if failed.is_set():return
        cmd = [sys.executable, str(Path(__file__).resolve()), '--home', str(a.home.resolve()),
               '--output', str(a.output), '--egl-device-id', a.egl_device_id, '--task', task['task']]
        try:
            with (a.output / (task['task'] + '.generation.log')).open('a') as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=7200)
        except subprocess.TimeoutExpired:
            failed.set()
            atomic_json(a.output / '.generation' / (task['task'] + '.exit.json'), {'returncode': None, 'timeout': True, 'ended': time.time()})
            raise
        atomic_json(a.output / '.generation' / (task['task'] + '.exit.json'), {'returncode': result.returncode, 'ended': time.time()})
        if result.returncode:
            failed.set()
            raise RuntimeError(task['task'] + ' initialization failed; see .generation and .chunks exit receipts')

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(generate, task) for task in definition()['tasks']]
        for job in jobs:
            job.result()
    if file_sha(freeze) != freeze_sha:
        raise ValueError('Simulator receipt changed during scene generation')
    build_plan_set(a.output)
    from evaluate_frozen import validate_suite
    print(json.dumps(validate_suite(a.output), indent=2))


if __name__ == '__main__':
    main()
