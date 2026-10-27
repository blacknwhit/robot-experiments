"""Compatibility CLI for seeded_async_sim; legacy helpers retain audit coverage."""
import argparse
import faulthandler
import gc
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import simulator_renderer
from episode_plan import atomic_json, file_sha
from scene_suite import DEFINITION
from seeded_evaluation import (PROTOCOL, HEALTH_VERSION, selected_episodes, partitions,
                               task_seeds, check_observation, set_policy_episode)


def exit_rows(children):
    rows = []
    for slot, child in enumerate(children):
        code = child.exitcode
        try:
            name = signal.Signals(-code).name if code is not None and code < 0 else None
        except ValueError:
            name = 'UNKNOWN'
        rows.append({'slot': slot, 'pid': child.pid, 'exitcode': code, 'signal': name})
    return rows


def logged_worker(slot, ids, task, output, host, port, seed_sha, max_replans):
    # Install this before importing MuJoCo. Native signals may bypass Python's
    # exception writer; the coordinator still records the original exit code.
    with (output / f'worker{slot}.log').open('w', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        faulthandler.enable(file=sys.stderr, all_threads=True)
        atomic_json(output / f'worker{slot}.started.json', {'slot': slot, 'ids': ids,
                    'pid': os.getpid(), 'hostname': socket.gethostname(), 'started': time.time()})
        worker(slot, ids, task, output, host, port, seed_sha, max_replans)


def worker(slot, ids, task, output, host, port, seed_sha, max_replans):
    from fresh_scene import create_scene, batch_observation, unbatch_action
    from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import PolicyWrapper
    import robocasa
    import robosuite
    env = model = observation = None
    rows = []
    episode = None
    began = None
    automatic_gc = gc.isenabled()
    try:
        simulator_renderer.install()
        # MuJoCo / EGL own native resources in Python reference cycles. Reclaim
        # cycles only after closing an episode, not during context construction,
        # drawing or readback. Normal reference-count deallocation still runs.
        gc.disable()
        if file_sha(DEFINITION) != seed_sha:
            raise ValueError('Seed definition changed during worker startup')
        spec, seeds = task_seeds(task)
        model = PolicyWrapper(host=host, port=port, task_name='robocasa/' + spec['task'],
                              base_seed=7, strict_determinism=True, trace_determinism=False)
        for episode in ids:
            env, observation, _ = create_scene('robocasa/' + spec['task'], spec['horizon'], seeds[episode])
            renderer_info = simulator_renderer.retain_environment(env)
            health = check_observation(observation)
            atomic_json(output / f'worker{slot}.initial.json', {'episode': episode, 'scene_seed': seeds[episode], 'health': health})
            set_policy_episode(model, episode)
            started = time.time()
            if began is None:
                began = started
            success = False
            requests = 0
            done = False
            while not done:
                actions = model.step(batch_observation(observation))['actions']
                observation, reward, terminated, truncated, _ = env.step(unbatch_action(actions))
                simulator_renderer.assert_healthy()
                check_observation(observation)
                if not np.isfinite(reward):
                    raise ValueError('Non-finite environment reward')
                success |= bool(reward)
                done = bool(terminated or truncated)
                requests += 1
                model.finish_step(np.asarray([done]))
                if max_replans is not None and requests >= max_replans:
                    break
            rows.append({'episode': episode, 'scene_seed': seeds[episode], 'success': success,
                         'episode_complete': done, 'requests': requests, 'health_checks': requests + 1,
                         'initial_health': health, 'renderer': renderer_info,
                         'started_at': started, 'finished_at': time.time()})
            simulator_renderer.release_all()
            env.close()
            env = None
            gc.collect()
            simulator_renderer.assert_healthy()
            atomic_json(output / f'worker{slot}.progress.json', {'complete': False, 'slot': slot, 'ids': ids, 'rows': rows})
        if file_sha(DEFINITION) != seed_sha:
            raise ValueError('Seed definition changed during evaluation')
        atomic_json(output / f'worker{slot}.json', {'slot': slot, 'ids': ids, 'rows': rows,
                    'started': began, 'ended': time.time(), 'metadata': model.server_metadata,
                    'simulator_versions': {'robocasa': robocasa.__version__, 'robosuite': robosuite.__version__,
                                           'rgb_adapter': simulator_renderer.VERSION}})
    except BaseException:
        error = {'episode': episode, 'traceback': traceback.format_exc(),
                 'completed_episodes': [row['episode'] for row in rows if row['episode_complete']]}
        if isinstance(observation, dict):
            arrays = {key: np.asarray(value) for key, value in observation.items()
                      if not np.asarray(value).dtype.hasobject}
            try:
                path = output / f'worker{slot}.failed_observation.npz'
                np.savez_compressed(path, **arrays)
                error['observation_file'] = path.name
            except Exception as save_error:
                error['observation_save_error'] = repr(save_error)
        atomic_json(output / f'worker{slot}.error.json', error)
        raise
    finally:
        try:
            if model is not None:
                model.close()
        finally:
            try:
                simulator_renderer.release_all()
                if env is not None:
                    env.close()
                    env = None
                gc.collect()
                simulator_renderer.assert_healthy()
            finally:
                if automatic_gc:
                    gc.enable()


def run(args):
    ids = selected_episodes(args.diagnostic_episodes)
    diagnostic = args.diagnostic_episodes is not None
    if args.diagnostic_max_replans is not None and (not diagnostic or args.diagnostic_max_replans < 1):
        raise ValueError('Replan limit is only allowed for explicit diagnostics')
    groups = partitions(ids, args.n_envs)
    spec, seeds = task_seeds(args.task)
    seed_sha = file_sha(DEFINITION)
    args.output.mkdir(parents=True, exist_ok=False)
    children = []
    began = time.time()
    ctx = mp.get_context('spawn')
    try:
        for slot, group in enumerate(groups):
            child = ctx.Process(target=logged_worker, args=(slot, group, args.task, args.output, args.host,
                                args.port, seed_sha, args.diagnostic_max_replans))
            child.start()
            children.append(child)
        deadline = time.monotonic() + args.timeout
        while any(child.is_alive() for child in children):
            if any(child.exitcode not in (None, 0) for child in children):
                atomic_json(args.output / 'WORKER_FAILURE.json', {'detected_at': time.time(),
                            'workers_before_cleanup': exit_rows(children)})
                raise RuntimeError('Simulator worker failed; inspect worker*.error.json and worker_exit.json')
            if time.monotonic() >= deadline:
                raise TimeoutError('Seed-only task evaluation exceeded timeout')
            time.sleep(.2)
        if any(child.exitcode != 0 for child in children):
            atomic_json(args.output / 'WORKER_FAILURE.json', {'detected_at': time.time(),
                        'workers_before_cleanup': exit_rows(children)})
            raise RuntimeError('Simulator worker failed; inspect worker exit codes')
        workers = [json.loads((args.output / f'worker{i}.json').read_text()) for i in range(args.n_envs)]
        rows = sorted((row for w in workers for row in w['rows']), key=lambda row: row['episode'])
        if [row['episode'] for row in rows] != ids:
            raise ValueError('Duplicate or missing episode IDs')
        metadata = workers[0]['metadata']
        versions = workers[0]['simulator_versions']
        if any(w['metadata'] != metadata or w['simulator_versions'] != versions for w in workers):
            raise ValueError('Mixed model or simulator identities')
        if file_sha(DEFINITION) != seed_sha:
            raise ValueError('Seed definition changed during evaluation')
        completed = all(row['episode_complete'] for row in rows)
        successes = [row['success'] for row in rows] if completed else None
        result = {'complete': completed, 'evaluation_kind': 'diagnostic' if diagnostic else 'formal',
                  'env': 'robocasa/' + spec['task'], 'episode_protocol': PROTOCOL,
                  'scene_seed_definition_sha256': seed_sha, 'episode_ids': ids,
                  'scene_seeds': [seeds[i] for i in ids], 'num_trials': len(ids), 'n_envs': args.n_envs,
                  'logical_environment_index': 0, 'seed': 7, 'n_action_steps': 5,
                  'max_episode_steps': spec['horizon'], 'diagnostic_max_replans': args.diagnostic_max_replans,
                  'reference_plan_required': False, 'reference_observation_comparison': False,
                  'observation_health_version': HEALTH_VERSION, 'episodes': rows, 'successes': successes,
                  'num_successes': sum(successes) if completed else None,
                  'requests': sum(row['requests'] for row in rows), 'policy_server_metadata': metadata,
                  'simulator_versions': versions, 'wall_timing': {'started_at': began, 'finished_at': time.time()},
                  'rollout_timing': {'started_at': min(w['started'] for w in workers), 'finished_at': max(w['ended'] for w in workers)}}
        atomic_json(args.output / 'result.json', result)
    finally:
        before = exit_rows(children)
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(timeout=15)
            if child.is_alive():
                child.kill()
                child.join(timeout=10)
        atomic_json(args.output / 'worker_exit.json', {'before_cleanup': before, 'after_cleanup': exit_rows(children)})


def _legacy_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--n-envs', type=int, choices=range(1, 51), default=4)
    p.add_argument('--diagnostic-episodes')
    p.add_argument('--diagnostic-max-replans', type=int)
    p.add_argument('--timeout', type=int, default=14400)
    run(p.parse_args())


from seeded_async_sim import main

if __name__ == '__main__':
    main()
