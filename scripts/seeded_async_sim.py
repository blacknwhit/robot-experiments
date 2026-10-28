"""One GL-owning process, independent environments, asynchronous batch-1 policies.

All simulation, RNG restoration and renderer lifecycle work stays on the main
thread. Network workers only receive a policy client and an immutable observation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import pickle
import signal
import socket
import threading
import time
import traceback

import numpy as np
import simulator_renderer
from episode_plan import atomic_json, file_sha
from scene_suite import DEFINITION
from seeded_evaluation import (PROTOCOL, HEALTH_VERSION, EXECUTION_VERSION, selected_episodes,
                               partitions, task_seeds, check_observation, set_policy_episode)


class EnvironmentRng:
    def __init__(self):
        self.capture()

    def capture(self):
        self.python = random.getstate()
        self.numpy = np.random.get_state()

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)


def diagnostic_fingerprint(env, observation):
    from fresh_scene import fingerprints
    from episode_plan import rng_snapshot
    result = fingerprints(env, observation)
    result['rng_sha256'] = hashlib.sha256(pickle.dumps(rng_snapshot(env.unwrapped.env), protocol=4)).hexdigest()
    return result


def run(args):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Simulation and GL must be owned by the main thread')
    ids = selected_episodes(args.diagnostic_episodes)
    # Internal process-worker fragments retain global IDs and are never scored
    # independently. The parent must assemble and validate the complete task.
    if hasattr(args, '_episode_ids'):
        ids = selected_episodes(','.join(map(str, args._episode_ids)))
    diagnostic = args.diagnostic_episodes is not None
    if args.diagnostic_max_replans is not None and (not diagnostic or args.diagnostic_max_replans < 1):
        raise ValueError('Replan limit is only allowed for explicit diagnostics')
    groups = partitions(ids, args.n_envs)
    spec, seeds = task_seeds(args.task)
    seed_sha = file_sha(DEFINITION)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'SIMULATION_PROCESS.json', {'pid': os.getpid(), 'hostname': socket.gethostname(),
                'started': time.time(), 'execution': EXECUTION_VERSION, 'n_envs': args.n_envs})
    automatic_gc = gc.isenabled()
    slots = []
    pending = {}
    rows = []
    active = None
    executor = None
    began = time.time()
    deadline = time.monotonic() + args.timeout
    metadata = None
    primary_error = None
    rgb_schedule = getattr(args, 'rgb_schedule', 'all')
    rgb_report = None

    def keep_live():
        live = [slot for slot in slots if slot.get('env') is not None]
        proofs = simulator_renderer.retain_environments([slot['env'] for slot in live])
        for slot, proof in zip(live, proofs):
            slot['renderer'] = proof

    def start_episode(slot):
        nonlocal active
        active = slot
        slot['episode'] = slot['ids'].pop(0)
        slot['phase'] = 'reset'
        slot['env'], slot['observation'], _ = create_scene('robocasa/' + spec['task'], spec['horizon'], seeds[slot['episode']])
        slot['rng'] = EnvironmentRng()
        keep_live()
        slot['health'] = check_observation(slot['observation'])
        set_policy_episode(slot['model'], slot['episode'])
        slot.update(requests=0, success=False, started=None)
        if getattr(args, 'diagnostic_trace', False):
            slot['trace'] = [diagnostic_fingerprint(slot['env'], slot['observation'])]
        atomic_json(args.output / f"env{slot['slot']}.initial.json", {'episode': slot['episode'],
                    'scene_seed': seeds[slot['episode']], 'health': slot['health'], 'renderer': slot['renderer']})
        if diagnostic:
            np.savez_compressed(args.output / f"episode{slot['episode']}.initial.npz",
                                **{k: np.asarray(v) for k, v in slot['observation'].items() if not np.asarray(v).dtype.hasobject})

    def submit(slot):
        if slot['started'] is None:
            slot['started'] = time.time()
        slot['phase'] = 'policy'
        # Never pass slot/env/renderer to a thread; one request per client at a time.
        future = executor.submit(slot['model'].step, batch_observation(slot['observation']))
        pending[future] = slot

    try:
        simulator_renderer.install()
        from fresh_scene import create_scene, batch_observation, unbatch_action
        from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import PolicyWrapper
        import robocasa
        import robosuite
        if rgb_schedule == 'final':
            from final_frame_rgb import install
            rgb_report = install()
        elif rgb_schedule != 'all':
            raise ValueError('Unknown RGB schedule')
        gc.disable()
        for index, group in enumerate(groups):
            slot = {'slot': index, 'ids': list(group), 'env': None, 'model': None, 'phase': 'connect', 'episode': None}
            slots.append(slot)
            active = slot
            slot['model'] = PolicyWrapper(host=args.host, port=args.port, task_name='robocasa/' + spec['task'],
                                          base_seed=7, strict_determinism=True, trace_determinism=False)
            if metadata is None:
                metadata = slot['model'].server_metadata
            elif metadata != slot['model'].server_metadata:
                raise ValueError('Mixed policy identities')
            start_episode(slot)
        # Finish initial EGL setup before the first policy CUDA Graph capture.
        executor = ThreadPoolExecutor(max_workers=args.n_envs, thread_name_prefix='policy-network')
        for slot in slots:
            submit(slot)
        while pending:
            if time.monotonic() >= deadline:
                raise TimeoutError('Seed-only evaluation exceeded timeout')
            completed, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
            for future in completed:
                slot = pending.pop(future)
                active = slot
                actions = unbatch_action(future.result()['actions'])
                if any(not np.isfinite(np.asarray(value)).all() for value in actions.values()):
                    raise ValueError('Non-finite action')
                slot['phase'] = 'simulation'
                slot['rng'].restore()
                try:
                    observation, reward, terminated, truncated, _ = slot['env'].step(actions)
                finally:
                    slot['rng'].capture()
                slot['observation'] = observation
                if getattr(args, 'diagnostic_trace', False):
                    from episode_plan import observation_hash
                    slot['trace'].append({**diagnostic_fingerprint(slot['env'], observation),
                        'action_sha256': observation_hash(actions), 'reward': float(reward),
                        'terminated': bool(terminated), 'truncated': bool(truncated)})
                simulator_renderer.assert_healthy()
                check_observation(observation)
                if not np.isfinite(reward):
                    raise ValueError('Non-finite environment reward')
                slot['success'] |= bool(reward)
                done = bool(terminated or truncated)
                slot['requests'] += 1
                slot['model'].finish_step(np.asarray([done]))
                limited = args.diagnostic_max_replans is not None and slot['requests'] >= args.diagnostic_max_replans
                if not done and not limited:
                    submit(slot)
                    continue
                episode = slot['episode']
                row = {'episode': episode, 'scene_seed': seeds[episode], 'success': slot['success'],
                       'episode_complete': done, 'requests': slot['requests'], 'health_checks': slot['requests'] + 1,
                       'initial_health': slot['health'], 'renderer': slot['renderer'],
                       'started_at': slot['started'], 'finished_at': time.time()}
                rows.append(row)
                if getattr(args, 'diagnostic_trace', False):
                    atomic_json(args.output / f'episode{episode}.trace.json', slot['trace'])
                slot['phase'] = 'close'
                env = slot['env']
                slot['env'] = None
                if rgb_report is not None:
                    rgb_report.release(env)
                keep_live()  # Free this env's GPU resources under their owner before env.close().
                env.close()
                del env
                gc.collect()
                simulator_renderer.assert_healthy()
                atomic_json(args.output / f"env{slot['slot']}.progress.json", {'slot': slot['slot'],
                            'last_episode': episode, 'last_requests': slot['requests'], 'finished_episodes': len(rows)})
                if slot['ids']:
                    start_episode(slot)
                    submit(slot)
                else:
                    slot['phase'] = 'complete'
        if file_sha(DEFINITION) != seed_sha:
            raise ValueError('Seed definition changed during evaluation')
        rows.sort(key=lambda row: row['episode'])
        if [row['episode'] for row in rows] != ids:
            raise ValueError('Duplicate or missing episode IDs')
    except BaseException:
        primary_error = traceback.format_exc()
        error = {'traceback': primary_error, 'slot': active['slot'] if active else None,
                 'episode': active['episode'] if active else None, 'phase': active['phase'] if active else None}
        if active and isinstance(active.get('observation'), dict):
            try:
                np.savez_compressed(args.output / 'failed_observation.npz',
                                    **{k: np.asarray(v) for k, v in active['observation'].items() if not np.asarray(v).dtype.hasobject})
            except Exception as save_error:
                error['observation_save_error'] = repr(save_error)
        atomic_json(args.output / 'SIMULATION_ERROR.json', error)
        raise
    finally:
        cleanup = []
        for future in pending:
            future.cancel()
        for slot in slots:
            if slot.get('model') is not None:
                try: slot['model'].close()
                except BaseException: cleanup.append(traceback.format_exc())
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        try: simulator_renderer.release_all()
        except BaseException: cleanup.append(traceback.format_exc())
        for slot in slots:
            if slot.get('env') is not None:
                try:
                    if rgb_report is not None:
                        rgb_report.release(slot['env'])
                    slot['env'].close()
                except BaseException: cleanup.append(traceback.format_exc())
                slot['env'] = None
        gc.collect()
        try: simulator_renderer.assert_healthy()
        except BaseException: cleanup.append(traceback.format_exc())
        if rgb_report is not None:
            try: rgb_report.close()
            except BaseException: cleanup.append(traceback.format_exc())
        if automatic_gc:
            gc.enable()
        if cleanup:
            atomic_json(args.output / 'CLEANUP_ERROR.json', {'errors': cleanup})
            if primary_error is None:
                raise RuntimeError('Simulator cleanup failed; see CLEANUP_ERROR.json')
    completed = all(row['episode_complete'] for row in rows)
    successes = [row['success'] for row in rows] if completed else None
    result = {'complete': completed, 'evaluation_kind': 'diagnostic' if diagnostic else 'formal',
              'rgb_schedule': rgb_schedule, 'rgb_counts': rgb_report() if rgb_report is not None else None,
              'initial_rgb': getattr(args, 'initial_rgb', 'all'),
              'env': 'robocasa/' + spec['task'], 'episode_protocol': PROTOCOL, 'simulator_execution': EXECUTION_VERSION,
              'scene_seed_definition_sha256': seed_sha, 'episode_ids': ids, 'scene_seeds': [seeds[i] for i in ids],
              'num_trials': len(ids), 'n_envs': args.n_envs, 'logical_environment_index': 0, 'seed': 7, 'n_action_steps': 5,
              'max_episode_steps': spec['horizon'], 'diagnostic_max_replans': args.diagnostic_max_replans,
              'reference_plan_required': False, 'reference_observation_comparison': False,
              'observation_health_version': HEALTH_VERSION, 'episodes': rows, 'successes': successes,
              'num_successes': sum(successes) if completed else None, 'requests': sum(row['requests'] for row in rows),
              'policy_server_metadata': metadata, 'simulator_versions': {'robocasa': robocasa.__version__,
                  'robosuite': robosuite.__version__, 'rgb_adapter': simulator_renderer.VERSION},
              'wall_timing': {'started_at': began, 'finished_at': time.time()},
              'rollout_timing': {'started_at': min(row['started_at'] for row in rows), 'finished_at': max(row['finished_at'] for row in rows)}}
    atomic_json(args.output / 'result.json', result)
    return result


def main():
    faulthandler.enable(all_threads=True)
    if hasattr(signal, 'SIGUSR1'):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--n-envs', type=int, choices=range(1, 51), default=4)
    p.add_argument('--diagnostic-episodes')
    p.add_argument('--diagnostic-max-replans', type=int)
    p.add_argument('--diagnostic-trace', action='store_true')
    p.add_argument('--rgb-schedule', choices=('all', 'final'), default='all')
    p.add_argument('--timeout', type=int, default=14400)
    args = p.parse_args()
    if args.diagnostic_trace and args.diagnostic_episodes is None:
        p.error('Trace capture is diagnostic only')
    run(args)


if __name__ == '__main__':
    main()
