"""Fixed episode-seed evaluation: no scene preparation or reference pixel hashes.

    Single GPU/task: --task CloseBlenderLid --gpu-ids 0 --n-envs 4
    Eight GPU suite: --all-tasks --gpu-ids 0,1,2,3,4,5,6,7
"""
import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from common import ROOT, VARIANTS, activate, environment, runtime, sha256
from egl_lifecycle import VERSION as EGL_LIFECYCLE_VERSION
from episode_plan import atomic_json
from evaluate import TASKS, gpu_assignments, tasks_for_part
from evaluate_task import bind_output_identity, check_port_available
from scene_suite import run_task_process, stop_process_group
from seeded_evaluation import PROTOCOL, HEALTH_VERSION, seed_suite, selected_episodes, partitions, validate_result


def identity_for(args):
    import yaml
    config = yaml.safe_load(args.config.read_text())
    if config['portable']['variant'] != args.variant:
        raise ValueError('Config variant mismatch')
    return {'variant': args.variant, 'checkpoint_sha256': sha256(args.checkpoint),
            'config_sha256': sha256(args.config), 'source_manifest_sha256': sha256(ROOT / 'provenance/source_files.sha256'),
            'sim_freeze_sha256': sha256(args.home / 'receipts/sim.freeze.txt'),
            'inference_backend': 'cuda-graph-pipeline', 'inference_backend_version': 'cpu_gpu_pipeline_batch1_v1',
            'episode_protocol': PROTOCOL, 'n_envs': args.n_envs,
            'scene_seed_definition_sha256': seed_suite()['scene_seed_definition_sha256'],
            'task_registry_sha256': seed_suite()['task_registry_sha256'],
            'observation_health_version': HEALTH_VERSION, 'reference_plan_required': False,
            'egl_lifecycle_version': EGL_LIFECYCLE_VERSION,
            'evaluation_kind': 'diagnostic' if args.diagnostic_episodes is not None else 'formal',
            'diagnostic_episode_ids': selected_episodes(args.diagnostic_episodes) if args.diagnostic_episodes is not None else None,
            'diagnostic_max_replans': args.diagnostic_max_replans,
            'entry_sources': {name: sha256(ROOT / 'scripts' / name) for name in
                              ('evaluate_seeded.py', 'seeded_episode_sim.py', 'seeded_evaluation.py', 'fresh_scene.py', 'egl_lifecycle.py')}}


def validate_receipt(path, identity, task, ids, diagnostic, max_replans):
    receipt = json.loads(path.with_suffix('.identity.json').read_text())
    if receipt.get('identity') != identity or receipt.get('result_sha256') != sha256(path):
        raise ValueError('Existing result identity or bytes differ; preserve it and use a new output')
    return validate_result(json.loads(path.read_text()), task, ids, identity['n_envs'], identity, diagnostic, max_replans)


def summarize(args, identity):
    if identity['evaluation_kind'] != 'formal':
        raise ValueError('Diagnostic results cannot be summarized as a benchmark')
    rows = []
    versions = None
    for task in TASKS:
        result = validate_receipt(args.output / 'formal' / (task['task'] + '.json'), identity,
                                  task['task'], list(range(50)), False, None)
        if versions is None:
            versions = result['simulator_versions']
        if result['simulator_versions'] != versions:
            raise ValueError('Mixed simulator versions across task results')
        rows.append({'task': task['task'], 'successes': result['num_successes'], 'trials': 50,
                     'result_sha256': sha256(args.output / 'formal' / (task['task'] + '.json'))})
    atomic_json(args.output / 'FINAL_ACCEPTED.json', {'identity': identity, 'episode_protocol': PROTOCOL,
                'tasks': rows, 'trials': 2500, 'successes': sum(row['successes'] for row in rows),
                'simulator_versions': versions, 'reference_plan_required': False,
                'pixel_or_trajectory_equivalence_certified': False})


def run_single(args, identity):
    import fcntl
    diagnostic = args.diagnostic_episodes is not None
    ids = selected_episodes(args.diagnostic_episodes)
    partitions(ids, args.n_envs)
    gpus, egls = gpu_assignments(args.gpu_ids, args.egl_device_ids)
    if len(gpus) != 1:
        raise ValueError('Single-task entry requires exactly one GPU')
    category = 'diagnostic' if diagnostic else 'formal'
    with (args.output / (args.task + '.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = args.output / category / (args.task + '.json')
        if target.exists():
            validate_receipt(target, identity, args.task, ids, diagnostic, args.diagnostic_max_replans)
            print('Already complete with matching identity:', args.task)
            return
        attempt = args.output / 'runs' / args.task / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        atomic_json(attempt / 'STARTED.json', {'identity': identity, 'task': args.task, 'episode_ids': ids,
                    'started': time.time(), 'hostname': socket.gethostname(), 'pid': os.getpid(),
                    'cuda_device': gpus[0], 'egl_device': egls[0]})
        server = None
        try:
            activate(args.variant)
            check_port_available(args.port_base)
            policy_env = environment(args.variant)
            policy_env.update(CUDA_VISIBLE_DEVICES=gpus[0], PYTHONHASHSEED='7', OPENPI_DATA_HOME=str(args.home / 'cache/openpi'))
            command = [sys.executable, str(ROOT / 'scripts/server.py'), '--variant', args.variant,
                       '--config', str(args.config), '--checkpoint', str(args.checkpoint),
                       '--port', str(args.port_base), '--inference-backend', 'cuda-graph-pipeline']
            # Normal backend keeps its first-request exact eager/graph check;
            # no every-request duplicate inference and no prerequisite smoke.
            with (attempt / 'server.log').open('w') as log:
                server = subprocess.Popen(command, env=policy_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Policy startup failed; see ' + str(attempt / 'server.log'))
                try:
                    with socket.create_connection(('127.0.0.1', args.port_base), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Policy startup timeout')
                    time.sleep(2)
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
            client = WebsocketClientPolicy(port=args.port_base)
            try:
                metadata = client.get_server_metadata()
            finally:
                client.close()
            for key in ('variant', 'checkpoint_sha256', 'config_sha256', 'inference_backend', 'inference_backend_version'):
                if metadata.get(key) != identity[key]:
                    raise ValueError('Foreign policy server: ' + key)
            sim_env = environment(args.variant, shim=False, overlay=False)
            sim_env.pop('CUDA_VISIBLE_DEVICES', None)
            sim_env.pop('LD_PRELOAD', None)
            sim_env.update(PYTHONPATH=os.pathsep.join((str(ROOT / 'scripts'), str(runtime(args.variant)))),
                           PYTHONHASHSEED='7', PYTHONFAULTHANDLER='1', MUJOCO_GL='egl',
                           PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=egls[0])
            command = [str(args.sim_python), str(ROOT / 'scripts/seeded_episode_sim.py'), '--task', args.task,
                       '--output', str(attempt / 'simulation'), '--port', str(args.port_base),
                       '--n-envs', str(args.n_envs), '--timeout', str(args.timeout)]
            if diagnostic:
                command += ['--diagnostic-episodes', args.diagnostic_episodes]
            if args.diagnostic_max_replans is not None:
                command += ['--diagnostic-max-replans', str(args.diagnostic_max_replans)]
            atomic_json(attempt / 'SIMULATOR_COMMAND.json', {'command': command})
            with (attempt / 'simulation.log').open('w') as log:
                returncode = run_task_process(command, env=sim_env, log=log, timeout=args.timeout + 120)
            atomic_json(attempt / 'SIMULATOR_EXIT.json', {'returncode': returncode, 'ended': time.time()})
            if returncode != 0:
                raise RuntimeError(f'Simulator exited {returncode}; see {attempt / "simulation.log"} and simulation/worker*.error.json')
            result = json.loads((attempt / 'simulation/result.json').read_text())
            validate_result(result, args.task, ids, args.n_envs, identity, diagnostic, args.diagnostic_max_replans)
            if identity_for(args) != identity:
                raise ValueError('Code, weights, configuration or environment receipts changed during evaluation')
            atomic_json(target, result)
            atomic_json(target.with_suffix('.identity.json'), {'identity': identity, 'result_sha256': sha256(target),
                        'evidence': str(attempt.relative_to(args.output))})
            atomic_json(attempt / 'COMPLETE.json', {'entry_passed': True, 'diagnostic_only': diagnostic,
                        'full_episodes_complete': result['complete'], 'episodes': ids, 'requests': result['requests']})
        except BaseException:
            atomic_json(attempt / 'ERROR.json', {'traceback': traceback.format_exc(), 'ended': time.time()})
            raise
        finally:
            if server is not None:
                stop_process_group(server)


def suite_commands(args):
    selected = tasks_for_part(args.part) if args.part else TASKS
    gpus, egls = gpu_assignments(args.gpu_ids, args.egl_device_ids)
    if len(gpus) > len(selected) or (args.part and len(gpus) != len(selected)):
        raise ValueError('Parts require four GPUs (part13: two); all-tasks accepts 1..50 distinct GPUs')
    commands = [[] for _ in gpus]
    for i, task in enumerate(selected):
        slot = i % len(gpus)
        command = [sys.executable, str(Path(__file__).resolve()), '--variant', args.variant,
                   '--home', str(args.home), '--config', str(args.config), '--checkpoint', str(args.checkpoint),
                   '--sim-python', str(args.sim_python), '--output', str(args.output), '--task', task['task'],
                   '--gpu-ids', gpus[slot], '--egl-device-ids', egls[slot], '--port-base', str(args.port_base + slot),
                   '--n-envs', str(args.n_envs), '--timeout', str(args.timeout)]
        commands[slot].append(command)
    return commands


def run_suite(args):
    failed = threading.Event()
    def run_slot(commands):
        for command in commands:
            if failed.is_set():
                return
            try:
                subprocess.run(command, check=True)
            except BaseException:
                failed.set()
                raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.commands)) as pool:
        futures = [pool.submit(run_slot, commands) for commands in args.commands]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=VARIANTS)
    for key in ('home', 'config', 'checkpoint', 'sim-python', 'output'):
        p.add_argument('--' + key, type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--task', choices=[task['task'] for task in TASKS])
    mode.add_argument('--part', type=int, choices=range(1, 14))
    mode.add_argument('--all-tasks', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    p.add_argument('--gpu-ids', '--gpu-id', default='0')
    p.add_argument('--egl-device-ids', '--egl-device-id')
    p.add_argument('--port-base', '--port', type=int, default=19090)
    p.add_argument('--n-envs', type=int, choices=range(1, 51), default=4)
    p.add_argument('--timeout', type=int, default=14400)
    p.add_argument('--diagnostic-episodes', help='Explicit subset of global episode IDs; never counted in formal results')
    p.add_argument('--diagnostic-max-replans', type=int, help='Bound diagnostic prefixes only; does not change the native horizon')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    suite = seed_suite()
    if args.check_only:
        print(json.dumps(suite, indent=2))
        return
    for key in ('variant', 'home', 'config', 'checkpoint', 'sim_python', 'output'):
        if getattr(args, key) is None:
            p.error('--' + key.replace('_', '-') + ' is required')
        if key != 'variant':
            setattr(args, key, getattr(args, key).resolve())
    if args.diagnostic_episodes is not None and not args.task:
        p.error('Diagnostic episode selection is single-task only')
    if args.diagnostic_max_replans is not None and (args.diagnostic_episodes is None or args.diagnostic_max_replans < 1):
        p.error('A positive replan limit requires explicit --diagnostic-episodes')
    partitions(selected_episodes(args.diagnostic_episodes), args.n_envs)
    gpus, _ = gpu_assignments(args.gpu_ids, args.egl_device_ids)
    if args.task and len(gpus) != 1:
        p.error('Single task requires one GPU')
    if args.timeout <= 0 or not 1024 <= args.port_base <= 65535 - len(gpus):
        p.error('Invalid timeout or port range')
    if args.part or args.all_tasks:
        args.commands = suite_commands(args)
    if args.dry_run:
        print(json.dumps({'suite': suite, 'task': args.task, 'episode_ids': selected_episodes(args.diagnostic_episodes),
                          'n_envs': args.n_envs, 'commands_by_gpu': getattr(args, 'commands', None)}, indent=2))
        return
    identity = identity_for(args)
    args.output.mkdir(parents=True, exist_ok=True)
    bind_output_identity(args.output, identity)
    if args.summarize:
        summarize(args, identity)
    elif args.task:
        run_single(args, identity)
    else:
        run_suite(args)
        if args.all_tasks:
            summarize(args, identity)


if __name__ == '__main__':
    main()
