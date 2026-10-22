"""Run a complete frozen RoboCasa suite, one task per GPU, eight environments per task.

This entry consumes real recorded plans. It never generates replacement scenes.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

from common import ROOT, VARIANTS
from episode_plan import SEEDED_FORMAT, file_sha, load_plan, load_plan_set
from evaluate import TASKS, gpu_assignments, tasks_for_part
from legacy_contract import require_same_trajectories


def validate_suite(directory):
    """Reject debug subsets, extra tasks, mixed sources and changed horizons."""
    directory = Path(directory)
    plans, digest = load_plan_set(directory)
    if set(plans) != {task['task'] for task in TASKS}:
        raise ValueError('Formal suite requires exactly the registered 50 tasks')
    sources = []
    traces = []
    for task in TASKS:
        plan = load_plan(directory / (task['task'] + '.json'), 'robocasa/' + task['task'], 50)
        if plan['num_trials'] != 50 or plan['max_episode_steps'] != task['horizon']:
            raise ValueError('Formal suite requires 50 episodes and the registered horizon: ' + task['task'])
        if plan['format'] == SEEDED_FORMAT:
            source = {key: plan[key] for key in ('suite_definition_sha256', 'generation_protocol',
                                               'generator_sha256', 'scene_factory_sha256', 'python_version', 'rng_algorithms')}
        else:
            metadata = plan['reference_result'].get('policy_server_metadata', {})
            require_same_trajectories(plan['reference_result'], plan['reference_result'], 50)
            traces.extend(plan['reference_result']['episode_trace_sha256'])
            source = {key: metadata.get(key) for key in ('variant', 'checkpoint_sha256', 'config_sha256')}
            if source['variant'] not in VARIANTS or any(
                not isinstance(source[key], str) or len(source[key]) != 64
                for key in ('checkpoint_sha256', 'config_sha256')
            ):
                raise ValueError('Missing scene source policy identity: ' + task['task'])
        sources.append({'format': plan['format'], **source, **{key: plan[key] for key in
            ('sim_freeze_sha256', 'robocasa_version', 'robosuite_version')}})
    if any(source != sources[0] for source in sources):
        raise ValueError('Scene plans mix source policies or simulator environments')
    if traces and len(set(traces)) != 2500:
        raise ValueError('Duplicate reference traces across tasks')
    return {'episode_plan_set_sha256': digest, 'tasks': 50, 'scored_episodes': 2500,
            'task_registry_sha256': file_sha(ROOT / 'assets/eval_tasks.json'),
            'scene_source': sources[0], 'historical_scene_equivalence': 'not_certified_by_this_check'}


def child_commands(args):
    base = [sys.executable, str(ROOT / 'scripts/evaluate_task.py'), '--variant', args.variant,
            '--home', str(args.home), '--config', str(args.config), '--checkpoint', str(args.checkpoint),
            '--sim-python', str(args.sim_python), '--output', str(args.output),
            '--scene-protocol', 'shared-scenes', '--plan-dir', str(args.plan_dir), '--n-envs', str(args.n_envs)]
    if args.summarize:
        return [base + ['--summarize']]
    gpus, egl = gpu_assignments(args.gpu_ids, args.egl_device_ids)
    selected = [TASKS[0]] if args.smoke else tasks_for_part(args.part)
    if not args.smoke and len(gpus) != len(selected):
        raise ValueError('Use four GPUs for parts 1..12, two GPUs for part 13; exactly one task per GPU')
    if not 1024 <= args.port_base <= 65535 - len(selected):
        raise ValueError('Invalid port base')
    commands = []
    for i, task in enumerate(selected):
        command = base + ['--task', task['task'], '--gpu-id', gpus[i], '--egl-device-id', egl[i],
                          '--port', str(args.port_base + i)]
        if args.smoke:
            command.append('--smoke')
        commands.append(command)
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-dir', required=True, type=Path)
    p.add_argument('--check-only', action='store_true', help='Validate 50 x 50 real plans without loading a model')
    p.add_argument('--variant', choices=VARIANTS)
    for key in ('home', 'config', 'checkpoint', 'sim-python', 'output'):
        p.add_argument('--' + key, type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_true')
    modes.add_argument('--summarize', action='store_true')
    modes.add_argument('--part', type=int, choices=range(1, 14))
    p.add_argument('--n-envs', type=int, choices=(1, 2, 4, 8), default=8)
    p.add_argument('--gpu-ids', default='0,1,2,3')
    p.add_argument('--egl-device-ids')
    p.add_argument('--port-base', type=int, default=19090)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    suite = validate_suite(args.plan_dir)
    if args.check_only:
        print(json.dumps(suite, indent=2))
        return
    for key in ('variant', 'home', 'config', 'checkpoint', 'sim_python', 'output'):
        if getattr(args, key) is None:
            p.error('--' + key.replace('_', '-') + ' is required')
    if not (args.smoke or args.summarize or args.part):
        p.error('Choose --smoke, --part 1..13 or --summarize')
    commands = child_commands(args)
    if args.dry_run:
        print(json.dumps({'suite': suite, 'commands': commands}, indent=2))
        return
    # Each child retains its own model/plan identity gate and per-task writer lock.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
        jobs = [pool.submit(subprocess.run, command, check=True) for command in commands]
        for job in jobs:
            job.result()
    if validate_suite(args.plan_dir) != suite:
        raise ValueError('Frozen suite changed during execution')


if __name__ == '__main__':
    main()
