import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from episode_plan import build_plan_set
from evaluate import TASKS
from evaluate_frozen import child_commands, validate_suite
from episode_plan import SEEDED_FORMAT, load_plan, restore_rng, file_sha
from scene_suite import DEFINITION, definition, seeded_rng, chunk_ranges, validate_chunk
import scene_suite
from legacy_contract import require_same_policy
import random
import numpy as np


class FrozenSuiteTests(unittest.TestCase):
    def test_launch_error_stops_queued_work_and_records_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(home=root, output=root, egl_device_id='0')
            failed = threading.Event()
            with patch('scene_suite.run_task_process', side_effect=OSError('test launch error')) as launch:
                with self.assertRaises(OSError):
                    scene_suite.generate_task_process(TASKS[0], args, {}, failed)
                scene_suite.generate_task_process(TASKS[1], args, {}, failed)
                self.assertEqual(launch.call_count, 1)
            self.assertTrue(failed.is_set())
            receipt = json.loads((root / '.generation' / (TASKS[0]['task'] + '.exit.json')).read_text())
            self.assertIn('test launch error', receipt['error'])

    @unittest.skipUnless(sys.platform == 'linux', 'POSIX process group and flock acceptance')
    def test_timeout_and_coordinator_exit_leave_no_child_writer(self):
        import fcntl
        for leave_early in (False, True):
            with self.subTest(leave_early=leave_early), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                child = root / 'child.py'
                child.write_text('import fcntl, pathlib, sys, time\np=pathlib.Path(sys.argv[1])\n'
                                 'with p.open("a+") as f:\n fcntl.flock(f,fcntl.LOCK_EX)\n'
                                 ' p.with_suffix(".ready").touch()\n time.sleep(60)\n')
                parent = root / 'parent.py'
                parent.write_text('import pathlib, subprocess, sys, time\n'
                                  'subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]])\n'
                                  'while not pathlib.Path(sys.argv[2]).with_suffix(".ready").exists():time.sleep(.01)\n'
                                  + ('' if leave_early else 'time.sleep(60)\n'))
                lock = root / 'writer.lock'
                with (root / 'console.log').open('w') as log:
                    command = [sys.executable, str(parent), str(child), str(lock)]
                    if leave_early:
                        self.assertEqual(scene_suite.run_task_process(command, env=os.environ, log=log, timeout=3), 0)
                    else:
                        with self.assertRaises(subprocess.TimeoutExpired):
                            scene_suite.run_task_process(command, env=os.environ, log=log, timeout=2)
                self.assertTrue(lock.with_suffix('.ready').exists())
                with lock.open('a+') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_chunked_initialization_covers_same_51_scenes_and_rejects_bad_chunks(self):
        task = TASKS[0];plan = self.seeded_plan(task);rows = []
        for start, count in chunk_ranges():
            chunk = {**plan, 'format': 'robocasa_scene_chunk_v1', 'chunk_start': start, 'chunk_count': count,
                     'generator_sha256': file_sha(Path(scene_suite.__file__)), 'resets': plan['resets'][start:start+count]}
            validate_chunk(chunk, task['task'], 'c'*64, start, count)
            self.assertLessEqual(count, 8);rows.extend(chunk['resets'])
            for mutation in (lambda c: c.update(chunk_start=99), lambda c: c['resets'].pop(),
                             lambda c: c.update(generator_sha256='f'*64)):
                bad = copy.deepcopy(chunk);mutation(bad)
                with self.assertRaises(ValueError):validate_chunk(bad, task['task'], 'c'*64, start, count)
        self.assertEqual(rows, plan['resets'])

    def seeded_plan(self, task):
        seeds = definition()['scene_seeds'][task['task']]
        return {'format': SEEDED_FORMAT, 'complete': True, 'env': 'robocasa/' + task['task'],
                'seed': 7, 'n_action_steps': 5, 'env_split': 'pretrain', 'num_trials': 50,
                'max_episode_steps': task['horizon'], 'robocasa_version': '1.0.0', 'robosuite_version': '1.5.2',
                'sim_freeze_sha256': 'c' * 64, 'suite_definition_sha256': file_sha(DEFINITION),
                'generator_sha256': file_sha(Path(scene_suite.__file__)), 'python_version': '3.11.14',
                'rng_algorithms': ['PCG64', 'MT19937', 'python_random_v3'],
                'generation_protocol': 'independent_rng_reset_only_v1', 'resets': [
                    {'episode': i, 'scene_seed': seed, 'initial_observation_sha256': 'd' * 64,
                     'initial_state_sha256': 'e' * 64} for i, seed in enumerate(seeds)]}

    def test_seeded_suite_needs_no_policy_and_cannot_be_used_as_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in TASKS:
                (root / (task['task'] + '.json')).write_text(json.dumps(self.seeded_plan(task)))
            build_plan_set(root)
            self.assertEqual(validate_suite(root)['scene_source']['suite_definition_sha256'], file_sha(DEFINITION))
            path = root / (TASKS[0]['task'] + '.json')
            plan = load_plan(path)
            self.assertNotIn('reference_result', plan)
            self.assertIn('pre_reset_rng', plan['resets'][0])
            with self.assertRaisesRegex(ValueError, 'Legacy'):
                require_same_policy(plan, {})
            for mutate in (lambda p: p['resets'][0].update(scene_seed=0),
                           lambda p: p.update(reference_result={}),
                           lambda p: p.update(suite_definition_sha256='f' * 64)):
                bad = self.seeded_plan(TASKS[0]); mutate(bad)
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_plan(path)

    def test_episode_rng_does_not_depend_on_model_consumption_or_worker_order(self):
        env = SimpleNamespace(rng=np.random.default_rng(77))
        seeds = definition()['scene_seeds'][TASKS[0]['task']]
        def observations(order, noise):
            rows = {}
            for i in order:
                env.rng.random(noise); np.random.random(noise)
                for _ in range(noise): random.random()
                restore_rng(env, seeded_rng(seeds[i]))
                rows[i] = (env.rng.random(3).tolist(), np.random.random(3).tolist(), random.random())
            return rows
        self.assertEqual(observations(range(51), 1), observations(reversed(range(51)), 73))

    def test_seeded_suite_rejects_mixed_generator_or_runtime(self):
        for field in ('generator_sha256', 'python_version', 'rng_algorithms'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for task in TASKS:
                    plan = self.seeded_plan(task)
                    if task['index'] == 49:
                        plan[field] = 'different'
                    (root / (task['task'] + '.json')).write_text(json.dumps(plan))
                build_plan_set(root)
                with self.assertRaises(ValueError):
                    validate_suite(root)

    def write_suite(self, root, mutate=None):
        for task in TASKS:
            ref = {'env': 'robocasa/' + task['task'], 'env_split': 'pretrain', 'seed': 7,
                   'n_action_steps': 5, 'n_envs': 1, 'max_episode_steps': task['horizon'], 'num_trials': 50,
                   'num_successes': 0, 'successes': [False] * 50,
                   'episode_trace_sha256': [f'{task["index"] * 50 + i:064x}' for i in range(50)],
                   'policy_server_metadata': {'variant': 'baseline', 'checkpoint_sha256': 'a' * 64,
                                              'config_sha256': 'b' * 64}}
            plan = {**ref, 'format': 'robocasa_frozen_episode_plan_v1', 'complete': True,
                    'sim_freeze_sha256': 'c' * 64, 'robocasa_version': '1.0.0', 'robosuite_version': '1.5.1',
                    'reference_result': copy.deepcopy(ref), 'resets': [
                        {'episode': i, 'initial_observation_sha256': 'd' * 64, 'initial_state_sha256': 'e' * 64,
                         'pre_reset_rng': {'env': {}, 'numpy': [], 'python': []}} for i in range(51)]}
            if mutate:
                mutate(task, plan)
            (root / (task['task'] + '.json')).write_text(json.dumps(plan))
        build_plan_set(root)

    def test_complete_suite_and_missing_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_suite(root)
            proof = validate_suite(root)
            self.assertEqual(proof['scored_episodes'], 2500)
            self.assertEqual(proof['historical_scene_equivalence'], 'not_certified_by_this_check')
            (root / 'PLAN_SET.json').unlink()
            (root / (TASKS[-1]['task'] + '.json')).unlink()
            build_plan_set(root)
            with self.assertRaisesRegex(ValueError, 'exactly'):
                validate_suite(root)

    def test_mixed_model_and_wrong_horizon_rejected(self):
        def model(task, plan):
            if task['index'] == 49:
                plan['reference_result']['policy_server_metadata']['checkpoint_sha256'] = 'f' * 64
        def horizon(task, plan):
            if task['index'] == 49:
                plan['max_episode_steps'] += 5
                plan['reference_result']['max_episode_steps'] += 5
        def duplicate_trace(task, plan):
            if task['index'] == 49:
                plan['reference_result']['episode_trace_sha256'][0] = '0' * 64
        def parallel_reference(task, plan):
            if task['index'] == 49:
                plan['reference_result']['n_envs'] = 8
        for mutation in (model, horizon, duplicate_trace, parallel_reference):
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as directory:
                self.write_suite(Path(directory), mutation)
                with self.assertRaises(ValueError):
                    validate_suite(directory)

    def test_all_parts_one_gpu_per_task_and_no_coverage_gaps(self):
        args = SimpleNamespace(variant='query_fixed', home='home', config='config', checkpoint='weights',
                               sim_python='sim', output='out', plan_dir='plans', n_envs=8, summarize=False,
                               gpu_ids='GPU-A,GPU-B,GPU-C,GPU-D', egl_device_ids='3,2,1,0',
                               smoke=False, port_base=19090, part=1)
        tasks = []
        for part in range(1, 14):
            args.part = part
            if part == 13:
                args.gpu_ids = 'GPU-A,GPU-B'
                args.egl_device_ids = '3,2'
            commands = child_commands(args)
            self.assertEqual(len(commands), 2 if part == 13 else 4)
            self.assertEqual(len({c[c.index('--gpu-id') + 1] for c in commands}), len(commands))
            tasks.extend(c[c.index('--task') + 1] for c in commands)
            for command in commands:
                self.assertEqual(command[command.index('--n-envs') + 1], '8')
                self.assertEqual(command[command.index('--scene-protocol') + 1], 'shared-scenes')
        self.assertEqual(tasks, [t['task'] for t in TASKS])
        args.part = 1
        with self.assertRaisesRegex(ValueError, 'four GPUs'):
            child_commands(args)
        args.smoke = True
        self.assertEqual(len(child_commands(args)), 1)


if __name__ == '__main__':
    unittest.main()
