import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import POSITION_VARIANTS
from configure import render
from checkpoint import choose_latest
from train_or_resume import command


class PositionHandoffTests(unittest.TestCase):
    def test_formal_is_explicit_and_keeps_scientific_recipe(self):
        for variant in POSITION_VARIANTS:
            with self.assertRaises(ValueError):
                render(variant, '/tmp/positions', 'run', 2)
            with self.assertRaises(ValueError):
                render(variant, '/tmp/positions', 'run', 2, formal=True, engineering=True)
            short = render(variant, '/tmp/positions', 'short', 2, stop=1500, engineering=True)
            formal = render(variant, '/tmp/positions', 'formal', 2, formal=True)
            self.assertEqual(formal['trainer']['stop_after_steps'], 70000)
            self.assertEqual(formal['trainer']['max_train_steps'], 100000)
            for key in ('framework', 'datasets', 'seed', 'trajectory_release_sha256'):
                self.assertEqual(short[key], formal[key])
            expected = copy.deepcopy(short['trainer']); expected['stop_after_steps'] = 70000
            self.assertEqual(expected, formal['trainer'])

    def test_repeated_scheduler_entry_never_silently_restarts_base(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            cfg = render(POSITION_VARIANTS[0], home, 'short', 2, stop=1500, engineering=True)
            path = home / 'config.yaml'; path.write_text(yaml.safe_dump(cfg))
            self.assertNotIn('--resume', command(path))
            (home / 'engineering_runs' / 'short').mkdir(parents=True)
            self.assertIn('--resume', command(path))

    def test_v3_recovery_requires_marker_before_deep_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            cfg = {'portable': {'variant': POSITION_VARIANTS[1], 'world_size': 2}}
            for step in (1, 1000):
                (run / 'checkpoints' / f'checkpoint-{step}').mkdir(parents=True)
            calls = []
            def validate(path, config):
                calls.append(path.name)
            with self.assertRaises(RuntimeError):
                choose_latest(run, cfg, validate)
            self.assertEqual(calls, [])
            marker = run / 'checkpoints/checkpoint-1/PORTABLE_COMPLETE.json'
            marker.write_text(json.dumps({'step': 1, 'world': 2}))
            selected, rejected = choose_latest(run, cfg, validate)
            self.assertEqual(selected.name, 'checkpoint-1')
            self.assertEqual(rejected[0]['step'], 1000)
            marker.write_text(json.dumps({'step': 1, 'world': 8}))
            with self.assertRaises(RuntimeError):
                choose_latest(run, cfg, validate)


if __name__ == '__main__':
    unittest.main()
