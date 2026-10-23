import copy
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import os
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import POSITION_VARIANTS
from configure import render
from checkpoint import choose_latest
from train_or_resume import command
from release_contract import verify_release
from train import training_environment


class PositionHandoffTests(unittest.TestCase):
    def test_isolated_outputs_reuse_resources_and_keep_scientific_recipe(self):
        with tempfile.TemporaryDirectory() as folder:
            home, output = Path(folder) / 'existing-resources', Path(folder) / 'new-abc'
            gt = home / 'data/gt_armobj_full_v1'
            gt.mkdir(parents=True)
            sampling = gt / 'sampling_config.json'
            sampling.write_text(json.dumps({'asset_root': str(home), 'seed': 42}))
            before = sampling.read_bytes()
            roots = []
            for variant in POSITION_VARIANTS:
                for mode, subdir, stop in [('smoke', 'smoke_runs', 2),
                                            ('engineering', 'engineering_runs', 1500),
                                            ('formal', 'runs', 70000)]:
                    base = render(variant, home, variant, 2, stop=stop, **{mode: True})
                    isolated = render(variant, home, variant, 2, stop=stop,
                                      experiment_root=output, **{mode: True})
                    self.assertEqual(Path(isolated['run_root_dir']), output / subdir)
                    self.assertEqual(isolated['framework'], base['framework'])
                    self.assertEqual(isolated['datasets'], base['datasets'])
                    self.assertEqual(isolated['trainer'], base['trainer'])
                    self.assertEqual(training_environment(isolated)['OPENPI_DATA_HOME'],
                                     str(home / 'cache/openpi'))
                    normalized = copy.deepcopy(isolated)
                    normalized['run_root_dir'] = base['run_root_dir']
                    normalized['portable'].pop('resource_home')
                    normalized['portable'].pop('experiment_root')
                    self.assertEqual(normalized, base)
                    roots.append(str(Path(isolated['run_root_dir']) / isolated['run_id']))
            self.assertEqual(len(roots), len(set(roots)))
            self.assertEqual(sampling.read_bytes(), before)
            self.assertFalse(output.exists())  # rendering never creates a run or copies resources

    def test_isolated_run_resume_does_not_select_existing_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            home, output = Path(folder) / 'resources', Path(folder) / 'abc'
            (home / 'runs/shared-name').mkdir(parents=True)
            cfg = render(POSITION_VARIANTS[0], home, 'shared-name', 2, formal=True,
                         experiment_root=output)
            file = Path(folder) / 'config.yaml'; file.write_text(yaml.safe_dump(cfg))
            self.assertNotIn('--resume', command(file))
            (output / 'runs/shared-name').mkdir(parents=True)
            self.assertIn('--resume', command(file))

    def test_output_option_is_scoped_and_legacy_cache_path_is_unchanged(self):
        for variant in ('baseline', 'query_fixed', 'query3d_v2', 'embedding3d_v2'):
            with self.assertRaises(ValueError):
                render(variant, '/tmp/resources', 'run', 2, experiment_root='/tmp/abc')
            cfg = render(variant, '/tmp/resources', 'run', 2)
            self.assertNotIn('resource_home', cfg['portable'])
            self.assertEqual(training_environment(cfg)['OPENPI_DATA_HOME'],
                             str(Path(cfg['run_root_dir']).parent / 'cache/openpi'))
        with self.assertRaises(ValueError):
            render(POSITION_VARIANTS[0], '/tmp/resources', 'run', 2, formal=True,
                   experiment_root='/tmp/resources')

    def test_training_launch_does_not_depend_on_inherited_cublas_configuration(self):
        for variant in POSITION_VARIANTS:
            cfg=render(variant,'/tmp/position','short',2,stop=1500,engineering=True)
            with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':'invalid'},clear=False):
                self.assertEqual(training_environment(cfg)['CUBLAS_WORKSPACE_CONFIG'],':4096:8')

    def test_current_source_and_recipe_must_both_match_before_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'provenance').mkdir()
            code=root/'source.py';code.write_bytes(b'x = 1\n')
            manifest=root/'provenance/source_files.sha256'
            manifest.write_text(hashlib.sha256(code.read_bytes()).hexdigest()+'  source.py\n')
            cfg={'trajectory_release_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
            self.assertEqual(verify_release(cfg,root),1)
            code.write_bytes(b'x = 2\n')
            with self.assertRaisesRegex(ValueError,'source changed'):verify_release(cfg,root)
            code.write_bytes(b'x = 1\n')
            cfg['trajectory_release_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'another code release'):verify_release(cfg,root)

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
