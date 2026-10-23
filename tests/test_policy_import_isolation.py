import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import common


class PolicyImportIsolationTests(unittest.TestCase):
    def test_v3_policy_import_prefers_shim_even_with_real_simulator_in_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            real = runtime / 'robocasa'
            shim = runtime / 'examples/pi05_robocasa365/server_compat/robocasa'
            real.mkdir(); shim.mkdir(parents=True)
            (real / '__init__.py').write_text("raise RuntimeError('wrong simulator imported by policy')")
            (shim / '__init__.py').write_text("source = 'policy-shim'")
            old_path = list(sys.path)
            old_module = sys.modules.pop('robocasa', None)
            try:
                with patch.object(common, 'runtime', return_value=runtime):
                    common.activate('embedding3d_pos_a')
                    package = importlib.import_module('robocasa')
                    self.assertEqual(package.source, 'policy-shim')
                    self.assertEqual(Path(package.__file__), shim / '__init__.py')
            finally:
                sys.path[:] = old_path
                sys.modules.pop('robocasa', None)
                if old_module is not None:
                    sys.modules['robocasa'] = old_module

    def test_v3_subprocess_path_order_matches_activation_and_simulator_excludes_shim(self):
        for variant in common.POSITION_VARIANTS:
            expected = list(map(str, common.import_paths(variant)))
            self.assertIn('server_compat', expected[0])
            self.assertEqual(common.environment(variant)['PYTHONPATH'].split(os.pathsep), expected)
            simulator = common.environment(variant, shim=False, overlay=False)['PYTHONPATH']
            self.assertNotIn('server_compat', simulator)
            self.assertEqual(simulator.split(os.pathsep)[0], str(common.runtime(variant)))

    def test_historical_import_order_is_preserved(self):
        for variant in ('baseline', 'query_fixed', 'query3d_v2', 'embedding3d_v2'):
            paths = common.import_paths(variant)
            self.assertEqual(paths[0], common.runtime(variant) / 'runtime')
            self.assertIn('server_compat', str(paths[-1]))


if __name__ == '__main__':
    unittest.main()
