import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('position_collector', Path(__file__).with_name('collect_status.py'))
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def check_terminal(self, attempt_start):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = dict(native_monitor_id='retired', current_job='known-job',
                         target_root='/data/xiangyuyue/ULLM-zf/RobotVla/test')
            (root / 'status.json').write_text(json.dumps(state))
            def get(route):
                if 'auto-retrain' in route:
                    return dict(ok=True, monitors=[])
                if '/detail?' in route:
                    return dict(ok=True, item=dict(state='SUCCEEDED', startTime='2026-10-20T19:08:00Z'))
                if '/workers?' in route:
                    return dict(ok=True, items=[dict(name='worker-known', phase='Succeeded')])
                return dict(ok=True, items=[dict(body='known application failure',
                    attributes={'lepton.workload-name': 'known-job'})])
            shared = {'current_attempt.json': dict(hostname='worker-known', started_unix=attempt_start),
                      'HOLD_FOR_MAIN.json': dict(error='known application failure')}
            result = SimpleNamespace(returncode=0, stdout=json.dumps(shared), stderr='')
            with patch.object(collector, 'D', root), patch.object(collector, 'get', get), \
                 patch.object(collector.subprocess, 'run', return_value=result), contextlib.redirect_stdout(io.StringIO()):
                collector.main()
            return json.loads((root / 'latest_status.json').read_text())

    def test_missing_native_keeps_terminal_diagnosis_without_claiming_current_owner(self):
        report = self.check_terminal(1792523286.0)
        self.assertFalse(report['identity_verified'])
        self.assertEqual(report['evidence_scope'], 'last_known_job_audit_no_native_owner')
        self.assertTrue(report['attempt_matches_current_worker'])
        self.assertIn('HOLD_FOR_MAIN.json', report['application'])

    def test_older_attempt_start_rejects_shared_evidence(self):
        report = self.check_terminal(1700000000.0)
        self.assertFalse(report['attempt_matches_current_worker'])
        self.assertNotIn('application', report)


if __name__ == '__main__':
    unittest.main()
