import datetime,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collect_native_diagnostic import collect

class NativeCollectorTests(unittest.TestCase):
    def snapshot(self,hostname,live=True):
        start='2026-10-22T23:00:00Z';epoch=datetime.datetime.fromisoformat(start.replace('Z','+00:00')).timestamp()
        job='pt-current' if live else 'pt-last'
        state={'reader':'fslgpu','diagnostic':{'native_monitor_id':'monitor','current_job':'pt-last',
            'target_root':'/data/xiangyuyue/ULLM-zf/RobotVla/diagnostic'}}
        native={'monitors':[{'id':'monitor','currentJobName':'pt-current'}] if live else []}
        app={'current_attempt.json':{'hostname':hostname,'started_unix':epoch+2},
            'phase.json':{'phase':'native_debug_replay'}}
        def get(path):
            self.assertIn('/'+job+'/',path)
            if '/detail?' in path:return {'item':{'state':'RUNNING' if live else 'SUCCEEDED','startTime':start}}
            if '/workers?' in path:return {'items':[{'name':'worker-real','phase':'Running' if live else 'Succeeded'}]}
            return {'items':[{'attributes':{'lepton.workload-name':job}},{'attributes':{'lepton.workload-name':'foreign'}}]}
        def save(path,value):path.write_text(json.dumps(value))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'checks/stamp';out.mkdir(parents=True)
            with patch('collect_native_diagnostic.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(app))):
                result=collect(state,native,out,get,save)
            report=json.loads((out/'native_diagnostic/report.json').read_text())
            self.assertEqual(len(json.loads((out/'native_diagnostic/strict_logs.json').read_text())),1)
            return result,report
    def test_current_identity_accepts_matching_worker_and_time(self):
        result,report=self.snapshot('worker-real')
        self.assertTrue(result['attempt_matches_worker']);self.assertIn('application',report)
        self.assertEqual(result['job'],'pt-current')
    def test_stale_worker_evidence_is_rejected(self):
        result,report=self.snapshot('worker-old')
        self.assertFalse(result['attempt_matches_worker']);self.assertNotIn('application',report)
    def test_removed_monitor_is_historical_not_live(self):
        result,report=self.snapshot('worker-real',False)
        self.assertFalse(result['live_native_identity']);self.assertEqual(result['evidence_scope'],'last_known_diagnostic_job')
        self.assertTrue(result['attempt_matches_worker'])
if __name__=='__main__':unittest.main()
