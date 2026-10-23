from pathlib import Path
import datetime,json,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
from collect_fslgpu_acceptance import collect
class CollectorTests(unittest.TestCase):
    def check(self,hostname):
        start='2026-10-23T08:06:44Z';epoch=datetime.datetime.fromisoformat(start.replace('Z','+00:00')).timestamp()
        app={'item':{'state':'RUNNING'},'instances':{'instances':[{'name':'live-host','state':'RUNNING','last_started_time':start}]}}
        data={'current_attempt.json':{'hostname':hostname,'started_unix':epoch+1},'FSLGPU_EVALUATION_ACCEPTED.json':{'passed':True}}
        state={'fslgpu_acceptance':{'app_id':'app','target_root':'/data/xiangyuyue/ULLM-zf/RobotVla/probe'}}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'checks/stamp';out.mkdir(parents=True)
            with patch('collect_fslgpu_acceptance.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(data))):
                return collect(state,out,lambda path:app,lambda p,v:p.write_text(json.dumps(v)))
    def test_current_instance_acceptance(self):self.assertTrue(self.check('live-host')['accepted'])
    def test_replaced_instance_rejects_previous_acceptance(self):self.assertFalse(self.check('old-host')['accepted'])
if __name__=='__main__':unittest.main()
