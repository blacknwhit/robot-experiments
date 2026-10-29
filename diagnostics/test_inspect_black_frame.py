import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('inspect_black_frame',Path(__file__).with_name('inspect_black_frame.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class InspectionTests(unittest.TestCase):
    def test_real_artifact_layout_reports_stage_and_only_target_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'parallel-runs/a/simulation/episodes/WaffleReheat/0';target.mkdir(parents=True)
            (target/'SIMULATION_ERROR.json').write_text(json.dumps({'phase':'simulation','episode':0,'traceback':'All-zero RGB'}))
            (target/'env0.initial.json').write_text(json.dumps({'scene_seed':288272014,'health':{'finite_state':True}}))
            (root/'parallel-runs/a/STARTED.json').write_text(json.dumps({
                'gpu':'3','egl':'1','tasks':['WaffleReheat'],'command':['sim','--n-envs','16']}))
            np.savez(target/'failed_observation.npz',**{'video.robot0_eye_in_hand':np.zeros((1,256,256,3),np.uint8),
                     'video.robot0_agentview_left':np.ones((1,256,256,3),np.uint8)})
            other=root/'parallel-runs/a/simulation/episodes/CloseFridge/0';other.mkdir(parents=True)
            (other/'SIMULATION_ERROR.json').write_text('{}')
            result=module.inspect(root,'WaffleReheat',0);self.assertEqual(len(result['matching_errors']),1)
            row=result['matching_errors'][0];self.assertEqual(row['phase'],'simulation')
            self.assertEqual(row['scene_seed'],288272014)
            self.assertTrue(row['scene_seed_matches'])
            self.assertEqual(result['batch_launches'][0]['gpu'],'3')
            self.assertEqual(result['batch_launches'][0]['egl'],'1')
            self.assertEqual(result['batch_launches'][0]['command'],['sim','--n-envs','16'])
            self.assertTrue(row['failed_camera_arrays']['video.robot0_eye_in_hand']['all_zero'])
            self.assertFalse(row['failed_camera_arrays']['video.robot0_agentview_left']['all_zero'])
            self.assertIsNone(row['initial_camera_arrays'])

    def test_no_error_is_not_claimed_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(module.inspect(tmp,'WaffleReheat',0)['status'],'no_matching_public_pipeline_artifact')

    def test_object_arrays_are_not_deserialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'unsafe.npz';np.savez(path,**{'video.camera':np.array([object()],dtype=object)})
            with self.assertRaises(ValueError):module.camera_stats(path)


if __name__=='__main__':unittest.main()
