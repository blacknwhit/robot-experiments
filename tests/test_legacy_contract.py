import copy,hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from legacy_contract import require_same_policy,require_same_trajectories,historical_inventory,check_historical
from rgb_last_frame import CameraCache,Guard


class LegacyContractTests(unittest.TestCase):
    def reference(self):
        return dict(env='robocasa/CloseBlenderLid',env_split='pretrain',n_envs=1,seed=7,n_action_steps=5,max_episode_steps=900,
            num_trials=50,num_successes=0,successes=[False]*50,episode_trace_sha256=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(50)],
            policy_server_metadata=dict(variant='baseline',checkpoint_sha256='weights',config_sha256='config'))

    def test_foreign_checkpoint_and_config_cannot_use_legacy_plan(self):
        ref=self.reference();identity=ref['policy_server_metadata'];plan={'reference_result':ref}
        plan['format']='robocasa_frozen_episode_plan_v1'
        require_same_policy(plan,identity)
        for key in identity:
            changed={**identity,key:'foreign'}
            with self.assertRaisesRegex(ValueError,'own serial plan'):require_same_policy(plan,changed)

    def test_formal_tail_is_checked_even_if_smoke_prefix_matches(self):
        ref=self.reference();actual=copy.deepcopy(ref)
        require_same_trajectories(actual,ref,50)
        smoke={**actual,'successes':actual['successes'][:8],'episode_trace_sha256':actual['episode_trace_sha256'][:8]}
        require_same_trajectories(smoke,ref,8)
        actual['episode_trace_sha256'][49]='f'*64
        with self.assertRaisesRegex(ValueError,'Legacy content differs'):require_same_trajectories(actual,ref,50)

    def test_historical_files_are_bound_and_missing_weight_proof_is_not_invented(self):
        ref=self.reference();identity=ref['policy_server_metadata'];actual=copy.deepcopy(ref)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'CloseBlenderLid.json';p.write_text(json.dumps(ref))
            files,digest=historical_inventory(directory,[{'task':'CloseBlenderLid'}])
            proof=check_historical(directory,files,'CloseBlenderLid',actual,identity,50)
            self.assertTrue(proof['checkpoint_sha_verified'])
            ref.pop('policy_server_metadata');p.write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'reference changed'):check_historical(directory,files,'CloseBlenderLid',actual,identity,50)
            files,new_digest=historical_inventory(directory,[{'task':'CloseBlenderLid'}]);self.assertNotEqual(digest,new_digest)
            self.assertFalse(check_historical(directory,files,'CloseBlenderLid',actual,identity,50)['checkpoint_sha_verified'])

    def test_no_plan_defaults_to_native_single_environment_and_rejects_parallel(self):
        args=[sys.executable,str(ROOT/'scripts/evaluate_task.py'),'--variant','baseline','--home','.','--config','config','--checkpoint','weights',
              '--sim-python','python','--output','out','--task','CloseBlenderLid','--dry-run','--smoke']
        ok=subprocess.run(args,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(ok.stdout)['n_envs'],1);self.assertEqual(json.loads(ok.stdout)['mode'],'capture')
        bad=subprocess.run(args+['--n-envs','8'],capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0);self.assertIn('own existing serial plan',bad.stderr)

    def test_grouped_history_and_alias_require_exact_weight_identity(self):
        ref=self.reference();ref['policy_server_metadata']['variant']='query'
        identity={**ref['policy_server_metadata'],'variant':'query3d_v2'}
        with tempfile.TemporaryDirectory() as directory:
            group=Path(directory)/'atomic_seen';group.mkdir()
            p=group/'CloseBlenderLid.json';p.write_text(json.dumps(ref))
            files,_=historical_inventory(directory,[{'task':'CloseBlenderLid'}])
            self.assertTrue(check_historical(directory,files,'CloseBlenderLid',ref,identity,50)['checkpoint_sha_verified'])
            with self.assertRaisesRegex(ValueError,'checkpoint_sha256'):
                check_historical(directory,files,'CloseBlenderLid',ref,{**identity,'checkpoint_sha256':'other'},50)
            ref['policy_server_metadata'].pop('checkpoint_sha256');p.write_text(json.dumps(ref))
            files,_=historical_inventory(directory,[{'task':'CloseBlenderLid'}])
            with self.assertRaisesRegex(ValueError,'variant'):check_historical(directory,files,'CloseBlenderLid',ref,identity,50)
            (Path(directory)/'CloseBlenderLid.json').write_text(json.dumps(ref))
            with self.assertRaisesRegex(ValueError,'Ambiguous'):historical_inventory(directory,[{'task':'CloseBlenderLid'}])

    def test_rgb_proof_requires_real_execution_and_correct_step_count(self):
        from evaluate_task import validate_rgb
        good={'simulation_optimization':dict(version='native_last_frame_rgb_v1',control_steps=1800,rgb_rendered=1080,rgb_reused=4320)}
        validate_rgb(good,2,900)
        with self.assertRaisesRegex(ValueError,'execution evidence'):validate_rgb(good,50,900)
        with self.assertRaisesRegex(ValueError,'execution evidence'):validate_rgb({},2,900)

    def test_rgb_cache_does_not_call_sensor_on_unread_steps_and_restores_real_value(self):
        calls=[]
        def sensor(cache):calls.append(cache['tick']);return np.full((2,2,3),cache['tick'],dtype=np.uint8)
        sensor.__modality__='image';guard=SimpleNamespace(skip=False)
        camera=CameraCache(SimpleNamespace(_sensor=sensor,_current_observed_value=np.zeros((2,2,3),dtype=np.uint8)),guard)
        for tick in range(1,6):
            guard.skip=tick<5;out=camera({'tick':tick})
        self.assertEqual(calls,[5]);self.assertTrue(np.array_equal(out,np.full((2,2,3),5,dtype=np.uint8)))
        self.assertEqual((camera.rendered,camera.reused),(1,4))
        wrapper=SimpleNamespace(video_delta_indices=np.array([0]),state_delta_indices=np.array([0]),n_action_steps=5,max_episode_steps=903,reward=[0]*900,done=[])
        self.assertEqual(Guard.chunk_steps(None,wrapper),3)
        wrapper.video_delta_indices=np.array([-1,0])
        with self.assertRaises(ValueError):Guard.chunk_steps(None,wrapper)

if __name__=='__main__':unittest.main()
