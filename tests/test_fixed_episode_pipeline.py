import ast,concurrent.futures,copy,json,random,socket,sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from episode_plan import rng_snapshot,restore_rng,load_plan,build_plan_set,load_plan_set
from evaluate_task import validate_fixed,check_port_available
import evaluate_task
from fixed_episode_sim import episode_partitions

class FixedEpisodeTests(unittest.TestCase):
    def test_every_environment_count_preserves_all_50_episode_ids(self):
        for count in range(1,51):
            groups=episode_partitions(50,count)
            self.assertEqual(len(groups),count)
            self.assertEqual(sorted(i for group in groups for i in group),list(range(50)))
            self.assertTrue(all(groups))
            self.assertLessEqual(max(map(len,groups))-min(map(len,groups)),1)
        for count in (0,51,-1,True):
            with self.assertRaises(ValueError):episode_partitions(50,count)
        with self.assertRaises(ValueError):episode_partitions(2,3)

    @unittest.skipUnless(sys.platform.startswith('linux'),'Real output and per-task flock')
    def test_direct_formal_launch_without_smoke_keeps_identity_and_plan_guards(self):
        import hashlib
        class ReachedPolicy(Exception):pass
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);home=root/'home';(home/'receipts').mkdir(parents=True)
            freeze=home/'receipts/sim.freeze.txt';freeze.write_text('test-runtime\n')
            config=root/'config.yaml';config.write_text('portable:\n  variant: baseline\n')
            checkpoint=root/'weights';checkpoint.write_bytes(b'checkpoint-one')
            plans=root/'plans';plans.mkdir()
            plan=json.loads((ROOT/'provenance/fixed_scene_reference_v2/CloseBlenderLid.json').read_text())
            plan['sim_freeze_sha256']=hashlib.sha256(freeze.read_bytes()).hexdigest()
            plan_path=plans/'CloseBlenderLid.json';plan_path.write_text(json.dumps(plan));build_plan_set(plans)
            base=['evaluate_task.py','--variant','baseline','--home',str(home),'--config',str(config),
                  '--checkpoint',str(checkpoint),'--sim-python',sys.executable,'--task','CloseBlenderLid',
                  '--scene-protocol','shared-scenes','--plan-dir',str(plans)]
            for count in (3,6,12,50):
                out=root/f'n{count}';args=base+['--output',str(out),'--n-envs',str(count)]
                with patch.object(sys,'argv',args),patch.object(evaluate_task,'activate',side_effect=ReachedPolicy):
                    with self.assertRaises(ReachedPolicy):evaluate_task.main()
                self.assertFalse((out/'SMOKE_ACCEPTED.json').exists())
                self.assertEqual(json.loads((out/'EVALUATION_IDENTITY.json').read_text())['n_envs'],count)
            checkpoint.write_bytes(b'checkpoint-two')
            args=base+['--output',str(root/'n6'),'--n-envs','6']
            with patch.object(sys,'argv',args),patch.object(evaluate_task,'activate') as launch:
                with self.assertRaisesRegex(ValueError,'Output identity differs'):evaluate_task.main()
                launch.assert_not_called()
            checkpoint.write_bytes(b'checkpoint-one')
            plan['resets'][0]['initial_state_sha256']='0'*64;plan_path.write_text(json.dumps(plan))
            with patch.object(sys,'argv',args),patch.object(evaluate_task,'activate') as launch:
                with self.assertRaisesRegex(ValueError,'file changed'):evaluate_task.main()
                launch.assert_not_called()
            legacy=[x for x in base[:base.index('--scene-protocol')]]+['--output',str(root/'legacy')]
            with patch.object(sys,'argv',legacy),patch.object(evaluate_task,'activate') as launch:
                with self.assertRaisesRegex(ValueError,'matching task --smoke'):evaluate_task.main()
                launch.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith('linux'),'Linux deployment socket semantics')
    def test_reuses_time_wait_but_rejects_live_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1];listener.listen()
            with self.assertRaises(OSError):check_port_available(port)
            with socket.create_connection(('127.0.0.1',port)) as client:
                connection,_=listener.accept()
                with connection:connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1),b'')
        check_port_available(port)
    def test_json_rng_roundtrip_preserves_all_three_streams(self):
        env=SimpleNamespace(rng=np.random.default_rng(7));random.seed(19);np.random.seed(23)
        snapshot=json.loads(json.dumps(rng_snapshot(env)))
        expected=(env.rng.random(20),np.random.random(20),[random.random() for _ in range(20)])
        env.rng.random(200);np.random.random(200);random.random()
        restore_rng(env,snapshot)
        self.assertTrue(np.array_equal(expected[0],env.rng.random(20)))
        self.assertTrue(np.array_equal(expected[1],np.random.random(20)))
        self.assertEqual(expected[2],[random.random() for _ in range(20)])

    def test_plan_rejects_duplicate_missing_and_unfinished_episodes(self):
        state=rng_snapshot(SimpleNamespace(rng=np.random.default_rng(7)))
        plan={'format':'robocasa_frozen_episode_plan_v1','complete':True,'env':'robocasa/CloseBlenderLid',
            'num_trials':2,'seed':7,'n_action_steps':5,'env_split':'pretrain','max_episode_steps':900,
            'resets':[{'episode':i,'pre_reset_rng':state,'initial_observation_sha256':'a'*64,'initial_state_sha256':'b'*64} for i in range(3)],
            'reference_result':{'num_trials':2,'num_successes':1,'successes':[False,True],'episode_trace_sha256':['a'*64,'b'*64],
                'env':'robocasa/CloseBlenderLid','seed':7,'n_action_steps':5,'env_split':'pretrain','max_episode_steps':900}}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'plan.json';path.write_text(json.dumps(plan));self.assertEqual(load_plan(path,plan['env'],2)['num_trials'],2)
            for mutate in (lambda d:d.update(complete=False),lambda d:d['resets'].pop(),lambda d:d['resets'][1].update(episode=0),
                           lambda d:d['reference_result'].update(successes=[0,1]),lambda d:d['reference_result'].update(seed=8),
                           lambda d:d['reference_result'].update(episode_trace_sha256=['a'*64]*2)):
                broken=copy.deepcopy(plan);mutate(broken);path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):load_plan(path)
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):load_plan(path,trials=50)
            with self.assertRaises(ValueError):load_plan(path,task='robocasa/CloseFridge')
            path.rename(Path(folder)/'CloseBlenderLid.json')
            (Path(folder)/'CloseBlenderLid.inprogress.json').write_text(json.dumps({'complete':False,'resets':[]}))
            digest=build_plan_set(folder);plans,actual=load_plan_set(folder)
            self.assertEqual(digest,actual);self.assertEqual(list(plans),['CloseBlenderLid'])
            path=Path(folder)/'CloseBlenderLid.json';changed=copy.deepcopy(plan);changed['resets'][0]['initial_state_sha256']='c'*64;path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'file changed'):load_plan_set(folder)
            with self.assertRaisesRegex(ValueError,'immutable'):build_plan_set(folder)

    def test_result_requires_exact_episode_coverage_and_scene_hashes(self):
        identity={k:k for k in ('variant','checkpoint_sha256','config_sha256','inference_backend','inference_backend_version')}
        metadata={**identity,'strict_full_state':{'all_loaded_values_exact_at_model_dtype':True},'cuda_graph_verify':True}
        plan={'env':'robocasa/CloseBlenderLid','max_episode_steps':900,'resets':[{'initial_observation_sha256':str(i)*64,'initial_state_sha256':'f'*64} for i in range(5)]}
        data={'complete':True,'env':plan['env'],'episode_protocol':'frozen_scene_plan_v1','episode_plan_sha256':'plan',
            'num_trials':4,'n_envs':4,'logical_environment_index':0,'episode_ids':list(range(4)),'seed':7,'n_action_steps':5,'max_episode_steps':900,
            'successes':[False,True,False,True],'num_successes':2,'episode_trace_sha256':[str(i)*64 for i in range(4)],
            'policy_server_metadata':metadata,'initial_scenes':[{'episode':i,'observation_sha256':str(i)*64,'state_sha256':'f'*64} for i in range(4)]}
        validate_fixed(data,plan,'plan',4,4,identity,True)
        for mutate in (lambda d:d.update(episode_ids=[0,1,1,3]),lambda d:d.update(n_envs=1),lambda d:d.update(episode_plan_sha256='foreign'),
                       lambda d:d['initial_scenes'][2].update(observation_sha256='bad'),lambda d:d['policy_server_metadata'].update(cuda_graph_verify=False)):
            bad=copy.deepcopy(data);mutate(bad)
            with self.assertRaises(ValueError):validate_fixed(bad,plan,'plan',4,4,identity,True)

    def test_actual_queue_preserves_request_mapping_and_exclusive_gpu_on_failure(self):
        tree=ast.parse((ROOT/'scripts/pipelined_policy.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('predict_action','close')]
        ns={'concurrent':concurrent,'time':time};exec(compile(ast.Module(body=methods,type_ignores=[]),'pipelined_policy.py','exec'),ns)
        pipe=SimpleNamespace(cpu=concurrent.futures.ThreadPoolExecutor(1),gpu=concurrent.futures.ThreadPoolExecutor(1),bound=threading.BoundedSemaphore(4),closed=False)
        active=0;maximum=0;lock=threading.Lock()
        def prepare(sample):time.sleep(.002);return sample['value']
        def execute(sample,batch):
            nonlocal active,maximum
            with lock:active+=1;maximum=max(active,maximum)
            try:
                time.sleep(.003)
                if sample['value']==5:raise ValueError('sentinel GPU failure')
                return batch*7
            finally:
                with lock:active-=1
        pipe.prepare=prepare;pipe.execute=execute
        try:
            with concurrent.futures.ThreadPoolExecutor(8) as callers:
                futures=[callers.submit(ns['predict_action'],pipe,[{'__pi05_rng_seed':i,'value':i}]) for i in range(20)]
                for i,f in enumerate(futures):
                    if i==5:
                        with self.assertRaisesRegex(ValueError,'sentinel'):f.result(timeout=5)
                    else:self.assertEqual(f.result(timeout=5),i*7)
            self.assertEqual(maximum,1)
        finally:ns['close'](pipe)
        with self.assertRaisesRegex(RuntimeError,'closed'):ns['predict_action'](pipe,[{'__pi05_rng_seed':0,'value':0}])

if __name__=='__main__':unittest.main()
