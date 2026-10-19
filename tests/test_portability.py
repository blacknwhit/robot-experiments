import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from configure import render
from checkpoint import choose_latest,required_files,validate
from evaluate import TASKS,tasks_for_part,validate_result

class PortabilityTests(unittest.TestCase):
    def test_scientific_recipes(self):
        for v in ['baseline','query_fixed','query3d_v2','embedding3d_v2']:
            for n in [2,4,8]:
                c=render(v,'/tmp/robot-test','test',n)
                self.assertEqual(c['datasets']['vla_data']['per_device_batch_size']*n,64)
                self.assertEqual(c['trainer']['gradient_accumulation_steps'],1)
                self.assertEqual(c['trainer']['max_train_steps'],100000)
                self.assertEqual(c['trainer']['stop_after_steps'],70000)
                self.assertEqual(c['seed'],42)
        fixed=render('query_fixed','/tmp/robot-test','fixed',8)
        self.assertTrue(fixed['datasets']['vla_data']['foresight_targets'])
        self.assertEqual(fixed['framework']['pi05']['foresight']['teacher_ema_decay'],1.0)
        self.assertEqual(fixed['framework']['pi05']['foresight']['num_queries'],64)
        self.assertEqual(render('query3d_v2','/tmp/x','q',2)['framework']['pi05']['point_track']['loss_weight'],3.75)
        self.assertEqual(render('embedding3d_v2','/tmp/x','e',2)['framework']['pi05']['point_track']['loss_weight'],4.0)

    def test_no_overlap_or_lost_evaluation_tasks(self):
        parts=[tasks_for_part(i) for i in range(1,14)]
        self.assertEqual([len(p) for p in parts],[4]*12+[2])
        names=[t['task'] for p in parts for t in p]
        self.assertEqual(len(set(names)),50)
        self.assertEqual([t['index'] for p in parts for t in p],list(range(50)))

    def test_incomplete_newer_checkpoint_is_skipped_and_wrong_world_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg={'portable':{'variant':'baseline','world_size':2}}
            for step in [1000,2000]:
                cp=root/'checkpoints'/f'checkpoint-{step}';cp.mkdir(parents=True)
                for n in required_files(2):
                    f=cp/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'x')
                (cp/'latest').write_text('pytorch_model')
                (cp.parent/f'steps_{step}_model.safetensors').write_bytes(b'x')
            (root/'checkpoints/checkpoint-2000/random_states_1.pkl').unlink()
            selected,rejected=choose_latest(root,cfg,lambda p,c:validate(p,c,deep=False))
            self.assertEqual(selected.name,'checkpoint-1000');self.assertEqual(rejected[0]['step'],2000)
            cfg['portable']['world_size']=8
            with self.assertRaises(RuntimeError):choose_latest(root,cfg,lambda p,c:validate(p,c,deep=False))

    def test_raw_acceptance_rejects_duplicate_traces_and_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            task=TASKS[0];cp=Path(tmp)/'weights';p=Path(tmp)/'result.json'
            d={'env':'robocasa/'+task['task'],'env_split':'pretrain','robocasa_version':'1.0.0','profile':'published_v1_0_0','seed':7,
                'n_envs':1,'max_episode_steps':task['horizon'],'horizon_multiplier':1.5,'n_action_steps':5,'num_trials':2,
                'strict_determinism':True,'trace_determinism':True,'renderer_backend':'egl','checkpoint':str(cp.resolve()),
                'policy_rng_mode':'stateless_request_v1','policy_seed_version':'sha256_u32_v1','successes':[True,False],
                'num_successes':1,'episode_trace_sha256':['a'*64,'b'*64],'protocol_comparable':False}
            p.write_text(json.dumps(d));validate_result(p,task,2,cp)
            d['policy_server_metadata']={'variant':'query_fixed','checkpoint_sha256':'foreign','config_sha256':'cfg','strict_full_state':{'all_loaded_values_exact_at_model_dtype':True}}
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):validate_result(p,task,2,cp,{'variant':'query_fixed','checkpoint_sha256':'correct','config_sha256':'cfg'})
            d['episode_trace_sha256']=['a'*64]*2;p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):validate_result(p,task,2,cp)
            d['episode_trace_sha256']=['a'*64,'b'*64];d['checkpoint']='foreign';p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):validate_result(p,task,2,cp)
if __name__=='__main__':unittest.main()
