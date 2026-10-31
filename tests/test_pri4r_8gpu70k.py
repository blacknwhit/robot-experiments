"""Public 8GPU70k recipe, launch/resume identity and fixed evaluation coverage."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from configure import render
from pri4r_contract import validate_recipe
from pri4r_profiles import PROFILE_8GPU70K
from pri4r_preflight import check_profile,complete_checkpoint
from train import launch_configuration,training_environment
from train_or_resume import command
from checkpoint import canonical_recipe,choose_latest


class EightGPU70kTests(unittest.TestCase):
    def make(self,**kwargs):
        args=dict(variant='pri4r_paper',home=Path(tempfile.gettempdir())/'pri4r70k-resources',
                  run_id='paper70k',world=8,stop=70000,formal=True,training_profile=PROFILE_8GPU70K,
                  experiment_root=Path(tempfile.gettempdir())/'pri4r70k-experiment')
        args.update(kwargs);return render(**args)

    def test_effective_budget_batch_schedule_and_baseline_fields(self):
        c=self.make();check_profile(c);t=c['trainer'];d=c['datasets']['vla_data']
        self.assertEqual((t['max_train_steps'],t['stop_after_steps']), (70000,70000))
        self.assertEqual((d['per_device_batch_size'],t['gradient_accumulation_steps'],t['expected_global_batch_size']),(16,1,128))
        self.assertEqual((t['num_warmup_steps'],t['learning_rate']['base'],t['ema_decay']),(10000,5e-5,.999))
        baseline=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
        for key in ('save_interval','optimizer','logging_frequency','gradient_clipping','save_first_update'):
            self.assertEqual(t[key],baseline['trainer'][key])
        for key in ('num_workers','prefetch_factor','official_filter_key','official_filter_seed','official_weight_alpha','data_mix'):
            self.assertEqual(d[key],baseline['datasets']['vla_data'][key])
        self.assertTrue(c['framework']['pi05']['gradient_checkpointing'])
        self.assertEqual(t['ema_device'],'cuda')

    def test_wrong_topology_mode_or_stop_rejected(self):
        for args in ({'world':4},{'world':1},{'stop':30000},{'stop':70001},
                     {'formal':False,'smoke':True},{'variant':'baseline'}):
            with self.assertRaises(ValueError):self.make(**args)

    def test_accumulation_cannot_be_reenabled_even_if_global_batch_matches(self):
        c=self.make();c['trainer']['gradient_accumulation_steps']=2
        c['datasets']['vla_data']['per_device_batch_size']=8
        with self.assertRaisesRegex(ValueError,'contract differs'):validate_recipe(c)

    def test_budget_and_baseline_settings_cannot_silently_drift(self):
        for field,value in (('max_train_steps',30000),('stop_after_steps',30000),('save_interval',1000)):
            c=self.make();c['trainer'][field]=value
            with self.assertRaises(ValueError):check_profile(c)
        for field,value in (('num_workers',4),('prefetch_factor',2)):
            c=self.make();c['datasets']['vla_data'][field]=value
            with self.assertRaises(ValueError):check_profile(c)

    def test_original30k_and_single_gpu_engineering_recipe_stay_separate(self):
        c=render('pri4r_paper',Path(tempfile.gettempdir())/'p','historical',4,30000,formal=True)
        validate_recipe(c);self.assertEqual(c['trainer']['max_train_steps'],30000)
        c=render('pri4r_paper',Path(tempfile.gettempdir())/'p','smoke',1,3,smoke=True)
        validate_recipe(c);self.assertEqual(c['trainer']['gradient_accumulation_steps'],32)
        with self.assertRaises(ValueError):check_profile(c)

    def test_eight_rank_launcher_and_same_command_resume(self):
        c=self.make();cmd,backend,ds=launch_configuration(c,Path('resolved.yaml'),Path('control'))
        self.assertEqual(cmd[cmd.index('--nproc_per_node')+1],'8')
        self.assertIsNone(ds);self.assertEqual(backend['mixed_precision'],'no')
        self.assertEqual(training_environment(c)['ACCELERATE_USE_DEEPSPEED'],'false')
        with tempfile.TemporaryDirectory() as tmp:
            c['run_root_dir']=str(Path(tmp)/'runs');path=Path(tmp)/'recipe.yaml'
            path.write_text(yaml.safe_dump(c));self.assertNotIn('--resume',command(path))
            (Path(c['run_root_dir'])/c['run_id']).mkdir(parents=True)
            self.assertIn('--resume',command(path))

    def test_old30k_checkpoint_cannot_be_selected_for_new_budget(self):
        c=self.make()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cp=root/'checkpoints/checkpoint-5000';cp.mkdir(parents=True)
            old=copy.deepcopy(c);old['trainer']['max_train_steps']=30000
            (cp/'PORTABLE_COMPLETE.json').write_text(json.dumps({'step':5000,'world':8,'recipe':canonical_recipe(old)}))
            with self.assertRaisesRegex(RuntimeError,'recipe/source/point timing policy differs'):
                choose_latest(root,c,validator=lambda *args:None)

    def test_incomplete70k_cannot_be_used_as_final_evaluation(self):
        c=self.make()
        with self.assertRaisesRegex(ValueError,'not completed'):
            complete_checkpoint(c,selector=lambda *_:(Path('checkpoints/checkpoint-65000'),[]))
        done=complete_checkpoint(c,selector=lambda *_:(Path('checkpoints/checkpoint-70000'),[]))
        self.assertTrue(done['ema_model'].endswith('steps_70000_model.safetensors'))

    def test_documented_cli_generates_real_profile_and_eight_gpu_eval_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=root/'resolved.yaml'
            subprocess.run([sys.executable,str(ROOT/'scripts/configure.py'),'--variant','pri4r_paper',
                '--training-profile',PROFILE_8GPU70K,'--home',str(root/'home'),'--experiment-root',str(root/'exp'),
                '--run-id','h8_70k','--gpus','8','--formal','--stop','70000','--out',str(cfg)],check=True,capture_output=True)
            c=yaml.safe_load(cfg.read_text());check_profile(c)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--config',str(cfg),'--dry-run'],
                check=True,capture_output=True,text=True)
            launch=json.loads(result.stdout)
            self.assertEqual((launch['world'],launch['micro_batch'],launch['accumulation'],launch['cosine_decay_steps']),(8,16,1,70000))
            args=[sys.executable,str(ROOT/'scripts/evaluate_parallel.py'),'--variant','pri4r_paper',
                '--home',str(root/'home'),'--config',str(cfg),'--checkpoint',str(root/'steps_70000_model.safetensors'),
                '--sim-python',sys.executable,'--output',str(root/'eval'),'--all-tasks','--gpu-ids','0,1,2,3,4,5,6,7',
                '--egl-device-ids','0,1,2,3,4,5,6,7','--n-envs','16','--tasks-per-gpu','50',
                '--simulator-cpu-threads','1','--rgb-schedule','final','--initial-rgb','deferred','--timeout','172800','--dry-run']
            preview=json.loads(subprocess.run(args,check=True,capture_output=True,text=True).stdout)
            self.assertEqual((len(preview['tasks']),len(preview['episode_ids']),len(preview['gpus'])),(50,50,8))
            self.assertEqual(preview['workers_per_gpu'],16)


if __name__=='__main__':unittest.main()
