"""Global64 must pass launch and native full-state checks, not just YAML parsing."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from configure import render
from pri4r_profiles import PROFILE_8GPU70K_B8,PROFILE_8GPU70K,expected_global_batch,load_profile
from pri4r_contract import validate_recipe
from pri4r_preflight import check_profile
from checkpoint import canonical_recipe,choose_latest
from train import launch_configuration


class B8Tests(unittest.TestCase):
    def make(self,**kwargs):
        args=dict(variant='pri4r_paper',home=Path(tempfile.gettempdir())/'pri4r-b8-resources',
            run_id='h8_b8_70k',world=8,stop=70000,formal=True,training_profile=PROFILE_8GPU70K_B8,
            experiment_root=Path(tempfile.gettempdir())/'pri4r-b8-experiment')
        args.update(kwargs);return render(**args)

    def test_batch_matches_baseline_and_fixed_with_pri4r_lr_ema(self):
        c=self.make();check_profile(c)
        self.assertEqual((c['datasets']['vla_data']['per_device_batch_size'],c['trainer']['gradient_accumulation_steps'],expected_global_batch(c)),(8,1,64))
        for variant in ('baseline','query_fixed'):
            base=render(variant,Path(tempfile.gettempdir())/'r','baseline',8)
            self.assertEqual(base['datasets']['vla_data']['per_device_batch_size'],8)
            self.assertEqual(expected_global_batch(base),expected_global_batch(c))
        self.assertEqual((c['trainer']['max_train_steps'],c['trainer']['stop_after_steps']),(70000,70000))
        self.assertEqual((c['trainer']['learning_rate']['base'],c['trainer']['num_warmup_steps'],c['trainer']['ema_decay']),(5e-5,10000,.999))

    def test_checkpointing_matches_selected_profile(self):
        c=self.make();profile=load_profile(PROFILE_8GPU70K_B8)
        self.assertEqual(c['framework']['pi05']['gradient_checkpointing'],profile['gradient_checkpointing'])
        self.assertEqual(c['framework']['pi05']['point_track']['checkpoint_fusion'],profile['checkpoint_fusion'])
        c['framework']['pi05']['gradient_checkpointing']=not profile['gradient_checkpointing']
        with self.assertRaisesRegex(ValueError,'checkpointing'):validate_recipe(c)

    def test_accumulation_and_global_field_cannot_bypass_profile(self):
        c=self.make();c['datasets']['vla_data']['per_device_batch_size']=4;c['trainer']['gradient_accumulation_steps']=2
        with self.assertRaises(ValueError):validate_recipe(c)
        c=self.make();c['trainer']['expected_global_batch_size']=128
        self.assertEqual(expected_global_batch(c),64)
        with self.assertRaises(ValueError):validate_recipe(c)

    def test_old_recipe_and_topology_stay_separate(self):
        old=self.make(training_profile=PROFILE_8GPU70K)
        validate_recipe(old);self.assertEqual(expected_global_batch(old),128)
        self.assertEqual(old['datasets']['vla_data']['per_device_batch_size'],16)
        with self.assertRaises(ValueError):self.make(world=4)
        with self.assertRaises(ValueError):self.make(stop=30000)

    def test_actual_launcher_admits_b8_global64(self):
        c=self.make();cmd,backend,ds=launch_configuration(c,Path('recipe.yaml'),Path('control'))
        self.assertEqual(cmd[cmd.index('--nproc_per_node')+1],'8');self.assertIsNone(ds)
        self.assertEqual(backend['backend'],'native_ddp')

    def test_b16_state_is_not_silently_resumed_as_b8(self):
        new=self.make();old=self.make(training_profile=PROFILE_8GPU70K)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cp=root/'checkpoints/checkpoint-5000';cp.mkdir(parents=True)
            (cp/'PORTABLE_COMPLETE.json').write_text(json.dumps({'step':5000,'world':8,'recipe':canonical_recipe(old)}))
            with self.assertRaisesRegex(RuntimeError,'recipe/source/point timing policy differs'):
                choose_latest(root,new,validator=lambda *args:None)

    def test_native_global64_save_restore_and_wrong_metadata_rejection(self):
        # Tiny real Adam update/save/restore with an explicit eight-rank metadata
        # fixture. This tests serialization contracts, not eight-device execution.
        import random,numpy as np,torch
        from safetensors.torch import save_file
        from pri4r_native_state import write_metadata,validate_native,audit_restored,restore_rng_strict
        c=self.make();model=torch.nn.Linear(3,2)
        spec=c['trainer']['optimizer']
        opt=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=tuple(spec['betas']),eps=spec['eps'],weight_decay=spec['weight_decay'])
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        model(torch.randn(4,3)).square().mean().backward();opt.step();opt.zero_grad();scheduler.step()
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint-1';cp.mkdir()
            save_file(model.state_dict(),str(cp/'model.safetensors'))
            torch.save(opt.state_dict(),cp/'optimizer.bin');torch.save(scheduler.state_dict(),cp/'custom_checkpoint_0.pkl')
            torch.save({'completed_steps':1,'consumed_batches':1},cp/'custom_checkpoint_1.pkl')
            for rank in range(8):
                random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank)
                torch.save({'random_state':random.getstate(),'numpy_random_seed':np.random.get_state(),
                    'torch_manual_seed':torch.get_rng_state(),'torch_cuda_manual_seed':[]},cp/f'random_states_{rank}.pkl')
            accelerator=SimpleNamespace(num_processes=8,gradient_accumulation_steps=1,process_index=0,unwrap_model=lambda m:m)
            fake=SimpleNamespace(model=model,optimizer=opt,accelerator=accelerator,total_batch_size=64,
                completed_steps=1,resume_from_checkpoint=str(cp),config=SimpleNamespace(output_dir=tmp))
            write_metadata(fake,cp);validate_native(cp,c,1)
            original=copy.deepcopy(model.state_dict());saved=copy.deepcopy(opt.state_dict())
            for p in model.parameters():p.data.zero_()
            for state in opt.state.values():state['exp_avg'].zero_()
            model.load_state_dict(original);opt.load_state_dict(saved);restore_rng_strict(cp,0);audit_restored(fake)
            self.assertTrue(json.loads((Path(tmp)/'native_restore_rank0.json').read_text())['model_adam_rng_bitwise_equal'])
            meta=json.loads((cp/'native_state.json').read_text());meta['global_batch']=128
            (cp/'native_state.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'topology/clocks'):validate_native(cp,c,1)

    def test_real_configure_and_training_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=root/'resolved.yaml'
            subprocess.run([sys.executable,str(ROOT/'scripts/configure.py'),'--variant','pri4r_paper',
                '--training-profile',PROFILE_8GPU70K_B8,'--home',str(root/'home'),'--experiment-root',str(root/'exp'),
                '--run-id','h8_b8_70k','--gpus','8','--formal','--stop','70000','--out',str(cfg)],check=True,capture_output=True)
            launch=json.loads(subprocess.run([sys.executable,str(ROOT/'scripts/train.py'),'--config',str(cfg),'--dry-run'],
                check=True,capture_output=True,text=True).stdout)
            self.assertEqual((launch['world'],launch['micro_batch'],launch['accumulation'],launch['global_batch']),(8,8,1,64))
            self.assertEqual(launch['cosine_decay_steps'],70000)
            self.assertEqual(launch['gradient_checkpointing'],load_profile(PROFILE_8GPU70K_B8)['gradient_checkpointing'])


if __name__=='__main__':unittest.main()
