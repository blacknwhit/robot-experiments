"""Native mixed-dtype save/restore and world-coordinate regression checks."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import Accelerator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import activate
activate('pri4r_paper',overlay=False)
from configure import render
from train import launch_configuration,training_environment
from pri4r_native_state import write_metadata,validate_native,audit_restored,discard_auxiliary,restore_rng_strict,accumulation_plugin
from starVLA.pri4r.reader import point_window
from openpi.models_pytorch.point_track import PointTrackConfig


class Progress:
    def __init__(self):self.value={'completed_steps':1,'consumed_batches':32}
    def state_dict(self):return self.value
    def load_state_dict(self,value):self.value=value


class MixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bf=torch.nn.Linear(3,4).to(torch.bfloat16)
        self.fp=torch.nn.Linear(4,2)
    def forward(self,x):return self.fp(self.bf(x.to(torch.bfloat16)).float())


class NativeTests(unittest.TestCase):
    def test_world_coordinates_do_not_follow_base_pose(self):
        tracks=np.array([[[4.,6.,1.]],[[4.1,6.2,1.3]],[[4.2,6.5,1.7]]])
        poses=np.repeat(np.eye(4)[None],3,axis=0)
        poses[:,:3,3]=[3.,4.,0.]
        poses[:,:2,:2]=[[0.,-1.],[1.,0.]]
        out=point_window(tracks,poses,np.array([0.,.05,.1]),0,horizon=3,coordinate_frame='world')
        np.testing.assert_allclose(out['point_current'],tracks[0])
        np.testing.assert_allclose(out['point_delta'][0,:2],np.diff(tracks[:,0],axis=0),rtol=1e-6)
        self.assertFalse(out['point_valid'][0,2])
        self.assertEqual(out['point_meta']['coordinate_frame'],'world')
        local=point_window(tracks,poses,np.array([0.,.05,.1]),0,horizon=3)
        self.assertFalse(np.array_equal(out['point_current'],local['point_current']))

    def test_coordinate_change_invalidates_head_signature(self):
        a=PointTrackConfig(version='pri4r_embedding_v1',coordinate_frame='world')
        b=PointTrackConfig(version='pri4r_embedding_v1',coordinate_frame='window_start_robot_base')
        self.assertFalse(torch.equal(a.signature(2048),b.signature(2048)))

    def test_launcher_preserves_selective_precision_and_rejects_offload(self):
        c=render('pri4r_paper','home','native',1,2,smoke=True)
        cmd,config,ds=launch_configuration(c,Path('recipe.yaml'),Path('control'))
        self.assertIn('torch.distributed.run',cmd);self.assertIsNone(ds)
        self.assertEqual(config['mixed_precision'],'no')
        env=training_environment(c)
        self.assertEqual(env['ACCELERATE_USE_DEEPSPEED'],'false')
        c['trainer']['optimizer_offload_cpu']=True
        with self.assertRaises(ValueError):launch_configuration(c,Path('a'),Path('b'))

    def test_mixed_dtype_real_accelerate_roundtrip_and_corruption(self):
        accelerator=Accelerator(cpu=True,mixed_precision='no')
        model=MixedModel()
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        model,optimizer=accelerator.prepare(model,optimizer)
        model(torch.randn(3,3)).square().mean().backward();optimizer.step();optimizer.zero_grad();scheduler.step()
        progress=Progress();accelerator.register_for_checkpointing(scheduler,progress)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint-1';accelerator.save_state(path)
            # CPU fixture has no CUDA RNG; represent the actual empty device list.
            rng=torch.load(path/'random_states_0.pkl',weights_only=False)
            rng.setdefault('torch_cuda_manual_seed',[]);torch.save(rng,path/'random_states_0.pkl')
            fake=SimpleNamespace(model=model,optimizer=optimizer,accelerator=accelerator,total_batch_size=128,
                completed_steps=1,resume_from_checkpoint=str(path),config=SimpleNamespace(output_dir=tmp))
            # This fixture emulates the production accumulation clock, not its workload.
            original_acc=accelerator.gradient_accumulation_steps
            accelerator.gradient_accumulation_steps=32
            write_metadata(fake,path)
            c=render('pri4r_paper','home','roundtrip',1,2,smoke=True)
            validate_native(path,c,1)
            for p in model.parameters():p.data.zero_()
            for state in optimizer.state.values():state['exp_avg'].zero_()
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]):
                accelerator.load_state(path)
            random.seed(100);np.random.seed(100);torch.manual_seed(100)
            restore_rng_strict(path,0)
            audit_restored(fake)
            proof=json.loads((Path(tmp)/'native_restore_rank0.json').read_text())
            self.assertTrue(proof['model_adam_rng_bitwise_equal'])
            self.assertFalse(proof['has_fp32_master_copy'])
            saved=torch.load(path/'optimizer.bin',weights_only=False)
            key=next(iter(saved['state']));del saved['state'][key];torch.save(saved,path/'optimizer.bin')
            with self.assertRaisesRegex(ValueError,'Missing native Adam'):validate_native(path,c,1)
            accelerator.gradient_accumulation_steps=original_acc

    def test_prefetched_epoch_boundary_does_not_step_early(self):
        accelerator=Accelerator(cpu=True,mixed_precision='no',gradient_accumulation_plugin=accumulation_plugin(4))
        model=torch.nn.Linear(1,1)
        loader=accelerator.prepare(torch.utils.data.DataLoader(torch.arange(4),batch_size=1))
        iterator=iter(loader);pending=[next(iterator) for _ in range(4)]
        self.assertTrue(accelerator.gradient_state.end_of_dataloader)
        boundaries=[]
        for _ in pending:
            with accelerator.accumulate(model):boundaries.append(accelerator.sync_gradients)
        self.assertEqual(boundaries,[False,False,False,True])
        with self.assertRaises(StopIteration):next(iterator)

    def test_discard_keeps_action_modules(self):
        policy=torch.nn.Module();policy.action=torch.nn.Linear(2,3)
        policy.point_track_head=torch.nn.Linear(3,4)
        policy.foresight=SimpleNamespace(enabled=False)
        policy.point_track=SimpleNamespace(version='pri4r_embedding_v1')
        before=policy.action.weight.detach().clone()
        self.assertEqual(discard_auxiliary(policy),16)
        self.assertFalse(hasattr(policy,'point_track_head'))
        torch.testing.assert_close(policy.action.weight,before,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
