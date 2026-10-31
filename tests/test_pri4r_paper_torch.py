"""CPU/GPU-independent mathematical gates; real multi-GPU acceptance is separate."""
import importlib.util
import sys
import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import activate
activate('pri4r_paper')
from openpi.models_pytorch.point_track import PointTrackConfig,point_track_loss_terms
from openpi.models_pytorch.pri4r_embedding import EmbeddingPointTrackHead
path=ROOT/'runtimes/pri4r_paper/starVLA/training/trainer_utils/checkpoint_state.py'
spec=importlib.util.spec_from_file_location('ema_under_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class MathTests(unittest.TestCase):
    def test_all_invalid_aux_keeps_action_step_ema_and_audit(self):
        from engineering_audit import native_audited_step
        torch.manual_seed(53)
        policy=torch.nn.Module()
        policy.shared=torch.nn.Linear(3,3,bias=False)
        policy.point_track_head=torch.nn.Linear(3,1,bias=False)
        opt=torch.optim.AdamW(policy.parameters(),lr=.01,weight_decay=1e-10)
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        ema=m.ExponentialMovingAverage(.999);ema.initialize(policy)
        with tempfile.TemporaryDirectory() as tmp:
            acc=SimpleNamespace(unwrap_model=lambda model:model,sync_gradients=True,
                gradient_accumulation_steps=1,process_index=0,num_processes=1,is_main_process=True)
            trainer=SimpleNamespace(model=SimpleNamespace(policy_model=policy),accelerator=acc,
                total_batch_size=128,completed_steps=0,config=SimpleNamespace(output_dir=tmp))
            def original(t,batch,_):
                x=torch.ones(2,3)
                shared=policy.shared(x)
                action=shared.square().mean()
                pred=policy.point_track_head(shared).view(2,1,1,1).expand(-1,-1,-1,3)
                valid=torch.full((2,1,1),bool(batch[0]['point_valid'].any()),dtype=torch.bool)
                total,count,_=point_track_loss_terms(pred,torch.zeros_like(pred),valid)
                aux=total/count.clamp_min(1)
                t._point_global_count=count
                expected=torch.autograd.grad(action,policy.shared.weight,retain_graph=True)[0]
                (action+aux).backward()
                if valid.any():
                    self.assertFalse(torch.equal(policy.shared.weight.grad,expected))
                else:
                    torch.testing.assert_close(policy.shared.weight.grad,expected,rtol=0,atol=0)
                opt.step();ema.update(policy);schedule.step();opt.zero_grad()
                return {'point_track_loss':aux.item()}
            names=['shared.weight','point_track_head.weight']
            with patch('engineering_audit.audited_parameter_names',return_value=names):
                for step,valid in enumerate((True,False)):
                    trainer.completed_steps=step
                    result=native_audited_step(original,trainer,[{'point_valid':torch.tensor([[valid]])}])
                    if not valid:self.assertEqual(result['point_track_loss'],0.)
            rows=[json.loads(x) for x in (Path(tmp)/'native_update_rank0.jsonl').read_text().splitlines()]
            last=rows[-1]['parameters']['point_track_head.weight']
            self.assertEqual(last['gradient_squared_norm'],0.)
            self.assertFalse(last['global_update_required'])
            self.assertGreater(last['update_max'],0.)  # Existing Adam momentum may still move the head.
            self.assertEqual(schedule.last_epoch,2)
            self.assertEqual(ema.num_updates,2)

    def test_auxiliary_reaches_prefix_hidden_and_head_layers(self):
        cfg=PointTrackConfig(enabled=True,version='pri4r_embedding_v1',horizon=3,num_points=8,
            hidden_dim=16,num_heads=4,embedding_layers=2,embedding_ff_multiplier=2,
            point_hidden_dim=8,fusion_hidden_dim=8,point_chunk_size=4)
        head=EmbeddingPointTrackHead(16,cfg)
        hidden=torch.randn(2,6,16,requires_grad=True);points=torch.randn(2,8,3)
        valid=torch.ones(2,6,dtype=torch.bool)
        pred=head(hidden,valid,points)
        target=torch.randn_like(pred);mask=torch.ones(pred.shape[:-1],dtype=torch.bool)
        s,n,_=point_track_loss_terms(pred,target,mask);(s/n).backward()
        self.assertTrue(torch.isfinite(hidden.grad).all());self.assertGreater(hidden.grad.abs().sum(),0)
        for name,p in head.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum(),0,name)

    def test_ema_formula_and_resume_exact(self):
        model=torch.nn.Linear(3,2);ema=m.ExponentialMovingAverage(.999);ema.initialize(model)
        before={k:v.clone() for k,v in ema.shadow_parameters.items()}
        with torch.no_grad():
            for p in model.parameters():p.add_(1)
        ema.update(model)
        for name,p in model.named_parameters():
            expected=before[name].mul(.999).add(p.float(),alpha=.001)
            torch.testing.assert_close(ema.shadow_parameters[name],expected,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ema.safetensors';ema.save_safetensors(path,model.state_dict())
            resumed=m.ExponentialMovingAverage(.999);resumed.initialize(model);resumed.load_safetensors(path)
            self.assertEqual(resumed.num_updates,1)
            for k in ema.shadow_parameters:torch.testing.assert_close(resumed.shadow_parameters[k],ema.shadow_parameters[k],rtol=0,atol=0)

    def test_mask_and_loss_match_independent_formula(self):
        p=torch.randn(2,8,3,3,requires_grad=True);t=torch.randn_like(p);v=torch.rand(2,8,3)>.4
        t[~v]=float('nan')
        total,count,_=point_track_loss_terms(p,t,v)
        torch.testing.assert_close(total/count,(p[v]-t[v]).abs().mean())
        (total/count).backward();self.assertTrue((p.grad[~v]==0).all())


if __name__=='__main__':unittest.main()
