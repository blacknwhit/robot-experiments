"""CPU/GPU-independent mathematical gates; real multi-GPU acceptance is separate."""
import importlib.util
import sys
import tempfile
import unittest
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
    def test_auxiliary_reaches_backbone_all_layers(self):
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
