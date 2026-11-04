"""Lightweight real Torch checks; no full VLM/data/GPU claims."""
import ast,copy,json,os,random,tempfile,types,unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from action_head import CausalActionEncoder,ActionConditionedHead,objective
from contract import digest
from checkpoints import latest_checkpoint,compatible_checkpoint
import state
from data import RankIndices,WindowDataset

source=Path(__file__).with_name('model.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='ConditionedHead')
space=dict(nn=nn,torch=torch,F=F,checkpoint=checkpoint,CausalActionEncoder=CausalActionEncoder)
exec(compile(ast.Module(body=[node],type_ignores=[]),'model.py#actual-ConditionedHead','exec'),space)
Head=space['ConditionedHead']

class Base(nn.Module):
 def __init__(self):
  super().__init__();self.point_mlp=nn.Sequential(nn.Linear(3,4),nn.GELU(),nn.Linear(4,16))
  self.time=nn.Linear(16,16);self.fusion_in=nn.Linear(32,8);self.fusion_out=nn.Linear(8,3)
  self.config=types.SimpleNamespace(point_chunk_size=3)
 def encode_time(self,h,v):return self.time(h)

class Toy(nn.Module):
 def __init__(self):super().__init__();self.linear=nn.Linear(3,2);self.unused=nn.Parameter(torch.ones(5))
 def forward(self,x):return self.linear(x)

class Tests(unittest.TestCase):
 def setUp(self):torch.manual_seed(42);np.random.seed(42);random.seed(42)
 def test_actual_fusion_matches_independent_head(self):
  h=Head(Base(),'C3');ref=ActionConditionedHead(copy.deepcopy(h.base));ref.load_state_dict(h.state_dict());h.eval();ref.eval()
  with torch.no_grad():h.action_projection.weight.normal_();ref.load_state_dict(h.state_dict())
  x=torch.randn(2,50,16,requires_grad=True);p=torch.randn(2,7,3);v=torch.ones(2,50,dtype=torch.bool);act=torch.randn(2,50,12,requires_grad=True)
  h.condition=act;out=h(x,v,p);want=ref((x,v,p),act.detach())
  torch.testing.assert_close(out,want,rtol=0,atol=0);out.square().mean().backward()
  self.assertIsNone(act.grad);self.assertGreater(x.grad.norm().item(),0)
  self.assertGreater(h.action_encoder.input[0].weight.grad.norm().item(),0)
 def test_zero_projection_equal_null_and_matched_initially(self):
  a=Head(Base(),'C1');b=Head(copy.deepcopy(a.base),'C3');b.load_state_dict(a.state_dict())
  a.eval();b.eval();a.condition=b.condition=torch.randn(1,50,12)
  x=torch.randn(1,50,16);p=torch.randn(1,5,3);v=torch.ones(1,50,dtype=torch.bool)
  self.assertTrue(torch.equal(a(x,v,p),b(x,v,p)))
 def test_gru_causality(self):
  m=CausalActionEncoder();a=torch.randn(2,50,12);b=a.clone();b[:,25:]+=100
  self.assertTrue(torch.equal(m(a)[:,:25],m(b)[:,:25]))
 def test_objective_manual_formula_and_nan_mask(self):
  pred=torch.ones(1,2,50,3);gt=torch.zeros_like(pred);v=torch.ones(1,2,50,dtype=torch.bool)
  self.assertAlmostEqual(objective(pred,gt,v).item(),8.,places=6)
  gt[:,1]=float('nan');v[:,1]=False
  self.assertAlmostEqual(objective(pred,gt,v).item(),8.,places=6)
  self.assertEqual(objective(pred,gt,torch.zeros_like(v)).item(),0)
 def test_rank_partition_resume_cursor(self):
  for world in (1,2,4,8):
   seq=[list(RankIndices(3,5,r,world)) for r in range(world)]
   self.assertEqual(sorted(sum(seq,[])),list(range(3*64,5*64)))
   self.assertTrue(all(len(x)==128//world for x in seq))
  self.assertEqual(list(RankIndices(4,5,3,8)),list(range(4*64+24,4*64+32)))
 def test_source_frame_zero_masks_only_auxiliary(self):
  sample={'action':np.ones((50,12),np.float32),'point_delta':np.ones((1024,50,3),np.float32),
          'point_valid':np.ones((1024,50),bool),'point_meta':{'coordinate_frame':'window_start_robot_base','units':'meters','frame_index':0}}
  class Fake:
   def __getitem__(self,index):return copy.deepcopy(sample)
  ds=WindowDataset.__new__(WindowDataset);ds.base=Fake();ds.seed=42
  row=ds[8];self.assertFalse(row['point_valid'].any());self.assertTrue(np.array_equal(row['action'],sample['action']))
  sample['point_meta']['frame_index']=1;self.assertTrue(ds[9]['point_valid'].all())
 def test_full_state_restores_unused_adam_and_rng(self):
  with tempfile.TemporaryDirectory() as tmp:
   model=Toy();opt=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
   def step(m,o):
    o.zero_grad();loss=m(torch.randn(4,3)).square().mean();loss.backward();o.step()
   step(model,opt);step(model,opt);cp=state.save(tmp,model,opt,2,'same',0,1)
   step(model,opt);expected=state.tree_signature({'model':model.state_dict(),'optimizer':opt.state_dict()})
   restored=Toy();newopt=torch.optim.AdamW(restored.parameters(),lr=.001,foreach=False)
   self.assertEqual(state.load(cp,restored,newopt,'same',1,0),2);step(restored,newopt)
   self.assertEqual(expected,state.tree_signature({'model':restored.state_dict(),'optimizer':newopt.state_dict()}))
   self.assertNotIn(restored.unused,newopt.state)
   (Path(tmp)/'step_000003').mkdir()
   self.assertEqual(latest_checkpoint(tmp,'same',1)[0],cp)
   self.assertIsNone(compatible_checkpoint(cp,'different',1))
   self.assertIsNone(compatible_checkpoint(cp,'same',8))
 def test_worker_declares_original_lr_math(self):
  text=Path(__file__).with_name('worker.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='lr_factor')
  import math
  ns={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'lr_factor','exec'),ns)
  c={'warmup_steps':1000,'schedule_steps':100000,'min_learning_rate':2.5e-6,'learning_rate':2.5e-5}
  self.assertEqual(ns['lr_factor'](0,c),1/1001);self.assertEqual(ns['lr_factor'](1000,c),1.)

if __name__=='__main__':unittest.main(verbosity=2)
