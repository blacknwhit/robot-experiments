"""Auxiliary-only action conditioning; native policy input and output preserved."""
import pyarrow
import copy,json,types,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from contract import ROOT,activate
from action_head import CausalActionEncoder,objective
from robot_eval import runtime as strict

class ConditionedHead(nn.Module):
 def __init__(self,base,arm):
  super().__init__();self.base=base;self.arm=arm;self.condition=None
  self.action_encoder=CausalActionEncoder();self.action_projection=nn.Linear(64,base.fusion_in.out_features)
  nn.init.zeros_(self.action_projection.weight);nn.init.zeros_(self.action_projection.bias)
 def forward(self,hidden,valid,points,point_valid=None):
  if self.arm=='C0':return hidden.new_zeros((hidden.shape[0],points.shape[1],50,3),dtype=torch.float32)
  if point_valid is not None:points=torch.where(point_valid[...,None],points,0)
  z=self.base.encode_time(hidden,valid);width=z.shape[-1];p=self.base.point_mlp(points.to(self.base.point_mlp[0].weight.dtype))
  pp=F.linear(p,self.base.fusion_in.weight[:,:width]);zp=F.linear(z,self.base.fusion_in.weight[:,width:],self.base.fusion_in.bias)
  actions=self.condition.detach()
  if self.arm=='C1':actions=torch.zeros_like(actions)
  assert actions.shape==(hidden.shape[0],50,12)
  ap=self.action_projection(self.action_encoder(actions));tp=zp+ap
  def fuse(x,t):return self.base.fusion_out(F.gelu(x[:,:,None]+t[:,None]))
  chunks=[]
  for k in range(0,points.shape[1],self.base.config.point_chunk_size):
   part=pp[:,k:k+self.base.config.point_chunk_size]
   chunks.append(checkpoint(fuse,part,tp,use_reentrant=False,preserve_rng_state=False) if self.training and torch.is_grad_enabled() else fuse(part,tp))
  return torch.cat(chunks,1)

class JointPolicy(nn.Module):
 def __init__(self,arm,plan):
  super().__init__();self.arm=arm;self.plan=plan
  from starVLA.training.train_starvla import build_framework,apply_config_compat
  cfg=apply_config_compat(OmegaConf.create(copy.deepcopy(plan['framework_config'])))
  cfg.output_dir=str(Path(plan['run'])/'unused');cfg.run_root_dir=str(Path(plan['run'])/'unused')
  cfg.framework.pi05.init_checkpoint=plan['framework_config']['framework']['pi05']['init_checkpoint'];cfg.framework.pi05.deterministic_images=False
  cfg.framework.pi05.point_track.auxiliary_warmup_steps=0
  torch.manual_seed(plan['seed']);np.random.seed(plan['seed']);random.seed(plan['seed'])
  f=build_framework(cfg);p=f.policy_model;f.run_output_dir=None
  self.native_dtypes={n:v.dtype for n,v in f.state_dict().items()}
  original=p.point_track_head;p.point_track_head=ConditionedHead(original,arm)
  for v in f.parameters():v.requires_grad_(True);v.data=v.data.float()
  if arm=='C0':
   for v in p.point_track_head.parameters():v.requires_grad_(False)
  p.gradient_checkpointing_enable();p.paligemma_with_expert.gradient_checkpointing=True
  original_prepare=f._prepare_batch
  def prepare(*args,**kwargs):
   result=original_prepare(*args,**kwargs)
   if kwargs.get('include_actions',False):
    actions=result[1];assert actions.shape[-2:]==(50,32)
    if torch.count_nonzero(actions[...,12:]):raise RuntimeError('Nonzero padded action dimensions')
    p.point_track_head.condition=actions[...,:12].detach()
   return result
  f._prepare_batch=prepare;self.framework=f;self.captured={}
  self.hook=p.point_track_head.register_forward_hook(lambda m,inp,out:self.captured.update(pred=out,prefix=inp[0]))
  strict.configure()
 def forward(self,samples,noise_seed,augment=True):
  p=self.framework.policy_model;device=next(p.parameters()).device
  # Each call has a persisted seed. The same microbatch layout is used across arms.
  torch.manual_seed(noise_seed);np.random.seed(noise_seed%(2**32));random.seed(noise_seed)
  noise=p.sample_noise((len(samples),50,32),device);t=p.sample_time(len(samples),device)
  original=p._preprocess_observation
  if not augment:p._preprocess_observation=lambda obs,train=True:original(obs,train=False)
  try:
   with torch.autocast('cuda',dtype=torch.bfloat16):
    output=self.framework(copy.deepcopy(samples),noise=noise,time=t)
    pred=self.captured['pred']
    if self.arm=='C0':aux=output['action_loss']*0
    else:
     target=torch.as_tensor(np.stack([s['point_delta'] for s in samples]),device=device)
     valid=torch.as_tensor(np.stack([s['point_valid'] for s in samples]),device=device)
     aux=objective(pred,target,valid)
  finally:p._preprocess_observation=original
  return {'action':output['action_loss'],'aux':aux}
 def parameters_by_group(self):
  auxiliary=[];main=[]
  for name,p in self.framework.named_parameters():
   if p.requires_grad:(auxiliary if 'point_track_head.' in name else main).append(p)
  return main,auxiliary
 def export_policy_state(self):
  return {n:v.detach().cpu().to(self.native_dtypes[n]).contiguous().clone() for n,v in self.framework.state_dict().items() if 'point_track_head.' not in n}
 def close_capture(self):self.captured.clear()
