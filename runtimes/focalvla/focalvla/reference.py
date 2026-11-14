"""Reference visibility and query-conditioned inference."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .adapter import NativeAdapter

class PredictionHead(nn.Module):
 def __init__(self,width,out):
  super().__init__();self.norm=nn.LayerNorm(width);self.proj=nn.Linear(width,out)
 def forward(self,x):return self.proj(self.norm(x).mean(1)).float()

def attention_layout(context_valid, semantic_valid=None, action_len=0):
 b,c=context_valid.shape;device=context_valid.device;q=16;s=0 if semantic_valid is None else semantic_valid.shape[1]
 p=c+q+s;total=p+action_len
 valid=torch.cat([context_valid,torch.ones(b,q,device=device,dtype=torch.bool),semantic_valid if s else context_valid[:,:0],torch.ones(b,action_len,device=device,dtype=torch.bool)],1)
 mask=torch.zeros((b,total,total),dtype=torch.bool,device=device)
 mask[:,:c,:c]=True
 mask[:,c:c+8,:c+8]=True
 mask[:,c+8:c+16,:c]=True;mask[:,c+8:c+16,c+8:c+16]=True
 if s:
  mask[:,c+16:p,:c]=True
  mask[:,c+16:p,c+16:p]=torch.ones(s,s,dtype=torch.bool,device=device).tril()
 if action_len:
  mask[:,p:,:c+16]=True;mask[:,p:,p:]=True
 mask &= valid[:,None,:] & valid[:,:,None]
 count=context_valid.long().sum(1,keepdim=True)
 pos_c=context_valid.long().cumsum(1)-1
 pos_q=count+torch.arange(q,device=device)[None]
 pos_s=count+q+torch.arange(s,device=device)[None]
 # Semantic labels are training-only: they never shift action RoPE positions.
 pos_a=count+q+torch.arange(action_len,device=device)[None]
 return mask,torch.cat([pos_c,pos_q,pos_s,pos_a],1),valid

class FocalPolicy(nn.Module):
 def __init__(self,resources,stats=None,load_base=True):
  super().__init__()
  self.framework=NativeAdapter(resources,load_base=load_base);p=self.framework.policy_model
  p.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
  p.paligemma_with_expert.gemma_expert.model.config._attn_implementation='eager'
  self.framework.float().requires_grad_(True)
  p.gradient_checkpointing_enable();p.paligemma_with_expert.gradient_checkpointing=True
  p.paligemma_with_expert.paligemma.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  self.geometry_queries=nn.Parameter(torch.randn(8,2048)*.02)
  self.motion_queries=nn.Parameter(torch.randn(8,2048)*.02)
  self.geometry_head=PredictionHead(2048,2048);self.motion_head=PredictionHead(2048,256)
  for key,dim in [('geo',2048),('motion',256)]:
   self.register_buffer(key+'_mean',torch.tensor(stats[key]['mean'],dtype=torch.float32) if stats else torch.zeros(dim))
   self.register_buffer(key+'_std',torch.tensor(stats[key]['std'],dtype=torch.float32) if stats else torch.ones(dim))
  import sentencepiece as spm
  self.tokenizer=spm.SentencePieceProcessor(model_file=resources['tokenizer'])

 @property
 def native(self):return self.framework.policy_model

 def semantic_tokens(self,texts,device):
  labels=[self.tokenizer.encode(t,add_eos=True) for t in texts]
  n=max(map(len,labels));target=torch.full((len(texts),n),-100,device=device,dtype=torch.long)
  inputs=torch.zeros_like(target);valid=torch.zeros_like(target,dtype=torch.bool)
  for i,t in enumerate(labels):
   target[i,:len(t)]=torch.tensor(t,device=device);inputs[i,:len(t)]=torch.tensor([self.tokenizer.bos_id()]+t[:-1],device=device);valid[i,:len(t)]=True
  return inputs,target,valid

 def prefix(self,observation,texts=None):
  p=self.native
  images,im_masks,lang,lang_mask,state=p._preprocess_observation(observation,train=False)
  c,valid,_=p.embed_prefix(images,im_masks,lang,lang_mask)
  queries=torch.cat([self.geometry_queries,self.motion_queries],0)[None].expand(c.shape[0],-1,-1)
  prefix=torch.cat([c,queries.to(c.dtype)],1);target=None;sem_valid=None
  if texts is not None:
   inputs,target,sem_valid=self.semantic_tokens(texts,c.device)
   emb=p.paligemma_with_expert.embed_language_tokens(inputs)*math.sqrt(c.shape[-1])
   prefix=torch.cat([prefix,emb],1)
  return prefix,valid,state,target,sem_valid

 def velocity(self,observation,noisy,time,texts=None,disconnect=False,return_context=False):
  # The paper's semantic branch shares weights, but is evaluated separately.
  # This implements the same block mask without GT text changing BF16 kernel
  # dimensions in the action path; context embeddings still receive both losses.
  p=self.native;prefix,cv,state,_,_=self.prefix(observation,None)
  sem_target=None;sv=None
  suffix,_,_,cond=p.embed_suffix(state,noisy,time)
  mask,positions,_=attention_layout(cv,sv,suffix.shape[1]);c=cv.shape[1];prefix_len=prefix.shape[1]
  if disconnect:mask[:,prefix_len:,c:c+16]=False
  (hidden,action),_=p.paligemma_with_expert.forward(attention_mask=p._prepare_attention_masks_4d(mask),position_ids=positions,inputs_embeds=[prefix,suffix],past_key_values=None,use_cache=False,adarms_cond=[None,cond])
  velocity=p.action_out_proj(action[:,-p.config.action_horizon:].float()).float()
  result=(velocity,hidden,sem_target,c)
  return (*result,prefix[:,:c],cv) if return_context else result

 def semantic_logits(self,context,cv,texts):
  inputs,target,sv=self.semantic_tokens(texts,context.device)
  embedded=self.native.paligemma_with_expert.embed_language_tokens(inputs)*math.sqrt(context.shape[-1])
  b,c=cv.shape;s=sv.shape[1];valid=torch.cat([cv,sv],1)
  mask=torch.zeros(b,c+s,c+s,device=context.device,dtype=torch.bool)
  mask[:,:c,:c]=True;mask[:,c:,:c]=True;mask[:,c:,c:]=torch.ones(s,s,device=context.device,dtype=torch.bool).tril()
  mask &= valid[:,None,:] & valid[:,:,None]
  pos=torch.cat([cv.long().cumsum(1)-1,cv.sum(1)[:,None]+torch.arange(s,device=context.device)[None]],1)
  (hidden,_),_=self.native.paligemma_with_expert.forward(attention_mask=self.native._prepare_attention_masks_4d(mask),position_ids=pos,inputs_embeds=[torch.cat([context,embedded],1),None],past_key_values=None,use_cache=False,adarms_cond=[None,None])
  head=self.native.paligemma_with_expert.paligemma.get_output_embeddings()
  return head(hidden[:,c:]).float(),target

 def forward(self,samples,targets,noise=None,time=None):
  p=self.native;obs,actions,_=self.framework._prepare_batch(samples,include_actions=True)
  actions=actions.float();noise=torch.randn_like(actions) if noise is None else noise
  time=p.sample_time(len(samples),actions.device) if time is None else time
  noisy=time[:,None,None]*noise+(1-time[:,None,None])*actions
  with torch.autocast('cuda',dtype=torch.bfloat16):
   vel,hidden,_,c,context,cv=self.velocity(obs,noisy,time,return_context=True)
   geo=self.geometry_head(hidden[:,c:c+8]);motion=self.motion_head(hidden[:,c+8:c+16])
   logits,sem_target=self.semantic_logits(context,cv,[s['subtask'] for s in samples])
  action=F.mse_loss(vel,noise-actions)
  token_loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),sem_target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(sem_target)
  semantic=(token_loss.sum(1)/(sem_target!=-100).sum(1).clamp_min(1)).mean()
  losses={}
  for key,pred in [('geo',geo),('motion',motion)]:
   label=targets[key].to(pred.device).float();valid=targets[key+'_valid'].to(pred.device)
   mean=getattr(self,key+'_mean');std=getattr(self,key+'_std')
   a=(pred-mean)/std;b=(label-mean)/std
   values=(a-b).square().mean(-1) if key=='geo' else F.smooth_l1_loss(a,b,reduction='none',beta=1).mean(-1)
   losses[key]=(values*valid).sum()/valid.sum().clamp_min(1)
  return {'loss':action+.05*losses['geo']+.05*losses['motion']+.01*semantic,'action':action,'semantic':semantic,**losses}

 @torch.inference_mode()
 def predict_action(self,samples,noise=None,num_steps=10,disconnect=False):
  p=self.native;obs,_,transformed=self.framework._prepare_batch(samples,include_actions=False)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   prefix,cv,state,_,_=self.prefix(obs)
   mask,pos,pv=attention_layout(cv)
   _,cache=p.paligemma_with_expert.forward(attention_mask=p._prepare_attention_masks_4d(mask),position_ids=pos,inputs_embeds=[prefix,None],past_key_values=None,use_cache=True,adarms_cond=[None,None])
   if disconnect:pv[:,cv.shape[1]:]=False
   x=torch.randn((len(samples),50,32),device=prefix.device,dtype=torch.float32) if noise is None else noise.clone()
   for k in range(num_steps):
    t=torch.full((len(samples),),1-k/num_steps,device=prefix.device)
    if not disconnect:v=p.denoise_step(state,pv,cache,x,t)
    else:
     # Keep positions fixed when ablating query conditioning.
     suffix,sp,sa,cond=p.embed_suffix(state,x,t)
     b,a=sp.shape;vis=torch.cat([pv[:,None,:].expand(b,a,-1),torch.ones(b,a,a,device=x.device,dtype=torch.bool)],-1)
     positions=cv.sum(1)[:,None]+16+torch.arange(a,device=x.device)[None]
     (_,out),_=p.paligemma_with_expert.forward(attention_mask=p._prepare_attention_masks_4d(vis),position_ids=positions,inputs_embeds=[None,suffix],past_key_values=cache,use_cache=False,adarms_cond=[None,cond])
     v=p.action_out_proj(out[:,-50:].float())
    x=x-v.float()/num_steps
  output=[]
  for i in range(len(samples)):
   out=self.framework._output_transform({'state':transformed['state'][i].cpu().numpy(),'actions':x[i].cpu().numpy()})
   output.append(np.asarray(out['actions'],dtype=np.float32))
  return np.stack(output)
