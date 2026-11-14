"""FOCAL training with one shared context pass and block FlashAttention."""
import copy
import math
import functools
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma as gemma
from .reference import FocalPolicy as ReferenceFocalPolicy
from .attention import make_layout,focal_flash


class FocalPolicy(ReferenceFocalPolicy):
    def __init__(self,resources,stats=None,load_base=True):
        super().__init__(resources,stats,load_base)
        self.backend='flash'
        self.native.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def prefix(self,observation,texts=None):
        p=self.native
        images,masks,lang,lang_mask,state=p._preprocess_observation(observation,train=False)
        batch=images[0].shape[0]
        image_tokens=p.paligemma_with_expert.embed_image(torch.cat(images,0))
        patches=image_tokens.shape[1]
        image_tokens=image_tokens.view(len(images),batch,patches,-1).permute(1,0,2,3).reshape(batch,len(images)*patches,-1)
        image_valid=torch.stack(masks,1)[:,:,None].expand(batch,len(images),patches).reshape(batch,-1)
        language=p.paligemma_with_expert.embed_language_tokens(lang)*math.sqrt(image_tokens.shape[-1])
        context=torch.cat([image_tokens,language],1);valid=torch.cat([image_valid,lang_mask],1)
        queries=torch.cat([self.geometry_queries,self.motion_queries],0)[None].expand(batch,-1,-1)
        prefix=torch.cat([context,queries.to(context.dtype)],1)
        target=sem_valid=None
        if texts is not None:
            inputs,target,sem_valid=self.semantic_tokens(texts,context.device)
            semantic=p.paligemma_with_expert.embed_language_tokens(inputs)*math.sqrt(context.shape[-1])
            prefix=torch.cat([prefix,semantic],1)
        return prefix,valid,state,target,sem_valid

    @functools.lru_cache(maxsize=8192)
    def _encode(self,text):
        return tuple(self.tokenizer.encode(text,add_eos=True))

    def semantic_tokens(self,texts,device):
        tokens=[self._encode(t) for t in texts];length=max(map(len,tokens),default=0)
        target=np.full((len(tokens),length),-100,dtype=np.int64)
        inputs=np.zeros_like(target);valid=np.zeros_like(target,dtype=np.bool_)
        for i,t in enumerate(tokens):
            target[i,:len(t)]=t;inputs[i,:len(t)]=(self.tokenizer.bos_id(),)+t[:-1];valid[i,:len(t)]=True
        return tuple(torch.from_numpy(a).to(device,non_blocking=True) for a in (inputs,target,valid))

    def prepare_batch(self,samples):
        # Full-data training transforms/pins batches in DataLoader workers.
        # Raw samples remain supported for inference and numerical comparisons.
        if isinstance(samples,dict):
            from .adapter import tree_map
            batch=tree_map(lambda x:x.to(self.geometry_queries.device,non_blocking=True),samples['inputs'])
            actions=batch.pop('actions')
            return self.framework._model_module.Observation.from_dict(batch),actions
        obs,actions,_=self.framework._prepare_batch(samples,include_actions=True)
        return obs,actions

    def fused_hidden(self,prefix,semantic,suffix,cv,sv,cond):
        p=self.native.paligemma_with_expert
        c=cv.shape[1];plen=prefix.shape[1];slen=semantic.shape[1]
        count=cv.long().sum(1,keepdim=True)
        # Preserve the reference paths' positions; semantics do not shift action.
        main_pos=torch.cat([cv.long().cumsum(1)-1,count+torch.arange(16,device=cv.device)[None],
                            count+16+torch.arange(suffix.shape[1],device=cv.device)[None]],1)
        sem_pos=count+torch.arange(slen,device=cv.device)[None]
        rotary=p.paligemma.language_model.rotary_emb
        # Native joint attention constructs rotary frequencies in Q's dtype,
        # not the FP32 residual-stream dtype.
        rotary_dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else prefix.dtype
        rotary_stub=torch.empty((),device=prefix.device,dtype=rotary_dtype)
        cos,sin=rotary(rotary_stub,main_pos)
        # The standalone HF semantic path uses its FP32 embedding stream for
        # RoPE, then autocast rounds Q/K at the attention matmul boundary.
        sem_cos,sem_sin=rotary(semantic,sem_pos)
        blocks=make_layout(cv,sv,suffix.shape[1])
        dense_mask=None
        if self.backend=='eager':
            # Numerical reference for the optimized block layout.
            total=plen+slen+suffix.shape[1]
            dense_mask=torch.zeros(cv.shape[0],total,total,dtype=torch.bool,device=cv.device)
            dense_mask[:,:c,:c]=True
            dense_mask[:,c:c+8,:c+8]=True
            dense_mask[:,c+8:plen,:c]=True;dense_mask[:,c+8:plen,c+8:plen]=True
            dense_mask[:,plen:plen+slen,:c]=True
            dense_mask[:,plen:plen+slen,plen:plen+slen]=torch.ones(slen,slen,device=cv.device,dtype=torch.bool).tril()
            dense_mask[:,plen+slen:,:plen]=True;dense_mask[:,plen+slen:,plen+slen:]=True
            valid=torch.cat([cv,torch.ones(cv.shape[0],16,device=cv.device,dtype=torch.bool),sv,torch.ones(cv.shape[0],suffix.shape[1],device=cv.device,dtype=torch.bool)],1)
            dense_mask &= valid[:,None,:]&valid[:,:,None]
            dense_mask=self.native._prepare_attention_masks_4d(dense_mask)

        def layer_forward(index,x,z,a):
            layers=[p.paligemma.language_model.layers[index],p.paligemma.language_model.layers[index],p.gemma_expert.model.layers[index]]
            hidden=[x,z,a];conditions=[None,None,cond];qs=[];ks=[];vs=[];gates=[]
            for value,layer,condition in zip(hidden,layers,conditions):
                norm,gate=layer.input_layernorm(value,cond=condition);gates.append(gate)
                shape=(*norm.shape[:-1],-1,layer.self_attn.head_dim)
                qs.append(layer.self_attn.q_proj(norm).view(shape).transpose(1,2))
                ks.append(layer.self_attn.k_proj(norm).view(shape).transpose(1,2))
                vs.append(layer.self_attn.v_proj(norm).view(shape).transpose(1,2))
            mq,mk=gemma.apply_rotary_pos_emb(torch.cat([qs[0],qs[2]],2),torch.cat([ks[0],ks[2]],2),cos,sin,unsqueeze_dim=1)
            sq,sk=gemma.apply_rotary_pos_emb(qs[1],ks[1],sem_cos,sem_sin,unsqueeze_dim=1)
            sq=sq.to(mq.dtype);sk=sk.to(mk.dtype)
            q=torch.cat([mq[:,:,:plen],sq,mq[:,:,plen:]],2)
            k=torch.cat([mk[:,:,:plen],sk,mk[:,:,plen:]],2)
            v=torch.cat(vs,2)
            scale=layers[0].self_attn.scaling
            if self.backend=='flash':
                attention=focal_flash(q,k,v,blocks,scale)
            else:
                attention=gemma.eager_attention_forward(layers[0].self_attn,q,k,v,dense_mask,scale)[0].flatten(2)
            result=[];offset=0
            for value,layer,condition,gate in zip(hidden,layers,conditions,gates):
                # Autocast performs the necessary cast; no BF16->FP32->BF16 round trip.
                out=layer.self_attn.o_proj(attention[:,offset:offset+value.shape[1]])
                out=gemma._gated_residual(value,out,gate)
                normalized,mlp_gate=layer.post_attention_layernorm(out,cond=condition)
                result.append(gemma._gated_residual(out,layer.mlp(normalized),mlp_gate))
                offset+=value.shape[1]
            return tuple(result)

        values=(prefix,semantic,suffix)
        for index in range(len(p.paligemma.language_model.layers)):
            if self.training and torch.is_grad_enabled():
                values=checkpoint(layer_forward,index,*values,use_reentrant=False,preserve_rng_state=False)
            else:
                values=layer_forward(index,*values)
        return p.paligemma.language_model.norm(values[0])[0],p.paligemma.language_model.norm(values[1])[0],p.gemma_expert.model.norm(values[2],cond=cond)[0]

    def forward(self,samples,targets,noise=None,time=None,return_outputs=False):
        p=self.native;obs,actions=self.prepare_batch(samples);actions=actions.float()
        noise=torch.randn_like(actions) if noise is None else noise
        time=p.sample_time(actions.shape[0],actions.device) if time is None else time
        noisy=time[:,None,None]*noise+(1-time[:,None,None])*actions
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prefix,cv,state,_,_=self.prefix(obs,None)
            inputs,sem_target,sv=self.semantic_tokens(tuple(samples['subtasks']) if isinstance(samples,dict) else tuple(s['subtask'] for s in samples),prefix.device)
            semantic=p.paligemma_with_expert.embed_language_tokens(inputs)*math.sqrt(prefix.shape[-1])
            suffix,_,_,cond=p.embed_suffix(state,noisy,time)
            h,sh,a=self.fused_hidden(prefix,semantic,suffix,cv,sv,cond)
            velocity=p.action_out_proj(a[:,-p.config.action_horizon:].float()).float()
            geo=self.geometry_head(h[:,cv.shape[1]:cv.shape[1]+8])
            motion=self.motion_head(h[:,cv.shape[1]+8:cv.shape[1]+16])
            logits=p.paligemma_with_expert.paligemma.get_output_embeddings()(sh).float()
        token=F.cross_entropy(logits.flatten(0,1),sem_target.flatten(),ignore_index=-100,reduction='none').view_as(sem_target)
        losses={'action':F.mse_loss(velocity,noise-actions),'semantic':(token.sum(1)/(sem_target!=-100).sum(1).clamp_min(1)).mean()}
        for key,pred in [('geo',geo),('motion',motion)]:
            mean=getattr(self,key+'_mean');std=getattr(self,key+'_std')
            label=targets[key].to(pred.device,non_blocking=True).float();valid=targets[key+'_valid'].to(pred.device,non_blocking=True)
            x=(pred-mean)/std;y=(label-mean)/std
            values=(x-y).square().mean(-1) if key=='geo' else F.smooth_l1_loss(x,y,reduction='none',beta=1).mean(-1)
            losses[key]=(values*valid).sum()/valid.sum().clamp_min(1)
        losses['loss']=losses['action']+.05*losses['geo']+.05*losses['motion']+.01*losses['semantic']
        return (losses,{'velocity':velocity,'geo':geo,'motion':motion}) if return_outputs else losses
