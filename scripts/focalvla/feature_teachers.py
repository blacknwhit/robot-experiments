"""Exact required teacher branches, with batched inputs and bounded RoPE reuse."""
from collections import OrderedDict
import gc
import types
import sys
import numpy as np
import torch
from torch.nn import functional as F
from focalvla.teachers import Teachers,tensor,coverage,pool


def load_frozen_teachers(resources):
    # All used parameters are restored by the upstream checkpoint loaders.
    # Avoid filling gigabytes of parameters with random values first. Keep tensor
    # factories/buffers intact; only nn.init routines are suppressed in this scope.
    from unittest.mock import patch
    names=('uniform_','normal_','trunc_normal_','constant_','ones_','zeros_',
           'xavier_uniform_','xavier_normal_','kaiming_uniform_','kaiming_normal_','orthogonal_')
    def unchanged(tensor,*args,**kwargs):return tensor
    with patch.multiple(torch.nn.init,**{name:unchanged for name in names}):
        return Teachers(resources)


def cache_rotary_positions(module,limit=8,static_grid=False):
    """Reuse embeddings of the same immutable position tensor within a forward.

    Hold the position tensor to prevent allocator pointer reuse. The arithmetic
    and intermediate dtypes match the upstream RoPE, with no fused FMA changes.
    """
    for rope in module.modules():
        if not hasattr(rope,'_compute_frequency_components') or not hasattr(rope,'_rotate_features'):
            continue
        cache=OrderedDict();rope._focal_rotary_bank=cache
        def forward(self,tokens,positions,_cache=cache):
            dim=tokens.shape[-1]//2
            key=(None if static_grid else positions.data_ptr(),tuple(positions.shape),tuple(positions.stride()),dim,tokens.dtype)
            if key not in _cache:
                maximum=int(positions.max())+1
                co,si=self._compute_frequency_components(dim,maximum,tokens.device,tokens.dtype)
                values=(positions,F.embedding(positions[...,0],co)[:,None],F.embedding(positions[...,0],si)[:,None],
                        F.embedding(positions[...,1],co)[:,None],F.embedding(positions[...,1],si)[:,None])
                _cache[key]=values
                if len(_cache)>limit:_cache.popitem(last=False)
            elif not static_grid:_cache.move_to_end(key)
            _,cy,sy,cx,sx=_cache[key]
            y,x=tokens.chunk(2,dim=-1)
            return torch.cat((y*cy+self._rotate_features(y)*sy,x*cx+self._rotate_features(x)*sx),dim=-1)
        rope.forward=types.MethodType(forward,rope)


class FeatureTeachers:
    def __init__(self,resources=None,base=None,cache_rope=True,static_grid=False,prune_outputs=False,cast_linears=False):
        base=load_frozen_teachers(resources) if base is None else base
        base.handle.remove()
        self.geometry=base.geometry.aggregator
        self.backbone=base.motion.backbone.model.da3.backbone
        self.projection=base.motion.dot_fc2
        self.flow3d=base.motion.flow_aggregator3d
        self.dim=base.motion.dim_feature
        self.patch_start=base.motion.patch_start_idx
        self.image_mean=base.motion.image_mean
        self.image_std=base.motion.image_std
        # Selection contains small batched GEMMs whose kernel can change with B.
        # Its argmin is discrete: preserve the original B=1 calculation per clip.
        selector_module=sys.modules[type(self.backbone.pretrained).__module__]
        selector=selector_module.select_reference_view
        selector=getattr(selector,'_focal_native',selector)
        def select_per_clip(x,strategy='saddle_balanced'):
            return torch.cat([selector(x[i:i+1],strategy) for i in range(len(x))])
        select_per_clip._focal_native=selector
        selector_module.select_reference_view=select_per_clip
        for module in (self.geometry,self.backbone,self.flow3d):
            if hasattr(module,'_profile_original_forward'):
                module.forward=module._profile_original_forward
                del module._profile_original_forward
        if prune_outputs:
            self.geometry.cached_layer_indices=(self.geometry.depth-1,)
            self.backbone.out_layers=[max(self.backbone.out_layers)]
            # FOCAL consumes only the normalized global half of DA3's final
            # concatenated feature; local concatenation/intermediate exports are unused.
            self.backbone.pretrained.cat_token=False
        if cast_linears:
            for module,dtype in [(self.geometry,torch.bfloat16),(self.backbone,torch.float16),
                                 (self.projection,torch.float16),(self.flow3d,torch.float16)]:
                for layer in module.modules():
                    if isinstance(layer,(torch.nn.Linear,torch.nn.Conv2d)):
                        layer.to(dtype=dtype)
        if cache_rope:
            for module in (self.geometry,self.backbone,self.flow3d):cache_rotary_positions(module,static_grid=static_grid)
        # Keep only the branches needed by the FOCAL targets. The strict upstream
        # checkpoint loader above validates all weights before unused heads leave.
        del base
        gc.collect();torch.cuda.empty_cache()

    @torch.inference_mode()
    def __call__(self,samples):
        count=len(samples)
        batch=len(samples)
        images=np.stack([s[0] for s in samples]);clips=np.stack([s[1] for s in samples]);masks=np.stack([s[2] for s in samples])
        image=tensor(images.reshape(-1,*images.shape[2:]),518).reshape(batch,3,3,518,518)
        motion_image=(tensor(clips.reshape(-1,*clips.shape[2:]),630)-self.image_mean)/self.image_std
        motion_image=F.interpolate(motion_image,(630,630),mode='bilinear',align_corners=False,antialias=True)
        masks=torch.as_tensor(masks,device='cuda',dtype=torch.float32)
        geo,motion,geo_valid,motion_valid=self._forward(image,motion_image,masks)
        assert torch.isfinite(geo).all() and torch.isfinite(motion).all()
        return {'geo':geo[:count].cpu().numpy(),'motion':motion[:count].cpu().numpy(),
                'geo_valid':geo_valid[:count].cpu().numpy(),'motion_valid':motion_valid[:count].cpu().numpy()}

    def _forward(self,image,motion_image,masks):
        batch=image.shape[0]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features,patch=self.geometry(image)
        f=features[-1][:,:,patch:].reshape(batch*3,1369,2048)
        value,valid=pool(f,coverage(masks.reshape(-1,*masks.shape[-2:]),37))
        value=value.reshape(batch,3,2048);valid=valid.reshape(batch,3)
        geo=(value*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        geo_valid=valid.any(1)
        del features,f,value,image
        image=motion_image
        with torch.autocast('cuda',dtype=torch.float16):
            features,_=self.backbone(image.reshape(batch,11,3,630,630),cam_token=None,export_feat_layers=[],ref_view_strategy='saddle_balanced')
            hidden=features[-1][0][...,-self.dim:]
            flow=self.flow3d(self.projection(hidden.detach()),image,patch_start_idx=self.patch_start)[-1]
        assert flow.shape==(batch,11,2025,256)
        motion,motion_valid=pool(flow[:,0],coverage(masks[:,0],45))
        return geo,motion,geo_valid,motion_valid
