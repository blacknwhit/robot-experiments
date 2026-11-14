"""Frozen full-frame VGGT / Track4World teachers and current-entity pooling."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def tensor(images, size):
    value = torch.as_tensor(np.asarray(images), device='cuda', dtype=torch.float32).permute(0,3,1,2)/255
    return F.interpolate(value, size=(size,size), mode='bilinear', align_corners=False, antialias=True)


def coverage(masks, size):
    value = F.interpolate(torch.as_tensor(masks, device='cuda', dtype=torch.float32)[:,None],
                          size=(size*14,size*14), mode='nearest')
    return F.avg_pool2d(value,14,14)[:,0].flatten(1)


def pool(features, weights):
    denominator = weights.sum(-1); valid = denominator > 0
    values = (features.float()*weights[...,None]).sum(-2)/torch.where(valid,denominator,torch.ones_like(denominator))[...,None]
    return values, valid


class Captured(Exception):
    pass


class Teachers:
    def __init__(self, resources):
        sys.path[:0] = [resources[k] for k in ('vggt_source','track4world_source','utils3d_source_parent')]
        from vggt.models.vggt import VGGT
        from track4world.nets.model import Track4World, DepthAnything3
        self.geometry = VGGT().eval()
        self.geometry.load_state_dict(torch.load(resources['vggt_weights'], map_location='cpu', weights_only=True), strict=True)
        self.geometry.requires_grad_(False).cuda()
        original = DepthAnything3.from_pretrained
        try:
            DepthAnything3.from_pretrained = classmethod(lambda cls,*a,**kw: original(resources['da3_weights'], local_files_only=True))
            cfg = json.loads((Path(resources['track4world_source'])/'track4world/config/eval/v1.json').read_text())
            self.motion = Track4World(**cfg['model'],seqlen=16,use_3d=True,use_model='depthanythingv3').eval()
        finally:
            DepthAnything3.from_pretrained = original
        missing, unexpected = self.motion.load_pretrained_with_remap(torch.load(resources['track4world_weights'], map_location='cpu', weights_only=True))
        assert not [k for k in missing if not k.startswith('backbone.model.da3_metric.')], missing
        assert not unexpected, unexpected
        self.motion.requires_grad_(False).cuda()
        self.captured = None
        self.handle = self.motion.flow_aggregator3d.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.captured = output[-1].detach()
        raise Captured()

    @torch.inference_mode()
    def __call__(self, images, clip, masks):
        # Teachers see full frames; masks are applied after token extraction.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features, patch = self.geometry.aggregator(tensor(images,518)[None])
        features = features[-1][0,:,patch:]
        assert features.shape == (3,1369,2048), features.shape
        values, valid = pool(features, coverage(masks,37))
        geo = values[valid].mean(0) if valid.any() else torch.zeros(2048,device='cuda')
        geo_valid = bool(valid.any()); geo = geo.cpu().numpy()
        del features, values
        inputs = (tensor(clip,630)-self.motion.image_mean)/self.motion.image_std
        assert inputs.shape[0] == 11
        try:
            self.motion.forward_point(inputs,current_batch_size=1,for_flow=True)
        except Captured:
            pass
        features, self.captured = self.captured, None
        assert features is not None and features.shape == (1,11,2025,256)
        values, valid = pool(features[:,0], coverage(masks[:1],45))
        motion = values[0].cpu().numpy()
        assert np.isfinite(geo).all() and np.isfinite(motion).all()
        return {'geo':geo,'motion':motion,'geo_valid':geo_valid,'motion_valid':bool(valid[0])}

