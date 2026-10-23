"""FP32 target-encoder EMA, independent of the policy's inference EMA.

Post-optimizer momentum update and checkpoint ownership follow I-JEPA's target
encoder. Constant 0.999 is the FoMoVLA recipe choice; 1.0 is our frozen control.
The inference encoder retains the original mixed precision in both variants.
"""
import hashlib
import math

import torch

FORMAT = 'siglip_fp32_shadow_v1'


def ensure_model_device(model, device):
    """The teacher is a homogeneous, plain module; avoid repeated full walks."""
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if next(model.parameters()).device != device:
        model.to(device)
    if model.training:
        model.eval()
    return device


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_ema_state(state, *, decay, identity, reference, updates=None):
    if state['format'] != FORMAT or state['decay'] != decay or state['identity'] != identity:
        raise ValueError('EMA teacher recipe/initial identity mismatch')
    if type(state['updates']) is not int or state['updates'] < 0:
        raise ValueError('Invalid teacher update count')
    if updates is not None and state['updates'] != updates:
        raise ValueError('Teacher update count does not match optimizer progress')
    shadow = state['shadow']
    if not reference or shadow.keys() != reference.keys():
        raise ValueError('EMA teacher parameter/buffer keys mismatch')
    for name, value in shadow.items():
        expected = reference[name]
        dtype = torch.float32 if expected.is_floating_point() else expected.dtype
        if value.shape != expected.shape or value.dtype != dtype:
            raise ValueError(f'EMA teacher shape/dtype mismatch: {name}')
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite teacher state: {name}')
    if state_hash(shadow) != state['shadow_sha256']:
        raise ValueError('EMA teacher checkpoint checksum mismatch')


class MomentumState:
    """Plain object: excluded from student modules, optimizer, and ZeRO casting."""

    def initialize_momentum(self, decay):
        if not math.isfinite(decay) or not 0. <= decay <= 1.:
            raise ValueError('EMA decay must be finite and in [0, 1]')
        self.decay, self.updates = float(decay), 0
        self.model.requires_grad_(False).eval()
        # Start from the actual mixed-precision initial teacher. Both variants
        # therefore produce exactly identical targets at optimizer step zero.
        self.shadow = {k: v.detach().float().clone() if v.is_floating_point() else v.detach().clone()
                       for k, v in self.model.state_dict().items()}
        self.identity.update(teacher_update=FORMAT, teacher_ema_decay=self.decay)

    def to(self, device):
        device = ensure_model_device(self.model, device)
        if next(iter(self.shadow.values())).device != device:
            self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        return self

    @torch.no_grad()
    def _refresh_encoder(self):
        for name, value in self.model.state_dict().items():
            value.copy_(self.shadow[name])
        self.model.requires_grad_(False).eval()

    @torch.no_grad()
    def update(self, student, step):
        if step != self.updates + 1:
            raise ValueError(f'Teacher updates must advance once: {self.updates} -> {step}')
        source = student.state_dict()
        if source.keys() != self.shadow.keys():
            raise ValueError('Student vision encoder and teacher keys differ')
        self.to(next(student.parameters()).device)
        if self.decay < 1.:
            for name, value in self.shadow.items():
                if value.shape != source[name].shape:
                    raise ValueError(f'Student/teacher shape mismatch: {name}')
                if value.is_floating_point():
                    # FP32 accumulation is required: tiny BF16 lerp updates can
                    # round to zero indefinitely. Only one layer is cast at a time.
                    value.lerp_(source[name].detach().float(), 1. - self.decay)
                else:
                    value.copy_(source[name])
            self._refresh_encoder()
        self.updates = step

    def state_dict(self):
        # One independent snapshot on either CPU or CUDA, with no second full
        # CPU clone after an already allocating GPU -> CPU transfer.
        shadow = {k: v.detach().to(device='cpu', copy=True) for k, v in self.shadow.items()}
        return dict(format=FORMAT, decay=self.decay, updates=self.updates,
                    identity=dict(self.identity), shadow=shadow, shadow_sha256=state_hash(shadow))

    def load_state_dict(self, state):
        validate_ema_state(state, decay=self.decay, identity=self.identity,
                           reference=self.model.state_dict())
        for name, value in state['shadow'].items():
            self.shadow[name].copy_(value)
        self.updates = state['updates']
        self._refresh_encoder()

    def audit(self):
        for name, value in self.model.state_dict().items():
            if not torch.equal(value, self.shadow[name].to(value.device, value.dtype)):
                raise ValueError(f'Teacher inference/shadow mismatch: {name}')
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ValueError('Teacher must stay in eval mode without gradients')
        return dict(updates=self.updates, decay=self.decay, shadow_sha256=state_hash(self.shadow),
                    encoder_matches_fp32_shadow=True, frozen_no_grad=True)
