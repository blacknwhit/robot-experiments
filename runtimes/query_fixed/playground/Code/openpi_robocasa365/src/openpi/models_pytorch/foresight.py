"""Multi-target asymmetric visual foresight (project specification V1.1).

Queries are continuous prefix embeddings, not vocabulary entries. All future
targets stay outside the policy input. The decoder is training-only.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ForesightConfig:
    enabled: bool = False
    version: str = "multi_target_asym_v1"
    num_queries: int = 64
    query_init_std: float = 0.02
    attention_mode: str = "x_to_q_then_action"
    future_offsets: tuple[int, ...] = (25, 50)
    future_view_keys: tuple[str, ...] = (
        "video.robot0_agentview_left", "video.robot0_eye_in_hand", "video.robot0_agentview_right")
    auxiliary_enabled: bool = True
    loss_weight: float = 0.1
    loss_warmup_optimizer_steps: int = 1000
    teacher_microbatch: int = 4
    teacher_ema_decay: float | None = None
    augmentation: str = "deterministic"
    target_cache: str | None = None

    @classmethod
    def parse(cls, value=None, *, horizon=50):
        if value is None:
            return cls()
        if isinstance(value, cls):
            result = value
        else:
            value = dict(value)
            for key in ("future_offsets", "future_view_keys"):
                if key in value:
                    value[key] = tuple(value[key])
            result = cls(**value)
        if result.enabled:
            if result.num_queries != 64 or result.version != "multi_target_asym_v1":
                raise ValueError("V1.1 requires exactly 64 shared queries")
            if result.attention_mode not in ("x_to_q_then_action", "bidirectional_xq"):
                raise ValueError("Unknown foresight attention mode")
            if result.future_offsets != (math.ceil(horizon / 2), horizon):
                raise ValueError("RoboCasa synchronous targets must match half/full action horizon")
            if len(set(result.future_offsets)) != 2 or min(result.future_offsets) <= 0:
                raise ValueError("Two distinct future times required")
            if not result.future_view_keys or len(set(result.future_view_keys)) != len(result.future_view_keys):
                raise ValueError("Unique explicit camera slots required")
            if result.teacher_microbatch < 1 or result.loss_warmup_optimizer_steps < 1:
                raise ValueError("Positive teacher microbatch and warmup required")
            if result.augmentation not in ("deterministic", "official_pytorch_temporal_sync_v1"):
                raise ValueError("Unknown foresight preprocessing contract")
            if result.augmentation != "deterministic" and result.target_cache is not None:
                raise ValueError("Random synchronized augmentation cannot reuse per-frame target caches")
            if result.teacher_ema_decay is not None:
                if not 0. <= result.teacher_ema_decay <= 1.:
                    raise ValueError("Teacher EMA decay must be in [0, 1]")
                if result.teacher_ema_decay < 1. and result.target_cache is not None:
                    raise ValueError("An evolving EMA teacher cannot use a static target cache")
        return result

    def weight(self, completed_steps):
        return self.loss_weight * min(max(int(completed_steps), 0) / self.loss_warmup_optimizer_steps, 1.)

    def signature(self, d, dv, grid):
        contract = asdict(self)
        # Execution placement and target caching do not change the learned model.
        for key in ("teacher_microbatch", "target_cache"):
            contract.pop(key)
        # Preserve signatures of already trained legacy fixed-teacher policies.
        if self.teacher_ema_decay is None:
            contract.pop("teacher_ema_decay")
        contract.update(d=d, dv=dv, grid=grid)
        return torch.tensor(list(hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).digest()), dtype=torch.uint8)


def append_queries(x, pad, ar, queries, *, bidirectional=False):
    q = queries.to(x.dtype).expand(x.shape[0], -1, -1)
    q_pad = torch.ones(q.shape[:2], device=x.device, dtype=torch.bool)
    q_ar = torch.zeros(q.shape[:2], device=ar.device, dtype=ar.dtype)
    if not bidirectional:
        q_ar[:, 0] = 1
    return torch.cat((x, q), 1), torch.cat((pad, q_pad), 1), torch.cat((ar, q_ar), 1)


class FutureDecoder(nn.Module):
    def __init__(self, d, dv, k=2, v=3, grid=(16, 16)):
        super().__init__()
        self.k, self.v, self.dv, self.grid = k, v, dv, tuple(grid)
        self.shared = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.heads = nn.ModuleList(nn.Linear(d, dv) for _ in range(k * v))

    def forward(self, hq):
        b, m, _ = hq.shape
        if m != 64:
            raise ValueError(f"Expected 64 Q, got {m}")
        u = self.shared(hq.to(self.shared[1].weight.dtype))
        z = torch.stack([head(u) for head in self.heads], dim=1)
        z = z.reshape(b, self.k, self.v, 8, 8, self.dv)
        z = z.permute(0, 1, 2, 5, 3, 4).reshape(-1, self.dv, 8, 8)
        z = F.interpolate(z.float(), size=self.grid, mode="bilinear", align_corners=False)
        return z.reshape(b, self.k, self.v, self.dv, -1).permute(0, 1, 2, 4, 3).contiguous()


def future_loss_terms(pred, target, valid):
    if pred.shape != target.shape or pred.ndim != 5 or valid.shape != pred.shape[:3]:
        raise ValueError("Expected matching [B,K,V,P,Dv] and bool [B,K,V]")
    if valid.dtype != torch.bool:
        raise TypeError("future_valid must be bool")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Nonfinite future prediction")
    # Invalid target storage is finite zero, including the all-invalid case.
    if not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite future target")
    p = F.normalize(pred.float(), dim=-1, eps=1e-6)
    t = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
    per_target = (1 - (p * t).sum(-1)).mean(-1)
    masked = torch.where(valid, per_target, torch.zeros_like(per_target))
    return masked.sum(), valid.sum().float(), per_target.detach()


def distributed_future_loss(local_sum, global_step_count, *, world_size, accumulation_steps):
    """For a trainer that averages DP gradients and divides each loss by G."""
    return local_sum * (world_size * accumulation_steps) / global_step_count.clamp_min(1)


def tensor_state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
