"""A/B heads: start-relative positions, optional same-point causal time blocks.

No future target or future validity is accepted by the head. The causal contract
applies to fused point/time features, not to the upstream bidirectional embedding.
"""
import torch
from torch import nn
from torch.nn import functional as F

from openpi.models_pytorch.pri4r_embedding import EmbeddingPointTrackHead


class CausalPointTimeBlock(nn.Module):
    def __init__(self, width, heads, ff_dim, horizon):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_dim), nn.GELU(), nn.Linear(ff_dim, width))
        # MultiheadAttention boolean True means blocked (opposite SDPA's bool convention).
        self.register_buffer("blocked_future", torch.ones(horizon, horizon, dtype=torch.bool).triu(1),
                             persistent=False)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.blocked_future.shape[0]:
            raise ValueError("Expected [batch*points,horizon,width]")
        q = self.attention_norm(x)
        x = x + self.attention(q, q, q, attn_mask=self.blocked_future, need_weights=False)[0]
        return x + self.ff(self.ff_norm(x))


class PositionPointHead(EmbeddingPointTrackHead):
    def __init__(self, input_dim, config):
        if config.objective_version != "relative_position_and_difference_v3":
            raise ValueError("Position head requires the V3 objective signature")
        # Common PointMLP/fusion/Embedding initialize before B-only parameters.
        super().__init__(input_dim, config)
        with torch.random.fork_rng(devices=[]):
            self.temporal = nn.ModuleList([
                CausalPointTimeBlock(config.fusion_hidden_dim, config.temporal_heads,
                                     config.temporal_ff_dim, config.horizon)
                for _ in range(config.temporal_layers)
            ])

    def _fuse_projected(self, point_projection, time_projection):
        hidden = F.gelu(point_projection[:, :, None, :] + time_projection[:, None, :, :])
        b, n, h, d = hidden.shape
        hidden = hidden.reshape(b * n, h, d)
        for layer in self.temporal:
            hidden = layer(hidden)
        return self.fusion_out(hidden).reshape(b, n, h, 3)
