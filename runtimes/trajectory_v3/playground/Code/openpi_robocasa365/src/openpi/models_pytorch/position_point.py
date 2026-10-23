"""A/B/C heads: start-relative positions and shared causal time decoding.

No future target or future validity is accepted by the head. The causal contract
applies to fused point/time features, not to the upstream bidirectional embedding.
"""
import torch
from torch import nn
from torch.nn import functional as F

from openpi.models_pytorch.pri4r_embedding import EmbeddingPointTrackHead
from openpi.models_pytorch.point_track import QueryPointTrackHead


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


class _PositionDecoder:
    """One implementation for B and C; the only difference is the Z_t source."""
    def _init_temporal(self, config):
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


class PositionPointHead(_PositionDecoder, EmbeddingPointTrackHead):
    def __init__(self, input_dim, config):
        if config.objective_version != "relative_position_and_difference_v3":
            raise ValueError("Position head requires the V3 objective signature")
        # Preserve A/B initialization order, state names and RNG behavior.
        super().__init__(input_dim, config)
        self._init_temporal(config)


class QueryPositionPointHead(_PositionDecoder, QueryPointTrackHead):
    """Final VLM Query features are Z_t directly, with B's identical point decoder."""
    def __init__(self, input_dim, config):
        if config.version != "pri4r_query_zt_v1" or config.objective_version != "relative_position_and_difference_v3":
            raise ValueError("Query position head requires the direct-Query V3 contract")
        super().__init__(input_dim, config)
        self._init_temporal(config)


def validate_position_policy(policy):
    """Check the actual modules, shared by training and action-only inference gates."""
    cfg = policy.point_track
    cfg.validate()
    if not cfg.enabled or cfg.objective_version != "relative_position_and_difference_v3":
        raise ValueError("Expected the V3 position objective")
    if policy.foresight.auxiliary_enabled or hasattr(policy, "future_decoder"):
        raise ValueError("V3 has no future-image decoder or auxiliary target")
    direct = cfg.version == "pri4r_query_zt_v1"
    if direct:
        fs = policy.foresight
        if (not fs.enabled or fs.version != "prefix_query_track_v1"
                or fs.num_queries != cfg.horizon or fs.attention_mode != "x_to_q_then_action"
                or not hasattr(policy, "future_queries")
                or tuple(policy.future_queries.shape) != (1, cfg.horizon, cfg.hidden_dim)
                or not isinstance(policy.point_track_head, QueryPositionPointHead)
                or hasattr(policy.point_track_head, "embedding")):
            raise ValueError("Query C requires direct final-normalized prefix Query features")
    elif (policy.foresight.enabled or hasattr(policy, "future_queries")
          or not isinstance(policy.point_track_head, PositionPointHead)):
        raise ValueError("Embedding A/B preserve the original action prefix")
    return direct
