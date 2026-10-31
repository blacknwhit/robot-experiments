"""Training-only Pri4R-style point trajectory supervision.

The published method supplies the time-query / PointMLP / FusionMLP structure,
1024 points, adjacent displacements and L1 supervision. Widths, layer counts and
initialization below are an explicit local reconstruction, not official code.
The caller supplies final-normalized observation tokens with the original
foresight-Q span excluded. Do not detach those tokens. Ground-truth future
positions and displacements are never inputs to this head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class PointTrackConfig:
    enabled: bool = False
    version: str = "pri4r_adjacent_v1"
    horizon: int = 50
    num_points: int = 1024
    hidden_dim: int = 256
    num_heads: int = 8
    embedding_layers: int = 2
    embedding_ff_multiplier: int = 2
    point_hidden_dim: int = 128
    fusion_hidden_dim: int = 128
    loss_weight: float = 1.0
    point_chunk_size: int = 256
    checkpoint_fusion: bool = False
    coordinate_frame: str = "window_start_robot_base"
    units: str = "meters"
    output_kind: str = "adjacent_displacement"

    @classmethod
    def parse(cls, value=None, *, horizon=50):
        result = cls(horizon=horizon) if value is None else (
            value if isinstance(value, cls) else cls(**dict(value)))
        result.validate()
        if result.enabled and result.horizon != horizon:
            raise ValueError("Point-track horizon must equal the action horizon")
        return result

    def validate(self):
        if self.version not in ("pri4r_adjacent_v1", "pri4r_query_zt_v1", "pri4r_embedding_v1"):
            raise ValueError("Unknown point-track implementation version")
        for name in ("horizon", "num_points", "hidden_dim", "num_heads",
                     "point_hidden_dim", "fusion_hidden_dim", "point_chunk_size", "embedding_layers", "embedding_ff_multiplier"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not math.isfinite(self.loss_weight) or self.loss_weight < 0:
            raise ValueError("loss_weight must be finite and nonnegative")
        if self.coordinate_frame not in ("window_start_robot_base", "world") or (
                self.units, self.output_kind) != ("meters", "adjacent_displacement"):
            raise ValueError("Point tracks require adjacent meter displacements in a fixed coordinate frame")

    def signature(self, input_dim):
        """Fingerprint the learned contract, excluding execution-only chunking."""
        contract = asdict(self)
        for key in ("enabled", "point_chunk_size", "checkpoint_fusion"):
            contract.pop(key)
        if self.version != "pri4r_embedding_v1":
            contract.pop("embedding_layers")
            contract.pop("embedding_ff_multiplier")
        if self.version == "pri4r_query_zt_v1":
            contract.pop("num_heads")  # No attention module exists in this head.
            contract.update(input_dim=input_dim, activation="gelu_exact",
                            point_time_order="B,N,H,XYZ", reduction="valid_point_time_xyz_mean",
                            prefix_contract="final_norm_query_span_direct_Zt",
                            concatenation_order="point_then_time", output_activation="identity")
        elif self.version == "pri4r_embedding_v1":
            contract.update(input_dim=input_dim, activation="gelu_exact", dropout=0.0,
                point_time_order="B,N,H,XYZ", reduction="valid_point_time_xyz_mean",
                concatenation_order="point_then_time", output_activation="identity",
                prefix_contract="original_final_norm_image_language_tokens",
                embedding_order="pre_norm_self_cross_ffn_final_norm", query_init_std=0.02)
        else:
            contract.update(input_dim=input_dim, ff_multiplier=2, dropout=0.0,
                        activation="gelu_exact", point_time_order="B,N,H,XYZ",
                        reduction="valid_point_time_xyz_mean", query_init_std=0.02,
                        prefix_contract="final_norm_observation_span_without_old_Q")
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).digest()
        return torch.tensor(list(digest), dtype=torch.uint8)


class PointTrackHead(nn.Module):
    """Read observation prefix once, then predict all point/time displacements.

    Args to forward:
        prefix_hidden: [B,L,input_dim], final-normalized observation tokens.
        prefix_valid: bool [B,L], excludes padding and the original Q span.
        point_current: [B,N,3], current positions in frozen F_t, in meters.
        point_current_valid: optional bool [B,N], validity at the current time.
    Returns:
        [B,N,H,3] adjacent displacements, in meters, without output clipping.

    The module never sees future labels. Indirect Q information carried by
    observation hidden states is permitted; there is no information-isolation
    or gradient-blocking operation here.
    """

    def __init__(self, input_dim: int, config: PointTrackConfig):
        super().__init__()
        config.validate()
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        self.config, self.input_dim = config, input_dim
        d = config.hidden_dim
        self.prefix_norm = nn.LayerNorm(input_dim)
        self.prefix_projection = nn.Linear(input_dim, d)
        self.time_queries = nn.Parameter(torch.empty(config.horizon, d))
        nn.init.normal_(self.time_queries, std=0.02)
        self.query_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(d, config.num_heads, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(d)
        self.time_ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.point_mlp = nn.Sequential(nn.Linear(3, config.point_hidden_dim), nn.GELU(),
                                       nn.Linear(config.point_hidden_dim, d))
        # Keep the ordinary concatenation-MLP weight layout for auditability.
        # Forward splits this weight, avoiding a [B,N,H,2*d] concatenation.
        self.fusion_in = nn.Linear(2 * d, config.fusion_hidden_dim)
        self.fusion_out = nn.Linear(config.fusion_hidden_dim, 3)
        self.register_buffer("point_track_signature", config.signature(input_dim), persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "point_track_signature"
        expected = self.config.signature(self.input_dim)
        incoming = state_dict.get(key)
        if incoming is not None and (incoming.shape != expected.shape or not torch.equal(
                incoming.detach().cpu(), expected)):
            error_msgs.append(f"{key}: point-track semantic contract mismatch")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def encode_time(self, prefix_hidden, prefix_valid):
        if prefix_hidden.ndim != 3 or prefix_hidden.shape[-1] != self.input_dim:
            raise ValueError("Expected prefix_hidden [B,L,input_dim]")
        if prefix_valid.dtype != torch.bool or prefix_valid.shape != prefix_hidden.shape[:2]:
            raise ValueError("Expected bool prefix_valid [B,L]")
        if prefix_hidden.shape[1] == 0 or not prefix_valid.any(dim=1).all():
            raise ValueError("Every sample needs at least one valid observation token")
        # Sanitize padding before normalization/projection; masked NaNs must not
        # enter GEMMs or their backward paths through a subsequent zero mask.
        source = torch.where(prefix_valid[..., None], prefix_hidden, 0)
        source = source.to(self.prefix_projection.weight.dtype)
        memory = self.prefix_projection(self.prefix_norm(source))
        queries = self.time_queries.to(memory.dtype).unsqueeze(0).expand(source.shape[0], -1, -1)
        update, _ = self.cross_attention(self.query_norm(queries), memory, memory,
                                        key_padding_mask=~prefix_valid, need_weights=False)
        context = queries + update
        return context + self.time_ff(self.ff_norm(context))

    def _fuse_projected(self, point_projection, time_projection):
        hidden = point_projection[:, :, None, :] + time_projection[:, None, :, :]
        return self.fusion_out(F.gelu(hidden))

    def forward(self, prefix_hidden, prefix_valid, point_current, point_current_valid=None):
        b = prefix_hidden.shape[0]
        expected = (b, self.config.num_points, 3)
        if point_current.shape != expected:
            raise ValueError(f"Expected point_current {expected}, got {tuple(point_current.shape)}")
        if point_current_valid is None:
            point_current_valid = torch.ones(point_current.shape[:2], dtype=torch.bool,
                                             device=point_current.device)
        if point_current_valid.dtype != torch.bool or point_current_valid.shape != point_current.shape[:2]:
            raise ValueError("Expected bool point_current_valid [B,N]")
        points = torch.where(point_current_valid[..., None], point_current, 0)
        context = self.encode_time(prefix_hidden, prefix_valid)
        encoded_points = self.point_mlp(points.to(self.point_mlp[0].weight.dtype))
        d = self.config.hidden_dim
        # W*[point;time]+b = Wp*point + Wt*time + b. Bias appears once.
        point_projection = F.linear(encoded_points, self.fusion_in.weight[:, :d])
        time_projection = F.linear(context, self.fusion_in.weight[:, d:], self.fusion_in.bias)
        chunks = []
        for start in range(0, self.config.num_points, self.config.point_chunk_size):
            part = point_projection[:, start:start + self.config.point_chunk_size]
            if self.config.checkpoint_fusion and self.training and torch.is_grad_enabled():
                result = checkpoint(self._fuse_projected, part, time_projection,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                result = self._fuse_projected(part, time_projection)
            chunks.append(result)
        return torch.cat(chunks, dim=1)


class QueryPointTrackHead(PointTrackHead):
    """Pri4R's two-MLP head with prefix Query outputs directly used as Z_t.

    No pooling, time queries, projection or extra attention transforms Z_t.
    Widths/activation are explicit local choices, not unpublished author code.
    Inherited fusion splits W[point;time]+b to avoid materializing B*N*H*2d.
    """

    def __init__(self, input_dim: int, config: PointTrackConfig):
        nn.Module.__init__(self)
        config.validate()
        if config.version not in ("pri4r_query_zt_v1", "pri4r_embedding_v1") or input_dim != config.hidden_dim:
            raise ValueError("Direct Z_t width must equal PointMLP output width")
        self.config, self.input_dim = config, input_dim
        self.point_mlp = nn.Sequential(nn.Linear(3, config.point_hidden_dim), nn.GELU(),
                                       nn.Linear(config.point_hidden_dim, input_dim))
        self.fusion_in = nn.Linear(2 * input_dim, config.fusion_hidden_dim)
        self.fusion_out = nn.Linear(config.fusion_hidden_dim, 3)
        self.register_buffer("point_track_signature", config.signature(input_dim), persistent=True)

    def encode_time(self, query_hidden, query_valid):
        if query_hidden.ndim != 3 or query_hidden.shape[1:] != (self.config.horizon, self.input_dim):
            raise ValueError("Direct Z_t must be [B,H,d], with one Query per predicted step")
        if (query_valid.dtype != torch.bool or query_valid.shape != query_hidden.shape[:2]
                or not query_valid.all()):
            raise ValueError("Every direct Z_t Query must be valid")
        return query_hidden.to(self.fusion_in.weight.dtype)


def query_prefix_span(prefix_hidden, prefix_valid, *, num_queries, horizon):
    """Select the appended final-normalized Query span without detaching it."""
    if num_queries != horizon or horizon < 1:
        raise ValueError("Direct Query span must match the prediction horizon")
    if (prefix_hidden.ndim != 3 or prefix_valid.dtype != torch.bool
            or prefix_valid.shape != prefix_hidden.shape[:2] or prefix_hidden.shape[1] <= num_queries):
        raise ValueError("Expected observation tokens followed by a nonempty Query span")
    hidden, valid = prefix_hidden[:, -num_queries:], prefix_valid[:, -num_queries:]
    if not valid.all():
        raise ValueError("Invalid token in appended Query span")
    return hidden, valid


def observation_prefix_without_queries(prefix_hidden, prefix_valid, *, query_enabled, num_queries):
    """Slice only the new reader's view; retain the original full prefix.

    This integration contract targets the existing append-at-end Query64 layout.
    It is not a generic 'drop the final 64 tokens' operation on arbitrary models.
    """
    if not query_enabled or num_queries != 64:
        raise ValueError("Query+Pri4R V1 requires the original appended Query64 span")
    if prefix_hidden.ndim != 3 or prefix_valid.dtype != torch.bool or prefix_valid.shape != prefix_hidden.shape[:2]:
        raise ValueError("Prefix hidden states and bool validity must have matching [B,L] spans")
    observation_length = prefix_hidden.shape[1] - num_queries
    if observation_length < 1 or not prefix_valid[:, observation_length:].all():
        raise ValueError("Expected a nonempty observation span followed by 64 valid query tokens")
    return prefix_hidden[:, :observation_length], prefix_valid[:, :observation_length]


def point_track_loss_terms(pred, target, valid):
    """Return local L1 sum, XYZ count, and detached mean-XYZ per-pair loss.

    A valid pair needs a valid current point and both adjacent GT positions;
    constructing that mask is the dataset's responsibility. Invalid storage may
    contain NaNs. Nonfinite *valid* labels and any nonfinite prediction raise.
    The all-invalid case keeps a differentiable zero connected to predictions.
    """
    if pred.ndim != 4 or pred.shape[-1] != 3 or pred.shape != target.shape:
        raise ValueError("Expected matching point deltas [B,N,H,3]")
    if valid.dtype != torch.bool or valid.shape != pred.shape[:-1]:
        raise ValueError("Expected bool point validity [B,N,H]")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("Nonfinite point-track prediction")
    if not torch.where(valid[..., None], torch.isfinite(target), True).all():
        raise FloatingPointError("Nonfinite valid point-track target")
    safe_target = torch.where(valid[..., None], target.detach(), 0).float()
    # Mask before subtracting as well as after: 0 * NaN is still NaN.
    safe_pred = torch.where(valid[..., None], pred, 0).float()
    error = (safe_pred - safe_target).abs()
    return error.sum(), valid.sum().to(torch.float32) * 3, error.mean(dim=-1).detach()


def distributed_point_track_loss(local_sum, global_step_count, *, world_size, accumulation_steps):
    """Correct gradient for DDP averaging plus loss/G gradient accumulation.

    global_step_count is the globally summed number of valid XYZ elements over
    the *whole optimizer-step accumulation window*. Prefetch and all-reduce the
    count once, as for the existing future loss. This helper has no collective;
    logging must separately globally sum numerators and divide by the count.
    """
    if world_size < 1 or accumulation_steps < 1:
        raise ValueError("world_size and accumulation_steps must be positive")
    return local_sum * (world_size * accumulation_steps) / global_step_count.clamp_min(1)
