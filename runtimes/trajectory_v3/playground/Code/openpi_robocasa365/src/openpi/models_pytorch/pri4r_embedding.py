"""Pri4R pi-family observation reader, separate from the original VLM prefix.

Paper Fig.2: learnable queries -> self/cross attention -> Z_t -> two-MLP head.
Unpublished details are explicit local choices: two pre-norm decoder blocks,
8 heads, FFN 2*d, exact GELU, no dropout, final LayerNorm. Current point-head
weights/layout, fusion and loss implementation are shared with Query50.
"""
import torch
from torch import nn

from openpi.models_pytorch.point_track import QueryPointTrackHead


class EmbeddingBlock(nn.Module):
    def __init__(self, d, heads, ff_multiplier):
        super().__init__()
        self.self_norm = nn.LayerNorm(d)
        self.self_attention = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_multiplier*d), nn.GELU(), nn.Linear(ff_multiplier*d, d))

    def forward(self, query, memory, valid):
        q = self.self_norm(query)
        query = query + self.self_attention(q, q, q, need_weights=False)[0]
        query = query + self.cross_attention(self.cross_norm(query), memory, memory,
                                             key_padding_mask=~valid, need_weights=False)[0]
        return query + self.ff(self.ff_norm(query))


class Pri4REmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.queries = nn.Parameter(torch.empty(1, config.horizon, d))
        nn.init.normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([EmbeddingBlock(d, config.num_heads, config.embedding_ff_multiplier)
                                    for _ in range(config.embedding_layers)])
        self.final_norm = nn.LayerNorm(d)

    def forward(self, hidden, valid):
        if hidden.ndim != 3 or hidden.shape[-1] != self.queries.shape[-1]:
            raise ValueError('Expected final VLM hidden states [B,L,d]')
        if valid.dtype != torch.bool or valid.shape != hidden.shape[:2] or not valid.any(dim=1).all():
            raise ValueError('Every sample needs a valid image/language token')
        # Mask before projection as well: invalid padding may contain NaNs.
        memory = torch.where(valid[..., None], hidden, 0).to(self.queries.dtype)
        query = self.queries.expand(hidden.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(query, memory, valid)
        return self.final_norm(query)


class EmbeddingPointTrackHead(QueryPointTrackHead):
    def __init__(self, input_dim, config):
        if config.version != 'pri4r_embedding_v1':
            raise ValueError('Embedding head requires its distinct checkpoint contract')
        # Initialize common MLPs first, identically to the existing head.
        super().__init__(input_dim, config)
        self.embedding = Pri4REmbedding(config)

    def encode_time(self, prefix_hidden, prefix_valid):
        return self.embedding(prefix_hidden, prefix_valid).to(self.fusion_in.weight.dtype)
