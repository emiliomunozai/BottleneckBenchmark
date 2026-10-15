"""Transformer encoder with a learned bottleneck token."""

import torch
from torch import nn

from bb.models.bottleneck import EncoderOutput, SequenceEncoder
from bb.registry import register_encoder


@register_encoder(
    "transformer",
    constructor_params=["d_model", "bottleneck_dim", "n_layers", "n_heads"],
    param_defaults={"n_layers": 2, "n_heads": 4},
)
class TransformerSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        bottleneck_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.to_bottleneck = nn.Linear(d_model, bottleneck_dim)
        self.out_dim = bottleneck_dim

    def encode(self, x):
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = self.embed(x)
        h = torch.cat([cls, tokens], dim=1)
        h = h + self.pos_emb[:, : h.size(1), :]
        h = self.encoder(h)
        h = self.norm(h[:, 0, :])
        z = self.to_bottleneck(h)
        return EncoderOutput(z=z)
