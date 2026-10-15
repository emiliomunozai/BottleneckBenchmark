"""Transformer decoder conditioned on the fixed bottleneck."""

import torch
import torch.nn as nn

from bb.models.bottleneck import SequenceDecoder
from bb.registry import register_decoder


@register_decoder(
    "transformer",
    constructor_params=["d_model", "n_layers", "n_heads"],
    param_defaults={"n_layers": 2, "n_heads": 4},
)
class TransformerSequenceDecoder(SequenceDecoder):
    def __init__(
        self,
        z_dim: int,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.memory_expand = nn.Linear(z_dim, seq_len * d_model)
        self.query = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.query_pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.memory_pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.query_pos, std=0.02)
        nn.init.normal_(self.memory_pos, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = z.size(0)
        memory = self.memory_expand(z).view(batch_size, self.seq_len, -1)
        memory = memory[:, :seq_len, :] + self.memory_pos[:, :seq_len, :]
        target = (self.query[:, :seq_len, :] + self.query_pos[:, :seq_len, :]).expand(batch_size, -1, -1)
        out = self.decoder(target, memory)
        out = self.norm(out)
        return self.head(out)
