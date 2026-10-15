"""SSM encoder: diagonal real state space model."""

import torch
import torch.nn as nn

from bb.models.bottleneck import EncoderOutput, SequenceEncoder
from bb.registry import register_encoder


class _SSMLayer(nn.Module):
    """Single diagonal real SSM layer: h_t = A * h_{t-1} + B * x_t, y_t = C * h_t + D * x_t."""

    def __init__(self, d_model: int):
        super().__init__()
        # log_A init=2 → sigmoid(2)≈0.88: slower decay, better long-range retention
        self.log_A = nn.Parameter(torch.full((d_model,), 2.0))
        self.B = nn.Linear(d_model, d_model, bias=False)
        self.C = nn.Linear(d_model, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, L, _ = x.shape
        A = 0.5 + 0.5 * torch.sigmoid(self.log_A)   # [d_model], A ∈ (0.5, 1.0)
        Bx = self.B(x)                               # [B, L, d_model]
        # float64 for scan: A^t underflows float32 at large t (e.g. 0.75^512 ≈ 3e-64)
        t_idx = torch.arange(L, device=x.device, dtype=torch.float64).view(1, L, 1)
        At = A.double().view(1, 1, -1) ** t_idx      # [1, L, d_model] in float64
        h = (torch.cumsum(Bx.double() / At, dim=1) * At).to(x.dtype)
        y = self.C(h) + self.D * x                   # skip connection
        return self.norm(y)


@register_encoder(
    "bi_ssm",
    constructor_params=["d_model", "bottleneck_dim", "n_layers"],
    param_defaults={"n_layers": 1},
)
class BiSSMSequenceEncoder(SequenceEncoder):
    """Bidirectional SSM encoder: forward + backward passes, concat final states."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        bottleneck_dim: int,
        n_layers: int = 1,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fwd_layers = nn.ModuleList([_SSMLayer(d_model) for _ in range(n_layers)])
        self.bwd_layers = nn.ModuleList([_SSMLayer(d_model) for _ in range(n_layers)])
        self.to_bottleneck = nn.Linear(2 * d_model, bottleneck_dim)
        self.out_dim = bottleneck_dim

    def encode(self, x: torch.Tensor) -> EncoderOutput:
        h = self.embed(x)
        fwd = h
        for layer in self.fwd_layers:
            fwd = layer(fwd)
        bwd = h.flip(1)
        for layer in self.bwd_layers:
            bwd = layer(bwd)
        combined = torch.cat([fwd[:, -1], bwd[:, -1]], dim=-1)
        z = self.to_bottleneck(combined)
        return EncoderOutput(z=z)
