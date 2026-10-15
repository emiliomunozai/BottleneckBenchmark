"""SSM decoder conditioned on the fixed bottleneck."""

import torch
import torch.nn as nn

from bb.models.bottleneck import SequenceDecoder
from bb.registry import register_decoder


@register_decoder(
    "ssm",
    constructor_params=["d_model", "n_layers"],
    param_defaults={"n_layers": 1},
)
class SSMSequenceDecoder(SequenceDecoder):
    """Stack of diagonal SSM layers; each step conditioned on pos_emb + z_proj(z)."""

    def __init__(
        self,
        z_dim: int,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.init_h = nn.Linear(z_dim, n_layers * d_model)

        self.log_A = nn.ParameterList([nn.Parameter(torch.full((d_model,), 2.0)) for _ in range(n_layers)])
        self.B = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)])
        self.C = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)])
        self.D = nn.ParameterList([nn.Parameter(torch.ones(d_model)) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.head = nn.Linear(d_model, vocab_size)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        b = z.size(0)
        pos = torch.arange(seq_len, device=z.device)
        h = self.pos_emb(pos).unsqueeze(0).expand(b, -1, -1) + self.z_proj(z).unsqueeze(1)

        # h_states provides per-layer initial hidden states from z.
        # Shape: [B, n_layers * d_model] → split per layer below.
        h_states = self.init_h(z)

        t_idx = torch.arange(seq_len, device=z.device, dtype=torch.float64).view(1, seq_len, 1)
        for i in range(self.n_layers):
            A = (0.5 + 0.5 * torch.sigmoid(self.log_A[i])).double().view(1, 1, -1)
            Bh = self.B[i](h)
            At = A ** t_idx  # A^0, A^1, ..., A^{L-1}  shape [1, L, d_model]

            # Zero-initial-state response: h_t = Σ_{k=0}^{t} A^{t-k} * Bh_k
            h_ssm = torch.cumsum(Bh.double() / At, dim=1) * At

            # Add initial-state contribution: h_0 drives A^{t+1}*h_0 at position t
            h0_i = h_states[:, i * self.d_model:(i + 1) * self.d_model]  # [B, d_model]
            h_ssm = h_ssm + A * At * h0_i.double().unsqueeze(1)

            y = self.C[i](h_ssm.to(h.dtype)) + self.D[i] * h
            h = self.norms[i](y)

        return self.head(h)
