"""
Minimal bottleneck composition for BB models.

Every encoder must emit a fixed-size latent ``z`` with shape ``[B, bottleneck_dim]``.
Every decoder must reconstruct token logits from that same latent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """Bottleneck representation for one batch."""

    z: torch.Tensor


class SequenceEncoder(nn.Module, ABC):
    """Maps token ids ``[B, L]`` to a bottleneck vector ``[B, D]``."""

    vocab_size: int
    seq_len: int
    out_dim: int

    def __init__(self, vocab_size: int, seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.out_dim = 0

    @abstractmethod
    def encode(self, x: torch.Tensor) -> EncoderOutput:
        ...


class SequenceDecoder(nn.Module, ABC):
    """Maps bottleneck ``[B, D]`` to token logits ``[B, L, V]``."""

    @abstractmethod
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        ...


class SequenceBottleneckModel(nn.Module):
    """Thin encoder-decoder composition used throughout the benchmark."""

    def __init__(self, encoder: SequenceEncoder, decoder: SequenceDecoder):
        super().__init__()
        if encoder.out_dim <= 0:
            raise ValueError("encoder.out_dim must be set before building the decoder")
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(x).z
        return self.decoder.decode(z, self.encoder.seq_len)


def resolve_bottleneck_dim(d_model: int, bottleneck_dim: int | None) -> int:
    """Fallback to ``d_model`` when configs omit the bottleneck width."""
    return int(bottleneck_dim) if bottleneck_dim is not None else int(d_model)
