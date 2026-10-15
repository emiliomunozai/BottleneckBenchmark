"""
Decoders: bottleneck ``[B, D]`` → logits ``[B, L, V]``.
"""

from bb.models.bottleneck import SequenceDecoder
from bb.models.decoders.lstm import LSTMSequenceDecoder
from bb.models.decoders.ssm import SSMSequenceDecoder
from bb.models.decoders.transformer import TransformerSequenceDecoder

__all__ = [
    "LSTMSequenceDecoder",
    "SSMSequenceDecoder",
    "SequenceDecoder",
    "TransformerSequenceDecoder",
]
