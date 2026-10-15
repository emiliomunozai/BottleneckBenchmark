"""One encoder, two decoders sharing one bottleneck z."""

import torch
import torch.nn as nn

from bb.models.bottleneck import SequenceEncoder, SequenceDecoder


class MultiTaskBottleneckModel(nn.Module):
    def __init__(self, encoder: SequenceEncoder, copy_decoder: SequenceDecoder, sort_decoder: SequenceDecoder):
        super().__init__()
        self.encoder      = encoder
        self.copy_decoder = copy_decoder
        self.sort_decoder = sort_decoder

    def forward(self, x: torch.Tensor):
        z   = self.encoder.encode(x).z
        seq = self.encoder.seq_len
        return self.copy_decoder.decode(z, seq), self.sort_decoder.decode(z, seq)
