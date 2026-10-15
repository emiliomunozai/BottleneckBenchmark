"""Bidirectional LSTM encoder with a fixed bottleneck projection."""

from torch import nn

from bb.models.bottleneck import EncoderOutput, SequenceEncoder
from bb.registry import register_encoder


@register_encoder(
    "bi_lstm",
    constructor_params=["d_model", "bottleneck_dim", "n_layers"],
    param_defaults={"n_layers": 1},
)
class BiLSTMSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        bottleneck_dim: int,
        n_layers: int,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.LSTM(
            d_model,
            d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.to_bottleneck = nn.Linear(2 * d_model, bottleneck_dim)
        self.out_dim = bottleneck_dim

    def encode(self, x):
        x = self.embed(x)
        _, (h_n, _) = self.rnn(x)
        h_last = h_n[-2:].transpose(0, 1).reshape(x.size(0), -1)
        z = self.to_bottleneck(h_last)
        return EncoderOutput(z=z)
