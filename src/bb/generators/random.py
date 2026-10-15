import torch
from bb.generators.base import BaseGenerator
from bb.registry import register_generator

@register_generator(
    "random",
    constructor_params=["input_noise", "corruption_mode", "mask_token_id", "pad_token_id", "structure"],
    param_defaults={"input_noise": 0.0, "corruption_mode": "replace", "mask_token_id": 0,
                     "pad_token_id": None, "structure": 0.0},
)
class RandomSequenceGenerator(BaseGenerator):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        *,
        input_noise: float = 0.0,
        corruption_mode: str = "replace",
        mask_token_id: int = 0,
        pad_token_id: int | None = None,
        structure: float = 0.0,
    ):
        super().__init__(
            seq_len,
            vocab_size,
            input_noise=input_noise,
            corruption_mode=corruption_mode,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )
        self.structure = float(structure)

    def sample(self, batch_size: int):
        if self.structure <= 0.0:
            return torch.randint(0, self.vocab_size, (batch_size, self.seq_len), dtype=torch.long)

        # Markov self-transition: each token repeats the previous with prob `structure`,
        # otherwise sampled uniformly. Reduces per-token entropy to (1-s)*log2(V).
        x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), dtype=torch.long)
        repeat = torch.rand(batch_size, self.seq_len - 1) < self.structure
        for t in range(1, self.seq_len):
            x[:, t] = torch.where(repeat[:, t - 1], x[:, t - 1], x[:, t])
        return x
