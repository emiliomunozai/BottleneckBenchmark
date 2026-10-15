import torch
from bb.generators.base import BaseGenerator
from bb.registry import register_generator


@register_generator(
    "palindrome",
    constructor_params=["input_noise", "corruption_mode", "mask_token_id", "pad_token_id"],
    param_defaults={"input_noise": 0.0, "corruption_mode": "replace", "mask_token_id": 0,
                     "pad_token_id": None},
)
class PalindromeGenerator(BaseGenerator):
    """Sequences that are palindromes: the second half mirrors the first (an odd
    middle token, if any, is unconstrained). Non-Markovian by construction --
    demonstrates BB's source interface on a structurally different generator."""

    def sample(self, batch_size: int):
        half = self.seq_len // 2
        first_half = torch.randint(0, self.vocab_size, (batch_size, half), dtype=torch.long)
        second_half = first_half.flip(dims=[1])
        if self.seq_len % 2 == 0:
            return torch.cat([first_half, second_half], dim=1)
        middle = torch.randint(0, self.vocab_size, (batch_size, 1), dtype=torch.long)
        return torch.cat([first_half, middle, second_half], dim=1)
