from abc import ABC, abstractmethod

import torch


def corrupt_input_tokens(
    x: torch.Tensor,
    vocab_size: int,
    noise_rate: float,
    *,
    pad_token_id: int | None = None,
    mode: str = "replace",
    mask_token_id: int = 0,
) -> torch.Tensor:
    """Corrupt input tokens by random replacement or masking."""
    if noise_rate <= 0 or vocab_size <= 1:
        return x

    if pad_token_id is None:
        eligible = torch.ones_like(x, dtype=torch.bool, device=x.device)
    else:
        eligible = x != pad_token_id
    flip = (torch.rand_like(x, dtype=torch.float32, device=x.device) < noise_rate) & eligible

    if mode == "mask":
        mask_value = torch.full_like(x, int(mask_token_id))
        return torch.where(flip, mask_value, x)
    if mode != "replace":
        raise ValueError(f"Unknown corruption mode: {mode!r}. Use 'replace' or 'mask'.")

    wrong = torch.randint(0, vocab_size, x.shape, device=x.device, dtype=x.dtype)
    wrong = torch.where(wrong == x, (x + 1) % vocab_size, wrong)
    return torch.where(flip, wrong, x)


class BaseGenerator(ABC):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        *,
        input_noise: float = 0.0,
        corruption_mode: str = "replace",
        mask_token_id: int = 0,
        pad_token_id: int | None = None,
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.input_noise = float(input_noise)
        self.corruption_mode = str(corruption_mode)
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = pad_token_id if pad_token_id is None else int(pad_token_id)

    def apply_input_corruption(
        self,
        x: torch.Tensor,
        split: str,
        *,
        apply_input_corruption: bool | None = None,
    ) -> torch.Tensor:
        should_corrupt = (
            split == "train"
            if apply_input_corruption is None
            else bool(apply_input_corruption)
        )
        if not should_corrupt or self.input_noise <= 0:
            return x
        return corrupt_input_tokens(
            x,
            self.vocab_size,
            self.input_noise,
            pad_token_id=self.pad_token_id,
            mode=self.corruption_mode,
            mask_token_id=self.mask_token_id,
        )

    @abstractmethod
    def sample(self, batch_size: int):
        """
        Returns x: input tensor [batch_size, seq_len] (LongTensor).
        The task computes targets y from x (e.g. sort, copy).
        """
        pass
