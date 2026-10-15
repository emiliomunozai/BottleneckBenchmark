import torch
import torch.nn.functional as F


def cross_entropy(logits, targets):
    """
    Cross-entropy for sequence predictions.
    logits: [B, L, V]   targets: [B, L]
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )


def shift_tolerant_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    window: int = 2,
    sigma: float | None = None,
) -> torch.Tensor:
    """
    Shift-tolerant cross-entropy for sequence copy tasks.

    At each position t the target is a soft distribution blending the
    one-hot targets from [t-window … t+window], weighted by a Gaussian
    with the given sigma.  A copy shifted by one position is penalised
    much less than with standard CE; a perfectly aligned copy still
    achieves the global optimum.

    Args:
        logits:  [B, L, V]
        targets: [B, L]  (integer token ids)
        window:  max positional offset to blend (default 2)
        sigma:   Gaussian std-dev over offsets; defaults to window/2,
                 so the window edge gets ~14% relative weight.

    Returns:
        Scalar loss.
    """
    if sigma is None:
        sigma = window / 2.0

    B, L, V = logits.shape

    offsets = torch.arange(-window, window + 1, device=targets.device)  # [2w+1]
    weights = torch.exp(-offsets.float() ** 2 / (2 * sigma ** 2))
    weights = weights / weights.sum()                                    # normalised

    one_hot = F.one_hot(targets, V).float()                              # [B, L, V]
    padded  = F.pad(one_hot, (0, 0, window, window))                     # [B, L+2w, V]

    soft_targets = torch.zeros_like(one_hot)
    for i, off in enumerate(offsets.tolist()):
        soft_targets += weights[i] * padded[:, window + off : window + off + L, :]

    log_probs = F.log_softmax(logits, dim=-1)                            # [B, L, V]
    return -(soft_targets * log_probs).sum(dim=-1).mean()