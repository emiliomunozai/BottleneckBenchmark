import torch
import torch.nn.functional as F
from bb.tasks.base import BaseTask
from bb.registry import register_task


@register_task("set", description="predict which tokens appear (set membership)")
class SetTask(BaseTask):

    def __init__(self, generator, loss_fn):
        self.generator = generator
        self._pad_idx = generator.vocab_size  # out-of-vocab sentinel for padding
        pad_idx = self._pad_idx
        # Override loss_fn so both train_step (which calls loss_fn directly) and
        # BaseTask.loss() use ignore_index — the pad sentinel is out-of-range for logits.
        self.loss_fn = lambda logits, targets: F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=pad_idx,
        )

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        *,
        apply_input_corruption: bool | None = None,
    ):
        x_clean = self.generator.sample(batch_size)
        L = x_clean.shape[1]
        y_list = []
        for b in range(batch_size):
            unique_sorted = x_clean[b].unique().sort().values  # [K], K <= vocab_size
            pad_len = L - unique_sorted.shape[0]
            pad = x_clean.new_full((pad_len,), self._pad_idx)
            y_list.append(torch.cat([unique_sorted, pad]))
        y = torch.stack(y_list)  # [B, L]
        x = self.generator.apply_input_corruption(
            x_clean, split, apply_input_corruption=apply_input_corruption
        )
        return x, y

    def evaluate(self, model, batch):
        x, y = batch
        preds = model(x).argmax(dim=-1)  # [B, L]
        mask = y != self._pad_idx
        if not mask.any():
            return 1.0
        return (preds[mask] == y[mask]).float().mean().item()
