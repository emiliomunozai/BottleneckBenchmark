import torch
from bb.tasks.base import BaseTask
from bb.registry import register_task


@register_task("sorting", description="sort input sequence")
class SortingTask(BaseTask):

    def __init__(self, generator, loss_fn):
        self.generator = generator
        self.loss_fn = loss_fn

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
        *,
        apply_input_corruption: bool | None = None,
    ):
        x_clean = self.generator.sample(batch_size)
        y = torch.sort(x_clean, dim=1).values
        x = self.generator.apply_input_corruption(
            x_clean,
            split,
            apply_input_corruption=apply_input_corruption,
        )
        return x, y
