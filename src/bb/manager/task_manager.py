"""Training loop for seq2seq tasks with optional discrete input corruption.

TaskManager orchestrates optimization/evaluation only. Data semantics (including token
corruption) live in the generator/task batch pipeline.
"""

import torch


class TaskManager:
    def __init__(self, task, model, optimizer, device):
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = []
        self._scheduler = None

    def _get_train_xy(self, batch_size: int):
        x, y = self.task.get_batch(batch_size, split="train")
        return x.to(self.device), y.to(self.device)

    def train_step(self, batch_size: int):
        self.model.train()
        x, y = self._get_train_xy(batch_size)
        logits = self.model(x)
        loss = self.task.loss_fn(logits, y)
        train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        return loss.item(), train_acc

    def eval_step(self, batch_size: int, *, noisy_inputs: bool = False):
        self.model.eval()
        with torch.no_grad():
            x, y = self.task.get_batch(
                batch_size,
                split="eval",
                apply_input_corruption=noisy_inputs,
            )
            x, y = x.to(self.device), y.to(self.device)
            val_loss = self.task.loss(self.model, (x, y)).item()
            val_acc = self.task.evaluate(self.model, (x, y))
            return val_loss, val_acc

    def _eval_both(self, batch_size: int) -> dict:
        val_clean_loss, val_clean_acc = self.eval_step(batch_size, noisy_inputs=False)
        val_noisy_loss, val_noisy_acc = self.eval_step(batch_size, noisy_inputs=True)
        # Keep val_* aliases for backwards compatibility, with noisy as primary.
        return {
            "val_clean_loss": val_clean_loss,
            "val_clean_acc": val_clean_acc,
            "val_noisy_loss": val_noisy_loss,
            "val_noisy_acc": val_noisy_acc,
            "val_loss": val_noisy_loss,
            "val_acc": val_noisy_acc,
        }

    def train(
        self,
        n_steps: int,
        batch_size: int = 64,
        eval_every: int = 200,
        early_stopping: bool = True,
        patience: int = 50,
        no_hope_threshold: float = 0.01,
        no_hope_after_evals: int = 50,
        no_hope_first_eval_at: int | None = None,
        warmup_steps: int = 0,
    ):
        """
        Train for up to n_steps. Returns (history, early_stopped, stopped_at_step).
        Early stopping monitors noisy eval accuracy (primary robustness metric).
        warmup_steps: if > 0, linearly ramps the optimizer's LR from 0 to its base
        value over the first `warmup_steps` steps, then holds it constant.
        """
        self.history = []
        self._scheduler = None
        if warmup_steps > 0:
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
            )
        early_stopped = False
        stopped_at_step = n_steps
        best_val_acc = -1.0
        best_state = None
        evals_without_improvement = 10
        min_steps = 20

        for step in range(n_steps):
            train_loss, train_acc = self.train_step(batch_size)
            current_step = step + 1

            if (
                early_stopping
                and no_hope_first_eval_at is not None
                and no_hope_first_eval_at > 0
                and current_step == no_hope_first_eval_at
                and current_step % eval_every != 0
            ):
                eval_metrics = self._eval_both(batch_size)
                self.history.append(
                    {
                        "step": current_step,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        **eval_metrics,
                    }
                )
                print(
                    f"Step {current_step}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_clean_loss={eval_metrics['val_clean_loss']:.4f}, val_clean_acc={eval_metrics['val_clean_acc']:.4f}, "
                    f"val_noisy_loss={eval_metrics['val_noisy_loss']:.4f}, val_noisy_acc={eval_metrics['val_noisy_acc']:.4f}"
                )
                if no_hope_threshold > 0 and eval_metrics["val_acc"] < no_hope_threshold:
                    early_stopped = True
                    stopped_at_step = current_step
                    print(
                        f"  Early stop (no hope at step {current_step}: "
                        f"val_noisy_acc={eval_metrics['val_acc']:.4f} < {no_hope_threshold})"
                    )
                    break

            if current_step % eval_every == 0:
                eval_metrics = self._eval_both(batch_size)
                self.history.append(
                    {
                        "step": current_step,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        **eval_metrics,
                    }
                )
                print(
                    f"Step {current_step}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_clean_loss={eval_metrics['val_clean_loss']:.4f}, val_clean_acc={eval_metrics['val_clean_acc']:.4f}, "
                    f"val_noisy_loss={eval_metrics['val_noisy_loss']:.4f}, val_noisy_acc={eval_metrics['val_noisy_acc']:.4f}"
                )

                if early_stopping:
                    num_evals = len(self.history)
                    if (
                        no_hope_threshold > 0
                        and num_evals >= no_hope_after_evals
                        and eval_metrics["val_acc"] < no_hope_threshold
                    ):
                        early_stopped = True
                        stopped_at_step = current_step
                        print(
                            f"  Early stop (no hope: val_noisy_acc={eval_metrics['val_acc']:.4f} "
                            f"< {no_hope_threshold} after {num_evals} evals)"
                        )
                        break
                    if eval_metrics["val_acc"] > best_val_acc:
                        best_val_acc = eval_metrics["val_acc"]
                        evals_without_improvement = 0
                        best_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                    else:
                        evals_without_improvement += 1
                    if current_step >= min_steps and evals_without_improvement >= patience:
                        early_stopped = True
                        stopped_at_step = current_step
                        if best_state is not None:
                            self.model.load_state_dict(best_state, strict=True)
                            self.model.to(self.device)
                        print(
                            f"  Early stop (no improvement for {patience} evals, "
                            f"best noisy val_acc={best_val_acc:.4f})"
                        )
                        break

        if not early_stopped:
            if self.history and self.history[-1]["step"] != n_steps:
                eval_metrics = self._eval_both(batch_size)
                self.history.append(
                    {
                        "step": n_steps,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        **eval_metrics,
                    }
                )
                print(
                    f"Step {n_steps}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_clean_loss={eval_metrics['val_clean_loss']:.4f}, val_clean_acc={eval_metrics['val_clean_acc']:.4f}, "
                    f"val_noisy_loss={eval_metrics['val_noisy_loss']:.4f}, val_noisy_acc={eval_metrics['val_noisy_acc']:.4f}"
                )
            elif not self.history:
                eval_metrics = self._eval_both(batch_size)
                self.history.append(
                    {
                        "step": n_steps,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        **eval_metrics,
                    }
                )
                print(
                    f"Step {n_steps}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_clean_loss={eval_metrics['val_clean_loss']:.4f}, val_clean_acc={eval_metrics['val_clean_acc']:.4f}, "
                    f"val_noisy_loss={eval_metrics['val_noisy_loss']:.4f}, val_noisy_acc={eval_metrics['val_noisy_acc']:.4f}"
                )

        return self.history, early_stopped, stopped_at_step

    def show_examples(self, n_examples=5):
        """Show some examples with clean-eval input and predictions."""
        self.model.eval()
        with torch.no_grad():
            x, y = self.task.get_batch(
                n_examples,
                split="eval",
                apply_input_corruption=False,
            )
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            preds = logits.argmax(dim=-1)

            x = x.cpu()
            y = y.cpu()
            preds = preds.cpu()

            print("\nExamples:")
            for i in range(n_examples):
                print(f"\nExample {i+1}:")
                print(f"  Input:  {x[i].tolist()}")
                print(f"  Target: {y[i].tolist()}")
                print(f"  Pred:   {preds[i].tolist()}")
                correct = (preds[i] == y[i]).sum().item()
                print(f"  Correct: {correct}/{len(y[i])}")
