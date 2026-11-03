"""
Forgetting map analysis: per-position accuracy for each checkpoint of a forgetting-map sweep.

For each (encoder, d_b, seed), loads the checkpoint, runs a large eval batch,
and records per-position accuracy.

Usage:
  uv run bb run -c configs/paper/forgetting_map.yaml -o data/results/forgetting_runs.csv   # keeps checkpoints
  uv run python scripts/paper/forgetting_map.py -i data/results/forgetting_runs.csv -o data/results/forgetting.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from bb.cli import _build_task_and_model
from bb.results import checkpoints_dir

CHECKPOINT_DIR = checkpoints_dir()
EVAL_BATCHES   = 20   # 20 x 128 = 2560 examples per checkpoint
BATCH_SIZE     = 128
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_position_accuracy(model, task, batch_size: int, n_batches: int, noisy: bool) -> list[float]:
    model.eval()
    L = task.generator.seq_len
    correct = torch.zeros(L, device=DEVICE)
    total   = torch.zeros(L, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = task.get_batch(batch_size, split="eval", apply_input_corruption=noisy)
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)                          # [B, L, V]
            preds  = logits.argmax(dim=-1)             # [B, L]
            correct += (preds == y).float().sum(dim=0) # [L]
            total   += batch_size
    return (correct / total).cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True, help="results CSV of the forgetting-map sweep")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    RESULTS_CSV, OUT_CSV = args.input, args.output

    if not RESULTS_CSV.exists():
        raise SystemExit(f"Results file not found: {RESULTS_CSV}. Run the sweep first.")

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} runs from {RESULTS_CSV}")

    records = []
    for _, row in df.iterrows():
        ckpt_path = CHECKPOINT_DIR / f"copy_{row['encoder']}_lstm_{int(row['run_id'])}.pt"
        if not ckpt_path.exists():
            print(f"  Missing checkpoint: {ckpt_path.name} — skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        cfg  = ckpt["run_config"]
        task, model, _ = _build_task_and_model(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)

        clean_acc = per_position_accuracy(model, task, BATCH_SIZE, EVAL_BATCHES, noisy=False)
        noisy_acc = per_position_accuracy(model, task, BATCH_SIZE, EVAL_BATCHES, noisy=True)

        for pos, (ca, na) in enumerate(zip(clean_acc, noisy_acc)):
            records.append({
                "run_id":        row["run_id"],
                "encoder":       row["encoder"],
                "bottleneck_dim": row["bottleneck_dim"],
                "seed":          row["seed"],
                "mean_val_noisy_acc": row["final_val_noisy_acc"],
                "position":      pos,
                "clean_acc":     ca,
                "noisy_acc":     na,
            })

        print(f"  {row['encoder']:12s} d_b={int(row['bottleneck_dim']):3d} seed={row['seed']}  "
              f"noisy_mean={row['final_val_noisy_acc']:.3f}  "
              f"pos_range=[{min(noisy_acc):.3f}, {max(noisy_acc):.3f}]")

        del model, task

    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out)} position-level rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
