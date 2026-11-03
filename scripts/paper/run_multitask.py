"""
Multi-task compression sweep: one encoder, two decoders (copy + sort) sharing one bottleneck z.

For each (encoder, d_b, seed), trains with combined copy+sort loss and reports
both accuracies. d_b^min_joint = smallest d_b where copy_acc >= 0.8 AND sort_acc >= 0.8.

Usage: uv run python scripts/paper/run_multitask.py -c configs/paper/multitask_copy_sort.yaml [-o results.csv]
"""

import argparse
import gc
import itertools
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from bb.cli import _component_kwargs
from bb.generators.random import RandomSequenceGenerator
from bb.losses import cross_entropy
from bb.models.multitask import MultiTaskBottleneckModel
from bb.registry import get_decoder, get_encoder

THRESHOLD = 0.8


def build_model(cfg: dict, bottleneck_dim: int):
    enc_info = get_encoder(cfg["encoder"])
    dec_info = get_decoder("lstm")

    gen = RandomSequenceGenerator(
        seq_len=cfg["sequence_length"],
        vocab_size=cfg["vocabulary_size"],
        input_noise=cfg.get("input_noise", 0.0),
        corruption_mode="replace",
        structure=cfg.get("structural_regularity", 0.0),
    )

    injected_enc = dict(vocab_size=cfg["vocabulary_size"], seq_len=cfg["sequence_length"],
                        d_model=cfg["d_model"], bottleneck_dim=bottleneck_dim)
    enc = enc_info["cls"](**_component_kwargs(enc_info, cfg, injected_enc))

    injected_dec = dict(z_dim=enc.out_dim, seq_len=cfg["sequence_length"],
                        d_model=cfg["d_model"], vocab_size=cfg["vocabulary_size"])
    dec_kwargs = _component_kwargs(dec_info, cfg, injected_dec)
    copy_dec = dec_info["cls"](**dec_kwargs)
    sort_dec = dec_info["cls"](**dec_kwargs)
    return gen, MultiTaskBottleneckModel(enc, copy_dec, sort_dec)


def run_one(cfg: dict, bottleneck_dim: int, seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    gen, model = build_model(cfg, bottleneck_dim)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_joint = 0.0
    best_copy = best_sort = 0.0
    no_hope_evals = 0
    t0 = time.perf_counter()

    for step in range(1, cfg["steps"] + 1):
        model.train()
        # sample x_clean, build copy and sort targets from same clean sequence
        x_clean = gen.sample(cfg["batch_size"]).to(device)
        y_copy  = x_clean.clone()
        y_sort  = torch.sort(x_clean, dim=1).values
        x_noisy = gen.apply_input_corruption(x_clean, "train").to(device)

        copy_logits, sort_logits = model(x_noisy)
        loss = cross_entropy(copy_logits, y_copy) + cross_entropy(sort_logits, y_sort)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                xc = gen.sample(cfg["batch_size"]).to(device)
                yc, ys = xc.clone(), torch.sort(xc, dim=1).values
                cl, sl = model(xc)
                copy_acc = (cl.argmax(-1) == yc).float().mean().item()
                sort_acc = (sl.argmax(-1) == ys).float().mean().item()
                joint    = min(copy_acc, sort_acc)
                if joint > best_joint:
                    best_joint, best_copy, best_sort = joint, copy_acc, sort_acc
                if joint < 0.01:
                    no_hope_evals += 1
                    if no_hope_evals >= cfg.get("no_hope_after_evals", 100):
                        break
                else:
                    no_hope_evals = 0

    return {
        "encoder": cfg["encoder"], "bottleneck_dim": bottleneck_dim, "seed": seed,
        "sequence_length": cfg["sequence_length"], "vocabulary_size": cfg["vocabulary_size"],
        "input_noise": cfg.get("input_noise", 0.0),
        "structural_regularity": cfg.get("structural_regularity", 0.0),
        "d_model": cfg["d_model"],
        "final_copy_acc": best_copy, "final_sort_acc": best_sort,
        "joint_success": int(best_copy >= THRESHOLD and best_sort >= THRESHOLD),
        "train_time_s": round(time.perf_counter() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", type=Path, default=Path("data/results/multitask.csv"))
    args = parser.parse_args()
    OUT_PATH = args.output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoders   = cfg["encoder"] if isinstance(cfg["encoder"], list) else [cfg["encoder"]]
    dbs        = cfg["bottleneck_dim"] if isinstance(cfg["bottleneck_dim"], list) else [cfg["bottleneck_dim"]]
    seeds      = cfg["seed"] if isinstance(cfg["seed"], list) else [cfg["seed"]]
    combos     = list(itertools.product(encoders, dbs, seeds))

    existing = set()
    if OUT_PATH.exists():
        done = pd.read_csv(OUT_PATH)
        for _, r in done.iterrows():
            existing.add((r["encoder"], int(r["bottleneck_dim"]), int(r["seed"])))
        print(f"Skipping {len(existing)} already-done runs")

    to_run = [(enc, db, s) for enc, db, s in combos if (enc, db, s) not in existing]
    print(f"Running {len(to_run)}/{len(combos)} configurations")

    for i, (enc, db, seed) in enumerate(to_run):
        run_cfg = {**cfg, "encoder": enc}
        print(f"  [{i+1}/{len(to_run)}] {enc} d_b={db} seed={seed}", flush=True)
        row = run_one(run_cfg, db, seed, device)
        row_df = pd.DataFrame([row])
        row_df.to_csv(OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False)
        print(f"    copy={row['final_copy_acc']:.3f}  sort={row['final_sort_acc']:.3f}"
              f"  joint={'YES' if row['joint_success'] else 'no'}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
