"""BB CLI: config-driven sweeps and checkpoint prediction."""

import argparse
import gc
import itertools
import math
import time
from pathlib import Path

import torch
import torch.multiprocessing as _tmp
import yaml
from filelock import FileLock

from bb.losses import cross_entropy, shift_tolerant_ce
from bb.manager.task_manager import TaskManager
from bb.models.bottleneck import SequenceBottleneckModel, resolve_bottleneck_dim
from bb.registry import (
    all_decoder_names,
    all_encoder_names,
    all_generator_names,
    all_model_param_keys,
    all_task_names,
    get_decoder,
    get_encoder,
    get_generator,
    get_task,
    param_defaults_from_models,
)
from bb.results import (
    append_results,
    append_run_detail,
    checkpoints_dir,
    default_results_path,
    get_next_run_id,
    load_existing_rows,
    overwrite_results,
    runs_detail_path,
)

LOSSES = {
    "cross_entropy": cross_entropy,
    "shift_tolerant_ce": shift_tolerant_ce,
}

SWEEP_PARAM_KEYS = [
    "encoder",
    "decoder",
    "task",
    "loss",
    "sequence_length",
    "vocabulary_size",
    "input_noise",
    "corruption_mode",
    "mask_token_id",
    "structural_regularity",
    "capacity_ratio",
    *sorted(all_model_param_keys()),
    "batch_size",
    "steps",
    "eval_every",
    "seed",
]

DEFAULTS = {
    "encoder": "lstm",
    "decoder": "lstm",
    "task": "sorting",
    "loss": "cross_entropy",
    "sequence_length": 32,
    "vocabulary_size": 64,
    "input_noise": 0.0,
    "corruption_mode": "replace",
    "mask_token_id": 0,
    "pad_token_id": None,
    "structural_regularity": 0.0,
    "d_model": 64,
    "bottleneck_dim": None,
    "capacity_ratio": None,
    "n_heads": 4,
    "batch_size": 64,
    "steps": 5000,
    "eval_every": 1000,
    "seed": None,
    **param_defaults_from_models(),
}


# ---------------------------------------------------------------------------
# Config loading & sweep expansion
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _expand_sweep(config: dict) -> list[dict]:
    """Expand list-valued params into a Cartesian product of scalar configs."""
    cfg = dict(config)
    if isinstance(cfg.get("sweep"), dict):
        base = {k: v for k, v in cfg.items() if k != "sweep"}
        cfg = {**base, **cfg["sweep"]}

    if isinstance(cfg.get("models"), list):
        shared = {k: v for k, v in cfg.items() if k != "models"}
        out = []
        for entry in cfg["models"]:
            if isinstance(entry, dict) and "encoder" in entry and "decoder" in entry:
                out.extend(_expand_sweep({**shared, **entry}))
        return out

    extra = {k: v for k, v in cfg.items() if k not in SWEEP_PARAM_KEYS}
    fixed, dims = {}, {}
    for key in SWEEP_PARAM_KEYS:
        value = cfg.get(key, DEFAULTS.get(key))
        if isinstance(value, list):
            dims[key] = value
        else:
            fixed[key] = value

    if not dims:
        return [{**extra, **fixed}]

    keys = list(dims)
    return [
        {**extra, **fixed, **dict(zip(keys, combo))}
        for combo in itertools.product(*(dims[k] for k in keys))
    ]


def _norm_sig_val(v):
    """Normalize a value for signature comparison: NaN→None, whole floats→int."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v == int(v):
            return int(v)
    return v


def _config_signature(cfg: dict) -> tuple:
    """Hashable identity of a run config (used to skip reruns)."""
    def _get(k):
        v = cfg.get(k)
        # bottleneck_dim is resolved at runtime when capacity_ratio is set;
        # normalize to None so raw configs match CSV rows.
        if k == "bottleneck_dim" and cfg.get("capacity_ratio") is not None:
            return None
        return _norm_sig_val(v)
    return tuple(_get(k) for k in SWEEP_PARAM_KEYS)


# ---------------------------------------------------------------------------
# Information-theoretic helpers
# ---------------------------------------------------------------------------

def compute_d_min(seq_len: int, vocab_size: int, structural_regularity: float) -> int:
    """Shannon lower bound on bottleneck dimension: ceil(H(τ) / 32).

    H(τ) = seq_len * (1 - structural_regularity) * log2(vocab_size) bits.
    Dividing by 32 converts bits to float32 dimensions.
    Returns at least 1.
    """
    if vocab_size <= 1:
        return 1
    entropy_bits = seq_len * (1.0 - structural_regularity) * math.log2(vocab_size)
    return max(1, math.ceil(entropy_bits / 32))


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _component_kwargs(info: dict, cfg: dict, injected: dict) -> dict:
    defaults = info.get("param_defaults", {})
    kwargs = dict(injected)
    for key in info.get("constructor_params", []):
        if key not in kwargs:
            kwargs[key] = cfg.get(key, defaults.get(key))
    return kwargs


def _build_task_and_model(cfg: dict):
    """Build task + bottleneck model from a config with defaults already merged."""
    loss_fn = LOSSES.get(cfg["loss"])
    if loss_fn is None:
        raise SystemExit(f"Unknown loss: {cfg['loss']}. Use: {', '.join(sorted(LOSSES))}.")

    encoder_info = get_encoder(cfg["encoder"])
    if encoder_info is None:
        raise SystemExit(f"Unknown encoder: {cfg['encoder']}. Use: {', '.join(sorted(all_encoder_names()))}.")

    decoder_info = get_decoder(cfg["decoder"])
    if decoder_info is None:
        raise SystemExit(f"Unknown decoder: {cfg['decoder']}. Use: {', '.join(sorted(all_decoder_names()))}.")

    task_info = get_task(cfg["task"])
    if task_info is None:
        raise SystemExit(f"Unknown task: {cfg['task']}. Use: {', '.join(sorted(all_task_names()))}.")

    source_name = cfg.get("source", "random")
    generator_info = get_generator(source_name)
    if generator_info is None:
        raise SystemExit(f"Unknown source: {source_name}. Use: {', '.join(sorted(all_generator_names()))}.")

    generator_injected = {
        "seq_len": cfg["sequence_length"],
        "vocab_size": cfg["vocabulary_size"],
    }
    if "structure" in generator_info.get("constructor_params", []):
        generator_injected["structure"] = cfg.get("structural_regularity", 0.0)

    generator = generator_info["cls"](**_component_kwargs(generator_info, cfg, generator_injected))
    task = task_info["cls"](generator, loss_fn=loss_fn)

    capacity_ratio = cfg.get("capacity_ratio")
    if capacity_ratio is not None:
        d_min = compute_d_min(
            cfg["sequence_length"],
            cfg["vocabulary_size"],
            cfg.get("structural_regularity", 0.0),
        )
        bottleneck_dim = min(cfg["d_model"], max(1, math.ceil(capacity_ratio * d_min)))
    else:
        bottleneck_dim = resolve_bottleneck_dim(cfg["d_model"], cfg.get("bottleneck_dim"))

    encoder = encoder_info["cls"](**_component_kwargs(encoder_info, cfg, {
        "vocab_size": cfg["vocabulary_size"],
        "seq_len": cfg["sequence_length"],
        "d_model": cfg["d_model"],
        "bottleneck_dim": bottleneck_dim,
    }))

    decoder = decoder_info["cls"](**_component_kwargs(decoder_info, cfg, {
        "z_dim": encoder.out_dim,
        "seq_len": cfg["sequence_length"],
        "d_model": cfg["d_model"],
        "vocab_size": cfg["vocabulary_size"],
    }))

    return task, SequenceBottleneckModel(encoder, decoder), bottleneck_dim


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _get_device(name: str) -> torch.device:
    if name == "auto":
        use_cuda = torch.cuda.is_available()
        dev = torch.device("cuda" if use_cuda else "cpu")
        print(f"Using device: {'cuda (GPU)' if use_cuda else 'cpu (GPU not available)'}")
        return dev
    print(f"Using device: {name}")
    return torch.device(name)


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _final_metrics(history: list, early_stopped: bool) -> dict:
    if not history:
        return {f"final_{k}": None for k in [
            "train_loss", "train_acc",
            "val_clean_loss", "val_clean_acc",
            "val_noisy_loss", "val_noisy_acc",
        ]}
    record = max(history, key=lambda h: h["val_noisy_acc"]) if early_stopped else history[-1]
    return {f"final_{k}": record[k] for k in [
        "train_loss", "train_acc",
        "val_clean_loss", "val_clean_acc",
        "val_noisy_loss", "val_noisy_acc",
    ]}


def _model_label(cfg: dict, bottleneck_dim: int) -> str:
    parts = [f"d_model={cfg['d_model']}", f"bottleneck_dim={bottleneck_dim}"]
    if cfg.get("n_layers") is not None:
        parts.append(f"n_layers={cfg['n_layers']}")
    uses_transformer = cfg["encoder"] == "transformer" or cfg["decoder"] == "transformer"
    if uses_transformer and cfg.get("n_heads") is not None:
        parts.append(f"n_heads={cfg['n_heads']}")
    return f"{cfg['encoder']}->{cfg['decoder']}({', '.join(parts)})"


def _save_checkpoint(model, cfg: dict, run_id: int) -> Path:
    ckpt_dir = checkpoints_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{cfg['task']}_{cfg['encoder']}_{cfg['decoder']}_{run_id}.pt"
    torch.save({"model_state_dict": model.state_dict(), "run_config": cfg}, path)
    return path


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def _run_one(cfg: dict, device: torch.device, run_id: int,
             save_model: bool = False, log_prefix: str = "") -> tuple[dict, list, Path | None]:
    """Train one config. Returns (result_row, history, checkpoint_path)."""
    if cfg.get("seed") is not None:
        torch.manual_seed(cfg["seed"])

    task, model, bottleneck_dim = _build_task_and_model(cfg)
    d_min = compute_d_min(
        cfg["sequence_length"],
        cfg["vocabulary_size"],
        cfg.get("structural_regularity", 0.0),
    )
    model = model.to(device)
    param_count = _count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = TaskManager(task=task, model=model, optimizer=optimizer, device=device)

    t0 = time.perf_counter()
    history, early_stopped, stopped_at_step = manager.train(
        n_steps=cfg["steps"],
        batch_size=cfg["batch_size"],
        eval_every=cfg["eval_every"],
        early_stopping=cfg.get("early_stopping", True),
        patience=cfg.get("patience", 3),
        no_hope_threshold=cfg.get("no_hope_threshold", 0.01),
        no_hope_after_evals=cfg.get("no_hope_after_evals", 3),
        no_hope_first_eval_at=cfg.get("no_hope_first_eval_at"),
        warmup_steps=cfg.get("warmup_steps", 0),
    )
    train_time_s = round(time.perf_counter() - t0, 2)

    row = {
        "run_id": run_id,
        "model": _model_label(cfg, bottleneck_dim),
        "encoder": cfg["encoder"],
        "decoder": cfg["decoder"],
        "task": cfg["task"],
        "loss": cfg["loss"],
        "sequence_length": cfg["sequence_length"],
        "vocabulary_size": cfg["vocabulary_size"],
        "input_noise": cfg.get("input_noise", 0.0),
        "corruption_mode": cfg.get("corruption_mode", "replace"),
        "mask_token_id": cfg.get("mask_token_id", 0),
        "pad_token_id": cfg.get("pad_token_id"),
        "structural_regularity": cfg.get("structural_regularity", 0.0),
        "d_model": cfg["d_model"],
        "bottleneck_dim": bottleneck_dim,
        "d_min": d_min,
        "capacity_ratio": cfg.get("capacity_ratio"),
        "n_layers": cfg.get("n_layers"),
        "n_heads": cfg.get("n_heads"),
        "param_count": param_count,
        "batch_size": cfg["batch_size"],
        "steps": cfg["steps"],
        "eval_every": cfg["eval_every"],
        "seed": cfg.get("seed"),
        "train_time_s": train_time_s,
        "early_stop": early_stopped,
        "stopped_at_step": stopped_at_step,
        **_final_metrics(history, early_stopped),
    }

    ckpt_path = None
    if save_model:
        ckpt_path = _save_checkpoint(model, cfg, run_id)
        print(f"{log_prefix}Saved checkpoint to {ckpt_path}")

    del model, optimizer, task, manager
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return row, history, ckpt_path


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _write_result(cfg, row, history, ckpt_path, out_path):
    """Append a completed run to CSV + detail file under a file lock."""
    lock = FileLock(str(out_path) + ".lock")
    with lock:
        _, current_explored = load_existing_rows(out_path, _config_signature)
        if _config_signature(cfg) not in current_explored:
            append_results([row], out_path)
            append_run_detail({
                "run_id": row["run_id"],
                "run_config": cfg,
                "history": history,
                "summary": row,
                "checkpoint": str(ckpt_path.resolve()) if ckpt_path else None,
            }, out_path)


def _worker_run_configs(args_tuple):
    """Worker entry point (module-level for multiprocessing pickling)."""
    configs_with_ids, out_path_str, device_str, save_model, worker_id = args_tuple
    device = _get_device(device_str)
    out_path = Path(out_path_str)
    for run_id, cfg in configs_with_ids:
        print(
            f"  [W{worker_id}] encoder={cfg['encoder']} decoder={cfg['decoder']} "
            f"task={cfg['task']} seq_len={cfg['sequence_length']} "
            f"vocab={cfg['vocabulary_size']} run_id={run_id}"
        )
        row, history, ckpt_path = _run_one(
            cfg, device, run_id,
            save_model=save_model,
            log_prefix=f"    [W{worker_id}] ",
        )
        _write_result(cfg, row, history, ckpt_path, out_path)


def cmd_run(args):
    configs = []
    for path in args.config:
        configs.extend(_expand_sweep(_load_config(path)))

    out_path = Path(args.output) if args.output else default_results_path()
    workers = args.workers

    existing_rows, explored = [], set()
    if not args.overwrite and out_path.exists():
        existing_rows, explored = load_existing_rows(out_path, _config_signature)
        print(f"Found {len(existing_rows)} existing result(s) in {out_path}")

    to_run = [c for c in configs if _config_signature(c) not in explored]
    skipped = len(configs) - len(to_run)
    if skipped:
        print(f"Skipping {skipped} already-explored configuration(s)")
    if not to_run:
        print("No new configurations to run.")
        return

    detail_path = runs_detail_path(out_path)
    # runs_detail.jsonl is a permanent append-only log — never cleared.
    base_id = get_next_run_id(out_path)

    # With --overwrite, reset the summary CSV before any workers start writing.
    if args.overwrite:
        overwrite_results([], out_path)

    print(f"Running {len(to_run)} configuration(s) with {workers} worker(s)")
    configs_with_ids = [(base_id + i, cfg) for i, cfg in enumerate(to_run)]

    if workers == 1:
        device = _get_device(args.device)
        for run_id, cfg in configs_with_ids:
            print(
                f"  [{run_id - base_id + 1}/{len(to_run)}] encoder={cfg['encoder']} "
                f"decoder={cfg['decoder']} task={cfg['task']} "
                f"seq_len={cfg['sequence_length']} vocab={cfg['vocabulary_size']} "
                f"input_noise={cfg['input_noise']}"
            )
            row, history, ckpt_path = _run_one(
                cfg, device, run_id,
                save_model=not args.no_save_model,
                log_prefix="    ",
            )
            _write_result(cfg, row, history, ckpt_path, out_path)
    else:
        # Partition configs round-robin across workers to avoid overlap.
        chunks = [configs_with_ids[i::workers] for i in range(workers)]
        chunk_args = [
            (chunk, str(out_path), args.device, not args.no_save_model, w)
            for w, chunk in enumerate(chunks)
        ]
        ctx = _tmp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            pool.map(_worker_run_configs, chunk_args)

    print(f"Done: {len(to_run)} run(s) -> {out_path} + {detail_path}")


def cmd_predict(args):
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = _get_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["run_config"]

    task, model, _ = _build_task_and_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    manager = TaskManager(task=task, model=model, optimizer=None, device=device)

    print(f"Checkpoint : {ckpt_path}")
    print(f"Encoder    : {cfg['encoder']}")
    print(f"Decoder    : {cfg['decoder']}")
    print(f"Task       : {cfg['task']}")
    print(f"Seq len    : {cfg['sequence_length']}")
    print(f"Vocab size : {cfg['vocabulary_size']}")

    n = args.n_examples
    clean_loss, clean_acc = manager.eval_step(batch_size=n, noisy_inputs=False)
    noisy_loss, noisy_acc = manager.eval_step(batch_size=n, noisy_inputs=True)
    print(f"\nEval clean ({n} samples): loss={clean_loss:.4f}  acc={clean_acc:.4f}")
    print(f"Eval noisy ({n} samples): loss={noisy_loss:.4f}  acc={noisy_acc:.4f}")
    manager.show_examples(n_examples=n)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        prog="bb",
        description="Bottleneck Benchmark (BB): encoder/decoder sweeps under a fixed latent bottleneck.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one or more YAML configs (list-valued fields are swept).")
    run_p.add_argument("-c", "--config", required=True, nargs="+", metavar="YAML")
    run_p.add_argument("-o", "--output", default=None, help="Results CSV path")
    run_p.add_argument("--overwrite", action="store_true", help="Replace existing results file")
    run_p.add_argument("--no-save-model", action="store_true", help="Skip checkpoint writing")
    run_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    run_p.add_argument("--workers", type=int, default=1, metavar="N",
                       help="Number of parallel training processes (default 1). "
                            "Each worker uses the same GPU; configs are split round-robin.")
    run_p.set_defaults(func=cmd_run)

    pred_p = sub.add_parser("predict", help="Load a checkpoint and show predictions.")
    pred_p.add_argument("checkpoint", help="Path to .pt checkpoint")
    pred_p.add_argument("-n", "--n-examples", type=int, default=5)
    pred_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    pred_p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
