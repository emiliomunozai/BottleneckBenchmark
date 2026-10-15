![ci](https://github.com/emiliomunozai/BottleneckBenchmark/actions/workflows/ci.yml/badge.svg)

# Bottleneck Benchmark (BB)

Factor Bottleneck Benchmark for sequence models under a fixed information budget. This repository ships the reference code as the Python package **`bb`** (sources under `src/bb/`).

## Overview

Every model must compress a sequence into a fixed bottleneck vector before reconstruction.

- An **encoder** maps tokens `[B, L]` to a latent `[B, bottleneck_dim]`.
- A **decoder** reconstructs logits `[B, L, V]` from that latent.

Encoders and decoders are independent — any encoder can pair with any decoder.

Active families:

- Encoders: `bi_lstm`, `bi_ssm`, `transformer`
- Decoders: `lstm`, `ssm`, `transformer` (the SSM decoder is supported but weak; sweeps use `bi_ssm` + `lstm` as the SSM track)

The transformer encoder uses a learned bottleneck token and the decoder expands it back into position-indexed memory slots, while still keeping the single latent vector contract.

## Setup

Python 3.12+. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (project workflows use `uv` only, not `pip`).

```powershell
cd <path-to-this-repo>
uv sync --group dev --group cpu   # or --group gpu for CUDA
```

## CLI

The Hatch entry point is **`bb`** (`pyproject.toml` → `bb.cli:main`). Typical invocations:

```powershell
uv run bb run -c configs/example_single.yaml
uv run bb predict data/checkpoints/copy_bi_lstm_lstm_0.pt -n 5
```

`bb run` takes one or more YAML configs. List-valued fields are swept (Cartesian product). There is no separate sweep command.

Use `uv run bb --help`, `uv run bb run --help`, and `uv run bb predict --help` for full flag lists.

### Quick test run (no output files)

To run a quick test that will not save any results or checkpoints, use the `--no-save-model` flag and point the output to a dummy location:

```powershell
uv run bb run -c configs/example_single.yaml --no-save-model -o nul
```
- On Linux or Mac, replace `-o nul` with `-o /dev/null`:
  ```sh
  uv run bb run -c configs/example_single.yaml --no-save-model -o /dev/null
  ```
This will train and print results to the screen, but skip writing results or models.

### Repository documentation

| Path | Purpose |
|------|---------|
| `docs/pending.md` | Task board (writing, code, experiments, submission) |
| `docs/logs.md` | Macro decisions and conclusions across runs |
| `docs/experiments.md` | Per-run tables and experiment-series notes |
| `docs/main.tex` | Paper draft (NeurIPS 2026 style) |

Options:

```
-c, --config YAML [YAML ...]   Config file(s)
-o, --output PATH              Results CSV (default: data/results/results.csv)
--overwrite                    Replace existing results
--no-save-model                Skip checkpoint writing
--device {auto,cpu,cuda}
```

## Config patterns

### Single model

```yaml
task: copy
loss: cross_entropy

encoder: bi_lstm
decoder: lstm

sequence_length: 32
vocabulary_size: 64
input_noise: 0.0
structural_regularity: 0.0
corruption_mode: replace

d_model: 128
bottleneck_dim: 32
n_layers: 2
n_heads: 4

batch_size: 64
steps: 5000
eval_every: 500
seed: 42
```

### Multiple explicit models

Use `models:` to define specific encoder-decoder pairs instead of sweeping all combinations:

```yaml
task: copy
loss: cross_entropy
sequence_length: [64, 128, 256]
vocabulary_size: [64, 128, 256]
input_noise: [0.0, 0.1]
corruption_mode: replace

models:
  - encoder: bi_lstm
    decoder: lstm
    n_layers: 2

  - encoder: transformer
    decoder: lstm
    n_layers: 2
    n_heads: 4

  - encoder: bi_ssm
    decoder: lstm
    n_layers: 2

d_model: 128
bottleneck_dim: 32
batch_size: 128
steps: 5000
eval_every: 1000
seed: 42
```

List-valued fields (`sequence_length`, `vocabulary_size`, `input_noise` above) are crossed with each model pair.

### Defaults

Any key omitted from a config falls back to the built-in defaults. `bottleneck_dim` defaults to `d_model` when not specified.

## Output

Each run writes:

- `data/results/results.csv` — one summary row per run
- `data/results/runs_detail.jsonl` — full config, step history, and checkpoint path per run
- `data/checkpoints/` — model weights (`.pt`)

Runs already present in the CSV (matched by config signature) are skipped automatically. Use `--overwrite` to start fresh.

## Validation

Two eval modes are tracked at each checkpoint:

- **clean**: clean input → clean target
- **noisy**: corrupted input → clean target

The noisy score is the primary robustness metric and drives early stopping.

## Extending

Add encoders in `src/bb/models/encoders/`, decoders in `src/bb/models/decoders/`. Register with the `@register_encoder` / `@register_decoder` decorators.

Contract:
- Encoders return `EncoderOutput(z=...)` with shape `[B, bottleneck_dim]`
- Decoders map that latent to logits `[B, L, V]`

## Project description

### Motivation

Comparing sequence model architectures is difficult when models differ simultaneously in depth, width, parameter count, and representational capacity. Performance differences may reflect architectural merit or simply differences in information throughput. BB isolates architecture as the experimental variable by forcing every model through a fixed-dimensional bottleneck, equalizing the information budget available for sequence reconstruction.

### Formulation

Let \( x \in \{0, \dots, V{-}1\}^L \) be an input sequence of length \( L \) over a vocabulary of size \( V \). The benchmark decomposes every model into two stages:

1. **Encoder** \( f_\theta : \mathbb{Z}^{L} \to \mathbb{R}^{d} \). Maps the discrete input to a single latent vector \( z = f_\theta(x) \) of fixed dimension \( d \) (the bottleneck dimension), regardless of sequence length.
2. **Decoder** \( g_\phi : \mathbb{R}^{d} \to \mathbb{R}^{L \times V} \). Reconstructs token-level logits from \( z \) alone, with no residual connection or skip path from the encoder.

The full model is their composition \( g_\phi(f_\theta(x)) \), trained end-to-end with a token-level loss.

### Tasks

Each task defines a deterministic mapping from the clean input \( x \) to a target sequence \( y \):

| Task | Target | What it measures |
|------|--------|------------------|
| **Copy** | \( y = x \) | Whether the bottleneck can preserve token identity and order |
| **Sorting** | \( y = \text{sort}(x) \) | Whether the bottleneck captures distributional content independent of position |
| **Set** | \( y = \text{sort}(\text{unique}(x)) \), right-padded with a sentinel | Whether the bottleneck captures the support of \( x \) (which tokens appear) without their counts or order |

Targets are always computed from the clean (uncorrupted) input.

### Structural regularity

Input sequences are generated with a controllable Markov self-transition bias governed by `structural_regularity = s ∈ [0, 1]`. The first token is sampled uniformly from \( \{0, \dots, V{-}1\} \). Each subsequent token either repeats the previous token (with probability \( s \)) or is sampled uniformly (with probability \( 1-s \)):

```
P(x[t] = j | x[t-1] = i) = s · 1[j == i]  +  (1-s) · (1/V)
```

This reduces per-token conditional entropy to \( (1-s) \cdot \log_2 V \) bits. At `s=0.0` (the default) sequences are fully i.i.d. uniform, recovering the original behavior. At `s=1.0` every token repeats the first, giving maximum regularity. Intermediate values produce runs of identical tokens whose expected length is \( 1/(1-s) \).

### Input corruption

To probe robustness, the encoder input may be corrupted at a configurable noise rate \( \eta \in [0, 1] \). Each token is independently corrupted with probability \( \eta \). Two corruption modes are supported:

- **Replace**: the token is replaced by a uniformly random token guaranteed to differ from the original.
- **Mask**: the token is replaced by a fixed mask token ID.

Corruption is applied to the encoder input only; the target \( y \) is always derived from the clean sequence. During training, corruption is applied by default. At evaluation time, two scores are reported: clean (no corruption) and noisy (corruption applied), allowing separation of modeling capacity from noise robustness.

### Loss functions

- **Cross-entropy**: standard token-level cross-entropy over the vocabulary, averaged across positions and batch.
- **Shift-tolerant cross-entropy**: at each position \( t \), the one-hot target is replaced by a soft distribution blending targets from neighboring positions \( [t{-}w, t{+}w] \), weighted by a Gaussian kernel with standard deviation \( \sigma = w/2 \). This relaxes the positional alignment requirement: a copy shifted by one position incurs a small penalty rather than full cross-entropy loss, while a perfectly aligned prediction still achieves the global optimum.

### Experimental protocol

A sweep configuration specifies sets of values for each hyperparameter (encoder, decoder, task, sequence length, vocabulary size, noise rate, etc.). The benchmark takes the Cartesian product of all list-valued fields and trains one model per combination. For each run:

1. A random seed is fixed (when provided) for reproducibility.
2. The encoder and decoder are instantiated with shared architectural parameters (\( d_\text{model} \), \( d \), number of layers, number of attention heads where applicable).
3. The model is trained with Adam (lr = 1e-3) for a fixed number of steps.
4. At regular intervals, clean and noisy validation accuracy are computed. Early stopping monitors noisy validation accuracy with a patience window and a no-hope threshold (minimum accuracy after a configurable number of evaluations).
5. The final reported metrics are taken from the last evaluation step, or from the best noisy validation checkpoint if early stopping triggered.

### Measured quantities

Each run produces a summary row containing:

- Architecture identifiers (encoder, decoder, task, loss)
- Data parameters (sequence length, vocabulary size, noise rate, corruption mode)
- Model parameters (d_model, bottleneck_dim, n_layers, n_heads, trainable parameter count)
- Training parameters (batch size, steps, eval frequency, seed)
- Final metrics: train loss/accuracy, clean validation loss/accuracy, noisy validation loss/accuracy
- Training time, early stopping status, and step at which training stopped

### Controlled variables

BB exposes four independently controllable difficulty factors:

| Factor | Config key | Effect |
|--------|-----------|--------|
| **Scale** | `sequence_length` | More tokens → more total information to compress |
| **Breadth** | `vocabulary_size` | Larger vocab → more bits per token |
| **Noise** | `input_noise` | Corrupted tokens → encoder must denoise before compressing |
| **Structural regularity** | `structural_regularity` | Higher value → more token repetition → lower effective entropy, more compressible |

The bottleneck dimension `bottleneck_dim` is the primary architectural variable. By fixing it across architectures the benchmark equalizes the information budget, so accuracy differences reflect compression efficiency rather than raw model width.
