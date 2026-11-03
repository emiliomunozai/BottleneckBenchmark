"""Rebuild the paper's tables from logged runs and check them against the published values.

Usage:
  uv run python scripts/paper/tables.py table2                  # print one table
  uv run python scripts/paper/tables.py all                     # print every table
  uv run python scripts/paper/tables.py verify                  # compare every cell with the paper
  uv run python scripts/paper/tables.py verify --results DIR    # same, on your own rerun
  uv run python scripts/paper/tables.py seeds > seeds.csv       # per-cell seed statistics (appendix)

Every table reads CSVs from --results (default: results/paper), one file per experiment:
  sweeps.csv       Tables 1-2   (bb run: configs/paper/{set,sort}_sweep_L.yaml, copy_sweep_*.yaml)
  decoders.csv     Table 3      (bb run: configs/paper/decoder_sweep_L.yaml)
  forgetting.csv   Table 4      (scripts/paper/forgetting_map.py)
  multitask.csv    Table 5      (scripts/paper/run_multitask.py)
  interaction.csv  Table 6      (bb run: configs/paper/factor_interaction.yaml)
  warmup.csv       Appendix     (bb run: configs/paper/warmup_*.yaml)
  extra_seeds.csv  Appendix     (bb run: configs/paper/extra_seeds_*.yaml)
  palindrome.csv   Appendix     (bb run: configs/paper/palindrome_copy.yaml)

d_b^min is the smallest tested d_b whose mean noisy-validation accuracy over seeds reaches THETA.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

THETA = 0.8
ARCHS = ["bi_lstm", "transformer", "bi_ssm"]
LABEL = {"bi_lstm": "BiLSTM", "transformer": "Transformer", "bi_ssm": "BiSSM",
         "lstm": "LSTM", "ssm": "SSM"}
FAIL = "---"

# Configuration signature: two runs with the same signature are repeats; the latest is kept.
SIGNATURE = ["task", "encoder", "decoder", "sequence_length", "vocabulary_size", "input_noise",
             "structural_regularity", "bottleneck_dim", "seed"]

BASELINE = dict(sequence_length=16, vocabulary_size=8, structural_regularity=0.0, input_noise=0.0)


def load(results: Path, name: str) -> pd.DataFrame:
    """Grid runs only (capacity_ratio set; off-grid probes are excluded), latest run per signature."""
    df = pd.read_csv(results / name).dropna(subset=["capacity_ratio"])
    return (df.sort_values("run_id")
              .drop_duplicates(subset=[c for c in SIGNATURE if c in df.columns], keep="last")
              .reset_index(drop=True))


def select(df: pd.DataFrame, **fixed) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, val in fixed.items():
        mask &= df[col] == val
    return df[mask]


def dbmin(grp: pd.DataFrame, acc: str = "final_val_noisy_acc") -> str:
    """Smallest d_b whose seed-mean accuracy reaches THETA, FAIL if none does, n/a if no runs."""
    if grp.empty:
        return "n/a"
    means = grp.groupby("bottleneck_dim")[acc].mean().sort_index()
    passing = means[means >= THETA]
    return str(int(passing.index[0])) if len(passing) else FAIL


# ---------------------------------------------------------------- tables

def table1(results: Path) -> pd.DataFrame:
    """Task anchor: d_b^min by task at the noiseless baseline (LSTM decoder)."""
    df = select(load(results, "sweeps.csv"), decoder="lstm", **BASELINE)
    rows = []
    for task, label in [("set", "Set"), ("sorting", "Sort"), ("copy", "Copy")]:
        rows.append({"Task": label, **{LABEL[a]: dbmin(select(df, task=task, encoder=a))
                                       for a in ARCHS}})
    return pd.DataFrame(rows)


T2_SWEEPS = [("L", "sequence_length", [8, 16, 32, 64, 96]),
             ("V", "vocabulary_size", [4, 8, 16, 32, 64]),
             ("s", "structural_regularity", [0.0, 0.25, 0.5, 0.75]),
             ("eta", "input_noise", [0.0, 0.1, 0.2, 0.3])]


def table2(results: Path) -> pd.DataFrame:
    """Copy task, one factor varied at a time from the baseline (LSTM decoder)."""
    df = select(load(results, "sweeps.csv"), task="copy", decoder="lstm")
    rows = []
    for label, col, levels in T2_SWEEPS:
        for level in levels:
            fixed = {**BASELINE, col: level}
            rows.append({"Sweep": label, "Level": level,
                         **{LABEL[a]: dbmin(select(df, encoder=a, **fixed)) for a in ARCHS}})
    return pd.DataFrame(rows)


def table3(results: Path) -> pd.DataFrame:
    """All nine encoder-decoder pairings on copy at L in {16, 32, 64}."""
    df = select(load(results, "decoders.csv"), task="copy", vocabulary_size=8,
                structural_regularity=0.0, input_noise=0.0)
    rows = []
    for enc in ["bi_lstm", "bi_ssm", "transformer"]:
        for dec in ["lstm", "transformer", "ssm"]:
            rows.append({"Encoder": LABEL[enc], "Decoder": LABEL[dec],
                         **{f"L={L}": dbmin(select(df, encoder=enc, decoder=dec, sequence_length=L))
                            for L in [16, 32, 64]}})
    return pd.DataFrame(rows)


T4_POSITIONS = [0, 3, 7, 11, 15]


def table4(results: Path) -> pd.DataFrame:
    """Per-position clean accuracy at the largest tested d_b below d_b^min."""
    df = pd.read_csv(results / "forgetting.csv")
    rows = []
    for arch in ARCHS:
        runs = df[df.encoder == arch].drop_duplicates(subset=["bottleneck_dim", "seed"])
        means = runs.groupby("bottleneck_dim")["mean_val_noisy_acc"].mean()
        db = int(means[means < THETA].index.max())
        at = df[(df.encoder == arch) & (df.bottleneck_dim == db)]
        rows.append({"Encoder": LABEL[arch], "d_b": str(db),
                     **{f"pos{p}": f"{at[at.position == p].clean_acc.mean():.3f}"
                        for p in T4_POSITIONS}})
    return pd.DataFrame(rows)


def table5(results: Path) -> pd.DataFrame:
    """Single-task vs joint copy+sort d_b^min; joint also restricted to the single-task grid."""
    single = select(load(results, "sweeps.csv"), decoder="lstm", **BASELINE)
    joint = pd.read_csv(results / "multitask.csv")
    rows = []
    for arch in ARCHS:
        j = joint[joint.encoder == arch]
        means = j.groupby("bottleneck_dim")[["final_copy_acc", "final_sort_acc"]].mean()
        ok = means[(means.final_copy_acc >= THETA) & (means.final_sort_acc >= THETA)]
        copy_grid = set(select(single, task="copy", encoder=arch).bottleneck_dim)
        ok_shared = ok[ok.index.isin(copy_grid)]
        rows.append({
            "Encoder": LABEL[arch],
            "copy": dbmin(select(single, task="copy", encoder=arch)),
            "sort": dbmin(select(single, task="sorting", encoder=arch)),
            "joint": str(int(ok.index[0])) if len(ok) else FAIL,
            "joint_shared": str(int(ok_shared.index[0])) if len(ok_shared) else FAIL,
        })
    return pd.DataFrame(rows)


def table6(results: Path) -> pd.DataFrame:
    """BiLSTM copy under joint V x s x eta stress at L in {16, 64}."""
    df = select(load(results, "interaction.csv"), task="copy", encoder="bi_lstm")
    rows = []
    for L in [16, 64]:
        for V in [8, 32, 64]:
            for s in [0.0, 0.25, 0.5]:
                rows.append({"L": L, "V": V, "s": s,
                             **{f"eta={eta}": dbmin(select(df, sequence_length=L, vocabulary_size=V,
                                                           structural_regularity=s, input_noise=eta))
                                for eta in [0.0, 0.15, 0.3]}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- appendix tables

WARMUP_CELLS = [("eta=0", dict(input_noise=0.0)), ("eta=0.1", dict(input_noise=0.1)),
                ("eta=0.2", dict(input_noise=0.2)), ("L=32", dict(sequence_length=32))]


def warmup(results: Path) -> pd.DataFrame:
    """Copy d_b^min without -> with a 1000-step learning-rate warmup."""
    base = select(load(results, "sweeps.csv"), task="copy", decoder="lstm")
    warm = load(results, "warmup.csv")
    return pd.DataFrame([{"Encoder": LABEL[a],
                          **{name: f"{dbmin(select(base, encoder=a, **{**BASELINE, **c}))} -> "
                                   f"{dbmin(select(warm, encoder=a, **{**BASELINE, **c}))}"
                             for name, c in WARMUP_CELLS}} for a in ARCHS])


def ceiling(results: Path) -> pd.DataFrame:
    """BiSSM at s=0.5, d_b=40 (grid ceiling): paper seeds plus extra seeds."""
    base = select(load(results, "sweeps.csv"), task="copy", decoder="lstm", encoder="bi_ssm",
                  **{**BASELINE, "structural_regularity": 0.5})
    extra = pd.read_csv(results / "extra_seeds.csv")
    acc = pd.concat([base, extra[extra.structural_regularity == 0.5]])
    acc = acc[acc.bottleneck_dim == 40].final_val_noisy_acc
    return pd.DataFrame([{"Cell": "BiSSM, s=0.5, d_b=40", "n_seeds": str(len(acc)),
                          "mean": f"{acc.mean():.3f}", "std": f"{acc.std():.3f}",
                          "seeds>=0.8": str(int((acc >= THETA).sum()))}])


def palindrome(results: Path) -> pd.DataFrame:
    """Copy d_b^min on the palindrome source (d_b set directly, so no capacity_ratio)."""
    df = pd.read_csv(results / "palindrome.csv")
    return pd.DataFrame([{"Encoder": LABEL[a], "palindrome": dbmin(df[df.encoder == a])}
                         for a in ["bi_lstm", "transformer"]])


# ---------------------------------------------------------------- seed statistics (appendix)

def cells(results: Path):
    """Every distinct d_b^min cell of Tables 1-3, 5 (joint) and 6 as (table, cell, runs, accuracy column)."""
    sweeps = select(load(results, "sweeps.csv"), decoder="lstm")
    seen = set()
    for task in ["set", "sorting", "copy"]:
        for a in ARCHS:
            grp = select(sweeps, task=task, encoder=a, **BASELINE)
            seen.add(tuple(grp.index))
            yield "1", f"{task}, {LABEL[a]}", grp, "final_val_noisy_acc"
    for label, col, levels in T2_SWEEPS:
        for level in levels:
            for a in ARCHS:
                grp = select(sweeps, task="copy", encoder=a, **{**BASELINE, col: level})
                if tuple(grp.index) not in seen:  # the baseline appears in every sweep
                    seen.add(tuple(grp.index))
                    yield "2", f"{label}={level}, {LABEL[a]}", grp, "final_val_noisy_acc"
    dec = select(load(results, "decoders.csv"), task="copy", vocabulary_size=8,
                 structural_regularity=0.0, input_noise=0.0)
    for enc in ["bi_lstm", "bi_ssm", "transformer"]:
        for d in ["lstm", "transformer", "ssm"]:
            for L in [16, 32, 64]:
                yield "3", f"{LABEL[enc]}-{LABEL[d]}, L={L}", \
                    select(dec, encoder=enc, decoder=d, sequence_length=L), "final_val_noisy_acc"
    joint = pd.read_csv(results / "multitask.csv")
    for a in ARCHS:
        yield "5", f"joint, {LABEL[a]}", joint[joint.encoder == a], ("final_copy_acc", "final_sort_acc")
    inter = select(load(results, "interaction.csv"), task="copy", encoder="bi_lstm")
    for L in [16, 64]:
        for V in [8, 32, 64]:
            for s in [0.0, 0.25, 0.5]:
                for eta in [0.0, 0.15, 0.3]:
                    yield "6", f"L={L}, V={V}, s={s}, eta={eta}", \
                        select(inter, sequence_length=L, vocabulary_size=V,
                               structural_regularity=s, input_noise=eta), "final_val_noisy_acc"


def cell_dbmin(grp: pd.DataFrame, acc, theta: float = THETA):
    """d_b^min as an int (None on failure); a pair of columns means both seed means must reach theta."""
    means = grp.groupby("bottleneck_dim")[list(acc) if isinstance(acc, tuple) else [acc]].mean()
    passing = means[(means >= theta).all(axis=1)]
    return int(passing.index[0]) if len(passing) else None


def seeds(results: Path) -> pd.DataFrame:
    """Per cell and tested width: seed mean, std, range; d_b^min at 0.75/0.80/0.85; seed-decided flag
    (dropping any one seed changes d_b^min). Table 5 statistics are for copy accuracy."""
    rows = []
    for table, cell, grp, acc in cells(results):
        d = {th: cell_dbmin(grp, acc, th) for th in (0.75, THETA, 0.85)}
        decided = any(cell_dbmin(grp[grp.seed != s], acc) != d[THETA] for s in grp.seed.unique())
        col = acc[0] if isinstance(acc, tuple) else acc
        for db, x in grp.groupby("bottleneck_dim")[col]:
            rows.append({"table": table, "cell": cell, "d_b": int(db), "n": len(x),
                         "mean": round(x.mean(), 3), "std": round(x.std(), 3),
                         "min": round(x.min(), 3), "max": round(x.max(), 3),
                         **{f"dbmin@{th}": FAIL if v is None else str(v) for th, v in d.items()},
                         "seed_decided": decided})
    return pd.DataFrame(rows)


TABLES = {"table1": table1, "table2": table2, "table3": table3,
          "table4": table4, "table5": table5, "table6": table6,
          "warmup": warmup, "ceiling": ceiling, "palindrome": palindrome}

# ---------------------------------------------------------------- published values
# Cells as printed in the camera-ready paper, row by row in the order the functions above emit.
# "---" = no tested d_b reached the threshold (the paper writes ">40" and ">120" for two such cells
# whose largest tested width falls just short).

F = FAIL
PUBLISHED = {
    "table1": [["4", "4", "4"], ["10", "10", "10"], ["20", "20", "40"]],
    "table2": [["10", "10", "10"], ["20", "20", "40"], ["30", F, F], ["120", F, F], ["128", F, F],
               ["10", "10", "20"], ["20", "20", "40"], ["20", "40", "40"], ["30", F, "60"],
               ["30", F, "120"],
               ["20", "20", "40"], ["10", "20", "80"], ["10", "10", F], ["10", "10", "10"],
               ["20", "20", "40"], ["20", F, "80"], ["40", F, F], [F, F, F]],
    "table3": [["20", "30", "120"], ["20", "30", "128"], ["40", "60", F],
               ["40", F, F], [F, F, F], [F, F, F],
               ["20", F, F], [F, F, F], ["40", F, F]],
    "table4": [["10", "0.909", "0.946", "0.887", "0.893", "0.955"],
               ["10", "0.387", "0.431", "0.521", "0.758", "0.999"],
               ["30", "0.964", "0.794", "0.671", "0.757", "0.968"]],
    "table5": [["20", "10", "15", "20"], ["20", "10", F, F], ["40", "10", "30", "40"]],
    "table6": [["20", "20", F], ["10", "20", F], ["10", "40", F],
               ["30", "60", F], ["40", "80", F], ["40", F, F],
               ["30", "120", F], ["60", F, F], ["80", F, F],
               ["120", "120", F], ["100", "128", F], ["60", F, F],
               [F, F, F], [F, F, F], [F, F, F],
               [F, F, F], [F, F, F], [F, F, F]],
    "warmup": [["20 -> 20", "20 -> 20", "40 -> 40", "30 -> 30"],
               ["20 -> 20", "--- -> 40", "--- -> ---", "--- -> 120"],
               ["40 -> 20", "80 -> 40", "--- -> ---", "--- -> ---"]],
    "ceiling": [["25", "0.791", "0.037", "8"]],
    "palindrome": [["10"], ["10"]],
}
KEY_COLS = {"table1": 1, "table2": 2, "table3": 2, "table4": 1, "table5": 1, "table6": 3,
            "warmup": 1, "ceiling": 1, "palindrome": 1}


def verify(results: Path) -> bool:
    all_ok = True
    for name, build in TABLES.items():
        try:
            df = build(results)
        except FileNotFoundError as e:
            print(f"{name}: SKIPPED (missing {Path(e.filename).name})")
            all_ok = False
            continue
        k = KEY_COLS[name]
        mismatches = []
        for (_, row), expected in zip(df.iterrows(), PUBLISHED[name]):
            got = [str(v) for v in row.iloc[k:]]
            for col, g, e in zip(df.columns[k:], got, expected):
                if g != e:
                    key = ", ".join(f"{c}={row[c]}" for c in df.columns[:k])
                    mismatches.append(f"    [{key}] {col}: computed {g}, paper {e}")
        n = sum(len(r) for r in PUBLISHED[name])
        print(f"{name}: {n - len(mismatches)}/{n} cells match")
        print("\n".join(mismatches)) if mismatches else None
        all_ok &= not mismatches
    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=[*TABLES, "all", "verify", "seeds"])
    parser.add_argument("--results", type=Path, default=Path("results/paper"))
    args = parser.parse_args()

    if args.what == "verify":
        raise SystemExit(0 if verify(args.results) else 1)
    if args.what == "seeds":
        df = seeds(args.results)
        print(df.to_csv(index=False), end="")
        cells_ = df.drop_duplicates(["table", "cell"])
        print(f"# {len(cells_)} cells, {cells_.seed_decided.sum()} seed-decided", file=sys.stderr)
        return
    for name in TABLES if args.what == "all" else [args.what]:
        print(f"=== {name} ===")
        print(TABLES[name](args.results).to_string(index=False), end="\n\n")


if __name__ == "__main__":
    main()
