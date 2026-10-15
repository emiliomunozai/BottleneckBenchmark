"""
Results layout: summary table (CSV) + append-only run detail (JSON Lines).
"""
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def _project_root() -> Path:
    """Project root (dir containing pyproject.toml); fallback to cwd when not found."""
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        if (cur / "pyproject.toml").exists():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return Path.cwd()


# data/
#   results/
#     results.csv       – one row per run (final metrics + hyperparameters)
#     runs_detail.jsonl – one JSON object per line: config, full step history, summary, checkpoint path
#   traces/     – optional legacy step CSVs (same columns as history in runs_detail)
#   figures/    – saved plots from bb report --save
#   checkpoints/ – model weights (.pt)
DATA_DIR = _project_root() / "data"
RESULTS_DIR = DATA_DIR / "results"
TRACES_DIR = DATA_DIR / "traces"
FIGURES_DIR = DATA_DIR / "figures"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

RESULTS_CSV = RESULTS_DIR / "results.csv"
_LEGACY_RESULTS = DATA_DIR / "results.csv"


def default_results_path() -> Path:
    """Path for the summary results table."""
    if not RESULTS_CSV.exists() and _LEGACY_RESULTS.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _LEGACY_RESULTS.replace(RESULTS_CSV)
    return RESULTS_CSV


def runs_detail_path(summary_csv: Path | str) -> Path:
    """JSONL detail file that pairs with a given results.csv (same directory, fixed name)."""
    p = Path(summary_csv).resolve()
    return p.parent / "runs_detail.jsonl"


def traces_dir() -> Path:
    """Directory for ad-hoc trace CSVs."""
    return TRACES_DIR


def figures_dir() -> Path:
    """Directory for saved analysis plots."""
    return FIGURES_DIR


def checkpoints_dir() -> Path:
    """Directory for saved model checkpoints."""
    return CHECKPOINTS_DIR


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return []
    if "model" not in df.columns:
        df["model"] = ""
    else:
        df["model"] = df["model"].fillna("")
    if "encoder" not in df.columns:
        df["encoder"] = ""
    else:
        df["encoder"] = df["encoder"].fillna("")
    if "decoder" not in df.columns:
        df["decoder"] = ""
    else:
        df["decoder"] = df["decoder"].fillna("")
    if "n_layers" not in df.columns:
        df["n_layers"] = 1
    else:
        df["n_layers"] = df["n_layers"].fillna(1).astype(int)
    if "input_noise" not in df.columns:
        if "target_noise" in df.columns:
            df["input_noise"] = df["target_noise"].fillna(0.0).astype(float)
        else:
            df["input_noise"] = 0.0
    else:
        df["input_noise"] = df["input_noise"].fillna(0.0).astype(float)
    if "target_noise" not in df.columns:
        df["target_noise"] = df["input_noise"]
    else:
        df["target_noise"] = df["target_noise"].fillna(df["input_noise"]).astype(float)
    return df.to_dict("records")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def append_results(rows: list[dict], path: Path | None = None) -> Path:
    """Append summary rows to results CSV. Returns path written."""
    path = Path(path) if path else default_results_path()
    existing = _read_csv(path)
    _write_csv(path, existing + rows)
    return path


def overwrite_results(rows: list[dict], path: Path | None = None) -> Path:
    """Overwrite results CSV. Returns path written."""
    path = Path(path) if path else default_results_path()
    _write_csv(path, rows)
    return path


def append_run_detail(record: dict, summary_csv: Path | str | None = None) -> Path:
    """Append one JSON line to runs_detail.jsonl (paired with summary_csv)."""
    summary = Path(summary_csv) if summary_csv else default_results_path()
    detail_path = runs_detail_path(summary)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
    with open(detail_path, "a", encoding="utf-8") as f:
        f.write(line)
    return detail_path


def clear_run_detail(summary_csv: Path | str | None = None) -> Path:
    """Truncate the detail JSONL (e.g. when overwriting the summary table)."""
    summary = Path(summary_csv) if summary_csv else default_results_path()
    p = runs_detail_path(summary)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    return p


def load_run_detail_record(detail_path: Path | str, run_id: int) -> dict | None:
    """Load the last record for run_id from a runs_detail.jsonl file."""
    path = Path(detail_path)
    if not path.exists():
        return None
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run_id") == run_id:
                last = rec
    return last


def load_existing_rows(path: Path, config_signature_fn) -> tuple[list[dict], set[tuple]]:
    """Load existing rows and their config signatures."""
    if not path.exists():
        return [], set()
    rows = _read_csv(path)
    return rows, {config_signature_fn(r) for r in rows}


def get_next_run_id(path: Path | None = None) -> int:
    """Next run_id (max + 1 in summary CSV, or 0)."""
    path = Path(path) if path else default_results_path()
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return 0
    if "run_id" not in df.columns or df.empty:
        return 0
    return int(df["run_id"].max()) + 1
