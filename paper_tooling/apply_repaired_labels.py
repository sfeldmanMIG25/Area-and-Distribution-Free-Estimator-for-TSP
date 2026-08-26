"""Write the repaired ground truth into every released results CSV.

Reads ``paper_tooling/labels_repaired.csv`` (built by
``paper_tooling/repair_labels.py``) and rewrites, in place:

  Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv
  Generalized_TSP_Analysis_ND/benchmark_checkpoints/base_ground_truth_nd.csv
  Generalized_TSP_Analysis/benchmark_results_2D_v3.csv
  Generalized_TSP_Analysis/benchmark_checkpoints/base_ground_truth_2d.csv
  tsplib_benchmark/results/all_models_tsplib.csv

Predictions are untouched.  An estimator's output depends on coordinates, not
on the label, so repairing a label rescores a row -- it does not re-run it.
``true_cost`` is replaced, ``gap_pct`` / ``abs_gap_pct`` / ``alpha`` are
recomputed from it, and nothing else moves.

Quarantined instances get an **empty** ``true_cost`` and
``status = quarantined_label_unrecoverable``.  See
``paper_tooling/label_quarantine.py`` for why the field is emptied rather than
filled with a sentinel.

The as-published file is preserved next to each target as
``<stem>_as_published.csv`` the first time this runs, and never overwritten
after that, so the pre-repair numbers stay auditable.

Usage
-----
    python paper_tooling/apply_repaired_labels.py           # apply
    python paper_tooling/apply_repaired_labels.py --dry-run # report only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper_tooling.label_quarantine import (  # noqa: E402
    QUARANTINE_LABEL_STATUS,
    QUARANTINE_STATUS,
)

P_LABELS = ROOT / "paper_tooling" / "labels_repaired.csv"
P_REPORT = ROOT / "paper_tooling" / "apply_repaired_labels.json"

CKPT_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints"
CKPT_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints"


def _targets() -> list[tuple[str, Path]]:
    """Aggregates first, then the per-model resume checkpoints they were built
    from -- a stale checkpoint would silently reintroduce the old label the next
    time a runner resumes."""
    t: list[tuple[str, Path]] = [
        ("nd", ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"),
        ("nd", CKPT_ND / "base_ground_truth_nd.csv"),
        ("2d", ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"),
        ("2d", CKPT_2D / "base_ground_truth_2d.csv"),
        ("tsplib", ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"),
        # Live parallel copies that the frontier bank reads. The ``*_pre_gart2``
        # files are historical snapshots and are deliberately left alone.
        ("tsplib", ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"),
        ("tsplib", ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_parallel_timings.csv"),
    ]
    for corpus, folder in (("nd", CKPT_ND), ("2d", CKPT_2D)):
        for p in sorted(folder.glob("results_*.csv")):
            if p.stem.endswith("_as_published"):
                continue
            t.append((corpus, p))
    return t


def _backup(path: Path) -> Path:
    bak = path.with_name(path.stem + "_as_published.csv")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def _apply(path: Path, corpus: str, labels: pd.DataFrame, dry: bool) -> dict:
    df = pd.read_csv(path)
    sub = labels[labels["corpus"] == corpus]
    repaired = dict(zip(sub["instance"].astype(str), sub["label_repaired"]))
    quar = set(sub.loc[sub["label_status"] == QUARANTINE_LABEL_STATUS,
                       "instance"].astype(str))

    key = df["instance"].astype(str)
    known = key.isin(repaired)
    is_quar = key.isin(quar)
    new_true = key.map(repaired)  # NaN for quarantined and for unknown

    old_true = pd.to_numeric(df["true_cost"], errors="coerce")
    moved = int((known & ~is_quar
                 & (new_true - old_true).abs().gt(1e-9)).sum())
    report = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": int(len(df)),
        "rows_unmatched": int((~known).sum()),
        "rows_quarantined": int(is_quar.sum()),
        "rows_true_cost_changed": moved,
    }
    if dry:
        return report

    df["true_cost"] = np.where(known, new_true, old_true)
    if "pred_cost" in df.columns:
        pred = pd.to_numeric(df["pred_cost"], errors="coerce")
        tc = pd.to_numeric(df["true_cost"], errors="coerce")
        gap = 100.0 * (pred - tc) / tc
        df["gap_pct"] = gap
        if "abs_gap_pct" in df.columns:
            df["abs_gap_pct"] = gap.abs()
    if "mst_length" in df.columns and "alpha" in df.columns:
        mst = pd.to_numeric(df["mst_length"], errors="coerce")
        df["alpha"] = pd.to_numeric(df["true_cost"], errors="coerce") / mst
    if "true_alpha" in df.columns and "mst_length" in df.columns:
        mst = pd.to_numeric(df["mst_length"], errors="coerce")
        df["true_alpha"] = pd.to_numeric(df["true_cost"], errors="coerce") / mst
    if "status" in df.columns:
        df.loc[is_quar, "status"] = QUARANTINE_STATUS
    _backup(path)
    df.to_csv(path, index=False)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labels = pd.read_csv(P_LABELS)
    reports = [_apply(p, corpus, labels, args.dry_run) for corpus, p in _targets()]
    out = {"dry_run": args.dry_run, "targets": reports}
    print(json.dumps(out, indent=2))
    if not args.dry_run:
        P_REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
