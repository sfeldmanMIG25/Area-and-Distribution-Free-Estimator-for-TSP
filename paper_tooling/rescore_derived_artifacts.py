"""Push the repaired ground truth into the derived per-instance frames.

``apply_repaired_labels.py`` owns the released benchmark CSVs.  This script
owns the *analysis* frames that carry their own copy of the label -- the
Held--Karp ladders, the Polyak sweep, and the study sidecars -- because a bank
rebuilt from a stale copy would reintroduce the defect one layer down.

Two rules, applied uniformly:

* ``true_cost`` (or the file's own label column) is replaced from
  ``labels_repaired.csv``.  Derived error columns are recomputed where the
  file defines them.
* Quarantined instances are **dropped**, not blanked.  These are internal
  analysis frames rather than released per-instance results, so removing the
  row is the honest representation: there is nothing to score.  The row count
  before and after is reported for every file so the drop is visible.

Run ``apply_repaired_labels.py`` first, then this, then the bank rebuilds
listed in ``REBUILD_ORDER`` below.

    python paper_tooling/rescore_derived_artifacts.py [--dry-run]
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
from paper_tooling.label_quarantine import quarantined  # noqa: E402

P_LABELS = ROOT / "paper_tooling" / "labels_repaired.csv"
P_REPORT = ROOT / "paper_tooling" / "rescore_derived_artifacts.json"
HERE = ROOT / "paper_tooling"

# file, label column, extra columns to recompute afterwards
FRAMES: list[tuple[Path, str]] = [
    (HERE / "hk1tree_frontier_nd.csv", "true_cost"),
    (HERE / "hk1tree_frontier_2d.csv", "true_cost"),
    (HERE / "hk1tree_frontier_train.csv", "true_cost"),
    (HERE / "hk1tree_frontier_train_d2.csv", "true_cost"),
    (HERE / "hk1tree_frontier_tsplib.csv", "true_cost"),
    (HERE / "hk1tree_frontier_tsplib_int.csv", "true_cost"),
    (HERE / "hk1tree_ladder.csv", "true_cost"),
    (HERE / "hk1tree_invariant_k1000.csv", "true_cost"),
    (HERE / "hk1tree_trainval_k1000.csv", "true_cost"),
    (HERE / "hk1tree_delaunay_trap.csv", "true_cost"),
    (HERE / "polyak_nd_sweep.csv", "true_cost"),
    (HERE / "polyak_nd_probe.csv", "true_cost"),
    (HERE / "polyak_nd_ub_sensitivity.csv", "true_cost"),
    (HERE / "polyak_nd_convergence_audit.csv", "true_cost"),
    (HERE / "v4_study_allmodels_per_instance.csv", "true_cost"),
    (HERE / "armA_verify_per_instance.csv", "true_cost"),
    (HERE / "support_arms_per_instance.csv", "true_cost"),
    # The 31-feature controls. rows_tsplib.csv is read directly by
    # build_paper_tables for the rank table, so a stale copy here would put a
    # pre-repair label into a published table.
    (HERE / "controls_31f" / "rows_2d.csv", "true_cost"),
    (HERE / "controls_31f" / "rows_nd.csv", "true_cost"),
    (HERE / "controls_31f" / "rows_tsplib.csv", "true_cost"),
]

# Rebuild these, in this order, after the frames are rescored.
REBUILD_ORDER = [
    "paper_tooling/hk1tree_frontier_analyze.py",
    "paper_tooling/hk1tree_solo_cost.py",
    "paper_tooling/hk1tree_polyak_nd_analyze.py",
    "paper_tooling/hk1tree_polyak_nd_frontier.py",
    "paper_tooling/report_31f_controls.py",
    "paper_tooling/frontier_manuscript_numbers.py",
    "paper_tooling/build_paper_tables.py",
    "paper_tooling/splice_tables.py",
]

# Derived error columns keyed on (numerator column, label column).
GAP_COLS = {
    "gap_pct": lambda pred, true: 100.0 * (pred - true) / true,
    "abs_gap_pct": lambda pred, true: (100.0 * (pred - true) / true).abs(),
    "err_pct": lambda pred, true: 100.0 * (pred - true) / true,
    "abs_err_pct": lambda pred, true: (100.0 * (pred - true) / true).abs(),
}
PRED_COLS = ("bound", "pred_cost", "pred")


def _backup(path: Path) -> None:
    bak = path.with_name(path.stem + "_as_published.csv")
    if not bak.exists():
        shutil.copy2(path, bak)


def _rescore(path: Path, label_col: str, repaired: dict[str, float],
             quar: frozenset[str], dry: bool) -> dict:
    if not path.exists():
        return {"path": path.name, "skipped": "missing"}
    df = pd.read_csv(path, low_memory=False)
    icol = "instance" if "instance" in df.columns else None
    if icol is None or label_col not in df.columns:
        return {"path": path.name, "skipped": "no instance/label column"}

    key = df[icol].astype(str)
    before = len(df)
    drop = key.isin(quar)
    known = key.isin(repaired)
    old = pd.to_numeric(df[label_col], errors="coerce")
    new = key.map(repaired)
    changed = int((known & ~drop & (new - old).abs().gt(1e-9)).sum())
    rep = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows_before": before,
        "rows_dropped_quarantined": int(drop.sum()),
        "rows_unmatched": int((~known & ~drop).sum()),
        "rows_label_changed": changed,
    }
    if dry:
        return rep

    _backup(path)
    df = df[~drop].copy()
    key = df[icol].astype(str)
    df[label_col] = np.where(key.isin(repaired), key.map(repaired),
                             pd.to_numeric(df[label_col], errors="coerce"))
    true = pd.to_numeric(df[label_col], errors="coerce")
    pred_col = next((c for c in PRED_COLS if c in df.columns), None)
    if pred_col is not None:
        pred = pd.to_numeric(df[pred_col], errors="coerce")
        for c, fn in GAP_COLS.items():
            if c in df.columns:
                df[c] = fn(pred, true)
    df.to_csv(path, index=False)
    rep["rows_after"] = int(len(df))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lab = pd.read_csv(P_LABELS)
    lab = lab[lab["label_repaired"].notna()]
    repaired = dict(zip(lab["instance"].astype(str), lab["label_repaired"].astype(float)))
    quar = quarantined()

    reports = [_rescore(p, c, repaired, quar, args.dry_run) for p, c in FRAMES]
    out = {"dry_run": args.dry_run, "frames": reports,
           "rebuild_order": REBUILD_ORDER}
    print(json.dumps(out, indent=2))
    if not args.dry_run:
        P_REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
