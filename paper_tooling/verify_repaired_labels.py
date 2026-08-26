"""Post-repair gate: no certified lower bound may exceed a repaired label.

``paper_tooling/label_certificate.py`` is the *discovery* instrument.  It
compares a proven lower bound ``B`` against a released label ``L`` that lives
in an integer-quantised metric, so it has to carry the quantisation slack
``n q / 2`` and can only refute a label that clears it.  That is what found the
defect, and its summary (``label_certificate.json``) is kept as the pre-repair
record.

This module is the *acceptance* test for the repair, and it is strictly
sharper.  Every repaired label is either an exact float64 optimum or the
float64 length of a real tour on the released coordinates, so it is an upper
bound on the optimum **in the same metric the 1-tree bound is computed in**.
The inequality is therefore exact, with no slack to hide in:

    B  <=  OPT  <=  L_repaired          for every instance that has a bound.

A single violation means the repair is wrong somewhere, and this script exits
non-zero.  Quarantined instances have no label and are excluded by
construction; they are counted and reported, never silently skipped.

    python paper_tooling/verify_repaired_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paper_tooling.label_quarantine import quarantined  # noqa: E402

HERE = ROOT / "paper_tooling"
P_LABELS = HERE / "labels_repaired.csv"
P_OUT = HERE / "verify_repaired_labels.json"

# Arithmetic noise of the ascent, relative. Anything above this is a real
# violation, not a rounding artefact.
REL_EPS = 1e-9

BOUND_SOURCES = [
    ("nd_test", HERE / "polyak_nd_sweep.csv"),
    ("nd_test_vj", HERE / "hk1tree_frontier_nd.csv"),
    ("nd_trainval", HERE / "hk1tree_trainval_k1000.csv"),
    ("nd_invariant", HERE / "hk1tree_invariant_k1000.csv"),
    ("2d", HERE / "hk1tree_frontier_2d.csv"),
    ("tsplib_int_metric", HERE / "hk1tree_frontier_tsplib_int.csv"),
]


def main() -> int:
    lab = pd.read_csv(P_LABELS)
    repaired = dict(zip(lab["instance"].astype(str),
                        lab["label_repaired"].astype(float)))
    quar = quarantined()

    report: dict = {"rel_eps": REL_EPS, "sources": {}, "violations": []}
    worst_overall = 0.0
    for name, path in BOUND_SOURCES:
        if not path.exists():
            report["sources"][name] = {"skipped": "missing"}
            continue
        df = pd.read_csv(path, usecols=lambda c: c in
                         {"instance", "bound", "status", "k"}, low_memory=False)
        if "status" in df.columns:
            df = df[df["status"].astype(str) == "ok"]
        best = df.groupby(df["instance"].astype(str))["bound"].max()
        n_quar = int(best.index.isin(quar).sum())
        best = best[~best.index.isin(quar)]
        lab_s = best.index.map(repaired)
        covered = pd.notna(lab_s)
        b = best[covered].to_numpy(dtype=float)
        L = pd.Series(lab_s)[covered.tolist()].to_numpy(dtype=float)
        excess = (b - L) / L
        bad = excess > REL_EPS
        worst = float(excess.max()) if len(excess) else float("nan")
        worst_overall = max(worst_overall, worst if worst == worst else 0.0)
        report["sources"][name] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "instances_with_bound": int(len(best)),
            "quarantined_excluded": n_quar,
            "no_repaired_label": int((~covered).sum()),
            "violations": int(bad.sum()),
            "worst_excess_rel": worst,
        }
        if bad.any():
            names = best.index[covered][bad][:20].tolist()
            report["violations"].extend(
                {"source": name, "instance": i} for i in names)

    total = sum(v.get("violations", 0) for v in report["sources"].values()
                if isinstance(v, dict))
    report["total_violations"] = int(total)
    report["worst_excess_rel_any_source"] = worst_overall
    report["verdict"] = "PASS" if total == 0 else "FAIL"
    P_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
