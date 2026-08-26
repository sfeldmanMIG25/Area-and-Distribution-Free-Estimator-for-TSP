"""Bring the canonical TSPLIB results file onto the final estimator roster.

Two edits, no others:

1. **Drop the withdrawn rows.** ``Chien``, ``Chien_extrap``, ``Kwon``,
   ``Kwon_extrap`` and ``Daganzo`` are removed. Their coefficients came from a
   secondary transcription and the primaries are paywalled with no obtainable
   open-access copy, so no published number may rest on them.

2. **Rescore ``Cavdar``.** The stored rows predate the rebuild against the
   primary document -- Cavdar's 2014 Georgia Tech dissertation, Ch. 4 -- which
   added the Eq. (21) finite-``n`` correction bounded to its fitted range
   ``n in [100, 975]`` and replaced the axis-aligned bounding box with the
   minimum-area enclosing rectangle found by a hull-edge scan.

Why this is not a full re-run of ``run_all_models_tsplib.py``
------------------------------------------------------------
The wall-clock columns in the canonical file do **not** come from that runner.
They were measured under a separate single-process protocol
(``serial_solo_median11_quiet_*``) and spliced in; a fresh threaded run would
overwrite every one of them with a contended value, which is the exact defect
``paper_tooling/restore_tsplib_serial_timings.py`` exists to undo. Predictions
for the models this change does not touch are functions of code that has not
changed, so re-running them can only add measurement noise to the timing
columns. This script therefore rewrites prediction cells only.

Cavdar's own timing cells are re-measured, because its implementation *did*
change: ``--retime`` reruns the estimator under the recorded protocol (single
process, one BLAS thread, median of ``REPEATS`` draws) and stamps a new
``timing_provenance``. Without ``--retime`` the timing cells are left alone and
the run reports that they describe the pre-rebuild code.

Usage
-----
    python paper_tooling/rescore_tsplib_final_roster.py --retime
    python paper_tooling/rescore_tsplib_final_roster.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tsplib_benchmark"))

from tsplib_parser import parse_tsplib_file  # noqa: E402
import classical_region_estimators as region_est  # noqa: E402

INSTANCES_DIR = ROOT / "tsplib_benchmark" / "instances"
CANONICAL = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
BACKUP = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_pre_final_roster.csv.bak"

WITHDRAWN = ["Chien", "Chien_extrap", "Kwon", "Kwon_extrap", "Daganzo"]
RESCORE = "Cavdar"
NATIVE_2D_TYPES = {"EUC_2D", "CEIL_2D"}
REPEATS = 11


def _coords_2d(name: str) -> np.ndarray | None:
    """Reproduce the runner's coordinate handling for a native 2D instance."""
    path = INSTANCES_DIR / f"{name}.tsp"
    if not path.exists():
        return None
    info = parse_tsplib_file(path)
    if info["edge_weight_type"] not in NATIVE_2D_TYPES:
        return None
    return info["raw_coords"].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retime", action="store_true",
                    help="re-measure Cavdar wall time (median of %d solo draws)" % REPEATS)
    args = ap.parse_args()

    df = pd.read_csv(CANONICAL)
    before_models = sorted(df["model"].unique())
    n_before = len(df)

    drop_mask = df["model"].isin(WITHDRAWN)
    print(f"dropping {int(drop_mask.sum())} rows across {len(WITHDRAWN)} withdrawn models")

    est = region_est.ESTIMATORS[RESCORE]
    idx = df.index[(df["model"] == RESCORE)]
    moved, unchanged, skipped = 0, 0, 0
    provenance_tag = f"serial_solo_median{REPEATS}_quiet_{date.today().isoformat()}"

    for i in idx:
        row = df.loc[i]
        if row["status"] != "ok":
            skipped += 1
            continue
        coords = _coords_2d(str(row["instance"]))
        if coords is None:
            raise RuntimeError(f"no native 2D coordinates for {row['instance']!r}, "
                               "but the stored row has status 'ok'")
        res = est.estimate(coords, 2, None)
        if res["status"] != "ok":
            raise RuntimeError(f"{row['instance']}: estimator now declines "
                               f"({res['status']}) an instance previously scored ok")
        pred = float(res["estimate"])
        old = float(row["pred_cost"])
        gap = (pred - float(row["true_cost"])) / float(row["true_cost"]) * 100.0
        df.at[i, "pred_cost"] = pred
        df.at[i, "gap_pct"] = gap
        df.at[i, "abs_gap_pct"] = abs(gap)
        if args.retime:
            draws = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                est.estimate(coords, 2, None)
                draws.append(time.perf_counter() - t0)
            total = float(np.median(draws))
            df.at[i, "total_time_s"] = total
            df.at[i, "feature_time_s"] = total
            df.at[i, "inference_time_s"] = 0.0
            ct = row.get("concorde_time_s")
            df.at[i, "speedup_vs_concorde"] = (float(ct) / total) if pd.notna(ct) else None
            df.at[i, "timing_provenance"] = provenance_tag
        if abs(pred - old) > 1e-9 * max(1.0, abs(old)):
            moved += 1
        else:
            unchanged += 1

    print(f"rescored {RESCORE}: {moved} predictions moved, {unchanged} unchanged, "
          f"{skipped} non-ok rows untouched")
    if args.retime:
        print(f"re-timed {RESCORE} under {provenance_tag}")
    else:
        print(f"WARNING: {RESCORE} timing cells still describe the pre-rebuild "
              "implementation. Pass --retime.")

    out = df[~drop_mask].reset_index(drop=True)
    ok = out[(out["model"] == RESCORE) & (out["status"] == "ok")]
    print(f"\n{RESCORE} on TSPLIB after rescore: N={len(ok)} "
          f"MAPE={ok['abs_gap_pct'].mean():.2f} median={ok['abs_gap_pct'].median():.2f} "
          f"MSPE={ok['gap_pct'].mean():.2f}")
    print(f"rows {n_before} -> {len(out)}; models {len(before_models)} -> "
          f"{out['model'].nunique()}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if not BACKUP.exists():
        shutil.copy2(CANONICAL, BACKUP)
        print(f"backup -> {BACKUP.name}")
    out.to_csv(CANONICAL, index=False)
    print(f"wrote {CANONICAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
