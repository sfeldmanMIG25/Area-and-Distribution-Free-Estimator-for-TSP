"""Cost of the 1-tree upper anchor, relative to GART 2.0, on TSPLIB EUC_2D.

WHAT THIS DOES AND DOES NOT MEASURE
-----------------------------------
It does NOT reproduce the published solo timing column. That protocol
(``serial_median11_quiet_*``: one estimator per process, 11 repeats, threads
pinned, quiet box) yields *absolute* milliseconds, and absolute milliseconds
require an idle machine. Run this on a loaded box and the absolutes are junk.

What it does measure is the *ratio*, which is what the middle-ground claim
actually needs. Every model is timed back-to-back inside one process, on the
same instance, in the same repeat, with the order rotated between repeats. A
background load inflates every arm of the comparison at once, so the ratio
survives what the absolute numbers do not. Absolute numbers stay pending until
the box is quiet, and the output says so.

Threads are pinned to 1 before numpy or lightgbm is imported, matching the
published protocol. JIT and first-predict are warmed outside the clock.

    python paper_tooling/hk1tree_cost_ratio.py [--repeats 5] [--max-n 3000]
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tsplib_benchmark", ROOT / "lgbm_model_v3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from held_karp_1tree import HeldKarp1Tree  # noqa: E402
from tsplib_parser import parse_tsplib_file  # noqa: E402
import classical_region_estimators as region_est  # noqa: E402

INSTANCE_DIR = ROOT / "tsplib_benchmark" / "instances"
OUT = ROOT / "paper_tooling" / "hk1tree_cost_ratio.csv"


def build_models():
    models = {}
    from lgbm_estimator_gart2 import TSP_GART2_Estimator
    models["GART_2.0"] = TSP_GART2_Estimator()
    models["MST_Only"] = region_est.ESTIMATORS["MST_Only"]
    for k in (10, 50, 100, 500, 1000):
        models[f"HK_1Tree_{k}"] = HeldKarp1Tree(iterations=k)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-n", type=int, default=3000)
    args = ap.parse_args()

    load = os.popen(
        'powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | '
        'Measure-Object -Property LoadPercentage -Average).Average"'
    ).read().strip()
    print(f"CPU load at start: {load}%   (ratios are robust to this; absolutes are not)")

    insts = []
    for f in sorted(INSTANCE_DIR.glob("*.tsp")):
        try:
            info = parse_tsplib_file(str(f))
        except Exception:
            continue
        if info["edge_weight_type"].upper() != "EUC_2D":
            continue
        if int(info["n"]) > args.max_n or info.get("raw_coords") is None:
            continue
        insts.append((f.stem, np.asarray(info["raw_coords"], dtype=np.float64)))
    print(f"{len(insts)} TSPLIB EUC_2D instances with n <= {args.max_n}")

    models = build_models()
    names = list(models)

    # Warm: JIT, first predict, page-ins -- all outside the clock.
    for nm in names:
        for _, X in insts[:3]:
            models[nm].estimate(X, X.shape[1])

    rows = []
    for r in range(args.repeats):
        order = names[r % len(names):] + names[: r % len(names)]   # rotate
        for nm in order:
            est = models[nm]
            for name, X in insts:
                t0 = time.perf_counter()
                est.estimate(X, X.shape[1])
                rows.append({"repeat": r, "model": nm, "instance": name,
                             "n": X.shape[0], "seconds": time.perf_counter() - t0})
        print(f"  repeat {r+1}/{args.repeats} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # Per-repeat corpus total, then the median across repeats -- the same shape
    # as the published "EUC_2D total ms" column.
    tot = df.groupby(["model", "repeat"]).seconds.sum().reset_index()
    med = tot.groupby("model").seconds.median() * 1000.0
    base = med["GART_2.0"]
    summary = pd.DataFrame({
        "total_ms_median": med.round(3),
        "x_GART_2.0": (med / base).round(3),
    }).sort_values("total_ms_median")
    print(f"\nEUC_2D corpus total ({len(insts)} instances), median of "
          f"{args.repeats} interleaved repeats:")
    print(summary.to_string())
    print("\nABSOLUTE ms ARE NOT COMPARABLE to the published solo column: this run "
          "is multi-model-in-one-process on a loaded box.\nThe x_GART_2.0 column is "
          "the measurement this script is for.")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
