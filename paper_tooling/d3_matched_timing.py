"""Size-matched cost of GART 2.0 against the Polyak 1-tree ascent at d = 2, 3, 4.

Why this exists
---------------
``hk1tree_polyak_nd.timing`` draws its sample with ``stratified``, which picks
``per_cell`` instances independently *within each dimension*.  The n-mix of the
d = 2 draw is therefore unrelated to the n-mix of the d = 3 draw -- in the
released sample the medians are n ~ 550 against n ~ 70 -- so a per-dimension
cost ratio read off that file is a size effect wearing a dimension's label.
The published aggregate over d in {2,3} is sound; splitting it is not.

This module draws the same cells but *matched on n*: the same size bands, the
same instance count per band, for every dimension it reports.  The question it
answers is whether d = 3 holds the cost/accuracy front on its own, or whether
d = 2 carries the published d in {2,3} row.  d = 4 is the control -- it is the
first dimension at which ``mst_utils.compute_mst`` abandons the Delaunay path
for the dense kernel, so if the size-matched ratios show a cliff it should
appear between 3 and 4 rather than between 2 and 3.

Protocol is ``hk1tree_polyak_nd.timing``'s, unchanged: one process, threads
pinned to 1, both estimators warmed then measured back to back in the same
session, each budget timed by its own ``polyak_bound`` call, median of
``repeats``.  Absolute milliseconds are load-dependent; the ratio is the datum.
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_tooling"))

from hk1tree_polyak import polyak_bound  # noqa: E402
from hk1tree_polyak_nd import FLOAT_FMT, load_base, tasks_from  # noqa: E402

OUT = ROOT / "paper_tooling"

#: The published accuracy buckets, so a cost ratio can be read against the
#: MAPE already measured on all 647/648 instances of each dimension.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("[5,10]", 5, 10),
    ("[20,100]", 20, 100),
    ("[200,500]", 200, 500),
    ("[600,1000]", 600, 1000),
)

#: Up to the largest crossover budget any of these dimensions needs (d = 2 at
#: k = 200).  Running the full published ladder would triple the wall clock to
#: measure rungs no verdict here reads.
BUDGETS: tuple[int, ...] = (0, 10, 25, 50, 100, 200, 500)

DIMS: tuple[int, ...] = (2, 3, 4)

SEED = 20260821


def matched_sample(base: pd.DataFrame, per_band: int, seed: int) -> pd.DataFrame:
    """``per_band`` instances from every (dimension, size band) cell.

    A cell short of ``per_band`` takes everything it has and is reported with
    its true count, so a thin cell is visible in the output rather than being
    silently padded from a neighbouring band.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for d in DIMS:
        for label, lo, hi in BANDS:
            g = base[(base.dimension == d)
                     & (base.n_customers >= lo)
                     & (base.n_customers <= hi)]
            if g.empty:
                print(f"  [warn] empty cell d={d} n in {label}")
                continue
            take = min(per_band, len(g))
            idx = rng.choice(len(g), size=take, replace=False)
            sel = g.iloc[np.sort(idx)].copy()
            sel["band"] = label
            parts.append(sel)
    return pd.concat(parts, ignore_index=True)


def run(per_band: int, repeats: int) -> Path:
    from lgbm_model_v3.lgbm_estimator_gart2 import TSP_GART2_Estimator

    base = load_base()
    sel = matched_sample(base, per_band, SEED)
    bands = dict(zip(sel.instance, sel.band))
    tasks = tasks_from(sel)
    gart = TSP_GART2_Estimator(str(ROOT / "lgbm_model_v3"))

    rows = []
    for i, t in enumerate(tasks, 1):
        X32 = np.asarray(t["coords"], dtype=np.float32)
        gart.estimate(X32, t["d"], t["grid_size"])   # warm the predict path
        polyak_bound(t["coords"], 5)                 # warm the JIT

        g = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            gart.estimate(X32, t["d"], t["grid_size"])
            g.append(time.perf_counter() - t0)

        cell = {"instance": t["instance"], "n": t["n"], "d": t["d"],
                "band": bands[t["instance"]],
                "gart_ms": 1000 * float(np.median(g))}
        for k in BUDGETS:
            reps = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                polyak_bound(t["coords"], k)
                reps.append(time.perf_counter() - t0)
            cell[f"hk_k{k}_ms"] = 1000 * float(np.median(reps))
        rows.append(cell)
        print(f"  [{i}/{len(tasks)}] d={t['d']} n={t['n']} "
              f"gart={cell['gart_ms']:.2f}ms k100={cell['hk_k100_ms']:.2f}ms",
              flush=True)

    df = pd.DataFrame(rows)
    p = OUT / globals().get("_OUT_NAME", "d3_matched_timing.csv")
    df.to_csv(p, index=False, float_format=FLOAT_FMT)
    print(f"\nwrote {p}  ({len(df)} instances)")
    return p


def main() -> None:
    global DIMS, BANDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dims", type=str, default=None,
                    help="comma-separated dimensions, e.g. 10,50,100")
    ap.add_argument("--bands", type=str, default=None,
                    help="comma-separated band labels, e.g. [600,1000]")
    ap.add_argument("--out", type=str, default="d3_matched_timing.csv")
    ap.add_argument("--smoke", action="store_true",
                    help="1 instance per cell, 1 repeat -- validates wiring only")
    a = ap.parse_args()
    if a.dims:
        DIMS = tuple(int(x) for x in a.dims.split(","))
    if a.bands:
        want = {s.strip() for s in a.bands.split(";")}
        BANDS = tuple(b for b in BANDS if b[0] in want)
        if not BANDS:
            raise SystemExit(f"no band matched {want}")
    globals()["_OUT_NAME"] = a.out
    if a.smoke:
        run(per_band=1, repeats=1)
    else:
        run(per_band=a.per_band, repeats=a.repeats)


if __name__ == "__main__":
    main()
