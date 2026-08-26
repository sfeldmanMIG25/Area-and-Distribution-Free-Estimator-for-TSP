"""Validate the 1-tree bound on TSPLIB, in TSPLIB's own metric.

This is the cleanest of the validations available, because it removes the one
ambiguity that clouds the synthetic corpora. TSPLIB optima are *published and
proven exact*, and they are exact with respect to a specific integer distance
convention: ``EUC_2D`` is ``nint(euclidean)``, ``CEIL_2D`` is ``ceil``, ``ATT``
is the pseudo-Euclidean rule, ``GEO`` the great-circle rule, ``EXPLICIT`` the
tabulated matrix. Building the bound on that same integer matrix and comparing
against the published optimum is an exact, metric-consistent test with no
label-precision escape hatch:

    bound <= published optimum, or the implementation is wrong.

Instances whose matrix does not fit in memory record a status row rather than
being silently dropped -- ``pla85900`` would need 55 GiB.

    python paper_tooling/hk1tree_tsplib.py [--budget 1000] [--max-n 8000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tsplib_benchmark"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from held_karp_1tree import one_tree_bound_from_matrix  # noqa: E402
from tsplib_parser import parse_tsplib_file  # noqa: E402

INSTANCE_DIR = ROOT / "tsplib_benchmark" / "instances"
OPTIMA = ROOT / "tsplib_benchmark" / "ground_truth" / "optima.csv"
OUT = ROOT / "paper_tooling" / "hk1tree_tsplib.csv"


def tsplib_matrix(info: dict) -> np.ndarray:
    """Distance matrix under the instance's declared EDGE_WEIGHT_TYPE.

    The parser already returns one for ATT, GEO and EXPLICIT. For the native
    Euclidean types it deliberately returns ``None`` (the benchmark's estimators
    take coordinates), so the rounding rule is applied here -- and it must be
    applied, because the published optimum counts rounded edges.
    """
    # Indexed, not .get(): a renamed key must raise here, not silently fall
    # through to the Euclidean branch. It already did once -- the parser calls
    # this "distance_matrix", an earlier draft asked for "dist_matrix", and
    # every EXPLICIT/GEO/ATT instance was quietly misrouted and lost.
    matrix = info["distance_matrix"]
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float64)

    coords = info["raw_coords"]
    if coords is None:
        raise ValueError("instance has neither a distance matrix nor coordinates")
    X = np.asarray(coords, dtype=np.float64)
    # (a-b)^2 expansion, not a broadcast difference: the latter allocates an
    # (n, n, d) temporary, which is 1 GiB at n = 8000 before the matrix itself.
    sq = (X * X).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    np.sqrt(D, out=D)
    ewt = info["edge_weight_type"].upper()
    if ewt in ("EUC_2D", "EUC_3D"):
        D = np.rint(D)          # TSPLIB nint
    elif ewt in ("CEIL_2D", "CEIL_3D"):
        D = np.ceil(D)
    else:
        raise NotImplementedError(f"no rounding rule wired for {ewt}")
    np.fill_diagonal(D, 0.0)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--max-n", type=int, default=8000)
    args = ap.parse_args()

    opt = pd.read_csv(OPTIMA)
    optima = dict(zip(opt["instance"].astype(str), opt["optimum"].astype(float)))

    rows = []
    files = sorted(INSTANCE_DIR.glob("*.tsp"))
    print(f"{len(files)} TSPLIB files, budget k={args.budget}, max_n={args.max_n}")
    for f in files:
        name = f.stem
        if name not in optima:
            continue
        try:
            info = parse_tsplib_file(str(f))
        except Exception as exc:
            rows.append({"instance": name, "status": f"parse_error:{type(exc).__name__}"})
            continue
        n = int(info["n"])
        ewt = info["edge_weight_type"].upper()
        if n > args.max_n:
            rows.append({"instance": name, "n": n, "edge_weight_type": ewt,
                         "status": "too_large_for_dense_matrix"})
            continue
        try:
            D = tsplib_matrix(info)
            t0 = time.perf_counter()
            res = one_tree_bound_from_matrix(D, args.budget)
            dt = time.perf_counter() - t0
        except Exception as exc:
            rows.append({"instance": name, "n": n, "edge_weight_type": ewt,
                         "status": f"{type(exc).__name__}:{exc}"})
            continue
        o = optima[name]
        rows.append({"instance": name, "n": n, "edge_weight_type": ewt,
                     "optimum": o, "bound": res.bound, "w0": res.initial_bound,
                     "gap_pct": 100.0 * (o - res.bound) / o,
                     "is_optimal": res.is_optimal, "seconds": dt, "status": "ok"})
        print(f"  {name:<12} n={n:<6} {ewt:<9} opt={o:<12.1f} bound={res.bound:<14.3f} "
              f"gap={100*(o-res.bound)/o:7.4f}%  {dt:7.2f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ok = df[df.status == "ok"]
    print(f"\nScored {len(ok)} of {len(df)} ({(df.status != 'ok').sum()} skipped/errored)")
    viol = ok[ok.gap_pct < -1e-9]
    print(f"\nVIOLATIONS (bound > published exact optimum): {len(viol)}")
    if len(viol):
        w = viol.loc[viol.gap_pct.idxmin()]
        print(f"  worst: {w.instance} bound={w.bound:.6f} optimum={w.optimum:.1f} "
              f"({w.gap_pct:+.6f}%)")
        print(viol[["instance", "n", "edge_weight_type", "optimum", "bound", "gap_pct"]].to_string(index=False))
    else:
        print("  none -- every bound is at or below the published optimum.")
    print(f"\nGap below optimum, by edge_weight_type:")
    print(ok.groupby("edge_weight_type").gap_pct.agg(["size", "mean", "median", "min", "max"]).round(4).to_string())
    print(f"\nOverall: mean {ok.gap_pct.mean():.4f}%  median {ok.gap_pct.median():.4f}%  "
          f"min {ok.gap_pct.min():.4f}%  max {ok.gap_pct.max():.4f}%")
    print(f"Closed exactly: {int(ok.is_optimal.sum())}/{len(ok)}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
