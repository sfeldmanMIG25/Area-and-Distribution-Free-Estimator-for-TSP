"""Clean single-process cost benchmark for the candidate feature groups.

The in-run timings in ``feature_screen.py`` are measured inside a 20-way
process pool and are inflated by memory-bandwidth contention. This script
measures the same work uncontended, and additionally measures a k-reduced
``local_id`` variant (only the neighbourhood sizes a shortlist actually needs),
since the module recomputes the local-PCA spectrum once per k.

Run:  python paper_tooling/feature_screen_cost_bench.py [k1,k2,...]
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from features_ext import group_degeneracy, group_local_id, group_mst_topology  # noqa: E402
from mst_utils import compute_mst  # noqa: E402

REPEATS = 5
CASES = [(100, 2), (1000, 2), (1000, 10), (1000, 50), (1000, 100)]


def timeit(fn, *args):
    best = np.inf
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def main():
    keep_k = None
    if len(sys.argv) > 1:
        keep_k = tuple(int(x) for x in sys.argv[1].split(","))

    rng = np.random.default_rng(0)
    rows = []
    for n, d in CASES:
        X = np.unique(rng.uniform(0, 1000, (n, d)), axis=0)
        mst = compute_mst(X)
        row = {
            "n": n, "d": d,
            "mst_existing_pipeline_ms": timeit(compute_mst, X),
            "local_id_full_ms": timeit(group_local_id.compute, X),
            "degeneracy_ms": timeit(group_degeneracy.compute, X),
            "mst_topology_ms": timeit(lambda c: group_mst_topology.compute(c, mst), X),
        }
        if keep_k:
            k_all, lb_all = group_local_id._K_VALUES, group_local_id._LB_K_VALUES
            try:
                group_local_id._K_VALUES = keep_k
                group_local_id._KMAX = max(keep_k)
                group_local_id._LB_K_VALUES = tuple(k for k in lb_all if k in keep_k)
                row[f"local_id_k{'_'.join(map(str, keep_k))}_ms"] = timeit(
                    group_local_id.compute, X)
            finally:
                group_local_id._K_VALUES = k_all
                group_local_id._KMAX = max(k_all)
                group_local_id._LB_K_VALUES = lb_all
        rows.append(row)

    df = pd.DataFrame(rows).set_index(["n", "d"])
    df["all_three_groups_ms"] = df[["local_id_full_ms", "degeneracy_ms",
                                    "mst_topology_ms"]].sum(axis=1)
    print(f"best of {REPEATS}, single process, single-threaded BLAS\n")
    print(df.round(2).to_string())
    out = os.path.join(ROOT, "paper_tooling", "feature_screen_cost_bench.csv")
    df.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
