"""
Supplemental baseline runner for the 7 TSPLIB EUC_2D instances that were
skipped by the original all-models run because they have n > 5000.

For each missing instance, compute BHH, Cavdar, Chien, MST_Ratio, Hilbert
predictions using the same formulas as `tsp_utils_2.py`, but with a
Delaunay-based O(n)-edge MST for MST_Ratio so memory is bounded.

Appends rows to `tsplib_benchmark/results/all_models_tsplib_supplemental.csv`
with the same schema as `all_models_tsplib.csv`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from tsplib_parser import parse_tsplib_file
import tsp_utils_2 as academic
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay

MISSING = [
    "brd14051", "d15112", "d18512",
    "rl11849", "rl5915", "rl5934", "usa13509",
]


def delaunay_mst_length(coords: np.ndarray) -> float:
    """Compute MST length using Delaunay edge set — O(n) edges in 2D."""
    n = coords.shape[0]
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
    rows, cols, dists = [], [], []
    for a, b in edges:
        d = float(np.linalg.norm(coords[a] - coords[b]))
        rows.append(a); cols.append(b); dists.append(d)
        rows.append(b); cols.append(a); dists.append(d)
    sp = csr_matrix((dists, (rows, cols)), shape=(n, n))
    mst = minimum_spanning_tree(sp)
    return float(mst.sum())


def mst_ratio_delaunay(coords: np.ndarray):
    t0 = time.perf_counter()
    coords_u = np.unique(coords, axis=0)
    n = len(coords_u)
    if n <= 1:
        return 0.0, 0.0, 0.0
    d = coords_u.shape[1]
    mst_len = delaunay_mst_length(coords_u)
    if d == 2:
        ratio = 1.075
    elif d == 3:
        ratio = 1.05
    else:
        ratio = 1.0 + (0.075 * (2.0 / d))
    return float(mst_len * ratio), time.perf_counter() - t0, float(mst_len)


def run_missing(ground_truth: dict) -> pd.DataFrame:
    rows = []
    skipped = []
    for name in MISSING:
        path = THIS_DIR / "instances" / f"{name}.tsp"
        if not path.exists():
            print(f"  SKIP {name}: .tsp not found")
            skipped.append((name, "not_downloaded"))
            continue
        try:
            info = parse_tsplib_file(str(path))
        except Exception as exc:
            print(f"  SKIP {name}: parse error {exc}")
            skipped.append((name, "parse_error"))
            continue
        if info["edge_weight_type"] != "EUC_2D":
            print(f"  SKIP {name}: not EUC_2D")
            skipped.append((name, "not_EUC_2D"))
            continue
        coords = info["raw_coords"].astype(np.float32)
        n = info["n"]
        true_cost = ground_truth.get(name)
        if true_cost is None:
            print(f"  SKIP {name}: no optimum")
            skipped.append((name, "no_optimum"))
            continue

        print(f"  Running {name} (n={n}) ...")

        # MST_Ratio via Delaunay
        try:
            pred, tsec, mst_len = mst_ratio_delaunay(coords)
            rows.append(make_row(name, n, "MST_Ratio", pred, true_cost, tsec, mst_len))
            print(f"    MST_Ratio: pred={pred:.1f} in {tsec:.3f}s, mst={mst_len:.1f}")
        except Exception as exc:
            print(f"    MST_Ratio FAILED: {exc}")
            mst_len = None

        # BHH
        try:
            pred, tsec = academic.estimate_tsp_bhh(coords)
            rows.append(make_row(name, n, "BHH", pred, true_cost, tsec, mst_len))
            print(f"    BHH: pred={pred:.1f} in {tsec:.3f}s")
        except Exception as exc:
            print(f"    BHH FAILED: {exc}")

        # Cavdar
        try:
            pred, tsec = academic.estimate_tsp_cavdar(coords)
            rows.append(make_row(name, n, "Cavdar", pred, true_cost, tsec, mst_len))
            print(f"    Cavdar: pred={pred:.1f} in {tsec:.3f}s")
        except Exception as exc:
            print(f"    Cavdar FAILED: {exc}")

        # Chien
        try:
            pred, tsec = academic.estimate_tsp_chien(coords)
            rows.append(make_row(name, n, "Chien", pred, true_cost, tsec, mst_len))
            print(f"    Chien: pred={pred:.1f} in {tsec:.3f}s")
        except Exception as exc:
            print(f"    Chien FAILED: {exc}")

        # Hilbert
        try:
            pred, tsec = academic.estimate_tsp_hilbert(coords)
            rows.append(make_row(name, n, "Hilbert", pred, true_cost, tsec, mst_len))
            print(f"    Hilbert: pred={pred:.1f} in {tsec:.3f}s")
        except Exception as exc:
            print(f"    Hilbert FAILED: {exc}")

    df = pd.DataFrame(rows)
    print("\nSkipped:", skipped)
    return df


def make_row(name, n, model, pred, true_cost, tsec, mst_len):
    gap = (pred - true_cost) / true_cost * 100.0
    return {
        "instance": name,
        "n": n,
        "edge_weight_type": "EUC_2D",
        "model": model,
        "pred_cost": float(pred),
        "true_cost": float(true_cost),
        "gap_pct": float(gap),
        "abs_gap_pct": abs(float(gap)),
        "total_time_s": float(tsec),
        "feature_time_s": float(tsec),
        "inference_time_s": 0.0,
        "mst_length": float(mst_len) if mst_len is not None else None,
        "alpha": None,
        "concorde_time_s": None,
        "speedup_vs_concorde": None,
        "mode": "native",
        "feature_dim": 2,
    }


def load_optima():
    df = pd.read_csv(THIS_DIR / "ground_truth" / "optima.csv")
    return dict(zip(df["instance"].astype(str), df["optimum"].astype(float)))


def main():
    gt = load_optima()
    df = run_missing(gt)
    out = THIS_DIR / "results" / "all_models_tsplib_supplemental.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
