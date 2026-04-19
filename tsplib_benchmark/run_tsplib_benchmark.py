"""
TSPLIB95 benchmark runner for the GART 3.0 (LGBM V3) estimator.

Pipeline
--------
For each TSPLIB instance in ``instances/``:

1. Parse the .tsp file (tsplib_parser.parse_tsplib_file).
2. If the instance is natively Euclidean (EUC_2D / CEIL_2D / ATT), feed the raw
   2D coordinates to the estimator directly. ATT is pseudo-Euclidean but is
   close enough to 2D Euclidean for the feature set; we use its native 2D
   coordinates.
3. Otherwise (GEO, EXPLICIT) run classical MDS on the TSPLIB distance matrix,
   auto-select a dimensionality (cap = ``MAX_MDS_DIM``), and pass the embedded
   coordinates to the estimator.
4. Compare the predicted tour length to the published optimum in
   ``ground_truth/optima.csv``.

The script never overwrites existing results: each run is written to a new
timestamped CSV under ``results/`` and a summary printed to stdout. Old result
files are left untouched.

CLI
---
    python tsplib_benchmark/run_tsplib_benchmark.py [--max-n N] [--include-over-cap]

Options:
    --max-n N           Skip instances with more than N nodes. Default: no cap.
    --include-over-cap  Include instances whose node count exceeds the model's
                        training range (n > 1000). Default: included; use
                        ``--exclude-over-cap`` to run only in-range instances.
    --exclude-over-cap  Skip instances with n > 1000.
    --tag LABEL         Suffix appended to the output filename.
    --max-mds-dim K     Hard cap on MDS dimensionality (default 100).
    --workers N         Number of parallel worker processes (default: cpu_count - 1).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lgbm_model_v3"))
sys.path.insert(0, str(THIS_DIR))

from tsplib_parser import parse_tsplib_file  # noqa: E402
from classical_mds import classical_mds      # noqa: E402
from exclusions import TRIANGLE_INEQ_VIOLATORS  # noqa: E402
from lgbm_estimator_v3 import (              # noqa: E402
    TSP_V3_LGBM_Estimator,
    _fast_centroid_stats,
    compute_mst_degrees,
)

from collections import deque              # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: E402
from scipy import stats                    # noqa: E402


INSTANCES_DIR = THIS_DIR / "instances"
GROUND_TRUTH_FILE = THIS_DIR / "ground_truth" / "optima.csv"
RESULTS_DIR = THIS_DIR / "results"
TRAINING_MAX_N = 1000     # Upper bound of n during training.
TRAINING_MAX_DIM = 50     # Upper bound of feature dimensionality during training.
DEFAULT_MAX_MDS_DIM = 100


# ---------------------------------------------------------------------------
# Shared estimator for ThreadPoolExecutor
# ---------------------------------------------------------------------------
_worker_estimator = None


def _worker_init():
    """Lazily initialize the shared LightGBM estimator. Idempotent — safe to
    call from every thread; only the first call actually loads the model."""
    global _worker_estimator
    if _worker_estimator is None:
        _worker_estimator = TSP_V3_LGBM_Estimator(str(REPO_ROOT / "lgbm_model_v3"))


def _hybrid_estimate(estimator, original_dist_matrix, mds_coords, d_feat):
    """Hybrid feature computation for non-Euclidean instances.

    Computes MST features (20 features) from the *original* TSPLIB distance
    matrix so the MST scale is correct. Computes geometric/centroid features
    (7 features) from the MDS-embedded coordinates so the model gets
    approximate spatial structure. Metadata features (n, d) are set from the
    MDS embedding.

    This avoids the distance-inflation problem where classical MDS on a
    non-Euclidean matrix produces embeddings with inflated pairwise distances,
    causing the estimator's internally-computed Euclidean MST to be 2-4x too
    long.

    Returns the same dict schema as TSP_V3_LGBM_Estimator.estimate().
    """
    n = original_dist_matrix.shape[0]
    coords = mds_coords.astype(np.float32)

    t0 = time.perf_counter()

    feats = {"n_customers": n, "dimension": d_feat}

    # --- Geometric features from MDS embedding (7 features) ---
    rngs = np.ptp(coords, axis=0).astype(float)
    rngs[rngs < 1e-9] = 1e-9
    log_hv = np.sum(np.log(rngs))
    hypervolume = np.exp(min(log_hv, 690.0))
    feats["bounding_hypervolume"] = hypervolume
    feats["node_density"] = n / hypervolume if hypervolume > 1e-15 else 0.0
    feats["aspect_ratio"] = np.max(rngs) / np.min(rngs)

    cent = np.mean(coords, axis=0, dtype=np.float32)
    c_mn, c_st, c_mx, c_raw = _fast_centroid_stats(coords, cent)
    feats["centroid_dist_mean"] = c_mn
    feats["centroid_dist_std"] = c_st
    feats["centroid_dist_max"] = c_mx
    feats["centroid_dist_iqr"] = float(np.subtract(*np.percentile(c_raw, [75, 25])))

    # --- MST features from ORIGINAL distance matrix (20 features) ---
    D = original_dist_matrix.astype(np.float64)
    np.fill_diagonal(D, 0)
    mst_csr = minimum_spanning_tree(D)
    edges = mst_csr.data
    mst_len = float(np.sum(edges))

    feats["mst_total_length"] = mst_len
    feats["mst_edge_mean"] = float(np.mean(edges))
    feats["mst_edge_std"] = float(np.std(edges))
    feats["mst_edge_skew"] = float(stats.skew(edges))
    feats["mst_edge_kurtosis"] = float(stats.kurtosis(edges))
    feats["mst_edge_max"] = float(np.max(edges))

    percs = np.percentile(edges, [10, 25, 50, 75, 90])
    for i, p in enumerate([10, 25, 50, 75, 90]):
        feats[f"mst_edge_q{p}"] = float(percs[i])

    # Clustering proxies
    k_dom = max(1, int(np.sqrt(n)))
    feats["mst_dominance_ratio"] = float(
        np.sum(np.partition(edges, -k_dom)[-k_dom:]) / (mst_len + 1e-9)
    )
    feats["mst_gap_ratio"] = float(feats["mst_edge_max"] / (percs[2] + 1e-9))

    rows, cols = mst_csr.nonzero()
    degrees = compute_mst_degrees(
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        n,
    )
    feats["mst_leaf_ratio"] = float(np.sum(degrees == 1) / n)
    feats["mst_degree_mean"] = float(np.mean(degrees))
    feats["mst_degree_std"] = float(np.std(degrees))
    feats["mst_degree_max"] = int(np.max(degrees))
    feats["large_edge_count"] = int(
        np.sum(edges > feats["mst_edge_mean"] + feats["mst_edge_std"])
    )

    # Diameter via two BFS passes on the MST
    adj = [[] for _ in range(n)]
    for i in range(len(rows)):
        adj[rows[i]].append((cols[i], edges[i]))
        adj[cols[i]].append((rows[i], edges[i]))

    def _farthest(start):
        dists = np.full(n, -1.0)
        dists[start] = 0.0
        q = deque([start])
        fn, md = start, 0.0
        while q:
            u = q.popleft()
            if dists[u] > md:
                md, fn = dists[u], u
            for v, w in adj[u]:
                if dists[v] < 0:
                    dists[v] = dists[u] + w
                    q.append(v)
        return fn, md

    node1, _ = _farthest(0)
    _, diam = _farthest(node1)
    feats["mst_diameter"] = float(diam)
    feats["mst_diameter_normalized"] = float(diam / (mst_len + 1e-9))

    t_feat = time.perf_counter() - t0

    # --- Inference ---
    t1 = time.perf_counter()
    df_input = pd.DataFrame(
        [{k: feats.get(k, 0.0) for k in estimator.features_required}]
    )
    alpha = float(estimator.model.predict(df_input)[0])
    alpha = np.clip(alpha, 1.0, 2.0)
    t_inf = time.perf_counter() - t1

    return {
        "estimate": float(alpha * mst_len),
        "alpha": float(alpha),
        "mst_length": mst_len,
        "feature_time": t_feat,
        "inference_time": t_inf,
    }


def _process_one_instance(args):
    """Worker function: parse, prep, and estimate a single TSPLIB instance.

    For native Euclidean instances, calls the standard estimator. For
    non-Euclidean instances (ATT, GEO, EXPLICIT), uses the hybrid path:
    MST features from the original distance matrix, geometric features from
    the MDS embedding.

    Returns either a result dict or a (name, skip_reason) tuple.
    """
    path, true_cost, max_mds_dim = args
    name = Path(path).stem
    global _worker_estimator

    try:
        t_parse0 = time.perf_counter()
        info = parse_tsplib_file(path)
        t_parse = time.perf_counter() - t_parse0
    except Exception as exc:
        return ("skip", name, f"parse error: {exc}")

    n = info["n"]
    is_native = info["is_native_euclidean"] and info["raw_coords"] is not None

    if is_native:
        # --- Standard path: estimator handles everything ---
        coords = info["raw_coords"].astype(np.float32)
        d_feat = coords.shape[1]
        mds_info = {
            "mode": "native",
            "chosen_dim": d_feat,
            "natural_dim": d_feat,
            "variance_retained": 1.0,
            "negative_eigvalue_mass": 0.0,
            "strain": 0.0,
        }
        try:
            t_est0 = time.perf_counter()
            res = _worker_estimator.estimate(coords, d_feat, grid_size=0)
            t_est = time.perf_counter() - t_est0
        except Exception as exc:
            return ("skip", name, f"estimate error: {exc}")

    else:
        # --- Hybrid path: MST from original matrix, geometry from MDS ---
        D_orig = info["distance_matrix"]
        try:
            t_prep0 = time.perf_counter()
            X, _eigs, mds_raw = classical_mds(
                D_orig, max_dim=max_mds_dim, variance_threshold=0.999
            )
            t_prep = time.perf_counter() - t_prep0
        except Exception as exc:
            return ("skip", name, f"mds error: {exc}")

        d_feat = X.shape[1]
        mds_info = dict(mds_raw)
        mds_info["mode"] = "hybrid"

        try:
            t_est0 = time.perf_counter()
            res = _hybrid_estimate(_worker_estimator, D_orig, X, d_feat)
            t_est = time.perf_counter() - t_est0
        except Exception as exc:
            return ("skip", name, f"hybrid estimate error: {exc}")

    pred = res["estimate"]
    gap = (pred - true_cost) / true_cost * 100.0
    over_dim = d_feat > TRAINING_MAX_DIM

    return ("ok", {
        "instance": name,
        "n": n,
        "edge_weight_type": info["edge_weight_type"],
        "mode": mds_info["mode"],
        "feature_dim": d_feat,
        "mds_natural_dim": mds_info["natural_dim"],
        "mds_variance_retained": mds_info["variance_retained"],
        "mds_negative_mass": mds_info["negative_eigvalue_mass"],
        "mds_strain": mds_info["strain"],
        "in_training_n_range": n <= TRAINING_MAX_N,
        "in_training_dim_range": d_feat <= TRAINING_MAX_DIM,
        "extrapolated": (n > TRAINING_MAX_N) or over_dim,
        "true_cost": true_cost,
        "pred_cost": pred,
        "alpha": res["alpha"],
        "mst_length": res["mst_length"],
        "gap_pct": gap,
        "abs_gap_pct": abs(gap),
        "parse_time_s": t_parse,
        "prep_time_s": 0.0 if is_native else t_prep,
        "feature_time_s": res["feature_time"],
        "inference_time_s": res["inference_time"],
        "total_est_time_s": t_est,
    })


def load_optima() -> dict:
    df = pd.read_csv(GROUND_TRUTH_FILE)
    return dict(zip(df["instance"].astype(str), df["optimum"].astype(float)))



def run_benchmark(
    exclude_over_cap: bool = False,
    max_n: int | None = None,
    tag: str = "",
    max_mds_dim: int = DEFAULT_MAX_MDS_DIM,
    workers: int | None = None,
) -> Path:
    """Run GART 3.0 on every TSPLIB .tsp file under ``instances/``.

    Uses ProcessPoolExecutor to parallelize across CPU cores. Each worker
    loads its own copy of the LightGBM model via the initializer, so the
    model object is never serialized across the process boundary.

    Returns the path to the newly-written results CSV.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    optima = load_optima()

    tsp_files = sorted(INSTANCES_DIR.glob("*.tsp"))
    if not tsp_files:
        raise SystemExit(
            f"No .tsp files in {INSTANCES_DIR}. Run download_tsplib.py first."
        )

    # --- Build work list (filter before dispatching to workers) ---
    work = []
    skipped = []
    for path in tsp_files:
        name = path.stem
        if name in TRIANGLE_INEQ_VIOLATORS:
            skipped.append((name, "triangle-inequality violator"))
            continue
        true_cost = optima.get(name)
        if true_cost is None:
            skipped.append((name, "no ground-truth optimum"))
            continue
        # Peek at file header for DIMENSION to apply n-filters cheaply
        # without full parse. Read first 20 lines.
        n_from_header = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.strip().upper().startswith("DIMENSION"):
                        n_from_header = int(line.split(":")[-1].strip().split()[0])
                        break
        except Exception:
            pass

        if n_from_header is not None:
            if max_n is not None and n_from_header > max_n:
                skipped.append((name, f"n={n_from_header} exceeds --max-n={max_n}"))
                continue
            if exclude_over_cap and n_from_header > TRAINING_MAX_N:
                skipped.append((name, f"n={n_from_header} > training cap {TRAINING_MAX_N}"))
                continue

        work.append((str(path), true_cost, max_mds_dim))

    n_workers = workers if workers else max(1, os.cpu_count() - 1)
    print(f"Dispatching {len(work)} instances across {n_workers} workers...")

    rows = []
    _worker_init()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one_instance, w): w for w in work}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="TSPLIB benchmark"
        ):
            result = future.result()
            if result[0] == "skip":
                skipped.append((result[1], result[2]))
            else:
                rows.append(result[1])

    # --- Persist results ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out_file = RESULTS_DIR / f"tsplib_results_{ts}{suffix}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)

    skip_file = RESULTS_DIR / f"tsplib_skipped_{ts}{suffix}.csv"
    with open(skip_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "reason"])
        for name, reason in skipped:
            w.writerow([name, reason])

    print_summary(df, skipped, out_file)
    return out_file


def print_summary(df: pd.DataFrame, skipped, out_file: Path):
    print()
    print("=" * 80)
    print(f" TSPLIB95 benchmark -- GART 3.0 (LGBM V3)")
    print("=" * 80)
    print(f"Result file    : {out_file}")
    print(f"Total runs     : {len(df)}")
    print(f"Skipped        : {len(skipped)}")
    if df.empty:
        return

    def _block(label: str, subset: pd.DataFrame):
        if subset.empty:
            print(f"{label:30s}: (no instances)")
            return
        mape = subset["abs_gap_pct"].mean()
        med  = subset["abs_gap_pct"].median()
        p90  = subset["abs_gap_pct"].quantile(0.90)
        mx   = subset["abs_gap_pct"].max()
        bias = subset["gap_pct"].mean()
        t = (subset["feature_time_s"] + subset["inference_time_s"]).mean() * 1000
        print(
            f"{label:30s}: n={len(subset):3d}  "
            f"MAPE={mape:6.3f}%  med={med:6.3f}%  p90={p90:6.3f}%  "
            f"max={mx:6.2f}%  bias={bias:+.3f}%  lat={t:6.2f}ms"
        )

    print()
    print("--- Overall ---")
    _block("ALL", df)
    _block("In training range (n<=1000)", df[df["in_training_n_range"]])
    _block("Extrapolated (n>1000)", df[~df["in_training_n_range"]])

    print()
    print("--- By edge-weight type ---")
    for ewt in sorted(df["edge_weight_type"].unique()):
        _block(ewt, df[df["edge_weight_type"] == ewt])

    print()
    print("--- By mode (native vs hybrid-MDS) ---")
    for mode in sorted(df["mode"].unique()):
        _block(mode, df[df["mode"] == mode])

    print()
    print("Top 10 largest |gap|:")
    top = df.nlargest(10, "abs_gap_pct")[
        ["instance", "n", "edge_weight_type", "mode", "true_cost", "pred_cost", "gap_pct"]
    ]
    print(top.to_string(index=False))
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None,
                    help="Hard ceiling on n; larger instances are skipped.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--include-over-cap", action="store_true",
                       help="Include instances with n > 1000 (default).")
    group.add_argument("--exclude-over-cap", action="store_true",
                       help="Exclude instances with n > 1000.")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional label appended to result filenames.")
    ap.add_argument("--max-mds-dim", type=int, default=DEFAULT_MAX_MDS_DIM,
                    help=f"Hard cap on MDS dimensionality (default {DEFAULT_MAX_MDS_DIM}).")
    ap.add_argument("--workers", type=int, default=None,
                    help="Number of parallel worker processes (default: cpu_count - 1).")
    args = ap.parse_args()

    run_benchmark(
        exclude_over_cap=args.exclude_over_cap,
        max_n=args.max_n,
        tag=args.tag,
        max_mds_dim=args.max_mds_dim,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
