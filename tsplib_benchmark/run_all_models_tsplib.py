"""
Run ALL available TSP estimation models on TSPLIB95 instances.

For each TSPLIB instance, runs every available model and records:
- Predicted cost
- Wall-clock time (feature extraction + inference)
- Gap vs optimum (%)
- Speedup vs Concorde solve time (where available)

Models tested:
  ML models:    LGBM_V3, Linear_V3, Interp_V3, GART_1.0
  Academic:     BHH, Cavdar, Chien, Vinel, Composite, MST_Ratio, Hilbert
  Baselines:    Fixed_Alpha (optimal alpha=1.136 from TSPLIB fit)

For native 2D Euclidean instances, the LGBM estimator uses Delaunay
triangulation for O(n)-edge MST construction, enabling instances up to
n=85,900+.  Non-Euclidean instances (GEO, ATT, EXPLICIT) use hybrid MDS
with Delaunay-accelerated MST when the MDS embedding is 2D.

Usage:
    python tsplib_benchmark/run_all_models_tsplib.py [--max-n N] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lgbm_model_v3"))
sys.path.insert(0, str(REPO_ROOT / "linear_model_v3"))
sys.path.insert(0, str(REPO_ROOT / "interpretable_model_v3"))
sys.path.insert(0, str(THIS_DIR))

# Core imports
from tsplib_parser import parse_tsplib_file
from classical_mds import classical_mds

# ML model imports
from lgbm_model_v3.lgbm_estimator_v3 import TSP_V3_LGBM_Estimator
from linear_model_v3.estimator_linear_v3 import TSP_V3_Linear_Estimator
from interpretable_model_v3.estimator_interpretable_v3 import TSP_Interpretable_Estimator

# Academic estimator imports
import tsp_utils_2 as academic
import joblib

# Hybrid estimation support
from lgbm_estimator_v3 import _fast_centroid_stats, compute_mst_degrees
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from scipy import stats
from collections import deque

INSTANCES_DIR = THIS_DIR / "instances"
GROUND_TRUTH_FILE = THIS_DIR / "ground_truth" / "optima.csv"
CONCORDE_TIMES_FILE = THIS_DIR / "concorde_solve_times.csv"
RESULTS_DIR = THIS_DIR / "results"
MAX_MDS_DIM = 100

# Native Euclidean types that can use academic estimators directly
_NATIVE_2D_TYPES = {"EUC_2D", "CEIL_2D"}

# Optimal fixed alpha from TSPLIB fitting (minimizes MAPE)
FIXED_ALPHA = 1.136


class GART_Adapter:
    """Adapter for GART 1.0 legacy model."""
    def __init__(self, model_dir):
        p = Path(model_dir) / "alpha_predictor_model.joblib"
        self.model = joblib.load(p)

    def estimate(self, coordinates, dimension, grid_size):
        cost, t_feat, t_inf = academic.estimate_tsp_ml_alpha(coordinates, self.model)
        return {"estimate": cost, "feature_time": t_feat, "inference_time": t_inf}


def _hybrid_estimate_generic(estimator, original_dist_matrix, mds_coords, d_feat):
    """Hybrid feature computation for non-Euclidean instances.

    MST features (20) from original distance matrix.
    Geometric features (7) from MDS embedding.
    Works for any V3 ML model with model.predict().
    """
    n = original_dist_matrix.shape[0]
    coords = mds_coords.astype(np.float32)

    t0 = time.perf_counter()

    feats = {"n_customers": n, "dimension": d_feat}

    # Geometric features from MDS embedding
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

    # MST features from ORIGINAL distance matrix
    # Use Delaunay triangulation for 2D embeddings (O(n) edges vs O(n^2))
    if d_feat == 2 and n >= 4:
        try:
            tri = Delaunay(coords)
            edges_set = set()
            for simplex in tri.simplices:
                for ii in range(3):
                    for jj in range(ii + 1, 3):
                        a, b = simplex[ii], simplex[jj]
                        if a > b:
                            a, b = b, a
                        edges_set.add((a, b))
            rows_d, cols_d, dists_d = [], [], []
            for a, b in edges_set:
                # Use original distance matrix for edge weights (preserves true scale)
                dist_ab = float(original_dist_matrix[a, b])
                rows_d.append(a); cols_d.append(b); dists_d.append(dist_ab)
                rows_d.append(b); cols_d.append(a); dists_d.append(dist_ab)
            sparse_graph = csr_matrix((dists_d, (rows_d, cols_d)), shape=(n, n))
            mst_csr = minimum_spanning_tree(sparse_graph)
        except Exception:
            # Fallback to dense matrix if Delaunay fails
            D = original_dist_matrix.astype(np.float64)
            np.fill_diagonal(D, 0)
            mst_csr = minimum_spanning_tree(D)
    else:
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

    # Diameter via two BFS passes
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

    # Inference
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


def load_optima():
    df = pd.read_csv(GROUND_TRUTH_FILE)
    return dict(zip(df["instance"].astype(str), df["optimum"].astype(float)))


def load_concorde_times():
    if CONCORDE_TIMES_FILE.exists():
        df = pd.read_csv(CONCORDE_TIMES_FILE)
        return dict(zip(df["instance"].astype(str), df["concorde_time_s"].astype(float)))
    return {}


def run_benchmark(max_n=None, workers=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    optima = load_optima()
    concorde_times = load_concorde_times()

    tsp_files = sorted(INSTANCES_DIR.glob("*.tsp"))
    if not tsp_files:
        raise SystemExit(f"No .tsp files in {INSTANCES_DIR}. Run download_tsplib.py first.")

    # --- Load ML models ---
    print("Loading ML models...")
    ml_models = {}
    try:
        ml_models["LGBM_V3"] = TSP_V3_LGBM_Estimator(str(REPO_ROOT / "lgbm_model_v3"))
        print("  LGBM_V3: OK")
    except Exception as e:
        print(f"  LGBM_V3: FAILED ({e})")
    try:
        ml_models["Linear_V3"] = TSP_V3_Linear_Estimator(str(REPO_ROOT / "linear_model_v3"))
        print("  Linear_V3: OK")
    except Exception as e:
        print(f"  Linear_V3: FAILED ({e})")
    try:
        ml_models["Interp_V3"] = TSP_Interpretable_Estimator(str(REPO_ROOT / "interpretable_model_v3"))
        print("  Interp_V3: OK")
    except Exception as e:
        print(f"  Interp_V3: FAILED ({e})")
    try:
        ml_models["GART_1.0"] = GART_Adapter(str(REPO_ROOT / "GART_1.0"))
        print("  GART_1.0: OK")
    except Exception as e:
        print(f"  GART_1.0: FAILED ({e})")

    # Academic estimators (2D Euclidean only)
    academic_estimators = {
        "BHH": academic.estimate_tsp_bhh,
        "Cavdar": academic.estimate_tsp_cavdar,
        "Chien": academic.estimate_tsp_chien,
        "Vinel": academic.estimate_tsp_vinel,
        "Composite": academic.estimate_tsp_composite,
        "MST_Ratio": academic.estimate_tsp_mst_ratio,
        "Hilbert": academic.estimate_tsp_hilbert,
    }

    # --- Parse all instances ---
    print(f"\nParsing {len(tsp_files)} TSPLIB instances...")
    instances = []
    for path in tsp_files:
        name = path.stem
        true_cost = optima.get(name)
        if true_cost is None:
            continue

        # Quick n check from header
        n_from_header = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.strip().upper().startswith("DIMENSION"):
                        n_from_header = int(line.split(":")[-1].strip().split()[0])
                        break
        except Exception:
            pass

        if max_n is not None and n_from_header is not None and n_from_header > max_n:
            continue

        instances.append((path, name, true_cost, n_from_header))

    print(f"  {len(instances)} instances to benchmark\n")

    # --- Process each instance ---
    results = []
    for path, name, true_cost, n_hint in tqdm(instances, desc="Benchmarking"):
        # Parse
        try:
            t_parse0 = time.perf_counter()
            info = parse_tsplib_file(str(path))
            t_parse = time.perf_counter() - t_parse0
        except Exception as exc:
            print(f"  SKIP {name}: parse error: {exc}")
            continue

        n = info["n"]
        ewt = info["edge_weight_type"]
        is_native = info["is_native_euclidean"] and info["raw_coords"] is not None
        is_2d = is_native and ewt in _NATIVE_2D_TYPES

        # Get coordinates for native instances
        coords_2d = None
        if is_native:
            coords_2d = info["raw_coords"].astype(np.float32)

        # MDS for non-native instances (compute once, reuse)
        mds_coords = None
        mds_dim = None
        D_orig = None
        if not is_native:
            D_orig = info["distance_matrix"]
            try:
                X, _eigs, mds_raw = classical_mds(D_orig, max_dim=MAX_MDS_DIM)
                mds_coords = X
                mds_dim = X.shape[1]
            except Exception as exc:
                print(f"  SKIP {name}: MDS error: {exc}")
                continue

        # Compute MST length once for fixed-alpha baseline
        mst_length = None

        # === Run each model ===

        # 1. ML models (work on all instances via hybrid)
        for model_name, model_obj in ml_models.items():
            # GART 1.0 only works on 2D Euclidean; skip large instances (dense MST)
            if model_name == "GART_1.0" and (not is_2d or n > 5000):
                continue

            try:
                t0 = time.perf_counter()
                if is_native:
                    d_feat = coords_2d.shape[1]
                    res = model_obj.estimate(coords_2d, d_feat, grid_size=0)
                elif model_name == "GART_1.0":
                    continue  # Already filtered above
                elif hasattr(model_obj, "features_required"):
                    res = _hybrid_estimate_generic(model_obj, D_orig, mds_coords, mds_dim)
                else:
                    continue
                total_time = time.perf_counter() - t0

                pred = res["estimate"]
                if mst_length is None:
                    mst_length = res.get("mst_length", None)

                gap = (pred - true_cost) / true_cost * 100.0
                concorde_t = concorde_times.get(name, None)
                speedup = concorde_t / total_time if concorde_t else None

                results.append({
                    "instance": name,
                    "n": n,
                    "edge_weight_type": ewt,
                    "model": model_name,
                    "pred_cost": pred,
                    "true_cost": true_cost,
                    "gap_pct": gap,
                    "abs_gap_pct": abs(gap),
                    "total_time_s": total_time,
                    "feature_time_s": res.get("feature_time", 0.0),
                    "inference_time_s": res.get("inference_time", 0.0),
                    "mst_length": res.get("mst_length", None),
                    "alpha": res.get("alpha", None),
                    "concorde_time_s": concorde_t,
                    "speedup_vs_concorde": speedup,
                    "mode": "native" if is_native else "hybrid",
                    "feature_dim": mds_dim if not is_native else (coords_2d.shape[1] if coords_2d is not None else 2),
                })
            except Exception as exc:
                print(f"  {model_name} FAILED on {name}: {exc}")

        # 2. Fixed-alpha baseline (uses MST from any previous model)
        if mst_length is not None:
            pred_fixed = FIXED_ALPHA * mst_length
            gap_fixed = (pred_fixed - true_cost) / true_cost * 100.0
            results.append({
                "instance": name,
                "n": n,
                "edge_weight_type": ewt,
                "model": "Fixed_Alpha",
                "pred_cost": pred_fixed,
                "true_cost": true_cost,
                "gap_pct": gap_fixed,
                "abs_gap_pct": abs(gap_fixed),
                "total_time_s": 0.0,  # negligible
                "feature_time_s": 0.0,
                "inference_time_s": 0.0,
                "mst_length": mst_length,
                "alpha": FIXED_ALPHA,
                "concorde_time_s": concorde_times.get(name, None),
                "speedup_vs_concorde": None,
                "mode": "native" if is_native else "hybrid",
                "feature_dim": mds_dim if not is_native else (coords_2d.shape[1] if coords_2d is not None else 2),
            })

        # 3. Academic estimators (2D Euclidean only, skip for n>5000 to avoid O(n^2) blowup)
        if is_2d and coords_2d is not None and n <= 5000:
            for est_name, est_func in academic_estimators.items():
                try:
                    t0 = time.perf_counter()
                    pred, est_time = est_func(coords_2d)
                    total_time = time.perf_counter() - t0

                    gap = (pred - true_cost) / true_cost * 100.0
                    concorde_t = concorde_times.get(name, None)
                    speedup = concorde_t / total_time if concorde_t else None

                    results.append({
                        "instance": name,
                        "n": n,
                        "edge_weight_type": ewt,
                        "model": est_name,
                        "pred_cost": pred,
                        "true_cost": true_cost,
                        "gap_pct": gap,
                        "abs_gap_pct": abs(gap),
                        "total_time_s": total_time,
                        "feature_time_s": est_time,
                        "inference_time_s": 0.0,
                        "mst_length": mst_length,
                        "alpha": None,
                        "concorde_time_s": concorde_t,
                        "speedup_vs_concorde": speedup,
                        "mode": "native",
                        "feature_dim": 2,  # academic estimators are 2D native only
                    })
                except Exception as exc:
                    print(f"  {est_name} FAILED on {name}: {exc}")

    # --- Save results (fixed filename, overwrites previous run) ---
    out_path = RESULTS_DIR / "all_models_tsplib.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} results to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ALL-MODELS TSPLIB BENCHMARK SUMMARY")
    print("=" * 70)

    # Exclude brg180 from summary
    df_clean = df[df.instance != "brg180"]

    for model in df_clean["model"].unique():
        sub = df_clean[df_clean["model"] == model]
        mape = sub["abs_gap_pct"].mean()
        median = sub["abs_gap_pct"].median()
        count = len(sub)
        avg_time = sub["total_time_s"].mean() * 1000  # ms
        print(f"  {model:15s}  n={count:3d}  MAPE={mape:6.2f}%  median={median:5.2f}%  avg_time={avg_time:8.1f}ms")

    print("=" * 70)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all models on TSPLIB95")
    parser.add_argument("--max-n", type=int, default=None, help="Skip instances > N nodes")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(max_n=args.max_n, workers=args.workers)
