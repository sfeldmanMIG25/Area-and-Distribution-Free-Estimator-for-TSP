"""
Run ALL available TSP estimation models on TSPLIB95 instances.

For each TSPLIB instance, runs every available model and records:
- Predicted cost
- Wall-clock time (feature extraction + inference)
- Gap vs optimum (%)
- Speedup vs Concorde solve time (where available)

Models tested:
  ML models:    LGBM_V3, Linear_V3, Interp_V3, GART_1.0
  Academic:     BHH, Cavdar--Sokol, MST_Only, MST_Ratio, Hilbert
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# Cap the native thread pools before numpy / LightGBM / torch initialise theirs.
# The instance loop is run on a thread pool, so without this every worker would
# spawn its own BLAS/OpenMP team and oversubscribe the machine. Set as defaults
# so an explicit shell value still wins.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lgbm_model_v3"))
sys.path.insert(0, str(REPO_ROOT / "lgbm_model_v4"))
sys.path.insert(0, str(REPO_ROOT / "linear_model_v3"))
sys.path.insert(0, str(REPO_ROOT / "interpretable_model_v3"))
sys.path.insert(0, str(REPO_ROOT / "nn_est_alpha_v3"))
sys.path.insert(0, str(THIS_DIR))

# Core imports
from feature_engineering_gart2 import TRAIN_GREEDY_RANGE, greedy_ratio_from_matrix
from tsplib_parser import parse_tsplib_file
from classical_mds import classical_mds
from exclusions import filter_metric_consistent, METRIC_RATIO_THRESHOLD

# ML model imports
from lgbm_model_v3.lgbm_estimator_v3 import TSP_V3_LGBM_Estimator
from lgbm_model_v3.lgbm_estimator_gart2 import TSP_GART2_Estimator
from lgbm_model_v4.lgbm_estimator_v4 import TSP_V4_LGBM_Estimator
from linear_model_v3.estimator_linear_v3 import TSP_V3_Linear_Estimator
from nn_est_alpha_v3.estimator_v3 import TSP_V3_Neural_Estimator
from interpretable_model_v3.estimator_interpretable_v3 import TSP_Interpretable_Estimator

# Academic estimator imports
import tsp_utils_2 as academic
import classical_region_estimators as region_est
from baselines_calibrated import AsymptoticMSTRatio, CalibratedMSTRatio
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

# --- Dense-allocation admission control -------------------------------------
# Every estimator in this suite bounds its own memory except GART 1.0:
#   * mst_utils.compute_mst caps the dense attempt at MST_DENSE_BUDGET_GB and
#     falls back to Delaunay (d<=3) or blocked Prim, both O(n) memory;
#   * feature_engineering_gart2 uses the dense greedy kernel only up to
#     DENSE_CAP = 3000 points, then switches to cKDTree candidate lists;
#   * lgbm_model_v4/feature_engineering keeps its own ~0.5 GB dense budget.
# GART 1.0 has no such gate. tsp_utils_2._calculate_gart_features runs an
# unconditional ``cdist(coords, coords)`` -- a dense n x n float64 matrix,
# 8*n^2 bytes -- solely to take a per-row minimum for the 1-NN statistics.
# On pla85900 (n = 85,900, every coordinate distinct) that single allocation is
# 54.98 GiB, and it happens inside a worker thread while the other workers hold
# their own instances. The instance list is sorted by filename, so at 16
# workers pla33810 (8.52 GiB), pla7397 (0.41 GiB) and pla85900 (54.98 GiB) are
# co-scheduled in the same wave: 63.9 GiB of dense distance matrix before the
# interpreter, the seven loaded models, or any other worker is counted. That is
# the hang -- the machine pages, it does not raise MemoryError.
#
# The hazard is one model's allocation on large instances, not the worker
# count, so the guard is scoped to that allocation rather than to the pool. A
# GART 1.0 call whose dense matrix would exceed GART1_DENSE_SERIALIZE_BYTES
# takes an exclusive lock; every other call, and every small instance, is
# unguarded and still runs pool-wide. Over the 82 native-2D TSPLIB instances,
# the ones below the 256 MiB default (n <= 5,792) sum to 0.75 GiB for the whole
# corpus, so the worst-case resident dense total is bounded by
#     max_i 8*n_i^2  +  sum_{8*n_j^2 <= T} 8*n_j^2  =  54.98 + 0.75 GiB
# at any worker count -- strictly less than the 63.5 GiB that --workers 4 can
# still reach, and it leaves 101 of the 111 instances running 16-wide.
# Override with TSPLIB_GART1_SERIALIZE_MB, mirroring MST_DENSE_BUDGET_GB.
GART1_DENSE_SERIALIZE_BYTES = int(
    float(os.environ.get("TSPLIB_GART1_SERIALIZE_MB", "256")) * (1 << 20)
)
_GART1_BIG_LOCK = threading.Lock()


def _dense_guard(model_name, n):
    """Exclusive lock for a GART 1.0 call big enough to matter; else a no-op.

    ``n`` is the instance dimension from the header. GART 1.0 deduplicates
    coordinates first, so the real matrix is at most 8*n^2 bytes -- the guard
    errs toward serialising.
    """
    if model_name == "GART_1.0" and 8 * n * n > GART1_DENSE_SERIALIZE_BYTES:
        return _GART1_BIG_LOCK
    return nullcontext()


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
    # Match feature_creator_v3 exactly. The model consumes all four of these;
    # omitting the two log features fed them to LightGBM as 0.0 on every
    # non-Euclidean instance, and computing node_density from the clipped
    # hypervolume saturated it whenever the MDS dimension pushed log_hv past 690.
    log_hv = float(np.sum(np.log(rngs)))
    log_density = float(np.log(n) - log_hv)
    feats["log_bounding_hypervolume"] = log_hv
    feats["bounding_hypervolume"] = float(np.exp(min(log_hv, 690.0)))
    feats["log_node_density"] = log_density
    feats["node_density"] = float(np.exp(max(min(log_density, 690.0), -690.0)))
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

    # greedy_nn_over_mst (GART 2.0 only) -- computed from the TRUE distance
    # matrix, never from the MDS embedding. The embedding distorts the metric:
    # on gr666 an embedding-derived numerator gives a ratio of 3.44, far
    # outside the [1.035, 2.209] range the model was fitted on, whereas the
    # true-matrix value is 1.384 and in distribution. Both the greedy tour and
    # the MST denominator here come from the same original metric, so the
    # feature means exactly what it meant during training.
    out_of_range = None
    if "greedy_nn_over_mst" in estimator.features_required:
        ratio = greedy_ratio_from_matrix(original_dist_matrix, mst_length=mst_len)
        feats["greedy_nn_over_mst"] = ratio
        lo, hi = TRAIN_GREEDY_RANGE
        if not (np.isfinite(ratio) and lo <= ratio <= hi):
            out_of_range = ratio

    t_feat = time.perf_counter() - t0

    # Never-silent contract: an out-of-distribution feature yields a status
    # row, not a quietly extrapolated prediction.
    if out_of_range is not None:
        return {"status": f"greedy_ratio_out_of_range:{out_of_range:.4f}",
                "estimate": float("nan"), "alpha": float("nan"),
                "mst_length": mst_len, "feature_time": t_feat,
                "inference_time": 0.0}

    # Inference. Fail loudly on a missing feature instead of substituting 0.0,
    # which previously hid two absent features from every non-Euclidean row.
    missing = [k for k in estimator.features_required if k not in feats]
    if missing:
        raise KeyError(f"hybrid feature builder omitted required features: {missing}")
    t1 = time.perf_counter()
    if hasattr(estimator, "predict_alpha"):
        # GART 2.0 predicts logit(alpha - 1), so the deployed alpha is
        # 1 + sigmoid(z), not z. Delegate to the estimator so the inverse
        # transform is defined in exactly one place (the model wrapper) and the
        # hybrid path cannot drift from the native one. Clipping the raw z here
        # instead would have produced plausible-looking, wrong predictions.
        alpha = estimator.predict_alpha(feats)
    else:
        # V3 / V4 / Linear / Interp predict alpha directly.
        df_input = pd.DataFrame([{k: feats[k] for k in estimator.features_required}])
        alpha = float(np.clip(float(estimator.model.predict(df_input)[0]), 1.0, 2.0))
    t_inf = time.perf_counter() - t1

    return {
        "estimate": float(alpha * mst_len),
        "alpha": float(alpha),
        "mst_length": mst_len,
        "feature_time": t_feat,
        "inference_time": t_inf,
        "status": "ok",
    }


def load_optima():
    df = pd.read_csv(GROUND_TRUTH_FILE)
    return dict(zip(df["instance"].astype(str), df["optimum"].astype(float)))


def load_concorde_times():
    if CONCORDE_TIMES_FILE.exists():
        df = pd.read_csv(CONCORDE_TIMES_FILE)
        return dict(zip(df["instance"].astype(str), df["concorde_time_s"].astype(float)))
    return {}


def _status_row(name, n, ewt, model, status, mode=None, feat_dim=None):
    """Emit a placeholder row for an instance-model pair we could not score.
    Never-silent contract: if a pair is not in the output CSV with status='ok',
    it must appear with a status explaining why."""
    return {
        "instance": name,
        "n": n,
        "edge_weight_type": ewt,
        "model": model,
        "pred_cost": np.nan,
        "true_cost": np.nan,
        "gap_pct": np.nan,
        "abs_gap_pct": np.nan,
        "total_time_s": np.nan,
        "feature_time_s": np.nan,
        "inference_time_s": np.nan,
        "mst_length": np.nan,
        "alpha": np.nan,
        "concorde_time_s": np.nan,
        "speedup_vs_concorde": np.nan,
        "mode": mode,
        "feature_dim": feat_dim,
        "status": status,
    }


def _run_one_instance(task, ml_models, academic_estimators, all_models, concorde_times):
    """Score every model on a single TSPLIB instance.

    Returns the list of row dicts for this instance. Pure with respect to the
    caller: it holds no cross-instance state, so the instances can be run
    concurrently and the rows re-assembled in the original order.
    """
    path, name, true_cost, n_hint = task
    rows = []

    # Parse
    try:
        t_parse0 = time.perf_counter()
        info = parse_tsplib_file(str(path))
        t_parse = time.perf_counter() - t_parse0
    except Exception as exc:
        print(f"  SKIP {name}: parse error: {exc}")
        for m in all_models:
            rows.append(_status_row(name, n_hint or 0, None, m, f"parse_error:{type(exc).__name__}"))
        return rows

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
            for m in all_models:
                rows.append(_status_row(name, n, ewt, m, f"mds_error:{type(exc).__name__}", mode="hybrid"))
            return rows

    # Compute MST length once for fixed-alpha baseline
    mst_length = None

    # === Run each model ===

    # 1. ML models (work on all instances via hybrid)
    for model_name, model_obj in ml_models.items():
        # GART 1.0 only works on 2D Euclidean (its feature set assumes 2D);
        # non-2D EWT is a correctness gate (different distance units), not
        # a performance gate, so it stays. The n>5000 gate is lifted for
        # consistency with the rest of the benchmark suite.
        if model_name == "GART_1.0" and not is_2d:
            rows.append(_status_row(name, n, ewt, model_name, "gart1_only_2d",
                                    mode="native" if is_native else "hybrid"))
            continue

        try:
            # The clock starts inside the guard, so a serialised GART 1.0 call
            # records its own runtime and not the time it spent queueing.
            with _dense_guard(model_name, n):
                t0 = time.perf_counter()
                if is_native:
                    d_feat = coords_2d.shape[1]
                    res = model_obj.estimate(coords_2d, d_feat, grid_size=0)
                elif model_name == "GART_1.0":
                    continue  # Already filtered above
                elif hasattr(model_obj, "features_required"):
                    res = _hybrid_estimate_generic(model_obj, D_orig, mds_coords, mds_dim)
                else:
                    rows.append(_status_row(name, n, ewt, model_name, "no_hybrid_path",
                                            mode="hybrid"))
                    continue
                total_time = time.perf_counter() - t0

            res_status = res.get("status", "ok")
            if res_status != "ok":
                rows.append(_status_row(
                    name, n, ewt, model_name, res_status,
                    mode="native" if is_native else "hybrid",
                    feat_dim=mds_dim if not is_native else (
                        coords_2d.shape[1] if coords_2d is not None else 2)))
                continue

            pred = res["estimate"]
            if mst_length is None:
                mst_length = res.get("mst_length", None)

            gap = (pred - true_cost) / true_cost * 100.0
            concorde_t = concorde_times.get(name, None)
            speedup = concorde_t / total_time if concorde_t else None

            rows.append({
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
                "status": "ok",
            })
        except Exception as exc:
            print(f"  {model_name} FAILED on {name}: {exc}")
            rows.append(_status_row(name, n, ewt, model_name, f"exception:{type(exc).__name__}",
                                    mode="native" if is_native else "hybrid",
                                    feat_dim=mds_dim if not is_native else (coords_2d.shape[1] if coords_2d is not None else 2)))

    # 2. Fixed-alpha baseline (uses MST from any previous model)
    if mst_length is not None:
        pred_fixed = FIXED_ALPHA * mst_length
        gap_fixed = (pred_fixed - true_cost) / true_cost * 100.0
        rows.append({
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
            "status": "ok",
        })
    else:
        rows.append(_status_row(name, n, ewt, "Fixed_Alpha", "no_mst_length",
                                mode="native" if is_native else "hybrid"))

    # 3. Academic estimators (2D Euclidean only: EUC_2D + CEIL_2D).
    # Never-silent: when an academic baseline cannot run, still emit a status row.
    # Non-2D edge-weight types (ATT / GEO / EXPLICIT) are skipped because the
    # true-cost optimum is computed with a non-Euclidean distance function,
    # so a Euclidean Cavdar / BHH / MST prediction lives in different
    # units from the ground truth and the gap is meaningless. No n-gate: the
    # academic estimators are O(n log n) at worst (convex hull + rotating
    # calipers / MST), so the entire TSPLIB 2D set runs in seconds.
    for est_name, est_func in academic_estimators.items():
        if not (is_2d and coords_2d is not None):
            rows.append(_status_row(name, n, ewt, est_name, "academic_non_2d",
                                    mode="hybrid"))
            continue
        try:
            t0 = time.perf_counter()
            if hasattr(est_func, "estimate"):
                # Region-aware estimator: it reports an out-of-domain
                # instance rather than raising, so record that status
                # instead of a meaningless prediction.
                res = est_func.estimate(coords_2d, 2, None)
                if res.get("status", "ok") != "ok":
                    rows.append(_status_row(name, n, ewt, est_name,
                                            res["status"], mode="native", feat_dim=2))
                    continue
                pred = res["estimate"]
                est_time = res.get("feature_time", 0.0) + res.get("inference_time", 0.0)
            else:
                pred, est_time = est_func(coords_2d)
            total_time = time.perf_counter() - t0

            gap = (pred - true_cost) / true_cost * 100.0
            concorde_t = concorde_times.get(name, None)
            speedup = concorde_t / total_time if concorde_t else None

            rows.append({
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
                "feature_dim": 2,
                "status": "ok",
            })
        except Exception as exc:
            print(f"  {est_name} FAILED on {name}: {exc}")
            rows.append(_status_row(name, n, ewt, est_name, f"exception:{type(exc).__name__}",
                                    mode="native", feat_dim=2))

    return rows


def run_benchmark(max_n=None, workers=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    optima = load_optima()
    concorde_times = load_concorde_times()

    tsp_files = sorted(INSTANCES_DIR.glob("*.tsp"))
    if not tsp_files:
        raise SystemExit(f"No .tsp files in {INSTANCES_DIR}. Run download_tsplib.py first.")

    # Every instance is run end-to-end. Non-metric outliers are dropped at
    # aggregate-metric time via filter_metric_consistent (true_cost/MST ratio
    # threshold = METRIC_RATIO_THRESHOLD); see exclusions.py for rationale.

    # --- Load ML models ---
    print("Loading ML models...")
    ml_models = {}
    try:
        ml_models["LGBM_V3"] = TSP_V3_LGBM_Estimator(str(REPO_ROOT / "lgbm_model_v3"))
        print("  LGBM_V3: OK")
    except Exception as e:
        print(f"  LGBM_V3: FAILED ({e})")
    try:
        ml_models["GART_2.0"] = TSP_GART2_Estimator(str(REPO_ROOT / "lgbm_model_v3"))
        print("  GART_2.0: OK")
    except Exception as e:
        print(f"  GART_2.0: FAILED ({e})")
    try:
        ml_models["LGBM_V4"] = TSP_V4_LGBM_Estimator(str(REPO_ROOT / "lgbm_model_v4"))
        print("  LGBM_V4: OK")
    except Exception as e:
        print(f"  LGBM_V4: FAILED ({e})")
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
        ml_models["NN_V3"] = TSP_V3_Neural_Estimator(str(REPO_ROOT / "nn_est_alpha_v3"))
        print("  NN_V3: OK")
    except Exception as e:
        print(f"  NN_V3: FAILED ({e})")
    try:
        ml_models["GART_1.0"] = GART_Adapter(str(REPO_ROOT / "GART_1.0"))
        print("  GART_1.0: OK")
    except Exception as e:
        print(f"  GART_1.0: FAILED ({e})")

    # Academic estimators (2D Euclidean only).
    # TSPLIB defines no sampling region, so BHH runs in 'hull' mode: the convex
    # hull of the instance is the documented plug-in for the region measure the
    # theorem assumes. Cavdar--Sokol needs no region at all -- it rotates each
    # instance into its minimum-area enclosing rectangle and takes the area from
    # there, which is what its source prescribes.
    #
    # Chien, Kwon--Golden--Wasil and Daganzo are gone, together with their
    # *_extrap rows. Their coefficients came from a secondary transcription and
    # the primaries are paywalled with no obtainable open-access copy, so this
    # benchmark prints no number that depends on them. Do not re-add.
    academic_estimators = {
        "BHH": region_est.ESTIMATORS["BHH"],
        "Cavdar": region_est.ESTIMATORS["Cavdar"],
        "MST_Only": region_est.ESTIMATORS["MST_Only"],
        "Asymptotic_MST": AsymptoticMSTRatio(),
        "Calibrated_MST_d": CalibratedMSTRatio("d"),
        "Calibrated_MST_dn": CalibratedMSTRatio("dn"),
        "MST_Ratio": academic.estimate_tsp_mst_ratio,
        "Hilbert": academic.estimate_tsp_hilbert,
    }

    # Full model universe — used for never-silent status logging. Every
    # (instance, model) pair must produce a row with a non-empty status.
    ALL_MODELS = list(ml_models.keys()) + ["Fixed_Alpha"] + list(academic_estimators.keys())

    # --- Parse all instances ---
    print(f"\nParsing {len(tsp_files)} TSPLIB instances...")
    results = []
    instances = []
    for path in tsp_files:
        name = path.stem
        true_cost = optima.get(name)
        n_from_header = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.strip().upper().startswith("DIMENSION"):
                        n_from_header = int(line.split(":")[-1].strip().split()[0])
                        break
        except Exception:
            pass

        if true_cost is None:
            # Never-silent: emit status rows for every model we would have tried.
            for m in ALL_MODELS:
                results.append(_status_row(name, n_from_header or 0, None, m, "no_optimum"))
            continue

        if max_n is not None and n_from_header is not None and n_from_header > max_n:
            for m in ALL_MODELS:
                results.append(_status_row(name, n_from_header, None, m, "max_n_cap"))
            continue

        instances.append((path, name, true_cost, n_from_header))

    print(f"  {len(instances)} instances to benchmark\n")

    # --- Process each instance (threaded) ---
    # Threads, not processes: the loaded LightGBM / sklearn / torch objects are
    # expensive to pickle, and the hot paths (MST, MDS, LightGBM predict) run in
    # native code that releases the GIL. Rows are collected per instance and
    # re-assembled in the original `instances` order so the CSV stays byte-stable.
    max_workers = workers if workers and workers > 0 else min(16, os.cpu_count() or 1)
    print(f"  using {max_workers} worker threads\n")

    per_instance = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one_instance, task, ml_models, academic_estimators,
                        ALL_MODELS, concorde_times): idx
            for idx, task in enumerate(instances)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Benchmarking"):
            per_instance[futures[fut]] = fut.result()

    for idx in range(len(instances)):
        results.extend(per_instance[idx])

    # --- Save results (fixed filename, overwrites previous run) ---
    out_path = RESULTS_DIR / "all_models_tsplib.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} results to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ALL-MODELS TSPLIB BENCHMARK SUMMARY")
    print("=" * 70)

    # Drop instances whose optimum exceeds METRIC_RATIO_THRESHOLD * MST, i.e.
    # instances mathematically incompatible with metric TSP (e.g. brg180).
    # Per-instance rows are still in ``df`` on disk; only aggregates below
    # use df_clean.
    df_clean = filter_metric_consistent(df)
    dropped = set(df["instance"].unique()) - set(df_clean["instance"].unique())
    if dropped:
        print(f"\nAggregate-metric filter dropped {len(dropped)} non-metric instance(s) "
              f"(true_cost/MST > {METRIC_RATIO_THRESHOLD}): {', '.join(sorted(dropped))}")

    # Per-status tally for visibility (never-silent principle)
    print("\nStatus breakdown across all (instance, model) pairs:")
    for st, cnt in df_clean["status"].value_counts().items():
        print(f"  {st:30s} {cnt}")
    print()

    for model in sorted(df_clean["model"].unique()):
        sub_all = df_clean[df_clean["model"] == model]
        sub_ok  = sub_all[sub_all["status"] == "ok"]
        mape    = sub_ok["abs_gap_pct"].mean() if len(sub_ok) else float("nan")
        median  = sub_ok["abs_gap_pct"].median() if len(sub_ok) else float("nan")
        count_ok = len(sub_ok)
        count_all = len(sub_all)
        avg_time = sub_ok["total_time_s"].mean() * 1000 if len(sub_ok) else float("nan")
        print(f"  {model:15s}  ok={count_ok:3d}/{count_all:3d}  MAPE={mape:6.2f}%  median={median:5.2f}%  avg_time={avg_time:8.1f}ms")

    print("=" * 70)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all models on TSPLIB95")
    parser.add_argument("--max-n", type=int, default=None, help="Skip instances > N nodes")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(max_n=args.max_n, workers=args.workers)
