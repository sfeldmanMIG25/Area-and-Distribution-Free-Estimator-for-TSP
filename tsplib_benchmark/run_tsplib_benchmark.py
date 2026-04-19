"""
TSPLIB95 benchmark runner — multi-model edition.

Pipeline
--------
For each TSPLIB instance in ``instances/``:

1. Parse the .tsp file (tsplib_parser.parse_tsplib_file).
2. If the instance is natively Euclidean (EUC_2D / CEIL_2D / ATT), use raw
   coordinates.
3. Otherwise (GEO, EXPLICIT) run classical MDS on the TSPLIB distance matrix
   once per instance (cap = ``MAX_MDS_DIM``), cache the embedded coordinates,
   and feed them to each estimator. For LGBM-kind estimators we use a hybrid
   path that computes MST features from the *original* distance matrix to
   avoid MDS-induced distance inflation. Generic estimators (NN, Linear,
   Interp) fall back to .estimate() on MDS coords (mode="mds_only").
4. Compare each estimator's prediction to the published optimum.

Output layout
-------------
    results/
      checkpoints/
        results_<model>.csv        # per-model checkpoint (enables partial regen)
      tsplib_results.csv           # canonical aggregated CSV with `model` column
      tsplib_skipped.csv           # canonical skip log

The canonical CSVs overwrite on every run. Older timestamped CSVs from earlier
runs are deleted once the new canonical files are written.

CLI
---
    python tsplib_benchmark/run_tsplib_benchmark.py [flags]

Options:
    --max-n N           Skip instances with more than N nodes.
    --include-over-cap  Include n > 1000 (default).
    --exclude-over-cap  Skip n > 1000.
    --max-mds-dim K     Hard cap on MDS dimensionality (default 100).
    --workers N         Parallel workers (default: cpu_count).
    --only M1,M2,...    Run only a subset of models (regen helper).
    --fresh             Delete per-model checkpoints before running.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lgbm_model_v3"))
sys.path.insert(0, str(REPO_ROOT / "lgbm_model_v4"))
sys.path.insert(0, str(REPO_ROOT / "linear_model_v3"))
sys.path.insert(0, str(REPO_ROOT / "nn_est_alpha_v3"))
sys.path.insert(0, str(REPO_ROOT / "interpretable_model_v3"))
sys.path.insert(0, str(THIS_DIR))

from tsplib_parser import parse_tsplib_file  # noqa: E402
from classical_mds import classical_mds      # noqa: E402
import tsp_utils_2 as academic               # noqa: E402
from lgbm_estimator_v3 import (              # noqa: E402
    TSP_V3_LGBM_Estimator,
    _fast_centroid_stats,
    compute_mst_degrees,
)
from lgbm_model_v4.lgbm_estimator_v4 import TSP_V4_LGBM_Estimator  # noqa: E402
from linear_model_v3.estimator_linear_v3 import TSP_V3_Linear_Estimator  # noqa: E402
from nn_est_alpha_v3.estimator_v3 import TSP_V3_Neural_Estimator  # noqa: E402
from interpretable_model_v3.estimator_interpretable_v3 import TSP_Interpretable_Estimator  # noqa: E402

from collections import deque              # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: E402
from scipy import stats                    # noqa: E402
from mst_utils import MSTResult            # noqa: E402


INSTANCES_DIR = THIS_DIR / "instances"
GROUND_TRUTH_FILE = THIS_DIR / "ground_truth" / "optima.csv"
RESULTS_DIR = THIS_DIR / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FINAL_RESULTS_FILE = RESULTS_DIR / "tsplib_results.csv"
FINAL_SKIPPED_FILE = RESULTS_DIR / "tsplib_skipped.csv"
TRAINING_MAX_N = 1000
TRAINING_MAX_DIM = 50
DEFAULT_MAX_MDS_DIM = 100

GART_FAMILY = {"Linear_V3", "LGBM_V3", "LGBM_V4", "NN_V3", "Interp_V3"}
EUC2D_TYPES = {"EUC_2D"}  # Scope gate for classical baselines.


# ---------------------------------------------------------------------------
# GART 1.0 legacy adapter (same class used by run_benchmark_2D_all.py)
# ---------------------------------------------------------------------------
class GART_Adapter:
    def __init__(self, model_dir):
        p = Path(model_dir) / "alpha_predictor_model.joblib"
        self.model = joblib.load(p)

    def estimate(self, coordinates, dimension, grid_size, precomputed_mst=None):
        cost, t_feat, t_inf = academic.estimate_tsp_ml_alpha(
            coordinates, self.model, precomputed_mst=precomputed_mst
        )
        return {"estimate": cost, "feature_time": t_feat, "inference_time": t_inf}


# ---------------------------------------------------------------------------
# Build an MSTResult from a precomputed distance matrix. compute_mst only
# accepts coordinates (it computes Euclidean distances internally), so for the
# hybrid path on non-native TSPLIB instances we wrap scipy's
# minimum_spanning_tree output into an MSTResult compatible with the
# estimator feature functions.
# ---------------------------------------------------------------------------
def _mst_from_distance_matrix(D):
    n = D.shape[0]
    D64 = D.astype(np.float64, copy=True)
    np.fill_diagonal(D64, 0.0)
    mst_csr = minimum_spanning_tree(D64)
    coo = mst_csr.tocoo()
    endpoints = np.stack([coo.row, coo.col], axis=1).astype(np.int32)
    edges = coo.data.astype(np.float32)
    degrees = np.zeros(n, dtype=np.int32)
    np.add.at(degrees, endpoints[:, 0], 1)
    np.add.at(degrees, endpoints[:, 1], 1)
    return MSTResult(n=n, edges=edges, endpoints=endpoints,
                     degrees=degrees, method="precomputed_dm")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def build_model_registry():
    """(name, factory, kind, scope).
      kind  : 'lgbm' (hybrid path) | 'generic' (.estimate on MDS) | 'classical' (coords->(pred,t))
      scope : 'all' (every instance) | 'euc2d' (EUC_2D only)
    """
    return [
        # --- Machine Learning Models (run on every instance) ---
        ("LGBM_V3",   lambda: TSP_V3_LGBM_Estimator(str(REPO_ROOT / "lgbm_model_v3")),   "lgbm",    "all"),
        ("LGBM_V4",   lambda: TSP_V4_LGBM_Estimator(str(REPO_ROOT / "lgbm_model_v4")),   "lgbm",    "all"),
        ("Linear_V3", lambda: TSP_V3_Linear_Estimator(str(REPO_ROOT / "linear_model_v3")), "generic", "all"),
        ("NN_V3",     lambda: TSP_V3_Neural_Estimator(str(REPO_ROOT / "nn_est_alpha_v3")), "generic", "all"),
        ("Interp_V3", lambda: TSP_Interpretable_Estimator(str(REPO_ROOT / "interpretable_model_v3")), "generic", "all"),
        ("GART",      lambda: GART_Adapter(str(REPO_ROOT / "GART_1.0")),                 "generic", "euc2d"),
        # --- Classical Academic Baselines (EUC_2D only) ---
        ("Cavdar",    lambda: academic.estimate_tsp_cavdar,    "classical", "euc2d"),
        ("BHH",       lambda: academic.estimate_tsp_bhh,       "classical", "euc2d"),
        ("MST_Ratio", lambda: academic.estimate_tsp_mst_ratio, "classical", "euc2d"),
        ("Chien",     lambda: academic.estimate_tsp_chien,     "classical", "euc2d"),
        ("Hilbert",   lambda: academic.estimate_tsp_hilbert,   "classical", "euc2d"),
        ("Kwon",      lambda: academic.estimate_tsp_kwon,      "classical", "euc2d"),
        ("Daganzo",   lambda: academic.estimate_tsp_daganzo,   "classical", "euc2d"),
    ]


# ---------------------------------------------------------------------------
# Hybrid feature computation (LGBM-kind on non-native instances)
# ---------------------------------------------------------------------------
def _hybrid_feature_vec(D_orig, mds_coords, d_feat):
    """Compute the 27-feature hybrid dict: MST from ORIGINAL distances,
    geometry from MDS embedding. Returns (feats, mst_len, feat_time_s)."""
    n = D_orig.shape[0]
    coords = mds_coords.astype(np.float32)
    t0 = time.perf_counter()

    feats = {"n_customers": n, "dimension": d_feat}

    rngs = np.ptp(coords, axis=0).astype(float)
    rngs[rngs < 1e-9] = 1e-9
    log_hv = float(np.sum(np.log(rngs)))
    hypervolume = np.exp(min(log_hv, 690.0))
    feats["bounding_hypervolume"] = hypervolume
    feats["node_density"] = n / hypervolume if hypervolume > 1e-15 else 0.0
    feats["aspect_ratio"] = float(np.max(rngs) / np.min(rngs))
    feats["log_bounding_hypervolume"] = log_hv
    feats["log_node_density"] = float(np.log(n) - log_hv)

    cent = np.mean(coords, axis=0, dtype=np.float32)
    c_mn, c_st, c_mx, c_raw = _fast_centroid_stats(coords, cent)
    feats["centroid_dist_mean"] = c_mn
    feats["centroid_dist_std"] = c_st
    feats["centroid_dist_max"] = c_mx
    feats["centroid_dist_iqr"] = float(np.subtract(*np.percentile(c_raw, [75, 25])))

    D = D_orig.astype(np.float64)
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
    feats["mst_dominance_ratio"] = float(np.sum(np.partition(edges, -k_dom)[-k_dom:]) / (mst_len + 1e-9))
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
    feats["large_edge_count"] = int(np.sum(edges > feats["mst_edge_mean"] + feats["mst_edge_std"]))

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
    return feats, mst_len, t_feat


def _lgbm_hybrid_predict(estimator, feats, mst_len):
    """Predict using an LGBM-style estimator from a precomputed feature dict."""
    t1 = time.perf_counter()
    df_input = pd.DataFrame([{k: feats.get(k, 0.0) for k in estimator.features_required}])
    alpha = float(estimator.model.predict(df_input)[0])
    alpha = float(np.clip(alpha, 1.0, 2.0))
    t_inf = time.perf_counter() - t1
    return {"estimate": float(alpha * mst_len), "alpha": alpha,
            "mst_length": mst_len, "inference_time": t_inf}


# ---------------------------------------------------------------------------
# Instance preparation (parse + optional MDS — once per instance)
# ---------------------------------------------------------------------------
def prepare_instance(path, max_mds_dim):
    t0 = time.perf_counter()
    info = parse_tsplib_file(path)
    t_parse = time.perf_counter() - t0

    n = info["n"]
    is_native = info["is_native_euclidean"] and info["raw_coords"] is not None

    prep = {
        "name": Path(path).stem,
        "n": n,
        "is_native": is_native,
        "edge_weight_type": info["edge_weight_type"],
        "parse_time": t_parse,
    }

    if is_native:
        raw = info["raw_coords"].astype(np.float32)
        prep["coords"] = raw
        prep["d_feat"] = raw.shape[1]
        prep["D_orig"] = None
        prep["mds_info"] = {"mode": "native", "chosen_dim": raw.shape[1],
                            "natural_dim": raw.shape[1], "variance_retained": 1.0,
                            "negative_eigvalue_mass": 0.0, "strain": 0.0}
        prep["prep_time"] = 0.0
    else:
        D = info["distance_matrix"]
        tp = time.perf_counter()
        X, _eigs, mds_raw = classical_mds(D, max_dim=max_mds_dim, variance_threshold=0.999)
        prep["prep_time"] = time.perf_counter() - tp
        prep["coords"] = X
        prep["d_feat"] = X.shape[1]
        prep["D_orig"] = D
        prep["mds_info"] = dict(mds_raw)
        prep["mds_info"]["mode"] = "hybrid"

    return prep


# ---------------------------------------------------------------------------
# Per-(instance, model) scoring
# ---------------------------------------------------------------------------
def _score_one(prep, estimator, model_name, kind, true_cost):
    t0 = time.perf_counter()
    status = "ok"
    alpha = float('nan')
    mst_len = float('nan')

    if kind == "classical":
        # Classical baselines: estimator is a bare callable coords -> (pred, time).
        # Kwon has a calibration range — record out-of-range as explicit status.
        if model_name == "Kwon" and prep["n"] > getattr(academic, "KWON_CALIBRATION_N_MAX", 300):
            status = "kwon_out_of_calibration"
            pred = float('nan')
            t_feat = t_inf = float('nan')
        else:
            pred_raw, t_total_cls = estimator(prep["coords"])
            pred = float(pred_raw)
            t_feat = t_inf = t_total_cls / 2.0
        res = {"estimate": pred, "feature_time": t_feat, "inference_time": t_inf}
        mode = "native"
    elif prep["is_native"]:
        res = estimator.estimate(prep["coords"], prep["d_feat"], grid_size=0)
        alpha = float(res.get("alpha", float('nan')))
        mst_len = float(res.get("mst_length", float('nan')))
        mode = "native"
    else:
        if kind == "lgbm":
            feats, mst_len, t_feat_hybrid = _hybrid_feature_vec(
                prep["D_orig"], prep["coords"], prep["d_feat"]
            )
            r = _lgbm_hybrid_predict(estimator, feats, mst_len)
            res = {"estimate": r["estimate"],
                   "feature_time": t_feat_hybrid,
                   "inference_time": r["inference_time"]}
            alpha = r["alpha"]
            mode = "hybrid"
        else:
            # Hybrid path for generic estimators (Linear_V3, NN_V3, Interp_V3,
            # GART) on non-native instances: compute MST from the ORIGINAL
            # distance matrix (once) and feed it via precomputed_mst so
            # coordinate-based features still use MDS coords but MST-derived
            # features reflect the true metric.
            mst_from_orig = _mst_from_distance_matrix(prep["D_orig"])
            res = estimator.estimate(prep["coords"], prep["d_feat"], grid_size=0,
                                     precomputed_mst=mst_from_orig)
            alpha = float(res.get("alpha", float('nan')))
            mst_len = float(res.get("mst_length", float(mst_from_orig.total_length)))
            mode = "hybrid"

    pred = float(res["estimate"]) if np.isfinite(res.get("estimate", float('nan'))) else float('nan')
    t_total = time.perf_counter() - t0
    gap = (pred - true_cost) / true_cost * 100.0 if (true_cost > 0 and np.isfinite(pred)) else float('nan')

    return {
        "model": model_name,
        "instance": prep["name"],
        "n": prep["n"],
        "edge_weight_type": prep["edge_weight_type"],
        "mode": mode,
        "status": status,
        "feature_dim": prep["d_feat"],
        "mds_natural_dim": prep["mds_info"].get("natural_dim"),
        "mds_variance_retained": prep["mds_info"].get("variance_retained"),
        "mds_negative_mass": prep["mds_info"].get("negative_eigvalue_mass"),
        "mds_strain": prep["mds_info"].get("strain"),
        "in_training_n_range": prep["n"] <= TRAINING_MAX_N,
        "in_training_dim_range": prep["d_feat"] <= TRAINING_MAX_DIM,
        "extrapolated": (prep["n"] > TRAINING_MAX_N) or (prep["d_feat"] > TRAINING_MAX_DIM),
        "true_cost": true_cost,
        "pred_cost": pred,
        "alpha": alpha,
        "mst_length": mst_len,
        "tsp_mst_ratio": (true_cost / mst_len) if (np.isfinite(mst_len) and mst_len > 0) else float('nan'),
        "parse_time_s": prep["parse_time"],
        "prep_time_s": prep["prep_time"],
        "feature_time_s": float(res.get("feature_time", 0.0)),
        "inference_time_s": float(res.get("inference_time", 0.0)),
        "total_est_time_s": t_total,
        "gap_pct": gap,
        "abs_gap_pct": abs(gap) if np.isfinite(gap) else float('nan'),
    }


# ---------------------------------------------------------------------------
# Work collection
# ---------------------------------------------------------------------------
def load_optima():
    df = pd.read_csv(GROUND_TRUTH_FILE)
    return dict(zip(df["instance"].astype(str), df["optimum"].astype(float)))


def collect_work(exclude_over_cap, max_n):
    tsp_files = sorted(INSTANCES_DIR.glob("*.tsp"))
    if not tsp_files:
        raise SystemExit(f"No .tsp files in {INSTANCES_DIR}. Run download_tsplib.py first.")
    optima = load_optima()
    work, skipped = [], []
    for path in tsp_files:
        name = path.stem
        true_cost = optima.get(name)
        if true_cost is None:
            skipped.append((name, "no ground-truth optimum"))
            continue
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
        work.append((str(path), true_cost))
    return work, skipped


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run_benchmark(exclude_over_cap=False, max_n=None, max_mds_dim=DEFAULT_MAX_MDS_DIM,
                  workers=None, only=None, fresh=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if fresh:
        for p in CHECKPOINT_DIR.glob("results_*.csv"):
            p.unlink()
            print(f"[fresh] removed {p.name}")

    work, skipped = collect_work(exclude_over_cap, max_n)
    print(f"--- TSPLIB: {len(work)} instances queued, {len(skipped)} pre-skipped ---")

    n_workers = workers if workers else max(1, os.cpu_count() or 2)

    # --- Prepare (parse + MDS) serially. Concurrent numpy.linalg.eigh on
    # large TSPLIB distance matrices is memory-heavy and unstable under
    # Python 3.14 + LAPACK; run one-at-a-time to avoid crashes. Model
    # scoring below still uses the full worker pool.
    prepared = []
    prep_skips = []
    for p, tc in tqdm(work, desc="prepare"):
        try:
            prep = prepare_instance(p, max_mds_dim)
            prepared.append((prep, tc))
        except Exception as exc:
            prep_skips.append((Path(p).stem, f"prepare error: {exc}"))
    skipped.extend(prep_skips)

    registry = build_model_registry()
    if only:
        wanted = {m.strip() for m in only.split(",")}
        registry = [(n, f, k, s) for n, f, k, s in registry if n in wanted]
        print(f"--- Running subset: {[n for n, _, _, _ in registry]} ---")

    # --- Per-model loop with checkpoint-and-skip ---
    for model_name, factory, kind, scope in registry:
        ckpt = CHECKPOINT_DIR / f"results_{model_name.lower()}.csv"
        if ckpt.exists() and not fresh:
            print(f"[SKIPPED] {model_name}: checkpoint exists at {ckpt.name}")
            continue

        # Apply scope gate — classical baselines restricted to EUC_2D only.
        if scope == "euc2d":
            scoped = [(p, tc) for p, tc in prepared if p["edge_weight_type"] in EUC2D_TYPES]
        else:
            scoped = prepared
        if not scoped:
            print(f"[SKIPPED] {model_name}: no instances match scope '{scope}'")
            continue

        print(f"[RUN] {model_name} ({kind}, scope={scope}, n={len(scoped)})")
        try:
            estimator = factory()
        except Exception as exc:
            print(f"    [ERROR] failed to load {model_name}: {exc}")
            continue

        rows = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_score_one, prep, estimator, model_name, kind, tc)
                    for prep, tc in scoped]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=model_name):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    rows.append({"model": model_name, "instance": "<error>",
                                 "error": str(exc), "abs_gap_pct": float('nan')})

        pd.DataFrame(rows).to_csv(ckpt, index=False)
        ok = [r for r in rows if np.isfinite(r.get("abs_gap_pct", float('nan')))]
        if ok:
            mape = float(np.mean([r["abs_gap_pct"] for r in ok]))
            print(f"    [SAVED] {model_name} | MAPE={mape:.3f}% | n_ok={len(ok)}/{len(rows)}")
        del estimator
        gc.collect()

    # --- Aggregate into canonical CSV + clean up legacy timestamped files ---
    csv_files = sorted(CHECKPOINT_DIR.glob("results_*.csv"))
    if not csv_files:
        print("[WARN] No per-model checkpoints to aggregate.")
        return None
    final_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    final_df.to_csv(FINAL_RESULTS_FILE, index=False)

    with open(FINAL_SKIPPED_FILE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "reason"])
        for name, reason in skipped:
            w.writerow([name, reason])

    removed = 0
    for p in RESULTS_DIR.glob("tsplib_results_*.csv"):
        p.unlink(); removed += 1
    for p in RESULTS_DIR.glob("tsplib_skipped_*.csv"):
        p.unlink(); removed += 1
    if removed:
        print(f"[cleanup] removed {removed} legacy timestamped CSVs")

    print_summary(final_df, skipped, FINAL_RESULTS_FILE)
    return FINAL_RESULTS_FILE


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(df, skipped, out_file):
    print()
    print("=" * 80)
    print(" TSPLIB95 benchmark -- multi-model")
    print("=" * 80)
    print(f"Result file    : {out_file}")
    print(f"Total rows     : {len(df)}")
    print(f"Skipped        : {len(skipped)}")
    if df.empty:
        return

    def _block(label, subset):
        if subset.empty:
            print(f"{label:34s}: (empty)")
            return
        mape = subset["abs_gap_pct"].mean()
        med = subset["abs_gap_pct"].median()
        p90 = subset["abs_gap_pct"].quantile(0.90)
        mx = subset["abs_gap_pct"].max()
        bias = subset["gap_pct"].mean()
        t = (subset["feature_time_s"] + subset["inference_time_s"]).mean() * 1000
        print(f"{label:34s}: n={len(subset):4d}  MAPE={mape:6.3f}%  med={med:6.3f}%  "
              f"p90={p90:6.3f}%  max={mx:6.2f}%  bias={bias:+.3f}%  lat={t:6.2f}ms")

    print("\n--- By model ---")
    for m in sorted(df["model"].astype(str).unique()):
        _block(m, df[df["model"] == m])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--include-over-cap", action="store_true")
    g.add_argument("--exclude-over-cap", action="store_true")
    ap.add_argument("--max-mds-dim", type=int, default=DEFAULT_MAX_MDS_DIM)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated model subset (regen helper).")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete per-model checkpoints before running.")
    args = ap.parse_args()

    run_benchmark(
        exclude_over_cap=args.exclude_over_cap,
        max_n=args.max_n,
        max_mds_dim=args.max_mds_dim,
        workers=args.workers,
        only=args.only,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
