"""Should the degenerate-geometry training augmentation be adopted?

Adapted from ``paper_tooling/augmentation_experiment.py``. That script recycled
half of the 2D benchmark's own ``line_noise`` instances as training data, which
measures recall, not generalisation. This one trains on a corpus of *different*
generators (collinear / subspace / curve / filament for the high-alpha regime,
lattice / hexlattice for the low-alpha regime) and evaluates on benchmark
families that were never touched.

    M0  standard training split only (must reproduce the shipped artifact)
    MA  standard training split + augment rows (TRAIN split only)

Both share the frozen hyperparameters in ``lgbm_model_v3/best_params_v3.json``,
seed 42, early stopping (100 rounds) on the untouched standard validation split,
and the preprocessing of ``lgbm_model_v3/LGBM_Alpha_Model_V3.py``.

Evaluation slices (all untouched by the augmentation):
    (a) 2D diverse benchmark, 2,580 instances, by generator class + ``grid``
    (b) ND test split, 16,920 rows of ``tsp_features_v3.csv``
    (c) TSPLIB EUC_2D, 78 instances (native path)
    (d) TSPLIB non-EUC_2D screened, 23 instances (hybrid MDS path)

Usage::

    python paper_tooling/augmentation_v2_experiment.py
    python paper_tooling/augmentation_v2_experiment.py --force
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
for _p in (ROOT_DIR, ROOT_DIR / "lgbm_model_v3", ROOT_DIR / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import feature_creator_v3 as fc  # noqa: E402
from tsp_utils_2 import canonicalize_coords_pca  # noqa: E402
from lgbm_model_v3.lgbm_estimator_v3 import (  # noqa: E402
    TSP_V3_LGBM_Estimator,
    _fast_centroid_stats,
    compute_mst_degrees,
)
from tsplib_parser import parse_tsplib_file  # noqa: E402
from classical_mds import classical_mds  # noqa: E402
from exclusions import STRUCTURAL_TRIANGLE_VIOLATORS  # noqa: E402

# --- paths ---------------------------------------------------------------
DATA_FILE = ROOT_DIR / "tsp_features_v3.csv"
PARAMS_FILE = ROOT_DIR / "lgbm_model_v3" / "best_params_v3.json"
SHIPPED_MODEL_FILE = ROOT_DIR / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib"

BENCH_2D_FEATURES = ROOT_DIR / "paper_tooling" / "augmentation_2d_features.csv"

AUG_DIR = ROOT_DIR / "augment"
# Order matters: later files supersede earlier ones on duplicate ``name``.
# ``repair_records.json`` re-solves 50 batch-1 instances with Concorde; those
# solution files were overwritten on disk, so the repaired metadata is the
# truthful one. ``batch2_records.json`` targets rho in [2, 100] (the mildly
# elongated regime where the benchmark's line_noise failures actually live).
AUG_RECORD_FILES = (
    AUG_DIR / "full_records.json",
    AUG_DIR / "repair_records.json",
    AUG_DIR / "batch2_records.json",
)
AUG_INSTANCES = AUG_DIR / "instances"
AUG_SOLUTIONS = AUG_DIR / "solutions"
AUG_FEATURES_CACHE = ROOT_DIR / "paper_tooling" / "augment_features_v3.csv"

TSPLIB_DIR = ROOT_DIR / "tsplib_benchmark"
TSPLIB_INSTANCES = TSPLIB_DIR / "instances"
TSPLIB_RESULTS = TSPLIB_DIR / "results" / "all_models_tsplib_repaired.csv"
TSPLIB_FEATURES_CACHE = ROOT_DIR / "paper_tooling" / "tsplib_features_v3.csv"

OUT_DIR = ROOT_DIR / "paper_tooling"
MODEL_DIR = OUT_DIR / "augmentation_v2_models"
RESULTS_FILE = OUT_DIR / "augmentation_v2_results.csv"
CRITERIA_FILE = OUT_DIR / "augmentation_v2_criteria.csv"
DIAG_FILE = OUT_DIR / "augmentation_v2_alpha_diagnostics.csv"

ADOPT_MODEL_FILE = ROOT_DIR / "lgbm_model_v3" / "lgbm_alpha_model_v3_augmented.joblib"
ADOPT_FEATURES_FILE = ROOT_DIR / "tsp_features_v3_augmented.csv"

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100
MAX_BOOST_ROUND = 3000
MAX_MDS_DIM = 100

GEN_CLASSES: dict[str, str] = {
    **{g: "Isotropic" for g in ("random", "normal", "triangular", "truncated_exponential")},
    **{g: "Biased" for g in ("squeezed_uniform", "uniform_triangular", "triangular_squeezed", "correlated")},
    **{g: "Geometric" for g in ("grid", "boundary", "x_central")},
    "clustered": "Clustered",
    "line_noise": "LineNoise",
}
GEN_RE = re.compile(r"^TSP-([a-z_0-9]+)-n(\d+)-g(\d+)")

# M0 must reproduce the shipped artifact exactly.
SHIPPED_BEST_ITERATION = 2031
SHIPPED_ND_TEST_MAPE = 0.8769


# =========================================================================
# feature extraction helpers
# =========================================================================
def _aug_extract_one(name: str) -> dict[str, object]:
    inst = fc.load_instance_data(f"{name}.json")
    if inst is None:
        raise FileNotFoundError(name)
    with (AUG_SOLUTIONS / f"{name}.sol.json").open() as fh:
        sol = json.load(fh)
    feats = fc.compute_features_for_instance_v3(inst, sol)
    if feats is None:
        raise ValueError(f"{name}: extractor returned None (n < 3)")
    return feats


def extract_augment_features(force: bool) -> pd.DataFrame:
    if AUG_FEATURES_CACHE.exists() and not force:
        df = pd.read_csv(AUG_FEATURES_CACHE)
        print(f"Cached augment features -> {AUG_FEATURES_CACHE.name} ({len(df)} rows)")
        return df

    merged: dict[str, dict] = {}
    for path in AUG_RECORD_FILES:
        with path.open() as fh:
            recs = json.load(fh)
        print(f"augment/{path.name}: {len(recs)} records")
        for rec in recs:
            rec["source_file"] = path.stem
            merged[rec["name"]] = rec
    print(f"  merged unique names: {len(merged)}")

    usable, dropped = [], {"integrity": 0, "no_solution": 0}
    for rec in merged.values():
        if not rec.get("integrity_ok"):
            dropped["integrity"] += 1
            continue
        if not (AUG_SOLUTIONS / f"{rec['name']}.sol.json").exists():
            dropped["no_solution"] += 1
            continue
        usable.append(rec)
    print(f"  usable={len(usable)}  dropped={dropped}")

    fc.INSTANCES_DIR = str(AUG_INSTANCES)
    fc.SOLUTIONS_DIR = str(AUG_SOLUTIONS)
    names = [r["name"] for r in usable]
    workers = max(1, os.cpu_count() or 1)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(tqdm(pool.map(_aug_extract_one, names), total=len(names), desc="augment features"))
    print(f"  extracted in {time.perf_counter() - t0:.1f}s")

    df = pd.DataFrame(rows)
    meta = pd.DataFrame(
        [{"instance_name": r["name"], "family": r["family"], "group": r["group"],
          "record_alpha": r["alpha"], "solver_used": r["solver_used"],
          "source_file": r["source_file"], "record_rho": r.get("rho")} for r in usable]
    )
    df = df.merge(meta, on="instance_name", how="left", validate="one_to_one")
    df.to_csv(AUG_FEATURES_CACHE, index=False)
    print(f"  cached -> {AUG_FEATURES_CACHE.name}")
    return df


def _hybrid_features(original_dist_matrix: np.ndarray, mds_coords: np.ndarray, d_feat: int) -> dict:
    """Verbatim port of run_all_models_tsplib._hybrid_estimate_generic's feature block."""
    n = original_dist_matrix.shape[0]
    coords = mds_coords.astype(np.float32)
    feats: dict[str, object] = {"n_customers": n, "dimension": d_feat}

    rngs = np.ptp(coords, axis=0).astype(float)
    rngs[rngs < 1e-9] = 1e-9
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
                dist_ab = float(original_dist_matrix[a, b])
                rows_d.append(a); cols_d.append(b); dists_d.append(dist_ab)
                rows_d.append(b); cols_d.append(a); dists_d.append(dist_ab)
            mst_csr = minimum_spanning_tree(csr_matrix((dists_d, (rows_d, cols_d)), shape=(n, n)))
        except Exception:
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
    feats["mst_dominance_ratio"] = float(np.sum(np.partition(edges, -k_dom)[-k_dom:]) / (mst_len + 1e-9))
    feats["mst_gap_ratio"] = float(feats["mst_edge_max"] / (percs[2] + 1e-9))

    rows, cols = mst_csr.nonzero()
    degrees = compute_mst_degrees(np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32), n)
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
    return feats


def _tsplib_extract_one(task: tuple[str, str, float]) -> dict[str, object]:
    name, ewt_expected, true_cost = task
    info = parse_tsplib_file(str(TSPLIB_INSTANCES / f"{name}.tsp"))
    is_native = info["is_native_euclidean"] and info["raw_coords"] is not None
    if is_native:
        coords = np.array(info["raw_coords"], dtype=np.float32)
        coords = canonicalize_coords_pca(coords).astype(np.float32, copy=False)
        est = _NATIVE_EST[0]
        feats, mst_len = est._compute_v3_features(coords, len(coords), coords.shape[1], 0)
        feats = dict(feats)
        mode, feat_dim = "native", int(coords.shape[1])
    else:
        D = info["distance_matrix"]
        X, _eigs, _mds = classical_mds(D, max_dim=MAX_MDS_DIM)
        feats = _hybrid_features(D, X, X.shape[1])
        mst_len = feats["mst_total_length"]
        mode, feat_dim = "hybrid", int(X.shape[1])
    feats["instance_name"] = name
    feats["edge_weight_type"] = info["edge_weight_type"]
    feats["mode"] = mode
    feats["feature_dim"] = feat_dim
    feats["mst_total_length"] = float(mst_len)
    feats["optimal_cost"] = float(true_cost)
    return feats


_NATIVE_EST: list[TSP_V3_LGBM_Estimator] = []


def extract_tsplib_features(force: bool) -> pd.DataFrame:
    if TSPLIB_FEATURES_CACHE.exists() and not force:
        df = pd.read_csv(TSPLIB_FEATURES_CACHE)
        print(f"Cached TSPLIB features -> {TSPLIB_FEATURES_CACHE.name} ({len(df)} rows)")
        return df

    res = pd.read_csv(TSPLIB_RESULTS, low_memory=False)
    v3 = res[(res["model"] == "LGBM_V3") & (res["status"] == "ok")].copy()
    print(f"TSPLIB LGBM_V3 ok rows: {len(v3)}")
    tasks = list(zip(v3["instance"].astype(str), v3["edge_weight_type"].astype(str), v3["true_cost"].astype(float)))

    _NATIVE_EST.clear()
    _NATIVE_EST.append(TSP_V3_LGBM_Estimator(str(ROOT_DIR / "lgbm_model_v3")))

    workers = max(1, min(8, os.cpu_count() or 1))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(tqdm(pool.map(_tsplib_extract_one, tasks), total=len(tasks), desc="TSPLIB features"))
    print(f"  extracted in {time.perf_counter() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(TSPLIB_FEATURES_CACHE, index=False)
    print(f"  cached -> {TSPLIB_FEATURES_CACHE.name}")
    return df


# =========================================================================
# shared plumbing
# =========================================================================
def attach_alpha(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    divisor = out["mst_total_length"].replace(0, 1e-9)
    out["alpha"] = (out["optimal_cost"] / divisor).clip(1.0, 2.0)
    if (out["optimal_cost"] <= 0).any():
        raise ValueError("optimal_cost contains non-positive values")
    return out


def build_features(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"frame missing model features: {missing}")
    return df[names]


def cost_metrics(model, X: pd.DataFrame, mst: np.ndarray, cost: np.ndarray) -> dict[str, float]:
    if len(X) == 0:
        raise ValueError("empty evaluation slice")
    best_it = getattr(model, "best_iteration_", None)
    pred_alpha = np.clip(model.predict(X, num_iteration=best_it), 1.0, 2.0)
    err = (pred_alpha * mst - cost) / cost
    return {
        "N": int(len(err)),
        "MAPE_pct": float(np.mean(np.abs(err)) * 100.0),
        "SDPE_pct": float(np.std(err, ddof=1) * 100.0) if len(err) > 1 else float("nan"),
        "MSPE_pct": float(np.mean(err) * 100.0),
        "MedAPE_pct": float(np.median(np.abs(err)) * 100.0),
    }


def alpha_regression(model, frame: pd.DataFrame, names: list[str]) -> dict[str, float]:
    best_it = getattr(model, "best_iteration_", None)
    pred = np.clip(model.predict(build_features(frame, names), num_iteration=best_it), 1.0, 2.0)
    true = frame["alpha"].to_numpy()
    slope, intercept = np.polyfit(true, pred, 1)
    return {
        "N": int(len(true)),
        "mean_true_alpha": float(np.mean(true)),
        "mean_pred_alpha": float(np.mean(pred)),
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": float(np.corrcoef(true, pred)[0, 1]),
    }


def fit_model(tag, X_tr, y_tr, X_val, y_val, params, force):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{tag}.joblib"
    if path.exists() and not force:
        print(f"[{tag}] cached -> {path.name}")
        return joblib.load(path)
    print(f"[{tag}] fitting on {len(X_tr)} train / {len(X_val)} val rows ...")
    t0 = time.perf_counter()
    model = lgb.LGBMRegressor(**params, n_estimators=MAX_BOOST_ROUND,
                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    joblib.dump(model, path)
    print(f"[{tag}] {(time.perf_counter() - t0) / 60:.2f} min, best_iteration={model.best_iteration_}")
    return model


# =========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-features", action="store_true")
    args = ap.parse_args()

    shipped = joblib.load(SHIPPED_MODEL_FILE)
    fnames = list(shipped.feature_name_)
    print(f"Model consumes {len(fnames)} features.\n")

    # ---------- slices ----------
    df2d = attach_alpha(pd.read_csv(BENCH_2D_FEATURES))
    if len(df2d) != 2580:
        raise SystemExit(f"expected 2580 2D benchmark rows, got {len(df2d)}")
    print(f"2D benchmark: {len(df2d)} rows, classes {df2d['gen_class'].value_counts().to_dict()}")

    corpus = attach_alpha(pd.read_csv(DATA_FILE))
    base_train = corpus[corpus["split"] == "train"]
    base_val = corpus[corpus["split"] == "val"]
    base_test = corpus[corpus["split"] == "test"]
    print(f"corpus: {len(corpus)} rows; train={len(base_train)} val={len(base_val)} test={len(base_test)}")

    aug = attach_alpha(extract_augment_features(args.force_features))
    print(f"augment: {len(aug)} rows; families {aug['family'].value_counts().to_dict()}")
    print(f"  by source file: {aug['source_file'].value_counts().to_dict()}")
    print(f"  by group: {aug['group'].value_counts().to_dict()}")
    print(f"  alpha range [{aug['alpha'].min():.4f}, {aug['alpha'].max():.4f}]")
    rho = aug["record_rho"]
    print(f"  rho: {int(rho.notna().sum())} with a target rho, range "
          f"[{rho.min():.3f}, {rho.max():.3f}]; {int(rho.isna().sum())} lattice/hexlattice (no rho)")
    rho_bins = pd.cut(rho, [0, 1, 2, 10, 50, 1e9], labels=["<=1", "1-2", "2-10", "10-50", ">50"])
    print(f"  rho buckets: {rho_bins.value_counts().sort_index().to_dict()}")
    print(f"  n buckets: {aug['n_customers'].value_counts().sort_index().to_dict()}")
    print(f"  n>=200 & alpha>=1.45: {int(((aug['n_customers'] >= 200) & (aug['alpha'] >= 1.45)).sum())}")
    print(f"  n>=200 & alpha<=1.05: {int(((aug['n_customers'] >= 200) & (aug['alpha'] <= 1.05)).sum())}")
    mild = aug[(aug["n_customers"] >= 200) & rho.between(2, 100)]
    print(f"  n>=200 & rho in [2,100]: {len(mild)} rows, alpha "
          f"[{mild['alpha'].min():.4f}, {mild['alpha'].max():.4f}]" if len(mild) else "  n>=200 & rho in [2,100]: 0")
    overlap = set(aug["instance_name"]) & set(corpus["instance_name"])
    if overlap:
        raise SystemExit(f"augment names collide with the corpus: {sorted(overlap)[:5]}")

    tsp = attach_alpha(extract_tsplib_features(args.force_features))
    tsp_euc = tsp[tsp["edge_weight_type"] == "EUC_2D"].copy()
    tsp_non = tsp[(tsp["edge_weight_type"] != "EUC_2D")
                  & (~tsp["instance_name"].isin(STRUCTURAL_TRIANGLE_VIOLATORS))].copy()
    print(f"TSPLIB: EUC_2D={len(tsp_euc)}  non-EUC screened={len(tsp_non)}")

    # ---------- verify the TSPLIB extraction reproduces the shipped run ----
    res = pd.read_csv(TSPLIB_RESULTS, low_memory=False)
    ref = res[(res["model"] == "LGBM_V3") & (res["status"] == "ok")][
        ["instance", "alpha", "gap_pct", "abs_gap_pct", "feature_dim"]
    ].rename(columns={"feature_dim": "feature_dim_csv", "alpha": "alpha_csv"})
    chk = tsp.merge(ref, left_on="instance_name", right_on="instance", validate="one_to_one").reset_index(drop=True)
    pred_alpha = np.clip(shipped.predict(build_features(chk, fnames)), 1.0, 2.0)
    dalpha = np.abs(pred_alpha - chk["alpha_csv"].to_numpy())
    nat = chk["mode"].to_numpy() == "native"
    print(f"TSPLIB extraction check vs shipped CSV: native max|dalpha|={dalpha[nat].max():.3e} "
          f"(N={int(nat.sum())}), hybrid max|dalpha|={dalpha[~nat].max():.3e} (N={int((~nat).sum())})")
    dim_mismatch = chk[chk["feature_dim"].astype(int) != chk["feature_dim_csv"].astype(int)]
    print(f"  embedding-dimension mismatches vs CSV: {len(dim_mismatch)}"
          + (f" -> {dim_mismatch['instance_name'].tolist()}" if len(dim_mismatch) else ""))
    if dalpha[nat].max() > 1e-9:
        worst = chk.loc[np.argsort(-np.where(nat, dalpha, -1))[:5], "instance_name"].tolist()
        raise SystemExit(f"native TSPLIB feature path does not reproduce the shipped run; worst: {worst}")
    if dalpha[~nat].max() > 1e-9:
        print("  NOTE: the hybrid MDS path is not bit-reproducible against the archived CSV "
              "(eigendecomposition round-off crossing LightGBM split thresholds). Both M0 and MA "
              "are scored on THIS extraction, so the comparison stays internally consistent.")
    for label, frame in (("EUC_2D", tsp_euc), ("nonEUC_screened", tsp_non)):
        m = cost_metrics(shipped, build_features(frame, fnames),
                         frame["mst_total_length"].to_numpy(), frame["optimal_cost"].to_numpy())
        print(f"  shipped {label}: N={m['N']} MAPE={m['MAPE_pct']:.4f} MSPE={m['MSPE_pct']:.4f}")

    # ---------- fit ----------
    with PARAMS_FILE.open() as fh:
        params = json.load(fh)
    print(f"\nfrozen hyperparameters: {params}")

    X_val, y_val = build_features(base_val, fnames), base_val["alpha"]
    X_base, y_base = build_features(base_train, fnames), base_train["alpha"]
    X_aug, y_aug = build_features(aug, fnames), aug["alpha"]
    X_plus = pd.concat([X_base, X_aug], ignore_index=True)
    y_plus = pd.concat([y_base, y_aug], ignore_index=True)

    m0 = fit_model("M0", X_base, y_base, X_val, y_val, params, args.force)
    ma = fit_model("MA", X_plus, y_plus, X_val, y_val, params, args.force)

    # ---------- M0 reproduction gate ----------
    X_test = build_features(base_test, fnames)
    mst_t, cost_t = base_test["mst_total_length"].to_numpy(), base_test["optimal_cost"].to_numpy()
    m0_test = cost_metrics(m0, X_test, mst_t, cost_t)
    ship_test = cost_metrics(shipped, X_test, mst_t, cost_t)
    max_pred_drift = float(np.max(np.abs(m0.predict(X_test) - shipped.predict(X_test))))
    print("\n--- M0 reproduction gate ---")
    print(f"  best_iteration: M0={m0.best_iteration_}  shipped={SHIPPED_BEST_ITERATION}")
    print(f"  ND test MAPE:   M0={m0_test['MAPE_pct']:.4f}  shipped-artifact={ship_test['MAPE_pct']:.4f}"
          f"  target={SHIPPED_ND_TEST_MAPE}")
    print(f"  max |pred_M0 - pred_shipped| over ND test = {max_pred_drift:.3e}")
    fails = []
    if int(m0.best_iteration_) != SHIPPED_BEST_ITERATION:
        fails.append(f"best_iteration {m0.best_iteration_} != {SHIPPED_BEST_ITERATION}")
    if abs(m0_test["MAPE_pct"] - SHIPPED_ND_TEST_MAPE) > 5e-4:
        fails.append(f"ND test MAPE {m0_test['MAPE_pct']:.4f} != {SHIPPED_ND_TEST_MAPE}")
    if max_pred_drift > 1e-9:
        fails.append(f"prediction drift {max_pred_drift:.3e} > 1e-9")
    if fails:
        raise SystemExit("M0 DOES NOT REPRODUCE THE SHIPPED ARTIFACT: " + "; ".join(fails))
    print("  OK: M0 is bit-identical to the shipped artifact.")

    # ---------- evaluate ----------
    slices: list[tuple[str, pd.DataFrame]] = [("2D_all", df2d)]
    for cls in ("Isotropic", "Biased", "Geometric", "Clustered", "LineNoise"):
        slices.append((f"2D_{cls}", df2d[df2d["gen_class"] == cls]))
    slices.append(("2D_grid_subgen", df2d[df2d["generator"] == "grid"]))
    slices.append(("ND_test", base_test))
    slices.append(("TSPLIB_EUC_2D", tsp_euc))
    slices.append(("TSPLIB_nonEUC_screened", tsp_non))

    rows = []
    for tag, model in (("M0", m0), ("MA", ma)):
        for sname, frame in slices:
            m = cost_metrics(model, build_features(frame, fnames),
                             frame["mst_total_length"].to_numpy(), frame["optimal_cost"].to_numpy())
            rows.append({"model": tag, "slice": sname, **{k: (v if k == "N" else round(v, 4))
                                                          for k, v in m.items()},
                         "n_trees": int(model.best_iteration_)})
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False)
    print("\n=== RESULTS (tour-cost percent error) ===")
    print(results[["model", "slice", "N", "MAPE_pct", "SDPE_pct", "MSPE_pct"]].to_string(index=False))

    def get(model_tag, sname, col):
        return float(results[(results["model"] == model_tag) & (results["slice"] == sname)][col].iloc[0])

    # ---------- LineNoise broken out by n bucket ----------
    ln = df2d[df2d["gen_class"] == "LineNoise"]
    buckets = [("n<=100", ln[ln["n_customers"] <= 100]),
               ("n 101-400", ln[(ln["n_customers"] > 100) & (ln["n_customers"] <= 400)]),
               ("n 401-1000", ln[(ln["n_customers"] > 400) & (ln["n_customers"] <= 1000)])]
    ln_rows = []
    for tag, model in (("M0", m0), ("MA", ma)):
        for bname, frame in buckets:
            m = cost_metrics(model, build_features(frame, fnames),
                             frame["mst_total_length"].to_numpy(), frame["optimal_cost"].to_numpy())
            ln_rows.append({"model": tag, "n_bucket": bname,
                            **{k: (v if k == "N" else round(v, 4)) for k, v in m.items()}})
    ln_df = pd.DataFrame(ln_rows)
    ln_df.to_csv(OUT_DIR / "augmentation_v2_linenoise_by_n.csv", index=False)
    print("\n=== 2D LineNoise by n bucket ===")
    print(ln_df[["model", "n_bucket", "N", "MAPE_pct", "SDPE_pct", "MSPE_pct"]].to_string(index=False))

    # ---------- alpha-slope diagnostic ----------
    ln_big = df2d[(df2d["gen_class"] == "LineNoise") & (df2d["n_customers"] >= 200)]
    diag = []
    for tag, model in (("M0", m0), ("MA", ma)):
        d = alpha_regression(model, ln_big, fnames)
        d["model"] = tag
        diag.append(d)
    diag_df = pd.DataFrame(diag)[["model", "N", "mean_true_alpha", "mean_pred_alpha", "slope", "intercept", "pearson_r"]]
    diag_df.to_csv(DIAG_FILE, index=False)
    print(f"\n=== ALPHA REGRESSION, 2D LineNoise n>=200 (N={len(ln_big)}) ===")
    print(diag_df.round(4).to_string(index=False))
    slope_m0 = float(diag_df[diag_df["model"] == "M0"]["slope"].iloc[0])
    slope_ma = float(diag_df[diag_df["model"] == "MA"]["slope"].iloc[0])

    # ---------- pre-registered criteria ----------
    crit = []

    def add(num, desc, value, ok, detail):
        crit.append({"criterion": num, "description": desc, "value": round(value, 4),
                     "verdict": "PASS" if ok else "FAIL", "detail": detail,
                     "disqualifying": num != 3})

    ln_mape_ma, ln_mspe_ma = get("MA", "2D_LineNoise", "MAPE_pct"), get("MA", "2D_LineNoise", "MSPE_pct")
    ln_mape_m0, ln_mspe_m0 = get("M0", "2D_LineNoise", "MAPE_pct"), get("M0", "2D_LineNoise", "MSPE_pct")
    add(1, "2D LineNoise MAPE < 5.00 (from 11.60)", ln_mape_ma, ln_mape_ma < 5.0,
        f"M0={ln_mape_m0:.4f} -> MA={ln_mape_ma:.4f}")
    add(2, "2D LineNoise MSPE within +/-3.0 of zero (from -11.58)", ln_mspe_ma, abs(ln_mspe_ma) <= 3.0,
        f"M0={ln_mspe_m0:.4f} -> MA={ln_mspe_ma:.4f}")

    g_m0, g_ma = get("M0", "2D_grid_subgen", "MSPE_pct"), get("MA", "2D_grid_subgen", "MSPE_pct")
    add(3, "grid sub-generator MSPE improves by >= 2.0 points (desirable)", g_m0 - g_ma, (g_m0 - g_ma) >= 2.0,
        f"M0={g_m0:.4f} -> MA={g_ma:.4f} (improvement {g_m0 - g_ma:.4f})")

    nd_m0, nd_ma = get("M0", "ND_test", "MAPE_pct"), get("MA", "ND_test", "MAPE_pct")
    add(4, "ND test MAPE regresses <= 0.05 points from 0.8769", nd_ma - SHIPPED_ND_TEST_MAPE,
        (nd_ma - SHIPPED_ND_TEST_MAPE) <= 0.05, f"M0={nd_m0:.4f} -> MA={nd_ma:.4f}")

    e_m0, e_ma = get("M0", "TSPLIB_EUC_2D", "MAPE_pct"), get("MA", "TSPLIB_EUC_2D", "MAPE_pct")
    add(5, "TSPLIB EUC_2D MAPE regresses <= 0.15 points from 3.271", e_ma - e_m0, (e_ma - e_m0) <= 0.15,
        f"M0={e_m0:.4f} -> MA={e_ma:.4f}")

    worst_cls, worst_delta = None, -1e9
    for cls in ("Isotropic", "Biased", "Clustered"):
        d = get("MA", f"2D_{cls}", "MAPE_pct") - get("M0", f"2D_{cls}", "MAPE_pct")
        if d > worst_delta:
            worst_delta, worst_cls = d, cls
    detail6 = "; ".join(
        f"{c}: {get('M0', f'2D_{c}', 'MAPE_pct'):.4f}->{get('MA', f'2D_{c}', 'MAPE_pct'):.4f}"
        for c in ("Isotropic", "Biased", "Clustered"))
    add(6, "no Isotropic/Biased/Clustered regression > 0.15 MAPE points", worst_delta, worst_delta <= 0.15,
        f"worst={worst_cls} (+{worst_delta:.4f}); {detail6}")

    ne_m0, ne_ma = get("M0", "TSPLIB_nonEUC_screened", "MAPE_pct"), get("MA", "TSPLIB_nonEUC_screened", "MAPE_pct")
    add(7, "TSPLIB non-EUC_2D screened MAPE regresses <= 0.30 points from 4.63", ne_ma - ne_m0,
        (ne_ma - ne_m0) <= 0.30, f"M0={ne_m0:.4f} -> MA={ne_ma:.4f}")

    add(8, "LineNoise alpha-slope (n>=200) >= 0.70 (from 0.28)", slope_ma, slope_ma >= 0.70,
        f"M0={slope_m0:.4f} -> MA={slope_ma:.4f}")

    crit_df = pd.DataFrame(crit)
    crit_df.to_csv(CRITERIA_FILE, index=False)
    print("\n=== PRE-REGISTERED CRITERIA ===")
    for c in crit:
        star = "" if c["disqualifying"] else "  (desirable only)"
        print(f"  {c['criterion']}. {c['verdict']:4s}  {c['description']}{star}\n"
              f"        {c['detail']}")

    hard_fail = [c["criterion"] for c in crit if c["verdict"] == "FAIL" and c["disqualifying"]]
    verdict = "REJECT" if hard_fail else "ADOPT"
    print(f"\n=== OVERALL VERDICT: {verdict} ===")
    if hard_fail:
        print(f"  disqualifying failures: {hard_fail}")

    # ---------- adoption artifacts ----------
    if verdict == "ADOPT":
        joblib.dump(ma, ADOPT_MODEL_FILE)
        print(f"  saved MA -> {ADOPT_MODEL_FILE}")
        corpus_raw = pd.read_csv(DATA_FILE)
        aug_raw = pd.read_csv(AUG_FEATURES_CACHE)
        aug_out = aug_raw[[c for c in corpus_raw.columns if c in aug_raw.columns]].copy()
        missing = [c for c in corpus_raw.columns if c not in aug_out.columns and c != "split"]
        if missing:
            raise SystemExit(f"augment rows lack corpus columns: {missing}")
        aug_out["split"] = "train"
        combined = pd.concat([corpus_raw, aug_out[corpus_raw.columns]], ignore_index=True)
        combined.to_csv(ADOPT_FEATURES_FILE, index=False)
        print(f"  wrote {ADOPT_FEATURES_FILE} ({len(combined)} rows, "
              f"{len(aug_out)} augment rows appended as split='train')")

    print("\nArtifacts:")
    for p in (RESULTS_FILE, CRITERIA_FILE, DIAG_FILE):
        print(f"  {p}")


if __name__ == "__main__":
    main()
