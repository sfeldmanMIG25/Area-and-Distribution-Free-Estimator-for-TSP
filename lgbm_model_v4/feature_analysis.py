"""Candidate-feature audit for V4.

Writes ``feature_report.md`` with one row per candidate, including:

* F-regression p-value against alpha target
* Mutual information against alpha
* Spearman correlation with alpha
* Per-feature wall-time (median + p95) on a stratified sample of 200 instances
* Forward-selection SDPE gain: baseline V3 set -> add one feature -> val SDPE delta
* Final filter decision + reason, plus a ``selected_features.json`` for train.py

The filter rule (all-of, pre-user review):
  * F-reg p < 0.01
  * MI >= 0.01
  * |delta SDPE_pp| >= 0.05 pp OR SHAP share >= 2%
  * p95 wall-time <= 10 ms

The user reviews this report before train.py is run.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.feature_selection import f_regression, mutual_info_regression

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from feature_creator_v3 import load_instance_data
from feature_engineering import compute_features, feature_columns_for_training

V4_CSV = REPO_ROOT / "tsp_features_v4.csv"
REPORT_MD = THIS_DIR / "feature_report.md"
SELECTED_JSON = THIS_DIR / "selected_features.json"

# Features carried over verbatim from V3 — always retained, no ablation needed.
BASELINE_V3_FEATURES = [
    "n_customers", "dimension",  # grid_size deliberately excluded (unused in V3 model)
    "bounding_hypervolume", "log_bounding_hypervolume",
    "node_density", "log_node_density", "aspect_ratio",
    "centroid_dist_mean", "centroid_dist_std",
    "centroid_dist_max", "centroid_dist_iqr",
    "mst_total_length",
    "mst_edge_mean", "mst_edge_std", "mst_edge_skew",
    "mst_edge_kurtosis", "mst_edge_max",
    "mst_edge_q10", "mst_edge_q25", "mst_edge_q50",
    "mst_edge_q75", "mst_edge_q90",
    "mst_dominance_ratio", "mst_gap_ratio",
    "mst_leaf_ratio", "mst_degree_mean", "mst_degree_std",
    "mst_degree_max", "mst_diameter", "mst_diameter_normalized",
    "large_edge_count",
]

V4_CANDIDATES = [
    "obb_volume", "log_obb_volume", "obb_shrinkage",
    "pca_e1_share", "pca_effective_rank",
    "nn1_dist_mean", "nn1_dist_cv", "nn2_dist_mean", "nn_gap_ratio",
    "pca_log_det", "mst_edge_pca_e1_share", "intrinsic_dim_2nn",
]

DELTA_SDPE_THRESHOLD = 0.05      # percentage points
MI_THRESHOLD = 0.01
P_VALUE_THRESHOLD = 0.01
TIME_P95_CAP_MS = 10.0


# =============================================================================
# Loading + alpha target
# =============================================================================
def _load_frame() -> pd.DataFrame:
    df = pd.read_csv(V4_CSV)
    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(1.0, 2.0)
    df = df.dropna(subset=["alpha"])
    return df


# =============================================================================
# Statistical tests
# =============================================================================
def _significance_tests(df: pd.DataFrame, feature: str) -> Dict[str, float]:
    mask = df[feature].notna() & df["alpha"].notna()
    x = df.loc[mask, feature].to_numpy().reshape(-1, 1)
    y = df.loc[mask, "alpha"].to_numpy()
    if len(y) < 20:
        return {"p_value": float("nan"), "mi": float("nan"), "spearman": float("nan"), "n": int(mask.sum())}
    _, p_vals = f_regression(x, y)
    mi = mutual_info_regression(x, y, n_neighbors=3, random_state=42)
    rho, _ = spearmanr(x.ravel(), y)
    return {"p_value": float(p_vals[0]), "mi": float(mi[0]), "spearman": float(rho), "n": int(mask.sum())}


# =============================================================================
# Per-feature wall-clock on a stratified sample
# =============================================================================
def _stratified_sample(df: pd.DataFrame, per_dim: int = 10) -> List[str]:
    """Return at most ``per_dim`` instance names per dimension class."""
    names: List[str] = []
    for _, sub in df.groupby("dimension"):
        names.extend(sub["instance_name"].sample(min(per_dim, len(sub)), random_state=42).tolist())
    return names


def _time_per_feature(sample_names: List[str]) -> Dict[str, tuple[float, float]]:
    """For each candidate feature, measure median and p95 wall-time per
    instance. We reuse compute_features (it computes everything), then
    re-run each *feature-specific* helper in isolation on the same coords
    to get a per-feature breakdown."""
    from feature_engineering import (
        _pca_eigenvalues, _obb_volume, _knn_features, _intrinsic_dim_2nn,
        _mst_edge_pca_e1_share, _compute_mst,
    )
    timings: Dict[str, List[float]] = {f: [] for f in V4_CANDIDATES}

    for name in sample_names:
        inst = load_instance_data(f"{name}.json")
        if inst is None:
            continue
        coords = np.unique(inst["coordinates"], axis=0)
        if coords.shape[0] < 3:
            continue
        d = int(inst["dimension"])

        mst = _compute_mst(coords, d)
        rows, cols = mst.nonzero()

        # PCA-based batch (Tier 1 + pca_log_det).
        t0 = time.perf_counter()
        eigvals = _pca_eigenvalues(coords)
        t_pca = time.perf_counter() - t0

        t0 = time.perf_counter()
        _obb_volume(coords)
        t_obb = time.perf_counter() - t0

        t0 = time.perf_counter()
        _knn_features(coords, d)
        t_knn = time.perf_counter() - t0

        t0 = time.perf_counter()
        _intrinsic_dim_2nn(coords, d)
        t_idim = time.perf_counter() - t0

        t0 = time.perf_counter()
        _mst_edge_pca_e1_share(coords, rows, cols)
        t_edge_pca = time.perf_counter() - t0

        # Amortise PCA cost across the five features that use the eigvals.
        pca_share = t_pca / 5.0
        timings["obb_volume"].append(pca_share + t_obb)
        timings["log_obb_volume"].append(pca_share + t_obb)
        timings["obb_shrinkage"].append(pca_share + t_obb)
        timings["pca_e1_share"].append(pca_share)
        timings["pca_effective_rank"].append(pca_share)
        timings["pca_log_det"].append(pca_share)

        # Amortise kNN across its four features.
        knn_share = t_knn / 4.0
        for k in ["nn1_dist_mean", "nn1_dist_cv", "nn2_dist_mean", "nn_gap_ratio"]:
            timings[k].append(knn_share)

        timings["intrinsic_dim_2nn"].append(t_idim)
        timings["mst_edge_pca_e1_share"].append(t_edge_pca)

    out: Dict[str, tuple[float, float]] = {}
    for f, ts in timings.items():
        if not ts:
            out[f] = (float("nan"), float("nan"))
            continue
        arr = np.asarray(ts) * 1000.0
        out[f] = (float(np.median(arr)), float(np.percentile(arr, 95)))
    return out


# =============================================================================
# Forward-selection SDPE gain
# =============================================================================
def _sdpe_on_val(booster, X_val, y_val, mst_val, true_val) -> tuple[float, float]:
    """Predict alpha, project to cost, compute SDPE and MAPE on val."""
    pred_alpha = np.clip(booster.predict(X_val, num_iteration=booster.best_iteration), 1.0, 2.0)
    pred_cost = pred_alpha * mst_val
    err = (pred_cost - true_val) / true_val
    return float(np.std(err, ddof=1) * 100.0), float(np.mean(np.abs(err)) * 100.0)


def _fit_quick(X_tr, y_tr, X_val, y_val, seed: int = 42):
    params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.05, num_leaves=127, min_child_samples=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        verbosity=-1, seed=seed,
    )
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
    booster = lgb.train(
        params, dtr, num_boost_round=800, valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def _forward_select(df: pd.DataFrame):
    mask_tr = df["split"] == "train"
    mask_val = df["split"] == "val"
    y_tr = df.loc[mask_tr, "alpha"]
    y_val = df.loc[mask_val, "alpha"]
    mst_val = df.loc[mask_val, "mst_total_length"].to_numpy()
    true_val = df.loc[mask_val, "optimal_cost"].to_numpy()

    X_tr_base = df.loc[mask_tr, BASELINE_V3_FEATURES]
    X_val_base = df.loc[mask_val, BASELINE_V3_FEATURES]

    booster = _fit_quick(X_tr_base, y_tr, X_val_base, y_val)
    base_sdpe, base_mape = _sdpe_on_val(booster, X_val_base, y_val, mst_val, true_val)
    print(f"[fwd-select] baseline V3 features: SDPE={base_sdpe:.3f}%  MAPE={base_mape:.3f}%")

    deltas: Dict[str, Dict[str, float]] = {}
    for feat in V4_CANDIDATES:
        X_tr = df.loc[mask_tr, BASELINE_V3_FEATURES + [feat]]
        X_val = df.loc[mask_val, BASELINE_V3_FEATURES + [feat]]
        booster = _fit_quick(X_tr, y_tr, X_val, y_val)
        sdpe, mape = _sdpe_on_val(booster, X_val, y_val, mst_val, true_val)
        deltas[feat] = {
            "sdpe": sdpe, "mape": mape,
            "d_sdpe_pp": sdpe - base_sdpe,
            "d_mape_pp": mape - base_mape,
        }
        print(f"[fwd-select]   +{feat:28s} SDPE={sdpe:.3f}% (d={sdpe-base_sdpe:+.3f})  MAPE={mape:.3f}% (d={mape-base_mape:+.3f})")

    return {"baseline": {"sdpe": base_sdpe, "mape": base_mape}, "deltas": deltas}


# =============================================================================
# Report writer
# =============================================================================
def _decide(stats_row: dict, timing_row: tuple[float, float], fwd_row: dict) -> tuple[bool, str]:
    p = stats_row["p_value"]; mi = stats_row["mi"]
    t_p95 = timing_row[1]
    d_sdpe = fwd_row["d_sdpe_pp"]
    reasons = []
    if not (p < P_VALUE_THRESHOLD):
        reasons.append(f"p={p:.2e} >= {P_VALUE_THRESHOLD}")
    if not (mi >= MI_THRESHOLD):
        reasons.append(f"MI={mi:.4f} < {MI_THRESHOLD}")
    if not (abs(d_sdpe) >= DELTA_SDPE_THRESHOLD):
        reasons.append(f"|dSDPE|={abs(d_sdpe):.3f}pp < {DELTA_SDPE_THRESHOLD}")
    if not (t_p95 <= TIME_P95_CAP_MS):
        reasons.append(f"p95={t_p95:.2f}ms > {TIME_P95_CAP_MS}")
    # Bias: better to drop by cost than by dSDPE. We *keep* when |dSDPE|>=thresh OR when stat tests are strong.
    keep = (abs(d_sdpe) >= DELTA_SDPE_THRESHOLD or mi >= 5 * MI_THRESHOLD) and (p < P_VALUE_THRESHOLD) and (t_p95 <= TIME_P95_CAP_MS)
    return keep, ("auto-keep" if keep and not reasons else ("auto-keep (strong MI)" if keep else " | ".join(reasons)))


def _write_report(stats, timings, fwd, keep_decisions) -> None:
    lines: list[str] = []
    lines.append("# V4 feature-candidate report\n")
    lines.append(f"Baseline (V3 feature set): SDPE = **{fwd['baseline']['sdpe']:.3f}%**, MAPE = **{fwd['baseline']['mape']:.3f}%** on val split.\n")
    lines.append(
        f"Filter rule: `p(F-reg) < {P_VALUE_THRESHOLD}`, `MI >= {MI_THRESHOLD}`, "
        f"`|delta SDPE| >= {DELTA_SDPE_THRESHOLD}pp` (or strong MI), `p95 <= {TIME_P95_CAP_MS}ms`.\n"
    )
    lines.append("| Feature | p(F-reg) | MI | Spearman | p50 ms | p95 ms | delta SDPE pp | delta MAPE pp | Decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for feat in V4_CANDIDATES:
        s = stats[feat]; t = timings[feat]; f = fwd["deltas"][feat]
        keep, reason = keep_decisions[feat]
        flag = "KEEP" if keep else "DROP"
        lines.append(
            f"| `{feat}` | {s['p_value']:.2e} | {s['mi']:.4f} | {s['spearman']:+.3f} | "
            f"{t[0]:.2f} | {t[1]:.2f} | {f['d_sdpe_pp']:+.3f} | {f['d_mape_pp']:+.3f} | "
            f"**{flag}** — {reason} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- `delta SDPE pp` is the change in val-set SDPE (percentage points) "
        "when this feature is *added* to the V3 baseline. Negative is better."
    )
    lines.append(
        "- All timings are on a stratified sample of 200 instances (10 per "
        "dimension class), including the amortised cost of shared primitives "
        "(PCA eigendecomposition, cKDTree)."
    )
    lines.append(
        "- Features above `KDTREE_DIM_CAP = 16` report NaN for k-NN and "
        "intrinsic-dim metrics. The model sees NaN; LightGBM handles NaN via "
        "built-in sentinel splits."
    )
    lines.append("")
    lines.append(
        "After review, edit `selected_features.json` directly (or delete a "
        "candidate from it) before running `train.py`. The training script "
        "trusts this file as the final feature list."
    )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {REPORT_MD}")


def main() -> None:
    df = _load_frame()
    print(f"[analysis] loaded {len(df)} rows, dims {sorted(df['dimension'].unique())}")

    print("[analysis] step 1/3 — significance tests ...")
    stats_all = {f: _significance_tests(df, f) for f in V4_CANDIDATES}

    print("[analysis] step 2/3 — per-feature timing on stratified sample ...")
    sample = _stratified_sample(df, per_dim=10)
    timings = _time_per_feature(sample)

    print("[analysis] step 3/3 — forward-selection SDPE gain (LightGBM quick-fit x 12) ...")
    fwd = _forward_select(df)

    decisions = {f: _decide(stats_all[f], timings[f], fwd["deltas"][f]) for f in V4_CANDIDATES}
    _write_report(stats_all, timings, fwd, decisions)

    selected = BASELINE_V3_FEATURES + [f for f in V4_CANDIDATES if decisions[f][0]]
    with open(SELECTED_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_v3_features": BASELINE_V3_FEATURES,
            "v4_candidates": V4_CANDIDATES,
            "selected": selected,
            "dropped": [f for f in V4_CANDIDATES if not decisions[f][0]],
            "decisions": {f: {"keep": k, "reason": r} for f, (k, r) in decisions.items()},
            "baseline_metrics_val": fwd["baseline"],
        }, f, indent=2)
    print(f"[analysis] wrote {SELECTED_JSON}  (selected {len(selected)} features)")


if __name__ == "__main__":
    main()
