"""Generate every numeric value needed to update Area_Free_Main.tex.

Produces a structured text dump ready for paper edits:
  - Abstract + Intro summary numbers (GART 2.0, MST Ratio, top classical)
  - Hyperparameter table (from best_params_v3.json + booster stats)
  - SHAP feature importance ranking (top 30)
  - ND benchmark: by-size and by-dim with 95% bootstrap SDPE CIs
  - 2D benchmark: by-size with bootstrap CIs
  - TSPLIB benchmark: by-size with bootstrap CIs (EUC_2D only, 78 instances)
  - Non-Euclidean TSPLIB: CEIL_2D, ATT, GEO, EXPLICIT
  - Spearman/Kendall rank correlations
  - Wall-time breakdown

All numbers are computed from the freshly regenerated benchmark CSVs;
none are hand-entered. Output is printed to stdout for the paper edit pass.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr, kendalltau

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RNG_SEED = 42
BOOT_B = 1000


def boot_sdpe_ci(err_pct, B=BOOT_B, seed=RNG_SEED, alpha=0.05):
    """Bootstrap 95% CI for SDPE (signed percent error std, Bessel)."""
    err_pct = np.asarray(err_pct, dtype=float)
    err_pct = err_pct[np.isfinite(err_pct)]
    n = len(err_pct)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boots[i] = np.std(err_pct[idx], ddof=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(np.std(err_pct, ddof=1)), float(lo), float(hi)


def row_stats(sub, g_col="gap_pct", a_col="abs_gap_pct",
              pred_col="pred_cost", true_col="true_cost",
              alpha_true=None, alpha_pred=None,
              time_col=None):
    """Return dict(sdpe, sdpe_lo, sdpe_hi, mape, median, r2, r2_alpha, time_ms, n)."""
    sub = sub.copy()
    sub = sub.dropna(subset=[g_col]) if g_col in sub.columns else sub
    if len(sub) == 0:
        return None
    g = sub[g_col].values.astype(float)
    a = sub[a_col].values.astype(float) if a_col in sub.columns else np.abs(g)
    sdpe, lo, hi = boot_sdpe_ci(g)
    mape = float(np.mean(a))
    med = float(np.median(a))
    # R^2 on tour cost
    y_true = sub[true_col].values.astype(float)
    y_pred = sub[pred_col].values.astype(float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_alpha = float("nan")
    if alpha_true is not None and alpha_pred is not None:
        at = np.asarray(alpha_true, dtype=float)
        ap = np.asarray(alpha_pred, dtype=float)
        mask = np.isfinite(at) & np.isfinite(ap)
        if mask.sum() > 1:
            ss_res_a = np.sum((at[mask] - ap[mask]) ** 2)
            ss_tot_a = np.sum((at[mask] - at[mask].mean()) ** 2)
            r2_alpha = 1.0 - ss_res_a / ss_tot_a if ss_tot_a > 0 else float("nan")
    t_ms = float("nan")
    if time_col is not None and time_col in sub.columns:
        t_ms = float(np.median(sub[time_col].dropna().values) * 1000.0) if sub[time_col].notna().any() else float("nan")
    return dict(n=len(sub), sdpe=sdpe, lo=lo, hi=hi, mape=mape, median=med,
                r2=r2, r2_alpha=r2_alpha, time_ms=t_ms)


def compute_true_alpha(df):
    if "true_cost" in df.columns and "mst_length" in df.columns:
        ml = df["mst_length"].values
        tc = df["true_cost"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(ml > 0, tc / ml, np.nan)
        return np.clip(a, 1.0, 2.0)
    return None


def compute_pred_alpha(df):
    if "pred_cost" in df.columns and "mst_length" in df.columns:
        ml = df["mst_length"].values
        pc = df["pred_cost"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(ml > 0, pc / ml, np.nan)
        return np.clip(a, 1.0, 2.0)
    return None


print("=" * 72)
print("PAPER NUMBER GENERATION — Area_Free_Main.tex update pack")
print("=" * 72)

# --- 2D benchmark ---
print("\n" + "=" * 72)
print("2D benchmark — 2580 instances, table tab:2d_by_size")
print("=" * 72)
d2 = pd.read_csv(ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv")
gt2 = pd.read_csv(ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv")[
    ["instance", "n_customers", "mst_length"]]
d2 = d2.merge(gt2, on="instance", how="left")
d2["alpha_true"] = np.clip(d2["true_cost"] / d2["mst_length"].replace(0, np.nan), 1.0, 2.0)
d2["alpha_pred"] = np.clip(d2["pred_cost"] / d2["mst_length"].replace(0, np.nan), 1.0, 2.0)
BUCKETS_2D = [(5, 10), (11, 50), (51, 100), (101, 500), (501, 1000)]
MODELS_2D = ["LGBM_V3", "LGBM_V4", "NN_V3", "GART", "MST_Ratio", "Hilbert", "Daganzo", "Kwon", "BHH", "Chien", "Cavdar"]
MODEL_LABEL = {"LGBM_V3": "GART 2.0", "LGBM_V4": "LGBM V4", "NN_V3": "NN V3",
               "GART": "GART 1.0", "MST_Ratio": "MST Ratio",
               "Hilbert": "Hilbert", "Daganzo": "Daganzo", "Kwon": "Kwon",
               "BHH": "BHH", "Chien": "Chien", "Cavdar": "Cavdar"}
for lo, hi in BUCKETS_2D + [("total", "total")]:
    if lo == "total":
        sub2 = d2.copy(); label = f"Total (n={sub2['instance'].nunique()})"
    else:
        sub2 = d2[(d2["n_customers"] >= lo) & (d2["n_customers"] <= hi)]
        label = f"[{lo},{hi}]  count={sub2['instance'].nunique()}"
    print(f"\n--- 2D bucket {label} ---")
    for m in MODELS_2D:
        sm = sub2[sub2["model"] == m]
        if len(sm) == 0:
            continue
        at = sm["alpha_true"].values if m in ("LGBM_V3", "LGBM_V4", "NN_V3", "GART") else None
        ap = sm["alpha_pred"].values if m in ("LGBM_V3", "LGBM_V4", "NN_V3", "GART") else None
        stats = row_stats(sm, alpha_true=at, alpha_pred=ap, time_col="prediction_time_s")
        if stats is None: continue
        r2a = f"{stats['r2_alpha']:.3f}" if np.isfinite(stats['r2_alpha']) else "---"
        print(f"  {MODEL_LABEL[m]:<12}  SDPE={stats['sdpe']:5.2f} [{stats['lo']:5.2f},{stats['hi']:5.2f}]  "
              f"MAPE={stats['mape']:5.2f}  |Med|={stats['median']:5.2f}  R2={stats['r2']:.3f}  "
              f"R2a={r2a}  t={stats['time_ms']:.2f}ms")

# --- ND benchmark ---
print("\n" + "=" * 72)
print("ND benchmark — 16920 instances, tables tab:nd_by_size / tab:nd_by_dim")
print("=" * 72)
dn = pd.read_csv(ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv", low_memory=False)
dn["dim"] = dn.get("dimension_y", dn.get("dimension", np.nan))
dn["n_c"] = dn.get("n_customers_y", dn.get("n_customers", np.nan))
# mst_length column
ml_col = next((c for c in ["mst_length", "mst_length_y", "mst_length_x"] if c in dn.columns), None)
if ml_col:
    dn["mst_length"] = dn[ml_col]
dn["alpha_true"] = np.clip(dn["true_cost"] / dn["mst_length"].replace(0, np.nan), 1.0, 2.0) if "mst_length" in dn.columns else np.nan
dn["alpha_pred"] = np.clip(dn["pred_cost"] / dn["mst_length"].replace(0, np.nan), 1.0, 2.0) if "mst_length" in dn.columns else np.nan
BUCKETS_ND_N = [(5, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]
MODELS_ND = ["LGBM_V3", "LGBM_V4", "MST_Ratio", "Hilbert"]
for lo, hi in BUCKETS_ND_N + [("total", "total")]:
    if lo == "total":
        subn = dn.copy(); label = f"Total (n={subn['instance'].nunique()})"
    else:
        subn = dn[(dn["n_c"] >= lo) & (dn["n_c"] <= hi)]
        label = f"n=[{lo},{hi}]  count={subn['instance'].nunique()}"
    print(f"\n--- ND bucket {label} ---")
    for m in MODELS_ND:
        sm = subn[subn["model"] == m]
        if len(sm) == 0:
            continue
        at = sm["alpha_true"].values if m in ("LGBM_V3", "LGBM_V4") else None
        ap = sm["alpha_pred"].values if m in ("LGBM_V3", "LGBM_V4") else None
        stats = row_stats(sm, alpha_true=at, alpha_pred=ap, time_col="prediction_time_s")
        if stats is None: continue
        r2a = f"{stats['r2_alpha']:.3f}" if np.isfinite(stats['r2_alpha']) else "---"
        print(f"  {MODEL_LABEL[m]:<12}  SDPE={stats['sdpe']:5.2f} [{stats['lo']:5.2f},{stats['hi']:5.2f}]  "
              f"MAPE={stats['mape']:5.2f}  |Med|={stats['median']:5.2f}  R2={stats['r2']:.3f}  "
              f"R2a={r2a}  t={stats['time_ms']:.2f}ms")

D_BUCKETS = [(2, 2), (3, 5), (6, 10), (15, 25), (30, 50), (100, 100)]
D_LABELS = ["d=2", "d=3-5", "d=6-10", "d=15-25", "d=30-50", "d=100"]
print("\n--- by dimension ---")
for (lo, hi), label in zip(D_BUCKETS, D_LABELS):
    subn = dn[(dn["dim"] >= lo) & (dn["dim"] <= hi)]
    n_inst = subn['instance'].nunique()
    print(f"\n{label}  count={n_inst}")
    for m in MODELS_ND:
        sm = subn[subn["model"] == m]
        if len(sm) == 0: continue
        at = sm["alpha_true"].values if m in ("LGBM_V3", "LGBM_V4") else None
        ap = sm["alpha_pred"].values if m in ("LGBM_V3", "LGBM_V4") else None
        stats = row_stats(sm, alpha_true=at, alpha_pred=ap, time_col="prediction_time_s")
        if stats is None: continue
        r2a = f"{stats['r2_alpha']:.3f}" if np.isfinite(stats['r2_alpha']) else "---"
        print(f"  {MODEL_LABEL[m]:<12}  SDPE={stats['sdpe']:5.2f} [{stats['lo']:5.2f},{stats['hi']:5.2f}]  "
              f"MAPE={stats['mape']:5.2f}  |Med|={stats['median']:5.2f}  R2={stats['r2']:.3f}  "
              f"R2a={r2a}  t={stats['time_ms']:.2f}ms")

# --- TSPLIB benchmark (EUC_2D only, 78 instances) ---
print("\n" + "=" * 72)
print("TSPLIB benchmark — EUC_2D only (paper's 78-instance subset), tab:tsplib_by_size")
print("=" * 72)
dt = pd.read_csv(ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv")
dt = dt[(dt["edge_weight_type"] == "EUC_2D") & (dt["status"] == "ok") & (dt["instance"] != "brg180")].copy()
ml_col = next((c for c in ["mst_length"] if c in dt.columns), None)
if ml_col:
    dt["alpha_true"] = np.clip(dt["true_cost"] / dt[ml_col].replace(0, np.nan), 1.0, 2.0)
    dt["alpha_pred"] = np.clip(dt["pred_cost"] / dt[ml_col].replace(0, np.nan), 1.0, 2.0)
BUCKETS_TSPLIB = [(51, 150), (151, 400), (401, 10**9)]
BUCKET_LABELS_TSPLIB = ["[51,150]", "[151,400]", ">400"]
MODELS_TSPLIB = ["LGBM_V3", "LGBM_V4", "GART_1.0", "MST_Ratio", "Hilbert", "Daganzo", "Kwon", "BHH", "Chien", "Cavdar"]
MODEL_LABEL_TSP = MODEL_LABEL | {"GART_1.0": "GART 1.0"}
for (lo, hi), lab in zip(BUCKETS_TSPLIB + [("total", "total")], BUCKET_LABELS_TSPLIB + ["Total"]):
    if lo == "total":
        subt = dt.copy()
    else:
        subt = dt[(dt["n"] >= lo) & (dt["n"] <= hi)]
    n_inst = subt['instance'].nunique()
    print(f"\n--- TSPLIB bucket {lab}  count={n_inst} ---")
    for m in MODELS_TSPLIB:
        sm = subt[subt["model"] == m]
        if len(sm) == 0: continue
        at = sm["alpha_true"].values if m in ("LGBM_V3", "LGBM_V4", "GART_1.0") and "alpha_true" in sm.columns else None
        ap = sm["alpha_pred"].values if m in ("LGBM_V3", "LGBM_V4", "GART_1.0") and "alpha_pred" in sm.columns else None
        stats = row_stats(sm, alpha_true=at, alpha_pred=ap, time_col="total_time_s")
        if stats is None: continue
        r2a = f"{stats['r2_alpha']:.3f}" if np.isfinite(stats['r2_alpha']) else "---"
        print(f"  {MODEL_LABEL_TSP[m]:<12}  SDPE={stats['sdpe']:5.2f} [{stats['lo']:5.2f},{stats['hi']:5.2f}]  "
              f"MAPE={stats['mape']:5.2f}  |Med|={stats['median']:5.2f}  R2={stats['r2']:.3f}  "
              f"R2a={r2a}  t={stats['time_ms']:.2f}ms")

# --- non-Euclidean TSPLIB ---
print("\n" + "=" * 72)
print("TSPLIB non-Euclidean (CEIL_2D, ATT, GEO, EXPLICIT), tab:tsplib_nonEuc")
print("=" * 72)
dtn = pd.read_csv(ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv")
dtn = dtn[(dtn["model"] == "LGBM_V3") & (dtn["status"] == "ok") & (dtn["instance"] != "brg180")].copy()
for ewt in ["CEIL_2D", "ATT", "GEO", "EXPLICIT"]:
    sub = dtn[dtn["edge_weight_type"] == ewt]
    if len(sub) == 0:
        print(f"  {ewt}: N=0 no rows"); continue
    mape = sub["abs_gap_pct"].mean()
    sdpe = float(np.std(sub["gap_pct"].values, ddof=1)) if len(sub) > 1 else float("nan")
    med = sub["abs_gap_pct"].median()
    ml = sub["mst_length"] if "mst_length" in sub.columns else None
    t_ms = float(np.median(sub["total_time_s"].dropna().values) * 1000.0) if "total_time_s" in sub.columns else float("nan")
    print(f"  {ewt:<14}  N={len(sub):2d}  MAPE={mape:5.2f}  SDPE={sdpe:5.2f}  Median={med:5.2f}  t={t_ms:.1f}ms")
# Total non-Euclidean
dtn_ne = dtn[dtn["edge_weight_type"].isin(["CEIL_2D", "ATT", "GEO", "EXPLICIT"])]
if len(dtn_ne) > 0:
    print(f"  TOTAL           N={len(dtn_ne):2d}  MAPE={dtn_ne['abs_gap_pct'].mean():5.2f}  "
          f"SDPE={float(np.std(dtn_ne['gap_pct'].values, ddof=1)):5.2f}  Median={dtn_ne['abs_gap_pct'].median():5.2f}")

# --- Rank correlations ---
print("\n" + "=" * 72)
print("Rank correlations (Spearman rho, Kendall tau) — GART 2.0 vs others")
print("=" * 72)
for tag, dfx in [("2D", d2), ("ND", dn), ("TSPLIB EUC_2D", dt)]:
    g = dfx[dfx["model"] == "LGBM_V3"][["instance", "pred_cost", "true_cost"]].dropna()
    if len(g) < 2: continue
    rho, _ = spearmanr(g["pred_cost"], g["true_cost"])
    tau, _ = kendalltau(g["pred_cost"], g["true_cost"])
    print(f"  GART 2.0 on {tag}: Spearman rho={rho:.4f}  Kendall tau={tau:.4f}")

# --- Hyperparameters + booster stats (new) ---
print("\n" + "=" * 72)
print("GART 2.0 hyperparameters (tab:hyperparams)")
print("=" * 72)
try:
    with open(ROOT / "lgbm_model_v3" / "best_params_v3.json") as f:
        bp = json.load(f)
    print(json.dumps(bp, indent=2))
except Exception as e:
    print(f"  best_params_v3.json: {e}")

try:
    booster = joblib.load(ROOT / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib")
    inner = booster.booster_ if hasattr(booster, "booster_") else booster
    n_trees = inner.num_trees() if hasattr(inner, "num_trees") else booster.n_estimators
    # leaves per tree
    mdl_dump = inner.dump_model()
    leaves = [t["num_leaves"] if "num_leaves" in t else t.get("tree_info", {}).get("num_leaves", 0) for t in mdl_dump["tree_info"]]
    if not leaves or max(leaves) == 0:
        # fall back
        leaves = [sum(1 for _ in range(1000) if False) for _ in mdl_dump["tree_info"]]
    # try tree depth
    def tree_depth(node):
        if node is None or "leaf_index" in node: return 0
        l = tree_depth(node.get("left_child"))
        r = tree_depth(node.get("right_child"))
        return 1 + max(l, r)
    depths = [tree_depth(t["tree_structure"]) for t in mdl_dump["tree_info"]]
    print(f"\nBooster stats (new V3):")
    print(f"  trees (after early stop): {n_trees}")
    print(f"  avg leaves/tree: {np.mean(leaves):.2f}   max leaves/tree: {max(leaves) if leaves else 0}")
    print(f"  avg depth: {np.mean(depths):.2f}   max depth: {max(depths) if depths else 0}")
except Exception as e:
    print(f"  booster stats failed: {e}")

# --- SHAP importance (top 30) ---
print("\n" + "=" * 72)
print("SHAP feature importance (tab:shap_top) — top 30 by mean |SHAP|")
print("=" * 72)
try:
    import shap
    from scipy import stats as sp_stats
    df_feat = pd.read_csv(ROOT / "tsp_features_v3.csv")
    booster = joblib.load(ROOT / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib")
    feature_names = booster.feature_name_ if hasattr(booster, "feature_name_") else list(booster.feature_name())
    test_df = df_feat[df_feat["split"] == "test"].sample(n=min(5000, (df_feat["split"] == "test").sum()),
                                                         random_state=RNG_SEED)
    X_test = test_df[feature_names]
    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_values(X_test)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    total_mass = mean_abs_shap.sum()
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    imp["share_pct"] = imp["mean_abs_shap"] / total_mass * 100
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    for i, row in imp.iterrows():
        print(f"  {i+1:2d}  {row['feature']:<28}  |SHAP|={row['mean_abs_shap']:.5f}  share={row['share_pct']:5.2f}%")
except Exception as e:
    print(f"  SHAP failed: {e}")
    # fallback: gain importance
    booster = joblib.load(ROOT / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib")
    feature_names = booster.feature_name_ if hasattr(booster, "feature_name_") else list(booster.feature_name())
    gains = booster.booster_.feature_importance(importance_type='gain') if hasattr(booster, "booster_") else booster.feature_importance(importance_type='gain')
    total = gains.sum()
    imp = pd.DataFrame({"feature": feature_names, "gain": gains}).sort_values("gain", ascending=False).reset_index(drop=True)
    print("(SHAP unavailable; showing gain-based importance as proxy)")
    for i, row in imp.iterrows():
        print(f"  {i+1:2d}  {row['feature']:<28}  gain={row['gain']:.1f}  share={row['gain']/total*100:5.2f}%")

# --- Wall-time split on TSPLIB (GART 2.0) ---
print("\n" + "=" * 72)
print("Wall-time breakdown (GART 2.0 on TSPLIB EUC_2D)")
print("=" * 72)
gart = dt[dt["model"] == "LGBM_V3"].copy()
if "feature_time_s" in gart.columns and "inference_time_s" in gart.columns and "total_time_s" in gart.columns:
    ft = gart["feature_time_s"].mean()
    it = gart["inference_time_s"].mean()
    tot = gart["total_time_s"].mean()
    # feature time = feature extraction + MST
    # inference = LightGBM inference
    pct_feat = ft / tot * 100
    pct_inf = it / tot * 100
    print(f"  mean feature+MST time: {ft*1000:.2f} ms  ({pct_feat:.2f}%)")
    print(f"  mean inference time:   {it*1000:.2f} ms  ({pct_inf:.2f}%)")
    print(f"  mean total time:       {tot*1000:.2f} ms")
