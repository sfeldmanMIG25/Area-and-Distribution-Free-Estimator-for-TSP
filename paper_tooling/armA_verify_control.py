"""Control: score the SAME opaque models with the study's own cached feature
table (v4_study_feature_cache.csv) using MY metric code and MY transform.

If the reported numbers reproduce here but not from raw instances, the reported
numbers are arithmetically right and the gap is re-extraction sensitivity.
If they do not reproduce even here, the reported numbers are wrong.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
sys.path.insert(0, str(ROOT))

FEATS31 = [
    "n_customers", "dimension", "log_bounding_hypervolume", "bounding_hypervolume",
    "log_node_density", "node_density", "aspect_ratio", "centroid_dist_mean",
    "centroid_dist_std", "centroid_dist_max", "centroid_dist_iqr", "mst_edge_mean",
    "mst_edge_std", "mst_edge_skew", "mst_edge_kurtosis", "mst_edge_max",
    "mst_edge_q10", "mst_edge_q25", "mst_edge_q50", "mst_edge_q75", "mst_edge_q90",
    "mst_dominance_ratio", "mst_gap_ratio", "mst_leaf_ratio", "mst_degree_mean",
    "mst_degree_std", "mst_degree_max", "mst_diameter", "mst_diameter_normalized",
    "large_edge_count", "greedy_nn_over_mst",
]
CLIP = (1.0, 2.0)
REPORTED = {
    ("augment", "FROZEN"): (11.980, 14.297), ("augment", "A"): (0.627, 1.133),
    ("bench2d", "FROZEN"): (2.904, 4.686), ("bench2d", "A"): (2.053, 3.172),
    ("nd_test", "FROZEN"): (0.620, 0.988), ("nd_test", "A"): (0.597, 0.984),
    ("tsplib_euc2d", "FROZEN"): (2.556, 2.955), ("tsplib_euc2d", "A"): (2.484, 3.242),
    ("tsplib_noneuc", "FROZEN"): (3.346, 3.897), ("tsplib_noneuc", "A"): (2.431, 3.001),
}


def summary(e):
    e = np.asarray(e, dtype=np.float64)
    return {"n": int(e.size), "mape": float(np.mean(np.abs(e))),
            "sdpe": float(np.std(e, ddof=1)), "mspe": float(np.mean(e))}


def main():
    import joblib

    C = pd.read_csv(HERE / "v4_study_feature_cache.csv", low_memory=False)
    gen = pd.read_csv(HERE / "augmentation_2d_features.csv",
                      usecols=["instance_name", "generator"])
    C = C.merge(gen, left_on="instance", right_on="instance_name", how="left")

    models = {"FROZEN": ROOT / "lgbm_model_v3" / "gart2_final.joblib",
              "A": HERE / "support_arms_models" / "A.joblib"}
    rows, keep = [], {}
    for tag, p in models.items():
        m = joblib.load(p)
        for st, g in C.groupby("stratum"):
            ok = g[(g.status == "ok") & g[FEATS31].notna().all(axis=1)
                   & g.true_cost.notna()].copy()
            z = m.predict(ok[FEATS31], num_iteration=m.best_iteration)
            a = np.clip(1.0 + 1.0 / (1.0 + np.exp(-z)), *CLIP)
            ok["pred_alpha"] = a
            ok["err_pct"] = (a * ok["mst_total_length"] - ok["true_cost"]) \
                / ok["true_cost"] * 100.0
            keep[(tag, st)] = ok
            s = summary(ok["err_pct"].to_numpy())
            rep = REPORTED.get((st, tag))
            rows.append({"model": tag, "stratum": st, **s,
                         "rep_mape": rep[0] if rep else np.nan,
                         "rep_sdpe": rep[1] if rep else np.nan,
                         "d_mape": abs(s["mape"] - rep[0]) if rep else np.nan,
                         "d_sdpe": abs(s["sdpe"] - rep[1]) if rep else np.nan})
    S = pd.DataFrame(rows)
    S.to_csv(HERE / "armA_verify_control_strata.csv", index=False)
    print("=== CONTROL: study's cached features, my metric code ===")
    print(S.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== grid MSPE and line_noise slope on cached features ===")
    for tag in models:
        ok = keep[(tag, "bench2d")]
        grid = ok[ok.generator == "grid"]
        ln = ok[(ok.generator == "line_noise") & (ok.n_customers >= 200)]
        ta = np.clip(ln["true_cost"] / ln["mst_total_length"], *CLIP).to_numpy()
        lr = stats.linregress(ta, ln["pred_alpha"].to_numpy())
        print(f"{tag:7s} grid MSPE {grid['err_pct'].mean():+.4f} (n={len(grid)})   "
              f"line_noise slope {lr.slope:.4f} (N={len(ln)})")


if __name__ == "__main__":
    main()
