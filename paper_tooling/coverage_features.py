"""Extract the 31 GART 2.0 features for the alpha-coverage corpus.

Uses ``lgbm_model_v4/feature_engineering.compute_features`` -- the extractor
that built ``tsp_features_v4.csv``, i.e. the training table the shipped model
was fitted on -- with the same ``np.unique`` row-dedupe the corpus build used.
Checked against the production inference extractor
``lgbm_model_v3/feature_engineering_gart2.compute_features``: the two agree to
1e-8 relative on all 31 features, so there is no train/serve skew to inherit.

Split assignment mirrors ``feature_creator_v3.create_stratified_split``:
70/20/10 train/val/test stratified by (dimension, n_customers, grid_size), seed
42.  The coverage test rows are a held-out slice of the NEW geometry and are
kept in their own stratum.  The ND test split is untouched, so the headline
comparison against the frozen model runs on exactly the same 16,920 rows it
was measured on.

Writes ``alpha_coverage/coverage_features.csv`` only.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "lgbm_model_v4"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

COV_ROOT = ROOT / "alpha_coverage"
INST_DIR = COV_ROOT / "instances"
SOL_DIR = COV_ROOT / "solutions"
OUT_CSV = COV_ROOT / "coverage_features.csv"

RANDOM_STATE = 42
ALPHA_CLIP = (1.0, 2.0)

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


def _one(name: str) -> dict | None:
    from feature_engineering import compute_features

    inst_path = INST_DIR / f"{name}.json"
    sol_path = SOL_DIR / f"{name}.sol.json"
    if not (inst_path.exists() and sol_path.exists()):
        return None
    with open(inst_path) as f:
        inst = json.load(f)
    with open(sol_path) as f:
        sol = json.load(f)

    coords = np.unique(np.asarray(inst["coordinates"], dtype=np.float64), axis=0)
    if coords.shape[0] < 3:
        return None
    feats = compute_features(coords, int(inst["dimension"]), int(inst["grid_size"]))
    geo = inst.get("coverage_geometry", {})
    feats.update({
        "instance_name": name,
        "optimal_cost": float(sol["optimal_cost"]),
        "grid_size": int(inst["grid_size"]),
        "coverage_family": inst.get("coverage_family"),
        "coverage_family_requested": inst.get("coverage_family_requested"),
        "coverage_group": inst.get("coverage_group"),
        "alpha_pred": geo.get("alpha_pred"),
        "rho": geo.get("rho"), "profile": geo.get("profile"),
        "mix": geo.get("mix"), "spacing": geo.get("spacing"),
        "rho_measured": geo.get("rho_measured"),
        "transverse_kurtosis": geo.get("transverse_kurtosis"),
        "solver": sol.get("optimal_solver"),
    })
    return feats


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """70/20/10 by (dimension, n_customers, grid_size) -- the corpus rule."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = df.copy()
    df["_r"] = rng.random(len(df))
    grp = df.groupby(["dimension", "n_customers", "grid_size"])
    frac = (grp["_r"].rank(method="first") - 1) / grp["_r"].transform("count")
    df["split"] = np.where(frac < 0.10, "test", np.where(frac < 0.30, "val", "train"))
    return df.drop(columns=["_r"])


def main() -> None:
    names = sorted(p.stem for p in INST_DIR.glob("*.json"))
    print(f"[cov-feat] {len(names)} coverage instances")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as ex:
        for i, r in enumerate(ex.map(_one, names, chunksize=16), 1):
            if r is not None:
                rows.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(names)}", flush=True)

    df = pd.DataFrame(rows)
    print(f"[cov-feat] extracted {len(df)} / {len(names)}")

    missing = [c for c in FEATS31 if c not in df.columns]
    if missing:
        raise RuntimeError(f"extractor did not return {missing}")

    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(*ALPHA_CLIP)
    n_bad = int(df["alpha"].isna().sum())
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)
    if n_bad:
        print(f"[cov-feat] dropped {n_bad} rows with unusable MST")

    # A label at the clip boundary means the raw ratio left [1, 2], which for a
    # metric instance is impossible and would signal a broken label. Report it.
    raw = df["optimal_cost"] / df["mst_total_length"]
    out_of_range = int(((raw < 1.0 - 1e-9) | (raw > 2.0 + 1e-9)).sum())
    print(f"[cov-feat] raw alpha outside [1,2] before clip: {out_of_range}")

    df = stratified_split(df)
    df["corpus"] = "cov"

    front = ["instance_name", "split", "corpus", "optimal_cost", "alpha",
             "coverage_family", "coverage_group", "alpha_pred", "rho", "profile",
             "mix", "spacing", "solver"]
    cols = front + [c for c in df.columns if c not in front]
    df[cols].to_csv(OUT_CSV, index=False)
    print(f"[cov-feat] wrote {OUT_CSV}  ({len(df)} rows, {len(cols)} cols)")
    print(df["split"].value_counts().to_string())
    print("\nalpha by n decade (all coverage rows):")
    b = pd.cut(df.n_customers, [0, 20, 50, 100, 200, 500, 1000, 10 ** 9])
    print(df.groupby(b, observed=True)["alpha"]
          .agg(["count", "min", "median", "max"]).round(3).to_string())
    print("\nalpha histogram in 10 bins over [1,2]:")
    print(np.histogram(df["alpha"], bins=10, range=(1.0, 2.0))[0].tolist())


if __name__ == "__main__":
    main()
