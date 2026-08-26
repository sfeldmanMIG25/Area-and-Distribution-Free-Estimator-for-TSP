"""Does degenerate-geometry augmentation fix GART 2.0's LineNoise failure mode?

GART 2.0 (``lgbm_model_v3``) under-predicts near-collinear ("line_noise") 2D
instances by ~11.6 percent, and the error is almost pure bias
(MSPE -11.58 vs MAPE 11.60). Hypothesis: the training corpus contains no
degenerate geometry, so the model never sees "large n AND high alpha" and has
collapsed to alpha ~ f(n, d).

This script tests that hypothesis WITHOUT solving any new TSP instance. The 210
line_noise instances of the 2D diverse benchmark are already solved, so half of
them can be recycled as training data at zero solver cost.

    M0  standard training split only (reproduces the shipped model)
    M1  standard training split + 105 AUG line_noise rows
    M2  same as M1, AUG rows upweighted x20 via LightGBM ``sample_weight``

All three share the frozen hyperparameters in
``lgbm_model_v3/best_params_v3.json``, seed 42, and early stopping (100 rounds)
on the untouched standard validation split, exactly as
``lgbm_model_v3/LGBM_Alpha_Model_V3.py`` does.

Usage::

    python paper_tooling/augmentation_experiment.py
    python paper_tooling/augmentation_experiment.py --force

Outputs ``paper_tooling/augmentation_results.csv``, caches the three refits in
``paper_tooling/augmentation_models/`` and the extracted 2D benchmark features
in ``paper_tooling/augmentation_2d_features.csv``.
"""

from __future__ import annotations

import os

# Must precede the joblib import on Windows (matches LGBM_Alpha_Model_V3.py).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Canonical feature extractor. Imported, never reimplemented: the augmented rows
# have to be byte-for-byte comparable with the training corpus.
import feature_creator_v3 as fc  # noqa: E402

# --- CONFIGURATION (mirrors lgbm_model_v3/LGBM_Alpha_Model_V3.py) ---
DATA_FILE = ROOT_DIR / "tsp_features_v3.csv"
PARAMS_FILE = ROOT_DIR / "lgbm_model_v3" / "best_params_v3.json"
SHIPPED_MODEL_FILE = ROOT_DIR / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib"

BENCH_DIR = ROOT_DIR / "Generalized_TSP_Analysis"
BENCH_INSTANCES = BENCH_DIR / "instances"
BENCH_TRUTH = BENCH_DIR / "benchmark_checkpoints" / "base_ground_truth_2d.csv"

MODEL_DIR = ROOT_DIR / "paper_tooling" / "augmentation_models"
RESULTS_FILE = ROOT_DIR / "paper_tooling" / "augmentation_results.csv"
FEATURES_CACHE = ROOT_DIR / "paper_tooling" / "augmentation_2d_features.csv"

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100
MAX_BOOST_ROUND = 3000
AUG_WEIGHT = 20.0
DROP_COLS = [
    "instance_name",
    "optimal_cost",
    "alpha",
    "split",
    "grid_size",
    "mst_total_length",
]

#: Generator -> paper generator class (paper_tooling/build_paper_tables.py).
GEN_CLASSES: dict[str, str] = {
    **{g: "Isotropic" for g in ("random", "normal", "triangular", "truncated_exponential")},
    **{g: "Biased" for g in ("squeezed_uniform", "uniform_triangular", "triangular_squeezed", "correlated")},
    **{g: "Geometric" for g in ("grid", "boundary", "x_central")},
    "clustered": "Clustered",
    "line_noise": "LineNoise",
}
GEN_RE = re.compile(r"^TSP-([a-z_0-9]+)-n(\d+)-g(\d+)")

#: Shipped-benchmark MAPE of the uniform ("random") generator class. The
#: extraction path is only trustworthy if it reproduces this.
SANITY_TARGET_MAPE_PCT = 1.5836
SANITY_TOLERANCE_PCT = 0.3

#: M0 must reproduce the shipped artifact; larger drift means the refit recipe
#: has drifted from LGBM_Alpha_Model_V3.py.
REPRO_TOLERANCE_MAPE_PCT = 0.05


# --------------------------------------------------------------------------
# Phase 1 — feature extraction for the already-solved 2D benchmark
# --------------------------------------------------------------------------
def generator_of(instance_name: str) -> str:
    match = GEN_RE.match(instance_name)
    if match is None:
        raise ValueError(f"instance name does not match TSP-<gen>-n<n>-g<g>: {instance_name}")
    return match.group(1)


def class_of(instance_name: str) -> str:
    gen = generator_of(instance_name)
    if gen not in GEN_CLASSES:
        raise ValueError(f"unknown generator {gen!r} in {instance_name}")
    return GEN_CLASSES[gen]


def _extract_one(row: tuple[str, float]) -> dict[str, object]:
    """Run the canonical V3 extractor on one benchmark instance."""
    instance_name, true_cost = row
    inst = fc.load_instance_data(f"{instance_name}.json")
    if inst is None:
        raise FileNotFoundError(f"instance file missing for {instance_name}")
    feats = fc.compute_features_for_instance_v3(inst, {"optimal_cost": float(true_cost)})
    if feats is None:
        raise ValueError(f"{instance_name}: extractor returned None (n < 3)")
    return feats


def extract_2d_features(force: bool) -> pd.DataFrame:
    """Extract the 30 model features for all 2,580 solved 2D benchmark instances."""
    if FEATURES_CACHE.exists() and not force:
        df = pd.read_csv(FEATURES_CACHE)
        print(f"Cached 2D benchmark features -> {FEATURES_CACHE.name} ({len(df)} rows)")
        return df

    truth = pd.read_csv(BENCH_TRUTH, usecols=["instance", "true_cost"])
    if truth["instance"].duplicated().any():
        raise ValueError("base_ground_truth_2d.csv contains duplicate instance names")
    print(f"Extracting V3 features for {len(truth)} 2D benchmark instances ...")

    # feature_creator_v3 resolves paths against its own module-level directory.
    # Repoint it at the benchmark corpus rather than duplicating the loader.
    fc.INSTANCES_DIR = str(BENCH_INSTANCES)

    payload = list(zip(truth["instance"], truth["true_cost"], strict=True))
    workers = max(1, os.cpu_count() or 1)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(pool.map(_extract_one, payload), total=len(payload), desc="V3 features"))
    print(f"  done in {time.perf_counter() - started:.1f}s")

    df = pd.DataFrame(records)
    df["generator"] = df["instance_name"].map(generator_of)
    df["gen_class"] = df["instance_name"].map(class_of)
    df.to_csv(FEATURES_CACHE, index=False)
    print(f"  cached -> {FEATURES_CACHE.name}")
    return df


def attach_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the clipped alpha label exactly as the training pipeline does."""
    out = df.copy()
    divisor = out["mst_total_length"].replace(0, 1e-9)
    out["alpha"] = (out["optimal_cost"] / divisor).clip(1.0, 2.0)
    if (out["optimal_cost"] <= 0).any():
        raise ValueError("optimal_cost contains non-positive values; percent errors undefined")
    return out


def build_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reproduce the exact 30-column training matrix, in model column order."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"extracted frame is missing model features: {missing}")
    return df[feature_names]


def sanity_check_extraction(df2d: pd.DataFrame, feature_names: list[str], subset: int) -> None:
    """Reproduce the shipped uniform-class MAPE, or abort."""
    print("\n--- Extraction sanity check (shipped model, 'random' generator) ---")
    shipped = joblib.load(SHIPPED_MODEL_FILE)
    rnd = df2d[df2d["generator"] == "random"].sort_values("instance_name").reset_index(drop=True)
    if rnd.empty:
        raise ValueError("no 'random' generator instances found")

    for label, frame in (
        (f"first {subset}", rnd.head(subset)),
        (f"all {len(rnd)}", rnd),
    ):
        pred_alpha = np.clip(shipped.predict(build_features(frame, feature_names)), 1.0, 2.0)
        err = (pred_alpha * frame["mst_total_length"].to_numpy() - frame["optimal_cost"].to_numpy())
        err = err / frame["optimal_cost"].to_numpy()
        mape = float(np.mean(np.abs(err)) * 100.0)
        print(f"  {label:>12}: MAPE = {mape:.4f}%  (shipped benchmark: {SANITY_TARGET_MAPE_PCT:.4f}%)")
        if label.startswith("all"):
            drift = abs(mape - SANITY_TARGET_MAPE_PCT)
            if drift > SANITY_TOLERANCE_PCT:
                raise SystemExit(
                    f"EXTRACTION MISMATCH: uniform-class MAPE {mape:.4f}% differs from the shipped "
                    f"{SANITY_TARGET_MAPE_PCT:.4f}% by {drift:.4f} points (> {SANITY_TOLERANCE_PCT}). "
                    "Everything downstream depends on this; aborting."
                )
            print(f"  OK: drift {drift:.4f} points <= {SANITY_TOLERANCE_PCT}")


# --------------------------------------------------------------------------
# Phase 2 — AUG / HELD split
# --------------------------------------------------------------------------
def split_line_noise(df2d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded 50/50 split of the 210 line_noise instances, stratified by n."""
    ln = df2d[df2d["generator"] == "line_noise"].copy()
    if len(ln) != 210:
        raise ValueError(f"expected 210 line_noise instances, found {len(ln)}")

    rng = np.random.default_rng(RANDOM_STATE)
    ln = ln.sort_values("instance_name").reset_index(drop=True)
    ln["_r"] = rng.random(len(ln))
    # n_customers is post-dedup; stratify on the generator's nominal n instead.
    ln["_n_nom"] = ln["instance_name"].str.extract(r"-n(\d+)-").astype(int)
    ln["_rank"] = ln.groupby("_n_nom")["_r"].rank(method="first")
    ln["_count"] = ln.groupby("_n_nom")["_r"].transform("count")
    is_aug = ln["_rank"] <= ln["_count"] / 2.0

    aug = ln[is_aug].drop(columns=["_r", "_rank", "_count"])
    held = ln[~is_aug].drop(columns=["_r", "_rank", "_count"])
    print(
        f"\nLineNoise split (seed {RANDOM_STATE}, stratified by nominal n): "
        f"AUG={len(aug)} HELD={len(held)}; "
        f"AUG n in [{aug['_n_nom'].min()}, {aug['_n_nom'].max()}], "
        f"HELD n in [{held['_n_nom'].min()}, {held['_n_nom'].max()}]"
    )
    if len(aug) != 105 or len(held) != 105:
        raise ValueError(f"split is not 105/105: AUG={len(aug)} HELD={len(held)}")
    return aug.drop(columns=["_n_nom"]), held.drop(columns=["_n_nom"])


# --------------------------------------------------------------------------
# Phase 3 — fit
# --------------------------------------------------------------------------
def load_frozen_params() -> dict[str, float | int]:
    with PARAMS_FILE.open() as handle:
        params = json.load(handle)
    print(f"  frozen hyperparameters: {params}")
    return params


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: np.ndarray | None,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, float | int],
) -> lgb.LGBMRegressor:
    """Fit with the frozen hyperparameters, early-stopped on the standard val split."""
    if X_train.empty or X_val.empty:
        raise ValueError("empty training or validation set")
    model = lgb.LGBMRegressor(
        **params,
        n_estimators=MAX_BOOST_ROUND,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def get_model(
    tag: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: np.ndarray | None,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, float | int],
    force: bool,
) -> lgb.LGBMRegressor:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{tag}.joblib"
    if path.exists() and not force:
        print(f"[{tag}] cached model found -> {path.name} (use --force to refit)")
        return joblib.load(path)
    print(f"[{tag}] fitting on {len(X_train)} train / {len(X_val)} val rows ...")
    started = time.perf_counter()
    model = fit_model(X_train, y_train, weights, X_val, y_val, params)
    elapsed = time.perf_counter() - started
    joblib.dump(model, path)
    print(f"[{tag}] done in {elapsed / 60.0:.2f} min, best_iteration={model.best_iteration_} -> {path.name}")
    return model


# --------------------------------------------------------------------------
# Phase 4 — evaluate
# --------------------------------------------------------------------------
def cost_metrics(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    mst: np.ndarray,
    cost: np.ndarray,
) -> dict[str, float]:
    """Percent-error statistics on tour cost = clip(alpha_hat, 1, 2) * MST."""
    if len(X) == 0:
        raise ValueError("evaluation slice is empty")
    pred_alpha = np.clip(model.predict(X, num_iteration=model.best_iteration_), 1.0, 2.0)
    err = (pred_alpha * mst - cost) / cost
    return {
        "N": int(len(err)),
        "MAPE_pct": float(np.mean(np.abs(err)) * 100.0),
        "SDPE_pct": float(np.std(err, ddof=1) * 100.0) if len(err) > 1 else float("nan"),
        "MSPE_pct": float(np.mean(err) * 100.0),
        "MedAPE_pct": float(np.median(np.abs(err)) * 100.0),
    }


def record(
    rows: list[dict[str, object]],
    model_tag: str,
    eval_slice: str,
    model: lgb.LGBMRegressor,
    frame: pd.DataFrame,
    feature_names: list[str],
    train_rows: int,
) -> None:
    metrics = cost_metrics(
        model,
        build_features(frame, feature_names),
        frame["mst_total_length"].to_numpy(),
        frame["optimal_cost"].to_numpy(),
    )
    rows.append(
        {
            "model": model_tag,
            "eval_slice": eval_slice,
            "N": metrics["N"],
            "MAPE_pct": round(metrics["MAPE_pct"], 4),
            "SDPE_pct": round(metrics["SDPE_pct"], 4),
            "MSPE_pct": round(metrics["MSPE_pct"], 4),
            "MedAPE_pct": round(metrics["MedAPE_pct"], 4),
            "n_trees": int(model.best_iteration_),
            "train_rows": train_rows,
        }
    )


def alpha_diagnostics(
    model: lgb.LGBMRegressor,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float]:
    """Mean predicted vs true alpha, and the OLS slope of predicted on true."""
    pred = np.clip(model.predict(build_features(frame, feature_names), num_iteration=model.best_iteration_), 1.0, 2.0)
    true = frame["alpha"].to_numpy()
    slope, intercept = np.polyfit(true, pred, 1)
    corr = float(np.corrcoef(true, pred)[0, 1])
    return {
        "mean_true_alpha": float(np.mean(true)),
        "mean_pred_alpha": float(np.mean(pred)),
        "slope_pred_on_true": float(slope),
        "intercept": float(intercept),
        "pearson_r": corr,
    }


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="refit models and re-extract features")
    parser.add_argument("--force-features", action="store_true", help="re-extract 2D features only")
    parser.add_argument("--sanity-subset", type=int, default=50, help="size of the small sanity subset")
    args = parser.parse_args()

    shipped = joblib.load(SHIPPED_MODEL_FILE)
    feature_names = list(shipped.feature_name_)
    print(f"Model consumes {len(feature_names)} features.")

    # --- Phase 1: features for the solved 2D benchmark -------------------
    df2d = extract_2d_features(force=args.force or args.force_features)
    df2d = attach_alpha(df2d)
    sanity_check_extraction(df2d, feature_names, args.sanity_subset)

    aug, held = split_line_noise(df2d)
    non_ln = df2d[df2d["generator"] != "line_noise"].copy()
    if len(non_ln) != 2370:
        raise ValueError(f"expected 2370 non-LineNoise instances, found {len(non_ln)}")

    # --- Phase 2: the standard corpus ------------------------------------
    print(f"\nLoading {DATA_FILE.name} ...")
    corpus = pd.read_csv(DATA_FILE)
    corpus = attach_alpha(corpus)
    counts = corpus["split"].value_counts().to_dict()
    print(f"  rows={len(corpus)}  splits={counts}")

    base_train = corpus[corpus["split"] == "train"]
    base_val = corpus[corpus["split"] == "val"]
    base_test = corpus[corpus["split"] == "test"]

    X_val, y_val = build_features(base_val, feature_names), base_val["alpha"]
    X_base, y_base = build_features(base_train, feature_names), base_train["alpha"]
    X_aug, y_aug = build_features(aug, feature_names), aug["alpha"]

    X_plus = pd.concat([X_base, X_aug], ignore_index=True)
    y_plus = pd.concat([y_base, y_aug], ignore_index=True)
    w_flat = np.concatenate([np.ones(len(X_base)), np.ones(len(X_aug))])
    w_boost = np.concatenate([np.ones(len(X_base)), np.full(len(X_aug), AUG_WEIGHT)])

    print("\n--- Fitting three models (frozen hyperparameters, seed 42) ---")
    params = load_frozen_params()
    models = {
        "M0": get_model("M0_baseline", X_base, y_base, None, X_val, y_val, params, args.force),
        "M1": get_model("M1_aug", X_plus, y_plus, w_flat, X_val, y_val, params, args.force),
        "M2": get_model("M2_aug_x20", X_plus, y_plus, w_boost, X_val, y_val, params, args.force),
    }
    train_rows = {"M0": len(X_base), "M1": len(X_plus), "M2": len(X_plus)}

    # --- Phase 3: reproducibility of M0 vs the shipped artifact ----------
    print("\n--- M0 vs shipped artifact (ND test split) ---")
    X_test = build_features(base_test, feature_names)
    mst_test = base_test["mst_total_length"].to_numpy()
    cost_test = base_test["optimal_cost"].to_numpy()
    m0_mape = cost_metrics(models["M0"], X_test, mst_test, cost_test)["MAPE_pct"]
    ship_mape = cost_metrics(shipped, X_test, mst_test, cost_test)["MAPE_pct"]
    print(f"  M0 MAPE {m0_mape:.4f}%   shipped MAPE {ship_mape:.4f}%   drift {abs(m0_mape - ship_mape):.4f}")
    if abs(m0_mape - ship_mape) > REPRO_TOLERANCE_MAPE_PCT:
        print("  WARNING: M0 does not reproduce the shipped model within tolerance.")

    # --- Phase 4: evaluate -----------------------------------------------
    rows: list[dict[str, object]] = []
    eval_slices: list[tuple[str, pd.DataFrame]] = [
        ("HELD_LineNoise", held),
        ("AUG_LineNoise(in-train for M1/M2)", aug),
        ("2D_nonLineNoise_all", non_ln),
    ]
    for cls in ("Isotropic", "Biased", "Geometric", "Clustered"):
        eval_slices.append((f"2D_{cls}", non_ln[non_ln["gen_class"] == cls]))
    eval_slices.append(("ND_test_split", base_test))

    for tag, model in models.items():
        for slice_name, frame in eval_slices:
            record(rows, tag, slice_name, model, frame, feature_names, train_rows[tag])
    for slice_name, frame in eval_slices:
        record(rows, "SHIPPED", slice_name, shipped, frame, feature_names, len(X_base))

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults -> {RESULTS_FILE}")

    pivot_cols = ["model", "eval_slice", "N", "MAPE_pct", "SDPE_pct", "MSPE_pct"]
    print("\n=== RESULTS (cost percent error) ===")
    print(results[pivot_cols].to_string(index=False))

    # --- Phase 5: alpha-level diagnostics on HELD ------------------------
    print("\n=== ALPHA DIAGNOSTICS on HELD LineNoise (N=105) ===")
    diag_rows = []
    for tag in ("SHIPPED", "M0", "M1", "M2"):
        model = shipped if tag == "SHIPPED" else models[tag]
        diag = alpha_diagnostics(model, held, feature_names)
        diag["model"] = tag
        diag_rows.append(diag)
    diag_df = pd.DataFrame(diag_rows)[
        ["model", "mean_true_alpha", "mean_pred_alpha", "slope_pred_on_true", "intercept", "pearson_r"]
    ]
    print(diag_df.round(4).to_string(index=False))
    diag_df.to_csv(RESULTS_FILE.with_name("augmentation_alpha_diagnostics.csv"), index=False)

    # Large-n subset: the regime the hypothesis says is unreachable.
    held_big = held[held["n_customers"] >= 200]
    print(f"\n=== ALPHA DIAGNOSTICS on HELD LineNoise, n >= 200 (N={len(held_big)}) ===")
    big_rows = []
    for tag in ("SHIPPED", "M0", "M1", "M2"):
        model = shipped if tag == "SHIPPED" else models[tag]
        diag = alpha_diagnostics(model, held_big, feature_names)
        diag["model"] = tag
        big_rows.append(diag)
    print(pd.DataFrame(big_rows)[
        ["model", "mean_true_alpha", "mean_pred_alpha", "slope_pred_on_true", "pearson_r"]
    ].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
