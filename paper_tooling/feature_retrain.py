"""Retrain GART 2.0 with the shortlisted local-geometry features.

Two models, differing ONLY in the feature set:

    M0   the 30 shipped features.  Required to reproduce the shipped artifact
         bit-identically (best_iteration 2031, ND test MAPE 0.8769,
         max |pred - shipped_pred| == 0.0).  The run aborts if it does not.
    MF   the 30 shipped features + the 5 shortlisted candidates.

Both use the frozen hyperparameters in ``lgbm_model_v3/best_params_v3.json``,
seed 42, early stopping 100 on the untouched validation split, and the exact
preprocessing of ``lgbm_model_v3/LGBM_Alpha_Model_V3.py``.  The train/val/test
assignment in ``tsp_features_v3.csv`` is used verbatim; no row moves.

The frozen hyperparameters were tuned for 30 columns.  ``feature_fraction``
in particular has a different meaning at 35 columns, so a single re-tuned
variant (MF_tuned) is reported separately and clearly labelled, never mixed
into the headline comparison.

New features (computed on the RAW, de-duplicated point cloud, exactly as in
``paper_tooling/feature_screen.py``; all five are invariant to the PCA
rotation the shipped extractor applies, so the frame is immaterial):

    mst_topology_straightness
    mst_topology_deg2_straight_mean
    degeneracy_pca_effective_rank
    local_id_evr1_median_k5
    local_id_pr_mean_k5

Stages (each cached, so the expensive ones run once):

    extract         5 features for all 106,272 rows of tsp_features_v3.csv
                        -> paper_tooling/features_extended.csv
    extract-tsplib  5 features for the 111 TSPLIB instances
                        -> paper_tooling/features_extended_tsplib.csv
    train           M0 (+ reproduction assertions) and MF
                        -> paper_tooling/feature_models/{M0,MF}.joblib
    retune          one Optuna re-tune of MF's hyperparameters
                        -> paper_tooling/feature_models/MF_tuned.joblib
    eval            every slice, the pre-registered criteria, SHAP
    all             extract + extract-tsplib + train + eval

The 2D benchmark slice reuses ``paper_tooling/feature_screen_frame.csv``,
whose 30-feature block was recomputed with the canonical extractor and which
reproduces the reported baseline exactly (verified in ``_check_frame``).

Never writes to ``lgbm_model_v3/``.
"""

from __future__ import annotations

import os
import sys

# Single-thread BLAS inside every worker: 20 processes x multithreaded BLAS
# thrashes on Windows.  Must precede the numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

import argparse
import json
import struct
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HERE = os.path.join(ROOT, "paper_tooling")
ND_INST = os.path.join(ROOT, "instances")
ND_FEATS = os.path.join(ROOT, "tsp_features_v3.csv")
SHIPPED_MODEL = os.path.join(ROOT, "lgbm_model_v3", "lgbm_alpha_model_v3.joblib")
BEST_PARAMS = os.path.join(ROOT, "lgbm_model_v3", "best_params_v3.json")

FRAME_2D = os.path.join(HERE, "feature_screen_frame.csv")
TSPLIB_FEATS = os.path.join(HERE, "tsplib_features_v3.csv")
TSPLIB_DIR = os.path.join(ROOT, "tsplib_benchmark")
TSPLIB_INST = os.path.join(TSPLIB_DIR, "instances")

EXT_CSV = os.path.join(HERE, "features_extended.csv")
EXT_TSPLIB_CSV = os.path.join(HERE, "features_extended_tsplib.csv")
MODEL_DIR = os.path.join(HERE, "feature_models")
REPORT_JSON = os.path.join(HERE, "feature_retrain_report.json")
SLICES_CSV = os.path.join(HERE, "feature_retrain_slices.csv")
CRITERIA_CSV = os.path.join(HERE, "feature_retrain_criteria.csv")
SHAP_CSV = os.path.join(HERE, "feature_retrain_shap.csv")
OPTUNA_DB = os.path.join(MODEL_DIR, "optuna_mf.db")

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100
MAX_BOOST_ROUND = 3000
MIN_BOOST_ROUND = 100
N_WORKERS = max(1, min(20, os.cpu_count() or 4))
MAX_MDS_DIM = 100

NEW_FEATURES = [
    "mst_topology_straightness",
    "mst_topology_deg2_straight_mean",
    "degeneracy_pca_effective_rank",
    "local_id_evr1_median_k5",
    "local_id_pr_mean_k5",
]

DROP_COLS = ["instance_name", "optimal_cost", "alpha", "split", "grid_size",
             "mst_total_length"]

# 2D benchmark sub-generator -> paper generator class.
GEN_CLASSES = {
    "Isotropic": {"random", "normal", "triangular", "truncated_exponential"},
    "Biased": {"squeezed_uniform", "uniform_triangular", "triangular_squeezed",
               "correlated"},
    "Geometric": {"grid", "boundary", "x_central"},
    "Clustered": {"clustered"},
    "LineNoise": {"line_noise"},
}
GEN_TO_CLASS = {g: c for c, gs in GEN_CLASSES.items() for g in gs}

# Shipped-model baselines, all independently reproduced by ``_check_frame``.
BASE = {
    "nd_test_mape": 0.8769192747205671,
    "best_iteration": 2031,
    "linenoise_mape": 11.591801,
    "linenoise_mspe": -11.574307,
    "grid_mspe": 8.480433,
    "tsplib_euc_mape": 3.2712819,
    "tsplib_noneuc_mape": 4.6326684,
    "isotropic_mape": 1.836292,
    "biased_mape": 1.993289,
    "clustered_mape": 2.606236,
    "linenoise_slope": 0.2941405,
}


# ==========================================================================
# 1. feature extraction
# ==========================================================================

_CACHE: dict = {}


def _groups():
    if "g" not in _CACHE:
        from features_ext import group_degeneracy, group_local_id, group_mst_topology
        from mst_utils import compute_mst
        _CACHE["g"] = (group_local_id, group_degeneracy, group_mst_topology,
                       compute_mst)
    return _CACHE["g"]


def _five(coords: np.ndarray, mst=None) -> dict:
    """The five shortlisted features on a point cloud.

    Every candidate group is evaluated in full and then subset, rather than
    calling a trimmed fast path, so the values are bit-identical to the ones
    the screen selected on.
    """
    g_lid, g_deg, g_top, compute_mst = _groups()
    if mst is None:
        mst = compute_mst(coords)
    out = {}
    out.update(g_lid.compute(coords))
    out.update(g_deg.compute(coords))
    out.update(g_top.compute(coords, mst))
    return {k: float(out[k]) for k in NEW_FEATURES}


def _load_bin(name: str) -> np.ndarray:
    """De-duplicated raw coordinates from an ``instances/*.bin`` file."""
    with open(os.path.join(ND_INST, name + ".bin"), "rb") as f:
        n, d, _grid = struct.unpack("III", f.read(12))
        dist_len = struct.unpack("I", f.read(4))[0]
        f.read(dist_len)
        buf = f.read(n * d * 4)
    coords = np.frombuffer(buf, dtype=np.float32).reshape(n, d)
    return np.unique(coords, axis=0).astype(np.float64)


def _worker_corpus(name: str) -> dict:
    try:
        row = _five(_load_bin(name))
    except Exception as exc:
        row = {k: np.nan for k in NEW_FEATURES}
        row["_error"] = repr(exc)
    row["instance_name"] = name
    return row


def stage_extract(force: bool = False) -> pd.DataFrame:
    if os.path.exists(EXT_CSV) and not force:
        df = pd.read_csv(EXT_CSV)
        print(f"[extract] cached {EXT_CSV} ({len(df)} rows)")
        return df

    names = pd.read_csv(ND_FEATS, usecols=["instance_name"])["instance_name"].tolist()
    print(f"[extract] {len(names)} instances on {N_WORKERS} workers ...")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, r in enumerate(pool.map(_worker_corpus, names, chunksize=64), 1):
            rows.append(r)
            if i % 10000 == 0:
                el = time.perf_counter() - t0
                print(f"    {i}/{len(names)}  {el:.0f}s  eta {el/i*(len(names)-i):.0f}s",
                      flush=True)
    df = pd.DataFrame(rows)
    n_err = int(df.get("_error", pd.Series(dtype=object)).notna().sum()) if "_error" in df else 0
    print(f"[extract] done in {time.perf_counter()-t0:.0f}s, errors={n_err}")
    if n_err:
        print(df.loc[df["_error"].notna(), ["instance_name", "_error"]].head(10).to_string())
        raise SystemExit("extraction errors -- refusing to continue")
    df = df[["instance_name"] + NEW_FEATURES]
    df.to_csv(EXT_CSV, index=False)
    print(f"[extract] -> {EXT_CSV}")
    return df


def _worker_tsplib(name: str) -> dict:
    """Five features for one TSPLIB instance.

    Rule, identical to every other corpus here: the candidates are functions
    of the point cloud the estimator actually sees.  For a native Euclidean
    instance that is the raw node coordinates; for a non-Euclidean instance it
    is the classical-MDS embedding the hybrid path builds, with its own
    Euclidean MST in that space.
    """
    try:
        sys.path.insert(0, TSPLIB_DIR)
        from tsplib_parser import parse_tsplib_file
        from classical_mds import classical_mds

        info = parse_tsplib_file(os.path.join(TSPLIB_INST, name + ".tsp"))
        if info["is_native_euclidean"] and info["raw_coords"] is not None:
            coords = np.asarray(info["raw_coords"], dtype=np.float64)
            mode = "native"
        else:
            X, _e, _m = classical_mds(info["distance_matrix"], max_dim=MAX_MDS_DIM)
            coords = np.asarray(X, dtype=np.float64)
            mode = "hybrid"
        coords = np.unique(coords, axis=0)
        row = _five(coords)
        row["_mode_check"] = mode
    except Exception as exc:
        row = {k: np.nan for k in NEW_FEATURES}
        row["_error"] = repr(exc)
    row["instance_name"] = name
    return row


def stage_extract_tsplib(force: bool = False) -> pd.DataFrame:
    if os.path.exists(EXT_TSPLIB_CSV) and not force:
        df = pd.read_csv(EXT_TSPLIB_CSV)
        print(f"[tsplib] cached {EXT_TSPLIB_CSV} ({len(df)} rows)")
        return df
    names = pd.read_csv(TSPLIB_FEATS)["instance_name"].astype(str).tolist()
    print(f"[tsplib] {len(names)} instances ...")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        rows = list(pool.map(_worker_tsplib, names, chunksize=1))
    df = pd.DataFrame(rows)
    if "_error" in df and df["_error"].notna().any():
        print(df.loc[df["_error"].notna(), ["instance_name", "_error"]].to_string())
        raise SystemExit("TSPLIB extraction errors -- refusing to continue")
    print(f"[tsplib] done in {time.perf_counter()-t0:.0f}s")
    df = df[["instance_name"] + NEW_FEATURES]
    df.to_csv(EXT_TSPLIB_CSV, index=False)
    print(f"[tsplib] -> {EXT_TSPLIB_CSV}")
    return df


# ==========================================================================
# 2. data assembly
# ==========================================================================

def load_corpus() -> tuple[pd.DataFrame, list, list]:
    """tsp_features_v3.csv + the 5 new columns, splits untouched."""
    df = pd.read_csv(ND_FEATS)
    base_order = list(df.columns)
    ext = pd.read_csv(EXT_CSV)

    before = df[["instance_name", "split"]].copy()
    df = df.merge(ext, on="instance_name", how="left", validate="one_to_one")

    # Split integrity: same rows, same order, same assignment.
    assert len(df) == len(before), "merge changed row count"
    assert (df["instance_name"].values == before["instance_name"].values).all(), \
        "merge reordered rows"
    assert (df["split"].values == before["split"].values).all(), "split changed"
    assert df[NEW_FEATURES].notna().all().all(), "missing new features"
    assert list(df.columns)[:len(base_order)] == base_order, "base columns moved"

    mst_div = df["mst_total_length"].replace(0, 1e-9)
    df["alpha"] = (df["optimal_cost"] / mst_div).clip(1.0, 2.0)

    feats30 = [c for c in base_order if c not in DROP_COLS]
    return df, feats30, feats30 + NEW_FEATURES


def load_2d() -> pd.DataFrame:
    d = pd.read_csv(FRAME_2D, low_memory=False)
    d = d[d["source"] == "2D"].reset_index(drop=True)
    d["gen_class"] = d["generator"].map(GEN_TO_CLASS)
    assert d["gen_class"].notna().all(), "unmapped 2D generator"
    assert d[NEW_FEATURES].notna().all().all(), "missing 2D new features"
    return d


def load_tsplib() -> pd.DataFrame:
    d = pd.read_csv(TSPLIB_FEATS)
    ext = pd.read_csv(EXT_TSPLIB_CSV)
    d = d.merge(ext, on="instance_name", how="left", validate="one_to_one")
    assert d[NEW_FEATURES].notna().all().all(), "missing TSPLIB new features"
    return d


# ==========================================================================
# 3. metrics
# ==========================================================================

def pct_err(model, df: pd.DataFrame, feats: list) -> np.ndarray:
    pred_alpha = np.clip(model.predict(df[feats], num_iteration=model.best_iteration_),
                         1.0, 2.0)
    pred_cost = pred_alpha * df["mst_total_length"].to_numpy()
    cost = df["optimal_cost"].to_numpy()
    return (pred_cost - cost) / cost


def summarize(err: np.ndarray) -> dict:
    err = np.asarray(err, dtype=float)
    return {
        "N": int(err.size),
        "MAPE": float(np.mean(np.abs(err)) * 100.0),
        "SDPE": float(np.std(err, ddof=1) * 100.0) if err.size > 1 else float("nan"),
        "MSPE": float(np.mean(err) * 100.0),
    }


def _check_frame(shipped) -> None:
    """The evaluation frames must reproduce the shipped model's reported error."""
    from tsplib_benchmark.exclusions import STRUCTURAL_TRIANGLE_VIOLATORS as SV
    f30 = list(shipped.feature_name_)

    d2 = load_2d()
    e2 = pct_err(shipped, d2, f30)
    ln = summarize(e2[d2["generator"] == "line_noise"])
    gr = summarize(e2[d2["generator"] == "grid"])
    assert abs(ln["MAPE"] - BASE["linenoise_mape"]) < 1e-3, ln
    assert abs(ln["MSPE"] - BASE["linenoise_mspe"]) < 1e-3, ln
    assert abs(gr["MSPE"] - BASE["grid_mspe"]) < 1e-3, gr

    dt = load_tsplib()
    et = pct_err(shipped, dt, f30)
    euc = summarize(et[dt["edge_weight_type"] == "EUC_2D"])
    ne = summarize(et[(dt["edge_weight_type"] != "EUC_2D")
                      & (~dt["instance_name"].isin(SV))])
    assert euc["N"] == 78 and abs(euc["MAPE"] - BASE["tsplib_euc_mape"]) < 1e-3, euc
    assert ne["N"] == 23 and abs(ne["MAPE"] - BASE["tsplib_noneuc_mape"]) < 1e-3, ne
    print("[check] 2D + TSPLIB frames reproduce the shipped baseline exactly")


# ==========================================================================
# 4. training
# ==========================================================================

def fit_model(df: pd.DataFrame, feats: list, params: dict, tag: str):
    import lightgbm as lgb

    tr, va = df["split"] == "train", df["split"] == "val"
    model = lgb.LGBMRegressor(**params, n_estimators=MAX_BOOST_ROUND,
                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    t0 = time.perf_counter()
    model.fit(df.loc[tr, feats], df.loc[tr, "alpha"],
              eval_set=[(df.loc[va, feats], df.loc[va, "alpha"])], eval_metric="rmse",
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    print(f"[fit:{tag}] {len(feats)} feats, best_iteration={model.best_iteration_}, "
          f"{time.perf_counter()-t0:.0f}s")
    return model


def stage_train(force: bool = False, retune_trials: int = 0):
    import joblib

    os.makedirs(MODEL_DIR, exist_ok=True)
    params = json.load(open(BEST_PARAMS))
    df, f30, fall = load_corpus()
    shipped = joblib.load(SHIPPED_MODEL)
    assert list(shipped.feature_name_) == f30, "feature order drift vs shipped model"

    p_m0 = os.path.join(MODEL_DIR, "M0.joblib")
    p_mf = os.path.join(MODEL_DIR, "MF.joblib")

    # --- M0: must reproduce the shipped artifact ---------------------------
    m0 = joblib.load(p_m0) if (os.path.exists(p_m0) and not force) else None
    if m0 is None:
        m0 = fit_model(df, f30, params, "M0")
        joblib.dump(m0, p_m0)

    te = df["split"] == "test"
    d_te = df.loc[te]
    m0_mape = summarize(pct_err(m0, d_te, f30))["MAPE"]
    p_new = m0.predict(df[f30], num_iteration=m0.best_iteration_)
    p_old = shipped.predict(df[f30], num_iteration=shipped.best_iteration_)
    max_diff = float(np.max(np.abs(p_new - p_old)))
    print(f"[M0] best_iteration={m0.best_iteration_} (expect {BASE['best_iteration']})")
    print(f"[M0] ND test MAPE={m0_mape:.4f} (expect {BASE['nd_test_mape']:.4f})")
    print(f"[M0] max |pred - shipped| = {max_diff:.3e} (expect 0.0)")
    if m0.best_iteration_ != BASE["best_iteration"]:
        raise SystemExit("M0 REPRODUCTION FAILED: best_iteration mismatch")
    if abs(m0_mape - BASE["nd_test_mape"]) > 5e-4:
        raise SystemExit("M0 REPRODUCTION FAILED: ND test MAPE mismatch")
    if max_diff != 0.0:
        raise SystemExit("M0 REPRODUCTION FAILED: predictions differ from the artifact")
    print("[M0] reproduction verified: bit-identical to lgbm_alpha_model_v3.joblib")

    # --- MF: frozen hyperparameters, 35 features ---------------------------
    if os.path.exists(p_mf) and not force:
        mf = joblib.load(p_mf)
        print(f"[MF] cached, best_iteration={mf.best_iteration_}")
    else:
        mf = fit_model(df, fall, params, "MF")
        joblib.dump(mf, p_mf)
        print(f"[MF] -> {p_mf}")

    # --- MF_tuned: one re-tune, reported separately ------------------------
    mft = None
    p_mft = os.path.join(MODEL_DIR, "MF_tuned.joblib")
    if retune_trials > 0:
        mft = stage_retune(df, fall, retune_trials, force=force)
    elif os.path.exists(p_mft):
        mft = joblib.load(p_mft)
        print(f"[MF_tuned] cached, best_iteration={mft.best_iteration_}")

    return df, f30, fall, m0, mf, mft


def stage_retune(df: pd.DataFrame, fall: list, n_trials: int, force: bool = False):
    """One Optuna re-tune of MF, same search space / sampler / objective as V3."""
    import joblib
    import lightgbm as lgb
    import optuna
    from optuna.integration import LightGBMPruningCallback
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tr, va = df["split"] == "train", df["split"] == "val"
    X_tr, y_tr = df.loc[tr, fall], df.loc[tr, "alpha"]
    X_va, y_va = df.loc[va, fall], df.loc[va, "alpha"]
    mst_va = df.loc[va, "mst_total_length"].to_numpy()
    cost_va = df.loc[va, "optimal_cost"].to_numpy()

    def objective(trial):
        p = {
            "objective": "regression_l2", "metric": "rmse", "boosting_type": "gbdt",
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 64, 512),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
        b = lgb.train(p, dtr, num_boost_round=MAX_BOOST_ROUND, valid_sets=[dva],
                      valid_names=["val"],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                 LightGBMPruningCallback(trial, metric="rmse",
                                                         valid_name="val")])
        pa = np.clip(b.predict(X_va, num_iteration=b.best_iteration), 1.0, 2.0)
        err = (pa * mst_va - cost_va) / np.where(cost_va == 0, 1e-9, cost_va)
        return float(np.std(err, ddof=1))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, group=True, seed=RANDOM_STATE),
        pruner=HyperbandPruner(min_resource=MIN_BOOST_ROUND, max_resource=MAX_BOOST_ROUND),
        study_name="lgbm_mf", storage=f"sqlite:///{OPTUNA_DB}", load_if_exists=True)
    done = len([t for t in study.trials if t.state.is_finished()])
    todo = max(0, n_trials - done)
    print(f"[retune] {done} trials on record, running {todo} more ...")
    if todo:
        study.optimize(objective, n_trials=todo)
    print(f"[retune] best val SDPE={study.best_value*100:.4f}%  params={study.best_params}")
    with open(os.path.join(MODEL_DIR, "best_params_mf.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
    mft = fit_model(df, fall, study.best_params, "MF_tuned")
    joblib.dump(mft, os.path.join(MODEL_DIR, "MF_tuned.joblib"))
    return mft


# ==========================================================================
# 5. evaluation
# ==========================================================================

def evaluate(models: dict, d2: pd.DataFrame, d_te: pd.DataFrame,
             dt: pd.DataFrame) -> pd.DataFrame:
    from tsplib_benchmark.exclusions import STRUCTURAL_TRIANGLE_VIOLATORS as SV

    ne_mask = (dt["edge_weight_type"] != "EUC_2D") & (~dt["instance_name"].isin(SV))
    rows = []
    for tag, (model, feats) in models.items():
        e2 = pct_err(model, d2, feats)
        for cls in ["Isotropic", "Biased", "Clustered", "Geometric", "LineNoise"]:
            rows.append({"model": tag, "slice": f"2D {cls}",
                         **summarize(e2[(d2["gen_class"] == cls).to_numpy()])})
        rows.append({"model": tag, "slice": "2D grid sub-gen",
                     **summarize(e2[(d2["generator"] == "grid").to_numpy()])})
        rows.append({"model": tag, "slice": "2D overall", **summarize(e2)})
        rows.append({"model": tag, "slice": "ND test",
                     **summarize(pct_err(model, d_te, feats))})
        et = pct_err(model, dt, feats)
        rows.append({"model": tag, "slice": "TSPLIB EUC_2D",
                     **summarize(et[(dt["edge_weight_type"] == "EUC_2D").to_numpy()])})
        rows.append({"model": tag, "slice": "TSPLIB non-EUC screened",
                     **summarize(et[ne_mask.to_numpy()])})
    return pd.DataFrame(rows)


def linenoise_by_n(models: dict, d2: pd.DataFrame) -> pd.DataFrame:
    ln = d2[d2["generator"] == "line_noise"]
    buckets = [("<=100", ln["n_customers"] <= 100),
               ("101-400", (ln["n_customers"] > 100) & (ln["n_customers"] <= 400)),
               ("401-1000", ln["n_customers"] > 400)]
    rows = []
    for tag, (model, feats) in models.items():
        e = pct_err(model, ln, feats)
        for label, m in buckets:
            rows.append({"model": tag, "n_bucket": label, **summarize(e[m.to_numpy()])})
    return pd.DataFrame(rows)


def mechanism(models: dict, d2: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats
    ln = d2[(d2["generator"] == "line_noise") & (d2["n_customers"] >= 200)]
    true_alpha = np.clip(ln["optimal_cost"] / ln["mst_total_length"], 1.0, 2.0).to_numpy()
    rows = []
    for tag, (model, feats) in models.items():
        pa = np.clip(model.predict(ln[feats], num_iteration=model.best_iteration_), 1.0, 2.0)
        lr = stats.linregress(true_alpha, pa)
        rows.append({"model": tag, "N": len(ln), "slope": float(lr.slope),
                     "intercept": float(lr.intercept), "pearson_r": float(lr.rvalue),
                     "pred_alpha_std": float(np.std(pa, ddof=1)),
                     "true_alpha_std": float(np.std(true_alpha, ddof=1))})
    return pd.DataFrame(rows)


def criteria(sl: pd.DataFrame, mech: pd.DataFrame, tag: str) -> pd.DataFrame:
    def val(slice_name, col="MAPE"):
        return float(sl[(sl.model == tag) & (sl.slice == slice_name)][col].iloc[0])

    slope = float(mech[mech.model == tag]["slope"].iloc[0])
    ln_mape, ln_mspe = val("2D LineNoise"), val("2D LineNoise", "MSPE")
    grid_mspe = val("2D grid sub-gen", "MSPE")
    nd = val("ND test")
    euc = val("TSPLIB EUC_2D")
    ne = val("TSPLIB non-EUC screened")
    iso, bia, clu = val("2D Isotropic"), val("2D Biased"), val("2D Clustered")
    worst_reg = max(iso - BASE["isotropic_mape"], bia - BASE["biased_mape"],
                    clu - BASE["clustered_mape"])

    c = [
        (1, "LineNoise MAPE < 5.00", f"{ln_mape:.3f} (from 11.591)", ln_mape < 5.00),
        (2, "LineNoise MSPE within +/-3.0", f"{ln_mspe:+.3f} (from -11.574)",
         abs(ln_mspe) <= 3.0),
        (3, "grid MSPE improves >= 2.0 pts",
         f"{grid_mspe:+.3f} (from +8.480, delta {BASE['grid_mspe']-grid_mspe:+.3f})",
         (BASE["grid_mspe"] - grid_mspe) >= 2.0),
        (4, "ND test MAPE regression <= 0.05",
         f"{nd:.4f} (from 0.8769, delta {nd-BASE['nd_test_mape']:+.4f})",
         (nd - BASE["nd_test_mape"]) <= 0.05),
        (5, "TSPLIB EUC_2D MAPE regression <= 0.15",
         f"{euc:.4f} (from 3.2713, delta {euc-BASE['tsplib_euc_mape']:+.4f})",
         (euc - BASE["tsplib_euc_mape"]) <= 0.15),
        (6, "no Iso/Biased/Clustered regression > 0.15",
         f"worst delta {worst_reg:+.4f} (Iso {iso-BASE['isotropic_mape']:+.3f}, "
         f"Bia {bia-BASE['biased_mape']:+.3f}, Clu {clu-BASE['clustered_mape']:+.3f})",
         worst_reg <= 0.15),
        (7, "TSPLIB non-EUC screened regression <= 0.30",
         f"{ne:.4f} (from 4.6327, delta {ne-BASE['tsplib_noneuc_mape']:+.4f})",
         (ne - BASE["tsplib_noneuc_mape"]) <= 0.30),
        (8, "LineNoise alpha slope (n>=200) >= 0.70",
         f"{slope:.4f} (from 0.294)", slope >= 0.70),
    ]
    return pd.DataFrame([{"model": tag, "criterion": i, "test": t, "value": v,
                          "verdict": "PASS" if ok else "FAIL"} for i, t, v, ok in c])


def shap_new(model, feats: list, d_te: pd.DataFrame, tag: str,
             n_sample: int = 5000) -> pd.DataFrame:
    import shap
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(d_te), size=min(n_sample, len(d_te)), replace=False)
    X = d_te.iloc[idx][feats]
    vals = shap.TreeExplainer(model).shap_values(X)
    mean_abs = np.abs(vals).mean(axis=0)
    out = pd.DataFrame({"model": tag, "feature": feats, "mean_abs_shap": mean_abs})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["share_pct"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum() * 100.0
    out["is_new"] = out["feature"].isin(NEW_FEATURES)
    return out


# ==========================================================================
# 6. driver
# ==========================================================================

def fmt(df: pd.DataFrame, floats=("MAPE", "SDPE", "MSPE")) -> str:
    d = df.copy()
    for c in floats:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.4f}")
    return d.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["extract", "extract-tsplib", "train", "retune", "eval", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retune-trials", type=int, default=0)
    args = ap.parse_args()

    if args.stage in ("extract", "all"):
        stage_extract(args.force)
    if args.stage in ("extract-tsplib", "all"):
        stage_extract_tsplib(args.force)
    if args.stage in ("extract", "extract-tsplib"):
        return

    import joblib
    shipped = joblib.load(SHIPPED_MODEL)
    _check_frame(shipped)

    df, f30, fall, m0, mf, mft = stage_train(args.force, args.retune_trials)
    if args.stage in ("train", "retune"):
        return

    d2, dt = load_2d(), load_tsplib()
    d_te = df.loc[df["split"] == "test"]
    models = {"M0": (m0, f30), "MF": (mf, fall)}
    if mft is not None:
        models["MF_tuned"] = (mft, fall)

    sl = evaluate(models, d2, d_te, dt)
    ln = linenoise_by_n(models, d2)
    mech = mechanism(models, d2)
    crit = pd.concat([criteria(sl, mech, t) for t in models], ignore_index=True)
    sh = pd.concat([shap_new(m, f, d_te, t) for t, (m, f) in models.items() if t != "M0"],
                   ignore_index=True)

    sl.to_csv(SLICES_CSV, index=False)
    crit.to_csv(CRITERIA_CSV, index=False)
    sh.to_csv(SHAP_CSV, index=False)

    print("\n================ MODEL BY SLICE ================")
    print(fmt(sl.pivot(index="slice", columns="model",
                       values=["N", "MAPE", "SDPE", "MSPE"]).reset_index(), floats=())
          if False else fmt(sl))
    print("\n============ LineNoise by n bucket =============")
    print(fmt(ln))
    print("\n================= MECHANISM ====================")
    print(mech.to_string(index=False))
    print("\n============ PRE-REGISTERED CRITERIA ===========")
    for t in models:
        sub = crit[crit.model == t]
        print(f"\n-- {t} --")
        print(sub[["criterion", "test", "value", "verdict"]].to_string(index=False))
        print(f"   {int((sub.verdict=='PASS').sum())}/8 PASS")
    print("\n============== SHAP (new features) =============")
    print(sh[sh.is_new].to_string(index=False))
    print("\n-- top 10 overall --")
    for t in sh.model.unique():
        print(sh[sh.model == t].head(10).to_string(index=False))

    n_pass = int((crit[crit.model == "MF"].verdict == "PASS").sum())
    verdict = "ADOPT" if n_pass == 8 else "REJECT"
    print(f"\n=== VERDICT (MF, frozen hyperparameters): {verdict} ({n_pass}/8) ===")

    json.dump({"slices": sl.to_dict("records"), "linenoise_by_n": ln.to_dict("records"),
               "mechanism": mech.to_dict("records"), "criteria": crit.to_dict("records"),
               "verdict": verdict, "n_pass_MF": n_pass},
              open(REPORT_JSON, "w"), indent=2)
    print(f"\nwrote {SLICES_CSV}\n      {CRITERIA_CSV}\n      {SHAP_CSV}\n      {REPORT_JSON}")


if __name__ == "__main__":
    main()
