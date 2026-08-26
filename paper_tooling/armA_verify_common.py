"""Shared loader / fitter / scorer for the arm-A robustness audit.

Independent re-implementation of the arm-A recipe that cross-checks itself
against support_arms_study.py by requiring that (a) an unaugmented refit
reproduces the frozen artifact bit-for-bit on predictions and (b) a full-dose
refit reproduces the shipped A.joblib bit-for-bit.

Writes nothing except armA_verify_* artifacts.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
for _p in (ROOT, HERE, ROOT / "lgbm_model_v3", ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FROZEN = ROOT / "lgbm_model_v3" / "gart2_final.joblib"
V4_TABLE = ROOT / "tsp_features_v4.csv"
FEAT_CACHE = HERE / "v4_study_feature_cache.csv"
GEN_MAP = HERE / "augmentation_2d_features.csv"
CACHE_PKL = HERE / "armA_verify_datacache.pkl"

SEED = 42
NUM_THREADS = 6
MAX_BOOST_ROUND = 4000
EARLY_STOP = 100
EPS = 1e-6
ALPHA_CLIP = (1.0, 2.0)

V3_FROZEN = {
    "learning_rate": 0.02597081024148143, "num_leaves": 148,
    "reg_alpha": 0.07764927124026656, "reg_lambda": 1.1865278867536643e-05,
    "feature_fraction": 0.5481439957854186, "bagging_fraction": 0.6087321895536237,
    "bagging_freq": 6, "min_child_samples": 23, "max_depth": -1,
}
MONO_COLS = ("n_customers", "dimension")


def to_z(alpha) -> np.ndarray:
    u = np.clip(np.asarray(alpha, dtype=np.float64) - 1.0, EPS, 1.0 - EPS)
    return np.log(u / (1.0 - u))


def to_alpha(z) -> np.ndarray:
    return 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def frozen_features() -> list[str]:
    import joblib
    return list(joblib.load(FROZEN).feature_name())


# ---------------------------------------------------------------- data
def build_cache() -> dict:
    feats31 = frozen_features()

    df = pd.read_csv(V4_TABLE)
    keep = ["instance_name", "split", "optimal_cost", "mst_total_length"] + feats31
    keep = list(dict.fromkeys(keep))
    df = df[keep].copy()
    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(*ALPHA_CLIP)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)

    base = pd.read_csv(HERE / "augment_features_v3.csv")
    gnn = pd.read_csv(HERE / "augment_greedy_nn.csv")
    aug = base.merge(gnn, on="instance_name", how="inner", validate="one_to_one")
    m = aug["mst_total_length"].replace(0, np.nan)
    aug["alpha"] = (aug["optimal_cost"] / m).clip(*ALPHA_CLIP)
    aug = aug.dropna(subset=["alpha"]).reset_index(drop=True)
    aug["split"] = "train"
    aug = aug[["instance_name", "split", "optimal_cost", "mst_total_length",
               "alpha", "family", "group"] + feats31].copy()

    # leakage assertions, re-derived not inherited
    names = set(aug.instance_name.astype(str))
    for sp in ("train", "val", "test"):
        ov = names & set(df.loc[df.split == sp, "instance_name"].astype(str))
        assert not ov, f"augment overlaps corpus/{sp}"

    C = pd.read_csv(FEAT_CACHE, low_memory=False)
    gen = pd.read_csv(GEN_MAP, usecols=["instance_name", "generator"])
    C = C.merge(gen, left_on="instance", right_on="instance_name", how="left")
    for st in C.stratum.unique():
        if st == "augment":
            continue
        ov = names & set(C.loc[C.stratum == st, "instance"].astype(str))
        assert not ov, f"augment overlaps scored stratum {st}"

    cache = {"corpus": df, "aug": aug, "cache": C, "feats31": feats31}
    with open(CACHE_PKL, "wb") as fh:
        pickle.dump(cache, fh, protocol=5)
    return cache


def load_cache(rebuild: bool = False) -> dict:
    if CACHE_PKL.exists() and not rebuild:
        with open(CACHE_PKL, "rb") as fh:
            return pickle.load(fh)
    return build_cache()


# ---------------------------------------------------------------- fit
def fit(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], seed: int = SEED):
    import lightgbm as lgb

    mono = [-1 if f in MONO_COLS else 0 for f in feats]
    params = dict(V3_FROZEN)
    params.update({
        "objective": "regression_l2", "metric": "None",
        "boosting_type": "gbdt", "seed": seed,
        "num_threads": NUM_THREADS, "verbosity": -1, "feature_pre_filter": False,
        "monotone_constraints": mono, "monotone_constraints_method": "basic",
    })
    mst_v = val["mst_total_length"].to_numpy()
    cost_v = val["optimal_cost"].to_numpy()

    def feval(preds, _ds):
        a = to_alpha(preds)
        err = (a * mst_v - cost_v) / cost_v
        return "cost_mape", float(np.mean(np.abs(err)) * 100.0), False

    dtr = lgb.Dataset(train[feats], label=to_z(train["alpha"].to_numpy()))
    dvl = lgb.Dataset(val[feats], label=to_z(val["alpha"].to_numpy()), reference=dtr)
    return lgb.train(params, dtr, num_boost_round=MAX_BOOST_ROUND,
                     valid_sets=[dvl], valid_names=["val"], feval=feval,
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                lgb.log_evaluation(0)])


# ---------------------------------------------------------------- score
def _summary(e: np.ndarray) -> dict:
    e = np.asarray(e, dtype=float)
    return {"n": int(e.size), "mape": float(np.mean(np.abs(e))),
            "sdpe": float(np.std(e, ddof=1)) if e.size > 1 else float("nan"),
            "mspe": float(np.mean(e))}


def score_frame(model, feats, frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame[(frame.status == "ok") & frame[feats].notna().all(axis=1)
               & frame.true_cost.notna()].copy()
    a = np.clip(to_alpha(model.predict(ok[feats], num_iteration=model.best_iteration)),
                *ALPHA_CLIP)
    ok["pred_alpha"] = a
    ok["pred_cost"] = a * ok["mst_total_length"].to_numpy()
    ok["err_pct"] = (ok["pred_cost"] - ok["true_cost"]) / ok["true_cost"] * 100.0
    return ok


def group_of(gen: pd.Series) -> np.ndarray:
    return np.where(gen == "grid", "grid",
                    np.where(gen == "line_noise", "line_noise",
                             np.where(gen == "clustered", "cluster", "others")))


def metrics(model, feats, C: pd.DataFrame, want_per_instance: bool = False) -> dict:
    out: dict = {}
    scored = {}
    for st, g in C.groupby("stratum"):
        ok = score_frame(model, feats, g)
        scored[st] = ok
        s = _summary(ok["err_pct"].to_numpy())
        out[f"{st}_mape"] = s["mape"]
        out[f"{st}_sdpe"] = s["sdpe"]
        out[f"{st}_mspe"] = s["mspe"]
        out[f"{st}_n"] = s["n"]

    b2 = scored["bench2d"].copy()
    b2["grp"] = group_of(b2["generator"])
    for grp, g in b2.groupby("grp"):
        s = _summary(g["err_pct"].to_numpy())
        out[f"b2_{grp}_mape"] = s["mape"]
        out[f"b2_{grp}_mspe"] = s["mspe"]
        out[f"b2_{grp}_sdpe"] = s["sdpe"]

    ln = b2[(b2.generator == "line_noise") & (b2.n_customers >= 200)]
    ta = np.clip(ln["true_cost"] / ln["mst_total_length"], *ALPHA_CLIP).to_numpy()
    lr = stats.linregress(ta, ln["pred_alpha"].to_numpy())
    out["linenoise_slope"] = float(lr.slope)
    out["linenoise_slope_n"] = int(len(ln))
    out["linenoise_slope_r"] = float(lr.rvalue)

    gr = b2[(b2.generator == "grid") & (b2.n_customers >= 200)]
    ta = np.clip(gr["true_cost"] / gr["mst_total_length"], *ALPHA_CLIP).to_numpy()
    lrg = stats.linregress(ta, gr["pred_alpha"].to_numpy())
    out["grid_slope"] = float(lrg.slope)

    out["trees"] = int(model.num_trees())
    out["best_iter"] = int(model.best_iteration)
    if want_per_instance:
        out["_scored"] = scored
    return out


KEYS = ["nd_test_mape", "nd_test_sdpe", "bench2d_mape", "bench2d_sdpe",
        "b2_grid_mape", "b2_grid_mspe", "b2_line_noise_mape", "b2_line_noise_mspe",
        "b2_cluster_mape", "b2_others_mape", "linenoise_slope", "grid_slope",
        "tsplib_euc2d_mape", "tsplib_euc2d_sdpe", "tsplib_noneuc_mape",
        "augment_mape", "trees"]
