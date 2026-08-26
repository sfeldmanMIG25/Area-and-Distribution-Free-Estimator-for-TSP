"""Train GART 2.0 -- the paper's production alpha estimator.

Feature set: V3's 30 features plus ``greedy_nn_over_mst`` (31 total).

Deliberately EXCLUDED:
  * ``grid_size``          -- generator fingerprint, leaks the instance family.
  * ``mst_total_length``   -- alpha = optimal_cost / mst_total_length, so
                              feeding MST magnitude to the model lets it
                              recover a component of the label. The V4
                              prototype re-admitted this column; a controlled
                              ablation showed it is worth <= 0.02 pp on every
                              stratum and actively hurts the out-of-distribution
                              corpus (+0.04 pp), so it stays out.

Protocol (unchanged from V3, deliberately):
  * Fit on ``split == 'train'`` only, early-stopped on ``split == 'val'``.
  * ``val`` is a checkpoint signal and is never trained on. There is no
    refit-on-train+val step -- the V4 prototype did that and it cost 0.015 pp
    on the ND test split and 0.134 pp on TSPLIB.
  * ``split == 'test'`` (d=100 held out entirely) is touched once, at the end.

Tuning:
  * Optuna TPE, seed 42, single objective = VALIDATION SDPE at cost level.
    SDPE is the paper's primary metric; the consistency thesis rests on it.
  * Every trial also records val MAPE, so the MAPE-minimising configuration
    is reported alongside the SDPE-minimising winner to expose the trade.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent

DATA_FILE = REPO / "tsp_features_v4.csv"      # only table carrying greedy_nn_over_mst
MODEL_OUT = THIS_DIR / "gart2_alpha_model.joblib"
PARAMS_OUT = THIS_DIR / "gart2_best_params.json"
NOTES_OUT = THIS_DIR / "gart2_run_notes.md"
STUDY_DB = THIS_DIR / "gart2_optuna.db"

RANDOM_STATE = 42
N_TRIALS = int(os.environ.get("GART2_TRIALS", "200"))
EARLY_STOP_MIN, EARLY_STOP_MAX = 50, 250
MAX_BOOST_ROUND = 4000
MIN_BOOST_ROUND = 100
NUM_THREADS = 6
ALPHA_CLIP = (1.0, 2.0)

EXTRA_FEATURE = "greedy_nn_over_mst"
BANNED = ("grid_size", "mst_total_length")


# =============================================================================
def build_feature_list() -> list[str]:
    """V3's exact 30 columns, in the trained order, plus the one new feature.

    Read off the shipped V3 booster so the inherited block cannot drift.
    """
    m3 = joblib.load(THIS_DIR / "lgbm_alpha_model_v3.joblib")
    v3 = list(m3.feature_name_)
    for b in BANNED:
        if b in v3:
            raise RuntimeError(f"shipped V3 unexpectedly contains {b!r}")
    if len(v3) != 30:
        raise RuntimeError(f"expected 30 V3 features, got {len(v3)}")
    return v3 + [EXTRA_FEATURE]


def load_data(features: list[str]):
    df = pd.read_csv(DATA_FILE)
    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(*ALPHA_CLIP)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(f"feature table is missing {missing}")

    parts = {}
    for name in ("train", "val", "test"):
        m = df["split"] == name
        parts[name] = {
            "X": df.loc[m, features],
            "y": df.loc[m, "alpha"],
            "mst": df.loc[m, "mst_total_length"].to_numpy(),
            "cost": df.loc[m, "optimal_cost"].to_numpy(),
            "dim": df.loc[m, "dimension"].to_numpy(),
        }
    return df, parts


def cost_metrics(alpha_pred, mst, true_cost) -> dict:
    """Cost-level metrics. SDPE is the paper's primary consistency metric."""
    a = np.clip(alpha_pred, *ALPHA_CLIP)
    err = (a * mst - true_cost) / true_cost
    return {
        "sdpe": float(np.std(err, ddof=1) * 100.0),
        "mape": float(np.mean(np.abs(err)) * 100.0),
        "mspe": float(np.mean(err ** 2) * 1e4),   # mean squared pct error, pp^2
        "bias": float(np.mean(err) * 100.0),
    }


# =============================================================================
def make_objective(parts):
    tr, vl = parts["train"], parts["val"]

    def objective(trial: optuna.Trial) -> float:
        es = trial.suggest_int("early_stopping_rounds", EARLY_STOP_MIN, EARLY_STOP_MAX)
        params = {
            "objective": "regression_l2",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "seed": RANDOM_STATE,
            "num_threads": NUM_THREADS,
            "verbosity": -1,
            "feature_pre_filter": False,
            # Span both V3's high-capacity region (num_leaves 148, depth -1)
            # and V4's heavily-regularised region (num_leaves 49, depth 6),
            # since the ablation showed each wins on a different stratum.
            # Floor at 5e-3: below that nothing converges inside MAX_BOOST_ROUND,
            # so those trials burn the full budget to produce an under-trained
            # model. V3 (0.0260) and V4 (0.0227) both sit well inside this range.
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 8),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
        }
        dtr = lgb.Dataset(tr["X"], label=tr["y"])
        dvl = lgb.Dataset(vl["X"], label=vl["y"], reference=dtr)
        booster = lgb.train(
            params, dtr, num_boost_round=MAX_BOOST_ROUND,
            valid_sets=[dvl], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(es, verbose=False),
                LightGBMPruningCallback(trial, metric="rmse", valid_name="val"),
                lgb.log_evaluation(0),
            ],
        )
        pred = booster.predict(vl["X"], num_iteration=booster.best_iteration)
        m = cost_metrics(pred, vl["mst"], vl["cost"])
        trial.set_user_attr("val_mape", m["mape"])
        trial.set_user_attr("val_sdpe", m["sdpe"])
        trial.set_user_attr("val_bias", m["bias"])
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        return m["sdpe"]          # PRIMARY OBJECTIVE

    return objective


def refit(params_in: dict, parts, features):
    """Final fit under the V3 protocol: train only, early-stopped on val."""
    p = dict(params_in)
    es = int(p.pop("early_stopping_rounds", 100))
    p.update({
        "objective": "regression_l2", "metric": "rmse", "boosting_type": "gbdt",
        "seed": RANDOM_STATE, "num_threads": NUM_THREADS, "verbosity": -1,
        "feature_pre_filter": False,
    })
    tr, vl = parts["train"], parts["val"]
    dtr = lgb.Dataset(tr["X"], label=tr["y"])
    dvl = lgb.Dataset(vl["X"], label=vl["y"], reference=dtr)
    booster = lgb.train(
        p, dtr, num_boost_round=MAX_BOOST_ROUND,
        valid_sets=[dvl], valid_names=["val"],
        callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)],
    )
    return booster, p, es


# =============================================================================
def main() -> None:
    features = build_feature_list()
    print(f"[gart2] {len(features)} features; excluded {BANNED}")
    df, parts = load_data(features)
    print(f"[gart2] train={len(parts['train']['X'])} "
          f"val={len(parts['val']['X'])} test={len(parts['test']['X'])}")

    sampler = TPESampler(multivariate=True, group=True, seed=RANDOM_STATE)
    pruner = HyperbandPruner(min_resource=MIN_BOOST_ROUND, max_resource=MAX_BOOST_ROUND)
    study = optuna.create_study(
        study_name="gart2", direction="minimize",
        sampler=sampler, pruner=pruner,
        storage=f"sqlite:///{STUDY_DB.as_posix()}", load_if_exists=True,
    )
    t0 = time.time()
    study.optimize(make_objective(parts), n_trials=N_TRIALS, show_progress_bar=False)
    tune_min = (time.time() - t0) / 60.0

    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    best_sdpe = min(done, key=lambda t: t.value)
    best_mape = min(done, key=lambda t: t.user_attrs.get("val_mape", 1e9))
    print(f"\n[gart2] {len(done)} complete / {len(study.trials)} trials in {tune_min:.1f} min")
    print(f"[gart2] SDPE-min trial #{best_sdpe.number}: "
          f"SDPE={best_sdpe.user_attrs['val_sdpe']:.4f}%  MAPE={best_sdpe.user_attrs['val_mape']:.4f}%")
    print(f"[gart2] MAPE-min trial #{best_mape.number}: "
          f"SDPE={best_mape.user_attrs['val_sdpe']:.4f}%  MAPE={best_mape.user_attrs['val_mape']:.4f}%")

    # ---- final model = SDPE-minimising configuration -----------------------
    booster, final_params, es = refit(best_sdpe.params, parts, features)
    te, vl = parts["test"], parts["val"]
    pred_te = booster.predict(te["X"], num_iteration=booster.best_iteration)
    pred_vl = booster.predict(vl["X"], num_iteration=booster.best_iteration)
    m_test = cost_metrics(pred_te, te["mst"], te["cost"])
    m_val = cost_metrics(pred_vl, vl["mst"], vl["cost"])
    print("\n[gart2] TEST (cost level):", {k: round(v, 4) for k, v in m_test.items()})

    # ---- the MAPE-minimising alternative, fitted the same way (for the trade)
    alt, _, _ = refit(best_mape.params, parts, features)
    m_test_alt = cost_metrics(alt.predict(te["X"], num_iteration=alt.best_iteration),
                              te["mst"], te["cost"])
    print("[gart2] TEST if we had picked MAPE-min:",
          {k: round(v, 4) for k, v in m_test_alt.items()})

    joblib.dump(booster, MODEL_OUT)
    payload = {
        "model": "GART 2.0",
        "artifact": MODEL_OUT.name,
        "n_features": len(features),
        "features": features,
        "excluded_features": list(BANNED),
        "protocol": "fit on train only, early-stopped on val; no refit on train+val",
        "objective_primary": "validation SDPE (cost level)",
        "optuna_trials_requested": N_TRIALS,
        "optuna_trials_complete": len(done),
        "optuna_minutes": tune_min,
        "seed": RANDOM_STATE,
        "best_iteration": int(booster.best_iteration),
        "early_stopping_rounds": es,
        "hyperparameters": best_sdpe.params,
        "lgb_params_final": {k: v for k, v in final_params.items()},
        "sdpe_min_trial": {
            "number": best_sdpe.number, "params": best_sdpe.params,
            "val_sdpe": best_sdpe.user_attrs["val_sdpe"],
            "val_mape": best_sdpe.user_attrs["val_mape"],
        },
        "mape_min_trial": {
            "number": best_mape.number, "params": best_mape.params,
            "val_sdpe": best_mape.user_attrs["val_sdpe"],
            "val_mape": best_mape.user_attrs["val_mape"],
            "test_metrics_if_selected": m_test_alt,
        },
        "val_metrics_final": m_val,
        "test_metrics_final": m_test,
    }
    PARAMS_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    NOTES_OUT.write_text(
        "# GART 2.0 training run\n\n"
        f"- features: {len(features)} (V3's 30 + `{EXTRA_FEATURE}`)\n"
        f"- excluded: {', '.join(BANNED)}\n"
        f"- protocol: train-only fit, early stop on val (val never trained on)\n"
        f"- objective: minimise validation SDPE (cost level)\n"
        f"- Optuna: {len(done)} complete trials, {tune_min:.1f} min, seed {RANDOM_STATE}\n"
        f"- picked trial #{best_sdpe.number}; best_iteration {booster.best_iteration}\n"
        f"- val:  SDPE={m_val['sdpe']:.4f}%  MAPE={m_val['mape']:.4f}%\n"
        f"- test: SDPE={m_test['sdpe']:.4f}%  MAPE={m_test['mape']:.4f}%  "
        f"MSPE={m_test['mspe']:.4f}pp^2  bias={m_test['bias']:+.4f}%\n"
        f"- MAPE-min alternative (trial #{best_mape.number}) would give test "
        f"SDPE={m_test_alt['sdpe']:.4f}%  MAPE={m_test_alt['mape']:.4f}%\n",
        encoding="utf-8",
    )
    print(f"\n[gart2] saved {MODEL_OUT.name}, {PARAMS_OUT.name}, {NOTES_OUT.name}")


if __name__ == "__main__":
    main()
