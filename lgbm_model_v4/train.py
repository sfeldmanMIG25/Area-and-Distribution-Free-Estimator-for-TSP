"""Train GART V4 LightGBM (multi-objective Optuna, precision-first).

Pipeline:
  1. Load tsp_features_v4.csv and selected_features.json.
  2. Optuna multi-objective study (minimize MAPE, minimize SDPE) using a
     multivariate TPE sampler and Hyperband pruner. Each trial trains
     LightGBM with early stopping on val and reports (MAPE, SDPE) on val.
  3. Pick the SDPE-minimum trial on the Pareto front that still has
     MAPE <= V3 baseline (5.5% cost-level). If no trial satisfies that,
     fall back to the lowest-SDPE trial overall (still reported).
  4. Refit on train + val with the picked hyperparameters (early-stop-derived
     num_boost_round) and report test-set MAPE, SDPE, R^2, MSE, MAE.
  5. Save model, params, and a Pareto-front plot.

Key design choices:
  * Cost-level SDPE (not alpha-level) because MAPE and SDPE are quoted on
    tour cost throughout the paper.
  * Inputs are split deterministically via the ``split`` column — same
    70/20/10 assignment as V3 (d=100 locked to test).
  * The HyperbandPruner prunes bad trials at 100 boosting rounds, freeing
    compute for promising configurations.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from feature_engineering import feature_columns_for_training  # noqa: F401 (sanity import)

V4_CSV = REPO_ROOT / "tsp_features_v4.csv"
SELECTED_JSON = THIS_DIR / "selected_features.json"
MODEL_OUTPUT = THIS_DIR / "lgbm_alpha_model_v4.joblib"
PARAMS_OUTPUT = THIS_DIR / "best_params_v4.json"
PARETO_PLOT = THIS_DIR / "pareto_front.png"
STUDY_DB = THIS_DIR / "optuna_study.db"
RUN_NOTES = THIS_DIR / "run_notes.md"

# Targets
ALPHA_CLIP = (1.0, 2.0)          # bounded ratio
MAPE_PARETO_CUTOFF = 5.5         # percent; chosen to match V3 2D MAPE
OPTUNA_N_TRIALS = 100            # per user spec
RANDOM_STATE = 42
EARLY_STOP_VAL = 50
EARLY_STOP_FINAL = 100


# =============================================================================
# Data
# =============================================================================
def _load() -> Tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(V4_CSV)
    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(*ALPHA_CLIP)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)
    with open(SELECTED_JSON, "r", encoding="utf-8") as f:
        selected = json.load(f)["selected"]
    return df, selected


def _split(df: pd.DataFrame, features: list[str]):
    mtr = df["split"] == "train"
    mvl = df["split"] == "val"
    mte = df["split"] == "test"
    X_tr = df.loc[mtr, features]; y_tr = df.loc[mtr, "alpha"]
    X_vl = df.loc[mvl, features]; y_vl = df.loc[mvl, "alpha"]
    X_te = df.loc[mte, features]; y_te = df.loc[mte, "alpha"]
    mst_vl = df.loc[mvl, "mst_total_length"].to_numpy()
    mst_te = df.loc[mte, "mst_total_length"].to_numpy()
    true_vl = df.loc[mvl, "optimal_cost"].to_numpy()
    true_te = df.loc[mte, "optimal_cost"].to_numpy()
    return (X_tr, y_tr, X_vl, y_vl, X_te, y_te, mst_vl, true_vl, mst_te, true_te)


# =============================================================================
# Metrics
# =============================================================================
def _cost_level_metrics(booster, X, y_alpha, mst, true_cost, num_iteration=None):
    alpha_pred = np.clip(booster.predict(X, num_iteration=num_iteration), *ALPHA_CLIP)
    pred_cost = alpha_pred * mst
    err = (pred_cost - true_cost) / true_cost
    sdpe = float(np.std(err, ddof=1) * 100.0)
    mape = float(np.mean(np.abs(err)) * 100.0)
    bias = float(np.mean(err) * 100.0)
    rmse = float(np.sqrt(mean_squared_error(y_alpha, alpha_pred)))
    r2 = float(r2_score(true_cost, pred_cost))
    mae = float(mean_absolute_error(true_cost, pred_cost))
    return {"sdpe": sdpe, "mape": mape, "bias": bias, "rmse_alpha": rmse, "r2_cost": r2, "mae_cost": mae}


# =============================================================================
# Optuna objective
# =============================================================================
def _build_objective(X_tr, y_tr, X_vl, y_vl, mst_vl, true_vl):
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": RANDOM_STATE,
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "max_depth": trial.suggest_int("max_depth", -1, 32),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-9, 1.0, log=True),
        }
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dvl = lgb.Dataset(X_vl, label=y_vl, reference=dtr)
        callbacks = [
            lgb.early_stopping(EARLY_STOP_VAL, verbose=False),
            lgb.log_evaluation(0),
            optuna.integration.LightGBMPruningCallback(trial, "rmse", valid_name="valid_1"),
        ]
        booster = lgb.train(
            params, dtr, num_boost_round=5000,
            valid_sets=[dtr, dvl], valid_names=["train", "valid_1"],
            callbacks=callbacks,
        )
        m = _cost_level_metrics(booster, X_vl, y_vl, mst_vl, true_vl, num_iteration=booster.best_iteration)
        trial.set_user_attr("best_iteration", booster.best_iteration)
        trial.set_user_attr("bias_val", m["bias"])
        return m["mape"], m["sdpe"]

    return objective


# =============================================================================
# Pick best trial
# =============================================================================
def _pick_best_trial(study: optuna.study.Study) -> optuna.trial.FrozenTrial:
    pareto = study.best_trials
    if not pareto:
        raise RuntimeError("No Pareto-optimal trial returned — study failed.")
    # Primary rule: lowest SDPE among trials with MAPE <= MAPE_PARETO_CUTOFF.
    eligible = [t for t in pareto if t.values and t.values[0] <= MAPE_PARETO_CUTOFF]
    if not eligible:
        # All Pareto trials exceed the MAPE budget. Fall back to lowest SDPE anyway
        # and log the compromise for the run notes.
        eligible = pareto
    return min(eligible, key=lambda t: t.values[1])


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    df, features = _load()
    print(f"[train] loaded {len(df)} rows, {len(features)} features")

    (X_tr, y_tr, X_vl, y_vl, X_te, y_te,
     mst_vl, true_vl, mst_te, true_te) = _split(df, features)

    print(f"[train] split — train={len(X_tr)}, val={len(X_vl)}, test={len(X_te)}")

    # ---- Optuna multi-objective study ----
    sampler = optuna.samplers.TPESampler(
        multivariate=True, group=True, seed=RANDOM_STATE,
        warn_independent_sampling=False,
    )
    pruner = optuna.pruners.HyperbandPruner(min_resource=100, max_resource=5000, reduction_factor=3)
    storage = f"sqlite:///{STUDY_DB.as_posix()}"
    study = optuna.create_study(
        study_name="gart_v4",
        directions=["minimize", "minimize"],  # MAPE, SDPE
        sampler=sampler, pruner=pruner,
        storage=storage, load_if_exists=True,
    )
    objective = _build_objective(X_tr, y_tr, X_vl, y_vl, mst_vl, true_vl)
    t0 = time.time()
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
    opt_time = time.time() - t0
    print(f"[train] Optuna finished in {opt_time/60:.1f} min across {OPTUNA_N_TRIALS} trials")

    best = _pick_best_trial(study)
    best_params = dict(best.params)
    best_iter = best.user_attrs.get("best_iteration", 3000)
    print(f"[train] picked trial #{best.number}: MAPE={best.values[0]:.3f}%  SDPE={best.values[1]:.3f}%")
    print(f"[train] best_iter={best_iter}  params={best_params}")

    # ---- Final fit on train + val ----
    X_full = pd.concat([X_tr, X_vl]); y_full = pd.concat([y_tr, y_vl])
    final_params = dict(best_params)
    final_params.update({"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": RANDOM_STATE})
    # Train with 1.1x the val-derived iteration count to let it breathe.
    num_rounds = int(max(100, round(best_iter * 1.1)))
    dtr = lgb.Dataset(X_full, label=y_full)
    booster = lgb.train(final_params, dtr, num_boost_round=num_rounds)

    # ---- Test-set metrics ----
    m_test = _cost_level_metrics(booster, X_te, y_te, mst_te, true_te)
    m_val = _cost_level_metrics(booster, X_vl, y_vl, mst_vl, true_vl)
    print("[train] TEST metrics (cost-level):")
    for k, v in m_test.items():
        print(f"    {k:12s} = {v:.4f}")

    # ---- Save artifacts ----
    joblib.dump(booster, MODEL_OUTPUT)
    with open(PARAMS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "hyperparameters": best_params,
            "num_boost_round": num_rounds,
            "best_trial_number": best.number,
            "val_metrics_best_trial": {"mape": best.values[0], "sdpe": best.values[1]},
            "test_metrics_final": m_test,
            "val_metrics_final": m_val,
            "pareto_size": len(study.best_trials),
            "optuna_trials": OPTUNA_N_TRIALS,
            "feature_set": features,
        }, f, indent=2)

    # ---- Pareto plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.values:
            ax.scatter(t.values[0], t.values[1], alpha=0.35, c="#666")
    for t in study.best_trials:
        ax.scatter(t.values[0], t.values[1], c="red", s=40, label="Pareto front" if t == study.best_trials[0] else None)
    ax.scatter(best.values[0], best.values[1], c="blue", marker="*", s=180, label="Picked")
    ax.axvline(MAPE_PARETO_CUTOFF, linestyle="--", color="black", alpha=0.4, label=f"MAPE budget ({MAPE_PARETO_CUTOFF}%)")
    ax.set_xlabel("MAPE (%) — val"); ax.set_ylabel("SDPE (%) — val")
    ax.set_title("V4 Optuna Pareto front (precision-first)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PARETO_PLOT, dpi=160); plt.close()

    # ---- Notes ----
    RUN_NOTES.write_text(
        "# V4 training run notes\n\n"
        f"- feature_set_size: {len(features)}\n"
        f"- optuna_trials: {OPTUNA_N_TRIALS}  (time: {opt_time/60:.1f} min)\n"
        f"- pareto_front_size: {len(study.best_trials)}\n"
        f"- picked_trial: #{best.number}\n"
        f"- val metrics (picked): MAPE={best.values[0]:.3f}%, SDPE={best.values[1]:.3f}%\n"
        f"- test metrics (final fit): MAPE={m_test['mape']:.3f}%, SDPE={m_test['sdpe']:.3f}%, "
        f"R2={m_test['r2_cost']:.4f}\n",
        encoding="utf-8",
    )
    print(f"[train] saved: {MODEL_OUTPUT}, {PARAMS_OUTPUT}, {PARETO_PLOT}, {RUN_NOTES}")


if __name__ == "__main__":
    main()
