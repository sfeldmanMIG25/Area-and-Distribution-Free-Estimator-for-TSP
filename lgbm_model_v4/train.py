"""Train GART V4 LightGBM (multi-objective Optuna, precision-first).

Pipeline:
  1. Load tsp_features_v4.csv and selected_features.json.
  2. Pass 1 — Optuna multi-objective study (minimize MAPE, minimize SDPE) on
     the full selected feature set. TPE sampler + Hyperband pruner.
     ``early_stopping_rounds`` is itself a tuned hyperparameter in
     [EARLY_STOP_MIN, EARLY_STOP_MAX].
  3. Pick the SDPE-minimum trial on the Pareto front that still has
     MAPE <= V3 baseline (MAPE_PARETO_CUTOFF). If none satisfies that,
     fall back to the lowest-SDPE trial overall.
  4. Zero-gain prune — refit the pass-1 winner and drop features whose
     total LightGBM gain is 0 (never split on).
  5. Pass 2 (conditional) — if any features were pruned, run a smaller
     Optuna study (REFINEMENT_TRIALS) on the pruned subset. Accept pass 2
     only if it both meets the MAPE budget and strictly improves SDPE.
  6. Refit on train + val with the winning hyperparameters + features and
     report test-set MAPE, SDPE, R^2, MSE, MAE.
  7. Save model, params, active Pareto plot, and run notes.

Key design choices:
  * Cost-level SDPE (not alpha-level) because MAPE and SDPE are quoted on
    tour cost throughout the paper.
  * Inputs are split deterministically via the ``split`` column — same
    70/20/10 assignment as V3 (d=100 locked to test).
  * The HyperbandPruner prunes bad trials at 100 boosting rounds, freeing
    compute for promising configurations.
  * Pass 2 re-tunes hyperparameters on the pruned feature set, closing the
    feature-selection / HP-tuning one-shot gap in the original V4 design.
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
OPTUNA_N_TRIALS = 100            # pass-1 trials on full feature set
REFINEMENT_TRIALS = 30           # pass-2 trials after zero-gain pruning
RANDOM_STATE = 42
EARLY_STOP_MIN = 50              # lower bound for tuned early-stopping rounds
EARLY_STOP_MAX = 200             # upper bound for tuned early-stopping rounds


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
        early_stopping_rounds = trial.suggest_int(
            "early_stopping_rounds", EARLY_STOP_MIN, EARLY_STOP_MAX
        )
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
            lgb.early_stopping(early_stopping_rounds, verbose=False),
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
# Helpers: Optuna study runner and zero-gain pruning
# =============================================================================
def _run_study(name: str, X_tr, y_tr, X_vl, y_vl, mst_vl, true_vl, n_trials: int):
    sampler = optuna.samplers.TPESampler(
        multivariate=True, group=True, seed=RANDOM_STATE,
        warn_independent_sampling=False,
    )
    pruner = optuna.pruners.HyperbandPruner(min_resource=100, max_resource=5000, reduction_factor=3)
    storage = f"sqlite:///{STUDY_DB.as_posix()}"
    study = optuna.create_study(
        study_name=name,
        directions=["minimize", "minimize"],  # MAPE, SDPE
        sampler=sampler, pruner=pruner,
        storage=storage, load_if_exists=True,
    )
    objective = _build_objective(X_tr, y_tr, X_vl, y_vl, mst_vl, true_vl)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t0
    best = _pick_best_trial(study)
    print(f"[{name}] {n_trials} trials in {elapsed/60:.1f} min — "
          f"picked #{best.number}: MAPE={best.values[0]:.3f}%  SDPE={best.values[1]:.3f}%")
    return study, best, elapsed


def _lgb_params_from_trial(trial_params: dict) -> tuple[dict, int]:
    """Split a trial's params into (LightGBM params dict, early_stopping_rounds)."""
    p = dict(trial_params)
    es = int(p.pop("early_stopping_rounds", EARLY_STOP_MIN))
    p.update({"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": RANDOM_STATE})
    return p, es


def _zero_gain_features(X_tr, y_tr, X_vl, y_vl, features, trial_params, best_iter) -> list[str]:
    """Refit once with the trial's hyperparameters, return features whose
    total LightGBM gain is 0 (never chosen as a split)."""
    params, es = _lgb_params_from_trial(trial_params)
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dvl = lgb.Dataset(X_vl, label=y_vl, reference=dtr)
    booster = lgb.train(
        params, dtr, num_boost_round=max(1000, int(best_iter * 1.1)),
        valid_sets=[dvl], valid_names=["valid_1"],
        callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)],
    )
    gains = booster.feature_importance(importance_type="gain")
    return [f for f, g in zip(features, gains) if g <= 0]


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    df, features = _load()
    print(f"[train] loaded {len(df)} rows, {len(features)} features")

    (X_tr, y_tr, X_vl, y_vl, X_te, y_te,
     mst_vl, true_vl, mst_te, true_te) = _split(df, features)

    print(f"[train] split — train={len(X_tr)}, val={len(X_vl)}, test={len(X_te)}")

    # ---- Pass 1: Optuna on the full selected feature set ----
    study1, best1, t1 = _run_study(
        "gart_v4", X_tr, y_tr, X_vl, y_vl, mst_vl, true_vl, OPTUNA_N_TRIALS
    )

    # ---- Zero-gain pruning: drop features the pass-1 best model never split on ----
    best1_iter = best1.user_attrs.get("best_iteration", 3000)
    dropped = _zero_gain_features(X_tr, y_tr, X_vl, y_vl, features,
                                  best1.params, best1_iter)
    print(f"[train] zero-gain features from pass-1 best model: {dropped or '(none)'}")

    best = best1
    active_study = study1
    total_time = t1
    selected_features = list(features)
    refinement = None
    if dropped:
        selected_features = [f for f in features if f not in dropped]
        X_tr2 = X_tr[selected_features]; X_vl2 = X_vl[selected_features]
        study2, best2, t2 = _run_study(
            "gart_v4_refined", X_tr2, y_tr, X_vl2, y_vl, mst_vl, true_vl,
            REFINEMENT_TRIALS
        )
        total_time += t2
        refinement = {
            "trials": REFINEMENT_TRIALS,
            "minutes": t2 / 60.0,
            "dropped": dropped,
            "kept_features": selected_features,
            "val_mape": best2.values[0],
            "val_sdpe": best2.values[1],
        }
        # Accept pass 2 only if it meets the MAPE cutoff AND strictly improves SDPE.
        meets_budget = best2.values[0] <= MAPE_PARETO_CUTOFF
        improves_sdpe = best2.values[1] < best1.values[1]
        if meets_budget and improves_sdpe:
            print(f"[train] pass-2 wins: SDPE {best1.values[1]:.3f}% -> {best2.values[1]:.3f}%")
            best = best2
            active_study = study2
            refinement["accepted"] = True
        else:
            print(f"[train] pass-2 rejected (meets_budget={meets_budget}, improves_sdpe={improves_sdpe}); keeping pass-1")
            selected_features = list(features)
            refinement["accepted"] = False

    best_iter = best.user_attrs.get("best_iteration", 3000)
    print(f"[train] final pick: MAPE={best.values[0]:.3f}%  SDPE={best.values[1]:.3f}%  best_iter={best_iter}")
    print(f"[train] final feature count: {len(selected_features)}")

    # ---- Final fit on train + val ----
    X_tr_f = X_tr[selected_features]; X_vl_f = X_vl[selected_features]
    X_te_f = X_te[selected_features]
    X_full = pd.concat([X_tr_f, X_vl_f]); y_full = pd.concat([y_tr, y_vl])
    final_params, _ = _lgb_params_from_trial(best.params)
    # Train with 1.1x the val-derived iteration count to let it breathe.
    num_rounds = int(max(100, round(best_iter * 1.1)))
    dtr = lgb.Dataset(X_full, label=y_full)
    booster = lgb.train(final_params, dtr, num_boost_round=num_rounds)

    # ---- Test-set metrics ----
    m_test = _cost_level_metrics(booster, X_te_f, y_te, mst_te, true_te)
    m_val = _cost_level_metrics(booster, X_vl_f, y_vl, mst_vl, true_vl)
    print("[train] TEST metrics (cost-level):")
    for k, v in m_test.items():
        print(f"    {k:12s} = {v:.4f}")

    # ---- Save artifacts ----
    joblib.dump(booster, MODEL_OUTPUT)
    with open(PARAMS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({
            "hyperparameters": best.params,
            "num_boost_round": num_rounds,
            "best_trial_number": best.number,
            "best_trial_study": active_study.study_name,
            "val_metrics_best_trial": {"mape": best.values[0], "sdpe": best.values[1]},
            "test_metrics_final": m_test,
            "val_metrics_final": m_val,
            "pareto_size": len(active_study.best_trials),
            "optuna_trials_pass1": OPTUNA_N_TRIALS,
            "refinement": refinement,
            "feature_set_full": features,
            "feature_set_final": selected_features,
        }, f, indent=2)

    # ---- Pareto plot (active study: whichever pass produced the winner) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for t in active_study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.values:
            ax.scatter(t.values[0], t.values[1], alpha=0.35, c="#666")
    for t in active_study.best_trials:
        ax.scatter(t.values[0], t.values[1], c="red", s=40, label="Pareto front" if t == active_study.best_trials[0] else None)
    ax.scatter(best.values[0], best.values[1], c="blue", marker="*", s=180, label="Picked")
    ax.axvline(MAPE_PARETO_CUTOFF, linestyle="--", color="black", alpha=0.4, label=f"MAPE budget ({MAPE_PARETO_CUTOFF}%)")
    ax.set_xlabel("MAPE (%) — val"); ax.set_ylabel("SDPE (%) — val")
    ax.set_title(f"V4 Optuna Pareto front — {active_study.study_name}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PARETO_PLOT, dpi=160); plt.close()

    # ---- Notes ----
    RUN_NOTES.write_text(
        "# V4 training run notes\n\n"
        f"- feature_set_full: {len(features)}\n"
        f"- feature_set_final: {len(selected_features)}"
        + (f"  (dropped zero-gain: {dropped})\n" if dropped else "\n")
        + f"- optuna_trials_pass1: {OPTUNA_N_TRIALS}  (time: {t1/60:.1f} min)\n"
        + (f"- optuna_trials_pass2: {REFINEMENT_TRIALS}  (accepted: {refinement['accepted']})\n" if refinement else "")
        + f"- total optuna minutes: {total_time/60:.1f}\n"
        + f"- active_study: {active_study.study_name}\n"
        + f"- pareto_front_size: {len(active_study.best_trials)}\n"
        + f"- picked_trial: #{best.number}\n"
        + f"- early_stopping_rounds (tuned): {best.params.get('early_stopping_rounds')}\n"
        + f"- val metrics (picked): MAPE={best.values[0]:.3f}%, SDPE={best.values[1]:.3f}%\n"
        + f"- test metrics (final fit): MAPE={m_test['mape']:.3f}%, SDPE={m_test['sdpe']:.3f}%, "
        + f"R2={m_test['r2_cost']:.4f}\n",
        encoding="utf-8",
    )
    print(f"[train] saved: {MODEL_OUTPUT}, {PARAMS_OUTPUT}, {PARETO_PLOT}, {RUN_NOTES}")


if __name__ == "__main__":
    main()
