"""GART 2.0 with monotone constraints on n_customers and dimension.

Rationale: the paper's consistency thesis says accuracy does not degrade as
instances get larger or higher-dimensional. Unconstrained, that is an
in-distribution empirical observation. A LightGBM monotone constraint turns it
into a construction guarantee that holds at *any* n and d, including well past
the training range where the paper makes its extrapolation claims.

Direction is -1 (non-increasing) on both: alpha = OPT / MST falls towards 1 as
the instance grows or the ambient dimension rises.

Two variants:
  both -- constrain n_customers and dimension
  n    -- constrain n_customers only

For each variant the constraints are active *during* Optuna, not bolted onto
hyperparameters tuned without them: a constrained optimum generally sits
somewhere else in hyperparameter space. The bolt-on model is also fitted, for
comparison.

Usage:
    python train_gart2_monotone.py --constrain both --trials 80
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from train_gart2 import (  # noqa: E402
    EARLY_STOP_MAX, EARLY_STOP_MIN, MAX_BOOST_ROUND, MIN_BOOST_ROUND,
    NUM_THREADS, RANDOM_STATE, build_feature_list, cost_metrics, load_data,
)

THIS_DIR = Path(__file__).resolve().parent
CONSTRAINED_COLS = {"both": ("n_customers", "dimension"), "n": ("n_customers",)}


def constraint_vector(features: list[str], which: str) -> list[int]:
    cols = CONSTRAINED_COLS[which]
    return [-1 if f in cols else 0 for f in features]


def _base_params(mono: list[int]) -> dict:
    return {
        "objective": "regression_l2", "metric": "rmse", "boosting_type": "gbdt",
        "seed": RANDOM_STATE, "num_threads": NUM_THREADS, "verbosity": -1,
        "feature_pre_filter": False,
        "monotone_constraints": mono,
        "monotone_constraints_method": "basic",
    }


def fit(params_in: dict, parts, features, mono):
    p = dict(params_in)
    es = int(p.pop("early_stopping_rounds", 100))
    p.update(_base_params(mono))
    tr, vl = parts["train"], parts["val"]
    dtr = lgb.Dataset(tr["X"], label=tr["y"])
    dvl = lgb.Dataset(vl["X"], label=vl["y"], reference=dtr)
    return lgb.train(p, dtr, num_boost_round=MAX_BOOST_ROUND,
                     valid_sets=[dvl], valid_names=["val"],
                     callbacks=[lgb.early_stopping(es, verbose=False),
                                lgb.log_evaluation(0)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constrain", choices=["both", "n"], required=True)
    ap.add_argument("--trials", type=int, default=80)
    args = ap.parse_args()

    features = build_feature_list()
    mono = constraint_vector(features, args.constrain)
    df, parts = load_data(features)
    tag = f"gart2_mono_{args.constrain}"
    print(f"[{tag}] constrained: {[f for f, m in zip(features, mono) if m]}")

    # ---- bolt-on: the unconstrained tuned hyperparameters + constraints ----
    unc = json.loads((THIS_DIR / "gart2_best_params.json").read_text(encoding="utf-8"))
    b_bolt = fit(unc["hyperparameters"], parts, features, mono)
    te, vl = parts["test"], parts["val"]
    m_bolt = cost_metrics(b_bolt.predict(te["X"], num_iteration=b_bolt.best_iteration),
                          te["mst"], te["cost"])
    print(f"[{tag}] bolt-on  test SDPE={m_bolt['sdpe']:.4f} MAPE={m_bolt['mape']:.4f}")

    # ---- re-tune with the constraints active -------------------------------
    def objective(trial: optuna.Trial) -> float:
        es = trial.suggest_int("early_stopping_rounds", EARLY_STOP_MIN, EARLY_STOP_MAX)
        p = _base_params(mono)
        p.update({
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
        })
        tr, v = parts["train"], parts["val"]
        dtr = lgb.Dataset(tr["X"], label=tr["y"])
        dvl = lgb.Dataset(v["X"], label=v["y"], reference=dtr)
        bo = lgb.train(p, dtr, num_boost_round=MAX_BOOST_ROUND,
                       valid_sets=[dvl], valid_names=["val"],
                       callbacks=[lgb.early_stopping(es, verbose=False),
                                  LightGBMPruningCallback(trial, "rmse", "val"),
                                  lgb.log_evaluation(0)])
        m = cost_metrics(bo.predict(v["X"], num_iteration=bo.best_iteration),
                         v["mst"], v["cost"])
        trial.set_user_attr("val_mape", m["mape"])
        trial.set_user_attr("val_sdpe", m["sdpe"])
        return m["sdpe"]

    study = optuna.create_study(
        study_name=tag, direction="minimize",
        sampler=TPESampler(multivariate=True, group=True, seed=RANDOM_STATE),
        pruner=HyperbandPruner(min_resource=MIN_BOOST_ROUND, max_resource=MAX_BOOST_ROUND),
        storage=f"sqlite:///{(THIS_DIR / (tag + '.db')).as_posix()}", load_if_exists=True)
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    mins = (time.time() - t0) / 60.0

    C = optuna.trial.TrialState.COMPLETE
    done = [t for t in study.trials if t.state == C and t.value is not None]
    best = min(done, key=lambda t: t.value)
    b_tuned = fit(best.params, parts, features, mono)
    m_tuned = cost_metrics(b_tuned.predict(te["X"], num_iteration=b_tuned.best_iteration),
                           te["mst"], te["cost"])
    m_val = cost_metrics(b_tuned.predict(vl["X"], num_iteration=b_tuned.best_iteration),
                         vl["mst"], vl["cost"])
    print(f"[{tag}] re-tuned test SDPE={m_tuned['sdpe']:.4f} MAPE={m_tuned['mape']:.4f} "
          f"({len(done)} trials, {mins:.1f} min)")

    # Ship whichever of bolt-on / re-tuned has the lower TEST-adjacent val SDPE.
    # Selection uses val, never test.
    m_bolt_val = cost_metrics(b_bolt.predict(vl["X"], num_iteration=b_bolt.best_iteration),
                              vl["mst"], vl["cost"])
    use_tuned = m_val["sdpe"] <= m_bolt_val["sdpe"]
    winner, wname = (b_tuned, "retuned") if use_tuned else (b_bolt, "bolt_on")
    joblib.dump(winner, THIS_DIR / f"{tag}_model.joblib")
    (THIS_DIR / f"{tag}_params.json").write_text(json.dumps({
        "variant": args.constrain,
        "constrained_features": list(CONSTRAINED_COLS[args.constrain]),
        "monotone_constraints": mono,
        "monotone_constraints_method": "basic",
        "features": features,
        "selected": wname,
        "val_sdpe_retuned": m_val["sdpe"], "val_sdpe_bolt_on": m_bolt_val["sdpe"],
        "test_retuned": m_tuned, "test_bolt_on": m_bolt,
        "hyperparameters_retuned": best.params,
        "hyperparameters_bolt_on": unc["hyperparameters"],
        "optuna_trials": len(done), "optuna_minutes": mins,
    }, indent=2), encoding="utf-8")
    print(f"[{tag}] shipped {wname} -> {tag}_model.joblib")


if __name__ == "__main__":
    main()
