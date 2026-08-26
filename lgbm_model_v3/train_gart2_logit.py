"""GART 2.0 variants with a logit target and MAPE-based early stopping.

alpha lives in [1, 2] by construction. Regressing alpha directly and clipping
afterwards treats that bound as an afterthought; regressing
``z = logit(alpha - 1)`` makes it structural -- the inverse map
``alpha = 1 + sigmoid(z)`` cannot leave the interval, so no clip is needed and
the model spends no capacity learning where the boundary is.

Early stopping on cost-level MAPE rather than RMSE-in-z aligns the stopping
rule with the metric that is actually reported. (Note that cost-level relative
error equals alpha relative error: pred_cost/true_cost = alpha_pred/alpha_true.)

Arms (all on the 31-feature GART 2.0 set):
  a  logit + esMAPE, frozen V3 hyperparameters, no tuning
  b  same as (a) plus monotone constraints on n_customers and dimension
  c  the tuned GART 2.0 hyperparameters with the logit target substituted

Usage:  python train_gart2_logit.py --arm a
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import joblib
import lightgbm as lgb
import numpy as np

from train_gart2 import (  # noqa: E402
    ALPHA_CLIP, MAX_BOOST_ROUND, NUM_THREADS, RANDOM_STATE,
    build_feature_list, cost_metrics, load_data,
)
from train_gart2_monotone import constraint_vector  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
EPS = 1e-6

# V3's shipped hyperparameters, frozen. Deliberately NOT tuned: the
# objective-alignment study showed that Optuna against in-distribution
# validation reliably buys ND and pays for it on TSPLIB.
V3_FROZEN = {
    "learning_rate": 0.02597081024148143, "num_leaves": 148,
    "reg_alpha": 0.07764927124026656, "reg_lambda": 1.1865278867536643e-05,
    "feature_fraction": 0.5481439957854186, "bagging_fraction": 0.6087321895536237,
    "bagging_freq": 6, "min_child_samples": 23, "max_depth": -1,
}
V3_EARLY_STOP = 100


def to_z(alpha: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(alpha, dtype=np.float64) - 1.0, EPS, 1.0 - EPS)
    return np.log(u / (1.0 - u))


def to_alpha(z: np.ndarray) -> np.ndarray:
    return 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def make_mape_feval(mst: np.ndarray, cost: np.ndarray):
    """Cost-level MAPE as a LightGBM eval function, in z-space."""
    def feval(preds, _ds):
        a = to_alpha(preds)
        err = (a * mst - cost) / cost
        return "cost_mape", float(np.mean(np.abs(err)) * 100.0), False
    return feval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b", "c"], required=True)
    args = ap.parse_args()

    features = build_feature_list()
    df, parts = load_data(features)
    tr, vl, te = parts["train"], parts["val"], parts["test"]

    if args.arm == "c":
        tuned = json.loads((THIS_DIR / "gart2_best_params.json").read_text(encoding="utf-8"))
        hp = dict(tuned["hyperparameters"])
        es = int(hp.pop("early_stopping_rounds", 100))
        mono = None
        tag = "gart2_logit_tuned"
    else:
        hp = dict(V3_FROZEN)
        es = V3_EARLY_STOP
        mono = constraint_vector(features, "both") if args.arm == "b" else None
        tag = "gart2_logit_v3hp" + ("_mono" if args.arm == "b" else "")

    params = dict(hp)
    params.update({
        "objective": "regression_l2",
        "metric": "None",            # early stopping is driven by the feval
        "boosting_type": "gbdt", "seed": RANDOM_STATE,
        "num_threads": NUM_THREADS, "verbosity": -1, "feature_pre_filter": False,
    })
    if mono is not None:
        params["monotone_constraints"] = mono
        params["monotone_constraints_method"] = "basic"

    z_tr, z_vl = to_z(tr["y"].to_numpy()), to_z(vl["y"].to_numpy())
    dtr = lgb.Dataset(tr["X"], label=z_tr)
    dvl = lgb.Dataset(vl["X"], label=z_vl, reference=dtr)
    booster = lgb.train(
        params, dtr, num_boost_round=MAX_BOOST_ROUND,
        valid_sets=[dvl], valid_names=["val"],
        feval=make_mape_feval(vl["mst"], vl["cost"]),
        callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)],
    )

    def score(part):
        a = to_alpha(booster.predict(part["X"], num_iteration=booster.best_iteration))
        return cost_metrics(a, part["mst"], part["cost"])

    m_val, m_test = score(vl), score(te)
    print(f"[{tag}] best_iter={booster.best_iteration}")
    print(f"[{tag}] val  {{'sdpe': {m_val['sdpe']:.4f}, 'mape': {m_val['mape']:.4f}}}")
    print(f"[{tag}] test {{'sdpe': {m_test['sdpe']:.4f}, 'mape': {m_test['mape']:.4f}, "
          f"'mspe': {m_test['mspe']:.4f}, 'bias': {m_test['bias']:+.4f}}}")

    joblib.dump(booster, THIS_DIR / f"{tag}_model.joblib")
    (THIS_DIR / f"{tag}_params.json").write_text(json.dumps({
        "arm": args.arm, "tag": tag, "target": "logit",
        "inverse": "alpha = 1 + sigmoid(z)",
        "early_stopping_metric": "cost_mape",
        "early_stopping_rounds": es,
        "monotone_constraints": mono,
        "monotone_constraints_method": "basic" if mono else None,
        "features": features, "hyperparameters": hp,
        "best_iteration": int(booster.best_iteration),
        "val_metrics": m_val, "test_metrics": m_test,
    }, indent=2), encoding="utf-8")
    print(f"[{tag}] saved {tag}_model.joblib")


if __name__ == "__main__":
    main()
