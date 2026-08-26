"""Freeze the chosen GART 2.0 artifact and write its provenance sidecar.

Chosen model: 31 features, logit target, cost-MAPE early stopping, frozen V3
hyperparameters, monotone constraints on n_customers and dimension.

Does NOT touch lgbm_alpha_model_v3.joblib -- the swap is a deliberate,
separate act.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
SRC = THIS / "gart2_logit_v3hp_mono_model.joblib"
DST = THIS / "gart2_final.joblib"
SIDECAR = THIS / "gart2_final.json"
SRC_PARAMS = THIS / "gart2_logit_v3hp_mono_params.json"


def main() -> None:
    src_meta = json.loads(SRC_PARAMS.read_text(encoding="utf-8"))
    booster = joblib.load(SRC)
    feats = list(booster.feature_name())

    df = pd.read_csv(REPO / "tsp_features_v4.csv",
                     usecols=["split", "optimal_cost", "mst_total_length"])
    mst = df["mst_total_length"].replace(0, np.nan)
    alpha = (df["optimal_cost"] / mst).clip(1.0, 2.0)
    keep = alpha.notna()
    n_train = int(((df["split"] == "train") & keep).sum())
    n_val = int(((df["split"] == "val") & keep).sum())
    n_test = int(((df["split"] == "test") & keep).sum())

    shutil.copyfile(SRC, DST)
    digest = hashlib.sha256(DST.read_bytes()).hexdigest()

    mono = src_meta["monotone_constraints"]
    payload = {
        "name": "GART 2.0",
        "artifact": DST.name,
        "sha256": digest,
        "derived_from": SRC.name,
        "seed": 42,

        "features_in_booster_order": feats,
        "n_features": len(feats),
        "excluded_features": {
            "grid_size": "generator fingerprint; leaks instance family",
            "mst_total_length":
                "alpha = optimal_cost / mst_total_length, so feeding MST "
                "magnitude lets the model recover a component of the label. "
                "Ablation: worth <=0.02 pp on every stratum, harmful on the "
                "augmentation corpus.",
        },

        "target_transform": {
            "forward": "z = log(u / (1 - u)) with u = clip(alpha - 1, 1e-6, 1 - 1e-6)",
            "inverse": "alpha = 1 + 1 / (1 + exp(-z))",
            "why": "alpha is bounded in [1, 2] by construction; the logit "
                   "target makes the bound structural so no post-hoc clip is "
                   "needed and the inverse cannot leave the interval.",
            "clip_after_inverse": False,
        },

        "monotone_constraints": mono,
        "monotone_constrained_features": [f for f, m in zip(feats, mono) if m == -1],
        "monotone_constraints_method": "basic",
        "monotone_direction": "-1 (non-increasing): alpha falls towards 1 as n "
                              "or d grows",

        "hyperparameters": src_meta["hyperparameters"],
        "hyperparameter_provenance":
            "V3's shipped values, frozen. Not tuned: Optuna against "
            "in-distribution validation reliably buys ND and pays for it on "
            "TSPLIB (reproduced in this study: 200 trials bought 0.011 pp ND "
            "and cost 0.16 pp TSPLIB MAPE).",
        "early_stopping_metric": "cost_mape (cost-level MAPE on the val split)",
        "early_stopping_rounds": src_meta["early_stopping_rounds"],
        "best_iteration": int(booster.best_iteration or booster.num_trees()),
        "num_trees": int(booster.num_trees()),

        "training_protocol":
            "Fit on split=='train' only, early-stopped on split=='val'. "
            "val is a checkpoint signal and is never trained on. No refit on "
            "train+val. split=='test' (d=100 held out) touched once.",
        "rows": {"train": n_train, "val": n_val, "test": n_test,
                 "total": n_train + n_val + n_test},
        "training_table": "tsp_features_v4.csv",

        "val_metrics": src_meta["val_metrics"],
        "test_metrics": src_meta["test_metrics"],
        "inference_module": "lgbm_model_v3/feature_engineering_gart2.py",
        "estimator_class": "lgbm_model_v3/lgbm_estimator_gart2.py::TSP_GART2_Estimator",
    }
    SIDECAR.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"froze {DST.name}  sha256={digest[:16]}...")
    print(f"  features={len(feats)}  trees={payload['num_trees']}  "
          f"best_iter={payload['best_iteration']}")
    print(f"  constrained={payload['monotone_constrained_features']}")
    print(f"  rows train/val/test = {n_train}/{n_val}/{n_test}")
    print(f"sidecar -> {SIDECAR.name}")


if __name__ == "__main__":
    main()
