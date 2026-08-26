"""Freeze the alpha-coverage-trained model as a named artifact.

Fits the shipped recipe -- 31 features, logit(alpha - 1), V3 hyperparameters,
monotone -1 on n_customers and dimension, early stopping on cost-level MAPE --
on the corpus PLUS the alpha-coverage corpus, at the project's canonical seed
42, and writes it beside a manifest in the same shape as ``gart2_final.json``.

Seed 42 is the canonical seed, not a selected one: the manifest carries the
full seven-seed band from ``coverage_study_results.json`` so the artifact
cannot be mistaken for a favourable draw.

``lgbm_model_v3/gart2_final.*`` is NOT touched.  Promoting this artifact to the
shipped model is a separate, explicit act.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
for _p in (ROOT, HERE, ROOT / "lgbm_model_v3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from armA_verify_common import V3_FROZEN, fit  # noqa: E402
from coverage_study import arm_frames, load_all  # noqa: E402

OUT_MODEL = ROOT / "lgbm_model_v3" / "gart2_alphacov.joblib"
OUT_JSON = ROOT / "lgbm_model_v3" / "gart2_alphacov.json"
SEED = 42


def main() -> None:
    frozen = joblib.load(ROOT / "lgbm_model_v3" / "gart2_final.joblib")
    feats31 = list(frozen.feature_name())

    corpus, cov, C, held = load_all(feats31)
    tr, vl = arm_frames(corpus, cov, "cov")
    print(f"[freeze] train {len(tr)} (corpus {int((corpus.split=='train').sum())} "
          f"+ coverage {int((cov.split=='train').sum())})  val {len(vl)}")

    booster = fit(tr, vl, feats31, seed=SEED)
    joblib.dump(booster, OUT_MODEL)
    sha = hashlib.sha256(OUT_MODEL.read_bytes()).hexdigest()

    study = json.loads((HERE / "coverage_study_results.json").read_text(encoding="utf-8"))
    per_seed = pd.read_csv(HERE / "coverage_study_per_seed.csv")
    this_seed = per_seed[(per_seed.arm == "cov") & (per_seed.seed == SEED)]

    OUT_JSON.write_text(json.dumps({
        "name": "GART 2.0 + alpha-coverage corpus",
        "artifact": OUT_MODEL.name,
        "sha256": sha,
        "supersedes": "gart2_final.joblib (NOT overwritten; promotion is a separate act)",
        "seed": SEED,
        "seed_note": "canonical seed, not selected. The seven-seed band is below; "
                     "no metric here should be quoted as a point estimate.",
        "features_in_booster_order": feats31,
        "n_features": len(feats31),
        "target_transform": {
            "forward": "z = log(u / (1 - u)) with u = clip(alpha - 1, 1e-6, 1 - 1e-6)",
            "inverse": "alpha = 1 + 1 / (1 + exp(-z))",
        },
        "monotone_constrained_features": ["n_customers", "dimension"],
        "monotone_constraints_method": "basic",
        "hyperparameters": V3_FROZEN,
        "early_stopping_metric": "cost_mape (cost-level MAPE on the val split)",
        "early_stopping_rounds": 100,
        "best_iteration": int(booster.best_iteration),
        "num_trees": int(booster.num_trees()),
        "training_protocol": (
            "Fit on split=='train' only, early-stopped on split=='val'. Identical "
            "to the shipped recipe; the only change is the training corpus."),
        "training_tables": ["tsp_features_v4.csv",
                            "alpha_coverage/coverage_features.csv"],
        "concatenation_order": "corpus block then coverage block, each sorted "
                               "ascending by instance_name; no shuffling",
        "rows": {
            "corpus_train": int((corpus.split == "train").sum()),
            "coverage_train": int((cov.split == "train").sum()),
            "train_total": int(len(tr)),
            "corpus_val": int((corpus.split == "val").sum()),
            "coverage_val": int((cov.split == "val").sum()),
            "val_total": int(len(vl)),
            "coverage_test_heldout": int((cov.split == "test").sum()),
            "nd_test_unchanged": 16920,
        },
        "this_seed_metrics": (this_seed.iloc[0].to_dict() if len(this_seed) else None),
        "seven_seed_band": study.get("cov"),
        "frozen_reference": study.get("frozen"),
        "control": "base arm at seed 42 reproduces gart2_final.joblib with "
                   "max |delta alpha| = 0 over all 20,475 scored instances",
    }, indent=2, default=float), encoding="utf-8")

    print(f"[freeze] wrote {OUT_MODEL.name} ({booster.num_trees()} trees) and "
          f"{OUT_JSON.name}")
    print(f"[freeze] sha256 {sha}")


if __name__ == "__main__":
    main()
