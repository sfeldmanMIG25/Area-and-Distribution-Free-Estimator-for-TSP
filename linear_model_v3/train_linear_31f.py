"""Linear control on GART 2.0's identical 31-column feature vector.

Replaces ``train_linear_v3.py`` as the same-feature control. Two things change
and nothing else:

  1. the feature matrix is GART 2.0's 31 columns in booster order, instead of
     the 28 that ``train_linear_v3.py`` selected (V3's 30 minus
     ``bounding_hypervolume`` and ``node_density``);
  2. the training table is ``tsp_features_v4.csv``, the table GART 2.0 is
     fitted on, instead of ``tsp_features_v3.csv``.

The estimator is unchanged: median imputation, standardisation, ordinary least
squares, fitted on ``split == 'train'`` only. There is nothing to tune in an
OLS fit, so "same tuning discipline" is vacuous here and no choice was made.

THE TWO RAW SCALE COLUMNS
-------------------------
``bounding_hypervolume`` and ``node_density`` reach 4.6e299 on this corpus, so
their sample variance (~1e599) is outside the float64 range and stock
``StandardScaler`` returns ``scale_ = nan``, mapping the column to NaN and
raising inside ``LinearRegression``. ``FiniteStandardScaler`` sets those
entries to 0.0 instead, so the columns keep their production positions in the
31-vector and contribute nothing. That is the accurate statement about a linear
form in an unbounded 300-decade covariate, and it costs no information: both
columns are exact ``exp()`` images of ``log_bounding_hypervolume`` and
``log_node_density``, which are in the same vector. ``train_linear_v3.py``
dropped them outright for the same underlying reason. ``--diagnose`` refits on
the 29 estimable columns and prints the difference, so "the two inert columns
cost this control nothing" is measured rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO))

from control_features_31f import (  # noqa: E402
    ALPHA_CLIP, DATA_FILE, FiniteStandardScaler, cost_metrics, load_splits,
    production_features,
)

MODEL_OUT = THIS / "linear_31f_model.joblib"
SIDECAR_OUT = THIS / "linear_31f_model.json"

#: Columns whose variance overflows float64 and are annihilated by the scaler.
INERT_UNDER_STANDARDISATION = ("bounding_hypervolume", "node_density")


def build_pipeline() -> Pipeline:
    """The V3 linear pipeline; only the scaler's NaN handling differs."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", FiniteStandardScaler()),
        ("model", LinearRegression()),
    ])


def fit_and_score(features, parts) -> tuple[Pipeline, dict]:
    pipe = build_pipeline()
    with warnings.catch_warnings():
        # The overflow in the two raw scale columns is expected and documented.
        warnings.simplefilter("ignore", RuntimeWarning)
        pipe.fit(parts["train"]["X"][features], parts["train"]["y"])
    scores = {}
    for split in ("train", "val", "test"):
        p = parts[split]
        pred = pipe.predict(p["X"][features])
        scores[split] = cost_metrics(pred, p["mst"], p["cost"])
    return pipe, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose", action="store_true",
                    help="also fit without the two inert columns and print the delta")
    args = ap.parse_args()

    features = production_features()
    parts = load_splits(features)
    print(f"[linear_31f] {len(features)} features, table {DATA_FILE.name}, "
          f"train={len(parts['train']['y'])} val={len(parts['val']['y'])} "
          f"test={len(parts['test']['y'])}")

    pipe, scores = fit_and_score(features, parts)

    # Confirm, rather than assume, which raw columns are annihilated.
    inert = pipe.named_steps["scaler"].undefined_columns(features)
    coefs = dict(zip(features, pipe.named_steps["model"].coef_.tolist()))
    print(f"[linear_31f] columns with non-finite scale (inert): {inert}")

    for split in ("val", "test"):
        m = scores[split]
        print(f"[linear_31f] {split:<5} sdpe={m['sdpe']:.4f} mape={m['mape']:.4f} "
              f"bias={m['bias']:+.4f}")

    delta = None
    if args.diagnose:
        keep = [f for f in features if f not in INERT_UNDER_STANDARDISATION]
        _, s29 = fit_and_score(keep, parts)
        delta = {k: {"sdpe": s29[k]["sdpe"] - scores[k]["sdpe"],
                     "mape": s29[k]["mape"] - scores[k]["mape"]}
                 for k in ("val", "test")}
        print(f"[linear_31f] diagnostic: dropping {list(INERT_UNDER_STANDARDISATION)} "
              f"({len(keep)} cols) changes test sdpe by {delta['test']['sdpe']:+.6f} pp, "
              f"mape by {delta['test']['mape']:+.6f} pp")

    joblib.dump(pipe, MODEL_OUT)
    SIDECAR_OUT.write_text(json.dumps({
        "name": "Linear (GART 2.0 features)",
        "model_key": "Linear_31F",
        "artifact": MODEL_OUT.name,
        "supersedes": "linear_alpha_model_v3.joblib",
        "role": "same-feature control -- isolates the model class against GART 2.0",
        "n_features": len(features),
        "features_in_model_order": features,
        "feature_source": "lgbm_model_v3/gart2_final.joblib::feature_name()",
        "training_table": DATA_FILE.name,
        "target": "alpha = clip(optimal_cost / mst_total_length, 1, 2), predicted directly",
        "alpha_clip": list(ALPHA_CLIP),
        "pipeline": "SimpleImputer(median) -> FiniteStandardScaler -> LinearRegression",
        "pipeline_provenance":
            "linear_model_v3/train_linear_v3.py, with StandardScaler replaced by "
            "FiniteStandardScaler: identical except that an undefined standardised "
            "entry becomes 0.0 instead of NaN",
        "training_protocol": "fit on split=='train' only; val and test never fitted on",
        "inert_columns": inert,
        "inert_columns_note":
            "sample variance exceeds the float64 range, so scale_ is nan and the "
            "column is mapped to 0.0. Present in the input vector at its production "
            "position, unusable by a linear form. Both are exact exp() images of log "
            "columns already in the vector, so no information is lost -- see "
            "diagnostic_drop_inert_delta_pp.",
        "diagnostic_drop_inert_delta_pp": delta,
        "coefficients": coefs,
        "intercept": float(pipe.named_steps["model"].intercept_),
        "metrics": scores,
    }, indent=2), encoding="utf-8")
    print(f"[linear_31f] saved {MODEL_OUT.name} and {SIDECAR_OUT.name}")


if __name__ == "__main__":
    main()
