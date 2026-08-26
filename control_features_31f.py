"""Shared data intake for the 31-feature same-feature controls.

The paper describes ``Linear_V3`` and ``NN_V3`` as controls on GART 2.0's
"identical feature vector". They are not: GART 2.0 consumes 31 columns of
``tsp_features_v4.csv``, ``NN_V3`` consumes 30 columns of
``tsp_features_v3.csv``, and ``Linear_V3`` consumes 28 (``train_linear_v3.py``
additionally drops ``bounding_hypervolume`` and ``node_density``). The two
tables also disagree on the label: ``optimal_cost`` and ``mst_total_length``
were both revised between v3 and v4, so the targets are not the same numbers.

This module is the single intake the 31-feature replacements share, so neither
of them can drift from the production model's inputs:

  * the feature list is read off the frozen booster, never from a literal;
  * the training table is the production table, ``tsp_features_v4.csv``;
  * the split assignment, the alpha definition and the alpha clip are the ones
    ``lgbm_model_v3/train_gart2.py`` uses.

Nothing here is model-class specific. Preprocessing (standardisation, quantile
transform) stays in the individual trainers, because that is part of the model
class being controlled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent
DATA_FILE = REPO / "tsp_features_v4.csv"
BOOSTER = REPO / "lgbm_model_v3" / "gart2_final.joblib"
SIDECAR = REPO / "lgbm_model_v3" / "gart2_final.json"

ALPHA_CLIP = (1.0, 2.0)
RANDOM_STATE = 42


class FiniteStandardScaler(StandardScaler):
    """``StandardScaler`` that emits 0.0 where standardisation is undefined.

    Two of GART 2.0's 31 columns, ``bounding_hypervolume`` and
    ``node_density``, reach 4.6e299. Their sample second moment is ~1e599,
    which is not representable in float64, so ``StandardScaler`` computes
    ``var_ = nan``, ``scale_ = nan`` and maps the entire column to NaN --
    which then raises inside any downstream linear estimator. The overflow is
    not an implementation artifact: the variance of that column genuinely
    exceeds the float64 range, so no standardisation of it exists.

    Rather than dropping the columns (which would change the feature vector
    and reintroduce the very mismatch these controls exist to remove), the
    non-finite entries are set to 0.0 -- their post-standardisation mean. The
    columns stay in the vector at their production positions and contribute
    nothing, which is the accurate statement about what a linear form can
    extract from them.

    This costs no information. Both columns are exact ``exp()`` images of
    ``log_bounding_hypervolume`` and ``log_node_density``, which are present
    in the same vector, so the pair is redundant by construction rather than
    merely unusable.

    Defined at module scope, not inside a training script, so the pickled
    pipeline unpickles in any process. ``nn_est_alpha_v3/estimator_v3.py``
    carries a ``sys.modules['__main__']`` injection hack precisely because
    ``StableV3Scaler`` was pickled from ``__main__``; this avoids repeating it.
    """

    def transform(self, X, copy=None):  # type: ignore[override]
        out = super().transform(X, copy=copy)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def undefined_columns(self, feature_names: List[str]) -> List[str]:
        """Names of the columns this scaler could not standardise."""
        return [f for f, s in zip(feature_names, self.scale_) if not np.isfinite(s)]


class Stable31Scaler:
    """``nn_est_alpha_v3/train_v3.py``'s ``StableV3Scaler``, unchanged.

    Quantile-to-normal, then standardise, then clip to +/-10. Reproduced here
    only so that it lives at module scope in an importable module: the V3 copy
    was pickled from ``__main__`` and ``estimator_v3.py`` has to inject the
    class into ``sys.modules['__main__']`` at load time to unpickle it.

    The quantile transform is also why the neural control tolerates the two raw
    scale columns that defeat the linear control -- rank-based mapping has no
    second moment to overflow.
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        from sklearn.preprocessing import QuantileTransformer
        self.qt = QuantileTransformer(output_distribution="normal",
                                      random_state=random_state)
        self.ss = StandardScaler()

    def fit(self, X):
        self.ss.fit(self.qt.fit_transform(X))
        return self

    def transform(self, X):
        return np.clip(self.ss.transform(self.qt.transform(X)), -10.0, 10.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def production_features() -> List[str]:
    """GART 2.0's 31 inputs, in booster order, cross-checked against the sidecar."""
    feats = list(joblib.load(BOOSTER).feature_name())
    declared = json.loads(SIDECAR.read_text(encoding="utf-8"))["features_in_booster_order"]
    if list(declared) != feats:
        raise RuntimeError("gart2_final sidecar disagrees with the booster feature order")
    if "mst_total_length" in feats:
        raise RuntimeError("mst_total_length is in the feature list; alpha = cost / mst leaks")
    return feats


def load_splits(features: List[str]) -> Dict[str, dict]:
    """``train`` / ``val`` / ``test`` parts of the production training table.

    ``X`` is a float64 frame in booster order, ``y`` is alpha in [1, 2], and
    ``mst`` / ``cost`` carry the cost-level scale so every metric in this study
    is computed the way ``train_gart2.cost_metrics`` computes it.
    """
    df = pd.read_csv(DATA_FILE)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(f"{DATA_FILE.name} is missing {missing}")

    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(*ALPHA_CLIP)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)

    parts: Dict[str, dict] = {}
    for name in ("train", "val", "test"):
        m = df["split"] == name
        X = df.loc[m, features].astype(np.float64)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        parts[name] = {
            "X": X,
            "y": df.loc[m, "alpha"].to_numpy(dtype=np.float64),
            "mst": df.loc[m, "mst_total_length"].to_numpy(dtype=np.float64),
            "cost": df.loc[m, "optimal_cost"].to_numpy(dtype=np.float64),
            "instance": df.loc[m, "instance_name"].to_numpy(),
        }
    return parts


def cost_metrics(alpha_pred, mst, true_cost) -> Dict[str, float]:
    """Cost-level metrics, identical to ``train_gart2.cost_metrics``."""
    a = np.clip(np.asarray(alpha_pred, dtype=np.float64), *ALPHA_CLIP)
    err = (a * mst - true_cost) / true_cost
    return {
        "sdpe": float(np.std(err, ddof=1) * 100.0),
        "mape": float(np.mean(np.abs(err)) * 100.0),
        "mspe": float(np.mean(err ** 2) * 1e4),
        "bias": float(np.mean(err) * 100.0),
    }
