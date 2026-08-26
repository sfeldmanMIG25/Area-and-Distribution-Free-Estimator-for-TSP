"""GART 2.0 inference wrapper -- the production estimator.

Wraps the frozen booster ``gart2_final.joblib`` (31 features, 1118 trees) and
its sidecar ``gart2_final.json``.

API-compatible with ``TSP_V3_LGBM_Estimator``:
``estimate(coordinates, dimension, grid_size)`` returns a dict with
``estimate``, ``alpha``, ``mst_length``, ``feature_time``, ``inference_time``.

``grid_size`` is accepted for call-site compatibility and ignored -- it is a
generator fingerprint and is not a model input.

THREE THINGS THAT MUST NOT DRIFT
--------------------------------
1. **Feature order** comes from the booster (``feature_name()``), never from a
   literal in this file. The sidecar's ``features_in_booster_order`` is checked
   against it at load time and a mismatch raises.
2. **The target is logit(alpha - 1), not alpha.** The booster's raw output is
   ``z``; the deployed alpha is ``1 + sigmoid(z)``. Feeding ``z`` straight
   through (or clipping it to [1, 2]) yields plausible-looking numbers that are
   wrong -- ``z`` happens to live in roughly the same range. The inverse is
   applied here, in exactly the form the sidecar records, and the recorded form
   is verified at load time.
3. **A missing feature raises.** It is never defaulted to 0.0. Silent zero-fill
   has caused two separate defects in this project.

Monotone constraints (non-increasing in ``n_customers`` and ``dimension``) are
baked into the trees at fit time and have no inference-time effect; they are
surfaced through :attr:`provenance` for reporting.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")

import joblib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from feature_engineering_gart2 import (  # noqa: E402
    ALPHA_CLIP,
    TRAIN_GREEDY_RANGE,
    compute_features,
    greedy_ratio_from_matrix,
)

MODEL_FILE = "gart2_final.joblib"
SIDECAR_FILE = "gart2_final.json"

#: The only target transform this wrapper knows how to invert. The sidecar must
#: declare exactly this, or the model in the directory is not the model this
#: code was written for and we refuse to guess.
_EXPECTED_INVERSE = "alpha = 1 + 1 / (1 + exp(-z))"


def _inverse_logit_target(z) -> np.ndarray:
    """``alpha = 1 + sigmoid(z)`` -- the inverse of ``z = logit(alpha - 1)``.

    Structurally bounded to (1, 2); the clip that follows in the caller is a
    belt-and-braces guard against a NaN/inf leaking in, not a correction.
    """
    z = np.asarray(z, dtype=np.float64)
    return 1.0 + 1.0 / (1.0 + np.exp(-z))


class TSP_GART2_Estimator:
    """Alpha-ratio tour-length estimator: cost ~= alpha(features) * MST."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else THIS_DIR
        path = self.model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"GART 2.0 model not found at {path}")
        self.model = joblib.load(path)

        # Feature order is the booster's, always.
        self.features_required = list(self.model.feature_name())
        if "mst_total_length" in self.features_required:
            raise RuntimeError(
                "model was trained with mst_total_length as an input; "
                "alpha = cost / mst_total_length, so this leaks the label"
            )

        self.sidecar: Dict[str, Any] = {}
        sc_path = self.model_dir / SIDECAR_FILE
        if sc_path.exists():
            self.sidecar = json.loads(sc_path.read_text(encoding="utf-8"))

            declared = self.sidecar.get("features_in_booster_order")
            if declared is not None and list(declared) != self.features_required:
                raise RuntimeError(
                    "sidecar feature order disagrees with the booster; refusing "
                    "to run. Booster has "
                    f"{len(self.features_required)} features, sidecar declares "
                    f"{len(declared)}."
                )

            inverse = (self.sidecar.get("target_transform") or {}).get("inverse")
            if inverse is not None and inverse.strip() != _EXPECTED_INVERSE:
                raise RuntimeError(
                    "sidecar declares a target inverse this wrapper cannot "
                    f"apply: {inverse!r}. Expected {_EXPECTED_INVERSE!r}."
                )
        elif self.model_dir == THIS_DIR:
            # A throwaway smoke-test directory is allowed to ship without a
            # sidecar; the production directory is not.
            raise FileNotFoundError(f"GART 2.0 sidecar not found at {sc_path}")

        self.provenance: Dict[str, Any] = {
            "model_file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_features": len(self.features_required),
            "num_trees": int(self.model.num_trees()),
            "target_transform": "logit(alpha - 1); inverse 1 + sigmoid(z)",
            "alpha_clip": list(ALPHA_CLIP),
            "monotone_constrained_features":
                self.sidecar.get("monotone_constrained_features", []),
            "monotone_direction": self.sidecar.get("monotone_direction"),
            "monotone_constraints_method":
                self.sidecar.get("monotone_constraints_method"),
        }

    # ------------------------------------------------------------------
    def predict_alpha(self, feats: Dict[str, float]) -> float:
        """Deployed alpha from a feature dict.

        Raises ``KeyError`` naming every absent feature rather than filling a
        default. Used by the estimator's own coordinate path and by the
        non-Euclidean hybrid builder in ``run_all_models_tsplib``.
        """
        missing = [k for k in self.features_required if k not in feats]
        if missing:
            raise KeyError(
                f"GART 2.0 requires {len(self.features_required)} features; "
                f"missing: {missing}"
            )
        row = {k: feats[k] for k in self.features_required}
        df = pd.DataFrame([row], columns=self.features_required)
        z = self.model.predict(df)
        alpha = _inverse_logit_target(z)[0]
        return float(np.clip(alpha, *ALPHA_CLIP))

    # ------------------------------------------------------------------
    def estimate(self, coordinates, dimension: int, grid_size=None) -> dict:
        """Estimate from coordinates (the native Euclidean path)."""
        coords = np.unique(np.asarray(coordinates, dtype=np.float32), axis=0)
        if coords.shape[0] < 3:
            return {"estimate": 0.0, "alpha": float("nan"), "mst_length": 0.0,
                    "feature_time": 0.0, "inference_time": 0.0, "status": "n<3"}

        t0 = time.perf_counter()
        feats = compute_features(coords, int(dimension))
        t_feat = time.perf_counter() - t0

        t1 = time.perf_counter()
        alpha = self.predict_alpha(feats)
        t_inf = time.perf_counter() - t1

        mst_len = float(feats["mst_total_length"])
        return {"estimate": alpha * mst_len, "alpha": alpha,
                "mst_length": mst_len, "feature_time": t_feat,
                "inference_time": t_inf, "status": "ok"}

    # ------------------------------------------------------------------
    def greedy_ratio_from_distance_matrix(self, D, mst_length=None) -> float:
        """Hybrid/non-Euclidean helper: the one extra feature, computed from
        the TRUE distance matrix (never from an MDS embedding)."""
        return greedy_ratio_from_matrix(D, mst_length)

    @staticmethod
    def greedy_ratio_in_training_range(value: float) -> bool:
        lo, hi = TRAIN_GREEDY_RANGE
        return bool(np.isfinite(value) and lo <= value <= hi)


if __name__ == "__main__":
    est = TSP_GART2_Estimator()
    print(json.dumps(est.provenance, indent=2))
    rng = np.random.default_rng(0)
    pts = (rng.random((500, 2), dtype=np.float32) * 100.0).astype(np.float32)
    print(est.estimate(pts, 2))
