"""Inference wrapper for the 31-feature linear control (``Linear_31F``).

Deliberately thin. Everything that determines *what the model sees* is
delegated to the production feature extractor, ``feature_engineering_gart2``,
so the control cannot drift from GART 2.0 by so much as a percentile
definition. ``estimator_linear_v3.py`` re-implements the feature block by hand
and has already diverged from it (it never computes ``bounding_hypervolume``,
``node_density`` or ``greedy_nn_over_mst``, and it zero-fills anything the
model asks for and it did not compute).

Interface parity with ``TSP_GART2_Estimator``:

  * ``estimate(coordinates, dimension, grid_size=None)`` -- the native path,
    including the same duplicate-point collapse and the same ``n < 3`` status;
  * ``features_required`` and ``predict_alpha(feats)`` -- what the TSPLIB
    runner's hybrid builder dispatches on, so this control scores the
    non-Euclidean instances through the same code path GART 2.0 uses, and is
    held to the same in-distribution guard on ``greedy_nn_over_mst``.

``estimator_linear_v3.py`` exposes neither, which is why every non-Euclidean
row for ``Linear_V3`` in the TSPLIB results carries ``no_hybrid_path``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
for _p in (str(REPO), str(REPO / "lgbm_model_v3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feature_engineering_gart2 import ALPHA_CLIP, compute_features  # noqa: E402
import control_features_31f  # noqa: E402,F401  (registers FiniteStandardScaler for unpickling)

MODEL_FILE = "linear_31f_model.joblib"
SIDECAR_FILE = "linear_31f_model.json"


class TSP_Linear31_Estimator:
    """Least-squares alpha on GART 2.0's identical 31-column feature vector."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else THIS_DIR
        path = self.model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"Linear_31F model not found at {path}")
        self.model = joblib.load(path)
        self.features_required = list(self.model.feature_names_in_)

        self.sidecar: Dict[str, Any] = json.loads(
            (self.model_dir / SIDECAR_FILE).read_text(encoding="utf-8"))
        declared = self.sidecar["features_in_model_order"]
        if list(declared) != self.features_required:
            raise RuntimeError("linear_31f sidecar feature order disagrees with the pipeline")
        if "mst_total_length" in self.features_required:
            raise RuntimeError("mst_total_length is a model input; alpha = cost / mst leaks")

    # ------------------------------------------------------------------
    def predict_alpha(self, feats: Dict[str, float]) -> float:
        """Alpha from a feature dict. Missing features raise; never defaulted."""
        missing = [k for k in self.features_required if k not in feats]
        if missing:
            raise KeyError(f"Linear_31F requires {len(self.features_required)} "
                           f"features; missing: {missing}")
        row = {k: feats[k] for k in self.features_required}
        df = pd.DataFrame([row], columns=self.features_required)
        return float(np.clip(self.model.predict(df)[0], *ALPHA_CLIP))

    # ------------------------------------------------------------------
    def estimate(self, coordinates, dimension: int, grid_size=None) -> dict:
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
        return {"estimate": alpha * mst_len, "alpha": alpha, "mst_length": mst_len,
                "feature_time": t_feat, "inference_time": t_inf, "status": "ok"}


if __name__ == "__main__":
    est = TSP_Linear31_Estimator()
    print(f"{len(est.features_required)} features; inert: {est.sidecar['inert_columns']}")
    rng = np.random.default_rng(0)
    print(est.estimate((rng.random((500, 2), dtype=np.float32) * 100.0), 2))
