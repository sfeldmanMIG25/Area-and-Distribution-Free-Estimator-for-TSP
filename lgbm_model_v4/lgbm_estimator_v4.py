"""GART V4 inference wrapper, API-compatible with the V3 estimator.

Exposes a single class :class:`TSP_V4_LGBM_Estimator` with ``.estimate``
returning the same dict shape (``estimate``, ``alpha``, ``mst_length``,
``feature_time``, ``inference_time``) that the benchmark runners expect.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from feature_engineering import compute_features

ALPHA_CLIP = (1.0, 2.0)


class TSP_V4_LGBM_Estimator:
    def __init__(self, model_dir: str | None = None):
        if model_dir is None:
            model_dir = str(THIS_DIR)
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "lgbm_alpha_model_v4.joblib")

        # The LightGBM booster knows its own feature names.
        self.features_required: list[str] = list(self.model.feature_name())

    def estimate(self, coordinates: np.ndarray, dimension: int, grid_size: int) -> dict:
        coords = np.asarray(coordinates, dtype=np.float32)
        coords = np.unique(coords, axis=0)
        n = coords.shape[0]

        if n < 3:
            # Not enough points to form features; fall back to MST = 0 path.
            return {
                "estimate": 0.0, "alpha": float("nan"),
                "mst_length": 0.0, "feature_time": 0.0, "inference_time": 0.0,
            }

        t0 = time.perf_counter()
        feats = compute_features(coords, int(dimension), int(grid_size))
        t_feat = time.perf_counter() - t0

        # Model expects only the trained columns, in the trained order.
        row = {k: feats.get(k, 0.0) for k in self.features_required}
        df = pd.DataFrame([row])

        t1 = time.perf_counter()
        alpha = float(np.clip(self.model.predict(df)[0], *ALPHA_CLIP))
        t_inf = time.perf_counter() - t1

        mst_len = float(feats["mst_total_length"])
        return {
            "estimate": alpha * mst_len,
            "alpha": alpha,
            "mst_length": mst_len,
            "feature_time": t_feat,
            "inference_time": t_inf,
        }


if __name__ == "__main__":
    # Smoke test.
    import numpy as np
    est = TSP_V4_LGBM_Estimator()
    rng = np.random.default_rng(0)
    pts = rng.random((200, 2), dtype=np.float32) * 100.0
    res = est.estimate(pts, 2, 100)
    print(res)
