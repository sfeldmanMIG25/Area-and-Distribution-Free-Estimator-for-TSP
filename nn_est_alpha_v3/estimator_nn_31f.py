"""Inference wrapper for the 31-feature neural control (``NN_31F``).

Same contract as ``linear_model_v3/estimator_linear_31f.py``: the feature block
is the production extractor ``feature_engineering_gart2.compute_features``, not
a hand-rolled copy, so the control and GART 2.0 consume a bit-identical vector.

Two failure modes in ``estimator_v3.py`` that are not reproduced here:

  * it computes its own 30-feature block, which has already drifted from the
    production extractor (float32 centroid accumulation, unguarded skew and
    kurtosis on near-degenerate edge sets);
  * it exposes ``features_required`` but no ``predict_alpha``, so the TSPLIB
    hybrid builder dispatches to ``estimator.model.predict(df)`` -- a method a
    ``torch.nn.Module`` does not have. Every non-Euclidean ``NN_V3`` row in the
    TSPLIB results is ``exception:AttributeError`` for exactly that reason.
    Exposing ``predict_alpha`` here gives the control a working hybrid path and
    subjects it to the same ``greedy_nn_over_mst`` in-distribution guard GART
    2.0 obeys.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
for _p in (str(REPO), str(REPO / "lgbm_model_v3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from feature_engineering_gart2 import ALPHA_CLIP, compute_features  # noqa: E402
from control_features_31f import Stable31Scaler  # noqa: E402,F401  (unpickle target)
from train_nn_31f import TSP_Leap_Model  # noqa: E402

MODEL_FILE = "nn_31f_model.pt"
SCALER_FILE = "nn_31f_scaler.joblib"
SIDECAR_FILE = "nn_31f_model.json"


class TSP_NN31_Estimator:
    """Gated-residual network alpha on GART 2.0's identical 31-column vector."""

    def __init__(self, model_dir: Optional[str] = None, checkpoint: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else THIS_DIR
        ckpt_path = Path(checkpoint) if checkpoint else self.model_dir / MODEL_FILE
        if not ckpt_path.exists():
            raise FileNotFoundError(f"NN_31F checkpoint not found at {ckpt_path}")

        self.scaler = joblib.load(self.model_dir / SCALER_FILE)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.features_required = list(ckpt["features"])
        self.seed = ckpt.get("seed")
        if "mst_total_length" in self.features_required:
            raise RuntimeError("mst_total_length is a model input; alpha = cost / mst leaks")

        self.sidecar: Dict[str, Any] = json.loads(
            (self.model_dir / SIDECAR_FILE).read_text(encoding="utf-8"))
        if list(self.sidecar["features_in_model_order"]) != self.features_required:
            raise RuntimeError("nn_31f sidecar feature order disagrees with the checkpoint")

        hp = ckpt["params"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TSP_Leap_Model(ckpt["input_dim"], hp["hidden_dim"],
                                    hp["num_blocks"], hp["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

    # ------------------------------------------------------------------
    def predict_alpha(self, feats: Dict[str, float]) -> float:
        """Alpha from a feature dict. Missing features raise; never defaulted."""
        missing = [k for k in self.features_required if k not in feats]
        if missing:
            raise KeyError(f"NN_31F requires {len(self.features_required)} "
                           f"features; missing: {missing}")
        vec = np.array([[float(feats[k]) for k in self.features_required]], dtype=np.float64)
        vec = self.scaler.transform(vec)
        with torch.no_grad():
            y = self.model(torch.tensor(vec, dtype=torch.float32,
                                        device=self.device)).cpu().item()
        return float(np.clip(1.0 + y, *ALPHA_CLIP))

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
    est = TSP_NN31_Estimator()
    print(f"{len(est.features_required)} features, seed {est.seed}, device {est.device}")
    rng = np.random.default_rng(0)
    print(est.estimate((rng.random((500, 2), dtype=np.float32) * 100.0), 2))
