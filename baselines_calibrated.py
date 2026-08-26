"""Calibrated constant-ratio baselines fitted on the training split.

Motivation
----------
The benchmark previously compared GART 2.0 against a fixed MST multiplier whose
``d >= 3`` values were author-chosen (``rho(d) = 1 + 0.15/d``). Those values sit
8-15 percentage points below the training-split mean of
``alpha = L_TSP / L_MST``, so that row measured an uncalibrated constant rather
than the best constant available without learning. This module replaces it with
constants fitted on the training split only, which is the honest reference point
for any claim that the learned ``alpha`` adds value.

Two calibrated baselines are provided:

``rho_d``
    One constant per dimension. This is the direct, fitted counterpart of the
    old fixed schedule.
``rho_dn``
    One constant per (dimension, size-bucket) cell. This is the strongest
    constant-ratio baseline obtainable without a learned model.

A third, literature-anchored constant is kept separately in
:data:`ASYMPTOTIC_RHO`: the ratio of the BHH tour constant to the Euclidean MST
constant for uniform points in the plane. It is defined only where both
constants are published, so it is applied at ``d = 2`` and reported as an
asymptotic reference rather than a tuned baseline.

The fitted table is written to ``calibrated_alpha_table.json`` so benchmark runs
load a frozen artifact instead of refitting, and so the fit is auditable.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

# Ratio of published asymptotic constants for i.i.d. uniform points in the unit
# square: beta_TSP = 0.7124 (Johnson, McGeoch & Rothberg 1996) over
# beta_MST = 0.6331 (Cortina-Borja & Robinson 2000). Defined for d = 2 only;
# no comparably pinned MST constant exists for d >= 3.
ASYMPTOTIC_RHO: dict[int, float] = {2: 0.7124 / 0.6331}

# Size buckets used by the (d, n) table. Right-closed, matching the reporting
# buckets used in the paper's per-size tables.
N_BUCKET_EDGES: tuple[int, ...] = (0, 10, 50, 100, 200, 500, 1_000_000)

TABLE_PATH = Path(__file__).parent / "calibrated_alpha_table.json"
FEATURES_CSV = Path(__file__).parent / "tsp_features_v3.csv"


def n_bucket_index(n: int) -> int:
    """Return the index of the size bucket containing ``n``."""
    for i in range(len(N_BUCKET_EDGES) - 1):
        if N_BUCKET_EDGES[i] < n <= N_BUCKET_EDGES[i + 1]:
            return i
    return len(N_BUCKET_EDGES) - 2


def fit_table(features_csv: Path = FEATURES_CSV, split: str = "train") -> dict:
    """Fit the constant-ratio tables on one split of the feature corpus.

    Only rows from ``split`` are used, so the test partition never informs the
    baseline. Cells are the mean of ``alpha`` over the rows in the cell.
    """
    import pandas as pd

    cols = ["n_customers", "dimension", "optimal_cost", "mst_total_length", "split"]
    df = pd.read_csv(features_csv, usecols=cols)
    df = df[df["split"] == split]
    df = df[df["mst_total_length"] > 0]
    alpha = (df["optimal_cost"] / df["mst_total_length"]).clip(1.0, 2.0)
    df = df.assign(alpha=alpha)

    by_d = df.groupby("dimension")["alpha"].mean()
    rho_d = {int(k): float(v) for k, v in by_d.items()}

    bucket = df["n_customers"].map(n_bucket_index)
    by_dn = df.assign(nb=bucket).groupby(["dimension", "nb"])["alpha"].mean()
    rho_dn = {f"{int(d)}|{int(nb)}": float(v) for (d, nb), v in by_dn.items()}

    return {
        "source_csv": str(features_csv.name),
        "split": split,
        "rows_used": int(len(df)),
        "n_bucket_edges": list(N_BUCKET_EDGES),
        "rho_d": rho_d,
        "rho_dn": rho_dn,
    }


def load_table(path: Path = TABLE_PATH) -> dict:
    """Load the frozen calibration table, fitting and writing it if absent."""
    if not path.exists():
        table = fit_table()
        path.write_text(json.dumps(table, indent=2), encoding="utf-8")
        return table
    return json.loads(path.read_text(encoding="utf-8"))


def _nearest_trained_dimension(d: int, trained: list[int]) -> int:
    """Return the trained dimension closest to ``d``.

    ``d = 100`` never appears in training, so the table extrapolates from the
    largest trained dimension. The choice is recorded here rather than hidden
    inside the estimator.
    """
    return min(trained, key=lambda t: (abs(t - d), t))


class CalibratedMSTRatio:
    """``L_hat = rho_hat * L_MST`` with ``rho_hat`` read from the fitted table.

    ``granularity='d'`` uses one constant per dimension; ``granularity='dn'``
    uses one constant per (dimension, size-bucket) cell.
    """

    def __init__(self, granularity: str = "d", table: dict | None = None):
        if granularity not in ("d", "dn"):
            raise ValueError("granularity must be 'd' or 'dn'")
        self.granularity = granularity
        self.table = table if table is not None else load_table()
        self._rho_d = {int(k): v for k, v in self.table["rho_d"].items()}
        self._rho_dn = {k: v for k, v in self.table["rho_dn"].items()}
        self._trained_dims = sorted(self._rho_d)

    def rho(self, d: int, n: int) -> float:
        d_eff = d if d in self._rho_d else _nearest_trained_dimension(d, self._trained_dims)
        if self.granularity == "dn":
            key = f"{d_eff}|{n_bucket_index(n)}"
            if key in self._rho_dn:
                return self._rho_dn[key]
        return self._rho_d[d_eff]

    def estimate(self, coordinates, dimension, grid_size=None, precomputed_mst=None):
        """Return the benchmark-runner result dict."""
        import tsp_utils_2 as academic

        coords = np.asarray(coordinates, dtype=np.float64)
        n = len(np.unique(coords, axis=0))
        if precomputed_mst is not None:
            mst_len, t_feat = float(precomputed_mst), 0.0
        else:
            mst_len, t_feat = academic.get_mst_length(coords)
        t1 = time.perf_counter()
        est = self.rho(int(dimension), int(n)) * float(mst_len)
        return {
            "estimate": est,
            "feature_time": t_feat,
            "inference_time": time.perf_counter() - t1,
            "status": "ok",
        }


class AsymptoticMSTRatio:
    """``L_hat = (beta_TSP / beta_MST) * L_MST`` using published constants.

    Defined only for dimensions listed in :data:`ASYMPTOTIC_RHO`. Callers must
    gate on dimension and record a status row rather than extrapolating, because
    no published MST constant pins the ratio for ``d >= 3``.
    """

    def __init__(self, rho_table: dict[int, float] | None = None):
        self.rho_table = dict(ASYMPTOTIC_RHO if rho_table is None else rho_table)

    def supports(self, dimension: int) -> bool:
        return int(dimension) in self.rho_table

    def estimate(self, coordinates, dimension, grid_size=None, precomputed_mst=None):
        import tsp_utils_2 as academic

        d = int(dimension)
        if d not in self.rho_table:
            # No published MST constant pins the ratio here, so report the gap
            # rather than extrapolating a literature value into a dimension the
            # literature does not cover.
            return {"estimate": float("nan"), "feature_time": 0.0,
                    "inference_time": 0.0, "status": "asymptotic_rho_undefined"}
        coords = np.asarray(coordinates, dtype=np.float64)
        if precomputed_mst is not None:
            mst_len, t_feat = float(precomputed_mst), 0.0
        else:
            mst_len, t_feat = academic.get_mst_length(coords)
        t1 = time.perf_counter()
        est = self.rho_table[d] * float(mst_len)
        return {
            "estimate": est,
            "feature_time": t_feat,
            "inference_time": time.perf_counter() - t1,
            "status": "ok",
        }


if __name__ == "__main__":
    tbl = fit_table()
    TABLE_PATH.write_text(json.dumps(tbl, indent=2), encoding="utf-8")
    tbl = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    print(f"Fitted on {tbl['rows_used']} rows of split '{tbl['split']}'.")
    print("rho_d:")
    for d in sorted(int(k) for k in tbl["rho_d"]):
        print(f"  d={d:<4d} {tbl['rho_d'][str(d)]:.4f}")
    print(f"Asymptotic reference: d=2 rho={ASYMPTOTIC_RHO[2]:.4f}")
    print(f"Wrote {TABLE_PATH}")
