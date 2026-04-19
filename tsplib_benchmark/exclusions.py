"""Principled, deterministic exclusion rule for non-metric TSPLIB instances.

Motivation
----------
GART 2.0 and the MDS-embedding pipeline both assume symmetric metric
distances. For any metric TSP, the double-tree construction proves

    L_MST  <=  L_TSP  <=  2 * L_MST.

An instance whose published optimum falls outside this band is
mathematically incompatible with the metric assumption -- triangle
inequality is violated somewhere in its distance matrix. We use both
bounds (with a 25% slack above for integer-rounded EXPLICIT matrices)
as a deterministic, model-agnostic filter: any instance with

    L_TSP / L_MST  <  METRIC_RATIO_LOWER
    L_TSP / L_MST  >  METRIC_RATIO_UPPER

is dropped from aggregate metrics. The instance is still downloaded,
benchmarked, and reported in per-instance output so that the non-metric
behavior is auditable; only the headline averages exclude it.

Design notes
------------
* The filter is applied at analysis time, not at download or benchmark
  time. Every instance in TSPLIB95 is run end-to-end so the per-instance
  CSV contains full results for brg180 (or any future outlier).
* The constant lives here so the paper can cite a single source of
  truth. Do not hard-code the threshold in call sites; import it.
* ``TRIANGLE_INEQ_VIOLATORS`` is retained (empty) for backwards
  compatibility with scripts that import it; new code should use
  ``filter_metric_consistent`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# -- Public constants ------------------------------------------------------

# Optimal-tour / MST ratio bounds for metric TSP. For any symmetric
# metric instance, L_MST <= L_TSP <= 2 * L_MST (double-tree). Instances
# whose published optimum falls outside [LOWER, UPPER] are non-metric.
# UPPER carries a 25% slack above the theoretical 2.0 ceiling to absorb
# integer rounding in EXPLICIT matrices. LOWER is exactly 1.0 — any
# ratio below that means the "optimum" is shorter than the MST, which
# is impossible in a metric space (e.g. brg180: ratio 0.44).
METRIC_RATIO_LOWER: float = 1.0
METRIC_RATIO_UPPER: float = 2.5
# Retained for import backwards-compat; points at UPPER.
METRIC_RATIO_THRESHOLD: float = METRIC_RATIO_UPPER

# Retained for backwards compatibility. With the ratio filter in place
# there is no static exclusion list; every instance runs and the
# principled filter handles aggregates.
TRIANGLE_INEQ_VIOLATORS: frozenset[str] = frozenset()


# -- Public helpers --------------------------------------------------------


def is_metric_consistent(true_cost: float, mst_length: float) -> bool:
    """Return True iff ``METRIC_RATIO_LOWER <= true_cost / mst_length <= METRIC_RATIO_UPPER``.

    If ``mst_length`` is non-positive, returns True (the ratio is
    undefined, so no evidence of non-metricity).
    """
    if mst_length is None or mst_length <= 0:
        return True
    ratio = true_cost / mst_length
    return METRIC_RATIO_LOWER <= ratio <= METRIC_RATIO_UPPER


def filter_metric_consistent(
    df: "pd.DataFrame",
    true_col: str = "true_cost",
    mst_col: str = "mst_length",
) -> "pd.DataFrame":
    """Return a copy of ``df`` with non-metric rows removed.

    A row is kept iff its optimal-tour / MST ratio is <= the threshold
    (or iff the required columns are missing, in which case the filter
    is a no-op).
    """
    if true_col not in df.columns or mst_col not in df.columns:
        return df.copy()
    ratio = df[true_col] / df[mst_col]
    mask = (
        ratio.notna()
        & (df[mst_col] > 0)
        & (ratio >= METRIC_RATIO_LOWER)
        & (ratio <= METRIC_RATIO_UPPER)
    )
    # Rows where ratio cannot be computed are kept (filter is a no-op there).
    mask = mask | df[mst_col].isna() | (df[mst_col] <= 0)
    return df[mask].copy()


def is_excluded(instance_name: str) -> bool:
    """Legacy name-based check. Always False now that filtering is ratio-based."""
    return instance_name in TRIANGLE_INEQ_VIOLATORS
