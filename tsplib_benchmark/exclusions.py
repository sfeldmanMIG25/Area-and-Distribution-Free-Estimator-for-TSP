"""Shared audited screen for metric-dependent TSPLIB aggregates.

Motivation
----------
GART 2.0 and the MDS-embedding pipeline both assume symmetric metric
distances. For any metric TSP, the double-tree construction proves

    L_MST  <=  L_TSP  <=  2 * L_MST.

The paper's non-Euclidean experiment uses an exhaustive distance-matrix
audit to distinguish structural from mild rounding-level triangle
violations. ``STRUCTURAL_TRIANGLE_VIOLATORS`` is the shared list used by
the paper tooling and aggregate helpers. Three disclosed mild violators
remain in the exploratory screen.

The tour/MST bounds below provide an additional model-agnostic guard: an
instance with

    L_TSP / L_MST  <  METRIC_RATIO_LOWER
    L_TSP / L_MST  >  METRIC_RATIO_UPPER

is also dropped from metric-dependent aggregates. The upper guard retains
25% slack for integer-rounded EXPLICIT matrices. All instances remain in
per-instance benchmark output so the screen is auditable.

Design notes
------------
* The filter is applied at analysis time, not at download or benchmark
  time. Every instance in TSPLIB95 is run end-to-end so the per-instance
  CSV contains full results for brg180 (or any future outlier).
* Names and ratio guards live here so analysis scripts use one protocol.
* ``TRIANGLE_INEQ_VIOLATORS`` aliases the structural list for backwards
  compatibility.
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

# Exhaustive distance-matrix audit reported in the paper. These violations
# are structural (worst direct-edge ratio at least 1.2), not small rounding
# artifacts.
STRUCTURAL_TRIANGLE_VIOLATORS: frozenset[str] = frozenset({
    "bays29",
    "brazil58",
    "brg180",
    "dantzig42",
    "gr17",
    "gr21",
    "gr24",
    "gr48",
    "gr120",
    "hk48",
})

# Retained in the exploratory 23-instance screen and disclosed explicitly.
MILD_TRIANGLE_VIOLATORS: frozenset[str] = frozenset({"fri26", "pa561", "swiss42"})

TRIANGLE_INEQ_VIOLATORS = STRUCTURAL_TRIANGLE_VIOLATORS


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
    instance_col: str = "instance",
) -> "pd.DataFrame":
    """Return a copy using the shared structural-name and ratio screen.

    The structural list is applied whenever ``instance_col`` is present.
    The tour/MST guard is additionally applied when both numeric columns
    are present.
    """
    keep = None
    if instance_col in df.columns:
        names = df[instance_col].astype(str).str.lower().str.replace(r"\.tsp$", "", regex=True)
        keep = ~names.isin(STRUCTURAL_TRIANGLE_VIOLATORS)
    if true_col not in df.columns or mst_col not in df.columns:
        return df.loc[keep].copy() if keep is not None else df.copy()
    ratio = df[true_col] / df[mst_col]
    ratio_keep = (
        ratio.notna()
        & (df[mst_col] > 0)
        & (ratio >= METRIC_RATIO_LOWER)
        & (ratio <= METRIC_RATIO_UPPER)
    )
    # Rows where ratio cannot be computed are kept (filter is a no-op there).
    ratio_keep = ratio_keep | df[mst_col].isna() | (df[mst_col] <= 0)
    keep = ratio_keep if keep is None else keep & ratio_keep
    return df.loc[keep].copy()


def is_excluded(instance_name: str) -> bool:
    """Return whether ``instance_name`` is in the structural audit list."""
    normalized = instance_name.lower()
    if normalized.endswith(".tsp"):
        normalized = normalized[:-4]
    return normalized in STRUCTURAL_TRIANGLE_VIOLATORS
