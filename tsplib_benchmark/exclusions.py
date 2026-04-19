"""Single source of truth for TSPLIB instances excluded from the pipeline.

Instances listed here violate the triangle inequality (metric TSP
assumption) and must be filtered out at every pipeline stage: download,
benchmark runners, analysis, and aggregate reporting. GART 2.0 and the
MDS embedding assume symmetric metric distances; non-metric matrices
produce ill-defined MSTs and degenerate embeddings.

Currently only ``brg180`` ("Bridges") is known to violate. Add new
instances here and every downstream component will pick up the change.
"""

from __future__ import annotations

# Instances whose distance matrix violates the triangle inequality.
TRIANGLE_INEQ_VIOLATORS: frozenset[str] = frozenset({"brg180"})


def is_excluded(instance_name: str) -> bool:
    """Return True if ``instance_name`` must be excluded from the pipeline."""
    return instance_name in TRIANGLE_INEQ_VIOLATORS
