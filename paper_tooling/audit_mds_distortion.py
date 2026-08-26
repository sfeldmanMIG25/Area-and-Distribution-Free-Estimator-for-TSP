"""Recompute embedding diagnostics for the paper's screened TSPLIB set."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


ROOT = Path(__file__).resolve().parents[1]
TSPLIB_DIR = ROOT / "tsplib_benchmark"
sys.path.insert(0, str(TSPLIB_DIR))

from classical_mds import classical_mds  # noqa: E402
from exclusions import STRUCTURAL_TRIANGLE_VIOLATORS  # noqa: E402
from tsplib_parser import parse_tsplib_file  # noqa: E402


# The canonical results file, not the derived ``*_repaired.csv``.  This audit
# needs nothing but the instance list and its distance label, both of which are
# instance-level facts, so depending on a derived file only added a way for the
# screen to go stale behind the manuscript.
RESULTS = TSPLIB_DIR / "results" / "all_models_tsplib.csv"
INSTANCES = TSPLIB_DIR / "instances"
OUTPUT = ROOT / "paper_reference" / "mds_distortion_screened.csv"
TYPES = {"CEIL_2D", "ATT", "GEO", "EXPLICIT"}


def distance_diagnostics(original: np.ndarray, embedded: np.ndarray) -> tuple[float, float]:
    upper = np.triu_indices_from(original, 1)
    observed = original[upper].astype(float)
    reconstructed = squareform(pdist(embedded))[upper]
    normalized_stress = float(
        np.sqrt(np.sum((observed - reconstructed) ** 2) / np.sum(observed ** 2))
    )
    correlation = float(np.corrcoef(observed, reconstructed)[0, 1])
    return normalized_stress, correlation


def ceil_distance_diagnostics(coords: np.ndarray) -> tuple[float, float]:
    """Estimate CEIL_2D rounding distortion without an O(n^2) matrix."""
    rng = np.random.default_rng(42)
    pair_count = min(1_000_000, max(100_000, len(coords) * 100))
    left = rng.integers(0, len(coords), size=pair_count)
    right = rng.integers(0, len(coords), size=pair_count)
    keep = left != right
    reconstructed = np.linalg.norm(coords[left[keep]] - coords[right[keep]], axis=1)
    observed = np.ceil(reconstructed)
    normalized_stress = float(
        np.sqrt(np.sum((observed - reconstructed) ** 2) / np.sum(observed ** 2))
    )
    correlation = float(np.corrcoef(observed, reconstructed)[0, 1])
    return normalized_stress, correlation


def main() -> None:
    results = pd.read_csv(RESULTS, low_memory=False)
    # Screen over instances, not over one model's rows.  The previous version
    # keyed on a hardcoded ``model == "LGBM_V3"`` purely to get one row per
    # instance, which silently tied this audit to whichever model happened to
    # be production when it was written.  ``edge_weight_type`` is constant per
    # instance, so de-duplicating on the pair is roster-independent.
    screen = results.loc[
        results["edge_weight_type"].isin(TYPES)
        & ~results["instance"].isin(STRUCTURAL_TRIANGLE_VIOLATORS),
        ["instance", "edge_weight_type"],
    ].drop_duplicates()
    if screen["instance"].duplicated().any():
        conflicting = sorted(screen.loc[screen["instance"].duplicated(), "instance"])
        raise ValueError(f"instances with conflicting edge_weight_type: {conflicting}")
    if len(screen) != 23:
        raise ValueError(f"expected 23 screened instances, found {len(screen)}")

    rows: list[dict[str, object]] = []
    for instance, edge_type in screen.itertuples(index=False, name=None):
        info = parse_tsplib_file(INSTANCES / f"{instance}.tsp")
        if edge_type == "CEIL_2D":
            embedded = info["raw_coords"].astype(float)
            chosen_dim = natural_dim = 2
            retained = 1.0
            negative_ratio = 0.0
            strain = 0.0
            mode = "native coordinates; sampled CEIL rounding diagnostic"
            stress, correlation = ceil_distance_diagnostics(embedded)
        else:
            original = info["distance_matrix"].astype(float)
            embedded, positive, mds = classical_mds(
                original, max_dim=100, variance_threshold=0.999
            )
            chosen_dim = int(mds["chosen_dim"])
            natural_dim = int(mds["natural_dim"])
            retained = float(mds["variance_retained"])
            negative_ratio = float(mds["negative_eigvalue_mass"] / positive.sum())
            strain = float(mds["strain"])
            mode = "positive-eigenvalue MDS"
            stress, correlation = distance_diagnostics(original, embedded)
        rows.append({
            "instance": instance,
            "edge_weight_type": edge_type,
            "mode": mode,
            "chosen_dim": chosen_dim,
            "natural_positive_dim": natural_dim,
            "positive_mass_retained": retained,
            "threshold_met": retained >= 0.999 - 1e-12,
            "negative_to_positive_eigenvalue_mass": negative_ratio,
            "classical_strain": strain,
            "normalized_distance_stress": stress,
            "distance_correlation": correlation,
        })

    output = pd.DataFrame(rows).sort_values(["edge_weight_type", "instance"])
    output.to_csv(OUTPUT, index=False)
    print(output.groupby("edge_weight_type")[
        ["positive_mass_retained", "normalized_distance_stress", "distance_correlation"]
    ].agg(["count", "median", "max", "min"]).to_string())
    print(f"threshold not met: {(~output['threshold_met']).sum()}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
