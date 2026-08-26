"""Global degeneracy and pile-up detectors for TSP point clouds.

Motivation
----------
The 30 production features summarise a cloud with a bounding box, a node
density, a 2-axis aspect ratio and MST edge statistics. None of them notices
that a cloud has *collapsed*: that its variance lives in far fewer than ``d``
directions, or that a large mass of points shares one coordinate value because
a generator clipped out-of-box samples onto a face instead of resampling them.

This module measures both kinds of collapse:

* **Spectral degeneracy** - the shape of the full PCA eigenvalue spectrum
  (all ``d`` eigenvalues, not the two that ``aspect_ratio`` uses), plus the
  per-axis shape moments and extreme-band occupancy in the PCA frame.
* **Coordinate-value degeneracy** - axis-aligned pile-ups, detected from the
  multiplicity distribution of coordinate values on each axis, with no
  hard-coded knowledge of a ``[0, G]`` box.

Frames and invariance
---------------------
Features named ``degeneracy_pca_*`` are computed in the cloud's own principal
-axis frame, which this module derives internally from ``coords``, so they are
invariant to the frame the caller supplies. The eight spectral features are
exactly rotation invariant. The six per-axis features
(``abs_skew_*``, ``kurtosis_*``, ``extreme_frac_*``) inherit PCA's one
unavoidable weakness: when two eigenvalues are *exactly* tied the eigenvectors
spanning that eigenspace are arbitrary, so the per-axis marginals are not
uniquely defined. Measured worst case on an exact square lattice
(lambda_2 / lambda_1 = 1.0000): kurtosis moves by 0.41 and extreme_frac by
0.09 under a random rotation. At lambda_2 / lambda_1 = 0.956 and below the
movement is 0.0 to float precision, so real clouds are unaffected.

The five coordinate-value features (``degeneracy_tied_point_fraction``,
``degeneracy_max_tie_mass``, ``degeneracy_max_axis_pileup``,
``degeneracy_mean_axis_pileup``, ``degeneracy_min_distinct_ratio``) are
**deliberately rotation sensitive**: an axis-aligned pile-up only exists
relative to some set of axes. They must be fed the *raw generator frame*.
Passing PCA-canonicalised coordinates (as ``feature_creator_v3`` does before
its own feature block) destroys exact coordinate ties and collapses all five
to their no-degeneracy values. A caller that wants both groups should call
``compute`` on the raw coordinates; the PCA rotation is applied internally.

Complexity
----------
``O(n * d^2 + d^3)`` for the covariance and its eigendecomposition, plus
``O(d * n log n)`` for the per-axis sorts. No convex hull, no solver, no
randomness. ``mst_csr`` is accepted for contract uniformity and ignored.

Scale invariance
----------------
Every feature is scale invariant. Spectral features are eigenvalue ratios;
skewness and kurtosis are standardised moments; tie tolerance and the extreme
band are fractions of each axis's own range. Multiplying ``coords`` by any
positive constant leaves all 19 outputs unchanged.
"""

from __future__ import annotations

import numpy as np

__all__ = ["feature_names", "compute"]


# Coordinate values on one axis are treated as tied when they differ by less
# than this fraction of that axis's range. Relative (hence scale invariant)
# and ~4 orders of magnitude above float64 round-off on a rotated cloud, yet
# ~6 orders below the spacing of an integer grid of size 1000.
_TIE_REL_TOL = 1e-9

# Half-width of the "extreme" band at each end of a PCA axis, as a fraction
# of that axis's range. A uniform axis puts 2 * 0.02 = 4% of its mass in it.
_EXTREME_BAND = 0.02

# Excess kurtosis is bounded below by -2 (Pearson: kurtosis >= skew^2 + 1).
# A fully collapsed axis has undefined moments; -2 is the saturating value.
_MIN_EXCESS_KURTOSIS = -2.0

_FEATURE_NAMES = [
    # (a) effective rank / participation ratio of the full PCA spectrum
    "degeneracy_pca_effective_rank",
    "degeneracy_pca_effective_rank_norm",
    "degeneracy_pca_participation_ratio",
    "degeneracy_pca_participation_ratio_norm",
    # (b) normalised eigenvalue-decay profile
    "degeneracy_pca_var_frac_1",
    "degeneracy_pca_var_frac_2",
    "degeneracy_pca_var_frac_3",
    "degeneracy_pca_spectral_entropy",
    # (c) coordinate-value degeneracy (raw frame; rotation sensitive)
    "degeneracy_tied_point_fraction",
    "degeneracy_max_tie_mass",
    "degeneracy_max_axis_pileup",
    "degeneracy_mean_axis_pileup",
    "degeneracy_min_distinct_ratio",
    # (d) per-axis shape moments in the PCA frame
    "degeneracy_pca_abs_skew_max",
    "degeneracy_pca_abs_skew_mean",
    "degeneracy_pca_kurtosis_min",
    "degeneracy_pca_kurtosis_max",
    # (e) extreme-band occupancy in the PCA frame
    "degeneracy_pca_extreme_frac_max",
    "degeneracy_pca_extreme_frac_mean",
]


def feature_names() -> list[str]:
    """Stable, ordered names of the 19 features this module produces."""
    return list(_FEATURE_NAMES)


def compute(coords: np.ndarray, mst_csr=None) -> dict[str, float]:
    """Compute the degeneracy feature group for one point cloud.

    Parameters
    ----------
    coords : (n, d) array_like
        Point coordinates. Pass the **raw generator frame**: the PCA rotation
        needed by the ``degeneracy_pca_*`` features is applied internally,
        while the coordinate-value features need un-rotated axes to see
        axis-aligned pile-ups.
    mst_csr : optional
        Accepted for contract uniformity with the sibling feature groups and
        ignored - this group needs no MST.

    Returns
    -------
    dict[str, float]
        Keyed exactly by :func:`feature_names`. All values are finite.

    Degenerate-input policy
    -----------------------
    ``n < 3``, an empty array, or a cloud with zero total variance returns the
    "maximally collapsed" sentinel set from :func:`_collapsed_defaults`:
    effective rank and participation ratio 1.0 (one occupied direction),
    every cumulative variance fraction 1.0, spectral entropy 0.0, both
    pile-up features 1.0, ``min_distinct_ratio`` 1.0, skewness 0.0, both
    kurtosis features -2.0, and extreme fractions 1.0. Tie features are still
    measured from the data when ``n >= 1`` so that ``n = 2`` duplicates are
    not silently reported as untied.
    A single zero-variance PCA axis contributes skew 0.0 and excess kurtosis
    -2.0 and an extreme fraction of 1.0 rather than a division by zero.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(-1, 1) if arr.size else np.empty((0, 1), dtype=np.float64)
    n, d = arr.shape
    d = max(int(d), 1)

    if n < 3 or not np.all(np.isfinite(arr)):
        out = _collapsed_defaults(d)
        if n >= 1 and np.all(np.isfinite(arr)):
            out.update(_coordinate_degeneracy(arr))
        return out

    out: dict[str, float] = {}
    out.update(_coordinate_degeneracy(arr))

    centred = arr - arr.mean(axis=0)
    # eigh on the (d, d) covariance: O(n d^2 + d^3), symmetric, deterministic.
    cov = np.atleast_2d(np.cov(centred, rowvar=False, bias=True))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    total = float(eigvals.sum())

    if total <= 0.0:
        # Every point identical (n >= 3 duplicates). Spectrum is undefined;
        # fall back to the collapsed sentinels but keep the measured ties.
        collapsed = _collapsed_defaults(d)
        collapsed.update(out)
        return collapsed

    out.update(_spectral_shape(eigvals, total, d))

    rotated = centred @ eigvecs[:, order]
    out.update(_pca_frame_shape(rotated))

    return {name: float(out[name]) for name in _FEATURE_NAMES}


# --------------------------------------------------------------------------
# (a) + (b) spectrum shape
# --------------------------------------------------------------------------

def _spectral_shape(eigvals: np.ndarray, total: float, d: int) -> dict[str, float]:
    """Effective rank, participation ratio and decay profile of the spectrum.

    ``p_i = lambda_i / sum(lambda)``.

    * effective rank  = exp(-sum p_i log p_i)      (Roy & Vetterli), in [1, d]
    * participation   = 1 / sum(p_i^2)                                in [1, d]
    * var_frac_k      = cumulative variance in the leading k components
    * spectral entropy= Shannon entropy of p, divided by log(d), in [0, 1]

    Both ``*_norm`` variants divide by ``d`` so the value is comparable across
    the d = 2..100 range the paper claims. For ``d == 1`` the normalisations
    are 1.0 and the entropy is 0.0 (log(1) = 0 would divide by zero).
    """
    p = eigvals / total
    nz = p[p > 0.0]
    entropy = float(-np.sum(nz * np.log(nz)))

    eff_rank = float(np.exp(entropy))
    part_ratio = float(1.0 / np.sum(p * p))

    # Numerical guard: both live in [1, d] analytically.
    eff_rank = min(max(eff_rank, 1.0), float(d))
    part_ratio = min(max(part_ratio, 1.0), float(d))

    cum = np.cumsum(p)
    return {
        "degeneracy_pca_effective_rank": eff_rank,
        "degeneracy_pca_effective_rank_norm": eff_rank / d,
        "degeneracy_pca_participation_ratio": part_ratio,
        "degeneracy_pca_participation_ratio_norm": part_ratio / d,
        "degeneracy_pca_var_frac_1": float(min(cum[0], 1.0)),
        "degeneracy_pca_var_frac_2": float(min(cum[min(1, d - 1)], 1.0)),
        "degeneracy_pca_var_frac_3": float(min(cum[min(2, d - 1)], 1.0)),
        "degeneracy_pca_spectral_entropy": (
            0.0 if d < 2 else float(min(max(entropy / np.log(d), 0.0), 1.0))
        ),
    }


# --------------------------------------------------------------------------
# (c) coordinate-value degeneracy
# --------------------------------------------------------------------------

def _coordinate_degeneracy(arr: np.ndarray) -> dict[str, float]:
    """Axis-aligned pile-up detectors, from coordinate-value multiplicities.

    For each axis the sorted values are grouped by single linkage with a
    tolerance of ``_TIE_REL_TOL`` times that axis's range, which makes the
    grouping scale invariant and immune to float representation noise without
    ever merging two distinct integer lattice values.

    From the group multiplicities ``m_1..m_k`` (summing to n):

    ``tied_point_fraction``
        Fraction of points that share a value with at least one other point
        on at least one axis (union over axes).
    ``max_tie_mass``
        ``max over axes of max_i m_i / n`` - the largest single pile-up.
    ``max_axis_pileup`` / ``mean_axis_pileup``
        ``1 - (N_eff - 1) / (k - 1)`` where ``N_eff = n^2 / sum(m_i^2)`` is
        the effective number of occupied values. This is 0 when the occupied
        values carry equal mass (a lattice, or uniform integers) and → 1 when
        one value hoards the mass (a clipped face). It is the pile-up
        detector that survives the birthday-paradox ties an integer grid
        produces by chance, which ``tied_point_fraction`` and ``max_tie_mass``
        do not. ``k == 1`` (whole axis collapsed) is defined as 1.0.
    ``min_distinct_ratio``
        ``min over axes of k / n``. 1.0 for continuous coordinates, ~1/sqrt(n)
        for a square lattice, 1/n for a collapsed axis.
    """
    n, d = arr.shape
    if n == 0:
        return {
            "degeneracy_tied_point_fraction": 1.0,
            "degeneracy_max_tie_mass": 1.0,
            "degeneracy_max_axis_pileup": 1.0,
            "degeneracy_mean_axis_pileup": 1.0,
            "degeneracy_min_distinct_ratio": 1.0,
        }

    tied_any = np.zeros(n, dtype=bool)
    max_tie_mass = 0.0
    min_distinct_ratio = 1.0
    pileups = np.empty(d, dtype=np.float64)

    for j in range(d):
        col = arr[:, j]
        order = np.argsort(col, kind="stable")
        sorted_col = col[order]
        span = float(sorted_col[-1] - sorted_col[0])

        if span <= 0.0:
            # Whole axis collapsed onto one value.
            tied_any[:] = True
            max_tie_mass = 1.0
            min_distinct_ratio = min(min_distinct_ratio, 1.0 / n)
            pileups[j] = 1.0
            continue

        tol = _TIE_REL_TOL * span
        breaks = np.diff(sorted_col) > tol
        group_id = np.concatenate(([0], np.cumsum(breaks)))
        counts = np.bincount(group_id)
        k = counts.size

        sizes_sorted = counts[group_id]
        sizes = np.empty(n, dtype=counts.dtype)
        sizes[order] = sizes_sorted
        tied_any |= sizes >= 2

        max_tie_mass = max(max_tie_mass, float(counts.max()) / n)
        min_distinct_ratio = min(min_distinct_ratio, k / n)

        if k > 1:
            n_eff = (float(n) ** 2) / float(np.sum(counts.astype(np.float64) ** 2))
            pileups[j] = min(max(1.0 - (n_eff - 1.0) / (k - 1.0), 0.0), 1.0)
        else:
            pileups[j] = 1.0

    return {
        "degeneracy_tied_point_fraction": float(tied_any.mean()),
        "degeneracy_max_tie_mass": float(max_tie_mass),
        "degeneracy_max_axis_pileup": float(pileups.max()),
        "degeneracy_mean_axis_pileup": float(pileups.mean()),
        "degeneracy_min_distinct_ratio": float(min_distinct_ratio),
    }


# --------------------------------------------------------------------------
# (d) + (e) per-axis shape in the PCA frame
# --------------------------------------------------------------------------

def _pca_frame_shape(rotated: np.ndarray) -> dict[str, float]:
    """Standardised moments and extreme-band occupancy per principal axis.

    Skewness is reported as ``|skew|`` because the sign of a PCA eigenvector
    is arbitrary; excess kurtosis is already sign invariant. A zero-variance
    axis gets skew 0.0, excess kurtosis ``_MIN_EXCESS_KURTOSIS`` (the
    saturating "maximally degenerate" value) and extreme fraction 1.0.

    ``kurtosis_min`` is the pile-up channel: mass parked at both ends of an
    axis drives excess kurtosis towards -2, well below the -1.2 of a uniform
    axis. ``extreme_frac_*`` counts points within ``_EXTREME_BAND`` of either
    end of the axis range, so a uniform axis scores ~0.04.
    """
    n, d = rotated.shape
    sd = rotated.std(axis=0)
    live = sd > 0.0

    abs_skew = np.zeros(d, dtype=np.float64)
    excess_kurt = np.full(d, _MIN_EXCESS_KURTOSIS, dtype=np.float64)
    extreme_frac = np.ones(d, dtype=np.float64)

    if np.any(live):
        z = rotated[:, live] / sd[live]
        z2 = z * z
        abs_skew[live] = np.abs((z2 * z).mean(axis=0))
        excess_kurt[live] = np.maximum(
            (z2 * z2).mean(axis=0) - 3.0, _MIN_EXCESS_KURTOSIS
        )

        sub = rotated[:, live]
        lo = sub.min(axis=0)
        hi = sub.max(axis=0)
        band = _EXTREME_BAND * (hi - lo)
        in_band = (sub <= lo + band) | (sub >= hi - band)
        extreme_frac[live] = in_band.sum(axis=0) / n

    return {
        "degeneracy_pca_abs_skew_max": float(abs_skew.max()),
        "degeneracy_pca_abs_skew_mean": float(abs_skew.mean()),
        "degeneracy_pca_kurtosis_min": float(excess_kurt.min()),
        "degeneracy_pca_kurtosis_max": float(excess_kurt.max()),
        "degeneracy_pca_extreme_frac_max": float(extreme_frac.max()),
        "degeneracy_pca_extreme_frac_mean": float(extreme_frac.mean()),
    }


# --------------------------------------------------------------------------
# sentinels
# --------------------------------------------------------------------------

def _collapsed_defaults(d: int) -> dict[str, float]:
    """Finite sentinels for input too degenerate to describe (see ``compute``)."""
    return {
        "degeneracy_pca_effective_rank": 1.0,
        "degeneracy_pca_effective_rank_norm": 1.0 / d,
        "degeneracy_pca_participation_ratio": 1.0,
        "degeneracy_pca_participation_ratio_norm": 1.0 / d,
        "degeneracy_pca_var_frac_1": 1.0,
        "degeneracy_pca_var_frac_2": 1.0,
        "degeneracy_pca_var_frac_3": 1.0,
        "degeneracy_pca_spectral_entropy": 0.0,
        "degeneracy_tied_point_fraction": 1.0,
        "degeneracy_max_tie_mass": 1.0,
        "degeneracy_max_axis_pileup": 1.0,
        "degeneracy_mean_axis_pileup": 1.0,
        "degeneracy_min_distinct_ratio": 1.0,
        "degeneracy_pca_abs_skew_max": 0.0,
        "degeneracy_pca_abs_skew_mean": 0.0,
        "degeneracy_pca_kurtosis_min": _MIN_EXCESS_KURTOSIS,
        "degeneracy_pca_kurtosis_max": _MIN_EXCESS_KURTOSIS,
        "degeneracy_pca_extreme_frac_max": 1.0,
        "degeneracy_pca_extreme_frac_mean": 1.0,
    }
