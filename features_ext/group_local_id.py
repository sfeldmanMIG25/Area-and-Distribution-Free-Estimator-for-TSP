"""Local intrinsic-dimensionality features (group ``local_id``).

Motivation
----------
Every feature in the production GART 2.0 extractor is *global*: bounding box,
node density, aspect ratio, centroid-distance moments and MST aggregates. None
of them can see that a point cloud is a *union* of locally one-dimensional
pieces. That is exactly the structure of the ``line_noise`` class (thin
segments plus large collinear masses pinned to a box face by coordinate
clipping) and, in the opposite direction, of the ``grid`` class (locally a
perfect 2-D lattice, which is *less* locally-1-D than an isotropic sample).

This module measures the local dimensionality of the cloud directly, in a way
that is generator-agnostic: it only ever looks at k-nearest-neighbour patches
and at inter-point distance ratios, so it fires the same way on a genuine
collinear cloud, on a polyline, and on an edge mass.

Design constraints honoured
---------------------------
* **Arbitrary d (2..100).** Local PCA never touches a d x d matrix when d is
  large: the patch spectrum is obtained from the (k+1) x (k+1) Gram matrix,
  whose non-zero eigenvalues are exactly those of the patch covariance. Per
  point the eigen cost is therefore O(min(k+1, d)^3), independent of d.
* **Complexity.** Neighbour search is a single scipy ``cKDTree`` query,
  O(n log n) in the low-d regime and O(n^2 d) in the fully degenerate worst
  case -- the same budget the dense MST already pays. Measured against a
  chunked BLAS brute force it is faster at every size the project uses, so
  there is no second backend. No convex hull, no tour, no solver.
* **Scale invariance.** Every feature is a ratio of eigenvalues, a ratio of
  distances, or a log-ratio of distances. All 30 are exactly scale invariant.
* **Rotation / translation invariance.** Every feature is a function of the
  pairwise distance structure alone, so all 30 are invariant to rigid motions
  (and to reflections). None is intended to be orientation sensitive; this
  group is deliberately complementary to the PCA-frame global features.
* **Determinism.** No randomness anywhere. Reference rows for the correlation
  integral are chosen by a fixed stride; neighbour ties are broken by a stable
  lexicographic sort on (exact squared distance, point index).
* **No target leakage.** Only ``coords`` is read. ``mst_csr`` is accepted for
  contract uniformity and is *not used*.

Degenerate-input policy
-----------------------
Every feature in this group is an intrinsic-dimension-like quantity whose
minimum meaningful value is 1 (a line). Wherever the estimand is undefined the
module returns the sentinel ``1.0``, which reads as "maximally degenerate /
locally one-dimensional". Concretely:

* ``n < 3``, non-finite coordinates, or ``d < 1`` -> *all* features are 1.0.
  A set of fewer than three points spans at most a line, so 1.0 is also the
  correct answer rather than merely a safe one.
* A k-NN patch whose centred scatter has zero trace (the point and all its
  neighbours coincide) contributes ``evr1 = evr2 = pr = 1.0`` and counts as
  one-dimensional in ``frac_1d``. The test is exact: the trace is a sum of
  squares and is zero only for a coincident patch.
* TwoNN and Levina-Bickel skip points whose first-neighbour distance is zero
  (duplicates) and points whose ``mu = r2/r1`` is within 1e-9 of 1 (exact
  ties, e.g. a perfect lattice or a perfectly equispaced chain), because
  ``log(mu)`` is then zero or pure floating-point noise and the estimator
  diverges. If fewer than three points survive, the estimator returns 1.0.
  The tie phenomenon itself is still reported, in bounded form, by
  ``local_id_frac_mu_lt_1p05``.
* The correlation-dimension features return 1.0 if the median pairwise
  distance is zero (all points coincide) or if the two radii cannot be
  separated.
* Unbounded estimators (TwoNN, Levina-Bickel, correlation dimension and their
  ratio) are clamped to ``[0.0, 100.0]``. 100 is the largest dimension the
  paper considers, so the clamp can only bind on pathological input.

When ``n <= k``, the k-th neighbourhood silently falls back to all ``n - 1``
available neighbours, so the ``k5`` / ``k10`` / ``k20`` blocks coincide on
tiny instances.

``compute`` de-duplicates and lexicographically sorts its input (a single
``np.unique(..., axis=0)``) before reading it. This is required for
correctness, not merely tidiness: neighbour ties are broken on point index, so
without a canonical index the features are a function of the caller's row
order rather than of the point set. Measured on the unfixed module, permuting
the rows of an unchanged 9-point cloud moved a feature by 0.75.

The returned dict is guaranteed to be keyed exactly by ``feature_names()`` and
to contain only finite floats.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["feature_names", "compute"]

# --- configuration ---------------------------------------------------------

_K_VALUES = (5, 10, 20)          # neighbourhood sizes for the local-PCA block
_KMAX = max(_K_VALUES)
_LB_K_VALUES = (10, 20)          # neighbourhood sizes for Levina-Bickel MLE

_ONE_D_THRESHOLD = 0.95          # evr1 above this => patch is "essentially 1-D"
_MU_TIE = 1.05                   # r2/r1 below this => first two NNs are tied
_MU_EPS = 1e-9                   # mu within this of 1 is treated as an exact tie
_ID_MAX = 100.0                  # clamp for unbounded ID estimators
_DEGENERATE = 1.0                # sentinel, see module docstring

_MAX_REF_ROWS = 256              # rows sampled for the correlation integral
_MAX_REF_CELLS = 2_000_000       # and a hard cap on sampled pair count
_PATCH_CHUNK_BYTES = 32 << 20    # memory cap for the (chunk, k+1, d) patches


def feature_names() -> list[str]:
    """Stable, ordered names for the 30 features in this group."""
    names: list[str] = []
    for k in _K_VALUES:
        names.extend(
            [
                f"local_id_evr1_mean_k{k}",
                f"local_id_evr1_median_k{k}",
                f"local_id_evr1_p10_k{k}",
                f"local_id_evr1_p90_k{k}",
                f"local_id_evr2_mean_k{k}",
                f"local_id_pr_mean_k{k}",
                f"local_id_frac_1d_k{k}",
            ]
        )
    names.extend(
        [
            "local_id_twonn",
            "local_id_twonn_mle",
            "local_id_mu_median",
            "local_id_frac_mu_lt_1p05",
            "local_id_corr_dim_lo",
            "local_id_corr_dim_mid",
            "local_id_corr_dim_ratio",
        ]
    )
    names.extend([f"local_id_lb_mle_k{k}" for k in _LB_K_VALUES])
    return names


# --- small helpers ---------------------------------------------------------


def _clamp_id(value: float) -> float:
    """Clamp an intrinsic-dimension estimate into [0, _ID_MAX]; NaN -> sentinel."""
    if not np.isfinite(value):
        return _DEGENERATE
    return float(min(max(value, 0.0), _ID_MAX))


def _sentinel_dict() -> dict[str, float]:
    return {name: _DEGENERATE for name in feature_names()}


def _reference_rows(n: int) -> np.ndarray:
    """Deterministic stride-selected row sample for the correlation integral."""
    budget = max(16, _MAX_REF_CELLS // max(n, 1))
    n_ref = int(min(_MAX_REF_ROWS, n, budget))
    if n_ref >= n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, n_ref).astype(np.int64))


def _neighbour_indices(coords: np.ndarray, m: int) -> np.ndarray:
    """(n, m) neighbour indices, self included, ordered by increasing distance.

    A single backend (``scipy.spatial.cKDTree``) is used at every dimension.
    Measured on this machine at n=1000 it costs 1.9 ms at d=2 and 6.5 ms at
    d=100, against 15 ms for a chunked BLAS brute force at either dimension;
    the two only converge at n=20000, d=100 (3.9 s vs 3.7 s). A second backend
    would therefore buy no speed while introducing a second, slightly
    different answer on tied input -- which of several *equidistant* points
    lands inside the k=5 prefix is genuinely ambiguous, and two implementations
    resolve it differently. One backend keeps the feature single-valued.

    Ties within the returned set are then re-ordered lexicographically on
    (distance, point index) so the prefix used for k=5 and k=10 does not
    depend on scipy's internal traversal order. That runs on the (n, m) index
    array, never on the (n, m, d) patch tensor, so it is free.

    Worst case is O(n^2 d) (fully degenerate tree), inside the stated budget.
    """
    n = coords.shape[0]
    if m >= n:
        return np.broadcast_to(np.arange(n, dtype=np.int64), (n, n)).copy()
    dist, sel = cKDTree(coords).query(coords, k=m, workers=-1)
    sel = sel.astype(np.int64)
    order = np.lexsort((sel, dist), axis=-1)
    return np.ascontiguousarray(np.take_along_axis(sel, order, axis=1))


def _top2_eigen(mat: np.ndarray) -> np.ndarray:
    """Clipped, descending eigenvalues of a batch of symmetric PSD matrices.

    Closed form for 1x1 and 2x2 (which is the whole 2-D benchmark, and avoids
    ~11 ms of batched LAPACK calls per instance); ``eigvalsh`` otherwise.
    """
    size = mat.shape[-1]
    if size == 1:
        lam = mat[:, 0, :1].copy()
    elif size == 2:
        tr = mat[:, 0, 0] + mat[:, 1, 1]
        det = mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 0, 1] * mat[:, 1, 0]
        disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
        lam = np.column_stack([0.5 * (tr + disc), 0.5 * (tr - disc)])
    else:
        lam = np.linalg.eigvalsh(mat)[:, ::-1]
    return np.maximum(lam, 0.0)


def _patch_spectrum(offs: np.ndarray, gram: np.ndarray | None, mk: int, d: int):
    """Per-point (evr1, evr2, participation ratio) for one neighbourhood size.

    ``offs`` is (c, m, d) query-centred patch offsets sorted by distance;
    ``gram`` is the matching (c, m, m) raw Gram or None. Whichever of the
    d x d scatter and the (mk x mk) Gram is smaller gets decomposed - their
    non-zero spectra are identical.
    """
    if d <= mk or gram is None:
        patch = offs[:, :mk, :]
        centred = patch - patch.mean(axis=1, keepdims=True)
        mat = np.swapaxes(centred, 1, 2) @ centred
    else:
        sub = gram[:, :mk, :mk]
        # Double-centring turns the query-centred Gram into the patch-mean
        # centred one. Working from query-centred offsets keeps the magnitudes
        # at neighbour-spacing scale, so there is no cancellation here.
        mat = sub - sub.mean(axis=2, keepdims=True) - sub.mean(axis=1, keepdims=True)
        mat += sub.mean(axis=(1, 2), keepdims=True)

    lam = _top2_eigen(mat)
    total = lam.sum(axis=1)
    good = total > 0.0
    safe = np.where(good, total, 1.0)

    evr1 = np.where(good, lam[:, 0] / safe, _DEGENERATE)
    if lam.shape[1] >= 2:
        evr2 = np.where(good, (lam[:, 0] + lam[:, 1]) / safe, _DEGENERATE)
    else:
        evr2 = np.full(lam.shape[0], _DEGENERATE)
    sumsq = np.einsum("ij,ij->i", lam, lam)
    pr = np.where(good & (sumsq > 0.0), safe**2 / np.where(sumsq > 0.0, sumsq, 1.0), _DEGENERATE)

    rank_cap = float(max(1, min(mk - 1, d)))
    return (
        np.clip(evr1, 0.0, 1.0),
        np.clip(evr2, 0.0, 1.0),
        np.clip(pr, 1.0, rank_cap),
    )


def _local_spectra(coords: np.ndarray):
    """Chunked pass producing exact k-NN distances and local PCA summaries.

    Returns ``(nn_dist, spectra)`` where ``nn_dist`` is (n, m) exact distances
    (column 0 is the point itself at distance 0, ascending afterwards) and
    ``spectra[k]`` is a tuple of (n,) arrays ``(evr1, evr2, pr)``.
    """
    n, d = coords.shape
    m = int(min(_KMAX + 1, n))
    cand = _neighbour_indices(coords, m)

    sizes = {k: int(min(k + 1, m)) for k in _K_VALUES}
    need_gram = any(d > mk for mk in sizes.values())
    chunk = int(max(1, min(n, _PATCH_CHUNK_BYTES // max(8 * m * d, 1))))

    nn_dist = np.empty((n, m), dtype=np.float64)
    spectra = {k: tuple(np.empty(n, dtype=np.float64) for _ in range(3)) for k in _K_VALUES}

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sel = cand[start:stop]
        # Offsets are already distance-ordered by _neighbour_indices. The
        # radii are recomputed here from coordinate differences rather than
        # taken from the backend, so they are exact and non-negative even for
        # coincident points; the extra sort only repairs ULP-level ties.
        offs = coords[sel] - coords[start:stop, None, :]
        exact = np.einsum("ijk,ijk->ij", offs, offs)
        nn_dist[start:stop] = np.sqrt(np.sort(exact, axis=1))

        gram = offs @ np.swapaxes(offs, 1, 2) if need_gram else None
        for k in _K_VALUES:
            evr1, evr2, pr = _patch_spectrum(offs, gram, sizes[k], d)
            spectra[k][0][start:stop] = evr1
            spectra[k][1][start:stop] = evr2
            spectra[k][2][start:stop] = pr

    return nn_dist, spectra


# --- estimators ------------------------------------------------------------


def _twonn(mu: np.ndarray) -> tuple[float, float]:
    """TwoNN intrinsic dimension (Facco et al. 2017) from ``mu = r2 / r1``.

    Returns (linear fit through the origin with the top 10% of mu discarded,
    plain maximum-likelihood estimate ``N / sum(log mu)``).
    """
    mu = mu[np.isfinite(mu) & (mu > 1.0 + _MU_EPS)]
    if mu.size < 3:
        return _DEGENERATE, _DEGENERATE

    log_sum = float(np.log(mu).sum())
    mle = _clamp_id(mu.size / log_sum) if log_sum > 0.0 else _DEGENERATE

    mu_sorted = np.sort(mu, kind="stable")
    n_mu = mu_sorted.size
    keep = int(np.floor(0.9 * n_mu))
    if keep < 2:
        return _DEGENERATE, mle
    # Empirical CDF F_i = i / n_mu for i = 1..keep; keep < n_mu keeps F_i < 1.
    f_emp = np.arange(1, keep + 1, dtype=np.float64) / n_mu
    x = np.log(mu_sorted[:keep])
    y = -np.log1p(-f_emp)
    denom = float(np.dot(x, x))
    fit = _clamp_id(float(np.dot(x, y)) / denom) if denom > 0.0 else _DEGENERATE
    return fit, mle


def _levina_bickel(nn_dist: np.ndarray, k: int) -> float:
    """Levina-Bickel maximum-likelihood ID at neighbourhood size k, averaged in
    the MacKay-Ghahramani (inverse-mean) form."""
    kk = int(min(k, nn_dist.shape[1] - 1))
    if kk < 2:
        return _DEGENERATE
    radii = nn_dist[:, 1 : kk + 1]
    valid = radii[:, 0] > 0.0
    if not np.any(valid):
        return _DEGENERATE
    radii = radii[valid]
    inv = np.mean(np.log(radii[:, -1:] / radii[:, :-1]), axis=1)
    inv = inv[np.isfinite(inv) & (inv > 0.0)]
    if inv.size == 0:
        return _DEGENERATE
    return _clamp_id(1.0 / float(inv.mean()))


def _correlation_dims(coords: np.ndarray) -> tuple[float, float, float]:
    """Correlation-integral slopes ``d log C(r) / d log r`` at two radius bands.

    The band edges are anchored on the median sampled pairwise distance, so
    they carry the instance's own length scale and the slopes come out scale
    invariant. Everything is done on squared distances (no sqrt) and with
    ``partition`` + counting rather than a full sort.
    """
    n = coords.shape[0]
    rows = _reference_rows(n)
    sq = np.einsum("ij,ij->i", coords, coords)
    block = coords[rows] @ coords.T
    block *= -2.0
    block += sq[None, :]
    block += sq[rows][:, None]
    np.maximum(block, 0.0, out=block)
    block[np.arange(len(rows)), rows] = np.inf     # drop self pairs

    flat = block.reshape(-1)
    n_pairs = flat.size - len(rows)
    if n_pairs < 8:
        return _DEGENERATE, _DEGENERATE, _DEGENERATE

    kth = (n_pairs - 1) // 2
    r2_hi = float(np.partition(flat, kth)[kth])
    if not np.isfinite(r2_hi) or r2_hi <= 0.0:
        return _DEGENERATE, _DEGENERATE, _DEGENERATE

    floor = 0.5 / n_pairs

    def integral(r2: float) -> float:
        return max(float(np.count_nonzero(flat < r2)) / n_pairs, floor)

    c_hi = integral(r2_hi)
    c_lo = integral(r2_hi / 16.0)      # radius r_hi / 4
    c_lolo = integral(r2_hi / 64.0)    # radius r_hi / 8
    if c_hi <= floor:
        return _DEGENERATE, _DEGENERATE, _DEGENERATE

    dim_mid = _clamp_id(np.log(c_hi / c_lo) / np.log(4.0))
    dim_lo = _clamp_id(np.log(c_lo / c_lolo) / np.log(2.0))
    ratio = _clamp_id(dim_lo / dim_mid) if dim_mid > 1e-9 else _DEGENERATE
    return dim_lo, dim_mid, ratio


# --- public entry point ----------------------------------------------------


def compute(coords: np.ndarray, mst_csr=None) -> dict[str, float]:
    """Compute the ``local_id`` feature group.

    Parameters
    ----------
    coords : (n, d) array
        Point coordinates in whatever frame the caller supplies. Every feature
        here is invariant to rigid motions and to uniform scaling, so the frame
        does not matter.
    mst_csr : ignored
        Accepted for uniformity with the other feature-group modules.

    Returns
    -------
    dict[str, float]
        Keyed exactly by ``feature_names()``; all values finite.
    """
    del mst_csr  # this group is purely geometric

    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 1 or not np.all(np.isfinite(arr)):
        return _sentinel_dict()

    # Canonicalise the point SET before anything reads it.
    #
    # ``_neighbour_indices`` breaks k-NN ties on point index. That is only a
    # valid tie-break if the index is itself a function of the point set, so
    # the caller's row order must not survive to this point. Without this,
    # permuting the rows of an unchanged cloud moved features by up to 0.75 at
    # n = 9. Duplicate rows are also collapsed: two coincident points make the
    # k = 5 prefix ambiguous and contribute a zero-radius neighbour that every
    # estimator below has to special-case anyway.
    #
    # This was previously left to the caller. The canonical extractor did do
    # it, so corpus values are unaffected -- but a contract that is only
    # honoured by convention is not a contract.
    arr = np.unique(arr, axis=0)
    if arr.shape[0] < 3:
        return _sentinel_dict()
    arr = np.ascontiguousarray(arr)

    nn_dist, spectra = _local_spectra(arr)

    out: dict[str, float] = {}
    for k in _K_VALUES:
        evr1, evr2, pr = spectra[k]
        out[f"local_id_evr1_mean_k{k}"] = float(evr1.mean())
        out[f"local_id_evr1_median_k{k}"] = float(np.median(evr1))
        out[f"local_id_evr1_p10_k{k}"] = float(np.quantile(evr1, 0.10))
        out[f"local_id_evr1_p90_k{k}"] = float(np.quantile(evr1, 0.90))
        out[f"local_id_evr2_mean_k{k}"] = float(evr2.mean())
        out[f"local_id_pr_mean_k{k}"] = float(pr.mean())
        out[f"local_id_frac_1d_k{k}"] = float(np.mean(evr1 > _ONE_D_THRESHOLD))

    r1 = nn_dist[:, 1]
    r2 = nn_dist[:, 2] if nn_dist.shape[1] > 2 else r1
    positive = r1 > 0.0
    if np.any(positive):
        mu = r2[positive] / r1[positive]
        out["local_id_mu_median"] = float(np.median(mu))
        out["local_id_frac_mu_lt_1p05"] = float(np.mean(mu < _MU_TIE))
        fit, mle = _twonn(mu)
    else:
        out["local_id_mu_median"] = _DEGENERATE
        out["local_id_frac_mu_lt_1p05"] = _DEGENERATE
        fit, mle = _DEGENERATE, _DEGENERATE
    out["local_id_twonn"] = fit
    out["local_id_twonn_mle"] = mle

    dim_lo, dim_mid, ratio = _correlation_dims(arr)
    out["local_id_corr_dim_lo"] = dim_lo
    out["local_id_corr_dim_mid"] = dim_mid
    out["local_id_corr_dim_ratio"] = ratio

    for k in _LB_K_VALUES:
        out[f"local_id_lb_mle_k{k}"] = _levina_bickel(nn_dist, k)

    # Contract guard: exact key set, finite values only.
    for name in feature_names():
        value = float(out.get(name, _DEGENERATE))
        out[name] = value if np.isfinite(value) else _DEGENERATE
    return {name: out[name] for name in feature_names()}
