"""Region-aware reimplementations of the classical tour-length estimators.

Why this module exists
----------------------
The classical estimators in :mod:`tsp_utils_2` were evaluated with the convex
hull of the realised sample substituted for the quantity their sources actually
require: the measure of the *sampling region*. That substitution is what made
them look uncompetitive, and it is why the manuscript excluded them. On the
synthetic benchmarks the sampling region is known exactly — every generator
draws coordinates inside ``[0, G]^d`` — so the substitution is unnecessary and
the estimators can be evaluated on their own terms.

Each estimator here therefore takes an explicit region measure and records how
that measure was obtained, so a reader can tell a source-faithful evaluation
from a documented plug-in:

``sampling_region``
    ``G^d``, the exact support of the generator. Available on both synthetic
    benchmarks. This is the input the sources assume.
``bounding_box``
    The product of the realised coordinate ranges. Used for TSPLIB, which
    defines no sampling region. Stated as a plug-in, never as the source form.
``convex_hull``
    The convex hull measure of the realised sample. Retained only as a
    sensitivity check against ``bounding_box``.

Every estimator also declares the domain its source covers. Callers gate on
:meth:`domain_status` and record a status row instead of extrapolating a fitted
formula past its calibration range.

What is not here, and why
-------------------------
Daganzo (1984a, 1984b), Chien (1992) and Kwon--Golden--Wasil (1995) were scored
by earlier revisions of this module and have been removed. Their DOIs resolve to
paywalled articles with no open-access location, so the coefficients we had were
transcribed from a secondary source (the literature review in Figliozzi 2008),
which the removed docstrings said outright. Secondary renderings disagree with
each other -- Cavdar's own table gives Chien a Daganzo-form coefficient of 0.69
against the 0.67 our Figliozzi-derived form implies, and renders Kwon's bracket
with different coefficients again -- so there is no way to pick one without
reading a primary we cannot obtain. We do not benchmark against constants we
have not read in the original. The manuscript still surveys these works as prior
art; it prints no number for them.

``Cavdar_region``, which fed :class:`CavdarSokol` the generator support ``G^2``,
is removed for a different reason: Cavdar defines ``A`` as the covering
rectangle of the *nodes*, and there is no sampling-region concept anywhere in
that work. Supplying ``G^2`` was our invention rather than a source-faithful
variant, and it was the one configuration in which the source's own Eq. (21)
correction made the result worse.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tsp_utils_2 import BETA_2D, BETA_3D, BETA_MST_2D, get_mst_length

RegionSource = Literal["sampling_region", "bounding_box", "convex_hull"]

# Asymptotic TSP constants for i.i.d. uniform points are imported above: beta_2
# is the Johnson-McGeoch-Rothberg (1996) estimate, beta_3 is Percus-Martin
# (1996). For d >= 4 no comparably pinned point estimate exists, so beta_tsp
# falls back to the large-d asymptotic sqrt(d / (2 pi e)); at n <= 1000 and
# d >= 4 the sample is nowhere near the asymptotic regime in any case.

# Convex-hull plug-ins are computed only up to this dimension. QHull's output
# size grows as n^floor(d/2), which is the exponential blow-up this paper cites
# as a reason not to build estimators on hull geometry: at d = 8 with n = 1000 a
# single hull can take minutes, and across 16,920 instances the run does not
# terminate in useful time. Above the cap we use the bounding box and record
# ``region_source='bounding_box'`` so the substitution is visible in the output.
# tsp_utils_2.CONVEX_HULL_MAX_DIM stays at its legacy value for other callers.
HULL_MAX_DIM = 3


def beta_tsp(d: int) -> float:
    """Return the BHH constant for dimension ``d``."""
    if d == 2:
        return BETA_2D
    if d == 3:
        return BETA_3D
    return math.sqrt(d / (2.0 * math.pi * math.e))


@dataclass(frozen=True)
class Region:
    """The measure of the region the points are treated as filling.

    ``measure`` is an area in 2D and a volume in ``d`` dimensions. ``root`` is
    ``measure**(1/d)`` computed in log space, because ``G^d`` overflows float64
    by ``d ~ 100`` at ``G = 10000``.
    """

    root: float
    source: RegionSource
    d: int

    @property
    def measure(self) -> float:
        """The region measure, or ``inf`` when it overflows float64."""
        log_m = self.d * math.log(self.root) if self.root > 0 else -math.inf
        return math.exp(log_m) if log_m < 709.0 else math.inf


def region_from_grid(grid_size: float, d: int) -> Region:
    """Exact sampling region for a generator drawing on ``[0, G]^d``."""
    if not (grid_size > 0):
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    return Region(root=float(grid_size), source="sampling_region", d=d)


def region_from_points(coords: np.ndarray, prefer_hull: bool = False) -> Region:
    """Plug-in region measure for instances with no defined sampling region."""
    coords = np.asarray(coords, dtype=np.float64)
    d = coords.shape[1]
    ranges = np.ptp(coords, axis=0).astype(float)
    ranges[ranges < 1e-9] = 1e-9
    if prefer_hull and len(coords) > d + 1 and d <= HULL_MAX_DIM:
        from scipy.spatial import ConvexHull

        try:
            vol = float(ConvexHull(coords).volume)
            if vol > 0:
                return Region(root=vol ** (1.0 / d), source="convex_hull", d=d)
        except Exception:
            pass
    # Geometric mean of the ranges == (prod ranges)**(1/d), stable at any d.
    return Region(root=float(np.exp(np.mean(np.log(ranges)))), source="bounding_box", d=d)


def resolve_region(coords: np.ndarray, d: int, grid_size: float | None,
                   prefer_hull: bool = False) -> Region:
    """Use the exact sampling region when the caller knows it, else plug in."""
    if grid_size is not None and grid_size > 0:
        return region_from_grid(grid_size, d)
    return region_from_points(coords, prefer_hull=prefer_hull)


class RegionEstimator:
    """Base class: an estimator that consumes a region measure.

    ``region_mode`` fixes, in advance, which measure the estimator receives:

    ``'sampling'``
        The generator's support ``[0, G]^d`` when the caller supplies ``G``.
        This is the source-faithful input, and it is the right one exactly when
        the generator is uniform on that support.
    ``'hull'``
        The convex hull of the realised sample. This is the standard practical
        plug-in and the only option when no sampling region exists, as on
        TSPLIB. It is also the fairer input on the degenerate classes: a
        near-collinear point set has a one-dimensional effective support, so
        handing an area-based estimator ``G^2`` inflates its prediction by an
        amount that says nothing about the estimator.
    ``'bbox'``
        The product of the realised coordinate ranges. Also the mode used by an
        estimator that derives its own frame and consumes no region at all.

    Where a source supports both forms they are registered as separate models
    and reported separately. The choice is made from a property of the data,
    never from the results.

    Subclasses implement :meth:`predict`. :meth:`estimate` is the interface the
    benchmark runners call, and it returns the runner's result dict plus the
    region provenance and the domain status.
    """

    name: str = "region_estimator"

    def __init__(self, region_mode: str = "hull"):
        if region_mode not in ("sampling", "hull", "bbox"):
            raise ValueError(f"unknown region_mode {region_mode!r}")
        self.region_mode = region_mode

    def _region(self, coords: np.ndarray, d: int, grid_size: float | None) -> Region:
        if self.region_mode == "sampling":
            return resolve_region(coords, d, grid_size, prefer_hull=False)
        return region_from_points(coords, prefer_hull=(self.region_mode == "hull"))

    def domain_status(self, coords: np.ndarray, n: int, d: int, region: Region) -> str:
        """Return ``'ok'`` or a status string naming why the source does not apply."""
        return "ok"

    def predict(self, coords: np.ndarray, n: int, d: int, region: Region) -> float:
        raise NotImplementedError

    def estimate(self, coordinates, dimension, grid_size=None) -> dict:
        t0 = time.perf_counter()
        coords = np.unique(np.asarray(coordinates, dtype=np.float64), axis=0)
        n = len(coords)
        d = int(dimension) if dimension else coords.shape[1]
        if n <= 1:
            return {"estimate": 0.0, "feature_time": 0.0, "inference_time": 0.0,
                    "region_source": "none", "status": "degenerate_n"}
        region = self._region(coords, d, grid_size)
        status = self.domain_status(coords, n, d, region)
        if status != "ok":
            return {"estimate": float("nan"), "feature_time": time.perf_counter() - t0,
                    "inference_time": 0.0, "region_source": region.source, "status": status}
        t1 = time.perf_counter()
        est = self.predict(coords, n, d, region)
        return {"estimate": float(est), "feature_time": t1 - t0,
                "inference_time": time.perf_counter() - t1,
                "region_source": region.source, "status": "ok"}


class BHH(RegionEstimator):
    """Beardwood--Halton--Hammersley asymptotic tour length.

    For ``n`` points drawn i.i.d. uniformly on a region of measure ``V`` in
    ``R^d``, ``L_n / n^{(d-1)/d} -> beta_d * V^{1/d}`` almost surely
    (Beardwood, Halton & Hammersley 1959). We evaluate

        L_hat = beta_d * n^{(d-1)/d} * V^{1/d}

    with ``V`` the *sampling region* measure, which is the quantity the theorem
    names. Substituting the convex hull of the realised sample is a different
    estimator and is reported separately.

    The theorem is an asymptotic statement for uniform i.i.d. points, so its
    matched domain is the uniform generator class at large ``n``. We still
    report it everywhere, because the size of its error off that domain is the
    point: it measures how much a density-and-region assumption costs.
    """

    name = "BHH"

    def domain_status(self, coords: np.ndarray, n: int, d: int, region: Region) -> str:
        return "ok"

    def predict(self, coords: np.ndarray, n: int, d: int, region: Region) -> float:
        return beta_tsp(d) * (n ** ((d - 1) / d)) * region.root


class MSTOnly(RegionEstimator):
    """``L_hat = L_MST``. The alpha = 1 null the R^2_alpha metric is measured against.

    This is the weakest possible MST-informed estimator and the correct floor
    for any claim that a learned multiplier adds value.
    """

    name = "MST_Only"

    def predict(self, coords: np.ndarray, n: int, d: int, region: Region) -> float:
        mst_len, _ = get_mst_length(coords)
        return float(mst_len)


def _min_area_rect_frame(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a planar point set into its minimum-area enclosing rectangle.

    Returns ``(rotated_coords, sides)`` where ``sides`` are the two side lengths
    of that rectangle, so ``sides[0] * sides[1]`` is its area.

    Cavdar prescribes rotating the graph so that "the ratio of the area of the
    convex hull to the area of the smallest rectangle covering all the nodes and
    lying parallel to the x-axis is maximum". Hull area is invariant under
    rotation, so maximising that ratio is exactly minimising the axis-aligned
    bounding rectangle -- i.e. finding the minimum-area enclosing rectangle.
    Freeman & Shapira (1975), which Cavdar cites for this, prove that rectangle
    has a side collinear with a hull edge, so scanning the hull edges is exact
    rather than a search over angles.
    """
    pts = np.asarray(coords, dtype=np.float64)
    try:
        from scipy.spatial import ConvexHull

        hull_pts = pts[ConvexHull(pts).vertices]
    except Exception:
        hull_pts = pts
    if len(hull_pts) < 2:
        return pts, np.ptp(pts, axis=0)

    best_area = math.inf
    best: tuple[np.ndarray, np.ndarray] | None = None
    edges = np.roll(hull_pts, -1, axis=0) - hull_pts
    for ex, ey in edges:
        norm = math.hypot(float(ex), float(ey))
        if norm < 1e-12:
            continue
        c, s = ex / norm, ey / norm
        # Rotate so this hull edge lies along +x.
        rot = np.array([[c, s], [-s, c]], dtype=np.float64)
        rp = pts @ rot.T
        sides = np.ptp(rp, axis=0)
        area = float(sides[0] * sides[1])
        if area < best_area:
            best_area, best = area, (rp, sides)
    if best is None:
        return pts, np.ptp(pts, axis=0)
    return best


class CavdarSokol(RegionEstimator):
    """Cavdar & Sokol's distribution-free estimator for a rectangular graph.

    Published model (dissertation Eq. 20 == EJOR Eq. 3):

        T ~ 2.791 * sqrt(n * cstdev_x * cstdev_y)
          + 0.2669 * sqrt(n * stdev_x * stdev_y * A / (cbar_x * cbar_y))

    with the authors' own notation (dissertation p. 66):

    ``stdev_j``   standard deviation of the coordinates on axis ``j``.
    ``cbar_j``    average distance of the nodes to the central axis of the
                  rectangle on axis ``j`` (the authors write ``c_x``, ``c_y``).
    ``cstdev_j``  standard deviation of those absolute distances.
    ``A``         area of the graph.

    Fit statistics: R^2 = 0.9956; coefficient 2.791 has SE 0.02714 and 0.2669
    has SE 0.003508 (dissertation Table 5).

    **Small-n correction (dissertation Eq. 21 == EJOR Eq. 4).** The model was
    trained on ``n`` in {3000, 4000, 5000, 5200, ..., 6600, 7000, 8000} and
    underestimates below ``n ~ 1000``. The authors fit the ratio of estimate to
    tour length,

        E/T = 0.9325 * exp(0.00005298 n) - 0.2972 * exp(-0.01452 n),

    with R^2 = 0.9867, and correct by dividing the raw estimate by it. That fit
    was made on ``n`` in {100, 125, ..., 975}, so we bound it to that range:

    * ``n < 100``  -- outside the fit. The authors' Fig. 54 shows E/T falling
      toward 0.4 as ``n`` approaches 10, but no fitted form is published there,
      so we evaluate the ratio at the lower endpoint ``n = 100`` rather than
      extrapolate. Flagged in the ``correction`` field of the result.
    * ``100 <= n <= 975`` -- the fitted range; applied as published.
    * ``n > 975`` -- no correction. The authors state that past roughly 1000
      nodes "the estimation is proper and stationary". Extrapolating Eq. 21
      upward is actively wrong: it grows without bound (E/T = 1.22 at n = 5000),
      which would contradict the training fit it is supposed to repair. This
      leaves a 1.8% step at the boundary, where the fitted ratio is 0.982.

    **Frame.** The model is defined on an axis-aligned rectangle, so the
    per-axis statistics are frame-dependent, and ``A`` is the area of the
    rectangle covering the nodes -- a statistic of the sample, not of any
    sampling region. We therefore apply the authors' own rule for
    non-rectangular graphs on every instance and rotate into the minimum-area
    enclosing rectangle before taking any statistic (see
    :func:`_min_area_rect_frame`), which also makes the estimator rotation
    invariant. ``region`` is not consumed at all; the estimator is registered in
    ``bbox`` mode only so that no convex hull is built twice.

    **Two caveats that are properties of the source, not of our evaluation.**
    The regression targets Lin-Kernighan tour lengths, not optimal tours, so it
    inherits LK's excess over optimal (the authors put LK within 1% and note the
    correction is a multiplicative rescale). And every training graph carries a
    node at each of the four corners, which pins the graph area to the covering
    rectangle; our instances do not.

    Primary source read:
      Cavdar, B. (2014). "A computation-implementation parallelization approach
      to time-sensitive applications." PhD dissertation, Georgia Institute of
      Technology, H. Milton Stewart School of Industrial and Systems
      Engineering (advisor J. Sokol). Chapter 4 is the tour-length estimation
      model. https://repository.gatech.edu/handle/1853/52322
    Published as:
      Cavdar, B., Sokol, J. (2015). "A distribution-free TSP tour length
      estimation model for random graphs." European Journal of Operational
      Research 243(2):588-598. https://doi.org/10.1016/j.ejor.2014.12.020

    RESOLVED 2026-08-12. This docstring and references.bib had previously
    disagreed about which repository record names the dissertation. The record
    above is the correct one: handle 1853/52322 resolves to that title, and the
    Georgia Tech catalogue attributes it to Cavdar, 2014, ISyE, advisor Sokol,
    with the tour-length estimation work in it. The competing entry that
    references.bib carried -- "Distribution-free approaches for the traveling
    salesman problem and the vehicle routing problem" at handle 1853/53506 --
    corresponds to no record that could be found and has been deleted from the
    bibliography. The equations, the constants and the Eq. (20) / Eq. (21)
    numbering implemented below were always read off the document itself; only
    the citation metadata was in doubt, and citing an unverified record is the
    same defect that removed Daganzo, Chien and Kwon from this module.
    """

    name = "Cavdar"
    A0, A1 = 2.791, 0.2669
    # Eq. 21 and the n-grid it was fitted on.
    CORR_A, CORR_B = 0.9325, 0.00005298
    CORR_C, CORR_D = 0.2972, 0.01452
    CORR_N_MIN, CORR_N_MAX = 100, 975

    def __init__(self, region_mode: str = "bbox", apply_correction: bool = True):
        super().__init__(region_mode)
        self.apply_correction = apply_correction

    @classmethod
    def correction_ratio(cls, n: int) -> float:
        """Published E/T ratio (Eq. 21), evaluated only inside its fitted range."""
        n_eff = min(max(n, cls.CORR_N_MIN), cls.CORR_N_MAX)
        return (cls.CORR_A * math.exp(cls.CORR_B * n_eff)
                - cls.CORR_C * math.exp(-cls.CORR_D * n_eff))

    def domain_status(self, coords: np.ndarray, n: int, d: int, region: Region) -> str:
        # The published model is defined per-axis on a planar rectangle. There is
        # no d-dimensional form in the source, so we do not invent one.
        return "ok" if d == 2 else "cavdar_not_2d"

    def predict(self, coords: np.ndarray, n: int, d: int, region: Region) -> float:
        # ``region`` is deliberately unused. Cavdar's A is the covering rectangle
        # of the NODES; the source has no sampling-region concept, so feeding it
        # the generator support G^2 would be our construction and not the
        # published model. An earlier ``Cavdar_region`` row did exactly that and
        # has been withdrawn.
        pts, sides = _min_area_rect_frame(coords)
        midpoint = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
        area = float(max(sides[0] * sides[1], 1e-12))

        stdev = np.clip(pts.std(axis=0).astype(np.float64), 1e-12, None)
        abs_dev = np.abs(pts - midpoint).astype(np.float64)
        c_bar = np.clip(abs_dev.mean(axis=0), 1e-12, None)
        cstdev = np.clip(abs_dev.std(axis=0), 1e-12, None)

        term1 = self.A0 * math.sqrt(n * cstdev[0] * cstdev[1])
        term2 = self.A1 * math.sqrt(
            n * stdev[0] * stdev[1] * area / (c_bar[0] * c_bar[1]))
        raw = term1 + term2
        if not self.apply_correction or n > self.CORR_N_MAX:
            return raw
        return raw / self.correction_ratio(n)


def asymptotic_mst_ratio_2d() -> float:
    """Published constant ratio ``beta_TSP / beta_MST`` for uniform planar points."""
    return BETA_2D / BETA_MST_2D


# Registry. ``BHH`` uses the convex-hull plug-in and is the row reported on the
# full benchmarks, where the generator density is not uniform on the sampling
# region. ``BHH_region`` receives the exact sampling region and is reported on
# the matched-domain (uniform generator) subset, where that is the quantity the
# BHH theorem names.
#
# ``Cavdar`` has one row and no ``_region`` twin: its area is a statistic of the
# node set in the source, so there is no second, region-fed form to report.
#
# The ``Daganzo``, ``Chien`` and ``Kwon`` keys, and their ``_region`` /
# ``_extrap`` twins, are gone. See the module docstring: their coefficients came
# from a secondary transcription and the primaries are not obtainable, so no row
# of ours may depend on them. Re-adding a key here without a primary source in
# hand reintroduces exactly the defect this removal exists to fix.
ESTIMATORS: dict[str, RegionEstimator] = {
    "BHH": BHH("hull"),
    "BHH_region": BHH("sampling"),
    "Cavdar": CavdarSokol("bbox"),
    "MST_Only": MSTOnly(),
}
