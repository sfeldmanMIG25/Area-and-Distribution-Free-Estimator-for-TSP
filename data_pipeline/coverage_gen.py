"""Alpha-coverage corpus: instances whose alpha is placed, not hoped for.

The defect
----------
The training split's alpha support collapses as n grows.  Measured on
``tsp_features_v4.csv`` (split == 'train', alpha = optimal_cost / mst_total_length):

    n range      count   min     median   max      p99
    (0,   10]    17442   1.1226  1.2609   1.9817   1.7249
    (10,  20]     2907   1.0679  1.1271   1.6370   1.3685
    (20,  50]     8721   1.0403  1.0984   1.5720   1.2797
    (50, 100]    14535   1.0315  1.0836   1.3852   1.2237
    (100,200]     2907   1.0328  1.0736   1.2291   1.1754
    (200,500]     8721   1.0084  1.0700   1.2068   1.1604
    (500,1e3]    14535   1.0000  1.0669   1.1890   1.1433

Zero training rows with alpha > 1.5 at n >= 200; all 212 rows above 1.7 have
n <= 10.  alpha is bounded in [1, 2] by construction, so the corpus occupies
roughly the bottom quarter of the range wherever n is large enough to matter.
The cause is structural: the ND corpus is a per-axis product of four 1-D laws
(uniform, clipped normal, clustered, correlated), every axis on the same
[0, G] scale.  No product of such laws can be a thin extended set at large n,
so no product of such laws can have alpha near 2 at large n.

The mechanism this module exploits
----------------------------------
Let the points lie on a 1-D curve, traversed in order p_1 .. p_n, with

    L = sum_i |p_{i+1} - p_i|        (chain length)
    D = |p_n - p_1|                  (endpoint separation)

When the transverse thickness is small compared with the spacing L/n, the MST
is that chain and the optimal tour is the chain plus the closing chord:

    MST -> L,    OPT -> L + D,    alpha -> 1 + D/L  ==  1 + kappa.

kappa = D/L is a pure shape number in [0, 1]: 1 for a straight line, 0 for a
closed curve.  It is computable from the coordinates with no solver, so a
target alpha can be *placed* by bisecting a family's shape parameter until
kappa hits alpha_target - 1.  That is what ``solve_kappa`` does, and it is why
this corpus can be uniform in alpha at every n instead of piling up at 2.

Two further knobs stop the corpus from being one generator wearing hats:

  * ``rho`` -- transverse thickness in units of the along-curve spacing.  As
    rho grows past ~1 the set stops reading as 1-D and alpha falls away from
    1 + kappa towards the planar prior.  A second, independent route to the
    middle of the range.
  * ``mix`` -- a fraction of the points drawn uniformly in the box instead of
    on the skeleton.  A third route, and one that does not look degenerate.

Distinctness from every evaluation generator (checked in code, not by name)
---------------------------------------------------------------------------
``paper_tooling/decontaminated_arm_protocol.md`` records that a previous
augmentation shipped a d=2 ``lattice`` family that was line-for-line the 2D
benchmark's ``grid`` generator.  The rule taken from that: an added generator
must be provably distinct from every generator used for evaluation.

Evaluation generators, enumerated from source:

  * ``data_pipeline/d2_benchmark_gen.py::DIST_MAP`` -- random, normal,
    triangular, squeezed_uniform, uniform_triangular, triangular_squeezed,
    boundary, x_central, truncated_exponential, clustered, grid, correlated.
    Eleven are i.i.d. per-point densities on the box; ``grid`` is a jittered
    square lattice; ``clustered`` is disc blobs.
  * ``data_pipeline/extend_line_noise.py::generate_line_noise`` -- one straight
    line of slope U(0.2, 5.0), x drawn uniformly, Gaussian offset along a
    single fixed perpendicular at sigma = 0.02 G, box-clipped.
  * ``data_pipeline/instance_io.py::DISTRIBUTION_MAP_1D`` -- the ND corpus:
    independent per-axis uniform / clipped-normal / clustered / correlated.

Every family below is a *curved or branched 1-D skeleton with a controlled
closure ratio*.  None of the fourteen produces one.  Concretely:

  * no lattice is generated here at any d, so the ``grid`` identity cannot recur;
  * ``arc`` is a circular arc of sweep theta >= 0.5 rad.  The sweep floor is
    load-bearing: it keeps the sagitta at >= 3% of the chord, so no instance in
    this family is a straight line, which is what ``line_noise`` is.  alpha up
    to 1.99 is still reachable there because kappa = 2 sin(theta/2)/theta is
    0.9896 at theta = 0.5;
  * ``fourier`` is a band-limited random space curve, ``serpentine`` a
    boustrophedon path, ``spider`` a k-armed star.  No evaluation generator has
    a skeleton at all;
  * transverse displacement is perpendicular to the LOCAL tangent of a curve,
    over the full (d-1)-dimensional orthogonal complement, and is offered in
    two profiles (Gaussian and uniform-in-ball).  ``line_noise`` offsets along
    one global perpendicular of a straight line.  The protocol also records
    that matching thickness alone is not a match at all -- a Gaussian band
    spans ~1.7x a clipped-uniform one at equal RMS -- so thickness here is a
    diversity knob and never a coverage claim.  Coverage is claimed on the
    achieved alpha histogram and nowhere else.

Solver path
-----------
``augment_gen.solve_with_fallback`` (Concorde first, LKH on timeout) with
``get_scale_factor`` -- the grid-only rule that produced the ND corpus.
``verification.solve_instance_robust`` and its ``get_robust_scale_factor``
damping are the defect that corrupted 15,016 labels and are not reachable from
here.  Every label passes the four exact integrity gates of
``augment_gen.verify_solution`` plus a Held--Karp 1-tree lower bound:
``bound <= optimal_cost`` is a hard gate, and n <= 20 is solved exactly by DP
inside ``solvers.concorde.run_concorde``.

Usage
-----
    python -m data_pipeline.coverage_gen --pilot
    python -m data_pipeline.coverage_gen --plan main --workers 16
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import math
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_pipeline.augment_gen import (
    _dedupe, _orthonormal_basis, _transverse_noise,
    measured_rho, transverse_kurtosis, verify_solution,
)
from held_karp_1tree import one_tree_bound_from_matrix
from mst_utils import mst_length
from solvers import tsplib_io
from solvers.concorde import _read_tour, _wsl_path
from solvers.config import CONCORDE_WSL_BIN, SOLVER_SCRATCH_DIR, get_scale_factor
from solvers.distance import compute_distance_matrix, compute_tour_length_numba
from solvers.exact import EXACT_N_MAX, held_karp
from solvers.lkh import run_lkh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NOT "coverage/": a top-level directory of that name becomes a namespace
# package on sys.path and shadows the real `coverage` distribution, which numba
# imports at load time. Doing so breaks every numba import in the repo.
COV_ROOT = os.path.join(REPO_ROOT, "alpha_coverage")
COV_INSTANCES_DIR = os.path.join(COV_ROOT, "instances")
COV_SOLUTIONS_DIR = os.path.join(COV_ROOT, "solutions")

NAME_PREFIX = "COV_"

# Directories a COV_ name must never collide with, and which must never end up
# holding a COV_ name. instances/ and Generalized_TSP_Analysis/instances/ are
# the ND corpus and the 2D benchmark; augment/ is the earlier arm.
GUARDED_DIRS = (
    os.path.join(REPO_ROOT, "instances"),
    os.path.join(REPO_ROOT, "solutions"),
    os.path.join(REPO_ROOT, "Generalized_TSP_Analysis", "instances"),
    os.path.join(REPO_ROOT, "Generalized_TSP_Analysis", "solutions"),
    os.path.join(REPO_ROOT, "augment", "instances"),
    os.path.join(REPO_ROOT, "augment", "solutions"),
)

# Held-Karp ascent budget for the label gate. Any budget yields a valid lower
# bound; more iterations tighten it and so make ``bound <= label`` a sharper
# test. The schedule keeps the O(k n^2) cost bounded at large n.
def hk_iterations_for(n: int) -> int:
    if n <= 300:
        return 300
    if n <= 700:
        return 150
    return 80

BOX_FILL = 0.90          # skeleton extent as a fraction of the grid
ARC_THETA_MIN = 0.5      # radians; see the module docstring -- never a line

# How far a family may miss its requested alpha bin before ``arc`` takes over.
ALPHA_PLACEMENT_TOL = 0.03


# ---------------------------------------------------------------------------
# Shape number
# ---------------------------------------------------------------------------

def predict_alpha(skeleton: np.ndarray) -> float:
    """Solver-free prediction of alpha for a thin set on this skeleton.

        alpha_pred = (L + D) / MST,   L = chain length in generation order,
                                      D = |last - first|, MST = MST of the set.

    The generation order is a Hamiltonian path, so ``L + D`` is a feasible tour
    on the SKELETON and ``alpha_pred`` is an upper bound on the skeleton's
    alpha, not a heuristic.  It is only a bound on the realised instance's
    alpha to the extent that the transverse noise leaves the MST unchanged, so
    a thick or very unevenly sampled set can come out slightly above it -- the
    audit measures by how much rather than assuming it cannot happen.  It is
    tight whenever the traversal the generator used is the one the optimal tour
    takes, which is the case for every family here:

      * simple curve -- MST is the chain, so this collapses to ``1 + D/L``,
        the closed-form closure law the module is built on;
      * branched skeleton (``spider``) -- MST is the star of arms, the chain is
        the out-and-back traversal at twice that, and the prediction is 2 as it
        should be.  ``1 + D/L`` would have said 1 and been wrong, which is why
        the MST is in the denominator rather than L.
    """
    seg = np.linalg.norm(np.diff(skeleton, axis=0), axis=1)
    L = float(seg.sum())
    if L <= 0.0:
        return float("nan")
    D = float(np.linalg.norm(skeleton[-1] - skeleton[0]))
    mst = float(mst_length(skeleton))
    if mst <= 0.0:
        return float("nan")
    return (L + D) / mst


def solve_shape(build, target: float, lo: float = 0.0, hi: float = 1.0,
                iters: int = 24, tol: float = 5e-3) -> Tuple[float, float]:
    """Bisect the shape scalar s until ``predict_alpha(build(s)) == target``.

    ``build(s)`` returns a skeleton and alpha_pred must be non-increasing in s
    (every family is built that way).  Returns (s, achieved_alpha_pred).  When
    the target lies outside the family's reachable range the nearer endpoint is
    returned, and the caller checks the residual -- see ``build_coords``.
    """
    a_lo, a_hi = predict_alpha(build(lo)), predict_alpha(build(hi))
    if not (a_hi - 1e-9 <= target <= a_lo + 1e-9):
        return (lo, a_lo) if abs(a_lo - target) <= abs(a_hi - target) else (hi, a_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = predict_alpha(build(mid))
        if abs(a - target) <= tol:
            return mid, a
        if a > target:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    return s, predict_alpha(build(s))


# ---------------------------------------------------------------------------
# Skeletons.  Each returns points in generation (traversal) order, centred on
# the origin and scaled to a BOX_FILL * G bounding extent by ``_place``.
# Each takes a single scalar s in [0, 1] with kappa non-increasing in s.
# ---------------------------------------------------------------------------

def _t_samples(rng, n: int, spacing: str) -> np.ndarray:
    """Sorted curve parameters in [0, 1].

    ``uniform`` -- i.i.d. uniform, so the gaps are exponential and the largest
    is ~log(n)/n of the curve.  ``strat`` -- one jittered sample per 1/n cell,
    so the spacing is near-regular.  Real routing point sets sit at both
    extremes and the two give visibly different ``mst_edge_*`` signatures, so
    both are swept.  It also matters mechanically: at small n the i.i.d. largest
    gap is a sizeable fraction of a closed curve, which lifts the reachable
    alpha floor away from 1.
    """
    if spacing == "strat":
        return (np.arange(n) + rng.uniform(0.0, 1.0, n)) / n
    return np.sort(rng.uniform(0.0, 1.0, n))


SPACINGS = ("uniform", "strat")


def _place(y: np.ndarray, d: int, grid_size: float, rng) -> np.ndarray:
    """Embed a k-dim skeleton into R^d, scale to BOX_FILL*G, centre in the box."""
    k = y.shape[1]
    basis = _orthonormal_basis(rng, d, min(k, d))      # (min(k,d), d)
    if k > basis.shape[0]:
        y = y[:, : basis.shape[0]]
    coords = y @ basis
    extent = float(np.max(np.ptp(coords, axis=0)))
    if extent > 0:
        coords = coords * (BOX_FILL * grid_size / extent)
    return coords + grid_size / 2.0 - coords.mean(axis=0)


def skel_arc(rng, n: int, d: int, grid_size: float, s: float,
             spacing: str = "uniform") -> np.ndarray:
    """Circular arc, sweep theta = ARC_THETA_MIN + s*(2pi - ARC_THETA_MIN).

    In the thin limit alpha = 1 + 2 sin(theta/2)/theta: 1.9896 at s=0 and 1 at
    s=1, monotone and in closed form.  This is the family that guarantees the
    corpus can reach any requested alpha bin at any n.
    """
    theta = ARC_THETA_MIN + s * (2.0 * math.pi - ARC_THETA_MIN)
    t = _t_samples(rng, n, spacing) * theta
    y = np.stack([np.cos(t), np.sin(t)], axis=1)
    return _place(y, d, grid_size, rng)


def skel_fourier(rng, n: int, d: int, grid_size: float, s: float,
                 spacing: str = "uniform", harmonics: int = 3) -> np.ndarray:
    """Band-limited random space curve plus a straightening drift.

    c(t) = drift(s) * t * e_1 + sum_j (a_j cos 2pi j t + b_j sin 2pi j t).
    The Fourier part alone is a closed loop (alpha_pred ~ 1); the drift opens
    it.  drift is largest at s=0 and zero at s=1.  At s=0 the curve is a gentle
    3-harmonic wave, not a line: transverse velocity reaches ~30% of the drift
    velocity, so the shape is visibly sinuous while alpha_pred is ~1.96.
    """
    kdim = min(3, max(2, d))
    t = _t_samples(rng, n, spacing)
    amp = rng.normal(size=(harmonics, 2, kdim)) / (np.arange(1, harmonics + 1)[:, None, None] ** 1.5)
    y = np.zeros((n, kdim))
    for j in range(harmonics):
        ang = 2.0 * math.pi * (j + 1) * t
        y += np.outer(np.cos(ang), amp[j, 0]) + np.outer(np.sin(ang), amp[j, 1])
    loop_scale = float(np.max(np.ptp(y, axis=0))) or 1.0
    drift = (1.0 - s) ** 3 * 12.0 * loop_scale
    y[:, 0] += drift * t
    return _place(y, d, grid_size, rng)


SERP_NU_MIN = 1.15


def serpentine_nu_max(n: int) -> float:
    """Largest sweep count that keeps the path one-dimensional.

    Rows sit at pitch 1/nu and points sit at spacing nu/n along a row, so the
    path only reads as 1-D while nu/n << 1/nu, i.e. nu << sqrt(n).  Past that
    the MST bridges between rows, the set is a filled square, and its alpha is
    the planar value -- which the corpus already has in abundance.  The probe
    caught exactly this: an uncapped nu made alpha_pred non-monotone and turned
    the family into a second copy of the uniform prior.
    """
    return max(SERP_NU_MIN + 0.5, math.sqrt(n) / 3.0)


def skel_serpentine(rng, n: int, d: int, grid_size: float, s: float,
                    spacing: str = "uniform") -> np.ndarray:
    """Boustrophedon (lawnmower) path with a continuous sweep count.

    nu sweeps of a unit-length row, stacked at pitch 1/nu so the path fills a
    unit square rather than becoming a filament.  Few sweeps leave the
    endpoints far apart (high alpha); more sweeps fold the path back until they
    nearly meet.  nu is continuous, so alpha_pred is continuous in s.

    Not a lattice and not the 2D benchmark's ``grid``: the points lie along a
    one-dimensional path with a partial final row, never on a product grid, and
    there is no per-site jitter law.  nu starts at 1.15, so the path always
    folds at least once and is never a straight run.
    """
    nu = SERP_NU_MIN + s * (serpentine_nu_max(n) - SERP_NU_MIN)
    u = _t_samples(rng, n, spacing) * nu              # position in sweeps
    row = np.floor(u)
    frac = u - row
    x = np.where(row % 2 == 0, frac, 1.0 - frac)
    y = row / nu
    return _place(np.stack([x, y], axis=1), d, grid_size, rng)


def skel_spider(rng, n: int, d: int, grid_size: float, s: float,
                spacing: str = "uniform", arms: int = 4) -> np.ndarray:
    """k-armed star with feet at radius 0.97*s, traversed in adjacent pairs.

    A branched skeleton -- no evaluation generator produces one -- and the only
    family here whose MST is not a path.  At s=0 the arms meet at a hub, the
    MST is the star (k*l) and the shortest tour runs up one arm, across to the
    neighbouring tip and back down it, giving alpha = 1 + sin(pi/k): 1.87 at
    k=3 down to 1.38 at k=8.  As s rises the feet slide out towards the tips
    and the set becomes k short radial ticks around a ring, so alpha falls
    towards 1.

    The traversal order IS that pairing, which is the point.  ``predict_alpha``
    is only as tight as the generation order is optimal, and an out-and-back
    order predicts 2 while the solver finds the tip-to-tip pairing: measured
    1.666 against a predicted 1.955 at k=4, exactly 1 + sin(pi/4) = 1.707 less
    the thickness correction.  Ordering the arms the way the optimal tour runs
    them removes the error rather than patching the predictor.

    Arm directions are equispaced with bounded jitter, never i.i.d. uniform.
    With i.i.d. angles two arms can nearly coincide, their point sets merge,
    the MST stops being the star, and alpha_pred runs above 2 -- the probe
    measured 2.33 at k=8 before this was fixed.
    """
    ang = 2.0 * math.pi * (np.arange(arms) + rng.uniform(0.3, 0.7, arms)) / arms
    tips = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    feet = tips * (0.97 * s)
    counts = np.full(arms, n // arms)
    counts[: n - int(counts.sum())] += 1

    out = []
    for a in range(arms):
        c = int(counts[a])
        if c <= 0:
            continue
        t = _t_samples(rng, c, spacing)
        # Odd-indexed arms run tip -> foot so that arm 2j and arm 2j+1 are
        # joined at their tips, which is the shortest way to cover a star.
        if a % 2 == 1:
            t = t[::-1]
        out.append(feet[a] + t[:, None] * (tips[a] - feet[a]))
    return _place(np.vstack(out)[:n], d, grid_size, rng)


def skel_polyline(rng, n: int, d: int, grid_size: float, s: float,
                  spacing: str = "uniform", segs: int = 4,
                  density: float = 1.0) -> np.ndarray:
    """Chain of straight segments with vertices on a circular arc of sweep theta.

    Adds the one axis the smooth families lack: point density that varies
    sharply from segment to segment.  Per-segment counts come from a Dirichlet
    with concentration ``density`` -- 0.3 gives a corpus where one segment
    carries most of the mass, 4.0 an almost even split -- while the closure
    knob is inherited from the arc that carries the vertices, so alpha is still
    placed exactly.

    Distinct from ``augment_gen.gen_polyline``, which puts its waypoints on the
    faces of the box in order to imitate the coordinate clipping in the 2D
    benchmark's ``line_noise``.  That is test-locus-targeted by construction and
    is disclosed as such in the arm-C protocol; this one is not, because its
    vertices lie on the same circular locus every other family here uses and no
    benchmark instance was looked at to choose any of its parameters.
    """
    theta = ARC_THETA_MIN + s * (2.0 * math.pi - ARC_THETA_MIN)
    segs = max(2, min(segs, max(2, n // 4)))
    phi = np.linspace(0.0, theta, segs + 1)
    verts = np.stack([np.cos(phi), np.sin(phi)], axis=1)

    share = rng.dirichlet(np.full(segs, density))
    counts = np.maximum(2, (share * n).astype(int))
    counts[-1] += n - int(counts.sum())
    if counts[-1] < 2:
        counts[0] += counts[-1] - 2
        counts[-1] = 2

    out = []
    for j in range(segs):
        c = int(counts[j])
        if c <= 0:
            continue
        t = _t_samples(rng, c, spacing)[:, None]
        out.append(verts[j] + t * (verts[j + 1] - verts[j]))
    return _place(np.vstack(out)[:n], d, grid_size, rng)


SKELETONS = {
    "arc": skel_arc,
    "fourier": skel_fourier,
    "serpentine": skel_serpentine,
    "spider": skel_spider,
    "polyline": skel_polyline,
}


# ---------------------------------------------------------------------------
# Instance assembly: skeleton -> thickness -> background mixture
# ---------------------------------------------------------------------------

def build_coords(rng, family: str, n: int, d: int, grid_size: float,
                 alpha_target: Optional[float], s_fixed: Optional[float],
                 rho: float, profile: str, mix: float, spacing: str,
                 family_kwargs: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return (coords, geometry).

    ``alpha_target`` places the skeleton by bisecting on the shape scalar;
    ``s_fixed`` uses a given scalar instead.  Exactly one must be supplied.
    """
    if (alpha_target is None) == (s_fixed is None):
        raise ValueError("give exactly one of alpha_target, s_fixed")

    n_skel = n if mix <= 0.0 else max(3, int(round(n * (1.0 - mix))))
    # One frozen seed for the whole bisection so ``build`` is a smooth
    # deterministic function of s and not a fresh random curve at each probe.
    skel_seed = int(rng.integers(0, 2 ** 32))

    def builder(fam: str, kwargs: Dict[str, Any]):
        fn = SKELETONS[fam]

        def build(s: float) -> np.ndarray:
            return fn(np.random.default_rng(skel_seed), n_skel, d, grid_size, s,
                      spacing=spacing, **kwargs)
        return build

    build = builder(family, family_kwargs)
    used_family, fallback = family, False

    if alpha_target is not None:
        s, a_pred = solve_shape(build, float(alpha_target))
        # A family that cannot reach the requested bin would silently pile its
        # instances at its own endpoint and break the uniform alpha coverage
        # this corpus exists to provide. ``arc`` spans the full range in closed
        # form, so it is the fallback and the substitution is recorded.
        if abs(a_pred - float(alpha_target)) > ALPHA_PLACEMENT_TOL and family != "arc":
            used_family, fallback = "arc", True
            build = builder("arc", {})
            s, a_pred = solve_shape(build, float(alpha_target))
    else:
        s = float(s_fixed)
        a_pred = predict_alpha(build(s))

    skeleton = build(s)
    chain = float(np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1)))

    coords = skeleton
    if rho > 0.0:
        tangent = np.gradient(skeleton, axis=0)
        coords = skeleton + _transverse_noise(rng, n_skel, d, tangent,
                                              rho * chain / n_skel, profile)

    if mix > 0.0:
        n_bg = n - n_skel
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        span = np.maximum(hi - lo, 0.02 * grid_size)
        bg = lo + rng.uniform(0.0, 1.0, (n_bg, d)) * span
        coords = np.vstack([coords, bg])

    coords = np.clip(np.asarray(coords, dtype=np.float64), 0.0, grid_size)
    coords = _dedupe(coords, grid_size, rng)

    geometry = {
        "shape_scalar": float(s),
        "alpha_pred": float(a_pred),
        "used_family": used_family,
        "family_fallback": bool(fallback),
        "rho": float(rho),
        "profile": profile,
        "mix": float(mix),
        "spacing": spacing,
        "rho_measured": measured_rho(coords),
        "transverse_kurtosis": transverse_kurtosis(coords),
        "skeleton_chain_length": chain,
    }
    return coords, geometry


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

def _slug(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}".replace(".", "p").replace("-", "m")
    return str(v)


@dataclass(frozen=True)
class CovSpec:
    family: str
    d: int
    n: int
    grid_size: int
    alpha_target: Optional[float] = None
    s_fixed: Optional[float] = None
    rho: float = 0.05
    profile: str = "perp"
    mix: float = 0.0
    spacing: str = "uniform"
    arms: int = 4
    segs: int = 4
    density: float = 1.0
    rep: int = 0
    group: str = ""

    @property
    def name(self) -> str:
        tgt = f"a{_slug(round(self.alpha_target, 3))}" if self.alpha_target is not None \
            else f"s{_slug(round(float(self.s_fixed), 3))}"
        bits = [f"{self.family}", f"d{self.d}", f"n{self.n}", f"g{self.grid_size}",
                tgt, f"rho{_slug(self.rho)}", self.profile, self.spacing]
        if self.mix > 0:
            bits.append(f"mix{_slug(self.mix)}")
        if self.family == "spider":
            bits.append(f"k{self.arms}")
        if self.family == "polyline":
            bits.append(f"seg{self.segs}")
            bits.append(f"dens{_slug(self.density)}")
        return NAME_PREFIX + "-".join(bits) + f"-r{self.rep}"

    @property
    def seed(self) -> int:
        return zlib.crc32(f"coverage-v1|{self.name}".encode("utf-8"))

    def family_kwargs(self) -> Dict[str, Any]:
        if self.family == "spider":
            return {"arms": self.arms}
        if self.family == "polyline":
            return {"segs": self.segs, "density": self.density}
        return {}

    def generate(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        return build_coords(rng, self.family, self.n, self.d, float(self.grid_size),
                            self.alpha_target, self.s_fixed, self.rho,
                            self.profile, self.mix, self.spacing, self.family_kwargs())


# ---------------------------------------------------------------------------
# Leakage gates
# ---------------------------------------------------------------------------

def _dir_names(directory: str) -> set:
    names = set()
    if not os.path.isdir(directory):
        return names
    for entry in os.scandir(directory):
        if not entry.is_file():
            continue
        stem = entry.name
        for ext in (".sol.json", ".json", ".bin"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        names.add(stem)
    return names


def assert_disjoint(specs: List[CovSpec]) -> None:
    wanted = [s.name for s in specs]
    if len(set(wanted)) != len(wanted):
        seen, dupes = set(), set()
        for w in wanted:
            (dupes if w in seen else seen).add(w)
        raise AssertionError(f"duplicate COV names in batch: {sorted(dupes)[:5]}")
    wanted_set = set(wanted)
    for directory in GUARDED_DIRS:
        existing = _dir_names(directory)
        clash = wanted_set & existing
        if clash:
            raise AssertionError(f"name collision with {directory}: {sorted(clash)[:5]}")
        prefixed = [x for x in existing if x.startswith(NAME_PREFIX)]
        if prefixed:
            raise AssertionError(f"{directory} already holds {NAME_PREFIX} names: {prefixed[:5]}")
    print(f"  names disjoint from {len(GUARDED_DIRS)} guarded corpora")


def assert_no_leakage() -> int:
    written = {f[:-5] for f in os.listdir(COV_INSTANCES_DIR) if f.endswith(".json")}
    for directory in GUARDED_DIRS:
        existing = _dir_names(directory)
        overlap = written & existing
        if overlap:
            raise AssertionError(f"LEAK: {len(overlap)} COV names in {directory}")
        prefixed = [x for x in existing if x.startswith(NAME_PREFIX)]
        if prefixed:
            raise AssertionError(f"LEAK: {directory} holds {NAME_PREFIX} names")
    return len(written)


# ---------------------------------------------------------------------------
# Build + solve + gate
# ---------------------------------------------------------------------------

def already_done(spec: CovSpec) -> bool:
    return (os.path.exists(os.path.join(COV_INSTANCES_DIR, f"{spec.name}.json"))
            and os.path.exists(os.path.join(COV_SOLUTIONS_DIR, f"{spec.name}.sol.json")))


def concorde_timeout_for(spec: CovSpec, base_s: int) -> int:
    """One flat budget, deliberately not raised for the large-n cells.

    Raising it to 900 s there was a mistake: the solve-time distribution at
    n=700 is median 14 s, p90 59 s, max 889 s, so a handful of instances pinned
    a pool shell for a quarter of an hour each and dragged throughput from an
    expected ~85/min down to 11.7/min. Truncating the tail costs a proven
    label on roughly 3% of the n>=700 rows, which fall back to LKH and are
    reported as heuristic in the audit; leaving it uncapped costs about ten
    hours. The bound gate still applies to every one of them.
    """
    return base_s


# Concorde runs inside WSL, and wsl.exe does not tolerate many simultaneous
# launches on this box. Measured on 48 n=200 coverage instances, re-solving the
# same set at three concurrencies:
#
#     workers=16   32/48 succeeded    84 solves/min   median call  3.2 s
#     workers= 8   32/48 succeeded    46 solves/min   median call  0.4 s
#     workers= 4   48/48 succeeded   643 solves/min   median call  0.3 s
#
# The failures are "Catastrophic failure" / "A connection attempt failed" from
# wsl.exe itself, after ~29 s of launch overhead, and each one silently
# demoted a label from proven-optimal to LKH-heuristic. The instances are not
# hard: the solve takes 0.3 s once the launch succeeds. Gating Concorde at 4
# leaves the other worker threads free for generation, MST and the 1-tree
# bound, and is both faster overall and the only configuration that produces
# exact labels.
# Concorde is single-threaded and the box has 20 cores. The per-instance
# Python work -- bisection, MST, the 1-tree bound, verification -- profiles at
# 0.2 to 1.4 s against a 6.2 s mean solve, so the pool size is what sets
# throughput. The solve-time tail is thin (0.5% over 300 s, 0.1% over 420 s)
# but each of those pins a shell for minutes, which is why the pool is wide
# rather than matched to the mean.
CONCORDE_CONCURRENCY = 20
CONCORDE_ATTEMPTS = 3


class WslPool:
    """A pool of persistent ``wsl bash`` shells, spawned serially, reused forever.

    ``wsl.exe`` cannot be launched concurrently on this machine.  Measured with
    a command that does nothing at all::

        wsl true x60, 1 worker    0/60 failed    median 0.10 s
        wsl true x60, 4 workers  32/60 failed    median 30.0 s
        wsl true x60, 8 workers  40/60 failed    median 30.0 s

    Every failure is ``rc=-1`` with "Catastrophic failure" or "A connection
    attempt failed" from wsl.exe itself, after a flat 30 s timeout.  It is not
    Concorde, not the instances, and not VM uptime -- recycling the VM with
    ``wsl --shutdown`` left the rate unchanged at 4 workers.  It is the launch.

    That cost 1,120 of the first 1,768 labels: each failed launch was taken as
    "Concorde cannot solve this" and demoted the instance to an LKH heuristic
    label, in a corpus built specifically to fix a labelling defect.

    The fix is to stop launching.  Shells are spawned one at a time under a
    global lock, then held open; commands go down stdin and completion is
    detected by a sentinel echoed after the command with its exit status.  The
    Linux processes themselves parallelise fine -- it was only ever the
    Windows-side launch that could not.
    """

    def __init__(self, size: int):
        self._launch_lock = threading.Lock()
        self._free: "queue.Queue[int]" = queue.Queue()
        self._procs: List[Optional[subprocess.Popen]] = [None] * size
        self._queues: List[Optional["queue.Queue[Optional[str]]"]] = [None] * size
        for i in range(size):
            self._spawn(i)
            self._free.put(i)

    def _spawn(self, i: int) -> None:
        with self._launch_lock:
            p = subprocess.Popen(
                ["wsl", "-e", "bash", "-s"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding="utf-8", errors="replace",
            )
            # Text-mode pipes translate "\n" to "\r\n" on Windows, and bash
            # then sees a trailing carriage return on every line: "whoami\r:
            # command not found". Write LF verbatim.
            p.stdin.reconfigure(newline="\n")
            time.sleep(0.35)          # keep launches from overlapping
        q: "queue.Queue[Optional[str]]" = queue.Queue()

        def reader(pipe, out_q):
            for line in pipe:
                out_q.put(line.rstrip("\n"))
            out_q.put(None)

        threading.Thread(target=reader, args=(p.stdout, q), daemon=True).start()
        self._procs[i], self._queues[i] = p, q

    def run(self, command: str, timeout_s: float) -> Tuple[int, str]:
        """Run a shell command in a pooled shell. Returns (exit_status, output)."""
        i = self._free.get()
        try:
            p, q = self._procs[i], self._queues[i]
            token = uuid.uuid4().hex
            marker = f"__COVDONE_{token}__"
            try:
                p.stdin.write(f"{command}\nprintf '%s%d\\n' {marker} $?\n")
                p.stdin.flush()
            except (BrokenPipeError, OSError):
                self._spawn(i)
                raise RuntimeError("wsl shell died before the command was sent")

            deadline = time.monotonic() + timeout_s
            lines: List[str] = []
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_and_respawn(i)
                    raise TimeoutError(f"wsl command exceeded {timeout_s}s")
                try:
                    line = q.get(timeout=min(remaining, 5.0))
                except queue.Empty:
                    continue
                if line is None:                       # shell exited
                    self._spawn(i)
                    raise RuntimeError("wsl shell closed unexpectedly")
                if line.startswith(marker):
                    return int(line[len(marker):] or 1), "\n".join(lines)
                lines.append(line)
        finally:
            self._free.put(i)

    def _kill_and_respawn(self, i: int) -> None:
        try:
            self._procs[i].kill()
        except Exception:
            pass
        self._spawn(i)


_WSL_POOL: Optional[WslPool] = None
_WSL_POOL_LOCK = threading.Lock()


def wsl_pool() -> WslPool:
    global _WSL_POOL
    with _WSL_POOL_LOCK:
        if _WSL_POOL is None:
            _WSL_POOL = WslPool(CONCORDE_CONCURRENCY)
    return _WSL_POOL


def concorde_via_pool(coords: np.ndarray, grid_size: int, timeout_s: int
                      ) -> Tuple[int, float, List[int], Optional[int]]:
    """Concorde through a pooled shell. Same scaling and TSPLIB writer as
    ``solvers.concorde``; the only difference is that no process is launched."""
    scale = get_scale_factor(float(grid_size))
    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join(SOLVER_SCRATCH_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    tsp_win = os.path.join(run_dir, f"{run_id}.tsp")
    try:
        tsplib_io.save_fast(tsp_win, coords, run_id, grid_size)
        wsl_tsp = _wsl_path(tsp_win)
        wsl_dir = _wsl_path(run_dir)
        wsl_tour = wsl_tsp.replace(".tsp", ".tour")

        # Concorde is verbose -- thousands of lines at n=1000. Sending all of it
        # back up the shell pipe, line by line through a reader thread holding
        # the GIL, cost more than the solve: 85 s per n=1000 instance against a
        # 16 s solve. Keep the log inside WSL and return only the one line the
        # integrity gate needs, or the tail on failure.
        start = time.perf_counter()
        rc, out = wsl_pool().run(
            f"cd {wsl_dir} && timeout -k 5 {timeout_s} {CONCORDE_WSL_BIN} "
            f"-o {wsl_tour} {wsl_tsp} > cc.log 2>&1; rc=$?; "
            f"if [ $rc -ne 0 ]; then tail -5 cc.log; "
            f"else grep -m1 'Optimal Solution' cc.log; fi; (exit $rc)",
            timeout_s + 60,
        )
        runtime = time.perf_counter() - start

        if rc in (124, 137):
            raise TimeoutError(f"Concorde exceeded {timeout_s}s")
        if rc != 0:
            raise RuntimeError(f"Concorde rc={rc}: {out.strip()[:200]}")

        tour_win = tsp_win.replace(".tsp", ".tour")
        if not os.path.exists(tour_win):
            raise FileNotFoundError("Concorde produced no tour file")

        tour = _read_tour(tour_win)
        length = compute_tour_length_numba(coords, np.array(tour), scale)
        m = re.search(r"Optimal Solution:\s*([0-9.]+)", out)
        reported = int(round(float(m.group(1)))) if m else None
        return length, runtime, tour, reported
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def solve_gated(coords: np.ndarray, grid_size: int, name: str, d: int,
                concorde_timeout_s: int, lkh_time_limit_s: int,
                lkh_proc_timeout_s: int = 900) -> Dict[str, Any]:
    """Concorde under the concurrency gate, retried, then LKH as a last resort.

    ``augment_gen.solve_with_fallback`` takes the first Concorde failure as
    final and drops straight to LKH.  That is right when Concorde has genuinely
    given up on a hard instance and wrong when wsl.exe simply refused to launch,
    which is what happened to 1,120 of the first 1,768 labels here.  Retrying
    distinguishes the two, and both the attempt count and the errors are stored
    so the distinction stays auditable instead of being inferred later.
    """
    scale = get_scale_factor(float(grid_size))
    wall_start = time.perf_counter()
    errors: List[str] = []

    for attempt in range(CONCORDE_ATTEMPTS):
        try:
            length, runtime, tour, reported = concorde_via_pool(
                coords, grid_size, concorde_timeout_s)
            return {
                "solver_used": "concorde", "concorde_error": None,
                "concorde_attempts": attempt + 1, "concorde_errors": errors,
                "solver_reported_cost": reported, "cost_int_scaled": int(length),
                "optimal_cost": float(length) / scale,
                "optimal_tour": [int(x) for x in tour],
                "scale_factor": float(scale), "solver_time_s": float(runtime),
                "wall_time_s": time.perf_counter() - wall_start,
            }
        except TimeoutError as exc:
            errors.append(f"TimeoutError: {exc}")
            break                       # a real timeout will just recur
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:200])
            time.sleep(1.5 * (attempt + 1))

    length, runtime, tour = run_lkh(name, coords, d, grid_size,
                                    time_limit_s=lkh_time_limit_s,
                                    proc_timeout_s=lkh_proc_timeout_s)
    return {
        "solver_used": "lkh", "concorde_error": errors[-1] if errors else None,
        "concorde_attempts": CONCORDE_ATTEMPTS, "concorde_errors": errors,
        "solver_reported_cost": None, "cost_int_scaled": int(length),
        "optimal_cost": float(length) / scale,
        "optimal_tour": [int(x) for x in tour],
        "scale_factor": float(scale), "solver_time_s": float(runtime),
        "wall_time_s": time.perf_counter() - wall_start,
    }


def solve_exact_small(coords: np.ndarray, grid_size: int) -> Dict[str, Any]:
    """Exact Held-Karp DP on the integer matrix, for n <= EXACT_N_MAX.

    Concorde is unstable on tiny instances -- ``solvers/concorde.py`` says so in
    its own docstring, and it duly returned "Catastrophic failure" on 208 of the
    first 300 n=20 coverage instances, each of which then fell back to LKH.  LKH
    at n=20 is optimal in practice but not proven, and a corpus built to fix a
    labelling defect should not quietly accept heuristic labels where an exact
    answer costs 23 seconds.  This is also the path ``run_concorde`` takes for
    n <= 20, so the small-n coverage labels are produced exactly as the ND
    corpus's own small-n labels were.
    """
    scale = get_scale_factor(float(grid_size))
    start = time.perf_counter()
    length, tour = held_karp(np.ascontiguousarray(coords, dtype=np.float64), scale)
    runtime = time.perf_counter() - start
    return {
        "solver_used": "held_karp_exact",
        "concorde_error": None,
        # The DP's own objective, checked against the tour re-measured from the
        # stored coordinates by GATE_SOLVER_REPORTED.
        "solver_reported_cost": int(length),
        "cost_int_scaled": int(length),
        "optimal_cost": float(length) / scale,
        "optimal_tour": [int(x) for x in tour],
        "scale_factor": float(scale),
        "solver_time_s": float(runtime),
        "wall_time_s": float(runtime),
    }


def build_and_solve(spec: CovSpec, concorde_timeout_s: int, lkh_time_limit_s: int,
                    write: bool = True, lkh_proc_timeout_s: int = 900) -> Dict[str, Any]:
    coords, geometry = spec.generate()
    if len(coords) <= EXACT_N_MAX:
        sol = solve_exact_small(coords, spec.grid_size)
    else:
        sol = solve_gated(coords, spec.grid_size, spec.name, spec.d,
                          concorde_timeout_for(spec, concorde_timeout_s),
                          lkh_time_limit_s, lkh_proc_timeout_s)
    check = verify_solution(coords, sol)

    mst_total = float(mst_length(coords))
    alpha = sol["optimal_cost"] / mst_total if mst_total > 0 else float("nan")

    # Held-Karp 1-tree lower bound. bound <= label is a hard gate: a label
    # below a valid lower bound is proof the label is wrong.
    # Held-Karp 1-tree lower bound, computed on THE MATRIX THE SOLVER READ.
    # The solver optimises the integer-rounded metric, so a float-Euclidean
    # bound is not comparable with the label: rounding lets the integer tour
    # fall a little under the float length, and the pilot duly refused two
    # sound instances at hk/cost = 1.00010 and 1.00007 -- exactly the
    # 0.29*sqrt(n)/scale rounding slack at n=100, scale=10. Same-metric is the
    # only version of this gate that means anything.
    hk = one_tree_bound_from_matrix(
        compute_distance_matrix(np.ascontiguousarray(coords, dtype=np.float64),
                                sol["scale_factor"]).astype(np.float64),
        hk_iterations_for(spec.n))
    hk_bound = float(hk.bound)
    hk_ok = hk_bound <= sol["cost_int_scaled"] * (1.0 + 1e-9)

    record = {
        "name": spec.name, "family": spec.family,
        "used_family": geometry["used_family"],
        "family_fallback": geometry["family_fallback"], "group": spec.group,
        "d": spec.d, "n": spec.n, "grid_size": spec.grid_size,
        "alpha_target": spec.alpha_target, "alpha_pred": geometry["alpha_pred"],
        "shape_scalar": geometry["shape_scalar"],
        "rho": spec.rho, "profile": spec.profile, "mix": spec.mix,
        "spacing": spec.spacing,
        "arms": spec.arms if spec.family == "spider" else None,
        "rho_measured": geometry["rho_measured"],
        "transverse_kurtosis": geometry["transverse_kurtosis"],
        "optimal_cost": sol["optimal_cost"], "mst_total_length": mst_total,
        "alpha": alpha,
        "solver_used": sol["solver_used"], "solver_time_s": sol["solver_time_s"],
        "wall_time_s": sol["wall_time_s"], "concorde_error": sol["concorde_error"],
        "concorde_attempts": sol.get("concorde_attempts", 0),
        "integrity_ok": check["ok"], "failed_gates": check["failed_gates"],
        "integrity_problems": check["problems"],
        "float_rel_dev": check["float_rel_dev"],
        "hk_bound": hk_bound, "hk_ratio": hk_bound / max(sol["cost_int_scaled"], 1e-12),
        "hk_ok": bool(hk_ok), "written": False,
    }

    if not check["ok"] or not hk_ok:
        return record

    if write:
        os.makedirs(COV_INSTANCES_DIR, exist_ok=True)
        os.makedirs(COV_SOLUTIONS_DIR, exist_ok=True)
        with open(os.path.join(COV_INSTANCES_DIR, f"{spec.name}.json"), "w") as f:
            json.dump({
                "instance_name": spec.name, "n_customers": spec.n,
                "dimension": spec.d, "grid_size": spec.grid_size,
                "distribution_type": f"cov_{geometry['used_family']}",
                "coverage_family": geometry["used_family"],
                "coverage_family_requested": spec.family,
                "coverage_group": spec.group,
                "coverage_geometry": geometry,
                "generation_seed": spec.seed,
                "coordinates": coords.tolist(),
            }, f)
        with open(os.path.join(COV_SOLUTIONS_DIR, f"{spec.name}.sol.json"), "w") as f:
            json.dump({
                "instance_name": spec.name,
                "optimal_cost": sol["optimal_cost"],
                "optimal_tour": sol["optimal_tour"],
                "optimal_solver": sol["solver_used"],
                "solver_time_s": sol["solver_time_s"],
                "wall_time_s": sol["wall_time_s"],
                "scale_factor": sol["scale_factor"],
                "cost_int_scaled": sol["cost_int_scaled"],
                "concorde_error": sol["concorde_error"],
                "concorde_attempts": sol.get("concorde_attempts", 0),
                "concorde_errors": sol.get("concorde_errors", []),
                "mst_total_length": mst_total, "alpha": alpha,
                "coverage_geometry": geometry,
                "integrity": {
                    "recomputed_int": check["recomputed_int"],
                    "matrix_int": check["matrix_int"],
                    "solver_reported_cost": check["solver_reported_cost"],
                    "float_tour_length": check["float_tour_length"],
                    "float_rel_dev": check["float_rel_dev"],
                    "hk_1tree_bound_int_scaled": hk_bound,
                    "hk_1tree_iterations": hk_iterations_for(spec.n),
                    "hk_1tree_metric": "integer distance matrix at scale_factor",
                    "hk_bound_le_cost": bool(hk_ok),
                },
            }, f)
        record["written"] = True
    return record


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

# alpha bin centres: 10 bins of width 0.1 spanning the whole admissible range.
ALPHA_BINS = tuple(round(1.05 + 0.1 * i, 2) for i in range(10))
# The small end is 30, not 20. Below EXACT_N_MAX = 20 the solve goes through
# ``solvers.exact.held_karp``, a pure-Python bitmask DP: 23 s per instance at
# n=20, and it holds the GIL, so sixteen worker threads deliver 2.3 solves per
# minute rather than 42. That is five hours for the least useful block in the
# corpus -- the training split already spans alpha 1.07 to 1.98 at n <= 20, so
# there is no coverage gap there. n=30 sits in the (20, 50] decade, where the
# corpus p99 is 1.28 and the max 1.57, so the gap is real; it is above
# EXACT_N_MAX, so Concorde solves it exactly in ~0.2 s. Same coverage, exact
# labels, no serial bottleneck. n=30 is in the ND corpus's own size ladder.
N_LADDER = (30, 50, 100, 200, 300, 500, 700, 1000)
D_LADDER = (2, 3, 5, 10, 20, 50)
GRIDS = (1000, 10000)
RHO_THIN = (0.02, 0.06, 0.15)
PROFILES = ("perp", "band")

# Only families whose generation order is the tour the solver actually finds
# may carry a targeted alpha bin, because that is what makes ``predict_alpha``
# tight.  Measured against Concorde: arc -0.010, fourier -0.014, serpentine
# -0.002, spider at k>=6 -0.008.  Spider at k=3..5 mispredicts by up to -0.17
# once the feet lift off the hub -- the star then admits shortcuts the fixed
# pairing does not take -- so those live in the untargeted blocks instead.
TARGETED_FAMILIES = ("arc", "fourier", "serpentine", "spider")
TARGETED_SPIDER_ARMS = (6, 8, 10)
FREE_SPIDER_ARMS = (3, 4, 5, 6, 8)


def pilot_plan() -> List[CovSpec]:
    """Small, wide probe: does 1 + kappa predict alpha, and what does it cost?"""
    specs: List[CovSpec] = []
    i = 0
    for fam in SKELETONS:
        for n, d in ((100, 2), (200, 5), (500, 10), (1000, 2)):
            for a in (1.15, 1.45, 1.75, 1.95):
                specs.append(CovSpec(fam, d, n, 1000, alpha_target=a, rho=0.06,
                                     profile="perp", spacing=SPACINGS[i % 2],
                                     group="pilot"))
                i += 1
    for rho in (0.5, 1.5, 4.0):
        specs.append(CovSpec("arc", 2, 500, 1000, alpha_target=1.85, rho=rho,
                             profile="perp", group="pilot_rho"))
    for mix in (0.2, 0.5, 0.8):
        specs.append(CovSpec("arc", 2, 500, 1000, alpha_target=1.85, rho=0.06,
                             mix=mix, group="pilot_mix"))
    return specs


def main_plan(reps: int = 3, seed: int = 20260812) -> List[CovSpec]:
    """The coverage corpus.

    Block A (targeted, thin): every (n, d) cell x every alpha bin, placed by
    bisection.  This is what makes the corpus uniform in alpha at every n.
    Family, grid, profile and rho rotate deterministically so no alpha bin is
    tied to one generator.

    Block B (thickness): the same cells at rho >> 1, where the set stops being
    1-D.  alpha lands wherever the geometry puts it; the point is a second,
    independent mechanism for interior alpha.

    Block C (mixture): skeleton plus uniform background, likewise untargeted.
    """
    rng = np.random.default_rng(seed)
    free_fams = list(SKELETONS)
    specs: List[CovSpec] = []

    # --- Block A ------------------------------------------------------------
    i = 0
    for n in N_LADDER:
        for d in D_LADDER:
            for a in ALPHA_BINS:
                for r in range(reps):
                    fam = TARGETED_FAMILIES[i % len(TARGETED_FAMILIES)]
                    specs.append(CovSpec(
                        family=fam, d=d, n=n, grid_size=GRIDS[i % len(GRIDS)],
                        alpha_target=float(a) + float(rng.uniform(-0.045, 0.045)),
                        rho=RHO_THIN[i % len(RHO_THIN)],
                        profile=PROFILES[i % len(PROFILES)],
                        spacing=SPACINGS[(i // 3) % len(SPACINGS)],
                        arms=TARGETED_SPIDER_ARMS[i % len(TARGETED_SPIDER_ARMS)],
                        rep=r, group="A_targeted"))
                    i += 1

    # --- Block B ------------------------------------------------------------
    for n in N_LADDER:
        for d in D_LADDER:
            for rho in (0.4, 1.0, 2.5, 6.0):
                for r in range(max(1, reps - 1)):
                    specs.append(CovSpec(
                        family=free_fams[i % len(free_fams)], d=d, n=n,
                        grid_size=GRIDS[i % len(GRIDS)],
                        alpha_target=float(rng.uniform(1.55, 1.99)),
                        rho=rho, profile=PROFILES[i % len(PROFILES)],
                        spacing=SPACINGS[(i // 3) % len(SPACINGS)],
                        arms=FREE_SPIDER_ARMS[i % len(FREE_SPIDER_ARMS)],
                        rep=r, group="B_thick"))
                    i += 1

    # --- Block C ------------------------------------------------------------
    for n in N_LADDER:
        for d in D_LADDER:
            for mix in (0.15, 0.35, 0.6, 0.85):
                for r in range(max(1, reps - 1)):
                    specs.append(CovSpec(
                        family=free_fams[i % len(free_fams)], d=d, n=n,
                        grid_size=GRIDS[i % len(GRIDS)],
                        alpha_target=float(rng.uniform(1.40, 1.99)),
                        rho=RHO_THIN[i % len(RHO_THIN)], mix=mix,
                        profile=PROFILES[i % len(PROFILES)],
                        spacing=SPACINGS[(i // 3) % len(SPACINGS)],
                        arms=FREE_SPIDER_ARMS[i % len(FREE_SPIDER_ARMS)],
                        rep=r, group="C_mix"))
                    i += 1

    return specs


def poly_plan(reps: int = 3, seed: int = 20260814) -> List[CovSpec]:
    """Block E: piecewise-linear skeletons with uneven per-segment density.

    Added after the main blocks because density heterogeneity along the route
    is a real property of real point sets and none of the four smooth families
    varies it.  Targeted on alpha like block A, so it widens the corpus without
    disturbing the uniform coverage.
    """
    rng = np.random.default_rng(seed)
    specs: List[CovSpec] = []
    i = 0
    for n in N_LADDER:
        for d in (2, 3, 5, 10, 20, 50):
            for a in ALPHA_BINS[::2]:                  # every other bin
                for r in range(reps):
                    specs.append(CovSpec(
                        family="polyline", d=d, n=n,
                        grid_size=GRIDS[i % len(GRIDS)],
                        alpha_target=float(a) + float(rng.uniform(-0.045, 0.045)),
                        rho=RHO_THIN[i % len(RHO_THIN)],
                        profile=PROFILES[i % len(PROFILES)],
                        spacing=SPACINGS[(i // 3) % len(SPACINGS)],
                        segs=(2, 3, 4, 6)[i % 4],
                        density=(0.3, 1.0, 4.0)[i % 3],
                        rep=r, group="E_polyline"))
                    i += 1
    return specs


def topup_plan(reps: int = 1, seed: int = 20260813) -> List[CovSpec]:
    """Level the achieved-alpha histogram after the main run.

    The placement predictor is an upper bound, so a targeted instance can land
    a little below its bin.  Rather than argue about how tight the bound is,
    this reads what was actually achieved from the written solutions and fills
    whichever (n decade, alpha bin) cells came out short, using the two
    families whose placement was measured tight (``arc`` and ``fourier``).
    Coverage is therefore verified on solved alpha, never claimed from a
    parameter.
    """
    from collections import Counter

    rng = np.random.default_rng(seed)
    got: "Counter[Any]" = Counter()
    for path in glob.glob(os.path.join(COV_SOLUTIONS_DIR, "*.sol.json")):
        with open(path) as f:
            sol = json.load(f)
        a = sol.get("alpha")
        if a is None:
            continue
        stem = os.path.basename(path)
        n = int(stem.split("-n")[1].split("-")[0])
        got[(_n_decade(n), min(9, int((a - 1.0) * 10)))] += 1

    # Level towards the median cell, not the fullest one: the fullest cell is
    # wherever the untargeted blocks happened to pile up, and chasing it would
    # multiply the corpus for no coverage gain.
    cells = [got.get((dec, b), 0) for dec in DECADE_REPRESENTATIVE for b in range(10)]
    target = int(np.median(cells)) if cells else 0
    print(f"[topup] cell counts min {min(cells, default=0)} median {target} "
          f"max {max(cells, default=0)}")

    specs: List[CovSpec] = []
    i = 0
    for dec, n_reps in DECADE_REPRESENTATIVE.items():
        for b in range(10):
            short = target - got.get((dec, b), 0)
            # ``rep`` counts draws WITHIN the cell. Deriving it from the global
            # counter instead lets two draws share every field: the parameter
            # rotations have a common period of 18, the alpha jitter is rounded
            # to three places in the name, and the dry run duly produced
            # colliding names.
            for k in range(max(0, min(short, TOPUP_CAP))):
                specs.append(CovSpec(
                    family=("arc", "fourier")[i % 2],
                    d=D_LADDER[i % len(D_LADDER)], n=n_reps[i % len(n_reps)],
                    grid_size=GRIDS[i % len(GRIDS)],
                    alpha_target=1.0 + 0.1 * b + float(rng.uniform(0.02, 0.08)),
                    rho=RHO_THIN[i % len(RHO_THIN)],
                    profile=PROFILES[i % len(PROFILES)],
                    spacing=SPACINGS[(i // 3) % len(SPACINGS)],
                    rep=1000 + k, group="D_topup"))
                i += 1
    return specs


DECADE_REPRESENTATIVE = {"20-99": (30, 50), "100-299": (100, 200),
                         "300-999": (300, 500, 700), "1000+": (1000,)}
TOPUP_CAP = 120


def _n_decade(n: int) -> str:
    if n < 100:
        return "20-99"
    if n < 300:
        return "100-299"
    if n < 1000:
        return "300-999"
    return "1000+"


PLANS = {"pilot": pilot_plan, "main": main_plan, "poly": poly_plan,
         "topup": topup_plan}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(specs: List[CovSpec], workers: int, concorde_timeout_s: int,
        lkh_time_limit_s: int, out_json: str, write: bool = True) -> List[Dict[str, Any]]:
    os.makedirs(COV_INSTANCES_DIR, exist_ok=True)
    os.makedirs(COV_SOLUTIONS_DIR, exist_ok=True)

    # Compile the numba kernels once, single-threaded, so the worker threads do
    # not all race into first-call compilation together.
    _pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    compute_distance_matrix(_pts, 1.0)
    mst_length(_pts)

    todo = [s for s in specs if not already_done(s)]
    print(f"[coverage] {len(specs)} specs, {len(specs) - len(todo)} already done, "
          f"{len(todo)} to run on {workers} workers")
    if not todo:
        return []

    lock = threading.Lock()
    records: List[Dict[str, Any]] = []
    done = [0]
    t0 = time.time()

    def work(spec: CovSpec) -> Dict[str, Any]:
        try:
            return build_and_solve(spec, concorde_timeout_s, lkh_time_limit_s, write=write)
        except Exception as exc:                       # never kill the batch
            return {"name": spec.name, "family": spec.family, "d": spec.d,
                    "n": spec.n, "grid_size": spec.grid_size, "written": False,
                    "integrity_ok": False, "hk_ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, s): s for s in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                records.append(rec)
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == len(todo):
                    el = time.time() - t0
                    ok = sum(1 for r in records if r.get("written"))
                    print(f"  {done[0]}/{len(todo)}  written={ok}  "
                          f"{el/60:.1f} min  eta {el/max(done[0],1)*(len(todo)-done[0])/60:.1f} min",
                          flush=True)

    with open(out_json, "w") as f:
        json.dump(records, f, indent=1)
    return records


def summarise(records: List[Dict[str, Any]]) -> None:
    ok = [r for r in records if r.get("written")]
    bad = [r for r in records if not r.get("written")]
    print(f"\n[coverage] written {len(ok)} / {len(records)}")
    if bad:
        from collections import Counter
        why = Counter()
        for r in bad:
            if r.get("error"):
                why[r["error"].split(":")[0]] += 1
            elif not r.get("hk_ok", True):
                why["hk_bound_above_cost"] += 1
            else:
                why["+".join(r.get("failed_gates") or ["unknown"])] += 1
        print("  refusals:", dict(why))
    if not ok:
        return
    from collections import Counter
    solvers = Counter(r["solver_used"] for r in ok)
    heur = sum(v for k, v in solvers.items() if k not in EXACT_SOLVERS)
    print(f"  solvers: {dict(solvers)}  -- heuristic labels {heur} "
          f"({heur / max(len(ok), 1) * 100:.1f}%)")

    a = np.array([r["alpha"] for r in ok])
    p = np.array([r["alpha_pred"] for r in ok])
    thin = np.array([r["rho"] <= 0.2 and r["mix"] == 0.0 for r in ok])
    print(f"  alpha: min {a.min():.3f}  med {np.median(a):.3f}  max {a.max():.3f}")
    if thin.any():
        err = a[thin] - p[thin]
        print(f"  thin-set predictor 1+kappa: n={int(thin.sum())} "
              f"mean err {err.mean():+.4f}  sd {err.std():.4f}  "
              f"max |err| {np.abs(err).max():.4f}")
    hk = np.array([r["hk_ratio"] for r in ok])
    print(f"  HK 1-tree / cost: min {hk.min():.4f}  med {np.median(hk):.4f}  max {hk.max():.4f}")
    hist, edges = np.histogram(a, bins=10, range=(1.0, 2.0))
    print("  alpha histogram [1,2] in 10 bins:", hist.tolist())


EXACT_SOLVERS = ("concorde", "held_karp_exact")


def drop_heuristic_labels() -> int:
    """Delete every written instance whose label came from LKH.

    Those labels exist only because wsl.exe refused to launch Concorde under
    16-way concurrency, not because the instances are hard -- see
    ``CONCORDE_CONCURRENCY``. Re-running the plan afterwards re-solves them
    through the gated Concorde path, or through the exact DP at n <= 20, since
    ``already_done`` only skips instances that still have both files.
    """
    removed = 0
    for path in sorted(glob.glob(os.path.join(COV_SOLUTIONS_DIR, "*.sol.json"))):
        with open(path) as f:
            sol = json.load(f)
        if sol.get("optimal_solver") in EXACT_SOLVERS:
            continue
        name = sol["instance_name"]
        os.remove(path)
        inst = os.path.join(COV_INSTANCES_DIR, f"{name}.json")
        if os.path.exists(inst):
            os.remove(inst)
        removed += 1
    print(f"[coverage] dropped {removed} heuristic (LKH) labels for re-solve")
    return removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS), default="pilot")
    ap.add_argument("--repair-heuristic", action="store_true",
                    help="delete all LKH-labelled instances before running so "
                         "they are re-solved exactly")
    ap.add_argument("--pilot", action="store_true", help="alias for --plan pilot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--concorde-timeout", type=int, default=300)
    ap.add_argument("--lkh-time-limit", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.repair_heuristic:
        drop_heuristic_labels()

    plan = "pilot" if args.pilot else args.plan
    specs = PLANS[plan](**({"reps": args.reps}
                           if plan in ("main", "poly", "topup") else {}))
    print(f"[coverage] plan={plan} specs={len(specs)}")
    assert_disjoint(specs)
    if args.dry_run:
        from collections import Counter
        print(Counter(s.family for s in specs))
        print(Counter(s.group for s in specs))
        return

    out = args.out or os.path.join(COV_ROOT, f"{plan}_records.json")
    os.makedirs(COV_ROOT, exist_ok=True)
    recs = run(specs, args.workers, args.concorde_timeout, args.lkh_time_limit, out)
    summarise(recs)
    print(f"[coverage] on disk: {assert_no_leakage()} instances, no leakage")


if __name__ == "__main__":
    main()
