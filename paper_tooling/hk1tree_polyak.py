"""Held--Karp 1-tree bound under a Polyak step rule, with a constructive
upper bound built from the coordinates alone.

WHY THIS EXISTS
---------------
``held_karp_1tree._ascend`` uses the Volgenant--Jonker schedule, whose only
adaptive move on a barren period is ``t *= 0.5``. There is no restart and no
floor, so ``t`` walks down to a float64 denormal and the loop's ``t > 0.0``
guard fires. Starting from ``t_0 = w(0) / (2n)`` that takes roughly

    63 iterations   (period 32 -> 16 -> 8 -> 4 -> 2 -> 1, one barren period each)
  + ~1080 halvings  (until t underflows to exactly 0.0)
  = ~1143 iterations,

which is precisely where ``iterations_used`` pins in the shipped ND sweep for
every ``d >= 15`` (measured median 1141--1143, against a requested budget of
2000). The high-dimensional plateau in ``hk1tree_frontier_nd_by_dimension.csv``
is therefore a step-size underflow, not a converged bound, and no dimensional
statement may be read off it.

THE POLYAK STEP
---------------
For a concave ``w`` with subgradient ``g`` at ``pi``, the Polyak step towards a
known target value ``UB >= max w`` is

    pi <- pi + gamma * (UB - w(pi)) / ||g||^2 * g,          gamma in (0, 2).

It is scale-free: the step shrinks automatically as ``w`` approaches ``UB``,
which is exactly the behaviour the V&J schedule has to fake with a halving
counter. ``gamma`` is halved after a run of barren iterations, so the sequence
is square-summable-ish in practice while remaining budget-independent.

    Polyak, B.T. (1969). "Minimization of unsmooth functionals." *USSR
        Computational Mathematics and Mathematical Physics* 9(3), 14--29.
    Held, M., Wolfe, P. and Crowder, H.P. (1974). "Validation of subgradient
        optimization." *Mathematical Programming* 6(1), 62--88.
        -- the ``gamma`` halving-on-barren-period rule used here.

WHERE THE UPPER BOUND COMES FROM, AND WHY THE CERTIFICATE SURVIVES IT
---------------------------------------------------------------------
``UB`` is a **nearest-neighbour tour from node 0, improved by 2-opt to local
optimality**, computed from the instance coordinates and nothing else. The
optimal cost is never read. :func:`constructive_upper_bound` is the only source
of ``UB`` in this module and it takes a distance matrix, not a label.

This matters twice over, and the two must not be confused:

1. **Validity.** ``w(pi) = L_1tree(pi) - 2 sum(pi) <= OPT`` holds for *every*
   real vector ``pi``. Nothing in the Lagrangian argument refers to how ``pi``
   was reached. So the bound this module returns is a certificate no matter
   what ``UB`` was -- a wrong ``UB``, a random ``UB``, or the optimum itself
   would all still yield a valid lower bound. Only the *rate* is affected.
2. **Unsupervisedness.** A method that consults the label to pick its step is
   not label-free even when its output is certified, and it may not be compared
   against a trained model as though it were. That is why the constructive tour
   is mandatory here and why the optimum-as-target arm exists only inside
   ``hk1tree_polyak_nd.py --mode ub``, is never used by the sweep, and is
   stamped ``ub_variant = optimum`` and reported as disqualified wherever it
   appears.

MONOTONICITY IN THE BUDGET
--------------------------
The schedule sees neither ``n`` nor ``k``: ``gamma`` depends only on the history
of incumbent improvements. The length-``k`` trajectory is therefore a strict
prefix of the length-``k'`` trajectory for ``k' > k``, and the reported bound is
the incumbent ``max_j w(pi_j)`` over the prefix -- a maximum over a nested
family, hence non-decreasing in ``k``. This is the same property that lets one
ascent be read off at every ladder budget, and it is checked by
``hk1tree_polyak_validate.py --structure`` (invariants I2 and I5).
"""

from __future__ import annotations

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from held_karp_1tree import _one_tree_coords, _one_tree_dense  # noqa: E402

try:
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap


#: Initial Polyak relaxation parameter. 2.0 is the classical upper end of the
#: convergent range ``(0, 2)``; Held--Wolfe--Crowder start there and halve.
GAMMA_0 = 2.0

#: Consecutive iterations without an incumbent improvement that trigger
#: ``gamma *= 0.5``. Fixed: it must depend on neither ``n`` nor the budget, or
#: prefix nesting -- and with it monotonicity in ``k`` -- is lost.
BARREN_LIMIT = 20

#: Below this ``gamma`` the step can no longer move the incumbent at float64
#: precision; the ascent stops and every later checkpoint repeats the incumbent.
#: Unlike the V&J underflow this is an *explicit* floor, is reported in
#: ``stopped_reason``, and is reached only after ~40 halvings.
GAMMA_MIN = 1e-12

#: Relative tolerance for "did this iterate improve the incumbent". Same
#: constant and same role as ``held_karp_1tree._IMPROVE_RTOL``.
IMPROVE_RTOL = 1e-12

#: Cap on 2-opt sweeps in the upper-bound construction. Bounds the constructive
#: cost at ``O(P n^2)``; every sweep past the cap is forfeited tour quality, not
#: correctness, since any tour is a valid ``UB``.
TWO_OPT_MAX_SWEEPS = 50

#: Largest ``n`` for which the float64 distance matrix is materialised here.
DENSE_MAX_N = 20000


# ---------------------------------------------------------------------------
# Constructive upper bound -- coordinates in, tour cost out. No label.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _nn_tour(D):
    """Nearest-neighbour tour from node 0. ``O(n^2)``."""
    n = D.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    tour = np.empty(n, dtype=np.int64)
    visited[0] = True
    tour[0] = 0
    cur = 0
    for i in range(1, n):
        best = 1.0e300
        bj = -1
        for j in range(n):
            if (not visited[j]) and D[cur, j] < best:
                best = D[cur, j]
                bj = j
        visited[bj] = True
        tour[i] = bj
        cur = bj
    return tour


@njit(cache=True)
def _tour_length(D, tour):
    n = tour.shape[0]
    total = 0.0
    for i in range(n):
        total += D[tour[i], tour[(i + 1) % n]]
    return total


@njit(cache=True)
def _two_opt(D, tour, max_sweeps):
    """2-opt to local optimality, full neighbourhood, first improvement.

    Reverses the inner segment in place. ``O(n^2)`` per sweep; the sweep count
    is capped because the upper bound only has to be *good*, not optimal.
    """
    n = tour.shape[0]
    sweeps = 0
    while sweeps < max_sweeps:
        improved = False
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # the two edges coincide; no move exists
                c = tour[j]
                e = tour[(j + 1) % n]
                delta = (D[a, c] + D[b, e]) - (D[a, b] + D[c, e])
                if delta < -1.0e-12:
                    lo = i + 1
                    hi = j
                    while lo < hi:
                        tmp = tour[lo]
                        tour[lo] = tour[hi]
                        tour[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
                    b = tour[i + 1]
        sweeps += 1
        if not improved:
            break
    return sweeps


def constructive_upper_bound(D: np.ndarray, two_opt: bool = True) -> tuple[float, int]:
    """A feasible tour's cost, from the distance matrix alone.

    Returns ``(cost, sweeps)``. The optimal cost is never consulted, here or by
    any caller in the sweep path.
    """
    n = D.shape[0]
    if n < 3:
        raise ValueError(f"a tour needs at least 3 nodes, got n={n}")
    tour = _nn_tour(D)
    sweeps = _two_opt(D, tour, TWO_OPT_MAX_SWEEPS) if two_opt else 0
    return float(_tour_length(D, tour)), int(sweeps)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolyakResult:
    bound: float
    initial_bound: float
    upper_bound: float
    iterations_used: int
    is_optimal: bool
    stopped_reason: str
    gamma_final: float
    n: int


# ---------------------------------------------------------------------------
# Ascent
# ---------------------------------------------------------------------------
def ascend_polyak(evaluate, n: int, budget: int, upper_bound: float,
                  checkpoints: tuple[int, ...] = (),
                  smoothing: float = 0.0) -> tuple[PolyakResult, dict[int, float]]:
    """Polyak subgradient ascent on ``w``, with checkpoint read-off.

    ``evaluate(pi) -> (L_1tree(pi), degrees)``.

    ``smoothing`` is Helsgaun's direction damping: the move direction is
    ``(1 - s) g_k + s g_{k-1}``. The *step length* is always computed from the
    raw subgradient's ``||g||^2``, which is the quantity Polyak's rule refers
    to. ``s = 0`` is plain Polyak and is the default; the sweep records which
    was used.

    ``upper_bound`` must be a feasible tour's cost. It scales the step and
    nothing else -- see the module docstring on why the certificate does not
    depend on it.
    """
    ks = sorted(set(int(k) for k in checkpoints)) if checkpoints else []
    out: dict[int, float] = {}

    pi = np.zeros(n, dtype=np.float64)
    length, degrees = evaluate(pi)
    w0 = float(length)
    w_best = w0
    out[0] = w0

    def finish(used: int, optimal: bool, reason: str, gamma: float):
        for k in ks:
            out.setdefault(k, w_best)
        return (PolyakResult(w_best, w0, float(upper_bound), used, optimal,
                             reason, gamma, n), out)

    if budget <= 0:
        return finish(0, False, "zero_budget", GAMMA_0)
    if not np.isfinite(upper_bound) or upper_bound <= 0.0:
        raise ValueError(f"upper_bound must be finite and positive, got {upper_bound}")

    gamma = GAMMA_0
    barren = 0
    g_prev = np.zeros(n, dtype=np.float64)
    first_step = True
    used = 0

    while used < budget:
        g = degrees.astype(np.float64) - 2.0
        gg = float(g @ g)
        if gg == 0.0:
            # Every degree is two: the 1-tree is a tour, weak duality closes.
            return finish(used, True, "one_tree_is_tour", gamma)

        slack = float(upper_bound) - w_best
        if slack <= 0.0:
            # The incumbent has reached a feasible tour's cost, so it is OPT.
            return finish(used, True, "bound_meets_tour", gamma)

        step = gamma * slack / gg
        direction = g if (first_step or smoothing == 0.0) else \
            (1.0 - smoothing) * g + smoothing * g_prev
        first_step = False
        g_prev = g
        pi = pi + step * direction

        length, degrees = evaluate(pi)
        w = float(length) - 2.0 * float(pi.sum())
        used += 1

        if w > w_best + IMPROVE_RTOL * max(1.0, abs(w_best)):
            w_best = w
            barren = 0
        else:
            barren += 1
            if barren >= BARREN_LIMIT:
                gamma *= 0.5
                barren = 0

        if used in ks:
            out[used] = w_best

        if gamma < GAMMA_MIN:
            return finish(used, False, "gamma_floor", gamma)

    return finish(used, False, "budget_exhausted", gamma)


def _kernel(coords: np.ndarray):
    """``(evaluate, D_or_None, n, backend)`` for a deduplicated point set."""
    X = np.ascontiguousarray(np.unique(np.asarray(coords, dtype=np.float64), axis=0))
    n = X.shape[0]
    if n <= DENSE_MAX_N:
        sq = (X * X).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(D, 0.0, out=D)
        np.sqrt(D, out=D)
        np.fill_diagonal(D, 0.0)
        return (lambda pi: _one_tree_dense(D, pi, 0)), D, n, "dense"
    return (lambda pi: _one_tree_coords(X, pi, 0)), None, n, "coords"


def polyak_bound(coords: np.ndarray, budget: int,
                 checkpoints: tuple[int, ...] = (),
                 smoothing: float = 0.0,
                 two_opt: bool = True,
                 upper_bound: float | None = None) -> dict:
    """Full pipeline for one Euclidean instance: dedup, matrix, UB, ascent.

    ``upper_bound`` overrides the constructive tour. It exists for
    :func:`sensitivity_ub` and must stay ``None`` on every scoring path.
    """
    t0 = time.perf_counter()
    evaluate, D, n, backend = _kernel(coords)
    if n < 3:
        return {"status": "degenerate_n", "n_unique": n}
    t_matrix = time.perf_counter() - t0

    t1 = time.perf_counter()
    if upper_bound is None:
        if D is None:
            raise NotImplementedError("constructive UB needs the dense matrix")
        ub, sweeps = constructive_upper_bound(D, two_opt=two_opt)
        ub_source = "nn_2opt" if two_opt else "nn"
    else:
        ub, sweeps, ub_source = float(upper_bound), 0, "supplied"
    t_ub = time.perf_counter() - t1

    t2 = time.perf_counter()
    res, ckpt = ascend_polyak(evaluate, n, budget, ub, checkpoints, smoothing)
    t_ascent = time.perf_counter() - t2

    return {"status": "ok", "n_unique": n, "backend": backend,
            "bound": res.bound, "initial_bound": res.initial_bound,
            "upper_bound": ub, "ub_source": ub_source, "two_opt_sweeps": sweeps,
            "iterations_used": res.iterations_used, "is_optimal": res.is_optimal,
            "stopped_reason": res.stopped_reason, "gamma_final": res.gamma_final,
            "matrix_seconds": t_matrix, "ub_seconds": t_ub,
            "ascent_seconds": t_ascent, "bounds": ckpt}


def polyak_bound_from_matrix(dist_matrix: np.ndarray, budget: int,
                             checkpoints: tuple[int, ...] = (),
                             smoothing: float = 0.0,
                             two_opt: bool = True,
                             upper_bound: float | None = None) -> dict:
    """Polyak ascent from an explicit symmetric distance matrix.

    The matrix twin of :func:`polyak_bound`, and the path for TSPLIB EXPLICIT,
    GEO and ATT instances, which have no usable Euclidean coordinates. Every
    step of the pipeline is metric-agnostic already: the 1-tree relaxation
    never invokes the triangle inequality (see ``held_karp_1tree`` module
    docstring), nearest-neighbour and 2-opt only read ``D``, and the Polyak
    step needs a feasible tour's cost, which the constructive tour supplies.
    So the certificate ``bound <= OPT`` survives on non-metric input; only the
    *tightness* of the bound depends on how metric the instance is.

    No deduplication is performed. Coincident nodes are a coordinate-space
    notion; in a distance matrix a zero off-diagonal entry is data, and
    collapsing it would change the instance the label was computed on.
    """
    t0 = time.perf_counter()
    D = np.ascontiguousarray(dist_matrix, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"dist_matrix must be square, got shape {D.shape}")
    if not np.isfinite(D).all():
        raise ValueError("dist_matrix contains non-finite entries")
    n = D.shape[0]
    if n < 3:
        return {"status": "degenerate_n", "n_unique": n}
    t_matrix = time.perf_counter() - t0

    t1 = time.perf_counter()
    if upper_bound is None:
        ub, sweeps = constructive_upper_bound(D, two_opt=two_opt)
        ub_source = "nn_2opt" if two_opt else "nn"
    else:
        ub, sweeps, ub_source = float(upper_bound), 0, "supplied"
    t_ub = time.perf_counter() - t1

    t2 = time.perf_counter()
    res, ckpt = ascend_polyak(lambda pi: _one_tree_dense(D, pi, 0), n, budget,
                              ub, checkpoints, smoothing)
    t_ascent = time.perf_counter() - t2

    return {"status": "ok", "n_unique": n, "backend": "matrix",
            "bound": res.bound, "initial_bound": res.initial_bound,
            "upper_bound": ub, "ub_source": ub_source, "two_opt_sweeps": sweeps,
            "iterations_used": res.iterations_used, "is_optimal": res.is_optimal,
            "stopped_reason": res.stopped_reason, "gamma_final": res.gamma_final,
            "matrix_seconds": t_matrix, "ub_seconds": t_ub,
            "ascent_seconds": t_ascent, "bounds": ckpt}


# ---------------------------------------------------------------------------
# Exact optima for validation
# ---------------------------------------------------------------------------
@njit(cache=True)
def _exact_dp(D):
    """Bellman--Held--Karp dynamic program. Exact OPT. ``O(n^2 2^n)``."""
    n = D.shape[0]
    full = 1 << (n - 1)
    INF = 1.0e300
    dp = np.full((full, n - 1), INF, dtype=np.float64)
    for j in range(n - 1):
        dp[1 << j, j] = D[n - 1, j]
    for mask in range(full):
        for j in range(n - 1):
            if dp[mask, j] >= INF:
                continue
            base = dp[mask, j]
            for k in range(n - 1):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                v = base + D[j, k]
                if v < dp[nm, k]:
                    dp[nm, k] = v
    best = INF
    for j in range(n - 1):
        v = dp[full - 1, j] + D[j, n - 1]
        if v < best:
            best = v
    return best


def exact_optimum(coords: np.ndarray) -> float:
    X = np.ascontiguousarray(np.unique(np.asarray(coords, dtype=np.float64), axis=0))
    n = X.shape[0]
    if n > 16:
        raise ValueError(f"exact DP refused at n={n}")
    sq = (X * X).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    np.sqrt(D, out=D)
    np.fill_diagonal(D, 0.0)
    return float(_exact_dp(np.ascontiguousarray(D)))


def brute_force_optimum(coords: np.ndarray) -> float:
    """Exhaustive permutation search. Independent of :func:`exact_optimum`, so
    the two cross-check each other rather than sharing a bug."""
    from itertools import permutations

    X = np.ascontiguousarray(np.unique(np.asarray(coords, dtype=np.float64), axis=0))
    n = X.shape[0]
    if n > 9:
        raise ValueError(f"brute force refused at n={n}")
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    best = math.inf
    for p in permutations(range(1, n)):
        t = (0,) + p
        c = sum(D[t[i], t[(i + 1) % n]] for i in range(n))
        if c < best:
            best = c
    return float(best)
