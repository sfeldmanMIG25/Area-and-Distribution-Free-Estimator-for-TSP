"""Held--Karp 1-tree lower bound, with the iteration budget exposed as a knob.

Why this module exists
----------------------
The manuscript positions GART 2.0 as occupying a middle ground between the
classical closed forms and the Held--Karp 1-tree bound. Only the lower anchor
was ever measured. This module supplies the upper one, as a scored estimator
that runs through the same benchmark harness as everything else.

The bound
---------
Fix a special node ``s``. A *1-tree* is a spanning tree on ``V \\ {s}`` plus the
two cheapest edges incident to ``s``; it has ``n`` edges and total degree ``2n``.
Every Hamiltonian tour is a 1-tree, so the minimum 1-tree is a relaxation.

Introduce node penalties ``pi`` and the transformed distances

    d_pi(i, j) = d(i, j) + pi_i + pi_j.

Under ``d_pi`` every tour costs exactly ``L_tour + 2 * sum(pi)``, because every
node has degree two. Minimising over the larger 1-tree family therefore gives,
for **any** real vector ``pi``,

    w(pi) = L_1tree(pi) - 2 * sum(pi)  <=  OPT.                    (Held & Karp)

``w`` is concave and piecewise linear in ``pi``, and a subgradient at ``pi`` is

    g_i = deg_i(1-tree under pi) - 2,

so ``max_pi w(pi)`` -- the Held--Karp bound -- is approached by subgradient
ascent. Note ``sum(g) = 0`` and ``g_s = 0`` identically, since the special node
has degree exactly two in every 1-tree by construction.

Validity does **not** require ``pi >= 0`` or ``d_pi >= 0``. The Lagrangian
argument above is sign-free, and Prim's algorithm is correct on negative edge
weights (unlike a shortest-path method). No clamping is applied.

References
----------
Held, M. and Karp, R.M. (1970). "The traveling-salesman problem and minimum
    spanning trees." *Operations Research* 18(6), 1138--1162.
    -- the 1-tree relaxation and the bound itself.
Held, M. and Karp, R.M. (1971). "The traveling-salesman problem and minimum
    spanning trees: Part II." *Mathematical Programming* 1(1), 6--25.
    -- the ascent, and the subgradient ``deg - 2``.
Volgenant, T. and Jonker, R. (1982). "A branch and bound algorithm for the
    symmetric traveling salesman problem based on the 1-tree relaxation."
    *European Journal of Operational Research* 9(1), 83--89.
    -- the upper-bound-free step-size schedule implemented in
    :func:`_ascend`: initial step ``L_1tree(0) / (2n)``, iterations grouped
    into periods, period doubling on a late improvement, step halving on a
    barren period. Exact recurrence is written out at the call site so a
    reader can check it against the paper rather than trusting this comment.
Helsgaun, K. (2000). "An effective implementation of the Lin--Kernighan
    traveling salesman heuristic." *EJOR* 126(1), 106--130, section 4.
    -- the ``0.7 g_k + 0.3 g_{k-1}`` direction smoothing applied on top of the
    Volgenant--Jonker schedule, and the doubling of the step alongside the
    period. Both are refinements of the V&J ascent, not replacements for it.

WHY THE PROJECT'S DELAUNAY MST FAST PATH CANNOT BE USED HERE
------------------------------------------------------------
``mst_utils.compute_mst`` computes an *exact Euclidean* MST, and for ``d <= 3``
it does so by extracting the MST from the Delaunay triangulation. That shortcut
rests on a theorem about the Euclidean metric: EMST is a subgraph of the
Delaunay graph, because for any non-Delaunay pair ``(u, v)`` some ``w`` lies in
the disk on ``uv`` as diameter, giving ``d(u,w) < d(u,v)`` and ``d(w,v) <
d(u,v)``, so ``(u,v)`` fails the cycle property.

That argument does not survive the penalty transform. Under ``d_pi`` the same
comparison becomes ``d(u,w) + pi_w < d(u,v) + pi_v``, which is false whenever
``pi_w`` exceeds ``pi_v`` by more than ``d(u,v) - d(u,w)``. ``d_pi`` is not a
Euclidean metric on any point set -- it is a Euclidean metric plus a node
potential -- and the MST genuinely moves under it. It has to: a spanning tree
``T`` costs ``L(T) + sum_i deg_T(i) pi_i``, so if the tree never changed the
subgradient would never change and the ascent would do nothing.

Restricting the ascent to the ``pi = 0`` Delaunay edge set is therefore not a
speed/accuracy trade -- it is a correctness failure with the dangerous sign.
An MST over a subgraph is *at least* as heavy as the true MST, so the
restricted ``w`` **over**-estimates ``w(pi)``, and an over-estimated lower bound
can exceed OPT. It is measured, not asserted: see
``paper_tooling/hk1tree_validate.py --delaunay-trap``.

What is used instead: an exact Prim over the complete graph under ``d_pi``,
recomputed every iteration. ``O(n^2)`` time per iteration, ``O(k * n^2)`` for a
``k``-iteration ascent, against ``O(n log n)`` for one Delaunay MST. That
asymptotic cost is the honest price of the upper anchor and is the quantity the
cost study is meant to report.

``mst_utils.compute_mst`` *is* still used, for the ``pi = 0`` plain-MST
reference that the ``w(0) >= L_MST`` invariant is checked against.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap


#: Largest ``n`` for which the full float64 distance matrix is materialised.
#: ``n^2 * 8`` bytes; 16384 is ~2.1 GB. Above this the coordinate kernel runs
#: instead, trading the stored matrix for a recomputed square root per pair.
DENSE_MAX_N = 16384

#: Volgenant--Jonker period length, in iterations. Fixed: it must depend on
#: neither ``n`` nor the iteration budget. See :func:`_ascend` for the measured
#: comparison against V&J's own ``n / 2``.
INITIAL_PERIOD = 32

#: Relative tolerance for "did this iterate improve the incumbent". Guards the
#: period bookkeeping against float64 noise in ``L_1tree - 2 sum(pi)``, which
#: is a difference of two quantities that both grow with ``|pi|``.
_IMPROVE_RTOL = 1e-12


# ---------------------------------------------------------------------------
# 1-tree kernels
# ---------------------------------------------------------------------------
# Both kernels return (L_1tree(pi), degrees). ``degrees`` sums to 2n: n-2 edges
# from the spanning tree on V \ {s}, plus the two edges at s.


@njit(cache=True)
def _one_tree_dense(D, pi, special):
    """Minimum 1-tree under ``d_pi`` from a precomputed distance matrix."""
    n = D.shape[0]
    INF = 1.0e300
    in_tree = np.zeros(n, dtype=np.bool_)
    best = np.full(n, INF, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    degrees = np.zeros(n, dtype=np.int64)

    # The special node is excluded from the spanning-tree phase.
    in_tree[special] = True
    start = 1 if special == 0 else 0
    best[start] = 0.0

    total = 0.0
    for _ in range(n - 1):
        u = -1
        b = INF
        for i in range(n):
            if (not in_tree[i]) and best[i] < b:
                b = best[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True
        pu = parent[u]
        if pu >= 0:
            total += b
            degrees[u] += 1
            degrees[pu] += 1
        piu = pi[u]
        for v in range(n):
            if in_tree[v]:
                continue
            w = D[u, v] + piu + pi[v]
            if w < best[v]:
                best[v] = w
                parent[v] = u

    # Two cheapest edges at the special node.
    b1 = INF
    b2 = INF
    i1 = -1
    i2 = -1
    pis = pi[special]
    for v in range(n):
        if v == special:
            continue
        w = D[special, v] + pis + pi[v]
        if w < b1:
            b2 = b1
            i2 = i1
            b1 = w
            i1 = v
        elif w < b2:
            b2 = w
            i2 = v
    total += b1 + b2
    degrees[special] += 2
    degrees[i1] += 1
    degrees[i2] += 1
    return total, degrees


@njit(cache=True)
def _one_tree_coords(X, pi, special):
    """Same 1-tree, distances recomputed from coordinates. ``O(n)`` memory."""
    n = X.shape[0]
    d = X.shape[1]
    INF = 1.0e300
    in_tree = np.zeros(n, dtype=np.bool_)
    best = np.full(n, INF, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    degrees = np.zeros(n, dtype=np.int64)

    in_tree[special] = True
    start = 1 if special == 0 else 0
    best[start] = 0.0

    total = 0.0
    for _ in range(n - 1):
        u = -1
        b = INF
        for i in range(n):
            if (not in_tree[i]) and best[i] < b:
                b = best[i]
                u = i
        if u == -1:
            break
        in_tree[u] = True
        pu = parent[u]
        if pu >= 0:
            total += b
            degrees[u] += 1
            degrees[pu] += 1
        piu = pi[u]
        for v in range(n):
            if in_tree[v]:
                continue
            s = 0.0
            for k in range(d):
                diff = X[v, k] - X[u, k]
                s += diff * diff
            w = math.sqrt(s) + piu + pi[v]
            if w < best[v]:
                best[v] = w
                parent[v] = u

    b1 = INF
    b2 = INF
    i1 = -1
    i2 = -1
    pis = pi[special]
    for v in range(n):
        if v == special:
            continue
        s = 0.0
        for k in range(d):
            diff = X[v, k] - X[special, k]
            s += diff * diff
        w = math.sqrt(s) + pis + pi[v]
        if w < b1:
            b2 = b1
            i2 = i1
            b1 = w
            i1 = v
        elif w < b2:
            b2 = w
            i2 = v
    total += b1 + b2
    degrees[special] += 2
    degrees[i1] += 1
    degrees[i2] += 1
    return total, degrees


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OneTreeResult:
    """Outcome of one ascent.

    Attributes
    ----------
    bound : float
        ``max_k w(pi_k)`` over every iterate visited -- the incumbent, not the
        final iterate. Subgradient ascent is not monotone in ``w(pi_k)``; only
        the incumbent is, and only the incumbent is reported.
    initial_bound : float
        ``w(0)``, the minimum 1-tree at zero penalties. Always ``>= L_MST``.
    iterations_used : int
        1-tree evaluations after the ``pi = 0`` one. Zero when the caller asked
        for zero, or when the ascent terminated early.
    is_optimal : bool
        True when some 1-tree had every degree equal to two. Such a 1-tree is a
        tour, so weak duality closes and ``bound == OPT`` exactly.
    """

    bound: float
    initial_bound: float
    iterations_requested: int
    iterations_used: int
    is_optimal: bool
    backend: str
    n: int


# ---------------------------------------------------------------------------
# Ascent
# ---------------------------------------------------------------------------


def _ascend(evaluate, n: int, iterations: int, special: int, backend: str) -> OneTreeResult:
    """Volgenant--Jonker subgradient ascent on ``w``.

    ``evaluate(pi) -> (L_1tree(pi), degrees)``.

    Schedule, written out so it can be checked against Volgenant & Jonker
    (1982) and Helsgaun (2000) section 4:

    * ``t_0 = w(0) / (2n)``. V&J scale the first step by the initial 1-tree
      length; no upper bound on OPT is needed anywhere, which is the property
      that makes this schedule usable as a standalone anchor.
    * Iterations are grouped into periods of length ``p``, and within a period
      the step decreases arithmetically to zero across the period:
      ``t_j = t_period * (p - j) / p`` for ``j = 0 .. p - 1``. The arithmetic
      within-period decrease is the part of V&J that does the work; holding
      ``t`` flat across a period and only adapting at the boundary is a
      different (LKH-style) ascent and converges far more slowly here.
    * Improvement on the **last** iteration of a period doubles both the period
      and the step -- the ascent is still climbing, so lengthen the stride.
    * A period with **no** improvement halves the step and the period.
    * Otherwise both are carried over unchanged.
    * Stop on exhausted budget, ``t`` underflow, or a zero subgradient.

    TWO CONSTANTS THAT ARE MEASURED, NOT QUOTED
    -------------------------------------------
    ``period_0 = 32``, a constant, where V&J specify ``n / 2``. V&J assume an
    ascent run to convergence; this one is run to a *budget*, and at ``n = 1000``
    a first period of 500 spends the entire budget at the initial step size.
    Measured mean gap to the optimum over 30 planar instances,
    ``n in [100, 1000]``, at budgets ``k = 10 / 50 / 100 / 500 / 1000``:

        period_0 = n/2, flat step  : 11.51 / 10.93 / 10.50 / 4.16 / 1.68 %
        period_0 = n/2, decreasing : 11.02 /  7.81 /  5.76 / 0.97 / 0.94 %
        period_0 = 32,  decreasing :  8.94 /  2.14 /  1.60 / 1.16 / 1.01 %

    The deviation from ``n / 2`` is therefore an evidence-backed choice for the
    budgeted regime, not a misreading of the source. Both constants are fixed
    here and depend on neither ``n`` nor ``k``.

    WHY THE SCHEDULE MUST NOT SEE THE BUDGET
    ----------------------------------------
    Budget-independence is what makes the bound exactly monotone in ``k``: the
    length-``k`` trajectory is a strict prefix of the length-``k'`` one for
    every ``k' > k``, so the incumbent is a maximum over a nested family. An
    earlier draft scaled ``period_0`` by ``k`` so small budgets would adapt the
    step sooner. It returned a *better* bound at ``k = 10`` than at ``k = 100``
    on uniform ``n = 200`` (9958.7 against 9475.2), because the two budgets
    were descending different trajectories. Budget-coupled scheduling and
    monotonicity in the budget cannot both hold; monotonicity wins.

    The direction is ``0.7 g_k + 0.3 g_{k-1}`` (Helsgaun), which damps the
    oscillation between adjacent 1-trees that plain ``g_k`` produces.
    """
    pi = np.zeros(n, dtype=np.float64)
    length, degrees = evaluate(pi)
    w0 = float(length)  # sum(pi) == 0
    w_best = w0

    if iterations <= 0:
        return OneTreeResult(w_best, w0, iterations, 0, False, backend, n)

    t = w0 / (2.0 * n)
    if not (t > 0.0) or not np.isfinite(t):
        # Degenerate geometry (all points coincident). Nothing to ascend.
        return OneTreeResult(w_best, w0, iterations, 0, False, backend, n)

    period = INITIAL_PERIOD
    g_prev = np.zeros(n, dtype=np.float64)
    first_step = True
    used = 0

    while used < iterations and t > 0.0:
        improved_anywhere = False
        improved_last = False
        # The budget may cut a period short, but the decay below divides by the
        # full ``period``, never by ``steps`` -- otherwise truncating the tail
        # would change the steps that come before it and break prefix nesting.
        steps = min(period, iterations - used)
        for j in range(steps):
            g = degrees.astype(np.float64) - 2.0
            if not np.any(g):
                # Every degree is two: the 1-tree is a tour, so the relaxation
                # is tight and w(pi) == OPT.
                return OneTreeResult(w_best, w0, iterations, used, True, backend, n)

            direction = g if first_step else 0.7 * g + 0.3 * g_prev
            first_step = False
            g_prev = g
            pi = pi + (t * (period - j) / period) * direction

            length, degrees = evaluate(pi)
            w = float(length) - 2.0 * float(pi.sum())
            used += 1

            if w > w_best + _IMPROVE_RTOL * max(1.0, abs(w_best)):
                w_best = w
                improved_anywhere = True
                improved_last = j == steps - 1
            else:
                improved_last = False

        if improved_last:
            period *= 2
            t *= 2.0
        elif not improved_anywhere:
            t *= 0.5
            period = max(1, period // 2)

    return OneTreeResult(w_best, w0, iterations, used, False, backend, n)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _check_iterations(iterations: int) -> int:
    if iterations is None:
        raise ValueError("iterations is required; it is the cost knob, not an option")
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    return iterations


def one_tree_bound_from_matrix(
    dist_matrix: np.ndarray, iterations: int, special: int = 0
) -> OneTreeResult:
    """Held--Karp bound from an explicit symmetric distance matrix.

    Works for any symmetric distances, metric or not -- the relaxation argument
    never uses the triangle inequality. This is the path for TSPLIB EXPLICIT,
    GEO and ATT instances, which have no usable Euclidean coordinates.
    """
    iterations = _check_iterations(iterations)
    D = np.ascontiguousarray(dist_matrix, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"dist_matrix must be square, got shape {D.shape}")
    n = D.shape[0]
    if n < 3:
        raise ValueError(f"a 1-tree needs at least 3 nodes, got n={n}")
    if not np.isfinite(D).all():
        raise ValueError("dist_matrix contains non-finite entries")
    if not (0 <= special < n):
        raise ValueError(f"special node {special} out of range for n={n}")

    return _ascend(
        lambda pi: _one_tree_dense(D, pi, special), n, iterations, special, "matrix"
    )


def one_tree_bound(
    coords: np.ndarray, iterations: int, special: int = 0, dense: Optional[bool] = None
) -> OneTreeResult:
    """Held--Karp bound for Euclidean points.

    ``dense`` forces the precomputed-matrix kernel on or off; the default picks
    the matrix when ``n <= DENSE_MAX_N`` and the coordinate kernel otherwise.
    Both compute the same 1-tree; only the memory/arithmetic trade differs.
    """
    iterations = _check_iterations(iterations)
    X = np.ascontiguousarray(coords, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"coords must be 2-D (n, d), got shape {X.shape}")
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"a 1-tree needs at least 3 nodes, got n={n}")
    if not np.isfinite(X).all():
        raise ValueError("coords contain non-finite entries")
    if not (0 <= special < n):
        raise ValueError(f"special node {special} out of range for n={n}")

    use_dense = (n <= DENSE_MAX_N) if dense is None else bool(dense)
    if use_dense:
        # (a-b)^2 expansion is ~5x faster than cdist here and the residual
        # error is irrelevant against a bound reported to 4 significant digits;
        # the clip removes the small negatives it can produce on the diagonal.
        sq = (X * X).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(D, 0.0, out=D)
        np.sqrt(D, out=D)
        np.fill_diagonal(D, 0.0)
        return _ascend(
            lambda pi: _one_tree_dense(D, pi, special), n, iterations, special, "dense"
        )
    return _ascend(
        lambda pi: _one_tree_coords(X, pi, special), n, iterations, special, "coords"
    )


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class HeldKarp1Tree:
    """Benchmark-scored Held--Karp 1-tree bound.

    Matches the estimator contract the runners call:
    ``estimate(coordinates, dimension, grid_size=None) -> dict`` with
    ``estimate``, ``feature_time``, ``inference_time``, ``status``.

    ``features_required`` is empty on purpose. This estimator consumes the
    point set itself, not a scalar feature dict, so there is no name to
    declare. It is present because the TSPLIB runner probes it on every model,
    and an empty list is the truthful answer rather than an absent attribute.

    Nothing here reads a value out of a dict with a default. Every input is
    validated and a bad one raises -- the project has shipped two defects from
    ``feats.get(k, 0.0)`` quietly substituting zero for an absent input.

    ``iterations`` is the whole point of the model: it is the cost knob. Each
    iteration is one exact ``O(n^2)`` Prim, so cost is ``O(k n^2)`` and the
    bound tightens monotonically in ``k`` (the incumbent is monotone by
    construction; the raw iterate is not -- see :func:`_ascend`).
    """

    def __init__(self, iterations: int = 100, special: int = 0):
        self.iterations = _check_iterations(iterations)
        if int(special) < 0:
            raise ValueError(f"special must be >= 0, got {special}")
        self.special = int(special)
        self.name = f"HK_1Tree_{self.iterations}"
        self.features_required: list[str] = []

    def __repr__(self) -> str:  # pragma: no cover
        return f"HeldKarp1Tree(iterations={self.iterations})"

    def estimate(self, coordinates, dimension=None, grid_size=None) -> dict:
        """``grid_size`` is accepted for call-site compatibility and unused --
        the bound is computed from the points and needs no region measure."""
        if coordinates is None:
            raise ValueError("coordinates is required")
        t0 = time.perf_counter()
        # Deduplicate to match the convention of every other estimator here.
        # Safe for a lower bound: in a metric space a duplicate point is
        # visited in place at zero cost, so OPT is unchanged by the removal.
        coords = np.unique(np.asarray(coordinates, dtype=np.float64), axis=0)
        n = coords.shape[0]
        if n < 3:
            return {
                "estimate": 0.0,
                "feature_time": time.perf_counter() - t0,
                "inference_time": 0.0,
                "status": "degenerate_n",
            }
        t_feat = time.perf_counter() - t0

        t1 = time.perf_counter()
        res = one_tree_bound(coords, self.iterations, special=self.special)
        return {
            "estimate": float(res.bound),
            "feature_time": t_feat,
            "inference_time": time.perf_counter() - t1,
            "status": "ok",
            "hk_initial_bound": res.initial_bound,
            "hk_iterations_used": res.iterations_used,
            "hk_is_optimal": res.is_optimal,
        }

    def estimate_from_matrix(self, dist_matrix) -> dict:
        """Non-Euclidean path: score EXPLICIT / GEO / ATT TSPLIB instances."""
        if dist_matrix is None:
            raise ValueError("dist_matrix is required")
        t1 = time.perf_counter()
        res = one_tree_bound_from_matrix(
            dist_matrix, self.iterations, special=self.special
        )
        return {
            "estimate": float(res.bound),
            "feature_time": 0.0,
            "inference_time": time.perf_counter() - t1,
            "status": "ok",
            "hk_initial_bound": res.initial_bound,
            "hk_iterations_used": res.iterations_used,
            "hk_is_optimal": res.is_optimal,
        }


#: Iteration budgets registered as separate scored models, so the cost/accuracy
#: trade appears as a curve in the benchmark output rather than a single point.
ITERATION_LADDER = (0, 10, 50, 100, 500, 1000)

ESTIMATORS: dict[str, HeldKarp1Tree] = {
    f"HK_1Tree_{k}": HeldKarp1Tree(iterations=k) for k in ITERATION_LADDER
}
