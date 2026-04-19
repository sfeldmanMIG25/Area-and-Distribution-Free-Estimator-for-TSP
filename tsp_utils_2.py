"""
TSP Academic Estimators Library (tsp_utils_2.py)

Contains a comprehensive set of TSP length estimators:
1. Exact DP (O(n^2 * 2^n)) -- misnamed historically as 'Held-Karp'; retained for
   backward compatibility. Paper uses this only as ground truth for n<=10
   verification, never as an estimator baseline.
2. Constructive (Christofides, Hilbert N-D, MST-Ratio)
3. Geometric/Asymptotic (BHH, Chien, Cavdar, Vinel, Kwon, Daganzo)
4. Simulation (2-Opt, EVT, Basel)
5. Machine Learning (GART 1.0)

Paper-active baselines (Area_Free_Main.tex):
- 2D / TSPLIB common subset: GART 2.0, MST_Ratio, Cavdar, BHH, Chien, Hilbert.
- ND: GART 2.0, MST_Ratio, Hilbert.
- TSPLIB-by-size adds: Kwon, Daganzo.
Functions marked DEPRECATED are retained for historical CSV
reproducibility only and emit a DeprecationWarning on call. They were
cut from the paper under the kill rule: an estimator whose wall time
equals or exceeds the optimal solver wall time at the benchmarked n is
not a valid baseline. The cut set:

- estimate_tsp_held_karp  (ground-truth DP, never a baseline)
- estimate_tsp_christofides
- estimate_tsp_evt
- estimate_tsp_2opt_distribution
- estimate_tsp_basel_willemain
- estimate_tsp_vinel       (redundant with BHH in 2D)
- estimate_tsp_composite   (dominated by GART)

Rules enforced:
- All functions operate on UNIQUE coordinates.
- All geometric functions use float casting to prevent integer overflow/zero-volume bugs.
- Hilbert estimator works in N-dimensions (requires 'hilbertcurve' pkg).
"""

import time
import math
import warnings
import numpy as np
import networkx as nx
import pandas as pd
from math import inf
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.stats import weibull_min
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

import os as _os, sys as _sys
_REPO_ROOT = _os.path.abspath(_os.path.dirname(__file__))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from mst_utils import compute_mst, mst_length as _mst_length

from hilbertcurve.hilbertcurve import HilbertCurve

# --- CONSTANTS ---
BETA_2D = 0.7124
BETA_3D = 0.6979

# Hard caps on geometric primitives that blow up in high dimensions.
# QHull/Delaunay simplex count scales as n^floor(d/2); beyond d≈8 both
# memory and wall-clock become impractical. The Euclidean MST is still
# correct via a dense distance matrix, and volume is safely approximated
# by the bounding-box product (what the original BHH/Vinel/Cavdar
# implementations fell back to for n <= d+1).
# Delaunay (QHull) vs dense cdist-MST crossover on n=1000: Delaunay is faster
# for d in {2, 3}, slower for d >= 4 (measured: d=5 Delaunay 1.8s vs dense 0.2s,
# d=6 Delaunay 13.5s vs dense 0.2s). Keep Delaunay capped at d=3.
DELAUNAY_MAX_DIM = 3
CONVEX_HULL_MAX_DIM = 8

# ====================================================================
# SHARED HELPERS
# ====================================================================

def get_mst_length(nodes_coords):
    """Calculates MST length on UNIQUE coordinates.

    Uses Delaunay triangulation (MST is a subgraph of the Delaunay graph,
    valid in any dimension d) for O(n log n) — but only up to
    :data:`DELAUNAY_MAX_DIM`, because QHull simplex count grows as
    n^floor(d/2). Beyond the cap, falls through to the dense O(n^2 d) matrix
    MST. Falls back to dense MST on Delaunay errors (degenerate points).
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0

    mst_length = _mst_length(coords)
    return mst_length, time.perf_counter() - start_time

def _run_2opt_fast(coords, n, max_iter=2000):
    """Fast 2-opt for simulation estimators."""
    tour = np.random.permutation(n)
    pts = coords[tour]
    cost = np.sum(np.sqrt(np.sum((pts[:-1] - pts[1:])**2, axis=1)))
    cost += np.linalg.norm(pts[-1] - pts[0])
    
    improved = True
    iter_count = 0
    limit = max(max_iter, n * 2)

    while improved and iter_count < limit:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n and i == 0: continue
                p1, p2 = tour[i], tour[i+1]
                p3, p4 = tour[j], tour[(j+1)%n]
                d_current = np.linalg.norm(coords[p1]-coords[p2]) + np.linalg.norm(coords[p3]-coords[p4])
                d_new = np.linalg.norm(coords[p1]-coords[p3]) + np.linalg.norm(coords[p2]-coords[p4])
                if d_new < d_current:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cost += (d_new - d_current)
                    improved = True
                    break 
            if improved: break
    return cost

def _get_random_tour_len(coords, n):
    perm = np.random.permutation(n)
    pts = coords[perm]
    d = np.sum(np.sqrt(np.sum((pts[:-1] - pts[1:])**2, axis=1)))
    d += np.linalg.norm(pts[-1] - pts[0])
    return d

# ====================================================================
# EXACT ESTIMATORS
# ====================================================================

def estimate_tsp_held_karp(nodes_coords):
    """DEPRECATED, not used in paper as a baseline.

    Exact DP (O(n^2 * 2^n)) -- misnamed historically as 'Held-Karp';
    retained for backward compatibility. Paper uses this only as
    ground truth for n <= 10 verification, never as an estimator
    baseline.
    """
    warnings.warn(
        "estimate_tsp_held_karp is deprecated; retained for n<=10 ground truth only.",
        DeprecationWarning, stacklevel=2,
    )
    start_time = time.perf_counter()
    unique_coords = np.unique(nodes_coords, axis=0)
    n = len(unique_coords)
    if n <= 1: return 0.0, 0.0
    if n == 2: return np.linalg.norm(unique_coords[0] - unique_coords[1]) * 2, time.perf_counter() - start_time
        
    dist_matrix = cdist(unique_coords, unique_coords)
    dp = [[inf] * n for _ in range(1 << n)]
    dp[1][0] = 0.0
    
    for r in range(2, n + 1):
        for subset in combinations(range(1, n), r - 1):
            mask = 1
            for bit in subset: mask |= 1 << bit
            for last in subset:
                prev = mask ^ (1 << last)
                best = inf
                temp = prev
                while temp:
                    bit = temp & -temp
                    i = bit.bit_length() - 1
                    temp ^= bit
                    cand = dp[prev][i] + dist_matrix[i][last]
                    if cand < best: best = cand
                dp[mask][last] = best
                
    full = (1 << n) - 1
    ans = inf
    for last in range(1, n):
        cand = dp[full][last] + dist_matrix[last][0]
        if cand < ans: ans = cand
    return ans, time.perf_counter() - start_time

# ====================================================================
# CONSTRUCTIVE ESTIMATORS
# ====================================================================

def estimate_tsp_christofides(nodes_coords):
    """DEPRECATED, cut from paper (wall-time exceeds optimal solver). Christofides 1.5x heuristic tour."""
    warnings.warn(
        "estimate_tsp_christofides is deprecated; cut under the solver-time kill rule.",
        DeprecationWarning, stacklevel=2,
    )
    start_time = time.perf_counter()
    unique_coords = np.unique(nodes_coords, axis=0)
    n = len(unique_coords)
    if n <= 1: return 0.0, 0.0
    
    dist_matrix = cdist(unique_coords, unique_coords)
    mst_csr = compute_mst(unique_coords).to_csr()
    mst_edges = zip(*mst_csr.nonzero())
    
    T = nx.Graph()
    T.add_nodes_from(range(n))
    for u, v in mst_edges:
        w = dist_matrix[u, v]
        T.add_edge(u, v, weight=w)
        
    odd_degree_nodes = [v for v, d in T.degree() if d % 2 == 1]
    subgraph = nx.Graph()
    k = len(odd_degree_nodes)
    for i in range(k):
        u = odd_degree_nodes[i]
        for j in range(i + 1, k):
            v = odd_degree_nodes[j]
            subgraph.add_edge(u, v, weight=dist_matrix[u, v])
            
    # Max weight matching on negated weights = Min weight matching
    for u, v, d in subgraph.edges(data=True): d['weight'] = -d['weight']
    matching = nx.max_weight_matching(subgraph, maxcardinality=True)
    
    M = nx.MultiGraph()
    M.add_nodes_from(range(n))
    M.add_edges_from(T.edges(data=True))
    for u, v in matching:
        w = dist_matrix[u, v]
        M.add_edge(u, v, weight=w)
        
    eulerian_circuit = list(nx.eulerian_circuit(M, source=0))

    visited = [False] * n
    tour = []
    for u, v in eulerian_circuit:
        if not visited[u]:
            visited[u] = True
            tour.append(u)
    if not visited[eulerian_circuit[-1][1]]: tour.append(eulerian_circuit[-1][1])
    
    cost = 0.0
    for i in range(n):
        u, v = tour[i], tour[(i+1)%n]
        cost += dist_matrix[u, v]
    return cost, time.perf_counter() - start_time

def _delaunay_mst_length(coords):
    """Thin shim kept for backwards compatibility — delegates to
    ``mst_utils.compute_mst`` (dense primary, OOM-triggered fallbacks)."""
    return _mst_length(coords)


def _legacy_delaunay_mst_length_unused(coords):
    n = coords.shape[0]
    tri = Delaunay(coords)
    edges = set()
    d = coords.shape[1]
    # simplex has d+1 vertices; enumerate all pairs
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                a, b = simplex[i], simplex[j]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
    rows, cols, dists = [], [], []
    for a, b in edges:
        dist = float(np.linalg.norm(coords[a] - coords[b]))
        rows += [a, b]; cols += [b, a]; dists += [dist, dist]
    sp = csr_matrix((dists, (rows, cols)), shape=(n, n))
    return float(minimum_spanning_tree(sp).sum())


def estimate_tsp_mst_ratio(nodes_coords):
    """MST-Ratio estimator: L ~ rho(d) * L_MST.

    Constants source: empirically calibrated near the Percus-Martin 1996
    asymptote beta_TSP / beta_MST for d-dimensional uniform points.
    For 2D uniform, Percus-Martin gives beta_TSP/beta_MST ~ 0.7124/0.6331
    ~ 1.125; the values 1.075 (d=2), 1.05 (d=3), and the tail form
    1 + 0.075*(2/d) (d>=4) are empirical near-asymptote constants used
    consistently across our benchmark CSVs.

    References:
      Percus, A.G., Martin, O.C. (1996). "Finite size and dimensional
        dependence in the Euclidean traveling salesman problem."
        Phys. Rev. E 54(2):1884. https://doi.org/10.1103/PhysRevE.54.1884
      Johnson, D.S., McGeoch, L.A., Rothberg, E.E. (1996). "Asymptotic
        experimental analysis for the Held-Karp traveling salesman bound."
        https://doi.org/10.1007/3-540-61310-2_18  (beta_2 = 0.7124).
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1:
        return 0.0, 0.0

    d = coords.shape[1]
    mst_len = _mst_length(coords)

    if d == 2:
        ratio = 1.075
    elif d == 3:
        ratio = 1.05
    else:
        ratio = 1.0 + (0.075 * (2.0 / d))
    return mst_len * ratio, time.perf_counter() - start_time

def estimate_tsp_hilbert(nodes_coords, p=16):
    """Bartholdi-Platzman (1982) space-filling-curve TSP heuristic.

    Constructive tour: sort points by their 1-D Hilbert-curve index,
    visit in order, close the loop. Paper uses this as a constructive
    upper-bound baseline (N-dimensional via hilbertcurve package).

    Reference:
      Bartholdi, J.J., Platzman, L.K. (1982). "An O(N log N)
        planar travelling salesman heuristic based on spacefilling
        curves." Oper. Res. Lett. 1(4):121-125.
        https://doi.org/10.1016/0167-6377(82)90012-8
    No asymptotic constant to calibrate; tour length is computed
    directly from the constructed ordering (p = 16-bit grid here).
    """
    start_time = time.perf_counter()

    # 1. Enforce Unique Coordinates (Standard Protocol)
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    # 2. Detect Dimensionality
    d = coords.shape[1]
    
    # 3. Scale Coordinates to Integer Grid [0, 2^p - 1]
    min_c = np.min(coords, axis=0)
    max_c = np.max(coords, axis=0)
    scale = max_c - min_c
    scale[scale < 1e-9] = 1.0 
    N_GRID = (1 << p) - 1
    norm_coords = (coords - min_c) / scale
    int_coords = (norm_coords * N_GRID).astype(int)
    
    # 4. Map N-D Coordinates to 1-D Hilbert Indices
    hc = HilbertCurve(p, d)
    points_list = int_coords.tolist()
    
    # Current hilbertcurve (>=2.x) exposes distances_from_points; older 2.0 used distances_from_coordinates.
    if hasattr(hc, 'distances_from_points'):
        hilbert_indices = hc.distances_from_points(points_list)
    elif hasattr(hc, 'distances_from_coordinates'):
        hilbert_indices = hc.distances_from_coordinates(points_list)
    else:
        raise AttributeError("Installed 'hilbertcurve' library has unsupported API; require >=2.0.")
    
    # 5. Sort Points by Hilbert Index
    sort_idx = np.argsort(hilbert_indices)
    sorted_coords = coords[sort_idx]
    
    # 6. Calculate Tour Length (Vectorized)
    deltas = sorted_coords[1:] - sorted_coords[:-1]
    tour_len = np.sum(np.sqrt(np.sum(deltas**2, axis=1)))
    
    # Close the loop
    tour_len += np.linalg.norm(sorted_coords[-1] - sorted_coords[0])
    
    return tour_len, time.perf_counter() - start_time

# ====================================================================
# SIMULATION ESTIMATORS
# ====================================================================

def estimate_tsp_evt(nodes_coords, samples=50):
    """DEPRECATED, cut from paper (dominated + simulation cost). Extreme-value-theory 2-opt fit."""
    warnings.warn(
        "estimate_tsp_evt is deprecated; cut as dominated baseline.",
        DeprecationWarning, stacklevel=2,
    )
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    local_optima = [_run_2opt_fast(coords, n) for _ in range(samples)]
    local_optima.sort()
    shape, loc, scale = weibull_min.fit(local_optima)
    estimated_min = loc
    if estimated_min < 0 or estimated_min > local_optima[0]:
        estimated_min = local_optima[0]
    return estimated_min, time.perf_counter() - start_time

def estimate_tsp_2opt_distribution(nodes_coords, samples=20):
    """DEPRECATED, cut from paper (dominated + simulation cost). 2-opt sample-distribution tail estimate."""
    warnings.warn(
        "estimate_tsp_2opt_distribution is deprecated; cut as dominated baseline.",
        DeprecationWarning, stacklevel=2,
    )
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    costs = [_run_2opt_fast(coords, n) for _ in range(samples)]
    mu = np.mean(costs)
    sigma = np.std(costs)
    if sigma < 1e-9: estimated_opt = mu
    else: estimated_opt = mu - 3.0 * sigma
    return estimated_opt, time.perf_counter() - start_time

def estimate_tsp_basel_willemain(nodes_coords):
    """DEPRECATED, cut from paper (wall-time ~= solver at n=100). Basel/Willemain random-tour sigma fit."""
    warnings.warn(
        "estimate_tsp_basel_willemain is deprecated; cut under the solver-time kill rule.",
        DeprecationWarning, stacklevel=2,
    )
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    BATCH_SIZE = 10000
    lengths = [_get_random_tour_len(coords, n) for _ in range(BATCH_SIZE)]
    sigma = np.std(lengths)
    if sigma < 1e-9: return 0.0, time.perf_counter() - start_time
    log_opt = 1.798 + 0.927 * np.log(sigma)
    est = np.exp(log_opt)
    return est, time.perf_counter() - start_time

# ====================================================================
# GEOMETRIC & ASYMPTOTIC ESTIMATORS
# ====================================================================

def estimate_tsp_chien(nodes_coords):
    """Chien (1992) single-route TSP length estimator.

    Formula (paper Eq. 423, 2D only):
        L = k1 * sqrt(n * A) + k2 * n / p
    with k1 = 0.98, k2 = 0, p = 1 (single-route, no depot term), so
        L = 0.98 * sqrt(n * A)
    where A is the convex-hull area of the points.

    Gated to d == 2. For d != 2 the expression is ill-defined
    (Chien's derivation is planar); we raise ValueError so callers
    can record status rather than silently producing a bogus value.

    Reference:
      Chien, T.W. (1992). "Operational estimators for the vehicle
        routing problem." Transportation Science 26(2):104-114.
        https://doi.org/10.1287/trsc.26.2.104
    Constants source: Chien (1992) Table 1 / Eq. (8), k1 = 0.98 for
    uniform 2D point distributions.
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1:
        return 0.0, 0.0
    d = coords.shape[1]
    if d != 2:
        raise ValueError(
            f"Chien: d={d} not supported (2D-only estimator). "
            f"Caller must gate on d and record status='chien_not_2d'."
        )

    if n > d + 1 and d <= CONVEX_HULL_MAX_DIM:
        A = float(ConvexHull(coords).volume)
    else:
        ranges = np.ptp(coords, axis=0).astype(float)
        ranges[ranges < 1e-9] = 1e-9
        A = float(np.prod(ranges))

    est = 0.98 * math.sqrt(n * A)
    return est, time.perf_counter() - start_time

def estimate_tsp_bhh(nodes_coords):
    """Beardwood-Halton-Hammersley (1959) asymptotic TSP length.

    Formula (paper Eq. 395):
        L ~ beta_d * n^((d-1)/d) * V^(1/d)
    Paper restricts BHH to d == 2 (beta_d for d >= 3 is not
    empirically pinned); this implementation falls back to the
    Gaussian-limit approximation beta_d ~ sqrt(d / (2*pi*e)) for
    d >= 4, retained only for completeness.

    References:
      Beardwood, Halton, Hammersley (1959). "The shortest path
        through many points." Proc. Camb. Phil. Soc. 55:299-327.
      Johnson, McGeoch, Rothberg (1996). "Asymptotic experimental
        analysis for the Held-Karp traveling salesman bound."
        https://doi.org/10.1007/3-540-61310-2_18
    Constants source: Johnson et al. (1996) Section 3.2, beta_2 =
    0.7124 +/- 0.0002; beta_3 ~ 0.6979 (Percus-Martin 1996,
    Phys. Rev. Lett. 76:1188,
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.76.1188);
    for d >= 4 we use the Rhee-Steele asymptotic
    beta_d ~ sqrt(d / (2 pi e)) as no tighter point estimate exists
    in the literature.

    Implementation note: at high d the bounding-box volume
    prod(ranges) overflows float64 (e.g. 10000**100 = 1e400).
    We therefore compute vol**(1/d) directly as the geometric
    mean exp(mean(log(ranges))), which is numerically stable
    at any d.
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    d = coords.shape[1]

    if n > d + 1 and d <= CONVEX_HULL_MAX_DIM:
        vol = ConvexHull(coords).volume
        vol_root = vol ** (1.0 / d) if vol > 0 else 0.0
    else:
        ranges = np.ptp(coords, axis=0).astype(float)
        ranges[ranges < 1e-9] = 1e-9
        # Geometric mean in log-space: equivalent to (prod ranges)**(1/d),
        # but stable at any d (no overflow from prod).
        vol_root = float(np.exp(np.mean(np.log(ranges))))

    if d == 2: beta = BETA_2D
    elif d == 3: beta = BETA_3D
    else: beta = math.sqrt(d / (2 * math.pi * math.e))

    exponent = (d - 1) / d
    est = beta * (n ** exponent) * vol_root
    return est, time.perf_counter() - start_time

def estimate_tsp_vinel(nodes_coords, b=0.768):
    """DEPRECATED, cut from paper (redundant with BHH in 2D). Vinel-style BHH-coefficient variant."""
    warnings.warn(
        "estimate_tsp_vinel is deprecated; redundant with BHH in 2D.",
        DeprecationWarning, stacklevel=2,
    )
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    d = coords.shape[1]

    if n > d + 1 and d <= CONVEX_HULL_MAX_DIM:
        vol = ConvexHull(coords).volume
    else:
        ranges = np.ptp(coords, axis=0).astype(float)
        ranges[ranges < 1e-9] = 1e-9
        vol = float(np.prod(ranges))

    geom_len_scale = math.pow(vol, 1.0 / d)
    n_scale = math.pow(n, (d - 1) / d)
    estimated_cost = b * n_scale * geom_len_scale
    return estimated_cost, time.perf_counter() - start_time

def mabr_rotate_2d(coords):
    """True 2D minimum-area bounding rectangle via rotating calipers.

    Uses the standard result that the MABR of a planar point set is aligned
    with one edge of its convex hull. Enumerates hull edges, rotates the hull
    so the edge is axis-aligned, measures the bounding-box area, and keeps
    the minimum. Returns the input coordinates expressed in that rotated
    frame (origin translated to the MABR corner, axes aligned with MABR).

    This is the orientation Cavdar & Sokol (2015) implicitly assume when they
    write "the length and width of a rectangular graph" — applying the paper's
    formula in any other frame distorts the per-axis dispersion statistics.

    Degenerate input (n < 3 or colinear hull) falls back to a centered copy
    of the input.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n < 3:
        return coords - coords.mean(axis=0)
    try:
        hull = ConvexHull(coords)
    except Exception:
        return coords - coords.mean(axis=0)
    hpts = coords[hull.vertices]
    m = len(hpts)
    if m < 2:
        return coords - coords.mean(axis=0)

    best_area = np.inf
    best_R = np.eye(2)
    best_origin = np.zeros(2)
    for i in range(m):
        p0 = hpts[i]
        p1 = hpts[(i + 1) % m]
        edge = p1 - p0
        L = float(np.linalg.norm(edge))
        if L < 1e-12:
            continue
        u = edge / L
        v = np.array([-u[1], u[0]])
        R = np.column_stack([u, v])  # rotate edge-frame <- world-frame via (X - p0) @ R
        proj = (hpts - p0) @ R
        dx = float(proj[:, 0].max() - proj[:, 0].min())
        dy = float(proj[:, 1].max() - proj[:, 1].min())
        area = dx * dy
        if area < best_area:
            best_area = area
            best_R = R
            best_origin = p0

    rotated = (coords - best_origin) @ best_R
    # deterministic sign: flip each axis so the extreme-magnitude point is positive
    for j in range(2):
        idx = int(np.argmax(np.abs(rotated[:, j])))
        if rotated[idx, j] < 0.0:
            rotated[:, j] = -rotated[:, j]
    return rotated


def canonicalize_coords_pca(coords):
    """Rotate a point cloud to its PCA principal-axis frame.

    ND-native generalization of the 2D minimum-area bounding rectangle (MABR):
    diagonalizes the sample covariance so axes align with the directions of
    greatest variance. For a uniform sample on an axis-aligned rectangle this
    recovers the original orientation; for an arbitrarily rotated cloud it
    removes the orientation-of-the-input noise that pollutes axis-dependent
    features (coordinate stdevs, midpoint distances, bounding-box ranges).

    The sign of each principal axis is fixed by making the coordinate with
    the largest absolute magnitude positive, so reflections of the input
    map to the same canonical frame.

    Complexity: O(n*d^2 + d^3); ND-safe, unlike true rotating-calipers MABR
    which is 2D-only.
    """
    arr = np.ascontiguousarray(coords, dtype=np.float64)
    n, d = arr.shape
    if n < 2 or d < 1:
        return arr.astype(np.asarray(coords).dtype, copy=True)
    centered = arr - arr.mean(axis=0)
    if d == 1:
        return centered.astype(np.asarray(coords).dtype, copy=False)
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    rotated = centered @ R
    for j in range(d):
        idx = int(np.argmax(np.abs(rotated[:, j])))
        if rotated[idx, j] < 0.0:
            rotated[:, j] = -rotated[:, j]
    return rotated.astype(np.asarray(coords).dtype, copy=False)


def estimate_tsp_cavdar(nodes_coords, a0=2.791, a1=0.2669):
    """Cavdar & Sokol (2015) distribution-free TSP tour-length estimator.

    Faithful implementation of Eq. (3) from Cavdar & Sokol (2015):

        T = 2.791 * sqrt(n * cstdev_x * cstdev_y)
          + 0.2669 * sqrt(n * stdev_x * stdev_y * A / (cbar_x * cbar_y))

    using the paper's exact definitions:
        stdev_{x,y}   : std of coordinates along each axis.
        cbar_{x,y}    : mean absolute distance from each node to the
                        *central midpoint axis* of the bounding rectangle
                        (i.e. to (min+max)/2, NOT to the mean).
        cstdev_{x,y}  : std of those absolute distances from the midpoint.
        A             : area of the (rectangular) graph = product of the
                        axis-aligned ranges (l_x * l_y in 2D).

    The paper assumes an axis-aligned rectangular graph. To apply the model
    to arbitrarily oriented point clouds without distorting its axis-dependent
    statistics, we first rotate the coordinates into the PCA principal-axis
    frame (ND-native analogue of the 2D minimum-area bounding rectangle). This
    makes the bounding-rectangle area and per-axis dispersions rotation-
    invariant without altering the paper's formula.

    ND extension: the 2D form is lifted using the standard geometric-mean
    convention
        term1 = a0 * n^((d-1)/d) * (prod_j cstdev_j)^(1/d)
        term2 = a1 * n^((d-1)/d) * A^(1/d) * (prod_j stdev_j / prod_j cbar_j)^(1/d)
    which reduces exactly to Eq. (3) for d = 2.

    Small-n correction (Cavdar & Sokol 2015 Eq. 4):
        E/T = 0.9325 * exp(0.00005298 * n) - 0.2972 * exp(-0.01452 * n)
    Divide raw estimate by this ratio for n < 1000 (paper's calibration range
    100 <= n <= 975).

    Reference:
      Cavdar, B., Sokol, J. (2015). "A distribution-free TSP tour length
      estimation model for random graphs." European Journal of Operational
      Research 243(2): 588-598. doi:10.1016/j.ejor.2014.12.020
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1:
        return 0.0, 0.0
    d = coords.shape[1]

    # Rotate into the frame the paper implicitly assumes (axis-aligned
    # rectangular graph). In 2D this is the true minimum-area bounding
    # rectangle via rotating calipers on the convex hull, exactly as Cavdar &
    # Sokol (2015) define "the rectangular graph". For d > 2 the paper offers
    # no definition (it is a 2D model); we fall back to PCA as a principled
    # ND extension, but the 2D branch is an exact match to the paper.
    if d == 2:
        coords = mabr_rotate_2d(coords)
    else:
        coords = canonicalize_coords_pca(coords)

    # A = convex-hull area (2D) / volume (ND). Paper, Section 4: for tests on
    # non-random graphs, "we used the area of the convex hull of the nodes as
    # A". For the paper's random rectangular training graphs the hull area
    # coincides with l_x * l_y by construction (they explicitly place a node
    # at each corner), so the convex-hull form is faithful to both cases and
    # strictly more accurate on real-world (non-corner-padded) instances.
    # Rotation into the MABR frame above leaves ConvexHull.volume invariant,
    # so the per-axis midpoint/stdev/cbar/cstdev statistics still come from
    # the canonical frame while A stays rotation-free by construction.
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    ranges = (hi - lo).astype(float)
    ranges[ranges < 1e-9] = 1e-9
    if n > d + 1 and d <= CONVEX_HULL_MAX_DIM:
        try:
            vol = float(ConvexHull(coords).volume)
        except Exception:
            vol = float(np.prod(ranges))
    else:
        vol = float(np.prod(ranges))
    if vol < 1e-12:
        vol = float(np.prod(ranges))

    # Paper: "average distance of nodes to the central horizontal and vertical
    # axes (the horizontal and vertical midpoint lines of the space)".
    midpoint = 0.5 * (hi + lo)

    stdev = coords.std(axis=0).astype(np.float64)
    stdev = np.where(stdev < 1e-12, 1e-12, stdev)
    abs_dev = np.abs(coords - midpoint).astype(np.float64)
    c_bar = abs_dev.mean(axis=0)
    c_bar = np.where(c_bar < 1e-12, 1e-12, c_bar)
    cstdev = abs_dev.std(axis=0)
    cstdev = np.where(cstdev < 1e-12, 1e-12, cstdev)

    n_scale = math.pow(n, (d - 1) / d)
    inv_d = 1.0 / d

    # Compute geometric-mean products in log space to stay finite in high d
    # (prod_j cstdev_j overflows float64 by d ~ 30 for typical grid sizes).
    log_geom_cstdev = float(np.sum(np.log(cstdev)) / d)
    log_geom_stdev = float(np.sum(np.log(stdev)) / d)
    log_geom_cbar = float(np.sum(np.log(c_bar)) / d)
    log_vol = float(np.sum(np.log(ranges)))  # full log-volume (not geom mean)

    term1 = a0 * n_scale * math.exp(log_geom_cstdev)
    term2 = a1 * n_scale * math.exp(log_vol * inv_d) * math.exp(log_geom_stdev - log_geom_cbar)

    estimated_cost = term1 + term2

    if n < 1000:
        corr = 0.9325 * math.exp(0.00005298 * n) - 0.2972 * math.exp(-0.01452 * n)
        estimated_cost = estimated_cost / corr

    return estimated_cost, time.perf_counter() - start_time

KWON_CALIBRATION_N_MAX = 300  # Kwon, Golden, Wasil (1995) calibration range upper bound.


def estimate_tsp_kwon(nodes_coords):
    """
    Kwon, Golden, Wasil (1995) TSP/VRP tour-length estimator.

    Kwon calibrated the form L_norm = (0.8326 - 0.0011*n + 1.1147*R/n) * sqrt(n)
    against unit-area service regions. We reproduce that convention by first
    rescaling coordinates so the bounding-box diagonal is 1, evaluating Kwon's
    expression on the normalized coordinates, and scaling the predicted length
    back by the original diagonal. This is the same "rescale then apply" wrapper
    used when Cavdar-Sokol (2015) benchmark Kwon in their Table 3; without it
    the -0.0011*n term drives the estimator negative at TSPLIB-scale n.

    Raises :class:`ValueError` when n exceeds the calibration range
    (``KWON_CALIBRATION_N_MAX = 300``). The coefficient ``-0.0011 * n`` drives
    the estimator negative past that size, so extrapolating is ill-defined.
    Callers MUST gate on n before invoking this function and record a status
    row rather than silently suppressing the failure.
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1:
        return 0.0, 0.0
    if n > KWON_CALIBRATION_N_MAX:
        raise ValueError(
            f"Kwon: n={n} > {KWON_CALIBRATION_N_MAX} (calibration range). "
            f"Caller must gate on n and record status='kwon_out_of_calibration'."
        )
    d = coords.shape[1]

    ranges = np.ptp(coords, axis=0).astype(float)
    diag = float(np.sqrt(np.sum(ranges ** 2)))
    if diag < 1e-12:
        return 0.0, time.perf_counter() - start_time

    coords_n = (coords - coords.min(axis=0)) / diag

    if n > d + 1 and d <= CONVEX_HULL_MAX_DIM:
        A = ConvexHull(coords_n).volume
    else:
        r_n = np.ptp(coords_n, axis=0).astype(float)
        r_n[r_n < 1e-9] = 1e-9
        A = float(np.prod(r_n))

    centroid = coords_n.mean(axis=0)
    R = float(np.mean(np.linalg.norm(coords_n - centroid, axis=1)))

    est_norm = (0.8326 - 0.0011 * n + 1.1147 * (R / max(n, 1))) * math.sqrt(n * A)
    est_norm = max(est_norm, 0.0)
    est = est_norm * diag  # rescale back to original units
    return est, time.perf_counter() - start_time


def estimate_tsp_daganzo(nodes_coords, k=0.57):
    """Daganzo (1984) TSP/CVRP-style tour length estimator.

    Formula (paper Eq. 416): L = 0.57 * sqrt(n * A).

    Structurally identical to BHH up to the choice of constant
    (BHH uses beta_2 = 0.7124); the two are included separately
    because the literature reports them as separate baselines.

    Reference:
      Daganzo, C.F. (1984). "The length of tours in zones of
        different shapes." Transportation Research B 18(2):135-145.
        https://doi.org/10.1016/0191-2615(84)90027-4
    Constants source: Daganzo (1984) Eq. (1), k = 0.57 for uniform
    points inside a disc / near-circular service region.
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1:
        return 0.0, 0.0
    d = coords.shape[1]

    if n > d + 1 and d <= CONVEX_HULL_MAX_DIM:
        A = ConvexHull(coords).volume
    else:
        ranges = np.ptp(coords, axis=0).astype(float)
        ranges[ranges < 1e-9] = 1e-9
        A = float(np.prod(ranges))

    est = k * math.sqrt(n * A)
    return est, time.perf_counter() - start_time


def estimate_tsp_composite(nodes_coords):
    """DEPRECATED, cut from paper (dominated by GART). Meta-estimator: max(MST, min(2MST, Vinel/Cavdar))."""
    warnings.warn(
        "estimate_tsp_composite is deprecated; dominated by GART 2.0.",
        DeprecationWarning, stacklevel=2,
    )
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    if n <= 10:
        cost, _ = estimate_tsp_held_karp(coords)
        return cost, time.perf_counter() - start_time

    d = coords.shape[1]
    mst_length = _mst_length(coords)

    if n < 100:
        est = estimate_tsp_vinel(coords)[0]
    else:
        est = estimate_tsp_cavdar(coords)[0]

    final_cost = max(mst_length, min(2 * mst_length, est))
    return final_cost, time.perf_counter() - start_time

# ====================================================================
# MACHINE LEARNING (GART 1.0)
# ====================================================================

def _calculate_gart_features(coords, precomputed_mst=None):
    """GART 1.0 legacy feature set (2D).

    MST topology features are computed from a Delaunay-sparse graph for 2D
    inputs (O(n log n)); 1-NN distances are still computed from the dense
    matrix because they require per-point nearest-neighbor lookups.
    """
    n = len(coords)
    d = coords.shape[1] if coords.ndim == 2 else 2
    features = {'n': n}
    if n > d + 1:
        hull = ConvexHull(coords)
        features['convex_hull_area'] = hull.volume
        features['convex_hull_perimeter'] = hull.area
        features['hull_vertex_count'] = len(hull.vertices)
        features['hull_ratio'] = features['hull_vertex_count'] / n
    else:
        features.update({'convex_hull_area': 0.0, 'convex_hull_perimeter': 0.0, 'hull_vertex_count': 0, 'hull_ratio': 0.0})

    ranges = np.ptp(coords, axis=0).astype(float)
    features['bounding_box_area'] = np.prod(ranges)

    # Dense matrix still needed for 1-NN distances (per-row min over neighbors).
    dist_matrix = cdist(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)
    one_nn = np.min(dist_matrix, axis=1)
    features['one_nn_dist_mean'] = one_nn.mean()
    features['one_nn_dist_std'] = one_nn.std()

    if n >= 2 and d >= 2:
        pca = PCA(n_components=2).fit(coords)
        ev = pca.explained_variance_
        features['pca_eigenvalue_ratio'] = ev[0] / ev[1] if ev[1] > 1e-9 else 1.0
    else:
        features['pca_eigenvalue_ratio'] = 1.0

    # MST via the project-wide utility (dense primary, OOM fallback).
    # If a precomputed MST is supplied (e.g. from an external non-Euclidean
    # distance matrix), use it instead of recomputing from coords.
    mst_result = precomputed_mst if precomputed_mst is not None else compute_mst(coords)
    mst_length = float(mst_result.total_length)
    degrees = mst_result.degrees
    features['mst_degree_mean'] = degrees.mean()
    features['mst_degree_max'] = degrees.max()
    features['mst_degree_std'] = degrees.std()
    features['mst_leaf_nodes_fraction'] = np.sum(degrees == 1) / n

    features['coord_std_dev_x'] = coords[:, 0].std()
    features['coord_std_dev_y'] = coords[:, 1].std()
    depot = coords[0]
    dists_depot = np.linalg.norm(coords[1:] - depot, axis=1) if n > 1 else np.array([0.0])
    features['avg_dist_from_depot'] = dists_depot.mean()
    features['max_dist_from_depot'] = dists_depot.max()

    return features, mst_length

def estimate_tsp_ml_alpha(nodes_coords, ml_model, precomputed_mst=None):
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0

    features_dict, mst_length = _calculate_gart_features(coords, precomputed_mst=precomputed_mst)
    t_feat = time.perf_counter() - start_time
    if mst_length == 0: return 0.0, time.perf_counter() - start_time

    feature_df = pd.DataFrame([features_dict])
    if hasattr(ml_model, "feature_name_"):
        feature_df = feature_df[ml_model.feature_name_]

    predicted_alpha = ml_model.predict(feature_df)[0]
    est = predicted_alpha * mst_length
    final_cost = min(max(mst_length, est), 2*mst_length)
    t_inf = time.perf_counter() - start_time - t_feat
    return final_cost, t_feat, t_inf