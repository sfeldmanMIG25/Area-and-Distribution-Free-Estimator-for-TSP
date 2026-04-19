"""GART V4 feature engineering (single source of truth).

Computes the full 42-candidate feature set for one TSP instance:

  * 29 inherited V3 features (geometric spread, centroid dispersion,
    MST topology)
  * Tier 1 (5): PCA-Oriented Bounding Box + spectral shape
  * Tier 2 (4): MST-based local density (exact 1-NN via MST cut property,
    2nd-incident-edge proxy for 2-NN) — defined at any dimension
  * Tier 3 (2): Mahalanobis log-volume, MST-edge PCA anisotropy
  * Tier 4 (1): Ripley's L deviation at the median-NN scale (clustered vs.
    regular vs. Poisson-random point pattern, computed in log-space)
  * Tier 5 (1): Greedy nearest-neighbour tour length normalised by MST length
    (Rosenkrantz-Stearns-Lewis 1977 tour-ordering upper bound)

Design:
  * Public entry point :func:`compute_features(coords, dimension, grid_size)`.
  * Returns a plain ``dict`` so callers can pd.DataFrame-ify or select columns.
  * The same function is used offline (for the CSV) and inline (in the
    V4 estimator) — guarantees train/serve parity by construction.
  * Strict — no silent fallbacks except the single allowed Delaunay → dense MST
    fallback (same policy as the rest of the pipeline).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import os
import sys

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import gammaln

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mst_utils import compute_mst

# Delaunay (QHull) crossover vs dense matrix: empirically dense is faster from
# d >= 4 upwards (measured d=5, n=1000 -> Delaunay ~1.8s vs dense ~0.2s).
# Keep Delaunay for d ∈ {2, 3} only. Dispatch lives in ``mst_utils.compute_mst``.
DELAUNAY_DIM_CAP = 3

# Log/exp safety rails — keep features in float64 range without overflow.
LOG_CAP = 690.0  # log(np.exp(690)) ~ 1e300, safely inside float64 max.

# Dense-matrix reuse budget for greedy-NN tour construction. If an n*n float32
# matrix fits this budget, we build it once and let greedy read rows of it
# (O(n^2) memory access, fastest). Otherwise greedy falls back to a cKDTree
# incremental-k path — O(n log n) time and O(n) memory, required for TSPLIB
# scale (n up to ~85k). Override with env var ``GREEDY_DENSE_BUDGET_GB``.
# Default 0.5 GB -> n_max ~ 11k; cdist peak during build is ~2x this.
_GREEDY_DENSE_BUDGET_BYTES = int(
    float(os.environ.get("GREEDY_DENSE_BUDGET_GB", "0.5")) * (1 << 30)
)


def _greedy_dense_feasible(n: int) -> bool:
    return (n * n * 4) <= _GREEDY_DENSE_BUDGET_BYTES


# =============================================================================
# MST — delegates to the project-wide utility (dense primary, fallback on OOM)
# =============================================================================
def _compute_mst(coords: np.ndarray, d: int) -> csr_matrix:
    """Return MST as a scipy CSR matrix. Preserved shim — the real computation
    lives in ``mst_utils.compute_mst``. ``d`` is kept for call-site compatibility
    but unused by the utility (dim is taken from ``coords.shape[1]``)."""
    return compute_mst(coords).to_csr()


# =============================================================================
# PCA helpers
# =============================================================================
def _pca_eigenvalues(coords: np.ndarray) -> np.ndarray:
    """Return sorted-descending eigenvalues of the coordinate covariance."""
    centred = coords - coords.mean(axis=0)
    # SVD is numerically more stable than covariance eigendecomposition for
    # flat / near-degenerate point sets.
    s = np.linalg.svd(centred, compute_uv=False)
    eigvals = (s ** 2) / max(coords.shape[0] - 1, 1)
    return np.sort(eigvals)[::-1]


def _obb_volume(coords: np.ndarray) -> tuple[float, float]:
    """PCA-oriented bounding box volume. Returns (obb_volume, log_obb_volume).
    The OBB is the tightest axis-aligned bounding box in the rotated (PCA)
    frame and is rotation-invariant — unlike raw coordinate-axis ptp.
    """
    centred = coords - coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    rotated = centred @ Vt.T
    ranges = np.ptp(rotated, axis=0)
    ranges = np.where(ranges < 1e-9, 1e-9, ranges)
    log_vol = float(np.sum(np.log(ranges)))
    vol = float(np.exp(min(log_vol, LOG_CAP)))
    return vol, log_vol


# =============================================================================
# MST-based local density
# =============================================================================
def _mst_local_density_stats(
    rows: np.ndarray, cols: np.ndarray, edges: np.ndarray, n: int
) -> Dict[str, float]:
    """Exact 1-NN distance and a 2-NN proxy derived from the MST.

    By the MST cut property, the edge from any vertex v to its nearest
    neighbour is always in the MST, so the minimum incident MST-edge weight
    equals the exact 1-NN distance. For the second-nearest neighbour there is
    no such guarantee, so we use the second-smallest incident MST-edge weight
    as a proxy (it is an upper bound on the true 2-NN distance).

    Works at any dimension — no concentration-of-measure cap needed.
    """
    if n < 3 or len(edges) == 0:
        return {
            "mst_nn1_mean": 0.0,
            "mst_nn1_cv": 0.0,
            "mst_nn2_proxy_mean": 0.0,
            "mst_nn_gap_ratio": 1.0,
        }
    incident: list[list[float]] = [[] for _ in range(n)]
    for i in range(len(rows)):
        w = float(edges[i])
        incident[int(rows[i])].append(w)
        incident[int(cols[i])].append(w)
    nn1 = np.zeros(n, dtype=np.float64)
    nn2_vals: list[float] = []
    for v in range(n):
        lst = incident[v]
        if not lst:
            continue
        lst.sort()
        nn1[v] = lst[0]
        if len(lst) >= 2:
            nn2_vals.append(lst[1])
    nn1_mean = float(np.mean(nn1))
    nn1_std = float(np.std(nn1))
    nn2_mean = float(np.mean(nn2_vals)) if nn2_vals else nn1_mean
    return {
        "mst_nn1_mean": nn1_mean,
        "mst_nn1_cv": (nn1_std / nn1_mean) if nn1_mean > 1e-12 else 0.0,
        "mst_nn2_proxy_mean": nn2_mean,
        "mst_nn_gap_ratio": (nn2_mean / nn1_mean) if nn1_mean > 1e-12 else 1.0,
    }


# =============================================================================
# Greedy nearest-neighbour tour — Rosenkrantz-Stearns-Lewis (1977) upper bound
# =============================================================================
def _greedy_nn_tour_length(
    coords: np.ndarray, D: Optional[np.ndarray] = None
) -> float:
    """Closed greedy-NN tour length with a canonical (centroid-nearest) start.

    Two paths — selected by the caller based on memory budget:
      * ``D`` is an (n, n) pairwise-distance matrix — each step is a masked
        argmin over one row (O(n^2) memory access, ~free with a cached matrix).
      * Otherwise, build a ``cKDTree`` once and do incremental-k nearest
        queries, skipping visited vertices — O(n log n) at low d, O(n) memory.
        This path is what lets TSPLIB instances at n ~ 85k run without a
        29 GB distance matrix.

    Output is identical to a brute-force greedy tour. Starting vertex is the
    point closest to the centroid so the result is reordering-invariant.
    RSL (1977) upper bound for metric TSP: within ½·(⌈log₂ n⌉+1)·OPT.
    """
    n = coords.shape[0]
    if n < 2:
        return 0.0

    centroid = coords.mean(axis=0)
    start = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    total = 0.0

    # Self-dispatch: if caller didn't pass D, build one when it fits the
    # budget. This keeps the fast path identical whether called from
    # compute_features or from the feature-timing benchmark.
    if D is None and _greedy_dense_feasible(n):
        D = cdist(coords, coords).astype(np.float32, copy=False)

    if D is not None:
        # Fast path: masked-argmin on a cached distance matrix row per step.
        for _ in range(n - 1):
            row = D[current].copy()
            row[visited] = np.inf
            nxt = int(np.argmin(row))
            total += float(row[nxt])
            visited[nxt] = True
            current = nxt
        total += float(np.linalg.norm(coords[current] - coords[start]))
        return total

    # Fallback: kd-tree with growing k so we never allocate n^2.
    tree = cKDTree(coords)
    k = min(32, n)
    for _ in range(n - 1):
        nxt = -1
        d_nxt = 0.0
        while True:
            dists, idxs = tree.query(coords[current], k=k)
            # k=1 returns scalars; normalize to arrays.
            idxs = np.atleast_1d(idxs)
            dists = np.atleast_1d(dists)
            mask_unvis = ~visited[idxs]
            if mask_unvis.any():
                j = int(np.argmax(mask_unvis))  # first unvisited in sorted order
                nxt = int(idxs[j])
                d_nxt = float(dists[j])
                break
            if k >= n:
                break
            k = min(k * 2, n)
        if nxt < 0:
            break
        total += d_nxt
        visited[nxt] = True
        current = nxt
    total += float(np.linalg.norm(coords[current] - coords[start]))
    return total


# =============================================================================
# Spatial point-pattern — Ripley's L deviation at the median NN scale
# =============================================================================
def _ripley_L_deviation(coords: np.ndarray, d: int) -> float:
    """Ripley's L-function deviation at r = median nearest-neighbour distance.

    For a homogeneous Poisson point process (CSR), L(r) = r exactly, so
    L(r) / r − 1 is 0 for CSR, positive for clustered point sets, and negative
    for regular / repulsive point sets (grids, quasi-crystal). The
    characteristic scale (median NN distance) auto-normalises for n and extent.

    Computed fully in log-space so it survives any ambient dimension.
    """
    n = coords.shape[0]
    if n < 3:
        return 0.0

    ranges = np.ptp(coords, axis=0).astype(np.float64)
    ranges = np.where(ranges < 1e-9, 1e-9, ranges)
    log_V = float(np.sum(np.log(ranges)))  # log bounding hypervolume
    tree = cKDTree(coords)

    # Characteristic scale r = median 1-NN distance. query k=2 (self + 1-NN).
    dd, _ = tree.query(coords, k=2)
    r = float(np.median(dd[:, 1]))
    if r <= 0.0:
        return 0.0

    # Number of *ordered* pairs within r; subtract the n self-pairs.
    pairs = tree.count_neighbors(tree, r)
    neighbour_pairs = pairs - n
    if neighbour_pairs <= 0:
        return -1.0

    # K(r) = (neighbour_pairs / n) / lambda ; lambda = n / V
    # log K = log(neighbour_pairs) - 2 log n + log V
    log_K = np.log(neighbour_pairs) - 2.0 * np.log(n) + log_V

    # Unit-ball volume c_d = pi^(d/2) / gamma(d/2 + 1); use lgamma in log-space.
    log_c_d = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)

    # L(r) = (K(r) / c_d)^(1/d)
    log_L = (log_K - log_c_d) / d
    L = float(np.exp(log_L))
    return L / r - 1.0


# =============================================================================
# MST-edge PCA
# =============================================================================
def _mst_edge_pca_e1_share(
    coords: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> float:
    """PCA share of the top eigenvalue among MST-edge direction vectors.

    Measures *edge-space* anisotropy: even if the point cloud is isotropic,
    the MST's edges may align with a dominant axis (e.g. long thin clusters
    connected by a single backbone). Value in [1/d, 1]."""
    if len(rows) < 2:
        return 1.0
    edge_vecs = coords[cols] - coords[rows]
    eigvals = _pca_eigenvalues(edge_vecs)
    total = float(np.sum(eigvals))
    if total <= 1e-12:
        return 1.0
    return float(eigvals[0] / total)


# =============================================================================
# Tree diameter (copied verbatim from V3 feature_creator for behaviour parity)
# =============================================================================
def _tree_diameter(mst_adj, n: int) -> float:
    def farthest(start: int):
        dists = np.full(n, -1.0)
        dists[start] = 0.0
        q = deque([start])
        fn, md = start, 0.0
        while q:
            u = q.popleft()
            if dists[u] > md:
                md = dists[u]; fn = u
            for v, w in mst_adj[u]:
                if dists[v] < 0:
                    dists[v] = dists[u] + w
                    q.append(v)
        return fn, md

    if n < 2:
        return 0.0
    n1, _ = farthest(0)
    _, diam = farthest(n1)
    return float(diam)


# =============================================================================
# Main entry point
# =============================================================================
def compute_features(
    coords: np.ndarray, dimension: int, grid_size: int
) -> Dict[str, float]:
    """Compute the 41-candidate V4 feature dict for one instance.

    ``coords`` is expected to be (n, d) float32; duplicate-row filtering
    should be done by the caller before invoking this.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D, got shape {coords.shape}")
    n, d_coords = coords.shape
    if d_coords != dimension:
        raise ValueError(
            f"coords dim {d_coords} disagrees with declared dimension {dimension}"
        )
    if n < 3:
        raise ValueError(f"feature extraction requires n >= 3 (got n={n})")

    out: Dict[str, float] = {
        "n_customers": int(n),
        "dimension": int(dimension),
        "grid_size": int(grid_size),
    }

    # --- Group 1: coordinate-axis spread --------------------------------------
    ranges = np.ptp(coords, axis=0).astype(np.float64)
    ranges = np.where(ranges < 1e-9, 1e-9, ranges)
    log_hv = float(np.sum(np.log(ranges)))
    out["log_bounding_hypervolume"] = log_hv
    out["bounding_hypervolume"] = float(np.exp(min(log_hv, LOG_CAP)))
    log_density = float(np.log(n) - log_hv)
    out["log_node_density"] = log_density
    out["node_density"] = float(np.exp(max(min(log_density, LOG_CAP), -LOG_CAP)))
    out["aspect_ratio"] = float(np.max(ranges) / np.min(ranges))

    # --- Group 2: centroid dispersion ----------------------------------------
    centroid = np.mean(coords, axis=0, dtype=np.float64)
    centroid_dists = np.linalg.norm(coords - centroid, axis=1)
    out["centroid_dist_mean"] = float(np.mean(centroid_dists))
    out["centroid_dist_std"] = float(np.std(centroid_dists))
    out["centroid_dist_max"] = float(np.max(centroid_dists))
    q75, q25 = np.percentile(centroid_dists, [75, 25])
    out["centroid_dist_iqr"] = float(q75 - q25)

    # --- Group 3: MST topology (V3 set) --------------------------------------
    mst = _compute_mst(coords, dimension)
    edges = mst.data.astype(np.float64)
    if len(edges) == 0:
        edges = np.array([0.0])

    mst_len = float(np.sum(edges))
    out["mst_total_length"] = mst_len
    e_mean = float(np.mean(edges))
    e_std = float(np.std(edges))
    out["mst_edge_mean"] = e_mean
    out["mst_edge_std"] = e_std
    out["mst_edge_skew"] = float(stats.skew(edges)) if e_std > 1e-9 else 0.0
    out["mst_edge_kurtosis"] = float(stats.kurtosis(edges)) if e_std > 1e-9 else 0.0
    out["mst_edge_max"] = float(np.max(edges))
    percs = np.percentile(edges, [10, 25, 50, 75, 90])
    for i, p in enumerate([10, 25, 50, 75, 90]):
        out[f"mst_edge_q{p}"] = float(percs[i])

    k_dom = max(1, int(np.sqrt(n)))
    if len(edges) >= k_dom:
        top_k = np.partition(edges, -k_dom)[-k_dom:]
        out["mst_dominance_ratio"] = float(np.sum(top_k) / mst_len) if mst_len > 1e-9 else 0.0
    else:
        out["mst_dominance_ratio"] = 1.0
    median_edge = float(percs[2])
    out["mst_gap_ratio"] = float(out["mst_edge_max"] / median_edge) if median_edge > 1e-9 else 0.0

    rows, cols = mst.nonzero()
    mst_adj = [[] for _ in range(n)]
    degrees = np.zeros(n, dtype=np.int32)
    for i in range(len(rows)):
        u, v, w = int(rows[i]), int(cols[i]), float(edges[i])
        mst_adj[u].append((v, w))
        mst_adj[v].append((u, w))
        degrees[u] += 1
        degrees[v] += 1

    out["mst_leaf_ratio"] = float(np.sum(degrees == 1) / n)
    out["mst_degree_mean"] = float(np.mean(degrees))
    out["mst_degree_std"] = float(np.std(degrees))
    out["mst_degree_max"] = int(np.max(degrees))
    diam = _tree_diameter(mst_adj, n)
    out["mst_diameter"] = diam
    out["mst_diameter_normalized"] = float(diam / mst_len) if mst_len > 1e-9 else 0.0
    out["large_edge_count"] = int(np.sum(edges > e_mean + e_std))

    # --- Group 4 (Tier 1): Oriented Bounding Box + PCA spectral --------------
    # One full SVD; Vt reused for OBB rotation, s for eigenvalues.
    centred = coords.astype(np.float64) - coords.mean(axis=0, dtype=np.float64)
    _, s_svd, Vt = np.linalg.svd(centred, full_matrices=False)
    eigvals = np.sort(s_svd ** 2 / max(n - 1, 1))[::-1]
    lam_sum = float(np.sum(eigvals))
    lam_sq_sum = float(np.sum(eigvals ** 2))
    out["pca_e1_share"] = float(eigvals[0] / lam_sum) if lam_sum > 1e-12 else 1.0
    out["pca_effective_rank"] = float(lam_sum ** 2 / lam_sq_sum) if lam_sq_sum > 1e-12 else 1.0
    rotated = centred @ Vt.T
    obb_ranges = np.ptp(rotated, axis=0)
    obb_ranges = np.where(obb_ranges < 1e-9, 1e-9, obb_ranges)
    log_obb_vol = float(np.sum(np.log(obb_ranges)))
    obb_vol = float(np.exp(min(log_obb_vol, LOG_CAP)))
    out["obb_volume"] = obb_vol
    out["log_obb_volume"] = log_obb_vol
    out["obb_shrinkage"] = float(obb_vol / out["bounding_hypervolume"]) if out["bounding_hypervolume"] > 1e-12 else 0.0

    # --- Group 5 (Tier 2): MST-based local density ---------------------------
    out.update(_mst_local_density_stats(rows, cols, edges, n))

    # --- Group 6 (Tier 3): Second-order shape ---------------------------------
    # log-determinant of coord covariance (Mahalanobis-style log-volume, safe
    # in any dimension). Zero eigenvalues are clipped with a floor.
    eig_floor = np.maximum(eigvals, 1e-12)
    out["pca_log_det"] = float(np.sum(np.log(eig_floor)))
    out["mst_edge_pca_e1_share"] = _mst_edge_pca_e1_share(coords, rows, cols)

    # --- Group 7 (Tier 4): Spatial point-pattern clustering ------------------
    out["ripley_L_dev"] = _ripley_L_deviation(coords, dimension)

    # --- Group 8 (Tier 5): Greedy-NN tour / MST ratio ------------------------
    # Scale-free tour-ordering signal: Rosenkrantz-Stearns-Lewis upper bound
    # normalised by the MST lower-bound proxy. Raw greedy length is omitted
    # since it is near-redundant with mst_total_length.
    # Dense fast path when the n*n float32 matrix fits the budget; kd-tree
    # fallback otherwise so TSPLIB-scale instances (n ~ 85k) stay in memory.
    if _greedy_dense_feasible(n):
        D_greedy = cdist(coords, coords).astype(np.float32, copy=False)
        greedy_len = _greedy_nn_tour_length(coords, D=D_greedy)
        del D_greedy
    else:
        greedy_len = _greedy_nn_tour_length(coords)
    out["greedy_nn_over_mst"] = float(greedy_len / mst_len) if mst_len > 1e-9 else 1.0

    return out


# List of identifier columns that the CSV carries alongside features.
IDENTIFIER_COLUMNS = ("instance_name", "optimal_cost", "split")

# List of metadata / size / dimension columns (also features the model uses).
METADATA_FEATURES = ("n_customers", "dimension", "grid_size")


def feature_columns_for_training() -> list[str]:
    """All candidate feature names in a deterministic order. Excludes the
    identifier / target columns."""
    return [
        # V3 metadata / scale  (grid_size kept in CSV but excluded from training — leakage)
        "n_customers", "dimension",
        # V3 geometric spread
        "bounding_hypervolume", "log_bounding_hypervolume",
        "node_density", "log_node_density", "aspect_ratio",
        "centroid_dist_mean", "centroid_dist_std",
        "centroid_dist_max", "centroid_dist_iqr",
        # V3 MST stats
        "mst_total_length",
        "mst_edge_mean", "mst_edge_std", "mst_edge_skew",
        "mst_edge_kurtosis", "mst_edge_max",
        "mst_edge_q10", "mst_edge_q25", "mst_edge_q50",
        "mst_edge_q75", "mst_edge_q90",
        "mst_dominance_ratio", "mst_gap_ratio",
        "mst_leaf_ratio", "mst_degree_mean", "mst_degree_std",
        "mst_degree_max", "mst_diameter", "mst_diameter_normalized",
        "large_edge_count",
        # V4 Tier 1
        "obb_volume", "log_obb_volume", "obb_shrinkage",
        "pca_e1_share", "pca_effective_rank",
        # V4 Tier 2 (MST-based local density)
        "mst_nn1_mean", "mst_nn1_cv", "mst_nn2_proxy_mean", "mst_nn_gap_ratio",
        # V4 Tier 3
        "pca_log_det", "mst_edge_pca_e1_share",
        # V4 Tier 4 (spatial point-pattern)
        "ripley_L_dev",
        # V4 Tier 5 (tour-ordering upper bound)
        "greedy_nn_over_mst",
    ]
