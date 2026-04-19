"""GART V4 feature engineering (single source of truth).

Computes the full 41-candidate feature set for one TSP instance:

  * 29 inherited V3 features (geometric spread, centroid dispersion,
    MST topology)
  * Tier 1 (5): PCA-Oriented Bounding Box + spectral shape
  * Tier 2 (4): local density via cKDTree (1-NN, 2-NN)
  * Tier 3 (3): Mahalanobis log-volume, MST-edge PCA, 2-NN intrinsic dim

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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mst_utils import compute_mst

# Dimension caps. QHull (Delaunay) blows up beyond ~10 dims; cKDTree still
# works in high d but falls back to linear scan and its own BallTree is
# preferable beyond d ~ 20 — we just cap cleanly instead.
# Delaunay (QHull) crossover vs dense matrix: empirically dense is faster from
# d >= 4 upwards (measured d=5, n=1000 -> Delaunay ~1.8s vs dense ~0.2s).
# Keep Delaunay for d ∈ {2, 3} only.
DELAUNAY_DIM_CAP = 3
KDTREE_DIM_CAP = 16

# Log/exp safety rails — keep features in float64 range without overflow.
LOG_CAP = 690.0  # log(np.exp(690)) ~ 1e300, safely inside float64 max.


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
# K-NN helpers
# =============================================================================
def _knn_features(coords: np.ndarray, d: int) -> Dict[str, float]:
    """Local density stats via cKDTree. Caps at KDTREE_DIM_CAP dims — beyond
    that, k-NN loses meaning because concentration-of-measure collapses the
    nearest-neighbour gap."""
    if d > KDTREE_DIM_CAP:
        return {
            "nn1_dist_mean": float("nan"),
            "nn1_dist_cv": float("nan"),
            "nn2_dist_mean": float("nan"),
            "nn_gap_ratio": float("nan"),
        }
    n = coords.shape[0]
    if n < 3:
        return {
            "nn1_dist_mean": 0.0,
            "nn1_dist_cv": 0.0,
            "nn2_dist_mean": 0.0,
            "nn_gap_ratio": 1.0,
        }
    tree = cKDTree(coords)
    # query k=3: self (0), 1st NN, 2nd NN.
    dd, _ = tree.query(coords, k=3)
    nn1 = dd[:, 1]
    nn2 = dd[:, 2]
    nn1_mean = float(np.mean(nn1))
    nn1_std = float(np.std(nn1))
    nn2_mean = float(np.mean(nn2))
    return {
        "nn1_dist_mean": nn1_mean,
        "nn1_dist_cv": (nn1_std / nn1_mean) if nn1_mean > 1e-12 else 0.0,
        "nn2_dist_mean": nn2_mean,
        "nn_gap_ratio": (nn2_mean / nn1_mean) if nn1_mean > 1e-12 else 1.0,
    }


def _intrinsic_dim_2nn(coords: np.ndarray, d: int) -> float:
    """Facco et al. 2017 two-nearest-neighbour intrinsic-dim estimator:
    mu_i = r_2(i) / r_1(i) >= 1; P(mu > x) = x^{-d_intrinsic};
    MLE: d_hat = N / sum(log mu_i).

    Returns NaN above KDTREE_DIM_CAP ambient dim (nearest-neighbour stats
    become unreliable)."""
    if d > KDTREE_DIM_CAP:
        return float("nan")
    n = coords.shape[0]
    if n < 3:
        return float(d)
    tree = cKDTree(coords)
    dd, _ = tree.query(coords, k=3)
    nn1 = dd[:, 1]
    nn2 = dd[:, 2]
    # Drop any coincident / near-coincident points to avoid log(1)=0 blow-ups.
    mask = (nn1 > 1e-12) & (nn2 > nn1 * (1 + 1e-9))
    if not mask.any():
        return float(d)
    mu = nn2[mask] / nn1[mask]
    log_mu = np.log(mu)
    total = float(np.sum(log_mu))
    if total <= 1e-12:
        return float(d)
    return float(mask.sum() / total)


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
    eigvals = _pca_eigenvalues(coords)
    lam_sum = float(np.sum(eigvals))
    lam_sq_sum = float(np.sum(eigvals ** 2))
    out["pca_e1_share"] = float(eigvals[0] / lam_sum) if lam_sum > 1e-12 else 1.0
    out["pca_effective_rank"] = float(lam_sum ** 2 / lam_sq_sum) if lam_sq_sum > 1e-12 else 1.0
    obb_vol, log_obb_vol = _obb_volume(coords)
    out["obb_volume"] = obb_vol
    out["log_obb_volume"] = log_obb_vol
    # obb_shrinkage in [0, 1]: obb is always <= axis-aligned bounding box.
    out["obb_shrinkage"] = float(obb_vol / out["bounding_hypervolume"]) if out["bounding_hypervolume"] > 1e-12 else 0.0

    # --- Group 5 (Tier 2): Local density via cKDTree --------------------------
    out.update(_knn_features(coords, dimension))

    # --- Group 6 (Tier 3): Second-order + manifold ---------------------------
    # log-determinant of coord covariance (Mahalanobis-style log-volume, safe
    # in any dimension). Zero eigenvalues are clipped with a floor.
    eig_floor = np.maximum(eigvals, 1e-12)
    out["pca_log_det"] = float(np.sum(np.log(eig_floor)))
    out["mst_edge_pca_e1_share"] = _mst_edge_pca_e1_share(coords, rows, cols)
    out["intrinsic_dim_2nn"] = _intrinsic_dim_2nn(coords, dimension)

    return out


# List of identifier columns that the CSV carries alongside features.
IDENTIFIER_COLUMNS = ("instance_name", "optimal_cost", "split")

# List of metadata / size / dimension columns (also features the model uses).
METADATA_FEATURES = ("n_customers", "dimension", "grid_size")


def feature_columns_for_training() -> list[str]:
    """All candidate feature names in a deterministic order. Excludes the
    identifier / target columns."""
    return [
        # V3 metadata / scale
        "n_customers", "dimension", "grid_size",
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
        # V4 Tier 2
        "nn1_dist_mean", "nn1_dist_cv", "nn2_dist_mean", "nn_gap_ratio",
        # V4 Tier 3
        "pca_log_det", "mst_edge_pca_e1_share", "intrinsic_dim_2nn",
    ]
