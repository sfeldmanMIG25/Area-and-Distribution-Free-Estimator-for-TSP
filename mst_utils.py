"""Unified Euclidean MST computation.

Single entry point: ``compute_mst(coords, cache_path=None)``. Dense primary;
on ``MemoryError`` falls back to a dimension-appropriate exact method:
  d in {2, 3} -> Delaunay + Kruskal (O(n) memory)
  d >= 4      -> blocked Prim (numba, O(n) memory, O(n^2) time)

Returns an ``MSTResult`` that bundles edge weights, endpoints, degrees, and
method tag. Downstream feature code should read from this object instead of
recomputing an scipy CSR MST. A lightweight ``.npz`` cache is supported so the
dataset-wide feature pipeline and the v3/v4 benchmark scripts can share one
computation per instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    _HAVE_NUMBA = False


CACHE_VERSION = 1

# Dense MST allocates an n*n float32 distance matrix plus scipy's internal
# buffers. If the estimated working set exceeds this budget (bytes), we skip
# the dense attempt and dispatch to the dimension-appropriate backend directly
# instead of waiting for a slow MemoryError. Override with env var
# ``MST_DENSE_BUDGET_GB``.
_DENSE_BUDGET_BYTES = int(float(os.environ.get("MST_DENSE_BUDGET_GB", "8")) * (1 << 30))


def _dense_is_feasible(n: int) -> bool:
    # n*n float32 dist + scipy sparse buffers (~3x dense matrix in worst case).
    est = n * n * 4 * 4
    return est <= _DENSE_BUDGET_BYTES


@dataclass
class MSTResult:
    """Exact Euclidean MST result.

    Attributes
    ----------
    n : int
        Number of vertices.
    edges : np.ndarray, shape (n-1,), float32
        Edge weights.
    endpoints : np.ndarray, shape (n-1, 2), int32
        Vertex-index pairs. Not guaranteed ordered.
    degrees : np.ndarray, shape (n,), int32
        Vertex degrees in the MST.
    method : str
        One of {"dense", "delaunay", "blocked_prim"}. Records which backend
        actually produced the tree.
    """

    n: int
    edges: np.ndarray
    endpoints: np.ndarray
    degrees: np.ndarray
    method: str

    @property
    def total_length(self) -> float:
        return float(self.edges.sum())

    @property
    def edge_count(self) -> int:
        return int(self.edges.shape[0])

    def to_csr(self) -> csr_matrix:
        """Reconstruct the one-directional CSR matrix scipy's MST returns."""
        u = self.endpoints[:, 0]
        v = self.endpoints[:, 1]
        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        return csr_matrix((self.edges, (lo, hi)), shape=(self.n, self.n))

    def save(self, path: str) -> None:
        np.savez(
            path,
            version=np.int32(CACHE_VERSION),
            n=np.int32(self.n),
            edges=self.edges.astype(np.float32, copy=False),
            endpoints=self.endpoints.astype(np.int32, copy=False),
            degrees=self.degrees.astype(np.int32, copy=False),
            method=np.array(self.method),
        )

    @classmethod
    def load(cls, path: str) -> "MSTResult":
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != CACHE_VERSION:
                raise ValueError(f"MST cache version mismatch at {path}")
            return cls(
                n=int(z["n"]),
                edges=z["edges"].astype(np.float32),
                endpoints=z["endpoints"].astype(np.int32),
                degrees=z["degrees"].astype(np.int32),
                method=str(z["method"]),
            )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _from_csr(mst_csr: csr_matrix, n: int, method: str) -> MSTResult:
    coo = mst_csr.tocoo()
    endpoints = np.stack([coo.row, coo.col], axis=1).astype(np.int32)
    edges = coo.data.astype(np.float32)
    degrees = np.zeros(n, dtype=np.int32)
    np.add.at(degrees, endpoints[:, 0], 1)
    np.add.at(degrees, endpoints[:, 1], 1)
    return MSTResult(n=n, edges=edges, endpoints=endpoints, degrees=degrees, method=method)


def _dense_mst(coords: np.ndarray) -> MSTResult:
    n = coords.shape[0]
    dm = cdist(coords, coords, "euclidean").astype(np.float32)
    np.fill_diagonal(dm, 0.0)
    mst_csr = minimum_spanning_tree(dm)
    del dm
    return _from_csr(mst_csr, n, method="dense")


def _delaunay_mst(coords: np.ndarray) -> MSTResult:
    n = coords.shape[0]
    tri = Delaunay(coords)
    simplices = tri.simplices
    d = coords.shape[1]
    if d == 2:
        edges_raw = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    elif d == 3:
        edges_raw = np.vstack([
            simplices[:, [0, 1]], simplices[:, [0, 2]], simplices[:, [0, 3]],
            simplices[:, [1, 2]], simplices[:, [1, 3]], simplices[:, [2, 3]]
        ])
    else:
        k = simplices.shape[1]
        combo_indices = np.array(list(combinations(range(k), 2)))
        edges_raw = np.vstack([simplices[:, c] for c in combo_indices])

    if edges_raw.size == 0:
        raise RuntimeError("Delaunay produced no edges")

    edges_raw.sort(axis=1)
    arr = np.unique(edges_raw, axis=0).astype(np.int32)
    diffs = coords[arr[:, 0]] - coords[arr[:, 1]]
    dists = np.linalg.norm(diffs, axis=1).astype(np.float32)
    rows = np.concatenate([arr[:, 0], arr[:, 1]])
    cols = np.concatenate([arr[:, 1], arr[:, 0]])
    data = np.concatenate([dists, dists])
    sp = csr_matrix((data, (rows, cols)), shape=(n, n))
    mst_csr = minimum_spanning_tree(sp)
    return _from_csr(mst_csr, n, method="delaunay")



if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _blocked_prim_kernel(coords: np.ndarray):
        """Exact Euclidean MST by Prim without materializing the full distance
        matrix. O(n) memory, O(n^2) time."""
        n = coords.shape[0]
        INF = np.float32(np.inf)
        min_dist = np.full(n, INF, dtype=np.float32)
        parent = np.full(n, -1, dtype=np.int32)
        in_tree = np.zeros(n, dtype=np.bool_)

        edges = np.empty(n - 1, dtype=np.float32)
        endpoints = np.empty((n - 1, 2), dtype=np.int32)

        min_dist[0] = 0.0
        edge_idx = 0
        for _ in range(n):
            u = -1
            best = INF
            for i in range(n):
                if (not in_tree[i]) and min_dist[i] < best:
                    best = min_dist[i]
                    u = i
            if u == -1:
                break
            in_tree[u] = True
            if parent[u] != -1:
                endpoints[edge_idx, 0] = parent[u]
                endpoints[edge_idx, 1] = u
                edges[edge_idx] = best
                edge_idx += 1
            # Update distances from u to all non-tree vertices.
            cu0 = coords[u, 0]
            # Generic-dimension distance update.
            for v in range(n):
                if in_tree[v]:
                    continue
                s = np.float32(0.0)
                for k in range(coords.shape[1]):
                    diff = coords[v, k] - coords[u, k]
                    s += diff * diff
                d = np.sqrt(s)
                if d < min_dist[v]:
                    min_dist[v] = d
                    parent[v] = u
        return edges[:edge_idx], endpoints[:edge_idx]


def _blocked_prim_mst(coords: np.ndarray) -> MSTResult:
    if not _HAVE_NUMBA:
        raise RuntimeError("Blocked Prim requires numba")
    coords32 = np.ascontiguousarray(coords, dtype=np.float32)
    edges, endpoints = _blocked_prim_kernel(coords32)
    n = coords32.shape[0]
    degrees = np.zeros(n, dtype=np.int32)
    np.add.at(degrees, endpoints[:, 0], 1)
    np.add.at(degrees, endpoints[:, 1], 1)
    return MSTResult(
        n=n,
        edges=edges.astype(np.float32),
        endpoints=endpoints.astype(np.int32),
        degrees=degrees,
        method="blocked_prim",
    )


def _fallback_mst(coords: np.ndarray, n: int, d: int) -> MSTResult:
    if d <= 3 and n >= d + 2:
        try:
            return _delaunay_mst(coords)
        except Exception:
            pass
    return _blocked_prim_mst(coords)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_mst(coords: np.ndarray, cache_path: Optional[str] = None) -> MSTResult:
    """Exact Euclidean MST with memory-aware fallback.

    Parameters
    ----------
    coords : (n, d) array
        Point coordinates. Converted to float32 internally.
    cache_path : str, optional
        If given, loads the MST from this ``.npz`` file when present, otherwise
        computes and saves to it. Useful for dataset-wide feature pipelines and
        downstream benchmark scripts so each instance's MST is computed once.
    """
    if cache_path is not None and os.path.exists(cache_path):
        try:
            return MSTResult.load(cache_path)
        except Exception:
            pass  # fall through to recompute

    coords = np.ascontiguousarray(coords, dtype=np.float32)
    n, d = coords.shape
    if n < 2:
        empty = MSTResult(
            n=n,
            edges=np.zeros(0, dtype=np.float32),
            endpoints=np.zeros((0, 2), dtype=np.int32),
            degrees=np.zeros(n, dtype=np.int32),
            method="trivial",
        )
        if cache_path is not None:
            empty.save(cache_path)
        return empty

    # Dynamic n x d dispatch. Thresholds come from measured crossovers on this
    # project (see mst_utils benchmarks): in d<=3 Delaunay dominates dense at
    # all n; in d>=4 Qhull simplex count explodes so dense wins up to its
    # memory limit; blocked-Prim is the universal O(n)-memory fallback.
    result = None
    if d <= 3 and n >= d + 2:
        try:
            result = _delaunay_mst(coords)
        except Exception:
            result = None
    if result is None and _dense_is_feasible(n):
        try:
            result = _dense_mst(coords)
        except MemoryError:
            result = None
    if result is None:
        result = _blocked_prim_mst(coords)

    if cache_path is not None:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            result.save(cache_path)
        except Exception:
            pass
    return result


def mst_length(coords: np.ndarray, cache_path: Optional[str] = None) -> float:
    """Convenience: exact Euclidean MST total length."""
    return compute_mst(coords, cache_path=cache_path).total_length
