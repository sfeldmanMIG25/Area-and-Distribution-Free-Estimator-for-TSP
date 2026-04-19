"""Numba-compiled distance matrix and tour length kernels.

Both enforce a minimum edge weight of 1 to keep integer solvers well-posed on
instances where scaled floating-point distances would round to zero.
"""
import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def compute_distance_matrix(coords: np.ndarray, scale_factor: float) -> np.ndarray:
    n, d = coords.shape
    dist_matrix = np.zeros((n, n), dtype=np.int32)

    for i in prange(n):
        for j in prange(i + 1, n):
            dist_sq = 0.0
            for k in range(d):
                diff = coords[i, k] - coords[j, k]
                dist_sq += diff * diff

            dist_val = np.sqrt(dist_sq) * scale_factor + 0.5
            dist_int = int(dist_val)

            if dist_int == 0:
                dist_int = 1

            dist_matrix[i, j] = dist_int
            dist_matrix[j, i] = dist_int

    return dist_matrix


@njit(fastmath=True, cache=True)
def compute_tour_length_numba(coords: np.ndarray, tour: np.ndarray, scale_factor: float) -> int:
    n = len(tour)
    total = 0

    for i in range(n):
        j = (i + 1) % n
        node1 = tour[i] - 1
        node2 = tour[j] - 1

        dist_sq = 0.0
        for k in range(coords.shape[1]):
            diff = coords[node1, k] - coords[node2, k]
            dist_sq += diff * diff

        dist_val = np.sqrt(dist_sq) * scale_factor + 0.5
        dist_int = int(dist_val)

        if dist_int == 0 and node1 != node2:
            dist_int = 1

        total += dist_int

    return total
