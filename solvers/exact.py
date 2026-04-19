"""Exact TSP solver for small N via Held-Karp DP.

Concorde is unstable for very small instances (crashes silently on N<=5 or so).
For N <= EXACT_N_MAX we solve by bitmask DP instead.
"""
from typing import List, Tuple

import numpy as np

from solvers.distance import compute_distance_matrix

EXACT_N_MAX = 20


def held_karp(coords: np.ndarray, scale: float) -> Tuple[int, List[int]]:
    """Return (tour_length_int, tour_1indexed) using Held-Karp DP on the integer
    distance matrix (same matrix Concorde would see)."""
    dist = compute_distance_matrix(coords, scale)
    n = dist.shape[0]
    if n < 2:
        return 0, [1] if n == 1 else []

    INF = np.iinfo(np.int64).max // 4
    # dp[mask][j] = min cost to start at 0, visit set `mask`, end at j
    size = 1 << n
    dp = np.full((size, n), INF, dtype=np.int64)
    parent = np.full((size, n), -1, dtype=np.int32)
    dp[1][0] = 0

    for mask in range(1, size):
        if not (mask & 1):
            continue
        for j in range(n):
            if not (mask & (1 << j)) or dp[mask][j] >= INF:
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                nmask = mask | (1 << k)
                cand = dp[mask][j] + int(dist[j, k])
                if cand < dp[nmask][k]:
                    dp[nmask][k] = cand
                    parent[nmask][k] = j

    full = size - 1
    best_cost = INF
    best_end = -1
    for j in range(1, n):
        cand = dp[full][j] + int(dist[j, 0])
        if cand < best_cost:
            best_cost = cand
            best_end = j

    tour = [best_end]
    mask = full
    while parent[mask][tour[-1]] != -1:
        prev = int(parent[mask][tour[-1]])
        mask ^= 1 << tour[-1]
        tour.append(prev)
    tour.reverse()
    return int(best_cost), [n_ + 1 for n_ in tour]
