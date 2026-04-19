"""TSPLIB file writers used to feed Concorde/LKH.

Two variants share the same shape but use different scaling rules:
- ``save_fast``: grid-only scale (get_scale_factor)
- ``save_robust``: grid-and-N scale (get_robust_scale_factor), returns the
  scale so callers can undo it on the solver's output.
"""
import numpy as np

from solvers.config import get_scale_factor, get_robust_scale_factor
from solvers.distance import compute_distance_matrix


def _write_tsplib(file_path: str, dist_matrix: np.ndarray, tsp_name: str, comment: str) -> None:
    n = dist_matrix.shape[0]
    with open(file_path, "w") as f:
        f.write(f"NAME : {tsp_name}\nTYPE : TSP\nCOMMENT : {comment}\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            f.write(" ".join(str(x) for x in dist_matrix[i]) + "\n")
        f.write("EOF\n")


def save_fast(file_path: str, coords: np.ndarray, tsp_name: str, grid_size: int) -> float:
    scale = get_scale_factor(float(grid_size))
    dist_matrix = compute_distance_matrix(coords, scale)
    _write_tsplib(file_path, dist_matrix, tsp_name, f"Grid={grid_size} Scale={scale}")
    return scale


def save_robust(file_path: str, coords: np.ndarray, tsp_name: str, grid_size: int) -> float:
    n = len(coords)
    scale = get_robust_scale_factor(float(grid_size), n)
    dist_matrix = compute_distance_matrix(coords, scale)
    _write_tsplib(file_path, dist_matrix, tsp_name, f"Grid={grid_size} N={n} Scale={scale:.2f}")
    return scale
