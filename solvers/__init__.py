"""Solver wrappers for Concorde and LKH.

Pass coordinates + grid_size in, get (tour_length, runtime, tour_nodes) out.
"""
from solvers.concorde import run_concorde, run_concorde_robust
from solvers.lkh import run_lkh
from solvers.config import (
    CONCORDE_WSL_BIN,
    LKH_EXECUTABLE_PATH,
    SOLVER_SCRATCH_DIR,
    get_scale_factor,
    get_robust_scale_factor,
)
from solvers.distance import compute_distance_matrix, compute_tour_length_numba

__all__ = [
    "run_concorde",
    "run_concorde_robust",
    "run_lkh",
    "CONCORDE_WSL_BIN",
    "LKH_EXECUTABLE_PATH",
    "SOLVER_SCRATCH_DIR",
    "get_scale_factor",
    "get_robust_scale_factor",
    "compute_distance_matrix",
    "compute_tour_length_numba",
]
