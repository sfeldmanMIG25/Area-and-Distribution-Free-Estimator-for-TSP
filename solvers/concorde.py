"""Concorde wrappers. Call ``run_concorde`` for the default pipeline or
``run_concorde_robust`` for the N-dependent scaling used by verification.

For N <= EXACT_N_MAX we skip Concorde (it crashes silently on tiny instances)
and solve exactly via Held-Karp.
"""
import os
import shutil
import subprocess
import time
import uuid
from typing import List, Tuple

import numpy as np

from solvers import tsplib_io
from solvers.config import CONCORDE_WSL_BIN, SOLVER_SCRATCH_DIR, get_scale_factor, get_robust_scale_factor
from solvers.distance import compute_tour_length_numba
from solvers.exact import EXACT_N_MAX, held_karp


def _wsl_path(win_path: str) -> str:
    """Deterministic C:\\foo\\bar -> /mnt/c/foo/bar. No subprocess."""
    p = win_path.replace("\\", "/")
    if len(p) < 2 or p[1] != ":":
        raise ValueError(f"Expected drive-qualified path, got: {win_path}")
    return f"/mnt/{p[0].lower()}{p[2:]}"


def _read_tour(tour_file_win: str) -> List[int]:
    with open(tour_file_win, "r") as f:
        lines = f.read().splitlines()
    if len(lines) < 2:
        raise ValueError("Concorde output invalid")
    return [int(n) + 1 for n in " ".join(lines[1:]).strip().split()]


def _invoke_concorde(coords: np.ndarray, scale: float, grid_size: int, tsp_writer) -> Tuple[int, float, List[int]]:
    run_id = str(uuid.uuid4())[:12]
    run_dir = os.path.join(SOLVER_SCRATCH_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    tsp_path_win = os.path.join(run_dir, f"{run_id}.tsp")

    try:
        tsp_writer(tsp_path_win, coords, run_id, grid_size)
        wsl_tsp = _wsl_path(tsp_path_win)
        wsl_tour = wsl_tsp.replace(".tsp", ".tour")

        start = time.perf_counter()
        result = subprocess.run(
            ["wsl", CONCORDE_WSL_BIN, "-o", wsl_tour, wsl_tsp],
            capture_output=True, text=True, cwd=run_dir,
        )
        runtime = time.perf_counter() - start

        if result.returncode != 0:
            raise RuntimeError(f"Concorde failed (rc={result.returncode}): {result.stderr.strip() or result.stdout.strip()}")

        tour_file_win = tsp_path_win.replace(".tsp", ".tour")
        if not os.path.exists(tour_file_win):
            raise FileNotFoundError(f"Concorde produced no tour file. stderr: {result.stderr.strip()}")

        tour_nodes = _read_tour(tour_file_win)
        tour_length = compute_tour_length_numba(coords, np.array(tour_nodes), scale)
        return tour_length, runtime, tour_nodes
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def run_concorde(solver_name: str, coords: np.ndarray, d: int, grid_size: int, timeout_s: int = 5000) -> Tuple[int, float, List[int]]:
    """Concorde with fast (grid-only) scaling. Returns (tour_length, runtime, tour).
    Small-N instances bypass Concorde via Held-Karp."""
    scale = get_scale_factor(float(grid_size))
    if len(coords) <= EXACT_N_MAX:
        start = time.perf_counter()
        length, tour = held_karp(coords, scale)
        return length, time.perf_counter() - start, tour
    return _invoke_concorde(coords, scale, grid_size, tsplib_io.save_fast)


def run_concorde_robust(coords: np.ndarray, grid_size: int) -> Tuple[int, float, List[int], float]:
    """Concorde with N-dependent scaling. Returns (tour_length, runtime, tour, scale).
    Small-N instances bypass Concorde via Held-Karp."""
    n = len(coords)
    scale = get_robust_scale_factor(float(grid_size), n)
    if n <= EXACT_N_MAX:
        start = time.perf_counter()
        length, tour = held_karp(coords, scale)
        return length, time.perf_counter() - start, tour, scale
    length, runtime, tour = _invoke_concorde(coords, scale, grid_size, tsplib_io.save_robust)
    return length, runtime, tour, scale
