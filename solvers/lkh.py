"""LKH-3 wrapper. Calls the Windows LKH executable in a scoped sandbox dir."""
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import List, Tuple

import numpy as np

from solvers import tsplib_io
from solvers.config import LKH_EXECUTABLE_PATH, SOLVER_SCRATCH_DIR, get_scale_factor
from solvers.distance import compute_tour_length_numba


def run_lkh(
    instance_name: str,
    coords: np.ndarray,
    d: int,
    grid_size: int,
    runs: int = 1,
    time_limit_s: int = 600,
    proc_timeout_s: int | None = None,
) -> Tuple[int, float, List[int]]:
    """Run LKH-3. Returns (tour_length, runtime, tour).

    ``time_limit_s`` is LKH's own TIME_LIMIT, which bounds the optimisation loop
    but not preprocessing (distance matrix, candidate set). Pass
    ``proc_timeout_s`` to additionally hard-bound the process; it raises
    ``subprocess.TimeoutExpired``. Default None keeps the previous behaviour.
    """
    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(SOLVER_SCRATCH_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    tsp_path = os.path.join(run_dir, f"{run_id}.tsp")
    par_path = os.path.join(run_dir, f"{run_id}.par")
    tour_path = os.path.join(run_dir, f"{run_id}.tour")

    try:
        tsplib_io.save_fast(tsp_path, coords, run_id, grid_size)

        with open(par_path, "w") as f:
            f.write(f"PROBLEM_FILE = {os.path.basename(tsp_path)}\n")
            f.write(f"TOUR_FILE = {os.path.basename(tour_path)}\n")
            f.write(f"RUNS = {runs}\n")
            f.write(f"TIME_LIMIT = {time_limit_s}\n")

        start_time = time.perf_counter()
        result = subprocess.run(
            [LKH_EXECUTABLE_PATH, os.path.basename(par_path)],
            capture_output=True, text=True, cwd=run_dir, timeout=proc_timeout_s,
        )
        lkh_time = time.perf_counter() - start_time

        if result.returncode != 0:
            raise ValueError(f"LKH error: {result.stderr}")
        if not os.path.exists(tour_path):
            raise ValueError("No LKH tour file")

        with open(tour_path, "r") as f:
            tour_content = f.read()
        match = re.search(r"TOUR_SECTION\s*([\s\d-]*?)\s*EOF", tour_content, re.DOTALL)
        if not match:
            raise ValueError("LKH tour parse failed")
        tour_nodes = [int(n) for n in match.group(1).strip().split() if int(n) != -1]

        scale = get_scale_factor(float(grid_size))
        tour_length = compute_tour_length_numba(coords, np.array(tour_nodes), scale)
        return tour_length, lkh_time, tour_nodes

    finally:
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
