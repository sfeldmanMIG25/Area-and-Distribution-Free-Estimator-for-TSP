"""Solver binary paths, scratch directory, and distance scaling helpers."""
import os

# Concorde lives in WSL but is not on PATH; invoke by absolute path.
CONCORDE_WSL_BIN = "/home/catst/concorde/TSP/concorde"

# LKH-3 Windows executable.
LKH_EXECUTABLE_PATH = "C:\\LKH\\LKH-3.exe"

# Per-run sandbox directory for solver I/O (Windows path; WSL sees it via /mnt/c).
SOLVER_SCRATCH_DIR = "C:\\Temp_TSP_Scratch"
os.makedirs(SOLVER_SCRATCH_DIR, exist_ok=True)


def get_scale_factor(grid_size: float) -> float:
    """Fixed scaling: normalize grid to a virtual ~10,000 for integer edge weights."""
    if grid_size <= 100:
        return 100.0
    if grid_size <= 1000:
        return 10.0
    return 1.0


def get_robust_scale_factor(grid_size: float, n_customers: int) -> float:
    """N-dependent scaling. Small-N instances are scaled down to avoid overflow."""
    if grid_size <= 100:
        base_scale = 100.0
    elif grid_size <= 1000:
        base_scale = 10.0
    else:
        base_scale = 1.0

    saturation_point = 500.0
    dampener = min(1.0, n_customers / saturation_point)
    return base_scale * dampener
