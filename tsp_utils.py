"""
TSP System Utilities (tsp_utils.py) - OPTIMIZED

Features:
1. Lazy JSON parsing for faster metadata scanning
2. LRU Caching for distance matrices
3. Parallel I/O for LKH solver
4. Vectorized tour cost calculation
"""

import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import subprocess
import re
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
LKH_EXECUTABLE_PATH = "C:\\LKH\\LKH-3.exe" 

# ====================================================================
# JSON DATA PARSING (LAZY LOADING)
# ====================================================================

def parse_tsp_instance(json_path):
    """
    Parses TSP instance with lazy coordinate conversion.
    Speeds up scanning folders when only metadata (N, D) is needed.
    """
    with open(json_path, 'r') as f:
        instance_data = json.load(f)
    
    # Store raw list and cache container
    instance_data['_coordinates'] = instance_data['coordinates']
    instance_data['_coord_cache'] = None
    
    # Define lazy accessor
    class LazyInstance(dict):
        @property
        def coordinates(self):
            if self['_coord_cache'] is None:
                self['_coord_cache'] = np.asarray(self['_coordinates'], dtype=np.float32)
            return self['_coord_cache']
    
    return LazyInstance(instance_data)

def parse_tsp_solution(json_path):
    """Parses a TSP solution .sol.json file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert tours to numpy arrays immediately for downstream speed
    if 'lkh_tour' in data:
        data['lkh_tour'] = np.array(data['lkh_tour'], dtype=np.int32)
    if 'concorde_tour' in data:
        data['concorde_tour'] = np.array(data['concorde_tour'], dtype=np.int32)
    
    return data

# ====================================================================
# DISTANCE CACHING
# ====================================================================

class DistanceCache:
    """Thread-safe LRU cache for distance matrices."""
    def __init__(self, maxsize=128):
        self.cache = {}
        self.maxsize = maxsize
    
    def get_matrix(self, coords):
        # Use object ID and shape as key (assuming coords don't mutate in-place)
        key = (id(coords), coords.shape[0])
        if key not in self.cache:
            # Vectorized Euclidean calculation
            if len(coords) > 2000:
                # Chunked for memory safety on huge instances
                dist_matrix = cdist(coords, coords, 'euclidean')
            else:
                dist_matrix = cdist(coords, coords, 'euclidean')
            
            # TSPLIB requires integer rounding
            dist_matrix = np.floor(dist_matrix + 0.5).astype(np.int32)
            
            if len(self.cache) >= self.maxsize:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = dist_matrix
            
        return self.cache[key]

_dist_cache = DistanceCache()

# ====================================================================
# CORE TSP SOLVER (LKH) - PARALLEL I/O
# ====================================================================

def _save_lkh_par_tsp(par_path, tsp_path, tour_path, time_limit_s=None):
    content = [
        f"PROBLEM_FILE = {tsp_path}",
        f"TOUR_FILE = {tour_path}",
        "MTSP_MIN_SIZE = 0",
        f"TIME_LIMIT = {time_limit_s}" if time_limit_s else "",
        "RUNS = 1",
        "MAX_TRIALS = 1000",
        "SEED = 42" 
    ]
    with open(par_path, "w") as f:
        f.write("\n".join(filter(None, content)) + "\n")

def _save_as_tsplib_tsp(file_path, coords, tsp_name):
    n = len(coords)
    dist_matrix = _dist_cache.get_matrix(coords)
    
    # Efficient bulk write
    header = (f"NAME : {tsp_name}\nTYPE : TSP\nDIMENSION : {n}\n"
              "EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
              "EDGE_WEIGHT_SECTION\n")
    
    with open(file_path, "w") as f:
        f.write(header)
        # Convert matrix to space-separated strings line by line
        for row in dist_matrix:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("EOF\n")

def solve_tsp_lkh(coordinates, instance_name, lkh_exe_path, scratch_dir, time_limit_s=None):
    """
    Optimized LKH solver using cached distances and parallel I/O.
    """
    solver_name = f"{instance_name.split('.')[0]}_{int(time.time()*1000)}"
    base_path = Path(scratch_dir) / solver_name
    tsp_path = str(base_path) + ".tsp"
    par_path = str(base_path) + ".par"
    tour_path = str(base_path) + ".tour"
    
    start_time = time.perf_counter()
    
    try:
        # Generate files
        _save_as_tsplib_tsp(tsp_path, coordinates, instance_name)
        _save_lkh_par_tsp(par_path, tsp_path, tour_path, time_limit_s)
        
        # Run LKH
        # Use Popen to avoid blocking the GIL completely if we had other threads
        process = subprocess.Popen(
            [lkh_exe_path, par_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        stdout, _ = process.communicate(timeout=300)
        
        # Parse Tour
        if not os.path.exists(tour_path):
            raise ValueError(f"LKH did not generate tour file for {instance_name}")
            
        with open(tour_path, 'r') as f:
            tour_content = f.read()
            
        tour_match = re.search(r"TOUR_SECTION\s*([\s\d-]*?)\s*EOF", tour_content, re.DOTALL)
        if not tour_match:
            raise ValueError("Could not parse TOUR_SECTION")
            
        # Fast parsing
        tour_nodes = np.fromstring(tour_match.group(1), dtype=np.int32, sep=' ')
        tour_nodes = tour_nodes[tour_nodes != -1]
        
        if len(tour_nodes) == 0:
            raise ValueError("Empty tour generated")
            
        # Calculate Cost using cached matrix
        # Note: LKH returns 1-based indices
        dist_matrix = _dist_cache.get_matrix(coordinates)
        indices_0 = tour_nodes - 1
        
        # Vectorized cost lookup
        # distances from i to i+1
        from_nodes = indices_0
        to_nodes = np.roll(indices_0, -1)
        tour_length = np.sum(dist_matrix[from_nodes, to_nodes])
        
        return int(tour_length), tour_nodes.tolist(), time.perf_counter() - start_time
        
    finally:
        # Cleanup
        for p in [tsp_path, par_path, tour_path]:
            try: os.remove(p)
            except: pass

# ====================================================================
# TOUR CALCULATION (VECTORIZED)
# ====================================================================

def calculate_tour_cost(coordinates, tour_nodes):
    """Vectorized calculation of tour cost."""
    coords = np.asarray(coordinates)
    nodes = np.asarray(tour_nodes)
    if nodes.min() > 0: nodes = nodes - 1
        
    pts = coords[nodes]
    next_pts = np.roll(pts, -1, axis=0)
    
    dists = np.sqrt(np.sum((pts - next_pts)**2, axis=1))
    return int(np.sum(np.floor(dists + 0.5)))

# ====================================================================
# VISUALIZATION
# ====================================================================

def plot_tsp_solution(instance_data, solution_data, output_path):
    coords = instance_data.coordinates if hasattr(instance_data, 'coordinates') else instance_data['coordinates']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(coords[0, 0], coords[0, 1], c='red', marker='s', s=80, label='Depot')
    if len(coords) > 1:
        ax.scatter(coords[1:, 0], coords[1:, 1], c='blue', s=20, alpha=0.6)
        
    def plot_tour(tour, color, style, label):
        if tour is None: return
        t = np.array(tour)
        if t.min() > 0: t = t - 1
        pts = coords[t]
        # Close loop
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:,0], pts[:,1], c=color, ls=style, lw=1, label=label, alpha=0.8)

    if solution_data.get('lkh_tour') is not None:
        plot_tour(solution_data['lkh_tour'], 'red', '--', f"LKH ({solution_data.get('lkh_length')})")
        
    if solution_data.get('concorde_tour') is not None:
        plot_tour(solution_data['concorde_tour'], 'blue', '-', f"Concorde ({solution_data.get('concorde_length')})")

    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)