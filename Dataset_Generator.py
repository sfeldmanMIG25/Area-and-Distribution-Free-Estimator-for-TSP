import numpy as np
import os
import subprocess
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import re
from tqdm import tqdm
import gc
from collections import defaultdict, Counter
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Dict, Any
import numba
from numba import njit, prange
import struct
import warnings
import multiprocessing as mp
import uuid
import shutil  # Added for directory cleanup

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")
VISUALS_DIR_INST = os.path.join(ROOT_DIR, "visuals", "instances")
VISUALS_DIR_SOL = os.path.join(ROOT_DIR, "visuals", "solutions")

os.makedirs(INSTANCES_DIR, exist_ok=True)
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR_INST, exist_ok=True)
os.makedirs(VISUALS_DIR_SOL, exist_ok=True)

LKH_EXECUTABLE_PATH = "C:\\LKH\\LKH-3.exe"
SOLVER_SCRATCH_DIR = "C:\\Temp_TSP_Scratch"
os.makedirs(SOLVER_SCRATCH_DIR, exist_ok=True)

GRID_SIZE_LIST = [100, 1000, 10000]
N_CUSTOMERS_LIST = [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                    200, 300, 400, 500, 600, 700, 800, 900, 1000]
DIMENSION_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
N_PER_CONFIGURATION = 82

# --- DYNAMIC SCALING HELPER ---
def get_scale_factor(grid_size: float) -> float:
    """
    Dynamic scaling to normalize virtual grid to ~10,000.
    100   -> Scale 100 -> 10,000
    1000  -> Scale 10  -> 10,000
    10000 -> Scale 1   -> 10,000
    """
    if grid_size <= 100: return 100.0
    if grid_size <= 1000: return 10.0
    return 1.0

# --- OPTIMIZED NUMBA FUNCTIONS ---
@njit(fastmath=True, cache=True)
def compute_distance_matrix(coords: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Ultra-fast distance matrix computation.
    ENFORCES MINIMUM DISTANCE OF 1.
    """
    n, d = coords.shape
    dist_matrix = np.zeros((n, n), dtype=np.int32)
    
    for i in prange(n):
        for j in prange(i + 1, n): # Start at i+1 to skip diagonal
            dist_sq = 0.0
            for k in range(d):
                diff = coords[i, k] - coords[j, k]
                dist_sq += diff * diff
            
            # Apply scaling
            dist_val = np.sqrt(dist_sq) * scale_factor + 0.5
            dist_int = int(dist_val)
            
            # ENFORCE MINIMUM DISTANCE
            # If distinct points round to 0, force to 1
            if dist_int == 0:
                dist_int = 1
                
            dist_matrix[i, j] = dist_int
            dist_matrix[j, i] = dist_int
            
    return dist_matrix

@njit(fastmath=True, cache=True)
def compute_tour_length_numba(coords: np.ndarray, tour: np.ndarray, scale_factor: float) -> int:
    """Compute tour length with Minimum Distance enforcement."""
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
        
        # Enforce Minimum Distance
        if dist_int == 0 and node1 != node2:
            dist_int = 1
            
        total += dist_int
    
    return total

@njit(fastmath=True, cache=True)
def make_unique_numba(coords: np.ndarray, grid_size: float, seed: int) -> np.ndarray:
    """Vectorized uniqueness check with early exit."""
    np.random.seed(seed)
    n, d = coords.shape
    max_retries = 10
    tolerance = 1e-6
    
    for attempt in range(max_retries):
        unique = True
        for i in range(n - 1):
            for j in range(i + 1, n):
                same = True
                for k in range(d):
                    if abs(coords[i, k] - coords[j, k]) > tolerance:
                        same = False
                        break
                if same:
                    unique = False
                    # Perturb the second point
                    for k in range(d):
                        coords[j, k] += np.random.uniform(-grid_size*1e-5, grid_size*1e-5)
                        coords[j, k] = max(0.0, min(grid_size, coords[j, k]))
                    break
            if not unique:
                break
        
        if unique:
            return coords
    
    return coords

# --- BINARY FILE HANDLING ---
def save_instance_binary(instance_path: str, data: Dict[str, Any]) -> None:
    binary_path = instance_path.replace('.json', '.bin')
    n = data['n_customers']
    d = data['dimension']
    coords = np.array(data['coordinates'], dtype=np.float32)
    
    with open(binary_path, 'wb') as f:
        f.write(struct.pack('III', n, d, data['grid_size']))
        dist_str = ''.join(data['distribution_types'])
        f.write(struct.pack('I', len(dist_str)))
        f.write(dist_str.encode('ascii'))
        f.write(coords.tobytes())
    
    with open(instance_path, 'w') as f:
        json.dump(data, f)

def load_instance_binary(binary_path: str) -> Dict[str, Any]:
    with open(binary_path, 'rb') as f:
        n, d, grid_size = struct.unpack('III', f.read(12))
        dist_len = struct.unpack('I', f.read(4))[0]
        dist_str = f.read(dist_len).decode('ascii')
        coords = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)
    
    return {
        'n_customers': n,
        'dimension': d,
        'grid_size': grid_size,
        'distribution_types': list(dist_str),
        'coordinates': coords.tolist()
    }

# --- DISTRIBUTION GENERATOR ---
class DistributionGenerator:
    def __init__(self):
        self.cache = {}
    
    def get_rng(self, seed: int) -> np.random.Generator:
        if seed not in self.cache:
            self.cache[seed] = np.random.default_rng(seed)
        return self.cache[seed]
    
    def generate_1d_random(self, n: int, seed: int, grid_size: float, **kwargs) -> np.ndarray:
        rng = self.get_rng(seed)
        return rng.uniform(0, grid_size, n)
    
    def generate_1d_normal(self, n: int, seed: int, grid_size: float, **kwargs) -> np.ndarray:
        rng = self.get_rng(seed)
        result = rng.normal(loc=grid_size/2, scale=grid_size/6, size=n)
        np.clip(result, 0, grid_size, out=result)
        return result
    
    def generate_1d_clustered(self, n: int, seed: int, grid_size: float, **kwargs) -> np.ndarray:
        rng = self.get_rng(seed)
        clust_n = max(2, int(np.sqrt(n) / 4))
        centers = rng.uniform(0, grid_size, clust_n)
        assignments = rng.integers(0, clust_n, size=n)
        stdev = grid_size * 0.05
        result = rng.normal(loc=centers[assignments], scale=stdev)
        np.clip(result, 0, grid_size, out=result)
        return result

    def generate_1d_correlated(self, n: int, seed: int, grid_size: float, base=None, **kwargs) -> np.ndarray:
        rng = self.get_rng(seed)
        if base is None:
            return rng.uniform(0, grid_size, n)
        noise = rng.normal(loc=0, scale=grid_size/10, size=n)
        result = base + noise
        np.clip(result, 0, grid_size, out=result)
        return result

dist_gen = DistributionGenerator()
DISTRIBUTION_MAP_1D = {
    'r': dist_gen.generate_1d_random, 'n': dist_gen.generate_1d_normal,
    'c': dist_gen.generate_1d_clustered, 'o': dist_gen.generate_1d_clustered,
    'i': dist_gen.generate_1d_normal, 'p': dist_gen.generate_1d_random,
    't': dist_gen.generate_1d_random, 'g': dist_gen.generate_1d_random,
    'k': dist_gen.generate_1d_correlated, 'l': dist_gen.generate_1d_random,
    's': dist_gen.generate_1d_random, 'b': dist_gen.generate_1d_random,
    'x': dist_gen.generate_1d_random, 'e': dist_gen.generate_1d_random,
    'a': dist_gen.generate_1d_random, 'h': dist_gen.generate_1d_random
}
DIST_LETTERS = list(DISTRIBUTION_MAP_1D.keys())

# --- SOLVER HELPERS (UPDATED FOR DYNAMIC SCALING) ---
def _save_as_tsplib_fast(file_path: str, coords: np.ndarray, tsp_name: str, grid_size: int) -> None:
    n = len(coords)
    # Determine scale factor dynamically
    scale_factor = get_scale_factor(float(grid_size))
    
    # Compute matrix with minimum distance enforcement
    dist_matrix = compute_distance_matrix(coords, scale_factor)
    
    with open(file_path, "w") as f:
        f.write(f"NAME : {tsp_name}\nTYPE : TSP\nCOMMENT : Grid={grid_size} Scale={scale_factor}\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row_str = " ".join([str(x) for x in dist_matrix[i]])
            f.write(row_str + "\n")
        f.write("EOF\n")

# --- BATCH GENERATOR ---
class InstanceBatchGenerator:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def generate_batch(self, batch_params: List[Tuple]) -> List[str]:
        generated = []
        for params in batch_params:
            n, d, dist_letters, seed, seq_j, grid_size = params
            coords = np.zeros((n, d), dtype=np.float32)
            base = None
            for j in range(d):
                letter = dist_letters[j]
                func = DISTRIBUTION_MAP_1D[letter]
                if letter == 'k' and base is not None:
                    coords[:, j] = func(n, seed + j, grid_size, base=base)
                else:
                    coords[:, j] = func(n, seed + j, grid_size)
                if base is None and letter != 'k':
                    base = coords[:, j]
            
            coords = make_unique_numba(coords, float(grid_size), seed)
            
            dist_string = ''.join(dist_letters)
            instance_name = f"N{n}_D{d}_G{grid_size}_{dist_string}_{seq_j}"
            instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
            
            instance_data = {
                "instance_name": instance_name,
                "n_customers": n, "dimension": d, "grid_size": grid_size,
                "distribution_types": dist_letters, "generation_seed": seed,
                "coordinates": coords.tolist()
            }
            save_instance_binary(instance_path, instance_data)
            generated.append(instance_name)
        return generated

# --- SOLVER EXECUTION (ROBUST SANDBOXING) ---
def solve_instance_batch(instance_names: List[str]) -> List[str]:
    solved = []
    for instance_name in instance_names:
        solution_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
        if os.path.exists(solution_path):
            solved.append(instance_name)
            continue
        
        instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
        binary_path = instance_path.replace('.json', '.bin')
        
        if os.path.exists(binary_path):
            inst_data = load_instance_binary(binary_path)
        elif os.path.exists(instance_path):
            with open(instance_path, 'r') as f:
                inst_data = json.load(f)
        else:
            continue
        
        coords = np.array(inst_data['coordinates'], dtype=np.float64)
        d = inst_data['dimension']
        grid_size = inst_data['grid_size']
        scale_factor = get_scale_factor(float(grid_size))
        
        # --- SOLVER 1: CONCORDE ---
        concorde_final = None
        concorde_time = None
        concorde_tour = None
        concorde_error = None
        
        try:
            c_len, c_time, c_nodes = _run_concorde_fast(instance_name, coords, d, grid_size)
            concorde_final = c_len / scale_factor
            concorde_time = c_time
            concorde_tour = c_nodes
        except Exception as e:
            concorde_error = str(e)
            if "edge too long" not in str(e):
                print(f"Concorde error on {instance_name}: {e}")

        # --- SOLVER 2: LKH ---
        lkh_final = None
        lkh_time = None
        lkh_tour = None
        lkh_error = None
        
        try:
            l_len, l_time, l_nodes = _run_lkh_fast(instance_name, coords, d, grid_size)
            lkh_final = l_len / scale_factor
            lkh_time = l_time
            lkh_tour = l_nodes
        except Exception as e:
            lkh_error = str(e)
            print(f"LKH Failed for {instance_name}: {e}")

        if lkh_final is None and concorde_final is None:
            continue # Both failed

        # Determine Optimal
        if concorde_final is not None and (lkh_final is None or concorde_final < lkh_final):
            opt_cost = concorde_final
            opt_tour = concorde_tour
            opt_solver = "concorde"
        elif lkh_final is not None:
            opt_cost = lkh_final
            opt_tour = lkh_tour
            opt_solver = "lkh"
        else:
            opt_cost = -1
            opt_tour = []
            opt_solver = "failed"

        if concorde_final is not None and lkh_final is not None:
            lkh_gap = (lkh_final - concorde_final) / concorde_final * 100.0
        else:
            lkh_gap = None

        sol_data = {
            "instance_name": instance_name,
            "optimal_cost": opt_cost,
            "optimal_tour": opt_tour,
            "optimal_solver": opt_solver,
            "concorde_length": concorde_final,
            "concorde_time_s": concorde_time,
            "concorde_tour": concorde_tour,
            "concorde_error": concorde_error,
            "lkh_length": lkh_final,
            "lkh_time_s": lkh_time,
            "lkh_tour": lkh_tour,
            "lkh_gap_pct": lkh_gap
        }
        
        with open(solution_path, 'w') as f:
            json.dump(sol_data, f, indent=2)
        solved.append(instance_name)
    return solved

def _run_concorde_fast(solver_name: str, coords: np.ndarray, d: int, grid_size: int) -> Tuple[int, float, List[int]]:
    # Create SANDBOX Directory
    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(SOLVER_SCRATCH_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    tsp_path_win = os.path.join(run_dir, f"{run_id}.tsp")
    
    try:
        # Save TSP into the sandbox
        _save_as_tsplib_fast(tsp_path_win, coords, run_id, grid_size)
        
        # Get WSL paths
        # Note: subprocess cwd will set the working dir, so files are relative to it.
        # But we need full paths for wslpath translation if we use absolute logic.
        # Simpler: Use relative filenames since we set CWD.
        
        clean_tsp_path_win = tsp_path_win.replace('\\', '/')
        # Get WSL equivalent of the input file
        wsl_tsp_path = subprocess.run(["wsl", "wslpath", "-a", clean_tsp_path_win], 
                                      capture_output=True, text=True, check=True).stdout.strip()
        
        # Tour output in the SAME directory
        wsl_tour_file = wsl_tsp_path.replace('.tsp', '.tour')
        
        concorde_cmd = ["wsl", "timeout", "5000", "concorde", "-o", wsl_tour_file, wsl_tsp_path]
        
        start_time = time.perf_counter()
        # CRITICAL FIX: Run with cwd=run_dir so ALL dump files stay inside this folder
        result = subprocess.run(concorde_cmd, capture_output=True, text=True, cwd=run_dir)
        runtime = time.perf_counter() - start_time
        
        if result.returncode != 0:
            raise ValueError(f"Concorde failed: {result.stderr}")
        
        # Read tour (it should be in run_dir/run_id.tour)
        tour_file_win = tsp_path_win.replace('.tsp', '.tour')
        if not os.path.exists(tour_file_win):
             raise ValueError("Concorde did not produce tour file")
             
        with open(tour_file_win, 'r') as f:
            tour_content = f.read()
        
        lines = tour_content.splitlines()
        if len(lines) < 2: raise ValueError("Concorde output invalid")
        tour_nodes = [int(n) + 1 for n in " ".join(lines[1:]).strip().split()]
        
        scale = get_scale_factor(float(grid_size))
        tour_length = compute_tour_length_numba(coords, np.array(tour_nodes), scale)
        
        return tour_length, runtime, tour_nodes
        
    finally:
        # SCORCHED EARTH CLEANUP: Delete the entire folder
        if os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
            except:
                pass

def _run_lkh_fast(instance_name: str, coords: np.ndarray, d: int, grid_size: int) -> Tuple[int, float, List[int]]:
    # Create SANDBOX Directory
    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(SOLVER_SCRATCH_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    tsp_path = os.path.join(run_dir, f"{run_id}.tsp")
    par_path = os.path.join(run_dir, f"{run_id}.par")
    tour_path = os.path.join(run_dir, f"{run_id}.tour")
    
    try:
        _save_as_tsplib_fast(tsp_path, coords, run_id, grid_size)
        
        with open(par_path, "w") as f:
            f.write(f"PROBLEM_FILE = {os.path.basename(tsp_path)}\n") # Use relative name
            f.write(f"TOUR_FILE = {os.path.basename(tour_path)}\n")   # Use relative name
            f.write("RUNS = 1\n")
            f.write("TIME_LIMIT = 600\n") 
        
        start_time = time.perf_counter()
        # CRITICAL FIX: Run with cwd=run_dir
        result = subprocess.run([LKH_EXECUTABLE_PATH, os.path.basename(par_path)], 
                                capture_output=True, text=True, cwd=run_dir)
        lkh_time = time.perf_counter() - start_time
        
        if result.returncode != 0: raise ValueError(f"LKH error: {result.stderr}")
        if not os.path.exists(tour_path): raise ValueError("No LKH tour file")
        
        with open(tour_path, 'r') as f: tour_content = f.read()
        match = re.search(r"TOUR_SECTION\s*([\s\d-]*?)\s*EOF", tour_content, re.DOTALL)
        if not match: raise ValueError("LKH tour parse failed")
        tour_nodes = [int(n) for n in match.group(1).strip().split() if int(n) != -1]
        
        scale = get_scale_factor(float(grid_size))
        tour_length = compute_tour_length_numba(coords, np.array(tour_nodes), scale)
        
        return tour_length, lkh_time, tour_nodes
        
    finally:
        # SCORCHED EARTH CLEANUP
        if os.path.exists(run_dir):
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
            except:
                pass

def generate_batch_wrapper(batch_params):
    generator = InstanceBatchGenerator()
    return generator.generate_batch(batch_params)

# --- VISUALIZATIONS ---
# Global Helper
def plot_single_tour(tour, color, style, label, ax=None, d=2, coords=None):
    if not tour: return
    idx = np.array(tour) - 1
    t_coords = coords[idx]
    t_coords = np.vstack([t_coords, t_coords[0]])
    
    if d == 2:
        ax.plot(t_coords[:, 0], t_coords[:, 1], 
                c=color, linestyle=style, label=label, alpha=0.8, linewidth=1.5)
    else:
        ax.plot(t_coords[:, 0], t_coords[:, 1], t_coords[:, 2], 
                c=color, linestyle=style, label=label, alpha=0.8, linewidth=1.5)

def viz_worker(file_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not file_name.endswith('.json'):
        return
    
    instance_name = file_name[:-5]
    instance_path = os.path.join(INSTANCES_DIR, file_name)
    
    inst_vis_path = os.path.join(VISUALS_DIR_INST, f"{instance_name}.png")
    sol_vis_path = os.path.join(VISUALS_DIR_SOL, f"{instance_name}.png")
    sol_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
    has_sol = os.path.exists(sol_path)
    
    if os.path.exists(inst_vis_path) and (not has_sol or os.path.exists(sol_vis_path)):
        return 1
    
    try:
        with open(instance_path, 'r') as f:
            inst_data = json.load(f)
        
        d = inst_data['dimension']
        if d not in [2, 3]: return 1
        
        coords = np.array(inst_data['coordinates'])
        
        if not os.path.exists(inst_vis_path):
            fig = plt.figure(figsize=(6, 6))
            if d == 2:
                ax = fig.add_subplot(111)
                ax.scatter(coords[:, 0], coords[:, 1], s=10, c='black')
            else:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=10, c='black')
            plt.title(f"Instance: {instance_name}", fontsize=9)
            plt.tight_layout()
            plt.savefig(inst_vis_path, dpi=100)
            plt.close(fig)
        
        if has_sol and not os.path.exists(sol_vis_path):
            with open(sol_path, 'r') as f:
                sol_data = json.load(f)
            
            fig = plt.figure(figsize=(8, 6))
            if d == 2:
                ax = fig.add_subplot(111)
                ax.scatter(coords[:, 0], coords[:, 1], s=15, c='k', zorder=5)
            else:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=15, c='k', zorder=5)
            
            c_tour = sol_data.get('concorde_tour')
            c_len = sol_data.get('concorde_length', 0)
            if c_tour:
                plot_single_tour(c_tour, 'blue', '-', f'Concorde: {c_len:.2f}', ax=ax, d=d, coords=coords)
                
            l_tour = sol_data.get('lkh_tour')
            l_len = sol_data.get('lkh_length', 0)
            if l_tour:
                plot_single_tour(l_tour, 'red', '--', f'LKH: {l_len:.2f}', ax=ax, d=d, coords=coords)
            
            plt.title(f"Solutions: {instance_name}\nGap: {sol_data.get('lkh_gap_pct', 0):.4f}%", fontsize=10)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(sol_vis_path, dpi=100)
            plt.close(fig)
    except:
        pass
    finally:
        plt.close('all')
    return 1

def visualize_instances_and_solutions():
    print("\n4. Generating visualizations...")
    instance_files = list(os.listdir(INSTANCES_DIR))
    num_workers = min(8, max(1, os.cpu_count() // 2))
    
    with mp.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(viz_worker, instance_files, chunksize=10), 
                  total=len(instance_files), desc="Visualizing"))

def main_optimized():
    print("=== OPTIMIZED TSP GENERATOR (SANDBOXED) ===")
    try:
        print("\n1. Scanning existing files...")
        existing_counts = Counter()
        existing_solutions = set()
        instance_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json')]
        solution_files = [f for f in os.listdir(SOLUTIONS_DIR) if f.endswith('.sol.json')]
        filename_pattern = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)\.json$")
        
        for f in instance_files:
            match = filename_pattern.match(f)
            if match:
                n, d, g, dist_str, seq_id = match.groups()
                existing_counts[(int(n), int(d), int(g))] += 1
        
        for f in solution_files:
            existing_solutions.add(f[:-9])
            
        print(f"   Indexed {len(instance_files)} instances and {len(solution_files)} solutions.")

        batch_generator = InstanceBatchGenerator(batch_size=50)
        needed_params = []
        for n in N_CUSTOMERS_LIST:
            for d in DIMENSION_LIST:
                for grid_size in GRID_SIZE_LIST:
                    current_count = existing_counts[(n, d, grid_size)]
                    needed = max(0, N_PER_CONFIGURATION - current_count)
                    if needed > 0:
                        start_seq = current_count + 1
                        for i in range(needed):
                            seq_j = start_seq + i
                            dist_letters = list(np.random.choice(DIST_LETTERS, d))
                            if d > 1:
                                while len(set(dist_letters)) < min(2, d):
                                    dist_letters = list(np.random.choice(DIST_LETTERS, d))
                            seed = hash((n, d, grid_size, ''.join(dist_letters), seq_j)) % 2**32
                            needed_params.append((n, d, dist_letters, seed, seq_j, grid_size))
        
        if needed_params:
            print(f"\n2. Generating {len(needed_params)} new instances...")
            num_workers = max(1, os.cpu_count() - 2)
            batch_size = min(100, len(needed_params) // num_workers + 1)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                batches = [needed_params[i:i + batch_size] for i in range(0, len(needed_params), batch_size)]
                futures = [executor.submit(generate_batch_wrapper, b) for b in batches]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Generating"): f.result()
        
        print("\n3. Solving instances...")
        all_instance_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json')]
        instances_to_solve = []
        for f in all_instance_files:
            if f[:-5] not in existing_solutions:
                instances_to_solve.append(f[:-5])
        
        if instances_to_solve:
            print(f"   Solving {len(instances_to_solve)} instances...")
            instances_to_solve.sort(key=lambda x: int(x.split('_')[0][1:]))
            num_workers = max(1, min(6, os.cpu_count()))
            solve_batch_size = min(10, len(instances_to_solve) // num_workers + 1)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                batches = [instances_to_solve[i:i + solve_batch_size] for i in range(0, len(instances_to_solve), solve_batch_size)]
                futures = [executor.submit(solve_instance_batch, b) for b in batches]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Solving"): f.result()
        
        visualize_instances_and_solutions()
        print("\n=== COMPLETE ===")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    print("Warming up JIT...")
    test_coords = np.random.rand(10, 2).astype(np.float32)
    _ = compute_distance_matrix(test_coords, 100.0)
    _ = compute_tour_length_numba(test_coords, np.arange(1, 11), 100.0)
    _ = make_unique_numba(test_coords, 100.0, 42)
    main_optimized()