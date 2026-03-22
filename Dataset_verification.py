import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import re
import sys
import shutil
import uuid
import subprocess
import time
from numba import njit, prange

# --- IMPORT GENERATOR FUNCTIONS ---
try:
    from Dataset_Generator import (
        compute_tour_length_numba, 
        compute_distance_matrix,
        _run_lkh_fast,
        load_instance_binary
    )
except ImportError:
    print("CRITICAL ERROR: 'Dataset_Generator.py' not found.")
    sys.exit(1)

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")
SOLVER_SCRATCH_DIR = "C:\\Temp_TSP_Scratch"

os.makedirs(SOLVER_SCRATCH_DIR, exist_ok=True)

# --- HELPER: RECONSTRUCT PARAMS ---
def reconstruct_params_from_filename(filename):
    pattern = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)\.json$")
    match = pattern.match(filename)
    if not match: return None
    n, d, g, dist_str, seq_id = match.groups()
    n, d, g, seq_id = int(n), int(d), int(g), int(seq_id)
    dist_letters = list(dist_str)
    seed = hash((n, d, g, dist_str, seq_id)) % 2**32
    return (n, d, dist_letters, seed, seq_id, g)

# --- HELPER: CLEANUP ---
def cleanup_artifacts(filename):
    base = filename[:-5] if filename.endswith('.json') else filename
    paths = [
        os.path.join(INSTANCES_DIR, f"{base}.json"),
        os.path.join(INSTANCES_DIR, f"{base}.bin"),
        os.path.join(SOLUTIONS_DIR, f"{base}.sol.json")
    ]
    for p in paths:
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

# --- NEW: ROBUST SCALING FORMULA ---
def get_robust_scale_factor(grid_size: float, n_customers: int) -> float:
    """
    Revised Scaling Formula:
    Relates Grid Size and N to scale down sparse instances naturally.
    """
    # 1. Determine Base Scale (Targeting ~10,000 virtual grid)
    if grid_size <= 100:
        base_scale = 100.0
    elif grid_size <= 1000:
        base_scale = 10.0
    else:
        base_scale = 1.0
        
    # 2. Apply N-based Dampening
    # If N is small (e.g. 9), we reduce the multiplier significantly.
    saturation_point = 500.0
    dampener = min(1.0, n_customers / saturation_point)
    
    final_scale = base_scale * dampener
    return  final_scale

# --- OVERRIDDEN: TSP SAVER WITH ROBUST SCALING ---
def _save_as_tsplib_robust(file_path: str, coords: np.ndarray, tsp_name: str, grid_size: int) -> float:
    """
    Saves TSP file using the N-dependent robust scaling.
    """
    n = len(coords)
    # Use the new formula
    scale_factor = get_robust_scale_factor(float(grid_size), n)
    
    # Compute matrix with minimum distance enforcement
    dist_matrix = compute_distance_matrix(coords, scale_factor)
    
    with open(file_path, "w") as f:
        f.write(f"NAME : {tsp_name}\nTYPE : TSP\nCOMMENT : Grid={grid_size} N={n} Scale={scale_factor:.2f}\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row_str = " ".join([str(x) for x in dist_matrix[i]])
            f.write(row_str + "\n")
        f.write("EOF\n")
        
    return scale_factor

# --- ROBUST CONCORDE SOLVER (NO TIMEOUT) ---
def _run_concorde_robust(coords: np.ndarray, grid_size: int) -> tuple:
    run_id = str(uuid.uuid4())[:12]
    run_dir = os.path.join(SOLVER_SCRATCH_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    tsp_filename = f"{run_id}.tsp"
    tour_filename = f"{run_id}.tour"
    tsp_path_win = os.path.join(run_dir, tsp_filename)
    
    try:
        # 1. Save with Robust Scaling
        scale = _save_as_tsplib_robust(tsp_path_win, coords, run_id, grid_size)
        
        # 2. WSL Path Conversion
        clean_tsp_path_win = tsp_path_win.replace('\\', '/')
        wsl_tsp_path = subprocess.run(["wsl", "wslpath", "-a", clean_tsp_path_win], 
                                      capture_output=True, text=True, check=True).stdout.strip()
        wsl_tour_file = wsl_tsp_path.replace('.tsp', '.tour')
        
        # 3. Run Concorde
        concorde_cmd = ["wsl", "concorde", "-o", wsl_tour_file, wsl_tsp_path]
        
        start_time = time.perf_counter()
        result = subprocess.run(concorde_cmd, capture_output=True, text=True, cwd=run_dir)
        runtime = time.perf_counter() - start_time
        
        if result.returncode != 0:
            raise RuntimeError(f"Concorde Crashed: {result.stderr}")
        
        # 4. Read Output
        tour_path_win = os.path.join(run_dir, tour_filename)
        if not os.path.exists(tour_path_win):
             raise FileNotFoundError(f"Concorde finished but produced no tour. Stderr: {result.stderr}")
             
        with open(tour_path_win, 'r') as f:
            tour_content = f.read()
        
        lines = tour_content.splitlines()
        if len(lines) < 2: 
            raise ValueError("Concorde output empty")
            
        tour_nodes = [int(n) + 1 for n in " ".join(lines[1:]).strip().split()]
        
        # 5. Compute Length (Using the exact same scale)
        tour_length = compute_tour_length_numba(coords, np.array(tour_nodes), scale)
        
        return tour_length, runtime, tour_nodes, scale

    finally:
        if os.path.exists(run_dir):
            try: shutil.rmtree(run_dir, ignore_errors=True)
            except: pass

# --- ROBUST WORKER ---
def solve_instance_robust(instance_name):
    instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    solution_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
    binary_path = instance_path.replace('.json', '.bin')

    if os.path.exists(binary_path):
        inst_data = load_instance_binary(binary_path)
    elif os.path.exists(instance_path):
        with open(instance_path, 'r') as f: inst_data = json.load(f)
    else:
        return "missing_file"

    coords = np.array(inst_data['coordinates'], dtype=np.float64)
    d = inst_data['dimension']
    grid_size = inst_data['grid_size']

    # Load existing to preserve LKH
    existing_sol = {}
    if os.path.exists(solution_path):
        try:
            with open(solution_path, 'r') as f: existing_sol = json.load(f)
        except: pass

    # --- CONCORDE ---
    concorde_final = existing_sol.get('concorde_length')
    concorde_tour = existing_sol.get('concorde_tour')
    concorde_time = existing_sol.get('concorde_time_s')
    
    if concorde_final is None:
        try:
            c_len, c_time, c_nodes, scale = _run_concorde_robust(coords, grid_size)
            concorde_final = c_len / scale
            concorde_time = c_time
            concorde_tour = c_nodes
        except Exception as e:
            return f"concorde_error: {str(e)}"

    # --- LKH ---
    lkh_final = existing_sol.get('lkh_length')
    lkh_tour = existing_sol.get('lkh_tour')
    lkh_time = existing_sol.get('lkh_time_s')
    
    if lkh_final is None:
        try:
            l_len, l_time, l_nodes = _run_lkh_fast(instance_name, coords, d, grid_size)
            # Use same robust scale logic for LKH reporting
            scale = get_robust_scale_factor(float(grid_size), len(coords))
            lkh_final = l_len / scale
            lkh_time = l_time
            lkh_tour = l_nodes
        except:
            pass

    # --- RESULT ---
    if concorde_final is not None and (lkh_final is None or concorde_final < lkh_final):
        opt_cost = concorde_final
        opt_tour = concorde_tour
        opt_solver = "concorde"
    elif lkh_final is not None:
        opt_cost = lkh_final
        opt_tour = lkh_tour
        opt_solver = "lkh"
    else:
        return "solver_failed"

    lkh_gap = None
    if concorde_final and lkh_final:
        lkh_gap = (lkh_final - concorde_final) / concorde_final * 100.0

    sol_data = {
        "instance_name": instance_name,
        "optimal_cost": opt_cost,
        "optimal_tour": opt_tour,
        "optimal_solver": opt_solver,
        "concorde_length": concorde_final,
        "concorde_time_s": concorde_time,
        "concorde_tour": concorde_tour,
        "lkh_length": lkh_final,
        "lkh_time_s": lkh_time,
        "lkh_tour": lkh_tour,
        "lkh_gap_pct": lkh_gap
    }

    with open(solution_path, 'w') as f:
        json.dump(sol_data, f, indent=2)
    
    return "solved"

# --- VERIFICATION ---
def verify_single_file(filename):
    res = {'filename': filename, 'needs_regen': False, 'needs_resolve': False, 'status': 'ok'}
    instance_path = os.path.join(INSTANCES_DIR, filename)
    solution_path = os.path.join(SOLUTIONS_DIR, filename.replace('.json', '.sol.json'))

    try:
        with open(instance_path, 'r') as f: inst_data = json.load(f)
        coords = np.array(inst_data['coordinates'])
        
        if len(coords) > 0:
            coords_view = coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).ravel()
            if len(np.unique(coords_view)) != len(coords):
                res['needs_regen'] = True
                res['status'] = 'stacked_coords'
                return res
        
        match = re.search(r"^N(\d+)_D(\d+)_G(\d+)_", filename)
        if match:
            fn_n, fn_d, fn_g = map(int, match.groups())
            if (fn_n != inst_data['n_customers'] or fn_d != inst_data['dimension'] or fn_g != inst_data['grid_size']):
                res['needs_regen'] = True
                res['status'] = 'meta_mismatch'
                return res
    except:
        res['needs_regen'] = True
        res['status'] = 'corrupt_inst'
        return res

    if not os.path.exists(solution_path):
        res['needs_resolve'] = True
        res['status'] = 'missing_sol'
        return res

    try:
        with open(solution_path, 'r') as f: sol_data = json.load(f)
        if sol_data.get('concorde_length') is None:
            res['needs_resolve'] = True
            res['status'] = 'concorde_missing'
    except:
        res['needs_resolve'] = True
        res['status'] = 'corrupt_sol'

    return res

# --- MAIN LOOP ---
def perform_repairs(files_to_regen, files_to_resolve):
    if files_to_regen:
        print(f"Regeneration needed for {len(files_to_regen)} files.")
        for f in files_to_regen:
            cleanup_artifacts(f)
            files_to_resolve.append(f)

    unique_to_solve = list(set([f[:-5] if f.endswith('.json') else f for f in files_to_resolve]))
    
    if unique_to_solve:
        print(f"\n>>> INITIATING ROBUST RE-SOLVE FOR {len(unique_to_solve)} INSTANCES <<<")
        print("    (Using N-dependent Scaling Formula)")
        
        unique_to_solve.sort(key=lambda x: int(x.split('_')[0][1:]))
        num_workers = max(1, os.cpu_count() - 2)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(solve_instance_robust, name): name for name in unique_to_solve}
            
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Robust Solving")
            for future in pbar:
                name = futures[future]
                try:
                    result = future.result()
                    if "error" in result:
                        pbar.write(f"FAILED {name}: {result}")
                except Exception as e:
                    pbar.write(f"CRITICAL EXCEPTION {name}: {e}")

def main():
    print("=== DATASET INTEGRITY CHECKER (ROBUST SCALING FIX) ===")
    all_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json')]
    
    print(f"Scanning {len(all_files)} instances...")
    stats = Counter()
    files_to_regen = []
    files_to_resolve = []
    
    with ProcessPoolExecutor() as executor:
        for res in tqdm(executor.map(verify_single_file, all_files), total=len(all_files), desc="Verifying"):
            stats[res['status']] += 1
            if res['needs_regen']: files_to_regen.append(res['filename'])
            elif res['needs_resolve']: files_to_resolve.append(res['filename'])

    print("\nSUMMARY:")
    print(f"Clean: {stats['ok']}")
    print(f"Issues: {len(files_to_regen) + len(files_to_resolve)}")
    print(dict(stats))
    
    if files_to_regen or files_to_resolve:
        perform_repairs(files_to_regen, files_to_resolve)
        print("\nRepairs attempted. Run verification again to confirm clean state.")
    else:
        print("\nDataset Perfect.")

if __name__ == "__main__":
    main()