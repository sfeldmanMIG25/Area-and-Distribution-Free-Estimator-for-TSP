"""Main dataset generator: enumerates configurations, writes instances, solves
them with Concorde + LKH, then renders visualizations.
"""
import json
import multiprocessing as mp
import os
import re
import sys
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# Ensure repo root is importable when running as ``python data_pipeline/generator.py``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.instance_io import (
    DISTRIBUTION_MAP_1D,
    DIST_LETTERS,
    make_unique_numba,
    save_instance_binary,
    load_instance_binary,
)
from solvers import run_concorde, run_lkh
from solvers.config import get_scale_factor
from solvers.distance import compute_distance_matrix, compute_tour_length_numba

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
ROOT_DIR = str(_ROOT)
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")
VISUALS_DIR_INST = os.path.join(ROOT_DIR, "visuals", "instances")
VISUALS_DIR_SOL = os.path.join(ROOT_DIR, "visuals", "solutions")

os.makedirs(INSTANCES_DIR, exist_ok=True)
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR_INST, exist_ok=True)
os.makedirs(VISUALS_DIR_SOL, exist_ok=True)

GRID_SIZE_LIST = [100, 1000, 10000]
N_CUSTOMERS_LIST = [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                    200, 300, 400, 500, 600, 700, 800, 900, 1000]
DIMENSION_LIST = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
N_PER_CONFIGURATION = 82


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
                if letter == "k" and base is not None:
                    coords[:, j] = func(n, seed + j, grid_size, base=base)
                else:
                    coords[:, j] = func(n, seed + j, grid_size)
                if base is None and letter != "k":
                    base = coords[:, j]

            coords = make_unique_numba(coords, float(grid_size), seed)

            dist_string = "".join(dist_letters)
            instance_name = f"N{n}_D{d}_G{grid_size}_{dist_string}_{seq_j}"
            instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")

            instance_data = {
                "instance_name": instance_name,
                "n_customers": n, "dimension": d, "grid_size": grid_size,
                "distribution_types": dist_letters, "generation_seed": seed,
                "coordinates": coords.tolist(),
            }
            save_instance_binary(instance_path, instance_data)
            generated.append(instance_name)
        return generated


def solve_instance_batch(instance_names: List[str]) -> List[str]:
    solved = []
    for instance_name in instance_names:
        solution_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
        if os.path.exists(solution_path):
            solved.append(instance_name)
            continue

        instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
        binary_path = instance_path.replace(".json", ".bin")

        if os.path.exists(binary_path):
            inst_data = load_instance_binary(binary_path)
        elif os.path.exists(instance_path):
            with open(instance_path, "r") as f:
                inst_data = json.load(f)
        else:
            continue

        coords = np.array(inst_data["coordinates"], dtype=np.float64)
        d = inst_data["dimension"]
        grid_size = inst_data["grid_size"]
        scale_factor = get_scale_factor(float(grid_size))

        concorde_final = concorde_time = concorde_tour = concorde_error = None
        try:
            c_len, c_time, c_nodes = run_concorde(instance_name, coords, d, grid_size)
            concorde_final = c_len / scale_factor
            concorde_time = c_time
            concorde_tour = c_nodes
        except Exception as e:
            concorde_error = str(e)
            if "edge too long" not in str(e):
                print(f"Concorde error on {instance_name}: {e}")

        lkh_final = lkh_time = lkh_tour = None
        try:
            l_len, l_time, l_nodes = run_lkh(instance_name, coords, d, grid_size)
            lkh_final = l_len / scale_factor
            lkh_time = l_time
            lkh_tour = l_nodes
        except Exception as e:
            print(f"LKH Failed for {instance_name}: {e}")

        if lkh_final is None and concorde_final is None:
            continue

        if concorde_final is not None and (lkh_final is None or concorde_final < lkh_final):
            opt_cost, opt_tour, opt_solver = concorde_final, concorde_tour, "concorde"
        elif lkh_final is not None:
            opt_cost, opt_tour, opt_solver = lkh_final, lkh_tour, "lkh"
        else:
            opt_cost, opt_tour, opt_solver = -1, [], "failed"

        lkh_gap = None
        if concorde_final is not None and lkh_final is not None:
            lkh_gap = (lkh_final - concorde_final) / concorde_final * 100.0

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
            "lkh_gap_pct": lkh_gap,
        }

        with open(solution_path, "w") as f:
            json.dump(sol_data, f, indent=2)
        solved.append(instance_name)
    return solved


def generate_batch_wrapper(batch_params):
    return InstanceBatchGenerator().generate_batch(batch_params)


def plot_single_tour(tour, color, style, label, ax=None, d=2, coords=None):
    if not tour:
        return
    idx = np.array(tour) - 1
    t_coords = coords[idx]
    t_coords = np.vstack([t_coords, t_coords[0]])
    if d == 2:
        ax.plot(t_coords[:, 0], t_coords[:, 1], c=color, linestyle=style,
                label=label, alpha=0.8, linewidth=1.5)
    else:
        ax.plot(t_coords[:, 0], t_coords[:, 1], t_coords[:, 2], c=color,
                linestyle=style, label=label, alpha=0.8, linewidth=1.5)


def viz_worker(file_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not file_name.endswith(".json"):
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
        with open(instance_path, "r") as f:
            inst_data = json.load(f)

        d = inst_data["dimension"]
        if d not in [2, 3]:
            return 1

        coords = np.array(inst_data["coordinates"])

        if not os.path.exists(inst_vis_path):
            fig = plt.figure(figsize=(6, 6))
            if d == 2:
                ax = fig.add_subplot(111)
                ax.scatter(coords[:, 0], coords[:, 1], s=10, c="black")
            else:
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=10, c="black")
            plt.title(f"Instance: {instance_name}", fontsize=9)
            plt.tight_layout()
            plt.savefig(inst_vis_path, dpi=100)
            plt.close(fig)

        if has_sol and not os.path.exists(sol_vis_path):
            with open(sol_path, "r") as f:
                sol_data = json.load(f)

            fig = plt.figure(figsize=(8, 6))
            if d == 2:
                ax = fig.add_subplot(111)
                ax.scatter(coords[:, 0], coords[:, 1], s=15, c="k", zorder=5)
            else:
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=15, c="k", zorder=5)

            c_tour = sol_data.get("concorde_tour")
            c_len = sol_data.get("concorde_length", 0)
            if c_tour:
                plot_single_tour(c_tour, "blue", "-", f"Concorde: {c_len:.2f}", ax=ax, d=d, coords=coords)

            l_tour = sol_data.get("lkh_tour")
            l_len = sol_data.get("lkh_length", 0)
            if l_tour:
                plot_single_tour(l_tour, "red", "--", f"LKH: {l_len:.2f}", ax=ax, d=d, coords=coords)

            plt.title(f"Solutions: {instance_name}\nGap: {sol_data.get('lkh_gap_pct', 0):.4f}%", fontsize=10)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(sol_vis_path, dpi=100)
            plt.close(fig)
    except Exception:
        pass
    finally:
        plt.close("all")
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
        instance_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith(".json")]
        solution_files = [f for f in os.listdir(SOLUTIONS_DIR) if f.endswith(".sol.json")]
        filename_pattern = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)\.json$")

        for f in instance_files:
            match = filename_pattern.match(f)
            if match:
                n, d, g, _dist_str, _seq_id = match.groups()
                existing_counts[(int(n), int(d), int(g))] += 1

        for f in solution_files:
            existing_solutions.add(f[:-9])

        print(f"   Indexed {len(instance_files)} instances and {len(solution_files)} solutions.")

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
                            seed = hash((n, d, grid_size, "".join(dist_letters), seq_j)) % 2**32
                            needed_params.append((n, d, dist_letters, seed, seq_j, grid_size))

        if needed_params:
            print(f"\n2. Generating {len(needed_params)} new instances...")
            num_workers = max(1, os.cpu_count() - 2)
            batch_size = min(100, len(needed_params) // num_workers + 1)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                batches = [needed_params[i:i + batch_size] for i in range(0, len(needed_params), batch_size)]
                futures = [executor.submit(generate_batch_wrapper, b) for b in batches]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                    f.result()

        print("\n3. Solving instances...")
        all_instance_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith(".json")]
        instances_to_solve = [f[:-5] for f in all_instance_files if f[:-5] not in existing_solutions]

        if instances_to_solve:
            print(f"   Solving {len(instances_to_solve)} instances...")
            instances_to_solve.sort(key=lambda x: int(x.split("_")[0][1:]))
            num_workers = max(1, min(6, os.cpu_count()))
            solve_batch_size = min(10, len(instances_to_solve) // num_workers + 1)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                batches = [instances_to_solve[i:i + solve_batch_size] for i in range(0, len(instances_to_solve), solve_batch_size)]
                futures = [executor.submit(solve_instance_batch, b) for b in batches]
                for f in tqdm(as_completed(futures), total=len(futures), desc="Solving"):
                    f.result()

        visualize_instances_and_solutions()
        print("\n=== COMPLETE ===")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Warming up JIT...")
    test_coords = np.random.rand(10, 2).astype(np.float32)
    _ = compute_distance_matrix(test_coords, 100.0)
    _ = compute_tour_length_numba(test_coords, np.arange(1, 11), 100.0)
    _ = make_unique_numba(test_coords, 100.0, 42)
    main_optimized()
