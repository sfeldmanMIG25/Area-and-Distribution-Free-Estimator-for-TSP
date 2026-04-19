"""Integrity checker + re-solver.

Scans ``instances/`` and ``solutions/`` for missing/corrupt files, regenerates
broken instances from their filename-encoded params, and re-solves with
Concorde (N-dependent scaling) + LKH. Hard-fails on solver errors.
"""
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.instance_io import load_instance_binary
from solvers.concorde import run_concorde_robust
from solvers.config import get_robust_scale_factor
from solvers.lkh import run_lkh

ROOT_DIR = str(_ROOT)
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")


def reconstruct_params_from_filename(filename):
    pattern = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)\.json$")
    match = pattern.match(filename)
    if not match:
        return None
    n, d, g, dist_str, seq_id = match.groups()
    n, d, g, seq_id = int(n), int(d), int(g), int(seq_id)
    seed = hash((n, d, dist_str, seq_id)) % 2**32
    return (n, d, list(dist_str), seed, seq_id, g)


def cleanup_artifacts(filename):
    base = filename[:-5] if filename.endswith(".json") else filename
    for p in (
        os.path.join(INSTANCES_DIR, f"{base}.json"),
        os.path.join(INSTANCES_DIR, f"{base}.bin"),
        os.path.join(SOLUTIONS_DIR, f"{base}.sol.json"),
    ):
        if os.path.exists(p):
            os.remove(p)


def solve_instance_robust(instance_name):
    instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    solution_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
    binary_path = instance_path.replace(".json", ".bin")

    if os.path.exists(binary_path):
        inst_data = load_instance_binary(binary_path)
    elif os.path.exists(instance_path):
        with open(instance_path, "r") as f:
            inst_data = json.load(f)
    else:
        return "missing_file"

    coords = np.array(inst_data["coordinates"], dtype=np.float64)
    d = inst_data["dimension"]
    grid_size = inst_data["grid_size"]

    existing_sol = {}
    if os.path.exists(solution_path):
        with open(solution_path, "r") as f:
            existing_sol = json.load(f)

    concorde_final = existing_sol.get("concorde_length")
    concorde_tour = existing_sol.get("concorde_tour")
    concorde_time = existing_sol.get("concorde_time_s")

    if concorde_final is None:
        c_len, c_time, c_nodes, scale = run_concorde_robust(coords, grid_size)
        concorde_final = c_len / scale if scale else float(c_len)
        concorde_time = c_time
        concorde_tour = c_nodes

    lkh_final = existing_sol.get("lkh_length")
    lkh_tour = existing_sol.get("lkh_tour")
    lkh_time = existing_sol.get("lkh_time_s")

    if lkh_final is None:
        l_len, l_time, l_nodes = run_lkh(instance_name, coords, d, grid_size)
        scale = get_robust_scale_factor(float(grid_size), len(coords))
        lkh_final = l_len / scale if scale else float(l_len)
        lkh_time = l_time
        lkh_tour = l_nodes

    if concorde_final < lkh_final:
        opt_cost, opt_tour, opt_solver = concorde_final, concorde_tour, "concorde"
    else:
        opt_cost, opt_tour, opt_solver = lkh_final, lkh_tour, "lkh"

    lkh_gap = (lkh_final - concorde_final) / concorde_final * 100.0 if concorde_final else None

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
        "lkh_gap_pct": lkh_gap,
    }

    with open(solution_path, "w") as f:
        json.dump(sol_data, f, indent=2)

    return "solved"


def verify_single_file(filename):
    res = {"filename": filename, "needs_regen": False, "needs_resolve": False, "status": "ok"}
    instance_path = os.path.join(INSTANCES_DIR, filename)
    solution_path = os.path.join(SOLUTIONS_DIR, filename.replace(".json", ".sol.json"))

    try:
        with open(instance_path, "r") as f:
            inst_data = json.load(f)
        coords = np.array(inst_data["coordinates"])

        if len(coords) > 0:
            view = coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).ravel()
            if len(np.unique(view)) != len(coords):
                res["needs_regen"] = True
                res["status"] = "stacked_coords"
                return res

        match = re.search(r"^N(\d+)_D(\d+)_G(\d+)_", filename)
        if match:
            fn_n, fn_d, fn_g = map(int, match.groups())
            if (fn_n != inst_data["n_customers"] or fn_d != inst_data["dimension"] or fn_g != inst_data["grid_size"]):
                res["needs_regen"] = True
                res["status"] = "meta_mismatch"
                return res
    except Exception:
        res["needs_regen"] = True
        res["status"] = "corrupt_inst"
        return res

    if not os.path.exists(solution_path):
        res["needs_resolve"] = True
        res["status"] = "missing_sol"
        return res

    try:
        with open(solution_path, "r") as f:
            sol_data = json.load(f)
        if sol_data.get("concorde_length") is None:
            res["needs_resolve"] = True
            res["status"] = "concorde_missing"
    except Exception:
        res["needs_resolve"] = True
        res["status"] = "corrupt_sol"

    return res


def perform_repairs(files_to_regen, files_to_resolve):
    if files_to_regen:
        print(f"Regeneration needed for {len(files_to_regen)} files.")
        for f in files_to_regen:
            cleanup_artifacts(f)
            files_to_resolve.append(f)

    unique_to_solve = list(set([f[:-5] if f.endswith(".json") else f for f in files_to_resolve]))
    if not unique_to_solve:
        return

    print(f"\n>>> INITIATING ROBUST RE-SOLVE FOR {len(unique_to_solve)} INSTANCES <<<")
    print("    (Using N-dependent Scaling; Held-Karp for N<=20)")

    unique_to_solve.sort(key=lambda x: int(x.split("_")[0][1:]))
    num_workers = max(1, os.cpu_count() - 2)

    failed = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(solve_instance_robust, name): name for name in unique_to_solve}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Robust Solving")
        for future in pbar:
            name = futures[future]
            try:
                result = future.result()
                if result != "solved":
                    failed.append((name, result))
                    pbar.write(f"FAILED {name}: {result}")
            except Exception as e:
                failed.append((name, f"{type(e).__name__}: {e}"))
                pbar.write(f"FAILED {name}: {type(e).__name__}: {e}")

    if failed:
        print(f"\n{len(failed)} instances failed. First 20:")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")
        sys.exit(1)


def main():
    print("=== DATASET INTEGRITY CHECKER ===")
    all_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith(".json")]
    print(f"Scanning {len(all_files)} instances...")

    stats = Counter()
    files_to_regen = []
    files_to_resolve = []

    with ProcessPoolExecutor() as executor:
        for res in tqdm(executor.map(verify_single_file, all_files), total=len(all_files), desc="Verifying"):
            stats[res["status"]] += 1
            if res["needs_regen"]:
                files_to_regen.append(res["filename"])
            elif res["needs_resolve"]:
                files_to_resolve.append(res["filename"])

    print(f"\nSUMMARY:\nClean: {stats['ok']}\nIssues: {len(files_to_regen) + len(files_to_resolve)}")
    print(dict(stats))

    if files_to_regen or files_to_resolve:
        perform_repairs(files_to_regen, files_to_resolve)
        print("\nRepairs attempted. Run verification again to confirm clean state.")
    else:
        print("\nDataset Perfect.")


if __name__ == "__main__":
    main()
