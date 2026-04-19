"""Integrity checker + re-solver.

Scans ``instances/`` and ``solutions/`` for missing/corrupt files. Corrupt
instances are regenerated deterministically from filename-encoded params using
the same seed formula as ``data_pipeline.generator``. Then re-solves with
Concorde (N-dependent scaling) + LKH. Hard-fails on solver errors.
"""
import json
import os
import re
import signal
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.instance_io import (
    DISTRIBUTION_MAP_1D,
    load_instance_binary,
    make_unique_numba,
    save_instance_binary,
)
from solvers.concorde import run_concorde_robust
from solvers.config import get_robust_scale_factor
from solvers.lkh import run_lkh

ROOT_DIR = str(_ROOT)
INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")

_FILENAME_RE = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)\.json$")


def shutdown_wsl():
    """Hard-kill WSL VM and any lingering wsl.exe processes."""
    try:
        subprocess.run(["wsl", "--shutdown"], timeout=15, capture_output=True)
    except Exception as e:
        print(f"wsl --shutdown failed: {e}", file=sys.stderr)
    try:
        subprocess.run(["taskkill", "/F", "/IM", "wsl.exe"], timeout=10, capture_output=True)
    except Exception:
        pass


def _install_sigint_handler():
    """Ctrl+C -> shut down WSL and exit hard. Prevents zombie concorde runs
    eating the WSL VM after the Python process dies."""
    def _handler(signum, frame):
        print("\n[SIGINT] Shutting down WSL and aborting...", file=sys.stderr)
        shutdown_wsl()
        os._exit(130)
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)


def reconstruct_params_from_filename(filename):
    """Return (n, d, dist_letters, seed, seq_j, grid_size) matching
    the seed formula in data_pipeline.generator (includes grid_size)."""
    match = _FILENAME_RE.match(filename)
    if not match:
        return None
    n, d, g, dist_str, seq_id = match.groups()
    n, d, g, seq_id = int(n), int(d), int(g), int(seq_id)
    dist_letters = list(dist_str)
    seed = hash((n, d, g, dist_str, seq_id)) % 2**32
    return (n, d, dist_letters, seed, seq_id, g)


def regenerate_instance(instance_name):
    """Regenerate a single instance from its filename-encoded params.
    Writes both .json and .bin. Returns True on success."""
    params = reconstruct_params_from_filename(f"{instance_name}.json")
    if params is None:
        return False
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

    instance_data = {
        "instance_name": instance_name,
        "n_customers": n,
        "dimension": d,
        "grid_size": grid_size,
        "distribution_types": dist_letters,
        "generation_seed": seed,
        "coordinates": coords.tolist(),
    }
    instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    save_instance_binary(instance_path, instance_data)
    return True


def cleanup_artifacts(filename):
    base = filename[:-5] if filename.endswith(".json") else filename
    for p in (
        os.path.join(INSTANCES_DIR, f"{base}.json"),
        os.path.join(INSTANCES_DIR, f"{base}.bin"),
        os.path.join(SOLUTIONS_DIR, f"{base}.sol.json"),
    ):
        if os.path.exists(p):
            os.remove(p)


def _load_instance(instance_name):
    """Strict load: prefer .bin, fall back to .json with utf-8. Validates
    coordinates shape against filename. Returns inst_data or raises."""
    match = _FILENAME_RE.match(f"{instance_name}.json")
    if not match:
        raise ValueError(f"bad filename: {instance_name}")
    fn_n, fn_d, fn_g = int(match.group(1)), int(match.group(2)), int(match.group(3))

    instance_path = os.path.join(INSTANCES_DIR, f"{instance_name}.json")
    binary_path = instance_path.replace(".json", ".bin")

    if os.path.exists(binary_path):
        inst_data = load_instance_binary(binary_path)
    else:
        with open(instance_path, "r", encoding="utf-8") as f:
            inst_data = json.load(f)

    coords = np.array(inst_data["coordinates"], dtype=np.float64)
    if coords.shape != (fn_n, fn_d):
        raise ValueError(f"shape {coords.shape} != ({fn_n},{fn_d})")
    if inst_data.get("grid_size") != fn_g:
        raise ValueError(f"grid_size {inst_data.get('grid_size')} != {fn_g}")
    return inst_data


def solve_instance_robust(instance_name):
    inst_data = _load_instance(instance_name)
    coords = np.array(inst_data["coordinates"], dtype=np.float64)
    d = inst_data["dimension"]
    grid_size = inst_data["grid_size"]

    solution_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
    existing_sol = {}
    if os.path.exists(solution_path):
        try:
            with open(solution_path, "r", encoding="utf-8") as f:
                existing_sol = json.load(f)
        except Exception:
            existing_sol = {}

    concorde_final = existing_sol.get("concorde_length")
    concorde_tour = existing_sol.get("concorde_tour")
    concorde_time = existing_sol.get("concorde_time_s")
    concorde_error = existing_sol.get("concorde_error")

    if concorde_final is None:
        try:
            c_len, c_time, c_nodes, scale = run_concorde_robust(coords, grid_size)
            concorde_final = c_len / scale if scale else float(c_len)
            concorde_time = c_time
            concorde_tour = c_nodes
            concorde_error = None
        except Exception as e:
            concorde_error = f"{type(e).__name__}: {e}"

    lkh_final = existing_sol.get("lkh_length")
    lkh_tour = existing_sol.get("lkh_tour")
    lkh_time = existing_sol.get("lkh_time_s")

    if lkh_final is None:
        l_len, l_time, l_nodes = run_lkh(instance_name, coords, d, grid_size)
        scale = get_robust_scale_factor(float(grid_size), len(coords))
        lkh_final = l_len / scale if scale else float(l_len)
        lkh_time = l_time
        lkh_tour = l_nodes

    if concorde_final is not None and concorde_final < lkh_final:
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
        "concorde_error": concorde_error,
        "lkh_length": lkh_final,
        "lkh_time_s": lkh_time,
        "lkh_tour": lkh_tour,
        "lkh_gap_pct": lkh_gap,
    }

    with open(solution_path, "w", encoding="utf-8") as f:
        json.dump(sol_data, f, indent=2)

    return "solved"


def _regen_worker(instance_name):
    try:
        cleanup_artifacts(instance_name)
        if regenerate_instance(instance_name):
            return (instance_name, "ok")
        return (instance_name, "unreconstructable")
    except Exception as e:
        return (instance_name, f"{type(e).__name__}: {e}")


def verify_single_file(filename):
    res = {"filename": filename, "needs_regen": False, "needs_resolve": False, "status": "ok"}
    instance_name = filename[:-5] if filename.endswith(".json") else filename

    try:
        inst_data = _load_instance(instance_name)
        coords = np.asarray(inst_data["coordinates"])
        if len(coords) > 0:
            view = coords.view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).ravel()
            if len(np.unique(view)) != len(coords):
                res["needs_regen"] = True
                res["status"] = "stacked_coords"
                return res
    except Exception as e:
        res["needs_regen"] = True
        res["status"] = f"corrupt_inst:{type(e).__name__}"
        return res

    solution_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
    if not os.path.exists(solution_path):
        res["needs_resolve"] = True
        res["status"] = "missing_sol"
        return res

    try:
        with open(solution_path, "r", encoding="utf-8") as f:
            sol_data = json.load(f)
        if sol_data.get("concorde_length") is None:
            res["needs_resolve"] = True
            res["status"] = "concorde_missing"
    except Exception:
        res["needs_resolve"] = True
        res["status"] = "corrupt_sol"

    return res


def perform_repairs(files_to_regen, files_to_resolve):
    num_workers = max(1, os.cpu_count() - 2)

    if files_to_regen:
        print(f"\n>>> REGENERATING {len(files_to_regen)} CORRUPT INSTANCES <<<")
        names = [f[:-5] if f.endswith(".json") else f for f in files_to_regen]
        regen_failed = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_regen_worker, name): name for name in names}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Regenerating")
            for future in pbar:
                name, status = future.result()
                if status != "ok":
                    regen_failed.append((name, status))
                    pbar.write(f"REGEN FAILED {name}: {status}")
                else:
                    files_to_resolve.append(name)
        if regen_failed:
            print(f"\n{len(regen_failed)} regeneration failures (first 20):")
            for name, err in regen_failed[:20]:
                print(f"  {name}: {err}")
            sys.exit(1)

    unique_to_solve = list({f[:-5] if f.endswith(".json") else f for f in files_to_resolve})
    if not unique_to_solve:
        return

    print(f"\n>>> INITIATING ROBUST RE-SOLVE FOR {len(unique_to_solve)} INSTANCES <<<")
    print("    (Using N-dependent Scaling; Held-Karp for N<=20)")

    unique_to_solve.sort(key=lambda x: int(x.split("_")[0][1:]))

    failed = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
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
    _install_sigint_handler()
    print("=== DATASET INTEGRITY CHECKER ===")
    print("    (Ctrl+C aborts cleanly and shuts down WSL)")
    all_files = [f for f in os.listdir(INSTANCES_DIR) if f.endswith(".json")]
    print(f"Scanning {len(all_files)} instances...")

    stats = Counter()
    files_to_regen = []
    files_to_resolve = []

    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as executor:
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
