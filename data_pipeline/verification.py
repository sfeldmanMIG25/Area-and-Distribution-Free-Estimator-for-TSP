"""Integrity checker + re-solver.

Grid-first verification: enumerates every expected (N, D, G, seq_j) combination,
finds what is missing or corrupt, regenerates and re-solves those, then deletes
any instance or solution file that does not belong to the expected grid.
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
    DIST_LETTERS,
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

# Must match generator.py exactly.
GRID_SIZE_LIST      = [100, 1000, 10000]
N_CUSTOMERS_LIST    = [5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                       200, 300, 400, 500, 600, 700, 800, 900, 1000]
DIMENSION_LIST      = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
N_PER_CONFIGURATION = 82

_GRID_CELLS = frozenset(
    (n, d, g)
    for n in N_CUSTOMERS_LIST
    for d in DIMENSION_LIST
    for g in GRID_SIZE_LIST
)
_TOTAL_EXPECTED = len(_GRID_CELLS) * N_PER_CONFIGURATION

_FILENAME_RE = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)$")


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def shutdown_wsl():
    try:
        subprocess.run(["wsl", "--shutdown"], timeout=15, capture_output=True)
    except Exception as e:
        print(f"wsl --shutdown failed: {e}", file=sys.stderr)
    try:
        subprocess.run(["taskkill", "/F", "/IM", "wsl.exe"], timeout=10, capture_output=True)
    except Exception:
        pass


def _install_sigint_handler():
    def _handler(signum, frame):
        print("\n[SIGINT] Shutting down WSL and aborting...", file=sys.stderr)
        shutdown_wsl()
        os._exit(130)
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)


# ---------------------------------------------------------------------------
# Instance I/O helpers
# ---------------------------------------------------------------------------

def _parse_instance_name(name):
    """Return (n, d, g, dist_str, seq_j) or None if name does not match."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    n, d, g, dist_str, seq_j = m.groups()
    return int(n), int(d), int(g), dist_str, int(seq_j)


def _load_instance(instance_name):
    """Strict load: prefer .bin, fall back to .json. Validates shape."""
    parsed = _parse_instance_name(instance_name)
    if parsed is None:
        raise ValueError(f"bad filename: {instance_name}")
    fn_n, fn_d, fn_g, _, _ = parsed

    binary_path = os.path.join(INSTANCES_DIR, f"{instance_name}.bin")
    json_path   = os.path.join(INSTANCES_DIR, f"{instance_name}.json")

    if os.path.exists(binary_path):
        inst_data = load_instance_binary(binary_path)
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            inst_data = json.load(f)

    coords = np.array(inst_data["coordinates"], dtype=np.float64)
    if coords.shape != (fn_n, fn_d):
        raise ValueError(f"shape {coords.shape} != ({fn_n},{fn_d})")
    if inst_data.get("grid_size") != fn_g:
        raise ValueError(f"grid_size {inst_data.get('grid_size')} != {fn_g}")
    return inst_data


def cleanup_artifacts(instance_name):
    for p in (
        os.path.join(INSTANCES_DIR, f"{instance_name}.json"),
        os.path.join(INSTANCES_DIR, f"{instance_name}.bin"),
        os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json"),
    ):
        if os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _coords_from_params(n, d, dist_letters, seed, grid_size):
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
    return make_unique_numba(coords, float(grid_size), seed)


def regenerate_instance(instance_name):
    """Regenerate a corrupt instance from its filename-encoded params."""
    parsed = _parse_instance_name(instance_name)
    if parsed is None:
        return False
    n, d, g, dist_str, seq_j = parsed
    dist_letters = list(dist_str)
    seed = hash((n, d, g, dist_str, seq_j)) % 2**32
    coords = _coords_from_params(n, d, dist_letters, seed, g)
    instance_data = {
        "instance_name": instance_name,
        "n_customers": n, "dimension": d, "grid_size": g,
        "distribution_types": dist_letters, "generation_seed": seed,
        "coordinates": coords.tolist(),
    }
    save_instance_binary(os.path.join(INSTANCES_DIR, f"{instance_name}.json"), instance_data)
    return True


def generate_new_instance(n, d, g, seq_j):
    """Generate a brand-new instance for a grid slot that has no file at all."""
    rng = np.random.default_rng(hash((n, d, g, seq_j, "new")) % 2**32)
    dist_letters = list(rng.choice(list(DIST_LETTERS), d))
    if d > 1:
        while len(set(dist_letters)) < min(2, d):
            dist_letters = list(rng.choice(list(DIST_LETTERS), d))
    dist_str = "".join(dist_letters)
    seed = hash((n, d, g, dist_str, seq_j)) % 2**32
    instance_name = f"N{n}_D{d}_G{g}_{dist_str}_{seq_j}"
    coords = _coords_from_params(n, d, dist_letters, seed, g)
    instance_data = {
        "instance_name": instance_name,
        "n_customers": n, "dimension": d, "grid_size": g,
        "distribution_types": dist_letters, "generation_seed": seed,
        "coordinates": coords.tolist(),
    }
    save_instance_binary(os.path.join(INSTANCES_DIR, f"{instance_name}.json"), instance_data)
    return instance_name


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def solve_instance_robust(instance_name):
    inst_data = _load_instance(instance_name)
    coords    = np.array(inst_data["coordinates"], dtype=np.float64)
    d         = inst_data["dimension"]
    grid_size = inst_data["grid_size"]

    solution_path = os.path.join(SOLUTIONS_DIR, f"{instance_name}.sol.json")
    existing_sol  = {}
    if os.path.exists(solution_path):
        try:
            with open(solution_path, "r", encoding="utf-8") as f:
                existing_sol = json.load(f)
        except Exception:
            existing_sol = {}

    concorde_final = existing_sol.get("concorde_length")
    concorde_tour  = existing_sol.get("concorde_tour")
    concorde_time  = existing_sol.get("concorde_time_s")
    concorde_error = existing_sol.get("concorde_error")

    if concorde_final is None:
        try:
            c_len, c_time, c_nodes, scale = run_concorde_robust(coords, grid_size)
            concorde_final = c_len / scale if scale else float(c_len)
            concorde_time  = c_time
            concorde_tour  = c_nodes
            concorde_error = None
        except Exception as e:
            concorde_error = f"{type(e).__name__}: {e}"

    lkh_final = existing_sol.get("lkh_length")
    lkh_tour  = existing_sol.get("lkh_tour")
    lkh_time  = existing_sol.get("lkh_time_s")

    if lkh_final is None:
        l_len, l_time, l_nodes = run_lkh(instance_name, coords, d, grid_size)
        scale     = get_robust_scale_factor(float(grid_size), len(coords))
        lkh_final = l_len / scale if scale else float(l_len)
        lkh_time  = l_time
        lkh_tour  = l_nodes

    if concorde_final is not None and concorde_final < lkh_final:
        opt_cost, opt_tour, opt_solver = concorde_final, concorde_tour, "concorde"
    else:
        opt_cost, opt_tour, opt_solver = lkh_final, lkh_tour, "lkh"

    lkh_gap = (lkh_final - concorde_final) / concorde_final * 100.0 if concorde_final else None

    sol_data = {
        "instance_name":  instance_name,
        "optimal_cost":   opt_cost,
        "optimal_tour":   opt_tour,
        "optimal_solver": opt_solver,
        "concorde_length": concorde_final,
        "concorde_time_s": concorde_time,
        "concorde_tour":   concorde_tour,
        "concorde_error":  concorde_error,
        "lkh_length":  lkh_final,
        "lkh_time_s":  lkh_time,
        "lkh_tour":    lkh_tour,
        "lkh_gap_pct": lkh_gap,
    }
    with open(solution_path, "w", encoding="utf-8") as f:
        json.dump(sol_data, f, indent=2)
    return "solved"


# ---------------------------------------------------------------------------
# Repair orchestration
# ---------------------------------------------------------------------------

def _regen_worker(instance_name):
    try:
        cleanup_artifacts(instance_name)
        if regenerate_instance(instance_name):
            return instance_name, "ok"
        return instance_name, "unreconstructable"
    except Exception as e:
        return instance_name, f"{type(e).__name__}: {e}"


def _new_gen_worker(args):
    n, d, g, seq_j = args
    try:
        name = generate_new_instance(n, d, g, seq_j)
        return name, "ok"
    except Exception as e:
        return f"N{n}_D{d}_G{g}_?_{seq_j}", f"{type(e).__name__}: {e}"


def perform_repairs(files_to_regen, missing_slots, files_to_resolve):
    num_workers = max(1, os.cpu_count())

    if files_to_regen:
        print(f"\nREGENERATING {len(files_to_regen)} corrupt instances")
        regen_failed = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_regen_worker, name): name for name in files_to_regen}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Regenerating"):
                name, status = future.result()
                if status != "ok":
                    regen_failed.append((name, status))
                else:
                    files_to_resolve.append(name)
        if regen_failed:
            print(f"{len(regen_failed)} regeneration failures:")
            for name, err in regen_failed[:20]:
                print(f"  {name}: {err}")
            sys.exit(1)

    if missing_slots:
        print(f"\nGENERATING {len(missing_slots)} missing grid instances")
        gen_failed = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_new_gen_worker, slot): slot for slot in missing_slots}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                name, status = future.result()
                if status != "ok":
                    gen_failed.append((name, status))
                else:
                    files_to_resolve.append(name)
        if gen_failed:
            print(f"{len(gen_failed)} generation failures:")
            for name, err in gen_failed[:20]:
                print(f"  {name}: {err}")
            sys.exit(1)

    unique_to_solve = list(set(files_to_resolve))
    if not unique_to_solve:
        return

    print(f"\nSOLVING {len(unique_to_solve)} instances")
    unique_to_solve.sort(key=lambda x: int(x.split("_")[0][1:]))
    failed = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(solve_instance_robust, name): name for name in unique_to_solve}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Solving")
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
        print(f"\n{len(failed)} solve failures (first 20):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main: grid-first verification
# ---------------------------------------------------------------------------

def main():
    _install_sigint_handler()
    print("=== DATASET INTEGRITY CHECKER (grid-first) ===")
    print(f"Expected grid: {len(N_CUSTOMERS_LIST)} N x {len(DIMENSION_LIST)} D x "
          f"{len(GRID_SIZE_LIST)} G x {N_PER_CONFIGURATION} instances = {_TOTAL_EXPECTED:,} total")

    # --- 1. Build expected set ---
    expected_cells = {
        (n, d, g): set(range(1, N_PER_CONFIGURATION + 1))
        for n in N_CUSTOMERS_LIST
        for d in DIMENSION_LIST
        for g in GRID_SIZE_LIST
    }

    # --- 2. Scan instances dir ---
    print("\nScanning instances/...")
    found = {}          # (n, d, g, seq_j) -> instance_name
    rogue_inst = []     # filenames to delete (not in grid)

    seen_names = set()
    for fname in os.listdir(INSTANCES_DIR):
        if fname.endswith(".bin"):
            stem = fname[:-4]
        elif fname.endswith(".json"):
            stem = fname[:-5]
        else:
            rogue_inst.append(os.path.join(INSTANCES_DIR, fname))
            continue

        if stem in seen_names:
            continue
        seen_names.add(stem)

        parsed = _parse_instance_name(stem)
        if parsed is None:
            rogue_inst.append(os.path.join(INSTANCES_DIR, fname))
            continue

        n, d, g, dist_str, seq_j = parsed
        if (n, d, g) not in expected_cells or seq_j not in expected_cells[(n, d, g)]:
            # Delete both .json and .bin for out-of-grid files
            for ext in (".json", ".bin"):
                p = os.path.join(INSTANCES_DIR, f"{stem}{ext}")
                if os.path.exists(p):
                    rogue_inst.append(p)
            continue

        found[(n, d, g, seq_j)] = stem

    # --- 3. Scan solutions dir ---
    print("Scanning solutions/...")
    rogue_sol = []      # solution files to delete

    for fname in os.listdir(SOLUTIONS_DIR):
        if not fname.endswith(".sol.json"):
            rogue_sol.append(os.path.join(SOLUTIONS_DIR, fname))
            continue
        stem   = fname[:-9]
        parsed = _parse_instance_name(stem)
        if parsed is None:
            rogue_sol.append(os.path.join(SOLUTIONS_DIR, fname))
            continue
        n, d, g, _, seq_j = parsed
        if (n, d, g) not in expected_cells or seq_j not in expected_cells[(n, d, g)]:
            rogue_sol.append(os.path.join(SOLUTIONS_DIR, fname))
            continue
        # Solution without a matching instance file
        if (n, d, g, seq_j) not in found:
            rogue_sol.append(os.path.join(SOLUTIONS_DIR, fname))

    # --- 4. Classify found instances ---
    files_to_regen  = []   # instance names: corrupt content
    files_to_resolve = []  # instance names: missing/incomplete solution

    for (n, d, g, seq_j), inst_name in found.items():
        try:
            _load_instance(inst_name)
        except Exception:
            files_to_regen.append(inst_name)
            continue

        sol_path = os.path.join(SOLUTIONS_DIR, f"{inst_name}.sol.json")
        if not os.path.exists(sol_path):
            files_to_resolve.append(inst_name)
            continue
        try:
            with open(sol_path, "r", encoding="utf-8") as f:
                sol = json.load(f)
            if sol.get("concorde_length") is None:
                files_to_resolve.append(inst_name)
        except Exception:
            files_to_resolve.append(inst_name)

    # --- 5. Find completely missing grid slots ---
    missing_slots = [
        (n, d, g, seq_j)
        for (n, d, g), seqs in expected_cells.items()
        for seq_j in seqs
        if (n, d, g, seq_j) not in found
    ]

    # --- 6. Report ---
    n_issues = (len(rogue_inst) + len(rogue_sol) + len(files_to_regen)
                + len(missing_slots) + len(files_to_resolve))
    print(f"\nSUMMARY")
    print(f"  Found instances:       {len(found):>8,}")
    print(f"  Missing grid slots:    {len(missing_slots):>8,}")
    print(f"  Corrupt instances:     {len(files_to_regen):>8,}")
    print(f"  Missing solutions:     {len(files_to_resolve):>8,}")
    print(f"  Rogue instance files:  {len(rogue_inst):>8,}")
    print(f"  Rogue solution files:  {len(rogue_sol):>8,}")
    print(f"  Total issues:          {n_issues:>8,}")

    if n_issues == 0:
        print("\nDataset is clean.")
        return

    # --- 7. Delete rogue files ---
    if rogue_inst or rogue_sol:
        print(f"\nDeleting {len(rogue_inst)} rogue instance files "
              f"and {len(rogue_sol)} rogue solution files...")
        for p in rogue_inst + rogue_sol:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    # --- 8. Regenerate corrupt, generate missing, solve unsolved ---
    perform_repairs(files_to_regen, missing_slots, files_to_resolve)
    print("\nRepairs complete. Run verification again to confirm clean state.")


if __name__ == "__main__":
    main()
