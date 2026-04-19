"""Integrity checker + re-solver for the 2D benchmark dataset.

Mirrors data_pipeline/verification.py but targets the 2D grid produced by
d2_benchmark_gen.py and extended by extend_line_noise.py.

Steps:
1. Unzip any *.zip archives sitting in Generalized_TSP_Analysis/instances/ and
   Generalized_TSP_Analysis/solutions/, then delete the zips (cleaner.py-style).
2. Enumerate the full expected 2D grid (base distributions + clustered configs
   + line_noise extension) for grid_sizes {1000, 10000} x SAMPLES_PER_CONFIG.
3. Scan files on disk, identify missing / corrupt / rogue items.
4. Regenerate missing or corrupt instances (via base_gen.generate_and_save_instance
   patched with line_noise) and (re)solve them (via base_gen.solve_single_instance).
"""
import json
import os
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline import d2_benchmark_gen as base_gen
from data_pipeline.extend_line_noise import generate_line_noise

# Patch the line_noise distribution into the base generator's DIST_MAP so
# both regeneration and solving handle every config uniformly.
base_gen.DIST_MAP['line_noise'] = generate_line_noise

# d2_benchmark_gen.py anchors its paths to its own __file__ (data_pipeline/),
# which produces a phantom data_pipeline/Generalized_TSP_Analysis/ tree. The
# real dataset lives at the repo root. Repoint base_gen's globals so both this
# script and base_gen's generator/solver functions read & write the right dir.
ROOT_DIR      = _ROOT / "Generalized_TSP_Analysis"
INSTANCES_DIR = ROOT_DIR / "instances"
SOLUTIONS_DIR = ROOT_DIR / "solutions"
VISUALS_DIR   = ROOT_DIR / "visualizations"
for d in (ROOT_DIR, INSTANCES_DIR, SOLUTIONS_DIR, VISUALS_DIR):
    d.mkdir(exist_ok=True)
base_gen.ROOT_DIR      = ROOT_DIR
base_gen.INSTANCES_DIR = INSTANCES_DIR
base_gen.SOLUTIONS_DIR = SOLUTIONS_DIR
base_gen.VISUALS_DIR   = VISUALS_DIR

# Filename patterns produced by base_gen.generate_and_save_instance:
#   TSP-{dist}-n{N}-g{G}-{sample}                       (non-clustered)
#   TSP-{dist}-n{N}-g{G}-c{cn}-r{cr_pct}-{sample}       (clustered)
_RE_PLAIN = re.compile(
    r"^TSP-(?P<dist>[a-z_]+)-n(?P<n>\d+)-g(?P<g>\d+)-(?P<sample>\d+)$"
)
_RE_CLUST = re.compile(
    r"^TSP-(?P<dist>clustered)-n(?P<n>\d+)-g(?P<g>\d+)"
    r"-c(?P<cn>\d+)-r(?P<cr>\d+)-(?P<sample>\d+)$"
)


# ---------------------------------------------------------------------------
# Step 1: unzip + delete archives
# ---------------------------------------------------------------------------

def unzip_and_cleanup(target_dirs):
    for directory in target_dirs:
        if not directory.exists():
            continue
        zips = sorted(directory.glob("*.zip"))
        if not zips:
            continue
        print(f"\nUnzipping {len(zips)} archive(s) in {directory}...")
        for zpath in tqdm(zips, desc=f"Unzip {directory.name}"):
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(directory)
            except zipfile.BadZipFile as e:
                print(f"  WARN: bad zip {zpath.name}: {e}", file=sys.stderr)
                continue
            os.remove(zpath)


# ---------------------------------------------------------------------------
# Step 2: build expected grid
# ---------------------------------------------------------------------------

def build_expected_grid():
    """Return dict {instance_name: params_tuple} matching base_gen.main()
    plus the line_noise extension."""
    line_configs = [
        {'n_points': n, 'dist_type': 'line_noise'}
        for n in base_gen.n_points_list
    ]
    all_configs = list(base_gen.BASE_CONFIGS) + line_configs

    expected = {}
    seq_j = 1
    for grid_size in base_gen.GRID_SIZE_LIST:
        for config in all_configs:
            for i in range(1, base_gen.SAMPLES_PER_CONFIG + 1):
                n = config['n_points']
                dist_type = config['dist_type']
                config_num = sum(ord(c) for c in dist_type)
                base_seed = config_num + seq_j * 1000 + n * 100 + grid_size + i
                if dist_type == 'clustered':
                    name = (f"TSP-clustered-n{n}-g{grid_size}"
                            f"-c{config['clust_n']}-r{int(config['clust_rad'] * 100)}-{i}")
                else:
                    name = f"TSP-{dist_type}-n{n}-g{grid_size}-{i}"
                expected[name] = (config, grid_size, i, seq_j, base_seed)
                seq_j += 1
    return expected


def _parse(name):
    m = _RE_CLUST.match(name) or _RE_PLAIN.match(name)
    return m.groupdict() if m else None


# ---------------------------------------------------------------------------
# Step 3: classify on-disk state
# ---------------------------------------------------------------------------

def _instance_ok(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        coords = data.get("coordinates")
        n = data.get("n_customers")
        return isinstance(coords, list) and len(coords) == n and n is not None
    except Exception:
        return False


def _solution_ok(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("lkh_length") is not None
    except Exception:
        return False


def classify(expected):
    expected_names = set(expected)
    rogue_inst, rogue_sol = [], []
    corrupt_inst, missing_inst = [], []
    missing_sol, corrupt_sol = [], []

    seen_inst = set()
    for fname in os.listdir(INSTANCES_DIR):
        fpath = INSTANCES_DIR / fname
        if not fname.endswith(".json"):
            rogue_inst.append(fpath)
            continue
        stem = fname[:-5]
        if _parse(stem) is None or stem not in expected_names:
            rogue_inst.append(fpath)
            continue
        seen_inst.add(stem)
        if not _instance_ok(fpath):
            corrupt_inst.append(stem)

    missing_inst = [n for n in expected_names if n not in seen_inst]

    seen_sol = set()
    for fname in os.listdir(SOLUTIONS_DIR):
        fpath = SOLUTIONS_DIR / fname
        if not fname.endswith(".sol.json"):
            rogue_sol.append(fpath)
            continue
        stem = fname[:-9]
        if _parse(stem) is None or stem not in expected_names:
            rogue_sol.append(fpath)
            continue
        seen_sol.add(stem)
        if not _solution_ok(fpath):
            corrupt_sol.append(stem)

    needs_solve = sorted(
        set(missing_inst) | set(corrupt_inst) | set(corrupt_sol)
        | (expected_names - seen_sol)
    )
    needs_regen = sorted(set(missing_inst) | set(corrupt_inst))

    return {
        "rogue_inst": rogue_inst,
        "rogue_sol": rogue_sol,
        "missing_inst": missing_inst,
        "corrupt_inst": corrupt_inst,
        "corrupt_sol": corrupt_sol,
        "needs_regen": needs_regen,
        "needs_solve": needs_solve,
    }


# ---------------------------------------------------------------------------
# Step 4: repair
# ---------------------------------------------------------------------------

def repair(expected, report):
    # Drop rogue files
    for p in report["rogue_inst"] + report["rogue_sol"]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    # Drop corrupt files so the regenerator writes fresh content
    for stem in report["corrupt_inst"]:
        p = INSTANCES_DIR / f"{stem}.json"
        if p.exists():
            os.remove(p)
    for stem in report["corrupt_sol"]:
        p = SOLUTIONS_DIR / f"{stem}.sol.json"
        if p.exists():
            os.remove(p)

    num_workers = max(1, (os.cpu_count() or 2) - 2)

    if report["needs_regen"]:
        params_list = [expected[name] for name in report["needs_regen"]]
        print(f"\nRegenerating {len(params_list)} instances...")
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(base_gen.generate_and_save_instance, p): p for p in params_list}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Regen"):
                fut.result()

    if report["needs_solve"]:
        params_list = [expected[name] for name in report["needs_solve"]]
        print(f"\nSolving {len(params_list)} instances...")
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(base_gen.solve_single_instance, p): p for p in params_list}
            failed = []
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Solve"):
                try:
                    fut.result()
                except Exception as e:
                    failed.append(f"{type(e).__name__}: {e}")
            if failed:
                print(f"\n{len(failed)} solve failures (first 10):")
                for msg in failed[:10]:
                    print(f"  {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== 2D DATASET INTEGRITY CHECKER ===")

    unzip_and_cleanup([INSTANCES_DIR, SOLUTIONS_DIR])

    expected = build_expected_grid()
    print(f"Expected instances: {len(expected):,} "
          f"(grids={base_gen.GRID_SIZE_LIST}, samples={base_gen.SAMPLES_PER_CONFIG})")

    report = classify(expected)
    print("\nSUMMARY")
    print(f"  Missing instances:   {len(report['missing_inst']):>6,}")
    print(f"  Corrupt instances:   {len(report['corrupt_inst']):>6,}")
    print(f"  Corrupt solutions:   {len(report['corrupt_sol']):>6,}")
    print(f"  Needs regen total:   {len(report['needs_regen']):>6,}")
    print(f"  Needs solve total:   {len(report['needs_solve']):>6,}")
    print(f"  Rogue instance fls:  {len(report['rogue_inst']):>6,}")
    print(f"  Rogue solution fls:  {len(report['rogue_sol']):>6,}")

    if not (report["needs_regen"] or report["needs_solve"]
            or report["rogue_inst"] or report["rogue_sol"]):
        print("\n2D dataset is clean.")
        return

    repair(expected, report)
    print("\nRepairs complete. Re-run verification2D to confirm clean state.")


if __name__ == "__main__":
    main()
