"""
Run LKH-3 on TSPLIB95 EUC_2D instances with n > 1000.

- Concorde is impractical at this scale; LKH-3 only.
- No per-instance timeout (LKH-3 converges naturally).
- ThreadPoolExecutor(max_workers=2) — each worker has its own scratch dir.
- Idempotent: skips instances already present in solver_wall_times_large.csv.
- Includes pla7397 as a scaling probe (MAX_TRIALS=1000 instead of 10000).

Output: tsplib_benchmark/results/solver_wall_times_large.csv
Columns: instance, n, lkh_time_s, lkh_tour_len, matches_optimum, published_optimum
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTANCE_DIR = REPO / "tsplib_benchmark" / "instances"
OPTIMA_CSV = REPO / "tsplib_benchmark" / "ground_truth" / "optima.csv"
RESULT_CSV = REPO / "tsplib_benchmark" / "results" / "solver_wall_times_large.csv"

LKH_BIN = "/home/mig25/LKH-3.0.13/LKH"
LKH_COST_RE = re.compile(r"Cost\.min\s*=\s*([0-9.]+)")

# Instances (n > 1000). pla7397 is used as a scaling probe (Task 3).
MEDIUM = [
    "pr1002", "u1060", "vm1084", "pcb1173", "d1291", "rl1304", "rl1323",
    "nrw1379", "fl1400", "u1432", "fl1577", "d1655", "vm1748", "u1817",
    "rl1889", "d2103", "u2152", "u2319", "pr2392", "pcb3038", "fl3795",
    "fnl4461",
]
LARGE = [
    "rl5915", "rl5934", "rl11849", "usa13509", "brd14051", "d15112", "d18512",
]
SCALING_PROBE = ["pla7397"]  # MAX_TRIALS=1000

ALL_INSTANCES = MEDIUM + LARGE + SCALING_PROBE

MAX_WORKERS = 2
DEFAULT_MAX_TRIALS = 10000
PROBE_MAX_TRIALS = 1000

# Thread-safe print + CSV append
_io_lock = threading.Lock()


def win_to_wsl(p: Path) -> str:
    s = str(p).replace("\\", "/")
    if len(s) > 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:]
        return f"/mnt/{drive}{rest}"
    return s


def parse_dimension(tsp_path: Path) -> int | None:
    with tsp_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DIMENSION"):
                try:
                    return int(line.split(":")[-1].strip().split()[0])
                except Exception:
                    return None
            if line == "NODE_COORD_SECTION" or line == "EOF":
                break
    return None


def load_optima() -> dict[str, int]:
    out: dict[str, int] = {}
    with OPTIMA_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[row["instance"]] = int(float(row["optimum"]))
            except Exception:
                pass
    return out


def load_done() -> set[str]:
    if not RESULT_CSV.exists():
        return set()
    done: set[str] = set()
    with RESULT_CSV.open("r", newline="") as f:
        for row in csv.DictReader(f):
            inst = row.get("instance", "").strip()
            if inst:
                done.add(inst)
    return done


def run_lkh_one(instance: str, max_trials: int) -> dict:
    """Run LKH-3 on a single instance. Returns result row dict."""
    tsp = INSTANCE_DIR / f"{instance}.tsp"
    n = parse_dimension(tsp)

    # Scratch dir (Windows-side, visible via /mnt in WSL)
    scratch = Path(tempfile.mkdtemp(
        prefix=f"lkh_{instance}_",
        dir=str(REPO / "tsplib_benchmark"),
    ))
    par_win = scratch / f"{instance}.par"
    tour_win = scratch / f"{instance}.tour"
    par_wsl = win_to_wsl(par_win)
    tour_wsl = win_to_wsl(tour_win)
    tsp_wsl = win_to_wsl(tsp)

    par_body = (
        f"PROBLEM_FILE = {tsp_wsl}\n"
        f"MAX_TRIALS = {max_trials}\n"
        f"RUNS = 1\n"
        f"SEED = 1\n"
        f"OUTPUT_TOUR_FILE = {tour_wsl}\n"
        f"TRACE_LEVEL = 1\n"
    )
    par_win.write_text(par_body, encoding="utf-8")

    with _io_lock:
        print(f"[start] {instance} n={n} max_trials={max_trials}", flush=True)

    run_cmd = ["wsl.exe", "-e", "bash", "-c", f"{LKH_BIN} {par_wsl}"]

    t0 = time.perf_counter()
    stdout = ""
    stderr = ""
    rc = -1
    crashed = False
    try:
        proc = subprocess.run(
            run_cmd, capture_output=True, text=True,
            errors="replace",
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        rc = proc.returncode
    except Exception as e:  # broad by design: we record the crash
        crashed = True
        stderr = f"Exception: {e}"
    wall = time.perf_counter() - t0

    cost = None
    # LKH prints "Cost.min = X" repeatedly; we take the LAST occurrence (best).
    matches = LKH_COST_RE.findall(stdout)
    if matches:
        try:
            cost = float(matches[-1])
        except Exception:
            cost = None

    ok = (rc == 0) and (cost is not None) and not crashed

    # Cleanup scratch
    for p in (par_win, tour_win):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    try:
        scratch.rmdir()
    except Exception:
        pass

    with _io_lock:
        status = "ok" if ok else ("crash" if crashed else f"fail(rc={rc})")
        cost_str = f"{cost:.0f}" if cost is not None else "None"
        print(f"[done ] {instance} n={n} wall={wall:.1f}s cost={cost_str} status={status}",
              flush=True)

    return {
        "instance": instance,
        "n": n if n is not None else "",
        "lkh_time_s": f"{wall:.6f}",
        "lkh_tour_len": f"{cost:.6f}" if cost is not None else "",
        "max_trials": max_trials,
        "status": "ok" if ok else ("crash" if crashed else f"fail(rc={rc})"),
        "stderr_tail": (stderr[-400:] if stderr else ""),
    }


def enrich_with_optimum(row: dict, optima: dict[str, int]) -> dict:
    inst = row["instance"]
    opt = optima.get(inst)
    published = opt if opt is not None else ""
    tour = row.get("lkh_tour_len", "")
    matches = ""
    if tour and opt is not None:
        try:
            tour_f = float(tour)
            matches = "True" if abs(tour_f - opt) < 0.5 else "False"
        except Exception:
            matches = ""
    row["published_optimum"] = published
    row["matches_optimum"] = matches
    return row


HEADER = [
    "instance", "n", "lkh_time_s", "lkh_tour_len",
    "matches_optimum", "published_optimum",
    "max_trials", "status", "stderr_tail",
]


def append_row(row: dict) -> None:
    write_header = not RESULT_CSV.exists()
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with _io_lock:
        with RESULT_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADER, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)


def main() -> None:
    optima = load_optima()
    done = load_done()

    # Build work list, preserve order: medium, large, probe
    work: list[tuple[str, int]] = []
    for inst in MEDIUM + LARGE:
        if inst in done:
            print(f"[skip ] {inst} already in CSV")
            continue
        work.append((inst, DEFAULT_MAX_TRIALS))
    for inst in SCALING_PROBE:
        if inst in done:
            print(f"[skip ] {inst} already in CSV (probe)")
            continue
        work.append((inst, PROBE_MAX_TRIALS))

    print(f"Running {len(work)} instances with {MAX_WORKERS} workers")
    print("=" * 70, flush=True)

    # Run large (n > 10000) serially first to avoid 2x big jobs colliding on RAM.
    # Everything else goes through the pool.
    def _size(inst: str) -> int:
        tsp = INSTANCE_DIR / f"{inst}.tsp"
        d = parse_dimension(tsp) or 0
        return d

    # Sort: smallest first (better throughput) — but keep giants last.
    work.sort(key=lambda t: _size(t[0]))

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        fut_map = {pool.submit(run_lkh_one, inst, mt): inst for inst, mt in work}
        for fut in as_completed(fut_map):
            inst = fut_map[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "instance": inst,
                    "n": _size(inst),
                    "lkh_time_s": "",
                    "lkh_tour_len": "",
                    "max_trials": "",
                    "status": f"exception: {e}",
                    "stderr_tail": "",
                }
            row = enrich_with_optimum(row, optima)
            append_row(row)

    t_total = time.perf_counter() - t_start
    print("=" * 70)
    print(f"All done in {t_total/60:.2f} min")
    print(f"CSV: {RESULT_CSV}")


if __name__ == "__main__":
    main()
