"""
Benchmark Concorde vs LKH-3 on TSPLIB95 EUC_2D instances with n <= 1000.

Records wall-clock seconds and tour lengths for each solver, then computes
per-instance "best" wall time = min of the two *when both agree on optimal*.
Concorde is the fallback when LKH disagrees (Concorde is provably optimal).

Output: tsplib_benchmark/results/solver_wall_times.csv
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTANCE_DIR = REPO / "tsplib_benchmark" / "instances"
RESULT_CSV = REPO / "tsplib_benchmark" / "results" / "solver_wall_times.csv"
TIMEOUT_S = 300

CONCORDE_BIN = "/usr/local/bin/concorde"
LKH_BIN = "/home/mig25/LKH-3.0.13/LKH"

COST_RE = re.compile(r"Optimal Solution:\s*([0-9.]+)")
LKH_COST_RE = re.compile(r"Cost\.min\s*=\s*([0-9.]+)")


def win_to_wsl(p: Path) -> str:
    s = str(p).replace("\\", "/")
    if len(s) > 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:]
        return f"/mnt/{drive}{rest}"
    return s


def parse_header(tsp_path: Path) -> tuple[int | None, str | None]:
    dim = None
    ewt = None
    with tsp_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DIMENSION"):
                try:
                    dim = int(line.split(":")[-1].strip().split()[0])
                except Exception:
                    pass
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                ewt = line.split(":")[-1].strip().split()[0]
            if line == "NODE_COORD_SECTION" or line == "EOF":
                break
            if dim is not None and ewt is not None:
                break
    return dim, ewt


def run_concorde(instance_path: Path, work_dir: Path) -> tuple[float, float | None, bool]:
    """Run Concorde in work_dir on a copy of the instance. Returns (wall_s, tour_len, ok)."""
    name = instance_path.stem
    local_tsp = work_dir / instance_path.name
    shutil.copy2(instance_path, local_tsp)

    wsl_work = win_to_wsl(work_dir)
    wsl_tsp = f"{wsl_work}/{instance_path.name}"
    # Concorde auto-detects EUC_2D from the file header; no -N flag needed.
    cmd = [
        "wsl", "-e", "bash", "-c",
        f"cd {wsl_work} && {CONCORDE_BIN} {wsl_tsp}",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=TIMEOUT_S,
        )
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        _cleanup_concorde(work_dir, name)
        return wall, None, False

    cost = None
    m = COST_RE.search(proc.stdout)
    if m:
        try:
            cost = float(m.group(1))
        except Exception:
            cost = None
    ok = (proc.returncode == 0) and (cost is not None)

    _cleanup_concorde(work_dir, name)
    return wall, cost, ok


def _cleanup_concorde(work_dir: Path, name: str) -> None:
    for ext in ("mas", "sav", "pul", "sol", "res", "tsp"):
        p = work_dir / f"{name}.{ext}"
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    # Also catch anything with the concorde prefix like O<pid>
    for p in work_dir.glob(f"{name}.*"):
        try:
            p.unlink()
        except Exception:
            pass


def run_lkh(instance_path: Path, work_dir: Path) -> tuple[float, float | None, bool]:
    """Run LKH-3 via a generated .par file written on Windows (visible through /mnt).
    Returns (wall_s, tour_len, ok).
    """
    name = instance_path.stem
    wsl_tsp = win_to_wsl(instance_path)

    par_win = work_dir / f"{name}.par"
    tour_win = work_dir / f"{name}.tour"
    par_wsl = win_to_wsl(par_win)
    tour_wsl = win_to_wsl(tour_win)

    par_body = (
        f"PROBLEM_FILE = {wsl_tsp}\n"
        f"MAX_TRIALS = 10000\n"
        f"RUNS = 1\n"
        f"SEED = 1\n"
        f"OUTPUT_TOUR_FILE = {tour_wsl}\n"
    )
    par_win.write_text(par_body, encoding="utf-8")

    run_cmd = ["wsl", "-e", "bash", "-c", f"{LKH_BIN} {par_wsl}"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            run_cmd, capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        for p in (par_win, tour_win):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        return wall, None, False

    cost = None
    m = LKH_COST_RE.search(proc.stdout)
    if m:
        try:
            cost = float(m.group(1))
        except Exception:
            cost = None
    ok = (proc.returncode == 0) and (cost is not None)

    for p in (par_win, tour_win):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    return wall, cost, ok


def main() -> None:
    # Collect EUC_2D instances with n <= 1000.
    candidates: list[tuple[str, int, Path]] = []
    for tsp in sorted(INSTANCE_DIR.glob("*.tsp")):
        dim, ewt = parse_header(tsp)
        if dim is None or ewt is None:
            continue
        if ewt == "EUC_2D" and dim <= 1000:
            candidates.append((tsp.stem, dim, tsp))

    print(f"Found {len(candidates)} EUC_2D instances with n <= 1000:")
    for name, n, _ in candidates:
        print(f"  {name:14s} n={n}")
    print()

    # Working scratch dir for Concorde (on Windows side, mapped into WSL).
    work_dir = REPO / "tsplib_benchmark" / "_concorde_scratch"
    work_dir.mkdir(parents=True, exist_ok=True)

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    N = len(candidates)
    for k, (name, n, tsp) in enumerate(candidates, 1):
        if n > 1000:
            continue

        # Concorde
        c_wall, c_cost, c_ok = run_concorde(tsp, work_dir)
        # LKH
        l_wall, l_cost, l_ok = run_lkh(tsp, work_dir)

        same = False
        if c_ok and l_ok and c_cost is not None and l_cost is not None:
            same = abs(c_cost - l_cost) <= 1e-6

        if same:
            if c_wall <= l_wall:
                best_time = c_wall
                best_solver = "concorde"
            else:
                best_time = l_wall
                best_solver = "lkh"
        else:
            if c_ok:
                best_time = c_wall
                best_solver = "concorde"
            elif l_ok:
                best_time = l_wall
                best_solver = "lkh"
            else:
                best_time = float("nan")
                best_solver = "none"

        c_wall_str = f"{c_wall:.3f}s" if c_ok else "timeout"
        l_wall_str = f"{l_wall:.3f}s" if l_ok else "fail"
        best_str = f"{best_time:.3f}s" if best_time == best_time else "nan"
        print(f"[{k}/{N}] {name} n={n} concorde={c_wall_str} lkh={l_wall_str} "
              f"same={same} best={best_str}", flush=True)

        rows.append({
            "instance": name,
            "n": n,
            "concorde_time_s": f"{c_wall:.6f}" if c_ok else "",
            "concorde_tour_len": f"{c_cost:.6f}" if c_cost is not None else "",
            "lkh_time_s": f"{l_wall:.6f}" if l_ok else "",
            "lkh_tour_len": f"{l_cost:.6f}" if l_cost is not None else "",
            "best_time_s": f"{best_time:.6f}" if best_time == best_time else "",
            "best_solver": best_solver,
            "same_optimal": str(same),
        })

    # Remove scratch dir
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    # Per-bucket averages.
    buckets = [(0, 150), (150, 400), (400, 1000)]
    bucket_rows: list[dict] = []
    print()
    print("Per-bucket summary:")
    for lo, hi in buckets:
        vals = []
        for r in rows:
            if r["best_time_s"] == "":
                continue
            nn = int(r["n"])
            if lo < nn <= hi:
                vals.append(float(r["best_time_s"]))
        if vals:
            vals_sorted = sorted(vals)
            mean = sum(vals) / len(vals)
            median = vals_sorted[len(vals_sorted) // 2] if len(vals_sorted) % 2 else \
                0.5 * (vals_sorted[len(vals_sorted) // 2 - 1] + vals_sorted[len(vals_sorted) // 2])
            print(f"  ({lo},{hi}]: count={len(vals)} mean={mean:.3f}s median={median:.3f}s")
            bucket_rows.append({
                "bucket": f"({lo},{hi}]",
                "count": len(vals),
                "mean_best_time_s": f"{mean:.6f}",
                "median_best_time_s": f"{median:.6f}",
            })
        else:
            print(f"  ({lo},{hi}]: (empty)")
            bucket_rows.append({
                "bucket": f"({lo},{hi}]", "count": 0,
                "mean_best_time_s": "", "median_best_time_s": "",
            })

    # Write CSV.
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "instance", "n", "concorde_time_s", "concorde_tour_len",
            "lkh_time_s", "lkh_tour_len", "best_time_s", "best_solver",
            "same_optimal",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        # Separator + bucket section.
        f.write("\n# bucket summary\n")
        bw = csv.DictWriter(f, fieldnames=[
            "bucket", "count", "mean_best_time_s", "median_best_time_s",
        ])
        bw.writeheader()
        for br in bucket_rows:
            bw.writerow(br)

    print()
    print(f"Wrote {RESULT_CSV}")


if __name__ == "__main__":
    main()
