"""
Parallel benchmark of Concorde and LKH-3 on TSPLIB95 EUC_2D instances with
n <= 1000.

Launches a ProcessPoolExecutor with max(1, cpu_count()-1) workers. Each worker
runs Concorde then LKH-3 sequentially inside its OWN scratch directory, so
.sav/.sol/.par artifacts never collide across workers.

Output: tsplib_benchmark/results/solver_wall_times.csv
(resumable: pre-existing rows for fully-populated instances are preserved and
skipped.)
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTANCE_DIR = REPO / "tsplib_benchmark" / "instances"
RESULT_CSV = REPO / "tsplib_benchmark" / "results" / "solver_wall_times.csv"

TIMEOUT_S = 300
SUBPROC_TIMEOUT = 320

CONCORDE_BIN = "/usr/local/bin/concorde"
LKH_BIN = "/home/mig25/LKH-3.0.13/LKH"

COST_RE = re.compile(r"Optimal Solution:\s*([0-9.]+)")
LKH_COST_RE = re.compile(r"Cost\.min\s*=\s*([0-9.]+)")

FIELDNAMES = [
    "instance", "n", "concorde_time_s", "concorde_tour_len",
    "lkh_time_s", "lkh_tour_len", "best_time_s", "best_solver",
    "same_optimal",
]


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


def _run_concorde(local_tsp: Path, work_dir: Path) -> tuple[float, float | None, bool]:
    wsl_work = win_to_wsl(work_dir)
    cmd = [
        "wsl", "-e", "bash", "-c",
        f"cd {wsl_work} && {CONCORDE_BIN} {local_tsp.name}",
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROC_TIMEOUT,
        )
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        return wall, None, False

    cost = None
    m = COST_RE.search(proc.stdout)
    if m:
        try:
            cost = float(m.group(1))
        except Exception:
            cost = None
    ok = (proc.returncode == 0) and (cost is not None)
    return wall, cost, ok


def _run_lkh(local_tsp: Path, work_dir: Path) -> tuple[float, float | None, bool]:
    """Par file is created inside work_dir with unique name so workers don't clash."""
    name = local_tsp.stem
    par_path = work_dir / f"{name}.par"
    tour_path = work_dir / f"{name}.tour"
    par_body = (
        f"PROBLEM_FILE = {local_tsp.name}\n"
        f"MAX_TRIALS = 10000\n"
        f"RUNS = 1\n"
        f"SEED = 1\n"
        f"OUTPUT_TOUR_FILE = {tour_path.name}\n"
    )
    par_path.write_text(par_body, encoding="utf-8")

    wsl_work = win_to_wsl(work_dir)
    run_cmd = [
        "wsl", "-e", "bash", "-c",
        f"cd {wsl_work} && {LKH_BIN} {par_path.name}",
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            run_cmd, capture_output=True, text=True, timeout=SUBPROC_TIMEOUT,
        )
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        return wall, None, False

    cost = None
    m = LKH_COST_RE.search(proc.stdout)
    if m:
        try:
            cost = float(m.group(1))
        except Exception:
            cost = None
    ok = (proc.returncode == 0) and (cost is not None)
    return wall, cost, ok


def run_one(args: tuple[str, int, str]) -> dict:
    """Worker function: run Concorde then LKH on one instance in a scratch dir."""
    name, n, tsp_str = args
    tsp_path = Path(tsp_str)

    scratch = Path(tempfile.mkdtemp(prefix="tsp_solver_"))
    try:
        local_tsp = scratch / tsp_path.name
        shutil.copy2(tsp_path, local_tsp)

        c_wall, c_cost, c_ok = _run_concorde(local_tsp, scratch)
        l_wall, l_cost, l_ok = _run_lkh(local_tsp, scratch)

        same = False
        if c_ok and l_ok and c_cost is not None and l_cost is not None:
            same = abs(c_cost - l_cost) <= 1e-6

        if same:
            if c_wall <= l_wall:
                best_time, best_solver = c_wall, "concorde"
            else:
                best_time, best_solver = l_wall, "lkh"
        elif c_ok:
            best_time, best_solver = c_wall, "concorde"
        elif l_ok:
            best_time, best_solver = l_wall, "lkh"
        else:
            best_time, best_solver = float("nan"), "none"

        return {
            "instance": name,
            "n": n,
            "concorde_time_s": f"{c_wall:.6f}" if c_ok else "",
            "concorde_tour_len": f"{c_cost:.6f}" if c_cost is not None else "",
            "lkh_time_s": f"{l_wall:.6f}" if l_ok else "",
            "lkh_tour_len": f"{l_cost:.6f}" if l_cost is not None else "",
            "best_time_s": f"{best_time:.6f}" if best_time == best_time else "",
            "best_solver": best_solver,
            "same_optimal": str(same),
            "_c_ok": c_ok,
            "_l_ok": l_ok,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def load_existing() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not RESULT_CSV.exists():
        return rows
    with RESULT_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("instance"):
                continue
            if r["instance"].startswith("#"):
                continue
            rows[r["instance"]] = r
    return rows


def is_row_complete(r: dict) -> bool:
    return bool(r.get("concorde_time_s")) and bool(r.get("lkh_time_s"))


def main() -> None:
    # Collect EUC_2D instances with n <= 1000.
    candidates: list[tuple[str, int, Path]] = []
    for tsp in sorted(INSTANCE_DIR.glob("*.tsp")):
        dim, ewt = parse_header(tsp)
        if dim is None or ewt is None:
            continue
        if ewt == "EUC_2D" and dim <= 1000:
            candidates.append((tsp.stem, dim, tsp))

    print(f"Found {len(candidates)} EUC_2D instances with n <= 1000.", flush=True)

    existing = load_existing()
    todo: list[tuple[str, int, str]] = []
    for name, n, p in candidates:
        if name in existing and is_row_complete(existing[name]):
            continue
        todo.append((name, n, str(p)))

    print(f"Already complete: {len(candidates) - len(todo)}", flush=True)
    print(f"Remaining to run: {len(todo)}", flush=True)

    if not todo:
        print("Nothing to do.", flush=True)
    else:
        # Cap workers at 8: each Concorde/LKH uses multiple threads internally
        # and WSL subprocess startup serializes; over-parallelizing on a 20-core
        # box causes CPU starvation and spurious 300s "timeouts".
        env_cap = int(os.environ.get("TSP_MAX_WORKERS", "0") or 0)
        default_cap = min(8, max(1, (os.cpu_count() or 2) - 1))
        max_workers = env_cap if env_cap > 0 else default_cap
        print(f"Launching ProcessPoolExecutor with {max_workers} workers.", flush=True)

        t0 = time.perf_counter()
        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_one, t): t for t in todo}
            N = len(futures)
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    r = fut.result()
                except Exception as e:
                    tname = futures[fut][0]
                    print(f"[{i}/{N}] {tname} FAILED: {e}", flush=True)
                    continue
                c_ok = r.pop("_c_ok", False)
                l_ok = r.pop("_l_ok", False)
                c_s = r["concorde_time_s"] or "timeout"
                l_s = r["lkh_time_s"] or "fail"
                print(
                    f"[{i}/{N}] {r['instance']:12s} n={r['n']:4d} "
                    f"concorde={c_s} lkh={l_s} same={r['same_optimal']} "
                    f"best={r['best_time_s'] or 'nan'}",
                    flush=True,
                )
                results.append(r)
                existing[r["instance"]] = r

                # Periodic flush so partial progress survives interruption.
                if i % 5 == 0 or i == N:
                    _write_csv(existing, candidates)

        wall = time.perf_counter() - t0
        print(f"\nParallel run finished in {wall:.1f}s ({wall/60:.1f} min).", flush=True)

    # Final write with bucket summary appended.
    _write_csv(existing, candidates, include_buckets=True)

    _print_bucket_summary(existing, candidates)


def _write_csv(
    existing: dict[str, dict],
    candidates: list[tuple[str, int, Path]],
    include_buckets: bool = False,
) -> None:
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for name, n, _ in candidates:
            r = existing.get(name)
            if r is None:
                continue
            w.writerow({k: r.get(k, "") for k in FIELDNAMES})
        if include_buckets:
            f.write("\n# bucket summary\n")
            bw = csv.DictWriter(f, fieldnames=[
                "bucket", "count", "mean_best_time_s", "median_best_time_s",
            ])
            bw.writeheader()
            for br in _bucket_rows(existing, candidates):
                bw.writerow(br)


def _bucket_rows(existing, candidates):
    rows_with_best = []
    for name, n, _ in candidates:
        r = existing.get(name)
        if r is None:
            continue
        if not r.get("best_time_s"):
            continue
        rows_with_best.append((n, float(r["best_time_s"])))
    buckets = [(0, 150), (150, 400), (400, 1000)]
    out = []
    for lo, hi in buckets:
        vals = [t for (nn, t) in rows_with_best if lo < nn <= hi]
        if vals:
            vals_sorted = sorted(vals)
            mean = sum(vals) / len(vals)
            mid = len(vals_sorted) // 2
            median = vals_sorted[mid] if len(vals_sorted) % 2 \
                else 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
            out.append({
                "bucket": f"({lo},{hi}]",
                "count": len(vals),
                "mean_best_time_s": f"{mean:.6f}",
                "median_best_time_s": f"{median:.6f}",
            })
        else:
            out.append({
                "bucket": f"({lo},{hi}]", "count": 0,
                "mean_best_time_s": "", "median_best_time_s": "",
            })
    return out


def _print_bucket_summary(existing, candidates):
    print("\nPer-bucket summary (best_time_s = fastest successful solver):")
    for br in _bucket_rows(existing, candidates):
        if br["count"]:
            print(
                f"  {br['bucket']}: count={br['count']} "
                f"mean={br['mean_best_time_s']}s "
                f"median={br['median_best_time_s']}s"
            )
        else:
            print(f"  {br['bucket']}: (empty)")

    # Disagreements.
    disagreements = []
    for name, n, _ in candidates:
        r = existing.get(name)
        if not r:
            continue
        if r.get("concorde_tour_len") and r.get("lkh_tour_len"):
            if r.get("same_optimal", "").lower() != "true":
                disagreements.append(
                    (name, n, r["concorde_tour_len"], r["lkh_tour_len"])
                )
    if disagreements:
        print("\nInstances where Concorde and LKH-3 disagreed on tour length:")
        for name, n, c, l in disagreements:
            print(f"  {name} n={n}: concorde={c} lkh={l}")
    else:
        print("\nConcorde and LKH-3 agreed on every successful instance.")

    print(f"\nWrote {RESULT_CSV}")


if __name__ == "__main__":
    main()
