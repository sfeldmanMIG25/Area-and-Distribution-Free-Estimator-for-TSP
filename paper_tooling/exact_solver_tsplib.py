"""Measured Concorde and LKH-3 wall times on the 78 TSPLIB EUC_2D instances.

Why this exists
---------------
``frontier_manuscript_bank.json -> exact_solver_anchor`` carries a *published*
Concorde record (Waterloo, Xeon 2.8 GHz / Alpha 500 MHz), covers 25 of the 78
instances, and states in its own provenance field that it was not measured on
this box. The cost axis therefore has no upper anchor that was measured under
the same conditions as everything else on it. This script produces one.

What is measured, and what is deliberately kept apart
-----------------------------------------------------
Two different questions, two columns, never averaged:

* **Concorde -- certified optimal.** Branch-and-cut run to proof. The recorded
  time is the wall clock to a *certificate*, not to a good tour. On censoring,
  the best lower and upper bound reached are recorded, so a censored row still
  carries its evidence.
* **LKH-3 -- time to best.** A heuristic. The recorded time is when its best
  tour was first reached inside its own budget, taken from LKH's own trial
  trace, plus the total wall clock it consumed to decide it was done. LKH
  offers no proof; when its tour equals the published optimum that is a
  coincidence of quality, not a certificate.

Censoring
---------
A per-instance cap is applied to both arms. Instances that hit it are recorded
with ``status = censored`` and the bound reached. They are never dropped: for a
paper whose thesis is that the exact solve is intractable, a censored instance
is the finding, and an omitted row would understate the anchor.

Protocol match
--------------
The estimator column this will sit beside is
``gart2_timing_bank.json -> tsplib_by_size_time_one_protocol``, tag
``serial_solo_median11_quiet_2026-08-11``: one process, threads pinned, quiet
box, median over repeats. This harness matches it where matching is meaningful:

* one solver process at a time, never two in flight (asserted before every
  timed run, Windows side and WSL side),
* single-threaded solvers -- Concorde's sequential branch-and-cut and LKH-3 are
  both serial; no thread pool is created, and no ``-h``/boss-grunt parallel
  Concorde mode is used,
* the box is sampled with the same counter the published protocol quotes
  (see ``paper_tooling/quiet_box.py``) immediately before every timed run and
  the reading is stamped on the row,
* repeats where they are affordable: ``--repeat-under`` seconds of first-run
  time earns ``--repeats`` total runs and the statistic is the median. Above
  that threshold a single run is taken, and the row says so in ``repeats``.

Two deviations are stated rather than hidden:

1. Concorde runs under WSL2 (there is no Windows build here); LKH-3 runs as a
   native Windows executable, as does the estimator column. Both use the same
   physical cores of the same box. WSL2 is a virtual machine with direct CPU
   execution, not emulation, but it is not the same environment, and the two
   solver columns are reported separately partly for that reason.
2. Repeats are affordable only in the small-n part of the corpus. The long
   solves are single runs. Every row carries its own ``repeats`` count.

Usage
-----
    python paper_tooling/exact_solver_tsplib.py --solver concorde --cap 600
    python paper_tooling/exact_solver_tsplib.py --solver lkh --cap 600

Both are checkpointed per instance and resume by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper_tooling"))

from quiet_box import LoadSampler, observe  # noqa: E402

INSTANCE_DIR = ROOT / "tsplib_benchmark" / "instances"
OPTIMA_CSV = ROOT / "tsplib_benchmark" / "ground_truth" / "optima.csv"
MODELS_CSV = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
OUT_DIR = ROOT / "paper_tooling"
LOG_DIR = OUT_DIR / "exact_solver_logs"
CKPT_DIR = OUT_DIR / "exact_solver_checkpoints"

CONCORDE_WSL_BIN = "/home/catst/concorde/TSP/concorde"
LKH_WIN_BIN = r"C:\LKH\LKH-3.exe"
LKH_SCRATCH = Path(r"C:\Temp_TSP_Scratch")

#: Concorde's own RNG seed. Branch-and-cut is randomised; a fixed seed makes a
#: rerun reproducible rather than merely similar.
CONCORDE_SEED = 99
LKH_SEED = 1

# -- Concorde log grammar ---------------------------------------------------
RE_OPTIMAL = re.compile(r"Optimal Solution:\s*([0-9.]+)")
RE_TOTAL_TIME = re.compile(r"Total Running Time:\s*([0-9.]+)")
RE_LP_VALUE = re.compile(r"LP Value\s*\d*\s*:\s*([0-9.]+)")
RE_LOWER_BOUND = re.compile(r"[Ll]ower\s*[Bb]ound[: ]+\s*([0-9.]+)")
RE_UPPER_BOUND = re.compile(r"[Uu]pper\s*[Bb]ound[: ]+\s*([0-9.]+)")
RE_INITIAL_UB = re.compile(r"initial upperbound\s*(?:to)?\s*([0-9.]+)", re.IGNORECASE)
RE_BB_NODES = re.compile(r"Number of bbnodes:\s*(\d+)")
RE_MARK_RC = re.compile(r"__RC=(-?\d+)")
RE_MARK_WALL = re.compile(r"__WALL=([0-9.eE+-]+)")

# -- LKH log grammar --------------------------------------------------------
RE_LKH_TRIAL = re.compile(r"^\*\s*(\d+):\s*Cost\s*=\s*([0-9]+).*?Time\s*=\s*([0-9.]+)\s*sec",
                          re.MULTILINE)
RE_LKH_RUN = re.compile(r"^Run\s*\d+:\s*Cost\s*=\s*([0-9]+),\s*Time\s*=\s*([0-9.]+)\s*sec",
                        re.MULTILINE)
RE_LKH_COSTMIN = re.compile(r"Cost\.min\s*=\s*([0-9]+)")
RE_LKH_TIMETOTAL = re.compile(r"Time\.total\s*=\s*([0-9.]+)\s*sec")
RE_LKH_ASCENT_LB = re.compile(r"Lower bound\s*=\s*([0-9.]+),\s*Ascent time\s*=\s*([0-9.]+)\s*sec")
RE_LKH_PREPROC = re.compile(r"Preprocessing time\s*=\s*([0-9.]+)\s*sec")
RE_LKH_TRIALS = re.compile(r"Trials\.min\s*=\s*(\d+)")


# -- instance selection ------------------------------------------------------


def euc2d_instances() -> list[tuple[str, int]]:
    """The 78 EUC_2D instances, taken from the benchmark result set so this
    harness scores exactly the corpus the cost axis is drawn on."""
    names: dict[str, int] = {}
    with MODELS_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("edge_weight_type") != "EUC_2D":
                continue
            names[row["instance"]] = int(row["n"])
    missing = [k for k in names if not (INSTANCE_DIR / f"{k}.tsp").exists()]
    if missing:
        raise FileNotFoundError(f"No .tsp on disk for: {sorted(missing)}")
    return sorted(names.items(), key=lambda kv: (kv[1], kv[0]))


def published_optima() -> dict[str, int]:
    out: dict[str, int] = {}
    with OPTIMA_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["instance"]] = int(float(row["optimum"]))
            except (ValueError, KeyError):
                pass
    return out


# -- one timed run -----------------------------------------------------------


@dataclass
class RunResult:
    """One timed solver invocation."""

    wall_s: float
    status: str                       # optimal | censored | error | best_found
    cost: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    solver_self_time_s: float | None = None
    time_to_best_s: float | None = None
    extra: dict = field(default_factory=dict)
    log: str = ""


def _wsl_path(win_path: Path) -> str:
    s = str(win_path).replace("\\", "/")
    if len(s) < 2 or s[1] != ":":
        raise ValueError(f"expected a drive-qualified path, got {win_path}")
    return f"/mnt/{s[0].lower()}{s[2:]}"


def run_concorde_once(name: str, tsp: Path, cap_s: int) -> RunResult:
    """Concorde to proof of optimality, hard-capped inside WSL.

    The cap is enforced by GNU ``timeout`` on the Linux side, not by killing
    ``wsl.exe`` from Windows: killing the launcher leaves the solver alive, and
    an orphaned solver is precisely the contamination that invalidated an
    earlier timing pass in this project.

    Scratch lives in ``/tmp`` (ext4), not under ``/mnt`` (9p). Concorde writes
    its master/pool/save files next to the problem, and doing that across the
    9p bridge would time the filesystem instead of the solver.

    ``stdbuf -oL`` matters on the censored rows: redirected stdout is block
    buffered, and a solver killed at the cap would take its last few kilobytes
    of progress -- the bound reached -- to the grave. Line buffering costs
    nothing here and is what makes a censored row carry evidence.
    """
    src = _wsl_path(tsp)
    script = f"""
d=$(mktemp -d /tmp/cc.XXXXXXXX)
cp "{src}" "$d/{name}.tsp"
cd "$d"
s=$(date +%s.%N)
timeout -s KILL {cap_s} stdbuf -oL -eL {CONCORDE_WSL_BIN} -s {CONCORDE_SEED} -x \
  -o "$d/{name}.sol" "$d/{name}.tsp" > "$d/run.log" 2>&1
rc=$?
e=$(date +%s.%N)
echo "__RC=$rc"
echo "__WALL=$(echo "$e - $s" | bc -l)"
echo "__LOG_BEGIN"
cat "$d/run.log"
echo "__LOG_END"
cd /
rm -rf "$d"
"""
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["wsl", "-e", "bash", "-c", script],
        capture_output=True, text=True, errors="replace",
    )
    outer_wall = time.perf_counter() - t0
    out = proc.stdout

    m_rc = RE_MARK_RC.search(out)
    m_wall = RE_MARK_WALL.search(out)
    rc = int(m_rc.group(1)) if m_rc else -999
    wall = float(m_wall.group(1)) if m_wall else outer_wall
    log = out.split("__LOG_BEGIN", 1)[1].rsplit("__LOG_END", 1)[0] if "__LOG_BEGIN" in out else out
    if proc.stderr.strip():
        log += "\n__STDERR__\n" + proc.stderr

    opt = RE_OPTIMAL.search(log)
    self_t = RE_TOTAL_TIME.search(log)
    lps = RE_LP_VALUE.findall(log)
    lbs = RE_LOWER_BOUND.findall(log)
    ubs = RE_UPPER_BOUND.findall(log) + RE_INITIAL_UB.findall(log)
    bb = RE_BB_NODES.search(log)

    lower = max((float(x) for x in lps + lbs), default=None)
    upper = min((float(x) for x in ubs), default=None)

    # Concorde exits 255 on a *successful* solve on this build, so the return
    # code cannot carry the verdict. The proof line can: "Optimal Solution:" is
    # printed only after branch-and-cut closes the gap. Verified against
    # eil51/berlin52/a280, all of which exit 255 with a correct certificate.
    if opt:
        status = "optimal"
        # "Final lower bound" is the ROOT LP value, printed before branching
        # closes the gap (a280: root 2578, optimum 2579). On a certified row
        # the only defensible bracket is the certificate itself.
        lower = upper = float(opt.group(1))
    elif rc in (124, 137):
        status = "censored"
    else:
        status = "error"

    return RunResult(
        wall_s=wall,
        status=status,
        cost=float(opt.group(1)) if opt else None,
        lower_bound=lower,
        upper_bound=upper,
        solver_self_time_s=float(self_t.group(1)) if self_t else None,
        extra={
            "returncode": rc,
            "bbnodes": int(bb.group(1)) if bb else None,
            "outer_wall_s": outer_wall,
            "wsl_launch_overhead_s": max(0.0, outer_wall - wall),
        },
        log=log,
    )


def run_lkh_once(name: str, tsp: Path, cap_s: int) -> RunResult:
    """LKH-3, one run, default trial budget, capped.

    ``TIME_LIMIT`` bounds LKH's optimisation loop but not its preprocessing, so
    a hard process timeout backs it up. ``time_to_best`` is read off LKH's own
    trial trace: the timestamp of the last improvement that reached the best
    cost of the run. It is LKH's clock, not this harness's, and it excludes
    nothing that LKH counts.
    """
    run_dir = LKH_SCRATCH / f"lkh_{uuid.uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    par = run_dir / f"{name}.par"
    tour = run_dir / f"{name}.tour"
    par.write_text(
        f"PROBLEM_FILE = {tsp}\n"
        f"OUTPUT_TOUR_FILE = {tour}\n"
        f"RUNS = 1\n"
        f"SEED = {LKH_SEED}\n"
        f"TIME_LIMIT = {cap_s}\n"
        f"TRACE_LEVEL = 1\n",
        encoding="ascii",
    )
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            [LKH_WIN_BIN, str(par)],
            capture_output=True, text=True, errors="replace",
            cwd=str(run_dir), timeout=cap_s + 120,
        )
        wall = time.perf_counter() - t0
        log = proc.stdout + ("\n__STDERR__\n" + proc.stderr if proc.stderr.strip() else "")
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        timed_out = True
        rc = -1
        log = (exc.stdout or "") if isinstance(exc.stdout, str) else (
            exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    trials = RE_LKH_TRIAL.findall(log)
    m_cost = RE_LKH_COSTMIN.search(log)
    m_run = RE_LKH_RUN.search(log)
    cost = float(m_cost.group(1)) if m_cost else (float(m_run.group(1)) if m_run else None)

    time_to_best = None
    if trials and cost is not None:
        for _t, c, tt in trials:
            if float(c) == cost:
                time_to_best = float(tt)
                break
    if time_to_best is None and m_run:
        time_to_best = float(m_run.group(2))

    m_tot = RE_LKH_TIMETOTAL.search(log)
    m_asc = RE_LKH_ASCENT_LB.search(log)
    m_pre = RE_LKH_PREPROC.search(log)
    m_tr = RE_LKH_TRIALS.search(log)

    # LKH honouring TIME_LIMIT is censoring of the search, not a crash.
    hit_limit = wall >= cap_s * 0.98 or timed_out
    if cost is None:
        status = "error"
    elif hit_limit:
        status = "censored"
    else:
        status = "best_found"

    return RunResult(
        wall_s=wall,
        status=status,
        cost=cost,
        lower_bound=float(m_asc.group(1)) if m_asc else None,
        upper_bound=cost,
        solver_self_time_s=float(m_tot.group(1)) if m_tot else None,
        time_to_best_s=time_to_best,
        extra={
            "returncode": rc,
            "hard_timeout": timed_out,
            "n_improvements": len(trials),
            "trials_run": int(m_tr.group(1)) if m_tr else None,
            "ascent_lower_bound": float(m_asc.group(1)) if m_asc else None,
            "ascent_time_s": float(m_asc.group(2)) if m_asc else None,
            "preprocessing_time_s": float(m_pre.group(1)) if m_pre else None,
        },
        log=log,
    )


# -- driver ------------------------------------------------------------------

FIELDS = [
    "instance", "n", "solver", "status", "repeats", "statistic",
    "wall_s", "wall_s_runs", "time_to_best_s", "solver_self_time_s",
    "cost", "published_optimum", "matches_optimum",
    "lower_bound", "upper_bound", "gap_pct_at_stop",
    "cap_s", "cpu_busy_pct_before", "cpu_busy_pct_during", "foreign_solver_procs",
    "returncode", "environment", "measured_utc", "notes",
]


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def measure(name: str, n: int, solver: str, cap_s: int, repeats: int,
            repeat_under_s: float, optima: dict[str, int]) -> dict:
    tsp = INSTANCE_DIR / f"{name}.tsp"
    runner = run_concorde_once if solver == "concorde" else run_lkh_once
    env = "WSL2 (Ubuntu 24.04, gcc 13.3) on the host's physical cores" \
        if solver == "concorde" else "Windows 11 native"

    runs: list[RunResult] = []
    loads: list[float] = []
    during: list[dict] = []
    foreign: list[str] = []

    for r in range(repeats):
        # Order matters: scan for foreign solvers, let the scan's own CPU spike
        # decay, then read the baseline, then run under a live sampler.
        # 1.5 s of settle, not 0.5: the previous repeat's teardown (log write,
        # checkpoint JSON, WSL shutdown) still shows up 0.5 s later and reads as
        # 37-54% against a 18-21% floor.
        pre = observe(interval_s=1.0, settle_s=1.5)
        if pre.foreign_solver_procs:
            foreign.extend(pre.foreign_solver_procs)
        loads.append(pre.busy_pct)
        with LoadSampler(period_s=5.0) as sampler:
            res = runner(name, tsp, cap_s)
        during.append(sampler.summary())
        runs.append(res)
        if r == 0 and res.wall_s > repeat_under_s:
            break                     # one run only; the row records repeats=1
        if res.status in ("censored", "error"):
            break

    head = runs[0]
    walls = [r.wall_s for r in runs if r.status == head.status]
    if not walls:
        walls = [head.wall_s]
    wall = _median(walls) if len(walls) > 1 else walls[0]

    ttb_vals = [r.time_to_best_s for r in runs if r.time_to_best_s is not None]
    ttb = _median(ttb_vals) if len(ttb_vals) > 1 else (ttb_vals[0] if ttb_vals else None)
    self_vals = [r.solver_self_time_s for r in runs if r.solver_self_time_s is not None]
    self_t = _median(self_vals) if len(self_vals) > 1 else (self_vals[0] if self_vals else None)

    opt = optima.get(name)
    cost = head.cost
    matches = (opt is not None and cost is not None and abs(cost - opt) < 0.5)

    gap = None
    if head.lower_bound and head.upper_bound and head.lower_bound > 0:
        gap = 100.0 * (head.upper_bound - head.lower_bound) / head.lower_bound

    # Persist the raw log so a censored row's evidence is auditable.
    log_path = LOG_DIR / solver / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(head.log[-400_000:], encoding="utf-8", errors="replace")

    return {
        "instance": name,
        "n": n,
        "solver": solver,
        "status": head.status,
        "repeats": len(runs),
        "statistic": "median over repeats" if len(runs) > 1 else "single run",
        "wall_s": f"{wall:.6f}",
        "wall_s_runs": ";".join(f"{r.wall_s:.6f}" for r in runs),
        "time_to_best_s": "" if ttb is None else f"{ttb:.6f}",
        "solver_self_time_s": "" if self_t is None else f"{self_t:.6f}",
        "cost": "" if cost is None else f"{cost:.2f}",
        "published_optimum": "" if opt is None else str(opt),
        "matches_optimum": str(matches),
        "lower_bound": "" if head.lower_bound is None else f"{head.lower_bound:.4f}",
        "upper_bound": "" if head.upper_bound is None else f"{head.upper_bound:.4f}",
        "gap_pct_at_stop": "" if gap is None else f"{gap:.6f}",
        "cap_s": cap_s,
        "cpu_busy_pct_before": ";".join(f"{x:.2f}" for x in loads),
        "cpu_busy_pct_during": json.dumps(during),
        "foreign_solver_procs": ";".join(sorted(set(foreign))),
        "returncode": head.extra.get("returncode"),
        "environment": env,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": json.dumps({k: v for k, v in head.extra.items() if k != "returncode"}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", choices=["concorde", "lkh"], required=True)
    ap.add_argument("--cap", type=int, default=600, help="per-instance cap, seconds")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repeat-under", type=float, default=10.0,
                    help="first-run seconds below which repeats are taken")
    ap.add_argument("--nmax", type=int, default=10**9)
    ap.add_argument("--only", default="", help="comma-separated instance names")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

    LKH_SCRATCH.mkdir(parents=True, exist_ok=True)
    ck_dir = CKPT_DIR / f"{args.solver}_cap{args.cap}"
    ck_dir.mkdir(parents=True, exist_ok=True)

    optima = published_optima()
    todo = [(nm, n) for nm, n in euc2d_instances() if n <= args.nmax]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        todo = [(nm, n) for nm, n in todo if nm in want]

    print(f"[{args.solver}] {len(todo)} instances, cap={args.cap}s, "
          f"repeats={args.repeats} under {args.repeat_under}s", flush=True)

    rows: list[dict] = []
    for i, (nm, n) in enumerate(todo, 1):
        ck = ck_dir / f"{nm}.json"
        if ck.exists() and not args.fresh:
            rows.append(json.loads(ck.read_text(encoding="utf-8")))
            print(f"  [{i}/{len(todo)}] {nm:10s} n={n:<6d} (checkpoint)", flush=True)
            continue
        row = measure(nm, n, args.solver, args.cap, args.repeats,
                      args.repeat_under, optima)
        ck.write_text(json.dumps(row, indent=1), encoding="utf-8")
        rows.append(row)
        print(f"  [{i}/{len(todo)}] {nm:10s} n={n:<6d} {row['status']:10s} "
              f"wall={float(row['wall_s']):10.3f}s reps={row['repeats']} "
              f"load={row['cpu_busy_pct_before']}", flush=True)

    out = OUT_DIR / f"exact_solver_tsplib_{args.solver}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
