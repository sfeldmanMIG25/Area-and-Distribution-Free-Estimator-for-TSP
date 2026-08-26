"""Cost of the 1-tree ladder on TSPLIB EUC_2D, under the published protocol.

PROTOCOL BEING REPRODUCED
-------------------------
``gart2_timing_bank.json -> tsplib_by_size_time_one_protocol``, tag
``serial_solo_median11_quiet_2026-08-11``:

* one estimator per process (this file is invoked once per model; ``--all``
  spawns the subprocesses back to back, never concurrently),
* single thread of control, no pool, OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1,
* 11 repeats, statistic = median over repeats, relative IQR and range retained,
* the timed region is the one ``run_all_models_tsplib.py`` uses:
  ``model.estimate(coords, 2, grid_size=0)`` for GART 2.0,
  ``est.estimate(coords, 2, None)`` for the region estimators,
  ``est(coords)`` for Hilbert. Parsing is outside the clock,
* coordinates are ``info["raw_coords"].astype(np.float32)``, byte-identical to
  what the runner hands every model,
* published cell = median over the 78 EUC_2D instances of the per-instance
  median over repeats,
* every timed call's prediction is retained so determinism can be checked.

TWO DOCUMENTED DEVIATIONS
-------------------------
1. **Repeat count below 11 at large budgets.** 11 repeats of the full ladder on
   all 78 instances is ~14 CPU-hours; the ladder is run at 11 repeats to
   ``k=100`` (which is where the frontier's crux sits), 3 to ``k=500``, 1 to
   ``k=2000``. Each run re-measures the cheaper checkpoints, so the repeat
   count's effect on the published statistic is *measured* rather than assumed.
2. **Checkpointed timing.** One ascent to ``K`` is run and the wall clock is
   read off at each ladder budget, instead of running nine independent
   ascents. The reading at ``k`` covers exactly what ``HK_1Tree_k.estimate()``
   covers -- dedup, matrix build, ``k`` Prim passes -- because the ascent's
   schedule is budget-independent, so a ``k``-budget run *is* the prefix of a
   ``K``-budget run. ``--mode direct`` calls the real estimator objects instead
   and exists to check that claim end to end.

3. **Size cap and memory screen (``--nmax``, ``--mem-frac``).** The 1-tree is
   Theta(k n^2), so the tail of the corpus dominates the wall clock: at k=500
   ``d18512`` alone costs about as much as the other 77 instances together.
   ``--nmax`` drops instances above a stated cap and
   :func:`screen_instances` additionally drops any instance whose *predicted*
   peak allocation exceeds a fraction of free RAM, so an oversized instance is
   excluded by arithmetic before it is attempted rather than by an OOM. Every
   exclusion is written to ``hk1tree_timing_<tag>_excluded.csv``. Comparators
   are run uncapped; the matched subset is taken downstream.

Absolute milliseconds require a quiet box. This script samples CPU load before,
during and after, and stamps every output row with it. Read the load before
quoting an absolute.

    python paper_tooling/hk1tree_frontier_timing.py --all
    python paper_tooling/hk1tree_frontier_timing.py --model HK_ladder --kmax 100 --repeats 11
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tsplib_benchmark", ROOT / "lgbm_model_v3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from held_karp_1tree import (  # noqa: E402
    DENSE_MAX_N,
    INITIAL_PERIOD,
    _IMPROVE_RTOL,
    HeldKarp1Tree,
    _one_tree_coords,
    _one_tree_dense,
)
from hk1tree_frontier_accuracy import CHECKPOINTS  # noqa: E402

#: Timing uses the *shipped* kernel switch, not the accuracy sweep's raised one:
#: a cost measurement has to reflect the path ``HeldKarp1Tree.estimate`` would
#: actually take, and that path drops to the coordinate kernel above 16384.
KERNEL_SWITCH_N = DENSE_MAX_N

TSPLIB_DIR = ROOT / "tsplib_benchmark" / "instances"
TSPLIB_RESULTS = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"
OUT_DIR = ROOT / "paper_tooling"

#: The comparator column, exactly as ``tsplib_by_size_time_one_protocol``
#: reports it. GART 2.0 is the anchor the ratio is taken against; the three
#: closed forms bracket it from below.
COMPARATORS = ("GART_2.0", "MST_Only", "Asymptotic_MST", "Calibrated_MST_dn", "Hilbert")


def cpu_load() -> float:
    """Whole-box CPU load, percent. Stamped on every row -- an absolute
    millisecond figure is only as good as the number next to it."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Measure-Object -Property "
             "LoadPercentage -Average).Average"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out)
    except Exception:
        return float("nan")


def free_physical_bytes() -> float:
    """Free physical RAM, bytes. Read before every batch so the memory screen
    below is sized against what the box actually has, not against a constant."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out) * 1024.0
    except Exception:
        return float("nan")


def predicted_peak_bytes(coords: np.ndarray) -> tuple[int, int]:
    """``(deduplicated n, predicted peak resident bytes of one timed call)``.

    Computed *before* the instance is timed. The dense kernel materialises the
    full ``n x n`` float64 distance matrix, and NumPy holds three such arrays at
    once while evaluating ``sq[:,None] + sq[None,:] - 2 X X^T`` (the two addends
    and the product, then the difference while two addends are still live), so
    the peak is ``3 * 8 * n^2`` and not the ``8 * n^2`` of the retained matrix.
    Above ``DENSE_MAX_N`` the shipped estimator drops to the coordinate kernel,
    which is O(n) in memory. GART 1.0's 54.98 GiB requirement on ``pla85900``
    is the precedent for screening this rather than discovering it.
    """
    n = int(np.unique(np.asarray(coords, dtype=np.float64), axis=0).shape[0])
    if n <= DENSE_MAX_N:
        return n, int(3 * 8 * n * n)
    return n, int(16 * n * int(coords.shape[1]) + 64 * n)


def screen_instances(insts, nmax: int, mem_frac: float, nmin: int = 0):
    """Apply the size cap and the memory budget. Returns ``(kept, dropped)``.

    ``dropped`` carries the reason, so an exclusion is reported rather than
    silently applied.
    """
    free = free_physical_bytes()
    budget = free * mem_frac
    kept, dropped = [], []
    for nm, X in insts:
        n_u, peak = predicted_peak_bytes(X)
        why = None
        if nmax and X.shape[0] > nmax:
            why = f"n={X.shape[0]} > nmax={nmax}"
        elif nmin and X.shape[0] < nmin:
            why = f"n={X.shape[0]} < nmin={nmin}"
        elif peak > budget:
            why = (f"predicted peak {peak / 2**30:.2f} GiB > {mem_frac:.0%} of "
                   f"{free / 2**30:.2f} GiB free")
        if why is None:
            kept.append((nm, X))
        else:
            dropped.append({"instance": nm, "n": X.shape[0], "n_unique": n_u,
                            "predicted_peak_GiB": peak / 2**30, "reason": why})
    print(f"memory screen: {free / 2**30:.2f} GiB free, budget "
          f"{budget / 2**30:.2f} GiB, kept {len(kept)}, dropped {len(dropped)}",
          flush=True)
    for d in dropped:
        print(f"  DROPPED {d['instance']} (n={d['n']}): {d['reason']}", flush=True)
    return kept, dropped


def load_instances() -> list[tuple[str, np.ndarray]]:
    """The 78 EUC_2D instances, in the runner's dtype, sorted by name."""
    from tsplib_parser import parse_tsplib_file

    res = pd.read_csv(TSPLIB_RESULTS)
    names = sorted(res[(res.model == "GART_2.0") &
                       (res.edge_weight_type == "EUC_2D")].instance.astype(str))
    out = []
    for nm in names:
        info = parse_tsplib_file(str(TSPLIB_DIR / f"{nm}.tsp"))
        out.append((nm, info["raw_coords"].astype(np.float32)))
    if len(out) != 78:
        raise ValueError(f"expected 78 EUC_2D instances, got {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Timed checkpointed ascent
# ---------------------------------------------------------------------------
def timed_ladder(coords: np.ndarray, ks: tuple[int, ...]) -> tuple[dict[int, float],
                                                                   dict[int, float]]:
    """``({k: seconds}, {k: bound})`` for one instance, from one ascent.

    The clock starts where ``HeldKarp1Tree.estimate`` starts it: before the
    dedup, so the reading at ``k`` is the whole cost of asking for a ``k``-budget
    bound, not just the ascent part of it. ``perf_counter`` is only called when
    a checkpoint is actually reached, so the instrumentation cannot show up in
    the small-``n`` cells.
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    budget = max(ks)
    secs: dict[int, float] = {}
    bounds: dict[int, float] = {}

    t0 = time.perf_counter()
    X = np.unique(np.asarray(coords, dtype=np.float64), axis=0)
    n = X.shape[0]
    X = np.ascontiguousarray(X)
    if n <= KERNEL_SWITCH_N:
        sq = (X * X).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(D, 0.0, out=D)
        np.sqrt(D, out=D)
        np.fill_diagonal(D, 0.0)

        def ev(pi):
            return _one_tree_dense(D, pi, 0)
    else:
        def ev(pi):
            return _one_tree_coords(X, pi, 0)

    pi = np.zeros(n, dtype=np.float64)
    length, degrees = ev(pi)
    w0 = float(length)
    w_best = w0

    nxt = 0
    if ks[0] == 0:
        secs[0] = time.perf_counter() - t0
        bounds[0] = w_best
        nxt = 1

    def close_out(used: int):
        el = time.perf_counter() - t0
        for k in ks:
            if k not in secs:
                secs[k] = el
                bounds[k] = w_best

    t = w0 / (2.0 * n)
    if budget <= 0 or not (t > 0.0) or not np.isfinite(t):
        close_out(0)
        return secs, bounds

    period = INITIAL_PERIOD
    g_prev = np.zeros(n, dtype=np.float64)
    first_step = True
    used = 0
    while used < budget and t > 0.0:
        improved_anywhere = False
        improved_last = False
        steps = min(period, budget - used)
        for j in range(steps):
            g = degrees.astype(np.float64) - 2.0
            if not np.any(g):
                close_out(used)
                return secs, bounds
            direction = g if first_step else 0.7 * g + 0.3 * g_prev
            first_step = False
            g_prev = g
            pi = pi + (t * (period - j) / period) * direction
            length, degrees = ev(pi)
            w = float(length) - 2.0 * float(pi.sum())
            used += 1
            if w > w_best + _IMPROVE_RTOL * max(1.0, abs(w_best)):
                w_best = w
                improved_anywhere = True
                improved_last = j == steps - 1
            else:
                improved_last = False
            if nxt < len(ks) and used == ks[nxt]:
                secs[ks[nxt]] = time.perf_counter() - t0
                bounds[ks[nxt]] = w_best
                nxt += 1
        if improved_last:
            period *= 2
            t *= 2.0
        elif not improved_anywhere:
            t *= 0.5
            period = max(1, period // 2)

    close_out(used)
    return secs, bounds


# ---------------------------------------------------------------------------
# Comparator call sites -- copied from run_all_models_tsplib.py
# ---------------------------------------------------------------------------
def build_comparator(name: str):
    import classical_region_estimators as region_est
    from baselines_calibrated import AsymptoticMSTRatio, CalibratedMSTRatio

    if name == "GART_2.0":
        from lgbm_estimator_gart2 import TSP_GART2_Estimator
        m = TSP_GART2_Estimator(str(ROOT / "lgbm_model_v3"))
        return lambda X: m.estimate(X, 2, grid_size=0)["estimate"]
    if name == "MST_Only":
        m = region_est.ESTIMATORS["MST_Only"]
        return lambda X: m.estimate(X, 2, None)["estimate"]
    if name == "Asymptotic_MST":
        m = AsymptoticMSTRatio()
        return lambda X: m.estimate(X, 2, None)["estimate"]
    if name == "Calibrated_MST_dn":
        m = CalibratedMSTRatio("dn")
        return lambda X: m.estimate(X, 2, None)["estimate"]
    if name == "Hilbert":
        import tsp_utils_2 as academic
        return lambda X: academic.estimate_tsp_hilbert(X)[0]
    raise ValueError(f"unknown comparator {name}")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_comparator(name: str, repeats: int, insts) -> pd.DataFrame:
    fn = build_comparator(name)
    for _, X in insts[:5]:
        fn(X)                                    # warm, outside the clock
    rows = []
    for r in range(repeats):
        load = cpu_load() if r in (0, repeats - 1) else float("nan")
        for nm, X in insts:
            t0 = time.perf_counter()
            pred = fn(X)
            dt = time.perf_counter() - t0
            rows.append({"model": name, "k": -1, "instance": nm, "n": X.shape[0],
                         "repeat": r, "seconds": dt, "pred": pred, "cpu_load": load})
        print(f"  {name} repeat {r + 1}/{repeats}", flush=True)
    return pd.DataFrame(rows)


def run_ladder(kmax: int, repeats: int, insts, part_path: Path | None = None) -> pd.DataFrame:
    """Checkpointed after every instance, and resumable.

    A ``k=2000`` sweep over all 78 instances is over an hour of single-threaded
    work, and this session's supervisor reaps background jobs at roughly 40
    minutes. Rather than shorten the measurement to fit the reaper -- which
    would mean publishing a budget nobody actually timed -- each
    (repeat, instance) result is appended to a part file as soon as it exists,
    and a restart skips what is already there. The measurement is unchanged;
    only its restartability is.
    """
    ks = tuple(k for k in CHECKPOINTS if k <= kmax)
    done: set[tuple[int, str]] = set()
    if part_path is not None and part_path.exists():
        prev = pd.read_csv(part_path)
        done = set(zip(prev.repeat.astype(int), prev.instance.astype(str)))
        print(f"  resuming: {len(done)} (repeat, instance) pairs already timed",
              flush=True)

    for _, X in insts[:3]:
        timed_ladder(X, (0, 5))          # warm the JIT, outside the clock

    rows = []
    for r in range(repeats):
        load = cpu_load()
        t_rep = time.perf_counter()
        for nm, X in insts:
            if (r, nm) in done:
                continue
            secs, bounds = timed_ladder(X, ks)
            new = [{"model": f"HK_1Tree_{k}", "k": k, "instance": nm,
                    "n": X.shape[0], "repeat": r, "seconds": secs[k],
                    "pred": bounds[k], "cpu_load": load} for k in ks]
            rows.extend(new)
            if part_path is not None:
                pd.DataFrame(new).to_csv(part_path, mode="a", index=False,
                                         header=not part_path.exists())
        print(f"  HK ladder<=k{kmax} repeat {r + 1}/{repeats} "
              f"({time.perf_counter() - t_rep:.0f}s, load {load}%)", flush=True)

    if part_path is not None and part_path.exists():
        return pd.read_csv(part_path)
    return pd.DataFrame(rows)


def run_direct(k: int, repeats: int, insts) -> pd.DataFrame:
    """Call the shipped estimator object. Cross-checks the checkpointed clock."""
    est = HeldKarp1Tree(iterations=k)
    for _, X in insts[:3]:
        est.estimate(X, 2, None)
    rows = []
    for r in range(repeats):
        load = cpu_load()
        for nm, X in insts:
            t0 = time.perf_counter()
            res = est.estimate(X, 2, None)
            dt = time.perf_counter() - t0
            rows.append({"model": f"HK_1Tree_{k}_direct", "k": k, "instance": nm,
                         "n": X.shape[0], "repeat": r, "seconds": dt,
                         "pred": res["estimate"], "cpu_load": load})
        print(f"  HK direct k={k} repeat {r + 1}/{repeats} (load {load}%)", flush=True)
    return pd.DataFrame(rows)


def run_interleaved(kmax: int, repeats: int, insts) -> pd.DataFrame:
    """GART 2.0 and the HK ladder in ONE process, arms rotated between repeats.

    This is the arm that survives a loaded box. Solo-per-process gives absolute
    milliseconds but only on a quiet machine; here both arms take the same
    background load in the same repeat, so the *ratio* -- which is the whole of
    the middle-ground claim -- is not at the mercy of what else is running.
    """
    ks = tuple(k for k in CHECKPOINTS if k <= kmax)
    gart = build_comparator("GART_2.0")
    for _, X in insts[:3]:
        gart(X)
        timed_ladder(X, (0, 5))
    rows = []
    for r in range(repeats):
        load = cpu_load()
        arms = ["GART_2.0", "HK"] if r % 2 == 0 else ["HK", "GART_2.0"]
        for arm in arms:
            for nm, X in insts:
                if arm == "GART_2.0":
                    t0 = time.perf_counter()
                    pred = gart(X)
                    rows.append({"model": "GART_2.0", "k": -1, "instance": nm,
                                 "n": X.shape[0], "repeat": r,
                                 "seconds": time.perf_counter() - t0, "pred": pred,
                                 "cpu_load": load})
                else:
                    secs, bounds = timed_ladder(X, ks)
                    for k in ks:
                        rows.append({"model": f"HK_1Tree_{k}", "k": k, "instance": nm,
                                     "n": X.shape[0], "repeat": r, "seconds": secs[k],
                                     "pred": bounds[k], "cpu_load": load})
        print(f"  interleaved<=k{kmax} repeat {r + 1}/{repeats} (load {load}%)", flush=True)
    return pd.DataFrame(rows)


PLAN = [
    # (model, kwargs, tag) -- executed back to back, one subprocess each.
    ("comparators", {"repeats": 11}, "cmp_r11"),
    ("HK_ladder", {"kmax": 100, "repeats": 11}, "ladder_k100_r11"),
    ("HK_direct", {"k": 100, "repeats": 2}, "direct_k100_r2"),
    ("HK_interleaved", {"kmax": 100, "repeats": 5}, "interleaved_k100_r5"),
    ("HK_ladder", {"kmax": 500, "repeats": 3}, "ladder_k500_r3"),
    ("HK_ladder", {"kmax": 2000, "repeats": 1}, "ladder_k2000_r1"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--kmax", type=int, default=100)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--tag", default="adhoc")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--nmax", type=int, default=0,
                    help="drop instances with more than NMAX points before timing; "
                         "0 disables the cap")
    ap.add_argument("--nmin", type=int, default=0,
                    help="drop instances with fewer than NMIN points; used to time "
                         "the capped tail on its own")
    ap.add_argument("--mem-frac", type=float, default=0.5,
                    help="fraction of free physical RAM an instance's predicted "
                         "peak may use before it is dropped")
    args = ap.parse_args()

    if args.all:
        # Sequential subprocesses: solo means solo, so nothing overlaps.
        for model, kw, tag in PLAN:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--model", model,
                   "--tag", tag, "--repeats", str(kw["repeats"])]
            if "kmax" in kw:
                cmd += ["--kmax", str(kw["kmax"])]
            if "k" in kw:
                cmd += ["--k", str(kw["k"])]
            print(f"\n=== {tag}: {' '.join(cmd[2:])}", flush=True)
            t0 = time.perf_counter()
            subprocess.run(cmd, check=True)
            print(f"=== {tag} done in {time.perf_counter() - t0:.0f}s", flush=True)
        return

    print(f"CPU load before: {cpu_load()}%", flush=True)
    insts = load_instances()
    print(f"{len(insts)} EUC_2D instances loaded (parsing outside the clock)", flush=True)

    insts, dropped = screen_instances(insts, args.nmax, args.mem_frac, args.nmin)
    if dropped:
        pd.DataFrame(dropped).to_csv(
            OUT_DIR / f"hk1tree_timing_{args.tag}_excluded.csv", index=False)

    if args.model == "comparators":
        df = pd.concat([run_comparator(m, args.repeats, insts) for m in COMPARATORS])
    elif args.model == "HK_ladder":
        df = run_ladder(args.kmax, args.repeats, insts,
                        OUT_DIR / f"hk1tree_timingpart_{args.tag}.csv")
    elif args.model == "HK_direct":
        df = run_direct(args.k, args.repeats, insts)
    elif args.model == "HK_interleaved":
        df = run_interleaved(args.kmax, args.repeats, insts)
    else:
        df = run_comparator(args.model, args.repeats, insts)

    df["cpu_load_end"] = cpu_load()
    out = OUT_DIR / f"hk1tree_timing_{args.tag}.csv"
    df.to_csv(out, index=False)
    print(f"CPU load after: {df['cpu_load_end'].iloc[0]}%")
    print(f"Wrote {out} ({len(df)} rows)")

    med = (df.groupby(["model", "instance"]).seconds.median()
             .groupby("model").median() * 1000.0)
    print("\nmedian over instances of per-instance median-of-repeats (ms):")
    print(med.round(4).to_string())


if __name__ == "__main__":
    main()
