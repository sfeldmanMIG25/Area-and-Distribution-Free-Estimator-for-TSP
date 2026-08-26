"""Cost of the 1-tree ladder on the two benchmarks Section 5 never priced.

WHAT IS MISSING AND WHY THIS EXISTS
-----------------------------------
The manuscript promises, at L121 and L216, that the Held--Karp 1-tree bound is
"scored on all four benchmarks on cost and accuracy together" and that
Section~5 "reports them in full". Section~5 prices two of the four:

    benchmark              accuracy              cost
    ---------------------  --------------------  --------------------------
    TSPLIB EUC_2D          Table 3               Table 3 (solo, 11 repeats)
    multidimensional       Table 4               Table 4 (serial sample)
    2D diverse (N=2580)    hk1tree_allbench      ABSENT   <-- this script
    TSPLIB non-EUC_2D      hk1tree_allbench      ABSENT   <-- this script

Accuracy on all four already exists and is gated
(``hk1tree_all_benchmarks.py --gates``). This script measures the two missing
cost columns, on the same protocol as the published one, so the two new cells
can be printed beside the two old ones.

PROTOCOL
--------
Reproduces ``gart2_timing_bank.json -> tsplib_by_size_time_one_protocol``, tag
``serial_solo_median11_quiet_2026-08-11``, the protocol of Table 3:

* **one estimator per process.** This file is invoked once per (corpus, arm).
  Arms are never co-measured: co-measurement was shown elsewhere in this
  project to move small-``n`` medians by up to 1.9x in GART 2.0's favour.
* **single thread of control**, no pool, OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1.
* **11 repeats**; the published statistic is the median over repeats, then the
  median over instances of that, with the relative IQR retained.
* **warm outside the clock**: numba JIT and each model's first predict are
  exercised on a handful of instances before the first timed call.
* **parsing outside the clock**: every instance is parsed and materialised
  before any repeat starts, and the timed region begins at the estimator's own
  entry point.
* box load is sampled from ``GetSystemTimes`` (see :func:`box_load_pct`) and
  stamped on every row.

PRIMARY LADDER: CHECKPOINTED, AS IN TABLE 3
-------------------------------------------
Table 3's ladder is *checkpointed*: one ascent to ``K`` with the clock read off
at each rung, which amortises the setup -- dedup, matrix build, and for Polyak
the constructive tour -- over the whole ladder. The two new cells are measured
the same way, because the point of them is to be printable beside Table 3.

The ``vj`` and ``polyak`` arms are the control for that choice: they call the
shipped single-budget entry points ``one_tree_bound(X, k)``,
``polyak_bound(X, k)`` and their matrix twins once per rung, so nothing is
amortised and each cell is the full cost of asking for a ``k``-budget bound.
That costs ``sum(BUDGETS)`` iterations per instance against ``max(BUDGETS)``
for the ladder, so on the 2{,}580-instance corpus it is run on a size-stratified
``--sample`` rather than the whole of it; on the 31-instance non-Euclidean
corpus it is run on all of them. Table 3 has one such cross-check, at one
budget on one corpus; these cover every budget.

ARMS
----
    gart2       the estimator, at its own benchmark's call site
    vj_ckpt     checkpointed Volgenant--Jonker ladder   -- PRIMARY
    polyak_ckpt checkpointed Polyak ladder              -- PRIMARY
    vj          one shipped VJ call per rung            -- amortisation control
    polyak      one shipped Polyak call per rung        -- amortisation control

The Polyak arms pay for the constructive tour (nearest neighbour improved by
2-opt) on every call, because a caller cannot skip it: the step length is sized
from the gap to that tour.

Usage
-----
    ...python.exe paper_tooling/hk1tree_cost_allbench.py --corpus 2d --arm gart2
    ...python.exe paper_tooling/hk1tree_cost_allbench.py --corpus 2d --arm vj_ckpt
    ...python.exe paper_tooling/hk1tree_cost_allbench.py --corpus 2d --arm vj \
        --sample 300 --repeats 5           # amortisation control
    ...python.exe paper_tooling/hk1tree_cost_allbench.py --corpus noneuc --arm polyak
    ...python.exe paper_tooling/hk1tree_cost_allbench.py --corpus tsplib --arm gart2 \
        --tag drift_A                      # drift control vs the published column

Output: paper_tooling/hk1tree_costtime_<tag>.csv
        paper_tooling/hk1tree_costpart_<tag>.csv   (resume file, per instance)
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import ctypes  # noqa: E402
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from ctypes import wintypes  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tsplib_benchmark", ROOT / "lgbm_model_v3",
           ROOT / "paper_tooling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# See hk1tree_all_benchmarks.py for why this shim exists: a bare ``coverage/``
# directory at the repo root becomes an importable namespace package once ROOT
# joins sys.path, and numba's ImportError guard then does not fire.
if "coverage" not in sys.modules:
    _cov = importlib.util.find_spec("coverage")
    if _cov is not None and _cov.origin is None:
        sys.modules["coverage"] = None  # type: ignore[assignment]

OUT_DIR = ROOT / "paper_tooling"
TSPLIB_DIR = ROOT / "tsplib_benchmark" / "instances"
TSPLIB_RESULTS = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"
BASE_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv"

#: The rungs Table 3 and Table 4 print. Cost is measured on exactly these so a
#: new cell can sit in the same row of the same table shape.
BUDGETS: tuple[int, ...] = (0, 10, 25, 50, 100, 200, 500)

#: Instances used to warm the JIT and the first predict, outside the clock.
WARM_N = 5


# ---------------------------------------------------------------------------
# Box load
# ---------------------------------------------------------------------------
class _FT(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD)]


def _ft(v: _FT) -> int:
    return (v.dwHighDateTime << 32) | v.dwLowDateTime


def _system_times() -> tuple[int, int, int]:
    idle, kern, user = _FT(), _FT(), _FT()
    if not ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
            ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
        raise OSError("GetSystemTimes failed")
    return _ft(idle), _ft(kern), _ft(user)


class BoxLoad:
    """Whole-box CPU load from ``GetSystemTimes``, sampled between calls.

    The WMI providers on this machine are not usable for this.
    ``Win32_Processor.LoadPercentage`` is already documented as unreliable here
    (``hk1tree_solo_cost.py`` -> ``load_sensor_warning``), and
    ``Win32_PerfFormattedData_PerfOS_Processor`` disagrees with the kernel's own
    accounting by a factor of three on this box: it reported 71% busy in the
    same window in which the sum over every running process's user+kernel time
    delta accounted for 25.6% of 20 logical processors, and its per-core values
    are quantised to 1/16. ``GetSystemTimes`` is the accounting those providers
    are derived from, read directly and differenced over a stated wall
    interval, so it cannot disagree with itself.
    """

    def __init__(self) -> None:
        self._prev = _system_times()

    def sample(self) -> float:
        """Percent busy since the previous sample. First call spans the run."""
        cur = _system_times()
        di = cur[0] - self._prev[0]
        dk = cur[1] - self._prev[1]
        du = cur[2] - self._prev[2]
        self._prev = cur
        tot = dk + du            # kernel time already includes idle
        if tot <= 0:
            return float("nan")
        return 100.0 * (1.0 - di / tot)


def spot_load(seconds: float = 1.0) -> float:
    b = BoxLoad()
    time.sleep(seconds)
    return b.sample()


def assert_solo() -> int:
    """Refuse to start if another copy of this harness is already timing.

    "One estimator per process" is only true if one process is running, and
    this project has already lost one 1-tree timing pass to a reaped background
    wrapper that left an orphan alive to race the run that replaced it
    (``hk1tree_solo_cost.py`` -> ``co_measurement_incident``). It happened
    again in this session: a driver shell outlived the job it was stopped with
    and started the next arm, so two arms measured each other for 45 minutes.
    Asserting it here rather than in the driver puts the check where it cannot
    be bypassed by a driver bug.
    """
    me = os.getpid()
    q = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*hk1tree_cost_allbench*' } | "
         "ForEach-Object { $_.ProcessId }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", q],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:                      # pragma: no cover
        print(f"  WARNING: solo check could not run ({exc}); proceeding")
        return 0
    others = [int(x) for x in out.split() if x.strip().isdigit() and int(x) != me]
    if others:
        raise SystemExit(
            f"REFUSING TO START: {len(others)} other hk1tree_cost_allbench "
            f"process(es) already running (pids {others}). Solo means solo; "
            f"kill them, delete the part file of whatever they were writing, "
            f"and start again.")
    print("  solo check: no other hk1tree_cost_allbench process running",
          flush=True)
    return len(others)


# ---------------------------------------------------------------------------
# Corpora -- parsed once, outside every clock
# ---------------------------------------------------------------------------
def load_2d() -> list[dict]:
    """The 2{,}580 diverse 2D instances, as ``run_benchmark_2D_all`` hands them.

    ``coordinates`` is passed to both arms byte-identically; ``grid_size`` and
    ``dimension`` are the runner's own, because GART 2.0's benchmark call site
    is ``estimate(coords, dimension, grid_size)`` and not the TSPLIB harness's
    ``grid_size=0``.
    """
    from tsp_utils import parse_tsp_instance

    base = pd.read_csv(BASE_2D)
    out = []
    for r in base.itertuples(index=False):
        coords = parse_tsp_instance(Path(r.file_path)).coordinates
        out.append({"instance": str(r.instance), "coords": np.asarray(coords),
                    "n": int(r.n_customers), "d": int(r.dimension),
                    "grid_size": r.grid_size, "true_cost": float(r.true_cost)})
    if len(out) != 2580:
        raise ValueError(f"expected 2580 2D instances, got {len(out)}")
    return out


def load_noneuc() -> list[dict]:
    """The non-EUC_2D TSPLIB instances the bound is scored on.

    Identical selection to ``hk1tree_all_benchmarks.tasks_noneuc`` -- reused
    rather than restated, so the cost corpus cannot drift from the accuracy
    corpus -- with the two extras the GART 2.0 arm needs: the raw coordinates
    for the natively Euclidean types, and the edge-weight type that decides
    which of the runner's two paths an instance takes.
    """
    from hk1tree_all_benchmarks import tasks_noneuc
    from tsplib_parser import parse_tsplib_file

    out = tasks_noneuc()
    for t in out:
        info = parse_tsplib_file(str(TSPLIB_DIR / f"{t['instance']}.tsp"))
        t["is_native"] = bool(info["is_native_euclidean"]
                              and info["raw_coords"] is not None)
        t["raw_coords"] = (info["raw_coords"].astype(np.float32)
                           if t["is_native"] else None)
        # The runner's hybrid path reads ``info["distance_matrix"]``; for the
        # CEIL_2D pair that is None and the native path is taken instead.
        t["parser_matrix"] = info["distance_matrix"]
    return out


def load_tsplib() -> list[dict]:
    """TSPLIB EUC_2D. Only the drift control uses this corpus."""
    from tsplib_parser import parse_tsplib_file

    res = pd.read_csv(TSPLIB_RESULTS)
    names = sorted(res[(res.model == "GART_2.0")
                       & (res.edge_weight_type == "EUC_2D")].instance.astype(str))
    out = []
    for nm in names:
        info = parse_tsplib_file(str(TSPLIB_DIR / f"{nm}.tsp"))
        X = info["raw_coords"].astype(np.float32)
        out.append({"instance": nm, "coords": X, "n": int(X.shape[0]), "d": 2,
                    "grid_size": 0, "is_native": True, "raw_coords": X})
    return out


LOADERS = {"2d": load_2d, "noneuc": load_noneuc, "tsplib": load_tsplib}


# ---------------------------------------------------------------------------
# Arms. Each returns [(model, k, seconds, prediction), ...] for one instance.
# ---------------------------------------------------------------------------
def make_gart2(corpus: str):
    """GART 2.0 at the call site the corpus's own benchmark runner uses.

    On ``2d`` and ``tsplib`` that is ``estimate(coords, d, grid_size)``. On
    ``noneuc`` the runner takes one of two paths, and the timed region is the
    whole of whichever it takes: for the natively Euclidean pair the direct
    call, and for the rest classical MDS *plus* the hybrid feature build. The
    published ``prediction_time_s`` column excludes the MDS because the runner
    computes it once and shares it across every model in the loop; a caller
    holding only a distance matrix cannot skip it, and the bound it is being
    priced against reads that same matrix directly, so charging it is the only
    comparison that means anything. Both readings are recorded and the
    estimate-only one is kept beside it.
    """
    from lgbm_estimator_gart2 import TSP_GART2_Estimator

    model = TSP_GART2_Estimator(str(ROOT / "lgbm_model_v3"))

    if corpus != "noneuc":
        def run(t: dict):
            t0 = time.perf_counter()
            res = model.estimate(t["coords"], t["d"], t["grid_size"])
            dt = time.perf_counter() - t0
            return [("GART_2.0", -1, dt, float(res["estimate"]),
                     res.get("status", "ok"), dt)]
        return run

    from classical_mds import classical_mds
    from run_all_models_tsplib import MAX_MDS_DIM, _hybrid_estimate_generic

    def run(t: dict):
        if t["is_native"]:
            X = t["raw_coords"]
            t0 = time.perf_counter()
            res = model.estimate(X, X.shape[1], grid_size=0)
            dt = time.perf_counter() - t0
            return [("GART_2.0", -1, dt, float(res["estimate"]),
                     res.get("status", "ok"), dt)]
        D = t["parser_matrix"]
        t0 = time.perf_counter()
        Xe, _eigs, _raw = classical_mds(D, max_dim=MAX_MDS_DIM)
        t_mds = time.perf_counter() - t0
        t1 = time.perf_counter()
        res = _hybrid_estimate_generic(model, D, Xe, Xe.shape[1])
        t_est = time.perf_counter() - t1
        return [("GART_2.0", -1, t_mds + t_est, float(res["estimate"]),
                 res.get("status", "ok"), t_est)]

    return run


def make_hk(corpus: str, ascent: str):
    """One shipped single-budget call per rung. Nothing is amortised.

    The VJ arm goes through ``HeldKarp1Tree.estimate``, the estimator object
    the benchmark runners instantiate, and *not* through the bare
    ``one_tree_bound``. The two are not the same call on a coordinate corpus:
    the estimator deduplicates first and the bare function does not, and 269 of
    the 2,580 2D instances carry coincident points, so timing the bare function
    would have priced a different instance than the one the accuracy sweep
    scored. The gate in ``hk1tree_cost_analyze.py`` caught exactly that.
    """
    from held_karp_1tree import HeldKarp1Tree
    from hk1tree_polyak import polyak_bound, polyak_bound_from_matrix

    from_matrix = corpus == "noneuc"

    if ascent == "vj":
        ests = {k: HeldKarp1Tree(iterations=k) for k in BUDGETS}

        def call(x, k):
            e = ests[k]
            r = e.estimate_from_matrix(x) if from_matrix else e.estimate(x)
            if r["status"] != "ok":
                return float("nan"), str(r["status"])
            return float(r["estimate"]), "ok"
    else:
        fn = polyak_bound_from_matrix if from_matrix else polyak_bound

        def call(x, k):
            r = fn(x, k)
            if r["status"] != "ok":
                return float("nan"), str(r["status"])
            return float(r["bound"]), "ok"

    key = "matrix" if from_matrix else "coords"

    def run(t: dict):
        x = t[key]
        rows = []
        for k in BUDGETS:
            t0 = time.perf_counter()
            b, st = call(x, k)
            rows.append((f"HK_1Tree_{ascent}_{k}", k,
                         time.perf_counter() - t0, b, st, float("nan")))
        return rows

    return run


#: The clock cannot be read from inside ``_ascend_checkpointed`` or
#: ``ascend_polyak`` -- neither takes an observer, and adding one would edit
#: the functions that produced every accuracy number in Section 5. Counting
#: evaluations is a faithful substitute: the m-th call to ``evaluate`` returns
#: the state at ``used = m - 1``, and both ascents read their checkpoint at
#: that ``used`` immediately afterwards, so the clock is read within a handful
#: of float operations of where the snapshot is taken.
def _clocked(evaluate, t0: float, marks: dict[int, float]):
    ks = set(BUDGETS)
    calls = [0]

    def ev(pi):
        r = evaluate(pi)
        u = calls[0]
        calls[0] += 1
        if u in ks:
            marks.setdefault(u, time.perf_counter() - t0)
        return r

    return ev


def make_ckpt(corpus: str, ascent: str):
    """The checkpointed ladder: one ascent, clock read off at each rung.

    Table 3's own protocol, and the primary measurement here. Both ascents
    have the prefix property -- neither schedule sees ``n`` or the budget -- so
    the incumbent read off at ``k`` is exactly ``bound(k)``, and
    ``hk1tree_cost_analyze.py`` gates that against the accuracy sweep rather
    than trusting it.
    """
    from held_karp_1tree import _one_tree_dense
    from hk1tree_frontier_accuracy import _ascend_checkpointed
    from hk1tree_polyak import GAMMA_0, ascend_polyak, constructive_upper_bound

    del GAMMA_0
    from_matrix = corpus == "noneuc"
    tag = "vjckpt" if ascent == "vj" else "pkckpt"

    def build(t: dict):
        """``(D, n)``. Same two paths the shipped entry points take."""
        if from_matrix:
            D = np.ascontiguousarray(t["matrix"], dtype=np.float64)
            return D, D.shape[0]
        X = np.ascontiguousarray(np.unique(
            np.asarray(t["coords"], dtype=np.float64), axis=0))
        sq = (X * X).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(D, 0.0, out=D)
        np.sqrt(D, out=D)
        np.fill_diagonal(D, 0.0)
        return D, X.shape[0]

    def run(t: dict):
        marks: dict[int, float] = {}
        t0 = time.perf_counter()
        D, n = build(t)
        ev = _clocked(lambda pi: _one_tree_dense(D, pi, 0), t0, marks)
        if ascent == "vj":
            bounds, _u, _o = _ascend_checkpointed(ev, n, BUDGETS)
        else:
            ub, _sweeps = constructive_upper_bound(D, two_opt=True)
            _res, bounds = ascend_polyak(ev, n, max(BUDGETS), ub, BUDGETS)
        el = time.perf_counter() - t0
        return [(f"HK_1Tree_{tag}_{k}", k, marks.get(k, el), bounds[k], "ok",
                 float("nan")) for k in BUDGETS]

    return run


ARMS = {"gart2": make_gart2,
        "vj": lambda c: make_hk(c, "vj"),
        "polyak": lambda c: make_hk(c, "polyak"),
        "vj_ckpt": lambda c: make_ckpt(c, "vj"),
        "polyak_ckpt": lambda c: make_ckpt(c, "polyak")}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def stratified_sample(insts: list[dict], size: int, seed: int = 20260812
                      ) -> list[dict]:
    """A size-stratified subsample, for the arms too dear to run on everything.

    Stratified on the manuscript's own 2D size buckets and drawn without
    replacement, so the sample spans the whole range the ladder's cost depends
    on instead of being dominated by the many small instances a uniform draw
    would return.
    """
    edges = ((5, 10), (11, 50), (51, 100), (101, 500), (501, 10 ** 9))
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    per = max(1, size // len(edges))
    for lo, hi in edges:
        pool = [t for t in insts if lo <= t["n"] <= hi]
        if not pool:
            continue
        take = min(per, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        out.extend(pool[i] for i in sorted(idx))
    return sorted(out, key=lambda t: t["instance"])


def run(tag: str, corpus: str, arm: str, repeats: int, insts: list[dict],
        nmax: int, sample: int) -> pd.DataFrame:
    part = OUT_DIR / f"hk1tree_costpart_{tag}.csv"
    done: set[tuple[int, str]] = set()
    if part.exists():
        prev = pd.read_csv(part)
        done = set(zip(prev.repeat.astype(int), prev.instance.astype(str)))
        print(f"  resuming: {len(done)} (repeat, instance) pairs already timed",
              flush=True)

    if nmax:
        kept = [t for t in insts if t["n"] <= nmax]
        print(f"  --nmax {nmax}: {len(kept)}/{len(insts)} instances kept")
        insts = kept
    if sample and sample < len(insts):
        insts = stratified_sample(insts, sample)
        print(f"  --sample: {len(insts)} instances, size-stratified")

    fn = ARMS[arm](corpus)

    warm = sorted(insts, key=lambda t: t["n"])[:WARM_N]
    t0 = time.perf_counter()
    for t in warm:
        fn(t)
    print(f"  warm-up on {len(warm)} smallest instances: "
          f"{time.perf_counter() - t0:.1f}s (outside the clock)", flush=True)

    load = BoxLoad()
    load.sample()
    for r in range(repeats):
        t_rep = time.perf_counter()
        wrote = 0
        # Appended per instance, not per repeat: one repeat of the 2D ladder is
        # tens of minutes and this session's supervisor reaps a background job
        # well inside that, so a per-repeat flush would discard everything the
        # reaped repeat had already measured.
        for t in insts:
            if (r, t["instance"]) in done:
                continue
            new = [{"arm": arm, "corpus": corpus, "model": model, "k": k,
                    "instance": t["instance"], "n": t["n"], "repeat": r,
                    "seconds": sec, "pred": pred, "status": status,
                    "seconds_estimate_only": sec_est, "box_load_pct": float("nan")}
                   for model, k, sec, pred, status, sec_est in fn(t)]
            pd.DataFrame(new).to_csv(part, mode="a", index=False,
                                     header=not part.exists())
            wrote += 1
        lp = load.sample()
        print(f"  {tag} repeat {r + 1}/{repeats}: "
              f"{time.perf_counter() - t_rep:.0f}s, {wrote} instances, "
              f"box load {lp:.1f}%", flush=True)
        if wrote:
            # One load reading per repeat, stamped on the rows it covers.
            d = pd.read_csv(part)
            d.loc[(d.repeat == r) & d.box_load_pct.isna(), "box_load_pct"] = lp
            d.to_csv(part, index=False)

    out = pd.read_csv(part)
    dest = OUT_DIR / f"hk1tree_costtime_{tag}.csv"
    out.to_csv(dest, index=False)
    print(f"\nWrote {dest} ({len(out)} rows)")

    med = (out.groupby(["model", "instance"]).seconds.median()
              .groupby("model").median() * 1000.0)
    print("median over instances of per-instance median-of-repeats (ms):")
    print(med.round(4).to_string())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, choices=sorted(LOADERS))
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--nmax", type=int, default=0,
                    help="drop instances above this n")
    ap.add_argument("--sample", type=int, default=0,
                    help="size-stratified subsample of this many instances; "
                         "0 uses the whole corpus")
    args = ap.parse_args()

    tag = args.tag or f"{args.corpus}_{args.arm}"
    assert_solo()
    print(f"box load before: {spot_load():.1f}%", flush=True)
    t0 = time.perf_counter()
    insts = LOADERS[args.corpus]()
    print(f"{len(insts)} {args.corpus} instances parsed in "
          f"{time.perf_counter() - t0:.0f}s (outside the clock)", flush=True)

    run(tag, args.corpus, args.arm, args.repeats, insts, args.nmax, args.sample)
    print(f"box load after: {spot_load():.1f}%")


if __name__ == "__main__":
    main()
