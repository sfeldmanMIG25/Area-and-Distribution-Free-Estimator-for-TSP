"""Cost of the *Polyak* 1-tree ladder on TSPLIB EUC_2D, solo protocol.

Why this exists
---------------
The cost axis prints a 1-tree ladder measured with the Volgenant-Jonker /
Helsgaun ascent (``hk1tree_frontier_bank.json ->
cost_tsplib_euc2d_solo_2026_08_11``). The ND arm reports the *Polyak* ascent
instead, because it is the stronger of the two there
(``frontier_manuscript_bank.json -> _sources.nd``). On TSPLIB the Polyak arm has
accuracy numbers but no protocol-matched cost: ``hk1tree_polyak_tsplib.csv``
records one ascent time per instance, replicated across its ``k`` rows, because
that sweep ran a single ascent to ``k=2000`` and read the *bounds* off at each
checkpoint. Those columns are an accuracy artefact and are not a per-``k`` cost.

So the axis currently mixes ascents across corpora and has no Polyak cost cell
on TSPLIB at all. This produces one, under the protocol the rest of the axis
uses.

Method, and why it needs no second implementation of the ascent
---------------------------------------------------------------
``ascend_polyak`` calls its ``evaluate`` callable exactly once before the loop
and exactly once per iteration. Wrapping that callable in a counter therefore
recovers the iteration index without touching the ascent, and a
``perf_counter`` read on the calls that land on a checkpoint gives the elapsed
time at each ladder budget from a *single* ascent to ``K``. There is no
re-implemented loop here to drift from the real one, which is the failure mode
the equivalent V&J harness has to guard against with a separate ``--mode
direct`` cross-check.

The clock starts before deduplication, so the reading at ``k`` is the whole
cost of asking for a ``k``-budget Polyak bound from coordinates: dedup, the
dense matrix, the constructive nearest-neighbour + 2-opt upper bound, and ``k``
ascent steps. The upper bound is part of that cost and has no counterpart in
the V&J arm -- Polyak's step length is scaled by the gap to a feasible tour, so
it cannot start without one. That difference is the point of the comparison and
is not netted out.

Protocol
--------
Matches ``serial_solo_median11_quiet_2026-08-11``: one estimator per process,
no pool, OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1, warm-up outside the clock, median
over repeats, median over instances of that. The corpus is capped at
``n <= 16384`` so the matched subset is identical to the V&J arm's 77.

    python paper_tooling/polyak_tsplib_timing.py --kmax 100 --repeats 11
    python paper_tooling/polyak_tsplib_timing.py --kmax 500 --repeats 3
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tsplib_benchmark", ROOT / "paper_tooling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hk1tree_polyak import (  # noqa: E402
    _kernel,
    ascend_polyak,
    constructive_upper_bound,
)
from quiet_box import LoadSampler, observe  # noqa: E402

TSPLIB_DIR = ROOT / "tsplib_benchmark" / "instances"
TSPLIB_RESULTS = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
OUT_DIR = ROOT / "paper_tooling"

CHECKPOINTS = (0, 10, 25, 50, 100, 200, 500, 1000, 2000)
#: Same cap as the V&J solo cost pass, so the matched subset is the same 77.
NMAX = 16384


def load_instances(nmax: int) -> list[tuple[str, np.ndarray]]:
    from tsplib_parser import parse_tsplib_file

    res = pd.read_csv(TSPLIB_RESULTS, usecols=["instance", "model", "edge_weight_type"])
    names = sorted(res[(res.model == "GART_2.0") &
                       (res.edge_weight_type == "EUC_2D")].instance.astype(str).unique())
    if len(names) != 78:
        raise ValueError(f"expected 78 EUC_2D instances, got {len(names)}")
    out = []
    for nm in names:
        info = parse_tsplib_file(str(TSPLIB_DIR / f"{nm}.tsp"))
        coords = info["raw_coords"].astype(np.float32)
        if coords.shape[0] > nmax:
            continue
        out.append((nm, coords))
    return out


class _TimedEvaluate:
    """Count ``ascend_polyak``'s evaluations and stamp the checkpoint times.

    ``ascend_polyak`` evaluates once at ``pi = 0`` (iteration 0) and once per
    ascent step, so call number ``1 + k`` is the state at ``used == k``.
    """

    def __init__(self, ev, ks: tuple[int, ...], t0: float) -> None:
        self._ev = ev
        self._ks = set(int(k) for k in ks)
        self._t0 = t0
        self.times: dict[int, float] = {}
        self._calls = 0

    def __call__(self, pi):
        out = self._ev(pi)
        self._calls += 1
        used = self._calls - 1
        if used in self._ks:
            self.times[used] = time.perf_counter() - self._t0
        return out


def timed_polyak_ladder(coords: np.ndarray, ks: tuple[int, ...]
                        ) -> tuple[dict[int, float], dict[int, float], dict]:
    """``({k: seconds}, {k: bound}, meta)`` for one instance, from one ascent."""
    ks = tuple(sorted(set(int(k) for k in ks)))
    budget = max(ks)

    t0 = time.perf_counter()
    evaluate, D, n, backend = _kernel(coords)
    if n < 3:
        raise ValueError(f"degenerate instance, n_unique={n}")
    if D is None:
        raise NotImplementedError("the constructive upper bound needs the dense matrix")
    ub, sweeps = constructive_upper_bound(D, two_opt=True)

    timer = _TimedEvaluate(evaluate, ks, t0)
    res, bounds = ascend_polyak(timer, n, budget, ub, ks, smoothing=0.0)
    elapsed = time.perf_counter() - t0

    # Checkpoints past an early stop cost exactly what the run cost.
    secs = {k: timer.times.get(k, elapsed) for k in ks}
    meta = {
        "n_unique": n,
        "backend": backend,
        "upper_bound": float(ub),
        "two_opt_sweeps": int(sweeps),
        "iterations_used": int(res.iterations_used),
        "stopped_reason": res.stopped_reason,
        "is_optimal": bool(res.is_optimal),
        "total_seconds": elapsed,
    }
    return secs, bounds, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmax", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--nmax", type=int, default=NMAX)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ks = tuple(k for k in CHECKPOINTS if k <= args.kmax)
    tag = args.tag or f"k{args.kmax}_r{args.repeats}"
    part = OUT_DIR / f"polyak_tsplib_timingpart_{tag}.csv"
    out = OUT_DIR / f"polyak_tsplib_timing_{tag}.csv"

    insts = load_instances(args.nmax)
    print(f"[polyak-cost] {len(insts)} instances (n <= {args.nmax}), "
          f"ks={ks}, repeats={args.repeats}", flush=True)

    pre = observe(interval_s=1.0, settle_s=1.5)
    print(f"  box before batch: {pre.busy_pct:.1f}% busy, "
          f"foreign solvers: {pre.foreign_solver_procs or 'none'}", flush=True)

    # Warm-up outside the clock: numba JIT on both kernels and the 2-opt.
    warm = insts[0][1]
    timed_polyak_ladder(warm, (0, 10))
    print("  warmed", flush=True)

    rows: list[dict] = []
    with LoadSampler(period_s=5.0) as sampler:
        for r in range(args.repeats):
            for nm, coords in insts:
                secs, bounds, meta = timed_polyak_ladder(coords, ks)
                for k in ks:
                    rows.append({
                        "repeat": r, "instance": nm, "n": int(coords.shape[0]),
                        "k": k, "seconds": secs[k], "bound": bounds.get(k),
                        **{kk: meta[kk] for kk in
                           ("n_unique", "backend", "iterations_used",
                            "stopped_reason", "two_opt_sweeps")},
                    })
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  repeat {r + 1}/{args.repeats} done "
                  f"({len(rows)} rows)", flush=True)
    load = sampler.summary()

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    per_inst = df.groupby(["instance", "k"]).seconds.median().reset_index()
    corpus = per_inst.groupby("k").seconds.median() * 1000.0
    summary = {
        "tag": tag,
        "protocol": {
            "matches": "serial_solo_median11_quiet_2026-08-11",
            "process": ("one estimator per process, single thread of control, no "
                        "pool, OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1"),
            "repeats": args.repeats,
            "statistic": "median over repeats; median over instances of that",
            "warm_up": "numba JIT warmed on one instance outside the clock",
            "timed_region": ("clock starts before dedup: dedup + dense matrix + "
                             "constructive NN/2-opt upper bound + k Polyak steps"),
            "cap": f"n <= {args.nmax}, matching the V&J solo cost pass",
            "box_during_pct": load,
        },
        "N_instances": int(per_inst.instance.nunique()),
        "polyak_ms_by_k": {str(int(k)): float(v) for k, v in corpus.items()},
    }
    (OUT_DIR / f"polyak_tsplib_timing_{tag}.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["polyak_ms_by_k"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
