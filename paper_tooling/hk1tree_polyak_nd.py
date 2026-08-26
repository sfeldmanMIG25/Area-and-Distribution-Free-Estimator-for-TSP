"""ND sweep of the Polyak Held--Karp 1-tree bound, and the probes that decide
whether its accuracy there is real.

MODES
-----
``--mode ub``       Upper-bound sensitivity. Runs the same ascent from three
                    step-scaling sources on the same instances: the
                    nearest-neighbour tour, the NN tour improved by 2-opt, and
                    -- as a *disqualified diagnostic only* -- the released
                    optimal cost. If the three converge to the same bound the
                    constructive tour is not load-bearing and the sweep's
                    unsupervisedness is not doing quiet work; if they do not,
                    the constructive number is the only reportable one.
``--mode probe``    Stratified sample, deep ladder to k = 10000, plus the
                    label's own provenance (Concorde-exact against LKH) so the
                    gap can be read against exact optima alone.
``--mode sweep``    The full 16,920-instance ND test split at the published
                    ladder. One row per (instance, k).
``--mode timing``   Serial, single-thread cost of the ascent against GART 2.0
                    re-measured in the same session, on a stratified sample.

Every scoring path builds its upper bound with
``hk1tree_polyak.constructive_upper_bound`` from coordinates only. The released
optimal cost enters exactly one function, :func:`ub_sensitivity`, and every row
it produces is stamped ``ub_source = optimum`` and reported as disqualified.
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "paper_tooling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hk1tree_polyak import polyak_bound  # noqa: E402

OUT = ROOT / "paper_tooling"
BASE_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
PROVENANCE = OUT / "nd_label_provenance.csv"

#: Same ladder as ``hk1tree_frontier_accuracy.CHECKPOINTS`` so the Polyak column
#: drops straight into the published table beside the V&J one.
CHECKPOINTS: tuple[int, ...] = (0, 10, 25, 50, 100, 200, 500, 1000, 2000)

#: The probe runs past the published ladder because the whole point of the
#: Polyak step is that it does not stall, so "where does it stop paying" is a
#: question the published ladder can no longer answer by construction.
DEEP_CHECKPOINTS: tuple[int, ...] = CHECKPOINTS + (5000, 10000)

#: pandas writes floats at 16 significant digits by default, which loses the
#: last ULP on some float64 bounds; %.17g round-trips exactly. Note the *reader*
#: needs fixing too -- pandas' default float parser is not correctly rounded, so
#: a value written exactly still reads back 1 ULP out unless the caller passes
#: float_precision="round_trip". Both ends matter, and both are 2e-16 relative:
#: twelve orders below the smallest figure this study reports, but an artifact a
#: paper cites should read back as written.
FLOAT_FMT = "%.17g"

SEED_PROBE = 20260818
SEED_UB = 20260819
SEED_TIMING = 20260820

_BIN_HEADER = struct.Struct("IIII")


def _load_bin(path: Path, n: int, d: int, grid: int) -> np.ndarray:
    blob = path.read_bytes()
    b_n, b_d, b_grid, dist_len = _BIN_HEADER.unpack_from(blob, 0)
    offset = _BIN_HEADER.size + dist_len
    if (b_n, b_d, b_grid) != (int(n), int(d), int(grid)):
        raise ValueError(f"{path.name}: stale .bin header")
    if len(blob) - offset != b_n * b_d * 4:
        raise ValueError(f"{path.name}: bad .bin body length")
    return np.frombuffer(blob, dtype=np.float32, count=b_n * b_d,
                         offset=offset).reshape(b_n, b_d).copy()


def load_base() -> pd.DataFrame:
    b = pd.read_csv(BASE_ND)
    if PROVENANCE.exists():
        prov = pd.read_csv(PROVENANCE)[["instance", "solver", "gap"]]
        b = b.merge(prov, on="instance", how="left")
    return b


def tasks_from(rows: pd.DataFrame) -> list[dict]:
    out = []
    for r in rows.itertuples(index=False):
        p = Path(r.file_path).with_suffix(".bin")
        out.append({"instance": r.instance,
                    "coords": _load_bin(p, r.n_customers, r.dimension, r.grid_size),
                    "true_cost": float(r.true_cost), "n": int(r.n_customers),
                    "d": int(r.dimension), "grid_size": int(r.grid_size),
                    "solver": getattr(r, "solver", None)})
    return out


def stratified(base: pd.DataFrame, per_cell: int, seed: int,
               n_lo: int = 0, n_hi: int = 10 ** 9) -> pd.DataFrame:
    """``per_cell`` instances per dimension, drawn inside ``[n_lo, n_hi]``."""
    b = base[(base.n_customers >= n_lo) & (base.n_customers <= n_hi)]
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in b.groupby("dimension"):
        idx = rng.choice(len(g), size=min(per_cell, len(g)), replace=False)
        parts.append(g.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def _run_sweep_one(task: dict) -> list[dict]:
    r = polyak_bound(task["coords"], max(CHECKPOINTS), checkpoints=CHECKPOINTS)
    common = {"instance": task["instance"], "n": task["n"], "d": task["d"],
              "true_cost": task["true_cost"], "solver": task["solver"]}
    if r["status"] != "ok":
        return [dict(common, k=k, bound=float("nan"), status=r["status"])
                for k in CHECKPOINTS]
    return [dict(common, k=k, bound=r["bounds"][k], status="ok",
                 n_unique=r["n_unique"], backend=r["backend"],
                 upper_bound=r["upper_bound"], ub_source=r["ub_source"],
                 two_opt_sweeps=r["two_opt_sweeps"],
                 iterations_used=r["iterations_used"], is_optimal=r["is_optimal"],
                 stopped_reason=r["stopped_reason"],
                 ub_seconds=r["ub_seconds"], matrix_seconds=r["matrix_seconds"],
                 ascent_seconds=r["ascent_seconds"])
            for k in CHECKPOINTS]


def _run_probe_one(task: dict) -> list[dict]:
    r = polyak_bound(task["coords"], max(DEEP_CHECKPOINTS),
                     checkpoints=DEEP_CHECKPOINTS)
    common = {"instance": task["instance"], "n": task["n"], "d": task["d"],
              "true_cost": task["true_cost"], "solver": task["solver"]}
    if r["status"] != "ok":
        return [dict(common, k=k, bound=float("nan"), status=r["status"])
                for k in DEEP_CHECKPOINTS]
    return [dict(common, k=k, bound=r["bounds"][k], status="ok",
                 upper_bound=r["upper_bound"], iterations_used=r["iterations_used"],
                 is_optimal=r["is_optimal"], stopped_reason=r["stopped_reason"],
                 ascent_seconds=r["ascent_seconds"])
            for k in DEEP_CHECKPOINTS]


def _run_ub_one(task: dict) -> list[dict]:
    """Three step-scaling sources, one instance, identical everything else."""
    rows = []
    variants = (("nn", dict(two_opt=False)),
                ("nn_2opt", dict(two_opt=True)),
                ("optimum", dict(upper_bound=task["true_cost"])))
    for tag, kw in variants:
        r = polyak_bound(task["coords"], max(CHECKPOINTS),
                         checkpoints=CHECKPOINTS, **kw)
        if r["status"] != "ok":
            continue
        for k in CHECKPOINTS:
            rows.append({"instance": task["instance"], "n": task["n"],
                         "d": task["d"], "true_cost": task["true_cost"],
                         "solver": task["solver"], "ub_variant": tag,
                         "k": k, "bound": r["bounds"][k],
                         "upper_bound": r["upper_bound"],
                         "iterations_used": r["iterations_used"],
                         "disqualified": tag == "optimum"})
    return rows


def _parallel(tasks: list[dict], fn, workers: int, tag: str) -> pd.DataFrame:
    tasks = sorted(tasks, key=lambda t: -t["n"])
    print(f"[{tag}] {len(tasks)} instances, n in [{tasks[-1]['n']}, {tasks[0]['n']}], "
          f"{workers} workers", flush=True)
    rows: list[dict] = []
    done = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, t) for t in tasks]
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                el = time.perf_counter() - t0
                print(f"  [{tag}] {done}/{len(tasks)}  {el:.0f}s, "
                      f"~{el / done * (len(tasks) - done):.0f}s left", flush=True)
    print(f"[{tag}] done in {time.perf_counter() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def ub_sensitivity(workers: int, per_cell: int = 12) -> Path:
    base = load_base()
    sel = stratified(base, per_cell, SEED_UB, n_lo=40, n_hi=250)
    df = _parallel(tasks_from(sel), _run_ub_one, workers, "ub")
    p = OUT / "polyak_nd_ub_sensitivity.csv"
    df.to_csv(p, index=False, float_format=FLOAT_FMT)
    return p


def probe(workers: int, per_cell: int = 24) -> Path:
    base = load_base()
    sel = stratified(base, per_cell, SEED_PROBE, n_lo=40, n_hi=250)
    df = _parallel(tasks_from(sel), _run_probe_one, workers, "probe")
    p = OUT / "polyak_nd_probe.csv"
    df.to_csv(p, index=False, float_format=FLOAT_FMT)
    return p


def sweep(workers: int, limit: int | None) -> Path:
    base = load_base()
    if limit:
        base = base.head(limit)
    df = _parallel(tasks_from(base), _run_sweep_one, workers, "sweep")
    df = df.sort_values(["instance", "k"])
    p = OUT / "polyak_nd_sweep.csv"
    df.to_csv(p, index=False, float_format=FLOAT_FMT)
    return p


#: Budgets the cost sweep measures. Deliberately short of the full ladder: each
#: rung is timed by an *independent* run, so the measurement cost is the sum of
#: the ladder, and the crossover the ND result turns on sits at the low end.
TIMED_CHECKPOINTS: tuple[int, ...] = (0, 10, 25, 50, 100, 200, 500)


def timing(per_cell: int = 6, repeats: int = 5) -> Path:
    """Serial single-thread cost, Polyak ladder against GART 2.0, same session.

    Both are measured back to back in this one process under the same thread
    caps, so the *ratio* is meaningful even though absolute milliseconds are
    load-dependent. The GART 2.0 call is the one ``run_benchmark_ND_final``
    makes, feature construction inside the clock.

    Each budget is timed by its own ``polyak_bound`` call rather than by
    stamping one long ascent. That costs more measurement time and needs no
    assumption about how a prefix's wall clock relates to the whole run's; the
    timed region is exactly what a user paying for budget ``k`` would run --
    dedup, matrix, constructive tour, ``k`` Prim passes.
    """
    from lgbm_model_v3.lgbm_estimator_gart2 import TSP_GART2_Estimator

    base = load_base()
    sel = stratified(base, per_cell, SEED_TIMING)
    tasks = tasks_from(sel)
    # The runner's own construction and the runner's own call, so the timed
    # region matches ``run_benchmark_ND_final._score_one`` line for line.
    gart = TSP_GART2_Estimator(str(ROOT / "lgbm_model_v3"))

    rows = []
    for i, t in enumerate(tasks, 1):
        X32 = np.asarray(t["coords"], dtype=np.float32)
        gart.estimate(X32, t["d"], t["grid_size"])       # warm predict path
        polyak_bound(t["coords"], 5)                     # warm JIT
        g = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            gart.estimate(X32, t["d"], t["grid_size"])
            g.append(time.perf_counter() - t0)

        cell = {"instance": t["instance"], "n": t["n"], "d": t["d"],
                "gart_ms": 1000 * float(np.median(g))}
        for k in TIMED_CHECKPOINTS:
            reps = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                polyak_bound(t["coords"], k)
                reps.append(time.perf_counter() - t0)
            cell[f"hk_k{k}_ms"] = 1000 * float(np.median(reps))
        rows.append(cell)
        if i % 10 == 0 or i == len(tasks):
            print(f"  [timing] {i}/{len(tasks)}", flush=True)

    df = pd.DataFrame(rows)
    p = OUT / "polyak_nd_timing.csv"
    df.to_csv(p, index=False, float_format=FLOAT_FMT)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ub", "probe", "sweep", "timing"],
                    required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-cell", type=int, default=None)
    a = ap.parse_args()

    if a.mode == "ub":
        p = ub_sensitivity(a.workers, a.per_cell or 12)
    elif a.mode == "probe":
        p = probe(a.workers, a.per_cell or 24)
    elif a.mode == "sweep":
        p = sweep(a.workers, a.limit)
    else:
        p = timing(a.per_cell or 6)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
