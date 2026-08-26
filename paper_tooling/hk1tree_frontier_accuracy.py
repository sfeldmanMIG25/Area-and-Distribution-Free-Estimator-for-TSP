"""Accuracy of the Held--Karp 1-tree bound across an ascent-budget ladder.

WHAT THIS PRODUCES
------------------
For every instance in four corpora -- TSPLIB EUC_2D (the stratum the published
timing column reports), the 2D benchmark, the full ND test split, and a
documented random subsample of the ND *train* split -- the bound at every
budget in :data:`CHECKPOINTS`, written one row per (corpus, instance, k).

Accuracy is scored with the manuscript's own metrics, computed downstream in
``hk1tree_frontier_analyze.py`` from ``err_pct = 100 (pred - true) / true``,
exactly as ``build_paper_tables.py`` line 503 defines it.

THE ONE OPTIMISATION, AND WHY IT IS EXACT
-----------------------------------------
Running the ladder naively costs ``sum(CHECKPOINTS)`` iterations per instance.
This script runs ``max(CHECKPOINTS)`` and reads the incumbent off at each
checkpoint, which is a 2.2x saving on the ladder below.

That is only legitimate because ``held_karp_1tree._ascend`` was deliberately
built so the schedule sees neither ``n`` nor the budget: the length-``k``
trajectory is a *strict prefix* of the length-``k'`` trajectory for every
``k' > k``, so ``bound(k) = max`` of the incumbent over the first ``k``
iterates. :func:`_ascend_checkpointed` below mirrors ``_ascend`` step for step.
Mirrored code drifts, so ``--verify`` re-runs the shipped
``held_karp_1tree.one_tree_bound`` at every ladder budget on a sample of
instances and asserts bit-exact agreement. Run it before trusting the sweep.

    python paper_tooling/hk1tree_frontier_accuracy.py --verify
    python paper_tooling/hk1tree_frontier_accuracy.py --corpus tsplib --workers 8

Output: paper_tooling/hk1tree_frontier_<corpus>.csv
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from held_karp_1tree import (  # noqa: E402
    INITIAL_PERIOD,
    _IMPROVE_RTOL,
    _one_tree_coords,
    _one_tree_dense,
)

#: The ladder. The task floor is {0, 10, 50, 100, 500}; 25/200 are free once the
#: ascent is checkpointed, and 1000/2000 are carried because the bound is still
#: tightening at 500 and the frontier claim needs to show where it stops paying.
CHECKPOINTS: tuple[int, ...] = (0, 10, 25, 50, 100, 200, 500, 1000, 2000)
K_MAX = max(CHECKPOINTS)

#: Above this the coordinate kernel runs instead of the stored matrix. Higher
#: than ``held_karp_1tree.DENSE_MAX_N`` on purpose: this script owns its memory
#: budget (one worker touching n=18512 holds 2.7 GB of a 42 GB free pool) and
#: the two kernels compute the same 1-tree, so the choice is pure speed here.
DENSE_MAX_N_LOCAL = 20000

SEED_TRAIN_SUBSAMPLE = 20260811
N_TRAIN_SUBSAMPLE = 2000

#: A second, planar-only train draw. The global train split is d in {2..50} and
#: only 4104 of its 69768 rows are planar, so a 2000-row uniform draw carries
#: ~120 planar instances -- too thin to fit a correction for the two planar
#: evaluation corpora on. This draw is still TRAIN-only; it is a different
#: stratum of the same split, not a different split.
SEED_TRAIN_D2 = 20260812
N_TRAIN_D2 = 900

OUT_DIR = ROOT / "paper_tooling"
TSPLIB_DIR = ROOT / "tsplib_benchmark" / "instances"
TSPLIB_RESULTS = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"
BASE_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv"
BASE_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
FEATURES = ROOT / "tsp_features_v3.csv"
ND_INSTANCES = ROOT / "instances"
ND_SOLUTIONS = ROOT / "solutions"


# ---------------------------------------------------------------------------
# Checkpointed ascent -- a mirror of held_karp_1tree._ascend
# ---------------------------------------------------------------------------
def _ascend_checkpointed(evaluate, n: int, checkpoints: tuple[int, ...],
                         special: int = 0) -> tuple[dict[int, float], int, bool]:
    """``{k: bound(k)}`` for every ``k`` in ``checkpoints``, in one ascent.

    Every line below has a counterpart in ``held_karp_1tree._ascend``; the only
    additions are the incumbent snapshots. Keep the two in step, and use
    ``--verify`` to prove they are.
    """
    ks = sorted(set(int(k) for k in checkpoints))
    budget = max(ks)
    out: dict[int, float] = {}

    pi = np.zeros(n, dtype=np.float64)
    length, degrees = evaluate(pi)
    w0 = float(length)
    w_best = w0

    def finish(used: int, optimal: bool) -> tuple[dict[int, float], int, bool]:
        # Every checkpoint at or past the stopping point sees the final
        # incumbent: the ascent produced no further iterate to improve on.
        for k in ks:
            out.setdefault(k, w_best)
        return out, used, optimal

    out[0] = w0
    if budget <= 0:
        return finish(0, False)

    t = w0 / (2.0 * n)
    if not (t > 0.0) or not np.isfinite(t):
        return finish(0, False)

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
                return finish(used, True)

            direction = g if first_step else 0.7 * g + 0.3 * g_prev
            first_step = False
            g_prev = g
            pi = pi + (t * (period - j) / period) * direction

            length, degrees = evaluate(pi)
            w = float(length) - 2.0 * float(pi.sum())
            used += 1

            if w > w_best + _IMPROVE_RTOL * max(1.0, abs(w_best)):
                w_best = w
                improved_anywhere = True
                improved_last = j == steps - 1
            else:
                improved_last = False

            if used in out or used in ks:
                out[used] = w_best

        if improved_last:
            period *= 2
            t *= 2.0
        elif not improved_anywhere:
            t *= 0.5
            period = max(1, period // 2)

    return finish(used, False)


def ladder_bounds_matrix(D: np.ndarray) -> dict:
    """Ladder from an explicit symmetric matrix -- the metric-consistent path.

    Used for the ``tsplib_int`` robustness corpus: TSPLIB's published optima are
    exact with respect to ``nint(euclidean)``, so scoring the bound on that same
    integer matrix removes the unit mismatch that the float64 corpus inherits
    from the benchmark's own convention.
    """
    D = np.ascontiguousarray(D, dtype=np.float64)
    n = D.shape[0]
    t0 = time.perf_counter()
    bounds, used, is_opt = _ascend_checkpointed(
        lambda pi: _one_tree_dense(D, pi, 0), n, CHECKPOINTS)
    return {"status": "ok", "n_unique": n, "backend": "matrix",
            "iterations_used": used, "is_optimal": bool(is_opt),
            "ascent_seconds": time.perf_counter() - t0, "bounds": bounds}


def ladder_bounds(coords: np.ndarray) -> dict:
    """Bound at every checkpoint for one point set, plus provenance."""
    X = np.unique(np.asarray(coords, dtype=np.float64), axis=0)
    n = X.shape[0]
    if n < 3:
        return {"status": "degenerate_n", "n_unique": n}
    X = np.ascontiguousarray(X)

    if n <= DENSE_MAX_N_LOCAL:
        sq = (X * X).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(D, 0.0, out=D)
        np.sqrt(D, out=D)
        np.fill_diagonal(D, 0.0)
        backend = "dense"

        def ev(pi):
            return _one_tree_dense(D, pi, 0)
    else:
        backend = "coords"

        def ev(pi):
            return _one_tree_coords(X, pi, 0)

    t0 = time.perf_counter()
    bounds, used, is_opt = _ascend_checkpointed(ev, n, CHECKPOINTS)
    return {"status": "ok", "n_unique": n, "backend": backend,
            "iterations_used": used, "is_optimal": bool(is_opt),
            "ascent_seconds": time.perf_counter() - t0, "bounds": bounds}


# ---------------------------------------------------------------------------
# Corpus loaders -- each yields (instance, coords, true_cost, extra dict)
# ---------------------------------------------------------------------------
_BIN_HEADER = struct.Struct("IIII")


def _load_bin(path: Path, n: int, d: int, grid: int) -> np.ndarray:
    """ND coordinate reader. Same header check as run_benchmark_ND_final."""
    blob = path.read_bytes()
    b_n, b_d, b_grid, dist_len = _BIN_HEADER.unpack_from(blob, 0)
    offset = _BIN_HEADER.size + dist_len
    if (b_n, b_d, b_grid) != (int(n), int(d), int(grid)):
        raise ValueError(f"{path.name}: stale .bin header")
    if len(blob) - offset != b_n * b_d * 4:
        raise ValueError(f"{path.name}: bad .bin body length")
    return np.frombuffer(blob, dtype=np.float32, count=b_n * b_d,
                         offset=offset).reshape(b_n, b_d).copy()


def tasks_tsplib() -> list[dict]:
    """The 78 EUC_2D instances GART 2.0 is scored on, with published optima.

    Coordinates are the raw float64 ones every other estimator receives, and
    the label is the same published integer optimum every other estimator is
    scored against. The bound therefore inherits the corpus's own metric
    mismatch (float64 prediction, nint label) rather than inventing a private
    convention -- which is the only way its MAPE is comparable to GART 2.0's.
    A metric-consistent integer-matrix check lives in hk1tree_tsplib.py.
    """
    from tsplib_parser import parse_tsplib_file

    res = pd.read_csv(TSPLIB_RESULTS)
    ref = res[(res.model == "GART_2.0") & (res.edge_weight_type == "EUC_2D")]
    labels = dict(zip(ref.instance.astype(str), ref.true_cost.astype(float)))

    out = []
    for name, true_cost in labels.items():
        info = parse_tsplib_file(str(TSPLIB_DIR / f"{name}.tsp"))
        coords = info["raw_coords"]
        if coords is None:
            raise ValueError(f"{name}: EUC_2D instance with no coordinates")
        out.append({"instance": name, "coords": np.asarray(coords, dtype=np.float64),
                    "true_cost": float(true_cost), "n": int(info["n"]), "d": 2})
    if len(out) != 78:
        raise ValueError(f"expected 78 EUC_2D instances, got {len(out)}")
    return out


#: Largest ``n`` scored in the integer-matrix robustness corpus. n=11849 needs
#: 1.1 GB for the matrix and 2000 dense Prims on it; the five instances above
#: this contribute nothing the smaller 73 do not, and the corpus exists only to
#: separate the metric artefact from the bound's real gap.
TSPLIB_INT_MAX_N = 8000


def tasks_tsplib_int() -> list[dict]:
    """The same EUC_2D instances, scored in TSPLIB's own ``nint`` metric.

    NOT comparable to the estimator column -- every estimator in the benchmark
    predicts in float64 against these integer labels. This corpus exists to
    measure how much of the float64 corpus's positive tail is that convention.
    """
    from tsplib_parser import parse_tsplib_file

    res = pd.read_csv(TSPLIB_RESULTS)
    ref = res[(res.model == "GART_2.0") & (res.edge_weight_type == "EUC_2D")]
    out = []
    for name, true_cost in zip(ref.instance.astype(str), ref.true_cost.astype(float)):
        info = parse_tsplib_file(str(TSPLIB_DIR / f"{name}.tsp"))
        n = int(info["n"])
        if n > TSPLIB_INT_MAX_N:
            continue
        X = np.asarray(info["raw_coords"], dtype=np.float64)
        sq = (X * X).sum(1)
        D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
        np.maximum(D, 0.0, out=D)
        np.sqrt(D, out=D)
        D = np.rint(D)                      # TSPLIB nint, the label's own metric
        np.fill_diagonal(D, 0.0)
        out.append({"instance": name, "matrix": D, "true_cost": float(true_cost),
                    "n": n, "d": 2})
    return out


def tasks_2d() -> list[dict]:
    from tsp_utils import parse_tsp_instance

    base = pd.read_csv(BASE_2D)
    out = []
    for r in base.itertuples(index=False):
        coords = parse_tsp_instance(Path(r.file_path)).coordinates
        out.append({"instance": r.instance, "coords": np.asarray(coords),
                    "true_cost": float(r.true_cost), "n": int(r.n_customers),
                    "d": int(r.dimension)})
    return out


def tasks_nd() -> list[dict]:
    base = pd.read_csv(BASE_ND)
    out = []
    for r in base.itertuples(index=False):
        p = Path(r.file_path).with_suffix(".bin")
        coords = _load_bin(p, r.n_customers, r.dimension, r.grid_size)
        out.append({"instance": r.instance, "coords": coords,
                    "true_cost": float(r.true_cost), "n": int(r.n_customers),
                    "d": int(r.dimension)})
    return out


def _tasks_train(size: int, seed: int, only_d2: bool) -> list[dict]:
    """Random subsample of the TRAIN split -- the only split a correction may
    be fitted on. Sizes and seeds are constants at the top of this file so the
    draws are reproducible and auditable."""
    feats = pd.read_csv(FEATURES, usecols=["instance_name", "split", "n_customers",
                                           "dimension", "grid_size"])
    train = feats[feats.split == "train"]
    if only_d2:
        train = train[train.dimension == 2]
    train = train.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train), size=min(size, len(train)), replace=False)
    sel = train.iloc[np.sort(idx)]

    out = []
    for r in sel.itertuples(index=False):
        name = r.instance_name
        sol = json.loads((ND_SOLUTIONS / f"{name}.sol.json").read_text())
        coords = _load_bin(ND_INSTANCES / f"{name}.bin", r.n_customers,
                           r.dimension, r.grid_size)
        out.append({"instance": name, "coords": coords,
                    "true_cost": float(sol["optimal_cost"]),
                    "n": int(r.n_customers), "d": int(r.dimension)})
    return out


def tasks_train() -> list[dict]:
    return _tasks_train(N_TRAIN_SUBSAMPLE, SEED_TRAIN_SUBSAMPLE, only_d2=False)


def tasks_train_d2() -> list[dict]:
    return _tasks_train(N_TRAIN_D2, SEED_TRAIN_D2, only_d2=True)


LOADERS = {"tsplib": tasks_tsplib, "tsplib_int": tasks_tsplib_int, "2d": tasks_2d,
           "nd": tasks_nd, "train": tasks_train, "train_d2": tasks_train_d2}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_one(task: dict) -> list[dict]:
    r = (ladder_bounds_matrix(task["matrix"]) if "matrix" in task
         else ladder_bounds(task["coords"]))
    common = {"instance": task["instance"], "n": task["n"], "d": task["d"],
              "true_cost": task["true_cost"], "n_unique": r["n_unique"]}
    if r["status"] != "ok":
        return [dict(common, k=k, bound=float("nan"), status=r["status"])
                for k in CHECKPOINTS]
    return [dict(common, k=k, bound=r["bounds"][k], status="ok",
                 backend=r["backend"], iterations_used=r["iterations_used"],
                 is_optimal=r["is_optimal"], ascent_seconds=r["ascent_seconds"])
            for k in CHECKPOINTS]


def run_corpus(corpus: str, workers: int, limit: int | None) -> Path:
    print(f"[{corpus}] loading instances ...", flush=True)
    tasks = LOADERS[corpus]()
    if limit:
        tasks = tasks[:limit]
    # Largest first: one n=18512 ascent outlasts the entire tail, so it must
    # start in the first wave or it sets the wall clock all by itself.
    tasks.sort(key=lambda t: -t["n"])
    print(f"[{corpus}] {len(tasks)} instances, n in [{tasks[-1]['n']}, {tasks[0]['n']}], "
          f"K_MAX={K_MAX}, {workers} workers", flush=True)

    rows: list[dict] = []
    done = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, t): t["instance"] for t in tasks}
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                el = time.perf_counter() - t0
                print(f"  [{corpus}] {done}/{len(tasks)}  {el:.0f}s elapsed, "
                      f"~{el / done * (len(tasks) - done):.0f}s left", flush=True)

    df = pd.DataFrame(rows).sort_values(["instance", "k"])
    out = OUT_DIR / f"hk1tree_frontier_{corpus}.csv"
    df.to_csv(out, index=False)
    print(f"[{corpus}] wrote {out}  ({len(df)} rows, "
          f"{time.perf_counter() - t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Verification of the mirrored ascent
# ---------------------------------------------------------------------------
def verify(n_cases: int = 40) -> None:
    """Assert the checkpointed ascent equals the shipped one at every budget.

    Bit-exact is the right bar, not 'close': both paths execute the same
    arithmetic in the same order, so any difference at all means the mirror has
    drifted from ``_ascend`` and the whole sweep is suspect.
    """
    from held_karp_1tree import one_tree_bound

    rng = np.random.default_rng(7)
    cases = []
    for i in range(n_cases):
        n = int(rng.integers(5, 400))
        d = int(rng.choice([2, 2, 3, 5, 10, 40]))
        style = i % 4
        X = rng.random((n, d)) * 1000.0
        if style == 1:                                   # clustered
            c = rng.random((max(2, n // 25), d)) * 1000.0
            X = c[rng.integers(0, len(c), n)] + rng.normal(0, 15, (n, d))
        elif style == 2:                                 # near-degenerate line
            X = np.tile(rng.random((n, 1)) * 1000.0, (1, d)) + rng.normal(0, 0.5, (n, d))
        elif style == 3:                                 # integer grid, many ties
            X = rng.integers(0, 12, (n, d)).astype(np.float64)
        cases.append(X)

    bad = 0
    for i, X in enumerate(cases):
        Xu = np.ascontiguousarray(np.unique(X, axis=0))
        if Xu.shape[0] < 3:
            continue
        mine = ladder_bounds(X)
        for k in CHECKPOINTS:
            ref = one_tree_bound(Xu, k, dense=Xu.shape[0] <= DENSE_MAX_N_LOCAL).bound
            got = mine["bounds"][k]
            if got != ref:
                bad += 1
                print(f"  MISMATCH case {i} n={Xu.shape[0]} k={k}: "
                      f"checkpointed={got!r} shipped={ref!r} "
                      f"rel={abs(got - ref) / max(1e-300, abs(ref)):.3e}")
    total = len(cases) * len(CHECKPOINTS)
    if bad:
        raise SystemExit(f"VERIFY FAILED: {bad}/{total} budgets disagree")
    print(f"VERIFY OK: {total} (case, budget) pairs bit-exact against "
          f"held_karp_1tree.one_tree_bound")

    # Monotonicity is what makes the checkpoint read-off meaningful at all.
    for i, X in enumerate(cases):
        b = ladder_bounds(X)
        if b["status"] != "ok":
            continue
        seq = [b["bounds"][k] for k in CHECKPOINTS]
        if any(seq[j + 1] < seq[j] for j in range(len(seq) - 1)):
            raise SystemExit(f"VERIFY FAILED: bound decreases in k on case {i}: {seq}")
    print(f"VERIFY OK: bound non-decreasing in k on all {len(cases)} cases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=[*LOADERS, "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    order = (["tsplib", "2d", "train", "train_d2", "nd"]
             if args.corpus == "all" else [args.corpus])
    for c in order:
        run_corpus(c, args.workers, args.limit)


if __name__ == "__main__":
    main()
