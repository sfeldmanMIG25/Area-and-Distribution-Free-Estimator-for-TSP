"""Score both 1-tree ascents on every benchmark the paper reports.

WHY THIS EXISTS
---------------
The Held--Karp 1-tree bound entered this paper late, as a comparator for two
strata only, and the two ascents were run on disjoint corpora: the
Volgenant--Jonker ascent on TSPLIB EUC_2D and the 2D benchmark, the Polyak
ascent on the multidimensional split. That left the comparison unable to
answer "which is better, and where" on the same footing as every estimator in
Table~\\ref{tab:benchmark_models}, each of which is scored on all four strata.

This script fills the missing cells so both ascents are scored on every
corpus. What was already computed is not recomputed:

    corpus              VJ ascent                      Polyak ascent
    ------------------  -----------------------------  --------------------
    TSPLIB EUC_2D       hk1tree_frontier_tsplib.csv    THIS SCRIPT
    2D benchmark        hk1tree_frontier_2d.csv        THIS SCRIPT
    multidimensional    hk1tree_frontier_nd.csv        polyak_nd_sweep.csv
    TSPLIB non-EUC_2D   THIS SCRIPT                    THIS SCRIPT

The non-Euclidean corpus is new to this family entirely. It is the stratum
where the bound's structural advantage over every learned estimator here is
sharpest: a 1-tree needs a symmetric distance matrix and nothing else -- no
coordinates, no embedding, no triangle inequality (see the ``held_karp_1tree``
module docstring on why the Lagrangian argument is metric-free). GART 2.0
reaches this stratum only through the MDS pipeline of Section
\\ref{sec:application}; the bound reads the matrix directly.

THE LADDER, AND WHY IT IS ONE ASCENT PER INSTANCE
-------------------------------------------------
Both ascents have the prefix property: with a schedule that sees neither ``n``
nor the budget, the length-``k`` trajectory is a strict prefix of the
length-``k'`` trajectory for ``k' > k``, so the incumbent read off at ``k`` is
exactly ``bound(k)``. One ascent to ``max(CHECKPOINTS)`` therefore yields the
whole ladder. ``--verify`` re-runs the shipped single-budget entry points at
every rung and asserts agreement, because that property is a claim about code
that can drift, not a theorem.

GATES
-----
``--gates`` runs three checks and writes the verdict to
``hk1tree_allbench_gates.json``:

  1. certificate -- ``bound <= optimum`` on every instance with a known
     optimum, at every budget. A violation is disqualifying: it means the
     "lower bound" is not one.
  2. exact DP    -- at ``n <= 10`` the label is replaced by an exhaustively
     computed optimum, so the certificate is checked against ground truth
     rather than against a stored tour that could itself be wrong.
  3. monotone    -- ``bound(k)`` is non-decreasing in ``k`` on every instance.

Usage
-----
    ...python.exe paper_tooling/hk1tree_all_benchmarks.py --verify
    ...python.exe paper_tooling/hk1tree_all_benchmarks.py --corpus noneuc --workers 8
    ...python.exe paper_tooling/hk1tree_all_benchmarks.py --corpus all --workers 8
    ...python.exe paper_tooling/hk1tree_all_benchmarks.py --gates

Output: paper_tooling/hk1tree_<ascent>_<corpus>.csv
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tsplib_benchmark", ROOT / "paper_tooling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# -- numba/coverage shim; see the note below before deleting -----------------
# numba's ``misc.coverage_support`` does ``import coverage`` inside a
# ``try/except ImportError`` and then subclasses ``coverage.types.Tracer``.
# No ``coverage`` distribution is installed here, so that guard normally fires
# and coverage support switches off. But any bare ``coverage/`` DIRECTORY at
# the repo root is importable as a PEP 420 namespace package once the repo root
# joins ``sys.path``, which the loop above just did to reach
# ``held_karp_1tree``. The import then succeeds, ``coverage.types`` does not
# exist, and numba dies on an AttributeError its guard was never written to
# catch. This is not hypothetical: it killed a sweep of this script mid-run.
# The writer responsible was ``data_pipeline/coverage_gen.py``, now fixed at the
# source -- it writes ``alpha_coverage/``. This shim is kept as insurance
# against the next output directory that lands on a stdlib-adjacent name, and
# is inert once a real ``coverage`` distribution is installed (that spec has a
# non-None ``origin``).
# Binding ``None`` in ``sys.modules`` makes ``import coverage`` raise
# ImportError again, which is exactly the state numba is built to handle.
# ORDER MATTERS TWICE: this has to run after the ``sys.path`` insertion above
# (or the namespace package is not yet findable and the check is a no-op) and
# before the first import that pulls numba in -- including in every spawned
# worker, which re-executes this module top to bottom.
if "coverage" not in sys.modules:
    _cov = importlib.util.find_spec("coverage")
    if _cov is not None and _cov.origin is None:
        sys.modules["coverage"] = None  # type: ignore[assignment]

from held_karp_1tree import (  # noqa: E402
    _one_tree_dense,
    one_tree_bound,
    one_tree_bound_from_matrix,
)
from hk1tree_frontier_accuracy import (  # noqa: E402
    CHECKPOINTS,
    _ascend_checkpointed,
    tasks_2d,
    tasks_tsplib,
    tasks_tsplib_int,
)
from hk1tree_polyak import (  # noqa: E402
    _exact_dp,
    polyak_bound,
    polyak_bound_from_matrix,
)

OUT_DIR = ROOT / "paper_tooling"
TSPLIB_DIR = ROOT / "tsplib_benchmark" / "instances"
TSPLIB_RESULTS = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"

K_MAX = max(CHECKPOINTS)

#: Largest ``n`` scored on the non-Euclidean corpus. The bound's cost is
#: ``Theta(k n^2)`` and the matrix is materialised, so ``pla33810`` (9.1 GB,
#: ~40 min/instance) and ``pla85900`` (59 GB) are out of budget. Both are
#: CEIL_2D, i.e. native-coordinate instances that are Euclidean up to a
#: ceiling; neither is a distance-matrix instance, so dropping them costs the
#: "needs only a metric" claim nothing. The cap matches the 8000 already used
#: by ``hk1tree_frontier_accuracy.TSPLIB_INT_MAX_N`` so the two robustness
#: corpora are gated identically.
NONEUC_MAX_N = 8000

#: Exhaustive DP is ``O(n^2 2^n)``. At n=10 that is 102,400 states -- instant.
EXACT_DP_MAX_N = 10

#: Certificate tolerance, relative to the label. Covers float64 accumulation in
#: the 1-tree sum over up to ~10^4 edges; anything above this is a real
#: violation, not arithmetic.
CERT_RTOL = 1e-9


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def tasks_noneuc() -> list[dict]:
    """TSPLIB95's non-EUC_2D instances, scored from their distance matrices.

    Returns every instance the parser can build a matrix for under
    :data:`NONEUC_MAX_N`, carrying the same published optimum every estimator
    in Table~\\ref{tab:tsplib_nonEuc} is scored against, plus the two flags the
    downstream screen needs: ``edge_weight_type`` and whether the instance is
    one of the structurally nonmetric EXPLICIT matrices.

    Nothing is screened out here. The bound is valid on non-metric input, so
    the screen is an analysis-time choice about comparability with GART 2.0,
    not a precondition of the measurement -- and making it here would throw
    away the evidence for that very point.
    """
    from exclusions import STRUCTURAL_TRIANGLE_VIOLATORS
    from tsplib_parser import parse_tsplib_file

    res = pd.read_csv(TSPLIB_RESULTS)
    ref = res[(res.model == "GART_2.0") & (res.edge_weight_type != "EUC_2D")]
    ref = ref.drop_duplicates(subset="instance")

    out = []
    for r in ref.itertuples(index=False):
        name = str(r.instance)
        n = int(r.n)
        if n > NONEUC_MAX_N:
            continue
        info = parse_tsplib_file(str(TSPLIB_DIR / f"{name}.tsp"))
        D = info["distance_matrix"]
        if D is None:
            # CEIL_2D. The parser skips the matrix for native-Euclidean types
            # because the estimators consume coordinates and pla85900 would
            # need 55 GiB. The bound needs the matrix, and it must be the
            # instance's own metric -- ceil of the Euclidean distance -- or it
            # would not be a lower bound on the label's optimum.
            ewt = str(info["edge_weight_type"])
            if ewt != "CEIL_2D":
                raise ValueError(f"{name}: no matrix and unexpected type {ewt}")
            X = np.asarray(info["raw_coords"], dtype=np.float64)
            sq = (X * X).sum(axis=1)
            D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
            np.maximum(D, 0.0, out=D)
            np.sqrt(D, out=D)
            np.ceil(D, out=D)
            np.fill_diagonal(D, 0.0)
        D = np.ascontiguousarray(D, dtype=np.float64)
        if D.shape != (int(info["n"]), int(info["n"])):
            raise ValueError(f"{name}: matrix shape {D.shape} != n={info['n']}")
        out.append({"instance": name, "matrix": D, "true_cost": float(r.true_cost),
                    "n": int(info["n"]), "d": 2,
                    "edge_weight_type": str(r.edge_weight_type),
                    "structural_violator": name in STRUCTURAL_TRIANGLE_VIOLATORS})
    return out


CORPORA = {
    "2d": (tasks_2d, "coords"),
    "tsplib": (tasks_tsplib, "coords"),
    "tsplib_int": (tasks_tsplib_int, "matrix"),
    "noneuc": (tasks_noneuc, "matrix"),
}

#: (ascent, corpus) pairs this script owns. The rest already exist on disk.
#: ``tsplib_int`` is not a reporting corpus -- it is the metric-consistent
#: twin of ``tsplib``, and it exists so the certificate gate has something
#: honest to check the planar ascents against. See :func:`gates`.
JOBS = [
    ("polyak", "2d"),
    ("polyak", "tsplib"),
    ("polyak", "tsplib_int"),
    ("vj", "noneuc"),
    ("polyak", "noneuc"),
]

#: Corpora whose prediction metric differs from their label's metric. On these
#: the raw excess over the label is a unit convention, not a broken bound, and
#: the certificate has to be checked on the twin named here instead.
METRIC_TWIN = {"tsplib": "tsplib_int"}


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def _dense(X: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix. Same expansion the shipped kernels use, so
    the ladder and the entry point see bit-identical input."""
    sq = (X * X).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    np.sqrt(D, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def _meta(task: dict) -> dict:
    keep = ("instance", "n", "d", "true_cost", "edge_weight_type",
            "structural_violator")
    return {k: task[k] for k in keep if k in task}


def _rows_failed(task: dict, ascent: str, status: str) -> list[dict]:
    return [dict(_meta(task), ascent=ascent, k=k, bound=float("nan"),
                 status=status) for k in CHECKPOINTS]


def _run_vj(task: dict) -> list[dict]:
    """Volgenant--Jonker ascent. No upper bound is consulted at any point."""
    t0 = time.perf_counter()
    if "matrix" in task:
        D = task["matrix"]
        n = D.shape[0]
        if n < 3:
            return _rows_failed(task, "vj", "degenerate_n")
        bounds, used, opt = _ascend_checkpointed(
            lambda pi: _one_tree_dense(D, pi, 0), n, CHECKPOINTS)
        backend = "matrix"
    else:
        X = np.ascontiguousarray(np.unique(np.asarray(task["coords"],
                                                      dtype=np.float64), axis=0))
        n = X.shape[0]
        if n < 3:
            return _rows_failed(task, "vj", "degenerate_n")
        D = _dense(X)
        bounds, used, opt = _ascend_checkpointed(
            lambda pi: _one_tree_dense(D, pi, 0), n, CHECKPOINTS)
        backend = "dense"
    el = time.perf_counter() - t0
    return [dict(_meta(task), ascent="vj", k=k, bound=bounds[k], status="ok",
                 n_unique=n, backend=backend, iterations_used=used,
                 is_optimal=opt, ascent_seconds=el) for k in CHECKPOINTS]


def _run_polyak(task: dict) -> list[dict]:
    """Polyak ascent against a constructive tour. The label is never read."""
    if "matrix" in task:
        r = polyak_bound_from_matrix(task["matrix"], K_MAX, checkpoints=CHECKPOINTS)
    else:
        r = polyak_bound(task["coords"], K_MAX, checkpoints=CHECKPOINTS)
    if r["status"] != "ok":
        return _rows_failed(task, "polyak", r["status"])
    return [dict(_meta(task), ascent="polyak", k=k, bound=r["bounds"][k],
                 status="ok", n_unique=r["n_unique"], backend=r["backend"],
                 upper_bound=r["upper_bound"], ub_source=r["ub_source"],
                 two_opt_sweeps=r["two_opt_sweeps"],
                 iterations_used=r["iterations_used"], is_optimal=r["is_optimal"],
                 stopped_reason=r["stopped_reason"], ub_seconds=r["ub_seconds"],
                 matrix_seconds=r["matrix_seconds"],
                 ascent_seconds=r["ascent_seconds"]) for k in CHECKPOINTS]


RUNNERS = {"vj": _run_vj, "polyak": _run_polyak}


def _parallel(tasks: list[dict], fn, workers: int, tag: str) -> pd.DataFrame:
    tasks = sorted(tasks, key=lambda t: -t["n"])
    print(f"[{tag}] {len(tasks)} instances, n in [{tasks[-1]['n']}, "
          f"{tasks[0]['n']}], {workers} workers", flush=True)
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
# Verification: the checkpointed ladder against the shipped entry points
# ---------------------------------------------------------------------------
#: ``verify`` re-runs a full ascent per rung, so it costs ``sum(CHECKPOINTS)``
#: iterations per instance against the sweep's ``max(CHECKPOINTS)``. Cap ``n``:
#: the property under test is that a ladder read-off equals a fresh run, which
#: is a statement about the schedule and is exercised identically at n=200 and
#: at n=7397. Both kernels and all four edge-weight types stay in the sample.
VERIFY_MAX_N = 250


def verify(sample: int = 6, seed: int = 20260812) -> None:
    """Re-run the shipped single-budget bounds at every rung and compare.

    The VJ arm must agree bit-exactly: ``_ascend_checkpointed`` mirrors
    ``_ascend`` step for step and any drift is a bug. The Polyak arm is checked
    for the same prefix property through its own public entry point.
    """
    rng = np.random.default_rng(seed)
    ne = [t for t in tasks_noneuc() if t["n"] <= VERIFY_MAX_N]
    picks = [ne[i] for i in rng.choice(len(ne), size=min(sample, len(ne)),
                                       replace=False)]
    worst_vj = 0.0
    worst_pk = 0.0
    for t in picks:
        D = t["matrix"]
        ladder, _, _ = _ascend_checkpointed(lambda pi: _one_tree_dense(D, pi, 0),
                                            D.shape[0], CHECKPOINTS)
        pk = polyak_bound_from_matrix(D, K_MAX, checkpoints=CHECKPOINTS)["bounds"]
        for k in CHECKPOINTS:
            ref_vj = one_tree_bound_from_matrix(D, k).bound
            worst_vj = max(worst_vj, abs(ladder[k] - ref_vj))
            if ladder[k] != ref_vj:
                raise AssertionError(
                    f"VJ ladder drift on {t['instance']} at k={k}: "
                    f"{ladder[k]!r} vs shipped {ref_vj!r}")
            ref_pk = polyak_bound_from_matrix(D, k)["bound"]
            worst_pk = max(worst_pk, abs(pk[k] - ref_pk))
            if pk[k] != ref_pk:
                raise AssertionError(
                    f"Polyak ladder drift on {t['instance']} at k={k}: "
                    f"{pk[k]!r} vs shipped {ref_pk!r}")
        print(f"  ok {t['instance']:>10s} n={t['n']:>5d}", flush=True)

    # The coordinate path is the one the 2D and TSPLIB corpora take.
    t2 = [t for t in tasks_2d() if t["n"] <= VERIFY_MAX_N]
    for i in rng.choice(len(t2), size=3, replace=False):
        t = t2[i]
        X = np.unique(np.asarray(t["coords"], dtype=np.float64), axis=0)
        Dx = _dense(X)
        full = polyak_bound(t["coords"], K_MAX, checkpoints=CHECKPOINTS)["bounds"]
        lad, _, _ = _ascend_checkpointed(lambda pi: _one_tree_dense(Dx, pi, 0),
                                         X.shape[0], CHECKPOINTS)
        for k in CHECKPOINTS:
            got = polyak_bound(t["coords"], k)["bound"]
            worst_pk = max(worst_pk, abs(full[k] - got))
            if full[k] != got:
                raise AssertionError(
                    f"Polyak coord ladder drift on {t['instance']} at k={k}: "
                    f"{full[k]!r} vs shipped {got!r}")
            # The VJ coordinate path must also match its shipped entry point.
            ref = one_tree_bound(X, k).bound
            worst_vj = max(worst_vj, abs(lad[k] - ref))
            if lad[k] != ref:
                raise AssertionError(
                    f"VJ coord ladder drift on {t['instance']} at k={k}: "
                    f"{lad[k]!r} vs shipped {ref!r}")
        print(f"  ok {t['instance']:>10s} n={t['n']:>5d} (coords)", flush=True)

    print(f"VERIFY PASS  max|ladder-shipped| vj={worst_vj:g} polyak={worst_pk:g}")


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _exact_optimum_matrix(D: np.ndarray) -> float:
    return float(_exact_dp(np.ascontiguousarray(D, dtype=np.float64)))


def gates() -> dict:
    """Certificate, exact-DP and monotonicity checks over everything on disk."""
    frames = []
    for ascent, corpus in _all_pairs():
        p = _out_path(ascent, corpus)
        if not p.exists():
            print(f"  (skip, absent) {p.name}")
            continue
        df = pd.read_csv(p)
        df["ascent"] = ascent
        df["corpus"] = corpus
        frames.append(df)
    if not frames:
        raise SystemExit("no ladder CSVs on disk; run the sweeps first")
    all_df = pd.concat(frames, ignore_index=True)

    ok = all_df[all_df.status == "ok"].copy()
    report: dict = {"rows_total": int(len(all_df)), "rows_ok": int(len(ok))}

    # -- Gate 1: certificate ------------------------------------------------
    # A corpus whose prediction metric differs from its label's metric cannot
    # settle this question. TSPLIB EUC_2D is the case: every estimator here
    # predicts in float64 and is scored against an integer optimum that is
    # exact in TSPLIB's own ``nint`` metric, and the bound deliberately
    # inherits that convention so its MAPE stays comparable. A float64 bound
    # above an nint label is then a unit artefact, and the real certificate
    # check is the same ladder re-run on the instance's own integer matrix.
    # Both are computed; only the metric-consistent one can fail the gate.
    ok["excess_pct"] = 100.0 * (ok.bound - ok.true_cost) / ok.true_cost
    consistent = ok[~ok.corpus.isin(METRIC_TWIN)]
    viol = consistent[consistent.bound > consistent.true_cost * (1.0 + CERT_RTOL)]
    cert = {"checked": int(len(consistent)), "violations": int(len(viol)),
            "max_excess_pct": float(consistent.excess_pct.max()),
            "rtol": CERT_RTOL,
            "corpora": sorted(consistent.corpus.unique().tolist())}
    if len(viol):
        cert["worst"] = (viol.nlargest(10, "excess_pct")
                         [["corpus", "ascent", "instance", "n", "k",
                           "bound", "true_cost", "excess_pct"]]
                         .to_dict("records"))
    report["gate_certificate"] = cert

    # The convention artefact, reported so it is never silently absorbed.
    conv: dict = {}
    for corpus, twin in METRIC_TWIN.items():
        f = ok[ok.corpus == corpus]
        t = ok[ok.corpus == twin]
        if not len(f):
            continue
        over = f[f.bound > f.true_cost * (1.0 + CERT_RTOL)]
        conv[corpus] = {
            "twin": twin,
            "float64_rows_above_label": int(len(over)),
            "float64_instances_above_label": sorted(over.instance.unique().tolist()),
            "float64_max_excess_pct": float(f.excess_pct.max()),
            "twin_rows_checked": int(len(t)),
            "twin_violations": int((t.bound > t.true_cost * (1.0 + CERT_RTOL)).sum()),
            "twin_max_excess_pct": (float(t.excess_pct.max()) if len(t)
                                    else float("nan")),
            "twin_instances": int(t.instance.nunique()) if len(t) else 0,
            "reading": ("every float64 row above the label is at or below it in "
                        "the instance's own nint metric; the bound is not "
                        "violated" if len(t) and
                        not (t.bound > t.true_cost * (1.0 + CERT_RTOL)).any()
                        else "UNRESOLVED: the metric twin does not clear it"),
        }
        # No violation tally is added here: the twin corpus is metric-consistent
        # and is already inside ``consistent`` above, so Gate 1 has counted it.
    report["gate_metric_convention"] = conv

    # -- Gate 2: exact DP at small n ---------------------------------------
    small = []
    for corpus in ("2d",):
        tasks = [t for t in CORPORA[corpus][0]() if t["n"] <= EXACT_DP_MAX_N]
        for t in tasks:
            X = np.unique(np.asarray(t["coords"], dtype=np.float64), axis=0)
            if X.shape[0] < 3:
                continue
            sq = (X * X).sum(axis=1)
            D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
            np.maximum(D, 0.0, out=D)
            np.sqrt(D, out=D)
            np.fill_diagonal(D, 0.0)
            small.append({"corpus": corpus, "instance": t["instance"],
                          "n": int(X.shape[0]), "opt": _exact_optimum_matrix(D)})
    dp = pd.DataFrame(small)
    dp_report = {"instances": int(len(dp)), "max_n": EXACT_DP_MAX_N}
    if len(dp):
        m = ok.merge(dp[["corpus", "instance", "opt"]], on=["corpus", "instance"])
        m["excess_vs_exact_pct"] = 100.0 * (m.bound - m.opt) / m.opt
        bad = m[m.bound > m.opt * (1.0 + CERT_RTOL)]
        dp_report.update({
            "ladder_rows_checked": int(len(m)),
            "violations": int(len(bad)),
            "max_excess_vs_exact_pct": float(m.excess_vs_exact_pct.max()),
            "label_vs_exact_max_abs_pct": float(
                (100.0 * (m.true_cost - m.opt) / m.opt).abs().max()),
        })
        if len(bad):
            dp_report["worst"] = (bad.nlargest(10, "excess_vs_exact_pct")
                                  [["corpus", "ascent", "instance", "n", "k",
                                    "bound", "opt"]].to_dict("records"))
    report["gate_exact_dp"] = dp_report

    # -- Gate 3: monotone in k ---------------------------------------------
    s = ok.sort_values(["corpus", "ascent", "instance", "k"])
    d = s.groupby(["corpus", "ascent", "instance"]).bound.diff()
    nonmono = s[d < -CERT_RTOL * s.bound.abs().clip(lower=1.0)]
    report["gate_monotone"] = {
        "series": int(s.groupby(["corpus", "ascent", "instance"]).ngroups),
        "violations": int(len(nonmono)),
        "min_step": float(d.min()) if len(d.dropna()) else 0.0,
    }

    report["PASS"] = bool(cert["violations"] == 0
                          and dp_report.get("violations", 0) == 0
                          and report["gate_monotone"]["violations"] == 0)

    # -- Coverage census ----------------------------------------------------
    report["coverage"] = (all_df.groupby(["corpus", "ascent"])
                          .agg(instances=("instance", "nunique"),
                               rows=("k", "size"),
                               n_min=("n", "min"), n_max=("n", "max"))
                          .reset_index().to_dict("records"))

    out = OUT_DIR / "hk1tree_allbench_gates.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "coverage"}, indent=1))
    print(f"\nwrote {out}")
    return report


def _all_pairs() -> list[tuple[str, str]]:
    """Every (ascent, corpus) with a ladder on disk, this script's or earlier."""
    return JOBS + [("vj", "2d"), ("vj", "tsplib"), ("vj", "nd"),
                   ("polyak", "nd"), ("vj", "tsplib_int")]


#: Ladders that predate this script keep the filenames they were published
#: under. Renaming them would orphan every artifact that already cites them.
LEGACY_PATHS = {
    ("vj", "2d"): "hk1tree_frontier_2d.csv",
    ("vj", "tsplib"): "hk1tree_frontier_tsplib.csv",
    ("vj", "tsplib_int"): "hk1tree_frontier_tsplib_int.csv",
    ("vj", "nd"): "hk1tree_frontier_nd.csv",
    ("polyak", "nd"): "polyak_nd_sweep.csv",
}


def _out_path(ascent: str, corpus: str) -> Path:
    legacy = LEGACY_PATHS.get((ascent, corpus))
    return OUT_DIR / (legacy or f"hk1tree_{ascent}_{corpus}.csv")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="all",
                    choices=["all", "2d", "tsplib", "tsplib_int", "noneuc"])
    ap.add_argument("--ascent", default="all", choices=["all", "vj", "polyak"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute a ladder whose CSV already exists")
    args = ap.parse_args()

    if args.verify:
        verify()
        return
    if args.gates:
        rep = gates()
        raise SystemExit(0 if rep["PASS"] else 1)

    jobs = [(a, c) for a, c in JOBS
            if (args.corpus in ("all", c) and args.ascent in ("all", a))]
    if not jobs:
        raise SystemExit(f"no job for corpus={args.corpus} ascent={args.ascent}")

    loaded: dict[str, list[dict]] = {}
    for ascent, corpus in jobs:
        out = _out_path(ascent, corpus)
        if out.exists() and not args.force:
            print(f"[{ascent}/{corpus}] {out.name} exists, skipping "
                  f"(--force to recompute)")
            continue
        if corpus not in loaded:
            t0 = time.perf_counter()
            loaded[corpus] = CORPORA[corpus][0]()
            print(f"[load] {corpus}: {len(loaded[corpus])} instances in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        df = _parallel(loaded[corpus], RUNNERS[ascent], args.workers,
                       f"{ascent}/{corpus}")
        df.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(df)} rows)\n", flush=True)


if __name__ == "__main__":
    main()
