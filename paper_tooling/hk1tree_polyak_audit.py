"""Two audits the ND Polyak result must survive, plus the mechanism check.

A1  LABEL-METRIC AUDIT. A bound above the released label looks like a broken
    certificate. It need not be one: the ND label is a solver tour scored in
    the *scaled integer* metric the solver was handed
    (``concorde_integer_length / scale_factor``, ``scale`` from
    ``solvers/config.get_scale_factor``), while the bound is computed in
    float64 on the released coordinates. Those are different metrics. This
    audit recomputes the stored tour's length in float64 -- the metric the
    bound lives in -- and asks whether the bound is above *that*. A bound above
    the float64 tour length is a real violation; a bound above the integer
    label alone is a unit mismatch, and the same one every estimator in the
    benchmark inherits.

A2  CONVERGENCE AUDIT. The shipped Volgenant--Jonker ascent stalls because its
    step underflows. Replacing it with a rule that stalls on an explicit
    ``gamma`` floor instead of an implicit denormal would be the same mistake
    with better manners. This audit re-runs each instance with the floor lowered
    by 40 halvings (``GAMMA_MIN`` 1e-12 -> 1e-24) and the budget raised tenfold
    (2000 -> 20000), and reports how much the bound moves. If it moves, the
    sweep is under-converged and the reported accuracy is a floor, not a value.

A3  MECHANISM. Why the relaxation is nearly exact in high dimension: the
    fraction of instances whose minimum 1-tree becomes a Hamiltonian cycle
    (relaxation provably tight, gap exactly zero), against dimension, alongside
    the relative spread of pairwise distances, which falls as ``O(1/sqrt(d))``.

    python paper_tooling/hk1tree_polyak_audit.py --all
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "paper_tooling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hk1tree_polyak import polyak_bound  # noqa: E402
from hk1tree_polyak_nd import _load_bin, load_base, stratified, tasks_from  # noqa: E402

OUT = ROOT / "paper_tooling"
SOLUTIONS = ROOT / "solutions"


# ---------------------------------------------------------------------------
# A1 -- is a bound above the label above the tour it is scored against?
# ---------------------------------------------------------------------------
def tour_length_float64(coords: np.ndarray, tour: list[int]) -> float:
    X = np.asarray(coords, dtype=np.float64)
    t = np.asarray(tour, dtype=np.int64)
    if t.shape[0] != X.shape[0] or len(np.unique(t)) != X.shape[0]:
        return float("nan")          # stored tour does not match the coordinates
    # LKH writes 1-based node ids, Concorde 0-based; the two solvers' tours sit
    # in the same field. Detect rather than assume.
    if t.min() == 1 and t.max() == X.shape[0]:
        t = t - 1
    elif t.min() != 0 or t.max() != X.shape[0] - 1:
        return float("nan")
    seg = X[t] - X[np.roll(t, -1)]
    return float(np.sqrt((seg * seg).sum(axis=1)).sum())


def audit_labels(sweep_csv: Path, k: int) -> dict:
    """Every row above its label, re-scored against the float64 tour."""
    d = pd.read_csv(sweep_csv, float_precision="round_trip")
    d = d[(d.k == k) & (d.status == "ok")].copy()
    d["e_pct"] = 100.0 * (d.bound - d.true_cost) / d.true_cost
    above = d[d.e_pct > 0].copy()

    base = load_base().set_index("instance")
    rows = []
    for r in above.itertuples(index=False):
        meta = base.loc[r.instance]
        sol = json.loads((SOLUTIONS / f"{r.instance}.sol.json").read_text())
        X = _load_bin(Path(meta.file_path).with_suffix(".bin"),
                      meta.n_customers, meta.dimension, meta.grid_size)
        f64 = tour_length_float64(X, sol["optimal_tour"])
        rows.append({"instance": r.instance, "n": r.n, "d": r.d,
                     "solver": sol.get("optimal_solver"),
                     "bound": r.bound, "label": r.true_cost,
                     "tour_float64": f64,
                     "e_vs_label_pct": r.e_pct,
                     "e_vs_float64_tour_pct":
                         100.0 * (r.bound - f64) / f64 if f64 == f64 else float("nan")})
    a = pd.DataFrame(rows)
    if len(a):
        a.to_csv(OUT / f"polyak_nd_label_audit_k{k}.csv", index=False)

    real = a[a.e_vs_float64_tour_pct > 1e-9] if len(a) else a
    unresolved = a[a.tour_float64.isna()] if len(a) else a
    return {
        "k": k,
        "N_scored": int(len(d)),
        "N_above_integer_label": int(len(above)),
        "pct_above_integer_label": float(100.0 * len(above) / max(1, len(d))),
        "max_excess_over_integer_label_pct":
            float(above.e_pct.max()) if len(above) else 0.0,
        "N_above_float64_tour": int(len(real)),
        "max_excess_over_float64_tour_pct":
            float(real.e_vs_float64_tour_pct.max()) if len(real) else 0.0,
        "N_stored_tour_unusable": int(len(unresolved)),
        "solver_of_rows_above_label":
            a.solver.value_counts().to_dict() if len(a) else {},
        "verdict": ("every row above the integer label is at or below the same "
                    "tour's float64 length -- a unit mismatch, not a broken bound"
                    if len(real) == 0 else
                    f"{len(real)} rows exceed the float64 tour length: REAL VIOLATION"),
    }


# ---------------------------------------------------------------------------
# A2 -- does restarting gamma buy anything the floor is hiding?
# ---------------------------------------------------------------------------
def audit_convergence(per_cell: int = 8, seed: int = 20260821) -> dict:
    """Lower the floor and lengthen the budget; measure what the bound gains."""
    import hk1tree_polyak as hp

    base = load_base()
    sel = stratified(base, per_cell, seed, n_lo=40, n_hi=250)
    tasks = tasks_from(sel)

    rows = []
    saved = hp.GAMMA_MIN
    for t in tasks:
        ref = polyak_bound(t["coords"], 2000)
        if ref["status"] != "ok":
            continue
        hp.GAMMA_MIN = 1e-24               # 40 more halvings before the floor
        deep = polyak_bound(t["coords"], 20000)
        hp.GAMMA_MIN = saved
        rows.append({"instance": t["instance"], "n": t["n"], "d": t["d"],
                     "true_cost": t["true_cost"],
                     "k2000": ref["bound"], "deep": deep["bound"],
                     "k2000_iters": ref["iterations_used"],
                     "deep_iters": deep["iterations_used"],
                     "gain_pct": 100.0 * (deep["bound"] - ref["bound"]) / ref["bound"],
                     "ape_k2000": 100.0 * abs(t["true_cost"] - ref["bound"]) / t["true_cost"],
                     "ape_deep": 100.0 * abs(t["true_cost"] - deep["bound"]) / t["true_cost"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "polyak_nd_convergence_audit.csv", index=False)
    return {
        "N": int(len(df)),
        "protocol": ("published run: budget 2000, GAMMA_MIN 1e-12. deep run: "
                     "budget 20000, GAMMA_MIN 1e-24 -- 40 further halvings"),
        "median_gain_pct": float(df.gain_pct.median()),
        "max_gain_pct": float(df.gain_pct.max()),
        "MAPE_published": float(df.ape_k2000.mean()),
        "MAPE_deep": float(df.ape_deep.mean()),
        "MAPE_by_d_published": {int(a): float(b) for a, b in
                                df.groupby("d").ape_k2000.mean().items()},
        "MAPE_by_d_deep": {int(a): float(b) for a, b in
                           df.groupby("d").ape_deep.mean().items()},
        "median_iters_published": float(df.k2000_iters.median()),
        "median_iters_deep": float(df.deep_iters.median()),
    }


# ---------------------------------------------------------------------------
# A3 -- the mechanism
# ---------------------------------------------------------------------------
def audit_mechanism(per_cell: int = 24, seed: int = 20260822) -> dict:
    """Tightness rate and distance concentration against dimension."""
    base = load_base()
    sel = stratified(base, per_cell, seed, n_lo=40, n_hi=250)
    tasks = tasks_from(sel)

    rows = []
    for t in tasks:
        r = polyak_bound(t["coords"], 2000)
        if r["status"] != "ok":
            continue
        X = np.unique(np.asarray(t["coords"], dtype=np.float64), axis=0)
        sq = (X * X).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * (X @ X.T), 0.0))
        iu = np.triu_indices(X.shape[0], 1)
        pd_ = D[iu]
        nn = np.partition(D + np.eye(X.shape[0]) * 1e18, 1, axis=1)[:, 1]
        rows.append({"instance": t["instance"], "n": t["n"], "d": t["d"],
                     "is_tight": bool(r["is_optimal"]),
                     "ape": 100.0 * abs(t["true_cost"] - r["bound"]) / t["true_cost"],
                     "cv_pairwise": float(pd_.std() / pd_.mean()),
                     "nn_over_mean": float(nn.mean() / pd_.mean()),
                     "w0_gap_pct": 100.0 * (t["true_cost"] - r["initial_bound"])
                     / t["true_cost"] if "initial_bound" in r else float("nan")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "polyak_nd_mechanism.csv", index=False)
    by_d = df.groupby("d").agg(N=("ape", "size"), tight_pct=("is_tight", "mean"),
                               MAPE=("ape", "mean"), cv=("cv_pairwise", "mean"),
                               nn_ratio=("nn_over_mean", "mean"))
    by_d["tight_pct"] *= 100.0
    return {"note": ("tight_pct = share of instances whose minimum 1-tree is a "
                     "Hamiltonian cycle, i.e. the relaxation closes exactly; "
                     "cv = coefficient of variation of pairwise distances, the "
                     "concentration the O(1/sqrt(d)) argument predicts"),
            "by_d": {int(k): {c: float(v) for c, v in r.items()}
                     for k, r in by_d.iterrows()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--labels", type=str, default=None,
                    help="path to a sweep csv to audit")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--mechanism", action="store_true")
    a = ap.parse_args()

    p_out = OUT / "polyak_audits.json"
    rep = json.loads(p_out.read_text()) if p_out.exists() else {}

    if a.labels:
        rep[f"A1_label_audit_k{a.k}"] = audit_labels(Path(a.labels), a.k)
        print(json.dumps(rep[f"A1_label_audit_k{a.k}"], indent=2))
    if a.all or a.convergence:
        print("[A2] convergence ...", flush=True)
        rep["A2_convergence"] = audit_convergence()
        print(json.dumps(rep["A2_convergence"], indent=2))
    if a.all or a.mechanism:
        print("[A3] mechanism ...", flush=True)
        rep["A3_mechanism"] = audit_mechanism()
        print(json.dumps(rep["A3_mechanism"], indent=2))

    p_out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {p_out}")


if __name__ == "__main__":
    main()
