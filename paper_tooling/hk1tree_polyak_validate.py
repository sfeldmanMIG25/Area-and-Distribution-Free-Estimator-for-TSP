"""Invariant checks for the Polyak 1-tree ascent, and the head-to-head against
the shipped Volgenant--Jonker ascent.

The Polyak ascent is admitted to the ND sweep only if it passes the same bar
the shipped ascent passed:

  I1  bound <= OPT on every instance with an independently computed optimum.
      Two independent optimum computations are used and cross-checked against
      each other -- exhaustive permutation search at n <= 9 and the
      Bellman--Held--Karp DP at n <= 13 -- so a shared bug cannot hide.
  I2  the incumbent is non-decreasing in the budget k.
  I3  w(0) >= L_MST, against the project's own ``mst_utils.compute_mst``.
  I4  the constructive upper bound is a real tour cost: UB >= OPT wherever OPT
      is known, and UB >= bound always.
  I5  the k-prefix property: an ascent run to budget k returns exactly what a
      budget-K ascent reports at checkpoint k. Without it the checkpointed
      sweep is not measuring what it claims.

Also produced here, because they are the two things a reader will ask about
the step rule itself:

  * the underflow diagnosis for the shipped ascent, measured rather than
    asserted (``--diagnose``);
  * Polyak against V&J at equal budget on the same instances (``--headtohead``).

    python paper_tooling/hk1tree_polyak_validate.py --all
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

from held_karp_1tree import one_tree_bound  # noqa: E402
from hk1tree_polyak import (  # noqa: E402
    brute_force_optimum,
    exact_optimum,
    polyak_bound,
)
from mst_utils import compute_mst  # noqa: E402

OUT = ROOT / "paper_tooling"
LADDER = (0, 10, 25, 50, 100, 200, 500, 1000, 2000)
DIMS = (2, 3, 5, 10, 20, 50, 100)


def make_case(rng, n: int, d: int, style: int) -> np.ndarray:
    X = rng.random((n, d)) * 1000.0
    if style == 1:                                    # clustered
        c = rng.random((max(2, n // 3), d)) * 1000.0
        X = c[rng.integers(0, len(c), n)] + rng.normal(0, 15, (n, d))
    elif style == 2:                                  # near-degenerate line
        X = np.tile(rng.random((n, 1)) * 1000.0, (1, d)) + rng.normal(0, 0.5, (n, d))
    elif style == 3:                                  # integer grid, many ties
        X = rng.integers(0, 12, (n, d)).astype(np.float64)
    return X


# ---------------------------------------------------------------------------
# I1 / I4 -- bound <= OPT, UB >= OPT, on independently solved instances
# ---------------------------------------------------------------------------
def check_optima(budget: int = 2000, seed: int = 20260814) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for d in DIMS:
        for n in range(5, 14):
            for style in range(4):
                X = make_case(rng, n, d, style)
                Xu = np.unique(X, axis=0)
                if Xu.shape[0] < 4:
                    continue
                dp = exact_optimum(Xu)
                bf = brute_force_optimum(Xu) if Xu.shape[0] <= 9 else float("nan")
                r = polyak_bound(Xu, budget, checkpoints=LADDER)
                if r["status"] != "ok":
                    continue
                rows.append({
                    "d": d, "n": int(Xu.shape[0]), "style": style,
                    "opt_dp": dp, "opt_bf": bf, "bound": r["bound"],
                    "w0": r["initial_bound"], "ub": r["upper_bound"],
                    "iters": r["iterations_used"], "is_opt": r["is_optimal"],
                    "stopped": r["stopped_reason"],
                    "viol_pct": 100.0 * (r["bound"] - dp) / dp,
                    "gap_pct": 100.0 * (dp - r["bound"]) / dp,
                    "ub_excess_pct": 100.0 * (r["upper_bound"] - dp) / dp,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "polyak_validate_optima.csv", index=False)

    bf = df.dropna(subset=["opt_bf"])
    cross = float(np.max(np.abs(bf.opt_dp - bf.opt_bf) / bf.opt_bf)) if len(bf) else 0.0
    return {
        "N_cases": int(len(df)),
        "dp_vs_bruteforce_max_rel_diff": cross,
        "I1_max_violation_pct": float(df.viol_pct.max()),
        "I1_violations_over_1e-9pct": int((df.viol_pct > 1e-9).sum()),
        "I4_min_ub_excess_pct": float(df.ub_excess_pct.min()),
        "I4_ub_below_opt": int((df.ub_excess_pct < -1e-9).sum()),
        "I4_ub_below_bound": int((df.ub < df.bound - 1e-9).sum()),
        "median_gap_pct": float(df.gap_pct.median()),
        "frac_closed_exactly_pct": float(100.0 * (df.gap_pct.abs() < 1e-9).mean()),
        "gap_pct_by_d": {int(k): float(v) for k, v in
                         df.groupby("d").gap_pct.median().items()},
        "ub_excess_pct_by_d": {int(k): float(v) for k, v in
                               df.groupby("d").ub_excess_pct.median().items()},
    }


# ---------------------------------------------------------------------------
# I2 / I3 / I5 -- monotone in k, w(0) >= L_MST, prefix property
# ---------------------------------------------------------------------------
def check_structure(seed: int = 20260815, n_cases: int = 60) -> dict:
    rng = np.random.default_rng(seed)
    mono_bad = mst_bad = prefix_bad = 0
    worst_mst = np.inf
    checked = 0
    for i in range(n_cases):
        n = int(rng.integers(20, 260))
        d = int(rng.choice(DIMS))
        X = make_case(rng, n, d, i % 4)
        Xu = np.ascontiguousarray(np.unique(X, axis=0))
        if Xu.shape[0] < 5:
            continue
        r = polyak_bound(Xu, max(LADDER), checkpoints=LADDER)
        if r["status"] != "ok":
            continue
        checked += 1
        seq = [r["bounds"][k] for k in LADDER]
        if any(seq[j + 1] < seq[j] - 0.0 for j in range(len(seq) - 1)):
            mono_bad += 1

        lmst = compute_mst(Xu).total_length
        if r["initial_bound"] < lmst - 1e-9 * max(1.0, lmst):
            mst_bad += 1
        worst_mst = min(worst_mst, (r["initial_bound"] - lmst) / max(1e-12, lmst))

        # I5: independent short runs must equal the checkpoint read-off.
        for k in (10, 100, 500):
            s = polyak_bound(Xu, k)
            if s["bound"] != r["bounds"][k]:
                prefix_bad += 1
    return {"N_cases": checked,
            "I2_monotonicity_failures": mono_bad,
            "I3_w0_below_LMST_failures": mst_bad,
            "I3_min_relative_slack_w0_over_LMST": float(worst_mst),
            "I5_prefix_mismatches": prefix_bad}


# ---------------------------------------------------------------------------
# The underflow diagnosis for the shipped ascent
# ---------------------------------------------------------------------------
def diagnose(seed: int = 20260816) -> dict:
    """Show that the V&J plateau is the step underflowing, not convergence."""
    from held_karp_1tree import INITIAL_PERIOD

    rng = np.random.default_rng(seed)
    rows = []
    for d in (2, 10, 20, 50, 100):
        for rep in range(4):
            X = rng.random((120, d)) * 1000.0
            for k in (2000, 8000):
                r = one_tree_bound(X, k)
                rows.append({"d": d, "rep": rep, "budget": k, "bound": r.bound,
                             "iters": r.iterations_used})
    df = pd.DataFrame(rows)
    p = df.pivot_table(index=["d", "rep"], columns="budget",
                       values=["bound", "iters"]).reset_index()
    identical = int(np.sum(np.isclose(p[("bound", 2000)], p[("bound", 8000)],
                                      rtol=0, atol=0)))
    # Predicted stall: 6 shrinking periods to reach length 1, then one halving
    # per iteration until t underflows from t0 = w(0)/(2n).
    periods_to_one = INITIAL_PERIOD.bit_length() + 1
    return {"N_pairs": int(len(p)),
            "pairs_with_identical_bound_at_k2000_and_k8000": identical,
            "median_iterations_used_at_budget_8000":
                float(df[df.budget == 8000].iters.median()),
            "iterations_used_by_d_at_budget_8000":
                {int(k): float(v) for k, v in
                 df[df.budget == 8000].groupby("d").iters.median().items()},
            "predicted_stall_iterations":
                f"~{periods_to_one} + ~1080 halvings to float64 underflow",
            "note": ("a budget of 8000 buys nothing over 2000 because the "
                     "V&J step reaches 0.0 and the loop's t > 0 guard fires")}


# ---------------------------------------------------------------------------
# Polyak against the shipped ascent, equal budget, same instances
# ---------------------------------------------------------------------------
def head_to_head(seed: int = 20260817, per_cell: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for d in DIMS:
        for n in (40, 120, 250):
            for rep in range(per_cell):
                X = rng.random((n, d)) * 1000.0
                vj = {k: one_tree_bound(X, k).bound for k in (100, 500, 2000)}
                pk = polyak_bound(X, 2000, checkpoints=(100, 500, 2000))
                for k in (100, 500, 2000):
                    rows.append({"d": d, "n": n, "rep": rep, "k": k,
                                 "vj": vj[k], "polyak": pk["bounds"][k],
                                 "ub": pk["upper_bound"]})
    df = pd.DataFrame(rows)
    df["polyak_higher"] = df.polyak > df.vj + 1e-9
    df["rel_gain_pct"] = 100.0 * (df.polyak - df.vj) / df.vj
    df.to_csv(OUT / "polyak_vs_vj.csv", index=False)
    out = {}
    for k, g in df.groupby("k"):
        out[int(k)] = {
            "N": int(len(g)),
            "polyak_higher_pct": float(100.0 * g.polyak_higher.mean()),
            "median_rel_gain_pct": float(g.rel_gain_pct.median()),
            "by_d_median_rel_gain_pct":
                {int(a): float(b) for a, b in
                 g.groupby("d").rel_gain_pct.median().items()},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--optima", action="store_true")
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--headtohead", action="store_true")
    a = ap.parse_args()
    run_all = a.all or not any((a.optima, a.structure, a.diagnose, a.headtohead))

    # Merge rather than overwrite: the four blocks are independently runnable
    # and each costs minutes, so a partial re-run must not drop the others.
    p_out = OUT / "polyak_validation.json"
    report = json.loads(p_out.read_text()) if p_out.exists() else {}
    if run_all or a.diagnose:
        print("[diagnose] shipped V&J step underflow ...", flush=True)
        report["vj_underflow_diagnosis"] = diagnose()
        print(json.dumps(report["vj_underflow_diagnosis"], indent=2))
    if run_all or a.optima:
        print("[I1/I4] bound <= OPT on independently solved instances ...", flush=True)
        report["optima"] = check_optima()
        print(json.dumps(report["optima"], indent=2))
    if run_all or a.structure:
        print("[I2/I3/I5] monotone / MST floor / prefix ...", flush=True)
        report["structure"] = check_structure()
        print(json.dumps(report["structure"], indent=2))
    if run_all or a.headtohead:
        print("[head-to-head] Polyak vs Volgenant--Jonker ...", flush=True)
        report["head_to_head"] = head_to_head()
        print(json.dumps(report["head_to_head"], indent=2))

    (OUT / "polyak_validation.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'polyak_validation.json'}")


if __name__ == "__main__":
    main()
