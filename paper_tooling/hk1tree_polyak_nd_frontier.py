"""Assemble the ND cost/accuracy frontier and bank every number it reports.

Accuracy comes from ``polyak_nd_sweep.csv`` (all 16,920 ND test instances).
Cost comes from ``polyak_nd_timing.csv`` (stratified sample, serial, single
thread, GART 2.0 re-measured in the same process so the ratio is meaningful).

TWO WEIGHTINGS, BOTH REPORTED
-----------------------------
The timing sample is stratified *per dimension*, 8 instances each, so it holds
d = 100 at 5.6 % where the corpus holds it at 34.9 %. The unweighted sample
median answers "what does a dimension cost"; the corpus-weighted median answers
"what does the ND benchmark cost". They differ and both are written out.

Outputs: polyak_nd_frontier.csv, polyak_nd_frontier_by_group.csv,
polyak_nd_bank.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "paper_tooling"

SWEEP = OUT / "polyak_nd_sweep.csv"
TIMING = OUT / "polyak_nd_timing.csv"
REF_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
BASE_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"

TIMED_KS = (0, 10, 25, 50, 100, 200, 500)
ALL_KS = (0, 10, 25, 50, 100, 200, 500, 1000, 2000)


def group_of(d: int) -> str:
    if d <= 3:
        return "d in {2,3}"
    if d <= 10:
        return "d in [4,10]"
    if d <= 50:
        return "d in [15,50]"
    return "d = 100"


GROUP_ORDER = ["d in {2,3}", "d in [4,10]", "d in [15,50]", "d = 100", "all ND"]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    v, w = np.asarray(values)[o], np.asarray(weights)[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


VJ_SWEEP = OUT / "hk1tree_frontier_nd.csv"
TOUR_AUDIT = OUT / "reference_tour_audit.csv"


def _envelope_and_robustness(sw: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Two things a reader will ask that the headline table cannot answer.

    *Envelope*: neither ascent is proved to attain ``max_pi w(pi)``, so the
    per-instance maximum of the Polyak and Volgenant--Jonker trajectories is a
    strictly better estimate of the Held--Karp bound than either alone. It is
    reported as a check on the Polyak column, not as the shipped method -- it
    costs two ascents.

    *Robustness*: 184 instances corpus-wide have a stored reference tour
    inconsistent with their released coordinates. 74 fall in the ND test split.
    Dropping them must not move the conclusion.
    """
    out: dict = {}
    if VJ_SWEEP.exists():
        vj = pd.read_csv(VJ_SWEEP, float_precision="round_trip")
        vj = vj[vj.status == "ok"]
        j = sw[["instance", "d", "k", "bound", "true_cost"]].merge(
            vj[["instance", "k", "bound"]], on=["instance", "k"],
            suffixes=("_pk", "_vj"))
        j["env"] = np.maximum(j.bound_pk, j.bound_vj)
        j["ape_env"] = 100.0 * (j.true_cost - j.env).abs() / j.true_cost
        g = j[j.k == 2000]
        out["two_ascent_envelope"] = {
            "what": "per-instance max of the Polyak and V&J bounds; costs two ascents",
            "MAPE_pct_by_k": {int(k): float(gg.ape_env.mean())
                              for k, gg in j.groupby("k")},
            "vj_higher_than_polyak_pct_at_k2000":
                float(100.0 * (g.bound_vj > g.bound_pk + 1e-9).mean()),
            "vj_higher_pct_by_d_at_k2000":
                {int(a): float(100.0 * b) for a, b in
                 (g.bound_vj > g.bound_pk + 1e-9).groupby(g.d).mean().items()},
        }
    if TOUR_AUDIT.exists():
        bad = set(pd.read_csv(TOUR_AUDIT).query("bucket == 'corrupt'").instance_name)
        keep, keepr = ~sw.instance.isin(bad), ~ref.instance.isin(bad)
        out["corrupt_reference_tour_cut"] = {
            "N_corrupt_in_ND_test": int(sw[sw.k == 0].instance.isin(bad).sum()),
            "GART_2.0_MAPE_pct_excluded": float(ref[keepr].ape.mean()),
            "polyak_MAPE_pct_by_k_excluded":
                {int(k): float(sw[keep & (sw.k == k)].ape.mean()) for k in ALL_KS},
        }
    return out


def main() -> None:
    sw = pd.read_csv(SWEEP, float_precision="round_trip")
    sw = sw[sw.status == "ok"].copy()
    sw["ape"] = 100.0 * (sw.bound - sw.true_cost).abs() / sw.true_cost
    sw["group"] = sw.d.map(group_of)

    ref = pd.read_csv(REF_ND, low_memory=False)
    ref = ref[(ref.model == "GART_2.0") & (ref.status == "ok")].copy()
    meta = pd.read_csv(BASE_ND)[["instance", "dimension"]]
    ref = ref.merge(meta, on="instance")
    ref["ape"] = 100.0 * (ref.pred_cost - ref.true_cost).abs() / ref.true_cost
    ref["group"] = ref.dimension.map(group_of)

    tm = pd.read_csv(TIMING)
    tm["group"] = tm.d.map(group_of)
    for k in TIMED_KS:
        tm[f"x_k{k}"] = tm[f"hk_k{k}_ms"] / tm.gart_ms
    # corpus weights: instances per dimension / corpus size, spread over the
    # sampled instances of that dimension.
    corpus_d = pd.read_csv(BASE_ND).dimension.value_counts()
    sample_d = tm.d.value_counts()
    tm["w"] = tm.d.map(lambda d: corpus_d[d] / len(pd.read_csv(BASE_ND))
                       / sample_d[d]) if False else \
        tm.d.map(lambda d: (corpus_d[d] / corpus_d.sum()) / sample_d[d])

    rows = []
    for g in GROUP_ORDER:
        s = sw if g == "all ND" else sw[sw.group == g]
        r = ref if g == "all ND" else ref[ref.group == g]
        t = tm if g == "all ND" else tm[tm.group == g]
        gart = float(r.ape.mean())
        by_k = {k: float(s[s.k == k].ape.mean()) for k in ALL_KS}
        hit = [k for k in ALL_KS if by_k[k] <= gart]
        kx = hit[0] if hit else None
        row = {"group": g, "N_instances": int(r.shape[0]),
               "GART_2.0_MAPE_pct": gart,
               "GART_2.0_ms": float(t.gart_ms.median()),
               "GART_2.0_ms_corpus_weighted":
                   weighted_median(t.gart_ms.to_numpy(), t.w.to_numpy()),
               "crossover_k": kx}
        for k in ALL_KS:
            row[f"MAPE_k{k}"] = by_k[k]
        for k in TIMED_KS:
            row[f"x_k{k}"] = float(t[f"x_k{k}"].median())
            row[f"ms_k{k}"] = float(t[f"hk_k{k}_ms"].median())
        if kx in TIMED_KS:
            row["crossover_cost_x"] = float(t[f"x_k{kx}"].median())
            row["crossover_ms"] = float(t[f"hk_k{kx}_ms"].median())
        rows.append(row)
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "polyak_nd_frontier_by_group.csv", index=False)

    # Corpus-weighted cost ratios, the honest "what does the ND benchmark cost".
    wq = {f"x_k{k}": weighted_median(tm[f"x_k{k}"].to_numpy(), tm.w.to_numpy())
          for k in TIMED_KS}

    # Pareto: is GART 2.0 dominated inside each group?
    dominated = {}
    for _, r in fr.iterrows():
        strict = [k for k in TIMED_KS
                  if r[f"x_k{k}"] < 1.0 and r[f"MAPE_k{k}"] < r["GART_2.0_MAPE_pct"]]
        dominated[r["group"]] = {
            "strictly_dominating_budgets": [int(k) for k in strict],
            "on_pareto_front": len(strict) == 0,
            "best": ({"k": int(strict[-1]),
                      "cost_x": float(r[f"x_k{strict[-1]}"]),
                      "MAPE_pct": float(r[f"MAPE_k{strict[-1]}"]),
                      "accuracy_factor": float(r["GART_2.0_MAPE_pct"]
                                               / r[f"MAPE_k{strict[-1]}"])}
                     if strict else None)}

    bank = {
        "_what": ("ND cost/accuracy frontier for GART 2.0 against the Held-Karp "
                  "1-tree bound under a Polyak step rule with a constructive "
                  "(nearest-neighbour + 2-opt) upper bound"),
        "_written_by": "paper_tooling/hk1tree_polyak_nd_frontier.py",
        "_sources": {
            "accuracy_1tree": "paper_tooling/polyak_nd_sweep.csv (16920 instances x 9 budgets)",
            "accuracy_GART_2.0": "Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv",
            "cost_both": "paper_tooling/polyak_nd_timing.csv (144 instances, 8 per dimension, serial, 1 thread, 5 repeats, median)",
            "validation": "paper_tooling/polyak_validation.json",
            "audits": "paper_tooling/polyak_audits.json",
            "ub_sensitivity": "paper_tooling/polyak_nd_ub_sensitivity.csv",
        },
        "method": {
            "step_rule": "Polyak: pi += gamma (UB - w(pi)) / ||g||^2 * g",
            "gamma_schedule": "gamma_0 = 2.0, halved after 20 consecutive barren iterations, explicit floor 1e-12",
            "upper_bound_source": "nearest-neighbour tour from node 0, 2-opt to local optimality (max 50 sweeps), computed from coordinates only",
            "label_used_in_ascent": False,
            "budget_independent_schedule": True,
        },
        "corpus": {"name": "ND test split", "N": int(sw.instance.nunique()),
                   "dimensions": sorted(int(x) for x in sw.d.unique()),
                   "n_min": int(sw.n.min()), "n_median": float(sw.n.median()),
                   "n_max": int(sw.n.max())},
        "GART_2.0_MAPE_pct": float(ref.ape.mean()),
        "polyak_MAPE_pct_by_k": {int(k): float(sw[sw.k == k].ape.mean()) for k in ALL_KS},
        "corpus_weighted_cost_ratio_by_k": {int(k.split("_k")[1]): v for k, v in wq.items()},
        "sample_median_cost_ratio_by_k": {int(k): float(tm[f"x_k{k}"].median())
                                          for k in TIMED_KS},
        "sample_median_ms": {"GART_2.0": float(tm.gart_ms.median()),
                             **{f"k{k}": float(tm[f"hk_k{k}_ms"].median())
                                for k in TIMED_KS}},
        "pareto_by_group": dominated,
        "by_group": fr.to_dict(orient="records"),
    }
    bank.update(_envelope_and_robustness(sw, ref))
    (OUT / "polyak_nd_bank.json").write_text(json.dumps(bank, indent=2))

    cols = (["group", "N_instances", "GART_2.0_MAPE_pct", "crossover_k",
             "crossover_cost_x"] +
            [f"MAPE_k{k}" for k in (50, 100, 200, 500, 2000)] +
            [f"x_k{k}" for k in (50, 100, 200, 500)])
    print(fr[cols].round(4).to_string(index=False))
    print("\ncorpus-weighted cost ratio by k:",
          {k: round(v, 3) for k, v in wq.items()})
    print("\npareto:", json.dumps(dominated, indent=2))
    print(f"\nwrote {OUT / 'polyak_nd_bank.json'}")


if __name__ == "__main__":
    main()
