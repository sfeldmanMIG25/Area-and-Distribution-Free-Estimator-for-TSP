"""Size-stratified cost/accuracy of GART 2.0 against the Polyak 1-tree ascent.

Why this exists
---------------
``hk1tree_polyak_nd`` reports the multidimensional frontier aggregated over the
whole test split, and stratified by *dimension* only.  That split is 62.7%
instances of at most 100 nodes, a regime in which the relaxation is very nearly
exact and very nearly free, so the aggregate verdict is largely a statement
about small instances.  Nothing in the published tables separates that from a
statement about the estimator.

This module re-reads the same sweep stratified by *size* as well, on the cells
the deployment argument actually rests on, and pairs each cell's accuracy with
a size-matched cost measurement from ``d3_matched_timing``.  It reports, per
(dimension, size band), whether either method dominates the other on both axes
or whether the cell is a genuine trade-off.

Accuracy is computed on every instance of a cell that both methods score; cost
is the median of the matched timing sample for that cell, which is smaller.
The two sample sizes are emitted beside every figure so a reader can see which
number rests on what.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_tooling"

SWEEP = OUT / "polyak_nd_sweep.csv"
TIMING = OUT / "d3_matched_timing_confirm.csv"
BASE_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
BENCH_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"

BANK = OUT / "size_stratified_bank.json"
TIDY = OUT / "size_stratified_cells.csv"

BANDS: tuple[tuple[str, str, int, int], ...] = (
    ("n20_100", "[20,100]", 20, 100),
    ("n200_500", "[200,500]", 200, 500),
    ("n600_1000", "[600,1000]", 600, 1000),
)
BUDGETS: tuple[int, ...] = (25, 50, 100, 200, 500)


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(pred - true) / true))


def main() -> None:
    base = pd.read_csv(BASE_ND)[["instance", "n_customers", "dimension"]]

    g = pd.read_csv(BENCH_ND, low_memory=False)
    g = g[(g.model == "GART_2.0") & (g.status == "ok")]
    g = g.merge(base, on="instance")

    sw = pd.read_csv(SWEEP)
    sw = sw[sw.status == "ok"]

    tm = pd.read_csv(TIMING)

    bank: dict[str, object] = {}
    rows = []

    # Corpus composition -- the reason the aggregate reads as it does.
    nd_all = g.copy()
    frac_small = 100.0 * float((nd_all.n_customers <= 100).mean())
    bank["corpus/scored_instances"] = int(len(nd_all))
    bank["corpus/frac_n_le_100_pct"] = round(frac_small, 2)
    bank["corpus/n_le_100_instances"] = int((nd_all.n_customers <= 100).sum())

    dims = sorted(tm.d.unique())
    for d in dims:
        for key, label, lo, hi in BANDS:
            tc = tm[(tm.d == d) & (tm.band == label)]
            if tc.empty:
                continue
            gc = g[(g.dimension == d) & (g.n_customers >= lo) & (g.n_customers <= hi)]
            sc = sw[(sw.d == d) & (sw.n >= lo) & (sw.n <= hi)]
            if gc.empty or sc.empty:
                continue

            # Pair the two on the instances both actually score.
            common = set(gc.instance) & set(sc.instance)
            if not common:
                continue
            gc = gc[gc.instance.isin(common)]
            g_mape = mape(gc.pred_cost.to_numpy(), gc.true_cost.to_numpy())
            g_ms = float(tc.gart_ms.median())

            pre = f"cell/d{d}/{key}"
            bank[f"{pre}/n_accuracy"] = int(len(gc))
            bank[f"{pre}/n_timing"] = int(len(tc))
            bank[f"{pre}/gart_mape_pct"] = round(g_mape, 4)
            bank[f"{pre}/gart_ms"] = round(g_ms, 3)

            dominated_by, dominates = [], []
            for k in BUDGETS:
                col = f"hk_k{k}_ms"
                if col not in tc.columns:
                    continue
                r = sc[(sc.k == k) & (sc.instance.isin(common))]
                if r.empty:
                    continue
                b_mape = mape(r.bound.to_numpy(), r.true_cost.to_numpy())
                b_ms = float(tc[col].median())
                bank[f"{pre}/bound_k{k}_mape_pct"] = round(b_mape, 4)
                bank[f"{pre}/bound_k{k}_ms"] = round(b_ms, 3)
                bank[f"{pre}/bound_k{k}_cost_x"] = round(b_ms / g_ms, 3)

                if b_ms < g_ms and b_mape < g_mape:
                    dominated_by.append(k)
                    verdict = "bound dominates"
                elif b_ms > g_ms and b_mape > g_mape:
                    dominates.append(k)
                    verdict = "GART dominates"
                else:
                    verdict = "trade-off"
                rows.append({"d": d, "band": label, "k": k,
                             "n_accuracy": len(gc), "n_timing": len(tc),
                             "gart_mape_pct": g_mape, "gart_ms": g_ms,
                             "bound_mape_pct": b_mape, "bound_ms": b_ms,
                             "cost_x": b_ms / g_ms, "verdict": verdict})

            bank[f"{pre}/gart_dominated"] = bool(dominated_by)
            bank[f"{pre}/budgets_dominating_gart"] = dominated_by
            bank[f"{pre}/budgets_gart_dominates"] = dominates

    tidy = pd.DataFrame(rows)
    tidy.to_csv(TIDY, index=False, float_format="%.6g")

    # Headline: at which (d, band) is GART on the front?
    front = (tidy.groupby(["d", "band"])["verdict"]
             .apply(lambda s: "bound dominates" not in set(s)))
    bank["summary/cells_gart_not_dominated"] = int(front.sum())
    bank["summary/cells_total"] = int(len(front))
    big = front[[i[1] == "[600,1000]" for i in front.index]]
    bank["summary/large_n_cells_gart_not_dominated"] = int(big.sum())
    bank["summary/large_n_cells_total"] = int(len(big))

    BANK.write_text(json.dumps(bank, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {BANK}  ({len(bank)} keys)")
    print(f"wrote {TIDY}  ({len(tidy)} rows)")
    print()
    print(tidy.pivot_table(index=["d", "band"], columns="k",
                           values="verdict", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
