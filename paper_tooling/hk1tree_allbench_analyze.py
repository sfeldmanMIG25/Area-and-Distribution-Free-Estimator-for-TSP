"""Score both 1-tree ascents against GART 2.0 on all four benchmarks.

Consumes the ladders written by ``hk1tree_all_benchmarks.py`` plus the two that
predate it, and emits one bank + one tidy CSV covering every
(corpus, ascent, budget) cell.

Metric definitions are *imported* from ``hk1tree_frontier_analyze`` rather than
restated, so a change there cannot silently desynchronise this file from the
Section~5 numbers it sits beside.

    ...python.exe paper_tooling/hk1tree_allbench_analyze.py

Output: paper_tooling/hk1tree_allbench_bank.json
        paper_tooling/hk1tree_allbench_table.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "paper_tooling", ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hk1tree_all_benchmarks import CHECKPOINTS, _out_path  # noqa: E402
from hk1tree_frontier_analyze import (  # noqa: E402
    REF_MODELS,
    err,
    fit_correction,
    load_ref,
    metrics,
)

OUT = ROOT / "paper_tooling"
TSPLIB_RESULTS = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"

#: The four reporting strata, in the order the manuscript uses. ``nd`` is
#: re-derived here rather than read from its published bank so that all four
#: sit on one instance-intersection rule and the two ascents are compared on
#: identical instance sets; its figures should reproduce the Section~5 ones.
CORPORA_REPORTED = ("2d", "nd", "tsplib", "noneuc")

#: Every (ascent, corpus) cell the paper now reports.
CELLS = [(a, c) for c in CORPORA_REPORTED for a in ("vj", "polyak")]

#: Which train draw each corpus's calibration constant is fitted on. Planar
#: corpora get the planar draw; there is no non-Euclidean training split at all,
#: so ``noneuc`` borrows the planar one and the bank records that it did. The
#: absence is not a gap in the experiment -- it is the point. A constant fitted
#: on Euclidean synthetic data is the only calibration available to *any* model
#: here, and the raw bound needs none.
CORR_SOURCE = {"2d": "train_d2", "tsplib": "train_d2", "noneuc": "train_d2",
               "nd": "train"}


def load_ladder(ascent: str, corpus: str) -> pd.DataFrame:
    p = _out_path(ascent, corpus)
    if not p.exists():
        raise SystemExit(f"missing ladder: {p}")
    d = pd.read_csv(p)
    bad = d[d.status != "ok"]
    if len(bad):
        print(f"  [{ascent}/{corpus}] {bad.instance.nunique()} not scored: "
              f"{bad.status.value_counts().to_dict()}")
    return d[d.status == "ok"].copy()


def load_ref_noneuc() -> pd.DataFrame:
    d = pd.read_csv(TSPLIB_RESULTS, low_memory=False)
    d = d[(d.edge_weight_type != "EUC_2D") & d.model.isin(REF_MODELS)
          & (d.status == "ok")]
    return d[["instance", "model", "pred_cost", "true_cost"]].copy()


def ref_for(corpus: str) -> pd.DataFrame:
    return load_ref_noneuc() if corpus == "noneuc" else load_ref(corpus)


def main() -> None:
    corr = {"train_d2": fit_correction("train_d2"), "train": fit_correction("train")}
    print("\ntrain-fitted c_k = median(true/bound), planar draw:")
    for k in CHECKPOINTS:
        print(f"  k={k:>5d}  c_k={corr['train_d2'][k]['median']:.6f}")

    bank: dict = {"checkpoints": list(CHECKPOINTS), "cells": {}}
    rows: list[dict] = []

    for corpus in CORPORA_REPORTED:
        ladders = {a: load_ladder(a, corpus) for a in ("vj", "polyak")}
        # The two ascents must be scored on the same instances or the
        # comparison between them is confounded by which corpus each saw.
        common = set.intersection(*(set(d.instance) for d in ladders.values()))
        ck = corr[CORR_SOURCE[corpus]]

        ref = ref_for(corpus)
        ref_inst = set(ref[ref.model == "GART_2.0"].instance)
        entry: dict = {
            "N_instances_scored": len(common),
            "correction_source": CORR_SOURCE[corpus],
            "reference_models": {},
            "ascents": {},
        }

        # -- reference estimators, on the instances the bound also scores ----
        for m, g in ref[ref.instance.isin(common)].groupby("model"):
            entry["reference_models"][m] = metrics(err(g.pred_cost, g.true_cost))

        for ascent, d in ladders.items():
            d = d[d.instance.isin(common)]
            raw, scaled = {}, {}
            for k, g in d.groupby("k"):
                e_raw = err(g.bound, g.true_cost)
                e_cal = err(g.bound * ck[int(k)]["median"], g.true_cost)
                raw[int(k)] = metrics(e_raw)
                scaled[int(k)] = metrics(e_cal)
                rows.append({
                    "corpus": corpus, "ascent": ascent, "k": int(k),
                    "N": int(len(g)),
                    "raw_MAPE_pct": raw[int(k)]["MAPE_pct"],
                    "raw_SDPE_pct": raw[int(k)]["SDPE_pct"],
                    "raw_MedAPE_pct": raw[int(k)]["MedAPE_pct"],
                    "scaled_MAPE_pct": scaled[int(k)]["MAPE_pct"],
                    "scaled_SDPE_pct": scaled[int(k)]["SDPE_pct"],
                    "c_k": ck[int(k)]["median"],
                    "frac_at_or_below_label_pct":
                        raw[int(k)]["frac_at_or_below_label_pct"],
                })
            entry["ascents"][ascent] = {
                "raw_MAPE_by_k": {str(k): raw[k]["MAPE_pct"] for k in raw},
                "raw_SDPE_by_k": {str(k): raw[k]["SDPE_pct"] for k in raw},
                "scaled_MAPE_by_k": {str(k): scaled[k]["MAPE_pct"] for k in scaled},
                "raw_full": {str(k): raw[k] for k in raw},
            }

        # -- which ascent wins, per budget -----------------------------------
        piv = {a: entry["ascents"][a]["raw_MAPE_by_k"] for a in ladders}
        entry["ascent_comparison"] = {
            str(k): {"vj_MAPE_pct": piv["vj"][str(k)],
                     "polyak_MAPE_pct": piv["polyak"][str(k)],
                     "better": ("polyak" if piv["polyak"][str(k)] < piv["vj"][str(k)]
                                else "vj")}
            for k in CHECKPOINTS}

        # -- per-instance max of the two ascents, the cost of running both ---
        both = (ladders["vj"][["instance", "k", "bound", "true_cost"]]
                .merge(ladders["polyak"][["instance", "k", "bound"]],
                       on=["instance", "k"], suffixes=("_vj", "_pk")))
        both = both[both.instance.isin(common)]
        both["best"] = both[["bound_vj", "bound_pk"]].max(axis=1)
        entry["max_of_both_MAPE_by_k"] = {
            str(int(k)): float(np.mean(np.abs(err(g.best, g.true_cost))))
            for k, g in both.groupby("k")}
        entry["polyak_strictly_higher_pct_by_k"] = {
            str(int(k)): float(100.0 * np.mean(g.bound_pk > g.bound_vj))
            for k, g in both.groupby("k")}

        if corpus == "noneuc":
            entry.update(_noneuc_detail(ladders, ref, common, ck))

        gart = entry["reference_models"].get("GART_2.0")
        if gart:
            entry["vs_GART2"] = _crux(entry, gart)
            entry["GART2_scored_on"] = len(ref_inst & common)
            entry["like_for_like_vs_GART2"] = _like_for_like(
                ladders, ref, common, ref_inst)
        bank["cells"][corpus] = entry
        print(f"[{corpus}] {len(common)} instances scored by both ascents")

    (OUT / "hk1tree_allbench_bank.json").write_text(
        json.dumps(bank, indent=1), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hk1tree_allbench_table.csv", index=False)
    print(f"\nwrote hk1tree_allbench_bank.json and hk1tree_allbench_table.csv "
          f"({len(df)} cells)")
    _report(bank)


def _crux(entry: dict, gart: dict) -> dict:
    """Cheapest budget at which each ascent, raw and scaled, beats GART 2.0."""
    out: dict = {"GART_2.0_MAPE_pct": gart["MAPE_pct"],
                 "GART_2.0_SDPE_pct": gart["SDPE_pct"]}
    for ascent, a in entry["ascents"].items():
        for tag, key in (("raw", "raw_MAPE_by_k"), ("scaled", "scaled_MAPE_by_k")):
            beats = [k for k in CHECKPOINTS if a[key][str(k)] < gart["MAPE_pct"]]
            out[f"{ascent}_{tag}"] = {
                "best_MAPE_pct": min(a[key].values()),
                "best_k": int(min(a[key], key=lambda z: a[key][z])),
                "first_k_beating_GART2": (int(min(beats)) if beats else None),
            }
    return out


def _like_for_like(ladders: dict, ref: pd.DataFrame, common: set,
                   ref_inst: set) -> dict:
    """Both estimators on exactly the instances both of them score.

    On three of the four strata this is the whole stratum and the block is
    redundant. On the non-Euclidean one it is not: GART 2.0 declines instances
    whose greedy-to-MST ratio falls outside its trained range, and the bound
    scores instances whose ``n`` exceeds this study's matrix budget. Quoting a
    margin across two different instance sets would be the easiest way to
    overstate the result, so the intersection is computed and named.
    """
    both = sorted(common & ref_inst)
    g = ref[(ref.model == "GART_2.0") & ref.instance.isin(both)]
    out: dict = {
        "N": len(both),
        "bound_only": sorted(common - ref_inst),
        "GART2_only": sorted(ref_inst - common),
        "GART_2.0_MAPE_pct": float(np.mean(np.abs(err(g.pred_cost, g.true_cost)))),
    }
    for ascent, d in ladders.items():
        d = d[d.instance.isin(both)]
        out[f"{ascent}_raw_MAPE_by_k"] = {
            str(int(k)): float(np.mean(np.abs(err(x.bound, x.true_cost))))
            for k, x in d.groupby("k")}
    return out


def _noneuc_detail(ladders: dict, ref: pd.DataFrame, common: set,
                   ck: dict) -> dict:
    """The screened/unscreened split, and the per-type breakdown.

    Table~\\ref{tab:tsplib_nonEuc} scores GART 2.0 on the 23 instances left
    after ten structurally nonmetric EXPLICIT matrices are dropped. The bound
    needs no such screen -- the relaxation is valid on any symmetric matrix --
    so both populations are reported and the difference between them is the
    measurement of what the screen costs.
    """
    out: dict = {}
    meta = (ladders["vj"][["instance", "edge_weight_type", "structural_violator", "n"]]
            .drop_duplicates("instance").set_index("instance"))
    screened = set(meta[~meta.structural_violator].index) & common
    out["N_all"] = len(common)
    out["N_screened"] = len(screened)
    out["screened_out"] = sorted(set(common) - screened)

    for tag, pop in (("all", common), ("screened", screened)):
        blk: dict = {}
        for ascent, d in ladders.items():
            g = d[d.instance.isin(pop)]
            blk[ascent] = {
                "raw_MAPE_by_k": {str(int(k)): float(np.mean(np.abs(
                    err(x.bound, x.true_cost)))) for k, x in g.groupby("k")},
                "scaled_MAPE_by_k": {str(int(k)): float(np.mean(np.abs(
                    err(x.bound * ck[int(k)]["median"], x.true_cost))))
                    for k, x in g.groupby("k")},
            }
        r = ref[ref.instance.isin(pop)]
        blk["reference_models"] = {
            m: metrics(err(x.pred_cost, x.true_cost)) for m, x in r.groupby("model")}
        out[f"population_{tag}"] = blk

    # -- by edge-weight type, at the budgets the manuscript quotes -----------
    by_type: dict = {}
    for ascent, d in ladders.items():
        d = d[d.instance.isin(common)]
        for (ewt, k), g in d.groupby(["edge_weight_type", "k"]):
            if int(k) not in (100, 500, 2000):
                continue
            by_type.setdefault(ewt, {}).setdefault(ascent, {})[str(int(k))] = {
                "N": int(len(g)),
                "raw_MAPE_pct": float(np.mean(np.abs(err(g.bound, g.true_cost)))),
            }
    out["by_edge_weight_type"] = by_type
    return out


def _report(bank: dict) -> None:
    print("\n" + "=" * 78)
    for corpus, e in bank["cells"].items():
        g = e["reference_models"].get("GART_2.0", {})
        print(f"\n### {corpus}  (N={e['N_instances_scored']})   "
              f"GART 2.0 MAPE={g.get('MAPE_pct', float('nan')):.4f}%")
        print(f"{'k':>6} | {'VJ raw':>9} {'VJ scaled':>10} | "
              f"{'PK raw':>9} {'PK scaled':>10} | {'better':>7}")
        for k in bank["checkpoints"]:
            a = e["ascents"]
            print(f"{k:>6} | {a['vj']['raw_MAPE_by_k'][str(k)]:>9.4f} "
                  f"{a['vj']['scaled_MAPE_by_k'][str(k)]:>10.4f} | "
                  f"{a['polyak']['raw_MAPE_by_k'][str(k)]:>9.4f} "
                  f"{a['polyak']['scaled_MAPE_by_k'][str(k)]:>10.4f} | "
                  f"{e['ascent_comparison'][str(k)]['better']:>7}")


if __name__ == "__main__":
    main()
