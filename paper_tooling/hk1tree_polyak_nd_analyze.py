"""Score the Polyak ND sweep against GART 2.0 and against the shipped ascent.

Metric definitions are copied from ``hk1tree_frontier_analyze.metrics``, which
copies them from ``build_paper_tables.group_metrics``: ``err_pct = 100 (pred -
true) / true`` and ``MAPE = mean |err_pct|``. Nothing new is defined here, so
the Polyak column is directly comparable to every published ND cell.

Three cuts are produced because the corpus-total figure alone would be
misleading in two separate directions:

* **by dimension**, because the relaxation's gap falls with ``d`` while
  GART 2.0's error also falls with ``d``, at different rates;
* **by n**, because a quarter of the ND split has ``n <= 10``, where the
  Held--Karp relaxation is frequently exact and the comparison is degenerate;
* **on the Concorde-labelled subset only**, because the LKH labels are tours,
  not proven optima, so a gap measured against them is an upper estimate of the
  duality gap and a lower estimate of the bound's accuracy.

    python paper_tooling/hk1tree_polyak_nd_analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "paper_tooling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "paper_tooling"
SWEEP = OUT / "polyak_nd_sweep.csv"
VJ_SWEEP = OUT / "hk1tree_frontier_nd.csv"
REF_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
BASE_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
PROVENANCE = OUT / "nd_label_provenance.csv"

REF_MODELS = ("GART_2.0", "MST_Only", "Asymptotic_MST", "Calibrated_MST_dn", "Hilbert")
N_BUCKETS = [(5, 10), (20, 100), (200, 500), (600, 1000)]


def err(pred, true):
    return 100.0 * (np.asarray(pred, float) - np.asarray(true, float)) / np.asarray(true, float)


def mape(e) -> float:
    e = np.asarray(e, float)
    return float(np.mean(np.abs(e[np.isfinite(e)])))


def n_bucket(n: int) -> str:
    for lo, hi in N_BUCKETS:
        if lo <= n <= hi:
            return f"n in [{lo},{hi}]"
    return "other"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    # pandas' default float parser is not correctly rounded and loses the
    # last ULP on some bounds; round_trip is exact. Immaterial to every
    # figure reported here, but the artifact should read back as written.
    sw = pd.read_csv(SWEEP, float_precision="round_trip")
    sw = sw[sw.status == "ok"].copy()
    sw["e"] = err(sw.bound, sw.true_cost)
    sw["bucket"] = sw.n.map(n_bucket)

    ref = pd.read_csv(REF_ND, low_memory=False)
    ref = ref[(ref.model.isin(REF_MODELS)) & (ref.status == "ok")].copy()
    meta = pd.read_csv(BASE_ND)[["instance", "dimension", "n_customers"]]
    ref = ref.merge(meta, on="instance")
    ref = ref.rename(columns={"dimension": "d", "n_customers": "n"})
    ref["e"] = err(ref.pred_cost, ref.true_cost)
    ref["bucket"] = ref.n.map(n_bucket)
    return sw, ref


def crossover(mape_by_k: dict[int, float], target: float) -> dict:
    """Smallest ladder budget whose MAPE is at or below ``target``."""
    ks = sorted(mape_by_k)
    hit = [k for k in ks if mape_by_k[k] <= target]
    if not hit:
        return {"k": None, "note": "no budget in the ladder reaches the target"}
    k = hit[0]
    return {"k": int(k), "MAPE_at_k": float(mape_by_k[k]), "target": float(target)}


def table_by(sw: pd.DataFrame, ref: pd.DataFrame, key: str) -> pd.DataFrame:
    gart = ref[ref.model == "GART_2.0"]
    rows = []
    for val, g in sw.groupby(key):
        by_k = {int(k): mape(gg.e) for k, gg in g.groupby("k")}
        gm = mape(gart[gart[key] == val].e)
        row = {key: val, "N": int(g[g.k == 0].shape[0]), "GART_2.0": gm}
        row.update({f"HK_k{k}": v for k, v in sorted(by_k.items())})
        c = crossover(by_k, gm)
        row["crossover_k"] = c["k"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired(sw: pd.DataFrame, ref: pd.DataFrame, k: int) -> dict:
    """Per-instance paired comparison at one budget."""
    gart = ref[ref.model == "GART_2.0"][["instance", "e"]].rename(columns={"e": "e_gart"})
    h = sw[sw.k == k][["instance", "d", "n", "bucket", "e"]].rename(columns={"e": "e_hk"})
    j = h.merge(gart, on="instance")
    j["hk_better"] = j.e_hk.abs() < j.e_gart.abs()
    return {"k": k, "N": int(len(j)),
            "hk_win_rate_pct": float(100.0 * j.hk_better.mean()),
            "MAPE_hk": mape(j.e_hk), "MAPE_gart": mape(j.e_gart),
            "win_rate_by_d_pct": {int(a): float(100 * b) for a, b in
                                  j.groupby("d").hk_better.mean().items()},
            "win_rate_by_bucket_pct": {str(a): float(100 * b) for a, b in
                                       j.groupby("bucket").hk_better.mean().items()}}


def main() -> None:
    sw, ref = load()
    ks = sorted(sw.k.unique())
    gart_all = mape(ref[ref.model == "GART_2.0"].e)
    by_k = {int(k): mape(g.e) for k, g in sw.groupby("k")}

    rep: dict = {
        "corpus": {"N_instances": int(sw.instance.nunique()),
                   "N_rows": int(len(sw)),
                   "ladder": [int(k) for k in ks]},
        "reference_models_MAPE": {m: mape(ref[ref.model == m].e) for m in REF_MODELS},
        "polyak_MAPE_by_k": by_k,
        "crossover_vs_GART_2.0": crossover(by_k, gart_all),
    }

    # per-dimension and per-n tables
    t_d = table_by(sw, ref, "d")
    t_d.to_csv(OUT / "polyak_nd_by_dimension.csv", index=False)
    t_n = table_by(sw, ref, "bucket")
    t_n.to_csv(OUT / "polyak_nd_by_nbucket.csv", index=False)
    rep["by_dimension_csv"] = str(OUT / "polyak_nd_by_dimension.csv")
    rep["by_nbucket_csv"] = str(OUT / "polyak_nd_by_nbucket.csv")

    # head-to-head against the shipped Volgenant--Jonker sweep
    if VJ_SWEEP.exists():
        vj = pd.read_csv(VJ_SWEEP, float_precision="round_trip")
        vj = vj[vj.status == "ok"].copy()
        vj["e"] = err(vj.bound, vj.true_cost)
        j = (sw[["instance", "k", "bound", "e"]]
             .merge(vj[["instance", "k", "bound", "e"]], on=["instance", "k"],
                    suffixes=("_pk", "_vj")))
        jd = j.merge(sw[sw.k == 0][["instance", "d"]], on="instance")
        rep["vs_shipped_VJ"] = {
            "MAPE_by_k": {int(k): {"polyak": mape(g.e_pk), "vj": mape(g.e_vj)}
                          for k, g in j.groupby("k")},
            "polyak_bound_higher_pct_at_k2000":
                float(100.0 * (j[j.k == 2000].bound_pk >
                               j[j.k == 2000].bound_vj + 1e-9).mean()),
            "MAPE_by_d_at_k2000":
                {int(a): {"polyak": mape(g.e_pk), "vj": mape(g.e_vj)}
                 for a, g in jd[jd.k == 2000].groupby("d")},
        }

    # label provenance cut
    if PROVENANCE.exists():
        prov = pd.read_csv(PROVENANCE)[["instance", "solver"]]
        s2 = sw.merge(prov, on="instance", how="left", suffixes=("", "_p"))
        conc = s2[s2.solver_p.fillna(s2.get("solver")) == "concorde"] \
            if "solver_p" in s2 else s2[s2.solver == "concorde"]
        g2 = ref[ref.model == "GART_2.0"].merge(prov, on="instance")
        rep["concorde_exact_subset"] = {
            "N_instances": int(conc.instance.nunique()),
            "GART_2.0_MAPE": mape(g2[g2.solver == "concorde"].e),
            "polyak_MAPE_by_k": {int(k): mape(g.e) for k, g in conc.groupby("k")},
            "by_d_at_k2000": {int(a): mape(g.e) for a, g in
                              conc[conc.k == 2000].groupby("d")},
        }

    # signed-error shape and certificate integrity
    for k in (100, 500, 2000):
        g = sw[sw.k == k]
        rep[f"shape_k{k}"] = {
            "MAPE": mape(g.e), "MedAPE": float(np.median(np.abs(g.e))),
            "mean_signed": float(g.e.mean()),
            "frac_above_label_pct": float(100.0 * (g.e > 0).mean()),
            "max_above_label_pct": float(g.e.max()),
            "is_optimal_pct": float(100.0 * g.is_optimal.mean())
            if "is_optimal" in g else None,
        }

    # paired tests
    rep["paired"] = {str(k): paired(sw, ref, k) for k in (50, 100, 200, 500, 2000)}

    (OUT / "polyak_nd_results.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("by_dimension_csv", "by_nbucket_csv")}, indent=2)[:6000])
    print("\n--- MAPE by dimension ---")
    print(t_d.round(4).to_string(index=False))
    print("\n--- MAPE by n bucket ---")
    print(t_n.round(4).to_string(index=False))
    print(f"\nwrote {OUT / 'polyak_nd_results.json'}")


if __name__ == "__main__":
    main()
