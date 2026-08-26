"""What each label defect does to the numbers the manuscript prints.

Three defects, established by ``label_certificate.py`` and the metric audit:

D1  METRIC.  Every released synthetic label is a solver tour length in an
    integer-quantised metric; every MST-based estimator predicts a float64
    Euclidean cost.  2D labels are ``sum nint(euclidean)`` on integer
    coordinates.  ND labels are ``solver_integer_length / scale``.
D2  linhp318 carries a ``FIXED_EDGES_SECTION`` the parser drops, so its label
    (41345) is the Hamiltonian-path optimum while its coordinates are lin318's
    and its tour optimum is 42029.
D3  35 ND test labels sit below a proven lower bound.

For each, this rebuilds exactly the table set ``build_paper_tables.main`` gates
on and diffs it against ``Area_Free_Main.tex`` with ``run_check``'s own
tolerance -- half a unit in the last printed place.  Nothing is written back
into the manuscript or into ``paper_tooling/tables/``.

The float64 reference for a defect-1 rescoring is the float64 length of the same
stored optimal tour.  That tour is feasible, so its length upper-bounds the
float64 optimum; on the 651 planar and 6,275 multidimensional instances where
the ascent closes exactly it equals the float64 optimum to a mean 0.0004% and
0.0007%, so it is a witness, not an approximation.

    python paper_tooling/label_defect_blast_radius.py
Outputs: paper_tooling/label_defect_gate_<variant>.csv   moved cells, per variant
         paper_tooling/label_defect_blast_radius.json    counts + headline moves
         paper_tooling/label_defect_ref_2d.csv           2D float64 tour lengths
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paper_tooling.build_paper_tables as B  # noqa: E402

OUT = ROOT / "paper_tooling"
P_REF_2D = OUT / "label_defect_ref_2d.csv"
P_AUDIT = OUT / "reference_tour_audit.csv"
P_CERT = OUT / "label_certificate.csv"
INST_2D = ROOT / "Generalized_TSP_Analysis" / "instances"
SOL_2D = ROOT / "Generalized_TSP_Analysis" / "solutions"
GT_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv"

LINHP = "linhp318"
LIN318_TOUR_OPTIMUM = 42029.0

ACCURACY_METRICS = frozenset({
    "SDPE_pct", "SDPE_lo", "SDPE_hi", "MAPE_pct", "MedAPE_pct", "MSPE_pct",
    "R2_alpha", "close5_pct", "close10_pct", "kendall_tau", "spearman_rho"})


# ---------------------------------------------------------------------------
# 2D float64 reference tours
# ---------------------------------------------------------------------------
def build_ref_2d() -> pd.DataFrame:
    """Float64 and nint lengths of every stored 2D optimal tour. Cached on disk."""
    if P_REF_2D.exists():
        return pd.read_csv(P_REF_2D)
    rows = []
    for rec in pd.read_csv(GT_2D).itertuples():
        name = rec.instance
        X = np.asarray(json.loads((INST_2D / f"{name}.json").read_text())["coordinates"],
                       dtype=np.float64)
        data = json.loads((SOL_2D / f"{name}.sol.json").read_text())
        tour = data.get("optimal_tour") or data.get("concorde_tour") or data.get("lkh_tour")
        t = np.asarray(tour, dtype=np.int64)
        n = X.shape[0]
        if t.size == n and t.min() == 1 and t.max() == n:
            t = t - 1
        if t.size != n or sorted(t.tolist()) != list(range(n)):
            rows.append({"instance": name, "n": n, "status": "not_a_permutation"})
            continue
        e = np.linalg.norm(X[t] - X[np.roll(t, -1)], axis=1)
        rows.append({"instance": name, "n": n, "stored_cost": float(rec.true_cost),
                     "tour_len_float": float(e.sum()), "tour_len_nint": float(np.rint(e).sum()),
                     "status": "ok"})
    df = pd.DataFrame(rows)
    df.to_csv(P_REF_2D, index=False)
    return df


# ---------------------------------------------------------------------------
# Table rebuild + diff
# ---------------------------------------------------------------------------
def build_tables(d2: pd.DataFrame, dn: pd.DataFrame, euc: pd.DataFrame,
                 non: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rnd = d2[d2["generator"] == "random"]
    specs = [
        ("2d_by_size", d2, B.B_2D_SIZE, "prediction_time_s"),
        ("2d_by_genclass", d2, B.B_GENCLASS, "prediction_time_s"),
        ("2d_random_by_size", rnd, B.B_2D_SIZE, "prediction_time_s"),
        ("2d_random_n10_80", rnd[(rnd["n_customers"] >= 10) & (rnd["n_customers"] <= 80)],
         [(r"$n\in[10,80]$", "n10_80", B._all())], "prediction_time_s"),
        ("nd_by_size", dn, B.B_ND_SIZE, "prediction_time_s"),
        ("nd_by_dim", dn, B.B_ND_DIM, "prediction_time_s"),
        ("tsplib_by_size", euc, B.B_TSPLIB, "total_time_s"),
    ]
    tables = {name: B.compute_table(df, bk, tc, name) for name, df, bk, tc in specs}
    tables["tsplib_nonEuc"] = B.compute_table(non, B.B_NONEUC, "total_time_s", "tsplib_nonEuc")
    tables["rank"] = pd.concat([B.compute_rank(d2, "2D", "2d"),
                                B.compute_rank(dn, "ND", "nd"),
                                B.compute_rank(B.with_31f_controls(euc),
                                               r"TSPLIB EUC\_2D", "tsplib_euc2d")],
                               ignore_index=True)
    return tables


def flatten(tables: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for name, tidy in tables.items():
        for _, r in tidy.iterrows():
            kb = B._clean(str(r["bucket"]))
            out[(name, kb, "", "bucket_count")] = r["bucket_count"]
            for metric in B.CHECK_METRICS:
                if metric in r.index:
                    out[(name, kb, str(r["display"]), metric)] = r[metric]
    return out


def diff_vs_tex(new: dict, old: dict) -> pd.DataFrame:
    rows = []
    for key, cell in sorted(old.items()):
        if key not in new or cell is None:
            continue
        nv = new[key]
        if isinstance(nv, str) or not np.isfinite(float(nv)):
            continue
        ov, ndec = cell
        r = round(float(nv), ndec)
        if abs(ov - r) > min(0.005, 0.5 * 10.0 ** -ndec):
            rows.append({"table": key[0], "bucket": key[1], "model": key[2], "metric": key[3],
                         "paper": ov, "repaired": r, "delta": r - ov})
    return pd.DataFrame(rows, columns=["table", "bucket", "model", "metric",
                                       "paper", "repaired", "delta"])


def swap_labels(df: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    o = df.copy()
    o["true_cost"] = o["instance"].map(ref)
    o = o[np.isfinite(o["true_cost"]) & (o["true_cost"] > 0)].copy()
    o["err_pct"] = 100.0 * (o["pred_cost"] - o["true_cost"]) / o["true_cost"]
    return o[np.isfinite(o["err_pct"])].copy()


def main() -> None:
    old = B.parse_tex(B.P_TEX)
    d2, _, _ = B.load_2d()
    dn, _, _, _ = B.load_nd()
    euc, non, _ = B.load_tsplib()

    ref2d = build_ref_2d()
    ref2d = ref2d[ref2d.status == "ok"].set_index("instance")
    audit = pd.read_csv(P_AUDIT)
    good = audit[(audit.split == "test") & (audit.bucket != "corrupt")].set_index("instance_name")
    wrong = list(pd.read_csv(P_CERT).query("corpus == 'ND'")["instance"]) if P_CERT.exists() else []

    d2_f64 = swap_labels(d2, ref2d["tour_len_float"])
    dn_f64 = swap_labels(dn[dn.instance.isin(good.index)], good["tour_len_float"])
    euc_res = euc.copy()
    m = euc_res["instance"] == LINHP
    euc_res.loc[m, "true_cost"] = LIN318_TOUR_OPTIMUM
    euc_res.loc[m, "err_pct"] = 100.0 * (euc_res.loc[m, "pred_cost"] - LIN318_TOUR_OPTIMUM) \
        / LIN318_TOUR_OPTIMUM
    euc_drop = euc[euc.instance != LINHP].copy()
    dn_drop = dn[~dn.instance.isin(wrong)].copy()
    dn_all = swap_labels(dn[dn.instance.isin(good.index) & ~dn.instance.isin(wrong)],
                         good["tour_len_float"])

    variants = {
        "as_published": (d2, dn, euc, non),
        "D1_float64_2d": (d2_f64, dn, euc, non),
        "D1_float64_nd": (d2, dn_f64, euc, non),
        "D1_float64_both": (d2_f64, dn_f64, euc, non),
        "D2_linhp318_rescored": (d2, dn, euc_res, non),
        "D2_linhp318_excluded": (d2, dn, euc_drop, non),
        f"D3_drop_{len(wrong)}_certified_wrong": (d2, dn_drop, euc, non),
        "all_repairs": (d2_f64, dn_all, euc_drop, non),
    }

    report = {"gated_cells": len(old), "variants": {}}
    for tag, frames in variants.items():
        moved = diff_vs_tex(flatten(build_tables(*frames)), old)
        moved.to_csv(OUT / f"label_defect_gate_{tag}.csv", index=False)
        acc = moved[moved.metric.isin(ACCURACY_METRICS)]
        report["variants"][tag] = {
            "cells_moved": int(len(moved)),
            "accuracy_cells": int(len(acc)),
            "timing_cells": int((moved.metric == "time_ms").sum()),
            "count_cells": int(moved.metric.isin(("N", "bucket_count")).sum()),
            "max_abs_accuracy_delta": float(acc.delta.abs().max()) if len(acc) else 0.0,
            "accuracy_cells_by_table": acc.groupby("table").size().to_dict() if len(acc) else {},
        }
        print(f"{tag:<32} {len(moved):>4} of {len(old)} gated cells move "
              f"({len(acc)} accuracy, {int((moved.metric == 'time_ms').sum())} timing, "
              f"{int(moved.metric.isin(('N', 'bucket_count')).sum())} counts)")

    (OUT / "label_defect_blast_radius.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT / 'label_defect_blast_radius.json'}")
    if report["variants"]["as_published"]["cells_moved"]:
        print("!! the as-published rebuild does not reproduce the manuscript; "
              "everything below is measured against a moving baseline", file=sys.stderr)


if __name__ == "__main__":
    main()
