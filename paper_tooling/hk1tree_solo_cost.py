"""The 1-tree's absolute cost, on the manuscript's own timing protocol.

WHY THIS EXISTS
---------------
Every cost figure in the manuscript's timing column comes from one protocol:
one estimator per process, 11 repeats, OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads
pinned to 1, JIT and first predict warmed outside the clock, median over
repeats published and the relative IQR retained, on a quiet box
(``gart2_timing_bank.json -> tsplib_by_size_time_one_protocol``, tag
``serial_solo_median11_quiet_2026-08-11``).

The first 1-tree cost pass did not meet that protocol. Its ladder was measured
at 20-73% CPU load and its headline arm was *interleaved* -- GART 2.0 and the
1-tree sharing one process. Co-measurement is not neutral in this project: it
was shown to move small-``n`` medians by up to 1.9x, in GART 2.0's favour. Those
numbers are therefore ratios only and cannot be printed beside the solo column.

This script consumes a re-measurement taken under the solo protocol
(``paper_tooling/hk1tree_frontier_timing.py``, tags ``solo_*``) and produces the
figures that *can* be printed beside it. It writes no timing of its own.

THE DECISIVE NUMBER
-------------------
The cost multiple of the 1-tree over GART 2.0 **at the accuracy crossover**,
where the crossover budget is located from the same ladder that supplies the
cost. It is reported per size bucket as well as for the corpus, because the
ordering reverses with ``n``: the 1-tree is far cheaper than GART 2.0 on small
instances and far dearer on large ones, so a single corpus number conceals the
fact on which the positioning turns.

    python paper_tooling/hk1tree_solo_cost.py            # analyse + write bank
    python paper_tooling/hk1tree_solo_cost.py --dry-run  # analyse only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_tooling"
if str(OUT) not in sys.path:
    sys.path.insert(0, str(OUT))

from hk1tree_frontier_analyze import (  # noqa: E402
    SIZE_BUCKETS,
    crossing_k,
    interp_time_ms,
)

#: The solo re-measurement. One file per process invocation; nothing here was
#: measured alongside anything else.
SOLO_SOURCES = {
    "solo_gart2_A": "GART 2.0, start of session",
    "solo_asympmst_A": "Asymptotic_MST, drift control",
    "solo_ladder_k500_r11": "1-tree ladder to k=500, 11 repeats, n <= 16384",
    "solo_direct_k50_r11": "1-tree via the shipped estimator object at k=50",
    "solo_ladder_k500_r3_tail": "the capped tail (d18512) at 3 repeats",
    "solo_gart2_B": "GART 2.0, end of session",
    "solo_asympmst_B": "Asymptotic_MST, end of session",
    "noisywindow_ladder_k500_r6": "the same ladder, 6 repeats, noisy window (control)",
}

#: Halfway through the 11 repeats. Both halves are now from the quiet window,
#: so this is a stability check on the published statistic, not a correction.
LADDER_WINDOW_SPLIT = 6

#: GART 2.0 was measured twice, once at each end of the session. The ladder ran
#: in the quiet window, so the *quiet* GART reading is the denominator: a cost
#: multiple must divide two numbers taken under the same conditions. The noisy
#: reading is retained and reported, not discarded.
GART_DENOM = "solo_gart2_B"

#: ``tsplib_by_size_time_one_protocol -> time_ms``, the cells this pass has to
#: be comparable with. Median over instances of the per-instance median-of-11.
PUBLISHED_MS = {
    "n in [51,150]": {"GART_2.0": 3.644, "Asymptotic_MST": 1.156},
    "n in [151,400]": {"GART_2.0": 4.514, "Asymptotic_MST": 1.840},
    "n > 400": {"GART_2.0": 20.544, "Asymptotic_MST": 9.273},
    "Total (all EUC_2D)": {"GART_2.0": 6.122, "Asymptotic_MST": 2.574},
}

#: Every rung the ascent to k=500 checkpoints. The five the re-measurement was
#: commissioned for are ``REQUESTED_KS``; ``0`` and ``200`` come free from the
#: same ascent and are kept because they bracket the crossovers, so the
#: interpolation below never has to extrapolate off the priced grid.
LADDER_KS = (0, 10, 25, 50, 100, 200, 500)
REQUESTED_KS = (10, 25, 50, 100, 500)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_solo() -> pd.DataFrame:
    """Every solo timing row, tagged with the process it came from."""
    frames = []
    for tag in SOLO_SOURCES:
        p = OUT / f"hk1tree_timing_{tag}.csv"
        if not p.exists():
            print(f"  MISSING {p.name} -- skipped")
            continue
        d = pd.read_csv(p)
        d["source"] = tag
        frames.append(d)
    if not frames:
        raise SystemExit("no solo timing files found; run hk1tree_frontier_timing.py")
    return pd.concat(frames, ignore_index=True)


def per_instance(t: pd.DataFrame) -> pd.DataFrame:
    """Per-instance median over repeats, with the relative IQR retained.

    The published statistic is a median of medians; the IQR is what says
    whether that median is worth quoting.
    """
    g = t.groupby(["source", "model", "k", "instance", "n"]).seconds
    out = g.agg(median="median", q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75), reps="size").reset_index()
    out["iqr_rel_pct"] = 100.0 * (out.q3 - out.q1) / out["median"]
    return out


def cell(pi: pd.DataFrame, source: str, model: str, bucket: str,
         keep: set[str]) -> dict | None:
    """One published-style cell: median over instances, restricted to ``keep``."""
    s = pi[(pi.source == source) & (pi.model == model) & pi.instance.isin(keep)]
    if bucket != "Total (all EUC_2D)":
        lo, hi = next((l, h) for lbl, l, h in SIZE_BUCKETS if lbl == bucket)
        s = s[(s.n >= lo) & (s.n <= hi)]
    if s.empty:
        return None
    return {"N": int(len(s)), "ms": 1000.0 * float(s["median"].median()),
            "total_ms": 1000.0 * float(s["median"].sum()),
            "median_iqr_rel_pct": float(s.iqr_rel_pct.median()),
            "repeats": int(s.reps.max())}


# ---------------------------------------------------------------------------
# Accuracy, on the same instances
# ---------------------------------------------------------------------------
def accuracy(keep: set[str]) -> dict:
    """MAPE of the raw bound at each ladder budget, and of GART 2.0, per bucket.

    ``err_pct = 100 (pred - true) / true`` and ``MAPE = mean|err_pct|``, the
    definition ``build_paper_tables.py`` uses for every accuracy cell in the
    manuscript. The bound is scored raw -- uncalibrated and certified -- because
    that is the object whose cost is being priced.
    """
    lad = pd.read_csv(OUT / "hk1tree_frontier_tsplib.csv")
    lad = lad[lad.instance.isin(keep)].copy()
    lad["err_pct"] = 100.0 * (lad.bound - lad.true_cost) / lad.true_cost

    res = pd.read_csv(ROOT / "tsplib_benchmark" / "results" /
                      "all_models_tsplib_repaired.csv")
    g2 = res[(res.model == "GART_2.0") & (res.edge_weight_type == "EUC_2D") &
             res.instance.isin(keep)].copy()

    out = {}
    for label, lo, hi in SIZE_BUCKETS:
        if label == "Total (all EUC_2D)":
            lsub, gsub = lad, g2
        else:
            lsub, gsub = lad[(lad.n >= lo) & (lad.n <= hi)], g2[(g2.n >= lo) & (g2.n <= hi)]
        by_k = {int(k): float(s.err_pct.abs().mean())
                for k, s in lsub.groupby("k")}
        out[label] = {"N": int(len(gsub)),
                      "gart2_mape_pct": float(gsub.abs_gap_pct.mean()),
                      "hk_raw_mape_pct_by_k": by_k}
    return out


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
def window_split(raw: pd.DataFrame, keep: set[str]) -> dict:
    """The ladder statistic recomputed on each half of the repeat sequence.

    If the two halves agree, the median over all 11 is not an artefact of when
    in the session the repeat happened to land.
    """
    lad = raw[(raw.source == "solo_ladder_k500_r11") & raw.instance.isin(keep)]
    out = {}
    for k in REQUESTED_KS:
        s = lad[lad.k == k]
        early = s[s.repeat < LADDER_WINDOW_SPLIT]
        late = s[s.repeat >= LADDER_WINDOW_SPLIT]
        if early.empty or late.empty:
            continue
        e = 1000.0 * early.groupby("instance").seconds.median().median()
        l = 1000.0 * late.groupby("instance").seconds.median().median()
        out[k] = {"repeats_0_5_ms": float(e), "repeats_6_10_ms": float(l),
                  "late_over_early": float(l / e)}
    return out


def build(pi: pd.DataFrame, raw: pd.DataFrame) -> dict:
    lad = pi[pi.source == "solo_ladder_k500_r11"]
    keep = set(lad.instance.unique())
    all78 = set(pi[pi.source == "solo_gart2_A"].instance.unique())
    excluded = sorted(all78 - keep)

    acc = accuracy(keep)
    rep: dict = {
        "protocol": {
            "tag": "serial_solo_median11_quiet_2026-08-11_1tree",
            "matches": ("gart2_timing_bank.json -> tsplib_by_size_time_one_protocol, "
                        "tag serial_solo_median11_quiet_2026-08-11"),
            "process": ("one estimator per process, single thread of control, no "
                        "pool, OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 1. Verified before "
                        "every batch by asserting zero other hk1tree python "
                        "processes were running."),
            "repeats_per_pair": 11,
            "statistic": ("median over repeats; median over instances of that; "
                          "relative IQR retained"),
            "warm_up": "numba JIT and each model's first call warmed outside the clock",
            "background_cpu_pct": ("11-15% measured on \\Processor(_Total)\\% "
                                   "Processor Time immediately before each batch, "
                                   "against 13-16% for the published column"),
            "load_sensor_warning": (
                "the harness's own cpu_load() reads Win32_Processor.LoadPercentage, "
                "which returned 81-96% at moments when \\Processor(_Total)\\% "
                "Processor Time returned 18.8-21.8%. The cpu_load column in the raw "
                "CSVs is therefore NOT a box-load measurement and the '20-73% load' "
                "that disqualified the first 1-tree pass is largely an artefact of "
                "that sensor. The first pass's real defect is co-measurement, not load."),
            "co_measurement_incident": (
                "a reaped background wrapper left an orphaned ladder process alive, "
                "which raced the resumed run between 17:09 and 18:20. Every repeat "
                "measured in that window carried duplicate (repeat, instance, k) rows "
                "and was discarded; the surviving solo repeats were retained. The "
                "published 11 repeats are all from the quiet window after the orphan "
                "was killed and zero-concurrency was asserted."),
            "cap": ("n <= 16384 = held_karp_1tree.DENSE_MAX_N, the estimator's own "
                    "dense/coordinate kernel switch. Excludes exactly one of the 78 "
                    "EUC_2D instances. Both arms are compared on the matched 77."),
            "memory_screen": ("predicted peak = 3 * 8 * n^2 bytes on the dense path "
                              "(three simultaneous n x n float64 temporaries), "
                              "computed before each instance and required to fit in "
                              "50% of free physical RAM. Largest admitted: d15112 at "
                              "5.11 GiB against 40.6 GiB free. Nothing was skipped on "
                              "memory grounds; the cap alone bound."),
        },
        "matched_subset": {
            "N": len(keep),
            "excluded": excluded,
            "excluded_n": {i: int(pi[pi.instance == i].n.iloc[0]) for i in excluded},
        },
        "by_bucket": {},
        "ladder_repeat_window_split": window_split(raw, keep),
        "load_sensitivity": {},
        "drift_control": {},
        "checkpointed_vs_direct_k50": {},
        "capped_tail": {},
    }

    # -- drift: today over published, on the FULL 78, both control estimators --
    for model, srcs in (("GART_2.0", ("solo_gart2_A", "solo_gart2_B")),
                        ("Asymptotic_MST", ("solo_asympmst_A", "solo_asympmst_B"))):
        for src in srcs:
            if pi[pi.source == src].empty:
                continue
            d = {}
            for label, *_ in SIZE_BUCKETS:
                c = cell(pi, src, model, label, all78)
                pub = PUBLISHED_MS[label][model]
                if c:
                    d[label] = {"today_ms": c["ms"], "published_ms": pub,
                                "today_over_published": c["ms"] / pub,
                                "median_iqr_rel_pct": c["median_iqr_rel_pct"],
                                "N": c["N"]}
            rep["drift_control"][f"{model}::{src}"] = d

    # -- the cost/accuracy table, matched subset, per bucket and total --------
    for label, *_ in SIZE_BUCKETS:
        g = cell(pi, GART_DENOM, "GART_2.0", label, keep)
        gn = cell(pi, "solo_gart2_A", "GART_2.0", label, keep)
        if g is None:
            continue
        hk_ms, hk_iqr, hk_tot = {}, {}, {}
        for k in LADDER_KS:
            c = cell(pi, "solo_ladder_k500_r11", f"HK_1Tree_{k}", label, keep)
            if c:
                hk_ms[k], hk_iqr[k], hk_tot[k] = c["ms"], c["median_iqr_rel_pct"], c["total_ms"]

        a = acc[label]
        mape = {k: v for k, v in a["hk_raw_mape_pct_by_k"].items() if k in hk_ms}
        cx = crossing_k(mape, a["gart2_mape_pct"])
        entry = {
            "N": g["N"],
            "gart2_ms": g["ms"],
            "gart2_total_ms": g["total_ms"],
            "gart2_median_iqr_rel_pct": g["median_iqr_rel_pct"],
            "gart2_ms_noisy_window": gn["ms"] if gn else None,
            "gart2_mape_pct": a["gart2_mape_pct"],
            "hk_ms_by_k": hk_ms,
            "hk_median_iqr_rel_pct_by_k": hk_iqr,
            "hk_total_ms_by_k": hk_tot,
            "hk_over_gart2_by_k": {k: v / g["ms"] for k, v in hk_ms.items()},
            "hk_over_gart2_total_by_k": {k: v / g["total_ms"] for k, v in hk_tot.items()},
            "hk_raw_mape_pct_by_k": mape,
            "crossover": dict(cx),
        }
        if cx["reached"]:
            kk = int(cx["ladder_k"])
            entry["crossover"].update({
                "cost_ms_at_ladder_k": hk_ms[kk],
                "cost_x_gart2_at_ladder_k": hk_ms[kk] / g["ms"],
                "cost_ms_at_interp_k": interp_time_ms(hk_ms, cx["interp_k"]),
                "cost_x_gart2_at_interp_k":
                    interp_time_ms(hk_ms, cx["interp_k"]) / g["ms"],
                "bound_mape_pct_at_ladder_k": mape[kk],
                "bound_mape_over_gart2_mape": mape[kk] / a["gart2_mape_pct"],
            })
        rep["by_bucket"][label] = entry

    # -- how much does background load move each arm? -------------------------
    # Both arms were measured twice, once in each window. If the 1-tree moves
    # more than GART 2.0 does, then a shared-process measurement of the pair
    # cannot be neutral, and the quiet-box requirement binds harder on the
    # 1-tree than on the estimator it is being priced against.
    for k in REQUESTED_KS:
        q = cell(pi, "solo_ladder_k500_r11", f"HK_1Tree_{k}", "Total (all EUC_2D)", keep)
        n_ = cell(pi, "noisywindow_ladder_k500_r6", f"HK_1Tree_{k}",
                  "Total (all EUC_2D)", keep)
        if q and n_:
            rep["load_sensitivity"][f"HK_1Tree_{k}"] = {
                "quiet_ms": q["ms"], "noisy_ms": n_["ms"],
                "noisy_over_quiet": n_["ms"] / q["ms"]}
    gq = cell(pi, GART_DENOM, "GART_2.0", "Total (all EUC_2D)", keep)
    gnz = cell(pi, "solo_gart2_A", "GART_2.0", "Total (all EUC_2D)", keep)
    if gq and gnz:
        rep["load_sensitivity"]["GART_2.0"] = {
            "quiet_ms": gq["ms"], "noisy_ms": gnz["ms"],
            "noisy_over_quiet": gnz["ms"] / gq["ms"]}

    # -- does the checkpointed clock agree with the shipped estimator? --------
    for label, *_ in SIZE_BUCKETS:
        d = cell(pi, "solo_direct_k50_r11", "HK_1Tree_50_direct", label, keep)
        c = cell(pi, "solo_ladder_k500_r11", "HK_1Tree_50", label, keep)
        if d and c:
            rep["checkpointed_vs_direct_k50"][label] = {
                "checkpointed_ms": c["ms"], "direct_ms": d["ms"],
                "direct_over_checkpointed": d["ms"] / c["ms"], "N": d["N"]}

    # -- what the cap costs, measured rather than asserted --------------------
    tail = pi[pi.source == "solo_ladder_k500_r3_tail"]
    if not tail.empty:
        hk, reps = {}, 0
        for k in LADDER_KS:
            s = tail[tail.model == f"HK_1Tree_{k}"]
            if not s.empty:
                hk[k] = 1000.0 * float(s["median"].iloc[0])
                reps = int(s.reps.iloc[0])
        gt = pi[(pi.source == GART_DENOM) & (pi.instance == str(tail.instance.iloc[0]))]
        g_ms = 1000.0 * float(gt["median"].iloc[0]) if not gt.empty else float("nan")
        rep["capped_tail"] = {
            "instance": str(tail.instance.iloc[0]), "n": int(tail.n.iloc[0]),
            "repeats": reps, "gart2_ms": g_ms, "hk_ms_by_k": hk,
            "hk_over_gart2_by_k": {k: v / g_ms for k, v in hk.items()},
            "why_excluded": ("above the estimator's own dense/coordinate kernel "
                             "switch DENSE_MAX_N=16384, and at k=500 it costs more "
                             "wall clock than the other 77 instances together"),
        }
    return rep


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------
SUPERSEDE = (
    "SUPERSEDED by cost_tsplib_euc2d_solo_2026_08_11 in "
    "paper_tooling/hk1tree_frontier_bank.json. Measured at 20-73% CPU load, and "
    "the headline arm co-measured GART 2.0 and the 1-tree in ONE process. "
    "Co-measurement moved small-n medians by up to 1.9x elsewhere in this "
    "project, in GART 2.0's favour, so these are ratios only and must not be "
    "printed beside the solo timing column. Kept for provenance -- do not quote "
    "an absolute millisecond figure from this key."
)


def write_bank(rep: dict) -> None:
    p = OUT / "hk1tree_frontier_bank.json"
    b = json.loads(p.read_text(encoding="utf-8"))
    if "cost_tsplib_euc2d_78" in b:
        b["cost_tsplib_euc2d_78"]["_superseded"] = SUPERSEDE
    b["cost_tsplib_euc2d_solo_2026_08_11"] = rep
    p.write_text(json.dumps(b, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {p}")

    q = OUT / "frontier_positioning_bank.json"
    f = json.loads(q.read_text(encoding="utf-8"))
    for key in ("cost_normalisation", "crossing"):
        if isinstance(f.get(key), dict):
            f[key]["_superseded"] = SUPERSEDE
    f["_superseded_by_solo_cost_pass"] = {
        "when": "2026-08-11",
        "what": SUPERSEDE,
        "contaminated_keys": ["cost_normalisation", "crossing", "table_total_corpus",
                              "table_by_bucket", "pareto_front_total_corpus",
                              "pareto_front_by_bucket", "gart2_dominators_total_corpus",
                              "gart2_dominators_by_bucket"],
        "replacement": ("paper_tooling/hk1tree_frontier_bank.json -> "
                        "cost_tsplib_euc2d_solo_2026_08_11"),
        "note": ("the two table_* and pareto_* keys mix solo comparator costs with "
                 "load-contaminated 1-tree costs, so their 1-tree rows are the "
                 "affected part; the estimator rows in them are already solo."),
    }
    f["crossing_solo_2026_08_11"] = {
        b: rep["by_bucket"][b]["crossover"] for b in rep["by_bucket"]}
    q.write_text(json.dumps(f, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {q}")


def write_csvs(pi: pd.DataFrame, rep: dict) -> None:
    pi.to_csv(OUT / "hk1tree_solo_cost_per_instance.csv", index=False)
    rows = []
    for bucket, e in rep["by_bucket"].items():
        rows.append({"bucket": bucket, "N": e["N"], "model": "GART_2.0", "k": np.nan,
                     "ms": e["gart2_ms"], "x_gart2": 1.0,
                     "iqr_rel_pct": e["gart2_median_iqr_rel_pct"],
                     "mape_pct": e["gart2_mape_pct"]})
        for k in sorted(e["hk_ms_by_k"]):
            rows.append({"bucket": bucket, "N": e["N"], "model": f"HK_1Tree_{k}", "k": k,
                         "ms": e["hk_ms_by_k"][k],
                         "x_gart2": e["hk_over_gart2_by_k"][k],
                         "iqr_rel_pct": e["hk_median_iqr_rel_pct_by_k"][k],
                         "mape_pct": e["hk_raw_mape_pct_by_k"].get(k, np.nan)})
    pd.DataFrame(rows).to_csv(OUT / "hk1tree_solo_cost_by_bucket.csv", index=False)
    print(f"Wrote {OUT / 'hk1tree_solo_cost_by_bucket.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = load_solo()
    pi = per_instance(raw)
    rep = build(pi, raw)
    print(json.dumps(rep, indent=1))
    if not args.dry_run:
        write_csvs(pi, rep)
        write_bank(rep)


if __name__ == "__main__":
    main()
