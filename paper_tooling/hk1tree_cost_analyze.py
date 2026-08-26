"""Assemble the two missing frontier cells, and define "corpus median".

INPUTS
------
* ``hk1tree_costtime_{2d,noneuc}_{gart2,vj_ckpt,polyak_ckpt}.csv`` -- the two
  missing cost cells, written by ``hk1tree_cost_allbench.py`` under the solo
  protocol of Table 3, plus the ``*_direct`` / ``noneuc_{vj,polyak}``
  amortisation controls.
* ``hk1tree_costtime_loadctl_tsplib_vjckpt.csv`` -- today's checkpointed TSPLIB
  ladder, paired instance by instance against the published quiet-window one,
  which is what turns background load from a confounder into a measured
  covariate.
* ``hk1tree_costtime_drift_{A,B}.csv`` -- GART 2.0 on TSPLIB EUC_2D at each end
  of the timing session, against the published cells of
  ``tsplib_by_size_time_one_protocol``. This is the only check that says
  whether today's absolute milliseconds may be printed beside yesterday's.
* the accuracy ladders of ``hk1tree_all_benchmarks.py`` and the per-budget
  training-split constants ``c_k`` of ``hk1tree_frontier_analyze``.

OUTPUTS
-------
* ``hk1tree_cost_frontier_bank.json`` -- every cell, plus the gates.
* ``hk1tree_cost_frontier_{2d,noneuc}.csv`` -- tidy, one row per
  (group, ascent, budget).
* ``hk1tree_cost_aggregation.csv`` -- the two cost aggregations side by side on
  all four benchmarks. See below.
* ``tables/frontier_2d_*.tex``, ``tables/frontier_noneuc.tex`` -- fragments in
  the shape of Tables 3 and 4. Not spliced; the prose pass owns the manuscript.

THE TWO COST AGGREGATIONS, AND WHY BOTH ARE PRINTED
---------------------------------------------------
The manuscript says "corpus median" six times and never defines it. It names
one specific statistic, the one Table 3's Time column carries:

    typical-instance cost = median over the corpus of each instance's
                            median over the 11 repeats

and the accuracy printed beside it is a MAPE, a *mean* over the same
instances, so the word "median" describes the cost column alone. That
statistic answers "what does one instance cost", and it does not compose: a
median of medians cannot be recovered from per-bucket medians at any weights,
which is why Table 3's three buckets cannot be made to yield the 0.90 the
abstract quotes.

    corpus throughput cost = sum over the corpus of those same per-instance
                             medians

answers the other question, "what does the whole benchmark cost", and *is*
reconstructible from the buckets. On TSPLIB EUC_2D the two disagree by more
than an order of magnitude -- the bound is 0.90 times GART 2.0 per typical
instance at k=25 and 24.9 times it per corpus -- because the corpus is
dominated by a handful of large instances on which the bound is quadratic.
Both are computed here, on all four benchmarks, so a reader can see which
question each answer belongs to.

    ...python.exe paper_tooling/hk1tree_cost_analyze.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "paper_tooling", ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hk1tree_all_benchmarks import _out_path  # noqa: E402
from hk1tree_cost_allbench import BUDGETS  # noqa: E402
from hk1tree_frontier_analyze import err, fit_correction, load_ref  # noqa: E402

OUT = ROOT / "paper_tooling"
TAB = OUT / "tables"
TSPLIB_RESULTS = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"

#: The manuscript's own 2D size buckets (``build_paper_tables.B_2D_SIZE``).
BUCKETS_2D = ((5, 10), (11, 50), (51, 100), (101, 500), (501, 1000))

#: ``tsplib_by_size_time_one_protocol`` -> ``time_ms``, GART 2.0, the cells the
#: drift control has to reproduce for today's absolutes to be printable beside
#: the published column.
PUBLISHED_GART2_TSPLIB_MS = {"n in [51,150]": 3.644, "n in [151,400]": 4.514,
                             "n > 400": 20.544, "Total (all EUC_2D)": 6.122}
TSPLIB_BUCKETS = (("n in [51,150]", 51, 150), ("n in [151,400]", 151, 400),
                  ("n > 400", 401, 10 ** 9))

CERT_RTOL = 1e-9

#: Relative tolerance for "the timed call returned the number the accuracy
#: sweep scored". The observed floor is one ULP, ~1.5e-16 relative.
AGREE_RTOL = 1e-12


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_time(tag: str) -> pd.DataFrame | None:
    p = OUT / f"hk1tree_costtime_{tag}.csv"
    if not p.exists():
        print(f"  MISSING {p.name}")
        return None
    return pd.read_csv(p)


def per_instance(t: pd.DataFrame) -> pd.DataFrame:
    """Per-instance median over repeats, with the relative IQR retained.

    Identical in definition to ``hk1tree_solo_cost.per_instance``; the
    published statistic is a median of medians and the IQR is what says
    whether that median is worth quoting.
    """
    g = t.groupby(["model", "k", "instance", "n"], dropna=False).seconds
    out = g.agg(median="median", q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75), fastest="min",
                reps="size").reset_index()
    out["iqr_rel_pct"] = 100.0 * (out.q3 - out.q1) / out["median"]
    return out


def cost_cell(pi: pd.DataFrame, model: str, keep: set[str]) -> dict | None:
    s = pi[(pi.model == model) & pi.instance.isin(keep)]
    if s.empty:
        return None
    return {"N": int(len(s)),
            "ms": 1000.0 * float(s["median"].median()),
            "total_ms": 1000.0 * float(s["median"].sum()),
            # Fastest-repeat companion to every median. Under a contended box
            # the median absorbs whatever else was running; the minimum over
            # repeats is the closest estimate of the uncontended cost the data
            # contains, and the ratio between the two measures the contention.
            # Both arms get it, so a ratio can be formed from either.
            "ms_fastest": 1000.0 * float(s["fastest"].median()),
            "total_ms_fastest": 1000.0 * float(s["fastest"].sum()),
            "median_iqr_rel_pct": float(s.iqr_rel_pct.median()),
            "max_iqr_rel_pct": float(s.iqr_rel_pct.max()),
            "repeats": int(s.reps.max())}


# ---------------------------------------------------------------------------
# Gates on the timing run's own predictions
# ---------------------------------------------------------------------------
def gate_timing_predictions(corpus: str, arms: dict[str, pd.DataFrame],
                            ladders: dict[str, pd.DataFrame],
                            truth: pd.Series) -> dict:
    """Four checks, all on the numbers the *timed* calls returned.

    A cost column is only about the same object as the accuracy column beside
    it if the timed call returned the scored number. That is checkable and is
    checked here rather than assumed:

    1. determinism -- every repeat of a given (instance, k) returned the same
       bound, so no cell is an average over a randomised computation;
    2. agreement   -- that bound equals the accuracy sweep's bound for the same
       (instance, ascent, k), which is what makes Table 3's cost column and its
       MAPE column describe one object;
    3. certificate -- ``bound <= optimum`` on every instance with a label;
    4. monotone    -- the bound is non-decreasing in the budget.
    """
    rep: dict = {}
    for arm, t in arms.items():
        if arm == "gart2" or t is None:
            continue
        ascent = "vj" if arm.startswith("vj") else "polyak"
        if t.empty:
            continue
        g = t.groupby(["instance", "k"]).pred
        spread = (g.max() - g.min()).abs()
        det = {"pairs": int(len(spread)),
               "nonzero_spread": int((spread > 0).sum()),
               "max_spread": float(spread.max()) if len(spread) else 0.0}

        first = t.drop_duplicates(["instance", "k"])[["instance", "k", "pred"]]
        lad = ladders[ascent][["instance", "k", "bound"]]
        m = first.merge(lad, on=["instance", "k"], how="inner")
        d = (m.pred - m.bound).abs()
        rel = d / m.bound.abs().clip(lower=1.0)
        # A handful of cells differ by one ULP: the sweep evaluated its ladder
        # inside a worker process and the shipped entry point re-derives the
        # same trajectory here, and the bound is round-tripped through CSV in
        # between. AGREE_RTOL is four orders of magnitude above that floor and
        # far below anything a MAPE cell could notice.
        agree = {"compared": int(len(m)),
                 "bit_identical": int((d == 0).sum()),
                 "mismatches": int((rel > AGREE_RTOL).sum()),
                 "rtol": AGREE_RTOL,
                 "max_abs_diff": float(d.max()) if len(m) else 0.0,
                 "max_rel_diff": float(rel.max()) if len(m) else 0.0}
        if arm.endswith("_ckpt"):
            agree["note"] = ("the checkpointed read-off against the shipped "
                             "single-budget entry point's ladder: this is the "
                             "prefix property, checked rather than assumed")

        j = first.assign(true_cost=first.instance.map(truth)).dropna(subset=["true_cost"])
        over = j[j.pred > j.true_cost * (1.0 + CERT_RTOL)]
        cert = {"checked": int(len(j)), "violations": int(len(over)),
                "max_excess_pct": (float((100.0 * (j.pred - j.true_cost)
                                          / j.true_cost).max()) if len(j) else float("nan"))}
        if len(over):
            cert["worst"] = over.nlargest(5, "pred")[
                ["instance", "k", "pred", "true_cost"]].to_dict("records")

        s = first.sort_values(["instance", "k"])
        diff = s.groupby("instance").pred.diff()
        mono = {"series": int(s.instance.nunique()),
                "violations": int((diff < -CERT_RTOL * s.pred.abs().clip(lower=1.0)).sum()),
                "min_step": float(diff.min()) if diff.notna().any() else 0.0}

        rep[arm] = {"determinism": det, "agreement_with_accuracy_sweep": agree,
                    "certificate": cert, "monotone_in_k": mono,
                    "PASS": bool(det["nonzero_spread"] == 0
                                 and agree["mismatches"] == 0
                                 and cert["violations"] == 0
                                 and mono["violations"] == 0)}
    rep["PASS"] = all(v["PASS"] for v in rep.values() if isinstance(v, dict))
    rep["corpus"] = corpus
    return rep


# ---------------------------------------------------------------------------
# Drift control
# ---------------------------------------------------------------------------
def drift_control() -> dict:
    """Today's GART 2.0 on TSPLIB EUC_2D against the published column.

    Absolute milliseconds are a property of the box as much as of the code.
    The published column was taken at 11-15% background load; this block
    measures the same estimator on the same 78 instances under the same
    harness at each end of the new session, so the reader is told the offset
    instead of having to assume there is none.
    """
    out: dict = {}
    for end in ("A", "B"):
        t = load_time(f"drift_{end}")
        if t is None:
            continue
        pi = per_instance(t)
        blk = {}
        for label, lo, hi in TSPLIB_BUCKETS:
            s = pi[(pi.n >= lo) & (pi.n <= hi)]
            if s.empty:
                continue
            ms = 1000.0 * float(s["median"].median())
            blk[label] = {"today_ms": ms, "published_ms": PUBLISHED_GART2_TSPLIB_MS[label],
                          "today_over_published": ms / PUBLISHED_GART2_TSPLIB_MS[label],
                          "N": int(len(s)),
                          "median_iqr_rel_pct": float(s.iqr_rel_pct.median())}
        ms = 1000.0 * float(pi["median"].median())
        blk["Total (all EUC_2D)"] = {
            "today_ms": ms,
            "published_ms": PUBLISHED_GART2_TSPLIB_MS["Total (all EUC_2D)"],
            "today_over_published": ms / PUBLISHED_GART2_TSPLIB_MS["Total (all EUC_2D)"],
            "N": int(len(pi)),
            "median_iqr_rel_pct": float(pi.iqr_rel_pct.median()),
            "box_load_pct_median": float(t.box_load_pct.median())}
        out[f"drift_{end}"] = blk
    if "drift_A" in out and "drift_B" in out:
        a = out["drift_A"]["Total (all EUC_2D)"]["today_ms"]
        b = out["drift_B"]["Total (all EUC_2D)"]["today_ms"]
        out["session_stability"] = {"start_ms": a, "end_ms": b, "end_over_start": b / a}
    return out


def load_control() -> dict:
    """How much the box moves each arm, measured per instance, not assumed.

    The published TSPLIB pass wrote every per-instance median it took to
    ``hk1tree_solo_cost_per_instance.csv``. Re-measuring the same instances
    with the same code under today's box and dividing instance by instance
    gives a *paired* inflation factor for each arm separately. It matters that
    it is per arm: the published pass already found the bound 1.25 times more
    expensive in a noisy window against 1.05 for GART 2.0, so a cost ratio
    taken on a loaded box is biased against the bound, and the size of that
    bias is exactly what this block measures.

    Returns the median paired ratio for GART 2.0 (from the drift control) and
    for the checkpointed 1-tree ladder at every budget.
    """
    p = OUT / "hk1tree_solo_cost_per_instance.csv"
    if not p.exists():
        return {"status": "missing hk1tree_solo_cost_per_instance.csv"}
    pub = pd.read_csv(p)
    pub["instance"] = pub.instance.astype(str)
    out: dict = {"published_source": p.name,
                 "published_protocol": "solo, 11 repeats, quiet window, 2026-08-11"}

    def paired(today: pd.DataFrame, today_model: str, pub_source: str,
               pub_model: str) -> dict | None:
        a = today[today.model == today_model].set_index("instance")["median"]
        b = pub[(pub.source == pub_source) & (pub.model == pub_model)] \
            .set_index("instance")["median"]
        j = pd.concat([a.rename("today"), b.rename("pub")], axis=1).dropna()
        if j.empty:
            return None
        r = (j.today / j["pub"]).to_numpy(float)
        return {"N": int(len(r)), "median_today_over_published": float(np.median(r)),
                "q25": float(np.quantile(r, 0.25)), "q75": float(np.quantile(r, 0.75)),
                "min": float(r.min()), "max": float(r.max())}

    t = load_time("drift_A")
    if t is not None:
        out["GART_2.0"] = paired(per_instance(t), "GART_2.0", "solo_gart2_B",
                                 "GART_2.0")
    t = load_time("loadctl_tsplib_vjckpt")
    if t is not None:
        pi = per_instance(t)
        blk = {}
        for k in BUDGETS:
            r = paired(pi, f"HK_1Tree_vjckpt_{k}", "solo_ladder_k500_r11",
                       f"HK_1Tree_{k}")
            if r:
                blk[str(k)] = r
        out["HK_1Tree_vj"] = blk
        g = out.get("GART_2.0")
        if g and blk:
            out["bias_in_a_cost_ratio"] = {
                str(k): v["median_today_over_published"]
                / g["median_today_over_published"] for k, v in blk.items()}
            out["reading"] = (
                "bias_in_a_cost_ratio is the factor by which a bound/GART 2.0 "
                "cost multiple measured on today's box exceeds the same "
                "multiple on the published quiet box, on the corpus where both "
                "are known. Above 1 means today's numbers overstate the "
                "bound's cost; divide by it to read a quiet-box estimate.")
    return out


# ---------------------------------------------------------------------------
# One benchmark
# ---------------------------------------------------------------------------
def groups_for(corpus: str, meta: pd.DataFrame) -> list[tuple[str, set[str]]]:
    """(label, instance set). The last entry is always the whole corpus."""
    if corpus == "2d":
        gs = []
        for lo, hi in BUCKETS_2D:
            sel = meta[(meta.n >= lo) & (meta.n <= hi)]
            gs.append((f"n in [{lo},{hi}]", set(sel.instance)))
        gs.append(("Total (all 2D)", set(meta.instance)))
        return gs
    gs = []
    for ewt, sel in meta.groupby("edge_weight_type"):
        gs.append((str(ewt), set(sel.instance)))
    # Section 6's Table 8 is scored on the screened set -- the EXPLICIT
    # matrices with structural triangle-inequality violations removed -- so the
    # screened row is what a reader compares against it. The unscreened Total
    # is the population the bound is actually valid on, since the relaxation
    # never invokes the triangle inequality, and the difference between the two
    # rows is the measurement of what the screen costs.
    gs.append(("Screened (metric)", set(meta[~meta.structural_violator].instance)))
    gs.append(("Total (all non-EUC_2D)", set(meta.instance)))
    return gs


#: arm -> (timing tag suffix, model-name infix, which accuracy ladder it is).
#: The two ``_ckpt`` arms are the primary measurement; the two bare ones are
#: the amortisation control and are run on a subsample on the 2D corpus, so
#: they are reported separately and never mixed into a published cell.
ARM_SPEC = {
    "vj_ckpt": ("vj_ckpt", "vjckpt", "vj"),
    "polyak_ckpt": ("polyak_ckpt", "pkckpt", "polyak"),
}
CONTROL_SPEC = {
    "2d": {"vj": ("vj_direct", "vj", "vj"),
           "polyak": ("polyak_direct", "polyak", "polyak")},
    "noneuc": {"vj": ("vj", "vj", "vj"),
               "polyak": ("polyak", "polyak", "polyak")},
}


def pick_gart2(corpus: str, bound_load: float) -> tuple[pd.DataFrame, dict]:
    """Choose which GART 2.0 reading is the denominator, and say why.

    The estimator is measured twice per corpus, once at each end of that
    corpus's block, because this box's background load moves during a session
    and a cost multiple has to divide two numbers taken under the same
    conditions -- the rule ``hk1tree_solo_cost.GART_DENOM`` applies to the
    published column.

    The published pass could pick by window because its two readings sat in a
    quiet and a noisy window it had characterised. Here the box-load column is
    not usable for the choice: a GART 2.0 repeat over 31 non-Euclidean
    instances lasts under a second, and a load sampled over a window that short
    is dominated by whatever happened to be scheduled in it. So the rule is
    deterministic and conservative instead -- take the **faster** reading,
    which is the denominator least favourable to the bound -- and print both,
    with their spread, so the effect of the choice is bounded on the page.
    """
    cands = {}
    for suffix in ("gart2", "gart2_B"):
        t = load_time(f"{corpus}_{suffix}")
        if t is not None and not t.empty:
            cands[suffix] = t
    if not cands:
        raise SystemExit(f"[{corpus}] no GART 2.0 timing arm")
    info = {k: {"box_load_pct_median": float(v.box_load_pct.median()),
                "ms": 1000.0 * float(per_instance(v)["median"].median())}
            for k, v in cands.items()}
    best = min(info, key=lambda k: info[k]["ms"])
    return cands[best], {
        "chosen": best,
        "rule": ("faster of the two readings: the denominator least "
                 "favourable to the bound"),
        "bound_arm_box_load_pct": bound_load,
        "candidates": info,
        "spread_pct": (100.0 * (max(x["ms"] for x in info.values())
                                / min(x["ms"] for x in info.values()) - 1.0))
        if len(info) > 1 else 0.0}


def analyse(corpus: str, ck: dict) -> dict:
    arms: dict[str, pd.DataFrame | None] = {}
    for a, (suf, _infix, _lad) in ARM_SPEC.items():
        arms[a] = load_time(f"{corpus}_{suf}")
    controls = {a: load_time(f"{corpus}_{suf}")
                for a, (suf, _i, _l) in CONTROL_SPEC[corpus].items()}
    missing = [a for a, t in arms.items() if t is None]
    if missing:
        raise SystemExit(f"[{corpus}] missing arm(s) {missing}; "
                         f"run hk1tree_cost_allbench.py")
    bound_load = float(np.median([t.box_load_pct.median() for t in arms.values()]))
    arms["gart2"], gart_choice = pick_gart2(corpus, bound_load)

    ladders = {a: pd.read_csv(_out_path(a, corpus)) for a in ("vj", "polyak")}
    ladders = {a: d[d.status == "ok"].copy() for a, d in ladders.items()}
    truth = (ladders["vj"].drop_duplicates("instance")
             .set_index("instance").true_cost)

    meta = (ladders["vj"].drop_duplicates("instance")
            [["instance", "n"] + (["edge_weight_type", "structural_violator"]
                                  if corpus == "noneuc" else [])].copy())

    pis = {a: per_instance(t) for a, t in arms.items() if t is not None}
    pis_ctl = {a: per_instance(t) for a, t in controls.items() if t is not None}

    # -- who is scored by whom -------------------------------------------------
    bound_inst = set(ladders["vj"].instance) & set(ladders["polyak"].instance)
    timed_bound = set(pis["vj_ckpt"].instance) & set(pis["polyak_ckpt"].instance)
    gart_ok = arms["gart2"][arms["gart2"].status == "ok"]
    timed_gart = set(gart_ok.instance)
    if corpus == "noneuc":
        ref = pd.read_csv(TSPLIB_RESULTS, low_memory=False)
        ref = ref[(ref.model == "GART_2.0") & (ref.edge_weight_type != "EUC_2D")
                  & (ref.status == "ok")]
    else:
        ref = load_ref(corpus)
        ref = ref[ref.model == "GART_2.0"]
    scored_gart = set(ref.instance.astype(str))
    matched = bound_inst & timed_bound & timed_gart & scored_gart
    meta = meta[meta.instance.isin(matched)]

    rep: dict = {
        "corpus": corpus,
        "protocol": {
            "tag": "serial_solo_median11_2026-08-12_costfill",
            "matches": ("gart2_timing_bank.json -> tsplib_by_size_time_one_protocol, "
                        "tag serial_solo_median11_quiet_2026-08-11: one estimator "
                        "per process, single thread of control, threads pinned to "
                        "1, 11 repeats, median over repeats then median over "
                        "instances, relative IQR retained, JIT and first predict "
                        "warmed outside the clock, parsing outside the clock, and "
                        "the same checkpointed ladder Table 3 reports"),
            "amortisation_control": ("the vj / polyak arms call the shipped "
                                     "single-budget entry point once per rung, so "
                                     "nothing is shared across the ladder; on 2d "
                                     "they run on a size-stratified subsample "
                                     "because sum(BUDGETS)/max(BUDGETS) iterations "
                                     "on 2,580 instances is not affordable"),
            "repeats": int(pis["vj_ckpt"].reps.max()),
            "budgets": list(BUDGETS),
            "statistic_cost_typical": ("median over instances of the per-instance "
                                       "median over repeats -- Table 3's Time column"),
            "statistic_cost_throughput": ("sum over instances of those same "
                                          "per-instance medians"),
            "statistic_accuracy": "MAPE = mean over instances of |percent error|",
            "gart2_call_site": ("estimate(coords, d, grid_size) on 2d; classical MDS "
                                "plus the hybrid feature build on the non-native "
                                "non-EUC_2D instances, direct on the native pair"),
            "gart2_denominator": gart_choice,
            "box_load_pct_median_bound_arms": bound_load,
        },
        "instance_accounting": {
            "bound_scored": len(bound_inst),
            "bound_timed": len(timed_bound),
            "gart2_timed_ok": len(timed_gart),
            "gart2_scored_in_benchmark": len(scored_gart & bound_inst),
            "matched": len(matched),
            "gart2_declined": sorted(timed_bound - timed_gart),
            "bound_only": sorted(bound_inst - scored_gart),
        },
        "gates": gate_timing_predictions(
            corpus,
            {**arms, **{f"{a}_direct": t for a, t in controls.items()}},
            ladders, truth),
        "groups": {},
    }
    if corpus == "noneuc":
        rep["instance_accounting"]["structural_violators_in_matched"] = sorted(
            meta[meta.structural_violator].instance)

    # -- GART 2.0's own accuracy, from the released benchmark ------------------
    ref_m = ref[ref.instance.astype(str).isin(matched)].copy()
    ref_m["instance"] = ref_m.instance.astype(str)
    ref_m["abs_err_pct"] = np.abs(err(ref_m.pred_cost, ref_m.true_cost))
    gart_err = ref_m.set_index("instance").abs_err_pct

    for label, inst in groups_for(corpus, meta):
        inst = inst & matched
        if not inst:
            continue
        g = cost_cell(pis["gart2"], "GART_2.0", inst)
        if g is None:
            continue
        entry: dict = {
            "N": len(inst),
            "gart2_ms": g["ms"], "gart2_total_ms": g["total_ms"],
            "gart2_ms_fastest": g["ms_fastest"],
            "gart2_total_ms_fastest": g["total_ms_fastest"],
            "gart2_median_iqr_rel_pct": g["median_iqr_rel_pct"],
            "gart2_MAPE_pct": float(gart_err.reindex(sorted(inst)).dropna().mean()),
            "ascents": {},
        }
        if corpus == "noneuc":
            est_only = arms["gart2"][arms["gart2"].instance.isin(inst)]
            e = (est_only.groupby("instance").seconds_estimate_only.median())
            entry["gart2_ms_estimate_only"] = 1000.0 * float(e.median())
            entry["gart2_ms_mds_share_pct"] = 100.0 * (
                1.0 - entry["gart2_ms_estimate_only"] / entry["gart2_ms"])

        for ascent, (_suf, infix, lad_key) in ARM_SPEC.items():
            tag = infix
            lad = ladders[lad_key]
            lad = lad[lad.instance.isin(inst)]
            blk: dict = {}
            for k in BUDGETS:
                c = cost_cell(pis[ascent], f"HK_1Tree_{tag}_{k}", inst)
                if c is None:
                    continue
                s = lad[lad.k == k]
                e_raw = np.abs(err(s.bound, s.true_cost))
                e_cal = np.abs(err(s.bound * ck[k]["median"], s.true_cost))
                blk[str(k)] = {
                    "ms": c["ms"], "total_ms": c["total_ms"],
                    "x_gart2_typical": c["ms"] / g["ms"],
                    "x_gart2_throughput": c["total_ms"] / g["total_ms"],
                    "ms_fastest": c["ms_fastest"],
                    "x_gart2_typical_fastest": c["ms_fastest"] / g["ms_fastest"],
                    "x_gart2_throughput_fastest":
                        c["total_ms_fastest"] / g["total_ms_fastest"],
                    "median_iqr_rel_pct": c["median_iqr_rel_pct"],
                    "max_iqr_rel_pct": c["max_iqr_rel_pct"],
                    "raw_MAPE_pct": float(e_raw.mean()),
                    "cal_MAPE_pct": float(e_cal.mean()),
                    "c_k": ck[k]["median"],
                    "win_rate_vs_gart2_pct": _win_rate(s, gart_err, ck[k]["median"]),
                }
            entry["ascents"][ascent] = blk
            entry.setdefault("crossover", {})[ascent] = _crossover(
                blk, entry["gart2_MAPE_pct"])
        rep["groups"][label] = entry

    rep["amortisation_control"] = _amortisation(corpus, pis, pis_ctl, matched)
    return rep


def _amortisation(corpus: str, pis: dict, pis_ctl: dict, matched: set) -> dict:
    """What the checkpointed ladder saves over calling the entry point per rung.

    Paired on the instance, on whatever subset the control was run on, so the
    factor is a measurement and not the difference between two corpora. A
    checkpointed cell divided by this factor is what a caller who asks for one
    budget and nothing else would pay.
    """
    out: dict = {}
    for ctl, (_suf, infix, _lad) in CONTROL_SPEC[corpus].items():
        if ctl not in pis_ctl:
            out[ctl] = {"status": "control arm not run"}
            continue
        prim = "vj_ckpt" if ctl == "vj" else "polyak_ckpt"
        p_infix = ARM_SPEC[prim][1]
        blk = {}
        for k in BUDGETS:
            a = pis_ctl[ctl]
            a = a[(a.model == f"HK_1Tree_{infix}_{k}")
                  & a.instance.isin(matched)].set_index("instance")["median"]
            b = pis[prim]
            b = b[(b.model == f"HK_1Tree_{p_infix}_{k}")
                  & b.instance.isin(matched)].set_index("instance")["median"]
            j = pd.concat([a.rename("direct"), b.rename("ckpt")], axis=1).dropna()
            if j.empty:
                continue
            r = (j.direct / j.ckpt).to_numpy(float)
            blk[str(k)] = {"N": int(len(r)),
                           "direct_ms": 1000.0 * float(j.direct.median()),
                           "ckpt_ms": 1000.0 * float(j.ckpt.median()),
                           "median_direct_over_ckpt": float(np.median(r)),
                           "q25": float(np.quantile(r, 0.25)),
                           "q75": float(np.quantile(r, 0.75))}
        out[ctl] = blk
    return out


def _win_rate(lad_k: pd.DataFrame, gart_err: pd.Series, c: float) -> dict:
    """Fraction of instances on which the bound's absolute error is smaller.

    Paired on the instance, raw and calibrated, so the aggregate margin is not
    the only thing supporting the claim.
    """
    e = np.asarray(np.abs(err(lad_k.bound, lad_k.true_cost)), float)
    ec = np.asarray(np.abs(err(lad_k.bound * c, lad_k.true_cost)), float)
    gm = gart_err.reindex(lad_k.instance.astype(str)).to_numpy(float)
    ok = np.isfinite(gm)
    if not ok.any():
        return {"raw": float("nan"), "cal": float("nan"), "N": 0}
    return {"raw": float(100.0 * np.mean(e[ok] < gm[ok])),
            "cal": float(100.0 * np.mean(ec[ok] < gm[ok])),
            "N": int(ok.sum())}


def _crossover(blk: dict, gart_mape: float) -> dict:
    """Cheapest priced budget at which the bound beats GART 2.0 on MAPE."""
    out = {}
    for tag, key in (("raw", "raw_MAPE_pct"), ("cal", "cal_MAPE_pct")):
        beats = [int(k) for k, v in blk.items() if v[key] < gart_mape]
        if beats:
            k = min(beats)
            out[tag] = {"k": k, "MAPE_pct": blk[str(k)][key],
                        "x_gart2_typical": blk[str(k)]["x_gart2_typical"],
                        "x_gart2_throughput": blk[str(k)]["x_gart2_throughput"],
                        "dominates_on_both_axes":
                            bool(blk[str(k)]["x_gart2_typical"] < 1.0)}
        else:
            out[tag] = {"k": None}
    return out


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def tidy(rep: dict) -> pd.DataFrame:
    rows = []
    for label, e in rep["groups"].items():
        rows.append({"corpus": rep["corpus"], "group": label, "N": e["N"],
                     "row": "GART_2.0", "ascent": "", "k": np.nan,
                     "ms": e["gart2_ms"], "total_ms": e["gart2_total_ms"],
                     "ms_fastest": e["gart2_ms_fastest"],
                     "x_gart2_typical": 1.0, "x_gart2_throughput": 1.0,
                     "x_gart2_typical_fastest": 1.0,
                     "iqr_rel_pct": e["gart2_median_iqr_rel_pct"],
                     "raw_MAPE_pct": e["gart2_MAPE_pct"],
                     "cal_MAPE_pct": np.nan, "c_k": np.nan,
                     "win_rate_raw_pct": np.nan, "win_rate_cal_pct": np.nan})
        for ascent, blk in e["ascents"].items():
            for k, v in blk.items():
                rows.append({"corpus": rep["corpus"], "group": label, "N": e["N"],
                             "row": f"HK_1Tree_{ascent}", "ascent": ascent, "k": int(k),
                             "ms": v["ms"], "total_ms": v["total_ms"],
                             "ms_fastest": v["ms_fastest"],
                             "x_gart2_typical": v["x_gart2_typical"],
                             "x_gart2_throughput": v["x_gart2_throughput"],
                             "x_gart2_typical_fastest":
                                 v["x_gart2_typical_fastest"],
                             "iqr_rel_pct": v["median_iqr_rel_pct"],
                             "raw_MAPE_pct": v["raw_MAPE_pct"],
                             "cal_MAPE_pct": v["cal_MAPE_pct"], "c_k": v["c_k"],
                             "win_rate_raw_pct": v["win_rate_vs_gart2_pct"]["raw"],
                             "win_rate_cal_pct": v["win_rate_vs_gart2_pct"]["cal"]})
    return pd.DataFrame(rows)


def latex_ladder(rep: dict, labels: list[str], ascent: str, caption: str,
                 lab: str) -> str:
    """Table 3's shape: one column group per stratum, (ms, x, Raw, Cal.)."""
    gs = [(l, rep["groups"][l]) for l in labels if l in rep["groups"]]
    ncol = 4 * len(gs)
    head = " & ".join(
        rf"\multicolumn{{4}}{{c}}{{{l.replace('_', chr(92) + '_')}, "
        rf"$N={e['N']:,}$}}".replace(",", "{,}") for l, e in gs)
    cmid = "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(gs)))
    sub = " & ".join([r"ms & $\times$ & Raw & Cal."] * len(gs))
    g_row = " & ".join(
        rf"{e['gart2_ms']:.2f} & 1.00 & \multicolumn{{2}}{{c}}{{{e['gart2_MAPE_pct']:.2f}}}"
        for _, e in gs)
    lines = [
        r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{lab}}}", r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}", r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}l{'r' * ncol}@{{}}}}", r"\toprule",
        rf" & {head} \\", cmid, rf"Row & {sub} \\", r"\midrule",
        rf"GART 2.0 & {g_row} \\", r"\midrule",
    ]
    for k in BUDGETS:
        cells = []
        for _, e in gs:
            v = e["ascents"][ascent].get(str(k))
            if v is None:
                cells.append("--- & --- & --- & ---")
                continue
            dom = v["x_gart2_typical"] < 1.0
            raw = (rf"\textbf{{{v['raw_MAPE_pct']:.2f}}}"
                   if dom and v["raw_MAPE_pct"] < e["gart2_MAPE_pct"]
                   else f"{v['raw_MAPE_pct']:.2f}")
            cal = (rf"\textbf{{{v['cal_MAPE_pct']:.2f}}}"
                   if dom and v["cal_MAPE_pct"] < e["gart2_MAPE_pct"]
                   else f"{v['cal_MAPE_pct']:.2f}")
            cells.append(f"{v['ms']:.2f} & {v['x_gart2_typical']:.2f} & {raw} & {cal}")
        lines.append(rf"$k={k}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def latex_ladder_compact(rep: dict, labels: list[str], ascent: str, caption: str,
                         lab: str) -> str:
    """Table 4's shape: (ms, x, MAPE) per group, so four groups still fit.

    The six-group non-Euclidean ladder is 25 columns wide in Table 3's shape,
    which no page carries. This drops the calibrated column -- it is in the
    tidy CSV -- and keeps the raw certified row, which is the one that needs no
    training split and is therefore the row the section's argument rests on.
    """
    gs = [(l, rep["groups"][l]) for l in labels if l in rep["groups"]]
    ncol = 3 * len(gs)
    head = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{{l.replace('_', chr(92) + '_')}, "
        rf"$N={e['N']:,}$}}".replace(",", "{,}") for l, e in gs)
    cmid = "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(gs)))
    sub = " & ".join([r"ms & $\times$ & MAPE"] * len(gs))
    g_row = " & ".join(rf"{e['gart2_ms']:.2f} & 1.00 & {e['gart2_MAPE_pct']:.2f}"
                       for _, e in gs)
    lines = [
        r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{lab}}}", r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}", r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}l{'r' * ncol}@{{}}}}", r"\toprule",
        rf" & {head} \\", cmid, rf"Row & {sub} \\", r"\midrule",
        rf"GART 2.0 & {g_row} \\", r"\midrule",
    ]
    for k in BUDGETS:
        cells = []
        for _, e in gs:
            v = e["ascents"][ascent].get(str(k))
            if v is None:
                cells.append("--- & --- & ---")
                continue
            dom = (v["x_gart2_typical"] < 1.0
                   and v["raw_MAPE_pct"] < e["gart2_MAPE_pct"])
            m = (rf"\textbf{{{v['raw_MAPE_pct']:.2f}}}" if dom
                 else f"{v['raw_MAPE_pct']:.2f}")
            cells.append(f"{v['ms']:.2f} & {v['x_gart2_typical']:.2f} & {m}")
        lines.append(rf"$k={k}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def aggregation_table(reps: dict) -> pd.DataFrame:
    """The two cost aggregations on every benchmark, at every priced budget.

    This is the table the phrase "corpus median" needs behind it.
    """
    rows = []
    for corpus, rep in reps.items():
        tot = [l for l in rep["groups"] if l.startswith("Total")]
        if not tot:
            continue
        e = rep["groups"][tot[0]]
        for ascent, blk in e["ascents"].items():
            for k, v in blk.items():
                rows.append({
                    "corpus": corpus, "group": tot[0], "N": e["N"],
                    "ascent": ascent, "k": int(k),
                    "gart2_typical_ms": e["gart2_ms"],
                    "bound_typical_ms": v["ms"],
                    "x_gart2_typical": v["x_gart2_typical"],
                    "x_gart2_typical_fastest": v["x_gart2_typical_fastest"],
                    "gart2_throughput_ms": e["gart2_total_ms"],
                    "bound_throughput_ms": v["total_ms"],
                    "x_gart2_throughput": v["x_gart2_throughput"],
                    "gart2_MAPE_pct": e["gart2_MAPE_pct"],
                    "bound_raw_MAPE_pct": v["raw_MAPE_pct"],
                    "bound_cal_MAPE_pct": v["cal_MAPE_pct"]})
    return pd.DataFrame(rows)


def published_tsplib_aggregation() -> pd.DataFrame:
    """The same two aggregations for the already-published TSPLIB EUC_2D cell.

    Read out of ``hk1tree_frontier_bank.json`` rather than recomputed, so the
    row the abstract's 0.90 comes from is printed in the same table as the new
    ones and can be compared line for line.
    """
    b = json.loads((OUT / "hk1tree_frontier_bank.json").read_text(encoding="utf-8"))
    e = b["cost_tsplib_euc2d_solo_2026_08_11"]["by_bucket"]["Total (all EUC_2D)"]
    ck = fit_correction("train_d2")
    lad = pd.read_csv(OUT / "hk1tree_frontier_tsplib.csv")
    keep = set(pd.read_csv(OUT / "hk1tree_solo_cost_per_instance.csv")
               .query("source == 'solo_ladder_k500_r11'").instance.astype(str))
    lad = lad[lad.instance.astype(str).isin(keep)]
    rows = []
    for k, ms in e["hk_ms_by_k"].items():
        s = lad[lad.k == int(k)]
        rows.append({
            "corpus": "tsplib (published)", "group": "Total (all EUC_2D)",
            "N": e["N"], "ascent": "vj", "k": int(k),
            "gart2_typical_ms": e["gart2_ms"], "bound_typical_ms": ms,
            "x_gart2_typical": e["hk_over_gart2_by_k"][k],
            "x_gart2_typical_fastest": float("nan"),
            "gart2_throughput_ms": e["gart2_total_ms"],
            "bound_throughput_ms": e["hk_total_ms_by_k"][k],
            "x_gart2_throughput": e["hk_over_gart2_total_by_k"][k],
            "gart2_MAPE_pct": e["gart2_mape_pct"],
            "bound_raw_MAPE_pct": float(np.abs(err(s.bound, s.true_cost)).mean()),
            "bound_cal_MAPE_pct": float(np.abs(
                err(s.bound * ck[int(k)]["median"], s.true_cost)).mean())})
    return pd.DataFrame(rows)


#: The numbers the manuscript will assert once the prose pass reaches these
#: cells, as ``costfront:`` paths a ``prose_manifest.Claim`` can point at.
#: Written into the bank so the registration is generated from the data rather
#: than transcribed, and so ``check_prose_numbers.py`` can resolve every one of
#: them the moment an anchor exists. ``suggested_anchor`` is a template, not a
#: match: the prose pass writes the sentence and the anchor follows it.
PENDING: list[tuple[str, str, str]] = [
    ("frontier.2d.gart2_mape",
     "cells/2d/groups/Total (all 2D)/gart2_MAPE_pct",
     "GART 2.0 reaches {v}\\% MAPE on the 2D diverse benchmark"),
    ("frontier.2d.gart2_ms",
     "cells/2d/groups/Total (all 2D)/gart2_ms",
     "at {v}~ms on the typical instance"),
    ("frontier.2d.vj_k50_mape",
     "cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/raw_MAPE_pct",
     "the raw certified bound reaches {v}\\% at a budget of 50"),
    ("frontier.2d.vj_k50_cost_x",
     "cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/x_gart2_typical",
     "at {v} times GART 2.0's cost"),
    ("frontier.2d.vj_k25_cal_mape",
     "cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/25/cal_MAPE_pct",
     "the calibrated row reaches {v}\\% at a budget of 25"),
    ("frontier.2d.vj_k25_cost_x",
     "cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/25/x_gart2_typical",
     "at {v} times GART 2.0's cost"),
    ("frontier.2d.vj_k50_win_rate",
     "cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/win_rate_vs_gart2_pct/raw",
     "winning the paired comparison on {v}\\% of instances"),
    ("frontier.2d.smallest_bucket_cost_x",
     "cells/2d/groups/n in [5,10]/ascents/vj_ckpt/25/x_gart2_typical",
     "costs {v} times GART 2.0 on the smallest bucket"),
    ("frontier.2d.smallest_bucket_mape",
     "cells/2d/groups/n in [5,10]/ascents/vj_ckpt/25/raw_MAPE_pct",
     "at {v}\\% MAPE"),
    ("frontier.2d.throughput_x_k50",
     "cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/x_gart2_throughput",
     "and {v} times its cost over the whole corpus"),
    ("frontier.noneuc.gart2_mape",
     "cells/noneuc/groups/Total (all non-EUC_2D)/gart2_MAPE_pct",
     "GART 2.0 obtains {v}\\% MAPE"),
    ("frontier.noneuc.gart2_ms",
     "cells/noneuc/groups/Total (all non-EUC_2D)/gart2_ms",
     "at {v}~ms including the MDS embedding"),
    ("frontier.noneuc.gart2_mds_share",
     "cells/noneuc/groups/Total (all non-EUC_2D)/gart2_ms_mds_share_pct",
     "of which the embedding is {v}\\%"),
    ("frontier.noneuc.pk_k500_mape",
     "cells/noneuc/groups/Total (all non-EUC_2D)/ascents/polyak_ckpt/500/raw_MAPE_pct",
     "against {v}\\% for the bound at a budget of 500"),
    ("frontier.noneuc.pk_k500_cost_x",
     "cells/noneuc/groups/Total (all non-EUC_2D)/ascents/polyak_ckpt/500/x_gart2_typical",
     "at {v} times the cost"),
    ("frontier.noneuc.pk_k500_win_rate",
     "cells/noneuc/groups/Total (all non-EUC_2D)/ascents/polyak_ckpt/500/"
     "win_rate_vs_gart2_pct/raw",
     "on {v}\\% of the instances both score"),
    ("frontier.noneuc.vj_k25_cost_x",
     "cells/noneuc/groups/Total (all non-EUC_2D)/ascents/vj_ckpt/25/x_gart2_typical",
     "the bound already beats it at {v} times the cost"),
    ("frontier.noneuc.vj_k25_mape",
     "cells/noneuc/groups/Total (all non-EUC_2D)/ascents/vj_ckpt/25/raw_MAPE_pct",
     "at {v}\\% MAPE"),
    ("frontier.corpus_median.tsplib_cost_x_k25",
     "corpus_median_definition/tsplib_published/x_gart2_typical_k25",
     "0.90 is the ratio of two corpus medians, {v}"),
    ("frontier.corpus_median.tsplib_throughput_x_k25",
     "corpus_median_definition/tsplib_published/x_gart2_throughput_k25",
     "against {v} for the same pair over the whole corpus"),
    ("protocol.load.gart2_inflation",
     "load_control/GART_2.0/median_today_over_published",
     "GART 2.0 measures {v} times its published cost on this box"),
    ("protocol.load.bound_inflation_k100",
     "load_control/HK_1Tree_vj/100/median_today_over_published",
     "against {v} for the bound"),
]


def corpus_median_block(agg: pd.DataFrame) -> dict:
    """The row the phrase "corpus median" names, printed as flat keys.

    Three figures in the manuscript -- 0.90x, 2.00\\% and 2.56\\% -- are quoted
    "on the corpus median" and appear in no table, and Table 3's three size
    buckets cannot be made to produce them: a median of medians does not
    compose. This block prints the row they come from, and the throughput ratio
    for the same pair beside it, so the reader can see both that the term is
    well defined and that it is not the corpus's total cost.
    """
    out: dict = {"definition": {
        "typical_instance_cost": ("median over the corpus of each instance's "
                                  "median over the timing repeats -- Table 3's "
                                  "Time column, and what 'corpus median' names"),
        "corpus_throughput_cost": ("sum over the corpus of those same "
                                   "per-instance medians"),
        "accuracy": ("MAPE, a mean over the same instances: 'median' in "
                     "'corpus median' describes the cost column only"),
        "non_composability": ("no weighting of Table 3's three bucket medians "
                              "reproduces its Total row, which is why 0.90 "
                              "cannot be reconstructed from what is printed"),
    }}
    for corpus, g in agg.groupby("corpus"):
        blk: dict = {}
        for _, r in g.iterrows():
            a, k = str(r.ascent), int(r.k)
            for field, col in (("x_gart2_typical", "x_gart2_typical"),
                               ("x_gart2_throughput", "x_gart2_throughput"),
                               ("bound_typical_ms", "bound_typical_ms"),
                               ("bound_throughput_ms", "bound_throughput_ms"),
                               ("bound_raw_MAPE_pct", "bound_raw_MAPE_pct"),
                               ("bound_cal_MAPE_pct", "bound_cal_MAPE_pct")):
                blk[f"{field}_k{k}" if len(g.ascent.unique()) == 1
                    else f"{a}_{field}_k{k}"] = float(r[col])
        blk["N"] = int(g.N.iloc[0])
        blk["gart2_typical_ms"] = float(g.gart2_typical_ms.iloc[0])
        blk["gart2_throughput_ms"] = float(g.gart2_throughput_ms.iloc[0])
        blk["gart2_MAPE_pct"] = float(g.gart2_MAPE_pct.iloc[0])
        key = {"tsplib (published)": "tsplib_published"}.get(corpus, corpus)
        out[key] = blk
    return out


def resolve_pending(bank: dict) -> list[dict]:
    """Check every pending manifest path resolves, and record its value."""
    rows = []
    for cid, path, anchor in PENDING:
        node: object = bank
        ok = True
        for part in path.split("/"):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        rows.append({"id": f"pending.{cid}", "expect": f"costfront:{path}",
                     "suggested_anchor": anchor,
                     "resolves": ok,
                     "value": float(node) if ok and isinstance(node, (int, float))
                     else None})
    bad = [r for r in rows if not r["resolves"]]
    if bad:
        print(f"  WARNING: {len(bad)} pending manifest path(s) do not resolve:")
        for r in bad:
            print(f"    {r['id']}  {r['expect']}")
    else:
        print(f"  all {len(rows)} pending manifest paths resolve")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=["2d", "noneuc"])
    args = ap.parse_args()

    TAB.mkdir(exist_ok=True)
    ck = fit_correction("train_d2")

    bank: dict = {"budgets": list(BUDGETS),
                  "correction_source": "train_d2",
                  "c_k": {str(k): ck[k]["median"] for k in BUDGETS},
                  "drift_control": drift_control(),
                  "load_control": load_control(),
                  "cells": {}}

    reps = {}
    for corpus in args.corpora:
        rep = analyse(corpus, ck)
        reps[corpus] = rep
        bank["cells"][corpus] = rep
        tidy(rep).to_csv(OUT / f"hk1tree_cost_frontier_{corpus}.csv", index=False)
        print(f"wrote hk1tree_cost_frontier_{corpus}.csv   gates PASS="
              f"{rep['gates']['PASS']}")

    if "2d" in reps:
        r = reps["2d"]
        (TAB / "frontier_2d_vj_part1.tex").write_text(latex_ladder(
            r, ["n in [5,10]", "n in [11,50]", "n in [51,100]"], "vj_ckpt",
            "2D diverse benchmark cost/accuracy ladder, Volgenant--Jonker "
            "ascent (part 1 of 2: $n\\le100$).", "tab:frontier_2d_vj_a"),
            encoding="utf-8")
        (TAB / "frontier_2d_vj_part2.tex").write_text(latex_ladder(
            r, ["n in [101,500]", "n in [501,1000]", "Total (all 2D)"], "vj_ckpt",
            "2D diverse benchmark cost/accuracy ladder, Volgenant--Jonker "
            "ascent (part 2 of 2: $n>100$, and the whole corpus).",
            "tab:frontier_2d_vj_b"), encoding="utf-8")
        (TAB / "frontier_2d_polyak_part1.tex").write_text(latex_ladder(
            r, ["n in [5,10]", "n in [11,50]", "n in [51,100]"], "polyak_ckpt",
            "2D diverse benchmark cost/accuracy ladder, Polyak ascent "
            "(part 1 of 2: $n\\le100$).", "tab:frontier_2d_pk_a"),
            encoding="utf-8")
        (TAB / "frontier_2d_polyak_part2.tex").write_text(latex_ladder(
            r, ["n in [101,500]", "n in [501,1000]", "Total (all 2D)"], "polyak_ckpt",
            "2D diverse benchmark cost/accuracy ladder, Polyak ascent "
            "(part 2 of 2: $n>100$, and the whole corpus).",
            "tab:frontier_2d_pk_b"), encoding="utf-8")
    if "noneuc" in reps:
        r = reps["noneuc"]
        labels = [l for l in r["groups"] if not l.startswith("Total")] + \
                 [l for l in r["groups"] if l.startswith("Total")]
        (TAB / "frontier_noneuc_vj.tex").write_text(latex_ladder(
            r, labels, "vj_ckpt", "Non-EUC\\_2D TSPLIB95 cost/accuracy ladder, "
            "Volgenant--Jonker ascent.", "tab:frontier_noneuc_vj"),
            encoding="utf-8")
        (TAB / "frontier_noneuc_polyak.tex").write_text(latex_ladder(
            r, labels, "polyak_ckpt", "Non-EUC\\_2D TSPLIB95 cost/accuracy ladder, "
            "Polyak ascent.", "tab:frontier_noneuc_pk"), encoding="utf-8")
        compact = ["EXPLICIT", "GEO", "Screened (metric)",
                   "Total (all non-EUC_2D)"]
        for asc, tag, name in (("vj_ckpt", "vj", "Volgenant--Jonker"),
                               ("polyak_ckpt", "pk", "Polyak")):
            (TAB / f"frontier_noneuc_{tag}_compact.tex").write_text(
                latex_ladder_compact(
                    r, compact, asc,
                    f"Non-EUC\\_2D TSPLIB95 cost/accuracy ladder, {name} ascent, "
                    "raw certified bound. Screened is the metric subset "
                    "Table~\\ref{tab:tsplib_nonEuc} scores; Total adds the "
                    "structurally nonmetric matrices, on which the relaxation "
                    "is still valid.",
                    f"tab:frontier_noneuc_{tag}"), encoding="utf-8")

    agg = pd.concat([published_tsplib_aggregation(), aggregation_table(reps)],
                    ignore_index=True)
    agg.to_csv(OUT / "hk1tree_cost_aggregation.csv", index=False)
    bank["corpus_median_definition"] = corpus_median_block(agg)
    bank["aggregation_definitions"] = {
        "typical_instance_cost": ("median over the corpus of each instance's median "
                                  "over the timing repeats; the statistic Table 3's "
                                  "Time column carries and the one the manuscript's "
                                  "undefined phrase 'corpus median' names"),
        "corpus_throughput_cost": ("sum over the corpus of those same per-instance "
                                   "medians; the cost of running the whole benchmark"),
        "accuracy": ("MAPE, a mean over the same instances, so 'median' in "
                     "'corpus median' describes the cost column only"),
        "why_both": ("a median of medians does not compose, so no weighting of "
                     "Table 3's three bucket medians reproduces its Total row; the "
                     "throughput statistic does compose and disagrees with it by "
                     "more than an order of magnitude on TSPLIB EUC_2D"),
    }
    bank["pending_manifest_entries"] = resolve_pending(bank)
    (OUT / "hk1tree_cost_frontier_bank.json").write_text(
        json.dumps(bank, indent=1), encoding="utf-8")
    pd.DataFrame(bank["pending_manifest_entries"]).to_csv(
        OUT / "hk1tree_cost_pending_claims.csv", index=False)
    print(f"\nwrote hk1tree_cost_frontier_bank.json, hk1tree_cost_aggregation.csv "
          f"({len(agg)} rows), and {len(list(TAB.glob('frontier_*.tex')))} fragments")


if __name__ == "__main__":
    main()
