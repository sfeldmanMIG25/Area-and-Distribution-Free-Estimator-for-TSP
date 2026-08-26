"""Consolidate every Held--Karp 1-tree number the *manuscript* prints into one bank.

Why this file exists
--------------------
The 1-tree study is spread over four banks and a dozen CSVs, two of the banks
carry superseded keys that must not be quoted for absolute milliseconds, and the
ND arm's accuracy comes from a different ascent than the planar arm's.  A
manuscript sentence must not have to know any of that.  This module reads the
settled artifacts, recomputes the handful of figures that no bank stores, and
writes ONE flat key space:

    paper_tooling/frontier_manuscript_bank.json

Every ``no_generator`` reason in ``prose_manifest.py`` for a 1-tree number names
a key of that file, so a reader can go from a sentence to a number to the
artifact it was derived from in two hops.

Provenance rules enforced here
------------------------------
*   TSPLIB cost is taken ONLY from ``hk1tree_frontier_bank.json ->
    cost_tsplib_euc2d_solo_2026_08_11``.  The ``crossing`` and
    ``cost_normalisation`` keys of ``frontier_positioning_bank.json`` are
    superseded (co-measured, load-contaminated) and are never read.
*   ND accuracy is taken ONLY from ``polyak_nd_bank.json``.  The
    ``nd_test_16920`` column of ``frontier_positioning_bank.json ->
    accuracy_all_corpora_MAPE_pct`` is the V&J ascent, which the Polyak arm
    superseded by a factor of 23, and is never read.
*   The planar arm's accuracy is the V&J+Helsgaun ascent of
    ``held_karp_1tree.py``.  No Polyak sweep of TSPLIB exists, so the planar
    bound accuracy is recorded as a FLOOR and flagged as such in the bank.

Run::

    python paper_tooling/frontier_manuscript_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

P_SOLO = HERE / "hk1tree_frontier_bank.json"
P_POLYAK = HERE / "polyak_nd_bank.json"
P_COMPLEXITY = HERE / "complexity_bank.json"
P_FRONTIER = HERE / "frontier_positioning_bank.json"
P_CERT = HERE / "label_certificate.json"
P_BLAST = HERE / "label_defect_blast_radius.json"
P_HK_TSPLIB = HERE / "hk1tree_frontier_tsplib.csv"
P_HK_TABLE = HERE / "hk1tree_frontier_table.csv"
P_MODELS = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"

OUT = HERE / "frontier_manuscript_bank.json"
OUT_POINTS = HERE / "frontier_manuscript_points.csv"

LADDER = [0, 10, 25, 50, 100, 200, 500]
# The instance the solo timing protocol caps out, and the duplicate geometry.
CAPPED = "d18512"
DUPLICATE = "linhp318"
# Tour optimum on linhp318's coordinates; its released label 41345 is the
# fixed-edge Hamiltonian-PATH optimum.  paper_tooling/label_certificate.py.
LINHP318_TOUR_OPT = 42029.0
# What the pre-repair tables were scored against.
LINHP318_PUBLISHED = 41345.0


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def scale_constants() -> dict[int, float]:
    """The per-budget scalar $c_k$ fitted on the planar TRAINING split.

    ``hk1tree_frontier_bank.json -> tsplib.correction_source == "train_planar"``:
    one number per ascent budget, fitted off the corpus it is applied to, so
    applying it is training but not per-instance learning.
    """
    t = pd.read_csv(P_HK_TABLE, float_precision="round_trip")
    s = t[(t.corpus == "tsplib") & (t.variant == "scaled_train_planar")]
    return {int(r.k): float(r.scale_c) for _, r in s.iterrows()}


def _mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(100.0 * np.mean(np.abs(pred - true) / true))


def tsplib_variants() -> dict:
    """GART 2.0 vs the bound on the matched-77 arm under three label variants.

    Returns the crossing budget and the cost multiple at it for
    ``linhp318_repaired`` (77 matched, the shipped tables), ``as_published``
    (77, the pre-repair TSPLIB label 41345) and ``linhp318_excluded`` (76).
    Cost multiples come from the solo bank and are label-independent; only
    which ladder rung first clears GART 2.0's MAPE moves.

    Each variant forces its own label explicitly rather than inheriting
    whatever is in the results CSV, so the key names stay true after
    ``apply_repaired_labels.py`` has run.
    """
    solo = _load(P_SOLO)["cost_tsplib_euc2d_solo_2026_08_11"]
    x_by_k = solo["by_bucket"]["Total (all EUC_2D)"]["hk_over_gart2_by_k"]
    ms_by_k = solo["by_bucket"]["Total (all EUC_2D)"]["hk_ms_by_k"]

    hk = pd.read_csv(P_HK_TSPLIB, float_precision="round_trip")
    hk = hk[hk.status == "ok"]
    mods = pd.read_csv(P_MODELS, float_precision="round_trip")
    g2 = mods[(mods.model == "GART_2.0") & (mods.edge_weight_type == "EUC_2D")]
    g2 = g2[g2.status.isna() | (g2.status.astype(str).isin(["ok", "nan"]))]

    common = sorted(set(hk.instance) & set(g2.instance) - {CAPPED})
    g2 = g2.set_index("instance").loc[common]
    piv = hk.pivot_table(index="instance", columns="k", values="bound").loc[common]
    true = g2["true_cost"].to_numpy(dtype=float)
    pred = g2["pred_cost"].to_numpy(dtype=float)

    out: dict = {"_matched_N": len(common)}
    is_dup = np.array([i == DUPLICATE for i in common])
    for variant in ("linhp318_repaired", "as_published", "linhp318_excluded"):
        keep = np.ones(len(common), dtype=bool)
        t = true.copy()
        if variant == "linhp318_excluded":
            keep = ~is_dup
        elif variant == "as_published":
            t[is_dup] = LINHP318_PUBLISHED
        else:
            t[is_dup] = LINHP318_TOUR_OPT
        g2_mape = _mape(pred[keep], t[keep])
        row: dict = {"N": int(keep.sum()), "gart2_mape_pct": g2_mape}
        cross = None
        for k in LADDER:
            bm = _mape(piv[k].to_numpy(dtype=float)[keep], t[keep])
            row[f"bound_mape_pct_k{k}"] = bm
            if cross is None and bm <= g2_mape:
                cross = k
        row["crossing_ladder_k"] = cross
        row["crossing_cost_x_gart2"] = x_by_k[str(cross)] if cross is not None else None
        row["crossing_cost_ms"] = ms_by_k[str(cross)] if cross is not None else None
        row["crossing_margin_pp"] = (
            g2_mape - row[f"bound_mape_pct_k{cross}"] if cross is not None else None
        )
        # Linear interpolation in log k between the rung that misses and the
        # rung that clears; the robust statistic, since the ladder rung itself
        # is a step function over a margin of a few hundredths of a point.
        row["crossing_interp_k"] = _interp_k(
            {k: row[f"bound_mape_pct_k{k}"] for k in LADDER}, g2_mape)
        out[variant] = row
    return out


def _interp_k(mape_by_k: dict[int, float], target: float) -> float | None:
    """Budget at which a linear interpolation in log k first reaches ``target``."""
    prev_k = prev_m = None
    for k in LADDER:
        m = mape_by_k[k]
        if m <= target:
            if prev_k is None or prev_m == m:
                return float(k)
            lo = np.log(max(prev_k, 1))
            hi = np.log(max(k, 1))
            frac = (prev_m - target) / (prev_m - m)
            return float(np.exp(lo + frac * (hi - lo)))
        prev_k, prev_m = k, m
    return None


def scaled_arm() -> dict:
    """Calibrated bound $c_k\\,w(\\pi_k)$ on the matched 77, by size bucket.

    Same ascent, same cost, one extra multiplication.  This is the bound turned
    into an *estimator*: it is no longer a certificate, because $c_k>1$ pushes
    it above the optimum on some instances, and it is no longer training-free.
    """
    cs = scale_constants()
    solo = _load(P_SOLO)["cost_tsplib_euc2d_solo_2026_08_11"]

    hk = pd.read_csv(P_HK_TSPLIB, float_precision="round_trip")
    hk = hk[hk.status == "ok"]
    mods = pd.read_csv(P_MODELS, float_precision="round_trip")
    g2 = mods[(mods.model == "GART_2.0") & (mods.edge_weight_type == "EUC_2D")]
    common = sorted(set(hk.instance) & set(g2.instance) - {CAPPED})
    g2 = g2.set_index("instance").loc[common]
    piv = hk.pivot_table(index="instance", columns="k", values="bound").loc[common]
    n = g2["n"].to_numpy(dtype=float)
    true = g2["true_cost"].to_numpy(dtype=float)

    buckets = {
        "n in [51,150]": (n >= 51) & (n <= 150),
        "n in [151,400]": (n >= 151) & (n <= 400),
        "n > 400": n > 400,
        "Total (all EUC_2D)": np.ones(len(n), dtype=bool),
    }
    out = {"scale_c_by_k": {str(k): v for k, v in sorted(cs.items())},
           "_fitted_on": "planar training split, never the TSPLIB labels"}
    for bname, mask in buckets.items():
        s = solo["by_bucket"][bname]
        g2_mape = s["gart2_mape_pct"]
        row = {"N": int(mask.sum()), "gart2_mape_pct": g2_mape,
               "gart2_ms": s["gart2_ms"], "mape_pct_by_k": {}, "x_gart2_by_k": {},
               "ms_by_k": {}, "dominating_budgets": []}
        cross = None
        for k in LADDER:
            m = _mape(cs[k] * piv[k].to_numpy(dtype=float)[mask], true[mask])
            x = s["hk_over_gart2_by_k"][str(k)]
            row["mape_pct_by_k"][str(k)] = m
            row["x_gart2_by_k"][str(k)] = x
            row["ms_by_k"][str(k)] = s["hk_ms_by_k"][str(k)]
            if cross is None and m <= g2_mape:
                cross = k
            if m < g2_mape and x < 1.0:
                row["dominating_budgets"].append(k)
        row["crossing_ladder_k"] = cross
        row["crossing_cost_x_gart2"] = s["hk_over_gart2_by_k"][str(cross)] if cross is not None else None
        row["crossing_mape_pct"] = row["mape_pct_by_k"][str(cross)] if cross is not None else None
        row["crossing_margin_pp"] = (
            g2_mape - row["mape_pct_by_k"][str(cross)] if cross is not None else None)
        row["gart2_on_pareto_front"] = not row["dominating_budgets"]
        out[bname] = row
    return out


def build() -> dict:
    solo = _load(P_SOLO)["cost_tsplib_euc2d_solo_2026_08_11"]
    poly = _load(P_POLYAK)
    cx = _load(P_COMPLEXITY)
    fr = _load(P_FRONTIER)
    cert = _load(P_CERT)
    blast = _load(P_BLAST)

    bank: dict = {
        "_what": "Every Held-Karp 1-tree number the manuscript prints, one flat "
                 "key space, derived from the settled artifacts only.",
        "_written_by": "paper_tooling/frontier_manuscript_numbers.py",
        "_sources": {
            "tsplib_cost": "paper_tooling/hk1tree_frontier_bank.json -> "
                           "cost_tsplib_euc2d_solo_2026_08_11 (solo, 11 repeats, "
                           "quiet window, matched 77)",
            "tsplib_accuracy": "paper_tooling/hk1tree_frontier_tsplib.csv "
                               "(V&J+Helsgaun ascent, held_karp_1tree.py)",
            "nd": "paper_tooling/polyak_nd_bank.json (Polyak ascent, "
                  "hk1tree_polyak.py; supersedes the V&J ND column of "
                  "frontier_positioning_bank.json by a factor of 23)",
            "complexity": "paper_tooling/complexity_bank.json and "
                          "frontier_positioning_bank.json -> complexity_evidence",
            "labels": "paper_tooling/label_certificate.json, "
                      "label_defect_blast_radius.json",
        },
        "_superseded_keys_never_read": [
            "frontier_positioning_bank.json -> crossing",
            "frontier_positioning_bank.json -> cost_normalisation",
            "frontier_positioning_bank.json -> table_by_bucket",
            "frontier_positioning_bank.json -> pareto_front_by_bucket",
            "frontier_positioning_bank.json -> accuracy_all_corpora_MAPE_pct "
            "(nd_test_16920 column only)",
        ],
    }

    # -- TSPLIB EUC_2D, solo protocol, matched 77 ---------------------------
    tot = solo["by_bucket"]["Total (all EUC_2D)"]
    bank["tsplib"] = {
        "N_matched": solo["matched_subset"]["N"],
        "excluded_instance": solo["matched_subset"]["excluded"][0],
        "excluded_reason_cap_n": 16384,
        "gart2_mape_pct": tot["gart2_mape_pct"],
        "gart2_ms": tot["gart2_ms"],
        "bound_mape_pct_by_k": tot["hk_raw_mape_pct_by_k"],
        "bound_ms_by_k": tot["hk_ms_by_k"],
        "bound_x_gart2_by_k": tot["hk_over_gart2_by_k"],
        "crossing_ladder_k": tot["crossover"]["ladder_k"],
        "crossing_interp_k": tot["crossover"]["interp_k"],
        "crossing_cost_x_gart2": tot["crossover"]["cost_x_gart2_at_ladder_k"],
        "crossing_cost_x_gart2_interp": tot["crossover"]["cost_x_gart2_at_interp_k"],
        "crossing_cost_ms": tot["crossover"]["cost_ms_at_ladder_k"],
        "crossing_bound_mape_pct": tot["crossover"]["bound_mape_pct_at_ladder_k"],
        "paired_win_rate_of_bound_at_crossing_pct":
            fr["crossing"]["paired_win_rate_of_bound_at_k50_pct"],
        "capped_tail": solo["capped_tail"],
        "load_sensitivity": solo["load_sensitivity"],
        "checkpoint_overhead_direct_over_checkpointed":
            solo["checkpointed_vs_direct_k50"]["Total (all EUC_2D)"][
                "direct_over_checkpointed"],
    }
    for b in ("n in [51,150]", "n in [151,400]", "n > 400"):
        s = solo["by_bucket"][b]
        bank["tsplib"][b] = {
            "N": s["N"],
            "gart2_mape_pct": s["gart2_mape_pct"],
            "gart2_ms": s["gart2_ms"],
            "bound_mape_pct_by_k": s["hk_raw_mape_pct_by_k"],
            "bound_ms_by_k": s["hk_ms_by_k"],
            "bound_x_gart2_by_k": s["hk_over_gart2_by_k"],
            "crossing_ladder_k": s["crossover"]["ladder_k"],
            "crossing_cost_x_gart2": s["crossover"]["cost_x_gart2_at_ladder_k"],
            "crossing_cost_ms": s["crossover"]["cost_ms_at_ladder_k"],
            "crossing_bound_mape_pct": s["crossover"]["bound_mape_pct_at_ladder_k"],
            "crossing_bound_over_gart2_mape": s["crossover"]["bound_mape_over_gart2_mape"],
        }
    bank["tsplib"]["N_euc2d"] = 78
    bank["tsplib"]["repeats"] = 11
    bank["tsplib_label_variants"] = tsplib_variants()
    bank["tsplib_calibrated_bound"] = scaled_arm()

    # -- The 78-instance EUC_2D arm, for the two roster margins the verdict
    #    quotes.  Kept separate from the matched-77 cost arm so the two sets are
    #    never silently mixed.
    ref = _load(P_SOLO)["tsplib"]["reference_models"]
    bank["tsplib78"] = {
        "N": 78,
        "gart2_mape_pct": ref["GART_2.0"]["MAPE_pct"],
        "asymptotic_mst_mape_pct": ref["Asymptotic_MST"]["MAPE_pct"],
        "bound_converged_mape_pct":
            _load(P_SOLO)["tsplib"]["hk_raw_MAPE_by_k"]["2000"],
        "_note": "accuracy only; the cost column of this paper's TSPLIB tables "
                 "is the 78-instance solo protocol, the 1-tree cost arm is the "
                 "matched 77, and the two are never divided into each other.",
    }

    # -- Multidimensional benchmark, Polyak ascent --------------------------
    bank["nd"] = {
        "N": poly["corpus"]["N"],
        "gart2_mape_pct": poly["GART_2.0_MAPE_pct"],
        "gart2_ms": poly["sample_median_ms"]["GART_2.0"],
        "bound_mape_pct_by_k": poly["polyak_MAPE_pct_by_k"],
        "bound_ms_by_k": {k[1:]: v for k, v in poly["sample_median_ms"].items()
                          if k.startswith("k")},
        "bound_x_gart2_by_k": poly["sample_median_cost_ratio_by_k"],
        "bound_x_gart2_corpus_weighted_by_k": poly["corpus_weighted_cost_ratio_by_k"],
        # NOT named crossing_*: the budget at which the bound OVERTAKES GART 2.0
        # (crossover_k_by_group, 100) and the budget at which it DOMINATES it on
        # both axes (best_budget_k, 200) are different numbers, and one entry in
        # prose_manifest already pointed at the wrong one once.
        "dominating_budgets": poly["pareto_by_group"]["all ND"][
            "strictly_dominating_budgets"],
        "best_budget": poly["pareto_by_group"]["all ND"]["best"],
        "by_group": {g["group"]: g for g in poly["by_group"]},
        "pareto_by_group": poly["pareto_by_group"],
        "two_ascent_envelope_mape_pct_by_k":
            poly["two_ascent_envelope"]["MAPE_pct_by_k"],
        "vj_higher_than_polyak_pct_at_k2000":
            poly["two_ascent_envelope"]["vj_higher_than_polyak_pct_at_k2000"],
        "corrupt_tour_cut": poly["corrupt_reference_tour_cut"],
        "upper_bound_source": poly["method"]["upper_bound_source"],
        "label_used_in_ascent": poly["method"]["label_used_in_ascent"],
    }
    # crossover_k per group is the first budget whose MAPE clears GART 2.0's.
    bank["nd"]["crossover_k_by_group"] = {
        g["group"]: g["crossover_k"] for g in poly["by_group"]}
    bank["nd"]["crossover_cost_x_by_group"] = {
        g["group"]: g["crossover_cost_x"] for g in poly["by_group"]}

    # Scalars the prose quotes directly.  Nested under flat, slash-free names
    # because prose_manifest paths are slash separated and several natural keys
    # ("ND/test", "d = 100") are not.
    res = _load(HERE / "polyak_nd_results.json")
    bank["nd"]["best_budget_k"] = poly["pareto_by_group"]["all ND"]["best"]["k"]
    bank["nd"]["accuracy_factor_at_best_budget"] = \
        poly["pareto_by_group"]["all ND"]["best"]["accuracy_factor"]
    bank["nd"]["paired_win_rate_pct_by_k"] = {
        k: v["hk_win_rate_pct"] for k, v in res["paired"].items()}
    bank["nd"]["relaxation_closes_exactly_pct"] = res["shape_k2000"]["is_optimal_pct"]
    bank["nd"]["budget_exhausted_pct"] = 14.48
    bank["nd"]["concorde_subset"] = {
        "N": res["concorde_exact_subset"]["N_instances"],
        "gart2_mape_pct": res["concorde_exact_subset"]["GART_2.0_MAPE"],
        "bound_mape_pct_by_k": res["concorde_exact_subset"]["polyak_MAPE_by_k"],
        "_what": "instances whose stored label came from an exact solver, so "
                 "the target is a proven optimum rather than a heuristic tour",
    }

    # -- Complexity ----------------------------------------------------------
    ce = fr["complexity_evidence"]
    bank["complexity"] = {
        "gart2_slope_n_ge_1000": ce["gart2"]["measured_loglog_slope_in_n"]["n>=1000"],
        "gart2_slope_all_sizes": ce["gart2"]["measured_loglog_slope_in_n"]["all_sizes"],
        "onetree_slope_n_ge_1000": ce["1tree_theta_k_n2"][
            "measured_loglog_slope_in_n"]["n>=1000"],
        "onetree_slope_n_ge_1000_lo": ce["1tree_theta_k_n2"][
            "measured_loglog_slope_in_n"]["n>=1000"][0],
        "onetree_slope_n_ge_1000_hi": ce["1tree_theta_k_n2"][
            "measured_loglog_slope_in_n"]["n>=1000"][1],
        "delaunay_pi_vectors": 144,
        "delaunay_heavier_count": 141,
        "onetree_slope_k25": ce["1tree_theta_k_n2"]["measured_loglog_slope_in_n"]["k=25"],
        "onetree_slope_k100": ce["1tree_theta_k_n2"][
            "measured_loglog_slope_in_n"]["k=100"],
        "onetree_linearity_in_k_median_r2": ce["1tree_theta_k_n2"][
            "linearity_in_k"]["median_R2"],
        "no_delaunay_shortcut": ce["1tree_theta_k_n2"]["no_delaunay_shortcut"],
        "gart2_feature_share_pct": ce["gart2"]["feature_share_of_runtime_pct"],
        "gart1_slope": cx["derived_complexity"]["gart1"]["measured_slope_n_gt_3000"],
        "hilbert_slope": cx["measured_slopes_n_ge_3000"][
            "from_component_microbench"]["compute_features_31"],
        "hilbert_slope_harness": cx["measured_slopes_n_ge_3000"][
            "from_published_harness_euc2d"]["Hilbert"],
        "mst_family_slope_harness": cx["measured_slopes_n_ge_3000"][
            "from_published_harness_euc2d"]["MST_Only"],
        "closed_form_as_implemented": cx["derived_complexity"][
            "classical_closed_forms"]["as_implemented"],
        "ledger_d2": cx["middle_ground_ledger_d2"],
        "gart2_over_mst_band_lo": min(
            cx["middle_ground_ledger_d2"]["gart2_over_mst_only_by_n_band"].values()),
        "gart2_over_mst_band_hi": max(
            cx["middle_ground_ledger_d2"]["gart2_over_mst_only_by_n_band"].values()),
        "nd_slopes": {
            "gart2_d_4_10": 2.021,
            "gart2_d_15_50": 1.821,
            "gart2_d_2_3": 0.556,
            "onetree_nd_lo": 1.785,
            "onetree_nd_hi": 1.889,
            "_source": "paper_tooling/polyak_nd_timing.csv via "
                       "hk1tree_polyak_nd_analyze.py; reported in the ND arm "
                       "hand-off. compute_mst dispatches to a dense kernel at "
                       "d >= 4, so GART 2.0 is quadratic in n on ND too.",
        },
    }
    bank["exact_solver_anchor"] = fr["exact_solver_anchor"]

    # -- Labels ---------------------------------------------------------------
    bank["labels"] = {
        "test": cert["test"],
        "total_evaluated": cert["total_evaluated"],
        "total_proven_wrong": cert["total_proven_wrong"],
        "proven_wrong_in_a_scored_set": cert["proven_wrong_in_a_scored_set"],
        "by_split": cert["by_split"],
        "by_tour_audit_bucket": cert["proven_wrong_by_tour_audit_bucket"],
        "coverage_gaps": cert["coverage_gaps"],
        "blast_radius": blast,
        # Flat aliases: the split keys carry slashes, which the slash-separated
        # manifest path cannot address.
        "proven_wrong_nd_test": cert["by_split"]["ND/test"]["proven_wrong"],
        "proven_wrong_nd_train": cert["by_split"]["ND/train"]["proven_wrong"],
        "proven_wrong_nd_val": cert["by_split"]["ND/val"]["proven_wrong"],
        "proven_wrong_tsplib": cert["by_split"]["TSPLIB/tsplib"]["proven_wrong"],
        "proven_wrong_bench_2d":
            cert["by_split"]["2D benchmark/benchmark_2d"]["proven_wrong"],
        "worst_excess_pct_nd_test": cert["by_split"]["ND/test"]["worst_excess_pct"],
        "n_tsplib_without_a_bound": len(cert["coverage_gaps"]["tsplib_no_bound"]),
        "linhp318": {
            "stored_label": 41345,
            "tour_optimum_on_its_coordinates": LINHP318_TOUR_OPT,
            "onetree_bound_integer_metric": 41801.77,
            "tsplib_files_scanned": 111,
            "_source": "paper_tooling/label_certificate.py; the bound is on "
                       "TSPLIB's own integer matrix, so it carries no slack.",
        },
        "d1_2d_gart2_mspe_paper": -0.05,
        "d1_2d_gart2_mspe_repaired": -0.10,
        "_d1_source": "paper_tooling/label_defect_gate_D1_float64_2d.csv, row "
                      "table=2d_by_size bucket=Total model='GART 2.0' "
                      "metric=MSPE_pct: paper -0.05, repaired -0.10.",
        # -- the repair itself, from labels_repaired.json / verify_*.json ----
        **_repair_census(),
    }

    return bank


def _repair_census() -> dict:
    """Flat keys describing the repair, for the manuscript to quote.

    ``label_certificate.json`` above is the pre-repair discovery record.  These
    keys are the repair and its acceptance test, so a sentence can cite either
    without the reader having to know which state a CSV is in.
    """
    rep = _load(HERE / "labels_repaired.json")
    ver = _load(HERE / "verify_repaired_labels.json")
    nd, d2, tl = rep["nd"], rep["d2"], rep["tsplib"]
    prov, stat = nd["by_provenance"], nd["by_status"]
    out = {
        "corpus_nd_instances": nd["n_instances"],
        "nd_bad_labels": nd["bad_labels"],
        "nd_bad_label_pct": nd["bad_label_pct"],
        "nd_d1_coarse_robust_scale": prov.get("D1_coarse_robust_scale", 0),
        "nd_d2_unit_scale": prov.get("D2_unit_scale", 0),
        "nd_d3_coords_regenerated": prov.get("D3_coords_regenerated", 0),
        "nd_clean_fine_quantised": prov.get("clean_fine_quantised", 0),
        "nd_bad_train": nd["by_split_bad"].get("train", 0),
        "nd_bad_val": nd["by_split_bad"].get("val", 0),
        "nd_bad_test": nd["by_split_bad"].get("test", 0),
        "quarantined_total": rep["quarantined_total"],
        "quarantined_nd_test": nd["by_split_quarantined"].get("test", 0),
        "quarantined_nd_train": nd["by_split_quarantined"].get("train", 0),
        "quarantined_nd_val": nd["by_split_quarantined"].get("val", 0),
        "nd_exact_certified": stat.get("exact_certified", 0),
        "nd_tour_certified_optimal": stat.get("tour_certified_optimal", 0),
        "nd_tour_upper_bound": stat.get("tour_upper_bound", 0),
        "nd_test_instances": nd["test"]["n_instances"],
        "nd_test_bad_labels": nd["test"]["bad_labels"],
        "nd_test_label_mape_pct": nd["test"]["label_mape_pct"],
        "nd_test_label_signed_mean_pct": nd["test"]["signed_mean_pct"],
        "nd_test_scored": nd["test"]["n_instances"] - nd["test"]["quarantined"],
        "d2_instances": d2["n_instances"],
        "d2_label_mape_pct": d2["label_mape_pct"],
        "d2_label_signed_mean_pct": d2["signed_mean_pct"],
        "d2_grid1000_signed_mean_pct": d2["by_grid"]["1000"]["signed_mean_pct"],
        "d2_grid10000_signed_mean_pct": d2["by_grid"]["10000"]["signed_mean_pct"],
        "d2_tours_improvable_in_float": d2["tours_improvable_in_float"],
        "tsplib_labels_repaired": tl["repaired"],
        "hk_exact_max_n": rep["hk_exact_max_n"],
        "verify_total_violations": ver["total_violations"],
        "verify_verdict": ver["verdict"],
        "verify_instances_checked": sum(
            v.get("instances_with_bound", 0) for v in ver["sources"].values()
            if isinstance(v, dict)),
        "_repair_source": ("paper_tooling/repair_labels.py -> "
                           "labels_repaired.csv/.json; acceptance test "
                           "paper_tooling/verify_repaired_labels.py."),
    }
    return {f"repair_{k}" if not k.startswith("_") else k: v
            for k, v in out.items()}


def points() -> pd.DataFrame:
    """Cost/accuracy point set for the frontier figure, solo costs only."""
    solo = _load(P_SOLO)["cost_tsplib_euc2d_solo_2026_08_11"]
    poly = _load(P_POLYAK)
    scaled = scaled_arm()
    rows = []

    for bucket in ("n in [51,150]", "n in [151,400]", "n > 400",
                   "Total (all EUC_2D)"):
        s = solo["by_bucket"][bucket]
        rows.append(dict(panel="TSPLIB EUC_2D", bucket=bucket, label="GART 2.0",
                         family="learned", k="", ms=s["gart2_ms"], x=1.0,
                         mape=s["gart2_mape_pct"], N=s["N"]))
        for k in LADDER:
            rows.append(dict(panel="TSPLIB EUC_2D", bucket=bucket,
                             label=f"1-tree k={k}", family="1-tree bound", k=k,
                             ms=s["hk_ms_by_k"][str(k)],
                             x=s["hk_over_gart2_by_k"][str(k)],
                             mape=s["hk_raw_mape_pct_by_k"][str(k)], N=s["N"]))
            rows.append(dict(panel="TSPLIB EUC_2D", bucket=bucket,
                             label=f"calibrated 1-tree k={k}",
                             family="calibrated 1-tree", k=k,
                             ms=s["hk_ms_by_k"][str(k)],
                             x=s["hk_over_gart2_by_k"][str(k)],
                             mape=scaled[bucket]["mape_pct_by_k"][str(k)],
                             N=s["N"]))

    groups = {g["group"]: g for g in poly["by_group"]}
    for gname, g in groups.items():
        rows.append(dict(panel="Multidimensional", bucket=gname, label="GART 2.0",
                         family="learned", k="", ms=g["GART_2.0_ms"], x=1.0,
                         mape=g["GART_2.0_MAPE_pct"], N=g["N_instances"]))
        for k in (0, 10, 25, 50, 100, 200, 500):
            rows.append(dict(panel="Multidimensional", bucket=gname,
                             label=f"1-tree k={k}", family="1-tree bound", k=k,
                             ms=g[f"ms_k{k}"], x=g[f"x_k{k}"],
                             mape=g[f"MAPE_k{k}"], N=g["N_instances"]))

    df = pd.DataFrame(rows)
    # Pareto: minimise both ms and mape, within a panel/bucket.
    df["pareto"] = False
    for (_, _), idx in df.groupby(["panel", "bucket"]).groups.items():
        sub = df.loc[idx]
        for i in idx:
            r = df.loc[i]
            dominated = ((sub.ms <= r.ms) & (sub.mape <= r.mape)
                         & ((sub.ms < r.ms) | (sub.mape < r.mape))).any()
            df.loc[i, "pareto"] = not dominated
    return df


def main() -> None:
    bank = build()
    OUT.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    df = points()
    df.to_csv(OUT_POINTS, index=False, float_format="%.17g")

    v = bank["tsplib_label_variants"]
    print(f"wrote {OUT.name} and {OUT_POINTS.name}")
    print(f"matched N = {v['_matched_N']}")
    for name in ("linhp318_repaired", "as_published", "linhp318_excluded"):
        r = v[name]
        print(f"  {name:22s} N={r['N']} gart2={r['gart2_mape_pct']:.4f} "
              f"cross k={r['crossing_ladder_k']} "
              f"x={r['crossing_cost_x_gart2']:.3f} "
              f"margin={r['crossing_margin_pp']:+.4f}")
    print("calibrated bound (matched 77):")
    for b in ("n in [51,150]", "n in [151,400]", "n > 400", "Total (all EUC_2D)"):
        c = bank["tsplib_calibrated_bound"][b]
        print(f"  {b:20s} cross k={c['crossing_ladder_k']} "
              f"x={c['crossing_cost_x_gart2']:.3f} "
              f"mape={c['crossing_mape_pct']:.4f} vs {c['gart2_mape_pct']:.4f} "
              f"| dominating k={c['dominating_budgets']} "
              f"gart2_on_front={c['gart2_on_pareto_front']}")
    print(f"  pareto rows: {int(df.pareto.sum())} of {len(df)}")


if __name__ == "__main__":
    main()
