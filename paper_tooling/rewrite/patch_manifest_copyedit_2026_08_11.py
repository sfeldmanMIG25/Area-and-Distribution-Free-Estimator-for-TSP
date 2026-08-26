"""Register every numeral the 2026-08-11 copy-edit introduced, and fix two entries.

Byte-level read_bytes / replace / write_bytes with exactly-once assertions, same
discipline as the manuscript patch.

1. ``intro.v4.n_features`` -- its anchor tail was "Two of those", which the F3
   anaphor repair replaced with "Three of those".
2. ``methods.probe.n_hi`` -- its reason silently added the "in the synthetic
   corpora" qualifier the manuscript sentence lacked (F4).
3. Forty-four new ``Claim`` entries appended: every numeral the copy-edit added,
   plus the three line-noise numerals the F6 finding named and the on-face median
   that pins their slice.
"""
from __future__ import annotations

from pathlib import Path

MANIFEST = Path(r"D:\Area-and-Distribution-Free-Estimator-for-TSP\paper_tooling\prose_manifest.py")

OLD_ANCHOR = (
    b'        anchor=r"a boosted variant on an extended {v}-feature block. Two of those",'
)
NEW_ANCHOR = (
    b'        anchor=r"a boosted variant on an extended {v}-feature block. Three of those",'
)

OLD_NHI = b'''            "paper_tooling/v4_study.py::PROBE_N_GRID upper endpoint = 4000, four times "
            "the largest node count evaluated anywhere in this paper (n=1000 in the "
            "synthetic corpora). Protocol constant. Settle by banking the probe "
            "protocol constants under probe_* keys."'''

NEW_NHI = b'''            "paper_tooling/v4_study.py::PROBE_N_GRID upper endpoint = 4000, four times "
            "the largest node count in the SYNTHETIC corpora (n=1000). It is NOT four "
            "times the largest node count this paper evaluates: the TSPLIB EUC_2D "
            "benchmark runs to n=18512 (d18512) and pla85900 is scored at n=85900, so "
            "the grid reaches 4.7 percent of the largest evaluated n. Until 2026-08-11 "
            "this reason carried the 'in the synthetic corpora' qualifier while the "
            "manuscript sentence claimed four times the largest node count anything in "
            "the paper is evaluated at; the sentence was narrowed to match. Protocol "
            "constant. Settle by banking the probe protocol constants under probe_* keys."'''

# ---------------------------------------------------------------------------
NEW_CLAIMS = rb'''
    # -- 2026-08-11 copy-edit: the constraint-transfer control -----------------
    #
    #    paper_tooling/constraint_transfer.py refits the extended-block variant
    #    with GART 2.0's two monotone constraints at seven seeds against a
    #    protocol registered before the first fit, and refits GART 2.0 itself as
    #    the matched control.  Its 1,934 ctrans_* keys are carried into
    #    paper_numbers.json by build_paper_tables.py, so every number below has a
    #    bank key and is checked, not merely recorded.
    Claim(
        id="abstract.ctrans.enforced_pct",
        anchor=r"for the network. That {v}\% is enforced inside the tree builder",
        expect="bank:cons_probe_gart_2_0_dimension_pct_nonincr_deployed",
        tol=("dp", 1),
        note="Restates the shipped model's probe figure in the clause that "
             "withdraws the inductive-bias reading of it.",
    ),
    Claim(
        id="abstract.ctrans.mono_pct",
        anchor=r"same constraints and recovers {v}\% on both axes at every one of seven seeds",
        expect="bank:ctrans_probe_v4_mono_32f_dimension_pct_nonincr_deployed_min",
        tol=("dp", 1),
        note="The MINIMUM over the seven seeds, because the sentence claims every "
             "seed. Median and max are 100.0 as well.",
    ),
    Claim(
        id="abstract.ctrans.cost_max_pp",
        anchor=r"seeds for at most {v} percentage points of MAPE: consistency is a flag",
        expect="bank:ctrans_cost_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
        note="Largest of the four per-stratum median paired constraint costs "
             "(nd_test -0.0022, bench2d 0.0391, tsplib_euc2d 0.0488, "
             "tsplib_noneuc 0.0456), so TSPLIB EUC_2D is the binding one.",
    ),
    Claim(
        id="abstract.ctrans.gart_tsplib_mape",
        anchor=r"constrained variant {v}\% against {~}\% on median MAPE with disjoint seed bands",
        expect="bank:ctrans_strata_gart2_refit_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="abstract.ctrans.mono_tsplib_mape",
        anchor=r"constrained variant {~}\% against {v}\% on median MAPE with disjoint seed bands",
        expect="bank:ctrans_strata_v4_mono_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="intro.ctrans.mono_pct",
        anchor=r"extended-block variant and recovers {v}\% on both axes at all seven",
        expect="bank:ctrans_probe_v4_mono_32f_dimension_pct_nonincr_deployed_min",
        tol=("dp", 1),
    ),
    Claim(
        id="intro.ctrans.cost_2d_pp",
        anchor=r"seeds, for a median {v} percentage points of 2D MAPE and at most",
        expect="bank:ctrans_cost_32f_bench2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="intro.ctrans.cost_max_pp",
        anchor=r"of 2D MAPE and at most {v} points on any stratum",
        expect="bank:ctrans_cost_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="intro.ctrans.gart_tsplib_mape",
        anchor=r"constrained variant {v}\% against {~}\% on median MAPE across seven seeds",
        expect="bank:ctrans_strata_gart2_refit_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="intro.ctrans.mono_tsplib_mape",
        anchor=r"constrained variant {~}\% against {v}\% on median MAPE across seven seeds",
        expect="bank:ctrans_strata_v4_mono_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="intro.ctrans.tsplib_p",
        anchor=r"with disjoint bands and a paired Wilcoxon $p={v}$ over {~} instances",
        expect="bank:ctrans_paired_tsplib_euc2d_wilcoxon_p",
        tol=("dp", 5),
    ),
    Claim(
        id="intro.ctrans.tsplib_n",
        anchor=r"Wilcoxon $p={~}$ over {v} instances. The accuracy also costs time",
        expect="bank:ctrans_paired_tsplib_euc2d_n",
        tol="exact",
    ),
    Claim(
        id="methods.ctrans.unc_dim_median",
        anchor=r"that variant holds a median {v}\% of the dimension sweeps and",
        expect="bank:ctrans_probe_v4_unc_32f_dimension_pct_nonincr_deployed_median",
        tol=("dp", 1),
        note="Median over seven unconstrained refits. The manuscript's 9.1 for "
             "this variant is the shipped artifact's single-seed value and is the "
             "MAXIMUM of this band (3.3-9.1), so the printed contrast against "
             "100% is conservative.",
    ),
    Claim(
        id="methods.ctrans.unc_n_median",
        anchor=r"of the dimension sweeps and {v}\% of the size sweeps, so the two figures",
        expect="bank:ctrans_probe_v4_unc_32f_n_customers_pct_nonincr_deployed_median",
        tol=("dp", 1),
        note="Median over seven unconstrained refits; the printed 78.6 is this "
             "band's maximum (33.2-78.6).",
    ),
    Claim(
        id="methods.ctrans.mono_pct",
        anchor=r"non-increasing constraints returns {v}\% non-increasing sweeps on both axes",
        expect="bank:ctrans_probe_v4_mono_32f_dimension_pct_nonincr_deployed_min",
        tol=("dp", 1),
    ),
    Claim(
        id="methods.ctrans.cost_2d_pp",
        anchor=r"a median {v} percentage points of MAPE on the 2D benchmark",
        expect="bank:ctrans_cost_32f_bench2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="methods.ctrans.cost_tsplib_pp",
        anchor=r"percentage points of MAPE on the 2D benchmark, {v} on TSPLIB EUC\_2D and",
        expect="bank:ctrans_cost_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="methods.ctrans.cost_noneuc_pp",
        anchor=r"{v} on the screened non-Euclidean set, and costs it nothing",
        expect="bank:ctrans_cost_32f_tsplib_noneuc_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="methods.ctrans.mono_2d_mape",
        anchor=r"keeps the 2D benchmark, {v}\% against {~}\% on median MAPE",
        expect="bank:ctrans_strata_v4_mono_32f_bench2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="methods.ctrans.gart_2d_mape",
        anchor=r"keeps the 2D benchmark, {~}\% against {v}\% on median MAPE",
        expect="bank:ctrans_strata_gart2_refit_bench2d_mape_median",
        tol=("dp", 3),
        note="A matched refit of GART 2.0 under the transfer protocol, not the "
             "shipped model's published 2.904 on this benchmark: the shipped fit "
             "is train-only with early stopping, the refit is not.",
    ),
    Claim(
        id="methods.ctrans.mono_tsplib_mape",
        anchor=r"loses TSPLIB EUC\_2D, {v}\% against {~}\%, disjoint bands at a paired",
        expect="bank:ctrans_strata_v4_mono_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="methods.ctrans.gart_tsplib_mape",
        anchor=r"loses TSPLIB EUC\_2D, {~}\% against {v}\%, disjoint bands at a paired",
        expect="bank:ctrans_strata_gart2_refit_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="methods.ctrans.tsplib_p",
        anchor=r"disjoint bands at a paired Wilcoxon $p={v}$ over {~} instances",
        expect="bank:ctrans_paired_tsplib_euc2d_wilcoxon_p",
        tol=("dp", 5),
    ),
    Claim(
        id="methods.ctrans.tsplib_n",
        anchor=r"Wilcoxon $p={~}$ over {v} instances. Monotone consistency",
        expect="bank:ctrans_paired_tsplib_euc2d_n",
        tol="exact",
    ),
    Claim(
        id="discussion.ctrans.mono_pct",
        anchor=r"constraints on the variant restores {v}\% on both axes at all seven seeds",
        expect="bank:ctrans_probe_v4_mono_32f_dimension_pct_nonincr_deployed_min",
        tol=("dp", 1),
    ),
    Claim(
        id="discussion.ctrans.cost_max_pp",
        anchor=r"seven seeds for at most {v} percentage points of MAPE on any stratum",
        expect="bank:ctrans_cost_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="conclusion.ctrans.cost_max_pp",
        anchor=r"seven seeds for at most {v} percentage points of MAPE, and the constrained",
        expect="bank:ctrans_cost_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="conclusion.ctrans.gart_tsplib_mape",
        anchor=r"constrained variant {v}\% against {~}\% on median MAPE with disjoint seven-seed bands",
        expect="bank:ctrans_strata_gart2_refit_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),
    Claim(
        id="conclusion.ctrans.mono_tsplib_mape",
        anchor=r"constrained variant {~}\% against {v}\% on median MAPE with disjoint seven-seed bands",
        expect="bank:ctrans_strata_v4_mono_32f_tsplib_euc2d_mape_median",
        tol=("dp", 3),
    ),

    # -- 2026-08-11 copy-edit: the two ranking measures the Conclusion omitted -
    Claim(
        id="conclusion.rank.close10_v4",
        anchor=r"the variant orders {v}\% of that band against",
        expect="bank:rank_tsplib_euc2d_gart_2_0_v4_features_close10_pct",
        tol=("dp", 2),
        note="Adverse: the extended-block variant leads GART 2.0 on the 10% "
             "close-pair band, which the sentence previously omitted while naming "
             "the variant in the clause before it.",
    ),
    Claim(
        id="conclusion.rank.close10_gart",
        anchor=r"of that band against GART 2.0's {v}\%, at Spearman",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close10_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="conclusion.rank.spearman_v4",
        anchor=r"at Spearman {v} against {~} and Kendall",
        expect="bank:rank_tsplib_euc2d_gart_2_0_v4_features_spearman_rho",
        tol=("dp", 6),
        note="Six decimals because the two coefficients differ in the fourth: "
             "0.999248 against 0.999222. Rounding to the house three decimals "
             "would print the adverse comparison as a tie.",
    ),
    Claim(
        id="conclusion.rank.spearman_gart",
        anchor=r"at Spearman {~} against {v} and Kendall",
        expect="bank:rank_tsplib_euc2d_gart_2_0_spearman_rho",
        tol=("dp", 6),
    ),
    Claim(
        id="conclusion.rank.kendall_v4",
        anchor=r"and Kendall {v} against {~}. It is also the more expensive estimator",
        expect="bank:rank_tsplib_euc2d_gart_2_0_v4_features_kendall_tau",
        tol=("dp", 6),
    ),
    Claim(
        id="conclusion.rank.kendall_gart",
        anchor=r"and Kendall {~} against {v}. It is also the more expensive estimator",
        expect="bank:rank_tsplib_euc2d_gart_2_0_kendall_tau",
        tol=("dp", 6),
    ),

    # -- 2026-08-11 copy-edit: the Conclusion now matches the provenance body --
    #    The clause it replaces asserted "their cost labels remain internally
    #    consistent", which Section 3.3 had already withdrawn.
    Claim(
        id="conclusion.provenance.solver_agree",
        anchor=r"the agreement of the two solvers on {v} of those rows",
        no_generator=(
            "paper_tooling/audit_reference_tours.py section (e), printed field "
            "'concorde_length == lkh_length' over bucket=='corrupt' = 166 of 184; "
            "all 184 corrupt rows carry both lengths and 18 disagree. Re-derived "
            "on 2026-08-11 by reading concorde_length and lkh_length out of "
            "solutions/<name>.sol.json for the 184 instance_name values with "
            "bucket=='corrupt' in paper_tooling/reference_tour_audit.csv, which "
            "does not itself carry the two solver columns. Settle by having "
            "audit_reference_tours.py write the solver columns into that CSV and "
            "bank the corrupt-bucket census under provenance_audit_* keys."
        ),
        tol="exact",
    ),
    Claim(
        id="conclusion.provenance.z_cells_over",
        anchor=r"because {v} of the {~} affected cells depart",
        no_generator=_zcell("printed field 'cells with |z| > 1.6' = 17"),
        tol="exact",
    ),
    Claim(
        id="conclusion.provenance.z_cells",
        anchor=r"because {~} of the {v} affected cells depart",
        no_generator=_zcell("printed field 'cells containing an affected row' = 29"),
        tol="exact",
    ),
    Claim(
        id="conclusion.provenance.z_threshold",
        anchor=r"unaffected neighbours by more than {v} standard deviations",
        no_generator=_zcell("the module constant THRESHOLD = 1.6"),
        tol=("dp", 1),
    ),
    Claim(
        id="conclusion.provenance.drop184_bound",
        anchor=r"moves every reported metric by less than {v} percentage points, but the released",
        no_generator=(
            "Sensitivity bound restated from Section 3.3: the largest change in "
            "aggregate MAPE across all benchmarked estimators when the 184 "
            "provenance-corrupt instances are dropped from the multidimensional "
            "benchmark. GART 2.0 itself moves 0.6201 -> 0.6182. No artifact "
            "exports the per-estimator delta table; the body sentence quotes the "
            "same bound. Settle by having audit_reference_tours.py emit a "
            "drop-184 re-score per estimator into paper_numbers.json under "
            "provenance_drop184_delta_* keys."
        ),
        tol=("dp", 2),
    ),

    # -- 2026-08-11 copy-edit: the Line Noise on-face relation is rank-like ----
    #    The printed 0.80 was the Pearson r on this slice, in a sentence whose
    #    claim is monotone tracking; the label was kept and the value replaced
    #    with the Spearman rho on the identical slice.
    Claim(
        id="results_2d.linenoise.on_face_median",
        anchor=r"a median {v}\% of points lie exactly on a face",
        no_generator=_LINENOISE(
            "median on-face fraction over the slice = 0.5985714286"),
        tol=("dp", 1),
        scale=100.0,
    ),
    Claim(
        id="results_2d.linenoise.rank_corr",
        anchor=r"tracks that fraction closely (Spearman {v}, rising monotonically",
        no_generator=_LINENOISE(
            "Spearman rho between on-face fraction and alpha = 0.8590870 "
            "(p = 2.49e-27). The Pearson r on the same slice is 0.8007114, which "
            "is what the manuscript printed as 0.80 under a Spearman label until "
            "2026-08-11; the label was correct for the claim and the value was "
            "not, so the value moved"),
        tol=("dp", 2),
    ),
    Claim(
        id="results_2d.linenoise.q1_alpha",
        anchor=r"rising monotonically from {v} to {~} across its quartiles",
        no_generator=_LINENOISE(
            "median alpha in the lowest on-face quartile = 1.1795406971"),
        tol=("dp", 3),
    ),
    Claim(
        id="results_2d.linenoise.q4_alpha",
        anchor=r"rising monotonically from {~} to {v} across its quartiles",
        no_generator=_LINENOISE(
            "median alpha in the highest on-face quartile = 1.4311880118; the "
            "four quartile medians are 1.1795, 1.2863, 1.3292, 1.4312, so the "
            "rise the sentence calls monotone is monotone"),
        tol=("dp", 3),
    ),
]
'''

# Reason helper for the four line-noise entries, inserted next to the others.
LINENOISE_HELPER = rb'''

def _LINENOISE(field: str) -> str:
    """Reason for a Line Noise on-face number: exact slice and recomputation."""
    return (
        f"{field}. Recomputed 2026-08-11 over the 2D benchmark's 210 "
        "Generalized_TSP_Analysis/instances/TSP-line_noise-n*.json files "
        "restricted to n>=200, which is 90 instances: on-face fraction is the "
        "share of points with an x or y coordinate equal to 0 or to the grid "
        "side G parsed from the instance name, and alpha is true_alpha from "
        "Generalized_TSP_Analysis/benchmark_checkpoints/base_ground_truth_2d.csv. "
        "paper_tooling/corpus_statistics.py::linenoise_geometry reads the same "
        "files but measures only rho_measured and kurtosis, so no artifact "
        "exports the on-face fraction or its correlation with alpha. Settle by "
        "having linenoise_geometry emit on_face_frac per instance and "
        "corpus_statistics bank the slice's median, its quartile alpha medians "
        "and its Spearman rho under corpus_linenoise_onface_* keys."
    )

'''

HELPER_ANCHOR = rb'''
def _zcell(row: str) -> str:'''


def main() -> int:
    raw = MANIFEST.read_bytes()
    before = len(raw)
    applied: list[str] = []
    missing: list[str] = []

    def sub(tag: str, old: bytes, new: bytes) -> None:
        nonlocal raw
        hits = raw.count(old)
        if hits != 1:
            missing.append(f"{tag} ({hits} matches)")
            return
        raw = raw.replace(old, new, 1)
        applied.append(tag)

    def crlf(b: bytes) -> bytes:
        return b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")

    sub("intro.v4.n_features-anchor", OLD_ANCHOR, NEW_ANCHOR)
    sub("methods.probe.n_hi-reason", crlf(OLD_NHI), crlf(NEW_NHI))
    sub("_LINENOISE-helper", crlf(HELPER_ANCHOR),
        crlf(LINENOISE_HELPER) + crlf(HELPER_ANCHOR))

    # Append the new claims by replacing the final list terminator.
    tail = b"\r\n        note=\"Stratum size. The predecessor scores all 23; GART 2.0 declines si1032.\",\r\n    ),\r\n]\r\n"
    if raw.count(tail) == 1:
        raw = raw.replace(tail, tail[:-len(b"]\r\n")] + NEW_CLAIMS.replace(b"\n", b"\r\n").lstrip(b"\r\n"), 1)
        applied.append("append-44-claims")
    else:
        missing.append(f"append-44-claims ({raw.count(tail)} matches)")

    print(f"applied : {len(applied)}")
    for t in applied:
        print(f"  + {t}")
    print(f"missing : {len(missing)}")
    for t in missing:
        print(f"  ! {t}")
    if missing:
        print("\nNOT WRITTEN.")
        return 1

    MANIFEST.write_bytes(raw)
    print(f"\nwritten: {before} -> {len(raw)} bytes (+{len(raw) - before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
