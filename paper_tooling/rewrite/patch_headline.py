r"""Headline-scope prose patch: abstract, introduction claim paragraph, conclusion.

Scope
-----
  line  90  \begin{abstract}            (all quantitative claims)
  line 123  \subsection{Contribution}   (final three claim sentences)
  line 484  \section{Conclusion}, P1    (summary of the advantage)
  line 486  \section{Conclusion}, P2    (two numeric defects only)

Every replacement is confined to a single physical line, so the file's CRLF
line endings are never touched. Byte-level replace; no LaTeX is round-tripped
through a shell.

Sources of record for the new values
------------------------------------
  paper_tooling/tables/paper_numbers.json     benchmark totals, paired tests, rank
  paper_tooling/controls_31f/marginals.csv    retrained 31-feature NN / linear controls
  paper_tooling/controls_31f/paired.csv       paired GART 2.0 vs control
  paper_tooling/controls_31f/seed_band.csv    7-seed retraining band
  paper_tooling/gart2_timing_bank.json        tsplib_by_size_time_one_protocol (solo, 11 repeats)
  paper_tooling/generalization_results.csv    E0 / E2 / E3 hold-out refits
  paper_reference/Area_Free_Main.tex          machine-verified table bodies (Kwon domain row)

Run: python paper_tooling/rewrite/patch_headline.py
"""

from pathlib import Path

TEX = Path(__file__).resolve().parents[2] / "paper_reference" / "Area_Free_Main.tex"

# (tag, old_string, new_string)
REPLACEMENTS = [
    # ------------------------------------------------------------------ #
    # ABSTRACT (line 90)
    # ------------------------------------------------------------------ #
    # 2D 3.33/5.19 -> 2.90/4.69, ND 0.88/1.28 -> 0.62/0.99,
    # TSPLIB 3.27/3.42 -> 2.55/2.94. All three triples were the predecessor's
    # (gart_2_0_v3_features). Keys: {2d_by_size,nd_by_dim,tsplib_by_size}
    # _total_gart_2_0_{mape,sdpe}_pct. Printed at 2 dp to match the tables.
    (
        "abstract.headline_triples",
        r"and obtain aggregate MAPE/SDPE of 3.33\%/5.19\%, 0.88\%/1.28\%, and 3.27\%/3.42\%.",
        r"and obtain aggregate MAPE/SDPE of 2.90\%/4.69\%, 0.62\%/0.99\%, and 2.55\%/2.94\%.",
    ),
    # Factor over calibrated rho(d,n) on ND: 1.814353 / 0.620115 = 2.9258 -> 2.9.
    # TSPLIB margin over calibrated rho(d,n): 3.765556 - 2.554094 = 1.2115 -> 1.21 pp.
    (
        "abstract.margin_over_rho_dn",
        r"which GART 2.0 beats by a factor of 2.1 on the multidimensional benchmark and by 0.49 percentage points on TSPLIB.",
        r"which GART 2.0 beats by a factor of 2.9 on the multidimensional benchmark and by 1.21 percentage points on TSPLIB.",
    ),
    # The asymptotic-ratio margin is now significant and favourable:
    # paired_tsplib_by_size_total_asymptotic_mst_ratio_mean_diff = -0.996751,
    # ci [-1.77005, -0.223225], wilcoxon_p = 0.00485326.
    # Dispersion gain 1 - 2.944122/4.752154 = 38.047% -> 38%.
    # The 3.54% best-constant bound is left verbatim: no generator emits it.
    (
        "abstract.tsplib_asymptotic",
        r"On TSPLIB no constant multiplier improves on 3.54\% MAPE, and GART 2.0's margin over the asymptotic ratio is not statistically distinguishable from zero; its gain there is 28\% lower dispersion.",
        r"On TSPLIB no constant multiplier improves on 3.54\% MAPE; GART 2.0 beats the asymptotic ratio by 1.00 percentage points (paired Wilcoxon $p=0.0049$) and cuts dispersion by 38\%.",
    ),
    # Kwon-domain GART 2.0: 2.035693 -> 2.04 (table row, line 736, reads 2.04).
    # Then the two results the abstract did not carry: the 2D loss to the
    # retrained same-feature network, and the cost against the closed forms.
    (
        "abstract.kwon_plus_nn_and_cost",
        r"against GART 2.0's 2.21\%.",
        r"against GART 2.0's 2.04\%. A feed-forward network on the identical 31-feature vector beats GART 2.0 on the 2D benchmark, 2.34\%/3.54\% against 2.90\%/4.69\%, at all seven retraining seeds, so the feature set and not the model class carries that benchmark; GART 2.0 wins the multidimensional benchmark at all seven seeds and TSPLIB EUC\_2D at six. The accuracy is not free: GART 2.0 needs 6.12~ms per TSPLIB EUC\_2D instance against 2.57--3.08~ms for the closed-form baselines, and above $n=400$ it is the slowest of the eight estimators timed.",
    ),
    # Geometry spread was the predecessor's: isotropic 1.837291, LineNoise
    # 11.595016. GART 2.0 is 1.48381 and 10.752207.
    (
        "abstract.geometry_spread",
        r"error rises from 1.84\% on isotropic instances to 11.60\% on near-collinear ones.",
        r"error rises from 1.48\% on isotropic instances to 10.75\% on near-collinear ones.",
    ),
    # ------------------------------------------------------------------ #
    # INTRODUCTION, contribution paragraph (line 123)
    # ------------------------------------------------------------------ #
    # Drops the "any baseline that does not share its feature vector"
    # superlative, which now hides a loss at every seed rather than a tie.
    # Corrects 30 -> 31 features, the 2.1 factor -> 2.9, and the 0.28 pp
    # TSPLIB margin -> 1.00 pp. Adds the cost qualification.
    (
        "intro.claim_block",
        r"GART 2.0 has the lowest aggregate error of any baseline that does not share its feature vector, on all three benchmarks. The margin ranges from a factor of 2.1 down to 0.28 percentage points depending on the benchmark. A feed-forward network on the identical 30 features is more accurate than the boosted ensemble on the 2D benchmark, so the feature representation rather than the model class carries that result.",
        r"GART 2.0 has the lowest aggregate MAPE and SDPE of every baseline on the multidimensional and TSPLIB EUC\_2D benchmarks, and of every baseline except the same-feature neural network on the 2D benchmark. Its margin over the strongest baseline that uses no learned model is a factor of 2.9 on the multidimensional set and 1.00 percentage points on TSPLIB EUC\_2D. A feed-forward network on the identical 31 features beats the boosted ensemble on the 2D benchmark, 2.34\% MAPE and 3.54\% SDPE against 2.90\% and 4.69\%, and does so at all seven retraining seeds, so the feature representation rather than the model class carries that result. The accuracy also costs time: on TSPLIB EUC\_2D GART 2.0 runs 2.0 to 2.4 times slower than the closed-form estimators it is meant to displace, and above $n=400$ it is the slowest of the eight estimators timed.",
    ),
    # ------------------------------------------------------------------ #
    # CONCLUSION, paragraph 1 (line 484)
    # ------------------------------------------------------------------ #
    # Production feature count is 31 (sidecar n_features).
    (
        "conclusion.n_features",
        r"GART 2.0 predicts the MST-to-tour scaling factor from 30 MST, centroid, and coordinate-range features.",
        r"GART 2.0 predicts the MST-to-tour scaling factor from 31 MST, centroid, and coordinate-range features.",
    ),
    # GART 2.0 is lowest on BOTH metrics on ND and TSPLIB EUC_2D, not SDPE
    # alone; on 2D the same-feature network leads, so name it.
    (
        "conclusion.superlative",
        r"it has the lowest aggregate SDPE on the multidimensional and TSPLIB EUC\_2D benchmarks, and the lowest of any estimator not sharing its feature vector on the 2D benchmark.",
        r"it has the lowest aggregate MAPE and SDPE on the multidimensional and TSPLIB EUC\_2D benchmarks, and leads every baseline except the same-feature neural network on the 2D benchmark.",
    ),
    # 2.1 -> 2.9 (ND vs calibrated rho(d,n)); 2.4 -> 2.7 (Kwon domain,
    # 5.402075 / 2.035693 = 2.6537); 0.28 pp -> 1.00 pp with the CI and p
    # that now make it significant; 28% -> 38% dispersion.
    (
        "conclusion.margin_sizes",
        r"It is a factor of 2.1 on MAPE over a calibrated constant on the multidimensional set, a factor of 2.4 over Kwon--Golden--Wasil on the planar uniform instances where that regression was fitted, and 0.28 percentage points on TSPLIB EUC\_2D, where the paired difference against the asymptotic ratio is not statistically distinguishable from zero and the real gain is 28\% lower dispersion.",
        r"It is a factor of 2.9 on MAPE over a calibrated constant on the multidimensional set, a factor of 2.7 over Kwon--Golden--Wasil on the planar uniform instances where that regression was fitted, and 1.00 percentage points on TSPLIB EUC\_2D over the asymptotic ratio, a paired difference with 95\% CI $[0.22, 1.77]$ and $p=0.0049$, together with 38\% lower dispersion.",
    ),
    # The 2D loss stated with its numbers and its seed count, plus the two
    # other places GART 2.0 is behind: TSPLIB close-pair ordering
    # (rank_tsplib_euc2d_*_close5_pct: 69.090909 against 70.909091 for both
    # gart_1_0 and gart_2_0_v3_features) and cost.
    (
        "conclusion.nn_ordering_cost",
        r"On the 2D benchmark a feed-forward network on the same 30 features is more accurate than the boosted ensemble, so the feature set rather than the model class carries that result.",
        r"On the 2D benchmark a feed-forward network on the same 31 features beats the boosted ensemble on both metrics, 2.34\%/3.54\% against 2.90\%/4.69\%, at all seven retraining seeds, so the feature set rather than the model class carries that benchmark. Ordering is not uniform either: GART 2.0 orders 69.1\% of the close TSPLIB pairs correctly against 70.9\% for both GART 1.0 and the predecessor feature set, although it leads both on the wider 10\% band and on global rank correlation. It is also the more expensive estimator, 6.12~ms per TSPLIB EUC\_2D instance against 2.57--3.08~ms for the closed forms, and the slowest of the eight timed above $n=400$.",
    ),
    # ------------------------------------------------------------------ #
    # CONCLUSION, paragraph 2 (line 486) -- numeric defects only
    # ------------------------------------------------------------------ #
    # WITHDRAWN BY THE SINGLE WRITER, 2026-08-11.  Two hunks lived here:
    #
    #   conclusion.geometry_spread  11.59/1.84 -> 10.75/1.48
    #   conclusion.holdout_cost     0.35 -> 0.18 MAPE points
    #
    # Both are subsumed by patch_limits.py, which runs earlier in the applier
    # order and rewrites the whole limitations paragraph (P09) plus the
    # withholding sentence (P09B).  P09 makes the identical geometry
    # correction and additionally retires the clause "Those rounds do not
    # settle whether the cause is training coverage or the feature set",
    # which Section 4.8 now settles; P09B makes the identical 0.35 -> 0.18
    # correction.  Keeping these two here only produced two MISSING anchors
    # that aborted the whole patch.  The corrections are in the manuscript --
    # verified after patch_limits ran -- they are simply not made from here.
]


def main() -> int:
    raw = TEX.read_bytes()
    original_len = len(raw)

    applied, missing, ambiguous = [], [], []

    for tag, old, new in REPLACEMENTS:
        old_b = old.encode("utf-8")
        new_b = new.encode("utf-8")
        hits = raw.count(old_b)
        if hits == 0:
            missing.append(tag)
            continue
        if hits > 1:
            ambiguous.append((tag, hits))
            continue
        raw = raw.replace(old_b, new_b, 1)
        applied.append(tag)

    if missing or ambiguous:
        print("patch_headline: ABORTED, no bytes written")
    else:
        TEX.write_bytes(raw)

    print(f"patch_headline: {len(applied)} applied, {len(missing)} missing, "
          f"{len(ambiguous)} ambiguous, of {len(REPLACEMENTS)} total")
    for tag in applied:
        print(f"  applied   {tag}")
    for tag in missing:
        print(f"  MISSING   {tag}  (anchor not found verbatim)")
    for tag, hits in ambiguous:
        print(f"  AMBIGUOUS {tag}  (anchor matched {hits} times)")
    if applied and not (missing or ambiguous):
        print(f"  bytes {original_len} -> {len(raw)}")

    return 0 if not (missing or ambiguous) else 1


if __name__ == "__main__":
    raise SystemExit(main())
