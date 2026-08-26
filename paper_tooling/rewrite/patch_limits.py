"""Prose patch: GENERALIZATION dependants, COVERAGE/LIMITATIONS, TIMING.

Scope owned by this patch
-------------------------
  * subsec:coverage          (lines 402-412 at time of drafting)  -- full rewrite
  * the timing paragraph     (line 430)                            -- full rewrite
  * subsec:generalization    (line 400, closing sentence only)     -- one sentence
  * the Conclusion's limitations paragraph (line 486)              -- rewrite
  * subsec:provenance        (line 206)  -- CROSS-SCOPE, see note on P07

Every replacement is line-local: no old_string spans a line terminator, so the
file's CRLF endings are never matched and never rewritten.  Where a replacement
introduces a new paragraph the CRLF pair is written explicitly.

Sources for every number are recorded next to each replacement:
  BANK   paper_tooling/tables/paper_numbers.json
  TIME   paper_tooling/gart2_timing_bank.json  (current_solo_derived_numbers /
         tsplib_by_size_time_one_protocol -- never the _superseded blocks)
  GEN    paper_tooling/generalization_results.csv
  ARMA   paper_tooling/support_arms_results.json, support_arms_slope.csv,
         armA_verify_roworder_noise.csv, armA_verify_oracle_constant.csv,
         armA_verify_gain_decomposition.csv, armA_verify_grid_identity.py
  ARMC   paper_tooling/armC_gate_table.csv, decontaminated_arm_protocol.md
  ABL    paper_tooling/feature_retrain_criteria.csv, adv_support.csv
  CORPUS tsp_features_v4.csv train split (recomputed 2026-08-11)
  TEX    a machine-verified table body in Area_Free_Main.tex

Run by the single applier.  Do not run concurrently with another .tex writer.
"""

from __future__ import annotations

from pathlib import Path

TEX = Path(__file__).resolve().parents[2] / "paper_reference" / "Area_Free_Main.tex"

CRLF = "\r\n"
PARA = CRLF + CRLF


# ---------------------------------------------------------------------------
# P01  subsec:generalization, closing sentence of line 400.
#      The section no longer "finds it is not" a coverage gap -- subsec:coverage
#      now settles the question and the answer is "both".
#      source: ABL (feature_retrain_criteria.csv rows M0/MF criterion 8;
#                    adv_support.csv slope_n200 for M0+aug / MF+aug)
# ---------------------------------------------------------------------------
P01_OLD = r"""Section~\ref{subsec:coverage} tests whether that cost is a training-coverage gap, and finds it is not."""

P01_NEW = r"""Section~\ref{subsec:coverage} decomposes that cost and finds it is a coverage gap and a feature gap at once."""


# ---------------------------------------------------------------------------
# P02  subsec:coverage, opening paragraph (line 404).
#      Replaced by two paragraphs: the corrected failure statement, and the
#      exact statement of the support gap.
#      11.59 -> 10.75   BANK 2d_by_genclass_linenoise_gart_2_0_mape_pct 10.7522
#                        (TEX tab:genclass GART 2.0 LineNoise row prints 10.75;
#                         11.59 is the "GART 2.0 (V3 features)" row)
#      1.84  -> 1.48    BANK 2d_by_genclass_isotropic_gart_2_0_mape_pct 1.48381
#      -11.57 -> -10.74 BANK 2d_by_genclass_linenoise_gart_2_0_mspe_pct -10.7377
#      slope 0.29 -> 0.36, corr 0.96 -> 0.95
#                        ARMA support_arms_slope.csv FROZEN slope 0.36211,
#                        r 0.95183, n=90 (armA_verify_slope.csv agrees)
#      alpha>1.5 at n>=200 : 0        CORPUS (also 0 at alpha>1.45)
#      max n where alpha>1.7 : 10     CORPUS (212 such rows)
#      train alpha p99 = 1.544        CORPUS
#      54% of 210 LineNoise above p99 CORPUS (0.5429)
# ---------------------------------------------------------------------------
P02_OLD = r"""Section~\ref{subsec:results_2d} reports 11.59\% MAPE on the near-collinear Line Noise class against 1.84\% on the isotropic class, and the error there is almost entirely one-sided: MSPE is $-11.57$, so the model systematically under-predicts. A natural explanation is missing training coverage. In the training split, $\alpha>1.45$ occurs essentially only at $n\le10$, so the region ``large $n$ and large $\alpha$'' is empty, and regressing predicted $\alpha$ on true $\alpha$ over Line Noise at $n\ge200$ gives a slope of 0.29 against a correlation of 0.96: the model ranks those instances almost perfectly and declines to move its $\alpha$ level."""

P02_NEW = (
    r"""Section~\ref{subsec:results_2d} reports 10.75\% MAPE on the near-collinear Line Noise class against 1.48\% on the isotropic class, and the error is almost entirely one-sided: MSPE is $-10.74$, so the model under-predicts. Two explanations compete. The training split may never show the model this regime, or the feature set may be unable to represent it. This section separates them, and both hold."""
    + PARA
    + r"""The support gap is exact rather than approximate. The training split contains zero instances with $\alpha>1.5$ at $n\ge200$, and every training instance with $\alpha>1.7$ has $n\le10$. Its $\alpha$ distribution has a 99th percentile of 1.544, and 54\% of the 210 benchmark Line Noise instances lie above that percentile. No tree can map large $n$ at high $\alpha$ onto a high $\alpha$ when the fit never contains the combination. The symptom matches: regressing predicted $\alpha$ on true $\alpha$ over the 90 Line Noise instances at $n\ge200$ gives a slope of 0.36 against a correlation of 0.95, so the model ranks those instances almost perfectly and declines to move its $\alpha$ level."""
)


# ---------------------------------------------------------------------------
# P03  subsec:coverage, second paragraph (line 406).
#      The old paragraph states criteria for two augmentation rounds whose
#      reported numbers are all predecessor-fitted and unverifiable against the
#      shipped model.  Replaced by the 2x2 ablation that settles the diagnosis.
#      0.294 baseline / 0.569 features  ABL feature_retrain_criteria.csv
#                                       M0 criterion 8 "0.2941 (from 0.294)",
#                                       MF criterion 8 "0.5690 (from 0.294)"
#      0.332 support / 0.798 both       ABL adv_support.csv slope_n200
#                                       M0+aug 0.33182, MF+aug 0.79755
#      0.362 released model             ARMA support_arms_slope.csv FROZEN
#      874 added instances              ABL feature_adversarial.py stage_support
#                                       docstring; ARMC protocol Sec.1 (24+850)
#      five added descriptors           35 - 30; the five are
#                                       mst_topology_straightness,
#                                       mst_topology_deg2_straight_mean,
#                                       degeneracy_pca_effective_rank,
#                                       local_id_evr1_median_k5,
#                                       local_id_pr_mean_k5
# ---------------------------------------------------------------------------
P03_OLD = r"""We tested that explanation directly by generating and solving new instances to fill the empty region, adding them to the training split only and refitting with the hyperparameters and seed held fixed. Success criteria were fixed before any augmented model was fitted: Line Noise MAPE below 5\%, its MSPE within $\pm3$, no more than 0.05 points of regression on the multidimensional benchmark, no more than 0.15 on TSPLIB EUC\_2D or on any other 2D class, and a slope of at least 0.70. The slope criterion is the substantive one, because a MAPE gain without it indicates a shifted prediction mass rather than a learned relationship."""

P03_NEW = r"""A $2\times2$ ablation separates the two causes (\texttt{paper\_tooling/feature\_adversarial.py}). Holding hyperparameters and seed fixed, we vary the feature set, extending the 30-feature baseline with five degeneracy, local-intrinsic-dimension and MST-topology descriptors, and we vary the training support, adding 874 newly generated and solved instances at large $n$ and high $\alpha$ to the training split only. Measured by the Line Noise slope at $n\ge200$, features alone move the baseline from 0.294 to 0.569, support alone moves it to 0.332, and the two together reach 0.798. Neither intervention approaches sufficiency alone and the two are not additive, so the deficit is joint. The ladder is anchored on the 30-feature model, whose slope on this slice is 0.294; the released 31-feature model reaches 0.362 on the same slice, so these figures compare the two interventions against each other rather than against the shipped estimator."""


# ---------------------------------------------------------------------------
# P04  subsec:coverage, third paragraph (line 408).
#      Old paragraph = the 578-instance and 275-instance rounds, all
#      predecessor-fitted.  Replaced by arm A and the row-order nuisance result.
#      frozen 2D MAPE 2.90 / arm A 2.05  ARMA support_arms_results.json
#                                        bench2d_overall_mape 2.90375 / 2.05333
#      LineNoise 10.75 -> 5.78           ARMA armA_verify_oracle_constant.csv
#                                        frozen_mape 10.7503, armA_mape 5.78116
#      grid MSPE +7.11 -> +1.54          ARMA support_arms_results.json G7_val
#      slope 0.362 -> 0.827              ARMA support_arms_slope.csv FROZEN / A
#      nine gates, clean sweep           ARMA support_arms_results.json
#                                        model A: n_pass 9, clean_sweep true
#      row-order band [0.637, 0.803],
#        mean 0.728, reported 0.827      ARMA armA_verify_roworder_noise.csv
#                                        linenoise_slope reps=8 min/max/mean,
#                                        armA_outside_range True
#      MAPE band [5.90, 6.36] vs 5.78    ARMA same file, b2_line_noise_mape
# ---------------------------------------------------------------------------
P04_OLD = r"""Two rounds failed the same five criteria. The first added 578 instances spanning collapsed geometry, with $\alpha$ up to 1.995. It cut Line Noise MAPE to 7.95\% but moved the slope only to 0.47 while the correlation fell to 0.76. Measuring the benchmark's own instances showed why the target was wrong: writing $\rho$ for the ratio of transverse thickness to inter-point spacing, Line Noise at $n\ge200$ has median $\rho$ between 25 and 46 and $\alpha$ only 1.26--1.37, whereas the added instances spanned $\rho\le1$. The failures are mildly elongated clouds, not collapsed lines. The second round therefore added 275 instances at $\rho\in[2,100]$, of which 250 sit at $n\ge200$ in the intended regime. Line Noise MAPE moved only to 7.76\%, the slope fell back to 0.43, and the correlation fell further to 0.67."""

P04_NEW = r"""Repairing the deficit was then attempted twice, against criteria fixed before either candidate was scored, and both attempts were rejected. The first candidate, arm A, adds all 874 instances to the training split and clears every one of its nine gates: 2D MAPE falls from 2.90\% to 2.05\%, Line Noise MAPE from 10.75\% to 5.78\%, the \texttt{grid} sub-generator's MSPE from $+7.11$ to $+1.54$, and the Line Noise slope rises from 0.362 to 0.827. We rejected it because that slope is not a property of the model. Eight permutations of the row order of the identical 874 training rows, at the same seed, give slopes spanning $[0.637,0.803]$ with mean 0.728, and the reported 0.827 lies above the entire band; Line Noise MAPE behaves the same way, with a band of $[5.90,6.36]$ against a reported 5.78. Row order carries no information, so a statistic this sensitive to it cannot be published as a point estimate."""


# ---------------------------------------------------------------------------
# P05  subsec:coverage, fourth paragraph (line 410).
#      Old paragraph = mechanism of the deleted rounds.  Replaced by the
#      same-generator contamination finding and arm C's result.
#      52.6% / 47.6% share of arm A's 2D gain
#                                   ARMA armA_verify_gain_decomposition.csv
#                                   share_of_total_gain_pct grid 52.6115,
#                                   line_noise 47.5604
#      generator identity, 12 of 42 (n,G) cells against 18 augment cells
#                                   ARMA armA_verify_grid_identity.py, re-run
#                                   2026-08-11: EXACT (n,G) collisions = 12
#      24 dropped / 850 retained / hexlattice retained / 7 seeds
#                                   ARMC decontaminated_arm_protocol.md Sec.1-2
#      ten of eleven gates          ARMC armC_gate_table.csv (gate 2 FAIL only)
#      slope median 0.735, 6/7      ARMC gate 6
#      grid MSPE +1.45              ARMC gate 7
#      ND 0.6201 -> 0.5903          ARMC gate 1; GEN E0 test_all 0.6201
#      swap Holm p 0.065 [0.043, 0.152], 1/7, frozen 0.0048
#                                   ARMC gate 2 median/band/per_seed/note
# ---------------------------------------------------------------------------
P05_OLD = r"""The mechanism is visible in the direction of the change. Under the augmented model the mean predicted $\alpha$ on those instances \emph{decreases}, from 1.199 to 1.190, while aggregate MAPE improves, and the improvement is confined to $n\le100$ while the $n\in[401,1000]$ bucket worsens by 2.34 points. The model is shifting a family-shaped mass of predictions, not tracking $\alpha$. A within-family control makes the contrast sharp: training on half the Line Noise instances and testing on the other half reaches a slope of 0.84 and 2.48\% MAPE. The relationship is learnable when the model can key on correlations specific to one generator, and it does not transfer to a different generator of the same geometry even at matched $\rho$."""

P05_NEW = (
    r"""A second defect is worse than fragility. Arm A's 2D gain is not spread across the benchmark: the \texttt{grid} family carries 52.6\% of it and Line Noise the remaining 47.6\%, and the augmentation's $d=2$ \texttt{lattice} generator is the benchmark's \texttt{grid} generator. Both place $\max(2,\lceil\sqrt{n}\,\rceil)$ sites per side at spacing $G/\max(2,\lceil\sqrt{n}\,\rceil)$ with cell centres at $(i+\tfrac{1}{2})$ times that spacing; both jitter uniformly over $\pm5\%$ of the spacing at the shared setting; the site sets are exactly equal at perfect-square $n$; and 12 of the benchmark's 42 $(n,G)$ \texttt{grid} cells reappear exactly among the augmentation's 18 $d=2$ \texttt{lattice} cells. Half of arm A's headline gain is measured on its own training generator, and any description of that comparison as a cross-family test is wrong."""
    + PARA
    + r"""Arm C removes the 24 contaminating rows, keeps the other 850, retains the triangular \texttt{hexlattice} rows because a triangular lattice is not among the benchmark's generators, and re-runs over seven seeds fixed in advance, reporting every statistic as a median over seeds with its full band. It clears ten of eleven gates: the Line Noise slope reaches a median of 0.735 with six of seven seeds at or above 0.70, \texttt{grid} MSPE falls to $+1.45$, and multidimensional MAPE improves from 0.6201\% to a median 0.5903\%. It fails gate 2. Under the distribution-free swap permutation, its dispersion advantage over the asymptotic ratio on TSPLIB reaches a Holm-adjusted $p$ of only 0.065 at the median, band $[0.043,0.152]$, significant in one seed of seven, where the released model reaches 0.0048. The augmentation buys a real and seed-stable gain on Line Noise and on the lattice failure at the price of the paper's one distribution-free dispersion claim, and we did not make that trade."""
)


# ---------------------------------------------------------------------------
# P06  subsec:coverage, closing paragraph (line 412).
#      Old paragraph = the transverse-shape targeting analysis of the deleted
#      rounds.  Replaced by the oracle-constant control, the two protocol
#      disclosures, and the verdict.
#      97.9% / 105.7%            ARMA armA_verify_oracle_constant.csv
#                                frac_of_armA_gain_by_constant 0.97913 / 1.05660
#      p = 0.70 / p = 0.37       ARMA same file, wilcoxon_p_armA_vs_recal
#                                0.69850 / 0.36750
#      rho targets profiled off the 210 benchmark Line Noise instances
#                                ARMC protocol Sec.4 obligation 3
#      0.70 threshold inherited; de-contamination rule found from an outcome
#                                ARMC protocol Sec.0(a), Sec.0(b)
# ---------------------------------------------------------------------------
P06_OLD = r"""We do not yet conclude from this that the feature set is at fault, because both rounds are explained by a targeting error rather than by a limit of the model. Matching $\rho$ turns out not to be sufficient: the benchmark's near-collinear instances are generated with noise fixed at 2\% of the grid and a line drawn across the box, so clipping produces a bounded, sub-Gaussian transverse band with kurtosis 2.10, whereas our added instances used isotropic Gaussian noise with kurtosis 3.22. At equal transverse RMS a Gaussian spans roughly 1.7 times the full width, which carries the point set past the one- to two-dimensional crossover, and that crossover is sharp near $\rho\approx8$ where the benchmark's instances sit. Measured against the benchmark at matched $\rho$, our second round therefore lands 0.14 to 0.45 below it in $\alpha$, and adding rows below the target locus moves predictions in the wrong direction. What these two rounds establish is narrower than a claim about the feature set: filling the empty region of the $(n,\alpha)$ plane does not by itself close the gap, and the transverse shape of the added instances matters as much as their elongation. The same augmentation did close the opposite-signed lattice failure, where the \texttt{grid} sub-generator's MSPE improved from $+8.48$ to $+2.94$ under training instances drawn from different lattice generators. Both augmented models were rejected and the released model is unchanged."""

P06_NEW = r"""Three disclosures bound what either arm would have been worth. A single multiplicative constant fitted per family on that same family, an oracle a deployed estimator does not have, recovers 97.9\% of arm A's Line Noise MAPE gain and 105.7\% of its \texttt{grid} gain, and arm A is not significantly better than the released model rescaled by that constant on either family ($p=0.70$ and $p=0.37$). Nearly all of the apparent gain is a family-level shift in level, which is why the slope and not the MAPE is the criterion that decides. Second, the augmentation's elongation targets were chosen by profiling the 210 benchmark Line Noise instances, so the added coverage was aimed at the evaluation set's locus; no instance and no label was seen, but the targeting is a weakness of the design and we report it. Third, the 0.70 slope threshold was carried over verbatim from an earlier round's criteria file rather than chosen blind, and arm C's de-contamination rule was found by reading a leave-one-family-out result before being re-registered and re-run from scratch. The degenerate-geometry failure is therefore diagnosed and not repaired: it is a coverage gap and a feature gap together, both candidate repairs were rejected, and the released model is unchanged."""


# ---------------------------------------------------------------------------
# P07  subsec:provenance, line 206.  CROSS-SCOPE -- flagged.
#      These two figures are outputs of generalization_results.csv, the artifact
#      this patch owns, and both are the predecessor's.  If the agent who owns
#      subsec:provenance has already replaced this sentence, this hunk will
#      report MISSING and nothing is lost.
#      0.877 -> 0.6201   GEN E0 test_all  0.6201
#      0.876 -> 0.6182   GEN E0 test_clean 0.6182 (shipped model, cleaned slice)
#      0.8757 -> 0.6182  same
#      0.8726 -> 0.6146  GEN E1 test_clean 0.6146 (refit on cleaned train)
#      the "every other estimator by less than 0.06 percentage points" clause is
#      left untouched: it is not re-derived by any artifact I can reach.
# ---------------------------------------------------------------------------
P07_OLD = r"""Removing all 184 instances from the multidimensional benchmark moves GART 2.0 from 0.877\% to 0.876\% MAPE and every other estimator by less than 0.06 percentage points, and refitting the model on the cleaned training split moves test MAPE from 0.8757\% to 0.8726\%."""

P07_NEW = r"""Removing all 184 instances from the multidimensional benchmark moves GART 2.0 from 0.6201\% to 0.6182\% MAPE and every other estimator by less than 0.06 percentage points, and refitting the model on the cleaned training split moves test MAPE from 0.6182\% to 0.6146\%."""


# ---------------------------------------------------------------------------
# P08  the timing paragraph, line 430.
#      82.6 / 16.3 / 1.2  ->  92.90 / 5.03 / 2.08
#                         TIME current_solo_derived_numbers
#                              .gart2_decomposition_euc2d_78
#      6.12 / 8.98 / 11.24 / 6.12 total EUC_2D medians, and the n>400 and
#      smallest-bucket figures, all from
#                         TIME tsplib_by_size_time_one_protocol.time_ms
#                              (GART_2.0 6.122, LGBM_V3 8.982, NN_V3 11.238,
#                               GART_1.0 6.119, MST_Only 3.083, Hilbert 2.642,
#                               Calibrated_MST_dn 2.621, Asymptotic_MST 2.574;
#                               n>400: 20.544 / 18.407 / 18.265 / 16.475 /
#                               9.136-9.545)
#      2.0-2.4x, 5.55x    TIME positioning_summary.classical_range_total
#                              [1.986, 2.378]; gart2_over_model_ratio
#                              n in [51,150] Hilbert 5.552
#      171 -> 121 ms      TEX tab:nd_by_size Total row (BANK
#                         nd_by_size_total_gart_2_0_time_ms 121.4773); 171 ms is
#                         the "GART 2.0 (V3 features)" row.  The direction of
#                         the old sentence reverses: 121 against 122 is a tie,
#                         not "slower".
#      272 -> 465 ms      TEX tab:nd_by_size [501,1000] row (BANK
#                         nd_by_size_501_1000_gart_2_0_time_ms 465.0184);
#                         272 ms is again the V3-features row.
#      3,799 ms reference tour: unchanged, TEX same table.
#      127 -> 239 ms      TIME current_solo_derived_numbers.d18512_ms
#                         GART_2.0 239.25 (LGBM_V3 155.79 is the old draw's
#                         lineage).
# ---------------------------------------------------------------------------
P08_OLD = r"""In the TSPLIB timing log, GART 2.0 spends 82.6\% of total time on feature extraction and MST construction and 16.3\% on LightGBM inference; the residual 1.2\% is model loading and array marshalling. Feature extraction, not tree traversal, is therefore the target for any speed work. On the multidimensional benchmark the median GART 2.0 prediction is slower than generating the reference tour itself, 171~ms against 122~ms, because most of those instances are small enough to solve outright. The speed argument holds only where a solve is expensive: at $n\in[501,1000]$ the reference tour takes 3{,}799~ms against GART's 272~ms, and on TSPLIB \texttt{d18512} GART predicts in 127~ms."""

P08_NEW = (
    r"""In the TSPLIB timing log, GART 2.0 spends 92.90\% of total time on feature extraction and MST construction and 5.03\% on LightGBM inference; the residual 2.08\% is model loading and array marshalling. Feature extraction, not tree traversal, is the target for any speed work, and it is also what the cost claim turns on. The Time column of Table~\ref{tab:tsplib_by_size} puts all eight models on one protocol: one estimator per process, a single thread, the median of 11 repeats. Against the learned baselines GART 2.0 is competitive, 6.12~ms over the full EUC\_2D set against 8.98~ms for the V3-feature variant and 11.24~ms for the same-feature network, and it ties GART 1.0 at 6.12~ms. Against the classical closed forms it is not: it costs 2.0 to 2.4 times the 2.57--3.08~ms those four take on the same set, and 5.55 times the Hilbert sort on the smallest bucket. At $n>400$ it is the slowest of all eight, 20.54~ms against 18.41 for GART 1.0, 18.27 for the network, 16.48 for the V3-feature variant, and 9.1--9.5 for the closed forms. The closed forms are the comparison this paper positions itself against, so the honest statement is comparable cost among learned estimators and two to five times the cost of the classical ones."""
    + PARA
    + r"""On the multidimensional benchmark the median GART 2.0 prediction and the median reference-tour generation cost the same, 121~ms against 122~ms, because most of those instances are small enough to solve outright. The speed argument holds only where a solve is expensive: at $n\in[501,1000]$ the reference tour takes 3{,}799~ms against GART 2.0's 465~ms, and on TSPLIB \texttt{d18512} GART 2.0 predicts in 239~ms."""
)


# ---------------------------------------------------------------------------
# P09  the Conclusion's limitations paragraph, line 486.
#      11.59 / 1.84 -> 10.75 / 1.48    BANK, as for P02
#      the "two rounds ... do not settle whether the cause is training coverage
#      or the feature set" clause is now false: subsec:coverage settles it.
#      slope 0.29 -> 0.43, corr 0.96 -> 0.67 were the deleted rounds' figures;
#      replaced by the ablation ladder (ABL).
#      0.35 -> 0.18 MAPE points        GEN largest withholding cost is E3,
#                                      test_n_gt200_le1000 0.4075 -> 0.5873
#                                      = 0.1798; the dimension arm is
#                                      0.5009 -> 0.5772 = 0.0763.
#      the 184-instance clause and the "less than 0.06 percentage points" bound
#      are left as they stand: not re-derived here.
# ---------------------------------------------------------------------------
P09_OLD = r"""Three limitations bound these results. Accuracy depends on geometry more than on scale: error is 11.59\% on the near-collinear class against 1.84\% on the isotropic class. Section~\ref{subsec:coverage} reports two rounds of targeted augmentation against this failure, 874 newly solved instances judged by criteria fixed in advance, both of which were rejected: the predicted-versus-true $\alpha$ slope on those instances rose only from 0.29 to 0.43 while its correlation fell from 0.96 to 0.67. Those rounds do not settle whether the cause is training coverage or the feature set, because both added instances that miss the benchmark's transverse-shape regime, and we report them as a bound on what naive augmentation achieves rather than as a diagnosis."""

P09_NEW = r"""Three limitations bound these results. Accuracy depends on geometry more than on scale: error is 10.75\% on the near-collinear class against 1.48\% on the isotropic class. Section~\ref{subsec:coverage} diagnoses that failure as a training-coverage gap and a feature gap together: a $2\times2$ ablation moves the predicted-versus-true $\alpha$ slope on the affected instances from 0.294 to 0.569 on features alone, to 0.332 on coverage alone, and to 0.798 on both. Two repairs were attempted against criteria fixed in advance, both drawing on the same 874 newly solved instances, and both were rejected: the first because its headline slope was the favourable extreme of a row-order nuisance distribution and half of its 2D gain was measured on its own training generator, the second because it forfeits the model's distribution-free dispersion advantage on TSPLIB. The failure is diagnosed and not fixed."""


P09B_OLD = r"""Withholding a dimension or the large-$n$ range from training costs up to 0.35 MAPE points and appears as a systematic offset rather than as scatter, so the estimator degrades predictably off its training grid."""

P09B_NEW = r"""Withholding a dimension or the large-$n$ range from training costs up to 0.18 MAPE points and appears as a systematic offset rather than as scatter, so the estimator degrades predictably off its training grid."""


REPLACEMENTS = [
    ("P01 generalization: forward reference to subsec:coverage", P01_OLD, P01_NEW),
    ("P02 coverage: failure statement + exact support gap", P02_OLD, P02_NEW),
    ("P03 coverage: 2x2 feature/support ablation", P03_OLD, P03_NEW),
    ("P04 coverage: arm A and the row-order nuisance", P04_OLD, P04_NEW),
    ("P05 coverage: generator contamination and arm C", P05_OLD, P05_NEW),
    ("P06 coverage: oracle constant, disclosures, verdict", P06_OLD, P06_NEW),
    ("P07 provenance (CROSS-SCOPE): E0/E1 refit figures", P07_OLD, P07_NEW),
    ("P08 timing paragraph", P08_OLD, P08_NEW),
    ("P09 conclusion: limitations, coverage half", P09_OLD, P09_NEW),
    ("P09B conclusion: withholding cost 0.35 -> 0.18", P09B_OLD, P09B_NEW),
]


def main() -> int:
    raw = TEX.read_bytes()
    text = raw.decode("utf-8")

    applied, missing, ambiguous = [], [], []
    for name, old, new in REPLACEMENTS:
        count = text.count(old)
        if count == 0:
            missing.append(name)
            continue
        if count > 1:
            ambiguous.append((name, count))
            continue
        text = text.replace(old, new, 1)
        applied.append(name)

    if applied:
        TEX.write_bytes(text.encode("utf-8"))

    print(f"patch_limits.py -> {TEX}")
    print(f"  applied   {len(applied)}/{len(REPLACEMENTS)}")
    for n in applied:
        print(f"    ok      {n}")
    print(f"  missing   {len(missing)}")
    for n in missing:
        print(f"    MISSING {n}")
    print(f"  ambiguous {len(ambiguous)}")
    for n, c in ambiguous:
        print(f"    AMBIG   {n}  ({c} occurrences; not applied)")
    return 1 if (missing or ambiguous) else 0


if __name__ == "__main__":
    raise SystemExit(main())
