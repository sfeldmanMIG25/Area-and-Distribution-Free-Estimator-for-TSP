"""Prose patch: Results (ND, 2D, TSPLIB, matched domain, rank/close-pair, paired tests).

Scope: paper_reference/Area_Free_Main.tex, prose only.  No table body is touched.
Every replacement is an exact byte-for-byte ``old`` quoted from the manuscript as
it stood at the time of drafting; the applier reports any that no longer match
rather than guessing.

SOURCES (one per changed figure; see the accompanying report for the full map)
  bank      -- paper_tooling/tables/paper_numbers.json
  controls  -- paper_tooling/controls_31f/{marginals,paired,seed_band}.csv
  coverage  -- paper_tooling/tables/coverage.csv
  rank      -- paper_tooling/tables/table_rank.csv
  posthoc   -- recomputed from tsplib_benchmark/results/all_models_tsplib.csv under
               build_paper_tables.OK_STATUS + exclusions.filter_metric_consistent
  sidecar   -- nn_est_alpha_v3/nn_31f_model.json, nn_est_alpha_v3/train_nn_31f.py

The file is CRLF.  No ``old`` string spans a line break; the one insertion that
adds paragraphs writes "\r\n\r\n" explicitly.
"""

from __future__ import annotations

from pathlib import Path

TEX = Path(__file__).resolve().parents[2] / "paper_reference" / "Area_Free_Main.tex"

NL = "\r\n"

# --------------------------------------------------------------------------
# New prose blocks that are long enough to be worth naming.
# --------------------------------------------------------------------------

# Inserted after the 2D headline paragraph (was line 362).
CONTROL_PARAS = (
    "A feed-forward network on GART 2.0's own 31 features is more accurate "
    "than the boosted ensemble on this benchmark, and the gap is large: "
    "2.34\\% MAPE and 3.54\\% SDPE against GART 2.0's 2.90\\% and 4.69\\%. Its "
    "hyperparameters are V3's, frozen and not retuned, the same provenance "
    "GART 2.0's own hyperparameters carry, so the two arms differ in model "
    "class and in nothing else. The paired difference in absolute percent "
    "error is 0.56 percentage points in the network's favour (Wilcoxon "
    "$p=1.7\\times10^{-33}$), and the paired difference in SDPE is 1.15 points, "
    "with a 95\\% bootstrap interval of $[0.98,1.32]$ and no sign reversal in "
    "1{,}000 resamples. The advantage does not carry to the other benchmarks: "
    "on the multidimensional test split GART 2.0 is ahead of the same network "
    "by 0.46 MAPE points and 1.03 SDPE points, and on TSPLIB EUC\\_2D by 0.33 "
    "and 0.34 points. The model class therefore matters on all three "
    "benchmarks, and it favours the network on one of them."
    + NL + NL +
    "Neither verdict is a point estimate, because the network's fit is "
    "seed-dependent. The configuration was fitted at seven torch seeds and the "
    "shipped artifact is the seed with the best validation SDPE, selected "
    "without reference to any test split. Across those seven seeds the network "
    "beats GART 2.0 on the 2D benchmark on both metrics at every seed, loses "
    "on the multidimensional split at every seed, and loses on TSPLIB EUC\\_2D "
    "at six of seven. The spread is wide: the network's multidimensional SDPE "
    "ranges over $[1.63,2.30]$ across seeds, a band of 0.67 points and nearly "
    "nine times the 0.08-point SDPE margin that the earlier version of this "
    "comparison turned on. Every network verdict in this paper is a count of "
    "seeds rather than a difference between two numbers."
    + NL + NL +
    "That earlier comparison rested on a control that could not settle it. The "
    "network reported as ``Neural net (same features)'' in "
    "Table~\\ref{tab:2d_by_size} is fitted on the V3 30-feature vector and the "
    "superseded label table rather than on GART 2.0's 31 features and the "
    "production table, and it computes its own feature block instead of "
    "calling the production extractor. It exposes no \\texttt{predict\\_alpha}, "
    "so the non-Euclidean hybrid builder dispatched to a method a "
    "\\texttt{torch.nn.Module} does not have and recorded "
    "\\texttt{AttributeError} on 29 of the 111 TSPLIB instances, leaving it 4 "
    "of the 23 screened non-Euclidean instances against GART 2.0's 22. On the "
    "2D benchmark it reaches 2.99\\%/4.61\\%, which is 0.09 MAPE points behind "
    "GART 2.0 and 0.08 SDPE points ahead, at Wilcoxon $p=0.44$ and bootstrap "
    "$p=0.25$ respectively. The comparison drawn from it was wrong in "
    "direction on MAPE and resolved nothing on either metric; retraining the "
    "control on the production feature vector reverses the direction and "
    "enlarges the difference."
)

# --------------------------------------------------------------------------
# (old, new, why) -- applied in order.
# --------------------------------------------------------------------------
PAIRS: list[tuple[str, str, str]] = [

    # ================= ND results, headline (line 343) =====================
    (
        "GART 2.0 obtains 0.88\\% MAPE and 1.28\\% SDPE overall. The strongest "
        "baseline that uses no learned model is the calibrated ratio "
        "$\\hat\\rho(d,n)$ at 1.81\\%/2.94\\%, so the learned $\\alpha$ buys a "
        "factor of 2.1 on MAPE and 2.3 on SDPE over",

        "GART 2.0 obtains 0.62\\% MAPE and 0.99\\% SDPE overall. The strongest "
        "baseline that uses no learned model is the calibrated ratio "
        "$\\hat\\rho(d,n)$ at 1.81\\%/2.94\\%, so the learned $\\alpha$ buys a "
        "factor of 2.9 on MAPE and 3.0 on SDPE over",

        "ND totals were the predecessor's. bank nd_by_dim_total_gart_2_0_"
        "{mape,sdpe}_pct = 0.620115 / 0.988113; ratios over "
        "calibrated_mst_ratio_hat_rho_d_n = 1.814353/0.620115 = 2.926 and "
        "2.943222/0.988113 = 2.979.",
    ),

    # ================= ND consistency paragraph (line 345) =================
    (
        "SDPE decreases from 1.98\\% at $n\\le10$ to 0.55\\% at $n\\in[501,1000]$, "
        "and across dimension groups from 2.97\\% at $d=2$ to 0.61\\% at "
        "$d\\in[30,50]$, then rises to 0.65\\% at the unseen $d=100$ while MAPE "
        "rises from 0.38\\% to 1.04\\%.",

        "SDPE decreases from 1.56\\% at $n\\le10$ to 0.46\\% at $n\\in[201,500]$ "
        "and then rises by 0.01 points to 0.47\\% at $n\\in[501,1000]$, the one "
        "non-monotone step across size. Across dimension groups it falls from "
        "2.37\\% at $d=2$ to 0.49\\% at $d\\in[30,50]$, and falls again, to "
        "0.45\\%, at the unseen $d=100$, while MAPE rises there from 0.30\\% to "
        "0.60\\%. The predecessor's SDPE rose at $d=100$, from 0.61\\% to "
        "0.65\\%; the shipped model's falls.",

        "Central consistency claim. bank nd_by_size_*_gart_2_0_sdpe_pct = "
        "1.5599 / 1.0339 / 0.6910 / 0.4882 / 0.4603 / 0.4739 (the last step "
        "rises by 0.0136, stated rather than rounded away); "
        "nd_by_dim_{d2,d30_50,d100}_gart_2_0_sdpe_pct = 2.3665 / 0.4904 / "
        "0.4507; MAPE 0.2962 -> 0.5966; predecessor "
        "nd_by_dim_{d30_50,d100}_gart_2_0_v3_features_sdpe_pct = 0.6104 -> "
        "0.6522.",
    ),

    # ================= 2D section opener (line 351) ========================
    (
        "GART 2.0 has SDPE below 6.3\\% in every size bucket and 3.05\\% in the "
        "$[501,1000]$ bucket. Its aggregate SDPE of 5.19\\% is less than half "
        "the 11.57\\%",

        "GART 2.0 has SDPE below 5.9\\% in every size bucket and 2.74\\% in the "
        "$[501,1000]$ bucket. Its aggregate SDPE of 4.69\\% is less than half "
        "the 11.57\\%",

        "bank 2d_by_size_*_gart_2_0_sdpe_pct buckets = 4.2818 / 5.8342 / "
        "5.5848 / 4.2450 / 2.7363, so the tight bound is 5.9; total 4.6869. "
        "'less than half 11.57' still holds (2 x 4.69 = 9.37).",
    ),

    # ================= 2D headline (line 362) ==============================
    (
        "GART 2.0 obtains 3.33\\% MAPE and 5.19\\% SDPE overall, against",
        "GART 2.0 obtains 2.90\\% MAPE and 4.69\\% SDPE overall, against",
        "bank 2d_by_size_total_gart_2_0_{mape,sdpe}_pct = 2.904163 / 4.686905.",
    ),

    # ---- delete the false NN sentence, append the control discussion ------
    (
        "The same-feature neural network reaches 2.99\\%/4.61\\% and is more "
        "accurate than the boosted ensemble on this benchmark; "
        "Appendix~\\ref{app:paired} quantifies that difference. The custom "
        "Hilbert sort has 31.01\\% MAPE against 14.24\\% SDPE, a gap that "
        "identifies its error as a near-constant positive offset rather than "
        "scatter, which is what a space-filling-curve tour should produce.",

        "The custom Hilbert sort has 31.01\\% MAPE against 14.24\\% SDPE, a gap "
        "that identifies its error as a near-constant positive offset rather "
        "than scatter, which is what a space-filling-curve tour should "
        "produce." + NL + NL + CONTROL_PARAS,

        "STRUCTURAL (a) and (b). The deleted sentence was true of neither "
        "control: against NN_V3 GART 2.0 is AHEAD on MAPE (2.904 vs 2.990, "
        "paired mean_diff -0.0863, Wilcoxon p = 0.4396, SDPE diff +0.0771, "
        "boot p = 0.252); against the retrained NN_31F the network wins both "
        "metrics (2.3436 / 3.5377) at 7/7 seeds, d|APE| +0.5606 Wilcoxon "
        "p = 1.73e-33, dSDPE +1.1492 CI [0.9778, 1.3225] boot p = 0.000 over "
        "1,000 resamples. ND and TSPLIB EUC paired diffs -0.4557 / -1.0273 "
        "and -0.3307 / -0.3363 favour GART 2.0 (7/7 and 6/7 seeds). Seed band "
        "on ND SDPE [1.6264, 2.2986] = 0.672 points against the old 0.0771 "
        "margin. Control defects from coverage.csv "
        "(TSPLIB,NN_V3,exception:AttributeError,29) and "
        "nn_est_alpha_v3/estimator_nn_31f.py; non-Euclidean coverage 4 vs 22 "
        "from controls_31f/marginals.csv.",
    ),

    # ================= 2D generator classes (line 364) =====================
    (
        "GART 2.0 ranges from 1.84\\% MAPE on the isotropic class to 11.60\\% "
        "on Line Noise",
        "GART 2.0 ranges from 1.48\\% MAPE on the isotropic class to 10.75\\% "
        "on Line Noise",
        "bank 2d_by_genclass_{isotropic,linenoise}_gart_2_0_mape_pct = "
        "1.483810 / 10.752207. The four baseline figures in the same sentence "
        "were verified unchanged and are left alone.",
    ),

    # ================= TSPLIB EUC_2D headline (line 380) ===================
    (
        "On the full EUC\\_2D benchmark GART 2.0 obtains 3.42\\% SDPE and "
        "3.27\\% MAPE, against",
        "On the full EUC\\_2D benchmark GART 2.0 obtains 2.94\\% SDPE and "
        "2.55\\% MAPE, against",
        "bank tsplib_by_size_total_gart_2_0_{sdpe,mape}_pct = 2.944122 / "
        "2.554094. The three baseline pairs in the same sentence were verified "
        "unchanged.",
    ),
    (
        "and GART 2.0's 3.27\\% improves on that bound by 0.27 percentage "
        "points.",
        "and GART 2.0's 2.55\\% improves on that bound by 0.99 percentage "
        "points.",
        "3.545 (MAPE-minimising constant, unchanged) - 2.554094 = 0.991.",
    ),
    (
        "The dispersion gap is the larger one, 3.42\\% SDPE against 4.75\\%,",
        "The dispersion gap is the larger one, 2.94\\% SDPE against 4.75\\%,",
        "bank tsplib_by_size_total_gart_2_0_sdpe_pct = 2.944122 against "
        "asymptotic_mst_ratio 4.752154; the dispersion gap (1.81 pp) is still "
        "larger than the MAPE gap (0.99 pp), so the sentence's claim survives.",
    ),

    # ================= TSPLIB size non-uniformity (line 382) ===============
    (
        "In the coarse $n>400$ bucket the GART and asymptotic-ratio SDPE "
        "intervals overlap, and in the post hoc $n>10{,}000$ slice the two are "
        "nearly tied on different metrics, with the asymptotic ratio lower on "
        "SDPE (1.46\\% versus 1.67\\%) and GART marginally lower on MAPE "
        "(2.71\\% versus 2.74\\%).",

        "In the coarse $n>400$ bucket the GART and asymptotic-ratio SDPE "
        "intervals overlap, on $[3.20,3.62]$. In the post hoc $n>10{,}000$ "
        "slice of five instances GART 2.0 is lower on both metrics, 0.57\\% "
        "MAPE and 0.72\\% SDPE against the asymptotic ratio's 2.74\\% and "
        "1.46\\%.",

        "1.67/2.71 were the predecessor's. Recomputed on the five EUC_2D "
        "instances with n > 10,000 (brd14051, d15112, d18512, rl11849, "
        "usa13509) under the same screens: GART 2.0 MAPE 0.5750 SDPE 0.7186; "
        "Asymptotic_MST 2.7401 / 1.4556; LGBM_V3 2.7056 / 1.6734. Interval "
        "overlap from bank tsplib_by_size_gt400_*_sdpe_{lo,hi} = GART "
        "[2.3163, 3.6153], asymptotic [3.2013, 4.6976].",
    ),

    # ================= Matched-domain comparison (line 392) ================
    (
        "On the 80 Kwon-domain instances it obtains 2.21\\% MAPE against "
        "Kwon's 5.40\\%; on the 50 Chien-domain instances 3.40\\% against "
        "Chien's 18.54\\%; and on all 210 uniform instances 1.58\\% against "
        "BHH's 8.89\\% and the $\\alpha=1$ floor's 15.58\\%. The same-feature "
        "neural network again edges the boosted ensemble, 1.52\\% against "
        "1.58\\%, though on these 210 instances that difference is not "
        "significant ($p=0.73$). A factor of 2.4 over Kwon--Golden--Wasil on "
        "its home ground",

        "On the 80 Kwon-domain instances it obtains 2.04\\% MAPE against "
        "Kwon's 5.40\\%; on the 50 Chien-domain instances 2.46\\% against "
        "Chien's 18.54\\%; and on all 210 uniform instances 1.32\\% against "
        "BHH's 8.89\\% and the $\\alpha=1$ floor's 15.58\\%. The V3-feature "
        "network is behind it here, 1.52\\% against 1.32\\%, though on these "
        "210 instances that difference is not significant ($p=0.069$). We have "
        "not scored the 31-feature network on this subset, so the reversal of "
        "Section~\\ref{subsec:results_2d} is untested on the matched domain. A "
        "factor of 2.7 over Kwon--Golden--Wasil on its home ground",

        "bank classical_b_{kwon,chien,random}_gart_2_0_mape_pct = 2.035693 / "
        "2.456358 / 1.317039; classical_b_random_neural_net_same_features_"
        "mape_pct = 1.517095, so GART 2.0 leads, not the network; "
        "paired_classical_b_random_neural_net_same_features_wilcoxon_p = "
        "0.06929 (0.737 is now the Kwon-domain comparison, n = 80). Factor "
        "5.402075 / 2.035693 = 2.654.",
    ),

    # ================= Close-pair ordering (line 420) ======================
    (
        "These percentages remain descriptive because the pairs share "
        "instances and we compute no block-bootstrap interval over them. The "
        "same-feature network orders TSPLIB close pairs better than GART 2.0, "
        "74.5\\% against 69.1\\%.",

        "Every pair set is enumerated exhaustively rather than sampled "
        "(Appendix~\\ref{app:rank}). These percentages remain descriptive "
        "because the pairs share instances and we compute no block-bootstrap "
        "interval over them. GART 2.0 orders TSPLIB close pairs worse than "
        "three models it is compared against: the same-feature network at "
        "74.5\\%, GART 1.0 at 70.9\\%, and the V3-feature predecessor, whose "
        "rank row is in the released tidy tables rather than in "
        "Table~\\ref{tab:rank}, also at 70.9\\%, against GART 2.0's 69.1\\%. "
        "Cost accuracy and pair ordering are distinct properties, and 55 pairs "
        "resolve none of these gaps.",

        "table_rank.csv, TSPLIB EUC_2D close5_pct: GART_2.0 69.0909, NN_V3 "
        "74.5455, GART_1.0 70.9091, LGBM_V3 70.9091. Exhaustive enumeration is "
        "already stated in Appendix app:rank; the body now says so too. The "
        "counts and percentages already in this paragraph were verified "
        "against table_rank.csv and left unchanged.",
    ),

    # ================= Asymptotic-regime caveat (line 422) =================
    (
        "The mixed post hoc result at extreme sizes should therefore be "
        "treated as an empirical boundary of the present model, not as proof "
        "of a particular convergence mechanism.",

        "GART 2.0's post hoc advantage at extreme sizes should therefore be "
        "treated as an observation on five instances, not as proof of a "
        "particular convergence mechanism.",

        "Consequential edit: the n > 10,000 result is no longer mixed (GART "
        "2.0 leads on both metrics), so 'mixed' contradicts the corrected "
        "line-382 sentence.",
    ),

    # ================= R^2_alpha on the n>400 bucket (line 424) ============
    (
        "The $n>400$ TSPLIB bucket has $R^2_\\alpha=-0.549$,",
        "The $n>400$ TSPLIB bucket has $R^2_\\alpha=-0.071$,",
        "bank tsplib_by_size_gt400_gart_2_0_r2_alpha = -0.071034 (-0.549 is "
        "LGBM_V3's). Still negative, so the paragraph's argument stands.",
    ),
    (
        "while tour-cost MAPE remains 3.75\\%.",
        "while tour-cost MAPE remains 2.90\\%.",
        "bank tsplib_by_size_gt400_gart_2_0_mape_pct = 2.902071.",
    ),

    # ================= Paired significance paragraph (line 426) ============
    (
        "The aggregate margins are not uniformly significant, and one of them "
        "is not significant at all.",
        "The aggregate margins are not uniformly significant, and one of the "
        "significant ones runs against GART 2.0.",
        "The TSPLIB margin is now significant (p = 4.85e-3) and the 2D margin "
        "is significant against GART 2.0, so the old topic sentence is false.",
    ),
    (
        "On TSPLIB EUC\\_2D the difference between GART 2.0 and the asymptotic "
        "MST ratio is $-0.28$ percentage points with a 95\\% interval of "
        "$[-1.05,+0.45]$ and $p=0.77$: on that benchmark GART 2.0's MAPE "
        "advantage is indistinguishable from zero, and its real gain is "
        "dispersion, 3.42\\% SDPE against 4.75\\%. On the 2D benchmark the "
        "feed-forward network on the identical feature vector is more accurate "
        "than the boosted ensemble by 0.34 percentage points, with a 95\\% "
        "interval of $[0.27,0.43]$. Gradient boosting is therefore not the "
        "source of the advantage on that benchmark; the feature set is.",

        "On TSPLIB EUC\\_2D the difference between GART 2.0 and the asymptotic "
        "MST ratio is $-1.00$ percentage points with a 95\\% interval of "
        "$[-1.77,-0.22]$ and $p=0.0049$: on that benchmark GART 2.0's MAPE "
        "advantage is small but resolvable, and its larger gain is dispersion, "
        "2.94\\% SDPE against 4.75\\%. On the 2D benchmark the paired "
        "difference against the feed-forward network on GART 2.0's own 31 "
        "features is $+0.56$ percentage points with a 95\\% interval of "
        "$[+0.48,+0.66]$, against GART 2.0. Gradient boosting is therefore not "
        "the source of the advantage on that benchmark; the feature set is, "
        "and a different learner exploits the same features better. That "
        "reversal is confined to 2D: on the multidimensional split and on "
        "TSPLIB EUC\\_2D the boosted ensemble wins at seven of seven and six "
        "of seven training seeds (Section~\\ref{subsec:results_2d}).",

        "STRUCTURAL (a). bank paired_tsplib_by_size_total_asymptotic_mst_ratio"
        "_{mean_diff,ci_lo,ci_hi,wilcoxon_p} = -0.996751, -1.770050, "
        "-0.223230, 0.00485326 -- the interval no longer straddles zero, so "
        "'indistinguishable from zero' inverts. The 2D clause was pointed at "
        "NN_V3, whose true paired difference (-0.0863, p = 0.4396) favours "
        "GART 2.0; it is repointed at NN_31F, controls_31f/paired.csv row "
        "2d/GART_2.0/NN_31F: mean_diff +0.5606, CI [0.4756, 0.6571]. Seed "
        "counts from controls_31f/seed_band.csv.",
    ),
]


def main() -> int:
    raw = TEX.read_bytes()
    text = raw.decode("utf-8")
    applied, missing = 0, []
    for old, new, why in PAIRS:
        if text.count(old) == 1:
            text = text.replace(old, new, 1)
            applied += 1
        else:
            missing.append((text.count(old), old[:90], why[:70]))
    TEX.write_bytes(text.encode("utf-8"))
    print(f"patch_results.py: applied {applied}/{len(PAIRS)}")
    for count, head, why in missing:
        print(f"  MISSING (count={count}): {head!r}\n    why: {why}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
