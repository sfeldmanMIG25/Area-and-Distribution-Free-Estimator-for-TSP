"""Claim -> authority table for every prose number in ``Area_Free_Main.tex``.

DATA ONLY. No formatting, no I/O, no pandas. ``build_prose_claim_map.py``
renders this into ``prose_claim_map.json`` (for the checker) and
``prose_claim_map.md`` (for humans). A change to either output format touches
the renderer, never this file.

Why this file exists
--------------------
``build_paper_tables.py --check`` verifies the table cells and nothing else.
Every number in the running prose is transcribed by hand, so when the
production model changed the tables followed and the prose did not.
``paper_tooling/prose_claim_audit.md`` catalogued the damage. This file is the
durable half of that audit: the *values* in the audit expire at the next model
swap, the *mapping* from a claim to the artifact that should generate it does
not.

Bucket semantics
----------------
``bucket`` answers one question: where does the authority for this number live?

    BANKED       a key exists in paper_tooling/tables/paper_numbers.json.
                 ``keys`` lists them. ``derivation`` is a Python expression over
                 those keys when the prose quotes a ratio, a margin, an argmin
                 or a bound rather than a raw cell.
    GENERATED    some script computes it and writes it to disk, but not into the
                 number bank. ``authority`` names the artifact, ``action`` names
                 the export that would make it BANKED.
    UNGENERATED  nothing computes it for the production model. ``action`` is the
                 concrete thing that would have to be written or run.
    STRUCTURAL   not a result: a dataset size, a design constant, a definition,
                 a prespecified threshold, a taxonomy. Should not move when the
                 model moves. ``note`` records why.

``drift=True`` marks a claim the manuscript *frames* as invariant whose value
nevertheless moved with the model. On a STRUCTURAL row that combination is the
defect itself: the manuscript is asserting a constant that is not one.

``checkable=False`` marks a claim no artifact can ever settle: an asymptotic
argument, a causal interpretation, or a historical fact about the process.
Those are the honest limit of mechanical checking.

Anchors
-------
``anchor`` is a literal substring of the .tex, chosen to survive reflowing and
renumbering. Line numbers are a 2026-08-11 snapshot (1,371 lines) and are a
hint, not a key. ``build_prose_claim_map.py --verify`` re-locates every anchor.
"""

from __future__ import annotations

NUMBER_BANK = "paper_tooling/tables/paper_numbers.json"
PAIRED = "paper_tooling/tables/paired_tests.csv"
COVERAGE = "paper_tooling/tables/coverage.csv"
SIDECAR = "lgbm_model_v3/gart2_final.json"
CALIB = "calibrated_alpha_table.json"
TOURS = "paper_tooling/reference_tour_audit.csv"
MDS = "paper_reference/mds_distortion_screened.csv"
CORPUS = "tsp_features_v4.csv"

FIELDS = ("id", "section", "line", "anchor", "quantity", "audit", "bucket",
          "authority", "keys", "derivation", "drift", "checkable", "note", "action")

# Claims the manuscript has ALREADY been corrected on since the audit snapshot,
# verified against the .tex and the number bank while building this map. Their
# ``audit`` verdict below is the audit's, and is now historical. Recorded rather
# than deleted because the mapping is the point: these rows show the machinery
# working, and they are the rows a checker should already pass.
#
# Section 4.8's close-pair paragraph (line 420) and the rank appendix (line 830)
# were rewritten to the exhaustive-enumeration definition; every figure in both
# now agrees with paper_numbers.json to the printed precision.
RECONCILED_IN_TEX = frozenset({
    "disc.pairs_2d", "disc.close5_2d", "disc.pairs_nd", "disc.close5_nd",
    "disc.close5_tsplib", "disc.close5_control", "disc.gain_nd", "disc.gain_2d",
    "disc.gain_tsplib", "disc.close10", "disc.nn_close_pairs",
    "appx.nd_pair_sample",
})


def C(id, section, line, anchor, quantity, audit, bucket, authority=None,
      keys=(), derivation=None, drift=False, checkable=True, note="", action=None):
    return dict(id=id, section=section, line=line, anchor=anchor, quantity=quantity,
                audit=audit, bucket=bucket, authority=authority, keys=list(keys),
                derivation=derivation, drift=drift, checkable=checkable,
                note=note, action=action)


# Shorthands for the two authority artifacts that recur.
def _bank(keys, derivation=None, **kw):
    return dict(authority=NUMBER_BANK, keys=list(keys), derivation=derivation, **kw)


CLAIMS = [

    # ---------------------------------------------------------------- Abstract
    C("abs.corpus_sizes", "Abstract", 90, "2{,}580 synthetic 2D instances",
      "benchmark sizes 2,580 / 16,920 / 78", "CORRECT", "STRUCTURAL",
      authority=NUMBER_BANK,
      keys=["2d_by_size_total_gart_2_0_n", "nd_by_size_total_gart_2_0_n",
            "tsplib_by_size_total_gart_2_0_n"],
      note="Dataset sizes are fixed by the corpus, not by the model. Banked only "
           "as a cross-check. HAZARD: the bank's N is per-model, so a future "
           "model that declines a EUC_2D instance would move the 78 without the "
           "corpus changing. Read the count off the coverage table, not off a "
           "model row."),
    C("abs.2d_headline", "Abstract", 90, "aggregate MAPE/SDPE of 3.33",
      "2D aggregate MAPE and SDPE", "STALE", "BANKED",
      **_bank(["2d_by_size_total_gart_2_0_mape_pct",
               "2d_by_size_total_gart_2_0_sdpe_pct"])),
    C("abs.nd_headline", "Abstract", 90, "0.88\\%/1.28",
      "ND aggregate MAPE and SDPE", "STALE", "BANKED",
      **_bank(["nd_by_size_total_gart_2_0_mape_pct",
               "nd_by_size_total_gart_2_0_sdpe_pct"])),
    C("abs.tsplib_headline", "Abstract", 90, "3.27\\%/3.42",
      "TSPLIB EUC_2D aggregate MAPE and SDPE", "STALE", "BANKED",
      **_bank(["tsplib_by_size_total_gart_2_0_mape_pct",
               "tsplib_by_size_total_gart_2_0_sdpe_pct"])),
    C("abs.identical_features", "Abstract", 90, "identical feature vector",
      "Linear and NN share GART's feature vector", "CONTRADICTED", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="Framed as a design invariant; it is not one. Sidecar n_features=31, "
           "Linear_V3/NN_V3 consume the 30-feature V3 vector. Five other "
           "occurrences say the same thing.",
      action="model_registry.py has no feature-set field. Add FEATURE_SETS: "
             "{model_key -> frozenset(features)} so 'shares its feature vector' "
             "becomes a computable predicate instead of a prose assertion."),
    C("abs.nd_factor", "Abstract", 90, "by a factor of 2.1 on the multidimensional",
      "GART advantage over rho(d,n) on ND", "STALE", "BANKED",
      **_bank(["nd_by_size_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
               "nd_by_size_total_gart_2_0_mape_pct"],
              derivation="k0 / k1")),
    C("abs.tsplib_margin", "Abstract", 90, "0.49 percentage points on TSPLIB",
      "GART margin over rho(d,n) on TSPLIB", "STALE", "BANKED",
      **_bank(["tsplib_by_size_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
               "tsplib_by_size_total_gart_2_0_mape_pct"],
              derivation="k0 - k1")),
    C("abs.oracle_constant_78", "Abstract", 90, "no constant multiplier improves on 3.54",
      "MAPE-minimising constant multiple of L_MST over the 78 EUC_2D", "CORRECT",
      "UNGENERATED",
      note="build_paper_tables.py evaluates Fixed_Alpha=1.136 but never searches "
           "for the optimum. The value was produced by an ad-hoc grid search that "
           "persisted nothing.",
      action="Add an oracle-constant search to build_paper_tables.py: over each "
             "instance set (78 EUC_2D, all 111, the 23 screened non-EUC_2D), grid "
             "c over true_cost/mst_length, bank c* and its MAPE as "
             "oracle_constant_<set>_{c,mape_pct}. Model-independent, so it also "
             "survives the arm A swap unchanged."),
    C("abs.tsplib_paired", "Abstract", 90, "not statistically distinguishable from zero",
      "paired GART vs asymptotic-ratio difference on TSPLIB", "CONTRADICTED",
      "GENERATED", authority=PAIRED,
      keys=["table=tsplib_by_size,bucket_slug=total,model_b=Asymptotic_MST"
            " -> mean_diff, ci_lo, ci_hi, wilcoxon_p"],
      action="paired_tests.csv is written by build_paper_tables.main() but never "
             "reaches the bank. Export paired_<table>_<bucket_slug>_<model_b_slug>_"
             "{n_pairs,mean_diff,ci_lo,ci_hi,wilcoxon_p} into paper_numbers.json. "
             "This one export covers every significance claim in the paper."),
    C("abs.dispersion_gain", "Abstract", 90, "lower dispersion",
      "GART dispersion gain over the asymptotic ratio on TSPLIB", "STALE", "BANKED",
      **_bank(["tsplib_by_size_total_gart_2_0_sdpe_pct",
               "tsplib_by_size_total_asymptotic_mst_ratio_sdpe_pct"],
              derivation="100 * (1 - k0 / k1)")),
    C("abs.kwon_home", "Abstract", 90, "reaches 5.40",
      "Kwon vs GART on the Kwon domain", "STALE", "BANKED",
      **_bank(["classical_b_kwon_kwon_golden_wasil_sampling_region_mape_pct",
               "classical_b_kwon_gart_2_0_mape_pct"])),
    C("abs.linenoise_span", "Abstract", 90, "isotropic instances to 11.60",
      "GART MAPE isotropic vs Line Noise", "STALE", "BANKED",
      **_bank(["2d_by_genclass_isotropic_gart_2_0_mape_pct",
               "2d_by_genclass_linenoise_gart_2_0_mape_pct"])),

    # ------------------------------------------------------------ Introduction
    C("intro.baseline_count", "Introduction", 123, "thirteen baselines",
      "size of the comparison roster", "STALE", "STRUCTURAL",
      authority="paper_tooling/build_paper_tables.py::TEX_MODELS + model_registry.MODEL_LABELS",
      drift=True,
      note="Framed as a fixed experimental design. It moved because the roster "
           "gained a 'GART 2.0 (V3 features)' row when the model changed, and the "
           "tidy tables carry a 15th (LGBM_V4). A roster count must never be "
           "hand-typed next to a roster defined in code.",
      action="Bank len(TEX_MODELS[roster]) - 1 per benchmark as "
             "roster_<benchmark>_n_baselines, and have splice_tables.py write the "
             "word form. Then the count cannot drift from the tables it describes."),
    C("intro.identical_features", "Introduction", 123, "a linear model and a neural network on the identical",
      "Linear and NN share GART's feature vector", "CONTRADICTED", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="Duplicate of abs.identical_features.", action="See abs.identical_features."),
    C("intro.superlative_all3", "Introduction", 123, "lowest aggregate error of any baseline",
      "GART is best on all three benchmarks among non-feature-sharing baselines",
      "CONTRADICTED", "BANKED",
      **_bank(["2d_by_size_total_*_{mape_pct,sdpe_pct}",
               "nd_by_size_total_*_{mape_pct,sdpe_pct}",
               "tsplib_by_size_total_*_{mape_pct,sdpe_pct}"],
              derivation="argmin over the shipped roster, excluding models that "
                         "share GART's feature vector"),
      note="Mechanically checkable ONLY once 'shares its feature vector' is a "
           "computable predicate; today the exclusion set is a prose judgement. "
           "Also roster-conditional: the claim's truth depends on whether "
           "LGBM_V4 ships.",
      action="Same FEATURE_SETS addition as abs.identical_features, plus a "
             "SHIPPED_ROSTER constant so the argmin is taken over the roster that "
             "actually reaches the .tex."),
    C("intro.margin_range", "Introduction", 123, "margin ranges from a factor of 2.1",
      "range of GART's margin across benchmarks", "STALE", "BANKED",
      **_bank(["nd_by_size_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
               "nd_by_size_total_gart_2_0_mape_pct",
               "tsplib_by_size_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
               "tsplib_by_size_total_gart_2_0_mape_pct"],
              derivation="max(k0/k1) ... min(k2-k3) over the three benchmarks")),
    C("intro.nn_beats_gart_2d", "Introduction", 123, "is more accurate than the boosted ensemble on the 2D",
      "NN_V3 beats GART on 2D", "CONTRADICTED", "BANKED",
      **_bank(["2d_by_size_total_gart_2_0_mape_pct",
               "2d_by_size_total_neural_net_same_features_mape_pct"],
              derivation="k0 > k1"),
      note="Ordering is banked. The significance attached to it in the "
           "Discussion is in paired_tests.csv (see disc.nn_paired)."),

    # --------------------------------------------------- Theory / Methodology
    C("meth.alpha_sd_by_d", "Theory", 161, "falls monotonically from 0.1837",
      "training-split std(alpha) at d=2 and d=50", "UNVERIFIABLE", "UNGENERATED",
      action="Groupby on the corpus: alpha = optimal_cost / mst_total_length "
             "(clip to [1,2] exactly as lgbm_model_v3/freeze_gart2_final.py does), "
             "restricted to split=='train', std by dimension. Bank as "
             "corpus_train_alpha_sd_d<d>.",
      note="VERIFIED the audit's suggestion, with a correction. The audit says "
           "'a tsp_features_v4.csv groupby' and that is the right table -- "
           "gart2_final.json records training_table=tsp_features_v4.csv. But this "
           "will not merely confirm 0.1837/0.0762: the manuscript's values came "
           "off tsp_features_v3.csv, which has 35 columns to v4's 48. Both carry "
           "106,272 rows, so the split is probably identical and the values "
           "probably hold -- but that has to be shown, not assumed. Note also "
           "that CLAUDE.md's '90,418 rows' for the training CSV is stale."),
    C("meth.feature_count_166", "Theory", 166, "describe its 30 input features",
      "feature count", "STALE", "STRUCTURAL", authority=SIDECAR, keys=["n_features"],
      drift=True,
      note="One of six places the manuscript states the feature count as a design "
           "constant. It is model-dependent (30 -> 31) and will move again.",
      action="Bank the sidecar's n_features as model_n_features and splice it, so "
             "all six occurrences come from one source."),
    C("meth.target_clip_prep", "Theory", 168, "target preparation clips them",
      "two raw corpus alphas are clipped before the logit", "CORRECT", "STRUCTURAL",
      authority=SIDECAR, keys=["target_transform.forward"],
      note="The clipping step is part of the frozen recipe and is recorded "
           "verbatim in the sidecar. Survives the model swap only if the next "
           "model keeps a bounded target."),
    C("meth.feature_count_172", "Theory", 172, "we extract 30 features",
      "feature count", "STALE", "STRUCTURAL", authority=SIDECAR, keys=["n_features"],
      drift=True, note="Duplicate of meth.feature_count_166.",
      action="See meth.feature_count_166."),
    C("meth.feature_split_19", "Theory", 188, "remaining 19 features summarize the MST",
      "feature family decomposition 11 + 19", "STALE", "STRUCTURAL",
      authority=SIDECAR, keys=["features_in_booster_order"], drift=True,
      note="11 + 19 = 30 no longer adds up: greedy_nn_over_mst makes 31 and "
           "belongs to no named family. The decomposition is a taxonomy the "
           "manuscript maintains by hand.",
      action="Add a FEATURE_FAMILIES map (feature -> family) next to the roster in "
             "model_registry.py and bank the per-family counts. Without it the "
             "taxonomy claims, the appendix tables and the SHAP family shares all "
             "stay hand-maintained."),
    C("meth.identical_212", "Theory", 212, "identical 30-feature vector",
      "Linear and NN share GART's feature vector", "CONTRADICTED", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="Duplicate of abs.identical_features.", action="See abs.identical_features."),
    C("meth.objective", "Theory", 214, "squared-error regression on",
      "training objective", "CONTRADICTED", "STRUCTURAL", authority=SIDECAR,
      keys=["target_transform.forward", "target_transform.inverse"], drift=True,
      note="Framed as a method definition; the recipe changed under it. The "
           "sidecar is authoritative and machine-readable.",
      action="Bank the sidecar's target_transform strings as model_target_forward "
             "/ model_target_inverse so the methods paragraph is spliced, not typed."),
    C("meth.inference_clip", "Theory", 214, "clipped to $[1.0,2.0]$ at inference",
      "existence of a post-hoc inference clip", "CONTRADICTED", "STRUCTURAL",
      authority=SIDECAR, keys=["target_transform.clip_after_inverse"], drift=True,
      note="clip_after_inverse=false. The bound is structural in the logit "
           "inverse; no clip exists to describe."),
    C("meth.alphahat_range", "Theory", 214, "fell in $[1.033,1.905]$",
      "range of predicted alpha on the ND test split", "UNVERIFIABLE", "UNGENERATED",
      action="Load lgbm_model_v3/gart2_final.joblib, score the 16,920 rows of "
             "tsp_features_v4.csv with split=='test', apply the sidecar's inverse "
             "transform, bank min/max as model_alphahat_test_{min,max}. Pure "
             "inference, no retraining, seconds of work.",
      note="VERIFIED the audit's suggestion is sufficient AND cheap. It is also "
           "the only claim in the paper that would substantiate the 'the bound is "
           "structural' argument empirically, so it is worth banking even though "
           "the clip sentence itself has to go."),
    C("meth.optuna", "Theory", 214, "100 Optuna TPE trials",
      "hyperparameter provenance", "CONTRADICTED", "STRUCTURAL", authority=SIDECAR,
      keys=["hyperparameter_provenance"], drift=True, checkable=False,
      note="NO MECHANICAL ROUTE. The sidecar records 'V3's shipped values, frozen. "
           "Not tuned' as free text. A checker can assert the field is unchanged; "
           "it cannot verify a historical statement about how the values were "
           "obtained. Attestation, not measurement."),
    C("meth.early_stopping", "Theory", 214, "validation RMSE supplied early stopping",
      "early-stopping metric", "CONTRADICTED", "STRUCTURAL", authority=SIDECAR,
      keys=["early_stopping_metric", "early_stopping_rounds"], drift=True,
      note="Sidecar says cost_mape on the val split. Machine-readable, unlike the "
           "tuning provenance above."),
    C("meth.tree_count_214", "Theory", 214, "2{,}031 trees",
      "number of boosted trees", "STALE", "GENERATED", authority=SIDECAR,
      keys=["num_trees", "best_iteration"],
      action="Mirror the sidecar's num_trees into the bank as model_num_trees. "
             "Quoted in three separate places (methods, complexity, "
             "tab:hyperparams) and wrong in all three."),
    C("meth.leaves_per_tree", "Theory", 214, "148 leaves per tree",
      "realised mean leaves per tree", "STALE", "UNGENERATED",
      note="The sidecar's hyperparameters.num_leaves=148 is the CAP, not the "
           "realised mean (147.8). Quoting the cap as the realisation happens to "
           "round to the same integer here, which is exactly why nobody caught it.",
      action="See meth.depth_stats -- same exporter."),
    C("meth.traversals", "Theory", 219, "traversals",
      "K = number of tree traversals per prediction", "STALE", "GENERATED",
      authority=SIDECAR, keys=["num_trees"],
      action="Same export as meth.tree_count_214."),
    C("meth.depth_stats", "Theory", 219, "mean root-to-leaf depth",
      "mean and max root-to-leaf depth", "STALE", "UNGENERATED",
      note="lgbm_model_v3/dump_hyperparams.py computes exactly these statistics "
           "but is hardcoded to lgbm_alpha_model_v3.joblib (the predecessor) and "
           "best_params_v3.json, and it prints to stdout without writing a file. "
           "So the code exists and the number does not.",
      action="Repoint dump_hyperparams.py at model_registry.PRODUCTION_BOOSTER and "
             "have it write paper_tooling/tables/booster_stats.json "
             "{num_trees, leaves_mean, leaves_max, leaf_depth_mean, leaf_depth_max, "
             "per_tree_max_depth_mean, n_features}; bank those keys. One exporter "
             "settles meth.leaves_per_tree, meth.depth_stats, meth.comparisons and "
             "the tab:hyperparams row group."),
    C("meth.comparisons", "Theory", 219, "comparisons per prediction",
      "K * Dbar comparisons per prediction", "STALE", "UNGENERATED",
      derivation="model_num_trees * booster_leaf_depth_mean",
      action="Falls out of booster_stats.json (meth.depth_stats) once that exists."),
    C("meth.complexity", "Theory", 219, "the pipeline costs",
      "asymptotic cost of the feature pipeline", "UNVERIFIABLE", "STRUCTURAL",
      authority="lgbm_model_v3/feature_engineering_gart2.py", drift=True,
      checkable=False,
      note="NO MECHANICAL ROUTE. An asymptotic statement cannot be settled by any "
           "artifact; it needs a human re-derivation against the inference module. "
           "It is on the drift list because its CONTENT moved with the model: the "
           "31st feature adds a greedy-nearest-neighbour pass that the stated "
           "bounds do not account for. A timing curve would be corroboration, not "
           "proof."),
    C("meth.shap_count", "Theory", 223, "We assess the 30-feature set",
      "feature count in the SHAP assessment", "STALE", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="Feature count again, this time inside the SHAP paragraph, where it also sets the row count of tab:shap_top.",
      action="See meth.feature_count_166."),
    C("meth.shap_shares", "Theory", 223, "dominance ratio",
      "SHAP family shares 26.4 / 22.2 / 10.0 / 4.6", "UNVERIFIABLE", "UNGENERATED",
      note="The audit's suggestion -- 're-run shap_analyzer_v3.py against "
           "gart2_final.joblib' -- is INSUFFICIENT, on three counts, all verified "
           "in the script. (1) It hardcodes DATA=tsp_features_v3.csv, which has no "
           "greedy_nn_over_mst column, so `X = X[expected]` raises KeyError on a "
           "31-feature booster: the data path must move to tsp_features_v4.csv too. "
           "(2) It prints the top TOP_K=10 rows and writes two PNGs; it persists no "
           "numeric artifact, so a re-run leaves nothing for a checker to read. "
           "(3) It reports PER-FEATURE share_pct only. The manuscript quotes "
           "FAMILY shares ('dimension and node count 22.2%', 'centroid 10.0%'), and "
           "no grouping map exists anywhere in the repo. Repointing the model alone "
           "settles nothing.",
      action="Rewrite as paper_tooling/shap_production.py: read PRODUCTION_BOOSTER "
             "and PRODUCTION_SIDECAR's training_table, sample the test split at "
             "seed 42, write ALL n_features rows to tables/shap_ranking.csv, and "
             "aggregate through the FEATURE_FAMILIES map (see "
             "meth.feature_split_19) into tables/shap_families.csv; bank both. "
             "Same exporter feeds tab:shap_top, which currently prints 30 rows for "
             "a 31-feature model."),

    # ------------------------------------------------------- Provenance (3.3)
    C("prov.184_count", "Provenance", 206, "184 instances",
      "count of instances whose stored tour disagrees with the coordinates",
      "CORRECT", "GENERATED", authority=TOURS,
      action="audit_reference_tours.py writes reference_tour_audit.csv and prints "
             "aggregates. Bank the counts as audit_tours_{n_corrupt,n_rounding,"
             "n_checked}. Model-independent, so it survives the arm A swap."),
    C("prov.drop184_gart", "Provenance", 206, "moves GART 2.0 from 0.877",
      "GART ND MAPE with and without the 184 instances", "STALE", "UNGENERATED",
      note="These are E0/E1 in generalization_results.csv, which reproduce the "
           "PREDECESSOR (0.8769 / 0.8757 at 2,031 trees).",
      action="See gen.* -- one repoint of generalization_experiments.py settles "
             "this row and all ten of Section 4.6."),
    C("prov.refit_clean", "Provenance", 206, "0.8757\\% to 0.8726",
      "refit on the cleaned training split", "STALE", "UNGENERATED",
      action="See gen.*."),
    C("prov.other_estimators", "Provenance", 206, "less than 0.06 percentage points",
      "effect of dropping the 184 on every non-GART estimator", "CORRECT",
      "UNGENERATED",
      note="Model-independent, and true, but nothing regenerates it: "
           "build_paper_tables.py has no 'audited instances removed' bucket, so "
           "the delta was computed once by hand.",
      action="Add a bucket to the 2D/ND/TSPLIB specs that masks out the instances "
             "listed in paper_tooling/corrupt_instances.txt, and bank the per-model "
             "delta. Cheap, model-independent, and it also settles "
             "concl.metric_shift_006."),

    # ------------------------------------------------- Benchmarking setup (4)
    C("bench.daganzo_offset", "Setup", 229, "carries a $+15.4",
      "Daganzo strip-constant bias and dispersion on uniform instances", "CORRECT",
      "BANKED",
      **_bank(["classical_b_random_daganzo_sampling_region_mspe_pct",
               "classical_b_random_daganzo_sampling_region_sdpe_pct"])),
    C("bench.hilbert_229", "Setup", 229, "custom Hilbert sort has 31.01",
      "Hilbert sort MAPE and SDPE on 2D", "CORRECT", "BANKED",
      **_bank(["2d_by_size_total_custom_hilbert_sort_mape_pct",
               "2d_by_size_total_custom_hilbert_sort_sdpe_pct"])),
    C("bench.same30_238", "Setup", 238, "learned estimators built on the same 30",
      "Linear and NN share GART's feature vector", "CONTRADICTED", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="Third of six occurrences of the same false parity statement.",
      action="See abs.identical_features."),
    C("bench.same30_244", "Setup", 244, "consume the identical 30-feature vector",
      "Linear and NN share GART's feature vector", "CONTRADICTED", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="Fourth occurrence, and the one the benchmark-model table leans on.",
      action="See abs.identical_features."),
    C("bench.full_coverage_78", "Setup", 320, "defined on this complete subset",
      "every 2D baseline scores all 78 EUC_2D instances", "CONTRADICTED",
      "GENERATED", authority=COVERAGE,
      note="Pre-existing, not model-driven: gated Kwon covers 5 of 78. GART covers "
           "78/78 (it declines si1032, an EXPLICIT instance outside this stratum).",
      action="coverage.csv is written by build_paper_tables.main() but not banked. "
             "Export coverage_<dataset>_<model_slug>_{n_used,n_declined} plus the "
             "decline reason string. Also settles app.si1032 and the '22 of 23' "
             "phrasing."),

    # -------------------------------------------- Multidimensional results 4.4.1
    C("nd.headline", "ND results", 343, "obtains 0.88\\% MAPE and 1.28",
      "GART ND MAPE and SDPE", "STALE", "BANKED",
      **_bank(["nd_by_size_total_gart_2_0_mape_pct",
               "nd_by_size_total_gart_2_0_sdpe_pct",
               "nd_by_size_total_gart_2_0_mspe_pct"])),
    C("nd.rho_dn", "ND results", 343, "at 1.81\\%/2.94",
      "calibrated rho(d,n) on ND", "CORRECT", "BANKED",
      **_bank(["nd_by_size_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
               "nd_by_size_total_calibrated_mst_ratio_hat_rho_d_n_sdpe_pct"])),
    C("nd.factors", "ND results", 343, "buys a factor of 2.1 on MAPE",
      "GART / rho(d,n) ratio on MAPE and SDPE", "STALE", "BANKED",
      **_bank(["nd_by_size_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
               "nd_by_size_total_gart_2_0_mape_pct",
               "nd_by_size_total_calibrated_mst_ratio_hat_rho_d_n_sdpe_pct",
               "nd_by_size_total_gart_2_0_sdpe_pct"],
              derivation="k0/k1 and k2/k3")),
    C("nd.lookup_102", "ND results", 343, "lookup table of 102 constants",
      "size of the calibrated rho(d,n) table", "CORRECT", "STRUCTURAL",
      authority=CALIB, keys=["rho_dn (102 cells)", "rho_d (17 cells)"],
      note="A property of the frozen baseline table, not of GART. Never "
           "model-dependent.",
      action="Bank len(rho_dn) as calib_rho_dn_cells so the count cannot drift "
             "from the JSON it describes."),
    C("nd.baseline_rows", "ND results", 343, "The same-feature linear model reaches",
      "ND baseline MAPE/SDPE rows", "CORRECT", "BANKED",
      **_bank(["nd_by_size_total_linear_same_features_{mape_pct,sdpe_pct}",
               "nd_by_size_total_calibrated_mst_ratio_hat_rho_d_{mape_pct,sdpe_pct}",
               "nd_by_size_total_l_mathrm_mst_alpha_1_{mape_pct,sdpe_pct}",
               "nd_by_size_total_custom_hilbert_sort_{mape_pct,sdpe_pct}",
               "nd_by_size_total_bhh_{mape_pct,sdpe_pct}"])),
    C("nd.sdpe_by_size", "ND results", 345, "SDPE decreases from 1.98",
      "ND SDPE at n<=10 and n in [501,1000]", "STALE", "BANKED",
      **_bank(["nd_by_size_5_10_gart_2_0_sdpe_pct",
               "nd_by_size_501_1000_gart_2_0_sdpe_pct"])),
    C("nd.sdpe_by_dim", "ND results", 345, "then rises to 0.65",
      "ND SDPE at d=2, d in [30,50], d=100 and its direction", "CONTRADICTED",
      "BANKED",
      **_bank(["nd_by_dim_d2_gart_2_0_sdpe_pct",
               "nd_by_dim_d30_50_gart_2_0_sdpe_pct",
               "nd_by_dim_d100_gart_2_0_sdpe_pct"],
              derivation="k2 > k1  (the sentence asserts a rise; the bank says "
                         "0.4507 < 0.4881, so it falls)"),
      note="The single most dangerous row in the audit: a DIRECTION, not just a "
           "value, and directions are what a reader remembers. A checker that only "
           "compares numbers would flag the digits and miss that the argument "
           "inverted. Worth encoding the inequality as the check."),
    C("nd.mape_by_dim", "ND results", 345, "while MAPE rises from 0.38",
      "ND MAPE at d=2 and d=100", "STALE", "BANKED",
      **_bank(["nd_by_dim_d2_gart_2_0_mape_pct", "nd_by_dim_d100_gart_2_0_mape_pct"],
              derivation="k1 > k0")),
    C("nd.bhh_exact_region", "ND results", 347, "cuts its MAPE from 39.07",
      "BHH with and without the exact sampling region", "CORRECT", "BANKED",
      **_bank(["nd_by_size_total_bhh_mape_pct",
               "nd_by_size_total_bhh_sampling_region_mape_pct"])),

    # ---------------------------------------------------------- 2D results 4.4.2
    C("2d.sdpe_bound", "2D results", 351, "SDPE below 6.3\\% in every size bucket",
      "tightest SDPE bound across 2D size buckets", "STALE", "BANKED",
      **_bank(["2d_by_size_5_10_gart_2_0_sdpe_pct", "2d_by_size_11_50_gart_2_0_sdpe_pct",
               "2d_by_size_51_100_gart_2_0_sdpe_pct", "2d_by_size_101_500_gart_2_0_sdpe_pct",
               "2d_by_size_501_1000_gart_2_0_sdpe_pct"],
              derivation="ceil(max(k0..k4) * 10) / 10"),
      note="A bound, not a cell. It stayed stale because 6.3 was still TRUE of the "
           "new model -- just no longer tight. A value-only checker cannot see a "
           "loose bound; the derivation has to be the tight one."),
    C("2d.sdpe_bucket_501", "2D results", 351, "3.05\\% in the",
      "2D SDPE in the [501,1000] bucket", "STALE", "BANKED",
      **_bank(["2d_by_size_501_1000_gart_2_0_sdpe_pct"])),
    C("2d.half_of_1157", "2D results", 351, "less than half the 11.57",
      "GART aggregate SDPE vs the alpha=1 floor on 2D", "STALE", "BANKED",
      **_bank(["2d_by_size_total_gart_2_0_sdpe_pct",
               "2d_by_size_total_l_mathrm_mst_alpha_1_sdpe_pct"],
              derivation="k0 < k1 / 2")),
    C("2d.headline", "2D results", 362, "obtains 3.33\\% MAPE and 5.19",
      "GART 2D MAPE and SDPE", "STALE", "BANKED",
      **_bank(["2d_by_size_total_gart_2_0_mape_pct",
               "2d_by_size_total_gart_2_0_sdpe_pct"])),
    C("2d.baseline_rows", "2D results", 362, "against 5.75\\%/9.38",
      "2D baseline MAPE/SDPE rows", "CORRECT", "BANKED",
      **_bank(["2d_by_size_total_calibrated_mst_ratio_hat_rho_d_n_{mape_pct,sdpe_pct}",
               "2d_by_size_total_gart_1_0_{mape_pct,sdpe_pct}",
               "2d_by_size_total_asymptotic_mst_ratio_{mape_pct,sdpe_pct}",
               "2d_by_size_total_l_mathrm_mst_alpha_1_{mape_pct,sdpe_pct}"])),
    C("2d.nn_wins", "2D results", 362, "same-feature neural network reaches 2.99",
      "NN_V3 vs GART on 2D", "CONTRADICTED", "BANKED",
      **_bank(["2d_by_size_total_neural_net_same_features_mape_pct",
               "2d_by_size_total_neural_net_same_features_sdpe_pct",
               "2d_by_size_total_gart_2_0_mape_pct",
               "2d_by_size_total_gart_2_0_sdpe_pct"],
              derivation="k0 < k2 (MAPE) and k1 < k3 (SDPE) -- only the second holds")),
    C("2d.hilbert_362", "2D results", 362, "31.01\\% MAPE against 14.24",
      "Hilbert sort on 2D", "CORRECT", "BANKED",
      **_bank(["2d_by_size_total_custom_hilbert_sort_mape_pct",
               "2d_by_size_total_custom_hilbert_sort_sdpe_pct"])),
    C("2d.genclass_span", "2D results", 364, "ranges from 1.84\\% MAPE on the isotropic",
      "GART MAPE isotropic vs Line Noise", "STALE", "BANKED",
      **_bank(["2d_by_genclass_isotropic_gart_2_0_mape_pct",
               "2d_by_genclass_linenoise_gart_2_0_mape_pct"])),
    C("2d.genclass_baselines", "2D results", 364, "calibrated ratio moves from 3.31",
      "baseline MAPE across generator classes", "CORRECT", "BANKED",
      **_bank(["2d_by_genclass_{isotropic,linenoise}_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
               "2d_by_genclass_{isotropic,linenoise}_asymptotic_mst_ratio_mape_pct",
               "2d_by_genclass_{isotropic,linenoise}_l_mathrm_mst_alpha_1_mape_pct"])),

    # --------------------------------------------------- TSPLIB EUC_2D 4.4.3
    C("tsplib.bucket_design", "TSPLIB", 368, "three prespecified size buckets",
      "78 instances split 23 / 16 / 39", "CORRECT", "STRUCTURAL",
      authority="paper_tooling/build_paper_tables.py::B_TSPLIB",
      keys=["tsplib_by_size_{51_150,151_400,gt400}_gart_2_0_n"],
      note="Bucket edges are a prespecified design choice; the occupancy counts "
           "follow from the corpus. Model-independent unless a model declines a "
           "EUC_2D instance."),
    C("tsplib.headline", "TSPLIB", 380, "obtains 3.42\\% SDPE and 3.27",
      "GART TSPLIB MAPE and SDPE", "STALE", "BANKED",
      **_bank(["tsplib_by_size_total_gart_2_0_sdpe_pct",
               "tsplib_by_size_total_gart_2_0_mape_pct"])),
    C("tsplib.baseline_rows", "TSPLIB", 380, "against 4.49\\%/3.77",
      "TSPLIB baseline SDPE/MAPE rows", "CORRECT", "BANKED",
      **_bank(["tsplib_by_size_total_calibrated_mst_ratio_hat_rho_d_n_{sdpe_pct,mape_pct}",
               "tsplib_by_size_total_asymptotic_mst_ratio_{sdpe_pct,mape_pct}",
               "tsplib_by_size_total_gart_1_0_{sdpe_pct,mape_pct}"])),
    C("tsplib.alpha_moments", "TSPLIB", 380, "has mean 1.1306",
      "mean and sd of true alpha over the 78 EUC_2D instances", "CORRECT",
      "UNGENERATED",
      note="Model-independent corpus statistic, and it will NOT move when the "
           "model swaps -- but nothing exports it, so it is unprotected against a "
           "change in the screen that selects the 78.",
      action="Bank alpha moments per instance set from the ground-truth frame that "
             "build_paper_tables already loads: "
             "corpus_alpha_<set>_{n,mean,sd,min,max}. Same exporter also settles "
             "app.alpha_range_22."),
    C("tsplib.oracle_constant", "TSPLIB", 380, "MAPE-minimising constant on these instances",
      "c* = 1.1275 and its 3.54% MAPE over the 78", "CORRECT", "UNGENERATED",
      action="See abs.oracle_constant_78 -- one oracle-constant exporter covers "
             "the 78, the 111 and the 23."),
    C("tsplib.bound_margin", "TSPLIB", 380, "improves on that bound by 0.27",
      "GART's margin over the best constant", "STALE", "UNGENERATED",
      derivation="oracle_constant_euc78_mape_pct - tsplib_by_size_total_gart_2_0_mape_pct",
      note="One banked term, one ungenerated term. Blocked on the oracle-constant "
           "exporter.",
      action="See abs.oracle_constant_78."),
    C("tsplib.dispersion_gap", "TSPLIB", 380, "dispersion gap is the larger one",
      "GART SDPE vs asymptotic-ratio SDPE on TSPLIB", "STALE", "BANKED",
      **_bank(["tsplib_by_size_total_gart_2_0_sdpe_pct",
               "tsplib_by_size_total_asymptotic_mst_ratio_sdpe_pct"])),
    C("tsplib.gt400_overlap", "TSPLIB", 382, "SDPE intervals overlap",
      "overlap of the two bootstrap SDPE intervals in the n>400 bucket", "CORRECT",
      "BANKED",
      **_bank(["tsplib_by_size_gt400_gart_2_0_sdpe_lo",
               "tsplib_by_size_gt400_gart_2_0_sdpe_hi",
               "tsplib_by_size_gt400_asymptotic_mst_ratio_sdpe_lo",
               "tsplib_by_size_gt400_asymptotic_mst_ratio_sdpe_hi"],
              derivation="k1 > k2 and k3 > k0")),
    C("tsplib.n_gt_10000", "TSPLIB", 382, "post hoc",
      "GART vs asymptotic ratio on the 5 instances with n>10,000", "CONTRADICTED",
      "UNGENERATED",
      note="No such bucket exists. B_TSPLIB stops at 'n>400', so the slice was cut "
           "by hand and never regenerated. It is the slice the paper leans on to "
           "concede a near-tie at large n -- and the concession is now false.",
      action="Add ('$n>10{,}000$', 'gt10000', _rng('n', 10001, 10**9)) to B_TSPLIB, "
             "or a separate post-hoc spec so it is not confused with the "
             "prespecified buckets. Then it is BANKED like every other bucket."),
    C("tsplib.d18512_ratio", "TSPLIB", 382, "more than $18\\times$ the training-size cap",
      "largest TSPLIB instance relative to the n<=1000 training cap", "CORRECT",
      "STRUCTURAL",
      note="18,512 / 1,000. Both terms are corpus/design constants. Would only "
           "move if the training-size cap moved."),
    C("tsplib.rho_d_vs_dn", "TSPLIB", 384, "gives 11.97\\% MAPE here",
      "rho(2) vs rho(2,n) on TSPLIB", "CORRECT", "BANKED",
      **_bank(["tsplib_by_size_total_calibrated_mst_ratio_hat_rho_d_mape_pct",
               "tsplib_by_size_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct"]),
      note="The constant 1.2610 itself is STRUCTURAL: calibrated_alpha_table.json "
           "rho_d['2']."),
    C("tsplib.classical_rows", "TSPLIB", 384, "Chien 30.41",
      "classical estimators on TSPLIB EUC_2D", "CORRECT", "BANKED",
      **_bank(["classical_a_tsplib_cavdar_sokol_mape_pct",
               "classical_a_tsplib_bhh_mape_pct",
               "classical_a_tsplib_chien_extrapolated_mape_pct",
               "classical_a_tsplib_daganzo_mape_pct",
               "classical_a_tsplib_kwon_golden_wasil_extrapolated_mape_pct"])),

    # ------------------------------------------------ Matched-domain (4.5)
    C("matched.classical_rows", "Matched domain", 390, "BHH falls from 23.82",
      "classical estimators, full vs matched domain", "CORRECT", "BANKED",
      **_bank(["classical_a_2d_{bhh,cavdar_sokol,daganzo}_mape_pct",
               "classical_b_random_{bhh,cavdar_sokol,daganzo,chien,kwon}_sampling_region_"
               "{mape_pct,mspe_pct}",
               "classical_b_kwon_kwon_golden_wasil_sampling_region_mape_pct"])),
    C("matched.kwon_domain", "Matched domain", 392, "80 Kwon-domain instances",
      "GART vs Kwon on the Kwon domain", "STALE", "BANKED",
      **_bank(["classical_b_kwon_gart_2_0_mape_pct",
               "classical_b_kwon_kwon_golden_wasil_sampling_region_mape_pct"])),
    C("matched.chien_domain", "Matched domain", 392, "50 Chien-domain instances",
      "GART vs Chien on the Chien domain", "STALE", "BANKED",
      **_bank(["classical_b_chien_gart_2_0_mape_pct",
               "classical_b_chien_chien_sampling_region_mape_pct"])),
    C("matched.uniform_210", "Matched domain", 392, "210 uniform instances",
      "GART vs BHH and the alpha=1 floor on the uniform subset", "STALE", "BANKED",
      **_bank(["classical_b_random_gart_2_0_mape_pct",
               "classical_b_random_bhh_sampling_region_mape_pct",
               "classical_b_random_l_mathrm_mst_alpha_1_mape_pct"])),
    C("matched.nn_edges", "Matched domain", 392, "again edges the boosted ensemble",
      "NN_V3 vs GART on the 210 uniform instances", "CONTRADICTED", "BANKED",
      **_bank(["classical_b_random_gart_2_0_mape_pct",
               "classical_b_random_neural_net_same_features_mape_pct"],
              derivation="k1 < k0")),
    C("matched.nn_pvalue", "Matched domain", 392, "not significant",
      "significance of the NN-GART difference on the 210", "STALE", "GENERATED",
      authority=PAIRED,
      keys=["table=classical,bucket_slug=b_random,model_b=NN_V3 -> wilcoxon_p"],
      note="p=0.73 is now the KWON-DOMAIN comparison (n=80); the 210-instance "
           "comparison is p=0.069. A p-value that migrated between rows is exactly "
           "what a bucket_slug-keyed export prevents.",
      action="See abs.tsplib_paired."),
    C("matched.kwon_factor", "Matched domain", 392, "factor of 2.4 over Kwon",
      "GART / Kwon ratio on the Kwon domain", "STALE", "BANKED",
      **_bank(["classical_b_kwon_kwon_golden_wasil_sampling_region_mape_pct",
               "classical_b_kwon_gart_2_0_mape_pct"], derivation="k0 / k1")),

    # ------------------------------------------------- Generalization (4.6)
    C("gen.held_out_sampling", "Generalization", 396, "therefore measures held-out sampling",
      "GART ND MAPE as the interpolation reference", "STALE", "UNGENERATED",
      action="GEN-REPOINT (see note on gen.e0_bitforbit)."),
    C("gen.e0_bitforbit", "Generalization", 396, "reproduces the released model bit-for-bit",
      "E0 baseline refit reproduces the shipped model", "CONTRADICTED", "UNGENERATED",
      note="CONFIRMED at code level. generalization_experiments.py pins "
           "DATA_FILE=tsp_features_v3.csv, PARAMS_FILE=lgbm_model_v3/"
           "best_params_v3.json and SHIPPED_MODEL_FILE=lgbm_model_v3/"
           "lgbm_alpha_model_v3.joblib -- all three are the PREDECESSOR. Its E0 row "
           "reproduces LGBM_V3 bit-for-bit (0.8769 MAPE, 2,031 trees) and the "
           "manuscript reports that as the production model's reproducibility "
           "calibration. Every number in Section 4.6 inherits the error.",
      action="GEN-REPOINT: point generalization_experiments.py at "
             "model_registry.PRODUCTION_BOOSTER / PRODUCTION_SIDECAR and read the "
             "recipe (training_table, target transform, monotone constraints, "
             "early-stopping metric, frozen hyperparameters) from the sidecar "
             "rather than restating it. BLOCKED IN THIS TASK: four refits is "
             "training, and this task forbids it. Also worth gating -- have E0 "
             "assert its output hashes to PRODUCTION_SIDECAR's sha256 and fail "
             "loudly otherwise, which is the check that would have caught this."),
    C("gen.16846", "Generalization", 396, "16{,}846 rows",
      "cleaned-split test row count and its MAPE", "STALE", "UNGENERATED",
      note="Row count 16,846 is correct and structural; the MAPE attached to it is "
           "the predecessor's.",
      action="GEN-REPOINT."),
    C("gen.leave_dims_out", "Generalization", 398, "Withholding both",
      "leave-dimensions-out refit at d=15 and d=25", "STALE", "UNGENERATED",
      action="GEN-REPOINT."),
    C("gen.neighbour_dims", "Generalization", 398, "move only from 0.53",
      "unaffected neighbouring dimensions", "STALE", "UNGENERATED",
      action="GEN-REPOINT."),
    C("gen.signed_error_d15", "Generalization", 398, "mean signed error at",
      "mean signed error shift at d=15", "STALE", "UNGENERATED",
      action="GEN-REPOINT."),
    C("gen.leave_large_n_out", "Generalization", 398, "Training only on",
      "leave-large-n-out refit", "STALE", "UNGENERATED", action="GEN-REPOINT."),
    C("gen.within_range", "Generalization", 400, "within the range the released model achieves",
      "held-out regimes land inside the released model's range", "CONTRADICTED",
      "UNGENERATED",
      note="The argument INVERTS, not just the digits: 0.85-0.90% was inside the "
           "predecessor's range and is well outside the production model's 0.62%. "
           "A number-only checker would report a small delta on a sentence whose "
           "conclusion has reversed.",
      action="GEN-REPOINT."),
    C("gen.cost_028", "Generalization", 400, "worth up to 0.28 MAPE points",
      "cost of withholding a dimension", "STALE", "UNGENERATED",
      note="Also internally inconsistent with the 0.35 pp at line 486.",
      action="GEN-REPOINT."),

    # ------------------------------------ Degenerate geometry / augmentation 4.7
    C("aug.linenoise_vs_isotropic", "Degenerate geometry", 404, "on the near-collinear Line Noise class",
      "GART MAPE on Line Noise vs isotropic", "STALE", "BANKED",
      **_bank(["2d_by_genclass_linenoise_gart_2_0_mape_pct",
               "2d_by_genclass_isotropic_gart_2_0_mape_pct"]),
      note="Internally inconsistent in the manuscript: 11.59 here, 11.60 at lines "
           "364 and 428, for the same quantity. Banking it removes the possibility."),
    C("aug.linenoise_mspe", "Degenerate geometry", 404, "so the model systematically under-predicts",
      "Line Noise MSPE", "STALE", "BANKED",
      **_bank(["2d_by_genclass_linenoise_gart_2_0_mspe_pct"])),
    C("aug.slope_corr", "Degenerate geometry", 404, "gives a slope of 0.29",
      "regression of predicted alpha on true alpha, Line Noise n>=200", "UNVERIFIABLE",
      "UNGENERATED",
      note="The audit proposes refitting on gart2_final predictions for the 210 "
           "Line Noise rows. That WOULD settle it, with one correction: the "
           "sentence restricts to n>=200, so the fit is not over all 210. "
           "paper_tooling/armA_verify_oracle_constant.csv carries slope_frozen for "
           "line_noise at n=210 -- a different subset, and part of the arm study, "
           "so it is not the authority here.",
      action="Bank an alpha-on-alpha OLS per stratum from the per-instance "
             "predictions build_paper_tables already has in hand: "
             "alphafit_<bucket>_{slope,intercept,pearson_r,n}, with the n>=200 "
             "restriction as an explicit bucket. Settles this, aug.half_linenoise "
             "and concl.slope_corr."),
    C("aug.alpha_gt_145", "Degenerate geometry", 404, "occurs essentially only at",
      "alpha>1.45 confined to n<=10 in the training split", "CORRECT", "UNGENERATED",
      note="Model-independent corpus fact, never re-derived.",
      action="Same corpus-statistics exporter as meth.alpha_sd_by_d: bank "
             "corpus_train_alpha_gt145_max_n or the n-histogram of alpha>1.45."),
    C("aug.success_criteria", "Degenerate geometry", 406, "Success criteria were fixed before any augmented model",
      "prespecified adoption thresholds for the augmentation", "STALE", "STRUCTURAL",
      drift=True, checkable=False,
      note="NO MECHANICAL ROUTE to the prespecification itself -- that a threshold "
           "was fixed in advance is a fact about the process, not a value. FLAGGED "
           "because the thresholds are stated as absolute budgets ('<=0.05 pp ND "
           "regression') measured against a baseline that no longer ships, so a "
           "constant reads as having moved. Whether the criteria are MET is "
           "checkable once aug.slope_corr is banked."),
    C("aug.round1", "Degenerate geometry", 408, "added 578 instances",
      "first augmentation round", "UNVERIFIABLE", "UNGENERATED",
      note="augmentation_experiment.py pins tsp_features_v3.csv, best_params_v3.json "
           "and lgbm_alpha_model_v3.joblib; augmentation_results.csv predates the "
           "model change.",
      action="AUG-REPOINT: same treatment as GEN-REPOINT, and equally blocked -- "
             "both rounds retrain. If the augmentation study is not re-run, the "
             "honest move is to label Section 4.7's round-1/round-2 numbers "
             "explicitly as the predecessor's, which is a prose decision, not a "
             "tooling one."),
    C("aug.round2", "Degenerate geometry", 408, "second round therefore added",
      "second augmentation round", "UNVERIFIABLE", "UNGENERATED",
      note="augmentation_v2_experiment.py, same three hardcodes.",
      action="AUG-REPOINT."),
    C("aug.corpus_rho_alpha", "Degenerate geometry", 408, "median $\\rho$ between",
      "median rho and alpha of the augmentation corpus", "CORRECT", "UNGENERATED",
      note="Model-independent corpus geometry; nothing exports it.",
      action="Bank descriptive statistics of the augmentation corpora alongside "
             "the corpus-statistics exporter."),
    C("aug.mean_pred_alpha", "Degenerate geometry", 410, "mean predicted",
      "mean predicted alpha before/after augmentation and the [401,1000] bucket",
      "UNVERIFIABLE", "UNGENERATED", action="AUG-REPOINT."),
    C("aug.half_linenoise", "Degenerate geometry", 410, "half the Line Noise",
      "recall experiment slope and MAPE", "UNVERIFIABLE", "UNGENERATED",
      action="AUG-REPOINT (plus the alpha-on-alpha exporter of aug.slope_corr)."),
    C("aug.geometry_facts", "Degenerate geometry", 412, "kurtosis",
      "kurtosis, width ratio, crossover at rho~8, alpha offset", "CORRECT",
      "UNGENERATED",
      note="Model-independent geometry, never re-derived.",
      action="Corpus-statistics exporter."),
    C("aug.grid_mspe", "Degenerate geometry", 412, "sub-generator's MSPE improved",
      "grid sub-generator MSPE before/after augmentation", "UNVERIFIABLE",
      "UNGENERATED", action="AUG-REPOINT."),

    # ------------------------------------------------------- Discussion (4.8)
    C("disc.rank_gart", "Discussion", 418, "Spearman",
      "GART Spearman/Kendall on all three benchmarks", "STALE", "BANKED",
      **_bank(["rank_2d_gart_2_0_{spearman_rho,kendall_tau}",
               "rank_nd_gart_2_0_{spearman_rho,kendall_tau}",
               "rank_tsplib_euc2d_gart_2_0_{spearman_rho,kendall_tau}"]),
      note="The audit called tab:rank unbacked because gen_paper_numbers.py "
           "persisted nothing. That is no longer true: build_paper_tables.py now "
           "computes rank statistics and banks them. Re-checked against the live "
           "bank."),
    C("disc.rank_control", "Discussion", 418, "control obtains 0.9961",
      "alpha=1 control rank statistics", "CORRECT", "BANKED",
      **_bank(["rank_2d_l_mathrm_mst_alpha_1_{spearman_rho,kendall_tau}",
               "rank_nd_l_mathrm_mst_alpha_1_{spearman_rho,kendall_tau}",
               "rank_tsplib_euc2d_l_mathrm_mst_alpha_1_{spearman_rho,kendall_tau}"])),
    C("disc.pairs_2d", "Discussion", 420, "2D pairs,",
      "2D close-pair universe at the 5% threshold", "STALE", "STRUCTURAL",
      authority=NUMBER_BANK, keys=["rank_2d_gart_2_0_close5_pairs"], drift=True,
      note="FLAGGED, and this one did NOT move with the model -- it moved with the "
           "GENERATOR. The manuscript says 74,835; the bank says 74,759, because "
           "build_paper_tables.py redefined the qualifying predicate (exact ties "
           "now excluded, and the enumeration is exhaustive rather than sampled). "
           "A pair universe is a design constant; when one silently changes "
           "underneath a fixed number, every percentage keyed to it moves too."),
    C("disc.close5_2d", "Discussion", 420, "Close-pair ordering is the discriminating measure",
      "GART 2D close-pair ordering accuracy at 5%", "STALE", "BANKED",
      **_bank(["rank_2d_gart_2_0_close5_pct"])),
    C("disc.pairs_nd", "Discussion", 420, "multidimensional pairs, and",
      "ND close-pair sampling design", "UNVERIFIABLE", "STRUCTURAL",
      authority=NUMBER_BANK,
      keys=["rank_nd_gart_2_0_pair_mode", "rank_nd_gart_2_0_close5_pairs",
            "rank_nd_gart_2_0_pair_seed"], drift=True,
      note="FLAGGED. The audit says the fixed-seed sampler 'lives only in "
           "gen_paper_numbers.py and persists nothing'. That is now out of date "
           "and the suggested fix -- re-run gen_paper_numbers.py -- would "
           "REINTRODUCE the superseded design. build_paper_tables.py enumerates ND "
           "pairs EXHAUSTIVELY (1,662,781 qualifying at 5%, 3,290,075 at 10%), "
           "records pair_mode='exhaustive' and sets pair_seed/pair_draws to 'n/a' "
           "because no draw is made. The 46,434-pair sample no longer exists as a "
           "concept. Line 830's appendix sentence describes the same dead design."),
    C("disc.close5_nd", "Discussion", 420, "multidimensional pairs, and ",
      "GART ND close-pair ordering accuracy at 5%", "UNVERIFIABLE", "BANKED",
      **_bank(["rank_nd_gart_2_0_close5_pct"]),
      note="Reclassified from the audit's UNVERIFIABLE: it is banked now, under "
           "the exhaustive definition (92.37%), not the sampled one."),
    C("disc.close5_tsplib", "Discussion", 420, "TSPLIB pairs correctly",
      "GART TSPLIB close-pair ordering accuracy at 5%", "STALE", "BANKED",
      **_bank(["rank_tsplib_euc2d_gart_2_0_close5_pct",
               "rank_tsplib_euc2d_gart_2_0_close5_pairs"])),
    C("disc.close5_control", "Discussion", 420, "for the",
      "alpha=1 control close-pair accuracy on all three", "CORRECT", "BANKED",
      **_bank(["rank_2d_l_mathrm_mst_alpha_1_close5_pct",
               "rank_nd_l_mathrm_mst_alpha_1_close5_pct",
               "rank_tsplib_euc2d_l_mathrm_mst_alpha_1_close5_pct"])),
    C("disc.gain_nd", "Discussion", 420, "multidimensional gain of",
      "ND close-pair gain over the control", "UNVERIFIABLE", "BANKED",
      **_bank(["rank_nd_gart_2_0_close5_pct",
               "rank_nd_l_mathrm_mst_alpha_1_close5_pct"], derivation="k0 - k1"),
      note="Reclassified from UNVERIFIABLE: both terms are banked."),
    C("disc.gain_2d", "Discussion", 420, "2D gain of",
      "2D close-pair gain over the control", "CONTRADICTED", "BANKED",
      **_bank(["rank_2d_gart_2_0_close5_pct",
               "rank_2d_l_mathrm_mst_alpha_1_close5_pct"], derivation="k0 - k1")),
    C("disc.gain_tsplib", "Discussion", 420, "pairs cannot support a",
      "TSPLIB close-pair gain over the control", "STALE", "BANKED",
      **_bank(["rank_tsplib_euc2d_gart_2_0_close5_pct",
               "rank_tsplib_euc2d_l_mathrm_mst_alpha_1_close5_pct"],
              derivation="k0 - k1")),
    C("disc.close10", "Discussion", 420, "At a 10\\% threshold",
      "GART close-pair accuracy at the 10% threshold", "STALE", "BANKED",
      **_bank(["rank_2d_gart_2_0_close10_pct", "rank_nd_gart_2_0_close10_pct",
               "rank_tsplib_euc2d_gart_2_0_close10_pct"])),
    C("disc.nn_close_pairs", "Discussion", 420, "orders TSPLIB close pairs better",
      "NN_V3 vs GART on TSPLIB close pairs", "STALE", "BANKED",
      **_bank(["rank_tsplib_euc2d_neural_net_same_features_close5_pct",
               "rank_tsplib_euc2d_gart_2_0_close5_pct"], derivation="k0 > k1")),
    C("disc.r2_alpha_gt400", "Discussion", 424, "R^2_\\alpha",
      "R^2_alpha in the TSPLIB n>400 bucket", "STALE", "BANKED",
      **_bank(["tsplib_by_size_gt400_gart_2_0_r2_alpha"])),
    C("disc.mape_gt400", "Discussion", 424, "tour-cost MAPE remains",
      "GART MAPE in the TSPLIB n>400 bucket", "STALE", "BANKED",
      **_bank(["tsplib_by_size_gt400_gart_2_0_mape_pct"])),
    C("disc.asym_paired", "Discussion", 426, "indistinguishable from zero",
      "paired GART vs asymptotic ratio on TSPLIB", "CONTRADICTED", "GENERATED",
      authority=PAIRED,
      keys=["table=tsplib_by_size,bucket_slug=total,model_b=Asymptotic_MST"],
      note="Same statistic as abs.tsplib_paired, quoted twice with the same stale "
           "value. Direction of the conclusion reverses: p moved from 0.77 to "
           "4.85e-3.",
      action="See abs.tsplib_paired."),
    C("disc.dispersion_gain", "Discussion", 426, "real gain is dispersion",
      "GART SDPE vs asymptotic-ratio SDPE on TSPLIB", "STALE", "BANKED",
      **_bank(["tsplib_by_size_total_gart_2_0_sdpe_pct",
               "tsplib_by_size_total_asymptotic_mst_ratio_sdpe_pct"])),
    C("disc.nn_paired", "Discussion", 426, "by 0.34 percentage points",
      "paired NN vs GART difference on 2D", "CONTRADICTED", "GENERATED",
      authority=PAIRED,
      keys=["table=2d_by_size,bucket_slug=total,model_b=NN_V3 -> mean_diff, ci_lo, "
            "ci_hi, wilcoxon_p"],
      note="The SIGN flips. The CI [0.27,0.43] excluded zero; the current one "
           "[-0.164,-0.008] sits on the other side.",
      action="See abs.tsplib_paired."),
    C("disc.boosting_conclusion", "Discussion", 426, "not the source of the advantage",
      "attribution of the 2D advantage to features rather than boosting",
      "CONTRADICTED", "GENERATED", authority=PAIRED, checkable=False,
      note="NO MECHANICAL ROUTE to the claim itself: it is a causal attribution, "
           "and no artifact can certify 'the feature set is the source'. Its "
           "PREMISE (that the NN beats the boosted ensemble) is banked, and that "
           "premise is now false, so the sentence falls with it. A checker can "
           "flag the premise and must not be asked to judge the inference.",
      action="Bind this sentence to disc.nn_paired in the manifest so that when "
             "the premise flips, the conclusion is surfaced for human review."),
    C("disc.genclass_five", "Discussion", 428, "on the biased class",
      "GART MAPE across the five generator classes", "STALE", "BANKED",
      **_bank(["2d_by_genclass_isotropic_gart_2_0_mape_pct",
               "2d_by_genclass_biased_gart_2_0_mape_pct",
               "2d_by_genclass_clustered_gart_2_0_mape_pct",
               "2d_by_genclass_geometric_gart_2_0_mape_pct",
               "2d_by_genclass_linenoise_gart_2_0_mape_pct"])),
    C("disc.six_times", "Discussion", 428, "six times the isotropic error",
      "Line Noise / isotropic error ratio", "STALE", "BANKED",
      **_bank(["2d_by_genclass_linenoise_gart_2_0_mape_pct",
               "2d_by_genclass_isotropic_gart_2_0_mape_pct"], derivation="k0 / k1")),

    # ------------------------------------------------------------- Timing (430)
    C("time.decomposition", "Timing", 430, "of total time on feature extraction",
      "82.6 / 16.3 / 1.2 time decomposition", "PENDING", "UNGENERATED",
      note="CONFIRMED as genuinely sourceless, and worse than the audit implies. "
           "tables/table_time_breakdown.csv is dated Apr 18, has no MST column at "
           "all (only pct_features / pct_inference / pct_other), reports 74.75 / "
           "25.03 / 0.22 for TSPLIB, and carries the stale ND count 16,907. It "
           "does not back the sentence even for the predecessor. Meanwhile "
           "all_models_tsplib.csv tags all 109 GART_2.0 rows "
           "'pending_no_serial_measurement', and build_paper_tables' "
           "publishable_times() guard withholds the whole group, which is why "
           "tsplib_by_size_total_gart_2_0_time_ms is null in the bank.",
      action="Requires a low-contention serial run, then "
             "restore_tsplib_serial_timings.py, then a decomposition exporter that "
             "separates MST construction from the rest of feature extraction. "
             "Route exists; it is a measurement, not a computation."),
    C("time.median_vs_reference", "Timing", 430, "slower than generating the reference tour",
      "GART median prediction time vs reference tour time on ND", "PENDING",
      "UNGENERATED",
      note="Numbers EXIST but are not trustworthy, which is the worst state to be "
           "in. nd_by_size_total_gart_2_0_time_ms = 121.5 ms is banked against the "
           "manuscript's 171 ms, so the sentence's direction reverses. But the ND "
           "frame carries no timing_provenance column, so publishable_times() "
           "passes it through unguarded -- the 121.5 ms has the same contended "
           "origin the TSPLIB guard exists to suppress. Do not adopt it.",
      action="Extend timing_provenance to the 2D and ND benchmark CSVs so the "
             "guard covers them, then re-measure. Until then the bank's 2D/ND "
             "time_ms keys should be treated as UNPUBLISHABLE, not as authority. "
             "The reference-tour times are computed by _ref_times() and reach the "
             ".tex fragments but are never banked -- export them as "
             "ref_tour_<table>_<bucket_slug>_time_ms."),
    C("time.bucket_501", "Timing", 430, "the reference tour takes",
      "reference tour vs GART at n in [501,1000]", "PENDING", "UNGENERATED",
      note="Bank says 465.0 ms against the manuscript's 272 ms; same provenance "
           "problem as time.median_vs_reference.",
      action="See time.median_vs_reference."),
    C("time.d18512", "Timing", 430, "GART predicts in",
      "GART prediction time on d18512", "PENDING", "UNGENERATED",
      note="all_models_tsplib.csv records total_time_s=0.7525 for GART_2.0 on "
           "d18512, tagged pending. Neither 127 ms nor 752 ms is a clean "
           "measurement.",
      action="See time.decomposition."),

    # ------------------------------------------------------- Application (5)
    C("app.oracle_111", "Application", 434, "MAPE-minimising constant",
      "c* = 1.1341 over the full 111-instance TSPLIB set", "CORRECT", "UNGENERATED",
      action="See abs.oracle_constant_78."),
    C("app.oracle_23", "Application", 434, "on these 23 instances",
      "c* = 1.1718 and 7.55% over the 23 screened non-EUC_2D", "CORRECT",
      "UNGENERATED", action="See abs.oracle_constant_78."),
    C("app.two_of_thirty", "Application", 436, "two of the model's",
      "MDS pipeline omits 2 of the model's features", "STALE", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="The denominator is the feature count, so it moved with it; the numerator (two omitted features) is a property of the MDS pipeline and did not.",
      action="See meth.feature_count_166."),
    C("app.tsplib_taxonomy", "Application", 438, "non-EUC\\_2D",
      "33 non-EUC_2D split 4/2/10/17, leaving 23 including seven EXPLICIT",
      "CORRECT", "STRUCTURAL",
      authority="tsplib_benchmark/exclusions.py",
      keys=["tsplib_nonEuc_{att,ceil2d,geo,explicit}_*_n"],
      note="A corpus taxonomy fixed by the TSPLIB headers and the screen. Never "
           "model-dependent -- but see app.seven_explicit, where the manuscript "
           "silently reuses the taxonomy count as a scoring count."),
    C("app.mds_cap", "Application", 442, "five of 19 MDS cases",
      "cases the embedding-dimension cap keeps below 99.9% stress", "CORRECT",
      "GENERATED", authority=MDS,
      action="audit_mds_distortion.py writes mds_distortion_screened.csv. Bank the "
             "below-target counts as mds_below_target_{att,geo,explicit,total} and "
             "mds_cases_total."),
    C("app.si1032", "Application", 448, "declines \\texttt{si1032}",
      "si1032 declined on the greedy-ratio gate; 22 scored of 23", "CORRECT",
      "GENERATED", authority=COVERAGE,
      note="Already corrected in the manuscript. The gate floor 1.035 is "
           "STRUCTURAL (an estimator config constant); the declined instance and "
           "its ratio 1.0260 come from coverage.csv.",
      action="See bench.full_coverage_78 for the coverage export."),
    C("app.noneuc_results", "Application", 480, "On the 22 screened instances",
      "non-Euclidean MDS pipeline results", "CORRECT", "BANKED",
      **_bank(["tsplib_nonEuc_total_gart_2_0_{sdpe_pct,mape_pct}",
               "tsplib_nonEuc_total_fixed_alpha_1_136_{sdpe_pct,mape_pct}",
               "tsplib_nonEuc_att_gart_2_0_mape_pct",
               "tsplib_nonEuc_ceil2d_*_sdpe_pct"])),
    C("app.alpha_range_22", "Application", 480, "ranges from 1.006",
      "range of true alpha over the 22 accepted non-EUC_2D instances", "CORRECT",
      "UNGENERATED",
      note="Model-independent, but tied to a model-dependent instance set (the 22 "
           "GART accepts), which is precisely the kind of coupling that breaks "
           "silently on a model swap.",
      action="Same corpus-alpha exporter as tsplib.alpha_moments, keyed on the "
             "accepted set rather than the screened set, so the coupling is "
             "explicit."),

    # ------------------------------------------------------- Conclusion (6)
    C("concl.feature_families", "Conclusion", 484, "MST, centroid, and coordinate-range features",
      "feature count and family taxonomy", "STALE", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features", "features_in_booster_order"], drift=True,
      note="Two defects in one clause: the count moved 30 -> 31, and the 31st "
           "(greedy_nn_over_mst) is a greedy-tour ratio belonging to none of the "
           "three named families.",
      action="See meth.feature_split_19 (FEATURE_FAMILIES) and "
             "meth.feature_count_166."),
    C("concl.sharing_vector", "Conclusion", 484, "two learned models sharing its feature vector",
      "Linear and NN share GART's feature vector", "CONTRADICTED", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="Sixth and final occurrence, in the sentence a reader is most likely to quote.",
      action="See abs.identical_features."),
    C("concl.lowest_sdpe", "Conclusion", 484, "lowest aggregate SDPE",
      "GART has the lowest SDPE on ND and TSPLIB", "CORRECT", "BANKED",
      **_bank(["nd_by_size_total_*_sdpe_pct", "tsplib_by_size_total_*_sdpe_pct"],
              derivation="argmin over the shipped roster"),
      note="Roster-conditional: fails on ND if GART 2.0 (V4 features) ships "
           "(0.9683 < 0.9881).",
      action="See intro.superlative_all3 (SHIPPED_ROSTER)."),
    C("concl.lowest_2d", "Conclusion", 484, "lowest of any estimator not sharing",
      "GART has the lowest 2D SDPE among non-feature-sharing estimators",
      "CONTRADICTED", "BANKED",
      **_bank(["2d_by_size_total_gart_2_0_sdpe_pct",
               "2d_by_size_total_neural_net_same_features_sdpe_pct"],
              derivation="k0 < k1"),
      action="See intro.superlative_all3."),
    C("concl.nd_factor", "Conclusion", 484, "factor of 2.1 on MAPE over a calibrated constant",
      "GART / rho(d,n) on ND", "STALE", "BANKED",
      **_bank(["nd_by_size_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
               "nd_by_size_total_gart_2_0_mape_pct"], derivation="k0 / k1")),
    C("concl.kwon_factor", "Conclusion", 484, "factor of 2.4 over Kwon",
      "GART / Kwon on the Kwon domain", "STALE", "BANKED",
      **_bank(["classical_b_kwon_kwon_golden_wasil_sampling_region_mape_pct",
               "classical_b_kwon_gart_2_0_mape_pct"], derivation="k0 / k1")),
    C("concl.tsplib_margin", "Conclusion", 484, "0.28 percentage points on TSPLIB",
      "GART margin over the asymptotic ratio on TSPLIB", "STALE", "BANKED",
      **_bank(["tsplib_by_size_total_asymptotic_mst_ratio_mape_pct",
               "tsplib_by_size_total_gart_2_0_mape_pct"], derivation="k0 - k1")),
    C("concl.paired_zero", "Conclusion", 484, "paired difference against the asymptotic ratio",
      "significance of that margin", "CONTRADICTED", "GENERATED", authority=PAIRED,
      action="See abs.tsplib_paired."),
    C("concl.dispersion_gain", "Conclusion", 484, "real gain is 28",
      "dispersion gain on TSPLIB", "STALE", "BANKED",
      **_bank(["tsplib_by_size_total_gart_2_0_sdpe_pct",
               "tsplib_by_size_total_asymptotic_mst_ratio_sdpe_pct"],
              derivation="100 * (1 - k0 / k1)")),
    C("concl.nn_wins", "Conclusion", 484, "same 30 features is more accurate",
      "NN beats the boosted ensemble on 2D", "CONTRADICTED", "BANKED",
      **_bank(["2d_by_size_total_neural_net_same_features_mape_pct",
               "2d_by_size_total_gart_2_0_mape_pct"], derivation="k0 < k1")),
    C("concl.noneuc", "Conclusion", 484, "hybrid MDS pipeline obtains",
      "non-Euclidean results restated", "CORRECT", "BANKED",
      **_bank(["tsplib_nonEuc_total_gart_2_0_mape_pct",
               "tsplib_nonEuc_total_fixed_alpha_1_136_mape_pct"])),
    C("concl.linenoise", "Conclusion", 486, "near-collinear class against",
      "Line Noise vs isotropic error", "STALE", "BANKED",
      **_bank(["2d_by_genclass_linenoise_gart_2_0_mape_pct",
               "2d_by_genclass_isotropic_gart_2_0_mape_pct"]),
      note="Says 11.59 where lines 364 and 428 say 11.60."),
    C("concl.slope_corr", "Conclusion", 486, "874 newly solved instances",
      "augmentation slope and correlation movement", "UNVERIFIABLE", "UNGENERATED",
      action="AUG-REPOINT plus the alpha-on-alpha exporter (aug.slope_corr)."),
    C("concl.withholding_035", "Conclusion", 486, "MAPE points",
      "cost of withholding a dimension or the large-n range", "STALE", "UNGENERATED",
      note="Inconsistent with the 0.28 pp at line 400 for the same experiment.",
      action="GEN-REPOINT."),
    C("concl.metric_shift_006", "Conclusion", 486, "less than 0.06 percentage points",
      "effect of removing the audited instances on every reported metric",
      "UNVERIFIABLE", "UNGENERATED", action="See prov.other_estimators."),

    # -------------------------------------------------------------- Appendices
    C("app.check_cell_count", "Appendix", 498, "re-derives all 1{,}528 table cells",
      "number of cells build_paper_tables --check verifies", "UNVERIFIABLE",
      "GENERATED", authority="paper_tooling/build_paper_tables.py --check",
      note="run_check() prints len(old) in its banner. The count moved with the "
           "resplice (tab:tsplib_nonEuc gained an N column, every results table "
           "gained a 'GART 2.0 (V3 features)' row), so it is stale but trivially "
           "recoverable. A self-describing count typed by hand into the document "
           "it describes is the same defect class in miniature.",
      action="Have run_check() write tables/check_manifest.json "
             "{n_cells, n_moved, n_missing, generated_at} and splice n_cells."),
    C("appx.out_of_interval_rows", "Appendix", 499, "out-of-interval rows",
      "the two named corpus rows with raw alpha outside [1,2]", "CORRECT",
      "UNGENERATED",
      note="Model-independent; the instance names and raw alphas were read off the "
           "corpus once by hand.",
      action="Corpus-statistics exporter: bank the out-of-interval row names and "
             "their raw alpha."),
    C("appx.feature_split", "Appendix", 566, "MST-derived features used by GART",
      "19 MST + 11 geometric feature tables", "STALE", "STRUCTURAL",
      authority=SIDECAR, keys=["features_in_booster_order"], drift=True,
      note="greedy_nn_over_mst appears in neither appendix table.",
      action="See meth.feature_split_19."),
    C("appx.shap_count", "Appendix", 616, "ranks all",
      "tab:shap_top covers all features", "STALE", "STRUCTURAL",
      authority=SIDECAR, keys=["n_features"], drift=True,
      note="30 rows for a 31-feature model.", action="See meth.shap_shares."),
    C("appx.shap_shares", "Appendix", 618, "dominance ratio contributes",
      "SHAP family shares", "UNVERIFIABLE", "UNGENERATED",
      note="Duplicate of meth.shap_shares, same numbers quoted twice.",
      action="See meth.shap_shares."),
    C("appx.feature_set_620", "Appendix", 620, "set described in Section",
      "feature count", "STALE", "STRUCTURAL", authority=SIDECAR, keys=["n_features"],
      drift=True,
      note="Feature count in the feature-selection appendix, closing the loop back to Section 3.2.",
      action="See meth.feature_count_166."),
    C("appx.nd_pair_sample", "Appendix", 830, "enumerated exhaustively, so the statistic is exact",
      "ND pair-sampling design", "CORRECT", "STRUCTURAL", authority=NUMBER_BANK,
      keys=["rank_nd_gart_2_0_pair_mode", "rank_nd_gart_2_0_pair_seed",
            "rank_nd_gart_2_0_close5_pairs"], drift=True,
      note="FLAGGED. The audit marked this CORRECT because the sample size is "
           "model-independent -- true, but the sampler no longer exists. The bank "
           "records pair_mode='exhaustive' and pair_seed='n/a'. A methods sentence "
           "describing a retired procedure is worse than a stale number, because "
           "no value check will ever fire on it."),
    C("appx.geometric_11", "Appendix", 1271, "lists the",
      "11 geometric and centroid features", "CORRECT", "STRUCTURAL",
      authority=SIDECAR, keys=["features_in_booster_order"],
      note="Correct for that table in isolation; only the document-level total is "
           "wrong."),
    C("appx.seven_explicit", "Appendix", 1368, "on these seven cases",
      "GART MAPE on the EXPLICIT non-EUC_2D cases", "STALE", "BANKED",
      **_bank(["tsplib_nonEuc_explicit_gart_2_0_mape_pct",
               "tsplib_nonEuc_explicit_gart_2_0_n"]),
      note="TWO defects, and the second is the dangerous one. The value is stale "
           "(3.88 is the predecessor's), and the DENOMINATOR is wrong: the bank's "
           "N for that bucket is 6, not 7, because GART declines si1032. The "
           "manuscript reuses the taxonomy count (seven EXPLICIT instances, "
           "app.tsplib_taxonomy) as a scoring count. Whenever a model can decline "
           "an instance, a set size must be read from coverage, never from the "
           "taxonomy."),
]
