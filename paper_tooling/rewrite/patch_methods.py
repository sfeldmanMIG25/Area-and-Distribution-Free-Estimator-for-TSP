"""Methods-section prose corrections for Area_Free_Main.tex.

Scope: Section 3 only -- the feature description (subsec:features), the
model-training paragraph (subsec:model_training), the complexity paragraph,
and the SHAP paragraph (subsec:shap). Prose only; no table bodies are touched.

Every replacement is a single-line substring edit, so the file's CRLF endings
are never crossed. Anchors were quoted verbatim from the file at
sha256-of-record time; a missing anchor is reported, never guessed at.

Sources for every number changed:
  lgbm_model_v3/gart2_final.json ................. n_features, num_trees,
                                                   target_transform,
                                                   monotone_constraints,
                                                   early_stopping_metric,
                                                   hyperparameter_provenance
  lgbm_model_v3/gart2_final.joblib (booster dump) . mean leaf depth 11.8286,
                                                   max leaf depth 40,
                                                   avg leaves 147.8005,
                                                   K*Dbar = 1.322e4
  paper_tooling/v4_study_gart2_probe.csv ......... monotonicity probe,
                                                   rows GART2_logit_v3hp_mono
                                                   and GART2_logit_v3hp
  paper_tooling/tables/shap_numbers.json ......... all SHAP shares, sample rows
  paper_tooling/tables/shap_ranking.csv .......... top-ten composition
  lgbm_model_v3/feature_engineering_gart2.py ..... TRAIN_GREEDY_RANGE,
                                                   DENSE_CAP, CAND_K
  tsplib_benchmark/run_all_models_tsplib.py ...... where the coverage gate fires
  paper_tooling/v4_study_noneuc_greedy_audit.csv . si1032 is the only decline

Usage:  python paper_tooling/rewrite/patch_methods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

TEX = Path(__file__).resolve().parents[2] / "paper_reference" / "Area_Free_Main.tex"


# (label, old_string, new_string)
EDITS: list[tuple[str, str, str]] = [

    # ---------------------------------------------------------------- 166 --
    # Feature count 30 -> 31 (gart2_final.json :: n_features).
    (
        "L166 feature count in the section preamble",
        "The following subsections describe its 30 input features, training data, "
        "model fitting, computational cost, and feature-importance analysis.",
        "The following subsections describe its 31 input features, training data, "
        "model fitting, computational cost, and feature-importance analysis.",
    ),

    # ---------------------------------------------------------------- 172 --
    # Feature count 30 -> 31, and the "three aspects" enumeration gains the
    # fourth: greedy_nn_over_mst is neither spread, nor an edge moment, nor
    # topology.
    (
        "L172 feature count and the three-aspect enumeration",
        "we extract 30 features from each TSP instance. The feature count reflects "
        "the need to characterize three distinct aspects of instance structure---"
        "spatial spread, edge-weight distribution, and branching topology---each "
        "requiring multiple statistics to capture the range of behaviors observed "
        "across distributions.",
        "we extract 31 features from each TSP instance. The feature count reflects "
        "the need to characterize four distinct aspects of instance structure---"
        "spatial spread, edge-weight distribution, branching topology, and the gap "
        "between a constructed tour and the MST---the first three requiring multiple "
        "statistics to capture the range of behaviors observed across distributions, "
        "the fourth a single ratio.",
    ),

    # ---------------------------------------------------------------- 176 --
    # Three groups -> four, with the per-group counts stated so the arithmetic
    # 11 + 10 + 9 + 1 = 31 is checkable from the prose.
    (
        "L176 group overview",
        "The features fall into three groups. Geometric and centroid descriptors "
        "quantify spatial spread and density. MST edge distribution statistics "
        "capture local uniformity through edge weight moments. Topological structure "
        "and clustering proxies detect higher-order patterns such as clusters and "
        "filaments.",
        "The features fall into four groups. Eleven geometric and centroid "
        "descriptors quantify spatial spread and density. Ten MST edge distribution "
        "statistics capture local uniformity through edge weight moments. Nine "
        "topological structure and clustering proxies detect higher-order patterns "
        "such as clusters and filaments. One constructive ratio, "
        "\\texttt{greedy\\_nn\\_over\\_mst}, compares a greedy nearest-neighbour tour "
        "against the MST.",
    ),

    # ---------------------------------------------------------------- 188 --
    # "The remaining 19" presupposed 11 + 19 = 30. With 31 inputs the MST block
    # is no longer the remainder, so it is named absolutely.
    (
        "L188 opening: 'the remaining 19' presupposes a 30-feature total",
        "The remaining 19 features summarize the MST. Ten describe",
        "Nineteen features summarize the MST. Ten describe",
    ),

    # 188 tail: introduce greedy_nn_over_mst, its bound property, and the
    # coverage gate it drives. Range [1.035, 2.209] is TRAIN_GREEDY_RANGE in
    # lgbm_model_v3/feature_engineering_gart2.py; the gate fires on the
    # non-Euclidean path in tsplib_benchmark/run_all_models_tsplib.py.
    (
        "L188 tail: the 31st feature and the coverage gate",
        "Nine topological features describe dominance, gaps, leaves, degrees, large "
        "edges, and the MST diameter. Definitions and computational costs are "
        "reported in Appendix~\\ref{app:features}.",
        "Nine topological features describe dominance, gaps, leaves, degrees, large "
        "edges, and the MST diameter. The thirty-first feature, "
        "\\texttt{greedy\\_nn\\_over\\_mst}, is the length of an exact greedy "
        "nearest-neighbour tour divided by $L_{\\mathrm{MST}}$. It is the only input "
        "GART 2.0 adds to the predecessor's set, and it is a constructed upper bound "
        "on the target: a greedy tour is a feasible tour, so its length is at least "
        "$L_{\\mathrm{TSP}}$ and the ratio is at least $\\alpha$. The feature also "
        "gates coverage. On the non-Euclidean path of "
        "Section~\\ref{sec:application}, where the metric is given but coordinates "
        "are not, the estimator returns a status row rather than a prediction for "
        "any instance whose ratio falls outside $[1.035,2.209]$, the range spanned by "
        "the training corpus. Together with the eleven geometric and centroid "
        "descriptors these make up the 31-feature vector. Definitions and "
        "computational costs are reported in Appendix~\\ref{app:features}.",
    ),

    # ---------------------------------------------------------------- 212 --
    # The refit same-feature controls consume 31 features
    # (paper_tooling/controls_31f/marginals.csv :: NN_31F, Linear_31F).
    (
        "L212 same-feature control vector width",
        "Section~\\ref{subsec:results} isolates the contribution of boosting itself "
        "by running a linear model and a feed-forward network on the identical "
        "30-feature vector.",
        "Section~\\ref{subsec:results} isolates the contribution of boosting itself "
        "by running a linear model and a feed-forward network on the identical "
        "31-feature vector.",
    ),

    # ---------------------------------------------------------------- 214 --
    # Five independent corrections plus the monotone constraints, which the
    # manuscript never mentioned:
    #   (a) 100 Optuna TPE trials minimizing validation SDPE -> not tuned
    #   (b) validation RMSE early stopping -> cost_mape, patience 100
    #   (c) 2,031 trees -> 1,118
    #   (d) "148 leaves per tree" quoted the num_leaves CAP as the realisation;
    #       both are now stated (cap 148, realised mean 147.80). The manifest
    #       anchors `contains {v} trees with {~} leaves per tree` are preserved
    #       verbatim so methods.model.n_trees and methods.model.leaves_per_tree
    #       keep resolving.
    #   (e) monotone constraints described, with the probe reported as an
    #       artifact-integrity check and the unconstrained control beside it
    (
        "L214 tuning protocol, tree counts, monotone constraints, probe",
        "Using seed 42, 100 Optuna TPE trials \\citep{akiba2019optuna} fit on the "
        "training split and minimized validation SDPE; validation RMSE supplied early "
        "stopping. The selected configuration was then fit on the training split "
        "only, again early-stopped on validation, with no train--validation refit. "
        "The test split was not used by the tuning objective and was evaluated after "
        "this final fit. The resulting ensemble contains 2{,}031 trees with 148 "
        "leaves per tree; its configuration is reported in "
        "Appendix~\\ref{app:hyperparams}.",

        "The hyperparameters are the predecessor's, frozen; no search was run for "
        "this model, so Section~\\ref{sec:evaluation} compares a tuned field against "
        "an untuned entrant. We ran the search and rejected it: Optuna TPE "
        "\\citep{akiba2019optuna} against in-distribution validation buys "
        "multidimensional accuracy and pays for it on TSPLIB, and 200 trials bought "
        "0.011 percentage points of multidimensional MAPE while costing 0.16 points "
        "of TSPLIB MAPE. Early stopping used cost-level MAPE on the validation split "
        "with a patience of 100 rounds. The model was fit at seed 42 on the training "
        "split only, with no train--validation refit, and the test split was scored "
        "once after that fit. Two inputs carry non-increasing monotone constraints, "
        "$n$ and $d$, encoding that $\\alpha$ falls towards 1 as an instance grows or "
        "the ambient dimension rises. Those constraints are enforced inside the tree "
        "builder, so a sum of such trees cannot increase along a ceteris-paribus "
        "sweep of either variable; a probe over 1{,}000 held-out instances "
        "accordingly returns 100\\% non-increasing sweeps on both axes with zero "
        "violations. That figure verifies the constraint was applied and is not "
        "evidence about the model's inductive bias. The informative control is the "
        "same fit with the constraints removed, which holds monotonicity on 8.4\\% of "
        "the dimension sweeps and 19.0\\% of the size sweeps. The resulting ensemble "
        "contains 1{,}118 trees with 148 leaves per tree as the growth cap and "
        "147.80 realized on average; its configuration is reported in "
        "Appendix~\\ref{app:hyperparams}.",
    ),

    # ---------------------------------------------------------------- 219 --
    # Booster geometry, recomputed from gart2_final.joblib.
    (
        "L219 traversal count and depth",
        "The trained model performs $K=2{,}031$ traversals with empirical mean "
        "root-to-leaf depth $\\bar D\\approx18.9$ (maximum 33). Because the trees are "
        "unbalanced, traversal cost is better represented by $\\bar D$ than by "
        "$\\lceil\\log_2 148\\rceil$; the ensemble therefore performs approximately "
        "$K\\bar D=3.8\\times10^4$ comparisons per prediction, independent of $n$.",

        "The trained model performs $K=1{,}118$ traversals with empirical mean "
        "root-to-leaf depth $\\bar D\\approx11.83$ (maximum 40). Because the trees are "
        "unbalanced, traversal cost is better represented by $\\bar D$ than by "
        "$\\lceil\\log_2 148\\rceil$; the ensemble therefore performs approximately "
        "$K\\bar D=1.3\\times10^4$ comparisons per prediction, independent of $n$.",
    ),

    # 219 tail: the 2D sentence implied an O(n log n) extraction. The greedy
    # tour does not take the Delaunay path -- feature_engineering_gart2.py
    # dispatches to a dense O(n^2) scan at DENSE_CAP = 3000 and to CAND_K = 16
    # neighbour lists above it -- so the leading term below n = 3000 is the
    # greedy tour, not the MST.
    (
        "L219 tail: greedy-tour extraction cost",
        "In 2D, constructing the Euclidean MST from Delaunay edges costs "
        "$O(n\\log n)$ \\citep{delaunay1934}; sorting MST edges adds $O(n\\log n)$, "
        "and the remaining topological statistics cost $O(n)$.",

        "In 2D, constructing the Euclidean MST from Delaunay edges costs "
        "$O(n\\log n)$ \\citep{delaunay1934}; sorting MST edges adds $O(n\\log n)$, "
        "and the remaining topological statistics cost $O(n)$. The greedy "
        "nearest-neighbour tour behind \\texttt{greedy\\_nn\\_over\\_mst} does not "
        "take that path: it scans a dense distance matrix in $O(n^2)$ for "
        "$n\\le3000$ and switches to precomputed 16-neighbour candidate lists with an "
        "exact nearest-unvisited fallback above that threshold. Below $n=3000$ the "
        "greedy tour, not the MST, is therefore the leading term of 2D extraction.",
    ),

    # ---------------------------------------------------------------- 223 --
    # Every share here was the predecessor's. Re-run against the production
    # booster (paper_tooling/shap_production.py -> shap_numbers.json).
    (
        "L223 SHAP paragraph",
        "We assess the 30-feature set with mean absolute SHAP magnitudes "
        "\\citep{lundberg2017shap} on a 5{,}000-row held-out sample. The MST "
        "dominance ratio contributes 26.4\\% of the total magnitude, and six of the "
        "top ten features are MST-derived. Dimension and node count jointly "
        "contribute 22.2\\%; centroid-distance descriptors contribute 10.0\\%, and the "
        "two bounding-hypervolume features contribute approximately 4.6\\%. "
        "Appendix~\\ref{app:shap} reports the complete ranking.",

        "We assess the 31-feature set with mean absolute SHAP magnitudes "
        "\\citep{lundberg2017shap} against the production booster, on a 5{,}000-row "
        "sample of the 16{,}920-row held-out test split. The MST dominance ratio "
        "contributes 26.5\\% of the total magnitude and "
        "\\texttt{greedy\\_nn\\_over\\_mst} 23.9\\%; those two features carry 50.4\\% "
        "of it between them. Four of the top ten features are MST-derived, five are "
        "geometric, and the greedy ratio is the remaining one. Node count and "
        "dimension "
        "jointly contribute 14.1\\%, the two bounding-hypervolume features 7.2\\%, and "
        "the four centroid-distance descriptors 3.0\\%. The greedy ratio ranking "
        "second on its first appearance is the case for adding it; it is also the "
        "input that lets the estimator decline an instance, since the coverage gate "
        "of Section~\\ref{sec:application} is a range test on this feature and no "
        "other. Appendix~\\ref{app:shap} reports the complete ranking.",
    ),
]


def main() -> int:
    raw = TEX.read_bytes()
    text = raw.decode("utf-8")

    applied: list[str] = []
    missing: list[str] = []
    ambiguous: list[str] = []

    for label, old, new in EDITS:
        count = text.count(old)
        if count == 0:
            missing.append(label)
            continue
        if count > 1:
            ambiguous.append(f"{label} (matches {count}x)")
            continue
        text = text.replace(old, new, 1)
        applied.append(label)

    if missing or ambiguous:
        print("patch_methods: NOT WRITTEN -- anchor problems")
        for label in missing:
            print(f"  MISSING   {label}")
        for label in ambiguous:
            print(f"  AMBIGUOUS {label}")
        print(f"applied 0 / {len(EDITS)}")
        return 1

    TEX.write_bytes(text.encode("utf-8"))
    print(f"patch_methods: applied {len(applied)} / {len(EDITS)}, missing 0")
    for label in applied:
        print(f"  OK  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
