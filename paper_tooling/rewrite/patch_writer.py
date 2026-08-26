r"""Single-writer pass: corrections no drafter owned, plus the contradictions
the four drafter patches created between them.

Runs LAST, after patch_methods, patch_results, patch_limits, patch_headline.
Every anchor is quoted from the manuscript as it stands after those four ran.
No table body belonging to build_paper_tables.py is touched: TEX_TABLES there
is {tab:nd_by_dim, tab:nd_by_size, tab:2d_by_size, tab:tsplib_by_size,
tab:tsplib_nonEuc, tab:rank} and none of them appears below.  tab:hyperparams
is hand-authored, is not in that set, and contributes no cell to the 1,910
machine-verified cells -- it is edited here because it otherwise contradicts
the corrected Section 3.4 prose outright.

W1  L127  paper outline still advertises Optuna hyperparameter tuning
W2  L244  same-feature baseline definition: 30-feature claim, now ambiguous
W3  L364  control paragraph omits the non-Euclidean set, where the network
          also wins
W4  L402  "the tuned hyperparameters" contradicts Section 3.4
W5  L438  discussion per-class MAPE list is the predecessor's, and the
          isotropic-to-LineNoise ratio with it
W6  L1297 appendix: "Optuna-tuned"
W7  L1301 appendix caption: Optuna TPE / 100 trials / validation-SDPE
W8  L1308 appendix table group heading: same
W9  L1331 appendix booster stats: 2,031 trees, 148.0 avg leaves, 18.89 avg
          depth, 33 max depth -- all the predecessor's

SOURCES
  lgbm_model_v3/gart2_final.json            num_trees 1118, n_features 31,
                                            hyperparameter_provenance,
                                            early_stopping_metric
  lgbm_model_v3/gart2_final.joblib          booster dump: avg leaves 147.8005,
                                            max 148, min 82, mean leaf depth
                                            11.8286, max leaf depth 40
  paper_tooling/tables/paper_numbers.json   2d_by_genclass_*_gart_2_0_mape_pct
                                            = 1.483810 isotropic, 1.819136
                                            biased, 2.184655 clustered,
                                            3.697179 geometric, 10.752207
                                            LineNoise; 10.752207 / 1.483810
                                            = 7.2464
  paper_tooling/controls_31f/marginals.csv  tsplib_noneuc GART_2.0 3.3441 /
                                            3.8931 on N=22, NN_31F 2.1370 /
                                            2.7330 on the same N=22
  paper_tooling/controls_31f/paired.csv     tsplib_noneuc GART_2.0 vs NN_31F
                                            mean_diff +1.2071,
                                            wilcoxon_p 0.050073
  paper_tooling/controls_31f/seed_band.csv  tsplib_noneuc
                                            mape_seeds_beating_gart 6 of 7
"""

from __future__ import annotations

from pathlib import Path

TEX = Path(__file__).resolve().parents[2] / "paper_reference" / "Area_Free_Main.tex"


EDITS: list[tuple[str, str, str]] = [

    # ------------------------------------------------------------------ W1
    # Section 3.4 no longer describes a tuning run for this model, so the
    # outline may not advertise one.  akiba2019optuna is still cited in 3.4
    # on the rejected search, so no bibliography entry is orphaned.
    (
        "W1 outline: hyperparameter tuning via Optuna",
        r"Section~\ref{sec:methodology} also covers model training with hyperparameter tuning via Optuna \citep{akiba2019optuna} and complexity analysis.",
        r"Section~\ref{sec:methodology} also covers model training, the frozen hyperparameters the model inherits from its predecessor, and complexity analysis.",
    ),

    # ------------------------------------------------------------------ W2
    # The rows labelled "same features" in the benchmark tables are NN_V3
    # (30 inputs) and Linear_V3 (28), not the production 31.  Renaming the
    # table rows would touch generated table bodies, so the definition states
    # the fact and points at the refitted controls instead.
    (
        "W2 same-feature baseline definition",
        r"\paragraph{Same-feature learned baselines.} A linear model and a feed-forward network consume the identical 30-feature vector and predict the same target. They isolate the contribution of gradient boosting from the contribution of the feature set.",
        r"\paragraph{Same-feature learned baselines.} A linear model and a feed-forward network predict the same target from the same feature block, isolating the contribution of gradient boosting from the contribution of the feature set. The rows carrying this label in the benchmark tables are fitted on the predecessor's 30-input vector rather than on GART 2.0's 31, and Section~\ref{subsec:results_2d} reports controls refitted on the production 31-feature vector; the distinction changes the verdict on the 2D benchmark.",
    ),

    # ------------------------------------------------------------------ W3
    # The control paragraph claimed the network's advantage is confined to
    # 2D.  It is not: the same network also leads on the screened
    # non-Euclidean set, on the same 22 instances GART 2.0 accepts.
    (
        "W3 control paragraph: non-Euclidean set omitted",
        r"The advantage does not carry to the other benchmarks: on the multidimensional test split GART 2.0 is ahead of the same network by 0.46 MAPE points and 1.03 SDPE points, and on TSPLIB EUC\_2D by 0.33 and 0.34 points. The model class therefore matters on all three benchmarks, and it favours the network on one of them.",
        r"The advantage does not carry to the other two benchmarks: on the multidimensional test split GART 2.0 is ahead of the same network by 0.46 MAPE points and 1.03 SDPE points, and on TSPLIB EUC\_2D by 0.33 and 0.34 points. It does reappear outside them. On the 22 screened non-Euclidean instances of Section~\ref{sec:application} the same network reaches 2.14\% MAPE and 2.73\% SDPE against GART 2.0's 3.34\% and 3.89\%, a paired difference of 1.21 MAPE points at $p=0.050$, with the network ahead at six of seven seeds. The model class therefore matters everywhere we measure it, and it favours the network on the 2D benchmark and on the non-Euclidean set.",
    ),

    # ------------------------------------------------------------------ W4
    # Section 3.4 now states the hyperparameters were frozen, not tuned.
    (
        "W4 generalization: 'the tuned hyperparameters'",
        r"We answer with three refits that hold the tuned hyperparameters and seed fixed and vary only the training data",
        r"We answer with three refits that hold the hyperparameters and seed fixed and vary only the training data",
    ),

    # ------------------------------------------------------------------ W5
    # Every figure in this list was the "GART 2.0 (V3 features)" row.
    # tab:genclass already prints the corrected values.
    (
        "W5 discussion per-class MAPE list",
        r"GART 2.0's MAPE is 1.84\% on the isotropic class, 2.00\% on the biased class, 2.60\% on the clustered class, 4.43\% on the geometric-structure class, and 11.60\% on Line Noise. Line Noise is the only class absent from training and it costs six times the isotropic error,",
        r"GART 2.0's MAPE is 1.48\% on the isotropic class, 1.82\% on the biased class, 2.18\% on the clustered class, 3.70\% on the geometric-structure class, and 10.75\% on Line Noise. Line Noise is the only class absent from training and it costs more than seven times the isotropic error,",
    ),

    # ------------------------------------------------------------------ W6
    (
        "W6 appendix hyperparams: 'Optuna-tuned' lead-in",
        r"Table~\ref{tab:hyperparams} reports the Optuna-tuned hyperparameters and the resulting booster statistics for the final GART 2.0 model.",
        r"Table~\ref{tab:hyperparams} reports the hyperparameters and the resulting booster statistics for the final GART 2.0 model. The hyperparameters are the predecessor's, frozen and not tuned for this model; Section~\ref{subsec:model_training} reports the search we ran against them and rejected.",
    ),

    # ------------------------------------------------------------------ W7
    (
        "W7 appendix hyperparams caption",
        r"\caption{GART 2.0 hyperparameters and trained-model statistics. Tuned block: Optuna TPE, 100 trials, validation-SDPE study objective; LightGBM uses squared-error regression.}",
        r"\caption{GART 2.0 hyperparameters and trained-model statistics. The hyperparameter block is inherited from the predecessor and frozen, not tuned for this model; LightGBM uses squared-error regression on the logit-transformed target, early-stopped on validation cost-level MAPE.}",
    ),

    # ------------------------------------------------------------------ W8
    (
        "W8 appendix hyperparams group heading",
        r"\multicolumn{3}{@{}l}{\textit{Tuned hyperparameters (Optuna TPE, 100 trials, validation-SDPE objective)}} \\",
        r"\multicolumn{3}{@{}l}{\textit{Hyperparameters (inherited from the predecessor, frozen, not tuned)}} \\",
    ),

    # ------------------------------------------------------------------ W9
    # Booster statistics, recomputed from gart2_final.joblib.  This is a
    # hand-authored appendix table, not one of the six generated tables.
    (
        "W9a appendix booster: tree count",
        r"  & trees (after early stop) & 2{,}031 \\",
        r"  & trees (after early stop) & 1{,}118 \\",
    ),
    (
        "W9b appendix booster: avg leaves per tree",
        r"  & avg leaves / tree        & 148.0 \\",
        r"  & avg leaves / tree        & 147.80 \\",
    ),
    (
        "W9c appendix booster: avg tree depth",
        r"  & avg tree depth           & 18.89 \\",
        r"  & avg tree depth           & 11.83 \\",
    ),
    (
        "W9d appendix booster: max tree depth",
        r"  & max tree depth           & 33 \\",
        r"  & max tree depth           & 40 \\",
    ),
]


def main() -> int:
    raw = TEX.read_bytes()
    text = raw.decode("utf-8")

    applied, missing, ambiguous = [], [], []
    for label, old, new in EDITS:
        count = text.count(old)
        if count == 0:
            missing.append(label)
            continue
        if count > 1:
            ambiguous.append((label, count))
            continue
        text = text.replace(old, new, 1)
        applied.append(label)

    if missing or ambiguous:
        print("patch_writer: ABORTED, no bytes written")
    else:
        TEX.write_bytes(text.encode("utf-8"))

    print(f"patch_writer: applied {len(applied)} / {len(EDITS)}, "
          f"missing {len(missing)}, ambiguous {len(ambiguous)}")
    for label in applied:
        print(f"  ok        {label}")
    for label in missing:
        print(f"  MISSING   {label}")
    for label, count in ambiguous:
        print(f"  AMBIGUOUS {label} ({count} occurrences)")
    return 1 if (missing or ambiguous) else 0


if __name__ == "__main__":
    raise SystemExit(main())
