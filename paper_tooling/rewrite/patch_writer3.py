r"""Single-writer pass 3: Appendix app:features and Appendix app:shap.

WHY THIS TOUCHES TABLES
-----------------------
Same standing as patch_writer2.py.  ``build_paper_tables.TEX_TABLES`` verifies
six labels and neither ``tab:shap_top`` nor any feature-specification table is
among them; ``--check`` reports 1,910 cells before and after, unchanged.  Both
appendices are hand-authored, both are the predecessor's, and both are now
contradicted by the corrected Section 3 prose:

  * Section 3.2 says the thirty-first feature is ``greedy_nn_over_mst`` and
    that "Definitions and computational costs are reported in
    Appendix~\ref{app:features}".  No feature table defined it, so that
    sentence pointed at nothing.  Appendix app:features listed 19 + 11 = 30.
  * Section 3.6 says the SHAP analysis covers 31 features, that the greedy
    ratio ranks second at 23.9%, and that four of the top ten are MST-derived.
    Appendix app:shap said 30 features, no greedy ratio, and six of the top
    ten MST-derived, with every share the predecessor's.

The 31 ranking rows are not transcribed by hand.  They are rebuilt at patch
time from ``paper_tooling/tables/shap_ranking.csv`` (produced by
``paper_tooling/shap_production.py`` against the production booster), so the
printed table is the artifact by construction.  The bar rule keeps the
existing convention of one millimetre per percentage point.

Aggregate figures below come from ``paper_tooling/tables/shap_numbers.json``:
  shap_n_features 31, shap_sample_rows 5000, shap_split_rows 16920
  mst_dominance_ratio 26.520597 (rank 1), greedy_nn_over_mst 23.855249 (rank 2)
  size_dimension 14.132265, bounding_hypervolume 7.158363, centroid 3.002367
  group mst 50.654571 on 19 features, geometric 25.490180 on 11, greedy
  23.855249 on 1
  top-ten composition 4 MST-derived / 5 geometric / 1 greedy

The greedy feature's definition and cost come from
``lgbm_model_v3/feature_engineering_gart2.py``: an exact greedy
nearest-neighbour tour, dense O(n^2) scan for n <= DENSE_CAP = 3000 and
CAND_K = 16 neighbour candidate lists with an exact nearest-unvisited fallback
above it, divided by L_MST; TRAIN_GREEDY_RANGE = (1.035, 2.209) drives the
coverage gate.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper_reference" / "Area_Free_Main.tex"
RANKING = ROOT / "paper_tooling" / "tables" / "shap_ranking.csv"


def build_shap_rows() -> str:
    """Rebuild the 31 body rows of tab:shap_top from the generated ranking."""
    out: list[str] = []
    with RANKING.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rank = int(row["rank"])
            feat = row["feature"].replace("_", r"\_")
            mean_abs = float(row["mean_abs_shap"])
            share = float(row["share_pct"])
            bar = max(share, 0.20)  # keep the smallest bars visible, as before
            out.append(
                f"{rank:<2d} & \\texttt{{{feat}}} & {mean_abs:.4f} & {share:5.2f} & "
                f"\\textcolor{{blue!70!black}}{{\\rule{{{bar:.2f}mm}}{{4pt}}}} \\\\"
            )
    return "\r\n".join(out)


OLD_SHAP_ROWS = "\r\n".join([
    r"1  & \texttt{mst\_dominance\_ratio}      & 0.0388 & 26.43 & \textcolor{blue!70!black}{\rule{26.4mm}{4pt}} \\",
    r"2  & \texttt{dimension}                  & 0.0171 & 11.66 & \textcolor{blue!70!black}{\rule{11.7mm}{4pt}} \\",
    r"3  & \texttt{n\_customers}               & 0.0155 & 10.55 & \textcolor{blue!70!black}{\rule{10.6mm}{4pt}} \\",
    r"4  & \texttt{centroid\_dist\_std}        & 0.0071 &  4.81 & \textcolor{blue!70!black}{\rule{4.8mm}{4pt}} \\",
    r"5  & \texttt{mst\_degree\_mean}          & 0.0068 &  4.62 & \textcolor{blue!70!black}{\rule{4.6mm}{4pt}} \\",
    r"6  & \texttt{mst\_diameter}              & 0.0063 &  4.30 & \textcolor{blue!70!black}{\rule{4.3mm}{4pt}} \\",
    r"7  & \texttt{mst\_diameter\_normalized}  & 0.0061 &  4.18 & \textcolor{blue!70!black}{\rule{4.2mm}{4pt}} \\",
    r"8  & \texttt{mst\_gap\_ratio}            & 0.0050 &  3.39 & \textcolor{blue!70!black}{\rule{3.4mm}{4pt}} \\",
    r"9  & \texttt{large\_edge\_count}         & 0.0044 &  2.98 & \textcolor{blue!70!black}{\rule{3.0mm}{4pt}} \\",
    r"10 & \texttt{bounding\_hypervolume}      & 0.0040 &  2.70 & \textcolor{blue!70!black}{\rule{2.7mm}{4pt}} \\",
    r"11 & \texttt{mst\_leaf\_ratio}           & 0.0039 &  2.66 & \textcolor{blue!70!black}{\rule{2.7mm}{4pt}} \\",
    r"12 & \texttt{mst\_degree\_max}           & 0.0034 &  2.32 & \textcolor{blue!70!black}{\rule{2.3mm}{4pt}} \\",
    r"13 & \texttt{centroid\_dist\_mean}       & 0.0030 &  2.05 & \textcolor{blue!70!black}{\rule{2.1mm}{4pt}} \\",
    r"14 & \texttt{log\_bounding\_hypervolume} & 0.0027 &  1.85 & \textcolor{blue!70!black}{\rule{1.9mm}{4pt}} \\",
    r"15 & \texttt{centroid\_dist\_max}        & 0.0024 &  1.61 & \textcolor{blue!70!black}{\rule{1.6mm}{4pt}} \\",
    r"16 & \texttt{centroid\_dist\_iqr}        & 0.0022 &  1.51 & \textcolor{blue!70!black}{\rule{1.5mm}{4pt}} \\",
    r"17 & \texttt{mst\_edge\_max}             & 0.0020 &  1.35 & \textcolor{blue!70!black}{\rule{1.4mm}{4pt}} \\",
    r"18 & \texttt{mst\_edge\_q75}             & 0.0017 &  1.19 & \textcolor{blue!70!black}{\rule{1.2mm}{4pt}} \\",
    r"19 & \texttt{aspect\_ratio}              & 0.0017 &  1.13 & \textcolor{blue!70!black}{\rule{1.1mm}{4pt}} \\",
    r"20 & \texttt{mst\_degree\_std}           & 0.0016 &  1.11 & \textcolor{blue!70!black}{\rule{1.1mm}{4pt}} \\",
    r"21 & \texttt{log\_node\_density}         & 0.0015 &  1.01 & \textcolor{blue!70!black}{\rule{1.0mm}{4pt}} \\",
    r"22 & \texttt{mst\_edge\_q90}             & 0.0014 &  0.97 & \textcolor{blue!70!black}{\rule{0.97mm}{4pt}} \\",
    r"23 & \texttt{mst\_edge\_q25}             & 0.0014 &  0.93 & \textcolor{blue!70!black}{\rule{0.93mm}{4pt}} \\",
    r"24 & \texttt{mst\_edge\_mean}            & 0.0012 &  0.81 & \textcolor{blue!70!black}{\rule{0.81mm}{4pt}} \\",
    r"25 & \texttt{node\_density}              & 0.0011 &  0.76 & \textcolor{blue!70!black}{\rule{0.76mm}{4pt}} \\",
    r"26 & \texttt{mst\_edge\_skew}            & 0.0011 &  0.72 & \textcolor{blue!70!black}{\rule{0.72mm}{4pt}} \\",
    r"27 & \texttt{mst\_edge\_q10}             & 0.0010 &  0.70 & \textcolor{blue!70!black}{\rule{0.70mm}{4pt}} \\",
    r"28 & \texttt{mst\_edge\_q50}             & 0.0010 &  0.68 & \textcolor{blue!70!black}{\rule{0.68mm}{4pt}} \\",
    r"29 & \texttt{mst\_edge\_kurtosis}        & 0.0009 &  0.59 & \textcolor{blue!70!black}{\rule{0.59mm}{4pt}} \\",
    r"30 & \texttt{mst\_edge\_std}             & 0.0006 &  0.41 & \textcolor{blue!70!black}{\rule{0.41mm}{4pt}} \\",
])


GREEDY_TABLE = "\r\n".join([
    r"\subsection{Constructive Ratio} \label{app:features_greedy}",
    r"",
    r"The fourth category holds a single feature. It is the only input GART 2.0 adds "
    r"to the predecessor's set, it ranks second by SHAP magnitude "
    r"(Appendix~\ref{app:shap}), and it is the feature the non-Euclidean coverage "
    r"gate of Section~\ref{sec:application} tests.",
    r"",
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Constructive Feature Specification (1 feature).}",
    r"\label{tab:greedy_features}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}lp{6.5cm}ll@{}}",
    r"\toprule",
    r"\textbf{Feature Name} & \textbf{Description \& Rationale} & \textbf{Complexity} & \textbf{Origin / Citation} \\ \midrule",
    r"\textbf{Greedy-to-MST Ratio} & Length of an exact greedy nearest-neighbour tour "
    r"divided by $L_{\mathrm{MST}}$. A greedy tour is feasible, so the ratio is a "
    r"constructed upper bound on the target $\alpha$. Values outside the training "
    r"range $[1.035,2.209]$ trigger the coverage gate. & $O(n^2 d)$ dense for "
    r"$n\le3000$; $O(nkd)$ with $k=16$ candidate neighbours above it & "
    r"\textbf{Proposed}; greedy construction of \citet{johnson1996asymptotic} \\ \bottomrule",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
    r"",
])


def edits() -> list[tuple[str, str, str]]:
    return [
        # ------------------------------------------------------ app:features
        (
            "S1 app:features lead-in: 19 + 11 = 30 omits the greedy ratio",
            r"Tables~\ref{tab:edge_features} and~\ref{tab:topo_features} list the 19 MST-derived features used by GART 2.0, complementing the 11 geometric and centroid features of Table~\ref{tab:geo_features} (\S\ref{subsec:features}).",
            r"Tables~\ref{tab:edge_features} and~\ref{tab:topo_features} list the 19 MST-derived features used by GART 2.0, complementing the 11 geometric and centroid features of Table~\ref{tab:geo_features} and the single constructive ratio of Table~\ref{tab:greedy_features}, 31 in all (\S\ref{subsec:features}).",
        ),
        (
            "S2 app:features: define the 31st feature",
            "\\end{tabular}%\r\n}\r\n\\end{table}\r\n\r\n"
            "\\section{Feature Importance: SHAP Values} \\label{app:shap}",
            "\\end{tabular}%\r\n}\r\n\\end{table}\r\n\r\n"
            + GREEDY_TABLE
            + "\r\n\\section{Feature Importance: SHAP Values} \\label{app:shap}",
        ),
        (
            "S3 tab:geo_features cross-reference",
            r"Table~\ref{tab:geo_features} lists the 11 geometric and centroid features that complement the MST-derived features in Tables~\ref{tab:edge_features}, \ref{tab:topo_features}.",
            r"Table~\ref{tab:geo_features} lists the 11 geometric and centroid features that complement the MST-derived features in Tables~\ref{tab:edge_features}, \ref{tab:topo_features} and the constructive ratio of Table~\ref{tab:greedy_features}.",
        ),

        # ---------------------------------------------------------- app:shap
        (
            "S4 app:shap lead-in",
            r"To validate the 30-feature set, we report the SHAP values \citep{lundberg2017shap} on the held-out test split using the saved booster (see \texttt{shap\_analyzer\_v3.py}). Table~\ref{tab:shap_top} ranks all 30 features by mean absolute SHAP contribution. Three observations motivate the final feature set.",
            r"To validate the 31-feature set, we report the SHAP values \citep{lundberg2017shap} against the production booster on a 5{,}000-row sample of the 16{,}920-row held-out test split (\texttt{paper\_tooling/shap\_production.py}). Table~\ref{tab:shap_top} ranks all 31 features by mean absolute SHAP contribution. Three observations motivate the final feature set.",
        ),
        (
            "S5 app:shap aggregate shares",
            r"The MST dominance ratio contributes 26.4\% of total mean absolute SHAP magnitude, and six of the top ten features are MST-derived. Dimension and size jointly contribute 22.2\%; centroid-distance descriptors contribute 10.0\%, and the two bounding-hypervolume descriptors contribute approximately 4.6\%. SHAP magnitude describes model reliance, not a causal effect or proof that a feature group improves generalization.",
            r"The MST dominance ratio contributes 26.5\% of total mean absolute SHAP magnitude and \texttt{greedy\_nn\_over\_mst} 23.9\%; those two features carry 50.4\% of it between them. Four of the top ten features are MST-derived, five are geometric, and the greedy ratio is the remaining one. Node count and dimension jointly contribute 14.1\%, the two bounding-hypervolume descriptors 7.2\%, and the four centroid-distance descriptors 3.0\%. By group, the nineteen MST-derived features carry 50.7\%, the eleven geometric features 25.5\%, and the single constructive ratio 23.9\%. SHAP magnitude describes model reliance, not a causal effect or proof that a feature group improves generalization.",
        ),
        (
            "S6 app:shap closing caveat",
            r"We used SHAP analysis as a diagnostic during feature reduction, yielding the 30-feature set described in Section~\ref{subsec:features}; because this selection used validation feedback, it should not be interpreted as an independent confirmatory test.",
            r"We used SHAP analysis as a diagnostic during feature reduction, yielding the 31-feature set described in Section~\ref{subsec:features}; because this selection used validation feedback, it should not be interpreted as an independent confirmatory test.",
        ),
        (
            "S7 tab:shap_top caption",
            r"\caption{All 30 GART 2.0 features ranked by mean absolute SHAP magnitude (5{,}000 held-out rows, seed 42). The bars show each feature's share of the total magnitude.}",
            r"\caption{All 31 GART 2.0 features ranked by mean absolute SHAP magnitude, computed against the production booster on a 5{,}000-row sample of the 16{,}920-row held-out test split. The bars show each feature's share of the total magnitude.}",
        ),
        (
            "S8 tab:shap_top body, rebuilt from shap_ranking.csv",
            OLD_SHAP_ROWS,
            build_shap_rows(),
        ),
    ]


def main() -> int:
    text = TEX.read_bytes().decode("utf-8")

    applied, missing, ambiguous = [], [], []
    for label, old, new in edits():
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
        print("patch_writer3: ABORTED, no bytes written")
    else:
        TEX.write_bytes(text.encode("utf-8"))

    print(f"patch_writer3: applied {len(applied)} / {len(edits())}, "
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
