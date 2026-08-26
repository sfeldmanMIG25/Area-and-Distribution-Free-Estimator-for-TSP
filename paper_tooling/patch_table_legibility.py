"""Replace \\resizebox shrink with wrapping columns on the four illegible tables.

Byte-level read/replace/write, same reason as patch_final_verification.py.

``\\resizebox{\\textwidth}{!}{...}`` scales a table to the text width whatever its
natural width is, so one wide unbreakable cell shrinks the whole table. Measured
on the compiled PDF against a 10pt body, four tables came out below \\scriptsize:

    Table 1  tab:benchmark_models   4.42pt   (a long Formula and a long Source column)
    Table 2  tab:dataset_counts     6.18pt   (footnote rows at 5.56pt)
    Table 7  tab:complexity         6.13pt   (a long "Dominant term" column)
    Table 13 tab:greedy_features    5.60pt   (one very long Complexity cell in an l column)

Table 13 is the clearest illustration: it sits directly under Table 12, which
carries the same column spec but no over-wide cell and renders at 9.15pt, so the
two tables on that page differ in size by two thirds.

The fix in every case is the same and removes the scale factor entirely: give
each column an explicit ``p`` width that sums to the text block, let the prose
wrap, and set the size with \\footnotesize so it is a declared size rather than
whatever the shrink happened to produce. Column widths are written as
``\\dimexpr F\\textwidth - Xpt`` where the subtracted amount covers the
\\tabcolsep gutters, so the fractions read as fractions of the text width.

Table 1 also becomes a longtable. It is the tallest of the four once its cells
wrap, and a longtable paginates with a repeated header instead of overflowing.

No number, no cell and no citation changes; this patch is column specs, sizes
and the table/longtable environment only.
"""
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

# -- Table 1: tab:benchmark_models -> longtable with wrapping columns ---------
T1_OLD = b"""\\begin{table}[!htbp]\r
\\centering\r
\\caption{Benchmark estimators. Every row is evaluated numerically. ``Domain'' states the range each source fits or derives; rows outside it are reported separately as extrapolations.}\r
\\label{tab:benchmark_models}\r
\\setlength{\\tabcolsep}{5pt}\r
\\renewcommand{\\arraystretch}{1.15}\r
\\resizebox{\\textwidth}{!}{%\r
\\begin{tabular}{@{}lllll@{}}\r
\\toprule\r
\\textbf{Estimator} & \\textbf{Formula} & \\textbf{Domain} & \\textbf{Cost} & \\textbf{Source} \\\\\r
\\midrule\r
"""

T1_NEW = b"""\\begingroup\r
\\setlength{\\tabcolsep}{4pt}\r
\\renewcommand{\\arraystretch}{1.25}\r
\\setlength{\\LTcapwidth}{\\textwidth}\r
\\begin{longtable}{@{}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.170\\textwidth-8pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.315\\textwidth-8pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.195\\textwidth-8pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.110\\textwidth-8pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.210\\textwidth-8pt\\relax}@{}}\r
\\caption{Benchmark estimators. Every row is evaluated numerically. ``Domain'' states the range each source fits or derives; rows outside it are reported separately as extrapolations.}\\label{tab:benchmark_models}\\\\\r
\\toprule\r
\\textbf{Estimator} & \\textbf{Formula} & \\textbf{Domain} & \\textbf{Cost} & \\textbf{Source} \\\\\r
\\midrule\r
\\endfirsthead\r
\\multicolumn{5}{@{}l}{\\footnotesize\\itshape Table~\\thetable{} continued from the previous page}\\\\\r
\\toprule\r
\\textbf{Estimator} & \\textbf{Formula} & \\textbf{Domain} & \\textbf{Cost} & \\textbf{Source} \\\\\r
\\midrule\r
\\endhead\r
"""

T1_TAIL_OLD = b"""Calibrated 1-tree & $c_k$ times either bound, $c_k$ fitted per budget on the training split & As trained & $\\Theta(k n^2 d)$ & This work \\\\\r
\\bottomrule\r
\\end{tabular}%\r
}\r
\\end{table}\r
"""

T1_TAIL_NEW = b"""Calibrated 1-tree & $c_k$ times either bound, $c_k$ fitted per budget on the training split & As trained & $\\Theta(k n^2 d)$ & This work \\\\\r
\\bottomrule\r
\\end{longtable}\r
\\endgroup\r
"""

# -- Table 2: tab:dataset_counts ---------------------------------------------
T2_OLD = b"""\\resizebox{\\textwidth}{!}{%\r
\\begin{tabular}{@{}llcp{6cm}@{}}\r
"""
T2_NEW = b"""\\setlength{\\tabcolsep}{4pt}\r
\\begin{tabular}{@{}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.275\\textwidth-8pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.335\\textwidth-8pt\\relax}\r
  >{\\footnotesize\\centering\\arraybackslash}p{\\dimexpr0.085\\textwidth-8pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.305\\textwidth-8pt\\relax}@{}}\r
"""

T2_TAIL_OLD = b"""\\multicolumn{4}{l}{\\footnotesize \\phantom{$^{\\dagger}$} \\texttt{Grid} is one of the three Geometric Struct.\\ generators; Table~\\ref{tab:genclass} reports it as its own row.}\r
\\end{tabular}%\r
}\r
\\end{table}\r
"""
T2_TAIL_NEW = b"""\\multicolumn{4}{@{}l}{\\footnotesize \\phantom{$^{\\dagger}$} \\texttt{Grid} is one of the three Geometric Struct.\\ generators; Table~\\ref{tab:genclass} reports it as its own row.}\r
\\end{tabular}\r
\\end{table}\r
"""

# -- Table 7: tab:complexity -------------------------------------------------
T7_OLD = b"""\\setlength{\\tabcolsep}{5pt}\r
\\renewcommand{\\arraystretch}{1.15}\r
\\resizebox{\\textwidth}{!}{%\r
\\begin{tabular}{@{}llll@{}}\r
"""
T7_NEW = b"""\\setlength{\\tabcolsep}{4pt}\r
\\renewcommand{\\arraystretch}{1.25}\r
\\begin{tabular}{@{}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.235\\textwidth-7pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.395\\textwidth-7pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.195\\textwidth-7pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.175\\textwidth-7pt\\relax}@{}}\r
"""

T7_TAIL_OLD = b"""Exact solver & $\\Theta(n^{2}2^{n})$ for the Held--Karp DP; branch-and-cut admits no polynomial bound & Yes, exact & No \\\\\r
\\bottomrule\r
\\end{tabular}%\r
}\r
\\end{table}\r
"""
T7_TAIL_NEW = b"""Exact solver & $\\Theta(n^{2}2^{n})$ for the Held--Karp DP; branch-and-cut admits no polynomial bound & Yes, exact & No \\\\\r
\\bottomrule\r
\\end{tabular}\r
\\end{table}\r
"""

# -- Table 13: tab:greedy_features -------------------------------------------
# The bare "resizebox + {@{}lp{6.5cm}ll@{}}" pair appears in all four feature
# tables, so the label is carried into the match to keep it unique. Only this
# one needs the change: the other three have no over-wide cell and already
# typeset at 9.1-9.5pt.
T13_OLD = b"""\\label{tab:greedy_features}\r
\\resizebox{\\textwidth}{!}{%\r
\\begin{tabular}{@{}lp{6.5cm}ll@{}}\r
"""
T13_NEW = b"""\\label{tab:greedy_features}\r
\\setlength{\\tabcolsep}{4pt}\r
\\begin{tabular}{@{}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.150\\textwidth-7pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.360\\textwidth-7pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.265\\textwidth-7pt\\relax}\r
  >{\\footnotesize\\raggedright\\arraybackslash}p{\\dimexpr0.225\\textwidth-7pt\\relax}@{}}\r
"""

T13_TAIL_OLD = b"""& \\textbf{Proposed}; greedy nearest-neighbour construction \\\\ \\bottomrule\r
\\end{tabular}%\r
}\r
\\end{table}\r
"""
T13_TAIL_NEW = b"""& \\textbf{Proposed}; greedy nearest-neighbour construction \\\\ \\bottomrule\r
\\end{tabular}\r
\\end{table}\r
"""

EDITS = [
    ("Table 1  head  -> longtable, wrapping columns", T1_OLD, T1_NEW),
    ("Table 1  tail  -> close longtable", T1_TAIL_OLD, T1_TAIL_NEW),
    ("Table 2  head  -> wrapping columns", T2_OLD, T2_NEW),
    ("Table 2  tail  -> drop resizebox brace", T2_TAIL_OLD, T2_TAIL_NEW),
    ("Table 7  head  -> wrapping columns", T7_OLD, T7_NEW),
    ("Table 7  tail  -> drop resizebox brace", T7_TAIL_OLD, T7_TAIL_NEW),
    ("Table 13 head  -> wrapping columns", T13_OLD, T13_NEW),
    ("Table 13 tail  -> drop resizebox brace", T13_TAIL_OLD, T13_TAIL_NEW),
]


def main() -> int:
    raw = TEX.read_bytes()
    before = len(raw)
    resize_before = raw.count(b"\\resizebox")
    for name, old, new in EDITS:
        hits = raw.count(old)
        if hits != 1:
            print(f"ABORT: {name}: expected exactly 1 match, found {hits}")
            return 1
        raw = raw.replace(old, new)
        print(f"  applied: {name}")

    if raw.count(b"\r\n") != raw.count(b"\r") or raw.count(b"\r\n") != raw.count(b"\n"):
        print("ABORT: line-ending damage")
        return 1
    if any(b > 0x7E for b in raw):
        print("ABORT: non-ASCII byte introduced")
        return 1
    resize_after = raw.count(b"\\resizebox")
    if resize_after != resize_before - 4:
        print(f"ABORT: resizebox count {resize_before} -> {resize_after}, expected -4")
        return 1

    TEX.write_bytes(raw)
    print(f"resizebox: {resize_before} -> {resize_after}")
    print(f"wrote {TEX} ({before} -> {len(raw)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
