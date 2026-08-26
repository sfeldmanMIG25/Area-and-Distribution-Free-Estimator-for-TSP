"""Emit the two Section 5 cost/accuracy ladders the cost bank now supports.

Tables 3 and 4 of the manuscript price the bound against GART 2.0 on TSPLIB
EUC\\_2D and on the multidimensional benchmark.  The other two benchmarks -- the
2D diverse set and the non-EUC\\_2D TSPLIB instances -- had an accuracy column
and no cost column until the timing session behind
``hk1tree_cost_frontier_bank.json``.  This writes their ladders in one shape:
budgets down, the two subgradient step rules across, so the step-rule comparison
the paper never made is readable off the same page as the cost.

Only the corpus total is printed.  The per-bucket ladders are in
``hk1tree_cost_frontier_2d.csv`` and ``hk1tree_cost_frontier_noneuc.csv``; the
two endpoints the prose quotes come from there.

Output goes to ``paper_tooling/tables/``.  These two tables are pasted into the
manuscript rather than spliced: their numbers come from the cost bank, not from
``build_paper_tables.py``, and their labels are deliberately absent from that
script's ``TEX_TABLES`` map so ``--check`` leaves them alone.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BANK = ROOT / "hk1tree_cost_frontier_bank.json"
OUT = ROOT / "tables"
BUDGETS = ("0", "10", "25", "50", "100", "200", "500")
ARMS = (("vj_ckpt", "Volgenant--Jonker"), ("polyak_ckpt", "Polyak"))


# Two decimals for time and for the cost multiple, three for MAPE: the same
# precisions Tables 3 and 4 already print, so the four ladders read alike.
def ms(x: float) -> str:
    return f"{x:.2f}"


def mult(x: float) -> str:
    # Three decimals below 0.1: the cheapest rungs on the non-Euclidean corpus
    # differ by a factor of five inside the first two decimals.
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def acc(x: float, better: bool) -> str:
    s = f"{x:.3f}"
    return rf"\textbf{{{s}}}" if better else s


def ladder(group: dict, caption: str, label: str, colsep: str) -> str:
    gart_ms, gart_mape = group["gart2_ms"], group["gart2_MAPE_pct"]
    rows = [rf"GART 2.0 & \multicolumn{{8}}{{c}}{{{ms(gart_ms)}~ms at "
            rf"$1.00\times$; {gart_mape:.3f}\% MAPE}} \\"]
    for k in BUDGETS:
        cells = []
        for arm, _ in ARMS:
            r = group["ascents"][arm][k]
            x = r["x_gart2_typical"]
            cells += [ms(r["ms"]), mult(x),
                      acc(r["raw_MAPE_pct"], x < 1 and r["raw_MAPE_pct"] < gart_mape),
                      acc(r["cal_MAPE_pct"], x < 1 and r["cal_MAPE_pct"] < gart_mape)]
        rows.append(f"$k={k}$ & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\setlength{{\tabcolsep}}{{{colsep}}}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Volgenant--Jonker step} & \multicolumn{4}{c}{Polyak step} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Row & ms & $\times$ & Raw & Cal. & ms & $\times$ & Raw & Cal. \\",
        r"\midrule",
        "\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def main() -> None:
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    two_d = bank["cells"]["2d"]["groups"]["Total (all 2D)"]
    (OUT / "frontier_2d_cost.tex").write_text(ladder(
        two_d,
        "2D diverse benchmark cost/accuracy ladder, all 2{,}580 instances, both "
        "subgradient step rules. Time is the median per-instance time in "
        "milliseconds on the solo protocol of Table~\\ref{tab:tsplib_by_size}; "
        "$\\times$ is that time over GART 2.0's on the same instances. Raw is the "
        "certified bound $w(\\pi_k)$; Cal.\\ is $c_k w(\\pi_k)$ at the same $c_k$ "
        "Table~\\ref{tab:frontier_tsplib} uses. Bold marks every cell that is both "
        "cheaper and more accurate than GART 2.0, that is, every rung at which "
        "GART 2.0 is strictly dominated.",
        "tab:frontier_2d", "5pt"), encoding="utf-8")

    ne = bank["cells"]["noneuc"]["groups"]["Total (all non-EUC_2D)"]
    (OUT / "frontier_noneuc_cost.tex").write_text(ladder(
        ne,
        "Non-EUC\\_2D TSPLIB95 cost/accuracy ladder, the 29 instances both methods "
        "score, both subgradient step rules. Columns are as in "
        "Table~\\ref{tab:frontier_2d}; GART 2.0's time includes the MDS embedding "
        "of Section~\\ref{sec:application}, which the bound does not need. The "
        "calibration constant $c_k$ is the planar one, applied here out of its "
        "fitted domain, so the Cal.\\ columns are indicative and the Raw columns "
        "are the certified result.",
        "tab:frontier_noneuc", "5pt"), encoding="utf-8")

    print("wrote", OUT / "frontier_2d_cost.tex")
    print("wrote", OUT / "frontier_noneuc_cost.tex")


if __name__ == "__main__":
    main()
