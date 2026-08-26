r"""Single-writer pass 4: three corrections found on the read-back of passes 1-3.

W4-1  The greedy-feature row added by patch_writer3.py attributed the greedy
      nearest-neighbour construction to \citet{johnson1996asymptotic}. That
      entry is "Asymptotic experimental analysis for the Held--Karp traveling
      salesman bound" and presents no such construction; the paper cites it
      elsewhere for the asymptotic MST ratio. An invented attribution is worse
      than none, so the citation is removed and the row follows the appendix's
      own "\textbf{Proposed}; <mechanism>" convention.

W4-2  The timing paragraph closed with "two to five times the cost of the
      classical ones" while the same paragraph reports 5.55 times the Hilbert
      sort on the smallest bucket. The summary understated its own maximum.
      Source: paper_tooling/gart2_timing_bank.json ::
      gart2_over_model_ratio."n in [51,150]".Hilbert = 5.552.

W4-3  The conclusion listed the two models that order TSPLIB close pairs
      better than GART 2.0 and omitted the third and best of them. The
      same-feature network reaches 74.5%, above GART 1.0 and the predecessor
      feature set at 70.9% and above GART 2.0 at 69.1%; the Discussion names
      all three, and the conclusion may not name fewer.
      Source: paper_tooling/tables/paper_numbers.json ::
      rank_tsplib_euc2d_neural_net_same_features_close5_pct = 74.545455.
"""

from __future__ import annotations

from pathlib import Path

TEX = Path(__file__).resolve().parents[2] / "paper_reference" / "Area_Free_Main.tex"


EDITS: list[tuple[str, str, str]] = [
    (
        "W4-1 greedy feature row: remove the invented attribution",
        r"$n\le3000$; $O(nkd)$ with $k=16$ candidate neighbours above it & \textbf{Proposed}; greedy construction of \citet{johnson1996asymptotic} \\ \bottomrule",
        r"$n\le3000$; $O(nkd)$ with $k=16$ candidate neighbours above it & \textbf{Proposed}; greedy nearest-neighbour construction \\ \bottomrule",
    ),
    (
        "W4-2 timing summary understated its own maximum",
        r"so the honest statement is comparable cost among learned estimators and two to five times the cost of the classical ones.",
        r"so the honest statement is comparable cost among learned estimators and two to five and a half times the cost of the classical ones.",
    ),
    (
        "W4-3 conclusion omitted the best close-pair orderer",
        r"GART 2.0 orders 69.1\% of the close TSPLIB pairs correctly against 70.9\% for both GART 1.0 and the predecessor feature set, although it leads both on the wider 10\% band and on global rank correlation.",
        r"GART 2.0 orders 69.1\% of the close TSPLIB pairs correctly against 74.5\% for the same-feature network and 70.9\% for both GART 1.0 and the predecessor feature set, although it leads the two GART variants on the wider 10\% band and on global rank correlation.",
    ),
]


def main() -> int:
    text = TEX.read_bytes().decode("utf-8")

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
        print("patch_writer4: ABORTED, no bytes written")
    else:
        TEX.write_bytes(text.encode("utf-8"))

    print(f"patch_writer4: applied {len(applied)} / {len(EDITS)}, "
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
