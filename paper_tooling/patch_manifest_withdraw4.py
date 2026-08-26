"""Register the two remaining result numerals whose sentences the rewrite moved.

The conclusion's 1.00 pp margin over the asymptotic ratio and the appendix's
restatement of the 210-instance matched panel are both bankable, so neither is
left to the recorded backlog. Everything still unregistered after this is a
constant quoted from a source, not a result of ours.
"""

from __future__ import annotations

import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "prose_manifest.py"

ANCHOR = '''    Claim(
        id="matched.cavdar_factor",'''

NEW = '''    Claim(
        id="conclusion.tsplib.asym_margin_pp",
        anchor=r"uniform instances, and {v} percentage points on TSPLIB EUC\\_2D",
        expect="= -1 * {paired_tsplib_by_size_total_asymptotic_mst_ratio_mean_diff}",
        tol=("dp", 2),
        note="Conclusion twin of intro.tsplib.asym_margin_pp. Registered because "
             "the sentence around it was rewritten when the Kwon comparison it "
             "used to sit beside was withdrawn, which re-keyed the occurrence.",
    ),
    Claim(
        id="appendix.classical.panel_b_n",
        anchor=r"The lower panel restricts to the {v} i.i.d.",
        expect="bank:classical_b_random_gart_2_0_n",
        tol="exact",
    ),
'''


def main() -> int:
    text = MANIFEST.read_text(encoding="utf-8")
    if "conclusion.tsplib.asym_margin_pp" in text:
        print("already registered; nothing to do")
        return 0
    if text.count(ANCHOR) != 1:
        print(f"ABORTED: {text.count(ANCHOR)} insertion points")
        return 1
    MANIFEST.write_text(text.replace(ANCHOR, NEW + ANCHOR), encoding="utf-8")
    print("registered 2 claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
