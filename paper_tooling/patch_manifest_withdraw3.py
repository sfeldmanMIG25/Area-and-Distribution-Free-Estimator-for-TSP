"""Register the Çavdar Eq. (21) claims the rewritten appendix asserts.

Third and last manifest pass. The four numbers here are ours, not the source's:
how much of the benchmark falls below the fitted range, the step we accept at
the upper endpoint, the ratio there, and what extrapolating would return. They
are backed by ``paper_tooling/cavdar_correction_bank.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "prose_manifest.py"

ANCHOR = '''    Claim(
        id="matched.cavdar_factor",'''

NEW = '''    # -- Appendix: Cavdar--Sokol's Eq. (21), bounded to its fitted range -----
    # Backed by paper_tooling/cavdar_correction_bank.py, which reads the
    # constants off CavdarSokol itself, so changing the implemented correction
    # moves these sentences instead of silently disagreeing with them.
    Claim(
        id="appendix.cavdar.n_below_fit",
        anchor=r"binds on this benchmark: {v} of the {~} 2D instances have",
        expect="bank:cavdar_corr_2d_n_below_min",
        tol="exact",
        note="Adverse disclosure: the source's own correction is out of range on "
             "51.16% of the 2D benchmark, so for slightly over half of it the "
             "ratio is held at the n=100 endpoint. Registered so the "
             "qualification cannot be dropped in a later pass.",
    ),
    Claim(
        id="appendix.cavdar.benchmark_n",
        anchor=r"binds on this benchmark: {~} of the {v} 2D instances have",
        expect="bank:cavdar_corr_2d_n_total",
        tol="exact",
    ),
    Claim(
        id="appendix.cavdar.extrap_ratio_5000",
        anchor=r"grows without bound ($E/T={v}$ at",
        expect="bank:cavdar_corr_ratio_extrap_5000",
        tol=("dp", 2),
    ),
    Claim(
        id="appendix.cavdar.boundary_step",
        anchor=r"That leaves a {v}\\% step at the upper boundary",
        expect="bank:cavdar_corr_step_at_n_max_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="appendix.cavdar.boundary_ratio",
        anchor=r"where the fitted ratio reads {v}.",
        expect="bank:cavdar_corr_ratio_at_n_max",
        tol=("dp", 3),
    ),
'''


def main() -> int:
    text = MANIFEST.read_text(encoding="utf-8")
    if "appendix.cavdar.n_below_fit" in text:
        print("already registered; nothing to do")
        return 0
    if text.count(ANCHOR) != 1:
        print(f"ABORTED: {text.count(ANCHOR)} insertion points")
        return 1
    MANIFEST.write_text(text.replace(ANCHOR, NEW + ANCHOR), encoding="utf-8")
    print("registered 5 Cavdar Eq. (21) claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
