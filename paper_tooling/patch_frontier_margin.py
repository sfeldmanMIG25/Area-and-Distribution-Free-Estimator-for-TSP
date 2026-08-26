"""Third frontier pass: print the crossing margin at a precision it can be checked at.

0.04 is 0.038795 rounded honestly, but two decimals span +/-0.005 against a
half-ulp band of 0.0005, so the sentence asserts less than its claim verifies.
The gate calls that UNDER_PRECISE and blocks, correctly. Three occurrences.

Byte-level and anchored, for the reason given in patch_frontier_manuscript.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

PATCHES = [
    ("tsplib.crossing_margin",
     r"times GART 2.0 for a margin of 0.04 percentage points",
     r"times GART 2.0 for a margin of 0.039 percentage points"),
    ("labels.step_margin",
     r"a step function over a margin of 0.04 percentage points",
     r"a step function over a margin of 0.039 percentage points"),
    ("labels.margin_restated",
     r"its margin is 0.59 percentage points rather than 0.04.",
     r"its margin is 0.59 percentage points rather than 0.039."),
]


def main(dry: bool) -> int:
    text = TEX.read_bytes().decode("utf-8")
    applied, bad = [], []
    for name, old, new in PATCHES:
        n = text.count(old)
        if n != 1:
            bad.append(f"{name}: {n} matches")
            continue
        text = text.replace(old, new, 1)
        applied.append(name)
    print(f"applied {len(applied)} / {len(PATCHES)}")
    for n in applied:
        print(f"  APPLIED  {n}")
    for n in bad:
        print(f"  MISSING/AMBIGUOUS  {n}")
    if bad:
        print("REFUSING to write.")
        return 1
    if dry:
        return 0
    out = text.encode("utf-8")
    if out.count(b"\n") != out.count(b"\r\n"):
        print("REFUSING to write: line endings are no longer uniformly CRLF.")
        return 1
    TEX.write_bytes(out)
    print(f"wrote {TEX} ({len(out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main("--dry" in sys.argv))
