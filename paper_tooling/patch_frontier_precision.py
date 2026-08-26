"""Second frontier pass: print three numbers at a precision their claim can be checked at.

Byte-level and anchored, for the same reason as
``patch_frontier_manuscript.py``: a heredoc has corrupted ``\\ref`` in this
project before.

*   ``108 seconds`` -> ``108.1 seconds``.  The generated value is 108.12; at
    zero decimals the sentence asserts less than its claim can verify.
*   ``53.0`` -> ``52.6``.  The interpolated crossing on the linhp318-excluded
    arm, recomputed in ``frontier_manuscript_numbers.py``, is 52.584.
*   ``1.79--1.89`` -> ``1.785--1.889``.  1.785 printed at two decimals sits
    exactly on the acceptance radius, which is a coin flip rather than a check.
*   ``237 milliseconds`` gains its protocol, because the Discussion quotes 239
    ms for the same instance off the published harness and two unattributed
    numbers for one measurement read as an error.

Run::

    python paper_tooling/patch_frontier_precision.py [--dry]
"""

from __future__ import annotations

import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

PATCHES: list[tuple[str, str, str]] = [
    ("exact.median_seconds",
     r"the published median is 108 seconds against",
     r"the published median is 108.1 seconds against"),
    ("labels.interp_crossing",
     r"the interpolated crossing moves only from 48.7 to 53.0",
     r"the interpolated crossing moves only from 48.7 to 52.6"),
    ("complexity.nd_bound_slopes",
     r"$d\in[15,50]$, against 1.79--1.89 for the bound",
     r"$d\in[15,50]$, against 1.785--1.889 for the bound"),
    ("tsplib.capped_tail_protocol",
     r"493 seconds against GART 2.0's 237 milliseconds.",
     r"493 seconds against GART 2.0's 237 milliseconds on those same three "
     r"repeats; the 239~ms quoted for this instance in "
     r"Section~\ref{subsec:results} is the published harness figure for the "
     r"same work under a different protocol."),
]


def main(dry: bool) -> int:
    text = TEX.read_bytes().decode("utf-8")
    applied, missing, ambiguous = [], [], []
    for name, old, new in PATCHES:
        n = text.count(old)
        if n == 0:
            missing.append(name)
        elif n > 1:
            ambiguous.append(f"{name} ({n})")
        else:
            text = text.replace(old, new, 1)
            applied.append(name)

    print(f"applied {len(applied)} / {len(PATCHES)}")
    for n in applied:
        print(f"  APPLIED  {n}")
    for n in missing:
        print(f"  MISSING  {n}")
    for n in ambiguous:
        print(f"  AMBIGUOUS {n}")
    if missing or ambiguous:
        print("REFUSING to write.")
        return 1
    if dry:
        print("--dry: nothing written")
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
