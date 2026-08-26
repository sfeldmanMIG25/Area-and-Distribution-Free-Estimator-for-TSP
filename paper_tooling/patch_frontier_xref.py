"""Fourth frontier pass: point the d18512 cross-reference at the right subsection.

The 239 ms figure lives in the Discussion, not in Results. The Discussion
subsection carried no label, so this adds one and repoints the reference.

Byte-level and anchored, for the reason given in patch_frontier_manuscript.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

PATCHES = [
    ("label.discussion",
     r"\subsection{Discussion}",
     r"\subsection{Discussion} \label{subsec:discussion}"),
    ("xref.d18512",
     r"quoted for this instance in Section~\ref{subsec:results} is the",
     r"quoted for this instance in Section~\ref{subsec:discussion} is the"),
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
