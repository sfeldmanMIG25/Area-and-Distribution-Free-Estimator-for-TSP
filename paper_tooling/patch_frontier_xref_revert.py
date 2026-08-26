r"""Fifth frontier pass: drop the Discussion label added in the fourth.

Labelling \subsection{Discussion} moved the prose checker's context digest for
every numeral in that subsection, which re-keyed 64 PRE-EXISTING backlog
entries as NEW and blocked the gate. Absorbing 64 unverified numbers into the
permanent baseline ledger as a side effect of a cosmetic cross-reference is a
bad trade, so the label goes and the sentence drops the \ref instead. The
cross-reference was only disambiguating two protocols for one instance, which
the sentence can do in words.

Byte-level and anchored, for the reason given in patch_frontier_manuscript.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

PATCHES = [
    ("unlabel.discussion",
     r"\subsection{Discussion} \label{subsec:discussion}",
     r"\subsection{Discussion}"),
    ("xref.d18512.drop",
     r"the 239~ms quoted for this instance in Section~\ref{subsec:discussion} "
     r"is the published harness figure for the same work under a different "
     r"protocol.",
     r"the 239~ms quoted for this instance in the Discussion above is the "
     r"published harness figure for the same work under a different protocol."),
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
