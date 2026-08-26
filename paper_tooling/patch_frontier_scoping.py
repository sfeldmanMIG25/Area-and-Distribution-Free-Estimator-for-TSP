r"""Sixth frontier pass: scope the two remaining unqualified superlatives.

"The strongest baseline that uses no learned model" is false as written once
the Held--Karp 1-tree bound is admitted -- the bound uses no learned model, no
training corpus, and beats GART 2.0 by 4.95x on the multidimensional benchmark.
Section~\ref{sec:frontier} says so; these two earlier sentences still asserted
the superlative unscoped. Each gains the roster qualifier and, where the claim
is the headline of a results paragraph, a forward pointer.

Byte-level and anchored, for the reason given in patch_frontier_manuscript.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

PATCHES = [
    ("intro.scope_margin",
     r"Its margin over the strongest baseline that uses no learned model is a "
     r"factor of 2.9 on the multidimensional set",
     r"Its margin over the strongest baseline in that roster that uses no "
     r"learned model is a factor of 2.9 on the multidimensional set"),
    ("results_nd.scope_superlative",
     r"The strongest baseline that uses no learned model is the calibrated "
     r"ratio $\hat\rho(d,n)$ at",
     r"The strongest baseline in the roster of "
     r"Section~\ref{subsec:bench_models} that uses no learned model is the "
     r"calibrated ratio $\hat\rho(d,n)$ at"),
    ("results_nd.frontier_pointer",
     r"conditioning on $n$ is worth 48.8\% of what is reachable here, so it "
     r"accounts for just under half of the gain rather than most of it.",
     r"conditioning on $n$ is worth 48.8\% of what is reachable here, so it "
     r"accounts for just under half of the gain rather than most of it. That "
     r"roster contains no certified bound. Section~\ref{subsec:frontier_nd} "
     r"admits one and it beats GART 2.0 on this benchmark on both cost and "
     r"accuracy, so the comparison in this paragraph ranks the estimators "
     r"against each other and not against everything available."),
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
