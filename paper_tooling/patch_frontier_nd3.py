"""Third pass on frontier_positioning.md.

Two corrections the full sweep forces on the second pass:

1. "converged under this step rule" is true only where the ascent stopped on
   the gamma floor or on an exact closure. 14.48 % of the ND split exhausts the
   2000-iteration budget instead -- 28.4 % at n in [200,500] and 42.4 % at
   n in [600,1000] -- so at large n the k = 2000 column is a floor on what the
   method reaches, not its converged value.
2. The exact-closure share splits two ways and the prose collapsed them: 36.60 %
   because the minimum 1-tree is itself a Hamiltonian cycle, 0.50 % because the
   incumbent reached the constructive tour's cost (which certifies that tour
   optimal). 37.10 % is the sum.
"""

from __future__ import annotations

from pathlib import Path

DOC = Path(__file__).resolve().parent / "frontier_positioning.md"

OLD_MECH = """pairwise distance; over the same range the share of instances whose minimum 1-tree *is* a
Hamiltonian cycle — the relaxation closing exactly, gap identically zero — rises from 8.3 % to
between 12.5 % and 37.5 %, and is 37.1 % across the whole ND split. High-dimensional Euclidean TSP
is not the regime the 1 % folklore describes."""

NEW_MECH = """pairwise distance; over the same range the share of instances whose minimum 1-tree *is* a
Hamiltonian cycle — the relaxation closing exactly, gap identically zero — rises from 8.3 % to
between 12.5 % and 37.5 %. Across the whole ND split the relaxation closes exactly on 37.10 % of
instances: 36.60 % because the minimum 1-tree is a Hamiltonian cycle and 0.50 % because the
incumbent reaches the constructive tour's cost, which certifies that tour optimal. The closure
rate is strongly size-driven — 95.0 % at n ≤ 10, 33.1 % at n ∈ [20,100], 2.1 % at n ∈ [200,500],
0.4 % at n ∈ [600,1000] — so it is not the whole explanation, and the dimension trend above holds
at fixed n ∈ [40,250]. High-dimensional Euclidean TSP is not the regime the 1 % folklore
describes."""

OLD_CONV = """  0.1234790356 % to 0.1234790356 %. The k = 2000 column is converged under this step rule. It is
  not proved to be `max_pi w(pi)`: on 4.28 % of instances — 21.9 % at d = 2, 0.1 % at d = 100 —"""

NEW_CONV = """  0.1234790356 % to 0.1234790356 %. Two limits attach to that. The audit sample is n ∈ [40,250],
  and across the full split 14.48 % of instances exhaust the 2000-iteration budget instead of
  reaching the floor — 2.4 % at n ∈ [20,100] but 28.4 % at n ∈ [200,500] and 42.4 % at
  n ∈ [600,1000] — so at large n the k = 2000 column is a floor on what this ascent reaches, not
  its converged value, and the bound's accuracy there is understated. Nor is the column proved to
  be `max_pi w(pi)`: on 4.28 % of instances — 21.9 % at d = 2, 0.1 % at d = 100 —"""

PATCHES = [("mechanism closure split", OLD_MECH, NEW_MECH),
           ("convergence scope", OLD_CONV, NEW_CONV)]


def main() -> None:
    raw = DOC.read_bytes()
    if b"\r\n" in raw:
        raise SystemExit("file has CRLF; the LF anchors below would miss")
    text = raw.decode("utf-8")
    applied, missing = [], []
    for name, old, new in PATCHES:
        if text.count(old) == 1:
            text = text.replace(old, new)
            applied.append(name)
        else:
            missing.append(f"{name} (found {text.count(old)}, expected 1)")
    for m in missing:
        print("MISSING:", m)
    if applied:
        DOC.write_bytes(text.encode("utf-8"))
    for a in applied:
        print("APPLIED:", a)
    print(f"{len(applied)} applied, {len(missing)} missing")


if __name__ == "__main__":
    main()
