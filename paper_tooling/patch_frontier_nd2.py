"""Second pass on frontier_positioning.md: replace the probe-scale audit
figures with the full-sweep ones, qualify the convergence claim with the
two-ascent envelope, and record the corrupt-reference-tour robustness cut.

Same discipline as patch_frontier_nd.py: byte-level, exact anchors,
applied-vs-missing report, numbers cross-checked against the artifacts.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "paper_tooling" / "frontier_positioning.md"

OLD_CONV = """* **Convergence, not a new stall.** Lowering the `gamma` floor by 40 halvings and raising the
  budget tenfold to 20000 moves the bound by at most 2.3 × 10⁻⁹ % and the sample MAPE from
  0.1234790356 % to 0.1234790356 %. The k = 2000 column is the converged Held–Karp bound."""

NEW_CONV = """* **Convergence, not a new stall.** Lowering the `gamma` floor by 40 halvings and raising the
  budget tenfold to 20000 moves the bound by at most 2.3 × 10⁻⁹ % and the sample MAPE from
  0.1234790356 % to 0.1234790356 %. The k = 2000 column is converged under this step rule. It is
  not proved to be `max_pi w(pi)`: on 4.28 % of instances — 21.9 % at d = 2, 0.1 % at d = 100 —
  the shipped V&J trajectory reaches a *higher* point, so neither ascent attains the exact
  Held–Karp bound at low d. Taking the per-instance maximum of the two ascents moves the corpus
  figure from 0.0663 % to 0.0632 % and the d = 2 figure from 0.583 % to 0.528 %, which sharpens
  the adverse finding rather than softening it; the Polyak column alone is reported because it is
  one self-contained method with one cost."""

OLD_AUDIT = """* **Rows above the label are a unit artefact, not a broken certificate.** 11.6 % of probe rows sit
  above the released label, by at most 0.0053 %. Every one of them is at or below the float64
  length of that label's own stored tour: the ND label is a solver tour scored in the scaled
  integer metric Concorde and LKH were handed, the bound is float64 on the released coordinates,
  and the corpus's own label quantisation is ±0.003–0.006 % at high d. All 50 such rows are
  LKH-labelled; none is Concorde-labelled."""

NEW_AUDIT = """* **Rows above the label are a unit artefact, not a broken certificate.** Across the full sweep at
  k = 2000, 3,200 of 16,920 instances (18.9 %) sit above the released label. The ND label is a
  solver tour scored in the scaled integer metric Concorde and LKH were handed; the bound is
  float64 on the released coordinates, and the corpus's own label quantisation is ±0.003–0.006 %
  at high d. For the 3,164 of those whose stored tour is consistent with the released coordinates,
  the bound is at or below that tour's own float64 length in every case — worst margin
  2.4 × 10⁻¹³ % — and the largest excess over the integer label is 0.856 %. The remaining 36
  (0.21 % of the corpus) have a stored tour whose float64 length is 1.1–291 % away from its label:
  those tours are inside the known 184-instance corrupt set found by
  `paper_tooling/audit_reference_tours.py`, they are not witnesses in float64, and no independent
  float64 witness for them exists here — the ascent's own constructive tour caps the bound by
  construction and so cannot serve as one. Dropping all 74 corrupt-tour instances that fall in the
  ND test split moves GART 2.0 from 0.6201 % to 0.6182 % and the bound at k = 200 from 0.1253 % to
  0.1216 %, so no conclusion turns on them."""

PATCHES = [("convergence bullet", OLD_CONV, NEW_CONV),
           ("label-audit bullet", OLD_AUDIT, NEW_AUDIT)]


def main() -> None:
    raw = DOC.read_bytes()
    if b"\r\n" in raw:
        raise SystemExit("file has CRLF; the LF anchors below would miss")
    text = raw.decode("utf-8")

    applied, missing = [], []
    for name, old, new in PATCHES:
        cnt = text.count(old)
        if cnt == 1:
            text = text.replace(old, new)
            applied.append(name)
        else:
            missing.append(f"{name} (found {cnt}, expected 1)")

    for m in missing:
        print("MISSING:", m)
    if applied:
        DOC.write_bytes(text.encode("utf-8"))
    for a in applied:
        print("APPLIED:", a)
    print(f"{len(applied)} applied, {len(missing)} missing; "
          f"{len(text.encode('utf-8'))} bytes")


if __name__ == "__main__":
    main()
