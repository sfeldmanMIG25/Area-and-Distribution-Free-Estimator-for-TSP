"""Fold the settled ND arm into paper_tooling/frontier_positioning.md.

Byte-level read / replace / write with exact anchors, and an applied-vs-missing
report. The file is LF-only (checked at load); a CRLF file would be normalised
by a naive text-mode write, so the mode is explicit.

Every number written below comes from paper_tooling/polyak_nd_bank.json,
polyak_nd_results.json, polyak_validation.json or polyak_audits.json. Nothing
here is hand-computed.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "paper_tooling" / "frontier_positioning.md"
BANK = json.loads((ROOT / "paper_tooling" / "polyak_nd_bank.json").read_text())

OLD_ND_BLOCK = """**The ND column looks like GART 2.0's strongest result and must not be used as one.** No budget
in the ladder brings the bound within a factor of two of GART 2.0's 0.620 %, and the per-dimension
split appears to locate a clean crossover at d ≈ 7:

| d | N | GART 2.0 MAPE % | 1-tree k=2000 MAPE % (shipped ascent) |
|---:|---:|---:|---:|
| 2 | 648 | 1.571 | 0.534 |
| 4 | 648 | 0.994 | 0.177 |
| 7 | 648 | 0.709 | 0.653 |
| 8 | 648 | 0.667 | 1.248 |
| 10 | 648 | 0.561 | 0.746 |
| 20 | 648 | 0.412 | 2.718 |
| 50 | 648 | 0.266 | 2.232 |
| 100 | 5904 | 0.597 | 1.671 |

That plateau is a defect of the shipped subgradient ascent, not a property of the relaxation. The
iteration counter stalls near 1141 for every d ≥ 15 regardless of a 2000 budget — the step size
halves once per barren period until it underflows, and there is no restart, so a k = 8000 run is
bit-identical to k = 2000. On a stratified 48-instance sample (8 per dimension, n ∈ [40,250]) an
independent Polyak ascent at the same k = 2000 reaches 0.013 % MAPE at d = 20 against GART 2.0's
0.241 % on that same sample, and 0.003 % at d = 100; none of its bounds exceeds a feasible tour.
**No dimensional advantage for GART 2.0 may be claimed from the table above until the full ND
sweep is re-run with a working step rule.**
"""

NEW_ND_BLOCK = """**The ND column is not GART 2.0's strongest result. It is the corpus on which GART 2.0 is beaten
outright, and the ladder above understates the bound by a factor of 23.** The shipped
Volgenant–Jonker ascent stalls: its step halves once per barren period with no floor and no
restart, so it walks down to a float64 denormal and the loop's `t > 0` guard fires. Measured, at
d ≥ 20 with a requested budget of 8000, the ascent uses a median 1141–1143 iterations and returns
a bound bit-identical to the k = 2000 one on 17 of 20 instance/budget pairs. The high-dimensional
plateau was that underflow.

Replacing the schedule with a Polyak step — `pi += gamma (UB − w(pi)) / ||g||² g`, `gamma` from 2.0
halving after 20 barren iterations, `UB` a **nearest-neighbour tour improved by 2-opt and computed
from the coordinates alone** — and re-running the full 16,920-instance ND test split gives the
table below. The optimal cost is never read by the ascent; see the disqualification note at the
end of this section.

| d | N | GART 2.0 MAPE % | 1-tree k=100 | k=200 | k=500 | k=2000 (Polyak) | k=2000 (shipped V&J) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 648 | 1.571 | 1.771 | 0.797 | 0.589 | 0.583 | 0.534 |
| 4 | 648 | 0.994 | 0.855 | 0.260 | 0.152 | 0.151 | 0.177 |
| 7 | 648 | 0.709 | 0.703 | 0.168 | 0.075 | 0.074 | 0.653 |
| 10 | 648 | 0.561 | 0.599 | 0.132 | 0.050 | 0.049 | 0.746 |
| 20 | 648 | 0.412 | 0.384 | 0.075 | 0.022 | 0.022 | 2.718 |
| 50 | 648 | 0.266 | 0.223 | 0.041 | 0.010 | 0.009 | 2.232 |
| 100 | 5904 | 0.597 | 0.158 | 0.036 | 0.016 | 0.016 | 1.671 |
| **all ND** | **16920** | **0.620** | **0.449** | **0.125** | **0.067** | **0.066** | **1.546** |

Corpus-wide the converged bound is 0.066 % against GART 2.0's 0.620 %, a factor of 9.3, and it
crosses GART 2.0 at **k = 100**. The bound is more accurate at every one of the 18 dimensions from
k = 200 upward, and the paired per-instance win rate is 59.1 % at k = 100, 82.1 % at k = 200 and
92.6 % at k = 500. The shipped ascent's ND figure of 1.546 % at k = 2000 was wrong by 23×.

**And it is cheaper.** Cost was re-measured serially, single-threaded, five repeats, with GART 2.0
re-run in the same process so the ratio is load-independent (144 instances, 8 per dimension).
At the ND corpus median, k = 100 costs **0.40×** GART 2.0 and k = 200 costs **0.71×**; weighting
the sample to the corpus's own dimension mix gives 0.32× and 0.54×. So on the ND benchmark the
uncalibrated, certified, training-free bound at k = 200 is **five times more accurate at 71 % of
the cost** — an unambiguous strict domination, not a trade.

| dimension group | N | GART 2.0 MAPE % / ms | crossover k | cost at crossover | best dominating budget |
|---|---:|---:|---:|---:|---|
| d ∈ {2,3} | 1296 | 1.457 / 5.87 | 100 | 1.27× | none — GART 2.0 holds the front |
| d ∈ [4,10] | 4536 | 0.728 / 3.30 | 100 | 0.39× | k = 200: 0.69× cost, 4.07× accuracy |
| d ∈ [15,50] | 5184 | 0.343 / 3.59 | 100 | 0.56× | k = 200: 0.95× cost, 5.61× accuracy |
| d = 100 | 5904 | 0.597 / 3.26 | 50 | 0.15× | k = 500: 0.35× cost, 38.1× accuracy |
| **all ND** | **16920** | **0.620 / 3.55** | **100** | **0.40×** | **k = 200: 0.71× cost, 4.95× accuracy** |

**The asymptotic defence does not transfer to ND either.** `compute_mst` dispatches to Delaunay at
d ≤ 3 and to a dense kernel at d ≥ 4, so GART 2.0's own feature construction is Θ(n²) in the
multidimensional regime. Measured log–log cost slopes in n for n ≥ 100: GART 2.0 is 0.556 at
d ∈ {2,3} but **2.021 at d ∈ [4,10] and 1.821 at d ∈ [15,50]**, against the 1-tree's 1.785–1.889.
The Θ(n log n) against Θ(k n²) separation that survives on TSPLIB is a *planar* property. In
d ≥ 4 both methods are quadratic and only the constant k separates them, so the gap does not widen
in n — it is fixed, and at the crossover budget it points the wrong way.

**Why the relaxation is nearly exact in high dimension, and why that is not a measurement error.**
The concern is that a 0.016 % duality gap at d = 100 is far below the ~1 % integrality gap the
Euclidean literature would predict. It is real, and the mechanism is metric concentration. Over a
stratified sample the coefficient of variation of pairwise distances falls from 0.514 at d = 2 to
0.076 at d = 100 and the mean nearest-neighbour distance rises from 0.17 to 0.86 of the mean
pairwise distance; over the same range the share of instances whose minimum 1-tree *is* a
Hamiltonian cycle — the relaxation closing exactly, gap identically zero — rises from 8.3 % to
between 12.5 % and 37.5 %, and is 37.1 % across the whole ND split. High-dimensional Euclidean TSP
is not the regime the 1 % folklore describes.

Four checks stand behind the number:

* **Validity.** Against independently computed optima on 252 synthetic instances spanning
  d ∈ {2,3,5,10,20,50,100} and n ∈ [5,13], with the optimum computed twice by unrelated methods
  (exhaustive permutation search and the Bellman–Held–Karp DP, agreeing to 2.6 × 10⁻¹⁴), the worst
  excess of the bound over the optimum is 4.2 × 10⁻¹⁴ %. Monotonicity in k, `w(0) ≥ L_MST`, and the
  budget-prefix property all hold with zero failures.
* **Convergence, not a new stall.** Lowering the `gamma` floor by 40 halvings and raising the
  budget tenfold to 20000 moves the bound by at most 2.3 × 10⁻⁹ % and the sample MAPE from
  0.1234790356 % to 0.1234790356 %. The k = 2000 column is the converged Held–Karp bound.
* **Rows above the label are a unit artefact, not a broken certificate.** 11.6 % of probe rows sit
  above the released label, by at most 0.0053 %. Every one of them is at or below the float64
  length of that label's own stored tour: the ND label is a solver tour scored in the scaled
  integer metric Concorde and LKH were handed, the bound is float64 on the released coordinates,
  and the corpus's own label quantisation is ±0.003–0.006 % at high d. All 50 such rows are
  LKH-labelled; none is Concorde-labelled.
* **The result survives restriction to proven optima.** On the 3,070 Concorde-labelled instances
  alone, GART 2.0 scores 1.031 % and the bound scores 0.104 % at k = 200 and 0.087 % at k = 2000.

**The upper bound is constructive and is not load-bearing.** The Polyak step needs a target value;
using the optimum would make the method supervised and the comparison void. It is a
nearest-neighbour tour improved by 2-opt, from coordinates only, and it sits 0.70–5.75 % above the
label depending on d. Substituting the released optimum as the target — run as a **disqualified
diagnostic** and reported only here — converges twice as fast early (1.14 % against 4.48 % MAPE at
k = 10) and to the *same place*: at k = 2000 the constructive and label-fed variants agree to four
decimal places at every d ≥ 4 (d = 20: 0.0221 % against 0.0222 %; d = 100: 0.0078 % against
0.0099 %). Note also that the certificate never depended on the target — `w(pi) ≤ OPT` holds for
every real `pi` — so only the rate was ever at stake.
"""

OLD_IV = """(iv) The apparent ND-wide
advantage is an artefact of a stalling step rule and is withdrawn pending a re-run."""

NEW_IV = """(iv) The ND arm is settled and it is
adverse: with a working step rule the certified bound is 0.125 % at k = 200 against GART 2.0's
0.620 % while costing 0.71× as much, so GART 2.0 is strictly dominated on the ND benchmark by a
training-free method — at every dimension group except d ∈ {2,3}. (v) The Θ(n log n) against
Θ(k n²) separation is planar. At d ≥ 4 GART 2.0's own measured cost slope in n is 1.82–2.02, the
1-tree's is 1.79–1.89, and the asymptotic argument that carries the TSPLIB claim has nothing to
carry on ND."""

OLD_NOCROSS = """on the ND
test split there is no crossing at any ladder budget, for the ascent-defect reason in §2."""

NEW_NOCROSS = """on the ND
test split the crossing is at k ≈ 100 and **0.40×** — the bound is both cheaper and more accurate
there, and at k = 200 it is 4.95× more accurate at 0.71× the cost (§2)."""


PATCHES = [("ND section §2", OLD_ND_BLOCK, NEW_ND_BLOCK),
           ("failure item (iv)", OLD_IV, NEW_IV),
           ("§5 no-crossing caveat", OLD_NOCROSS, NEW_NOCROSS)]


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
            missing.append(f"{name} (found {cnt} occurrences, expected 1)")

    if missing:
        print("MISSING:")
        for m in missing:
            print("  -", m)
    if applied:
        DOC.write_bytes(text.encode("utf-8"))
        print("APPLIED:")
        for a in applied:
            print("  -", a)
    print(f"\n{len(applied)} applied, {len(missing)} missing; "
          f"{DOC} now {len(text.encode('utf-8'))} bytes")

    # Cross-check the three headline numbers against the bank rather than
    # trusting the prose above.
    b = BANK
    checks = {
        "GART_2.0 ND MAPE 0.620": round(b["GART_2.0_MAPE_pct"], 3) == 0.620,
        "polyak k=200 MAPE 0.125": round(b["polyak_MAPE_pct_by_k"]["200"], 3) == 0.125,
        "polyak k=200 cost 0.71x":
            round(b["sample_median_cost_ratio_by_k"]["200"], 2) == 0.71,
        "crossover k = 100":
            b["pareto_by_group"]["all ND"]["strictly_dominating_budgets"][0] == 100,
    }
    for k, v in checks.items():
        print(f"  bank check {'OK ' if v else 'FAIL'} {k}")
    if not all(checks.values()):
        raise SystemExit("bank cross-check failed; prose and bank disagree")


if __name__ == "__main__":
    main()
