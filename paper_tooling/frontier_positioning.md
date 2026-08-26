# Cost–accuracy positioning of GART 2.0 against the Held–Karp 1-tree bound

Numbers exported to `paper_tooling/frontier_positioning_bank.json`; the point sets plotted in
§6 are `frontier_positioning_points.csv` and `frontier_positioning_points_by_bucket.csv`.
Not spliced into `Area_Free_Main.tex`.

---

## 0. What is measured, and under what conditions

The cost axis is TSPLIB EUC_2D, N = 78, statistic = median over instances of the per-instance
median over repeats, solo protocol (one estimator per process, `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB`
threads = 1, JIT and first predict warmed outside the clock). The accuracy axis is MAPE over the
same 78 instances.

Two measurement sessions contribute. The 14 roster and classical cells were taken on a quiet box
(13–16 % background load). The 1-tree ladder was taken this pass at 20–73 % load, median 47 %.
The two are joined by ratio, not by assumption: GART 2.0 was re-measured alongside the ladder and
returned 6.072 ms against a published 6.122 ms, a reproduction of 0.8 %. Every 1-tree cell below
is its ratio to that co-measured GART 2.0 cell, multiplied by the published GART 2.0 cell.
**Ratios are the primary quantity; absolute milliseconds are advisory.** Relative IQR over 11
repeats is 5.2–7.6 %; the k = 500 and k = 2000 rungs carry 3 and 1 repeats respectively and
reproduce the 11-repeat rungs within 3 %.

Two rows exist for every ascent budget. `1-tree k (bound)` is the certified Held–Karp lower
bound — its MAPE is a duality gap, one-sided by construction, and is not the same quantity as an
estimator's error. `1-tree k × train scalar` multiplies that bound by a single scalar
`c_k = median over the training split of (true / bound)`, fitted on the same split GART 2.0 was
fitted on and applied unchanged to every evaluation corpus; it is an estimator, and it forfeits
the bound guarantee. Both are reported because the second is the fair like-for-like comparator
and the first is the one that carries a certificate.

---

## 1. The frontier

TSPLIB EUC_2D, N = 78. `T(·)` abbreviates `Θ(·)`. Complexity for the MST-based family splits on
dimension because `compute_mst` dispatches to Delaunay + Kruskal at d ≤ 3 and to a dense kernel at
d ≥ 4; the ND figures inherit that split.

| estimator | family | complexity | median ms | x GART 2.0 | MAPE % | Pareto |
|---|---|---|---:|---:|---:|:--:|
| Daganzo | closed form | T(n) | 0.61 | 0.100 | 44.477 | **yes** |
| BHH | closed form | T(n) | 0.65 | 0.105 | 25.158 | **yes** |
| Kwon (extrap.) | closed form | T(n) | 0.72 | 0.117 | 54.425 | - |
| Chien (extrap.) | closed form | T(n) | 0.73 | 0.119 | 30.408 | - |
| Cavdar-Sokol | closed form | T(n) | 0.79 | 0.129 | 23.199 | **yes** |
| 1-tree k=0, x train scalar | 1-tree + train scalar | T(k n^2) | 2.05 | 0.335 | 3.701 | **yes** |
| 1-tree k=0 (bound) | 1-tree bound | T(k n^2) | 2.05 | 0.335 | 10.341 | - |
| Asymptotic_MST | MST ratio | T(n log n) d<=3 / T(n^2) d>=4 | 2.57 | 0.420 | 3.551 | **yes** |
| Calibrated rho(d,n) | MST ratio | T(n log n) d<=3 / T(n^2) d>=4 | 2.62 | 0.428 | 3.766 | - |
| Hilbert curve | space-filling curve | T(n log n) | 2.64 | 0.432 | 45.203 | - |
| MST_Only | MST ratio | T(n log n) d<=3 / T(n^2) d>=4 | 3.08 | 0.504 | 11.350 | - |
| 1-tree k=10, x train scalar | 1-tree + train scalar | T(k n^2) | 3.48 | 0.568 | 2.594 | **yes** |
| 1-tree k=10 (bound) | 1-tree bound | T(k n^2) | 3.48 | 0.568 | 8.317 | - |
| 1-tree k=25, x train scalar | 1-tree + train scalar | T(k n^2) | 5.58 | 0.912 | 1.972 | **yes** |
| 1-tree k=25 (bound) | 1-tree bound | T(k n^2) | 5.58 | 0.912 | 3.552 | - |
| GART 1.0 | learned | T(n log n) d<=3 / T(n^2) d>=4 | 6.12 | 1.000 | 8.456 | - |
| **GART 2.0** | learned | T(n log n) d<=3 / T(n^2) d>=4 | 6.12 | 1.000 | 2.554 | - |
| LGBM_V3 (predecessor) | learned | T(n log n) d<=3 / T(n^2) d>=4 | 8.98 | 1.467 | 3.271 | - |
| 1-tree k=50, x train scalar | 1-tree + train scalar | T(k n^2) | 9.18 | 1.500 | 1.889 | **yes** |
| 1-tree k=50 (bound) | 1-tree bound | T(k n^2) | 9.18 | 1.500 | 2.521 | - |
| LGBM_V4 (unpublished) | learned | T(n log n) d<=3 / T(n^2) d>=4 | 9.81 | 1.603 | 2.930 | - |
| NN_V3 (same features) | learned | T(n log n) d<=3 / T(n^2) d>=4 | 11.24 | 1.836 | 3.287 | - |
| 1-tree k=100, x train scalar | 1-tree + train scalar | T(k n^2) | 16.19 | 2.645 | 1.491 | **yes** |
| 1-tree k=100 (bound) | 1-tree bound | T(k n^2) | 16.19 | 2.645 | 2.036 | - |
| 1-tree k=200, x train scalar | 1-tree + train scalar | T(k n^2) | 30.07 | 4.911 | 1.465 | **yes** |
| 1-tree k=200 (bound) | 1-tree bound | T(k n^2) | 30.07 | 4.911 | 1.995 | - |
| 1-tree k=500, x train scalar | 1-tree + train scalar | T(k n^2) | 71.39 | 11.661 | 1.076 | **yes** |
| 1-tree k=500 (bound) | 1-tree bound | T(k n^2) | 71.39 | 11.661 | 1.491 | - |
| 1-tree k=1000, x train scalar | 1-tree + train scalar | T(k n^2) | 147.90 | 24.158 | 0.981 | **yes** |
| 1-tree k=1000 (bound) | 1-tree bound | T(k n^2) | 147.90 | 24.158 | 1.328 | - |
| 1-tree k=2000, x train scalar | 1-tree + train scalar | T(k n^2) | 277.43 | 45.316 | 0.888 | **yes** |
| 1-tree k=2000 (bound) | 1-tree bound | T(k n^2) | 277.43 | 45.316 | 1.197 | - |

**Exact solver, off the top of the axis.** Concorde solved 25 of these 78 instances in the
published Waterloo record: median 108.12 s, minimum 0.13 s (`berlin52`, n = 52), maximum
1.118 × 10⁷ s (`d2103`, n = 2103 — 129 days). Against GART 2.0's 6.122 ms that is a factor of
1.8 × 10⁴ at the median and 1.8 × 10⁹ at the maximum. Those times come from other hardware and
carry order-of-magnitude weight only; the load-bearing argument is complexity, because
branch-and-cut over an exponential subtour family admits no polynomial bound while GART 2.0 is
Θ(n log n) in d ≤ 3.

**Complexity is measured, not merely asserted.** The 1-tree's Θ(k n²) is confirmed on both
arguments: cost is linear in k (median R² = 0.999 across the 78 instances, 81 % above R² = 0.995),
and the log–log slope in n is 1.94 across all sizes and 2.08–2.13 restricted to n ≥ 1000. The
n² term is irreducible — a Delaunay-restricted 1-tree is heavier than the exact one at 141 of 144
π-vectors drawn from real ascent trajectories, and a Delaunay-restricted ascent exceeds the
published optimum on 8 of 49 instances, so the sparse shortcut that rescues the MST does not
rescue the 1-tree. GART 2.0's log–log slope in n is 0.975 for n ≥ 1000, consistent with the
Θ(n log n) it inherits from Delaunay MST plus KD-tree k-NN features; 92.9 % of its runtime is
feature construction and inference is O(1) in n.

**One cost cell is known to be inflated and is not corrected here.** `compute_mst` selects the
dense kernel over blocked Prim for every d ≥ 4 instance that fits in memory — identical MST
lengths, 31–134× slower. Every MST-based cost at d ≥ 4, GART 2.0's included, carries that penalty.

---

## 2. Accuracy across all three corpora

The table below is accuracy only, and its ND column is the **shipped Volgenant–Jonker** ascent,
which the rest of this section shows to be broken. Cost for TSPLIB is in §1 and cost for ND is
further down this section; the 2D column has no cost measurement. The three cost axes are not
interchangeable: median n is 90 on the 2D benchmark and 75 on the ND test split against 408 on
TSPLIB, which places both synthetic corpora in the regime where the 1-tree is *cheaper* than
GART 2.0 rather than dearer.

| model (V&J ascent) | TSPLIB EUC_2D (78) | 2D bench (2580) | ND test (16920) — superseded |
|---|---:|---:|---:|
| **GART 2.0** | **2.554** | **2.904** | **0.620** |
| 1-tree k=0 -- bound / x scalar | 10.341 / 3.701 | 13.899 / 5.664 | 4.851 / 3.073 |
| 1-tree k=10 -- bound / x scalar | 8.317 / 2.594 | 8.312 / 4.225 | 4.233 / 2.508 |
| 1-tree k=25 -- bound / x scalar | 3.552 / 1.972 | 3.565 / 2.542 | 2.804 / 1.445 |
| 1-tree k=50 -- bound / x scalar | 2.521 / 1.889 | 2.661 / 2.315 | 1.884 / 1.485 |
| 1-tree k=100 -- bound / x scalar | 2.036 / 1.491 | 2.008 / 1.749 | 1.746 / 1.427 |
| 1-tree k=200 -- bound / x scalar | 1.995 / 1.465 | 1.844 / 1.602 | 1.637 / 1.422 |
| 1-tree k=500 -- bound / x scalar | 1.491 / 1.076 | 1.238 / 1.081 | 1.561 / 1.465 |
| 1-tree k=1000 -- bound / x scalar | 1.328 / 0.981 | 1.096 / 0.980 | 1.549 / 1.475 |
| 1-tree k=2000 -- bound / x scalar | 1.197 / 0.888 | 0.982 / 0.881 | 1.546 / 1.477 |

**The ND column is not GART 2.0's strongest result. It is the corpus on which GART 2.0 is beaten
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

Corpus-wide the bound at k = 2000 is 0.066 % against GART 2.0's 0.620 %, a factor of 9.3, and it
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
between 12.5 % and 37.5 %. Across the whole ND split the relaxation closes exactly on 37.10 % of
instances: 36.60 % because the minimum 1-tree is a Hamiltonian cycle and 0.50 % because the
incumbent reaches the constructive tour's cost, which certifies that tour optimal. The closure
rate is strongly size-driven — 95.0 % at n ≤ 10, 33.1 % at n ∈ [20,100], 2.1 % at n ∈ [200,500],
0.4 % at n ∈ [600,1000] — so it is not the whole explanation, and the dimension trend above holds
at fixed n ∈ [40,250]. High-dimensional Euclidean TSP is not the regime the 1 % folklore
describes.

Four checks stand behind the number:

* **Validity.** Against independently computed optima on 252 synthetic instances spanning
  d ∈ {2,3,5,10,20,50,100} and n ∈ [5,13], with the optimum computed twice by unrelated methods
  (exhaustive permutation search and the Bellman–Held–Karp DP, agreeing to 2.6 × 10⁻¹⁴), the worst
  excess of the bound over the optimum is 4.2 × 10⁻¹⁴ %. Monotonicity in k, `w(0) ≥ L_MST`, and the
  budget-prefix property all hold with zero failures.
* **Convergence, not a new stall.** Lowering the `gamma` floor by 40 halvings and raising the
  budget tenfold to 20000 moves the bound by at most 2.3 × 10⁻⁹ % and the sample MAPE from
  0.1234790356 % to 0.1234790356 %. Two limits attach to that. The audit sample is n ∈ [40,250],
  and across the full split 14.48 % of instances exhaust the 2000-iteration budget instead of
  reaching the floor — 2.4 % at n ∈ [20,100] but 28.4 % at n ∈ [200,500] and 42.4 % at
  n ∈ [600,1000] — so at large n the k = 2000 column is a floor on what this ascent reaches, not
  its converged value, and the bound's accuracy there is understated. Nor is the column proved to
  be `max_pi w(pi)`: on 4.28 % of instances — 21.9 % at d = 2, 0.1 % at d = 100 —
  the shipped V&J trajectory reaches a *higher* point, so neither ascent attains the exact
  Held–Karp bound at low d. Taking the per-instance maximum of the two ascents moves the corpus
  figure from 0.0663 % to 0.0632 % and the d = 2 figure from 0.583 % to 0.528 %, which sharpens
  the adverse finding rather than softening it; the Polyak column alone is reported because it is
  one self-contained method with one cost.
* **Rows above the label are a unit artefact, not a broken certificate.** Across the full sweep at
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
  0.1216 %, so no conclusion turns on them.
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

---

## 3. Where the ordering inverts

Per-bucket cost and accuracy, TSPLIB EUC_2D. Cells are `median ms / × GART 2.0 / MAPE %`.

| model | n in [51,150]<br>ms / xG / MAPE% | n in [151,400]<br>ms / xG / MAPE% | n > 400<br>ms / xG / MAPE% |
|---|---:|---:|---:|
| **GART 2.0** | 3.64 / 1.00 / 2.009 | 4.51 / 1.00 / 2.489 | 20.54 / 1.00 / 2.902 |
| Asymptotic_MST | 1.16 / 0.32 / 3.555 | 1.84 / 0.41 / 3.499 | 9.27 / 0.45 / 3.570 |
| LGBM_V4 (unpublished) | 5.72 / 1.57 / 2.283 | 7.16 / 1.59 / 2.526 | 33.58 / 1.63 / 3.478 |
| 1-tree k=25 (bound) | 0.57 / 0.16 / 2.910 | 1.54 / 0.34 / 4.738 | 79.88 / 3.89 / 3.443 |
| 1-tree k=50 (bound) | 1.00 / 0.27 / 2.090 | 2.72 / 0.60 / 3.762 | 138.39 / 6.74 / 2.266 |
| 1-tree k=100 (bound) | 1.82 / 0.50 / 1.563 | 5.07 / 1.12 / 2.977 | 252.03 / 12.27 / 1.929 |
| 1-tree k=200 (bound) | 3.54 / 0.97 / 1.569 | 10.03 / 2.22 / 2.962 | 534.21 / 26.00 / 1.850 |
| 1-tree k=25, x train scalar | 0.57 / 0.16 / 1.692 | 1.54 / 0.34 / 3.100 | 79.88 / 3.89 / 1.675 |
| 1-tree k=50, x train scalar | 1.00 / 0.27 / 1.684 | 2.72 / 0.60 / 3.080 | 138.39 / 6.74 / 1.522 |
| 1-tree k=100, x train scalar | 1.82 / 0.50 / 1.199 | 5.07 / 1.12 / 2.396 | 252.03 / 12.27 / 1.292 |

| bucket | N | GART 2.0 on the Pareto front? | strictly dominating configurations |
|---|---:|:--:|---|
| n in [51,150] | 23 | **no** | 1-tree k=25 × scalar (0.16×, 1.692 %), k=50 × scalar (0.27×, 1.684 %), k=100 × scalar (0.50×, 1.199 %), **k=100 bound (0.50×, 1.563 %)**, k=200 × scalar (0.97×, 1.222 %), k=200 bound (0.97×, 1.569 %) |
| n in [151,400] | 16 | yes | none |
| n > 400 | 39 | yes | none |
| all EUC_2D | 78 | **no** | 1-tree k=25 × scalar (0.912×, 1.972 %) |

The corpus-total row and the bucket rows disagree, and the bucket rows are the defensible ones.
`1-tree k=25 × scalar` dominates GART 2.0 on the corpus pair of statistics because the median cost
is set by the small and mid-size instances while the mean MAPE is set by the large ones — the
domination holds in no single bucket except the smallest. The bucket-A entry is not a summary
artefact: at n ∈ [51,150] the *uncalibrated, certified* bound at k = 100 costs half of GART 2.0 and
is 22 % more accurate, and that is a genuine strict domination.

---

## 4. The frontier claim, in the form the evidence supports

GART 2.0 is cheaper than an exact solver by four to nine orders of magnitude and by an unbounded
asymptotic margin, and that part of the requirement is not in doubt. The second and third parts —
cheaper than the Held–Karp 1-tree bound, and occupying a middle ground between that bound and the
classical closed forms — hold only under a size qualification that the unqualified claim omits.
The 1-tree's cost is Θ(k n²) against GART 2.0's Θ(n log n), so their ordering is a function of n
and k rather than a fixed fact: at n > 400 the accuracy-matching 1-tree costs 6.7× GART 2.0, at
n ∈ [151,400] it costs 0.60×, and at n ∈ [51,150] it costs 0.27×. GART 2.0's fixed feature cost,
92.9 % of its runtime, is what makes it the more expensive of the two below roughly n = 400.
The middle-ground position therefore holds at n ∈ [151,400] and n > 400, where GART 2.0 sits on
the measured Pareto front, and fails at n ∈ [51,150], where the certified bound at k = 100 is
simultaneously half the cost and 22 % more accurate. Against the five region-based closed forms
the accuracy separation is overwhelming — 2.554 % against 23.2–54.4 % — but against the MST-ratio
models that actually populate the published timing column the phrase inverts: GART 2.0 is 1.00
MAPE points from `Asymptotic_MST` (3.551 % at 0.42× the cost) and 1.36 points from the converged
bound (1.197 %), so on the comparators the paper times, GART 2.0's accuracy is closer to the cheap
anchor than to the upper anchor. The claim the evidence does support is an **asymptotic** one:
GART 2.0 buys bound-quality accuracy at Θ(n log n) where the relaxation pays Θ(k n²), the measured
exponents are 0.975 and 2.08–2.13 for n ≥ 1000, and the cost advantage therefore widens without
limit in n even though the crossover at the corpus median is only 1.5×.

**What failed, named plainly.** (i) The unqualified cost ordering "GART 2.0 is cheaper than the
1-tree bound" is false below n ≈ 400 and false at the corpus median for any budget k ≤ 25.
(ii) GART 2.0 is not on the corpus-median Pareto front once the 1-tree family is admitted, and it
is strictly dominated at n ∈ [51,150] by a certified bound requiring no training. (iii) The claim
that GART 2.0's accuracy sits closer to the 1-tree than to the closed forms is true only if
"closed forms" means the area-based estimators; it is false for the MST-ratio models in the timing
table, and the manuscript must not use one phrase for both groups. (iv) The ND arm is settled and it is
adverse: with a working step rule the certified bound is 0.125 % at k = 200 against GART 2.0's
0.620 % while costing 0.71× as much, so GART 2.0 is strictly dominated on the ND benchmark by a
training-free method — at every dimension group except d ∈ {2,3}. (v) The Θ(n log n) against
Θ(k n²) separation is planar. At d ≥ 4 GART 2.0's own measured cost slope in n is 1.82–2.02, the
1-tree's is 1.79–1.89, and the asymptotic argument that carries the TSPLIB claim has nothing to
carry on ND.

**What survives, and is worth stating.** Within the 20-model published roster GART 2.0 is the most
accurate model on all three corpora — 2.554 % against the roster runner-up's 3.271 % on TSPLIB,
2.904 % against 2.990 % on the 2D benchmark, 0.620 % against 0.877 % on the ND test split — and it
lies on the roster's own Pareto front. It reaches, at one MST and one boosted-tree evaluation, an
accuracy the Lagrangian relaxation needs roughly fifty subgradient iterations to match, and it does
so with no ascent, no per-instance tuning, and no dependence on a step-size schedule that the
adversarial pass showed to be the relaxation's practical weak point.

One caveat attaches to that roster statement. `LGBM_V4`, which lives in `lgbm_model_v4/` and appears
in no table of the manuscript, is more accurate than GART 2.0 on the 2D benchmark (2.733 % against
2.904 %) and on the ND test split (0.619 % against 0.620 %). It does not dominate on TSPLIB, where
it is both less accurate and 1.603× the cost, and no solo-protocol cost exists for it on the other
two corpora, so the domination is unverified — but the roster-leader claim is safe only because V4
is excluded from the roster.

---

## 5. The single most useful number

**The Held–Karp 1-tree bound reaches GART 2.0's TSPLIB EUC_2D accuracy at k = 50 subgradient
iterations (interpolated k = 48.9), where it costs 1.50× GART 2.0 — 9.18 ms against 6.12 ms.**

Supporting detail. GART 2.0's MAPE is 2.554 %; the raw bound is 3.552 % at k = 25 and 2.521 % at
k = 50. The paired win rate corroborates the crossing independently: at k = 50 the bound is closer
to the optimum on exactly 50.0 % of the 78 instances, a dead heat. The ratio is robust to load —
the interleaved arm, in which GART 2.0 and the ladder run back to back in one process, gives
1.43×, and correcting the 5.5 % over-read of the mirrored checkpoint clock found by the
adversarial pass gives ≈ 1.42×. An independent re-measurement reproduced the ratio at 1.53× and
1.50× on two passes with reversed process order.

Four qualifications travel with the number. It is a corpus-median statistic: the same k = 50
crossing costs 0.27× at n ≤ 150, 0.60× at n ∈ [151,400], 6.74× at n > 400, and 57.8× on corpus
*total* cost, which the five largest instances dominate. On SDPE rather than MAPE the crossing
moves to k ≈ 57 and 1.65×. On the 2D benchmark the MAPE crossing is k ≈ 42 and 1.30×; on the ND
test split the crossing is at k ≈ 100 and **0.40×** — the bound is both cheaper and more accurate
there, and at k = 200 it is 4.95× more accurate at 0.71× the cost (§2). And the
matched quantities are not the same kind of thing: the bound's MAPE is a duality gap, one-sided by
construction, with signed quartiles at k = 50 of [−2.552, −0.849] and a minimum of −13.917.

---

## 6. Figure specification — cost/accuracy scatter with Pareto frontier

Deliverable: one figure, two panels, for `paper_reference/`. Data: `frontier_positioning_points.csv`
(panel A) and `frontier_positioning_points_by_bucket.csv` filtered to `bucket == "n in [51,150]"`
(panel B). Do not recompute anything.

### Axes and scaling

Both axes **log₁₀**, in both panels. This is not stylistic. Cost spans 0.615 ms to 277.4 ms, a
factor of 451; accuracy spans 0.888 % to 54.4 %, a factor of 61. On linear axes 14 of the 32 points
collapse into the lower-left corner and the frontier's shape is unreadable.

- **x** = median wall-clock ms per instance. Range 0.4 to 400. Major ticks at 0.5, 1, 2, 5, 10, 20,
  50, 100, 200, labelled as plain numbers, not exponents. Second x-axis on top, sharing the scale,
  labelled `× GART 2.0` with ticks at 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 45 — this is the axis the
  reader should trust, and the caption must say so.
- **y** = MAPE %. Range 0.7 to 60. Major ticks at 1, 2, 5, 10, 20, 50. Do not invert; lower is
  better and the frontier runs down-and-right.
- Panel B: x range 0.15 to 8, y range 1.0 to 5.

### Points

| series | marker | count | treatment |
|---|---|---|---|
| closed forms (Daganzo, BHH, Kwon, Chien, Cavdar) | open circle | 5 | neutral grey |
| space-filling / MST-ratio (Hilbert, MST_Only, Asymptotic_MST, Calibrated ρ(d,n)) | open square | 4 | neutral grey |
| learned (GART 1.0, LGBM_V3, LGBM_V4, NN_V3) | filled diamond, small | 4 | neutral grey |
| **GART 2.0** | filled diamond, 2× size | 1 | single accent colour, the only saturated non-ramp element |
| 1-tree raw bound, k ∈ {0,10,25,50,100,200,500,1000,2000} | filled circle on a **solid** connecting line, ordered by k | 9 | sequential ramp keyed to log k, light → dark |
| 1-tree × train scalar, same k | open triangle on a **dashed** connecting line | 9 | same ramp |

The two 1-tree series must be drawn as connected trajectories, not as free scatter: they are one
algorithm at nine budgets, and the line is the message. The raw and scaled series share x exactly,
so they will stack vertically at nine x positions — that vertical offset is the value of the
train-fitted correction and should read as such.

### Frontier

Draw the Pareto staircase over the union of all 32 points as a heavy neutral step line (right,
then down), behind the markers, at ~35 % opacity. Frontier membership is precomputed in the
`pareto` column; do not recompute it. Members are Daganzo, BHH, Cavdar, 1-tree k=0/10/25/50/100/
200/500/1000/2000 × scalar, and Asymptotic_MST.

**GART 2.0 is off this frontier and the figure must show that, not hide it.** Draw a short leader
line from the GART 2.0 marker to `1-tree k=25 × scalar` (5.58 ms, 1.972 %) annotated
"dominated at the corpus median". Shade the strict-domination quadrant relative to GART 2.0
(x < 6.12, y < 2.554) at ~8 % opacity.

### Annotations in panel A

1. Horizontal reference line at y = 2.554 (GART 2.0's MAPE), thin, dotted, full width.
2. Vertical reference line at x = 6.12, same style.
3. Callout on `1-tree k=50 (bound)` at (9.18, 2.521): "bound matches GART 2.0 at k ≈ 49, 1.50× cost".
4. Complexity in the legend text, one line per family — `Θ(n)`, `Θ(n log n)`, `Θ(n log n) d≤3 /
   Θ(n²) d≥4`, `Θ(k n²)` — not encoded as a visual channel.
5. Right-edge axis break with a caret and the text "Concorde (exact): median 1.08 × 10⁵ ms,
   max 1.12 × 10¹⁰ ms". Do not extend the axis to reach it; that would need seven more decades and
   would flatten everything else.

### Panel B — the inversion

Same encodings, restricted to n ∈ [51,150] (N = 23), titled "n ∈ [51,150]: the ordering inverts".
Plot GART 2.0 (3.64, 2.009), 1-tree bound at k = 25/50/100/200, 1-tree × scalar at the same
budgets, Asymptotic_MST (1.16, 3.555) and LGBM_V4 (5.72, 2.283). Shade the domination quadrant
(x < 3.64, y < 2.009) and place `1-tree k=100 (bound)` (1.82, 1.563) visibly inside it, directly
labelled "half the cost, 22 % more accurate — certified bound, no training".

### Do not

- Do not add error bars. Relative IQR is 5.2–7.6 %, smaller than the markers at this scale. State
  the range in the caption instead, and mark the k = 500 and k = 2000 points with a hollow centre
  because they carry 3 and 1 repeats respectively.
- Do not plot the 2D or ND corpora on these axes. Cost was measured on TSPLIB only, and their
  median n of 90 and 75 puts them in a different cost regime.
- Do not use colour alone to separate families; marker shape must carry it in greyscale.
- Do not use gridlines darker than 15 % grey, and none on the minor log ticks.

### Caption must contain

The cost statistic and solo protocol; the two-session normalisation and GART 2.0's 0.8 %
reproduction; the 5.2–7.6 % IQR and the reduced repeat counts at k ≥ 500; and the statement that
the bound's MAPE is a one-sided duality gap rather than an estimator's error.
