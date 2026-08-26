# Pre-registered protocol: de-contaminated augmented arm ("arm C")

**Written 2026-08-11 by the coordinating agent, BEFORE arm C was trained or scored.**

Unlike `support_arms_gates.json`, this document states plainly what was and was not
blind. Read §0 before treating anything here as pre-registration.

---

## 0. Disclosure of what is not blind

Three things must be disclosed rather than claimed away.

**(a) The 0.70 slope threshold is inherited, not fresh.** `support_arms_gates.json`
(2026-08-11 01:16) claims it was "written BEFORE any candidate model was trained or
scored." That overstates it. At 2026-08-10 22:02:37 `augmentation_v2_criteria.csv`
already carried criterion 8, *"LineNoise alpha-slope (n>=200) >= 0.70 (from 0.28)"*, and
scored a different augmented arm against it (FAIL, 0.4286). Gate 6's threshold is that
number verbatim; gate 7's "<= +4.0" restates an outcome already shown reachable. Arm A
itself was not scored before its gates were fixed, and the earlier arm used a different
incumbent (the V3-features model, ND 0.8769) — but the *thresholds* were not chosen blind.
The same thresholds are carried forward here, and this paragraph is the disclosure. Any
paper text describing them as pre-registered must carry it too.

**(b) The de-contamination rule was found by looking at outcomes.** The adversarial
review ran leave-one-family-out and observed that removing the d=2 `lattice` rows
*improves* grid. The rule below is outcome-independent as stated, but it was not
discovered blind. It is re-registered here and re-run from scratch; the previous run's
numbers are discarded, not carried forward.

**(c) Arm A's reported point estimates are the favourable extreme of a nuisance
distribution.** Eight row-order permutations of the identical 874 rows at seed 42 give a
LineNoise slope band [0.6367, 0.8028], mean 0.7279, sd 0.056. The reported 0.8268 lies
above the entire band. Same pattern on line_noise MAPE (band [5.901, 6.358], reported
5.781) and TSPLIB non-EUC MAPE (band [2.510, 2.870], reported 2.431). Row order carries
zero information, so any statistic this sensitive to it cannot be published as a point.
§2 exists because of this.

---

## 1. The de-contamination rule (outcome-independent, fixed here)

> Remove from the training augmentation every row whose generator is provably the
> evaluation set's own generator.

Applied concretely: **drop the 24 d=2 `lattice` rows.** Justification, verified from
source and not from outcomes:

- `data_pipeline/augment_gen.py:392` (`gen_lattice`, d=2) and
  `data_pipeline/d2_benchmark_gen.py:238` (`generate_grid`) both place
  `m = max(2, ceil(sqrt(n)))` sites per side at spacing `G/m`, with cell centres
  `(i + 0.5) * spacing`.
- At `jitter=0.05` the jitter law is identical to the benchmark's
  `(rng.random(2) - 0.5) * (spacing * 0.1)` = U(±0.05·spacing).
- For perfect-square `n` the site sets are **exactly equal** (n=100: 100/100 sites;
  n=400: 400/400). `AUG_lattice-d2-n400-g1000-jitter0p05-r0` is
  `TSP-grid-n400-g1000-*` with a different RNG draw and nothing else.

Therefore the docstring at `augment_gen.py:99` — "DIFFERENT generators … which is what
makes the evaluation a cross-family generalisation test" — is **false for d=2 lattice**
and must be corrected in the code as part of this work.

`hexlattice` is **retained**: a triangular lattice is not among the benchmark's 13
generators, so square-lattice performance learned from it is genuine transfer across
lattice type. This is the mechanism the grid result actually rests on.

Rows removed: 24. Rows retained: 850. Both counts are to be verified, not assumed.

---

## 2. Nuisance protocol (the fix for §0c)

Fixed before running:

- **Concatenation order is fixed in writing**: corpus rows first, then augmentation rows;
  each block sorted ascending by `instance_name`. No shuffling of the concatenation.
- **k = 7 seeds**: 42, 1, 7, 97, 123, 1729, 2024. Chosen to include both extremes already
  observed for arm A (42 the best, 1729 the worst), so the set cannot flatter arm C.
- **Every reported statistic is the median over the 7 seeds**, accompanied by the full
  min–max band. No point estimate from a single seed appears anywhere.
- **Row-order sensitivity is reported, not hidden**: 8 row-order permutations at the
  median seed, band reported alongside.

---

## 3. Gates

Arm C ships only on a clean sweep. Gates 1–9 carry over from `support_arms_gates.json`
with the amendments below; 10 and 11 are new and close holes found by review.

| # | Gate | Rule |
|---|---|---|
| 1 | ND no-regression | median MAPE <= 0.6401 AND median SDPE <= 1.0081 |
| 2 | TSPLIB dispersion vs Asymptotic_MST | **AMENDED**: must pass under the distribution-free **swap permutation** (Holm p < 0.05), not only Pitman–Morgan, AND `detectable_holm` must be **True** (observed SD ratio below the MDE at its own Holm threshold). Arm A passed PM at 0.0058 but failed the swap permutation at 0.13 and sat above its MDE — gate 2 as originally written never read either. |
| 3 | Mean gain vs Calibrated_MST_dn | median gain > MDE, Holm p < 0.05 |
| 4 | Monotonicity probe | **AMENDED**: reported as an artifact-integrity check only. A sum of monotone-constrained trees cannot violate a ceteris-paribus sweep, so 100% is not evidence about the augmentation. Must still be 100%, and the unconstrained-refit control must be reported beside it. |
| 5 | Extraction cost | ratio <= 1.35 (expected exactly 1.00; arm C adds no features) |
| 6 | LineNoise slope, n>=200 | **AMENDED to the band**: median >= 0.70 **AND at least 6 of 7 seeds individually >= 0.70**. The point-estimate form is what let arm A's coin flip through. |
| 7 | grid MSPE | median <= +4.0 |
| 8 | 2D class regression | no class worse than frozen by > 0.15 MAPE, at the median |
| 9 | TSPLIB non-Euclidean | median no worse than frozen (MAPE 3.3441, SDPE 3.8931); coverage must not drop below 22/23 |
| **10** | **TSPLIB EUC_2D dispersion cost** | **NEW.** Closes my own gate hole. SDPE ratio vs frozen must stay **below the design's resolving power**: ratio < 1.174 (the MDE at N=78, r_xy 0.87), AND both robust-scale ratios (MAD, 10%-trimmed SD) <= 1.00. Rationale: a cost we provably cannot detect is publishable if disclosed; one we can detect is not. A hard non-regression rule would be over-strict — the review judged the regression publishable when described honestly. |
| **11** | **A2 one-constant control, slope view** | **NEW.** The oracle per-family constant already beats the candidate on MAPE and that is disclosed as a failed control, not re-litigated. What must hold is the separation the finding actually rests on: candidate LineNoise slope (median) must exceed the recalibrated frozen model's slope (~0.402) by >= 0.25. If it does not, there is no deployable advantage and the arm does not ship. |

---

## 4. Reporting obligations, independent of outcome

These are published whether arm C passes or fails:

1. The TSPLIB EUC_2D dispersion regression, with the PM p, the SD-ratio CI, the MDE, the
   `ts225` concentration, and the robust-scale disagreement.
2. The A2 control as a **failed** control on MAPE (constant recovers 97.9% on line_noise,
   105.7% on grid), with the slope view as the reason the finding survives.
3. The line_noise design provenance: `augment_gen.py:1019-1030` states the augmentation's
   rho targets were chosen by profiling the 210 benchmark line_noise instances. That is
   test-locus-targeted coverage. Not leakage — no instance or label was seen — but it must
   be disclosed, because the coverage was aimed at the evaluation set's locus.
4. §0(a) and §0(b) of this document.
5. The ND gain described correctly: net MAPE improves, but the median instance moves
   ~0.008 pp and ~46% of held-out instances get worse. It is a trade of accuracy on
   already-easy instances for accuracy on the harder half, not a uniform improvement.
6. The augment stratum, if reported at all, as a **held-out refit** value with the
   in-sample figure excluded.

---

## 5. Stop conditions

- If arm C fails any gate, the frozen model `gart2_final.joblib` ships and arm C is
  written up as a negative result. Do not re-cut the rule.
- Do not tune the de-contamination rule to recover a failed gate. The rule in §1 is fixed.
- If a gate looks close, report the band and call it a fail. Arm A's 9/9 was produced by
  exactly the impulse to read a favourable draw as a result.
