# Pre-registered falsification test: does GART 2.0 occupy a useful niche?

**Written 2026-08-11, BEFORE any Held–Karp measurement was returned.** The HK
implementation was still in its build phase when this was fixed.

## The question, in the author's terms

> Run the 1-tree for as many passes as it takes to get a tighter solution than our model
> and record the result. If we get a tighter solution in less absolute time, and Held–Karp
> scales well in high n, and is dimension agnostic, then we can definitively argue that our
> approach is not useful — we have bled from a classical estimator into one that is more
> advanced. But if we hold a middle ground with bounded complexity and moderate performance
> then there is a case to be made.

This is a test the paper can fail. It is written that way on purpose.

## Why the niche argument is the real claim

GART 2.0 is not competing to be the most accurate approximation to the optimum; the Held–Karp
bound and an exact solver both sit above it on accuracy. It is competing to occupy a band:
**materially more accurate than the closed forms, at a cost far below anything that iterates.**
If that band turns out to be empty — if a well-implemented 1-tree is more accurate *and*
cheaper *and* has no domain restriction — then a learned α predictor buys nothing, and the
honest conclusion is that the work bled out of the closed-form regime without arriving
anywhere useful.

## Definitions, fixed here

- **Accuracy** is absolute percentage error against the known optimum, |pred − opt| / opt,
  reported as MAPE and MedAPE per stratum. For HK, `pred` is the bound.
- **Tighter than GART 2.0** means strictly lower MAPE on the same instance set, same screen.
  GART 2.0's targets: ND test 0.6201, 2D benchmark 2.9042, TSPLIB EUC_2D 2.5541,
  TSPLIB non-Euclidean 3.3441 (on the 22 it covers).
- **Cost** is wall-clock under the published solo protocol: one estimator per process,
  11 repeats, threads pinned to 1, JIT and first predict warmed outside the clock, median
  published, IQR retained. Any number not measured that way is not comparable.
- **k** is the subgradient ascent iteration count.

## The competitor must be the STRONGEST reasonable HK, not the weakest

A raw 1-tree bound is systematically below the optimum, so its signed error is negative by
construction and its MAPE overstates its weakness as an *estimator*. Testing against the raw
bound alone would be beating a straw man in our own favour.

Both must be reported:
1. **Raw bound** — the honest lower bound, signed error reported, described as a bound.
2. **Scaled HK** — the bound times a single multiplicative constant fitted **on the training
   split only**, never on the evaluation set. This is HK-as-an-estimator and it is the
   competitor that matters for the niche question.

The kill condition below is evaluated against whichever of the two is stronger at each k.

## KILL CONDITION — GART 2.0 has no useful niche

All three must hold. If they do, the paper says so.

- **K1 — dominated on the cost/accuracy plane.** HK reaches strictly lower MAPE than GART 2.0
  in strictly less absolute time, on the same instances under the same protocol, on the ND
  test split *and* on TSPLIB EUC_2D.
- **K2 — scales no worse in n.** HK's measured time exponent in n is no larger than GART 2.0's,
  so its advantage does not evaporate on large instances. Measured, not assumed, across the
  available n range and reported with the fitted exponents.
- **K3 — dimension agnostic where we are not.** HK holds its accuracy across d without
  retraining and without a domain gate, on the same dimensions GART 2.0 covers *and* on
  d = 100, which GART 2.0 only reaches by extrapolation.

K3 carries real weight and should not be waved through. GART 2.0 is trained on d ∈ {2..50},
declines out-of-domain instances (it refuses `si1032` on the greedy-ratio guard), and needs a
training corpus at all. A 1-tree needs a metric and nothing else. If HK matches it on accuracy
and cost, that generality alone is close to decisive.

## SURVIVAL CONDITION — the middle ground is real

All three must hold:

- **S1** — GART 2.0's cost sits strictly between the classical closed forms and HK-at-crossover.
  Closed forms are 2.574–3.083 ms on the TSPLIB EUC_2D total; GART 2.0 is 6.122 ms. The open
  question is only where HK-at-crossover falls.
- **S2** — GART 2.0's accuracy is strictly better than every closed form on every stratum.
- **S3** — reaching GART 2.0's accuracy costs HK materially more time than GART 2.0 takes.
  "Materially" is fixed here as ≥ 3×, so a marginal result is not read as a win.

## The single decisive measurement

**k\*** = the smallest iteration count at which HK becomes tighter than GART 2.0, per stratum,
and the wall-clock cost at k\*.

Three outcomes, all reportable:
- k\* is small and cheap → K1 is in play; the niche is under threat.
- k\* is large and expensive → S3 holds; the middle ground is real.
- **k\* does not exist** → HK plateaus above GART 2.0's error and no iteration count reaches
  it. Plausible on ND, where GART 2.0 is at 0.6201 MAPE and a converged 1-tree typically sits
  about 1% below optimum. This would be the strongest available result and therefore the one
  to scrutinise hardest: an ascent that plateaus because its step rule is broken looks
  identical to one that has converged, and would manufacture exactly this outcome.
  Convergence must be demonstrated, not inferred from a flat curve.

## Feasibility check, stated for completeness

GART 2.0 must be faster than the exact solver. Concorde times are recorded per instance in
the TSPLIB results. Report the ratio; it is expected to be overwhelming and is a floor, not
an achievement.

## Stop conditions

- Do not tune k, the step schedule, or the stratum set after seeing which way the result
  falls. The schedule is fixed by Volgenant–Jonker before measurement.
- If HK wins, the paper reports that GART 2.0 is dominated. Do not rescue it by weakening the
  competitor, restricting the strata, or quoting the raw bound where the scaled bound is
  stronger.
- A stratum-dependent answer is a legitimate outcome and is probably the most likely one.
  Report it per stratum rather than averaging into a single verdict.
