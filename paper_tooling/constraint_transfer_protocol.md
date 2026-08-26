# Constraint-transfer protocol — pre-registered

**Written:** 2026-08-11T12:14:53-07:00 (2026-08-11T19:14:53Z)
**Repository HEAD at pre-registration:** `f585549226e19b66ec187bd3502a38f7dec8835f`
**Status:** frozen. Nothing below may be altered after the first measurement is
taken. Any deviation forced by the data is recorded as a numbered addendum at
the bottom, never by editing the rule.

---

## 1. The gap this measurement closes

The manuscript's shipping argument is that aggregate error does not select which
estimator to ship, consistency does. GART 2.0 scores 100% on both axes of the
swept-feature monotonicity probe; the extended-block variant `LGBM_V4` scores
9.1% on `dimension` and 78.6% on `n_customers` while beating GART 2.0 on three of
four strata.

But the distinguishing property is a LightGBM training flag. The paper measures
what *removing* the constraint buys GART 2.0 — its unconstrained twin
`GART2_logit_v3hp` scores 8.4% / 19.0% — and never measures what *adding* it
costs the rival. The shipping argument therefore rests on a configuration
difference that has not been priced. This protocol prices it.

## 2. The one variable

`LGBM_V4` is refit changing exactly one thing: `monotone_constraints` set to
`-1` on `n_customers` and `dimension` and `0` elsewhere, with
`monotone_constraints_method = "basic"` — the identical specification carried by
the shipped GART 2.0 artifact (`lgbm_model_v3/gart2_final.json`).

Held fixed at `lgbm_model_v4/train.py`'s final-fit values:

| Held fixed | Value |
|---|---|
| Feature block | the 32 columns of `lgbm_model_v4/selected_features.json` |
| Target | `alpha = clip(optimal_cost / mst_total_length, 1, 2)`, regressed directly |
| Deployment map | `clip(raw, 1, 2)` (`v4_study._to_alpha(·, "alpha")`) |
| Hyperparameters | `lgbm_model_v4/best_params_v4.json → hyperparameters`, minus `early_stopping_rounds` |
| Boosting rounds | 5382, fixed (no early stopping in the final fit) |
| Splits | the `split` column of `tsp_features_v4.csv`; fit on `train ∪ val`, test untouched |
| Objective / metric | `regression` / `rmse`, `verbosity = -1` |

## 3. Reproduction gate (must pass before any arm is reported)

Arm A (unconstrained, 32 features, seed 42) must reproduce the shipped
`lgbm_model_v4/lgbm_alpha_model_v4.joblib` **bit-identically** — identical
`Booster.model_to_string()`. If it does not, the harness is not refitting
`LGBM_V4` and no number in this study is reportable.

## 4. Falsification gate (must pass before any probe number is reported)

The probe is `v4_study._sweep_monotonicity` with `v4_study.PROBE_D_GRID` (22
points, 2..200), `v4_study.PROBE_N_GRID` (24 points, 5..4000),
`PROBE_TOL = 1e-9`, and the 1000-instance ND-test base sampled at
`random_state = 42`. It is imported, never reimplemented, so every number is
directly comparable with `v4_study_gart2_probe.csv` and
`consistency_31f_probe.csv`.

`consistency_31f.build_falsification` supplies three synthetic predictors with
known monotonicity and `consistency_31f._assert_falsification` checks them: the
strictly increasing one must score 0%, the strictly decreasing one 100%, the
oscillating one below 100%, on both axes. If that check fails, no model's pass
rate from this session is reportable.

## 5. Arms

| Arm | Features | Constraint | Purpose |
|---|---|---|---|
| A | 32 | none | reproduces shipped `LGBM_V4`; the baseline the cost is measured against |
| B | 32 | `-1` on `n_customers`, `dimension` | **the headline arm** |
| C | 31 (32 minus `mst_total_length`) | none | symmetric arm: isolates the extra feature |
| D | 31 (32 minus `mst_total_length`) | `-1` on `n_customers`, `dimension` | symmetric arm: constraint on GART 2.0's feature block |
| G | 31 (GART 2.0's block) | `-1` on `n_customers`, `dimension` | GART 2.0's own recipe, refit at k seeds, so the production model carries a nuisance band too |

`LGBM_V4`'s 32 columns are exactly GART 2.0's 31 plus `mst_total_length`, so C
and D are the clean symmetric arm and no third feature set is invented.

The shipped GART 2.0 artifact (`gart2_final.joblib`) is additionally
re-evaluated, unmodified, through the same evaluation code in the same session.

### What the symmetric arm can and cannot establish

Three variables still separate arm D from arm G: the target transform (raw
clipped alpha vs. logit), the hyperparameters (V4's Optuna values vs. V3's
frozen values), and the training protocol (refit on `train ∪ val` at a fixed
round count vs. fit on `train` with early stopping on `val`). The symmetric arm
therefore **bounds** the contribution of `mst_total_length` and of the
constraint; it does not identify the residual, and the report must say so rather
than attributing the residual to any one of the three.

## 6. Nuisance control

Every arm is fit at **k = 7 seeds**: 42, 43, 44, 45, 46, 47, 48. Seed 42 is the
shipped configuration. The LightGBM `seed` parameter is the only thing that
varies across seeds; it reseeds bagging and feature subsampling, which is the
nuisance distribution two earlier candidates in this project were rejected for
having been read at its favourable extreme.

**The headline statistic is the MEDIAN over the seven seeds. The full band
(min, max) and every per-seed value are reported. No single-seed point may be
quoted as a headline number.**

## 7. Measurement surfaces

Accuracy is measured on the four strata of `v4_study_feature_cache.csv` through
`v4_study._predict_cost`, which is the function every published stratum cell is
computed from:

| Stratum | Instances scored |
|---|---|
| `nd_test` | 16,920 |
| `bench2d` | 2,580 |
| `tsplib_euc2d` | 78 |
| `tsplib_noneuc` | 22 of 23 (screened) |

`augment` (874) is reported as a fifth, **non-decisive** stratum: it is an
out-of-distribution corpus, it is not one of the four strata the shipping
argument is argued on, and it enters no clause of the decision rule.

Both `MAPE` and `SDPE` are recorded per stratum. Consistency is measured by
`pct_nonincr_deployed` on both probe axes.

## 8. Definitions, fixed before looking

**Strata where unconstrained `LGBM_V4` leads GART 2.0.** Read off the existing
published artifact `v4_study_allmodels_strata.csv`, so the target set cannot move
after the fact. On MAPE, `LGBM_V4` vs `GART2_logit_v3hp_mono`:

| Stratum | `LGBM_V4` | GART 2.0 | Lead (pp) |
|---|---|---|---|
| `nd_test` | 0.6187 | 0.6201 | **+0.0014** |
| `bench2d` | 2.7320 | 2.9037 | **+0.1717** |
| `tsplib_euc2d` | 2.9295 | 2.5562 | −0.3733 (GART 2.0 leads) |
| `tsplib_noneuc` | 2.9564 | 3.3464 | **+0.3900** |

**Materiality floor: 0.05 pp.** Chosen because it sits below both real gaps
(0.17 pp, 0.39 pp) and above the `nd_test` gap (0.0014 pp). Consequence, stated
now rather than discovered later: **`nd_test` cannot register as a retained
advantage under any outcome**, because unconstrained `LGBM_V4`'s lead there is
already 36× below the floor. `nd_test` is reported in full and is excluded from
the decision clause.

**Decisive strata: `bench2d` and `tsplib_noneuc`** — the strata where
unconstrained `LGBM_V4`'s lead exceeds the materiality floor.

**RETAINS (per stratum, per metric).** Arm B retains its advantage on a stratum
for a metric when both hold:
1. median(arm B) is lower than median(arm G) by ≥ 0.05 pp; and
2. the two full seven-seed bands `[min, max]` do not overlap.

**RETAINS (per stratum).** The stratum counts as RETAINED only when RETAINS holds
for **both** MAPE and SDPE. If exactly one metric satisfies it, the stratum is
`MIXED`. If neither does, the stratum is `LOST`.

**Probe PASS.** Both the median and the **minimum** over the seven seeds of
`pct_nonincr_deployed` equal 100.0 on **both** `dimension` and `n_customers`.
Anything short of that is `PROBE_PARTIAL`, reported with the per-seed values.

**Supporting, not gating.** A paired Wilcoxon on absolute per-instance errors at
the median seed is reported for the two TSPLIB strata, whose n is small. It
informs the write-up; it is not a clause of the rule.

## 9. The decision rule

> **If constrained `LGBM_V4` reaches 100% on both probe axes AND retains a
> material accuracy advantage over GART 2.0 on the strata where unconstrained V4
> leads, then GART 2.0 should not be the shipped model, and the paper must say
> so.**

Resolved against the definitions in §8:

| Outcome | Condition | Consequence |
|---|---|---|
| **SHIP-V4** | probe PASS **and** both decisive strata RETAINED | GART 2.0 should not be the shipped model. The paper must say so — as the finding, not as a hedge. The shipping argument is rewritten around constrained `LGBM_V4`. |
| **ARGUMENT-COMPLETE** | probe PASS **and** neither decisive stratum RETAINED | The rival's accuracy edge and the consistency property are not simultaneously attainable on that feature set. This becomes one of the paper's strongest results, not a footnote. |
| **FRONTIER** | anything else — probe partial, exactly one decisive stratum RETAINED, or any `MIXED` | Report the frontier. No verdict is forced, and the report states which axis was partially retained and by how much. |

An adverse outcome is never softened. If SHIP-V4 fires, it is reported as
SHIP-V4.

## 10. Outputs

| Artifact | Contents |
|---|---|
| `paper_tooling/constraint_transfer_perseed.csv` | one row per (arm, seed, stratum, metric) — the full per-seed table |
| `paper_tooling/constraint_transfer_probe.csv` | one row per (arm, seed, swept axis), plus the falsification controls |
| `paper_tooling/constraint_transfer_summary.csv` | median / min / max per (arm, stratum, metric) and per (arm, axis) |
| `paper_tooling/constraint_transfer_verdict.json` | the rule evaluated, clause by clause |
| `paper_tooling/constraint_transfer_repro.json` | reproduction-gate and falsification-gate results, library versions |
| bank keys `ctrans_*` in `paper_tooling/tables/paper_numbers.json` | every number above, under stable keys |

The manuscript is **not** edited by this study. The copy-edit agent owns
`Area_Free_Main.tex`. Any number this study puts into prose later must be
registered in `paper_tooling/prose_manifest.py` against a `ctrans_*` bank key.

---

## Addenda

*(None at pre-registration. Numbered entries only; the rule above is never edited.)*
