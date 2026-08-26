# Prose claim audit — `paper_reference/Area_Free_Main.tex`

Catalogue only. No manuscript edits were made.

**Snapshot**: line numbers are against the file as of `2026-08-11 01:37` (1,371 lines).
The manuscript is being edited concurrently — §5.3 (`Results on non-Euclidean TSPLIB95`),
`tab:tsplib_nonEuc`, and the non-Euclidean sentence of the Conclusion were rewritten
*during* this audit and are already correct. Everything else below is as it stands.
A working copy is at
`C:\Users\catst\AppData\Local\Temp\claude\D--Area-and-Distribution-Free-Estimator-for-TSP\186bd023-35c7-4e67-a732-3273a5b65b82\scratchpad\snapshot.tex`.

**Scope**: prose only — abstract, body text, captions, footnotes. Table bodies excluded.

---

## What changed in the model (ground truth: `lgbm_model_v3/gart2_final.json`, `gart2_final.joblib`)

| Property | Manuscript says | Artifact says |
|---|---|---|
| Feature count | 30 | **31** (`greedy_nn_over_mst` added; `grid_size` and `mst_total_length` excluded) |
| Target | squared-error regression on $\alpha$ | squared error on `z = logit(clip(alpha-1, 1e-6, 1-1e-6))`, inverse `alpha = 1 + sigmoid(z)` |
| Inference clip | clipped to $[1.0,2.0]$ | `clip_after_inverse: false` — the bound is structural, no clip exists |
| Monotone constraints | *not mentioned anywhere* | `-1` on `n_customers` and `dimension`, method `basic` |
| Hyperparameters | 100 Optuna TPE trials minimising validation SDPE | **not tuned** — V3's shipped values frozen (`hyperparameter_provenance`) |
| Early stopping | validation RMSE | `cost_mape` on the val split |
| Trees | 2,031 | **1,118** (`best_iteration`) |
| Depth | mean root-to-leaf 18.9, max 33 | mean leaf depth **11.83**, max **40**; mean per-tree max depth 20.77 |
| Leaves/tree | 148 | 147.8 avg, 148 max |
| Coverage gate | *not mentioned in body prose* | declines instances whose greedy-to-MST ratio is outside the trained range (floor 1.035) |

The linear model and the neural network still consume the **30-feature V3 vector**.
Every "identical feature vector" / "same 30 features" phrase is now false.

**Roster hazard**: results tables now carry a `GART 2.0 (V3 features)` row (14 rows, not
thirteen baselines), and the released tidy tables (`paper_tooling/tables/paper_numbers.json`)
also carry `GART 2.0 (V4 features)` = LGBM_V4, which **beats** GART 2.0 on ND
(0.9683/0.6187 vs 0.9881/0.6201) and on 2D (4.248/2.733 vs 4.687/2.904). Every
superlative must be re-scoped against whichever roster ships.

---

## Catalogue

Verdict key: **CORRECT** · **STALE** (number changed) · **UNVERIFIABLE** (no artifact backs it) ·
**CONTRADICTED** (artifact says something qualitatively different) · **PENDING** (timing; see §Timing)

### Abstract (line 90)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 90 | "2,580 synthetic 2D instances, 16,920 held-out synthetic instances…78 TSPLIB95 EUC\_2D instances" | 2,580 / 16,920 / 78 | 2,580 / 16,920 / 78; GART 2.0 scores all three in full (`coverage.csv`) | CORRECT |
| 90 | "obtain aggregate MAPE/SDPE of 3.33%/5.19%…" (2D) | 3.33 / 5.19 | **2.904 / 4.687** | STALE (predecessor) |
| 90 | "…0.88%/1.28%…" (ND) | 0.88 / 1.28 | **0.6201 / 0.9881** | STALE (predecessor) |
| 90 | "…and 3.27%/3.42%" (TSPLIB) | 3.27 / 3.42 | **2.554 / 2.944** | STALE (predecessor) |
| 90 | "a linear model and a neural network built on the identical feature vector" | identical feature vector | Linear/NN use the 30-feature V3 vector; GART 2.0 uses 31 | CONTRADICTED |
| 90 | "GART 2.0 beats [$\hat\rho(d,n)$] by a factor of 2.1 on the multidimensional benchmark" | 2.1× | 1.8144 / 0.6201 = **2.93×** | STALE |
| 90 | "and by 0.49 percentage points on TSPLIB" | 0.49 pp | 3.7656 − 2.5541 = **1.21 pp** | STALE |
| 90 | "On TSPLIB no constant multiplier improves on 3.54% MAPE" | 3.54 | Grid search over $c\,L_{\rm MST}$ on the 78: min at $c=1.1275$, **3.545%** | CORRECT |
| 90 | "GART 2.0's margin over the asymptotic ratio is not statistically distinguishable from zero" | p ≫ 0.05 | Paired diff **−1.00 pp [−1.77, −0.22], Wilcoxon p = 4.85e−3** (`paired_tests.csv`) | CONTRADICTED |
| 90 | "its gain there is 28% lower dispersion" | 28% | 1 − 2.944/4.752 = **38.0%** | STALE |
| 90 | "the Kwon–Golden–Wasil regression reaches 5.40% MAPE within its fitted range against GART 2.0's 2.21%" | 5.40 / 2.21 | Kwon 5.402 CORRECT; GART 2.0 **2.036** | STALE (GART figure) |
| 90 | "error rises from 1.84% on isotropic instances to 11.60% on near-collinear ones" | 1.84 / 11.60 | **1.484 / 10.752** | STALE |

### Introduction (line 123)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 123 | "We compare it against thirteen baselines" | 13 | Results tables now print 14 comparison rows (adds `GART 2.0 (V3 features)`); tidy tables add a 15th (`GART 2.0 (V4 features)`) | STALE |
| 123 | "a linear model and a neural network on the identical feature vector" | identical | 30 vs 31 features | CONTRADICTED |
| 123 | "GART 2.0 has the lowest aggregate error of any baseline that does not share its feature vector, on all three benchmarks" | superlative ×3 | 2D SDPE: **NN_V3 4.610 < GART 4.687**, and NN no longer shares the feature vector. ND: LGBM_V4 0.9683/0.6187 beats GART if V4 ships. | CONTRADICTED |
| 123 | "The margin ranges from a factor of 2.1 down to 0.28 percentage points" | 2.1× / 0.28 pp | **2.93×** down to **1.00 pp** | STALE |
| 123 | "A feed-forward network on the identical 30 features is more accurate than the boosted ensemble on the 2D benchmark" | NN wins 2D | GART 2.0 MAPE **2.904 < NN 2.990**; paired diff −0.086 pp [−0.164, −0.008], p = 0.44. NN retains only the lower SDPE. | CONTRADICTED |

### Theory / Methodology (§2–§3)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 161 | "standard deviation of $\alpha$…falls monotonically from 0.1837 at $d=2$ to 0.0762 at $d=50$" | 0.1837 / 0.0762 | Training-corpus statistic, independent of the model. Not re-derived here. | UNVERIFIABLE — settle with a `tsp_features_v4.csv` groupby on `split=='train'` |
| 166 | "The following subsections describe its 30 input features" | 30 | **31** | STALE |
| 168 | "Two raw corpus values fall outside this interval…target preparation clips them" | clip in target prep | Still true: `u = clip(alpha-1, 1e-6, 1-1e-6)` before the logit | CORRECT |
| 172 | "we extract 30 features from each TSP instance" | 30 | **31** | STALE |
| 188 | "The remaining 19 features summarize the MST." | 11 geo + 19 MST = 30 | 11 + 19 + 1 (`greedy_nn_over_mst`) = 31; the new feature is in neither appendix table | STALE |
| 212 | "running a linear model and a feed-forward network on the identical 30-feature vector" | identical | Those two run 30; GART 2.0 runs 31 | CONTRADICTED |
| 214 | "The LightGBM learner uses squared-error regression on $\alpha$" | target = $\alpha$ | Squared error on **logit($\alpha-1$)** | CONTRADICTED |
| 214 | "predictions are clipped to $[1.0,2.0]$ at inference" | post-hoc clip exists | `clip_after_inverse: false` — no such clip | CONTRADICTED |
| 214 | "The clip did not activate on the 16,920 multidimensional test predictions, which fell in $[1.033,1.905]$" | range [1.033, 1.905] | Moot (no clip). Range not recomputed — ND benchmark CSV has no `mst_length` column | UNVERIFIABLE — settle by scoring `gart2_final.joblib` on the test split and recording $\hat\alpha$ min/max |
| 214 | "Using seed 42, 100 Optuna TPE trials fit on the training split and minimized validation SDPE" | Optuna-tuned | `hyperparameter_provenance`: "V3's shipped values, frozen. **Not tuned**" | CONTRADICTED |
| 214 | "validation RMSE supplied early stopping" | RMSE | `early_stopping_metric: cost_mape` | CONTRADICTED |
| 214 | "The resulting ensemble contains 2,031 trees with 148 leaves per tree" | 2,031 / 148 | **1,118 trees**; 147.8 avg leaves (max 148) | STALE |
| 219 | "performs $K=2{,}031$ traversals" | 2,031 | **1,118** | STALE |
| 219 | "empirical mean root-to-leaf depth $\bar D\approx18.9$ (maximum 33)" | 18.9 / 33 | Mean leaf depth **11.83**, max **40** (mean per-tree max depth 20.77) | STALE |
| 219 | "approximately $K\bar D=3.8\times10^4$ comparisons per prediction" | 3.8e4 | 1118 × 11.83 ≈ **1.3e4** | STALE |
| 219 | "the pipeline costs $O(n^2d+nd^2+d^3)$…In 2D…$O(n\log n)$" | complexity | Does not account for the greedy-nearest-neighbour feature's cost | UNVERIFIABLE — settle against `lgbm_model_v3/feature_engineering_gart2.py` |
| 223 | "We assess the 30-feature set with mean absolute SHAP magnitudes…dominance ratio 26.4%…dimension and node count 22.2%…centroid 10.0%…hypervolume ≈4.6%" | 30, 26.4, 22.2, 10.0, 4.6 | Feature count is 31. Shares are for the predecessor. | STALE (count) + UNVERIFIABLE (shares) — settle by re-running `shap_analyzer_v3.py` against `gart2_final.joblib` |

### Provenance audit (§3.3)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 206 | "Removing all 184 instances…moves GART 2.0 from 0.877% to 0.876% MAPE" | 0.877 → 0.876 | Those are E0/E1 in `generalization_results.csv`, which reproduce the **predecessor** (0.8769 / 0.8757, 2,031 trees) | STALE |
| 206 | "refitting the model on the cleaned training split moves test MAPE from 0.8757% to 0.8726%" | 0.8757 → 0.8726 | Predecessor refit (E1) | STALE |
| 206 | "every other estimator by less than 0.06 percentage points" | < 0.06 pp | Baseline-only, unaffected by the model change | CORRECT |

### Benchmarking setup (§4.1–§4.3)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 229 | "Daganzo's strip constant carries a $+15.4$% offset on uniform instances with only 9.8% SDPE" | +15.4 / 9.8 | MSPE +15.437, SDPE 9.818 | CORRECT |
| 229 | "the custom Hilbert sort has 31.01% MAPE against 14.24% SDPE" | 31.01 / 14.24 | 31.01 / 14.24 (2D total) | CORRECT |
| 238 | "learned estimators built on the same 30 features" | same 30 | 30 vs GART's 31 | CONTRADICTED |
| 244 | "A linear model and a feed-forward network consume the identical 30-feature vector" | identical | 30 vs 31 | CONTRADICTED |
| 320 | "Every numerically evaluated 2D baseline is defined on this complete subset" (78 EUC\_2D) | full coverage | GART 2.0 covers 78/78 (it declines only `si1032`, an EXPLICIT instance outside this stratum). But gated `Kwon` covers **5 of 78** (`coverage.csv`) | CONTRADICTED (pre-existing, not model-driven) |

### Multidimensional results (§4.4.1)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 343 | "GART 2.0 obtains 0.88% MAPE and 1.28% SDPE overall" | 0.88 / 1.28 | **0.6201 / 0.9881** (MSPE +0.1822) | STALE |
| 343 | "the calibrated ratio $\hat\rho(d,n)$ at 1.81%/2.94%" | 1.81 / 2.94 | 1.8144 / 2.9432 | CORRECT |
| 343 | "the learned $\alpha$ buys a factor of 2.1 on MAPE and 2.3 on SDPE" | 2.1× / 2.3× | **2.93× / 2.98×** | STALE |
| 343 | "a lookup table of 102 constants" | 102 | `calibrated_alpha_table.json` → `rho_dn` has 102 cells | CORRECT |
| 343 | "linear 2.73/3.71, $\hat\rho(d)$ 6.23/7.30, $\alpha=1$ 9.68/7.26, Hilbert 21.20/14.38, BHH 28.01/13.54" | baseline rows | All match `paper_numbers.json` | CORRECT |
| 345 | "SDPE decreases from 1.98% at $n\le10$ to 0.55% at $n\in[501,1000]$" | 1.98 → 0.55 | **1.56 → 0.47** | STALE |
| 345 | "from 2.97% at $d=2$ to 0.61% at $d\in[30,50]$, then **rises** to 0.65% at the unseen $d=100$" | 2.97 → 0.61 → 0.65 (rise) | **2.37 → 0.49 → 0.45** — SDPE now *falls* at $d=100$ | CONTRADICTED (direction reversed) |
| 345 | "while MAPE rises from 0.38% to 1.04%" | 0.38 → 1.04 | **0.30 → 0.60** (still rises) | STALE |
| 347 | "Supplying the exact region…cuts its MAPE from 39.07% to the 28.01%" | 39.07 → 28.01 | 39.067 → 28.009 | CORRECT |

### 2D results (§4.4.2)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 351 | "GART 2.0 has SDPE below 6.3% in every size bucket" | bound 6.3 | Buckets are 4.28 / 5.83 / 5.58 / 4.24 / 2.74 → tight bound is **5.9** | STALE (bound derived from predecessor's 6.29) |
| 351 | "and 3.05% in the $[501,1000]$ bucket" | 3.05 | **2.74** | STALE |
| 351 | "Its aggregate SDPE of 5.19% is less than half the 11.57%" | 5.19 | **4.687** (conclusion "less than half" still holds) | STALE |
| 362 | "GART 2.0 obtains 3.33% MAPE and 5.19% SDPE overall" | 3.33 / 5.19 | **2.904 / 4.687** | STALE |
| 362 | "against 5.75%/9.38%…7.35%/7.95% GART 1.0, 9.18%/11.57%…17.91%/10.28%" | baseline rows | All match | CORRECT |
| 362 | "The same-feature neural network reaches 2.99%/4.61% **and is more accurate than the boosted ensemble** on this benchmark" | NN wins | NN 2.990/4.610 values CORRECT, but GART MAPE **2.904 < 2.990**. NN wins on SDPE only (4.610 < 4.687). | CONTRADICTED |
| 362 | "the custom Hilbert sort has 31.01% MAPE against 14.24% SDPE" | 31.01 / 14.24 | matches | CORRECT |
| 364 | "GART 2.0 ranges from 1.84% MAPE on the isotropic class to 11.60% on Line Noise" | 1.84 / 11.60 | **1.484 / 10.752** | STALE |
| 364 | "the calibrated ratio moves from 3.31% to 20.37%, the asymptotic ratio 7.10% to 25.80%, the $\alpha=1$ floor 17.12% to 34.06%" | baseline rows | All match | CORRECT |

### TSPLIB EUC\_2D results (§4.4.3)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 368 | "all 78 EUC\_2D instances…in three prespecified size buckets (23/16/39)" | 78, 23/16/39 | matches | CORRECT |
| 380 | "GART 2.0 obtains 3.42% SDPE and 3.27% MAPE" | 3.42 / 3.27 | **2.944 / 2.554** | STALE |
| 380 | "against 4.49%/3.77%…4.75%/3.55%…7.44%/8.46% for GART 1.0" | baseline rows | matches | CORRECT |
| 380 | "$\alpha$ on these 78 instances has mean 1.1306 and standard deviation 0.0558" | 1.1306 / 0.0558 | Recomputed `true_cost/mst_length`: mean 1.1306, sd 0.0558 | CORRECT |
| 380 | "The MAPE-minimising constant on these instances is $1.1275$ and reaches 3.54%" | 1.1275 / 3.54 | 1.1275 → 3.545% | CORRECT |
| 380 | "GART 2.0's 3.27% improves on that bound by 0.27 percentage points" | 3.27 / 0.27 pp | **2.554%, improving by 0.99 pp** | STALE |
| 380 | "The dispersion gap is the larger one, 3.42% SDPE against 4.75%" | 3.42 vs 4.75 | **2.944 vs 4.752** | STALE |
| 382 | "In the coarse $n>400$ bucket the GART and asymptotic-ratio SDPE intervals overlap" | overlap | GART 3.05 [2.32, 3.62] vs asymptotic 4.03 [3.20, 4.70] — still overlap, but only on [3.20, 3.62] | CORRECT (margin now much narrower) |
| 382 | "in the post hoc $n>10{,}000$ slice the two are nearly tied…asymptotic ratio lower on SDPE (1.46% versus 1.67%) and GART marginally lower on MAPE (2.71% versus 2.74%)" | near-tie | On the 5 instances with $n>10{,}000$: **GART 2.0 SDPE 0.72 / MAPE 0.575** vs asymptotic 1.456 / 2.740. GART now wins both decisively. (1.67/2.71 are the predecessor's.) | CONTRADICTED |
| 382 | "The largest instance, \texttt{d18512}, is more than $18\times$ the training-size cap" | 18× | 18,512 / 1,000 | CORRECT |
| 384 | "$\hat\rho(2)=1.2610$ gives 11.97% MAPE here against 3.77% for $\hat\rho(2,n)$" | 11.97 / 3.77 | 11.971 / 3.766 | CORRECT |
| 384 | "Çavdar–Sokol 23.20%, BHH 25.16%, Chien 30.41%, Daganzo 44.48%, Kwon 54.43%" | classical rows | All match | CORRECT |

### Matched-domain comparison (§4.5)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 390 | "BHH falls from 23.82% to 8.89%…Çavdar–Sokol 25.54% to 10.79%…Kwon 41.71% to 5.40% with MSPE $-0.05$%…Daganzo $+15.44$%…Chien 18.54% with $+17.90$% bias" | classical rows | All match `paper_numbers.json` | CORRECT |
| 392 | "On the 80 Kwon-domain instances it obtains 2.21% MAPE against Kwon's 5.40%" | 2.21 | **2.036** | STALE |
| 392 | "on the 50 Chien-domain instances 3.40% against Chien's 18.54%" | 3.40 | **2.456** | STALE |
| 392 | "on all 210 uniform instances 1.58% against BHH's 8.89% and the $\alpha=1$ floor's 15.58%" | 1.58 | **1.317** (BHH 8.889 and floor 15.578 CORRECT) | STALE |
| 392 | "The same-feature neural network **again edges the boosted ensemble**, 1.52% against 1.58%" | NN wins | GART **1.317 < NN 1.517**; paired diff −0.200 pp [−0.424, +0.019] | CONTRADICTED |
| 392 | "though on these 210 instances that difference is not significant ($p=0.73$)" | p = 0.73 | **p = 0.069** on the 210 uniform. (p = 0.737 is now the Kwon-domain NN comparison, n = 80.) | STALE |
| 392 | "A factor of 2.4 over Kwon–Golden–Wasil on its home ground" | 2.4× | 5.402 / 2.036 = **2.65×** | STALE |

### Generalization (§4.6) — entire section describes the predecessor

`paper_tooling/generalization_results.csv` E0/SHIPPED = 0.8769 MAPE, 1.2765 SDPE, **2,031 trees**.
That is LGBM_V3, not the shipped 31-feature model (1,118 trees, 0.6201 MAPE).

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 396 | "Its 0.88% MAPE therefore measures held-out sampling" | 0.88 | **0.6201** | STALE |
| 396 | "A baseline refit reproduces the released model bit-for-bit, at 0.8769% MAPE over the full 16,920-row test split and 2,031 trees" | bit-for-bit | E0 reproduces the **predecessor**, not the released model | CONTRADICTED |
| 396 | "The 0.8757% quoted in Section~\ref{subsec:provenance}…on the 16,846 rows" | 0.8757 / 16,846 | Row count 16,846 CORRECT; MAPE is the predecessor's | STALE |
| 398 | "Withholding both $d=15$ and $d=25$…11.8% of the training rows, raises test MAPE at $d=15$ from 0.62% to 0.90% and at $d=25$ from 0.48% to 0.62%" | 0.62→0.90, 0.48→0.62 | Predecessor refits (E2) | STALE — settle by re-running `generalization_experiments.py` against `gart2_final` |
| 398 | "$d\in\{10,20,30\}$ move only from 0.53% to 0.54%" | 0.53 → 0.54 | Predecessor (E0/E2) | STALE |
| 398 | "mean signed error at $d=15$ moves from $-0.005$% to $+0.532$%" | −0.005 → +0.532 | Predecessor | STALE |
| 398 | "Training only on $n\le200$…raises MAPE from 0.49% to 0.85%…SDPE moves only from 0.55% to 0.61%…mean signed error rises from $+0.35$% to $+0.80$%, a factor of 2.3" | E3 numbers | Predecessor | STALE |
| 400 | "Both held-out regimes therefore land at 0.85–0.90% MAPE, **within the range the released model achieves on the full test split**" | inside the range | Released model is now **0.6201%** on the full test split — 0.85–0.90% is well outside it. The sentence's whole argument inverts. | CONTRADICTED |
| 400 | "worth up to 0.28 MAPE points" | 0.28 pp | Predecessor (E2 at $d=15$: 0.6234 → 0.8994) | STALE |

### Degenerate-geometry / augmentation (§4.7) — measured against the predecessor

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 404 | "reports 11.59% MAPE on the near-collinear Line Noise class against 1.84% on the isotropic class" | 11.59 / 1.84 | **10.752 / 1.484**. Also internally inconsistent — lines 364 and 428 say 11.60 for the same quantity. | STALE |
| 404 | "MSPE is $-11.57$, so the model systematically under-predicts" | −11.57 | **−10.738** (sign and story survive) | STALE |
| 404 | "regressing predicted $\alpha$ on true $\alpha$ over Line Noise at $n\ge200$ gives a slope of 0.29 against a correlation of 0.96" | 0.29 / 0.96 | Predecessor-fitted | UNVERIFIABLE — settle by refitting the regression on `gart2_final` predictions for the 210 Line Noise rows |
| 404 | "In the training split, $\alpha>1.45$ occurs essentially only at $n\le10$" | corpus fact | Model-independent | CORRECT (not re-derived) |
| 406 | Success criteria (Line Noise MAPE < 5%, MSPE ±3, ≤0.05 pp ND regression, ≤0.15 TSPLIB/2D, slope ≥ 0.70) | thresholds | Fixed in advance against the predecessor baseline; the regression budgets are relative to a baseline that no longer ships | STALE (framing) |
| 408 | "added 578 instances…cut Line Noise MAPE to 7.95% but moved the slope only to 0.47 while the correlation fell to 0.76" | 7.95 / 0.47 / 0.76 | Predecessor-based | UNVERIFIABLE — `paper_tooling/augmentation_results.csv` predates the model change |
| 408 | "The second round therefore added 275 instances…Line Noise MAPE moved only to 7.76%, the slope fell back to 0.43, and the correlation fell further to 0.67" | 7.76 / 0.43 / 0.67 | Predecessor-based | UNVERIFIABLE — `augmentation_v2_results.csv` |
| 408 | "median $\rho$ between 25 and 46 and $\alpha$ only 1.26–1.37" | data fact | Model-independent | CORRECT (not re-derived) |
| 410 | "mean predicted $\alpha$…decreases, from 1.199 to 1.190…the $n\in[401,1000]$ bucket worsens by 2.34 points" | 1.199 / 1.190 / 2.34 | Predecessor-based | UNVERIFIABLE |
| 410 | "training on half the Line Noise instances…reaches a slope of 0.84 and 2.48% MAPE" | 0.84 / 2.48 | Predecessor-based | UNVERIFIABLE |
| 412 | "kurtosis 2.10…kurtosis 3.22…roughly 1.7 times the full width…crossover sharp near $\rho\approx8$…lands 0.14 to 0.45 below it in $\alpha$" | data facts | Model-independent geometry claims | CORRECT (not re-derived) |
| 412 | "the \texttt{grid} sub-generator's MSPE improved from $+8.48$ to $+2.94$" | +8.48 → +2.94 | Predecessor-based | UNVERIFIABLE |

### Discussion (§4.8)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 418 | "GART 2.0 obtains Spearman $\rho$/Kendall $\tau$ of 0.9993/0.981 on 2D, 1.0000/0.998 on the multidimensional set, and 0.9991/0.985 on TSPLIB EUC\_2D" | rank stats | Recomputed for the shipped model: **2D 0.9994 / 0.9828**; **TSPLIB 0.9992 / 0.9858**. (0.9993/0.9806 and 0.9991/0.9852 are LGBM_V3's exactly.) ND not recomputed. | STALE |
| 418 | "the $\alpha=1$ control obtains 0.9961/0.962, 0.9996/0.986, and 0.9986/0.980" | control rows | Recomputed 2D 0.9961/0.9615, TSPLIB 0.9986/0.9799; ND per table 0.9996/0.9857 | CORRECT |
| 420 | "GART 2.0 orders 72.8% of 74,835 2D pairs…correctly" | 72.8 | Pair count 74,835 CORRECT; GART 2.0 = **74.47%** (72.76% is LGBM_V3) | STALE |
| 420 | "89.6% of a fixed 46,434-pair sample from the multidimensional set" | 89.6 / 46,434 | Predecessor value; the fixed-seed sampler lives only in `paper_tooling/gen_paper_numbers.py` (dated Apr 19) and persists nothing | UNVERIFIABLE — settle by re-running `gen_paper_numbers.py` |
| 420 | "70.9% of 55 TSPLIB pairs" | 70.9 / 55 | 55 pairs CORRECT; GART 2.0 = **69.09%** (70.91% is LGBM_V3) | STALE |
| 420 | "against 71.1%, 68.8%, and 61.8% for the $\alpha=1$ control" | control | 2D 71.10, ND 68.85, TSPLIB 61.82 | CORRECT |
| 420 | "The multidimensional gain of 20.7 percentage points is the substantive one" | 20.7 pp | Depends on the unverified ND value | UNVERIFIABLE |
| 420 | "the 2D gain of 1.7 points is not [substantive]" | 1.7 pp | **3.37 pp** (74.47 − 71.10) | CONTRADICTED |
| 420 | "on TSPLIB 55 pairs cannot support a 9-point claim" | 9 pp | **7.3 pp** (69.09 − 61.82) | STALE |
| 420 | "At a 10% threshold GART obtains 82.0%, 94.6%, and 78.3%" | 82.0 / 94.6 / 78.3 | 2D **83.59**, TSPLIB **79.25**; ND unverified | STALE |
| 420 | "The same-feature network orders TSPLIB close pairs better than GART 2.0, 74.5% against 70.9%" | 74.5 vs 70.9 | NN 74.55 CORRECT; GART **69.09**. Direction survives, number does not. | STALE |
| 424 | "The $n>400$ TSPLIB bucket has $R^2_\alpha=-0.549$" | −0.549 | **−0.071** (−0.549 is LGBM_V3's exactly) | STALE |
| 424 | "while tour-cost MAPE remains 3.75%" | 3.75 | **2.90** | STALE |
| 426 | "On TSPLIB EUC\_2D the difference between GART 2.0 and the asymptotic MST ratio is $-0.28$ pp with a 95% interval of $[-1.05,+0.45]$ and $p=0.77$: …indistinguishable from zero" | −0.28, p = 0.77 | **−1.00 pp [−1.77, −0.22], p = 4.85e−3** — now significant | CONTRADICTED |
| 426 | "its real gain is dispersion, 3.42% SDPE against 4.75%" | 3.42 vs 4.75 | **2.944 vs 4.752** | STALE |
| 426 | "the feed-forward network…is more accurate than the boosted ensemble by 0.34 percentage points, with a 95% interval of $[0.27,0.43]$" | +0.34 pp | **−0.086 pp [−0.164, −0.008], p = 0.44** — sign flips, GART is now (insignificantly) ahead | CONTRADICTED |
| 426 | "Gradient boosting is therefore not the source of the advantage on that benchmark; the feature set is." | conclusion | Rests on the reversed comparison above | CONTRADICTED |
| 428 | "GART 2.0's MAPE is 1.84% isotropic, 2.00% biased, 2.60% clustered, 4.43% geometric-structure, 11.60% Line Noise" | five values | **1.484 / 1.822 / 2.185 / 3.697 / 10.752** (the five quoted values are LGBM_V3's row exactly) | STALE |
| 428 | "it costs six times the isotropic error" | 6× | 10.752 / 1.484 = **7.25×** | STALE |

### Timing (line 430) — all PENDING

Per instruction, no new wall-clock measurement was taken and none is proposed. The box
is loaded. The 82.6/16.3/1.2 decomposition was verified only for the predecessor
(82.57 / 16.28 / 1.15); the 31-feature model adds a greedy-nearest-neighbour pass and no
valid serial timing exists for it.

| Line | Claim as written | Asserts | Status | Verdict |
|---|---|---|---|---|
| 430 | "GART 2.0 spends 82.6% of total time on feature extraction and MST construction and 16.3% on LightGBM inference; the residual 1.2%…" | 82.6 / 16.3 / 1.2 | Predecessor-verified only. `paper_tooling/tables/table_time_breakdown.csv` is dated Apr 18, reports 74.75 / 25.03 / 0.22 for TSPLIB, and uses the stale ND count 16,907 — it does not back the sentence either. | PENDING — needs a fresh serial timing run on an idle machine |
| 430 | "the median GART 2.0 prediction is slower than generating the reference tour itself, 171 ms against 122 ms" | 171 vs 122 | 171 ms is the predecessor's row. The respliced ND table now shows GART 2.0 at 121 ms against the same 122 ms reference — **the direction of the sentence reverses**. Do not adopt either number as a result. | PENDING (and the qualitative claim is at risk) |
| 430 | "at $n\in[501,1000]$ the reference tour takes 3,799 ms against GART's 272 ms" | 272 | 272 ms is the predecessor's row; the respliced table shows 465 ms | PENDING |
| 430 | "on TSPLIB \texttt{d18512} GART predicts in 127 ms" | 127 | The current `all_models_tsplib.csv` records `GART_2.0` `total_time_s` = 0.7525 for `d18512` (LGBM_V3 = 0.261). Neither is a clean measurement. | PENDING |

### Application §5 — already corrected during this audit

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 434 | "a fixed $\alpha=1.136$ close to the MAPE-minimising constant $1.134$ over the full 111-instance TSPLIB set" | 1.134 | Grid search over all 111: min at **1.1341** | CORRECT |
| 434 | "The MAPE-minimising constant on these 23 instances is $1.1718$ and reaches 7.55%" | 1.1718 / 7.55 | 1.1718 → 7.550% | CORRECT |
| 436 | "omitted two of the model's **thirty** features" | 30 | **31** | STALE |
| 438 | "33 non-EUC\_2D: 4 CEIL\_2D, 2 ATT, 10 GEO, 17 EXPLICIT…leaving 23 instances, including seven EXPLICIT" | counts | matches | CORRECT |
| 442 | "The cap prevents five of 19 MDS cases from reaching 99.9%" | 5 of 19 | `tab:mds_diagnostics` below-target: 1 + 0 + 4 = 5, of 2 + 10 + 7 = 19 | CORRECT |
| 448 | Caption: "GART 2.0 declines \texttt{si1032}, whose greedy-to-MST ratio $1.0260$…scored on 22 instances against 23" | 22/23, 1.0260, 1.035 | `coverage.csv`: `greedy_ratio_out_of_range:1.0260` | CORRECT (already fixed) |
| 480 | "On the 22 screened instances it accepts, GART 2.0 has aggregate SDPE 3.89% and MAPE 3.34%…10.48% and 8.26% for the fixed $\alpha=1.136$…7.55%…predecessor…4.88% SDPE and 4.63% MAPE…$\alpha$ ranges from 1.006 to 1.511…ATT 3.38% against 3.66%…CEIL\_2D…within 1.3 percentage points" | all | 3.893 / 3.344 on n=22; fixed 10.483 / 8.261 on n=23; V3 4.877 / 4.633 on n=23; α ∈ [1.0063, 1.5108]; ATT 3.38 vs 3.66; CEIL\_2D SDPE spread 5.16–6.44 = 1.28 pp | CORRECT (already fixed) |

### Conclusion (§6)

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 484 | "predicts the MST-to-tour scaling factor from **30** MST, centroid, and coordinate-range features" | 30, three families | **31**, and the 31st (`greedy_nn_over_mst`) is a greedy-tour ratio — none of the three named families | STALE + CONTRADICTED (taxonomy) |
| 484 | "two learned models sharing its feature vector" | sharing | Linear/NN are on the 30-feature V3 vector | CONTRADICTED |
| 484 | "it has the lowest aggregate SDPE on the multidimensional and TSPLIB EUC\_2D benchmarks" | superlative | Holds against the roster printed in the .tex tables. Fails on ND if `GART 2.0 (V4 features)` ships (0.9683 < 0.9881). | CORRECT-conditional — re-scope against the final roster |
| 484 | "and the lowest of any estimator not sharing its feature vector on the 2D benchmark" | superlative | **NN_V3 SDPE 4.610 < GART 4.687**, and NN does not share the 31-feature vector | CONTRADICTED |
| 484 | "a factor of 2.1 on MAPE over a calibrated constant on the multidimensional set" | 2.1× | **2.93×** | STALE |
| 484 | "a factor of 2.4 over Kwon–Golden–Wasil" | 2.4× | **2.65×** | STALE |
| 484 | "and 0.28 percentage points on TSPLIB EUC\_2D" | 0.28 pp | **1.00 pp** | STALE |
| 484 | "where the paired difference against the asymptotic ratio is not statistically distinguishable from zero" | p ≫ 0.05 | **p = 4.85e−3** | CONTRADICTED |
| 484 | "and the real gain is 28% lower dispersion" | 28% | **38.0%** | STALE |
| 484 | "a feed-forward network on the same 30 features is more accurate than the boosted ensemble" | NN wins 2D | GART MAPE 2.904 < NN 2.990 | CONTRADICTED |
| 484 | "On the 22 of 23 screened non-EUC\_2D instances it accepts, the hybrid MDS pipeline obtains 3.34% MAPE against 8.26%…with 7.55% the bound" | all | matches | CORRECT (already fixed) |
| 486 | "error is **11.59%** on the near-collinear class against 1.84% on the isotropic class" | 11.59 / 1.84 | **10.752 / 1.484**; also inconsistent with the 11.60 used at lines 364 and 428 | STALE |
| 486 | "874 newly solved instances…the slope on those instances rose only from 0.29 to 0.43 while its correlation fell from 0.96 to 0.67" | 0.29→0.43, 0.96→0.67 | Predecessor-based | UNVERIFIABLE |
| 486 | "Withholding a dimension or the large-$n$ range from training costs up to **0.35** MAPE points" | 0.35 pp | Predecessor (E3: 0.4944 → 0.8478). Also inconsistent with the 0.28 pp at line 400. | STALE |
| 486 | "removing them moves every reported metric by less than 0.06 percentage points" | < 0.06 pp | Predecessor-measured for the GART rows | UNVERIFIABLE |

### Appendices

| Line | Claim as written | Asserts | Correct current value | Verdict |
|---|---|---|---|---|
| 498 | "\texttt{build\_paper\_tables.py --check} re-derives all **1,528** table cells" | 1,528 | Cell count changed with the resplice (`tab:tsplib_nonEuc` gained an $N$ column and lost the mean-$\alpha$ row group; a `GART 2.0 (V3 features)` row was added to every results table) | UNVERIFIABLE — settle by running `paper_tooling/build_paper_tables.py --check` and reading the reported count |
| 499 | "the two released out-of-interval rows are \path{N1000_D15_G100_...} (raw $\alpha=0.974823$) and \path{N5_D2_G100_oc_33} (raw $\alpha=2.059518$)" | corpus rows | Model-independent | CORRECT (not re-derived) |
| 566 | "list the **19** MST-derived features used by GART 2.0, complementing the **11** geometric and centroid features" | 19 + 11 = 30 | Total is 31; `greedy_nn_over_mst` appears in neither appendix table | STALE |
| 616 | "To validate the **30-feature** set…Table~\ref{tab:shap_top} ranks all **30** features" | 30 | **31** | STALE |
| 618 | "dominance ratio contributes 26.4%…Dimension and size jointly 22.2%…centroid 10.0%…hypervolume ≈4.6%" | SHAP shares | Predecessor SHAP run | UNVERIFIABLE — settle by re-running `shap_analyzer_v3.py` on `gart2_final.joblib` |
| 620 | "yielding the **30-feature** set described in Section~\ref{subsec:features}" | 30 | **31** | STALE |
| 830 | "Multidimensional pair counts are a fixed-seed sample of 46,434 qualifying pairs" | 46,434 | Sample size is model-independent | CORRECT |
| 1271 | "lists the **11** geometric and centroid features" | 11 | 11 is right for that table; the document total is 31, not 30 | CORRECT (in isolation) |
| 1368 | "GART 2.0 obtains **3.88%** MAPE on these seven cases" | 3.88, 7 cases | **3.013% on six of the seven** — GART 2.0 declines `si1032`. 3.88% is the predecessor's value on all 7. | STALE + coverage error |

---

## Omissions (not false, but the prose no longer describes the shipped model)

1. **Monotone constraints** appear nowhere in the manuscript. The shipped model applies
   non-increasing constraints on `n_customers` and `dimension`.
2. **The logit target** appears nowhere. §3.3 still describes a raw-$\alpha$ regression with
   an inference-time clip.
3. **`greedy_nn_over_mst`** appears in no feature table, no SHAP ranking, no complexity
   analysis, and no cost discussion — despite being the single largest source of the
   improvement (`v4_study_summary.json`: 90.6% of the ND MAPE gain, 64.3% of the 2D gain).
4. **The greedy-ratio coverage gate** is described only in the caption of
   `tab:tsplib_nonEuc`. §3 and §4 never state that the estimator can decline an instance,
   so "all N instances" phrasing elsewhere is unqualified.
5. **`mst_total_length` and `grid_size` exclusions** — §3.2 explains only the
   `mst_total_length` exclusion; `grid_size` (generator fingerprint) is not mentioned.

---

## Table bodies out of scope but still stale (flagged for the table tool)

- `tab:hyperparams` (line ~1295): 2,031 trees / avg depth 18.89 / max depth 33 /
  objective `regression_l2 (squared error)` / "Optuna TPE, 100 trials, validation-SDPE
  objective". All four are wrong for `gart2_final` (1,118 / 11.83 or 20.77 / 40 /
  squared error on the logit target / hyperparameters frozen from V3, not tuned).
- `tab:rank` (line ~837): the `GART 2.0` rows are LGBM_V3's values verbatim. Verified by
  recomputation: 2D 0.9994/0.9828/74.47/83.59 and TSPLIB 0.9992/0.9858/69.09/79.25 for the
  shipped model. No artifact backs this table — `gen_paper_numbers.py` (Apr 19) prints to
  stdout and persists nothing.
- `tab:paired` (line ~874): every row is stale. Current values are in
  `paper_tooling/tables/paired_tests.csv` (regenerated 01:19).
- `tab:shap_top` (line ~627): 30 rows for a 31-feature model.
- `tab:benchmark_models` rows for Linear / Neural net say "on the 30 features" — correct
  for those two models, but the surrounding prose implies parity with GART 2.0.
