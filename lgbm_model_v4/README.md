# GART V4 — precision-first LightGBM rebuild

**Status: scaffolded, not trained.** All scripts exist and are wired together; the
feature CSV, Optuna tuning, and model artifact are not yet generated. A
dataset-corruption prerequisite must be resolved before training (see
[Prerequisite: dataset corruption](#prerequisite-dataset-corruption-to-finish)).

## Why V4

On the 2D benchmark, GART 1.0 (legacy MAPE 7.35 %, SDPE **7.95 %**) edged out GART
2.0 (MAPE 4.98 %, SDPE 8.14 %) on precision alone. The cause is **axis-aligned
bounding-box features**: V3's `bounding_hypervolume = prod(ptp(coords, axis=0))`
is not rotation-invariant, so a slanted-line cluster is scored the same as an
isotropic square of identical extent. GART 1.0 used `ConvexHull.volume`
(rotation-invariant) and won on precision where anisotropy mattered.

V4 fixes this by adding a small rotation-invariant spectral feature block on top
of the V3 MST-topology set, and retunes hyperparameters with a precision-first
objective.

## The 42 candidate features

**V3 baseline (29)** — carried forward verbatim: `n_customers`, `dimension`,
`grid_size`, `bounding_hypervolume`, `log_bounding_hypervolume`,
`node_density`, `log_node_density`, `aspect_ratio`, centroid dispersion
(mean/std/max/iqr), full MST-edge stats (mean/std/skew/kurtosis/max/q10-q90),
MST topology stats (dominance ratio, gap ratio, leaf ratio, degree stats,
diameter, large-edge count).

**Tier 1 — PCA-Oriented Bounding Box (rotation-invariant shape) — 5 features**
- `obb_volume` — PCA-rotated bounding volume
- `log_obb_volume`
- `obb_shrinkage = obb_volume / bounding_hypervolume` in [0, 1]
- `pca_e1_share = λ₁ / Σλᵢ`
- `pca_effective_rank = (Σλᵢ)² / Σλᵢ²`

**Tier 2 — MST-based local density (defined at any d) — 4 features**
- `mst_nn1_mean` — mean over vertices of the minimum incident MST-edge weight.
  By the MST cut property this equals the exact 1-NN distance at every vertex.
- `mst_nn1_cv = mst_nn1_std / mst_nn1_mean`
- `mst_nn2_proxy_mean` — mean of the second-smallest incident MST-edge weight
  over vertices of MST-degree ≥ 2. Upper bound on the true 2-NN distance.
- `mst_nn_gap_ratio = mst_nn2_proxy_mean / mst_nn1_mean`

These replace the earlier cKDTree-based 1-NN / 2-NN block, which was capped at
d ≤ 16 and emitted NaN beyond that. The MST-based block is well-defined at
every dimension, costs O(n) extra work over already-computed MST edges, and
keeps the exact 1-NN component from the original formulation.

**Tier 3 — second-order shape — 2 features**
- `pca_log_det = Σ log λᵢ` (Mahalanobis-style log-volume)
- `mst_edge_pca_e1_share` (edge-vector anisotropy)

**Tier 4 — spatial point-pattern — 1 feature**
- `ripley_L_dev = L(r) / r − 1` at `r = median 1-NN distance`, computed in
  log-space so it is numerically stable at any dimension. Ripley's L is 0 for
  a homogeneous Poisson point process (CSR), positive for clustered point
  sets, and negative for regular / repulsive point sets (grids, lattices).
  This captures a distinct shape axis from OBB / PCA / MST — the
  clustering / regularity tendency of the point cloud itself — and separates
  clustered (+5.5) from grid (−0.1) from uniform (~0) cleanly on smoke tests.

**Tier 5 — tour-ordering upper bound — 1 feature**
- `greedy_nn_over_mst = greedy_nn_tour_length / mst_total_length`. The greedy
  nearest-neighbour tour from the centroid-nearest vertex (Rosenkrantz-Stearns-
  Lewis 1977; within ½·(⌈log₂ n⌉+1)·OPT for metric TSP). Ratio form is
  scale-free and non-redundant with `mst_total_length`. This is the only V4
  feature that encodes *tour* (sequential) information — every other feature
  is either a geometric summary or an MST-tree stat, so this axis is
  orthogonal and strongly correlated with OPT/MST on non-uniform point sets.

Full definitions and the rotation-invariance argument are in
[`feature_engineering.py`](feature_engineering.py).

## Files in this folder

| File | Purpose |
|---|---|
| `feature_engineering.py` | Single source of truth — `compute_features(coords, dim, grid_size)`. Used both offline (CSV) and inline (inference). |
| `build_features_csv.py` | Runs `compute_features` over every row in `tsp_features_v3.csv`, writes `tsp_features_v4.csv` in the repo root. ThreadPoolExecutor wrapper. |
| `feature_analysis.py` | Significance + timing + forward-selection SDPE ablation. Writes `feature_report.md` and `selected_features.json`. |
| `train.py` | Optuna multi-objective (MAPE, SDPE) with TPE multivariate + Hyperband pruner, 100 trials. Final fit → `lgbm_alpha_model_v4.joblib` + `best_params_v4.json` + `pareto_front.png`. |
| `lgbm_estimator_v4.py` | Inference wrapper, API-compatible with V3 estimator (`.estimate(coords, d, grid)` → dict). Drop-in for the benchmark runners. |

## Prerequisite: dataset corruption (to finish)

`data_recovery_v2.py` in the repo root ran during this session and found:

- **82,643** binaries OK unchanged.
- **3,261** binaries had valid headers but garbage bodies — deleted, JSON became
  the authoritative source (matches the stored `generation_seed`; coords are
  correct).
- **4,882** JSONs had valid bins — rewrote JSON from bin.
- **101** instances had both sources corrupt but the `generation_seed` text was
  still extractable from the broken JSON — regenerated coords deterministically
  from the seed (verified byte-identical on a known-good instance).
- **24** truly lost (23 in grid → dropped from the V3 CSV; grid went 90,418 → 90,395).

**The 101 seed-recovered instances now have FRESH coordinates but STALE
`solutions/<name>.sol.json` files.** The solution files were generated from the
old (corrupt) coord body, so their `optimal_cost` is the optimum of a different
point set. For those 101 rows in `tsp_features_v4.csv`, `α = optimal_cost /
mst_total_length` is meaningless — training on this label is contaminated.

**What must happen before training V4:**

1. Identify the 101 seed-recovered instances. They're the instances that were
   regenerated by `data_recovery_v2.py` (the ones in `both corrupt, recovered`
   bucket) — a re-run of the recovery script with an extended log would list
   them. Alternatively: iterate every instance in `tsp_features_v3.previous.csv`
   not in `tsp_features_v3.csv` and check which ones are now fresh.
2. Re-solve those 101 with Concorde and LKH-3 (see `Dataset_Generator.py`
   `solve_instance_batch`). Each instance is ≤ n=1000, d ≤ 100 — expect under
   10 min wall-time total on a 20-core box.
3. Rebuild `tsp_features_v3.csv` by re-running `feature_creator_v3.py` with
   the fresh solutions.
4. Only then rebuild `tsp_features_v4.csv` with `build_features_csv.py`.

Details of the recovery pass are in `data_recovery_v2.py` and
`data_recovery_lost_v2.txt` at the repo root.

## Finish V4 — recipe

Assumes the prerequisite above is resolved. Run from the repo root with
`PYTHONUTF8=1` set (Python 3.14 on Windows still defaults stdout to cp1252).

```bash
# 1. Build the 42-feature CSV for all 90,395 instances (~6-10 min on 20 cores).
PYTHONUTF8=1 python lgbm_model_v4/build_features_csv.py

# 2. Generate the feature report and the auto-filtered selected_features.json
#    (~20-30 min: stats + per-feature timing + 12 forward-selection LightGBM fits).
PYTHONUTF8=1 python lgbm_model_v4/feature_analysis.py

# 3. REVIEW lgbm_model_v4/feature_report.md
#    Edit selected_features.json directly to override any feature decision.

# 4. Train (100 Optuna multi-objective trials + final fit, ~1.5-3 h).
PYTHONUTF8=1 python lgbm_model_v4/train.py
```

Training writes:

- `lgbm_alpha_model_v4.joblib` — final booster
- `best_params_v4.json` — picked hyperparameters + val/test metrics
- `pareto_front.png` — Pareto front with the picked trial highlighted
- `optuna_study.db` — SQLite study (can be re-attached to resume / inspect)
- `run_notes.md` — summary

## Benchmark integration (after V4 is trained)

Add `LGBM_V4` alongside `LGBM_V3` in the 2D / TSPLIB / ND runners. Schedules
already have the V3 entry; add a parallel line that loads the V4 estimator:

```python
# In run_benchmark_2D_all.py and run_benchmark_ND_final.py schedules:
('LGBM_V4', lambda: TSP_V4_LGBM_Estimator(str(SCRIPT_DIR / 'lgbm_model_v4'))),
```

Add the corresponding import:

```python
from lgbm_model_v4.lgbm_estimator_v4 import TSP_V4_LGBM_Estimator
```

The V4 estimator returns the same dict shape as V3 (`estimate`, `alpha`,
`mst_length`, `feature_time`, `inference_time`), so `worker_run_estimator`
works unchanged.

## Decisions baked into V4 (for future reference)

- **Optuna objective:** multi-objective `(minimize MAPE, minimize SDPE)` with a
  MAPE cutoff of 5.5 % when picking the final point from the Pareto front —
  that's V3's 2D MAPE, so V4 must not regress. Inside this budget we minimise
  SDPE.
- **Sampler:** `TPESampler(multivariate=True, group=True)` — handles
  inter-parameter correlations (num_leaves × min_child_samples) better than
  default independent TPE.
- **Pruner:** `HyperbandPruner(min_resource=100, max_resource=5000)` — kills
  trials that show no promise by boosting round 100. Effectively multiplies the
  100-trial budget.
- **Target variable:** α (clipped to [1, 2]) — cost is reconstructed as
  α · mst_total_length. Metrics are reported on cost.
- **Split:** inherited 70/20/10 stratified split from `tsp_features_v3.csv`
  (`split` column). `d = 100` is locked to test.
- **Delaunay dim cap:** 3 (empirical crossover at d = 4 for n = 1000 —
  Delaunay 1.8 s vs dense 0.2 s). Propagated to every estimator and to
  `tsp_utils_2.py`. See the comment on `DELAUNAY_DIM_CAP` in
  `feature_engineering.py`.
- **Local-density via MST, not cKDTree:** the Tier-2 block is derived from
  MST incident edges so every feature is defined at every dimension — no
  NaN-branch dead weight for high-d rows. The exact 1-NN component is
  preserved via the MST cut property.
- **Inference shape-compat:** V4 estimator is drop-in for V3 — same
  constructor signature and return dict.

## Future improvements (GART 3.0 candidates)

Deferred from V4 to keep the current iteration scoped; to be revisited once the
V4 Pareto front is measured.

- **Retarget the model on OPT / 1-tree instead of OPT / MST.** The Held-Karp
  1-tree (MST on V \ {v₀} plus the two cheapest edges at v₀, maximised over a
  sample of pivots v₀) is a strictly tighter lower bound than the MST and is
  the basis of the tightest known LP relaxation (Held-Golden 1970). Changing
  the target from α = OPT / mst_total_length to α = OPT / one_tree_bound would
  concentrate the residual the booster has to learn and should reduce SDPE at
  equal compute. This is a *fundamental* modeling change (affects training,
  inference, and the paper's framing) — hence GART 3.0, not a V4 patch.
  Implementation note: 1-tree is O(n log n) on top of the already-computed
  MST, so the only real cost is redoing the full training run.
- **Multi-start greedy-NN tour.** V4 uses the centroid-nearest start for
  determinism. A max-over-k-starts variant would tighten the upper bound at
  k× compute — worth measuring if training budget allows.

## Open questions for the next session

1. **Should V4 retrain against a 3D/ND benchmark specifically to close the
   precision gap in higher dimensions?** The 2D shortfall vs GART 1.0 was the
   trigger, but the N-D story isn't complete without a fresh N-D run.
2. **Neural_V3 checkpoint:** still absent. Paper doesn't reference Neural_V3,
   so it's not blocking — but if the runs are rebuilt, decide whether to train
   a V4 neural head in parallel (shared feature CSV).
3. **Kwon n ≤ 300 gate:** currently hard-coded in `tsp_utils_2.estimate_tsp_kwon`
   and the 2D runner records `status='kwon_out_of_calibration'` rows. Paper
   should cite 300 as the calibration upper bound (Kwon-Golden-Wasil 1995) —
   confirm before writing.
