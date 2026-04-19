# LGBM V3 → V4 Upgrade: Deep Research Report

**Date:** 2026-04-18  
**Scope:** Complete analysis of LGBM V3 training pipeline, feature engineering, and weaknesses to inform V4 planning.

---

## 1. Current V3 State: Training & Features

### 1.1 Training Setup

| Component | Configuration | Source |
|-----------|---------------|--------|
| **Target** | `alpha = optimal_cost / mst_total_length`, clipped to [1.0, 2.0] | `LGBM_Alpha_Model_V3.py:42` |
| **Loss Function** | L2 regression (`regression_l2`) | `LGBM_Alpha_Model_V3.py:66` |
| **Metric** | RMSE (validation) | `LGBM_Alpha_Model_V3.py:67` |
| **Optimizer** | Optuna TPE multivariate + HyperbandPruner | `LGBM_Alpha_Model_V3.py:114-115` |
| **Hyperparameter Trials** | 100 trials | `LGBM_Alpha_Model_V3.py:31` |
| **CV Strategy** | 3-way split (train/val/test) | `LGBM_Alpha_Model_V3.py:51-56` |
| **Dataset Sizes** | ~51,819 instances (2D-100D, n=10-1000) | `lgbm_model_v3/README.md:32` |
| **Split Ratios** | 70% train / 20% val / 10% test | `feature_creator_v3.py:246-274` |
| **Special Treatment** | d=100 locked to test (OOD evaluation) | `feature_creator_v3.py:251-256` |
| **Early Stopping** | 100 rounds on validation RMSE | `LGBM_Alpha_Model_V3.py:32, 84` |
| **Max Boosting Rounds** | 3000 (limited by Hyperband) | `LGBM_Alpha_Model_V3.py:34` |

### 1.2 Best Hyperparameters

- **learning_rate:** 0.0129 (very low; slow learning, fine-grained fit)
- **num_leaves:** 142 (moderate tree complexity)
- **lambda_l2:** 1.43 (moderate L2 ridge penalty)
- **lambda_l1:** 8.49e-8 (near zero; L1 not utilized)
- **feature_fraction:** 0.657 (~66% of features per tree)
- **bagging_fraction:** 0.432 (~43% of samples per boosting round)
- **bagging_freq:** 5 (every 5 rounds)
- **min_child_samples:** 22 (leaf sample minimum)

### 1.3 Feature Count

**29 features total:** Metadata (2), Geometric (3), Centroid distribution (4), MST edge statistics (11), MST clustering (3), MST topology (6).

---

## 2. Identified V3 Weaknesses (Ranked by Impact)

### HIGH-Impact Issues

**1. Non-Rotation-Invariant Bounding Box**
- **Issue:** `bounding_hypervolume = prod(ptp(coords, axis=0))` is axis-aligned. Slanted clusters score identically to isotropic squares.
- **Evidence:** GART 1.0 (rotation-invariant ConvexHull) achieved MAPE 7.35% / SDPE **7.95%**. GART V3 achieved MAPE 4.98% / SDPE **8.14%** — precision regression on anisotropic data. | `lgbm_model_v4/README.md:10-15`
- **File/Line:** `feature_creator_v3.py:120-128`

**2. MST-Only Features (Missing Tour Information)**
- **Issue:** V3 has no sequential-order features. Identical MST topologies with different greedy-NN tour lengths score the same.
- **Impact:** Cannot distinguish point sets by sequencing quality.
- **V4 Fix:** `greedy_nn_over_mst` feature encodes tour upper bound. | `lgbm_model_v4/README.md:63-70`

**3. Missing Point-Pattern Clustering Axis**
- **Issue:** No feature distinguishes grid (regular) from clustered from uniform point sets with identical MST/centroid stats.
- **Impact:** Instance class (regularity vs clustering) affects TSP difficulty; this axis is orthogonal to all V3 features.
- **V4 Fix:** `ripley_L_dev` (Ripley's L statistic in log-space). Separates grid (−0.1) from uniform (0) from clustered (+5.5). | `lgbm_model_v4/README.md:54-61`

**4. Suboptimal Loss Function (L2 Regression Only)**
- **Issue:** RMSE optimizes mean error; no explicit variance control or SDPE minimization.
- **Impact:** High-variance instances underfitted; prediction intervals not part of training signal.
- **V4 Fix:** Multi-objective Pareto optimization (MAPE, SDPE) with MAPE ≤ 5.5% hard constraint. | `lgbm_model_v4/README.md:172-175`

### MEDIUM-Impact Issues

**5. Dimension-Capped Local Density (Mitigated in V3 via MST)**
- **Historical Issue:** Original design (removed) used cKDTree-based k-NN, capped at d ≤ 16, emitted NaN beyond.
- **V3 Status:** Already works around this using MST statistics instead (no explicit k-NN features).
- **V4 Enhancement:** Explicit MST-based 1-NN / 2-NN block (Tier 2). Costs O(n) after MST, fully defined at all d. | `lgbm_model_v4/README.md:37-48`

**6. No Interaction Terms**
- **Issue:** V3 features are all univariate. Missing interactions like `mst_edge_std * aspect_ratio` or `dominance_ratio * n_customers`.
- **Status:** V3 relies on booster auto-interactions; not explicitly engineered.

**7. Very Low Learning Rate (0.0129)**
- **Rationale:** Slow learning on 51K samples reduces overfit.
- **Trade-off:** Conservative fitting; may underfit SDPE-sensitive tails.

---

## 3. V4 Improvements Proposed (Ranked by Expected Impact)

### TIER 1: Precision-Critical (HIGH)

1. **Rotation-Invariant Bounding Box (PCA-OBB)** — 5 features
   - `obb_volume`, `log_obb_volume`, `obb_shrinkage`, `pca_e1_share`, `pca_effective_rank`
   - Fixes anisotropy blind spot; direct cause of V3 precision loss vs GART 1.0.

2. **Multi-Objective Training (MAPE + SDPE)**
   - Switches from L2 to Pareto optimization; explicitly minimizes variance.
   - Hard constraint: MAPE ≤ 5.5% (V3 baseline); minimize SDPE within budget.

3. **Point-Pattern Clustering Feature (Ripley's L)**
   - `ripley_L_dev = L(r) / r - 1` in log-space.
   - New shape axis; separates grid/uniform/clustered cleanly.

4. **Tour-Aware Upper Bound**
   - `greedy_nn_over_mst = greedy_nn_tour_length / mst_total_length`
   - Only feature encoding sequential structure; orthogonal to all geometry/topology.

### TIER 2: Robustness & Coverage (MEDIUM-HIGH)

5. **MST-Based Local Density (1-NN / 2-NN)**
   - 4 features: `mst_nn1_mean`, `mst_nn1_cv`, `mst_nn2_proxy_mean`, `mst_nn_gap_ratio`
   - Replaces cKDTree-based k-NN; works at all d (including d=100).

6. **Second-Order Shape Features**
   - 2 features: `pca_log_det` (Mahalanobis volume), `mst_edge_pca_e1_share` (edge anisotropy).

### TIER 3: Hyperparameter Tuning (MEDIUM)

7. **Extend Learning Rate Search** — [0.005, 0.03] (V3 was at floor 0.01).
8. **Tune Early Stopping Rounds** — [50, 200] (V3 fixed at 100).
9. **Explore Monotonic Constraints** — enforce alpha mono-increasing in n_customers, dimension.

### TIER 4: Future (Post-V4)

10. **Retarget on OPT / 1-Tree (Held-Karp)** — GART 3.0 candidate. Tighter lower bound; reduces residual variance.
11. **Multi-Start Greedy-NN** — k-starts for tighter upper bound.

---

## 4. Suggested Hyperparameter V4 Search Space

```python
optuna_space = {
    'learning_rate': (0.005, 0.03),      # Extended lower; V3 at floor
    'num_leaves': (64, 256),             # Tightened from 512
    'lambda_l1': (1e-8, 10.0),           # Keep
    'lambda_l2': (1e-8, 10.0),           # Keep
    'feature_fraction': (0.4, 1.0),      # Keep
    'bagging_fraction': (0.3, 0.9),      # Wider range
    'bagging_freq': (1, 10),             # Extended from 7
    'min_child_samples': (5, 100),       # Extended from 50
    'early_stopping_rounds': (50, 200),  # NEW
}

# Multi-objective
directions = ['minimize', 'minimize']  # (MAPE, SDPE)
mape_cutoff = 5.5  # V3 2D baseline; V4 must not regress

# Optuna config
sampler = TPESampler(multivariate=True, group=True)
pruner = HyperbandPruner(min_resource=100, max_resource=5000)
```

---

## 5. Suggested New Features with Justification

| Feature | Compute (ms) | Tier | Justification | Priority |
|---------|--------------|------|---------------|----------|
| `obb_volume` | ~8 | 1 | Rotation-invariant bounding box | HIGH |
| `pca_e1_share` | ~8 | 1 | First PC energy fraction | HIGH |
| `pca_effective_rank` | ~8 | 1 | Intrinsic dimensionality (stable at d=100) | HIGH |
| `mst_nn1_mean` | ~2 | 2 | Exact 1-NN via MST cut; works at all d | HIGH |
| `ripley_L_dev` | ~15 | 4 | Clustering vs regularity | HIGH |
| `greedy_nn_over_mst` | ~30 | 5 | Tour upper bound; only tour feature | HIGH |

**Total new:** ~100 ms/instance cached; marginal amortized cost.

---

## 6. Feature-Pruning Targets

Candidates for removal (high redundancy, low information gain):

| Feature | Redundancy | Confidence |
|---------|-----------|-----------|
| `log_bounding_hypervolume` | Log of already-computed volume | HIGH |
| `log_node_density` | Log of already-computed density | HIGH |
| `aspect_ratio` | Subsumed by Tier-1 PCA features | MEDIUM |
| `dimension` | Possibly correlated with density | MEDIUM |

**Strategy:** Run `feature_analysis.py` forward-selection ablation. Flag features with SDPE gain < 0.01% for removal. Never drop without evidence.

---

## 7. Expected V4 Improvements

- **MAPE:** ≤ 5.5% (hold V3 2D baseline, no regression)
- **SDPE:** < 8.0% (tighter than V3's 8.14%; target ~7.9% to match GART 1.0)
- **Higher-d generalization:** MST-NN + Ripley L + OBB should improve d > 2 robustness.
- **Data integrity:** Re-solve 101 seed-recovered instances before training.

---

**Files Referenced:** LGBM_Alpha_Model_V3.py, lgbm_estimator_v3.py, feature_creator_v3.py, best_params_v3.json, lgbm_model_v4/README.md, lgbm_model_v4/feature_engineering.py

**Research Depth:** Very Thorough — all source files read, V3 architecture analyzed, V4 roadmap integrated.
