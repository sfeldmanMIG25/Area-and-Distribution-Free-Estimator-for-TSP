# LGBM V3 (GART 2.0) - LightGBM Alpha Estimator

The primary model for TSP tour length estimation. Predicts an adjustment
coefficient alpha = optimal_cost / MST_length, enabling fast tour length
estimation as `alpha * MST_length`.

## Files

| File | Purpose |
|------|---------|
| `LGBM_Alpha_Model_V3.py` | Training pipeline with Optuna hyperparameter search (100 trials). Loads `tsp_features_v3.csv`, splits into train/val/test, tunes LightGBM, and saves the final model. |
| `lgbm_estimator_v3.py` | Production estimator class (`TSP_V3_LGBM_Estimator`). Computes the 29-feature V3 feature set on-the-fly with Numba acceleration, loads the trained model, and returns predictions. |
| `LGBM_Alpha_Model_V3_2d_test.py` | 2D-specific benchmark test variant. |
| `quick_check_lgmb_v3.py` | Quick validation script for sanity-checking model loads. |
| `lgbm_alpha_model_v3.joblib` | Trained LightGBM model artifact (~50 MB). |
| `best_params_v3.json` | Best hyperparameters from Optuna search. |
| `feature_importance_v3.png` | Feature importance plot (top 30 features by split count). |

## Usage

```python
from lgbm_model_v3.lgbm_estimator_v3 import TSP_V3_LGBM_Estimator

estimator = TSP_V3_LGBM_Estimator('path/to/lgbm_model_v3')
result = estimator.estimate(coordinates, dimension=2, grid_size=1000)
# result = {'estimate': float, 'alpha': float, 'mst_length': float,
#           'feature_time': float, 'inference_time': float}
```

## Training details

- **Dataset:** 51,819 synthetic instances (2D-100D, n=10-1000, various distributions)
- **Target:** alpha clipped to [1.0, 2.0]
- **Tuning:** 100 Optuna trials minimizing validation RMSE
- **Final model:** Trained on train+val sets at the optimal iteration count
- **Key hyperparameters:** lr=0.0129, num_leaves=142, feature_fraction=0.66, bagging_fraction=0.43

## Feature set (29 features)

The V3 feature set is dimension-agnostic and MST-centric:

- **Metadata (2):** n_customers, dimension
- **Geometric spread (3):** bounding_hypervolume, node_density, aspect_ratio
- **Centroid distribution (4):** mean, std, max, IQR of centroid distances
- **MST edge statistics (11):** total_length, mean, std, skew, kurtosis, max, q10/q25/q50/q75/q90
- **MST clustering proxies (3):** dominance_ratio, gap_ratio, large_edge_count
- **MST topology (6):** leaf_ratio, degree_mean/std/max, diameter, diameter_normalized
