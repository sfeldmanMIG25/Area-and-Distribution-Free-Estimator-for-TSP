# Linear V3 - Linear Regression Baseline

Baseline linear regression model using the V3 feature set with log-transformed
features and standardization. Serves as a lower-bound comparison for the
gradient-boosted models.

## Files

| File | Purpose |
|------|---------|
| `train_linear_v3.py` | Training script with sklearn Pipeline (StandardScaler + SimpleImputer). Applies robust log transforms to explosive features (hypervolume, node_density). |
| `estimator_linear_v3.py` | Production estimator class (`TSP_V3_Linear_Estimator`) with Numba-accelerated feature computation. |
| `d2_test_linear.py` | 2D benchmark test script. |
| `linear_alpha_model_v3.joblib` | Trained linear model artifact. |

## Usage

```python
from linear_model_v3.estimator_linear_v3 import TSP_V3_Linear_Estimator
estimator = TSP_V3_Linear_Estimator('path/to/linear_model_v3')
result = estimator.estimate(coordinates, dimension=2, grid_size=1000)
```
