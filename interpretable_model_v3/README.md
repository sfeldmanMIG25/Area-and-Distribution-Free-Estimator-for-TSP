# Interpretable V3 - Mixture-of-Experts Model

A mixture-of-experts model combining interpretability with gradient-boosted
feature interactions. Uses a decision tree router (depth 3) to gate input
instances to specialized expert models, providing transparent routing logic
while leveraging LightGBM-derived features.

## Files

| File | Purpose |
|------|---------|
| `train_interpretable_v3.py` | Training script that builds a routing tree + expert models from LGBM feature interactions. |
| `estimator_interpretable_v3.py` | Production estimator class (`TSP_Interpretable_Estimator`). |
| `d2_test_interpretable.py` | 2D benchmark test script. |
| `model_artifacts/` | Saved routing tree and expert model components. |

## Usage

```python
from interpretable_model_v3.estimator_interpretable_v3 import TSP_Interpretable_Estimator
estimator = TSP_Interpretable_Estimator('path/to/interpretable_model_v3')
result = estimator.estimate(coordinates, dimension=2, grid_size=1000)
```
