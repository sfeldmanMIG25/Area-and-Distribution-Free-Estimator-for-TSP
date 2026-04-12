# Neural Network V3 - PyTorch Alpha Estimator

Deep learning approach using a PyTorch neural network with QuantileTransformer
preprocessing and weighted sampling for robust alpha prediction.

## Files

| File | Purpose |
|------|---------|
| `train_v3.py` | Training pipeline with Optuna hyperparameter search (100 trials), weighted sampling, and 250-epoch training. |
| `estimator_v3.py` | Production estimator class (`TSP_V3_Neural_Estimator`) with StableV3Scaler wrapper. |
| `d2_test.py` | 2D benchmark test script. |

## Dependencies

Requires PyTorch in addition to the standard scientific stack.

## Usage

```python
from nn_est_alpha_v3.estimator_v3 import TSP_V3_Neural_Estimator
estimator = TSP_V3_Neural_Estimator('path/to/nn_est_alpha_v3')
result = estimator.estimate(coordinates, dimension=2, grid_size=1000)
```
