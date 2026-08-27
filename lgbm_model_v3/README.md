# lgbm_model_v3 — GART 2.0 (released) and the legacy V3 model

This directory holds two models. **GART 2.0 is the released model.** The V3
files are legacy. Do not use the V3 files for new work.

## GART 2.0 (released model)

GART 2.0 predicts the ratio alpha = tour cost / MST length, and returns the
estimate `alpha * MST_length`. The frozen booster has 31 input features and
1,118 trees. It uses a logit-transformed target and monotone constraints.

| File | Purpose |
|------|---------|
| `gart2_final.joblib` | The frozen released booster. The paper's results come from this file. |
| `gart2_final.json` | Sidecar with the booster's metadata (features, trees, split sizes). |
| `lgbm_estimator_gart2.py` | Released wrapper class `TSP_GART2_Estimator`. Loads `gart2_final.joblib` and computes features on the fly. |
| `feature_engineering_gart2.py` | Feature computation for the 31-feature set. |
| `train_gart2.py`, `train_gart2_logit.py`, `train_gart2_monotone.py` | Training scripts, in the order the model developed. |
| `freeze_gart2_final.py` | Freezes the trained booster and writes the sidecar. |
| `gart2_optuna.db` | Optuna study for the hyperparameter search the paper reports and rejects. |
| `gart2_logit_*.joblib`, `gart2_mono_*.joblib` | Control boosters for the paper's tuning and constraint comparisons. |

### Usage

```python
from lgbm_model_v3.lgbm_estimator_gart2 import TSP_GART2_Estimator

estimator = TSP_GART2_Estimator()
result = estimator.estimate(coordinates, dimension=2)
# result contains: estimate, alpha, mst_length, timings
```

## Legacy V3 model (do not use)

The V3 model is the predecessor of GART 2.0. It is kept so that old results
stay reproducible. It is not the model the paper calls GART 2.0.

| File | Purpose |
|------|---------|
| `LGBM_Alpha_Model_V3.py` | Legacy V3 training pipeline (Optuna search on `tsp_features_v3.csv`). |
| `lgbm_estimator_v3.py` | Legacy V3 wrapper class `TSP_V3_LGBM_Estimator` (29 features). |
| `LGBM_Alpha_Model_V3_2d_test.py` | Legacy 2D benchmark test variant. |
| `lgbm_alpha_model_v3.joblib` | Legacy V3 model artifact. |
| `best_params_v3.json`, `optuna_study_v3.db` | Legacy V3 search outputs. |
| `feature_importance_v3.png`, `test_metrics_v3.json` | Legacy V3 diagnostics. |
