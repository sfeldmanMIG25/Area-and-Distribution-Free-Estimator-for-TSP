# GART 1.0 - Original Alpha Predictor

The original Generalized Alpha-Ratio TSP estimator from the thesis. Uses an
LGBM model with the V1 feature set (2D-only, fewer features).

## Files

| File | Purpose |
|------|---------|
| `alpha_predictor_model.joblib` | Trained GART 1.0 model artifact (~52 MB). |

## Notes

This model is retained for backward compatibility and benchmark comparisons.
GART 2.0 (lgbm_model_v3) supersedes it with a dimension-agnostic feature set
and improved accuracy.
