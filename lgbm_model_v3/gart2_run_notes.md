# GART 2.0 training run

- features: 31 (V3's 30 + `greedy_nn_over_mst`)
- excluded: grid_size, mst_total_length
- protocol: train-only fit, early stop on val (val never trained on)
- objective: minimise validation SDPE (cost level)
- Optuna: 60 complete trials, 37.0 min, seed 42
- picked trial #61; best_iteration 1840
- val:  SDPE=1.1458%  MAPE=0.6372%
- test: SDPE=0.9885%  MAPE=0.6222%  MSPE=1.0165pp^2  bias=+0.1985%
- MAPE-min alternative (trial #188) would give test SDPE=0.9818%  MAPE=0.6101%
