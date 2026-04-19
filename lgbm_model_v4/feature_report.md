# V4 feature-candidate report

Baseline (V3 feature set): SDPE = **1.415%**, MAPE = **0.802%** on val split.

Filter rule: `p(F-reg) < 0.01`, `MI >= 0.01`, `|delta SDPE| >= 0.05pp` (or strong MI), `p95 <= 10.0ms`.

| Feature | p(F-reg) | MI | Spearman | p50 ms | p95 ms | delta SDPE pp | delta MAPE pp | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `obb_volume` | 1.00e+00 | 0.0000 | -0.849 | 0.17 | 3.65 | -0.005 | -0.002 | **DROP** — p=1.00e+00 >= 0.01 | MI=0.0000 < 0.01 | |dSDPE|=0.005pp < 0.05 |
| `log_obb_volume` | 0.00e+00 | 1.0668 | -0.849 | 0.17 | 3.65 | -0.013 | -0.007 | **KEEP** — auto-keep (strong MI) |
| `obb_shrinkage` | 1.61e-03 | 0.0139 | -0.700 | 0.17 | 3.65 | +0.009 | +0.002 | **DROP** — |dSDPE|=0.009pp < 0.05 |
| `pca_e1_share` | 0.00e+00 | 0.8446 | +0.875 | 0.02 | 0.27 | -0.004 | -0.005 | **KEEP** — auto-keep (strong MI) |
| `pca_effective_rank` | 0.00e+00 | 1.1591 | -0.887 | 0.02 | 0.27 | -0.005 | -0.005 | **KEEP** — auto-keep (strong MI) |
| `mst_nn1_mean` | 1.84e-264 | 0.4674 | -0.182 | 0.03 | 0.11 | -0.009 | -0.006 | **KEEP** — auto-keep (strong MI) |
| `mst_nn1_cv` | 0.00e+00 | 0.7155 | +0.491 | 0.03 | 0.11 | -0.007 | -0.010 | **KEEP** — auto-keep (strong MI) |
| `mst_nn2_proxy_mean` | 2.42e-193 | 0.4960 | -0.159 | 0.03 | 0.11 | -0.005 | -0.006 | **KEEP** — auto-keep (strong MI) |
| `mst_nn_gap_ratio` | 0.00e+00 | 0.6539 | +0.612 | 0.03 | 0.11 | -0.004 | -0.002 | **KEEP** — auto-keep (strong MI) |
| `pca_log_det` | 0.00e+00 | 0.9656 | -0.815 | 0.02 | 0.27 | -0.003 | -0.006 | **KEEP** — auto-keep (strong MI) |
| `mst_edge_pca_e1_share` | 0.00e+00 | 1.3512 | +0.939 | 0.08 | 1.14 | -0.014 | -0.003 | **KEEP** — auto-keep (strong MI) |
| `ripley_L_dev` | 0.00e+00 | 0.5843 | -0.254 | 0.70 | 24.86 | -0.010 | -0.010 | **DROP** — |dSDPE|=0.010pp < 0.05 | p95=24.86ms > 10.0 |
| `greedy_nn_over_mst` | 0.00e+00 | 1.6742 | +0.939 | 0.76 | 9.31 | -0.256 | -0.165 | **KEEP** — auto-keep |

## Notes
- `delta SDPE pp` is the change in val-set SDPE (percentage points) when this feature is *added* to the V3 baseline. Negative is better.
- All timings are on a stratified sample of 200 instances (10 per dimension class), including the amortised cost of shared primitives (PCA eigendecomposition, MST-incident-edge sort).
- All 40 candidate features are defined at every dimension — the Tier-2 local-density block is derived from MST incident edges, which avoids the concentration-of-measure collapse that makes kd-tree k-NN unreliable past d ~ 16.

After review, edit `selected_features.json` directly (or delete a candidate from it) before running `train.py`. The training script trusts this file as the final feature list.