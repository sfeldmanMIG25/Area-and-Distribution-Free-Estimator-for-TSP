# V4 feature-candidate report

Baseline (V3 feature set): SDPE = **1.419%**, MAPE = **0.801%** on val split.

Filter rule: `p(F-reg) < 0.01` AND `p95 <= 10.0ms` AND (`dSDPE <= -0.05pp` OR [`dSDPE <= -0.025pp` AND `MI >= 0.050`]).
(Sign-sensitive: features that *worsen* SDPE can no longer be auto-kept by MI alone.)

| Feature | p(F-reg) | MI | Spearman | p50 ms | p95 ms | delta SDPE pp | delta MAPE pp | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `obb_volume` | 1.00e+00 | 0.0000 | -0.876 | 0.14 | 5.11 | -0.002 | +0.001 | **DROP** — p=1.00e+00 >= 0.01 | dSDPE=-0.002pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.0000) |
| `log_obb_volume` | 0.00e+00 | 1.1103 | -0.876 | 0.14 | 5.11 | -0.006 | -0.003 | **DROP** — dSDPE=-0.006pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=1.1103) |
| `obb_shrinkage` | 1.00e+00 | 0.0000 | -0.227 | 0.14 | 5.11 | +0.011 | +0.002 | **DROP** — p=1.00e+00 >= 0.01 | dSDPE=+0.011pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.0000) |
| `pca_e1_share` | 0.00e+00 | 0.8446 | +0.875 | 0.02 | 0.51 | -0.010 | -0.007 | **DROP** — dSDPE=-0.010pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.8446) |
| `pca_effective_rank` | 0.00e+00 | 1.1591 | -0.887 | 0.02 | 0.51 | -0.009 | -0.003 | **DROP** — dSDPE=-0.009pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=1.1591) |
| `mst_nn1_mean` | 1.84e-264 | 0.4674 | -0.182 | 0.03 | 0.09 | -0.008 | -0.007 | **DROP** — dSDPE=-0.008pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.4674) |
| `mst_nn1_cv` | 0.00e+00 | 0.7155 | +0.491 | 0.03 | 0.09 | -0.001 | -0.006 | **DROP** — dSDPE=-0.001pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.7155) |
| `mst_nn2_proxy_mean` | 2.42e-193 | 0.4960 | -0.159 | 0.03 | 0.09 | -0.004 | -0.006 | **DROP** — dSDPE=-0.004pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.4960) |
| `mst_nn_gap_ratio` | 0.00e+00 | 0.6539 | +0.612 | 0.03 | 0.09 | -0.000 | -0.003 | **DROP** — dSDPE=-0.000pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.6539) |
| `pca_log_det` | 0.00e+00 | 0.9877 | -0.833 | 0.02 | 0.51 | +0.006 | +0.003 | **DROP** — dSDPE=+0.006pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.9877) |
| `mst_edge_pca_e1_share` | 0.00e+00 | 1.3512 | +0.939 | 0.06 | 1.52 | -0.004 | +0.009 | **DROP** — dSDPE=-0.004pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=1.3512) |
| `ripley_L_dev` | 0.00e+00 | 0.4645 | -0.443 | 0.52 | 16.95 | -0.014 | -0.006 | **DROP** — p95=16.95ms > 10.0 | dSDPE=-0.014pp fails gate (need <=-0.05 or <=-0.025 with MI>=0.050; MI=0.4645) |
| `greedy_nn_over_mst` | 0.00e+00 | 1.6742 | +0.939 | 0.53 | 6.77 | -0.258 | -0.162 | **KEEP** — keep (dSDPE gate) |

## Notes
- `delta SDPE pp` is the change in val-set SDPE (percentage points) when this feature is *added* to the V3 baseline. Negative is better.
- All timings are on a stratified sample of 200 instances (10 per dimension class), including the amortised cost of shared primitives (PCA eigendecomposition, MST-incident-edge sort).
- All 40 candidate features are defined at every dimension — the Tier-2 local-density block is derived from MST incident edges, which avoids the concentration-of-measure collapse that makes kd-tree k-NN unreliable past d ~ 16.

After review, edit `selected_features.json` directly (or delete a candidate from it) before running `train.py`. The training script trusts this file as the final feature list.