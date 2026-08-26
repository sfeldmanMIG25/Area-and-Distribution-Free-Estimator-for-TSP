# `features_extended.csv` — partially superseded, retained deliberately

Status as of 2026-08-11. **Do not delete this file, and do not repoint its
consumers wholesale.** The supersession is column-scoped, not file-scoped.

## What the file is

`paper_tooling/features_extended.csv` — 106,272 rows (one per corpus instance),
six columns, written by `paper_tooling/feature_retrain.py` (`extract` stage,
cached; `--force` to rebuild):

| column | status |
|---|---|
| `instance_name` | key, current |
| `mst_topology_straightness` | **SUPERSEDED** |
| `mst_topology_deg2_straight_mean` | superseded in principle, numerically unchanged |
| `degeneracy_pca_effective_rank` | **current — this file is the only source** |
| `local_id_evr1_median_k5` | **current — this file is the only source** |
| `local_id_pr_mean_k5` | **current — this file is the only source** |

## What changed underneath it

The `features_ext` MST-topology extractors were repaired for permutation
dependence and MST tie-degeneracy (`features_ext/group_mst_topology.py`). The
values in this file predate that repair. Measured against the replacement:

| feature | rows changed (>1e-9) | max abs delta |
|---|---|---|
| `mst_topology_straightness` | 8 of 106,272 | 0.0216 |
| `mst_topology_deg2_straight_mean` | 0 | 2.2e-16 |

The degeneracy and local-intrinsic-dimension columns were **not** touched by the
repair, so they are not stale.

## Replacement

`paper_tooling/support_arms_feats_corpus.csv` — same 106,272 rows, but only
`instance_name` plus the two repaired `mst_topology_*` columns. It does **not**
carry the three degeneracy / local-ID columns, which is why this file cannot be
retired.

## Consumers, and why each is left alone

| consumer | uses | action |
|---|---|---|
| `feature_retrain.py` | producer + reader (cache) | none — rebuild with `--force` if the repaired `mst_topology_*` values are wanted here |
| `feature_reproduce.py` | reader | none — it audits the corpus as shipped |
| `feature_adversarial.py` (`a5`) | reader, name set only | none — it never reads the values |
| `support_arms_features.py` (`diff_vs_cached`) | reader | none — this file **is** the intended pre-repair snapshot |
| `support_arms_study.py` (`EXT_OLD`) | reader, degeneracy group | none — those columns are current |

No manuscript table, figure or quoted scalar is derived from this file.
