"""Independent contamination audit of arm A.

Adversarial re-derivation. Nothing here reuses support_arms_* summary outputs;
every number is recomputed from the raw feature cache, the corpus table and the
booster artifacts. Writes only paper_tooling/armA_verify_*.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
OUT = {}

FEATS = list(joblib.load(ROOT / "lgbm_model_v3" / "gart2_final.joblib").feature_name())

# ---------------------------------------------------------------- data ----
C = pd.read_csv(HERE / "v4_study_feature_cache.csv", low_memory=False)
V4 = pd.read_csv(ROOT / "tsp_features_v4.csv", low_memory=False)
AUGMETA = pd.read_csv(HERE / "augment_features_v3.csv",
                      usecols=["instance_name", "family", "group", "record_rho",
                               "n_customers", "dimension"])
GEN = pd.read_csv(HERE / "augmentation_2d_features.csv",
                  usecols=["instance_name", "generator", "gen_class"])

C = C.merge(GEN.rename(columns={"instance_name": "instance",
                                "generator": "gen2"}),
            on="instance", how="left")
C["fam"] = C["gen2"].fillna(C.get("generator", pd.Series(index=C.index, dtype=object)))

strata = {s: g for s, g in C.groupby("stratum")}
print("[strata]", {k: len(v) for k, v in strata.items()})

aug = strata["augment"].copy()
b2d = strata["bench2d"].copy()
aug = aug.drop(columns=[c for c in ("family", "group", "record_rho")
                        if c in aug.columns])
aug = aug.merge(AUGMETA.rename(columns={"instance_name": "instance"})[
    ["instance", "family", "group", "record_rho"]], on="instance", how="left")
print("[augment families]", aug["family"].value_counts().to_dict())

# ------------------------------------------------- 1. identifier overlap ---
names = {s: set(g["instance"].astype(str)) for s, g in C.groupby("stratum")}
corpus = {sp: set(V4.loc[V4.split == sp, "instance_name"].astype(str))
          for sp in ("train", "val", "test")}
ov = {}
for s, ns in names.items():
    for sp, cs in corpus.items():
        ov[f"{s}__corpus_{sp}"] = len(ns & cs)
for a, b in (("augment", "bench2d"), ("augment", "tsplib_euc2d"),
             ("augment", "tsplib_noneuc"), ("augment", "nd_test"),
             ("bench2d", "nd_test")):
    if a in names and b in names:
        ov[f"{a}__{b}"] = len(names[a] & names[b])
OUT["identifier_overlap"] = ov
print("[overlap]", json.dumps(ov, indent=1))

# augment names must also be absent from the raw v4 table entirely
OUT["augment_names_in_v4_any_split"] = int(
    len(names["augment"] & set(V4["instance_name"].astype(str))))

# ------------------------------------------ 2. corpus train feature pool ---
tr = V4[V4.split == "train"].copy()
va = V4[V4.split == "val"].copy()
OUT["corpus_split_sizes"] = {"train": len(tr), "val": len(va),
                             "test": int((V4.split == "test").sum())}

pool = {
    "corpus_train": tr[FEATS].to_numpy(float),
    "augment": aug[FEATS].to_numpy(float),
    "bench2d": b2d[FEATS].to_numpy(float),
}
for k, v in pool.items():
    print(f"[pool] {k}: {v.shape}, nan={np.isnan(v).sum()}")

# --------------------------------------------- 3. metric M1: ECDF ranks ----
# Normalisation reference = the pooled corpus TRAIN distribution only, so the
# benchmark and the augmentation are both mapped through a yardstick that
# neither of them defines.
ref = pool["corpus_train"]


def ecdf_transform(X: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        r = np.sort(ref[:, j])
        out[:, j] = np.searchsorted(r, X[:, j], side="left") / len(r)
    return out


Z = {k: ecdf_transform(v) for k, v in pool.items()}


def nn_dist(A: np.ndarray, B: np.ndarray, exclude_self: bool = False,
            chunk: int = 512) -> np.ndarray:
    """Min Euclidean distance from each row of A to any row of B."""
    best = np.full(len(A), np.inf)
    b2 = (B ** 2).sum(1)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        d2 = (a ** 2).sum(1)[:, None] + b2[None, :] - 2.0 * a @ B.T
        np.maximum(d2, 0.0, out=d2)
        if exclude_self:
            for k in range(len(a)):
                d2[k, i + k] = np.inf
        best[i:i + chunk] = np.sqrt(d2.min(1))
    return best


def nn_arg(A: np.ndarray, B: np.ndarray, chunk: int = 512):
    best = np.full(len(A), np.inf)
    idx = np.zeros(len(A), dtype=int)
    b2 = (B ** 2).sum(1)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        d2 = (a ** 2).sum(1)[:, None] + b2[None, :] - 2.0 * a @ B.T
        np.maximum(d2, 0.0, out=d2)
        j = d2.argmin(1)
        best[i:i + chunk] = np.sqrt(d2[np.arange(len(a)), j])
        idx[i:i + chunk] = j
    return best, idx


d_aug, i_aug = nn_arg(Z["bench2d"], Z["augment"])
d_trn = nn_dist(Z["bench2d"], Z["corpus_train"])
d_own = nn_dist(Z["bench2d"], Z["bench2d"], exclude_self=True)

b2d = b2d.reset_index(drop=True)
b2d["nn_aug"] = d_aug
b2d["nn_train"] = d_trn
b2d["nn_own"] = d_own
b2d["nn_aug_name"] = aug["instance"].to_numpy()[i_aug]
b2d["nn_aug_family"] = aug["family"].to_numpy()[i_aug]

rows = []
for fam, g in b2d.groupby("fam"):
    rows.append({
        "family": fam, "n": len(g),
        "nn_aug_min": g.nn_aug.min(), "nn_aug_p01": g.nn_aug.quantile(.01),
        "nn_aug_p05": g.nn_aug.quantile(.05), "nn_aug_med": g.nn_aug.median(),
        "nn_train_min": g.nn_train.min(), "nn_train_p05": g.nn_train.quantile(.05),
        "nn_train_med": g.nn_train.median(),
        "nn_own_min": g.nn_own.min(), "nn_own_med": g.nn_own.median(),
        "ratio_med_aug_over_train": g.nn_aug.median() / g.nn_train.median(),
        "ratio_med_aug_over_own": g.nn_aug.median() / g.nn_own.median(),
        "n_aug_closer_than_train": int((g.nn_aug < g.nn_train).sum()),
        "n_aug_closer_than_own": int((g.nn_aug < g.nn_own).sum()),
    })
M1 = pd.DataFrame(rows).sort_values("family")
M1.to_csv(HERE / "armA_verify_nn_m1.csv", index=False)
print("\n[M1 ECDF-rank Euclidean, 31 feats]")
print(M1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ----------------------------------- 4. metric M2: leaf-kernel (arm A) -----
A = joblib.load(HERE / "support_arms_models" / "A.joblib")
NT = A.best_iteration


def leaves(X):
    return A.predict(X, num_iteration=NT, pred_leaf=True).astype(np.int32)


L_b = leaves(b2d[FEATS])
L_a = leaves(aug[FEATS])
rs = np.random.default_rng(0)
sub = rs.choice(len(tr), size=min(25000, len(tr)), replace=False)
L_t = leaves(tr.iloc[sub][FEATS])


def max_leaf_match(Lq: np.ndarray, Lr: np.ndarray, chunk: int = 64):
    """Max over reference rows of the fraction of trees sharing a leaf."""
    best = np.zeros(len(Lq))
    arg = np.zeros(len(Lq), dtype=int)
    for i in range(0, len(Lq), chunk):
        q = Lq[i:i + chunk]
        m = (q[:, None, :] == Lr[None, :, :]).mean(2)
        j = m.argmax(1)
        best[i:i + chunk] = m[np.arange(len(q)), j]
        arg[i:i + chunk] = j
    return best, arg


m_aug, j_aug = max_leaf_match(L_b, L_a)
m_trn, _ = max_leaf_match(L_b, L_t)
b2d["leafmatch_aug"] = m_aug
b2d["leafmatch_train"] = m_trn
b2d["leafmatch_aug_name"] = aug["instance"].to_numpy()[j_aug]

rows = []
for fam, g in b2d.groupby("fam"):
    rows.append({"family": fam, "n": len(g),
                 "leafmatch_aug_max": g.leafmatch_aug.max(),
                 "leafmatch_aug_p95": g.leafmatch_aug.quantile(.95),
                 "leafmatch_aug_med": g.leafmatch_aug.median(),
                 "leafmatch_train_max": g.leafmatch_train.max(),
                 "leafmatch_train_med": g.leafmatch_train.median(),
                 "n_aug_beats_train": int((g.leafmatch_aug > g.leafmatch_train).sum())})
M2 = pd.DataFrame(rows).sort_values("family")
M2.to_csv(HERE / "armA_verify_nn_m2_leaf.csv", index=False)
print("\n[M2 leaf-kernel match fraction, arm A, 1203 trees]")
print(M2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ----------------------------- 5. does the gain track proximity? -----------
PI = pd.read_csv(HERE / "support_arms_per_instance.csv")
pv = PI[PI.stratum == "bench2d"].pivot_table(index="instance", columns="model",
                                             values="err_pct")
b2d = b2d.set_index("instance")
common = b2d.index.intersection(pv.index)
b2d = b2d.loc[common]
pv = pv.loc[common]
b2d["ape_frozen"] = pv["FROZEN"].abs()
b2d["ape_A"] = pv["A"].abs()
b2d["gain"] = b2d.ape_frozen - b2d.ape_A

dec_rows = []
for fam in ("line_noise", "grid", None):
    g = b2d if fam is None else b2d[b2d.fam == fam]
    q = pd.qcut(g.nn_aug, 5, labels=False, duplicates="drop")
    for k in sorted(pd.unique(q.dropna())):
        s = g[q == k]
        dec_rows.append({"family": fam or "ALL_2580", "quintile": int(k) + 1,
                         "n": len(s), "nn_aug_med": s.nn_aug.median(),
                         "ape_frozen": s.ape_frozen.mean(),
                         "ape_A": s.ape_A.mean(), "gain": s.gain.mean()})
M3 = pd.DataFrame(dec_rows)
M3.to_csv(HERE / "armA_verify_gain_by_proximity.csv", index=False)
print("\n[M3 arm-A gain by distance-to-nearest-augment quintile]")
print(M3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

for fam in ("line_noise", "grid"):
    g = b2d[b2d.fam == fam]
    from scipy import stats
    r = stats.spearmanr(g.nn_aug, g.gain)
    rl = stats.spearmanr(g.leafmatch_aug, g.gain)
    OUT[f"spearman_gain_vs_nnaug_{fam}"] = [float(r.statistic), float(r.pvalue)]
    OUT[f"spearman_gain_vs_leafmatch_{fam}"] = [float(rl.statistic), float(rl.pvalue)]

# ------------------------------------------- 6. the extreme cases ----------
ext = b2d.sort_values("nn_aug").head(15)[
    ["fam", "n_customers", "dimension", "nn_aug", "nn_train", "nn_own",
     "nn_aug_name", "nn_aug_family", "leafmatch_aug", "leafmatch_train",
     "ape_frozen", "ape_A", "gain"]]
ext.to_csv(HERE / "armA_verify_extremes_m1.csv")
print("\n[closest 15 benchmark instances to any augmentation instance, M1]")
print(ext.to_string(float_format=lambda x: f"{x:.4f}"))

ext2 = b2d.sort_values("leafmatch_aug", ascending=False).head(15)[
    ["fam", "n_customers", "dimension", "leafmatch_aug", "leafmatch_train",
     "nn_aug", "nn_train", "leafmatch_aug_name", "ape_frozen", "ape_A", "gain"]]
ext2.to_csv(HERE / "armA_verify_extremes_m2.csv")
print("\n[highest leaf-kernel overlap with an augmentation instance]")
print(ext2.to_string(float_format=lambda x: f"{x:.4f}"))

json.dump(OUT, open(HERE / "armA_verify_contamination.json", "w"), indent=2)
print("\n[done]")
