"""Causal test: which augmentation families actually buy the 2D gates?

Refits arm A's exact recipe with augmentation subsets removed, and adds a
held-out-augmentation arm so the 0.627 augment MAPE can be read as fit vs
memorisation.

    A_full     corpus train + all 874 augment rows            (= arm A)
    A_nolat    corpus train + augment minus lattice/hexlattice
    A_no1d     corpus train + augment minus the 1-D families
    A_half     corpus train + a random 50% of augment; the other 50% is scored
               OUT OF SAMPLE

Everything else -- features, target, monotone constraints, hyperparameters,
seed, early stopping on the untouched val split -- is imported unchanged from
support_arms_study.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from support_arms_study import (fit_arm, frozen_features, load_corpus,   # noqa
                                load_augment, to_alpha, ALPHA_CLIP)

MDL = HERE / "armA_verify_lofo_models"
MDL.mkdir(exist_ok=True)

ONE_D = ("collinear", "subspace", "curve", "filament", "polyline")
LAT = ("lattice", "hexlattice")

feats = frozen_features()
corpus = load_corpus(feats)
aug = load_augment(feats)
meta = pd.read_csv(HERE / "augment_features_v3.csv",
                   usecols=["instance_name", "family"])
aug = aug.drop(columns=[c for c in ["family"] if c in aug.columns]).merge(
    meta, on="instance_name", how="left", validate="one_to_one")
print("[augment families]", aug.family.value_counts().to_dict())

tr = corpus[corpus.split == "train"]
va = corpus[corpus.split == "val"]
cols = tr.columns.intersection(aug.columns)

rs = np.random.default_rng(0)
half = rs.permutation(len(aug))[: len(aug) // 2]
in_half = aug.index.isin(aug.index[half])

SUBSETS = {
    "A_full":  aug,
    "A_nolat": aug[~aug.family.isin(LAT)],
    "A_no1d":  aug[~aug.family.isin(ONE_D)],
    "A_half":  aug[in_half],
}
HELDOUT = {"A_half": aug[~in_half]}

models = {}
for tag, sub in SUBSETS.items():
    p = MDL / f"{tag}.joblib"
    if p.exists():
        models[tag] = joblib.load(p)
        print(f"[fit:{tag}] cached ({len(sub)} aug rows), "
              f"best_iter={models[tag].best_iteration}")
        continue
    trx = pd.concat([tr, sub[cols]], ignore_index=True, sort=False)
    assert set(sub.instance_name).isdisjoint(set(va.instance_name))
    print(f"[fit:{tag}] {len(sub)} augment rows -> {len(trx)} train rows")
    b = fit_arm(trx, va, feats, tag)
    joblib.dump(b, p)
    models[tag] = b

models["FROZEN"] = joblib.load(ROOT / "lgbm_model_v3" / "gart2_final.joblib")
models["A_ship"] = joblib.load(HERE / "support_arms_models" / "A.joblib")

# ------------------------------------------------------------ scoring ----
C = pd.read_csv(HERE / "v4_study_feature_cache.csv", low_memory=False)
gen = pd.read_csv(HERE / "augmentation_2d_features.csv",
                  usecols=["instance_name", "generator"])
C = C.merge(gen.rename(columns={"instance_name": "instance"}), on="instance",
            how="left")
C = C[(C.status == "ok") & C[feats].notna().all(axis=1) & C.true_cost.notna()]
b2 = C[C.stratum == "bench2d"].copy()
nd = C[C.stratum == "nd_test"].copy()
tl = C[C.stratum == "tsplib_euc2d"].copy()


def pred(m, frame):
    a = np.clip(to_alpha(m.predict(frame[feats], num_iteration=m.best_iteration)),
                *ALPHA_CLIP)
    return a, (a * frame.mst_total_length.to_numpy() - frame.true_cost.to_numpy()
               ) / frame.true_cost.to_numpy() * 100.0


rows = []
for tag, m in models.items():
    _, e2 = pred(m, b2)
    _, en = pred(m, nd)
    _, et = pred(m, tl)
    g = b2.generator.to_numpy()
    ln = b2[(b2.generator == "line_noise") & (b2.n_customers >= 200)]
    a_ln, _ = pred(m, ln)
    true_a = np.clip(ln.true_cost / ln.mst_total_length, *ALPHA_CLIP).to_numpy()
    slope = np.polyfit(true_a, a_ln, 1)[0]
    r = {"model": tag,
         "bench2d_MAPE": np.abs(e2).mean(),
         "grid_MSPE": e2[g == "grid"].mean(),
         "grid_MAPE": np.abs(e2[g == "grid"]).mean(),
         "linenoise_MAPE": np.abs(e2[g == "line_noise"]).mean(),
         "linenoise_slope_N90": slope,
         "other11_MAPE": np.abs(e2[~np.isin(g, ["grid", "line_noise"])]).mean(),
         "nd_test_MAPE": np.abs(en).mean(),
         "nd_test_SDPE": np.std(en / 100.0, ddof=1) * 100.0,
         "tsplib_euc2d_MAPE": np.abs(et).mean()}
    if tag in HELDOUT:
        ho = C[C.instance.isin(set(HELDOUT[tag].instance_name))]
        _, eh = pred(m, ho)
        r["heldout_augment_MAPE"] = np.abs(eh).mean()
        r["heldout_augment_n"] = len(ho)
        ins = C[C.instance.isin(set(SUBSETS[tag].instance_name))]
        _, ei = pred(m, ins)
        r["insample_augment_MAPE"] = np.abs(ei).mean()
    else:
        aall = C[C.stratum == "augment"]
        _, ea = pred(m, aall)
        r["insample_augment_MAPE"] = np.abs(ea).mean()
    rows.append(r)

R = pd.DataFrame(rows)
R.to_csv(HERE / "armA_verify_lofo_results.csv", index=False)
pd.set_option("display.width", 250)
print("\n" + R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nGATE 6 threshold: linenoise slope >= 0.70 | "
      "GATE 7 threshold: grid MSPE <= +4.0 | "
      "GATE 1: nd MAPE <= 0.6401 and SDPE <= 1.0081")
