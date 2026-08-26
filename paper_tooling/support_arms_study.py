"""Do support (augmentation) and three cheap features buy anything over frozen GART 2.0?

Arms, all on the frozen recipe -- logit(alpha-1) target, cost-MAPE early
stopping (100 rounds) on the untouched val split, monotone -1 on n_customers
and dimension, V3 frozen hyperparameters, seed 42, num_threads 6:

    R0  31 features, no augmentation.  Reproduction control: must be
        bit-identical to lgbm_model_v3/gart2_final.joblib, otherwise the whole
        comparison is confounded and the run aborts.
    A   31 features + augmentation rows in TRAIN ONLY.  Zero extraction cost.
    B   34 features (frozen + the three cheap ones) + the same augmentation.

The frozen artifact itself is the control, scored by this pipeline. Gates are
pre-registered in support_arms_gates.json and were written before any arm was
trained.

Augmentation rows go to TRAIN only -- never val, never test. That is asserted,
not assumed.

Writes only under paper_tooling/. Never touches lgbm_model_v3/gart2_final.*,
paper_reference/, or the production benchmark directories.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "10")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
for _p in (ROOT, HERE, ROOT / "lgbm_model_v3", ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FROZEN = ROOT / "lgbm_model_v3" / "gart2_final.joblib"
V4_TABLE = ROOT / "tsp_features_v4.csv"
FEAT_CACHE = HERE / "v4_study_feature_cache.csv"
GEN_MAP = HERE / "augmentation_2d_features.csv"

F_CORPUS = HERE / "support_arms_feats_corpus.csv"
F_AUGMENT = HERE / "support_arms_feats_augment.csv"
F_BENCH2D = HERE / "support_arms_feats_bench2d.csv"
F_TSPLIB = HERE / "support_arms_feats_tsplib.csv"
EXT_OLD = HERE / "features_extended.csv"          # degeneracy: group untouched

MODEL_DIR = HERE / "support_arms_models"
GATES_JSON = HERE / "support_arms_gates.json"

NEW3 = ["mst_topology_straightness", "mst_topology_deg2_straight_mean",
        "degeneracy_pca_effective_rank"]

SEED = 42
NUM_THREADS = 6
MAX_BOOST_ROUND = 4000
EARLY_STOP = 100
EPS = 1e-6
ALPHA_CLIP = (1.0, 2.0)

# V3's shipped hyperparameters, frozen (lgbm_model_v3/train_gart2_logit.py).
V3_FROZEN = {
    "learning_rate": 0.02597081024148143, "num_leaves": 148,
    "reg_alpha": 0.07764927124026656, "reg_lambda": 1.1865278867536643e-05,
    "feature_fraction": 0.5481439957854186, "bagging_fraction": 0.6087321895536237,
    "bagging_freq": 6, "min_child_samples": 23, "max_depth": -1,
}
MONO_COLS = ("n_customers", "dimension")


# =========================================================================
# target transform
# =========================================================================
def to_z(alpha) -> np.ndarray:
    u = np.clip(np.asarray(alpha, dtype=np.float64) - 1.0, EPS, 1.0 - EPS)
    return np.log(u / (1.0 - u))


def to_alpha(z) -> np.ndarray:
    return 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def cost_metrics(alpha_pred, mst, true_cost) -> dict:
    a = np.clip(alpha_pred, *ALPHA_CLIP)
    err = (a * mst - true_cost) / true_cost
    return {"n": int(len(err)),
            "mape": float(np.mean(np.abs(err)) * 100.0),
            "sdpe": float(np.std(err, ddof=1) * 100.0) if len(err) > 1 else float("nan"),
            "mspe": float(np.mean(err) * 100.0)}


# =========================================================================
# data
# =========================================================================
def frozen_features() -> list[str]:
    import joblib
    return list(joblib.load(FROZEN).feature_name())


def load_corpus(feats31: list[str]) -> pd.DataFrame:
    """tsp_features_v4.csv + the three arm-B columns, splits untouched."""
    df = pd.read_csv(V4_TABLE)
    before = df[["instance_name", "split"]].copy()

    topo = pd.read_csv(F_CORPUS)
    deg = pd.read_csv(EXT_OLD, usecols=["instance_name", "degeneracy_pca_effective_rank"])
    df = df.merge(topo, on="instance_name", how="left", validate="one_to_one")
    df = df.merge(deg, on="instance_name", how="left", validate="one_to_one")

    assert len(df) == len(before), "merge changed row count"
    assert (df["instance_name"].values == before["instance_name"].values).all()
    assert (df["split"].values == before["split"].values).all(), "split changed"
    assert df[NEW3].notna().all().all(), "missing arm-B features on the corpus"

    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(*ALPHA_CLIP)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)
    missing = [c for c in feats31 if c not in df.columns]
    assert not missing, f"corpus missing frozen features {missing}"
    return df


def load_augment(feats31: list[str]) -> pd.DataFrame:
    """The augmentation corpus, assembled into the same schema. TRAIN only."""
    base = pd.read_csv(HERE / "augment_features_v3.csv")
    gnn = pd.read_csv(HERE / "augment_greedy_nn.csv")
    new = pd.read_csv(F_AUGMENT)
    df = base.merge(gnn, on="instance_name", how="inner", validate="one_to_one")
    df = df.merge(new, on="instance_name", how="inner", validate="one_to_one")

    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(*ALPHA_CLIP)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)
    df["split"] = "train"

    missing = [c for c in feats31 + NEW3 if c not in df.columns]
    assert not missing, f"augment missing {missing}"
    return df


def assert_no_leakage(corpus: pd.DataFrame, aug: pd.DataFrame) -> dict:
    """Augmentation names must not appear in ANY split of the scored corpora."""
    aug_names = set(aug["instance_name"].astype(str))
    report = {"n_augment": len(aug_names)}
    for split in ("train", "val", "test"):
        names = set(corpus.loc[corpus.split == split, "instance_name"].astype(str))
        ov = aug_names & names
        report[f"overlap_corpus_{split}"] = len(ov)
        assert not ov, f"augment overlaps corpus/{split}: {sorted(ov)[:5]}"
    C = pd.read_csv(FEAT_CACHE, usecols=["stratum", "instance"], low_memory=False)
    for st, g in C.groupby("stratum"):
        ov = aug_names & set(g["instance"].astype(str))
        report[f"overlap_{st}"] = len(ov)
        if st != "augment":
            assert not ov, f"augment overlaps scored stratum {st}"
    return report


# =========================================================================
# training
# =========================================================================
def fit_arm(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], tag: str):
    import lightgbm as lgb

    mono = [-1 if f in MONO_COLS else 0 for f in feats]
    params = dict(V3_FROZEN)
    params.update({
        "objective": "regression_l2",
        "metric": "None",                 # early stopping is driven by the feval
        "boosting_type": "gbdt", "seed": SEED,
        "num_threads": NUM_THREADS, "verbosity": -1, "feature_pre_filter": False,
        "monotone_constraints": mono,
        "monotone_constraints_method": "basic",
    })

    mst_v = val["mst_total_length"].to_numpy()
    cost_v = val["optimal_cost"].to_numpy()

    def feval(preds, _ds):
        a = to_alpha(preds)
        err = (a * mst_v - cost_v) / cost_v
        return "cost_mape", float(np.mean(np.abs(err)) * 100.0), False

    dtr = lgb.Dataset(train[feats], label=to_z(train["alpha"].to_numpy()))
    dvl = lgb.Dataset(val[feats], label=to_z(val["alpha"].to_numpy()), reference=dtr)
    t0 = time.perf_counter()
    booster = lgb.train(params, dtr, num_boost_round=MAX_BOOST_ROUND,
                        valid_sets=[dvl], valid_names=["val"], feval=feval,
                        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                   lgb.log_evaluation(0)])
    print(f"[fit:{tag}] {len(feats)} feats, {len(train)} train rows, "
          f"best_iter={booster.best_iteration}, trees={booster.num_trees()}, "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    return booster


def stage_train(force: bool = False) -> dict:
    import joblib

    MODEL_DIR.mkdir(exist_ok=True)
    feats31 = frozen_features()
    feats34 = feats31 + NEW3

    corpus = load_corpus(feats31)
    aug = load_augment(feats31)
    leak = assert_no_leakage(corpus, aug)
    print(f"[data] corpus={len(corpus)} augment={len(aug)}  leakage report={leak}")

    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    te = corpus[corpus.split == "test"]
    print(f"[data] train={len(tr)} val={len(va)} test={len(te)}")

    tr_aug = pd.concat([tr, aug[tr.columns.intersection(aug.columns)]],
                       ignore_index=True, sort=False)
    # Augmentation must not have leaked into the early-stopping or test signal.
    assert len(tr_aug) == len(tr) + len(aug)
    assert set(aug.instance_name).isdisjoint(set(va.instance_name))
    assert set(aug.instance_name).isdisjoint(set(te.instance_name))

    arms = {}
    specs = [("R0", tr, feats31), ("A", tr_aug, feats31), ("B", tr_aug, feats34)]
    for tag, train_df, feats in specs:
        p = MODEL_DIR / f"{tag}.joblib"
        if p.exists() and not force:
            arms[tag] = (joblib.load(p), feats)
            print(f"[fit:{tag}] cached, best_iter={arms[tag][0].best_iteration}")
            continue
        b = fit_arm(train_df, va, feats, tag)
        joblib.dump(b, p)
        arms[tag] = (b, feats)

    # ---- R0 must reproduce the frozen artifact --------------------------
    frozen = joblib.load(FROZEN)
    r0 = arms["R0"][0]
    pf = frozen.predict(te[feats31], num_iteration=frozen.best_iteration)
    pr = r0.predict(te[feats31], num_iteration=r0.best_iteration)
    md = float(np.max(np.abs(pf - pr)))
    print(f"[R0] trees {r0.num_trees()} vs frozen {frozen.num_trees()}; "
          f"max |pred - frozen| = {md:.3e}")
    if md > 1e-12:
        raise SystemExit(
            "R0 REPRODUCTION FAILED: this pipeline does not reproduce the frozen "
            f"model (max abs pred diff {md:.3e}). Arm A/B results would be "
            "confounded by a recipe difference, so the run stops here.")
    print("[R0] reproduction verified: pipeline is faithful to the frozen recipe")

    arms["FROZEN"] = (frozen, feats31)
    return {"arms": arms, "corpus": corpus, "aug": aug, "leak": leak,
            "feats31": feats31, "feats34": feats34}


# =========================================================================
# evaluation frames
# =========================================================================
def build_frames() -> dict:
    """Scoring frames, all with the three arm-B columns joined on."""
    C = pd.read_csv(FEAT_CACHE, low_memory=False)
    b2 = pd.read_csv(F_BENCH2D)
    tl = pd.read_csv(F_TSPLIB)
    ag = pd.read_csv(F_AUGMENT)
    cp = pd.read_csv(F_CORPUS)
    deg = pd.read_csv(EXT_OLD, usecols=["instance_name", "degeneracy_pca_effective_rank"])
    corpus_new = cp.merge(deg, on="instance_name", how="left", validate="one_to_one")

    new_by_inst = pd.concat([b2, tl, ag, corpus_new], ignore_index=True)
    new_by_inst = new_by_inst.drop_duplicates("instance_name", keep="first")

    C = C.merge(new_by_inst, left_on="instance", right_on="instance_name",
                how="left")
    gen = pd.read_csv(GEN_MAP, usecols=["instance_name", "generator", "gen_class"])
    C = C.merge(gen.rename(columns={"gen_class": "gen_class_map"}),
                left_on="instance", right_on="instance_name", how="left",
                suffixes=("", "_g"))
    C["generator"] = C["generator"]
    return {"cache": C}


def score(model, feats, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    ok = frame[(frame.status == "ok") & frame[feats].notna().all(axis=1)
               & frame.true_cost.notna()].copy()
    a = to_alpha(model.predict(ok[feats], num_iteration=model.best_iteration))
    a = np.clip(a, *ALPHA_CLIP)
    ok["pred_cost"] = a * ok["mst_total_length"].to_numpy()
    ok["pred_alpha"] = a
    ok["err_pct"] = (ok["pred_cost"] - ok["true_cost"]) / ok["true_cost"] * 100.0
    return ok, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["train", "eval", "all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    st = stage_train(args.force)
    json.dump({"leakage": st["leak"]},
              open(HERE / "support_arms_leakage.json", "w"), indent=2)
    if args.stage == "train":
        return

    from support_arms_eval import run_eval           # noqa: WPS433
    run_eval(st, build_frames())


if __name__ == "__main__":
    main()
