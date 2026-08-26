"""Structural-change study for the GART consistency thesis.

Four structural changes are tested against the shipped LGBM_V3 recipe. The
figure of merit is *dispersion*, not average accuracy: SDPE is the standard
deviation of the signed percent error e_i = 100 (pred - true) / true, and the
discriminating measure is the difficulty-normalised skill SDPE / CV(alpha)
computed per bucket, which must fall monotonically as the problem gives the
estimator more to work with.

    A  monotone_constraints on n_customers and/or dimension
    B  stacking of LGBM_V3 / LGBM_V4 / NN_V3 / Linear_V3, weights fit
       out of evaluation (validation split, plus a K-fold OOF cross-check)
    C  a variance-targeting custom objective, and a two-stage OOF residual
       correction
    D  LightGBM quantile regressions and the empirical coverage of the
       resulting 90 % interval

Every candidate is scored on all four evaluation strata: the ND test split
(16,920, which carries the held-out d = 100 extrapolation bucket), the 2D
diverse-generator benchmark (2,580), TSPLIB95 EUC_2D (78) and the novel-
geometry augmentation corpus (874).

Because alpha = L_TSP / L_MST and the prediction is alpha_hat * L_MST, the
MST length cancels out of the signed percent error exactly:

    e = 100 (alpha_hat L_MST - alpha L_MST) / (alpha L_MST)
      = 100 (alpha_hat - alpha) / alpha

so SDPE in cost space and SDPE in alpha space are the same number. All the
scoring below works in alpha space and is identical to the cost-space number
the paper reports.

CLI
    python structural_study.py fit-a        # monotone constraints
    python structural_study.py fit-b        # stacking
    python structural_study.py fit-c        # SDPE-directed objectives
    python structural_study.py fit-d        # quantile intervals
    python structural_study.py report       # tables from cached predictions
    python structural_study.py all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "paper_tooling")):
    if p not in sys.path:
        sys.path.insert(0, p)

import monotonicity as mono  # noqa: E402
import ood_harness as oh  # noqa: E402

OUT = REPO / "paper_tooling" / "structural_out"
PRED_DIR = OUT / "preds"
MODEL_DIR = OUT / "models"
SEED = 42
THREADS = 6
ALPHA_CLIP = (1.0, 2.0)
MAX_ROUNDS = 3000
EARLY_STOP = 100

#: Strata keys used throughout, in report order.
STRATA = ("ND", "2D", "TSPLIB", "AUG")

#: Map to the ood_harness suite. ND has no harness stratum: it is the held-out
#: test split of the training corpus, scored from the ND benchmark CSV.
HARNESS_KEY = {"2D": "bench2d", "TSPLIB": "tsplib_euc2d", "AUG": "augment", "ND": None}

#: Baselines the dispersion family is run against. Asymptotic_MST is the
#: paper's headline reference; the other two are the incumbent and the known
#: stronger-but-slower sibling.
DISPERSION_BASELINES = ("Asymptotic_MST", "LGBM_V3", "LGBM_V4")

#: Degenerate 2D generator classes where the point model is known to fail.
DEGENERATE_2D = ("line_noise", "grid")

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


# ==========================================================================
# 1. Shipped artefacts and the frozen recipe
# ==========================================================================


def _shipped_v3():
    return joblib.load(REPO / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib")


def _shipped_v4():
    return joblib.load(REPO / "lgbm_model_v4" / "lgbm_alpha_model_v4.joblib")


V3_FEATURES: list[str] = list(_shipped_v3().feature_name_)
V4_FEATURES: list[str] = list(_shipped_v4().feature_name())
HP_V3: dict = json.loads((REPO / "lgbm_model_v3" / "best_params_v3.json").read_text())


def base_params(**over) -> dict:
    """The frozen V3 hyperparameters, capped at ``num_threads`` per fit."""
    p = dict(
        HP_V3,
        objective="regression_l2",
        metric="rmse",
        boosting_type="gbdt",
        seed=SEED,
        num_threads=THREADS,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    p.update(over)
    return p


_HP_V4_RAW: dict = json.loads(
    (REPO / "lgbm_model_v4" / "best_params_v4.json").read_text()
)["hyperparameters"]
#: sklearn-style names in the shipped V4 file -> native LightGBM names.
_V4_RENAME = {"reg_alpha": "lambda_l1", "reg_lambda": "lambda_l2"}
HP_V4: dict = {
    _V4_RENAME.get(k, k): v for k, v in _HP_V4_RAW.items() if k != "early_stopping_rounds"
}
V4_EARLY_STOP: int = int(_HP_V4_RAW["early_stopping_rounds"])


def base_params_v4(**over) -> dict:
    """The shipped V4 hyperparameters, so the OOF V4 arm matches the artefact."""
    p = dict(
        HP_V4,
        objective="regression_l2",
        metric="rmse",
        boosting_type="gbdt",
        seed=SEED,
        num_threads=THREADS,
        verbose=-1,
        deterministic=True,
        force_row_wise=True,
    )
    p.update(over)
    return p


def constraint_vector(features: list[str], on: tuple[str, ...]) -> list[int]:
    """-1 (non-increasing) on the named features, 0 elsewhere."""
    return [-1 if f in on else 0 for f in features]


# ==========================================================================
# 2. Training corpus
# ==========================================================================


_TRAIN_CACHE: pd.DataFrame | None = None


def train_frame() -> pd.DataFrame:
    """``tsp_features_v3.csv`` joined to the v4 greedy feature, plus alpha."""
    global _TRAIN_CACHE
    if _TRAIN_CACHE is not None:
        return _TRAIN_CACHE
    df = pd.read_csv(REPO / "tsp_features_v3.csv")
    df["alpha"] = (df["optimal_cost"] / df["mst_total_length"].replace(0, 1e-9)).clip(*ALPHA_CLIP)
    extra = [c for c in V4_FEATURES if c not in df.columns]
    if extra:
        v4 = pd.read_csv(REPO / "tsp_features_v4.csv", usecols=["instance_name", *extra])
        df = df.merge(v4, on="instance_name", how="left")
    _TRAIN_CACHE = df
    return df


def splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = train_frame()
    return (
        df[df["split"] == "train"].reset_index(drop=True),
        df[df["split"] == "val"].reset_index(drop=True),
        df[df["split"] == "test"].reset_index(drop=True),
    )


# ==========================================================================
# 3. Evaluation corpora
# ==========================================================================


@dataclass
class Corpus:
    key: str
    feats: pd.DataFrame           # instance-indexed, carries every model's features
    mst: pd.Series
    truth: pd.Series              # true tour length
    alpha_true: pd.Series
    n: pd.Series
    d: pd.Series
    group: pd.Series              # generator / family label, for by-class coverage
    axes: dict[str, list[tuple[int, int]]]
    baselines: dict[str, pd.Series] = field(default_factory=dict)  # instance -> pred_cost

    @property
    def index(self) -> pd.Index:
        return self.truth.index

    def cost(self, alpha: np.ndarray) -> pd.Series:
        return pd.Series(np.asarray(alpha, float) * self.mst.to_numpy(), index=self.index)


def _nd_baselines(idx: pd.Index) -> dict[str, pd.Series]:
    nd = pd.read_csv(
        REPO / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv",
        usecols=["model", "instance", "pred_cost", "status"],
        low_memory=False,
    )
    nd = nd[nd["status"] == "ok"]
    wide = nd.pivot_table(index="instance", columns="model", values="pred_cost", aggfunc="first")
    return {str(m): wide[m].reindex(idx) for m in wide.columns}


def _load_nd() -> Corpus:
    g = pd.read_csv(
        REPO / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
    ).set_index("instance")
    idx = pd.Index(g.index, name="instance")
    _, _, te = splits()
    feats = te.set_index("instance_name").reindex(idx)
    feats.index.name = "instance"
    return Corpus(
        key="ND",
        feats=feats,
        mst=g["mst_length"].astype(float),
        truth=g["true_cost"].astype(float),
        alpha_true=(g["true_cost"] / g["mst_length"]).astype(float),
        n=g["n_customers"].astype(float),
        d=g["dimension"].astype(float),
        group=g["distribution"].astype(str),
        axes={"size": mono.ND_SIZE, "dim": mono.ND_DIM},
        baselines=_nd_baselines(idx),
    )


def _load_2d() -> Corpus:
    f = pd.read_csv(REPO / "paper_tooling" / "augmentation_2d_features.csv")
    f = f.merge(
        pd.read_csv(REPO / "paper_tooling" / "v4_study_greedy_2d.csv")[
            ["instance_name", "greedy_nn_over_mst"]
        ],
        on="instance_name",
        how="left",
    ).set_index("instance_name")
    f.index.name = "instance"
    st = oh.load_suite()["bench2d"]
    idx = f.index
    return Corpus(
        key="2D",
        feats=f,
        mst=st.mst.reindex(idx),
        truth=st.truth.reindex(idx),
        alpha_true=(st.truth / st.mst).reindex(idx),
        n=f["n_customers"].astype(float),
        d=f["dimension"].astype(float),
        group=f["generator"].astype(str),
        axes={"size": mono.TWOD_SIZE},
        baselines={k: v.reindex(idx) for k, v in st.baselines.items()},
    )


def _load_tsplib() -> Corpus:
    f = pd.read_csv(REPO / "paper_tooling" / "tsplib_features_v3.csv")
    f = f[f["edge_weight_type"] == "EUC_2D"].merge(
        pd.read_csv(REPO / "paper_tooling" / "v4_study_greedy_tsplib.csv")[
            ["instance_name", "greedy_nn_over_mst"]
        ],
        on="instance_name",
        how="left",
    ).set_index("instance_name")
    f.index.name = "instance"
    st = oh.load_suite()["tsplib_euc2d"]
    idx = pd.Index(sorted(set(f.index) & set(st.truth.index)), name="instance")
    f = f.loc[idx]
    return Corpus(
        key="TSPLIB",
        feats=f,
        mst=st.mst.reindex(idx),
        truth=st.truth.reindex(idx),
        alpha_true=(st.truth / st.mst).reindex(idx),
        n=f["n_customers"].astype(float),
        d=pd.Series(2.0, index=idx),
        group=pd.Series("EUC_2D", index=idx),
        axes={"size": mono.TSPLIB_SIZE},
        baselines={k: v.reindex(idx) for k, v in st.baselines.items()},
    )


def _load_aug() -> Corpus:
    f = pd.read_csv(REPO / "paper_tooling" / "augment_features_v3.csv").merge(
        pd.read_csv(REPO / "paper_tooling" / "augment_greedy_nn.csv"), on="instance_name", how="left"
    ).set_index("instance_name")
    f.index.name = "instance"
    st = oh.load_suite()["augment"]
    idx = pd.Index(sorted(f.index), name="instance")
    f = f.loc[idx]
    base = {k: v.reindex(idx) for k, v in st.baselines.items()}
    # The augmentation stratum ships no LGBM_V4 column; score it here so the
    # stack has the same arms everywhere rather than a hole on one stratum.
    m4 = _shipped_v4()
    base["LGBM_V4"] = pd.Series(
        np.clip(m4.predict(f[V4_FEATURES]), *ALPHA_CLIP) * f["mst_total_length"].to_numpy(),
        index=idx,
    )
    return Corpus(
        key="AUG",
        feats=f,
        mst=f["mst_total_length"].astype(float),
        truth=f["optimal_cost"].astype(float),
        alpha_true=(f["optimal_cost"] / f["mst_total_length"]).astype(float),
        n=f["n_customers"].astype(float),
        d=f["dimension"].astype(float),
        group=f["family"].astype(str),
        axes={"size": mono.AUG_SIZE, "dim": mono.AUG_DIM},
        baselines=base,
    )


_CORPORA: dict[str, Corpus] | None = None


def corpora() -> dict[str, Corpus]:
    global _CORPORA
    if _CORPORA is None:
        _CORPORA = {"ND": _load_nd(), "2D": _load_2d(), "TSPLIB": _load_tsplib(), "AUG": _load_aug()}
    return _CORPORA


# ==========================================================================
# 4. Prediction store
# ==========================================================================


def save_preds(name: str, alphas: dict[str, np.ndarray]) -> None:
    """Persist one candidate's alpha predictions, one CSV per stratum."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for key, a in alphas.items():
        path = PRED_DIR / f"{key}.csv"
        idx = corpora()[key].index
        cur = pd.read_csv(path).set_index("instance") if path.exists() else pd.DataFrame(index=idx)
        cur.index.name = "instance"
        cur = cur.reindex(idx)
        cur[name] = np.asarray(a, float)
        cur.reset_index().to_csv(path, index=False)


def load_preds(key: str) -> pd.DataFrame:
    path = PRED_DIR / f"{key}.csv"
    if not path.exists():
        return pd.DataFrame(index=corpora()[key].index)
    return pd.read_csv(path).set_index("instance").reindex(corpora()[key].index)


def candidate_names() -> list[str]:
    seen: list[str] = []
    for key in STRATA:
        for c in load_preds(key).columns:
            if c not in seen:
                seen.append(c)
    return seen


# ==========================================================================
# 5. Fitting
# ==========================================================================


def fit(
    params: dict,
    features: list[str] = None,
    label: str = "",
    train: pd.DataFrame = None,
    val: pd.DataFrame = None,
    init_score: float | None = None,
    rounds: int = MAX_ROUNDS,
    early_stop: int = EARLY_STOP,
) -> lgb.Booster:
    features = features or V3_FEATURES
    if train is None or val is None:
        tr, va, _ = splits()
        train = tr if train is None else train
        val = va if val is None else val
    t0 = time.time()
    kw = {}
    if init_score is not None:
        kw["init_score"] = np.full(len(train), float(init_score))
    dtr = lgb.Dataset(train[features], label=train["alpha"], **kw)
    vkw = {"init_score": np.full(len(val), float(init_score))} if init_score is not None else {}
    dval = lgb.Dataset(val[features], label=val["alpha"], reference=dtr, **vkw)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stop, verbose=False)],
    )
    if label:
        print(f"  [fit] {label:<28} iter={booster.best_iteration:>5}  {time.time() - t0:5.1f}s")
    return booster


def predict_alpha(
    booster: lgb.Booster,
    features: list[str],
    frame: pd.DataFrame,
    offset: float = 0.0,
    clip: bool = True,
) -> np.ndarray:
    raw = booster.predict(frame[features], num_iteration=booster.best_iteration) + offset
    return np.clip(raw, *ALPHA_CLIP) if clip else raw


def predict_everywhere(
    booster: lgb.Booster, features: list[str] = None, offset: float = 0.0
) -> dict[str, np.ndarray]:
    features = features or V3_FEATURES
    return {k: predict_alpha(booster, features, c.feats, offset) for k, c in corpora().items()}


# ==========================================================================
# 6. Scoring
# ==========================================================================


def spe(alpha_hat: np.ndarray, alpha_true: np.ndarray) -> np.ndarray:
    """Signed percent error. Identical in alpha space and cost space."""
    return 100.0 * (np.asarray(alpha_hat, float) - alpha_true) / alpha_true


def _summarise(e: np.ndarray) -> dict:
    e = e[np.isfinite(e)]
    return {
        "n": int(e.size),
        "sdpe": float(e.std(ddof=1)) if e.size > 1 else np.nan,
        "mape": float(np.abs(e).mean()) if e.size else np.nan,
        "bias": float(e.mean()) if e.size else np.nan,
    }


def stratum_table(names: list[str] = None) -> pd.DataFrame:
    """SDPE / MAPE / bias per candidate per stratum, plus the ND sub-slices."""
    names = names or candidate_names()
    rows = []
    for key in STRATA:
        c = corpora()[key]
        pr = load_preds(key)
        at = c.alpha_true.to_numpy()
        for name in names:
            if name not in pr.columns:
                continue
            e = spe(pr[name].to_numpy(), at)
            rows.append({"model": name, "stratum": key, "slice": "all", **_summarise(e)})
            if key == "ND":
                dd = c.d.to_numpy()
                nn = c.n.to_numpy()
                for lab, m in (
                    ("d<=50 (in-distribution)", dd <= 50),
                    ("d=100 (extrapolation)", dd == 100),
                    ("n in [501,1000]", (nn >= 501) & (nn <= 1000)),
                    ("d<=50 & n in [501,1000]", (dd <= 50) & (nn >= 501) & (nn <= 1000)),
                ):
                    rows.append({"model": name, "stratum": key, "slice": lab, **_summarise(e[m])})
    return pd.DataFrame(rows)


def _mono_stratum(key: str, names: list[str]) -> mono.Stratum:
    """Wrap cached candidate predictions in the monotonicity harness's format."""
    c = corpora()[key]
    meta = pd.DataFrame(
        {
            "instance": c.index,
            "n": c.n.to_numpy(),
            "d": c.d.to_numpy(),
            "alpha": c.alpha_true.to_numpy(),
        }
    )
    pr = load_preds(key)
    frames = []
    for name in names:
        if name not in pr.columns:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "instance": c.index,
                    "pred_cost": pr[name].to_numpy() * c.mst.to_numpy(),
                    "true_cost": c.truth.to_numpy(),
                }
            )
        )
    long = pd.concat(frames, ignore_index=True).merge(meta, on="instance", how="inner")
    long["e"] = 100.0 * (long["pred_cost"] - long["true_cost"]) / long["true_cost"]
    return mono.Stratum(key, long, meta, c.axes)


def bucket_table(names: list[str] = None) -> pd.DataFrame:
    """Per-bucket SDPE and difficulty-normalised skill, via monotonicity.py."""
    names = names or candidate_names()
    strata = [_mono_stratum(k, names) for k in STRATA]
    return mono.build_bucket_table(strata)


def monotonicity_table(names: list[str] = None) -> pd.DataFrame:
    names = names or candidate_names()
    strata = [_mono_stratum(k, names) for k in STRATA]
    bt = mono.build_bucket_table(strata)
    return mono.build_monotonicity(strata, bt)


def skill_across_dimension(names: list[str] = None, bt: pd.DataFrame = None) -> pd.DataFrame:
    """The discriminating measure: SDPE / CV(alpha) per dimension bucket on ND.

    ``build_monotonicity`` deliberately drops the d = 100 extrapolation bucket
    from its trend tests, so the monotone verdict here is reported twice: over
    the in-distribution buckets only, and over the full trajectory including
    the held-out d = 100 bucket.
    """
    names = names or candidate_names()
    bt = bucket_table(names) if bt is None else bt
    sub = bt[(bt.stratum == "ND") & (bt.axis == "dim")].sort_values(["model", "bucket_index"])
    rows = []
    for model, g in sub.groupby("model"):
        sk = g["skill_vs_cv_alpha"].to_numpy(float)
        labs = list(g["bucket"])
        sk_in = sk[~g["extrapolation"].to_numpy()]
        rec = {"model": model}
        rec.update({f"skill_{lab}": v for lab, v in zip(labs, sk)})
        rec.update(
            {
                "mono_in_dist": bool(np.all(np.diff(sk_in) < 0)) if sk_in.size > 1 else False,
                "mono_with_d100": bool(np.all(np.diff(sk) < 0)) if sk.size > 1 else False,
                "n_decreasing_in_dist": int((np.diff(sk_in) < 0).sum()),
                "n_pairs_in_dist": int(sk_in.size - 1),
                "skill_d50": float(sk_in[-1]) if sk_in.size else np.nan,
                "skill_d100": float(sk[-1]) if g["extrapolation"].any() else np.nan,
                "d100_over_d50": float(sk[-1] / sk_in[-1]) if sk_in.size and sk[-1] else np.nan,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def skill_across_size(names: list[str] = None, bt: pd.DataFrame = None) -> pd.DataFrame:
    """Difficulty-normalised skill across the n axis, per stratum."""
    names = names or candidate_names()
    bt = bucket_table(names) if bt is None else bt
    sub = bt[bt.axis == "size"].sort_values(["stratum", "model", "bucket_index"])
    rows = []
    for (stratum, model), g in sub.groupby(["stratum", "model"]):
        sk = g["skill_vs_cv_alpha"].to_numpy(float)
        rec = {"stratum": stratum, "model": model}
        rec.update({f"skill_{lab}": v for lab, v in zip(g["bucket"], sk)})
        rec["monotone"] = bool(np.all(np.diff(sk) < 0)) if sk.size > 1 else False
        rec["n_decreasing"] = int((np.diff(sk) < 0).sum())
        rec["n_pairs"] = int(sk.size - 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def dispersion_family(
    names: list[str] = None, baselines: tuple[str, ...] = DISPERSION_BASELINES
) -> pd.DataFrame:
    """Paired dispersion tests, Holm-adjusted within each candidate's family.

    The family for one candidate is (stratum x baseline); Holm is applied over
    that family so the reported p is the multiplicity-corrected one.
    """
    names = names or candidate_names()
    rows = []
    for key in STRATA:
        c = corpora()[key]
        pr = load_preds(key)
        # The swap permutation materialises an (n_perm x n) float matrix, so it
        # is thinned on the large strata. The Pitman-Morgan test that Holm is
        # applied to is analytic and unaffected.
        n_perm = int(np.clip(2_000_000 // max(len(c.index), 1), 1_000, oh.N_PERMUTATION))
        n_boot = oh.N_BOOTSTRAP if len(c.index) <= 5_000 else 500
        for name in names:
            if name not in pr.columns:
                continue
            cand_cost = c.cost(pr[name].to_numpy())
            for b in baselines:
                if b not in c.baselines:
                    continue
                rec = oh.compare_dispersion(
                    cand_cost, c.baselines[b], c.truth, name, b, metric="signed",
                    n_boot=n_boot, n_perm=n_perm,
                )
                rec["stratum"] = key
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = []
    for name, g in df.groupby("model_a"):
        g = g.copy()
        g["p_holm"] = oh.holm_bonferroni(g["p_pitman_morgan"])
        g["family_size"] = len(g)
        out.append(g)
    keep = [
        "model_a", "stratum", "model_b", "n_pairs", "sdpe_a", "sdpe_b", "sd_ratio",
        "pct_lower", "ratio_ci_low", "ratio_ci_high", "p_pitman_morgan", "p_holm",
        "p_swap_permutation", "mde_sd_ratio", "detectable", "family_size",
    ]
    return pd.concat(out, ignore_index=True)[keep]


def dispersion_vs_candidate(names: list[str], reference: str) -> pd.DataFrame:
    """Paired dispersion of each candidate against another *candidate*.

    Needed for experiment A, whose correct control is the unconstrained refit
    made under identical conditions rather than the shipped artefact.
    """
    rows = []
    for key in STRATA:
        c = corpora()[key]
        pr = load_preds(key)
        if reference not in pr.columns:
            continue
        ref_cost = c.cost(pr[reference].to_numpy())
        n_perm = int(np.clip(2_000_000 // max(len(c.index), 1), 1_000, oh.N_PERMUTATION))
        n_boot = oh.N_BOOTSTRAP if len(c.index) <= 5_000 else 500
        for name in names:
            if name not in pr.columns or name == reference:
                continue
            rec = oh.compare_dispersion(
                c.cost(pr[name].to_numpy()), ref_cost, c.truth, name, reference,
                metric="signed", n_boot=n_boot, n_perm=n_perm,
            )
            rec["stratum"] = key
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = []
    for _name, g in df.groupby("model_a"):
        g = g.copy()
        g["p_holm"] = oh.holm_bonferroni(g["p_pitman_morgan"])
        out.append(g)
    keep = [
        "model_a", "stratum", "model_b", "n_pairs", "sdpe_a", "sdpe_b", "sd_ratio",
        "pct_lower", "ratio_ci_low", "ratio_ci_high", "p_pitman_morgan", "p_holm",
    ]
    return pd.concat(out, ignore_index=True)[keep]


# ==========================================================================
# 7. Experiment A -- monotone constraints
# ==========================================================================


A_SPECS = [
    ("A0_unconstrained", (), None),
    ("A1_n_basic", ("n_customers",), "basic"),
    ("A1_n_intermediate", ("n_customers",), "intermediate"),
    ("A1_n_advanced", ("n_customers",), "advanced"),
    ("A2_d_basic", ("dimension",), "basic"),
    ("A2_d_intermediate", ("dimension",), "intermediate"),
    ("A2_d_advanced", ("dimension",), "advanced"),
    ("A3_nd_basic", ("n_customers", "dimension"), "basic"),
    ("A3_nd_intermediate", ("n_customers", "dimension"), "intermediate"),
    ("A3_nd_advanced", ("n_customers", "dimension"), "advanced"),
]


def experiment_a() -> None:
    print("\n=== A. monotone constraints ===")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tr, va, _ = splits()
    for name, on, method in A_SPECS:
        p = base_params()
        if on:
            p["monotone_constraints"] = constraint_vector(V3_FEATURES, on)
            p["monotone_constraints_method"] = method
        b = fit(p, V3_FEATURES, label=name, train=tr, val=va)
        b.save_model(str(MODEL_DIR / f"{name}.txt"), num_iteration=b.best_iteration)
        save_preds(name, predict_everywhere(b))
    print("  [A] done")


# ==========================================================================
# 8. Experiment B -- stacking with honest weight fitting
# ==========================================================================


BASE_ARMS = ("LGBM_V3", "LGBM_V4", "NN_V3", "Linear_V3")


def _nn_alpha(frame: pd.DataFrame) -> np.ndarray:
    return oh._augment_nn_predictions(frame)


def base_alpha(frame: pd.DataFrame) -> pd.DataFrame:
    """Alpha from each shipped base model on an arbitrary feature frame."""
    m3, m4 = _shipped_v3(), _shipped_v4()
    lin = joblib.load(REPO / "linear_model_v3" / "linear_alpha_model_v3.joblib")
    return pd.DataFrame(
        {
            "LGBM_V3": np.clip(m3.predict(frame[V3_FEATURES]), *ALPHA_CLIP),
            "LGBM_V4": np.clip(m4.predict(frame[V4_FEATURES]), *ALPHA_CLIP),
            "NN_V3": np.clip(_nn_alpha(frame), *ALPHA_CLIP),
            "Linear_V3": np.clip(lin.predict(frame[list(lin.feature_names_in_)]), *ALPHA_CLIP),
        },
        index=frame.index,
    )


def _nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares weights on the *relative* error.

    Minimising sum_i ((sum_k w_k p_ik) - y_i)^2 / y_i^2 is the right target
    here: it is the squared relative error whose spread is SDPE. Dividing both
    the design and the response by y turns that into an ordinary NNLS problem
    with design ``P / y`` and response ``1``.
    """
    from scipy import optimize

    w, _ = optimize.nnls(P / y[:, None], np.ones_like(y))
    return w


def _ridge_weights(P: np.ndarray, y: np.ndarray, lam: float = 1.0):
    """Ridge meta-learner with an intercept, weighted to the relative error.

    Weighting each row by 1 / y^2 makes the fitted criterion the squared
    relative error, matching SDPE, while keeping the learned map a plain
    affine function of the base alphas so it is deployable without knowing y.
    """
    from sklearn.linear_model import Ridge

    r = Ridge(alpha=lam, fit_intercept=True)
    r.fit(P, y, sample_weight=1.0 / y**2)
    return r.coef_.copy(), float(r.intercept_)


def _oof_lgbm_alpha(df: pd.DataFrame, features: list[str], params: dict, folds: int = 5,
                    tag: str = "", rounds: int = MAX_ROUNDS,
                    early_stop: int = EARLY_STOP) -> np.ndarray:
    """K-fold out-of-fold alpha on ``df``, stratified by dimension.

    The held-out fold doubles as the early-stopping set for that fold, which
    is the same protocol the shipped models use (train fitted, val watched);
    the out-of-fold prediction is still never used to fit that fold's trees.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.full(len(df), np.nan)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    strat = df["dimension"].astype(int).to_numpy()
    for i, (itr, iva) in enumerate(skf.split(df, strat)):
        sub_tr, sub_va = df.iloc[itr], df.iloc[iva]
        b = fit(params, features, label=f"{tag}fold{i}", train=sub_tr, val=sub_va,
                rounds=rounds, early_stop=early_stop)
        y[iva] = predict_alpha(b, features, sub_va)
    return y


def experiment_b() -> None:
    print("\n=== B. stacking ===")
    tr, va, _ = splits()

    # --- weights fit on the validation split (never an evaluation set) ------
    Pv = base_alpha(va)
    yv = va["alpha"].to_numpy()
    w_avg = np.full(len(BASE_ARMS), 1.0 / len(BASE_ARMS))
    w_nnls = _nnls_weights(Pv[list(BASE_ARMS)].to_numpy(), yv)
    coef, icpt = _ridge_weights(Pv[list(BASE_ARMS)].to_numpy(), yv)

    # --- weights fit on out-of-fold training predictions (cross-check) ------
    # Only the two LightGBM arms can be honestly refit out of fold here; the
    # NN and the linear model are shipped artefacts trained on this same split,
    # so their in-fold predictions would be optimistic. The OOF stack is
    # therefore reported as a two-arm tree stack, which is also the stack that
    # does not inherit a d = 100 collapse.
    oof3 = _oof_lgbm_alpha(tr, V3_FEATURES, base_params(), tag="B-oof-v3-")
    oof4 = _oof_lgbm_alpha(tr, V4_FEATURES, base_params_v4(), tag="B-oof-v4-",
                           rounds=6000, early_stop=V4_EARLY_STOP)
    Po = np.column_stack([oof3, oof4])
    w_oof = _nnls_weights(Po, tr["alpha"].to_numpy())

    weights = {
        "B1_avg4": ("val", list(BASE_ARMS), w_avg, 0.0),
        "B2_nnls4": ("val", list(BASE_ARMS), w_nnls, 0.0),
        "B3_ridge4": ("val", list(BASE_ARMS), coef, icpt),
        "B4_nnls_trees_val": ("val", ["LGBM_V3", "LGBM_V4"],
                              _nnls_weights(Pv[["LGBM_V3", "LGBM_V4"]].to_numpy(), yv), 0.0),
        "B5_nnls_trees_oof": ("oof-train", ["LGBM_V3", "LGBM_V4"], w_oof, 0.0),
    }
    (OUT).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"stack": k, "fit_on": src, **dict(zip(arms, np.round(w, 6))), "intercept": ic}
            for k, (src, arms, w, ic) in weights.items()
        ]
    ).to_csv(OUT / "stack_weights.csv", index=False)
    for k, (src, arms, w, ic) in weights.items():
        print(f"  [B] {k:<20} fit_on={src:<10} " + "  ".join(f"{a}={v:+.4f}" for a, v in zip(arms, w))
              + (f"  intercept={ic:+.5f}" if ic else ""))

    # --- score the base arms and the stacks on every stratum ---------------
    for key, c in corpora().items():
        P = base_alpha(c.feats)
        for arm in BASE_ARMS:
            save_preds(f"BASE_{arm}", {key: P[arm].to_numpy()})
        for k, (_src, arms, w, ic) in weights.items():
            a = P[arms].to_numpy() @ w + ic
            save_preds(k, {key: np.clip(a, *ALPHA_CLIP)})
    print("  [B] done")


# ==========================================================================
# 9. Experiment C -- SDPE-directed objectives
# ==========================================================================


def variance_objective(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Custom LightGBM objective penalising deviation from the *running mean*
    relative error -- i.e. it targets variance, not magnitude.

    With r_i = (p_i - y_i) / y_i and rbar the current batch mean of r,

        L          = 0.5 * sum_i (r_i - rbar)^2
        dL/dp_i    = (r_i - rbar) / y_i     (the rbar term cancels exactly,
                                             because sum_i (r_i - rbar) = 0)
        d2L/dp_i^2 = (1 - 1/N) / y_i^2

    This is deliberately *not* the 1/alpha^2-weighted L2 -- squared relative
    error -- that the objective-study team is running: the mean is subtracted,
    so only the spread is penalised. The price is that the loss is almost
    invariant to a level shift, which leaves the fitted model's location
    arbitrary; it is fixed afterwards by one scalar calibration constant fit
    on the training split.
    """
    y = dataset.get_label()
    iv = 1.0 / y
    r = (preds - y) * iv
    grad = (r - r.mean()) * iv
    hess = (1.0 - 1.0 / max(y.size, 2)) * iv * iv
    return grad, hess


def _variance_eval(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
    y = dataset.get_label()
    r = 100.0 * (preds - y) / y
    return "sdpe", float(r.std(ddof=1)), False


def experiment_c() -> None:
    print("\n=== C. SDPE-directed objectives ===")
    tr, va, _ = splits()
    ytr = tr["alpha"].to_numpy()

    # --- C1: variance-targeting custom objective ---------------------------
    p = base_params(objective=variance_objective, metric="None")
    dtr = lgb.Dataset(tr[V3_FEATURES], label=ytr, init_score=np.full(len(tr), float(ytr.mean())))
    dva = lgb.Dataset(
        va[V3_FEATURES],
        label=va["alpha"],
        reference=dtr,
        init_score=np.full(len(va), float(ytr.mean())),
    )
    t0 = time.time()
    bc = lgb.train(
        p,
        dtr,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dva],
        valid_names=["val"],
        feval=_variance_eval,
        callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True, verbose=False)],
    )
    off = float(ytr.mean())
    print(f"  [fit] {'C1_variance_obj':<28} iter={bc.best_iteration:>5}  {time.time() - t0:5.1f}s")

    # scalar multiplicative calibration, fit on the TRAINING split only
    raw_tr = bc.predict(tr[V3_FEATURES], num_iteration=bc.best_iteration) + off
    scale = float(np.sum(raw_tr / ytr) / np.sum((raw_tr / ytr) ** 2))
    print(f"        train-fit calibration scale = {scale:.6f}")

    def c1_alpha(frame):
        raw = bc.predict(frame[V3_FEATURES], num_iteration=bc.best_iteration) + off
        return np.clip(raw * scale, *ALPHA_CLIP)

    save_preds("C1_variance_obj", {k: c1_alpha(c.feats) for k, c in corpora().items()})
    save_preds("C1_variance_obj_uncal",
               {k: np.clip(bc.predict(c.feats[V3_FEATURES], num_iteration=bc.best_iteration) + off,
                           *ALPHA_CLIP) for k, c in corpora().items()})

    # --- C2: two-stage OOF residual correction -----------------------------
    oof = _oof_lgbm_alpha(tr, V3_FEATURES, base_params(), tag="C-oof-")
    stage1 = fit(base_params(), V3_FEATURES, label="C2_stage1", train=tr, val=va)

    tr2 = tr.copy()
    tr2["alpha"] = (ytr - oof) / oof            # signed relative residual
    va2 = va.copy()
    a1_va = predict_alpha(stage1, V3_FEATURES, va)
    va2["alpha"] = (va["alpha"].to_numpy() - a1_va) / a1_va
    resid = fit(base_params(), V3_FEATURES, label="C2_residual", train=tr2, val=va2)

    def c2_alpha(frame):
        a1 = predict_alpha(stage1, V3_FEATURES, frame)
        s = resid.predict(frame[V3_FEATURES], num_iteration=resid.best_iteration)
        return np.clip(a1 * (1.0 + s), *ALPHA_CLIP)

    save_preds("C2_two_stage", {k: c2_alpha(c.feats) for k, c in corpora().items()})
    save_preds("C2_stage1_only",
               {k: predict_alpha(stage1, V3_FEATURES, c.feats) for k, c in corpora().items()})
    print("  [C] done")


# ==========================================================================
# 10. Experiment D -- prediction intervals
# ==========================================================================


def experiment_d() -> None:
    print("\n=== D. quantile prediction intervals ===")
    tr, va, _ = splits()
    val_q: dict[str, np.ndarray] = {}
    for q in QUANTILES:
        tag = f"D_q{int(q * 100):02d}"
        p = base_params(objective="quantile", alpha=q, metric="quantile")
        b = fit(p, V3_FEATURES, label=tag, train=tr, val=va)
        save_preds(tag, {k: predict_alpha(b, V3_FEATURES, c.feats) for k, c in corpora().items()})
        val_q[tag] = predict_alpha(b, V3_FEATURES, va)

    # --- split-conformal widening (CQR), calibrated on the validation split --
    # score_i = max(lo_i - y_i, y_i - hi_i); the (1-a)(n+1)/n empirical
    # quantile of the score is added to both ends. Calibrated on val only, so
    # no evaluation stratum touches the calibration.
    yv = va["alpha"].to_numpy()
    lo, hi = val_q["D_q05"], val_q["D_q95"]
    score = np.maximum(lo - yv, yv - hi)
    n = score.size
    delta = float(np.quantile(score, min(1.0, np.ceil((n + 1) * 0.90) / n), method="higher"))
    lo25, hi75 = val_q["D_q25"], val_q["D_q75"]
    s50 = np.maximum(lo25 - yv, yv - hi75)
    delta50 = float(np.quantile(s50, min(1.0, np.ceil((n + 1) * 0.50) / n), method="higher"))
    print(f"  [D] conformal widening (val, n={n}): 90% delta={delta:.6f} alpha units, "
          f"50% delta={delta50:.6f}")
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "conformal_delta.json").write_text(
        json.dumps({"n_calibration": n, "delta90": delta, "delta50": delta50}, indent=2)
    )
    for key in STRATA:
        pr = load_preds(key)
        save_preds("D_cq05", {key: np.clip(pr["D_q05"].to_numpy() - delta, *ALPHA_CLIP)})
        save_preds("D_cq95", {key: np.clip(pr["D_q95"].to_numpy() + delta, *ALPHA_CLIP)})
        save_preds("D_cq25", {key: np.clip(pr["D_q25"].to_numpy() - delta50, *ALPHA_CLIP)})
        save_preds("D_cq75", {key: np.clip(pr["D_q75"].to_numpy() + delta50, *ALPHA_CLIP)})
    print("  [D] done")


def coverage_table(conformal: bool = False) -> pd.DataFrame:
    """Empirical coverage and mean width of the 90 % interval, by stratum.

    ``conformal=True`` scores the split-conformal (CQR) widened interval whose
    single additive constant was calibrated on the validation split.
    """
    pre = "D_cq" if conformal else "D_q"
    need = [f"{pre}05", f"{pre}95", f"{pre}25", f"{pre}75"]
    rows = []
    for key in STRATA:
        c = corpora()[key]
        pr = load_preds(key)
        if any(x not in pr.columns for x in need):
            continue
        lo, hi = pr[f"{pre}05"].to_numpy(), pr[f"{pre}95"].to_numpy()
        at = c.alpha_true.to_numpy()
        inside = (at >= lo) & (at <= hi)
        width = 100.0 * (hi - lo) / at          # interval width as % of truth
        lo25, hi75 = pr[f"{pre}25"].to_numpy(), pr[f"{pre}75"].to_numpy()
        in50 = (at >= lo25) & (at <= hi75)
        # Is the width at least a useful difficulty signal where it fails to
        # cover? Rank-correlate it against the median arm's absolute error.
        abs_err = np.abs(spe(pr["D_q50"].to_numpy(), at)) if "D_q50" in pr.columns else None

        def rec(lab, m):
            from scipy import stats as _st

            rho = np.nan
            if abs_err is not None and m.sum() > 10:
                rho = float(_st.spearmanr(width[m], abs_err[m]).statistic)
            return {
                "width_vs_abserr_rho": rho,
                "interval": "conformal" if conformal else "raw",
                "stratum": key,
                "slice": lab,
                "n": int(m.sum()),
                "cov90_pct": 100.0 * float(inside[m].mean()) if m.sum() else np.nan,
                "cov50_pct": 100.0 * float(in50[m].mean()) if m.sum() else np.nan,
                "mean_width_pct": float(width[m].mean()) if m.sum() else np.nan,
                "median_width_pct": float(np.median(width[m])) if m.sum() else np.nan,
                "below_lo_pct": 100.0 * float((at[m] < lo[m]).mean()) if m.sum() else np.nan,
                "above_hi_pct": 100.0 * float((at[m] > hi[m]).mean()) if m.sum() else np.nan,
            }

        all_m = np.ones(len(c.index), bool)
        rows.append(rec("all", all_m))
        if key == "ND":
            dd, nn = c.d.to_numpy(), c.n.to_numpy()
            rows.append(rec("d<=50 (in-distribution)", dd <= 50))
            rows.append(rec("d=100 (extrapolation)", dd == 100))
            rows.append(rec("n in [501,1000]", (nn >= 501) & (nn <= 1000)))
        if key == "2D":
            g = c.group.to_numpy()
            for gen in DEGENERATE_2D:
                rows.append(rec(f"generator={gen}", g == gen))
            rows.append(rec("non-degenerate generators", ~np.isin(g, DEGENERATE_2D)))
        if key == "AUG":
            g = c.group.to_numpy()
            for fam in sorted(set(g)):
                rows.append(rec(f"family={fam}", g == fam))
    return pd.DataFrame(rows)


def sliced_dispersion(
    names: list[str], reference: str, slices: dict[str, np.ndarray] = None
) -> pd.DataFrame:
    """Paired dispersion on ND sub-slices, notably the d = 100 held-out bucket.

    The whole-stratum test averages the extrapolation bucket away; the paper's
    claim is specifically about behaviour there, so it gets its own test.
    """
    c = corpora()["ND"]
    pr = load_preds("ND")
    dd, nn = c.d.to_numpy(), c.n.to_numpy()
    slices = slices or {
        "d<=50 (in-distribution)": dd <= 50,
        "d=100 (extrapolation)": dd == 100,
        "n in [501,1000]": (nn >= 501) & (nn <= 1000),
    }
    rows = []
    for lab, m in slices.items():
        idx = c.index[m]
        truth = c.truth.loc[idx]
        ref = c.cost(pr[reference].to_numpy()).loc[idx]
        for name in names:
            if name not in pr.columns or name == reference:
                continue
            rec = oh.compare_dispersion(
                c.cost(pr[name].to_numpy()).loc[idx], ref, truth, name, reference,
                metric="signed", n_boot=500, n_perm=1000,
            )
            rec["slice"] = lab
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = []
    for _n, g in df.groupby("model_a"):
        g = g.copy()
        g["p_holm"] = oh.holm_bonferroni(g["p_pitman_morgan"])
        out.append(g)
    keep = ["model_a", "slice", "model_b", "n_pairs", "sdpe_a", "sdpe_b", "sd_ratio",
            "ratio_ci_low", "ratio_ci_high", "p_pitman_morgan", "p_holm"]
    return pd.concat(out, ignore_index=True)[keep]


def constraint_binding(n_probe: int = 400, seed: int = SEED) -> pd.DataFrame:
    """Does a monotone constraint actually make the prediction monotone?

    Each probe holds one real test instance's features fixed and sweeps the
    named feature across a grid, including values beyond the training range
    (d up to 200, n up to 4000). A constrained model must return a
    non-increasing curve on every probe *by construction*; an unconstrained
    one only does so where the data happened to teach it. This separates a
    structural guarantee from an empirical accident.
    """
    rng = np.random.default_rng(seed)
    c = corpora()["ND"]
    probe = c.feats.iloc[rng.choice(len(c.feats), n_probe, replace=False)].copy()
    grids = {
        "dimension": np.array([2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200], float),
        "n_customers": np.array([5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000], float),
    }
    boosters = {name: lgb.Booster(model_file=str(MODEL_DIR / f"{name}.txt"))
                for name in [s[0] for s in A_SPECS]
                if (MODEL_DIR / f"{name}.txt").exists()}
    rows = []
    for name, b in boosters.items():
        for feat, grid in grids.items():
            curves = np.empty((len(probe), grid.size))
            for j, v in enumerate(grid):
                q = probe.copy()
                q[feat] = v
                curves[:, j] = np.clip(b.predict(q[V3_FEATURES]), *ALPHA_CLIP)
            diffs = np.diff(curves, axis=1)
            rows.append(
                {
                    "model": name,
                    "swept_feature": feat,
                    "n_probes": len(probe),
                    "pct_non_increasing": 100.0 * float((diffs <= 1e-12).all(axis=1).mean()),
                    "pct_pairs_non_increasing": 100.0 * float((diffs <= 1e-12).mean()),
                    "mean_total_drop": float((curves[:, 0] - curves[:, -1]).mean()),
                    "max_violation": float(np.maximum(diffs, 0).max()),
                }
            )
    return pd.DataFrame(rows)


# ==========================================================================
# 11. Reporting
# ==========================================================================


def _fmt(df: pd.DataFrame, floats: int = 4) -> str:
    with pd.option_context("display.width", 220, "display.max_columns", 60,
                           "display.max_rows", 400,
                           "display.float_format", lambda v: f"{v:.{floats}f}"):
        return df.to_string(index=False)


def report(names: list[str] = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = names or candidate_names()
    reference = [n for n in ("BASE_LGBM_V3", "BASE_LGBM_V4", "BASE_NN_V3", "BASE_Linear_V3")
                 if n in names]
    order = reference + [n for n in names if n not in reference]

    st = stratum_table(order)
    st.to_csv(OUT / "stratum_scores.csv", index=False)
    wide = st[st["slice"] == "all"].pivot(index="model", columns="stratum", values="sdpe")
    wide = wide.reindex(order)[list(STRATA)].add_prefix("SDPE_")
    nd = st[st.stratum == "ND"].pivot(index="model", columns="slice", values="sdpe")
    nd = nd.reindex(order)[["d<=50 (in-distribution)", "d=100 (extrapolation)",
                            "n in [501,1000]"]]
    nd.columns = ["SDPE_ND_d<=50", "SDPE_ND_d=100", "SDPE_ND_n501-1000"]
    print("\n########## SDPE by stratum and ND slice ##########")
    print(_fmt(wide.join(nd).reset_index()))

    bt = bucket_table(order)
    bt.to_csv(OUT / "bucket_table.csv", index=False)

    sk = skill_across_dimension(order, bt)
    sk.to_csv(OUT / "skill_dimension.csv", index=False)
    print("\n########## difficulty-normalised skill across dimension (ND) ##########")
    print(_fmt(sk))

    sz = skill_across_size(order, bt)
    sz.to_csv(OUT / "skill_size.csv", index=False)

    mt = monotonicity_table(order)
    mt.to_csv(OUT / "monotonicity.csv", index=False)

    dsp = dispersion_family(order)
    dsp.to_csv(OUT / "dispersion.csv", index=False)
    print("\n########## dispersion vs Asymptotic_MST (Holm within candidate) ##########")
    print(_fmt(dsp[dsp.model_b == "Asymptotic_MST"]))

    if "A0_unconstrained" in order:
        ctl = dispersion_vs_candidate(order, "A0_unconstrained")
        ctl.to_csv(OUT / "dispersion_vs_A0.csv", index=False)
        print("\n########## dispersion vs the unconstrained refit control ##########")
        print(_fmt(ctl))

    sl = sliced_dispersion(order, "A0_unconstrained") if "A0_unconstrained" in order else None
    if sl is not None and not sl.empty:
        sl.to_csv(OUT / "dispersion_nd_slices.csv", index=False)
        print("\n########## ND slice dispersion vs the unconstrained control ##########")
        print(_fmt(sl[sl["slice"] == "d=100 (extrapolation)"]))

    if "BASE_LGBM_V4" in order:
        s4 = sliced_dispersion([n for n in order if n.startswith("B")], "BASE_LGBM_V4")
        if not s4.empty:
            s4.to_csv(OUT / "dispersion_nd_slices_vs_v4.csv", index=False)
            print("\n########## stacks vs LGBM_V4 on the ND slices ##########")
            print(_fmt(s4))

    cb = constraint_binding()
    if not cb.empty:
        cb.to_csv(OUT / "constraint_binding.csv", index=False)
        print("\n########## does the constraint bind? (swept-feature probes) ##########")
        print(_fmt(cb, 3))

    cov = pd.concat([coverage_table(False), coverage_table(True)], ignore_index=True)
    if not cov.empty:
        cov.to_csv(OUT / "coverage.csv", index=False)
        print("\n########## interval coverage: raw quantile vs split-conformal ##########")
        print(_fmt(cov, 2))

    print(f"\n[report] tables written to {OUT}")


# ==========================================================================
# 12. CLI
# ==========================================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "command",
        choices=["fit-a", "fit-b", "fit-c", "fit-d", "report", "all", "check"],
    )
    ap.add_argument("--models", default=None, help="comma-separated subset for report")
    a = ap.parse_args(argv)

    if a.command == "check":
        m3 = _shipped_v3()
        for k, c in corpora().items():
            ref = np.clip(m3.predict(c.feats[V3_FEATURES]), *ALPHA_CLIP)
            e = spe(ref, c.alpha_true.to_numpy())
            base = c.baselines.get("LGBM_V3")
            eb = spe(base.to_numpy() / c.mst.to_numpy(), c.alpha_true.to_numpy())
            print(f"{k:<7} n={len(c.index):>6}  baselines={len(c.baselines):>2}  "
                  f"d={sorted(set(c.d.astype(int)))[:4]}..  "
                  f"V3 SDPE refit={e.std(ddof=1):.4f} shipped={np.nanstd(eb, ddof=1):.4f}")
        return 0

    if a.command in ("fit-a", "all"):
        experiment_a()
    if a.command in ("fit-b", "all"):
        experiment_b()
    if a.command in ("fit-c", "all"):
        experiment_c()
    if a.command in ("fit-d", "all"):
        experiment_d()
    if a.command in ("report", "all"):
        report(a.models.split(",") if a.models else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
