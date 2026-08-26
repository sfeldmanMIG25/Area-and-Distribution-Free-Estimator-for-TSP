"""Price the monotone constraint on the rival: what does ADDING it cost LGBM_V4?

The manuscript argues that aggregate error does not select which estimator to
ship, consistency does.  GART 2.0 scores 100% on both axes of the swept-feature
monotonicity probe; ``LGBM_V4`` scores 9.1% on ``dimension`` and 78.6% on
``n_customers`` while beating GART 2.0 on three of four strata.  The paper
measures what *removing* the constraint buys GART 2.0 (its unconstrained twin
scores 8.4% / 19.0%) and never measures what *adding* it costs the rival.  This
module measures that.

The decision rule is pre-registered in ``constraint_transfer_protocol.md``,
written before the first fit, and is not restated here so the two cannot drift.
Everything below is the measurement.

ONE VARIABLE
------------
``LGBM_V4`` is refit with ``monotone_constraints`` of -1 on ``n_customers`` and
``dimension`` and ``monotone_constraints_method = "basic"`` -- the identical
specification carried by the shipped GART 2.0 artifact, taken from
``train_gart2_monotone.constraint_vector`` rather than rewritten.  Feature
block, target, deployment map, hyperparameters, boosting rounds and splits are
``lgbm_model_v4/train.py``'s final-fit values, unchanged.

REPRODUCTION GATE
-----------------
Arm ``V4_unc_32f`` at seed 42 must reproduce ``lgbm_alpha_model_v4.joblib``
bit-identically (``Booster.model_to_string()``).  If it does not, this harness
is not refitting ``LGBM_V4`` and nothing here is reportable.  The same check is
run for ``GART2_refit`` against ``gart2_final.joblib`` and reported as a
diagnostic -- the protocol did not pre-register it as a gate, so it is not
treated as one.

FALSIFICATION GATE
------------------
The probe is ``v4_study._sweep_monotonicity`` with ``v4_study``'s own grids,
tolerance and 1000-instance base, imported and never reimplemented, so every
number is directly comparable with ``v4_study_gart2_probe.csv`` and
``consistency_31f_probe.csv``.  ``consistency_31f``'s three synthetic controls
run first and must discriminate before any model's pass rate is reported.

NUISANCE CONTROL
----------------
Every arm is fit at seven seeds.  The headline statistic is the median; the full
band and every per-seed value are written out.  Two candidates in this project
have already been rejected for a headline that turned out to be the favourable
extreme of exactly this distribution.

Usage:
    python paper_tooling/constraint_transfer.py fit
    python paper_tooling/constraint_transfer.py eval
    python paper_tooling/constraint_transfer.py probe
    python paper_tooling/constraint_transfer.py verdict
    python paper_tooling/constraint_transfer.py all
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE), str(REPO / "lgbm_model_v3"),
           str(REPO / "lgbm_model_v4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import v4_study as V4  # noqa: E402  probe grids, tolerance, sweep, deploy map

MODELS_DIR = HERE / "constraint_transfer_models"
P_PERSEED = HERE / "constraint_transfer_perseed.csv"
P_TSPLIB_ROWS = HERE / "constraint_transfer_tsplib_rows.csv"
P_PROBE = HERE / "constraint_transfer_probe.csv"
P_SUMMARY = HERE / "constraint_transfer_summary.csv"
P_VERDICT = HERE / "constraint_transfer_verdict.json"
P_REPRO = HERE / "constraint_transfer_repro.json"

FEAT_CACHE = HERE / "v4_study_feature_cache.csv"
V4_DIR = REPO / "lgbm_model_v4"
G_DIR = REPO / "lgbm_model_v3"

#: Pre-registered.  Seed 42 is the shipped configuration.
SEEDS = (42, 43, 44, 45, 46, 47, 48)

#: Pre-registered.  ``augment`` is measured but is not one of the four.
STRATA = ("nd_test", "bench2d", "tsplib_euc2d", "tsplib_noneuc")
STRATA_ALL = STRATA + ("augment",)

MONO_COLS = ("n_customers", "dimension")
MATERIALITY_PP = 0.05
DECISIVE = ("bench2d", "tsplib_noneuc")

ARMS = {
    # label            block      mono   recipe
    "V4_unc_32f":     ("v4_32",   False, "v4"),
    "V4_mono_32f":    ("v4_32",   True,  "v4"),
    "V4_unc_31f":     ("v4_31",   False, "v4"),
    "V4_mono_31f":    ("v4_31",   True,  "v4"),
    "GART2_refit":    ("gart_31", True,  "gart"),
}

ARM_ROLES = {
    "V4_unc_32f": "reproduces shipped LGBM_V4; baseline the cost is measured against",
    "V4_mono_32f": "HEADLINE: LGBM_V4 + GART 2.0's monotone constraint, one variable changed",
    "V4_unc_31f": "symmetric arm: V4 recipe without mst_total_length, unconstrained",
    "V4_mono_31f": "symmetric arm: V4 recipe without mst_total_length, constrained",
    "GART2_refit": "GART 2.0's own recipe refit at k seeds, so production carries a band too",
    "LGBM_V4[shipped]": "published artifact, re-evaluated in this session",
    "GART_2.0[shipped]": "published artifact, re-evaluated in this session",
}

TARGET = {"v4": "alpha", "gart": "logit"}


# =============================================================================
# Feature blocks
# =============================================================================
def _blocks() -> dict[str, list[str]]:
    """The three feature blocks, each read off an existing artifact.

    ``v4_32`` is the shipped V4 booster's own order; ``gart_31`` is the shipped
    GART 2.0 booster's own order.  ``v4_31`` is ``v4_32`` minus
    ``mst_total_length`` -- V4's 32 columns are exactly GART 2.0's 31 plus that
    one, which is what makes the symmetric arm clean and why no third feature
    set is invented.
    """
    sel = json.loads((V4_DIR / "selected_features.json").read_text(encoding="utf-8"))["selected"]
    v4b = list(joblib.load(V4_DIR / "lgbm_alpha_model_v4.joblib").feature_name())
    if v4b != sel:
        raise SystemExit("selected_features.json disagrees with the shipped V4 booster order")

    from train_gart2 import build_feature_list
    g = build_feature_list()
    side = json.loads((G_DIR / "gart2_final.json").read_text(encoding="utf-8"))
    if g != side["features_in_booster_order"]:
        raise SystemExit("build_feature_list disagrees with the frozen GART 2.0 sidecar")

    extra = [f for f in v4b if f not in g]
    if extra != ["mst_total_length"]:
        raise SystemExit(f"V4 minus GART 2.0 is {extra}, expected ['mst_total_length']")

    return {"v4_32": v4b, "v4_31": [f for f in v4b if f != "mst_total_length"],
            "gart_31": g}


def _mono_vector(feats: list[str]) -> list[int]:
    """GART 2.0's own constraint vector builder, not a second copy of the rule."""
    from train_gart2_monotone import constraint_vector
    v = constraint_vector(feats, "both")
    if [f for f, m in zip(feats, v) if m] != [c for c in MONO_COLS if c in feats]:
        raise SystemExit(f"constraint vector does not pin {MONO_COLS}")
    return v


# =============================================================================
# Data
# =============================================================================
def _load_table() -> pd.DataFrame:
    """``tsp_features_v4.csv`` with the alpha label, identically to both recipes.

    ``lgbm_model_v4/train.py::_load`` and ``lgbm_model_v3/train_gart2.py::
    load_data`` compute the label with the same three lines; doing it once here
    keeps a single 85 MB read and cannot diverge between the arms.
    """
    df = pd.read_csv(REPO / "tsp_features_v4.csv")
    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(1.0, 2.0)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)
    counts = df["split"].value_counts().to_dict()
    if counts != {"train": 69768, "val": 19584, "test": 16920}:
        raise SystemExit(f"unexpected split sizes {counts}")
    return df


# =============================================================================
# Fitting
# =============================================================================
def _fit_v4(df, feats, mono, seed) -> lgb.Booster:
    bp = json.loads((V4_DIR / "best_params_v4.json").read_text(encoding="utf-8"))
    p = dict(bp["hyperparameters"])
    p.pop("early_stopping_rounds", None)
    p.update({"objective": "regression", "metric": "rmse", "verbosity": -1,
              "seed": seed})
    if mono:
        p["monotone_constraints"] = _mono_vector(feats)
        p["monotone_constraints_method"] = "basic"
    # Row ORDER matters: LightGBM's bagging draws by position, so the
    # train-then-val concatenation of ``train.py::main`` is part of the recipe.
    # Masking the table in place instead interleaves the two splits and changes
    # every bagging subset -- the reproduction gate catches exactly that.
    mtr, mvl = df["split"] == "train", df["split"] == "val"
    X = pd.concat([df.loc[mtr, feats], df.loc[mvl, feats]])
    y = pd.concat([df.loc[mtr, "alpha"], df.loc[mvl, "alpha"]])
    return lgb.train(p, lgb.Dataset(X, label=y),
                     num_boost_round=int(bp["num_boost_round"]))


def _fit_gart(df, feats, mono, seed) -> lgb.Booster:
    from train_gart2 import MAX_BOOST_ROUND, NUM_THREADS
    from train_gart2_logit import V3_EARLY_STOP, V3_FROZEN, make_mape_feval, to_z

    p = dict(V3_FROZEN)
    p.update({"objective": "regression_l2", "metric": "None",
              "boosting_type": "gbdt", "seed": seed, "num_threads": NUM_THREADS,
              "verbosity": -1, "feature_pre_filter": False})
    if mono:
        p["monotone_constraints"] = _mono_vector(feats)
        p["monotone_constraints_method"] = "basic"
    tr, vl = df[df.split == "train"], df[df.split == "val"]
    dtr = lgb.Dataset(tr[feats], label=to_z(tr["alpha"].to_numpy()))
    dvl = lgb.Dataset(vl[feats], label=to_z(vl["alpha"].to_numpy()), reference=dtr)
    return lgb.train(
        p, dtr, num_boost_round=MAX_BOOST_ROUND, valid_sets=[dvl],
        valid_names=["val"],
        feval=make_mape_feval(vl["mst_total_length"].to_numpy(),
                              vl["optimal_cost"].to_numpy()),
        callbacks=[lgb.early_stopping(V3_EARLY_STOP, verbose=False),
                   lgb.log_evaluation(0)])


def _path(arm: str, seed: int) -> Path:
    return MODELS_DIR / f"{arm}_s{seed}.joblib"


def cmd_fit(args) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    blocks = _blocks()
    df = _load_table()
    print(f"[fit] {len(df)} labelled rows | blocks "
          f"{ {k: len(v) for k, v in blocks.items()} }")

    repro: dict = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "lightgbm": lgb.__version__, "numpy": np.__version__,
                   "pandas": pd.__version__, "python": platform.python_version(),
                   "seeds": list(SEEDS), "fits": {}}

    for arm, (block, mono, recipe) in ARMS.items():
        feats = blocks[block]
        for seed in SEEDS:
            p = _path(arm, seed)
            if p.exists() and not args.force:
                print(f"  [skip] {p.name}")
                continue
            t0 = time.time()
            b = (_fit_v4 if recipe == "v4" else _fit_gart)(df, feats, mono, seed)
            joblib.dump(b, p)
            print(f"  [fit ] {arm:<13s} seed {seed}  trees={b.num_trees():<5d} "
                  f"{time.time()-t0:6.1f}s -> {p.name}")

    # ---- reproduction gate: arm A at seed 42 must BE the shipped V4 ----------
    ship_v4 = joblib.load(V4_DIR / "lgbm_alpha_model_v4.joblib")
    mine_v4 = joblib.load(_path("V4_unc_32f", 42))
    gate = mine_v4.model_to_string() == ship_v4.model_to_string()
    repro["gate_v4_bit_identical"] = bool(gate)
    print(f"\n[gate] V4_unc_32f@42 == lgbm_alpha_model_v4.joblib : {gate}")
    if not gate:
        raise SystemExit("reproduction gate FAILED: the harness is not refitting "
                         "LGBM_V4, so no arm in this study is reportable")

    # ---- diagnostic (not a pre-registered gate): GART 2.0's recipe ----------
    ship_g = joblib.load(G_DIR / "gart2_final.joblib")
    mine_g = joblib.load(_path("GART2_refit", 42))
    same = mine_g.model_to_string() == ship_g.model_to_string()
    repro["diagnostic_gart_bit_identical"] = bool(same)
    repro["diagnostic_gart_trees_refit"] = int(mine_g.num_trees())
    repro["diagnostic_gart_trees_shipped"] = int(ship_g.num_trees())
    print(f"[diag] GART2_refit@42 == gart2_final.joblib        : {same} "
          f"(trees {mine_g.num_trees()} vs {ship_g.num_trees()})")

    for arm, (block, mono, _) in ARMS.items():
        repro["fits"][arm] = {
            "features": len(blocks[block]), "monotone": mono,
            "role": ARM_ROLES[arm],
            "trees_by_seed": {str(s): int(joblib.load(_path(arm, s)).num_trees())
                              for s in SEEDS}}
    P_REPRO.write_text(json.dumps(repro, indent=2), encoding="utf-8")
    print(f"[fit] -> {P_REPRO}")


# =============================================================================
# Strata evaluation
# =============================================================================
def _eval_frame(blocks) -> tuple[pd.DataFrame, pd.Series]:
    """The benchmark feature cache and ONE eligibility mask for every arm.

    ``v4_study.cmd_evalall`` masks per model, on that model's own columns.  Here
    the mask is computed once on the union of all three blocks, so every arm is
    scored on a byte-identical instance set and a difference between two arms
    cannot be a difference in which rows they were allowed to see.
    """
    C = pd.read_csv(FEAT_CACHE)
    union = sorted({f for b in blocks.values() for f in b} | {"mst_total_length"})
    ok = (C.status == "ok") & C[union].notna().all(axis=1) & C.true_cost.notna()
    got = C[ok].groupby("stratum").size().to_dict()
    want = {"nd_test": 16920, "bench2d": 2580, "tsplib_euc2d": 78,
            "tsplib_noneuc": 22, "augment": 874}
    if got != want:
        raise SystemExit(f"eligible instance counts {got} != published {want}")
    print(f"[eval] one mask, {int(ok.sum())} instances: {got}")
    return C, ok


def _metrics(err_pct: np.ndarray) -> dict[str, float]:
    return {"mape": float(np.mean(np.abs(err_pct))),
            "sdpe": float(np.std(err_pct, ddof=1)),
            "bias": float(np.mean(err_pct)),
            "mspe": float(np.mean(err_pct ** 2))}


def _score(model, feats, target, C, ok) -> tuple[list[dict], pd.DataFrame]:
    rows, tsp = [], []
    for stratum in STRATA_ALL:
        sub = C[ok & (C.stratum == stratum)]
        pred, _ = V4._predict_cost(model, feats, target, sub)
        truth = sub.true_cost.to_numpy(dtype=float)
        err = (pred - truth) / truth * 100.0
        rows.append({"stratum": stratum, "n": int(len(sub)), **_metrics(err)})
        if stratum.startswith("tsplib"):
            tsp.append(pd.DataFrame({"stratum": stratum,
                                     "instance": sub.instance.to_numpy(),
                                     "abs_err_pct": np.abs(err)}))
    return rows, pd.concat(tsp, ignore_index=True)


def cmd_eval(args) -> None:
    blocks = _blocks()
    C, ok = _eval_frame(blocks)

    out, tsp_rows = [], []
    shipped = [("LGBM_V4[shipped]", V4_DIR / "lgbm_alpha_model_v4.joblib",
                blocks["v4_32"], "alpha"),
               ("GART_2.0[shipped]", G_DIR / "gart2_final.joblib",
                blocks["gart_31"], "logit")]
    for label, path, feats, target in shipped:
        rows, tsp = _score(joblib.load(path), feats, target, C, ok)
        for r in rows:
            out.append({"arm": label, "seed": "shipped", "role": ARM_ROLES[label], **r})
        tsp_rows.append(tsp.assign(arm=label, seed="shipped"))

    for arm, (block, _, recipe) in ARMS.items():
        for seed in SEEDS:
            rows, tsp = _score(joblib.load(_path(arm, seed)), blocks[block],
                               TARGET[recipe], C, ok)
            for r in rows:
                out.append({"arm": arm, "seed": str(seed), "role": ARM_ROLES[arm], **r})
            tsp_rows.append(tsp.assign(arm=arm, seed=str(seed)))
        print(f"  [eval] {arm}")

    T = pd.DataFrame(out)
    T.to_csv(P_PERSEED, index=False)
    pd.concat(tsp_rows, ignore_index=True).to_csv(P_TSPLIB_ROWS, index=False)
    print(f"\n[eval] -> {P_PERSEED}  ({len(T)} rows)")

    for metric in ("mape", "sdpe"):
        p = (T[T.seed != "shipped"]
             .pivot_table(index="arm", columns="stratum", values=metric, aggfunc="median")
             .reindex(columns=list(STRATA_ALL)))
        s = (T[T.seed == "shipped"]
             .pivot_table(index="arm", columns="stratum", values=metric)
             .reindex(columns=list(STRATA_ALL)))
        print(f"\n=== {metric.upper()} — median over {len(SEEDS)} seeds "
              f"(shipped artifacts are single points) ===")
        print(pd.concat([s, p]).to_string(float_format=lambda x: f"{x:.4f}"))


# =============================================================================
# Swept-feature monotonicity probe
# =============================================================================
def cmd_probe(args) -> None:
    # Imported here, not at module scope: consistency_31f pins the BLAS thread
    # count at import, and the fit stage must not run under a thread count the
    # shipped artifact was not trained with.
    import consistency_31f as C31

    blocks = _blocks()
    base = C31._probe_base()
    print(f"[probe] {len(base)} ND-test instances @ seed {V4.SEED} | "
          f"d grid {len(V4.PROBE_D_GRID)} pts | n grid {len(V4.PROBE_N_GRID)} pts | "
          f"tol {V4.PROBE_TOL:g}")

    def sweep(model, feats, target, label, seed):
        got = []
        for col, grid in (("dimension", V4.PROBE_D_GRID),
                          ("n_customers", V4.PROBE_N_GRID)):
            r = V4._sweep_monotonicity(model, feats, base, col, grid,
                                       tol=V4.PROBE_TOL, target=target)
            r.update({"arm": label, "seed": seed, "swept": col,
                      "n_probes": len(base), "grid_points": len(grid),
                      "role": ARM_ROLES.get(label, "")})
            got.append(r)
        return got

    print("\n[falsification] the probe must be able to fail")
    ctrl = []
    for ad in C31.build_falsification(blocks["v4_32"]):
        ctrl += sweep(ad, ad.feats, ad.target, ad.label, "control")
    C31._assert_falsification([{**r, "model": r["arm"]} for r in ctrl])

    rows = list(ctrl)
    print("\n[shipped]")
    for label, path, feats, target in (
            ("LGBM_V4[shipped]", V4_DIR / "lgbm_alpha_model_v4.joblib",
             blocks["v4_32"], "alpha"),
            ("GART_2.0[shipped]", G_DIR / "gart2_final.joblib",
             blocks["gart_31"], "logit")):
        got = sweep(joblib.load(path), feats, target, label, "shipped")
        rows += got
        print(f"  {label:<18s} dim {got[0]['pct_nonincr_deployed']:6.1f}%  "
              f"n {got[1]['pct_nonincr_deployed']:6.1f}%")

    print("\n[arms]")
    for arm, (block, _, recipe) in ARMS.items():
        for seed in SEEDS:
            got = sweep(joblib.load(_path(arm, seed)), blocks[block],
                        TARGET[recipe], arm, str(seed))
            rows += got
            print(f"  {arm:<13s} seed {seed}  dim {got[0]['pct_nonincr_deployed']:6.1f}%  "
                  f"n {got[1]['pct_nonincr_deployed']:6.1f}%  "
                  f"viol max (raw) d {got[0]['viol_max_raw']:.2e} "
                  f"n {got[1]['viol_max_raw']:.2e}")

    cols = ["arm", "seed", "role", "swept", "n_probes", "grid_points", "n_pairs",
            "pct_nonincr_deployed", "n_viol_deployed", "viol_med_deployed",
            "viol_p99_deployed", "viol_max_deployed", "pct_nonincr_raw",
            "n_viol_raw", "viol_med_raw", "viol_p99_raw", "viol_max_raw"]
    T = pd.DataFrame(rows).reindex(columns=cols)
    T.to_csv(P_PROBE, index=False)
    print(f"\n[probe] -> {P_PROBE}")
    print(T.pivot_table(index=["arm", "seed"], columns="swept",
                        values="pct_nonincr_deployed", sort=False)
           .to_string(float_format=lambda x: f"{x:.1f}"))


# =============================================================================
# Summary + the pre-registered rule
# =============================================================================
def _band(v: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(v)), "min": float(np.min(v)),
            "max": float(np.max(v)), "n_seeds": int(len(v))}


def cmd_verdict(args) -> None:
    from scipy import stats as sstats

    T = pd.read_csv(P_PERSEED)
    P = pd.read_csv(P_PROBE)
    R = pd.read_csv(P_TSPLIB_ROWS)
    fit = T[T.seed != "shipped"]

    srows = []
    for arm in ARMS:
        for stratum in STRATA_ALL:
            for metric in ("mape", "sdpe", "bias"):
                v = fit[(fit.arm == arm) & (fit.stratum == stratum)][metric].to_numpy(float)
                srows.append({"kind": "strata", "arm": arm, "key": stratum,
                              "metric": metric, **_band(v)})
        for swept in ("dimension", "n_customers"):
            v = P[(P.arm == arm) & (P.swept == swept)]["pct_nonincr_deployed"].to_numpy(float)
            srows.append({"kind": "probe", "arm": arm, "key": swept,
                          "metric": "pct_nonincr_deployed", **_band(v)})
    for _, r in T[T.seed == "shipped"].iterrows():
        for metric in ("mape", "sdpe", "bias"):
            srows.append({"kind": "strata", "arm": r["arm"], "key": r["stratum"],
                          "metric": metric, "median": float(r[metric]),
                          "min": float(r[metric]), "max": float(r[metric]),
                          "n_seeds": 1})
    for _, r in P[P.seed == "shipped"].iterrows():
        srows.append({"kind": "probe", "arm": r["arm"], "key": r["swept"],
                      "metric": "pct_nonincr_deployed",
                      "median": float(r["pct_nonincr_deployed"]),
                      "min": float(r["pct_nonincr_deployed"]),
                      "max": float(r["pct_nonincr_deployed"]), "n_seeds": 1})
    S = pd.DataFrame(srows)
    S.to_csv(P_SUMMARY, index=False)
    print(f"[verdict] -> {P_SUMMARY}")

    def band(arm, kind, key, metric="mape"):
        r = S[(S.arm == arm) & (S["kind"] == kind) & (S.key == key)
              & (S.metric == metric)]
        if len(r) != 1:
            raise SystemExit(f"summary has {len(r)} rows for {arm}/{key}/{metric}")
        return r.iloc[0]

    # -- probe clause -------------------------------------------------------
    probe = {}
    for axis in ("dimension", "n_customers"):
        b = band("V4_mono_32f", "probe", axis, "pct_nonincr_deployed")
        probe[axis] = {"median": b["median"], "min": b["min"], "max": b["max"],
                       "pass": bool(b["median"] == 100.0 and b["min"] == 100.0)}
    probe_pass = all(v["pass"] for v in probe.values())

    # -- accuracy clause ----------------------------------------------------
    strata_verdict = {}
    for stratum in STRATA:
        per_metric = {}
        for metric in ("mape", "sdpe"):
            v4 = band("V4_mono_32f", "strata", stratum, metric)
            g = band("GART2_refit", "strata", stratum, metric)
            margin = float(g["median"] - v4["median"])
            disjoint = bool(v4["max"] < g["min"] or g["max"] < v4["min"])
            per_metric[metric] = {
                "v4_mono_median": v4["median"], "v4_mono_min": v4["min"],
                "v4_mono_max": v4["max"], "gart_median": g["median"],
                "gart_min": g["min"], "gart_max": g["max"],
                "margin_pp_v4_better": round(margin, 6),
                "meets_materiality": bool(margin >= MATERIALITY_PP),
                "bands_disjoint": disjoint,
                "retains": bool(margin >= MATERIALITY_PP and disjoint)}
        n_ret = sum(per_metric[m]["retains"] for m in ("mape", "sdpe"))
        strata_verdict[stratum] = {
            "state": "RETAINED" if n_ret == 2 else ("MIXED" if n_ret == 1 else "LOST"),
            "decisive": stratum in DECISIVE, "by_metric": per_metric}

    retained = [s for s in DECISIVE if strata_verdict[s]["state"] == "RETAINED"]
    if probe_pass and len(retained) == len(DECISIVE):
        outcome = "SHIP-V4"
    elif probe_pass and not retained and all(
            strata_verdict[s]["state"] == "LOST" for s in DECISIVE):
        outcome = "ARGUMENT-COMPLETE"
    else:
        outcome = "FRONTIER"

    # -- supporting paired test on the two small TSPLIB strata --------------
    paired = {}
    for stratum in ("tsplib_euc2d", "tsplib_noneuc"):
        med_seed = {}
        for arm in ("V4_mono_32f", "GART2_refit", "V4_unc_32f"):
            sub = fit[(fit.arm == arm) & (fit.stratum == stratum)]
            med_seed[arm] = str(sub.loc[(sub["mape"] - sub["mape"].median()).abs()
                                        .idxmin(), "seed"])
        a = R[(R.arm == "V4_mono_32f") & (R.seed == med_seed["V4_mono_32f"])
              & (R.stratum == stratum)].set_index("instance")["abs_err_pct"]
        b = R[(R.arm == "GART2_refit") & (R.seed == med_seed["GART2_refit"])
              & (R.stratum == stratum)].set_index("instance")["abs_err_pct"]
        common = a.index.intersection(b.index)
        d = a.loc[common].to_numpy() - b.loc[common].to_numpy()
        paired[stratum] = {
            "median_seed": med_seed, "n": int(len(common)),
            "mean_diff_pp_neg_favours_v4_mono": float(np.mean(d)),
            "n_v4_mono_better": int((d < 0).sum()),
            "wilcoxon_p": float(sstats.wilcoxon(a.loc[common], b.loc[common]).pvalue)}

    # -- symmetric arm: bound the extra feature, do not identify the residual
    sym = {}
    for metric in ("mape", "sdpe"):
        sym[metric] = {}
        for stratum in STRATA:
            g = {a: float(band(a, "strata", stratum, metric)["median"])
                 for a in ("V4_unc_32f", "V4_mono_32f", "V4_unc_31f",
                           "V4_mono_31f", "GART2_refit")}
            sym[metric][stratum] = {
                "medians": {k: round(v, 6) for k, v in g.items()},
                "constraint_cost_32f_pp": round(g["V4_mono_32f"] - g["V4_unc_32f"], 6),
                "constraint_cost_31f_pp": round(g["V4_mono_31f"] - g["V4_unc_31f"], 6),
                "feature_value_unconstrained_pp": round(g["V4_unc_31f"] - g["V4_unc_32f"], 6),
                "feature_value_constrained_pp": round(g["V4_mono_31f"] - g["V4_mono_32f"], 6),
                "residual_mono31f_vs_gart_pp": round(g["V4_mono_31f"] - g["GART2_refit"], 6)}

    payload = {
        "protocol": "paper_tooling/constraint_transfer_protocol.md",
        "protocol_written": "2026-08-11T12:14:53-07:00",
        "seeds": list(SEEDS), "materiality_pp": MATERIALITY_PP,
        "decisive_strata": list(DECISIVE),
        "probe_clause": {"pass": probe_pass, "by_axis": probe},
        "accuracy_clause": strata_verdict,
        "outcome": outcome,
        "supporting_paired_tests": paired,
        "symmetric_arm": sym,
        "nd_test_note": ("Excluded from the decision clause by the protocol: "
                         "unconstrained LGBM_V4's published lead there is "
                         "0.0014 pp, already 36x below the 0.05 pp materiality "
                         "floor fixed before looking."),
        "symmetric_arm_note": ("Three variables still separate V4_mono_31f from "
                               "GART2_refit -- target transform, hyperparameters, "
                               "train+val refit. The symmetric arm bounds the "
                               "contribution of mst_total_length and of the "
                               "constraint; it does not identify the residual."),
    }
    P_VERDICT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[verdict] -> {P_VERDICT}")

    print(f"\n=== PROBE (V4_mono_32f) ===")
    for ax, v in probe.items():
        print(f"  {ax:<12s} median {v['median']:6.1f}  band [{v['min']:.1f}, "
              f"{v['max']:.1f}]  pass={v['pass']}")
    print(f"\n=== ACCURACY vs GART2_refit (median of {len(SEEDS)} seeds) ===")
    for stratum, sv in strata_verdict.items():
        m = sv["by_metric"]["mape"]; s = sv["by_metric"]["sdpe"]
        print(f"  {stratum:<14s} {sv['state']:<8s} decisive={sv['decisive']!s:<5s} "
              f"MAPE {m['v4_mono_median']:.4f} vs {m['gart_median']:.4f} "
              f"(margin {m['margin_pp_v4_better']:+.4f} pp, disjoint={m['bands_disjoint']}) | "
              f"SDPE {s['v4_mono_median']:.4f} vs {s['gart_median']:.4f} "
              f"(margin {s['margin_pp_v4_better']:+.4f} pp, disjoint={s['bands_disjoint']})")
    print(f"\n=== OUTCOME: {outcome} ===")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_fit)
    sub.add_parser("eval").set_defaults(fn=cmd_eval)
    sub.add_parser("probe").set_defaults(fn=cmd_probe)
    sub.add_parser("verdict").set_defaults(fn=cmd_verdict)
    sub.add_parser("all").set_defaults(
        fn=lambda a: (cmd_fit(a), cmd_eval(a), cmd_probe(a), cmd_verdict(a)))
    a = ap.parse_args()
    if not hasattr(a, "force"):
        a.force = False
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
