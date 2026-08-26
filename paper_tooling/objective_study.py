"""Objective / target-parameterisation study for the GART 2.0 alpha model.

The shipped model minimises squared error on alpha, but the paper reports
percent error on cost.  Because

    (cost_hat - cost) / cost = (alpha_hat - alpha) / alpha,

MAPE on cost is E[|alpha_hat - alpha| / alpha] and SDPE on cost is
sd((alpha_hat - alpha)/alpha).  The matched training losses are therefore
L1 on alpha with weight 1/alpha, and L2 on alpha with weight 1/alpha^2.

This script keeps the existing 30 features and the existing train/val/test
split fixed and varies only the objective, the sample weights, the target
parameterisation, the early-stopping metric and the hyperparameters.

Stages (``--stage``):
    A      reproduce the shipped artifact bit-identically
    BCD    objective sweep, target transforms, early-stopping metric
    E      Optuna TPE search minimising validation MAPE on cost
    tests  paired Wilcoxon + bootstrap for the winner vs the shipped model
    all    A, BCD, E, tests

Usage:
    python paper_tooling/objective_study.py --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLING = os.path.join(ROOT, "paper_tooling")
MODEL_DIR = os.path.join(TOOLING, "objective_models")
RESULTS_CSV = os.path.join(TOOLING, "objective_study.csv")
SHIPPED = os.path.join(ROOT, "lgbm_model_v3", "lgbm_alpha_model_v3.joblib")
BEST_PARAMS = os.path.join(ROOT, "lgbm_model_v3", "best_params_v3.json")

SEED = 42
NUM_THREADS = 6
MAX_ROUNDS = 3000
ES_ROUNDS = 100

ALPHA_LO, ALPHA_HI = 1.0, 2.0

# Decision rule, fixed in advance.
ND_TOL = 0.05      # max allowed ND-test MAPE regression, percentage points
TSPLIB_TOL = 0.15  # max allowed TSPLIB MAPE regression
CLASS_TOL = 0.15   # max allowed regression on any reported 2D class


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _feature_names() -> list[str]:
    return list(joblib.load(SHIPPED).feature_name_)


class Data:
    """All four evaluation sets plus the ND training split."""

    def __init__(self) -> None:
        self.features = _feature_names()
        self._load_nd()
        self._load_2d()
        self._load_tsplib()
        self._load_augment()

    # -- (train) + (a) ND test -------------------------------------------
    def _load_nd(self) -> None:
        df = pd.read_csv(os.path.join(ROOT, "tsp_features_v3.csv"))
        mst = df["mst_total_length"].replace(0, 1e-9)
        df["alpha"] = (df["optimal_cost"] / mst).clip(ALPHA_LO, ALPHA_HI)
        drop = ["instance_name", "optimal_cost", "alpha", "split",
                "grid_size", "mst_total_length"]
        X = df.drop(columns=[c for c in drop if c in df.columns])
        assert list(X.columns) == self.features, "feature order drift"

        tr, va, te = (df["split"] == s for s in ("train", "val", "test"))
        self.X_train, self.y_train = X[tr], df.loc[tr, "alpha"].to_numpy()
        self.X_val, self.y_val = X[va], df.loc[va, "alpha"].to_numpy()
        self.val = dict(X=X[va],
                        mst=df.loc[va, "mst_total_length"].to_numpy(),
                        cost=df.loc[va, "optimal_cost"].to_numpy())
        self.nd_test = dict(X=X[te],
                            mst=df.loc[te, "mst_total_length"].to_numpy(),
                            cost=df.loc[te, "optimal_cost"].to_numpy())

    # -- (b) 2D benchmark -------------------------------------------------
    def _load_2d(self) -> None:
        f = pd.read_csv(os.path.join(TOOLING, "augmentation_2d_features.csv"))
        gt = pd.read_csv(os.path.join(
            ROOT, "Generalized_TSP_Analysis", "benchmark_checkpoints",
            "base_ground_truth_2d.csv"))
        m = f.merge(gt[["instance", "true_cost"]], left_on="instance_name",
                    right_on="instance", validate="1:1")
        # generator class straight from the instance-name grammar
        gen = m["instance_name"].str.extract(r"^TSP-(.+?)-n\d+-g")[0]
        assert (gen == m["generator"]).all(), "generator grammar mismatch"
        self.bench2d = dict(X=m[self.features],
                            mst=m["mst_total_length"].to_numpy(),
                            cost=m["true_cost"].to_numpy(),
                            gen=gen.to_numpy(),
                            gen_class=m["gen_class"].to_numpy())

    # -- (c) TSPLIB EUC_2D ------------------------------------------------
    def _load_tsplib(self) -> None:
        f = pd.read_csv(os.path.join(TOOLING, "tsplib_features_v3.csv"))
        r = pd.read_csv(os.path.join(ROOT, "tsplib_benchmark", "results",
                                     "all_models_tsplib.csv"))
        r = r[(r["model"] == "LGBM_V3") & (r["status"] == "ok")]
        m = f.merge(r[["instance", "true_cost", "edge_weight_type"]],
                    left_on="instance_name", right_on="instance",
                    suffixes=("", "_r"), validate="1:1")
        m = m[m["edge_weight_type"] == "EUC_2D"].reset_index(drop=True)
        self.tsplib = dict(X=m[self.features],
                           mst=m["mst_total_length"].to_numpy(),
                           cost=m["true_cost"].to_numpy(),
                           name=m["instance_name"].to_numpy())

    # -- (d) novel-geometry augmentation corpus ---------------------------
    def _load_augment(self) -> None:
        a = pd.read_csv(os.path.join(TOOLING, "augment_features_v3.csv"))
        self.augment = dict(X=a[self.features],
                            mst=a["mst_total_length"].to_numpy(),
                            cost=a["optimal_cost"].to_numpy(),
                            family=a["family"].to_numpy())


# --------------------------------------------------------------------------
# target parameterisations
# --------------------------------------------------------------------------

_EPS = 1e-6


def _t_ident(a):
    return a


def _i_ident(z):
    return z


def _t_log(a):
    return np.log(a)


def _i_log(z):
    return np.exp(z)


def _t_logit(a):
    u = np.clip(a - ALPHA_LO, _EPS, 1.0 - _EPS)
    return np.log(u / (1.0 - u))


def _i_logit(z):
    return ALPHA_LO + 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


TRANSFORMS = {"alpha": (_t_ident, _i_ident),
              "log": (_t_log, _i_log),
              "logit": (_t_logit, _i_logit)}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def pct_err(pred_alpha, mst, cost):
    return (np.clip(pred_alpha, ALPHA_LO, ALPHA_HI) * mst - cost) / cost


def summarise(e):
    return dict(N=int(len(e)),
                MAPE=float(np.mean(np.abs(e)) * 100.0),
                SDPE=float(np.std(e, ddof=1) * 100.0),
                MSPE=float(np.mean(e) * 100.0))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def weights_for(alpha, scheme):
    if scheme == "none":
        return None
    if scheme == "inv_alpha":
        return 1.0 / alpha
    if scheme == "inv_alpha2":
        return 1.0 / alpha ** 2
    raise ValueError(scheme)


def make_cost_mape_eval(inv, mst, cost):
    """sklearn-style eval_metric: f(y_true, y_pred) -> (name, value, higher_is_better)."""
    def _f(y_true, y_pred):
        a = np.clip(inv(np.asarray(y_pred)), ALPHA_LO, ALPHA_HI)
        return "cost_mape", float(np.mean(np.abs(a * mst - cost) / cost) * 100.0), False
    return _f


NATIVE_METRIC = {"regression_l2": "rmse", "regression_l1": "l1",
                 "huber": "huber", "mape": "mape"}


def fit(cfg, d, params=None, threads=NUM_THREADS):
    """Train one configuration.  Returns (booster, extra)."""
    tf, inv = TRANSFORMS[cfg.get("target", "alpha")]
    p = dict(json.load(open(BEST_PARAMS)) if params is None else params)
    p.update(cfg.get("param_override", {}))

    w_tr = weights_for(d.y_train, cfg.get("weight", "none"))
    w_va = weights_for(d.y_val, cfg.get("weight", "none"))

    kw = dict(objective=cfg.get("objective", "regression_l2"),
              n_estimators=MAX_ROUNDS, random_state=SEED,
              n_jobs=threads, verbose=-1)
    if cfg.get("es", "native") == "cost_mape":
        kw["metric"] = "None"
        eval_metric = make_cost_mape_eval(inv, d.val["mst"], d.val["cost"])
    else:
        eval_metric = NATIVE_METRIC[cfg.get("objective", "regression_l2")]

    model = lgb.LGBMRegressor(**p, **kw)
    model.fit(d.X_train, tf(d.y_train), sample_weight=w_tr,
              eval_set=[(d.X_val, tf(d.y_val))], eval_sample_weight=
              None if w_va is None else [w_va],
              eval_metric=eval_metric,
              callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False,
                                            first_metric_only=True)])
    return model, inv


def predict_alpha(model, inv, X, factor=1.0):
    z = model.predict(X, num_iteration=model.best_iteration_)
    return np.clip(inv(np.asarray(z)) * factor, ALPHA_LO, ALPHA_HI)


def smearing_factor(model, inv, d, kind):
    """Retransformation correction estimated on the validation split."""
    z = model.predict(d.X_val, num_iteration=model.best_iteration_)
    if kind == "duan":            # Duan smearing for the log link
        resid = np.log(d.y_val) - np.asarray(z)
        return float(np.mean(np.exp(resid)))
    a = inv(np.asarray(z))        # plain multiplicative calibration
    return float(np.mean(d.y_val) / np.mean(a))


# --------------------------------------------------------------------------
# evaluation over all four sets
# --------------------------------------------------------------------------

def evaluate(name, model, inv, d, factor=1.0, extra=None):
    rows = []

    def add(setname, e, n_note=""):
        r = dict(config=name, set=setname)
        r.update(summarise(e))
        r["note"] = n_note
        rows.append(r)

    e_nd = pct_err(predict_alpha(model, inv, d.nd_test["X"], factor),
                   d.nd_test["mst"], d.nd_test["cost"])
    add("nd_test", e_nd)

    b = d.bench2d
    e2 = pct_err(predict_alpha(model, inv, b["X"], factor), b["mst"], b["cost"])
    add("2d_all", e2)
    for c in sorted(set(b["gen_class"])):
        add(f"2d_class:{c}", e2[b["gen_class"] == c])
    for g in ("line_noise", "grid"):
        add(f"2d_gen:{g}", e2[b["gen"] == g])

    t = d.tsplib
    e_t = pct_err(predict_alpha(model, inv, t["X"], factor), t["mst"], t["cost"])
    add("tsplib_euc2d", e_t)

    a = d.augment
    e_a = pct_err(predict_alpha(model, inv, a["X"], factor), a["mst"], a["cost"])
    add("augment", e_a)

    va = pct_err(predict_alpha(model, inv, d.val["X"], factor),
                 d.val["mst"], d.val["cost"])
    add("val", va)

    for r in rows:
        r["best_iteration"] = int(model.best_iteration_)
        if extra:
            r["note"] = (r["note"] + " " + extra).strip()
    return rows, dict(nd=e_nd, tsplib=e_t, augment=e_a, val=va, bench2d=e2)


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------

BASE_CONFIGS = [
    dict(name="A_baseline_l2", objective="regression_l2", weight="none"),
    dict(name="B1_l1_unw", objective="regression_l1", weight="none"),
    dict(name="B2_l1_w_inv_alpha", objective="regression_l1", weight="inv_alpha"),
    dict(name="B3_l2_w_inv_alpha2", objective="regression_l2", weight="inv_alpha2"),
    dict(name="B4a_huber_unw", objective="huber", weight="none"),
    dict(name="B4b_huber_w_inv_alpha", objective="huber", weight="inv_alpha"),
    dict(name="B4c_huber_d0.02_unw", objective="huber", weight="none",
         param_override=dict(alpha=0.02)),
    dict(name="B5_mape_builtin", objective="mape", weight="none"),
    dict(name="C1_l2_log", objective="regression_l2", weight="none", target="log"),
    dict(name="C2_l2_logit", objective="regression_l2", weight="none", target="logit"),
]


def run_bcd(d):
    rows, errs, models = [], {}, {}
    for es in ("native", "cost_mape"):
        for base in BASE_CONFIGS:
            cfg = dict(base)
            cfg["es"] = es
            name = cfg["name"] + ("" if es == "native" else "_esMAPE")
            t0 = time.time()
            model, inv = fit(cfg, d)
            r, e = evaluate(name, model, inv, d)
            rows += r
            errs[name] = e
            models[name] = (model, inv, 1.0)
            print(f"  {name:34s} it={model.best_iteration_:5d} "
                  f"ND={e['nd'].__abs__().mean()*100:.4f} "
                  f"TSPLIB={np.abs(e['tsplib']).mean()*100:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

            # retransformation corrections share the fitted booster
            if cfg.get("target") in ("log", "logit"):
                kind = "duan" if cfg["target"] == "log" else "mult"
                f = smearing_factor(model, inv, d, kind)
                cname = name + f"_corr"
                r, e = evaluate(cname, model, inv, d, factor=f,
                                extra=f"corr_factor={f:.6f}")
                rows += r
                errs[cname] = e
                models[cname] = (model, inv, f)
                print(f"  {cname:34s} factor={f:.6f} "
                      f"ND={np.abs(e['nd']).mean()*100:.4f} "
                      f"TSPLIB={np.abs(e['tsplib']).mean()*100:.4f}", flush=True)
    return rows, errs, models


# --------------------------------------------------------------------------
# Optuna
# --------------------------------------------------------------------------

def run_optuna(d, best_cfg, n_trials=150, n_jobs=3):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 512),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.4, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            max_depth=trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12, 16]),
        )
        cfg = dict(best_cfg)
        cfg["es"] = "cost_mape"
        model, inv = fit(cfg, d, params=params)
        e = pct_err(predict_alpha(model, inv, d.val["X"]),
                    d.val["mst"], d.val["cost"])
        return float(np.mean(np.abs(e)) * 100.0)

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs,
                   show_progress_bar=False)
    print(f"  optuna: {len(study.trials)} trials in {time.time()-t0:.0f}s, "
          f"best val cost-MAPE {study.best_value:.4f}")
    print(f"  best params: {json.dumps(study.best_params)}")
    return study


# --------------------------------------------------------------------------
# paired tests
# --------------------------------------------------------------------------

def paired_tests(e_new, e_ref, label, n_boot=1000, seed=SEED):
    from scipy.stats import wilcoxon
    a, b = np.abs(e_new) * 100.0, np.abs(e_ref) * 100.0
    dif = a - b
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        stat, p = float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dif), size=(n_boot, len(dif)))
    means = dif[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(set=label, n=int(len(dif)),
                mean_abs_new=float(a.mean()), mean_abs_ref=float(b.mean()),
                mean_paired_diff=float(dif.mean()),
                boot_ci_lo=float(lo), boot_ci_hi=float(hi),
                wilcoxon_stat=float(stat), wilcoxon_p=float(p),
                n_better=int((dif < 0).sum()), n_worse=int((dif > 0).sum()))


# --------------------------------------------------------------------------
# decision rule
# --------------------------------------------------------------------------

def screen(rows_df, baseline="A_baseline_l2"):
    piv = rows_df.pivot_table(index="config", columns="set", values="MAPE")
    base = piv.loc[baseline]
    class_cols = [c for c in piv.columns if c.startswith("2d_class:")
                  or c.startswith("2d_gen:")]
    out = []
    for cfg in piv.index:
        if cfg == baseline:
            continue
        r = piv.loc[cfg]
        d_nd = r["nd_test"] - base["nd_test"]
        d_tl = r["tsplib_euc2d"] - base["tsplib_euc2d"]
        worst_cls = max(r[c] - base[c] for c in class_cols)
        ok = ((d_nd < 0 or d_tl < 0)
              and d_nd <= ND_TOL and d_tl <= TSPLIB_TOL
              and worst_cls <= CLASS_TOL)
        out.append(dict(config=cfg, nd_mape=r["nd_test"], d_nd=d_nd,
                        tsplib_mape=r["tsplib_euc2d"], d_tsplib=d_tl,
                        worst_2d_class_delta=worst_cls,
                        augment_mape=r["augment"], passes=ok))
    return pd.DataFrame(out).sort_values(["passes", "tsplib_mape", "nd_mape"],
                                         ascending=[False, True, True])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["A", "BCD", "E", "all"])
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--optuna-jobs", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading data ...", flush=True)
    d = Data()
    print(f"  train {len(d.X_train)}  val {len(d.X_val)}  "
          f"nd_test {len(d.nd_test['X'])}  2d {len(d.bench2d['X'])}  "
          f"tsplib {len(d.tsplib['X'])}  augment {len(d.augment['X'])}",
          flush=True)

    # ---- A: reproduction check ----
    print("\n[A] reproduction check", flush=True)
    shipped = joblib.load(SHIPPED)
    model, inv = fit(dict(name="A", objective="regression_l2", weight="none",
                          es="native"), d)
    p_new = model.predict(d.nd_test["X"], num_iteration=model.best_iteration_)
    p_old = shipped.predict(d.nd_test["X"], num_iteration=shipped.best_iteration_)
    maxdiff = float(np.abs(p_new - p_old).max())
    e = pct_err(np.clip(p_new, ALPHA_LO, ALPHA_HI), d.nd_test["mst"],
                d.nd_test["cost"])
    s = summarise(e)
    print(f"  best_iteration {model.best_iteration_} (expected 2031)")
    print(f"  max abs prediction difference {maxdiff}")
    print(f"  ND test MAPE {s['MAPE']:.4f} (expected 0.8769)")
    assert model.best_iteration_ == 2031, "best_iteration mismatch"
    assert maxdiff == 0.0, "predictions differ from shipped artifact"
    assert abs(s["MAPE"] - 0.8769) < 5e-4, "ND MAPE mismatch"
    print("  REPRODUCTION OK", flush=True)

    # shipped-model reference errors on every set
    ref_rows, ref_err = evaluate("SHIPPED", shipped, _i_ident, d)
    if args.stage == "A":
        for r in ref_rows:
            print(r)
        return

    all_rows = list(ref_rows)
    all_err = {"SHIPPED": ref_err}
    all_models = {}

    # ---- B / C / D ----
    print("\n[B/C/D] objective, target transform and early-stopping sweep",
          flush=True)
    rows, errs, models = run_bcd(d)
    all_rows += rows
    all_err.update(errs)
    all_models.update(models)

    df = pd.DataFrame(all_rows)
    sc = screen(df)
    print("\n--- screen after B/C/D (sorted by TSPLIB MAPE) ---")
    print(sc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # pick the objective family to hand to Optuna: best validation cost MAPE
    val_mape = df[df["set"] == "val"].set_index("config")["MAPE"]
    cand = [c for c in val_mape.index if c not in ("SHIPPED",)]
    best_name = val_mape.loc[cand].idxmin()
    print(f"\nBest configuration by validation cost-MAPE: {best_name} "
          f"({val_mape[best_name]:.4f})")

    # ---- E ----
    if args.stage in ("E", "all"):
        stem = best_name.replace("_esMAPE", "").replace("_corr", "")
        best_cfg = next(c for c in BASE_CONFIGS if c["name"] == stem)
        print(f"\n[E] Optuna TPE on {stem}, {args.trials} trials, "
              f"objective = validation MAPE on cost", flush=True)
        study = run_optuna(d, best_cfg, n_trials=args.trials,
                           n_jobs=args.optuna_jobs)
        cfg = dict(best_cfg)
        cfg["es"] = "cost_mape"
        model, inv = fit(cfg, d, params=study.best_params)
        rows, e = evaluate("E_optuna_" + stem, model, inv, d,
                           extra=json.dumps(study.best_params))
        all_rows += rows
        all_err["E_optuna_" + stem] = e
        all_models["E_optuna_" + stem] = (model, inv, 1.0)
        with open(os.path.join(MODEL_DIR, "optuna_best_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=2)

        # same hyperparameters, plain L2 objective -- isolates objective vs HP
        cfg2 = dict(BASE_CONFIGS[0])
        cfg2["es"] = "cost_mape"
        m2, inv2 = fit(cfg2, d, params=study.best_params)
        rows, e2 = evaluate("E_optuna_l2_control", m2, inv2, d)
        all_rows += rows
        all_err["E_optuna_l2_control"] = e2
        all_models["E_optuna_l2_control"] = (m2, inv2, 1.0)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote {RESULTS_CSV} ({len(df)} rows)")

    sc = screen(df)
    sc.to_csv(os.path.join(TOOLING, "objective_study_screen.csv"), index=False)
    print("\n--- final screen (decision rule) ---")
    print(sc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- save anything that passes, and run the paired tests ----
    winners = sc[sc["passes"]]
    print("\n--- paired tests vs shipped model ---")
    if winners.empty:
        print("  No configuration passes the pre-registered decision rule.")
    for cfg in winners["config"]:
        model, inv, factor = all_models[cfg]
        joblib.dump(dict(model=model, inverse=cfg, factor=factor),
                    os.path.join(MODEL_DIR, f"{cfg}.joblib"))
    top = list(winners["config"])[:3]
    tests = []
    for cfg in top:
        for s in ("tsplib", "augment"):
            t = paired_tests(all_err[cfg][s], ref_err[s], s)
            t["config"] = cfg
            tests.append(t)
            print(f"  {cfg:34s} {s:8s} d={t['mean_paired_diff']:+.4f} "
                  f"CI[{t['boot_ci_lo']:+.4f},{t['boot_ci_hi']:+.4f}] "
                  f"W={t['wilcoxon_stat']:.0f} p={t['wilcoxon_p']:.4g} "
                  f"better/worse={t['n_better']}/{t['n_worse']}")
    if tests:
        pd.DataFrame(tests).to_csv(
            os.path.join(TOOLING, "objective_study_paired_tests.csv"),
            index=False)


if __name__ == "__main__":
    sys.exit(main())
