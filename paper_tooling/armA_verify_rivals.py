"""Task 4: reproduce the oracle-constant audit, then try to BUILD a deployable rival.

The oracle constant is not deployable -- it needs a family label at inference.
The question that decides whether arm A is the only way to get its behaviour is
whether some label-free post-hoc correction of the FROZEN model, given exactly
the same 874 augmentation rows, matches arm A on every stratum.

Rivals (all label-free at inference, all built on the frozen booster):
  global_const   one multiplicative constant, fitted on val + augmentation
  iso_alpha      isotonic map on the predicted alpha, fitted on val + augmentation
  knn_ratio      distance-shrunk kNN correction factor read off the 874 rows
  clf_const      learned regime classifier -> per-regime constant fitted on the
                 augmentation rows only (never on the benchmark)
  resid_gbm      second-stage booster on the frozen logit residual (2x inference)

Reference points: FROZEN, arm A, and the per-family ORACLE constant (in-sample
on the very family it is scored on -- an upper bound, not a rival).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

import armA_verify_common as K

HERE = Path(__file__).resolve().parent
ALPHA_CLIP = K.ALPHA_CLIP


def summ(e: np.ndarray) -> dict:
    e = np.asarray(e, float)
    return {"n": int(e.size), "mape": float(np.mean(np.abs(e))),
            "sdpe": float(np.std(e, ddof=1)), "mspe": float(np.mean(e))}


def best_constant(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    r = true / pred
    grid = np.linspace(r.min(), r.max(), 4001)
    mp = np.array([np.mean(np.abs((g * pred - true) / true)) * 100.0 for g in grid])
    k = int(np.argmin(mp))
    return float(grid[k]), float(mp[k])


def evaluate(name: str, alpha_by_inst: dict, C: pd.DataFrame) -> list[dict]:
    rows = []
    for st, g in C.groupby("stratum"):
        gg = g[g.status == "ok"].copy()
        a = gg["instance"].map(alpha_by_inst)
        gg = gg[a.notna()].copy()
        a = np.clip(a.dropna().to_numpy(float), *ALPHA_CLIP)
        err = (a * gg["mst_total_length"].to_numpy() - gg["true_cost"].to_numpy()) \
            / gg["true_cost"].to_numpy() * 100.0
        rows.append({"model": name, "group": st, **summ(err)})
        if st == "bench2d":
            gg["err"] = err
            gg["pa"] = a
            gg["grp"] = K.group_of(gg["generator"])
            for grp, h in gg.groupby("grp"):
                rows.append({"model": name, "group": f"2d:{grp}", **summ(h["err"].to_numpy())})
            rows.append({"model": name, "group": "2d:ALL", **summ(err)})
            ln = gg[(gg.generator == "line_noise") & (gg.n_customers >= 200)]
            ta = np.clip(ln["true_cost"] / ln["mst_total_length"], *ALPHA_CLIP).to_numpy()
            rows.append({"model": name, "group": "slope:line_noise",
                         "n": len(ln), "mape": np.nan, "sdpe": np.nan,
                         "mspe": float(stats.linregress(ta, ln["pa"].to_numpy()).slope)})
    return rows


def main() -> None:
    D = K.load_cache()
    corpus, aug, C, feats = D["corpus"], D["aug"], D["cache"], D["feats31"]
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]

    frozen = joblib.load(K.FROZEN)
    armA = joblib.load(HERE / "armA_verify_models" / "A_full.joblib")

    Cok = C[C.status == "ok"].copy()
    zf = frozen.predict(Cok[feats], num_iteration=frozen.best_iteration)
    af = np.clip(K.to_alpha(zf), *ALPHA_CLIP)
    za = armA.predict(Cok[feats], num_iteration=armA.best_iteration)
    aa = np.clip(K.to_alpha(za), *ALPHA_CLIP)
    inst = Cok["instance"].astype(str).to_numpy()
    A_FROZEN = dict(zip(inst, af))
    A_ARMA = dict(zip(inst, aa))

    # calibration corpus for the deployable rivals: val (out of sample for the
    # frozen model) plus the 874 augmentation rows. Never the benchmark.
    cal = pd.concat([va.assign(src="val"), aug.assign(src="aug")], ignore_index=True,
                    sort=False)
    zc = frozen.predict(cal[feats], num_iteration=frozen.best_iteration)
    cal["pa"] = np.clip(K.to_alpha(zc), *ALPHA_CLIP)
    cal["ta"] = cal["alpha"].to_numpy()
    cal["pred_cost"] = cal["pa"] * cal["mst_total_length"]

    results: list[dict] = []
    results += evaluate("FROZEN", A_FROZEN, C)
    results += evaluate("armA", A_ARMA, C)

    # ---------------- rival 1: one global constant ------------------------
    c, _ = best_constant(cal["pred_cost"].to_numpy(), cal["optimal_cost"].to_numpy())
    results += evaluate("R_global_const", {k: v * c for k, v in A_FROZEN.items()}, C)
    print(f"[R_global_const] c = {c:.5f}")

    # ---------------- rival 2: isotonic map on alpha ----------------------
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=1.0, y_max=2.0, out_of_bounds="clip")
    iso.fit(cal["pa"].to_numpy(), cal["ta"].to_numpy())
    results += evaluate("R_iso_alpha",
                        {k: float(iso.predict([v])[0]) for k, v in A_FROZEN.items()}, C)

    # ---------------- rival 3: kNN ratio patch ----------------------------
    from scipy.spatial import cKDTree
    mu = tr[feats].mean().to_numpy()
    sd = tr[feats].std(ddof=0).replace(0, 1.0).to_numpy()
    # 13 augmentation rows have non-finite mst_edge_skew/kurtosis (perfectly
    # uniform MST edge lengths). LightGBM tolerates that; a KD-tree does not,
    # so those coordinates are imputed at the training mean (z = 0).
    Za = (aug[feats].to_numpy(float) - mu) / sd
    Za = np.where(np.isfinite(Za), Za, 0.0)
    za_pred = np.clip(K.to_alpha(frozen.predict(aug[feats],
                                                num_iteration=frozen.best_iteration)),
                      *ALPHA_CLIP)
    ratio_a = aug["alpha"].to_numpy() / za_pred          # alpha correction factor
    tree = cKDTree(Za)
    Zq = (Cok[feats].to_numpy(float) - mu) / sd
    dist, idx = tree.query(Zq, k=10)
    for tau in (1.0, 3.0):
        w = np.exp(-dist / tau)
        fac = (w * ratio_a[idx]).sum(1) / np.maximum(w.sum(1), 1e-12)
        shrink = w.max(1)                                # 0 far away, 1 on top of a row
        corr = 1.0 + shrink * (fac - 1.0)
        results += evaluate(f"R_knn_tau{tau:g}", dict(zip(inst, af * corr)), C)

    # ---------------- rival 4: learned regime -> constant -----------------
    import lightgbm as lgb
    lab = pd.concat([
        tr[feats].assign(_y="none"),
        aug[feats].assign(_y=aug["family"].to_numpy()),
    ], ignore_index=True)
    classes = sorted(lab["_y"].unique())
    cmap = {c_: i for i, c_ in enumerate(classes)}
    # 69,768 "none" rows against 874 regime rows collapses an unweighted
    # multiclass model onto "none" for everything, including held-out
    # augmentation rows. Balanced weights are required for this rival to be a
    # real rival rather than a straw man.
    ylab = lab["_y"].map(cmap).to_numpy()
    cnt = np.bincount(ylab, minlength=len(classes))
    wts = (len(ylab) / (len(classes) * cnt))[ylab]
    clf = lgb.train(
        {"objective": "multiclass", "num_class": len(classes), "learning_rate": 0.1,
         "num_leaves": 63, "verbosity": -1, "seed": 42, "num_threads": K.NUM_THREADS,
         "feature_pre_filter": False},
        lgb.Dataset(lab[feats], label=ylab, weight=wts), num_boost_round=300)
    # per-regime constant, fitted on the augmentation rows of that regime only
    const = {"none": 1.0}
    for f_, g in aug.groupby("family"):
        p = np.clip(K.to_alpha(frozen.predict(g[feats],
                                              num_iteration=frozen.best_iteration)),
                    *ALPHA_CLIP) * g["mst_total_length"].to_numpy()
        const[f_], _ = best_constant(p, g["optimal_cost"].to_numpy())
    pr = clf.predict(Cok[feats])
    who = np.array([classes[i] for i in pr.argmax(1)])
    conf = pr.max(1)
    for thr in (0.5, 0.8):
        cvec = np.array([const[w] if cf >= thr else 1.0 for w, cf in zip(who, conf)])
        results += evaluate(f"R_clf_const_t{thr:g}", dict(zip(inst, af * cvec)), C)
    # soft version: blend the regime constants by class probability
    cvec = pr @ np.array([const[c_] for c_ in classes])
    results += evaluate("R_clf_const_soft", dict(zip(inst, af * cvec)), C)
    print(f"[R_clf_const] regime constants = "
          f"{ {k: round(v, 4) for k, v in const.items()} }")
    b2m = (Cok["stratum"] == "bench2d").to_numpy()
    diag = pd.crosstab(Cok.loc[b2m, "generator"].fillna("other"), who[b2m])
    print("[R_clf_const] predicted regime on bench2d (rows = true generator):")
    print(diag.loc[[i for i in ("grid", "line_noise", "clustered") if i in diag.index]]
          .to_string())
    print("[R_clf_const] mean confidence on grid/line_noise: "
          f"{conf[b2m][Cok.loc[b2m, 'generator'].isin(['grid', 'line_noise']).to_numpy()].mean():.3f}")

    # ---------------- rival 5: residual booster ---------------------------
    tr_aug = pd.concat([tr, aug[[c_ for c_ in tr.columns if c_ in aug.columns]]],
                       ignore_index=True, sort=False)
    for df in (tr_aug, va):
        df["_zf"] = frozen.predict(df[feats], num_iteration=frozen.best_iteration)
    mst_v = va["mst_total_length"].to_numpy()
    cost_v = va["optimal_cost"].to_numpy()
    zf_v = va["_zf"].to_numpy()

    def feval(preds, _ds):
        a = np.clip(K.to_alpha(zf_v + preds), *ALPHA_CLIP)
        return "cost_mape", float(np.mean(np.abs((a * mst_v - cost_v) / cost_v)) * 100), False

    dtr = lgb.Dataset(tr_aug[feats], label=K.to_z(tr_aug["alpha"].to_numpy())
                      - tr_aug["_zf"].to_numpy())
    dvl = lgb.Dataset(va[feats], label=K.to_z(va["alpha"].to_numpy()) - zf_v,
                      reference=dtr)
    rp = dict(K.V3_FROZEN)
    rp.update({"objective": "regression_l2", "metric": "None", "seed": 42,
               "num_threads": K.NUM_THREADS, "verbosity": -1,
               "feature_pre_filter": False})
    rb = lgb.train(rp, dtr, num_boost_round=2000, valid_sets=[dvl], feval=feval,
                   callbacks=[lgb.early_stopping(100, verbose=False),
                              lgb.log_evaluation(0)])
    zr = zf + rb.predict(Cok[feats], num_iteration=rb.best_iteration)
    results += evaluate("R_resid_gbm", dict(zip(inst, np.clip(K.to_alpha(zr), *ALPHA_CLIP))), C)
    print(f"[R_resid_gbm] trees {rb.num_trees()}")

    # 5b. the plain residual booster stops after a handful of trees because the
    # early-stopping set (val) contains no degenerate geometry, so it cannot see
    # the correction it is being asked to learn. Give the rival its best shot:
    # hold out a quarter of the augmentation rows and put them in the stopper.
    rs = np.random.default_rng(0).permutation(len(aug))
    a_fit, a_stop = aug.iloc[rs[len(aug) // 4:]], aug.iloc[rs[:len(aug) // 4]]
    cols = [c_ for c_ in tr.columns if c_ in aug.columns]
    fit2 = pd.concat([tr, a_fit[cols]], ignore_index=True, sort=False)
    stop2 = pd.concat([va, a_stop[cols]], ignore_index=True, sort=False)
    for df in (fit2, stop2):
        df["_zf"] = frozen.predict(df[feats], num_iteration=frozen.best_iteration)
    mst_s = stop2["mst_total_length"].to_numpy()
    cost_s = stop2["optimal_cost"].to_numpy()
    zf_s = stop2["_zf"].to_numpy()

    def feval2(preds, _ds):
        a = np.clip(K.to_alpha(zf_s + preds), *ALPHA_CLIP)
        return "cost_mape", float(np.mean(np.abs((a * mst_s - cost_s) / cost_s)) * 100), False

    d2 = lgb.Dataset(fit2[feats], label=K.to_z(fit2["alpha"].to_numpy()) - fit2["_zf"].to_numpy())
    v2 = lgb.Dataset(stop2[feats], label=K.to_z(stop2["alpha"].to_numpy()) - zf_s, reference=d2)
    rb2 = lgb.train(rp, d2, num_boost_round=2000, valid_sets=[v2], feval=feval2,
                    callbacks=[lgb.early_stopping(100, verbose=False),
                               lgb.log_evaluation(0)])
    zr2 = zf + rb2.predict(Cok[feats], num_iteration=rb2.best_iteration)
    results += evaluate("R_resid_gbm_v2",
                        dict(zip(inst, np.clip(K.to_alpha(zr2), *ALPHA_CLIP))), C)
    print(f"[R_resid_gbm_v2] trees {rb2.num_trees()}")

    # 5c. residual booster fitted on the augmentation rows ALONE -- the purest
    # "patch only where the frozen model is known to be wrong" rival.
    d3 = lgb.Dataset(a_fit[feats], label=K.to_z(a_fit["alpha"].to_numpy())
                     - frozen.predict(a_fit[feats], num_iteration=frozen.best_iteration))
    v3 = lgb.Dataset(a_stop[feats], label=K.to_z(a_stop["alpha"].to_numpy())
                     - frozen.predict(a_stop[feats], num_iteration=frozen.best_iteration),
                     reference=d3)
    rp3 = dict(rp)
    rp3.update({"metric": "l2", "min_child_samples": 5, "num_leaves": 15})
    rb3 = lgb.train(rp3, d3, num_boost_round=2000, valid_sets=[v3],
                    callbacks=[lgb.early_stopping(100, verbose=False),
                               lgb.log_evaluation(0)])
    zr3 = zf + rb3.predict(Cok[feats], num_iteration=rb3.best_iteration)
    results += evaluate("R_resid_augonly",
                        dict(zip(inst, np.clip(K.to_alpha(zr3), *ALPHA_CLIP))), C)
    print(f"[R_resid_augonly] trees {rb3.num_trees()}")

    RES = pd.DataFrame(results)
    RES.to_csv(HERE / "armA_verify_rivals.csv", index=False)

    # ---------------- oracle constant audit (reproduce prior claim) -------
    b2 = C[(C.stratum == "bench2d") & (C.status == "ok")].copy()
    b2["af"] = b2["instance"].map(A_FROZEN)
    b2["aa"] = b2["instance"].map(A_ARMA)
    orc = []
    for gen in ("line_noise", "grid"):
        g = b2[b2.generator == gen]
        true = g["true_cost"].to_numpy()
        pf = g["af"].to_numpy() * g["mst_total_length"].to_numpy()
        pa = g["aa"].to_numpy() * g["mst_total_length"].to_numpy()
        c_, mape_rc = best_constant(pf, true)
        e_f = np.abs((pf - true) / true) * 100
        e_r = np.abs((c_ * pf - true) / true) * 100
        e_a = np.abs((pa - true) / true) * 100
        w = stats.wilcoxon(e_a, e_r)
        gain_c = e_f.mean() - e_r.mean()
        gain_a = e_f.mean() - e_a.mean()
        ln = g[g.n_customers >= 200]
        ta = np.clip(ln["true_cost"] / ln["mst_total_length"], *ALPHA_CLIP).to_numpy()
        sl_f = stats.linregress(ta, ln["af"].to_numpy()).slope
        sl_r = stats.linregress(ta, np.clip(ln["af"] * c_, *ALPHA_CLIP)).slope
        sl_a = stats.linregress(ta, ln["aa"].to_numpy()).slope
        orc.append({"family": gen, "n": len(g), "oracle_constant": c_,
                    "frozen_mape": e_f.mean(), "recal_mape": e_r.mean(),
                    "armA_mape": e_a.mean(),
                    "gain_constant": gain_c, "gain_armA": gain_a,
                    "frac_of_armA_gain_by_constant": gain_c / gain_a,
                    "wilcoxon_p_armA_vs_recal": float(w.pvalue),
                    "slope_frozen": sl_f, "slope_recal": sl_r, "slope_armA": sl_a})
    OR = pd.DataFrame(orc)
    OR.to_csv(HERE / "armA_verify_oracle_constant.csv", index=False)

    pd.set_option("display.width", 240)
    f4 = lambda x: f"{x:.4f}"  # noqa: E731
    print("\n============ ORACLE PER-FAMILY CONSTANT (not deployable) ============")
    print(OR.to_string(index=False, float_format=f4))
    print("\n============ RIVALS ============")
    piv = RES.pivot_table(index="model", columns="group", values="mape")
    order = ["FROZEN", "armA", "R_global_const", "R_iso_alpha", "R_knn_tau1",
             "R_knn_tau3", "R_clf_const_t0.5", "R_clf_const_t0.8", "R_clf_const_soft", "R_resid_gbm",
             "R_resid_gbm_v2", "R_resid_augonly"]
    cols = ["nd_test", "2d:ALL", "2d:grid", "2d:line_noise", "2d:cluster",
            "2d:others", "tsplib_euc2d", "tsplib_noneuc"]
    print("MAPE")
    print(piv.reindex(order)[cols].to_string(float_format=f4))
    sl = RES[RES.group == "slope:line_noise"].set_index("model")["mspe"]
    sd = RES.pivot_table(index="model", columns="group", values="sdpe")
    print("\nline_noise calibration slope (n>=200) / tsplib_euc2d SDPE / nd SDPE")
    print(pd.DataFrame({"slope": sl.reindex(order),
                        "tsplib_euc2d_sdpe": sd.reindex(order)["tsplib_euc2d"],
                        "nd_sdpe": sd.reindex(order)["nd_test"]})
          .to_string(float_format=f4))
    json.dump({"global_const": c, "regime_const": const},
              open(HERE / "armA_verify_rivals.json", "w"), indent=2)
    print("\nwrote armA_verify_rivals.csv, armA_verify_oracle_constant.csv")


if __name__ == "__main__":
    main()
