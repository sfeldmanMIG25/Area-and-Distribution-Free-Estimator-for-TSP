"""Balanced extension of the shortlist + leave-one-out ablation.

``feature_screen.py``'s greedy maximises out-of-fold R^2 on the whole residual.
LineNoise alone is 45% of the residual sum of squares, so that objective is
dominated by the under-prediction case and stops improving the grid
over-prediction case early. The brief asks for features that help BOTH
directions, so this script:

  1. continues the greedy from the shortlist ``feature_screen.py`` produced,
     using a two-sided objective: minimise |MSPE_LineNoise| + |MSPE_grid|
     (the two signed biases the paper reports), rejecting any addition that
     lowers overall out-of-fold R^2;
  2. runs a leave-one-out ablation on the final set, which is what actually
     attributes value to each feature once the others are present.

Same frame, same grouped folds, same corrector hyperparameters as the screen.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import feature_screen as FS  # noqa: E402

MAX_TOTAL = 8
EXT_CSV = os.path.join(ROOT, "paper_tooling", "feature_screen_extended.csv")
ABL_CSV = os.path.join(ROOT, "paper_tooling", "feature_screen_ablation.csv")


def main():
    import joblib
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    frame = pd.read_csv(FS.FRAME_CSV)
    model = joblib.load(FS.MODEL_PATH)
    feat30 = list(model.feature_name_)
    frame = frame.dropna(subset=feat30 + ["optimal_cost", "mst_total_length"]).copy()
    a = (frame["optimal_cost"] / frame["mst_total_length"].replace(0, 1e-9)).clip(1.0, 2.0).to_numpy()
    r = np.clip(model.predict(frame[feat30]), 1.0, 2.0) - a

    is_ln = (frame["generator"] == "line_noise").to_numpy()
    is_grid = (frame["generator"] == "grid").to_numpy()
    is_ndk = (frame["generator"] == "ND_corr_hi").to_numpy()
    is_2d = (frame["source"] == "2D").to_numpy()
    SUB = {"all": None, "LineNoise": is_ln, "grid": is_grid,
           "ND_corr_hi": is_ndk, "2D_bench": is_2d}

    groups = FS.cv_group_labels(frame)
    folds = list(GroupKFold(n_splits=6).split(np.zeros((len(frame), 1)), r, groups))

    def oof(cols):
        pred = np.full(len(frame), np.nan)
        M = frame[cols].to_numpy(float)
        for tr, te in folds:
            m = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=15,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=FS.RANDOM_STATE,
                n_jobs=os.cpu_count(), verbose=-1, deterministic=True,
                force_row_wise=True)
            m.fit(M[tr], r[tr])
            pred[te] = m.predict(M[te])
        return pred

    def rep(pred):
        e = r - pred
        out = {}
        for nm, m in SUB.items():
            y, p, ee, aa = (r, pred, e, a) if m is None else (r[m], pred[m], e[m], a[m])
            out[f"R2_{nm}"] = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
            out[f"MSPE_{nm}"] = float(np.mean(ee / aa) * 100)
            out[f"MAPE_{nm}"] = float(np.mean(np.abs(ee / aa)) * 100)
        out["bias2"] = abs(out["MSPE_LineNoise"]) + abs(out["MSPE_grid"])
        return out

    # ---- pool: same group-stratified construction as the screen -----------
    corr = pd.read_csv(FS.CORR_CSV)
    group_of = dict(zip(corr.feature, corr.group))
    pool = []
    for g in FS.GROUP_MODULES:
        gs = corr[corr.group == g]
        take = set(gs.head(8)["feature"])
        take |= set(gs.reindex(gs.rho_resid_LN.abs().sort_values(ascending=False).index).head(4)["feature"])
        take |= set(gs.reindex(gs.rho_resid_grid.abs().sort_values(ascending=False).index).head(4)["feature"])
        pool += [f for f in gs["feature"] if f in take]

    chosen = pd.read_csv(FS.SHORT_CSV)["feature"].tolist()
    print(f"starting from the R2-greedy shortlist ({len(chosen)}): {chosen}\n")

    cur = rep(oof(feat30 + chosen))
    base = rep(oof(feat30))
    rows = [{"step": 0, "feature": "(base30)", "group": "", **base},
            {"step": len(chosen), "feature": "+".join(chosen), "group": "R2-greedy", **cur}]
    print(f"  base30              bias2={base['bias2']:6.2f}  R2={base['R2_all']:+.4f}  "
          f"MSPE_LN={base['MSPE_LineNoise']:+.2f}% MSPE_grid={base['MSPE_grid']:+.2f}%")
    print(f"  R2-greedy shortlist bias2={cur['bias2']:6.2f}  R2={cur['R2_all']:+.4f}  "
          f"MSPE_LN={cur['MSPE_LineNoise']:+.2f}% MSPE_grid={cur['MSPE_grid']:+.2f}%")

    # ---- two-sided greedy extension ---------------------------------------
    while len(chosen) < MAX_TOTAL:
        best, best_f = None, None
        for f in pool:
            if f in chosen:
                continue
            m = rep(oof(feat30 + chosen + [f]))
            if m["R2_all"] < cur["R2_all"] - 1e-4:      # must not cost overall fit
                continue
            if m["bias2"] < cur["bias2"] - 1e-3 and (best is None or m["bias2"] < best["bias2"]):
                best, best_f = m, f
        if best_f is None:
            print(f"[extend] no two-sided gain at step {len(chosen)+1}; stopping")
            break
        chosen.append(best_f)
        cur = best
        rows.append({"step": len(chosen), "feature": best_f,
                     "group": group_of[best_f], **best})
        print(f"  +{best_f:38s} bias2={cur['bias2']:6.2f}  R2={cur['R2_all']:+.4f}  "
              f"MSPE_LN={cur['MSPE_LineNoise']:+.2f}% MSPE_grid={cur['MSPE_grid']:+.2f}%  "
              f"MAPE_LN={cur['MAPE_LineNoise']:.2f}% MAPE_grid={cur['MAPE_grid']:.2f}%")

    ext = pd.DataFrame(rows)
    ext.to_csv(EXT_CSV, index=False)
    print(f"\nfinal shortlist ({len(chosen)}): {chosen}")

    # ---- leave-one-out ablation -------------------------------------------
    print("\n=== leave-one-out ablation of the final shortlist ===")
    full = cur
    arows = [{"model": f"base30+shortlist{len(chosen)} (full)", **full}]
    for f in chosen:
        m = rep(oof(feat30 + [x for x in chosen if x != f]))
        m["model"] = f"drop {f}"
        m["dR2_all"] = m["R2_all"] - full["R2_all"]
        m["dR2_LineNoise"] = m["R2_LineNoise"] - full["R2_LineNoise"]
        m["dR2_grid"] = m["R2_grid"] - full["R2_grid"]
        m["dbias_LN_pp"] = abs(m["MSPE_LineNoise"]) - abs(full["MSPE_LineNoise"])
        m["dbias_grid_pp"] = abs(m["MSPE_grid"]) - abs(full["MSPE_grid"])
        arows.append(m)
        print(f"  drop {f:38s} dR2={m['dR2_all']:+.4f}  "
              f"LN bias {m['dbias_LN_pp']:+.2f}pp  grid bias {m['dbias_grid_pp']:+.2f}pp")

    abl = pd.DataFrame(arows)
    lead = ["model", "R2_all", "dR2_all", "R2_LineNoise", "dR2_LineNoise", "R2_grid",
            "dR2_grid", "MSPE_LineNoise", "MAPE_LineNoise", "MSPE_grid", "MAPE_grid",
            "dbias_LN_pp", "dbias_grid_pp"]
    abl = abl[[c for c in lead if c in abl] + [c for c in abl if c not in lead]]
    abl.to_csv(ABL_CSV, index=False)
    print("\n" + abl.round(4).to_string(index=False))
    print(f"\nwrote {EXT_CSV}\n      {ABL_CSV}")


if __name__ == "__main__":
    main()
