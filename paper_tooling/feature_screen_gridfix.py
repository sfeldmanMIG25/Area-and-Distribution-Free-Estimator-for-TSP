"""Grid-targeted extension of the shortlist.

The R^2-greedy shortlist leaves the grid over-prediction at MSPE +7.0%, while
the whole 30-feature ``local_id`` group reaches +4.8%. So the grid fix exists
but needs more than one local-PCA feature. This script greedily extends the
shortlist to at most 8 features minimising |MSPE_grid|, subject to overall
out-of-fold R^2 staying within ``R2_TOL`` of the R^2-greedy shortlist, and
reports the resulting two-sided position.
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
R2_TOL = 0.05
OUT_CSV = os.path.join(ROOT, "paper_tooling", "feature_screen_gridfix.csv")


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

    corr = pd.read_csv(FS.CORR_CSV)
    group_of = dict(zip(corr.feature, corr.group))
    pool = []
    for g in FS.GROUP_MODULES:
        gs = corr[corr.group == g]
        take = set(gs.head(8)["feature"])
        take |= set(gs.reindex(gs.rho_resid_LN.abs().sort_values(ascending=False).index).head(4)["feature"])
        take |= set(gs.reindex(gs.rho_resid_grid.abs().sort_values(ascending=False).index).head(6)["feature"])
        pool += [f for f in gs["feature"] if f in take]

    chosen = pd.read_csv(FS.SHORT_CSV)["feature"].tolist()
    cur = rep(oof(feat30 + chosen))
    floor = cur["R2_all"] - R2_TOL
    rows = [{"step": len(chosen), "feature": "+".join(chosen), "group": "R2-greedy", **cur}]
    print(f"start: {chosen}\n  R2={cur['R2_all']:+.4f} MSPE_LN={cur['MSPE_LineNoise']:+.2f}% "
          f"MSPE_grid={cur['MSPE_grid']:+.2f}%   (R2 floor for extension {floor:.4f})\n")

    while len(chosen) < MAX_TOTAL:
        best, best_f = None, None
        for f in pool:
            if f in chosen:
                continue
            m = rep(oof(feat30 + chosen + [f]))
            if m["R2_all"] < floor:
                continue
            if abs(m["MSPE_grid"]) < abs(cur["MSPE_grid"]) - 1e-3 and \
               (best is None or abs(m["MSPE_grid"]) < abs(best["MSPE_grid"])):
                best, best_f = m, f
        if best_f is None:
            print(f"[gridfix] no grid gain within the R2 floor at step {len(chosen)+1}; stopping")
            break
        chosen.append(best_f)
        cur = best
        rows.append({"step": len(chosen), "feature": best_f, "group": group_of[best_f], **best})
        print(f"  +{best_f:38s} R2={cur['R2_all']:+.4f}  MSPE_LN={cur['MSPE_LineNoise']:+.2f}% "
              f"MSPE_grid={cur['MSPE_grid']:+.2f}%  MAPE_LN={cur['MAPE_LineNoise']:.2f}% "
              f"MAPE_grid={cur['MAPE_grid']:.2f}%  bias2={cur['bias2']:.2f}")

    print(f"\nfinal ({len(chosen)}): {chosen}")

    # leave-one-out on the final set
    full = cur
    print("\n=== leave-one-out ablation ===")
    for f in chosen:
        m = rep(oof(feat30 + [x for x in chosen if x != f]))
        rows.append({"step": -1, "feature": f"drop {f}", "group": group_of[f], **m,
                     "dR2_all": m["R2_all"] - full["R2_all"],
                     "dbias_LN_pp": abs(m["MSPE_LineNoise"]) - abs(full["MSPE_LineNoise"]),
                     "dbias_grid_pp": abs(m["MSPE_grid"]) - abs(full["MSPE_grid"])})
        print(f"  drop {f:38s} dR2={rows[-1]['dR2_all']:+.4f}  "
              f"LN bias {rows[-1]['dbias_LN_pp']:+.2f}pp  grid bias {rows[-1]['dbias_grid_pp']:+.2f}pp")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n" + df.round(4).to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
