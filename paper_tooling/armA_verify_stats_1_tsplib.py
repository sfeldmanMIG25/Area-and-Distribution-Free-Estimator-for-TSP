"""Independent re-derivation of the two TSPLIB significance claims for arm A.

Nothing here calls compare_dispersion / compare / evaluate_candidate. Baselines
are read straight off the TSPLIB results CSV with the same structural screen the
harness uses, then every statistic is recomputed from first principles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tsplib_benchmark.exclusions import filter_metric_consistent  # noqa: E402

TSPLIB_CSV = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
SHIPPED = "LGBM_V3"


def load_euc2d():
    raw = pd.read_csv(TSPLIB_CSV)
    anchor = raw[raw.model == SHIPPED].drop_duplicates("instance").set_index("instance")
    euc = anchor.index[anchor["edge_weight_type"] == "EUC_2D"]
    anchor_sub = anchor.loc[sorted(euc)]
    screened = filter_metric_consistent(anchor_sub.reset_index()).set_index("instance")
    idx = pd.Index(sorted(screened.index), name="instance")
    sub = raw[raw.instance.isin(idx)]
    wide = sub.pivot_table(index="instance", columns="model", values="pred_cost",
                           aggfunc="first").reindex(idx)
    truth = screened.loc[idx, "true_cost"].astype(float)
    n_cust = screened.loc[idx, "n"].astype(float)
    return truth, wide, n_cust


def spe(pred, truth):
    return ((pred - truth) / truth * 100.0).astype(float)


def ape(pred, truth):
    return ((pred - truth).abs() / truth * 100.0).astype(float)


def pitman_morgan(x, y):
    """Return (r_uv, t, p_two_sided, df)."""
    n = x.size
    u, v = x + y, x - y
    r = float(np.corrcoef(u, v)[0, 1])
    t = r * np.sqrt((n - 2) / (1 - r * r))
    p = float(2 * stats.t.sf(abs(t), n - 2))
    return r, float(t), p, n - 2


def pm_ci_ratio(x, y, conf=0.95):
    """Exact PM confidence interval for sd(x)/sd(y) via the Fisher z on r_uv.

    var ratio k^2 solves rho_uv = (k^2-1)/sqrt((k^2+2rk+1)(k^2-2rk+1)) with
    r = corr(x,y). Invert numerically at each end of the r_uv CI.
    """
    from scipy import optimize
    n = x.size
    u, v = x + y, x - y
    r_uv = float(np.corrcoef(u, v)[0, 1])
    r_xy = float(np.corrcoef(x, y)[0, 1])
    z = np.arctanh(r_uv)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(0.5 + conf / 2)
    lo_r, hi_r = np.tanh(z - crit * se), np.tanh(z + crit * se)

    def rho_of_k(k):
        den2 = (k * k + 2 * r_xy * k + 1) * (k * k - 2 * r_xy * k + 1)
        return (k * k - 1.0) / np.sqrt(den2)

    def solve(target):
        f = lambda k: rho_of_k(k) - target  # noqa: E731
        return float(optimize.brentq(f, 1e-6, 1e6, xtol=1e-12))

    return solve(lo_r), solve(hi_r), r_uv, r_xy


def levene_bf(x, y):
    return float(stats.levene(x, y, center="median").pvalue)


def swap_perm_p(x, y, n_perm=200_000, seed=42):
    rng = np.random.default_rng(seed)
    xc, yc = x - x.mean(), y - y.mean()
    obs = abs(np.log(x.std(ddof=1) / y.std(ddof=1)))
    swap = rng.random((n_perm, x.size)) < 0.5
    px = np.where(swap, yc, xc)
    py = np.where(swap, xc, yc)
    perm = np.abs(np.log(px.std(axis=1, ddof=1) / py.std(axis=1, ddof=1)))
    return float((1 + np.sum(perm >= obs - 1e-15)) / (n_perm + 1))


def holm(p):
    p = np.asarray(p, dtype=float)
    m = p.size
    out = np.empty(m)
    order = np.argsort(p, kind="mergesort")
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def main():
    truth, wide, n_cust = load_euc2d()
    PI = pd.read_csv(HERE / "support_arms_per_instance.csv")
    pi = PI[PI.stratum == "tsplib_euc2d"]
    preds = {tag: pi[pi.model == tag].set_index("instance")["pred_cost"].reindex(truth.index)
             for tag in ("FROZEN", "R0", "A", "B")}

    out = {}
    print(f"n TSPLIB EUC_2D after screen: {len(truth)}")
    # sanity: FROZEN in the arm study should equal the shipped LGBM_V3 column
    d = (preds["FROZEN"] - wide[SHIPPED]).abs()
    print(f"max |FROZEN(arm study) - LGBM_V3(prod csv)| = {d.max():.6e}")
    print(f"max |R0 - FROZEN| = {(preds['R0'] - preds['FROZEN']).abs().max():.6e}")

    # ================= CLAIM 1: dispersion vs Asymptotic_MST =============
    print("\n=== dispersion vs Asymptotic_MST (signed pct error) ===")
    base = wide["Asymptotic_MST"]
    y = spe(base, truth).to_numpy()
    rows = []
    for tag in ("FROZEN", "A", "B"):
        x = spe(preds[tag], truth).to_numpy()
        sa, sb = x.std(ddof=1), y.std(ddof=1)
        r_uv, t, p, df = pitman_morgan(x, y)
        lo, hi, _, r_xy = pm_ci_ratio(x, y)
        rows.append({
            "model": tag, "n": x.size, "sdpe_a": sa, "sdpe_b": sb,
            "sd_ratio": sa / sb, "ci_lo": lo, "ci_hi": hi,
            "r_xy": r_xy, "r_uv": r_uv, "t": t, "df": df,
            "p_pm_raw": p, "p_bf": levene_bf(x, y), "p_swap": swap_perm_p(x, y),
        })
    D = pd.DataFrame(rows)
    print(D.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    out["dispersion_vs_asymptotic"] = D.to_dict("records")

    # ================= CLAIM 2: mean gain vs Calibrated_MST_dn ===========
    print("\n=== mean APE gain vs Calibrated_MST_dn ===")
    baseb = wide["Calibrated_MST_dn"]
    b = ape(baseb, truth)
    rows = []
    for tag in ("FROZEN", "A", "B"):
        a = ape(preds[tag], truth)
        ok = a.notna() & b.notna()
        diff = (a[ok] - b[ok]).to_numpy()
        n = diff.size
        w = stats.wilcoxon(diff, zero_method="wilcox")
        tt = stats.ttest_1samp(diff, 0.0)
        sd = diff.std(ddof=1)
        boot = np.random.default_rng(42).integers(0, n, size=(20000, n))
        bm = diff[boot].mean(axis=1)
        rows.append({
            "model": tag, "n": n, "mape_a": a[ok].mean(), "mape_b": b[ok].mean(),
            "mean_gain": -diff.mean(), "median_gain": -np.median(diff),
            "sd_diff": sd, "dz": diff.mean() / sd,
            "gain_ci_lo": -np.percentile(bm, 97.5), "gain_ci_hi": -np.percentile(bm, 2.5),
            "wins_a": int((diff < 0).sum()), "wins_b": int((diff > 0).sum()),
            "p_wilcoxon_raw": float(w.pvalue), "p_ttest_raw": float(tt.pvalue),
        })
    M = pd.DataFrame(rows)
    print(M.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    out["mean_vs_calibrated_dn"] = M.to_dict("records")

    # ================= THE GAP: FROZEN vs A raw SDPE =====================
    print("\n=== HEAD-TO-HEAD: arm A vs FROZEN, tsplib_euc2d dispersion ===")
    xa = spe(preds["A"], truth).to_numpy()
    xf = spe(preds["FROZEN"], truth).to_numpy()
    sa, sf = xa.std(ddof=1), xf.std(ddof=1)
    r_uv, t, p, df = pitman_morgan(xa, xf)
    lo, hi, _, r_xy = pm_ci_ratio(xa, xf)
    p_swap = swap_perm_p(xa, xf)
    p_bf = levene_bf(xa, xf)
    print(f"SDPE  A={sa:.4f}  FROZEN={sf:.4f}  ratio={sa/sf:.4f} (+{(sa/sf-1)*100:.2f}%)")
    print(f"PM: r_xy={r_xy:.4f} r_uv={r_uv:.4f} t={t:.4f} df={df} p={p:.4g}")
    print(f"PM 95% CI on SD ratio: [{lo:.4f}, {hi:.4f}]")
    print(f"Brown-Forsythe p={p_bf:.4g}   swap-permutation p={p_swap:.4g}")
    # also variance-of-APE and MAD, and the paired abs-error scale
    va = np.var(xa, ddof=1)
    vf = np.var(xf, ddof=1)
    print(f"variance ratio = {va/vf:.4f}")
    out["A_vs_FROZEN_dispersion"] = {
        "n": int(xa.size), "sdpe_A": sa, "sdpe_FROZEN": sf, "sd_ratio": sa / sf,
        "pct_change": (sa / sf - 1) * 100, "r_xy": r_xy, "r_uv": r_uv,
        "t": t, "df": int(df), "p_pitman_morgan": p, "ci_lo": lo, "ci_hi": hi,
        "p_brown_forsythe": p_bf, "p_swap_permutation": p_swap,
    }

    # who drives it -- squared-error decomposition of the variance change
    print("\n--- decomposition of the SDPE increase ---")
    ca, cf = xa - xa.mean(), xf - xf.mean()
    contrib = (ca**2 - cf**2) / (xa.size - 1)
    dfc = pd.DataFrame({
        "instance": truth.index, "n": n_cust.to_numpy(),
        "spe_frozen": xf, "spe_A": xa,
        "d_var_contrib": contrib,
    }).sort_values("d_var_contrib", ascending=False)
    tot = contrib.sum()
    print(f"total variance change = {tot:+.5f}  (var A {va:.4f} - var F {vf:.4f} = {va-vf:+.4f})")
    print(f"instances worsening variance: {(contrib > 0).sum()}/{contrib.size}")
    for k in (1, 2, 3, 5, 10):
        print(f"  top-{k:2d} share of total increase: "
              f"{dfc.d_var_contrib.head(k).sum() / tot * 100:6.1f}%")
    print("\ntop 10 contributors:")
    print(dfc.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nbottom 5 (variance-reducing):")
    print(dfc.tail(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    dfc.to_csv(HERE / "armA_verify_tsplib_variance_decomp.csv", index=False)

    # leave-one-out on the SD ratio
    loo = []
    for i in range(xa.size):
        m = np.ones(xa.size, dtype=bool)
        m[i] = False
        loo.append(xa[m].std(ddof=1) / xf[m].std(ddof=1))
    loo = np.array(loo)
    j = int(np.argmin(loo))
    print(f"\nleave-one-out SD ratio: min {loo.min():.4f} (drop "
          f"{truth.index[j]}), max {loo.max():.4f}, full {sa/sf:.4f}")
    # how many must be dropped to get ratio <= 1
    order = np.argsort(-contrib)
    for k in range(1, 12):
        m = np.ones(xa.size, dtype=bool)
        m[order[:k]] = False
        r = xa[m].std(ddof=1) / xf[m].std(ddof=1)
        if r <= 1.0:
            print(f"dropping the top {k} contributor(s) brings the ratio to {r:.4f} <= 1")
            break
    else:
        print("dropping up to 11 top contributors never brings the ratio to <= 1")

    json.dump(out, open(HERE / "armA_verify_tsplib_stats.json", "w"), indent=2, default=float)
    print("\nwrote armA_verify_tsplib_stats.json, armA_verify_tsplib_variance_decomp.csv")


if __name__ == "__main__":
    main()
