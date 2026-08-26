"""Do the Pitman-Morgan assumptions hold, and do the robust cross-checks agree?

PM is a t-test on corr(x+y, x-y) and is exact only under bivariate normality of
the paired error vectors. It is known to be badly non-robust to kurtosis. The
harness already computes two distribution-free alternatives; this asks whether
they agree, and what the gate verdict would be on each.
"""
from __future__ import annotations

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

import ood_harness as oh  # noqa: E402

pd.set_option("display.width", 260)


def arm_preds(tag):
    PI = pd.read_csv(HERE / "support_arms_per_instance.csv", low_memory=False)
    p = PI[PI.model == tag]
    return dict(zip(p["instance"].astype(str), p["pred_cost"].astype(float)))


def main():
    s = oh.load_suite()["tsplib_euc2d"]
    truth = s.truth
    base = s.baselines["Asymptotic_MST"]
    y = oh.signed_percent_error(base, truth).to_numpy()

    print("=== normality of the paired signed-percent-error vectors (n=78) ===")
    rows = []
    for tag in ("FROZEN", "A"):
        x = oh.signed_percent_error(
            pd.Series(arm_preds(tag)).reindex(truth.index), truth).to_numpy()
        u, v = x + y, x - y
        for nm, vec in (("x=" + tag, x), ("y=Asymptotic", y), ("u=x+y", u), ("v=x-y", v)):
            rows.append({
                "arm": tag, "vector": nm,
                "skew": float(stats.skew(vec)),
                "excess_kurt": float(stats.kurtosis(vec)),
                "shapiro_p": float(stats.shapiro(vec).pvalue),
                "jarque_bera_p": float(stats.jarque_bera(vec).pvalue),
            })
    N = pd.DataFrame(rows).drop_duplicates(subset=["vector"], keep="first")
    print(N.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nPM requires bivariate normality of (x,y); it is a t-test on r(u,v)")
    print("and its type-I error is inflated under excess kurtosis.")

    print("\n=== gate-2 verdict under each dispersion test, Holm-adjusted "
          "inside the m=18 family the study used ===")
    for tag in ("FROZEN", "A", "B"):
        d = oh.dispersion_verdict({tag: arm_preds(tag)}, stratum="tsplib_euc2d")
        r = d[d.model_b == "Asymptotic_MST"].iloc[0]
        print(f"\n--- {tag}  (sd_ratio {r.sd_ratio:.4f}) ---")
        for label, praw, pholm in (
            ("Pitman-Morgan  (primary, parametric)", r.p_pitman_morgan, r.p_holm),
            ("Brown-Forsythe (robust, unpaired)   ", r.p_brown_forsythe, r.p_bf_holm),
            ("swap permutation (distribution-free)", r.p_swap_permutation, r.p_swap_holm),
        ):
            verdict = "PASS" if pholm < 0.05 else "FAIL"
            print(f"  {label}: p_raw {praw:<10.4g} p_holm {pholm:<10.4g} -> gate2 {verdict}")

    print("\n=== bootstrap CI on the SD ratio (harness values) ===")
    for tag in ("FROZEN", "A", "B"):
        d = oh.dispersion_verdict({tag: arm_preds(tag)}, stratum="tsplib_euc2d")
        r = d[d.model_b == "Asymptotic_MST"].iloc[0]
        print(f"  {tag:7s} ratio {r.sd_ratio:.4f}  CI [{r.ratio_ci_low:.4f}, "
              f"{r.ratio_ci_high:.4f}]  mde_holm {r.mde_sd_ratio_holm:.4f}  "
              f"detectable_holm {bool(r.detectable_holm)}")

    # ---- trimmed / robust scale, to see if one point is doing the work ----
    print("\n=== robust scale of the signed percent error, tsplib_euc2d ===")
    rows = []
    for tag in ("FROZEN", "A", "B"):
        x = oh.signed_percent_error(
            pd.Series(arm_preds(tag)).reindex(truth.index), truth).to_numpy()
        rows.append({
            "model": tag, "sd": x.std(ddof=1),
            "IQR": float(np.subtract(*np.percentile(x, [75, 25]))),
            "MAD_scaled": float(stats.median_abs_deviation(x, scale="normal")),
            "sd_trim10": float(stats.mstats.trimmed_std(x, limits=0.10, ddof=1)),
            "p95_abs": float(np.percentile(np.abs(x), 95)),
            "max_abs": float(np.abs(x).max()),
        })
    R = pd.DataFrame(rows)
    R["sd_ratio_vs_frozen"] = R["sd"] / R.loc[R.model == "FROZEN", "sd"].iloc[0]
    R["MAD_ratio_vs_frozen"] = R["MAD_scaled"] / R.loc[R.model == "FROZEN",
                                                       "MAD_scaled"].iloc[0]
    R["trim_ratio_vs_frozen"] = R["sd_trim10"] / R.loc[R.model == "FROZEN",
                                                       "sd_trim10"].iloc[0]
    print(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- provenance check: is the study's FROZEN the production LGBM_V3? --
    print("\n=== is the arm study's FROZEN the production LGBM_V3? ===")
    prod = s.baselines["LGBM_V3"]
    fz = pd.Series(arm_preds("FROZEN")).reindex(truth.index)
    rel = ((fz - prod).abs() / prod * 100.0)
    print(f"  n compared {rel.notna().sum()}")
    print(f"  max relative diff {rel.max():.4f}%   median {rel.median():.6f}%   "
          f"n_exact {(rel < 1e-9).sum()}/{len(rel)}")
    print(f"  MAPE study-FROZEN {oh.absolute_percent_error(fz, truth).mean():.4f}  "
          f"vs production LGBM_V3 {oh.absolute_percent_error(prod, truth).mean():.4f}")
    print(f"  SDPE study-FROZEN {oh.signed_percent_error(fz, truth).std(ddof=1):.4f}  "
          f"vs production LGBM_V3 "
          f"{oh.signed_percent_error(prod, truth).std(ddof=1):.4f}")
    worst = rel.sort_values(ascending=False).head(5)
    print("  largest relative disagreements:")
    for k, v in worst.items():
        print(f"    {k:12s} {v:.4f}%")


if __name__ == "__main__":
    main()
