"""Power accounting for both TSPLIB claims, at the level each is actually made.

Gate 3 pre-registers an MDE at alpha = 0.05 but makes its claim at the
Holm-adjusted level. Gate 2's harness computes both. This recomputes the mean
claim's detection floor at the Holm threshold that actually binds, and reports
the Type-M (exaggeration) ratio for each claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ood_harness as oh  # noqa: E402

pd.set_option("display.width", 240)


def arm_preds(tag):
    PI = pd.read_csv(HERE / "support_arms_per_instance.csv", low_memory=False)
    p = PI[PI.model == tag]
    return dict(zip(p["instance"].astype(str), p["pred_cost"].astype(float)))


def type_m(delta, sd, n, alpha):
    """E[|estimate| | significant] / |true effect|, if the true effect were delta."""
    from scipy import stats
    se = sd / np.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, n - 1) * se
    xs = np.linspace(delta - 8 * se, delta + 8 * se, 40001)
    w = stats.norm.pdf(xs, delta, se)
    sig = np.abs(xs) >= crit
    if not sig.any() or w[sig].sum() == 0:
        return np.nan
    return float(np.average(np.abs(xs[sig]), weights=w[sig]) / abs(delta))


def main():
    s = oh.load_suite()["tsplib_euc2d"]
    truth = s.truth
    baseb = s.baselines["Calibrated_MST_dn"]

    print("=== GATE 3: mean APE gain vs Calibrated_MST_dn, detection floor ===")
    rows = []
    for tag in ("FROZEN", "A", "B"):
        cand = pd.Series(arm_preds(tag)).reindex(truth.index)
        a = oh.absolute_percent_error(cand, truth)
        b = oh.absolute_percent_error(baseb, truth)
        ok = a.notna() & b.notna()
        diff = (a[ok] - b[ok]).to_numpy()
        n, sd, gain = diff.size, diff.std(ddof=1), -diff.mean()

        v = oh.evaluate_candidate(arm_preds(tag), tag)
        c = v.comparisons
        row = c[(c.stratum == "tsplib_euc2d") & (c.model_b == "Calibrated_MST_dn")].iloc[0]
        p_all = c["p_wilcoxon"].to_numpy(dtype=float)
        mask = ~((c.stratum == "tsplib_euc2d") & (c.model_b == "Calibrated_MST_dn")).to_numpy()
        a_holm = oh.holm_threshold(p_all[mask])

        mde05 = oh.min_detectable_difference(sd, n, 0.05)
        mde_holm = oh.min_detectable_difference(sd, n, a_holm) if a_holm > 0 else np.nan
        mde_bonf = oh.min_detectable_difference(sd, n, 0.05 / v.family_size)
        rows.append({
            "model": tag, "n": n, "gain": gain, "sd_diff": sd,
            "p_raw": row.p_wilcoxon, "p_holm": row.p_holm,
            "holm_alpha_emp": a_holm,
            "MDE@0.05": mde05, "gain/MDE05": gain / mde05,
            "MDE@holm_emp": mde_holm, "gain/MDEholm": gain / mde_holm,
            f"MDE@0.05/{v.family_size}": mde_bonf, "gain/MDEbonf": gain / mde_bonf,
            "typeM@0.05": type_m(gain, sd, n, 0.05),
            "typeM@holm": type_m(gain, sd, n, a_holm) if a_holm > 0 else np.nan,
        })
    M = pd.DataFrame(rows)
    print(M.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  gate 3 pre-registers MDE at alpha=0.05 but claims significance at")
    print("  the Holm level. gain/MDEholm < 1 means the claim was made inside the")
    print("  regime the study's own power calculation calls undetectable.")

    print("\n=== GATE 2: dispersion vs Asymptotic_MST, detection floor ===")
    rows = []
    for tag in ("FROZEN", "A", "B"):
        d = oh.dispersion_verdict({tag: arm_preds(tag)}, stratum="tsplib_euc2d")
        r = d[d.model_b == "Asymptotic_MST"].iloc[0]
        rows.append({
            "model": tag, "sd_ratio": r.sd_ratio,
            "boot_CI": f"[{r.ratio_ci_low:.3f}, {r.ratio_ci_high:.3f}]",
            "r_xy": r.r_xy, "p_holm": r.p_holm,
            "MDE@0.05": r.mde_sd_ratio, "detectable": bool(r.detectable),
            "MDE@holm": r.mde_sd_ratio_holm, "detectable_holm": bool(r.detectable_holm),
            "margin_holm": r.mde_sd_ratio_holm - r.sd_ratio,
        })
    D = pd.DataFrame(rows)
    print(D.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  detectable_holm is the harness's own 80%-power flag. Negative")
    print("  margin_holm = the observed ratio is weaker than what n=78 can")
    print("  resolve at the level the claim is made.")

    print("\n=== what would n=78 need to resolve arm A's own SDPE regression? ===")
    xa = oh.signed_percent_error(pd.Series(arm_preds("A")).reindex(truth.index),
                                 truth).to_numpy()
    xf = oh.signed_percent_error(pd.Series(arm_preds("FROZEN")).reindex(truth.index),
                                 truth).to_numpy()
    r_xy = float(np.corrcoef(xa, xf)[0, 1])
    k_down = oh.variance_ratio_mde(78, r_xy, alpha=0.05)
    print(f"  r_xy(A, FROZEN) = {r_xy:.4f}")
    print(f"  smallest DETECTABLE improvement at n=78, alpha=.05: ratio <= {k_down:.4f}")
    print(f"  by symmetry the smallest detectable REGRESSION is roughly "
          f"ratio >= {1/k_down:.4f}  (i.e. +{(1/k_down-1)*100:.1f}% SDPE)")
    print(f"  observed regression: ratio {xa.std(ddof=1)/xf.std(ddof=1):.4f} "
          f"(+{(xa.std(ddof=1)/xf.std(ddof=1)-1)*100:.1f}%)")
    for n in (78, 150, 300, 600, 1200):
        k = oh.variance_ratio_mde(n, r_xy, alpha=0.05)
        print(f"    n={n:5d}: detectable regression threshold "
              f"+{(1/k-1)*100:5.1f}%")


if __name__ == "__main__":
    main()
