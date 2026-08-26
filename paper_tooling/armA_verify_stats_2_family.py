"""What family were the two TSPLIB claims actually corrected inside?

Reads the harness's own verdict tables rather than re-deriving them, because the
question here is about the *construction* of the family, not the arithmetic.
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


def main():
    suite = oh.load_suite()
    print("=== baselines defined per stratum (BASELINE_ORDER only) ===")
    for k, s in suite.items():
        names = [b for b in oh.BASELINE_ORDER if b in s.baselines]
        print(f"{k:16s} n={s.n_instances:6d}  n_baselines_in_order={len(names)}")

    for tag in ("FROZEN", "A"):
        preds = arm_preds(tag)
        print(f"\n############ {tag} ############")

        # ---- dispersion family as the study ran it: ONE stratum ----------
        d1 = oh.dispersion_verdict({tag: preds}, stratum="tsplib_euc2d")
        r1 = d1[d1.model_b == "Asymptotic_MST"].iloc[0]
        print(f"\n[dispersion, study family = tsplib_euc2d only]  m={len(d1)}")
        print(f"  ratio {r1.sd_ratio:.4f}  p_raw {r1.p_pitman_morgan:.4g}  "
              f"p_holm {r1.p_holm:.4g}  holm_alpha {r1.holm_alpha:.4g}")
        print(f"  mde_sd_ratio (unadj) {r1.mde_sd_ratio:.4f}   "
              f"mde_sd_ratio_holm {r1.mde_sd_ratio_holm:.4f}")
        print(f"  detectable (unadj) {bool(r1.detectable)}   "
              f"detectable_holm {bool(r1.detectable_holm)}   <-- harness's own power flag")
        print(f"  bootstrap ratio CI [{r1.ratio_ci_low:.4f}, {r1.ratio_ci_high:.4f}]")
        # rank of this test inside its family
        ps = np.sort(d1.p_pitman_morgan.dropna().to_numpy())
        rank = int(np.searchsorted(ps, r1.p_pitman_morgan) + 1)
        print(f"  this test is rank {rank}/{len(ps)} by raw p -> Holm multiplier "
              f"{len(ps) - rank + 1}")

        # ---- dispersion family if run the way the MEAN test is run -------
        allrows = []
        for key in suite:
            dd = oh.dispersion_verdict({tag: preds}, stratum=key)
            if len(dd):
                allrows.append(dd)
        big = pd.concat(allrows, ignore_index=True)
        big2 = oh.adjust_family(big.drop(columns=["p_holm", "p_bh", "family_size"]),
                                "p_pitman_morgan")
        rb = big2[(big2.stratum == "tsplib_euc2d") & (big2.model_b == "Asymptotic_MST")].iloc[0]
        print(f"\n[dispersion, all-strata family (parity with the mean test)]  m={len(big2)}")
        print(f"  p_raw {rb.p_pitman_morgan:.4g} -> p_holm {rb.p_holm:.4g}  "
              f"(sig at 0.05: {rb.p_holm < 0.05})")

        # ---- mean family -------------------------------------------------
        v = oh.evaluate_candidate(preds, tag)
        c = v.comparisons
        row = c[(c.stratum == "tsplib_euc2d") & (c.model_b == "Calibrated_MST_dn")].iloc[0]
        print(f"\n[mean, study family = all strata x all baselines]  m={v.family_size}")
        print(f"  gain {-row.mean_diff:.4f}  p_wilcoxon {row.p_wilcoxon:.4g}  "
              f"p_holm {row.p_holm:.4g}  p_perm_holm {row.p_perm_holm:.4g}")
        ps = np.sort(c.p_wilcoxon.dropna().to_numpy())
        rank = int(np.searchsorted(ps, row.p_wilcoxon) + 1)
        print(f"  rank {rank}/{len(ps)} by raw p -> Holm multiplier {len(ps) - rank + 1}")
        print(f"  per-stratum test counts: {c.groupby('stratum').size().to_dict()}")
        # what if the mean family were the ONE stratum, like the dispersion one?
        sub = c[c.stratum == "tsplib_euc2d"].copy()
        sub2 = oh.adjust_family(sub.drop(columns=["p_holm", "p_bh", "family_size"]),
                                "p_wilcoxon")
        rs = sub2[sub2.model_b == "Calibrated_MST_dn"].iloc[0]
        print(f"  [counterfactual: same-stratum-only family m={len(sub2)}] "
              f"p_holm {rs.p_holm:.4g}")
        # combined family: both claims together, all strata
        comb = np.concatenate([c.p_wilcoxon.dropna().to_numpy(),
                               big2.p_pitman_morgan.dropna().to_numpy()])
        adj = oh.holm_bonferroni(comb)
        i_mean = int(np.where(c.p_wilcoxon.dropna().to_numpy() == row.p_wilcoxon)[0][0])
        n_mean = int(c.p_wilcoxon.notna().sum())
        j_disp = n_mean + int(np.where(
            big2.p_pitman_morgan.dropna().to_numpy() == rb.p_pitman_morgan)[0][0])
        print(f"\n[both claims in ONE family, all strata]  m={comb.size}")
        print(f"  mean claim p_holm {adj[i_mean]:.4g}   dispersion claim p_holm "
              f"{adj[j_disp]:.4g}")

        # ---- losses that survive Holm -----------------------------------
        L = v.losses()
        print(f"\n  significant LOSSES under Holm: {len(L)}")
        if len(L):
            print(L[["stratum", "model_b", "n_pairs", "mape_a", "mape_b",
                     "mean_diff", "p_holm"]].to_string(index=False,
                                                       float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
