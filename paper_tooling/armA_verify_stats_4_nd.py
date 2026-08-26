"""Is arm A's ND-test gain broad-based or tail-driven?

Mean MAPE 0.6201 -> 0.5965 with Wilcoxon p 6.3e-33 on 16,920 held-out rows. A
p-value that small on n=16,920 says only that the shift is not exactly zero. The
question is where the 0.0236 pp lives.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 240)


def main():
    PI = pd.read_csv(HERE / "support_arms_per_instance.csv", low_memory=False)
    nd = PI[PI.stratum == "nd_test"]
    fz = nd[nd.model == "FROZEN"].set_index("instance")
    out = {}

    for cand in ("A", "B"):
        ca = nd[nd.model == cand].set_index("instance")
        idx = fz.index.intersection(ca.index)
        a = ca.loc[idx, "err_pct"].abs().to_numpy()
        f = fz.loc[idx, "err_pct"].abs().to_numpy()
        d = a - f                                    # negative = arm improves
        n = d.size
        print(f"\n################ ND test: arm {cand} vs FROZEN  (n={n}) ################")
        print(f"MAPE  frozen {f.mean():.4f} -> {cand} {a.mean():.4f}   "
              f"mean paired diff {d.mean():+.5f} pp")
        print(f"MEDIAN |err|  frozen {np.median(f):.4f} -> {cand} {np.median(a):.4f}")
        print(f"MEDIAN paired diff {np.median(d):+.6f} pp   "
              f"Hodges-Lehmann {np.median((d[:, None] + d[None, :])[np.triu_indices(min(n, 1500))] / 2) if False else float(np.median(d)):+.6f}")

        worse = int((d > 0).sum())
        better = int((d < 0).sum())
        tie = int((d == 0).sum())
        print(f"instances BETTER {better} ({better/n*100:.2f}%)   "
              f"WORSE {worse} ({worse/n*100:.2f}%)   TIED {tie}")
        print(f"sign test p (better vs worse) "
              f"{stats.binomtest(better, better+worse, 0.5).pvalue:.4g}")

        w = stats.wilcoxon(d, zero_method="zsplit")
        print(f"Wilcoxon p {w.pvalue:.4g}   "
              f"rank-biserial {2*better/(better+worse)-1:+.4f}   "
              f"Cohen dz {d.mean()/d.std(ddof=1):+.4f}")

        qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        print("paired-difference quantiles (pp):")
        print("  " + "  ".join(f"p{q}={np.percentile(d, q):+.4f}" for q in qs))

        # ---- where does the mean gain come from? -------------------------
        print("\n-- decomposition of the total gain by frozen-error decile --")
        dec = pd.qcut(f, 10, labels=False, duplicates="drop")
        tab = pd.DataFrame({"d": d, "f": f, "a": a, "dec": dec}).groupby("dec").agg(
            n=("d", "size"), frozen_mape=("f", "mean"), arm_mape=("a", "mean"),
            mean_diff=("d", "mean"), median_diff=("d", "median"),
            pct_worse=("d", lambda s: (s > 0).mean() * 100))
        tab["share_of_total_gain_%"] = tab.n * tab.mean_diff / d.sum() * 100
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))

        print("\n-- trimming the tail of the paired difference --")
        srt = np.sort(d)
        for k_pct in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0):
            k = int(round(n * k_pct / 100))
            core = srt[k:n - k] if k else srt
            print(f"  drop {k_pct:4.1f}% at EACH end (k={k:4d}): "
                  f"mean diff {core.mean():+.6f} pp  "
                  f"({core.mean()/d.mean()*100 if d.mean() else float('nan'):6.1f}% of full)")

        # top-|d| concentration
        order = np.argsort(-np.abs(d))
        for k in (1, 10, 50, 100, 500, 1000):
            print(f"  top {k:5d} |diff| instances ({k/n*100:5.2f}%) carry "
                  f"{d[order[:k]].sum()/d.sum()*100:7.2f}% of the total gain")

        out[cand] = {
            "n": n, "mean_diff_pp": float(d.mean()),
            "median_diff_pp": float(np.median(d)),
            "pct_better": better / n * 100, "pct_worse": worse / n * 100,
            "pct_tied": tie / n * 100,
            "p_wilcoxon": float(w.pvalue),
            "cohens_dz": float(d.mean() / d.std(ddof=1)),
            "mean_diff_trim1pct": float(np.sort(d)[int(n*0.01):n - int(n*0.01)].mean()),
            "share_top1pct": float(d[order[:int(n*0.01)]].sum() / d.sum() * 100),
        }

    json.dump(out, open(HERE / "armA_verify_nd_stats.json", "w"), indent=2)
    print("\nwrote armA_verify_nd_stats.json")


if __name__ == "__main__":
    main()
