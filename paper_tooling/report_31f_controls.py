"""Compare the 31-feature controls against GART 2.0 on all four strata.

Reads the published benchmark CSVs for the incumbents and
``paper_tooling/controls_31f/rows_*.csv`` for the replacements, applies the
screens ``build_paper_tables`` applies, and reports MAPE, SDPE and the paired
tests.

TWO PAIRED TESTS, BECAUSE THE CLAIM AT ISSUE IS ABOUT DISPERSION
-----------------------------------------------------------------
``build_paper_tables.paired_test`` tests the mean paired difference in |APE|,
i.e. a MAPE comparison, and it is imported here rather than reimplemented so
the convention (sign, bootstrap seed, B, Wilcoxon zero-method) cannot drift.
It says nothing about SDPE, and the manuscript's neural-network claim is an
SDPE claim. SDPE is a property of a sample, not of an instance, so there is no
per-instance difference to rank; the paired analogue is a bootstrap over
instances that recomputes *both* models' SDPE on the same resampled instance
set, which preserves the pairing and the shared instance-difficulty variance.
:func:`paired_sdpe_boot` does that.

Sign convention matches the repository's: negative favours GART 2.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "paper_tooling")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tsplib_benchmark.exclusions import filter_metric_consistent  # noqa: E402
from paper_tooling.build_paper_tables import (  # noqa: E402
    BOOT_B, OK_STATUS, RNG_SEED, boot_sdpe_ci, paired_test,
)
from model_registry import GART  # noqa: E402

CTRL = ROOT / "paper_tooling" / "controls_31f"
P_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
P_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
P_TSPLIB = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"

INCUMBENTS = ["NN_V3", "Linear_V3"]
CONTROLS = ["NN_31F", "Linear_31F"]
ROSTER = [GART] + CONTROLS + INCUMBENTS


def _usable(df: pd.DataFrame) -> pd.DataFrame:
    keep = (df["status"].isin(OK_STATUS) & np.isfinite(df["pred_cost"])
            & np.isfinite(df["true_cost"]) & (df["true_cost"] > 0))
    out = df.loc[keep].copy()
    out["err_pct"] = 100.0 * (out["pred_cost"] - out["true_cost"]) / out["true_cost"]
    return out[np.isfinite(out["err_pct"])].copy()


def load_stratum(name: str) -> pd.DataFrame:
    """Published rows for the incumbents, fresh rows for the controls."""
    published = {"nd": P_ND, "2d": P_2D, "tsplib_euc": P_TSPLIB,
                 "tsplib_noneuc": P_TSPLIB}[name]
    fresh = {"nd": "rows_nd.csv", "2d": "rows_2d.csv",
             "tsplib_euc": "rows_tsplib.csv", "tsplib_noneuc": "rows_tsplib.csv"}[name]

    pub = pd.read_csv(published, low_memory=False)
    new = pd.read_csv(CTRL / fresh, low_memory=False)
    if name.startswith("tsplib"):
        cols = ["instance", "model", "pred_cost", "true_cost", "status",
                "edge_weight_type", "mst_length"]
        df = pd.concat([pub[cols], new[cols]], ignore_index=True)
        # mst_length is persisted on only some model rows but is an instance
        # property; broadcast it exactly as build_paper_tables.load_tsplib does,
        # because filter_metric_consistent's tour/MST ratio guard is a no-op
        # without it and the screened N would silently change.
        ml = (df.dropna(subset=["mst_length"]).drop_duplicates("instance")
                .set_index("instance")["mst_length"])
        df["mst_length"] = df["instance"].map(ml)
        df = _usable(df)
        df = (df[df.edge_weight_type == "EUC_2D"] if name == "tsplib_euc"
              else df[df.edge_weight_type != "EUC_2D"])
        return filter_metric_consistent(df)

    cols = ["instance", "model", "pred_cost", "true_cost", "status"]
    return _usable(pd.concat([pub[cols], new[cols]], ignore_index=True))


def paired_sdpe_boot(df: pd.DataFrame, a: str, b: str, B: int = BOOT_B,
                     seed: int = RNG_SEED) -> dict:
    """Bootstrap the SDPE difference (a - b) over the instances both models score."""
    wide = df.pivot_table(index="instance", columns="model", values="err_pct",
                          aggfunc="first")
    if a not in wide.columns or b not in wide.columns:
        return {"n_pairs": 0}
    pair = wide[[a, b]].dropna()
    ea, eb = pair[a].to_numpy(float), pair[b].to_numpy(float)
    n = len(ea)
    if n < 3:
        return {"n_pairs": n}
    point = float(np.std(ea, ddof=1) - np.std(eb, ddof=1))
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boots[i] = np.std(ea[idx], ddof=1) - np.std(eb[idx], ddof=1)
    frac_neg = float(np.mean(boots < 0.0))
    return {
        "n_pairs": n, "sdpe_diff": point,
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
        "boot_p": float(2.0 * min(frac_neg, 1.0 - frac_neg)),
    }


def main() -> None:
    strata = ["nd", "2d", "tsplib_euc", "tsplib_noneuc"]
    titles = {"nd": "Multidimensional test split", "2d": "2D benchmark",
              "tsplib_euc": "TSPLIB EUC_2D", "tsplib_noneuc": "TSPLIB non-Euclidean"}

    marg_rows, pair_rows = [], []
    for s in strata:
        df = load_stratum(s)
        print(f"\n=== {titles[s]} " + "=" * (58 - len(titles[s])))
        print(f"{'model':<12}{'N':>7}{'MAPE':>9}{'SDPE':>9}   SDPE 95% CI")
        for m in ROSTER:
            sub = df[df.model == m]
            if not len(sub):
                print(f"{m:<12}{'--':>7}   (absent)")
                continue
            e = sub["err_pct"].to_numpy(float)
            sd, lo, hi = boot_sdpe_ci(e)
            mape = float(np.mean(np.abs(e)))
            print(f"{m:<12}{len(sub):>7}{mape:>9.4f}{sd:>9.4f}   [{lo:.4f}, {hi:.4f}]")
            marg_rows.append({"stratum": s, "model": m, "N": len(sub),
                              "MAPE_pct": mape, "SDPE_pct": sd,
                              "SDPE_lo": lo, "SDPE_hi": hi})

        print(f"\n  paired vs {GART}   (negative favours {GART})")
        print(f"  {'model':<12}{'n':>6}{'d|APE|':>10}{'95% CI':>22}{'wilcoxon p':>13}"
              f"{'dSDPE':>10}{'95% CI':>22}{'boot p':>10}")
        for m in CONTROLS + INCUMBENTS:
            if not len(df[df.model == m]):
                continue
            mask = pd.Series(True, index=df.index)
            pt = paired_test(df, GART, m, mask)
            ps = paired_sdpe_boot(df, GART, m)
            nan = float("nan")
            ape_ci = f"[{pt['ci_lo']:.4f}, {pt['ci_hi']:.4f}]"
            sd_ci = f"[{ps.get('ci_lo', nan):.4f}, {ps.get('ci_hi', nan):.4f}]"
            print(f"  {m:<12}{pt['n_pairs']:>6}{pt['mean_diff']:>10.4f}{ape_ci:>22}"
                  f"{pt['wilcoxon_p']:>13.3e}{ps.get('sdpe_diff', nan):>10.4f}"
                  f"{sd_ci:>22}{ps.get('boot_p', nan):>10.4f}")
            pair_rows.append({"stratum": s, "model_a": GART, "model_b": m, **pt,
                              **{f"sdpe_{k}": v for k, v in ps.items()}})

    # --- neural seed band ---------------------------------------------------
    # The shipped seed is one draw. Reporting it alone on a question decided by
    # tenths of a percentage point is the failure mode that got the augmented
    # arms rejected, so the whole band is reported and the verdict is stated as
    # a count of seeds, not as a comparison of two point estimates.
    print("\n=== Neural control: spread over training seeds " + "=" * 15)
    band_rows = []
    for s in strata:
        df = load_stratum(s)
        seeds = sorted(m for m in df.model.unique() if m.startswith("NN_31F_s"))
        if not seeds:
            continue
        gart = df[df.model == GART]["err_pct"].to_numpy(float)
        g_sd, g_mape = float(np.std(gart, ddof=1)), float(np.mean(np.abs(gart)))
        sd = {m: float(np.std(df[df.model == m]["err_pct"], ddof=1)) for m in seeds}
        mp = {m: float(np.mean(np.abs(df[df.model == m]["err_pct"]))) for m in seeds}
        n_sd = sum(v < g_sd for v in sd.values())
        n_mp = sum(v < g_mape for v in mp.values())
        print(f"\n{titles[s]}   ({GART}: MAPE {g_mape:.4f}, SDPE {g_sd:.4f})")
        print(f"  SDPE  median {np.median(list(sd.values())):.4f}  "
              f"band [{min(sd.values()):.4f}, {max(sd.values()):.4f}]  "
              f"-> beats {GART} on {n_sd}/{len(seeds)} seeds")
        print(f"  MAPE  median {np.median(list(mp.values())):.4f}  "
              f"band [{min(mp.values()):.4f}, {max(mp.values()):.4f}]  "
              f"-> beats {GART} on {n_mp}/{len(seeds)} seeds")
        band_rows.append({"stratum": s, "n_seeds": len(seeds),
                          "gart_mape": g_mape, "gart_sdpe": g_sd,
                          "sdpe_median": float(np.median(list(sd.values()))),
                          "sdpe_min": min(sd.values()), "sdpe_max": max(sd.values()),
                          "sdpe_seeds_beating_gart": n_sd,
                          "mape_median": float(np.median(list(mp.values()))),
                          "mape_min": min(mp.values()), "mape_max": max(mp.values()),
                          "mape_seeds_beating_gart": n_mp,
                          **{f"sdpe_{k.split('_s')[-1]}": v for k, v in sd.items()}})
    pd.DataFrame(band_rows).to_csv(CTRL / "seed_band.csv", index=False)

    pd.DataFrame(marg_rows).to_csv(CTRL / "marginals.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(CTRL / "paired.csv", index=False)
    print(f"\nwrote {CTRL / 'marginals.csv'} and {CTRL / 'paired.csv'}")


if __name__ == "__main__":
    main()
