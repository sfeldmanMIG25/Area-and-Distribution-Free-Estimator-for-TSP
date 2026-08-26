"""Pre-empt the attack that was run on arm A, on arm C's own numbers.

Arm A's 9/9 fell because its reported statistics were the favourable extreme of
a nuisance distribution nobody had drawn. These are the same probes, run on arm
C before anyone else runs them:

* where the protocol-order value sits inside its own row-order permutation band;
* how fragile each gate verdict is across all 15 arm C fits (7 seeds + 8 perms);
* what the 24 removed rows actually bought, paired by seed (arm A minus arm C);
* what the 850 retained rows bought, paired by seed (arm C minus order-only R0);
* whether gate 2's verdict survives every reasonable reading of "the median".

Writes armC_attack.json / armC_attack.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import armC_common as CC  # noqa: E402


def main() -> None:
    R = pd.read_csv(HERE / "armC_runs.csv").set_index("run")
    S = pd.read_csv(HERE / "armC_significance.csv").set_index("run")
    C = [f"C_s{s}" for s in CC.SEEDS]
    A = [f"A_s{s}" for s in CC.SEEDS]
    Z = [f"R0p_s{s}" for s in CC.SEEDS]
    PM = [r for r in R.index if r.startswith("Cperm")]
    out: dict = {}

    # ---- 1. row-order position of the protocol-order value --------------
    med_seed = 123
    obs = float(R.loc[f"C_s{med_seed}", "linenoise_slope"])
    perm = np.sort([float(R.loc[r, "linenoise_slope"]) for r in PM])
    out["row_order_position"] = {
        "median_seed": med_seed,
        "protocol_order_slope": obs,
        "permutation_band": [float(perm.min()), float(perm.max())],
        "permutation_median": float(np.median(perm)),
        "permutation_sd": float(np.std(perm, ddof=1)),
        "pct_of_permutations_at_or_below_observed":
            float((perm <= obs).mean() * 100),
        "observed_outside_band": bool(obs < perm.min() or obs > perm.max()),
        "armA_comparison": ("arm A reported 0.8268 while its own 8-permutation "
                            "band was [0.6367, 0.8028] -- above the entire band. "
                            "Arm C's protocol-order value sits inside its band."),
    }

    # ---- 2. gate fragility over all 15 arm C fits -----------------------
    allC = C + PM
    sl = np.array([float(R.loc[r, "linenoise_slope"]) for r in allC])
    out["gate6_fragility"] = {
        "n_fits": len(allC),
        "slope_median": float(np.median(sl)),
        "slope_band": [float(sl.min()), float(sl.max())],
        "fits_below_0.70": int((sl < CC.G6_SLOPE_MIN).sum()),
        "fits_below_0.70_pct": float((sl < CC.G6_SLOPE_MIN).mean() * 100),
        "closest_pass": float(sl[sl >= CC.G6_SLOPE_MIN].min()),
        "worst_seed_margin": float(min(
            float(R.loc[r, "linenoise_slope"]) for r in C) - CC.G6_SLOPE_MIN),
    }

    # ---- 3/4. paired deltas ---------------------------------------------
    rows = []
    keys = ["linenoise_slope", "b2_grid_mspe", "bench2d_mape", "nd_test_mape",
            "tsplib_euc2d_sdpe", "tsplib_noneuc_mape", "b2_line_noise_mape"]
    for k in keys:
        dAC = np.array([float(R.loc[a, k]) - float(R.loc[c, k])
                        for a, c in zip(A, C)])
        dCZ = np.array([float(R.loc[c, k]) - float(R.loc[z, k])
                        for c, z in zip(C, Z)])
        rows.append({
            "metric": k,
            "A_minus_C_median": float(np.median(dAC)),
            "A_minus_C_lo": float(dAC.min()), "A_minus_C_hi": float(dAC.max()),
            "A_minus_C_sign_consistent": bool(np.all(dAC > 0) or np.all(dAC < 0)),
            "C_minus_R0protocol_median": float(np.median(dCZ)),
            "C_minus_R0protocol_lo": float(dCZ.min()),
            "C_minus_R0protocol_hi": float(dCZ.max()),
            "C_minus_R0protocol_sign_consistent":
                bool(np.all(dCZ > 0) or np.all(dCZ < 0)),
        })
    P = pd.DataFrame(rows)
    P.to_csv(HERE / "armC_attack.csv", index=False)
    out["paired_deltas"] = P.to_dict(orient="records")

    # ---- 5. gate 2 under every reasonable reading -----------------------
    sw = np.array([float(S.loc[r, "disp_p_swap_holm"]) for r in C])
    det = np.array([bool(S.loc[r, "disp_detectable_holm"]) for r in C])
    out["gate2_robustness"] = {
        "swap_holm_per_seed": {int(s): round(float(v), 5)
                               for s, v in zip(CC.SEEDS, sw)},
        "reading_median_of_p": {"value": float(np.median(sw)),
                                "verdict": "FAIL" if np.median(sw) >= 0.05 else "PASS"},
        "reading_median_seed_model": {
            "value": float(S.loc[f"C_s{med_seed}", "disp_p_swap_holm"]),
            "verdict": "FAIL"},
        "reading_majority_of_seeds": {"n_pass": int((sw < 0.05).sum()),
                                      "verdict": "FAIL"},
        "reading_all_seeds": {"verdict": "FAIL"},
        "reading_best_seed": {"value": float(sw.min()),
                              "verdict": "PASS (single seed -- the exact defect "
                                         "this protocol exists to forbid)"},
        "detectable_holm_seeds": f"{int(det.sum())}/7",
        "pitman_morgan_holm_median":
            float(np.median([float(S.loc[r, "disp_p_pm_holm"]) for r in C])),
        "swap_raw_p_median":
            float(np.median([float(S.loc[r, "disp_p_swap_raw"]) for r in C])),
        "r_xy_frozen": float(S.loc["FROZEN", "disp_r_xy"]),
        "r_xy_candidate_median":
            float(np.median([float(S.loc[r, "disp_r_xy"]) for r in C])),
        "mechanism": ("The raw swap-permutation p is significant (median 0.0093); "
                      "it is the Holm correction over the 18-test dispersion "
                      "family that removes it. The candidate's error vector is "
                      "also far less correlated with the baseline's than the "
                      "frozen model's is (r_xy 0.22 vs 0.46), which is what the "
                      "distribution-free test is sensitive to and the "
                      "parametric Pitman-Morgan test is not."),
        "armA_same_protocol_seeds_passing":
            int(sum(float(S.loc[r, "disp_p_swap_holm"]) < 0.05 for r in A)),
    }

    json.dump(out, open(HERE / "armC_attack.json", "w"), indent=2)

    print("=== row-order position of the protocol-order value ===")
    for k, v in out["row_order_position"].items():
        print(f"  {k}: {v}")
    print("\n=== gate 6 fragility over all 15 arm C fits ===")
    for k, v in out["gate6_fragility"].items():
        print(f"  {k}: {v}")
    print("\n=== paired deltas (same seed) ===")
    print(P.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== gate 2 under every reading ===")
    for k, v in out["gate2_robustness"].items():
        print(f"  {k}: {v}")
    print("\nwrote armC_attack.json armC_attack.csv")


if __name__ == "__main__":
    main()
