"""Score arm C against gates 1-11 and produce the s4 reporting obligations.

Every band-form statistic is reported as the median over the 7 named seeds with
the full min-max band and the per-seed values. Nothing here reads a single seed
as a result.

Writes armC_gate_table.csv, armC_obligations.json, armC_perseed.csv,
armC_dispersion_vs_frozen.csv, armC_nd_decomposition.csv.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import armA_verify_common as K  # noqa: E402
import armC_common as CC  # noqa: E402
import ood_harness as oh  # noqa: E402

RUNS = ["C_s%d" % s for s in CC.SEEDS]


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    R = pd.read_csv(HERE / "armC_runs.csv").set_index("run")
    S = pd.read_csv(HERE / "armC_significance.csv").set_index("run")
    with open(HERE / "armC_preds.pkl", "rb") as fh:
        P = pickle.load(fh)
    median_seed = P["median_seed"]
    fz = R.loc["FROZEN"]

    def col(name, runs=RUNS, frame=R):
        return [float(frame.loc[r, name]) for r in runs]

    per = pd.DataFrame({"seed": list(CC.SEEDS)})
    gates: list[dict] = []

    def add(gid, name, rule, b, ok, unit="", per_seed=None, note=""):
        gates.append({
            "gate": gid, "name": name, "rule": rule,
            "median": round(b["median"], 6), "band_lo": round(b["lo"], 6),
            "band_hi": round(b["hi"], 6), "verdict": verdict(ok), "unit": unit,
            "per_seed": "" if per_seed is None else
            " ".join(f"{v:.4f}" for v in per_seed), "note": note,
        })

    # ---------------- gate 1: ND no-regression --------------------------
    m1, s1 = col("nd_test_mape"), col("nd_test_sdpe")
    per["nd_mape"], per["nd_sdpe"] = m1, s1
    b_m, b_s = CC.band(m1), CC.band(s1)
    g1 = b_m["median"] <= CC.G1_MAPE_MAX and b_s["median"] <= CC.G1_SDPE_MAX
    add(1, "ND no-regression", f"median MAPE <= {CC.G1_MAPE_MAX} AND "
        f"median SDPE <= {CC.G1_SDPE_MAX}", b_m, g1, "MAPE %", m1,
        f"SDPE median {b_s['median']:.4f} [{b_s['lo']:.4f}, {b_s['hi']:.4f}] "
        f"vs cap {CC.G1_SDPE_MAX}; frozen {fz.nd_test_mape:.4f}/{fz.nd_test_sdpe:.4f}")

    # ---------------- gate 2: dispersion vs Asymptotic_MST --------------
    sw = col("disp_p_swap_holm", frame=S)
    pm = col("disp_p_pm_holm", frame=S)
    bf = col("disp_p_bf_holm", frame=S)
    det = [bool(S.loc[r, "disp_detectable_holm"]) for r in RUNS]
    ratio = col("disp_sd_ratio", frame=S)
    per["disp_sd_ratio"], per["p_swap_holm"] = ratio, sw
    per["p_pm_holm"], per["p_bf_holm"], per["detectable_holm"] = pm, bf, det
    b_sw = CC.band(sw)
    g2 = (b_sw["median"] < 0.05) and (float(np.median(ratio)) < 1.0) \
        and (sum(det) > len(det) / 2)
    add(2, "TSPLIB dispersion vs Asymptotic_MST",
        "swap-permutation Holm p < 0.05 AND detectable_holm True "
        "(Pitman-Morgan alone insufficient)", b_sw, g2, "Holm p (swap)", sw,
        f"PM Holm median {np.median(pm):.4g} [{min(pm):.4g}, {max(pm):.4g}]; "
        f"BF Holm median {np.median(bf):.4g}; "
        f"detectable_holm {sum(det)}/7; "
        f"sd_ratio median {np.median(ratio):.4f} "
        f"[{min(ratio):.4f}, {max(ratio):.4f}]; "
        f"seeds with swap p<0.05: {sum(p < 0.05 for p in sw)}/7; "
        f"frozen swap Holm p {float(S.loc['FROZEN','disp_p_swap_holm']):.4g}")

    # ---------------- gate 3: mean gain vs Calibrated_MST_dn ------------
    gn, mde, ph = (col("mean_gain", frame=S), col("mean_mde", frame=S),
                   col("mean_p_holm", frame=S))
    per["mean_gain"], per["mean_mde"], per["mean_p_holm"] = gn, mde, ph
    b_g = CC.band(gn)
    g3 = (b_g["median"] > float(np.median(mde))) and (float(np.median(ph)) < 0.05)
    add(3, "Mean gain vs Calibrated_MST_dn", "median gain > MDE AND Holm p < 0.05",
        b_g, g3, "APE points", gn,
        f"MDE median {np.median(mde):.4f} [{min(mde):.4f}, {max(mde):.4f}]; "
        f"Holm p median {np.median(ph):.4g} [{min(ph):.4g}, {max(ph):.4g}]; "
        f"gain>MDE in {sum(a > b for a, b in zip(gn, mde))}/7 seeds; "
        f"frozen gain {float(S.loc['FROZEN','mean_gain']):.4f} "
        f"(MDE {float(S.loc['FROZEN','mean_mde']):.4f})")

    # ---------------- gate 4: monotonicity (artifact-integrity) ---------
    md, mn = col("mono_dimension_pct"), col("mono_n_customers_pct")
    vd = max(col("mono_dimension_maxviol") + col("mono_n_customers_maxviol"))
    per["mono_d_pct"], per["mono_n_pct"] = md, mn
    b_md = CC.band(md)
    g4 = all(v == 100.0 for v in md + mn) and vd == 0.0
    u = R.loc["Cuncon"]
    add(4, "Monotonicity probe (artifact integrity only)",
        "100% non-increasing on both axes; unconstrained control reported beside",
        b_md, g4, "% non-increasing", md,
        f"n_customers axis {min(mn):.1f}-{max(mn):.1f}%; max violation {vd:.1e}; "
        f"UNCONSTRAINED CONTROL (same rows, same seed {median_seed}, "
        f"constraints removed): dimension {u.mono_dimension_pct:.1f}%, "
        f"n_customers {u.mono_n_customers_pct:.1f}%, max violation "
        f"{max(u.mono_dimension_maxviol, u.mono_n_customers_maxviol):.3e}. "
        "A sum of monotone-constrained trees cannot violate the sweep, so 100% "
        "is evidence the constraint was applied, not evidence about arm C.")

    # ---------------- gate 5: extraction cost ---------------------------
    same = all(bool(R.loc[r, "features_equal_frozen"]) for r in RUNS)
    b5 = {"median": 1.0, "lo": 1.0, "hi": 1.0, "k": 7}
    add(5, "Extraction cost", f"ratio <= {CC.G5_COST_RATIO_MAX}", b5, same, "ratio",
        None, "Structural, not timed: every arm C booster carries the identical "
        f"31 frozen feature names (verified for {sum(bool(R.loc[r,'features_equal_frozen']) for r in RUNS)}/7 "
        "seeds), so the extraction code path is unchanged and the ratio is "
        "exactly 1.00 by construction. No wall-clock was measured.")

    # ---------------- gate 6: LineNoise slope (band form) ---------------
    sl = col("linenoise_slope")
    per["linenoise_slope"] = sl
    b_sl = CC.band(sl)
    n_ok = sum(v >= CC.G6_SLOPE_MIN for v in sl)
    g6 = (b_sl["median"] >= CC.G6_SLOPE_MIN) and (n_ok >= CC.G6_MIN_SEEDS)
    add(6, "LineNoise slope, n>=200",
        f"median >= {CC.G6_SLOPE_MIN} AND >= {CC.G6_MIN_SEEDS}/7 seeds "
        f">= {CC.G6_SLOPE_MIN}", b_sl, g6, "OLS slope", sl,
        f"seeds >= 0.70: {n_ok}/7 (fails: "
        + ", ".join(f"seed {s}={v:.4f}" for s, v in zip(CC.SEEDS, sl)
                    if v < CC.G6_SLOPE_MIN) + "); frozen "
        f"{fz.linenoise_slope:.4f}; N={int(fz.linenoise_slope_n)}")

    # ---------------- gate 7: grid MSPE ---------------------------------
    gm = col("b2_grid_mspe")
    per["grid_mspe"] = gm
    b_gm = CC.band(gm)
    g7 = b_gm["median"] <= CC.G7_GRID_MSPE_MAX
    add(7, "grid MSPE", f"median <= +{CC.G7_GRID_MSPE_MAX}", b_gm, g7, "MSPE %", gm,
        f"frozen {fz.b2_grid_mspe:+.4f}")

    # ---------------- gate 8: 2D class regression -----------------------
    worst, wname = [], []
    for r in RUNS:
        d = {g: float(R.loc[r, f"b2_{g}_mape"]) - CC.PROD_2D[g] for g in CC.PROD_2D}
        k = max(d, key=d.get)
        worst.append(d[k])
        wname.append(k)
    per["worst_class_delta"] = worst
    b_w = CC.band(worst)
    g8 = b_w["median"] <= CC.G8_MAX_DELTA
    add(8, "2D class regression", f"no class worse than frozen by > "
        f"{CC.G8_MAX_DELTA} MAPE, at the median", b_w, g8, "delta MAPE", worst,
        "worst class per seed: " + ", ".join(sorted(set(wname)))
        + f"; seeds over threshold {sum(v > CC.G8_MAX_DELTA for v in worst)}/7")

    # ---------------- gate 9: TSPLIB non-Euclidean ----------------------
    ne_m, ne_s = col("tsplib_noneuc_mape"), col("tsplib_noneuc_sdpe")
    cov = [int(R.loc[r, "tsplib_noneuc_coverage"]) for r in RUNS]
    per["noneuc_mape"], per["noneuc_sdpe"], per["noneuc_cov"] = ne_m, ne_s, cov
    b_ne = CC.band(ne_m)
    g9 = (b_ne["median"] <= CC.G9_NONEUC_MAPE
          and float(np.median(ne_s)) <= CC.G9_NONEUC_SDPE
          and min(cov) >= CC.G9_MIN_COVERAGE)
    add(9, "TSPLIB non-Euclidean",
        f"median MAPE <= {CC.G9_NONEUC_MAPE} AND median SDPE <= "
        f"{CC.G9_NONEUC_SDPE} AND coverage >= {CC.G9_MIN_COVERAGE}/23",
        b_ne, g9, "MAPE %", ne_m,
        f"SDPE median {np.median(ne_s):.4f} [{min(ne_s):.4f}, {max(ne_s):.4f}] "
        f"vs {CC.G9_NONEUC_SDPE}; coverage {min(cov)}-{max(cov)}/23; "
        f"same-harness frozen MAPE {fz.tsplib_noneuc_mape:.4f} / SDPE "
        f"{fz.tsplib_noneuc_sdpe:.4f} (protocol literal 3.3441/3.8931 used as "
        "the binding, stricter threshold)")

    # ---------------- gate 10: TSPLIB EUC_2D dispersion cost ------------
    sd = col("tsplib_euc2d_sdpe")
    mad = col("tsplib_euc2d_mad")
    tsd = col("tsplib_euc2d_trimsd10")
    r_sd = [v / fz.tsplib_euc2d_sdpe for v in sd]
    r_mad = [v / fz.tsplib_euc2d_mad for v in mad]
    r_tsd = [v / fz.tsplib_euc2d_trimsd10 for v in tsd]
    per["euc2d_sdpe"], per["euc2d_sd_ratio"] = sd, r_sd
    per["euc2d_mad_ratio"], per["euc2d_trimsd_ratio"] = r_mad, r_tsd
    b_r = CC.band(r_sd)
    g10 = (b_r["median"] < CC.G10_SD_RATIO_MAX
           and float(np.median(r_mad)) <= CC.G10_ROBUST_RATIO_MAX
           and float(np.median(r_tsd)) <= CC.G10_ROBUST_RATIO_MAX)
    add(10, "TSPLIB EUC_2D dispersion cost",
        f"SDPE ratio vs frozen < {CC.G10_SD_RATIO_MAX} AND MAD ratio <= "
        f"{CC.G10_ROBUST_RATIO_MAX} AND 10%-trimmed SD ratio <= "
        f"{CC.G10_ROBUST_RATIO_MAX}", b_r, g10, "SDPE ratio", r_sd,
        f"MAD ratio median {np.median(r_mad):.4f} "
        f"[{min(r_mad):.4f}, {max(r_mad):.4f}]; trimmed-SD ratio median "
        f"{np.median(r_tsd):.4f} [{min(r_tsd):.4f}, {max(r_tsd):.4f}]; "
        f"frozen SDPE {fz.tsplib_euc2d_sdpe:.4f}, MAD {fz.tsplib_euc2d_mad:.4f}, "
        f"trimmed SD {fz.tsplib_euc2d_trimsd10:.4f}")

    # ---------------- gate 11: A2 one-constant control, slope view ------
    marg = [v - CC.G11_RECAL_FROZEN_SLOPE for v in sl]
    per["slope_margin_vs_recal"] = marg
    b_mg = CC.band(marg)
    g11 = b_mg["median"] >= CC.G11_MIN_MARGIN
    add(11, "A2 one-constant control, slope view",
        f"median slope - recalibrated frozen slope "
        f"({CC.G11_RECAL_FROZEN_SLOPE:.4f}) >= {CC.G11_MIN_MARGIN}",
        b_mg, g11, "slope margin", marg,
        f"seeds meeting margin {sum(v >= CC.G11_MIN_MARGIN for v in marg)}/7")

    G = pd.DataFrame(gates)
    G.to_csv(HERE / "armC_gate_table.csv", index=False)
    per.to_csv(HERE / "armC_perseed.csv", index=False)

    # =================== reporting obligations (s4) ======================
    ob: dict = {}
    s = oh.load_suite()["tsplib_euc2d"]
    truth = s.truth
    fzp = pd.Series({k: v for k, v in P["preds_all"]["FROZEN"].items()
                     if k in truth.index}, dtype=float).reindex(truth.index)

    # --- 1: EUC_2D dispersion regression vs FROZEN -----------------------
    disp_rows = []
    for r in RUNS:
        cp = pd.Series({k: v for k, v in P["preds_all"][r].items()
                        if k in truth.index}, dtype=float).reindex(truth.index)
        d = oh.compare_dispersion(cp, fzp, truth, r, "FROZEN")
        e_c = ((cp - truth) / truth * 100.0).dropna()
        e_f = ((fzp - truth) / truth * 100.0).dropna()
        ix = [i for i in e_c.index if "ts225" in str(i)]
        keep = [i for i in e_c.index if i not in ix]
        disp_rows.append({
            "run": r, "sdpe_cand": d["sdpe_a"], "sdpe_frozen": d["sdpe_b"],
            "sd_ratio": d["sd_ratio"], "ci_low": d["ratio_ci_low"],
            "ci_high": d["ratio_ci_high"], "r_xy": d["r_xy"],
            "p_pitman_morgan": d["p_pitman_morgan"],
            "p_swap_permutation": d["p_swap_permutation"],
            "p_brown_forsythe": d["p_brown_forsythe"],
            "mde_sd_ratio": d["mde_sd_ratio"], "detectable": d["detectable"],
            "mad_ratio": CC._mad(e_c.to_numpy()) / CC._mad(e_f.to_numpy()),
            "trimsd_ratio": (CC._trimmed_sd(e_c.to_numpy())
                             / CC._trimmed_sd(e_f.to_numpy())),
            "ts225_err_cand": float(e_c[ix[0]]) if ix else np.nan,
            "ts225_err_frozen": float(e_f[ix[0]]) if ix else np.nan,
            "sd_ratio_ex_ts225": float(np.std(e_c[keep], ddof=1)
                                       / np.std(e_f[keep], ddof=1)),
        })
    DZ = pd.DataFrame(disp_rows)
    DZ.to_csv(HERE / "armC_dispersion_vs_frozen.csv", index=False)
    ob["1_euc2d_dispersion_regression"] = {
        "sd_ratio_vs_frozen": CC.band(DZ.sd_ratio),
        "bootstrap_ci_low": CC.band(DZ.ci_low), "bootstrap_ci_high": CC.band(DZ.ci_high),
        "p_pitman_morgan": CC.band(DZ.p_pitman_morgan),
        "p_swap_permutation": CC.band(DZ.p_swap_permutation),
        "mde_sd_ratio": CC.band(DZ.mde_sd_ratio),
        "detectable_seeds": f"{int(DZ.detectable.sum())}/7",
        "mad_ratio": CC.band(DZ.mad_ratio), "trimsd_ratio": CC.band(DZ.trimsd_ratio),
        "sd_ratio_excluding_ts225": CC.band(DZ.sd_ratio_ex_ts225),
        "ts225_signed_error_pct_cand": CC.band(DZ.ts225_err_cand),
        "ts225_signed_error_pct_frozen": float(DZ.ts225_err_frozen.iloc[0]),
        "gate2_vs_Asymptotic_MST_sd_ratio": CC.band(ratio),
    }

    # --- 2: A2 one-constant control -------------------------------------
    def oracle(gen: str, cand_run: str) -> dict:
        f = P["b2"]["FROZEN"]
        c = P["b2"][cand_run]
        f = f[f.generator == gen]
        c = c[c.generator == gen]
        true, pred = f.true_cost.to_numpy(), f.pred_cost.to_numpy()
        grid = np.linspace((true / pred).min(), (true / pred).max(), 4001)
        mapes = np.array([np.mean(np.abs((g * pred - true) / true)) * 100 for g in grid])
        k = int(np.argmin(mapes))
        fm = float(np.mean(np.abs(f.err_pct)))
        cm = float(np.mean(np.abs(c.err_pct)))
        ta = np.clip(true / f.mst_total_length.to_numpy(), *K.ALPHA_CLIP)
        from scipy import stats as st
        sl_recal = float(st.linregress(
            ta, np.clip(grid[k] * f.pred_alpha.to_numpy(), *K.ALPHA_CLIP)).slope)
        return {"constant": float(grid[k]), "frozen_mape": fm,
                "recal_mape": float(mapes[k]), "cand_mape": cm,
                "gain_constant": fm - float(mapes[k]), "gain_cand": fm - cm,
                "frac_of_cand_gain_by_constant":
                    (fm - float(mapes[k])) / (fm - cm) if fm != cm else np.nan,
                "slope_recal_frozen": sl_recal}

    a2 = {}
    for gen in ("line_noise", "grid"):
        per_run = [oracle(gen, r) for r in RUNS]
        a2[gen] = {
            "oracle_constant": per_run[0]["constant"],
            "frozen_mape": per_run[0]["frozen_mape"],
            "frozen_recalibrated_mape": per_run[0]["recal_mape"],
            "candidate_mape": CC.band([p["cand_mape"] for p in per_run]),
            "frac_of_candidate_gain_the_constant_already_buys":
                CC.band([p["frac_of_cand_gain_by_constant"] for p in per_run]),
            "recalibrated_frozen_slope": per_run[0]["slope_recal_frozen"],
        }
    a2["slope_view"] = {
        "candidate_linenoise_slope": CC.band(sl),
        "recalibrated_frozen_slope": CC.G11_RECAL_FROZEN_SLOPE,
        "margin": CC.band(marg), "required": CC.G11_MIN_MARGIN,
    }
    ob["2_A2_control"] = a2

    # --- 3: line_noise design provenance --------------------------------
    ob["3_line_noise_provenance"] = {
        "source": "data_pipeline/augment_gen.py:1018-1031 (batch 2 note)",
        "statement": (
            "The augmentation's rho targets were chosen by profiling the 210 "
            "benchmark line_noise instances: their rho medians (0.94 / 5.23 / "
            "25.19 / 45.66 across n buckets, max 96.3) drove batch 2's rho grid "
            "of (2, 4, 8, 16, 32, 64, 100). No benchmark instance and no "
            "benchmark label was ever read into training, so this is not "
            "leakage; but the coverage was deliberately aimed at the evaluation "
            "set's locus and must be disclosed as test-locus-targeted coverage."),
        "batch2_rho_grid": [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0],
    }

    # --- 4: non-blind disclosures ---------------------------------------
    ob["4_pre_registration_caveats"] = {
        "0a_threshold_inherited": (
            "The 0.70 slope threshold of gate 6 is not fresh. "
            "augmentation_v2_criteria.csv (2026-08-10 22:02:37) already carried "
            "'LineNoise alpha-slope (n>=200) >= 0.70 (from 0.28)' and scored an "
            "earlier augmented arm against it (FAIL, 0.4286). Gate 7's '<= +4.0' "
            "likewise restates an outcome already shown reachable. Arm C itself "
            "was not scored before its gates were fixed, but the thresholds were "
            "not chosen blind."),
        "0b_rule_found_by_looking": (
            "The de-contamination rule was discovered by adversarial "
            "leave-one-family-out, which observed that removing the d=2 lattice "
            "rows improves grid. The rule as stated is outcome-independent and "
            "was re-registered and re-run from scratch here; the previous run's "
            "numbers are discarded, not carried forward."),
        "0c_row_order_defect_this_run_fixes": (
            "Arm A's reported point estimates were the favourable extreme of "
            "their own nuisance distribution (slope 0.8268 above the entire "
            "8-permutation band [0.6367, 0.8028]). Every arm C statistic here is "
            "a 7-seed median with the full band."),
    }

    # --- 5: ND gain described correctly ---------------------------------
    D = K.load_cache()
    C, feats = D["cache"], D["feats31"]
    ndf = C[C.stratum == "nd_test"]
    frozen = joblib.load(K.FROZEN)
    fo = K.score_frame(frozen, feats, ndf).set_index("instance")
    nd_rows = []
    for r in RUNS:
        m = joblib.load(HERE / "armC_models" / f"{r}.joblib")
        co = K.score_frame(m, feats, ndf).set_index("instance")
        ix = fo.index.intersection(co.index)
        af, ac = fo.loc[ix, "err_pct"].abs(), co.loc[ix, "err_pct"].abs()
        d = (ac - af)
        nd_rows.append({"run": r, "n": int(len(ix)),
                        "mape_frozen": float(af.mean()), "mape_cand": float(ac.mean()),
                        "net_gain_pp": float(af.mean() - ac.mean()),
                        "median_delta_pp": float(d.median()),
                        "pct_worse": float((d > 0).mean() * 100),
                        "pct_better": float((d < 0).mean() * 100),
                        "p90_improvement_pp": float(-d.quantile(0.10))})
    ND = pd.DataFrame(nd_rows)
    ND.to_csv(HERE / "armC_nd_decomposition.csv", index=False)
    ob["5_nd_gain_decomposition"] = {
        "net_mape_gain_pp": CC.band(ND.net_gain_pp),
        "median_instance_delta_pp": CC.band(ND.median_delta_pp),
        "pct_of_held_out_instances_that_get_worse": CC.band(ND.pct_worse),
        "statement": ("Net ND MAPE improves, but the median instance moves only "
                      "a few thousandths of a percentage point and roughly half "
                      "of held-out instances get worse. It is a trade of accuracy "
                      "on already-easy instances for accuracy on the harder half, "
                      "not a uniform improvement."),
    }

    # --- 6: augment stratum as a held-out refit -------------------------
    HO = pd.read_csv(HERE / "armC_augment_heldout.csv")
    ob["6_augment_stratum_held_out"] = {
        "held_out_5fold_refit_MAPE": float(HO.err_pct.abs().mean()),
        "held_out_n": int(len(HO)),
        "held_out_by_fold": {int(k): round(float(v), 4) for k, v
                             in HO.groupby("fold").err_pct.apply(
                                 lambda x: x.abs().mean()).items()},
        "in_sample_MAPE_EXCLUDED": CC.band(col("augment_mape")),
        "frozen_MAPE_on_same_rows": float(fz.augment_mape),
        "statement": ("The in-sample augment figure is excluded per the protocol. "
                      "The held-out value is a 5-fold refit over the 850 retained "
                      "rows: each fold's rows are scored by a model trained "
                      "without them."),
    }

    # --- row-order sensitivity (protocol s2 bullet 4) --------------------
    pm_runs = [r for r in R.index if r.startswith("Cperm")]
    ob["row_order_sensitivity_at_median_seed"] = {
        "median_seed": int(median_seed), "n_permutations": len(pm_runs),
        "definition": ("median seed = the seed whose LineNoise slope is the "
                       "median of the 7, i.e. the gate-6 statistic"),
        "linenoise_slope": CC.band([float(R.loc[r, "linenoise_slope"]) for r in pm_runs]),
        "protocol_order_value_at_that_seed":
            float(R.loc[f"C_s{median_seed}", "linenoise_slope"]),
        "b2_line_noise_mape": CC.band([float(R.loc[r, "b2_line_noise_mape"]) for r in pm_runs]),
        "b2_grid_mspe": CC.band([float(R.loc[r, "b2_grid_mspe"]) for r in pm_runs]),
        "tsplib_euc2d_sdpe": CC.band([float(R.loc[r, "tsplib_euc2d_sdpe"]) for r in pm_runs]),
        "permutation_seeds_below_gate6_threshold":
            sum(float(R.loc[r, "linenoise_slope"]) < CC.G6_SLOPE_MIN for r in pm_runs),
    }

    # --- order-only control: protocol order, no augmentation -------------
    r0 = [f"R0p_s{s}" for s in CC.SEEDS]
    ob["order_only_control_R0_protocol_order"] = {
        "why": ("Protocol s2's fixed concatenation order re-sorts the corpus "
                "block and is itself a row-order change relative to the frozen "
                "fit. This arm isolates that change: same 7 seeds, protocol "
                "order, zero augmentation rows."),
        "linenoise_slope": CC.band([float(R.loc[r, "linenoise_slope"]) for r in r0]),
        "b2_grid_mspe": CC.band([float(R.loc[r, "b2_grid_mspe"]) for r in r0]),
        "bench2d_mape": CC.band([float(R.loc[r, "bench2d_mape"]) for r in r0]),
        "nd_test_mape": CC.band([float(R.loc[r, "nd_test_mape"]) for r in r0]),
        "tsplib_euc2d_sdpe": CC.band([float(R.loc[r, "tsplib_euc2d_sdpe"]) for r in r0]),
    }

    # --- arm A under the identical protocol, for the delta ---------------
    ar = [f"A_s{s}" for s in CC.SEEDS]
    ob["arm_A_under_the_same_protocol"] = {
        "linenoise_slope": CC.band([float(R.loc[r, "linenoise_slope"]) for r in ar]),
        "b2_grid_mspe": CC.band([float(R.loc[r, "b2_grid_mspe"]) for r in ar]),
        "bench2d_mape": CC.band([float(R.loc[r, "bench2d_mape"]) for r in ar]),
        "tsplib_euc2d_sdpe": CC.band([float(R.loc[r, "tsplib_euc2d_sdpe"]) for r in ar]),
        "disp_p_swap_holm": CC.band(col("disp_p_swap_holm", ar, S)),
        "seeds_with_swap_p_below_0p05":
            f"{sum(float(S.loc[r,'disp_p_swap_holm']) < 0.05 for r in ar)}/7",
    }

    ob["directional_requirement_bench2d_overall"] = {
        "target": 2.9042, "candidate": CC.band(col("bench2d_mape")),
        "met": bool(CC.band(col("bench2d_mape"))["median"] < 2.9042),
    }

    json.dump(ob, open(HERE / "armC_obligations.json", "w"), indent=2, default=str)

    # =================== print =========================================
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print("\n=============== ARM C GATE TABLE (median over 7 seeds, "
          "full min-max band) ===============")
    print(G[["gate", "name", "median", "band_lo", "band_hi", "verdict",
             "unit"]].to_string(index=False))
    n_pass = int((G.verdict == "PASS").sum())
    print(f"\nSWEEP: {n_pass}/11 gates pass. "
          f"FAILED: {', '.join(str(g) for g in G.loc[G.verdict=='FAIL','gate'])}"
          if n_pass < 11 else f"\nSWEEP: {n_pass}/11 -- clean.")
    print("\n--- per-seed values ---")
    print(per.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- notes ---")
    for _, r in G.iterrows():
        print(f"\n[gate {r.gate}] {r.verdict}  {r['name']}")
        print(f"  rule   : {r.rule}")
        print(f"  median : {r['median']:.6f}  "
              f"band [{r.band_lo:.6f}, {r.band_hi:.6f}]")
        if r.per_seed:
            print(f"  seeds  : {r.per_seed}")
        if r.note:
            print(f"  note   : {r.note}")
    print("\nwrote armC_gate_table.csv armC_perseed.csv armC_obligations.json "
          "armC_dispersion_vs_frozen.csv armC_nd_decomposition.csv")


if __name__ == "__main__":
    main()
