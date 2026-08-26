"""Gate scoring for the support/feature arms. Imported by support_arms_study.py.

Gates 1-4 and 6-9 plus both adversarial checks. Gate 5 (extraction cost) is a
separate single-process measurement in support_arms_cost.py, because it has to
run interleaved and undisturbed.
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

SEED = 42
ALPHA_CLIP = (1.0, 2.0)
NEW3 = ["mst_topology_straightness", "mst_topology_deg2_straight_mean",
        "degeneracy_pca_effective_rank"]

# Production control numbers supplied by the coordinator (see the gates file).
PROD = {
    "nd_mape": 0.6201, "nd_sdpe": 0.9881,
    "bench2d_mape": 2.9042,
    "grid_mape": 7.1121, "line_noise_mape": 10.7522,
    "cluster_mape": 2.1847, "others_mape": 1.7191,
}
GATE1_MAPE_MAX = PROD["nd_mape"] + 0.02
GATE1_SDPE_MAX = PROD["nd_sdpe"] + 0.02

PROBE_GRID_POINTS = 24
PROBE_N_INSTANCES = 1000
PROBE_TOL = 1e-9


def to_alpha(z) -> np.ndarray:
    return 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _summary(err_pct: np.ndarray) -> dict:
    e = np.asarray(err_pct, dtype=float)
    return {"n": int(e.size), "mape": float(np.mean(np.abs(e))),
            "sdpe": float(np.std(e, ddof=1)) if e.size > 1 else float("nan"),
            "mspe": float(np.mean(e))}


def _score(model, feats, frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame[(frame.status == "ok") & frame[feats].notna().all(axis=1)
               & frame.true_cost.notna()].copy()
    a = np.clip(to_alpha(model.predict(ok[feats], num_iteration=model.best_iteration)),
                *ALPHA_CLIP)
    ok["pred_alpha"] = a
    ok["pred_cost"] = a * ok["mst_total_length"].to_numpy()
    ok["err_pct"] = (ok["pred_cost"] - ok["true_cost"]) / ok["true_cost"] * 100.0
    return ok


# =========================================================================
# gate 4: monotonicity probe
# =========================================================================
def _log_int_grid(lo: int, hi: int, k: int) -> list[int]:
    g = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), k)).astype(int))
    return [int(v) for v in g]


PROBE_D_GRID = _log_int_grid(2, 200, PROBE_GRID_POINTS)
PROBE_N_GRID = _log_int_grid(5, 4000, PROBE_GRID_POINTS)


def monotonicity(model, feats, base: pd.DataFrame, col: str, grid) -> dict:
    preds = np.empty((len(base), len(grid)), dtype=np.float64)
    for j, g in enumerate(grid):
        X = base.copy()
        X[col] = g
        preds[:, j] = model.predict(X[feats], num_iteration=model.best_iteration)
    out = {}
    for kind, P in (("raw", preds), ("deployed", to_alpha(preds))):
        d = np.diff(P, axis=1)
        viol = d > PROBE_TOL
        inc = d[viol]
        out[f"pct_nonincr_{kind}"] = float((~viol.any(axis=1)).mean()) * 100.0
        out[f"n_viol_{kind}"] = int(viol.sum())
        out[f"viol_max_{kind}"] = float(inc.max()) if inc.size else 0.0
    return out


# =========================================================================
# gates 2 and 3: TSPLIB significance, via the shipped harness
# =========================================================================
def tsplib_significance(preds_all: dict, label: str) -> dict:
    """Dispersion vs Asymptotic_MST and mean gain vs Calibrated_MST_dn.

    ``preds_all`` maps instance -> predicted tour length across EVERY stratum,
    so ``evaluate_candidate`` builds the same multiplicity family the frozen
    model was judged in. Both tests are read off the harness rather than
    reimplemented, so the Holm correction is comparable by construction.
    """
    import ood_harness as oh

    res: dict = {"model": label}
    # -- gate 2 ----------------------------------------------------------
    try:
        d = oh.dispersion_verdict({label: preds_all}, stratum="tsplib_euc2d")
        row = d[d.model_b == "Asymptotic_MST"]
        if len(row):
            r = row.iloc[0]
            res.update({
                "disp_n": int(r["n_pairs"]), "disp_ratio": float(r["sd_ratio"]),
                "disp_p_holm": float(r["p_holm"]),
                "disp_family_size": int(r["family_size"])
                if "family_size" in r and np.isfinite(r["family_size"]) else len(d),
                "disp_mde_holm": float(r.get("mde_sd_ratio_holm", np.nan)),
            })
    except Exception as e:  # noqa: BLE001
        res["disp_error"] = f"{type(e).__name__}: {e}"

    # -- gate 3 ----------------------------------------------------------
    try:
        v = oh.evaluate_candidate(preds_all, label)
        c = v.comparisons
        c = c[(c.stratum == "tsplib_euc2d") & (c.model_b == "Calibrated_MST_dn")]
        if len(c):
            r = c.iloc[0]
            suite = oh.load_suite()["tsplib_euc2d"]
            pa = pd.Series({k: val for k, val in preds_all.items()
                            if k in suite.truth.index}, dtype=float)
            pb = suite.baselines["Calibrated_MST_dn"]
            idx = [i for i in pa.index if i in pb.index and np.isfinite(pb[i])]
            a = oh.absolute_percent_error(pa.loc[idx], suite.truth.loc[idx])
            b = oh.absolute_percent_error(pb.loc[idx], suite.truth.loc[idx])
            diff = (a - b).to_numpy()
            sd = float(np.std(diff, ddof=1))
            res.update({
                "mean_n": int(r["n_pairs"]),
                "mean_gain": -float(r["mean_diff"]),
                "mean_p_holm": float(r["p_holm"]),
                "mean_family_size": int(v.family_size),
                "mean_mde": float(oh.min_detectable_difference(sd, len(diff))),
                "mape_model": float(r["mape_a"]), "mape_baseline": float(r["mape_b"]),
            })
    except Exception as e:  # noqa: BLE001
        res["mean_error"] = f"{type(e).__name__}: {e}"
    return res


# =========================================================================
# adversarial checks
# =========================================================================
def adv_near_duplicate(corpus: pd.DataFrame, aug: pd.DataFrame,
                       bench: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """How close, in feature space, are the added instances to the benchmark?

    For each degenerate benchmark instance we compare two distances: to the
    nearest AUGMENTATION instance, and to the nearest OTHER BENCHMARK instance
    of the same generator. If the augmentation set were a disguised copy of the
    benchmark, the first would be at or below the second. The benchmark's own
    within-family spacing is the only honest yardstick here -- an absolute
    distance means nothing without it.
    """
    from scipy.spatial import cKDTree

    tr = corpus[corpus.split == "train"]
    mu = tr[feats].mean().to_numpy()
    sd = tr[feats].std(ddof=0).replace(0, 1.0).to_numpy()

    def Z(df):
        return (df[feats].to_numpy(dtype=float) - mu) / sd

    Za = Z(aug)
    tree_a = cKDTree(Za)
    rows = []
    for gen, g in bench.groupby("generator"):
        if len(g) < 2:
            continue
        Zb = Z(g)
        d_aug, _ = tree_a.query(Zb, k=1)
        d_own, _ = cKDTree(Zb).query(Zb, k=2)      # k=1 is the point itself
        d_own = d_own[:, 1]
        rows.append({
            "generator": gen, "n_bench": len(g),
            "nn_to_augment_median": float(np.median(d_aug)),
            "nn_to_augment_p05": float(np.percentile(d_aug, 5)),
            "nn_to_augment_min": float(d_aug.min()),
            "nn_within_family_median": float(np.median(d_own)),
            "ratio_median": float(np.median(d_aug) / max(np.median(d_own), 1e-12)),
            "n_closer_to_aug_than_own_family": int((d_aug < d_own).sum()),
        })
    return pd.DataFrame(rows).sort_values("ratio_median")


def adv_one_constant(frozen_scored: pd.DataFrame, cand_scored: pd.DataFrame,
                     generator: str) -> dict:
    """Does the arm beat the frozen model rescaled by ONE in-sample constant?

    The constant is fitted on the same family it is evaluated on, which is the
    most generous version of the control possible: it cannot be beaten by any
    other single constant. If the arm's gain does not clearly exceed this, the
    arm has learned a family offset and not a mechanism.
    """
    f = frozen_scored[frozen_scored.generator == generator]
    c = cand_scored[cand_scored.generator == generator]
    true = f["true_cost"].to_numpy()
    pred = f["pred_cost"].to_numpy()

    # Constant minimising MAPE is the weighted median of true/pred; solve on a
    # fine grid instead, which is exact enough and assumption-free.
    ratio = true / pred
    grid = np.linspace(ratio.min(), ratio.max(), 4001)
    mapes = np.array([np.mean(np.abs((g * pred - true) / true)) * 100.0 for g in grid])
    k = int(np.argmin(mapes))
    best_c, best_mape = float(grid[k]), float(mapes[k])

    cand_mape = float(np.mean(np.abs(c["err_pct"])))
    frozen_mape = float(np.mean(np.abs(f["err_pct"])))
    return {
        "generator": generator, "n": int(len(f)),
        "frozen_mape": frozen_mape,
        "frozen_recal_constant": best_c,
        "frozen_recal_mape": best_mape,
        "gain_from_one_constant": frozen_mape - best_mape,
        "candidate_mape": cand_mape,
        "gain_from_candidate": frozen_mape - cand_mape,
        "candidate_beats_constant_by": best_mape - cand_mape,
    }


# =========================================================================
# driver
# =========================================================================
def run_eval(st: dict, frames: dict) -> None:
    arms = st["arms"]
    corpus = st["corpus"]
    aug = st["aug"]
    feats31 = st["feats31"]
    C = frames["cache"]

    order = ["FROZEN", "R0", "A", "B"]
    arms = {k: arms[k] for k in order if k in arms}

    strata_rows, gate_rows, per_inst = [], [], []
    probe_rows, sig_rows = [], []

    # base frame for the monotonicity probe: same protocol as the shipped probe
    nd = corpus[corpus.split == "test"].reset_index(drop=True)
    probe_base = nd.sample(min(PROBE_N_INSTANCES, len(nd)), random_state=SEED).copy()

    scored: dict[str, dict] = {}
    for tag, (model, feats) in arms.items():
        scored[tag] = {}
        for stratum, g in C.groupby("stratum"):
            ok = _score(model, feats, g)
            scored[tag][stratum] = ok
            s = _summary(ok["err_pct"].to_numpy())
            strata_rows.append({"model": tag, "stratum": stratum,
                                "n_stratum": int(len(g)), **s,
                                "coverage": len(ok) / len(g)})
            pi = ok[["instance", "pred_cost", "true_cost", "err_pct",
                     "pred_alpha", "generator"]].copy()
            pi.insert(0, "stratum", stratum)
            pi.insert(0, "model", tag)
            per_inst.append(pi)

        # gate 4
        for col, grid in (("dimension", PROBE_D_GRID), ("n_customers", PROBE_N_GRID)):
            r = monotonicity(model, feats, probe_base, col, grid)
            r.update({"model": tag, "swept": col})
            probe_rows.append(r)

        # gates 2 + 3
        allp = {}
        for stratum, ok in scored[tag].items():
            allp.update(dict(zip(ok["instance"].astype(str),
                                 ok["pred_cost"].astype(float))))
        sig_rows.append(tsplib_significance(allp, tag))
        print(f"[eval] {tag} scored", flush=True)

    S = pd.DataFrame(strata_rows)
    P = pd.DataFrame(probe_rows)
    G = pd.DataFrame(sig_rows)
    PI = pd.concat(per_inst, ignore_index=True)
    S.to_csv(HERE / "support_arms_strata.csv", index=False)
    P.to_csv(HERE / "support_arms_probe.csv", index=False)
    G.to_csv(HERE / "support_arms_significance.csv", index=False)
    PI.to_csv(HERE / "support_arms_per_instance.csv", index=False)

    # ---- 2D breakdown ---------------------------------------------------
    def group_of(gen: pd.Series) -> pd.Series:
        return np.where(gen == "grid", "grid",
                        np.where(gen == "line_noise", "line_noise",
                                 np.where(gen == "clustered", "cluster", "others")))

    b2_rows = []
    for tag in arms:
        ok = scored[tag]["bench2d"].copy()
        ok["grp"] = group_of(ok["generator"])
        for grp, g in ok.groupby("grp"):
            b2_rows.append({"model": tag, "group": grp, **_summary(g["err_pct"].to_numpy())})
        b2_rows.append({"model": tag, "group": "ALL", **_summary(ok["err_pct"].to_numpy())})
        for gc, g in ok.dropna(subset=["gen_class"]).groupby("gen_class"):
            b2_rows.append({"model": tag, "group": f"class:{gc}",
                            **_summary(g["err_pct"].to_numpy())})
    B2 = pd.DataFrame(b2_rows)
    B2.to_csv(HERE / "support_arms_bench2d.csv", index=False)

    # ---- gate 6: LineNoise slope ---------------------------------------
    slope_rows = []
    for tag in arms:
        ok = scored[tag]["bench2d"]
        ln = ok[(ok.generator == "line_noise") & (ok.n_customers >= 200)]
        ta = np.clip(ln["true_cost"] / ln["mst_total_length"], *ALPHA_CLIP).to_numpy()
        lr = stats.linregress(ta, ln["pred_alpha"].to_numpy())
        slope_rows.append({"model": tag, "n": int(len(ln)), "slope": float(lr.slope),
                           "intercept": float(lr.intercept), "r": float(lr.rvalue)})
    SL = pd.DataFrame(slope_rows)
    SL.to_csv(HERE / "support_arms_slope.csv", index=False)

    # ---- gates table ----------------------------------------------------
    def b2val(tag, grp, col="mape"):
        r = B2[(B2.model == tag) & (B2.group == grp)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    for tag in arms:
        nd_r = S[(S.model == tag) & (S.stratum == "nd_test")].iloc[0]
        ne_r = S[(S.model == tag) & (S.stratum == "tsplib_noneuc")].iloc[0]
        sig = G[G.model == tag].iloc[0]
        pd_ = P[(P.model == tag) & (P.swept == "dimension")].iloc[0]
        pn_ = P[(P.model == tag) & (P.swept == "n_customers")].iloc[0]
        sl = SL[SL.model == tag].iloc[0]
        fz_ne = S[(S.model == "FROZEN") & (S.stratum == "tsplib_noneuc")].iloc[0]

        g1 = (nd_r["mape"] <= GATE1_MAPE_MAX) and (nd_r["sdpe"] <= GATE1_SDPE_MAX)
        g2 = bool(sig.get("disp_ratio", np.nan) < 1.0
                  and sig.get("disp_p_holm", 1.0) < 0.05)
        g3 = bool(sig.get("mean_gain", -1) > 0 and sig.get("mean_p_holm", 1.0) < 0.05)
        g4 = (pd_["pct_nonincr_deployed"] >= 99.0
              and pn_["pct_nonincr_deployed"] >= 99.0
              and max(pd_["viol_max_deployed"], pn_["viol_max_deployed"]) <= 1e-3)
        g6 = sl["slope"] >= 0.70
        g7 = b2val(tag, "grid", "mspe") <= 4.0
        worst = max(b2val(tag, g) - PROD[f"{g}_mape"]
                    for g in ("grid", "line_noise", "cluster", "others"))
        g8 = worst <= 0.15
        g9 = (ne_r["mape"] <= fz_ne["mape"] + 1e-9) and (ne_r["n"] >= fz_ne["n"])

        gate_rows.append({
            "model": tag,
            "G1_nd": "PASS" if g1 else "FAIL",
            "G1_val": f"MAPE {nd_r['mape']:.4f}/<= {GATE1_MAPE_MAX:.4f}, "
                      f"SDPE {nd_r['sdpe']:.4f}/<= {GATE1_SDPE_MAX:.4f}",
            "G2_disp": "PASS" if g2 else "FAIL",
            "G2_val": f"ratio {sig.get('disp_ratio', float('nan')):.4f}, "
                      f"Holm p {sig.get('disp_p_holm', float('nan')):.4g}, "
                      f"family {sig.get('disp_family_size', 'na')}",
            "G3_mean": "PASS" if g3 else "FAIL",
            "G3_val": f"gain {sig.get('mean_gain', float('nan')):.4f} "
                      f"(MDE {sig.get('mean_mde', float('nan')):.4f}), "
                      f"Holm p {sig.get('mean_p_holm', float('nan')):.4g}, "
                      f"family {sig.get('mean_family_size', 'na')}",
            "G4_mono": "PASS" if g4 else "FAIL",
            "G4_val": f"d {pd_['pct_nonincr_deployed']:.1f}% / n "
                      f"{pn_['pct_nonincr_deployed']:.1f}%, max viol "
                      f"{max(pd_['viol_max_deployed'], pn_['viol_max_deployed']):.2e}",
            "G6_slope": "PASS" if g6 else "FAIL",
            "G6_val": f"{sl['slope']:.4f} (N={sl['n']})",
            "G7_grid": "PASS" if g7 else "FAIL",
            "G7_val": f"MSPE {b2val(tag, 'grid', 'mspe'):+.4f}",
            "G8_2dclass": "PASS" if g8 else "FAIL",
            "G8_val": f"worst delta {worst:+.4f}",
            "G9_noneuc": "PASS" if g9 else "FAIL",
            "G9_val": f"MAPE {ne_r['mape']:.4f} vs frozen {fz_ne['mape']:.4f}, "
                      f"cov {int(ne_r['n'])}/{int(ne_r['n_stratum'])}",
            "bench2d_overall_mape": b2val(tag, "ALL"),
        })
    GA = pd.DataFrame(gate_rows)
    GA.to_csv(HERE / "support_arms_gate_results.csv", index=False)

    # ---- adversarial ----------------------------------------------------
    bench = scored["FROZEN"]["bench2d"]
    dup = adv_near_duplicate(corpus, aug, bench, feats31)
    dup.to_csv(HERE / "support_arms_adv_nearduplicate.csv", index=False)

    recal = []
    for tag in arms:
        if tag == "FROZEN":
            continue
        for gen in ("line_noise", "grid"):
            r = adv_one_constant(scored["FROZEN"]["bench2d"],
                                 scored[tag]["bench2d"], gen)
            r["model"] = tag
            recal.append(r)
    RC = pd.DataFrame(recal)
    RC.to_csv(HERE / "support_arms_adv_recalibration.csv", index=False)

    # ---- correlation ----------------------------------------------------
    sub = corpus[["greedy_nn_over_mst", "mst_topology_straightness"]].dropna()
    pear = float(np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1])
    spear = float(stats.spearmanr(sub.iloc[:, 0], sub.iloc[:, 1]).statistic)
    corr = {"n": int(len(sub)), "pearson": pear, "spearman": spear}
    tr = corpus[corpus.split == "train"]
    lown = tr[tr.n_customers >= 200]
    corr["pearson_n_ge_200"] = float(np.corrcoef(
        lown["greedy_nn_over_mst"], lown["mst_topology_straightness"])[0, 1])
    json.dump(corr, open(HERE / "support_arms_correlation.json", "w"), indent=2)

    # ---- print ----------------------------------------------------------
    pd.set_option("display.width", 250)
    f4 = lambda x: f"{x:.4f}"  # noqa: E731
    print("\n================= STRATA =================")
    print(S.to_string(index=False, float_format=f4))
    print("\n================= 2D BREAKDOWN =================")
    print(B2.to_string(index=False, float_format=f4))
    print("\n================= LINENOISE SLOPE (n>=200) =================")
    print(SL.to_string(index=False, float_format=f4))
    print("\n================= MONOTONICITY =================")
    print(P.to_string(index=False, float_format=f4))
    print("\n================= TSPLIB SIGNIFICANCE =================")
    print(G.to_string(index=False, float_format=f4))
    print("\n================= GATES =================")
    for _, r in GA.iterrows():
        print(f"\n--- {r['model']} --- (2D overall MAPE {r['bench2d_overall_mape']:.4f})")
        for i in (1, 2, 3, 4, 6, 7, 8, 9):
            k = [c for c in GA.columns if c.startswith(f"G{i}_") and not c.endswith("_val")][0]
            print(f"  gate {i}: {r[k]:4s}  {r[k + '_val'] if k + '_val' in r else r[k[:3] + '_val']}")
    print("\n================= ADVERSARIAL: near-duplicate =================")
    print(dup.to_string(index=False, float_format=f4))
    print("\n================= ADVERSARIAL: one-constant recalibration =================")
    print(RC.to_string(index=False, float_format=f4))
    print("\n================= CORRELATION =================")
    print(json.dumps(corr, indent=2))
    print("\nwrote support_arms_{strata,bench2d,slope,probe,significance,"
          "gate_results,per_instance,adv_*}.csv")
