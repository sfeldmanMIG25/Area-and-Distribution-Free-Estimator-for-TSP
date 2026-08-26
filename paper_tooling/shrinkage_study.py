"""Honest out-of-sample test of MST-constant shrinkage for GART.

Question
--------
Does ``alpha_shrunk = w * alpha_model + (1 - w) * rho`` genuinely improve the
estimator out of sample, or is the TSPLIB gain an artefact of having selected
the blend on the test set?

Pre-registered protocol (fixed BEFORE any TSPLIB number is computed)
--------------------------------------------------------------------
1.  Selection corpora (TSPLIB excluded, never trained on):
        - 2D benchmark, 2,580 instances, d = 2
        - augmentation corpus, 874 instances, d in {2,3,5,10,20,50}
    Reference-only corpora:
        - validation split of tsp_features_v3.csv (in-distribution)
2.  Candidate models: LGBM_V3 (shipped GART 2.0) and LGBM_V4.
3.  Shrinkage target ``rho``:
        Variant A ("asymptotic-anchored", the pre-committed one):
            d = 2   -> rho = beta_TSP / beta_MST = 0.7124 / 0.6331 = 1.125257
            d != 2  -> no published MST constant exists, so the train-split
                       calibrated per-dimension constant rho_d is used
                       (calibrated_alpha_table.json, fitted on split='train').
        Variant B ("pure calibrated"): rho = rho_d for every d, including d = 2.
4.  Selection rule: the (model, w) pair minimising MAPE on the instance-pooled
    OOD selection set (2D benchmark + augmentation, 3,454 instances) under
    Variant A. w searched on [0, 1] at 0.001 resolution.
5.  The chosen (model, w) is then applied ONCE to TSPLIB EUC_2D and compared,
    with a paired Wilcoxon test and a 1,000-resample paired bootstrap, against
    exactly two pre-declared baselines: LGBM_V3 alone, and the asymptotic
    constant alone. No further search on TSPLIB.
6.  Fragility checks with the same w: ND test split and TSPLIB non-EUC screened.

Because shrinkage is linear in the prediction, ``cost_shrunk(w) = w * cost_model
+ (1 - w) * rho * L_MST`` is identical to shrinking alpha, so everything is done
in cost space using the shipped per-instance predictions where they exist.

Outputs
-------
paper_tooling/shrinkage_results.csv   tidy long table (w curves + summaries)
stdout                                the report
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RNG_SEED = 20260810
ASYM_RHO_2D = 0.7124 / 0.6331  # 1.1252567...
W_GRID_COARSE = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 4)
W_GRID_FINE = np.round(np.arange(0.0, 1.0 + 1e-9, 0.001), 4)
N_BOOT = 1000

FEATURES_V3 = REPO / "tsp_features_v3.csv"
FEATURES_V4 = REPO / "tsp_features_v4.csv"
AUG_FEATURES = REPO / "paper_tooling" / "augment_features_v3.csv"
AUG_GREEDY_CACHE = REPO / "paper_tooling" / "augment_greedy_nn.csv"
AUG_INSTANCES = REPO / "augment" / "instances"
BENCH_2D = REPO / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
GT_2D = (REPO / "Generalized_TSP_Analysis" / "benchmark_checkpoints"
         / "base_ground_truth_2d.csv")
TSPLIB = REPO / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
SCREENED = REPO / "paper_reference" / "mds_distortion_screened.csv"
CALIB_TABLE = REPO / "calibrated_alpha_table.json"
OUT_CSV = REPO / "paper_tooling" / "shrinkage_results.csv"


# ---------------------------------------------------------------------------
# rho lookups
# ---------------------------------------------------------------------------
_TBL = json.loads(CALIB_TABLE.read_text(encoding="utf-8"))
_RHO_D = {int(k): float(v) for k, v in _TBL["rho_d"].items()}
_TRAINED_DIMS = sorted(_RHO_D)


def rho_calibrated(d) -> float:
    """Train-split calibrated per-dimension constant, nearest-dim fallback."""
    d = int(d)
    if d in _RHO_D:
        return _RHO_D[d]
    return _RHO_D[min(_TRAINED_DIMS, key=lambda t: (abs(t - d), t))]


def rho_variant_a(d) -> float:
    """Asymptotic where published (d=2), calibrated elsewhere."""
    return ASYM_RHO_2D if int(d) == 2 else rho_calibrated(d)


def rho_vec(dims, variant: str) -> np.ndarray:
    f = rho_variant_a if variant == "A" else rho_calibrated
    return np.array([f(d) for d in dims], dtype=float)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def pct_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return 100.0 * (pred - true) / true


def summarise(pe: np.ndarray) -> dict:
    pe = np.asarray(pe, dtype=float)
    bias = float(pe.mean())
    sd = float(pe.std(ddof=1)) if pe.size > 1 else 0.0
    return {
        "n": int(pe.size),
        "MAPE": float(np.abs(pe).mean()),
        "MSPE": bias,
        "SDPE": sd,
        "RMSPE": float(np.sqrt((pe ** 2).mean())),
        "bias_sq": bias ** 2,
        "var": float(pe.var(ddof=1)) if pe.size > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# corpus builders -> DataFrame(instance, dimension, n, true_cost, mst_length,
#                              pred_V3, pred_V4, rho_A, rho_B)
# ---------------------------------------------------------------------------
def build_2d_benchmark() -> pd.DataFrame:
    raw = pd.read_csv(BENCH_2D, usecols=["model", "instance", "pred_cost"])
    piv = raw.pivot_table(index="instance", columns="model", values="pred_cost")
    gt = pd.read_csv(
        GT_2D, usecols=["instance", "n_customers", "true_cost", "mst_length"]
    ).set_index("instance")
    j = piv.join(gt, how="inner")
    out = pd.DataFrame({
        "instance": j.index,
        "dimension": 2,
        "n": j["n_customers"].values,
        "true_cost": j["true_cost"].values,
        "mst_length": j["mst_length"].values,
        "pred_V3": j["LGBM_V3"].values,
        "pred_V4": j["LGBM_V4"].values,
    })
    # Shipped constant rows: Asymptotic_MST is 1.12526 * L_MST; the row labelled
    # MST_Ratio in THIS file is a legacy 1.075 ratio, not the asymptotic one.
    out["pred_rho_A"] = j["Asymptotic_MST"].values
    out["pred_rho_B"] = j["Calibrated_MST_d"].values
    return out


def _augment_greedy_feature() -> pd.Series:
    """greedy_nn_over_mst for the augmentation corpus (cached)."""
    if AUG_GREEDY_CACHE.exists():
        c = pd.read_csv(AUG_GREEDY_CACHE)
        return c.set_index("instance_name")["greedy_nn_over_mst"]

    from concurrent.futures import ThreadPoolExecutor
    from lgbm_model_v4.feature_engineering import _greedy_nn_tour_length

    feats = pd.read_csv(
        AUG_FEATURES, usecols=["instance_name", "mst_total_length"]
    )

    def one(row):
        name, mst_len = row
        with open(AUG_INSTANCES / f"{name}.json", "r") as fh:
            coords = np.asarray(json.load(fh)["coordinates"], dtype=np.float64)
        g = _greedy_nn_tour_length(coords)
        return name, float(g / mst_len) if mst_len > 1e-9 else 1.0

    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, feats.itertuples(index=False, name=None)))

    c = pd.DataFrame(res, columns=["instance_name", "greedy_nn_over_mst"])
    c.to_csv(AUG_GREEDY_CACHE, index=False)
    return c.set_index("instance_name")["greedy_nn_over_mst"]


def build_augment(m3, f3, m4, f4) -> pd.DataFrame:
    df = pd.read_csv(AUG_FEATURES)
    df["greedy_nn_over_mst"] = df["instance_name"].map(_augment_greedy_feature())
    a3 = np.clip(m3.predict(df[f3]), 1.0, 2.0)
    a4 = np.clip(m4.predict(df[f4]), 1.0, 2.0)
    L = df["mst_total_length"].values
    out = pd.DataFrame({
        "instance": df["instance_name"].values,
        "dimension": df["dimension"].values,
        "n": df["n_customers"].values,
        "true_cost": df["optimal_cost"].values,
        "mst_length": L,
        "pred_V3": a3 * L,
        "pred_V4": a4 * L,
    })
    out["pred_rho_A"] = rho_vec(out["dimension"], "A") * L
    out["pred_rho_B"] = rho_vec(out["dimension"], "B") * L
    return out


def build_split(split: str, m3, f3, m4, f4) -> pd.DataFrame:
    d3 = pd.read_csv(FEATURES_V3)
    d3 = d3[d3["split"] == split].reset_index(drop=True)
    d4 = pd.read_csv(FEATURES_V4)
    d4 = d4[d4["split"] == split].set_index("instance_name")
    d4 = d4.loc[d3["instance_name"].values].reset_index()

    L = d3["mst_total_length"].values
    a3 = np.clip(m3.predict(d3[f3]), 1.0, 2.0)
    a4 = np.clip(m4.predict(d4[f4]), 1.0, 2.0)
    out = pd.DataFrame({
        "instance": d3["instance_name"].values,
        "dimension": d3["dimension"].values,
        "n": d3["n_customers"].values,
        "true_cost": d3["optimal_cost"].values,
        "mst_length": L,
        "pred_V3": a3 * L,
        "pred_V4": a4 * L,
    })
    out["pred_rho_A"] = rho_vec(out["dimension"], "A") * L
    out["pred_rho_B"] = rho_vec(out["dimension"], "B") * L
    return out


def build_tsplib(subset: str) -> pd.DataFrame:
    raw = pd.read_csv(TSPLIB)
    if subset == "euc":
        keep = raw[raw["edge_weight_type"] == "EUC_2D"]["instance"].unique()
    else:
        keep = pd.read_csv(SCREENED)["instance"].unique()
    raw = raw[raw["instance"].isin(keep)]

    base = (raw[raw["model"] == "LGBM_V3"]
            .set_index("instance")[["n", "true_cost", "mst_length",
                                    "pred_cost", "feature_dim"]])
    v4 = raw[raw["model"] == "LGBM_V4"].set_index("instance")
    v4_pred = v4["pred_cost"].where(v4["status"] == "ok")

    dims = base["feature_dim"].fillna(2).astype(int).values
    L = base["mst_length"].values
    out = pd.DataFrame({
        "instance": base.index,
        "dimension": dims,
        "n": base["n"].values,
        "true_cost": base["true_cost"].values,
        "mst_length": L,
        "pred_V3": base["pred_cost"].values,
        "pred_V4": v4_pred.reindex(base.index).values,
    })
    out["pred_rho_A"] = rho_vec(dims, "A") * L
    out["pred_rho_B"] = rho_vec(dims, "B") * L
    return out


# ---------------------------------------------------------------------------
# shrinkage machinery
# ---------------------------------------------------------------------------
def shrunk_pe(df: pd.DataFrame, model: str, variant: str, w) -> np.ndarray:
    pm = df[f"pred_{model}"].values
    pr = df[f"pred_rho_{variant}"].values
    return pct_err(w * pm + (1.0 - w) * pr, df["true_cost"].values)


def w_curve(df: pd.DataFrame, model: str, variant: str, grid) -> pd.DataFrame:
    rows = []
    for w in grid:
        s = summarise(shrunk_pe(df, model, variant, w))
        s["w"] = float(w)
        rows.append(s)
    return pd.DataFrame(rows)


def best_w(df: pd.DataFrame, model: str, variant: str) -> tuple:
    mapes = np.array([np.abs(shrunk_pe(df, model, variant, w)).mean()
                      for w in W_GRID_FINE])
    i = int(np.argmin(mapes))
    return float(W_GRID_FINE[i]), float(mapes[i])


def paired_tests(pe_a: np.ndarray, pe_b: np.ndarray, rng) -> dict:
    """Paired comparison of |pe_a| against |pe_b| (a = candidate, b = baseline).

    Negative mean difference means the candidate has smaller absolute error.
    """
    from scipy.stats import wilcoxon

    d = np.abs(pe_a) - np.abs(pe_b)
    if np.allclose(d, 0):
        return {"mean_diff": 0.0, "wilcoxon_p": 1.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "win_rate": 0.5}
    stat, p = wilcoxon(np.abs(pe_a), np.abs(pe_b))
    n = d.size
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = d[idx].mean(axis=1)
    return {
        "mean_diff": float(d.mean()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "win_rate": float((d < 0).mean()),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")
    import joblib
    import lightgbm  # noqa: F401

    rng = np.random.default_rng(RNG_SEED)
    rows_out = []

    def rec(**kw):
        rows_out.append(kw)

    m3 = joblib.load(REPO / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib")
    m3.set_params(verbosity=-1)
    f3 = list(m3.feature_name_)
    m4 = joblib.load(REPO / "lgbm_model_v4" / "lgbm_alpha_model_v4.joblib")
    f4 = list(m4.feature_name())

    print("Loading corpora ...")
    sets = {
        "2D_bench": build_2d_benchmark(),
        "augment": build_augment(m3, f3, m4, f4),
        "val": build_split("val", m3, f3, m4, f4),
        "nd_test": build_split("test", m3, f3, m4, f4),
    }
    sets["pooled_OOD"] = pd.concat(
        [sets["2D_bench"], sets["augment"]], ignore_index=True
    )
    for k, v in sets.items():
        print(f"  {k:12s} n={len(v):6d}  dims={sorted(v['dimension'].unique())[:8]}")

    # ---------------- Section 1: baseline behaviour ------------------------
    print("\n" + "=" * 78)
    print("1. BASELINE BEHAVIOUR ON SELECTION / REFERENCE CORPORA (no TSPLIB)")
    print("=" * 78)
    hdr = f"{'set':12s} {'estimator':22s} {'n':>6s} {'MAPE':>8s} {'MSPE':>8s} {'SDPE':>8s}"
    print(hdr)
    for name, df in sets.items():
        for est, col in [("LGBM_V3", "pred_V3"), ("LGBM_V4", "pred_V4"),
                         ("rho variant A", "pred_rho_A"),
                         ("rho variant B (calib)", "pred_rho_B")]:
            pe = pct_err(df[col].values, df["true_cost"].values)
            pe = pe[np.isfinite(pe)]
            s = summarise(pe)
            print(f"{name:12s} {est:22s} {s['n']:6d} {s['MAPE']:8.3f} "
                  f"{s['MSPE']:+8.3f} {s['SDPE']:8.3f}")
            rec(section="baseline", set=name, estimator=est, w=np.nan,
                variant="", **s)
        print()

    # ---------------- Section 2: w curves ----------------------------------
    print("=" * 78)
    print("2. MAPE AS A FUNCTION OF w  (w=1 -> pure model, w=0 -> pure rho)")
    print("=" * 78)
    curve_sets = ["2D_bench", "augment", "val", "pooled_OOD"]
    for variant in ("A", "B"):
        vlabel = ("A: asymptotic@d=2 + calibrated@d!=2" if variant == "A"
                  else "B: calibrated rho_d everywhere")
        for model in ("V3", "V4"):
            print(f"\n--- target {vlabel} | model LGBM_{model} ---")
            print("  w   " + "".join(f"{s:>12s}" for s in curve_sets))
            curves = {s: w_curve(sets[s], model, variant, W_GRID_COARSE)
                      for s in curve_sets}
            for i, w in enumerate(W_GRID_COARSE):
                line = f"{w:5.2f}"
                for s in curve_sets:
                    line += f"{curves[s]['MAPE'].iloc[i]:12.4f}"
                print(line)
            for s in curve_sets:
                for _, r in curves[s].iterrows():
                    rec(section="w_curve", set=s, estimator=f"LGBM_{model}",
                        variant=variant, w=r["w"], n=int(r["n"]),
                        MAPE=r["MAPE"], MSPE=r["MSPE"], SDPE=r["SDPE"],
                        RMSPE=r["RMSPE"], bias_sq=r["bias_sq"], var=r["var"])
            print("  optimal w (0.001 grid):")
            for s in curve_sets + ["nd_test"]:
                bw, bm = best_w(sets[s], model, variant)
                m1 = np.abs(shrunk_pe(sets[s], model, variant, 1.0)).mean()
                print(f"    {s:12s} w*={bw:.3f}  MAPE*={bm:.4f}  "
                      f"MAPE(w=1)={m1:.4f}  gain={m1 - bm:+.4f}")
                rec(section="w_star", set=s, estimator=f"LGBM_{model}",
                    variant=variant, w=bw, n=len(sets[s]), MAPE=bm,
                    MAPE_at_w1=m1, gain=m1 - bm)

    # ---------------- Section 3: the pre-committed choice ------------------
    print("\n" + "=" * 78)
    print("3. PRE-COMMITTED SELECTION (pooled OOD = 2D bench + augment, variant A)")
    print("=" * 78)
    cands = {}
    for model in ("V3", "V4"):
        bw, bm = best_w(sets["pooled_OOD"], model, "A")
        cands[model] = (bw, bm)
        print(f"  LGBM_{model}: w*={bw:.3f}  pooled-OOD MAPE*={bm:.4f}")
    chosen_model = min(cands, key=lambda k: cands[k][1])
    chosen_w = cands[chosen_model][0]
    print(f"  -> CHOSEN: model=LGBM_{chosen_model}, w={chosen_w:.3f}, target=variant A")
    rec(section="selection", set="pooled_OOD", estimator=f"LGBM_{chosen_model}",
        variant="A", w=chosen_w, MAPE=cands[chosen_model][1])

    # corpus-balanced sensitivity (each corpus weighted equally)
    def bal_mape(model, w):
        return np.mean([np.abs(shrunk_pe(sets[s], model, "A", w)).mean()
                        for s in ("2D_bench", "augment")])
    bal = {m: W_GRID_FINE[int(np.argmin([bal_mape(m, w) for w in W_GRID_FINE]))]
           for m in ("V3", "V4")}
    print(f"  sensitivity, corpus-balanced pooling: w* = "
          + ", ".join(f"LGBM_{k}:{v:.3f}" for k, v in bal.items()))
    for k, v in bal.items():
        rec(section="selection_sensitivity", set="corpus_balanced_OOD",
            estimator=f"LGBM_{k}", variant="A", w=float(v))

    # ---------------- Section 4: single TSPLIB evaluation ------------------
    print("\n" + "=" * 78)
    print("4. SINGLE PRE-COMMITTED TSPLIB EUC_2D EVALUATION")
    print("=" * 78)
    tl = build_tsplib("euc")
    pe_shrunk = shrunk_pe(tl, chosen_model, "A", chosen_w)
    pe_v3 = pct_err(tl["pred_V3"].values, tl["true_cost"].values)
    pe_rho = pct_err(tl["pred_rho_A"].values, tl["true_cost"].values)

    named = [(f"shrunk (LGBM_{chosen_model}, w={chosen_w:.3f})", pe_shrunk),
             ("LGBM_V3 alone", pe_v3),
             ("asymptotic constant alone", pe_rho)]
    print(f"{'estimator':42s} {'n':>4s} {'MAPE':>8s} {'MSPE':>8s} {'SDPE':>8s} {'RMSPE':>8s}")
    for label, pe in named:
        s = summarise(pe)
        print(f"{label:42s} {s['n']:4d} {s['MAPE']:8.3f} {s['MSPE']:+8.3f} "
              f"{s['SDPE']:8.3f} {s['RMSPE']:8.3f}")
        rec(section="tsplib_euc", set="tsplib_euc2d", estimator=label,
            variant="A", w=chosen_w if "shrunk" in label else np.nan, **s)

    print("\n  Paired tests, |pe(shrunk)| - |pe(baseline)| (negative favours shrunk):")
    for blabel, pe_b in [("LGBM_V3 alone", pe_v3),
                         ("asymptotic constant alone", pe_rho)]:
        t = paired_tests(pe_shrunk, pe_b, rng)
        print(f"    vs {blabel:28s} mean_diff={t['mean_diff']:+.4f} pp  "
              f"95% CI [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}]  "
              f"Wilcoxon p={t['wilcoxon_p']:.4g}  win_rate={t['win_rate']:.3f}")
        rec(section="tsplib_paired_test", set="tsplib_euc2d",
            estimator=f"shrunk_vs_{blabel}", variant="A", w=chosen_w, **t)

    # ---- reference only: the oracle / selected-on-test picture -------------
    print("\n  [REFERENCE ONLY - selected ON the test set, therefore INVALID "
          "as evidence]")
    for model in ("V3", "V4"):
        tw, tm = best_w(tl, model, "A")
        m1 = np.abs(shrunk_pe(tl, model, "A", 1.0)).mean()
        m50 = np.abs(shrunk_pe(tl, model, "A", 0.5)).mean()
        print(f"    LGBM_{model}: TSPLIB-oracle w={tw:.3f} -> MAPE {tm:.4f} ; "
              f"w=1 -> {m1:.4f} ; w=0.5 (naive average) -> {m50:.4f}")
        rec(section="tsplib_oracle_reference", set="tsplib_euc2d",
            estimator=f"LGBM_{model}", variant="A", w=tw, MAPE=tm,
            MAPE_at_w1=m1, MAPE_at_w050=m50)
        t = paired_tests(shrunk_pe(tl, model, "A", 0.5), pe_v3, rng)
        print(f"      w=0.5 blend vs LGBM_V3 alone: mean_diff="
              f"{t['mean_diff']:+.4f} pp, Wilcoxon p={t['wilcoxon_p']:.4g} "
              f"(cherry-picked, do not cite)")
        rec(section="tsplib_oracle_reference_test", set="tsplib_euc2d",
            estimator=f"LGBM_{model}_w0.5_vs_V3", variant="A", w=0.5, **t)

    print("\n  TSPLIB EUC_2D w curve (post-hoc, for contrast with the OOD "
          "curves above):")
    print("  w   " + "".join(f"{'LGBM_' + m:>12s}" for m in ("V3", "V4")))
    for w in W_GRID_COARSE:
        line = f"{w:5.2f}"
        for model in ("V3", "V4"):
            line += f"{np.abs(shrunk_pe(tl, model, 'A', w)).mean():12.4f}"
        print(line)
        for model in ("V3", "V4"):
            s = summarise(shrunk_pe(tl, model, "A", w))
            rec(section="w_curve_posthoc", set="tsplib_euc2d",
                estimator=f"LGBM_{model}", variant="A", w=float(w), **s)

    # ---- diagnostic: is the OOD optimum actually beyond w = 1? -------------
    print("\n  [diagnostic, outside the pre-committed [0,1] grid] extended "
          "search w in [0, 1.5]:")
    ext = np.round(np.arange(0.0, 1.5 + 1e-9, 0.001), 4)
    for sname in ("2D_bench", "augment", "pooled_OOD", "val", "nd_test"):
        df = sets[sname]
        mp = np.array([np.abs(shrunk_pe(df, chosen_model, "A", w)).mean()
                       for w in ext])
        i = int(np.argmin(mp))
        print(f"    {sname:12s} w*_ext={ext[i]:.3f}  MAPE={mp[i]:.4f}")
        rec(section="w_star_extended", set=sname,
            estimator=f"LGBM_{chosen_model}", variant="A", w=float(ext[i]),
            MAPE=float(mp[i]))
    mpt = np.array([np.abs(shrunk_pe(tl, chosen_model, "A", w)).mean()
                    for w in ext])
    it = int(np.argmin(mpt))
    print(f"    {'tsplib_euc2d':12s} w*_ext={ext[it]:.3f}  MAPE={mpt[it]:.4f}"
          "   <- the only corpus with an interior optimum")
    rec(section="w_star_extended", set="tsplib_euc2d",
        estimator=f"LGBM_{chosen_model}", variant="A", w=float(ext[it]),
        MAPE=float(mpt[it]))

    # ---------------- Section 5: bias / variance ---------------------------
    print("\n" + "=" * 78)
    print("5. BIAS vs VARIANCE DECOMPOSITION  (RMSPE^2 = MSPE^2 + Var(pe))")
    print("=" * 78)
    print(f"{'set':14s} {'estimator':30s} {'MSPE^2':>10s} {'Var':>10s} "
          f"{'RMSPE^2':>10s} {'bias share':>11s}")
    bv_sets = [("tsplib_euc2d", tl), ("2D_bench", sets["2D_bench"]),
               ("augment", sets["augment"]), ("nd_test", sets["nd_test"])]
    for sname, df in bv_sets:
        trio = [("model LGBM_V3",
                 pct_err(df["pred_V3"].values, df["true_cost"].values)),
                ("model LGBM_" + chosen_model,
                 pct_err(df[f"pred_{chosen_model}"].values, df["true_cost"].values)),
                ("rho constant (variant A)",
                 pct_err(df["pred_rho_A"].values, df["true_cost"].values)),
                (f"shrunk LGBM_V3 w=0.5",
                 shrunk_pe(df, "V3", "A", 0.5)),
                (f"shrunk chosen w={chosen_w:.3f}",
                 shrunk_pe(df, chosen_model, "A", chosen_w))]
        for label, pe in trio:
            s = summarise(pe)
            share = s["bias_sq"] / (s["RMSPE"] ** 2) if s["RMSPE"] > 0 else 0.0
            print(f"{sname:14s} {label:30s} {s['bias_sq']:10.3f} {s['var']:10.3f} "
                  f"{s['RMSPE'] ** 2:10.3f} {share:11.3f}")
            rec(section="bias_variance", set=sname, estimator=label,
                variant="A", w=chosen_w if "shrunk" in label else np.nan,
                bias_share=share, **s)
        print()

    # ---------------- Section 6: fragility ---------------------------------
    print("=" * 78)
    print("6. FRAGILITY: SAME w APPLIED TO OTHER DOMAINS")
    print("=" * 78)
    non_euc = build_tsplib("screened")
    n_v4_ok = int(np.isfinite(non_euc["pred_V4"].values).sum())
    print(f"  note: LGBM_V4 produces a prediction for only {n_v4_ok}/"
          f"{len(non_euc)} non-EUC screened instances (hybrid-MDS path raises "
          "KeyError), so that set is reported with LGBM_V3.\n")

    frag = [("nd_test (16,920)", sets["nd_test"], None),
            ("val (19,584)", sets["val"], None),
            ("2D_bench (2,580)", sets["2D_bench"], None),
            ("augment (874)", sets["augment"], None),
            ("tsplib_euc2d (78)", tl, None),
            ("tsplib_non_euc_screened (23)", non_euc, "V3")]
    print(f"{'set':30s} {'model':6s} {'MAPE(w=1)':>11s} {'MAPE(w*)':>11s} "
          f"{'delta':>9s} {'MSPE(w=1)':>11s} {'MSPE(w*)':>11s}")
    # Both the pre-committed choice and the shipped V3 are reported, because
    # the headline TSPLIB claim was made about V3.
    for label, df, force in frag:
        models = [force] if force else sorted({chosen_model, "V3"})
        for model in models:
            pe1 = shrunk_pe(df, model, "A", 1.0)
            pew = shrunk_pe(df, model, "A", chosen_w)
            # also report the TSPLIB-oracle w, to show what adopting it costs
            s1, sw = summarise(pe1), summarise(pew)
            print(f"{label:30s} {model:6s} {s1['MAPE']:11.4f} "
                  f"{sw['MAPE']:11.4f} {sw['MAPE'] - s1['MAPE']:+9.4f} "
                  f"{s1['MSPE']:+11.4f} {sw['MSPE']:+11.4f}")
            rec(section="fragility", set=label, estimator=f"LGBM_{model}",
                variant="A", w=chosen_w, MAPE=sw["MAPE"],
                MAPE_at_w1=s1["MAPE"], gain=s1["MAPE"] - sw["MAPE"],
                MSPE=sw["MSPE"], SDPE=sw["SDPE"], n=s1["n"])

    print("\n  Counterfactual: what if the TSPLIB-selected blend had been "
          "shipped anyway? (w = 0.5, the naive average)")
    print(f"{'set':30s} {'model':6s} {'MAPE(w=1)':>11s} {'MAPE(w=0.5)':>12s} "
          f"{'delta':>9s} {'MSPE(w=0.5)':>12s}")
    for label, df, force in frag:
        models = [force] if force else sorted({chosen_model, "V3"})
        for model in models:
            s1 = summarise(shrunk_pe(df, model, "A", 1.0))
            s5 = summarise(shrunk_pe(df, model, "A", 0.5))
            print(f"{label:30s} {model:6s} {s1['MAPE']:11.4f} "
                  f"{s5['MAPE']:12.4f} {s5['MAPE'] - s1['MAPE']:+9.4f} "
                  f"{s5['MSPE']:+12.4f}")
            rec(section="counterfactual_w050", set=label,
                estimator=f"LGBM_{model}", variant="A", w=0.5,
                MAPE=s5["MAPE"], MAPE_at_w1=s1["MAPE"],
                gain=s1["MAPE"] - s5["MAPE"], MSPE=s5["MSPE"],
                SDPE=s5["SDPE"], n=s5["n"])

    # ---------------- Section 7: calibrated vs asymptotic target -----------
    print("\n" + "=" * 78)
    print("7. SHRINKING TOWARD THE CALIBRATED CONSTANT INSTEAD (variant B)")
    print("=" * 78)
    print(f"{'set':22s} {'model':6s} {'w*(A)':>8s} {'MAPE*(A)':>10s} "
          f"{'w*(B)':>8s} {'MAPE*(B)':>10s} {'MAPE(w=1)':>10s} "
          f"{'MAPE(A,w=.5)':>13s} {'MAPE(B,w=.5)':>13s}")
    all_sets = list(sets.items()) + [("tsplib_euc2d", tl),
                                     ("tsplib_non_euc", non_euc)]
    for sname, df in all_sets:
        models = ["V3"] if sname == "tsplib_non_euc" else sorted(
            {chosen_model, "V3"})
        for model in models:
            wa, ma = best_w(df, model, "A")
            wb, mb = best_w(df, model, "B")
            m1 = np.abs(shrunk_pe(df, model, "A", 1.0)).mean()
            a5 = np.abs(shrunk_pe(df, model, "A", 0.5)).mean()
            b5 = np.abs(shrunk_pe(df, model, "B", 0.5)).mean()
            print(f"{sname:22s} {model:6s} {wa:8.3f} {ma:10.4f} {wb:8.3f} "
                  f"{mb:10.4f} {m1:10.4f} {a5:13.4f} {b5:13.4f}")
            rec(section="target_comparison", set=sname,
                estimator=f"LGBM_{model}", variant="A_vs_B", w=wa,
                MAPE=ma, w_B=wb, MAPE_B=mb, MAPE_at_w1=m1,
                MAPE_A_w050=a5, MAPE_B_w050=b5, n=len(df))

    pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
