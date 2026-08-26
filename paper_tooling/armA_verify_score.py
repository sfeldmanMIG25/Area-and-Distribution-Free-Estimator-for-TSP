"""Independent scoring of FROZEN / R0 / A against the reported arm-A numbers.

Reads only the feature tables this verification produced
(``armA_verify_feats_*.csv``) plus raw label sources. Treats the joblib files
as opaque predictors. MAPE / SDPE / MSPE are reimplemented from definition.

Also runs the transform ablation: the same pipeline with the inverse-logit step
removed, to prove the gate is not vacuous.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
for _p in (ROOT, HERE, ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FEATS31 = [
    "n_customers", "dimension", "log_bounding_hypervolume", "bounding_hypervolume",
    "log_node_density", "node_density", "aspect_ratio", "centroid_dist_mean",
    "centroid_dist_std", "centroid_dist_max", "centroid_dist_iqr", "mst_edge_mean",
    "mst_edge_std", "mst_edge_skew", "mst_edge_kurtosis", "mst_edge_max",
    "mst_edge_q10", "mst_edge_q25", "mst_edge_q50", "mst_edge_q75", "mst_edge_q90",
    "mst_dominance_ratio", "mst_gap_ratio", "mst_leaf_ratio", "mst_degree_mean",
    "mst_degree_std", "mst_degree_max", "mst_diameter", "mst_diameter_normalized",
    "large_edge_count", "greedy_nn_over_mst",
]
CLIP = (1.0, 2.0)

MODELS = {
    "FROZEN": ROOT / "lgbm_model_v3" / "gart2_final.joblib",
    "R0": HERE / "support_arms_models" / "R0.joblib",
    "A": HERE / "support_arms_models" / "A.joblib",
}

REPORTED = {
    ("augment", "FROZEN"): (11.980, 14.297), ("augment", "A"): (0.627, 1.133),
    ("bench2d", "FROZEN"): (2.904, 4.686), ("bench2d", "A"): (2.053, 3.172),
    ("nd_test", "FROZEN"): (0.620, 0.988), ("nd_test", "A"): (0.597, 0.984),
    ("tsplib_euc2d", "FROZEN"): (2.556, 2.955), ("tsplib_euc2d", "A"): (2.484, 3.242),
    ("tsplib_noneuc", "FROZEN"): (3.346, 3.897), ("tsplib_noneuc", "A"): (2.431, 3.001),
}


# =========================================================================
# metrics, from definition
# =========================================================================
def summary(err_pct: np.ndarray) -> dict:
    e = np.asarray(err_pct, dtype=np.float64)
    return {"n": int(e.size),
            "mape": float(np.mean(np.abs(e))),
            "sdpe": float(np.std(e, ddof=1)) if e.size > 1 else float("nan"),
            "mspe": float(np.mean(e))}


def alpha_correct(z) -> np.ndarray:
    """alpha = 1 + sigmoid(z): the documented inverse of z = logit(alpha-1)."""
    z = np.asarray(z, dtype=np.float64)
    return 1.0 + 1.0 / (1.0 + np.exp(-z))


def alpha_naive(z) -> np.ndarray:
    """The WRONG pipeline: treat the raw booster output as alpha directly."""
    return np.asarray(z, dtype=np.float64)


# =========================================================================
# labels, from raw sources
# =========================================================================
def labels_nd_test(names) -> pd.Series:
    out = {}
    d = ROOT / "solutions"
    for nm in names:
        j = json.loads((d / f"{nm}.sol.json").read_text(encoding="utf-8"))
        out[nm] = float(j["optimal_cost"])
    return pd.Series(out, name="true_cost")


def labels_bench2d(names) -> pd.Series:
    out = {}
    d = ROOT / "Generalized_TSP_Analysis" / "solutions"
    for nm in names:
        j = json.loads((d / f"{nm}.sol.json").read_text(encoding="utf-8"))
        out[nm] = float(j["optimal_cost"])
    return pd.Series(out, name="true_cost")


def labels_augment() -> pd.Series:
    out = {}
    for p in ("pilot_records.json", "pilot_records_v2.json", "full_records.json",
              "batch2_records.json", "repair_records.json"):
        f = ROOT / "augment" / p
        if not f.exists():
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r.get("optimal_cost") is not None and r.get("written"):
                out[str(r["name"])] = float(r["optimal_cost"])
    return pd.Series(out, name="true_cost")


def labels_tsplib() -> pd.Series:
    df = pd.read_csv(ROOT / "tsplib_benchmark" / "ground_truth" / "optima.csv")
    return pd.Series(df["optimum"].astype(float).to_numpy(),
                     index=df["instance"].astype(str), name="true_cost")


def bench2d_generator() -> pd.Series:
    import glob
    out = {}
    for f in glob.glob(str(ROOT / "Generalized_TSP_Analysis" / "instances" / "*.json")):
        j = json.loads(Path(f).read_text(encoding="utf-8"))
        out[str(j["instance_name"])] = str(j.get("distribution_type", "unknown"))
    return pd.Series(out, name="generator")


# =========================================================================
# frame assembly
# =========================================================================
def build_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    # ---- nd_test --------------------------------------------------------
    nd = pd.read_csv(HERE / "armA_verify_feats_nd_test.csv")
    nd = nd[nd.status == "ok"].copy()
    nd["true_cost"] = nd["instance"].map(labels_nd_test(nd["instance"].tolist()))
    frames["nd_test"] = nd

    # ---- bench2d --------------------------------------------------------
    b2 = pd.read_csv(HERE / "armA_verify_feats_bench2d.csv")
    b2 = b2[b2.status == "ok"].copy()
    b2["true_cost"] = b2["instance"].map(labels_bench2d(b2["instance"].tolist()))
    b2["generator"] = b2["instance"].map(bench2d_generator())
    frames["bench2d"] = b2

    # ---- augment --------------------------------------------------------
    ag = pd.read_csv(HERE / "armA_verify_feats_augment.csv")
    ag = ag[ag.status == "ok"].copy()
    lab = labels_augment()
    ag["true_cost"] = ag["instance"].map(lab)
    n_all = len(ag)
    ag = ag.dropna(subset=["true_cost"]).copy()
    print(f"[augment] {n_all} extracted, {len(ag)} with a solved optimum")
    frames["augment"] = ag

    # ---- tsplib ---------------------------------------------------------
    from tsplib_benchmark.exclusions import filter_metric_consistent

    tl = pd.read_csv(HERE / "armA_verify_feats_tsplib.csv")
    tl["true_cost"] = tl["instance"].map(labels_tsplib())
    scr = filter_metric_consistent(
        tl.rename(columns={"mst_total_length": "mst_length"}),
        true_col="true_cost", mst_col="mst_length", instance_col="instance")
    keep = set(scr["instance"])
    tl = tl[tl["instance"].isin(keep)].copy()
    euc = tl[tl.edge_weight_type == "EUC_2D"].copy()
    non = tl[tl.edge_weight_type != "EUC_2D"].copy()
    print(f"[tsplib] screened -> euc2d {len(euc)}, noneuc {len(non)} "
          f"(noneuc status!=ok: {int((non.status != 'ok').sum())})")
    frames["tsplib_euc2d"] = euc[euc.status == "ok"].copy()
    frames["tsplib_noneuc"] = non[non.status == "ok"].copy()
    return frames


def score(model, frame: pd.DataFrame, transform) -> pd.DataFrame:
    ok = frame[frame[FEATS31].notna().all(axis=1) & frame.true_cost.notna()].copy()
    z = model.predict(ok[FEATS31], num_iteration=model.best_iteration)
    a = np.clip(transform(z), *CLIP)
    ok["pred_alpha"] = a
    ok["pred_cost"] = a * ok["mst_total_length"].to_numpy()
    ok["err_pct"] = (ok["pred_cost"] - ok["true_cost"]) / ok["true_cost"] * 100.0
    return ok


# =========================================================================
def main() -> None:
    import joblib

    frames = build_frames()
    boosters = {k: joblib.load(v) for k, v in MODELS.items()}

    rows, per_inst = [], {}
    for tag, b in boosters.items():
        for st, fr in frames.items():
            ok = score(b, fr, alpha_correct)
            per_inst[(tag, st)] = ok
            s = summary(ok["err_pct"].to_numpy())
            rep = REPORTED.get((st, tag))
            rows.append({
                "model": tag, "stratum": st, **s,
                "rep_mape": rep[0] if rep else np.nan,
                "rep_sdpe": rep[1] if rep else np.nan,
                "d_mape": abs(s["mape"] - rep[0]) if rep else np.nan,
                "d_sdpe": abs(s["sdpe"] - rep[1]) if rep else np.nan,
            })
    S = pd.DataFrame(rows)
    S.to_csv(HERE / "armA_verify_strata.csv", index=False)
    print("\n=========== STRATA (independent recomputation) ===========")
    print(S.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- transform ablation --------------------------------------------
    abl = []
    for tag in ("FROZEN", "A"):
        for st in ("nd_test", "bench2d", "augment"):
            ok = score(boosters[tag], frames[st], alpha_naive)
            abl.append({"model": tag, "stratum": st, "pipeline": "NO_INVERSE",
                        **summary(ok["err_pct"].to_numpy())})
            ok2 = per_inst[(tag, st)]
            abl.append({"model": tag, "stratum": st, "pipeline": "correct",
                        **summary(ok2["err_pct"].to_numpy())})
    AB = pd.DataFrame(abl).sort_values(["model", "stratum", "pipeline"])
    AB.to_csv(HERE / "armA_verify_transform_ablation.csv", index=False)
    print("\n=========== TRANSFORM ABLATION ===========")
    print(AB.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- raw z range, to show why omitting the inverse is seductive -----
    z = boosters["FROZEN"].predict(frames["nd_test"][FEATS31],
                                   num_iteration=boosters["FROZEN"].best_iteration)
    print(f"\nraw z on nd_test: min {z.min():.4f} p50 {np.median(z):.4f} "
          f"max {z.max():.4f};  1+sigmoid(z): min {alpha_correct(z).min():.4f} "
          f"max {alpha_correct(z).max():.4f}")

    # ---- 2D breakdown + gate 6 slope + grid MSPE ------------------------
    def grp(g):
        return np.where(g == "grid", "grid",
                        np.where(g == "line_noise", "line_noise",
                                 np.where(g == "clustered", "cluster", "others")))

    b2rows, slrows = [], []
    for tag in boosters:
        ok = per_inst[(tag, "bench2d")].copy()
        ok["grp"] = grp(ok["generator"])
        for g, sub in ok.groupby("grp"):
            b2rows.append({"model": tag, "group": g, **summary(sub["err_pct"].to_numpy())})
        b2rows.append({"model": tag, "group": "ALL", **summary(ok["err_pct"].to_numpy())})
        ln = ok[(ok.generator == "line_noise") & (ok.n_customers >= 200)]
        ta = np.clip(ln["true_cost"] / ln["mst_total_length"], *CLIP).to_numpy()
        lr = stats.linregress(ta, ln["pred_alpha"].to_numpy())
        slrows.append({"model": tag, "n": int(len(ln)), "slope": float(lr.slope),
                       "intercept": float(lr.intercept), "r": float(lr.rvalue)})
    B2 = pd.DataFrame(b2rows)
    SL = pd.DataFrame(slrows)
    B2.to_csv(HERE / "armA_verify_bench2d.csv", index=False)
    SL.to_csv(HERE / "armA_verify_slope.csv", index=False)
    print("\n=========== 2D BREAKDOWN ===========")
    print(B2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=========== LINE_NOISE SLOPE (n>=200) ===========")
    print(SL.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- TSPLIB significance via the shipped harness --------------------
    import ood_harness as oh
    sig = []
    for tag in boosters:
        allp = {}
        for st in frames:
            ok = per_inst[(tag, st)]
            allp.update(dict(zip(ok["instance"].astype(str),
                                 ok["pred_cost"].astype(float))))
        r = {"model": tag}
        try:
            d = oh.dispersion_verdict({tag: allp}, stratum="tsplib_euc2d")
            row = d[d.model_b == "Asymptotic_MST"]
            if len(row):
                r.update({"disp_ratio": float(row.iloc[0]["sd_ratio"]),
                          "disp_p_holm": float(row.iloc[0]["p_holm"]),
                          "disp_n": int(row.iloc[0]["n_pairs"])})
        except Exception as e:  # noqa: BLE001
            r["disp_error"] = f"{type(e).__name__}: {e}"
        try:
            v = oh.evaluate_candidate(allp, tag)
            c = v.comparisons
            c = c[(c.stratum == "tsplib_euc2d") & (c.model_b == "Calibrated_MST_dn")]
            if len(c):
                r.update({"mean_gain": -float(c.iloc[0]["mean_diff"]),
                          "mean_p_holm": float(c.iloc[0]["p_holm"]),
                          "mean_n": int(c.iloc[0]["n_pairs"]),
                          "mape_model": float(c.iloc[0]["mape_a"])})
        except Exception as e:  # noqa: BLE001
            r["mean_error"] = f"{type(e).__name__}: {e}"
        sig.append(r)
    G = pd.DataFrame(sig)
    G.to_csv(HERE / "armA_verify_significance.csv", index=False)
    print("\n=========== TSPLIB SIGNIFICANCE ===========")
    print(G.to_string(index=False))

    pd.concat([v.assign(model=k[0], stratum=k[1])
               for k, v in per_inst.items()], ignore_index=True)[
        ["model", "stratum", "instance", "pred_alpha", "pred_cost", "true_cost",
         "err_pct"]].to_csv(HERE / "armA_verify_per_instance.csv", index=False)
    print("\nwrote armA_verify_{strata,bench2d,slope,significance,"
          "transform_ablation,per_instance}.csv")


if __name__ == "__main__":
    main()
