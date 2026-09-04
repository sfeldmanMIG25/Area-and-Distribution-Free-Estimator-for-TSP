"""SHAP explanation of GART 2.0 on the 2,580-instance 2D diverse benchmark,
broken out by generator class and sub-generator -- built to explain WHY the
Line Noise (near-collinear) class is where the model errs most.

Computes, on the frozen booster (``lgbm_model_v3/gart2_final.joblib``), in the
logit(alpha - 1) space it actually predicts (the booster's raw margin output,
since the sigmoid is applied outside the model, not inside it):

1. Per class and per sub-generator: mean |SHAP| per feature, ranked, with each
   feature's share of that group's total mean |SHAP|.
2. Per class: mean predicted alpha, mean realized (true) alpha, and -- for the
   features that matter most on Isotropic -- the mean signed SHAP on Line
   Noise vs on Isotropic.
3. For the features that matter most on the rest of the benchmark (all
   classes except Line Noise), the fraction of Line Noise rows whose feature
   value falls outside the training split's [1st, 99th] percentile band.

Outputs:
  paper_tooling/shap_2d_by_class.csv   -- tidy: class, generator, feature,
                                           mean_abs_shap, mean_signed_shap,
                                           share (pct of group total |SHAP|)
  paper_tooling/shap_2d_summary.json   -- per-class alphas, top-5 lists,
                                           Line-Noise-vs-Isotropic signed SHAP,
                                           out-of-training-support fractions

Never reads a CSV/JSON instance file into the LLM's context -- this script is
the only thing that touches them; it prints small aggregate numbers.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
LGBM_V3_DIR = os.path.join(ROOT, "lgbm_model_v3")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Bare (not package-qualified) import, matching lgbm_estimator_gart2.py's own
# sys.path setup -- the numba cache for greedy_nn_tour_length's JIT kernel is
# keyed to the bare module name "feature_engineering_gart2" and misses (with
# a ModuleNotFoundError on reload) if imported as "lgbm_model_v3.feature_engineering_gart2".
if LGBM_V3_DIR not in sys.path:
    sys.path.insert(0, LGBM_V3_DIR)

import joblib  # noqa: E402
import shap  # noqa: E402

from feature_engineering_gart2 import compute_features  # noqa: E402

GT_2D = os.path.join(ROOT, "Generalized_TSP_Analysis", "benchmark_checkpoints",
                      "base_ground_truth_2d.csv")
MODEL_PATH = os.path.join(ROOT, "lgbm_model_v3", "gart2_final.joblib")
TRAIN_CSV = os.path.join(ROOT, "tsp_features_v4.csv")
OUT_CSV = os.path.join(THIS_DIR, "shap_2d_by_class.csv")
OUT_JSON = os.path.join(THIS_DIR, "shap_2d_summary.json")

GEN_RE = re.compile(r"^TSP-([a-z_0-9]+)-n(\d+)-g(\d+)")

# Same six reporting buckets as paper_tooling/build_paper_tables.py's
# GEN_CLASSES: the five-class design taxonomy with Geometric split into its
# unrepresented member (grid) and its represented members (boundary,
# x_central), because that split is exactly what this analysis needs for
# "per sub-generator for Line Noise and grid".
GEN_CLASSES: dict[str, frozenset[str]] = {
    "Isotropic": frozenset({"random", "normal", "triangular", "truncated_exponential"}),
    "Biased": frozenset({"squeezed_uniform", "uniform_triangular", "triangular_squeezed", "correlated"}),
    "GeometricGrid": frozenset({"grid"}),
    "GeometricOther": frozenset({"boundary", "x_central"}),
    "Clustered": frozenset({"clustered"}),
    "LineNoise": frozenset({"line_noise"}),
}
EXPECTED_COUNTS = {"Isotropic": 840, "Biased": 840, "GeometricGrid": 210,
                    "GeometricOther": 420, "Clustered": 60, "LineNoise": 210}

TOP_K = 5


def load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(GT_2D)
    gen = gt["instance"].str.extract(GEN_RE)
    if gen[0].isna().any():
        raise RuntimeError("2D: instance names do not match the generator grammar")
    gt["generator"] = gen[0]
    gt["gen_class"] = pd.NA
    for cls, members in GEN_CLASSES.items():
        gt.loc[gt["generator"].isin(members), "gen_class"] = cls
    if gt["gen_class"].isna().any():
        bad = sorted(set(gt.loc[gt["gen_class"].isna(), "generator"]))
        raise RuntimeError(f"unmapped generators: {bad}")
    counts = gt.drop_duplicates("instance").groupby("gen_class").size().to_dict()
    for cls, expect in EXPECTED_COUNTS.items():
        if counts.get(cls, 0) != expect:
            raise RuntimeError(f"class {cls}: expected {expect} instances, got {counts.get(cls, 0)}")
    return gt


def build_feature_matrix(gt: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    rows = []
    n = len(gt)
    for i, rec in enumerate(gt.itertuples(index=False)):
        with open(rec.file_path, "r", encoding="utf-8") as f:
            inst = json.load(f)
        coords = np.asarray(inst["coordinates"], dtype=np.float32)
        feats = compute_features(coords, int(rec.dimension))
        rows.append(feats)
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  features: {i + 1}/{n}")
    X = pd.DataFrame(rows)
    mst_check = (X["mst_total_length"].to_numpy() - gt["mst_length"].to_numpy())
    rel = np.abs(mst_check) / np.maximum(gt["mst_length"].to_numpy(), 1e-9)
    if rel.max() > 1e-4:
        print(f"  warning: mst_total_length mismatch vs ground truth, max relative diff {rel.max():.2e}")
    return X[feature_order]


def group_shap_table(sv: np.ndarray, features: list[str], mask: np.ndarray) -> pd.DataFrame:
    """mean |SHAP| / mean signed SHAP / share(%) per feature over rows[mask]."""
    sub = sv[mask]
    mean_abs = np.abs(sub).mean(axis=0)
    mean_signed = sub.mean(axis=0)
    total = mean_abs.sum()
    share = 100.0 * mean_abs / total if total > 0 else np.zeros_like(mean_abs)
    return pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
        "share": share,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def top5(tbl: pd.DataFrame) -> list[dict]:
    return [
        {"feature": r.feature, "mean_abs_shap": float(r.mean_abs_shap),
         "mean_signed_shap": float(r.mean_signed_shap), "share_pct": float(r.share)}
        for r in tbl.head(TOP_K).itertuples(index=False)
    ]


def main() -> None:
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    feature_order = list(model.feature_name())
    print(f"  {len(feature_order)} features, {model.num_trees()} trees")

    print("Loading 2D benchmark ground truth...")
    gt = load_ground_truth().reset_index(drop=True)
    print(f"  {len(gt)} instances")

    print("Computing per-instance features (compute_features, GART 2.0 extractor)...")
    X = build_feature_matrix(gt, feature_order)

    print("Predicting (raw margin = logit(alpha-1), and deployed alpha)...")
    z = model.predict(X)
    alpha_pred = np.clip(1.0 + 1.0 / (1.0 + np.exp(-z)), 1.0, 2.0)
    gt["alpha_pred"] = alpha_pred
    gt["alpha_true"] = gt["true_cost"] / gt["mst_length"]

    print("Computing SHAP values (TreeExplainer, raw/logit-space output)...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # defensive: some SHAP/LightGBM combos return a list
        sv = sv[0]
    sv = np.asarray(sv)
    assert sv.shape == (len(gt), len(feature_order)), sv.shape

    # ---- tidy CSV: class-level and generator-level rows -------------------
    tidy_rows = []
    class_tables: dict[str, pd.DataFrame] = {}
    for cls in GEN_CLASSES:
        mask = (gt["gen_class"] == cls).to_numpy()
        tbl = group_shap_table(sv, feature_order, mask)
        class_tables[cls] = tbl
        for r in tbl.itertuples(index=False):
            tidy_rows.append({"class": cls, "generator": "", "feature": r.feature,
                               "mean_abs_shap": r.mean_abs_shap,
                               "mean_signed_shap": r.mean_signed_shap,
                               "share": r.share})

    generator_tables: dict[str, pd.DataFrame] = {}
    gen_to_class = {g: cls for cls, members in GEN_CLASSES.items() for g in members}
    for gname in sorted(gen_to_class):
        mask = (gt["generator"] == gname).to_numpy()
        tbl = group_shap_table(sv, feature_order, mask)
        generator_tables[gname] = tbl
        for r in tbl.itertuples(index=False):
            tidy_rows.append({"class": gen_to_class[gname], "generator": gname,
                               "feature": r.feature, "mean_abs_shap": r.mean_abs_shap,
                               "mean_signed_shap": r.mean_signed_shap, "share": r.share})

    tidy = pd.DataFrame(tidy_rows)
    tidy.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(tidy)} rows)")

    # ---- per-class alpha summary ------------------------------------------
    alpha_summary = {}
    for cls in GEN_CLASSES:
        sub = gt[gt["gen_class"] == cls]
        alpha_summary[cls] = {
            "n": int(len(sub)),
            "mean_pred_alpha": float(sub["alpha_pred"].mean()),
            "mean_true_alpha": float(sub["alpha_true"].mean()),
            "mean_signed_error_pred_minus_true": float((sub["alpha_pred"] - sub["alpha_true"]).mean()),
        }
    overall = {
        "n": int(len(gt)),
        "mean_pred_alpha": float(gt["alpha_pred"].mean()),
        "mean_true_alpha": float(gt["alpha_true"].mean()),
        "mean_signed_error_pred_minus_true": float((gt["alpha_pred"] - gt["alpha_true"]).mean()),
    }

    # Same block, keyed by sub-generator (e.g. "grid", "line_noise") rather
    # than by reporting class -- the paper names the jittered lattice "grid"
    # and the near-collinear generator "line_noise", not by their class label.
    alpha_by_generator = {}
    for gname in sorted(gen_to_class):
        sub = gt[gt["generator"] == gname]
        alpha_by_generator[gname] = {
            "n": int(len(sub)),
            "mean_pred_alpha": float(sub["alpha_pred"].mean()),
            "mean_true_alpha": float(sub["alpha_true"].mean()),
            "mean_signed_error_pred_minus_true": float((sub["alpha_pred"] - sub["alpha_true"]).mean()),
        }

    # ---- Line Noise vs Isotropic: signed SHAP on Isotropic's top features -
    iso_top_features = class_tables["Isotropic"].head(TOP_K)["feature"].tolist()
    iso_mask = (gt["gen_class"] == "Isotropic").to_numpy()
    ln_mask = (gt["gen_class"] == "LineNoise").to_numpy()
    idx = {f: feature_order.index(f) for f in iso_top_features}
    signed_iso = {f: float(sv[iso_mask, idx[f]].mean()) for f in iso_top_features}
    signed_ln = {f: float(sv[ln_mask, idx[f]].mean()) for f in iso_top_features}
    line_noise_vs_isotropic = {
        "top_features_used": iso_top_features,
        "note": "top-5 by mean |SHAP| on the Isotropic class",
        "mean_signed_shap_isotropic": signed_iso,
        "mean_signed_shap_linenoise": signed_ln,
        "delta_linenoise_minus_isotropic": {
            f: signed_ln[f] - signed_iso[f] for f in iso_top_features
        },
    }

    # ---- Line Noise vs training support, on the features that matter for -
    # ---- the rest of the benchmark (all classes except Line Noise) --------
    rest_mask = ~ln_mask
    rest_tbl = group_shap_table(sv, feature_order, rest_mask)
    rest_top_features = rest_tbl.head(TOP_K)["feature"].tolist()

    print("Loading training split percentiles...")
    train = pd.read_csv(TRAIN_CSV, usecols=["split"] + rest_top_features)
    train = train[train["split"] == "train"]

    train_bounds = {}
    frac_outside = {}
    # X and gt share row order/index (both built from the same gt frame)
    X_ln = X.loc[gt.index[ln_mask], :]
    for f in rest_top_features:
        lo, hi = np.percentile(train[f].to_numpy(), [1, 99])
        train_bounds[f] = [float(lo), float(hi)]
        outside = ((X_ln[f] < lo) | (X_ln[f] > hi)).to_numpy()
        frac_outside[f] = float(outside.mean())

    out_of_support = {
        "top_features_used": rest_top_features,
        "note": "top-5 by mean |SHAP| on all classes except Line Noise (n="
                f"{int(rest_mask.sum())}); train bounds from tsp_features_v4.csv split=='train' (n={len(train)})",
        "train_p1_p99": train_bounds,
        "frac_linenoise_outside_train_p1_p99": frac_outside,
    }

    # Raw (non-SHAP) descriptive stats of the same five features, over the 210
    # Line Noise rows -- the numbers a reader needs to see how far outside
    # train_p1_p99 those rows actually sit (e.g. greedy_nn_over_mst's mean and
    # [min, max] on Line Noise vs its training band above).
    linenoise_feature_stats = {}
    for f in rest_top_features:
        v = X_ln[f].to_numpy()
        lo99, hi99 = np.percentile(v, [1, 99])
        linenoise_feature_stats[f] = {
            "mean": float(v.mean()), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()),
            "p1": float(lo99), "p99": float(hi99),
        }

    summary = {
        "n_instances": int(len(gt)),
        "features": feature_order,
        "alpha": {"overall": overall, "by_class": alpha_summary, "by_generator": alpha_by_generator},
        "top5_by_class": {cls: top5(class_tables[cls]) for cls in GEN_CLASSES},
        "top5_by_generator": {g: top5(generator_tables[g]) for g in generator_tables},
        "line_noise_vs_isotropic_signed_shap": line_noise_vs_isotropic,
        "line_noise_out_of_training_support": out_of_support,
        "linenoise_feature_stats": linenoise_feature_stats,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT_JSON}")

    # ---- plain-language console summary ------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Overall: mean pred alpha={overall['mean_pred_alpha']:.3g}, "
          f"mean true alpha={overall['mean_true_alpha']:.3g}, "
          f"mean signed error={overall['mean_signed_error_pred_minus_true']:.3g}")
    for cls in GEN_CLASSES:
        a = alpha_summary[cls]
        print(f"  {cls:16s} n={a['n']:4d}  pred={a['mean_pred_alpha']:.3g}  "
              f"true={a['mean_true_alpha']:.3g}  err={a['mean_signed_error_pred_minus_true']:.3g}")

    print(f"\nTop {TOP_K} features on Isotropic (mean |SHAP|, share%):")
    for r in class_tables["Isotropic"].head(TOP_K).itertuples(index=False):
        print(f"  {r.feature:24s} |SHAP|={r.mean_abs_shap:.4g}  signed={r.mean_signed_shap:+.4g}  share={r.share:.3g}%")

    print(f"\nSame features on Line Noise:")
    for f in iso_top_features:
        print(f"  {f:24s} signed_iso={signed_iso[f]:+.4g}  signed_ln={signed_ln[f]:+.4g}  "
              f"delta={signed_ln[f]-signed_iso[f]:+.4g}")

    print(f"\nTop {TOP_K} features on the rest of the benchmark (excl. Line Noise) "
          f"and Line Noise's out-of-[p1,p99]-training-support fraction:")
    for f in rest_top_features:
        lo, hi = train_bounds[f]
        print(f"  {f:24s} train[p1,p99]=[{lo:.4g},{hi:.4g}]  "
              f"frac_LineNoise_outside={frac_outside[f]:.3g}")

    print(f"\nLine Noise's own distribution on those {TOP_K} features (raw values, n=210):")
    for f in rest_top_features:
        s = linenoise_feature_stats[f]
        print(f"  {f:24s} mean={s['mean']:.4g} median={s['median']:.4g} "
              f"min={s['min']:.4g} max={s['max']:.4g} p1={s['p1']:.4g} p99={s['p99']:.4g}")
    print("=" * 78)


if __name__ == "__main__":
    main()
