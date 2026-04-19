"""
Comprehensive TSPLIB95 benchmark analysis for GART 3.0.

Generates publication-quality figures and tables for the paper. Reads the
latest hybrid_delaunay results CSV and produces:

1. Accuracy by instance-size group (small/medium/large/very-large)
2. Accuracy by edge-weight type and MDS embedding dimension
3. Fixed-alpha baseline comparison
4. GART 3.0 vs synthetic-benchmark models (from Generalized_TSP_Analysis)
5. Timing analysis

Figures are saved to tsplib_benchmark/figures/ for inclusion in the LaTeX paper.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
RESULTS_DIR = THIS_DIR / "results"
FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(THIS_DIR))
from exclusions import filter_metric_consistent, METRIC_RATIO_THRESHOLD  # noqa: E402

# Matplotlib style for publication
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

SIZE_BINS = [0, 100, 500, 1000, 5000, 100000]
SIZE_LABELS = ["Small\n(n≤100)", "Medium\n(101–500)", "Large\n(501–1000)",
               "Very Large\n(1001–5000)", "Massive\n(n>5000)"]


def load_latest_results():
    """Load the most recent hybrid_delaunay results CSV."""
    csvs = sorted(RESULTS_DIR.glob("tsplib_results_*hybrid_delaunay*.csv"))
    if not csvs:
        csvs = sorted(RESULTS_DIR.glob("tsplib_results_*.csv"))
    if not csvs:
        raise FileNotFoundError("No result CSVs in results/")
    path = csvs[-1]
    print(f"Loading: {path.name}")
    df = pd.read_csv(path)
    df["size_group"] = pd.cut(df["n"], bins=SIZE_BINS, labels=SIZE_LABELS)
    df["true_alpha"] = df["true_cost"] / df["mst_length"]
    return df


def load_synthetic_benchmarks():
    """Load the 2D synthetic benchmark results for model comparison."""
    ckpt = REPO_ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints"
    frames = []
    for f in sorted(ckpt.glob("results_*.csv")):
        df = pd.read_csv(f)
        if "abs_gap_pct" not in df.columns and "gap_pct" in df.columns:
            df["abs_gap_pct"] = df["gap_pct"].abs()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =========================================================================
# Figure 1: Accuracy by instance size
# =========================================================================
def fig_accuracy_by_size(df):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    groups = df.groupby("size_group", observed=True)

    # Panel A: box plot of gap_pct
    ax = axes[0]
    data = [grp["gap_pct"].values for _, grp in groups]
    labels = [name for name, _ in groups]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.5))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Prediction Gap (%)")
    ax.set_title("(a) Gap Distribution by Instance Size")

    # Panel B: MAPE bar chart
    ax = axes[1]
    mape = groups["abs_gap_pct"].mean()
    counts = groups["abs_gap_pct"].count()
    bars = ax.bar(range(len(mape)), mape.values, color=colors[:len(mape)],
                  alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(mape)))
    ax.set_xticklabels(mape.index, fontsize=8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(b) Mean Absolute Percentage Error")
    for i, (v, c) in enumerate(zip(mape.values, counts.values)):
        ax.text(i, v + 0.2, f"n={c}", ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tsplib_accuracy_by_size.png")
    plt.close()
    print("  Saved: tsplib_accuracy_by_size.png")


# =========================================================================
# Figure 2: Accuracy by edge-weight type and mode
# =========================================================================
def fig_accuracy_by_type(df):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    types = df.groupby("edge_weight_type")
    type_names = sorted(df["edge_weight_type"].unique())
    mape = [types.get_group(t)["abs_gap_pct"].mean() for t in type_names]
    counts = [len(types.get_group(t)) for t in type_names]
    colors_map = {"EUC_2D": "#4c72b0", "CEIL_2D": "#55a868", "ATT": "#c44e52",
                  "GEO": "#8172b2", "EXPLICIT": "#ccb974"}
    colors = [colors_map.get(t, "gray") for t in type_names]
    bars = ax.bar(range(len(type_names)), mape, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(type_names)))
    ax.set_xticklabels(type_names, fontsize=9)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("GART 3.0 Accuracy by TSPLIB Edge-Weight Type")
    for i, (v, c) in enumerate(zip(mape, counts)):
        ax.text(i, v + 0.3, f"n={c}", ha="center", fontsize=7)
    # Annotate any non-metric outlier that slipped through (true_cost/MST > threshold).
    if {"true_cost", "mst_length"}.issubset(df.columns):
        ratio = df["true_cost"] / df["mst_length"]
        outliers = df[ratio > METRIC_RATIO_THRESHOLD]
        if not outliers.empty and "EXPLICIT" in type_names:
            names_str = ", ".join(outliers["instance"].tolist())
            ax.annotate(
                f"{names_str} excluded\n(true_cost/MST > {METRIC_RATIO_THRESHOLD})",
                xy=(type_names.index("EXPLICIT"), mape[type_names.index("EXPLICIT")]),
                xytext=(type_names.index("EXPLICIT") + 0.5, max(mape) * 0.8),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.5),
            )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tsplib_accuracy_by_type.png")
    plt.close()
    print("  Saved: tsplib_accuracy_by_type.png")


# =========================================================================
# Figure 3: Fixed-alpha baseline comparison
# =========================================================================
def fig_fixed_alpha_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel A: scatter of true alpha vs GART predicted alpha
    ax = axes[0]
    native = df[df["mode"] == "native"]
    hybrid = df[df["mode"] == "hybrid"]
    ax.scatter(native["true_alpha"], native["alpha"], s=12, alpha=0.5,
               label="Native (Euclidean)", color="#4c72b0", edgecolors="none")
    ax.scatter(hybrid["true_alpha"], hybrid["alpha"], s=20, alpha=0.7,
               label="Hybrid (MDS)", color="#c44e52", marker="^", edgecolors="none")
    lims = [0.9, 1.6]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True α (optimal / MST)")
    ax.set_ylabel("Predicted α")
    ax.set_title("(a) Predicted vs True α")
    ax.legend(loc="upper left", framealpha=0.8)

    # Panel B: MAPE comparison bar chart
    ax = axes[1]
    # Compute optimal fixed alpha
    opt = minimize_scalar(
        lambda a: ((a * df.mst_length - df.true_cost) / df.true_cost).abs().mean() * 100,
        bounds=(1.0, 2.0), method="bounded")
    fixed_alphas = {"MST only\n(α=1.0)": 1.0,
                    f"Optimal fixed\n(α={opt.x:.3f})": opt.x,
                    "GART 3.0": None}
    mapes = {}
    for label, alpha_val in fixed_alphas.items():
        if alpha_val is not None:
            pred = alpha_val * df["mst_length"]
            mapes[label] = ((pred - df.true_cost) / df.true_cost).abs().mean() * 100
        else:
            mapes[label] = df["abs_gap_pct"].mean()

    colors = ["#ccb974", "#55a868", "#4c72b0"]
    bars = ax.bar(range(len(mapes)), list(mapes.values()), color=colors,
                  alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(mapes)))
    ax.set_xticklabels(list(mapes.keys()), fontsize=8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(b) Fixed α Baseline vs GART 3.0")
    for i, v in enumerate(mapes.values()):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tsplib_fixed_alpha_comparison.png")
    plt.close()
    print("  Saved: tsplib_fixed_alpha_comparison.png")


# =========================================================================
# Figure 4: Synthetic benchmark model comparison
# =========================================================================
def fig_synthetic_model_comparison(synth_df):
    if synth_df.empty:
        print("  Skipped: no synthetic benchmark data")
        return

    # Filter to models with abs_gap_pct
    valid = synth_df.dropna(subset=["abs_gap_pct"])
    if valid.empty:
        return

    grp = valid.groupby("model").agg(
        mape=("abs_gap_pct", "mean"),
        avg_time=("prediction_time_s", "mean"),
        avg_opt_time=("optimal_solve_time_s", "mean"),
    )
    grp["speedup"] = grp["avg_opt_time"] / grp["avg_time"]
    grp = grp.sort_values("mape")

    # Select key models for clean visualization
    key_models = ["LGBM_V3", "Neural_V3", "Interp_V3", "Linear_V3", "GART",
                  "Composite", "MST_Ratio", "Cavdar", "BHH", "Chien", "Hilbert"]
    avail = [m for m in key_models if m in grp.index]
    grp_sel = grp.loc[avail]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # Panel A: MAPE comparison
    ax = axes[0]
    colors = ["#4c72b0" if "LGBM" in m or "GART" == m else "#999999"
              for m in grp_sel.index]
    colors[0] = "#2ca02c"  # Highlight LGBM_V3
    ax.barh(range(len(grp_sel)), grp_sel["mape"].values, color=colors,
            alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(grp_sel)))
    ax.set_yticklabels(grp_sel.index, fontsize=8)
    ax.set_xlabel("MAPE (%) on Synthetic 2D Benchmark")
    ax.set_title("(a) Accuracy (lower is better)")
    ax.invert_yaxis()
    for i, v in enumerate(grp_sel["mape"].values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)

    # Panel B: speedup vs optimal solver
    ax = axes[1]
    ax.barh(range(len(grp_sel)), grp_sel["speedup"].values, color=colors,
            alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(grp_sel)))
    ax.set_yticklabels(grp_sel.index, fontsize=8)
    ax.set_xlabel("Speedup over Optimal Solver (×)")
    ax.set_title("(b) Speed (higher is better)")
    ax.invert_yaxis()
    ax.set_xscale("log")
    for i, v in enumerate(grp_sel["speedup"].values):
        ax.text(v * 1.1, i, f"{v:.0f}×", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "synthetic_model_comparison.png")
    plt.close()
    print("  Saved: synthetic_model_comparison.png")


# =========================================================================
# Figure 5: MDS dimension vs accuracy for non-Euclidean instances
# =========================================================================
def fig_mds_dimension_analysis(df):
    hybrid = df[df["mode"] == "hybrid"].copy()
    if hybrid.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 3.2))
    # Color by edge weight type
    for ewt, marker, color in [("EXPLICIT", "s", "#ccb974"),
                                 ("GEO", "^", "#8172b2"),
                                 ("ATT", "o", "#c44e52")]:
        sub = hybrid[hybrid["edge_weight_type"] == ewt]
        if sub.empty:
            continue
        ax.scatter(sub["feature_dim"], sub["abs_gap_pct"], s=30, marker=marker,
                   color=color, alpha=0.7, edgecolors="black", linewidth=0.3,
                   label=ewt)

    ax.set_xlabel("MDS Embedding Dimension")
    ax.set_ylabel("|Gap| (%)")
    ax.set_title("Accuracy vs Embedding Dimension (Non-Euclidean Instances)")
    ax.legend(framealpha=0.8)

    # Exclude non-metric outliers from y-axis scale
    non_outlier = filter_metric_consistent(hybrid)
    if not non_outlier.empty:
        ax.set_ylim(0, non_outlier["abs_gap_pct"].max() * 1.2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tsplib_mds_dimension_analysis.png")
    plt.close()
    print("  Saved: tsplib_mds_dimension_analysis.png")


# =========================================================================
# Figure 6: Prediction gap vs instance size (scatter)
# =========================================================================
def fig_gap_vs_n(df):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    native = df[df["mode"] == "native"]
    hybrid = filter_metric_consistent(df[df["mode"] == "hybrid"])

    ax.scatter(native["n"], native["gap_pct"], s=12, alpha=0.5,
               color="#4c72b0", label="Native Euclidean", edgecolors="none")
    ax.scatter(hybrid["n"], hybrid["gap_pct"], s=20, alpha=0.7,
               color="#c44e52", marker="^", label="Hybrid MDS", edgecolors="none")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Nodes (n)")
    ax.set_ylabel("Prediction Gap (%)")
    ax.set_title("GART 3.0 Prediction Gap vs Instance Size (TSPLIB)")
    ax.legend(loc="upper left", framealpha=0.8)

    # Training range annotation
    ax.axvline(1000, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(1100, ax.get_ylim()[1] * 0.9, "Training\ncap", fontsize=7,
            color="red", alpha=0.6)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tsplib_gap_vs_n.png")
    plt.close()
    print("  Saved: tsplib_gap_vs_n.png")


# =========================================================================
# Summary table (LaTeX)
# =========================================================================
def generate_summary_table(df):
    """Print a LaTeX-ready summary table."""
    groups = [
        ("All instances", df),
        ("EUC\\_2D", df[df["edge_weight_type"] == "EUC_2D"]),
        ("CEIL\\_2D", df[df["edge_weight_type"] == "CEIL_2D"]),
        ("GEO (hybrid)", df[df["edge_weight_type"] == "GEO"]),
        ("ATT (hybrid)", df[df["edge_weight_type"] == "ATT"]),
        ("EXPLICIT (hybrid)", df[df["edge_weight_type"] == "EXPLICIT"]),
        ("n \\leq 1000", df[df["in_training_n_range"]]),
        ("n > 1000", df[~df["in_training_n_range"]]),
    ]

    lines = []
    lines.append("\\begin{tabular}{l r r r r r r}")
    lines.append("\\toprule")
    lines.append("Subset & Count & MAPE (\\%) & Median (\\%) & Bias (\\%) & p90 (\\%) & Lat. (ms) \\\\")
    lines.append("\\midrule")
    for label, sub in groups:
        if sub.empty:
            continue
        n = len(sub)
        mape = sub["abs_gap_pct"].mean()
        med = sub["abs_gap_pct"].median()
        bias = sub["gap_pct"].mean()
        p90 = sub["abs_gap_pct"].quantile(0.90)
        lat = (sub["feature_time_s"] + sub["inference_time_s"]).mean() * 1000
        lines.append(f"{label} & {n} & {mape:.2f} & {med:.2f} & {bias:+.2f} & {p90:.2f} & {lat:.1f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    table = "\n".join(lines)
    with open(FIGURES_DIR / "tsplib_summary_table.tex", "w") as f:
        f.write(table)
    print("  Saved: tsplib_summary_table.tex")
    print(table)


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 60)
    print("TSPLIB95 Benchmark Analysis for GART 3.0")
    print("=" * 60)

    df = load_latest_results()
    synth = load_synthetic_benchmarks()

    print(f"\nTSPLIB instances: {len(df)}")
    print(f"Synthetic benchmark rows: {len(synth)}")
    print()

    # Exclude non-metric outliers from aggregate stats (true_cost/MST > threshold).
    df_clean = filter_metric_consistent(df)
    dropped = sorted(set(df["instance"]) - set(df_clean["instance"]))
    if dropped:
        print(f"Aggregate filter dropped {len(dropped)} non-metric instance(s) "
              f"(true_cost/MST > {METRIC_RATIO_THRESHOLD}): {', '.join(dropped)}")

    print("Generating figures...")
    fig_accuracy_by_size(df_clean)
    fig_accuracy_by_type(df)  # Keep full set so the annotation can flag outliers
    fig_fixed_alpha_comparison(df_clean)
    fig_synthetic_model_comparison(synth)
    fig_mds_dimension_analysis(df)
    fig_gap_vs_n(df)

    print("\nGenerating LaTeX table...")
    generate_summary_table(df_clean)

    # Print key stats for the report
    print("\n" + "=" * 60)
    print("KEY FINDINGS FOR THE PAPER")
    print("=" * 60)

    euc = df_clean[df_clean["edge_weight_type"] == "EUC_2D"]
    hybrid = df_clean[df_clean["mode"] == "hybrid"]
    in_range = df_clean[df_clean["in_training_n_range"]]
    extrap = df_clean[~df_clean["in_training_n_range"]]

    print(f"\n1. Overall MAPE (metric-consistent, true_cost/MST <= {METRIC_RATIO_THRESHOLD}): "
          f"{df_clean.abs_gap_pct.mean():.2f}%")
    print(f"   Median: {df_clean.abs_gap_pct.median():.2f}%")
    print(f"   Bias: {df_clean.gap_pct.mean():+.2f}%")

    print(f"\n2. EUC_2D MAPE: {euc.abs_gap_pct.mean():.2f}%")
    print(f"   This on {len(euc)} real-world instances vs synthetic training data")

    print(f"\n3. Hybrid (non-Euclidean) MAPE: {hybrid.abs_gap_pct.mean():.2f}%")
    print(f"   GEO: {df_clean[df_clean.edge_weight_type=='GEO'].abs_gap_pct.mean():.2f}%")
    print(f"   ATT: {df_clean[df_clean.edge_weight_type=='ATT'].abs_gap_pct.mean():.2f}%")
    print(f"   EXPLICIT: {df_clean[df_clean.edge_weight_type=='EXPLICIT'].abs_gap_pct.mean():.2f}%")

    print(f"\n4. In-training-range (n<=1000): MAPE={in_range.abs_gap_pct.mean():.2f}%")
    print(f"   Extrapolated (n>1000): MAPE={extrap.abs_gap_pct.mean():.2f}%")
    print(f"   Degradation: {extrap.abs_gap_pct.mean() - in_range.abs_gap_pct.mean():.2f}pp")

    opt_fixed = minimize_scalar(
        lambda a: ((a * df_clean.mst_length - df_clean.true_cost) / df_clean.true_cost).abs().mean() * 100,
        bounds=(1.0, 2.0), method="bounded")
    print(f"\n5. Fixed-alpha baseline: best alpha={opt_fixed.x:.4f} -> MAPE={opt_fixed.fun:.2f}%")
    print(f"   GART 3.0 improvement: {opt_fixed.fun - df_clean.abs_gap_pct.mean():.2f}pp")

    if not synth.empty:
        synth_valid = synth.dropna(subset=["abs_gap_pct"])
        if not synth_valid.empty:
            lgbm_synth = synth_valid[synth_valid["model"] == "LGBM_V3"]
            if not lgbm_synth.empty:
                print(f"\n6. Synthetic 2D benchmark (LGBM_V3): MAPE={lgbm_synth.abs_gap_pct.mean():.2f}%")
                print(f"   TSPLIB EUC_2D: MAPE={euc.abs_gap_pct.mean():.2f}%")
                print(f"   Generalization gap: {euc.abs_gap_pct.mean() - lgbm_synth.abs_gap_pct.mean():.2f}pp")

    print(f"\n7. Timing (EUC_2D avg):")
    lat = (euc.feature_time_s + euc.inference_time_s).mean() * 1000
    print(f"   Feature+Inference: {lat:.1f} ms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
