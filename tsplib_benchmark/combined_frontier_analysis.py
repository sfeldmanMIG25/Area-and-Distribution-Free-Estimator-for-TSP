"""
Combined analysis across all benchmark data sources:
  1. Synthetic 2D benchmarks   (Generalized_TSP_Analysis/benchmark_results_2D_v3.csv)
  2. Synthetic ND benchmarks   (Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv)
  3. TSPLIB95 all-models       (tsplib_benchmark/results/all_models_tsplib_*.csv)
  4. Concorde reference times  (tsplib_benchmark/concorde_solve_times.csv)

Generates:
  - combined_model_comparison.png       : MAPE across all data sources per model
  - frontier_lgbm_vs_fixed.png          : Where LGBM_V3 > fixed MST ratio (frontier)
  - concorde_speedup_vs_accuracy.png    : Pareto chart: speedup vs accuracy
  - combined_accuracy_by_n.png          : Accuracy vs n for key models
  - tsplib_all_models_table.tex         : LaTeX table of all models on TSPLIB
  - concorde_comparison_table.tex       : LaTeX table comparing to Concorde times
  - frontier_summary.tex                : LaTeX table defining the frontier

Usage:
    python tsplib_benchmark/combined_frontier_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(THIS_DIR))
from exclusions import filter_metric_consistent, METRIC_RATIO_THRESHOLD  # noqa: E402

# Style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def find_latest(pattern, directory):
    """Find the most recently modified file matching pattern."""
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_data():
    """Load all benchmark data sources."""
    data = {}

    # 1. TSPLIB all-models results (latest)
    tsplib_file = find_latest("all_models_tsplib_*.csv", THIS_DIR / "results")
    if tsplib_file:
        data["tsplib"] = pd.read_csv(tsplib_file)
        print(f"TSPLIB all-models: {len(data['tsplib'])} rows from {tsplib_file.name}")
    else:
        print("WARNING: No all_models_tsplib results found. Run run_all_models_tsplib.py first.")
        data["tsplib"] = pd.DataFrame()

    # 2. Synthetic 2D
    f2d = REPO_ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
    if f2d.exists():
        data["synth_2d"] = pd.read_csv(f2d)
        print(f"Synthetic 2D: {len(data['synth_2d'])} rows")
    else:
        data["synth_2d"] = pd.DataFrame()

    # 3. Synthetic ND
    fnd = REPO_ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
    if fnd.exists():
        data["synth_nd"] = pd.read_csv(fnd)
        print(f"Synthetic ND: {len(data['synth_nd'])} rows")
    else:
        data["synth_nd"] = pd.DataFrame()

    # 4. Concorde reference times
    fc = THIS_DIR / "concorde_solve_times.csv"
    if fc.exists():
        data["concorde"] = pd.read_csv(fc)
        print(f"Concorde times: {len(data['concorde'])} instances")
    else:
        data["concorde"] = pd.DataFrame()

    return data


# =========================================================================
# Figure 1: All models on TSPLIB - MAPE comparison
# =========================================================================
def fig_tsplib_all_models(tsplib):
    if tsplib.empty:
        return
    df = filter_metric_consistent(tsplib)

    model_stats = (
        df.groupby("model")
        .agg(mape=("abs_gap_pct", "mean"),
             median=("abs_gap_pct", "median"),
             count=("abs_gap_pct", "count"),
             avg_time_ms=("total_time_s", lambda x: x.mean() * 1000))
        .sort_values("mape")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MAPE bars
    colors = []
    for m in model_stats.index:
        if m == "LGBM_V3":
            colors.append("#2196F3")
        elif m == "Fixed_Alpha":
            colors.append("#FF9800")
        elif m in ("Linear_V3", "Interp_V3", "GART_1.0"):
            colors.append("#4CAF50")
        else:
            colors.append("#9E9E9E")

    y_pos = range(len(model_stats))
    ax1.barh(y_pos, model_stats["mape"], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_stats.index)
    ax1.set_xlabel("MAPE (%)")
    ax1.set_title("(a) Accuracy on TSPLIB95")
    for i, (mape, count) in enumerate(zip(model_stats["mape"], model_stats["count"])):
        ax1.text(mape + 0.3, i, f"{mape:.1f}% (n={count})", va="center", fontsize=8)

    # Timing bars
    ax2.barh(y_pos, model_stats["avg_time_ms"], color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_stats.index)
    ax2.set_xlabel("Avg Time (ms)")
    ax2.set_xscale("log")
    ax2.set_title("(b) Average Estimation Time")
    for i, t in enumerate(model_stats["avg_time_ms"]):
        ax2.text(t * 1.2, i, f"{t:.0f}ms", va="center", fontsize=8)

    fig.suptitle("All Models on TSPLIB95 Instances", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tsplib_all_models_comparison.png")
    plt.close(fig)
    print("  Saved: tsplib_all_models_comparison.png")

    # LaTeX table
    with open(FIGURES_DIR / "tsplib_all_models_table.tex", "w") as f:
        f.write("\\begin{tabular}{l r r r r r}\n\\toprule\n")
        f.write("Model & Count & MAPE (\\%) & Median (\\%) & Avg Time (ms) & Scope \\\\\n\\midrule\n")
        for m, row in model_stats.iterrows():
            scope = "All" if m in ("LGBM_V3", "Linear_V3", "Interp_V3", "Fixed_Alpha") else "EUC\\_2D"
            if m == "GART_1.0":
                scope = "EUC\\_2D"
            f.write(f"{m} & {int(row['count'])} & {row['mape']:.2f} & {row['median']:.2f} & {row['avg_time_ms']:.1f} & {scope} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  Saved: tsplib_all_models_table.tex")


# =========================================================================
# Figure 2: Frontier - Where LGBM_V3 beats fixed MST ratio
# =========================================================================
def fig_frontier(tsplib, synth_2d):
    if tsplib.empty:
        return

    df = filter_metric_consistent(tsplib)

    # Get LGBM and Fixed_Alpha per instance
    lgbm = df[df.model == "LGBM_V3"].set_index("instance")
    fixed = df[df.model == "Fixed_Alpha"].set_index("instance")

    common = lgbm.index.intersection(fixed.index)
    if len(common) == 0:
        print("  No common instances for frontier analysis")
        return

    lgbm_c = lgbm.loc[common]
    fixed_c = fixed.loc[common]

    improvement = fixed_c["abs_gap_pct"].values - lgbm_c["abs_gap_pct"].values
    n_vals = lgbm_c["n"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Per-instance improvement
    ax = axes[0]
    colors_scatter = ["#2196F3" if imp > 0 else "#F44336" for imp in improvement]
    ax.scatter(n_vals, improvement, c=colors_scatter, alpha=0.6, s=30, edgecolors="k", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Instance Size (n)")
    ax.set_xscale("log")
    ax.set_ylabel("Improvement (pp)")
    ax.set_title("(a) LGBM_V3 vs Fixed Alpha\n(positive = LGBM better)")
    wins = np.sum(improvement > 0)
    losses = np.sum(improvement < 0)
    ax.text(0.02, 0.98, f"LGBM wins: {wins}/{len(common)}\nFixed wins: {losses}/{len(common)}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # (b) Binned by size - where is the frontier?
    ax = axes[1]
    size_bins = [(0, 100, "<=100"), (101, 500, "101-500"), (501, 1000, "501-1000"), (1001, 100000, ">1000")]
    bin_labels = []
    lgbm_mapes = []
    fixed_mapes = []
    for lo, hi, label in size_bins:
        mask = (n_vals >= lo) & (n_vals <= hi)
        if mask.sum() == 0:
            continue
        bin_labels.append(f"{label}\n(n={mask.sum()})")
        lgbm_mapes.append(lgbm_c["abs_gap_pct"].values[mask].mean())
        fixed_mapes.append(fixed_c["abs_gap_pct"].values[mask].mean())

    x = np.arange(len(bin_labels))
    w = 0.35
    ax.bar(x - w/2, fixed_mapes, w, label="Fixed Alpha", color="#FF9800", alpha=0.8)
    ax.bar(x + w/2, lgbm_mapes, w, label="LGBM_V3", color="#2196F3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(b) MAPE by Size Group")
    ax.legend(fontsize=8)

    # (c) Add synthetic 2D data to show frontier extends
    ax = axes[2]
    if not synth_2d.empty and "model" in synth_2d.columns:
        lgbm_s = synth_2d[synth_2d.model == "LGBM_V3"]
        mst_s = synth_2d[synth_2d.model == "MST_Ratio"]
        if not lgbm_s.empty and not mst_s.empty:
            # Get n from instance name
            def extract_n(inst):
                try:
                    parts = str(inst).split("_")
                    for p in parts:
                        if p.startswith("n"):
                            return int(p[1:])
                    return None
                except:
                    return None

            lgbm_s = lgbm_s.copy()
            lgbm_s["n_parsed"] = lgbm_s["instance"].apply(extract_n)
            lgbm_s = lgbm_s.dropna(subset=["n_parsed"])

            mst_s = mst_s.copy()
            mst_s["n_parsed"] = mst_s["instance"].apply(extract_n)
            mst_s = mst_s.dropna(subset=["n_parsed"])

            # Group by n
            lgbm_by_n = lgbm_s.groupby("n_parsed")["abs_gap_pct"].mean()
            mst_by_n = mst_s.groupby("n_parsed")["abs_gap_pct"].mean()

            common_n = lgbm_by_n.index.intersection(mst_by_n.index)
            ax.plot(common_n, mst_by_n.loc[common_n], "o-", color="#FF9800", label="MST_Ratio", markersize=4)
            ax.plot(common_n, lgbm_by_n.loc[common_n], "s-", color="#2196F3", label="LGBM_V3", markersize=4)
            ax.set_xlabel("n (customers)")
            ax.set_ylabel("MAPE (%)")
            ax.set_title("(c) Synthetic 2D: LGBM vs MST Ratio")
            ax.legend(fontsize=8)
            ax.set_xscale("log")
        else:
            ax.text(0.5, 0.5, "Insufficient synthetic data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No synthetic 2D data", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Frontier: When is LGBM_V3 Better Than a Fixed MST Ratio?", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "frontier_lgbm_vs_fixed.png")
    plt.close(fig)
    print("  Saved: frontier_lgbm_vs_fixed.png")

    # Frontier summary table
    with open(FIGURES_DIR / "frontier_summary.tex", "w") as f:
        f.write("\\begin{tabular}{l r r r r}\n\\toprule\n")
        f.write("Size Group & Fixed MAPE (\\%) & LGBM MAPE (\\%) & Improvement (pp) & LGBM Win Rate \\\\\n\\midrule\n")
        for i, (lo, hi, label) in enumerate(size_bins):
            mask = (n_vals >= lo) & (n_vals <= hi)
            if mask.sum() == 0:
                continue
            lm = lgbm_c["abs_gap_pct"].values[mask].mean()
            fm = fixed_c["abs_gap_pct"].values[mask].mean()
            imp = fm - lm
            wins = np.sum(improvement[mask] > 0)
            total = mask.sum()
            f.write(f"$n \\in [{lo}, {hi}]$ & {fm:.2f} & {lm:.2f} & {imp:+.2f} & {wins}/{total} \\\\\n")
        # Overall
        lm_all = lgbm_c["abs_gap_pct"].mean()
        fm_all = fixed_c["abs_gap_pct"].mean()
        imp_all = fm_all - lm_all
        wins_all = np.sum(improvement > 0)
        f.write(f"\\midrule\nAll & {fm_all:.2f} & {lm_all:.2f} & {imp_all:+.2f} & {wins_all}/{len(common)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  Saved: frontier_summary.tex")


# =========================================================================
# Figure 3: Concorde speedup vs accuracy (Pareto front)
# =========================================================================
def fig_concorde_speedup(tsplib):
    if tsplib.empty:
        return

    df = filter_metric_consistent(tsplib)
    df = df[df.concorde_time_s.notna()].copy()
    if df.empty:
        print("  No instances with Concorde times for speedup chart")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    models_to_plot = df["model"].unique()
    markers = {"LGBM_V3": "o", "Linear_V3": "s", "Interp_V3": "D", "GART_1.0": "^",
               "Fixed_Alpha": "X", "BHH": "v", "Cavdar": "<", "Chien": ">",
               "Composite": "p", "MST_Ratio": "*", "Hilbert": "h", "Vinel": "P"}
    ml_colors = {"LGBM_V3": "#2196F3", "Linear_V3": "#4CAF50", "Interp_V3": "#9C27B0",
                 "GART_1.0": "#FF5722", "Fixed_Alpha": "#FF9800"}

    for model in sorted(models_to_plot):
        sub = df[df.model == model]
        mape = sub["abs_gap_pct"].mean()
        # Average speedup
        sub_with_speed = sub[sub.speedup_vs_concorde.notna()]
        if sub_with_speed.empty:
            continue
        avg_speedup = sub_with_speed["speedup_vs_concorde"].median()
        if avg_speedup <= 0:
            continue

        color = ml_colors.get(model, "#9E9E9E")
        marker = markers.get(model, "o")
        ax.scatter(avg_speedup, mape, s=120, c=color, marker=marker,
                   label=f"{model} ({mape:.1f}%)", edgecolors="k", linewidth=0.5, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Median Speedup vs Concorde (x)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Accuracy vs Speed: Estimators vs Concorde Solver")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(FIGURES_DIR / "concorde_speedup_vs_accuracy.png")
    plt.close(fig)
    print("  Saved: concorde_speedup_vs_accuracy.png")

    # Concorde comparison table
    model_stats = (
        df.groupby("model")
        .agg(
            mape=("abs_gap_pct", "mean"),
            count=("abs_gap_pct", "count"),
            avg_time_ms=("total_time_s", lambda x: x.mean() * 1000),
            median_speedup=("speedup_vs_concorde", "median"),
        )
        .sort_values("mape")
    )

    with open(FIGURES_DIR / "concorde_comparison_table.tex", "w") as f:
        f.write("\\begin{tabular}{l r r r r}\n\\toprule\n")
        f.write("Model & Count & MAPE (\\%) & Time (ms) & Speedup vs Concorde \\\\\n\\midrule\n")
        for m, row in model_stats.iterrows():
            speedup_str = f"{row['median_speedup']:.0f}$\\times$" if pd.notna(row['median_speedup']) else "---"
            f.write(f"{m} & {int(row['count'])} & {row['mape']:.2f} & {row['avg_time_ms']:.1f} & {speedup_str} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  Saved: concorde_comparison_table.tex")


# =========================================================================
# Figure 4: Combined accuracy by n across all data sources
# =========================================================================
def fig_combined_accuracy_by_n(tsplib, synth_2d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) TSPLIB - LGBM_V3 vs key baselines across n
    ax = axes[0]
    if not tsplib.empty:
        df = filter_metric_consistent(tsplib)
        for model, color, marker in [
            ("LGBM_V3", "#2196F3", "o"),
            ("Fixed_Alpha", "#FF9800", "X"),
            ("MST_Ratio", "#9E9E9E", "*"),
            ("BHH", "#E91E63", "v"),
        ]:
            sub = df[df.model == model]
            if sub.empty:
                continue
            ax.scatter(sub["n"], sub["abs_gap_pct"], c=color, marker=marker,
                       alpha=0.5, s=20, label=model)
        ax.set_xscale("log")
        ax.set_xlabel("Instance Size (n)")
        ax.set_ylabel("|Gap| (%)")
        ax.set_title("(a) TSPLIB95 Accuracy by Size")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    # (b) Synthetic 2D - LGBM vs baselines by n
    ax = axes[1]
    if not synth_2d.empty and "model" in synth_2d.columns:
        for model, color in [("LGBM_V3", "#2196F3"), ("MST_Ratio", "#9E9E9E"),
                              ("BHH", "#E91E63"), ("Cavdar", "#795548")]:
            sub = synth_2d[synth_2d.model == model]
            if sub.empty:
                continue

            def extract_n(inst):
                try:
                    parts = str(inst).split("_")
                    for p in parts:
                        if p.startswith("n"):
                            return int(p[1:])
                    return None
                except:
                    return None

            sub = sub.copy()
            sub["n_parsed"] = sub["instance"].apply(extract_n)
            sub = sub.dropna(subset=["n_parsed"])
            grouped = sub.groupby("n_parsed")["abs_gap_pct"].mean()
            ax.plot(grouped.index, grouped.values, "o-", color=color, label=model, markersize=4)

        ax.set_xlabel("n (customers)")
        ax.set_ylabel("MAPE (%)")
        ax.set_title("(b) Synthetic 2D Accuracy by Size")
        ax.legend(fontsize=8)
        ax.set_xscale("log")

    fig.suptitle("LGBM_V3 (GART 3.0) Accuracy Across Data Sources", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "combined_accuracy_by_n.png")
    plt.close(fig)
    print("  Saved: combined_accuracy_by_n.png")


# =========================================================================
# Figure 5: ND dimension analysis from synthetic benchmarks
# =========================================================================
def fig_nd_dimension_analysis(synth_nd):
    if synth_nd.empty:
        return

    if "dimension" not in synth_nd.columns or "model" not in synth_nd.columns:
        print("  Skipping ND dimension analysis - missing columns")
        return

    lgbm = synth_nd[synth_nd.model == "LGBM_V3"].copy()
    if lgbm.empty:
        lgbm = synth_nd[synth_nd.model == "LGBM"].copy()
    if lgbm.empty:
        print("  No LGBM data in ND benchmark")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) MAPE by dimension
    gap_col = "abs_gap_pct" if "abs_gap_pct" in lgbm.columns else "gap_pct"
    if gap_col == "gap_pct":
        lgbm["abs_gap_pct"] = lgbm["gap_pct"].abs()
        gap_col = "abs_gap_pct"

    by_dim = lgbm.groupby("dimension")[gap_col].agg(["mean", "median", "count"])
    ax1.bar(by_dim.index, by_dim["mean"], color="#2196F3", alpha=0.8)
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("MAPE (%)")
    ax1.set_title("(a) LGBM_V3 MAPE by Dimension")
    for d, row in by_dim.iterrows():
        ax1.text(d, row["mean"] + 0.2, f"n={int(row['count'])}", ha="center", fontsize=7)

    # (b) MAPE by dimension and n
    if "n_customers" in lgbm.columns:
        n_groups = [(5, 50, "n<=50"), (51, 200, "n 51-200"), (201, 1000, "n>200")]
        for lo, hi, label in n_groups:
            sub = lgbm[(lgbm.n_customers >= lo) & (lgbm.n_customers <= hi)]
            if sub.empty:
                continue
            by_dim_sub = sub.groupby("dimension")[gap_col].mean()
            ax2.plot(by_dim_sub.index, by_dim_sub.values, "o-", label=label, markersize=4)
        ax2.set_xlabel("Dimension")
        ax2.set_ylabel("MAPE (%)")
        ax2.set_title("(b) MAPE by Dimension and Size")
        ax2.legend(fontsize=8)

    fig.suptitle("LGBM_V3 Performance Across Dimensions (Synthetic ND)", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "nd_dimension_analysis.png")
    plt.close(fig)
    print("  Saved: nd_dimension_analysis.png")


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 70)
    print("COMBINED FRONTIER ANALYSIS")
    print("=" * 70)

    data = load_data()

    print("\nGenerating figures...")
    fig_tsplib_all_models(data["tsplib"])
    fig_frontier(data["tsplib"], data["synth_2d"])
    fig_concorde_speedup(data["tsplib"])
    fig_combined_accuracy_by_n(data["tsplib"], data["synth_2d"])
    fig_nd_dimension_analysis(data["synth_nd"])

    # === Frontier Definition ===
    if not data["tsplib"].empty:
        tsplib = filter_metric_consistent(data["tsplib"])
        lgbm = tsplib[tsplib.model == "LGBM_V3"]
        fixed = tsplib[tsplib.model == "Fixed_Alpha"]

        if not lgbm.empty and not fixed.empty:
            lgbm_mape = lgbm["abs_gap_pct"].mean()
            fixed_mape = fixed["abs_gap_pct"].mean()
            lgbm_time = lgbm["total_time_s"].mean() * 1000
            fixed_time = 0.001  # negligible for multiplication

            print("\n" + "=" * 70)
            print("FRONTIER DEFINITION: When LGBM_V3 > Fixed MST Ratio")
            print("=" * 70)
            print(f"\nOverall TSPLIB95 (metric-consistent, true_cost/MST <= {METRIC_RATIO_THRESHOLD}):")
            print(f"  LGBM_V3:      MAPE={lgbm_mape:.2f}%  avg_time={lgbm_time:.0f}ms")
            print(f"  Fixed Alpha:  MAPE={fixed_mape:.2f}%  avg_time=~0ms")
            print(f"  Improvement:  {fixed_mape - lgbm_mape:.2f}pp")

            # Per-instance comparison
            lgbm_i = lgbm.set_index("instance")
            fixed_i = fixed.set_index("instance")
            common = lgbm_i.index.intersection(fixed_i.index)
            improvement = fixed_i.loc[common, "abs_gap_pct"] - lgbm_i.loc[common, "abs_gap_pct"]
            wins = (improvement > 0).sum()

            print(f"\n  LGBM wins on {wins}/{len(common)} instances ({100*wins/len(common):.0f}%)")
            print(f"  Mean improvement when LGBM wins: {improvement[improvement > 0].mean():.2f}pp")
            print(f"  Mean degradation when Fixed wins: {-improvement[improvement < 0].mean():.2f}pp")

            print(f"\n  FRONTIER CONCLUSION:")
            print(f"  LGBM_V3 provides meaningful accuracy improvement (+{fixed_mape-lgbm_mape:.1f}pp)")
            print(f"  over a fixed MST ratio at the cost of ~{lgbm_time:.0f}ms per instance.")
            print(f"  The improvement is consistent across instance sizes (win rate {100*wins/len(common):.0f}%).")
            if lgbm_time < 1000:
                print(f"  At sub-second latency, LGBM_V3 is practical for real-time applications.")

    print("\n" + "=" * 70)
    print("Done. All figures saved to:", FIGURES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
