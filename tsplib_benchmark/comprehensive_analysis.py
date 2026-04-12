"""
Comprehensive Benchmark Analysis for GART 3.0 Paper

Generates:
- Model comparison across all data sources
- Frontier analysis (when GART 3.0 wins vs MST vs optimal)
- Speedup vs accuracy Pareto frontier
- Dimension-specific performance (D>2)
- High-quality figures for publication
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
FIGURES_DIR = REPO_ROOT / "tsplib_benchmark" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "serif",
    }
)


def find_latest(pattern, directory):
    """Find the most recently modified file matching pattern."""
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_all_data():
    """Load all benchmark data sources."""
    data = {}

    # TSPLIB all-models results
    tsplib_file = find_latest(
        "all_models_tsplib_*.csv", REPO_ROOT / "tsplib_benchmark" / "results"
    )
    if tsplib_file:
        data["tsplib"] = pd.read_csv(tsplib_file)
        print(f"TSPLIB: {len(data['tsplib'])} rows from {tsplib_file.name}")

    # Synthetic 2D
    f2d = REPO_ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
    if f2d.exists():
        data["synth_2d"] = pd.read_csv(f2d)
        print(f"Synthetic 2D: {len(data['synth_2d'])} rows")

    # Synthetic ND
    fnd = REPO_ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
    if fnd.exists():
        data["synth_nd"] = pd.read_csv(fnd)
        print(f"Synthetic ND: {len(data['synth_nd'])} rows")

    # Concorde times
    fc = REPO_ROOT / "tsplib_benchmark" / "concorde_solve_times.csv"
    if fc.exists():
        data["concorde"] = pd.read_csv(fc)
        print(f"Concorde times: {len(data['concorde'])} instances")

    return data


def compute_frontier_stats(tsplib):
    """Compute detailed frontier statistics."""
    lgbm = tsplib[tsplib.model == "LGBM_V3"].copy()
    mst = tsplib[tsplib.model == "MST_Ratio"].set_index("instance")
    lgbm_idx = lgbm.set_index("instance")

    common = lgbm_idx.index.intersection(mst.index)

    # Overall comparison
    lgbm_gap = lgbm_idx.loc[common, "abs_gap_pct"]
    mst_gap = mst.loc[common, "abs_gap_pct"]

    wins = (lgbm_gap < mst_gap).sum()
    losses = (lgbm_gap > mst_gap).sum()
    ties = (lgbm_gap == mst_gap).sum()

    improvement_when_wins = (
        mst_gap[lgbm_gap < mst_gap] - lgbm_gap[lgbm_gap < mst_gap]
    ).mean()
    degradation_when_loses = (
        lgbm_gap[lgbm_gap > mst_gap] - mst_gap[lgbm_gap > mst_gap]
    ).mean()

    # By instance size
    results = {
        "overall": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": 100 * wins / len(common),
            "lgbm_mape": lgbm_gap.mean(),
            "mst_mape": mst_gap.mean(),
            "improvement_pp": mst_gap.mean() - lgbm_gap.mean(),
            "improvement_when_wins": improvement_when_wins,
            "degradation_when_loses": degradation_when_loses,
        }
    }

    # By size bins
    for lo, hi, label in [
        (0, 50, "tiny"),
        (50, 100, "small"),
        (100, 200, "medium"),
        (200, 500, "large"),
        (500, 10000, "xlarge"),
    ]:
        mask = (lgbm["n"] >= lo) & (lgbm["n"] < hi)
        if mask.sum() > 0:
            sub = lgbm[mask].set_index("instance")
            c = sub.index.intersection(mst.index)
            if len(c) > 0:
                results[label] = {
                    "n": len(c),
                    "lgbm_mape": sub.loc[c, "abs_gap_pct"].mean(),
                    "mst_mape": mst.loc[c, "abs_gap_pct"].mean(),
                    "wins": (
                        sub.loc[c, "abs_gap_pct"] < mst.loc[c, "abs_gap_pct"]
                    ).sum(),
                }

    # By edge type
    for etype in lgbm["edge_weight_type"].unique():
        sub = lgbm[lgbm["edge_weight_type"] == etype].set_index("instance")
        c = sub.index.intersection(mst.index)
        if len(c) > 0:
            results[f"etype_{etype}"] = {
                "n": len(c),
                "lgbm_mape": sub.loc[c, "abs_gap_pct"].mean(),
                "mst_mape": mst.loc[c, "abs_gap_pct"].mean(),
            }

    return results


def create_model_comparison_table(data):
    """Create comprehensive model comparison table."""
    models_info = {}

    # TSPLIB results
    if "tsplib" in data and not data["tsplib"].empty:
        tsplib = data["tsplib"]
        for model in tsplib["model"].unique():
            sub = tsplib[tsplib.model == model]
            models_info[model] = {
                "tsplib_mape": sub["abs_gap_pct"].mean(),
                "tsplib_std": sub["abs_gap_pct"].std(),
                "tsplib_n": len(sub),
                "tsplib_time_ms": sub["total_time_s"].mean() * 1000,
            }

    # Synthetic 2D
    if "synth_2d" in data and not data["synth_2d"].empty:
        synth = data["synth_2d"]
        for model in synth["model"].unique():
            sub = synth[synth.model == model]
            if model in models_info:
                models_info[model]["synth2d_mape"] = sub["abs_gap_pct"].mean()
                models_info[model]["synth2d_n"] = len(sub)

    # Synthetic ND - uses gap_pct not abs_gap_pct
    if "synth_nd" in data and not data["synth_nd"].empty:
        synth = data["synth_nd"]
        # Create abs_gap_pct column
        if "gap_pct" in synth.columns and "abs_gap_pct" not in synth.columns:
            synth = synth.copy()
            synth["abs_gap_pct"] = synth["gap_pct"].abs()
            data["synth_nd"] = synth

        for model in synth["model"].unique():
            sub = synth[synth.model == model]
            if model in models_info:
                models_info[model]["nd_mape"] = sub["abs_gap_pct"].mean()
                models_info[model]["nd_dimensions"] = (
                    synth["dimension"].unique().tolist()
                )

    return models_info


def plot_frontier_analysis(tsplib, output_dir):
    """Create frontier analysis visualization."""
    lgbm = tsplib[tsplib.model == "LGBM_V3"].copy()
    mst = tsplib[tsplib.model == "MST_Ratio"].set_index("instance")
    lgbm_idx = lgbm.set_index("instance")
    common = lgbm_idx.index.intersection(mst.index)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) LGBM vs MST scatter
    ax = axes[0]
    ax.scatter(
        mst.loc[common, "abs_gap_pct"],
        lgbm_idx.loc[common, "abs_gap_pct"],
        alpha=0.6,
        s=30,
        c="#2196F3",
        edgecolors="k",
        linewidth=0.3,
    )
    max_val = max(
        mst.loc[common, "abs_gap_pct"].max(), lgbm_idx.loc[common, "abs_gap_pct"].max()
    )
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Equal")
    ax.set_xlabel("MST Ratio MAPE (%)")
    ax.set_ylabel("GART 3.0 MAPE (%)")
    ax.set_title("(a) GART 3.0 vs MST Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) By instance size
    ax = axes[1]
    sizes = ["tiny", "small", "medium", "large", "xlarge"]
    lgbm_mapes = []
    mst_mapes = []
    for label in sizes:
        key = f"size_{label}"
        if key in tsplib.columns:
            sub = tsplib[tsplib[key] == True] if key in tsplib.columns else None

    # Simplified: plot by n bins
    n_bins = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 100000)]
    n_labels = ["≤50", "50-100", "100-200", "200-500", "500-1K", ">1K"]
    lgbm_by_n = []
    mst_by_n = []

    for (lo, hi), label in zip(n_bins, n_labels):
        mask = (lgbm["n"] >= lo) & (lgbm["n"] < hi)
        if mask.sum() > 0:
            sub_lgbm = lgbm[mask]
            sub_mst = tsplib[
                (tsplib.model == "MST_Ratio") & (tsplib.n >= lo) & (tsplib.n < hi)
            ]
            lgbm_by_n.append(sub_lgbm["abs_gap_pct"].mean())
            mst_by_n.append(sub_mst["abs_gap_pct"].mean() if len(sub_mst) > 0 else 0)
        else:
            lgbm_by_n.append(0)
            mst_by_n.append(0)

    x = np.arange(len(n_labels))
    width = 0.35
    ax.bar(x - width / 2, lgbm_by_n, width, label="GART 3.0", color="#2196F3")
    ax.bar(x + width / 2, mst_by_n, width, label="MST Ratio", color="#9E9E9E")
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels, rotation=45)
    ax.set_xlabel("Instance Size (n)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(b) Accuracy by Instance Size")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # (c) Win rate pie chart
    ax = axes[2]
    mst_gap = mst.loc[common, "abs_gap_pct"]
    lgbm_gap = lgbm_idx.loc[common, "abs_gap_pct"]
    wins = (lgbm_gap < mst_gap).sum()
    losses = (lgbm_gap > mst_gap).sum()
    sizes_pie = [wins, losses]
    labels_pie = [f"Wins\n{wins}", f"Losses\n{losses}"]
    colors = ["#4CAF50", "#F44336"]
    ax.pie(
        sizes_pie, labels=labels_pie, colors=colors, autopct="%1.0f%%", startangle=90
    )
    ax.set_title(f"(c) GART 3.0 Win Rate\n({wins + losses} instances)")

    plt.tight_layout()
    fig.savefig(output_dir / "frontier_detailed_analysis.png", dpi=200)
    plt.close(fig)
    print(f"Saved: frontier_detailed_analysis.png")


def plot_pareto_frontier(data, output_dir):
    """Create speedup vs accuracy Pareto frontier."""
    if "tsplib" not in data or data["tsplib"].empty:
        return

    tsplib = data["tsplib"]
    concorde = data.get("concorde", pd.DataFrame())

    # Get model averages
    model_stats = (
        tsplib.groupby("model")
        .agg(
            {
                "abs_gap_pct": "mean",
                "total_time_s": lambda x: x.mean() * 1000,  # Convert to ms
            }
        )
        .reset_index()
    )

    # Add Concorde speedup if available (just get median time, don't merge on model)
    if not concorde.empty and "concorde_time_s" in concorde.columns:
        median_concorde = concorde["concorde_time_s"].median()
        model_stats["concorde_median_time"] = median_concorde

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = {
        "LGBM_V3": "#2196F3",
        "Interp_V3": "#9C27B0",
        "Linear_V3": "#4CAF50",
        "GART_1.0": "#FF5722",
        "Fixed_Alpha": "#FF9800",
        "MST_Ratio": "#607D8B",
        "BHH": "#E91E63",
        "Cavdar": "#795548",
        "Chien": "#00BCD4",
        "Composite": "#673AB7",
        "Hilbert": "#CDDC39",
        "Vinel": "#F06292",
    }

    markers = {
        "LGBM_V3": "o",
        "Interp_V3": "D",
        "Linear_V3": "s",
        "GART_1.0": "^",
        "Fixed_Alpha": "X",
        "MST_Ratio": "*",
        "BHH": "v",
        "Cavdar": "<",
        "Chien": ">",
        "Composite": "p",
        "Hilbert": "h",
        "Vinel": "P",
    }

    for _, row in model_stats.iterrows():
        model = row["model"]
        mape = row["abs_gap_pct"]
        time_ms = row["total_time_s"]

        color = colors.get(model, "#9E9E9E")
        marker = markers.get(model, "o")

        ax.scatter(
            time_ms,
            mape,
            s=150,
            c=color,
            marker=marker,
            edgecolors="k",
            linewidth=0.5,
            zorder=5,
            label=model,
        )

        ax.annotate(
            model,
            (time_ms, mape),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Average Inference Time (ms)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Speedup vs Accuracy Pareto Frontier\n(TSPLIB95)")
    ax.grid(True, alpha=0.3)

    # Add Pareto region indicator
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.3, label="10% MAPE threshold")
    ax.axvline(x=100, color="blue", linestyle="--", alpha=0.3, label="100ms threshold")

    plt.tight_layout()
    fig.savefig(output_dir / "pareto_frontier.png", dpi=200)
    plt.close(fig)
    print(f"Saved: pareto_frontier.png")


def plot_dimension_analysis(synth_nd, output_dir):
    """Plot performance by dimension for ND benchmarks."""
    if synth_nd.empty or "dimension" not in synth_nd.columns:
        return

    # Focus on LGBM models
    lgbm_models = [m for m in synth_nd["model"].unique() if "LGBM" in m or "GART" in m]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) MAPE by dimension
    ax = axes[0]
    for model in lgbm_models[:5]:
        sub = synth_nd[synth_nd.model == model]
        by_dim = sub.groupby("dimension")["abs_gap_pct"].mean()
        ax.plot(by_dim.index, by_dim.values, "o-", label=model, markersize=6)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(a) MAPE by Dimension")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) MAPE by dimension and n
    ax = axes[1]
    if "n_customers" in synth_nd.columns:
        for dim in sorted(synth_nd["dimension"].unique())[:4]:
            sub = synth_nd[synth_nd.dimension == dim]
            by_n = sub.groupby("n_customers")["abs_gap_pct"].mean()
            ax.plot(by_n.index, by_n.values, "o-", label=f"D={dim}", markersize=5)

    ax.set_xlabel("Number of Customers (n)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(b) MAPE by Size for Different Dimensions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    fig.savefig(output_dir / "dimension_analysis.png", dpi=200)
    plt.close(fig)
    print(f"Saved: dimension_analysis.png")


def generate_latex_tables(models_info, frontier_stats, output_dir):
    """Generate LaTeX tables for the paper."""

    # Table 1: Model comparison
    table1 = "\\begin{table}[!ht]\n\\centering\n\\caption{Model Performance Comparison}\n\\label{tab:model_comparison}\n\\resizebox{\\textwidth}{!}{\%\n\\begin{tabular}{l c c c c}\n\\toprule\nModel & TSPLIB MAPE (\\%) & Time (ms) & synth-2D MAPE & ND MAPE \\\\\n\\midrule\n"

    for model, info in sorted(
        models_info.items(), key=lambda x: x[1].get("tsplib_mape", 999)
    ):
        mape = info.get("tsplib_mape", "—")
        time_ms = info.get("tsplib_time_ms", 0)
        s2d = info.get("synth2d_mape", "—")
        nd = info.get("nd_mape", "—")

        if isinstance(mape, float):
            mape = f"{mape:.2f}"
        if isinstance(time_ms, float):
            time_ms = f"{time_ms:.1f}"
        if isinstance(s2d, float):
            s2d = f"{s2d:.2f}"
        if isinstance(nd, float):
            nd = f"{nd:.2f}"

        table1 += f"{model} & {mape} & {time_ms} & {s2d} & {nd} \\\\\n"

    table1 += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}"

    with open(output_dir / "model_comparison.tex", "w") as f:
        f.write(table1)
    print(f"Saved: model_comparison.tex")

    # Table 2: Frontier summary
    fs = frontier_stats.get("overall", {})
    table2 = f"""\\begin{{table}}[ht]
\\centering
\\caption{{Frontier Analysis: GART 3.0 vs MST Ratio}}
\\label{{tab:frontier}}
\\begin{{tabular}}{{l r}}
\\toprule
Metric & Value \\\\
\\midrule
Total Instances & {fs.get("wins", 0) + fs.get("losses", 0)} \\\\
GART 3.0 Wins & {fs.get("wins", 0)} ({fs.get("win_rate", 0):.0f}\\%) \\\\
GART 3.0 Losses & {fs.get("losses", 0)} \\\\
GART 3.0 MAPE & {fs.get("lgbm_mape", 0):.2f}\\% \\\\
MST Ratio MAPE & {fs.get("mst_mape", 0):.2f}\\% \\\\
Improvement & +{fs.get("improvement_pp", 0):.2f}pp \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    with open(output_dir / "frontier_summary.tex", "w") as f:
        f.write(table2)
    print(f"Saved: frontier_summary.tex")


def main():
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK ANALYSIS FOR GART 3.0")
    print("=" * 70)

    # Load data
    data = load_all_data()

    # Compute frontier statistics
    if "tsplib" in data:
        frontier_stats = compute_frontier_stats(data["tsplib"])
        print("\n=== Frontier Statistics ===")
        for key, val in frontier_stats.items():
            if isinstance(val, dict):
                print(f"{key}: {val}")
    else:
        frontier_stats = {}

    # Model comparison
    models_info = create_model_comparison_table(data)
    print("\n=== Model Comparison ===")
    for model, info in sorted(
        models_info.items(), key=lambda x: x[1].get("tsplib_mape", 999)
    )[:5]:
        print(
            f"{model}: MAPE={info.get('tsplib_mape', 'N/A'):.2f}%, Time={info.get('tsplib_time_ms', 0):.1f}ms"
        )

    # Generate figures
    print("\n=== Generating Figures ===")
    if "tsplib" in data:
        plot_frontier_analysis(data["tsplib"], FIGURES_DIR)
        plot_pareto_frontier(data, FIGURES_DIR)

    if "synth_nd" in data:
        plot_dimension_analysis(data["synth_nd"], FIGURES_DIR)

    # Generate LaTeX tables
    print("\n=== Generating LaTeX Tables ===")
    generate_latex_tables(models_info, frontier_stats, FIGURES_DIR)

    print("\n" + "=" * 70)
    print("Analysis complete. All outputs saved to:", FIGURES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
