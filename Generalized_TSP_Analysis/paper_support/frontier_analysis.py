"""
Cross-benchmark frontier analysis for the paper.

Reads the canonical 2D and ND benchmark CSVs produced by the
Generalized_TSP_Analysis pipelines and computes per-model MAPE/gap metrics
plus an MST-ratio frontier comparison (baseline = MST_Ratio * 1.22 vs GART).

Outputs
-------
Generalized_TSP_Analysis/paper_support/per_model_metrics.csv
Generalized_TSP_Analysis/paper_support/frontier_results.csv
"""
from pathlib import Path
import re
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
TWO_D_CSV = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
ND_CSV = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"

_INST_RE = re.compile(r"TSP-boundary-n(\d+)-g(\d+)-(\d+)")


def parse_2d_instance(inst):
    m = _INST_RE.match(str(inst))
    if not m:
        return None, None, None
    return int(m.group(1)), 2, int(m.group(2))


def compute_metrics(group):
    mape = mean_absolute_percentage_error(group["true_cost"], group["pred_cost"]) * 100
    return pd.Series({
        "MAPE": mape,
        "Mean_Gap": group["gap_pct"].mean(),
        "Mean_Abs_Gap": group["abs_gap_pct"].mean(),
        "Count": len(group),
    })


def add_frontier_metrics(df):
    df_mst = df[df["model"] == "MST_Ratio"]
    df_models = df[df["model"] != "MST_Ratio"]
    merged = pd.merge(
        df_models, df_mst[["instance", "pred_cost"]],
        on="instance", suffixes=("", "_mst"),
    )
    merged["baseline_mape"] = (
        (merged["true_cost"] - merged["pred_cost_mst"] * 1.22).abs()
        / merged["true_cost"]
    ) * 100
    merged["gart_mape"] = (
        (merged["true_cost"] - merged["pred_cost"]).abs() / merged["true_cost"]
    ) * 100
    merged["n_group"] = pd.cut(
        merged["n_customers"],
        bins=[0, 100, 1000, 100_000],
        labels=["Small", "Medium", "Large"],
    )
    return (
        merged.groupby(["model", "n_group"], observed=True)[["baseline_mape", "gart_mape"]]
        .mean()
        .reset_index()
    )


def load_combined():
    df_2d = pd.read_csv(TWO_D_CSV)
    df_nd = pd.read_csv(ND_CSV)
    df_2d[["n_customers", "dimension", "grid_size"]] = df_2d["instance"].apply(
        lambda x: pd.Series(parse_2d_instance(x))
    )
    df_2d["distribution"] = "boundary"
    df_nd["abs_gap_pct"] = df_nd["gap_pct"].abs()
    return pd.concat([df_2d, df_nd], ignore_index=True)


def main():
    df = load_combined()
    per_model = (
        df.groupby(["model", "dimension", "n_customers"])
        .apply(compute_metrics, include_groups=False)
        .reset_index()
    )
    frontier = add_frontier_metrics(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_model.to_csv(OUT_DIR / "per_model_metrics.csv", index=False)
    frontier.to_csv(OUT_DIR / "frontier_results.csv", index=False)

    print("Frontier Analysis Results:")
    print(frontier.to_string(index=False))
    print(f"\nWrote: {OUT_DIR / 'per_model_metrics.csv'}")
    print(f"Wrote: {OUT_DIR / 'frontier_results.csv'}")


if __name__ == "__main__":
    main()
