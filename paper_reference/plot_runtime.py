"""Runtime plot for GART 2.0 vs Concorde on TSPLIB95.

Bins instances by n, plots median GART (feature_time + inference_time stacked)
against median Concorde solve time on a log y-axis. Instances with
concorde_time_s > 3600 are drawn as a ">1 h" annotation since the solver was
not measured past that cutoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TSPLIB_CSV = REPO_ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
CONCORDE_CSV = REPO_ROOT / "tsplib_benchmark" / "concorde_solve_times.csv"
OUT_PNG = HERE / "plot_runtime.png"

BINS = [10, 50, 200, 400, 1500, np.inf]
LABELS = ["10-50", "51-200", "201-400", "401-1500", "1501+"]
CUTOFF_HOURS = 1.0


def load() -> pd.DataFrame:
    df = pd.read_csv(TSPLIB_CSV)
    df = df[df.model == "LGBM_V3"]
    df = df[df.status == "ok"].copy()
    if CONCORDE_CSV.exists():
        ct = pd.read_csv(CONCORDE_CSV).set_index("instance")["concorde_time_s"]
        df["concorde_time_s"] = df["instance"].map(ct)
    df["n_bin"] = pd.cut(df["n"], bins=[-0.1] + BINS, labels=LABELS, right=True)
    return df


def main() -> None:
    if not TSPLIB_CSV.exists():
        sys.exit(f"Missing {TSPLIB_CSV}")
    df = load()

    agg = df.groupby("n_bin", observed=True).agg(
        n_count=("instance", "count"),
        feat_median=("feature_time_s", "median"),
        inf_median=("inference_time_s", "median"),
        concorde_median=("concorde_time_s", "median"),
    )
    agg = agg.reindex(LABELS).dropna(subset=["feat_median"])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(agg))
    width = 0.38

    ax.bar(x - width / 2, agg["feat_median"], width,
           label="GART: feature extraction", color="#2ba3a3")
    ax.bar(x - width / 2, agg["inf_median"], width,
           bottom=agg["feat_median"],
           label="GART: inference", color="#8fd4d4")

    concorde_medians = agg["concorde_median"].fillna(CUTOFF_HOURS * 3600.0)
    bar_conc = ax.bar(x + width / 2, concorde_medians, width,
                      label="Concorde: solve", color="#7f7f7f")

    for idx, val in enumerate(agg["concorde_median"]):
        if np.isnan(val):
            ax.text(x[idx] + width / 2, CUTOFF_HOURS * 3600.0 * 1.15,
                    ">1 h (not measured)", ha="center", va="bottom",
                    fontsize=8, color="#444", rotation=0)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=0)
    ax.set_xlabel("TSPLIB instance size (n)")
    ax.set_ylabel("Median wall time (seconds, log scale)")
    ax.set_title("Runtime: GART 2.0 (stacked: features + inference) vs Concorde")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
