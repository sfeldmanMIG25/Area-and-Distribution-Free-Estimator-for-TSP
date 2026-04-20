"""
Regenerate the three boxplot PNGs used by Area_Free_Main.tex.

Writes directly into paper_reference/ (same dir as this script and the .tex),
so Overleaf uploads are self-contained.

Usage:
    python paper_reference/regenerate_boxplots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PATH_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
PATH_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
PATH_TSPLIB = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"

# Model ordering and display names. Must match the model keys in the CSVs.
MODEL_ORDER_2D = [
    ("LGBM_V3", "GART 2.0"),
    ("GART", "GART 1.0"),
    ("MST_Ratio", "MST Ratio"),
    ("BHH", "BHH"),
    ("Cavdar", "Cavdar"),
    ("Chien", "Chien"),
    ("Daganzo", "Daganzo"),
    ("Kwon", "Kwon"),
    ("Hilbert", "Hilbert"),
]

MODEL_ORDER_ND = [
    ("LGBM_V3", "GART 2.0"),
    ("MST_Ratio", "MST Ratio"),
    ("BHH", "BHH"),
    ("Hilbert", "Hilbert"),
]

MODEL_ORDER_TSPLIB = [
    ("LGBM_V3", "GART 2.0"),
    ("GART_1.0", "GART 1.0"),
    ("MST_Ratio", "MST Ratio"),
    ("BHH", "BHH"),
    ("Cavdar", "Cavdar"),
    ("Chien", "Chien"),
    ("Daganzo", "Daganzo"),
    ("Kwon", "Kwon"),
    ("Hilbert", "Hilbert"),
]


def _load(path: Path, extra_filter=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    if extra_filter is not None:
        df = extra_filter(df)
    # Use signed gap_pct = 100 * (pred - true) / true when present; else compute.
    if "gap_pct" not in df.columns:
        df["gap_pct"] = 100.0 * (df["pred_cost"] - df["true_cost"]) / df["true_cost"]
    return df


def _tsplib_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "edge_weight_type" in df.columns:
        df = df[df["edge_weight_type"] == "EUC_2D"].copy()
    # Paper filter L_TSP/L_MST <= 2.5 using mst_length column when available.
    if "mst_length" in df.columns:
        ratio = df["true_cost"] / df["mst_length"]
        df = df[ratio <= 2.5].copy()
    return df


def make_boxplot(
    df: pd.DataFrame,
    order: list[tuple[str, str]],
    out_path: Path,
    ylim: tuple[float, float],
    title: str,
) -> None:
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for key, display in order:
        sub = df[df["model"] == key]
        if len(sub) == 0:
            continue
        groups.append(sub["gap_pct"].to_numpy())
        labels.append(f"{display}\n(N={len(sub)})")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="#888", linewidth=0.8, zorder=1)
    bp = ax.boxplot(
        groups,
        tick_labels=labels,
        showfliers=False,
        whis=(5, 95),
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(bp["boxes"])))
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.55)

    ax.set_ylim(*ylim)
    ax.set_ylabel("Signed % error  (100·(pred-true)/true)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    if PATH_2D.exists():
        df2 = _load(PATH_2D)
        make_boxplot(
            df2,
            MODEL_ORDER_2D,
            HERE / "boxplot_2d_errors.png",
            ylim=(-60.0, 60.0),
            title="2D synthetic benchmark — signed error by model",
        )
    else:
        print(f"SKIP 2D: {PATH_2D} not found")

    if PATH_ND.exists():
        dfn = _load(PATH_ND)
        make_boxplot(
            dfn,
            MODEL_ORDER_ND,
            HERE / "boxplot_nd_errors.png",
            ylim=(-60.0, 60.0),
            title="Multi-dimensional benchmark — signed error by model",
        )
    else:
        print(f"SKIP ND: {PATH_ND} not found")

    if PATH_TSPLIB.exists():
        dft = _load(PATH_TSPLIB, extra_filter=_tsplib_filter)
        make_boxplot(
            dft,
            MODEL_ORDER_TSPLIB,
            HERE / "boxplot_tsplib_errors.png",
            ylim=(-30.0, 80.0),
            title="TSPLIB95 EUC\\_2D benchmark — signed error by model",
        )
    else:
        print(f"SKIP TSPLIB: {PATH_TSPLIB} not found")


if __name__ == "__main__":
    main()
