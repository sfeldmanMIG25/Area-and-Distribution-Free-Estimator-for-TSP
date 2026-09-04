"""Regenerate the publication figures used by ``Area_Free_Main.tex``.

Each figure uses a common, finite-prediction instance set so its title and
every box always refer to the same sample.  Chien (1992), Kwon et al. (1995)
and Daganzo (1984) appear in no figure and in no table: their primaries are
paywalled with no obtainable open-access copy, the coefficients we had came
from a secondary transcription, and the paper prints no number that rests on
them.  Of the classical estimators only BHH and Cavdar--Sokol are scored, and
they appear in Table~\ref{tab:classical} rather than in these boxplots, whose
axes their errors would dominate.

Usage:
    py -3.14 paper_reference/regenerate_boxplots.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from paper_tooling.model_registry import GART  # noqa: E402

PATH_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
PATH_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
PATH_TSPLIB = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"

# Held--Karp 1-tree ladders. Same instance keys as the benchmark CSVs above and
# the same ``true_cost`` to machine precision (verified: max relative difference
# 0.0 on all three corpora), so the bound's signed error is on the identical
# denominator as every estimator box beside it.
PATH_HK_2D = ROOT / "paper_tooling" / "hk1tree_frontier_2d.csv"
PATH_HK_ND = ROOT / "paper_tooling" / "polyak_nd_sweep.csv"
PATH_HK_TSPLIB = ROOT / "paper_tooling" / "hk1tree_frontier_tsplib.csv"

# The ascent budget the bound is drawn at. 100 is the rung the manuscript's two
# ladders turn on: on the multidimensional benchmark it is where the bound first
# overtakes GART 2.0, and on TSPLIB it is where the raw certificate beats it in
# the smallest size bucket at under half the cost. One budget across all three
# panels so a reader compares boxes and not budgets.
HK_BUDGET = 100
HK_KEY = "HK_1tree"
HK_LABEL = r"Held–Karp 1-tree bound ($k{=}100$)"

# (benchmark-CSV key, box label). The keys come from the registry so a model
# swap does not have to be repeated here; the box labels stay short because
# these are figure legends, not table rows.
#
# Each roster is exactly the estimator roster of the corresponding manuscript
# table plus the certified bound: Table~\ref{tab:2d_by_size} for FULL_2D,
# Table~\ref{tab:nd_by_dim} for FULL_ND, Table~\ref{tab:tsplib_by_size} for
# FULL_TSPLIB. ``NN_V3`` and ``Linear_V3`` are deliberately absent -- they were
# withdrawn from the manuscript (they are fitted on 30 and 28 columns against
# GART 2.0's 31, so they are not controls on the same feature vector), and the
# figures outlived them by one revision. ``Calibrated_MST_d`` is absent for the
# same reason: Section~\ref{subsec:bench_models} states explicitly that
# $\hat\rho(d)$ is not printed as a baseline. The earlier boosted model on the
# 30-feature block (``model_registry.PREDECESSOR``) is absent for the same
# reason again: it carries no ``MODEL_LABELS`` entry, so it is in no manuscript
# table, and a box with no row beside it is a model the reader cannot look up.
GART_2_0 = (GART, "GART 2.0")
HK_BOUND = (HK_KEY, HK_LABEL)

FULL_2D = [
    GART_2_0,
    HK_BOUND,
    ("GART", "GART 1.0"),
    ("Calibrated_MST_dn", r"Calibrated MST ratio $\hat\rho(d,n)$"),
    ("Asymptotic_MST", "Asymptotic MST ratio"),
    ("MST_Only", r"$L_{\mathrm{MST}}$ ($\alpha=1$)"),
    ("Hilbert", "Custom Hilbert sort"),
]
FULL_ND = [
    GART_2_0,
    HK_BOUND,
    ("Calibrated_MST_dn", r"Calibrated MST ratio $\hat\rho(d,n)$"),
    ("MST_Only", r"$L_{\mathrm{MST}}$ ($\alpha=1$)"),
    ("BHH_region", "BHH (sampling region)"),
    ("Hilbert", "Custom Hilbert sort"),
]
FULL_TSPLIB = [
    GART_2_0,
    HK_BOUND,
    ("GART_1.0", "GART 1.0"),
    ("Calibrated_MST_dn", r"Calibrated MST ratio $\hat\rho(d,n)$"),
    ("Asymptotic_MST", "Asymptotic MST ratio"),
    ("MST_Only", r"$L_{\mathrm{MST}}$ ($\alpha=1$)"),
    ("Hilbert", "Custom Hilbert sort"),
]

COLORS = {
    "GART 2.0": "#0077BB",
    "GART 1.0": "#009988",
    HK_LABEL: "#AA3377",
    r"Calibrated MST ratio $\hat\rho(d,n)$": "#EE7733",
    "Asymptotic MST ratio": "#CC3311",
}
DEFAULT_COLOR = "#BBBBBB"
HATCHES = ("", "//", "\\\\", "..", "xx", "++", "oo", "--")

# ---------------------------------------------------------------------------
# Y-axis: linear, standard Tukey boxes (1.5 IQR whiskers, fliers drawn), one
# window per panel.  Author decision 2026-09-03.
#
# The earlier signed-square-root ruler shared by all three panels needed three
# conventions (5/95 whiskers, hidden fliers, the root transform) that no
# caption or body sentence stated.  A Tukey box on a linear axis needs none.
# The price is dynamic range: the rosters' 1.5 IQR whiskers span -42 .. +71
# while the boxes the paper exists to show are 0.8 to 6 pp tall, so a window
# wide enough for every whisker turns GART 2.0 into a hairline.  Each panel is
# therefore windowed to its own contenders and the weak rows run off the
# frame.  Clipping is never silent: every clipped side carries a marker at the
# frame edge with the value the tail reaches, and a box that lies wholly
# outside the window prints its quartiles, so the figure certifies its own
# clipping and the captions carry no caveat.
#
# Windows are (low, high) in percent.  1.5 IQR whisker reach of the contenders,
# measured 2026-09-03 over the paired samples the panels draw:
#     ND      GART 2.0 -1.4..+1.9   bound -1.4..0   rho(d,n) -2.5..+3.4
#     2D      GART 2.0 -5.3..+5.9   bound -2.4..0   rho(d,n) -9.5..+8.6
#             asymptotic -27.8..+11.3   GART 1.0 -14.1..+23.4
#     TSPLIB  GART 2.0 -3.4..+7.2   bound -3.0..+0.7  rho(d,n) -9.9..+11.6
#             asymptotic -9.9..+9.0     GART 1.0 -7.4..+22.3
WINDOW_2D = (-30.0, 40.0)
WINDOW_ND = (-8.0, 8.0)
WINDOW_TSPLIB = (-22.0, 30.0)

# Authored at the printed size.  These used to be 10.2 inches wide and were
# then set at 0.86--0.96\textwidth, so pdflatex shrank them by about 1.7 and an
# 8 pt tick label reached the page at under 5 pt.  Drawing at the text width
# (about 6.5 in for this class) and setting them at \textwidth means the point
# sizes below are the point sizes the reader gets.
FIG_W = 6.8
FIG_H = 4.1

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "gap_pct" not in df.columns:
        df["gap_pct"] = 100.0 * (df["pred_cost"] - df["true_cost"]) / df["true_cost"]
    finite = np.isfinite(pd.to_numeric(df["gap_pct"], errors="coerce"))
    return df.loc[finite].copy()


def _hk_rows(path: Path) -> pd.DataFrame:
    """The certified 1-tree bound at ``HK_BUDGET``, shaped like a model's rows.

    Returned with the same three columns the boxplot reads -- ``model``,
    ``instance``, ``gap_pct`` -- so it joins the benchmark frame without any
    special case downstream.  ``gap_pct`` is computed against the ladder's own
    ``true_cost`` column, which equals the benchmark CSV's to machine precision.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["k"] == HK_BUDGET) & (df["status"] == "ok")].copy()
    if df.empty:
        raise ValueError(f"No k={HK_BUDGET} rows with status 'ok' in {path}")
    out = pd.DataFrame(
        {
            "model": HK_KEY,
            "instance": df["instance"].to_numpy(),
            "gap_pct": 100.0 * (df["bound"] - df["true_cost"]) / df["true_cost"],
        }
    )
    return out.loc[np.isfinite(out["gap_pct"])].copy()


def _paired_subset(
    df: pd.DataFrame,
    models: list[tuple[str, str]],
    instances: set[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Return rows for an exact common instance set and verify equal coverage."""
    keys = [key for key, _ in models]
    work = df[df["model"].isin(keys)].copy()
    duplicates = work.duplicated(subset=["model", "instance"], keep=False)
    if duplicates.any():
        pairs = work.loc[duplicates, ["model", "instance"]].drop_duplicates()
        raise ValueError(f"Duplicate model-instance rows: {pairs.to_dict('records')}")
    if instances is None:
        coverage = [set(work.loc[work["model"] == key, "instance"]) for key in keys]
        instances = set.intersection(*coverage)
    work = work[work["instance"].isin(instances)].copy()
    counts = work.groupby("model")["instance"].nunique().reindex(keys)
    if counts.isna().any() or counts.nunique() != 1 or int(counts.iloc[0]) != len(instances):
        raise ValueError(f"Unpaired model coverage: {counts.to_dict()}")
    return work, len(instances)


def _clip_marks(ax: plt.Axes, data: list[np.ndarray], window: tuple[float, float]) -> None:
    """Mark every side of every box the window cuts off.

    A whisker that meets the frame edge would read as ending there.  Each
    clipped side gets a triangle on the edge and the value the tail reaches;
    a box lying wholly outside the window also prints its quartiles, since the
    reader would otherwise see an empty column.
    """
    low, high = window
    span = high - low
    inset = 0.012 * span
    style = {
        "fontsize": 6.3,
        "ha": "center",
        "color": "#222222",
        "zorder": 7,
        "bbox": {"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    }

    def num(x: float) -> str:
        return f"{x:.0f}".replace("-", "−")

    for index, sample in enumerate(data, start=1):
        q1, q3 = np.percentile(sample, [25, 75])
        lo, hi = float(sample.min()), float(sample.max())
        box_text = f"box {num(q1)} to {num(q3)}"
        if hi > high:
            text = num(hi) if q1 <= high else f"{box_text}\nmax {num(hi)}"
            ax.plot(index, high, marker="^", color="#222222", markersize=4, clip_on=False, zorder=6)
            ax.text(index, high - inset, text, va="top", **style)
        if lo < low:
            text = num(lo) if q3 >= low else f"{box_text}\nmin {num(lo)}"
            ax.plot(index, low, marker="v", color="#222222", markersize=4, clip_on=False, zorder=6)
            ax.text(index, low + inset, text, va="bottom", **style)


def _draw_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    models: list[tuple[str, str]],
    window: tuple[float, float],
) -> None:
    labels = [label for _, label in models]
    data = [df.loc[df["model"] == key, "gap_pct"].to_numpy(dtype=float) for key, _ in models]
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        whis=1.5,
        showfliers=True,
        widths=0.62,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "#444444", "linewidth": 0.9},
        capprops={"color": "#444444", "linewidth": 0.9},
        boxprops={"edgecolor": "#333333", "linewidth": 0.8},
        flierprops={
            "marker": ".",
            "markersize": 2.2,
            "markerfacecolor": "#666666",
            "markeredgecolor": "none",
            "alpha": 0.35,
        },
    )
    for index, (box, label) in enumerate(zip(bp["boxes"], labels)):
        box.set_facecolor(COLORS.get(label, DEFAULT_COLOR))
        box.set_alpha(0.82)
        box.set_hatch(HATCHES[index % len(HATCHES)])

    ax.set_ylim(*window)
    _clip_marks(ax, data, window)

    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_ylabel("Signed percent error (%)")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, linestyle=":")
    ax.tick_params(axis="x", rotation=34)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _save(fig: plt.Figure, stem: str) -> None:
    """Write the figure through a scratch name, then move it into place.

    A file watcher on this machine opens new files in ``paper_reference/``
    within about a second of their being created, and ``savefig`` onto a name
    it is already holding fails with ``OSError: [Errno 22]`` partway through a
    run -- leaving one format of one panel stale while the others are current,
    which is the worst possible outcome for a figure set that has to be read
    together. ``paper_tooling/_build_paper.py`` sidesteps the same watcher for
    the PDF build by the same means.
    """
    fig.tight_layout(pad=1.2)
    for ext in ("pdf", "png"):
        final = HERE / f"{stem}.{ext}"
        scratch = HERE / f".{stem}.tmp.{ext}"
        fig.savefig(scratch)
        for attempt in range(6):
            try:
                os.replace(scratch, final)
                break
            except OSError as exc:
                if attempt == 5:
                    scratch.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"could not move {scratch.name} onto {final.name}: {exc}"
                    ) from exc
                time.sleep(1.0)
    plt.close(fig)
    print(f"Wrote paper_reference/{stem}.pdf and .png")


def plot_2d(df: pd.DataFrame) -> None:
    df = pd.concat([df, _hk_rows(PATH_HK_2D)], ignore_index=True)
    paired, n = _paired_subset(df, FULL_2D)
    print(f"2D panel: N = {n:,} per estimator")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _draw_boxplot(ax, paired, FULL_2D, WINDOW_2D)
    _save(fig, "boxplot_2d_errors")


def plot_nd(df: pd.DataFrame) -> None:
    df = pd.concat([df, _hk_rows(PATH_HK_ND)], ignore_index=True)
    paired, n = _paired_subset(df, FULL_ND)
    print(f"ND panel: N = {n:,} per estimator")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _draw_boxplot(ax, paired, FULL_ND, WINDOW_ND)
    _save(fig, "boxplot_nd_errors")


def plot_tsplib(df: pd.DataFrame) -> None:
    df = df[df["edge_weight_type"] == "EUC_2D"].copy()
    if "mst_length" in df.columns:
        metric_rows = df[df["mst_length"].notna()].copy()
        ratio = metric_rows["true_cost"] / metric_rows["mst_length"]
        metric_instances = set(metric_rows.loc[ratio <= 2.5, "instance"])
        df = df[df["instance"].isin(metric_instances)].copy()
    df = pd.concat([df, _hk_rows(PATH_HK_TSPLIB)], ignore_index=True)

    paired, n = _paired_subset(df, FULL_TSPLIB)
    print(f"TSPLIB panel: N = {n} per estimator")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _draw_boxplot(ax, paired, FULL_TSPLIB, WINDOW_TSPLIB)
    _save(fig, "boxplot_tsplib_errors")


def main() -> None:
    plot_2d(_load(PATH_2D))
    plot_nd(_load(PATH_ND))
    plot_tsplib(_load(PATH_TSPLIB))


if __name__ == "__main__":
    main()
