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
    ("Asymptotic_MST", "Asymptotic MST ratio"),
    ("Calibrated_MST_dn", r"Calibrated MST ratio $\hat\rho(d,n)$"),
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
# Y-axis: one signed square-root ruler, identical on all three panels.
#
# These rosters span too wide a dynamic range for a linear axis and sit too
# close to zero for a logarithmic one.  Measured interquartile ranges, in
# percentage points, over the paired samples the panels actually draw:
#
#     2D      0.87 (Held-Karp) .. 16.62 (Hilbert)      19.1 : 1
#     ND      0.54 (Held-Karp) .. 22.09 (Hilbert)      41.1 : 1
#     TSPLIB  1.08 (Held-Karp) .. 14.01 (Hilbert)      12.9 : 1
#
# with a pooled whisker envelope of -41.35 .. +64.87.
#
# A linear axis over that envelope turns the tightest boxes into hairlines:
# GART 2.0's 2.81 pp box on the 2D panel is 3% of the axis and its 0.81 pp box
# on the multidimensional panel is 1%.  Those are the boxes the paper exists to
# show, and in the superseded linear panels the median could not be told from
# the quartile edges on either GART 2.0 or the bound.
#
# A logarithmic or symmetric-log axis fails the other way.  Height on a log arm
# is a *ratio*, not a dispersion, so a box straddling zero stretches without
# bound while a box far from zero is crushed.  Under the superseded symlog
# treatment the multidimensional BHH box (6.96 pp, from -31.72 to -24.76, a
# ratio of only 1.28) renders shorter than the GART 2.0 box (0.81 pp), which
# inverts the ranking a boxplot is read for.  Lowering the threshold below the
# smallest interquartile range does not repair this: a box that straddles zero
# always contains the linear core, so it cannot be moved onto the log arm at
# any threshold, and shrinking the core only stretches it further.
#
# sign(x) * sqrt(|x|) is the middle term.  It is smooth and strictly increasing
# through zero, so it is a single ruler for the whole panel with no linear/log
# seam for a box to straddle; it preserves sign and order exactly; and it
# compresses gently enough that box height still tracks dispersion.  It takes
# the multidimensional panel's 41:1 spread to 3.8:1 on the page and the 2D
# panel's 19:1 to 5.8:1, leaving every box legible without reordering any of
# them.  Two percent and twenty percent land a fixed 18% of the axis apart on
# every panel, so the same visual gap means the same thing in all three
# figures.
#
# Ticks are labelled at real percentages, not at roots, and the tick set and
# the limits are shared by all three panels so the reader learns the axis once.
# YLIM clears the largest whisker in any panel (64.87), so nothing is clipped
# on any figure and no caption has to carry a clipping caveat.
#
# Unlike a log axis, a root axis *compresses* towards zero, so the ticks have to
# thin out there rather than bunch up: 1 and 2 are only 0.41 root-units apart
# out of the 16.73 the axis spans, which is 2.5% of its height, or under 5 pt at
# the printed size against a 7.5 pt label.  Adding 0.5 and 2 collides the labels
# outright.  The set below keeps every neighbouring pair at least 5.5% of the
# axis apart (the tightest is 5 to 10), which is about 10 pt in print.
YLIM = 70.0
ROOT_TICKS = [-60, -40, -20, -10, -5, -1, 0, 1, 5, 10, 20, 40, 60]


def _signed_root(x):
    """Forward transform of the y-axis: sign-preserving square root."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.sqrt(np.abs(x))


def _signed_square(u):
    """Inverse of :func:`_signed_root`."""
    u = np.asarray(u, dtype=float)
    return np.sign(u) * np.square(u)

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


def _draw_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    models: list[tuple[str, str]],
    title: str,
) -> None:
    labels = [label for _, label in models]
    data = [df.loc[df["model"] == key, "gap_pct"].to_numpy(dtype=float) for key, _ in models]
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        widths=0.62,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "#444444", "linewidth": 0.9},
        capprops={"color": "#444444", "linewidth": 0.9},
        boxprops={"edgecolor": "#333333", "linewidth": 0.8},
    )
    for index, (box, label) in enumerate(zip(bp["boxes"], labels)):
        box.set_facecolor(COLORS.get(label, DEFAULT_COLOR))
        box.set_alpha(0.82)
        box.set_hatch(HATCHES[index % len(HATCHES)])

    # One ruler, the same on every panel.  See the YLIM / ROOT_TICKS block above
    # for why it is neither linear nor logarithmic.  Nothing may fall outside
    # YLIM: silent clipping of a whisker would misstate a model's tail, and the
    # captions state no clipping caveat, so assert rather than trusting that the
    # envelope stays where it was measured.
    reach = max(
        max(abs(float(np.percentile(s, 5))), abs(float(np.percentile(s, 95))))
        for s in data
    )
    if reach > YLIM:
        raise ValueError(
            f"whisker reach {reach:.2f} exceeds YLIM={YLIM:g} on '{title}'; "
            "raise YLIM (and revisit the captions) rather than clipping"
        )
    ax.set_yscale("function", functions=(_signed_root, _signed_square))
    ax.set_ylim(-YLIM, YLIM)
    ax.set_yticks(ROOT_TICKS)
    ax.set_yticklabels([f"{t:g}" for t in ROOT_TICKS])
    ax.minorticks_off()
    ylabel = "Signed percent error (%)\nsigned square-root scale"

    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_title(title, loc="center", fontweight="bold", pad=8)
    ax.set_ylabel(ylabel)
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
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _draw_boxplot(ax, paired, FULL_2D, f"2D diverse synthetic benchmark (N = {n:,} per estimator)")
    _save(fig, "boxplot_2d_errors")


def plot_nd(df: pd.DataFrame) -> None:
    df = pd.concat([df, _hk_rows(PATH_HK_ND)], ignore_index=True)
    paired, n = _paired_subset(df, FULL_ND)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _draw_boxplot(ax, paired, FULL_ND,
                  f"Multidimensional benchmark (N = {n:,} per estimator)")
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
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _draw_boxplot(ax, paired, FULL_TSPLIB, f"TSPLIB95 EUC_2D benchmark (N = {n} per estimator)")
    _save(fig, "boxplot_tsplib_errors")


def main() -> None:
    plot_2d(_load(PATH_2D))
    plot_nd(_load(PATH_ND))
    plot_tsplib(_load(PATH_TSPLIB))


if __name__ == "__main__":
    main()
