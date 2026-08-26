"""Accuracy of GART 2.0 against one classical planar estimator and one bound.

What replaced what
------------------
The superseded version of this figure was a single log--log cost/accuracy
scatter on TSPLIB only.  It carried four competing encodings at once -- a raw
1-tree ladder, a *second* calibrated ladder, colour standing for size bucket,
and open rings standing for Pareto optimality -- so nothing in it could be read
without the key, and the key needed two boxes.  It is replaced by one question
asked twice:

    how far is each estimate from the optimal tour,
    (a) on each of the three benchmarks, and (b) as dimension grows?

Three series, one metric, no cost axis (cost is Table~\\ref{tab:frontier_nd}'s
job), and exactly one bound series: the **raw** certified Held--Karp 1-tree
lower bound after 100 ascent steps.  The calibrated variant is not drawn.  100
steps is the same rung the three boxplot figures draw the bound at, so a reader
who has seen those is looking at the same bound here.

Why Cavdar--Sokol stops at panel (a)'s first two rows
----------------------------------------------------
It is planar by construction: its area term is the rectangle covering the nodes
and its dispersion terms are per-axis about that rectangle's midpoint lines, so
``classical_region_estimators.CavdarSokol`` gates itself to ``d == 2`` and the
multidimensional runner never scores it.  There is therefore no
Cavdar--Sokol number on the multidimensional benchmark and none is drawn.  That
absence is the paper's premise, so the panel states it in words rather than
leaving a gap the reader has to explain.

Sources -- every value is recomputed here from per-instance rows
---------------------------------------------------------------
Estimators   ``Generalized_TSP_Analysis/benchmark_results_2D_v3.csv``
             ``Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv``
             ``tsplib_benchmark/results/all_models_tsplib.csv``
Bound        ``paper_tooling/hk1tree_frontier_2d.csv``      (2D)
             ``paper_tooling/polyak_nd_sweep.csv``          (multidimensional)
             ``paper_tooling/hk1tree_frontier_tsplib.csv``  (TSPLIB)

These are exactly the six inputs ``regenerate_boxplots.py`` reads, screened the
same way, so the two figures cannot disagree.  Each panel is drawn on the
instance set all of its series cover, and :func:`_check_banks` asserts the
recomputed aggregates against ``paper_tooling/polyak_nd_bank.json`` and
``paper_tooling/polyak_nd_by_dimension.csv`` before anything is written.

Usage::

    py -3.14 paper_reference/regenerate_frontier_figure.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from paper_tooling.model_registry import GART  # noqa: E402

CAVDAR = "Cavdar"

PATH_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
PATH_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
PATH_TSPLIB = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"

PATH_HK_2D = ROOT / "paper_tooling" / "hk1tree_frontier_2d.csv"
PATH_HK_ND = ROOT / "paper_tooling" / "polyak_nd_sweep.csv"
PATH_HK_TSPLIB = ROOT / "paper_tooling" / "hk1tree_frontier_tsplib.csv"

BANK_ND = ROOT / "paper_tooling" / "polyak_nd_bank.json"
BY_DIM_ND = ROOT / "paper_tooling" / "polyak_nd_by_dimension.csv"

# Ascent budget the bound is drawn at, shared with the three boxplot figures.
HK_BUDGET = 100

# Palette carried over from regenerate_boxplots.py so GART 2.0 and the bound
# keep the colour they have in the sibling figures.  Marker shape, not colour,
# separates the three series, so the figure survives greyscale printing.
C_GART = "#0077BB"
C_CAVDAR = "#EE7733"
C_BOUND = "#AA3377"
GREY = "#555555"

L_GART = "GART 2.0"
L_CAVDAR = "Çavdar–Sokol"
L_BOUND = f"Held–Karp lower bound ({HK_BUDGET} ascent steps)"

SERIES = [
    (L_GART, C_GART, "o", 6.4),
    (L_CAVDAR, C_CAVDAR, "s", 6.0),
    (L_BOUND, C_BOUND, "D", 5.6),
]

FIG_W = 6.8
FIG_H = 3.9

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
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Loading


def _estimator_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "abs_gap_pct" not in df.columns:
        df["abs_gap_pct"] = 100.0 * (df["pred_cost"] - df["true_cost"]).abs() / df["true_cost"]
    finite = np.isfinite(pd.to_numeric(df["abs_gap_pct"], errors="coerce"))
    return df.loc[finite].copy()


def _bound_rows(path: Path) -> pd.DataFrame:
    """The certified 1-tree bound at ``HK_BUDGET``, as ``instance``/``abs_gap_pct``.

    The bound is a lower bound, so its signed error is negative on every row;
    the panels plot distance from the optimum, so the sign is taken off here and
    the axis label says "absolute".
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["k"] == HK_BUDGET) & (df["status"] == "ok")].copy()
    if df.empty:
        raise ValueError(f"no k={HK_BUDGET} rows with status 'ok' in {path}")
    out = pd.DataFrame(
        {
            "instance": df["instance"].to_numpy(),
            "abs_gap_pct": 100.0 * (df["bound"] - df["true_cost"]).abs() / df["true_cost"],
        }
    )
    return out.loc[np.isfinite(out["abs_gap_pct"])].copy()


def _tsplib_screen(df: pd.DataFrame) -> pd.DataFrame:
    """The EUC_2D + metric screen ``regenerate_boxplots.plot_tsplib`` applies."""
    df = df[df["edge_weight_type"] == "EUC_2D"].copy()
    metric_rows = df[df["mst_length"].notna()].copy()
    ratio = metric_rows["true_cost"] / metric_rows["mst_length"]
    keep = set(metric_rows.loc[ratio <= 2.5, "instance"])
    return df[df["instance"].isin(keep)].copy()


def _mape_row(
    bench: pd.DataFrame, bound: pd.DataFrame, models: list[str]
) -> tuple[dict[str, float], int]:
    """MAPE per series on the instance set every series in the row covers."""
    coverage = [set(bench.loc[bench["model"] == m, "instance"]) for m in models]
    coverage.append(set(bound["instance"]))
    common = set.intersection(*coverage)
    values = {
        m: float(
            bench.loc[
                (bench["model"] == m) & (bench["instance"].isin(common)), "abs_gap_pct"
            ].mean()
        )
        for m in models
    }
    values["bound"] = float(bound.loc[bound["instance"].isin(common), "abs_gap_pct"].mean())
    return values, len(common)


def collect() -> tuple[list[dict], pd.DataFrame]:
    """Panel (a) rows and panel (b) per-dimension frame."""
    rows = []

    v, n = _mape_row(_estimator_rows(PATH_2D), _bound_rows(PATH_HK_2D), [GART, CAVDAR])
    rows.append(
        {
            "label": "2D synthetic benchmark",
            "sub": f"{n:,} instances, $d = 2$",
            "gart": v[GART],
            "cavdar": v[CAVDAR],
            "bound": v["bound"],
            "n": n,
        }
    )

    v, n = _mape_row(_estimator_rows(PATH_ND), _bound_rows(PATH_HK_ND), [GART])
    rows.append(
        {
            "label": "Multidimensional benchmark",
            "sub": f"{n:,} instances, $d = 2$ to $100$",
            "gart": v[GART],
            "cavdar": None,
            "bound": v["bound"],
            "n": n,
        }
    )

    v, n = _mape_row(
        _tsplib_screen(_estimator_rows(PATH_TSPLIB)),
        _bound_rows(PATH_HK_TSPLIB),
        [GART, CAVDAR],
    )
    rows.append(
        {
            "label": "TSPLIB EUC_2D",
            "sub": f"{n} instances, $d = 2$",
            "gart": v[GART],
            "cavdar": v[CAVDAR],
            "bound": v["bound"],
            "n": n,
        }
    )

    # Panel (b): the same multidimensional pairing, split by dimension.
    bench = _estimator_rows(PATH_ND)
    bench = bench[bench["model"] == GART]
    bound = _bound_rows(PATH_HK_ND)
    common = set(bench["instance"]) & set(bound["instance"])
    bench = bench[bench["instance"].isin(common)].copy()
    bound = bound[bound["instance"].isin(common)].copy()
    bench["d"] = bench["instance"].str.extract(r"_D(\d+)_").astype(int)
    bound = bound.merge(bench[["instance", "d"]], on="instance", validate="one_to_one")
    by_dim = pd.DataFrame(
        {
            "gart": bench.groupby("d")["abs_gap_pct"].mean(),
            "bound": bound.groupby("d")["abs_gap_pct"].mean(),
            "n": bench.groupby("d")["abs_gap_pct"].size(),
        }
    ).reset_index()
    return rows, by_dim


def _check_banks(rows: list[dict], by_dim: pd.DataFrame) -> None:
    """Fail loudly if a recomputed value has drifted from its stored artifact."""
    bank = json.loads(BANK_ND.read_text())
    nd = next(r for r in rows if r["label"].startswith("Multidimensional"))
    checks = [
        ("ND GART 2.0 MAPE", nd["gart"], bank["GART_2.0_MAPE_pct"]),
        ("ND bound MAPE", nd["bound"], bank["polyak_MAPE_pct_by_k"][str(HK_BUDGET)]),
        ("ND N", float(nd["n"]), float(bank["corpus"]["N"])),
    ]
    stored = pd.read_csv(BY_DIM_ND).set_index("d")
    for _, r in by_dim.iterrows():
        d = int(r["d"])
        checks.append((f"d={d} GART 2.0", r["gart"], stored.loc[d, "GART_2.0"]))
        checks.append((f"d={d} bound", r["bound"], stored.loc[d, f"HK_k{HK_BUDGET}"]))
        checks.append((f"d={d} N", float(r["n"]), float(stored.loc[d, "N"])))
    bad = [
        f"{name}: recomputed {got:.6f} vs banked {want:.6f}"
        for name, got, want in checks
        if not np.isclose(got, want, rtol=1e-6, atol=1e-9)
    ]
    if bad:
        raise SystemExit("figure inputs disagree with the number bank:\n  " + "\n  ".join(bad))
    print(f"bank check: {len(checks)} values agree with polyak_nd_bank.json "
          f"and polyak_nd_by_dimension.csv")


# ---------------------------------------------------------------------------
# Drawing


def _panel_a(ax, rows: list[dict]) -> None:
    ys = list(range(len(rows)))
    for y, row in zip(ys, rows):
        present = [row["gart"], row["bound"]] + ([row["cavdar"]] if row["cavdar"] else [])
        ax.plot([min(present), max(present)], [y, y], "-", color="#DDDDDD", lw=3.0, zorder=1,
                solid_capstyle="round")

        marks = [(row["gart"], C_GART, "o", 6.4, 9), (row["bound"], C_BOUND, "D", 5.6, -13)]
        if row["cavdar"]:
            marks.append((row["cavdar"], C_CAVDAR, "s", 6.0, 9))
        for value, colour, marker, size, dy in marks:
            ax.plot([value], [y], marker, color=colour, ms=size, mew=0.8, mec="white", zorder=4)
            ax.annotate(
                f"{value:.2f}",
                (value, y),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                fontsize=7.2,
                color=colour,
                fontweight="bold",
                zorder=6,
            )
        if row["cavdar"] is None:
            ax.annotate(
                "Çavdar–Sokol is planar:\nno value exists for $d > 2$",
                (row["bound"], y),
                textcoords="offset points",
                xytext=(26, 0),
                ha="left",
                va="center",
                fontsize=7.0,
                color=GREY,
                style="italic",
                linespacing=1.3,
                zorder=6,
            )

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']}\n{r['sub']}" for r in rows], linespacing=1.45)
    ax.set_ylim(len(rows) - 0.55, -0.55)
    ax.set_xscale("log")
    ax.set_xlim(0.33, 42)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("Mean absolute error (% of optimal tour length, log scale)")
    ax.set_title("(a) Accuracy on each benchmark", loc="left", fontweight="bold", pad=17)
    ax.text(0.0, 1.025, "every series scored on the same instances",
            transform=ax.transAxes, fontsize=7.2, color=GREY, va="bottom")
    ax.grid(axis="x", color="#D8D8D8", lw=0.6, ls=":")
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)


def _panel_b(ax, by_dim: pd.DataFrame) -> None:
    d = by_dim["d"].to_numpy(dtype=float)
    ax.plot(d, by_dim["gart"], "-o", color=C_GART, lw=1.6, ms=4.2, mew=0.7, mec="white",
            zorder=4, label=L_GART)
    ax.plot(d, by_dim["bound"], "--D", color=C_BOUND, lw=1.6, ms=4.0, mew=0.7, mec="white",
            zorder=3, label=L_BOUND)

    ax.axvspan(72, 140, color="#F0F0F0", zorder=0)
    ax.annotate(
        "$d = 100$:\nheld out\nfrom training",
        xy=(100, 1.72),
        ha="center",
        va="top",
        fontsize=6.8,
        color=GREY,
        linespacing=1.35,
    )
    ax.annotate(
        "no Çavdar–Sokol curve:\nthe estimator is planar",
        xy=(2.05, 0.30),
        ha="left",
        va="top",
        fontsize=7.0,
        color=C_CAVDAR,
        style="italic",
        linespacing=1.35,
    )

    ax.set_xscale("log")
    ax.set_xlim(1.75, 145)
    ax.set_xticks([2, 3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.95)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Dimension $d$ (log scale)")
    ax.set_ylabel("Mean absolute error\n(% of optimal tour length)")
    ax.set_title("(b) Accuracy as dimension grows", loc="left", fontweight="bold", pad=17)
    ax.text(0.0, 1.025, "multidimensional benchmark, 16,846 instances",
            transform=ax.transAxes, fontsize=7.2, color=GREY, va="bottom")
    ax.grid(color="#D8D8D8", lw=0.6, ls=":")
    ax.set_axisbelow(True)


def _legend(fig) -> None:
    handles = [
        Line2D([], [], color=c, ls="none", marker=m, ms=s, mew=0.8, mec="white", label=lab)
        for lab, c, m, s in SERIES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.5,
        columnspacing=2.0,
    )


def _save(fig: plt.Figure, stem: str) -> None:
    """Write through a scratch name; see the note in ``regenerate_boxplots._save``."""
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


def main() -> None:
    rows, by_dim = collect()
    _check_banks(rows, by_dim)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H), gridspec_kw={"width_ratios": [1.32, 1.0], "wspace": 0.32}
    )
    _panel_a(ax_a, rows)
    _panel_b(ax_b, by_dim)
    # Explicit margins rather than tight_layout: panel (a)'s two-line row labels
    # plus a figure-level legend give tight_layout nothing stable to solve, and it
    # silently ignored the reserved band, printing the legend over panel (b)'s
    # title.  savefig(bbox="tight") still crops the outer whitespace.
    fig.subplots_adjust(left=0.245, right=0.985, top=0.775, bottom=0.165, wspace=0.44)
    _legend(fig)
    _save(fig, "frontier_cost_accuracy")

    for r in rows:
        cav = f"{r['cavdar']:.2f}" if r["cavdar"] else "n/a"
        print(f"  {r['label']:<28s} N={r['n']:>6,}  GART 2.0={r['gart']:.2f}  "
              f"Cavdar={cav:>5s}  bound(K={HK_BUDGET})={r['bound']:.2f}")


if __name__ == "__main__":
    main()
