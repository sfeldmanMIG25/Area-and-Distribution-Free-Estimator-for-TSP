"""Regenerate the two scaling figures.

Figure A, ``accuracy_grid``: GART 2.0 mean absolute percentage error over the
multidimensional benchmark, as a grid in (dimension, instance size).  One field,
two axes.  The paper's thesis is that accuracy improves in *both* arguments, so
the picture is a single gradient running toward the large-n, large-d corner.
The held-out d=100 row is separated because it is extrapolation, not fit.

Figure B, ``cost_scaling``: per-instance wall time against instance size on the
TSPLIB EUC_2D benchmark, for the closed-form classical estimator, GART 2.0, the
certified 1-tree bound, and an exact solve.  Every series is measured on the
same machine.  Concorde runs that hit the wall-clock cap are not drawn; the
solver's fitted log-log trend continues past its last solved instance and exits
the top of the chart, which is the story: the exact solve leaves the measurable
window while the bound climbs steeply and the estimator barely climbs at all.

Run:
    <project python> paper_reference/regenerate_scaling_figures.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ND_CSV = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
TSPLIB_CSV = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
CONCORDE_CSV = ROOT / "paper_tooling" / "exact_solver_tsplib_concorde.csv"
BOUND_CSV = ROOT / "paper_tooling" / "hk1tree_solo_cost_per_instance.csv"

GART = "GART_2.0"
BOUND_K = 100  # the ascent budget the error-distribution figures also draw

# House style, matching regenerate_boxplots.py.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.facecolor": "white",
})

C_GART = "#1f78b4"
C_CLASSICAL = "#f57c20"
C_BOUND = "#a3216f"
C_EXACT = "#4d4d4d"

SIZE_EDGES = [(5, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]
SIZE_LABELS = [f"{lo}–{hi}" for lo, hi in SIZE_EDGES]
INSTANCE_RE = re.compile(r"^N(\d+)_D(\d+)_", re.I)


def _bucket(n: int) -> int | None:
    for i, (lo, hi) in enumerate(SIZE_EDGES):
        if lo <= n <= hi:
            return i
    return None


# ---------------------------------------------------------------- figure A
def build_accuracy_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(ND_CSV, usecols=["model", "instance", "status", "abs_gap_pct"])
    df = df[(df.model == GART) & (df.status == "ok")].copy()
    parsed = df.instance.str.extract(INSTANCE_RE)
    df["n"] = pd.to_numeric(parsed[0], errors="coerce")
    df["d"] = pd.to_numeric(parsed[1], errors="coerce")
    df = df.dropna(subset=["n", "d", "abs_gap_pct"])
    df["bucket"] = df.n.astype(int).map(_bucket)
    df = df.dropna(subset=["bucket"])
    df["bucket"] = df.bucket.astype(int)

    mape = df.pivot_table(index="d", columns="bucket", values="abs_gap_pct", aggfunc="mean")
    counts = df.pivot_table(index="d", columns="bucket", values="abs_gap_pct", aggfunc="size")
    mape = mape.reindex(columns=range(len(SIZE_EDGES)))
    counts = counts.reindex(columns=range(len(SIZE_EDGES)))
    mape.index = mape.index.astype(int)
    counts.index = counts.index.astype(int)
    return mape.sort_index(), counts.sort_index()


def draw_accuracy_grid(mape: pd.DataFrame, counts: pd.DataFrame) -> None:
    dims = list(mape.index)
    trained = [d for d in dims if d != 100]
    held = [d for d in dims if d == 100]
    order = trained + held

    grid = mape.loc[order].to_numpy(dtype=float)
    nrow, ncol = grid.shape

    fig, ax = plt.subplots(figsize=(7.0, 0.30 * nrow + 1.9))
    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    im = ax.imshow(grid, aspect="auto", cmap="YlGnBu",
                   norm=LogNorm(vmin=max(vmin, 1e-3), vmax=vmax))

    for i in range(nrow):
        for j in range(ncol):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            shade = im.norm(v)
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.4,
                    color="white" if shade > 0.62 else "#1a1a1a")

    ax.set_xticks(range(ncol))
    ax.set_xticklabels(SIZE_LABELS)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([("100 (held-out)" if d == 100 else str(d)) for d in order])
    ax.set_xlabel("Instance size $n$")
    ax.set_ylabel("Dimension $d$")

    if held:
        split = len(trained) - 0.5
        ax.axhline(split, color="#c0392b", lw=1.8)
        ax.get_yticklabels()[-1].set_color("#c0392b")

    ax.set_title("GART 2.0 mean absolute percentage error (%) by dimension and instance size",
                 pad=26, loc="left", fontweight="bold")
    ax.text(0, 1.035, "error falls toward the large-instance, high-dimension corner "
                      "of the trained grid; the held-out row runs the other way",
            transform=ax.transAxes, fontsize=8.2, color="#555555")

    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("Mean absolute percentage error (% of reference tour cost, log scale)", fontsize=8)

    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"accuracy_grid.{suffix}")
    plt.close(fig)
    print(f"wrote accuracy_grid.pdf/.png  ({nrow} dimensions x {ncol} size bands, "
          f"{int(np.nansum(counts.to_numpy())):,} instances)")


# ---------------------------------------------------------------- figure B
def build_cost_series() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    tl = pd.read_csv(TSPLIB_CSV, usecols=["instance", "n", "model", "total_time_s",
                                          "edge_weight_type"])
    tl = tl[tl.edge_weight_type == "EUC_2D"]
    for key, model in (("gart", GART), ("classical", "Cavdar")):
        sub = tl[tl.model == model].dropna(subset=["total_time_s", "n"])
        out[key] = (sub.groupby("n", as_index=False)["total_time_s"]
                       .median()
                       .assign(ms=lambda x: x.total_time_s * 1e3))

    # ``median`` is SECONDS in this file, and it carries several timing
    # sessions; keep only the published quiet-window solo protocol.
    bd = pd.read_csv(BOUND_CSV)
    bd = bd[(bd.k == BOUND_K)
            & bd.source.isin(["solo_ladder_k500_r11", "solo_ladder_k500_r3_tail"])]
    bd = bd.dropna(subset=["median", "n"])
    out["bound"] = (bd.groupby("n", as_index=False)["median"].median()
                      .assign(ms=lambda x: x["median"] * 1e3))

    cc = pd.read_csv(CONCORDE_CSV, usecols=["instance", "n", "status", "wall_s", "cap_s"])
    cc = cc.dropna(subset=["wall_s", "n"])
    cc["ms"] = cc.wall_s * 1e3
    st = cc.status.astype(str).str.lower()
    cc = cc[st.isin(["optimal", "censored"])].copy()   # drops the linhp318 parse error
    cc["censored"] = cc.status.astype(str).str.lower() == "censored"
    out["exact"] = cc
    return out


def draw_cost_scaling(series: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    spec = [
        ("classical", C_CLASSICAL, "o", "Çavdar–Sokol (closed form)"),
        ("gart", C_GART, "o", "GART 2.0"),
        ("bound", C_BOUND, "D", f"Held–Karp 1-tree bound ($k{{=}}{BOUND_K}$)"),
    ]
    for key, colour, marker, label in spec:
        d = series[key].sort_values("n")
        ax.plot(d.n, d.ms, marker=marker, ms=3.4, lw=1.4, color=colour,
                label=label, alpha=0.9, markeredgewidth=0)

    ex = series["exact"].sort_values("n")
    obs, cen = ex[~ex.censored], ex[ex.censored]
    ax.scatter(obs.n, obs.ms, marker="s", s=13, color=C_EXACT,
               label="Exact solve (Concorde)", zorder=4, linewidths=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Instance size $n$")
    ax.set_ylabel("Wall time per instance (ms)")
    ax.set_title("Cost of an estimate, a certified bound, and an exact solve",
                 pad=10, loc="left", fontweight="bold")
    ax.grid(True, which="major", ls=":", lw=0.6, color="#bbbbbb")

    # Axis limits from the measured data alone, then freeze them so the exact
    # solver's trend can leave through the top edge instead of stretching it.
    all_ms = np.concatenate([series[k].ms.to_numpy() for k in ("classical", "gart", "bound")]
                            + [obs.ms.to_numpy()])
    all_n = np.concatenate([series[k].n.to_numpy() for k in ("classical", "gart", "bound")]
                           + [obs.n.to_numpy()])
    ax.set_xlim(all_n.min() * 0.85, all_n.max() * 1.15)
    ax.set_ylim(all_ms.min() * 0.55, all_ms.max() * 1.8)
    ax.autoscale(False)

    # The exact curve runs the full length of its data, like every other
    # series: solid through log-binned medians of the solved runs, then dashed
    # at the fitted tail slope until it leaves the top of the chart.  The
    # censored runs (cost > cap) sit under that continuation, so the exit
    # understates nothing.
    FIT_MIN_N = 200
    big = obs[obs.n >= FIT_MIN_N]
    slope, _ = np.polyfit(np.log10(big.n), np.log10(big.ms), 1)
    n_solved = obs.n.max()
    edges = np.geomspace(obs.n.min(), n_solved, 9)
    which = np.clip(np.digitize(obs.n, edges), 1, len(edges) - 1)
    binned = (obs.assign(b=which).groupby("b")
                 .agg(n=("n", "median"), ms=("ms", "median")).sort_values("n"))
    ax.plot(binned.n, binned.ms, color=C_EXACT, lw=1.3, alpha=0.8, zorder=3)
    n0, y0 = binned.n.iloc[-1], binned.ms.iloc[-1]
    xs = np.geomspace(n0, ax.get_xlim()[1], 200)
    ax.plot(xs, y0 * (xs / n0) ** slope, color=C_EXACT, lw=1.3, alpha=0.8,
            ls="--", zorder=3)
    print(f"  exact curve: {len(binned)} binned medians to n={int(n0)}, tail slope "
          f"{slope:.2f} from {len(big)} solved runs with n >= {FIT_MIN_N}; "
          f"{len(cen)} capped runs not drawn")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False,
              ncol=2, borderaxespad=0.0, handletextpad=0.6, columnspacing=1.8)

    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"cost_scaling.{suffix}")
    plt.close(fig)
    print(f"wrote cost_scaling.pdf/.png  (exact: {len(obs)} observed, {len(cen)} censored)")


def main() -> int:
    mape, counts = build_accuracy_grid()
    draw_accuracy_grid(mape, counts)
    lo = np.nanmin(mape.to_numpy())
    hi = np.nanmax(mape.to_numpy())
    print(f"  accuracy range: {lo:.3f}% to {hi:.3f}%")

    series = build_cost_series()
    draw_cost_scaling(series)
    for key in ("classical", "gart", "bound"):
        d = series[key]
        print(f"  {key:10s} n={int(d.n.min())}..{int(d.n.max())}  "
              f"ms {d.ms.min():.2f}..{d.ms.max():.1f}")
    ex = series["exact"]
    print(f"  exact      n={int(ex.n.min())}..{int(ex.n.max())}  "
          f"ms {ex.ms.min():.1f}..{ex.ms.max():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
