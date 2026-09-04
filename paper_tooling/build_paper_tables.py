"""Authoritative generator for every results table and prose number in the paper.

Reads the repaired benchmark artifacts, applies the audited screens, and emits
tidy CSVs, spliceable LaTeX row fragments, a JSON number bank, a coverage table
and paired significance tests into ``paper_tooling/tables/``.

Conventions (bucket edges, ``boot_sdpe_ci``, seed 42, B=1000) are inherited
verbatim from ``paper_tooling/gen_paper_numbers.py``.

Usage:
    python paper_tooling/build_paper_tables.py            # regenerate everything
    python paper_tooling/build_paper_tables.py --check    # + diff vs the .tex
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tsplib_benchmark.exclusions import filter_metric_consistent  # noqa: E402

# -- Inputs / outputs (repoint here only) ----------------------------------
P_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
P_2D_GT = ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv"
P_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
P_ND_GT = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
P_TSPLIB = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
# Per-instance predictions for the two refits on GART 2.0's own 31 features,
# written by paper_tooling/score_31f_controls.py. They are NOT in the benchmark
# CSVs above (that script deliberately writes outside them so scoring a control
# cannot rewrite the authoritative results), so the rank table has to reach for
# them here or the manuscript's 80.0% close-pair figure -- the largest ordering
# deficit it records for GART 2.0 -- ships with no released row behind it.
P_31F_TSPLIB = Path(__file__).resolve().parent / "controls_31f" / "rows_tsplib.csv"
P_TEX = ROOT / "paper_reference" / "Area_Free_Main.tex"
OUT = Path(__file__).resolve().parent / "tables"

RNG_SEED = 42
BOOT_B = 1000
OK_STATUS = frozenset({"ok", "repaired_hybrid_fixed_scaling"})

# Column 6 of the 8-column appendix tables. The fragments now emit MSPE there; plain
# R^2 was dropped because cross-instance cost spans orders of magnitude, so it reads
# 0.99x for every estimator. ``_IN_TEX`` is what Area_Free_Main.tex still prints --
# set it to TEX_STD_COL6 once the new fragments are spliced in, so --check keeps
# comparing the same metric on both sides.
TEX_STD_COL6 = "MSPE_pct"
TEX_STD_COL6_IN_TEX = "MSPE_pct"

# -- Model roster (add a model = add one line) ------------------------------
# The roster, the display names and the identity of the production model all
# live in ``paper_tooling/model_registry.py`` -- the single place a model swap
# is edited. ``GART`` is the local alias for the production key: bolded in
# every table, and the left-hand side of every paired test. What each row is
# called, and whether it is printed at all, is ``MODEL_LABELS``: a model with no
# label there is scored and not reported.
from paper_tooling.model_registry import (  # noqa: E402
    ALPHA_MODELS,
    GART,
    MODEL_LABELS,
)
from paper_tooling.paired_bank import paired_bank_numbers  # noqa: E402
from paper_tooling.generalization_bank import (  # noqa: E402
    generalization_bank_numbers,
    load_results as load_generalization_results,
)
from paper_tooling.consistency_bank import (  # noqa: E402
    consistency_bank_numbers,
    load_results as load_consistency_results,
)
# Exporters that own keys this module does not compute. Each reads back the
# sidecar it wrote, so nothing heavy (lightgbm, shap, a coordinate pass) is
# pulled into a table rebuild -- only the keys are re-merged.
from paper_tooling.alphahat_range import alphahat_bank_numbers  # noqa: E402
from paper_tooling.corpus_statistics import corpus_bank_numbers  # noqa: E402
from paper_tooling.constraint_transfer_bank import (  # noqa: E402
    carried_numbers as constraint_transfer_carried,
)
from paper_tooling.shap_production import shap_bank_numbers  # noqa: E402
from paper_tooling.shap_by_dimension import shap_band_bank_numbers  # noqa: E402
from paper_tooling.cavdar_correction_bank import (  # noqa: E402
    cavdar_correction_bank_numbers,
)
# Ordered rosters that reach the .tex fragments (tidy CSVs carry every model).
# NB the bare string "GART" below is the legacy GART 1.0 key, not ``GART``.
#
# WHAT IS DELIBERATELY ABSENT, AND WHY -- do not re-add without reading this.
#
# ``PREDECESSOR``: the earlier boosted model on the 30-feature block. It is
# dominated by ``LGBM_V4`` on all three benchmarks, so the manuscript reports a
# single ablation of the shipped model instead of two, and reporting the weaker
# one alongside it added a row and no result. It is still benchmarked and it is
# still the embedding donor; it simply carries no ``MODEL_LABELS`` entry, which
# is what keeps it out of the rosters, the tidy tables and the number bank at
# once. Re-adding it here without restoring that label is a silent no-op.
#
# ``NN_V3`` / ``Linear_V3`` / ``NN_31F`` / ``Linear_31F``: the model-class
# controls. They isolated learner from feature set, which is not a comparison
# this paper makes any more. Their artifacts, their rows in every tidy CSV and
# their paired tests are all still produced; only the manuscript rosters drop
# them. The result they carry is recorded in prose instead, in the appendix
# passage on model-class controls (``app:hyperparams``), which is where the
# per-class timings and the feature-count discrepancy are stated; the numbers
# quoted there come from ``tables/table_2d_by_genclass.csv``.
#
# ``Calibrated_MST_d``: dominated by ``Calibrated_MST_dn`` in every bucket of
# every benchmark (ND total 6.24% vs 1.82% MAPE, 2D 11.10% vs 5.76%, TSPLIB
# 11.95% vs 3.74%) and dominated by the alpha=1 floor on TSPLIB (11.95% vs
# 11.37%) and on the `grid` generator (21.12% vs 4.54%). It is an ablation of
# the calibrated lookup -- "does conditioning on n matter" -- not a baseline.
#
# ``MST_Ratio``: the superseded author-chosen schedule; already out.
#
# ``MST_Only`` and ``Asymptotic_MST`` ARE both kept on the planar benchmarks
# even though both are a global constant times L_MST, because they are not
# interchangeable there: on the `grid` generator alpha=1 beats GART 2.0
# (4.54% vs 7.07% MAPE) and the asymptotic ratio does not (9.43%), so dropping
# either one drops a result the other does not carry. The exact relation
# SDPE(c) = c*SDPE(1) holds for aggregate dispersion only.
TEX_MODELS: dict[str, list[str]] = {
    "2d": [GART, "GART", "Calibrated_MST_dn",
           "Asymptotic_MST", "MST_Only", "Hilbert"],
    "nd": [GART, "Calibrated_MST_dn",
           "MST_Only", "BHH_region", "Hilbert"],
    "tsplib": [GART, "GART_1.0", "Asymptotic_MST",
               "Calibrated_MST_dn", "MST_Only", "Hilbert"],
    # Not fixed by the roster brief. MST_Ratio is dropped like everywhere else.
    # Every other estimator is academic_non_2d / no_hybrid_path outside CEIL_2D and
    # stays in the tidy CSV only.
    #
    # COVERAGE IS NO LONGER UNIFORM HERE. Fixed_Alpha scores all 23 screened
    # non-Euclidean instances; GART 2.0 scores 22, declining one on the
    # greedy_nn_over_mst in-distribution guard rather than extrapolating. The Total
    # row is therefore NOT a like-for-like comparison on identical instance sets --
    # see coverage.csv, and the paired tests, which intersect per instance.
    "tsplib_nonEuc": [GART, "Fixed_Alpha"],
}

# Compact body tables for Section 4: one row per bucket, MAPE and SDPE per
# estimator, no CI / median / MSPE / R2 / time columns (those stay in the
# released full tables). Keyed by the tidy table each one is a projection of, so
# --check verifies the printed cells against the same regenerated frame. The
# roster per benchmark is the body roster in TEX_MODELS.
COMPACT_TABLES: dict[str, str] = {"tab:results_nd": "nd_by_dim",
                                  "tab:results_2d": "2d_by_genclass"}
COMPACT_MODELS: dict[str, list[str]] = {
    "nd_by_dim": TEX_MODELS["nd"],
    # Six estimator pairs overflow the text width by 31 pt even at
    # \footnotesize; five fit at \small. Hilbert is the constructive heuristic,
    # its 2D aggregate is quoted in the prose, and the full per-class table
    # stays in the released artifacts.
    "2d_by_genclass": [m for m in TEX_MODELS["2d"] if m != "Hilbert"],
}

# -- Classical-estimator table (the paper's central new result) --------------
#
# TWO ESTIMATORS, NOT FIVE -- and the reduction is a sourcing decision, not a
# results-driven one. Daganzo, Chien and Kwon--Golden--Wasil are gone from every
# panel. Their coefficients entered this repository through the literature
# review in Figliozzi (2008); all three primaries are paywalled with no
# obtainable open-access copy, and the available secondary renderings disagree
# with one another, so nothing here may depend on them. The manuscript still
# surveys the three works; it prints no number for them. ``Cavdar_region`` is
# gone as well -- it fed Cavdar the generator support G^2, which that source
# never asks for.
#
# (a) full benchmarks: BHH with the convex-hull plug-in, Cavdar--Sokol on the
# minimum-area enclosing rectangle its source prescribes.
CLASSICAL_FULL = ["BHH", "Cavdar", GART, "Calibrated_MST_dn"]
# (b) matched-domain: the 210 i.i.d. uniform draws, where BHH's region measure
# is exact. Cavdar--Sokol appears with the same single row it has in panel A --
# it consumes no region, so the panel changes its instance set and nothing else,
# which is what makes the pair of rows readable as a domain effect.
CLASSICAL_MATCHED = [GART, "GART", "BHH_region", "Cavdar", "MST_Only"]
# Paired tests to keep beyond the display roster of panel B.
CLASSICAL_PAIRED_EXTRA = ["BHH"]
# Published n ranges; the sub-domain is taken from the estimator's own usable
# rows. Empty: the two entries here gated Chien to n in [5,30] and Kwon to
# n in [10,80], and both estimators are withdrawn. BHH and Cavdar--Sokol carry
# no fitted node-count range, so there is no sub-domain left to restrict to.
CLASSICAL_SUBDOMAINS: list[tuple[str, str, str]] = []

# -- Rank agreement (tab:rank) ----------------------------------------------
# Global rank correlation plus close-pair ordering accuracy. A pair of instances
# QUALIFIES at threshold t when their reference costs differ by less than t
# relative to the larger of the two; the estimator scores it CONCORDANTLY when it
# orders the pair the way the reference does (a tie in the prediction is counted
# as discordant, and pairs whose reference costs tie exactly are not qualifying
# because they carry no reference ordering). The qualifying set depends only on
# the reference costs, so every estimator on a benchmark is scored on the same
# pairs; only the per-model finite-prediction mask can shrink it.
RANK_THRESHOLDS = (0.05, 0.10)
# Every benchmark here is enumerated exhaustively, so the statistic is exact and
# no seed enters it: 2D has C(2580,2)=3,326,910 pairs, TSPLIB EUC_2D C(78,2)=3,003
# and the multidimensional set C(16920,2)=143,134,740. The last is never
# materialised -- ``_exhaustive_qualifying`` sorts the reference costs and locates
# each element's qualifying partners by binary search, so only the pairs that
# actually qualify (1,662,781 at 5% and 3,290,075 at 10%) are ever built, in
# bounded row blocks. Sampling survives only as a fallback for a future benchmark
# past PAIR_EXHAUSTIVE_MAX; ``pair_seed`` and ``pair_draws`` are PAIR_NA whenever
# the enumeration is exhaustive, because no draw was made.
PAIR_EXHAUSTIVE_MAX = 1_000_000_000
PAIR_SAMPLE_DRAWS = 2_000_000
PAIR_SEED = RNG_SEED
PAIR_NA = "n/a"
# Candidate slots per row block, and the relative widening applied to the binary
# search bound. See ``_exhaustive_qualifying`` for why the bound is widened.
PAIR_CHUNK = 1_000_000
PAIR_SLACK = 1.0 + 1e-9
# Roster and row order of tab:rank, read off the manuscript table. Keyed by the
# bucket slug ``compute_rank`` stamps on its rows. NB the bare string "GART" is
# the legacy GART 1.0 key, not ``GART`` the production alias.
RANK_MODELS: dict[str, list[str]] = {
    "2d": [GART, "GART", "Calibrated_MST_dn", "Asymptotic_MST", "MST_Only"],
    "nd": [GART, "Calibrated_MST_dn", "MST_Only", "Hilbert"],
    "tsplib_euc2d": [GART, "GART_1.0", "Asymptotic_MST", "MST_Only"],
}

# -- Buckets ----------------------------------------------------------------
BucketFn = Callable[[pd.DataFrame], "pd.Series[bool]"]
Bucket = tuple[str, str, BucketFn]


def _rng(col: str, lo: float, hi: float) -> BucketFn:
    return lambda d: (d[col] >= lo) & (d[col] <= hi)


def _all() -> BucketFn:
    return lambda d: pd.Series(True, index=d.index)


def _eq(col: str, val: object) -> BucketFn:
    return lambda d: d[col] == val


def _size_buckets(edges: Sequence[tuple[int, int]]) -> list[Bucket]:
    return [(f"$[{lo},{hi}]$", f"{lo}_{hi}", _rng("n_customers", lo, hi)) for lo, hi in edges] + [
        (r"\textbf{Total}", "total", _all())]


B_2D_SIZE = _size_buckets([(5, 10), (11, 50), (51, 100), (101, 500), (501, 1000)])
B_ND_SIZE = _size_buckets([(5, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, 1000)])
B_ND_DIM: list[Bucket] = [
    ("$d=2$", "d2", _rng("dimension", 2, 2)),
    (r"$d\in[3,5]$", "d3_5", _rng("dimension", 3, 5)),
    (r"$d\in[6,10]$", "d6_10", _rng("dimension", 6, 10)),
    (r"$d\in[15,25]$", "d15_25", _rng("dimension", 15, 25)),
    (r"$d\in[30,50]$", "d30_50", _rng("dimension", 30, 50)),
    (r"$d=100$\textsuperscript{$\dagger$}", "d100", _rng("dimension", 100, 100)),
    (r"\textbf{Total}", "total", _all()),
]
B_TSPLIB: list[Bucket] = [
    (r"$n\in[51,150]$", "51_150", _rng("n", 51, 150)),
    (r"$n\in[151,400]$", "151_400", _rng("n", 151, 400)),
    ("$n>400$", "gt400", _rng("n", 401, 10**9)),
    (r"\textbf{Total}", "total", _all()),
]
B_NONEUC: list[Bucket] = [
    ("ATT", "att", _eq("edge_weight_type", "ATT")),
    (r"CEIL\_2D", "ceil2d", _eq("edge_weight_type", "CEIL_2D")),
    ("GEO", "geo", _eq("edge_weight_type", "GEO")),
    ("EXPLICIT screened", "explicit", _eq("edge_weight_type", "EXPLICIT")),
    (r"\textbf{Total}", "total", _all()),
]

# The five stress classes of tab:dataset_counts, with the Geometric class split
# into the two rows the results table reports.
#
# WHY GEOMETRIC IS SPLIT AND THE OTHER FOUR ARE NOT
# --------------------------------------------------
# ``grid`` is the one Geometric generator with no counterpart in the training
# corpus, and it is the second-worst sub-generator on the whole benchmark: GART
# 2.0 reads +7.11 MSPE on it against +2.78 for the 630-instance Geometric
# aggregate that used to carry it, and every one of its 210 errors is positive.
# Reporting it inside that aggregate hid an unrepresented generator behind two
# represented ones, which is exactly the defect the Line Noise row exists to
# prevent. The split is by TRAINING COVERAGE, not by geometry: the design
# taxonomy of tab:dataset_counts is drawn from the instance-generation
# literature and is unchanged, ``grid`` remains a Geometric generator there, and
# the two sub-rows sum to that table's 630. Promoting ``grid`` to a sixth
# top-level class would instead misdescribe the benchmark's design.
#
# Tuple is (members, expected instance count, row label, bank/bucket slug). The
# slugs of the four unsplit classes are frozen -- prose_manifest keys on
# ``2d_by_genclass_isotropic_*`` and ``2d_by_genclass_linenoise_*``.
GEN_CLASSES: dict[str, tuple[frozenset[str], int, str, str]] = {
    "Isotropic": (frozenset({"random", "normal", "triangular", "truncated_exponential"}),
                  840, "Isotropic", "isotropic"),
    "Biased": (frozenset({"squeezed_uniform", "uniform_triangular", "triangular_squeezed", "correlated"}),
               840, "Biased", "biased"),
    "GeometricGrid": (frozenset({"grid"}), 210,
                      r"Geometric: Grid\textsuperscript{$\dagger$}", "geometric_grid"),
    "GeometricOther": (frozenset({"boundary", "x_central"}), 420,
                       "Geometric: Boundary, X-Central", "geometric_other"),
    "Clustered": (frozenset({"clustered"}), 60, "Clustered", "clustered"),
    "LineNoise": (frozenset({"line_noise"}), 210,
                  r"Line Noise\textsuperscript{$\dagger$}", "linenoise"),
}
B_GENCLASS: list[Bucket] = [(label, slug, _eq("gen_class", k))
                            for k, (_, _, label, slug) in GEN_CLASSES.items()] + [
    (r"\textbf{Total}", "total", _all())
]
GEN_RE = re.compile(r"^TSP-([a-z_0-9]+)-n(\d+)-g(\d+)")


# -- Statistics -------------------------------------------------------------
def boot_sdpe_ci(err_pct: Iterable[float], B: int = BOOT_B, seed: int = RNG_SEED,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap 95% CI for SDPE (signed percent error std, Bessel)."""
    e = np.asarray(err_pct, dtype=float)
    e = e[np.isfinite(e)]
    n = len(e)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    if n < 3:  # point estimate only; a 2-point bootstrap CI is not reported
        return float(np.std(e, ddof=1)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for i in range(B):
        boots[i] = np.std(e[rng.integers(0, n, n)], ddof=1)
    return float(np.std(e, ddof=1)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 2:
        return float("nan")
    ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y[m] - yhat[m]) ** 2)) / ss_tot


# -- Timing provenance guard ------------------------------------------------
# ``timing_provenance`` is written by paper_tooling/restore_tsplib_serial_timings.py.
# Rows tagged with a "pending_" reason had their only wall-clock measurement taken
# under a contended parallel run; the value was withheld rather than published.
# GART 2.0 postdates the serial reference run, so every one of its timing rows is
# pending until a low-contention run exists.
TIMING_PROVENANCE_COL = "timing_provenance"
PENDING_PREFIX = "pending_"
WITHHELD_TIMINGS: dict[tuple[str, str], int] = {}


def publishable_times(sub: pd.DataFrame, time_col: str, model: str,
                      table: str = "?") -> "pd.Series[float]":
    """Wall-clock values safe to publish, or an empty series if any are withheld.

    A median over only the non-withheld rows of a group would silently report a
    biased subset as if it were the whole group, so if any row in the group is
    pending the entire group's timing is withheld and the cell renders '---'.
    Frames without the provenance column (2D, ND) keep the previous behaviour.
    """
    t = sub[time_col].dropna()
    if TIMING_PROVENANCE_COL not in sub.columns:
        return t
    pending = sub[TIMING_PROVENANCE_COL].astype(str).str.startswith(PENDING_PREFIX)
    n = int(pending.sum())
    if n:
        WITHHELD_TIMINGS[(table, model)] = WITHHELD_TIMINGS.get((table, model), 0) + n
        return t.iloc[0:0]
    return t


def report_withheld_timings() -> None:
    """Print a loud summary of every timing cell suppressed by the guard."""
    if not WITHHELD_TIMINGS:
        return
    print("\n" + "!" * 82, file=sys.stderr)
    print("WITHHELD TIMINGS -- these cells render '---', they are NOT measurements of 0",
          file=sys.stderr)
    print("Cause: no low-contention (serial) wall-clock run exists for these rows.",
          file=sys.stderr)
    print("Fix:   re-run the benchmark serially, then rerun restore_tsplib_serial_timings.py.",
          file=sys.stderr)
    print("!" * 82, file=sys.stderr)
    for (table, model), n in sorted(WITHHELD_TIMINGS.items()):
        print(f"  {table:<22}{model:<22}{n:>5} pending rows", file=sys.stderr)
    print("", file=sys.stderr)


def group_metrics(sub: pd.DataFrame, model: str, time_col: str,
                  table: str = "?") -> dict[str, float | str]:
    """All per-(bucket, model) metrics. R^2_alpha is '---' when undefined."""
    e = sub["err_pct"].to_numpy(dtype=float)
    sdpe, lo, hi = boot_sdpe_ci(e)
    r2a: float | str = "---"
    if model in ALPHA_MODELS and "mst_length" in sub.columns:
        ml = sub["mst_length"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = np.where(ml > 0, sub["true_cost"].to_numpy(float) / ml, np.nan)
            ap = np.where(ml > 0, sub["pred_cost"].to_numpy(float) / ml, np.nan)
        ok = np.isfinite(at) & np.isfinite(ap)
        if ok.sum() >= 3 and float(np.std(at[ok])) >= 1e-4:
            v = _r2(at[ok], ap[ok])
            r2a = v if np.isfinite(v) else "---"
    t = publishable_times(sub, time_col, model, table)
    return {
        "N": int(len(sub)),
        "SDPE_pct": sdpe, "SDPE_lo": lo, "SDPE_hi": hi,
        "MAPE_pct": float(np.mean(np.abs(e))),
        "MedAPE_pct": float(np.median(np.abs(e))),
        "MSPE_pct": float(np.mean(e)),
        "RMSPE_pct": float(np.sqrt(np.mean(e ** 2))),
        "R2": _r2(sub["true_cost"].to_numpy(float), sub["pred_cost"].to_numpy(float)),
        "R2_alpha": r2a,
        "time_ms": float(np.median(t) * 1000.0) if len(t) else float("nan"),
    }


def paired_test(df: pd.DataFrame, model_a: str, model_b: str,
                bucket_mask: "pd.Series[bool]") -> dict[str, float]:
    """Paired |APE| difference (a - b) on instances where both models are finite."""
    nan = float("nan")
    empty = {"n_pairs": 0, "mean_diff": nan, "ci_lo": nan, "ci_hi": nan, "wilcoxon_p": nan}
    sub = df.loc[bucket_mask, ["instance", "model", "err_pct"]]
    wide = sub.pivot_table(index="instance", columns="model", values="err_pct", aggfunc="first")
    if model_a not in wide.columns or model_b not in wide.columns:
        return empty
    pair = wide[[model_a, model_b]].dropna()
    d = (pair[model_a].abs() - pair[model_b].abs()).to_numpy(dtype=float)
    n = len(d)
    if n == 0:
        return empty
    rng = np.random.default_rng(RNG_SEED)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT_B)])
    p = float("nan")
    if n >= 1 and not np.allclose(d, 0.0):
        p = float(wilcoxon(d, zero_method="wilcox").pvalue)
    return {"n_pairs": n, "mean_diff": float(d.mean()),
            "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "wilcoxon_p": p}


Qualifying = tuple[np.ndarray, np.ndarray, np.ndarray]


def _qualifies(ti: np.ndarray, tj: np.ndarray, t: float) -> "np.ndarray":
    """The qualifying-pair predicate, defined once: close, and not an exact tie."""
    return (np.abs(ti - tj) / np.maximum(ti, tj) < t) & (np.sign(ti - tj) != 0)


def _chunk_edges(cnt: np.ndarray, budget: int) -> np.ndarray:
    """Row-block boundaries so each block spans roughly ``budget`` candidate slots."""
    cum = np.cumsum(cnt)
    total = int(cum[-1])
    cuts = np.searchsorted(cum, np.arange(budget, total + budget, budget)) + 1
    return np.unique(np.concatenate(([0], cuts, [len(cnt)]))).clip(0, len(cnt))


def _exhaustive_qualifying(true: np.ndarray, t: float) -> Qualifying:
    """Every qualifying pair at threshold ``t``, without materialising C(n,2).

    Sort the reference costs ascending. For ``j`` above ``i`` in that order
    ``max(s_i, s_j) = s_j``, so the relative gap is ``1 - s_i/s_j``, which is
    non-decreasing in ``s_j``: the partners of ``i`` therefore form one
    contiguous run, whose end binary search locates. The bound is widened by
    PAIR_SLACK and the shipped predicate re-evaluated inside the window, because
    ``s_j < s_i/(1-t)`` is the algebraic rearrangement of the predicate and not
    its bit-for-bit equal -- on the multidimensional set the two disagree on one
    pair at the 10% threshold. Costs must be strictly positive for the
    monotonicity argument to hold; ``_finalize`` already screens on that and the
    guard below refuses to run if it ever stops being true.

    Returns ``(i, j, reference_sign)`` with ``i<j`` in lexicographic order, which
    is the order ``np.triu_indices`` produces, so the enumeration is
    interchangeable with a full scan down to the last bit of the mean.
    """
    n = len(true)
    if n < 2:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, float)
    if not np.all(true > 0):
        raise RuntimeError("close-pair enumeration needs strictly positive reference costs")
    order = np.argsort(true, kind="stable")
    s = true[order]
    start = np.arange(1, n + 1, dtype=np.int64)
    stop = np.maximum(np.searchsorted(s, s / (1.0 - t) * PAIR_SLACK, side="right"), start)
    cnt = stop - start
    edges = _chunk_edges(cnt, PAIR_CHUNK)
    ii: list[np.ndarray] = []
    jj: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = cnt[lo:hi]
        k = int(c.sum())
        if not k:
            continue
        src = np.repeat(np.arange(lo, hi, dtype=np.int64), c)
        dst = (np.repeat(start[lo:hi], c) + np.arange(k, dtype=np.int64)
               - np.repeat(np.cumsum(c) - c, c))
        keep = _qualifies(s[src], s[dst], t)
        if keep.any():
            ii.append(order[src[keep]])
            jj.append(order[dst[keep]])
    if not ii:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, float)
    a, b = np.concatenate(ii), np.concatenate(jj)
    a, b = np.minimum(a, b), np.maximum(a, b)
    k = np.lexsort((b, a))
    a, b = a[k].astype(np.int32), b[k].astype(np.int32)
    return a, b, np.sign(true[a] - true[b])


def _sampled_qualifying(true: np.ndarray, seed: int) -> dict[float, Qualifying]:
    """Fallback for a benchmark past PAIR_EXHAUSTIVE_MAX: a seeded pair sample.

    Draw PAIR_SAMPLE_DRAWS index pairs, discard self-pairs and duplicates, and
    score the distinct remainder. Reproducible bit-for-bit from ``seed``, which
    the caller stamps on every row so the sample can be identified.
    """
    n = len(true)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, PAIR_SAMPLE_DRAWS, dtype=np.int64)
    b = rng.integers(0, n, PAIR_SAMPLE_DRAWS, dtype=np.int64)
    keep = a != b
    lo, hi = np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])
    key = np.unique(lo * n + hi)  # distinct unordered pairs, ascending
    pi, pj = (key // n).astype(np.int32), (key % n).astype(np.int32)
    ti, tj = true[pi], true[pj]
    sign = np.sign(ti - tj)
    return {t: (pi[m], pj[m], sign[m])
            for t in RANK_THRESHOLDS for m in [_qualifies(ti, tj, t)]}


def compute_rank(df: pd.DataFrame, benchmark: str, slug: str,
                 seed: int = PAIR_SEED) -> pd.DataFrame:
    """Rank agreement with the reference tour cost, one row per model.

    Spearman and Kendall are taken over the model's own instances; close-pair
    accuracy over the shared qualifying pair set (see RANK_THRESHOLDS). The
    qualifying set depends only on the reference costs, so it is built once per
    benchmark -- exhaustively unless the benchmark is past PAIR_EXHAUSTIVE_MAX,
    in which case ``seed`` selects a reproducible sample instead.
    """
    ref = (df[["instance", "true_cost"]].drop_duplicates("instance")
             .sort_values("instance").reset_index(drop=True))
    inst = ref["instance"].to_numpy()
    true = ref["true_cost"].to_numpy(dtype=float)
    pos = {s: k for k, s in enumerate(inst)}
    n_inst = len(inst)
    qual: dict[float, Qualifying]
    if n_inst * (n_inst - 1) // 2 <= PAIR_EXHAUSTIVE_MAX:
        mode, seed_out, draws = "exhaustive", PAIR_NA, PAIR_NA
        qual = {t: _exhaustive_qualifying(true, t) for t in RANK_THRESHOLDS}
    else:
        mode, seed_out, draws = "sampled", seed, PAIR_SAMPLE_DRAWS
        qual = _sampled_qualifying(true, seed)

    order = {m: k for k, m in enumerate(MODEL_LABELS)}
    rows: list[dict[str, object]] = []
    for model in sorted(set(df["model"]), key=lambda m: order.get(m, 999)):
        if model not in MODEL_LABELS:
            continue
        sub = df.loc[df["model"] == model, ["instance", "pred_cost"]].dropna()
        sub = sub.drop_duplicates("instance")
        if len(sub) < 3:
            continue
        pred = np.full(len(inst), np.nan)
        pred[[pos[s] for s in sub["instance"]]] = sub["pred_cost"].to_numpy(dtype=float)
        have = np.isfinite(pred)
        row: dict[str, object] = {
            "table": "rank", "bucket": benchmark, "bucket_slug": slug,
            "bucket_count": int(len(inst)), "model": model,
            "display": MODEL_LABELS[model], "N": int(have.sum()),
            "spearman_rho": float(spearmanr(pred[have], true[have])[0]),
            "kendall_tau": float(kendalltau(pred[have], true[have])[0]),
            "pair_mode": mode, "pair_seed": seed_out, "pair_draws": draws,
            "pair_universe": int(n_inst),
        }
        for t in RANK_THRESHOLDS:
            a, b, sign = qual[t]
            ok = have[a] & have[b]
            n_pairs = int(ok.sum())
            tag = f"close{int(round(t * 100))}"
            row[f"{tag}_pairs"] = n_pairs
            row[f"{tag}_pct"] = (
                float(100.0 * np.mean(np.sign(pred[a[ok]] - pred[b[ok]]) == sign[ok]))
                if n_pairs else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# -- Loading ----------------------------------------------------------------
def _merge_gt(res: pd.DataFrame, gt: pd.DataFrame, cols: list[str], tag: str) -> pd.DataFrame:
    before = len(res)
    out = res.merge(gt[["instance"] + cols], on="instance", how="left", validate="m:1")
    if len(out) != before:
        raise RuntimeError(f"{tag}: ground-truth join changed row count {before} -> {len(out)}")
    missing = out[cols].isna().any(axis=1).sum()
    if missing:
        raise RuntimeError(f"{tag}: ground-truth join left {missing} rows without {cols}")
    return out


def _finalize(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into usable rows and a per-(model, status) coverage ledger."""
    cov = (df.groupby(["model", "status"], dropna=False).size()
             .rename("n_rows").reset_index())
    cov.insert(0, "dataset", dataset)
    cov["status_kept"] = cov["status"].isin(OK_STATUS)
    keep = (df["status"].isin(OK_STATUS) & np.isfinite(df["pred_cost"])
            & np.isfinite(df["true_cost"]) & (df["true_cost"] > 0))
    used = df.loc[keep].copy()
    used["err_pct"] = 100.0 * (used["pred_cost"] - used["true_cost"]) / used["true_cost"]
    used = used[np.isfinite(used["err_pct"])].copy()
    n_used = used.groupby("model").size().rename("n_used")
    cov = cov.merge(n_used, left_on="model", right_index=True, how="left").fillna({"n_used": 0})
    cov["n_used"] = cov["n_used"].astype(int)
    return used, cov


def load_2d() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    gt = pd.read_csv(P_2D_GT)
    df = _merge_gt(pd.read_csv(P_2D, low_memory=False), gt,
                   ["n_customers", "grid_size", "mst_length"], "2D")
    gen = df["instance"].str.extract(GEN_RE)
    if gen[0].isna().any():
        raise RuntimeError("2D: instance names do not match the generator grammar")
    df["generator"] = gen[0]
    df["gen_class"] = pd.NA
    for cls, (members, _, _, _) in GEN_CLASSES.items():
        df.loc[df["generator"].isin(members), "gen_class"] = cls
    if df["gen_class"].isna().any():
        raise RuntimeError(f"2D: unmapped generators {sorted(set(df.loc[df.gen_class.isna(), 'generator']))}")
    per_class = df.drop_duplicates("instance").groupby("gen_class").size().to_dict()
    for cls, (_, expect, _, _) in GEN_CLASSES.items():
        if per_class.get(cls, 0) != expect:
            raise RuntimeError(f"2D generator class {cls}: expected {expect} instances, got {per_class.get(cls, 0)}")
    # The class rows must still partition the 2,580-instance benchmark: a split
    # that dropped or double-counted a generator would otherwise pass the
    # per-class check above and only show up as a wrong Total.
    total = sum(c for _, c, _, _ in GEN_CLASSES.values())
    n_inst = int(df["instance"].nunique())
    if total != n_inst:
        raise RuntimeError(f"2D generator classes sum to {total} instances, benchmark holds {n_inst}")
    used, cov = _finalize(df, "2D")
    return used, cov, _ref_times(gt, B_2D_SIZE)


def load_nd() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]]:
    gt = pd.read_csv(P_ND_GT)
    df = _merge_gt(pd.read_csv(P_ND, low_memory=False), gt,
                   ["n_customers", "dimension", "grid_size", "mst_length"], "ND")
    used, cov = _finalize(df, "ND")
    return used, cov, _ref_times(gt, B_ND_SIZE), _ref_times(gt, B_ND_DIM)


def load_tsplib() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(P_TSPLIB, low_memory=False)
    # mst_length is only persisted on some model rows; it is an instance property.
    ml = df.dropna(subset=["mst_length"]).drop_duplicates("instance").set_index("instance")["mst_length"]
    df["mst_length"] = df["instance"].map(ml)
    used, cov = _finalize(df, "TSPLIB")
    euc = filter_metric_consistent(used[used["edge_weight_type"] == "EUC_2D"])
    non = filter_metric_consistent(used[used["edge_weight_type"] != "EUC_2D"])
    return euc, non, cov


def with_31f_controls(euc: pd.DataFrame) -> pd.DataFrame:
    """``euc`` plus the two 31-feature refits, for the rank table only.

    Returned frame carries only the columns ``compute_rank`` reads, so it must
    not be fed to ``compute_table``: the controls have no timing, no status and
    no coverage row, and printing them in an accuracy table would imply a
    provenance they do not have. The accuracy figures for these two models are
    reported from ``controls_31f/marginals.csv`` instead.

    Scored on exactly the instances ``euc`` holds -- the control file also
    carries the non-Euclidean instances -- so every model in the rank table is
    ordered over the same qualifying pair set.
    """
    if not P_31F_TSPLIB.exists():
        print(f"note: {P_31F_TSPLIB.name} missing; rank table omits the 31-feature refits")
        return euc
    keep = set(euc["instance"])
    ctrl = pd.read_csv(P_31F_TSPLIB)
    ctrl = ctrl[ctrl["model"].isin(("NN_31F", "Linear_31F")) & ctrl["instance"].isin(keep)]
    ctrl = ctrl.dropna(subset=["pred_cost"])[["instance", "model", "pred_cost"]]
    true = euc.drop_duplicates("instance").set_index("instance")["true_cost"]
    for m, g in ctrl.groupby("model"):
        if len(g) != len(keep):
            raise RuntimeError(f"31F control {m}: {len(g)} of {len(keep)} EUC_2D instances scored; "
                               f"a partial roster would be ranked against a different universe")
    ctrl["true_cost"] = ctrl["instance"].map(true)
    return pd.concat([euc[["instance", "model", "pred_cost", "true_cost"]], ctrl],
                     ignore_index=True)


def _ref_times(gt: pd.DataFrame, buckets: list[Bucket]) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, slug, fn in buckets:
        s = gt.loc[fn(gt), "optimal_solve_time_s"].dropna()
        out[slug] = float(np.median(s) * 1000.0) if len(s) else float("nan")
    return out


# -- Table assembly ---------------------------------------------------------
def compute_table(df: pd.DataFrame, buckets: list[Bucket], time_col: str, table: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    order = {m: i for i, m in enumerate(MODEL_LABELS)}
    for label, slug, fn in buckets:
        sel = df.loc[fn(df)]
        count = int(sel["instance"].nunique())
        for model in sorted(set(sel["model"]), key=lambda m: order.get(m, 999)):
            if model not in MODEL_LABELS:
                continue
            sub = sel[sel["model"] == model]
            if len(sub) == 0:
                continue
            rows.append({"table": table, "bucket": label, "bucket_slug": slug,
                         "bucket_count": count, "model": model,
                         "display": MODEL_LABELS[model],
                         **group_metrics(sub, model, time_col, table)})
    return pd.DataFrame(rows)


def _classical_row(panel: str, label: str, slug: str, count: int, model: str,
                   sub: pd.DataFrame, time_col: str) -> dict[str, object]:
    """One classical-table row; an estimator absent from a benchmark yields N=0."""
    row: dict[str, object] = {
        "table": "classical", "panel": panel, "bucket": label, "bucket_slug": slug,
        "bucket_count": count, "model": model, "display": MODEL_LABELS.get(model, model)}
    if len(sub) == 0:
        nan = float("nan")
        row.update({"N": 0, "SDPE_pct": nan, "SDPE_lo": nan, "SDPE_hi": nan,
                    "MAPE_pct": nan, "MedAPE_pct": nan, "MSPE_pct": nan,
                    "RMSPE_pct": nan, "R2": nan, "R2_alpha": "---", "time_ms": nan})
        return row
    row.update(group_metrics(sub, model, time_col, f"classical/{panel}"))
    return row


def compute_classical(d2: pd.DataFrame, euc: pd.DataFrame) -> pd.DataFrame:
    """Panel A: convex-hull plug-ins on the complete benchmarks.

    Panel B: the matched domain where the sampling region is exact (i.i.d. uniform
    draws), with each published regression restricted to its own fitted n range and
    GART 2.0 re-evaluated on exactly those instances for a like-for-like comparison.
    """
    rows: list[dict[str, object]] = []
    for label, slug, df, tcol in [
        ("Full 2D benchmark", "a_2d", d2, "prediction_time_s"),
        (r"TSPLIB EUC\_2D", "a_tsplib", euc, "total_time_s"),
    ]:
        count = int(df["instance"].nunique())
        for m in CLASSICAL_FULL:
            rows.append(_classical_row("A", label, slug, count, m, df[df["model"] == m], tcol))
    rnd = d2[d2["generator"] == "random"]
    count = int(rnd["instance"].nunique())
    for m in CLASSICAL_MATCHED:
        rows.append(_classical_row("B", r"2D \texttt{random} (uniform on $[0,G]^2$)\\ matched domain", "b_random",
                                   count, m, rnd[rnd["model"] == m], "prediction_time_s"))
    for anchor, label, slug in CLASSICAL_SUBDOMAINS:
        inst = set(rnd.loc[rnd["model"] == anchor, "instance"])
        sd = rnd[rnd["instance"].isin(inst)]
        for m in (anchor, GART, "MST_Only"):
            rows.append(_classical_row("B", label, slug, len(inst), m,
                                       sd[sd["model"] == m], "prediction_time_s"))
    return pd.DataFrame(rows)


def write_tex_classical(tidy: pd.DataFrame, path: Path, nobreak: bool = False) -> None:
    """7-column table: group | estimator | N | SDPE [CI] | MAPE | |Median| | MSPE.

    ``nobreak`` terminates every row but each domain's last with ``\\\\*``, so a
    page break cannot land inside a domain and strand its ``\\multirow`` label.
    Set it for the manuscript's longtable copy; the per-panel fragments are not
    typeset as longtables and leave it off. Unlike ``write_tex_std`` the
    ``\\multirow`` lead shares a line with the domain's first data row, so every
    line but the last is protected rather than every line but the first and last.
    """
    blocks: list[str] = []
    for slug in tidy["bucket_slug"].drop_duplicates():
        sel = tidy[tidy["bucket_slug"] == slug]
        label = str(sel.iloc[0]["bucket"])
        lines: list[str] = []
        for i, (_, r) in enumerate(sel.iterrows()):
            name = _texname(MODEL_LABELS.get(str(r["model"]), str(r["model"])))
            if r["model"] == GART:
                name = rf"\textbf{{{name}}}"
            n = int(r["N"])
            ci = (rf" \scriptsize[{_f(r['SDPE_lo'], 2)},\,{_f(r['SDPE_hi'], 2)}]"
                  if n >= 3 else r" \scriptsize[---,\,---]")
            lead = rf"\multirow{{{len(sel)}}}{{*}}{{\makecell[l]{{{label}}}}}" if i == 0 else ""
            lines.append(f"{lead} & {name} & {_thou(n)} & {_f(r['SDPE_pct'], 2)}{ci} "
                         f"& {_f(r['MAPE_pct'], 2)} & {_f(r['MedAPE_pct'], 2)} "
                         f"& {_f(r['MSPE_pct'], 2)} \\\\")
        if nobreak:
            lines[:-1] = [ln + "*" for ln in lines[:-1]]
        blocks.append("\n".join(lines))
    path.write_text("\n\\midrule\n".join(blocks) + "\n", encoding="utf-8")


def _sig3(x: float) -> str:
    """Three significant figures, trailing zeros kept (author, 2026-09-03).

    Every numeric cell the builders emit prints at this precision; the ``nd``
    argument of :func:`_f` is retained by the callers for provenance only.
    """
    if not np.isfinite(x):
        return "---"
    if abs(x) >= 1000:
        return _thou(int(round(x)))
    s = f"{x:#.3g}".rstrip(".")
    return "$-$" + s[1:] if s.startswith("-") else s


def _thou(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _f(x: object, nd: int) -> str:
    if isinstance(x, str):
        return x
    return _sig3(float(x))


def write_tex_std(tidy: pd.DataFrame, buckets: list[Bucket], models: list[str],
                  ref_times: dict[str, float] | None, path: Path,
                  nobreak: bool = False) -> None:
    """8-column appendix table: bucket | model | SDPE[CI] | MAPE | Med | MSPE | R2a | t.

    Column 6 carries MSPE, not plain R^2: cross-instance tour cost spans orders of
    magnitude, so R^2 sits at 0.99x for every estimator and separates nothing. R^2
    is still computed and kept in the tidy CSVs.

    ``nobreak`` terminates every row but each bucket's last with ``\\\\*``, which
    forbids a page break there. Set it for the tables the manuscript typesets as
    a ``longtable``: those break across pages, and a break inside a bucket would
    strand that bucket's ``\\multirow`` label. It is inert in a plain ``tabular``,
    which never breaks, so the flag records intent rather than changing output
    for the tables that stay floats.
    """
    blocks: list[str] = []
    for label, slug, _ in buckets:
        sel = tidy[tidy["bucket_slug"] == slug].set_index("model")
        present = [m for m in models if m in sel.index]
        if not present:
            continue
        present.sort(key=lambda m: (float(sel.loc[m, "SDPE_pct"]), models.index(m)))
        n_rows = len(present) + (1 if ref_times else 0)
        count = _thou(int(sel.iloc[0]["bucket_count"]))
        lines = [rf"\multirow{{{n_rows}}}{{*}}{{\makecell[l]{{{label}\\ Count={count}}}}}"]
        for m in present:
            r = sel.loc[m]
            name, sd = _texname(MODEL_LABELS[m]), _f(r["SDPE_pct"], 2)
            if m == GART:
                name, sd = rf"\textbf{{{name}}}", rf"\textbf{{{sd}}}"
            ci = rf" \scriptsize[{_f(r['SDPE_lo'], 2)},\,{_f(r['SDPE_hi'], 2)}]"
            lines.append(f"  & {name} & {sd}{ci} & {_f(r['MAPE_pct'], 2)} & {_f(r['MedAPE_pct'], 2)} "
                         f"& {_f(r[TEX_STD_COL6], 2)} & {_f(r['R2_alpha'], 3)} "
                         f"& {_sig3(float(r['time_ms']))} \\\\")
        if ref_times:
            t = ref_times.get(slug, float("nan"))
            ms = _thou(int(round(t))) if np.isfinite(t) else "---"
            lines.append(rf"  & \textit{{Reference-tour generation}} & --- & --- & --- & --- & --- & {ms}~ms \\")
        if nobreak:  # lines[0] is the \multirow lead; lines[-1] closes the bucket
            lines[1:-1] = [ln + "*" for ln in lines[1:-1]]
        blocks.append("\n".join(lines))
    path.write_text("\n\\midrule\n".join(blocks) + "\n", encoding="utf-8")


def write_tex_compact(tidy: pd.DataFrame, buckets: list[Bucket], models: list[str],
                      path: Path) -> None:
    """Body rows of a compact results table: bucket | Count | (MAPE, SDPE) per model.

    One row per bucket in the order ``buckets`` gives; the estimator columns are
    fixed in the order ``models`` gives so the manuscript header (written once
    by hand, with the same order) stays aligned. GART 2.0's cells are bold, the
    convention Section 4 states. A model absent from a bucket prints ``---``.
    """
    lines: list[str] = []
    for label, slug, _ in buckets:
        sel = tidy[tidy["bucket_slug"] == slug].set_index("model")
        if sel.empty:
            continue
        cells = [label, _thou(int(sel.iloc[0]["bucket_count"]))]
        for m in models:
            if m in sel.index:
                mape, sd = _f(sel.loc[m, "MAPE_pct"], 2), _f(sel.loc[m, "SDPE_pct"], 2)
            else:
                mape = sd = "---"
            if m == GART:
                mape, sd = rf"\textbf{{{mape}}}", rf"\textbf{{{sd}}}"
            cells += [mape, sd]
        lines.append(" & ".join(cells) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Column heads for the compact tables. The full display names of MODEL_LABELS
# are too wide for six estimator pairs on one text width; --check keys the
# parsed cells by column position and MODEL_LABELS, never by this header text.
COMPACT_HEAD: dict[str, str] = {
    GART: "GART 2.0", "GART": "GART 1.0",
    "Calibrated_MST_dn": r"$\hat\rho(d,n)$", "Asymptotic_MST": "Asymptotic",
    "MST_Only": r"$\alpha=1$", "BHH_region": "BHH", "Hilbert": "Hilbert",
}


def compact_header(models: list[str]) -> str:
    """The two-row header matching write_tex_compact, for pasting once by hand."""
    k = len(models)
    spec = "@{}lr" + "rr" * k + "@{}"
    names = " & ".join(rf"\multicolumn{{2}}{{c}}{{{COMPACT_HEAD.get(m, _texname(MODEL_LABELS[m]))}}}"
                       for m in models)
    rules = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(k))
    sub = " & ".join(["MAPE", "SDPE"] * k)
    return (rf"\begin{{tabular}}{{{spec}}}" "\n" r"\toprule" "\n"
            rf" & & {names} \\" "\n" f"{rules}\n"
            rf"Bucket & Count & {sub} \\" "\n" r"\midrule")


def write_tex_noneuc(tidy: pd.DataFrame, buckets: list[Bucket], models: list[str], path: Path) -> None:
    """8-column table: type | count | estimator | N | SDPE [CI] | MAPE | |Median| | MSPE.

    ``Count`` is the stratum size; ``N`` is what the estimator actually scores, and the
    two differ. GART 2.0 declines one EXPLICIT instance on its in-distribution guard,
    so its Total aggregates 22 instances against 23 for the references. Printing N per
    row keeps that asymmetry on the page instead of hiding it behind a Total that is
    not a like-for-like comparison.
    """
    blocks: list[str] = []
    for label, slug, _ in buckets:
        sel = tidy[tidy["bucket_slug"] == slug].set_index("model")
        present = [m for m in models if m in sel.index]
        if not present:
            continue
        total = slug == "total"
        cnt = str(int(sel.iloc[0]["bucket_count"]))
        head_cnt = rf"\textbf{{{cnt}}}" if total else cnt
        lines: list[str] = []
        for i, m in enumerate(present):
            r = sel.loc[m]
            n = int(r["N"])
            ci = f"[{_f(r['SDPE_lo'], 2)}, {_f(r['SDPE_hi'], 2)}]" if n >= 3 else "[---, ---]"
            cells = [str(n), f"{_f(r['SDPE_pct'], 2)} {ci}", _f(r["MAPE_pct"], 2),
                     _f(r["MedAPE_pct"], 2), _f(r["MSPE_pct"], 2)]
            name = _texname(MODEL_LABELS[m])
            if total and m == GART:  # the manuscript bolds only the GART row of the Total group
                name, cells = rf"\textbf{{{name}}}", [rf"\textbf{{{c}}}" for c in cells]
            lead = (rf"\multirow{{{len(present)}}}{{*}}{{{label}}} & \multirow{{{len(present)}}}{{*}}{{{head_cnt}}}"
                    if i == 0 else " &")
            lines.append(f"{lead} & {name} & " + " & ".join(cells) + " \\\\")
        blocks.append("\n".join(lines))
    path.write_text("\n\\midrule\n".join(blocks) + "\n", encoding="utf-8")


def write_tex_rank(tidy: pd.DataFrame, path: Path) -> None:
    """7-column table: benchmark | model | N | Spearman | Kendall | close 5% | close 10%.

    Row order is the roster order in ``RANK_MODELS``, not a performance sort: the
    production model leads each block and the $\\alpha=1$ control closes the ones
    that carry it, which is what the caption refers to.
    """
    blocks: list[str] = []
    for slug in tidy["bucket_slug"].drop_duplicates():
        sel = tidy[tidy["bucket_slug"] == slug].set_index("model")
        present = [m for m in RANK_MODELS.get(slug, []) if m in sel.index]
        if not present:
            continue
        label = str(sel.iloc[0]["bucket"])
        lines: list[str] = []
        for i, m in enumerate(present):
            r = sel.loc[m]
            name = (rf"\textbf{{{_texname(MODEL_LABELS[m])}}}" if m == GART
                    else _texname(MODEL_LABELS[m]))
            lead = rf"\multirow{{{len(present)}}}{{*}}{{{label}}}" if i == 0 else ""
            lines.append(f"{lead} & {name} & {_thou(int(r['N']))} "
                         f"& {_f(r['spearman_rho'], 4)} & {_f(r['kendall_tau'], 4)} "
                         f"& {_f(r['close5_pct'], 2)} & {_f(r['close10_pct'], 2)} \\\\")
        blocks.append("\n".join(lines))
    path.write_text("\n\\midrule\n".join(blocks) + "\n", encoding="utf-8")


# The MODEL_LABELS strings double as bank-key slugs (_slug is applied to them),
# so they have to stay ASCII: adding the cedilla there would rename every
# cavdar_sokol key in paper_numbers.json. The typeset name has no such
# constraint, and the running prose spells it with the diacritic, so the table
# body did too until the roster rebuild reintroduced the ASCII form. This map is
# applied on the way into the .tex only; _clean folds it back for --check.
TEX_NAME = {"Cavdar--Sokol": r"\c{C}avdar--Sokol"}


def _texname(display: str) -> str:
    return TEX_NAME.get(display, display)


# -- --check: parse the manuscript ------------------------------------------
_TEXTB = re.compile(r"\\(?:textbf|textit|mathbf)\{([^{}]*)\}")
_MAKECELL = re.compile(r"\\makecell\[l\]\{(.*?)\\\\\s*Count=")
_CI = re.compile(r"^([-\d.]+)\s*\[\s*([-\d.]+|-{3})\s*,\s*(?:\\,)?\s*([-\d.]+|-{3})\s*\]$")
TEX_TABLES = {"tab:nd_by_dim": "nd_by_dim", "tab:nd_by_size": "nd_by_size",
              "tab:2d_by_size": "2d_by_size",
              "tab:tsplib_by_size": "tsplib_by_size",
              "tab:tsplib_nonEuc": "tsplib_nonEuc", "tab:rank": "rank",
              # Both of these are written by this script and spliced into the
              # manuscript by splice_tables.py, but until now neither was read
              # back: --check covered six of the eight generated tables, so a
              # roster change could move a cell in the classical panel or the
              # per-generator panel and no gate would see it. They are the two
              # tables the withdrawal of Daganzo/Chien/Kwon rewrote most.
              "tab:genclass": "2d_by_genclass", "tab:classical": "classical",
              # Compact Section 4 body tables (write_tex_compact); projections of
              # the two tidy frames named, parsed by the compact branch below.
              **COMPACT_TABLES}
# Metrics offered to --check and to the number bank. A tidy frame contributes only
# the ones it actually carries, so table-specific columns (the rank block) can sit
# in the same list without every other table having to define them.
CHECK_METRICS = ("N", "SDPE_pct", "SDPE_lo", "SDPE_hi", "MAPE_pct", "MedAPE_pct",
                 "MSPE_pct", "R2", "R2_alpha", "time_ms",
                 "spearman_rho", "kendall_tau", "close5_pct", "close10_pct")
NUMBER_METRICS = ("SDPE_pct", "SDPE_lo", "SDPE_hi", "MAPE_pct", "MedAPE_pct", "MSPE_pct",
                  "RMSPE_pct", "R2", "R2_alpha", "time_ms",
                  "spearman_rho", "kendall_tau", "close5_pct", "close10_pct",
                  "close5_pairs", "close10_pairs",
                  "pair_mode", "pair_seed", "pair_draws", "pair_universe")


def _clean(cell: str) -> str:
    c = cell.strip()
    for _ in range(2):
        c = _TEXTB.sub(r"\1", c)
    # ``\\*`` is a row terminator that forbids a page break after the row; the
    # longtable appendix tables carry it so a bucket's \multirow label cannot be
    # split across pages. Stripped before the bare ``\\`` or the star would
    # survive into the last cell of every protected row and fail to parse.
    # ``\c{C}`` is folded back to ``C`` so a typeset name with a diacritic still
    # matches the ASCII display string that generated it (see TEX_NAME).
    return (c.replace(r"\scriptsize", "").replace("{,}", "").replace("~ms", "")
             .replace(r"\c{C}", "C").replace("$-$", "-")
             .replace("\\\\*", "").replace(r"\\", "").strip())


def _bucket_lead(cell: str) -> str:
    """Read a bucket label out of a ``\\multirow{n}{*}{\\makecell[l]{...}}`` lead.

    Peeled one wrapper at a time rather than with a single greedy regex, because
    the labels themselves contain braces (``2D \\texttt{random} ...``) and a
    greedy match would swallow the wrapper's own closing brace.
    """
    c = cell.strip()
    for pat in (r"\\multirow\{\d+\}\{\*\}\{(.*)\}$", r"\\makecell\[l\]\{(.*)\}$"):
        m = re.match(pat, c)
        if m:
            c = m.group(1).strip()
    return _clean(c)


def _num(s: str) -> tuple[float, int] | None:
    """Parse a printed cell into (value, decimals shown)."""
    try:
        v = float(s)
    except ValueError:
        return None
    return v, len(s.partition(".")[2])


def parse_tex(path: Path) -> dict[tuple[str, str, str, str], tuple[float, int] | None]:
    """Return {(table, bucket, model, metric): (value, decimals)} from the manuscript."""
    text = path.read_text(encoding="utf-8")
    out: dict[tuple[str, str, str, str], tuple[float, int] | None] = {}
    table, bucket = None, None
    compact: list[str] | None = None
    for raw in text.split("\n"):
        lab = re.search(r"\\label\{(tab:[\w:]+)\}", raw)
        if lab:
            table = TEX_TABLES.get(lab.group(1))
            compact = (COMPACT_MODELS[COMPACT_TABLES[lab.group(1)]]
                       if lab.group(1) in COMPACT_TABLES else None)
            continue
        if compact is not None:
            # bucket | Count | (MAPE, SDPE) x models. Gated on the label, not the
            # width: 2 + 2k cells collides with the appendix shape at k = 3.
            cells = [_clean(c) for c in raw.split("&")]
            if len(cells) != 2 + 2 * len(compact) or _num(cells[1]) is None:
                continue  # header rows, rules, blank lines
            bucket = cells[0]
            out[(table, bucket, "", "bucket_count")] = _num(cells[1])
            for i, m in enumerate(compact):
                disp = MODEL_LABELS[m]
                out[(table, bucket, disp, "MAPE_pct")] = _num(cells[2 + 2 * i])
                out[(table, bucket, disp, "SDPE_pct")] = _num(cells[3 + 2 * i])
            continue
        if r"\begin{table}" in raw:
            continue
        if table is None:
            continue
        mc = _MAKECELL.search(raw)
        if mc:
            bucket = _clean(mc.group(1))
            cnt = re.search(r"Count=([\d{,}]*\d)", raw)
            if cnt:
                out[(table, bucket, "", "bucket_count")] = _num(cnt.group(1).replace("{,}", ""))
        if "SDPE (\\%)" in raw or r"\toprule" in raw:  # header row
            continue
        cells = [_clean(c) for c in raw.split("&")]
        if table == "tsplib_nonEuc":
            # type | count | estimator | N | SDPE [CI] | MAPE | |Median| | MSPE.
            # Tested before the appendix shape: this table is also eight cells wide,
            # so the cell count alone cannot tell the two layouts apart.
            if len(cells) != 8:
                continue
            if cells[0]:
                bucket = _clean(re.sub(r"\\multirow\{\d+\}\{\*\}\{(.*)\}", r"\1", cells[0]))
                cnt = re.search(r"\{(\d+)\}$", cells[1])
                if cnt:
                    out[(table, bucket, "", "bucket_count")] = _num(cnt.group(1))
            model = cells[2]
            if not model:
                continue
            m = _CI.match(cells[4].replace(" ", ""))
            for k, v in {"N": _num(cells[3]),
                         "SDPE_pct": _num(m.group(1)) if m else None,
                         "SDPE_lo": _num(m.group(2)) if m else None,
                         "SDPE_hi": _num(m.group(3)) if m else None,
                         "MAPE_pct": _num(cells[5]), "MedAPE_pct": _num(cells[6]),
                         "MSPE_pct": _num(cells[7])}.items():
                out[(table, bucket, model, k)] = v
        elif table == "rank":
            # benchmark | model | N | Spearman | Kendall | close <5% | close <10%.
            # Gated on the table name, not on the width: seven cells is not unique
            # either (the classical table is seven wide), and a width-only branch
            # would shadow whichever layout it were tested before.
            if len(cells) != 7:
                continue
            if cells[0]:
                bucket = _clean(re.sub(r"\\multirow\{\d+\}\{\*\}\{(.*)\}", r"\1", cells[0]))
            model, n = cells[1], _num(cells[2])
            if not model or n is None:  # header row: $N$ does not parse
                continue
            for k, v in {"N": n, "spearman_rho": _num(cells[3]),
                         "kendall_tau": _num(cells[4]), "close5_pct": _num(cells[5]),
                         "close10_pct": _num(cells[6])}.items():
                out[(table, bucket, model, k)] = v
        elif table == "classical":
            # domain | estimator | N | SDPE [CI] | MAPE | |Median| | MSPE.
            # Seven cells wide, like the rank block, so this is gated on the
            # table name too. Its \makecell carries no ``Count=``, so _MAKECELL
            # never fires and the domain label is read off the row lead here.
            if len(cells) != 7:
                continue
            if cells[0]:
                bucket = _bucket_lead(cells[0])
            model = cells[1]
            if not model:
                continue
            m = _CI.match(cells[3].replace(" ", ""))
            for k, v in {"N": _num(cells[2]),
                         "SDPE_pct": _num(m.group(1)) if m else None,
                         "SDPE_lo": _num(m.group(2)) if m else None,
                         "SDPE_hi": _num(m.group(3)) if m else None,
                         "MAPE_pct": _num(cells[4]), "MedAPE_pct": _num(cells[5]),
                         "MSPE_pct": _num(cells[6])}.items():
                out[(table, bucket, model, k)] = v
        elif len(cells) == 8 and bucket:  # appendix shape
            model, metrics = cells[1], cells[2:]
            if "Reference-tour" in model:
                continue
            m = _CI.match(metrics[0].replace(" ", ""))
            vals = {"SDPE_pct": _num(m.group(1)) if m else None,
                    "SDPE_lo": _num(m.group(2)) if m else None,
                    "SDPE_hi": _num(m.group(3)) if m else None,
                    "MAPE_pct": _num(metrics[1]), "MedAPE_pct": _num(metrics[2]),
                    TEX_STD_COL6_IN_TEX: _num(metrics[3]), "R2_alpha": _num(metrics[4]),
                    "time_ms": _num(metrics[5])}
            for k, v in vals.items():
                out[(table, bucket, model, k)] = v
    return out


def run_check(tables: dict[str, pd.DataFrame]) -> None:
    old = parse_tex(P_TEX)
    new: dict[tuple[str, str, str, str], object] = {}
    for name, tidy in tables.items():
        for _, r in tidy.iterrows():
            key_b = _clean(str(r["bucket"]))
            new[(name, key_b, "", "bucket_count")] = r["bucket_count"]
            # Both R2 and MSPE_pct are offered so the comparison works whichever
            # one column six of the manuscript currently prints; only one of the
            # two is ever parsed out of the .tex.
            # "N" is printed by the non-Euclidean and rank tables, where the
            # estimators do not all score the same instances; elsewhere unmatched.
            for metric in CHECK_METRICS:
                if metric in r.index:
                    new[(name, key_b, str(r["display"]), metric)] = r[metric]
    moved, appeared, vanished, missing = [], [], [], []
    for key, cell in sorted(old.items()):
        if key not in new:
            missing.append(key)
            continue
        nv = new[key]
        nvf = float(nv) if not isinstance(nv, str) and np.isfinite(float(nv)) else None
        if cell is None and nvf is not None:
            appeared.append((key, nvf))
        elif cell is not None and nvf is None:
            vanished.append((key, cell[0]))
        elif cell is not None and nvf is not None:
            ov, ndec = cell
            # Compare at the precision the manuscript actually prints: half a unit
            # in the last printed place, never looser than the historical 0.005.
            # Without this the four-decimal rank correlations would be invisible to
            # --check, since every plausible tau shift is far below 0.005.
            if abs(ov - round(nvf, ndec)) > min(0.005, 0.5 * 10.0 ** -ndec):
                moved.append((key, ov, round(nvf, ndec), nvf))
    # The manuscript states this count in prose ("re-derives all N table
    # cells"), and until now it was a hand-typed figure with no generator behind
    # it -- so a roster change that added or removed table rows made the
    # sentence wrong silently. Bank it, and the prose checker owns it like any
    # other number.
    bank_path = OUT / "paper_numbers.json"
    if bank_path.exists():
        banked = json.loads(bank_path.read_text(encoding="utf-8"))
        if banked.get("manuscript_table_cells") != len(old):
            banked["manuscript_table_cells"] = len(old)
            bank_path.write_text(json.dumps(banked, indent=2, sort_keys=True),
                                 encoding="utf-8")
    print(f"\n{'=' * 82}\n--check: manuscript vs regenerated ({len(old)} manuscript cells)\n{'=' * 82}")
    print(f"printed value changes: {len(moved)} | now defined: {len(appeared)} | now '---': {len(vanished)} "
          f"| cell not regenerated: {len(missing)}\n")
    if moved:
        print(f"{'table':<15}{'bucket':<22}{'model':<22}{'metric':<11}{'paper':>10}{'new':>10}{'delta':>10}{'raw':>12}")
        for (t, b, m, k), ov, nv, raw in moved:
            print(f"{t:<15}{b:<22}{m:<22}{k:<11}{ov:>10.3f}{nv:>10.3f}{nv - ov:>+10.3f}{raw:>12.4f}")
    for tag, items in (("NOW DEFINED (was ---)", appeared), ("NOW '---' (was numeric)", vanished)):
        if items:
            print(f"\n{tag}:")
            for (t, b, m, k), v in items:
                print(f"  {t:<15}{b:<22}{m:<22}{k:<12}{v:>10.3f}")
    if missing:
        print("\nMANUSCRIPT CELLS WITH NO REGENERATED COUNTERPART:")
        for t, b, m, k in missing:
            print(f"  {t:<15}{b:<22}{m:<22}{k}")
    # --check covers table cells only. The prose numbers are a separate surface
    # with its own history of going stale; check_prose_numbers.py owns them.
    print("\nTable cells only -- prose numbers are not checked here. Run:"
          "\n  python paper_tooling/check_prose_numbers.py")


# -- Orchestration ----------------------------------------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="diff regenerated values against Area_Free_Main.tex")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d2, cov2, ref2 = load_2d()
    dn, covn, refn_size, refn_dim = load_nd()
    euc, non, covt = load_tsplib()
    rnd = d2[d2["generator"] == "random"]

    # Trailing flag: the manuscript typesets this table as a longtable, so its
    # rows carry the no-break terminator (see write_tex_std). 2d_by_size and
    # tsplib_by_size joined that set when they stopped being \resizebox floats;
    # 2d_by_size in particular used to be split by hand across two floats, and
    # the longtable now paginates it instead.
    specs = [
        ("2d_by_size", d2, B_2D_SIZE, "prediction_time_s", "2d", ref2, True),
        ("2d_by_genclass", d2, B_GENCLASS, "prediction_time_s", "2d", None, True),
        ("2d_random_by_size", rnd, B_2D_SIZE, "prediction_time_s", "2d", None, False),
        ("2d_random_n10_80", rnd[(rnd["n_customers"] >= 10) & (rnd["n_customers"] <= 80)],
         [(r"$n\in[10,80]$", "n10_80", _all())], "prediction_time_s", "2d", None, False),
        ("nd_by_size", dn, B_ND_SIZE, "prediction_time_s", "nd", refn_size, True),
        ("nd_by_dim", dn, B_ND_DIM, "prediction_time_s", "nd", refn_dim, True),
        ("tsplib_by_size", euc, B_TSPLIB, "total_time_s", "tsplib", None, True),
    ]
    tables: dict[str, pd.DataFrame] = {}
    for name, df, buckets, tcol, roster, refs, nobreak in specs:
        tidy = compute_table(df, buckets, tcol, name)
        tidy.to_csv(OUT / f"table_{name}.csv", index=False)
        write_tex_std(tidy, buckets, TEX_MODELS[roster], refs,
                      OUT / f"table_{name}.tex", nobreak=nobreak)
        tables[name] = tidy
    for label, name in COMPACT_TABLES.items():
        buckets = B_ND_DIM if name == "nd_by_dim" else B_GENCLASS
        models = COMPACT_MODELS[name]
        stem = label.split(":", 1)[1]
        frag = OUT / f"table_{stem}.tex"
        write_tex_compact(tables[name], buckets, models, frag)
        (OUT / f"table_{stem}_header.tex").write_text(
            compact_header(models) + "\n", encoding="utf-8")
        # The whole tabular, header included, for splice_tables.TABULAR: a roster
        # change moves the column count, which the body-only splice cannot follow.
        note = ""
        if any("dagger" in lbl for lbl, _, _ in buckets):
            note = (rf"\multicolumn{{{2 + 2 * len(models)}}}{{l}}"
                    r"{\footnotesize $^{\dagger}$ Not represented in training.} \\" "\n")
        (OUT / f"table_{stem}_tabular.tex").write_text(
            compact_header(models) + "\n" + frag.read_text(encoding="utf-8")
            + "\\bottomrule\n" + note + "\\end{tabular}", encoding="utf-8")
    ne = compute_table(non, B_NONEUC, "total_time_s", "tsplib_nonEuc")
    ne.to_csv(OUT / "table_tsplib_nonEuc.csv", index=False)
    write_tex_noneuc(ne, B_NONEUC, TEX_MODELS["tsplib_nonEuc"], OUT / "table_tsplib_nonEuc.tex")
    tables["tsplib_nonEuc"] = ne

    rank = pd.concat([compute_rank(d2, "2D", "2d"),
                      compute_rank(dn, "Multidimensional", "nd"),
                      compute_rank(with_31f_controls(euc), r"TSPLIB EUC\_2D", "tsplib_euc2d")],
                     ignore_index=True)
    rank.to_csv(OUT / "table_rank.csv", index=False)
    write_tex_rank(rank, OUT / "table_rank.tex")
    tables["rank"] = rank

    classical = compute_classical(d2, euc)
    classical.to_csv(OUT / "table_classical.csv", index=False)
    write_tex_classical(classical, OUT / "table_classical.tex", nobreak=True)
    for panel, tag in [("A", "full"), ("B", "matched")]:
        sub = classical[classical["panel"] == panel]
        sub.to_csv(OUT / f"table_classical_{tag}.csv", index=False)
        write_tex_classical(sub, OUT / f"table_classical_{tag}.tex")
    tables["classical"] = classical

    pd.concat([cov2, covn, covt], ignore_index=True).to_csv(OUT / "coverage.csv", index=False)

    pt: list[dict[str, object]] = []
    for name, df, buckets in [("2d_by_size", d2, B_2D_SIZE), ("nd_by_size", dn, B_ND_SIZE),
                              ("tsplib_by_size", euc, B_TSPLIB)]:
        for label, slug, fn in buckets:
            mask = fn(df)
            for m in sorted(set(df["model"])):
                if m == GART or m not in MODEL_LABELS:
                    continue
                pt.append({"table": name, "bucket": label, "bucket_slug": slug, "model_a": GART,
                           "model_b": m, "display_b": MODEL_LABELS[m], **paired_test(df, GART, m, mask)})
    # Matched-domain head-to-heads: the classical panel-B supports, where each
    # published regression is only compared on the instances it actually covers.
    for label, slug, sub in (
        [(r"2D \texttt{random}", "b_random", rnd)]
        + [(lbl, sg, rnd[rnd["instance"].isin(set(rnd.loc[rnd["model"] == anchor, "instance"]))])
           for anchor, lbl, sg in CLASSICAL_SUBDOMAINS]
    ):
        for m in CLASSICAL_MATCHED + CLASSICAL_PAIRED_EXTRA:
            if m == GART or m not in MODEL_LABELS:
                continue
            pt.append({"table": "classical", "bucket": label, "bucket_slug": slug, "model_a": GART,
                       "model_b": m, "display_b": MODEL_LABELS[m],
                       **paired_test(sub, GART, m, pd.Series(True, index=sub.index))})
    paired = pd.DataFrame(pt)
    paired.to_csv(OUT / "paired_tests.csv", index=False)

    numbers: dict[str, object] = {}
    for name, tidy in tables.items():
        for _, r in tidy.iterrows():
            base = f"{name}_{r['bucket_slug']}_{_slug(str(r['display']))}"
            numbers[f"{base}_n"] = int(r["N"])
            for metric in NUMBER_METRICS:
                if metric not in r.index:
                    continue
                v = r[metric]
                numbers[f"{base}_{_slug(metric)}"] = v if isinstance(v, str) else (
                    round(float(v), 6) if np.isfinite(float(v)) else None)
    # Readable aliases for the headline values the prose quotes.
    for short, full in [("2d", "2d_by_size_total"), ("nd", "nd_by_size_total"),
                        ("tsplib_euc2d", "tsplib_by_size_total"), ("tsplib_nonEuc", "tsplib_nonEuc_total"),
                        ("2d_linenoise", "2d_by_genclass_linenoise")]:
        for metric in ("sdpe_pct", "mape_pct", "medape_pct", "n"):
            src = f"{full}_gart_2_0_{metric}"
            if src in numbers:
                numbers[f"{short}_gart_{metric.replace('_pct', '')}"] = numbers[src]
    # Significance tests. Same function paper_tooling/paired_bank.py exposes as a
    # standalone merge, so a full rebuild and a paired-only refresh emit identical
    # keys. Without this every p-value and bootstrap interval in the prose is
    # unbackable -- paired_tests.csv was written and then never read.
    numbers.update(paired_bank_numbers(paired))
    # Leave-out refits. Same function paper_tooling/generalization_bank.py exposes
    # as a standalone merge, so a full rebuild and a generalization-only refresh
    # emit identical keys. Without this a full rebuild would delete the keys the
    # refits wrote, and Section 4.6 would go back to being unbackable prose.
    numbers.update(generalization_bank_numbers(load_generalization_results()))
    # Swept-feature monotonicity and the ND dispersion series, for the
    # production model and the two comparators that beat it on raw accuracy.
    # Same function paper_tooling/consistency_bank.py exposes as a standalone
    # merge. Returns {} until consistency_31f.py has been run, so a fresh
    # clone rebuilds tables without these keys rather than failing.
    numbers.update(consistency_bank_numbers(*load_consistency_results()))
    # Keys owned by the standalone exporters: SHAP attribution, the production
    # model's realised alpha range, and the model-independent corpus facts.
    # Same reason as the two above -- a full rebuild would otherwise delete
    # them, and the claims that point at them would go back to being
    # unverifiable. Each returns {} when its exporter has not been run, so a
    # fresh clone rebuilds tables without them rather than failing.
    for owner, fn in (("shap_production", shap_bank_numbers),
                      ("shap_by_dimension", shap_band_bank_numbers),
                      ("alphahat_range", alphahat_bank_numbers),
                      ("corpus_statistics", corpus_bank_numbers),
                      ("cavdar_correction_bank", cavdar_correction_bank_numbers),
                      ("constraint_transfer_bank", constraint_transfer_carried)):
        carried = fn()
        if carried:
            numbers.update(carried)
        else:
            print(f"note: no banked keys from {owner}; run "
                  f"paper_tooling/{owner}.py to generate them")
    # Two banner counts are banked by the checkers themselves rather than
    # derived here: manuscript_table_cells by run_check below, and
    # frontier_table_cells by paper_tooling/check_frontier_tables.py. Neither is
    # recomputable from anything in this function, so a full rebuild has to
    # carry them forward -- same reason the exporters above are carried. Without
    # this, running --check (which rebuilds before it checks) deleted
    # frontier_table_cells and the prose claim pointing at it failed until the
    # frontier checker happened to be run again, making the gate order-dependent.
    prior_path = OUT / "paper_numbers.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        for carried_key in ("frontier_table_cells", "manuscript_table_cells"):
            if carried_key in prior:
                numbers.setdefault(carried_key, prior[carried_key])
    (OUT / "paper_numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {len(tables)} tables + coverage.csv, paired_tests.csv, paper_numbers.json to {OUT}")
    print("\nCOVERAGE (rows dropped by status):")
    cov = pd.concat([cov2, covn, covt], ignore_index=True)
    print(cov[["dataset", "model", "status", "n_rows", "status_kept", "n_used"]].to_string(index=False))
    report_withheld_timings()
    if args.check:
        run_check(tables)


if __name__ == "__main__":
    main()
