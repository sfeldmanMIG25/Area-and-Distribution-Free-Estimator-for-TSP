"""
regenerate_benchmark_tables.py
==============================

Master, idempotent script that regenerates the canonical benchmark tables
and box-plot figures used in Section 5 of the GART 2.0 paper.

v3 changes (2026-04-17):
    - Display name "GART 3.0" renamed to "GART 2.0" everywhere. The
      internal model column in the CSVs is still ``LGBM_V3``.
    - R^2_alpha is now computed *within each bucket* (not reused from the
      full dataset). A bucket with < 3 instances, or where true_alpha
      std < 1e-4, emits ``---``.
    - TSPLIB aggregates drop instances whose optimum exceeds the
      double-tree bound (true_cost / MST > METRIC_RATIO_THRESHOLD = 2.5,
      i.e. 25% above the metric ceiling of 2). This is a deterministic,
      model-agnostic filter; brg180 is the only current TSPLIB instance
      that trips it, but future additions are handled automatically.
    - TSPLIB by-size buckets replaced with finer splits:
      n<=50, [51,150], [151,500], [501,2000], [2001,10000], n>10000, ALL.
    - New non-Euclidean TSPLIB summary table ``table_tsplib_nonEuc.csv``
      with one row per non-Euclidean edge_weight_type (CEIL_2D, ATT,
      GEO, EXPLICIT-metric; non-metric outliers removed from EXPLICIT).

Instances that fail the double-tree bound (true_cost / MST >
METRIC_RATIO_THRESHOLD) are dropped from every aggregate; GART 2.0
assumes a symmetric metric distance matrix.
"""

from __future__ import annotations

import gc
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tsplib_benchmark"))
from exclusions import METRIC_RATIO_THRESHOLD, filter_metric_consistent  # noqa: E402
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
PATH_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
PATH_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
PATH_TSPLIB = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
PATH_TSPLIB_SUPPL = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_supplemental.csv"
OUT_DIR = ROOT / "paper_reference"
TAB_DIR = OUT_DIR / "scripts" / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Model roster (Vinel dropped)
# --------------------------------------------------------------------------- #
MODELS_2D = [
    ("GART 2.0", "LGBM_V3"),
    ("MST Ratio", "MST_Ratio"),
    ("Cavdar",    "Cavdar"),
    ("BHH",       "BHH"),
    ("Chien",     "Chien"),
    ("Hilbert",   "Hilbert"),
]

# ND: only dim-agnostic trio
MODELS_ND = [
    ("GART 2.0", "LGBM_V3"),
    ("MST Ratio", "MST_Ratio"),
    ("Hilbert",   "Hilbert"),
]

# TSPLIB: same six as 2D
MODELS_TSPLIB = MODELS_2D

# Models that predict alpha directly (R^2_alpha is reported)
ALPHA_MODELS = {"LGBM_V3", "MST_Ratio"}
ALPHA_MST_RATIO_2D = 1.075  # MST_Ratio multiplier used for 2D instances.

# --------------------------------------------------------------------------- #
# Memory-safety utilities
# --------------------------------------------------------------------------- #
def load_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(columns))


def stream_filter(
    path: Path,
    columns: Sequence[str],
    model_filter: Optional[Iterable[str]] = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    keep = set(model_filter) if model_filter is not None else None
    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=list(columns), chunksize=chunksize):
        if keep is not None and "model" in chunk.columns:
            chunk = chunk[chunk["model"].isin(keep)]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return out


def free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _r2_safe(y_true: np.ndarray, y_pred: np.ndarray,
             min_n: int = 3, min_std: float = 0.0) -> float:
    """R^2 with safety rails.

    Returns NaN when the group has fewer than ``min_n`` finite pairs, or
    when the std of ``y_true`` falls below ``min_std`` (treated as
    near-constant, R^2 ill-defined).
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < min_n:
        return np.nan
    s = float(np.std(y_true[mask]))
    if s <= min_std:
        return np.nan
    return float(r2_score(y_true[mask], y_pred[mask]))


def compute_metrics_row(
    df_group: pd.DataFrame,
    display_name: str,
    raw_name: str,
    support_alpha: bool,
    time_col: str,
) -> dict:
    """Compute all per-model metrics, using ``time_col`` (seconds) for timing.

    ``time_col`` is the name of the 'total wall-clock' column in the loaded
    frame, e.g. ``total_time_s`` (TSPLIB) or ``prediction_time_s`` (2D/ND).
    ``time_ms`` is reported as that column * 1000.
    """
    g = df_group
    n = len(g)
    if n == 0:
        return {
            "model": display_name,
            "N": 0,
            "MAPE_pct": np.nan,
            "SDPE_pct": np.nan,
            "Median_abs_pct_err": np.nan,
            "R2": np.nan,
            "R2_alpha": "---",
            "time_ms": np.nan,
        }

    true_c = g["true_cost"].to_numpy(dtype=float)
    pred_c = g["pred_cost"].to_numpy(dtype=float)
    mask = np.isfinite(true_c) & np.isfinite(pred_c) & (true_c != 0)
    true_c = true_c[mask]
    pred_c = pred_c[mask]

    pe = (pred_c - true_c) / true_c * 100.0
    ape = np.abs(pe)

    mape = float(ape.mean())
    sdpe = float(pe.std(ddof=1)) if len(pe) > 1 else np.nan
    medape = float(np.median(ape))
    r2 = _r2_safe(true_c, pred_c)

    # R^2 alpha -- computed *within this bucket only*, using this bucket's
    # rows. Emits '---' if fewer than 3 instances or true_alpha std < 1e-4.
    r2a = "---"
    if support_alpha and "mst_length" in g.columns:
        ml = g.loc[mask, "mst_length"].to_numpy(dtype=float)
        vm = np.isfinite(ml) & (ml > 0)
        if vm.sum() >= 3:
            alpha_true = true_c[vm] / ml[vm]
            alpha_pred = pred_c[vm] / ml[vm]
            r2a_val = _r2_safe(alpha_true, alpha_pred,
                               min_n=3, min_std=1e-4)
            if np.isfinite(r2a_val):
                r2a = f"{r2a_val:.4f}"

    # Timing (always total wall-clock, reported in ms)
    if time_col in g.columns:
        t = g[time_col].to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        time_ms = float(t.mean() * 1000.0) if len(t) else np.nan
    else:
        time_ms = np.nan

    return {
        "model": display_name,
        "N": int(mask.sum()),
        "MAPE_pct": round(mape, 4),
        "SDPE_pct": round(sdpe, 4) if np.isfinite(sdpe) else np.nan,
        "Median_abs_pct_err": round(medape, 4),
        "R2": round(r2, 4) if np.isfinite(r2) else np.nan,
        "R2_alpha": r2a,
        "time_ms": round(time_ms, 4) if np.isfinite(time_ms) else np.nan,
    }


# --------------------------------------------------------------------------- #
# Bucket helpers
# --------------------------------------------------------------------------- #
SIZE_BUCKETS_GENERIC = [
    ("[5,10]",      lambda n: (n >= 5)    & (n <= 10)),
    ("[11,50]",     lambda n: (n >= 11)   & (n <= 50)),
    ("[51,100]",    lambda n: (n >= 51)   & (n <= 100)),
    ("[101,200]",   lambda n: (n >= 101)  & (n <= 200)),
    ("[201,500]",   lambda n: (n >= 201)  & (n <= 500)),
    ("[501,1000]",  lambda n: (n >= 501)  & (n <= 1000)),
    ("ALL",         lambda n: np.ones_like(n, dtype=bool)),
]

DIM_BUCKETS_ND = [
    ("d=2",      lambda d: d == 2),
    ("d=3-5",    lambda d: (d >= 3) & (d <= 5)),
    ("d=6-10",   lambda d: (d >= 6) & (d <= 10)),
    ("d=15-25",  lambda d: (d >= 15) & (d <= 25)),
    ("d=30-50", lambda d: (d >= 30) & (d <= 50)),
    ("d=100",    lambda d: d == 100),
    ("ALL",      lambda d: np.ones_like(d, dtype=bool)),
]

SIZE_BUCKETS_TSPLIB = [
    ("n<=50",        lambda n: n <= 50),
    ("[51,150]",     lambda n: (n >= 51)   & (n <= 150)),
    ("[151,500]",    lambda n: (n >= 151)  & (n <= 500)),
    ("[501,2000]",   lambda n: (n >= 501)  & (n <= 2000)),
    ("[2001,10000]", lambda n: (n >= 2001) & (n <= 10000)),
    ("n>10000",      lambda n: n > 10000),
    ("ALL",          lambda n: np.ones_like(n, dtype=bool)),
]

# Non-metric instances (true_cost / MST > METRIC_RATIO_THRESHOLD) are
# dropped from every aggregate via ``filter_metric_consistent``. The
# threshold is imported from ``tsplib_benchmark/exclusions.py``.


def build_table(
    df: pd.DataFrame,
    bucket_col: str,
    buckets: list,
    models: List[tuple],
    time_col: str,
    label_name: str = "bucket",
) -> pd.DataFrame:
    rows = []
    col_vals = df[bucket_col].to_numpy()
    for bname, bfn in buckets:
        bmask = bfn(col_vals)
        sub = df.loc[bmask]
        for disp, raw in models:
            g = sub[sub["model"] == raw]
            row = compute_metrics_row(
                g, disp, raw, support_alpha=(raw in ALPHA_MODELS), time_col=time_col,
            )
            row[label_name] = bname
            rows.append(row)
    out = pd.DataFrame(rows)
    cols = [label_name, "model", "N", "MAPE_pct", "SDPE_pct",
            "Median_abs_pct_err", "R2", "R2_alpha", "time_ms"]
    return out[cols]


# --------------------------------------------------------------------------- #
# Data loaders (per-benchmark)
# --------------------------------------------------------------------------- #
_N_PATTERN = re.compile(r"-n(\d+)-")


def load_2d() -> pd.DataFrame:
    """Load 2D benchmark. Derive n from instance; derive mst_length from MST_Ratio."""
    cols = [
        "model", "instance",
        "pred_cost", "true_cost",
        "prediction_time_s", "feature_time_s", "inference_time_s",
    ]
    raw_names = {m[1] for m in MODELS_2D}
    df = stream_filter(PATH_2D, cols, model_filter=raw_names, chunksize=50_000)
    n_vals = df["instance"].str.extract(_N_PATTERN, expand=False)
    df["n"] = pd.to_numeric(n_vals, errors="coerce")
    df = df.dropna(subset=["n"]).copy()
    df["n"] = df["n"].astype(int)

    # Derive mst_length per instance from MST_Ratio row:
    # MST_Ratio pred_cost = 1.075 * mst_length  =>  mst_length = pred_cost / 1.075
    mst_rows = df[df["model"] == "MST_Ratio"][["instance", "pred_cost"]].copy()
    mst_rows["mst_length"] = mst_rows["pred_cost"] / ALPHA_MST_RATIO_2D
    mst_map = dict(zip(mst_rows["instance"], mst_rows["mst_length"]))
    df["mst_length"] = df["instance"].map(mst_map)
    # total_time_s alias: prediction_time_s (2D has no explicit total col)
    df["total_time_s"] = df["prediction_time_s"]
    return df


def load_nd() -> pd.DataFrame:
    cols = [
        "model", "instance",
        "pred_cost", "true_cost",
        "prediction_time_s", "feature_time_s", "inference_time_s",
        "n_customers", "dimension", "mst_length",
    ]
    raw_names = {m[1] for m in MODELS_ND}
    df = stream_filter(PATH_ND, cols, model_filter=raw_names, chunksize=100_000)
    df = df.dropna(subset=["n_customers", "dimension", "true_cost"]).copy()
    df["n_customers"] = df["n_customers"].astype(int)
    df["dimension"] = df["dimension"].astype(int)
    df = df.rename(columns={"n_customers": "n", "dimension": "d"})
    df["total_time_s"] = df["prediction_time_s"]
    return df


def load_tsplib_common() -> pd.DataFrame:
    """TSPLIB loader: merges the original CSV with the supplemental CSV of
    large-instance baseline runs, then restricts to the EUC_2D common set
    covered by every model in ``MODELS_TSPLIB``.

    Drops non-metric instances (true_cost / MST > METRIC_RATIO_THRESHOLD)
    before any aggregation.
    """
    cols = [
        "instance", "n", "edge_weight_type", "model",
        "pred_cost", "true_cost", "mst_length",
        "total_time_s", "feature_time_s", "inference_time_s",
    ]
    raw_names = {m[1] for m in MODELS_TSPLIB}
    df = load_columns(PATH_TSPLIB, cols)

    # Merge supplemental rows if present
    if PATH_TSPLIB_SUPPL.exists():
        sup = pd.read_csv(PATH_TSPLIB_SUPPL, usecols=cols)
        df = pd.concat([df, sup], ignore_index=True)

    # Drop non-metric instances (true_cost / MST > METRIC_RATIO_THRESHOLD).
    # GART 2.0 assumes symmetric metric distances; anything past the
    # double-tree bound is outside the model's defined regime.
    df = filter_metric_consistent(df)

    df = df[df["edge_weight_type"] == "EUC_2D"]
    df = df[df["model"].isin(raw_names)].copy()

    # De-dup: prefer the first occurrence on (instance, model)
    df = df.drop_duplicates(subset=["instance", "model"], keep="first")

    inst_per_model = df.groupby("model")["instance"].apply(set)
    if len(inst_per_model) == 0:
        return df
    common = set.intersection(*inst_per_model.values)
    df = df[df["instance"].isin(common)].copy()
    return df


def load_tsplib_nonEuc() -> pd.DataFrame:
    """Load TSPLIB non-Euclidean rows (GART 2.0 only) for the summary
    table in Section 6. Returns a frame restricted to edge_weight_type in
    {CEIL_2D, ATT, GEO, EXPLICIT} with triangle-inequality violators
    removed. Includes ``feature_dim`` for embedding-dimension reporting.
    """
    cols = [
        "instance", "n", "edge_weight_type", "model",
        "pred_cost", "true_cost", "mst_length",
        "total_time_s", "feature_time_s", "inference_time_s",
        "feature_dim",
    ]
    df = pd.read_csv(PATH_TSPLIB, usecols=cols)
    df = filter_metric_consistent(df)
    df = df[df["edge_weight_type"].isin(["CEIL_2D", "ATT", "GEO", "EXPLICIT"])]
    df = df[df["model"] == "LGBM_V3"].copy()
    return df


def build_tsplib_nonEuc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the §6 non-Euclidean summary: one row per EWT with
    N, MAPE, SDPE, median |% err|, R^2_alpha, avg time (ms), avg k, k range.
    """
    rows = []
    for ewt in ["CEIL_2D", "ATT", "GEO", "EXPLICIT"]:
        sub = df[df["edge_weight_type"] == ewt].copy()
        if sub.empty:
            continue
        true_c = sub["true_cost"].to_numpy(dtype=float)
        pred_c = sub["pred_cost"].to_numpy(dtype=float)
        ml = sub["mst_length"].to_numpy(dtype=float)
        t = sub["total_time_s"].to_numpy(dtype=float)
        k = sub["feature_dim"].to_numpy(dtype=float)

        mask = (np.isfinite(true_c) & np.isfinite(pred_c)
                & (true_c != 0))
        true_c = true_c[mask]
        pred_c = pred_c[mask]
        ml = ml[mask]
        t = t[mask]
        k = k[mask]

        pe = (pred_c - true_c) / true_c * 100.0
        ape = np.abs(pe)
        mape = float(ape.mean())
        sdpe = float(pe.std(ddof=1)) if len(pe) > 1 else np.nan
        medape = float(np.median(ape))

        # R^2_alpha within this EWT only.
        vm = np.isfinite(ml) & (ml > 0)
        r2a = "---"
        if vm.sum() >= 3:
            a_true = true_c[vm] / ml[vm]
            a_pred = pred_c[vm] / ml[vm]
            v = _r2_safe(a_true, a_pred, min_n=3, min_std=1e-4)
            if np.isfinite(v):
                r2a = f"{v:.4f}"

        tfin = t[np.isfinite(t)]
        time_ms = float(tfin.mean() * 1000.0) if len(tfin) else np.nan

        kfin = k[np.isfinite(k)]
        k_mean = float(kfin.mean()) if len(kfin) else np.nan
        k_min = int(kfin.min()) if len(kfin) else -1
        k_max = int(kfin.max()) if len(kfin) else -1

        # Label: "EXPLICIT" gets clarified as "EXPLICIT-metric" in the paper.
        label = "EXPLICIT-metric" if ewt == "EXPLICIT" else ewt

        rows.append({
            "edge_weight_type": label,
            "N": int(mask.sum()),
            "MAPE_pct": round(mape, 4),
            "SDPE_pct": round(sdpe, 4) if np.isfinite(sdpe) else np.nan,
            "Median_abs_pct_err": round(medape, 4),
            "R2_alpha": r2a,
            "time_ms": round(time_ms, 4) if np.isfinite(time_ms) else np.nan,
            "avg_k": round(k_mean, 2) if np.isfinite(k_mean) else np.nan,
            "k_range": f"[{k_min},{k_max}]" if k_min >= 0 else "---",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Time-breakdown table
# --------------------------------------------------------------------------- #
def time_breakdown(df: pd.DataFrame, dataset_label: str,
                   models: List[tuple]) -> pd.DataFrame:
    """Compute average percentage of total_time_s spent in features /
    MST / inference / other for each model.

    None of the three source CSVs have a dedicated ``mst_time_s`` column --
    MST computation is bundled inside feature extraction. We therefore report
    ``pct_mst = NaN`` for every row (the dataset is flagged 'partial' in the
    returned frame via the ``notes`` column). Percentages are computed
    per-row then averaged to match a macro-average interpretation.
    """
    rows = []
    for disp, raw in models:
        sub = df[df["model"] == raw].copy()
        if sub.empty:
            continue

        total = sub["total_time_s"].to_numpy(dtype=float)
        feat = sub.get("feature_time_s",
                       pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)
        inf = sub.get("inference_time_s",
                      pd.Series([np.nan] * len(sub))).to_numpy(dtype=float)

        valid = np.isfinite(total) & (total > 0)
        if not valid.any():
            rows.append({
                "dataset": dataset_label,
                "model": disp,
                "pct_features": np.nan,
                "pct_mst": np.nan,
                "pct_inference": np.nan,
                "pct_other": np.nan,
                "avg_total_ms": np.nan,
                "N": int(len(sub)),
            })
            continue

        pct_feat = np.where(valid & np.isfinite(feat),
                            feat / total * 100.0, np.nan)
        pct_inf = np.where(valid & np.isfinite(inf),
                           inf / total * 100.0, np.nan)
        # MST not logged as a distinct column -> NaN.
        pct_mst = np.full(len(sub), np.nan)

        # "Other" = residual. We can only compute it when the log actually
        # separates feature & inference; otherwise it is NaN.
        pct_other = np.where(
            valid & np.isfinite(feat) & np.isfinite(inf),
            100.0 - (pct_feat + pct_inf),
            np.nan,
        )

        rows.append({
            "dataset": dataset_label,
            "model": disp,
            "pct_features": round(float(np.nanmean(pct_feat)), 2)
                            if np.isfinite(pct_feat).any() else np.nan,
            "pct_mst": np.nan,
            "pct_inference": round(float(np.nanmean(pct_inf)), 2)
                             if np.isfinite(pct_inf).any() else np.nan,
            "pct_other": round(float(np.nanmean(pct_other)), 2)
                         if np.isfinite(pct_other).any() else np.nan,
            "avg_total_ms": round(float(np.nanmean(total[valid])) * 1000.0, 4),
            "N": int(valid.sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def preview(df: pd.DataFrame, title: str, n: Optional[int] = None):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 20,
    ):
        if n is None:
            print(df.to_string(index=False))
        else:
            print(df.head(n).to_string(index=False))


# --------------------------------------------------------------------------- #
# Box-plot utility
# --------------------------------------------------------------------------- #
def boxplot(
    df: pd.DataFrame,
    models: List[tuple],
    title: str,
    out_path: Path,
    ylim: tuple = (-60.0, 60.0),
):
    true_c = df["true_cost"].to_numpy(dtype=float)
    pred_c = df["pred_cost"].to_numpy(dtype=float)
    pe_all = np.where(
        (true_c != 0) & np.isfinite(true_c) & np.isfinite(pred_c),
        (pred_c - true_c) / true_c * 100.0,
        np.nan,
    )
    df = df.copy()
    df["pct_err"] = pe_all

    labels = [m[0] for m in models]
    data = [
        df.loc[df["model"] == m[1], "pct_err"].dropna().to_numpy()
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(
        data, labels=labels, showfliers=True, patch_artist=True,
        flierprops=dict(marker=".", markersize=2.5, alpha=0.35),
        medianprops=dict(color="black", linewidth=1.5),
    )
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c",
               "#d62728", "#9467bd", "#8c564b"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Percent error  (pred - true) / true  [%]")
    ax.set_ylim(*ylim)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Loading 2D benchmark ...")
    df2d = load_2d()
    print(f"  2D rows loaded: {len(df2d):,}")

    print("Loading ND benchmark (chunked) ...")
    dfnd = load_nd()
    print(f"  ND rows loaded: {len(dfnd):,}")

    print("Loading TSPLIB benchmark (+ supplemental) ...")
    dftl = load_tsplib_common()
    common_inst = dftl["instance"].nunique()
    print(f"  TSPLIB common EUC_2D instances (intersection): {common_inst}")
    print(f"  TSPLIB rows loaded: {len(dftl):,}")

    # ------------------- Table 1: 2D by size ------------------- #
    tab_2d = build_table(
        df2d, bucket_col="n",
        buckets=SIZE_BUCKETS_GENERIC,
        models=MODELS_2D, label_name="n_bucket",
        time_col="total_time_s",
    )
    p = TAB_DIR / "table_2d_by_size.csv"
    tab_2d.to_csv(p, index=False)
    preview(tab_2d, f"table_2d_by_size.csv  ->  {p}")

    # ------------------- Table 2: ND by size ------------------- #
    tab_nd_sz = build_table(
        dfnd, bucket_col="n",
        buckets=SIZE_BUCKETS_GENERIC,
        models=MODELS_ND, label_name="n_bucket",
        time_col="total_time_s",
    )
    p = TAB_DIR / "table_nd_by_size.csv"
    tab_nd_sz.to_csv(p, index=False)
    preview(tab_nd_sz, f"table_nd_by_size.csv  ->  {p}")

    # ------------------- Table 3: ND by dim ------------------- #
    tab_nd_d = build_table(
        dfnd, bucket_col="d",
        buckets=DIM_BUCKETS_ND,
        models=MODELS_ND, label_name="d_bucket",
        time_col="total_time_s",
    )
    p = TAB_DIR / "table_nd_by_dim.csv"
    tab_nd_d.to_csv(p, index=False)
    preview(tab_nd_d, f"table_nd_by_dim.csv  ->  {p}")

    # ------------------- Table 4: TSPLIB by size ------------------- #
    tab_tl = build_table(
        dftl, bucket_col="n",
        buckets=SIZE_BUCKETS_TSPLIB,
        models=MODELS_TSPLIB, label_name="n_bucket",
        time_col="total_time_s",
    )
    p = TAB_DIR / "table_tsplib_by_size.csv"
    tab_tl.to_csv(p, index=False)
    preview(tab_tl, f"table_tsplib_by_size.csv  ->  {p}")

    # ------------------- Table 5: TSPLIB non-Euclidean (§6) ------------ #
    print(f"Building TSPLIB non-Euclidean summary "
          f"(GART 2.0 only, true_cost/MST > {METRIC_RATIO_THRESHOLD} dropped) ...")
    df_ne = load_tsplib_nonEuc()
    tab_ne = build_tsplib_nonEuc_table(df_ne)
    p = TAB_DIR / "table_tsplib_nonEuc.csv"
    tab_ne.to_csv(p, index=False)
    preview(tab_ne, f"table_tsplib_nonEuc.csv  ->  {p}")

    # ------------------- Table 6: time breakdown ------------------- #
    br_2d = time_breakdown(df2d, "2D", MODELS_2D)
    br_nd = time_breakdown(dfnd, "ND", MODELS_ND)
    br_tl = time_breakdown(dftl, "TSPLIB", MODELS_TSPLIB)
    tab_br = pd.concat([br_2d, br_nd, br_tl], ignore_index=True)
    p = TAB_DIR / "table_time_breakdown.csv"
    tab_br.to_csv(p, index=False)
    preview(tab_br, f"table_time_breakdown.csv  ->  {p}")

    # ------------------- Box plots ------------------- #
    print("\nGenerating box plots ...")
    boxplot(df2d, MODELS_2D,
            title="Percent error by estimator - 2D benchmark (n in [5,1000])",
            out_path=OUT_DIR / "boxplot_2d_errors.png", ylim=(-60.0, 60.0))
    boxplot(dfnd, MODELS_ND,
            title="Percent error by estimator - ND benchmark (d in [2,100])",
            out_path=OUT_DIR / "boxplot_nd_errors.png", ylim=(-60.0, 60.0))
    boxplot(dftl, MODELS_TSPLIB,
            title=f"Percent error by estimator - TSPLIB EUC_2D "
                  f"(common subset, {common_inst} instances)",
            out_path=OUT_DIR / "boxplot_tsplib_errors.png",
            ylim=(-60.0, 60.0))

    # ------------------- ALL-row summaries ------------------- #
    print("\n" + "#" * 78)
    print("ALL-row summaries (full tables)")
    print("#" * 78)
    for tab, name in [
        (tab_2d,    "2D by size"),
        (tab_nd_sz, "ND by size"),
        (tab_nd_d,  "ND by dim"),
        (tab_tl,    "TSPLIB by size"),
        (tab_ne,    "TSPLIB non-Euclidean"),
        (tab_br,    "time breakdown (all rows)"),
    ]:
        print(f"\n-- {name} --")
        with pd.option_context("display.max_columns", None,
                               "display.width", 200):
            print(tab.to_string(index=False))

    free(df2d, dfnd, dftl)
    print("\nDone.")


if __name__ == "__main__":
    main()
