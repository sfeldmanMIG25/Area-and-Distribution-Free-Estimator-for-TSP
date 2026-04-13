"""
Data loader module for TSP estimation analysis.
Provides documented schemas and loader functions for all benchmark datasets.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Dataset paths ──────────────────────────────────────────────────────────
PATHS = {
    "2d_benchmark": ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv",
    "nd_benchmark": ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv",
    "extended_dims": ROOT / "benchmark_extended_dims.csv",
    "tsplib_all": ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv",
    "tsplib_hybrid": ROOT / "tsplib_benchmark" / "results" / "tsplib_results_hybrid_delaunay.csv",
}

# ── Column schemas (for reference — not enforced at load time) ─────────────
SCHEMAS = {
    "2d_benchmark": {
        # 30,961 rows · 12 models · 2D instances only
        "model":              "BHH | Cavdar | Chien | Composite | GART | Hilbert | Interp_V3 | LGBM_V3 | Linear_V3 | MST_Ratio | Neural_V3 | Vinel",
        "instance":           "Instance id  (e.g. TSP-boundary-n10-g10000-5)",
        "pred_cost":          "Predicted tour length",
        "true_cost":          "Optimal tour length (Concorde / LKH)",
        "prediction_time_s":  "Total prediction time (s)",
        "feature_time_s":     "Feature extraction time (s)",
        "inference_time_s":   "Model inference time (s)",
        "optimal_solve_time_s": "Concorde solve time (s)",
        "gap_pct":            "Signed gap: (pred − true) / true × 100",
        "abs_gap_pct":        "Absolute percentage gap",
    },
    "nd_benchmark": {
        # 94,482 rows · 18 models · d ∈ {2,3,4,5} · has Concorde times
        "instance":           "Instance id",
        "n_customers":        "Number of cities",
        "dimension":          "Spatial dimension (2–5)",
        "grid_size":          "Coordinate grid resolution",
        "distribution":       "Distribution type",
        "true_cost":          "Optimal tour length",
        "mst_length":         "MST total weight",
        "optimal_solve_time_s": "Concorde solve time (s)",
        "model":              "Model name (18 models incl. LGBM_V3)",
        "pred_cost":          "Predicted tour length",
        "prediction_time_s":  "Total prediction time (s)",
        "feature_time_s":     "Feature extraction time (s)",
        "inference_time_s":   "Model inference time (s)",
        "pred_alpha":         "Predicted α = pred_cost / mst_length",
        "gap_pct":            "Signed gap",
        "speedup_pct":        "Concorde time / prediction time",
    },
    "extended_dims": {
        # 10,800 rows · LGBM_V3 + MST_Ratio · d ∈ {2..50, 100}
        "model":        "LGBM_V3 | MST_Ratio",
        "instance":     "Instance id",
        "n":            "Number of cities",
        "dimension":    "Spatial dimension (2–100, 18 values)",
        "distribution": "Distribution type (all 'unknown')",
        "pred_cost":    "Predicted tour length",
        "true_cost":    "Optimal tour length",
        "gap_pct":      "Signed gap",
        "abs_gap_pct":  "Absolute percentage gap",
        "time_s":       "Total prediction time (s)",
    },
    "tsplib_all": {
        # ~880 rows · 11 models · 110 instances
        "instance":           "TSPLIB instance name",
        "n":                  "Number of cities",
        "edge_weight_type":   "EUC_2D | CEIL_2D | ATT | GEO | EXPLICIT",
        "model":              "Model name",
        "pred_cost":          "Predicted tour length",
        "true_cost":          "Optimal tour length",
        "gap_pct":            "Signed gap",
        "abs_gap_pct":        "Absolute percentage gap",
        "total_time_s":       "Total prediction time (s)",
        "feature_time_s":     "Feature extraction time (s)",
        "inference_time_s":   "Model inference time (s)",
        "mst_length":         "MST total weight",
        "alpha":              "Predicted α",
        "concorde_time_s":    "Concorde solve time (may be NaN)",
        "speedup_vs_concorde": "Concorde / prediction time",
        "mode":               "native | hybrid",
        "feature_dim":        "Spatial dimension used for features (2 for native; MDS dim for non-Euclidean)",
    },
    "tsplib_hybrid": {
        # 111 rows · LGBM_V3 only · MDS diagnostics
        "instance":             "TSPLIB instance name",
        "n":                    "Number of cities",
        "edge_weight_type":     "Distance type",
        "mode":                 "native | hybrid",
        "feature_dim":          "Dimension used for features (MDS dim for non-Euclidean)",
        "mds_natural_dim":      "Natural MDS dim for 99.9% variance",
        "mds_variance_retained":"Fraction of variance retained",
        "mds_negative_mass":    "Sum of negative eigenvalues (non-metricity indicator)",
        "mds_strain":           "MDS strain",
        "true_cost":            "Optimal tour length",
        "pred_cost":            "Predicted tour length",
        "alpha":                "Predicted α",
        "mst_length":           "MST total weight",
        "gap_pct":              "Signed gap",
        "abs_gap_pct":          "Absolute percentage gap",
        "total_est_time_s":     "Total estimation time (s)",
    },
}

# ── Model display names ───────────────────────────────────────────────────
PAPER_MODELS = {
    "LGBM_V3":    "GART 3.0",
    "MST_Ratio":  "MST Ratio",
    "BHH":        "BHH",
    "Cavdar":     "Cavdar",
    "Chien":      "Chien",
    "Hilbert":    "Hilbert",
    "Vinel":      "Vinel",
    "Fixed_Alpha":"Fixed α",
}

# Models shown in the paper's 2D benchmark table
PAPER_2D_MODELS = ["LGBM_V3", "MST_Ratio", "Cavdar", "BHH", "Chien", "Hilbert"]


# ── Loaders ────────────────────────────────────────────────────────────────

def load_2d():
    """2D benchmark (30,961 rows, 12 models)."""
    return pd.read_csv(PATHS["2d_benchmark"])

def load_nd():
    """ND benchmark (94,482 rows, 18 models, d=2–5, has Concorde times)."""
    return pd.read_csv(PATHS["nd_benchmark"])

def load_extended():
    """Extended-dims benchmark (10,800 rows, LGBM_V3 + MST_Ratio, d=2–100)."""
    return pd.read_csv(PATHS["extended_dims"])

def load_tsplib():
    """TSPLIB all-models results (~880 rows, 11 models)."""
    # Try canonical name first, fall back to timestamped
    p = PATHS["tsplib_all"]
    if not p.exists():
        candidates = sorted(p.parent.glob("all_models_tsplib_*.csv"))
        if candidates:
            p = candidates[-1]  # most recent
    return pd.read_csv(p)

def load_tsplib_hybrid():
    """TSPLIB hybrid results with MDS diagnostics (111 rows, LGBM_V3)."""
    p = PATHS["tsplib_hybrid"]
    if not p.exists():
        candidates = sorted(p.parent.glob("tsplib_results_*hybrid*.csv"))
        if candidates:
            p = candidates[-1]
    return pd.read_csv(p)

def rename_models(df, col="model"):
    """Rename model codes to paper display names."""
    df = df.copy()
    df[col] = df[col].map(PAPER_MODELS).fillna(df[col])
    return df
