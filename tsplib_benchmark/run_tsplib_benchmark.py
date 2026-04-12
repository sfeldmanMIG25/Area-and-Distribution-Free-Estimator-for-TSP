"""
TSPLIB95 benchmark runner for the GART 3.0 (LGBM V3) estimator.

Pipeline
--------
For each TSPLIB instance in ``instances/``:

1. Parse the .tsp file (tsplib_parser.parse_tsplib_file).
2. If the instance is natively Euclidean (EUC_2D / CEIL_2D / ATT), feed the raw
   2D coordinates to the estimator directly. ATT is pseudo-Euclidean but is
   close enough to 2D Euclidean for the feature set; we use its native 2D
   coordinates.
3. Otherwise (GEO, EXPLICIT) run classical MDS on the TSPLIB distance matrix,
   auto-select a dimensionality (cap = ``MAX_MDS_DIM``), and pass the embedded
   coordinates to the estimator.
4. Compare the predicted tour length to the published optimum in
   ``ground_truth/optima.csv``.

The script never overwrites existing results: each run is written to a new
timestamped CSV under ``results/`` and a summary printed to stdout. Old result
files are left untouched.

CLI
---
    python tsplib_benchmark/run_tsplib_benchmark.py [--max-n N] [--include-over-cap]

Options:
    --max-n N           Skip instances with more than N nodes. Default: no cap.
    --include-over-cap  Include instances whose node count exceeds the model's
                        training range (n > 1000). Default: included; use
                        ``--exclude-over-cap`` to run only in-range instances.
    --exclude-over-cap  Skip instances with n > 1000.
    --tag LABEL         Suffix appended to the output filename.
    --max-mds-dim K     Hard cap on MDS dimensionality (default 100).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lgbm_model_v3"))
sys.path.insert(0, str(THIS_DIR))

from tsplib_parser import parse_tsplib_file  # noqa: E402
from classical_mds import classical_mds      # noqa: E402
from lgbm_estimator_v3 import TSP_V3_LGBM_Estimator  # noqa: E402


INSTANCES_DIR = THIS_DIR / "instances"
GROUND_TRUTH_FILE = THIS_DIR / "ground_truth" / "optima.csv"
RESULTS_DIR = THIS_DIR / "results"
TRAINING_MAX_N = 1000     # Upper bound of n during training.
TRAINING_MAX_DIM = 50     # Upper bound of feature dimensionality during training.
DEFAULT_MAX_MDS_DIM = 100


def load_optima() -> dict:
    df = pd.read_csv(GROUND_TRUTH_FILE)
    return dict(zip(df["instance"].astype(str), df["optimum"].astype(float)))


def prepare_instance(info, max_mds_dim):
    """Turn a parsed TSPLIB record into the (coords, d_feature, mds_info) triple
    that the estimator expects.

    For native-Euclidean edge types we return the raw coordinates as-is. For
    GEO/EXPLICIT we run classical MDS on the distance matrix and return the
    embedded points.
    """
    if info["is_native_euclidean"] and info["raw_coords"] is not None:
        coords = info["raw_coords"].astype(np.float32)
        d = coords.shape[1]
        return coords, d, {
            "mode": "native",
            "chosen_dim": d,
            "natural_dim": d,
            "variance_retained": 1.0,
            "negative_eigvalue_mass": 0.0,
            "strain": 0.0,
        }

    # MDS path
    D = info["distance_matrix"]
    X, _eigs, mds_info = classical_mds(
        D, max_dim=max_mds_dim, variance_threshold=0.999
    )
    mds_info = dict(mds_info)
    mds_info["mode"] = "mds"
    return X.astype(np.float32), X.shape[1], mds_info


def run_benchmark(
    exclude_over_cap: bool = False,
    max_n: int | None = None,
    tag: str = "",
    max_mds_dim: int = DEFAULT_MAX_MDS_DIM,
) -> Path:
    """Run GART 3.0 on every TSPLIB .tsp file under ``instances/``.

    Returns the path to the newly-written results CSV.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    optima = load_optima()
    estimator = TSP_V3_LGBM_Estimator(str(REPO_ROOT / "lgbm_model_v3"))

    tsp_files = sorted(INSTANCES_DIR.glob("*.tsp"))
    if not tsp_files:
        raise SystemExit(
            f"No .tsp files in {INSTANCES_DIR}. Run download_tsplib.py first."
        )

    rows = []
    skipped = []
    for path in tqdm(tsp_files, desc="TSPLIB benchmark"):
        name = path.stem
        try:
            t_parse0 = time.perf_counter()
            info = parse_tsplib_file(path)
            t_parse = time.perf_counter() - t_parse0
        except Exception as exc:
            skipped.append((name, f"parse error: {exc}"))
            continue

        n = info["n"]
        if max_n is not None and n > max_n:
            skipped.append((name, f"n={n} exceeds --max-n={max_n}"))
            continue
        if exclude_over_cap and n > TRAINING_MAX_N:
            skipped.append((name, f"n={n} > training cap {TRAINING_MAX_N}"))
            continue

        true_cost = optima.get(name)
        if true_cost is None:
            skipped.append((name, "no ground-truth optimum"))
            continue

        try:
            t_prep0 = time.perf_counter()
            coords, d_feat, mds_info = prepare_instance(info, max_mds_dim)
            t_prep = time.perf_counter() - t_prep0
        except Exception as exc:
            skipped.append((name, f"prep error: {exc}"))
            continue

        if d_feat > TRAINING_MAX_DIM:
            over_dim = True
        else:
            over_dim = False

        try:
            t_est0 = time.perf_counter()
            res = estimator.estimate(coords, d_feat, grid_size=0)
            t_est = time.perf_counter() - t_est0
        except Exception as exc:
            skipped.append((name, f"estimate error: {exc}"))
            continue

        pred = res["estimate"]
        gap = (pred - true_cost) / true_cost * 100.0
        rows.append({
            "instance": name,
            "n": n,
            "edge_weight_type": info["edge_weight_type"],
            "mode": mds_info["mode"],
            "feature_dim": d_feat,
            "mds_natural_dim": mds_info["natural_dim"],
            "mds_variance_retained": mds_info["variance_retained"],
            "mds_negative_mass": mds_info["negative_eigvalue_mass"],
            "mds_strain": mds_info["strain"],
            "in_training_n_range": n <= TRAINING_MAX_N,
            "in_training_dim_range": d_feat <= TRAINING_MAX_DIM,
            "extrapolated": (n > TRAINING_MAX_N) or over_dim,
            "true_cost": true_cost,
            "pred_cost": pred,
            "alpha": res["alpha"],
            "mst_length": res["mst_length"],
            "gap_pct": gap,
            "abs_gap_pct": abs(gap),
            "parse_time_s": t_parse,
            "prep_time_s": t_prep,
            "feature_time_s": res["feature_time"],
            "inference_time_s": res["inference_time"],
            "total_est_time_s": t_est,
        })

    # --- Persist results ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    out_file = RESULTS_DIR / f"tsplib_results_{ts}{suffix}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)

    skip_file = RESULTS_DIR / f"tsplib_skipped_{ts}{suffix}.csv"
    with open(skip_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "reason"])
        for name, reason in skipped:
            w.writerow([name, reason])

    print_summary(df, skipped, out_file)
    return out_file


def print_summary(df: pd.DataFrame, skipped, out_file: Path):
    print()
    print("=" * 80)
    print(f" TSPLIB95 benchmark -- GART 3.0 (LGBM V3)")
    print("=" * 80)
    print(f"Result file    : {out_file}")
    print(f"Total runs     : {len(df)}")
    print(f"Skipped        : {len(skipped)}")
    if df.empty:
        return

    def _block(label: str, subset: pd.DataFrame):
        if subset.empty:
            print(f"{label:30s}: (no instances)")
            return
        mape = subset["abs_gap_pct"].mean()
        med  = subset["abs_gap_pct"].median()
        p90  = subset["abs_gap_pct"].quantile(0.90)
        mx   = subset["abs_gap_pct"].max()
        bias = subset["gap_pct"].mean()
        t = (subset["feature_time_s"] + subset["inference_time_s"]).mean() * 1000
        print(
            f"{label:30s}: n={len(subset):3d}  "
            f"MAPE={mape:6.3f}%  med={med:6.3f}%  p90={p90:6.3f}%  "
            f"max={mx:6.2f}%  bias={bias:+.3f}%  lat={t:6.2f}ms"
        )

    print()
    print("--- Overall ---")
    _block("ALL", df)
    _block("In training range (n<=1000)", df[df["in_training_n_range"]])
    _block("Extrapolated (n>1000)", df[~df["in_training_n_range"]])

    print()
    print("--- By edge-weight type ---")
    for ewt in sorted(df["edge_weight_type"].unique()):
        _block(ewt, df[df["edge_weight_type"] == ewt])

    print()
    print("--- By mode (native vs MDS) ---")
    for mode in sorted(df["mode"].unique()):
        _block(mode, df[df["mode"] == mode])

    print()
    print("Top 10 largest |gap|:")
    top = df.nlargest(10, "abs_gap_pct")[
        ["instance", "n", "edge_weight_type", "mode", "true_cost", "pred_cost", "gap_pct"]
    ]
    print(top.to_string(index=False))
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=None,
                    help="Hard ceiling on n; larger instances are skipped.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--include-over-cap", action="store_true",
                       help="Include instances with n > 1000 (default).")
    group.add_argument("--exclude-over-cap", action="store_true",
                       help="Exclude instances with n > 1000.")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional label appended to result filenames.")
    ap.add_argument("--max-mds-dim", type=int, default=DEFAULT_MAX_MDS_DIM,
                    help=f"Hard cap on MDS dimensionality (default {DEFAULT_MAX_MDS_DIM}).")
    args = ap.parse_args()

    run_benchmark(
        exclude_over_cap=args.exclude_over_cap,
        max_n=args.max_n,
        tag=args.tag,
        max_mds_dim=args.max_mds_dim,
    )


if __name__ == "__main__":
    main()
