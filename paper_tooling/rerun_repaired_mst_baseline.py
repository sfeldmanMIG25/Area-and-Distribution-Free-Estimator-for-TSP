"""Prediction-only rerun of the repaired MST-Ratio baseline.

This script never invokes a solver and never overwrites canonical benchmark
files.  It carries forward every non-MST row unchanged and recomputes only
``MST_Ratio`` predictions from the MST lengths already recorded in the
benchmark ground-truth checkpoints (or in TSPLIB's result rows).
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_registry import EMBEDDING_DONOR  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

TWO_D_INPUT = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
TWO_D_GROUND_TRUTH = (
    ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv"
)
TWO_D_OUTPUT = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_repaired.csv"

ND_INPUT = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
ND_GROUND_TRUTH = (
    ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
)
ND_OUTPUT = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_repaired.csv"

TSPLIB_INPUT = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
TSPLIB_OUTPUT = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib_repaired.csv"


def mst_ratio(dimension: pd.Series) -> pd.Series:
    """Return the dimensional schedule used by estimate_tsp_mst_ratio."""
    dimension = pd.to_numeric(dimension, errors="raise").astype(float)
    return np.where(
        dimension == 2,
        0.7124 / 0.6331,
        np.where(dimension == 3, 1.05, 1.0 + (0.075 * (2.0 / dimension))),
    )


def validate(
    frame: pd.DataFrame, label: str, expected_models: set[str], repaired_rows: pd.Series
) -> None:
    duplicates = frame.duplicated(["model", "instance"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["model", "instance"]].head(10).to_dict("records")
        raise ValueError(f"{label}: duplicate (model, instance) rows: {examples}")
    observed = set(frame["model"].dropna().unique())
    if observed != expected_models:
        raise ValueError(f"{label}: model coverage changed: {observed ^ expected_models}")
    numeric = [column for column in ("pred_cost", "gap_pct", "abs_gap_pct", "alpha") if column in frame]
    for column in numeric:
        values = pd.to_numeric(frame.loc[repaired_rows, column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: non-finite repaired MST_Ratio {column} values")


def validate_non_mst_rows_unchanged(source: pd.DataFrame, output: pd.DataFrame, label: str) -> None:
    mask = ~source["model"].eq("MST_Ratio")
    pd.testing.assert_frame_equal(
        source.loc[mask].reset_index(drop=True),
        output.loc[mask].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def write_preserving_non_mst_tokens(
    source_path: Path,
    output_path: Path,
    source: pd.DataFrame,
    repaired: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Write repaired MST rows while retaining every other source CSV token."""
    raw = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    keys = ["model", "instance"]
    if not raw[keys].equals(repaired[keys].astype(str)):
        raise ValueError(f"{label}: repaired row order differs from source")
    mst_rows = raw["model"].eq("MST_Ratio")
    serialized = repaired.loc[mst_rows].to_csv(index=False)
    replacement = pd.read_csv(StringIO(serialized), dtype=str, keep_default_na=False)
    raw.loc[mst_rows, :] = replacement.to_numpy()
    raw.to_csv(output_path, index=False)
    persisted = pd.read_csv(output_path, low_memory=False)
    validate_non_mst_rows_unchanged(source, persisted, label)
    return persisted


def replace_predictions(
    results: pd.DataFrame,
    mst_length: pd.Series,
    dimension: pd.Series,
    label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    output = results.copy()
    mask = output["model"].eq("MST_Ratio")
    if not mask.any():
        raise ValueError(f"{label}: no MST_Ratio rows found")
    mst = pd.to_numeric(mst_length, errors="coerce")
    dims = pd.to_numeric(dimension, errors="coerce")
    true_cost = pd.to_numeric(output["true_cost"], errors="coerce")
    eligible = mask & mst.notna() & dims.notna() & (mst > 0) & (dims > 0) & true_cost.notna() & (true_cost > 0)
    if not eligible.any():
        raise ValueError(f"{label}: no eligible MST_Ratio rows found")
    output.loc[eligible, "pred_cost"] = (mst[eligible] * mst_ratio(dims[eligible])).to_numpy()
    output.loc[eligible, "gap_pct"] = (
        (output.loc[eligible, "pred_cost"] - true_cost[eligible]) / true_cost[eligible] * 100.0
    )
    output.loc[eligible, "abs_gap_pct"] = output.loc[eligible, "gap_pct"].abs()
    if "alpha" in output:
        output.loc[eligible, "alpha"] = output.loc[eligible, "pred_cost"] / mst[eligible]
    print(f"{label}: repaired {eligible.sum()} of {mask.sum()} MST_Ratio rows")
    return output, eligible


def rerun_with_checkpoint(
    source: Path,
    checkpoint: Path,
    output: Path,
    label: str,
) -> pd.DataFrame:
    results = pd.read_csv(source, low_memory=False)
    expected_models = set(results["model"].dropna().unique())
    ground_truth = pd.read_csv(checkpoint, low_memory=False)
    required = {"instance", "mst_length", "dimension"}
    if required - set(ground_truth):
        raise ValueError(f"{label}: checkpoint is missing {required - set(ground_truth)}")
    joined = results.merge(
        ground_truth[["instance", "mst_length", "dimension"]],
        on="instance",
        how="left",
        validate="many_to_one",
    )
    repaired, eligible = replace_predictions(joined, joined["mst_length"], joined["dimension"], label)
    repaired = repaired[results.columns]
    validate(repaired, label, expected_models, eligible)
    validate_non_mst_rows_unchanged(results, repaired, label)
    return write_preserving_non_mst_tokens(source, output, results, repaired, label)


def rerun_tsplib() -> pd.DataFrame:
    results = pd.read_csv(TSPLIB_INPUT, low_memory=False)
    expected_models = set(results["model"].dropna().unique())
    for column in ("mst_length", "feature_dim"):
        if column not in results:
            raise ValueError(f"TSPLIB: missing {column}")
    # Native-Euclidean MST rows already carry these inputs.  Hybrid rows were
    # historically emitted as empty ``academic_non_2d`` placeholders, even
    # though the donor row for the same instance stores the original-distance
    # MST and the selected MDS dimension.  Copy only those persisted inputs so
    # the paper's hybrid reference is reproducible as rho(k) * MST(D).
    #
    # The donor is ``EMBEDDING_DONOR`` (see model_registry), not the production
    # model: these are instance-level facts that must be present on every row,
    # and the production model records NaN for instances it declines.
    prepared = results.copy()
    donor = (
        results.loc[results["model"].eq(EMBEDDING_DONOR),
                    ["instance", "mst_length", "feature_dim", "true_cost"]]
        .set_index("instance")
    )
    mst_rows = prepared["model"].eq("MST_Ratio")
    hybrid_rows = mst_rows & prepared["mst_length"].isna()
    for column in ("mst_length", "feature_dim", "true_cost"):
        prepared.loc[hybrid_rows, column] = prepared.loc[hybrid_rows, "instance"].map(
            donor[column]
        )
    prepared.loc[hybrid_rows, "mode"] = "hybrid"
    prepared.loc[hybrid_rows, "status"] = "repaired_hybrid_fixed_scaling"

    repaired, eligible = replace_predictions(
        prepared, prepared["mst_length"], prepared["feature_dim"], "TSPLIB"
    )
    validate(repaired, "TSPLIB", expected_models, eligible)
    validate_non_mst_rows_unchanged(results, repaired, "TSPLIB")
    return write_preserving_non_mst_tokens(
        TSPLIB_INPUT, TSPLIB_OUTPUT, results, repaired, "TSPLIB"
    )


def report(label: str, frame: pd.DataFrame, instance_size: pd.Series) -> None:
    mst = frame.loc[frame["model"].eq("MST_Ratio")].copy()
    mst["_size"] = mst["instance"].map(instance_size)
    evaluated = mst.loc[np.isfinite(pd.to_numeric(mst["abs_gap_pct"], errors="coerce"))].copy()
    print(
        f"{label}: rows={len(frame)}, MST_Ratio rows={len(mst)}, "
        f"evaluated={len(evaluated)}, instances={evaluated['instance'].nunique()}"
    )
    for bucket, sub in [("total", evaluated), *bucket_rows(evaluated)]:
        mean = sub["abs_gap_pct"].mean()
        median = sub["abs_gap_pct"].median()
        sdpe = sub["gap_pct"].std(ddof=1)
        print(f"  {bucket}: N={len(sub)}, MAPE={mean:.6f}, median={median:.6f}, SDPE={sdpe:.6f}")


def bucket_rows(frame: pd.DataFrame):
    if "n_customers" in frame:
        values = frame["n_customers"]
    elif "n" in frame:
        values = frame["n"]
    else:
        values = frame["_size"]
    for lo, hi in ((5, 10), (11, 50), (51, 100), (101, 500), (501, 1000), (1001, np.inf)):
        sub = frame.loc[(values >= lo) & (values <= hi)]
        if not sub.empty:
            yield f"n=[{lo},{'inf' if np.isinf(hi) else hi}]", sub


def main() -> None:
    two_d = rerun_with_checkpoint(TWO_D_INPUT, TWO_D_GROUND_TRUTH, TWO_D_OUTPUT, "2D")
    nd = rerun_with_checkpoint(ND_INPUT, ND_GROUND_TRUTH, ND_OUTPUT, "ND")
    tsplib = rerun_tsplib()
    two_d_size = pd.read_csv(TWO_D_GROUND_TRUTH).set_index("instance")["n_customers"]
    nd_size = pd.read_csv(ND_GROUND_TRUTH).set_index("instance")["n_customers"]
    report("2D", two_d, two_d_size)
    report("ND", nd, nd_size)
    report("TSPLIB", tsplib, pd.Series(dtype=float))
    print(f"Wrote {TWO_D_OUTPUT}")
    print(f"Wrote {ND_OUTPUT}")
    print(f"Wrote {TSPLIB_OUTPUT}")


if __name__ == "__main__":
    main()
