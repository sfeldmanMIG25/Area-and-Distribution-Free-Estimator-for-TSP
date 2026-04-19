"""Build ``tsp_features_v4.csv`` — 41-candidate feature CSV aligned to the
baseline split carried in ``tsp_features.csv`` / ``tsp_features_v3.csv``.

For each instance listed in the V3 CSV:
  * load coords (via load_instance_data — bin-then-json, strict),
  * compute all 41 V4 features (``feature_engineering.compute_features``),
  * preserve the existing train/val/test split assignment.

Writes ``tsp_features_v4.csv`` in the repo root.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from feature_creator_v3 import load_instance_data   # strict loader
from feature_engineering import compute_features

BASELINE_CSV = REPO_ROOT / "tsp_features_v3.csv"
OUTPUT_CSV = REPO_ROOT / "tsp_features_v4.csv"
INSTANCES_DIR = REPO_ROOT / "instances"
SOLUTIONS_DIR = REPO_ROOT / "solutions"


def _process_one(row: dict) -> dict | None:
    name = row["instance_name"]
    split = row["split"]
    optimal_cost = row["optimal_cost"]
    sol_path = SOLUTIONS_DIR / f"{name}.sol.json"
    if not sol_path.exists():
        return None
    inst = load_instance_data(f"{name}.json")
    if inst is None:
        return None
    coords = np.unique(inst["coordinates"], axis=0)
    if coords.shape[0] < 3:
        return None
    feats = compute_features(coords, int(inst["dimension"]), int(inst.get("grid_size", 0)))
    feats["instance_name"] = name
    feats["split"] = split
    feats["optimal_cost"] = float(optimal_cost)
    return feats


def main() -> None:
    df = pd.read_csv(BASELINE_CSV, usecols=["instance_name", "split", "optimal_cost"])
    rows = df.to_dict(orient="records")
    print(f"[v4-csv] computing 41-feature CSV for {len(rows)} instances")

    num_workers = max(1, os.cpu_count() or 1)
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for feat in tqdm(ex.map(_process_one, rows, chunksize=32), total=len(rows), desc="v4 features"):
            if feat is not None:
                results.append(feat)

    print(f"[v4-csv] computed: {len(results)} / {len(rows)}")
    out = pd.DataFrame(results)
    # Keep identifier columns first for readability.
    front = ["instance_name", "split", "optimal_cost"]
    out = out[front + [c for c in out.columns if c not in front]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[v4-csv] wrote {OUTPUT_CSV}  ({len(out)} rows, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
