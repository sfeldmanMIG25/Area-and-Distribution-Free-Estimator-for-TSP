"""
Run Kwon (1995) and Daganzo (1984) on every EUC_2D TSPLIB95 instance and
append the rows to all_models_tsplib_supplemental.csv so build_tsplib_tables.py
picks them up at merge time. Existing Kwon/Daganzo rows (if any) are
overwritten (keep last).
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from tsplib_parser import parse_tsplib_file
import tsp_utils_2 as academic

INSTANCES_DIR = THIS_DIR / "instances"
RESULTS_DIR   = THIS_DIR / "results"
SUP_CSV       = RESULTS_DIR / "all_models_tsplib_supplemental.csv"
OPTIMA_CSV    = THIS_DIR / "ground_truth" / "optima.csv"


def load_optima():
    df = pd.read_csv(OPTIMA_CSV)
    return dict(zip(df["instance"].astype(str), df["optimum"].astype(float)))


def make_row(name, n, model, pred, true_cost, tsec, mst_len):
    gap = (pred - true_cost) / true_cost * 100.0 if true_cost else np.nan
    return {
        "instance": name,
        "n": n,
        "edge_weight_type": "EUC_2D",
        "model": model,
        "pred_cost": float(pred),
        "true_cost": float(true_cost),
        "gap_pct": float(gap),
        "abs_gap_pct": float(abs(gap)),
        "total_time_s": float(tsec),
        "feature_time_s": float(tsec),
        "inference_time_s": 0.0,
        "mst_length": float(mst_len) if mst_len is not None else np.nan,
        "alpha": np.nan,
        "concorde_time_s": np.nan,
        "speedup_vs_concorde": np.nan,
        "mode": "native",
        "feature_dim": 2,
    }


def main():
    optima = load_optima()
    tsp_files = sorted(INSTANCES_DIR.glob("*.tsp"))

    new_rows = []
    for path in tsp_files:
        name = path.stem
        true_cost = optima.get(name)
        if true_cost is None:
            continue
        try:
            info = parse_tsplib_file(str(path))
        except Exception as e:
            print(f"  SKIP {name}: parse error ({e})")
            continue
        if info["edge_weight_type"] != "EUC_2D":
            continue
        if info["raw_coords"] is None:
            continue
        coords = info["raw_coords"].astype(np.float32)
        n = info["n"]

        try:
            pred, tsec = academic.estimate_tsp_kwon(coords)
            new_rows.append(make_row(name, n, "Kwon", pred, true_cost, tsec, None))
        except Exception as e:
            print(f"  Kwon FAILED on {name}: {e}")

        try:
            pred, tsec = academic.estimate_tsp_daganzo(coords)
            new_rows.append(make_row(name, n, "Daganzo", pred, true_cost, tsec, None))
        except Exception as e:
            print(f"  Daganzo FAILED on {name}: {e}")

    print(f"\nComputed {len(new_rows)} Kwon/Daganzo rows across {len(new_rows)//2} EUC_2D instances.")

    if SUP_CSV.exists():
        existing = pd.read_csv(SUP_CSV)
        # drop any prior Kwon/Daganzo rows; keep everything else
        existing = existing[~existing["model"].isin(["Kwon", "Daganzo"])]
        merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        merged = pd.DataFrame(new_rows)

    merged.to_csv(SUP_CSV, index=False)
    print(f"Wrote {len(merged)} total rows -> {SUP_CSV}")


if __name__ == "__main__":
    main()
