"""
Extended N-Dimensional Benchmark: GART 3.0 vs MST Ratio
Samples instances across d=2 to d=100, runs LGBM_V3 and MST_Ratio.
"""
import os, sys, time
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR / "lgbm_model_v3"))

from tsp_utils import parse_tsp_instance, parse_tsp_solution
import tsp_utils_2 as academic
from lgbm_model_v3.lgbm_estimator_v3 import TSP_V3_LGBM_Estimator

INSTANCES_DIR = SCRIPT_DIR / "instances"
SOLUTIONS_DIR = SCRIPT_DIR / "solutions"
OUTPUT_FILE = SCRIPT_DIR / "benchmark_extended_dims.csv"
SAMPLES_PER_DIM = 300  # enough for statistical power, fast enough to run

def find_all_solved_instances():
    sol_stems = {f.stem.replace('.sol', '') for f in SOLUTIONS_DIR.glob("*.sol.json")}
    inst_stems = {f.stem for f in INSTANCES_DIR.glob("*.json") if not f.stem.endswith('.sol')}
    return sorted(sol_stems & inst_stems)

def extract_info(inst_name):
    try:
        inst_data = parse_tsp_instance(INSTANCES_DIR / f"{inst_name}.json")
        sol_data = parse_tsp_solution(SOLUTIONS_DIR / f"{inst_name}.sol.json")
        tc = sol_data['optimal_cost']
        if tc <= 0: return None
        return {
            "instance": inst_name,
            "file_path": str(INSTANCES_DIR / f"{inst_name}.json"),
            "n_customers": inst_data['n_customers'],
            "dimension": inst_data['dimension'],
            "grid_size": inst_data.get('grid_size', 1000),
            "distribution": inst_data.get('distribution_type', 'unknown'),
            "true_cost": tc,
        }
    except:
        return None

def run_models(row, gart):
    try:
        inst_data = parse_tsp_instance(Path(row['file_path']))
    except:
        return []
    coords = inst_data.coordinates
    results = []

    # GART
    try:
        res = gart.estimate(coords, row['dimension'], row['grid_size'])
        pred = res['estimate']
        gap = ((pred - row['true_cost']) / row['true_cost']) * 100
        results.append({'model': 'LGBM_V3', 'instance': row['instance'],
            'n': row['n_customers'], 'dimension': row['dimension'],
            'distribution': row['distribution'],
            'pred_cost': pred, 'true_cost': row['true_cost'],
            'gap_pct': gap, 'abs_gap_pct': abs(gap),
            'time_s': res.get('feature_time', 0) + res.get('inference_time', 0)})
    except Exception as e:
        pass

    # MST Ratio
    try:
        pred_mst, t = academic.estimate_tsp_mst_ratio(coords)
        gap = ((pred_mst - row['true_cost']) / row['true_cost']) * 100
        results.append({'model': 'MST_Ratio', 'instance': row['instance'],
            'n': row['n_customers'], 'dimension': row['dimension'],
            'distribution': row['distribution'],
            'pred_cost': pred_mst, 'true_cost': row['true_cost'],
            'gap_pct': gap, 'abs_gap_pct': abs(gap), 'time_s': t})
    except:
        pass
    return results

def main():
    print("=== Extended Benchmark: GART vs MST Ratio (sampled) ===")
    all_inst = find_all_solved_instances()
    print(f"Found {len(all_inst)} solved instances. Extracting metadata...")

    base = []
    for name in tqdm(all_inst):
        info = extract_info(name)
        if info: base.append(info)
    base_df = pd.DataFrame(base)
    print(f"Valid: {len(base_df)} across dims {sorted(base_df['dimension'].unique())}")

    # Sample per dimension
    sampled_parts = []
    for d, grp in base_df.groupby('dimension'):
        sampled_parts.append(grp.sample(n=min(SAMPLES_PER_DIM, len(grp)), random_state=42))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    print(f"Sampled {len(sampled)} instances ({SAMPLES_PER_DIM}/dim)")
    print(sampled.groupby('dimension')['instance'].count())

    # Load GART
    print("\nLoading GART 3.0...")
    gart = TSP_V3_LGBM_Estimator(str(SCRIPT_DIR / 'lgbm_model_v3'))

    # Run
    print("Running benchmarks...")
    results = []
    for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
        results.extend(run_models(row, gart))

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} results to {OUTPUT_FILE}")

    # Summary
    print("\n=== SUMMARY ===")
    for model in ['LGBM_V3', 'MST_Ratio']:
        mdf = df[df['model'] == model]
        print(f"\n{model}:")
        for d in sorted(mdf['dimension'].unique()):
            dd = mdf[mdf['dimension'] == d]
            print(f"  d={d:3d}: n={len(dd):4d}, MAPE={dd['abs_gap_pct'].mean():6.2f}%, median={dd['abs_gap_pct'].median():6.2f}%")
        print(f"  ALL:   n={len(mdf):4d}, MAPE={mdf['abs_gap_pct'].mean():6.2f}%")

if __name__ == "__main__":
    main()
