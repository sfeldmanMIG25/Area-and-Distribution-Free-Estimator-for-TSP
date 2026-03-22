import os
import sys
import pandas as pd
import numpy as np
import json
import time
import warnings
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
from estimator_linear_v3 import TSP_V3_Linear_Estimator

# --- CONFIGURATION (Absolute Paths) ---
INSTANCES_DIR = r"C:\Area-and-Distribution-Free-Estimators-for-TSP\Generalized_TSP_Analysis\instances"
SOLUTIONS_DIR = r"C:\Area-and-Distribution-Free-Estimators-for-TSP\Generalized_TSP_Analysis\solutions"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def run_benchmark():
    print("--- Running Linear Baseline V3 Benchmark ---")
    if not os.path.exists(INSTANCES_DIR) or not os.path.exists(SOLUTIONS_DIR):
        sys.exit("CRITICAL: Paths not found.")

    estimator = TSP_V3_Linear_Estimator(MODEL_DIR)
    
    # Gather pairs
    all_inst = [f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json')]
    results = []
    
    print(f"Benchmarking {len(all_inst)} instances...")
    for filename in tqdm(all_inst, desc="Linear V3"):
        base_name = filename.replace('.json', '')
        sol_path = os.path.join(SOLUTIONS_DIR, f"{base_name}.sol.json")
        if not os.path.exists(sol_path): continue
            
        with open(os.path.join(INSTANCES_DIR, filename), 'r') as f: inst_data = json.load(f)
        with open(sol_path, 'r') as f: sol_data = json.load(f)

        optimal_cost = sol_data.get('optimal_cost', 0)
        if optimal_cost <= 0: continue

        res = estimator.estimate(inst_data['coordinates'], inst_data['dimension'], inst_data.get('grid_size', 1000))
        pe = (res['estimate'] - optimal_cost) / optimal_cost
        
        results.append({
            'opt': optimal_cost, 'pred': res['estimate'], 'pe_pct': pe * 100,
            'time_total': res['feature_time'] + res['inference_time'],
            'opt_time': sol_data.get('concorde_time_s', sol_data.get('lkh_time_s', 1e-9))
        })

    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)
    mse = mean_squared_error(df['opt'], df['pred'])
    
    print("\n" + "="*45)
    print("      BENCHMARK REPORT (LINEAR V3)      ")
    print("="*45)
    print(f"Total Instances      : {len(df)}")
    print(f"R^2 Score:           {r2_score(df['opt'], df['pred']):.5f}")
    print(f"MAPE (Accuracy) %:   {df['pe_pct'].abs().mean():.4f}%")
    print(f"SD of Error %:       {df['pe_pct'].std():.4f}%")
    print("-" * 45)
    print(f"Avg Prediction Time: {df['time_total'].mean():.6f}s")
    print(f"Avg Speedup Factor:  {(df['opt_time'] / df['time_total']).mean():.2f}x")
    print("="*45)

if __name__ == "__main__":
    run_benchmark()