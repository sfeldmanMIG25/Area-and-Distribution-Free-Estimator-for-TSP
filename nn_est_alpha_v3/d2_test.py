import os
import sys
import pandas as pd
import numpy as np
import json
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import joblib
import torch

# --- SILENCE ALL WARNINGS ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PYTHONWARNINGS'] = 'ignore'
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# --- CUSTOM SCALER DEFINITION (Required for successful loading) ---
class StableV3Scaler:
    def __init__(self):
        self.qt = QuantileTransformer(output_distribution='normal', random_state=42)
        self.ss = StandardScaler()

    def transform(self, X):
        X_qt = self.qt.transform(X)
        X_ss = self.ss.transform(X_qt)
        return np.clip(X_ss, -10.0, 10.0)

# Import the estimator class
from estimator_v3 import TSP_V3_Neural_Estimator

# --- CONFIGURATION (Strict Absolute Paths) ---
INSTANCES_DIR = r"C:\Area-and-Distribution-Free-Estimators-for-TSP\Generalized_TSP_Analysis\instances"
SOLUTIONS_DIR = r"C:\Area-and-Distribution-Free-Estimators-for-TSP\Generalized_TSP_Analysis\solutions"

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(MODEL_DIR, 'nn_benchmark_results_2d.csv')

def run_nn_benchmark():
    print("--- Running Neural Network V3 Benchmark (Strict Path Processing) ---")
    
    # 1. Validation
    if not os.path.exists(INSTANCES_DIR) or not os.path.exists(SOLUTIONS_DIR):
        sys.exit(f"CRITICAL: Directories not found.\nInstances: {INSTANCES_DIR}\nSolutions: {SOLUTIONS_DIR}")

    # 2. Load Estimator
    estimator = TSP_V3_Neural_Estimator(MODEL_DIR)

    # 3. Match Files strictly by folder contents (No CSV filtering)
    all_inst = set(f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json'))
    all_sol = set(f for f in os.listdir(SOLUTIONS_DIR) if f.endswith('.sol.json'))
    
    files_to_process = []
    for inst_file in all_inst:
        base_name = inst_file.replace('.json', '')
        sol_file = f"{base_name}.sol.json"
        if sol_file in all_sol:
            files_to_process.append(inst_file)
            
    files_to_process.sort()
    
    if not files_to_process:
        sys.exit("CRITICAL: No matching instance/solution pairs found in the specified paths.")
        
    print(f"Found {len(files_to_process)} benchmark pairs.")

    # 4. Processing Loop
    results = []
    
    for filename in tqdm(files_to_process, desc="Benchmarking NN"):
        inst_path = os.path.join(INSTANCES_DIR, filename)
        sol_path = os.path.join(SOLUTIONS_DIR, filename.replace('.json', '.sol.json'))
        
        with open(inst_path, 'r') as f:
            inst_data = json.load(f)
        with open(sol_path, 'r') as f:
            sol_data = json.load(f)

        optimal_cost = sol_data.get('optimal_cost')
        if optimal_cost is None or optimal_cost <= 0:
            continue

        # Solver timing for speedup reporting
        opt_time = 1e-9
        if sol_data.get('optimal_solver') == 'concorde':
            opt_time = sol_data.get('concorde_time_s', 1e-9)
        elif sol_data.get('optimal_solver') == 'lkh':
            opt_time = sol_data.get('lkh_time_s', 1e-9)

        # Run Neural Estimation
        est_result = estimator.estimate(
            inst_data['coordinates'], 
            inst_data['dimension'], 
            inst_data.get('grid_size', 1000)
        )
        
        pred_cost = est_result['estimate']
        sdpe = (pred_cost - optimal_cost) / optimal_cost
        
        results.append({
            'instance_name': inst_data['instance_name'],
            'n_customers': inst_data['n_customers'],
            'optimal_cost': optimal_cost,
            'predicted_cost': pred_cost,
            'sdpe_pct': sdpe * 100,
            'abs_pct_error': abs(sdpe) * 100,
            'time_feat': est_result['feature_time'],
            'time_pred': est_result['inference_time'],
            'time_total': est_result['feature_time'] + est_result['inference_time'],
            'optimal_time': opt_time
        })

    # 5. Global Metrics Analysis
    if not results:
        sys.exit("CRITICAL: No valid results were generated.")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    df['speedup_factor'] = df['optimal_time'] / df['time_total']
    sdpe_mean = df['sdpe_pct'].mean()
    sdpe_std = df['sdpe_pct'].std()
    mape = df['abs_pct_error'].mean()
    rmse = np.sqrt(mean_squared_error(df['optimal_cost'], df['predicted_cost']))
    r2 = r2_score(df['optimal_cost'], df['predicted_cost'])
    
    print("\n" + "="*45)
    print("      BENCHMARK REPORT (NEURAL V3)      ")
    print("="*45)
    print(f"Instances Processed: {len(df)}")
    print(f"R^2 Score:           {r2:.5f}")
    print(f"RMSE:                {rmse:.4f}")
    print("-" * 45)
    print(f"SDPE (Bias) %:       {sdpe_mean:.4f}%  (± {sdpe_std:.4f})")
    print(f"MAPE (Accuracy) %:   {mape:.4f}%")
    print("-" * 45)
    print(f"Avg Prediction Time: {df['time_total'].mean():.6f}s")
    print(f"Avg Optimal Time:    {df['optimal_time'].mean():.6f}s")
    print(f"Avg Speedup Factor:  {df['speedup_factor'].mean():.2f}x")
    print("="*45)

    # 6. Generate Plots
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sdpe_pct'], bins=30, kde=True)
    plt.title('Neural V3: SDPE Distribution (Bias)')
    plt.axvline(0, color='r', linestyle='--')
    plt.savefig(os.path.join(MODEL_DIR, 'nn_v3_sdpe_dist.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['optimal_cost'], df['predicted_cost'], alpha=0.6)
    plt.plot([df['optimal_cost'].min(), df['optimal_cost'].max()], 
             [df['optimal_cost'].min(), df['optimal_cost'].max()], 'r--')
    plt.title(f'Neural V3: Predicted vs Optimal Cost (R2={r2:.4f})')
    plt.savefig(os.path.join(MODEL_DIR, 'nn_v3_pred_vs_opt.png'))
    plt.close()
    
    print(f"\n✅ Benchmark complete. Results saved in {MODEL_DIR}")

if __name__ == "__main__":
    run_nn_benchmark()