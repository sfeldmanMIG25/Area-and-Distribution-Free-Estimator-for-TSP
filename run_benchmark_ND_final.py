import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from glob import glob
import warnings
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import functools
import gc
import joblib

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

# --- Add model subdirectories to sys.path ---
sys.path.append(str(SCRIPT_DIR / "linear_model_v3"))
sys.path.append(str(SCRIPT_DIR / "lgbm_model_v3"))
sys.path.append(str(SCRIPT_DIR / "nn_est_alpha_v3"))
sys.path.append(str(SCRIPT_DIR / "interpretable_model_v3"))
sys.path.append(str(SCRIPT_DIR / "GART_1.0"))

# --- IMPORTS ---
from tsp_utils import parse_tsp_instance, parse_tsp_solution
import tsp_utils_2 as academic

# Import V3 Estimators
from linear_model_v3.estimator_linear_v3 import TSP_V3_Linear_Estimator
from lgbm_model_v3.lgbm_estimator_v3 import TSP_V3_LGBM_Estimator
from nn_est_alpha_v3.estimator_v3 import TSP_V3_Neural_Estimator
from interpretable_model_v3.estimator_interpretable_v3 import TSP_Interpretable_Estimator

ROOT_DIR = SCRIPT_DIR
RESULTS_DIR = ROOT_DIR / "Generalized_TSP_Analysis_ND"
INSTANCES_DIR = ROOT_DIR / "instances"
SOLUTIONS_DIR = ROOT_DIR / "solutions"
METADATA_FILE = ROOT_DIR / "tsp_features_v3.csv"
CHECKPOINT_DIR = RESULTS_DIR / "benchmark_checkpoints"
FINAL_RESULTS_FILE = RESULTS_DIR / "benchmark_results_ND_final.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# =============================================================================
# 0. GART Adapter
# =============================================================================
class GART_Adapter:
    def __init__(self, model_dir):
        p = Path(model_dir) / "alpha_predictor_model.joblib"
        self.model = joblib.load(p)

    def estimate(self, coordinates, dimension, grid_size):
        cost, t_feat, t_inf = academic.estimate_tsp_ml_alpha(coordinates, self.model)
        return {'estimate': cost, 'feature_time': t_feat, 'inference_time': t_inf}

# =============================================================================
# 1. Base Data Generation
# =============================================================================
def get_test_set_instances():
    """Identify N-Dimensional Test Set from V3 Feature CSV."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"FATAL: Metadata file not found at {METADATA_FILE}")
        
    df = pd.read_csv(METADATA_FILE)
    if 'split' not in df.columns or 'instance_name' not in df.columns:
        raise ValueError("Metadata file missing 'split' or 'instance_name'.")
        
    # Filter for test set
    test_df = df[df['split'] == 'test']
    return set(test_df['instance_name'].tolist())

def extract_base_info(inst_name):
    inst_path = INSTANCES_DIR / f"{inst_name}.json"
    sol_path = SOLUTIONS_DIR / f"{inst_name}.sol.json"
    
    if not inst_path.exists() or not sol_path.exists():
        return None
        
    try:
        inst_data = parse_tsp_instance(inst_path)
        sol_data = parse_tsp_solution(sol_path)
        
        # Lazy loading
        coords = inst_data.coordinates
        mst_len, _ = academic.get_mst_length(coords)
        
        opt_solver = sol_data.get('optimal_solver')
        opt_time = 0.0
        if opt_solver == 'concorde': 
            opt_time = sol_data.get('concorde_time_s', 0.0)
        elif opt_solver in ['lkh', 'lkh_only', 'lkh_only_timed']: 
            opt_time = sol_data.get('lkh_time_s', 0.0)
            
        return {
            "instance": inst_name,
            "file_path": str(inst_path),
            "n_customers": inst_data['n_customers'],
            "dimension": inst_data['dimension'],
            "grid_size": inst_data.get('grid_size', 1000),
            "distribution": inst_data.get('distribution_type', 'unknown'),
            "true_cost": sol_data['optimal_cost'],
            "mst_length": mst_len,
            "optimal_solve_time_s": opt_time
        }
    except:
        return None

def generate_base_dataframe(instance_names):
    base_file = CHECKPOINT_DIR / "base_ground_truth_nd.csv"
    if base_file.exists(): 
        return pd.read_csv(base_file)
        
    print("Generating base ground truth for N-Dimensional Test Set...")
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    max_workers = os.cpu_count() or 4
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(extract_base_info, instance_names), total=len(instance_names)):
            if res: results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv(base_file, index=False)
    return df

# =============================================================================
# 2. Model Execution Logic
# =============================================================================
def worker_run_estimator(row_dict, model_name, estimator_obj):
    inst_path = Path(row_dict['file_path'])
    try:
        inst_data = parse_tsp_instance(inst_path)
    except:
        return None
        
    coords = inst_data.coordinates
    d = row_dict['dimension']
    grid_size = row_dict['grid_size']
    
    t_feat = 0.0
    t_inf = 0.0
    pred_cost = 0.0
    
    try:
        if hasattr(estimator_obj, 'estimate'):
            # V3 Models and GART
            res = estimator_obj.estimate(coords, d, grid_size)
            pred_cost = res['estimate']
            t_feat = res.get('feature_time', 0.0)
            t_inf = res.get('inference_time', 0.0)
        else:
            # Academic Estimators
            pred_cost, t_total = estimator_obj(coords)
            t_feat = t_inf = t_total / 2
    except Exception as e:
        print(f"Error in {model_name} for {inst_path.stem}: {e}")
        return None
        
    return {
        'model': model_name,
        'instance': row_dict['instance'],
        'pred_cost': pred_cost,
        'true_cost': row_dict['true_cost'],
        'prediction_time_s': t_feat + t_inf,
        'feature_time_s': t_feat,
        'inference_time_s': t_inf,
        'optimal_solve_time_s': row_dict['optimal_solve_time_s']
    }

def process_model(model_name, factory, base_df):
    print(f"Processing {model_name}...")
    output_csv = CHECKPOINT_DIR / f"results_{model_name.lower()}.csv"
    
    if output_csv.exists():
        print(f"    [SKIPPED] Existing results found.")
        return

    try:
        estimator_instance = factory() if callable(factory) else factory
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    max_workers = os.cpu_count() or 4
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        partial_worker = functools.partial(worker_run_estimator, model_name=model_name, estimator_obj=estimator_instance)
        # Pass rows as dicts
        futures = [executor.submit(partial_worker, row.to_dict()) for _, row in base_df.iterrows()]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = future.result()
            if res: results.append(res)
            
    for row in results:
        if row['true_cost'] > 0:
            row['gap_pct'] = ((row['pred_cost'] - row['true_cost']) / row['true_cost']) * 100
            row['abs_gap_pct'] = abs(row['gap_pct'])
            row['speedup_pct'] = ((row['optimal_solve_time_s'] - (row['feature_time_s'] + row['inference_time_s'])) / row['optimal_solve_time_s']) * 100
        else:
            row['gap_pct'] = -100.0
            row['abs_gap_pct'] = 100.0
            row['speedup_pct'] = 0.0
            
    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        avg_gap = np.mean([r['abs_gap_pct'] for r in results])
        print(f"    [SAVED] {model_name} | MAPE: {avg_gap:.2f}%")
        
    del estimator_instance
    gc.collect()

# =============================================================================
# 3. Reporting & Plots
# =============================================================================
def calculate_metrics_and_print(df):
    print("\n" + "="*90)
    print("                      FINAL BENCHMARK SUMMARY (N-D)                      ")
    print("="*90)
    
    df['error'] = df['pred_cost'] - df['true_cost']
    df['sq_error'] = df['error'] ** 2
    
    def nonzero_mean(series):
        nz = series[series > 1e-9]
        return nz.mean() if not nz.empty else 0.0
        
    grp = df.groupby('model')
    summ = pd.DataFrame()
    
    # Accuracy
    summ['MAPE (%)'] = grp['abs_gap_pct'].mean()
    summ['SDPE (%)'] = grp['gap_pct'].std()
    summ['Bias (%)'] = grp['gap_pct'].mean()
    summ['R2'] = grp.apply(lambda x: r2_score(x['true_cost'], x['pred_cost']))
    summ['MSE'] = grp['sq_error'].mean()
    
    # Timing
    summ['Avg Total (s)'] = grp['prediction_time_s'].mean()
    summ['Avg Inf (s)'] = grp['inference_time_s'].apply(nonzero_mean)
    
    # Speedup
    avg_opt_time = df['optimal_solve_time_s'].mean()
    summ['Speedup (x)'] = avg_opt_time / summ['Avg Total (s)']
    
    pd.options.display.float_format = '{:,.4f}'.format
    cols = [
        'MAPE (%)', 'SDPE (%)', 'Bias (%)', 'R2', 'MSE',
        'Avg Total (s)', 'Avg Inf (s)', 'Speedup (x)'
    ]
    print(summ[cols].sort_values('MAPE (%)'))
    print("="*90)
    print(f"Global Average Optimal Solve Time: {avg_opt_time:.4f} s")
    print("="*90)

def generate_plots(df, out_dir):
    print("\nGenerating plots...")
    
    # Time Boxplot
    plt.figure(figsize=(14, 8))
    opt_df = df.drop_duplicates(subset=['instance'])[['optimal_solve_time_s']].rename(columns={'optimal_solve_time_s': 'time_s'})
    opt_df['model'] = 'Optimal_Solver'
    
    model_df = df[['model', 'prediction_time_s']].rename(columns={'prediction_time_s': 'time_s'})
    plot_df = pd.concat([model_df, opt_df], ignore_index=True)
    
    sns.boxplot(data=plot_df, x='model', y='time_s', showfliers=False)
    plt.yscale('log')
    plt.title('Prediction Time (Log Scale)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_ND_plot_times.png")
    plt.close()
    
    # Error Boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='model', y='gap_pct', showfliers=False)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Percentage Error Distribution', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_ND_plot_errors.png")
    plt.close()

# =============================================================================
# 4. Main
# =============================================================================
def main():
    print("--- N-Dimensional TSP Benchmark (V3 Integration + GART) ---")
    
    try:
        test_instances = get_test_set_instances()
    except Exception as e:
        print(f"Error loading test set: {e}")
        return
        
    if not test_instances:
        print("No test instances found in metadata.")
        return
        
    base_df = generate_base_dataframe(test_instances)
    
    schedule = [
        # --- Classic Academic Estimators ---
        #('Cavdar', lambda: academic.estimate_tsp_cavdar),
        #('Vinel', lambda: academic.estimate_tsp_vinel),
        #('Composite', lambda: academic.estimate_tsp_composite),
        #('BHH', lambda: academic.estimate_tsp_bhh),
        ('MST_Ratio', lambda: academic.estimate_tsp_mst_ratio),
        #('Chien', lambda: academic.estimate_tsp_chien),
        # ('Christofides', lambda: academic.estimate_tsp_christofides),
        ('Hilbert', lambda: academic.estimate_tsp_hilbert),
        # --- Simulation / Sampling ---
        # ('EVT', lambda: academic.estimate_tsp_evt),
        # ('2Opt_Dist', lambda: academic.estimate_tsp_2opt_distribution),
        # ('Basel', lambda: academic.estimate_tsp_basel_willemain)
        # --- Machine Learning Models ---
        ('Linear_V3', lambda: TSP_V3_Linear_Estimator(str(SCRIPT_DIR / 'linear_model_v3'))),
        ('LGBM_V3', lambda: TSP_V3_LGBM_Estimator(str(SCRIPT_DIR / 'lgbm_model_v3'))),
        ('Neural_V3', lambda: TSP_V3_Neural_Estimator(str(SCRIPT_DIR / 'nn_est_alpha_v3'))),
        ('Interp_V3', lambda: TSP_Interpretable_Estimator(str(SCRIPT_DIR / 'interpretable_model_v3'))),
        
        # --- GART (Legacy ML) ---
        #('GART', lambda: GART_Adapter(str(SCRIPT_DIR / 'GART_1.0'))),
    ]

    for name, factory in schedule:
        process_model(name, factory, base_df)

    print("\n--- Aggregating Results ---")
    csv_files = glob(str(CHECKPOINT_DIR / "results_*.csv"))
    if csv_files:
        final_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        final_df.to_csv(FINAL_RESULTS_FILE, index=False)
        print(f"Saved combined results to {FINAL_RESULTS_FILE}")
        
        calculate_metrics_and_print(final_df)
        generate_plots(final_df, RESULTS_DIR)

    print("\n✅ ND Benchmark Complete.")

if __name__ == "__main__":
    main()