import os
import pandas as pd
import numpy as np
import joblib
import json
import time
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import stats
import warnings
# --- SILENCE ALL WARNINGS ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PYTHONWARNINGS'] = 'ignore'
# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

INSTANCES_DIR = r"C:\Area-and-Distribution-Free-Estimators-for-TSP\Generalized_TSP_Analysis\instances"
SOLUTIONS_DIR = r"C:\Area-and-Distribution-Free-Estimators-for-TSP\Generalized_TSP_Analysis\solutions"

# Model Path
MODEL_FILE = os.path.join(ROOT_DIR, 'lgbm_alpha_model_v3.joblib')
OUTPUT_CSV = os.path.join(ROOT_DIR, 'benchmark_results_v3_2d.csv')

# --- V3 FEATURE GENERATION (Exact Copy from feature_creator_v3.py) ---

def _compute_tree_diameter(mst_adj, n):
    """Compute the weighted diameter of the tree using two BFS runs."""
    def farthest(start_node):
        distances = np.full(n, -1.0)
        distances[start_node] = 0.0
        queue = deque([start_node])
        farthest_node, max_dist = start_node, 0.0
        
        while queue:
            u = queue.popleft()
            if distances[u] > max_dist:
                max_dist = distances[u]
                farthest_node = u
            
            for v, weight in mst_adj[u]:
                if distances[v] < 0:
                    distances[v] = distances[u] + weight
                    queue.append(v)
        
        final_farthest_node = np.argmax(distances)
        return final_farthest_node, distances[final_farthest_node]

    if n < 2: return 0.0
    node1, _ = farthest(0)
    _, diameter = farthest(node1)
    return diameter

def compute_features_for_instance_v3(inst_data):
    """
    Computes the MST-Centric Feature Set V3.
    """
    # Parse data depending on format (dict vs raw json)
    if isinstance(inst_data.get('coordinates'), list):
        coords = np.array(inst_data['coordinates'], dtype=np.float32)
    else:
        coords = inst_data['coordinates']
        
    n = inst_data['n_customers']
    d = inst_data['dimension']
    
    # Handle duplicates just in case, though V3 generator handles this
    coords = np.unique(coords, axis=0)
    n = len(coords)

    if n < 3:
        # Fallback for trivial cases
        return None, 0.0

    features = {
        'n_customers': n, 
        'dimension': d,
    }

    # === Group 2: Geometric Spread ===
    dim_ranges = np.ptp(coords, axis=0).astype(float)
    dim_ranges[dim_ranges < 1e-9] = 1e-9 
    
    features['bounding_hypervolume'] = np.prod(dim_ranges)
    features['node_density'] = n / features['bounding_hypervolume'] if features['bounding_hypervolume'] > 0 else 0
    features['aspect_ratio'] = np.max(dim_ranges) / np.min(dim_ranges)

    centroid = np.mean(coords, axis=0)
    centroid_dists = np.linalg.norm(coords - centroid, axis=1)
    
    features['centroid_dist_mean'] = np.mean(centroid_dists)
    features['centroid_dist_std'] = np.std(centroid_dists)
    features['centroid_dist_max'] = np.max(centroid_dists)
    q75, q25 = np.percentile(centroid_dists, [75, 25])
    features['centroid_dist_iqr'] = q75 - q25

    # === Group 3: Topological Structure (MST) ===
    dist_matrix = cdist(coords, coords, 'euclidean') + 1e-9
    np.fill_diagonal(dist_matrix, 0)
    
    mst_csr = minimum_spanning_tree(dist_matrix)
    mst_edges = mst_csr.data 
    
    if len(mst_edges) == 0: mst_edges = np.array([0.0])
        
    total_mst_len = np.sum(mst_edges)
    features['mst_total_length'] = total_mst_len

    mst_edge_mean = np.mean(mst_edges)
    mst_edge_std = np.std(mst_edges)
    
    features['mst_edge_mean'] = mst_edge_mean
    features['mst_edge_std'] = mst_edge_std
    
    if mst_edge_std > 1e-9:
        features['mst_edge_skew'] = stats.skew(mst_edges)
        features['mst_edge_kurtosis'] = stats.kurtosis(mst_edges)
    else:
        features['mst_edge_skew'] = 0.0
        features['mst_edge_kurtosis'] = 0.0

    max_edge = np.max(mst_edges)
    features['mst_edge_max'] = max_edge
    
    percs = np.percentile(mst_edges, [10, 25, 50, 75, 90])
    features['mst_edge_q10'] = percs[0]
    features['mst_edge_q25'] = percs[1]
    features['mst_edge_q50'] = percs[2]
    features['mst_edge_q75'] = percs[3]
    features['mst_edge_q90'] = percs[4]

    # --- ADVANCED MST PROXIES (V3 SPECIFIC) ---
    
    # 1. MST Dominance Ratio
    k_dom = max(1, int(np.sqrt(n)))
    if len(mst_edges) >= k_dom:
        top_k_edges = np.partition(mst_edges, -k_dom)[-k_dom:]
        features['mst_dominance_ratio'] = np.sum(top_k_edges) / total_mst_len if total_mst_len > 1e-9 else 0.0
    else:
        features['mst_dominance_ratio'] = 1.0

    # 2. Gap Ratio
    median_edge = percs[2]
    features['mst_gap_ratio'] = (max_edge / median_edge) if median_edge > 1e-9 else 0.0

    # Topology Stats
    mst_adj = [[] for _ in range(n)]
    degrees = np.zeros(n, dtype=int)
    rows, cols = mst_csr.nonzero()
    
    for i in range(len(rows)):
        u, v, dist = rows[i], cols[i], mst_edges[i]
        mst_adj[u].append((v, dist))
        mst_adj[v].append((u, dist))
        degrees[u] += 1
        degrees[v] += 1
    
    # 3. Leaf Ratio
    features['mst_leaf_ratio'] = np.sum(degrees == 1) / n

    features['mst_degree_mean'] = np.mean(degrees)
    features['mst_degree_std'] = np.std(degrees)
    features['mst_degree_max'] = np.max(degrees)
    
    diameter = _compute_tree_diameter(mst_adj, n)
    features['mst_diameter'] = diameter
    
    # 4. Normalized Diameter
    features['mst_diameter_normalized'] = diameter / total_mst_len if total_mst_len > 1e-9 else 0.0
    
    features['large_edge_count'] = np.sum(mst_edges > mst_edge_mean + mst_edge_std)
            
    return features, total_mst_len

# --- BENCHMARK LOGIC ---

def run_benchmark():
    print("--- Running LGBM V3 Model Benchmark (2D Test Set) ---")
    
    # 1. Validation
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file not found: {MODEL_FILE}")
        return
    if not os.path.exists(INSTANCES_DIR) or not os.path.exists(SOLUTIONS_DIR):
        print(f"ERROR: Directories not found.\nInstances: {INSTANCES_DIR}\nSolutions: {SOLUTIONS_DIR}")
        return

    # 2. Load Model
    print(f"Loading model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    
    # Robustly get feature names
    try:
        model_features = model.feature_name_
    except AttributeError:
        model_features = model.booster_.feature_name()
        
    print(f"Model loaded. Expecting {len(model_features)} features.")

    # 3. Match Files
    all_inst = set(f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json'))
    all_sol = set(f for f in os.listdir(SOLUTIONS_DIR) if f.endswith('.sol.json'))
    
    # Match instance "X.json" with solution "X.sol.json"
    files_to_process = []
    for inst_file in all_inst:
        base_name = inst_file.replace('.json', '')
        sol_file = f"{base_name}.sol.json"
        if sol_file in all_sol:
            files_to_process.append(inst_file)
            
    files_to_process.sort()
    
    if not files_to_process:
        print("ERROR: No matching instance/solution pairs found.")
        return
        
    print(f"Found {len(files_to_process)} benchmark pairs.")

    # 4. Process Loop
    results = []
    
    for filename in tqdm(files_to_process, desc="Benchmarking"):
        inst_path = os.path.join(INSTANCES_DIR, filename)
        sol_path = os.path.join(SOLUTIONS_DIR, filename.replace('.json', '.sol.json'))
        
        try:
            with open(inst_path, 'r') as f: inst_data = json.load(f)
            with open(sol_path, 'r') as f: sol_data = json.load(f)

            optimal_cost = sol_data.get('optimal_cost')
            if optimal_cost is None or optimal_cost <= 0: continue

            # Get optimal solve time (safe default)
            opt_time = 1e-9
            if sol_data.get('optimal_solver') == 'concorde':
                opt_time = sol_data.get('concorde_time_s', 1e-9)
            elif sol_data.get('optimal_solver') == 'lkh':
                opt_time = sol_data.get('lkh_time_s', 1e-9)
            
            # --- Feature Phase ---
            t0 = time.perf_counter()
            features, mst_len = compute_features_for_instance_v3(inst_data)
            t1 = time.perf_counter()
            
            if features is None: continue
            
            # Prepare for prediction
            df_feat = pd.DataFrame([features])
            
            # Ensure columns match model expectation (fill missing with NaN)
            for feat in model_features:
                if feat not in df_feat.columns:
                    df_feat[feat] = np.nan
            
            # Reorder
            X_pred = df_feat[model_features]

            # --- Prediction Phase ---
            t2 = time.perf_counter()
            pred_alpha = model.predict(X_pred)[0]
            t3 = time.perf_counter()
            
            # Metrics
            pred_alpha = np.clip(pred_alpha, 1.0, 2.0)
            pred_cost = pred_alpha * mst_len
            
            # SDPE = (Pred - Opt) / Opt
            sdpe = (pred_cost - optimal_cost) / optimal_cost
            
            results.append({
                'instance_name': inst_data['instance_name'],
                'n_customers': inst_data['n_customers'],
                'optimal_cost': optimal_cost,
                'predicted_cost': pred_cost,
                'sdpe_pct': sdpe * 100,
                'abs_pct_error': abs(sdpe) * 100,
                'time_feat': t1 - t0,
                'time_pred': t3 - t2,
                'time_total': (t1 - t0) + (t3 - t2),
                'optimal_time': opt_time
            })
            
        except Exception as e:
            continue

    # 5. Analysis
    if not results:
        print("No valid results generated.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Speedup
    df['speedup_factor'] = df['optimal_time'] / df['time_total']
    
    # Metrics
    sdpe_mean = df['sdpe_pct'].mean()
    sdpe_std = df['sdpe_pct'].std()
    mape = df['abs_pct_error'].mean()
    rmse = np.sqrt(mean_squared_error(df['optimal_cost'], df['predicted_cost']))
    r2 = r2_score(df['optimal_cost'], df['predicted_cost'])
    
    print("\n" + "="*40)
    print("      BENCHMARK REPORT (V3 MODEL)      ")
    print("="*40)
    print(f"Instances Processed: {len(df)}")
    print(f"R^2 Score:           {r2:.5f}")
    print(f"RMSE:                {rmse:.4f}")
    print("-" * 40)
    print(f"SDPE (Bias) %:       {sdpe_mean:.4f}%  (± {sdpe_std:.4f})")
    print(f"MAPE (Accuracy) %:   {mape:.4f}%")
    print("-" * 40)
    print(f"Avg Prediction Time: {df['time_total'].mean():.6f}s")
    print(f"Avg Optimal Time:    {df['optimal_time'].mean():.6f}s")
    print(f"Avg Speedup Factor:  {df['speedup_factor'].mean():.2f}x")
    print("="*40)

    # 6. Plots
    print(f"\nGenerating plots in {ROOT_DIR}...")
    sns.set_style("whitegrid")
    
    # SDPE Hist
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sdpe_pct'], bins=30, kde=True)
    plt.title('Distribution of Signed Deviation Percentage Error (SDPE)')
    plt.xlabel('Error % ((Predicted - Optimal) / Optimal)')
    plt.axvline(0, color='r', linestyle='--')
    plt.savefig(os.path.join(ROOT_DIR, 'v3_sdpe_dist.png'))
    plt.close()

    # Scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(df['optimal_cost'], df['predicted_cost'], alpha=0.6)
    plt.plot([df['optimal_cost'].min(), df['optimal_cost'].max()], 
             [df['optimal_cost'].min(), df['optimal_cost'].max()], 'r--')
    plt.xlabel('Optimal Cost')
    plt.ylabel('Predicted Cost')
    plt.title(f'Predicted vs Optimal Cost (R2={r2:.4f})')
    plt.savefig(os.path.join(ROOT_DIR, 'v3_pred_vs_opt.png'))
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    run_benchmark()