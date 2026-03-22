import os
import sys

# --- CRITICAL FIX FOR WINDOWS/JOBLIB ---
# Prevent sklearn/joblib from spawning subprocesses inside workers.
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, os.cpu_count()))
# ---------------------------------------

import numpy as np
import json
import struct
from collections import deque
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# SOURCE OF TRUTH FOR SPLITS (If available)
BASELINE_DATA_FILE = os.path.join(ROOT_DIR, 'tsp_features.csv') 
OUTPUT_FILE = os.path.join(ROOT_DIR, 'tsp_features_v3.csv')

INSTANCES_DIR = os.path.join(ROOT_DIR, "instances")
SOLUTIONS_DIR = os.path.join(ROOT_DIR, "solutions")

RANDOM_STATE = 42

# --- BINARY LOADER (OPTIMIZED) ---
def load_instance_data(instance_name):
    """
    Loads instance data, prioritizing the fast binary format.
    Falls back to JSON if binary is missing.
    """
    base_name = instance_name[:-5] if instance_name.endswith('.json') else instance_name
    
    bin_path = os.path.join(INSTANCES_DIR, f"{base_name}.bin")
    json_path = os.path.join(INSTANCES_DIR, f"{base_name}.json")
    
    # Try Binary First
    if os.path.exists(bin_path):
        # Strict loading: if files are corrupt, script will crash (per user rules)
        with open(bin_path, 'rb') as f:
            n, d, grid_size = struct.unpack('III', f.read(12))
            dist_len = struct.unpack('I', f.read(4))[0]
            _ = f.read(dist_len) # Skip dist string
            coords_buffer = f.read(n * d * 4)
            coords = np.frombuffer(coords_buffer, dtype=np.float32).reshape(n, d)
        return {
            'instance_name': base_name, 'n_customers': n, 
            'dimension': d, 'grid_size': grid_size, 'coordinates': coords
        }

    # Fallback to JSON
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            data['coordinates'] = np.array(data['coordinates'], dtype=np.float32)
            data['instance_name'] = base_name
            return data
            
    return None

# --- V3 FEATURE LOGIC (MST-Centric + Clustering Proxies) ---

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

def compute_features_for_instance_v3(inst_data, sol_data):
    """
    Compute the 25-feature, dimension-agnostic, MST-centric feature set.
    Includes advanced clustering proxies (Dominance, Leaf Ratio, Gap Ratio).
    """
    coords = inst_data['coordinates']
    coords = np.unique(coords, axis=0)
    
    n = len(coords)
    d = inst_data['dimension']
    grid_size = inst_data.get('grid_size', 0)

    if n < 3: return None
    
    features = {
        'instance_name': inst_data['instance_name'],
        'n_customers': n, 
        'dimension': d,
        'grid_size': grid_size
    }
    
    features['optimal_cost'] = sol_data.get('optimal_cost', np.nan)

    # === Group 2: Geometric Spread ===
    
    dim_ranges = np.ptp(coords, axis=0).astype(float)
    dim_ranges[dim_ranges < 1e-9] = 1e-9 
    
    features['bounding_hypervolume'] = np.prod(dim_ranges)
    
    if features['bounding_hypervolume'] > 1e-12:
        features['node_density'] = n / features['bounding_hypervolume']
    else:
        features['node_density'] = 0.0
        
    min_range = np.min(dim_ranges)
    if min_range > 1e-12:
        features['aspect_ratio'] = np.max(dim_ranges) / min_range
    else:
        features['aspect_ratio'] = 1.0

    centroid = np.mean(coords, axis=0)
    centroid_dists = np.linalg.norm(coords - centroid, axis=1)
    
    features['centroid_dist_mean'] = np.mean(centroid_dists)
    features['centroid_dist_std'] = np.std(centroid_dists)
    features['centroid_dist_max'] = np.max(centroid_dists)
    q75, q25 = np.percentile(centroid_dists, [75, 25])
    features['centroid_dist_iqr'] = q75 - q25

    # === Group 3: Topological Structure ===
    
    dist_matrix = cdist(coords, coords, 'euclidean') + 1e-9
    np.fill_diagonal(dist_matrix, 0)
    
    mst_csr = minimum_spanning_tree(dist_matrix)
    mst_edges = mst_csr.data 
    
    if len(mst_edges) == 0: 
        mst_edges = np.array([0.0])
        
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

    # --- ADVANCED MST CLUSTERING PROXIES ---
    
    # 1. MST Dominance Ratio (Sum of top sqrt(N) edges / Total Length)
    k_dom = max(1, int(np.sqrt(n)))
    # partition moves largest k elements to the end
    if len(mst_edges) >= k_dom:
        top_k_edges = np.partition(mst_edges, -k_dom)[-k_dom:]
        features['mst_dominance_ratio'] = np.sum(top_k_edges) / total_mst_len if total_mst_len > 1e-9 else 0.0
    else:
        features['mst_dominance_ratio'] = 1.0

    # 3. Gap Ratio (Max Edge / Median Edge)
    median_edge = percs[2]
    features['mst_gap_ratio'] = (max_edge / median_edge) if median_edge > 1e-9 else 0.0

    # Topology stats (Degree & Diameter)
    mst_adj = [[] for _ in range(n)]
    degrees = np.zeros(n, dtype=int)
    rows, cols = mst_csr.nonzero()
    
    for i in range(len(rows)):
        u, v, dist = rows[i], cols[i], mst_edges[i]
        mst_adj[u].append((v, dist))
        mst_adj[v].append((u, dist))
        degrees[u] += 1
        degrees[v] += 1
    
    # 2. Leaf Ratio (Count of Degree=1 / N)
    features['mst_leaf_ratio'] = np.sum(degrees == 1) / n

    features['mst_degree_mean'] = np.mean(degrees)
    features['mst_degree_std'] = np.std(degrees)
    features['mst_degree_max'] = np.max(degrees)
    
    diameter = _compute_tree_diameter(mst_adj, n)
    features['mst_diameter'] = diameter
    
    # 4. Normalized Diameter (Diameter / Total Length)
    features['mst_diameter_normalized'] = diameter / total_mst_len if total_mst_len > 1e-9 else 0.0
    
    features['large_edge_count'] = np.sum(mst_edges > mst_edge_mean + mst_edge_std)
            
    return features

def process_file_worker(filename):
    """Worker function for ProcessPoolExecutor"""
    if not filename.endswith('.json'): return None
    
    # Check strict existence to avoid crashes
    base_name = filename[:-5]
    sol_filename = f"{base_name}.sol.json"
    sol_path = os.path.join(SOLUTIONS_DIR, sol_filename)
    
    if not os.path.exists(sol_path): return None
    
    inst = load_instance_data(filename)
    if not inst: return None
    
    with open(sol_path, 'r') as f:
        sol = json.load(f)
        
    return compute_features_for_instance_v3(inst, sol)

def create_stratified_split(df):
    """
    Replicates the exact split logic from feature_creator.py
    if the baseline split file is missing.
    """
    print("Applying Fresh Stratified Split Logic...")
    
    # 1. d=100 -> TEST
    mask_d100 = df['dimension'] == 100
    df_d100 = df[mask_d100].copy()
    df_others = df[~mask_d100].copy()
    
    df_d100['split'] = 'test'
    print(f"Locked {len(df_d100)} instances (d=100) into Test.")
    
    if len(df_others) > 0:
        # Group by [Dimension, N, Grid]
        stratum_cols = ['dimension', 'n_customers', 'grid_size']
        
        # Shuffle within groups
        df_others = df_others.groupby(stratum_cols, group_keys=False).apply(
            lambda x: x.sample(frac=1, random_state=RANDOM_STATE)
        )
        
        # Calculate ratio within the group
        df_others['group_frac'] = df_others.groupby(stratum_cols).cumcount() / \
                                  df_others.groupby(stratum_cols)['instance_name'].transform('count')
        
        # 70% Train, 20% Val, 10% Test
        conditions = [
            df_others['group_frac'] < 0.10,          # 10% Test
            df_others['group_frac'] < (0.10 + 0.20)  # 20% Val (Cumulative 0.30)
        ]
        choices = ['test', 'val']
        
        df_others['split'] = np.select(conditions, choices, default='train')
        df_others = df_others.drop(columns=['group_frac'])
    
    return pd.concat([df_others, df_d100], ignore_index=True)

# --- MAIN ---
if __name__ == '__main__':
    print(f"--- Feature Generator V3 (Auto-Split + New Proxies) ---")
    
    split_map = None
    files_to_process = []
    
    # 1. Check for Baseline Split (V1)
    if os.path.exists(BASELINE_DATA_FILE):
        print(f"Loading split map from: {BASELINE_DATA_FILE}")
        base_df = pd.read_csv(BASELINE_DATA_FILE, usecols=['instance_name', 'split'])
        split_map = dict(zip(base_df['instance_name'], base_df['split']))
        target_instances = list(split_map.keys())
        files_to_process = [f"{name}.json" for name in target_instances]
    else:
        print(f"Baseline file '{BASELINE_DATA_FILE}' not found.")
        print("Scanning instance directory for all available files...")
        files_to_process = [f for f in os.listdir(INSTANCES_DIR) if f.endswith('.json')]
    
    print(f"Targeting {len(files_to_process)} instances.")

    # 2. Compute Features in Parallel
    num_workers = max(1, os.cpu_count())
    print(f"Starting computation with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_file_worker, files_to_process), 
                            total=len(files_to_process), desc="Computing V3 Features"))

    all_features = [res for res in results if res is not None]

    if not all_features:
        print("Error: No features generated.")
        sys.exit(1)

    # 3. Merge & Split
    df = pd.DataFrame(all_features)
    
    if split_map:
        # Apply existing splits
        df['split'] = df['instance_name'].map(split_map)
        missing_mask = df['split'].isna()
        if missing_mask.sum() > 0:
            print(f"Warning: {missing_mask.sum()} instances missing from baseline split map. Dropping them.")
            df = df.dropna(subset=['split'])
    else:
        # Generate fresh splits
        df = create_stratified_split(df)

    # 4. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ V3 Features saved to {OUTPUT_FILE}")
    print("Split Consistency Check:")
    print(df['split'].value_counts(normalize=True).map("{:.2%}".format))