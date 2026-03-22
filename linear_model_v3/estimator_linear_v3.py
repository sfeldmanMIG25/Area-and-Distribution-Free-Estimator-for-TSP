import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from collections import deque
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numba
import warnings

warnings.filterwarnings("ignore")

@numba.njit(fastmath=True, cache=True, parallel=True)
def _fast_centroid_stats(coords, centroid):
    """Numba-accelerated centroid stats."""
    n = coords.shape[0]
    dists = np.zeros(n, dtype=np.float32)
    for i in numba.prange(n):
        s = 0.0
        for j in range(coords.shape[1]):
            d = coords[i, j] - centroid[j]
            s += d*d
        dists[i] = np.sqrt(s)
    return np.mean(dists), np.std(dists), np.max(dists), dists

class TSP_V3_Linear_Estimator:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model = joblib.load(os.path.join(model_dir, 'linear_alpha_model_v3.joblib'))
        # Retrieve feature names expected by the model
        try:
            self.feature_order = self.model.feature_names_in_
        except AttributeError:
            self.feature_order = None

    def _compute_features_raw(self, coords, n, d, grid_size):
        """Computes features with explicit Log-Transform for stability."""
        feats = {'n_customers': n, 'dimension': d}
        
        # --- Stability: Log-Features ---
        rngs = np.ptp(coords, axis=0).astype(float)
        rngs[rngs < 1e-9] = 1e-9
        
        # Calculate log_hypervolume directly to avoid overflow
        log_hv = np.sum(np.log(rngs))
        
        # We compute log versions directly as the linear model likely selected them
        feats['log_bounding_hypervolume'] = log_hv
        feats['log_node_density'] = np.log(n) - log_hv
        
        feats['aspect_ratio'] = np.max(rngs) / np.min(rngs)
        
        # Centroid
        cent = np.mean(coords, axis=0, dtype=np.float32)
        c_mn, c_st, c_mx, c_raw = _fast_centroid_stats(coords, cent)
        feats['centroid_dist_mean'], feats['centroid_dist_std'] = c_mn, c_st
        feats['centroid_dist_max'] = c_mx
        feats['centroid_dist_iqr'] = np.subtract(*np.percentile(c_raw, [75, 25]))

        # MST
        dist_mat = cdist(coords, coords, 'euclidean').astype(np.float32)
        np.fill_diagonal(dist_mat, 0)
        mst = minimum_spanning_tree(dist_mat)
        edges = mst.data
        mst_len = np.sum(edges)
        
        feats['mst_total_length'] = mst_len
        feats['mst_edge_mean'], feats['mst_edge_std'] = np.mean(edges), np.std(edges)
        feats['mst_edge_skew'] = stats.skew(edges) if len(edges) > 2 else 0.0
        feats['mst_edge_kurtosis'] = stats.kurtosis(edges) if len(edges) > 2 else 0.0
        feats['mst_edge_max'] = np.max(edges)
        
        percs = np.percentile(edges, [10, 25, 50, 75, 90])
        for i, p in enumerate([10, 25, 50, 75, 90]): 
            feats[f'mst_edge_q{p}'] = percs[i]
        
        # Proxies
        k_dom = max(1, int(np.sqrt(n)))
        feats['mst_dominance_ratio'] = np.sum(np.partition(edges, -k_dom)[-k_dom:]) / (mst_len + 1e-9)
        feats['mst_gap_ratio'] = feats['mst_edge_max'] / (percs[2] + 1e-9)
        
        degrees = np.zeros(n); rows, cols = mst.nonzero()
        for i in range(len(rows)): degrees[rows[i]]+=1; degrees[cols[i]]+=1
        
        feats['mst_leaf_ratio'] = np.sum(degrees == 1) / n
        feats['mst_degree_mean'] = np.mean(degrees)
        feats['mst_degree_std'] = np.std(degrees)
        feats['mst_degree_max'] = np.max(degrees)
        feats['large_edge_count'] = np.sum(edges > feats['mst_edge_mean'] + feats['mst_edge_std'])
        
        # Diameter
        adj = [[] for _ in range(n)]
        for i in range(len(rows)):
            adj[rows[i]].append((cols[i], edges[i])); adj[cols[i]].append((rows[i], edges[i]))
        def farthest(start):
            dists = np.full(n, -1.0); dists[start] = 0.0; q = deque([start])
            fn, md = start, 0.0
            while q:
                u = q.popleft()
                if dists[u] > md: md, fn = dists[u], u
                for v, w in adj[u]:
                    if dists[v] < 0: dists[v] = dists[u] + w; q.append(v)
            return fn, md
        n1, _ = farthest(0); _, diam = farthest(n1)
        feats['mst_diameter'] = diam
        feats['mst_diameter_normalized'] = diam / (mst_len + 1e-9)
        
        return feats, mst_len

    def estimate(self, coordinates, dimension, grid_size):
        coords = np.array(coordinates, dtype=np.float32)
        t0 = time.perf_counter()
        f_dict, mst_len = self._compute_features_raw(coords, len(coords), dimension, grid_size)
        t_feat = time.perf_counter() - t0
        
        # DataFrame construction matches training pipeline expectation
        df_input = pd.DataFrame([f_dict])
        
        # Alignment with model features
        if self.feature_order is not None:
            for col in self.feature_order:
                if col not in df_input.columns:
                    df_input[col] = 0.0
            df_input = df_input[self.feature_order]
        else:
            cols_to_drop = ['mst_total_length']
            df_input = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns])
            
        t1 = time.perf_counter()
        alpha = self.model.predict(df_input)[0]
        t_inf = time.perf_counter() - t1
        
        alpha = np.clip(alpha, 1.0, 2.0)
        return {
            'estimate': alpha * mst_len,
            'alpha': alpha,
            'mst_length': mst_len,
            'feature_time': t_feat,
            'inference_time': t_inf
        }

# --- N-DIMENSIONAL TEST SET BENCHMARK ---

if __name__ == "__main__":
    print("=== Linear Regression V3 Performance Benchmark ===")
    
    # 1. Setup Paths
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_F = os.path.join(CUR_DIR, '..', 'tsp_features_v3.csv')
    INST_D = os.path.join(CUR_DIR, '..', 'instances')
    
    # 2. Load Test Definitions
    if not os.path.exists(DATA_F):
        print(f"CRITICAL: {DATA_F} not found.")
        exit(1)
        
    df_full = pd.read_csv(DATA_F)
    test_defs = df_full[df_full['split'] == 'test']
    
    if test_defs.empty:
        print("Warning: No test split found. Running on sample.")
        test_defs = df_full.sample(min(100, len(df_full)))

    # 3. Initialize & Benchmark
    estimator = TSP_V3_Linear_Estimator(CUR_DIR)
    results = []
    
    print(f"Benchmarking {len(test_defs)} N-Dimensional instances...")
    for _, row in tqdm(test_defs.iterrows(), total=len(test_defs)):
        inst_path = os.path.join(INST_D, f"{row['instance_name']}.json")
        if not os.path.exists(inst_path): continue
            
        with open(inst_path, 'r') as f:
            data = json.load(f)
            
        res = estimator.estimate(
            data['coordinates'], 
            data['dimension'], 
            data.get('grid_size', 1000)
        )
        
        pe = (res['estimate'] - row['optimal_cost']) / row['optimal_cost']
        results.append({
            'opt': row['optimal_cost'], 
            'pred': res['estimate'], 
            'pe': pe * 100,
            'feat_t': res['feature_time'],
            'inf_t': res['inference_time']
        })

    # 4. Report
    if results:
        df_r = pd.DataFrame(results)
        mse = mean_squared_error(df_r['opt'], df_r['pred'])
        mape = df_r['pe'].abs().mean()
        sdpe = df_r['pe'].std()
        
        print("\n" + "="*50)
        print("      BENCHMARK REPORT (LINEAR V3)      ")
        print("="*50)
        print(f"Total Instances      : {len(df_r)}")
        print(f"MSE                  : {mse:,.2f}")
        print(f"Avg % Error (MAPE)   : {mape:.4f}%")
        print(f"SD of % Error (SDPE) : {sdpe:.4f}%")
        print("-" * 50)
        print(f"Avg Feature Time     : {df_r['feat_t'].mean()*1000:.3f} ms")
        print(f"Avg Inference Time   : {df_r['inf_t'].mean()*1000:.3f} ms")
        print(f"Avg Total Latency    : {(df_r['feat_t'].mean() + df_r['inf_t'].mean())*1000:.3f} ms")
        print("="*50)
    else:
        print("Error: No instances processed.")