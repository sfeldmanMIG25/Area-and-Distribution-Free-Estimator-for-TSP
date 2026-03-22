import os
import sys

# --- CRITICAL: SILENCE LIGHTGBM C++ CORE ---
# This must be set before lightgbm is imported
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
# -------------------------------------------

import time
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import deque
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numba
import warnings

# Silence Python-level warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- NUMBA ACCELERATIONS ---

@numba.njit(fastmath=True, cache=True, parallel=True)
def _fast_centroid_stats(coords, centroid):
    """Accelerated centroid distance calculation."""
    n = coords.shape[0]
    dists = np.zeros(n, dtype=np.float32)
    for i in numba.prange(n):
        s = 0.0
        for j in range(coords.shape[1]):
            d = coords[i, j] - centroid[j]
            s += d*d
        dists[i] = np.sqrt(s)
    return np.mean(dists), np.std(dists), np.max(dists), dists

@numba.njit(fastmath=True, cache=True)
def compute_mst_degrees(rows, cols, n):
    """Fast degree computation for MST adjacency."""
    degrees = np.zeros(n, dtype=np.int32)
    for i in range(len(rows)):
        degrees[rows[i]] += 1
        degrees[cols[i]] += 1
    return degrees

# --- ESTIMATOR CLASS ---

class TSP_V3_LGBM_Estimator:
    """
    Optimized LightGBM V3 Accessor.
    Computes V3 Feature Set on-the-fly with 100D stability and silent execution.
    """
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(model_dir, 'lgbm_alpha_model_v3.joblib')
        if not os.path.exists(model_path):
            sys.exit(f"CRITICAL ERROR: LGBM Model not found at {model_path}")
            
        self.model = joblib.load(model_path)
        
        # --- REPAIR MODEL PARAMETERS TO FORCE SILENCE ---
        # Overwrite parameter aliases that trigger LightGBM warnings during predict()
        self.model.set_params(verbosity=-1)
        if hasattr(self.model, 'booster_'):
            self.model.booster_.params['verbosity'] = -1
        
        # Extract expected feature names
        try:
            self.features_required = self.model.feature_name_
        except AttributeError:
            self.features_required = self.model.booster_.feature_name()
            
        self._feature_cache = {}

    def _compute_v3_features(self, coords, n, d, grid_size):
        """Precise V3 feature set implementation with log-stabilization for high dimensions."""
        
        # Cache Check
        cache_key = hash(coords.tobytes())
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        feats = {'n_customers': n, 'dimension': d}
        
        # Geometric Spread & Hypervolume Stability for 100D
        rngs = np.ptp(coords, axis=0).astype(float)
        rngs[rngs < 1e-9] = 1e-9
        
        # Stable Log-Space product
        log_hv = np.sum(np.log(rngs))
        # Cap at float64 ceiling (~1e300) to prevent INF errors in LGBM input
        hypervolume = np.exp(min(log_hv, 690.0)) 
        
        feats['bounding_hypervolume'] = hypervolume
        feats['node_density'] = n / hypervolume if hypervolume > 1e-15 else 0.0
        feats['aspect_ratio'] = np.max(rngs) / np.min(rngs)
        
        # Centroid Stats (Numba)
        cent = np.mean(coords, axis=0, dtype=np.float32)
        c_mn, c_st, c_mx, c_raw = _fast_centroid_stats(coords, cent)
        feats['centroid_dist_mean'], feats['centroid_dist_std'], feats['centroid_dist_max'] = c_mn, c_st, c_mx
        feats['centroid_dist_iqr'] = np.subtract(*np.percentile(c_raw, [75, 25]))

        # MST Block
        dist_mat = cdist(coords, coords, 'euclidean').astype(np.float32)
        np.fill_diagonal(dist_mat, 0)
        mst_csr = minimum_spanning_tree(dist_mat)
        edges = mst_csr.data
        mst_len = np.sum(edges)
        
        feats['mst_total_length'] = mst_len
        feats['mst_edge_mean'], feats['mst_edge_std'] = np.mean(edges), np.std(edges)
        feats['mst_edge_skew'], feats['mst_edge_kurtosis'] = stats.skew(edges), stats.kurtosis(edges)
        feats['mst_edge_max'] = np.max(edges)
        
        percs = np.percentile(edges, [10, 25, 50, 75, 90])
        for i, p in enumerate([10, 25, 50, 75, 90]): 
            feats[f'mst_edge_q{p}'] = percs[i]
        
        # Clustering Proxies
        k_dom = max(1, int(np.sqrt(n)))
        feats['mst_dominance_ratio'] = np.sum(np.partition(edges, -k_dom)[-k_dom:]) / (mst_len + 1e-9)
        feats['mst_gap_ratio'] = feats['mst_edge_max'] / (percs[2] + 1e-9)
        
        rows, cols = mst_csr.nonzero()
        degrees = compute_mst_degrees(rows, cols, n)
        feats['mst_leaf_ratio'] = np.sum(degrees == 1) / n
        feats['mst_degree_mean'], feats['mst_degree_std'] = np.mean(degrees), np.std(degrees)
        feats['mst_degree_max'] = np.max(degrees)
        feats['large_edge_count'] = np.sum(edges > feats['mst_edge_mean'] + feats['mst_edge_std'])
        
        # Diameter
        adj = [[] for _ in range(n)]
        for i in range(len(rows)):
            adj[rows[i]].append((cols[i], edges[i]))
            adj[cols[i]].append((rows[i], edges[i]))
            
        def farthest(start):
            dists = np.full(n, -1.0); dists[start] = 0.0; q = deque([start])
            fn, md = start, 0.0
            while q:
                u = q.popleft()
                if dists[u] > md: md, fn = dists[u], u
                for v, w in adj[u]:
                    if dists[v] < 0: dists[v] = dists[u] + w; q.append(v)
            return fn, md
        
        node1, _ = farthest(0)
        _, diam = farthest(node1)
        feats['mst_diameter'] = diam
        feats['mst_diameter_normalized'] = diam / (mst_len + 1e-9)

        result = (feats, mst_len)
        self._feature_cache[cache_key] = result
        return result

    def estimate(self, coordinates, dimension, grid_size):
        """Main entry point. Returns predictions and silent timing metrics."""
        coords = np.array(coordinates, dtype=np.float32)
        n = len(coords)
        
        # 1. Feature Generation
        t0 = time.perf_counter()
        f_dict, mst_len = self._compute_v3_features(coords, n, dimension, grid_size)
        t_feat = time.perf_counter() - t0
        
        # 2. DataFrame formatting
        t1 = time.perf_counter()
        df_input = pd.DataFrame([{k: f_dict.get(k, 0.0) for k in self.features_required}])
        
        # 3. Inference
        alpha = self.model.predict(df_input)[0]
        t_inf = time.perf_counter() - t1
        
        alpha = np.clip(alpha, 1.0, 2.0)
        
        return {
            'estimate': float(alpha * mst_len),
            'alpha': float(alpha),
            'mst_length': float(mst_len),
            'feature_time': t_feat,
            'inference_time': t_inf
        }

# --- N-DIMENSIONAL BENCHMARK ---

if __name__ == "__main__":
    print("=== LightGBM V3 Performance Benchmark (N-Dimensional) ===")
    
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_F = os.path.join(CUR_DIR, '..', 'tsp_features_v3.csv')
    INST_D = os.path.join(CUR_DIR, '..', 'instances')
    
    if not os.path.exists(DATA_F):
        sys.exit(f"CRITICAL ERROR: Features file not found at {DATA_F}")
        
    df_full = pd.read_csv(DATA_F)
    test_defs = df_full[df_full['split'] == 'test']
    
    if test_defs.empty:
        test_defs = df_full.sample(min(200, len(df_full)))

    estimator = TSP_V3_LGBM_Estimator(CUR_DIR)
    results = []
    
    print(f"Benchmarking {len(test_defs)} instances...")
    for _, row in tqdm(test_defs.iterrows(), total=len(test_defs)):
        instance_name = row['instance_name']
        optimal_cost = row['optimal_cost']
        
        inst_path = os.path.join(INST_D, f"{instance_name}.json")
        if not os.path.exists(inst_path): continue
            
        with open(inst_path, 'r') as f:
            data = json.load(f)
            
        res = estimator.estimate(
            data['coordinates'], 
            data['dimension'], 
            data.get('grid_size', 1000)
        )
        
        pe = (res['estimate'] - optimal_cost) / optimal_cost
        results.append({
            'opt': optimal_cost, 'pred': res['estimate'], 'pe': pe * 100,
            'feat_t': res['feature_time'], 'inf_t': res['inference_time']
        })

    if results:
        df_r = pd.DataFrame(results)
        mape = df_r['pe'].abs().mean()
        sdpe = df_r['pe'].std()
        
        print("\n" + "="*50)
        print("         FINAL BENCHMARK RESULTS (LGBM V3)")
        print("="*50)
        print(f"Total Instances      : {len(df_r)}")
        print(f"MSE                  : {mean_squared_error(df_r['opt'], df_r['pred']):,.2f}")
        print(f"Avg % Error (MAPE)   : {mape:.4f}%")
        print(f"SD of % Error (SDPE) : {sdpe:.4f}%")
        print("-" * 50)
        print(f"Avg Feature Time     : {df_r['feat_t'].mean()*1000:.3f} ms")
        print(f"Avg Inference Time   : {df_r['inf_t'].mean()*1000:.3f} ms")
        print(f"Avg Total Latency    : {(df_r['feat_t'].mean() + df_r['inf_t'].mean())*1000:.3f} ms")
        print("="*50)