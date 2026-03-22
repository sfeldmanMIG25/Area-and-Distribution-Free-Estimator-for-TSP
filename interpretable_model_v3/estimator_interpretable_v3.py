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
    n = coords.shape[0]
    dists = np.zeros(n, dtype=np.float32)
    for i in numba.prange(n):
        s = 0.0
        for j in range(coords.shape[1]):
            d = coords[i, j] - centroid[j]
            s += d*d
        dists[i] = np.sqrt(s)
    return np.mean(dists), np.std(dists), np.max(dists), dists

class TSP_Interpretable_Estimator:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        artifacts_dir = os.path.join(model_dir, 'model_artifacts')
        self.router = joblib.load(os.path.join(artifacts_dir, 'router.joblib'))
        self.experts = joblib.load(os.path.join(artifacts_dir, 'experts.joblib'))
        
        # Pre-fetch router features to ensure strict alignment during inference
        if hasattr(self.router, 'feature_names_in_'):
            self.router_features = self.router.feature_names_in_
        else:
            self.router_features = None
        
        with open(os.path.join(artifacts_dir, 'model_metadata.json'), 'r') as f:
            # Convert string keys back to int for leaf IDs
            self.metadata = {int(k): v for k, v in json.load(f).items()}
            
    def _compute_base_features(self, coords, n, d, grid_size):
        """Computes V3 features with Log-Space stability."""
        feats = {'n_customers': n, 'dimension': d}
        
        # Log-Space Stability for High Dimensions
        rngs = np.ptp(coords, axis=0).astype(float)
        rngs[rngs < 1e-9] = 1e-9
        log_hv = np.sum(np.log(rngs))
        
        # Store log versions explicitly as expected by the router/linear models
        feats['log_bounding_hypervolume'] = log_hv
        feats['log_node_density'] = np.log(n) - log_hv
        feats['aspect_ratio'] = np.max(rngs) / np.min(rngs)
        
        # Centroid Stats
        cent = np.mean(coords, axis=0, dtype=np.float32)
        c_mn, c_st, c_mx, c_raw = _fast_centroid_stats(coords, cent)
        feats['centroid_dist_mean'], feats['centroid_dist_std'] = c_mn, c_st
        feats['centroid_dist_max'] = c_mx
        feats['centroid_dist_iqr'] = np.subtract(*np.percentile(c_raw, [75, 25]))

        # MST Construction
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
        for i, p in enumerate([10, 25, 50, 75, 90]): feats[f'mst_edge_q{p}'] = percs[i]
        
        # Proxies
        k_dom = max(1, int(np.sqrt(n)))
        feats['mst_dominance_ratio'] = np.sum(np.partition(edges, -k_dom)[-k_dom:]) / (mst_len + 1e-9)
        feats['mst_gap_ratio'] = feats['mst_edge_max'] / (percs[2] + 1e-9)
        
        degs = np.zeros(n, dtype=int)
        rows, cols = mst.nonzero()
        for i in range(len(rows)): 
            degs[rows[i]] += 1
            degs[cols[i]] += 1
        
        feats['mst_leaf_ratio'] = np.sum(degs == 1) / n
        feats['mst_degree_mean'], feats['mst_degree_std'] = np.mean(degs), np.std(degs)
        feats['mst_degree_max'] = np.max(degs)
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
        
        # 1. Base Features
        f_dict, mst_len = self._compute_base_features(coords, len(coords), dimension, grid_size)
        
        # 2. Routing Setup
        df_base = pd.DataFrame([f_dict]).fillna(0)
        
        # Align columns to what the Router expects
        if self.router_features is not None:
            for col in self.router_features:
                if col not in df_base.columns:
                    df_base[col] = 0.0
            df_router_input = df_base[self.router_features]
        else:
            df_router_input = df_base

        # 3. Get Regime (Leaf ID)
        leaf_id = int(self.router.apply(df_router_input)[0])
        
        # 4. Expert Feature Preparation
        leaf_info = self.metadata.get(leaf_id)
        if not leaf_info:
            # Fallback if leaf logic fails
            t_feat = time.perf_counter() - t0
            return {'estimate': 1.5 * mst_len, 'alpha': 1.5, 'mst_length': mst_len, 
                    'feature_time': t_feat, 'inference_time': 0, 'regime_id': -1}

        # Construct interactions required by this specific leaf
        df_leaf = df_base.copy()
        interaction_cols = []
        
        for fa, fb in leaf_info['interactions']:
            col_name = f"{fa}_x_{fb}"
            val_a = df_base.get(fa, 0.0).item() if fa in df_base else 0.0
            val_b = df_base.get(fb, 0.0).item() if fb in df_base else 0.0
            df_leaf[col_name] = val_a * val_b
            interaction_cols.append(col_name)
            
        t_feat = time.perf_counter() - t0
        
        # 5. Inference
        t1 = time.perf_counter()
        expert = self.experts[leaf_id]
        
        # CRITICAL FIX: Filter DataFrame to ONLY the features this expert knows
        # Order doesn't strictly matter for sklearn DataFrames, but strict set matching does.
        required_features = leaf_info['base_features'] + interaction_cols
        df_expert_input = df_leaf[required_features]
        
        alpha = expert.predict(df_expert_input)[0]
        t_inf = time.perf_counter() - t1
        
        alpha = np.clip(alpha, 1.0, 2.0)
        
        return {
            'estimate': alpha * mst_len,
            'alpha': alpha,
            'mst_length': mst_len,
            'feature_time': t_feat,
            'inference_time': t_inf,
            'regime_id': leaf_id
        }

if __name__ == "__main__":
    print("=== Interpretable V3 Benchmark (N-Dim) ===")
    CUR = os.path.dirname(os.path.abspath(__file__))
    DATA_F = os.path.join(CUR, '..', 'tsp_features_v3.csv')
    INST_D = os.path.join(CUR, '..', 'instances')
    
    if not os.path.exists(DATA_F): exit(1)
    df_full = pd.read_csv(DATA_F)
    test_defs = df_full[df_full['split'] == 'test']
    
    est = TSP_Interpretable_Estimator(CUR)
    results = []
    
    for _, row in tqdm(test_defs.iterrows(), total=len(test_defs)):
        p = os.path.join(INST_D, f"{row['instance_name']}.json")
        if not os.path.exists(p): continue
        with open(p, 'r') as f: data = json.load(f)
        
        res = est.estimate(data['coordinates'], data['dimension'], data.get('grid_size', 1000))
        pe = (res['estimate'] - row['optimal_cost']) / row['optimal_cost']
        results.append({'opt': row['optimal_cost'], 'pred': res['estimate'], 'pe': pe, 
                        'ft': res['feature_time'], 'it': res['inference_time']})

    df_r = pd.DataFrame(results)
    print(f"\nMAPE: {df_r['pe'].abs().mean():.4%} | SDPE: {df_r['pe'].std():.4%}")
    print(f"Avg Time: {(df_r['ft'] + df_r['it']).mean()*1000:.3f} ms")