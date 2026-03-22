import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from collections import deque
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import numba
import warnings

# --- SILENCE ALL WARNINGS ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- CUSTOM SCALER DEFINITION ---
# This class must be defined here for the logic to work
class StableV3Scaler:
    def __init__(self):
        self.qt = QuantileTransformer(output_distribution='normal', random_state=42)
        self.ss = StandardScaler()

    def transform(self, X):
        X_qt = self.qt.transform(X)
        X_ss = self.ss.transform(X_qt)
        return np.clip(X_ss, -10.0, 10.0)

# --- ARCHITECTURE DEFINITIONS ---

class GatedResidualBlock_Inf(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.glu = nn.Sequential(nn.Linear(dim, dim * 2), nn.GLU(dim=-1))
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),   
            nn.GELU(),                 
            nn.Dropout(dropout),       
            nn.Linear(dim * 2, dim),   
            nn.Dropout(dropout * 0.5)  
        )
    def forward(self, x):
        return self.ffn(self.glu(self.norm(x))) + x

class TSP_Leap_Inference(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout=0.1):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([GatedResidualBlock_Inf(hidden_dim, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.GELU(), 
            nn.Linear(hidden_dim // 2, 1), 
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks: x = block(x)
        return self.head(self.final_norm(x))

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

# --- ESTIMATOR CLASS ---

class TSP_V3_Neural_Estimator:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- CRITICAL FIX: Namespace Injection ---
        # The scaler was originally pickled in a script run as __main__.
        # When loaded from another script (like run_benchmark_2D_all.py), 
        # pickle expects to find 'StableV3Scaler' in sys.modules['__main__'].
        # We manually inject it here to prevent the AttributeError.
        if not hasattr(sys.modules['__main__'], 'StableV3Scaler'):
            sys.modules['__main__'].StableV3Scaler = StableV3Scaler
        # -----------------------------------------

        # Load Scaler
        scaler_path = os.path.join(model_dir, 'nn_alpha_v3_scaler.joblib')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # Load Model
        model_path = os.path.join(model_dir, 'nn_alpha_v3_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location='cpu')
        self.features_required = checkpoint['features']
        
        bp = checkpoint['params']
        self.model = TSP_Leap_Inference(
            checkpoint['input_dim'], bp['hidden_dim'], bp['num_blocks'], dropout=bp.get('dropout', 0.1)
        )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _compute_v3_features(self, coords, n, d, grid_size):
        """High-stability V3 feature set computation."""
        feats = {'n_customers': n, 'dimension': d}
        rngs = np.ptp(coords, axis=0).astype(float); rngs[rngs < 1e-9] = 1e-9
        
        # --- STABLE HYPERVOLUME INTAKE ---
        log_hv = np.sum(np.log(rngs))
        hypervolume = np.exp(min(log_hv, 690.0)) 
        
        feats['bounding_hypervolume'] = hypervolume
        feats['node_density'] = n / hypervolume if hypervolume > 1e-15 else 0.0
        feats['aspect_ratio'] = np.max(rngs) / np.min(rngs)
        
        cent = np.mean(coords, axis=0, dtype=np.float32)
        c_mn, c_st, c_mx, c_raw = _fast_centroid_stats(coords, cent)
        feats['centroid_dist_mean'], feats['centroid_dist_std'], feats['centroid_dist_max'] = c_mn, c_st, c_mx
        feats['centroid_dist_iqr'] = np.subtract(*np.percentile(c_raw, [75, 25]))

        dist_mat = cdist(coords, coords, 'euclidean').astype(np.float32); np.fill_diagonal(dist_mat, 0)
        mst = minimum_spanning_tree(dist_mat); edges = mst.data; mst_len = np.sum(edges)
        feats['mst_total_length'] = mst_len
        feats['mst_edge_mean'], feats['mst_edge_std'] = np.mean(edges), np.std(edges)
        feats['mst_edge_skew'], feats['mst_edge_kurtosis'] = stats.skew(edges), stats.kurtosis(edges)
        feats['mst_edge_max'] = np.max(edges)
        percs = np.percentile(edges, [10, 25, 50, 75, 90])
        for i, p in enumerate([10, 25, 50, 75, 90]): feats[f'mst_edge_q{p}'] = percs[i]
        
        k_dom = max(1, int(np.sqrt(n)))
        feats['mst_dominance_ratio'] = np.sum(np.partition(edges, -k_dom)[-k_dom:]) / (mst_len + 1e-9)
        feats['mst_gap_ratio'] = feats['mst_edge_max'] / (percs[2] + 1e-9)
        
        degs = np.zeros(n); r, c = mst.nonzero()
        for i in range(len(r)): degs[r[i]]+=1; degs[c[i]]+=1
        feats['mst_leaf_ratio'] = np.sum(degs == 1) / n
        feats['mst_degree_mean'], feats['mst_degree_std'], feats['mst_degree_max'] = np.mean(degs), np.std(degs), np.max(degs)
        feats['large_edge_count'] = np.sum(edges > feats['mst_edge_mean'] + feats['mst_edge_std'])
        
        adj = [[] for _ in range(n)]
        for i in range(len(r)):
            adj[r[i]].append((c[i], edges[i])); adj[c[i]].append((r[i], edges[i]))
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
        """Inference with strict input truncation."""
        coords = np.array(coordinates, dtype=np.float32)
        t0 = time.perf_counter()
        f_dict, mst_len = self._compute_v3_features(coords, len(coords), dimension, grid_size)
        t_feat = time.perf_counter() - t0
        
        # Build vector and strictly cap to handle the '32 region' (float32 limit)
        vec = np.array([[f_dict[k] for k in self.features_required]])
        # Clean residual INF/NaNs from any other features
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e38, neginf=-1e38) 
        
        vec_tf = self.scaler.transform(vec)
        
        t1 = time.perf_counter()
        with torch.no_grad():
            x = torch.tensor(vec_tf, dtype=torch.float32).to(self.device)
            alpha_scaled = self.model(x).cpu().item()
        t_inf = time.perf_counter() - t1
        
        alpha = 1.0 + alpha_scaled
        return {
            'estimate': alpha * mst_len,
            'alpha': alpha,
            'mst_length': mst_len,
            'feature_time': t_feat,
            'inference_time': t_inf
        }

if __name__ == "__main__":
    print("=== Accurate V3 Neural Benchmark ===")
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_F = os.path.join(CUR_DIR, '..', 'tsp_features_v3.csv')
    INST_D = os.path.join(CUR_DIR, '..', 'instances')
    
    if not os.path.exists(DATA_F):
        print(f"Data file not found: {DATA_F}")
        exit(1)

    df_full = pd.read_csv(DATA_F)
    test_defs = df_full[df_full['split'] == 'test']
    if test_defs.empty: test_defs = df_full.sample(min(100, len(df_full)))

    estimator = TSP_V3_Neural_Estimator(CUR_DIR)
    results = []
    
    for _, row in tqdm(test_defs.iterrows(), total=len(test_defs)):
        p = os.path.join(INST_D, f"{row['instance_name']}.json")
        if not os.path.exists(p): continue
        with open(p, 'r') as f: data = json.load(f)
            
        res = estimator.estimate(data['coordinates'], data['dimension'], data.get('grid_size', 1000))
        pe = (res['estimate'] - row['optimal_cost']) / row['optimal_cost']
        results.append({'opt': row['optimal_cost'], 'pred': res['estimate'], 'pe': pe * 100, 
                        'feat_t': res['feature_time'], 'inf_t': res['inference_time']})

    df_r = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"Total Instances      : {len(df_r)}")
    print(f"MSE                  : {mean_squared_error(df_r['opt'], df_r['pred']):,.2f}")
    print(f"Avg % Error (MAPE)   : {df_r['pe'].abs().mean():.4f}%")
    print(f"SD of % Error (SDPE) : {df_r['pe'].std():.4f}%")
    print("-" * 50)
    print(f"Avg Feature Time     : {df_r['feat_t'].mean()*1000:.3f} ms")
    print(f"Avg Inference Time   : {df_r['inf_t'].mean()*1000:.3f} ms")
    print("="*50)