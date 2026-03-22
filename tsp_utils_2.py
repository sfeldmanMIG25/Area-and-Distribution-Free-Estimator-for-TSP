"""
TSP Academic Estimators Library (tsp_utils_2.py)

Contains a comprehensive set of TSP length estimators:
1. Exact (Held-Karp)
2. Constructive (Christofides, Hilbert N-D, MST-Ratio)
3. Geometric/Asymptotic (BHH, Chien, Cavdar, Vinel)
4. Simulation (2-Opt, EVT, Basel)
5. Machine Learning (GART 1.0)

Rules enforced:
- All functions operate on UNIQUE coordinates.
- All geometric functions use float casting to prevent integer overflow/zero-volume bugs.
- Chien estimator corrected for N-dimensional scaling.
- Hilbert estimator works in N-dimensions (requires 'hilbertcurve' pkg).
"""

import time
import math
import numpy as np
import networkx as nx
import pandas as pd
from math import inf
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import weibull_min
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

from hilbertcurve.hilbertcurve import HilbertCurve

# --- CONSTANTS ---
BETA_2D = 0.7124
BETA_3D = 0.6979

# ====================================================================
# SHARED HELPERS
# ====================================================================

def get_mst_length(nodes_coords):
    """Calculates MST length on UNIQUE coordinates."""
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    dist_matrix = cdist(coords, coords)
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = mst.sum()
    return mst_length, time.perf_counter() - start_time

def _run_2opt_fast(coords, n, max_iter=2000):
    """Fast 2-opt for simulation estimators."""
    tour = np.random.permutation(n)
    pts = coords[tour]
    cost = np.sum(np.sqrt(np.sum((pts[:-1] - pts[1:])**2, axis=1)))
    cost += np.linalg.norm(pts[-1] - pts[0])
    
    improved = True
    iter_count = 0
    limit = max(max_iter, n * 2)

    while improved and iter_count < limit:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n and i == 0: continue
                p1, p2 = tour[i], tour[i+1]
                p3, p4 = tour[j], tour[(j+1)%n]
                d_current = np.linalg.norm(coords[p1]-coords[p2]) + np.linalg.norm(coords[p3]-coords[p4])
                d_new = np.linalg.norm(coords[p1]-coords[p3]) + np.linalg.norm(coords[p2]-coords[p4])
                if d_new < d_current:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    cost += (d_new - d_current)
                    improved = True
                    break 
            if improved: break
    return cost

def _get_random_tour_len(coords, n):
    perm = np.random.permutation(n)
    pts = coords[perm]
    d = np.sum(np.sqrt(np.sum((pts[:-1] - pts[1:])**2, axis=1)))
    d += np.linalg.norm(pts[-1] - pts[0])
    return d

# ====================================================================
# EXACT ESTIMATORS
# ====================================================================

def estimate_tsp_held_karp(nodes_coords):
    """Exact TSP solver (DP) for small N."""
    start_time = time.perf_counter()
    unique_coords = np.unique(nodes_coords, axis=0)
    n = len(unique_coords)
    if n <= 1: return 0.0, 0.0
    if n == 2: return np.linalg.norm(unique_coords[0] - unique_coords[1]) * 2, time.perf_counter() - start_time
        
    dist_matrix = cdist(unique_coords, unique_coords)
    dp = [[inf] * n for _ in range(1 << n)]
    dp[1][0] = 0.0
    
    for r in range(2, n + 1):
        for subset in combinations(range(1, n), r - 1):
            mask = 1
            for bit in subset: mask |= 1 << bit
            for last in subset:
                prev = mask ^ (1 << last)
                best = inf
                temp = prev
                while temp:
                    bit = temp & -temp
                    i = bit.bit_length() - 1
                    temp ^= bit
                    cand = dp[prev][i] + dist_matrix[i][last]
                    if cand < best: best = cand
                dp[mask][last] = best
                
    full = (1 << n) - 1
    ans = inf
    for last in range(1, n):
        cand = dp[full][last] + dist_matrix[last][0]
        if cand < ans: ans = cand
    return ans, time.perf_counter() - start_time

# ====================================================================
# CONSTRUCTIVE ESTIMATORS
# ====================================================================

def estimate_tsp_christofides(nodes_coords):
    start_time = time.perf_counter()
    unique_coords = np.unique(nodes_coords, axis=0)
    n = len(unique_coords)
    if n <= 1: return 0.0, 0.0
    
    dist_matrix = cdist(unique_coords, unique_coords)
    mst_csr = minimum_spanning_tree(dist_matrix)
    mst_edges = zip(*mst_csr.nonzero())
    
    T = nx.Graph()
    T.add_nodes_from(range(n))
    for u, v in mst_edges:
        w = dist_matrix[u, v]
        T.add_edge(u, v, weight=w)
        
    odd_degree_nodes = [v for v, d in T.degree() if d % 2 == 1]
    subgraph = nx.Graph()
    k = len(odd_degree_nodes)
    for i in range(k):
        u = odd_degree_nodes[i]
        for j in range(i + 1, k):
            v = odd_degree_nodes[j]
            subgraph.add_edge(u, v, weight=dist_matrix[u, v])
            
    # Max weight matching on negated weights = Min weight matching
    for u, v, d in subgraph.edges(data=True): d['weight'] = -d['weight']
    matching = nx.max_weight_matching(subgraph, maxcardinality=True)
    
    M = nx.MultiGraph()
    M.add_nodes_from(range(n))
    M.add_edges_from(T.edges(data=True))
    for u, v in matching:
        w = dist_matrix[u, v]
        M.add_edge(u, v, weight=w)
        
    try: 
        eulerian_circuit = list(nx.eulerian_circuit(M, source=0))
    except nx.NetworkXError: 
        return 0.0, time.perf_counter() - start_time
        
    visited = [False] * n
    tour = []
    for u, v in eulerian_circuit:
        if not visited[u]:
            visited[u] = True
            tour.append(u)
    if not visited[eulerian_circuit[-1][1]]: tour.append(eulerian_circuit[-1][1])
    
    cost = 0.0
    for i in range(n):
        u, v = tour[i], tour[(i+1)%n]
        cost += dist_matrix[u, v]
    return cost, time.perf_counter() - start_time

def estimate_tsp_mst_ratio(nodes_coords):
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    d = coords.shape[1]
    dist_matrix = cdist(coords, coords)
    mst_len = minimum_spanning_tree(dist_matrix).sum()
    
    if d == 2: ratio = 1.075
    elif d == 3: ratio = 1.05
    else: ratio = 1.0 + (0.075 * (2.0/d))
    return mst_len * ratio, time.perf_counter() - start_time

def estimate_tsp_hilbert(nodes_coords, p=16):
    """
    Estimates TSP cost using an N-dimensional Hilbert Space Filling Curve.
    """
    start_time = time.perf_counter()

    # 1. Enforce Unique Coordinates (Standard Protocol)
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    # 2. Detect Dimensionality
    d = coords.shape[1]
    
    # 3. Scale Coordinates to Integer Grid [0, 2^p - 1]
    min_c = np.min(coords, axis=0)
    max_c = np.max(coords, axis=0)
    scale = max_c - min_c
    scale[scale < 1e-9] = 1.0 
    N_GRID = (1 << p) - 1
    norm_coords = (coords - min_c) / scale
    int_coords = (norm_coords * N_GRID).astype(int)
    
    # 4. Map N-D Coordinates to 1-D Hilbert Indices
    hc = HilbertCurve(p, d)
    points_list = int_coords.tolist()
    
    # --- ROBUST API CHECK ---
    # Prioritize 'distances_from_points' (current pip version)
    if hasattr(hc, 'distances_from_points'):
        hilbert_indices = hc.distances_from_points(points_list)
    elif hasattr(hc, 'distances_from_coordinates'):
        # v2.0 (Early)
        hilbert_indices = hc.distances_from_coordinates(points_list)
    elif hasattr(hc, 'distance_from_coordinates'):
        # v1.x (Singular) - might not support batch processing
        try:
            # Try batch first (some intermediate versions)
            hilbert_indices = hc.distance_from_coordinates(points_list)
        except (TypeError, ValueError, AttributeError):
            # Fallback to loop
            hilbert_indices = [hc.distance_from_coordinates(pt) for pt in points_list]
    else:
        raise AttributeError("Installed 'hilbertcurve' library has unknown API. Check version.")
    
    # 5. Sort Points by Hilbert Index
    sort_idx = np.argsort(hilbert_indices)
    sorted_coords = coords[sort_idx]
    
    # 6. Calculate Tour Length (Vectorized)
    deltas = sorted_coords[1:] - sorted_coords[:-1]
    tour_len = np.sum(np.sqrt(np.sum(deltas**2, axis=1)))
    
    # Close the loop
    tour_len += np.linalg.norm(sorted_coords[-1] - sorted_coords[0])
    
    return tour_len, time.perf_counter() - start_time

# ====================================================================
# SIMULATION ESTIMATORS
# ====================================================================

def estimate_tsp_evt(nodes_coords, samples=50):
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    local_optima = [_run_2opt_fast(coords, n) for _ in range(samples)]
    local_optima.sort()
    try:
        shape, loc, scale = weibull_min.fit(local_optima)
        estimated_min = loc
        if estimated_min < 0 or estimated_min > local_optima[0]: estimated_min = local_optima[0]
    except: estimated_min = local_optima[0]
    return estimated_min, time.perf_counter() - start_time

def estimate_tsp_2opt_distribution(nodes_coords, samples=20):
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    costs = [_run_2opt_fast(coords, n) for _ in range(samples)]
    mu = np.mean(costs)
    sigma = np.std(costs)
    if sigma < 1e-9: estimated_opt = mu
    else: estimated_opt = mu - 3.0 * sigma
    return estimated_opt, time.perf_counter() - start_time

def estimate_tsp_basel_willemain(nodes_coords):
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    BATCH_SIZE = 10000
    lengths = [_get_random_tour_len(coords, n) for _ in range(BATCH_SIZE)]
    sigma = np.std(lengths)
    if sigma < 1e-9: return 0.0, time.perf_counter() - start_time
    try:
        log_opt = 1.798 + 0.927 * np.log(sigma)
        est = np.exp(log_opt)
    except: est = 0.0
    return est, time.perf_counter() - start_time

# ====================================================================
# GEOMETRIC & ASYMPTOTIC ESTIMATORS
# ====================================================================

def estimate_tsp_chien(nodes_coords):
    """
    Implements Chien's Estimator with corrected N-dimensional scaling.
    Original (2D): L = 2D_avg + 0.57 * sqrt(n) * sqrt(A)
    Generalized:   L = 2D_avg + 0.57 * n^((d-1)/d) * V^(1/d)
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    d = coords.shape[1]
    
    # Float cast fix for hypervolume
    ranges = np.ptp(coords, axis=0).astype(float)
    ranges[ranges < 1e-9] = 1e-9 
    V = np.prod(ranges)
    
    try: geom_length_scale = math.pow(V, 1/d)
    except ValueError: geom_length_scale = 0.0
        
    centroid = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - centroid, axis=1)
    D_avg = np.mean(dists)
    
    # Corrected Node Scaling for d dimensions: n^((d-1)/d)
    exponent = (d - 1) / d
    n_scale = math.pow(n, exponent)
    
    est = 2 * D_avg + 0.57 * n_scale * geom_length_scale
    return est, time.perf_counter() - start_time

def estimate_tsp_bhh(nodes_coords):
    """
    Beardwood-Halton-Hammersley Theorem.
    L ~ beta * n^((d-1)/d) * V^(1/d)
    """
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    d = coords.shape[1]

    try:
        if n > d + 1:
            vol = ConvexHull(coords).volume
        else:
            raise Exception("Hull fail")
    except:
        ranges = np.ptp(coords, axis=0).astype(float)
        ranges[ranges < 1e-9] = 1e-9
        vol = np.prod(ranges)
        
    if d == 2: beta = BETA_2D
    elif d == 3: beta = BETA_3D
    else: beta = math.sqrt(d / (2 * math.pi * math.e))
    
    exponent = (d - 1) / d
    est = beta * (n ** exponent) * (vol ** (1/d))
    return est, time.perf_counter() - start_time

def estimate_tsp_vinel(nodes_coords, b=0.768):
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    d = coords.shape[1]
    
    try:
        if n > d + 1:
            vol = ConvexHull(coords).volume
        else:
            raise Exception("Hull fail")
    except:
        ranges = np.ptp(coords, axis=0).astype(float)
        ranges[ranges < 1e-9] = 1e-9
        vol = np.prod(ranges)   
        
    geom_len_scale = math.pow(vol, 1/d)
    n_scale = math.pow(n, (d - 1) / d)
    estimated_cost = b * n_scale * geom_len_scale
    return estimated_cost, time.perf_counter() - start_time

def estimate_tsp_cavdar(nodes_coords, a0=2.791, a1=0.2669):
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    d = coords.shape[1]
    
    try:
        if n > d + 1:
            vol = ConvexHull(coords).volume
        else:
            raise Exception("Hull fail")
    except:
        ranges = np.ptp(coords, axis=0).astype(float)
        ranges[ranges < 1e-9] = 1e-9
        vol = np.prod(ranges)
        
    mu = coords.mean(axis=0)
    stdev = coords.std(axis=0)
    abs_dev = np.abs(coords - mu)
    c_bar = abs_dev.mean(axis=0)
    c_bar[c_bar < 1e-9] = 1e-9
    cstdev = np.sqrt(np.mean((abs_dev - c_bar)**2, axis=0))
    
    prod_stdev = np.prod(stdev)
    prod_c_bar = np.prod(c_bar)
    prod_cstdev = np.prod(cstdev)
    
    n_scale = math.pow(n, (d - 1) / d)
    inv_d = 1.0 / d
    
    term1_geom = math.pow(prod_cstdev, inv_d)
    term1 = a0 * n_scale * term1_geom
    
    vol_scale = math.pow(vol, inv_d)
    dev_ratio = prod_stdev / prod_c_bar
    term2_geom = math.pow(dev_ratio, inv_d)
    term2 = a1 * n_scale * vol_scale * term2_geom
    
    estimated_cost = term1 + term2
    
    if n < 1000:
        corr = 0.9325 * math.exp(0.00005298 * n) - 0.2972 * math.exp(-0.01452 * n)
        if corr > 0.1: estimated_cost = estimated_cost / corr
            
    return estimated_cost, time.perf_counter() - start_time

def estimate_tsp_composite(nodes_coords):
    """Meta-estimator returning max(MST, min(2MST, Vinel/Cavdar))."""
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    if n <= 10:
        cost, _ = estimate_tsp_held_karp(coords)
        return cost, time.perf_counter() - start_time
    
    dist_matrix = cdist(coords, coords)
    mst_length = minimum_spanning_tree(dist_matrix).sum()
    
    if n < 100:
        est = estimate_tsp_vinel(coords)[0]
    else:
        est = estimate_tsp_cavdar(coords)[0]
        
    final_cost = max(mst_length, min(2 * mst_length, est))
    return final_cost, time.perf_counter() - start_time

# ====================================================================
# MACHINE LEARNING (GART 1.0)
# ====================================================================

def _calculate_gart_features(coords):
    n = len(coords)
    features = {'n': n}
    try:
        hull = ConvexHull(coords)
        features['convex_hull_area'] = hull.volume
        features['convex_hull_perimeter'] = hull.area
        features['hull_vertex_count'] = len(hull.vertices)
        features['hull_ratio'] = features['hull_vertex_count'] / n
    except:
        features.update({'convex_hull_area': 0, 'convex_hull_perimeter': 0, 'hull_vertex_count': 0, 'hull_ratio': 0})
    
    ranges = np.ptp(coords, axis=0).astype(float)
    features['bounding_box_area'] = np.prod(ranges)
    
    dist_matrix = cdist(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)
    one_nn = np.min(dist_matrix, axis=1)
    features['one_nn_dist_mean'] = one_nn.mean()
    features['one_nn_dist_std'] = one_nn.std()
    
    try:
        pca = PCA(n_components=2).fit(coords)
        ev = pca.explained_variance_
        features['pca_eigenvalue_ratio'] = ev[0] / ev[1] if ev[1] > 1e-9 else 1.0
    except:
        features['pca_eigenvalue_ratio'] = 1.0
        
    np.fill_diagonal(dist_matrix, 0)
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = mst.sum()
    degrees = np.count_nonzero(mst.toarray() + mst.toarray().T, axis=1)
    features['mst_degree_mean'] = degrees.mean()
    features['mst_degree_max'] = degrees.max()
    features['mst_degree_std'] = degrees.std()
    features['mst_leaf_nodes_fraction'] = np.sum(degrees == 1) / n
    
    features['coord_std_dev_x'] = coords[:, 0].std()
    features['coord_std_dev_y'] = coords[:, 1].std()
    depot = coords[0]
    dists_depot = np.linalg.norm(coords[1:] - depot, axis=1) if n > 1 else np.array([0.0])
    features['avg_dist_from_depot'] = dists_depot.mean()
    features['max_dist_from_depot'] = dists_depot.max()
    
    return features, mst_length

def estimate_tsp_ml_alpha(nodes_coords, ml_model):
    start_time = time.perf_counter()
    coords = np.unique(nodes_coords, axis=0)
    n = len(coords)
    if n <= 1: return 0.0, 0.0
    
    features_dict, mst_length = _calculate_gart_features(coords)
    t_feat = time.perf_counter() - start_time
    if mst_length == 0: return 0.0, time.perf_counter() - start_time
    
    try: feature_cols = ml_model.feature_name_
    except AttributeError: feature_cols = sorted(features_dict.keys()) 
    feature_df = pd.DataFrame([features_dict])
    if hasattr(ml_model, "feature_name_"): feature_df = feature_df[ml_model.feature_name_]

    predicted_alpha = ml_model.predict(feature_df)[0]
    est = predicted_alpha * mst_length
    final_cost = min(max(mst_length, est), 2*mst_length)
    t_inf = time.perf_counter() - start_time - t_feat
    return final_cost, t_feat, t_inf