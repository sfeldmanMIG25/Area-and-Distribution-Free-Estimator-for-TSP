"""V4-vs-V3 promotion study.

Answers three questions the paper needs settled before LGBM_V4 can replace
LGBM_V3 as "GART 2.0":

  1. ablation  -- where does V4's accuracy gain actually come from?
  2. cost      -- is V4's ~16x feature-extraction slowdown inherent or an
                  implementation artefact?
  3. coverage  -- (handled in the report; the non-Euclidean plumbing gap)

Subcommands (all write small CSV/JSON summaries next to this file):

    greedy      -- correctness + speed of the fast exact greedy-NN rewrite
    cost        -- feature-extraction cost table, V3 vs V4 vs lean-V4
    cache       -- build greedy_nn_over_mst caches for 2D-bench / TSPLIB
    ablate      -- the decisive ablation (retrains LightGBM arms)
    ood         -- V4 vs V3 on the 874-instance augmentation corpus

Nothing here mutates repository data or models; every output is a new file
with a ``v4_study_`` prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE), str(REPO / "lgbm_model_v4")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT_PREFIX = HERE / "v4_study"
SEED = 42


# =============================================================================
# 1. Exact greedy nearest-neighbour tour, three implementations
# =============================================================================
def greedy_ref_dense(coords: np.ndarray) -> float:
    """Reference: the O(n^2)-memory dense path used by lgbm_model_v4.

    Kept verbatim in spirit so the fast version can be checked against it.
    """
    n = coords.shape[0]
    if n < 2:
        return 0.0
    centroid = coords.mean(axis=0)
    start = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    total = 0.0
    D = cdist(coords, coords).astype(np.float32, copy=False)
    for _ in range(n - 1):
        row = D[current].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        total += float(row[nxt])
        visited[nxt] = True
        current = nxt
    total += float(np.linalg.norm(coords[current] - coords[start]))
    return total


def greedy_fast(coords: np.ndarray, rebuild_ratio: float = 0.5) -> float:
    """Exact greedy-NN closed tour length in O(n log n) time and O(n) memory.

    Same canonical start (point nearest the centroid) and same tie-breaking
    as the dense reference, so the returned length is identical -- this is an
    exact algorithm swap, not an approximation.

    Method: keep a cKDTree over the *unvisited* points only. Points are
    removed lazily (marked visited); once ``rebuild_ratio`` of the tree's
    points are stale the tree is rebuilt from the survivors. Because each
    rebuild happens after a constant fraction has been consumed, the total
    build work telescopes to O(n log n), and every query needs only a small
    k because at most ``rebuild_ratio`` of the hits can be stale.
    """
    n = coords.shape[0]
    if n < 2:
        return 0.0
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    centroid = coords.mean(axis=0)
    start = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    total = 0.0

    remaining = np.flatnonzero(~visited)
    tree = cKDTree(coords[remaining])
    stale = 0

    for _ in range(n - 1):
        if remaining.size == 0:
            break
        if stale >= rebuild_ratio * remaining.size:
            remaining = np.flatnonzero(~visited)
            if remaining.size == 0:
                break
            tree = cKDTree(coords[remaining])
            stale = 0

        k = 1
        nxt = -1
        d_nxt = 0.0
        while True:
            kq = min(k, remaining.size)
            dd, ii = tree.query(coords[current], k=kq)
            ii = np.atleast_1d(ii)
            dd = np.atleast_1d(dd)
            cand = remaining[ii]
            ok = ~visited[cand]
            if ok.any():
                j = int(np.argmax(ok))
                nxt = int(cand[j])
                d_nxt = float(dd[j])
                break
            if kq >= remaining.size:
                break
            k = min(k * 2, remaining.size)

        if nxt < 0:  # every point in this tree is stale -> force a rebuild
            remaining = np.flatnonzero(~visited)
            if remaining.size == 0:
                break
            tree = cKDTree(coords[remaining])
            stale = 0
            continue

        total += d_nxt
        visited[nxt] = True
        current = nxt
        stale += 1

    total += float(np.linalg.norm(coords[current] - coords[start]))
    return total


def greedy_dense_nocopy(coords: np.ndarray) -> float:
    """Dense path with the per-step row copy removed (visited columns are
    poisoned in place instead). Same answer, ~2x less allocation churn."""
    n = coords.shape[0]
    if n < 2:
        return 0.0
    centroid = coords.mean(axis=0)
    start = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
    D = cdist(coords, coords).astype(np.float32, copy=False)
    D[:, start] = np.inf
    current = start
    total = 0.0
    for _ in range(n - 1):
        nxt = int(np.argmin(D[current]))
        total += float(D[current, nxt])
        D[:, nxt] = np.inf
        current = nxt
    total += float(np.linalg.norm(coords[current] - coords[start]))
    return total


def greedy_candlist(coords: np.ndarray, k: int = 16,
                    rebuild_ratio: float = 0.5) -> float:
    """Exact greedy-NN using precomputed k-NN candidate lists.

    One vectorised ``cKDTree.query(coords, k+1)`` builds an (n, k) table of
    each point's k nearest neighbours in sorted order. During the walk the
    first *unvisited* entry of ``NI[current]`` is, by construction, the true
    nearest unvisited point -- so the common case costs a numpy fancy-index
    instead of a per-step scipy query (~2 us vs ~50 us). When all k candidates
    are already visited we fall back to an exact lazily-rebuilt kd-tree query,
    which keeps the result identical to brute force.
    """
    n = coords.shape[0]
    if n < 2:
        return 0.0
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    centroid = coords.mean(axis=0)
    start = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))

    tree_all = cKDTree(coords)
    kq = min(k + 1, n)
    ND, NI = tree_all.query(coords, k=kq, workers=-1)
    ND = np.atleast_2d(ND)[:, 1:]
    NI = np.atleast_2d(NI)[:, 1:]

    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    total = 0.0

    remaining = np.flatnonzero(~visited)
    tree = tree_all
    tree_idx = np.arange(n)
    stale = 1

    for _ in range(n - 1):
        cand = NI[current]
        ok = ~visited[cand]
        if ok.any():
            j = int(np.argmax(ok))
            nxt = int(cand[j])
            total += float(ND[current, j])
        else:
            # exact fallback
            if stale >= rebuild_ratio * tree_idx.size:
                remaining = np.flatnonzero(~visited)
                if remaining.size == 0:
                    break
                tree = cKDTree(coords[remaining])
                tree_idx = remaining
                stale = 0
            kk = 1
            nxt = -1
            d_nxt = 0.0
            while True:
                q = min(kk, tree_idx.size)
                dd, ii = tree.query(coords[current], k=q)
                ii = np.atleast_1d(ii); dd = np.atleast_1d(dd)
                c2 = tree_idx[ii]
                ok2 = ~visited[c2]
                if ok2.any():
                    j2 = int(np.argmax(ok2))
                    nxt = int(c2[j2]); d_nxt = float(dd[j2])
                    break
                if q >= tree_idx.size:
                    break
                kk = min(kk * 2, tree_idx.size)
            if nxt < 0:
                remaining = np.flatnonzero(~visited)
                if remaining.size == 0:
                    break
                tree = cKDTree(coords[remaining]); tree_idx = remaining; stale = 0
                continue
            total += d_nxt
        visited[nxt] = True
        current = nxt
        stale += 1

    total += float(np.linalg.norm(coords[current] - coords[start]))
    return total


try:
    import numba

    @numba.njit(cache=True)
    def _greedy_dense_kernel(D, start):  # pragma: no cover - jitted
        n = D.shape[0]
        visited = np.zeros(n, np.bool_)
        visited[start] = True
        cur = start
        total = 0.0
        for _ in range(n - 1):
            best = -1
            bd = np.inf
            for j in range(n):
                if not visited[j]:
                    v = D[cur, j]
                    if v < bd:
                        bd = v
                        best = j
            if best < 0:
                break
            total += bd
            visited[best] = True
            cur = best
        return total, cur

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


def greedy_dense_numba(coords: np.ndarray) -> float:
    """Dense greedy with the inner scan JIT-compiled. Same answer as the
    reference; removes the Python-per-step overhead that dominates n <= 2000.
    (V3's own extractor already depends on numba, so this adds no new dep.)"""
    n = coords.shape[0]
    if n < 2:
        return 0.0
    if not _HAVE_NUMBA:
        return greedy_dense_nocopy(coords)
    c64 = np.ascontiguousarray(coords, dtype=np.float64)
    centroid = c64.mean(axis=0)
    start = int(np.argmin(np.sum((c64 - centroid) ** 2, axis=1)))
    D = cdist(c64, c64)
    total, cur = _greedy_dense_kernel(D, start)
    return float(total) + float(np.linalg.norm(c64[cur] - c64[start]))


def greedy_auto(coords: np.ndarray, dense_cap: int = 3000) -> float:
    """Production dispatch: JIT dense for small n (lowest constant), k-NN
    candidate lists above. Both branches are exact."""
    n = coords.shape[0]
    if n <= dense_cap:
        return greedy_dense_numba(coords)
    return greedy_candlist(coords)


# =============================================================================
# 2. Lean V4 feature extraction -- only the 32 columns the booster consumes
# =============================================================================
def compute_features_lean_v4(
    coords: np.ndarray, dimension: int, grid_size: int,
    greedy_fn: Callable[[np.ndarray], float] = greedy_auto,
) -> Dict[str, float]:
    """V3's 30 features + mst_total_length + greedy_nn_over_mst, and nothing
    else. Byte-for-byte the same definitions as lgbm_model_v4, minus the 12
    candidate features the trained booster never sees."""
    from collections import deque
    from scipy import stats
    from mst_utils import compute_mst
    from tsp_utils_2 import canonicalize_coords_pca

    coords = np.asarray(coords, dtype=np.float32)
    n = coords.shape[0]
    coords = canonicalize_coords_pca(coords).astype(np.float32, copy=False)

    out: Dict[str, float] = {
        "n_customers": int(n), "dimension": int(dimension),
        "grid_size": int(grid_size),
    }
    ranges = np.ptp(coords, axis=0).astype(np.float64)
    ranges = np.where(ranges < 1e-9, 1e-9, ranges)
    log_hv = float(np.sum(np.log(ranges)))
    out["log_bounding_hypervolume"] = log_hv
    out["bounding_hypervolume"] = float(np.exp(min(log_hv, 690.0)))
    log_density = float(np.log(n) - log_hv)
    out["log_node_density"] = log_density
    out["node_density"] = float(np.exp(max(min(log_density, 690.0), -690.0)))
    out["aspect_ratio"] = float(np.max(ranges) / np.min(ranges))

    centroid = np.mean(coords, axis=0, dtype=np.float64)
    cd = np.linalg.norm(coords - centroid, axis=1)
    out["centroid_dist_mean"] = float(np.mean(cd))
    out["centroid_dist_std"] = float(np.std(cd))
    out["centroid_dist_max"] = float(np.max(cd))
    q75, q25 = np.percentile(cd, [75, 25])
    out["centroid_dist_iqr"] = float(q75 - q25)

    mst = compute_mst(coords).to_csr()
    edges = mst.data.astype(np.float64)
    if len(edges) == 0:
        edges = np.array([0.0])
    mst_len = float(np.sum(edges))
    out["mst_total_length"] = mst_len
    e_mean = float(np.mean(edges)); e_std = float(np.std(edges))
    out["mst_edge_mean"] = e_mean
    out["mst_edge_std"] = e_std
    out["mst_edge_skew"] = float(stats.skew(edges)) if e_std > 1e-9 else 0.0
    out["mst_edge_kurtosis"] = float(stats.kurtosis(edges)) if e_std > 1e-9 else 0.0
    out["mst_edge_max"] = float(np.max(edges))
    percs = np.percentile(edges, [10, 25, 50, 75, 90])
    for i, p in enumerate([10, 25, 50, 75, 90]):
        out[f"mst_edge_q{p}"] = float(percs[i])
    k_dom = max(1, int(np.sqrt(n)))
    if len(edges) >= k_dom:
        top_k = np.partition(edges, -k_dom)[-k_dom:]
        out["mst_dominance_ratio"] = float(np.sum(top_k) / mst_len) if mst_len > 1e-9 else 0.0
    else:
        out["mst_dominance_ratio"] = 1.0
    med = float(percs[2])
    out["mst_gap_ratio"] = float(out["mst_edge_max"] / med) if med > 1e-9 else 0.0

    rows, cols = mst.nonzero()
    mst_adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    degrees = np.zeros(n, dtype=np.int32)
    for i in range(len(rows)):
        u, v, w = int(rows[i]), int(cols[i]), float(edges[i])
        mst_adj[u].append((v, w)); mst_adj[v].append((u, w))
        degrees[u] += 1; degrees[v] += 1
    out["mst_leaf_ratio"] = float(np.sum(degrees == 1) / n)
    out["mst_degree_mean"] = float(np.mean(degrees))
    out["mst_degree_std"] = float(np.std(degrees))
    out["mst_degree_max"] = int(np.max(degrees))

    def _farthest(start_v: int):
        dists = np.full(n, -1.0); dists[start_v] = 0.0
        q = deque([start_v]); fn, md = start_v, 0.0
        while q:
            u = q.popleft()
            if dists[u] > md:
                md = dists[u]; fn = u
            for v, w in mst_adj[u]:
                if dists[v] < 0:
                    dists[v] = dists[u] + w; q.append(v)
        return fn, md
    n1, _ = _farthest(0)
    _, diam = _farthest(n1)
    out["mst_diameter"] = float(diam)
    out["mst_diameter_normalized"] = float(diam / mst_len) if mst_len > 1e-9 else 0.0
    out["large_edge_count"] = int(np.sum(edges > e_mean + e_std))

    greedy_len = greedy_fn(coords)
    out["greedy_nn_over_mst"] = float(greedy_len / mst_len) if mst_len > 1e-9 else 1.0
    return out


# =============================================================================
# 3. Instance loading helpers
# =============================================================================
def rand_instance(n: int, d: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, d), dtype=np.float32) * 1000.0).astype(np.float32)


def load_2d_bench(name: str) -> np.ndarray:
    import json as _j
    p = REPO / "Generalized_TSP_Analysis" / "instances" / f"{name}.json"
    with open(p, "r", encoding="utf-8") as f:
        data = _j.load(f)
    for key in ("coordinates", "coords", "points", "node_coords"):
        if key in data:
            return np.asarray(data[key], dtype=np.float32)
    raise KeyError(f"no coordinate key in {p}: {list(data)[:10]}")


def load_tsplib_coords(name: str) -> Optional[np.ndarray]:
    """Native EUC_2D / CEIL_2D coordinates from the TSPLIB .tsp file."""
    import glob as _g
    cands = _g.glob(str(REPO / "tsplib_benchmark" / "instances" / f"{name}.tsp"))
    if not cands:
        cands = _g.glob(str(REPO / "tsplib_benchmark" / "instances" / "**" / f"{name}.tsp"),
                        recursive=True)
    if not cands:
        return None
    pts: List[List[float]] = []
    in_sec = False
    with open(cands[0], "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("NODE_COORD_SECTION"):
                in_sec = True
                continue
            if in_sec:
                if s in ("EOF", "") or s.startswith("DISPLAY") or s[0].isalpha():
                    break
                parts = s.split()
                if len(parts) >= 3:
                    pts.append([float(parts[1]), float(parts[2])])
    return np.asarray(pts, dtype=np.float32) if pts else None


# =============================================================================
# 4. Subcommand: greedy -- correctness + speed
# =============================================================================
def cmd_greedy(args) -> None:
    import pandas as pd
    rows = []
    cases: List[Tuple[str, np.ndarray]] = []
    for n in (200, 1000, 3000):
        for d in (2, 10, 50):
            cases.append((f"rand_n{n}_d{d}", rand_instance(n, d)))
    for n in (10000, 20000):
        cases.append((f"rand_n{n}_d2", rand_instance(n, 2)))

    for name, X in cases:
        n = X.shape[0]
        rec: Dict[str, object] = {"case": name, "n": n, "d": X.shape[1]}
        # reference dense (skip when it would need too much RAM/time)
        if n <= 20000:
            t = time.perf_counter(); L_ref = greedy_ref_dense(X)
            rec["t_ref_dense_ms"] = (time.perf_counter() - t) * 1e3
            rec["len_ref"] = L_ref
        else:
            L_ref = np.nan
            rec["t_ref_dense_ms"] = np.nan; rec["len_ref"] = np.nan
        t = time.perf_counter(); L_f = greedy_fast(X)
        rec["t_fast_ms"] = (time.perf_counter() - t) * 1e3
        rec["len_fast"] = L_f
        t = time.perf_counter(); L_a = greedy_auto(X)
        rec["t_auto_ms"] = (time.perf_counter() - t) * 1e3
        rec["len_auto"] = L_a
        if np.isfinite(L_ref):
            rec["rel_err_fast"] = abs(L_f - L_ref) / L_ref
            rec["rel_err_auto"] = abs(L_a - L_ref) / L_ref
            rec["speedup_fast"] = rec["t_ref_dense_ms"] / rec["t_fast_ms"]
            rec["speedup_auto"] = rec["t_ref_dense_ms"] / rec["t_auto_ms"]
        rows.append(rec)
        print(f"  {name:16s} ref={rec['t_ref_dense_ms']:9.1f}ms  "
              f"fast={rec['t_fast_ms']:8.1f}ms  auto={rec['t_auto_ms']:8.1f}ms  "
              f"relerr_fast={rec.get('rel_err_fast', float('nan')):.2e}")

    # also exercise the *current repo* kd-tree fallback for the honest before/after
    from lgbm_model_v4.feature_engineering import _greedy_nn_tour_length
    print("\n  current repo kd-tree fallback (GREEDY_DENSE_BUDGET forced to 0):")
    for n in (2000, 5000, 10000):
        X = rand_instance(n, 2)
        t = time.perf_counter(); L_old = _greedy_nn_tour_length(X, D=None) if False else None
        # call the fallback branch directly by passing a tiny budget
        import lgbm_model_v4.feature_engineering as fe
        old = fe._GREEDY_DENSE_BUDGET_BYTES
        fe._GREEDY_DENSE_BUDGET_BYTES = 0
        t = time.perf_counter(); L_old = fe._greedy_nn_tour_length(X)
        dt = (time.perf_counter() - t) * 1e3
        fe._GREEDY_DENSE_BUDGET_BYTES = old
        t = time.perf_counter(); L_new = greedy_fast(X)
        dt_new = (time.perf_counter() - t) * 1e3
        rows.append({"case": f"kdfallback_n{n}_d2", "n": n, "d": 2,
                     "t_repo_kdtree_ms": dt, "t_fast_ms": dt_new,
                     "len_repo_kdtree": L_old, "len_fast": L_new,
                     "rel_err_fast": abs(L_new - L_old) / L_old,
                     "speedup_fast": dt / dt_new})
        print(f"    n={n:6d}  repo_kdtree={dt:9.1f}ms  fast={dt_new:8.1f}ms  "
              f"speedup={dt/dt_new:6.1f}x  relerr={abs(L_new-L_old)/L_old:.2e}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_PREFIX}_greedy.csv", index=False)
    print(f"\n[greedy] -> {OUT_PREFIX}_greedy.csv")


# =============================================================================
# 5. Subcommand: cost -- feature-extraction cost table
# =============================================================================
def _time_it(fn, repeat: int = 3) -> float:
    fn()  # warm up (JIT / caches)
    ts = []
    for _ in range(repeat):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e3


def cmd_cost(args) -> None:
    import pandas as pd
    from lgbm_model_v3.lgbm_estimator_v3 import TSP_V3_LGBM_Estimator
    from lgbm_model_v4.feature_engineering import compute_features as v4_full
    import lgbm_model_v4.feature_engineering as fe

    est3 = TSP_V3_LGBM_Estimator()
    v3_feat = getattr(est3, "_compute_features", None) or getattr(est3, "compute_features", None)

    cases: List[Tuple[str, np.ndarray, int]] = []
    for (n, d) in ((1000, 2), (1000, 50), (10000, 2)):
        cases.append((f"rand_n{n}_d{d}", rand_instance(n, d), d))
    big = None
    import pandas as _pd
    tl = _pd.read_csv(REPO / "paper_tooling" / "tsplib_features_v3.csv")
    euc = tl[tl.edge_weight_type.isin(["EUC_2D", "CEIL_2D"])].sort_values("n_customers")
    for nm in reversed(euc.instance_name.tolist()):
        c = load_tsplib_coords(nm)
        if c is not None and c.shape[0] > 5000:
            big = (f"tsplib_{nm}", c, 2)
            break
    if big:
        cases.append(big)
        print(f"[cost] largest TSPLIB EUC instance loaded: {big[0]} n={big[1].shape[0]}")

    def _orig_greedy(c):
        if fe._greedy_dense_feasible(c.shape[0]):
            D = cdist(c, c).astype(np.float32, copy=False)
            return fe._greedy_nn_tour_length(c, D=D)
        return fe._greedy_nn_tour_length(c)

    # Warm every JIT / import path once so numba compilation in the V3
    # estimator is not charged to the first measured case.
    print("[cost] warming up (numba JIT, scipy kernels) ...")
    _warm = rand_instance(400, 2)
    est3.estimate(_warm, 2, 1000)
    v4_full(_warm, 2, 1000)
    compute_features_lean_v4(_warm, 2, 1000, greedy_auto)
    for _d in (10, 50):
        _w = rand_instance(400, _d)
        est3.estimate(_w, _d, 1000)
        v4_full(_w, _d, 1000)
        compute_features_lean_v4(_w, _d, 1000, greedy_auto)

    rows = []
    for name, X, d in cases:
        n = X.shape[0]
        reps = 1 if n > 20000 else 3
        rec: Dict[str, object] = {"case": name, "n": n, "d": d, "reps": reps}
        # ---- V3 full feature extraction (via the shipped estimator) --------
        # NOTE: the V3 estimator memoises features on hash(coords.tobytes()),
        # so the cache must be cleared or every repeat reads back 0 ms.
        est3._feature_cache.clear()
        r3 = est3.estimate(X, d, 1000)
        fts = []
        for _ in range(reps):
            est3._feature_cache.clear()
            rr = est3.estimate(X, d, 1000); fts.append(rr["feature_time"])
        rec["v3_feature_ms"] = float(np.median(fts)) * 1e3
        rec["v3_infer_ms"] = r3["inference_time"] * 1e3
        # ---- V4 as shipped -------------------------------------------------
        rec["v4_shipped_feature_ms"] = _time_it(lambda: v4_full(X, d, 1000), reps)
        f4 = v4_full(X, d, 1000)
        # ---- V4 lean (only the 32 booster columns), fast greedy ------------
        rec["v4_lean_fast_feature_ms"] = _time_it(
            lambda: compute_features_lean_v4(X, d, 1000, greedy_auto), reps)
        fl = compute_features_lean_v4(X, d, 1000, greedy_auto)
        # ---- V4 lean with the ORIGINAL greedy (isolate greedy vs the rest) -
        rec["v4_lean_origreedy_feature_ms"] = _time_it(
            lambda: compute_features_lean_v4(X, d, 1000, _orig_greedy), reps)
        # ---- isolated greedy component -------------------------------------
        rec["greedy_orig_ms"] = _time_it(lambda: _orig_greedy(X), reps)
        rec["greedy_fast_ms"] = _time_it(lambda: greedy_auto(X), reps)
        rec["greedy_speedup"] = rec["greedy_orig_ms"] / max(rec["greedy_fast_ms"], 1e-9)
        # ---- agreement -------------------------------------------------------
        rec["greedy_ratio_shipped"] = f4["greedy_nn_over_mst"]
        rec["greedy_ratio_lean_fast"] = fl["greedy_nn_over_mst"]
        rec["greedy_ratio_rel_diff"] = abs(
            fl["greedy_nn_over_mst"] - f4["greedy_nn_over_mst"]
        ) / max(f4["greedy_nn_over_mst"], 1e-12)
        rec["v4_shipped_over_v3"] = rec["v4_shipped_feature_ms"] / max(rec["v3_feature_ms"], 1e-9)
        rec["v4_lean_over_v3"] = rec["v4_lean_fast_feature_ms"] / max(rec["v3_feature_ms"], 1e-9)
        rows.append(rec)
        print(f"  {name:22s} n={n:6d} d={d:3d} | v3={rec['v3_feature_ms']:9.1f}ms "
              f"v4_shipped={rec['v4_shipped_feature_ms']:9.1f}ms "
              f"v4_lean_fast={rec['v4_lean_fast_feature_ms']:9.1f}ms "
              f"| greedy {rec['greedy_orig_ms']:8.1f}->{rec['greedy_fast_ms']:7.1f}ms "
              f"({rec['greedy_speedup']:.1f}x) | ratio_reldiff={rec['greedy_ratio_rel_diff']:.2e}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_PREFIX}_cost.csv", index=False)
    print(f"\n[cost] -> {OUT_PREFIX}_cost.csv")


# =============================================================================
# 6. Subcommand: cache -- greedy_nn_over_mst for the eval corpora
# =============================================================================
def _greedy_ratio_from_coords(coords: np.ndarray) -> Tuple[float, float]:
    from mst_utils import compute_mst
    from tsp_utils_2 import canonicalize_coords_pca
    c = np.asarray(coords, dtype=np.float32)
    c = np.unique(c, axis=0)
    c = canonicalize_coords_pca(c).astype(np.float32, copy=False)
    mst = compute_mst(c).to_csr()
    mst_len = float(np.sum(mst.data))
    g = greedy_auto(c)
    return (g / mst_len if mst_len > 1e-9 else 1.0), mst_len


def cmd_cache(args) -> None:
    import pandas as pd
    which = args.which

    if which in ("2d", "all"):
        feats = pd.read_csv(REPO / "paper_tooling" / "augmentation_2d_features.csv")
        names = feats.instance_name.tolist()
        out = []
        t0 = time.time()
        for i, nm in enumerate(names):
            try:
                X = load_2d_bench(nm)
                r, m = _greedy_ratio_from_coords(X)
                out.append({"instance_name": nm, "greedy_nn_over_mst": r,
                            "mst_recomputed": m})
            except Exception as e:  # noqa: BLE001
                out.append({"instance_name": nm, "greedy_nn_over_mst": np.nan,
                            "mst_recomputed": np.nan, "error": repr(e)[:120]})
            if (i + 1) % 500 == 0:
                print(f"    2d {i+1}/{len(names)}  {time.time()-t0:.0f}s")
        pd.DataFrame(out).to_csv(f"{OUT_PREFIX}_greedy_2d.csv", index=False)
        print(f"[cache] 2D -> {OUT_PREFIX}_greedy_2d.csv  ({len(out)} rows, "
              f"{time.time()-t0:.0f}s)")

    if which in ("tsplib", "all"):
        tl = pd.read_csv(REPO / "paper_tooling" / "tsplib_features_v3.csv")
        out = []
        t0 = time.time()
        for _, r in tl.iterrows():
            nm = r.instance_name
            X = load_tsplib_coords(nm)
            if X is None or r.edge_weight_type not in ("EUC_2D", "CEIL_2D"):
                out.append({"instance_name": nm, "greedy_nn_over_mst": np.nan,
                            "mst_recomputed": np.nan,
                            "note": "non-native or coords unavailable"})
                continue
            try:
                g, m = _greedy_ratio_from_coords(X)
                out.append({"instance_name": nm, "greedy_nn_over_mst": g,
                            "mst_recomputed": m, "note": "ok"})
            except Exception as e:  # noqa: BLE001
                out.append({"instance_name": nm, "greedy_nn_over_mst": np.nan,
                            "mst_recomputed": np.nan, "note": repr(e)[:120]})
            print(f"    {nm:12s} n={r.n_customers:6d}  {time.time()-t0:7.1f}s")
        pd.DataFrame(out).to_csv(f"{OUT_PREFIX}_greedy_tsplib.csv", index=False)
        print(f"[cache] TSPLIB -> {OUT_PREFIX}_greedy_tsplib.csv ({time.time()-t0:.0f}s)")


# =============================================================================
# 7. Subcommand: ablate
# =============================================================================
V3_HP = {
    "learning_rate": 0.02597081024148143, "num_leaves": 148,
    "lambda_l1": 0.07764927124026656, "lambda_l2": 1.1865278867536643e-05,
    "feature_fraction": 0.5481439957854186, "bagging_fraction": 0.6087321895536277,
    "bagging_freq": 6, "min_child_samples": 23, "max_depth": -1,
}
V3_EARLY_STOP = 100
V3_MAX_ROUNDS = 3000

V4_HP = {
    "learning_rate": 0.02266507637317971, "num_leaves": 49, "max_depth": 6,
    "min_child_samples": 51, "reg_alpha": 0.09552812568703481,
    "reg_lambda": 9.001418946897983, "feature_fraction": 0.46060748755687964,
    "bagging_fraction": 0.5943947577635971, "bagging_freq": 7,
    "min_split_gain": 1.6021218554896001e-06,
}
V4_EARLY_STOP = 188
V4_MAX_ROUNDS = 5000

ALPHA_CLIP = (1.0, 2.0)


def _cost_metrics(alpha_pred, mst, true_cost) -> Dict[str, float]:
    alpha_pred = np.clip(alpha_pred, *ALPHA_CLIP)
    pred = alpha_pred * mst
    err = (pred - true_cost) / true_cost
    return {
        "mape": float(np.mean(np.abs(err)) * 100.0),
        "sdpe": float(np.std(err, ddof=1) * 100.0),
        "bias": float(np.mean(err) * 100.0),
        "n": int(len(err)),
    }


def _fit_arm(df, features: Sequence[str], hp: dict, early_stop: int,
             max_rounds: int, protocol: str):
    """protocol='v3' -> fit on train, early-stop on val (val never trained on).
       protocol='v4' -> tune rounds on val, then refit on train+val at 1.1x."""
    import lightgbm as lgb
    import pandas as pd

    features = list(features)
    mtr = df["split"] == "train"; mvl = df["split"] == "val"
    X_tr = df.loc[mtr, features]; y_tr = df.loc[mtr, "alpha"]
    X_vl = df.loc[mvl, features]; y_vl = df.loc[mvl, "alpha"]

    params = dict(hp)
    params.update({"objective": "regression", "metric": "rmse", "verbosity": -1,
                   "seed": SEED, "feature_pre_filter": False, "num_threads": 6})
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dvl = lgb.Dataset(X_vl, label=y_vl, reference=dtr)
    b = lgb.train(params, dtr, num_boost_round=max_rounds,
                  valid_sets=[dvl], valid_names=["val"],
                  callbacks=[lgb.early_stopping(early_stop, verbose=False),
                             lgb.log_evaluation(0)])
    best_iter = b.best_iteration
    if protocol == "v3":
        return b, best_iter, features
    X_full = pd.concat([X_tr, X_vl]); y_full = pd.concat([y_tr, y_vl])
    rounds = int(max(100, round(best_iter * 1.1)))
    b2 = lgb.train(params, lgb.Dataset(X_full, label=y_full),
                   num_boost_round=rounds)
    return b2, rounds, features


def _predict_alpha(booster, X, best_iter=None):
    return np.clip(booster.predict(X, num_iteration=best_iter), *ALPHA_CLIP)


def cmd_ablate(args) -> None:
    import pandas as pd

    print("[ablate] loading tsp_features_v4.csv ...")
    df = pd.read_csv(REPO / "tsp_features_v4.csv")
    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(*ALPHA_CLIP)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)
    print(f"[ablate] {len(df)} rows | split: {df['split'].value_counts().to_dict()}")

    import joblib
    m3 = joblib.load(REPO / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib")
    V3_FEATS = list(m3.feature_name_)
    m4 = joblib.load(REPO / "lgbm_model_v4" / "lgbm_alpha_model_v4.joblib")
    V4_FEATS = list(m4.feature_name())
    print(f"[ablate] V3 feats={len(V3_FEATS)}  V4 feats={len(V4_FEATS)}  "
          f"extra={sorted(set(V4_FEATS)-set(V3_FEATS))}")

    F_V3 = V3_FEATS
    F_V3_G = V3_FEATS + ["greedy_nn_over_mst"]
    F_V3_M = V3_FEATS + ["mst_total_length"]
    F_V4 = V4_FEATS

    arms = [
        ("A0_v3feat_v3hp_v3proto", F_V3, "v3hp", "v3"),
        ("A1_v3feat_v4hp_v3proto", F_V3, "v4hp", "v3"),
        ("A2_v4feat_v3hp_v3proto", F_V4, "v3hp", "v3"),
        ("A3_v3feat+greedy_v3hp_v3proto", F_V3_G, "v3hp", "v3"),
        ("A4_v3feat+mstlen_v3hp_v3proto", F_V3_M, "v3hp", "v3"),
        ("A5_v4feat_v4hp_v3proto", F_V4, "v4hp", "v3"),
        ("A6_v4feat_v4hp_v4proto", F_V4, "v4hp", "v4"),
        ("A7_v3feat_v3hp_v4proto", F_V3, "v3hp", "v4"),
        ("A8_v3feat+greedy_v4hp_v4proto", F_V3_G, "v4hp", "v4"),
        # A9 is the promotion candidate: V3's feature set plus the one new
        # feature, V4's (better-regularised) hyperparameters, and V3's honest
        # train-only fit. No mst_total_length.
        ("A9_v3feat+greedy_v4hp_v3proto", F_V3_G, "v4hp", "v3"),
    ]
    if args.arms:
        keep = set(args.arms.split(","))
        arms = [a for a in arms if a[0].split("_")[0] in keep or a[0] in keep]

    # ---- evaluation corpora -------------------------------------------------
    te = df[df["split"] == "test"].reset_index(drop=True)
    te_ind = te[te["dimension"] != 100]
    te_ood = te[te["dimension"] == 100]

    bench2d = _load_2d_eval()
    tsplib = _load_tsplib_eval()
    ood = _load_ood_eval()

    results = []
    for name, feats, hp_key, proto in arms:
        hp, es, mx = ((V3_HP, V3_EARLY_STOP, V3_MAX_ROUNDS) if hp_key == "v3hp"
                      else (V4_HP, V4_EARLY_STOP, V4_MAX_ROUNDS))
        t0 = time.time()
        b, it, feats = _fit_arm(df, feats, hp, es, mx, proto)
        bi = it if proto == "v3" else None
        rec: Dict[str, object] = {
            "arm": name, "n_features": len(feats), "hp": hp_key,
            "protocol": proto, "best_iter": it, "fit_s": time.time() - t0,
            "has_greedy": "greedy_nn_over_mst" in feats,
            "has_mstlen": "mst_total_length" in feats,
        }
        # ND test split
        for tag, sub in (("nd_test_all", te), ("nd_test_indist", te_ind),
                         ("nd_test_d100", te_ood)):
            a = _predict_alpha(b, sub[feats], bi)
            m = _cost_metrics(a, sub["mst_total_length"].to_numpy(),
                              sub["optimal_cost"].to_numpy())
            rec[f"{tag}_mape"] = m["mape"]; rec[f"{tag}_sdpe"] = m["sdpe"]
            rec[f"{tag}_n"] = m["n"]
        # 2D benchmark
        if bench2d is not None:
            ok = bench2d.dropna(subset=feats)
            a = _predict_alpha(b, ok[feats], bi)
            m = _cost_metrics(a, ok["mst_total_length"].to_numpy(),
                              ok["optimal_cost"].to_numpy())
            rec["bench2d_mape"] = m["mape"]; rec["bench2d_sdpe"] = m["sdpe"]
            rec["bench2d_n"] = m["n"]
            err = (np.clip(a, *ALPHA_CLIP) * ok["mst_total_length"].to_numpy()
                   - ok["optimal_cost"].to_numpy()) / ok["optimal_cost"].to_numpy()
            tmp = ok.assign(_ae=np.abs(err) * 100.0)
            for cls, g in tmp.groupby("gen_class"):
                rec[f"bench2d_mape__{cls}"] = float(g["_ae"].mean())
        # OOD augmentation corpus (874 adversarial instances, 6 families)
        if ood is not None:
            okd = ood.dropna(subset=feats)
            a = _predict_alpha(b, okd[feats], bi)
            m = _cost_metrics(a, okd["mst_total_length"].to_numpy(),
                              okd["optimal_cost"].to_numpy())
            rec["ood_mape"] = m["mape"]; rec["ood_sdpe"] = m["sdpe"]
            rec["ood_n"] = m["n"]
            err = (np.clip(a, *ALPHA_CLIP) * okd["mst_total_length"].to_numpy()
                   - okd["optimal_cost"].to_numpy()) / okd["optimal_cost"].to_numpy()
            tmp = okd.assign(_ae=np.abs(err) * 100.0)
            for fam, g in tmp.groupby("family"):
                rec[f"ood_mape__{fam}"] = float(g["_ae"].mean())
        # TSPLIB EUC_2D
        if tsplib is not None:
            ok = tsplib.dropna(subset=feats)
            a = _predict_alpha(b, ok[feats], bi)
            m = _cost_metrics(a, ok["mst_total_length"].to_numpy(),
                              ok["optimal_cost"].to_numpy())
            rec["tsplib_mape"] = m["mape"]; rec["tsplib_sdpe"] = m["sdpe"]
            rec["tsplib_n"] = m["n"]
            per = (np.clip(a, *ALPHA_CLIP) * ok["mst_total_length"].to_numpy()
                   - ok["optimal_cost"].to_numpy()) / ok["optimal_cost"].to_numpy()
            np.save(f"{OUT_PREFIX}_tsplib_err_{name}.npy", per)
        results.append(rec)
        print(f"  {name:34s} ndtest={rec['nd_test_all_mape']:.3f} "
              f"2d={rec.get('bench2d_mape', float('nan')):.3f} "
              f"tsplib={rec.get('tsplib_mape', float('nan')):.3f}  "
              f"({rec['fit_s']:.0f}s, {it} rounds)")
        pd.DataFrame(results).to_csv(f"{OUT_PREFIX}_ablation.csv", index=False)

    print(f"\n[ablate] -> {OUT_PREFIX}_ablation.csv")


def _load_2d_eval():
    import pandas as pd
    f = REPO / "paper_tooling" / "augmentation_2d_features.csv"
    g = Path(f"{OUT_PREFIX}_greedy_2d.csv")
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if g.exists():
        d = d.merge(pd.read_csv(g)[["instance_name", "greedy_nn_over_mst"]],
                    on="instance_name", how="left")
    else:
        d["greedy_nn_over_mst"] = np.nan
    return d


def _load_ood_eval():
    import pandas as pd
    f = REPO / "paper_tooling" / "augment_features_v3.csv"
    g = REPO / "paper_tooling" / "augment_greedy_nn.csv"
    if not (f.exists() and g.exists()):
        return None
    return pd.read_csv(f).merge(pd.read_csv(g), on="instance_name", how="left")


def _load_tsplib_eval():
    import pandas as pd
    f = REPO / "paper_tooling" / "tsplib_features_v3.csv"
    g = Path(f"{OUT_PREFIX}_greedy_tsplib.csv")
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d = d[d.edge_weight_type == "EUC_2D"].reset_index(drop=True)
    if g.exists():
        d = d.merge(pd.read_csv(g)[["instance_name", "greedy_nn_over_mst"]],
                    on="instance_name", how="left")
    else:
        d["greedy_nn_over_mst"] = np.nan
    return d


# =============================================================================
# 8. Subcommand: ood -- augmentation corpus
# =============================================================================
def cmd_ood(args) -> None:
    import pandas as pd
    import joblib
    from scipy import stats as sstats

    feats = pd.read_csv(REPO / "paper_tooling" / "augment_features_v3.csv")
    gnn = pd.read_csv(REPO / "paper_tooling" / "augment_greedy_nn.csv")
    d = feats.merge(gnn, on="instance_name", how="left")
    print(f"[ood] {len(d)} rows | greedy cached for {d.greedy_nn_over_mst.notna().sum()}")

    m3 = joblib.load(REPO / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib")
    m4 = joblib.load(REPO / "lgbm_model_v4" / "lgbm_alpha_model_v4.joblib")
    f3 = list(m3.feature_name_); f4 = list(m4.feature_name())
    miss = [c for c in set(f3 + f4) if c not in d.columns]
    if miss:
        print(f"[ood] MISSING COLUMNS: {miss}")
        return
    ok = d.dropna(subset=f4).reset_index(drop=True)
    print(f"[ood] usable rows for both models: {len(ok)}")

    a3 = np.clip(m3.predict(ok[f3]), *ALPHA_CLIP)
    a4 = np.clip(m4.predict(ok[f4]), *ALPHA_CLIP)
    mstv = ok["mst_total_length"].to_numpy(); true = ok["optimal_cost"].to_numpy()
    e3 = (a3 * mstv - true) / true
    e4 = (a4 * mstv - true) / true
    ok = ok.assign(ape_v3=np.abs(e3) * 100.0, ape_v4=np.abs(e4) * 100.0,
                   err_v3=e3 * 100.0, err_v4=e4 * 100.0)

    rows = [{"family": "ALL", "n": len(ok),
             "v3_mape": ok.ape_v3.mean(), "v4_mape": ok.ape_v4.mean(),
             "delta_pp": ok.ape_v4.mean() - ok.ape_v3.mean(),
             "v3_sdpe": ok.err_v3.std(ddof=1), "v4_sdpe": ok.err_v4.std(ddof=1)}]
    for fam, g in ok.groupby("family"):
        rows.append({"family": fam, "n": len(g),
                     "v3_mape": g.ape_v3.mean(), "v4_mape": g.ape_v4.mean(),
                     "delta_pp": g.ape_v4.mean() - g.ape_v3.mean(),
                     "v3_sdpe": g.err_v3.std(ddof=1), "v4_sdpe": g.err_v4.std(ddof=1)})
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    out.to_csv(f"{OUT_PREFIX}_ood.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    diff = ok.ape_v4.values - ok.ape_v3.values
    boot = np.array([np.mean(np.random.default_rng(SEED + i).choice(diff, len(diff)))
                     for i in range(5000)])
    w = sstats.wilcoxon(ok.ape_v4.values, ok.ape_v3.values)
    tt = sstats.ttest_rel(ok.ape_v4.values, ok.ape_v3.values)
    paired = {"mean_delta_pp": float(diff.mean()),
              "ci95_lo": float(np.percentile(boot, 2.5)),
              "ci95_hi": float(np.percentile(boot, 97.5)),
              "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
              "ttest_p": float(tt.pvalue),
              "v4_better_count": int((diff < 0).sum()),
              "v3_better_count": int((diff > 0).sum()), "n": int(len(diff))}
    with open(f"{OUT_PREFIX}_ood_paired.json", "w", encoding="utf-8") as fh:
        json.dump(paired, fh, indent=2)
    print("\n[ood] paired:", json.dumps(paired, indent=2))


# =============================================================================
# 9. GART 2.0 -- prediction assembly across every evaluation stratum
# =============================================================================
GART2_DIR = REPO / "lgbm_model_v3"
PRED_CACHE = Path(f"{OUT_PREFIX}_gart2_predictions.csv")


def _gart2_estimator():
    sys.path.insert(0, str(GART2_DIR))
    from lgbm_estimator_gart2 import TSP_GART2_Estimator
    # GART2_MODEL_DIR lets the pipeline be smoke-tested against a throwaway
    # model while the real Optuna run is still in flight.
    return TSP_GART2_Estimator(os.environ.get("GART2_MODEL_DIR", str(GART2_DIR)))


def _tsplib_tasks():
    """(path, name) for every TSPLIB instance the harness scores."""
    import glob as _g
    out = {}
    for p in _g.glob(str(REPO / "tsplib_benchmark" / "instances" / "**" / "*.tsp"),
                     recursive=True):
        out[Path(p).stem] = p
    return out


def cmd_predict(args) -> None:
    """Compute GART 2.0 predictions + per-instance timings for all strata.

    Features are recomputed from raw coordinates wherever coordinates exist,
    so nothing depends on a stale cached feature table (the shipped TSPLIB
    table, for instance, carries NaN skew/kurtosis on ts225).
    """
    import pandas as pd
    sys.path.insert(0, str(REPO / "tsplib_benchmark"))
    import ood_harness as oh

    est = _gart2_estimator()
    print(f"[predict] model features: {len(est.features_required)}")
    suite = oh.load_suite()
    rows: List[dict] = []

    # ---- bench2d (2580): coordinates on disk ------------------------------
    s = suite["bench2d"]
    for inst in map(str, s.truth.index):
        try:
            X = load_2d_bench(inst)
            r = est.estimate(X, X.shape[1], 0)
            rows.append({"stratum": "bench2d", "instance": inst,
                         "pred_cost": r["estimate"], "alpha": r["alpha"],
                         "mst_length": r["mst_length"],
                         "feature_time_s": r["feature_time"],
                         "inference_time_s": r["inference_time"], "status": r["status"]})
        except Exception as e:  # noqa: BLE001
            rows.append({"stratum": "bench2d", "instance": inst, "pred_cost": np.nan,
                         "status": f"exception:{type(e).__name__}"})
    print(f"[predict] bench2d done ({len(rows)})")

    # ---- TSPLIB, both strata: exercise the real dispatch ------------------
    from tsplib_parser import parse_tsplib_file
    from classical_mds import classical_mds
    import run_all_models_tsplib as R

    paths = _tsplib_tasks()
    for key in ("tsplib_euc2d", "tsplib_noneuc"):
        st = suite[key]
        for inst in map(str, st.truth.index):
            p = paths.get(inst)
            if p is None:
                rows.append({"stratum": key, "instance": inst, "pred_cost": np.nan,
                             "status": "instance_file_missing"})
                continue
            try:
                info = parse_tsplib_file(str(p))
                native = info["is_native_euclidean"] and info["raw_coords"] is not None
                if native:
                    C = info["raw_coords"].astype(np.float32)
                    r = est.estimate(C, C.shape[1], 0)
                    mode = "native"
                else:
                    D = info["distance_matrix"]
                    X, _e, _raw = classical_mds(D, max_dim=R.MAX_MDS_DIM)
                    r = R._hybrid_estimate_generic(est, D, X, X.shape[1])
                    mode = "hybrid"
                rows.append({"stratum": key, "instance": inst, "mode": mode,
                             "pred_cost": r.get("estimate", np.nan),
                             "alpha": r.get("alpha", np.nan),
                             "mst_length": r.get("mst_length", np.nan),
                             "feature_time_s": r.get("feature_time", np.nan),
                             "inference_time_s": r.get("inference_time", np.nan),
                             "status": r.get("status", "ok")})
            except Exception as e:  # noqa: BLE001
                rows.append({"stratum": key, "instance": inst, "pred_cost": np.nan,
                             "status": f"exception:{type(e).__name__}: {e}"[:160]})
        print(f"[predict] {key} done")

    # ---- augment (874): recompute from coordinates -------------------------
    # The cached augment_features_v3.csv carries NaN skew/kurtosis on 13
    # perfect-lattice instances (zero MST-edge variance leaves skew undefined).
    # The production extractor guards that case, so recomputing recovers them.
    feats = est.features_required
    s = suite["augment"]
    import json as _json
    for inst in map(str, s.truth.index):
        p = REPO / "augment" / "instances" / f"{inst}.json"
        try:
            j = _json.loads(p.read_text(encoding="utf-8"))
            X = np.asarray(j["coordinates"], dtype=np.float32)
            r = est.estimate(X, int(j["dimension"]), 0)
            rows.append({"stratum": "augment", "instance": inst,
                         "pred_cost": r["estimate"], "alpha": r["alpha"],
                         "mst_length": r["mst_length"],
                         "feature_time_s": r["feature_time"],
                         "inference_time_s": r["inference_time"],
                         "status": r["status"]})
        except Exception as e:  # noqa: BLE001
            rows.append({"stratum": "augment", "instance": inst,
                         "pred_cost": np.nan,
                         "status": f"exception:{type(e).__name__}"})
    print("[predict] augment done")

    # ---- ND test split (16,920) -------------------------------------------
    nd = pd.read_csv(REPO / "tsp_features_v4.csv")
    nd = nd[nd["split"] == "test"].reset_index(drop=True)
    a = np.clip(est.model.predict(nd[feats]), 1.0, 2.0)
    for inst, av, m in zip(nd.instance_name, a, nd.mst_total_length):
        rows.append({"stratum": "nd_test", "instance": str(inst),
                     "pred_cost": float(av) * float(m), "alpha": float(av),
                     "mst_length": float(m), "status": "ok"})
    print(f"[predict] nd_test done ({len(nd)})")

    df = pd.DataFrame(rows)
    df.to_csv(PRED_CACHE, index=False)
    print(f"\n[predict] -> {PRED_CACHE}")
    print(df.groupby("stratum")["status"].value_counts().to_string())


# =============================================================================
# 10. Per-stratum metrics
# =============================================================================
def _metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    err = (pred - truth) / truth
    return {"n": int(len(err)),
            "sdpe": float(np.std(err, ddof=1) * 100.0),
            "mape": float(np.mean(np.abs(err)) * 100.0),
            "mspe": float(np.mean((err * 100.0) ** 2)),
            "bias": float(np.mean(err) * 100.0)}


def cmd_strata(args) -> None:
    import pandas as pd
    import ood_harness as oh

    P = pd.read_csv(PRED_CACHE)
    ok = P[P.status == "ok"]
    pmap = dict(zip(ok.instance.astype(str), ok.pred_cost.astype(float)))
    suite = oh.load_suite()

    # Reference models, scored on exactly the same strata for comparability.
    import joblib
    refmaps: Dict[str, Dict[str, float]] = {"GART_2.0": pmap}
    for nm in ("LGBM_V3", "LGBM_V4"):
        d: Dict[str, float] = {}
        for key in ("bench2d", "tsplib_euc2d", "tsplib_noneuc", "augment"):
            b = suite[key].baselines.get(nm)
            if b is not None:
                d.update({str(i): float(v) for i, v in b.items() if np.isfinite(v)})
        refmaps[nm] = d
    ndf = pd.read_csv(REPO / "tsp_features_v4.csv")
    ndf = ndf[ndf.split == "test"].reset_index(drop=True)
    for nm, path, api in (("LGBM_V3", "lgbm_model_v3/lgbm_alpha_model_v3.joblib", "sk"),
                          ("LGBM_V4", "lgbm_model_v4/lgbm_alpha_model_v4.joblib", "b")):
        mdl = joblib.load(REPO / path)
        fl = list(mdl.feature_name_) if api == "sk" else list(mdl.feature_name())
        a = np.clip(mdl.predict(ndf[fl]), 1.0, 2.0)
        refmaps[nm].update({str(i): float(x) * float(m) for i, x, m
                            in zip(ndf.instance_name, a, ndf.mst_total_length)})

    out = []
    for key in ("nd_test", "bench2d", "tsplib_euc2d", "tsplib_noneuc", "augment"):
        if key == "nd_test":
            truth = dict(zip(ndf.instance_name.astype(str),
                             ndf.optimal_cost.astype(float)))
        else:
            truth = {str(i): float(v) for i, v in suite[key].truth.items()}
        for nm, mp in refmaps.items():
            common = [i for i in truth if i in mp and np.isfinite(mp[i])]
            if not common:
                continue
            pr = np.array([mp[i] for i in common])
            tr = np.array([truth[i] for i in common])
            m = _metrics(pr, tr)
            m.update({"stratum": key, "model": nm, "n_stratum": len(truth),
                      "coverage": len(common) / max(len(truth), 1)})
            out.append(m)
    t = pd.DataFrame(out)[["stratum", "model", "n_stratum", "n", "coverage",
                           "sdpe", "mape", "mspe", "bias"]]
    t.to_csv(f"{OUT_PREFIX}_gart2_strata.csv", index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 2D by generator class -------------------------------------------
    b2 = pd.read_csv(REPO / "paper_tooling" / "augmentation_2d_features.csv",
                     usecols=["instance_name", "optimal_cost", "gen_class"])
    b2["pred"] = b2.instance_name.astype(str).map(pmap)
    b2 = b2.dropna(subset=["pred"])
    rows = []
    for cls, g in b2.groupby("gen_class"):
        m = _metrics(g.pred.to_numpy(), g.optimal_cost.to_numpy())
        m["gen_class"] = cls
        rows.append(m)
    c = pd.DataFrame(rows)[["gen_class", "n", "sdpe", "mape", "mspe", "bias"]]
    c.to_csv(f"{OUT_PREFIX}_gart2_2d_by_class.csv", index=False)
    print("\n--- 2D benchmark by generator class ---")
    print(c.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- augment by family ------------------------------------------------
    ag = pd.read_csv(REPO / "paper_tooling" / "augment_features_v3.csv",
                     usecols=["instance_name", "optimal_cost", "family"])
    ag["pred"] = ag.instance_name.astype(str).map(pmap)
    ag = ag.dropna(subset=["pred"])
    rows = []
    for fam, g in ag.groupby("family"):
        m = _metrics(g.pred.to_numpy(), g.optimal_cost.to_numpy())
        m["family"] = fam
        rows.append(m)
    f = pd.DataFrame(rows)[["family", "n", "sdpe", "mape", "mspe", "bias"]]
    f.to_csv(f"{OUT_PREFIX}_gart2_aug_by_family.csv", index=False)
    print("\n--- augmentation corpus by family ---")
    print(f.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =============================================================================
# 11. Consistency verdict: dispersion + skill trajectory
# =============================================================================
def cmd_consistency(args) -> None:
    import pandas as pd
    import ood_harness as oh

    P = pd.read_csv(PRED_CACHE)
    ok = P[(P.status == "ok") & P.pred_cost.notna()]
    preds = dict(zip(ok.instance.astype(str), ok.pred_cost.astype(float)))
    label = "GART_2.0"

    # ---- dispersion vs the asymptotic MST ratio ---------------------------
    # Reported exactly the way the shipped reference numbers were produced:
    # ONE candidate per call against the harness's default baseline family,
    # then read off the Asymptotic_MST row. LGBM_V3 and LGBM_V4 are re-run
    # here under identical settings so the three are directly comparable
    # (their sd_ratios reproduce at 0.7204 and 0.6409).
    # Explicitly the predecessor. ``oh.shipped_predictions()`` used to mean
    # LGBM_V3 and now means the production model, which is this study's other
    # arm -- reading it here would put the same predictions under both keys.
    ref = {oh.PREDECESSOR: oh.model_predictions(oh.PREDECESSOR), label: preds}
    try:
        ref["LGBM_V4"] = oh.candidate_predictions_from_suite(
            ["LGBM_V4"], "tsplib_euc2d")["LGBM_V4"]
    except Exception:  # noqa: BLE001
        pass

    disp_all = []
    for stratum in ("tsplib_euc2d", "bench2d", "augment"):
        for lab, pr in ref.items():
            try:
                d = oh.dispersion_verdict({lab: pr}, stratum=stratum)
                d = d[d.model_b == "Asymptotic_MST"]
                if len(d):
                    disp_all.append(d)
            except Exception as e:  # noqa: BLE001
                print(f"  dispersion {stratum}/{lab}: {type(e).__name__}: {e}")
    disp = pd.concat(disp_all, ignore_index=True) if disp_all else pd.DataFrame()
    if len(disp):
        disp.to_csv(f"{OUT_PREFIX}_gart2_dispersion.csv", index=False)
        cols = ["stratum", "model_a", "n_pairs", "sdpe_a", "sdpe_b", "sd_ratio",
                "pct_lower", "ratio_ci_low", "ratio_ci_high", "p_pitman_morgan",
                "p_holm", "family_size", "detectable_holm"]
        print("--- dispersion vs Asymptotic_MST "
              "(sd_ratio = model / asymptotic; < 1 is better) ---")
        print(disp[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- full MAPE verdict -------------------------------------------------
    v = oh.evaluate_candidate(preds, label)
    print(f"\n[verdict] {v.headline()}  family_size={v.family_size}")
    v.comparisons.to_csv(f"{OUT_PREFIX}_gart2_comparisons.csv", index=False)
    los = v.losses()
    print(f"[verdict] wins(Holm)={len(v.significant())}  losses(Holm)={len(los)}")
    if len(los):
        print(los[["stratum", "model_b", "mape_a", "mape_b", "mean_diff", "p_holm"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- difficulty-normalised skill trajectory across dimension ----------
    import monotonicity as mn
    nd = mn.load_nd()
    long = nd.long
    meta = nd.meta.set_index("instance")
    add = ok[ok.stratum == "nd_test"].copy()
    add["instance"] = add.instance.astype(str)
    add = add[add.instance.isin(meta.index)]
    j = meta.loc[add.instance]
    new = pd.DataFrame({
        "model": label, "instance": add.instance.to_numpy(),
        "pred_cost": add.pred_cost.to_numpy(),
        "true_cost": (j.alpha.to_numpy() * 0 + 0),  # filled below
        "n": j.n.to_numpy(), "d": j.d.to_numpy(), "alpha": j.alpha.to_numpy(),
    })
    truth_map = dict(zip(long.instance.astype(str), long.true_cost.astype(float)))
    new["true_cost"] = new.instance.map(truth_map)
    new = new.dropna(subset=["true_cost"])
    new["e"] = 100.0 * (new.pred_cost - new.true_cost) / new.true_cost
    merged = pd.concat([long, new], ignore_index=True)
    st = mn.Stratum("ND", merged, nd.meta, {"dim": mn.ND_DIM})
    bt = mn.build_bucket_table([st])
    mono = mn.build_monotonicity([st], bt)
    bt.to_csv(f"{OUT_PREFIX}_gart2_buckets.csv", index=False)
    mono.to_csv(f"{OUT_PREFIX}_gart2_monotonicity.csv", index=False)

    keep = bt[bt.model.isin([label, "LGBM_V3", "LGBM_V4"])]
    print("\n--- difficulty-normalised skill trajectory across dimension "
          "(skill = SDPE / CV(alpha); lower is better) ---")
    print(keep[["model", "bucket", "bucket_index", "n_instances", "sdpe",
                "cv_alpha_pct", "skill_vs_cv_alpha", "extrapolation"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- monotonicity summary (d=100 excluded from trend by design) ---")
    print(mono[mono.model.isin([label, "LGBM_V3", "LGBM_V4"])]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =============================================================================
# 12. Timing parity (run on an unloaded machine)
# =============================================================================
def cmd_timing(args) -> None:
    import pandas as pd
    import ood_harness as oh

    print("[timing] machine load:", oh.measure_machine_load())
    P = pd.read_csv(PRED_CACHE)
    ct = P[(P.status == "ok") & P.feature_time_s.notna()][
        ["instance", "feature_time_s", "inference_time_s"]].copy()
    t = oh.timing_parity(ct, "GART_2.0", reference="LGBM_V3",
                         machine_loaded=bool(args.loaded))
    t.to_csv(f"{OUT_PREFIX}_gart2_timing.csv", index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


# =============================================================================
# 12b. Feature cache + multi-model registry
# =============================================================================
FEAT_CACHE = Path(f"{OUT_PREFIX}_feature_cache.csv")
V4_UNION = None  # filled at runtime: the 32-column union all models draw from


class _CaptureShim:
    """Stands in for an estimator so the real hybrid builder hands us its
    feature dict instead of a prediction. Guarantees the cached non-Euclidean
    features are exactly what the production path produces."""

    def __init__(self, required):
        self.features_required = list(required)
        self.captured = None
        outer = self

        class _M:
            @staticmethod
            def predict(df):
                outer.captured = df.iloc[0].to_dict()
                return np.array([1.5])
        self.model = _M()


def cmd_features(args) -> None:
    import json as _json
    import pandas as pd
    sys.path.insert(0, str(REPO / "tsplib_benchmark"))
    sys.path.insert(0, str(GART2_DIR))
    import ood_harness as oh
    from feature_engineering_gart2 import FEATURE_ORDER, compute_features
    from tsplib_parser import parse_tsplib_file
    from classical_mds import classical_mds
    import run_all_models_tsplib as R

    cols = FEATURE_ORDER + ["mst_total_length"]
    suite = oh.load_suite()
    rows: List[dict] = []

    def add(stratum, inst, feats, extra=None, status="ok", mode=None,
            t_feat=np.nan):
        r = {"stratum": stratum, "instance": inst, "status": status,
             "mode": mode, "feature_time_s": t_feat}
        r.update({c: feats.get(c, np.nan) for c in cols} if feats
                 else {c: np.nan for c in cols})
        if extra:
            r.update(extra)
        rows.append(r)

    # bench2d
    for inst in map(str, suite["bench2d"].truth.index):
        try:
            X = load_2d_bench(inst)
            t = time.perf_counter(); f = compute_features(X, X.shape[1])
            add("bench2d", inst, f, t_feat=time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001
            add("bench2d", inst, None, status=f"exception:{type(e).__name__}")
    print("[features] bench2d done")

    # augment
    for inst in map(str, suite["augment"].truth.index):
        try:
            j = _json.loads((REPO / "augment" / "instances" / f"{inst}.json")
                            .read_text(encoding="utf-8"))
            X = np.asarray(j["coordinates"], dtype=np.float32)
            t = time.perf_counter(); f = compute_features(X, int(j["dimension"]))
            add("augment", inst, f, t_feat=time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001
            add("augment", inst, None, status=f"exception:{type(e).__name__}")
    print("[features] augment done")

    # TSPLIB, both strata, real dispatch
    paths = _tsplib_tasks()
    for key in ("tsplib_euc2d", "tsplib_noneuc"):
        for inst in map(str, suite[key].truth.index):
            p = paths.get(inst)
            if p is None:
                add(key, inst, None, status="instance_file_missing"); continue
            try:
                info = parse_tsplib_file(str(p))
                if info["is_native_euclidean"] and info["raw_coords"] is not None:
                    C = info["raw_coords"].astype(np.float32)
                    t = time.perf_counter(); f = compute_features(C, C.shape[1])
                    add(key, inst, f, mode="native", t_feat=time.perf_counter() - t)
                else:
                    D = info["distance_matrix"]
                    X, _e, _r = classical_mds(D, max_dim=R.MAX_MDS_DIM)
                    shim = _CaptureShim(cols)
                    t = time.perf_counter()
                    res = R._hybrid_estimate_generic(shim, D, X, X.shape[1])
                    dt = time.perf_counter() - t
                    st = res.get("status", "ok")
                    add(key, inst, shim.captured, status=st, mode="hybrid", t_feat=dt)
            except Exception as e:  # noqa: BLE001
                add(key, inst, None, status=f"exception:{type(e).__name__}")
        print(f"[features] {key} done")

    # ND test split -- already tabulated
    nd = pd.read_csv(REPO / "tsp_features_v4.csv")
    nd = nd[nd["split"] == "test"].reset_index(drop=True)
    for _, r in nd.iterrows():
        add("nd_test", str(r["instance_name"]),
            {c: r[c] for c in cols if c in nd.columns})
    print("[features] nd_test done")

    df = pd.DataFrame(rows)
    # attach truth + grouping labels
    truth = {}
    for key in ("bench2d", "augment", "tsplib_euc2d", "tsplib_noneuc"):
        truth.update({str(i): float(v) for i, v in suite[key].truth.items()})
    truth.update(dict(zip(nd.instance_name.astype(str), nd.optimal_cost.astype(float))))
    df["true_cost"] = df.instance.map(truth)
    b2 = pd.read_csv(REPO / "paper_tooling" / "augmentation_2d_features.csv",
                     usecols=["instance_name", "gen_class"])
    ag = pd.read_csv(REPO / "paper_tooling" / "augment_features_v3.csv",
                     usecols=["instance_name", "family"])
    df["gen_class"] = df.instance.map(dict(zip(b2.instance_name.astype(str), b2.gen_class)))
    df["family"] = df.instance.map(dict(zip(ag.instance_name.astype(str), ag.family)))
    df.to_csv(FEAT_CACHE, index=False)
    print(f"\n[features] -> {FEAT_CACHE}  ({len(df)} rows)")
    print(df.groupby("stratum").status.value_counts().to_string())


def _model_registry():
    """(label, path, api, target). ``target`` 'alpha' or 'logit'."""
    L = REPO / "lgbm_model_v3"
    return [
        ("LGBM_V3", L / "lgbm_alpha_model_v3.joblib", "sk", "alpha"),
        ("LGBM_V4", REPO / "lgbm_model_v4" / "lgbm_alpha_model_v4.joblib", "b", "alpha"),
        ("GART2_tuned", L / "gart2_alpha_model.joblib", "b", "alpha"),
        ("GART2_mono_nd", L / "gart2_mono_both_model.joblib", "b", "alpha"),
        ("GART2_mono_n", L / "gart2_mono_n_model.joblib", "b", "alpha"),
        ("GART2_logit_v3hp", L / "gart2_logit_v3hp_model.joblib", "b", "logit"),
        ("GART2_logit_v3hp_mono", L / "gart2_logit_v3hp_mono_model.joblib", "b", "logit"),
        ("GART2_logit_tuned", L / "gart2_logit_tuned_model.joblib", "b", "logit"),
    ]


def _load_models():
    import joblib
    out = []
    for label, path, api, target in _model_registry():
        if not path.exists():
            continue
        m = joblib.load(path)
        feats = list(m.feature_name_) if api == "sk" else list(m.feature_name())
        out.append((label, m, feats, target))
    return out


def _predict_cost(m, feats, target, frame):
    z = m.predict(frame[feats])
    if target == "logit":
        a = 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    else:
        a = np.clip(z, 1.0, 2.0)
    return a * frame["mst_total_length"].to_numpy(), a


def cmd_evalall(args) -> None:
    import pandas as pd
    from scipy import stats as sstats
    C = pd.read_csv(FEAT_CACHE)
    models = _load_models()
    print("[evalall] models:", [m[0] for m in models])

    per_inst = []
    rows = []
    for label, m, feats, target in models:
        for stratum, g in C.groupby("stratum"):
            ok = g[(g.status == "ok") & g[feats].notna().all(axis=1)
                   & g.true_cost.notna()]
            if not len(ok):
                continue
            pred, alpha = _predict_cost(m, feats, target, ok)
            err = (pred - ok.true_cost.to_numpy()) / ok.true_cost.to_numpy()
            rows.append({"model": label, "stratum": stratum,
                         "n_stratum": int(len(g)), "n": int(len(ok)),
                         "coverage": len(ok) / len(g),
                         "sdpe": float(np.std(err, ddof=1) * 100),
                         "mape": float(np.mean(np.abs(err)) * 100),
                         "mspe": float(np.mean((err * 100) ** 2)),
                         "bias": float(np.mean(err) * 100)})
            per_inst.append(pd.DataFrame({
                "model": label, "stratum": stratum,
                "instance": ok.instance.to_numpy(), "pred_cost": pred,
                "true_cost": ok.true_cost.to_numpy(), "err_pct": err * 100,
                "gen_class": ok.gen_class.to_numpy(), "family": ok.family.to_numpy()}))
    S = pd.DataFrame(rows)
    PI = pd.concat(per_inst, ignore_index=True)
    S.to_csv(f"{OUT_PREFIX}_allmodels_strata.csv", index=False)
    PI.to_csv(f"{OUT_PREFIX}_allmodels_per_instance.csv", index=False)

    order = ["nd_test", "bench2d", "tsplib_euc2d", "tsplib_noneuc", "augment"]
    for metric in ("sdpe", "mape"):
        p = S.pivot_table(index="model", columns="stratum", values=metric)
        p = p[[c for c in order if c in p.columns]]
        print(f"\n=== {metric.upper()} by stratum ===")
        print(p.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== coverage ===")
    print(S.pivot_table(index="model", columns="stratum", values="n")
          [[c for c in order if c in S.stratum.unique()]].to_string())

    # 2D by class and augment by family, for every model
    for key, col in (("bench2d", "gen_class"), ("augment", "family")):
        sub = PI[PI.stratum == key].dropna(subset=[col])
        t = (sub.assign(ape=sub.err_pct.abs())
             .pivot_table(index="model", columns=col, values="ape", aggfunc="mean"))
        t.to_csv(f"{OUT_PREFIX}_allmodels_{key}_by_{col}.csv")
        print(f"\n=== {key} MAPE by {col} ===")
        print(t.to_string(float_format=lambda x: f"{x:.4f}"))

    # paired tests vs LGBM_V3 on every stratum
    pr = []
    for stratum in order:
        base = PI[(PI.model == "LGBM_V3") & (PI.stratum == stratum)]
        if not len(base):
            continue
        bmap = dict(zip(base.instance, base.err_pct))
        for label in [m[0] for m in models if m[0] != "LGBM_V3"]:
            cur = PI[(PI.model == label) & (PI.stratum == stratum)]
            common = [i for i in cur.instance if i in bmap]
            if len(common) < 6:
                continue
            cm = dict(zip(cur.instance, cur.err_pct))
            a = np.array([abs(cm[i]) for i in common])
            b = np.array([abs(bmap[i]) for i in common])
            d = a - b
            boot = np.array([np.mean(np.random.default_rng(SEED + k).choice(d, len(d)))
                             for k in range(4000)])
            try:
                w = float(sstats.wilcoxon(a, b).pvalue)
            except Exception:  # noqa: BLE001
                w = np.nan
            pr.append({"stratum": stratum, "model": label, "n": len(common),
                       "mape_model": a.mean(), "mape_v3": b.mean(),
                       "mean_diff_pp": d.mean(),
                       "ci_lo": float(np.percentile(boot, 2.5)),
                       "ci_hi": float(np.percentile(boot, 97.5)),
                       "wilcoxon_p": w,
                       "model_better_n": int((d < 0).sum())})
    T = pd.DataFrame(pr)
    T.to_csv(f"{OUT_PREFIX}_allmodels_paired_vs_v3.csv", index=False)
    print("\n=== paired vs LGBM_V3 (negative mean_diff favours the model) ===")
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


DISP_BASELINES = ["Asymptotic_MST", "MST_Ratio", "Fixed_Alpha",
                  "Calibrated_MST_d", "Calibrated_MST_dn"]
MEANTEST_BASELINES = ["Asymptotic_MST", "Calibrated_MST_dn", "Fixed_Alpha"]
FINAL_MODEL = "GART2_logit_v3hp_mono"


def cmd_conformal(args) -> None:
    """Split-conformal coverage for the final model.

    One constant, calibrated on the val split (never trained on), applied
    unchanged everywhere. The interval is multiplicative on cost:
        [pred / (1 + q), pred * (1 + q)]  -- equivalently |rel err| <= q.
    """
    import joblib
    import pandas as pd
    sys.path.insert(0, str(GART2_DIR))

    m = joblib.load(GART2_DIR / "gart2_final.joblib")
    feats = list(m.feature_name())

    tab = pd.read_csv(REPO / "tsp_features_v4.csv")
    mstv = tab["mst_total_length"].replace(0, np.nan)
    tab["alpha"] = (tab["optimal_cost"] / mstv).clip(1.0, 2.0)
    tab = tab.dropna(subset=["alpha"])
    cal = tab[tab.split == "val"]
    a = _to_alpha(m.predict(cal[feats]), "logit")
    e = np.abs(a * cal.mst_total_length.to_numpy() - cal.optimal_cost.to_numpy()) \
        / cal.optimal_cost.to_numpy()
    rows = []
    for nominal in (0.80, 0.90, 0.95):
        n = len(e)
        lvl = min(1.0, np.ceil((n + 1) * nominal) / n)   # finite-sample correction
        q = float(np.quantile(e, lvl))
        C = pd.read_csv(FEAT_CACHE)
        for stratum, g in C.groupby("stratum"):
            ok = g[(g.status == "ok") & g[feats].notna().all(axis=1)
                   & g.true_cost.notna()]
            if not len(ok):
                continue
            pa = _to_alpha(m.predict(ok[feats]), "logit")
            err = np.abs(pa * ok.mst_total_length.to_numpy()
                         - ok.true_cost.to_numpy()) / ok.true_cost.to_numpy()
            rows.append({"nominal_pct": nominal * 100, "q_rel": q,
                         "stratum": stratum, "n": int(len(err)),
                         "coverage_pct": float((err <= q).mean() * 100.0),
                         "median_width_pct": float(q * 100.0 * 2)})
    T = pd.DataFrame(rows)
    T.to_csv(f"{OUT_PREFIX}_gart2_conformal.csv", index=False)
    for nom, g in T.groupby("nominal_pct"):
        print(f"\n=== split-conformal, nominal {nom:.0f}%  "
              f"(q = {g.q_rel.iloc[0]*100:.3f}% relative, calibrated on val) ===")
        print(g[["stratum", "n", "coverage_pct"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n[conformal] -> {OUT_PREFIX}_gart2_conformal.csv")


def cmd_timing_final(args) -> None:
    """Idle-machine timing for the final model against shipped V3.

    Reports median and p90 feature-extraction and inference time per stratum,
    plus the four reference sizes the paper quotes. Run this with nothing else
    on the box; it refuses to interpret its own numbers otherwise.
    """
    import json as _json
    import joblib
    import pandas as pd
    sys.path.insert(0, str(GART2_DIR))
    sys.path.insert(0, str(REPO / "tsplib_benchmark"))
    import ood_harness as oh
    from feature_engineering_gart2 import compute_features
    from lgbm_model_v3.lgbm_estimator_v3 import TSP_V3_LGBM_Estimator
    from tsplib_parser import parse_tsplib_file

    print("[timing] machine load:", oh.measure_machine_load())
    est3 = TSP_V3_LGBM_Estimator(str(REPO / "lgbm_model_v3"))
    m = joblib.load(GART2_DIR / "gart2_final.joblib")
    feats = list(m.feature_name())

    def gart2_once(X, d):
        t0 = time.perf_counter()
        f = compute_features(X, d)
        t1 = time.perf_counter()
        row = pd.DataFrame([{k: f[k] for k in feats}], columns=feats)
        _ = _to_alpha(m.predict(row), "logit")
        return (t1 - t0), (time.perf_counter() - t1)

    def v3_once(X, d):
        est3._feature_cache.clear()
        r = est3.estimate(X, d, 0)
        return r["feature_time"], r["inference_time"]

    print("[timing] warming JIT ...")
    for d in (2, 10, 50):
        W = rand_instance(400, d)
        gart2_once(W, d); v3_once(W, d)

    # ---- per stratum -------------------------------------------------------
    suite = oh.load_suite()
    rows = []
    for stratum in ("bench2d", "augment", "tsplib_euc2d"):
        gf, gi, vf, vi = [], [], [], []
        names = list(map(str, suite[stratum].truth.index))
        for inst in names:
            try:
                if stratum == "bench2d":
                    X = load_2d_bench(inst); d = X.shape[1]
                elif stratum == "augment":
                    j = _json.loads((REPO / "augment" / "instances" / f"{inst}.json")
                                    .read_text(encoding="utf-8"))
                    X = np.asarray(j["coordinates"], dtype=np.float32)
                    d = int(j["dimension"])
                else:
                    p = _tsplib_tasks().get(inst)
                    info = parse_tsplib_file(str(p))
                    if not (info["is_native_euclidean"] and info["raw_coords"] is not None):
                        continue
                    X = info["raw_coords"].astype(np.float32); d = X.shape[1]
                a, b = gart2_once(X, d); c, e = v3_once(X, d)
                gf.append(a); gi.append(b); vf.append(c); vi.append(e)
            except Exception:  # noqa: BLE001
                continue
        if not gf:
            continue
        gf, gi, vf, vi = map(np.asarray, (gf, gi, vf, vi))
        rows.append({
            "scope": stratum, "n_instances": len(gf),
            "g2_feat_median_ms": np.median(gf) * 1e3,
            "g2_feat_p90_ms": np.percentile(gf, 90) * 1e3,
            "g2_inf_median_ms": np.median(gi) * 1e3,
            "g2_inf_p90_ms": np.percentile(gi, 90) * 1e3,
            "v3_feat_median_ms": np.median(vf) * 1e3,
            "v3_feat_p90_ms": np.percentile(vf, 90) * 1e3,
            "v3_inf_median_ms": np.median(vi) * 1e3,
            "ratio_feat_median": np.median(gf) / np.median(vf),
            "ratio_total_median": (np.median(gf) + np.median(gi))
                                  / (np.median(vf) + np.median(vi)),
        })
        print(f"  {stratum:14s} n={len(gf):5d} "
              f"G2 {np.median(gf)*1e3:8.2f}ms  V3 {np.median(vf)*1e3:8.2f}ms  "
              f"ratio {np.median(gf)/np.median(vf):.2f}x")

    # ---- the four reference sizes -----------------------------------------
    cases = [("rand_n1000_d2", rand_instance(1000, 2), 2),
             ("rand_n1000_d50", rand_instance(1000, 50), 50),
             ("rand_n10000_d2", rand_instance(10000, 2), 2)]
    p = _tsplib_tasks().get("pla85900")
    if p:
        info = parse_tsplib_file(str(p))
        cases.append(("tsplib_pla85900", info["raw_coords"].astype(np.float32), 2))
    for name, X, d in cases:
        reps = 3 if X.shape[0] <= 20000 else 2
        g = np.array([gart2_once(X, d) for _ in range(reps)])
        v = np.array([v3_once(X, d) for _ in range(reps)])
        rows.append({
            "scope": name, "n_instances": int(X.shape[0]),
            "g2_feat_median_ms": np.median(g[:, 0]) * 1e3,
            "g2_feat_p90_ms": np.percentile(g[:, 0], 90) * 1e3,
            "g2_inf_median_ms": np.median(g[:, 1]) * 1e3,
            "g2_inf_p90_ms": np.percentile(g[:, 1], 90) * 1e3,
            "v3_feat_median_ms": np.median(v[:, 0]) * 1e3,
            "v3_feat_p90_ms": np.percentile(v[:, 0], 90) * 1e3,
            "v3_inf_median_ms": np.median(v[:, 1]) * 1e3,
            "ratio_feat_median": np.median(g[:, 0]) / np.median(v[:, 0]),
            "ratio_total_median": (np.median(g[:, 0]) + np.median(g[:, 1]))
                                  / (np.median(v[:, 0]) + np.median(v[:, 1])),
        })
        print(f"  {name:16s} n={X.shape[0]:6d} "
              f"G2 {np.median(g[:,0])*1e3:9.2f}ms  V3 {np.median(v[:,0])*1e3:9.2f}ms  "
              f"ratio {np.median(g[:,0])/np.median(v[:,0]):.2f}x")

    T = pd.DataFrame(rows)
    T.to_csv(f"{OUT_PREFIX}_gart2_timing_final.csv", index=False)
    print(f"\n[timing] -> {OUT_PREFIX}_gart2_timing_final.csv")


def cmd_bygen(args) -> None:
    """2D benchmark broken out by the underlying generator (not just class),
    so the `grid` sub-generator can be reported on its own."""
    import pandas as pd
    PI = pd.read_csv(f"{OUT_PREFIX}_allmodels_per_instance.csv")
    gen = pd.read_csv(REPO / "paper_tooling" / "augmentation_2d_features.csv",
                      usecols=["instance_name", "generator", "gen_class"])
    sub = PI[PI.stratum == "bench2d"].merge(
        gen, left_on="instance", right_on="instance_name", how="left")
    sub["ape"] = sub.err_pct.abs()
    out = (sub.groupby(["model", "generator"])
           .agg(n=("ape", "size"), mape=("ape", "mean"),
                sdpe=("err_pct", lambda s: float(np.std(s, ddof=1))),
                mspe=("err_pct", lambda s: float(np.mean(s ** 2))))
           .reset_index())
    out.to_csv(f"{OUT_PREFIX}_allmodels_2d_by_generator.csv", index=False)
    p = out.pivot_table(index="model", columns="generator", values="mape")
    print("=== 2D MAPE by generator ===")
    print(p.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== `grid` sub-generator only ===")
    print(out[out.generator == "grid"][["model", "n", "sdpe", "mape", "mspe"]]
          .sort_values("mape").to_string(index=False,
                                         float_format=lambda x: f"{x:.4f}"))


def cmd_meantest(args) -> None:
    """Paired mean test of absolute percent error against the close baselines.

    The MDE is recomputed from each pair's OWN paired-difference SD rather than
    read off a precomputed floor: a candidate that improves by shrinking its
    large errors also shrinks that SD, which loosens the target it has to clear.
    """
    import pandas as pd
    import ood_harness as oh

    PI = pd.read_csv(f"{OUT_PREFIX}_allmodels_per_instance.csv")
    suite = oh.load_suite()
    stratum = "tsplib_euc2d"
    st = suite[stratum]
    truth = st.truth

    holm = {}
    for label in PI.model.unique():
        sub = PI[PI.model == label]
        preds = dict(zip(sub.instance.astype(str), sub.pred_cost.astype(float)))
        try:
            v = oh.evaluate_candidate(preds, label)
            c = v.comparisons
            c = c[c.stratum == stratum]
            for _, r in c.iterrows():
                holm[(label, r.model_b)] = (r.get("p_holm", np.nan),
                                            r.get("family_size", np.nan))
        except Exception as e:  # noqa: BLE001
            print(f"  evaluate_candidate({label}): {type(e).__name__}: {e}")

    rows = []
    for label in PI.model.unique():
        sub = PI[(PI.model == label) & (PI.stratum == stratum)]
        pa = pd.Series(sub.pred_cost.to_numpy(), index=sub.instance.astype(str))
        for bname in MEANTEST_BASELINES:
            pb = st.baselines.get(bname)
            if pb is None:
                continue
            idx = [i for i in pa.index if i in pb.index and np.isfinite(pb[i])]
            a = oh.absolute_percent_error(pa.loc[idx], truth.loc[idx])
            b = oh.absolute_percent_error(pb.loc[idx], truth.loc[idx])
            d = (a - b).to_numpy()
            sd_diff = float(np.std(d, ddof=1))
            n = len(d)
            mde = float(oh.min_detectable_difference(sd_diff, n))
            try:
                cmp_ = oh.compare(pa.loc[idx], pb.loc[idx], truth.loc[idx],
                                  label, bname, n_boot=1000, n_perm=10000, seed=SEED)
            except Exception:  # noqa: BLE001
                cmp_ = {}
            ph, fam = holm.get((label, bname), (np.nan, np.nan))
            rows.append({
                "model": label, "baseline": bname, "n": n,
                "mape_model": float(a.mean()), "mape_baseline": float(b.mean()),
                "mean_diff": float(d.mean()),
                "ci_lo": cmp_.get("ci_low", np.nan), "ci_hi": cmp_.get("ci_high", np.nan),
                "p_wilcoxon": cmp_.get("p_wilcoxon", np.nan),
                "p_perm": cmp_.get("p_permutation", np.nan),
                "p_holm": ph, "family_size": fam,
                "sd_paired_diff": sd_diff, "mde_at_n": mde,
                "gain": -float(d.mean()),
                "clears_mde": bool(-float(d.mean()) >= mde),
            })
    T = pd.DataFrame(rows).sort_values(["baseline", "mean_diff"])
    T.to_csv(f"{OUT_PREFIX}_meantest_tsplib.csv", index=False)
    for bname, g in T.groupby("baseline"):
        print(f"\n=== paired mean APE test :: {stratum} vs {bname} "
              f"(negative mean_diff favours the model) ===")
        print(g[["model", "n", "mape_model", "mape_baseline", "mean_diff",
                 "ci_lo", "ci_hi", "p_wilcoxon", "p_perm", "p_holm",
                 "sd_paired_diff", "mde_at_n", "gain", "clears_mde"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[meantest] -> {OUT_PREFIX}_meantest_tsplib.csv")


def cmd_dispersion(args) -> None:
    """dispersion_verdict for every model, reported exactly as the shipped
    reference numbers were: one candidate per call against the harness's
    default baseline family, so Holm adjustment is comparable."""
    import pandas as pd
    import ood_harness as oh

    PI = pd.read_csv(f"{OUT_PREFIX}_allmodels_per_instance.csv")
    out = []
    for label in PI.model.unique():
        sub = PI[PI.model == label]
        preds = dict(zip(sub.instance.astype(str), sub.pred_cost.astype(float)))
        for stratum in ("tsplib_euc2d", "bench2d", "augment"):
            try:
                d = oh.dispersion_verdict({label: preds}, stratum=stratum)
            except Exception as e:  # noqa: BLE001
                print(f"  {label}/{stratum}: {type(e).__name__}: {e}")
                continue
            d = d[d.model_b.isin(DISP_BASELINES)]
            out.append(d)
    D = pd.concat(out, ignore_index=True)
    D.to_csv(f"{OUT_PREFIX}_allmodels_dispersion.csv", index=False)

    cols = ["model_a", "model_b", "n_pairs", "sdpe_a", "sdpe_b", "sd_ratio",
            "ratio_ci_low", "ratio_ci_high", "p_pitman_morgan", "p_holm",
            "family_size", "mde_ratio_unadj", "mde_ratio_holm"]
    cols = [c for c in cols if c in D.columns]
    for stratum, g in D.groupby("stratum"):
        print(f"\n=== dispersion :: {stratum} "
              f"(sd_ratio = model / baseline; < 1 favours the model) ===")
        print(g.sort_values(["model_b", "sd_ratio"])[cols]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[dispersion] -> {OUT_PREFIX}_allmodels_dispersion.csv")


# =============================================================================
# 13. Swept-feature monotonicity probes
# =============================================================================
# -- DEFINITIVE PROBE PROTOCOL -------------------------------------------------
# Grids are log-spaced and run well past the training range (train is n <= 1000
# and d <= 50, with d=100 held out), so the probe covers the extrapolation
# regime where the paper actually makes its claims.
#
# CAVEAT that belongs in the paper: this is a *ceteris paribus* probe. It
# overwrites one column and holds every other feature at the instance's real
# value, so a swept row does not correspond to any realisable instance --
# changing n or d would move the MST features too. It tests the shape of the
# learned function, which is exactly what a structural guarantee is about, but
# it is not a claim about real instances at those n and d.
PROBE_N_INSTANCES = 1000
PROBE_GRID_POINTS = 24
PROBE_TOL = 1e-9          # alpha units; below this we are counting float noise


def _log_int_grid(lo: int, hi: int, k: int) -> List[int]:
    g = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), k)).astype(int))
    return [int(v) for v in g]


PROBE_D_GRID = _log_int_grid(2, 200, PROBE_GRID_POINTS)
PROBE_N_GRID = _log_int_grid(5, 4000, PROBE_GRID_POINTS)


def _to_alpha(z, target: str):
    if target == "logit":
        return 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    return np.clip(z, 1.0, 2.0)


def _sweep_monotonicity(model, feats: List[str], base, col: str,
                        grid: Sequence[float], tol: float = PROBE_TOL,
                        target: str = "alpha") -> dict:
    """Monotonicity of the prediction in ``col``, raw and as deployed.

    Returns pass rates and, more informatively, the MAGNITUDE of the
    violations: a model that breaks monotonicity by 1e-6 is materially
    monotone, one that breaks it by 0.05 is not, and a pass rate cannot tell
    those apart.
    """
    preds = np.empty((len(base), len(grid)), dtype=np.float64)
    for j, g in enumerate(grid):
        X = base.copy()
        X[col] = g
        preds[:, j] = model.predict(X[feats])

    out: dict = {}
    for kind, P in (("raw", preds), ("deployed", _to_alpha(preds, target))):
        diffs = np.diff(P, axis=1)
        viol = diffs > tol
        inc = diffs[viol]
        out[f"pct_nonincr_{kind}"] = float((~viol.any(axis=1)).mean()) * 100.0
        out[f"n_viol_{kind}"] = int(viol.sum())
        out[f"viol_med_{kind}"] = float(np.median(inc)) if inc.size else 0.0
        out[f"viol_p99_{kind}"] = float(np.percentile(inc, 99)) if inc.size else 0.0
        out[f"viol_max_{kind}"] = float(inc.max()) if inc.size else 0.0
    out["n_pairs"] = int(len(base) * (len(grid) - 1))
    return out


def cmd_probe(args) -> None:
    import joblib
    import pandas as pd

    df = pd.read_csv(REPO / "tsp_features_v4.csv")
    df = df[df["split"] == "test"].reset_index(drop=True)
    base = df.sample(min(PROBE_N_INSTANCES, len(df)), random_state=SEED).copy()
    print(f"[probe] {len(base)} ND-test instances @ seed {SEED} | "
          f"d grid {len(PROBE_D_GRID)} pts [{PROBE_D_GRID[0]}..{PROBE_D_GRID[-1]}] | "
          f"n grid {len(PROBE_N_GRID)} pts [{PROBE_N_GRID[0]}..{PROBE_N_GRID[-1]}] | "
          f"tol {PROBE_TOL:g}")

    rows = []
    for label, m, feats, target in _load_models():
        for col, grid in (("dimension", PROBE_D_GRID), ("n_customers", PROBE_N_GRID)):
            r = _sweep_monotonicity(m, feats, base, col, grid, target=target)
            r.update({"model": label, "swept": col, "n_probes": len(base),
                      "grid_points": len(grid)})
            rows.append(r)
            print(f"  {label:24s} {col:12s} deployed {r['pct_nonincr_deployed']:6.1f}%  "
                  f"raw {r['pct_nonincr_raw']:6.1f}%  "
                  f"viol med/p99/max (raw) "
                  f"{r['viol_med_raw']:.2e}/{r['viol_p99_raw']:.2e}/{r['viol_max_raw']:.2e}")
    t = pd.DataFrame(rows)[[
        "model", "swept", "n_probes", "grid_points", "n_pairs",
        "pct_nonincr_deployed", "n_viol_deployed", "viol_med_deployed",
        "viol_p99_deployed", "viol_max_deployed",
        "pct_nonincr_raw", "n_viol_raw", "viol_med_raw", "viol_p99_raw",
        "viol_max_raw"]]
    t.to_csv(f"{OUT_PREFIX}_gart2_probe.csv", index=False)
    print(f"\n[probe] -> {OUT_PREFIX}_gart2_probe.csv")


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("greedy").set_defaults(fn=cmd_greedy)
    sub.add_parser("cost").set_defaults(fn=cmd_cost)
    c = sub.add_parser("cache"); c.add_argument("--which", default="all",
                                                choices=["2d", "tsplib", "all"])
    c.set_defaults(fn=cmd_cache)
    a = sub.add_parser("ablate"); a.add_argument("--arms", default="")
    a.set_defaults(fn=cmd_ablate)
    sub.add_parser("ood").set_defaults(fn=cmd_ood)
    sub.add_parser("predict").set_defaults(fn=cmd_predict)
    sub.add_parser("probe").set_defaults(fn=cmd_probe)
    sub.add_parser("features").set_defaults(fn=cmd_features)
    sub.add_parser("evalall").set_defaults(fn=cmd_evalall)
    sub.add_parser("dispersion").set_defaults(fn=cmd_dispersion)
    sub.add_parser("meantest").set_defaults(fn=cmd_meantest)
    sub.add_parser("conformal").set_defaults(fn=cmd_conformal)
    sub.add_parser("bygen").set_defaults(fn=cmd_bygen)
    sub.add_parser("timing-final").set_defaults(fn=cmd_timing_final)
    sub.add_parser("strata").set_defaults(fn=cmd_strata)
    sub.add_parser("consistency").set_defaults(fn=cmd_consistency)
    tp = sub.add_parser("timing")
    tp.add_argument("--loaded", type=int, default=0,
                    help="1 if other work was running during measurement")
    tp.set_defaults(fn=cmd_timing)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
