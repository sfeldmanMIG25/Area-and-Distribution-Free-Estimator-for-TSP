"""Incremental-value screen for the candidate feature groups in ``features_ext/``.

The screening target is NOT alpha. It is the signed residual of the shipped
model,

    r_i = alpha_hat_i - alpha_i,
    alpha_i     = clip(optimal_cost_i / mst_total_length_i, 1, 2)
    alpha_hat_i = clip(LGBM_V3.predict(X30_i), 1, 2)

which is exactly the quantity the model gets wrong (positive = over-predict,
negative = under-predict). A candidate feature is valuable to the extent it
explains r.

Evaluation frame
----------------
A. The full 2,580-instance 2D benchmark (``Generalized_TSP_Analysis/``). The
   30 shipped features are recomputed with the canonical extractor
   ``feature_creator_v3.compute_features_for_instance_v3`` so the residual is
   the residual of the real pipeline. These instances are disjoint from the
   training corpus (different name space), so every 2D residual is honest.
B. A stratified sample of the multidimensional corpus (``instances/`` +
   ``tsp_features_v3.csv``), drawn only from the ``val``/``test`` splits so the
   residual is out-of-sample there too. Stratification is
   (dimension x n_customers), 18 x 12 = 216 cells.

Candidate features are computed on ``np.unique(coords, axis=0)`` in the RAW
frame (no PCA rotation). ``local_id`` and ``mst_topology`` are rotation
invariant so the frame is irrelevant to them; ``degeneracy``'s coordinate-tie
block is rotation sensitive by design and the raw frame is the one in which an
axis-aligned pile-up exists. Adopting any group-(c) degeneracy feature
therefore requires computing it BEFORE ``canonicalize_coords_pca``.

Outputs (all small CSVs next to this file):
    feature_screen_frame.csv        the evaluation frame (cached, reused)
    feature_screen_correlations.csv correlation + redundancy table
    feature_screen_incremental.csv  out-of-fold R^2 on the residual
    feature_screen_shortlist.csv    ranked shortlist
    feature_screen_cost.csv         measured per-instance cost
"""

from __future__ import annotations

import os
import sys

# Single-thread BLAS in every process: the candidate groups call eigh/BLAS and
# a process pool of 20 x multithreaded BLAS thrashes on Windows.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HERE = os.path.join(ROOT, "paper_tooling")
BENCH_DIR = os.path.join(ROOT, "Generalized_TSP_Analysis")
BENCH_INST = os.path.join(BENCH_DIR, "instances")
BENCH_SOL = os.path.join(BENCH_DIR, "solutions")
ND_INST = os.path.join(ROOT, "instances")
ND_FEATS = os.path.join(ROOT, "tsp_features_v3.csv")
MODEL_PATH = os.path.join(ROOT, "lgbm_model_v3", "lgbm_alpha_model_v3.joblib")

FRAME_CSV = os.path.join(HERE, "feature_screen_frame.csv")
CORR_CSV = os.path.join(HERE, "feature_screen_correlations.csv")
INCR_CSV = os.path.join(HERE, "feature_screen_incremental.csv")
SHORT_CSV = os.path.join(HERE, "feature_screen_shortlist.csv")
COST_CSV = os.path.join(HERE, "feature_screen_cost.csv")

ND_PER_CELL = 42          # 216 cells -> ~9,072 held-out ND rows
ND_CORR_SUPPLEMENT = 600  # extra k-heavy ND rows (line-like transfer test)
RANDOM_STATE = 42
N_WORKERS = max(1, min(20, (os.cpu_count() or 4)))

GROUP_MODULES = ("local_id", "degeneracy", "mst_topology")

# 2D benchmark generator -> paper generator class (paper_tooling/build_paper_tables.py)
GEN_CLASSES = {
    "Isotropic": {"random", "normal", "triangular", "truncated_exponential"},
    "Biased": {"squeezed_uniform", "uniform_triangular", "triangular_squeezed", "correlated"},
    "Geometric": {"grid", "boundary", "x_central"},
    "Clustered": {"clustered"},
    "LineNoise": {"line_noise"},
}


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------

_CACHE: dict = {}


def _modules():
    if "mods" not in _CACHE:
        from features_ext import group_degeneracy, group_local_id, group_mst_topology
        _CACHE["mods"] = {
            "local_id": group_local_id,
            "degeneracy": group_degeneracy,
            "mst_topology": group_mst_topology,
        }
    return _CACHE["mods"]


def _compute_mst():
    if "mst" not in _CACHE:
        from mst_utils import compute_mst
        _CACHE["mst"] = compute_mst
    return _CACHE["mst"]


def _candidates(coords: np.ndarray) -> dict:
    """All candidate features on the raw, deduplicated cloud."""
    mods = _modules()
    mst = _compute_mst()(coords)
    out = {}
    t0 = time.perf_counter()
    out.update(mods["local_id"].compute(coords))
    t1 = time.perf_counter()
    out.update(mods["degeneracy"].compute(coords))
    t2 = time.perf_counter()
    out.update(mods["mst_topology"].compute(coords, mst))
    t3 = time.perf_counter()
    out["_t_local_id_ms"] = (t1 - t0) * 1e3
    out["_t_degeneracy_ms"] = (t2 - t1) * 1e3
    out["_t_mst_topology_ms"] = (t3 - t2) * 1e3
    return out


def worker_2d(name: str):
    """Recompute the shipped 30 features + all candidates for one 2D instance."""
    try:
        import feature_creator_v3 as fc

        with open(os.path.join(BENCH_INST, name + ".json")) as f:
            inst = json.load(f)
        with open(os.path.join(BENCH_SOL, name + ".sol.json")) as f:
            sol = json.load(f)
        inst["coordinates"] = np.asarray(inst["coordinates"], dtype=np.float64)
        inst["instance_name"] = name

        feats = fc.compute_features_for_instance_v3(inst, sol)
        if feats is None:
            return None
        feats["optimal_cost"] = float(sol["optimal_cost"])

        coords = np.unique(inst["coordinates"], axis=0).astype(np.float64)
        feats.update(_candidates(coords))
        feats["source"] = "2D"
        feats["generator"] = inst.get("distribution_type", "unknown")
        return feats
    except Exception as exc:  # surface, do not silently drop
        return {"instance_name": name, "source": "2D", "_error": repr(exc)}


def worker_nd(name: str):
    """Candidate features only; the 30 shipped features come from the CSV."""
    try:
        path = os.path.join(ND_INST, name + ".bin")
        import struct

        with open(path, "rb") as f:
            n, d, _g = struct.unpack("III", f.read(12))
            dist_len = struct.unpack("I", f.read(4))[0]
            f.read(dist_len)
            buf = f.read(n * d * 4)
        coords = np.frombuffer(buf, dtype=np.float32).reshape(n, d)
        coords = np.unique(coords, axis=0).astype(np.float64)
        out = _candidates(coords)
        out["instance_name"] = name
        out["source"] = "ND"
        return out
    except Exception as exc:
        return {"instance_name": name, "source": "ND", "_error": repr(exc)}


# --------------------------------------------------------------------------
# frame construction
# --------------------------------------------------------------------------

def nd_letters(name: str, dimension: int) -> str:
    """Per-axis distribution letters are embedded in the ND instance name.

    ``N1000_D10_G10000_akcalbecno_33`` -> letters ``akcalbecno`` (one per axis).
    """
    for p in name.split("_")[3:]:
        if len(p) == dimension and p.isalpha():
            return p
    return ""


def nd_generator_class(name: str, dimension: int) -> str:
    """Letter families (data_pipeline/instance_io.py DISTRIBUTION_MAP_1D):

      k    -> correlated: ``base + gaussian noise`` then CLIPPED to the box.
              This is the ND analogue of the 2D LineNoise generator, including
              the boundary pile-up, so ``ND_corr_hi`` is the cross-generator
              transfer test for any LineNoise fix.
      c,o  -> clustered      n,i -> normal      everything else -> uniform

    Letters are drawn uniformly from 16 codes, so E[frac_k] = 1/16 = 0.0625.
    The thresholds below are ~4x / ~2.5x expectation.
    """
    s = nd_letters(name, dimension)
    if not s:
        return "ND_unknown"
    d = len(s)
    f_k = sum(ch == "k" for ch in s) / d
    f_c = sum(ch in "co" for ch in s) / d
    f_n = sum(ch in "ni" for ch in s) / d
    if f_k >= 0.25:
        return "ND_corr_hi"
    if f_k >= 0.10:
        return "ND_corr_mid"
    if f_c >= 0.30:
        return "ND_clus_hi"
    if f_n >= 0.30:
        return "ND_norm_hi"
    return "ND_plain"


def build_frame() -> pd.DataFrame:
    if os.path.exists(FRAME_CSV):
        print(f"[frame] reusing {FRAME_CSV}")
        return pd.read_csv(FRAME_CSV)

    # ---- part A: the 2,580-instance 2D benchmark -------------------------
    names_2d = sorted(f[:-5] for f in os.listdir(BENCH_INST) if f.endswith(".json"))
    print(f"[frame] 2D benchmark: {len(names_2d)} instances")

    rows_2d = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(worker_2d, nm) for nm in names_2d]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r is not None:
                rows_2d.append(r)
            if i % 500 == 0:
                print(f"  2D {i}/{len(names_2d)}  {time.perf_counter()-t0:.0f}s")
    df2 = pd.DataFrame(rows_2d)
    errs = df2["_error"].notna().sum() if "_error" in df2 else 0
    print(f"[frame] 2D done in {time.perf_counter()-t0:.0f}s, errors={errs}")
    if errs:
        print(df2.loc[df2["_error"].notna(), ["instance_name", "_error"]].head(5).to_string())
        df2 = df2[df2["_error"].isna()].drop(columns=["_error"])

    df2["gen_class"] = "unknown"
    for cls, members in GEN_CLASSES.items():
        df2.loc[df2["generator"].isin(members), "gen_class"] = cls
    unmapped = sorted(set(df2.loc[df2.gen_class == "unknown", "generator"]))
    if unmapped:
        raise RuntimeError(f"unmapped 2D generators: {unmapped}")

    # ---- part B: stratified held-out sample of the ND corpus -------------
    meta_cols = ["instance_name", "n_customers", "dimension", "split",
                 "optimal_cost", "mst_total_length"]
    import joblib
    model = joblib.load(MODEL_PATH)
    feat30 = list(model.feature_name_)
    nd = pd.read_csv(ND_FEATS, usecols=sorted(set(meta_cols) | set(feat30)))
    nd = nd[nd["split"].isin(["val", "test"])]
    nd["gcls"] = [nd_generator_class(nm, int(d))
                  for nm, d in zip(nd.instance_name, nd.dimension)]
    rng = np.random.default_rng(RANDOM_STATE)
    picks = []
    for (_d, _n), grp in nd.groupby(["dimension", "n_customers"], sort=True):
        k = min(ND_PER_CELL, len(grp))
        idx = rng.choice(grp.index.to_numpy(), size=k, replace=False)
        picks.append(grp.loc[np.sort(idx)])
    nd_s = pd.concat(picks, ignore_index=True)
    # Supplement the line-like ND bucket: 'k'-heavy instances are only ~4% of
    # the corpus but they are the cross-generator transfer test for a
    # LineNoise fix, so give that subset enough rows to measure.
    extra = nd[(nd.gcls == "ND_corr_hi") & (~nd.instance_name.isin(set(nd_s.instance_name)))]
    if len(extra) > ND_CORR_SUPPLEMENT:
        extra = extra.loc[np.sort(rng.choice(extra.index.to_numpy(),
                                             size=ND_CORR_SUPPLEMENT, replace=False))]
    nd_s = pd.concat([nd_s, extra], ignore_index=True).drop(columns=["gcls"])
    print(f"[frame] ND sample: {len(nd_s)} rows over "
          f"{nd_s.dimension.nunique()} dims x {nd_s.n_customers.nunique()} sizes "
          f"(held-out only: {sorted(nd_s.split.unique())}); "
          f"ND_corr_hi supplement={len(extra)}")

    rows_nd = []
    t0 = time.perf_counter()
    names_nd = nd_s["instance_name"].tolist()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(worker_nd, nm) for nm in names_nd]
        for i, fu in enumerate(as_completed(futs), 1):
            rows_nd.append(fu.result())
            if i % 1000 == 0:
                print(f"  ND {i}/{len(names_nd)}  {time.perf_counter()-t0:.0f}s")
    dfn = pd.DataFrame(rows_nd)
    errs = dfn["_error"].notna().sum() if "_error" in dfn else 0
    print(f"[frame] ND done in {time.perf_counter()-t0:.0f}s, errors={errs}")
    if errs:
        print(dfn.loc[dfn["_error"].notna(), ["instance_name", "_error"]].head(5).to_string())
        dfn = dfn[dfn["_error"].isna()].drop(columns=["_error"])

    dfn = dfn.merge(nd_s, on="instance_name", how="inner")
    dfn["generator"] = [nd_generator_class(nm, int(d))
                        for nm, d in zip(dfn.instance_name, dfn.dimension)]
    dfn["gen_class"] = dfn["generator"]

    frame = pd.concat([df2, dfn], ignore_index=True, sort=False)
    frame = frame.drop(columns=[c for c in ("split", "grid_size") if c in frame])
    frame.to_csv(FRAME_CSV, index=False)
    print(f"[frame] wrote {FRAME_CSV}: {frame.shape}")
    return frame


def cv_group_labels(frame: pd.DataFrame) -> np.ndarray:
    """Group label for GroupKFold: a model must never see a generator it is
    scored on.

    2D rows -> the sub-generator (13 labels: line_noise, grid, boundary, ...).
    ND rows -> the letter-family class. ``ND_plain`` and ``ND_corr_mid`` are
    over half the frame on their own, and a single group that large forces one
    CV fold to be half the data, so those two are split by dimension band.
    ``ND_corr_hi`` -- the line-like ND bucket, the cross-generator transfer
    test -- is deliberately left whole.
    """
    def band(d):
        d = int(d)
        return "d2-6" if d <= 6 else "d7-25" if d <= 25 else "d26-50" if d <= 50 else "d100"

    out = []
    for src, gen, dim in zip(frame["source"], frame["generator"], frame["dimension"]):
        if src == "2D":
            out.append(gen)
        elif gen in ("ND_plain", "ND_corr_mid"):
            out.append(f"{gen}_{band(dim)}")
        else:
            out.append(gen)
    return np.asarray(out)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def main():
    import joblib
    from scipy import stats
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.model_selection import GroupKFold
    import lightgbm as lgb

    frame = build_frame()
    model = joblib.load(MODEL_PATH)
    feat30 = list(model.feature_name_)

    from features_ext import group_degeneracy, group_local_id, group_mst_topology
    group_of = {}
    for gname, mod in (("local_id", group_local_id),
                       ("degeneracy", group_degeneracy),
                       ("mst_topology", group_mst_topology)):
        for f in mod.feature_names():
            group_of[f] = gname
    cands_all = [c for c in group_of if c in frame.columns]

    # ---- target -----------------------------------------------------------
    frame = frame.dropna(subset=feat30 + ["optimal_cost", "mst_total_length"]).copy()
    alpha = (frame["optimal_cost"] / frame["mst_total_length"].replace(0, 1e-9)).clip(1.0, 2.0)
    ahat = np.clip(model.predict(frame[feat30]), 1.0, 2.0)
    frame["alpha"] = alpha.to_numpy()
    frame["resid"] = ahat - alpha.to_numpy()

    a = frame["alpha"].to_numpy()
    r = frame["resid"].to_numpy()

    print("\n=== evaluation frame ===")
    print(f"rows={len(frame)}  2D={int((frame.source=='2D').sum())}  "
          f"ND={int((frame.source=='ND').sum())}")
    summ = (frame.assign(pe=100 * frame.resid / frame.alpha)
                 .groupby("gen_class")
                 .agg(n=("pe", "size"), MAPE=("pe", lambda s: np.abs(s).mean()),
                      MSPE=("pe", "mean"))
                 .sort_values("MSPE"))
    print(summ.round(3).to_string())

    is_ln = (frame["generator"] == "line_noise").to_numpy()
    is_grid = (frame["generator"] == "grid").to_numpy()
    is_ndk = (frame["generator"] == "ND_corr_hi").to_numpy()
    is_2d = (frame["source"] == "2D").to_numpy()
    SUBSETS = {"all": None, "LineNoise": is_ln, "grid": is_grid,
               "ND_corr_hi": is_ndk, "2D_bench": is_2d}
    print(f"\nLineNoise n={is_ln.sum()}  grid n={is_grid.sum()}  "
          f"ND_corr_hi n={is_ndk.sum()}  2D n={is_2d.sum()}")

    # ---- drop degenerate candidates --------------------------------------
    X_c = frame[cands_all].astype(float).replace([np.inf, -np.inf], np.nan)
    bad = X_c.columns[X_c.isna().any() | (X_c.nunique() <= 1)].tolist()
    if bad:
        print(f"\n[drop] constant / non-finite candidates ({len(bad)}): {bad}")
    cands = [c for c in cands_all if c not in bad]
    print(f"[cand] screening {len(cands)} candidates")

    Xc = frame[cands].to_numpy(float)
    X30 = frame[feat30].to_numpy(float)

    def sp(x, y):
        if len(x) < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return np.nan
        return float(stats.spearmanr(x, y).statistic)

    # ---- 2. correlation + MI with the residual ---------------------------
    mi = mutual_info_regression(Xc, r, random_state=RANDOM_STATE, n_neighbors=3)

    # ---- 3. redundancy vs the existing 30 --------------------------------
    def max_cross(rows_mask):
        """|Spearman| of each candidate against its closest existing feature."""
        A = Xc[rows_mask] if rows_mask is not None else Xc
        B = X30[rows_mask] if rows_mask is not None else X30
        ra = np.apply_along_axis(stats.rankdata, 0, A)
        rb = np.apply_along_axis(stats.rankdata, 0, B)
        ra = (ra - ra.mean(0)) / (ra.std(0) + 1e-12)
        rb = (rb - rb.mean(0)) / (rb.std(0) + 1e-12)
        return np.abs(ra.T @ rb) / len(A)

    cross = max_cross(None)
    # Pooled over d = 2..100 every intrinsic-dimension estimator is ~0.9
    # correlated with `dimension` purely because it estimates dimension. The
    # 2D-only view is the honest "does this restate an existing feature at
    # fixed ambient dimension" measure.
    cross2d = max_cross(is_2d)

    rows = []
    for j, c in enumerate(cands):
        k, k2 = int(np.argmax(cross[j])), int(np.argmax(cross2d[j]))
        rows.append({
            "feature": c, "group": group_of[c],
            "rho_resid_all": sp(Xc[:, j], r),
            "rho_resid_LN": sp(Xc[is_ln, j], r[is_ln]),
            "rho_resid_grid": sp(Xc[is_grid, j], r[is_grid]),
            "rho_resid_ND_corr_hi": sp(Xc[is_ndk, j], r[is_ndk]),
            "mi_resid": float(mi[j]),
            "max_abs_rho_vs_30": float(cross[j, k]),
            "closest_existing": feat30[k],
            "max_abs_rho_vs_30_2Donly": float(cross2d[j, k2]),
            "closest_existing_2D": feat30[k2],
        })
    corr = pd.DataFrame(rows)
    # Selection score: informative about the residual, not a restatement of an
    # existing feature at fixed dimension, and useful in BOTH failure subsets.
    corr["two_sided"] = np.minimum(corr.rho_resid_LN.abs().fillna(0),
                                   corr.rho_resid_grid.abs().fillna(0))
    corr["score"] = (corr["mi_resid"]
                     * (1.0 - corr["max_abs_rho_vs_30_2Donly"].clip(0, 1))
                     * (0.5 + corr["two_sided"]))
    corr = corr.sort_values("score", ascending=False).reset_index(drop=True)
    corr.to_csv(CORR_CSV, index=False)
    print("\n=== 2/3. residual correlation + redundancy (top 30 by score) ===")
    show = ["feature", "group", "rho_resid_all", "rho_resid_LN", "rho_resid_grid",
            "rho_resid_ND_corr_hi", "mi_resid", "max_abs_rho_vs_30",
            "max_abs_rho_vs_30_2Donly", "closest_existing_2D", "score"]
    print(corr[show].head(30).round(4).to_string(index=False))

    # ---- 4. incremental out-of-fold R^2 ----------------------------------
    groups = cv_group_labels(frame)
    n_splits = 6
    gkf = GroupKFold(n_splits=n_splits)
    gs = pd.Series(groups).value_counts()
    print(f"\n[cv] GroupKFold n_splits={n_splits} over {gs.size} generator groups "
          f"(largest {gs.iloc[0]} = {100*gs.iloc[0]/len(frame):.0f}% of frame)")

    FOLDS = list(gkf.split(Xc, r, groups))
    for fi, (_tr, te) in enumerate(FOLDS):
        print(f"    fold {fi}: n_test={len(te):6d}  LN={int(is_ln[te].sum()):4d} "
              f"grid={int(is_grid[te].sum()):4d}  "
              f"groups={sorted(set(groups[te]))[:4]}...")

    def oof(cols):
        pred = np.full(len(frame), np.nan)
        M = frame[cols].to_numpy(float)
        for tr, te in FOLDS:
            m = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=15,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_STATE,
                n_jobs=N_WORKERS, verbose=-1, deterministic=True, force_row_wise=True)
            m.fit(M[tr], r[tr])
            pred[te] = m.predict(M[te])
        return pred

    def metrics(pred, tag=""):
        """Everything reported downstream, for one out-of-fold prediction.

        `pred` is an estimate of the shipped model's signed residual, so the
        corrected error is (r - pred). MSPE/MAPE are that error as a percentage
        of alpha, i.e. directly comparable to the paper's error tables.
        """
        e = r - pred
        out = {}
        for name, m in SUBSETS.items():
            y, p, ee, aa = (r, pred, e, a) if m is None else (r[m], pred[m], e[m], a[m])
            sst = np.sum((y - y.mean()) ** 2)
            out[f"R2_{name}"] = float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan
            out[f"MSPE_{name}"] = float(np.mean(ee / aa) * 100)
            out[f"MAPE_{name}"] = float(np.mean(np.abs(ee / aa)) * 100)
        out["config"] = tag
        return out

    configs = {"no_correction": None, "base30": feat30}
    for g in GROUP_MODULES:
        gc = [c for c in cands if group_of[c] == g]
        if gc:
            configs[f"base30+{g}"] = feat30 + gc
    configs["base30+all_candidates"] = feat30 + cands

    t0 = time.perf_counter()
    incr_rows, oof_store = [], {}
    for name, cols in configs.items():
        p = np.zeros(len(frame)) if cols is None else oof(cols)
        oof_store[name] = p
        m = metrics(p, name)
        m["n_features"] = 0 if cols is None else len(cols)
        incr_rows.append(m)
        print(f"  [{time.perf_counter()-t0:5.0f}s] {name:24s} "
              f"R2_all={m['R2_all']:+.4f} R2_LN={m['R2_LineNoise']:+.4f} "
              f"R2_grid={m['R2_grid']:+.4f} | MSPE_LN={m['MSPE_LineNoise']:+.2f}% "
              f"MSPE_grid={m['MSPE_grid']:+.2f}%")

    def finish(rows):
        df = pd.DataFrame(rows)
        b = df.loc[df.config == "base30"].iloc[0]
        for s in SUBSETS:
            df[f"liftR2_{s}"] = df[f"R2_{s}"] - b[f"R2_{s}"]
        lead = ["config", "n_features"] + [f"R2_{s}" for s in SUBSETS] \
               + [f"liftR2_{s}" for s in SUBSETS] \
               + [f"MSPE_{s}" for s in SUBSETS] + [f"MAPE_{s}" for s in SUBSETS]
        return df[lead]

    # ---- 5. greedy forward selection, at most 8 --------------------------
    # The pool must be stratified by group. A single global ranking is
    # swamped by local_id (30 of 61 candidates, and its members are mutually
    # near-duplicates), which would keep the strongest group out of the greedy
    # search entirely. The composite score only prunes; the greedy decides.
    pool = []
    for g in GROUP_MODULES:
        gsub = corr[corr.group == g]
        take = set(gsub.head(8)["feature"])
        take |= set(gsub.reindex(gsub.rho_resid_LN.abs().sort_values(
            ascending=False).index).head(4)["feature"])
        take |= set(gsub.reindex(gsub.rho_resid_grid.abs().sort_values(
            ascending=False).index).head(4)["feature"])
        pool += [f for f in gsub["feature"] if f in take]
    print(f"\n[greedy] group-stratified pool of {len(pool)}:")
    for f in pool:
        print(f"    {group_of[f]:13s} {f}")

    base_r2 = metrics(oof_store["base30"])["R2_all"]
    chosen, hist, best_now = [], [], base_r2
    for step in range(8):
        best = None
        for f in pool:
            if f in chosen:
                continue
            m = metrics(oof(feat30 + chosen + [f]), f)
            if m["R2_all"] > best_now + 1e-5 and (best is None or m["R2_all"] > best["R2_all"]):
                best = m
        if best is None:
            print(f"[greedy] no further gain at step {step+1}; stopping")
            break
        best["step"] = step + 1
        best["feature"] = best.pop("config")
        best["group"] = group_of[best["feature"]]
        best["gain_R2_all"] = best["R2_all"] - best_now
        chosen.append(best["feature"])
        best_now = best["R2_all"]
        hist.append(best)
        print(f"  step {step+1}: +{best['feature']:38s} R2={best_now:+.4f} "
              f"(gain {best['gain_R2_all']:+.4f}) | "
              f"MSPE_LN={best['MSPE_LineNoise']:+.2f}% MAPE_LN={best['MAPE_LineNoise']:.2f}% "
              f"MSPE_grid={best['MSPE_grid']:+.2f}% MAPE_grid={best['MAPE_grid']:.2f}%")

    if hist:
        m = dict(hist[-1])
        m["config"] = f"base30+shortlist{len(chosen)}"
        m["n_features"] = 30 + len(chosen)
        incr_rows.append(m)
    incr = finish(incr_rows)
    incr.to_csv(INCR_CSV, index=False)

    short = pd.DataFrame(hist)
    if len(short):
        b = pd.DataFrame(incr_rows).set_index("config").loc["base30"]
        # Which failure does each addition actually fix? Compare the remaining
        # signed bias in each subset against the base30 corrector.
        short["fixes_LineNoise_MSPE_pp"] = b["MSPE_LineNoise"] - short["MSPE_LineNoise"]
        short["fixes_grid_MSPE_pp"] = b["MSPE_grid"] - short["MSPE_grid"]
        short = short.merge(corr[["feature", "rho_resid_LN", "rho_resid_grid",
                                  "rho_resid_ND_corr_hi", "mi_resid",
                                  "max_abs_rho_vs_30", "max_abs_rho_vs_30_2Donly",
                                  "closest_existing_2D"]], on="feature", how="left")
        cols = ["step", "feature", "group", "gain_R2_all", "R2_all", "R2_LineNoise",
                "R2_grid", "MSPE_LineNoise", "MAPE_LineNoise", "MSPE_grid",
                "MAPE_grid", "rho_resid_LN", "rho_resid_grid",
                "max_abs_rho_vs_30_2Donly", "mi_resid"]
        short = short[cols + [c for c in short.columns if c not in cols]]
    short.to_csv(SHORT_CSV, index=False)

    print("\n=== 4. incremental out-of-fold R^2 on the residual ===")
    print(incr.round(4).to_string(index=False))
    print("\n=== 5. greedy shortlist ===")
    if len(short):
        print(short.round(4).to_string(index=False))

    # ---- cost -------------------------------------------------------------
    tcols = ["_t_local_id_ms", "_t_degeneracy_ms", "_t_mst_topology_ms"]
    cost = (frame.assign(nb=pd.cut(frame.n_customers, [0, 50, 200, 600, 1001]),
                         db=pd.cut(frame.dimension, [0, 2, 10, 50, 100]))
                 .groupby(["db", "nb"], observed=True)[tcols].mean())
    cost["total_ms"] = cost.sum(axis=1)
    cost.to_csv(COST_CSV)
    print("\n=== added cost per instance (ms), by dimension band x n band ===")
    print(cost.round(2).to_string())
    tot = frame[tcols].sum(axis=1)
    print(f"\nmean total added ms/instance: {tot.mean():.2f} "
          f"(2D {tot[is_2d].mean():.2f}, ND {tot[~is_2d].mean():.2f}); "
          f"worst cell n=1000,d=100: "
          f"{tot[(frame.n_customers > 600) & (frame.dimension == 100)].mean():.2f}")
    for c in tcols:
        print(f"  {c:22s} mean {frame[c].mean():7.2f}  "
              f"p90 {frame[c].quantile(0.9):7.2f}  max {frame[c].max():7.2f}")

    print(f"\nwrote:\n  {CORR_CSV}\n  {INCR_CSV}\n  {SHORT_CSV}\n  {COST_CSV}")


if __name__ == "__main__":
    main()
