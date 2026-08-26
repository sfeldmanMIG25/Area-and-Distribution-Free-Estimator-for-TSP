"""Adversarial audit of the 5 local-geometry features added to GART 2.0.

The prior run (``paper_tooling/feature_retrain.py``) reported MF (35 features)
beating M0 (the 30 shipped features) on 2D LineNoise, 11.59 -> 5.94 MAPE.
This script tries to REFUTE the claim that the gain is a genuine, generator
invariant fix for degenerate geometry rather than a shortcut.

Default posture is REFUTED.  An attack only returns SURVIVED on unambiguous
evidence.

    a1  leave-one-generator-out retraining
    a2  novel degenerate geometry (the 874 augmentation instances)
    a3  shortcut probe (synthetic cases where elongation and local
        one-dimensionality disagree, with Concorde ground truth)
    a4  cost honesty
    a5  leakage audit

Every stage caches.  Nothing is written to ``lgbm_model_v3/`` and no
augmentation or benchmark instance ever enters a training split.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

import argparse
import ast
import hashlib
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "paper_tooling")
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import feature_retrain as FR  # noqa: E402  (reuses the exact training contract)

NEW = FR.NEW_FEATURES
RANDOM_STATE = FR.RANDOM_STATE
N_WORKERS = FR.N_WORKERS

AUG_DIR = os.path.join(ROOT, "augment")
AUG_INST = os.path.join(AUG_DIR, "instances")
AUG_FEATS_V3 = os.path.join(HERE, "augment_features_v3.csv")

OUT_AUG = os.path.join(HERE, "adv_augment_frame.csv")
OUT_AUG_RES = os.path.join(HERE, "adv_augment_results.csv")
OUT_LOGO = os.path.join(HERE, "adv_logo_results.csv")
OUT_LOGO_MECH = os.path.join(HERE, "adv_logo_mechanism.csv")
OUT_PROBE = os.path.join(HERE, "adv_probe_cases.csv")
OUT_PROBE_RES = os.path.join(HERE, "adv_probe_results.csv")
OUT_COST = os.path.join(HERE, "adv_cost.csv")
OUT_LEAK = os.path.join(HERE, "adv_leak.json")
OUT_REPORT = os.path.join(HERE, "adv_report.json")
PROBE_CACHE = os.path.join(HERE, "adv_probe_solved.json")

LOGO_DIR = os.path.join(HERE, "feature_models", "logo")


# ==========================================================================
# shared helpers
# ==========================================================================

def alpha_pred(model, X: pd.DataFrame, feats: list) -> np.ndarray:
    return np.clip(model.predict(X[feats], num_iteration=model.best_iteration_), 1.0, 2.0)


def err_from_alpha(model, df: pd.DataFrame, feats: list) -> np.ndarray:
    """Relative cost error, identical definition to feature_retrain.pct_err."""
    pa = alpha_pred(model, df, feats)
    pred_cost = pa * df["mst_total_length"].to_numpy()
    cost = df["optimal_cost"].to_numpy()
    return (pred_cost - cost) / cost


def load_models():
    import joblib
    md = os.path.join(HERE, "feature_models")
    m0 = joblib.load(os.path.join(md, "M0.joblib"))
    mf = joblib.load(os.path.join(md, "MF.joblib"))
    p = os.path.join(md, "MF_tuned.joblib")
    mft = joblib.load(p) if os.path.exists(p) else None
    f30 = list(m0.feature_name_)
    fall = list(mf.feature_name_)
    assert fall[:30] == f30 and fall[30:] == NEW, "feature order drift"
    return m0, mf, mft, f30, fall


def five_for_coords(coords: np.ndarray) -> dict:
    """The five shortlisted features, via the identical code path as the retrain."""
    return FR._five(np.unique(np.asarray(coords, dtype=np.float64), axis=0))


# ==========================================================================
# ATTACK 1 -- leave-one-generator-out
# ==========================================================================

def _nd_class_series(df: pd.DataFrame) -> pd.Series:
    from feature_screen import nd_generator_class
    return pd.Series([nd_generator_class(nm, int(d))
                      for nm, d in zip(df["instance_name"], df["dimension"])],
                     index=df.index)


def _k_frac(df: pd.DataFrame) -> pd.Series:
    from feature_screen import nd_letters
    out = []
    for nm, d in zip(df["instance_name"], df["dimension"]):
        s = nd_letters(nm, int(d))
        out.append(sum(ch == "k" for ch in s) / len(s) if s else np.nan)
    return pd.Series(out, index=df.index)


def generator_census(df: pd.DataFrame) -> pd.DataFrame:
    """What generators does the TRAINING corpus actually contain?"""
    g = _nd_class_series(df)
    kf = _k_frac(df)
    rows = []
    for split in ["train", "val", "test"]:
        m = df["split"] == split
        for cls, cm in g[m].value_counts().items():
            rows.append({"split": split, "nd_class": cls, "n": int(cm)})
    cen = pd.DataFrame(rows)
    # per-dimension prevalence of the 'k' (correlated + clipped) axis generator
    dd = pd.DataFrame({"dimension": df["dimension"], "k_frac": kf,
                       "split": df["split"]})
    dd = dd[dd["split"] == "train"]
    prev = dd.groupby("dimension")["k_frac"].agg(
        any_k=lambda s: float((s > 0).mean()),
        mean_k=lambda s: float(s.mean()),
        ge_25=lambda s: float((s >= 0.25).mean())).reset_index()
    return cen, prev


LOGO_SPECS = {
    # tag                 predicate on the corpus frame
    "corr_hi":  lambda df, g, kf, : g == "ND_corr_hi",
    "corr_all": lambda df, g, kf, : kf >= 0.10,
    "k_any_lowd": lambda df, g, kf, : (kf > 0) & (df["dimension"] <= 10),
    "clus":     lambda df, g, kf, : g == "ND_clus_hi",
    "d2":       lambda df, g, kf, : df["dimension"] == 2,
    "quantized": lambda df, g, kf, : (df["dimension"] <= 3) & (df["grid_size"] <= 1000),
}

LOGO_WHY = {
    "corr_hi": "drop k-dominant instances (f_k>=0.25): the ND analogue of LineNoise, "
               "correlated-then-CLIPPED axes with boundary pile-up",
    "corr_all": "drop every instance with f_k>=0.10 -- a much larger cut of the "
                "line-like family",
    "k_any_lowd": "drop every d<=10 instance containing ANY correlated-clipped axis: "
                  "the strictest removal of the line-like generator that leaves a "
                  "trainable corpus",
    "clus": "placebo: drop an unrelated generator (clustered). The LineNoise gain "
            "must NOT depend on this.",
    "d2": "drop the whole d=2 slice: does the 2D LineNoise gain need 2D training data?",
    "quantized": "drop the most lattice-like training data (d<=3, coarse integer grid) "
                 "-- the closest thing the corpus has to the 'grid' generator",
}


def stage_logo(force: bool = False, only: list | None = None) -> pd.DataFrame:
    import joblib

    os.makedirs(LOGO_DIR, exist_ok=True)
    params = json.load(open(FR.BEST_PARAMS))
    df, f30, fall = FR.load_corpus()
    d2 = FR.load_2d()
    d_te = df.loc[df["split"] == "test"]

    g = _nd_class_series(df)
    kf = _k_frac(df)

    cen, prev = generator_census(df)
    print("\n[a1] TRAINING-CORPUS GENERATOR CENSUS")
    print(cen.pivot(index="nd_class", columns="split", values="n").fillna(0).astype(int).to_string())
    print("\n[a1] prevalence of the 'k' (correlated+clipped) axis generator in TRAIN")
    print(prev.to_string(index=False))

    tags = list(LOGO_SPECS) if only is None else only
    rows, mech_rows = [], []
    for tag in tags:
        drop = LOGO_SPECS[tag](df, g, kf).to_numpy()
        keep = ~drop
        n_drop_tr = int((drop & (df["split"] == "train").to_numpy()).sum())
        n_drop_va = int((drop & (df["split"] == "val").to_numpy()).sum())
        print(f"\n[a1:{tag}] {LOGO_WHY[tag]}")
        print(f"[a1:{tag}] dropping {n_drop_tr} train + {n_drop_va} val rows "
              f"({100*n_drop_tr/ (df['split']=='train').sum():.1f}% of train)")

        sub = df.loc[keep].reset_index(drop=True)
        for name, feats in (("M0", f30), ("MF", fall)):
            p = os.path.join(LOGO_DIR, f"{tag}_{name}.joblib")
            if os.path.exists(p) and not force:
                m = joblib.load(p)
                print(f"[a1:{tag}:{name}] cached, best_iteration={m.best_iteration_}")
            else:
                m = FR.fit_model(sub, feats, params, f"{tag}:{name}")
                joblib.dump(m, p)

            e2 = FR.pct_err(m, d2, feats)
            ln_m = (d2["generator"] == "line_noise").to_numpy()
            gr_m = (d2["generator"] == "grid").to_numpy()
            base = {"holdout": tag, "model": name, "n_drop_train": n_drop_tr}
            rows.append({**base, "slice": "2D LineNoise", **FR.summarize(e2[ln_m])})
            rows.append({**base, "slice": "2D grid", **FR.summarize(e2[gr_m])})
            rows.append({**base, "slice": "2D overall", **FR.summarize(e2)})
            rows.append({**base, "slice": "ND test",
                         **FR.summarize(FR.pct_err(m, d_te, feats))})

            from scipy import stats as st
            ln = d2[(d2["generator"] == "line_noise") & (d2["n_customers"] >= 200)]
            ta = np.clip(ln["optimal_cost"] / ln["mst_total_length"], 1.0, 2.0).to_numpy()
            pa = alpha_pred(m, ln, feats)
            lr = st.linregress(ta, pa)
            mech_rows.append({"holdout": tag, "model": name, "N": len(ln),
                              "slope": float(lr.slope), "pearson_r": float(lr.rvalue),
                              "pred_alpha_std": float(np.std(pa, ddof=1))})

    res = pd.DataFrame(rows)
    mech = pd.DataFrame(mech_rows)
    res.to_csv(OUT_LOGO, index=False)
    mech.to_csv(OUT_LOGO_MECH, index=False)
    cen.to_csv(os.path.join(HERE, "adv_logo_census.csv"), index=False)
    prev.to_csv(os.path.join(HERE, "adv_logo_k_prevalence.csv"), index=False)

    print("\n[a1] RESULTS")
    print(res.pivot_table(index=["holdout", "slice"], columns="model",
                          values=["MAPE", "MSPE"]).round(4).to_string())
    print("\n[a1] LineNoise alpha slope (n>=200)")
    print(mech.pivot(index="holdout", columns="model", values="slope").round(4).to_string())
    return res


# ==========================================================================
# ATTACK 2 -- novel degenerate geometry (augmentation corpus)
# ==========================================================================

def _aug_worker(name: str) -> dict:
    import json as _json
    from feature_creator_v3 import compute_features_for_instance_v3
    try:
        inst = _json.load(open(os.path.join(AUG_INST, name + ".json")))
        sol = _json.load(open(os.path.join(AUG_DIR, "solutions", name + ".sol.json")))
        f = compute_features_for_instance_v3(inst, sol)
        if f is None:
            raise RuntimeError("extractor returned None")
        coords = np.unique(np.asarray(inst["coordinates"], dtype=np.float64), axis=0)
        f.update(FR._five(coords))
        ig = sol.get("integrity", {}) or {}
        f["sol_solver"] = sol.get("solver_used")
        f["float_tour_length"] = ig.get("float_tour_length")
        f["float_rel_dev"] = ig.get("float_rel_dev")
        f["float_within_tolerance"] = ig.get("float_within_tolerance")
        f["_ok"] = True
    except Exception as exc:                                    # noqa: BLE001
        f = {"instance_name": name, "_ok": False, "_error": repr(exc)}
    return f


def stage_augment_extract(force: bool = False) -> pd.DataFrame:
    if os.path.exists(OUT_AUG) and not force:
        d = pd.read_csv(OUT_AUG)
        print(f"[a2] cached {OUT_AUG} ({len(d)} rows)")
        return d

    names = sorted(f[:-5] for f in os.listdir(AUG_INST) if f.endswith(".json"))
    print(f"[a2] extracting 30+5 features for {len(names)} augmentation instances ...")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        rows = list(pool.map(_aug_worker, names, chunksize=8))
    d = pd.DataFrame(rows)
    bad = d[~d["_ok"].astype(bool)]
    if len(bad):
        print(bad[["instance_name", "_error"]].head(10).to_string())
        raise SystemExit("augmentation extraction errors")
    print(f"[a2] done in {time.perf_counter()-t0:.0f}s")

    # family / group / batch, from the solved records.  Later files supersede
    # earlier ones: repair_records re-solved 50 of the full_records entries.
    BATCH = {"pilot_records": "batch1", "pilot_records_v2": "batch1",
             "full_records": "batch1", "repair_records": "batch1",
             "batch2_records": "batch2"}
    meta = {}
    for src in ("pilot_records", "pilot_records_v2", "full_records", "batch2_records",
                "repair_records"):
        p = os.path.join(AUG_DIR, src + ".json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            if not r.get("written", True):
                continue
            meta[r["name"]] = {"family": r["family"], "group": r.get("group", ""),
                               "batch": BATCH[src], "source_file": src,
                               "record_alpha": r.get("alpha"), "rho": r.get("rho")}
    for c in ("family", "batch", "source_file", "record_alpha", "rho"):
        d[c] = d["instance_name"].map(lambda k, c=c: meta.get(k, {}).get(c))
    orphan = sorted(d.loc[d["family"].isna(), "instance_name"])
    if orphan:
        print(f"[a2] {len(orphan)} instance files have no solved record -- excluded: "
              f"{orphan}")
        d = d[d["family"].notna()].reset_index(drop=True)
    d["alpha_true"] = d["optimal_cost"] / d["mst_total_length"]
    # Alternative ground truth: the true Euclidean length of the returned tour.
    # `optimal_cost` is optimal for the ROUNDED integer matrix; on degenerate
    # geometry with a coarse scale factor the two disagree by up to ~2.7%.
    d["alpha_float"] = pd.to_numeric(d["float_tour_length"], errors="coerce") \
        / d["mst_total_length"]

    print(f"[a2] ground-truth quality: solver {d['sol_solver'].value_counts().to_dict()}; "
          f"float_rel_dev max {d['float_rel_dev'].max():.3e}, mean "
          f"{d['float_rel_dev'].mean():.3e}; outside 1e-3 tolerance "
          f"{int((~d['float_within_tolerance'].astype(bool)).sum())}/{len(d)}")

    # integrity: the freshly extracted 30 features must match the frozen file.
    # bounding_hypervolume overflows float64 in high d, so compare relatively.
    if os.path.exists(AUG_FEATS_V3):
        ref = pd.read_csv(AUG_FEATS_V3)
        m = d.merge(ref, on="instance_name", suffixes=("", "_ref"), how="inner")
        cols = [c for c in ref.columns
                if c + "_ref" in m.columns and pd.api.types.is_numeric_dtype(ref[c])]
        rel = {c: float(np.nanmax(np.abs(m[c] - m[c + "_ref"])
                                  / np.maximum(np.abs(m[c + "_ref"]), 1e-12)))
               for c in cols}
        worst_c = max(rel, key=rel.get)
        print(f"[a2] integrity vs augment_features_v3.csv: {len(m)} shared rows, "
              f"max RELATIVE diff over {len(cols)} numeric cols = {rel[worst_c]:.3e} "
              f"({worst_c})")

    d = d.drop(columns=[c for c in ("_ok", "_error") if c in d])
    d.to_csv(OUT_AUG, index=False)
    print(f"[a2] -> {OUT_AUG}")
    return d


def stage_augment_eval(force: bool = False) -> pd.DataFrame:
    d = stage_augment_extract(force)
    m0, mf, mft, f30, fall = load_models()
    models = {"M0": (m0, f30), "MF": (mf, fall)}
    if mft is not None:
        models["MF_tuned"] = (mft, fall)

    rows = []
    for tag, (m, feats) in models.items():
        pa = alpha_pred(m, d, feats)
        for gt in ("alpha_true", "alpha_float"):
            at = d[gt].to_numpy()
            e = (pa - at) / at
            good = np.isfinite(e)
            for (fam, batch), idx in d.groupby(["family", "batch"]).groups.items():
                sel = d.index.isin(idx) & good
                rows.append({"model": tag, "gt": gt, "family": fam, "batch": batch,
                             **FR.summarize(e[sel])})
            for batch, idx in d.groupby("batch").groups.items():
                sel = d.index.isin(idx) & good
                rows.append({"model": tag, "gt": gt, "family": "ALL", "batch": batch,
                             **FR.summarize(e[sel])})
            rows.append({"model": tag, "gt": gt, "family": "ALL", "batch": "ALL",
                         **FR.summarize(e[good])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_AUG_RES, index=False)

    main_gt = res[res.gt_ == "alpha_true"] if "gt_" in res else res[res["gt"] == "alpha_true"]
    piv = main_gt.pivot_table(index=["batch", "family"], columns="model",
                              values=["N", "MAPE", "MSPE"])
    print("\n[a2] AUGMENTATION CORPUS -- MAPE / MSPE by family x batch "
          "(ground truth = integer-optimal Concorde cost)")
    print(piv.round(4).to_string())

    alt = res[res["gt"] == "alpha_float"]
    print("\n[a2] same, with ground truth = float length of the optimal tour "
          "(rounding-robust)")
    print(alt.pivot_table(index=["batch", "family"], columns="model",
                          values=["MAPE", "MSPE"]).round(4).to_string())

    # --- does the gain concentrate where the geometry is actually degenerate?
    print("\n[a2] stratified by TRUE alpha (degeneracy proxy: alpha -> 2 is a "
          "one-dimensional set)")
    bins = [0.99, 1.10, 1.25, 1.50, 1.75, 2.01]
    d = d.assign(alpha_bin=pd.cut(d["alpha_true"], bins))
    srows = []
    for tag, (m, feats) in models.items():
        pa = alpha_pred(m, d, feats)
        e = (pa - d["alpha_true"].to_numpy()) / d["alpha_true"].to_numpy()
        for b, idx in d.groupby("alpha_bin", observed=True).groups.items():
            sel = d.index.isin(idx)
            srows.append({"model": tag, "alpha_bin": str(b), **FR.summarize(e[sel])})
    sdf = pd.DataFrame(srows)
    print(sdf.pivot_table(index="alpha_bin", columns="model",
                          values=["N", "MAPE", "MSPE"]).round(4).to_string())
    sdf.to_csv(os.path.join(HERE, "adv_augment_by_alpha.csv"), index=False)

    # --- is MF worth more than one constant fitted on the same family? -----
    # Oracle recalibration: the single multiplicative factor per family that
    # minimises MAPE, fitted ON the family being scored.  This is the most
    # generous possible level correction, so it upper-bounds what a level
    # shift alone can buy.
    print("\n[a2] oracle per-family multiplicative recalibration "
          "(fitted on the family it is scored on -- an upper bound on what a "
          "pure level shift can achieve)")
    rrows = []
    for fam, idx in d.groupby("family").groups.items():
        sel = d.index.isin(idx)
        at = d.loc[sel, "alpha_true"].to_numpy()
        row = {"family": fam, "N": int(sel.sum())}
        for tag, (m, feats) in models.items():
            pa = alpha_pred(m, d.loc[sel], feats)
            grid = np.linspace(0.8, 1.4, 6001)
            mapes = np.mean(np.abs((grid[:, None] * pa[None, :] - at[None, :]) / at),
                            axis=1) * 100.0
            j = int(np.argmin(mapes))
            row[f"{tag}_raw"] = float(np.mean(np.abs((pa - at) / at)) * 100.0)
            row[f"{tag}_recal"] = float(mapes[j])
            row[f"{tag}_factor"] = float(grid[j])
        rrows.append(row)
    rdf = pd.DataFrame(rrows)
    cols = ["family", "N", "M0_raw", "M0_recal", "M0_factor", "MF_raw", "MF_recal",
            "MF_factor"]
    print(rdf[[c for c in cols if c in rdf]].round(4).to_string(index=False))
    rdf.to_csv(os.path.join(HERE, "adv_augment_recal.csv"), index=False)
    return res


# ==========================================================================
# ATTACK 3 -- shortcut probe
# ==========================================================================

def _rot(d: int, seed: int) -> np.ndarray:
    """Deterministic random rotation, so no case is axis-aligned by accident."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def _fit_box(x: np.ndarray, span: float = 1000.0) -> np.ndarray:
    """Scale so the LONGEST axis spans `span`; keeps relative shape exactly."""
    x = x - x.min(axis=0)
    s = float(x.max())
    return x * (span / s) if s > 0 else x


def _arc_resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a densely sampled curve to n points equally spaced in arc length."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, j]) for j in range(pts.shape[1])], axis=1)


def probe_cases() -> dict:
    """Synthetic clouds where global elongation and LOCAL dimensionality disagree.

    slab_*   : high aspect ratio, locally d-dimensional  -> LOW alpha
    spiral_* : aspect ratio ~1, a single curled 1-D filament -> HIGH alpha
    liss_*   : aspect ratio ~1, a 1-D curve filling a d-cube -> HIGH alpha
    """
    C = {}

    def slab(n, d, lengths, seed):
        rng = np.random.default_rng(seed)
        x = rng.random((n, d)) * np.asarray(lengths, dtype=float)
        return _fit_box(x @ _rot(d, seed + 1).T)

    # --- controls -------------------------------------------------------
    C["ctl_iso_d2"] = slab(1000, 2, [1000, 1000], 11)
    C["ctl_iso_d3"] = slab(1000, 3, [1000, 1000, 1000], 12)
    rng = np.random.default_rng(13)
    ln = np.zeros((1000, 2))
    ln[:, 0] = np.sort(rng.random(1000)) * 1000.0
    ln[:, 1] = rng.standard_normal(1000) * 0.02
    C["ctl_collinear_d2"] = _fit_box(ln @ _rot(2, 14).T)

    # --- A: elongated but genuinely 2-D / 3-D ---------------------------
    #  transverse thickness / nn-spacing = sqrt(n*W/L); >=4 means locally 2-D
    C["slab_ar20_d2"] = slab(1000, 2, [1000, 50], 21)     # 7.1 spacings across
    C["slab_ar50_d2"] = slab(1000, 2, [1000, 20], 22)     # 4.5 spacings across
    C["slab_ar100_d2"] = slab(2000, 2, [1000, 10], 23)    # 4.5 spacings across
    C["slab_ar17_d3"] = slab(1000, 3, [1000, 60, 60], 24)

    # --- B: compact but locally one-dimensional -------------------------
    def spiral(n, turns, seed):
        th = np.linspace(0.05, 2 * np.pi * turns, 400_000)
        pts = np.stack([th * np.cos(th), th * np.sin(th)], axis=1)
        p = _arc_resample(pts, n)
        return _fit_box(p @ _rot(2, seed).T)

    C["spiral5_d2"] = spiral(1000, 5, 31)      # arm sep / spacing ~ 12
    C["spiral10_d2"] = spiral(1000, 10, 32)    # arm sep / spacing ~ 3

    def lissajous(n, d, seed):
        # distinct incommensurate frequencies -> the curve fills a d-cube
        freq = np.array([1.0 + 0.61803398875 * (j + 1) for j in range(d)])
        ph = np.linspace(0, 1, d, endpoint=False) * np.pi
        t = np.linspace(0, 2 * np.pi * 3.0, 400_000)
        pts = np.stack([np.sin(freq[j] * t + ph[j]) for j in range(d)], axis=1)
        p = _arc_resample(pts, n)
        return _fit_box(p @ _rot(d, seed).T)

    C["liss_d3"] = lissajous(1000, 3, 41)
    C["liss_d10"] = lissajous(1000, 10, 42)
    return C


def _solve_case(name: str, coords: np.ndarray) -> dict:
    from mst_utils import compute_mst
    from solvers.concorde import run_concorde_robust

    mst = compute_mst(coords)
    mst_len = float(np.sum(mst.edges))
    t0 = time.perf_counter()
    cost_int, _rt, tour, scale = run_concorde_robust(coords, 1000)
    tour = np.asarray(tour, dtype=int) - 1
    p = coords[tour]
    tour_len = float(np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)))
    # Optimality slack from integer rounding of the EXPLICIT matrix.
    slack = len(coords) * 0.5 / scale / tour_len
    return {"name": name, "n": int(len(coords)), "d": int(coords.shape[1]),
            "mst_total_length": mst_len, "optimal_cost": tour_len,
            "alpha_true": tour_len / mst_len,
            "concorde_int_cost": int(cost_int), "scale": float(scale),
            "rounding_slack_rel": float(slack),
            "solve_s": time.perf_counter() - t0}


def stage_probe(force: bool = False) -> pd.DataFrame:
    from feature_creator_v3 import compute_features_for_instance_v3

    cases = probe_cases()
    solved = json.load(open(PROBE_CACHE)) if (os.path.exists(PROBE_CACHE) and not force) else {}

    rows = []
    for name, coords in cases.items():
        coords = np.unique(coords, axis=0)
        if name not in solved:
            print(f"[a3] solving {name}  n={len(coords)} d={coords.shape[1]} ...", flush=True)
            solved[name] = _solve_case(name, coords)
            json.dump(solved, open(PROBE_CACHE, "w"), indent=1)
            print(f"      alpha_true={solved[name]['alpha_true']:.4f} "
                  f"({solved[name]['solve_s']:.0f}s)")
        s = solved[name]
        inst = {"instance_name": name, "coordinates": coords.tolist(),
                "dimension": int(coords.shape[1]), "grid_size": 1000}
        f = compute_features_for_instance_v3(inst, {"optimal_cost": s["optimal_cost"]})
        f.update(FR._five(coords))
        f["mst_total_length"] = s["mst_total_length"]
        f["alpha_true"] = s["alpha_true"]
        f["rounding_slack_rel"] = s["rounding_slack_rel"]
        # descriptive geometry, not model input
        rng_ = np.ptp(coords, axis=0)
        f["bbox_aspect_raw"] = float(rng_.max() / max(rng_.min(), 1e-12))
        rows.append(f)

    d = pd.DataFrame(rows)
    d.to_csv(OUT_PROBE, index=False)

    m0, mf, mft, f30, fall = load_models()
    models = {"M0": (m0, f30), "MF": (mf, fall)}
    if mft is not None:
        models["MF_tuned"] = (mft, fall)

    out = d[["instance_name", "n_customers", "dimension", "alpha_true",
             "aspect_ratio", "mst_topology_straightness",
             "degeneracy_pca_effective_rank", "local_id_evr1_median_k5",
             "local_id_pr_mean_k5", "mst_topology_deg2_straight_mean",
             "rounding_slack_rel"]].copy()
    for tag, (m, feats) in models.items():
        pa = alpha_pred(m, d, feats)
        out[f"alpha_{tag}"] = pa
        out[f"err_{tag}_pct"] = (pa - d["alpha_true"]) / d["alpha_true"] * 100.0
    out.to_csv(OUT_PROBE_RES, index=False)

    stage_stability()

    print("\n[a3] SHORTCUT PROBE (alpha_true from Concorde)")
    show = out.rename(columns={"instance_name": "case", "n_customers": "n",
                               "dimension": "d", "aspect_ratio": "AR",
                               "mst_topology_straightness": "straight",
                               "degeneracy_pca_effective_rank": "effrank",
                               "local_id_evr1_median_k5": "evr1",
                               "local_id_pr_mean_k5": "pr"})
    cols = ["case", "n", "d", "alpha_true", "AR", "straight", "effrank", "evr1", "pr",
            "alpha_M0", "alpha_MF", "err_M0_pct", "err_MF_pct"]
    print(show[cols].round(4).to_string(index=False))
    return out


def stage_stability() -> pd.DataFrame:
    """Are the MST-topology features well defined on degenerate geometry?

    On a perfect lattice the Euclidean MST is massively non-unique, so every
    MST-derived feature depends on tie-breaking.  Perturb the coordinates by a
    relative 1e-9 -- far below any physically meaningful scale -- and see how
    far the features and the prediction move.
    """
    d = pd.read_csv(OUT_AUG) if os.path.exists(OUT_AUG) else stage_augment_extract()
    m0, mf, mft, f30, fall = load_models()
    rng = np.random.default_rng(2024)

    rows = []
    for fam in ("lattice", "hexlattice", "collinear", "curve", "subspace"):
        names = d.loc[d["family"] == fam, "instance_name"].tolist()[:10]
        for nm in names:
            inst = json.load(open(os.path.join(AUG_INST, nm + ".json")))
            c = np.unique(np.asarray(inst["coordinates"], dtype=np.float64), axis=0)
            eps = 1e-9 * float(np.abs(c).max())
            c2 = c + rng.standard_normal(c.shape) * eps
            v1, v2 = FR._five(c), FR._five(np.unique(c2, axis=0))
            rows.append({"family": fam, "instance_name": nm,
                         **{k: abs(v1[k] - v2[k]) for k in NEW}})
    s = pd.DataFrame(rows)
    print("\n[a3b] MST-tie stability: max |feature change| under a relative-1e-9 "
          "coordinate jitter")
    print(s.groupby("family")[NEW].max().round(5).to_string())
    s.to_csv(os.path.join(HERE, "adv_stability.csv"), index=False)
    return s


def stage_signal() -> pd.DataFrame:
    """Does MF USE the new signal, or merely carry it?

    Fit two one-feature baselines on the TRAIN split only:
        LIN   alpha = a + b * mst_topology_straightness   (2 parameters)
        ISO   isotonic regression of alpha on straightness (monotone, 1 feature)
    Then score them on every degenerate slice next to M0 and MF.  If a
    one-feature rule matches or beats the 35-feature ensemble where the
    geometry is degenerate, the ensemble is not converting the feature into
    an alpha level.
    """
    from sklearn.isotonic import IsotonicRegression

    df, f30, fall = FR.load_corpus()
    tr = df[df["split"] == "train"]
    s_tr = tr["mst_topology_straightness"].to_numpy()
    a_tr = tr["alpha"].to_numpy()
    b, a0 = np.polyfit(s_tr, a_tr, 1)
    iso = IsotonicRegression(y_min=1.0, y_max=2.0, out_of_bounds="clip").fit(s_tr, a_tr)
    print(f"\n[a6] one-feature baselines fitted on TRAIN only: "
          f"LIN alpha = {a0:.4f} + {b:.4f}*straightness ; ISO isotonic")

    m0, mf, mft, _f30, _fall = load_models()

    def score(frame: pd.DataFrame, at: np.ndarray, label: str, rows: list):
        s = frame["mst_topology_straightness"].to_numpy()
        preds = {"M0": alpha_pred(m0, frame, f30), "MF": alpha_pred(mf, frame, fall),
                 "LIN": np.clip(a0 + b * s, 1.0, 2.0),
                 "ISO": np.clip(iso.predict(s), 1.0, 2.0)}
        for tag, pa in preds.items():
            e = (pa - at) / at
            rows.append({"slice": label, "model": tag, **FR.summarize(e)})

    rows = []
    d2 = FR.load_2d()
    a2d = np.clip(d2["optimal_cost"] / d2["mst_total_length"], 1.0, 2.0).to_numpy()
    for gen in ("line_noise", "grid"):
        m = (d2["generator"] == gen).to_numpy()
        score(d2[m], a2d[m], f"2D {gen}", rows)
    score(d2, a2d, "2D all", rows)
    te = df[df["split"] == "test"]
    score(te, te["alpha"].to_numpy(), "ND test", rows)

    if os.path.exists(OUT_AUG):
        ag = pd.read_csv(OUT_AUG)
        for fam, idx in ag.groupby("family").groups.items():
            sub = ag.loc[idx]
            score(sub, sub["alpha_true"].to_numpy(), f"aug {fam}", rows)
    if os.path.exists(OUT_PROBE):
        pr = pd.read_csv(OUT_PROBE)
        score(pr, pr["alpha_true"].to_numpy(), "probe all", rows)

    r = pd.DataFrame(rows)
    print("\n[a6] MAPE: 35-feature ensemble vs one-feature rules on straightness")
    print(r.pivot(index="slice", columns="model", values="MAPE").round(3)
          [["M0", "MF", "LIN", "ISO"]].to_string())
    print("\n[a6] MSPE (signed bias)")
    print(r.pivot(index="slice", columns="model", values="MSPE").round(3)
          [["M0", "MF", "LIN", "ISO"]].to_string())
    r.to_csv(os.path.join(HERE, "adv_signal.csv"), index=False)
    return r


def stage_support(force: bool = False) -> pd.DataFrame:
    """Feature problem, or training-support problem?

    The training split contains ZERO instances with alpha > 1.5 at n >= 200 and
    ZERO with straightness > 0.8 at n >= 200.  Every training instance with
    alpha > 1.7 has n <= 10.  So no tree can map "large, straight, high alpha"
    to a high alpha -- it has never seen the combination.

    Diagnostic: add the 874 augmentation instances (large n, straight, alpha up
    to 1.995) to TRAIN only, refit M0 and MF, and score on the 2D benchmark,
    which is disjoint from both corpora.  If M0+aug catches MF+aug, the deficit
    was support, not features.  These models are diagnostics; they are never
    used for any headline number, and the augmentation rows are added to the
    train split only -- never to val, and never to anything scored.
    """
    import joblib
    os.makedirs(LOGO_DIR, exist_ok=True)
    params = json.load(open(FR.BEST_PARAMS))
    df, f30, fall = FR.load_corpus()
    ag = pd.read_csv(OUT_AUG)

    add = ag.rename(columns={"alpha_true": "alpha"}).copy()
    add["split"] = "train"
    missing = [c for c in fall + ["alpha", "split", "mst_total_length", "optimal_cost"]
               if c not in add.columns]
    assert not missing, f"augmentation frame missing {missing}"
    add["alpha"] = add["alpha"].clip(1.0, 2.0)
    big = pd.concat([df, add[df.columns.intersection(add.columns)]],
                    ignore_index=True, sort=False)
    n_hi = int(((add["alpha"] > 1.5) & (add["n_customers"] >= 200)).sum())
    print(f"\n[a7] adding {len(add)} augmentation rows to TRAIN "
          f"({n_hi} of them are alpha>1.5 at n>=200, a combination the corpus "
          f"has 0 of); val and every evaluation slice untouched")

    d2 = FR.load_2d()
    ln_m = (d2["generator"] == "line_noise").to_numpy()
    gr_m = (d2["generator"] == "grid").to_numpy()
    at = np.clip(d2["optimal_cost"] / d2["mst_total_length"], 1.0, 2.0).to_numpy()
    from scipy import stats as st
    sel200 = ln_m & (d2["n_customers"] >= 200).to_numpy()

    rows = []
    for name, feats in (("M0+aug", f30), ("MF+aug", fall)):
        p = os.path.join(LOGO_DIR, f"support_{name.replace('+','_')}.joblib")
        if os.path.exists(p) and not force:
            m = joblib.load(p)
        else:
            m = FR.fit_model(big, feats, params, name)
            joblib.dump(m, p)
        e = FR.pct_err(m, d2, feats)
        pa = alpha_pred(m, d2, feats)
        lr = st.linregress(at[sel200], pa[sel200])
        rows.append({"model": name, "slice": "2D LineNoise", **FR.summarize(e[ln_m]),
                     "slope_n200": float(lr.slope)})
        rows.append({"model": name, "slice": "2D grid", **FR.summarize(e[gr_m]),
                     "slope_n200": np.nan})
        rows.append({"model": name, "slice": "2D overall", **FR.summarize(e),
                     "slope_n200": np.nan})
    r = pd.DataFrame(rows)
    print(r.round(4).to_string(index=False))
    r.to_csv(os.path.join(HERE, "adv_support.csv"), index=False)
    return r


# ==========================================================================
# ATTACK 4 -- cost honesty
# ==========================================================================

def stage_cost(force: bool = False, reps: int = 5) -> pd.DataFrame:
    if os.path.exists(OUT_COST) and not force:
        d = pd.read_csv(OUT_COST)
        print(f"[a4] cached {OUT_COST}")
        print(d.round(3).to_string(index=False))
        return d

    from feature_creator_v3 import compute_features_for_instance_v3
    from mst_utils import compute_mst
    from features_ext import group_degeneracy, group_local_id, group_mst_topology

    rows = []
    for d_ in (2, 50, 100):
        for n in (1000,):
            rng = np.random.default_rng(1000 * d_ + n)
            coords = np.unique(np.floor(rng.random((n, d_)) * 10000.0), axis=0)
            inst = {"instance_name": f"bench_n{n}_d{d_}", "coordinates": coords.tolist(),
                    "dimension": d_, "grid_size": 10000}
            sol = {"optimal_cost": 0.0}

            def timeit(fn, k=reps):
                ts = []
                for _ in range(k):
                    t0 = time.perf_counter()
                    fn()
                    ts.append((time.perf_counter() - t0) * 1000.0)
                return float(np.median(ts))

            t_base = timeit(lambda: compute_features_for_instance_v3(inst, sol))
            t_mst = timeit(lambda: compute_mst(coords))
            mst = compute_mst(coords)
            t_lid = timeit(lambda: group_local_id.compute(coords))
            t_deg = timeit(lambda: group_degeneracy.compute(coords))
            t_top = timeit(lambda: group_mst_topology.compute(coords, mst))
            t_asrun = timeit(lambda: FR._five(coords))

            rows.append({"n": n, "d": d_, "base30_ms": t_base, "mst_only_ms": t_mst,
                         "local_id_ms": t_lid, "degeneracy_ms": t_deg,
                         "mst_topology_ms": t_top,
                         "marginal_shared_mst_ms": t_lid + t_deg + t_top,
                         "as_run_ms": t_asrun,
                         "marginal_pct_of_base": (t_lid + t_deg + t_top) / t_base * 100.0,
                         "as_run_pct_of_base": t_asrun / t_base * 100.0})
            print(f"[a4] n={n} d={d_}: base30={t_base:.1f}ms  marginal(shared MST)="
                  f"{t_lid+t_deg+t_top:.1f}ms ({rows[-1]['marginal_pct_of_base']:.1f}%)"
                  f"  as-run={t_asrun:.1f}ms", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT_COST, index=False)
    print("\n[a4] synthetic, n=1000, single process")
    print(d.round(2).to_string(index=False))

    stage_cost_tsplib(force)
    return d


def _tsplib_cost_worker(name: str) -> dict:
    """Base-30 and marginal cost on one real TSPLIB EUC_2D instance.

    Both are measured back to back in the SAME process so the ratio does not
    depend on whatever machine produced the shipped timing log.  Run
    single-process: 20-way contention roughly doubles every number here.
    """
    from mst_utils import compute_mst
    from feature_creator_v3 import compute_features_for_instance_v3
    from features_ext import group_degeneracy, group_local_id, group_mst_topology
    sys.path.insert(0, FR.TSPLIB_DIR)
    from tsplib_parser import parse_tsplib_file

    info = parse_tsplib_file(os.path.join(FR.TSPLIB_INST, name + ".tsp"))
    raw = np.asarray(info["raw_coords"], dtype=np.float64)
    coords = np.unique(raw, axis=0)
    inst = {"instance_name": name, "coordinates": raw.tolist(),
            "dimension": 2, "grid_size": 0}

    t0 = time.perf_counter()
    compute_features_for_instance_v3(inst, {"optimal_cost": 0.0})
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter(); mst = compute_mst(coords); t_mst = time.perf_counter() - t0
    t0 = time.perf_counter(); group_local_id.compute(coords); t_l = time.perf_counter() - t0
    t0 = time.perf_counter(); group_degeneracy.compute(coords); t_d = time.perf_counter() - t0
    t0 = time.perf_counter(); group_mst_topology.compute(coords, mst); t_t = time.perf_counter() - t0
    return {"instance": name, "n_pts": int(len(coords)), "base30_s": t_base,
            "mst_s": t_mst, "local_id_s": t_l, "degeneracy_s": t_d,
            "mst_topology_s": t_t, "marginal_s": t_l + t_d + t_t}


def stage_cost_tsplib(force: bool = False) -> pd.DataFrame:
    """Recompute the paper's 82.6% split with the new block added.

    The paper's figure is the MEAN over LGBM_V3 rows with edge_weight_type
    EUC_2D in tsplib_benchmark/results/all_models_tsplib.csv; reproduced here
    to 82.57 / 16.28 / 1.15 before anything is added.
    """
    p = os.path.join(HERE, "adv_cost_tsplib.csv")
    log = os.path.join(ROOT, "tsplib_benchmark", "results", "all_models_tsplib.csv")
    t = pd.read_csv(log)
    g = t[(t["model"] == "LGBM_V3") & (t["edge_weight_type"] == "EUC_2D")].copy()
    ft, it, tt = (g["feature_time_s"].mean(), g["inference_time_s"].mean(),
                  g["total_time_s"].mean())
    print(f"\n[a4] paper baseline reproduced: N={len(g)} EUC_2D, feature "
          f"{ft/tt*100:.2f}%  inference {it/tt*100:.2f}%  residual {(tt-ft-it)/tt*100:.2f}%"
          f"  (paper: 82.6 / 16.3 / 1.2)")

    if os.path.exists(p) and not force:
        m = pd.read_csv(p)
    else:
        names = g["instance"].astype(str).tolist()
        print(f"[a4] timing base-30 and the new block on all {len(names)} EUC_2D "
              f"instances, SINGLE PROCESS ...", flush=True)
        m = pd.DataFrame([_tsplib_cost_worker(nm) for nm in names])
        m.to_csv(p, index=False)

    j = g.merge(m, on="instance", how="inner", validate="one_to_one")
    # (a) ratio measured entirely in this environment
    r_here = j["marginal_s"].sum() / j["base30_s"].sum()
    print(f"[a4] same-environment measurement on the 78 EUC_2D instances: "
          f"base-30 mean {j['base30_s'].mean()*1000:.1f} ms, new block mean "
          f"{j['marginal_s'].mean()*1000:.1f} ms, median "
          f"{j['marginal_s'].median()*1000:.1f} ms, max "
          f"{j['marginal_s'].max()*1000:.0f} ms "
          f"(n={int(j.loc[j['marginal_s'].idxmax(),'n_pts'])})")
    print(f"[a4] feature-extraction time multiplier, measured here: "
          f"x{1+r_here:.2f} (marginal is {r_here*100:.0f}% of base-30)")

    # (b) apply that multiplier to the shipped log the paper's number came from
    ft2 = ft * (1 + r_here)
    tt2 = tt + (ft2 - ft)
    print(f"[a4] paper split rescaled by the measured multiplier: feature "
          f"{ft2/tt2*100:.2f}%  inference {it/tt2*100:.2f}%  residual "
          f"{(tt2-ft2-it)/tt2*100:.2f}%   (was 82.57 / 16.28 / 1.15)")
    print(f"[a4] mean total prediction time {tt*1000:.1f} ms -> {tt2*1000:.1f} ms "
          f"({tt2/tt:.2f}x)")
    j[["instance", "n_pts", "base30_s", "feature_time_s", "inference_time_s",
       "total_time_s", "local_id_s", "degeneracy_s", "mst_topology_s",
       "marginal_s"]].to_csv(os.path.join(HERE, "adv_cost_tsplib_joined.csv"), index=False)
    return j


# ==========================================================================
# ATTACK 5 -- leakage audit
# ==========================================================================

BANNED_NAMES = {"optimal_cost", "tour", "alpha", "solution", "sol_data", "concorde",
                "lkh", "held_karp", "record_alpha", "opt_cost", "tsp_cost"}
BANNED_RNG = {"random", "rand", "randn", "shuffle", "permutation", "choice",
              "default_rng", "RandomState", "seed", "sample", "uniform"}


def _audit_source(path: str) -> dict:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    hits_target, hits_rng, imports = [], [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            hits_target.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            hits_target.append((node.lineno, node.attr))
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_RNG:
            hits_rng.append((node.lineno, node.attr))
        elif isinstance(node, ast.Name) and node.id in BANNED_RNG:
            hits_rng.append((node.lineno, node.id))
    # code-only view: strip docstrings/comments before the textual scan
    code_lines = [ln.split("#")[0] for ln in src.splitlines()]
    text_hits = {w: sum(w in ln for ln in code_lines) for w in
                 ("optimal_cost", "tour", "solve", "solution")}
    return {"file": os.path.relpath(path, ROOT), "sha256": hashlib.sha256(
        src.encode()).hexdigest()[:16], "n_lines": len(src.splitlines()),
        "imports": sorted(imports), "target_symbol_hits": hits_target,
        "rng_symbol_hits": hits_rng, "text_hits_in_code": text_hits}


def stage_leak() -> dict:
    out = {"static": [], "data": {}, "determinism": {}}
    for f in ("group_local_id.py", "group_degeneracy.py", "group_mst_topology.py",
              "__init__.py"):
        out["static"].append(_audit_source(os.path.join(ROOT, "features_ext", f)))

    print("\n[a5] STATIC AUDIT of features_ext/")
    for s in out["static"]:
        print(f"  {s['file']:<40} {s['n_lines']:>4} lines  sha {s['sha256']}")
        print(f"      imports: {', '.join(s['imports']) or '(none)'}")
        print(f"      target-symbol hits: {s['target_symbol_hits'] or 'NONE'}")
        print(f"      rng-symbol hits:    {s['rng_symbol_hits'] or 'NONE'}")
        print(f"      code-text hits:     {s['text_hits_in_code']}")

    # --- data-level: nothing evaluated on may appear in a training split ---
    corpus = pd.read_csv(FR.ND_FEATS, usecols=["instance_name", "split"])
    train_names = set(corpus.loc[corpus.split == "train", "instance_name"])
    fit_names = set(corpus.loc[corpus.split.isin(["train", "val"]), "instance_name"])
    all_names = set(corpus["instance_name"])

    aug = sorted(f[:-5] for f in os.listdir(AUG_INST) if f.endswith(".json"))
    bench = FR.load_2d()["instance_name"].astype(str).tolist()
    tsplib = pd.read_csv(FR.TSPLIB_FEATS)["instance_name"].astype(str).tolist()

    for label, names in (("augmentation", aug), ("2D benchmark", bench), ("TSPLIB", tsplib)):
        ov_t = sorted(set(names) & train_names)
        ov_f = sorted(set(names) & fit_names)
        ov_a = sorted(set(names) & all_names)
        out["data"][label] = {"n": len(names), "in_train": len(ov_t),
                              "in_train_or_val": len(ov_f), "in_corpus_at_all": len(ov_a),
                              "examples": ov_a[:5]}
        print(f"[a5] {label:<14} n={len(names):<5} overlap with train={len(ov_t)}  "
              f"train+val={len(ov_f)}  corpus(any split)={len(ov_a)}")

    # --- the 5 columns used for training came from instances/*.bin only ----
    ext = pd.read_csv(FR.EXT_CSV)
    out["data"]["features_extended_rows"] = len(ext)
    out["data"]["features_extended_matches_corpus"] = bool(
        set(ext["instance_name"]) == all_names)
    print(f"[a5] features_extended.csv rows={len(ext)}  "
          f"name set == corpus: {out['data']['features_extended_matches_corpus']}")

    # --- determinism: recompute a sample twice, and from a shuffled input --
    rng = np.random.default_rng(7)
    sample = list(rng.choice(sorted(all_names), size=40, replace=False))
    a = pd.DataFrame([FR._worker_corpus(nm) for nm in sample]).set_index("instance_name")
    b = pd.DataFrame([FR._worker_corpus(nm) for nm in sample]).set_index("instance_name")
    same = float(np.nanmax(np.abs(a[NEW].to_numpy() - b[NEW].to_numpy())))
    ref = pd.read_csv(FR.EXT_CSV).set_index("instance_name").loc[sample, NEW]
    vs_file = float(np.nanmax(np.abs(a[NEW].to_numpy() - ref.to_numpy())))

    # row-permutation invariance, raw and after the canonicalisation every
    # production caller applies (np.unique sorts lexicographically)
    perm_worst = perm_canon = 0.0
    perm_by_n = {}
    for nm in sample[:24]:
        c = FR._load_bin(nm)
        v1 = FR._five(c)
        for s in (3, 11):
            p = np.random.default_rng(s).permutation(len(c))
            dr = max(abs(v1[k] - FR._five(c[p])[k]) for k in NEW)
            dc = max(abs(v1[k] - FR._five(np.unique(c[p], axis=0))[k]) for k in NEW)
            perm_worst, perm_canon = max(perm_worst, dr), max(perm_canon, dc)
            perm_by_n[len(c)] = max(perm_by_n.get(len(c), 0.0), dr)
    # scale invariance
    scale_worst = 0.0
    for nm in sample[:12]:
        c = FR._load_bin(nm)
        v1, v2 = FR._five(c), FR._five(c * 7.13 + 41.0)
        scale_worst = max(scale_worst, max(abs(v1[k] - v2[k]) for k in NEW))

    out["determinism"] = {"n_sample": len(sample), "repeat_max_diff": same,
                          "vs_training_file_max_diff": vs_file,
                          "row_permutation_max_diff_raw": perm_worst,
                          "row_permutation_max_diff_canonicalised": perm_canon,
                          "row_permutation_by_n": perm_by_n,
                          "scale_shift_max_diff": scale_worst}
    print(f"[a5] determinism: repeat max diff {same:.3e}; vs training file {vs_file:.3e}; "
          f"scale+shift {scale_worst:.3e}")
    print(f"[a5] row-permutation: raw {perm_worst:.3e}, after np.unique canonicalisation "
          f"{perm_canon:.3e}  (worst n: "
          f"{sorted(perm_by_n.items(), key=lambda kv: -kv[1])[:4]})")

    json.dump(out, open(OUT_LEAK, "w"), indent=2, default=str)
    print(f"[a5] -> {OUT_LEAK}")
    return out


# ==========================================================================
# driver
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["a1", "logo", "a2", "augment", "a3", "probe", "a4", "cost",
                             "a5", "leak", "a6", "signal", "a7", "support", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated LOGO tags")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None

    if args.stage in ("a5", "leak", "all"):
        stage_leak()
    if args.stage in ("a4", "cost", "all"):
        stage_cost(args.force)
    if args.stage in ("a2", "augment", "all"):
        stage_augment_eval(args.force)
    if args.stage in ("a3", "probe", "all"):
        stage_probe(args.force)
    if args.stage in ("a1", "logo", "all"):
        stage_logo(args.force, only)
    if args.stage in ("a6", "signal", "all"):
        stage_signal()
    if args.stage in ("a7", "support", "all"):
        stage_support(args.force)


if __name__ == "__main__":
    main()
