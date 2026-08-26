"""Remaining checks: R0 numeric identity, monotonicity probe, and the
per-stratum blast radius of the cache-vs-production feature divergence.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"

FEATS31 = list(joblib.load(ROOT / "lgbm_model_v3" / "gart2_final.joblib").feature_name())
CLIP = (1.0, 2.0)
PROBE_TOL = 1e-9
SEED = 42


def alpha(z):
    return 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def log_int_grid(lo, hi, k):
    return [int(v) for v in
            np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), k)).astype(int))]


def main():
    frozen = joblib.load(ROOT / "lgbm_model_v3" / "gart2_final.joblib")
    r0 = joblib.load(HERE / "support_arms_models" / "R0.joblib")
    A = joblib.load(HERE / "support_arms_models" / "A.joblib")

    # ---- R0 identity -----------------------------------------------------
    print("=========== R0 vs FROZEN ===========")
    for tag, b in (("FROZEN", frozen), ("R0", r0)):
        s = b.model_to_string()
        print(f"  {tag:7s} trees={b.num_trees()} best_iter={b.best_iteration} "
              f"nfeat={b.num_feature()} sha256(model_to_string)="
              f"{hashlib.sha256(s.encode()).hexdigest()[:32]}")
    fp = ROOT / "lgbm_model_v3" / "gart2_final.joblib"
    rp = HERE / "support_arms_models" / "R0.joblib"
    print(f"  file sha256 equal: "
          f"{hashlib.sha256(fp.read_bytes()).hexdigest() == hashlib.sha256(rp.read_bytes()).hexdigest()}")

    nd = pd.read_csv(HERE / "armA_verify_feats_nd_test.csv")
    b2 = pd.read_csv(HERE / "armA_verify_feats_bench2d.csv")
    ag = pd.read_csv(HERE / "armA_verify_feats_augment.csv")
    tl = pd.read_csv(HERE / "armA_verify_feats_tsplib.csv")
    big = pd.concat([nd, b2, ag, tl], ignore_index=True)
    big = big[big[FEATS31].notna().all(axis=1)]
    pf = frozen.predict(big[FEATS31], num_iteration=frozen.best_iteration)
    pr = r0.predict(big[FEATS31], num_iteration=r0.best_iteration)
    print(f"  scored set n={len(big)}   max |pred_R0 - pred_frozen| = "
          f"{np.max(np.abs(pf - pr)):.3e}   exact ties: {int((pf == pr).sum())}/{len(big)}")

    # ---- monotonicity probe ---------------------------------------------
    print("\n=========== MONOTONICITY PROBE ===========")
    base = nd[nd.status == "ok"].sample(1000, random_state=SEED).copy()
    rows = []
    for tag, b in (("FROZEN", frozen), ("A", A)):
        for col, grid in (("dimension", log_int_grid(2, 200, 24)),
                          ("n_customers", log_int_grid(5, 4000, 24))):
            P = np.empty((len(base), len(grid)))
            for j, g in enumerate(grid):
                X = base.copy()
                X[col] = g
                P[:, j] = b.predict(X[FEATS31], num_iteration=b.best_iteration)
            for kind, M in (("raw", P), ("deployed", alpha(P))):
                d = np.diff(M, axis=1)
                viol = d > PROBE_TOL
                rows.append({"model": tag, "swept": col, "kind": kind,
                             "pct_nonincr": float((~viol.any(axis=1)).mean()) * 100.0,
                             "n_viol": int(viol.sum()),
                             "viol_max": float(d[viol].max()) if viol.any() else 0.0})
    P = pd.DataFrame(rows)
    P.to_csv(HERE / "armA_verify_probe.csv", index=False)
    print(P.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # ---- blast radius ----------------------------------------------------
    print("\n=========== CACHE vs PRODUCTION-FAITHFUL FEATURES ===========")
    C = pd.read_csv(HERE / "v4_study_feature_cache.csv", low_memory=False)
    src = {"nd_test": nd, "bench2d": b2, "augment": ag,
           "tsplib_euc2d": tl, "tsplib_noneuc": tl}
    out = []
    for st, M in src.items():
        c = C[(C.stratum == st) & (C.status == "ok")]
        j = c.merge(M[M.status == "ok"], on="instance", suffixes=("_c", "_m"))
        if not len(j):
            continue
        fc = j[[f + "_c" for f in FEATS31] + ["mst_total_length_c"]].rename(
            columns=lambda s: s[:-2])
        fm = j[[f + "_m" for f in FEATS31] + ["mst_total_length_m"]].rename(
            columns=lambda s: s[:-2])
        pc = np.clip(alpha(frozen.predict(fc[FEATS31], num_iteration=frozen.best_iteration)),
                     *CLIP) * fc["mst_total_length"].to_numpy()
        pm = np.clip(alpha(frozen.predict(fm[FEATS31], num_iteration=frozen.best_iteration)),
                     *CLIP) * fm["mst_total_length"].to_numpy()
        rel = np.abs(pc - pm) / pc * 100.0
        dn = (j["n_customers_c"] != j["n_customers_m"]).sum()
        out.append({"stratum": st, "n": len(j),
                    "n_pred_differ": int((rel > 1e-6).sum()),
                    "pct_differ": round(float((rel > 1e-6).mean() * 100), 1),
                    "max_rel_pct": round(float(rel.max()), 4),
                    "mean_rel_pct": round(float(rel.mean()), 5),
                    "n_with_different_n_customers": int(dn)})
    B = pd.DataFrame(out)
    B.to_csv(HERE / "armA_verify_blast_radius.csv", index=False)
    print(B.to_string(index=False))


if __name__ == "__main__":
    main()
