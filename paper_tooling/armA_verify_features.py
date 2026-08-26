"""Independent re-extraction of the 31 GART 2.0 features from raw instance files.

Deliberately does NOT import support_arms_features.py, support_arms_eval.py or
support_arms_study.py, and does not read v4_study_feature_cache.csv. Everything
is rebuilt from:

    instances/*.bin                              (nd_test)
    Generalized_TSP_Analysis/instances/*.json    (bench2d)
    augment/instances/*.json                     (augment)
    tsplib_benchmark/instances/*.tsp             (tsplib_euc2d / tsplib_noneuc)

using the canonical production extractor
``lgbm_model_v3/feature_engineering_gart2.compute_features`` on the native path
and the production hybrid builder
``tsplib_benchmark/run_all_models_tsplib._hybrid_estimate_generic`` on the
non-Euclidean path (harvested through a shim so this script -- not the
production wrapper -- owns the target transform).

Writes only paper_tooling/armA_verify_*.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
for _p in (ROOT, ROOT / "lgbm_model_v3", ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

N_WORKERS = 8
MAX_MDS_DIM = 100

FEATS31 = [
    "n_customers", "dimension", "log_bounding_hypervolume", "bounding_hypervolume",
    "log_node_density", "node_density", "aspect_ratio", "centroid_dist_mean",
    "centroid_dist_std", "centroid_dist_max", "centroid_dist_iqr", "mst_edge_mean",
    "mst_edge_std", "mst_edge_skew", "mst_edge_kurtosis", "mst_edge_max",
    "mst_edge_q10", "mst_edge_q25", "mst_edge_q50", "mst_edge_q75", "mst_edge_q90",
    "mst_dominance_ratio", "mst_gap_ratio", "mst_leaf_ratio", "mst_degree_mean",
    "mst_degree_std", "mst_degree_max", "mst_diameter", "mst_diameter_normalized",
    "large_edge_count", "greedy_nn_over_mst",
]
OUTCOLS = ["instance", "status", "mode", "mst_total_length"] + FEATS31


# ---------------------------------------------------------------------------
# coordinate loaders (native path)
# ---------------------------------------------------------------------------
def coords_nd(name: str):
    with open(ROOT / "instances" / f"{name}.bin", "rb") as f:
        n, d, _g = struct.unpack("III", f.read(12))
        dist_len = struct.unpack("I", f.read(4))[0]
        f.read(dist_len)
        buf = f.read(n * d * 4)
    return np.frombuffer(buf, dtype=np.float32).reshape(n, d), int(d)


def coords_bench2d(name: str):
    j = json.loads((ROOT / "Generalized_TSP_Analysis" / "instances" / f"{name}.json")
                   .read_text(encoding="utf-8"))
    c = np.asarray(j["coordinates"], dtype=np.float32)
    return c, int(j.get("dimension", c.shape[1]))


def coords_augment(name: str):
    j = json.loads((ROOT / "augment" / "instances" / f"{name}.json")
                   .read_text(encoding="utf-8"))
    c = np.asarray(j["coordinates"], dtype=np.float32)
    return c, int(j.get("dimension", c.shape[1]))


NATIVE_LOADERS = {"nd_test": coords_nd, "bench2d": coords_bench2d,
                  "augment": coords_augment}


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
class _FeatureShim:
    """Stands in for TSP_GART2_Estimator so the hybrid builder hands us its
    feature dict instead of a prediction. The transform stays in our hands."""

    def __init__(self):
        self.features_required = list(FEATS31)
        self.captured = None

    def predict_alpha(self, feats):
        self.captured = dict(feats)
        return 1.5


def _row_native(stratum: str, name: str) -> dict:
    from feature_engineering_gart2 import compute_features

    coords, dim = NATIVE_LOADERS[stratum](name)
    # Exactly what TSP_GART2_Estimator.estimate() does before extraction.
    coords = np.unique(np.asarray(coords, dtype=np.float32), axis=0)
    if coords.shape[0] < 3:
        return {"instance": name, "status": "n<3", "mode": "native"}
    f = compute_features(coords, dim)
    out = {"instance": name, "status": "ok", "mode": "native"}
    out.update({k: float(f[k]) for k in FEATS31})
    out["mst_total_length"] = float(f["mst_total_length"])
    return out


def _row_tsplib(name: str) -> dict:
    from classical_mds import classical_mds
    from feature_engineering_gart2 import compute_features
    from tsplib_parser import parse_tsplib_file
    import run_all_models_tsplib as R

    info = parse_tsplib_file(str(ROOT / "tsplib_benchmark" / "instances" / f"{name}.tsp"))
    native = bool(info["is_native_euclidean"]) and info["raw_coords"] is not None
    out = {"instance": name, "edge_weight_type": str(info["edge_weight_type"]),
           "n_parsed": int(info["n"])}
    if native:
        coords = np.asarray(info["raw_coords"], dtype=np.float32)
        coords = np.unique(coords, axis=0)
        f = compute_features(coords, coords.shape[1])
        out.update({"status": "ok", "mode": "native",
                    "mst_total_length": float(f["mst_total_length"])})
        out.update({k: float(f[k]) for k in FEATS31})
        return out

    D = info["distance_matrix"]
    X, _e, _m = classical_mds(D, max_dim=MAX_MDS_DIM)
    shim = _FeatureShim()
    res = R._hybrid_estimate_generic(shim, D, X, X.shape[1])
    out["mode"] = "hybrid"
    out["status"] = res.get("status", "ok")
    out["mst_total_length"] = float(res["mst_length"])
    if shim.captured is not None:
        out.update({k: float(shim.captured[k]) for k in FEATS31})
    return out


def _one(task) -> dict:
    stratum, name = task
    try:
        if stratum == "tsplib":
            r = _row_tsplib(name)
        else:
            r = _row_native(stratum, name)
    except Exception as exc:  # noqa: BLE001
        r = {"instance": name, "status": f"error:{type(exc).__name__}",
             "_error": repr(exc)}
    r["stratum"] = stratum
    return r


def run(stratum: str, names, out: Path, force: bool) -> pd.DataFrame:
    if out.exists() and not force:
        df = pd.read_csv(out)
        print(f"[{stratum}] cached {out.name} ({len(df)} rows)", flush=True)
        return df
    print(f"[{stratum}] extracting {len(names)} instances on {N_WORKERS} workers",
          flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, r in enumerate(pool.map(_one, [(stratum, n) for n in names],
                                       chunksize=16), 1):
            rows.append(r)
            if i % 2000 == 0:
                el = time.perf_counter() - t0
                print(f"    {i}/{len(names)}  eta {el / i * (len(names) - i):.0f}s",
                      flush=True)
    df = pd.DataFrame(rows)
    bad = df[df["status"].astype(str).str.startswith("error")]
    print(f"[{stratum}] done, hard errors={len(bad)}", flush=True)
    if len(bad):
        print(bad[["instance", "_error"]].head(10).to_string())
    df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["nd_test", "bench2d", "augment", "tsplib", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.stage in ("bench2d", "all"):
        names = sorted(p.stem for p in
                       (ROOT / "Generalized_TSP_Analysis" / "instances").glob("*.json"))
        if a.limit:
            names = names[:a.limit]
        run("bench2d", names, HERE / "armA_verify_feats_bench2d.csv", a.force)

    if a.stage in ("augment", "all"):
        names = sorted(p.stem for p in (ROOT / "augment" / "instances").glob("*.json"))
        if a.limit:
            names = names[:a.limit]
        run("augment", names, HERE / "armA_verify_feats_augment.csv", a.force)

    if a.stage in ("tsplib", "all"):
        names = sorted(p.stem for p in
                       (ROOT / "tsplib_benchmark" / "instances").glob("*.tsp"))
        if a.limit:
            names = names[:a.limit]
        run("tsplib", names, HERE / "armA_verify_feats_tsplib.csv", a.force)

    if a.stage in ("nd_test", "all"):
        v4 = pd.read_csv(ROOT / "tsp_features_v4.csv", usecols=["instance_name", "split"])
        names = sorted(v4.loc[v4.split == "test", "instance_name"].astype(str))
        if a.limit:
            names = names[:a.limit]
        run("nd_test", names, HERE / "armA_verify_feats_nd_test.csv", a.force)


if __name__ == "__main__":
    main()
