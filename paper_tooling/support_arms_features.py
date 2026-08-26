"""Recompute the three arm-B features on every corpus the study scores on.

The three kept features (SHAP ranks 2/3/11 in the prior screen, 27.3% of
attribution, +32% extraction cost at d=2 and +9.8% at d=100):

    mst_topology_straightness
    mst_topology_deg2_straight_mean
    degeneracy_pca_effective_rank

``group_mst_topology`` was repaired for this study (canonical point ordering,
canonical tie-broken MST, tolerance-aware diameter selection), so its two
features MUST be recomputed rather than read from ``features_extended.csv``.
``group_degeneracy`` was not touched, so its one feature is reused wherever a
cached copy exists and recomputed only where none does.

Corpora:
    corpus    106,272 rows of tsp_features_v3/v4.csv, from instances/*.bin
    augment       875 solved augmentation instances, from augment/instances
    bench2d     2,580 2D benchmark instances
    tsplib        111 TSPLIB instances, native or classical-MDS hybrid

Every stage is cached; ``--force`` recomputes. Writes only under paper_tooling/.
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
for _p in (ROOT, ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Another agent owns the production benchmark; cap ourselves well under it.
N_WORKERS = 10
MAX_MDS_DIM = 100

TOPO2 = ["mst_topology_straightness", "mst_topology_deg2_straight_mean"]
DEGEN1 = ["degeneracy_pca_effective_rank"]
ARM_B_FEATURES = TOPO2 + DEGEN1

OUT_CORPUS = HERE / "support_arms_feats_corpus.csv"
OUT_AUGMENT = HERE / "support_arms_feats_augment.csv"
OUT_BENCH2D = HERE / "support_arms_feats_bench2d.csv"
OUT_TSPLIB = HERE / "support_arms_feats_tsplib.csv"


# --------------------------------------------------------------------------
# coordinate loaders
# --------------------------------------------------------------------------
def load_bin(name: str) -> np.ndarray:
    """De-duplicated raw coordinates from instances/*.bin (as feature_retrain)."""
    with open(ROOT / "instances" / f"{name}.bin", "rb") as f:
        n, d, _grid = struct.unpack("III", f.read(12))
        dist_len = struct.unpack("I", f.read(4))[0]
        f.read(dist_len)
        buf = f.read(n * d * 4)
    coords = np.frombuffer(buf, dtype=np.float32).reshape(n, d)
    return np.unique(coords, axis=0).astype(np.float64)


def load_augment(name: str) -> np.ndarray:
    j = json.loads((ROOT / "augment" / "instances" / f"{name}.json")
                   .read_text(encoding="utf-8"))
    return np.unique(np.asarray(j["coordinates"], dtype=np.float64), axis=0)


def load_bench2d(name: str) -> np.ndarray:
    j = json.loads((ROOT / "Generalized_TSP_Analysis" / "instances" / f"{name}.json")
                   .read_text(encoding="utf-8"))
    for key in ("coordinates", "coords", "points", "node_coords"):
        if key in j:
            return np.unique(np.asarray(j[key], dtype=np.float64), axis=0)
    raise KeyError(name)


def load_tsplib(name: str) -> np.ndarray:
    """Whatever cloud the estimator actually sees: raw nodes, or the MDS embedding."""
    from tsplib_parser import parse_tsplib_file
    from classical_mds import classical_mds

    info = parse_tsplib_file(str(ROOT / "tsplib_benchmark" / "instances" / f"{name}.tsp"))
    if info["is_native_euclidean"] and info["raw_coords"] is not None:
        coords = np.asarray(info["raw_coords"], dtype=np.float64)
    else:
        X, _e, _m = classical_mds(info["distance_matrix"], max_dim=MAX_MDS_DIM)
        coords = np.asarray(X, dtype=np.float64)
    return np.unique(coords, axis=0)


LOADERS = {"corpus": load_bin, "augment": load_augment,
           "bench2d": load_bench2d, "tsplib": load_tsplib}


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
_G: dict = {}


def _groups():
    if not _G:
        from features_ext import group_degeneracy, group_mst_topology
        _G["top"] = group_mst_topology
        _G["deg"] = group_degeneracy
    return _G["top"], _G["deg"]


def _one(args) -> dict:
    which, name, need_degen = args
    row: dict = {"instance_name": name}
    try:
        g_top, g_deg = _groups()
        coords = LOADERS[which](name)
        # Let the topology group canonicalise and build its own MST: that is
        # the repaired path, and handing it a tree built on non-canonical
        # coordinates would bypass the whole fix.
        row.update({k: float(v) for k, v in g_top.compute(coords).items()
                    if k in TOPO2})
        if need_degen:
            row.update({k: float(v) for k, v in g_deg.compute(coords).items()
                        if k in DEGEN1})
    except Exception as exc:  # noqa: BLE001
        for k in TOPO2 + (DEGEN1 if need_degen else []):
            row[k] = np.nan
        row["_error"] = repr(exc)
    return row


def run(which: str, names: list[str], out: Path, need_degen: bool,
        force: bool) -> pd.DataFrame:
    if out.exists() and not force:
        df = pd.read_csv(out)
        print(f"[{which}] cached {out.name} ({len(df)} rows)")
        return df
    print(f"[{which}] {len(names)} instances on {N_WORKERS} workers ...", flush=True)
    t0 = time.perf_counter()
    rows = []
    tasks = [(which, n, need_degen) for n in names]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, r in enumerate(pool.map(_one, tasks, chunksize=32), 1):
            rows.append(r)
            if i % 10000 == 0:
                el = time.perf_counter() - t0
                print(f"    {i}/{len(names)}  eta {el/i*(len(names)-i):.0f}s", flush=True)
    df = pd.DataFrame(rows)
    n_err = int(df["_error"].notna().sum()) if "_error" in df else 0
    print(f"[{which}] done, errors={n_err}")
    if n_err:
        print(df.loc[df["_error"].notna(), ["instance_name", "_error"]].head(10).to_string())
        raise SystemExit(f"{which}: extraction errors -- refusing to continue")
    cols = ["instance_name"] + TOPO2 + (DEGEN1 if need_degen else [])
    df = df[cols]
    df.to_csv(out, index=False)
    print(f"[{which}] -> {out}")
    return df


# --------------------------------------------------------------------------
def diff_vs_cached(new: pd.DataFrame) -> None:
    """Quantify how far the repaired extractor moved the shipped corpus values."""
    old_path = HERE / "features_extended.csv"
    if not old_path.exists():
        print("[diff] no features_extended.csv to compare against")
        return
    old = pd.read_csv(old_path, usecols=["instance_name"] + TOPO2)
    m = old.merge(new[["instance_name"] + TOPO2], on="instance_name",
                  suffixes=("_old", "_new"), validate="one_to_one")
    print(f"\n=== repaired extractor vs features_extended.csv ({len(m)} rows) ===")
    rows = []
    for k in TOPO2:
        d = (m[f"{k}_new"] - m[f"{k}_old"]).abs()
        rows.append({"feature": k, "n_changed_gt_1e-9": int((d > 1e-9).sum()),
                     "pct_changed": float((d > 1e-9).mean() * 100),
                     "max_abs_delta": float(d.max()),
                     "mean_abs_delta_where_changed":
                         float(d[d > 1e-9].mean()) if (d > 1e-9).any() else 0.0})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6g}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["corpus", "augment", "bench2d", "tsplib", "all", "diff"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.stage in ("corpus", "all"):
        names = pd.read_csv(ROOT / "tsp_features_v4.csv",
                            usecols=["instance_name"])["instance_name"].astype(str).tolist()
        df = run("corpus", names, OUT_CORPUS, need_degen=False, force=args.force)
        diff_vs_cached(df)
    if args.stage == "diff":
        diff_vs_cached(pd.read_csv(OUT_CORPUS))
    if args.stage in ("augment", "all"):
        names = pd.read_csv(HERE / "augment_features_v3.csv",
                            usecols=["instance_name"])["instance_name"].astype(str).tolist()
        run("augment", names, OUT_AUGMENT, need_degen=True, force=args.force)
    if args.stage in ("bench2d", "all"):
        names = pd.read_csv(HERE / "augmentation_2d_features.csv",
                            usecols=["instance_name"])["instance_name"].astype(str).tolist()
        run("bench2d", names, OUT_BENCH2D, need_degen=True, force=args.force)
    if args.stage in ("tsplib", "all"):
        names = pd.read_csv(HERE / "tsplib_features_v3.csv",
                            usecols=["instance_name"])["instance_name"].astype(str).tolist()
        run("tsplib", names, OUT_TSPLIB, need_degen=True, force=args.force)


if __name__ == "__main__":
    main()
