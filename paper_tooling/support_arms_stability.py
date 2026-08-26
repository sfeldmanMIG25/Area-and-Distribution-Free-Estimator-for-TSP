"""Stability diagnostics for the two feature-extractor defects.

Defect 1 (group_mst_topology): a relative-1e-9 coordinate jitter is reported to
move ``mst_topology_straightness`` by 0.178 on hexlattice and 0.054 on square
lattice, because the Euclidean MST is massively non-unique on exact lattices.

Defect 2 (group_local_id): k-NN ties resolved on point index make the features
depend on row order for 7 <= n <= 20.

This script measures both, before and after the fix, on the same generators.
Run with ``--tag before`` / ``--tag after``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features_ext import group_local_id, group_mst_topology  # noqa: E402

SEED = 42
TOPO_KEYS = ["mst_topology_straightness", "mst_topology_deg2_straight_mean"]


# --------------------------------------------------------------------------
# generators: exactly the degenerate geometries the grid failure lives in
# --------------------------------------------------------------------------
def square_lattice(m: int) -> np.ndarray:
    g = np.arange(m, dtype=np.float64)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def hex_lattice(m: int) -> np.ndarray:
    rows = []
    for i in range(m):
        off = 0.5 * (i % 2)
        for j in range(m):
            rows.append([j + off, i * np.sqrt(3.0) / 2.0])
    return np.asarray(rows, dtype=np.float64)


def collinear(n: int, rng) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([t, np.zeros(n)])


def line_noise(n: int, rng) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([t, rng.normal(0.0, 0.01, n)])


def isotropic(n: int, rng) -> np.ndarray:
    return rng.random((n, 2))


def cube_lattice(m: int) -> np.ndarray:
    g = np.arange(m, dtype=np.float64)
    xx, yy, zz = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def cases():
    rng = np.random.default_rng(SEED)
    return [
        ("square_lattice_m15", square_lattice(15)),
        ("square_lattice_m20", square_lattice(20)),
        ("hexlattice_m15", hex_lattice(15)),
        ("hexlattice_m20", hex_lattice(20)),
        ("cube_lattice_m7", cube_lattice(7)),
        ("collinear_n300", collinear(300, rng)),
        ("line_noise_n300", line_noise(300, rng)),
        ("isotropic_n300", isotropic(300, rng)),
        ("isotropic_n1000", isotropic(1000, rng)),
    ]


# --------------------------------------------------------------------------
def topo(coords: np.ndarray) -> dict:
    """Always let the module build its own MST -- that is the path under test."""
    return group_mst_topology.compute(coords, None)


def jitter_sensitivity(rel: float, n_rep: int = 5) -> pd.DataFrame:
    rows = []
    for name, X in cases():
        scale = float(np.ptp(X, axis=0).max())
        base = topo(X)
        rng = np.random.default_rng(SEED)
        moves = {k: [] for k in TOPO_KEYS}
        for _ in range(n_rep):
            Xj = X + rng.normal(0.0, rel * scale, X.shape)
            f = topo(Xj)
            for k in TOPO_KEYS:
                moves[k].append(abs(f[k] - base[k]))
        row = {"case": name, "n": len(X), "d": X.shape[1], "rel_jitter": rel}
        for k in TOPO_KEYS:
            row[f"{k}__base"] = base[k]
            row[f"{k}__max_move"] = float(np.max(moves[k]))
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_sensitivity(n_rep: int = 5) -> pd.DataFrame:
    """Row-order sensitivity of both groups."""
    rows = []
    rng0 = np.random.default_rng(SEED)
    sizes = [7, 9, 12, 15, 20, 40]
    for n in sizes:
        X = rng0.random((n, 2))
        base_l = group_local_id.compute(X)
        base_t = topo(X)
        keys_l = list(base_l)
        mv_l, mv_t = 0.0, 0.0
        rng = np.random.default_rng(SEED + n)
        for _ in range(n_rep):
            p = rng.permutation(n)
            fl = group_local_id.compute(X[p])
            ft = topo(X[p])
            mv_l = max(mv_l, max(abs(fl[k] - base_l[k]) for k in keys_l))
            mv_t = max(mv_t, max(abs(ft[k] - base_t[k]) for k in TOPO_KEYS))
        rows.append({"case": f"isotropic_n{n}", "n": n,
                     "local_id__max_move": mv_l, "mst_topology__max_move": mv_t})
    # lattices too: heavy exact ties
    for name, X in [("square_lattice_m5", square_lattice(5)),
                    ("hexlattice_m5", hex_lattice(5))]:
        base_l = group_local_id.compute(X)
        base_t = topo(X)
        keys_l = list(base_l)
        mv_l, mv_t = 0.0, 0.0
        rng = np.random.default_rng(SEED)
        for _ in range(n_rep):
            p = rng.permutation(len(X))
            fl = group_local_id.compute(X[p])
            ft = topo(X[p])
            mv_l = max(mv_l, max(abs(fl[k] - base_l[k]) for k in keys_l))
            mv_t = max(mv_t, max(abs(ft[k] - base_t[k]) for k in TOPO_KEYS))
        rows.append({"case": name, "n": len(X),
                     "local_id__max_move": mv_l, "mst_topology__max_move": mv_t})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="before")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent
    js = pd.concat([jitter_sensitivity(r) for r in (1e-9, 1e-7, 1e-5)],
                   ignore_index=True)
    ps = permutation_sensitivity()
    js.to_csv(out / f"support_arms_stability_jitter_{args.tag}.csv", index=False)
    ps.to_csv(out / f"support_arms_stability_perm_{args.tag}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"=== JITTER sensitivity [{args.tag}] (max |move| over 5 reps) ===")
    print(js[["case", "n", "d", "rel_jitter",
              "mst_topology_straightness__base",
              "mst_topology_straightness__max_move",
              "mst_topology_deg2_straight_mean__max_move"]]
          .to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\n=== PERMUTATION sensitivity [{args.tag}] (max |move| over 5 reps) ===")
    print(ps.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nworst straightness jitter move @1e-9: "
          f"{js[js.rel_jitter == 1e-9]['mst_topology_straightness__max_move'].max():.6g}")
    print(f"worst local_id perm move: {ps['local_id__max_move'].max():.6g}")
    print(f"worst mst_topology perm move: {ps['mst_topology__max_move'].max():.6g}")


if __name__ == "__main__":
    main()
