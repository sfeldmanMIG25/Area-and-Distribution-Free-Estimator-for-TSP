"""Correctness checks for the repaired extractor.

1. ``compute_pair`` is bit-identical to ``compute`` on the two deployed
   features, across degenerate and generic geometry.
2. The scipy-backed canonical MST selects exactly the tree the reference
   Python union-find selects.
3. Permutation invariance holds for both feature groups.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from features_ext import group_local_id, group_mst_topology as GT  # noqa: E402

PAIR = list(GT.PAIR_NAMES)


def _reference_kruskal(coords, pu, pv, pw, bucket):
    """Textbook union-find Kruskal, the thing the scipy path must reproduce."""
    order = np.lexsort((pv, pu, bucket))
    n = coords.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    out = []
    for i in order:
        a, b = find(int(pu[i])), find(int(pv[i]))
        if a != b:
            parent[a] = b
            out.append((int(pu[i]), int(pv[i])))
            if len(out) == n - 1:
                break
    return {tuple(sorted(e)) for e in out}


def cases():
    rng = np.random.default_rng(11)
    out = []
    for m in (6, 11, 17):
        g = np.arange(m, dtype=float)
        xx, yy = np.meshgrid(g, g, indexing="ij")
        out.append((f"square_m{m}", np.column_stack([xx.ravel(), yy.ravel()])))
        r = []
        for i in range(m):
            for j in range(m):
                r.append([j + 0.5 * (i % 2), i * np.sqrt(3) / 2])
        out.append((f"hex_m{m}", np.asarray(r, float)))
    for n in (5, 40, 300, 1200):
        out.append((f"iso_n{n}", rng.random((n, 2))))
        out.append((f"iso_n{n}_d7", rng.random((n, 7))))
    t = np.linspace(0, 1, 400)
    out.append(("collinear", np.column_stack([t, np.zeros(400)])))
    out.append(("line_noise", np.column_stack([t, rng.normal(0, 0.01, 400)])))
    out.append(("dupes", np.repeat(rng.random((30, 2)), 3, axis=0)))
    return out


def main() -> None:
    n_fail = 0

    print("=== compute_pair == compute (deployed features) ===")
    worst = 0.0
    for name, X in cases():
        a = GT.compute(X)
        b = GT.compute_pair(X)
        d = max(abs(a[k] - b[k]) for k in PAIR)
        worst = max(worst, d)
        if d != 0.0:
            n_fail += 1
            print(f"  MISMATCH {name}: {d:.3e}")
    print(f"  {len(cases())} cases, worst |compute_pair - compute| = {worst:.3e}")

    print("\n=== scipy canonical MST == reference union-find Kruskal ===")
    from scipy.spatial import cKDTree
    checked = 0
    for name, X in cases():
        C, _o = GT.canonical_coords(X)
        u, v = GT._compute_mst_edges(C)
        w = np.linalg.norm(C[u] - C[v], axis=1)
        k = w > 0
        u, v, w = u[k], v[k], w[k]
        if u.size < 2:
            continue
        cu, cv, rep = GT._canonical_mst(C, u, v, w)
        if not rep:
            continue
        wmax = float(w.max())
        tol = GT.TIE_REL_TOL * wmax
        pr = cKDTree(C).query_pairs(r=wmax * (1 + 10 * GT.TIE_REL_TOL),
                                    output_type="ndarray")
        pu, pv = pr[:, 0], pr[:, 1]
        pw = np.linalg.norm(C[pu] - C[pv], axis=1)
        m = pw > 0
        pu, pv, pw = pu[m], pv[m], pw[m]
        ref = _reference_kruskal(C, pu, pv, pw, GT._cluster_index(pw, tol))
        got = {tuple(sorted(e)) for e in zip(cu.tolist(), cv.tolist())}
        checked += 1
        if ref != got:
            n_fail += 1
            print(f"  MISMATCH {name}: {len(ref ^ got)} differing edges")
    print(f"  {checked} repaired cases all matched the reference")

    print("\n=== permutation invariance ===")
    rng = np.random.default_rng(3)
    worst_t = worst_l = 0.0
    for name, X in cases():
        base_t = GT.compute(X)
        base_l = group_local_id.compute(X)
        for _ in range(3):
            p = rng.permutation(len(X))
            worst_t = max(worst_t, max(abs(GT.compute(X[p])[k] - base_t[k])
                                       for k in GT.feature_names()))
            fl = group_local_id.compute(X[p])
            worst_l = max(worst_l, max(abs(fl[k] - base_l[k]) for k in base_l))
    print(f"  mst_topology worst move = {worst_t:.3e}")
    print(f"  local_id     worst move = {worst_l:.3e}")
    if worst_t != 0.0 or worst_l != 0.0:
        n_fail += 1

    print(f"\n{'ALL CHECKS PASSED' if not n_fail else f'{n_fail} CHECK(S) FAILED'}")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
