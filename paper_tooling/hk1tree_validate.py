"""Validate the Held--Karp 1-tree bound against every instance with a known optimum.

Four checks, in decreasing order of how much they would hurt to fail.

1. ``bound <= OPT``. A lower bound that exceeds the optimum is not a bound.
   Run at the *largest* budget on the *whole* corpus. That is sufficient for
   the entire ladder: the bound is monotone in the budget, so
   ``bound(k) <= bound(k_max) <= OPT`` for every ``k <= k_max``. Validating the
   tightest point validates all the looser ones.

   METRIC CONSISTENCY. The stored ``true_cost`` for the planar corpus is an
   integer: those instances were solved under the TSPLIB ``EUC_2D`` convention,
   where every edge is ``nint(euclidean)``. The 1-tree bound -- like every
   MST-based estimator in this repository -- works in float64 Euclidean. The
   two metrics are not the same and the optimum under rounding can be *below*
   the float64 optimum. So a bound above ``true_cost`` is escalated rather than
   recorded: the stored optimal tour is reloaded and its length recomputed in
   float64. That tour is feasible, so its float64 length is an upper bound on
   the float64 optimum, and comparing against it is metric-consistent. Only a
   bound above *that* is a real violation.

2. ``w(0) >= L_MST``, against ``mst_utils.compute_mst`` -- the project's own MST.
   Note this is an inequality, not an equality. A 1-tree has ``n`` edges and
   contains a spanning tree, so dropping either edge at the special node leaves
   a spanning tree of weight ``w(0) - (that edge)``. Hence ``w(0) >= L_MST``
   with equality only if a zero-length edge is available; on distinct points the
   1-tree bound is strictly stronger than the MST bound at zero penalties.

3. Monotonicity of the bound in the iteration count.

4. ``--delaunay-trap``: the measured counter-example for the shortcut this
   implementation deliberately refuses. See the module docstring of
   ``held_karp_1tree`` for why the Delaunay candidate set is invalid under the
   penalty-modified distances.

Usage
-----
    python paper_tooling/hk1tree_validate.py invariant [--budget 1000] [--limit N]
    python paper_tooling/hk1tree_validate.py ladder [--sample 1500]
    python paper_tooling/hk1tree_validate.py delaunay-trap [--sample 200]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from held_karp_1tree import ITERATION_LADDER, one_tree_bound  # noqa: E402
from mst_utils import compute_mst  # noqa: E402

GT_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv"
GT_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
OUT_DIR = ROOT / "paper_tooling"

_BIN_HEADER = struct.Struct("IIII")


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------


def load_coords(path: str, n: int, d: int, grid: int) -> np.ndarray:
    """Coordinates via the ``.bin`` sibling when present, else the ``.json``.

    The header check mirrors ``run_benchmark_ND_final.load_coords_fast``: a
    stale binary raises rather than silently falling back, because a silent
    coordinate swap is exactly what would invalidate the comparison.
    """
    bin_path = Path(path).with_suffix(".bin")
    if not bin_path.exists():
        with open(path) as f:
            return np.asarray(json.load(f)["coordinates"], dtype=np.float64)
    blob = bin_path.read_bytes()
    b_n, b_d, b_grid, dist_len = _BIN_HEADER.unpack_from(blob, 0)
    offset = _BIN_HEADER.size + dist_len
    if (b_n, b_d, b_grid) != (int(n), int(d), int(grid)):
        raise ValueError(f"{bin_path.name}: stale header {(b_n, b_d, b_grid)}")
    if len(blob) - offset != b_n * b_d * 4:
        raise ValueError(f"{bin_path.name}: truncated coordinate body")
    arr = np.frombuffer(blob, np.float32, count=b_n * b_d, offset=offset)
    return arr.reshape(b_n, b_d).astype(np.float64)


def reference_tour_lengths(inst_path: str, coords: np.ndarray):
    """Recompute the stored optimal tour in float64 and under ``nint`` rounding.

    Returns ``(float64_length, nint_length)``, or ``(nan, nan)`` when the stored
    tour is missing or is not a permutation of the released coordinates. The
    repository already documents 184 instances whose stored tour disagrees with
    their coordinates; those are reported, never silently repaired.
    """
    sol = Path(str(inst_path).replace("instances", "solutions"))
    sol = sol.with_suffix("")
    sol = Path(str(sol) + ".sol.json")
    if not sol.exists():
        return float("nan"), float("nan")
    data = json.loads(sol.read_text())
    tour = data.get("optimal_tour") or data.get("concorde_tour") or data.get("lkh_tour")
    if tour is None:
        return float("nan"), float("nan")
    t = np.asarray(tour, dtype=np.int64)
    n = coords.shape[0]
    if t.size == n and t.min() == 1 and t.max() == n:
        t = t - 1
    if t.size != n or sorted(t.tolist()) != list(range(n)):
        return float("nan"), float("nan")
    nxt = np.roll(t, -1)
    l64 = float(np.linalg.norm(coords[t] - coords[nxt], axis=1).sum())
    ci = np.rint(coords)
    lint = float(np.rint(np.linalg.norm(ci[t] - ci[nxt], axis=1)).sum())
    return l64, lint


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _invariant_worker(rec: dict) -> dict:
    coords = load_coords(rec["file_path"], rec["n_customers"], rec["dimension"], rec["grid_size"])
    uniq = np.unique(coords, axis=0)
    out = {"instance": rec["instance"], "n": int(rec["n_customers"]),
           "d": int(rec["dimension"]), "corpus": rec["corpus"],
           "true_cost": float(rec["true_cost"])}
    if uniq.shape[0] < 3:
        out["status"] = "degenerate_n"
        return out

    res = one_tree_bound(uniq, rec["budget"])
    out["bound"] = res.bound
    out["w0"] = res.initial_bound
    out["is_optimal"] = res.is_optimal
    # Project MST machinery, as the reference for check 2. compute_mst works in
    # float32, so the comparison carries a float32-scale relative tolerance.
    out["mst_len"] = float(compute_mst(uniq).total_length)
    out["status"] = "ok"

    # Escalation path for check 1: only pay for the solution file when the
    # bound actually sits above the stored (possibly rounded-metric) label.
    if res.bound > float(rec["true_cost"]):
        l64, lint = reference_tour_lengths(rec["file_path"], coords)
        out["ref_tour_float64"] = l64
        out["ref_tour_nint"] = lint
    return out


def _ladder_worker(rec: dict) -> list[dict]:
    coords = load_coords(rec["file_path"], rec["n_customers"], rec["dimension"], rec["grid_size"])
    uniq = np.unique(coords, axis=0)
    if uniq.shape[0] < 3:
        return []
    rows = []
    for k in ITERATION_LADDER:
        res = one_tree_bound(uniq, k)
        rows.append({"instance": rec["instance"], "n": int(rec["n_customers"]),
                     "d": int(rec["dimension"]), "corpus": rec["corpus"], "k": k,
                     "bound": res.bound, "true_cost": float(rec["true_cost"]),
                     "is_optimal": res.is_optimal})
    return rows


def _delaunay_worker(rec: dict) -> dict:
    """Run the same ascent but restrict every MST to the pi=0 Delaunay edges.

    This is the shortcut the real implementation refuses. It is executed here
    only to measure what refusing it buys.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import Delaunay

    coords = load_coords(rec["file_path"], rec["n_customers"], rec["dimension"], rec["grid_size"])
    X = np.unique(coords, axis=0)
    n = X.shape[0]
    if n < 5 or X.shape[1] != 2:
        return {}

    tri = Delaunay(X)
    s = tri.simplices
    E = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]])
    E.sort(axis=1)
    E = np.unique(E, axis=0)
    base = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)

    # Node 0's two cheapest edges are taken over ALL nodes, exactly as the real
    # kernel does; only the spanning-tree phase is restricted. That isolates
    # the candidate-set effect from any other difference.
    d0 = np.linalg.norm(X - X[0], axis=1)

    keep = (E[:, 0] != 0) & (E[:, 1] != 0)
    Ek, basek = E[keep], base[keep]

    def one_tree(pi):
        w = basek + pi[Ek[:, 0]] + pi[Ek[:, 1]]
        # scipy reads an implicit zero as "no edge", so shift every weight
        # positive. A constant added to all edges cannot change which spanning
        # tree is minimum, because every spanning tree has the same edge count.
        shift = max(0.0, -w.min()) + 1.0
        g = csr_matrix((w + shift, (Ek[:, 0], Ek[:, 1])), shape=(n, n))
        mst = minimum_spanning_tree(g).tocoo()
        if mst.nnz != n - 2:
            # Delaunay minus the special node came apart: scipy returned a
            # forest, not a tree. Raise rather than under-count the total.
            raise RuntimeError(f"Delaunay subgraph disconnected: {mst.nnz} edges, expected {n-2}")
        total = float(mst.data.sum() - shift * mst.nnz)
        deg = np.zeros(n, dtype=np.int64)
        np.add.at(deg, mst.row, 1)
        np.add.at(deg, mst.col, 1)
        cand = d0 + pi[0] + pi
        cand[0] = np.inf
        i1, i2 = np.argsort(cand)[:2]
        total += cand[i1] + cand[i2]
        deg[0] += 2
        deg[i1] += 1
        deg[i2] += 1
        return total, deg

    pi = np.zeros(n)
    try:
        total, deg = one_tree(pi)
    except RuntimeError:
        return {}
    w_best = total
    t = total / (2.0 * n)
    period, gprev, first, used = 32, np.zeros(n), True, 0
    while used < rec["budget"] and t > 0:
        ia = il = False
        steps = min(period, rec["budget"] - used)
        for j in range(steps):
            g = deg.astype(float) - 2.0
            if not g.any():
                break
            direction = g if first else 0.7 * g + 0.3 * gprev
            first, gprev = False, g
            pi = pi + (t * (period - j) / period) * direction
            try:
                total, deg = one_tree(pi)
            except RuntimeError:
                return {}
            w = total - 2.0 * pi.sum()
            used += 1
            if w > w_best + 1e-12 * max(1.0, abs(w_best)):
                w_best, ia, il = w, True, j == steps - 1
            else:
                il = False
        if il:
            period *= 2
            t *= 2.0
        elif not ia:
            t *= 0.5
            period = max(1, period // 2)

    exact = one_tree_bound(X, rec["budget"])
    l64, _ = reference_tour_lengths(rec["file_path"], coords)
    return {"instance": rec["instance"], "n": n, "delaunay_bound": w_best,
            "exact_bound": exact.bound, "true_cost": float(rec["true_cost"]),
            "ref_tour_float64": l64}


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def load_corpus() -> pd.DataFrame:
    a = pd.read_csv(GT_2D).assign(corpus="2D")
    b = pd.read_csv(GT_ND).assign(corpus="ND")
    cols = ["instance", "file_path", "n_customers", "dimension", "grid_size",
            "true_cost", "mst_length", "corpus"]
    return pd.concat([a[cols], b[cols]], ignore_index=True)


def run_pool(fn, recs, workers, flatten=False):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(fn, recs, chunksize=8), 1):
            if flatten:
                out.extend(r)
            elif r:
                out.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(recs)}", flush=True)
    return pd.DataFrame(out)


def cmd_invariant(args):
    df = load_corpus()
    if args.limit:
        df = df.sample(args.limit, random_state=0)
    recs = df.to_dict("records")
    for r in recs:
        r["budget"] = args.budget
    print(f"[invariant] {len(recs)} instances at k={args.budget}, {args.workers} workers")
    res = run_pool(_invariant_worker, recs, args.workers)
    path = OUT_DIR / f"hk1tree_invariant_k{args.budget}.csv"
    res.to_csv(path, index=False)

    ok = res[res.status == "ok"].copy()
    print(f"\nEvaluated {len(ok)} instances ({(res.status != 'ok').sum()} degenerate).")

    # --- check 1
    ok["excess_vs_label"] = ok.bound - ok.true_cost
    flagged = ok[ok.excess_vs_label > 0]
    print(f"\n[1] bound vs stored true_cost: {len(flagged)} instances above the label.")
    if len(flagged):
        worst = flagged.loc[flagged.excess_vs_label.idxmax()]
        print(f"    worst by absolute excess: {worst.instance} "
              f"bound={worst.bound:.6f} label={worst.true_cost:.6f} "
              f"(+{worst.excess_vs_label:.6f}, {100*worst.excess_vs_label/worst.true_cost:+.6f}%)")
        have = flagged.dropna(subset=["ref_tour_float64"])
        real = have[have.bound > have.ref_tour_float64 * (1 + 1e-9)]
        print(f"    of those, metric-consistent recheck against the stored tour "
              f"recomputed in float64: {len(have)} recomputable, {len(real)} REAL violations")
        if len(real):
            w = real.loc[(real.bound - real.ref_tour_float64).idxmax()]
            print(f"    !!! REAL VIOLATION worst: {w.instance} bound={w.bound:.6f} "
                  f"float64 tour={w.ref_tour_float64:.6f} (+{w.bound-w.ref_tour_float64:.6f})")
        else:
            print("    -> every flagged case is a rounded-metric label, not a bound failure.")
        nores = flagged[flagged.ref_tour_float64.isna()] if "ref_tour_float64" in flagged else flagged
        if len(nores):
            print(f"    {len(nores)} flagged instances have no usable stored tour "
                  f"(cannot be rechecked in a consistent metric)")
    gaps = 100.0 * (ok.true_cost - ok.bound) / ok.true_cost
    print(f"    gap to label: mean {gaps.mean():.4f}%  median {gaps.median():.4f}%  "
          f"p99 {gaps.quantile(0.99):.4f}%  max {gaps.max():.4f}%")

    # --- check 2
    tol = 1e-5 * ok.mst_len.abs()          # compute_mst accumulates in float32
    bad_mst = ok[ok.w0 < ok.mst_len - tol]
    print(f"\n[2] w(0) >= L_MST (mst_utils): {len(bad_mst)} violations of "
          f"{len(ok)} (float32 tolerance 1e-5 relative)")
    ratio = ok.w0 / ok.mst_len
    print(f"    w(0)/L_MST: min {ratio.min():.6f}  mean {ratio.mean():.6f}  max {ratio.max():.6f}")
    print(f"    -> the pi=0 1-tree is stronger than the MST bound by "
          f"{100*(ratio.mean()-1):.2f}% on average")
    print(f"\nWrote {path}")


def cmd_ladder(args):
    df = load_corpus()
    df = df.sample(min(args.sample, len(df)), random_state=1)
    recs = df.to_dict("records")
    print(f"[ladder] {len(recs)} instances x {list(ITERATION_LADDER)}, {args.workers} workers")
    res = run_pool(_ladder_worker, recs, args.workers, flatten=True)
    path = OUT_DIR / "hk1tree_ladder.csv"
    res.to_csv(path, index=False)

    res["gap"] = 100.0 * (res.true_cost - res.bound) / res.true_cost
    piv = res.pivot_table(index="k", values="gap", aggfunc=["mean", "median", "min"])
    piv.columns = ["mean_gap%", "median_gap%", "min_gap%"]
    print("\nGap below the stored optimum, by iteration budget:")
    print(piv.round(4).to_string())

    wide = res.pivot_table(index="instance", columns="k", values="bound")
    ks = list(ITERATION_LADDER)
    viol = 0
    for a, b in zip(ks, ks[1:]):
        viol += int((wide[b] < wide[a] - 1e-9).sum())
    print(f"\nMonotonicity in k: {viol} decreasing steps over "
          f"{len(wide)} instances x {len(ks)-1} consecutive pairs")
    print(f"Closed exactly (subgradient reached zero) at k=1000: "
          f"{int(res[res.k == 1000].is_optimal.sum())}/{len(wide)}")
    print(f"\nWrote {path}")


def cmd_delaunay_trap(args):
    df = load_corpus()
    df = df[(df.corpus == "2D") & (df.n_customers.between(50, 1000))]
    df = df.sample(min(args.sample, len(df)), random_state=2)
    recs = df.to_dict("records")
    for r in recs:
        r["budget"] = args.budget
    print(f"[delaunay-trap] {len(recs)} planar instances at k={args.budget}")
    res = run_pool(_delaunay_worker, recs, args.workers)
    path = OUT_DIR / "hk1tree_delaunay_trap.csv"
    res.to_csv(path, index=False)

    ref = res.ref_tour_float64.fillna(res.true_cost)
    res["delaunay_excess%"] = 100.0 * (res.delaunay_bound - ref) / ref
    res["exact_excess%"] = 100.0 * (res.exact_bound - ref) / ref
    bad_d = int((res["delaunay_excess%"] > 1e-9).sum())
    bad_e = int((res["exact_excess%"] > 1e-9).sum())
    print(f"\nAgainst a feasible tour recomputed in the same metric:")
    print(f"  Delaunay-restricted ascent exceeds it on {bad_d}/{len(res)} instances "
          f"(max +{res['delaunay_excess%'].max():.4f}%)")
    print(f"  exact complete-graph ascent exceeds it on {bad_e}/{len(res)} instances "
          f"(max {res['exact_excess%'].max():+.4f}%)")
    print(f"\nWrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("invariant", cmd_invariant), ("ladder", cmd_ladder),
                     ("delaunay-trap", cmd_delaunay_trap)):
        s = sub.add_parser(name)
        s.add_argument("--budget", type=int, default=1000)
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--sample", type=int, default=1500)
        s.add_argument("--workers", type=int, default=4)
        s.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
