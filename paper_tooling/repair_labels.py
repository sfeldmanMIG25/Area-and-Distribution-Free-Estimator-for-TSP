"""Build the repaired-label table for every scored corpus in this project.

Why this exists
---------------
The multidimensional corpus was solved twice by two code paths with different
distance quantisation, and the second path's costs were written over the
first's.  ``solvers/config.py::get_robust_scale_factor`` multiplies the base
integer scale by ``min(1, n/500)``, so for ``n < 500`` the integer distance
matrix handed to the solver is ``500/n`` times coarser than the pipeline's own
resolution, and the cost reported back is un-scaled by that same coarse factor.
At ``n = 5`` one integer unit is 100 grid units.  The cross-check never caught
it because ``data_pipeline/verification.py`` divides LKH's fast-scale length by
the *robust* factor, inflating it by the same ``500/n``, so the corrupt coarse
value always wins the ``min``.

A second, smaller variant (93 solutions, all written in one 21:49 batch) used
``scale = 1.0``, which is neither rule.

A third population is not a label defect at all: 184 instances had their
coordinates regenerated after solving, so the stored tour does not describe the
released point set.  For those neither the label nor the tour is verifiable and
no lower bound is tight enough to reconstruct the optimum.  They are
quarantined, not repaired.

What a repair is
----------------
Every repaired label is recomputed *from the released coordinates*, never
rescaled from the stored number:

``held_karp_dp_float64``
    Exact float64 optimum by Held--Karp dynamic programming.  Applied to every
    instance with ``n <= HK_EXACT_MAX``.  This is a certificate.

``stored_tour_float64``
    The float64 length of the stored optimal tour, measured on the released
    coordinates.  This is an upper bound on the optimum, and it is exact
    whenever the stored tour is the optimal one -- which the quantisation
    defect does not disturb, because the defect corrupts the *reported length*
    and not the *chosen permutation*.  Where a Held--Karp 1-tree lower bound
    closes on it the status is upgraded to ``tour_certified_optimal``.

``stored_tour_float64_2opt``
    As above, then improved by a float-metric 2-opt descent.  Used on the 2D
    benchmark, whose tours are optimal for the *rounded* metric and therefore
    occasionally beatable in float64.

``tsplib_published``
    No repair.  The published optima are correct for the TSPLIB metric.

``tsplib_fixed_edge_parse``
    ``linhp318`` only.  Its published 41,345 is the fixed-edge Hamiltonian-path
    optimum; our parser walks past the ``FIXED_EDGES_SECTION`` and reads the
    file as a plain tour problem on coordinates bit-identical to ``lin318``.
    The tour optimum on those coordinates is therefore ``lin318``'s published
    optimum, 42,029.

Outputs
-------
``paper_tooling/labels_repaired.csv``   one row per instance, every corpus
``paper_tooling/labels_repaired.json``  summary counts

Quarantined rows carry an EMPTY ``label_repaired``.  That is deliberate: this
project has shipped two silent zero-fill bugs, so a sentinel that reads as data
is the wrong answer.  Anything that scores a quarantined row without consulting
``label_status`` propagates NaN instead of a plausible number, and
``paper_tooling/label_quarantine.py`` turns that into a hard error.

Usage
-----
    python paper_tooling/repair_labels.py            # full rebuild
    python paper_tooling/repair_labels.py --verify   # rebuild and diff vs disk
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "paper_tooling" / "labels_repaired.csv"
OUT_JSON = ROOT / "paper_tooling" / "labels_repaired.json"

ND_INST = ROOT / "instances"
ND_SOL = ROOT / "solutions"
D2_INST = ROOT / "Generalized_TSP_Analysis" / "instances"
D2_SOL = ROOT / "Generalized_TSP_Analysis" / "solutions"
P_TOUR_AUDIT = ROOT / "paper_tooling" / "reference_tour_audit.csv"
P_POLYAK = ROOT / "paper_tooling" / "polyak_nd_sweep.csv"
P_TSPLIB = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"

# Held--Karp DP is 2^n * n^2; 12 is 0.6 M states and still sub-second per
# instance. Above that the DP stops being cheap enough to run corpus-wide.
HK_EXACT_MAX = 10
# A 1-tree bound within this relative distance of the stored tour certifies the
# tour as optimal. The ascent's own arithmetic noise is ~1e-9 relative.
CERT_REL_TOL = 1e-7
# reference_tour_audit's ">1% and not explained by rounding" class.
QUARANTINE_BUCKET = "corrupt"

LINHP318_TOUR_OPTIMUM = 42029.0


# -- shared geometry --------------------------------------------------------


def _tour_edges(coords: np.ndarray, tour: list[int], one_based: bool) -> np.ndarray:
    idx = np.asarray(tour, dtype=np.int64)
    if one_based:
        idx = idx - 1
    seg = coords[idx] - coords[np.roll(idx, -1)]
    return np.sqrt(np.einsum("ij,ij->i", seg, seg))


def _held_karp(coords: np.ndarray) -> float:
    """Exact float64 optimal tour length. O(2^n n^2); caller gates on n."""
    n = coords.shape[0]
    if n <= 3:
        if n <= 1:
            return 0.0
        return float(_tour_edges(coords, list(range(1, n + 1)), True).sum())
    diff = coords[:, None, :] - coords[None, :, :]
    dm = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    m = n - 1  # node 0 is the fixed start
    full = 1 << m
    INF = math.inf
    dp = np.full((full, m), INF)
    dp[:, :] = INF
    for j in range(m):
        dp[1 << j, j] = dm[0, j + 1]
    for mask in range(full):
        row = dp[mask]
        for j in range(m):
            cur = row[j]
            if cur == INF or not (mask >> j) & 1:
                continue
            base = dm[j + 1]
            for k in range(m):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                cand = cur + base[k + 1]
                if cand < dp[nm, k]:
                    dp[nm, k] = cand
    last = dp[full - 1] + dm[1:, 0]
    return float(last.min())


def _two_opt(coords: np.ndarray, tour: np.ndarray, max_rounds: int = 60) -> tuple[np.ndarray, float]:
    """First-improvement 2-opt in the float metric. Returns (tour, length)."""
    n = coords.shape[0]
    if n < 4:
        return tour, float(_tour_edges(coords, list(tour + 1), True).sum())
    diff = coords[:, None, :] - coords[None, :, :]
    dm = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    t = tour.copy()
    for _ in range(max_rounds):
        improved = False
        for i in range(n - 1):
            a, b = t[i], t[i + 1]
            d_ab = dm[a, b]
            js = np.arange(i + 2, n)
            if js.size == 0:
                continue
            cs = t[js]
            ds = t[(js + 1) % n]
            delta = dm[a, cs] + dm[b, ds] - d_ab - dm[cs, ds]
            k = int(np.argmin(delta))
            if delta[k] < -1e-12:
                j = int(js[k])
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                improved = True
        if not improved:
            break
    seg = coords[t] - coords[np.roll(t, -1)]
    return t, float(np.sqrt(np.einsum("ij,ij->i", seg, seg)).sum())


# -- ND corpus --------------------------------------------------------------


def _fast_scale(grid: float) -> float:
    """data_pipeline/generator.py::get_scale_factor -- resolution is grid/10000."""
    return 100.0 if grid <= 100 else (10.0 if grid <= 1000 else 1.0)


def _robust_scale(grid: float, n: int) -> float:
    """solvers/config.py::get_robust_scale_factor -- the defect."""
    return _fast_scale(grid) * min(1.0, n / 500.0)


def _quantised_cost(edges: np.ndarray, scale: float) -> float:
    q = np.floor(edges * scale + 0.5).astype(np.int64)
    q[q == 0] = 1
    return float(q.sum()) / scale


def _nd_worker(names: list[str]) -> list[tuple]:
    rows: list[tuple] = []
    for name in names:
        try:
            with open(ND_INST / f"{name}.json", "rb") as fh:
                inst = json.load(fh)
            with open(ND_SOL / f"{name}.sol.json", "rb") as fh:
                sol = json.load(fh)
        except Exception as exc:  # unreadable pair -> quarantine downstream
            rows.append((name, -1, -1, -1.0, "", math.nan, math.nan, math.nan,
                         math.nan, math.nan, math.nan, "READ_ERROR"))
            continue
        coords = np.asarray(inst["coordinates"], dtype=np.float64)
        n = int(inst.get("n_customers", coords.shape[0]))
        d = int(inst.get("dimension", coords.shape[1]))
        grid = float(inst.get("grid_size", 0) or 0)
        label = sol.get("optimal_cost")
        tour = sol.get("optimal_tour") or []
        solver = str(sol.get("optimal_solver", "") or "")
        label = float(label) if label is not None else math.nan
        if len(tour) != n or sorted(tour) != list(range(1, n + 1)):
            rows.append((name, n, d, grid, solver, label, math.nan, math.nan,
                         math.nan, math.nan, math.nan, "BAD_PERMUTATION"))
            continue
        edges = _tour_edges(coords, tour, one_based=True)
        tour_float = float(edges.sum())
        fs, rs = _fast_scale(grid), _robust_scale(grid, n)
        rep_fast = _quantised_cost(edges, fs)
        rep_rob = _quantised_cost(edges, rs)
        rep_unit = _quantised_cost(edges, 1.0)
        exact = _held_karp(coords) if n <= HK_EXACT_MAX else math.nan
        rows.append((name, n, d, grid, solver, label, tour_float,
                     rep_fast, rep_rob, rep_unit, exact, ""))
    return rows


def _classify_nd(row: pd.Series) -> str:
    """Which code path produced this stored cost."""
    if row["read_error"]:
        return "unreadable"
    label = row["label_stored"]
    if not np.isfinite(label):
        return "no_stored_cost"
    eps = 1e-6 * max(1.0, abs(label))
    hit_fast = abs(row["repro_fast_scale"] - label) <= eps
    hit_rob = abs(row["repro_robust_scale"] - label) <= eps
    hit_unit = abs(row["repro_unit_scale"] - label) <= eps
    if row["bucket"] == QUARANTINE_BUCKET:
        return "D3_coords_regenerated"
    if hit_rob and not hit_fast:
        return "D1_coarse_robust_scale"
    if hit_unit and not hit_fast and not hit_rob:
        return "D2_unit_scale"
    if hit_fast:
        return "clean_fine_quantised"
    return "unreproduced"


def scan_nd(workers: int) -> pd.DataFrame:
    names = sorted(p.stem for p in ND_INST.glob("*.json"))
    print(f"ND: scanning {len(names):,} instances on {workers} workers ...", flush=True)
    chunks = [names[i::workers * 4] for i in range(workers * 4)]
    out: list[tuple] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(_nd_worker, chunks), 1):
            out.extend(part)
            print(f"  chunk {i}/{len(chunks)}  rows={len(out):,}", flush=True)
    df = pd.DataFrame(out, columns=[
        "instance", "n", "d", "grid_size", "solver", "label_stored", "tour_float64",
        "repro_fast_scale", "repro_robust_scale", "repro_unit_scale",
        "exact_optimum", "read_error"])

    audit = pd.read_csv(P_TOUR_AUDIT).rename(columns={"instance_name": "instance"})
    df = df.merge(audit[["instance", "split", "bucket", "mst"]], on="instance", how="left")
    df["provenance"] = df.apply(_classify_nd, axis=1)

    # Best available certified lower bound, for upgrading upper bounds.
    bound = (pd.read_csv(P_POLYAK, usecols=["instance", "bound", "status"])
             .query("status == 'ok'")
             .groupby("instance")["bound"].max())
    df["lower_bound"] = df["instance"].map(bound)
    return df


# -- 2D benchmark -----------------------------------------------------------


def _d2_worker(names: list[str]) -> list[tuple]:
    rows: list[tuple] = []
    for name in names:
        with open(D2_INST / f"{name}.json", "rb") as fh:
            inst = json.load(fh)
        with open(D2_SOL / f"{name}.sol.json", "rb") as fh:
            sol = json.load(fh)
        coords = np.asarray(inst["coordinates"], dtype=np.float64)
        n = coords.shape[0]
        grid = float(inst.get("grid_size", 0) or 0)
        label = float(sol.get("optimal_cost"))
        tour = sol.get("optimal_tour") or []
        one_based = min(tour) == 1
        idx = np.asarray(tour, dtype=np.int64) - (1 if one_based else 0)
        tour_float = float(_tour_edges(coords, tour, one_based).sum())
        exact = _held_karp(coords) if n <= HK_EXACT_MAX else math.nan
        _, refined = _two_opt(coords, idx)
        rows.append((name, n, grid, label, tour_float, refined, exact))
    return rows


def scan_2d(workers: int) -> pd.DataFrame:
    names = sorted(p.name[:-len(".sol.json")] for p in D2_SOL.glob("*.sol.json"))
    print(f"2D: scanning {len(names):,} instances on {workers} workers ...", flush=True)
    chunks = [names[i::workers * 2] for i in range(workers * 2)]
    out: list[tuple] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_d2_worker, chunks):
            out.extend(part)
    return pd.DataFrame(out, columns=[
        "instance", "n", "grid_size", "label_stored", "tour_float64",
        "tour_float64_2opt", "exact_optimum"])


# -- assembly ---------------------------------------------------------------


def build() -> tuple[pd.DataFrame, dict]:
    workers = max(1, min(20, (os.cpu_count() or 4)))
    nd = scan_nd(workers)
    d2 = scan_2d(workers)

    # ---- ND rows
    quarantine = nd["provenance"].isin({"D3_coords_regenerated", "unreadable",
                                        "no_stored_cost", "BAD_PERMUTATION"})
    # A label strictly below its own MST is impossible in a metric space.
    impossible = nd["label_stored"] < nd["mst"] * (1 - 1e-9)
    quarantine = quarantine | impossible.fillna(False)

    repaired = np.where(np.isfinite(nd["exact_optimum"]),
                        nd["exact_optimum"], nd["tour_float64"])
    method = np.where(np.isfinite(nd["exact_optimum"]),
                      "held_karp_dp_float64", "stored_tour_float64")
    status = np.where(np.isfinite(nd["exact_optimum"]),
                      "exact_certified", "tour_upper_bound")
    closed = (nd["lower_bound"].notna()
              & (nd["tour_float64"] - nd["lower_bound"]
                 <= CERT_REL_TOL * nd["tour_float64"].abs()))
    status = np.where(closed & ~np.isfinite(nd["exact_optimum"]),
                      "tour_certified_optimal", status)
    nd["label_repaired"] = np.where(quarantine, np.nan, repaired)
    nd["repair_method"] = np.where(quarantine, "quarantine", method)
    nd["label_status"] = np.where(quarantine, "quarantined_no_recoverable_truth", status)
    nd["corpus"] = "nd"

    # ---- 2D rows: the tour is optimal for the ROUNDED metric, so the float
    # optimum can be strictly shorter. Take the better of the two, and use the
    # DP optimum outright where it is available.
    best = np.minimum(d2["tour_float64"], d2["tour_float64_2opt"])
    use_exact = np.isfinite(d2["exact_optimum"])
    d2["label_repaired"] = np.where(use_exact, d2["exact_optimum"], best)
    d2["repair_method"] = np.where(
        use_exact, "held_karp_dp_float64",
        np.where(d2["tour_float64_2opt"] < d2["tour_float64"] - 1e-12,
                 "stored_tour_float64_2opt", "stored_tour_float64"))
    d2["label_status"] = np.where(use_exact, "exact_certified", "tour_upper_bound")
    d2["corpus"] = "2d"
    d2["d"] = 2
    d2["split"] = "benchmark_2d"
    d2["provenance"] = "nint_metric_winners_curse"

    # ---- TSPLIB rows
    tl = pd.read_csv(P_TSPLIB, usecols=["instance", "n", "edge_weight_type", "true_cost"])
    tl = tl.drop_duplicates("instance").reset_index(drop=True)
    tl["label_stored"] = tl["true_cost"]
    is_lin = tl["instance"].str.lower().str.replace(r"\.tsp$", "", regex=True) == "linhp318"
    tl["label_repaired"] = np.where(is_lin, LINHP318_TOUR_OPTIMUM, tl["true_cost"])
    tl["repair_method"] = np.where(is_lin, "tsplib_fixed_edge_parse", "tsplib_published")
    tl["label_status"] = np.where(is_lin, "exact_certified", "published_exact")
    tl["provenance"] = np.where(is_lin, "parser_fixed_edge_section", "tsplib_published")
    tl["corpus"] = "tsplib"
    tl["split"] = "tsplib"
    tl["grid_size"] = np.nan
    tl["d"] = np.where(tl["edge_weight_type"].isin(["EUC_2D", "CEIL_2D", "ATT"]), 2.0, np.nan)

    cols = ["corpus", "instance", "split", "n", "d", "grid_size", "provenance",
            "label_stored", "label_repaired", "label_status", "repair_method",
            "tour_float64", "exact_optimum", "lower_bound", "mst",
            "repro_fast_scale", "repro_robust_scale", "repro_unit_scale"]
    for frame in (nd, d2, tl):
        for c in cols:
            if c not in frame.columns:
                frame[c] = np.nan
    out = pd.concat([nd[cols], d2[cols], tl[cols]], ignore_index=True)
    out["delta_pct"] = 100.0 * (out["label_stored"] - out["label_repaired"]) / out["label_repaired"]

    summary = _summarise(out)
    return out, summary


def _summarise(out: pd.DataFrame) -> dict:
    nd = out[out["corpus"] == "nd"]
    d2 = out[out["corpus"] == "2d"]
    bad = nd["provenance"].isin({"D1_coarse_robust_scale", "D2_unit_scale"})
    quar = nd["label_status"] == "quarantined_no_recoverable_truth"
    nd_test = nd[nd["split"] == "test"]

    def _mape(frame: pd.DataFrame) -> float:
        m = frame["delta_pct"].abs().dropna()
        return float(m.mean()) if len(m) else float("nan")

    return {
        "generated_by": "paper_tooling/repair_labels.py",
        "hk_exact_max_n": HK_EXACT_MAX,
        "cert_rel_tol": CERT_REL_TOL,
        "nd": {
            "n_instances": int(len(nd)),
            "by_provenance": {k: int(v) for k, v in
                              nd["provenance"].value_counts().items()},
            "by_status": {k: int(v) for k, v in
                          nd["label_status"].value_counts().items()},
            "bad_labels": int(bad.sum()),
            "bad_label_pct": round(100.0 * float(bad.sum()) / len(nd), 4),
            "quarantined": int(quar.sum()),
            "by_split_bad": {k: int(v) for k, v in
                             nd.loc[bad, "split"].value_counts().items()},
            "by_split_quarantined": {k: int(v) for k, v in
                                     nd.loc[quar, "split"].value_counts().items()},
            "test": {
                "n_instances": int(len(nd_test)),
                "bad_labels": int(nd_test["provenance"].isin(
                    {"D1_coarse_robust_scale", "D2_unit_scale"}).sum()),
                "quarantined": int((nd_test["label_status"]
                                    == "quarantined_no_recoverable_truth").sum()),
                "label_mape_pct": _mape(nd_test),
                "signed_mean_pct": float(nd_test["delta_pct"].mean(skipna=True)),
            },
        },
        "d2": {
            "n_instances": int(len(d2)),
            "label_mape_pct": _mape(d2),
            "signed_mean_pct": float(d2["delta_pct"].mean(skipna=True)),
            "by_grid": {
                str(int(g)): {
                    "n": int((d2["grid_size"] == g).sum()),
                    "label_mape_pct": _mape(d2[d2["grid_size"] == g]),
                    "signed_mean_pct": float(d2.loc[d2["grid_size"] == g,
                                                    "delta_pct"].mean()),
                } for g in sorted(d2["grid_size"].dropna().unique())
            },
            "tours_improvable_in_float": int(
                (d2["repair_method"] == "stored_tour_float64_2opt").sum()),
        },
        "tsplib": {
            "n_instances": int((out["corpus"] == "tsplib").sum()),
            "repaired": int((out["repair_method"] == "tsplib_fixed_edge_parse").sum()),
        },
        "quarantined_total": int((out["label_status"]
                                  == "quarantined_no_recoverable_truth").sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and diff against the table already on disk")
    args = ap.parse_args()

    out, summary = build()
    if args.verify and OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        merged = old.merge(out, on=["corpus", "instance"], suffixes=("_old", "_new"))
        diff = merged[(merged["label_repaired_old"].fillna(-1)
                       - merged["label_repaired_new"].fillna(-1)).abs() > 1e-6]
        print(f"--verify: {len(diff)} of {len(merged)} labels differ")
        return 1 if len(diff) else 0

    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
