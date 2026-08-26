"""Audit stored reference tours against the released instance coordinates.

For every instance present in BOTH instances/ and solutions/, recompute the length
of the stored `optimal_tour` on the stored `coordinates` and compare it with the
stored `optimal_cost`. Separates two distinct failure modes:

  * rounding_artifact -- stored_cost is exactly the TSPLIB integer-rounded length
    of the stored tour (benign; per-edge nint() rounding).
  * corrupt           -- stored_cost disagrees with the stored tour by >1% and is
    not explained by rounding (the solver saw different coordinates).

Writes paper_tooling/reference_tour_audit.csv and paper_tooling/corrupt_instances.txt.
Prints only aggregate summaries.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INST_DIR = os.path.join(REPO, "instances")
SOL_DIR = os.path.join(REPO, "solutions")
FEATURES_CSV = os.path.join(REPO, "tsp_features_v3.csv")
OUT_CSV = os.path.join(REPO, "paper_tooling", "reference_tour_audit.csv")
OUT_CORRUPT = os.path.join(REPO, "paper_tooling", "corrupt_instances.txt")

OK_TOL = 0.001
CORRUPT_TOL = 0.01
N_BUCKETS = [(5, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]


def audit_one(name: str) -> dict:
    """Load one instance/solution pair and measure the stored tour."""
    row = {
        "instance_name": name,
        "n": -1,
        "d": -1,
        "grid_size": -1,
        "stored_cost": float("nan"),
        "tour_len_float": float("nan"),
        "tour_len_rounded": float("nan"),
        "rel_mismatch": float("nan"),
        "rounding_explains": False,
        "bucket": "",
        "concorde_length": float("nan"),
        "lkh_length": float("nan"),
        "concorde_error": "",
    }
    try:
        with open(os.path.join(INST_DIR, name + ".json"), "rb") as fh:
            inst = json.load(fh)
        with open(os.path.join(SOL_DIR, name + ".sol.json"), "rb") as fh:
            sol = json.load(fh)
    except Exception as exc:  # unreadable pair -> its own failure class
        row["bucket"] = "read_error:" + type(exc).__name__
        return row

    coords = np.asarray(inst["coordinates"], dtype=np.float64)
    n = int(inst.get("n_customers", coords.shape[0]))
    row["n"] = n
    row["d"] = int(inst.get("dimension", coords.shape[1]))
    row["grid_size"] = int(inst.get("grid_size", -1))
    row["stored_cost"] = float(sol["optimal_cost"])
    for key in ("concorde_length", "lkh_length"):
        val = sol.get(key)
        if val is not None:
            row[key] = float(val)
    err = sol.get("concorde_error")
    row["concorde_error"] = "" if err is None else str(err)

    t = list(sol["optimal_tour"])
    if not t:
        row["bucket"] = "tour_not_permutation"
        return row
    if max(t) == n:
        t = [x - 1 for x in t]
    if t[0] == t[-1]:
        t = t[:-1]
    if sorted(t) != list(range(n)):
        row["bucket"] = "tour_not_permutation"
        return row

    idx = np.asarray(t, dtype=np.int64)
    seg = coords[idx] - coords[np.roll(idx, -1)]
    edges = np.sqrt(np.einsum("ij,ij->i", seg, seg))
    tour_float = float(edges.sum())
    tour_rounded = float(np.floor(edges + 0.5).sum())
    row["tour_len_float"] = tour_float
    row["tour_len_rounded"] = tour_rounded

    stored = row["stored_cost"]
    row["rounding_explains"] = bool(
        abs(stored - tour_rounded) <= 1e-6 * max(1.0, tour_rounded)
    )
    if tour_float > 0:
        rel = (stored - tour_float) / tour_float
    else:
        rel = float("nan")
    row["rel_mismatch"] = rel

    if np.isfinite(rel) and abs(rel) <= OK_TOL:
        row["bucket"] = "ok"
    elif row["rounding_explains"]:
        row["bucket"] = "rounding_artifact"
    elif np.isfinite(rel) and abs(rel) > CORRUPT_TOL:
        row["bucket"] = "corrupt"
    else:
        row["bucket"] = "minor_mismatch"
    return row


def _names(directory: str, suffix: str) -> set:
    cut = len(suffix)
    return {
        e.name[:-cut]
        for e in os.scandir(directory)
        if e.is_file() and e.name.endswith(suffix)
    }


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.{nd}f}"


def counts_table(df, index_col, title):
    print(f"\n  {title}")
    tab = pd.crosstab(df[index_col], df["bucket"])
    tab["TOTAL"] = tab.sum(axis=1)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(tab.to_string())


def main() -> None:
    t0 = time.time()
    sample_n = 0
    if "--sample" in sys.argv:
        sample_n = int(sys.argv[sys.argv.index("--sample") + 1])

    inst_names = _names(INST_DIR, ".json")
    sol_names = _names(SOL_DIR, ".sol.json")
    matched = sorted(inst_names & sol_names)

    feat = pd.read_csv(
        FEATURES_CSV, usecols=["instance_name", "mst_total_length", "split"]
    )
    print("=" * 78)
    print("REFERENCE TOUR AUDIT")
    print("=" * 78)
    print(f"instance .json files            : {len(inst_names)}")
    print(f"solution .sol.json files        : {len(sol_names)}")
    print(f"matched instance/solution pairs : {len(matched)}")
    print(f"tsp_features_v3.csv rows        : {len(feat)}")
    print(f"matched pairs present in feats  : {len(set(matched) & set(feat.instance_name))}")

    if sample_n and sample_n < len(matched):
        rng = random.Random(20260810)
        matched = sorted(rng.sample(matched, sample_n))
        print(f"*** SAMPLE MODE: auditing {len(matched)} instances (seed 20260810) ***")

    workers = os.cpu_count() or 4
    print(f"\nauditing {len(matched)} pairs with {workers} processes ...")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(audit_one, matched, chunksize=64), 1):
            rows.append(r)
            if i % 20000 == 0:
                print(f"    {i}/{len(matched)}  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df = df.merge(feat, how="left", on="instance_name")
    df = df.rename(columns={"mst_total_length": "mst"})
    df["split"] = df["split"].fillna("MISSING")
    df["alpha_stored"] = df["stored_cost"] / df["mst"]
    df["alpha_recomputed"] = df["tour_len_float"] / df["mst"]

    # ---------------- (a) bucket counts ----------------
    print("\n" + "-" * 78)
    print("(a) BUCKET COUNTS")
    print("-" * 78)
    overall = df["bucket"].value_counts().sort_index()
    total = len(df)
    print("  overall")
    for b, c in overall.items():
        print(f"    {b:22s} {c:8d}  ({100.0*c/total:.4f}%)")
    print(f"    {'TOTAL':22s} {total:8d}")
    counts_table(df, "split", "by split")
    counts_table(df, "d", "by dimension d")

    # ---------------- (b) alpha range violations ----------------
    print("\n" + "-" * 78)
    print("(b) ALPHA RANGE VIOLATIONS (alpha = cost / mst_total_length)")
    print("-" * 78)
    for col in ("alpha_stored", "alpha_recomputed"):
        lo = df[col] < 1.0
        hi = df[col] > 2.0
        print(f"\n  {col}: n_finite={int(df[col].notna().sum())}"
              f"  (<1.0)={int(lo.sum())}  (>2.0)={int(hi.sum())}")
        g = pd.DataFrame({"split": df["split"], "lt1": lo, "gt2": hi})
        agg = g.groupby("split").agg(n=("lt1", "size"), lt1=("lt1", "sum"),
                                     gt2=("gt2", "sum"))
        print(agg.to_string())

    # ---------------- (c) rel_mismatch distribution ----------------
    print("\n" + "-" * 78)
    print("(c) rel_mismatch = (stored_cost - tour_len_float) / tour_len_float")
    print("-" * 78)
    rm = df["rel_mismatch"].to_numpy(dtype=float)
    fin = rm[np.isfinite(rm)]
    print(f"  n finite            : {fin.size}")
    print(f"  min                 : {fmt(float(fin.min()), 8)}")
    print(f"  p01                 : {fmt(float(np.percentile(fin, 1)), 8)}")
    print(f"  median              : {fmt(float(np.median(fin)), 8)}")
    print(f"  p99                 : {fmt(float(np.percentile(fin, 99)), 8)}")
    print(f"  max                 : {fmt(float(fin.max()), 8)}")
    a = np.abs(fin)
    for thr in (0.001, 0.01, 0.1):
        k = int((a > thr).sum())
        print(f"  frac |rel| > {thr:<6g}: {k}/{fin.size} = {k/fin.size:.6f}")

    # ---------------- (d) corrupt roster ----------------
    print("\n" + "-" * 78)
    print("(d) CORRUPT BUCKET")
    print("-" * 78)
    cor = df[df["bucket"] == "corrupt"]
    print(f"  total corrupt: {len(cor)}")
    show = cor.head(50)
    if len(cor):
        hdr = (f"{'instance_name':<58}{'n':>6}{'d':>5}{'split':>7}"
               f"{'stored_cost':>16}{'tour_len_float':>17}{'mst':>16}"
               f"{'a_stored':>11}{'a_recomp':>11}")
        print("  " + hdr)
        for _, r in show.iterrows():
            print("  "
                  f"{r.instance_name[:57]:<58}{int(r.n):>6}{int(r.d):>5}"
                  f"{str(r.split):>7}{r.stored_cost:>16.4f}{r.tour_len_float:>17.4f}"
                  f"{r.mst:>16.4f}{r.alpha_stored:>11.4f}{r.alpha_recomputed:>11.4f}")
        if len(cor) > 50:
            print(f"  ... showing first 50 of {len(cor)} corrupt instances")

    # ---------------- (e) solver agreement on corrupt rows ----------------
    print("\n" + "-" * 78)
    print("(e) SOLVER FIELDS AMONG CORRUPT ROWS")
    print("-" * 78)
    if len(cor):
        both = cor["concorde_length"].notna() & cor["lkh_length"].notna()
        eq = both & np.isclose(cor["concorde_length"], cor["lkh_length"],
                               rtol=1e-9, atol=1e-9)
        print(f"  rows with both concorde_length and lkh_length : {int(both.sum())}")
        print(f"  concorde_length == lkh_length                 : {int(eq.sum())}")
        print(f"  concorde_length != lkh_length                 : {int((both & ~eq).sum())}")
        print(f"  missing one or both lengths                   : {int((~both).sum())}")
        print("\n  concorde_error value counts:")
        ce = cor["concorde_error"].replace("", "<none>")
        for v, c in ce.value_counts().items():
            print(f"    {str(v)[:70]:<72} {c}")
    else:
        print("  (no corrupt rows)")

    # ---------------- (f) rounding artifact magnitude vs n ----------------
    print("\n" + "-" * 78)
    print("(f) ROUNDING ARTIFACT BUCKET")
    print("-" * 78)
    ra = df[df["bucket"] == "rounding_artifact"].copy()
    print(f"  total rounding_artifact: {len(ra)}")
    if len(ra):
        ra["abs_rel"] = ra["rel_mismatch"].abs()
        print(f"  median |rel_mismatch|  : {fmt(float(ra['abs_rel'].median()), 8)}")
        print(f"  max    |rel_mismatch|  : {fmt(float(ra['abs_rel'].max()), 8)}")
        print("\n  median |rel_mismatch| by n bucket:")
        print(f"    {'n range':<14}{'count':>9}{'median|rel|':>16}{'max|rel|':>16}")
        for lo, hi in N_BUCKETS:
            sub = ra[(ra["n"] >= lo) & (ra["n"] <= hi)]
            if len(sub):
                print(f"    [{lo},{hi}]".ljust(18)
                      + f"{len(sub):>5}{float(sub['abs_rel'].median()):>16.8f}"
                        f"{float(sub['abs_rel'].max()):>16.8f}")
            else:
                print(f"    [{lo},{hi}]".ljust(18) + f"{0:>5}{'-':>16}{'-':>16}")

    # ---------------- outputs ----------------
    cols = ["instance_name", "split", "n", "d", "grid_size", "stored_cost",
            "tour_len_float", "tour_len_rounded", "mst", "alpha_stored",
            "alpha_recomputed", "rel_mismatch", "rounding_explains", "bucket"]
    df[cols].to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    with open(OUT_CORRUPT, "w", encoding="utf-8") as fh:
        for name in cor["instance_name"]:
            fh.write(name + "\n")
    print("\n" + "-" * 78)
    print(f"wrote {OUT_CSV}  ({len(df)} rows)")
    print(f"wrote {OUT_CORRUPT}  ({len(cor)} names)")
    print(f"total wall time: {time.time()-t0:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
