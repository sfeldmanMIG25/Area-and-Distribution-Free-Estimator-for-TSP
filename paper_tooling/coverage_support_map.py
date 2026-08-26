"""The alpha support of the training split, before and after the fix.

This is the defect statement and its repair in one table.  Everything is read
off the two feature tables; nothing is modelled.

Writes ``paper_tooling/coverage_support_map.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V4 = ROOT / "tsp_features_v4.csv"
COV = ROOT / "alpha_coverage" / "coverage_features.csv"
OUT = ROOT / "paper_tooling" / "coverage_support_map.csv"

BINS = [0, 10, 20, 50, 100, 200, 500, 1000, 10 ** 9]
LABELS = ["<=10", "11-20", "21-50", "51-100", "101-200", "201-500", "501-1000", ">1000"]


def load() -> pd.DataFrame:
    base = pd.read_csv(V4, usecols=["instance_name", "split", "optimal_cost",
                                    "mst_total_length", "n_customers", "dimension"])
    m = base["mst_total_length"].replace(0, np.nan)
    base["alpha"] = (base["optimal_cost"] / m).clip(1.0, 2.0)
    base = base.dropna(subset=["alpha"])
    base["corpus"] = "nd"

    cov = pd.read_csv(COV, usecols=["instance_name", "split", "alpha",
                                    "n_customers", "dimension"])
    cov["corpus"] = "cov"
    return pd.concat([base, cov], ignore_index=True)


def table(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    d = df.copy()
    d["nb"] = pd.Categorical(pd.cut(d.n_customers, BINS, labels=LABELS), LABELS, True)
    g = d.groupby("nb", observed=False)["alpha"]
    out = pd.DataFrame({
        "arm": tag, "rows": g.size(), "min": g.min(), "median": g.median(),
        "p99": g.quantile(0.99), "max": g.max(),
        "n_gt_1p3": d[d.alpha > 1.3].groupby("nb", observed=False).size(),
        "n_gt_1p5": d[d.alpha > 1.5].groupby("nb", observed=False).size(),
        "n_gt_1p7": d[d.alpha > 1.7].groupby("nb", observed=False).size(),
    })
    return out.reset_index()


def main() -> None:
    df = load()
    tr_old = df[(df.split == "train") & (df.corpus == "nd")]
    tr_new = df[df.split == "train"]

    t = pd.concat([table(tr_old, "before"), table(tr_new, "after")], ignore_index=True)
    t.to_csv(OUT, index=False)

    for tag in ("before", "after"):
        print(f"\n=== training split alpha support: {tag} ===")
        print(t[t.arm == tag].drop(columns=["arm"]).to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))

    print("\n=== alpha histogram over [1,2] in 10 bins, training split ===")
    for tag, d in (("before", tr_old), ("after", tr_new)):
        print(f"  {tag:7s} all n   ", np.histogram(d.alpha, 10, (1.0, 2.0))[0].tolist())
    for lo, hi, name in ((100, 10 ** 9, "n>=100"), (200, 10 ** 9, "n>=200"),
                         (500, 10 ** 9, "n>=500")):
        for tag, d in (("before", tr_old), ("after", tr_new)):
            s = d[(d.n_customers >= lo) & (d.n_customers < hi)]
            print(f"  {tag:7s} {name:8s}", np.histogram(s.alpha, 10, (1.0, 2.0))[0].tolist())

    print("\n=== coverage rows by dimension ===")
    cov = df[df.corpus == "cov"]
    print(cov.groupby("dimension")["alpha"]
          .agg(["count", "min", "median", "max"]).round(3).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
