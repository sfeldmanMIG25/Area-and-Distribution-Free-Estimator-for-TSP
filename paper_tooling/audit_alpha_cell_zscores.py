"""Standardize the corrupt rows' alpha against their own (d,n) cell.

Section 3.2 of the manuscript asserted that the 184 provenance-corrupt rows'
alpha = L_TSP / L_MST "is close to the distribution of the unaffected rows".
No tool computed that, and the sentence's two halves were not reproducible
under any single convention.  This settles it under one stated convention.

Convention
----------
Cell            = one (d, n) pair of ``reference_tour_audit.csv``.
Affected rows   = ``bucket == "corrupt"`` (the >1% disagreements of
                  ``audit_reference_tours.py``).
Reference       = the *unaffected* rows of the same cell, i.e. every row of
                  that cell whose bucket is not ``corrupt``.
Statistic       = z = (mean alpha_stored over the cell's affected rows
                       - mean alpha_stored over the cell's reference rows)
                      / sd(alpha_stored over the cell's reference rows), ddof=1.

The row-level sd is the divisor, not the standard error of the affected mean.
That is the conservative choice: it asks whether the affected rows look like
*draws* from the unaffected distribution, and it gives a |z| smaller by a
factor of sqrt(n_affected) than a test of the means would.

Reads  paper_tooling/reference_tour_audit.csv (written by audit_reference_tours.py).
Writes paper_tooling/reference_tour_alpha_zscores.csv, one row per cell.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, "reference_tour_audit.csv")
OUT_CSV = os.path.join(HERE, "reference_tour_alpha_zscores.csv")
AFFECTED_BUCKET = "corrupt"
THRESHOLD = 1.6


def build() -> pd.DataFrame:
    df = pd.read_csv(IN_CSV)
    df = df[np.isfinite(df["alpha_stored"])]
    rows = []
    for (dim, n), cell in df.groupby(["d", "n"], sort=True):
        aff = cell[cell["bucket"] == AFFECTED_BUCKET]
        if not len(aff):
            continue
        ref = cell[cell["bucket"] != AFFECTED_BUCKET]
        mu = float(ref["alpha_stored"].mean())
        sd = float(ref["alpha_stored"].std(ddof=1))
        rows.append({
            "d": int(dim), "n": int(n),
            "n_affected": int(len(aff)), "n_reference": int(len(ref)),
            "alpha_mean_affected": float(aff["alpha_stored"].mean()),
            "alpha_mean_reference": mu, "alpha_sd_reference": sd,
            "z": (float(aff["alpha_stored"].mean()) - mu) / sd if sd > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["d", "n"]).reset_index(drop=True)


def main() -> None:
    t = build()
    t.to_csv(OUT_CSV, index=False)
    over = t[t["z"].abs() > THRESHOLD]
    worst = t.loc[t["z"].abs().idxmax()]
    clean_dims = sorted(set(t["d"]) - set(over["d"]))

    print("=" * 78)
    print("alpha of the provenance-corrupt rows, standardized within its (d,n) cell")
    print("=" * 78)
    print(f"  cells containing an affected row : {len(t)}")
    print(f"  affected rows covered            : {int(t.n_affected.sum())}")
    print(f"  smallest reference group         : {int(t.n_reference.min())}")
    print(f"  cells with |z| > {THRESHOLD}            : {len(over)}"
          f"  (holding {int(over.n_affected.sum())} affected rows)")
    print(f"  median |z| over cells            : {t['z'].abs().median():.3f}")
    print(f"  largest departure                : z = {worst.z:.4f} "
          f"at d={int(worst.d)}, n={int(worst.n)} on {int(worst.n_affected)} row(s)")
    print(f"  dimensions with every cell <= {THRESHOLD}: {clean_dims}")
    print()
    print(t.to_string(index=False,
                      formatters={"z": lambda v: f"{v:9.4f}",
                                  "alpha_mean_affected": lambda v: f"{v:.6f}",
                                  "alpha_mean_reference": lambda v: f"{v:.6f}",
                                  "alpha_sd_reference": lambda v: f"{v:.6f}"}))
    print(f"\nwrote {OUT_CSV}  ({len(t)} cells)")


if __name__ == "__main__":
    main()
