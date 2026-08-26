"""Audit every label in the alpha-coverage corpus, from the files on disk.

Reads the written solutions rather than the run's in-memory records, so a row
that was never written cannot be counted as sound.  Reports, per stratum:

  * solver composition -- how many labels are provably optimal (Concorde, and
    the exact Held-Karp DP at n <= 20) versus heuristic (LKH);
  * the four integrity gates from ``augment_gen.verify_solution``, re-run here
    against the stored coordinates;
  * the Held-Karp 1-tree gate ``bound <= label``, on the integer matrix the
    solver read;
  * how close the achieved alpha came to the placement prediction.

Writes ``paper_tooling/coverage_audit.csv``.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COV = ROOT / "alpha_coverage"
OUT = ROOT / "paper_tooling" / "coverage_audit.csv"


def _one(name: str) -> dict:
    from data_pipeline.augment_gen import verify_solution
    from mst_utils import mst_length

    with open(COV / "instances" / f"{name}.json") as f:
        inst = json.load(f)
    with open(COV / "solutions" / f"{name}.sol.json") as f:
        sol = json.load(f)

    coords = np.asarray(inst["coordinates"], dtype=np.float64)
    check = verify_solution(coords, {
        "optimal_tour": sol["optimal_tour"], "scale_factor": sol["scale_factor"],
        "cost_int_scaled": sol["cost_int_scaled"], "optimal_cost": sol["optimal_cost"],
        "solver_reported_cost": sol["integrity"].get("solver_reported_cost"),
    })
    geo = inst.get("coverage_geometry", {})
    integ = sol.get("integrity", {})
    mst = float(mst_length(coords))
    return {
        "instance": name, "n": int(inst["n_customers"]), "d": int(inst["dimension"]),
        "grid": int(inst["grid_size"]), "family": inst.get("coverage_family"),
        "group": inst.get("coverage_group"), "solver": sol.get("optimal_solver"),
        "alpha": sol["optimal_cost"] / mst if mst > 0 else np.nan,
        "alpha_pred": geo.get("alpha_pred"), "rho": geo.get("rho"),
        "mix": geo.get("mix"), "spacing": geo.get("spacing"),
        "integrity_ok": bool(check["ok"]),
        "failed_gates": ";".join(check["failed_gates"]),
        "float_rel_dev": check["float_rel_dev"],
        "hk_bound": integ.get("hk_1tree_bound_int_scaled"),
        "hk_ratio": (integ.get("hk_1tree_bound_int_scaled") or np.nan)
        / max(sol["cost_int_scaled"], 1e-12),
        "hk_ok": bool(integ.get("hk_bound_le_cost", False)),
        "mst_recomputed": mst,
        "mst_stored": sol.get("mst_total_length"),
    }


def main() -> None:
    names = sorted(p.stem for p in (COV / "instances").glob("*.json"))
    print(f"[audit] {len(names)} instances")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as ex:
        for i, r in enumerate(ex.map(_one, names, chunksize=16), 1):
            rows.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(names)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    exact = df.solver.isin(["concorde", "held_karp_exact"])
    print(f"\n[audit] rows                {len(df)}")
    print(f"[audit] solver              {df.solver.value_counts().to_dict()}")
    print(f"[audit] provably optimal    {int(exact.sum())} "
          f"({exact.mean()*100:.1f}%)  heuristic {int((~exact).sum())}")
    print(f"[audit] integrity gates     pass {int(df.integrity_ok.sum())} / {len(df)}")
    if (~df.integrity_ok).any():
        print("        failures:", df.loc[~df.integrity_ok, 'failed_gates'].value_counts().to_dict())
    print(f"[audit] HK bound <= label   pass {int(df.hk_ok.sum())} / {len(df)}")
    print(f"[audit] HK bound / label    min {df.hk_ratio.min():.4f} "
          f"median {df.hk_ratio.median():.4f} max {df.hk_ratio.max():.4f}")
    print(f"[audit] stored vs recomputed MST  max rel dev "
          f"{float(((df.mst_recomputed - df.mst_stored).abs() / df.mst_recomputed).max()):.2e}")
    print(f"[audit] float-vs-int tour dev     median {df.float_rel_dev.median():.2e} "
          f"max {df.float_rel_dev.max():.2e}")

    thin = df[(df.rho <= 0.2) & (df.mix == 0.0)]
    e = (thin.alpha - thin.alpha_pred)
    print(f"\n[audit] placement, thin targeted rows (n={len(thin)}): "
          f"mean {e.mean():+.4f}  sd {e.std():.4f}  p95|e| {e.abs().quantile(.95):.4f}")
    over = df.alpha - df.alpha_pred
    print(f"[audit] alpha above the skeleton bound: {int((over > 0.01).sum())} rows "
          f"> 0.01, max {over.max():+.4f} -- the bound is on the noiseless "
          f"skeleton, so transverse noise can lift a realised set past it")
    print(df.groupby("family").apply(
        lambda g: pd.Series({
            "n": len(g), "alpha_min": g.alpha.min(), "alpha_med": g.alpha.median(),
            "alpha_max": g.alpha.max(),
            "pred_err_mean": (g.alpha - g.alpha_pred).mean(),
            "pred_err_sd": (g.alpha - g.alpha_pred).std()}),
        include_groups=False).round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
