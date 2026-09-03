"""Does the geometric feature group track the sampling distribution rather than alpha?

Subtest for the argument map (2026-09-02). Companion to shap_by_dimension.py.

Each instance name carries one generator letter per axis
(N{n}_D{d}_G{grid}_{letters}_{seed}); the letter map in
data_pipeline/instance_io.py resolves 11 letters to uniform, 2 to normal,
2 to clustered, 1 to correlated. Per instance we form the three free axis
shares (normal, clustered, correlated; uniform is the complement), the
"generator mix".

Within each (exact d, n-band) cell we demean the SHAP contribution of each
feature group, the true logit target z, and the mix shares, then regress the
group contribution on (i) the mix alone, (ii) z alone, (iii) both. The unique
parts

    unique_mix = R2(mix + z) - R2(z)     variation that tracks the distribution
                                         but carries nothing about the target
    unique_z   = R2(mix + z) - R2(mix)   variation that tracks the target beyond
                                         anything the distribution explains

answer, per dimension band and per group, whether a group is "memorizing the
distribution" (large unique_mix, small unique_z) or carrying instance
information (the reverse). R2(z ~ mix) is printed as the confound check: if the
mix explained z itself, the two would be inseparable.

Writes paper_tooling/tables/shap_distribution_bias.csv. Does not touch
paper_numbers.json.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOSTER = ROOT / "lgbm_model_v3" / "gart2_final.joblib"
TABLE = ROOT / "tsp_features_v4.csv"
OUT = ROOT / "paper_tooling" / "tables" / "shap_distribution_bias.csv"

LETTER_KIND = {**{c: "uniform" for c in "rptglsbxeah"},
               "n": "normal", "i": "normal",
               "c": "clustered", "o": "clustered",
               "k": "correlated"}
NAME_RE = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)$")

D_BANDS = [("d2", 2, 2), ("d3_5", 3, 5), ("d6_10", 6, 10),
           ("d15_25", 15, 25), ("d30_50", 30, 50), ("d100", 100, 100)]
N_EDGES = [5, 10, 50, 100, 200, 500, 1000]
N_LABELS = ["5_10", "11_50", "51_100", "101_200", "201_500", "501_1000"]

GROUPS = {
    "size_dim": ["n_customers", "dimension"],
    "geometric": ["log_bounding_hypervolume", "bounding_hypervolume",
                  "log_node_density", "node_density", "aspect_ratio",
                  "centroid_dist_mean", "centroid_dist_std",
                  "centroid_dist_max", "centroid_dist_iqr"],
    "mst_edge": ["mst_edge_mean", "mst_edge_std", "mst_edge_skew",
                 "mst_edge_kurtosis", "mst_edge_max", "mst_edge_q10",
                 "mst_edge_q25", "mst_edge_q50", "mst_edge_q75", "mst_edge_q90"],
    "mst_topology": ["mst_dominance_ratio", "mst_gap_ratio", "mst_leaf_ratio",
                     "mst_degree_mean", "mst_degree_std", "mst_degree_max",
                     "mst_diameter", "mst_diameter_normalized",
                     "large_edge_count"],
    "greedy": ["greedy_nn_over_mst"],
}
# the two carriers from shap_by_dimension.py, reported on their own as well
SINGLES = ["greedy_nn_over_mst", "mst_dominance_ratio"]


def _mix_shares(name: str, d: int) -> tuple[float, float, float]:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"unparsed instance name: {name!r}")
    letters = m.group(4)
    if len(letters) != d:
        raise ValueError(f"{name!r}: {len(letters)} letters for d={d}")
    kinds = [LETTER_KIND[c] for c in letters]
    return (kinds.count("normal") / d, kinds.count("clustered") / d,
            kinds.count("correlated") / d)


def _demean_by(inv: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    counts = np.bincount(inv)
    if v.ndim == 1:
        return v - (np.bincount(inv, weights=v) / counts)[inv]
    out = np.empty_like(v)
    for j in range(v.shape[1]):
        out[:, j] = v[:, j] - (np.bincount(inv, weights=v[:, j]) / counts)[inv]
    return out


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of an OLS fit of y on X (intercept included)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 0.0 if ss_tot == 0 else 1.0 - float(np.sum(resid ** 2)) / ss_tot


def main() -> int:
    booster = joblib.load(BOOSTER)
    feats = list(booster.feature_name())
    cols = ["instance_name", "split", "optimal_cost", "mst_total_length", *feats]
    df = pd.read_csv(TABLE, usecols=cols, low_memory=False)
    df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"test rows: {len(df)}")

    contrib = booster.predict(df[feats], pred_contrib=True)[:, :-1]
    col = {f: i for i, f in enumerate(feats)}
    group_contrib = {g: contrib[:, [col[f] for f in fs]].sum(axis=1) for g, fs in GROUPS.items()}
    for f in SINGLES:
        group_contrib[f] = contrib[:, col[f]]

    alpha = (df["optimal_cost"] / df["mst_total_length"]).to_numpy()
    u = np.clip(alpha - 1.0, 1e-6, 1 - 1e-6)
    z = np.log(u / (1 - u))

    d = df["dimension"].to_numpy().astype(int)
    n = df["n_customers"].to_numpy().astype(int)
    mix = np.array([_mix_shares(nm, dd) for nm, dd in zip(df["instance_name"], d)])
    n_band = pd.cut(n, bins=N_EDGES, labels=N_LABELS, include_lowest=True).astype(str)
    cell_key = np.array([f"{a}|{b}" for a, b in zip(d, n_band)])

    rows = []
    for band, lo, hi in D_BANDS:
        m = (d >= lo) & (d <= hi)
        if m.sum() == 0:
            continue
        _, inv = np.unique(cell_key[m], return_inverse=True)
        zw = _demean_by(inv, z[m])
        mixw = _demean_by(inv, mix[m])
        # drop mix columns with no within-cell variation (e.g. no correlated axes in band)
        keep = mixw.std(axis=0) > 0
        mixw = mixw[:, keep]
        r2_z_on_mix = _r2(zw, mixw)
        for g, c in group_contrib.items():
            cw = _demean_by(inv, c[m])
            r2_mix = _r2(cw, mixw)
            r2_z = _r2(cw, zw[:, None])
            r2_both = _r2(cw, np.column_stack([mixw, zw]))
            rows.append({
                "band": band, "rows": int(m.sum()), "group": g,
                "var_within": float(np.mean(cw ** 2)),
                "r2_mix": r2_mix, "r2_z": r2_z, "r2_both": r2_both,
                "unique_mix": r2_both - r2_z, "unique_z": r2_both - r2_mix,
                "r2_z_on_mix": r2_z_on_mix,
            })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    order = [b for b, _, _ in D_BANDS if b in set(res["band"])]
    print("\n=== confound check: within-cell R^2 of the true target z on the generator mix ===")
    print(res.groupby("band", sort=False)["r2_z_on_mix"].first().loc[order].round(3).to_string())

    for metric, title in [("r2_mix", "R^2 of group contribution on generator mix alone"),
                          ("r2_z", "R^2 of group contribution on true target z alone"),
                          ("unique_mix", "unique to mix: tracks the distribution, not the target"),
                          ("unique_z", "unique to z: tracks the target beyond the distribution")]:
        print(f"\n=== {title} (percent) ===")
        piv = res.pivot_table(index="group", columns="band", values=metric)
        piv = piv.loc[["geometric", "mst_edge", "mst_topology", "greedy", "size_dim",
                       "greedy_nn_over_mst", "mst_dominance_ratio"], order]
        print((100 * piv).round(1).to_string())
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
