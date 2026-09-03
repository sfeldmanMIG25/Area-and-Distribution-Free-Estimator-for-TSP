"""SHAP by dimension band, and what a (d, n) constant cannot carry.

Subtest for the argument map (2026-09-02). Two measurements on the released
booster over the full 16,920-row test split, no sampling:

1. Mean |SHAP| share per feature inside each dimension band (the usual reliance
   number, now stratified).
2. Within-cell variance attribution. A cell is (exact d, n-band), the finest
   thing a calibrated per-cell constant rho(d, n) can key on. Subtracting the
   cell mean from every SHAP column and from the prediction leaves the
   instance-level variation that no constant can supply. With z_hat_w the
   within-cell prediction and c_j_w the within-cell contribution of feature j,
   Var(z_hat_w) = sum_j Cov(c_j_w, z_hat_w) exactly, so
   share_j = Cov(c_j_w, z_hat_w) / Var(z_hat_w) sums to one over features.
   Also reported: the within-cell R^2 of z_hat against the true logit target,
   i.e. how much of what a (d, n) constant misses the model actually recovers.

Attribution is in the booster's raw logit space, z = logit(alpha - 1). The
within-cell R^2 is reported in both logit space and alpha space; the alpha-space
figure is the one that maps onto the percent-error tables.

Writes paper_tooling/tables/shap_by_dimension.csv and, unless --no-bank, merges
a compact key set (prefix ``shap_band_``) into paper_numbers.json via the
sidecar ``shap_band_numbers.json`` so a table rebuild re-carries them. The bank
step also reads tables/shap_distribution_bias.csv when present, for the
generator-mix confound row and the unique-to-target shares.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOSTER = ROOT / "lgbm_model_v3" / "gart2_final.joblib"
TABLE = ROOT / "tsp_features_v4.csv"
TABLES = ROOT / "paper_tooling" / "tables"
OUT = TABLES / "shap_by_dimension.csv"
BIAS_CSV = TABLES / "shap_distribution_bias.csv"
BANK = TABLES / "paper_numbers.json"
SIDECAR_KEYS = TABLES / "shap_band_numbers.json"
KEY_PREFIX = "shap_band_"
CARRIERS = ["greedy_nn_over_mst", "mst_dominance_ratio"]

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


def _demean_by(inv: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Subtract the group mean (groups given by integer codes ``inv``)."""
    v = np.asarray(v, dtype=float)
    counts = np.bincount(inv)
    if v.ndim == 1:
        means = np.bincount(inv, weights=v) / counts
        return v - means[inv]
    out = np.empty_like(v)
    for j in range(v.shape[1]):
        means = np.bincount(inv, weights=v[:, j]) / counts
        out[:, j] = v[:, j] - means[inv]
    return out


def bank_numbers(res: pd.DataFrame) -> dict[str, object]:
    """Compact key set for the manuscript: per band, the two carriers' within-cell
    shares and truth correlations, the five group shares, both R^2 figures, row and
    cell counts; plus, from the distribution-bias CSV, the confound row and the
    unique-to-target / unique-to-mix shares of the carriers and groups."""
    grp_of = {f: g for g, fs in GROUPS.items() for f in fs}
    numbers: dict[str, object] = {}
    for band, sub in res.groupby("band", sort=False):
        p = f"{KEY_PREFIX}{band}_"
        first = sub.iloc[0]
        numbers[p + "rows"] = int(first["rows"])
        numbers[p + "cells"] = int(first["cells"])
        numbers[p + "r2_within_logit"] = round(float(first["r2_within_band"]), 6)
        numbers[p + "r2_within_alpha"] = round(float(first["r2_within_alpha"]), 6)
        numbers[p + "within_frac_of_target_var_pct"] = round(100 * float(first["within_frac_of_target_var"]), 6)
        by_feat = sub.set_index("feature")
        for f in CARRIERS:
            numbers[p + f"within_share_{f}_pct"] = round(100 * float(by_feat.loc[f, "within_cell_share"]), 6)
            numbers[p + f"abs_share_{f}_pct"] = round(100 * float(by_feat.loc[f, "abs_share"]), 6)
            numbers[p + f"corr_truth_{f}"] = round(float(by_feat.loc[f, "corr_with_truth_within"]), 6)
        numbers[p + "within_share_carriers_pct"] = round(
            100 * float(by_feat.loc[CARRIERS, "within_cell_share"].sum()), 6)
        grp = sub.assign(group=sub["feature"].map(grp_of)).groupby("group")
        for g, s in grp["within_cell_share"].sum().items():
            numbers[p + f"group_{g}_within_share_pct"] = round(100 * float(s), 6)
        for g, s in grp["abs_share"].sum().items():
            numbers[p + f"group_{g}_abs_share_pct"] = round(100 * float(s), 6)
    if BIAS_CSV.exists():
        bias = pd.read_csv(BIAS_CSV)
        for band, sub in bias.groupby("band", sort=False):
            p = f"{KEY_PREFIX}{band}_"
            numbers[p + "target_r2_on_mix_pct"] = round(100 * float(sub["r2_z_on_mix"].iloc[0]), 6)
            for _, r in sub.iterrows():
                g = r["group"]
                numbers[p + f"unique_z_{g}_pct"] = round(100 * float(r["unique_z"]), 6)
                numbers[p + f"unique_mix_{g}_pct"] = round(100 * float(r["unique_mix"]), 6)
    return numbers


def shap_band_bank_numbers() -> dict[str, object]:
    """Keys this module owns, read back from its sidecar (see shap_production.py
    for why: build_paper_tables.main() rewrites the bank wholesale)."""
    if not SIDECAR_KEYS.exists():
        return {}
    return json.loads(SIDECAR_KEYS.read_text(encoding="utf-8"))


def merge_bank(numbers: dict[str, object]) -> tuple[int, int, int]:
    SIDECAR_KEYS.write_text(json.dumps(numbers, indent=2, sort_keys=True), encoding="utf-8")
    bank: dict[str, object] = json.loads(BANK.read_text(encoding="utf-8"))
    stale = [k for k in bank if k.startswith(KEY_PREFIX) and k not in numbers]
    changed = sum(1 for k, v in numbers.items() if k in bank and bank[k] != v)
    fresh = sum(1 for k in numbers if k not in bank)
    for k in stale:
        del bank[k]
    bank.update(numbers)
    BANK.write_text(json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8")
    return fresh, changed, len(stale)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-bank", action="store_true")
    args = ap.parse_args(argv)

    booster = joblib.load(BOOSTER)
    feats = list(booster.feature_name())
    assert len(feats) == 31, feats
    cols = ["split", "optimal_cost", "mst_total_length", *feats]
    df = pd.read_csv(TABLE, usecols=cols, low_memory=False)
    df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"test rows: {len(df)}")

    X = df[feats]
    contrib = booster.predict(X, pred_contrib=True)  # (n, 31 + 1), last = bias
    C = contrib[:, :-1]
    z_hat = contrib.sum(axis=1)

    alpha = (df["optimal_cost"] / df["mst_total_length"]).to_numpy()
    u = np.clip(alpha - 1.0, 1e-6, 1 - 1e-6)
    z = np.log(u / (1 - u))
    alpha_hat = 1.0 + 1.0 / (1.0 + np.exp(-z_hat))

    d = df["dimension"].to_numpy().astype(int)
    n = df["n_customers"].to_numpy().astype(int)
    n_band = pd.cut(n, bins=N_EDGES, labels=N_LABELS, include_lowest=True).astype(str)
    cell_key = np.array([f"{a}|{b}" for a, b in zip(d, n_band)])

    rows = []
    for band, lo, hi in D_BANDS:
        m = (d >= lo) & (d <= hi)
        if m.sum() == 0:
            continue
        Cb, zb, zhb = C[m], z[m], z_hat[m]
        codes, inv = np.unique(cell_key[m], return_inverse=True)

        # 1. mean |SHAP| share
        mean_abs = np.abs(Cb).mean(axis=0)
        abs_share = mean_abs / mean_abs.sum()

        # 2. within-cell attribution
        Cw, zw, zhw = _demean_by(inv, Cb), _demean_by(inv, zb), _demean_by(inv, zhb)
        var_zh = float(np.mean(zhw ** 2))
        cov_share = np.array([np.mean(Cw[:, j] * zhw) for j in range(Cw.shape[1])]) / var_zh
        r2_within = 1.0 - float(np.mean((zw - zhw) ** 2) / np.mean(zw ** 2))
        aw, ahw = _demean_by(inv, alpha[m]), _demean_by(inv, alpha_hat[m])
        r2_within_alpha = 1.0 - float(np.mean((aw - ahw) ** 2) / np.mean(aw ** 2))
        within_frac = float(np.mean(zw ** 2) / zb.var())
        corr_truth = np.array([
            np.corrcoef(Cw[:, j], zw)[0, 1] if Cw[:, j].std() > 0 else 0.0
            for j in range(Cw.shape[1])
        ])

        for j, f in enumerate(feats):
            rows.append({
                "band": band, "rows": int(m.sum()), "cells": int(len(codes)),
                "feature": f, "abs_share": abs_share[j],
                "within_cell_share": cov_share[j],
                "corr_with_truth_within": corr_truth[j],
                "r2_within_band": r2_within,
                "r2_within_alpha": r2_within_alpha,
                "within_frac_of_target_var": within_frac,
            })

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    grp_of = {f: g for g, fs in GROUPS.items() for f in fs}
    res["group"] = res["feature"].map(grp_of)
    order = [b for b, _, _ in D_BANDS if b in set(res["band"])]

    print("\n=== per band: rows, cells, within-cell fraction of target variance, "
          "within-cell R^2 of GART ===")
    print(res.groupby("band", sort=False)[
        ["rows", "cells", "within_frac_of_target_var", "r2_within_band", "r2_within_alpha"]
    ].first().loc[order].round(3).to_string())

    print("\n=== group shares by band (percent) ===")
    ga = res.pivot_table(index="group", columns="band", values="abs_share", aggfunc="sum")
    gw = res.pivot_table(index="group", columns="band", values="within_cell_share", aggfunc="sum")
    print("mean|SHAP| share\n" + (100 * ga[order]).round(1).to_string())
    print("\nwithin-cell variance share\n" + (100 * gw[order]).round(1).to_string())

    print("\n=== top 5 features per band by within-cell variance share ===")
    for b in order:
        sub = res[res["band"] == b].sort_values("within_cell_share", ascending=False).head(5)
        print(f"\n{b}:")
        for _, r in sub.iterrows():
            print(f"  {r['feature']:<26} share {100 * r['within_cell_share']:6.1f}%   "
                  f"corr {r['corr_with_truth_within']:+.3f}   "
                  f"(mean|SHAP| {100 * r['abs_share']:.1f}%)")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if not args.no_bank:
        fresh, changed, stale = merge_bank(bank_numbers(res.drop(columns=["group"])))
        print(f"bank: {fresh} new, {changed} updated, {stale} stale removed "
              f"(prefix {KEY_PREFIX}, sidecar {SIDECAR_KEYS.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
