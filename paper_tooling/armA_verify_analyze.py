"""Analysis of the arm-A robustness sweep. Prints tables, writes summary CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 260)
F4 = lambda x: f"{x:.4f}"  # noqa: E731

R = pd.read_csv(HERE / "armA_verify_runs.csv")
ARMA = R[R.key == "seed_aug_42"].iloc[0]
FROZ = R[R.key == "seed_noaug_42"].iloc[0]

METRICS = ["bench2d_mape", "b2_grid_mspe", "b2_line_noise_mape",
           "linenoise_slope", "nd_test_mape", "tsplib_euc2d_mape",
           "tsplib_euc2d_sdpe", "tsplib_noneuc_mape"]

out: dict = {}

# ========================================================== 1. dose-response
print("=" * 100)
print("1. DOSE-RESPONSE  (LightGBM seed fixed at 42; 8 random nested subsets per dose)")
print("=" * 100)
d = R[R.block == "dose"]
tab = d.groupby("dose")[METRICS].agg(["mean", "std"])
rows = []
for dose, g in d.groupby("dose"):
    r = {"dose": dose, "n_aug": int(g.n_aug.iloc[0]), "reps": len(g)}
    for m in METRICS:
        r[m] = g[m].mean()
        r[m + "_sd"] = g[m].std()
    rows.append(r)
DR = pd.DataFrame(rows)
DR.to_csv(HERE / "armA_verify_dose_response.csv", index=False)
show = ["dose", "n_aug", "reps"] + [c for m in METRICS for c in (m, m + "_sd")]
print(DR[["dose", "n_aug", "reps", "bench2d_mape", "bench2d_mape_sd",
          "b2_grid_mspe", "b2_grid_mspe_sd", "b2_line_noise_mape",
          "b2_line_noise_mape_sd", "linenoise_slope", "linenoise_slope_sd"]]
      .to_string(index=False, float_format=F4))
print()
print(DR[["dose", "nd_test_mape", "nd_test_mape_sd", "tsplib_euc2d_mape",
          "tsplib_euc2d_mape_sd", "tsplib_euc2d_sdpe", "tsplib_euc2d_sdpe_sd",
          "tsplib_noneuc_mape", "tsplib_noneuc_mape_sd"]]
      .to_string(index=False, float_format=F4))

print("\n-- fraction of the full-dose change already delivered at each dose --")
frac_rows = []
for m in METRICS:
    v0 = DR.loc[DR.dose == 0.0, m].iloc[0]
    v1 = DR.loc[DR.dose == 1.0, m].iloc[0]
    span = v1 - v0
    r = {"metric": m, "dose0": v0, "dose1": v1}
    for dose in DR.dose:
        if dose in (0.0,):
            continue
        v = DR.loc[DR.dose == dose, m].iloc[0]
        r[f"f@{dose:g}"] = (v - v0) / span if abs(span) > 1e-12 else np.nan
    # rank correlation of dose vs metric, over the individual runs
    sub = d[d.dose > 0]
    r["spearman_dose_vs_metric_pos_doses"] = stats.spearmanr(sub["dose"], sub[m]).statistic
    frac_rows.append(r)
FR = pd.DataFrame(frac_rows)
FR.to_csv(HERE / "armA_verify_dose_fraction.csv", index=False)
print(FR.to_string(index=False, float_format=F4))

# ================================================ 2. row-order noise at fixed data
print("\n" + "=" * 100)
print("2. NUISANCE FLOOR: same 874 rows, only the ROW ORDER differs (dose = 1.0)")
print("=" * 100)
p1 = d[d.dose == 1.0]
noise = []
for m in METRICS:
    v = p1[m].to_numpy()
    noise.append({"metric": m, "reps": len(v), "mean": v.mean(), "sd": v.std(ddof=1),
                  "min": v.min(), "max": v.max(),
                  "armA_shipped": ARMA[m], "frozen": FROZ[m],
                  "armA_pct_of_perm_dist": float((v <= ARMA[m]).mean() * 100.0),
                  "armA_outside_range": bool(ARMA[m] < v.min() or ARMA[m] > v.max())})
NZ = pd.DataFrame(noise)
NZ.to_csv(HERE / "armA_verify_roworder_noise.csv", index=False)
print(NZ.to_string(index=False, float_format=F4))

sl = p1["linenoise_slope"].to_numpy()
print(f"\ngate G6 (line_noise slope >= 0.70) on the row-order permutations: "
      f"{int((sl >= 0.70).sum())}/{len(sl)} pass; shipped arm A = {ARMA['linenoise_slope']:.4f}")

# ============================================================ 3. seed pairing
print("\n" + "=" * 100)
print("3. SEED SENSITIVITY -- PAIRED refits (both arms refit at each seed)")
print("=" * 100)
s = R[R.block == "seed"]
piv = s.pivot_table(index="seed", columns="arm", values=METRICS)
seed_rows = []
for seed, g in s.groupby("seed"):
    a = g[g.arm == "aug"].iloc[0]
    b = g[g.arm == "noaug"].iloc[0]
    r = {"seed": seed}
    for m in METRICS:
        r[m + "_noaug"] = b[m]
        r[m + "_aug"] = a[m]
        r[m + "_delta"] = a[m] - b[m]
    seed_rows.append(r)
SD = pd.DataFrame(seed_rows)
SD.to_csv(HERE / "armA_verify_seed_paired.csv", index=False)
for m in METRICS:
    c = [f"{m}_noaug", f"{m}_aug", f"{m}_delta"]
    print(f"-- {m}")
    print(SD[["seed"] + c].to_string(index=False, float_format=F4))

print("\n-- ordering: does augmentation ever lose at a matched seed? --")
inv = []
for m in METRICS:
    dd = SD[m + "_delta"].to_numpy()
    better_is_low = m != "linenoise_slope"
    wins = (dd < 0).sum() if better_is_low else (dd > 0).sum()
    inv.append({"metric": m, "n_seeds": len(dd), "aug_wins": int(wins),
                "aug_loses": int(len(dd) - wins),
                "mean_delta": dd.mean(), "min_delta": dd.min(), "max_delta": dd.max()})
IV = pd.DataFrame(inv)
IV.to_csv(HERE / "armA_verify_seed_ordering.csv", index=False)
print(IV.to_string(index=False, float_format=F4))

# also: cross-seed overlap -- can a lucky no-aug seed beat an unlucky aug seed?
print("\n-- cross-seed range overlap (no-aug best vs aug worst) --")
ov = []
for m in METRICS:
    na = s[s.arm == "noaug"][m].to_numpy()
    au = s[s.arm == "aug"][m].to_numpy()
    lower_better = m != "linenoise_slope"
    ov.append({"metric": m,
               "noaug_best": na.min() if lower_better else na.max(),
               "noaug_worst": na.max() if lower_better else na.min(),
               "aug_best": au.min() if lower_better else au.max(),
               "aug_worst": au.max() if lower_better else au.min(),
               "ranges_overlap": bool((na.min() <= au.max()) and (au.min() <= na.max()))})
print(pd.DataFrame(ov).to_string(index=False, float_format=F4))

# ============================================================ 4. composition
print("\n" + "=" * 100)
print("4. COMPOSITION -- leave-one-family-out vs SIZE-MATCHED random removal")
print("=" * 100)
L = R[R.block == "lofo"]
comp = []
for fam, g in L.groupby("family"):
    drop = g[g["mode"] == "drop"].iloc[0]
    only = g[g["mode"] == "only"].iloc[0]
    rnd = g[g["mode"].str.startswith("rand_match")]
    r = {"family": fam, "fam_n": int(drop["fam_n"])}
    for m in METRICS:
        r[m + "_drop"] = drop[m]
        r[m + "_rand"] = rnd[m].mean()
        r[m + "_rand_sd"] = rnd[m].std()
        r[m + "_only"] = only[m]
        # load-bearing = how much worse than a matched random removal
        r[m + "_LB"] = drop[m] - rnd[m].mean()
    comp.append(r)
CP = pd.DataFrame(comp)
CP.to_csv(HERE / "armA_verify_composition.csv", index=False)
for m in ["bench2d_mape", "b2_grid_mspe", "b2_line_noise_mape", "linenoise_slope",
          "nd_test_mape"]:
    print(f"\n-- {m}   (frozen {FROZ[m]:.4f}, arm A {ARMA[m]:.4f})")
    c = ["family", "fam_n", m + "_drop", m + "_rand", m + "_rand_sd", m + "_LB",
         m + "_only"]
    print(CP[c].sort_values(m + "_LB", ascending=False)
          .to_string(index=False, float_format=F4))

json.dump({"armA": {m: float(ARMA[m]) for m in METRICS},
           "frozen": {m: float(FROZ[m]) for m in METRICS}},
          open(HERE / "armA_verify_anchor.json", "w"), indent=2)
print("\nwrote armA_verify_{dose_response,dose_fraction,roworder_noise,"
      "seed_paired,seed_ordering,composition}.csv")
