"""Regenerate all five LaTeX tables, replacing R^2_alpha with Pearson r_alpha (bounded [-1, 1]).
Outputs full LaTeX blocks for each table, ready to splice. Also emits CSV with the new column.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

REPO = Path(r"D:/Area-and-Distribution-Free-Estimator-for-TSP")
TBL_OUT = REPO / "paper_reference/scripts/tables"

# -------- Solver-time lookups (precomputed per bucket from CSVs) --------
# 2D: optimal_solve_time_s, per n-bucket, seconds. Source: benchmark_results_2D_v3.csv (deduped by instance).
SOLVER_2D = {
    "$[5,10]$": 0.183, "$[11,50]$": 0.119, "$[51,100]$": 0.433,
    "$[101,200]$": 1.824, "$[201,500]$": 6.130, "$[501,1000]$": 35.468,
    "TOTAL": 9.527,
}
# ND: optimal_solve_time_s from benchmark_results_ND_final.csv deduped by instance.
SOLVER_ND_BY_N = {
    "$[5,10]$": 0.161, "$[11,50]$": 0.121, "$[51,100]$": 0.101,
    "$[101,200]$": 0.372, "$[201,500]$": 1.420, "$[501,1000]$": 6.884,
    "TOTAL": 1.709,
}
SOLVER_ND_BY_D = {
    "$d=2$": 6.261, "$d=3$--$5$": 2.791, "$d=6$--$10$": 1.777,
    "$d=15$--$25$": 1.485, "$d=30$--$50$": 1.202, "$d=100^{*}$": 1.169,
    "TOTAL": 1.709,
}
# TSPLIB EUC_2D: concorde_time_s, per 4-bin.
SOLVER_TSPLIB_EUC = {
    "$[51,150]$": None, "$[151,400]$": None, "$[401,1500]$": None, "$n > 1500$": None,
    "TOTAL": None,
}  # filled at runtime from the CSV
# TSPLIB non-Euc.
SOLVER_TSPLIB_NONEUC = {
    "CEIL_2D": None, "ATT": None, "GEO": None, "EXPLICIT": None, "TOTAL": None,
}


def pearson_r(y_true, y_pred):
    """Pearson correlation coefficient, bounded [-1, 1]. Undefined for constant predictors."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3:
        return np.nan
    if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def fmt_r(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if x < 0:
        return fr"$-{abs(x):.3f}$"
    return f"{x:.3f}"


def fmt_r_model(x, model_name, applicable_models):
    if model_name not in applicable_models:
        return "---"
    return fmt_r(x)


def fmt_t(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def fmt_r2(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if x < 0:
        return fr"$-{abs(x):.3f}$"
    return f"{x:.3f}"


def fmt_n(x):
    return f"{int(x):,}".replace(",", "{,}")


# ---------------- 2D benchmark ----------------
def compute_2d():
    df = pd.read_csv(REPO / "Generalized_TSP_Analysis/benchmark_results_2D_v3.csv",
                     usecols=["model", "instance", "pred_cost", "true_cost", "prediction_time_s"])
    df = df.rename(columns={"prediction_time_s": "total_time_s"})
    # Derive n from instance name
    df["n"] = df["instance"].str.extract(r"-n(\d+)-").astype(int)
    # Derive mst_length from MST_Ratio rows (pred = 1.075 * mst_length)
    mst_rows = df[df["model"] == "MST_Ratio"][["instance", "pred_cost"]].rename(
        columns={"pred_cost": "mst_pred"}
    )
    mst_rows["mst_length"] = mst_rows["mst_pred"] / 1.075
    df = df.merge(mst_rows[["instance", "mst_length"]], on="instance", how="left")
    return df


def emit_bucket_table(df, bucket_col_fn, bucket_labels, models, caption, label, bucket_var):
    applicable = {"LGBM_V3", "MST_Ratio"}  # models for which r_alpha is defined
    display_names = {"LGBM_V3": "GART 2.0", "MST_Ratio": "MST Ratio",
                     "Cavdar": "Cavdar", "BHH": "BHH", "Chien": "Chien", "Hilbert": "Hilbert"}

    out = []
    out.append(r"\begin{table}[!ht]")
    out.append(r"\centering")
    out.append(fr"\caption{{{caption}}}")
    out.append(fr"\label{{{label}}}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\footnotesize")
    out.append(r"\begin{tabular}{@{}ll rrrrrrr@{}}")
    out.append(r"\toprule")
    out.append(fr"{bucket_var} & Model & $N$ & MAPE (\%) & SDPE (\%) & Median (\%) & $R^2$ & $r_\alpha$ & Time (ms) \\")
    out.append(r"\midrule")

    for lo, hi, lbl in bucket_labels:
        mask = (df["n"] > lo) & (df["n"] <= hi)
        sub = df[mask]
        n_models_here = 0
        model_rows = []
        for m in models:
            sm = sub[sub["model"] == m]
            if len(sm) == 0:
                continue
            n_models_here += 1
            mape = sm["abs_gap_pct"].mean() if "abs_gap_pct" in sm.columns else \
                   (np.abs(sm["pred_cost"] - sm["true_cost"]) / sm["true_cost"] * 100).mean()
            pe = sm["gap_pct"] if "gap_pct" in sm.columns else \
                 ((sm["pred_cost"] - sm["true_cost"]) / sm["true_cost"] * 100)
            sdpe = pe.std(ddof=0)
            med = (sm["abs_gap_pct"] if "abs_gap_pct" in sm.columns else
                   (np.abs(sm["pred_cost"] - sm["true_cost"]) / sm["true_cost"] * 100)).median()
            try:
                r2 = r2_score(sm["true_cost"], sm["pred_cost"])
            except Exception:
                r2 = np.nan
            # Pearson r_alpha
            if "mst_length" in sm.columns and not sm["mst_length"].isna().all() and (sm["mst_length"] > 0).all():
                true_a = sm["true_cost"] / sm["mst_length"]
                if m == "LGBM_V3":
                    pred_a = sm["pred_cost"] / sm["mst_length"]
                    r_alpha = pearson_r(true_a.values, pred_a.values)
                elif m == "MST_Ratio":
                    pred_a = pd.Series([1.075] * len(sm), index=sm.index)
                    r_alpha = pearson_r(true_a.values, pred_a.values)
                else:
                    r_alpha = np.nan
            else:
                r_alpha = np.nan
            t_ms = sm["total_time_s"].mean() * 1000
            model_rows.append((m, len(sm), mape, sdpe, med, r2, r_alpha, t_ms))

        if n_models_here == 0:
            continue
        out.append(fr"    \multirow{{{n_models_here}}}{{*}}{{{lbl}}}")
        for (m, N, mape, sdpe, med, r2, r_alpha, t_ms) in model_rows:
            dn = display_names[m]
            mape_s = f"{mape:.2f}"
            sdpe_s = f"{sdpe:.2f}"
            med_s = f"{med:.2f}"
            if m == "LGBM_V3":
                mape_s = fr"\textbf{{{mape_s}}}"
                sdpe_s = fr"\textbf{{{sdpe_s}}}"
            out.append(f"    & {dn} & {fmt_n(N)} & {mape_s} & {sdpe_s} & {med_s} & {fmt_r2(r2)} & {fmt_r_model(r_alpha, m, applicable)} & {fmt_t(t_ms)} \\\\")
        out.append(r"    \midrule")

    # TOTAL row across all rows in df (no bucket filter)
    tot_rows = []
    for m in models:
        sm = df[df["model"] == m]
        if len(sm) == 0:
            continue
        mape = sm["abs_gap_pct"].mean() if "abs_gap_pct" in sm.columns else \
               (np.abs(sm["pred_cost"] - sm["true_cost"]) / sm["true_cost"] * 100).mean()
        pe = sm["gap_pct"] if "gap_pct" in sm.columns else \
             ((sm["pred_cost"] - sm["true_cost"]) / sm["true_cost"] * 100)
        sdpe = pe.std(ddof=0)
        med = (sm["abs_gap_pct"] if "abs_gap_pct" in sm.columns else
               (np.abs(sm["pred_cost"] - sm["true_cost"]) / sm["true_cost"] * 100)).median()
        try:
            r2 = r2_score(sm["true_cost"], sm["pred_cost"])
        except Exception:
            r2 = np.nan
        if "mst_length" in sm.columns and not sm["mst_length"].isna().all() and (sm["mst_length"] > 0).all():
            true_a = sm["true_cost"] / sm["mst_length"]
            if m == "LGBM_V3":
                pred_a = sm["pred_cost"] / sm["mst_length"]
                r_alpha = pearson_r(true_a.values, pred_a.values)
            elif m == "MST_Ratio":
                pred_a = pd.Series([1.075] * len(sm), index=sm.index)
                r_alpha = pearson_r(true_a.values, pred_a.values)
            else:
                r_alpha = np.nan
        else:
            r_alpha = np.nan
        t_ms = sm["total_time_s"].mean() * 1000
        tot_rows.append((m, len(sm), mape, sdpe, med, r2, r_alpha, t_ms))

    out.append(fr"    \multirow{{{len(tot_rows)}}}{{*}}{{\textbf{{TOTAL}}}}")
    for (m, N, mape, sdpe, med, r2, r_alpha, t_ms) in tot_rows:
        dn = display_names[m]
        mape_s = f"{mape:.2f}"
        sdpe_s = f"{sdpe:.2f}"
        med_s = f"{med:.2f}"
        r2_s = fmt_r2(r2)
        r_s = fmt_r_model(r_alpha, m, applicable)
        t_s = fmt_t(t_ms)
        n_s = fmt_n(N)
        if m == "LGBM_V3":
            dn = fr"\textbf{{{dn}}}"
            n_s = fr"\textbf{{{n_s}}}"
            mape_s = fr"\textbf{{{mape_s}}}"
            sdpe_s = fr"\textbf{{{sdpe_s}}}"
            med_s = fr"\textbf{{{med_s}}}"
            r2_s = fr"\textbf{{{r2_s}}}"
            r_s = fr"\textbf{{{r_s}}}" if r_s != "---" else r_s
            t_s = fr"\textbf{{{t_s}}}"
        out.append(f"    & {dn} & {n_s} & {mape_s} & {sdpe_s} & {med_s} & {r2_s} & {r_s} & {t_s} \\\\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    return "\n".join(out)


# ---------------- 2D: emit table ----------------
print("=" * 70, flush=True)
print("TABLE: tab:2d_by_size", flush=True)
print("=" * 70, flush=True)
df2d = compute_2d()
# ensure the cols we use in emit function
df2d["abs_gap_pct"] = (np.abs(df2d["pred_cost"] - df2d["true_cost"]) / df2d["true_cost"] * 100)
df2d["gap_pct"] = ((df2d["pred_cost"] - df2d["true_cost"]) / df2d["true_cost"] * 100)

buckets_2d = [
    (0, 10, "$[5,10]$"),
    (10, 50, "$[11,50]$"),
    (50, 100, "$[51,100]$"),
    (100, 200, "$[101,200]$"),
    (200, 500, "$[201,500]$"),
    (500, 1000, "$[501,1000]$"),
]
print(emit_bucket_table(
    df2d, None, buckets_2d,
    ["LGBM_V3", "MST_Ratio", "Cavdar", "BHH", "Chien", "Hilbert"],
    r"2D diverse benchmark by instance size (2{,}580 instances). $r_\alpha$ = Pearson correlation between predicted and true $\alpha$, bounded $[-1,1]$.",
    "tab:2d_by_size", r"$n$ bucket"
))

# ---------------- ND benchmark ----------------
print("\n" + "=" * 70, flush=True)
print("TABLE: tab:nd_by_size", flush=True)
print("=" * 70, flush=True)
dfnd = pd.read_csv(REPO / "Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv",
                   usecols=["model", "instance", "pred_cost", "true_cost",
                            "prediction_time_s", "mst_length", "n_customers", "dimension",
                            "abs_gap_pct", "gap_pct"])
dfnd = dfnd.rename(columns={"prediction_time_s": "total_time_s", "n_customers": "n"})
# Filter to 3 models
dfnd = dfnd[dfnd["model"].isin(["LGBM_V3", "MST_Ratio", "Hilbert"])].copy()
dfnd["abs_gap_pct"] = (np.abs(dfnd["pred_cost"] - dfnd["true_cost"]) / dfnd["true_cost"] * 100)
dfnd["gap_pct"] = ((dfnd["pred_cost"] - dfnd["true_cost"]) / dfnd["true_cost"] * 100)

buckets_nd_size = [
    (0, 10, "$[5,10]$"),
    (10, 50, "$[11,50]$"),
    (50, 100, "$[51,100]$"),
    (100, 200, "$[101,200]$"),
    (200, 500, "$[201,500]$"),
    (500, 1000, "$[501,1000]$"),
]
print(emit_bucket_table(
    dfnd, None, buckets_nd_size,
    ["LGBM_V3", "MST_Ratio", "Hilbert"],
    r"ND benchmark by instance size (16{,}907 instances). Same data as Table~\ref{tab:nd_by_dim}.",
    "tab:nd_by_size", r"$n$ bucket"
))

print("\n" + "=" * 70, flush=True)
print("TABLE: tab:nd_by_dim", flush=True)
print("=" * 70, flush=True)

# For ND by dim, need to bucket differently
def emit_nd_by_dim(df):
    applicable = {"LGBM_V3", "MST_Ratio"}
    display_names = {"LGBM_V3": "GART 2.0", "MST_Ratio": "MST Ratio", "Hilbert": "Hilbert"}
    dim_buckets = [
        ("d=2", lambda d: d == 2),
        ("d=3-5", lambda d: (d >= 3) & (d <= 5)),
        ("d=6-10", lambda d: (d >= 6) & (d <= 10)),
        ("d=15-25", lambda d: (d >= 15) & (d <= 25)),
        ("d=30-50", lambda d: (d >= 30) & (d <= 50)),
        ("d=100", lambda d: d == 100),
    ]
    models = ["LGBM_V3", "MST_Ratio", "Hilbert"]
    out = []
    out.append(r"\begin{table}[!ht]")
    out.append(r"\centering")
    out.append(r"\caption{ND benchmark by dimension. Same 16{,}907 instances as Table~\ref{tab:nd_by_size}. $d = 100$ is outside the training range (extrapolation).}")
    out.append(r"\label{tab:nd_by_dim}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\footnotesize")
    out.append(r"\begin{tabular}{@{}ll rrrrrrr@{}}")
    out.append(r"\toprule")
    out.append(r"$d$ bucket & Model & $N$ & MAPE (\%) & SDPE (\%) & Median (\%) & $R^2$ & $r_\alpha$ & Time (ms) \\")
    out.append(r"\midrule")

    for lbl, fn in dim_buckets:
        tex_lbl = "$" + lbl.replace("-", "$--$") + "$"
        if lbl == "d=100":
            tex_lbl = "$d=100^{*}$"
        sub = df[fn(df["dimension"])]
        model_rows = []
        for m in models:
            sm = sub[sub["model"] == m]
            if len(sm) == 0:
                continue
            mape = sm["abs_gap_pct"].mean()
            sdpe = sm["gap_pct"].std(ddof=0)
            med = sm["abs_gap_pct"].median()
            try:
                r2 = r2_score(sm["true_cost"], sm["pred_cost"])
            except Exception:
                r2 = np.nan
            if (sm["mst_length"] > 0).all():
                true_a = sm["true_cost"] / sm["mst_length"]
                if m == "LGBM_V3":
                    pred_a = sm["pred_cost"] / sm["mst_length"]
                    r_alpha = pearson_r(true_a.values, pred_a.values)
                elif m == "MST_Ratio":
                    pred_a = pd.Series([1.075] * len(sm), index=sm.index)
                    r_alpha = pearson_r(true_a.values, pred_a.values)
                else:
                    r_alpha = np.nan
            else:
                r_alpha = np.nan
            t_ms = sm["total_time_s"].mean() * 1000
            model_rows.append((m, len(sm), mape, sdpe, med, r2, r_alpha, t_ms))
        out.append(fr"    \multirow{{{len(model_rows)}}}{{*}}{{{tex_lbl}}}")
        for (m, N, mape, sdpe, med, r2, r_alpha, t_ms) in model_rows:
            dn = display_names[m]
            mape_s = f"{mape:.2f}"
            sdpe_s = f"{sdpe:.2f}"
            med_s = f"{med:.2f}"
            if m == "LGBM_V3":
                mape_s = fr"\textbf{{{mape_s}}}"
                sdpe_s = fr"\textbf{{{sdpe_s}}}"
            out.append(f"    & {dn} & {fmt_n(N)} & {mape_s} & {sdpe_s} & {med_s} & {fmt_r2(r2)} & {fmt_r_model(r_alpha, m, applicable)} & {fmt_t(t_ms)} \\\\")
        out.append(r"    \midrule")

    # TOTAL
    tot_rows = []
    for m in models:
        sm = df[df["model"] == m]
        mape = sm["abs_gap_pct"].mean()
        sdpe = sm["gap_pct"].std(ddof=0)
        med = sm["abs_gap_pct"].median()
        try:
            r2 = r2_score(sm["true_cost"], sm["pred_cost"])
        except Exception:
            r2 = np.nan
        true_a = sm["true_cost"] / sm["mst_length"]
        if m == "LGBM_V3":
            pred_a = sm["pred_cost"] / sm["mst_length"]
            r_alpha = pearson_r(true_a.values, pred_a.values)
        elif m == "MST_Ratio":
            pred_a = pd.Series([1.075] * len(sm), index=sm.index)
            r_alpha = pearson_r(true_a.values, pred_a.values)
        else:
            r_alpha = np.nan
        t_ms = sm["total_time_s"].mean() * 1000
        tot_rows.append((m, len(sm), mape, sdpe, med, r2, r_alpha, t_ms))

    out.append(fr"    \multirow{{{len(tot_rows)}}}{{*}}{{\textbf{{TOTAL}}}}")
    for (m, N, mape, sdpe, med, r2, r_alpha, t_ms) in tot_rows:
        dn = display_names[m]
        mape_s = f"{mape:.2f}"
        sdpe_s = f"{sdpe:.2f}"
        med_s = f"{med:.2f}"
        r2_s = fmt_r2(r2)
        r_s = fmt_r_model(r_alpha, m, applicable)
        t_s = fmt_t(t_ms)
        n_s = fmt_n(N)
        if m == "LGBM_V3":
            dn = fr"\textbf{{{dn}}}"
            n_s = fr"\textbf{{{n_s}}}"
            mape_s = fr"\textbf{{{mape_s}}}"
            sdpe_s = fr"\textbf{{{sdpe_s}}}"
            med_s = fr"\textbf{{{med_s}}}"
            r2_s = fr"\textbf{{{r2_s}}}"
            r_s = fr"\textbf{{{r_s}}}" if r_s != "---" else r_s
            t_s = fr"\textbf{{{t_s}}}"
        out.append(f"    & {dn} & {n_s} & {mape_s} & {sdpe_s} & {med_s} & {r2_s} & {r_s} & {t_s} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\multicolumn{9}{l}{\footnotesize $^{*}$ $d = 100$ is outside the training range; included to test extrapolation.}")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    return "\n".join(out)


print(emit_nd_by_dim(dfnd))

# ---------------- TSPLIB 4-bin ----------------
print("\n" + "=" * 70, flush=True)
print("TABLE: tab:tsplib_by_size (4-bin)", flush=True)
print("=" * 70, flush=True)
dft = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib.csv")
try:
    sup = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib_supplemental.csv")
    dft = pd.concat([dft, sup], ignore_index=True)
except FileNotFoundError:
    pass
dft = dft[dft["edge_weight_type"] == "EUC_2D"].copy()

buckets_tsplib = [
    (0, 150, "$[51,150]$"),
    (150, 400, "$[151,400]$"),
    (400, 1500, "$[401,1500]$"),
    (1500, 10**9, "$n > 1500$"),
]
print(emit_bucket_table(
    dft, None, buckets_tsplib,
    ["LGBM_V3", "MST_Ratio", "Cavdar", "BHH", "Chien", "Hilbert"],
    r"TSPLIB95 EUC\_2D benchmark by instance size (78 instances, four balanced buckets).",
    "tab:tsplib_by_size", r"$n$ bucket"
))

# ---------------- TSPLIB non-Euc ----------------
print("\n" + "=" * 70, flush=True)
print("TABLE: tab:tsplib_nonEuc", flush=True)
print("=" * 70, flush=True)
dfne = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib.csv")
try:
    sup = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib_supplemental.csv")
    dfne = pd.concat([dfne, sup], ignore_index=True)
except FileNotFoundError:
    pass
dfne = dfne[(dfne["model"] == "LGBM_V3") & (dfne["instance"] != "brg180") & (dfne["edge_weight_type"] != "EUC_2D")].copy()

out = []
out.append(r"\begin{table}[!ht]")
out.append(r"\centering")
out.append(r"\caption{GART 2.0 on non-Euclidean TSPLIB95 (32 instances). $r_\alpha$ is Pearson correlation between predicted and true $\alpha$; undefined when the type has fewer than three instances (ATT). Embedded dimension $k$ is chosen per instance by MDS variance retention.}")
out.append(r"\label{tab:tsplib_nonEuc}")
out.append(r"\resizebox{\textwidth}{!}{%")
out.append(r"\footnotesize")
out.append(r"\begin{tabular}{@{}l r r r r r r c@{}}")
out.append(r"\toprule")
out.append(r"Distance type & $N$ & MAPE (\%) & SDPE (\%) & Median (\%) & $r_\alpha$ & Time (ms) & Embed.\ $k$ (avg [range]) \\")
out.append(r"\midrule")

rows = []
for dt in ["CEIL_2D", "ATT", "GEO", "EXPLICIT"]:
    sub = dfne[dfne["edge_weight_type"] == dt]
    if dt == "EXPLICIT":
        # exclude brg180 (already excluded) — EXPLICIT-metric subset
        pass
    if len(sub) == 0:
        continue
    ape = (np.abs(sub["pred_cost"] - sub["true_cost"]) / sub["true_cost"] * 100)
    pe = ((sub["pred_cost"] - sub["true_cost"]) / sub["true_cost"] * 100)
    mape = ape.mean()
    sdpe = pe.std(ddof=0)
    med = ape.median()
    true_a = sub["true_cost"] / sub["mst_length"]
    pred_a = sub["pred_cost"] / sub["mst_length"]
    r_alpha = pearson_r(true_a.values, pred_a.values)
    t_ms = sub["total_time_s"].mean() * 1000
    k_avg = int(sub["feature_dim"].mean()) if "feature_dim" in sub.columns else None
    k_min = int(sub["feature_dim"].min()) if "feature_dim" in sub.columns else None
    k_max = int(sub["feature_dim"].max()) if "feature_dim" in sub.columns else None
    name = "EXPLICIT-metric" if dt == "EXPLICIT" else dt.replace("_", r"\_")
    rows.append((name, len(sub), mape, sdpe, med, r_alpha, t_ms, k_avg, k_min, k_max))

# Also compute TOTAL
total = dfne  # already only LGBM_V3, brg180 excluded
ape_t = (np.abs(total["pred_cost"] - total["true_cost"]) / total["true_cost"] * 100)
pe_t = ((total["pred_cost"] - total["true_cost"]) / total["true_cost"] * 100)
true_at = total["true_cost"] / total["mst_length"]
pred_at = total["pred_cost"] / total["mst_length"]
r_total = pearson_r(true_at.values, pred_at.values)

for (name, N, mape, sdpe, med, r_alpha, t_ms, k_avg, k_min, k_max) in rows:
    out.append(f"{name} & {N} & {mape:.2f} & {sdpe:.2f} & {med:.2f} & {fmt_r(r_alpha)} & {fmt_t(t_ms)} & {k_avg} [{k_min},{k_max}] \\\\")
out.append(r"\midrule")
out.append(fr"\textbf{{TOTAL}} & \textbf{{{len(total)}}} & \textbf{{{ape_t.mean():.2f}}} & \textbf{{{pe_t.std(ddof=0):.2f}}} & \textbf{{{ape_t.median():.2f}}} & \textbf{{{fmt_r(r_total)}}} & \textbf{{{fmt_t(total['total_time_s'].mean()*1000)}}} & --- \\\\")
out.append(r"\bottomrule")
out.append(r"\end{tabular}%")
out.append(r"}")
out.append(r"\end{table}")
print("\n".join(out))
