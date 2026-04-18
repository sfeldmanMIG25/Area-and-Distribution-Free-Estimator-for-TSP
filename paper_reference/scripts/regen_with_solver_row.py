"""Regenerate all five LaTeX tables. Solver time is integrated as a single divider row
at the end of each bucket (inside the Time column, in seconds), NOT as a separate column.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

REPO = Path(r"D:/Area-and-Distribution-Free-Estimator-for-TSP")


def pearson_r(y_true, y_pred):
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


def fmt_r2(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if x < 0:
        return fr"$-{abs(x):.3f}$"
    return f"{x:.3f}"


def fmt_r_model(x, model_name, applicable):
    if model_name not in applicable:
        return "---"
    return fmt_r(x)


def fmt_t_ms(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def fmt_solver_cell(s):
    """Format solver wall-clock in scientific-notation milliseconds (for direct comparison with model Time column).
    Returns strings like '$6.6\\times 10^{2}$~ms', '$2.2\\times 10^{4}$~ms', '$1.3\\times 10^{9}$~ms'."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "---"
    ms = s * 1000.0
    if ms <= 0:
        return "---"
    exp = int(np.floor(np.log10(ms)))
    mant = ms / 10**exp
    # Normalize so 10.0 → 1.0e+1
    if mant >= 10:
        mant /= 10
        exp += 1
    return fr"${mant:.1f}\times 10^{{{exp}}}$~ms"


def fmt_n(x):
    return f"{int(x):,}".replace(",", "{,}")


DISPLAY = {"LGBM_V3": "GART 2.0", "MST_Ratio": "MST Ratio", "Cavdar": "Cavdar",
           "BHH": "BHH", "Chien": "Chien", "Hilbert": "Hilbert"}
APPLICABLE_R_ALPHA = {"LGBM_V3", "MST_Ratio"}


def compute_bucket_row(sub, model_code):
    sub = sub.dropna(subset=["true_cost", "pred_cost"]).copy()
    sub["true_cost"] = pd.to_numeric(sub["true_cost"], errors="coerce")
    sub["pred_cost"] = pd.to_numeric(sub["pred_cost"], errors="coerce")
    sub = sub.dropna(subset=["true_cost", "pred_cost"])
    mape = sub["abs_gap_pct"].mean()
    sdpe = sub["gap_pct"].std(ddof=0)
    med = sub["abs_gap_pct"].median()
    try:
        r2 = r2_score(sub["true_cost"], sub["pred_cost"])
    except Exception:
        r2 = np.nan
    if "mst_length" in sub.columns and (sub["mst_length"] > 0).all() and not sub["mst_length"].isna().any():
        true_a = sub["true_cost"] / sub["mst_length"]
        if model_code == "LGBM_V3":
            pred_a = sub["pred_cost"] / sub["mst_length"]
            r_alpha = pearson_r(true_a.values, pred_a.values)
        elif model_code == "MST_Ratio":
            pred_a = pd.Series([1.075] * len(sub), index=sub.index)
            r_alpha = pearson_r(true_a.values, pred_a.values)
        else:
            r_alpha = np.nan
    else:
        r_alpha = np.nan
    t_ms = sub["total_time_s"].mean() * 1000
    return dict(N=len(sub), MAPE=mape, SDPE=sdpe, MED=med, R2=r2, R2A=r_alpha, T=t_ms)


def emit_model_row(stats, model_code, bold_all=False):
    dn = DISPLAY[model_code]
    mape = f"{stats['MAPE']:.2f}"
    sdpe = f"{stats['SDPE']:.2f}"
    med = f"{stats['MED']:.2f}"
    r2 = fmt_r2(stats["R2"])
    r_a = fmt_r_model(stats["R2A"], model_code, APPLICABLE_R_ALPHA)
    t = fmt_t_ms(stats["T"])
    n = fmt_n(stats["N"])
    if model_code == "LGBM_V3":
        mape = fr"\textbf{{{mape}}}"
        sdpe = fr"\textbf{{{sdpe}}}"
    if bold_all and model_code == "LGBM_V3":
        dn = fr"\textbf{{{dn}}}"
        n = fr"\textbf{{{n}}}"
        med = fr"\textbf{{{med}}}"
        r2 = fr"\textbf{{{r2}}}"
        if r_a != "---":
            r_a = fr"\textbf{{{r_a}}}"
        t = fr"\textbf{{{t}}}"
    return f"    & {dn} & {n} & {mape} & {sdpe} & {med} & {r2} & {r_a} & {t} \\\\"


def emit_solver_row(solver_s, solver_label="Optimal solver"):
    cell = fmt_solver_cell(solver_s)
    # Light-italic divider row. Time column shows solver wall-clock in seconds (unit suffixed).
    return fr"    & \textit{{{solver_label}}} & --- & --- & --- & --- & --- & --- & {cell} \\"


def emit_bucket_table(df, buckets, models, caption, label, bucket_var,
                     solver_per_bucket, solver_total,
                     solver_label_per_bucket=None):
    solver_label_per_bucket = solver_label_per_bucket or {}
    out = [r"\begin{table}[!ht]",
           r"\centering",
           fr"\caption{{{caption}}}",
           fr"\label{{{label}}}",
           r"\resizebox{\textwidth}{!}{%",
           r"\footnotesize",
           r"\begin{tabular}{@{}ll rrrrrrr@{}}",
           r"\toprule",
           fr"{bucket_var} & Model & $N$ & MAPE (\%) & SDPE (\%) & Median (\%) & $R^2$ & $r_\alpha$ & Time (ms) \\",
           r"\midrule"]

    for (lo, hi, lbl) in buckets:
        mask = (df["n"] > lo) & (df["n"] <= hi)
        sub = df[mask]
        rows = []
        for m in models:
            sm = sub[sub["model"] == m]
            if len(sm) == 0:
                continue
            stats = compute_bucket_row(sm, m)
            rows.append((m, stats))
        if not rows:
            continue
        solver_label = solver_label_per_bucket.get(lbl, "Optimal solver")
        # multirow spans model rows + 1 solver row
        out.append(fr"    \multirow{{{len(rows) + 1}}}{{*}}{{{lbl}}}")
        for m, stats in rows:
            out.append(emit_model_row(stats, m))
        out.append(emit_solver_row(solver_per_bucket.get(lbl), solver_label))
        out.append(r"    \midrule")

    # TOTAL row block
    tot_rows = []
    for m in models:
        sm = df[df["model"] == m]
        if len(sm) == 0:
            continue
        stats = compute_bucket_row(sm, m)
        tot_rows.append((m, stats))
    out.append(fr"    \multirow{{{len(tot_rows) + 1}}}{{*}}{{\textbf{{TOTAL}}}}")
    for m, stats in tot_rows:
        out.append(emit_model_row(stats, m, bold_all=True))
    out.append(emit_solver_row(solver_total, "Optimal solver"))

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    return "\n".join(out)


# ============ 2D ============
df2d = pd.read_csv(REPO / "Generalized_TSP_Analysis/benchmark_results_2D_v3.csv",
                   usecols=["model", "instance", "pred_cost", "true_cost",
                            "prediction_time_s", "optimal_solve_time_s"])
df2d["total_time_s"] = df2d["prediction_time_s"]
df2d["n"] = df2d["instance"].str.extract(r"-n(\d+)-").astype(int)
df2d["abs_gap_pct"] = np.abs(df2d["pred_cost"] - df2d["true_cost"]) / df2d["true_cost"] * 100
df2d["gap_pct"] = (df2d["pred_cost"] - df2d["true_cost"]) / df2d["true_cost"] * 100
mst_rows = df2d[df2d["model"] == "MST_Ratio"][["instance", "pred_cost"]].rename(columns={"pred_cost": "mst_pred"})
mst_rows["mst_length"] = mst_rows["mst_pred"] / 1.075
df2d = df2d.merge(mst_rows[["instance", "mst_length"]], on="instance", how="left")

per_inst_2d = df2d[["instance", "n", "optimal_solve_time_s"]].drop_duplicates("instance")
buckets_2d = [(0, 10, "$[5,10]$"), (10, 50, "$[11,50]$"), (50, 100, "$[51,100]$"),
              (100, 200, "$[101,200]$"), (200, 500, "$[201,500]$"), (500, 1000, "$[501,1000]$")]
solver_2d = {lbl: per_inst_2d[(per_inst_2d["n"] > lo) & (per_inst_2d["n"] <= hi)]["optimal_solve_time_s"].mean()
             for lo, hi, lbl in buckets_2d}
solver_2d_total = per_inst_2d["optimal_solve_time_s"].mean()

print("=" * 70)
print("TABLE: tab:2d_by_size")
print("=" * 70)
print(emit_bucket_table(
    df2d, buckets_2d,
    ["LGBM_V3", "MST_Ratio", "Cavdar", "BHH", "Chien", "Hilbert"],
    r"2D diverse benchmark by instance size (2{,}580 instances). The \textit{Optimal solver} row shows mean wall-clock time of the ground-truth solver per bucket (\textsc{Concorde} or \textsc{LKH-3}, whichever produced the training tour).",
    "tab:2d_by_size", r"$n$ bucket",
    solver_2d, solver_2d_total
))

# ============ ND ============
dfnd = pd.read_csv(REPO / "Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv",
                   usecols=["model", "instance", "pred_cost", "true_cost",
                            "prediction_time_s", "mst_length", "n_customers", "dimension",
                            "abs_gap_pct", "gap_pct", "optimal_solve_time_s"],
                   low_memory=False)
dfnd["total_time_s"] = dfnd["prediction_time_s"]
dfnd = dfnd.rename(columns={"n_customers": "n"})
dfnd["solve_s"] = pd.to_numeric(dfnd["optimal_solve_time_s"], errors="coerce")
dfnd["n"] = pd.to_numeric(dfnd["n"], errors="coerce")
dfnd["dimension"] = pd.to_numeric(dfnd["dimension"], errors="coerce")
dfnd = dfnd[dfnd["model"].isin(["LGBM_V3", "MST_Ratio", "Hilbert"])].copy()

per_inst_nd = dfnd.dropna(subset=["solve_s", "n", "dimension"]).drop_duplicates("instance")[
    ["instance", "n", "dimension", "solve_s"]]
buckets_nd_size = [(0, 10, "$[5,10]$"), (10, 50, "$[11,50]$"), (50, 100, "$[51,100]$"),
                   (100, 200, "$[101,200]$"), (200, 500, "$[201,500]$"), (500, 1000, "$[501,1000]$")]
solver_nd_size = {lbl: per_inst_nd[(per_inst_nd["n"] > lo) & (per_inst_nd["n"] <= hi)]["solve_s"].mean()
                  for lo, hi, lbl in buckets_nd_size}
solver_nd_size_total = per_inst_nd["solve_s"].mean()

print("\n" + "=" * 70)
print("TABLE: tab:nd_by_size")
print("=" * 70)
print(emit_bucket_table(
    dfnd, buckets_nd_size,
    ["LGBM_V3", "MST_Ratio", "Hilbert"],
    r"ND benchmark by instance size (16{,}907 instances). Same data as Table~\ref{tab:nd_by_dim}. \textit{Optimal solver} row = mean \textsc{Concorde}/\textsc{LKH-3} wall time per bucket.",
    "tab:nd_by_size", r"$n$ bucket",
    solver_nd_size, solver_nd_size_total
))


def emit_nd_by_dim(df, per_inst):
    dim_buckets = [
        ("$d=2$", lambda d: d == 2),
        ("$d=3$--$5$", lambda d: (d >= 3) & (d <= 5)),
        ("$d=6$--$10$", lambda d: (d >= 6) & (d <= 10)),
        ("$d=15$--$25$", lambda d: (d >= 15) & (d <= 25)),
        ("$d=30$--$50$", lambda d: (d >= 30) & (d <= 50)),
        ("$d=100^{*}$", lambda d: d == 100),
    ]
    models = ["LGBM_V3", "MST_Ratio", "Hilbert"]
    out = [r"\begin{table}[!ht]", r"\centering",
           r"\caption{ND benchmark by dimension. Same 16{,}907 instances as Table~\ref{tab:nd_by_size}. $d = 100$ is outside the training range (extrapolation). \textit{Optimal solver} row shows mean \textsc{Concorde}/\textsc{LKH-3} wall time.}",
           r"\label{tab:nd_by_dim}",
           r"\resizebox{\textwidth}{!}{%",
           r"\footnotesize",
           r"\begin{tabular}{@{}ll rrrrrrr@{}}",
           r"\toprule",
           r"$d$ bucket & Model & $N$ & MAPE (\%) & SDPE (\%) & Median (\%) & $R^2$ & $r_\alpha$ & Time (ms) \\",
           r"\midrule"]
    for lbl, fn in dim_buckets:
        sub = df[fn(df["dimension"])]
        psub = per_inst[fn(per_inst["dimension"])]
        solver_s = psub["solve_s"].mean()
        rows = []
        for m in models:
            sm = sub[sub["model"] == m]
            if len(sm) == 0:
                continue
            stats = compute_bucket_row(sm, m)
            rows.append((m, stats))
        out.append(fr"    \multirow{{{len(rows) + 1}}}{{*}}{{{lbl}}}")
        for m, stats in rows:
            out.append(emit_model_row(stats, m))
        out.append(emit_solver_row(solver_s))
        out.append(r"    \midrule")

    tot_rows = []
    solver_total = per_inst["solve_s"].mean()
    for m in models:
        sm = df[df["model"] == m]
        stats = compute_bucket_row(sm, m)
        tot_rows.append((m, stats))
    out.append(fr"    \multirow{{{len(tot_rows) + 1}}}{{*}}{{\textbf{{TOTAL}}}}")
    for m, stats in tot_rows:
        out.append(emit_model_row(stats, m, bold_all=True))
    out.append(emit_solver_row(solver_total))

    out.append(r"\bottomrule")
    out.append(r"\multicolumn{9}{l}{\footnotesize $^{*}$ $d = 100$ is outside the training range; included to test extrapolation.}")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table}")
    return "\n".join(out)


print("\n" + "=" * 70)
print("TABLE: tab:nd_by_dim")
print("=" * 70)
print(emit_nd_by_dim(dfnd, per_inst_nd))

# ============ TSPLIB EUC_2D ============
dft = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib.csv")
try:
    sup = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib_supplemental.csv")
    dft = pd.concat([dft, sup], ignore_index=True)
except FileNotFoundError:
    pass
dft = dft[dft["edge_weight_type"] == "EUC_2D"].copy()
if "abs_gap_pct" not in dft.columns or dft["abs_gap_pct"].isna().all():
    dft["abs_gap_pct"] = np.abs(dft["pred_cost"] - dft["true_cost"]) / dft["true_cost"] * 100
if "gap_pct" not in dft.columns or dft["gap_pct"].isna().all():
    dft["gap_pct"] = (dft["pred_cost"] - dft["true_cost"]) / dft["true_cost"] * 100

buckets_tsplib = [(0, 150, "$[51,150]$"), (150, 400, "$[151,400]$"),
                  (400, 1500, "$[401,1500]$"), (1500, 10**9, "$n > 1500$")]

# Placeholder: fresh solver times to be filled in by background agent
# For now, use existing concorde_time_s (mean where valid) as a best-effort placeholder.
per_inst_ts = dft[dft["model"] == "LGBM_V3"][["instance", "n", "concorde_time_s"]].drop_duplicates("instance")
solver_ts = {}
for lo, hi, lbl in buckets_tsplib:
    m = per_inst_ts[(per_inst_ts["n"] > lo) & (per_inst_ts["n"] <= hi)]
    v = m["concorde_time_s"].mean()
    solver_ts[lbl] = v
solver_ts_total = per_inst_ts["concorde_time_s"].mean()

# Try to read fresh solver_wall_times.csv if present
fresh_path = REPO / "tsplib_benchmark/results/solver_wall_times.csv"
fresh_solver_ts = None
if fresh_path.exists():
    try:
        dfs = pd.read_csv(fresh_path)
        dfs["n"] = pd.to_numeric(dfs["n"], errors="coerce")
        dfs["best_time_s"] = pd.to_numeric(dfs["best_time_s"], errors="coerce")
        dfs = dfs.dropna(subset=["n", "best_time_s"])
        fresh_solver_ts = {}
        for lo, hi, lbl in buckets_tsplib:
            m = dfs[(dfs["n"] > lo) & (dfs["n"] <= hi)]
            if len(m) > 0:
                fresh_solver_ts[lbl] = m["best_time_s"].mean()
        print(f"[info] loaded fresh solver times from {fresh_path.name}: {fresh_solver_ts}")
    except Exception as exc:
        print(f"[warn] could not parse {fresh_path.name}: {exc}")

if fresh_solver_ts:
    # Merge: fresh overrides for buckets where n <= 1000 can be measured
    for k, v in fresh_solver_ts.items():
        solver_ts[k] = v

print("\n" + "=" * 70)
print("TABLE: tab:tsplib_by_size")
print("=" * 70)
print(emit_bucket_table(
    dft, buckets_tsplib,
    ["LGBM_V3", "MST_Ratio", "Cavdar", "BHH", "Chien", "Hilbert"],
    r"TSPLIB95 EUC\_2D benchmark by instance size (78 instances, four balanced buckets). \textit{Optimal solver} row shows the faster of \textsc{Concorde} and \textsc{LKH-3} (when both find the same optimum) per bucket, measured on this machine for $n \le 1000$; larger buckets use the training-pipeline records (\textsc{Concorde} where available).",
    "tab:tsplib_by_size", r"$n$ bucket",
    solver_ts, solver_ts_total
))

# ============ TSPLIB non-Euc ============
dfne = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib.csv")
try:
    sup = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib_supplemental.csv")
    dfne = pd.concat([dfne, sup], ignore_index=True)
except FileNotFoundError:
    pass
dfne = dfne[(dfne["model"] == "LGBM_V3") & (dfne["instance"] != "brg180") & (dfne["edge_weight_type"] != "EUC_2D")].copy()

out = [r"\begin{table}[!ht]",
       r"\centering",
       r"\caption{GART 2.0 on non-Euclidean TSPLIB95 (32 instances). $r_\alpha$ = Pearson correlation; undefined when $N < 3$ (ATT). Embedded dimension $k$ chosen by MDS variance retention. \textsc{Concorde} is inapplicable to non-Euclidean distance types; solver time is unavailable.}",
       r"\label{tab:tsplib_nonEuc}",
       r"\resizebox{\textwidth}{!}{%",
       r"\footnotesize",
       r"\begin{tabular}{@{}l r r r r r r c@{}}",
       r"\toprule",
       r"Distance type & $N$ & MAPE (\%) & SDPE (\%) & Median (\%) & $r_\alpha$ & Time (ms) & Embed.\ $k$ (avg [range]) \\",
       r"\midrule"]

for dt in ["CEIL_2D", "ATT", "GEO", "EXPLICIT"]:
    sub = dfne[dfne["edge_weight_type"] == dt]
    if len(sub) == 0:
        continue
    ape = np.abs(sub["pred_cost"] - sub["true_cost"]) / sub["true_cost"] * 100
    pe = (sub["pred_cost"] - sub["true_cost"]) / sub["true_cost"] * 100
    true_a = sub["true_cost"] / sub["mst_length"]
    pred_a = sub["pred_cost"] / sub["mst_length"]
    r_alpha = pearson_r(true_a.values, pred_a.values)
    t_ms = sub["total_time_s"].mean() * 1000
    k_avg = int(sub["feature_dim"].mean())
    k_min = int(sub["feature_dim"].min())
    k_max = int(sub["feature_dim"].max())
    name = "EXPLICIT-metric" if dt == "EXPLICIT" else dt.replace("_", r"\_")
    out.append(f"{name} & {len(sub)} & {ape.mean():.2f} & {pe.std(ddof=0):.2f} & {ape.median():.2f} & {fmt_r(r_alpha)} & {fmt_t_ms(t_ms)} & {k_avg} [{k_min},{k_max}] \\\\")

total = dfne
ape_t = np.abs(total["pred_cost"] - total["true_cost"]) / total["true_cost"] * 100
pe_t = (total["pred_cost"] - total["true_cost"]) / total["true_cost"] * 100
true_at = total["true_cost"] / total["mst_length"]
pred_at = total["pred_cost"] / total["mst_length"]
r_total = pearson_r(true_at.values, pred_at.values)
out.append(r"\midrule")
out.append(fr"\textbf{{TOTAL}} & \textbf{{{len(total)}}} & \textbf{{{ape_t.mean():.2f}}} & \textbf{{{pe_t.std(ddof=0):.2f}}} & \textbf{{{ape_t.median():.2f}}} & \textbf{{{fmt_r(r_total)}}} & \textbf{{{fmt_t_ms(total['total_time_s'].mean()*1000)}}} & --- \\\\")
out.append(r"\bottomrule")
out.append(r"\end{tabular}%")
out.append(r"}")
out.append(r"\end{table}")

print("\n" + "=" * 70)
print("TABLE: tab:tsplib_nonEuc")
print("=" * 70)
print("\n".join(out))
