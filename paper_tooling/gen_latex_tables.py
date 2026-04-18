"""Emit LaTeX tabular blocks from the regenerated CSVs. stdout only; caller splices."""
import pandas as pd
from pathlib import Path

TBL = Path(r"D:/Area-and-Distribution-Free-Estimator-for-TSP/paper_reference/scripts/tables")

def fmt_int(n):
    try:
        n = int(n)
    except:
        return "---"
    return f"{n:,}".replace(",", "{,}")

def fmt_float(x, prec=2):
    try:
        x = float(x)
        if pd.isna(x):
            return "---"
    except:
        return "---"
    return f"{x:.{prec}f}"

def fmt_r2(x):
    try:
        x = float(x)
        if pd.isna(x):
            return "---"
    except:
        return "---"
    if x < 0:
        return f"$-{abs(x):.3f}$"
    return f"{x:.3f}"

def fmt_r2_alpha(x):
    s = str(x).strip()
    if s in ("---", "", "nan"):
        return "---"
    try:
        x = float(x)
    except:
        return "---"
    if pd.isna(x):
        return "---"
    if x < 0:
        return f"$-{abs(x):.3f}$"
    return f"{x:.3f}"

def fmt_time(x):
    try:
        x = float(x)
        if pd.isna(x):
            return "---"
    except:
        return "---"
    if x >= 100:
        return f"{x:.0f}"
    elif x >= 10:
        return f"{x:.1f}"
    else:
        return f"{x:.2f}"

def render_row(r, bold_gart=False):
    m = r["model"]
    n = fmt_int(r["N"])
    mape = fmt_float(r["MAPE_pct"])
    sdpe = fmt_float(r["SDPE_pct"])
    med = fmt_float(r["Median_abs_pct_err"])
    r2 = fmt_r2(r["R2"])
    r2a = fmt_r2_alpha(r["R2_alpha"])
    t = fmt_time(r["time_ms"])
    if bold_gart and "GART" in m:
        mape = f"\\textbf{{{mape}}}"
        sdpe = f"\\textbf{{{sdpe}}}"
    return f"    & {m} & {n} & {mape} & {sdpe} & {med} & {r2} & {r2a} & {t} \\\\"

def emit_table(csv_path, bucket_col, bucket_label, caption, label, models_in_order):
    df = pd.read_csv(csv_path)
    # Skip empty buckets (N=0 or all NaN)
    df = df[df["N"].fillna(0) > 0].copy()
    # Group ordered by unique bucket appearance
    bucket_order = list(dict.fromkeys(df[bucket_col]))
    print(f"% TABLE: {label}")
    for bucket in bucket_order:
        sub = df[df[bucket_col] == bucket]
        rows = []
        for m in models_in_order:
            sel = sub[sub["model"] == m]
            if not sel.empty:
                rows.append(sel.iloc[0])
        if not rows:
            continue
        n_rows = len(rows)
        bucket_tex = str(bucket).replace("<=", "\\le ").replace(">", "$>$").replace("<", "$<$")
        # Escape bracket buckets
        if bucket_tex.startswith("["):
            bucket_tex = "$" + bucket_tex.replace("$>$", ">").replace("$<$", "<") + "$"
        elif bucket == "ALL":
            bucket_tex = "\\textbf{ALL}"
        elif "d=" in str(bucket):
            bucket_tex = "$" + str(bucket).replace("d=", "d=") + "$"
            bucket_tex = bucket_tex.replace("-", "$--$")
        print(f"    \\multirow{{{n_rows}}}{{*}}{{{bucket_tex}}}")
        for i, r in enumerate(rows):
            is_all = str(bucket) == "ALL"
            print(render_row(r, bold_gart=(not is_all) or ("GART" in r["model"])))
        print("    \\midrule")
    print(f"% END TABLE: {label}")
    print()

# Emit for each CSV
print("=" * 60)
print("=== 2D by size ===")
emit_table(TBL / "table_2d_by_size.csv", "n_bucket", "$n$ bucket",
           "2D benchmark by size", "tab:2d_by_size",
           ["GART 2.0", "MST Ratio", "Cavdar", "BHH", "Chien", "Hilbert"])

print("=" * 60)
print("=== ND by size ===")
emit_table(TBL / "table_nd_by_size.csv", "n_bucket", "$n$ bucket",
           "ND by size", "tab:nd_by_size",
           ["GART 2.0", "MST Ratio", "Hilbert"])

print("=" * 60)
print("=== ND by dim ===")
emit_table(TBL / "table_nd_by_dim.csv", "d_bucket", "$d$ bucket",
           "ND by dim", "tab:nd_by_dim",
           ["GART 2.0", "MST Ratio", "Hilbert"])

print("=" * 60)
print("=== TSPLIB by size ===")
emit_table(TBL / "table_tsplib_by_size.csv", "n_bucket", "$n$ bucket",
           "TSPLIB by size", "tab:tsplib_by_size",
           ["GART 2.0", "MST Ratio", "Cavdar", "BHH", "Chien", "Hilbert"])

# Non-Euc: different schema
print("=" * 60)
print("=== TSPLIB non-Euc ===")
df = pd.read_csv(TBL / "table_tsplib_nonEuc.csv")
for _, r in df.iterrows():
    dt = r["edge_weight_type"].replace("_", "\\_")
    n = fmt_int(r["N"])
    mape = fmt_float(r["MAPE_pct"])
    sdpe = fmt_float(r["SDPE_pct"])
    med = fmt_float(r["Median_abs_pct_err"])
    r2a = fmt_r2_alpha(r["R2_alpha"])
    t = fmt_time(r["time_ms"])
    k_avg = fmt_float(r["avg_k"], prec=0)
    k_range = str(r["k_range"])
    print(f"    {dt} & {n} & {mape} & {sdpe} & {med} & {r2a} & {t} & {k_avg} {k_range} \\\\")
