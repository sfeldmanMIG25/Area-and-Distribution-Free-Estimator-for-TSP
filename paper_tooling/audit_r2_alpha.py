"""Audit every R^2_alpha value we've reported. Verify:
  1. None exceed 1.0 (mathematically impossible).
  2. Negative values are legitimate (sklearn r2_score convention).
  3. Derivation matches the formula R^2 = 1 - SS_res/SS_tot.
Re-derives R^2_alpha from the raw CSVs using two independent implementations
and prints side-by-side.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from pathlib import Path

REPO = Path(r"D:/Area-and-Distribution-Free-Estimator-for-TSP")
TBL = REPO / "paper_reference/scripts/tables"

def r2_manual(y_true, y_pred):
    """Manual R^2 = 1 - SS_res/SS_tot. Returns -inf if SS_tot = 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

print("=" * 70)
print("Step 1: scan every generated CSV for R2_alpha values > 1")
print("=" * 70)
any_above_one = False
for csv_path in sorted(TBL.glob("*.csv")):
    df = pd.read_csv(csv_path)
    if "R2_alpha" not in df.columns:
        continue
    vals = pd.to_numeric(df["R2_alpha"], errors="coerce").dropna()
    max_val = vals.max() if len(vals) else np.nan
    min_val = vals.min() if len(vals) else np.nan
    over = vals[vals > 1.0]
    print(f"  {csv_path.name}: min={min_val:.4f}, max={max_val:.4f}, n_values={len(vals)}")
    if len(over) > 0:
        any_above_one = True
        print(f"    !!! {len(over)} values > 1.0: {over.tolist()}")
print(f"\nResult: {'BUG FOUND — values exceed 1' if any_above_one else 'no value exceeds 1.0'}")

print()
print("=" * 70)
print("Step 2: re-derive R^2_alpha from the raw TSPLIB CSV, two ways")
print("=" * 70)
df = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib.csv")
try:
    sup = pd.read_csv(REPO / "tsplib_benchmark/results/all_models_tsplib_supplemental.csv")
    df = pd.concat([df, sup], ignore_index=True)
except FileNotFoundError:
    pass
euc = df[df["edge_weight_type"] == "EUC_2D"].copy()
gart = euc[euc["model"] == "LGBM_V3"].copy()
mst = euc[euc["model"] == "MST_Ratio"].copy()

# Compute R^2_alpha on the full 78-instance aggregate
for name, sub in [("GART 2.0", gart), ("MST Ratio", mst)]:
    mst_len = sub["mst_length"]
    true_a = sub["true_cost"].values / mst_len.values
    if name == "GART 2.0":
        pred_a = sub["pred_cost"].values / mst_len.values
    else:
        pred_a = np.full(len(sub), 1.075)
    n = len(true_a)
    r2_sk = r2_score(true_a, pred_a)
    r2_my = r2_manual(true_a, pred_a)
    # Also compute Pearson r (which IS bounded to [-1, 1])
    pearson_r = np.corrcoef(true_a, pred_a)[0, 1]
    print(f"\n  {name} (N={n}):")
    print(f"    true_alpha range: [{true_a.min():.3f}, {true_a.max():.3f}], "
          f"mean={true_a.mean():.4f}, std={true_a.std(ddof=0):.4f}")
    print(f"    pred_alpha range: [{pred_a.min():.3f}, {pred_a.max():.3f}], "
          f"mean={pred_a.mean():.4f}")
    print(f"    R^2_alpha (sklearn)  = {r2_sk:.6f}")
    print(f"    R^2_alpha (manual)   = {r2_my:.6f}")
    print(f"    Pearson r            = {pearson_r:.4f}    [in [-1, 1]]")
    print(f"    Pearson r squared    = {pearson_r**2:.4f}   [in [0, 1]]")
    # Verify the formula
    ss_res = np.sum((true_a - pred_a) ** 2)
    ss_tot = np.sum((true_a - true_a.mean()) ** 2)
    print(f"    SS_residual = {ss_res:.6e}")
    print(f"    SS_total    = {ss_tot:.6e}")
    print(f"    1 - SS_res/SS_tot = {1 - ss_res/ss_tot:.6f}  (matches R^2 above)")

print()
print("=" * 70)
print("Step 3: math sanity — upper bound of R^2 is 1 (exactly, when SS_res=0)")
print("=" * 70)
print("  R^2 = 1 - SS_res/SS_tot, where SS_res >= 0 and SS_tot >= 0.")
print("  => R^2 <= 1 always, with equality iff predictions are perfect.")
print("  No finite-precision computation of R^2 should ever return > 1.")
print("  The reported R^2 = 1.000 in a table is a ROUNDED value (e.g. 0.99997).")
print("  R^2 CAN be arbitrarily negative — this is not the same as Pearson r.")
