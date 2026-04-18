# GART Paper — Remaining Task Instructions

## Status of current session work (before this file was written)

### Already completed this session
- `tsp_utils_2.py`: `estimate_tsp_mst_ratio` updated to use Delaunay triangulation (O(n log n)) instead of full O(n²) `cdist`. New helper `_delaunay_mst_length` added above the function. `Delaunay` and `csr_matrix` added to imports.
- `tsp_utils_2.py`: `estimate_tsp_kwon` and `estimate_tsp_daganzo` already exist (from prior session).
- TSPLIB: Kwon + Daganzo backfilled into `tsplib_benchmark/results/all_models_tsplib_supplemental.csv` (198 rows total).
- `paper_reference/Area_Free_Main.tex`: Concorde ">1 h (not measured)" cutoff applied to TSPLIB by_size table; Figure 4 removed.

---

## Task 23 — Never-silent refactor of run_all_models_tsplib

**File**: `tsplib_benchmark/run_all_models_tsplib.py`

**What**: Every (instance × model) pair must emit a CSV row. Currently some pairs silently `continue`.

**How**:
1. Add a `_status_row(name, n, ewt, model, status, mode, feat_dim)` helper that returns a dict with all columns set and `gap_pct=NaN`, `pred_cost=NaN`.
2. Define `ALL_MODELS` list at the top of `main()` — all model names including Kwon, Daganzo.
3. Replace every silent `continue` in loops with `rows.append(_status_row(..., status="<reason>"))`.
4. For academic estimators: n>5000 → `status="academic_n_gt_5000"`.
5. Print a summary at the end: per-model ok/total counts.

**Vinel fill for 7 large instances**: Already handled in `run_supplemental_baselines.py`. Re-running that script appends to the supplemental CSV.

---

## Task 24 — Rebuild 2D + ND aggregate tables + CIs; wire into paper

### Step 1: Add Kwon + Daganzo to 2D benchmark

**File**: `run_benchmark_2D_all.py`

**Note**: File has trailing null bytes after `main()` — handle with `open(path, 'rb').read()` then strip nulls before writing, or just edit the text portion directly.

**What to add** in the `schedule = [...]` list, after `('Hilbert', ...)`:
```python
('Kwon',    lambda: academic.estimate_tsp_kwon),
('Daganzo', lambda: academic.estimate_tsp_daganzo),
```

**Caveat**: Kwon was calibrated for n ≤ ~300 and 2D only. For the paper, include Kwon in 2D tables but add footnote that accuracy degrades for large n. Do NOT add Kwon/Daganzo to ND schedule (they're 2D-specific formulas).

### Step 2: Add Kwon + Daganzo to ND benchmark

**File**: `run_benchmark_ND_final.py`

**Decision**: Do NOT add Kwon/Daganzo to ND schedule — they are calibrated for 2D planar distributions only and produce meaningless results in d > 2. Note this in the paper.

### Step 3: Run 2D benchmark

```bash
cd D:/Area-and-Distribution-Free-Estimator-for-TSP
python run_benchmark_2D_all.py
```

Output: `Generalized_TSP_Analysis/benchmark_results_2D_v3.csv`

**Warning**: This takes hours. It runs checkpoint-per-model, so Kwon + Daganzo will create new checkpoint files and be merged into the final CSV. Existing model checkpoints are NOT re-run.

### Step 4: Run ND benchmark (only if ND results are stale)

```bash
python run_benchmark_ND_final.py
```

Output: `Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv`

### Step 5: Compute metrics and update paper tables

**Script to run for verification**:
```bash
python -c "
import pandas as pd, numpy as np
df = pd.read_csv('Generalized_TSP_Analysis/benchmark_results_2D_v3.csv')
models = ['BHH','Cavdar','Chien','Hilbert','MST_Ratio','Vinel','Kwon','Daganzo','Linear_V3','LGBM_V3','Neural_V3','Interp_V3']
for m in models:
    sub = df[df['model']==m]
    if len(sub)==0: continue
    err = (sub['pred_cost']-sub['true_cost'])/sub['true_cost']
    sdpe = err.std(ddof=1)*100
    mape = err.abs().mean()*100
    print(f'{m}: SDPE={sdpe:.1f}% MAPE={mape:.1f}% n={len(sub)}')
"
```

**Paper table locations**:
- 2D by-size table: `\begin{table}...\end{table}` around `tab:2d_by_size` in `Area_Free_Main.tex`
- ND by-size table: `tab:nd_by_size`
- ND by-dim table: `tab:nd_by_dim`

**Format**: Rows for Kwon and Daganzo should appear after Hilbert and before ML models. Only show Kwon in 2D table (not ND).

### Step 6: Regenerate boxplots

**Script**: Write a self-contained script that reads the benchmark CSVs and saves:
- `paper_reference/boxplot_2d_errors.png` — 2D signed relative errors by model
- `paper_reference/boxplot_nd_errors.png` — ND signed relative errors by model
- `paper_reference/boxplot_tsplib_errors.png` — TSPLIB signed relative errors by model

**Template** (adapt existing plot logic in `run_benchmark_2D_all.py` or `run_benchmark_ND_final.py`):
```python
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

MODEL_ORDER = ['BHH','Cavdar','Chien','Hilbert','MST_Ratio','Vinel','Kwon','Daganzo',
               'Linear_V3','LGBM_V3','Neural_V3','Interp_V3']

df = pd.read_csv('Generalized_TSP_Analysis/benchmark_results_2D_v3.csv')
df['signed_err_pct'] = (df['pred_cost'] - df['true_cost']) / df['true_cost'] * 100

fig, ax = plt.subplots(figsize=(12, 5))
order = [m for m in MODEL_ORDER if m in df['model'].unique()]
sns.boxplot(data=df, x='model', y='signed_err_pct', order=order, ax=ax, showfliers=False)
ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
ax.set_xlabel(''); ax.set_ylabel('Signed Relative Error (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('paper_reference/boxplot_2d_errors.png', dpi=150)
```

---

## Task 25 — Fold §4 complexity into §3

**File**: `paper_reference/Area_Free_Main.tex`

**What**: The complexity discussion (inference: 4431 trees × avg depth 16.3) currently lives in §4. Move it to §3.2 (Model Architecture / Training), where the model design is described. Delete the standalone complexity subsection from §4.

**How**: 
1. Find the complexity paragraph in §4 (search for `4431` or `16.3`).
2. Cut it.
3. Paste it into §3.2 after the hyperparameter table or after the α-clip disclosure.
4. Recompile: `cd paper_reference && latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error Area_Free_Main.tex`

---

## Task 26 — Move MDS mechanics to appendix

**File**: `paper_reference/Area_Free_Main.tex`

**What**: The MDS (multidimensional scaling) mechanics paragraph in §3 (feature engineering) is implementation detail, not conceptual. Move to appendix.

**How**:
1. Find the MDS mechanics block (search for `MDS` or `multidimensional scaling` in §3).
2. Cut the paragraph(s).
3. Add to appendix with a forward reference: `(see Appendix~\ref{app:mds} for implementation details)`.
4. Create `\section{MDS Feature Projection}\label{app:mds}` in the appendix.
5. Recompile.

---

## Task 28 — Runtime plot

**What**: Bar or scatter showing GART inference time vs Concorde/LKH-3 solve time. Stacked bars: feature_time + inference_time.

**Data source**: TSPLIB results CSV — columns `feature_time_s`, `inference_time_s`, `concorde_time_s`.

**Script location to write**: `paper_reference/plot_runtime.py` (self-contained, saves PNG to `paper_reference/`).

**Key design**:
- X-axis: instance size n (binned: [10,50], [51,200], [201,400], [401,1500], 1500+)
- Y-axis: median time in seconds (log scale)
- GART: stacked bar (feature_time + inference_time)
- Concorde: single bar (concorde_time_s); for n>1500, show ">1 h" annotation
- Color: GART teal, Concorde gray

---

## Task 29 — Training-data section rewrite

**File**: `paper_reference/Area_Free_Main.tex`

**What**: The training-data description section should clearly state:
- Training instances: n ∈ {10..1000}, d ∈ {2..50}
- Test instances include d=100 (held out; model never saw d > 50 in training)
- Distributions: uniform random, clustered, structured (describe how instances were generated)
- Total training set: ~59,247 instances; test set: ~14,457 instances

**Where**: §3.1 or the dataset/training section — search for `n \leq 1000` or `training` in §3.

---

## Compile command (run after every paper edit)

```bash
cd D:/Area-and-Distribution-Free-Estimator-for-TSP/paper_reference
latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error Area_Free_Main.tex
```

Then open the PDF to verify.

---

## Key file paths

| Purpose | Path |
|---|---|
| LaTeX source | `paper_reference/Area_Free_Main.tex` |
| BibTeX | `paper_reference/references.bib` |
| Academic estimators | `tsp_utils_2.py` |
| 2D benchmark runner | `run_benchmark_2D_all.py` |
| ND benchmark runner | `run_benchmark_ND_final.py` |
| TSPLIB runner | `tsplib_benchmark/run_all_models_tsplib.py` |
| TSPLIB supplemental | `tsplib_benchmark/run_supplemental_baselines.py` |
| 2D results CSV | `Generalized_TSP_Analysis/benchmark_results_2D_v3.csv` |
| ND results CSV | `Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv` |
| TSPLIB canonical CSV | `tsplib_benchmark/results/all_models_tsplib_20260412_155608.csv` |
| TSPLIB supplemental CSV | `tsplib_benchmark/results/all_models_tsplib_supplemental.csv` |
| Boxplot 2D | `paper_reference/boxplot_2d_errors.png` |
| Boxplot ND | `paper_reference/boxplot_nd_errors.png` |
| Boxplot TSPLIB | `paper_reference/boxplot_tsplib_errors.png` |
