"""Emit new 4-bin TSPLIB by-size LaTeX table with all six models and per-bucket R^2_alpha."""
import pandas as pd, numpy as np
from sklearn.metrics import r2_score

df = pd.read_csv(r'D:/Area-and-Distribution-Free-Estimator-for-TSP/tsplib_benchmark/results/all_models_tsplib.csv')
try:
    sup = pd.read_csv(r'D:/Area-and-Distribution-Free-Estimator-for-TSP/tsplib_benchmark/results/all_models_tsplib_supplemental.csv')
    df = pd.concat([df, sup], ignore_index=True)
except FileNotFoundError:
    pass

euc = df[df['edge_weight_type'] == 'EUC_2D'].copy()

MODELS = [('LGBM_V3', 'GART 2.0'), ('MST_Ratio', 'MST Ratio'),
          ('Cavdar', 'Cavdar'), ('BHH', 'BHH'),
          ('Chien', 'Chien'), ('Hilbert', 'Hilbert')]

def r2_alpha(sub):
    if 'mst_length' not in sub.columns: return float('nan')
    mst = sub['mst_length']
    if (mst <= 0).any() or mst.isna().any(): return float('nan')
    true_a = sub['true_cost'] / mst
    if sub['model'].iloc[0] == 'LGBM_V3':
        pred_a = sub['pred_cost'] / mst
    elif sub['model'].iloc[0] == 'MST_Ratio':
        pred_a = pd.Series([1.075] * len(sub), index=sub.index)
    else:
        return float('nan')
    if len(true_a) < 3: return float('nan')
    if true_a.std() < 1e-6: return float('nan')
    try:
        return r2_score(true_a, pred_a)
    except Exception:
        return float('nan')

def bucket_stats(sub):
    N = len(sub)
    if N == 0: return None
    ape = sub['abs_gap_pct']
    pe = sub['gap_pct']
    try:
        r2 = r2_score(sub['true_cost'], sub['pred_cost'])
    except Exception:
        r2 = float('nan')
    return dict(
        N=N,
        MAPE=ape.mean(),
        SDPE=pe.std(ddof=0),
        MED=ape.median(),
        R2=r2,
        R2A=r2_alpha(sub),
        T=sub['total_time_s'].mean() * 1000,
    )

bounds = [(0, 150), (150, 400), (400, 1500), (1500, 10**9)]
labels = ['$[51,150]$', '$[151,400]$', '$[401,1500]$', '$n > 1500$']

def fmt_r2(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '---'
    if x < 0: return fr'$-{abs(x):.3f}$'
    return f'{x:.3f}'

def fmt_r2a(x, name):
    if name not in ('GART 2.0', 'MST Ratio'): return '---'
    return fmt_r2(x)

def fmt_t(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '---'
    if x >= 100: return f'{x:.0f}'
    if x >= 10: return f'{x:.1f}'
    return f'{x:.2f}'

def fmt_n(x):
    return f'{x:,}'.replace(',', '{,}')

out = []
out.append(r'\begin{table}[!ht]')
out.append(r'\centering')
out.append(r'\caption{TSPLIB95 EUC\_2D benchmark by instance size (78 instances, four balanced buckets).}')
out.append(r'\label{tab:tsplib_by_size}')
out.append(r'\resizebox{\textwidth}{!}{%')
out.append(r'\footnotesize')
out.append(r'\begin{tabular}{@{}ll rrrrrrr@{}}')
out.append(r'\toprule')
out.append(r'$n$ bucket & Model & $N$ & MAPE (\%) & SDPE (\%) & Median (\%) & $R^2$ & $R^2_\alpha$ & Time (ms) \\')
out.append(r'\midrule')

def fmt_row(stats, name, is_gart=False, bold_all=False):
    mape = f'{stats["MAPE"]:.2f}'
    sdpe = f'{stats["SDPE"]:.2f}'
    med = f'{stats["MED"]:.2f}'
    r2 = fmt_r2(stats['R2'])
    r2a = fmt_r2a(stats['R2A'], name)
    t = fmt_t(stats['T'])
    n = fmt_n(stats['N'])
    if is_gart:
        mape = fr'\textbf{{{mape}}}'
        sdpe = fr'\textbf{{{sdpe}}}'
    if bold_all:
        name = fr'\textbf{{{name}}}'
        n = fr'\textbf{{{n}}}'
        mape = fr'\textbf{{{mape.replace(r"\textbf", "").strip("{}")}}}' if r'\textbf' in mape else fr'\textbf{{{mape}}}'
        sdpe = fr'\textbf{{{sdpe.replace(r"\textbf", "").strip("{}")}}}' if r'\textbf' in sdpe else fr'\textbf{{{sdpe}}}'
        med = fr'\textbf{{{med}}}'
        r2 = fr'\textbf{{{r2}}}'
        r2a = fr'\textbf{{{r2a}}}' if r2a != '---' else r2a
        t = fr'\textbf{{{t}}}'
    return f'    & {name} & {n} & {mape} & {sdpe} & {med} & {r2} & {r2a} & {t} \\\\'

for i, (lo, hi) in enumerate(bounds):
    mask = (euc['n'] > lo) & (euc['n'] <= hi)
    sub_all = euc[mask]
    out.append(fr'    \multirow{{{len(MODELS)}}}{{*}}{{{labels[i]}}}')
    for code, name in MODELS:
        sub = sub_all[sub_all['model'] == code]
        stats = bucket_stats(sub)
        if stats is None: continue
        out.append(fmt_row(stats, name, is_gart=(code == 'LGBM_V3')))
    out.append(r'    \midrule')

out.append(fr'    \multirow{{{len(MODELS)}}}{{*}}{{\textbf{{TOTAL}}}}')
for code, name in MODELS:
    sub = euc[euc['model'] == code]
    stats = bucket_stats(sub)
    if stats is None: continue
    out.append(fmt_row(stats, name, is_gart=(code == 'LGBM_V3'), bold_all=(code == 'LGBM_V3')))

out.append(r'\bottomrule')
out.append(r'\end{tabular}%')
out.append(r'}')
out.append(r'\end{table}')
print('\n'.join(out))
