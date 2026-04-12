import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import re

os.chdir(r'D:\Area-and-Distribution-Free-Estimator-for-TSP')

# Load data
df_2d = pd.read_csv('Generalized_TSP_Analysis/benchmark_results_2D_v3.csv')
df_nd = pd.read_csv('Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv')

# Parse 2D instances
def parse_2d_instance(inst):
    match = re.match(r'TSP-boundary-n(\d+)-g(\d+)-(\d+)', inst)
    if match:
        n = int(match.group(1))
        g = int(match.group(2))
        return n, 2, g
    return None, None, None

df_2d[['n_customers', 'dimension', 'grid_size']] = df_2d['instance'].apply(lambda x: pd.Series(parse_2d_instance(x)))

# Combine datasets
df_2d['distribution'] = 'boundary'
df_nd['abs_gap_pct'] = df_nd['gap_pct'].abs()
df = pd.concat([df_2d, df_nd], ignore_index=True)

# Function to compute metrics
def compute_metrics(group):
    true = group['true_cost']
    pred = group['pred_cost']
    mape = mean_absolute_percentage_error(true, pred) * 100
    mean_gap = group['gap_pct'].mean()
    mean_abs_gap = group['abs_gap_pct'].mean()
    return pd.Series({'MAPE': mape, 'Mean_Gap': mean_gap, 'Mean_Abs_Gap': mean_abs_gap, 'Count': len(group)})

metrics = df.groupby(['model', 'dimension', 'n_customers']).apply(compute_metrics).reset_index()

def add_frontier_metrics(df):
    df_mst = df[df['model'] == 'MST_Ratio']
    df_models = df[df['model'] != 'MST_Ratio']
    # Merge MST baseline pred_cost
    merged = pd.merge(df_models, df_mst[['instance', 'pred_cost']], on='instance', suffixes=('', '_mst'))
    merged['baseline_mape'] = (abs(merged['true_cost'] - (merged['pred_cost_mst'] * 1.22)) / merged['true_cost']) * 100
    merged['gart_mape'] = (abs(merged['true_cost'] - merged['pred_cost']) / merged['true_cost']) * 100
    # Categorize
    bins = [0, 100, 1000, 100000]
    merged['n_group'] = pd.cut(merged['n_customers'], bins=bins, labels=['Small', 'Medium', 'Large'])
    return merged.groupby(['model', 'n_group'], observed=True)[['baseline_mape', 'gart_mape']].mean()

frontier_results = add_frontier_metrics(df)
print('Frontier Analysis Results:')
print(frontier_results)
