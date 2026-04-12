import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
import re

# Load data
df_2d = pd.read_csv('Generalized_TSP_Analysis/benchmark_results_2D_v3.csv')
df_nd = pd.read_csv('Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv')

# Parse 2D instances
def parse_2d_instance(inst):
    match = re.match(r'TSP-boundary-n(\d+)-g(\d+)-(\d+)', inst)
    if match:
        n = int(match.group(1))
        g = int(match.group(2))
        return n, 2, g  # d=2
    return None, None, None

df_2d[['n_customers', 'dimension', 'grid_size']] = df_2d['instance'].apply(lambda x: pd.Series(parse_2d_instance(x)))

# For ND, already has n_customers, dimension, grid_size

# Combine datasets
df_2d['distribution'] = 'boundary'  # assuming all boundary
df_nd['abs_gap_pct'] = df_nd['gap_pct'].abs()
df = pd.concat([df_2d, df_nd], ignore_index=True)

# Models of interest
models = df['model'].unique()
lgbm_models = [m for m in models if 'LGBM' in m]
base_models = ['MST_Ratio', 'Linear_V3', 'BHH']

# Function to compute metrics
def compute_metrics(group):
    true = group['true_cost']
    pred = group['pred_cost']
    mape = mean_absolute_percentage_error(true, pred) * 100
    mean_gap = group['gap_pct'].mean()
    mean_abs_gap = group['abs_gap_pct'].mean()
    return pd.Series({'MAPE': mape, 'Mean_Gap': mean_gap, 'Mean_Abs_Gap': mean_abs_gap, 'Count': len(group)})

# Group by model, dimension, n_customers
metrics = df.groupby(['model', 'dimension', 'n_customers']).apply(compute_metrics).reset_index()

# For LGBM_V3
lgbm_v3 = metrics[metrics['model'] == 'LGBM_V3']

# Compare to baselines
baselines = metrics[metrics['model'].isin(base_models + ['LGBM_V3'])]

# Plot MAPE vs n for different d
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
dimensions = [2, 3, 4, 5]
for i, d in enumerate(dimensions[:3]):
    data = baselines[baselines['dimension'] == d]
    sns.lineplot(data=data, x='n_customers', y='Mean_Abs_Gap', hue='model', ax=axes[i])
    axes[i].set_title(f'Dimension {d}')
    axes[i].set_xlabel('Number of Customers (n)')
    axes[i].set_ylabel('Mean Absolute Percentage Error (%)')
    axes[i].set_yscale('log')

plt.tight_layout()
plt.savefig('model_comparison.png')
# plt.show()

# Where LGBM works best
best_regions = lgbm_v3.groupby('dimension')['Mean_Abs_Gap'].min()
print("Best MAPE for LGBM_V3 by dimension:")
print(best_regions)

# Compare to optimal solve time
df['time_ratio'] = df['prediction_time_s'] / df['optimal_solve_time_s']
time_comparison = df.groupby(['model', 'dimension', 'n_customers'])['time_ratio'].mean().reset_index()

# Plot time speedup
fig, ax = plt.subplots(figsize=(10, 6))
for model in ['LGBM_V3', 'MST_Ratio', 'Linear_V3']:
    data = time_comparison[time_comparison['model'] == model]
    for d in [2,3,4]:
        subset = data[data['dimension'] == d]
        ax.plot(subset['n_customers'], subset['time_ratio'], label=f'{model} d={d}', marker='o')

ax.set_xlabel('n_customers')
ax.set_ylabel('Prediction Time / Optimal Solve Time')
ax.set_title('Time Ratio Comparison')
ax.legend()
plt.savefig('time_comparison.png')
# plt.show()

# Compare LGBM_V3 to baselines
comparison = baselines.pivot_table(values='Mean_Abs_Gap', index=['dimension', 'n_customers'], columns='model').reset_index()
comparison['LGBM_vs_MST'] = comparison['LGBM_V3'] / comparison['MST_Ratio']
comparison['LGBM_vs_Linear'] = comparison['LGBM_V3'] / comparison['Linear_V3']
if 'BHH' in comparison.columns:
    comparison['LGBM_vs_BHH'] = comparison['LGBM_V3'] / comparison['BHH']
elif 'BHH_Asymptotic' in comparison.columns:
    comparison['LGBM_vs_BHH'] = comparison['LGBM_V3'] / comparison['BHH_Asymptotic']

print("Comparison ratios (LGBM_V3 / baseline):")
print(comparison[['dimension', 'n_customers', 'LGBM_vs_MST', 'LGBM_vs_Linear', 'LGBM_vs_BHH']].describe())

# Best regions for LGBM
print("\nBest n ranges for LGBM_V3 by dimension:")
for d in dimensions:
    data = lgbm_v3[lgbm_v3['dimension'] == d]
    if not data.empty:
        min_error = data['Mean_Abs_Gap'].min()
        best_n = data.loc[data['Mean_Abs_Gap'] == min_error, 'n_customers'].values
        print(f"Dimension {d}: Best MAPE {min_error:.3f} at n={best_n}")

# When to use LGBM: when optimal solve time > some threshold
threshold = 10  # seconds
df['use_estimator'] = df['optimal_solve_time_s'] > threshold
estimator_usage = df[df['use_estimator']].groupby(['model', 'dimension'])['abs_gap_pct'].mean()
print("\nMAPE when optimal solve > 10s:")
print(estimator_usage)