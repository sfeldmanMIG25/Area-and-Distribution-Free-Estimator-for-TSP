"""
SHAP verification helper. Prints the top-K feature ranking that should match
Area_Free_Main.tex Table \\ref{tab:shap_top}, and writes PNG summary plots to
paper_tooling/ (not paper_reference/, so paper_reference stays Overleaf-clean).
"""
import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'tsp_features_v3.csv')
MODEL = os.path.join(ROOT, 'lgbm_model_v3', 'lgbm_alpha_model_v3.joblib')
GEN = os.path.join(ROOT, 'paper_tooling', 'shap_plots')
os.makedirs(GEN, exist_ok=True)

TOP_K = 10
SAMPLE_SIZE = 5000

DROP_COLS = ['instance_name', 'optimal_cost', 'alpha', 'split', 'grid_size']


def main():
    print("Loading model and test data...")
    model = joblib.load(MODEL)
    expected = list(model.feature_name_)

    df = pd.read_csv(DATA)
    test = df[df['split'] == 'test'].copy()
    print(f"Test rows: {len(test)}")

    X = test.drop(columns=[c for c in DROP_COLS if c in test.columns])
    X = X[expected]

    if len(X) > SAMPLE_SIZE:
        X = X.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"SHAP sample: {len(X)} rows on {len(expected)} features")

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    mean_abs = np.abs(sv).mean(axis=0)
    ranking = pd.DataFrame({
        'feature':      expected,
        'mean_abs_shap': mean_abs,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    total = ranking['mean_abs_shap'].sum()
    ranking['share_pct'] = 100 * ranking['mean_abs_shap'] / total
    ranking['cum_share_pct'] = ranking['share_pct'].cumsum()

    print("\nTop features by mean |SHAP|:")
    print(ranking.head(TOP_K).to_string(index=False))

    print("\nDiff this ranking against Area_Free_Main.tex Table \\ref{tab:shap_top}.\n")

    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X, plot_type='bar', max_display=TOP_K, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(GEN, 'shap_bar.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 8))
    shap.summary_plot(sv, X, max_display=TOP_K, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(GEN, 'shap_summary.png'), dpi=150)
    plt.close()
    print(f"Wrote {GEN}/shap_bar.png and shap_summary.png")


if __name__ == '__main__':
    main()
