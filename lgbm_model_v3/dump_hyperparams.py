"""
Verification helper. Prints the hyperparameters and trained-model stats that
should match the values hard-coded into paper_reference/Area_Free_Main.tex.
Run this whenever best_params_v3.json or the saved booster changes, then
diff the printed values against the paper's Table \\ref{tab:hyperparams}.
No files are written; the paper_reference folder stays Overleaf-clean.
"""
import os
import json
import joblib
import numpy as np

ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(ROOT_DIR, 'lgbm_alpha_model_v3.joblib')
PARAMS_FILE = os.path.join(ROOT_DIR, 'best_params_v3.json')


def fmt(x):
    if isinstance(x, float):
        if abs(x) < 1e-3 or abs(x) >= 1e4:
            return f"{x:.3e}"
        return f"{x:.4g}"
    return str(x)


def booster_stats(booster):
    dump = booster.dump_model()
    trees = dump['tree_info']
    n_trees = len(trees)

    depths, leaf_counts = [], []
    for t in trees:
        leaves = []
        def walk(node, depth):
            if 'leaf_index' in node:
                leaves.append(depth)
            else:
                walk(node['left_child'],  depth + 1)
                walk(node['right_child'], depth + 1)
        walk(t['tree_structure'], 0)
        depths.append(max(leaves) if leaves else 0)
        leaf_counts.append(len(leaves))

    return {
        'n_trees':         n_trees,
        'max_depth_worst': int(max(depths)),
        'avg_depth':       float(np.mean(depths)),
        'avg_leaves':      float(np.mean(leaf_counts)),
        'max_leaves':      int(max(leaf_counts)),
    }


def main():
    with open(PARAMS_FILE) as f:
        params = json.load(f)

    model = joblib.load(MODEL_FILE)
    booster = model.booster_
    bs = booster_stats(booster)
    feature_names = list(model.feature_name_)

    print("=== GART 2.0 hyperparameters (Optuna TPE, 100 trials, RMSE) ===")
    print(f"  learning_rate    : {fmt(params['learning_rate'])}")
    print(f"  num_leaves       : {fmt(params['num_leaves'])}")
    print(f"  lambda_l1        : {fmt(params['lambda_l1'])}")
    print(f"  lambda_l2        : {fmt(params['lambda_l2'])}")
    print(f"  feature_fraction : {fmt(params['feature_fraction'])}")
    print(f"  bagging_fraction : {fmt(params['bagging_fraction'])}")
    print(f"  bagging_freq     : {fmt(params['bagging_freq'])}")
    print(f"  min_child_samples: {fmt(params['min_child_samples'])}")
    print("=== Trained-booster statistics ===")
    print(f"  trees (after early stop) : {bs['n_trees']}")
    print(f"  avg leaves per tree      : {bs['avg_leaves']:.2f}")
    print(f"  max leaves (any tree)    : {bs['max_leaves']}")
    print(f"  avg tree depth           : {bs['avg_depth']:.2f}")
    print(f"  max tree depth           : {bs['max_depth_worst']}")
    print(f"=== Feature set ({len(feature_names)}) ===")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    print("\nDiff these values against Area_Free_Main.tex Table \\ref{tab:hyperparams}.")


if __name__ == '__main__':
    main()
