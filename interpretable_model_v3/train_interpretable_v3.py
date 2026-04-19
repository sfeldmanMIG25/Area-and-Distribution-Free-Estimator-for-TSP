import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.multitest import multipletests

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, '..', 'tsp_features_v3.csv')
LGBM_MODEL = os.path.join(ROOT_DIR, '..', 'lgbm_model_v3', 'lgbm_alpha_model_v3.joblib')

MODEL_DIR = os.path.join(ROOT_DIR, 'model_artifacts')
ROUTER_FILE = os.path.join(MODEL_DIR, 'router.joblib')
EXPERTS_FILE = os.path.join(MODEL_DIR, 'experts.joblib')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')

RANDOM_STATE = 42
BH_FDR = 0.01          # Benjamini-Hochberg false-discovery rate
ROUTER_DEPTH = 3
TOP_K_INTERACTIONS = 30

os.makedirs(MODEL_DIR, exist_ok=True)


# --- Feature-matrix helper ---------------------------------------------------

# Columns dropped from the trainer's X matrix. Raw bounding_hypervolume and
# node_density are dropped in favour of their pre-computed log siblings
# (log_bounding_hypervolume / log_node_density) which are already in the CSV —
# this keeps scale-sensitive ridge experts numerically stable without renaming
# any feature and lets LGBM-mined SHAP interactions align 1:1 by name.
DROP_COLS = [
    'instance_name', 'optimal_cost', 'alpha', 'split', 'mst_total_length',
    'grid_size',
    'bounding_hypervolume', 'node_density',
]


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)


# --- LGBM-guided interaction mining ------------------------------------------

def get_lgbm_interactions(X_sample, lgbm_path):
    """Use SHAP interaction values on the LGBM V3 booster to rank feature
    pairs. Feature names match 1:1 between the booster and X_sample because
    both use the raw (no-extra-log-transform) V3 column names."""
    print(f"Mining interactions from {lgbm_path} ...")
    wrapper = joblib.load(lgbm_path)
    booster = wrapper.booster_
    required_feats = booster.feature_name()

    # Align columns — LGBM trainer also dropped grid_size but kept raw
    # bounding_hypervolume / node_density. If any booster-feature is not in
    # X_sample (because we dropped it here), fill with 0: SHAP will still
    # rank remaining pairs correctly and we simply won't propose pairs that
    # include a dropped feature.
    X_aligned = X_sample.copy()
    for col in required_feats:
        if col not in X_aligned.columns:
            X_aligned[col] = 0.0
    X_aligned = X_aligned[required_feats]

    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_interaction_values(X_aligned.iloc[:1000])
    mean_int = np.abs(shap_vals).mean(0)
    np.fill_diagonal(mean_int, 0)

    # Only keep pairs where BOTH features still exist in the live X_sample
    # (filter out pairs involving dropped-then-zero-filled columns).
    live = set(X_sample.columns)
    pairs = []
    for i in range(len(required_feats)):
        for j in range(i + 1, len(required_feats)):
            a, b = required_feats[i], required_feats[j]
            if a not in live or b not in live:
                continue
            val = mean_int[i, j]
            if val > 1e-5:
                pairs.append({'feature_a': a, 'feature_b': b, 'strength': float(val)})
    pairs.sort(key=lambda x: x['strength'], reverse=True)
    top_pairs = pairs[:TOP_K_INTERACTIONS]

    print("Top 3 interactions:")
    for p in top_pairs[:3]:
        print(f"  {p['feature_a']} * {p['feature_b']} ({p['strength']:.4f})")
    return top_pairs


# --- Training loop -----------------------------------------------------------

def train_interpretable_model():
    print("--- Training Interpretable V3 Model ---")
    df = pd.read_csv(DATA_FILE)
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = (df['optimal_cost'] / mst_divisor).clip(1.0, 2.0)
    X_all = _feature_matrix(df)
    y_all = df['alpha']

    # Train only — val/test held out for evaluation.
    train_mask = df['split'] == 'train'
    test_mask = df['split'] == 'test'
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    mst_test = df.loc[test_mask, 'mst_total_length'].to_numpy()
    true_cost_test = df.loc[test_mask, 'optimal_cost'].to_numpy()

    # 1. Mine interactions using the existing LGBM V3 booster as a guide.
    if os.path.exists(LGBM_MODEL):
        interactions = get_lgbm_interactions(X_train, LGBM_MODEL)
    else:
        print("Warning: LGBM model not found. Skipping interaction mining.")
        interactions = []

    # 2. Router (shallow decision tree) on base features.
    print(f"\nTraining Regime Router (Depth {ROUTER_DEPTH}) ...")
    router = DecisionTreeRegressor(
        max_depth=ROUTER_DEPTH, min_samples_leaf=50, random_state=RANDOM_STATE,
    )
    router.fit(X_train, y_train)
    leaf_indices = router.apply(X_train)
    unique_leaves = np.unique(leaf_indices)
    print(f"Router identified {len(unique_leaves)} regimes.")

    # 3. Local ridge experts per leaf.
    experts, leaf_metadata = {}, {}
    for leaf_id in unique_leaves:
        mask = leaf_indices == leaf_id
        X_leaf = X_train[mask].copy()
        y_leaf = y_train[mask]
        print(f"\n  Leaf {leaf_id}: {len(X_leaf)} samples")

        # 3a. Engineer candidate interactions from LGBM-mined pairs.
        created_interactions = []
        for p in interactions:
            fa, fb = p['feature_a'], p['feature_b']
            if fa in X_leaf.columns and fb in X_leaf.columns:
                col_name = f"{fa}_x_{fb}"
                X_leaf[col_name] = X_leaf[fa] * X_leaf[fb]
                created_interactions.append(col_name)

        # 3b. Feature selection — F-regression is scale-invariant, so no
        # intermediate StandardScaler is needed. Multi-test corrected with
        # Benjamini-Hochberg at BH_FDR.
        var = X_leaf.var()
        active_cols = var[var > 1e-9].index.tolist()
        if not active_cols:
            print("    Skipping leaf (no variance).")
            continue
        _, p_values = f_regression(X_leaf[active_cols], y_leaf)
        reject, _, _, _ = multipletests(p_values, alpha=BH_FDR, method='fdr_bh')
        final_features = np.array(active_cols)[reject].tolist()
        if not final_features:
            print("    Skipping leaf (no feature survived BH).")
            continue

        surviving_interactions = [f for f in final_features if f in created_interactions]
        print(f"    Selected {len(final_features)} features "
              f"({len(surviving_interactions)} interactions) after BH@{BH_FDR}.")

        # 3c. Ridge expert.
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RidgeCV(alphas=[0.1, 1.0, 10.0])),
        ])
        pipe.fit(X_leaf[final_features], y_leaf)
        score = pipe.score(X_leaf[final_features], y_leaf)
        print(f"    Train R^2: {score:.4f}")

        experts[int(leaf_id)] = pipe
        needed_interactions = [c.split('_x_') for c in surviving_interactions]
        leaf_metadata[int(leaf_id)] = {
            'base_features': [f for f in final_features if f not in surviving_interactions],
            'interactions': needed_interactions,
            'description': f"R2={score:.2f}, N={len(X_leaf)}",
        }

    # 4. Save artifacts.
    print("\nSaving artifacts ...")
    joblib.dump(router, ROUTER_FILE)
    joblib.dump(experts, EXPERTS_FILE)
    with open(METADATA_FILE, 'w') as f:
        json.dump(leaf_metadata, f, indent=2)

    # 5. Test metrics — route test rows through the experts.
    print("\n--- Evaluating on held-out test set ---")
    test_leaves = router.apply(X_test)
    preds = np.full(len(X_test), np.nan, dtype=np.float64)
    for leaf_id, pipe in experts.items():
        rows = np.where(test_leaves == leaf_id)[0]
        if len(rows) == 0:
            continue
        meta = leaf_metadata[leaf_id]
        X_rows = X_test.iloc[rows].copy()
        for fa, fb in meta['interactions']:
            X_rows[f"{fa}_x_{fb}"] = X_rows[fa] * X_rows[fb]
        cols = meta['base_features'] + [f"{a}_x_{b}" for a, b in meta['interactions']]
        preds[rows] = pipe.predict(X_rows[cols])

    ok = ~np.isnan(preds)
    pred_alpha = np.clip(preds[ok], 1.0, 2.0)
    y_ok = y_test.to_numpy()[ok]
    rmse = float(np.sqrt(mean_squared_error(y_ok, pred_alpha)))
    r2 = float(r2_score(y_ok, pred_alpha))
    pred_cost = pred_alpha * mst_test[ok]
    err = (pred_cost - true_cost_test[ok]) / true_cost_test[ok]
    mape = float(np.mean(np.abs(err)) * 100.0)
    sdpe = float(np.std(err, ddof=1) * 100.0)
    print(f"  alpha RMSE: {rmse:.6f}")
    print(f"  alpha R^2 : {r2:.6f}")
    print(f"  cost  MAPE: {mape:.3f}%")
    print(f"  cost  SDPE: {sdpe:.3f}%")
    print(f"  Routed   : {ok.sum()}/{len(preds)} test rows")
    print(f"Interpretable Model V3 saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_interpretable_model()
