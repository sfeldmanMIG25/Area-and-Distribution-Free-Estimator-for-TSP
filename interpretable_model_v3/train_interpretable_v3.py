import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.multitest import multipletests

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, '..', 'tsp_features_v3.csv')
LGBM_MODEL = os.path.join(ROOT_DIR, '..', 'lgbm_model_v3', 'lgbm_alpha_model_v3.joblib')

MODEL_DIR = os.path.join(ROOT_DIR, 'model_artifacts')
IMPUTER_FILE = os.path.join(MODEL_DIR, 'imputer.joblib')
BASE_MODEL_FILE = os.path.join(MODEL_DIR, 'base_ridge.joblib')
ROUTER_FILE = os.path.join(MODEL_DIR, 'router.joblib')
EXPERTS_FILE = os.path.join(MODEL_DIR, 'experts.joblib')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')

RANDOM_STATE = 42
BH_FDR = 0.01
ROUTER_DEPTH = 3
TOP_K_INTERACTIONS = 30
# Router sees only interpretable regime features so routing decisions are
# transparent AND we can build a per-leaf (n, d) support box for OOD detection.
ROUTER_FEATURES = ['dimension', 'n_customers', 'log_node_density']

os.makedirs(MODEL_DIR, exist_ok=True)

DROP_COLS = [
    'instance_name', 'optimal_cost', 'alpha', 'split', 'mst_total_length',
    'grid_size',
    'bounding_hypervolume', 'node_density',
]


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X.replace([np.inf, -np.inf], np.nan)


def _signed_log1p(a: np.ndarray) -> np.ndarray:
    # Bounded, sign-preserving transform so interaction products don't explode
    # when either factor has a heavy tail (e.g. mst_diameter at d=100).
    return np.sign(a) * np.log1p(np.abs(a))


def _add_log_interactions(X: pd.DataFrame, pairs):
    X = X.copy()
    created = []
    for p in pairs:
        fa, fb = p['feature_a'], p['feature_b']
        if fa in X.columns and fb in X.columns:
            name = f"{fa}_x_{fb}"
            X[name] = _signed_log1p(X[fa].to_numpy()) * _signed_log1p(X[fb].to_numpy())
            created.append(name)
    return X, created


def get_lgbm_interactions(X_sample, lgbm_path):
    print(f"Mining interactions from {lgbm_path} ...")
    wrapper = joblib.load(lgbm_path)
    booster = wrapper.booster_
    required_feats = booster.feature_name()

    X_aligned = X_sample.copy()
    for col in required_feats:
        if col not in X_aligned.columns:
            X_aligned[col] = 0.0
    X_aligned = X_aligned[required_feats]

    explainer = shap.TreeExplainer(booster)
    shap_vals = explainer.shap_interaction_values(X_aligned.iloc[:200])
    mean_int = np.abs(shap_vals).mean(0)
    np.fill_diagonal(mean_int, 0)

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


def train_interpretable_model():
    print("--- Training Interpretable V3 Model (stacked, OOD-safe) ---")
    df = pd.read_csv(DATA_FILE)
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = (df['optimal_cost'] / mst_divisor).clip(1.0, 2.0)
    X_raw = _feature_matrix(df)
    y_all = df['alpha']

    train_mask = df['split'] == 'train'
    test_mask = df['split'] == 'test'

    # (3) Global median imputation — fit on train only.
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_raw[train_mask])
    X_all = pd.DataFrame(
        imputer.transform(X_raw), columns=X_raw.columns, index=X_raw.index,
    )

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    mst_test = df.loc[test_mask, 'mst_total_length'].to_numpy()
    true_cost_test = df.loc[test_mask, 'optimal_cost'].to_numpy()

    # (4) Global ridge baseline — the hard-to-beat linear benchmark. Leaf
    # experts will predict residuals on top of this, so the worst-case expert
    # damage is bounded by how much the base model already explains.
    print("\nTraining global ridge base model ...")
    base_model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
    ])
    base_model.fit(X_train, y_train)
    base_train = base_model.predict(X_train)
    base_test = base_model.predict(X_test)
    print(f"  base train R^2: {base_model.score(X_train, y_train):.4f}")

    r_train = y_train.to_numpy() - base_train

    # LGBM-mined interaction pairs (used as log1p*log1p products).
    if os.path.exists(LGBM_MODEL):
        interactions = get_lgbm_interactions(X_train, LGBM_MODEL)
    else:
        print("Warning: LGBM model not found. Skipping interaction mining.")
        interactions = []

    # (1) Router on explicit regime features, predicts residual.
    router_cols = [c for c in ROUTER_FEATURES if c in X_train.columns]
    print(f"\nTraining router on {router_cols} (depth {ROUTER_DEPTH}) ...")
    router = DecisionTreeRegressor(
        max_depth=ROUTER_DEPTH, min_samples_leaf=200, random_state=RANDOM_STATE,
    )
    router.fit(X_train[router_cols], r_train)
    leaf_indices = router.apply(X_train[router_cols])
    unique_leaves = np.unique(leaf_indices)
    print(f"Router identified {len(unique_leaves)} regimes.")

    experts, leaf_metadata = {}, {}
    for leaf_id in unique_leaves:
        mask = leaf_indices == leaf_id
        X_leaf = X_train[mask]
        r_leaf = r_train[mask]
        print(f"\n  Leaf {leaf_id}: {len(X_leaf)} samples")

        X_leaf_int, created = _add_log_interactions(X_leaf, interactions)

        var = X_leaf_int.var()
        active_cols = var[var > 1e-9].index.tolist()
        if not active_cols:
            print("    Skipping leaf (no variance).")
            continue
        _, p_values = f_regression(X_leaf_int[active_cols], r_leaf)
        p_values = np.nan_to_num(p_values, nan=1.0)
        reject, _, _, _ = multipletests(p_values, alpha=BH_FDR, method='fdr_bh')
        final_features = np.array(active_cols)[reject].tolist()
        if not final_features:
            print("    No feature survived BH — expert predicts zero residual.")
            continue

        surviving_interactions = [f for f in final_features if f in created]
        print(f"    Selected {len(final_features)} features "
              f"({len(surviving_interactions)} interactions) after BH@{BH_FDR}.")

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        ])
        pipe.fit(X_leaf_int[final_features], r_leaf)
        score = pipe.score(X_leaf_int[final_features], r_leaf)
        print(f"    Train residual R^2: {score:.4f}")

        # (1) Per-leaf training support box on router features — a test row
        # whose (n, d, density) falls outside this box is OOD for this expert
        # and we fall back to base-only prediction (residual = 0).
        support = {
            c: [float(X_leaf[c].min()), float(X_leaf[c].max())]
            for c in router_cols
        }

        experts[int(leaf_id)] = pipe
        leaf_metadata[int(leaf_id)] = {
            'base_features': [f for f in final_features if f not in surviving_interactions],
            'interactions': [c.split('_x_') for c in surviving_interactions],
            'support_box': support,
            'description': f"residual_R2={score:.2f}, N={len(X_leaf)}",
        }

    print("\nSaving artifacts ...")
    joblib.dump(imputer, IMPUTER_FILE)
    joblib.dump(base_model, BASE_MODEL_FILE)
    joblib.dump(router, ROUTER_FILE)
    joblib.dump(experts, EXPERTS_FILE)
    with open(METADATA_FILE, 'w') as f:
        json.dump(
            {'router_features': router_cols, 'leaves': leaf_metadata},
            f, indent=2,
        )

    print("\n--- Evaluating on held-out test set ---")
    test_leaves = router.apply(X_test[router_cols])
    residual = np.zeros(len(X_test), dtype=np.float64)
    ood_count = 0

    for leaf_id, pipe in experts.items():
        rows = np.where(test_leaves == leaf_id)[0]
        if len(rows) == 0:
            continue
        meta = leaf_metadata[leaf_id]
        in_support = np.ones(len(rows), dtype=bool)
        for c, (lo, hi) in meta['support_box'].items():
            vals = X_test[c].iloc[rows].to_numpy()
            in_support &= (vals >= lo) & (vals <= hi)
        use_rows = rows[in_support]
        ood_count += int((~in_support).sum())
        if len(use_rows) == 0:
            continue
        X_use, _ = _add_log_interactions(X_test.iloc[use_rows].copy(), interactions)
        cols = meta['base_features'] + [f"{a}_x_{b}" for a, b in meta['interactions']]
        residual[use_rows] = pipe.predict(X_use[cols])

    pred_alpha = np.clip(base_test + residual, 1.0, 2.0)
    y_true = y_test.to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, pred_alpha)))
    r2 = float(r2_score(y_true, pred_alpha))
    pred_cost = pred_alpha * mst_test
    err = (pred_cost - true_cost_test) / true_cost_test
    mape = float(np.mean(np.abs(err)) * 100.0)
    sdpe = float(np.std(err, ddof=1) * 100.0)

    base_only = np.clip(base_test, 1.0, 2.0)
    base_rmse = float(np.sqrt(mean_squared_error(y_true, base_only)))
    base_r2 = float(r2_score(y_true, base_only))
    base_err = (base_only * mst_test - true_cost_test) / true_cost_test
    base_mape = float(np.mean(np.abs(base_err)) * 100.0)
    base_sdpe = float(np.std(base_err, ddof=1) * 100.0)

    print(f"  alpha RMSE: {rmse:.6f}  (base-only: {base_rmse:.6f})")
    print(f"  alpha R^2 : {r2:.6f}   (base-only: {base_r2:.6f})")
    print(f"  cost  MAPE: {mape:.3f}%   (base-only: {base_mape:.3f}%)")
    print(f"  cost  SDPE: {sdpe:.3f}%   (base-only: {base_sdpe:.3f}%)")
    print(f"  OOD rows fell back to base: {ood_count}/{len(X_test)}")
    print(f"Interpretable Model V3 saved to {MODEL_DIR}")


if __name__ == "__main__":
    train_interpretable_model()
