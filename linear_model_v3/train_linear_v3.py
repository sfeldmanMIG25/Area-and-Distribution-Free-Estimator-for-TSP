import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, '..', 'tsp_features_v3.csv')
MODEL_FILE = os.path.join(ROOT_DIR, 'linear_alpha_model_v3.joblib')

RANDOM_STATE = 42


def _test_metrics(pipeline, X_test, y_test, mst_test, true_cost_test):
    """Compute cost-space MAPE and SDPE alongside alpha-space RMSE/R^2."""
    pred_alpha = pipeline.predict(X_test).clip(1.0, 2.0)
    rmse = np.sqrt(mean_squared_error(y_test, pred_alpha))
    r2 = r2_score(y_test, pred_alpha)
    pred_cost = pred_alpha * mst_test
    err = (pred_cost - true_cost_test) / true_cost_test
    mape = float(np.mean(np.abs(err)) * 100.0)
    sdpe = float(np.std(err, ddof=1) * 100.0)
    return rmse, r2, mape, sdpe


def train_linear_model():
    print("--- Training Linear Regression Baseline (Full V3 Set) ---")
    if not os.path.exists(DATA_FILE):
        print(f"CRITICAL: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)

    # 1. Target.
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = (df['optimal_cost'] / mst_divisor).clip(1.0, 2.0)
    y = df['alpha']

    # 2. Feature matrix. Drop identifiers and the two raw explosive columns —
    # their log-space siblings (log_bounding_hypervolume, log_node_density)
    # are already present in the CSV, so we prefer those for the linear model.
    drop_cols = [
        'instance_name', 'optimal_cost', 'alpha', 'split', 'mst_total_length',
        'grid_size',
        'bounding_hypervolume', 'node_density',
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan)

    # 3. Split — train only (val/test are held out for evaluation).
    train_mask = df['split'] == 'train'
    test_mask = df['split'] == 'test'

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    mst_test = df.loc[test_mask, 'mst_total_length'].to_numpy()
    true_cost_test = df.loc[test_mask, 'optimal_cost'].to_numpy()

    print(f"Training on {X_train.shape[1]} features with {len(X_train)} samples...")

    # 4. Pipeline.
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on test in both alpha-space and cost-space.
    rmse, r2, mape, sdpe = _test_metrics(pipeline, X_test, y_test, mst_test, true_cost_test)
    print("\n--- Test Results ---")
    print(f"  alpha RMSE: {rmse:.6f}")
    print(f"  alpha R^2 : {r2:.6f}")
    print(f"  cost  MAPE: {mape:.3f}%")
    print(f"  cost  SDPE: {sdpe:.3f}%")

    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train_linear_model()
