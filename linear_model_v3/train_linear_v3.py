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

def robust_log_transform(df, cols):
    """Safely log-transforms explosive columns to prevent overflow."""
    for col in cols:
        if col in df.columns:
            # Clean infinite/NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(1e-9)
            df[col] = df[col].clip(lower=1e-9)
            
            # Create log feature and drop original
            df[f'log_{col}'] = np.log(df[col])
            df.drop(columns=[col], inplace=True)
            print(f"  Applied Log Transform: {col} -> log_{col}")
    return df

def train_linear_model():
    print("--- Training Linear Regression Baseline (Full V3 Set) ---")
    if not os.path.exists(DATA_FILE):
        print(f"CRITICAL: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)

    # 1. Target Preparation
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = (df['optimal_cost'] / mst_divisor).clip(1.0, 2.0)
    y = df['alpha']

    # 2. Feature Cleaning (Drop Metadata)
    drop_cols = ['instance_name', 'optimal_cost', 'alpha', 'split', 'mst_total_length', 'grid_size']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Stability: Log-Transform Explosive Features
    # These explode in high dimensions (e.g. 100D), causing linear regression weights to fail
    explosive_cols = ['bounding_hypervolume', 'node_density']
    X = robust_log_transform(X, explosive_cols)

    # 4. Clean Residual Infs
    X = X.replace([np.inf, -np.inf], np.nan)

    # 5. Split Data
    train_mask = (df['split'] == 'train') | (df['split'] == 'val')
    test_mask = (df['split'] == 'test')
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"Training on {X_train.shape[1]} features with {len(X_train)} samples...")

    # 6. Pipeline Construction
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Handle any NaNs
        ('scaler', StandardScaler()),                  # Linear models require scaling
        ('model', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    # 7. Evaluation
    y_pred = pipeline.predict(X_test).clip(1.0, 2.0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Validation Results ---")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R^2 : {r2:.6f}")

    # 8. Save
    joblib.dump(pipeline, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_linear_model()