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
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, '..', 'tsp_features_v3.csv')
LGBM_MODEL = "C:\\Area-and-Distribution-Free-Estimators-for-TSP\\lgbm_model_v3\\lgbm_alpha_model_v3.joblib"

MODEL_DIR = os.path.join(ROOT_DIR, 'model_artifacts')
ROUTER_FILE = os.path.join(MODEL_DIR, 'router.joblib')
EXPERTS_FILE = os.path.join(MODEL_DIR, 'experts.joblib')
METADATA_FILE = os.path.join(MODEL_DIR, 'model_metadata.json')

RANDOM_STATE = 42
SIGNIFICANCE_ALPHA = 0.01  # Strict p-value for interactions
ROUTER_DEPTH = 3           # Depth of regime splits
TOP_K_INTERACTIONS = 30    # Candidates to harvest from LGBM

os.makedirs(MODEL_DIR, exist_ok=True)

# --- UTILS ---

def robust_log_transform(df, cols):
    """Matches V3 Logic."""
    for col in cols:
        if col in df.columns:
            # Replace inf/-inf
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Impute for log safety
            df[col] = df[col].fillna(1e-9).clip(lower=1e-9)
            # Log transform
            df[f'log_{col}'] = np.log(df[col])
            df.drop(columns=[col], inplace=True)
    return df

def get_lgbm_interactions(X_sample, lgbm_path):
    """Uses SHAP to find the most potent non-linear interactions."""
    print(f"Mining interactions from {lgbm_path}...")
    
    # Load wrapper and extract booster
    wrapper = joblib.load(lgbm_path)
    booster = wrapper.booster_
    
    # Ensure columns match
    required_feats = booster.feature_name()
    X_aligned = X_sample.copy()
    for col in required_feats:
        if col not in X_aligned.columns:
            X_aligned[col] = 0.0
    X_aligned = X_aligned[required_feats]
    
    # Compute SHAP Interactions
    explainer = shap.TreeExplainer(booster)
    # limit sample size for speed
    shap_vals = explainer.shap_interaction_values(X_aligned.iloc[:1000])
    
    # Get mean absolute interaction strength (off-diagonal)
    mean_int = np.abs(shap_vals).mean(0)
    np.fill_diagonal(mean_int, 0)
    
    pairs = []
    for i in range(len(required_feats)):
        for j in range(i + 1, len(required_feats)):
            val = mean_int[i, j]
            if val > 1e-5:
                pairs.append({
                    'feature_a': required_feats[i], 
                    'feature_b': required_feats[j], 
                    'strength': val
                })
                
    pairs.sort(key=lambda x: x['strength'], reverse=True)
    top_pairs = pairs[:TOP_K_INTERACTIONS]
    
    print(f"Top 3 Identified Interactions:")
    for p in top_pairs[:3]:
        print(f"  {p['feature_a']} * {p['feature_b']} ({p['strength']:.4f})")
        
    return top_pairs

def train_interpretable_model():
    print("--- Training Interpretable V3 Model ---")
    
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = (df['optimal_cost'] / mst_divisor).clip(1.0, 2.0)
    
    # Drop Metadata
    drop_cols = ['instance_name', 'optimal_cost', 'alpha', 'split', 'mst_total_length', 'grid_size']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['alpha']
    
    # 2. Log Transform Base Features (V3 Standard)
    X = robust_log_transform(X, ['bounding_hypervolume', 'node_density'])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    train_mask = (df['split'] == 'train') | (df['split'] == 'val')
    X_train, y_train = X[train_mask], y[train_mask]
    
    # 3. Mine Interactions (Using LGBM as the guide)
    if os.path.exists(LGBM_MODEL):
        interactions = get_lgbm_interactions(X_train, LGBM_MODEL)
    else:
        print("Warning: LGBM model not found. Skipping interaction mining.")
        interactions = []

    # 4. Train Router (Decision Tree)
    print(f"\nTraining Regime Router (Depth {ROUTER_DEPTH})...")
    # Router uses ONLY base features to split
    router = DecisionTreeRegressor(max_depth=ROUTER_DEPTH, min_samples_leaf=50, random_state=RANDOM_STATE)
    router.fit(X_train, y_train)
    
    # Get leaf indices
    leaf_indices = router.apply(X_train)
    unique_leaves = np.unique(leaf_indices)
    print(f"Router identified {len(unique_leaves)} distinct regimes (Leaves).")
    
    # 5. Train Local Experts
    experts = {}
    leaf_metadata = {}
    
    base_features = list(X_train.columns)
    
    for leaf_id in unique_leaves:
        mask = (leaf_indices == leaf_id)
        X_leaf = X_train[mask].copy()
        y_leaf = y_train[mask]
        
        print(f"\n  Leaf {leaf_id}: {len(X_leaf)} samples")
        
        # A. Engineer Candidate Interactions
        # We act locally: an interaction might be significant here but not elsewhere
        created_interactions = []
        for p in interactions:
            fa, fb = p['feature_a'], p['feature_b']
            # Handle Log prefix mismatch if LGBM used raw names
            # Map LGBM names to current X names if needed
            if fa not in X_leaf.columns and f"log_{fa}" in X_leaf.columns: fa = f"log_{fa}"
            if fb not in X_leaf.columns and f"log_{fb}" in X_leaf.columns: fb = f"log_{fb}"
            
            if fa in X_leaf.columns and fb in X_leaf.columns:
                col_name = f"{fa}_x_{fb}"
                X_leaf[col_name] = X_leaf[fa] * X_leaf[fb]
                created_interactions.append(col_name)

        # B. Feature Selection (P-Value Filter)
        # We check ALL features (Base + New Interactions) for significance in this leaf
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_leaf)
        
        # Remove constant columns
        var = X_leaf.var()
        active_cols = var[var > 1e-9].index.tolist()
        
        if not active_cols:
            print("    Skipping leaf (no variance).")
            continue
            
        X_active = X_leaf[active_cols]
        X_scaled_active = scaler.fit_transform(X_active)
        
        f_scores, p_values = f_regression(X_scaled_active, y_leaf)
        
        significant_mask = p_values < SIGNIFICANCE_ALPHA
        final_features = np.array(active_cols)[significant_mask].tolist()
        
        # Identify which interactions survived
        surviving_interactions = [f for f in final_features if f in created_interactions]
        print(f"    Selected {len(final_features)} features ({len(surviving_interactions)} interactions).")
        
        # C. Train Linear Model (RidgeCV)
        # Using RidgeCV to handle multicollinearity better than Lasso for inference stability
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RidgeCV(alphas=[0.1, 1.0, 10.0]))
        ])
        
        pipe.fit(X_leaf[final_features], y_leaf)
        score = pipe.score(X_leaf[final_features], y_leaf)
        print(f"    Train R^2: {score:.4f}")
        
        # D. Store Artifacts
        experts[int(leaf_id)] = pipe
        
        # Parse interactions for metadata storage (to allow dynamic compute)
        needed_interactions = []
        for inter_col in surviving_interactions:
            parts = inter_col.split('_x_')
            needed_interactions.append(parts) # [feature_a, feature_b]

        leaf_metadata[int(leaf_id)] = {
            'base_features': [f for f in final_features if f not in surviving_interactions],
            'interactions': needed_interactions,
            'description': f"R2={score:.2f}, N={len(X_leaf)}"
        }

    # 6. Save Everything
    print("\nSaving artifacts...")
    joblib.dump(router, ROUTER_FILE)
    joblib.dump(experts, EXPERTS_FILE)
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(leaf_metadata, f, indent=2)
        
    print(f"✅ Interpretable Model V3 Saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_interpretable_model()