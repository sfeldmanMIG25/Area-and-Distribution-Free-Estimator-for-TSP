import os
import sys

# --- CRITICAL FIX FOR WINDOWS/JOBLIB ---
# Prevent thread contention in libraries like Joblib/Optuna
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, os.cpu_count()))
# ---------------------------------------

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import json

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# CORRECTED PATH: Look one folder up for the feature file
DATA_FILE = os.path.join(ROOT_DIR, '..', 'tsp_features_v3.csv') 

MODEL_DIR = os.path.join(ROOT_DIR) # Save in the script's own directory
MODEL_OUTPUT_FILE = os.path.join(MODEL_DIR, 'lgbm_alpha_model_v3.joblib')
PARAMS_OUTPUT_FILE = os.path.join(MODEL_DIR, 'best_params_v3.json')
PLOT_IMPORTANCE_FILE = os.path.join(MODEL_DIR, 'feature_importance_v3.png')

RANDOM_STATE = 42
OPTUNA_N_TRIALS = 100 # Kept as requested
EARLY_STOPPING_ROUNDS = 100

def load_and_preprocess(data_path):
    '''
    Loads V3 data, creates alpha target, and splits into train/val/test.
    Robustly handles the V3 feature set structure.
    '''
    print(f"Loading and preprocessing data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"CRITICAL ERROR: Data file not found at {data_path}")
        print("Please run 'feature_creator_v3.py' first to generate the dataset.")
        return None
        
    df = pd.read_csv(data_path)

    # Validate required columns exist before proceeding
    required_cols = ['mst_total_length', 'optimal_cost', 'split']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Dataset missing required columns: {required_cols}")
        return None

    # Calculate Alpha (Target)
    # Protection against div by zero
    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = df['optimal_cost'] / mst_divisor
    
    # Clip alpha to reasonable bounds (TSP specific theory: 1.0 <= alpha <= 2.0 approx)
    df['alpha'] = df['alpha'].clip(1.0, 2.0)

    y = df['alpha']
    
    # Drop Metadata and Components of Target
    features_to_drop = [
        'instance_name', 'optimal_cost', 'alpha', 'split', 'grid_size'
        'mst_total_length'
    ]
    
    existing_cols_to_drop = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)

    # --- Feature Configuration ---
    # We treat 'dimension' and 'n_customers' as numerical to allow extrapolation
    # The V3 feature set is designed to be dimension-agnostic, so we do NOT enforce categories.
    categorical_features = [] 
            
    # --- Splitting ---
    train_mask = (df['split'] == 'train')
    val_mask = (df['split'] == 'val')
    test_mask = (df['split'] == 'test')
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Data Split Summary:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val  : {len(X_val)} samples")
    print(f"  Test : {len(X_test)} samples")
    
    if len(X_train) == 0:
        print("Error: Training set is empty.")
        return None

    # Full training set for final model
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_full, y_train_full, categorical_features


def optuna_objective(trial, X_train, y_train, X_val, y_val, categorical_features):
    '''
    The objective function for Optuna to minimize (RMSE).
    Hyperparameter ranges kept exactly as requested.
    '''
    params = {
        'objective': 'regression_l2',
        'metric': 'rmse',
        'n_estimators': 3000, # High number, controlled by early_stopping
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        
        # --- Parameters to Tune (As requested) ---
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 64, 512), 
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }

    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        categorical_feature=categorical_features
    )
    
    y_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    y_pred_clipped = y_pred.clip(1.0, 2.0)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_clipped))
    return rmse


if __name__ == '__main__':
    # Create the output directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("--- 1. Loading and Preprocessing V3 Data ---")
    preprocess_result = load_and_preprocess(DATA_FILE)
    
    if preprocess_result is None:
        print("Exiting due to data loading error.")
    else:
        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            X_train_full, y_train_full,
            categorical_features
        ) = preprocess_result
        
        print(f"\n--- 2. Running Optuna Hyperparameter Tuning ({OPTUNA_N_TRIALS} trials) ---")
        # Optuna verbosity
        optuna.logging.set_verbosity(optuna.logging.INFO)
        
        objective_func = lambda trial: optuna_objective(
            trial, X_train, y_train, X_val, y_val, categorical_features
        )
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective_func, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
        
        print(f"\nOptuna tuning complete. Best RMSE: {study.best_value:.6f}")
        print("Best parameters found:")
        print(study.best_params)

        print(f"\n--- 3. Saving Best Parameters ---")
        with open(PARAMS_OUTPUT_FILE, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"Parameters saved to {PARAMS_OUTPUT_FILE}")

        print("\n--- 4. Training Final Model ---")
        
        # 1. Get the best hyperparameters from the study
        best_params = study.best_params
        
        # 2. Find the optimal n_estimators using these params on the split data
        print("Finding optimal number of trees using early stopping...")
        temp_model = lgb.LGBMRegressor(
            **best_params,
            n_estimators=5000, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        temp_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            categorical_feature=categorical_features
        )
        
        n_best_trees = temp_model.best_iteration_
        if n_best_trees is None or n_best_trees == 0:
            print("Warning: Early stopping did not find an optimal iteration. Defaulting to 100.")
            n_best_trees = 100
            
        print(f"Optimal number of trees found: {n_best_trees}")
        
        # 3. Train the *actual* final model on ALL training data (Train + Val)
        print("Training final model on combined (train + val) dataset...")
        final_params = best_params.copy()
        final_params['n_estimators'] = n_best_trees
        
        final_model = lgb.LGBMRegressor(**final_params, random_state=RANDOM_STATE, n_jobs=-1)
        
        final_model.fit(
            X_train_full,
            y_train_full,
            categorical_feature=categorical_features
        )
        
        print("Final model trained.")
        
        print("\n--- 5. Evaluating Final Model on Test Set ---")
        y_pred = final_model.predict(X_test)
        y_pred_clipped = y_pred.clip(1.0, 2.0)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
        test_mae = mean_absolute_error(y_test, y_pred_clipped)
        test_r2 = r2_score(y_test, y_pred_clipped)

        print("\n--- Final LightGBM V3 Model Test Results ---")
        print(f"  Final Test RMSE: {test_rmse:.6f}")
        print(f"  Final Test MAE : {test_mae:.6f}")
        print(f"  Final Test R^2 : {test_r2:.6f}")

        print(f"\n--- 6. Saving Model and Plots ---")
        print(f"Saving model to {MODEL_OUTPUT_FILE}...")
        joblib.dump(final_model, MODEL_OUTPUT_FILE)
        
        print(f"Saving feature importance plot to {PLOT_IMPORTANCE_FILE}...")
        try:
            # Increase max_num_features to see the new proxies
            lgb.plot_importance(final_model, max_num_features=30, figsize=(10, 15), importance_type='split')
            plt.title('LightGBM V3 Feature Importance (Top 30)')
            plt.tight_layout()
            plt.savefig(PLOT_IMPORTANCE_FILE)
            plt.close()
            print("Plot saved successfully.")
        except Exception as e:
            print(f"Could not generate plot: {e}")
        
        print("\n✅ Process complete.")