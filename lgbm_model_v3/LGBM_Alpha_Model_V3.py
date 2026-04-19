import os
import json

# --- CRITICAL FIX FOR WINDOWS/JOBLIB ---
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, os.cpu_count()))
# ---------------------------------------

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, '..', 'tsp_features_v3.csv')

MODEL_DIR = ROOT_DIR
MODEL_OUTPUT_FILE = os.path.join(MODEL_DIR, 'lgbm_alpha_model_v3.joblib')
PARAMS_OUTPUT_FILE = os.path.join(MODEL_DIR, 'best_params_v3.json')
METRICS_OUTPUT_FILE = os.path.join(MODEL_DIR, 'test_metrics_v3.json')
PLOT_IMPORTANCE_FILE = os.path.join(MODEL_DIR, 'feature_importance_v3.png')
OPTUNA_DB_FILE = os.path.join(MODEL_DIR, 'optuna_study_v3.db')

RANDOM_STATE = 42
OPTUNA_N_TRIALS = 100
EARLY_STOPPING_ROUNDS = 100
MIN_BOOST_ROUND = 100
MAX_BOOST_ROUND = 3000


def load_and_preprocess(data_path):
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)

    mst_divisor = df['mst_total_length'].replace(0, 1e-9)
    df['alpha'] = (df['optimal_cost'] / mst_divisor).clip(1.0, 2.0)
    y = df['alpha']

    # Drop identifiers, grid_size (leak), and mst_total_length: since
    # alpha = optimal_cost / mst_total_length, feeding MST magnitude to the
    # model lets it recover a component of the label.
    drop_cols = ['instance_name', 'optimal_cost', 'alpha', 'split', 'grid_size',
                 'mst_total_length']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    mst_val = df.loc[val_mask, 'mst_total_length'].to_numpy()
    cost_val = df.loc[val_mask, 'optimal_cost'].to_numpy()
    mst_test = df.loc[test_mask, 'mst_total_length'].to_numpy()
    cost_test = df.loc[test_mask, 'optimal_cost'].to_numpy()

    print(f"  train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, mst_val, cost_val, mst_test, cost_test


def optuna_objective(trial, X_train, y_train, X_val, y_val, mst_val, cost_val):
    params = {
        'objective': 'regression_l2',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 64, 512),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }
    dtr = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        LightGBMPruningCallback(trial, metric='rmse', valid_name='val'),
    ]
    booster = lgb.train(
        params, dtr, num_boost_round=MAX_BOOST_ROUND,
        valid_sets=[dval], valid_names=['val'],
        callbacks=callbacks,
    )
    pred_alpha = np.clip(booster.predict(X_val, num_iteration=booster.best_iteration), 1.0, 2.0)
    pred_cost = pred_alpha * mst_val
    err = (pred_cost - cost_val) / np.where(cost_val == 0, 1e-9, cost_val)
    return float(np.std(err, ddof=1))


def _test_metrics(model, X_test, y_test, mst_test, cost_test):
    pred_alpha = np.clip(model.predict(X_test, num_iteration=model.best_iteration_), 1.0, 2.0)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_alpha)))
    mae = float(mean_absolute_error(y_test, pred_alpha))
    r2 = float(r2_score(y_test, pred_alpha))
    pred_cost = pred_alpha * mst_test
    err = (pred_cost - cost_test) / cost_test
    mape = float(np.mean(np.abs(err)) * 100.0)
    sdpe = float(np.std(err, ddof=1) * 100.0)
    return {'alpha_rmse': rmse, 'alpha_mae': mae, 'alpha_r2': r2,
            'cost_mape_pct': mape, 'cost_sdpe_pct': sdpe}


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, y_train, X_val, y_val, X_test, y_test, mst_val, cost_val, mst_test, cost_test = load_and_preprocess(DATA_FILE)

    print(f"\n--- Running Optuna ({OPTUNA_N_TRIALS} trials, TPE multivariate + Hyperband) ---")
    sampler = TPESampler(multivariate=True, group=True, seed=RANDOM_STATE)
    pruner = HyperbandPruner(min_resource=MIN_BOOST_ROUND, max_resource=MAX_BOOST_ROUND)
    study = optuna.create_study(
        direction='minimize', sampler=sampler, pruner=pruner,
        study_name='lgbm_v3', storage=f'sqlite:///{OPTUNA_DB_FILE}',
        load_if_exists=True,
    )
    study.optimize(
        lambda t: optuna_objective(t, X_train, y_train, X_val, y_val, mst_val, cost_val),
        n_trials=OPTUNA_N_TRIALS, show_progress_bar=True,
    )
    best_params = study.best_params
    print(f"Best val RMSE: {study.best_value:.6f}")
    print(f"Best params: {best_params}")
    with open(PARAMS_OUTPUT_FILE, 'w') as f:
        json.dump(best_params, f, indent=4)

    # --- Final fit: train-only, early-stopped on val, tested on test. No
    # refitting on train+val; val stays purely a checkpoint signal.
    print("\n--- Final fit on train, early-stopped on val ---")
    final_model = lgb.LGBMRegressor(
        **best_params,
        n_estimators=MAX_BOOST_ROUND,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], eval_metric='rmse',
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    print(f"Best iteration: {final_model.best_iteration_}")

    metrics = _test_metrics(final_model, X_test, y_test, mst_test, cost_test)
    print("\n--- Test metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" + ("%" if 'pct' in k else ""))
    with open(METRICS_OUTPUT_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(final_model, MODEL_OUTPUT_FILE)
    print(f"\nModel saved to {MODEL_OUTPUT_FILE}")

    try:
        lgb.plot_importance(final_model, max_num_features=30, figsize=(10, 15), importance_type='split')
        plt.title('LightGBM V3 Feature Importance (Top 30)')
        plt.tight_layout()
        plt.savefig(PLOT_IMPORTANCE_FILE)
        plt.close()
        print(f"Importance plot -> {PLOT_IMPORTANCE_FILE}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
