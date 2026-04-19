import os
import sys
import copy
import joblib
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, QuantileTransformer

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, '..', 'tsp_features_v3.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'nn_alpha_v3_model.pt')
SCALER_PATH = os.path.join(ROOT_DIR, 'nn_alpha_v3_scaler.joblib')
METRICS_PATH = os.path.join(ROOT_DIR, 'test_metrics_v3.json')
OPTUNA_DB = os.path.join(ROOT_DIR, 'optuna_study_v3.db')

RANDOM_STATE = 42
OPTUNA_TRIALS = 100
OPTUNA_EPOCHS = 50          # inner trial budget (up from 15)
OPTUNA_MIN_EPOCHS = 5       # Hyperband min resource
EPOCHS = 250
BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


# --- Data-stability pipe -----------------------------------------------------
class StableV3Scaler:
    def __init__(self):
        self.qt = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.ss = StandardScaler()

    def fit(self, X):
        X_qt = self.qt.fit_transform(X)
        self.ss.fit(X_qt)
        return self

    def transform(self, X):
        return np.clip(self.ss.transform(self.qt.transform(X)), -10.0, 10.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --- Architecture -------------------------------------------------------------
class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.glu = nn.Sequential(nn.Linear(dim, dim * 2), nn.GLU(dim=-1))
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x):
        r = x
        x = self.norm(x)
        return self.ffn(self.glu(x)) + r


class TSP_Leap_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout=0.1):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.final_norm(x))


class TSPDataset(Dataset):
    def __init__(self, X, y):
        y_vals = y.values if hasattr(y, 'values') else y
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_vals, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def get_sampler_weights(self):
        y_np = self.y.numpy().flatten()
        # y = alpha - 1 in [0, 1]. Hard instances (high alpha) weighted more.
        # A sample at y=1 is sampled ~6x as often as a sample at y=0.
        return torch.tensor(1.0 + y_np * 5.0, dtype=torch.float32)


# --- Data loading -------------------------------------------------------------
def load_v3_clean():
    if not os.path.exists(DATA_FILE):
        sys.exit(f"CRITICAL: {DATA_FILE} not found.")
    df = pd.read_csv(DATA_FILE)

    mst_len = df['mst_total_length'].replace(0, 1e-9)
    df['alpha_raw'] = (df['optimal_cost'] / mst_len).clip(1.0, 2.0)
    y = df['alpha_raw'] - 1.0
    valid = ~y.isna()
    df, y = df[valid], y[valid]

    drop_cols = ['instance_name', 'optimal_cost', 'mst_total_length', 'alpha_raw', 'split', 'grid_size']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'

    scaler = StableV3Scaler()
    X_tr = scaler.fit_transform(X_raw[train_mask])
    X_vl = scaler.transform(X_raw[val_mask])
    X_te = scaler.transform(X_raw[test_mask])
    joblib.dump(scaler, SCALER_PATH)

    mst_val = df.loc[val_mask, 'mst_total_length'].to_numpy()
    cost_val = df.loc[val_mask, 'optimal_cost'].to_numpy()
    mst_test = df.loc[test_mask, 'mst_total_length'].to_numpy()
    cost_test = df.loc[test_mask, 'optimal_cost'].to_numpy()
    return (X_tr, y[train_mask], X_vl, y[val_mask], X_te, y[test_mask],
            mst_val, cost_val, mst_test, cost_test, X_raw.columns.tolist())


# --- GPU-resident helpers -----------------------------------------------------
def to_gpu_tensor(arr, dtype=torch.float32):
    a = arr.values if hasattr(arr, 'values') else arr
    return torch.as_tensor(a, dtype=dtype, device=DEVICE)


def eval_sdpe_gpu(model, X_gpu, mst_np, cost_np, batch_size=8192):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_gpu.size(0), batch_size):
            outs.append(model(X_gpu[i:i+batch_size]).squeeze(-1))
    preds = torch.cat(outs).cpu().numpy()
    pred_alpha = np.clip(preds + 1.0, 1.0, 2.0)
    pred_cost = pred_alpha * mst_np
    err = (pred_cost - cost_np) / np.where(cost_np == 0, 1e-9, cost_np)
    return float(np.std(err, ddof=1))


# --- Optuna objective: 50 epochs + Hyperband pruning --------------------------
def objective(trial, X_tr, y_tr, X_vl, y_vl, mst_vl, cost_vl):
    h_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    n_blocks = trial.suggest_int("num_blocks", 4, 8)
    dropout = trial.suggest_float("dropout", 0.05, 0.2)
    lr = trial.suggest_float("lr", 1e-5, 2e-3, log=True)

    model = TSP_Leap_Model(X_tr.shape[1], h_dim, n_blocks, dropout).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.1)

    tr_loader = DataLoader(TSPDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(TSPDataset(X_vl, y_vl), batch_size=BATCH_SIZE * 2)

    best_v = float('inf')
    for epoch in range(OPTUNA_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        v_sdpe = eval_val_sdpe(model, vl_loader, mst_vl, cost_vl)
        trial.report(v_sdpe, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_v = min(best_v, v_sdpe)
    return best_v


# --- Final fit: train only, keep best-val checkpoint --------------------------
def final_fit(bp, X_tr, y_tr, X_vl, y_vl, mst_vl, cost_vl):
    model = TSP_Leap_Model(X_tr.shape[1], bp['hidden_dim'], bp['num_blocks'], bp['dropout']).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=bp['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=0.1)

    ds_train = TSPDataset(X_tr, y_tr)
    sampler = WeightedRandomSampler(ds_train.get_sampler_weights(), len(ds_train), replacement=True)
    loader = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)
    vl_loader = DataLoader(TSPDataset(X_vl, y_vl), batch_size=BATCH_SIZE * 2)

    best_v = float('inf')
    best_state = None
    best_epoch = -1
    for epoch in tqdm(range(EPOCHS), desc="Final"):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        v_sdpe = eval_val_sdpe(model, vl_loader, mst_vl, cost_vl)
        if v_sdpe < best_v:
            best_v = v_sdpe
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val SDPE = {best_v:.6f} @ epoch {best_epoch}")
    return model


# --- Test metrics (cost-space MAPE + SDPE) ------------------------------------
def eval_test(model, X_te, y_te, mst_te, cost_te):
    model.eval()
    loader = DataLoader(TSPDataset(X_te, y_te), batch_size=BATCH_SIZE * 2)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy().ravel())
    preds = np.concatenate(preds)
    pred_alpha = np.clip(preds + 1.0, 1.0, 2.0)
    pred_cost = pred_alpha * mst_te
    err = (pred_cost - cost_te) / cost_te
    return {
        'cost_mape_pct': float(np.mean(np.abs(err)) * 100.0),
        'cost_sdpe_pct': float(np.std(err, ddof=1) * 100.0),
    }


if __name__ == "__main__":
    X_tr, y_tr, X_vl, y_vl, X_te, y_te, mst_vl, cost_vl, mst_te, cost_te, feats = load_v3_clean()
    print(f"Device: {DEVICE}  features: {len(feats)}  train: {len(y_tr)}")

    sampler = TPESampler(multivariate=True, group=True, seed=RANDOM_STATE)
    pruner = HyperbandPruner(min_resource=OPTUNA_MIN_EPOCHS, max_resource=OPTUNA_EPOCHS)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner,
        study_name='nn_v3', storage=f'sqlite:///{OPTUNA_DB}', load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, X_tr, y_tr, X_vl, y_vl, mst_vl, cost_vl), n_trials=OPTUNA_TRIALS)
    bp = study.best_params
    print(f"Best trial val loss = {study.best_value:.6f}")
    print(f"Best params: {bp}")

    model = final_fit(bp, X_tr, y_tr, X_vl, y_vl, mst_vl, cost_vl)

    metrics = eval_test(model, X_te, y_te, mst_te, cost_te)
    print("\n--- Test metrics ---")
    print(f"  cost MAPE: {metrics['cost_mape_pct']:.3f}%")
    print(f"  cost SDPE: {metrics['cost_sdpe_pct']:.3f}%")
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': X_tr.shape[1],
        'features': feats,
        'params': bp,
    }, MODEL_PATH)
    print(f"V3 NN model saved to {MODEL_PATH}")
