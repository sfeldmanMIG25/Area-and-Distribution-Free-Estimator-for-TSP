import os
import sys
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import optuna
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, '..', 'tsp_features_v3.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'nn_alpha_v3_model.pt')
SCALER_PATH = os.path.join(ROOT_DIR, 'nn_alpha_v3_scaler.joblib')

RANDOM_STATE = 42
OPTUNA_TRIALS = 100
EPOCHS = 250
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA STABILITY ENGINE ---

class StableV3Scaler:
    """Combines Quantile mapping with strict normalization and clipping."""
    def __init__(self):
        self.qt = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.ss = StandardScaler()

    def fit(self, X):
        # 1. Map to Normal distribution (neutralizes skew)
        X_qt = self.qt.fit_transform(X)
        # 2. Ensure unit variance (tames outliers)
        self.ss.fit(X_qt)
        return self

    def transform(self, X):
        X_qt = self.qt.transform(X)
        X_ss = self.ss.transform(X_qt)
        # 3. Final safety clip to prevent extreme inputs during inference
        return np.clip(X_ss, -10.0, 10.0)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# --- ARCHITECTURE: PRE-NORM GATED RESIDUAL NETWORK ---

class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # Pre-normalization for extreme stability
        self.norm = nn.LayerNorm(dim)
        
        # 1. Gating Step
        self.glu = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GLU(dim=-1)
        )
        # 2. Stacked Processing Step
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout * 0.5)
        )
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.glu(x)
        x = self.ffn(x)
        return x + residual

class TSP_Leap_Model(nn.Module):
    """Generational leap: Residual architecture with dynamic GLU feature selection."""
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout=0.1):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # Final Norm before head
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() 
        )
        
        # Proper Initialization to prevent early inf
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head(x)

# --- DATASET ---

class TSPDataset(Dataset):
    def __init__(self, X, y):
        # Convert Series/Array to Torch strictly
        y_vals = y.values if hasattr(y, 'values') else y
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_vals, dtype=torch.float32).view(-1, 1)
        
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

    def get_sampler_weights(self):
        y_np = self.y.numpy().flatten()
        # High difficulty instances (alpha > 1.5) get higher weights
        weights = 1.0 + np.abs(y_np - 0.2) * 5.0
        return torch.tensor(weights, dtype=torch.float32)

def load_v3_clean():
    if not os.path.exists(DATA_FILE):
        sys.exit(f"CRITICAL: {DATA_FILE} not found.")
    
    df = pd.read_csv(DATA_FILE)
    
    # 1. Target Construction & Cleaning
    mst_len = df['mst_total_length'].replace(0, 1e-9)
    df['alpha_raw'] = (df['optimal_cost'] / mst_len).clip(1.0, 2.0)
    y = (df['alpha_raw'] - 1.0)
    
    # Remove NaNs from target
    valid_mask = ~y.isna()
    df = df[valid_mask]
    y = y[valid_mask]
    
    # 2. Feature Extraction
    drop_cols = ['instance_name', 'optimal_cost', 'mst_total_length', 'alpha_raw', 'split','grid_size']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'
    
    # 3. Strict Normalization
    scaler = StableV3Scaler()
    X_tr = scaler.fit_transform(X_raw[train_mask])
    X_vl = scaler.transform(X_raw[val_mask])
    X_te = scaler.transform(X_raw[test_mask])
    
    joblib.dump(scaler, SCALER_PATH)
    return X_tr, y[train_mask], X_vl, y[val_mask], X_te, y[test_mask], X_raw.columns.tolist()

# --- TRAINING LOOP ---

def objective(trial, X_tr, y_tr, X_vl, y_vl):
    h_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    n_blocks = trial.suggest_int("num_blocks", 4, 8)
    dropout = trial.suggest_float("dropout", 0.05, 0.2)
    lr = trial.suggest_float("lr", 1e-5, 2e-3, log=True)
    
    model = TSP_Leap_Model(X_tr.shape[1], h_dim, n_blocks, dropout).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.1) # Stable loss
    
    tr_loader = DataLoader(TSPDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(TSPDataset(X_vl, y_vl), batch_size=BATCH_SIZE * 2)
    
    best_v = float('inf')
    for epoch in range(15):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in vl_loader:
                v_loss += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        v_loss /= len(vl_loader)
        
        trial.report(v_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        best_v = min(best_v, v_loss)
    return best_v

if __name__ == "__main__":
    X_tr, y_tr, X_vl, y_vl, X_te, y_te, feats = load_v3_clean()
    
    print(f"Starting Accuracy-Focused Search on {DEVICE}...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: objective(t, X_tr, y_tr, X_vl, y_vl), n_trials=OPTUNA_TRIALS)
    
    bp = study.best_params
    print(f"Optimal Configuration: {bp}")
    
    # Final training
    model = TSP_Leap_Model(X_tr.shape[1], bp['hidden_dim'], bp['num_blocks'], bp['dropout']).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=bp['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=0.1)
    
    ds_full = TSPDataset(np.vstack([X_tr, X_vl]), pd.concat([y_tr, y_vl]))
    sampler = WeightedRandomSampler(ds_full.get_sampler_weights(), len(ds_full), replacement=True)
    loader = DataLoader(ds_full, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)
    
    for epoch in tqdm(range(EPOCHS), desc="Final Accuracy Run"):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': X_tr.shape[1],
        'features': feats,
        'params': bp
    }, MODEL_PATH)
    print(f"✅ V3 Pre-Norm Model saved to {MODEL_PATH}")