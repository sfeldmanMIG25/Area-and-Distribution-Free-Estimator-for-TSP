"""Neural control on GART 2.0's identical 31-column feature vector.

Replaces ``train_v3.py`` as the same-feature control. The architecture
(``TSP_Leap_Model``: linear stem, gated residual blocks, sigmoid head), the
loss (Huber, delta=0.1), the optimiser (AdamW, weight_decay 1e-3), the
schedule (cosine over 250 epochs), the alpha-weighted resampling (p ~ 1 + 5y),
the gradient clip, the AMP setup and the best-validation-SDPE checkpoint rule
are all carried over unchanged. Three things differ:

  1. the input is GART 2.0's 31 columns in booster order, not V3's 30;
  2. the training table is ``tsp_features_v4.csv``, the table GART 2.0 is
     fitted on, not ``tsp_features_v3.csv``;
  3. the torch RNG is seeded (see SEEDING below).

HYPERPARAMETERS: FROZEN, NOT RETUNED
------------------------------------
``hidden_dim``, ``num_blocks``, ``dropout`` and ``lr`` are V3's Optuna winner,
selected against the 30-input problem, and they are reused here verbatim.

The reason is symmetry with the model this is a control for. GART 2.0's own
hyperparameters are V3's, frozen -- its sidecar records "V3's shipped values,
frozen. Not tuned", after a 200-trial study bought 0.011 pp on the
multidimensional split and cost 0.16 pp on TSPLIB. Both arms of the comparison
therefore carry hyperparameters chosen on the 30-feature problem and refitted
on 31 features, and the only thing that varies between them is the model class,
which is what a control is for. Giving the network a fresh search budget that
the production model was denied would convert a model-class comparison into a
tuning-budget comparison, and it would do so in the direction that flatters the
conclusion the manuscript currently draws.

The transfer is also cheap to justify on the merits. ``input_dim`` enters the
architecture in exactly one place, ``self.stem = nn.Linear(input_dim,
hidden_dim)``; every downstream tensor is ``hidden_dim``-shaped. Going from 30
to 31 inputs widens the stem's fan-in by 3.3% and changes nothing else, so the
tuned quantities are not meaningfully functions of it.

``--tune`` re-runs V3's Optuna study (same sampler, pruner, ranges, trial and
epoch budgets) on the 31-input problem, into a separate study database. It is a
falsification check on the paragraph above, not the shipped configuration.

SEEDING
-------
``train_v3.py`` never seeded torch, so the shipped 30-feature network is not
reproducible and has no measured run-to-run spread. That is a problem here
specifically: this control decides a headline claim whose current margin on the
2D benchmark is 0.08 pp of SDPE, and the measured spread of this configuration
over seeds is 0.67 pp -- nine times the margin. ``--seeds`` fits the same
configuration over several seeds and records the whole band, so no reported
number is a single draw.

Seeding torch does not make the fit bit-reproducible: two runs at seed 42 gave
test SDPE 2.1627 and 2.1645. The non-determinism is cuDNN kernel selection and
atomics, below the level a seed controls. The seed band is therefore a lower
bound on run-to-run spread, not an exhaustive account of it.

WHICH SEED SHIPS
----------------
The seed with the best VALIDATION SDPE, which is the rule already used to pick
the epoch within a fit. Extending the same rule one level up is consistent, and
it touches nothing but ``split == 'val'``. Shipping "the first seed listed"
would have shipped seed 42, whose validation SDPE is mid-band but whose test
SDPE is second-worst of seven -- an arbitrary draw presented as a result.

Usage:
    python nn_est_alpha_v3/train_nn_31f.py --seeds 42,1,2,3,4,5,6
    python nn_est_alpha_v3/train_nn_31f.py --reship   # re-select from saved seeds
    python nn_est_alpha_v3/train_nn_31f.py --tune
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO))

from control_features_31f import (  # noqa: E402
    ALPHA_CLIP, DATA_FILE, RANDOM_STATE, Stable31Scaler, cost_metrics,
    load_splits, production_features,
)

warnings.filterwarnings("ignore")

SCALER_OUT = THIS / "nn_31f_scaler.joblib"
MODEL_OUT = THIS / "nn_31f_model.pt"
SIDECAR_OUT = THIS / "nn_31f_model.json"
SEED_DIR = THIS / "nn_31f_seeds"
TUNE_DB = THIS / "optuna_31f.db"
TUNE_OUT = THIS / "nn_31f_tuned_params.json"

# V3's Optuna winner, read off nn_alpha_v3_model.pt. Frozen -- see the module
# docstring for why this is not retuned.
V3_TUNED = {
    "hidden_dim": 192,
    "num_blocks": 4,
    "dropout": 0.05871254182522992,
    "lr": 0.0009842315738502602,
}

EPOCHS = 250
BATCH_SIZE = 4096
OPTUNA_TRIALS = 100
OPTUNA_EPOCHS = 50
OPTUNA_MIN_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()


# --- Architecture: verbatim from train_v3.py --------------------------------
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
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
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


# --- Data -------------------------------------------------------------------
def prepare(features):
    """Scale on train only, hand back GPU-resident tensors plus cost-space refs."""
    parts = load_splits(features)
    scaler = Stable31Scaler()
    X = {k: scaler.fit_transform(parts[k]["X"]) if k == "train"
         else scaler.transform(parts[k]["X"]) for k in ("train", "val", "test")}
    tens = {k: torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for k, v in X.items()}
    # y = alpha - 1 in [0, 1]: the sigmoid head's native range.
    y_tr = torch.as_tensor(parts["train"]["y"] - 1.0,
                           dtype=torch.float32, device=DEVICE).view(-1, 1)
    return parts, scaler, tens, y_tr


def predict_alpha(model, X_gpu, batch=8192):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_gpu.size(0), batch):
            outs.append(model(X_gpu[i:i + batch]).squeeze(-1))
    return np.clip(torch.cat(outs).cpu().numpy().astype(np.float64) + 1.0, *ALPHA_CLIP)


def val_sdpe(model, X_gpu, part):
    return cost_metrics(predict_alpha(model, X_gpu), part["mst"], part["cost"])["sdpe"]


# --- Training ---------------------------------------------------------------
def fit(hp, tens, y_tr, parts, seed, epochs=EPOCHS, trial=None):
    """One fit. Identical to train_v3.final_fit apart from the explicit seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = TSP_Leap_Model(tens["train"].shape[1], hp["hidden_dim"],
                           hp["num_blocks"], hp["dropout"]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=0.1)
    amp = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    weights = (1.0 + 5.0 * y_tr.squeeze(-1)).clamp_min(1e-6)
    n = tens["train"].size(0)

    best_v, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        model.train()
        idx_all = torch.multinomial(weights, n, replacement=True)
        for i in range(0, n, BATCH_SIZE):
            idx = idx_all[i:i + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = criterion(model(tens["train"][idx]), y_tr[idx])
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp.step(optimizer)
            amp.update()
        scheduler.step()

        v = val_sdpe(model, tens["val"], parts["val"])
        if v < best_v:
            best_v, best_state, best_epoch = v, copy.deepcopy(model.state_dict()), epoch
        if trial is not None:
            import optuna
            trial.report(v, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_v, best_epoch


def score_all(model, tens, parts):
    return {k: cost_metrics(predict_alpha(model, tens[k]), parts[k]["mst"], parts[k]["cost"])
            for k in ("train", "val", "test")}


# --- Optuna falsification arm ------------------------------------------------
def tune(tens, y_tr, parts):
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler

    def objective(trial):
        hp = {
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 512, step=64),
            "num_blocks": trial.suggest_int("num_blocks", 4, 8),
            "dropout": trial.suggest_float("dropout", 0.05, 0.2),
            "lr": trial.suggest_float("lr", 1e-5, 2e-3, log=True),
        }
        _, best_v, _ = fit(hp, tens, y_tr, parts, RANDOM_STATE,
                           epochs=OPTUNA_EPOCHS, trial=trial)
        return best_v

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, group=True, seed=RANDOM_STATE),
        pruner=HyperbandPruner(min_resource=OPTUNA_MIN_EPOCHS, max_resource=OPTUNA_EPOCHS),
        study_name="nn_31f", storage=f"sqlite:///{TUNE_DB.as_posix()}", load_if_exists=True)
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    return study


def ship_seed(band: dict) -> int:
    """The seed rule: best validation SDPE. Never looks at ``split == 'test'``."""
    return min(band, key=lambda s: band[s]["val"]["sdpe"])


def reship() -> None:
    """Re-apply :func:`ship_seed` to the checkpoints already on disk."""
    sidecar = json.loads(SIDECAR_OUT.read_text(encoding="utf-8"))
    band = {int(k): v for k, v in sidecar["seed_band"].items()}
    seed = ship_seed(band)
    torch.save(torch.load(SEED_DIR / f"nn_31f_seed{seed}.pt", map_location="cpu",
                          weights_only=False), MODEL_OUT)
    sidecar["shipped_seed"] = seed
    sidecar["shipped_seed_rule"] = "best validation SDPE among the fitted seeds"
    SIDECAR_OUT.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"[nn_31f] shipped seed {seed} (val sdpe {band[seed]['val']['sdpe']:.4f}, "
          f"test sdpe {band[seed]['test']['sdpe']:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42",
                    help="comma-separated torch seeds; the shipped artifact is "
                         "chosen from them by best validation SDPE, not by order")
    ap.add_argument("--tune", action="store_true",
                    help="run the Optuna falsification arm on the 31-input problem")
    ap.add_argument("--reship", action="store_true",
                    help="re-apply the best-validation-SDPE seed rule to saved "
                         "checkpoints, without refitting")
    args = ap.parse_args()

    if args.reship:
        reship()
        return

    features = production_features()
    parts, scaler, tens, y_tr = prepare(features)
    print(f"[nn_31f] {len(features)} features, table {DATA_FILE.name}, device {DEVICE}, "
          f"train={len(parts['train']['y'])} val={len(parts['val']['y'])} "
          f"test={len(parts['test']['y'])}")

    if args.tune:
        study = tune(tens, y_tr, parts)
        done = [t for t in study.trials if t.value is not None]
        print(f"[nn_31f] tune: {len(done)} complete / {len(study.trials)} trials")
        print(f"[nn_31f] tune: best val SDPE {study.best_value:.6f} at {study.best_params}")
        frozen_v = fit(V3_TUNED, tens, y_tr, parts, RANDOM_STATE, epochs=OPTUNA_EPOCHS)[1]
        print(f"[nn_31f] tune: frozen V3 config under the same {OPTUNA_EPOCHS}-epoch "
              f"budget scores {frozen_v:.6f}")
        TUNE_OUT.write_text(json.dumps({
            "role": "falsification check on freezing V3's hyperparameters; NOT shipped",
            "search_space": "identical to nn_est_alpha_v3/train_v3.py",
            "n_trials_requested": OPTUNA_TRIALS,
            "n_trials_complete": len(done),
            "epochs_per_trial": OPTUNA_EPOCHS,
            "best_params": study.best_params,
            "best_val_sdpe": study.best_value,
            "frozen_v3_params": V3_TUNED,
            "frozen_v3_val_sdpe_same_budget": frozen_v,
        }, indent=2), encoding="utf-8")
        print(f"[nn_31f] wrote {TUNE_OUT.name}")
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    SEED_DIR.mkdir(exist_ok=True)
    band = {}
    for seed in seeds:
        model, best_v, best_epoch = fit(V3_TUNED, tens, y_tr, parts, seed)
        m = score_all(model, tens, parts)
        band[seed] = {"val": m["val"], "test": m["test"],
                      "best_val_sdpe": best_v, "best_epoch": best_epoch}
        print(f"[nn_31f] seed {seed:>3}: val sdpe={m['val']['sdpe']:.4f} "
              f"mape={m['val']['mape']:.4f} | test sdpe={m['test']['sdpe']:.4f} "
              f"mape={m['test']['mape']:.4f} (best epoch {best_epoch})")
        torch.save({"state_dict": model.state_dict(), "input_dim": len(features),
                    "features": features, "params": V3_TUNED, "seed": seed},
                   SEED_DIR / f"nn_31f_seed{seed}.pt")
    joblib.dump(scaler, SCALER_OUT)

    ship = ship_seed(band)
    torch.save(torch.load(SEED_DIR / f"nn_31f_seed{ship}.pt", map_location="cpu",
                          weights_only=False), MODEL_OUT)
    print(f"[nn_31f] shipping seed {ship} (best validation SDPE "
          f"{band[ship]['val']['sdpe']:.4f})")

    t_sdpe = [band[s]["test"]["sdpe"] for s in seeds]
    t_mape = [band[s]["test"]["mape"] for s in seeds]
    print(f"[nn_31f] test SDPE over {len(seeds)} seeds: median {np.median(t_sdpe):.4f} "
          f"band [{min(t_sdpe):.4f}, {max(t_sdpe):.4f}]")
    print(f"[nn_31f] test MAPE over {len(seeds)} seeds: median {np.median(t_mape):.4f} "
          f"band [{min(t_mape):.4f}, {max(t_mape):.4f}]")

    SIDECAR_OUT.write_text(json.dumps({
        "name": "Neural net (GART 2.0 features)",
        "model_key": "NN_31F",
        "artifact": MODEL_OUT.name,
        "scaler": SCALER_OUT.name,
        "supersedes": "nn_alpha_v3_model.pt",
        "role": "same-feature control -- isolates the model class against GART 2.0",
        "n_features": len(features),
        "features_in_model_order": features,
        "feature_source": "lgbm_model_v3/gart2_final.joblib::feature_name()",
        "training_table": DATA_FILE.name,
        "target": "y = alpha - 1 in [0, 1]; sigmoid head, so alpha = 1 + y is "
                  "structurally bounded and the clip is a guard, not a correction",
        "alpha_clip": list(ALPHA_CLIP),
        "architecture": "TSP_Leap_Model, verbatim from nn_est_alpha_v3/train_v3.py",
        "hyperparameters": V3_TUNED,
        "hyperparameter_provenance":
            "V3's Optuna winner on the 30-input problem, frozen. Deliberately not "
            "retuned: GART 2.0 likewise carries V3's frozen hyperparameters, so "
            "both arms of the comparison were tuned on the same 30-feature problem "
            "and only the model class differs. See nn_31f_tuned_params.json for the "
            "falsification arm.",
        "training_protocol":
            "fit on split=='train' only; val used solely to pick the best-SDPE "
            "checkpoint and never trained on; no refit on train+val",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "shipped_seed": ship,
        "shipped_seed_rule": "best validation SDPE among the fitted seeds",
        "seed_band": band,
        "seed_band_note":
            "train_v3.py did not seed torch, so the 30-feature predecessor has no "
            "measured run-to-run spread. The band is reported because the claim "
            "this control informs turns on a margin smaller than it. Seeding does "
            "not make the fit bit-reproducible on CUDA -- two runs at seed 42 gave "
            "test SDPE 2.1627 and 2.1645 -- so the band understates the spread.",
        "test_sdpe_median": float(np.median(t_sdpe)),
        "test_sdpe_band": [float(min(t_sdpe)), float(max(t_sdpe))],
        "test_mape_median": float(np.median(t_mape)),
        "test_mape_band": [float(min(t_mape)), float(max(t_mape))],
    }, indent=2), encoding="utf-8")
    print(f"[nn_31f] saved {MODEL_OUT.name}, {SCALER_OUT.name}, {SIDECAR_OUT.name}")


if __name__ == "__main__":
    main()
