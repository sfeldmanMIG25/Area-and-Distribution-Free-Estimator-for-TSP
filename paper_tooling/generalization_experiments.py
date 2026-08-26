"""Generalization vs. interpolation experiments for GART 2.0.

Four refits, all sharing the shipped model's recipe -- training table, feature
list, target transform, monotone constraints, frozen hyperparameters, early
stopping rule and seed -- so the training data is the only thing that varies:

    E0  baseline refit on the shipped split (reproducibility calibration)
    E1  clean labels: drop the 184 desynchronised instances everywhere
    E2  leave-dimensions-out: no d in {15, 25} in train/val
    E3  leave-large-n-out: train/val restricted to n_customers <= 200

Nothing about the recipe is restated here.  It is read from
``model_registry.PRODUCTION_SIDECAR`` (the frozen provenance record) and from
the parameter block of ``model_registry.PRODUCTION_BOOSTER`` itself, and the
two are cross-checked against each other.  This module previously hardcoded
``tsp_features_v3.csv``, ``best_params_v3.json`` and
``lgbm_alpha_model_v3.joblib`` -- all three the *predecessor* -- so every number
in Section~\\ref{subsec:generalization} described a model the paper does not
ship.  Reading the recipe instead of restating it is what makes that class of
error impossible rather than merely fixed.

E0 is the load-bearing run.  It must reproduce the shipped booster
*bit-for-bit*: identical tree count, identical serialised model text, and
``max |delta pred| == 0`` over every row of the corpus.  If it does not, the
script stops, because every E1/E2/E3 delta is then measured against the wrong
baseline.  Two ways a plausible-but-wrong refit could pass a weaker check are
gated explicitly:

* the target is ``logit(alpha - 1)``, not ``alpha``, so a path that forgets the
  inverse map produces finite, non-absurd-looking predictions.  ``--show-transform``
  (on by default) scores the shipped booster with the inverse omitted and
  asserts the result is grossly wrong, so the bit-for-bit check is not vacuous.
* the booster carries monotone constraints on ``n_customers`` and ``dimension``.
  Dropping them still fits and still scores well.  The constraint vector is
  taken from the sidecar and cross-checked against the booster's own parameter
  block.

Usage::

    python paper_tooling/generalization_experiments.py
    python paper_tooling/generalization_experiments.py --experiments E2 E3 --force

Outputs ``paper_tooling/generalization_results.csv``, merges the same numbers
into ``paper_tooling/tables/paper_numbers.json`` via
``paper_tooling/generalization_bank.py``, and caches each refit in
``paper_tooling/generalization_models/``.
"""

from __future__ import annotations

import os

# Must precede the joblib import on Windows (matches LGBM_Alpha_Model_V3.py).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generalization_bank import merge_into_bank  # noqa: E402
from model_registry import (  # noqa: E402
    PRODUCTION_BOOSTER,
    PRODUCTION_SIDECAR,
    REPO_ROOT,
)

CORRUPT_FILE = REPO_ROOT / "paper_tooling" / "corrupt_instances.txt"
MODEL_DIR = REPO_ROOT / "paper_tooling" / "generalization_models"
RESULTS_FILE = REPO_ROOT / "paper_tooling" / "generalization_results.csv"

HELD_OUT_DIMS = (15, 25)
CONTROL_DIMS = (10, 20, 30)
SMALL_N_MAX = 200
LARGE_N_MAX = 1000

#: The logit transform, spelled exactly as the sidecar records it.  These are
#: assertions, not configuration: if the shipped model's transform ever changes
#: the strings stop matching and this script refuses to run rather than
#: silently applying the old inverse to a new booster.
EXPECTED_FORWARD = "z = log(u / (1 - u)) with u = clip(alpha - 1, 1e-6, 1 - 1e-6)"
EXPECTED_INVERSE = "alpha = 1 + 1 / (1 + exp(-z))"
LOGIT_EPS = 1e-6

#: Scoring the booster with the inverse map omitted must be *grossly* wrong,
#: not subtly wrong.  Anything under this and the bit-for-bit check would be
#: passing for reasons unrelated to the transform.
NO_INVERSE_MIN_MAPE_PCT = 5.0

#: One letter per axis in ``N{n}_D{d}_G{grid}_{letters}_{seq}``; 16 letter codes
#: collapse onto 4 distinct 1-D generators (see data_pipeline/instance_io.py).
FAMILY_BY_LETTER: dict[str, str] = {
    **{c: "uniform" for c in "rptglsbxeah"},
    **{c: "normal" for c in "ni"},
    **{c: "clustered" for c in "co"},
    **{c: "correlated" for c in "k"},
}
NON_UNIFORM_FAMILIES = ("normal", "clustered", "correlated")
NAME_PATTERN = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)$")

ALL_EXPERIMENTS = ("E0", "E1", "E2", "E3")


# --------------------------------------------------------------------------
# Recipe: read, never restate
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Recipe:
    """Everything needed to reproduce the shipped fit, read from the artifact."""

    data_file: Path
    features: tuple[str, ...]
    hyperparameters: dict[str, float | int]
    monotone: tuple[int, ...]
    monotone_method: str
    seed: int
    num_threads: int
    max_boost_round: int
    early_stopping_rounds: int
    sha256: str
    booster_text_sha256: str
    n_trees: int


def parse_booster_params(text: str) -> dict[str, str]:
    """Parse the ``parameters:`` block LightGBM serialises into every model."""
    start = text.find("\nparameters:\n")
    end = text.find("\nend of parameters", start)
    if start < 0 or end < 0:
        raise ValueError("shipped booster carries no parameter block")
    out: dict[str, str] = {}
    for line in text[start:end].splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]") and ":" in line:
            key, _, value = line[1:-1].partition(":")
            out[key.strip()] = value.strip()
    return out


def load_recipe() -> tuple[lgb.Booster, Recipe]:
    """Load the shipped booster and derive the fit recipe from it.

    The sidecar and the booster's own parameter block are independent records
    of the same fit.  Both are read and cross-checked; a disagreement means one
    of them has drifted and the refit would be against an unknown target.
    """
    digest = hashlib.sha256(PRODUCTION_BOOSTER.read_bytes()).hexdigest()
    sidecar = json.loads(PRODUCTION_SIDECAR.read_text(encoding="utf-8"))
    if sidecar["sha256"] != digest:
        raise SystemExit(
            f"{PRODUCTION_BOOSTER.name} hashes to {digest[:16]}... but "
            f"{PRODUCTION_SIDECAR.name} records {sidecar['sha256'][:16]}...; "
            "the shipped artifact and its provenance record disagree")

    transform = sidecar["target_transform"]
    if transform["forward"] != EXPECTED_FORWARD or transform["inverse"] != EXPECTED_INVERSE:
        raise SystemExit(
            "the shipped model's target transform is not the logit map this "
            f"script implements.\n  sidecar forward: {transform['forward']}\n"
            f"  sidecar inverse: {transform['inverse']}")
    if transform.get("clip_after_inverse", False):
        raise SystemExit("sidecar now asks for a post-inverse clip; update cost_metrics")

    booster = joblib.load(PRODUCTION_BOOSTER)
    text = booster.model_to_string()
    params = parse_booster_params(text)

    features = tuple(sidecar["features_in_booster_order"])
    if tuple(booster.feature_name()) != features:
        raise SystemExit("sidecar feature order does not match the booster's own")

    mono = tuple(int(v) for v in sidecar["monotone_constraints"])
    mono_booster = tuple(int(v) for v in params.get("monotone_constraints", "").split(",") if v != "")
    if mono_booster and mono_booster != mono:
        raise SystemExit("sidecar monotone constraints disagree with the booster's")
    if len(mono) != len(features):
        raise SystemExit("monotone constraint vector length != feature count")

    seed = int(params["seed"])
    if seed != int(sidecar["seed"]):
        raise SystemExit("sidecar seed disagrees with the booster's")

    hp = dict(sidecar["hyperparameters"])
    # The parameter block rounds to 6 significant figures; the sidecar carries
    # full precision.  Use the sidecar, verify against the block.
    for sidecar_key, block_key in (("learning_rate", "learning_rate"),
                                   ("num_leaves", "num_leaves"),
                                   ("reg_alpha", "lambda_l1"),
                                   ("reg_lambda", "lambda_l2"),
                                   ("feature_fraction", "feature_fraction"),
                                   ("bagging_fraction", "bagging_fraction"),
                                   ("bagging_freq", "bagging_freq"),
                                   ("min_child_samples", "min_data_in_leaf"),
                                   ("max_depth", "max_depth")):
        want, got = float(hp[sidecar_key]), float(params[block_key])
        if not np.isclose(want, got, rtol=1e-5, atol=0.0):
            raise SystemExit(f"hyperparameter {sidecar_key}: sidecar {want} vs booster {got}")

    recipe = Recipe(
        data_file=REPO_ROOT / sidecar["training_table"],
        features=features,
        hyperparameters=hp,
        monotone=mono,
        monotone_method=sidecar["monotone_constraints_method"],
        seed=seed,
        num_threads=int(params["num_threads"]),
        max_boost_round=int(params["num_iterations"]),
        early_stopping_rounds=int(sidecar["early_stopping_rounds"]),
        sha256=digest,
        booster_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        n_trees=int(booster.num_trees()),
    )
    print(f"Recipe read from {PRODUCTION_BOOSTER.name} (sha256 {digest[:16]}...)")
    print(f"  table={recipe.data_file.name}  features={len(features)}  seed={seed}  "
          f"threads={recipe.num_threads}  max_rounds={recipe.max_boost_round}  "
          f"early_stop={recipe.early_stopping_rounds}")
    print(f"  target=logit(alpha-1)  constrained="
          f"{[f for f, m in zip(features, mono) if m]}  trees={recipe.n_trees}")
    return booster, recipe


# --------------------------------------------------------------------------
# Target transform
# --------------------------------------------------------------------------
def to_z(alpha: np.ndarray) -> np.ndarray:
    """Forward map, verbatim from the sidecar's ``target_transform.forward``."""
    u = np.clip(np.asarray(alpha, dtype=np.float64) - 1.0, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(u / (1.0 - u))


def to_alpha(z: np.ndarray) -> np.ndarray:
    """Inverse map.  Cannot leave [1, 2], so no post-hoc clip is applied."""
    return 1.0 + 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def predict_alpha(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return to_alpha(booster.predict(X, num_iteration=booster.best_iteration))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_corpus(recipe: Recipe) -> pd.DataFrame:
    """Load the training table exactly as ``train_gart2.load_data`` does."""
    print(f"Loading {recipe.data_file} ...")
    df = pd.read_csv(recipe.data_file)
    mst = df["mst_total_length"].replace(0, np.nan)
    df["alpha"] = (df["optimal_cost"] / mst).clip(1.0, 2.0)
    df = df.dropna(subset=["alpha"]).reset_index(drop=True)
    if (df["optimal_cost"] <= 0).any():
        raise ValueError("optimal_cost contains non-positive values; percent errors undefined")
    missing = [c for c in recipe.features if c not in df.columns]
    if missing:
        raise ValueError(f"{recipe.data_file.name} is missing {missing}")
    print(f"  rows={len(df)}  splits={df['split'].value_counts().to_dict()}")
    return df


def load_corrupt_names(path: Path) -> set[str]:
    """Read the audited list of instances whose stored tour is desynchronised."""
    names = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if not names:
        raise ValueError(f"{path} contained no instance names")
    print(f"  corrupt instances listed: {len(names)}")
    return names


def dominant_non_uniform_family(instance_name: str) -> str:
    """Family with the most axes after excluding uniform.

    Ties resolve to ``mixed``; instances with no non-uniform axis are
    ``all_uniform``.
    """
    match = NAME_PATTERN.match(instance_name)
    if match is None:
        raise ValueError(f"instance name does not match N{{n}}_D{{d}}_G{{g}}_{{letters}}_{{seq}}: {instance_name}")
    letters = match.group(4)
    counts = {fam: 0 for fam in NON_UNIFORM_FAMILIES}
    for char in letters:
        family = FAMILY_BY_LETTER.get(char)
        if family is None:
            raise ValueError(f"unknown generator letter {char!r} in {instance_name}")
        if family != "uniform":
            counts[family] += 1
    top = max(counts.values())
    if top == 0:
        return "all_uniform"
    winners = [fam for fam, cnt in counts.items() if cnt == top]
    return winners[0] if len(winners) == 1 else "mixed"


# --------------------------------------------------------------------------
# Fit / evaluate
# --------------------------------------------------------------------------
def make_mape_feval(mst: np.ndarray, cost: np.ndarray):
    """Cost-level MAPE as a LightGBM eval function, in z-space."""
    def feval(preds, _ds):
        err = (to_alpha(preds) * mst - cost) / cost
        return "cost_mape", float(np.mean(np.abs(err)) * 100.0), False
    return feval


def fit_model(
    X_train: pd.DataFrame,
    alpha_train: np.ndarray,
    X_val: pd.DataFrame,
    alpha_val: np.ndarray,
    mst_val: np.ndarray,
    cost_val: np.ndarray,
    recipe: Recipe,
) -> lgb.Booster:
    """Fit under the shipped recipe, early-stopped on cost-level val MAPE."""
    if X_train.empty or X_val.empty:
        raise ValueError("empty training or validation set")
    params = dict(recipe.hyperparameters)
    params.update({
        "objective": "regression_l2",
        "metric": "None",  # early stopping is driven by the feval
        "boosting_type": "gbdt",
        "seed": recipe.seed,
        "num_threads": recipe.num_threads,
        "verbosity": -1,
        "feature_pre_filter": False,
        "monotone_constraints": list(recipe.monotone),
        "monotone_constraints_method": recipe.monotone_method,
    })
    dtrain = lgb.Dataset(X_train, label=to_z(alpha_train))
    dval = lgb.Dataset(X_val, label=to_z(alpha_val), reference=dtrain)
    return lgb.train(
        params, dtrain, num_boost_round=recipe.max_boost_round,
        valid_sets=[dval], valid_names=["val"],
        feval=make_mape_feval(mst_val, cost_val),
        callbacks=[lgb.early_stopping(recipe.early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(0)],
    )


def get_model(
    tag: str,
    X_train: pd.DataFrame,
    alpha_train: np.ndarray,
    X_val: pd.DataFrame,
    alpha_val: np.ndarray,
    mst_val: np.ndarray,
    cost_val: np.ndarray,
    recipe: Recipe,
    force: bool,
) -> lgb.Booster:
    """Refit ``tag`` unless a cached artifact for *this recipe* already exists.

    The cache filename carries the shipped artifact's hash, so caches written
    against a previous production model are unreachable rather than silently
    reused.  That is precisely how this module came to report the predecessor.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{tag}_{recipe.sha256[:12]}.joblib"
    if path.exists() and not force:
        print(f"[{tag}] cached model found -> {path.name} (use --force to refit)")
        return joblib.load(path)
    print(f"[{tag}] fitting on {len(X_train)} train / {len(X_val)} val rows ...")
    started = time.perf_counter()
    model = fit_model(X_train, alpha_train, X_val, alpha_val, mst_val, cost_val, recipe)
    elapsed = time.perf_counter() - started
    joblib.dump(model, path)
    print(f"[{tag}] done in {elapsed / 60.0:.2f} min, best_iteration={model.best_iteration} -> {path.name}")
    return model


def cost_metrics(
    model: lgb.Booster,
    X: pd.DataFrame,
    mst: np.ndarray,
    cost: np.ndarray,
) -> dict[str, float]:
    """Percent-error statistics on tour cost = alpha_hat * MST."""
    if len(X) == 0:
        raise ValueError("evaluation slice is empty")
    err = (predict_alpha(model, X) * mst - cost) / cost
    return {
        "N": int(len(err)),
        "MAPE_pct": float(np.mean(np.abs(err)) * 100.0),
        "SDPE_pct": float(np.std(err, ddof=1) * 100.0) if len(err) > 1 else float("nan"),
        "MedAPE_pct": float(np.median(np.abs(err)) * 100.0),
        "MSPE_pct": float(np.mean(err) * 100.0),
    }


def record(
    rows: list[dict[str, object]],
    experiment: str,
    eval_slice: str,
    model: lgb.Booster,
    df: pd.DataFrame,
    X: pd.DataFrame,
    mask: pd.Series,
    train_rows: int | str,
) -> None:
    """Evaluate ``model`` on ``mask`` and append one tidy result row."""
    if not mask.any():
        raise ValueError(f"{experiment}/{eval_slice}: evaluation mask selected zero rows")
    metrics = cost_metrics(
        model,
        X[mask],
        df.loc[mask, "mst_total_length"].to_numpy(),
        df.loc[mask, "optimal_cost"].to_numpy(),
    )
    rows.append(
        {
            "experiment": experiment,
            "eval_slice": eval_slice,
            "N": metrics["N"],
            "MAPE_pct": round(metrics["MAPE_pct"], 4),
            "SDPE_pct": round(metrics["SDPE_pct"], 4),
            "MedAPE_pct": round(metrics["MedAPE_pct"], 4),
            "MSPE_pct": round(metrics["MSPE_pct"], 4),
            "n_trees": int(model.best_iteration),
            "train_rows": train_rows,
        }
    )


def check_removed(mask_before: pd.Series, mask_after: pd.Series, label: str) -> None:
    """Fail loudly if a filter that should remove rows removed none."""
    removed = int(mask_before.sum() - mask_after.sum())
    if removed <= 0:
        raise ValueError(f"{label}: filter removed {removed} rows; expected a positive count")
    print(f"  {label}: removed {removed} rows ({int(mask_before.sum())} -> {int(mask_after.sum())})")


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------
def show_transform_is_load_bearing(
    shipped: lgb.Booster, X: pd.DataFrame, mst: np.ndarray, cost: np.ndarray
) -> dict[str, float]:
    """Score the shipped booster with the inverse logit omitted.

    The point of this control is that the bit-for-bit check below is only
    meaningful if applying the transform matters.  A refit path written for a
    plain-``alpha`` target would not raise -- it would produce finite,
    plausible-looking numbers -- so the failure has to be demonstrated rather
    than assumed.  Raw ``z`` sits near ``log(0.13/0.87) ~ -1.9``; read as an
    ``alpha`` it prices every tour at (or below) its MST.
    """
    z = shipped.predict(X, num_iteration=shipped.best_iteration)
    correct = (to_alpha(z) * mst - cost) / cost
    naive = (np.clip(z, 1.0, 2.0) * mst - cost) / cost
    unclipped = (z * mst - cost) / cost
    out = {
        "correct_mape_pct": float(np.mean(np.abs(correct)) * 100.0),
        "no_inverse_clipped_mape_pct": float(np.mean(np.abs(naive)) * 100.0),
        "no_inverse_raw_mape_pct": float(np.mean(np.abs(unclipped)) * 100.0),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "alpha_min": float(to_alpha(z).min()),
        "alpha_max": float(to_alpha(z).max()),
    }
    print("\nTransform control (shipped booster, full test split):")
    print(f"  raw z in [{out['z_min']:.4f}, {out['z_max']:.4f}]  ->  "
          f"alpha = 1 + sigmoid(z) in [{out['alpha_min']:.4f}, {out['alpha_max']:.4f}]")
    print(f"  with the inverse logit    : MAPE {out['correct_mape_pct']:.4f}%")
    print(f"  inverse omitted, clipped  : MAPE {out['no_inverse_clipped_mape_pct']:.4f}%")
    print(f"  inverse omitted, raw z    : MAPE {out['no_inverse_raw_mape_pct']:.4f}%")
    if out["no_inverse_clipped_mape_pct"] < NO_INVERSE_MIN_MAPE_PCT:
        raise SystemExit(
            "transform control failed: omitting the inverse logit changed MAPE by "
            f"less than {NO_INVERSE_MIN_MAPE_PCT} points, so the bit-for-bit check "
            "below would not be evidence that the transform is applied correctly")
    print(f"  -> the transform is load-bearing "
          f"({out['no_inverse_clipped_mape_pct'] / out['correct_mape_pct']:.0f}x worse without it); "
          "the bit-for-bit check is not vacuous")
    return out


def assert_bit_for_bit(fresh: lgb.Booster, shipped: lgb.Booster, recipe: Recipe,
                       X: pd.DataFrame) -> None:
    """E0 must reproduce the shipped booster exactly, or nothing downstream holds."""
    problems: list[str] = []

    n_fresh, n_shipped = int(fresh.best_iteration), int(shipped.best_iteration)
    if n_fresh != n_shipped or n_fresh != recipe.n_trees:
        problems.append(f"tree count: E0 {n_fresh} vs shipped {n_shipped} "
                        f"(sidecar {recipe.n_trees})")

    text = fresh.model_to_string(num_iteration=n_fresh)
    fresh_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if fresh_digest != recipe.booster_text_sha256:
        problems.append(f"serialised model text: E0 {fresh_digest[:16]}... vs "
                        f"shipped {recipe.booster_text_sha256[:16]}...")

    delta = np.abs(fresh.predict(X, num_iteration=n_fresh)
                   - shipped.predict(X, num_iteration=n_shipped))
    max_delta = float(delta.max())
    if max_delta != 0.0:
        problems.append(f"max |delta pred| = {max_delta:.3e} over {len(delta)} rows (want exactly 0)")

    print("\nE0 bit-for-bit check against the shipped artifact:")
    print(f"  trees            : E0 {n_fresh}  shipped {n_shipped}")
    print(f"  model text sha256: E0 {fresh_digest[:16]}...  shipped {recipe.booster_text_sha256[:16]}...")
    print(f"  max |delta pred| : {max_delta:.3e} over {len(delta)} corpus rows")
    if problems:
        raise SystemExit(
            "E0 does NOT reproduce the shipped booster:\n  - " + "\n  - ".join(problems)
            + "\nEvery E1/E2/E3 delta would be measured against the wrong baseline. Stopping.")
    print("  -> E0 reproduces the shipped booster bit-for-bit.")


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(ALL_EXPERIMENTS),
        default=list(ALL_EXPERIMENTS),
        help="which refits to run (E0 is always needed as the comparison baseline)",
    )
    parser.add_argument("--force", action="store_true", help="refit even if a cached model exists")
    parser.add_argument("--no-bank", action="store_true", help="skip the number-bank merge")
    args = parser.parse_args()
    requested = [e for e in ALL_EXPERIMENTS if e in set(args.experiments)]

    shipped, recipe = load_recipe()
    df = load_corpus(recipe)
    X = df[list(recipe.features)]
    alpha = df["alpha"].to_numpy()
    mst_all = df["mst_total_length"].to_numpy()
    cost_all = df["optimal_cost"].to_numpy()

    is_train = df["split"] == "train"
    is_val = df["split"] == "val"
    is_test = df["split"] == "test"

    corrupt = load_corrupt_names(CORRUPT_FILE)
    is_corrupt = df["instance_name"].isin(corrupt)
    missing = len(corrupt) - int(is_corrupt.sum())
    if missing:
        raise ValueError(f"{missing} corrupt instance names are absent from {recipe.data_file.name}")

    family = df.loc[is_test, "instance_name"].map(dominant_non_uniform_family)

    # Evaluation slices on the test split.
    slices: dict[str, pd.Series] = {
        "test_all": is_test,
        "test_clean": is_test & ~is_corrupt,
        "test_d15": is_test & (df["dimension"] == 15),
        "test_d25": is_test & (df["dimension"] == 25),
        "test_d10_20_30": is_test & df["dimension"].isin(CONTROL_DIMS),
        "test_n_gt200_le1000": is_test & (df["n_customers"] > SMALL_N_MAX) & (df["n_customers"] <= LARGE_N_MAX),
        "test_n_le200": is_test & (df["n_customers"] <= SMALL_N_MAX),
    }
    for group in ["all_uniform", *NON_UNIFORM_FAMILIES, "mixed"]:
        member = pd.Series(False, index=df.index)
        member.loc[family.index[family == group]] = True
        if member.any():
            slices[f"test_genmix_{group}"] = member

    # --- transform control, before anything depends on the transform -----
    show_transform_is_load_bearing(
        shipped, X[is_test], mst_all[is_test.to_numpy()], cost_all[is_test.to_numpy()])

    rows: list[dict[str, object]] = []

    def refit(tag: str, tr: pd.Series, va: pd.Series) -> lgb.Booster:
        va_np = va.to_numpy()
        return get_model(tag, X[tr], alpha[tr.to_numpy()], X[va], alpha[va_np],
                         mst_all[va_np], cost_all[va_np], recipe, args.force)

    # --- E0: baseline refit on the shipped split -------------------------
    n_train_e0 = int(is_train.sum())
    model_e0 = refit("E0", is_train, is_val)
    assert_bit_for_bit(model_e0, shipped, recipe, X)
    for name, mask in slices.items():
        record(rows, "E0", name, model_e0, df, X, mask, n_train_e0)

    # --- Shipped artifact, same test rows --------------------------------
    if tuple(shipped.feature_name()) != tuple(X.columns):
        raise ValueError("shipped model feature names do not match the rebuilt feature matrix")
    record(rows, "SHIPPED", "test_all", shipped, df, X, is_test, "")

    e0 = next(r for r in rows if r["experiment"] == "E0" and r["eval_slice"] == "test_all")
    sh = next(r for r in rows if r["experiment"] == "SHIPPED")
    if (e0["MAPE_pct"], e0["SDPE_pct"], e0["n_trees"]) != (sh["MAPE_pct"], sh["SDPE_pct"], sh["n_trees"]):
        raise SystemExit("E0 and SHIPPED disagree on test_all after a bit-for-bit match; impossible")
    sidecar_test = json.loads(PRODUCTION_SIDECAR.read_text(encoding="utf-8"))["test_metrics"]
    print(f"\nE0 test_all MAPE {e0['MAPE_pct']:.4f}%  SDPE {e0['SDPE_pct']:.4f}%  "
          f"trees {e0['n_trees']}")
    print(f"  sidecar test_metrics: MAPE {sidecar_test['mape']:.4f}%  "
          f"SDPE {sidecar_test['sdpe']:.4f}%")
    for label, got, want in (("MAPE", e0["MAPE_pct"], sidecar_test["mape"]),
                             ("SDPE", e0["SDPE_pct"], sidecar_test["sdpe"])):
        if abs(float(got) - float(want)) > 5e-4:
            raise SystemExit(f"E0 {label} {got} disagrees with the sidecar's {want}")
    print("  -> agrees with the frozen provenance record.")

    # --- E1: clean labels ------------------------------------------------
    if "E1" in requested:
        print("\n=== E1 clean labels ===")
        tr, va = is_train & ~is_corrupt, is_val & ~is_corrupt
        check_removed(is_train, tr, "E1 train")
        check_removed(is_val, va, "E1 val")
        check_removed(is_test, slices["test_clean"], "E1 test")
        model = refit("E1", tr, va)
        record(rows, "E1", "test_clean", model, df, X, slices["test_clean"], int(tr.sum()))

    # --- E2: leave-dimensions-out ---------------------------------------
    if "E2" in requested:
        print("\n=== E2 leave-dimensions-out ===")
        keep = ~df["dimension"].isin(HELD_OUT_DIMS)
        tr, va = is_train & keep, is_val & keep
        check_removed(is_train, tr, "E2 train")
        check_removed(is_val, va, "E2 val")
        model = refit("E2", tr, va)
        for name in ("test_d15", "test_d25", "test_d10_20_30"):
            record(rows, "E2", name, model, df, X, slices[name], int(tr.sum()))

    # --- E3: leave-large-n-out ------------------------------------------
    if "E3" in requested:
        print("\n=== E3 leave-large-n-out ===")
        keep = df["n_customers"] <= SMALL_N_MAX
        tr, va = is_train & keep, is_val & keep
        check_removed(is_train, tr, "E3 train")
        check_removed(is_val, va, "E3 val")
        model = refit("E3", tr, va)
        for name in ("test_n_gt200_le1000", "test_n_le200"):
            record(rows, "E3", name, model, df, X, slices[name], int(tr.sum()))

    merged = write_results(rows)
    if not args.no_bank:
        merge_into_bank(merged)


def write_results(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Merge new rows into the tidy results CSV and print a markdown summary."""
    new = pd.DataFrame(rows)
    if RESULTS_FILE.exists():
        old = pd.read_csv(RESULTS_FILE)
        old = old[~old["experiment"].isin(set(new["experiment"]))]
        new = pd.concat([old, new], ignore_index=True)
    order = {name: i for i, name in enumerate(["E0", "SHIPPED", "E1", "E2", "E3"])}
    new = new.sort_values(
        by=["experiment", "eval_slice"],
        key=lambda s: s.map(order) if s.name == "experiment" else s,
    ).reset_index(drop=True)
    new.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults -> {RESULTS_FILE}")

    header = ["experiment", "eval_slice", "N", "MAPE_pct", "SDPE_pct", "MedAPE_pct", "MSPE_pct", "n_trees", "train_rows"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for _, row in new.iterrows():
        cells = [str(row[c]) for c in header]
        print("| " + " | ".join(cells) + " |")
    return new


if __name__ == "__main__":
    main()
