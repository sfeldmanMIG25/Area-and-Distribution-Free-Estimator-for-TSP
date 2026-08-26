"""SHAP attribution for the *production* booster, per feature and per family.

Why this replaces ``shap_analyzer_v3.py``
-----------------------------------------
The manuscript's SHAP paragraph was computed for the predecessor.  Repointing
the old script at ``gart2_final.joblib`` would not have been enough, for three
reasons, all of them properties of that script rather than of the model:

1. it hardcodes ``tsp_features_v3.csv``, which has no ``greedy_nn_over_mst``
   column, so ``X = X[expected]`` raises ``KeyError`` against a 31-feature
   booster.  The data path has to move to the sidecar's ``training_table``;
2. it persists nothing numeric -- it prints the top ten rows and writes two
   PNGs -- so a re-run leaves no artifact a checker can read;
3. it reports per-feature shares only, while the manuscript quotes *family*
   shares ("dimension and node count jointly contribute ...").  The grouping
   existed nowhere in the repository, so the family numbers were unreproducible
   in principle, not merely stale.

This module fixes all three: it reads the model through ``model_registry``,
writes every feature and every family to CSV, and aggregates through
``model_registry.FEATURE_FAMILIES``, which is checked against the booster's own
roster before anything is computed.

What the numbers mean, and one thing they no longer mean
--------------------------------------------------------
SHAP decomposes the booster's **raw margin**.  The predecessor regressed
``alpha`` directly, so its attributions were in units of ``alpha``.  The
production model regresses ``z = logit(alpha - 1)``, so these attributions are
in units of ``z``.  Shares are invariant to the overall scale, but the logit is
non-linear, so a feature's share is *not* the same quantity it was: attribution
in ``z`` weights instances near the interval's ends differently from attribution
in ``alpha``.  The ranking is therefore comparable to the predecessor's only as
a statement about model reliance, which is all the manuscript claims for it.

Sampling matches the superseded script exactly -- test split, 5,000 rows,
``random_state=42``, mean absolute SHAP -- so any movement in the numbers is
attributable to the model and the feature set, not to a changed protocol.

Output
------
``tables/shap_ranking.csv``   one row per feature: mean |SHAP|, share, rank
``tables/shap_families.csv``  one row per family and per coarse group
Bank keys                     ``shap_feature_<feature>_{mean_abs,share_pct,rank}``
                              ``shap_family_<family>_{share_pct,n_features}``
                              ``shap_group_<group>_{share_pct,n_features}``
                              ``shap_top10_mst_derived``, ``shap_n_features``,
                              ``shap_sample_rows``

CLI
---
    python paper_tooling/shap_production.py
    python paper_tooling/shap_production.py --no-bank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import (  # noqa: E402
    FAMILY_GROUPS,
    FEATURE_FAMILIES,
    PRODUCTION_BOOSTER,
    PRODUCTION_SIDECAR,
    REPO_ROOT,
    check_feature_families,
)

TABLES = REPO_ROOT / "paper_tooling" / "tables"
OUT_RANKING = TABLES / "shap_ranking.csv"
OUT_FAMILIES = TABLES / "shap_families.csv"
BANK = TABLES / "paper_numbers.json"
#: This module's keys, kept separately so a full table rebuild can re-merge them.
SIDECAR_KEYS = TABLES / "shap_numbers.json"

SAMPLE_SIZE = 5_000
SEED = 42
SPLIT = "test"
KEY_PREFIX = "shap_"


def compute(sample_size: int = SAMPLE_SIZE, seed: int = SEED, split: str = SPLIT):
    """Return ``(ranking, families, meta)`` for the production booster."""
    import shap

    sidecar = json.loads(PRODUCTION_SIDECAR.read_text(encoding="utf-8"))
    booster = joblib.load(PRODUCTION_BOOSTER)
    feats = list(booster.feature_name())
    if feats != list(sidecar["features_in_booster_order"]):
        raise SystemExit("booster roster disagrees with its sidecar")
    check_feature_families(feats)

    table = REPO_ROOT / sidecar["training_table"]
    df = pd.read_csv(table, usecols=["split", *feats], low_memory=False)
    X = df[df["split"] == split][feats]
    n_split = len(X)
    if n_split == 0:
        raise SystemExit(f"no rows with split == {split!r} in {table.name}")
    if n_split > sample_size:
        X = X.sample(n=sample_size, random_state=seed)

    sv = shap.TreeExplainer(booster).shap_values(X)
    mean_abs = np.abs(np.asarray(sv)).mean(axis=0)

    ranking = pd.DataFrame({
        "feature": feats,
        "family": [FEATURE_FAMILIES[f] for f in feats],
        "group": [FAMILY_GROUPS[FEATURE_FAMILIES[f]] for f in feats],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = float(ranking["mean_abs_shap"].sum())
    ranking["share_pct"] = 100.0 * ranking["mean_abs_shap"] / total
    ranking["cum_share_pct"] = ranking["share_pct"].cumsum()
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    fam = (
        ranking.groupby("family", as_index=False)
        .agg(n_features=("feature", "size"), share_pct=("share_pct", "sum"),
             mean_abs_shap=("mean_abs_shap", "sum"))
        .assign(group=lambda d: d["family"].map(FAMILY_GROUPS), level="family")
    )
    grp = (
        ranking.groupby("group", as_index=False)
        .agg(n_features=("feature", "size"), share_pct=("share_pct", "sum"),
             mean_abs_shap=("mean_abs_shap", "sum"))
        .assign(family=lambda d: d["group"], level="group")
    )
    families = (
        pd.concat([fam, grp], ignore_index=True)[
            ["level", "family", "group", "n_features", "mean_abs_shap", "share_pct"]
        ]
        .sort_values(["level", "share_pct"], ascending=[True, False])
        .reset_index(drop=True)
    )

    top10 = ranking.head(10)
    meta = {
        "n_features": len(feats),
        "sample_rows": int(len(X)),
        "split_rows": int(n_split),
        "split": split,
        "seed": seed,
        "top10_mst_derived": int((top10["group"] == "mst").sum()),
        "top10_geometric": int((top10["group"] == "geometric").sum()),
        "top10_greedy": int((top10["group"] == "greedy").sum()),
        "artifact_sha256": sidecar["sha256"],
        "attribution_space": "raw margin z = logit(alpha - 1)",
    }
    return ranking, families, meta


def _slug(s: str) -> str:
    return str(s).lower().replace("-", "_")


def bank_numbers(ranking: pd.DataFrame, families: pd.DataFrame, meta: dict):
    numbers: dict[str, object] = {}
    for _, r in ranking.iterrows():
        f = _slug(r["feature"])
        numbers[f"{KEY_PREFIX}feature_{f}_mean_abs"] = round(float(r["mean_abs_shap"]), 9)
        numbers[f"{KEY_PREFIX}feature_{f}_share_pct"] = round(float(r["share_pct"]), 6)
        numbers[f"{KEY_PREFIX}feature_{f}_rank"] = int(r["rank"])
    for _, r in families.iterrows():
        kind = "family" if r["level"] == "family" else "group"
        name = _slug(r["family"])
        numbers[f"{KEY_PREFIX}{kind}_{name}_share_pct"] = round(float(r["share_pct"]), 6)
        numbers[f"{KEY_PREFIX}{kind}_{name}_n_features"] = int(r["n_features"])
    numbers[f"{KEY_PREFIX}n_features"] = meta["n_features"]
    numbers[f"{KEY_PREFIX}sample_rows"] = meta["sample_rows"]
    numbers[f"{KEY_PREFIX}split_rows"] = meta["split_rows"]
    numbers[f"{KEY_PREFIX}top10_mst_derived"] = meta["top10_mst_derived"]
    numbers[f"{KEY_PREFIX}top10_geometric"] = meta["top10_geometric"]
    numbers[f"{KEY_PREFIX}top10_greedy"] = meta["top10_greedy"]
    return numbers


def shap_bank_numbers() -> dict[str, object]:
    """This module's bank keys, from the sidecar it writes.

    ``build_paper_tables.main()`` rewrites ``paper_numbers.json`` wholesale, so
    without a hook of this shape a full table rebuild silently deletes every key
    written here. Reading the sidecar rather than recomputing keeps the table
    builder free of ``shap`` and of a 5,000-row TreeExplainer pass.
    """
    if not SIDECAR_KEYS.exists():
        return {}
    return json.loads(SIDECAR_KEYS.read_text(encoding="utf-8"))


def merge_bank(numbers: dict[str, object]) -> tuple[int, int, int]:
    SIDECAR_KEYS.write_text(json.dumps(numbers, indent=2, sort_keys=True),
                            encoding="utf-8")
    bank: dict[str, object] = json.loads(BANK.read_text(encoding="utf-8"))
    stale = [k for k in bank if k.startswith(KEY_PREFIX) and k not in numbers]
    changed = sum(1 for k, v in numbers.items() if k in bank and bank[k] != v)
    fresh = sum(1 for k in numbers if k not in bank)
    for k in stale:
        del bank[k]
    bank.update(numbers)
    BANK.write_text(json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8")
    return fresh, changed, len(stale)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-bank", action="store_true")
    a = ap.parse_args(argv)

    ranking, families, meta = compute(a.sample_size, a.seed)
    TABLES.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUT_RANKING, index=False)
    families.to_csv(OUT_FAMILIES, index=False)

    print(f"model sha256 {meta['artifact_sha256'][:16]}...  "
          f"{meta['n_features']} features, {meta['sample_rows']} of "
          f"{meta['split_rows']} {SPLIT} rows, seed {meta['seed']}")
    print(f"attribution space: {meta['attribution_space']}\n")
    print(ranking[["rank", "feature", "family", "mean_abs_shap",
                   "share_pct", "cum_share_pct"]].round(6).to_string(index=False))
    print("\n=== by family ===")
    print(families.round(4).to_string(index=False))
    print(f"\nMST-derived in the top ten: {meta['top10_mst_derived']}")
    print(f"wrote {OUT_RANKING.name}, {OUT_FAMILIES.name}")

    if not a.no_bank:
        fresh, changed, stale = merge_bank(bank_numbers(ranking, families, meta))
        print(f"bank: {fresh} new, {changed} updated, {stale} stale removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
