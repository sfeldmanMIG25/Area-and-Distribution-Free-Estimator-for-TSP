"""Observed range of the production model's predicted ``alpha`` on the test split.

Why this exists
---------------
The manuscript stated that predictions are clipped to ``[1, 2]`` at inference
and that "the clip did not activate".  Both halves were wrong about the shipped
model, in opposite directions.  ``gart2_final.json`` records
``target_transform.clip_after_inverse = false``: the learner regresses
``z = logit(alpha - 1)`` and inverts with ``alpha = 1 + sigma(z)``, so the bound
is *structural*.  There is no clip to activate.  Reporting "the clip did not
activate" therefore credits a safeguard that does not exist, and it invites the
reader to conclude the interval was checked empirically when in fact nothing
could have left it.

What replaces the claim is the one empirical fact that is still worth stating:
how far inside the structural bound the predictions actually sit.  A range that
hugged 1.000 / 2.000 would mean the logit was saturating and the bound was doing
real work; a range comfortably inside means it is not.  That is a fact about the
model, so it needs a generator.

Nothing here retrains or rewrites the artifact.  It is one forward pass over the
16,920 test rows.

Output
------
``tables/alphahat_range.csv``   one row per split, plus the transform metadata
Bank keys                       ``model_alphahat_<split>_{n,min,max,mean,sd}``
                                ``model_alphahat_clip_after_inverse``
                                ``model_alphahat_transform_inverse``

CLI
---
    python paper_tooling/alphahat_range.py            # write CSV + merge bank
    python paper_tooling/alphahat_range.py --no-bank  # print only
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
    PRODUCTION_BOOSTER,
    PRODUCTION_SIDECAR,
    REPO_ROOT,
)

TABLES = REPO_ROOT / "paper_tooling" / "tables"
OUT_CSV = TABLES / "alphahat_range.csv"
BANK = TABLES / "paper_numbers.json"
#: This module's keys, kept separately so a full table rebuild can re-merge them.
SIDECAR_KEYS = TABLES / "alphahat_numbers.json"

KEY_PREFIX = "model_alphahat_"


def inverse_transform(z: np.ndarray) -> np.ndarray:
    """``alpha = 1 + 1 / (1 + exp(-z))`` -- the sidecar's recorded inverse.

    Written with :func:`scipy.special.expit`'s numerically stable branch inline
    so the module does not need scipy: for very negative ``z`` the naive form
    overflows in ``exp(-z)``.  The bound is structural precisely because this
    function maps the whole real line into ``(1, 2)``, so an implementation that
    can overflow to ``inf`` would destroy the property being reported.
    """
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return 1.0 + out


def score_split(split: str = "test") -> dict:
    """Forward-pass the production booster over one split of its training table."""
    sidecar = json.loads(PRODUCTION_SIDECAR.read_text(encoding="utf-8"))
    booster = joblib.load(PRODUCTION_BOOSTER)
    feats = list(booster.feature_name())
    if feats != list(sidecar["features_in_booster_order"]):
        raise SystemExit(
            "booster feature order disagrees with the sidecar; the artifact and "
            "its provenance record have drifted apart"
        )

    table = REPO_ROOT / sidecar["training_table"]
    df = pd.read_csv(table, usecols=["split", *feats], low_memory=False)
    rows = df[df["split"] == split]
    if rows.empty:
        raise SystemExit(f"no rows with split == {split!r} in {table.name}")

    z = booster.predict(rows[feats])
    alpha = inverse_transform(np.asarray(z, dtype=float))

    expected = sidecar.get("rows", {}).get(split)
    return {
        "split": split,
        "n": int(alpha.size),
        "n_expected": int(expected) if expected is not None else None,
        "alphahat_min": float(alpha.min()),
        "alphahat_max": float(alpha.max()),
        "alphahat_mean": float(alpha.mean()),
        "alphahat_sd": float(alpha.std(ddof=1)),
        "structural_lower": 1.0,
        "structural_upper": 2.0,
        "clip_after_inverse": bool(sidecar["target_transform"]["clip_after_inverse"]),
        "transform_inverse": sidecar["target_transform"]["inverse"],
    }


def bank_numbers(records: list[dict]) -> dict[str, object]:
    numbers: dict[str, object] = {}
    for r in records:
        s = r["split"]
        numbers[f"{KEY_PREFIX}{s}_n"] = r["n"]
        for field in ("min", "max", "mean", "sd"):
            numbers[f"{KEY_PREFIX}{s}_{field}"] = round(r[f"alphahat_{field}"], 9)
    first = records[0]
    numbers[f"{KEY_PREFIX}clip_after_inverse"] = first["clip_after_inverse"]
    numbers[f"{KEY_PREFIX}transform_inverse"] = first["transform_inverse"]
    return numbers


def alphahat_bank_numbers() -> dict[str, object]:
    """This module's bank keys, from the sidecar it writes.

    ``build_paper_tables.main()`` rewrites ``paper_numbers.json`` wholesale, so
    without a hook of this shape a full table rebuild silently deletes every key
    written here -- the same failure ``paired_bank`` and ``generalization_bank``
    exist to prevent. Reading the sidecar rather than re-scoring keeps the table
    builder free of lightgbm and of a 106,272-row forward pass.
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
    # All three by default: the bank drops any ``model_alphahat_*`` key this run
    # does not produce, so a narrower default would silently delete keys
    # whenever someone ran the module without arguments.
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--no-bank", action="store_true")
    a = ap.parse_args(argv)

    records = [score_split(s) for s in a.splits]
    out = pd.DataFrame(records)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.drop(columns=["transform_inverse"]).to_string(index=False))
    print(f"\ninverse transform: {records[0]['transform_inverse']}")
    print(f"clip_after_inverse: {records[0]['clip_after_inverse']} "
          f"-- the bound is structural, nothing is enforced at inference")
    print(f"wrote {OUT_CSV}")

    if not a.no_bank:
        fresh, changed, stale = merge_bank(bank_numbers(records))
        print(f"bank: {fresh} new, {changed} updated, {stale} stale removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
