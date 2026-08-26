"""Fold ``generalization_results.csv`` into the number bank under stable keys.

``paper_tooling/generalization_experiments.py`` has always written a tidy CSV
and has never written any of it into ``paper_numbers.json``.  Every sentence in
Section~\\ref{subsec:generalization} -- the bit-for-bit baseline, the two
leave-out refits, the bias shifts, the cost of withholding a stratum -- and the
matching limitation in the conclusion therefore had no bank key, so no prose
claim could point at one and ``check_prose_numbers.py`` could not verify a
single one of them.  ``prose_claim_map.md`` groups exactly ten UNGENERATED
claims under GEN-REPOINT for that reason.  This module closes the gap, on the
same pattern as ``paper_tooling/paired_bank.py``.

Key shape
---------
``gen_<experiment>_<eval_slice>_<field>``

``experiment`` and ``eval_slice`` are slugged from the CSV's own columns, so
the keys are produced by the data rather than restated here.  No model name
appears in a key: the experiment tags name a *protocol* (baseline refit,
leave-dimensions-out, leave-large-n-out), and a claim written against these
keys therefore survives a production-model swap untouched.  ``gen_shipped_*``
carries the shipped artifact scored on the same rows, which is what makes the
E0 reproducibility statement checkable rather than asserted.

Sign convention
---------------
``mspe_pct`` is the **mean signed** percent error, not a squared quantity --
the column has carried that meaning since the module was written, and the
manuscript's "mean signed error" sentences quote it.  Negative means the
estimator under-predicts tour cost.

Two entry points, one function
------------------------------
``generalization_experiments.main()`` calls :func:`merge_into_bank` after
writing the CSV, and ``build_paper_tables.main()`` calls
:func:`generalization_bank_numbers` before writing the bank, so a full rebuild
emits these keys natively instead of dropping them.  Running this module
directly merges the same keys from the existing CSV.  All three paths call the
same function, so they cannot drift.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import REPO_ROOT  # noqa: E402

TABLES = REPO_ROOT / "paper_tooling" / "tables"
P_RESULTS = REPO_ROOT / "paper_tooling" / "generalization_results.csv"
P_BANK = TABLES / "paper_numbers.json"

#: Result columns worth banking. ``experiment``/``eval_slice`` are identity.
GEN_FIELDS = ("N", "MAPE_pct", "SDPE_pct", "MedAPE_pct", "MSPE_pct", "n_trees", "train_rows")
INT_FIELDS = frozenset({"N", "n_trees", "train_rows"})

KEY_PREFIX = "gen_"


def _slug(s: str) -> str:
    """Same slug rule as ``build_paper_tables._slug`` -- keys must agree."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def gen_key(experiment: str, eval_slice: str, field: str) -> str:
    return f"{KEY_PREFIX}{_slug(experiment)}_{_slug(eval_slice)}_{_slug(field)}"


def generalization_bank_numbers(results) -> dict[str, object]:
    """Bank entries for every row of a generalization-results frame.

    ``results`` is the frame ``generalization_experiments`` writes to
    ``generalization_results.csv``.  Non-finite and blank values are stored as
    ``None``, matching how the rest of the bank records an undefined cell, so
    ``Sources.resolve`` reports "undefined in the generated artifact" instead
    of silently comparing against a NaN.  ``train_rows`` is blank on the
    ``SHIPPED`` row by design -- that row is a scoring of a frozen artifact,
    not a refit.
    """
    numbers: dict[str, object] = {}
    collisions: list[str] = []
    if results is None or len(results) == 0:
        return numbers
    for _, r in results.iterrows():
        for field in GEN_FIELDS:
            if field not in r.index:
                continue
            key = gen_key(str(r["experiment"]), str(r["eval_slice"]), field)
            if key in numbers:
                collisions.append(key)
            try:
                fv = float(r[field])
            except (TypeError, ValueError):
                numbers[key] = None
                continue
            if not math.isfinite(fv):
                numbers[key] = None
            elif field in INT_FIELDS:
                numbers[key] = int(fv)
            else:
                numbers[key] = round(fv, 9)
    if collisions:
        raise SystemExit(
            "generalization_bank: key collision -- two rows slug to the same key, so "
            f"one would silently overwrite the other: {sorted(set(collisions))[:5]}")
    return numbers


def load_results():
    """Read ``generalization_results.csv``, or ``None`` if it has not been run."""
    import pandas as pd

    if not P_RESULTS.exists():
        return None
    return pd.read_csv(P_RESULTS)


def merge_into_bank(results) -> dict[str, object]:
    """Merge the generalization keys into ``paper_numbers.json`` in place."""
    if not P_BANK.exists():
        raise SystemExit(f"missing {P_BANK}; run build_paper_tables.py first")
    added = generalization_bank_numbers(results)
    bank: dict[str, object] = json.loads(P_BANK.read_text(encoding="utf-8"))

    stale = [k for k in bank if k.startswith(KEY_PREFIX) and k not in added]
    changed = sum(1 for k, v in added.items() if k in bank and bank[k] != v)
    fresh = sum(1 for k in added if k not in bank)
    for k in stale:
        del bank[k]
    bank.update(added)
    P_BANK.write_text(json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8")

    print(f"generalization_bank: {len(results)} rows -> {len(added)} keys "
          f"({fresh} new, {changed} updated, {len(stale)} stale removed); "
          f"bank now holds {len(bank)} keys")
    return added


def main() -> int:
    results = load_results()
    if results is None:
        raise SystemExit(f"missing {P_RESULTS}; run generalization_experiments.py first")
    merge_into_bank(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
