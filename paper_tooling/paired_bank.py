"""Fold ``tables/paired_tests.csv`` into the number bank under stable keys.

``build_paper_tables.main()`` has always written ``paired_tests.csv`` and has
never written any of it into ``paper_numbers.json``.  Every significance
statement in the manuscript -- the paired mean difference, its bootstrap
interval, the Wilcoxon $p$ -- therefore had no bank key, so no prose claim could
point at one and ``check_prose_numbers.py`` could not verify a single one of
them.  This module closes that gap.  It is the single highest-leverage export
available: one function covers every significance claim in the paper.

Key shape
---------
``paired_<table>_<bucket_slug>_<display_b>_<field>``

``display_b`` is slugged from the CSV's own ``display_b`` column -- the label
``model_registry.MODEL_LABELS`` already assigned to the *comparison* model -- so
the keys match the rest of the bank (``tsplib_by_size_total_asymptotic_mst_ratio``
+ ``_mape_pct`` for the table cell, ``paired_tsplib_by_size_total_``
``asymptotic_mst_ratio`` + ``_wilcoxon_p`` for the test).  ``model_a`` is the
production model by construction and is deliberately *not* in the key: the
comparison is always "production vs this baseline", so a claim written against
these keys survives a production-model swap untouched.  No model name is
written here; the identifiers come from the data.

Sign convention (inherited from ``build_paper_tables.paired_test``)
-------------------------------------------------------------------
``mean_diff``/``ci_lo``/``ci_hi`` are ``|APE| of the production model`` minus
``|APE| of the baseline``, averaged over the instances where both are finite.
**Negative favours the production model.**  Prose that quotes a baseline's
advantage quotes the negated value, so those claims must negate in their
``expect`` expression rather than pointing at a different key.

Two entry points, one function
------------------------------
``build_paper_tables.main()`` calls :func:`paired_bank_numbers` before writing
the bank, so a full rebuild emits these keys natively.  Running this module
directly merges the same keys into the existing bank from the existing
``paired_tests.csv``, which is what to use when the paired tests have not
changed and a full table rebuild is not warranted.  Both paths call the same
function, so the two cannot drift.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import REPO_ROOT  # noqa: E402

TABLES = REPO_ROOT / "paper_tooling" / "tables"
P_PAIRED = TABLES / "paired_tests.csv"
P_BANK = TABLES / "paper_numbers.json"

# The columns worth banking. ``bucket``/``display_b`` are identity, not results.
PAIRED_FIELDS = ("n_pairs", "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p")

KEY_PREFIX = "paired_"


def _slug(s: str) -> str:
    """Same slug rule as ``build_paper_tables._slug`` -- keys must agree."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def paired_key(table: str, bucket_slug: str, display_b: str, field: str) -> str:
    return f"{KEY_PREFIX}{table}_{bucket_slug}_{_slug(display_b)}_{field}"


def paired_bank_numbers(pt) -> dict[str, object]:
    """Bank entries for every row of a paired-tests frame.

    ``pt`` is the frame ``build_paper_tables`` writes to ``paired_tests.csv``.
    Non-finite values are stored as ``None``, matching how the rest of the bank
    records an undefined cell, so ``Sources.resolve`` reports "undefined in the
    generated artifact" instead of silently comparing against a NaN.
    """
    import math

    numbers: dict[str, object] = {}
    collisions: list[str] = []
    for _, r in pt.iterrows():
        for field in PAIRED_FIELDS:
            if field not in r.index:
                continue
            key = paired_key(str(r["table"]), str(r["bucket_slug"]),
                             str(r["display_b"]), field)
            if key in numbers:
                collisions.append(key)
            v = r[field]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                numbers[key] = None
                continue
            if not math.isfinite(fv):
                numbers[key] = None
            elif field == "n_pairs":
                numbers[key] = int(fv)
            else:
                numbers[key] = round(fv, 9)
    if collisions:
        raise SystemExit(
            "paired_bank: key collision -- two rows slug to the same key, so one "
            f"would silently overwrite the other: {sorted(set(collisions))[:5]}")
    return numbers


def main() -> int:
    import pandas as pd

    if not P_PAIRED.exists():
        raise SystemExit(f"missing {P_PAIRED}; run build_paper_tables.py first")
    if not P_BANK.exists():
        raise SystemExit(f"missing {P_BANK}; run build_paper_tables.py first")

    pt = pd.read_csv(P_PAIRED)
    added = paired_bank_numbers(pt)
    bank: dict[str, object] = json.loads(P_BANK.read_text(encoding="utf-8"))

    stale = [k for k in bank if k.startswith(KEY_PREFIX) and k not in added]
    changed = sum(1 for k, v in added.items() if k in bank and bank[k] != v)
    fresh = sum(1 for k in added if k not in bank)
    for k in stale:
        del bank[k]
    bank.update(added)
    P_BANK.write_text(json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8")

    print(f"paired_bank: {len(pt)} rows -> {len(added)} keys "
          f"({fresh} new, {changed} updated, {len(stale)} stale removed); "
          f"bank now holds {len(bank)} keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
