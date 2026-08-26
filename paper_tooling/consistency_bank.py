"""Fold the consistency measurements into the number bank under stable keys.

``paper_tooling/consistency_31f.py`` measures the criterion the manuscript
actually argues from -- swept-feature monotonicity and the dispersion series --
for the two comparators that were never scored on it (``NN_31F``,
``LGBM_V4``), for the same-feature linear control, and for the production
model, in one session on one protocol.  None of it reaches
``tables/paper_numbers.json`` on its own, so no prose claim could point at a
bank key and ``check_prose_numbers.py`` could not verify one.  This module
closes that gap on the pattern of ``paper_tooling/generalization_bank.py``.

Key shapes
----------
``cons_probe_<model>_<swept>_<field>``        one swept-feature probe cell
``cons_nd_<axis>_<bucket>_<model>_<field>``   one ND dispersion cell
``cons_series_<axis>_<model>_<field>``        one series-monotonicity verdict

Model names appear in these keys, unlike ``gen_*``.  That is deliberate and is
the opposite trade-off: the whole point of the measurement is a *named*
comparison between the production model and two specific excluded
comparators, so a claim written against ``cons_probe_nn_31f_dimension_...``
must break if the model behind that name changes.  A protocol-shaped key would
survive a silent substitution, which is the defect this measurement exists to
prevent.

Booleans are banked as 0/1 integers because the bank's JSON consumers compare
numerically; ``strict_monotone_decreasing`` is the one field where the claim
in the prose is the boolean itself rather than a rounded quantity.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import REPO_ROOT  # noqa: E402

HERE = REPO_ROOT / "paper_tooling"
TABLES = HERE / "tables"
P_BANK = TABLES / "paper_numbers.json"
P_PROBE = HERE / "consistency_31f_probe.csv"
P_BUCKETS = HERE / "consistency_31f_nd_buckets.csv"
P_SERIES = HERE / "consistency_31f_nd_series.csv"

KEY_PREFIX = "cons_"

PROBE_FIELDS = ("n_probes", "grid_points", "n_pairs",
                "pct_nonincr_deployed", "n_viol_deployed", "viol_med_deployed",
                "viol_p99_deployed", "viol_max_deployed",
                "pct_nonincr_raw", "n_viol_raw", "viol_med_raw",
                "viol_p99_raw", "viol_max_raw",
                "pct_nonincr_deployed_tol1em6", "n_viol_deployed_tol1em6",
                "pct_nonincr_raw_tol1em6", "n_viol_raw_tol1em6")
BUCKET_FIELDS = ("N", "SDPE_pct", "SDPE_lo", "SDPE_hi", "MAPE_pct", "bias_pct")
SERIES_FIELDS = ("n_buckets", "n_decreasing_steps", "n_steps",
                 "strict_monotone_decreasing", "first", "last",
                 "ratio_last_first", "final_step", "falls_at_final_bucket")

INT_FIELDS = frozenset({"N", "n_probes", "grid_points", "n_pairs", "n_buckets",
                        "n_decreasing_steps", "n_steps", "n_viol_deployed",
                        "n_viol_raw", "n_viol_deployed_tol1em6",
                        "n_viol_raw_tol1em6"})
BOOL_FIELDS = frozenset({"strict_monotone_decreasing", "falls_at_final_bucket"})


def _slug(s: str) -> str:
    """Same slug rule as ``build_paper_tables._slug`` -- keys must agree."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _store(numbers: dict, key: str, field: str, value, collisions: list) -> None:
    if key in numbers:
        collisions.append(key)
    if field in BOOL_FIELDS:
        numbers[key] = int(bool(value)) if str(value).strip() != "" else None
        return
    try:
        fv = float(value)
    except (TypeError, ValueError):
        numbers[key] = None
        return
    if not math.isfinite(fv):
        numbers[key] = None
    elif field in INT_FIELDS:
        numbers[key] = int(fv)
    else:
        numbers[key] = round(fv, 9)


def consistency_bank_numbers(probe=None, buckets=None, series=None) -> dict[str, object]:
    """Bank entries for every cell of the three consistency artifacts."""
    numbers: dict[str, object] = {}
    collisions: list[str] = []

    if probe is not None and len(probe):
        for _, r in probe.iterrows():
            head = f"{KEY_PREFIX}probe_{_slug(r['model'])}_{_slug(r['swept'])}"
            for f in PROBE_FIELDS:
                if f in r.index:
                    _store(numbers, f"{head}_{_slug(f)}", f, r[f], collisions)

    if buckets is not None and len(buckets):
        for _, r in buckets.iterrows():
            head = (f"{KEY_PREFIX}nd_{_slug(r['axis'])}_{_slug(r['bucket_slug'])}"
                    f"_{_slug(r['model'])}")
            for f in BUCKET_FIELDS:
                if f in r.index:
                    _store(numbers, f"{head}_{_slug(f)}", f, r[f], collisions)

    if series is not None and len(series):
        for _, r in series.iterrows():
            head = f"{KEY_PREFIX}series_{_slug(r['axis'])}_{_slug(r['model'])}"
            for f in SERIES_FIELDS:
                if f in r.index:
                    _store(numbers, f"{head}_{_slug(f)}", f, r[f], collisions)

    if collisions:
        raise SystemExit(
            "consistency_bank: key collision -- two rows slug to the same key, so "
            f"one would silently overwrite the other: {sorted(set(collisions))[:5]}")
    return numbers


def load_results():
    """The three CSVs, or ``(None, None, None)`` if the study has not been run."""
    import pandas as pd

    if not (P_PROBE.exists() and P_BUCKETS.exists() and P_SERIES.exists()):
        return None, None, None
    return (pd.read_csv(P_PROBE), pd.read_csv(P_BUCKETS), pd.read_csv(P_SERIES))


def merge_into_bank(probe, buckets, series) -> dict[str, object]:
    """Merge the consistency keys into ``paper_numbers.json`` in place."""
    if not P_BANK.exists():
        raise SystemExit(f"missing {P_BANK}; run build_paper_tables.py first")
    added = consistency_bank_numbers(probe, buckets, series)
    bank: dict[str, object] = json.loads(P_BANK.read_text(encoding="utf-8"))

    stale = [k for k in bank if k.startswith(KEY_PREFIX) and k not in added]
    changed = sum(1 for k, v in added.items() if k in bank and bank[k] != v)
    fresh = sum(1 for k in added if k not in bank)
    for k in stale:
        del bank[k]
    bank.update(added)
    P_BANK.write_text(json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8")

    print(f"consistency_bank: {len(probe)}+{len(buckets)}+{len(series)} rows -> "
          f"{len(added)} keys ({fresh} new, {changed} updated, "
          f"{len(stale)} stale removed); bank now holds {len(bank)} keys")
    return added


def main() -> int:
    probe, buckets, series = load_results()
    if probe is None:
        raise SystemExit(f"missing {P_PROBE}; run consistency_31f.py all first")
    merge_into_bank(probe, buckets, series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
