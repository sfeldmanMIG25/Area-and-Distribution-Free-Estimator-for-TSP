"""Bank the facts about Çavdar–Sokol's Eq. (21) finite-``n`` correction.

Why this exists
---------------
Appendix~\\ref{app:bench_details} now states four things about that correction
that are ours to defend rather than the source's to assert:

* how much of the 2D benchmark falls *below* the range Eq. (21) was fitted over,
  so the correction is evaluated at its lower endpoint instead of extrapolated;
* the size of the discontinuity we accept at the upper endpoint by refusing to
  extrapolate there either, and the ratio at that endpoint;
* what Eq. (21) would return if it *were* extrapolated upward, which is the
  reason we do not.

Each is a number in the manuscript, so each needs a generator. Both endpoint
figures come from :meth:`CavdarSokol.correction_ratio` itself, so a change to
the implemented constants moves the prose; the coverage counts come from the
scored benchmark.

Output
------
``tables/cavdar_correction.json``   the sidecar this module owns
Bank keys                           ``cavdar_corr_{n_min,n_max}``
                                    ``cavdar_corr_ratio_at_{n_min,n_max}``
                                    ``cavdar_corr_step_at_n_max_pct``
                                    ``cavdar_corr_ratio_extrap_5000``
                                    ``cavdar_corr_2d_{n_total,n_below_min,pct_below_min}``

CLI
---
    python paper_tooling/cavdar_correction_bank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from classical_region_estimators import CavdarSokol  # noqa: E402

OUT = Path(__file__).resolve().parent / "tables"
SIDECAR = OUT / "cavdar_correction.json"
P_2D_GT = (ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints"
           / "base_ground_truth_2d.csv")

#: The value Eq. (21) reaches if extrapolated to a node count far above the
#: range it was fitted over. Quoted in the appendix as the reason the
#: implementation refuses to extrapolate upward.
EXTRAP_PROBE_N = 5000


def _ratio_unbounded(n: int) -> float:
    """Eq. (21) with the fitted-range clamp removed. Never used to predict."""
    import math
    c = CavdarSokol
    return c.CORR_A * math.exp(c.CORR_B * n) - c.CORR_C * math.exp(-c.CORR_D * n)


def compute() -> dict[str, object]:
    c = CavdarSokol
    r_lo = c.correction_ratio(c.CORR_N_MIN)
    r_hi = c.correction_ratio(c.CORR_N_MAX)
    keys: dict[str, object] = {
        "cavdar_corr_n_min": int(c.CORR_N_MIN),
        "cavdar_corr_n_max": int(c.CORR_N_MAX),
        "cavdar_corr_ratio_at_n_min": round(r_lo, 6),
        "cavdar_corr_ratio_at_n_max": round(r_hi, 6),
        # The estimate is DIVIDED by the ratio, so dropping the correction at
        # n_max multiplies the prediction by r_hi. The step is that in percent.
        "cavdar_corr_step_at_n_max_pct": round(100.0 * (1.0 / r_hi - 1.0), 6),
        "cavdar_corr_ratio_extrap_5000": round(_ratio_unbounded(EXTRAP_PROBE_N), 6),
    }
    gt = pd.read_csv(P_2D_GT)
    n = gt["n_customers"].astype(int)
    keys["cavdar_corr_2d_n_total"] = int(len(n))
    keys["cavdar_corr_2d_n_below_min"] = int((n < c.CORR_N_MIN).sum())
    keys["cavdar_corr_2d_pct_below_min"] = round(
        100.0 * float((n < c.CORR_N_MIN).mean()), 6)
    return keys


def cavdar_correction_bank_numbers() -> dict[str, object]:
    """This module's bank keys, from the sidecar it writes.

    ``build_paper_tables.main()`` rewrites ``paper_numbers.json`` wholesale, so
    without a hook of this shape a full table rebuild deletes every key written
    here and the claims pointing at them go unverifiable.
    """
    if not SIDECAR.exists():
        return {}
    return json.loads(SIDECAR.read_text(encoding="utf-8"))


def main() -> int:
    keys = compute()
    OUT.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(keys, indent=2, sort_keys=True), encoding="utf-8")
    for k, v in sorted(keys.items()):
        print(f"  {k} = {v}")
    bank = OUT / "paper_numbers.json"
    if bank.exists():
        numbers = json.loads(bank.read_text(encoding="utf-8"))
        numbers.update(keys)
        bank.write_text(json.dumps(numbers, indent=2, sort_keys=True), encoding="utf-8")
        print(f"merged {len(keys)} keys into {bank.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
