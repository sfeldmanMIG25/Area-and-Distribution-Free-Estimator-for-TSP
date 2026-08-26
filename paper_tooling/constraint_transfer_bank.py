"""Fold the constraint-transfer measurement into the number bank.

``paper_tooling/constraint_transfer.py`` prices the one thing the manuscript's
shipping argument rests on and never measured: what the monotone constraint
costs the rival when it is *added*, rather than what removing it buys the
production model.  None of it reaches ``tables/paper_numbers.json`` on its own,
so no prose claim could point at a bank key and ``check_prose_numbers.py`` could
not verify one.  This module closes that gap on the pattern of
``paper_tooling/consistency_bank.py``.

Key shapes
----------
``ctrans_strata_<arm>_<stratum>_<metric>_<stat>``   median / min / max over seeds
``ctrans_strata_<arm>_s<seed>_<stratum>_<metric>``  one per-seed cell
``ctrans_probe_<arm>_<swept>_<stat>``               median / min / max over seeds
``ctrans_probe_<arm>_s<seed>_<swept>_<field>``      one per-seed probe cell
``ctrans_cost_<block>_<stratum>_<metric>_<stat>``   paired-by-seed constraint cost
``ctrans_paired_<stratum>_<field>``                 supporting Wilcoxon on TSPLIB
``ctrans_verdict_<clause>``                         the rule, clause by clause
``ctrans_gate_<name>``                              reproduction / falsification

Arm names appear in these keys for the same reason they do in ``cons_*``: the
measurement is a *named* comparison, so a claim written against
``ctrans_strata_v4_mono_32f_bench2d_mape_median`` must break if the arm behind
that name changes.  A protocol-shaped key would survive a silent substitution.

Booleans are banked as 0/1 integers because the bank's JSON consumers compare
numerically, and the pass/fail clauses of the pre-registered rule are exactly
the places where the claim in the prose is the boolean itself.

The paired constraint cost is banked **paired by seed** (median over the seven
per-seed differences), not as a difference of medians.  The two are not equal
and the paired one is the quantity with a defensible band: both arms of a pair
saw the identical bagging draw.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
P_BANK = TABLES / "paper_numbers.json"

P_PERSEED = HERE / "constraint_transfer_perseed.csv"
P_PROBE = HERE / "constraint_transfer_probe.csv"
P_VERDICT = HERE / "constraint_transfer_verdict.json"
P_REPRO = HERE / "constraint_transfer_repro.json"

KEY_PREFIX = "ctrans_"

STRATA = ("nd_test", "bench2d", "tsplib_euc2d", "tsplib_noneuc", "augment")
METRICS = ("mape", "sdpe", "bias")
PROBE_FIELDS = ("pct_nonincr_deployed", "n_viol_deployed", "viol_max_deployed",
                "pct_nonincr_raw", "n_viol_raw", "viol_max_raw")
COST_PAIRS = {"32f": ("V4_mono_32f", "V4_unc_32f"),
              "31f": ("V4_mono_31f", "V4_unc_31f")}

INT_FIELDS = frozenset({"n", "n_seeds", "n_pairs", "n_probes", "grid_points",
                        "n_viol_deployed", "n_viol_raw", "n_v4_mono_better",
                        "n_seeds_cost_positive"})
BOOL_FIELDS = frozenset({"pass", "retains", "meets_materiality",
                         "bands_disjoint", "decisive", "gate"})


def _slug(s: str) -> str:
    """Same slug rule as ``build_paper_tables._slug`` -- keys must agree."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _store(numbers: dict, key: str, value, collisions: list,
           kind: str = "float") -> None:
    if key in numbers:
        collisions.append(key)
    if kind == "bool":
        numbers[key] = int(bool(value))
        return
    try:
        fv = float(value)
    except (TypeError, ValueError):
        numbers[key] = None
        return
    if not math.isfinite(fv):
        numbers[key] = None
    elif kind == "int":
        numbers[key] = int(fv)
    else:
        numbers[key] = round(fv, 9)


def _kind(field: str) -> str:
    if field in BOOL_FIELDS:
        return "bool"
    return "int" if field in INT_FIELDS else "float"


def _band(v: np.ndarray) -> dict[str, float]:
    return {"median": float(np.median(v)), "min": float(np.min(v)),
            "max": float(np.max(v))}


def constraint_transfer_numbers(perseed, probe, verdict, repro) -> dict[str, object]:
    numbers: dict[str, object] = {}
    col: list[str] = []

    # ---- strata: per-seed cells, then the band -----------------------------
    fit = perseed[perseed.seed != "shipped"]
    for _, r in perseed.iterrows():
        tag = ("shipped" if r["seed"] == "shipped" else f"s{r['seed']}")
        head = f"{KEY_PREFIX}strata_{_slug(r['arm'])}_{tag}_{_slug(r['stratum'])}"
        _store(numbers, f"{head}_n", r["n"], col, "int")
        for m in METRICS:
            _store(numbers, f"{head}_{m}", r[m], col)
    for arm, g0 in fit.groupby("arm"):
        for stratum, g in g0.groupby("stratum"):
            head = f"{KEY_PREFIX}strata_{_slug(arm)}_{_slug(stratum)}"
            _store(numbers, f"{head}_n_seeds", len(g), col, "int")
            for m in METRICS:
                for stat, v in _band(g[m].to_numpy(float)).items():
                    _store(numbers, f"{head}_{m}_{stat}", v, col)

    # ---- probe: per-seed cells, then the band ------------------------------
    for _, r in probe.iterrows():
        tag = r["seed"] if r["seed"] in ("shipped", "control") else f"s{r['seed']}"
        head = f"{KEY_PREFIX}probe_{_slug(r['arm'])}_{tag}_{_slug(r['swept'])}"
        _store(numbers, f"{head}_n_pairs", r["n_pairs"], col, "int")
        for f in PROBE_FIELDS:
            _store(numbers, f"{head}_{_slug(f)}", r[f], col, _kind(f))
    seeded = probe[~probe.seed.isin(["shipped", "control"])]
    for arm, g0 in seeded.groupby("arm"):
        for swept, g in g0.groupby("swept"):
            head = f"{KEY_PREFIX}probe_{_slug(arm)}_{_slug(swept)}"
            _store(numbers, f"{head}_n_seeds", len(g), col, "int")
            for f in PROBE_FIELDS:
                for stat, v in _band(g[f].to_numpy(float)).items():
                    _store(numbers, f"{head}_{_slug(f)}_{stat}", v, col)

    # ---- paired-by-seed constraint cost ------------------------------------
    for block, (mono, unc) in COST_PAIRS.items():
        for stratum in STRATA:
            for m in ("mape", "sdpe"):
                a = (fit[(fit.arm == mono) & (fit.stratum == stratum)]
                     .set_index("seed")[m])
                b = (fit[(fit.arm == unc) & (fit.stratum == stratum)]
                     .set_index("seed")[m])
                d = (a - b).reindex(sorted(a.index)).to_numpy(float)
                head = f"{KEY_PREFIX}cost_{block}_{_slug(stratum)}_{m}"
                for stat, v in _band(d).items():
                    _store(numbers, f"{head}_{stat}", v, col)
                _store(numbers, f"{head}_n_seeds_cost_positive",
                       int((d > 0).sum()), col, "int")

    # ---- the rule, clause by clause ----------------------------------------
    _store(numbers, f"{KEY_PREFIX}verdict_materiality_pp",
           verdict["materiality_pp"], col)
    _store(numbers, f"{KEY_PREFIX}verdict_n_seeds", len(verdict["seeds"]), col, "int")
    _store(numbers, f"{KEY_PREFIX}verdict_probe_pass",
           verdict["probe_clause"]["pass"], col, "bool")
    for axis, v in verdict["probe_clause"]["by_axis"].items():
        head = f"{KEY_PREFIX}verdict_probe_{_slug(axis)}"
        for stat in ("median", "min", "max"):
            _store(numbers, f"{head}_{stat}", v[stat], col)
        _store(numbers, f"{head}_pass", v["pass"], col, "bool")
    for stratum, sv in verdict["accuracy_clause"].items():
        head = f"{KEY_PREFIX}verdict_{_slug(stratum)}"
        _store(numbers, f"{head}_decisive", sv["decisive"], col, "bool")
        _store(numbers, f"{head}_retained",
               sv["state"] == "RETAINED", col, "bool")
        _store(numbers, f"{head}_lost", sv["state"] == "LOST", col, "bool")
        for m, cell in sv["by_metric"].items():
            for f in ("margin_pp_v4_better", "meets_materiality",
                      "bands_disjoint", "retains"):
                _store(numbers, f"{head}_{m}_{_slug(f)}", cell[f], col, _kind(f))
    for outcome in ("SHIP-V4", "ARGUMENT-COMPLETE", "FRONTIER"):
        _store(numbers, f"{KEY_PREFIX}verdict_outcome_is_{_slug(outcome)}",
               verdict["outcome"] == outcome, col, "bool")

    # ---- supporting paired tests -------------------------------------------
    for stratum, p in verdict["supporting_paired_tests"].items():
        head = f"{KEY_PREFIX}paired_{_slug(stratum)}"
        for f in ("n", "mean_diff_pp_neg_favours_v4_mono", "n_v4_mono_better",
                  "wilcoxon_p"):
            _store(numbers, f"{head}_{_slug(f)}", p[f], col, _kind(f))

    # ---- symmetric arm: the four decompositions ----------------------------
    for m, byst in verdict["symmetric_arm"].items():
        for stratum, cell in byst.items():
            head = f"{KEY_PREFIX}sym_{_slug(stratum)}_{m}"
            for f in ("constraint_cost_32f_pp", "constraint_cost_31f_pp",
                      "feature_value_unconstrained_pp",
                      "feature_value_constrained_pp",
                      "residual_mono31f_vs_gart_pp"):
                _store(numbers, f"{head}_{_slug(f)}", cell[f], col)

    # ---- gates --------------------------------------------------------------
    _store(numbers, f"{KEY_PREFIX}gate_v4_bit_identical",
           repro["gate_v4_bit_identical"], col, "bool")
    _store(numbers, f"{KEY_PREFIX}gate_gart_bit_identical_diagnostic",
           repro["diagnostic_gart_bit_identical"], col, "bool")
    ctrl = probe[probe.seed == "control"]
    for _, r in ctrl.iterrows():
        _store(numbers,
               f"{KEY_PREFIX}gate_falsification_{_slug(r['arm'])}_{_slug(r['swept'])}",
               r["pct_nonincr_deployed"], col)

    if col:
        raise SystemExit("constraint_transfer_bank: key collision -- two rows slug "
                         f"to the same key: {sorted(set(col))[:5]}")
    return numbers


def load_results():
    for p in (P_PERSEED, P_PROBE, P_VERDICT, P_REPRO):
        if not p.exists():
            raise SystemExit(f"missing {p}; run constraint_transfer.py all first")
    return (pd.read_csv(P_PERSEED, dtype={"seed": str}),
            pd.read_csv(P_PROBE, dtype={"seed": str}),
            json.loads(P_VERDICT.read_text(encoding="utf-8")),
            json.loads(P_REPRO.read_text(encoding="utf-8")))


def carried_numbers() -> dict[str, object]:
    """The keys a full table rebuild must carry, or ``{}`` before the study runs.

    ``build_paper_tables.py`` writes ``paper_numbers.json`` wholesale, so a
    rebuild deletes every key it does not itself emit.  This is the tolerant
    entry point it calls: a fresh clone rebuilds tables without these keys
    rather than failing, and a repository where the study has run keeps them.
    """
    if not all(p.exists() for p in (P_PERSEED, P_PROBE, P_VERDICT, P_REPRO)):
        return {}
    return constraint_transfer_numbers(*load_results())


def merge_into_bank(perseed, probe, verdict, repro) -> dict[str, object]:
    if not P_BANK.exists():
        raise SystemExit(f"missing {P_BANK}; run build_paper_tables.py first")
    added = constraint_transfer_numbers(perseed, probe, verdict, repro)
    bank: dict[str, object] = json.loads(P_BANK.read_text(encoding="utf-8"))

    stale = [k for k in bank if k.startswith(KEY_PREFIX) and k not in added]
    changed = sum(1 for k, v in added.items() if k in bank and bank[k] != v)
    fresh = sum(1 for k in added if k not in bank)
    for k in stale:
        del bank[k]
    bank.update(added)
    P_BANK.write_text(json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8")

    print(f"constraint_transfer_bank: {len(perseed)}+{len(probe)} rows -> "
          f"{len(added)} keys ({fresh} new, {changed} updated, "
          f"{len(stale)} stale removed); bank now holds {len(bank)} keys")
    return added


def main() -> int:
    merge_into_bank(*load_results())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
