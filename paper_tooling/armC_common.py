"""Arm C harness: the de-contaminated augmented arm, under the fixed nuisance
protocol of ``paper_tooling/decontaminated_arm_protocol.md``.

Arm C = arm A minus the 24 d=2 ``lattice`` augmentation rows (protocol s1),
trained under the fixed concatenation order and 7-seed nuisance protocol (s2),
judged on gates 1-11 (s3).

Everything about the recipe other than the removed rows and the fixed row order
is inherited unchanged from ``armA_verify_common``: the same 31 frozen features,
the same logit(alpha-1) target, the same monotone constraint -1 on
``n_customers`` and ``dimension``, the same V3 frozen hyperparameters, and the
same fitter. Augmentation rows go to TRAIN ONLY.

Writes nothing except ``armC_*`` artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import armA_verify_common as K  # noqa: E402

# ---------------------------------------------------------------- protocol s2
#: The 7 named seeds. Fixed in the protocol before arm C existed; chosen to
#: include both extremes already observed for arm A (42 best, 1729 worst).
SEEDS: tuple[int, ...] = (42, 1, 7, 97, 123, 1729, 2024)

#: Row-order permutations at the median seed, protocol s2 bullet 4.
N_ROW_PERMS = 8

#: The de-contamination rule, protocol s1: drop every augmentation row whose
#: generator is provably the evaluation set's own generator. For d=2 the
#: augmentation's ``gen_lattice`` and the 2D benchmark's ``generate_grid``
#: place the same site lattice, so every ``AUG_lattice-d2-*`` row is dropped.
D2_LATTICE_PREFIX = "AUG_lattice-d2-"
EXPECT_REMOVED = 24
EXPECT_RETAINED = 850

# ---------------------------------------------------------------- references
#: Frozen incumbent, as scored by THIS harness (armA_verify_repro.json).
FROZEN_REF = {
    "nd_test_mape": 0.620114641624737,
    "nd_test_sdpe": 0.9881129828885468,
    "bench2d_mape": 2.903749242765989,
    "b2_grid_mspe": 7.111779785317605,
    "b2_grid_mape": 7.111779785317605,
    "b2_line_noise_mape": 10.750295974830022,
    "b2_cluster_mape": 2.1757887161999587,
    "b2_others_mape": 1.719090387492025,
    "linenoise_slope": 0.3621117687188572,
    "tsplib_euc2d_mape": 2.5561741354541048,
    "tsplib_euc2d_sdpe": 2.955086628622842,
    "tsplib_noneuc_mape": 3.3464150371593533,
    "tsplib_noneuc_sdpe": 3.8974530161134324,
    "tsplib_noneuc_n": 22,
}

#: Production control numbers the gates file supplies for gate 8 (2D classes).
PROD_2D = {"grid": 7.1121, "line_noise": 10.7522, "cluster": 2.1847, "others": 1.7191}

#: Gate thresholds, verbatim from the protocol.
G1_MAPE_MAX = 0.6401
G1_SDPE_MAX = 1.0081
G6_SLOPE_MIN = 0.70
G6_MIN_SEEDS = 6
G7_GRID_MSPE_MAX = 4.0
G8_MAX_DELTA = 0.15
G9_NONEUC_MAPE = 3.3441          # protocol literal
G9_NONEUC_SDPE = 3.8931          # protocol literal
G9_MIN_COVERAGE = 22
G10_SD_RATIO_MAX = 1.174
G10_ROBUST_RATIO_MAX = 1.00
G11_RECAL_FROZEN_SLOPE = 0.4023826203247474   # armA_verify_oracle_constant.csv
G11_MIN_MARGIN = 0.25
G5_COST_RATIO_MAX = 1.35


# ---------------------------------------------------------------- data
def d2_lattice_mask(aug: pd.DataFrame) -> pd.Series:
    return aug["instance_name"].astype(str).str.startswith(D2_LATTICE_PREFIX)


def armC_aug(aug: pd.DataFrame) -> pd.DataFrame:
    """The 850 retained augmentation rows. Counts verified, not assumed."""
    drop = d2_lattice_mask(aug)
    n_drop = int(drop.sum())
    keep = aug[~drop.to_numpy()]
    if n_drop != EXPECT_REMOVED or len(keep) != EXPECT_RETAINED:
        raise SystemExit(
            f"de-contamination row counts wrong: removed {n_drop} "
            f"(expected {EXPECT_REMOVED}), retained {len(keep)} "
            f"(expected {EXPECT_RETAINED})"
        )
    return keep.reset_index(drop=True)


def protocol_train(corpus_train: pd.DataFrame, aug_rows: pd.DataFrame | None,
                   row_perm: int | None = None) -> pd.DataFrame:
    """Training frame in the order fixed in writing by protocol s2.

    Corpus rows first, then augmentation rows; each block sorted ascending by
    ``instance_name``; no shuffling of the concatenation. ``row_perm`` is used
    only by the row-order sensitivity study, which deliberately breaks the
    fixed order to measure how much the reported statistics owe to it.
    """
    a = corpus_train.sort_values("instance_name", kind="mergesort")
    blocks = [a]
    if aug_rows is not None and len(aug_rows):
        b = aug_rows.sort_values("instance_name", kind="mergesort")
        blocks.append(b[[c for c in a.columns if c in b.columns]])
    out = pd.concat(blocks, ignore_index=True, sort=False)
    if row_perm is not None:
        rng = np.random.default_rng(row_perm)
        out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    return out


# ---------------------------------------------------------------- scoring
def _mad(v: np.ndarray) -> float:
    """Median absolute deviation about the median, unscaled."""
    v = np.asarray(v, dtype=float)
    return float(np.median(np.abs(v - np.median(v))))


def _trimmed_sd(v: np.ndarray, prop: float = 0.10) -> float:
    """SD of the symmetrically 10%-trimmed sample."""
    v = np.sort(np.asarray(v, dtype=float))
    k = int(np.floor(len(v) * prop))
    t = v[k:len(v) - k] if len(v) - 2 * k > 1 else v
    return float(np.std(t, ddof=1))


def extended_metrics(model, feats, C: pd.DataFrame) -> dict:
    """``armA_verify_common.metrics`` plus what gates 9-11 need.

    Adds robust scale statistics on the TSPLIB EUC_2D signed error vector, the
    non-Euclidean SDPE and coverage, and the per-instance predicted tour lengths
    for every scored stratum (the input ``ood_harness`` wants).
    """
    m = K.metrics(model, feats, C, want_per_instance=True)
    scored = m.pop("_scored")

    e = scored["tsplib_euc2d"]["err_pct"].to_numpy(dtype=float)
    m["tsplib_euc2d_mad"] = _mad(e)
    m["tsplib_euc2d_trimsd10"] = _trimmed_sd(e, 0.10)
    m["tsplib_euc2d_iqr"] = float(np.subtract(*np.percentile(e, [75, 25])))
    m["tsplib_noneuc_coverage"] = int(len(scored["tsplib_noneuc"]))

    preds_all: dict[str, float] = {}
    for st, ok in scored.items():
        preds_all.update(dict(zip(ok["instance"].astype(str),
                                  ok["pred_cost"].astype(float))))
    m["_preds_all"] = preds_all
    m["_euc2d_err"] = dict(zip(scored["tsplib_euc2d"]["instance"].astype(str),
                               scored["tsplib_euc2d"]["err_pct"].astype(float)))
    m["_b2"] = scored["bench2d"][["instance", "generator", "n_customers",
                                  "pred_alpha", "pred_cost", "true_cost",
                                  "mst_total_length", "err_pct"]].copy()
    return m


#: Scalar metrics carried through every run record.
SCALARS = [
    "nd_test_mape", "nd_test_sdpe", "nd_test_mspe",
    "bench2d_mape", "bench2d_sdpe",
    "b2_grid_mape", "b2_grid_mspe", "b2_line_noise_mape", "b2_line_noise_mspe",
    "b2_cluster_mape", "b2_others_mape",
    "linenoise_slope", "linenoise_slope_n", "grid_slope",
    "tsplib_euc2d_mape", "tsplib_euc2d_sdpe", "tsplib_euc2d_mad",
    "tsplib_euc2d_trimsd10", "tsplib_euc2d_iqr",
    "tsplib_noneuc_mape", "tsplib_noneuc_sdpe", "tsplib_noneuc_coverage",
    "augment_mape", "trees", "best_iter",
]


def band(vals) -> dict:
    """Median with the full min-max band. The only reporting form s2 allows."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if not v.size:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "k": 0}
    return {"median": float(np.median(v)), "lo": float(v.min()),
            "hi": float(v.max()), "k": int(v.size)}


def fmt_band(b: dict, nd: int = 4) -> str:
    return f"{b['median']:.{nd}f} [{b['lo']:.{nd}f}, {b['hi']:.{nd}f}]"
