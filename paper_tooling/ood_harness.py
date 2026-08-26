"""Out-of-distribution referee harness for TSP tour-length estimators.

This module is the *judge*, not a candidate. It is written before any
candidate improvement exists and is deliberately not tuned to one: it takes an
arbitrary map ``{instance_name: predicted_tour_length}`` and returns a
multiplicity-controlled significance verdict against every baseline that is
defined on the same instances, over a fixed four-stratum out-of-distribution
suite assembled from artifacts already on disk.

Suite (see :func:`load_suite`)
------------------------------
``tsplib_euc2d``  78 real TSPLIB95 EUC_2D instances (external provenance).
``tsplib_noneuc`` 23 real TSPLIB95 non-EUC_2D instances, structurally
                  screened (GEO / EXPLICIT / CEIL_2D / ATT).
``bench2d``       2,580 synthetic 2D diverse-generator instances, grouped into
                  the five paper classes.
``augment``       874 synthetic novel-geometry instances whose generators
                  (collinear, curve, subspace, lattice, hexlattice, filament)
                  appear nowhere in the training corpus.

Statistical core
----------------
:func:`compare` never collapses to one number. It reports the mean paired
difference in absolute percent error with a paired bootstrap CI, a Wilcoxon
signed-rank p, a paired sign-flip permutation p, the median paired difference,
win/loss/tie counts, Cohen's d_z and the rank-biserial correlation.

:func:`evaluate_candidate` builds the whole verdict table and applies
Holm-Bonferroni and Benjamini-Hochberg across the *entire* family of tests in
that verdict. A significance claim must survive adjustment.

Nothing here writes outside ``paper_tooling/``.

CLI
---
    python paper_tooling/ood_harness.py --validate
    python paper_tooling/ood_harness.py --suite
    python paper_tooling/ood_harness.py --power
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import (  # noqa: E402
    EMBEDDING_DONOR,
    GART,
    PREDECESSOR,
    PRODUCTION_BOOSTER,
    PRODUCTION_SIDECAR,
)

OUT_DIR = Path(__file__).resolve().parent
AUGMENT_BASELINE_CACHE = OUT_DIR / "ood_augment_baselines.csv"

TSPLIB_CSV = REPO_ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
BENCH2D_DIR = REPO_ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints"
BENCH2D_TRUTH = BENCH2D_DIR / "base_ground_truth_2d.csv"
AUGMENT_FEATURES = OUT_DIR / "augment_features_v3.csv"
#: 31st feature, held separately because the cached feature table predates it.
AUGMENT_GREEDY = OUT_DIR / "augment_greedy_nn.csv"
AUGMENT_RECORDS = [
    REPO_ROOT / "augment" / "full_records.json",
    REPO_ROOT / "augment" / "batch2_records.json",
]

SEED = 42
N_BOOTSTRAP = 1_000
N_PERMUTATION = 10_000

#: Name used everywhere for the shipped model / reference arm.
#:
#: This was pinned to the predecessor for as long as the candidate-arm studies
#: were open, because moving it would have moved their baseline mid-review.
#: Those studies are closed, so it now follows the registry.
#:
#: ``SHIPPED`` is a *reference* role, not an *anchor* role, and the two are not
#: interchangeable.  The anchor -- the row a stratum reads instance-level truth
#: from -- must cover every instance, and the production model does not: it
#: records no ``true_cost`` and no ``mst_length`` on the instances it declines.
#: Pointing the anchor here would put a NaN truth into ``tsplib_noneuc`` for
#: ``si1032`` while ``n_instances`` still said 23, silently dropping the
#: instance from every comparison on that stratum.  The anchor therefore stays
#: on ``EMBEDDING_DONOR``; see ``_load_tsplib_strata``.
SHIPPED = GART

#: Slower-than-this against the shipped model gets flagged.
TIMING_FLAG_RATIO = 1.5

# Generator token -> paper class, for the 2D diverse benchmark. The instance
# grammar is TSP-<generator>-n<n>-g<G>-<i>; the clustered generator carries
# extra -c<k>-r<r> parameters before the replicate index, so the generator
# token is parsed non-greedily up to the -n<digits> field.
GENERATOR_CLASS: dict[str, str] = {
    "random": "Isotropic",
    "normal": "Isotropic",
    "triangular": "Isotropic",
    "truncated_exponential": "Isotropic",
    "squeezed_uniform": "Biased",
    "uniform_triangular": "Biased",
    "triangular_squeezed": "Biased",
    "correlated": "Biased",
    "grid": "Geometric",
    "boundary": "Geometric",
    "x_central": "Geometric",
    "clustered": "Clustered",
    "line_noise": "LineNoise",
}
GENERATOR_RE = r"^TSP-(.+?)-n\d+-"

# Checkpoint file -> canonical baseline name for the 2D benchmark. The
# checkpoint stores GART 1.0 under the bare name "GART"; everything else is
# renamed only where the TSPLIB run uses a different label, so one baseline
# name means the same estimator on every stratum.
BENCH2D_RENAME = {"GART": "GART_1.0"}

#: Canonical baseline set required by the verdict, in report order. Not every
#: one is defined on every stratum; coverage is recorded, never imputed.
BASELINE_ORDER = [
    "Asymptotic_MST",      # 1.1253 * L_MST at d=2 (published beta_TSP/beta_MST)
    "Calibrated_MST_d",    # rho_hat(d)   from calibrated_alpha_table.json
    "Calibrated_MST_dn",   # rho_hat(d,n) from calibrated_alpha_table.json
    "MST_Only",            # the alpha = 1 floor
    "Linear_V3",           # same-feature linear model
    "NN_V3",               # same-feature neural network
    "Interp_V3",           # same-feature interpretable model
    PREDECESSOR,           # the model the production model displaced
    "GART_1.0",            # previous generation
    "LGBM_V4",
    "BHH",                 # repaired classical region estimators below
    "Daganzo",
    "Cavdar",
    "Chien",
    "Chien_extrap",
    "Kwon",
    "Kwon_extrap",
    "Hilbert",
    "MST_Ratio",
    "Fixed_Alpha",
]


# --------------------------------------------------------------------------
# Suite definition
# --------------------------------------------------------------------------


@dataclass
class Stratum:
    """One out-of-distribution stratum with its truth, baselines and metadata."""

    key: str
    label: str
    real_world: bool
    provenance: str
    ood_rationale: str
    truth: pd.Series                      # instance -> true tour length
    mst: pd.Series                        # instance -> L_MST
    meta: pd.DataFrame                    # instance -> n_customers, dimension, group
    baselines: dict[str, pd.Series] = field(default_factory=dict)
    timings: pd.DataFrame | None = None   # model, instance, feature_time_s, inference_time_s

    @property
    def n_instances(self) -> int:
        return int(len(self.truth))

    @property
    def synthetic(self) -> bool:
        return not self.real_world

    def coverage(self) -> pd.DataFrame:
        """Per-baseline finite-prediction counts on this stratum."""
        rows = []
        for name, pred in self.baselines.items():
            aligned = pred.reindex(self.truth.index)
            n_fin = int(np.isfinite(aligned.to_numpy(dtype=float)).sum())
            rows.append(
                {
                    "stratum": self.key,
                    "baseline": name,
                    "n_stratum": self.n_instances,
                    "n_finite": n_fin,
                    "coverage_frac": n_fin / self.n_instances if self.n_instances else np.nan,
                }
            )
        return pd.DataFrame(rows)


def _pivot_predictions(df: pd.DataFrame, index: pd.Index) -> dict[str, pd.Series]:
    wide = df.pivot_table(index="instance", columns="model", values="pred_cost", aggfunc="first")
    return {str(m): wide[m].reindex(index) for m in wide.columns}


def _load_tsplib_strata() -> dict[str, Stratum]:
    from tsplib_benchmark.exclusions import filter_metric_consistent

    raw = pd.read_csv(TSPLIB_CSV)
    # The anchor supplies instance-level truth (edge_weight_type, n, true_cost,
    # L_MST), which every stratum needs on every instance. It must therefore be
    # the donor, not the reference: the production model leaves those columns
    # empty on the instances it declines, so anchoring on it would put NaN truth
    # into the non-EUC stratum for si1032 and quietly shrink every comparison
    # there from 23 pairs to 22 while still reporting n_stratum = 23.
    anchor = (
        raw[raw["model"] == EMBEDDING_DONOR]
        .drop_duplicates("instance")
        .set_index("instance")
    )

    euc_names = anchor.index[anchor["edge_weight_type"] == "EUC_2D"]
    non_names = anchor.index[anchor["edge_weight_type"] != "EUC_2D"]

    out: dict[str, Stratum] = {}
    specs = [
        (
            "tsplib_euc2d",
            "TSPLIB95 EUC_2D",
            euc_names,
            "Real-world TSPLIB95 instances, external provenance, released before "
            "this project existed.",
            "Out of distribution by provenance: hand-curated real point sets "
            "(city coordinates, drilling boards, VLSI) that no synthetic generator "
            "in the training corpus produces. Sizes reach n = 85,900, far beyond "
            "the n <= 1000 training cap.",
        ),
        (
            "tsplib_noneuc",
            "TSPLIB95 non-EUC_2D (screened)",
            non_names,
            "Real-world TSPLIB95 instances with GEO / EXPLICIT / CEIL_2D / ATT "
            "metrics, after the structural triangle-violation screen.",
            "Out of distribution by metric as well as provenance: the training "
            "corpus is entirely Euclidean-in-R^d, while these carry geodesic, "
            "pseudo-Euclidean and tabulated distance matrices reached through an "
            "MDS embedding.",
        ),
    ]
    for key, label, names, provenance, rationale in specs:
        sub = raw[raw["instance"].isin(names)]
        anchor_sub = anchor.loc[sorted(names)]
        screened = filter_metric_consistent(anchor_sub.reset_index()).set_index("instance")
        idx = pd.Index(sorted(screened.index), name="instance")
        sub = sub[sub["instance"].isin(idx)]

        meta = pd.DataFrame(
            {
                "n_customers": screened.loc[idx, "n"].astype(float),
                "dimension": 2.0,
                "group": screened.loc[idx, "edge_weight_type"].astype(str),
            },
            index=idx,
        )
        timings = sub[["model", "instance", "feature_time_s", "inference_time_s"]].copy()
        out[key] = Stratum(
            key=key,
            label=label,
            real_world=True,
            provenance=provenance,
            ood_rationale=rationale,
            truth=screened.loc[idx, "true_cost"].astype(float),
            mst=screened.loc[idx, "mst_length"].astype(float),
            meta=meta,
            baselines=_pivot_predictions(sub, idx),
            timings=timings,
        )
    return out


def _load_bench2d_stratum() -> Stratum:
    truth = pd.read_csv(BENCH2D_TRUTH)
    gen = truth["instance"].str.extract(GENERATOR_RE)[0]
    unknown = sorted(set(gen.dropna()) - set(GENERATOR_CLASS))
    if unknown or gen.isna().any():
        raise ValueError(f"unparsed 2D generators: {unknown or 'name grammar mismatch'}")
    truth = truth.assign(generator=gen, group=gen.map(GENERATOR_CLASS)).set_index("instance")
    idx = truth.index

    frames, timing_frames = [], []
    for path in sorted(BENCH2D_DIR.glob("results_*.csv")):
        d = pd.read_csv(path)
        d["model"] = d["model"].astype(str).replace(BENCH2D_RENAME)
        frames.append(d[["model", "instance", "pred_cost"]])
        timing_frames.append(d[["model", "instance", "feature_time_s", "inference_time_s"]])
    allpred = pd.concat(frames, ignore_index=True)

    meta = pd.DataFrame(
        {
            "n_customers": truth["n_customers"].astype(float),
            "dimension": truth["dimension"].astype(float),
            "group": truth["group"].astype(str),
            "subgroup": truth["generator"].astype(str),
        },
        index=idx,
    )
    return Stratum(
        key="bench2d",
        label="2D diverse-generator benchmark",
        real_world=False,
        provenance="Synthetic, 13 planar generators, exactly solved.",
        ood_rationale=(
            "Out of distribution by geometry, not by provenance: the generators "
            "overlap the training corpus but the benchmark deliberately "
            "over-samples degenerate shape regimes (near-collinear Line Noise, "
            "boundary and lattice geometry) that are rare in training, and the "
            "instances themselves are a held-out split."
        ),
        truth=truth["true_cost"].astype(float),
        mst=truth["mst_length"].astype(float),
        meta=meta,
        baselines=_pivot_predictions(allpred, idx),
        timings=pd.concat(timing_frames, ignore_index=True),
    )


def _augment_records() -> pd.DataFrame:
    rows = []
    for path in AUGMENT_RECORDS:
        batch = "batch2" if "batch2" in path.name else "batch1"
        for rec in json.loads(path.read_text(encoding="utf-8")):
            rows.append({"instance": rec["name"], "batch": batch})
    return pd.DataFrame(rows).drop_duplicates("instance").set_index("instance")


def compute_augment_baselines(force: bool = False) -> pd.DataFrame:
    """Score the model baselines on the cached augmentation features.

    The augmentation set ships features but no baseline predictions, so this
    evaluates the production model and its siblings directly from
    ``augment_features_v3.csv`` and caches the result. Coordinate-dependent
    baselines (GART 1.0, the classical region estimators) are not produced
    here; they are reported as absent rather than imputed.

    The production column is scored from ``PRODUCTION_BOOSTER``, not from a
    renamed predecessor prediction. Two things differ and both matter: the
    production booster needs a 31st feature (``greedy_nn_over_mst``, carried in
    a sidecar CSV because the cached feature table predates it), and it predicts
    in logit space, so its output goes through the sidecar's inverse transform
    with no clip rather than through ``np.clip(..., 1, 2)``.
    """
    if AUGMENT_BASELINE_CACHE.exists() and not force:
        cached = pd.read_csv(AUGMENT_BASELINE_CACHE).set_index("instance")
        if SHIPPED in cached.columns and PREDECESSOR in cached.columns:
            return cached
        # A cache written before the production model swapped holds the
        # predecessor under the old reference name and no production column.
        # Recomputing is the only honest response; reusing it would report the
        # predecessor's augmentation numbers under the production label.
        print(f"augment cache predates the current production model "
              f"({SHIPPED!r} absent); recomputing", file=sys.stderr)

    import joblib

    from baselines_calibrated import ASYMPTOTIC_RHO, CalibratedMSTRatio

    feats = pd.read_csv(AUGMENT_FEATURES).set_index("instance_name")
    feats.index.name = "instance"
    mst = feats["mst_total_length"].astype(float)
    n = feats["n_customers"].astype(int)
    d = feats["dimension"].astype(int)
    out = pd.DataFrame(index=feats.index)

    out[SHIPPED] = _augment_production_alpha(feats) * mst

    lgbm = joblib.load(REPO_ROOT / "lgbm_model_v3" / "lgbm_alpha_model_v3.joblib")
    out[PREDECESSOR] = (
        np.clip(lgbm.predict(feats[list(lgbm.feature_name_)]), 1.0, 2.0) * mst
    )

    lin = joblib.load(REPO_ROOT / "linear_model_v3" / "linear_alpha_model_v3.joblib")
    out["Linear_V3"] = np.clip(lin.predict(feats[list(lin.feature_names_in_)]), 1.0, 2.0) * mst

    out["NN_V3"] = _augment_nn_predictions(feats) * mst

    out["MST_Only"] = mst
    rho_d = CalibratedMSTRatio("d")
    rho_dn = CalibratedMSTRatio("dn")
    out["Calibrated_MST_d"] = np.array([rho_d.rho(di, ni) for di, ni in zip(d, n)]) * mst
    out["Calibrated_MST_dn"] = np.array([rho_dn.rho(di, ni) for di, ni in zip(d, n)]) * mst
    # Published only for the dimensions in ASYMPTOTIC_RHO; NaN elsewhere by
    # design, so the coverage table shows the gap instead of hiding it.
    out["Asymptotic_MST"] = [
        ASYMPTOTIC_RHO[di] * m if di in ASYMPTOTIC_RHO else np.nan for di, m in zip(d, mst)
    ]

    out.reset_index().to_csv(AUGMENT_BASELINE_CACHE, index=False)
    return out


def _augment_production_alpha(feats: pd.DataFrame) -> np.ndarray:
    """Predicted ``alpha`` from the production booster on the augmentation set.

    ``augment_features_v3.csv`` was written for the 30-feature predecessor, so
    ``greedy_nn_over_mst`` is joined in from ``augment_greedy_nn.csv``. That
    join is required to be total, and the check is specifically on the joined
    column rather than on the frame as a whole. The distinction matters: 13
    lattice and hexlattice instances have a degenerate MST whose edge lengths
    are all equal, so ``mst_edge_skew`` and ``mst_edge_kurtosis`` are genuinely
    undefined there. Those NaNs are data, and LightGBM's own missing-value
    branch is the right handling for them -- the predecessor path relied on
    exactly that. A NaN arriving from a failed join is not data, and would
    silently produce a finite prediction for an instance the model never saw.
    """
    import joblib

    sidecar = json.loads(PRODUCTION_SIDECAR.read_text(encoding="utf-8"))
    booster = joblib.load(PRODUCTION_BOOSTER)
    names = list(booster.feature_name())

    x = feats.copy()
    joined = [c for c in names if c not in x.columns]
    if joined:
        extra = pd.read_csv(AUGMENT_GREEDY).set_index("instance_name")
        extra.index.name = "instance"
        have = [c for c in joined if c in extra.columns]
        x = x.join(extra[have], how="left")
    still = [c for c in names if c not in x.columns]
    if still:
        raise SystemExit(f"augmentation features lack booster columns: {still}")
    if joined and x[joined].isna().any().any():
        bad = sorted(c for c in joined if x[c].isna().any())
        raise SystemExit(
            f"the joined production-only features are incomplete on the "
            f"augmentation set: {bad}. These come from a sidecar keyed on "
            f"instance name, so a NaN here is a failed join, not a "
            f"degenerate instance."
        )
    undefined = {c: int(x[c].isna().sum()) for c in names if x[c].isna().any()}
    if undefined:
        print(f"augment: undefined feature values passed to LightGBM's missing "
              f"branch (degenerate MSTs): {undefined}", file=sys.stderr)

    z = np.asarray(booster.predict(x[names]), dtype=float)
    if sidecar["target_transform"]["clip_after_inverse"]:
        raise SystemExit("sidecar now records a post-inverse clip; update this path")
    # alpha = 1 + sigma(z); the overflow-safe branch keeps the bound structural.
    s = np.empty_like(z)
    pos = z >= 0
    s[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    s[~pos] = e / (1.0 + e)
    return 1.0 + s


def _augment_nn_predictions(feats: pd.DataFrame) -> np.ndarray:
    """Batch the shipped neural network over cached features (alpha = 1 + head)."""
    import torch

    import nn_est_alpha_v3.estimator_v3 as nn_mod

    sys.modules["__main__"].StableV3Scaler = nn_mod.StableV3Scaler  # pickle namespace
    import joblib

    scaler = joblib.load(REPO_ROOT / "nn_est_alpha_v3" / "nn_alpha_v3_scaler.joblib")
    ckpt = torch.load(REPO_ROOT / "nn_est_alpha_v3" / "nn_alpha_v3_model.pt", map_location="cpu")
    bp = ckpt["params"]
    net = nn_mod.TSP_Leap_Inference(
        ckpt["input_dim"], bp["hidden_dim"], bp["num_blocks"], dropout=bp.get("dropout", 0.1)
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    x = feats[list(ckpt["features"])].to_numpy(dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1e38, neginf=-1e38)
    with torch.no_grad():
        head = net(torch.tensor(scaler.transform(x), dtype=torch.float32)).squeeze(-1).numpy()
    return 1.0 + head


def _load_augment_stratum() -> Stratum:
    feats = pd.read_csv(AUGMENT_FEATURES).set_index("instance_name")
    feats.index.name = "instance"
    idx = pd.Index(sorted(feats.index), name="instance")
    feats = feats.loc[idx]
    batch = _augment_records().reindex(idx)["batch"].fillna("batch1")

    preds = compute_augment_baselines().reindex(idx)
    meta = pd.DataFrame(
        {
            "n_customers": feats["n_customers"].astype(float),
            "dimension": feats["dimension"].astype(float),
            "group": feats["family"].astype(str),
            "subgroup": batch.astype(str) + "/" + feats["group"].astype(str),
            "batch": batch.astype(str),
        },
        index=idx,
    )
    return Stratum(
        key="augment",
        label="Novel-geometry augmentation set",
        real_world=False,
        provenance="Synthetic, 6 generator families absent from training, exactly solved.",
        ood_rationale=(
            "The strongest generator-level shift available locally: collinear, "
            "curve, subspace, lattice, hexlattice and filament generators appear "
            "nowhere in the training corpus, and they populate the "
            "large-n / large-alpha region that the training split leaves empty. "
            "Batch 2 is kept separable because it is known to sit below the "
            "benchmark's alpha locus and is therefore much easier than batch 1."
        ),
        truth=feats["optimal_cost"].astype(float),
        mst=feats["mst_total_length"].astype(float),
        meta=meta,
        baselines={c: preds[c] for c in preds.columns},
        timings=None,
    )


_SUITE_CACHE: dict[str, Stratum] | None = None


def load_suite(refresh: bool = False) -> dict[str, Stratum]:
    """Build (and memoise) the four-stratum out-of-distribution suite."""
    global _SUITE_CACHE
    if _SUITE_CACHE is not None and not refresh:
        return _SUITE_CACHE
    suite = _load_tsplib_strata()
    suite["bench2d"] = _load_bench2d_stratum()
    suite["augment"] = _load_augment_stratum()
    _SUITE_CACHE = suite
    return suite


def suite_summary() -> pd.DataFrame:
    """One row per stratum: counts, provenance and the OOD rationale."""
    rows = []
    for s in load_suite().values():
        rows.append(
            {
                "stratum": s.key,
                "label": s.label,
                "n_instances": s.n_instances,
                "real_world": s.real_world,
                "n_baselines_defined": sum(
                    bool(np.isfinite(v.to_numpy(dtype=float)).any()) for v in s.baselines.values()
                ),
                "provenance": s.provenance,
                "ood_rationale": s.ood_rationale,
            }
        )
    return pd.DataFrame(rows)


def coverage_table() -> pd.DataFrame:
    """Finite-prediction coverage for every baseline on every stratum."""
    return pd.concat([s.coverage() for s in load_suite().values()], ignore_index=True)


# --------------------------------------------------------------------------
# Statistical core
# --------------------------------------------------------------------------


def absolute_percent_error(pred: pd.Series, truth: pd.Series) -> pd.Series:
    """Per-instance |pred - truth| / truth * 100, aligned on ``truth``'s index."""
    p = pd.to_numeric(pred.reindex(truth.index), errors="coerce").astype(float)
    t = truth.astype(float)
    ape = (p - t).abs() / t * 100.0
    return ape.where(np.isfinite(ape.to_numpy(dtype=float)))


def _rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation from signed ranks of ``diff``."""
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    total = ranks.sum()
    return float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / total)


def compare(
    pred_a: pd.Series,
    pred_b: pd.Series,
    truth: pd.Series,
    name_a: str,
    name_b: str,
    n_boot: int = N_BOOTSTRAP,
    n_perm: int = N_PERMUTATION,
    seed: int = SEED,
) -> dict:
    """Paired comparison of two estimators on the instances where both are finite.

    ``diff = APE(a) - APE(b)``, so a negative mean favours ``a``. Every quantity
    is computed on the identical paired subset; nothing is imputed, and the
    subset size is reported so unequal-coverage comparisons are visible.
    """
    ape_a = absolute_percent_error(pred_a, truth)
    ape_b = absolute_percent_error(pred_b, truth)
    both = ape_a.notna() & ape_b.notna()
    a = ape_a[both].to_numpy(dtype=float)
    b = ape_b[both].to_numpy(dtype=float)
    n = a.size

    rec = {
        "model_a": name_a,
        "model_b": name_b,
        "n_pairs": n,
        "n_finite_a": int(ape_a.notna().sum()),
        "n_finite_b": int(ape_b.notna().sum()),
        "n_stratum": int(len(truth)),
        "equal_coverage": bool(ape_a.notna().sum() == ape_b.notna().sum() == n),
        "mape_a": float(np.mean(a)) if n else np.nan,
        "mape_b": float(np.mean(b)) if n else np.nan,
    }
    if n < 2:
        rec.update(
            {
                k: np.nan
                for k in (
                    "mean_diff", "ci_low", "ci_high", "median_diff", "sd_diff",
                    "p_wilcoxon", "p_permutation", "cohens_dz", "rank_biserial",
                )
            }
        )
        rec.update({"wins_a": 0, "wins_b": 0, "ties": 0})
        return rec

    diff = a - b
    rng = np.random.default_rng(seed)

    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[boot_idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    if np.allclose(diff, 0.0):
        p_w = 1.0
    else:
        p_w = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)

    obs = float(np.mean(diff))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = (signs * diff).mean(axis=1)
    p_perm = float((1 + np.sum(np.abs(perm_means) >= abs(obs) - 1e-15)) / (n_perm + 1))

    sd = float(np.std(diff, ddof=1))
    rec.update(
        {
            "mean_diff": obs,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "median_diff": float(np.median(diff)),
            "sd_diff": sd,
            "p_wilcoxon": p_w,
            "p_permutation": p_perm,
            "cohens_dz": float(obs / sd) if sd > 0 else np.nan,
            "rank_biserial": _rank_biserial(diff),
            "wins_a": int(np.sum(diff < 0)),
            "wins_b": int(np.sum(diff > 0)),
            "ties": int(np.sum(diff == 0)),
        }
    )
    return rec


# --------------------------------------------------------------------------
# Dispersion (scale) comparison
# --------------------------------------------------------------------------


def signed_percent_error(pred: pd.Series, truth: pd.Series) -> pd.Series:
    """Per-instance (pred - truth) / truth * 100.

    The paper's SDPE is the standard deviation of *this* quantity, not of the
    absolute percent error. On TSPLIB EUC_2D it gives 2.9441 for the production
    model against 4.7522 for the asymptotic ratio. (The 3.4237 this docstring
    used to quote as the production figure is the predecessor's, and the
    "28% lower dispersion" it was attached to was computed from that pair.)
    """
    p = pd.to_numeric(pred.reindex(truth.index), errors="coerce").astype(float)
    t = truth.astype(float)
    spe = (p - t) / t * 100.0
    return spe.where(np.isfinite(spe.to_numpy(dtype=float)))


def _pitman_morgan(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Pitman-Morgan test of equal variances in paired samples.

    Under H0 ``var(x) = var(y)``, the sum ``u = x + y`` and difference
    ``v = x - y`` are uncorrelated because ``cov(u, v) = var(x) - var(y)``.
    The test is therefore a t-test of ``r(u, v) = 0`` on ``n - 2`` degrees of
    freedom. This is the correct paired analogue of an F or Levene test and is
    materially more powerful than treating the two error vectors as
    independent samples, because the errors are strongly correlated across
    instances.

    Returns ``(r_uv, t, p)``.
    """
    n = x.size
    if n < 4:
        return np.nan, np.nan, np.nan
    u, v = x + y, x - y
    if np.std(u) == 0 or np.std(v) == 0:
        return np.nan, np.nan, np.nan
    r = float(np.corrcoef(u, v)[0, 1])
    r = min(max(r, -1 + 1e-15), 1 - 1e-15)
    t = r * np.sqrt((n - 2) / (1 - r * r))
    p = float(2 * stats.t.sf(abs(t), n - 2))
    return r, float(t), p


def compare_dispersion(
    pred_a: pd.Series,
    pred_b: pd.Series,
    truth: pd.Series,
    name_a: str,
    name_b: str,
    metric: str = "signed",
    n_boot: int = N_BOOTSTRAP,
    n_perm: int = N_PERMUTATION,
    seed: int = SEED,
) -> dict:
    """Paired comparison of error *dispersion* rather than error location.

    At n = 78 the mean test cannot resolve anything below about 1.1 APE points,
    but the scale comparison can, and dispersion is where the shipped model's
    real TSPLIB advantage lives. Reports, on the instances where both are
    finite:

    * ``sdpe_a`` / ``sdpe_b`` and their ratio, with a paired bootstrap CI on
      the ratio (the interpretable form: a CI excluding 1 is the claim)
    * ``p_pitman_morgan`` -- the paired-samples variance test, primary
    * ``p_brown_forsythe`` -- Levene on deviations from the median, robust but
      treats the two vectors as independent, so it is the conservative
      cross-check rather than the headline
    * ``p_swap_permutation`` -- distribution-free: each vector is centred on
      its own mean, then within-instance pairs are swapped at random, which
      isolates scale from location
    * ``r_xy`` -- the correlation between the two error vectors, which is what
      makes the paired test worth running

    ``metric='signed'`` (default) matches the paper's SDPE. ``metric='absolute'``
    scores the dispersion of absolute percent error instead.
    """
    fn = {"signed": signed_percent_error, "absolute": absolute_percent_error}[metric]
    ea, eb = fn(pred_a, truth), fn(pred_b, truth)
    both = ea.notna() & eb.notna()
    x = ea[both].to_numpy(dtype=float)
    y = eb[both].to_numpy(dtype=float)
    n = x.size

    rec = {
        "model_a": name_a,
        "model_b": name_b,
        "metric": metric,
        "n_pairs": n,
        "n_finite_a": int(ea.notna().sum()),
        "n_finite_b": int(eb.notna().sum()),
        "equal_coverage": bool(ea.notna().sum() == eb.notna().sum() == n),
    }
    nan_keys = (
        "sdpe_a", "sdpe_b", "sd_ratio", "pct_lower", "ratio_ci_low", "ratio_ci_high",
        "r_xy", "r_uv", "t_pitman_morgan", "p_pitman_morgan", "p_brown_forsythe",
        "p_swap_permutation", "mde_sd_ratio", "mde_pct_lower", "detectable",
    )
    if n < 4:
        rec.update(dict.fromkeys(nan_keys, np.nan))
        return rec

    sd_a = float(np.std(x, ddof=1))
    sd_b = float(np.std(y, ddof=1))
    ratio = sd_a / sd_b if sd_b > 0 else np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bx, by = x[idx], y[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_ratio = bx.std(axis=1, ddof=1) / by.std(axis=1, ddof=1)
    boot_ratio = boot_ratio[np.isfinite(boot_ratio)]
    lo, hi = (np.percentile(boot_ratio, [2.5, 97.5]) if boot_ratio.size else (np.nan, np.nan))

    r_uv, t_pm, p_pm = _pitman_morgan(x, y)
    p_bf = float(stats.levene(x, y, center="median").pvalue)

    xc, yc = x - x.mean(), y - y.mean()
    obs = abs(np.log(sd_a / sd_b)) if sd_b > 0 and sd_a > 0 else np.nan
    swap = rng.random((n_perm, n)) < 0.5
    px = np.where(swap, yc, xc)
    py = np.where(swap, xc, yc)
    perm = np.abs(np.log(px.std(axis=1, ddof=1) / py.std(axis=1, ddof=1)))
    p_swap = float((1 + np.sum(perm >= obs - 1e-15)) / (n_perm + 1))

    r_xy = float(np.corrcoef(x, y)[0, 1]) if n > 2 else np.nan
    mde_ratio = variance_ratio_mde(n, r_xy)

    rec.update(
        {
            "sdpe_a": sd_a,
            "sdpe_b": sd_b,
            "sd_ratio": ratio,
            "pct_lower": 100.0 * (1.0 - ratio) if np.isfinite(ratio) else np.nan,
            "ratio_ci_low": float(lo),
            "ratio_ci_high": float(hi),
            "r_xy": r_xy,
            "r_uv": r_uv,
            "t_pitman_morgan": t_pm,
            "p_pitman_morgan": p_pm,
            "p_brown_forsythe": p_bf,
            "p_swap_permutation": p_swap,
            "mde_sd_ratio": mde_ratio,
            "mde_pct_lower": 100.0 * (1.0 - mde_ratio) if np.isfinite(mde_ratio) else np.nan,
            "detectable": bool(np.isfinite(mde_ratio) and np.isfinite(ratio) and ratio <= mde_ratio),
        }
    )
    return rec


def _pitman_morgan_power(k: float, r: float, n: int, alpha: float) -> float:
    """Power of the paired variance test at SD ratio ``k`` and pairing corr ``r``.

    With ``sigma_y = 1`` and ``sigma_x = k``, the population correlation of the
    sum and difference is ``(k^2 - 1) / sqrt((k^2 + 2rk + 1)(k^2 - 2rk + 1))``;
    power follows from the Fisher-z test of that correlation against zero.
    """
    if n < 5 or not np.isfinite(k) or k <= 0:
        return np.nan
    den2 = (k * k + 2 * r * k + 1) * (k * k - 2 * r * k + 1)
    if den2 <= 0:
        return np.nan
    rho_uv = (k * k - 1.0) / np.sqrt(den2)
    rho_uv = min(max(rho_uv, -1 + 1e-12), 1 - 1e-12)
    ncp = np.arctanh(rho_uv) * np.sqrt(n - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z - ncp) + stats.norm.cdf(-z - ncp))


def variance_ratio_mde(
    n: int, r_xy: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Largest SD ratio (< 1) the paired variance test can detect.

    Returns ``k`` such that a candidate whose SDPE is ``k`` times the
    baseline's is detected with probability ``power``. The pairing correlation
    ``r_xy`` matters a great deal: highly correlated error vectors make far
    smaller scale differences detectable than independent samples would.
    """
    if not np.isfinite(r_xy) or n < 6:
        return np.nan
    r = min(max(float(r_xy), -0.999), 0.999)
    f = lambda k: _pitman_morgan_power(k, r, n, alpha) - power  # noqa: E731
    lo, hi = 1e-4, 1.0 - 1e-9
    if not np.isfinite(f(lo)) or f(lo) < 0:
        return np.nan
    if f(hi) > 0:
        return float(hi)
    return float(optimize.brentq(f, lo, hi, xtol=1e-9))


def dispersion_verdict(
    candidates: dict,
    stratum: str = "tsplib_euc2d",
    baselines: list[str] | None = None,
    metric: str = "signed",
    n_boot: int = N_BOOTSTRAP,
    n_perm: int = N_PERMUTATION,
    seed: int = SEED,
) -> pd.DataFrame:
    """Scale comparison for one or more candidates against baselines on a stratum.

    ``candidates`` maps label to ``{instance: predicted_tour_length}``. Holm and
    BH are applied over the whole returned table, so the family is every
    candidate-by-baseline dispersion test, exactly as for the mean tests.
    """
    s = load_suite()[stratum]
    names = baselines or [b for b in BASELINE_ORDER if b in s.baselines]
    rows = []
    for label, preds in candidates.items():
        cand = pd.Series(dict(preds), dtype=float).reindex(s.truth.index)
        for name in names:
            if name not in s.baselines or name == label:
                continue  # a self-comparison is degenerate, not a test
            rec = compare_dispersion(
                cand, s.baselines[name], s.truth, label, name,
                metric=metric, n_boot=n_boot, n_perm=n_perm, seed=seed,
            )
            rec["stratum"] = stratum
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = adjust_family(out, "p_pitman_morgan")
    out["p_bf_holm"] = holm_bonferroni(out["p_brown_forsythe"])
    out["p_swap_holm"] = holm_bonferroni(out["p_swap_permutation"])

    # The MDE above is the unadjusted floor. Inside a family, the threshold a
    # test must beat is tighter, so recompute the detectable SD ratio at each
    # row's own Holm threshold. This is the floor that actually binds.
    p = out["p_pitman_morgan"].to_numpy(dtype=float)
    holm_alpha, mde_holm = [], []
    for i in range(len(out)):
        a_i = holm_threshold(np.delete(p, i))
        holm_alpha.append(a_i)
        mde_holm.append(
            variance_ratio_mde(int(out["n_pairs"].iloc[i]), out["r_xy"].iloc[i], alpha=a_i)
            if a_i > 0 else np.nan
        )
    out["holm_alpha"] = holm_alpha
    out["mde_sd_ratio_holm"] = mde_holm
    out["mde_pct_lower_holm"] = 100.0 * (1.0 - np.asarray(mde_holm, dtype=float))
    out["detectable_holm"] = (
        out["sd_ratio"].to_numpy(dtype=float) <= np.asarray(mde_holm, dtype=float)
    )
    front = ["stratum", "model_a", "model_b", "n_pairs", "sdpe_a", "sdpe_b", "sd_ratio"]
    return out[front + [c for c in out.columns if c not in front]]


# --------------------------------------------------------------------------
# Alternative paired statistics (variance reduction on the mean scale)
# --------------------------------------------------------------------------

#: What each paired statistic actually measures, in the terms a practitioner
#: would have to be told. A statistic that reaches significance but names no
#: quantity anyone cares about is not a result.
STATISTIC_MEANING: dict[str, str] = {
    "ape_diff": (
        "Percentage points of absolute percent error saved per instance. "
        "Directly interpretable and the quantity the paper already reports."
    ),
    "logratio_diff_signed": (
        "NOT AN ACCURACY STATISTIC. log(pred_a/truth) - log(pred_b/truth) "
        "cancels the truth term exactly and equals log(pred_a/pred_b), so it "
        "measures only whether one model predicts systematically larger than "
        "the other. It is significant whenever two models differ in bias, "
        "including when the more biased one is more accurate. Reported here "
        "solely to document why it must not be used."
    ),
    "abs_logratio_diff": (
        "Reduction in absolute log accuracy ratio, x100. The multiplicative "
        "analogue of the APE difference: symmetric in over- and "
        "under-prediction, and for errors below roughly 20% it is numerically "
        "close to percentage points of relative error. Interpretable, with a "
        "sentence of explanation."
    ),
    "sq_pct_diff": (
        "Reduction in squared percent error. Units are squared percentage "
        "points, dominated by the few worst instances. Harder to explain and "
        "usually higher-variance, not lower."
    ),
    "sq_logratio_diff": (
        "Reduction in squared log accuracy ratio, x100 scale. Same "
        "tail-dominance problem as the squared percent error."
    ),
    "ape_blocked": (
        "The same percentage points of absolute percent error, but tested "
        "within instance-size blocks so that any systematic size effect is "
        "removed from the error term. Identical estimand to ape_diff, so it "
        "costs nothing in interpretability; it only helps if the paired "
        "difference genuinely varies with n."
    ),
}

#: Size-block edges, matching the calibrated table's n buckets.
N_BLOCK_EDGES = [0, 10, 50, 100, 200, 500, 10**9]


def paired_statistics(
    pred_a: pd.Series, pred_b: pd.Series, truth: pd.Series
) -> dict[str, np.ndarray]:
    """Per-instance paired differences under each candidate loss, aligned.

    All are oriented so that negative favours ``pred_a``.
    """
    p = pd.to_numeric(pred_a.reindex(truth.index), errors="coerce").astype(float)
    q = pd.to_numeric(pred_b.reindex(truth.index), errors="coerce").astype(float)
    t = truth.astype(float)
    ok = np.isfinite(p) & np.isfinite(q) & np.isfinite(t) & (t > 0) & (p > 0) & (q > 0)
    idx = truth.index[ok.to_numpy()]
    p, q, t = p[ok].to_numpy(), q[ok].to_numpy(), t[ok].to_numpy()

    ea, eb = 100.0 * (p - t) / t, 100.0 * (q - t) / t
    la, lb = 100.0 * np.log(p / t), 100.0 * np.log(q / t)
    return {
        "index": idx,
        "ape_diff": np.abs(ea) - np.abs(eb),
        "logratio_diff_signed": la - lb,
        "abs_logratio_diff": np.abs(la) - np.abs(lb),
        "sq_pct_diff": ea**2 - eb**2,
        "sq_logratio_diff": la**2 - lb**2,
        "ape_blocked": np.abs(ea) - np.abs(eb),
    }


def variance_components(diff: np.ndarray, blocks: np.ndarray) -> dict:
    """ANOVA split of paired-difference variance into between- and within-block.

    ``between_frac`` is the ceiling on what blocking can buy: if it is near
    zero the paired difference does not vary with the blocking variable and a
    blocked test has the same error variance as the unblocked one.
    """
    n = diff.size
    labels, inv = np.unique(blocks, return_inverse=True)
    k = labels.size
    if n < k + 2 or k < 2:
        return {"k_blocks": k, "sd_within": np.nan, "between_frac": np.nan, "df": np.nan}
    grand = diff.mean()
    means = np.array([diff[inv == j].mean() for j in range(k)])
    counts = np.array([(inv == j).sum() for j in range(k)])
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_within = float(np.sum((diff - means[inv]) ** 2))
    ss_total = ss_between + ss_within
    return {
        "k_blocks": int(k),
        "sd_within": float(np.sqrt(ss_within / (n - k))),
        "between_frac": float(ss_between / ss_total) if ss_total > 0 else np.nan,
        "df": int(n - k),
    }


def scan_paired_statistics(
    candidates: dict,
    stratum: str = "tsplib_euc2d",
    baselines: list[str] | None = None,
    kinds: list[str] | None = None,
    n_boot: int = N_BOOTSTRAP,
    n_perm: int = N_PERMUTATION,
    seed: int = SEED,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Does a variance-reduced paired statistic make the mean claim decidable?

    For a paired test at fixed ``n``, significance depends only on the
    standardised effect ``d_z = mean / sd`` of the paired differences, so
    changing the loss helps only if it raises ``|d_z|``. The scan therefore
    reports ``dz``, the ``dz_needed`` for significance at this ``n``, and their
    ratio, alongside the usual test output. Raw MDEs are in each statistic's
    own units and are not comparable across rows.

    Searching several statistics for one that reaches significance is itself
    multiple testing, so Holm and BH run over the whole scan.
    """
    s = load_suite()[stratum]
    names = baselines or [b for b in BASELINE_ORDER if b in s.baselines]
    kinds = kinds or [k for k in STATISTIC_MEANING]
    blocks_all = pd.cut(
        s.meta["n_customers"], bins=N_BLOCK_EDGES, labels=False, include_lowest=True
    )
    rows = []
    for label, preds in candidates.items():
        cand = pd.Series(dict(preds), dtype=float).reindex(s.truth.index)
        for name in names:
            if name not in s.baselines or name == label:
                continue
            stats_map = paired_statistics(cand, s.baselines[name], s.truth)
            inst = stats_map["index"]
            blocks = blocks_all.reindex(inst).to_numpy()
            for kind in kinds:
                d = np.asarray(stats_map[kind], dtype=float)
                n = d.size
                if n < 4:
                    continue
                rng = np.random.default_rng(seed)
                mean = float(d.mean())
                sd = float(d.std(ddof=1))

                vc = variance_components(d, blocks)
                if kind == "ape_blocked":
                    sd_eff = vc["sd_within"]
                    df = vc["df"]
                else:
                    sd_eff, df = sd, n - 1
                se = sd_eff / np.sqrt(n) if np.isfinite(sd_eff) else np.nan
                t_stat = mean / se if se and np.isfinite(se) and se > 0 else np.nan
                p_t = float(2 * stats.t.sf(abs(t_stat), df)) if np.isfinite(t_stat) else np.nan

                boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
                perm = (signs * d).mean(axis=1)
                p_perm = float((1 + np.sum(np.abs(perm) >= abs(mean) - 1e-15)) / (n_perm + 1))
                p_w = (
                    float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
                    if not np.allclose(d, 0.0) else 1.0
                )

                dz = mean / sd_eff if sd_eff and np.isfinite(sd_eff) and sd_eff > 0 else np.nan
                dz_needed = float(stats.t.ppf(1 - alpha / 2, df) / np.sqrt(n))
                rows.append(
                    {
                        "stratum": stratum,
                        "model_a": label,
                        "model_b": name,
                        "statistic": kind,
                        "n_pairs": n,
                        "mean_diff": mean,
                        "sd_diff": sd,
                        "sd_effective": sd_eff,
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "dz": dz,
                        "dz_needed": dz_needed,
                        "dz_shortfall_x": abs(dz) / dz_needed if np.isfinite(dz) else np.nan,
                        "mde_own_units": min_detectable_difference(sd_eff, n, alpha),
                        "p_t": p_t,
                        "p_wilcoxon": p_w,
                        "p_permutation": p_perm,
                        "n_needed": _n_for_delta(abs(mean), sd_eff, alpha, 0.80),
                        "block_between_frac": vc["between_frac"],
                        "k_blocks": vc["k_blocks"],
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = adjust_family(out, "p_wilcoxon")
    out["sig_raw"] = out["p_wilcoxon"] < alpha
    out["sig_holm"] = out["p_holm"] < alpha
    return out


# --------------------------------------------------------------------------
# Multiplicity control
# --------------------------------------------------------------------------


def holm_bonferroni(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error rate)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    out = np.full(m, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="mergesort")]
    k = order.size
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (k - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (false discovery rate)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    out = np.full(m, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="mergesort")]
    k = order.size
    running = 1.0
    for rank in range(k - 1, -1, -1):
        i = order[rank]
        running = min(running, k / (rank + 1) * p[i])
        out[i] = min(1.0, running)
    return out


def adjust_family(df: pd.DataFrame, p_col: str = "p_wilcoxon", prefix: str = "") -> pd.DataFrame:
    """Attach Holm and BH adjusted p-values over every row of ``df``.

    The family is the whole table handed in. Callers must not subset first and
    then adjust; that would understate the family size.
    """
    out = df.copy()
    out[f"{prefix}p_holm"] = holm_bonferroni(out[p_col])
    out[f"{prefix}p_bh"] = benjamini_hochberg(out[p_col])
    out[f"{prefix}family_size"] = int(np.isfinite(pd.to_numeric(out[p_col])).sum())
    return out


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


def _paired_t_power(delta: float, sd: float, n: int, alpha: float) -> float:
    """Two-sided paired t-test power via the non-central t distribution.

    ``scipy.stats.nct`` returns NaN for large non-centralities, so the normal
    approximation takes over once the exact routine loses precision. Both agree
    to well under a percentage point wherever the exact form is evaluable.
    """
    if n < 2 or sd <= 0 or not np.isfinite(delta):
        return np.nan
    df = n - 1
    ncp = abs(delta) * np.sqrt(n) / sd
    crit = stats.t.ppf(1 - alpha / 2, df)
    with np.errstate(all="ignore"):
        p = float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))
    if not np.isfinite(p):
        z = stats.norm.ppf(1 - alpha / 2)
        p = float(stats.norm.sf(z - ncp) + stats.norm.cdf(-z - ncp))
    return min(1.0, max(0.0, p))


def min_detectable_difference(
    sd_diff: float, n: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest mean paired difference a two-sided paired t-test can find.

    Solved exactly against the non-central t distribution rather than the
    normal approximation, which is optimistic at these sample sizes.
    """
    if not np.isfinite(sd_diff) or sd_diff <= 0 or n < 2:
        return np.nan
    f = lambda d: _paired_t_power(d, sd_diff, n, alpha) - power  # noqa: E731
    hi = sd_diff
    for _ in range(60):
        if f(hi) > 0:
            break
        hi *= 2
    else:
        return np.nan
    return float(optimize.brentq(f, 1e-12, hi, xtol=1e-10))


def power_analysis(
    reference: str = SHIPPED,
    baselines: list[str] | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """Detection floor per stratum and baseline, from observed dispersion.

    ``mde_ape_points`` is the smallest true mean paired difference in absolute
    percent error that the stratum's sample size can detect at the requested
    power. ``mde_wilcoxon`` rescales it by the asymptotic relative efficiency
    of the signed-rank test under normality (0.955), which is the test actually
    reported. ``observed_diff`` is the gap we in fact see, so
    ``detectable`` says whether the stratum can ever resolve it.
    """
    rows = []
    for s in load_suite().values():
        if reference not in s.baselines:
            continue
        ref = s.baselines[reference]
        names = baselines or [b for b in BASELINE_ORDER if b in s.baselines]
        for name in names:
            if name == reference or name not in s.baselines:
                continue
            ape_r = absolute_percent_error(ref, s.truth)
            ape_b = absolute_percent_error(s.baselines[name], s.truth)
            both = ape_r.notna() & ape_b.notna()
            n = int(both.sum())
            if n < 2:
                continue
            diff = (ape_r[both] - ape_b[both]).to_numpy(dtype=float)
            sd = float(np.std(diff, ddof=1))
            mde = min_detectable_difference(sd, n, alpha, power)
            obs = float(np.mean(diff))
            rows.append(
                {
                    "stratum": s.key,
                    "reference": reference,
                    "baseline": name,
                    "n_pairs": n,
                    "sd_paired_diff": sd,
                    "observed_diff": obs,
                    "mde_ape_points": mde,
                    "mde_wilcoxon": mde / np.sqrt(0.955) if np.isfinite(mde) else np.nan,
                    "detectable": bool(np.isfinite(mde) and abs(obs) >= mde),
                    "n_needed_for_observed": _n_for_delta(abs(obs), sd, alpha, power),
                }
            )
    return pd.DataFrame(rows)


def holm_threshold(other_pvals, alpha: float = 0.05) -> float:
    """Largest p-value that still survives Holm given the rest of the family.

    Holm is step-down, so a test's effective threshold depends on how many
    other tests are rejected before it. The guaranteed worst case is
    ``alpha / m``, but if most of the family is overwhelmingly significant the
    real threshold is much looser. This bisects for the exact value against
    the observed p-values of the other ``m - 1`` tests.
    """
    others = np.asarray([p for p in np.asarray(other_pvals, dtype=float) if np.isfinite(p)])
    m = others.size + 1

    def survives(p: float) -> bool:
        adj = holm_bonferroni(np.concatenate([[p], others]))
        return bool(adj[0] < alpha)

    if not survives(alpha / m):
        return 0.0
    lo, hi = alpha / m, alpha
    if survives(hi):
        return float(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if survives(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def plan_mean_target(
    stratum: str = "tsplib_euc2d",
    baseline: str = "Asymptotic_MST",
    reference: str = SHIPPED,
    n_target: int = 220,
    # None -> take it from the verdict this function already computes below.
    # It was hardcoded to 59, the family size before the production model
    # displaced the predecessor into the baseline roster; the guaranteed Holm
    # floor alpha/m depends on it, so a stale value understates the threshold.
    family_size: int | None = None,
    power: float = 0.80,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """What a larger real-world stratum would buy, stated as a target number.

    Holds the dispersion of the paired differences and the baseline's MAPE at
    their observed values and asks: at ``n_target`` instances, inside a family
    of ``family_size`` tests, how large must the candidate's mean advantage be,
    and what candidate MAPE does that imply? Reports three alpha levels --
    unadjusted, the guaranteed Holm floor ``alpha / m``, and the empirical Holm
    threshold implied by the shipped model's own verdict.
    """
    s = load_suite()[stratum]
    ape_r = absolute_percent_error(s.baselines[reference], s.truth)
    ape_b = absolute_percent_error(s.baselines[baseline], s.truth)
    both = ape_r.notna() & ape_b.notna()
    diff = (ape_r[both] - ape_b[both]).to_numpy(dtype=float)
    sd = float(np.std(diff, ddof=1))
    base_mape = float(ape_b[both].mean())
    n_now = int(both.sum())

    verdict = evaluate_candidate(shipped_predictions(), reference, n_boot=200, n_perm=1000)
    p_all = verdict.comparisons["p_wilcoxon"].to_numpy(dtype=float)
    mask = ~(
        (verdict.comparisons["stratum"] == stratum)
        & (verdict.comparisons["model_b"] == baseline)
    ).to_numpy()
    a_emp = holm_threshold(p_all[mask], alpha)
    if family_size is None:
        family_size = verdict.family_size

    levels = [
        ("unadjusted", alpha),
        (f"Holm guaranteed (alpha/{family_size})", alpha / family_size),
        ("Holm empirical (this family)", a_emp),
    ]
    rows = []
    for n in sorted({n_now, n_target}):
        for label, a in levels:
            mde = min_detectable_difference(sd, n, a, power) if a > 0 else np.nan
            rows.append(
                {
                    "stratum": stratum,
                    "baseline": baseline,
                    "n_instances": n,
                    "alpha_level": label,
                    "alpha": a,
                    "sd_paired_diff": sd,
                    "baseline_mape": base_mape,
                    "required_mean_gain": mde,
                    "required_candidate_mape": base_mape - mde if np.isfinite(mde) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _n_for_delta(delta: float, sd: float, alpha: float, power: float) -> float:
    """Smallest paired sample size that detects ``delta`` at ``power``.

    Exact integer bisection on the non-central t power function. An earlier
    version walked a geometric grid (n -> ceil(1.25 n) + 1) and returned the
    first grid point that cleared the target, which overstated the answer by up
    to 25%: for the TSPLIB EUC_2D case it returned 1,384 when the true minimum
    is 1,155. Power is monotone in n, so bisection is both exact and cheap.
    """
    if not np.isfinite(delta) or delta <= 0 or sd <= 0:
        return np.nan
    if _paired_t_power(delta, sd, 3, alpha) >= power:
        return 3.0
    hi = 4
    while hi < 100_000_000:
        if _paired_t_power(delta, sd, hi, alpha) >= power:
            break
        hi *= 2
    else:
        return np.nan
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _paired_t_power(delta, sd, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return float(hi)


# --------------------------------------------------------------------------
# Estimation-time parity
# --------------------------------------------------------------------------


def timing_parity(
    candidate_times: pd.DataFrame,
    label: str,
    reference: str = SHIPPED,
    machine_loaded: bool = True,
    flag_ratio: float = TIMING_FLAG_RATIO,
) -> pd.DataFrame:
    """Median and p90 wall clock per stratum, against the shipped model.

    ``candidate_times`` needs columns ``instance``, ``feature_time_s`` and
    ``inference_time_s``; total time is their sum. ``machine_loaded`` is
    recorded on every row because a timing measured on a loaded machine does
    not support a parity claim, whatever the ratio says.
    """
    need = {"instance", "feature_time_s", "inference_time_s"}
    missing = need - set(candidate_times.columns)
    if missing:
        raise ValueError(f"candidate_times missing columns: {sorted(missing)}")
    cand = candidate_times.copy()
    cand["total_s"] = cand["feature_time_s"].astype(float) + cand["inference_time_s"].astype(float)
    cand = cand.set_index("instance")["total_s"]

    rows = []
    for s in load_suite().values():
        c = cand.reindex(s.truth.index).dropna()
        if c.empty:
            continue
        row = {
            "stratum": s.key,
            "candidate": label,
            "n_timed": int(c.size),
            "median_s": float(np.median(c)),
            "p90_s": float(np.percentile(c, 90)),
            "machine_loaded": bool(machine_loaded),
        }
        ref = _reference_times(s, reference)
        if ref is not None:
            r = ref.reindex(c.index).dropna()
            common = c.reindex(r.index).dropna()
            row["ref_median_s"] = float(np.median(r)) if r.size else np.nan
            row["ref_p90_s"] = float(np.percentile(r, 90)) if r.size else np.nan
            row["ratio_median"] = (
                row["median_s"] / row["ref_median_s"] if row.get("ref_median_s") else np.nan
            )
            row["ratio_p90"] = (
                row["p90_s"] / row["ref_p90_s"] if row.get("ref_p90_s") else np.nan
            )
            row["n_ref_common"] = int(common.size)
            row["flag_slower"] = bool(
                np.isfinite(row["ratio_median"]) and row["ratio_median"] > flag_ratio
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _reference_times(s: Stratum, reference: str) -> pd.Series | None:
    if s.timings is None:
        return None
    t = s.timings[s.timings["model"] == reference]
    if t.empty:
        return None
    tot = t["feature_time_s"].astype(float) + t["inference_time_s"].astype(float)
    return pd.Series(tot.to_numpy(), index=t["instance"].to_numpy()).groupby(level=0).first()


def measure_machine_load(sample_s: float = 0.4) -> dict:
    """Best-effort snapshot of machine load, to stamp on a timing report."""
    info: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import psutil

        info["cpu_percent"] = float(psutil.cpu_percent(interval=sample_s))
        info["load_ok"] = info["cpu_percent"] < 25.0
    except Exception:
        info["cpu_percent"] = None
        info["load_ok"] = None
        info["note"] = "psutil unavailable; machine load must be asserted by the operator"
    return info


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


@dataclass
class Verdict:
    label: str
    suite: pd.DataFrame
    coverage: pd.DataFrame
    comparisons: pd.DataFrame
    by_group: pd.DataFrame
    power: pd.DataFrame
    family_size: int
    timing: pd.DataFrame | None = None

    def significant(self, adjust: str = "holm", alpha: float = 0.05) -> pd.DataFrame:
        """Rows where the candidate wins and survives the requested adjustment."""
        col = {"holm": "p_holm", "bh": "p_bh", "raw": "p_wilcoxon"}[adjust]
        c = self.comparisons
        return c[(c["mean_diff"] < 0) & (c[col] < alpha)]

    def losses(self, adjust: str = "holm", alpha: float = 0.05) -> pd.DataFrame:
        """Rows where a baseline beats the candidate and survives adjustment."""
        col = {"holm": "p_holm", "bh": "p_bh", "raw": "p_wilcoxon"}[adjust]
        c = self.comparisons
        return c[(c["mean_diff"] > 0) & (c[col] < alpha)]

    def headline(self) -> str:
        s = self.significant()
        return (
            f"{self.label}: {len(s)}/{self.family_size} comparisons are wins "
            f"significant at Holm-adjusted alpha=0.05."
        )


def evaluate_candidate(
    predictions_by_instance,
    label: str,
    baselines: list[str] | None = None,
    strata: list[str] | None = None,
    candidate_times: pd.DataFrame | None = None,
    machine_loaded: bool = True,
    n_boot: int = N_BOOTSTRAP,
    n_perm: int = N_PERMUTATION,
    seed: int = SEED,
) -> Verdict:
    """Judge one candidate against every baseline on every stratum.

    ``predictions_by_instance`` maps instance name to predicted tour length
    (not alpha). Instances the candidate does not cover are simply absent; each
    comparison then runs on the instances where the candidate and that one
    baseline are both finite, and the coverage columns make the restriction
    visible. The full set of tests produced here is the multiplicity family.
    """
    pred = pd.Series(dict(predictions_by_instance), dtype=float)
    suite = load_suite()
    keys = strata or list(suite)

    rows, group_rows = [], []
    for key in keys:
        s = suite[key]
        cand = pred.reindex(s.truth.index)
        names = baselines or [b for b in BASELINE_ORDER if b in s.baselines]
        for name in names:
            if name not in s.baselines:
                continue
            rec = compare(
                cand, s.baselines[name], s.truth, label, name,
                n_boot=n_boot, n_perm=n_perm, seed=seed,
            )
            rec["stratum"] = key
            rec["real_world"] = s.real_world
            rows.append(rec)

        ape_c = absolute_percent_error(cand, s.truth)
        for level in ("group", "subgroup", "batch"):
            if level not in s.meta.columns:
                continue
            for g, sub in s.meta.groupby(level):
                v = ape_c.reindex(sub.index).dropna()
                if v.empty:
                    continue
                group_rows.append(
                    {
                        "stratum": key,
                        "level": level,
                        "group": str(g),
                        "n": int(v.size),
                        "mape": float(v.mean()),
                        "median_ape": float(v.median()),
                        "sdpe": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                    }
                )

    comparisons = pd.DataFrame(rows)
    if not comparisons.empty:
        comparisons = adjust_family(comparisons, "p_wilcoxon")
        comparisons["p_perm_holm"] = holm_bonferroni(comparisons["p_permutation"])
        comparisons["p_perm_bh"] = benjamini_hochberg(comparisons["p_permutation"])
        front = ["stratum", "model_a", "model_b", "n_pairs", "mape_a", "mape_b", "mean_diff"]
        comparisons = comparisons[front + [c for c in comparisons.columns if c not in front]]
    family_size = int(np.isfinite(comparisons.get("p_wilcoxon", pd.Series(dtype=float))).sum())

    timing = None
    if candidate_times is not None:
        timing = timing_parity(candidate_times, label, machine_loaded=machine_loaded)

    return Verdict(
        label=label,
        suite=suite_summary(),
        coverage=coverage_table(),
        comparisons=comparisons,
        by_group=pd.DataFrame(group_rows),
        power=power_analysis(),
        family_size=family_size,
        timing=timing,
    )


def model_predictions(model: str) -> dict[str, float]:
    """One model's per-instance predictions across every stratum it covers.

    Split out from :func:`shipped_predictions` when the reference moved to the
    production model: callers that wanted *the predecessor's* predictions were
    reaching for ``shipped_predictions()`` and would silently have received the
    production model's instead, under the predecessor's label.
    """
    out: dict[str, float] = {}
    for s in load_suite().values():
        if model not in s.baselines:
            continue
        p = s.baselines[model].dropna()
        out.update({str(k): float(v) for k, v in p.items()})
    if not out:
        raise SystemExit(f"no stratum carries predictions for {model!r}")
    return out


def shipped_predictions() -> dict[str, float]:
    """The production model's per-instance predictions across the whole suite.

    Used to validate the harness against the banked numbers: feeding these in
    as a candidate must reproduce the production model's reported behaviour
    exactly.
    """
    return model_predictions(SHIPPED)


# --------------------------------------------------------------------------
# Validation and CLI
# --------------------------------------------------------------------------

#: Reference values the harness must reproduce, read from the number bank.
#:
#: They used to be literals, and they were the *predecessor's* literals
#: throughout (TSPLIB EUC_2D 3.2713, 2D 3.3348, Line Noise 11.59, ND 0.8770).
#: Every check passed while describing the arm the harness was no longer
#: judging, which is the failure mode a self-validation is supposed to catch.
#: Any figure quoted from an ``ood_*.csv`` produced before this change is the
#: predecessor's.
#:
#: Reading them from ``tables/paper_numbers.json`` instead of retyping the
#: production values does two jobs at once: the harness can no longer validate
#: against a displaced model, and agreement between the harness's own
#: computation and the table builder's becomes the check rather than an
#: assumption. The keys are built from the registry's display label, so a
#: further model swap needs no edit here either.
BANK_PATH = OUT_DIR / "tables" / "paper_numbers.json"

KNOWN_BANK_KEYS = {
    "tsplib_euc2d_mape": "tsplib_by_size_total_{p}_mape_pct",
    "tsplib_euc2d_n": "tsplib_by_size_total_{p}_n",
    "bench2d_mape": "2d_by_size_total_{p}_mape_pct",
    "bench2d_n": "2d_by_size_total_{p}_n",
    "bench2d_linenoise_mape": "2d_by_genclass_linenoise_{p}_mape_pct",
    "nd_mape": "nd_by_size_total_{p}_mape_pct",
    "tsplib_euc2d_vs_asymptotic_diff":
        "paired_tsplib_by_size_total_asymptotic_mst_ratio_mean_diff",
    "tsplib_euc2d_vs_asymptotic_p":
        "paired_tsplib_by_size_total_asymptotic_mst_ratio_wilcoxon_p",
}


def production_slug() -> str:
    """Bank-key slug for the production model's display label."""
    import re

    from model_registry import MODEL_LABELS

    return re.sub(r"[^a-z0-9]+", "_", MODEL_LABELS[GART].lower()).strip("_")


def load_known() -> dict[str, float]:
    """Production reference values, from the number bank."""
    if not BANK_PATH.exists():
        raise SystemExit(
            f"missing {BANK_PATH}; run paper_tooling/build_paper_tables.py first"
        )
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    p = production_slug()
    out, missing = {}, []
    for name, template in KNOWN_BANK_KEYS.items():
        key = template.format(p=p)
        if key not in bank or bank[key] is None:
            missing.append(key)
        else:
            out[name] = float(bank[key])
    if missing:
        raise SystemExit(
            "the number bank has no entry for these production values, so the "
            f"harness cannot validate itself: {missing}"
        )
    return out


def validate(verbose: bool = True) -> pd.DataFrame:
    """Run the production model through the harness and check the banked numbers."""
    known = load_known()
    v = evaluate_candidate(shipped_predictions(), SHIPPED)
    c = v.comparisons
    euc = c[c["stratum"] == "tsplib_euc2d"]
    asym = euc[euc["model_b"] == "Asymptotic_MST"].iloc[0]
    b2d = c[c["stratum"] == "bench2d"].iloc[0]
    ln = v.by_group[(v.by_group["stratum"] == "bench2d") & (v.by_group["group"] == "LineNoise")]

    checks = [
        ("TSPLIB EUC_2D MAPE", known["tsplib_euc2d_mape"], asym["mape_a"], 0.001),
        ("TSPLIB EUC_2D n", known["tsplib_euc2d_n"], asym["n_pairs"], 0),
        ("vs asymptotic mean diff", known["tsplib_euc2d_vs_asymptotic_diff"],
         asym["mean_diff"], 0.01),
        ("vs asymptotic Wilcoxon p", known["tsplib_euc2d_vs_asymptotic_p"],
         asym["p_wilcoxon"], 0.01),
        ("2D benchmark MAPE", known["bench2d_mape"], b2d["mape_a"], 0.001),
        ("2D benchmark n", known["bench2d_n"], b2d["n_stratum"], 0),
        ("2D LineNoise MAPE", known["bench2d_linenoise_mape"],
         float(ln["mape"].iloc[0]), 0.01),
        ("TSPLIB non-EUC n", 23, int(c[c["stratum"] == "tsplib_noneuc"]["n_stratum"].iloc[0]), 0),
        ("ND benchmark MAPE (external check)", known["nd_mape"], _nd_mape(), 0.001),
    ]
    rows = [
        {
            "check": name,
            "expected": exp,
            "observed": obs,
            "tolerance": tol,
            "pass": bool(abs(float(obs) - float(exp)) <= tol),
        }
        for name, exp, obs, tol in checks
    ]
    out = pd.DataFrame(rows)
    if verbose:
        print(out.to_string(index=False))
        print(f"\nALL PASS: {bool(out['pass'].all())}")
    return out


def candidate_predictions_from_suite(models: list[str], stratum: str) -> dict[str, dict]:
    """Wrap existing baselines as candidates, so already-scored models can be judged.

    Used to run the referee on the shipped model and on LGBM_V4 without
    re-scoring anything.
    """
    s = load_suite()[stratum]
    return {
        m: {str(k): float(v) for k, v in s.baselines[m].dropna().items()}
        for m in models
        if m in s.baselines
    }


def selftest(verbose: bool = True) -> pd.DataFrame:
    """Check the multiplicity and power routines against closed-form values.

    These are the parts of the referee no dataset can validate, so they are
    checked against hand-computable references instead.
    """
    p = np.array([0.001, 0.01, 0.03, 0.04, 0.20])
    holm = holm_bonferroni(p)          # 5p, 4p, 3p, 2p, 1p with running max
    bh = benjamini_hochberg(p)         # 5p/k with running min from the top
    rows = [
        ("Holm monotone", True, bool(np.all(np.diff(holm) >= -1e-12))),
        ("Holm first = m*p", 0.005, float(holm[0])),
        ("Holm 3rd = 3p", 0.09, float(holm[2])),
        # 0.04*2 = 0.08 < 0.09, so the step-down running max must hold 0.09.
        ("Holm running max holds", 0.09, float(holm[3])),
        ("BH monotone", True, bool(np.all(np.diff(bh) >= -1e-12))),
        ("BH largest = p", 0.20, float(bh[4])),
        ("BH k=3", 0.05, float(bh[2])),
        ("Holm capped at 1", 1.0, float(holm_bonferroni([0.6, 0.7])[1])),
        ("adjust ignores NaN", 1, int(np.isfinite(holm_bonferroni([0.5, np.nan])).sum())),
    ]
    # Power: the textbook normal approximation for a paired t-test at
    # alpha=0.05, power=0.80 is delta = 2.8016 * sd / sqrt(n); the exact
    # non-central-t solve must sit slightly above it and converge to it.
    approx = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) / np.sqrt(1000)
    exact = min_detectable_difference(1.0, 1000)
    rows.append(("MDE approaches normal approx", True, bool(0 < exact - approx < 0.005)))
    rows.append(("MDE exact exceeds approx at n=10", True,
                 bool(min_detectable_difference(1.0, 10) > 2.8016 / np.sqrt(10))))
    rows.append(("MDE achieves the target power", True,
                 bool(abs(_paired_t_power(min_detectable_difference(2.0, 78), 2.0, 78, 0.05)
                          - 0.80) < 1e-6)))
    # Variance-ratio MDE must hit its target power, tighten with n, and loosen
    # as the pairing correlation falls (independent samples are the worst case).
    rows.append(("variance MDE achieves target power", True,
                 bool(abs(_pitman_morgan_power(variance_ratio_mde(78, 0.9), 0.9, 78, 0.05)
                          - 0.80) < 1e-6)))
    rows.append(("variance MDE tightens with n", True,
                 bool(variance_ratio_mde(400, 0.9) > variance_ratio_mde(78, 0.9))))
    rows.append(("pairing correlation helps", True,
                 bool(variance_ratio_mde(78, 0.9) > variance_ratio_mde(78, 0.1))))
    # Pitman-Morgan on identical vectors is degenerate; on a pure rescale it
    # must fire, and it must be symmetric under swapping the two arms.
    rng0 = np.random.default_rng(7)
    base = rng0.normal(0, 1, 200)
    pm_r, _, pm_p = _pitman_morgan(base * 0.5, base + rng0.normal(0, 0.3, 200))
    rows.append(("Pitman-Morgan detects a rescale", True, bool(pm_p < 0.01)))
    rows.append(("Pitman-Morgan sign follows the larger arm", True, bool(pm_r < 0)))
    # Holm threshold: with a family of overwhelmingly significant others the
    # effective threshold must sit above the guaranteed alpha/m floor.
    tight = holm_threshold(np.full(58, 1e-12))
    rows.append(("Holm threshold beats alpha/m", True, bool(tight > 0.05 / 59)))
    rows.append(("Holm threshold capped at alpha", True, bool(tight <= 0.05 + 1e-12)))
    rows.append(("Holm threshold with hard family", True,
                 bool(holm_threshold(np.full(58, 0.9)) <= 0.05 / 59 + 1e-9)))
    # The signed log-ratio statistic must be provably truth-free: swapping the
    # ground truth for noise leaves it untouched, which is why it can never be
    # reported as an accuracy comparison. The APE difference must move.
    ti = pd.Index([f"j{k}" for k in range(30)])
    rng1 = np.random.default_rng(3)
    tt = pd.Series(rng1.uniform(100, 300, 30), index=ti)
    pa = pd.Series(tt.to_numpy() * rng1.uniform(0.9, 1.1, 30), index=ti)
    pb = pd.Series(tt.to_numpy() * rng1.uniform(0.9, 1.1, 30), index=ti)
    noise = pd.Series(rng1.uniform(1, 1e6, 30), index=ti)
    s_real = paired_statistics(pa, pb, tt)
    s_fake = paired_statistics(pa, pb, noise)
    rows.append(("signed log-ratio ignores truth", True, bool(
        np.max(np.abs(s_real["logratio_diff_signed"] - s_fake["logratio_diff_signed"])) < 1e-9)))
    rows.append(("APE difference uses truth", True, bool(
        np.max(np.abs(s_real["ape_diff"] - s_fake["ape_diff"])) > 1.0)))
    rows.append(("signed log-ratio equals log(a/b)", True, bool(np.allclose(
        s_real["logratio_diff_signed"], 100 * np.log(pa.to_numpy() / pb.to_numpy())))))
    # Exact sample size must be the true minimum, not a grid overshoot.
    rows.append(("n_for_delta is exact minimum", True, bool(
        _paired_t_power(0.2795633, 3.3891469, 1156, 0.05) >= 0.80
        and _paired_t_power(0.2795633, 3.3891469, 1155, 0.05) < 0.80)))

    # compare() must be exactly antisymmetric and coverage-honest.
    idx = pd.Index([f"i{k}" for k in range(50)])
    rng = np.random.default_rng(0)
    truth = pd.Series(rng.uniform(100, 200, 50), index=idx)
    a = truth * rng.normal(1.0, 0.02, 50)
    b = truth * rng.normal(1.0, 0.05, 50)
    b.iloc[:10] = np.nan
    fwd = compare(a, b, truth, "a", "b", n_boot=200, n_perm=500)
    rev = compare(b, a, truth, "b", "a", n_boot=200, n_perm=500)
    rows.append(("compare antisymmetric", True,
                 bool(abs(fwd["mean_diff"] + rev["mean_diff"]) < 1e-12)))
    rows.append(("compare pairs on both-finite", 40, fwd["n_pairs"]))
    rows.append(("compare flags unequal coverage", False, fwd["equal_coverage"]))
    rows.append(("same p both directions", True,
                 bool(abs(fwd["p_wilcoxon"] - rev["p_wilcoxon"]) < 1e-12)))

    out = pd.DataFrame(
        [
            {
                "check": n,
                "expected": e,
                "observed": o,
                "pass": bool(o == e) if isinstance(e, bool) or isinstance(e, int)
                else bool(abs(float(o) - float(e)) < 1e-9),
            }
            for n, e, o in rows
        ]
    )
    if verbose:
        print(out.to_string(index=False))
        print(f"\nSELFTEST ALL PASS: {bool(out['pass'].all())}")
    return out


def _nd_mape() -> float:
    """MAPE of the shipped model on the ND benchmark (not a suite stratum).

    The ND benchmark is a held-out synthetic split, not an OOD stratum, so it
    is checked here only to confirm the harness sees the same shipped model
    the paper reports.
    """
    path = REPO_ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
    d = pd.read_csv(path, usecols=["model", "abs_gap_pct"], low_memory=False)
    return float(d.loc[d["model"] == SHIPPED, "abs_gap_pct"].mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true", help="check the published numbers")
    ap.add_argument("--suite", action="store_true", help="print the suite and coverage tables")
    ap.add_argument("--power", action="store_true", help="print the per-stratum detection floor")
    ap.add_argument("--verdict", action="store_true", help="full verdict for the shipped model")
    ap.add_argument("--selftest", action="store_true", help="check the statistics routines")
    ap.add_argument("--dispersion", action="store_true", help="scale verdict on TSPLIB EUC_2D")
    ap.add_argument("--statistics", action="store_true", help="alternative paired statistics")
    ap.add_argument("--plan", action="store_true", help="mean-difference target at a larger n")
    ap.add_argument("--n-target", type=int, default=220, help="planned real-world stratum size")
    ap.add_argument("--out", type=Path, default=None, help="directory for CSV output")
    a = ap.parse_args(argv)
    if not any([a.validate, a.suite, a.power, a.verdict, a.selftest, a.dispersion,
                a.plan, a.statistics]):
        a.selftest = a.validate = a.suite = a.power = True

    frames: dict[str, pd.DataFrame] = {}
    if a.selftest:
        print("\n=== SELFTEST (statistics routines) ===")
        frames["selftest"] = selftest()
    if a.suite:
        s = suite_summary()
        print("\n=== SUITE ===")
        print(s.drop(columns=["provenance", "ood_rationale"]).to_string(index=False))
        cov = coverage_table()
        print("\n=== COVERAGE (finite predictions per baseline) ===")
        print(
            cov.pivot(index="baseline", columns="stratum", values="n_finite")
            .fillna(0).astype(int).to_string()
        )
        frames["suite"] = s
        frames["coverage"] = cov
    if a.power:
        p = power_analysis()
        print("\n=== POWER: minimum detectable mean paired APE difference ===")
        print(p.round(4).to_string(index=False))
        frames["power"] = p
    if a.validate:
        print("\n=== VALIDATION ===")
        frames["validation"] = validate()
    if a.dispersion:
        cands = candidate_predictions_from_suite([GART, "LGBM_V4"], "tsplib_euc2d")
        for metric in ("signed", "absolute"):
            d = dispersion_verdict(cands, "tsplib_euc2d", metric=metric)
            print(f"\n=== DISPERSION on tsplib_euc2d ({metric} percent error) ===")
            print(
                d[
                    ["model_a", "model_b", "n_pairs", "sdpe_a", "sdpe_b", "sd_ratio",
                     "pct_lower", "ratio_ci_low", "ratio_ci_high", "r_xy",
                     "p_pitman_morgan", "p_brown_forsythe", "p_swap_permutation",
                     "p_holm", "p_bh", "holm_alpha", "mde_sd_ratio",
                     "mde_sd_ratio_holm", "detectable", "detectable_holm"]
                ].round(4).to_string(index=False)
            )
            print(f"family size = {int(d['family_size'].iloc[0])}")
            frames[f"dispersion_{metric}"] = d
    if a.statistics:
        cands = candidate_predictions_from_suite([GART, "LGBM_V4"], "tsplib_euc2d")
        close = ["Asymptotic_MST", "MST_Ratio", "Fixed_Alpha", "Calibrated_MST_dn",
                 "MST_Only", "NN_V3"]
        r = scan_paired_statistics(cands, "tsplib_euc2d", baselines=close)
        print("\n=== ALTERNATIVE PAIRED STATISTICS on tsplib_euc2d ===")
        print(
            r[["model_a", "model_b", "statistic", "n_pairs", "mean_diff", "sd_effective",
               "dz", "dz_needed", "dz_shortfall_x", "p_wilcoxon", "p_holm",
               "sig_raw", "sig_holm", "n_needed", "block_between_frac"]]
            .round(4).to_string(index=False)
        )
        print(f"family size = {int(r['family_size'].iloc[0])}")
        print("\nWhat each statistic means:")
        for k, v in STATISTIC_MEANING.items():
            print(f"  {k}: {v}")
        frames["statistic_scan"] = r
    if a.plan:
        p = plan_mean_target(n_target=a.n_target)
        print(f"\n=== PLANNING: mean-difference target at n = {a.n_target} ===")
        print(p.round(5).to_string(index=False))
        frames["plan"] = p
    if a.verdict:
        v = evaluate_candidate(shipped_predictions(), SHIPPED)
        print(f"\n=== VERDICT ({v.headline()}) ===")
        print(v.comparisons.round(4).to_string(index=False))
        frames["verdict"] = v.comparisons
        frames["by_group"] = v.by_group

    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        for name, df in frames.items():
            df.to_csv(a.out / f"ood_{name}.csv", index=False)
        print(f"\nWrote {len(frames)} CSVs to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
