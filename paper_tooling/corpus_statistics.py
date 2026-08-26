"""Model-independent corpus facts, exported and banked.

Why this exists
---------------
Seven manuscript claims describe the *corpora*, not any model: the spread of
``alpha`` by dimension in the training split, the two rows whose raw ``alpha``
falls outside ``[1, 2]``, the moments of ``alpha`` over the 78 TSPLIB EUC_2D
instances, the range of ``alpha`` over the non-Euclidean set, where
``alpha > 1.45`` lives, and the geometry of the near-collinear instances the
augmentation rounds tried to match.  None of them moves when the production
model changes -- which is exactly why none of them had a generator, and why a
stale one would be harder to notice than a stale benchmark number.

Two of them are not purely corpus facts, and the difference matters:

* ``app.alpha_range_22`` describes ``alpha`` over "the 22 screened instances it
  accepts".  The *set* is chosen by the production model's coverage, so the
  statistic is model-coupled even though ``alpha`` is not.  This module keys the
  accepted set off the registry's production model and reports the screened
  (23) and accepted (22) sets side by side, so the coupling is on the page
  instead of implicit.
* the transverse-profile facts are measured from instance coordinates through
  ``data_pipeline.augment_gen``'s own ``measured_rho`` / ``transverse_kurtosis``,
  the same primitives the generator used, so the comparison is like-for-like.

Sources
-------
``tsp_features_v4.csv``                    training corpus (the sidecar's table)
``tsplib_benchmark/results/all_models_tsplib.csv``  TSPLIB truth + the screen
``Generalized_TSP_Analysis/instances``     benchmark Line Noise coordinates
``augment/{full,batch2}_records.json``     the two augmentation rounds

Output
------
``tables/corpus_alpha_by_dimension.csv``
``tables/corpus_alpha_sets.csv``
``tables/corpus_out_of_interval.csv``
``tables/corpus_linenoise_geometry.csv``
Bank keys prefixed ``corpus_``.

CLI
---
    python paper_tooling/corpus_statistics.py
    python paper_tooling/corpus_statistics.py --no-bank
    python paper_tooling/corpus_statistics.py --skip-geometry   # no coordinate pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import GART, PRODUCTION_SIDECAR, REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

TABLES = REPO_ROOT / "paper_tooling" / "tables"
BANK = TABLES / "paper_numbers.json"
#: This module's keys, kept separately so a full table rebuild can re-merge them.
SIDECAR_KEYS = TABLES / "corpus_numbers.json"

TSPLIB_CSV = REPO_ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
BENCH2D_INSTANCES = REPO_ROOT / "Generalized_TSP_Analysis" / "instances"
AUGMENT_RECORDS = {
    "batch1": REPO_ROOT / "augment" / "full_records.json",
    "batch2": REPO_ROOT / "augment" / "batch2_records.json",
}

KEY_PREFIX = "corpus_"

#: Threshold and size cut the manuscript uses for the coverage-hole claim.
ALPHA_HOLE_THRESHOLD = 1.45
ALPHA_HOLE_N_CUT = 10

#: Size cut for the Line Noise geometry claims ("Line Noise at n >= 200").
LINENOISE_N_CUT = 200

#: Augmentation families that are near-collinear / low-intrinsic-dimension, and
#: so belong in the rho comparison. ``lattice`` and ``hexlattice`` target the
#: opposite-signed failure and carry no rho.
NEAR_COLLINEAR_FAMILIES = frozenset({"collinear", "curve", "filament", "subspace"})


# --------------------------------------------------------------------------
# Training corpus
# --------------------------------------------------------------------------


def _training_frame() -> pd.DataFrame:
    """Training table with raw and clipped ``alpha``, exactly as the freeze does.

    ``freeze_gart2_final.py`` computes ``alpha = optimal_cost / mst_total_length``
    with zero MST lengths turned into NaN and then clips to ``[1, 2]``.  Both the
    raw and the clipped column are kept: the clipped one is the target, the raw
    one is what the out-of-interval audit is about.
    """
    sidecar = json.loads(PRODUCTION_SIDECAR.read_text(encoding="utf-8"))
    table = REPO_ROOT / sidecar["training_table"]
    df = pd.read_csv(
        table,
        usecols=["instance_name", "split", "dimension", "n_customers",
                 "optimal_cost", "mst_total_length"],
        low_memory=False,
    )
    raw = df["optimal_cost"] / df["mst_total_length"].replace(0, np.nan)
    df["raw_alpha"] = raw
    df["alpha"] = raw.clip(1.0, 2.0)
    return df


def alpha_by_dimension(df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
    sub = df[(df["split"] == split) & df["alpha"].notna()]
    out = (
        sub.groupby("dimension")["alpha"]
        .agg(n="size", mean="mean", sd=lambda s: s.std(ddof=1),
             min="min", max="max")
        .reset_index()
        .sort_values("dimension")
        .reset_index(drop=True)
    )
    out.insert(0, "split", split)
    return out


def out_of_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose *raw* alpha leaves ``[1, 2]`` and is therefore clipped."""
    m = df["raw_alpha"].notna() & ((df["raw_alpha"] < 1.0) | (df["raw_alpha"] > 2.0))
    return (
        df.loc[m, ["instance_name", "split", "n_customers", "dimension", "raw_alpha"]]
        .sort_values("raw_alpha")
        .reset_index(drop=True)
    )


def alpha_hole(df: pd.DataFrame, split: str = "train") -> dict:
    """Where ``alpha > 1.45`` lives in the training split, by instance size."""
    sub = df[(df["split"] == split) & df["alpha"].notna()]
    hi = sub[sub["alpha"] > ALPHA_HOLE_THRESHOLD]
    at_or_below = int((hi["n_customers"] <= ALPHA_HOLE_N_CUT).sum())
    return {
        "threshold": ALPHA_HOLE_THRESHOLD,
        "n_rows": int(len(hi)),
        "max_n": int(hi["n_customers"].max()) if len(hi) else 0,
        "n_le_cut": at_or_below,
        "frac_le_cut": at_or_below / len(hi) if len(hi) else np.nan,
        "n_above_cut": int(len(hi) - at_or_below),
        "sizes_above_cut": sorted(
            int(v) for v in hi.loc[hi["n_customers"] > ALPHA_HOLE_N_CUT,
                                   "n_customers"].unique()
        ),
    }


# --------------------------------------------------------------------------
# TSPLIB corpus
# --------------------------------------------------------------------------


def tsplib_alpha_sets() -> pd.DataFrame:
    """``alpha`` moments over the screened EUC_2D / non-EUC_2D sets.

    ``alpha`` is recomputed as ``true_cost / mst_length`` rather than read from
    the CSV's ``alpha`` column: that column is per-model and holds the model's
    *predicted* multiplier, so averaging it would report a model output under a
    corpus label -- the same defect ``model_registry`` exists to prevent.

    The instance-level facts come from the embedding donor, which covers every
    instance; the *accepted* set is then intersected with the production model's
    finite predictions, and that intersection is the only model-dependent thing
    here.
    """
    from tsplib_benchmark.exclusions import filter_metric_consistent

    from model_registry import EMBEDDING_DONOR

    raw = pd.read_csv(TSPLIB_CSV)
    donor = raw[raw["model"] == EMBEDDING_DONOR].drop_duplicates("instance")
    screened = filter_metric_consistent(donor).set_index("instance")
    screened["alpha"] = screened["true_cost"] / screened["mst_length"]

    prod = raw[raw["model"] == GART].drop_duplicates("instance").set_index("instance")
    accepted = prod.index[prod["pred_cost"].notna()]

    euc = screened[screened["edge_weight_type"] == "EUC_2D"]
    non = screened[screened["edge_weight_type"] != "EUC_2D"]
    sets = {
        "tsplib_euc2d_screened": euc,
        "tsplib_euc2d_accepted": euc.loc[euc.index.intersection(accepted)],
        "tsplib_noneuc_screened": non,
        "tsplib_noneuc_accepted": non.loc[non.index.intersection(accepted)],
    }
    rows = []
    for name, frame in sets.items():
        a = frame["alpha"].dropna()
        rows.append({
            "set": name,
            "model_coupled": name.endswith("_accepted"),
            "coverage_model": GART if name.endswith("_accepted") else None,
            "n": int(len(a)),
            "mean": float(a.mean()) if len(a) else np.nan,
            "sd": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
            "min": float(a.min()) if len(a) else np.nan,
            "max": float(a.max()) if len(a) else np.nan,
            "declined": ";".join(sorted(set(frame.index) - set(accepted))) or "",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Near-collinear geometry
# --------------------------------------------------------------------------


def linenoise_geometry(n_cut: int = LINENOISE_N_CUT) -> pd.DataFrame:
    """Realised rho and transverse kurtosis for the near-collinear corpora.

    Three populations, measured through the *same* two functions so the
    manuscript's "matched rho" comparison is like-for-like:

    ``benchmark``  the 2D benchmark's ``line_noise`` instances (the target)
    ``batch1``     augmentation round 1 (isotropic transverse noise)
    ``batch2``     augmentation round 2, aimed at ``rho in [2, 100]``

    Kurtosis is recomputed from coordinates for all three rather than read from
    a record: the manuscript's ``2.10`` against ``3.22`` is the whole shape
    argument, and a comparison between a measured number and a remembered one
    is not a comparison.  The augmentation records' own ``rho`` (nominal, what
    the generator was asked for) is kept alongside ``rho_measured`` (what the
    clipped point set has), because the manuscript quotes the nominal target
    for the added instances and a realised value for the benchmark.
    """
    from data_pipeline.augment_gen import measured_rho, transverse_kurtosis

    def measure(coords: np.ndarray) -> tuple[float, float]:
        c = np.asarray(coords, dtype=float)
        return measured_rho(c), transverse_kurtosis(c)

    rows = []
    truth = pd.read_csv(
        REPO_ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints"
        / "base_ground_truth_2d.csv"
    ).set_index("instance")
    for path in sorted(BENCH2D_INSTANCES.glob("TSP-line_noise-n*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rho, kur = measure(rec["coordinates"])
        name = rec["instance_name"]
        rows.append({
            "population": "benchmark",
            "instance": name,
            "n": int(rec["n_customers"]),
            "dimension": int(rec["dimension"]),
            "rho_nominal": np.nan,
            "rho_measured": rho,
            "kurtosis": kur,
            "alpha": (float(truth.loc[name, "true_alpha"])
                      if name in truth.index else np.nan),
        })

    aug_dir = REPO_ROOT / "augment" / "instances"
    for batch, path in AUGMENT_RECORDS.items():
        if not path.exists():
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            if rec.get("family") not in NEAR_COLLINEAR_FAMILIES:
                continue
            coord_file = aug_dir / f"{rec['name']}.json"
            rho_m, kur = np.nan, np.nan
            if coord_file.exists():
                inst = json.loads(coord_file.read_text(encoding="utf-8"))
                rho_m, kur = measure(inst["coordinates"])
            rows.append({
                "population": batch,
                "instance": rec["name"],
                "n": int(rec["n"]),
                "dimension": int(rec["d"]),
                "rho_nominal": rec.get("rho", np.nan),
                "rho_measured": rec.get("rho_measured", rho_m),
                "kurtosis": kur,
                "alpha": rec.get("alpha", np.nan),
            })

    df = pd.DataFrame(rows)
    df["large_n"] = df["n"] >= n_cut
    return df


def transverse_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Transverse kurtosis at ``d = 2``, the only dimension where it compares.

    ``transverse_kurtosis`` is ``E[t^4] / E[t^2]^2`` for the *norm* of the
    transverse displacement.  In ``d`` dimensions that norm lives in ``d - 1``
    dimensions, so isotropic Gaussian noise gives ``3.0`` only at ``d = 2`` and
    falls towards ``1`` as ``d`` grows.  Pooling dimensions therefore compares a
    2D benchmark against a mostly-high-dimensional augmentation set through a
    statistic whose reference value is different in each -- which reads as a
    shape difference that is really a dimension difference.  The manuscript's
    "2.10 against 3.22" is a 2D statement, so this restricts to ``d = 2``.
    """
    sub = df[df["dimension"] == 2]
    out = (
        sub.groupby("population")["kurtosis"]
        .agg(n_instances="size", median="median", mean="mean",
             q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    out["gaussian_reference"] = 3.0
    return out


def linenoise_by_size(df: pd.DataFrame, n_cut: int) -> pd.DataFrame:
    """Per-size medians, which is the shape the manuscript's bands are in.

    The claim is "Line Noise at n >= 200 has median rho between 25 and 46" --
    a band over the *per-size* medians, not a pooled median.  Pooling would
    answer a different question, so the band is computed the way it is stated.
    """
    sub = df[(df["population"] == "benchmark") & (df["n"] >= n_cut)]
    return (
        sub.groupby("n")[["rho_measured", "alpha", "kurtosis"]]
        .median()
        .reset_index()
        .sort_values("n")
        .reset_index(drop=True)
    )


def _summarise_geometry(df: pd.DataFrame, n_cut: int) -> pd.DataFrame:
    out = []
    for (pop, large), sub in df.groupby(["population", "large_n"]):
        rho = pd.to_numeric(sub["rho_measured"], errors="coerce").dropna()
        nom = pd.to_numeric(sub["rho_nominal"], errors="coerce").dropna()
        alpha = pd.to_numeric(sub["alpha"], errors="coerce").dropna()
        kur = pd.to_numeric(sub["kurtosis"], errors="coerce").dropna()
        out.append({
            "population": pop,
            "n_ge_cut": bool(large),
            "n_cut": n_cut,
            "n_instances": int(len(sub)),
            "rho_nominal_min": float(nom.min()) if len(nom) else np.nan,
            "rho_nominal_max": float(nom.max()) if len(nom) else np.nan,
            "rho_median": float(rho.median()) if len(rho) else np.nan,
            "rho_min": float(rho.min()) if len(rho) else np.nan,
            "rho_max": float(rho.max()) if len(rho) else np.nan,
            "alpha_median": float(alpha.median()) if len(alpha) else np.nan,
            "alpha_min": float(alpha.min()) if len(alpha) else np.nan,
            "alpha_max": float(alpha.max()) if len(alpha) else np.nan,
            "kurtosis_median": float(kur.median()) if len(kur) else np.nan,
            "kurtosis_mean": float(kur.mean()) if len(kur) else np.nan,
        })
    return pd.DataFrame(out).sort_values(["population", "n_ge_cut"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Banking
# --------------------------------------------------------------------------


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, 9)


def bank_numbers(by_dim, sets, ooi, hole, geo, by_size, shape) -> dict[str, object]:
    numbers: dict[str, object] = {}

    for _, r in by_dim.iterrows():
        d = int(r["dimension"])
        numbers[f"{KEY_PREFIX}train_alpha_d{d}_n"] = int(r["n"])
        for f in ("mean", "sd", "min", "max"):
            numbers[f"{KEY_PREFIX}train_alpha_d{d}_{f}"] = _num(r[f])
    sd = by_dim["sd"].to_numpy(dtype=float)
    numbers[f"{KEY_PREFIX}train_alpha_sd_monotone_decreasing"] = bool(
        np.all(np.diff(sd) <= 0)
    )
    numbers[f"{KEY_PREFIX}train_alpha_sd_dimensions"] = int(len(by_dim))

    for _, r in sets.iterrows():
        s = r["set"]
        numbers[f"{KEY_PREFIX}alpha_{s}_n"] = int(r["n"])
        for f in ("mean", "sd", "min", "max"):
            numbers[f"{KEY_PREFIX}alpha_{s}_{f}"] = _num(r[f])

    numbers[f"{KEY_PREFIX}out_of_interval_n"] = int(len(ooi))
    for i, r in ooi.iterrows():
        numbers[f"{KEY_PREFIX}out_of_interval_{i}_name"] = str(r["instance_name"])
        numbers[f"{KEY_PREFIX}out_of_interval_{i}_split"] = str(r["split"])
        numbers[f"{KEY_PREFIX}out_of_interval_{i}_raw_alpha"] = _num(r["raw_alpha"])

    numbers[f"{KEY_PREFIX}train_alpha_gt145_n"] = hole["n_rows"]
    numbers[f"{KEY_PREFIX}train_alpha_gt145_max_n"] = hole["max_n"]
    numbers[f"{KEY_PREFIX}train_alpha_gt145_n_le_10"] = hole["n_le_cut"]
    numbers[f"{KEY_PREFIX}train_alpha_gt145_frac_le_10"] = _num(hole["frac_le_cut"])
    numbers[f"{KEY_PREFIX}train_alpha_gt145_n_above_10"] = hole["n_above_cut"]

    if geo is not None:
        for _, r in geo.iterrows():
            tag = f"{r['population']}_{'large_n' if r['n_ge_cut'] else 'small_n'}"
            numbers[f"{KEY_PREFIX}linenoise_{tag}_n_instances"] = int(r["n_instances"])
            for f in ("rho_nominal_min", "rho_nominal_max", "rho_median",
                      "rho_min", "rho_max", "alpha_median", "alpha_min",
                      "alpha_max", "kurtosis_median", "kurtosis_mean"):
                numbers[f"{KEY_PREFIX}linenoise_{tag}_{f}"] = _num(r[f])

    if by_size is not None and len(by_size):
        # The band over per-size medians -- the form the manuscript states.
        for col, short in (("rho_measured", "rho"), ("alpha", "alpha"),
                           ("kurtosis", "kurtosis")):
            v = pd.to_numeric(by_size[col], errors="coerce").dropna()
            numbers[f"{KEY_PREFIX}linenoise_benchmark_large_n_median_{short}_band_lo"] = \
                _num(v.min())
            numbers[f"{KEY_PREFIX}linenoise_benchmark_large_n_median_{short}_band_hi"] = \
                _num(v.max())
        numbers[f"{KEY_PREFIX}linenoise_benchmark_large_n_sizes"] = int(len(by_size))

    if shape is not None and len(shape):
        for _, r in shape.iterrows():
            p = str(r["population"])
            numbers[f"{KEY_PREFIX}kurtosis_d2_{p}_n_instances"] = int(r["n_instances"])
            for f in ("median", "mean", "q25", "q75"):
                numbers[f"{KEY_PREFIX}kurtosis_d2_{p}_{f}"] = _num(r[f])
        numbers[f"{KEY_PREFIX}kurtosis_d2_gaussian_reference"] = 3.0
    return numbers


def corpus_bank_numbers() -> dict[str, object]:
    """This module's bank keys, from the sidecar it writes.

    ``build_paper_tables.main()`` rewrites ``paper_numbers.json`` wholesale, so
    without a hook of this shape a full table rebuild silently deletes every key
    written here. Reading the sidecar rather than recomputing keeps the table
    builder free of a coordinate pass over 1,084 instance files.
    """
    if not SIDECAR_KEYS.exists():
        return {}
    return json.loads(SIDECAR_KEYS.read_text(encoding="utf-8"))


def merge_bank(numbers: dict[str, object], full: bool) -> tuple[int, int, int]:
    """Merge into the bank.

    ``full`` says whether this run produced every ``corpus_`` key it is capable
    of.  A partial run (``--skip-geometry``) must not delete the keys it did not
    regenerate, so pruning only happens on a full run.
    """
    carried = corpus_bank_numbers() if not full else {}
    bank: dict[str, object] = json.loads(BANK.read_text(encoding="utf-8"))
    stale = (
        [k for k in bank if k.startswith(KEY_PREFIX) and k not in numbers]
        if full else []
    )
    changed = sum(1 for k, v in numbers.items() if k in bank and bank[k] != v)
    fresh = sum(1 for k in numbers if k not in bank)
    for k in stale:
        del bank[k]
    SIDECAR_KEYS.write_text(
        json.dumps({**carried, **numbers}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    bank.update(numbers)
    BANK.write_text(json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8")
    return fresh, changed, len(stale)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-bank", action="store_true")
    ap.add_argument("--skip-geometry", action="store_true",
                    help="skip the coordinate pass over the Line Noise instances")
    ap.add_argument("--n-cut", type=int, default=LINENOISE_N_CUT)
    a = ap.parse_args(argv)

    TABLES.mkdir(parents=True, exist_ok=True)

    df = _training_frame()
    by_dim = alpha_by_dimension(df)
    ooi = out_of_interval(df)
    hole = alpha_hole(df)
    sets = tsplib_alpha_sets()

    by_dim.to_csv(TABLES / "corpus_alpha_by_dimension.csv", index=False)
    ooi.to_csv(TABLES / "corpus_out_of_interval.csv", index=False)
    sets.to_csv(TABLES / "corpus_alpha_sets.csv", index=False)

    print("=== training-split alpha by dimension ===")
    print(by_dim.round(6).to_string(index=False))
    sd = by_dim["sd"].to_numpy(dtype=float)
    print(f"monotone non-increasing: {bool(np.all(np.diff(sd) <= 0))}  "
          f"(d={int(by_dim['dimension'].iloc[0])}: {sd[0]:.4f} -> "
          f"d={int(by_dim['dimension'].iloc[-1])}: {sd[-1]:.4f})")

    print("\n=== alpha over the TSPLIB sets ===")
    print(sets.round(6).to_string(index=False))

    print("\n=== raw alpha outside [1, 2] ===")
    print(ooi.round(6).to_string(index=False))

    print(f"\n=== alpha > {ALPHA_HOLE_THRESHOLD} in the training split ===")
    print(f"{hole['n_rows']} rows, max n = {hole['max_n']}, "
          f"{hole['n_le_cut']} at n <= {ALPHA_HOLE_N_CUT} "
          f"({100 * hole['frac_le_cut']:.2f}%), "
          f"{hole['n_above_cut']} above it at n in {hole['sizes_above_cut']}")

    geo = by_size = shape = None
    if not a.skip_geometry:
        raw_geo = linenoise_geometry(a.n_cut)
        raw_geo.to_csv(TABLES / "corpus_linenoise_geometry.csv", index=False)
        geo = _summarise_geometry(raw_geo, a.n_cut)
        by_size = linenoise_by_size(raw_geo, a.n_cut)
        by_size.to_csv(TABLES / "corpus_linenoise_by_size.csv", index=False)
        print(f"\n=== near-collinear geometry (cut at n = {a.n_cut}) ===")
        print(geo.round(4).to_string(index=False))
        print(f"\n=== benchmark Line Noise, per-size medians (n >= {a.n_cut}) ===")
        print(by_size.round(4).to_string(index=False))
        r = pd.to_numeric(by_size["rho_measured"], errors="coerce")
        al = pd.to_numeric(by_size["alpha"], errors="coerce")
        print(f"band over per-size medians: rho [{r.min():.2f}, {r.max():.2f}], "
              f"alpha [{al.min():.4f}, {al.max():.4f}]")
        shape = transverse_shape(raw_geo)
        shape.to_csv(TABLES / "corpus_transverse_shape.csv", index=False)
        print("\n=== transverse kurtosis at d = 2 (isotropic Gaussian = 3.0) ===")
        print(shape.round(4).to_string(index=False))

    if not a.no_bank:
        fresh, changed, stale = merge_bank(
            bank_numbers(by_dim, sets, ooi, hole, geo, by_size, shape),
            full=not a.skip_geometry,
        )
        print(f"\nbank: {fresh} new, {changed} updated, {stale} stale removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
