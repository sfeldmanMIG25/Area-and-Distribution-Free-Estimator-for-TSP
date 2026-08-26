"""Single source of truth for the model identifiers used across paper tooling.

Every script that has to name "the production model" or "the model it replaced"
imports from here instead of writing the string literal.  When the production
model changes, exactly one line in this file changes and every derived table,
figure, audit and prose-number generator follows.

Why this file exists
--------------------
``paper_tooling/gen_paper_numbers.py`` (legacy, superseded by
``build_paper_tables.py``) hardcoded ``MODEL_LABEL = {"LGBM_V3": "GART 2.0"}``
and then computed statistics on ``df.model == "LGBM_V3"`` while printing them
under the label "GART 2.0".  When the production model moved from ``LGBM_V3``
to ``GART_2.0`` the label stayed correct and the data silently did not.  That
class of defect is only preventable by never writing the identifier twice.
``paper_reference/plot_runtime.py`` carried the same defect in figure form.

Why here and not in ``build_paper_tables.py``
---------------------------------------------
The constants are consumed from two directories (``paper_tooling/`` and
``paper_reference/``), and importing the table builder just to read a string
would drag in scipy, pandas and ``tsplib_benchmark.exclusions``.  This mirrors
``tsplib_benchmark/exclusions.py``, the module the repository already uses for
shared screening constants: a dependency-free leaf module.  Consumers reach it
either as ``from model_registry import ...`` (with ``paper_tooling/`` on
``sys.path``) or as ``from paper_tooling.model_registry import ...`` (with the
repo root on ``sys.path``); both resolve to this file.

Roles, not aliases
------------------
``GART`` and ``PREDECESSOR`` are *roles*.  Do not assume the predecessor is
merely an older copy of the production model: the two differ in feature set,
target transform and, importantly, in coverage.  A script that needs a value
for **every** instance must not key off ``GART``, because the production model
is allowed to decline instances outside its support (on TSPLIB it declines
``si1032``).  Use ``EMBEDDING_DONOR`` for instance-level facts that have to be
present on every row.

Swapping the production model
-----------------------------
1. Change ``GART`` below.
2. In ``MODEL_LABELS``, give the new key the bare "GART 2.0" and either drop the
   old key (if it is no longer reported) or demote it to a self-describing
   label naming what distinguishes it -- never an internal version tag, which
   means nothing to a reader.  Display strings must stay unique: the ``--check``
   differ and the number bank key on them.
3. Set ``PREDECESSOR`` to the key just displaced.
4. Repoint ``PRODUCTION_BOOSTER`` / ``PRODUCTION_SIDECAR`` at the new frozen
   artifact, so booster statistics and SHAP rankings follow.
5. Re-check ``EMBEDDING_DONOR`` against its coverage note above.
Nothing outside this file should need editing.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The production model.  Change this line when the production model changes.
GART = "GART_2.0"

# The model the production model replaced.  Still benchmarked and still the
# full-coverage reference (see ``EMBEDDING_DONOR``), but NO LONGER REPORTED: it
# is dominated by ``LGBM_V4`` on every stratum, so the manuscript shows a single
# ablation rather than two, and this key deliberately carries no entry in
# ``MODEL_LABELS``.  Dropping the label is what removes it from the rosters, the
# tidy tables and the number bank in one place; the benchmark runners are
# untouched and keep scoring it.
PREDECESSOR = "LGBM_V3"

# Alias for readers who prefer the role name over the product name.  Same
# object, not a second source of truth -- ``GART`` remains the line to edit.
PRODUCTION = GART

# Frozen booster behind ``GART``, and its sidecar.  Booster statistics (tree
# count, leaves per tree, depth, feature count) and SHAP rankings quoted in the
# manuscript must be read from these, never from a predecessor artifact.
PRODUCTION_BOOSTER = REPO_ROOT / "lgbm_model_v3" / "gart2_final.joblib"
PRODUCTION_SIDECAR = REPO_ROOT / "lgbm_model_v3" / "gart2_final.json"

# Donor row for instance-level facts recorded by the benchmark runner rather
# than produced by a model: the original-distance MST length and the MDS
# embedding dimension ``k``.  These are properties of the instance, so any
# model row carries the same value, but the donor must cover *every* instance.
# ``PREDECESSOR`` is used deliberately: it never declines an instance, whereas
# ``GART`` records NaN for the instances it refuses to extrapolate on.
EMBEDDING_DONOR = PREDECESSOR

# Feature -> family.  The manuscript quotes SHAP contributions by *family*
# ("dimension and node count jointly contribute ...", "centroid-distance
# descriptors contribute ..."), and it states the roster as a family
# decomposition ("11 geometric ... the remaining 19 ... summarize the MST").
# Until this map existed both were maintained by hand, which is how a 30-feature
# split survived a move to 31 features: ``greedy_nn_over_mst`` belongs to none of
# the manuscript's named families and so was never noticed missing.
#
# The keys must be exactly ``PRODUCTION_SIDECAR``'s
# ``features_in_booster_order``; :func:`check_feature_families` enforces that, so
# a feature added to the booster fails loudly here rather than silently dropping
# out of a family share.
FEATURE_FAMILIES: dict[str, str] = {
    # -- geometric / instance-level (11) -----------------------------------
    "n_customers": "size_dimension",
    "dimension": "size_dimension",
    "bounding_hypervolume": "bounding_hypervolume",
    "log_bounding_hypervolume": "bounding_hypervolume",
    "node_density": "node_density",
    "log_node_density": "node_density",
    "aspect_ratio": "shape",
    "centroid_dist_mean": "centroid",
    "centroid_dist_std": "centroid",
    "centroid_dist_max": "centroid",
    "centroid_dist_iqr": "centroid",
    # -- MST edge-weight distribution (10) ---------------------------------
    "mst_edge_mean": "mst_edge",
    "mst_edge_std": "mst_edge",
    "mst_edge_skew": "mst_edge",
    "mst_edge_kurtosis": "mst_edge",
    "mst_edge_max": "mst_edge",
    "mst_edge_q10": "mst_edge",
    "mst_edge_q25": "mst_edge",
    "mst_edge_q50": "mst_edge",
    "mst_edge_q75": "mst_edge",
    "mst_edge_q90": "mst_edge",
    # -- MST topology (9) --------------------------------------------------
    "mst_dominance_ratio": "mst_topology",
    "mst_gap_ratio": "mst_topology",
    "mst_leaf_ratio": "mst_topology",
    "mst_degree_mean": "mst_topology",
    "mst_degree_std": "mst_topology",
    "mst_degree_max": "mst_topology",
    "mst_diameter": "mst_topology",
    "mst_diameter_normalized": "mst_topology",
    "large_edge_count": "mst_topology",
    # -- greedy-tour ratio (1) ---------------------------------------------
    # The feature the 31st slot added.  It is not MST-derived (it is a ratio of
    # a constructed tour to the MST) and not a geometric descriptor, so it gets
    # its own family rather than being folded into one it does not belong to.
    "greedy_nn_over_mst": "greedy_ratio",
}

# Coarse rollup used for the manuscript's "11 geometric / 19 MST" sentence.
FAMILY_GROUPS: dict[str, str] = {
    "size_dimension": "geometric",
    "bounding_hypervolume": "geometric",
    "node_density": "geometric",
    "shape": "geometric",
    "centroid": "geometric",
    "mst_edge": "mst",
    "mst_topology": "mst",
    "greedy_ratio": "greedy",
}


def check_feature_families(features) -> None:
    """Fail if ``FEATURE_FAMILIES`` and the booster roster have drifted apart."""
    roster = list(features)
    missing = [f for f in roster if f not in FEATURE_FAMILIES]
    extra = [f for f in FEATURE_FAMILIES if f not in roster]
    unmapped = sorted({v for v in FEATURE_FAMILIES.values()} - set(FAMILY_GROUPS))
    problems = []
    if missing:
        problems.append(f"booster features with no family: {missing}")
    if extra:
        problems.append(f"families naming absent features: {extra}")
    if unmapped:
        problems.append(f"families with no group: {unmapped}")
    if problems:
        raise SystemExit(
            "model_registry.FEATURE_FAMILIES is out of date with the production "
            "booster:\n  " + "\n  ".join(problems)
        )


# Human-readable label per benchmark-CSV ``model`` key.  Insertion order is the
# row order inside every bucket, so the production model comes first.
#
# A key with no entry here is scored but not reported: every roster, every tidy
# table, every paired test and every bank key is filtered on this mapping.
# ``PREDECESSOR`` is the standing example -- it is benchmarked and it is the
# embedding donor, and it appears in no manuscript table.
#
# Labels are self-describing, never internal version tags.  "GART 1.0" is the
# one version number a reader sees, because it is a published predecessor with
# a citation; a bare "V4" naming an artifact of our own release history is not
# a comparison a reader can make.
MODEL_LABELS: dict[str, str] = {
    GART: "GART 2.0",
    "GART": "GART 1.0",
    "GART_1.0": "GART 1.0",
    "MST_Ratio": "Fixed MST scaling",
    "Hilbert": "Custom Hilbert sort",
    # The next two labels name the feature block each model was actually fitted
    # on, because neither is a control on GART 2.0's feature vector:
    # ``Linear_V3`` is fitted on 28 columns and ``NN_V3`` on 30, against GART
    # 2.0's 31, and both are fitted on ``tsp_features_v3.csv`` rather than the
    # production ``tsp_features_v4.csv``, whose optimal_cost and
    # mst_total_length were both revised. They previously read "(same
    # features)", which was false; the manuscript now states the discrepancy in
    # prose, in the appendix passage on model-class controls
    # (``app:hyperparams``).
    #
    # ``Linear_31F`` / ``NN_31F`` ARE same-feature refits: same 31 columns in
    # booster order, same production table, same train/val split discipline, and
    # at inference the same extractor
    # (``feature_engineering_gart2.compute_features``).
    #
    # All four are scored, and all four keep their rows in every tidy CSV and in
    # the paired tests; none of them carries a manuscript table row (see the
    # roster note in build_paper_tables.py). Deleting these labels would drop
    # them from the tidy CSVs as well, which is not the decision that was made.
    # Display strings must stay unique -- the --check differ keys on them.
    "Linear_V3": "Linear (28-feature block)",
    "NN_V3": "Neural net (30-feature block)",
    "Linear_31F": "Linear (GART 2.0 features)",
    "NN_31F": "Neural net (GART 2.0 features)",
    "BHH": "BHH",
    "Cavdar": "Cavdar--Sokol",
    "Calibrated_MST_d": r"Calibrated MST ratio $\hat\rho(d)$",
    "Calibrated_MST_dn": r"Calibrated MST ratio $\hat\rho(d,n)$",
    "MST_Only": r"$L_{\mathrm{MST}}$ ($\alpha=1$)",
    "Asymptotic_MST": "Asymptotic MST ratio",
    # Sampling-region plug-in variant. Only BHH has one: its theorem names the
    # measure of the region, so feeding it the exact generator support G^d is a
    # source-faithful second reading. Cavdar--Sokol takes its area from the
    # rectangle covering the nodes and has no region input, so it has no twin --
    # the withdrawn ``Cavdar_region`` supplied G^2 to a model that never asked
    # for one.
    "BHH_region": "BHH (sampling region)",
    # Daganzo, Chien and Kwon--Golden--Wasil are withdrawn: unobtainable
    # primaries, coefficients transcribed from a secondary source. Their labels
    # are deleted too, so a stray key cannot quietly acquire a display name and
    # reappear in a table -- every roster here is filtered on MODEL_LABELS.
    "Fixed_Alpha": r"Fixed $\alpha=1.136$",
    "LGBM_V4": "GART 2.0 (32-feature variant)",
    "Interp_V3": "Interpretable (same features)",
}

# Short alias kept for callers that only need the two role labels.
LABELS = MODEL_LABELS

# Models whose prediction is a multiple of L_MST -> eligible for R^2_alpha.
ALPHA_MODELS = frozenset({
    GART, PREDECESSOR,
    "LGBM_V4", "NN_V3", "Linear_V3", "NN_31F", "Linear_31F",
    "GART", "GART_1.0", "MST_Ratio",
    "Calibrated_MST_d", "Calibrated_MST_dn", "Asymptotic_MST", "MST_Only",
    "Fixed_Alpha",
})


def assert_not_superseded(replacement: str = "paper_tooling/build_paper_tables.py") -> None:
    """Refuse to run a generator that predates the current production model.

    Several 2026-04 scripts emit LaTeX table blocks with ``"LGBM_V3"`` wired to
    the display name "GART 2.0". They also read pre-repair inputs and apply the
    superseded screens, so repointing them at ``GART`` would make them *look*
    current while still being wrong. Refusing is the honest fix.

    Set ``PAPER_ALLOW_LEGACY=1`` to run one deliberately (e.g. to reproduce a
    historical table); its output must not be spliced into the manuscript.
    """
    import os
    import sys

    if os.environ.get("PAPER_ALLOW_LEGACY") == "1":
        print(f"PAPER_ALLOW_LEGACY=1: running superseded generator "
              f"{Path(sys.argv[0]).name}. Output is historical -- do not splice.",
              file=sys.stderr)
        return
    raise SystemExit(
        f"{Path(sys.argv[0]).name} is superseded by {replacement}.\n"
        f"It hardcodes the predecessor ({PREDECESSOR}) under the production "
        f"label {MODEL_LABELS[GART]!r} and reads pre-repair inputs, so its "
        f"tables describe the wrong model.\n"
        f"Use {replacement}, then paper_tooling/splice_tables.py.\n"
        f"Set PAPER_ALLOW_LEGACY=1 to run it anyway for historical comparison."
    )


def label(model_key: str) -> str:
    """Display name for ``model_key``, falling back to the key itself."""
    return MODEL_LABELS.get(model_key, model_key)


def require_production_rows(df, column: str = "model"):
    """Return the ``GART`` rows of ``df``, or exit if there are none.

    Guards every consumer against the failure this module exists to prevent: a
    benchmark CSV predating the model swap silently yielding an empty frame,
    which a plotting or aggregation script then renders as an empty axis or a
    NaN table cell.
    """
    rows = df[df[column] == GART]
    if len(rows) == 0:
        present = sorted(map(str, df[column].dropna().unique()))
        raise SystemExit(
            f"No rows for the production model {GART!r}. The input predates "
            f"the current model. Models present: {present}"
        )
    return rows
