"""Manifest of the numbers the manuscript *asserts in prose*, and where each comes from.

``build_paper_tables.py --check`` verifies the table bodies.  It verifies nothing
in the prose, and the prose is where this project's stale numbers have survived:
a production-model swap moved almost every metric while the sentences quoting
them stayed put.  See ``paper_tooling/prose_claim_audit.md`` for the damage.

This file is the durable half of the fix.  It records, for one asserted number
at a time, **which generated artifact backs it** -- never the value itself.  When
the production model changes the values move; the mapping does not.  That is the
whole design constraint: a manifest entry must survive a model swap untouched.

Schema
------
Each entry is one ``Claim``.  One claim = one number.  Never bundle two numbers
into an entry: tolerance, provenance and the mismatch message are all per-number.

``anchor``
    Verbatim manuscript text with the numerals punched out as holes.  **Not a
    line number** -- the manuscript is edited concurrently and line numbers rot
    within hours.  Two hole markers:

    ``{v}``  the number this claim owns.  Exactly one per entry.
    ``{~}``  a numeral that lives inside the anchor but belongs to a *different*
             entry.  It is deliberately *not* counted as covered, so if no other
             entry claims it the checker reports it UNREGISTERED.

    Whitespace in the anchor matches any whitespace run, so a claim may straddle
    a hard line break.  Everything else is matched literally (LaTeX and all).
    The anchor must match exactly once; zero or several matches is an error, not
    a skip -- a claim that quietly stops being checked is the failure mode this
    file exists to prevent.

``expect``
    Where the true value comes from.  Three forms:

    ``"bank:<key>"`` / ``"<key>"``
        A key of ``paper_tooling/tables/paper_numbers.json`` (the ``bank:``
        prefix is optional and is the default source).
    ``"sidecar:<dotted.path>"``
        A path into the frozen production sidecar named by
        ``model_registry.PRODUCTION_SIDECAR``.  Use this for facts about the
        model artifact -- tree count, feature count, split sizes.  It follows
        the production model automatically when ``model_registry.GART`` changes.
    ``"frontier:<slash/separated/path>"``
        A path into ``paper_tooling/frontier_manuscript_bank.json``, the
        Held--Karp study's consolidated key space.  Separate from the bank
        because ``build_paper_tables.py`` rewrites ``paper_numbers.json``
        wholesale and a frontier key added there would not survive.
    ``"allbench:<slash/separated/path>"``
        A path into ``paper_tooling/hk1tree_allbench_bank.json``: the bound's
        **accuracy** on all four benchmarks, including the 2D diverse and
        non-EUC\_2D corpora the bank carries no bound row for.
    ``"costfront:<slash/separated/path>"``
        A path into ``paper_tooling/hk1tree_cost_frontier_bank.json``: the
        bound's **cost** on those same two corpora, plus the drift control that
        says whether its absolute milliseconds may be printed beside Table 3's.
        Written by ``paper_tooling/hk1tree_cost_analyze.py``.
    The three slash-separated sources use ``/`` and not ``.`` because their own
    keys contain dots (``GART_2.0_MAPE_pct``) and bucket labels
    (``n in [51,150]``), so a dotted path would be ambiguous.
    ``"= <expression>"``
        Arithmetic over ``{source-spec}`` placeholders, e.g.
        ``"= {a_mape} / {b_mape}"``.  Use it for every margin, ratio and
        percentage-point difference so the *derivation* is checked instead of a
        hand-computed constant being trusted.  Only ``+ - * / ** ()`` and
        ``abs/min/max`` are permitted.

``no_generator``
    Set instead of ``expect`` to register a number that no current tool
    produces.  The reason must say what would have to be run to settle it.  This
    is the only sanctioned way for an asserted number to go unverified, and it
    is reported as its own state so the backlog stays visible.

``tol``
    Per-entry, because a percentage printed to two decimals and a tree count
    need different rules.  Every mode resolves to one **acceptance radius** and
    the verdict is ``|printed - generated| <= radius``.  The radius is a
    function of the generated value and the mode only -- never of how many
    decimals the sentence happens to print.  (It used to be the latter, which
    let an author widen the band tenfold by deleting a decimal point and made
    the verdict non-monotone in the error; see the tolerance note in
    ``check_prose_numbers.py``.)

    ``"printed"``   (default) half a unit in the last of ``SIG_FIGS``
                    significant figures of the generated value -- "correct to
                    two significant figures", clamped so it is never coarser
                    than a whole unit.  Scale-free, so it behaves the same on a
                    p-value of 0.0049 and on a MAPE of 9.8; the
                    ``max(0.005, 0.5%)`` cap it replaced did not, and was 103% of
                    the former while rejecting every honest rendering of the
                    latter's [1,10) neighbors.
    ``("dp", k)``   this claim is asserted to ``k`` decimal places: half a unit
                    in the ``k``-th decimal.  Tightening only -- ``k`` coarser
                    than the significant-figure floor is raised to the floor, so
                    a two-decimal declaration cannot hand a p-value of 0.0048533
                    a band wider than the p-value.  ``k`` also sets the precision
                    the prose is required to print: fewer decimals than the claim
                    can be checked at is reported as UNDER_PRECISE and in the
                    PRECISION POLICY list, never as a silent pass.
    ``("abs", x)``  absolute radius ``x``.  The sanctioned way to go *looser*
                    than the floor, e.g. for a number the prose deliberately
                    rounds hard.  Explicit and greppable, unlike a typographic
                    accident -- so justify it in ``note``.
    ``("rel", x)``  relative radius (fraction, not percent).  Same role.
    ``"exact"``     equality; for counts.

    A claim can always be satisfied by a correct sentence.  Whatever the mode,
    the checker prints the rendering that would pass -- the generated value at
    ``required_dp`` -- and ``--selftest`` P5 sweeps the guarantee that writing it
    yields MATCHED.  A tolerance a correct sentence cannot satisfy is a bug in
    this file or in the rule, not in the manuscript.

``scale``
    Multiplier applied to the generated value before comparing, for the cases
    where the bank stores a fraction and the prose prints a percentage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

try:  # paper_tooling/ on sys.path -- how check_prose_numbers imports this module
    from model_registry import GART, MODEL_LABELS
except ImportError:  # repo root on sys.path
    from paper_tooling.model_registry import GART, MODEL_LABELS

# Product family behind the production model: "GART 2.0" -> "GART".  Derived,
# never written twice, so an arm swap renames the version-tag rule by itself.
_LABEL = MODEL_LABELS[GART]
_FAMILY_MATCH = re.match(r"[^\W\d_]+", _LABEL)
_FAMILY = re.escape(_FAMILY_MATCH.group(0) if _FAMILY_MATCH else _LABEL)

# Tolerance spec: "printed" | "exact" | ("dp", int) | ("abs", float) | ("rel", float)
Tol = "str | tuple[str, float]"


@dataclass(frozen=True)
class Claim:
    """One asserted number in the manuscript prose, and its provenance."""

    id: str
    anchor: str
    expect: str | None = None
    no_generator: str | None = None
    tol: object = "printed"
    scale: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if (self.expect is None) == (self.no_generator is None):
            raise ValueError(f"{self.id}: set exactly one of expect / no_generator")
        if self.anchor.count("{v}") != 1:
            raise ValueError(f"{self.id}: anchor needs exactly one {{v}} hole")


# ---------------------------------------------------------------------------
# Declared incidental patterns.
#
# Whole *classes* of numeral that are structurally not results.  Without these
# the UNREGISTERED list fills with seeds and subscripts and stops being read,
# which defeats the point.  Each pattern is matched against the text
# immediately around a numeral; the checker prints a hit count per pattern, so a
# rule that swallows more than it should is visible rather than silent.
#
# Add a pattern only for a *class*.  A one-off numeral that is genuinely not a
# result belongs in CLAIMS with ``no_generator``, where it carries a reason.
#
# Earning a place here
# --------------------
# A rule must suppress a numeral that is **in the manuscript now**, and the
# comment above it must name one.  A rule that fires zero times is not free: it
# is a standing licence to hide a future result, and this list accumulated seven
# such rules before anyone counted.  ``log-base`` (``\log_2``) and ``norm-order``
# (``\ell_2``) never fired and were redundant besides -- both digits are
# subscripts, which the checker rejects structurally.  ``float-guard``
# (``1e-9``), ``\bV\d\b`` (the digit of "V3" is never tokenised: a letter
# precedes it) and ``\bTSPLIB\s?\d+`` were removed for the same reason.  The
# TSPLIB alternative was worse than inert: "TSPLIB95" never tokenises either, so
# the only numeral it ever owned was the *result* 55 in "on TSPLIB 55 pairs
# cannot support a 7-point claim".
# ---------------------------------------------------------------------------
INCIDENTAL_PATTERNS: list[tuple[str, str]] = [
    # "Using seed 42, 100 Optuna TPE trials"; "5{,}000 held-out rows, seed 42".
    ("rng-seed", r"seeds?~?\s*=?\s*\d+"),
    # The index of a dimension slice, not a measurement: "from 0.1837 at $d=2$
    # to 0.0762 at $d=50$".  The lookbehind keeps the rule off "grid=100" and
    # any other identifier that merely ends in "d".
    ("dimension-index", r"\$?(?<![A-Za-z])d\s*=\s*\d+\$?"),
    # The endpoints of a dimension GROUP, not measurements: the 4 and 10 of
    # "It holds at $d\in[4,10]$, at $d\in[15,50]$ and at $d=100$".  Exactly the
    # same class as dimension-index above, which only covers the scalar
    # "$d=50$" form and so leaves every interval endpoint exposed; six such
    # numerals are in Section~\ref{sec:frontier} now.  Both brace and bracket
    # forms, since the manuscript writes $d\in\{2,3\}$ and $d\in[15,50]$.
    ("dimension-group", r"d\\in\s*(?:\\\{|\[)\s*\d+\s*,\s*\d+\s*(?:\\\}|\])"),
    # A product name, not a measurement: "GART 2.0" and its "GART 1.0"
    # predecessor.  The family is read from model_registry, so an arm swap
    # carries the rule with it and no model name is written twice.
    ("version-tag", rf"\b{_FAMILY}\b[^\w\n]{{0,3}}\d\.\d"),
    # Solver releases in the reproducibility note: "Concorde 03.12.19" (whose
    # "03.12" the two-decimal rule would otherwise promote) and "LKH-3 3.0.13".
    ("software-version", r"\b\d+\.\d+\.\d+\b"),
    # A journal locator inside a prose citation note: the 19, 469 and 478 of
    # "Computers \& Operations Research} 19(6):469--478".
    ("citation-locator", r"\b\d+\(\d+\):\d+(?:--|-)\d+"),
]


# Written once, cited by the three ``discussion.paired.2d_nn.*`` entries.
_NN31F_REASON_TEXT = (
    "Repointed from NN_V3 to the retrained 31-feature control NN_31F on "
    "2026-08-11. NN_31F is scored by paper_tooling/score_31f_controls.py into "
    "paper_tooling/controls_31f/paired.csv, row stratum=2d, model_a=GART_2.0, "
    "model_b=NN_31F: mean_diff +0.5605555, ci_lo +0.4755659, ci_hi +0.6570722, "
    "wilcoxon_p 1.735e-33. That file is not exported into paper_numbers.json, "
    "so there is no bank key to point at. Settle by extending "
    "paper_tooling/paired_bank.py to export controls_31f/*.csv under "
    "paired_31f_* keys, then restore expect= on all three entries."
)


# ---------------------------------------------------------------------------
# CLAIMS
#
# Two blocks.
#
# 1. SEED SET -- 13 entries chosen to exercise every mechanism of the checker
#    (bank key, sidecar key, derived expression, scaled expression, each
#    tolerance mode, {~} hand-off, and no_generator).
# 2. SIGNIFICANCE -- every paired-test number the manuscript prints, registered
#    against the ``paired_*`` bank keys that ``paper_tooling/paired_bank.py``
#    now exports from ``tables/paired_tests.csv``.  Before that export the file
#    was written and never read, so not one p-value or bootstrap interval in
#    the paper had a generator to check against.
#
# Populating the remaining claims catalogued in prose_claim_map.md is a
# separate job; that mapping records which exports collapse the rest in blocks.
#
# Most of these are expected to report MISMATCHED right now.  That is the
# correct output: the manuscript prose is stale and has not been rewritten yet.
# The checker's job is to say so; rewriting is deliberately not done here,
# because the arm A decision will move the values again.
# Feature counts of the two predecessor-vector controls.  Read off the released
# artifacts, not off any table: the "(same features)" label in the results
# tables is inaccurate and is corrected in the prose instead, because the label
# is the join key ``build_paper_tables.py --check`` differs on.
_LINEAR_V3_NFEAT_REASON = (
    "Input width of the Linear_V3 control: "
    "joblib.load('linear_model_v3/linear_alpha_model_v3.joblib').n_features_in_ "
    "== 28, and its feature_names_in_ omits bounding_hypervolume, node_density "
    "and greedy_nn_over_mst relative to GART 2.0's 31. No artifact exports it. "
    "Settle by having model_registry emit a control-arm sidecar with "
    "n_features_in_ per baseline, then point at control_linear_v3_n_features."
)
_NN_V3_NFEAT_REASON = (
    "Input width of the NN_V3 control: "
    "torch.load('nn_est_alpha_v3/nn_alpha_v3_model.pt')['input_dim'] == 30 and "
    "state_dict['stem.weight'].shape == (192, 30); the missing input relative to "
    "GART 2.0's 31 is greedy_nn_over_mst. No artifact exports it. Settle as for "
    "_LINEAR_V3_NFEAT_REASON, under control_nn_v3_n_features."
)

# The MAPE-minimizing constant multiple of L_MST over the 23 screened
# non-EUC_2D instances.  Flagged UNGENERATED in prose_claims.py as
# app.oracle_constant_23; one exporter covers the 78, the 111 and the 23.
_ORACLE_REASON = (
    "MAPE-minimizing constant multiple of L_MST over the 23 screened non-EUC_2D "
    "TSPLIB95 instances: c* = 1.1718 reaching 7.55% MAPE. Recorded in "
    "paper_tooling/prose_claims.py as app.oracle_constant_23, state CORRECT / "
    "UNGENERATED. Settle with the oracle-constant exporter named there: "
    "oracle_constant_<set>_{c,mape_pct}, model-independent, covering the 78 "
    "EUC_2D, the 111 and these 23."
)

_ORACLE_78_REASON = (
    "MAPE-minimizing constant multiple of L_MST over the 78 TSPLIB EUC_2D "
    "instances: c* = 1.1275 reaching 3.52% MAPE. Same status as "
    "app.oracle_constant.c: settle with the oracle-constant exporter named in "
    "paper_tooling/prose_claims.py, oracle_constant_<set>_{c,mape_pct}, "
    "model-independent, covering the 78 EUC_2D, the 111 and the 23."
)

_ORACLE_111_REASON = (
    "MAPE-minimizing constant multiple of L_MST over the full 111-instance "
    "TSPLIB set: c* = 1.134, the value the fixed alpha=1.136 reference is "
    "compared against. Settle with the same oracle-constant exporter as "
    "app.oracle_constant.c, which covers the 78, the 111 and the 23."
)

_FIXED_ALPHA_REASON = (
    "Design constant of the Fixed_Alpha reference estimator, the multiplier "
    "1.136 hardcoded in the benchmark roster; the bank spells it inside its "
    "own key names (tsplib_nonEuc_total_fixed_alpha_1_136_*) but exports no "
    "key holding the value itself. A roster constant, not a measurement. "
    "Settle by banking the roster constants."
)

_BUDGET25_REASON = (
    "Ascent-budget rung k=25: the budget from which GART 2.0 is strictly "
    "better than the raw Volgenant--Jonker bound on both cost and accuracy in "
    "the plane, read off Table tab:frontier_2d and the two larger size bands "
    "of tab:sizestrat. A rung of the measured ladder, chosen for the "
    "sentence; no bank key names the first-dominating budget. Settle by "
    "exporting a first-dominating-budget key from the frontier bank."
)

_WORST_EXCESS_REASON = (
    "Worst relative excess of a released label over its own certified bound "
    "among the 145,013 B <= L checks: 5.8e-15, computed by "
    "paper_tooling/verify_repaired_labels.py, whose banked summary "
    "(frontier:labels/repair_verify_*) carries the check count and the "
    "violation count but not the worst excess. The prose prints the mantissa "
    "5.8 before a literal 10^{-15}. Settle by banking "
    "repair_verify_worst_excess alongside the other two."
)


def _generalization(row: str) -> str:
    """Reason text for a generalization-refit number: exact experiment and value."""
    return (
        f"paper_tooling/generalization_experiments.py: {row}. The refit "
        "protocol holds hyperparameters and seed fixed and varies only the "
        "training data; its summary is released with the results but not "
        "exported into paper_numbers.json, so there is no bank key. Settle by "
        "banking the E-series refit metrics under generalization_* keys."
    )


def _row31f(fname: str, row: str, value: str) -> str:
    """Reason text for a 31-feature-control number: exact file, row and value."""
    return (
        f"paper_tooling/controls_31f/{fname}, row {row}: {value}. Written by "
        "paper_tooling/score_31f_controls.py and not exported into "
        "paper_numbers.json, so there is no bank key. Settle by extending "
        "paper_tooling/paired_bank.py to export controls_31f/*.csv under "
        "controls_31f_* keys, then replace no_generator with expect."
    )


def _timing(row: str) -> str:
    """Reason text for a timing number: exact path inside the timing bank."""
    return (
        f"paper_tooling/gart2_timing_bank.json, {row}. One estimator per "
        "process, single thread, median of 11 repeats. That file is not "
        "exported into paper_numbers.json. Settle by banking "
        "tsplib_by_size_time_one_protocol under timing_* keys."
    )


def _probe(row: str) -> str:
    """Reason text for a swept-feature monotonicity number: exact file and row."""
    return (
        f"{row}. Ceteris-paribus sweep of one column over a log grid with every "
        "other feature held at its real value, 1,000 test instances at seed 42, "
        "tolerance 1e-9, produced by paper_tooling/v4_study.py::_sweep_monotonicity. "
        "Neither probe CSV is exported into paper_numbers.json, so there is no "
        "bank key. Settle by having v4_study.py write pct_nonincr_deployed into "
        "paper_numbers.json under probe_<model>_<axis>_pct_nonincr keys."
    )


# The released GART 2.0 booster is absent from v4_study._model_registry(), so the
# canonical v4_study_gart2_probe.csv has no row for it.  Re-measured under the
# identical protocol into v4_study_gart2_probe_shipped.csv.
_PROBE_SHIPPED = "paper_tooling/v4_study_gart2_probe_shipped.csv, model=GART_2.0"
_PROBE_V4 = "paper_tooling/v4_study_gart2_probe.csv, model=LGBM_V4"



def _LINENOISE(field: str) -> str:
    """Reason for a Line Noise on-face number: exact slice and recomputation."""
    return (
        f"{field}. Recomputed 2026-08-11 over the 2D benchmark's 210 "
        "Generalized_TSP_Analysis/instances/TSP-line_noise-n*.json files "
        "restricted to n>=200, which is 90 instances: on-face fraction is the "
        "share of points with an x or y coordinate equal to 0 or to the grid "
        "side G parsed from the instance name, and alpha is true_alpha from "
        "Generalized_TSP_Analysis/benchmark_checkpoints/base_ground_truth_2d.csv. "
        "paper_tooling/corpus_statistics.py::linenoise_geometry reads the same "
        "files but measures only rho_measured and kurtosis, so no artifact "
        "exports the on-face fraction or its correlation with alpha. Settle by "
        "having linenoise_geometry emit on_face_frac per instance and "
        "corpus_statistics bank the slice's median, its quartile alpha medians "
        "and its Spearman rho under corpus_linenoise_onface_* keys."
    )


_NN_V3_KWON_REASON = (
    "NN_V3 on the 80 Kwon-domain instances, recomputed with "
    "build_paper_tables.load_2d(): restrict to generator=='random', take the "
    "instance set of the Kwon_region rows, and evaluate err_pct for "
    "model=='NN_V3' -- MAPE 2.0576802%, SDPE 2.5472796% (ddof=1) on N=80. "
    "build_paper_tables.CLASSICAL_SUBDOMAINS scores only the anchor estimator, "
    "GART 2.0 and MST_Only in that bucket, so table_classical_matched.csv has no "
    "NN_V3 row there and there is no bank key. Settle by adding NN_V3 to the "
    "subdomain model list -- which would also add a printed row to tab:classical."
)

# ---------------------------------------------------------------------------
_V4_NFEAT_REASON = (
    "lgbm_model_v4/best_params_v4.json, field feature_set_full = 32: GART 2.0's "
    "31 booster features plus the raw MST length. model_registry.py exposes the "
    "production feature count as a sidecar read but has no equivalent for the "
    "V4 variant, so there is no bank key. Settle by adding the V4 sidecar to "
    "model_registry and exporting its feature count alongside sidecar:n_features."
)


def _MATCHED(model: str, panel: str, value: str) -> str:
    """Reason for a matched-domain number on an estimator with no printed row."""
    return (
        f"{model} on the {panel} panel: {value}. Recomputed with "
        "build_paper_tables.load_2d() restricted to generator=='random' and, for "
        "the two sub-panels, to the instance set of the anchor estimator's rows, "
        "exactly as build_paper_tables.compute_classical builds panel B; error is "
        "100*(pred_cost-true_cost)/true_cost and SDPE is its std with ddof=1. "
        "build_paper_tables.CLASSICAL_SUBDOMAINS scores only the anchor estimator, "
        "GART 2.0 and MST_Only, so table_classical_matched.csv has no row for this "
        "model and there is no bank key. Settle by adding the production-feature "
        "refits to the subdomain model list, which would also add printed rows to "
        "tab:classical."
    )


# _RANK31F was the reason text for the two 31-feature close-pair numbers while
# no rank row was released for them. It was settled on 2026-08-11 the way it
# said to settle it: build_paper_tables.with_31f_controls merges
# controls_31f/rows_tsplib.csv into the rank input, so table_rank.csv carries
# NN_31F and Linear_31F and both claims now hold a bank key. The helper had no
# other caller and is gone with them.


def _V4COST(value: str) -> str:
    """Reason for a number read from the V4 extractor cost micro-benchmark."""
    return (
        f"paper_tooling/v4_study_cost.csv: {value}. Written by "
        "paper_tooling/v4_study.py; four cases, reps=3 except pla85900 at reps=1. "
        "Not exported into paper_numbers.json. Settle by banking the extractor "
        "cost ratios under v4_cost_* keys."
    )

_LARGE_BUCKET = (
    "Bucket boundary, not a measurement: the lower edge of the '$n>400$' row of "
    "build_paper_tables.B_TSPLIB, _rng('n', 401, 10**9). Bucket edges are design "
    "constants and are not written to paper_numbers.json, which banks the cells "
    "they produce. Settle by emitting the bucket edges alongside bucket_count."
)

_SIZE_BUCKET = (
    "Bucket boundary, not a measurement: an edge of the [201,500] row of "
    "build_paper_tables.B_ND_SIZE, _size_buckets([... (201,500) ...]). Same class "
    "as _LARGE_BUCKET; settle the same way."
)


def _GATE_CONST(which: str) -> str:
    """Reason for an endpoint of the released greedy-ratio coverage gate."""
    return (
        f"lgbm_model_v3/feature_engineering_gart2.py::TRAIN_GREEDY_RANGE = "
        f"(1.035, 2.209), {which}. Its docstring states the interval is taken over "
        "the FULL corpus (tsp_features_v4.csv, 106,272 rows), not over the training "
        "split, whose own range is [1.046482, 2.129495]; the manuscript now says so. "
        "The constant is hardcoded in the inference module and no artifact exports "
        "it. Settle by having corpus_statistics.py emit both ranges under "
        "corpus_greedy_range_*."
    )


CLAIMS: list[Claim] = [
    # -- size-stratified multidimensional frontier (Section 5.3) -------------
    # Generator: paper_tooling/size_stratified_analyze.py, which pairs the
    # Polyak sweep's per-cell accuracy with the size-matched cost sample from
    # paper_tooling/d3_matched_timing.py.
    Claim(
        id="sizestrat.corpus.frac_small_pct",
        anchor=r"10{,}569, or {v}\%, carry at most 100 nodes",
        expect="sizestrat:corpus/frac_n_le_100_pct",
        note="Share of the scored ND split at n <= 100; the composition that "
             "makes the corpus aggregate a small-instance statement.",
    ),
    Claim(
        id="sizestrat.corpus.small_instances",
        anchor=r"Of the {~} scored instances {v}, or 62.7",
        expect="sizestrat:corpus/n_le_100_instances",
    ),
    Claim(
        id="sizestrat.corpus.scored",
        anchor=r"Of the {v} scored instances {~}, or 62.7",
        expect="sizestrat:corpus/scored_instances",
    ),
    # withdrawn sizestrat.summary.undominated_cells: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn sizestrat.summary.total_cells: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn sizestrat.d2.large.mape: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn sizestrat.d2.large.ms: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    Claim(
        id="sizestrat.d2.large.k500_mape",
        anchor=r"The bound reads {v}\% MAPE at {~}~ms for $k=500$ at $d=2$",
        expect="sizestrat:cell/d2/n600_1000/bound_k500_mape_pct",
    ),
    Claim(
        id="sizestrat.d2.large.k500_ms",
        anchor=r"The bound reads {~}\% MAPE at {v}~ms for $k=500$ at $d=2$",
        expect="sizestrat:cell/d2/n600_1000/bound_k500_ms",
    ),
    # withdrawn sizestrat.d3.large.mape: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn sizestrat.d3.large.ms: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    Claim(
        id="sizestrat.d3.large.k200_mape",
        anchor=r"{v}\% at {~}~ms for $k=200$ at $d=3$",
        expect="sizestrat:cell/d3/n600_1000/bound_k200_mape_pct",
    ),
    Claim(
        id="sizestrat.d3.large.k200_ms",
        anchor=r"{~}\% at {v}~ms for $k=200$ at $d=3$",
        expect="sizestrat:cell/d3/n600_1000/bound_k200_ms",
    ),
    # withdrawn sizestrat.d4.large.mape: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn sizestrat.d4.large.ms: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    Claim(
        id="sizestrat.d4.large.k200_mape",
        anchor=r"and {v}\% at {~}~ms for the same budget at $d=4$",
        expect="sizestrat:cell/d4/n600_1000/bound_k200_mape_pct",
    ),
    Claim(
        id="sizestrat.d4.large.k200_ms",
        anchor=r"and {~}\% at {v}~ms for the same budget at $d=4$",
        expect="sizestrat:cell/d4/n600_1000/bound_k200_ms",
    ),
    Claim(
        id="sizestrat.d100.large.mape",
        anchor=r"its own accuracy degrades to {v}\% against {~}\% at $d=50$",
        expect="sizestrat:cell/d100/n600_1000/gart_mape_pct",
    ),
    Claim(
        id="sizestrat.d50.large.mape",
        anchor=r"its own accuracy degrades to {~}\% against {v}\% at $d=50$",
        expect="sizestrat:cell/d50/n600_1000/gart_mape_pct",
    ),
    # -- corpus sizes: integers, exact tolerance, two different sources -------
    Claim(
        id="abstract.corpus.n_nd",
        anchor=r"The multidimensional benchmark is {v} held-out instances",
        expect="sidecar:rows.test",
        tol="exact",
        note="Held-out split size, read from the frozen artifact rather than the "
             "benchmark CSV so it tracks whatever model ships.",
    ),

    # -- headline metrics: printed-precision tolerance, {~} hand-off ----------

    # -- derived: ratio over two bank keys ------------------------------------

    # -- derived: percentage-point difference ---------------------------------

    # -- derived + scaled: a fraction quoted as a percentage ------------------

    # -- model-artifact facts: sidecar source, follows the production model ---
    # withdrawn methods.model.n_features: section roadmap sentence deleted; the 31-feature count is asserted at the start of subsec:features and in subsec:shap
    # withdrawn methods.model.n_trees: cut in the 2026-09-03 audit pass (p09, author-approved small cut); the tree count now lives in the hyperparameter table only
    # withdrawn methods.model.leaves_per_tree: cut in the 2026-09-03 audit pass (p09); the leaf cap now lives in the hyperparameter table only

    # -- a genuinely unbackable number, registered with what would settle it --
    Claim(
        id="theory.alpha_sd_at_d2",
        anchor=r"falls monotonically from {v} at $d=2$",
        no_generator=(
            "Training-corpus dispersion of alpha, not a model output; no current "
            "generator emits it. Settle with a groupby on tsp_features_v4.csv "
            "restricted to split=='train', then promote to a bank key."
        ),
    ),

    # -- printed-precision on a one-decimal metric, region-variant sensitive --
    # withdrawn metrics.bhh_region.mspe_uniform: metric-justification illustration cut per author; the values stay machine-checked in the classical tables
    # withdrawn metrics.bhh_region.sdpe_uniform: metric-justification illustration cut per author; the values stay machine-checked in the classical tables

    # -----------------------------------------------------------------------
    # SIGNIFICANCE
    #
    # Sign convention, inherited from build_paper_tables.paired_test and carried
    # unchanged into the bank: mean_diff / ci_lo / ci_hi are |APE| of the
    # production model minus |APE| of the baseline, so a NEGATIVE value favors
    # the production model. The manuscript states two of these comparisons from
    # the *baseline's* point of view, which is why those entries negate in
    # ``expect`` rather than pointing at some other key. Negating in the
    # expression is deliberate: it keeps one generator per quantity and puts the
    # orientation in the derivation line of a mismatch report, where a reader
    # can see it.
    #
    # The sign also lives in the anchor as literal LaTeX ($-{v}$, $[-{v},+{~}]$).
    # If a future run flips a sign the anchor stops matching and the entry is
    # reported ANCHOR_MISSING -- loud, and the correct outcome, because a
    # sentence that says "GART is behind by" when it is now ahead needs a human,
    # not a re-rounded number.
    # -----------------------------------------------------------------------
    Claim(
        id="discussion.paired.tsplib_asym.mean_diff",
        anchor=r"the asymptotic MST ratio is $-{v}$ percentage points",
        expect="= -1 * {paired_tsplib_by_size_total_asymptotic_mst_ratio_mean_diff}",
        tol=("dp", 2),
        note="Also the premise of the two qualitative sentences that hang off "
             "this test -- 'not statistically distinguishable from zero' in the "
             "abstract and 'not statistically distinguishable from zero' in the "
             "conclusion. Neither carries a numeral of its own, so neither can "
             "be anchored; if this entry or the p-value below moves, both "
             "sentences need re-reading.",
    ),
    Claim(
        id="discussion.paired.tsplib_asym.ci_lo",
        anchor=r"percentage points with a 95\% interval of $[-{v},-{~}]$",
        expect="= -1 * {paired_tsplib_by_size_total_asymptotic_mst_ratio_ci_lo}",
        tol=("dp", 2),
        note="Anchor repointed 2026-08-11. The interval no longer straddles zero "
             "-- it is [-1.77, -0.22] -- so the old '$[-{v},+{~}]$' anchor stopped "
             "matching, which is the loud failure the sign-in-the-anchor design "
             "was built for.",
    ),
    Claim(
        id="discussion.paired.tsplib_asym.ci_hi",
        anchor=r"percentage points with a 95\% interval of $[-{~},-{v}]$",
        expect="= -1 * {paired_tsplib_by_size_total_asymptotic_mst_ratio_ci_hi}",
        tol=("dp", 2),
        note="Both ends are now negative and the sentence states the comparison "
             "from the baseline's point of view, so both ends negate. Before "
             "2026-08-11 the upper end was positive and was quoted in the table's "
             "own orientation without negation.",
    ),
    Claim(
        id="discussion.paired.tsplib_asym.p",
        anchor=r"and $p={v}$, so the MAPE advantage is small",
        expect="bank:paired_tsplib_by_size_total_asymptotic_mst_ratio_wilcoxon_p",
        tol=("abs", 0.0001),  # p-values print at two figures; exempt from 3 s.f. (author, 2026-09-03)
    ),
    # The three entries below were repointed on 2026-08-11 from NN_V3 to the
    # retrained 31-feature control NN_31F.  The old control was defective, not
    # merely stale: it was fitted on the predecessor's 30-input vector and on
    # the superseded label table, it computed its own feature block instead of
    # calling the production extractor, and it exposed no ``predict_alpha`` so
    # the hybrid builder recorded ``AttributeError`` on 29 of 111 TSPLIB
    # instances.  Its true paired difference on 2D (-0.0863, Wilcoxon p = 0.44)
    # favoured GART 2.0, so the sentence these entries anchored -- "the network
    # is more accurate ... by 0.34 percentage points" -- was wrong in direction
    # and not significant either way.  Refitted on the production vector the
    # network wins by +0.5606 points at p = 1.7e-33.
    #
    # NN_31F is not in paper_numbers.json, so these three have no bank key and
    # are registered ``no_generator`` rather than left silently unchecked. The
    # reason names the exact row that settles each one.
    Claim(
        id="results.nd.mape",
        anchor=r"GART 2.0 obtains {v}\% MAPE and {~}\% SDPE overall (Table~\ref{tab:results_nd}), a factor",
        expect="bank:nd_by_dim_total_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.nd.sdpe",
        anchor=r"GART 2.0 obtains {~}\% MAPE and {v}\% SDPE overall (Table~\ref{tab:results_nd}), a factor",
        expect="bank:nd_by_dim_total_gart_2_0_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.2d.mape",
        anchor=r"GART 2.0 obtains {v}\% MAPE and {~}\% SDPE overall (Table~\ref{tab:results_2d}, Figure~\ref{fig:boxplot_2d}), roughly half",
        expect="bank:2d_by_size_total_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.2d.sdpe",
        anchor=r"GART 2.0 obtains {~}\% MAPE and {v}\% SDPE overall (Table~\ref{tab:results_2d}, Figure~\ref{fig:boxplot_2d}), roughly half",
        expect="bank:2d_by_size_total_gart_2_0_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.sdpe",
        anchor=r"benchmark GART 2.0 obtains {~}\% MAPE and {v}\% SDPE",
        expect="bank:tsplib_by_size_total_gart_2_0_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.mape",
        anchor=r"benchmark GART 2.0 obtains {v}\% MAPE and {~}\% SDPE",
        expect="bank:tsplib_by_size_total_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    # matched.kwon_domain.gart_mape and matched.chien_domain.gart_mape are
    # deleted, not re-pointed. They quoted GART 2.0 on the Chien and Kwon fitted
    # node ranges; both estimators are withdrawn (unobtainable primaries), the
    # sub-domain panels they anchored are gone from tab:classical, and the
    # sentence now reports the one remaining matched panel.
    Claim(
        id="matched.uniform_domain.gart_mape",
        anchor=r"uniform instances it obtains {v}\% MAPE against",
        expect="bank:classical_b_random_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.uniform_domain.cavdar_mape",
        anchor=r"against \c{C}avdar--Sokol's {v}\%, BHH's",
        expect="bank:classical_b_random_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.uniform_domain.bhh_mape",
        anchor=r"\c{C}avdar--Sokol's {~}\%, BHH's {v}\% and the",
        expect="bank:classical_b_random_bhh_sampling_region_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.uniform_domain.floor_mape",
        anchor=r"and the $\alpha=1$ floor's {v}\%.",
        expect="bank:classical_b_random_l_mathrm_mst_alpha_1_mape_pct",
        tol="printed",
    ),
    Claim(
        id="discussion.r2alpha.tsplib_gt400",
        anchor=r"TSPLIB bucket has $R^2_\alpha=-{v}$. The model predicts",
        expect="= -1 * {tsplib_by_size_gt400_gart_2_0_r2_alpha}",
        tol=("dp", 3),
        note="Adverse result: negative R^2_alpha means the model predicts alpha "
             "worse than the bucket mean. The minus sign is in the anchor, so a "
             "sign flip reports ANCHOR_MISSING rather than a re-rounded number.",
    ),

    # -----------------------------------------------------------------------
    # SHAP SHARES, registered 2026-08-11.
    #
    # Every share is stated twice, in Section 3.6 and again in Appendix
    # app:shap, and before this rewrite the two agreed only because both were
    # the predecessor's. Both copies are anchored so they cannot drift apart:
    # a future SHAP re-run that updates one and not the other fails here.
    # The bank carries 121 shap_* keys from paper_tooling/shap_production.py.
    # -----------------------------------------------------------------------
    Claim(
        id="methods.shap.dominance_share",
        anchor=r"the MST dominance ratio contributes {v}\% of the total and",
        expect="bank:shap_feature_mst_dominance_ratio_share_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="methods.shap.greedy_share",
        anchor=r"of the total and \texttt{greedy\_nn\_over\_mst} {v}\%",
        expect="bank:shap_feature_greedy_nn_over_mst_share_pct",
        tol=("dp", 1),
    ),
    # withdrawn methods.shap.top2_share: cut in the verbosity sweep; the two component shares remain in prose and tab:shap_top carries the full ranking
    Claim(
        id="methods.shap.size_dimension_share",
        anchor=r"Node count and dimension jointly contribute {v}\%. Appendix",
        expect="bank:shap_family_size_dimension_share_pct",
        tol=("dp", 1),
    ),
    # withdrawn methods.shap.bounding_share: third-tier share breakdown cut from prose; the full ranking remains in Table tab:shap_top
    # withdrawn methods.shap.centroid_share: third-tier share breakdown cut from prose; the full ranking remains in Table tab:shap_top
    Claim(
        id="appendix.shap.group_mst_share",
        anchor=r"the nineteen MST-derived features carry {v}\%",
        expect="bank:shap_group_mst_share_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="appendix.shap.group_geometric_share",
        anchor=r"the eleven geometric features {v}\%",
        expect="bank:shap_group_geometric_share_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="appendix.shap.group_greedy_share",
        anchor=r"the single constructive ratio {v}\%",
        expect="bank:shap_group_greedy_share_pct",
        tol=("dp", 1),
    ),
    # -----------------------------------------------------------------------
    # referee pass, 2026-08-11: R1-R6 + the non-Euclidean omission
    #
    # Every numeral written or re-keyed by scratchpad/patch_referee7*.py is
    # registered here.  Where paper_numbers.json carries the quantity the entry
    # points at the bank key and is checked; where it does not -- the 31-feature
    # controls, the two baseline feature counts, the timing bank, the oracle
    # constant, the MDS stress table and the manuscript cell count -- the entry
    # is ``no_generator`` and its reason names the exact artifact and row.
    # -----------------------------------------------------------------------

    # -- abstract ------------------------------------------------------------

    # -- introduction --------------------------------------------------------

    # bench.controls.production_features is withdrawn with the sentence it
    # anchored: Section 4.1 no longer describes the 30-feature predecessor as a
    # reported row, so "the same learner on 30 of the 31 inputs" is gone. The
    # production feature count is still checked at its other sites.
    # withdrawn app.oracle_constant.c: caption statement removed; the body assertions are tracked by application.oracle_c_23_restated and application.oracle_mape_23_total
    # withdrawn app.oracle_constant.mape: caption statement removed; the body assertions are tracked by application.oracle_mape_23_restated and application.oracle_mape_23_total

    # =======================================================================
    # Re-audit pass of 2026-08-11: seven findings closed.  Every number the
    # rewrite introduced is registered here rather than baselined.
    # =======================================================================

    # -- N1: the 2D superlative.  NN_V3 is one of the thirteen enumerated
    #    baselines and a printed row of tab:2d_by_size, and its aggregate SDPE
    #    is below GART 2.0's, so "lowest MAPE and SDPE of every baseline except
    #    the refitted network" was false on the 2D stratum.
    # withdrawn discussion.timing.classical_lo: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn discussion.timing.classical_hi: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone

    # -- N4: SDPE is the primary precision metric (Section 4.3), and on the
    #    Kwon domain the V3-feature network's SDPE is below GART 2.0's, so
    #    "wins on every matched domain" failed on the paper's own metric.
    # withdrawn provenance.z.n_cells: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.n_rows: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.min_reference: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.n_over: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.n_cells_again: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.threshold: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.rows_over: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.n_over_again: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.n_rows_again: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.max: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.z.max_d: journey narrative removed per author directive (editorial restructure)

    # -- N7: GART 2.0 (V4 features).  Scored on every benchmark, present in the
    #    tidy tables, absent from the manuscript, and better than the released
    #    model on three of the four strata.  Enumerated here with the reason it
    #    was not released.  The released model's own probe row did not exist --
    #    v4_study._model_registry() omits PRODUCTION_BOOSTER -- so it was
    #    re-measured under the identical protocol.

    # -- appendices ----------------------------------------------------------
    Claim(
        id="appendix.code.table_cells",
        anchor=r"re-derives all {v} benchmark table cells",
        expect="bank:manuscript_table_cells",
        tol="exact",
        note="Settled. build_paper_tables.py --check now writes its own banner "
             "count to paper_numbers.json as manuscript_table_cells, so this is "
             "checked rather than typed. It had to be: withdrawing Daganzo, "
             "Chien, Kwon--Golden--Wasil and Cavdar_region cut 13 rows out of "
             "tab:classical, and while the figure was ungenerated a roster "
             "change of that size could move the count without anything saying "
             "so. Rose from 1,431 to 1,921 when --check was extended to "
             "tab:classical and tab:genclass, the two generated tables it had "
             "never read back.",
    ),
    Claim(
        id="datasets.2d.composition_caption_n",
        anchor=r"2D benchmark dataset composition ({v} instances)",
        expect="bank:2d_by_size_total_gart_2_0_n",
        tol="exact",
        note="The caption of tab:dataset_counts. Its context hash moved when the "
             "table dropped its \\resizebox for wrapping columns, which surfaced "
             "it as unregistered; registered against the bank rather than "
             "recorded, since the benchmark size is a generated quantity.",
    ),
    Claim(
        id="appendix.code.frontier_table_cells",
        anchor=r"does the same for the {v} cells of the cost/accuracy",
        expect="bank:frontier_table_cells",
        tol="exact",
        note="check_frontier_tables.py banks its own banner count the same way "
             "build_paper_tables.py --check does, so the second of the two "
             "verification counts the appendix quotes is generated rather than "
             "typed.",
    ),
    Claim(
        id="appendix.mds.explicit_gart_mape",
        anchor=r"and obtains {v}\% MAPE on the six it accepts",
        expect="bank:tsplib_nonEuc_explicit_gart_2_0_mape_pct",
        tol=("dp", 2),
        note="R1: the EXPLICIT stratum is GART 2.0's own row, N=6. The "
             "seven-instance figure it was transposed with before 2026-08-11 "
             "belonged to a row the manuscript no longer reports.",
    ),
    Claim(
        id="appendix.mds.geo_stress_max",
        anchor=r"maximum stress {v} for GEO against {~} for screened EXPLICIT",
        no_generator=(
            "Maximum normalized_distance_stress over the GEO rows of "
            "paper_reference/mds_distortion_screened.csv, written by "
            "paper_tooling/audit_mds_distortion.py: 0.2362897 at the GEO maximum "
            "against 0.1208360 for screened EXPLICIT. That CSV is not exported "
            "into paper_numbers.json. Settle by banking per-type stress extrema "
            "as mds_stress_<type>_max."
        ),
        tol=("dp", 4),
    ),
    # =======================================================================
    # Open-baseline-set pass.  The manuscript enumerated thirteen baselines and
    # excluded its two strongest comparators, NN_31F and LGBM_V4, so every
    # "lowest of every baseline" sentence held by enumeration.  Both are now in
    # the set, the superiority claims are re-founded on the consistency
    # criterion, and every number that move introduced is registered below.
    # The swept-feature probe numbers now have real bank keys (cons_*), written
    # by paper_tooling/consistency_bank.py, so they use expect rather than
    # no_generator.
    # =======================================================================

    # -- abstract: the V4 feature count and the open-set consistency claim ----
    # The same count, now also carried by the row's display name. The label was
    # "GART 2.0 (V4 features)" until the internal version tags were taken out of
    # the manuscript; "32-feature variant" says what V4 meant, and puts a
    # checkable numeral at both sites that name the row.
    Claim(
        id="methods.probe.twin_dim",
        anchor=r"against {v}\% of the dimension sweeps and",
        expect="bank:cons_probe_gart2_logit_v3hp_dimension_pct_nonincr_deployed",
        tol=("dp", 1),
    ),
    Claim(
        id="methods.probe.twin_n",
        anchor=r"and {v}\% of the size sweeps for the same fit with the constraints removed",
        expect="bank:cons_probe_gart2_logit_v3hp_n_customers_pct_nonincr_deployed",
        tol=("dp", 1),
    ),
    # -- Section 3.4, within-cell SHAP decomposition (shap_by_dimension.py) --
    Claim(
        id="shap.within.greedy_d2",
        anchor=r"greedy-to-MST ratio holds {v}\% of the SHAP variance at $d=2$",
        expect="bank:shap_band_d2_within_share_greedy_nn_over_mst_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="shap.within.dominance_d2",
        anchor=r"the MST dominance ratio holds {v}\% to {~}\%",
        expect="bank:shap_band_d2_within_share_mst_dominance_ratio_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="shap.within.dominance_d100",
        anchor=r"the MST dominance ratio holds {~}\% to {v}\%",
        expect="bank:shap_band_d100_within_share_mst_dominance_ratio_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="shap.within.r2_alpha_d2",
        anchor=r"from $R^2$ {v} at $d=2$ to {~} at the withheld $d=100$",
        expect="bank:shap_band_d2_r2_within_alpha",
        tol=("dp", 3),
    ),
    Claim(
        id="shap.within.r2_alpha_d100",
        anchor=r"from $R^2$ {~} at $d=2$ to {v} at the withheld $d=100$",
        expect="bank:shap_band_d100_r2_within_alpha",
        tol=("dp", 3),
    ),
    Claim(
        id="shap.within.mix_d2",
        anchor=r"The generator mix explains {v}\% of the within-cell target at $d=2$",
        expect="bank:shap_band_d2_target_r2_on_mix_pct",
        tol=("dp", 1),
    ),

    # -- Section 4.4: the screen count at the two 2D sites -------------------
    # The GART 2.0 side of that same comparison IS bankable, so it is checked
    # rather than left to the ablation's no_generator note.
    # matched.v4.{kwon,chien}_{mape,sdpe} are deleted with their panels. The
    # sentence reported the extended-block ablation on three matched panels; the
    # Chien and Kwon panels required those two estimators' fitted node ranges and
    # both estimators are withdrawn, so one panel is left.
    # matched.v4.p_kwon and matched.v4.p_chien are deleted with their panels.

    # -- Section 4.6: close-pair ordering over the open roster ---------------

    # -- Section 4.6: the V4 cost comparison, now single-protocol ------------

    # -- Section 5: the accepted count and the V4 row ------------------------
    # withdrawn application.noneuc.screen_n: Total-row readback cut per author; the values are printed and machine-checked in tab:tsplib_nonEuc
    # withdrawn application.noneuc.accept_n: Total-row readback cut per author; the values are printed and machine-checked in tab:tsplib_nonEuc

    # -- conclusion ----------------------------------------------------------
    # -- pre-existing numbers the open-set pass re-keyed ---------------------
    # None of these values or sentences changed; they re-key because the
    # checker addresses an occurrence by a digest of the text around it and the
    # open-set rewrite moved that text. They were in the recorded backlog. They
    # are registered rather than re-baselined so the next context shift cannot
    # un-key them a second time.

    # -- Section 4.4 / 4.5, rewritten for the two surviving classical rows ---
    # Every numeral in those two paragraphs is registered here. They were
    # rewritten wholesale when Daganzo, Chien and Kwon--Golden--Wasil were
    # withdrawn, so none of them can ride on the recorded backlog.
    # withdrawn results.tsplib.cavdar_mape: cut in the 2026-09-03 audit pass (p16, table readback); the value is Table tab:results_tsplib's
    # withdrawn results.tsplib.bhh_mape: cut in the 2026-09-03 audit pass (p16, table readback); the value is Table tab:results_tsplib's
    # withdrawn results.tsplib.bhh_mspe: cut in the 2026-09-03 audit pass (p16, table readback); the value is Table tab:results_tsplib's
    # withdrawn results.tsplib.cavdar_mspe: cut in the 2026-09-03 audit pass (p16, table readback); the value is Table tab:results_tsplib's
    Claim(
        id="matched.bhh.full_mape",
        anchor=r"BHH falls from {v}\% MAPE on the full 2D benchmark",
        expect="bank:classical_a_2d_bhh_mape_pct",
        tol="printed",
    ),
    Claim(
        id="matched.bhh.uniform_mape",
        anchor=r"on the full 2D benchmark to {v}\% here",
        expect="bank:classical_b_random_bhh_sampling_region_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.bhh.uniform_mspe",
        anchor=r"a systematic underprediction, $-{v}$\% signed against",
        expect="= -1 * {classical_b_random_bhh_sampling_region_mspe_pct}",
        tol=("dp", 2),
        note="Adverse result kept explicit: BHH's residual on its own matched "
             "domain is a bias, not noise -- the signed error exceeds the "
             "dispersion. Printed as a magnitude after a literal minus sign, "
             "hence the sign flip.",
    ),
    Claim(
        id="matched.bhh.uniform_sdpe",
        anchor=r"$-8.65$\% signed against {v}\% SDPE, the direction",
        expect="bank:classical_b_random_bhh_sampling_region_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.cavdar.full_mape",
        anchor=r"\c{C}avdar--Sokol from {v}\% to {~}\%. \c{C}avdar--Sokol itself",
        expect="bank:classical_a_2d_cavdar_sokol_mape_pct",
        tol="printed",
    ),
    Claim(
        id="matched.cavdar.uniform_mape",
        anchor=r"\c{C}avdar--Sokol from {~}\% to {v}\%. \c{C}avdar--Sokol itself",
        expect="bank:classical_b_random_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.cavdar.uniform_sdpe",
        anchor=r"at the widest dispersion in the panel, {v}\% SDPE",
        expect="bank:classical_b_random_cavdar_sokol_sdpe_pct",
        tol="printed",
        note="Adverse result: the widest dispersion in the matched panel, "
             "wider than the alpha=1 floor's. Registered so a later run cannot "
             "quietly drop the qualification.",
    ),
    Claim(
        id="matched.cavdar.uniform_medape",
        anchor=r"the lower median absolute error here, {v}\%, at the widest",
        expect="bank:classical_b_random_cavdar_sokol_medape_pct",
        tol=("dp", 2),
        note="The anchor read 'under a third' until the final verification "
             "pass. 2.84/8.16 = 0.348, which is over a third. The relation is a "
             "word, so no numeric check could have caught it; the wording now "
             "matches the two numbers either side of it.",
    ),
    Claim(
        id="matched.uniform_domain.n",
        anchor=r"on both metrics. On the {v} uniform instances it obtains",
        expect="bank:classical_b_random_gart_2_0_n",
        tol="exact",
    ),
    # -- Appendix: Cavdar--Sokol's Eq. (21), bounded to its fitted range -----
    # Backed by paper_tooling/cavdar_correction_bank.py, which reads the
    # constants off CavdarSokol itself, so changing the implemented correction
    # moves these sentences instead of silently disagreeing with them.
    Claim(
        id="appendix.cavdar.n_below_fit",
        anchor=r"binds on this benchmark: {v} of the {~} 2D instances have",
        expect="bank:cavdar_corr_2d_n_below_min",
        tol="exact",
        note="Adverse disclosure: the source's own correction is out of range on "
             "51.16% of the 2D benchmark, so for slightly over half of it the "
             "ratio is held at the n=100 endpoint. Registered so the "
             "qualification cannot be dropped in a later pass.",
    ),
    Claim(
        id="appendix.cavdar.benchmark_n",
        anchor=r"binds on this benchmark: {~} of the {v} 2D instances have",
        expect="bank:cavdar_corr_2d_n_total",
        tol="exact",
    ),
    Claim(
        id="appendix.cavdar.extrap_ratio_5000",
        anchor=r"grows without bound ($E/T={v}$ at",
        expect="bank:cavdar_corr_ratio_extrap_5000",
        tol=("dp", 2),
    ),
    Claim(
        id="appendix.cavdar.boundary_step",
        anchor=r"leaving a {v}\% step at the upper boundary",
        expect="bank:cavdar_corr_step_at_n_max_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="appendix.cavdar.boundary_ratio",
        anchor=r"where the fitted ratio reads {v}.",
        expect="bank:cavdar_corr_ratio_at_n_max",
        tol=("dp", 3),
    ),
    # withdrawn appendix.classical.panel_b_n: Appendix H deleted (eyeball pass 2026-09-04); the 210 survives under matched.uniform_domain.n
    # withdrawn matched.cavdar_factor: cut in the verbosity sweep; the factor is arithmetic over the 1.31 and 8.16 MAPE figures in the same sentence
    Claim(
        id="discussion.rank.tsplib_close5_gart1",
        anchor=r"GART 1.0 trails GART 2.0 on the TSPLIB close-pair statistic at the tighter threshold, at {v}\%.",
        expect="bank:rank_tsplib_euc2d_gart_1_0_close5_pct",
        tol=("dp", 1),
        note="Re-anchored after the editorial restructure: the close-pair sentence "
             "was rewritten and this is the one GART 1.0 figure it still prints.",
    ),
    # withdrawn discussion.rank.tsplib_close5_pairs: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn application.noneuc.gart_sdpe: Total-row readback cut per author; the values are printed and machine-checked in tab:tsplib_nonEuc
    # =======================================================================
    # Structural pass of 2026-08-11: S1 (LGBM_V3 admitted to the enumerated
    # baseline set, sixteen -> seventeen), S2 (`grid` split out of the
    # Geometric Struct. aggregate and named as the second generator absent
    # from training), and findings N-A, N-D, N-E, N-G, N-H, N-I, N-J.
    #
    # Every numeral those edits wrote is registered below. The bank gained
    # per-row keys for the split (2d_by_genclass_geometric_grid_* and
    # _geometric_other_*) and for the two 31-feature refits in the rank table,
    # so most of these are `expect` rather than `no_generator`.
    # =======================================================================

    # -- N-G: the timed field is fourteen, not the eight of one table --------

    # -- N-I: the coverage gate is a corpus range, not a training range ------
    Claim(
        id="methods.greedy_gate.corpus_rows",
        anchor=r"the range the ratio spans over the full {v}-row corpus, training, validation and test",
        no_generator=(
            "Row count of tsp_features_v4.csv, the table "
            "lgbm_model_v3/gart2_final.json names as training_table: 106,272. The "
            "corpus size is asserted elsewhere in the manuscript from the same file "
            "and is not exported into paper_numbers.json. Settle by banking "
            "len(tsp_features_v4.csv) under corpus_n_rows."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.greedy_gate.train_lo",
        anchor=r"The training split alone spans the narrower $[{v},2.1295]$",
        no_generator=(
            "Minimum of greedy_nn_over_mst over split=='train' in tsp_features_v4.csv "
            "(69,768 rows): 1.046482. The released constant "
            "feature_engineering_gart2.TRAIN_GREEDY_RANGE = (1.035, 2.209) is the "
            "range over the FULL corpus, as its own docstring states; this entry "
            "records the training split's own range, which the manuscript now "
            "distinguishes from it. No artifact exports either. Settle by having "
            "corpus_statistics.py emit both under corpus_greedy_range_*."
        ),
        tol=("dp", 4),
    ),
    Claim(
        id="methods.greedy_gate.train_hi",
        anchor=r"The training split alone spans the narrower $[{~},{v}]$",
        no_generator=(
            "Maximum of greedy_nn_over_mst over split=='train' in tsp_features_v4.csv "
            "(69,768 rows): 2.129495. Companion to methods.greedy_gate.train_lo; "
            "settle the same way."
        ),
        tol=("dp", 4),
    ),
    # withdrawn application.greedy_gate.train_lo: caption statement removed; the training-split range [1.0465,2.1295] is asserted in subsec:features

    # -- N-J: the tuning comparison, and which two boosters it is over -------
    Claim(
        id="methods.optuna.trials_complete",
        anchor=r"validation, {v} completed and {~} pruned",
        no_generator=(
            "lgbm_model_v3/gart2_optuna.db, study 'gart2': "
            "SELECT state, COUNT(*) FROM trials GROUP BY state -> COMPLETE 60, "
            "PRUNED 140, total 200. The study database is not read by any exporter. "
            "Settle by having a small exporter emit the trial census under "
            "optuna_gart2_trials_*."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.optuna.trials_pruned",
        anchor=r"validation, {~} completed and {v} pruned",
        no_generator=(
            "lgbm_model_v3/gart2_optuna.db, study 'gart2': PRUNED 140 of 200 trials. "
            "Companion to methods.optuna.trials_complete; settle the same way."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.optuna.tuned_nd",
        anchor=r"the tuned booster reaches {v}\% on the multidimensional test split",
        no_generator=(
            "paper_tooling/v4_study_allmodels_strata.csv, row model=GART2_logit_tuned, "
            "stratum=nd_test: mape 0.611239. Written by "
            "paper_tooling/v4_study.py::cmd_evalall over v4_study_feature_cache.csv, "
            "and not exported into paper_numbers.json. Settle by banking "
            "v4_study_allmodels_strata.csv under v4_strata_* keys."
        ),
        tol=("dp", 4),
    ),
    Claim(
        id="methods.optuna.untuned_nd",
        anchor=r"against {v}\% for the frozen hyperparameters",
        no_generator=(
            "paper_tooling/v4_study_allmodels_strata.csv, row model=GART2_logit_v3hp, "
            "stratum=nd_test: mape 0.622598. This is the control the tuning sentence "
            "is measured against -- same logit target, same unconstrained fit, "
            "hyperparameters the only difference -- and 0.622598 - 0.611239 = 0.011359 "
            "is the 0.011 the sentence quotes. Scoring the tuned booster against the "
            "SHIPPED model instead gives 0.008876, because the released model also "
            "carries the monotone constraint; that pairing would confound the search "
            "with the constraint. Settle by banking v4_study_allmodels_strata.csv."
        ),
        tol=("dp", 4),
    ),
    Claim(
        id="methods.optuna.tuned_tsplib",
        anchor=r"the frozen hyperparameters, and {v}\% against 2.47\% on TSPLIB EUC\_2D",
        no_generator=(
            "paper_tooling/v4_study_allmodels_strata.csv, row model=GART2_logit_tuned, "
            "stratum=tsplib_euc2d: mape 2.636384. Settle by banking "
            "v4_study_allmodels_strata.csv under v4_strata_* keys."
        ),
        tol=("dp", 4),
    ),
    Claim(
        id="methods.optuna.untuned_tsplib",
        anchor=r"and {~}\% against {v}\% on TSPLIB EUC\_2D",
        no_generator=(
            "paper_tooling/v4_study_allmodels_strata.csv, row model=GART2_logit_v3hp, "
            "stratum=tsplib_euc2d: mape 2.472478. 2.636384 - 2.472478 = 0.163906 is "
            "the 0.16 the sentence quotes. Settle by banking "
            "v4_study_allmodels_strata.csv under v4_strata_* keys."
        ),
        tol=("dp", 4),
    ),

    # -- N-D: the probe protocol, published with the claim it bounds ---------
    # withdrawn methods.probe.grid_points: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn methods.probe.d_lo: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn methods.probe.d_hi: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn methods.probe.n_lo: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn methods.probe.n_hi: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn methods.probe.tolerance_base: tolerance sentence cut in the condensation; PROBE_TOL remains a code constant and no prose asserts it

    # -- N-A: the rho(d) -> rho(d,n) gap is not the largest step -------------
    # withdrawn results_2d.grid_n: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn results_2d.grid_mape: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn results_2d.geom_other_mape: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn results_2d.geom_other_n: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    Claim(
        id="results_2d.grid_alpha_mean",
        anchor=r"Realized $\alpha$ averages {v} on \texttt{grid}",
        no_generator=(
            "Mean realised alpha = true_cost / mst_length over the 210 grid instances "
            "of the 2D benchmark: 1.0522, range [1.0104, 1.3900]. Recomputed with "
            "build_paper_tables.load_2d() restricted to generator=='grid' and "
            "model==GART. compute_table banks error metrics, not the realised target, "
            "so there is no key. Settle by banking per-bucket mean alpha alongside "
            "R2_alpha, which is already computed from the same two columns."
        ),
        tol=("dp", 2),
    ),
    Claim(
        id="results_2d.linenoise_alpha_mean",
        anchor=r"on \texttt{grid} against {v} on Line Noise",
        no_generator=(
            "Mean realised alpha over the 210 line_noise instances of the 2D "
            "benchmark: 1.5560, range [1.1403, 1.9950]. Same recomputation as "
            "results_2d.grid_alpha_mean with generator=='line_noise'. The two "
            "unrepresented generators bracket the corpus mean of 1.2418 from below "
            "and above, which is why their errors are signed in opposite directions. "
            "Settle the same way."
        ),
        tol=("dp", 2),
    ),

    # -- S2 / N-F: the per-class ladder in Section 4.6 -----------------------
    # withdrawn discussion.genclass.geom_other_mape: journey narrative removed per author directive (editorial restructure); the value is still checked at its Section 4.5 site by results_2d.geom_other_mape
    # withdrawn discussion.genclass.grid_mape: journey narrative removed per author directive (editorial restructure); the value is still checked at its Section 4.5 site by results_2d.grid_mape
    # withdrawn discussion.genclass.worst_represented: journey narrative removed per author directive (editorial restructure)
    # withdrawn discussion.genclass.grid_sdpe: journey narrative removed per author directive (editorial restructure)
    # withdrawn discussion.genclass.grid_floor_mape: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn appendix.genclass.geom_class_n: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.genclass.geom_class_mspe: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.genclass.geom_other_mspe: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.genclass.benchmark_n: caption sum-check sentence removed; the 2{,}580 total is asserted in subsec:datasets and the tab:frontier_2d caption

    # -- N-H: the network beats the extended-block variant on five of eight --
    Claim(
        id="methods.greedy_gate.corpus_lo",
        anchor=r"whose ratio falls outside $[{v},2.209]$",
        no_generator=_GATE_CONST("lower endpoint 1.035; the observed minimum is 1.035414"),
        tol=("dp", 3),
    ),
    Claim(
        id="methods.greedy_gate.corpus_hi",
        anchor=r"whose ratio falls outside $[{~},{v}]$",
        no_generator=_GATE_CONST("upper endpoint 2.209; the observed maximum is 2.209372"),
        tol=("dp", 3),
    ),
    # withdrawn application.greedy_gate.corpus_lo: caption statement removed; the corpus gate range [1.035,2.209] is asserted in subsec:features
    # withdrawn application.si1032.greedy_ratio: caption digit cut in the condensation; si1032's decline at the coverage gate is stated without the ratio
    # withdrawn application.noneuc.gart_scored_n: caption N-accounting removed; the 22-of-23 accounting is asserted in the body of the non-Euclidean results section
    # withdrawn application.noneuc.reference_scored_n: caption N-accounting removed; the 22-of-23 accounting is asserted in the body of the non-Euclidean results section
    Claim(
        id="methods.optuna.n_trials",
        anchor=r"estimator (TPE) ran {v} trials against",
        no_generator=(
            "lgbm_model_v3/gart2_optuna.db, study 'gart2': SELECT COUNT(*) FROM trials "
            "-> 200. Settle as for methods.optuna.trials_complete."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.optuna.nd_gain_pp",
        anchor=r"moved multidimensional MAPE by {v} percentage points",
        no_generator=(
            "paper_tooling/v4_study_allmodels_strata.csv, stratum=nd_test: "
            "GART2_logit_v3hp mape 0.622598 minus GART2_logit_tuned mape 0.611239 = "
            "0.011359. Tuned against untuned with the hyperparameters the only "
            "difference; see methods.optuna.untuned_nd for why the released model is "
            "the wrong control. Settle by banking v4_study_allmodels_strata.csv."
        ),
        tol=("dp", 3),
    ),
    Claim(
        id="methods.optuna.tsplib_cost_pp",
        anchor=r"percentage points while costing {v} points of TSPLIB MAPE",
        no_generator=(
            "paper_tooling/v4_study_allmodels_strata.csv, stratum=tsplib_euc2d: "
            "GART2_logit_tuned mape 2.636384 minus GART2_logit_v3hp mape 2.472478 = "
            "0.163906. Settle by banking v4_study_allmodels_strata.csv."
        ),
        tol=("dp", 2),
    ),
    Claim(
        id="methods.probe.gart_both_axes",
        anchor=r"held-out instances returns {v}\% non-increasing sweeps on both axes",
        expect="bank:cons_probe_gart_2_0_dimension_pct_nonincr_deployed",
        tol="printed",
        note="The dimension and node-count axes both read 100.0 with zero violations, "
             "so either key backs the sentence; the dimension one is named because it "
             "is the axis the comparators fail worst on.",
    ),
    # withdrawn results_nd.bhh_region_mape: Total-row readback sentence cut 2026-09-03 (simple-stupid pass, author rule); the numeral survives at the head of the BHH paragraph, where results_nd.bhh_region_mape_restated checks it
    # withdrawn results_nd.bhh_region_sdpe: Total-row readback sentence cut 2026-09-03 (simple-stupid pass, author rule); the SDPE cell is machine-checked in tab:results_nd and appears in no surviving body sentence
    # withdrawn results_nd.sdpe_smallest_bucket: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    Claim(
        id="results_nd.rho_dn_mape",
        anchor=r"the calibrated ratio $\hat\rho(d,n)$ at {v}\%/{~}\%. That baseline",
        expect="bank:nd_by_dim_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
        tol=("dp", 2),
        note="The strongest non-learned baseline IN THE ROSTER. The sentence "
             "now says so, because the Held--Karp bound of "
             "Section~\\ref{sec:frontier} uses no learned model either and "
             "beats GART 2.0 on this benchmark.",
    ),
    # withdrawn results_nd.sdpe_201_500: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn results_nd.bucket_201_lo: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn results_nd.bucket_500_hi: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn discussion.genclass.isotropic_mape: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn discussion.genclass.biased_mape: journey narrative removed per author directive (editorial restructure)
    # withdrawn discussion.genclass.clustered_mape: journey narrative removed per author directive (editorial restructure)
    # withdrawn discussion.genclass.linenoise_mape: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    Claim(
        id="discussion.timing.feature_share_pct",
        anchor=r"GART 2.0 spends {v}\% of its wall time on feature extraction",
        no_generator=_timing(
            "tsplib_by_size_time_one_protocol: the feature_time_s + mst_time_s share of "
            "total_time_s over the 78 EUC_2D instances, 92.90%"),
        tol=("dp", 2),
    ),
    # -- 2026-08-11 copy-edit: the constraint-transfer control -----------------
    #
    #    paper_tooling/constraint_transfer.py refits the extended-block variant
    #    with GART 2.0's two monotone constraints at seven seeds against a
    #    protocol registered before the first fit, and refits GART 2.0 itself as
    #    the matched control.  Its 1,934 ctrans_* keys are carried into
    #    paper_numbers.json by build_paper_tables.py, so every number below has a
    #    bank key and is checked, not merely recorded.

    # -- 2026-08-11 copy-edit: the two ranking measures the Conclusion omitted -

    # -- 2026-08-11 copy-edit: the Conclusion now matches the provenance body --
    #    The clause it replaces asserted "their cost labels remain internally
    #    consistent", which Section 3.3 had already withdrawn.

    # -- 2026-08-11 copy-edit: the Line Noise on-face relation is rank-like ----
    #    The printed 0.80 was the Pearson r on this slice, in a sentence whose
    #    claim is monotone tracking; the label was kept and the value replaced
    #    with the Spearman rho on the identical slice.
    Claim(
        id="results_2d.linenoise.on_face_median",
        anchor=r"a median {v}\% of points lie exactly on a face",
        no_generator=_LINENOISE(
            "median on-face fraction over the slice = 0.5985714286"),
        tol=("dp", 1),
        scale=100.0,
    ),
    Claim(
        id="results_2d.linenoise.rank_corr",
        anchor=r"tracks that fraction closely (Spearman {v}, rising monotonically",
        no_generator=_LINENOISE(
            "Spearman rho between on-face fraction and alpha = 0.8590870 "
            "(p = 2.49e-27). The Pearson r on the same slice is 0.8007114, which "
            "is what the manuscript printed as 0.80 under a Spearman label until "
            "2026-08-11; the label was correct for the claim and the value was "
            "not, so the value moved"),
        tol=("dp", 2),
    ),
    Claim(
        id="results_2d.linenoise.q1_alpha",
        anchor=r"rising monotonically from {v} to {~} across its quartiles",
        no_generator=_LINENOISE(
            "median alpha in the lowest on-face quartile = 1.1795406971"),
        tol=("dp", 3),
    ),
    Claim(
        id="results_2d.linenoise.q4_alpha",
        anchor=r"rising monotonically from {~} to {v} across its quartiles",
        no_generator=_LINENOISE(
            "median alpha in the highest on-face quartile = 1.4311880118; the "
            "four quartile medians are 1.1795, 1.2863, 1.3292, 1.4312, so the "
            "rise the sentence calls monotone is monotone"),
        tol=("dp", 3),
    ),
    # =======================================================================
    # FRONTIER -- the Held--Karp 1-tree comparator (Section~\ref{sec:frontier}),
    # its complexity table, and the labels its certificate refutes.
    #
    # Every entry resolves against ``frontier:``, a source added to
    # ``check_prose_numbers.py`` for this block: slash-separated paths into
    # ``paper_tooling/frontier_manuscript_bank.json``, which
    # ``frontier_manuscript_numbers.py`` derives from the settled artifacts.
    # These are ``expect`` and not ``no_generator`` deliberately -- a 1-tree
    # number is exactly the kind that would otherwise be typed once and never
    # checked again, and the bank is regenerable from the raw sweeps.
    #
    # Two provenance rules are enforced inside that generator rather than here,
    # and no entry below can bypass them: TSPLIB cost comes only from the solo
    # quiet-window pass (the co-measured ``crossing`` key of
    # frontier_positioning_bank.json is superseded), and multidimensional
    # accuracy comes only from the Polyak arm (the V&J column of the same file
    # is superseded by a factor of 23).
    # =======================================================================

    # -- abstract ------------------------------------------------------------

    # -- introduction --------------------------------------------------------

    # -- Section 3.3, the withdrawn sentence ---------------------------------
    # withdrawn provenance.certified_wrong_total: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.certified_wrong_in_184: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.certified_wrong_total_restated: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.corrupt_tour_count_in_new_sentence: journey narrative removed per author directive (editorial restructure)
    # withdrawn provenance.corrupt_tour_count_removal: journey narrative removed per author directive (editorial restructure)

    # -- Discussion timing numbers that re-keyed when the paragraph gained a
    #    pointer to the new section.  Pre-existing prose, now given generators.
    Claim(
        id="discussion.nd_gart2_time_large_n",
        anchor=r"the reference tour takes {~}~ms against GART 2.0's {v}~ms",
        expect="bank:nd_by_size_501_1000_gart_2_0_time_ms",
        tol=("dp", 0),
    ),
    Claim(
        id="discussion.d18512_gart2_time",
        anchor=r"GART 2.0 predicts in {v}~ms",
        no_generator=(
            "GART 2.0's total time on TSPLIB d18512 as recorded by the "
            "published benchmark harness: 239 ms, "
            "tsplib_benchmark/results/all_models_tsplib_repaired.csv, row "
            "instance=d18512 model=GART_2.0, column total_time_s. That column "
            "is not aggregated into paper_numbers.json, which carries bucket "
            "medians only. The solo three-repeat re-measurement of the same "
            "work is 237.42 ms and is quoted separately in "
            "Section~\\ref{subsec:frontier_tsplib}; the two are different "
            "protocols, not a disagreement. Settle by exporting a "
            "per-instance timing table under tsplib_instance_*_time_ms keys."
        ),
        tol=("dp", 0),
    ),

    # -- 5.1 the bound and the ascent ----------------------------------------
    # withdrawn frontier.ascent.vj_higher_pct: cut in the verbosity sweep; arm-vs-arm bookkeeping; the released ladder banks carry both arms
    # withdrawn frontier.ascent.converged_single: cut in the verbosity sweep; arm-vs-arm bookkeeping; the released ladder banks carry both arms
    # withdrawn frontier.ascent.converged_envelope: cut in the verbosity sweep; arm-vs-arm bookkeeping; the released ladder banks carry both arms

    # -- 5.2 TSPLIB ----------------------------------------------------------
    Claim(
        id="frontier.tsplib.repeats",
        anchor=r"We time each instance as the median of {v} repeats",
        expect="frontier:tsplib/repeats",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.matched_n",
        anchor=r"We compare the two on {v} of the {~} EUC\_2D instances",
        expect="frontier:tsplib/N_matched",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.euc2d_n",
        anchor=r"We compare the two on {~} of the {v} EUC\_2D instances",
        expect="frontier:tsplib/N_euc2d",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.raw_crossing_k",
        anchor=r"reaches parity at budget {v}, at {~} times the cost",
        expect="frontier:tsplib/crossing_ladder_k",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.raw_crossing_cost_x",
        anchor=r"reaches parity at budget {~}, at {v} times the cost",
        expect="frontier:tsplib/crossing_cost_x_gart2",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.raw_crossing_margin",
        anchor=r"and a margin of {v} percentage points",
        expect="= {frontier:tsplib/gart2_mape_pct}"
               " - {frontier:tsplib/crossing_bound_mape_pct}",
        tol=("dp", 3),
        note="Stated as a margin, so the margin is what is checked. It is 0.04 "
             "points, which is why the printed crossing rung is not a robust "
             "statistic; see frontier.labels.interp_crossing_excluded.",
    ),
    Claim(
        id="frontier.tsplib.paired_win_rate",
        anchor=r"with a paired win rate of exactly {v}\%",
        expect="frontier:tsplib/paired_win_rate_of_bound_at_crossing_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.tsplib.cal_crossing_k",
        anchor=r"on both cost and accuracy at an ascent budget of {v}: {~} times the cost",
        expect="frontier:tsplib_calibrated_bound/Total (all EUC_2D)/crossing_ladder_k",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.cal_crossing_cost_x",
        anchor=r"at an ascent budget of {~}: {v} times the cost at",
        expect="frontier:tsplib_calibrated_bound/Total (all EUC_2D)/x_gart2_by_k/25",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.cal_crossing_mape",
        anchor=r"times the cost at {v}\% MAPE against {~}\%, on the same instances",
        expect="frontier:tsplib_calibrated_bound/Total (all EUC_2D)/mape_pct_by_k/25",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.cal_crossing_gart2",
        anchor=r"times the cost at {~}\% MAPE against {v}\%, on the same instances",
        expect="frontier:tsplib/gart2_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.small_bucket_n",
        anchor=r"On the {v} smallest instances the calibrated bound",
        expect="frontier:tsplib/n in [51,150]/N",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.small_cal_cost_x",
        anchor=r"at that same budget costs {v} times GART 2.0 at {~}\% MAPE against {~}\%",
        expect="frontier:tsplib_calibrated_bound/n in [51,150]/x_gart2_by_k/25",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.small_cal_mape",
        anchor=r"at that same budget costs {~} times GART 2.0 at {v}\% MAPE against {~}\%",
        expect="frontier:tsplib_calibrated_bound/n in [51,150]/mape_pct_by_k/25",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.small_gart2_mape",
        anchor=r"at that same budget costs {~} times GART 2.0 at {~}\% MAPE against {v}\%",
        expect="frontier:tsplib/n in [51,150]/gart2_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.small_raw_accuracy_gain",
        anchor=r"uncalibrated certificate reaches {v}\% lower error at {~} times the cost. In the middle bucket",
        expect="= 100 * (1 - {frontier:tsplib/n in [51,150]/crossing_bound_over_gart2_mape})",
        tol=("dp", 1),
        note="The uncalibrated, certified bound against the released estimator "
             "in the bucket where the estimator loses on both axes.",
    ),
    Claim(
        id="frontier.tsplib.small_raw_cost_x",
        anchor=r"uncalibrated certificate reaches {~}\% lower error at {v} times the cost. In the middle bucket",
        expect="frontier:tsplib/n in [51,150]/crossing_cost_x_gart2",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.large_raw_cost_x",
        anchor=r"the raw bound matches at {v} times the cost and again nothing dominates",
        expect="frontier:tsplib/n > 400/crossing_cost_x_gart2",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.load_sensitivity_bound",
        anchor=r"its median rises by a factor of {v} between a quiet and a noisy window",
        expect="frontier:tsplib/load_sensitivity/HK_1Tree_50/noisy_over_quiet",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.load_sensitivity_gart2",
        anchor=r"against {v} for GART 2.0. Every published repeat",
        expect="frontier:tsplib/load_sensitivity/GART_2.0/noisy_over_quiet",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.capped_x_at_crossing",
        anchor=r"the bound costs {v} times GART 2.0 at the crossing budget",
        expect="frontier:tsplib/capped_tail/hk_over_gart2_by_k/50",
        tol=("dp", 0),
    ),
    Claim(
        id="frontier.tsplib.capped_x_at_top",
        anchor=r"and {v} times at the top of the ladder",
        expect="frontier:tsplib/capped_tail/hk_over_gart2_by_k/500",
        tol=("dp", 0),
    ),
    Claim(
        id="frontier.tsplib.capped_seconds",
        anchor=r"at the top of the ladder, {v} seconds against",
        expect="frontier:tsplib/capped_tail/hk_ms_by_k/500",
        tol=("dp", 0),
        scale=0.001,
        note="Bank stores milliseconds; the sentence prints seconds.",
    ),
    Claim(
        id="costacct.d18512_bound_x_top",
        anchor=r"at the crossing budget and {v} times as much, {~} seconds",
        expect="frontier:tsplib/capped_tail/hk_over_gart2_by_k/500",
        tol=("dp", 0),
    ),
    Claim(
        id="frontier.nd.load_sensitivity_gart2_body",
        anchor=r"{~} against {v} between a quiet and a noisy window, so every published repeat",
        expect="frontier:tsplib/load_sensitivity/GART_2.0/noisy_over_quiet",
        tol=("dp", 2),
    ),
    Claim(
        id="costacct.d18512_bound_x_crossing",
        anchor=r"costs {v} times as much at the crossing budget and {~} times as much",
        expect="frontier:tsplib/capped_tail/hk_over_gart2_by_k/50",
        tol=("dp", 0),
    ),
    Claim(
        id="costacct.d18512_bound_top_s",
        anchor=r"times as much, {v} seconds, at the top of the ladder",
        expect="frontier:tsplib/capped_tail/hk_ms_by_k/500",
        tol=("dp", 0),
        scale=0.001,
        note="Bank stores milliseconds; the sentence prints seconds.",
    ),
    Claim(
        id="frontier.nd.load_sensitivity_bound_body",
        anchor=r"{v} against {~} between a quiet and a noisy window, so every published repeat",
        expect="frontier:tsplib/load_sensitivity/HK_1Tree_50/noisy_over_quiet",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.capped_gart2_ms",
        anchor=r"seconds against {v} milliseconds (the",
        expect="frontier:tsplib/capped_tail/gart2_ms",
        tol=("dp", 0),
    ),
    Claim(
        id="frontier.tsplib.harness_d18512_ms",
        anchor=r"(the {v}~ms of Section~\ref{subsec:cost_accounting} is the same work",
        no_generator=(
            "Same 239 ms as discussion.d18512_gart2_time, cross-referenced here "
            "so that two figures for one instance -- 237.42 ms solo, 239 ms "
            "published harness -- read as two protocols rather than as a "
            "disagreement. Source: "
            "tsplib_benchmark/results/all_models_tsplib_repaired.csv, row "
            "instance=d18512 model=GART_2.0, column total_time_s. Same "
            "settle-by: export a per-instance timing table."
        ),
        tol=("dp", 0),
    ),
    Claim(
        id="frontier.tsplib.table_caption_n",
        anchor=r"cost/accuracy ladder over the {v} matched instances",
        expect="frontier:tsplib/N_matched",
        tol="exact",
    ),

    # -- 5.3 multidimensional ------------------------------------------------
    Claim(
        id="frontier.nd.best_budget",
        anchor=r"At an ascent budget of {v} (Table~\ref{tab:frontier_nd}) the bound reaches",
        expect="frontier:nd/best_budget_k",
        tol="exact",
    ),
    Claim(
        id="frontier.nd.bound_mape",
        anchor=r"the bound reaches {v}\% MAPE against GART 2.0's {~}\%, a factor of",
        expect="frontier:nd/bound_mape_pct_by_k/200",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.nd.gart2_mape",
        anchor=r"the bound reaches {~}\% MAPE against GART 2.0's {v}\%, a factor of",
        expect="frontier:nd/gart2_mape_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.nd.accuracy_factor",
        anchor=r"a factor of {v}, at {~} times the cost, and it wins the paired comparison",
        expect="frontier:nd/accuracy_factor_at_best_budget",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.nd.cost_x",
        anchor=r"a factor of {~}, at {v} times the cost, and it wins the paired comparison",
        expect="frontier:nd/bound_x_gart2_by_k/200",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.nd.paired_win_rate",
        anchor=r"wins the paired comparison on {v}\% of instances",
        expect="frontier:nd/paired_win_rate_pct_by_k/200",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.nd.crossing_k",
        anchor=r"overtakes GART 2.0 at a budget of {v}, where it costs {~} times as much",
        expect="frontier:nd/crossover_k_by_group/all ND",
        tol="exact",
        note="The budget at which the bound first OVERTAKES GART 2.0, which is "
             "not the budget at which it dominates: nd/best_budget_k is 200 and "
             "is what frontier.nd.best_budget points at.",
    ),
    Claim(
        id="frontier.nd.crossing_cost_x",
        anchor=r"overtakes GART 2.0 at a budget of {~}, where it costs {v} times as much",
        expect="frontier:nd/crossover_cost_x_by_group/all ND",
        tol=("dp", 2),
    ),
    # withdrawn frontier.nd.corpus_weighted_budget: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn frontier.nd.corpus_weighted_cost_x: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn frontier.nd.d100_budget: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn frontier.nd.d100_cost_x: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn frontier.nd.d100_accuracy_factor: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    Claim(
        id="frontier.nd.planar_group_cost_x",
        anchor=r"matching GART 2.0's accuracy costs the bound {v} times as much",
        expect="frontier:nd/crossover_cost_x_by_group/d in {2,3}",
        tol=("dp", 2),
        note="The one dimension group on which GART 2.0 stays on the front.",
    ),
    Claim(
        id="frontier.nd.concorde_subset_n",
        anchor=r"On the {v} scored instances whose label the generation run recorded as a Concorde-proven optimum",
        expect="frontier:nd/concorde_subset/N",
        tol="exact",
    ),
    Claim(
        id="frontier.nd.concorde_gart2_mape",
        anchor=r"GART 2.0 scores {v}\% against the bound's {~}\% at a budget of",
        expect="frontier:nd/concorde_subset/gart2_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.nd.concorde_bound_mape",
        anchor=r"GART 2.0 scores {~}\% against the bound's {v}\% at a budget of",
        expect="frontier:nd/concorde_subset/bound_mape_pct_by_k/200",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.nd.concorde_budget",
        anchor=r"against the bound's {~}\% at a budget of {v}. That status",
        expect="frontier:nd/best_budget_k",
        tol="exact",
    ),
    Claim(
        id="frontier.nd.closes_exactly_pct",
        anchor=r"The relaxation closes exactly on {v}\% of the scored multidimensional split",
        expect="frontier:nd/relaxation_closes_exactly_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.nd.table_caption_n",
        anchor=r"cost/accuracy ladder, all {v} scored instances, Polyak ascent",
        expect="frontier:nd/N",
        tol="exact",
    ),

    # -- 5.4 complexity ------------------------------------------------------
    Claim(
        id="frontier.complexity.delaunay_vectors",
        anchor=r"At {v} potential vectors drawn from real ascent trajectories",
        expect="frontier:complexity/delaunay_pi_vectors",
        tol="exact",
    ),
    Claim(
        id="frontier.complexity.delaunay_heavier",
        anchor=r"heavier than the exact one at {v} of them",
        expect="frontier:complexity/delaunay_heavier_count",
        tol="exact",
        note="Refutes the shortcut rather than assuming it is unavailable; this "
             "is why the bound is a complexity class above the MST family.",
    ),
    Claim(
        id="frontier.complexity.gart2_exponent",
        anchor=r"cost exponents in $n$ agree: {v} for GART 2.0 above a thousand nodes",
        expect="frontier:complexity/gart2_slope_n_ge_1000",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.complexity.bound_exponent_lo",
        anchor=r"above a thousand nodes against {v}--{~} for the bound over the same window",
        expect="frontier:complexity/onetree_slope_n_ge_1000_lo",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.complexity.bound_exponent_hi",
        anchor=r"above a thousand nodes against {~}--{v} for the bound over the same window",
        expect="frontier:complexity/onetree_slope_n_ge_1000_hi",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.complexity.mst_band_lo",
        anchor=r"GART 2.0 costs a flat {v} to {~} times the MST-ratio family",
        expect="frontier:complexity/gart2_over_mst_band_lo",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.complexity.mst_band_hi",
        anchor=r"GART 2.0 costs a flat {~} to {v} times the MST-ratio family",
        expect="frontier:complexity/gart2_over_mst_band_hi",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.complexity.ratio_at_1000",
        anchor=r"the ratio moves from {v} at a thousand nodes to {~} at sixteen thousand",
        expect="frontier:complexity/ledger_d2/n_1000/onetree_k50_over_gart2",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.complexity.ratio_at_16000",
        anchor=r"the ratio moves from {~} at a thousand nodes to {v} at sixteen thousand",
        expect="frontier:complexity/ledger_d2/n_16000/onetree_k50_over_gart2",
        tol="exact",
    ),
    Claim(
        id="frontier.complexity.nd_gart2_lowd",
        anchor=r"cost exponent in $n$ is {v} at $d\in[4,10]$ and {~} at $d\in[15,50]$",
        expect="frontier:complexity/nd_slopes/gart2_d_4_10",
        tol=("dp", 2),
        note="The Theta(n log n) defence is planar; compute_mst takes a dense "
             "kernel from d=4 upward, so on ND GART 2.0 is quadratic too.",
    ),
    Claim(
        id="frontier.complexity.nd_gart2_highd",
        anchor=r"cost exponent in $n$ is {~} at $d\in[4,10]$ and {v} at $d\in[15,50]$",
        expect="frontier:complexity/nd_slopes/gart2_d_15_50",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.complexity.nd_bound_lo",
        # Re-anchored 2026-08-12: the sentence named a function, \texttt{compute\_mst},
        # and now describes what it does. Same number, same sentence, same claim.
        anchor=r"against {v}--{~} for the bound, because the MST construction",
        expect="frontier:complexity/nd_slopes/onetree_nd_lo",
        tol="printed",
    ),
    Claim(
        id="frontier.complexity.nd_bound_hi",
        # Re-anchored 2026-08-12: see nd_bound_lo above.
        anchor=r"against {~}--{v} for the bound, because the MST construction",
        expect="frontier:complexity/nd_slopes/onetree_nd_hi",
        tol="printed",
    ),
    Claim(
        id="frontier.complexity.nd_gart2_planar",
        anchor=r"that exponent is {v}. On that benchmark both families are quadratic",
        expect="frontier:complexity/nd_slopes/gart2_d_2_3",
        tol=("dp", 2),
    ),

    # -- 5.5 exact solver ----------------------------------------------------
    # withdrawn frontier.exact.gart2_ms: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone

    # -- 5.6 labels ----------------------------------------------------------
    # withdrawn frontier.labels.total_evaluated: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.refuted_total: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.refuted_scored: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.refuted_nd_test: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.worst_excess: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.overlap_count: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.overlap_total: journey narrative removed per author directive (editorial restructure)
    Claim(
        id="frontier.labels.corrupt_population",
        anchor=r"The {v} quarantined instances above carry no recoverable label",
        no_generator=(
            "Size of the reference-tour audit's corrupt bucket, 184, restated "
            "from Section 3.3. Same artifact and same settle-by as "
            "provenance.corrupt_tour_count_in_new_sentence: "
            "paper_tooling/reference_tour_audit.csv, rows with "
            "bucket=='corrupt'."
        ),
        tol="exact",
    ),
    Claim(
        id="frontier.labels.linhp318_label",
        anchor=r"so its published {v} is a path optimum no tour on those coordinates can attain",
        expect="frontier:labels/linhp318/stored_label",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.linhp318_tour_opt",
        anchor=r"scored there against the tour optimum on those coordinates, ${v}$, \texttt{lin318}",
        expect="frontier:labels/linhp318/tour_optimum_on_its_coordinates",
        tol="exact",
    ),
    # withdrawn frontier.labels.linhp318_bound: journey narrative removed per author directive (editorial restructure)
    Claim(
        id="frontier.labels.tsplib_files_scanned",
        anchor=r"over the full {v}-instance TSPLIB set",
        expect="frontier:labels/linhp318/tsplib_files_scanned",
        tol="exact",
        note="Re-anchored after the editorial restructure: the file-scan sentence "
             "is gone; the same 111, the size of the full TSPLIB set, is now "
             "asserted in Section 6's oracle-constant sentence.",
    ),
    # withdrawn frontier.labels.gart2_mape_as_published: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.gart2_mape_repaired: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.interp_crossing_as_published: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.interp_crossing_repaired: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.repaired_margin: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.cal_margin: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.repaired_margin_restated: journey narrative removed per author directive (editorial restructure)

    # -- 5.7 verdict ---------------------------------------------------------
    Claim(
        id="frontier.verdict.cheaper_above_n",
        anchor=r"cheaper than a 1-tree bound above roughly {v} nodes in the plane",
        no_generator=(
            "Upper edge of the TSPLIB size buckets of Table "
            "\\ref{tab:tsplib_by_size}, n=400, quoted as the size above which "
            "the ordering favors GART 2.0. It names a bucket boundary this "
            "paper fixed a priori, not a fitted changepoint; no artifact "
            "estimates a crossover in n because the ladder is measured per "
            "bucket. Settle by fitting the cost/accuracy crossover as a "
            "continuous function of n over "
            "paper_tooling/hk1tree_solo_cost_per_instance.csv."
        ),
        tol="exact",
    ),
    Claim(
        id="frontier.verdict.dearer_below_n",
        anchor=r"in the plane and dearer below roughly {v} (Appendix Table~\ref{tab:frontier_tsplib})",
        no_generator=(
            "Upper edge of the smallest TSPLIB size bucket, n=150, quoted as "
            "the size below which the certified bound wins on both axes. Same "
            "status and same settle-by as frontier.verdict.cheaper_above_n: it "
            "is the bucket boundary, not a fitted changepoint."
        ),
        tol="exact",
    ),
    Claim(
        id="frontier.verdict.margin_over_mst_ratio",
        anchor=r"separated by {v} percentage points on the full {~}-instance EUC\_2D set",
        expect="= {frontier:tsplib78/asymptotic_mst_mape_pct}"
               " - {frontier:tsplib78/gart2_mape_pct}",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.verdict.euc2d_n",
        anchor=r"percentage points on the full {v}-instance EUC\_2D set",
        expect="frontier:tsplib78/N",
        tol="exact",
    ),
    Claim(
        id="frontier.verdict.margin_to_converged_bound",
        anchor=r"EUC\_2D set, against {v} points from the converged bound on the same set",
        expect="= {frontier:tsplib78/gart2_mape_pct}"
               " - {frontier:tsplib78/bound_converged_mape_pct}",
        tol=("dp", 2),
        note="The point of the pair: GART 2.0 sits closer to the cheap MST "
             "anchor than to the converged bound, so 'accuracy closer to the "
             "bound than to the closed forms' is true only against the "
             "area-based estimators.",
    ),

    # -- conclusion ----------------------------------------------------------
    # -- Numbers the label repair of Section 5.8 moved, re-registered ---------
    # The close-pair census is keyed on the reference costs, so quarantining
    # 74 multidimensional instances and repairing linhp318 moved every count
    # and every percentage in these two paragraphs and in Appendix A.
    Claim(
        id="rank.close5.pairs_2d",
        anchor=r"orders 74.5\% of {v} 2D pairs",
        expect="bank:rank_2d_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="rank.close5.pairs_nd",
        anchor=r"2D pairs, 92.4\% of {v} multidimensional pairs",
        expect="bank:rank_nd_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="rank.close5.gart_tsplib_pct",
        anchor=r"multidimensional pairs, and {v}\% of {~} TSPLIB pairs correctly",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close5_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close5.pairs_tsplib",
        anchor=r"multidimensional pairs, and {~}\% of {v} TSPLIB pairs correctly",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="rank.close5.floor_2d",
        anchor=r"TSPLIB pairs correctly, against {v}\%, {~}\%, and {~}\% for the $\alpha=1$ control",
        expect="bank:rank_2d_l_mathrm_mst_alpha_1_close5_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close5.floor_nd",
        anchor=r"TSPLIB pairs correctly, against {~}\%, {v}\%, and {~}\% for the $\alpha=1$ control",
        expect="bank:rank_nd_l_mathrm_mst_alpha_1_close5_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close5.floor_tsplib",
        anchor=r"TSPLIB pairs correctly, against {~}\%, {~}\%, and {v}\% for the $\alpha=1$ control",
        expect="bank:rank_tsplib_euc2d_l_mathrm_mst_alpha_1_close5_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close5.gain_nd_pp",
        anchor=r"The multidimensional gain of {v} percentage points is the substantive one",
        expect=("= {rank_nd_gart_2_0_close5_pct} - {rank_nd_l_mathrm_mst_alpha_1_close5_pct}"),
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close5.gain_2d_pp",
        anchor=r"The 2D gain of {v} points is not",
        expect=("= {rank_2d_gart_2_0_close5_pct} - {rank_2d_l_mathrm_mst_alpha_1_close5_pct}"),
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close5.gain_tsplib_pp",
        anchor=r"pairs cannot support an {v}-point claim",
        expect=("= {rank_tsplib_euc2d_gart_2_0_close5_pct} - "
                "{rank_tsplib_euc2d_l_mathrm_mst_alpha_1_close5_pct}"),
        tol=("dp", 0),
    ),
    Claim(
        id="rank.close10.gart_2d",
        anchor=r"At a 10\% threshold GART 2.0 obtains {v}\%, {~}\%, and {~}\%, against",
        expect="bank:rank_2d_gart_2_0_close10_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close10.gart_nd",
        anchor=r"At a 10\% threshold GART 2.0 obtains {~}\%, {v}\%, and {~}\%, against",
        expect="bank:rank_nd_gart_2_0_close10_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close10.gart_tsplib",
        anchor=r"At a 10\% threshold GART 2.0 obtains {~}\%, {~}\%, and {v}\%, against",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close10_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close10.floor_2d",
        anchor=r"At a 10\% threshold GART 2.0 obtains {~}\%, {~}\%, and {~}\%, against {v}\%, {~}\%, and {~}\% for the $\alpha=1$ control",
        expect="bank:rank_2d_l_mathrm_mst_alpha_1_close10_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close10.floor_nd",
        anchor=r"At a 10\% threshold GART 2.0 obtains {~}\%, {~}\%, and {~}\%, against {~}\%, {v}\%, and {~}\% for the $\alpha=1$ control",
        expect="bank:rank_nd_l_mathrm_mst_alpha_1_close10_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close10.floor_tsplib",
        anchor=r"At a 10\% threshold GART 2.0 obtains {~}\%, {~}\%, and {~}\%, against {~}\%, {~}\%, and {v}\% for the $\alpha=1$ control",
        expect="bank:rank_tsplib_euc2d_l_mathrm_mst_alpha_1_close10_pct",
        tol=("dp", 1),
    ),
    # withdrawn appendix.rank.nd_universe: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.nd_n: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.quarantined_nd: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.nd_close5_pairs: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.nd_close10_pairs: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.2d_close5_pairs: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.2d_close10_pairs: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.tsplib_close5_pairs: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.tsplib_close10_pairs: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.direct_scan_pairs: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    Claim(
        id="rank.close5.pairs_tsplib_restated",
        anchor=r"is not, and {v} TSPLIB pairs cannot support",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="rank.close10.threshold",
        anchor=r"claim. At a {v}\% threshold GART 2.0 obtains",
        no_generator=("Design constant, not a measurement: the wider of the two "
                      "close-pair bands, fixed a priori in build_paper_tables. "
                      "The bank stores the cells the threshold produces, not "
                      "the threshold."),
        tol="exact",
    ),
    # withdrawn appendix.rank.close5_threshold: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.tsplib_universe: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # withdrawn appendix.rank.nd_universe_rounded: wave-7 cut: detail appendix removed, grids live in the released tidy tables
    # -- Section 5.8, the repair. Numbers come from
    # paper_tooling/labels_repaired.json via frontier_manuscript_bank.json.
    # withdrawn dataset.repair.corpus_pct: journey narrative removed per author directive (editorial restructure)
    Claim(
        id="provenance.quarantine.nd_test",
        anchor=r"{v} of the {~} fall in the multidimensional test partition",
        expect="frontier:labels/repair_quarantined_nd_test",
        tol="exact",
    ),
    Claim(
        id="provenance.quarantine.scored_nd",
        anchor=r"scored on the remaining ${v}$ instances throughout",
        expect="frontier:labels/repair_nd_test_scored",
        tol="exact",
    ),
    # withdrawn labels.mech.d1: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.mech.d2: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.mech.clean: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.mech.bad_total: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.mech.corpus_n: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.mech.corpus_pct: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.mech.bad_train: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.mech.bad_val: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.mech.bad_test: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.repair.exact_n: journey narrative removed per author directive (editorial restructure)
    Claim(
        id="labels.repair.hk_max_n",
        anchor=r"At $n\le{v}$ the label is the exact Held--Karp dynamic-programming optimum",
        expect="frontier:labels/repair_hk_exact_max_n",
        tol="exact",
        note="Re-anchored after the editorial restructure: the repair-mechanics "
             "paragraph is gone; the exact-solve size cap survives in the Label "
             "Certification description of how the released labels are built.",
    ),
    Claim(
        id="labels.repair.certified_by_bound",
        anchor=r"proves it optimal after the fact for ${v}$ instances",
        expect="frontier:labels/repair_nd_tour_certified_optimal",
        tol="exact",
        note="Re-anchored after the editorial restructure into the Label "
             "Certification subsection.",
    ),
    # withdrawn labels.repair.d2_n: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.repair.d2_signed: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.repair.d2_signed_g1000: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.repair.d2_signed_g10000: journey narrative removed per author directive (editorial restructure)
    # withdrawn labels.repair.d2_improvable: journey narrative removed per author directive (editorial restructure)
    Claim(
        id="labels.verify.n",
        anchor=r"Across the {v} bound checks the released artifacts supply there are zero violations of $B\le L$",
        expect="frontier:labels/repair_verify_instances_checked",
        tol="exact",
        note="Re-anchored after the editorial restructure: the Label Certification "
             "copy of the verification count. The Label Validation copy of the "
             "same fact is provenance.verify.n.",
    ),
    # withdrawn labels.verify.violations: journey narrative removed per author directive (editorial restructure); the count is now written as the word 'zero', which carries no numeral to check
    # withdrawn labels.effect.nd_label_mape: journey narrative removed per author directive (editorial restructure)
    Claim(
        id="conclusion.repair.quarantined",
        anchor=r"The {v} instances ({~}\%) that fail are \emph{quarantined}, not scored",
        expect="frontier:labels/repair_quarantined_total",
        tol="exact",
        note="Re-anchored after the editorial restructure: the conclusion sentence "
             "is gone and the quarantine count now lives in Section 3's Label "
             "Validation subsection.",
    ),
    # withdrawn frontier.labels.raw_margin_as_published: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.interp_crossing_excluded: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.gart2_mape_excluded_new: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.crossing_k_excluded: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.crossing_k_repaired: journey narrative removed per author directive (editorial restructure)
    # withdrawn frontier.labels.crossing_x_repaired: journey narrative removed per author directive (editorial restructure)
    # withdrawn results_nd.sdpe_501_1000: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn figure.boxplot_2d.hk_budget: caption now inherits axes and bound budget from fig:boxplot_nd by reference; the budget is asserted once, in that caption
    # withdrawn figure.boxplot_tsplib.hk_budget: caption now inherits axes and bound budget from fig:boxplot_nd by reference; the budget is asserted once, in that caption

    # -- Section 6: the certified bound on the non-Euclidean set --------------
    # Section 6 previously ended without scoring the comparator the rest of the
    # paper is judged against, on the one corpus where the bound needs no
    # embedding and the estimator does. These five entries are the answer, read
    # from the all-benchmark bank rather than from paper_numbers.json, which
    # carries no bound row for this corpus.
    Claim(
        id="application.bound.like_for_like_n",
        anchor=r"On the {v} instances both methods score over all {~} non-EUC\_2D files --- a set wider",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/N",
        tol="exact",
        note="Instances scored by both GART 2.0 and the 1-tree bound; the bound "
             "additionally scores brg180 and si1032, GART 2.0 pla33810 and pla85900.",
    ),
    Claim(
        id="application.bound.gart_mape",
        anchor=r"GART 2.0 obtains {v}\% MAPE against {~}\% for the raw bound",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/GART_2.0_MAPE_pct",
    ),
    Claim(
        id="application.bound.vj_k100",
        anchor=r"MAPE against {v}\% for the raw bound at an ascent budget",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/vj_raw_MAPE_by_k/100",
    ),
    Claim(
        id="application.bound.polyak_k500",
        anchor=r"At a budget of {~} the Polyak step reaches {v}\%, a factor of",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/polyak_raw_MAPE_by_k/500",
        note="The sentence used to read as one ascent at two budgets. 1.27 is the "
             "Volgenant--Jonker bound and 0.51 the Polyak one; Volgenant--Jonker at "
             "500 is 1.14 and Polyak at 100 is 1.99, so the printed pair belonged to "
             "neither arm. The anchors now carry the step name.",
    ),
    Claim(
        id="application.bound.factor",
        anchor=r"the Polyak step reaches {~}\%, a factor of {v} against GART 2.0",
        expect="= {allbench:cells/noneuc/like_for_like_vs_GART2/GART_2.0_MAPE_pct}"
               " / {allbench:cells/noneuc/like_for_like_vs_GART2/polyak_raw_MAPE_by_k/500}",
        tol=("dp", 1),
        note="Stated as a ratio, so the ratio is what is checked.",
    ),

    # -- Conclusion: the three numbers that carry the paper's one finding -----

    # -- Section 5.2: what "corpus median" is a median of ---------------------
    Claim(
        id="frontier.tsplib.throughput_x",
        anchor=r"pair costs {v} times as much, because the bound is cheap",
        expect="costfront:corpus_median_definition/tsplib_published/x_gart2_throughput_k25",
        tol="printed",
        note="The second aggregation of the same pair at the same budget: the sum "
             "of the per-instance medians rather than their median. Printed beside "
             "the 0.90 so the phrase 'corpus median' cannot be read as the cost of "
             "running the corpus.",
    ),

    # -- Section 5.4: the bound priced on the other two benchmarks ------------
    # Everything below reads from hk1tree_cost_frontier_bank.json, the cost half
    # of the all-benchmark study. paper_numbers.json carries no bound row for
    # either corpus and the accuracy bank carries no cost column, so this is the
    # only source for the pairs these sentences assert.
    # withdrawn frontier.2d.gart_ms: cut in the verbosity sweep; readback of tab:frontier_2d's GART row; the table carries the value
    # withdrawn frontier.2d.gart_mape: cut in the verbosity sweep; readback of tab:frontier_2d's GART row; the table carries the value
    Claim(
        id="frontier.2d.raw_crossing_k",
        anchor=r"Volgenant--Jonker step at a budget of {v}, at {~} times the cost",
        expect="costfront:cells/2d/groups/Total (all 2D)/crossover/vj_ckpt/raw/k",
        tol="exact",
    ),
    Claim(
        id="frontier.2d.raw_crossing_cost_x",
        anchor=r"at a budget of {~}, at {v} times the cost and a paired win rate",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/x_gart2_typical",
        tol=("dp", 2),
    ),
    # withdrawn frontier.2d.raw_crossing_mape: cut in the verbosity sweep; readback of a tab:frontier_2d cell; the table carries the value
    Claim(
        id="frontier.2d.raw_crossing_win",
        anchor=r"and a paired win rate of {v}\%; the calibrated row passes",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/"
               "win_rate_vs_gart2_pct/raw",
        tol=("dp", 1),
    ),
    # withdrawn frontier.2d.cal_crossing_cost_x: cut in the verbosity sweep; readback of a tab:frontier_2d cell; the table carries the value
    # withdrawn frontier.2d.cal_crossing_mape: cut in the verbosity sweep; readback of a tab:frontier_2d cell; the table carries the value
    Claim(
        id="frontier.2d.small_bucket_n",
        anchor=r"On the {v} instances with $n\le10$, the raw bound",
        expect="costfront:cells/2d/groups/n in [5,10]/N",
        tol="exact",
    ),
    Claim(
        id="frontier.2d.small_bucket_k",
        anchor=r"the raw bound at a budget of {v} reads {~}\% MAPE against",
        no_generator=(
            "An ascent budget: a rung of the ladder Table~\\ref{tab:frontier_2d} "
            "prints, chosen for the sentence rather than measured. The rung set is "
            "costfront:cells/2d/protocol/budgets. Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.2d.small_bucket_mape",
        anchor=r"at a budget of {~} reads {v}\% MAPE against GART 2.0's",
        expect="costfront:cells/2d/groups/n in [5,10]/ascents/vj_ckpt/25/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.2d.small_bucket_gart_mape",
        anchor=r"MAPE against GART 2.0's {v}\% at {~} times the cost, winning",
        expect="costfront:cells/2d/groups/n in [5,10]/gart2_MAPE_pct",
        note="The trailing ', and wins' keeps this anchor off the abstract's "
             "sentence about the multidimensional benchmark, which is otherwise "
             "word-for-word the same shape.",
        tol="printed",
    ),
    Claim(
        id="frontier.2d.small_bucket_cost_x",
        anchor=r"at {v} times the cost, winning {~}\% of the paired comparisons",
        expect="costfront:cells/2d/groups/n in [5,10]/ascents/vj_ckpt/25/x_gart2_typical",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.2d.small_bucket_win",
        anchor=r"winning {v}\% of the paired comparisons. On the",
        expect="costfront:cells/2d/groups/n in [5,10]/ascents/vj_ckpt/25/"
               "win_rate_vs_gart2_pct/raw",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.2d.large_bucket_n",
        anchor=r"On the {v} instances with $n>500$, nothing dominates",
        expect="costfront:cells/2d/groups/n in [501,1000]/N",
        tol="exact",
    ),
    Claim(
        id="frontier.2d.large_bucket_cost_x",
        anchor=r"only at the top budget, and there it costs {v} times as much",
        expect="costfront:cells/2d/groups/n in [501,1000]/ascents/vj_ckpt/500/x_gart2_typical",
        tol="printed",
        note="The largest bucket is where the reversal of Section 5.2 runs the other "
             "way: more accurate only at the top rung, and only at 26 times the cost.",
    ),
    Claim(
        id="frontier.2d.caption_n",
        anchor=r"cost/accuracy ladder, all {v} instances, both step rules",
        expect="costfront:cells/2d/groups/Total (all 2D)/N",
        tol="exact",
    ),

    Claim(
        id="frontier.noneuc.matched_n",
        anchor=r"Over the {v} instances both methods score, GART 2.0 costs",
        expect="costfront:cells/noneuc/instance_accounting/matched",
        tol="exact",
        note="The bound scores 31; GART 2.0 declines brg180 on the metric screen and "
             "si1032 on the coverage gate, so 29 are matched.",
    ),
    Claim(
        id="frontier.noneuc.gart_ms",
        anchor=r"both methods score, GART 2.0 costs {v}~ms, of which the embedding",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/gart2_ms",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.noneuc.mds_share",
        anchor=r"of which the embedding is {v}\%, at {~}\% MAPE",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/gart2_ms_mds_share_pct",
        tol=("dp", 2),
        note="The MDS embedding is the part of the estimator's cost the bound does "
             "not pay at all, so it is separated out rather than folded in.",
    ),
    Claim(
        id="frontier.noneuc.gart_mape",
        anchor=r"the embedding is {~}\%, at {v}\% MAPE. The Polyak step",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/gart2_MAPE_pct",
        tol="printed",
    ),
    Claim(
        id="frontier.noneuc.pk500_k",
        anchor=r"The Polyak step at a budget of {v} reaches",
        no_generator=(
            "An ascent budget: the top rung of Table~\\ref{tab:frontier_noneuc}, "
            "chosen for the sentence rather than measured. Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.noneuc.pk500_mape",
        anchor=r"at a budget of {~} reaches {v}\% MAPE at {~} times that cost",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "polyak_ckpt/500/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.noneuc.pk500_cost_x",
        anchor=r"MAPE at {v} times that cost, winning all {~} paired comparisons",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "polyak_ckpt/500/x_gart2_typical",
        tol=("dp", 2),
        note="The accuracy factor Section 6 quotes, now priced: it is bought below "
             "the estimator's own cost, not above it.",
    ),
    Claim(
        id="frontier.noneuc.pk500_win_n",
        anchor=r"winning all {v} paired comparisons; at a budget of",
        expect="costfront:cells/noneuc/instance_accounting/matched",
        tol="exact",
        note="A 100% paired win rate, so the count of wins is the matched count.",
    ),
    Claim(
        id="frontier.noneuc.pk200_k",
        anchor=r"paired comparisons; at a budget of {v} it reads {~}\% for",
        no_generator=(
            "An ascent budget: a rung of Table~\\ref{tab:frontier_noneuc}, chosen "
            "for the sentence rather than measured. Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.noneuc.pk200_mape",
        anchor=r"at a budget of {~} it reads {v}\% for {~} times the cost, again on every",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "polyak_ckpt/200/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.noneuc.pk200_cost_x",
        anchor=r"for {v} times the cost, again on every instance",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "polyak_ckpt/200/x_gart2_typical",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.noneuc.vj_crossing_k",
        anchor=r"the Volgenant--Jonker step at a budget of {v} costs {~} times GART 2.0",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/crossover/"
               "vj_ckpt/raw/k",
        tol="exact",
    ),
    Claim(
        id="frontier.noneuc.vj_crossing_cost_x",
        anchor=r"at a budget of {~} costs {v} times GART 2.0 and reads",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "vj_ckpt/25/x_gart2_typical",
        tol=("dp", 3),
        note="The cheapest strict domination in the study, which is why it is "
             "printed to three decimals rather than two.",
    ),
    Claim(
        id="frontier.noneuc.vj_crossing_mape",
        anchor=r"times GART 2.0 and reads {v}\% against {~}\%. The two step rules",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "vj_ckpt/25/raw_MAPE_pct",
        tol="printed",
    ),
    Claim(
        id="frontier.noneuc.vj_crossing_gart_mape",
        anchor=r"and reads {~}\% against {v}\%. The two step rules",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/gart2_MAPE_pct",
        tol="printed",
    ),
    Claim(
        id="frontier.noneuc.caption_n",
        anchor=r"ladder over the {v} instances both methods score",
        expect="costfront:cells/noneuc/instance_accounting/matched",
        tol="exact",
    ),

    # -- Section 5.4: the two step rules, scored against each other -----------
    # withdrawn frontier.steps.vj_lead_top_k: sentence corrected in the final consistency pass; 500 is no longer asserted there
    Claim(
        id="frontier.steps.vj_2d_mape",
        anchor=r"widest at a budget of {~}, {v}\% against {~}\%.",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/25/raw_MAPE_pct",
        tol="printed",
    ),
    Claim(
        id="frontier.steps.pk_2d_mape",
        anchor=r"widest at a budget of {~}, {~}\% against {v}\%.",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/polyak_ckpt/25/raw_MAPE_pct",
        tol="printed",
    ),
    Claim(
        id="frontier.steps.2d_k",
        anchor=r"the gap is widest at a budget of {v},",
        no_generator=(
            "An ascent budget: the rung at which the two step rules are compared on "
            "the 2D corpus. Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.steps.noneuc_k",
        anchor=r"Polyak leads from a budget of {v} on the non-Euclidean corpus",
        no_generator=(
            "An ascent budget: the rung from which the Polyak arm leads on the "
            "non-Euclidean corpus, read off Table~\\ref{tab:frontier_noneuc}. "
            "Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.steps.pk_noneuc_mape",
        anchor=r"on the non-Euclidean corpus, {v}\% against {~}\%. The timing sessions",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "polyak_ckpt/200/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.steps.vj_noneuc_mape",
        anchor=r"corpus, {~}\% against {v}\%. The timing sessions behind",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "vj_ckpt/200/raw_MAPE_pct",
        tol="printed",
    ),

    # -- Section 5.4: the validity gates over both cost tables ----------------
    # withdrawn frontier.gates.pairs_2d: cut in the verbosity sweep; QA paragraph removed; the released code carries the checks
    # withdrawn frontier.gates.pairs_noneuc: cut in the verbosity sweep; QA paragraph removed; the released code carries the checks
    # withdrawn frontier.gates.series_2d: cut in the verbosity sweep; QA paragraph removed; the released code carries the checks
    # withdrawn frontier.gates.series_noneuc: cut in the verbosity sweep; QA paragraph removed; the released code carries the checks

    # -- Section 5.4: what the loaded box does to the cost multiples ----------
    # The bias runs against the bound, so it is reported rather than corrected
    # for silently. Every entry here is a paired control measured in the same
    # session as the tables it qualifies.
    Claim(
        id="frontier.load.gart_inflation",
        anchor=r"puts GART 2.0 at {v} times its published cost and the bound at",
        expect="costfront:load_control/GART_2.0/median_today_over_published",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.load.bound_inflation_lo",
        anchor=r"and the bound at {v} to {~} times, so every cost multiple",
        expect="costfront:load_control/HK_1Tree_vj/0/median_today_over_published",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.load.bound_inflation_hi",
        anchor=r"and the bound at {~} to {v} times, so every cost multiple",
        expect="costfront:load_control/HK_1Tree_vj/200/median_today_over_published",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.load.bias_lo",
        anchor=r"cost multiple above is between {v} and {~} of its quiet-box value",
        expect="costfront:load_control/bias_in_a_cost_ratio/0",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.load.bias_hi",
        anchor=r"cost multiple above is between {~} and {v} of its quiet-box value",
        expect="costfront:load_control/bias_in_a_cost_ratio/200",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.load.flip_k",
        anchor=r"the raw Volgenant--Jonker row at a budget of {v} on the 2D corpus",
        no_generator=(
            "An ascent budget: the one rung whose domination the load correction "
            "removes, identified by comparing every rung's multiple against the "
            "per-budget bias. Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.load.flip_before",
        anchor=r"on the 2D corpus, whose {v} becomes {~}",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/200/x_gart2_typical",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.load.flip_after",
        anchor=r"whose {~} becomes {v}. Running the released",
        expect="= {costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/200/"
               "x_gart2_typical} / {costfront:load_control/bias_in_a_cost_ratio/200}",
        tol=("dp", 2),
        note="Stated as the corrected multiple, so the correction is what is "
             "checked rather than a hand-computed constant.",
    ),
    Claim(
        id="frontier.amort.2d_lo",
        anchor=r"sharing nothing between rungs, costs {v} to {~} times the checkpointed",
        expect="costfront:cells/2d/amortisation_control/polyak/25/median_direct_over_ckpt",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.amort.2d_hi",
        anchor=r"costs {~} to {v} times the checkpointed ladder on the 2D corpus",
        expect="costfront:cells/2d/amortisation_control/vj/500/median_direct_over_ckpt",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.amort.noneuc_lo",
        anchor=r"on the 2D corpus and {v} to {~} times on the non-Euclidean one",
        expect="costfront:cells/noneuc/amortisation_control/polyak/25/median_direct_over_ckpt",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.amort.noneuc_hi",
        anchor=r"and {~} to {v} times on the non-Euclidean one",
        expect="costfront:cells/noneuc/amortisation_control/vj/500/median_direct_over_ckpt",
        tol=("dp", 2),
    ),

    # =======================================================================
    # Editorial restructure of 2026-08-26.  The process/journey narrative was
    # removed per author directive; sections were reordered and retitled, and
    # the surviving facts moved to new homes (label validation into Section 3's
    # Label Validation subsection, label certification into Section 6's Label
    # Certification subsection).  Every tier-1 numeral the restructure surfaced
    # as unregistered is registered below rather than absorbed into the
    # baseline; tier-2 incidental integers ride the recorded backlog.
    # =======================================================================

    # -- introduction: two literature figures -------------------------------
    Claim(
        id="intro.lastmile.share_pct",
        anchor=r"accounts for approximately {v}\% of total shipping costs by one industry estimate",
        no_generator=(
            "Literature figure quoted from the cited industry estimate "
            "(finmile2025), not a quantity this project generates. Settle only "
            "against the source itself; no artifact can back it."
        ),
        tol="exact",
    ),
    Claim(
        id="related.varol.sub1pct",
        anchor=r"report sub-{v}\% deviation on standard distributions",
        no_generator=(
            "Literature figure quoted from varol2023neural's own reported "
            "accuracy, not a quantity this project generates. Settle only "
            "against the source itself."
        ),
        tol="exact",
    ),

    # -- Section 3: target preparation and the inverse transform ------------
    # withdrawn methods.target.train_outlier: clipping mechanics moved out of the body; app:code prints both raw values at full precision
    # withdrawn methods.target.val_outlier: clipping mechanics moved out of the body; app:code prints both raw values at full precision
    Claim(
        id="methods.transform.pred_lo",
        anchor=r"test predictions $\hat\alpha$ fell in $[{v},{~}]$",
        no_generator=(
            "Minimum released-model prediction over the 16,920 multidimensional "
            "test rows: 1.029, the check that the logit inverse is not "
            "saturating. Recomputed from the released per-instance predictions; "
            "not exported into paper_numbers.json. Settle by banking the "
            "prediction extrema under prediction_range_*."
        ),
        tol=("dp", 3),
    ),
    Claim(
        id="methods.transform.pred_hi",
        anchor=r"test predictions $\hat\alpha$ fell in $[{~},{v}]$",
        no_generator=(
            "Maximum released-model prediction over the 16,920 multidimensional "
            "test rows: 1.922. Companion to methods.transform.pred_lo; settle "
            "the same way."
        ),
        tol=("dp", 3),
    ),

    # -- Section 3.3, Label Validation: the two screens ---------------------
    Claim(
        id="provenance.quarantine.pct",
        anchor=r"The {~} instances ({v}\%) that fail are \emph{quarantined}",
        expect="= 100 * {frontier:labels/repair_quarantined_total}"
               " / {frontier:labels/repair_corpus_nd_instances}",
        tol=("dp", 3),
        note="The quarantine share, checked as the derivation 184/106,272 so "
             "the two counts and the percentage cannot drift apart.",
    ),
    Claim(
        id="provenance.verify.n",
        anchor=r"Across the {v} bound checks the released artifacts supply",
        expect="frontier:labels/repair_verify_instances_checked",
        tol="exact",
        note="The Label Validation copy of the verification count; the Label "
             "Certification copy of the same fact is labels.verify.n.",
    ),
    Claim(
        id="provenance.verify.worst_excess",
        anchor=r"the worst relative excess being ${v}\times10^{-15}$",
        no_generator=_WORST_EXCESS_REASON,
        tol=("dp", 1),
    ),
    Claim(
        id="labels.verify.worst_excess",
        anchor=r"zero violations of $B\le L$, the worst relative excess being ${v}\times10^{-15}$",
        no_generator=_WORST_EXCESS_REASON,
        tol=("dp", 1),
    ),

    # -- Section 4.3, the metric definitions --------------------------------
    # withdrawn metrics.sdpe.bias_example_pct: cut in the verbosity sweep; illustrative hypothetical, not a measurement
    # withdrawn metrics.sdpe.pct_factor: the prose gloss on the leading factor of 100 was cut in pass 1 (Section 4 lead, 2026-09-04); the 100 remains in the displayed SDPE formula
    Claim(
        id="metrics.ci.level_pct",
        anchor=r"Each SDPE carries a {v}\% bootstrap confidence interval, suppressed",
        no_generator=(
            "Confidence level of the bootstrap intervals, fixed in "
            "build_paper_tables.py; a protocol constant, not a measurement. "
            "Settle by banking the bootstrap protocol constants."
        ),
        tol="exact",
    ),

    # -- Section 4.4/4.5, results restatements ------------------------------
    Claim(
        id="results.tsplib95.euc2d_n",
        anchor=r"The benchmark contains {v} EUC\_2D instances with $n$ from",
        expect="bank:tsplib_by_size_total_gart_2_0_n",
        tol="exact",
    ),
    # withdrawn results_nd.sdpe_d2: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    # withdrawn results_nd.sdpe_d30_50: cut to the trend on 2026-09-02 after tab:results_nd / tab:results_2d took the values (author decision); the cells are machine-checked in the table
    Claim(
        id="results_nd.bhh_region_mape_restated",
        anchor=r"The constant dominates BHH's {v}\% error",
        expect="bank:nd_by_size_total_bhh_sampling_region_mape_pct",
        tol="printed",
        note="Restatement of results_nd.bhh_region_mape at the head of the "
             "paragraph that explains the figure.",
    ),
    Claim(
        id="results_nd.uniform_axes_share",
        anchor=r"Only {v}\% of axes are uniform, so the region measure",
        no_generator=(
            "Share of corpus axes with a uniform marginal: 68.7% of the "
            "2,444,256 axes, the census Section 3.2 prints from the corpus "
            "generation metadata. No artifact exports it. Settle by having "
            "corpus_statistics.py bank the axis-type census under "
            "corpus_axis_mix_*."
        ),
        tol=("dp", 1),
    ),
    Claim(
        id="results.tsplib.alpha_sd",
        anchor=r"has mean {~} and standard deviation {v}, so a single well-chosen constant",
        no_generator=(
            "Standard deviation of realized alpha over the 78 TSPLIB EUC_2D "
            "instances: 0.0558 (mean 1.1306), recomputed from the released "
            "per-instance results. No artifact exports the TSPLIB alpha "
            "moments. Settle by banking them under corpus_tsplib_alpha_*."
        ),
        tol=("dp", 4),
    ),
    Claim(
        id="results.tsplib.oracle_c",
        anchor=r"The MAPE-minimizing constant on these instances is ${v}$ and reaches {~}\%",
        no_generator=_ORACLE_78_REASON,
        tol=("dp", 4),
    ),
    Claim(
        id="results.tsplib.oracle_mape",
        anchor=r"constant on these instances is ${~}$ and reaches {v}\%, so no constant multiplier",
        no_generator=_ORACLE_78_REASON,
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.sdpe_overlap_lo",
        anchor=r"SDPE intervals overlap, on $[{v},{~}]$",
        expect="= max({tsplib_by_size_gt400_gart_2_0_sdpe_lo},"
               " {tsplib_by_size_gt400_asymptotic_mst_ratio_sdpe_lo})",
        tol=("dp", 2),
        note="The overlap of the two bucket CIs, checked as the derivation "
             "max(lo_a, lo_b) so the printed interval tracks both bands.",
    ),
    Claim(
        id="results.tsplib.sdpe_overlap_hi",
        anchor=r"SDPE intervals overlap, on $[{~},{v}]$",
        expect="= min({tsplib_by_size_gt400_gart_2_0_sdpe_hi},"
               " {tsplib_by_size_gt400_asymptotic_mst_ratio_sdpe_hi})",
        tol=("dp", 2),
    ),

    # -- Section 4.5, the generalization refits -----------------------------
    Claim(
        id="general.e0.mape",
        anchor=r"bit-for-bit at {v}\% MAPE over the {~} scored rows",
        expect="bank:nd_by_dim_total_gart_2_0_mape_pct",
        tol="printed",
        note="The baseline refit reproduces the released model bit-for-bit, so "
             "its MAPE is the released model's banked figure at four decimals.",
    ),
    Claim(
        id="general.e1.train_share",
        anchor=r"from training, {v}\% of the training rows, raises test MAPE",
        no_generator=_generalization(
            "E1 leave-dimensions-out: the d=15 and d=25 strata are 11.8% of "
            "the training rows"),
        tol=("dp", 1),
    ),
    Claim(
        id="general.e1.d15_before",
        anchor=r"raises test MAPE at $d=15$ from {v}\% to {~}\% and at $d=25$ from",
        no_generator=_generalization(
            "E1: test MAPE at d=15 under the full training set, 0.50%"),
        tol=("dp", 2),
    ),
    Claim(
        id="general.e1.d15_after",
        anchor=r"raises test MAPE at $d=15$ from {~}\% to {v}\% and at $d=25$ from",
        no_generator=_generalization(
            "E1: test MAPE at d=15 with d=15 and d=25 withheld, 0.58%"),
        tol=("dp", 2),
    ),
    Claim(
        id="general.e1.d25_before",
        anchor=r"and at $d=25$ from {v}\% to {~}\%. The neighboring",
        no_generator=_generalization(
            "E1: test MAPE at d=25 under the full training set, 0.35%"),
        tol=("dp", 2),
    ),
    Claim(
        id="general.e1.d25_after",
        anchor=r"and at $d=25$ from {~}\% to {v}\%. The neighboring",
        no_generator=_generalization(
            "E1: test MAPE at d=25 with d=15 and d=25 withheld, 0.38%"),
        tol=("dp", 2),
    ),
    Claim(
        id="general.e2.msigned_before",
        anchor=r"mean signed error rises from $+{v}$\% to $+{~}$\%, a factor of",
        no_generator=_generalization(
            "E2 leave-large-n-out: mean signed error on n in (200,1000] under "
            "the full training set, +0.28%"),
        tol=("dp", 2),
    ),
    Claim(
        id="general.e2.msigned_after",
        anchor=r"mean signed error rises from $+{~}$\% to $+{v}$\%, a factor of",
        no_generator=_generalization(
            "E2: mean signed error on n in (200,1000] when training only on "
            "n<=200, +0.51%"),
        tol=("dp", 2),
    ),
    Claim(
        id="general.e2.msigned_factor",
        anchor=r"a factor of {v}. The refits vary only the training data",
        no_generator=_generalization(
            "E2: the ratio of the two mean signed errors, 0.51/0.28 = 1.8"),
        tol=("dp", 1),
    ),
    # withdrawn coverage.decontaminated_slope: near-collinear section compressed (author 2026-09-03); the median-slope figure of the de-contaminated candidate is gone

    # -- Section 4.6, rank agreement and calibration ------------------------
    # withdrawn rank.global.gart_2d_rho: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.gart_2d_tau: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.gart_nd_rho: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.gart_nd_tau: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.gart_tsplib_rho: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.gart_tsplib_tau: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.floor_2d_rho: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.floor_2d_tau: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.floor_nd_rho: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.floor_nd_tau: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.floor_tsplib_rho: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    # withdrawn rank.global.floor_tsplib_tau: cut in the verbosity sweep; readback of tab:rank; the table carries the values
    Claim(
        id="rank.close5.threshold",
        anchor=r"For pairs satisfying $|L_A-L_B|/\max(L_A,L_B)<{v}\%$, GART 2.0 orders",
        no_generator=("Design constant, the narrower close-pair band. Same "
                      "status as rank.close10.threshold."),
        tol="exact",
    ),
    Claim(
        id="rank.close5.gart_2d",
        anchor=r"GART 2.0 orders {v}\% of {~} 2D pairs",
        expect="bank:rank_2d_gart_2_0_close5_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close5.gart_nd",
        anchor=r"2D pairs, {v}\% of {~} multidimensional pairs",
        expect="bank:rank_nd_gart_2_0_close5_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="discussion.gt400.mape",
        anchor=r"while its tour-cost MAPE remains {v}\%, so the cost-level result",
        expect="bank:tsplib_by_size_gt400_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="discussion.paired.ci_level_pct",
        anchor=r"percentage points with a {v}\% interval of $[-{~},-{~}]$",
        no_generator=(
            "Confidence level of the paired bootstrap interval, fixed in "
            "build_paper_tables.py; a protocol constant, not a measurement. "
            "Settle by banking the bootstrap protocol constants."
        ),
        tol="exact",
    ),
    # withdrawn discussion.disp.gart_sdpe: the 2.92/4.74 SDPE restatement was cut from 4.7 in pass 1 (2026-09-04); the figures stay in 4.3.3 under results.tsplib.sdpe
    # withdrawn discussion.disp.asym_sdpe: the 2.92/4.74 SDPE restatement was cut from 4.7 in pass 1 (2026-09-04); the figures stay in 4.3.3 under results.tsplib.sdpe

    # -- Section 4.7, cost accounting ---------------------------------------
    Claim(
        id="costacct.lgbm_share_pct",
        anchor=r"against {v}\% on LightGBM inference",
        no_generator=_timing(
            "tsplib_by_size_time_one_protocol: the lgbm inference share of "
            "total_time_s over the 78 EUC_2D instances, 5.03%"),
        tol=("dp", 2),
    ),
    # withdrawn costacct.mst_rows_lo: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn costacct.mst_rows_hi: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    # withdrawn costacct.gt400_gart_ms: cut in the 2026-09-02 tightening pass (author-approved pairs); the sentence carrying the value is gone
    Claim(
        id="costacct.nd_median_gart_ms",
        anchor=r"cost the same, {v}~ms against {~}~ms, since most",
        expect="bank:nd_by_size_total_gart_2_0_time_ms",
        tol=("dp", 0),
    ),
    Claim(
        id="costacct.nd_median_ref_ms",
        anchor=r"cost the same, {~}~ms against {v}~ms, since most",
        no_generator=(
            "Median reference-tour generation time over the multidimensional "
            "benchmark: 122 ms, from the released benchmark harness timing "
            "columns, which paper_numbers.json does not aggregate. Settle by "
            "exporting reference-tour timing under nd_reference_time_* keys."
        ),
        tol=("dp", 0),
    ),
    Claim(
        id="costacct.nd_large_ref_ms",
        anchor=r"the reference tour takes {v}~ms against GART 2.0's {~}~ms",
        no_generator=(
            "Median reference-tour generation time on the n in [501,1000] "
            "bucket of the multidimensional benchmark: 3,799 ms, from the "
            "released benchmark harness timing columns, which "
            "paper_numbers.json does not aggregate. Settle by exporting "
            "reference-tour timing under nd_reference_time_* keys."
        ),
        tol=("dp", 0),
    ),
    Claim(
        id="costacct.d18512_scale",
        anchor=r"and more than ${v}\times$ the largest training instance",
        no_generator=(
            "d18512's node count as a multiple of the n=1000 training cap: "
            "18,512 / 1,000 = 18.5, printed floored as 'more than 18x'. "
            "Arithmetic over two constants asserted elsewhere; settle by "
            "deriving the multiple in the bank."
        ),
        tol="exact",
    ),

    # -- the ascent-budget-25 rung, at its three sites ----------------------
    Claim(
        id="abstract.frontier.budget25",
        anchor=r"strictly better than the bound on both axes from an ascent budget of {v} upward. The reason is structural",
        no_generator=_BUDGET25_REASON,
        tol="exact",
    ),
    Claim(
        id="frontier.verdict.budget25",
        anchor=r"strictly better than the bound on both axes from an ascent budget of {v} upward in the plane",
        no_generator=_BUDGET25_REASON,
        tol="exact",
    ),
    # withdrawn conclusion.frontier.budget25: the conclusion sentence carrying the budget was cut in the 2026-09-02 tightening pass; abstract and verdict copies remain registered

    # -- Section 6: the fixed multiplier and the oracle constants -----------
    Claim(
        id="application.fixed_alpha.constant",
        anchor=r"a fixed $\alpha={v}$ close to the MAPE-minimizing constant",
        no_generator=_FIXED_ALPHA_REASON,
        tol=("dp", 3),
    ),
    Claim(
        id="application.oracle_constant_111",
        anchor=r"close to the MAPE-minimizing constant ${v}$ over the full {~}-instance TSPLIB set",
        no_generator=_ORACLE_111_REASON,
        tol=("dp", 3),
    ),
    Claim(
        id="application.oracle_c_23_restated",
        anchor=r"The MAPE-minimizing constant over these {~} instances is ${v}$ and reaches",
        no_generator=_ORACLE_REASON,
        tol=("dp", 4),
    ),
    Claim(
        id="application.oracle_mape_23_restated",
        anchor=r"over these {~} instances is ${~}$ and reaches {v}\%, the floor any constant",
        no_generator=_ORACLE_REASON,
        tol=("dp", 2),
    ),
    Claim(
        id="application.oracle_mape_23_floor_repeat",
        anchor=r"clears the {v}\% constant-multiplier floor",
        no_generator=_ORACLE_REASON,
        tol=("dp", 2),
    ),
    # withdrawn application.fixed_alpha.constant_scored: Total-row readback cut per author; the values are printed and machine-checked in tab:tsplib_nonEuc
    # withdrawn application.oracle_mape_23_total: Total-row readback cut per author; the values are printed and machine-checked in tab:tsplib_nonEuc

    # -- appendices ----------------------------------------------------------
    Claim(
        id="appendix.training.test_share_restated",
        anchor=r"lifts the corpus-level test share to the ${v}\%$ above",
        expect="= 100 * {sidecar:rows.test} / {sidecar:rows.total}",
        tol="printed",
        note="Checked as the derivation 16,920/106,272 over the frozen sidecar "
             "so the share follows whatever model ships.",
    ),
    # withdrawn appendix.lit.chien_daganzo_cavdar: Appendix F rewritten wholesale (author 2026-09-03): the unscored estimators get one attribution sentence; the secondary-record coefficient discussion is gone
    # withdrawn appendix.lit.chien_daganzo_choi: Appendix F rewritten wholesale (author 2026-09-03): coefficient discussion gone
    # withdrawn appendix.lit.chien_review_coeff: Appendix F rewritten wholesale (author 2026-09-03): coefficient discussion gone
    # withdrawn appendix.lit.chien_formula_coeff: Appendix F rewritten wholesale (author 2026-09-03): coefficient discussion gone
    Claim(
        id="appendix.controls.linear_clustered_mape",
        anchor=r"reaching {v}\% MAPE on the clustered class against {~}\%, eight times",
        expect="bank:2d_by_genclass_clustered_linear_28_feature_block_mape_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="appendix.controls.released_clustered_mape",
        anchor=r"on the clustered class against {v}\%, eight times the released model's error",
        expect="bank:2d_by_genclass_clustered_gart_2_0_mape_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="appendix.controls.linear_clustered_r2alpha",
        anchor=r"eight times the released model's error, at $R^2_\alpha=-{v}$",
        expect="= -1 * {2d_by_genclass_clustered_linear_28_feature_block_r2_alpha}",
        tol=("dp", 2),
        note="Adverse result kept explicit; the minus sign is in the anchor so "
             "a sign flip reports ANCHOR_MISSING.",
    ),
    # -----------------------------------------------------------------------
    # Near-collinear section (author 2026-09-03): SHAP attribution of the released
    # booster by 2D generator class, paper_tooling/shap_2d_summary.json, written
    # by paper_tooling/shap_2d_by_class.py.
    # -----------------------------------------------------------------------
    Claim(
        id="shap.n_instances",
        anchor=r"run over the {v} diverse 2D benchmark instances",
        expect="shap2d:n_instances",
        tol="exact",
    ),
    Claim(
        id="shap.isotropic.n",
        anchor=r"On the Isotropic class ({v} instances)",
        expect="shap2d:alpha/by_class/Isotropic/n",
        tol="exact",
    ),
    Claim(
        id="shap.isotropic.greedy_share",
        anchor=r"Greedy-to-MST Ratio carries {v}\% of the total mean",
        expect="shap2d:top5_by_class/Isotropic/0/share_pct",
    ),
    Claim(
        id="shap.isotropic.dominance_share",
        anchor=r"the MST Dominance Ratio is next at {v}\%",
        expect="shap2d:top5_by_class/Isotropic/1/share_pct",
    ),
    Claim(
        id="shap.isotropic.dimension_share",
        anchor=r"then dimension at {v}\%",
        expect="shap2d:top5_by_class/Isotropic/2/share_pct",
    ),
    Claim(
        id="shap.isotropic.n_share",
        anchor=r"$n$ at {v}\%, and normalized",
        expect="shap2d:top5_by_class/Isotropic/3/share_pct",
    ),
    Claim(
        id="shap.isotropic.diameter_share",
        anchor=r"normalized MST diameter at {v}\%",
        expect="shap2d:top5_by_class/Isotropic/4/share_pct",
    ),
    Claim(
        id="shap.isotropic.pred_alpha",
        anchor=r"Isotropic predictions are unbiased: mean predicted $\alpha$ is {v} against a mean true $\alpha$ of {~}",
        expect="shap2d:alpha/by_class/Isotropic/mean_pred_alpha",
    ),
    Claim(
        id="shap.isotropic.true_alpha",
        anchor=r"Isotropic predictions are unbiased: mean predicted $\alpha$ is {~} against a mean true $\alpha$ of {v}",
        expect="shap2d:alpha/by_class/Isotropic/mean_true_alpha",
    ),
    Claim(
        id="shap.linenoise.n",
        anchor=r"On Line Noise ({v} instances) the same feature",
        expect="shap2d:alpha/by_class/LineNoise/n",
        tol="exact",
    ),
    Claim(
        id="shap.greedy.signed_iso",
        anchor=r"rises from $+{v}$ on Isotropic to $+{~}$ on Line Noise",
        expect="shap2d:line_noise_vs_isotropic_signed_shap/mean_signed_shap_isotropic/greedy_nn_over_mst",
    ),
    Claim(
        id="shap.greedy.signed_ln",
        anchor=r"rises from $+{~}$ on Isotropic to $+{v}$ on Line Noise",
        expect="shap2d:line_noise_vs_isotropic_signed_shap/mean_signed_shap_linenoise/greedy_nn_over_mst",
    ),
    Claim(
        id="shap.dominance.signed_iso",
        anchor=r"MST Dominance Ratio's from $+{v}$ to $+{~}$, both",
        expect="shap2d:line_noise_vs_isotropic_signed_shap/mean_signed_shap_isotropic/mst_dominance_ratio",
    ),
    Claim(
        id="shap.dominance.signed_ln",
        anchor=r"MST Dominance Ratio's from $+{~}$ to $+{v}$, both",
        expect="shap2d:line_noise_vs_isotropic_signed_shap/mean_signed_shap_linenoise/mst_dominance_ratio",
    ),
    Claim(
        id="shap.linenoise.pred_alpha",
        anchor=r"The prediction still falls short: mean predicted $\alpha$ is {v} against a mean true $\alpha$ of {~}",
        expect="shap2d:alpha/by_class/LineNoise/mean_pred_alpha",
    ),
    Claim(
        id="shap.linenoise.true_alpha",
        anchor=r"The prediction still falls short: mean predicted $\alpha$ is {~} against a mean true $\alpha$ of {v}",
        expect="shap2d:alpha/by_class/LineNoise/mean_true_alpha",
    ),
    Claim(
        id="shap.linenoise.shortfall",
        anchor=r"a shortfall of {v}, the largest of any class",
        expect="= -1 * {shap2d:alpha/by_class/LineNoise/mean_signed_error_pred_minus_true}",
    ),
    Claim(
        id="shap.grid.shortfall",
        anchor=r"the jittered-grid class is next at {v}",
        expect="shap2d:alpha/by_class/GeometricGrid/mean_signed_error_pred_minus_true",
    ),
    Claim(
        id="shap.train.greedy_p1",
        anchor=r"Greedy-to-MST Ratio spans $[{v},{~}]$ between its 1st and 99th",
        expect="shap2d:line_noise_out_of_training_support/train_p1_p99/greedy_nn_over_mst/0",
    ),
    Claim(
        id="shap.train.greedy_p99",
        anchor=r"Greedy-to-MST Ratio spans $[{~},{v}]$ between its 1st and 99th",
        expect="shap2d:line_noise_out_of_training_support/train_p1_p99/greedy_nn_over_mst/1",
    ),
    Claim(
        id="shap.linenoise.greedy_mean",
        anchor=r"on Line Noise it averages {v}, and",
        expect="shap2d:linenoise_feature_stats/greedy_nn_over_mst/mean",
    ),
    Claim(
        id="shap.linenoise.greedy_outside",
        anchor=r"and {v}\% of the class's rows fall outside that band",
        expect="shap2d:line_noise_out_of_training_support/frac_linenoise_outside_train_p1_p99/greedy_nn_over_mst",
        scale=100.0,
    ),
    Claim(
        id="shap.linenoise.dominance_outside",
        anchor=r"the MST Dominance Ratio is {v}\% outside",
        expect="shap2d:line_noise_out_of_training_support/frac_linenoise_outside_train_p1_p99/mst_dominance_ratio",
        scale=100.0,
    ),
    Claim(
        id="shap.linenoise.others_outside",
        anchor=r"dimension, $n$, and MST diameter are {v}\% outside",
        expect="shap2d:line_noise_out_of_training_support/frac_linenoise_outside_train_p1_p99/dimension",
        scale=100.0, tol="exact",
    ),
    # -----------------------------------------------------------------------
    # Pass 1, Section 4.3 (2026-09-04): three prose-only figures the critic asked
    # to see registered (BHH convex-hull form on the multidimensional split;
    # Hilbert sort MAPE and SDPE on the 2D diverse benchmark).
    # -----------------------------------------------------------------------
    Claim(
        id="results_nd.bhh_hull_mape",
        anchor=r"cuts its MAPE from {v}\% to 28.0\%",
        expect="bank:nd_by_size_total_bhh_mape_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="results_2d.hilbert_mape",
        anchor=r"The custom Hilbert sort has {v}\% MAPE against",
        expect="bank:2d_by_genclass_total_custom_hilbert_sort_mape_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="results_2d.hilbert_sdpe",
        anchor=r"MAPE against {v}\% SDPE, so its error is",
        expect="bank:2d_by_genclass_total_custom_hilbert_sort_sdpe_pct",
        tol=("dp", 1),
    ),
    # -----------------------------------------------------------------------
    # Pass 1, Section 4.6 (2026-09-04): the released model's TSPLIB paired p was
    # printed as 0.0048 (truncated) against 0.0049 in 4.7; corrected and registered.
    # -----------------------------------------------------------------------
    Claim(
        id="coverage.paired.tsplib_asym.p_restated",
        anchor=r"against {v} for the released model",
        expect="bank:paired_tsplib_by_size_total_asymptotic_mst_ratio_wilcoxon_p",
        tol=("abs", 0.0001),  # p-values print at two figures; exempt from 3 s.f. (author, 2026-09-03)
    ),
    # -----------------------------------------------------------------------
    # Pass 1, Section 7 (2026-09-04): the conclusion restates four body figures
    # (Section 6 non-Euclidean bound vs GART 2.0; Section 4.6 class MAPEs).
    # -----------------------------------------------------------------------
    Claim(
        id="conclusion.bound.vj_k100",
        anchor=r"the bound obtains {v}\% MAPE at an ascent budget of {~} against GART 2.0's",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/vj_raw_MAPE_by_k/100",
        tol=("dp", 2),
    ),
    Claim(
        id="conclusion.bound.gart_mape",
        anchor=r"against GART 2.0's {v}\%. Above a few hundred nodes",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/GART_2.0_MAPE_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="conclusion.coverage.linenoise_mape",
        anchor=r"Error on the near-collinear class is {v}\% MAPE against",
        expect="bank:2d_by_genclass_linenoise_gart_2_0_mape_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="conclusion.coverage.isotropic_mape",
        anchor=r"MAPE against {v}\% on the isotropic class",
        expect="bank:2d_by_genclass_isotropic_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
]
