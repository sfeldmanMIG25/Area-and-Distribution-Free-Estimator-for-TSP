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
                    latter's [1,10) neighbours.
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
# Feature counts of the two predecessor-vector controls.  Read off the shipped
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

# The MAPE-minimising constant multiple of L_MST over the 23 screened
# non-EUC_2D instances.  Flagged UNGENERATED in prose_claims.py as
# app.oracle_constant_23; one exporter covers the 78, the 111 and the 23.
_ORACLE_REASON = (
    "MAPE-minimising constant multiple of L_MST over the 23 screened non-EUC_2D "
    "TSPLIB95 instances: c* = 1.1718 reaching 7.55% MAPE. Recorded in "
    "paper_tooling/prose_claims.py as app.oracle_constant_23, state CORRECT / "
    "UNGENERATED. Settle with the oracle-constant exporter named there: "
    "oracle_constant_<set>_{c,mape_pct}, model-independent, covering the 78 "
    "EUC_2D, the 111 and these 23."
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


# The shipped GART 2.0 booster is absent from v4_study._model_registry(), so the
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


def _zcell(row: str) -> str:
    """Reason text for a provenance alpha z-score: exact file and printed field."""
    return (
        f"paper_tooling/audit_alpha_cell_zscores.py, {row}. Convention stated in "
        "that module's docstring: per (d,n) cell, the affected rows' mean "
        "alpha_stored standardized by the mean and ddof=1 standard deviation of "
        "the same cell's unaffected rows, reading "
        "paper_tooling/reference_tour_audit.csv and writing "
        "paper_tooling/reference_tour_alpha_zscores.csv. Not exported into "
        "paper_numbers.json. Settle by banking that CSV under "
        "provenance_alpha_z_* keys."
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
    """Reason for an endpoint of the shipped greedy-ratio coverage gate."""
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
        anchor=r"Of the {~} scored instances {v}, or 62.74",
        expect="sizestrat:corpus/n_le_100_instances",
    ),
    Claim(
        id="sizestrat.corpus.scored",
        anchor=r"Of the {v} scored instances {~}, or 62.74",
        expect="sizestrat:corpus/scored_instances",
    ),
    Claim(
        id="sizestrat.summary.undominated_cells",
        anchor=r"GART 2.0 is undominated in {v} of the 18 cells",
        expect="sizestrat:summary/cells_gart_not_dominated",
    ),
    Claim(
        id="sizestrat.summary.total_cells",
        anchor=r"undominated in 10 of the {v} cells",
        expect="sizestrat:summary/cells_total",
    ),
    Claim(
        id="sizestrat.d2.large.mape",
        anchor=r"GART 2.0 reaches {v}\% MAPE at {~}~ms at $d=2$",
        expect="sizestrat:cell/d2/n600_1000/gart_mape_pct",
    ),
    Claim(
        id="sizestrat.d2.large.ms",
        anchor=r"GART 2.0 reaches {~}\% MAPE at {v}~ms at $d=2$",
        expect="sizestrat:cell/d2/n600_1000/gart_ms",
    ),
    Claim(
        id="sizestrat.d2.large.k500_mape",
        anchor=r"is both dearer and looser at {v}\% and {~}~ms",
        expect="sizestrat:cell/d2/n600_1000/bound_k500_mape_pct",
    ),
    Claim(
        id="sizestrat.d2.large.k500_ms",
        anchor=r"is both dearer and looser at {~}\% and {v}~ms",
        expect="sizestrat:cell/d2/n600_1000/bound_k500_ms",
    ),
    Claim(
        id="sizestrat.d3.large.mape",
        anchor=r"{v}\% at {~}~ms at $d=3$, against",
        expect="sizestrat:cell/d3/n600_1000/gart_mape_pct",
    ),
    Claim(
        id="sizestrat.d3.large.ms",
        anchor=r"{~}\% at {v}~ms at $d=3$, against",
        expect="sizestrat:cell/d3/n600_1000/gart_ms",
    ),
    Claim(
        id="sizestrat.d3.large.k200_mape",
        anchor=r"at $d=3$, against {v}\% at {~}~ms for $k=200$",
        expect="sizestrat:cell/d3/n600_1000/bound_k200_mape_pct",
    ),
    Claim(
        id="sizestrat.d3.large.k200_ms",
        anchor=r"at $d=3$, against {~}\% at {v}~ms for $k=200$",
        expect="sizestrat:cell/d3/n600_1000/bound_k200_ms",
    ),
    Claim(
        id="sizestrat.d4.large.mape",
        anchor=r"and {v}\% at {~}~ms at $d=4$, against",
        expect="sizestrat:cell/d4/n600_1000/gart_mape_pct",
    ),
    Claim(
        id="sizestrat.d4.large.ms",
        anchor=r"and {~}\% at {v}~ms at $d=4$, against",
        expect="sizestrat:cell/d4/n600_1000/gart_ms",
    ),
    Claim(
        id="sizestrat.d4.large.k200_mape",
        anchor=r"at $d=4$, against {v}\% at {~}~ms for the same budget",
        expect="sizestrat:cell/d4/n600_1000/bound_k200_mape_pct",
    ),
    Claim(
        id="sizestrat.d4.large.k200_ms",
        anchor=r"at $d=4$, against {~}\% at {v}~ms for the same budget",
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
    Claim(
        id="methods.model.n_features",
        anchor=r"The following subsections describe its {v} input features",
        expect="sidecar:n_features",
        tol="exact",
    ),
    Claim(
        id="methods.model.n_trees",
        anchor=r"The resulting ensemble contains {v} trees with {~} leaves per tree",
        expect="sidecar:num_trees",
        tol="exact",
    ),
    Claim(
        id="methods.model.leaves_per_tree",
        anchor=r"ensemble contains {~} trees with {v} leaves per tree",
        expect="sidecar:hyperparameters.num_leaves",
        tol="exact",
    ),

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
    Claim(
        id="metrics.bhh_region.mspe_uniform",
        anchor=r"BHH given the exact sampling region carries a $-{v}$\% offset",
        expect="= -1 * {2d_random_by_size_total_bhh_sampling_region_mspe_pct}",
        tol=("dp", 2),
        note="Replaces metrics.daganzo.sdpe_uniform. The example of a bias that "
             "SDPE alone cannot see used to be Daganzo's strip constant; that "
             "estimator is withdrawn (unobtainable primary), so the sentence now "
             "uses BHH on the same i.i.d.-uniform subset, where the signed error "
             "-8.65 exceeds the 7.76 dispersion and makes the same point. The "
             "prose prints the magnitude after a literal minus sign, hence the "
             "sign flip in the expression.",
    ),
    Claim(
        id="metrics.bhh_region.sdpe_uniform",
        anchor=r"offset on uniform instances with only {v}\% SDPE",
        expect="bank:2d_random_by_size_total_bhh_sampling_region_sdpe_pct",
        tol=("dp", 2),
        note="The *_sampling_region_* variant is the right provenance: the "
             "sentence says 'given the exact sampling region'. The plain BHH key "
             "for the same subset reads 17.63.",
    ),

    # -----------------------------------------------------------------------
    # SIGNIFICANCE
    #
    # Sign convention, inherited from build_paper_tables.paired_test and carried
    # unchanged into the bank: mean_diff / ci_lo / ci_hi are |APE| of the
    # production model minus |APE| of the baseline, so a NEGATIVE value favours
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
        anchor=r"$ and $p={v}$: on that benchmark",
        expect="bank:paired_tsplib_by_size_total_asymptotic_mst_ratio_wilcoxon_p",
        tol=("dp", 2),
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
        anchor=r"GART 2.0 obtains {v}\% MAPE and {~}\% SDPE overall. The strongest",
        expect="bank:nd_by_dim_total_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.nd.sdpe",
        anchor=r"GART 2.0 obtains {~}\% MAPE and {v}\% SDPE overall. The strongest",
        expect="bank:nd_by_dim_total_gart_2_0_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.2d.mape",
        anchor=r"GART 2.0 obtains {v}\% MAPE and {~}\% SDPE overall, against",
        expect="bank:2d_by_size_total_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.2d.sdpe",
        anchor=r"GART 2.0 obtains {~}\% MAPE and {v}\% SDPE overall, against",
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
        anchor=r"and the $\alpha=1$ floor's {v}\%. A factor of",
        expect="bank:classical_b_random_l_mathrm_mst_alpha_1_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="discussion.r2alpha.tsplib_gt400",
        anchor=r"TSPLIB bucket has $R^2_\alpha=-{v}$,",
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
        anchor=r"contributes {v}\% of the total magnitude and",
        expect="bank:shap_feature_mst_dominance_ratio_share_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="methods.shap.greedy_share",
        anchor=r"total magnitude and \texttt{greedy\_nn\_over\_mst} {v}\%",
        expect="bank:shap_feature_greedy_nn_over_mst_share_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="methods.shap.top2_share",
        anchor=r"those two features carry {v}\% of it between them. Four of the top ten features are MST-derived, five are geometric, and the greedy ratio is the remaining one. Node count and dimension jointly contribute {~}\%, the two bounding-hypervolume features",
        expect="= {shap_feature_mst_dominance_ratio_share_pct}"
               " + {shap_feature_greedy_nn_over_mst_share_pct}",
        tol=("dp", 1),
        note="Anchor runs long only to separate this copy from the appendix one, "
             "whose sentence says 'bounding-hypervolume descriptors' where "
             "Section 3.6 says 'features'.",
    ),
    Claim(
        id="methods.shap.size_dimension_share",
        anchor=r"Node count and dimension jointly contribute {v}\%, the two bounding-hypervolume features",
        expect="bank:shap_family_size_dimension_share_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="methods.shap.bounding_share",
        anchor=r"the two bounding-hypervolume features {v}\%",
        expect="bank:shap_family_bounding_hypervolume_share_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="methods.shap.centroid_share",
        anchor=r"the four centroid-distance descriptors {v}\%. The greedy ratio ranking second",
        expect="bank:shap_family_centroid_share_pct",
        tol=("dp", 1),
    ),
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
    Claim(
        id="app.oracle_constant.c",
        anchor=r"constant over these instances is ${v}$ at {~}\%. MSPE",
        no_generator=_ORACLE_REASON,
        tol=("dp", 4),
    ),
    Claim(
        id="app.oracle_constant.mape",
        anchor=r"constant over these instances is ${~}$ at {v}\%. MSPE",
        no_generator=_ORACLE_REASON,
        tol=("dp", 2),
    ),

    # =======================================================================
    # Re-audit pass of 2026-08-11: seven findings closed.  Every number the
    # rewrite introduced is registered here rather than baselined.
    # =======================================================================

    # -- N1: the 2D superlative.  NN_V3 is one of the thirteen enumerated
    #    baselines and a printed row of tab:2d_by_size, and its aggregate SDPE
    #    is below GART 2.0's, so "lowest MAPE and SDPE of every baseline except
    #    the refitted network" was false on the 2D stratum.
    Claim(
        id="discussion.timing.classical_lo",
        anchor=r"an order of magnitude over the {v}--{~}~ms of the two classical closed forms",
        expect="bank:tsplib_by_size_total_bhh_time_ms",
        tol=("dp", 3),
        note="Was keyed on Daganzo; BHH is the cheapest surviving classical row.",
    ),
    Claim(
        id="discussion.timing.classical_hi",
        anchor=r"an order of magnitude over the {~}--{v}~ms of the two classical closed forms",
        expect="bank:tsplib_by_size_total_cavdar_sokol_time_ms",
        tol=("dp", 3),
    ),

    # -- N4: SDPE is the primary precision metric (Section 4.3), and on the
    #    Kwon domain the V3-feature network's SDPE is below GART 2.0's, so
    #    "wins on every matched domain" failed on the paper's own metric.
    Claim(
        id="provenance.z.n_cells",
        anchor=r"The {~} rows fall in {v} $(d,n)$ cells",
        no_generator=_zcell("printed field 'cells containing an affected row' = 29"),
        tol="exact",
    ),
    Claim(
        id="provenance.z.n_rows",
        anchor=r"The {v} rows fall in {~} $(d,n)$ cells",
        no_generator=_zcell("printed field 'affected rows covered' = 184"),
        tol="exact",
    ),
    Claim(
        id="provenance.z.min_reference",
        anchor=r"none of which retains fewer than {v} unaffected rows",
        no_generator=_zcell("printed field 'smallest reference group' = 219"),
        tol="exact",
    ),
    Claim(
        id="provenance.z.n_over",
        anchor=r"{v} of the {~} cells depart by more than {~} standard deviations",
        no_generator=_zcell("printed field 'cells with |z| > 1.6' = 17"),
        tol="exact",
    ),
    Claim(
        id="provenance.z.n_cells_again",
        anchor=r"{~} of the {v} cells depart by more than {~} standard deviations",
        no_generator=_zcell("printed field 'cells containing an affected row' = 29"),
        tol="exact",
    ),
    Claim(
        id="provenance.z.threshold",
        anchor=r"depart by more than {v} standard deviations and those",
        no_generator=_zcell(
            "the module constant THRESHOLD = 1.6, the bound the superseded "
            "sentence asserted and the one this sentence reports against"),
        tol=("dp", 1),
    ),
    Claim(
        id="provenance.z.rows_over",
        anchor=r"and those {~} hold {v} of the {~} rows",
        no_generator=_zcell("printed field 'cells with |z| > 1.6' holding-count = 95"),
        tol="exact",
    ),
    Claim(
        id="provenance.z.n_over_again",
        anchor=r"and those {v} hold {~} of the {~} rows",
        no_generator=_zcell("printed field 'cells with |z| > 1.6' = 17"),
        tol="exact",
    ),
    Claim(
        id="provenance.z.n_rows_again",
        anchor=r"hold {~} of the {v} rows; the largest departure",
        no_generator=_zcell("printed field 'affected rows covered' = 184"),
        tol="exact",
    ),
    Claim(
        id="provenance.z.max",
        anchor=r"the largest departure is $-{v}$, at $d={~}$",
        no_generator=_zcell(
            "printed field 'largest departure' = -15.9261, the cell d=25, n=1000, "
            "which holds 1 affected row against 245 unaffected"),
        tol=("dp", 2),
    ),
    Claim(
        id="provenance.z.max_d",
        anchor=r"the largest departure is $-{~}$, at $d={v}$",
        no_generator=_zcell(
            "printed field 'largest departure' cell coordinate d = 25"),
        tol="exact",
    ),

    # -- N7: GART 2.0 (V4 features).  Scored on every benchmark, present in the
    #    tidy tables, absent from the manuscript, and better than the shipped
    #    model on three of the four strata.  Enumerated here with the reason it
    #    was not shipped.  The shipped model's own probe row did not exist --
    #    v4_study._model_registry() omits PRODUCTION_BOOSTER -- so it was
    #    re-measured under the identical protocol.

    # -- appendices ----------------------------------------------------------
    Claim(
        id="appendix.code.table_cells",
        anchor=r"re-derives all {v} table cells",
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
        anchor=r"(maximum stress {v} for GEO and {~} for screened EXPLICIT)",
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
        anchor=r"which holds monotonicity on {v}\% of the dimension sweeps and",
        expect="bank:cons_probe_gart2_logit_v3hp_dimension_pct_nonincr_deployed",
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
    Claim(
        id="application.noneuc.screen_n",
        anchor=r"On the {~} of the {v} screened instances it accepts",
        expect="bank:tsplib_nonEuc_total_fixed_alpha_1_136_n",
        tol="exact",
    ),
    Claim(
        id="application.noneuc.accept_n",
        anchor=r"On the {v} of the {~} screened instances it accepts",
        expect="bank:tsplib_nonEuc_total_gart_2_0_n",
        tol="exact",
    ),

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
    Claim(
        id="results.tsplib.cavdar_mape",
        anchor=r"\c{C}avdar--Sokol obtains {v}\% MAPE and BHH",
        expect="bank:classical_a_tsplib_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.bhh_mape",
        anchor=r"MAPE and BHH {v}\%. TSPLIB instances",
        expect="bank:classical_a_tsplib_bhh_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.bhh_mspe",
        anchor=r"Both overpredict, by $+{v}$\% and",
        expect="bank:classical_a_tsplib_bhh_mspe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.cavdar_mspe",
        anchor=r"by $+{~}$\% and $+{v}$\% respectively",
        expect="bank:classical_a_tsplib_cavdar_sokol_mspe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.bhh.full_mape",
        anchor=r"BHH falls from {v}\% MAPE on the full 2D benchmark",
        expect="bank:classical_a_2d_bhh_mape_pct",
        tol=("dp", 2),
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
        anchor=r"\% signed against {v}\% SDPE, which is what an asymptotic",
        expect="bank:classical_b_random_bhh_sampling_region_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.cavdar.full_mape",
        anchor=r"\c{C}avdar--Sokol falls from {v}\% to",
        expect="bank:classical_a_2d_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.cavdar.uniform_mape",
        anchor=r"\c{C}avdar--Sokol falls from {~}\% to {v}\%, and here only",
        expect="bank:classical_b_random_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.cavdar.uniform_sdpe",
        anchor=r"the least even: at {v}\% SDPE its dispersion",
        expect="bank:classical_b_random_cavdar_sokol_sdpe_pct",
        tol=("dp", 2),
        note="Adverse result: the widest dispersion in the matched panel, "
             "wider than the alpha=1 floor's. Registered so a later run cannot "
             "quietly drop the qualification.",
    ),
    Claim(
        id="matched.cavdar.uniform_medape",
        anchor=r"its median absolute error of {v}\% sits at just over a third",
        expect="bank:classical_b_random_cavdar_sokol_medape_pct",
        tol=("dp", 2),
        note="The anchor read 'under a third' until the final verification "
             "pass. 2.84/8.16 = 0.348, which is over a third. The relation is a "
             "word, so no numeric check could have caught it; the wording now "
             "matches the two numbers either side of it.",
    ),
    Claim(
        id="matched.uniform_domain.n",
        anchor=r"suggest. On the {v} uniform instances it obtains",
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
        anchor=r"That leaves a {v}\% step at the upper boundary",
        expect="bank:cavdar_corr_step_at_n_max_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="appendix.cavdar.boundary_ratio",
        anchor=r"where the fitted ratio reads {v}.",
        expect="bank:cavdar_corr_ratio_at_n_max",
        tol=("dp", 3),
    ),
    Claim(
        id="appendix.classical.panel_b_n",
        anchor=r"The lower panel restricts to the {v} i.i.d.",
        expect="bank:classical_b_random_gart_2_0_n",
        tol="exact",
    ),
    Claim(
        id="matched.cavdar_factor",
        anchor=r"A factor of {v} over \c{C}avdar--Sokol on i.i.d.\ uniform draws",
        expect="= {classical_b_random_cavdar_sokol_mape_pct}"
               " / {classical_b_random_gart_2_0_mape_pct}",
        tol=("dp", 1),
        note="Replaces matched.kwon_factor. Kwon--Golden--Wasil was the strongest "
             "classical estimator this paper reported and is now withdrawn "
             "(unobtainable primary), so the headline factor is taken over the "
             "strongest surviving one: 8.162786 / 1.313575 = 6.214 on the 210 "
             "i.i.d.-uniform instances. The factor is LARGER than the 2.7 it "
             "replaces because the comparator is weaker on its own domain, not "
             "because GART 2.0 improved -- 1.313575 is unchanged.",
    ),
    Claim(
        id="discussion.rank.tsplib_close5_gart1",
        anchor=r"and trails it after, at {v}\%, the only ordering change",
        expect="bank:rank_tsplib_euc2d_gart_1_0_close5_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="discussion.rank.tsplib_close5_pairs",
        anchor=r"distinct properties, and {v} pairs resolve none of these gaps",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="application.noneuc.gart_sdpe",
        anchor=r"GART 2.0 has aggregate SDPE {v}\% and MAPE",
        expect="bank:tsplib_nonEuc_total_gart_2_0_sdpe_pct",
        tol=("dp", 2),
    ),
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
        anchor=r"the training split alone spans the narrower $[{v},2.1295]$",
        no_generator=(
            "Minimum of greedy_nn_over_mst over split=='train' in tsp_features_v4.csv "
            "(69,768 rows): 1.046482. The shipped constant "
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
        anchor=r"the training split alone spans the narrower $[{~},{v}]$",
        no_generator=(
            "Maximum of greedy_nn_over_mst over split=='train' in tsp_features_v4.csv "
            "(69,768 rows): 2.129495. Companion to methods.greedy_gate.train_lo; "
            "settle the same way."
        ),
        tol=("dp", 4),
    ),
    Claim(
        id="application.greedy_gate.train_lo",
        anchor=r"the training split's own minimum is ${v}$, and \texttt{si1032} is below both",
        no_generator=(
            "Minimum of greedy_nn_over_mst over split=='train' in tsp_features_v4.csv: "
            "1.046482. Same fact as methods.greedy_gate.train_lo, restated in the "
            "tab:tsplib_nonEuc caption so the decline decision states which floor it "
            "used. si1032's 1.0260 is below both floors, so the decision is invariant "
            "to the choice. Settle as for methods.greedy_gate.train_lo."
        ),
        tol=("dp", 4),
    ),

    # -- N-J: the tuning comparison, and which two boosters it is over -------
    Claim(
        id="methods.optuna.trials_complete",
        anchor=r"200 trials, {v} of them completed and 140 pruned",
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
        anchor=r"200 trials, {~} of them completed and {v} pruned",
        no_generator=(
            "lgbm_model_v3/gart2_optuna.db, study 'gart2': PRUNED 140 of 200 trials. "
            "Companion to methods.optuna.trials_complete; settle the same way."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.optuna.tuned_nd",
        anchor=r"The tuned booster reaches {v}\% on the multidimensional test split",
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
        anchor=r"against {v}\% for the same booster on the frozen hyperparameters",
        no_generator=(
            "paper_tooling/v4_study_allmodels_strata.csv, row model=GART2_logit_v3hp, "
            "stratum=nd_test: mape 0.622598. This is the control the tuning sentence "
            "is measured against -- same logit target, same unconstrained fit, "
            "hyperparameters the only difference -- and 0.622598 - 0.611239 = 0.011359 "
            "is the 0.011 the sentence quotes. Scoring the tuned booster against the "
            "SHIPPED model instead gives 0.008876, because the shipped model also "
            "carries the monotone constraint; that pairing would confound the search "
            "with the constraint. Settle by banking v4_study_allmodels_strata.csv."
        ),
        tol=("dp", 4),
    ),
    Claim(
        id="methods.optuna.tuned_tsplib",
        anchor=r"the frozen hyperparameters, and {v}\% against 2.4725\% on TSPLIB EUC\_2D",
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
    Claim(
        id="methods.probe.grid_points",
        anchor=r"swept on log-spaced grids of {v} points",
        expect="bank:cons_probe_gart_2_0_n_customers_grid_points",
        note=(
            "paper_tooling/v4_study.py::PROBE_GRID_POINTS = 24, the requested grid "
            "size for both axes; the n grid keeps all 24 and the d grid dedupes to 22 "
            "distinct integers after rounding. Recorded in "
            "paper_tooling/v4_study_gart2_probe.csv column grid_points. A design "
            "constant of the probe, not a measurement. Settle by banking the probe "
            "protocol constants under probe_* keys."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.probe.d_lo",
        anchor=r"points, $d$ from {v} to 200 and $n$ from 5 to 4{,}000",
        no_generator=(
            "paper_tooling/v4_study.py::PROBE_D_GRID = _log_int_grid(2, 200, 24); the "
            "lower endpoint is 2. Protocol constant. Settle by banking the probe "
            "protocol constants under probe_* keys."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.probe.d_hi",
        anchor=r"points, $d$ from {~} to {v} and $n$ from 5 to 4{,}000",
        no_generator=(
            "paper_tooling/v4_study.py::PROBE_D_GRID upper endpoint = 200, twice the "
            "largest dimension evaluated anywhere in this paper (d=100). Protocol "
            "constant. Settle by banking the probe protocol constants under probe_*."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.probe.n_lo",
        anchor=r"$d$ from 2 to 200 and $n$ from {v} to 4{,}000",
        no_generator=(
            "paper_tooling/v4_study.py::PROBE_N_GRID = _log_int_grid(5, 4000, 24); the "
            "lower endpoint is 5. Protocol constant. Settle by banking the probe "
            "protocol constants under probe_* keys."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.probe.n_hi",
        anchor=r"$n$ from {~} to {v}, against a tolerance",
        no_generator=(
            "paper_tooling/v4_study.py::PROBE_N_GRID upper endpoint = 4000, four times "
            "the largest node count in the SYNTHETIC corpora (n=1000). It is NOT four "
            "times the largest node count this paper evaluates: the TSPLIB EUC_2D "
            "benchmark runs to n=18512 (d18512) and pla85900 is scored at n=85900, so "
            "the grid reaches 4.7 percent of the largest evaluated n. Until 2026-08-11 "
            "this reason carried the 'in the synthetic corpora' qualifier while the "
            "manuscript sentence claimed four times the largest node count anything in "
            "the paper is evaluated at; the sentence was narrowed to match. Protocol "
            "constant. Settle by banking the probe protocol constants under probe_* keys."
        ),
        tol="exact",
    ),
    Claim(
        id="methods.probe.tolerance_base",
        anchor=r"against a tolerance of ${v}^{-9}$ in $\alpha$ units",
        no_generator=(
            "paper_tooling/v4_study.py::PROBE_TOL = 1e-9, the acceptance tolerance in "
            "alpha units below which a sweep difference is counted as float noise "
            "rather than as a monotonicity violation. The manuscript prints it as "
            "10^-9, so the base 10 is the numeral tokenised here. Protocol constant. "
            "Settle by banking the probe protocol constants under probe_* keys."
        ),
        tol="exact",
    ),

    # -- N-A: the rho(d) -> rho(d,n) gap is not the largest step -------------
    Claim(
        id="results_2d.grid_n",
        anchor=r"On the {v} \texttt{grid} instances GART 2.0 reads",
        expect="bank:2d_by_genclass_geometric_grid_gart_2_0_n",
        tol="exact",
    ),
    Claim(
        id="results_2d.grid_mape",
        anchor=r"\texttt{grid} instances GART 2.0 reads {v}\% MAPE against",
        expect="bank:2d_by_genclass_geometric_grid_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results_2d.geom_other_mape",
        anchor=r"MAPE against {v}\% on the 420 Geometric Struct.\ instances that are represented",
        expect="bank:2d_by_genclass_geometric_other_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results_2d.geom_other_n",
        anchor=r"MAPE against {~}\% on the {v} Geometric Struct.\ instances that are represented",
        expect="bank:2d_by_genclass_geometric_other_gart_2_0_n",
        tol="exact",
    ),
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
    Claim(
        id="discussion.genclass.geom_other_mape",
        anchor=r"on the biased class, {v}\% on the two represented geometric-structure generators",
        expect="bank:2d_by_genclass_geometric_other_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="discussion.genclass.grid_mape",
        anchor=r"on the clustered class, {v}\% on the \texttt{grid} generator",
        expect="bank:2d_by_genclass_geometric_grid_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="discussion.genclass.worst_represented",
        anchor=r"no represented row exceeds {v}\%, which locates the model's weakness",
        expect="bank:2d_by_genclass_clustered_gart_2_0_mape_pct",
        tol=("dp", 2),
        note="Clustered is the worst of the four represented rows at 2.1847; the two "
             "unrepresented rows, grid at 7.1121 and Line Noise at 10.7522, are the "
             "only ones above it.",
    ),
    Claim(
        id="discussion.genclass.grid_sdpe",
        anchor=r"the failure is almost pure bias: SDPE is {v}, the lowest GART 2.0 records",
        expect="bank:2d_by_genclass_geometric_grid_gart_2_0_sdpe_pct",
        tol=("dp", 2),
        note="1.9225, against 2.2309 / 2.5157 / 2.6082 / 2.7188 / 6.1656 on the other "
             "five rows. The error on grid is a level shift, not scatter: MSPE equals "
             "MAPE to the last digit at +7.1121.",
    ),
    Claim(
        id="discussion.genclass.grid_floor_mape",
        anchor=r"the $\alpha=1$ floor at {v}\%, and \texttt{grid} is the only row",
        expect="bank:2d_by_genclass_geometric_grid_l_mathrm_mst_alpha_1_mape_pct",
        tol=("dp", 2),
        note="4.5047 against GART 2.0's 7.1121. grid is the only bucket anywhere in "
             "the paper where the alpha=1 floor is more accurate than the shipped "
             "model on MAPE.",
    ),
    Claim(
        id="appendix.genclass.geom_class_n",
        anchor=r"Reported inside the {v}-instance class aggregate",
        expect="= {2d_by_genclass_geometric_grid_gart_2_0_n}"
               " + {2d_by_genclass_geometric_other_gart_2_0_n}",
        tol="exact",
        note="630, the Geometric Struct. count of tab:dataset_counts, which the two "
             "reported sub-rows still sum to.",
    ),
    Claim(
        id="appendix.genclass.geom_class_mspe",
        anchor=r"carries that aggregate to $+{v}$ MSPE",
        expect="= ({2d_by_genclass_geometric_grid_gart_2_0_n}"
               " * {2d_by_genclass_geometric_grid_gart_2_0_mspe_pct}"
               " + {2d_by_genclass_geometric_other_gart_2_0_n}"
               " * {2d_by_genclass_geometric_other_gart_2_0_mspe_pct})"
               " / ({2d_by_genclass_geometric_grid_gart_2_0_n}"
               " + {2d_by_genclass_geometric_other_gart_2_0_n})",
        tol=("dp", 2),
        note="The instance-weighted mean of the two sub-rows reproduces the +2.7842 "
             "the undivided Geometric row used to print, which is the point: grid's "
             "+7.1121 over 210 instances is what carried it.",
    ),
    Claim(
        id="appendix.genclass.geom_other_mspe",
        anchor=r"while the two represented generators sit at $+{v}$",
        expect="bank:2d_by_genclass_geometric_other_gart_2_0_mspe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="appendix.genclass.benchmark_n",
        anchor=r"the six row counts still sum to the {v} instances of the benchmark",
        expect="bank:2d_by_genclass_total_gart_2_0_n",
        tol="exact",
        note="build_paper_tables.load_2d asserts this sum: each class count is checked "
             "against GEN_CLASSES and their total against the benchmark's instance "
             "count, so a split that dropped or double-counted a generator fails there "
             "rather than printing a wrong Total.",
    ),

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
    Claim(
        id="application.greedy_gate.corpus_lo",
        anchor=r"falls below the in-distribution floor ${v}$ the gate applies",
        no_generator=_GATE_CONST("lower endpoint 1.035, the floor si1032 fails"),
        tol=("dp", 3),
    ),
    Claim(
        id="application.si1032.greedy_ratio",
        anchor=r"\texttt{si1032}, whose greedy-to-MST ratio ${v}$ falls below",
        no_generator=(
            "greedy_nn_over_mst of si1032 on the hybrid non-Euclidean path: 1.0260. "
            "Computed inside tsplib_benchmark/run_all_models_tsplib.py::_run_one_instance "
            "and used only by the coverage gate, so the instance is recorded with a "
            "status row rather than a prediction and the value reaches no results CSV. "
            "Settle by persisting the gate input on declined rows."
        ),
        tol=("dp", 4),
    ),
    Claim(
        id="application.noneuc.gart_scored_n",
        anchor=r"GART 2.0 is therefore scored on {v} instances against 23 for the reference",
        expect="bank:tsplib_nonEuc_total_gart_2_0_n",
        tol="exact",
    ),
    Claim(
        id="application.noneuc.reference_scored_n",
        anchor=r"scored on {~} instances against {v} for the reference",
        expect="bank:tsplib_nonEuc_total_fixed_alpha_1_136_n",
        tol="exact",
        note="Fixed_Alpha scores all 23; GART 2.0 declines si1032. The asymmetry "
             "is why the Total row is not a like-for-like comparison.",
    ),
    Claim(
        id="methods.optuna.n_trials",
        anchor=r"on TSPLIB, and {v} trials, 60 of them completed",
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
            "difference; see methods.optuna.untuned_nd for why the shipped model is "
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
        anchor=r"held-out instances accordingly returns {v}\% non-increasing sweeps on both axes",
        expect="bank:cons_probe_gart_2_0_dimension_pct_nonincr_deployed",
        tol=("dp", 1),
        note="The dimension and node-count axes both read 100.0 with zero violations, "
             "so either key backs the sentence; the dimension one is named because it "
             "is the axis the comparators fail worst on.",
    ),
    Claim(
        id="results_nd.bhh_region_mape",
        anchor=r"BHH given the exact sampling region {v}\%/{~}\%.",
        expect="bank:nd_by_size_total_bhh_sampling_region_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results_nd.bhh_region_sdpe",
        anchor=r"BHH given the exact sampling region {~}\%/{v}\%.",
        expect="bank:nd_by_size_total_bhh_sampling_region_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results_nd.sdpe_smallest_bucket",
        anchor=r"size buckets, from {v}\% at $n\le10$",
        expect="bank:nd_by_size_5_10_gart_2_0_sdpe_pct",
        tol=("dp", 2),
        note="Re-anchored 2026-08-11: the anchor used to carry the tail of the "
             "preceding sentence, which now ends with the pointer to "
             "Section~\\ref{subsec:frontier_nd} instead of 'most of it.'.",
    ),
    Claim(
        id="results_nd.rho_dn_mape",
        anchor=r"the calibrated ratio $\hat\rho(d,n)$ at {v}\%/{~}\%, so the learned",
        expect="bank:nd_by_dim_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct",
        tol=("dp", 2),
        note="The strongest non-learned baseline IN THE ROSTER. The sentence "
             "now says so, because the Held--Karp bound of "
             "Section~\\ref{sec:frontier} uses no learned model either and "
             "beats GART 2.0 on this benchmark.",
    ),
    Claim(
        id="results_nd.sdpe_201_500",
        anchor=r"at $n\le10$ to {v}\% at $n\in[201,500]$",
        expect="bank:nd_by_size_201_500_gart_2_0_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results_nd.bucket_201_lo",
        anchor=r"to 0.46\% at $n\in[{v},500]$ and 0.45",
        no_generator=_SIZE_BUCKET,
        tol="exact",
    ),
    Claim(
        id="results_nd.bucket_500_hi",
        anchor=r"to 0.46\% at $n\in[{~},{v}]$ and 0.45",
        no_generator=_SIZE_BUCKET,
        tol="exact",
    ),
    Claim(
        id="discussion.genclass.isotropic_mape",
        anchor=r"GART 2.0's MAPE is {v}\% on the isotropic class",
        expect="bank:2d_by_genclass_isotropic_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="discussion.genclass.biased_mape",
        anchor=r"on the isotropic class, {v}\% on the biased class",
        expect="bank:2d_by_genclass_biased_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="discussion.genclass.clustered_mape",
        anchor=r"geometric-structure generators, {v}\% on the clustered class",
        expect="bank:2d_by_genclass_clustered_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="discussion.genclass.linenoise_mape",
        anchor=r"\texttt{grid} generator, and {v}\% on Line Noise. The two worst rows",
        expect="bank:2d_by_genclass_linenoise_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
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
    Claim(
        id="provenance.certified_wrong_total",
        anchor=r"certifies {v} stored costs across the whole corpus",
        expect="frontier:labels/total_proven_wrong",
        tol="exact",
    ),
    Claim(
        id="provenance.certified_wrong_in_184",
        anchor=r"coordinates, and {v} of those {~} are inside these {~} rows",
        expect="frontier:labels/by_tour_audit_bucket/corrupt",
        tol="exact",
    ),
    Claim(
        id="provenance.certified_wrong_total_restated",
        anchor=r"coordinates, and {~} of those {v} are inside these {~} rows",
        expect="frontier:labels/total_proven_wrong",
        tol="exact",
    ),
    Claim(
        id="provenance.corrupt_tour_count_in_new_sentence",
        anchor=r"coordinates, and {~} of those {~} are inside these {v} rows",
        no_generator=(
            "Size of the reference-tour audit's corrupt bucket, 184. Recomputed "
            "by paper_tooling/audit_reference_tours.py into "
            "reference_tour_audit.csv, rows with bucket=='corrupt'; that file "
            "is not exported into paper_numbers.json, so there is no bank key. "
            "Settle by having audit_reference_tours.py emit its four bucket "
            "counts under provenance_bucket_* keys."
        ),
        tol="exact",
    ),
    Claim(
        id="provenance.corrupt_tour_count_removal",
        anchor=r"removing all {v} instances from the multidimensional benchmark",
        no_generator=(
            "Same 184 as provenance.corrupt_tour_count_in_new_sentence, in the "
            "sentence that follows it. Pre-existing prose; it re-keys against "
            "prose_baseline only because the sentence before it was rewritten "
            "when the 'stale field is the tour permutation' claim was withdrawn."
        ),
        tol="exact",
    ),

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
    Claim(
        id="frontier.ascent.vj_higher_pct",
        anchor=r"the higher bound on {v}\% of instances",
        expect="frontier:nd/vj_higher_than_polyak_pct_at_k2000",
        tol=("dp", 2),
        note="Evidence that neither ascent attains max_pi w(pi), which is why "
             "every accuracy figure in this section is a floor.",
    ),
    Claim(
        id="frontier.ascent.converged_single",
        anchor=r"multidimensional error from {v}\% to {~}\%",
        expect="frontier:nd/bound_mape_pct_by_k/2000",
        tol=("dp", 4),
    ),
    Claim(
        id="frontier.ascent.converged_envelope",
        anchor=r"multidimensional error from {~}\% to {v}\%",
        expect="frontier:nd/two_ascent_envelope_mape_pct_by_k/2000",
        tol=("dp", 4),
    ),

    # -- 5.2 TSPLIB ----------------------------------------------------------
    Claim(
        id="frontier.tsplib.repeats",
        anchor=r"a single thread, the median of {v} repeats",
        expect="frontier:tsplib/repeats",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.matched_n",
        anchor=r"both arms are compared on the {v} of the {~} EUC\_2D instances",
        expect="frontier:tsplib/N_matched",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.euc2d_n",
        anchor=r"both arms are compared on the {~} of the {v} EUC\_2D instances",
        expect="frontier:tsplib/N_euc2d",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.raw_crossing_k",
        anchor=r"MAPE at an ascent budget of {v}, where it costs {~} times GART 2.0",
        expect="frontier:tsplib/crossing_ladder_k",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.raw_crossing_cost_x",
        anchor=r"MAPE at an ascent budget of {~}, where it costs {v} times GART 2.0",
        expect="frontier:tsplib/crossing_cost_x_gart2",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.raw_crossing_margin",
        anchor=r"times GART 2.0 for a margin of {v} percentage points",
        expect="= {frontier:tsplib/gart2_mape_pct}"
               " - {frontier:tsplib/crossing_bound_mape_pct}",
        tol=("dp", 3),
        note="Stated as a margin, so the margin is what is checked. It is 0.04 "
             "points, which is why the printed crossing rung is not a robust "
             "statistic; see frontier.labels.interp_crossing_excluded.",
    ),
    Claim(
        id="frontier.tsplib.paired_win_rate",
        anchor=r"where the paired win rate is exactly {v}\%",
        expect="frontier:tsplib/paired_win_rate_of_bound_at_crossing_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.tsplib.cal_crossing_k",
        anchor=r"At a budget of {v} it costs {~} times GART 2.0 and reaches",
        expect="frontier:tsplib_calibrated_bound/Total (all EUC_2D)/crossing_ladder_k",
        tol="exact",
    ),
    Claim(
        id="frontier.tsplib.cal_crossing_cost_x",
        anchor=r"At a budget of {~} it costs {v} times GART 2.0 and reaches",
        expect="frontier:tsplib_calibrated_bound/Total (all EUC_2D)/x_gart2_by_k/25",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.cal_crossing_mape",
        anchor=r"times GART 2.0 and reaches {v}\% MAPE against {~}\%: strict domination",
        expect="frontier:tsplib_calibrated_bound/Total (all EUC_2D)/mape_pct_by_k/25",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.cal_crossing_gart2",
        anchor=r"times GART 2.0 and reaches {~}\% MAPE against {v}\%: strict domination",
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
        note="The uncalibrated, certified bound against the shipped estimator "
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
        anchor=r"its median rising by a factor of {v} between a quiet and a noisy window",
        expect="frontier:tsplib/load_sensitivity/HK_1Tree_50/noisy_over_quiet",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.tsplib.load_sensitivity_gart2",
        anchor=r"at the crossing budget against {v} for GART 2.0",
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
        anchor=r"that single instance takes {v} seconds against GART 2.0's",
        expect="frontier:tsplib/capped_tail/hk_ms_by_k/500",
        tol=("dp", 0),
        scale=0.001,
        note="Bank stores milliseconds; the sentence prints seconds.",
    ),
    Claim(
        id="frontier.tsplib.capped_gart2_ms",
        anchor=r"seconds against GART 2.0's {v} milliseconds on those same three repeats",
        expect="frontier:tsplib/capped_tail/gart2_ms",
        tol=("dp", 0),
    ),
    Claim(
        id="frontier.tsplib.harness_d18512_ms",
        anchor=r"the {v}~ms quoted for this instance in",
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
        anchor=r"cost/accuracy ladder, {v} instances matched between both arms",
        expect="frontier:tsplib/N_matched",
        tol="exact",
    ),

    # -- 5.3 multidimensional ------------------------------------------------
    Claim(
        id="frontier.nd.best_budget",
        anchor=r"At an ascent budget of {v} the bound reaches",
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
    Claim(
        id="frontier.nd.corpus_weighted_budget",
        anchor=r"the cost ratio at a budget of {v} is {~}",
        expect="frontier:nd/best_budget_k",
        tol="exact",
    ),
    Claim(
        id="frontier.nd.corpus_weighted_cost_x",
        anchor=r"the cost ratio at a budget of {~} is {v}",
        expect="frontier:nd/bound_x_gart2_corpus_weighted_by_k/200",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.nd.d100_budget",
        anchor=r"a budget of {v} costs {~} times GART 2.0 and is {~} times more accurate",
        expect="frontier:nd/pareto_by_group/d = 100/best/k",
        tol="exact",
    ),
    Claim(
        id="frontier.nd.d100_cost_x",
        anchor=r"a budget of {~} costs {v} times GART 2.0 and is {~} times more accurate",
        expect="frontier:nd/pareto_by_group/d = 100/best/cost_x",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.nd.d100_accuracy_factor",
        anchor=r"a budget of {~} costs {~} times GART 2.0 and is {v} times more accurate",
        expect="frontier:nd/pareto_by_group/d = 100/best/accuracy_factor",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.nd.planar_group_cost_x",
        anchor=r"where matching its accuracy costs the bound {v} times as much",
        expect="frontier:nd/crossover_cost_x_by_group/d in {2,3}",
        tol=("dp", 2),
        note="The one dimension group on which GART 2.0 stays on the front.",
    ),
    Claim(
        id="frontier.nd.concorde_subset_n",
        anchor=r"On the {v} instances whose label came from an exact solver",
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
        anchor=r"against the bound's {~}\% at a budget of {v}. And the relaxation",
        expect="frontier:nd/best_budget_k",
        tol="exact",
    ),
    Claim(
        id="frontier.nd.closes_exactly_pct",
        anchor=r"rather than a bound, on {v}\% of the split",
        expect="frontier:nd/relaxation_closes_exactly_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.nd.table_caption_n",
        anchor=r"cost/accuracy ladder, all {v} held-out instances, Polyak ascent",
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
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.complexity.nd_bound_hi",
        # Re-anchored 2026-08-12: see nd_bound_lo above.
        anchor=r"against {~}--{v} for the bound, because the MST construction",
        expect="frontier:complexity/nd_slopes/onetree_nd_hi",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.complexity.nd_gart2_planar",
        anchor=r"is that exponent {v}. On that benchmark both families are quadratic",
        expect="frontier:complexity/nd_slopes/gart2_d_2_3",
        tol=("dp", 2),
    ),

    # -- 5.5 exact solver ----------------------------------------------------
    Claim(
        id="frontier.exact.n_solves",
        anchor=r"Over the {v} TSPLIB instances for which a Concorde solve time is on record",
        expect="frontier:exact_solver_anchor/N_instances_with_a_recorded_solve",
        tol="exact",
    ),
    Claim(
        id="frontier.exact.median_seconds",
        anchor=r"the published median is {v} seconds against GART 2.0's",
        expect="frontier:exact_solver_anchor/median_s",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.exact.gart2_ms",
        anchor=r"seconds against GART 2.0's {v} milliseconds, and the recorded range",
        expect="frontier:exact_solver_anchor/gart2_median_ms",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.exact.min_seconds",
        anchor=r"the recorded range runs from {v} seconds to more than eleven million",
        expect="frontier:exact_solver_anchor/min_s",
        tol=("dp", 2),
    ),

    # -- 5.6 labels ----------------------------------------------------------
    Claim(
        id="frontier.labels.total_evaluated",
        anchor=r"Applying it to all {v} labelled instances in this project",
        expect="frontier:labels/total_evaluated",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.refuted_total",
        anchor=r"against the released labels, refutes {v} of them, and {~} of those sit in a set",
        expect="frontier:labels/total_proven_wrong",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.refuted_scored",
        anchor=r"against the released labels, refutes {~} of them, and {v} of those sit in a set",
        expect="frontier:labels/proven_wrong_in_a_scored_set",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.refuted_nd_test",
        anchor=r"By split they are {v} in the multidimensional test partition",
        expect="frontier:labels/proven_wrong_nd_test",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.worst_excess",
        anchor=r"the worst overshoots its label by {v}\%",
        expect="frontier:labels/worst_excess_pct_nd_test",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.labels.overlap_count",
        anchor=r"is near-total, since {v} of the {~} lie inside that section's {~} inconsistent-tour instances",
        expect="frontier:labels/by_tour_audit_bucket/corrupt",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.overlap_total",
        anchor=r"is near-total, since {~} of the {v} lie inside that section's {~} inconsistent-tour instances",
        expect="frontier:labels/total_proven_wrong",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.corrupt_population",
        anchor=r"lie inside that section's {v} inconsistent-tour instances",
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
        anchor=r"Its stored label, {v}, is the fixed-edge Hamiltonian-path optimum",
        expect="frontier:labels/linhp318/stored_label",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.linhp318_tour_opt",
        anchor=r"the tour optimum on those coordinates is {v}",
        expect="frontier:labels/linhp318/tour_optimum_on_its_coordinates",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.linhp318_bound",
        anchor=r"with no slack at all, is {v}. Every estimator",
        expect="frontier:labels/linhp318/onetree_bound_integer_metric",
        tol=("dp", 0),
    ),
    Claim(
        id="frontier.labels.tsplib_files_scanned",
        anchor=r"A scan of all {v} files found no second unrecognised section keyword",
        expect="frontier:labels/linhp318/tsplib_files_scanned",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.gart2_mape_as_published",
        anchor=r"matched-corpus MAPE from {v}\% to {~}\% and leaves the raw",
        expect="frontier:tsplib_label_variants/as_published/gart2_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.labels.gart2_mape_repaired",
        anchor=r"matched-corpus MAPE from {~}\% to {v}\% and leaves the raw",
        expect="frontier:tsplib_label_variants/linhp318_repaired/gart2_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.labels.interp_crossing_as_published",
        anchor=r"the interpolated crossing moves only from {v} to {~} under the repair",
        expect="frontier:tsplib_label_variants/as_published/crossing_interp_k",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.labels.interp_crossing_repaired",
        anchor=r"moves only from {~} to {v} under the repair and to",
        expect="frontier:tsplib_label_variants/linhp318_repaired/crossing_interp_k",
        tol=("dp", 1),
        note="The robust statistic. The printed ladder rung moves 50 -> 100 "
             "over a 0.04-point margin; the interpolated crossing moves 4.",
    ),
    Claim(
        id="frontier.labels.repaired_margin",
        anchor=r"cuts the margin at that rung from {~} to {v} percentage points",
        expect="= {frontier:tsplib_label_variants/linhp318_repaired/crossing_margin_pp}",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.labels.cal_margin",
        anchor=r"because its margin is {v} percentage points rather than {~}",
        expect="frontier:tsplib_calibrated_bound/Total (all EUC_2D)/crossing_margin_pp",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.labels.repaired_margin_restated",
        anchor=r"its margin is {~} percentage points rather than {v}.",
        expect="= {frontier:tsplib_label_variants/linhp318_repaired/crossing_margin_pp}",
        tol=("dp", 3),
    ),

    # -- 5.7 verdict ---------------------------------------------------------
    Claim(
        id="frontier.verdict.cheaper_above_n",
        anchor=r"it is cheaper above roughly {v} nodes in the plane",
        no_generator=(
            "Upper edge of the TSPLIB size buckets of Table "
            "\\ref{tab:tsplib_by_size}, n=400, quoted as the size above which "
            "the ordering favours GART 2.0. It names a bucket boundary this "
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
        anchor=r"and dearer below roughly {v}. And it is not uniformly closer",
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
        anchor=r"set against {v} points from the converged bound on the same set",
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
        anchor=r"the 2D gain of {v} points is not",
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
        anchor=r"threshold GART obtains {v}\%, {~}\%, and {~}\%",
        expect="bank:rank_2d_gart_2_0_close10_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close10.gart_nd",
        anchor=r"threshold GART obtains {~}\%, {v}\%, and {~}\%",
        expect="bank:rank_nd_gart_2_0_close10_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="rank.close10.gart_tsplib",
        anchor=r"threshold GART obtains {~}\%, {~}\%, and {v}\%",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close10_pct",
        tol=("dp", 1),
    ),
    Claim(
        id="appendix.rank.nd_universe",
        anchor=r"$\binom{16846}{2}={v}$ on the multidimensional set",
        expect=("= {rank_nd_gart_2_0_n} * ({rank_nd_gart_2_0_n} - 1) / 2"),
        tol="exact",
    ),
    Claim(
        id="appendix.rank.nd_n",
        anchor=r"$\binom{{v}}{2}=141{,}885{,}435$ on the multidimensional set",
        expect="bank:rank_nd_gart_2_0_n",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.quarantined_nd",
        anchor=r"multidimensional set, the {v} quarantined instances of",
        expect="frontier:labels/repair_quarantined_nd_test",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.nd_close5_pairs",
        anchor=r"Of the multidimensional pairs {v} qualify at the 5\% threshold",
        expect="bank:rank_nd_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.nd_close10_pairs",
        anchor=r"qualify at the 5\% threshold and {v} at 10\%",
        expect="bank:rank_nd_gart_2_0_close10_pairs",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.2d_close5_pairs",
        anchor=r"the corresponding 2D counts are {v} and {~},",
        expect="bank:rank_2d_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.2d_close10_pairs",
        anchor=r"the corresponding 2D counts are {~} and {v},",
        expect="bank:rank_2d_gart_2_0_close10_pairs",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.tsplib_close5_pairs",
        anchor=r"the TSPLIB EUC\_2D counts {v} and {~}.",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.tsplib_close10_pairs",
        anchor=r"the TSPLIB EUC\_2D counts {~} and {v}.",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close10_pairs",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.direct_scan_pairs",
        anchor=r"agree pair-for-pair with a direct scan of all {v} pairs.",
        expect=("= {rank_nd_gart_2_0_n} * ({rank_nd_gart_2_0_n} - 1) / 2"),
        tol="exact",
    ),
    Claim(
        id="rank.close5.pairs_tsplib_restated",
        anchor=r"and on TSPLIB {v} pairs cannot support",
        expect="bank:rank_tsplib_euc2d_gart_2_0_close5_pairs",
        tol="exact",
    ),
    Claim(
        id="rank.close10.threshold",
        anchor=r"claim. At a {v}\% threshold GART obtains",
        no_generator=("Design constant, not a measurement: the wider of the two "
                      "close-pair bands, fixed a priori in build_paper_tables. "
                      "The bank stores the cells the threshold produces, not "
                      "the threshold."),
        tol="exact",
    ),
    Claim(
        id="appendix.rank.close5_threshold",
        anchor=r"qualify at the {v}\% threshold and",
        no_generator=("Design constant, the narrower close-pair band. Same "
                      "status as rank.close10.threshold."),
        tol="exact",
    ),
    Claim(
        id="appendix.rank.tsplib_universe",
        anchor=r"$\binom{78}{2}={v}$ on TSPLIB EUC\_2D",
        expect="= {tsplib_by_size_total_gart_2_0_n} * ({tsplib_by_size_total_gart_2_0_n} - 1) / 2",
        tol="exact",
    ),
    Claim(
        id="appendix.rank.nd_universe_rounded",
        anchor=r"The {v} million multidimensional pairs are never held at once",
        no_generator=("Rounded restatement of appendix.rank.nd_universe "
                      "(141,885,435) in the same paragraph, to three "
                      "significant figures. The exact figure is checked one "
                      "sentence earlier."),
        tol="exact",
    ),
    # -- Section 5.8, the repair. Numbers come from
    # paper_tooling/labels_repaired.json via frontier_manuscript_bank.json.
    Claim(
        id="dataset.repair.corpus_pct",
        anchor=r"returned a cost that is wrong on {v}\% of the corpus",
        expect="frontier:labels/repair_nd_bad_label_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="provenance.quarantine.nd_test",
        anchor=r"{v} of them fall in the multidimensional test partition",
        expect="frontier:labels/repair_quarantined_nd_test",
        tol="exact",
    ),
    Claim(
        id="provenance.quarantine.scored_nd",
        anchor=r"scored on the remaining {v} instances throughout",
        expect="frontier:labels/repair_nd_test_scored",
        tol="exact",
    ),
    Claim(
        id="labels.mech.d1",
        anchor=r"one path: {v} carry the coarse verification scale",
        expect="frontier:labels/repair_nd_d1_coarse_robust_scale",
        tol="exact",
    ),
    Claim(
        id="labels.mech.d2",
        anchor=r"coarse verification scale, {v} the unit",
        expect="frontier:labels/repair_nd_d2_unit_scale",
        tol="exact",
    ),
    Claim(
        id="labels.mech.clean",
        anchor=r"scale, and {v} the generator's own resolution",
        expect="frontier:labels/repair_nd_clean_fine_quantised",
        tol="exact",
    ),
    Claim(
        id="labels.mech.bad_total",
        anchor=r"{v} labels of {~} are therefore wrong",
        expect="frontier:labels/repair_nd_bad_labels",
        tol="exact",
    ),
    Claim(
        id="labels.mech.corpus_n",
        anchor=r"{~} labels of {v} are therefore wrong",
        expect="frontier:labels/repair_corpus_nd_instances",
        tol="exact",
    ),
    Claim(
        id="labels.mech.corpus_pct",
        anchor=r"are therefore wrong, {v}\% of the corpus",
        expect="frontier:labels/repair_nd_bad_label_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="labels.mech.bad_train",
        anchor=r"of the corpus, {v} in training",
        expect="frontier:labels/repair_nd_bad_train",
        tol="exact",
    ),
    Claim(
        id="labels.mech.bad_val",
        anchor=r"in training, {v} in validation",
        expect="frontier:labels/repair_nd_bad_val",
        tol="exact",
    ),
    Claim(
        id="labels.mech.bad_test",
        anchor=r"in validation and {v} in test",
        expect="frontier:labels/repair_nd_bad_test",
        tol="exact",
    ),
    Claim(
        id="labels.repair.exact_n",
        anchor=r"For the {v} instances with $n\le10$ we solve exactly with the Held--Karp",
        expect="frontier:labels/repair_nd_exact_certified",
        tol="exact",
    ),
    Claim(
        id="labels.repair.hk_max_n",
        anchor=r"Above $n={v}$ the label becomes the float64 length",
        expect="frontier:labels/repair_hk_exact_max_n",
        tol="exact",
    ),
    Claim(
        id="labels.repair.certified_by_bound",
        anchor=r"upgraded to certified: {v} instances",
        expect="frontier:labels/repair_nd_tour_certified_optimal",
        tol="exact",
    ),
    Claim(
        id="labels.repair.d2_n",
        anchor=r"because all {v} of its labels are $\mathrm{nint}$ sums",
        expect="frontier:labels/repair_d2_instances",
        tol="exact",
    ),
    Claim(
        id="labels.repair.d2_signed",
        anchor=r"mean signed error $-{v}\%$ overall",
        expect="= -1 * {frontier:labels/repair_d2_label_signed_mean_pct}",
        tol=("dp", 3),
    ),
    Claim(
        id="labels.repair.d2_signed_g1000",
        anchor=r"$-{v}\%$ at $G=1000$ against",
        expect="= -1 * {frontier:labels/repair_d2_grid1000_signed_mean_pct}",
        tol=("dp", 3),
    ),
    Claim(
        id="labels.repair.d2_signed_g10000",
        anchor=r"against $-{v}\%$ at $G=10^4$",
        expect="= -1 * {frontier:labels/repair_d2_grid10000_signed_mean_pct}",
        tol=("dp", 3),
    ),
    Claim(
        id="labels.repair.d2_improvable",
        anchor=r"and {v} of the stored tours are beatable in float64",
        expect="frontier:labels/repair_d2_tours_improvable_in_float",
        tol="exact",
    ),
    Claim(
        id="labels.verify.n",
        anchor=r"Across the {v} bound checks that six independent sources supply",
        expect="frontier:labels/repair_verify_instances_checked",
        tol="exact",
    ),
    Claim(
        id="labels.verify.violations",
        anchor=r"there are {v} violations",
        expect="frontier:labels/repair_verify_total_violations",
        tol="exact",
    ),
    Claim(
        id="labels.effect.nd_label_mape",
        anchor=r"MAPE {v}\%, {~} defective labels low",
        expect="frontier:labels/repair_nd_test_label_mape_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="conclusion.repair.quarantined",
        anchor=r"The remaining {v} instances (0.173\%) disagree",
        expect="frontier:labels/repair_quarantined_total",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.raw_margin_as_published",
        anchor=r"cuts the margin at that rung from {v} to {~} percentage points",
        expect="= {frontier:tsplib_label_variants/as_published/crossing_margin_pp}",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.labels.interp_crossing_excluded",
        anchor=r"under the repair and to {v} under deletion",
        expect="frontier:tsplib_label_variants/linhp318_excluded/crossing_interp_k",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.labels.gart2_mape_excluded_new",
        anchor=r"Deleting the instance instead would move the crossing to {~} and the multiple to {v}",
        expect="frontier:tsplib_label_variants/linhp318_excluded/crossing_cost_x_gart2",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.labels.crossing_k_excluded",
        anchor=r"Deleting the instance instead would move the crossing to {v} and the multiple to {~}",
        expect="frontier:tsplib_label_variants/linhp318_excluded/crossing_ladder_k",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.crossing_k_repaired",
        anchor=r"leaves the raw bound's crossing budget at {v} and its cost multiple at {~}",
        expect="frontier:tsplib_label_variants/linhp318_repaired/crossing_ladder_k",
        tol="exact",
    ),
    Claim(
        id="frontier.labels.crossing_x_repaired",
        anchor=r"leaves the raw bound's crossing budget at {~} and its cost multiple at {v}",
        expect="frontier:tsplib_label_variants/linhp318_repaired/crossing_cost_x_gart2",
        tol=("dp", 2),
    ),
    Claim(
        id="results_nd.sdpe_501_1000",
        anchor=r"and {v}\% at $n\in[501,1000]$. Against the as-published",
        expect="bank:nd_by_size_501_1000_gart_2_0_sdpe_pct",
        tol=("dp", 2),
    ),    # =======================================================================
    # Baseline-roster cut of 2026-08-12.  The model-class controls (linear /
    # feed-forward, on the V3 block and on the production block) are out of the
    # manuscript, and the two GART 2.0 variants are reported as ablations rather
    # than as baselines.  Every number that survives that move under a new
    # sentence is registered here rather than absorbed into the backlog.
    # =======================================================================

    # =======================================================================
    # FIGURE CAPTIONS -- numbers that are a drawing decision rather than a
    # measurement.  The ascent budget the three error-distribution figures draw
    # the certified bound at is a constant in the plotting script, not a value
    # any artifact emits, so there is nothing for it to disagree with.  It is
    # registered anyway, in all three captions, because an unregistered numeral
    # in a caption is exactly how a figure and its caption drift apart: change
    # HK_BUDGET in the script and these three anchors are what fails.
    # =======================================================================
    Claim(
        id="figure.boxplot_nd.hk_budget",
        anchor=r"drawn at the ascent budget $k={v}$ at which Table",
        no_generator=(
            "Drawing decision, not a measurement: "
            "paper_reference/regenerate_boxplots.py::HK_BUDGET, the ascent "
            "budget the bound's box is computed at. Settle by having the "
            "figure script emit its constants to a small JSON the checker can "
            "read, the way the table builder emits paper_numbers.json."
        ),
    ),
    Claim(
        id="figure.boxplot_2d.hk_budget",
        anchor=r"drawn at the same ascent budget $k={v}$ as in Figures",
        no_generator=(
            "Same constant as figure.boxplot_nd.hk_budget: "
            "paper_reference/regenerate_boxplots.py::HK_BUDGET."
        ),
    ),
    Claim(
        id="figure.boxplot_tsplib.hk_budget",
        anchor=r"Section~\ref{sec:frontier} at ascent budget $k={v}$. This is the stratum",
        no_generator=(
            "Same constant as figure.boxplot_nd.hk_budget: "
            "paper_reference/regenerate_boxplots.py::HK_BUDGET."
        ),
    ),

    # -- Section 6: the certified bound on the non-Euclidean set --------------
    # Section 6 previously ended without scoring the comparator the rest of the
    # paper is judged against, on the one corpus where the bound needs no
    # embedding and the estimator does. These five entries are the answer, read
    # from the all-benchmark bank rather than from paper_numbers.json, which
    # carries no bound row for this corpus.
    Claim(
        id="application.bound.like_for_like_n",
        anchor=r"On the {v} of this set that both methods score",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/N",
        tol="exact",
        note="Instances scored by both GART 2.0 and the 1-tree bound; the bound "
             "additionally scores brg180 and si1032, GART 2.0 pla33810 and pla85900.",
    ),
    Claim(
        id="application.bound.gart_mape",
        anchor=r"both methods score, GART 2.0 obtains {v}\% MAPE against",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/GART_2.0_MAPE_pct",
    ),
    Claim(
        id="application.bound.vj_k100",
        anchor=r"MAPE against {v}\% for the raw certified bound at an ascent budget",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/vj_raw_MAPE_by_k/100",
    ),
    Claim(
        id="application.bound.polyak_k500",
        anchor=r"under the Volgenant--Jonker step and {v}\% at a budget of {~} under "
               r"the Polyak step",
        expect="allbench:cells/noneuc/like_for_like_vs_GART2/polyak_raw_MAPE_by_k/500",
        note="The sentence used to read as one ascent at two budgets. 1.27 is the "
             "Volgenant--Jonker bound and 0.51 the Polyak one; Volgenant--Jonker at "
             "500 is 1.14 and Polyak at 100 is 1.99, so the printed pair belonged to "
             "neither arm. The anchors now carry the step name.",
    ),
    Claim(
        id="application.bound.factor",
        anchor=r"at a budget of {~} under the Polyak step, a factor of {v}",
        expect="= {allbench:cells/noneuc/like_for_like_vs_GART2/GART_2.0_MAPE_pct}"
               " / {allbench:cells/noneuc/like_for_like_vs_GART2/polyak_raw_MAPE_by_k/500}",
        tol=("dp", 1),
        note="Stated as a ratio, so the ratio is what is checked.",
    ),

    # -- Conclusion: the three numbers that carry the paper's one finding -----

    # -- Section 5.2: what "corpus median" is a median of ---------------------
    Claim(
        id="frontier.tsplib.throughput_x",
        anchor=r"puts the same pair at {v} instead, because the bound is cheap",
        expect="costfront:corpus_median_definition/tsplib_published/x_gart2_throughput_k25",
        tol=("dp", 2),
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
    Claim(
        id="frontier.2d.gart_ms",
        anchor=r"GART 2.0 costs {v}~ms on the typical instance at",
        expect="costfront:cells/2d/groups/Total (all 2D)/gart2_ms",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.2d.gart_mape",
        anchor=r"on the typical instance at {v}\% MAPE. The raw certified bound",
        expect="costfront:cells/2d/groups/Total (all 2D)/gart2_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.2d.raw_crossing_k",
        anchor=r"Volgenant--Jonker step at a budget of {v}, at {~} times the cost",
        expect="costfront:cells/2d/groups/Total (all 2D)/crossover/vj_ckpt/raw/k",
        tol="exact",
    ),
    Claim(
        id="frontier.2d.raw_crossing_cost_x",
        anchor=r"at a budget of {~}, at {v} times the cost for {~}\% MAPE",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/x_gart2_typical",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.2d.raw_crossing_mape",
        anchor=r"times the cost for {v}\% MAPE and a paired win rate",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.2d.raw_crossing_win",
        anchor=r"and a paired win rate of {v}\%; the calibrated row",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/50/"
               "win_rate_vs_gart2_pct/raw",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.2d.cal_crossing_cost_x",
        anchor=r"a rung earlier, at {v} times the cost for {~}\%. The reversal",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/25/x_gart2_typical",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.2d.cal_crossing_mape",
        anchor=r"times the cost for {v}\%. The reversal is monotone",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/25/cal_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.2d.small_bucket_n",
        anchor=r"On the {v} instances of its smallest size bucket",
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
        anchor=r"MAPE against GART 2.0's {v}\% at {~} times the cost, and wins",
        expect="costfront:cells/2d/groups/n in [5,10]/gart2_MAPE_pct",
        note="The trailing ', and wins' keeps this anchor off the abstract's "
             "sentence about the multidimensional benchmark, which is otherwise "
             "word-for-word the same shape.",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.2d.small_bucket_cost_x",
        anchor=r"at {v} times the cost, and wins {~}\% of the paired comparisons",
        expect="costfront:cells/2d/groups/n in [5,10]/ascents/vj_ckpt/25/x_gart2_typical",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.2d.small_bucket_win",
        anchor=r"and wins {v}\% of the paired comparisons. On the",
        expect="costfront:cells/2d/groups/n in [5,10]/ascents/vj_ckpt/25/"
               "win_rate_vs_gart2_pct/raw",
        tol=("dp", 1),
    ),
    Claim(
        id="frontier.2d.large_bucket_n",
        anchor=r"On the {v} instances of its largest, nothing on the ladder",
        expect="costfront:cells/2d/groups/n in [501,1000]/N",
        tol="exact",
    ),
    Claim(
        id="frontier.2d.large_bucket_cost_x",
        anchor=r"only at the top budget, and there it costs {v} times as much",
        expect="costfront:cells/2d/groups/n in [501,1000]/ascents/vj_ckpt/500/x_gart2_typical",
        tol=("dp", 2),
        note="The largest bucket is where the reversal of Section 5.2 runs the other "
             "way: more accurate only at the top rung, and only at 26 times the cost.",
    ),
    Claim(
        id="frontier.2d.caption_n",
        anchor=r"cost/accuracy ladder, all {v} instances, both subgradient step rules",
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
        tol=("dp", 3),
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
        anchor=r"MAPE at {v} times that cost and wins all {~} paired comparisons",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "polyak_ckpt/500/x_gart2_typical",
        tol=("dp", 2),
        note="The accuracy factor Section 6 quotes, now priced: it is bought below "
             "the estimator's own cost, not above it.",
    ),
    Claim(
        id="frontier.noneuc.pk500_win_n",
        anchor=r"and wins all {v} paired comparisons; at a budget of",
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
        anchor=r"times GART 2.0 and reads {v}\% against {~}\%. The accuracy factor",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "vj_ckpt/25/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.noneuc.vj_crossing_gart_mape",
        anchor=r"and reads {~}\% against {v}\%. The accuracy factor",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/gart2_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.noneuc.caption_n",
        anchor=r"cost/accuracy ladder, the {v} instances both methods score",
        expect="costfront:cells/noneuc/instance_accounting/matched",
        tol="exact",
    ),

    # -- Section 5.4: the two step rules, scored against each other -----------
    Claim(
        id="frontier.steps.vj_lead_top_k",
        anchor=r"at every budget below {v} on the 2D corpus",
        no_generator=(
            "An ascent budget: the top rung of Table~\\ref{tab:frontier_2d}, above "
            "which the comparison is not run. Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.steps.vj_2d_mape",
        anchor=r"on the 2D corpus, {v}\% against {~}\% at a budget of",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/25/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.steps.pk_2d_mape",
        anchor=r"{~}\% against {v}\% at a budget of {~}, and Polyak leads it",
        expect="costfront:cells/2d/groups/Total (all 2D)/ascents/polyak_ckpt/25/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.steps.2d_k",
        anchor=r"at a budget of {v}, and Polyak leads it from a budget of",
        no_generator=(
            "An ascent budget: the rung at which the two step rules are compared on "
            "the 2D corpus. Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.steps.noneuc_k",
        anchor=r"and Polyak leads it from a budget of {v} on the non-Euclidean corpus",
        no_generator=(
            "An ascent budget: the rung from which the Polyak arm leads on the "
            "non-Euclidean corpus, read off Table~\\ref{tab:frontier_noneuc}. "
            "Nothing to settle."
        ),
    ),
    Claim(
        id="frontier.steps.pk_noneuc_mape",
        anchor=r"on the non-Euclidean corpus, {v}\% against {~}\%. The assignment",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "polyak_ckpt/200/raw_MAPE_pct",
        tol=("dp", 3),
    ),
    Claim(
        id="frontier.steps.vj_noneuc_mape",
        anchor=r"corpus, {~}\% against {v}\%. The assignment of Section",
        expect="costfront:cells/noneuc/groups/Total (all non-EUC_2D)/ascents/"
               "vj_ckpt/200/raw_MAPE_pct",
        tol=("dp", 3),
    ),

    # -- Section 5.4: the validity gates over both cost tables ----------------
    Claim(
        id="frontier.gates.pairs_2d",
        anchor=r"zero spread across all {v} instance--budget pairs of the 2D corpus",
        expect="costfront:cells/2d/gates/vj_ckpt/determinism/pairs",
        tol="exact",
    ),
    Claim(
        id="frontier.gates.pairs_noneuc",
        anchor=r"of the 2D corpus and all {v} of the non-Euclidean one",
        expect="costfront:cells/noneuc/gates/vj_ckpt/determinism/pairs",
        tol="exact",
    ),
    Claim(
        id="frontier.gates.series_2d",
        anchor=r"monotone in $k$, over all {v} series on the 2D corpus",
        expect="costfront:cells/2d/gates/vj_ckpt/monotone_in_k/series",
        tol="exact",
    ),
    Claim(
        id="frontier.gates.series_noneuc",
        anchor=r"series on the 2D corpus and all {v} on the non-Euclidean one",
        expect="costfront:cells/noneuc/gates/vj_ckpt/monotone_in_k/series",
        tol="exact",
    ),

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
        anchor=r"printed above is between {v} and {~} of what the same pair",
        expect="costfront:load_control/bias_in_a_cost_ratio/0",
        tol=("dp", 2),
    ),
    Claim(
        id="frontier.load.bias_hi",
        anchor=r"printed above is between {~} and {v} of what the same pair",
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
        anchor=r"whose {~} becomes {v}. Second, the checkpointed protocol",
        expect="= {costfront:cells/2d/groups/Total (all 2D)/ascents/vj_ckpt/200/"
               "x_gart2_typical} / {costfront:load_control/bias_in_a_cost_ratio/200}",
        tol=("dp", 2),
        note="Stated as the corrected multiple, so the correction is what is "
             "checked rather than a hand-computed constant.",
    ),
    Claim(
        id="frontier.amort.2d_lo",
        anchor=r"shares nothing between rungs costs {v} to {~} times the checkpointed",
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
]
