"""Verify the numbers the manuscript asserts in *prose* against generated artifacts.

``build_paper_tables.py --check`` covers the table bodies and nothing else.  This
is its sibling for the running text: same input file, same reporting shape.  Run
them together::

    python paper_tooling/build_paper_tables.py --check      # table cells
    python paper_tooling/check_prose_numbers.py             # prose numbers
    python paper_tooling/check_prose_numbers.py --tex COPY  # ...against a copy

What it does
------------
1.  Masks the regions of ``Area_Free_Main.tex`` that are not prose: the preamble,
    ``%`` comments, ``\\iffalse`` blocks, anything after ``\\end{document}``,
    verbatim-like bodies including ``\\begin{comment}``, tabular bodies, display
    math in every form (``equation``, ``\\[...\\]``, ``$$...$$``),
    reference-only command arguments and layout-parameter settings.  Every mask
    is named and its suppressed-numeral count is printed, so nothing disappears
    without a line in the report.  See ``mask_regions`` for why the set is what
    it is and what is deliberately left out.
2.  Resolves each ``prose_manifest.CLAIMS`` anchor to a character span **in the
    unmasked text** and pulls out the numeral it owns.  An anchor that resolves
    only inside a masked region binds to nothing: the number it names is not in
    the compiled PDF, so it is reported ANCHOR_MASKED rather than checked.
3.  Compares that numeral against the generated value -- a ``paper_numbers.json``
    key, a path into the frozen production sidecar, or an arithmetic expression
    over either -- against an acceptance radius derived from the *generated*
    value and the entry's declared tolerance.
4.  Independently scans the prose for numerals that *look like asserted results*
    and reports any that no manifest entry covers.

States
------
``MATCHED``       prose agrees with the generated value.
``MISMATCHED``    it does not; both values, the radius and the derivation are
                  printed.
``UNDER_PRECISE`` the prose numeral *is* the generated value, rounded honestly
                  at the precision the sentence prints -- but that precision is
                  too coarse for the claim's acceptance band, so the sentence
                  asserts nothing checkable.  "a factor of 3" for a generated
                  2.925833 is the shape of it.  Blocking, like MISMATCHED, so
                  dropping digits is not a way past the gate; the difference is
                  the message, which names the number of decimals to print and
                  the rendering to print.
``NO_GENERATOR``  registered as unbackable, with the reason and what would
                  settle it.  Not a failure, but it is listed every run.
``UNREGISTERED``  a numeral that reads as an asserted result and is in no
                  manifest entry.  This is the loud one: it is how the next
                  stale number gets in.  Split against ``prose_baseline.json``
                  into **NEW** (not in the recorded backlog -- always printed in
                  full, always blocking) and **BACKLOG** (recorded, printed but
                  not blocking, pageable with ``--limit``/``--offset``).
``ANCHOR_MISSING`` / ``ANCHOR_AMBIGUOUS`` / ``ANCHOR_MASKED``
                  a manifest entry stopped matching, matches in several live
                  places, or matches only in text the reader never sees.  All
                  loud -- a claim that silently stops being checked is exactly
                  the failure being defended against, so none of the three is a
                  skip.
``CLAIM WITHDRAWN`` a claim that was in the manifest when the baseline was
                  written is not in it now.  Deleting a claim used to be the
                  cheapest way to make a MISMATCHED number stop blocking: the
                  finding moved from ``mismatched`` into ``new`` and the total
                  did not move.  The baseline now records the roster, so the
                  removal is itself a finding, reported with the state the claim
                  was last known to be in.
``VACATED``       a recorded occurrence that no longer appears.  Not blocking --
                  deleting an unverified number is progress -- and, unlike the
                  count-based predecessor's ``STALE``, not a free slot either.

Exit code
---------
Non-zero on MISMATCHED, on UNDER_PRECISE, on any anchor fault, on a NEW
unregistered numeral, and on a claim withdrawn from the manifest.
The recorded backlog does **not** block: with ~550 unverified numbers pending a
prose rewrite the exit code was permanently 1, which is a gate nobody can use
and everybody eventually disables.  ``--strict`` restores all-or-nothing for the
day the backlog is empty.

``--update-baseline`` and ``--migrate-baseline`` exit with the gate code of the
state they leave behind, computed by reloading the record they just wrote.  A
baseline operation may record a finding; it may not be the thing that decides
whether the run passed.  See ``prose_baseline.py`` for the occurrence-addressed
record, the permanent ledger of everything ever absorbed, and why a vacated slot
can no longer swallow the next number that happens to share its value.

Tolerance
---------
Two questions, answered separately, because conflating them is what made the
previous rule unsatisfiable on some claims and vacuous on others.

*Accuracy* is one acceptance radius per claim, computed from the generated value
and the manifest's declared tolerance and compared against
``|printed - expected|``.  It is deliberately **not** derived from how many
decimals the sentence prints -- see the note above ``sig_dp`` for the
non-monotone verdicts that produced.  The default band is half a unit in the
last of ``SIG_FIGS`` significant figures, which is scale-free; the old
``max(0.005, 0.5%)`` was not, and was 103% of a p-value of 0.0048533 at one end
while rejecting every honest one-decimal rendering of 2.925833 at the other.

*Precision* is a separate pass over the numeral's own decimal count.  It never
moves the radius.  It splits a rejected numeral into wrong (``MISMATCHED``) and
too coarse (``UNDER_PRECISE``), and it drives the PRECISION POLICY section,
which lists every asserted result the manuscript prints to fewer decimals than
its claim can be checked at -- a typography finding about the paper, with the
rendering it should carry instead.

Asserted result vs incidental numeral
-------------------------------------
The classifier is deliberately biased towards over-reporting: a false
UNREGISTERED costs one manifest line, a false negative costs a wrong number in a
published paper.  **A numeral is an asserted result unless a rule can say
concretely why it is not.**  In order:

  0. it is not inside a masked region, and not an exponent or subscript (the
     one structural rejection);
  1. it is not matched by a declared ``INCIDENTAL_PATTERNS`` rule; and
  2. TIER 1 promotes early, before the single-digit rule below: a result unit
     or comparison marker (``\\%``, "percentage point(s)", bare "points",
     "pp", ``ms``, "seconds", "factor of", ``\\times``, "-fold", a ``p =``/
     ``p <`` construct), the low end of a written range ("8 to 15"), or two or
     more printed decimal places;
  3. a bare single-digit integer is rejected -- in this manuscript those are
     always notation (interval bounds, coefficients, ``$\\alpha=1$``);
  4. everything else is asserted.

Why step 4 is a default and not a test
--------------------------------------
It used to be a test: an integer or one-decimal value counted only if its
sentence contained a noun from a hand-maintained ``RESULT_NOUNS`` list, and
everything else fell into a silent "no-marker" bucket.  That bucket held 60
numerals, among them the Section 3 provenance counts, both close-pair
populations, the "55 TSPLIB pairs" and a stale table-cell count.  A noun list
cannot enumerate the ways a result can be phrased, so it was replaced by a
default that errs the other way.

The same reasoning retired five structural rules and four declared ones, each of
which fired **zero** times on the manuscript while licensing a blind spot:
``struct:year`` (any bare 1900-2099 value -- e.g. a stale tree count of 2031),
``struct:cross-reference``, ``struct:sci-notation-base``,
``struct:thousands-fragment``, ``struct:latex-length`` (whose "in" unit matched
the English word: it hid "standardized deviations under 1.6"), plus
``declared:log-base``, ``declared:norm-order``, ``declared:float-guard`` and two
alternatives of ``declared:version-tag``.  A suppression rule now has to name a
numeral in the current manuscript that it correctly suppresses.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prose_baseline  # noqa: E402
from model_registry import GART, PRODUCTION_SIDECAR, REPO_ROOT  # noqa: E402
from prose_manifest import CLAIMS, INCIDENTAL_PATTERNS, Claim  # noqa: E402

P_TEX = REPO_ROOT / "paper_reference" / "Area_Free_Main.tex"
P_BANK = REPO_ROOT / "paper_tooling" / "tables" / "paper_numbers.json"
P_BASELINE = REPO_ROOT / "paper_tooling" / "prose_baseline.json"
# The Held--Karp 1-tree study's consolidated key space, written by
# paper_tooling/frontier_manuscript_numbers.py.  It is a separate source rather
# than more paper_numbers.json keys because build_paper_tables.py owns that file
# and rewrites it wholesale; a frontier key added there would not survive the
# next regeneration.  Paths are SLASH separated, not dot separated: the bank's
# own keys contain dots ("GART_2.0_MAPE_pct") and bucket labels
# ("n in [51,150]"), so a dotted path would be ambiguous.
P_FRONTIER = REPO_ROOT / "paper_tooling" / "frontier_manuscript_bank.json"
P_ALLBENCH = REPO_ROOT / "paper_tooling" / "hk1tree_allbench_bank.json"
# The cost half of the same study: the 2D diverse and non-EUC_2D benchmarks,
# which hk1tree_allbench_bank.json scores for accuracy and prices for nothing.
# Written by paper_tooling/hk1tree_cost_analyze.py.  Separate for the same
# reason P_FRONTIER is: it has its own generator, and folding it into a file
# another script rewrites wholesale would lose it on the next regeneration.
P_COSTFRONT = REPO_ROOT / "paper_tooling" / "hk1tree_cost_frontier_bank.json"
# The size-stratified reading of the same ND sweep, written by
# paper_tooling/size_stratified_analyze.py.  Separate from P_FRONTIER for the
# reason that file is separate from paper_numbers.json: its generator owns it
# and rewrites it wholesale.  Keys are flat -- they already contain slashes as
# ordinary characters ("cell/d3/n600_1000/gart_ms"), so this resolves like
# ``bank:`` rather than walking a nested path.
P_SIZESTRAT = REPO_ROOT / "paper_tooling" / "size_stratified_bank.json"
# SHAP attribution of the released booster by 2D generator class, written by
# paper_tooling/shap_2d_by_class.py; backs the near-collinear "why" paragraph.
P_SHAP2D = REPO_ROOT / "paper_tooling" / "shap_2d_summary.json"


# -- Sources ----------------------------------------------------------------
class Sources:
    """The generated artifacts a claim may point at."""

    def __init__(self) -> None:
        self.bank: dict[str, object] = json.loads(P_BANK.read_text(encoding="utf-8"))
        self.sidecar: dict[str, object] = json.loads(
            PRODUCTION_SIDECAR.read_text(encoding="utf-8"))
        self.frontier: dict[str, object] = json.loads(
            P_FRONTIER.read_text(encoding="utf-8"))
        # The 1-tree bound scored on all four benchmarks, including the two
        # corpora paper_numbers.json never carried a bound row for.
        self.allbench: dict[str, object] = json.loads(
            P_ALLBENCH.read_text(encoding="utf-8"))
        # The cost column for those same two corpora. Absent on a fresh clone
        # until the timing session has been run, so a claim that points here
        # reports a missing path rather than crashing the whole check.
        self.costfront: dict[str, object] = (
            json.loads(P_COSTFRONT.read_text(encoding="utf-8"))
            if P_COSTFRONT.exists() else {})
        # Absent on a fresh clone until the matched timing session has been
        # run, so a claim pointing here reports a missing path and does not
        # crash the whole check.
        self.sizestrat: dict[str, object] = (
            json.loads(P_SIZESTRAT.read_text(encoding="utf-8"))
            if P_SIZESTRAT.exists() else {})
        self.shap2d: dict[str, object] = (
            json.loads(P_SHAP2D.read_text(encoding="utf-8"))
            if P_SHAP2D.exists() else {})

    def resolve(self, spec: str) -> float:
        """Look up one ``bank:``/``sidecar:``/``frontier:``/``allbench:``/
        ``costfront:`` spec.

        Raises ``KeyError`` when the path is absent.  ``sidecar:`` paths are dot
        separated; the other three are slash separated.
        """
        kind, _, key = spec.partition(":")
        if not _:
            kind, key = "bank", spec
        if kind == "bank":
            if key not in self.bank:
                raise KeyError(f"no such paper_numbers.json key: {key!r}")
            val = self.bank[key]
        elif kind == "sizestrat":
            if key not in self.sizestrat:
                raise KeyError(f"no such size_stratified_bank.json key: {key!r}")
            val = self.sizestrat[key]
        elif kind in ("sidecar", "frontier", "allbench", "costfront", "shap2d"):
            val = {"sidecar": self.sidecar, "frontier": self.frontier,
                   "allbench": self.allbench, "costfront": self.costfront,
                   "shap2d": self.shap2d}[kind]
            sep = "." if kind == "sidecar" else "/"
            for part in key.split(sep):
                if isinstance(val, list) and part.isdigit() and int(part) < len(val):
                    val = val[int(part)]
                    continue
                if not isinstance(val, dict) or part not in val:
                    raise KeyError(f"no such {kind} path: {key!r} (at {part!r})")
                val = val[part]
        else:
            raise KeyError(f"unknown source {kind!r} in {spec!r}")
        if val is None or (isinstance(val, str) and set(val) <= {"-"}):
            raise KeyError(f"{spec!r} is undefined ({val!r}) in the generated artifact")
        return float(val)


_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
                  ast.Load, ast.Name)
_ALLOWED_FUNCS = {"abs", "min", "max"}
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def evaluate(expect: str, src: Sources) -> tuple[float, str]:
    """Return ``(value, derivation)`` for one ``expect`` spec.

    A plain spec is a single lookup.  A leading ``=`` marks an arithmetic
    expression whose ``{...}`` placeholders are source specs; the returned
    derivation shows the substituted numbers so a mismatch report explains
    itself instead of asserting a constant.
    """
    if not expect.lstrip().startswith("="):
        return src.resolve(expect), expect

    body = expect.lstrip()[1:].strip()
    parts: list[str] = []

    def sub(m: re.Match[str]) -> str:
        v = src.resolve(m.group(1))
        parts.append(f"{m.group(1)}={v:.6g}")
        return repr(v)

    filled = _PLACEHOLDER.sub(sub, body)
    tree = ast.parse(filled, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax {type(node).__name__} in {expect!r}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCS:
            raise ValueError(f"disallowed name {node.id!r} in {expect!r}")
    env = {"abs": abs, "min": min, "max": max}
    value = float(eval(compile(tree, "<expect>", "eval"), {"__builtins__": {}}, env))
    return value, f"{body}  with  " + ", ".join(parts)


# -- Tolerance and precision ------------------------------------------------
#
# Two different questions, kept apart on purpose.  Conflating them is what made
# the previous rule simultaneously unsatisfiable on some claims and useless on
# others.
#
#   ACCURACY   Is the printed number the same number as the generated one?
#              Answered by ONE acceptance radius r, a function of the generated
#              value and the manifest's declared tolerance and of nothing else.
#              ``MATCHED`` iff ``|printed - expected| <= r``.
#
#   PRECISION  Does the sentence print enough digits for that answer to be worth
#              anything?  Answered separately, from the numeral's own decimal
#              count.  It never moves r.  It decides, for a numeral r rejects,
#              whether the number is WRONG (``MISMATCHED``) or merely too COARSE
#              to resolve (``UNDER_PRECISE``), and it drives the PRECISION
#              POLICY list -- a finding about the manuscript's typography, not
#              about its arithmetic.
#
# Why accuracy must stay typography-free
# --------------------------------------
# The design rejected twice already is "round the generated value to however
# many decimals the sentence prints, then compare".  ``build_paper_tables.py
# --check`` can key on printed precision because a table cell's precision is
# fixed by the generator and uniform down a column.  Prose precision is chosen
# sentence by sentence, so keying the *acceptance set* on it hands the author a
# silent 10-100x widener: drop a decimal and the band grows tenfold.  Worse, it
# is non-monotone -- against a generated 9.818014 it passed "10" (err 0.182)
# while failing "9.9" (err 0.082), and "10" and "10.0", the same quantity,
# disagreed.  ``compare`` therefore still never receives the numeral's decimal
# count: the acceptance set is the interval ``[expected-r, expected+r]`` for a
# constant r, so monotonicity and typography-independence are structural.
#
# Note what UNDER_PRECISE does and does not do.  It never *accepts* anything --
# it is blocking, exactly like MISMATCHED, so coarsening a numeral is not an
# escape route from the gate.  It only re-labels a failure whose cause is the
# number of digits rather than the digits themselves, and prints the rendering
# that would satisfy the claim.  Under the old rule "10" for 9.818014 was
# certified; here it is refused, with "print at least 1dp -- 9.8".
#
# Why the radius is 2 significant figures and not 0.5%
# ----------------------------------------------------
# The old default was ``max(0.005, 0.5% of |x|)``.  Both terms were wrong at the
# ends of the range this manuscript actually spans.
#
#   * The 0.5% term made honest renderings unsatisfiable in the [1,10) decade.
#     For a generated 2.925833 the band was +/-0.0146: the honest one-decimal
#     rendering "2.9" (err 0.026) FAILED, and so did every other rendering an
#     author would plausibly write.  A gate that stays red after the sentence is
#     corrected is a gate that gets switched off.
#   * The 0.005 floor was unconditional, so it swamped sub-unit quantities: a
#     p-value of 0.0048533 got a +/-0.005 band, i.e. 103% of the value, and
#     accepted anything in [0, 0.0098].  Every significance claim in the paper
#     is that size.
#
# Replacing both with "half a unit in the last of SIG_FIGS significant figures"
# fixes both ends with one rule, because significant figures are scale-free
# where a relative floor plus an absolute floor is not.  The resulting band is
# 0.5%-5% of the value in every decade at or below 1 (it was 0.5%-103%), and is
# clamped at half an integer unit above 10 -- prose does not round counts to
# trailing zeros, and 5% of 1{,}118 would have accepted "1{,}100".
#
# Calibration is unchanged where it was already right: against the generated
# 9.818014 the band is +/-0.05, so "9.8" (err 0.018) passes and "9.9" (err
# 0.082) -- the smallest error the tool has ever called wrong -- still fails.
SIG_FIGS = 3  # author, 2026-09-03: three significant figures everywhere

# Slack for binary-float representation only: a few double-precision ULPs at the
# magnitude in play.  It has to scale with the value -- 16{,}920 cannot be
# compared to an absolute 1e-12 -- but it must stay far below the smallest band
# it guards, or it becomes a second, hidden tolerance.  At 1e-12 the slack on
# 16{,}920 is 1.7e-8, against a five-decimal half-ulp of 5e-6; at the 1e-9 this
# started as it was 1.7e-5, three times that half-ulp, and the self-test's P3
# caught it certifying a numeral 2e-5 away as an honest 5dp rounding.
_EPS = 1e-12


def _slack(magnitude: float) -> float:
    return _EPS * max(1.0, abs(magnitude))


def _decade(x: float) -> int:
    """``floor(log10(x))`` for ``x > 0``, corrected for log10's edge error."""
    e = math.floor(math.log10(x))
    if 10.0 ** (e + 1) <= x:
        e += 1
    elif 10.0 ** e > x:
        e -= 1
    return e


def sig_dp(expected: float, sig: int = SIG_FIGS) -> int:
    """Decimal places needed to state ``expected`` to ``sig`` significant figures.

    Never negative: 2 s.f. of 1{,}118 is "1{,}100", but prose writes counts in
    full, so the floor for anything at or above 10 is a whole unit.
    """
    x = abs(expected)
    if not math.isfinite(x) or x == 0.0:
        return sig
    return max(0, sig - 1 - _decade(x))


def radius(expected: float, tol: object) -> tuple[float, str]:
    """Acceptance radius for one claim, plus a one-line derivation for the report.

    Depends only on ``expected`` and the manifest's declared tolerance -- never
    on the matched text.

    ``dp`` is a *tightening* knob, now against the significant-figure floor
    rather than against a relative cap: ``("dp", k)`` reads "this claim is
    asserted to k decimal places", and resolves to half a unit in the k-th
    decimal, or to the s.f. floor if k is coarser than the value can bear.  That
    second clause is what stops a declared ``("dp", 2)`` from handing a p-value
    of 0.0048533 a band wider than itself.  It is also why ``("dp", k)`` can no
    longer be unsatisfiable: the radius is always at least the half-ulp of the
    precision the claim is required to print at, so the honest rendering at that
    precision always lands inside it.  ``--selftest`` P5 sweeps exactly that.

    Widening past the floor is still possible only through ``abs``/``rel``,
    which are explicit, per-claim and greppable in ``prose_manifest.py``.
    """
    floor_dp = sig_dp(expected)
    if tol == "exact":
        return 0.0, "exact"
    if tol == "printed":
        return _half_ulp(floor_dp), (f"half-ulp at {floor_dp}dp "
                                     f"= {SIG_FIGS} s.f. of {expected:.6g}")
    mode, arg = tol  # type: ignore[misc]
    if mode == "dp":
        k = max(int(arg), floor_dp)
        how = f"half-ulp at {k}dp"
        if k != int(arg):
            how += (f" (declared {int(arg)}dp is coarser than the {SIG_FIGS} s.f. "
                    f"floor of {expected:.6g}, so the floor wins)")
        return _half_ulp(k), how
    if mode == "abs":
        return float(arg), f"declared abs {float(arg):g}"
    if mode == "rel":
        return float(arg) * abs(expected), f"declared rel {float(arg):g}"
    raise ValueError(f"unknown tolerance {tol!r}")


def _half_ulp(dp: int) -> float:
    return 0.5 * 10.0 ** -int(dp)


def required_dp(expected: float, tol: object) -> int:
    """Coarsest decimal precision at which this claim can be stated and checked.

    The smallest ``k`` whose half-ulp fits inside the acceptance radius, so that
    ``round(expected, k)`` is guaranteed to be MATCHED.  Derived from the radius
    rather than from the tolerance mode, so every mode -- including ``abs`` and
    ``rel`` -- gets a precision requirement consistent with the band it asked
    for.  ``exact`` has no radius to fit inside, so it needs however many
    decimals the value itself has.
    """
    r, _ = radius(expected, tol)
    if r <= 0.0:
        for k in range(0, 13):
            if round(expected, k) == expected:
                return k
        return 12
    for k in range(0, 13):
        if _half_ulp(k) <= r * (1.0 + 1e-12):
            return k
    return 12


def target(expected: float, tol: object) -> str:
    """The rendering the manuscript should carry: honest, and inside the band."""
    return f"{expected:.{required_dp(expected, tol)}f}"


def honest_at(printed: float, expected: float, ndec: int) -> bool:
    """Is ``printed`` the generated value correctly rounded to its own ``ndec``?

    The one place typography is read, and it is read only to classify a failure
    the acceptance radius has already declared.  A numeral that satisfies this
    is not a wrong number -- it is the right number at too coarse a precision --
    which is why UNDER_PRECISE can never launder a wrong value.
    """
    return abs(printed - expected) <= _half_ulp(ndec) + _slack(expected)


def verdict(printed: float, expected: float, tol: object, ndec: int) -> str:
    """The four accuracy states, decided on two independent questions.

    Q1, by ``compare`` alone: is the numeral inside the policy band?  ``ndec``
    cannot widen or narrow that set, so every property ``--selftest`` proves
    about the radius still holds unchanged.

    Q2, by ``honest_at``: is the numeral the generated value correctly rounded
    to the precision the sentence actually prints?

        band  honest  ->  MATCHED
        band  coarse  ->  OVER_PRECISE     prints more digits than it gets right
        wide  honest  ->  UNDER_PRECISE    right number, too coarse to check
        wide  coarse  ->  MISMATCHED

    OVER_PRECISE exists because the band is scale-free (half a unit in the last
    of ``SIG_FIGS`` significant figures) while the manuscript's house style
    prints percentages to two decimals.  For a generated 2.904163 the band is
    +/-0.05, so a band-only test certifies ``2.94\\%`` -- a sentence wrong in the
    digit it chose to print.  Requiring honesty at the printed precision closes
    that without reintroducing typography into acceptance: a numeral spelled
    with more digits is a *different assertion*, not the same one respelled, and
    ``10`` / ``10.0`` / ``10.00`` for a generated 9.818014 still agree (all three
    fail Q1, so Q2 only sorts them into UNDER_PRECISE vs MISMATCHED).
    """
    inside = compare(printed, expected, tol)[0]
    honest = honest_at(printed, expected, ndec)
    if inside:
        return "MATCHED" if honest else "OVER_PRECISE"
    return "UNDER_PRECISE" if honest else "MISMATCHED"


def selftest_tolerance(verbose: bool = True) -> int:
    """Sweep the tolerance rule for the properties it is supposed to have.

    Not an assertion that the rule is monotone -- a demonstration.  For each
    (generated value, declared tolerance) pair it enumerates candidate prose
    numerals on the 0..5-decimal grids around the value and checks:

    P1 MONOTONE   every MATCHED numeral is strictly closer to the generated
                  value than every MISMATCHED one.  The old rule failed this
                  16 times out of 40 pairs (it passed "10" for 9.818014 while
                  failing "9.9").
    P2 TYPOGRAPHY-FREE
                  the same quantity spelled at different precisions -- "10",
                  "10.0", "10.00" -- always gets the same *acceptance* verdict.
                  The old rule disagreed with itself on exactly that pair.  The
                  three-way state may legitimately differ between those two
                  spellings, because they are different assertions: "10" says
                  the value is in [9.5,10.5], which is true of 9.818014 and
                  merely coarse, while "10.0" says it is in [9.95,10.05], which
                  is false.  Both are refused; only the diagnosis differs.
    P3 NO LAUNDERING
                  every UNDER_PRECISE numeral is the generated value correctly
                  rounded at its own printed precision.  A wrong number can
                  therefore never reach that state, which is what makes it safe
                  for it to carry a gentler message than MISMATCHED.
    P4 COARSENING IS NOT AN ESCAPE
                  UNDER_PRECISE is never MATCHED and is always in the blocking
                  set, so dropping digits cannot turn a red gate green.
    P5 SATISFIABLE AT THE DECLARED PRECISION
                  the rendering the report asks for is MATCHED, *and* the
                  precision it asks for is the one the claim declares -- ``k``
                  for ``("dp", k)``, the s.f. floor for ``"printed"``, and the
                  floor rather than ``k`` when the declaration is coarser than
                  the value can bear.  Stated that way it is not self-consistent
                  with whatever radius happens to be in force, so it has teeth:
                  the old rule fails it on every value in the [1,10) decade,
                  including the ``("dp",1)`` claim whose band was +/-0.0146 and
                  which therefore no one-decimal numeral could satisfy.
    P6 BAND IS NOT VACUOUS
                  an inferred band is at most 5% of the value it guards, which
                  is what "correct to SIG_FIGS significant figures" means.  The
                  old rule's unconditional 0.005 floor failed this on every
                  sub-unit quantity -- 103% of a p-value of 0.0048533 -- which is
                  the size of every significance claim in the manuscript.
                  Declared ``abs``/``rel`` bands are exempt: going wider than the
                  floor is what they are for.

    Returns the number of violations; 0 is a pass.
    """
    gens = [0.0031, 0.0048533, 0.0075795, 0.0625, 0.086286, 0.62, 1.0, 1.05,
            2.904163, 2.925833, 3.33, 9.818014, 10.0, 16.5, 28.7, 38.05, 62.5,
            99.995, 148.0, 1118.0, 2031.0, 16920.0]
    tols: list[object] = ["printed", "exact", ("dp", 0), ("dp", 1), ("dp", 2),
                          ("dp", 3), ("dp", 4), ("abs", 0.5), ("rel", 0.02)]
    n_pairs = n_probe = v1 = v2 = v3 = v4 = v5 = v6 = 0
    notes: list[str] = []
    for gen in gens:
        for tol in tols:
            n_pairs += 1
            probes: list[tuple[float, bool, str]] = []
            for k in range(0, 6):
                step = 10.0 ** -k
                base = round(gen / step)
                for j in range(-60, 61):
                    v = (base + j) * step
                    if v < 0:
                        continue
                    ok, _, err, _ = compare(v, gen, tol)
                    probes.append((err, ok, f"{v:.{k}f}"))
                    # P3/P4: the three-way state, decided with this spelling's
                    # own decimal count.
                    state = verdict(v, gen, tol, k)
                    if state == "UNDER_PRECISE":
                        if abs(v - gen) > _half_ulp(k) * (1.0 + 1e-9):
                            v3 += 1
                            notes.append(
                                f"  P3 gen={gen:g} tol={tol}: {v:.{k}f} is "
                                f"UNDER_PRECISE but is not an honest {k}dp rounding")
                        if ok or state not in BLOCKING_STATES:
                            v4 += 1
                            notes.append(
                                f"  P4 gen={gen:g} tol={tol}: {v:.{k}f} is "
                                f"UNDER_PRECISE yet does not block")
            n_probe += len(probes)
            hit = [p for p in probes if p[1]]
            miss = [p for p in probes if not p[1]]
            if hit and miss:
                worst_hit = max(hit)
                best_miss = min(miss)
                if worst_hit[0] > best_miss[0] + 1e-12:
                    v1 += 1
                    notes.append(f"  P1 gen={gen:g} tol={tol}: MATCHED {worst_hit[2]!r} "
                                 f"(err {worst_hit[0]:.6g}) > MISMATCHED {best_miss[2]!r} "
                                 f"(err {best_miss[0]:.6g})")
            # P2: one value, many spellings.
            for probe in {p[2] for p in probes}:
                val = float(probe)
                verdicts = {compare(float(f"{val:.{k}f}"), gen, tol)[0]
                            for k in range(0, 6) if float(f"{val:.{k}f}") == val}
                if len(verdicts) > 1:
                    v2 += 1
                    notes.append(f"  P2 gen={gen:g} tol={tol}: {probe!r} verdict depends "
                                 f"on how many trailing zeros it is written with")
            # P5: the rendering the report tells the author to write must pass,
            # and the precision it asks for must be the declared one.
            need = required_dp(gen, tol)
            want = float(target(gen, tol))
            if verdict(want, gen, tol, need) != "MATCHED":
                v5 += 1
                notes.append(f"  P5 gen={gen:g} tol={tol}: the prescribed rendering "
                             f"{target(gen, tol)!r} at {need}dp is not MATCHED "
                             f"(r={radius(gen, tol)[0]:.6g})")
            inferred = tol == "printed" or (isinstance(tol, tuple) and tol[0] == "dp")
            if inferred:
                declared = (sig_dp(gen) if tol == "printed"
                            else max(int(tol[1]), sig_dp(gen)))  # type: ignore[index]
                at_declared = float(f"{gen:.{declared}f}")
                if not compare(at_declared, gen, tol)[0]:
                    v5 += 1
                    notes.append(
                        f"  P5 gen={gen:g} tol={tol}: the honest {declared}dp "
                        f"rendering {at_declared:.{declared}f} is REJECTED "
                        f"(r={radius(gen, tol)[0]:.6g} < half-ulp "
                        f"{_half_ulp(declared):.6g}) -- no correct rendering at the "
                        f"declared precision can pass")
                elif need != declared:
                    v5 += 1
                    notes.append(
                        f"  P5 gen={gen:g} tol={tol}: claim is stated to {declared}dp "
                        f"but the tool prescribes {need}dp")
                r_ = radius(gen, tol)[0]
                if gen != 0 and r_ > 0.05 * abs(gen) * (1.0 + 1e-9):
                    v6 += 1
                    notes.append(
                        f"  P6 gen={gen:g} tol={tol}: band +/-{r_:.6g} is "
                        f"{r_ / abs(gen):.1%} of the value")
    if verbose:
        print(f"\n{BAR}\ntolerance self-test: {n_pairs} (value, tolerance) pairs, "
              f"{n_probe} probe numerals\n{BAR}")
        for label, v in (("P1 monotone in the error", v1),
                         ("P2 acceptance independent of typography", v2),
                         ("P3 UNDER_PRECISE is always an honest rounding", v3),
                         ("P4 UNDER_PRECISE always blocks", v4),
                         ("P5 satisfiable at the declared precision", v5),
                         ("P6 inferred band is at most 5% of the value", v6)):
            print(f"  {label:<48}: {'PASS' if not v else f'FAIL ({v})'}")
        for note in notes[:20]:
            print(note)
        if len(notes) > 20:
            print(f"  ... {len(notes) - 20} more")
    return v1 + v2 + v3 + v4 + v5 + v6


def compare(printed: float, expected: float, tol: object) -> tuple[bool, str, float, float]:
    """Return ``(ok, rendering, error, r)``.  No parameter carries typography."""
    r, how = radius(expected, tol)
    err = abs(printed - expected)
    ok = err <= r + _slack(max(abs(expected), r))
    return ok, f"{expected:.5g} +/-{r:.2g}", err, r


# -- Region masking ---------------------------------------------------------
#
# The rule the mask enforces: **an anchor may only bind to text that reaches the
# compiled PDF, and only such text may yield an asserted numeral.**  A region
# that LaTeX drops is not prose, and certifying a number inside one is worse than
# not checking it -- the report says MATCHED about a sentence no reader will see.
#
# Two phases, because two different things are going on.
#
# PHASE 1 -- INERT regions.  Inside a ``%`` comment TeX never sees the tokens at
# all, and inside a verbatim-like body every character is catcode-12 text.  A
# ``\[`` or an ``\iffalse`` in either one is not a delimiter, so the phase-2 scan
# must not see it.  Phase 1 therefore *blanks* what it finds (spaces, same
# length, so every offset stays a valid index into the original) and phase 2 runs
# on the blanked copy.  Without this, one ``\iffalse`` parked in a comment would
# swallow the rest of the manuscript.
#
# PHASE 2 -- everything whose delimiters are live: the preamble, ``\iffalse``,
# text after ``\end{document}``, display math, table bodies, reference arguments.
INERT_ENVS = ("verbatim", "lstlisting", "filecontents", "comment")
MASKED_ENVS = ("tabular", "tabularx", "tabular*", "array", "equation", "align",
               "gather", "displaymath", "eqnarray", "thebibliography")
# Commands whose braced argument is a cross-reference or a length, never a result.
REF_COMMANDS = (r"cite\w*", "ref", "autoref", "eqref", "pageref", "label",
                "includegraphics", "input", "include", "bibliography",
                "bibliographystyle", "usepackage", "documentclass", "setlength",
                "vspace", "hspace", "url", "cmidrule", "addtolength", "geometry")
# Two-argument layout redefinitions -- ``\renewcommand{\arraystretch}{1.15}``.
# The first braced group names a LaTeX length or factor and the second sets it;
# neither is typeset.  REF_COMMANDS cannot express this, because its regex masks
# *one* braced group, which here is the name, leaving the value exposed: twelve
# ``\arraystretch`` settings were sitting in the recorded backlog as asserted
# results.  The parameter names are enumerated rather than accepting any
# ``\newcommand``, so a macro that really does define prose ("\newcommand{\ntest}
# {16{,}920}") stays visible to the scan instead of vanishing into the mask.
LAYOUT_PARAMS = ("arraystretch", "tabcolsep", "arrayrulewidth", "doublerulesep",
                 "baselinestretch", "extrarowheight")
# A TeX conditional token.  ``\iff`` is the math relation and ``\ifthenelse`` is
# the ifthen package's macro; neither opens a conditional and neither takes a
# ``\fi``, so counting them would push the depth past the real ``\fi``.
_COND_TOKEN = re.compile(r"\\(?:if[a-zA-Z@]*|else|fi)(?![a-zA-Z@])")
_NOT_CONDITIONAL = frozenset({r"\iff", r"\ifthenelse"})
_END_DOCUMENT = r"\end{document}"


def comment_spans(text: str) -> list[tuple[int, int, str]]:
    """``%`` to end of line.  A '%' starts a comment only when the run of
    backslashes before it is even (an odd run means it is the literal ``\\%``
    the results prose is full of)."""
    spans: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.split("\n"):
        for i, ch in enumerate(line):
            if ch != "%":
                continue
            back = len(line[:i]) - len(line[:i].rstrip("\\"))
            if back % 2 == 0:
                spans.append((pos + i, pos + len(line), "comment"))
                break
        pos += len(line) + 1
    return spans


def _blank(text: str, spans: list[tuple[int, int, str]]) -> str:
    """``text`` with every masked character replaced by a space, offsets intact."""
    buf = list(text)
    for a, b, _name in spans:
        for i in range(a, min(b, len(buf))):
            if buf[i] != "\n":       # keep line structure for the comment scan
                buf[i] = " "
    return "".join(buf)


def env_spans(text: str, envs: tuple[str, ...]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for env in envs:
        e = re.escape(env)
        for m in re.finditer(rf"\\begin\{{{e}\*?\}}.*?\\end\{{{e}\*?\}}", text, re.S):
            spans.append((m.start(), m.end(), f"env:{env}"))
    return spans


def _brace_end(text: str, pos: int) -> int | None:
    """Index of the ``}`` closing the group whose ``{`` ends at ``pos``."""
    depth, i = 1, pos
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def longtable_spans(text: str) -> list[tuple[int, int, str]]:
    r"""``longtable`` bodies, with the caption *argument* left live.

    A ``longtable`` carries its caption inside the environment, unlike the
    ``table`` float whose caption sits outside the ``tabular``.  Masking the
    environment whole would therefore hide caption prose the reader sees -- and
    the appendix tables converted to ``longtable`` state their scored instance
    counts there.  So the mask is laid down in pieces around each caption
    argument.  Everything else between the delimiters is header, rule or body:
    owned by ``build_paper_tables.py --check``, not by this scan.

    ``longtable`` cannot join ``MASKED_ENVS`` for the same reason: that helper
    masks a whole environment and has no notion of a live hole inside one.
    """
    spans: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\\begin\{longtable\*?\}.*?\\end\{longtable\*?\}",
                         text, re.S):
        cursor = m.start()
        for cap in re.finditer(r"\\caption\s*(?:\[[^\]]*\])?\s*\{", m.group(0)):
            lo = m.start() + cap.end()
            hi = _brace_end(text, lo)
            if hi is None or hi > m.end():
                continue
            spans.append((cursor, lo, "env:longtable"))
            cursor = hi
        spans.append((cursor, m.end(), "env:longtable"))
    return spans


def iffalse_spans(text: str) -> list[tuple[int, int, str]]:
    r"""``\iffalse`` regions, with TeX's own skipping semantics.

    While TeX skips a false branch it still *counts* conditionals, so the ``\fi``
    that closes an ``\iffalse`` is the one that balances every ``\if...`` opened
    inside it -- a ``\fi`` belonging to a real nested conditional does not end the
    region.  An ``\else`` seen at depth 1 does end it: the else-branch is typeset
    and is live prose.  An ``\else`` deeper than that belongs to the nested
    conditional and is ignored, exactly as TeX ignores it.

    A nested ``\iffalse`` needs no special case: it increments the depth like any
    other conditional, and its own span is contained in the outer one, so
    ``build_mask`` keeps the outer name.

    An **unclosed** ``\iffalse`` is masked to end of file and named
    ``iffalse:unclosed``, so the report says which it was.  Such a document does
    not compile at all, so nothing after the token is in any PDF; and the two
    failure directions are not symmetric.  Over-masking is loud -- anchors turn
    into blocking ANCHOR_MASKED faults and banked numerals turn into printed
    STALE entries.  Under-masking is silent, and silence is the defect this whole
    mechanism exists to remove.
    """
    spans: list[tuple[int, int, str]] = []
    end_of_last = 0
    for opener in re.finditer(r"\\iffalse(?![a-zA-Z@])", text):
        if opener.start() < end_of_last:
            continue                      # nested inside a region already taken
        depth, stop, name = 1, len(text), "iffalse:unclosed"
        for tok in _COND_TOKEN.finditer(text, opener.end()):
            t = tok.group(0)
            if t == r"\fi":
                depth -= 1
                if depth == 0:
                    stop, name = tok.end(), "iffalse"
                    break
            elif t == r"\else":
                if depth == 1:
                    stop, name = tok.start(), "iffalse"
                    break
            elif t not in _NOT_CONDITIONAL:
                depth += 1
        spans.append((opener.start(), stop, name))
        end_of_last = stop
    return spans


def mask_regions(text: str) -> list[tuple[int, int, str]]:
    """Named character spans that are not prose.

    Spans are returned outermost-suppressor-first, because ``build_mask`` lets
    the first span to claim a character name it: a table inside an ``\\iffalse``
    should report as ``iffalse``, not as ``env:tabular``.
    """
    # -- phase 1: regions in which LaTeX delimiters are not delimiters ---------
    inert = comment_spans(text)
    inert += env_spans(_blank(text, inert), INERT_ENVS)
    scan_text = _blank(text, inert)

    # -- phase 2: everything else, read off the blanked copy ------------------
    spans: list[tuple[int, int, str]] = []
    doc = scan_text.find(r"\begin{document}")
    if doc > 0:
        spans.append((0, doc, "preamble"))
    spans += [s for s in inert if s[2] == "comment"]
    # Anything after \end{document} is discarded by LaTeX -- the classic parking
    # spot for a cut paragraph, and every numeral in it is invisible to a reader.
    end = scan_text.find(_END_DOCUMENT)
    if end >= 0 and end + len(_END_DOCUMENT) < len(text):
        spans.append((end + len(_END_DOCUMENT), len(text), "after-end-document"))
    spans += iffalse_spans(scan_text)
    spans += [s for s in inert if s[2] != "comment"]
    # Display math in its two unenvironmented forms.  ``\begin{equation}`` and
    # friends are covered by MASKED_ENVS below; these two are not, and the
    # ``\[...\]`` at the SDPE definition was leaking its leading factor of 100
    # into the prose scan, where it was classified as an asserted result.  The
    # ``(?<!\\)`` guard keeps ``\\[2ex]`` -- a row break with a vertical skip,
    # not an open-display -- from opening a region that would run to the next
    # genuine ``\]``.
    for m in re.finditer(r"(?<!\\)\\\[.*?(?<!\\)\\\]", scan_text, re.S):
        spans.append((m.start(), m.end(), "display-math"))
    for m in re.finditer(r"(?<!\\)\$\$.*?(?<!\\)\$\$", scan_text, re.S):
        spans.append((m.start(), m.end(), "display-math"))
    spans += env_spans(scan_text, MASKED_ENVS)
    spans += longtable_spans(scan_text)
    for m in re.finditer(
            rf"\\(?:re|provide)?newcommand\s*\{{\s*\\(?:{'|'.join(LAYOUT_PARAMS)})\s*\}}"
            rf"\s*\{{[^{{}}]*\}}", scan_text):
        spans.append((m.start(), m.end(), "layout-param"))
    for m in re.finditer(rf"\\(?:{'|'.join(REF_COMMANDS)})\s*(?:\[[^\]]*\])?\s*\{{[^{{}}]*\}}",
                         scan_text):
        spans.append((m.start(), m.end(), "xref-arg"))
    return spans


def build_mask(text: str, spans: list[tuple[int, int, str]]) -> list[str | None]:
    mask: list[str | None] = [None] * len(text)
    for a, b, name in spans:
        for i in range(a, min(b, len(text))):
            if mask[i] is None:
                mask[i] = name
    return mask


# Masked regions an *anchor* may legally straddle.  ``xref-arg`` is masked so
# that the digits inside ``\cite{akiba2019optuna}`` or ``\ref{tab:2d}`` are not
# scanned as results -- but the command itself is typeset, so a sentence that
# contains one is still prose the reader sees, and an anchor is allowed to span
# it.  Every other mask marks text that is either absent from the PDF (comment,
# preamble, iffalse, after-end-document, env:comment, verbatim, layout-param) or
# owned by ``build_paper_tables.py --check`` (tabular, longtable, display math),
# and an anchor that touches one is not checking the running text.  A
# ``longtable`` caption is the exception the mask itself carves out, so an anchor
# there binds to live text and needs no entry here.  New mask names are
# opaque by default: a region is added to this set only with a reason for why the
# text around it still reaches the reader.
ANCHOR_PERMEABLE = frozenset({"xref-arg"})


def masks_over(mask: list[str | None], a: int, b: int) -> list[str]:
    """Distinct mask names covering ``text[a:b]``, in first-seen order."""
    out: list[str] = []
    for name in mask[a:b]:
        if name is not None and name not in out:
            out.append(name)
    return out


# -- Numeral scan and classification ----------------------------------------
# A thousands group may be written either the manuscript's way (``16{,}920``, the
# LaTeX idiom that keeps the comma from being typeset as punctuation) or plainly
# (``16,920``).  Both must tokenise as ONE numeral: with only the ``{,}`` form
# recognised, a registered claim anchored on "1,118" captured "1" and compared it
# against 1118.  Grouped alternatives are tried first so they win over the
# bare-integer branch.
#
# The plain form needs a guard the ``{,}`` form does not.  Every plain
# "digit,3-digit" string in this manuscript is in fact interval notation --
# ``$[51,100]$``, ``$n\in(200,1000]$``, the size-bucket column heads -- and
# reading those as 51100 and 2001000 would invent numerals while hiding the real
# bounds.  So the plain form is refused when a bracket sits on either side.  The
# residual cost is that a genuine thousands value wrapped in parentheses, "(1,118)",
# still splits; the manuscript writes real thousands as ``{,}``, and splitting is
# the pre-existing behaviour rather than a new one.
NUM = (r"(?:\d{1,3}(?:\{,\}\d{3})+"                 # 16{,}920 -- always thousands
       r"|(?<![\[(])\d{1,3}(?:,\d{3})+(?![\])])"    # 16,920 -- unless it is [a,b]
       r"|\d+)(?:\.\d+)?")
NUM_TOKEN = re.compile(rf"(?<![A-Za-z0-9_.]){NUM}(?![A-Za-z0-9])")

UNIT_AFTER = re.compile(
    r"^\s*(?:\\,)?(?:\\%|%|\\?~?\s*(?:ms\b|milliseconds?\b|seconds?\b|s\b)"
    r"|\s*(?:percentage\s+points?|percent\b|pp\b)|\s*(?:\\times|×)|-fold\b"
    # Bare "points" is the manuscript's shorthand for percentage points --
    # "the 2D gain of 3.4 points", "a 7-point claim".  Both are margins between
    # two printed percentages, i.e. the derived-margin class this tool exists
    # for, and both were invisible while only the spelled-out form counted.
    r"|-points?\b|\s+points?\b"
    r"|\s*times\s+(?:lower|higher|faster|slower|larger|smaller|more|less)\b)")
FACTOR_BEFORE = re.compile(
    r"(?:factor\s+of|by\s+a\s+factor\s+of|ratio\s+of|p\s*[=<>]\s*|speedup\s+of)\s*$")
# The low end of a stated range: "sat 8 to 15 percentage points below",
# "ranges from 2 to 61".  Without this the high end is checked and the low end
# is not, which is how half a range goes stale.  The dash form ("$n=5$--30") is
# deliberately not matched: there the low end sits inside math mode.
RANGE_AFTER = re.compile(r"^\s+to\s+\d")
# A count of a named class: "4 CEIL\_2D, 2 ATT, 10 GEO, and 17 EXPLICIT".  The
# manuscript spells small counts as words everywhere except when tabulating
# corpus composition, so these two digits are counts, not notation, and they
# are as liable to go stale as the 10 and 17 beside them.
LABEL_AFTER = re.compile(r"^\s+[A-Z][A-Z0-9\\_]{2,}")


@dataclass(frozen=True)
class Numeral:
    start: int
    end: int
    raw: str
    value: float
    ndec: int


def parse_numeral(raw: str) -> tuple[float, int]:
    clean = raw.replace("{,}", "").replace(",", "")
    return float(clean), len(clean.partition(".")[2])




def classify(text: str, num: Numeral, incidental: list[tuple[str, re.Pattern[str]]]
             ) -> tuple[bool, str]:
    """Return ``(is_asserted, rule)``.  ``rule`` names why, for the report."""
    before = text[max(0, num.start - 40):num.start]
    after = text[num.end:num.end + 40]

    # 0. structural rejection.
    #
    # One rule, because one rule is all the manuscript justifies: the "2" of
    # $O(n^2)$, the "2" of $R^2_\alpha$, the "3" of $O(n^3)$.  Five further
    # structural rules used to sit here and all five have been removed; see the
    # module docstring for which, and why a rule that fires zero times is a
    # liability rather than spare capacity.
    if before.endswith(("^", "^{", "_", "_{")):
        return False, "struct:exponent-or-subscript"

    # 1. declared incidental classes
    ctx = text[max(0, num.start - 24):num.end + 24]
    for name, pat in incidental:
        for m in pat.finditer(ctx):
            lo = max(0, num.start - 24) + m.start()
            if lo <= num.start < lo + len(m.group(0)):
                return False, f"declared:{name}"

    # 2. tier 1 -- carries a result unit, a range, a comparison marker, or >=2
    #    decimals.  These run before the single-digit rule below, so a genuine
    #    one-digit result ("a 7-point claim", "8 to 15 percentage points") is
    #    promoted rather than swallowed.
    if UNIT_AFTER.match(after):
        return True, "tier1:unit"
    if RANGE_AFTER.match(after):
        return True, "tier1:range"
    if LABEL_AFTER.match(after):
        return True, "tier1:label-count"
    if FACTOR_BEFORE.search(before):
        return True, "tier1:comparison"
    if num.ndec >= 2:
        return True, "tier1:precision"

    # 3. bare single-digit integers.
    #
    # In this manuscript a one-digit count in prose is written as a word ("five
    # classical estimators", "Ten describe the..."), so every bare 0-9 that
    # survives to here is notation: an interval bound ($\alpha\in[1,2]$), a
    # coefficient ($2\,L_{\mathrm{MST}}$), or a fixed value ($\alpha=1$).
    # Without this the bound and coefficient digits alone contributed ~90 false
    # positives and buried the real ones.
    if num.ndec == 0 and num.value < 10:
        return False, "tier2:bare-single-digit"

    # 4. everything else is asserted.
    #
    # This used to require a noun from a hand-maintained RESULT_NOUNS list in
    # the enclosing sentence, and anything else was dropped as "no-marker".
    # That bucket held 60 numerals and it was not incidental: it contained the
    # provenance counts of Section 3, both close-pair populations (74{,}759 and
    # 1{,}662{,}781), the "55 TSPLIB pairs", the "3.4 points" 2D margin and the
    # stale "1{,}528 table cells".  A numeral is a result unless a rule above
    # can say concretely why it is not; a list of nouns cannot.
    return True, "tier2:unmarked"


# -- Anchors ----------------------------------------------------------------
_HOLE = re.compile(r"\{v\}|\{~\}")


def compile_anchor(anchor: str) -> re.Pattern[str]:
    """Literal LaTeX with ``{v}``/``{~}`` holes -> regex; whitespace is elastic."""
    out: list[str] = []
    pos = 0
    for m in _HOLE.finditer(anchor):
        lit = re.escape(anchor[pos:m.start()])
        out.append(re.sub(r"(?:\\[ \t\n])+|\s+", r"\\s+", lit))
        out.append(f"({NUM})" if m.group(0) == "{v}" else f"(?:{NUM})")
        pos = m.end()
    lit = re.escape(anchor[pos:])
    out.append(re.sub(r"(?:\\[ \t\n])+|\s+", r"\\s+", lit))
    return re.compile("".join(out))


# -- Checking ---------------------------------------------------------------
@dataclass
class Result:
    claim: Claim
    state: str
    printed: float | None = None
    expected: float | None = None
    rendering: str = ""
    derivation: str = ""
    detail: str = ""
    span: tuple[int, int] | None = None
    shadow: str = ""
    ndec: int | None = None        # decimals the manuscript actually prints
    need_dp: int | None = None     # decimals it has to print to be checkable
    want: str = ""                 # the rendering that would satisfy the claim


def _anchor_fault(mask: list[str | None], m: re.Match[str]) -> str:
    """Why this anchor match is not live prose, or ``""`` if it is.

    Two independent conditions, because they fail differently.  The numeral may
    not sit in *any* masked region: a claim's value has to be a number the
    reader sees, and a digit inside ``\\ref{}`` is not one.  The surrounding
    match may not sit in a masked region either, except the permeable ones --
    otherwise the anchor is quoting a comment, the preamble or a table body, and
    the sentence it purports to check is not the sentence being typeset.
    """
    over_value = masks_over(mask, m.start(1), m.end(1))
    if over_value:
        return f"value lies in {'+'.join(over_value)}"
    opaque = [n for n in masks_over(mask, m.start(), m.end())
              if n not in ANCHOR_PERMEABLE]
    if opaque:
        return f"anchor spans {'+'.join(opaque)}"
    return ""


def check_claims(text: str, src: Sources, mask: list[str | None]) -> list[Result]:
    seen: set[str] = set()
    results: list[Result] = []
    for claim in CLAIMS:
        if claim.id in seen:
            raise SystemExit(f"duplicate manifest id {claim.id!r}")
        seen.add(claim.id)
        hits = list(compile_anchor(claim.anchor).finditer(text))
        if not hits:
            results.append(Result(claim, "ANCHOR_MISSING",
                                  detail="anchor text is no longer in the manuscript"))
            continue
        # An anchor may only bind to text that reaches the PDF.  Without this a
        # claim could resolve against a commented-out draft of its own sentence
        # and be reported MATCHED while the typeset line said something else --
        # the tool certifying a number no reader will ever see.
        live, dead = [], []
        for h in hits:
            fault = _anchor_fault(mask, h)
            (dead if fault else live).append((h, fault))
        if not live:
            where = ", ".join(f"line {line_of(text, h.start())} ({f})" for h, f in dead)
            results.append(Result(
                claim, "ANCHOR_MASKED",
                detail=f"{len(dead)} match(es), none in prose that reaches the PDF: {where}"))
            continue
        if len(live) > 1:
            lines = [line_of(text, h.start()) for h, _ in live]
            results.append(Result(claim, "ANCHOR_AMBIGUOUS",
                                  detail=f"{len(live)} live matches, lines {lines}"))
            continue
        m = live[0][0]
        # A live match plus a masked twin is not an error -- the live one is
        # authoritative -- but the twin is reported, because a commented-out
        # copy of a results sentence is exactly how a stale number gets restored.
        shadow = ("; ".join(f"line {line_of(text, h.start())} ({f})" for h, f in dead)
                  if dead else "")
        printed, ndec = parse_numeral(m.group(1))
        span = (m.start(1), m.end(1))
        if claim.no_generator:
            results.append(Result(claim, "NO_GENERATOR", printed=printed, ndec=ndec,
                                  detail=claim.no_generator, span=span, shadow=shadow))
            continue
        try:
            raw, derivation = evaluate(claim.expect, src)  # type: ignore[arg-type]
        except (KeyError, ValueError) as exc:
            results.append(Result(claim, "MISMATCHED", printed=printed, ndec=ndec,
                                  detail=f"source error: {exc}", span=span, shadow=shadow))
            continue
        expected = raw * claim.scale
        ok, rendering, err, r = compare(printed, expected, claim.tol)
        _, how = radius(expected, claim.tol)
        need = required_dp(expected, claim.tol)
        want = target(expected, claim.tol)
        state = verdict(printed, expected, claim.tol, ndec)
        if state == "MATCHED":
            detail = ""
        elif state == "UNDER_PRECISE":
            detail = (f"{printed:g} is {expected:.6g} rounded honestly to {ndec}dp, so "
                      f"the sentence is not wrong -- it is too coarse to check: "
                      f"{ndec}dp spans +/-{_half_ulp(ndec):g} against a band of "
                      f"{r:.6g} = {how}. Print at least {need}dp: {want}")
        elif state == "OVER_PRECISE":
            detail = (f"{printed:g} is inside the band ({r:.6g} = {how}) but is not "
                      f"{expected:.6g} rounded to the {ndec}dp the sentence prints: "
                      f"off by {err:.6g} against a half-ulp of {_half_ulp(ndec):g}. "
                      f"The digit it chose to print is wrong. Write {want} "
                      f"(or {expected:.{ndec}f} to keep {ndec}dp)")
        else:
            detail = (f"off by {err:.6g}; tolerance {r:.6g} = {how}; "
                      f"correct rendering at {need}dp is {want}")
        results.append(Result(claim, state,
                              printed=printed, expected=expected, rendering=rendering,
                              derivation=derivation, span=span, shadow=shadow,
                              ndec=ndec, need_dp=need, want=want, detail=detail))
    return results


def scan(text: str, mask: list[str | None]
         ) -> tuple[list[tuple[Numeral, str]], list[tuple[Numeral, str]]]:
    """Split every prose numeral into (asserted, incidental), each tagged by rule."""
    incidental = [(n, re.compile(p)) for n, p in INCIDENTAL_PATTERNS]
    asserted: list[tuple[Numeral, str]] = []
    rejected: list[tuple[Numeral, str]] = []
    for m in NUM_TOKEN.finditer(text):
        value, ndec = parse_numeral(m.group(0))
        num = Numeral(m.start(), m.end(), m.group(0), value, ndec)
        region = mask[m.start()]
        if region is not None:
            rejected.append((num, f"masked:{region}"))
            continue
        is_asserted, rule = classify(text, num, incidental)
        (asserted if is_asserted else rejected).append((num, rule))
    return asserted, rejected


# -- Reporting --------------------------------------------------------------
BAR = "=" * 82


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def excerpt(text: str, num: Numeral, width: int = 46) -> str:
    lo, hi = max(0, num.start - width), min(len(text), num.end + width)
    s = re.sub(r"\s+", " ", text[lo:hi])
    return ("..." if lo else "") + s + ("..." if hi < len(text) else "")


ANCHOR_FAULTS = ("ANCHOR_MISSING", "ANCHOR_AMBIGUOUS", "ANCHOR_MASKED")
FAULT_BLURB = {
    "ANCHOR_MISSING": "anchor text is gone (claim is unchecked)",
    "ANCHOR_AMBIGUOUS": "anchor resolves in several places (claim is unchecked)",
    "ANCHOR_MASKED": "anchor resolves only outside the typeset prose, so the "
                     "number is not in the PDF (claim is unchecked)",
}
# Every state that must turn the gate red.  UNDER_PRECISE is in it deliberately:
# it is a softer diagnosis, not a softer outcome, or dropping a decimal would be
# a way out of the gate.  OVER_PRECISE likewise: printing a digit you did not
# get right is a wrong sentence, however close the value is to the band.
BLOCKING_STATES = frozenset({"MISMATCHED", "UNDER_PRECISE", "OVER_PRECISE",
                             *ANCHOR_FAULTS})


def unregistered_of(results: list[Result], asserted: list[tuple[Numeral, str]]
                    ) -> list[tuple[Numeral, str]]:
    """Asserted numerals that no manifest entry's ``{v}`` span covers."""
    covered: list[tuple[int, int]] = [r.span for r in results if r.span]
    return [(n, rule) for n, rule in asserted
            if not any(a <= n.start and n.end <= b for a, b in covered)]


# Sections that --limit/--offset may truncate.  Structural rather than
# conventional: every pageable section calls _page, _page refuses any name that
# is not listed here, and no blocking section is listed.  A blocking finding
# that can be paged out of the report is a blocking finding nobody reads, which
# is how the backlog swallowed a proven-wrong number in the first place.
PAGEABLE_SECTIONS = frozenset({"backlog", "vacated"})


def _page(section: str, rows: list, limit: int, offset: int) -> tuple[list, int, int]:
    """``(page, skipped_before, remaining_after)`` for one pageable section."""
    if section not in PAGEABLE_SECTIONS:
        raise AssertionError(
            f"refusing to paginate {section!r}: only {sorted(PAGEABLE_SECTIONS)} "
            f"may be truncated. Blocking findings are printed in full.")
    lo = max(0, offset)
    hi = len(rows) if limit <= 0 else lo + limit
    page = rows[lo:hi]
    return page, lo, len(rows) - (lo + len(page))


def keyed(text: str, unregistered: list[tuple[Numeral, str]]
          ) -> list[tuple[Numeral, str, str, str]]:
    """Attach the baseline entry key and a context example to each numeral.

    The key is occurrence-addressed: section, value, classifier rule *and* a
    digest of the normalized prose around the numeral.  Without the digest the
    record grants fungible slots, and a slot vacated by one edit silently
    absorbs the next number that happens to share its value -- see the schema-2
    note in ``prose_baseline.py``.
    """
    index = prose_baseline.section_index(text)
    norm = prose_baseline.normalize(text)
    return [(n, rule, prose_baseline.entry_key(
                prose_baseline.section_at(index, n.start), n.raw, rule,
                prose_baseline.context_digest(norm, n.start)),
             excerpt(text, n))
            for n, rule in unregistered]


def report(text: str, results: list[Result], asserted: list[tuple[Numeral, str]],
           rejected: list[tuple[Numeral, str]], limit: int, show_incidental: bool,
           tex_path: Path, part: "prose_baseline.Partition",
           base: "prose_baseline.Baseline", cen: "prose_baseline.Census",
           offset: int = 0, strict: bool = False) -> int:
    unregistered = unregistered_of(results, asserted)

    by_state: dict[str, list[Result]] = {}
    for r in results:
        by_state.setdefault(r.state, []).append(r)

    print(f"\n{BAR}\n--check-prose: manuscript prose vs generated artifacts "
          f"({len(CLAIMS)} manifest entries, production model {GART})\n"
          f"manuscript: {tex_path}\n{BAR}")
    print(f"matched: {len(by_state.get('MATCHED', []))} "
          f"| mismatched: {len(by_state.get('MISMATCHED', []))} "
          f"| under-precise: {len(by_state.get('UNDER_PRECISE', []))} "
          f"| over-precise: {len(by_state.get('OVER_PRECISE', []))} "
          f"| no generator: {len(by_state.get('NO_GENERATOR', []))} "
          f"| anchor faults: {sum(len(by_state.get(s, [])) for s in ANCHOR_FAULTS)} "
          f"| unregistered: {len(unregistered)} "
          f"({len(part.new)} NEW, {len(part.backlog)} recorded)")
    for line in prose_baseline.banner(base, GART):
        print(line)
    if strict:
        print("--strict: the recorded backlog blocks as well.")
    print()

    # Blocking findings first, and none of them is pageable. --limit and
    # --offset reach the recorded backlog and the vacated list only; see _page.
    if cen.withdrawn:
        print(f"CLAIM WITHDRAWN -- in the manifest when {base.path.name} was written, "
              f"gone now ({len(cen.withdrawn)}):")
        for cid, d in cen.withdrawn:
            state = str(d.get("state", "?"))
            mark = "!!" if state == "MISMATCHED" else "  "
            values = ""
            if d.get("printed") is not None:
                values = f", paper {d.get('printed')} vs generated {d.get('expected')}"
            print(f"  {mark} {cid:<38}last state {state}{values}")
            if d.get("expect"):
                print(f"     {'':<38}was checked against {d['expect']}")
        print("  a withdrawn claim releases its numeral back into the unregistered\n"
              "  pool. Restore it, or acknowledge the withdrawal with\n"
              "  --update-baseline --reason \"...\" -- which names it in the ledger "
              "permanently.\n")

    # Never truncated. The whole point of the record is that this section is
    # short enough to read every time, so a number nothing verifies cannot enter
    # the manuscript by hiding in a 550-row tail.
    if base.legacy:
        # A schema-1 record grants no coverage, so every unregistered numeral is
        # formally NEW. Printing 600 of them would bury the MISMATCHED section
        # below; the actionable finding is the record's format, not the list.
        print(f"MIGRATION REQUIRED -- {len(part.new)} unregistered numeral(s), none "
              f"of them covered: {base.path.name} is schema "
              f"{prose_baseline.LEGACY_SCHEMA} and cannot say which occurrence an "
              f"entry stands for.\n"
              f"  -> --migrate-baseline --reason \"...\"  (lossless; absorbs nothing)\n")
    elif part.new:
        print(f"NEW UNREGISTERED -- asserted result, in no manifest entry and not in "
              f"the recorded backlog ({len(part.new)}):")
        print(f"  {'line':>5}{'value':>12}  {'rule':<20}context")
        for num, rule, key in part.new:
            print(f"  {line_of(text, num.start):>5}{num.raw:>12}  {rule:<20}"
                  f"{excerpt(text, num)}")
            print(f"  {'':>5}{'':>12}  {'':<20}key: {key}")
            hint = part.reword_hint(key)
            if hint:
                print(f"  {'':>5}{'':>12}  {'':<20}looks like a reword of a vacated "
                      f"entry: {hint}")
        print("  -> register each in prose_manifest.py, or record them with "
              "--update-baseline --reason \"...\"\n")

    if by_state.get("MISMATCHED"):
        print("MISMATCHED -- prose disagrees with the generated value:")
        print(f"  {'claim':<38}{'line':>6}{'paper':>12}{'generated':>18}  provenance")
        for r in sorted(by_state["MISMATCHED"], key=lambda x: x.claim.id):
            ln = line_of(text, r.span[0]) if r.span else 0
            pv = f"{r.printed:g}" if r.printed is not None else "--"
            ev = r.rendering or "--"
            print(f"  {r.claim.id:<38}{ln:>6}{pv:>12}{ev:>18}  {r.claim.expect}")
            if r.derivation and r.derivation != r.claim.expect:
                print(f"  {'':<74}  = {r.derivation}")
            if r.detail:
                print(f"  {'':<74}  ! {r.detail}")
            if r.shadow:
                print(f"  {'':<74}  ~ non-prose twin at {r.shadow}")
        print()

    if by_state.get("UNDER_PRECISE"):
        print("UNDER_PRECISE -- prose rounds the generated value honestly, but at a "
              "precision that cannot resolve it (blocking):")
        print(f"  {'claim':<38}{'line':>6}{'paper':>12}{'generated':>18}  provenance")
        for r in sorted(by_state["UNDER_PRECISE"], key=lambda x: x.claim.id):
            ln = line_of(text, r.span[0]) if r.span else 0
            pv = f"{r.printed:g}" if r.printed is not None else "--"
            print(f"  {r.claim.id:<38}{ln:>6}{pv:>12}{r.rendering:>18}  {r.claim.expect}")
            if r.derivation and r.derivation != r.claim.expect:
                print(f"  {'':<74}  = {r.derivation}")
            print(f"  {'':<74}  ! {r.detail}")
            if r.shadow:
                print(f"  {'':<74}  ~ non-prose twin at {r.shadow}")
        print()

    if by_state.get("OVER_PRECISE"):
        print("OVER_PRECISE -- prose is inside the band but prints a digit it did not "
              "get right (blocking):")
        print(f"  {'claim':<38}{'line':>6}{'paper':>12}{'generated':>18}  provenance")
        for r in sorted(by_state["OVER_PRECISE"], key=lambda x: x.claim.id):
            ln = line_of(text, r.span[0]) if r.span else 0
            pv = f"{r.printed:g}" if r.printed is not None else "--"
            print(f"  {r.claim.id:<38}{ln:>6}{pv:>12}{r.rendering:>18}  {r.claim.expect}")
            if r.derivation and r.derivation != r.claim.expect:
                print(f"  {'':<74}  = {r.derivation}")
            print(f"  {'':<74}  ! {r.detail}")
            if r.shadow:
                print(f"  {'':<74}  ~ non-prose twin at {r.shadow}")
        print()

    # PRECISION POLICY is a statement about the manuscript's typography, not
    # about its arithmetic, so it lists every claim that prints too few digits
    # whatever its accuracy state -- including MATCHED ones, where the number
    # happens to land inside the band but the sentence still under-reports it.
    # Only the UNDER_PRECISE rows block; the rest are advice for the rewrite.
    thin = [r for r in results
            if r.ndec is not None and r.need_dp is not None and r.ndec < r.need_dp]
    if thin:
        print(f"PRECISION POLICY -- asserted results printed to fewer decimals than "
              f"the claim can be checked at ({len(thin)}):")
        print(f"  {'claim':<38}{'line':>6}{'prints':>8}{'needs':>7}  "
              f"{'should read':<14}state")
        for r in sorted(thin, key=lambda x: x.claim.id):
            ln = line_of(text, r.span[0]) if r.span else 0
            print(f"  {r.claim.id:<38}{ln:>6}{r.ndec:>6}dp{r.need_dp:>5}dp  "
                  f"{r.want:<14}{r.state}")
        print("  -> fix by printing more digits, or, if the sentence must round "
              "hard, declare ('abs', x) on the claim and say why in its note.\n")

    for state in ANCHOR_FAULTS:
        if by_state.get(state):
            print(f"{state} -- {FAULT_BLURB[state]}:")
            for r in by_state[state]:
                print(f"  {r.claim.id:<38}{r.detail}")
                print(f"  {'':<38}anchor: {r.claim.anchor}")
            print()

    if by_state.get("NO_GENERATOR"):
        print("NO_GENERATOR -- registered as unbackable, reason recorded:")
        for r in sorted(by_state["NO_GENERATOR"], key=lambda x: x.claim.id):
            ln = line_of(text, r.span[0]) if r.span else 0
            print(f"  {r.claim.id:<38}{ln:>6}{r.printed:>12g}")
            print(f"  {'':<38}{r.detail}")
            if r.shadow:
                print(f"  {'':<38}~ non-prose twin at {r.shadow}")
        print()

    if by_state.get("MATCHED"):
        print("MATCHED:")
        for r in sorted(by_state["MATCHED"], key=lambda x: x.claim.id):
            ln = line_of(text, r.span[0]) if r.span else 0
            print(f"  {r.claim.id:<38}{ln:>6}{r.printed:>12g}{r.rendering:>18}  "
                  f"{r.claim.expect.split(':')[-1][:36]}")
            if r.shadow:
                print(f"  {'':<38}~ non-prose twin at {r.shadow}")
        print()

    if part.backlog:
        page, lo, after = _page("backlog", part.backlog, limit, offset)
        print(f"BACKLOG UNREGISTERED -- recorded in {base.path.name}, not blocking "
              f"({len(part.backlog)} total, showing {len(page)} from offset {lo}):")
        print(f"  {'line':>5}{'value':>12}  {'rule':<20}context")
        for num, rule, _key in page:
            print(f"  {line_of(text, num.start):>5}{num.raw:>12}  {rule:<20}"
                  f"{excerpt(text, num)}")
        if lo or after:
            print(f"  ... {lo} before, {after} after "
                  f"(page with --offset N; --limit 0 prints all)")
        print()

    if part.vacated:
        rows = sorted(part.vacated.items())
        page, lo, after = _page("vacated", rows, limit, 0)
        n = sum(part.vacated.values())
        print(f"VACATED -- {n} recorded occurrence(s) in {len(part.vacated)} key(s) no "
              f"longer appear (the prose was fixed, reworded or deleted; prune with "
              f"--update-baseline):")
        for key, k in page:
            print(f"  - {key}" + (f"  (x{k})" if k > 1 else ""))
        if after:
            print(f"  ... {after} more")
        print("  a vacated entry is NOT a free slot: nothing can occupy it but the\n"
              "  same number in the same sentence, so the next occurrence of that\n"
              "  value is NEW and blocks.\n")

    counts: dict[str, int] = {}
    for _, rule in rejected:
        counts[rule] = counts.get(rule, 0) + 1
    print("classified incidental (not checked; counts shown so no rule hides work):")
    for rule, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {rule:<34}{n:>6}")
    if show_incidental:
        print("\nincidental detail (audit a rule you suspect of over-suppressing):")
        print(f"  {'line':>5}{'value':>12}  {'rule':<28}context")
        for num, rule in rejected:
            if rule.startswith("masked:"):
                continue  # masked regions are owned by build_paper_tables.py --check
            print(f"  {line_of(text, num.start):>5}{num.raw:>12}  {rule:<28}"
                  f"{excerpt(text, num, 34)}")

    # Blocking = proven wrong (MISMATCHED), proven uncheckable as written
    # (UNDER_PRECISE), silently unchecked (anchor faults), newly unverified, or
    # withdrawn from the manifest since the record was written. The recorded
    # backlog is none of those: it is known, counted and printed above, and it
    # is the prose rewrite's job.
    #
    # Withdrawal counts because deleting a claim used to be the one edit that
    # reduced the blocking count without settling anything: three MISMATCHED
    # claims dropped from the manifest moved three findings from `mismatched`
    # into `new` and left the total at 38.
    mismatched = len(by_state.get("MISMATCHED", []))
    coarse = len(by_state.get("UNDER_PRECISE", []))
    faults = sum(len(by_state.get(s, [])) for s in ANCHOR_FAULTS)
    withdrawn = len(cen.withdrawn)
    failed = (mismatched + coarse + faults + len(part.new) + withdrawn
              + (len(part.backlog) if strict else 0))
    print(f"\n{BAR}\n{'FAIL' if failed else 'PASS'}: {failed} blocking finding(s) "
          f"= {mismatched} mismatched + {coarse} under-precise + {faults} anchor "
          f"fault(s) + {len(part.new)} new unregistered + {withdrawn} claim(s) "
          f"withdrawn"
          f"{f' + {len(part.backlog)} backlog (--strict)' if strict else ''}. "
          f"\n{len(by_state.get('NO_GENERATOR', []))} registered without a generator; "
          f"{len(part.backlog)} in the recorded backlog"
          f"{'' if strict else ' (not blocking)'}"
          f"{'' if cen.known else '; the record predates the manifest census, so a '
                                  'withdrawn claim cannot be detected -- re-record'}"
          f".\n{BAR}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", type=Path, default=P_TEX, metavar="PATH",
                    help="manuscript to check (default: the live "
                         "paper_reference/Area_Free_Main.tex). Point it at a copy to "
                         "exercise the checker without touching the manuscript.")
    ap.add_argument("--limit", type=int, default=40,
                    help="max BACKLOG rows to print, 0 for all (default 40). NEW "
                         "unregistered rows are never truncated.")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many BACKLOG rows, for paging through it")
    ap.add_argument("--baseline", type=Path, default=P_BASELINE, metavar="PATH",
                    help="recorded backlog (default: paper_tooling/prose_baseline.json)")
    ap.add_argument("--strict", action="store_true",
                    help="block on the recorded backlog too -- the end state, once "
                         "the backlog has been worked off")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the baseline from this run. Requires --reason, "
                         "refuses while an anchor fault stands, prints and "
                         "permanently ledgers every finding it absorbs and every "
                         "claim withdrawn from the manifest. It exits with the gate "
                         "code of the state it leaves behind, so it cannot turn a "
                         "MISMATCHED run green.")
    ap.add_argument("--migrate-baseline", action="store_true",
                    help="convert a schema-1 (count-based) record to schema 2 "
                         "(occurrence-addressed). Lossless: the occurrences the "
                         "counts covered stay recorded and the rest stay NEW, so it "
                         "absorbs nothing. Requires --reason.")
    ap.add_argument("--ledger", action="store_true",
                    help="print every record event the baseline has ever had -- "
                         "each absorbed finding and each withdrawn claim -- and exit")
    ap.add_argument("--reason", default="",
                    help="why the baseline is being rewritten; stored in the file "
                         "and echoed by every later run")
    ap.add_argument("--show-incidental", action="store_true",
                    help="expand the incidental-classification summary")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the full result set to this path")
    ap.add_argument("--selftest", action="store_true",
                    help="sweep the tolerance rule for monotonicity and typography "
                         "independence, then exit. Reads no manuscript.")
    args = ap.parse_args()

    if args.selftest:
        return 1 if selftest_tolerance() else 0

    text = args.tex.read_text(encoding="utf-8")
    spans = mask_regions(text)
    mask = build_mask(text, spans)
    src = Sources()

    results = check_claims(text, src, mask)
    asserted, rejected = scan(text, mask)
    base = prose_baseline.load(args.baseline)

    if args.ledger:
        for line in prose_baseline.render_ledger(base):
            print(line)
        return 0

    items = keyed(text, unregistered_of(results, asserted))
    part = prose_baseline.partition(items, base)
    states = claim_states(text, results)
    cen = prose_baseline.census(states, base)
    code = report(text, results, asserted, rejected, args.limit, args.show_incidental,
                  args.tex, part, base, cen, args.offset, args.strict)

    if args.json:
        payload = {
            "production_model": GART,
            "manuscript": str(args.tex),
            "baseline": {"path": str(args.baseline), "exists": base.exists,
                         "recorded": base.recorded, "entries": base.total},
            "claims": [{"id": r.claim.id, "state": r.state, "line":
                        line_of(text, r.span[0]) if r.span else None,
                        "printed": r.printed, "expected": r.expected,
                        "printed_dp": r.ndec, "required_dp": r.need_dp,
                        "should_read": r.want,
                        "radius": (radius(r.expected, r.claim.tol)[0]
                                   if r.expected is not None else None),
                        "expect_spec": r.claim.expect, "derivation": r.derivation,
                        "detail": r.detail, "shadow": r.shadow} for r in results],
            "new_unregistered": [{"line": line_of(text, n.start), "value": n.raw,
                                  "rule": rule, "key": key, "context": excerpt(text, n)}
                                 for n, rule, key in part.new],
            "backlog_unregistered": [{"line": line_of(text, n.start), "value": n.raw,
                                      "rule": rule, "key": key,
                                      "context": excerpt(text, n)}
                                     for n, rule, key in part.backlog],
            "vacated_baseline": part.vacated,
            "claims_withdrawn": [{"id": cid, **d} for cid, d in cen.withdrawn],
            "claims_added": cen.added,
            "census_known": cen.known,
            "gate": gate_breakdown(results, part, cen, args.strict),
            "incidental_counts": {r: sum(1 for _, x in rejected if x == r)
                                  for r in {x for _, x in rejected}},
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # UNDER_PRECISE travels with MISMATCHED into the record's note: both are
    # registered claims that are blocking at the moment the record is written.
    blocking_claims = [r.claim.id for r in results
                       if r.state in ("MISMATCHED", "UNDER_PRECISE")]
    faults = sum(1 for r in results if r.state in ANCHOR_FAULTS)

    if args.migrate_baseline:
        plan = prose_baseline.plan_migration(items, base)
        prose_baseline.migrate(base, plan, cen, args.reason, model=GART, tex=args.tex,
                               claim_states=states, mismatched=blocking_claims)
        # The promise a migration makes: the schema-1 verdict is preserved
        # exactly, so the occurrences the counts did not cover are still NEW.
        return after_write(args, base, items, blocking_claims, faults,
                           expect_new=plan.uncovered)

    if args.update_baseline:
        prose_baseline.record(base, part, cen, args.reason, model=GART, tex=args.tex,
                              claim_states=states, anchor_faults=faults,
                              mismatched=blocking_claims)
        return after_write(args, base, items, blocking_claims, faults, expect_new=[])
    return code


def claim_states(text: str, results: list[Result]) -> dict[str, dict]:
    """The manifest roster as this run saw it, for the record's census.

    Stored with each claim's verdict and printed value, so that when a claim
    later disappears from ``prose_manifest.py`` the report can say *what it was*
    -- "withdrawn while MISMATCHED, paper 3.33 vs generated 3.19" -- instead of
    merely noting that the roster shrank.
    """
    return {r.claim.id: {"state": r.state,
                         "expect": r.claim.expect or "(no generator)",
                         "line": line_of(text, r.span[0]) if r.span else None,
                         "printed": r.printed, "expected": r.expected}
            for r in results}


def gate_breakdown(results: list[Result], part: "prose_baseline.Partition",
                   cen: "prose_baseline.Census", strict: bool) -> dict[str, int]:
    """The blocking count, itemised.  ``report`` prints it; ``--json`` exports it."""
    by = {}
    for r in results:
        by[r.state] = by.get(r.state, 0) + 1
    g = {"mismatched": by.get("MISMATCHED", 0),
         "under_precise": by.get("UNDER_PRECISE", 0),
         "anchor_faults": sum(by.get(s, 0) for s in ANCHOR_FAULTS),
         "new_unregistered": len(part.new),
         "claims_withdrawn": len(cen.withdrawn),
         "backlog_blocking": len(part.backlog) if strict else 0}
    g["blocking"] = sum(g.values())
    return g


def after_write(args, base: "prose_baseline.Baseline", items: list,
                blocking_claims: list[str], faults: int,
                expect_new: list[str]) -> int:
    """Re-derive the gate from the record just written, and return that code.

    The defect this exists for: ``--update-baseline`` used to ``return 0``
    unconditionally, so a CI step reading only the exit code went permanently
    green while 16 MISMATCHED claims stood.  A baseline operation is allowed to
    record a finding; it is not allowed to be the thing that decides whether the
    run passed.  So the record is reloaded from disk and the gate is recomputed
    against it -- the same arithmetic an ordinary run would do next.

    ``expect_new`` additionally holds a write to its own promise.  A migration
    claims to absorb nothing; if the post-write partition disagrees with the
    pre-write NEW set, the conversion silently changed a verdict and the run
    fails loudly instead of reporting success.
    """
    fresh = prose_baseline.load(args.baseline)
    post = prose_baseline.partition(items, fresh)
    got = sorted(k for _n, _r, k in post.new)
    if got != sorted(expect_new):
        print(f"\n{BAR}\nFAIL: the write changed the verdict it promised not to "
              f"change.\n  expected {len(expect_new)} NEW after writing, found "
              f"{len(got)}\n{BAR}")
        return 1
    blocking = len(blocking_claims) + faults + len(post.new)
    if args.strict:
        blocking += len(post.backlog)
    print(f"\n{BAR}\n{'FAIL' if blocking else 'PASS'}: gate after the write = "
          f"{blocking} blocking finding(s) = {len(blocking_claims)} mismatched or "
          f"under-precise + {faults} anchor fault(s) + {len(post.new)} new "
          f"unregistered"
          f"{f' + {len(post.backlog)} backlog (--strict)' if args.strict else ''}."
          f"\nRecording a finding does not settle it. "
          f"{'Fix the numbers or register them.' if blocking else ''}\n{BAR}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
