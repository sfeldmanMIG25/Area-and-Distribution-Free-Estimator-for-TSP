"""Occurrence-addressed record of UNREGISTERED prose numerals, plus the gate it feeds.

The problem this solves
-----------------------
``check_prose_numbers.py`` classifies ~600 prose numerals as asserted results
that no manifest entry covers.  Every one is a real finding, and clearing them
is a prose-rewrite job.  Meanwhile the checker exits 1 on every run and prints
the first 40 of the 600.  Both halves fail:

* a gate that can never pass is turned off, and
* a genuinely new unverified number lands somewhere in the tail of the list and
  moves a counter from 602 to 603, which is not a signal anyone reads.

So: record the known backlog once, deliberately, and key the exit code to what
is *new* against that record plus the states that are proven wrong (MISMATCHED,
anchor faults, a claim withdrawn from the manifest).  The backlog stays printed
on every run -- visible, countable, pageable -- and stops blocking.

Why the record is occurrence-addressed (schema 2)
-------------------------------------------------
Schema 1 keyed an entry on ``section :: value :: rule`` and stored a *count*.
Coverage was then "this key may occur up to N times", which makes findings
**fungible**: any occurrence could sit in any slot with the same key.  Four
consequences, all demonstrated on the live manuscript rather than hypothesised:

* a routine reword left 20 slots vacated across 19 keys -- the *normal* state
  after any prose edit;
* inserting a genuinely new sentence ("A late refit reaches 0.88\\% MAPE on the
  withheld fold.") landed in BACKLOG rather than NEW, because a
  ``0.88 :: tier1:unit`` slot happened to be free.  No flag, no diff, blocking
  count unchanged;
* deleting a claim from ``prose_manifest.py`` demoted MISMATCHED to
  unregistered, and the vacated-slot rule then absorbed the released numeral;
* ``--update-baseline`` returned 0 unconditionally, so the whole record could be
  rewritten green with 16 MISMATCHED outstanding.

Schema 2 removes fungibility at the root.  An entry is one *occurrence*:

``section``  the nearest enclosing ``\\section``/``\\subsection`` (its
             ``\\label`` when it has one, else its slugged title), or
             ``frontmatter`` before the first sectioning command.
``value``    the numeral exactly as the manuscript prints it.
``rule``     the classifier rule that called it an asserted result.
``ctx:...``  a digest of the *normalized* text around the numeral.

The digest is what makes a slot non-transferable.  A recorded entry is matched
only by an occurrence in the same section, with the same value, the same rule
**and the same surrounding words**.  Nothing else can fill it, so a vacated slot
is not a slot at all: it is a deletion, reported as VACATED, and the next
occurrence of that value is NEW.

Normalization before digesting, so the digest survives what should not matter:
whitespace and line reflow collapse, LaTeX spacing macros (``\\,`` ``~`` ...)
collapse, case folds, and **every numeral -- including the focus numeral --
becomes ``#``**.  Editing a neighbouring number therefore does not rot this
one's identity; the value already lives in the key.

The cost, stated rather than hidden
-----------------------------------
Rewording a sentence that contains a backlogged number changes its digest, so
the occurrence is reported NEW (blocking) and the old one VACATED.  Schema 1
hid that, which is precisely how a real finding was absorbed.  Two mitigations,
neither of which touches the gate:

* the report pairs each NEW with a VACATED that shares its section, value and
  rule and labels it ``looks like a reword of ...``, so a run full of rewords
  reads as one at a glance; and
* acknowledging them is one command -- but that command now writes the absorbed
  keys into a permanent ledger, so the acknowledgement is a fact about the
  repository rather than a way to make a run green.

The honest resolution of a reworded backlog entry is to register the number in
``prose_manifest.py``.  Absorbing is the escape hatch, and it leaves a mark.

Every way of silencing a finding, and what now stops it
-------------------------------------------------------
``--update-baseline`` exits with the gate code of the state it leaves behind, so
recording cannot make MISMATCHED stop blocking -- it can only record alongside
it.  Absorption needs ``--reason``, prints each absorbed key, appends a ledger
event that is never rewritten, and increments lifetime counters that are
recomputed from the ledger on every write and printed by every later run.
Withdrawing a claim from the manifest is detected against the recorded census,
blocks until acknowledged, and -- when the claim was MISMATCHED at the time --
is named in the banner of every subsequent run, forever.

What is deliberately *not* defended: deleting ``prose_baseline.json`` outright.
Nothing inside a file can survive its own deletion.  The banner prints the
ledger length on every run, so a re-seeded record announces itself as one.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 2
LEGACY_SCHEMA = 1

WARNING = (
    "Recorded backlog of unverified prose numbers. Adding an entry here stops "
    "check_prose_numbers.py from blocking on it. Do not edit by hand and do not "
    "regenerate to make a red run green -- every entry added is a number the "
    "paper asserts and nothing verifies. Entries are occurrence-addressed: the "
    "ctx: component is a digest of the surrounding prose, so a slot cannot be "
    "filled by a different sentence. See ledger[] for every finding this file "
    "has ever absorbed and every claim ever withdrawn from the manifest."
)

SEP = " :: "
CTX_PREFIX = "ctx:"

# Half-width, in normalized characters, of the context window behind the digest.
# Wide enough that two different sentences asserting the same value in the same
# section separate; narrow enough that an edit elsewhere in the paragraph does
# not rot it.
CTX_HALF = 72
CTX_HEX = 12

# Sectioning commands, most specific first, so \subsection wins inside a \section.
_SECTION_CMD = re.compile(
    r"\\(section|subsection|subsubsection)\*?\s*\{", re.M)
_APPENDIX = re.compile(r"\\appendix\b")
_LABEL_AFTER = re.compile(r"\A\s*\\label\s*\{([^{}]*)\}")

# One numeral, in any of the manuscript's spellings, for masking purposes only.
_NUM_RUN = re.compile(r"\d(?:\d|\{,\}(?=\d)|,(?=\d\d\d)|\.(?=\d))*")
# LaTeX spacing macros that carry no words.
_SPACE_MACRO_CHARS = ",;:! "


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "untitled"


def _braced(text: str, open_brace: int) -> tuple[str, int]:
    """Balanced ``{...}`` starting at ``open_brace``; returns (body, end)."""
    depth, i = 0, open_brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i], i + 1
        i += 1
    return text[open_brace + 1:], len(text)


def section_index(text: str) -> list[tuple[int, str]]:
    """``(start_offset, section_key)`` for every sectioning command, in order.

    The key prefers the section's ``\\label`` over its title: a title can be
    reworded without changing what the section *is*, and rewording a heading
    should not reclassify every number underneath it.
    """
    app = _APPENDIX.search(text)
    app_at = app.start() if app else len(text)
    out: list[tuple[int, str]] = []
    top = ""
    for m in _SECTION_CMD.finditer(text):
        title, end = _braced(text, m.end() - 1)
        lab = _LABEL_AFTER.match(text[end:end + 80])
        name = _slug(lab.group(1)) if lab else _slug(re.sub(r"\\[a-zA-Z]+|[{}$]", "", title))
        if m.group(1) == "section":
            top = name
            key = name
        else:
            key = f"{top}/{name}" if top else name
        if m.start() >= app_at:
            key = f"appendix:{key}"
        out.append((m.start(), key))
    return out


def section_at(index: list[tuple[int, str]], pos: int) -> str:
    """Section key covering ``pos``; ``frontmatter`` before the first heading."""
    key = "frontmatter"
    for start, name in index:
        if start > pos:
            break
        key = name
    return key


# -- Context addressing ------------------------------------------------------
@dataclass(frozen=True)
class NormText:
    """A normalized copy of the manuscript, plus a map from original offsets."""

    norm: str
    o2n: list[int]


def normalize(text: str) -> NormText:
    """Fold away everything a digest must not depend on.

    Whitespace runs and LaTeX spacing macros collapse to one space, case folds,
    and every numeral collapses to ``#``.  The returned offset map lets a caller
    locate an original character in the normalized string, so the digest window
    is measured in normalized characters and a reflow moves nothing.
    """
    out: list[str] = []
    n = len(text)
    o2n = [0] * (n + 1)
    i = 0
    prev_space = True
    while i < n:
        o2n[i] = len(out)
        ch = text[i]
        if ch.isspace() or ch == "~":
            if not prev_space:
                out.append(" ")
                prev_space = True
            i += 1
            continue
        if ch == "\\" and i + 1 < n and text[i + 1] in _SPACE_MACRO_CHARS:
            o2n[i + 1] = len(out)
            if not prev_space:
                out.append(" ")
                prev_space = True
            i += 2
            continue
        if ch.isdigit():
            m = _NUM_RUN.match(text, i)
            end = m.end() if m else i + 1
            for j in range(i, end):
                o2n[j] = len(out)
            out.append("#")
            prev_space = False
            i = end
            continue
        out.append(ch.lower())
        prev_space = False
        i += 1
    for j in range(i, n + 1):
        o2n[j] = len(out)
    return NormText("".join(out), o2n)


def context_digest(nt: NormText, start: int) -> str:
    """Identity of the sentence position a numeral occupies."""
    p = nt.o2n[start] if start < len(nt.o2n) else len(nt.norm)
    window = nt.norm[max(0, p - CTX_HALF):p + CTX_HALF]
    return CTX_PREFIX + hashlib.sha256(window.encode("utf-8")).hexdigest()[:CTX_HEX]


def entry_key(section: str, value: str, rule: str, digest: str) -> str:
    return f"{section}{SEP}{value}{SEP}{rule}{SEP}{digest}"


def legacy_key_of(key: str) -> str:
    """The schema-1 key a schema-2 key would have had: drop the ``ctx:`` part."""
    head, sep, tail = key.rpartition(SEP)
    return head if sep and tail.startswith(CTX_PREFIX) else key


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- The record --------------------------------------------------------------
@dataclass
class Baseline:
    """A loaded record, or an empty one when no file exists yet."""

    path: Path
    exists: bool
    schema: int = SCHEMA
    legacy: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)
    first_seen: dict[str, str] = field(default_factory=dict)
    legacy_counts: dict[str, int] = field(default_factory=dict)
    claims: dict[str, dict] = field(default_factory=dict)
    has_census: bool = False
    recorded: dict[str, object] = field(default_factory=dict)
    ledger: list[dict] = field(default_factory=list)
    lifetime: dict[str, object] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _as_list(v: object) -> list:
    return list(v) if isinstance(v, (list, tuple)) else []


def _normalize_event(e: dict, index: int) -> dict:
    """Give a schema-1 ``history`` entry the shape a schema-2 ledger event has.

    Done at load rather than at migration so that an absorption recorded before
    the conversion keeps counting against the record's lifetime.  Schema 1 wrote
    ``absorbed_new``/``absorbed_keys`` only into the *latest* record, which is
    how one further routine re-record made an absorbed finding disappear; the
    keys did survive in ``history``, and this is where they are read back.

    The first event in a ledger is the seed (a record with nothing before it
    takes in the whole backlog by definition); every later one is an update, and
    what it took in was absorbed.  That is the same rule ``record`` applies now.
    """
    if "kind" in e:
        out = dict(e)
        out["mismatched_at_record"] = _as_list(e.get("mismatched_at_record"))
        return out
    out = dict(e)
    keys = _as_list(e.get("absorbed_keys"))
    out["kind"] = "seed" if index == 0 else "update"
    out["absorbed"] = [] if index == 0 else keys
    if index == 0:
        out["seeded"] = keys
    out["pruned"] = []  # schema 1 stored a count, not the keys
    out["pruned_count"] = int(e.get("pruned_stale", 0) or 0)
    out["claims_withdrawn"] = []  # schema 1 had no manifest census
    out["mismatched_at_record"] = _as_list(e.get("mismatched_claims"))
    out["from_schema"] = LEGACY_SCHEMA
    return out


def load(path: Path) -> Baseline:
    if not path.exists():
        return Baseline(path=path, exists=False)
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema = raw.get("schema")
    entries = raw.get("entries", {})
    if schema == LEGACY_SCHEMA:
        # Readable, but it grants no coverage: a count cannot say *which*
        # occurrence it covers, which is the property the gate now depends on.
        return Baseline(
            path=path, exists=True, schema=LEGACY_SCHEMA, legacy=True,
            legacy_counts={k: int(v["count"]) for k, v in entries.items()},
            examples={k: str(v.get("example", "")) for k, v in entries.items()},
            recorded=raw.get("recorded", {}),
            ledger=[_normalize_event(e, i)
                    for i, e in enumerate(raw.get("history", []))])
    if schema != SCHEMA:
        raise SystemExit(
            f"{path.name}: schema {schema!r}, expected {SCHEMA} (or {LEGACY_SCHEMA} "
            f"to migrate). Refusing to guess -- re-record with --update-baseline.")
    return Baseline(
        path=path, exists=True, schema=SCHEMA,
        counts={k: int(v["count"]) for k, v in entries.items()},
        examples={k: str(v.get("example", "")) for k, v in entries.items()},
        first_seen={k: str(v.get("first_recorded", "")) for k, v in entries.items()},
        claims=dict(raw.get("claims", {})),
        has_census="claims" in raw,
        recorded=raw.get("recorded", {}),
        ledger=[_normalize_event(e, i)
                for i, e in enumerate(raw.get("ledger", []))],
        lifetime={})  # recomputed from the ledger; never trusted from the file


# -- Partitioning the current run against the record -------------------------
@dataclass
class Partition:
    """The current unregistered list split against the record."""

    new: list[tuple[object, str, str]] = field(default_factory=list)
    backlog: list[tuple[object, str, str]] = field(default_factory=list)
    vacated: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)

    def reword_hint(self, key: str) -> str | None:
        """A VACATED entry that shares this key's section, value and rule.

        Advice for the reader only.  It is never consulted by the gate and never
        widens coverage: pairing a NEW with a VACATED does not stop the NEW from
        blocking.  Its whole job is to let a run full of rewords be recognised
        as one without reading 40 contexts.
        """
        want = legacy_key_of(key)
        for cand in self.vacated:
            if legacy_key_of(cand) == want:
                return cand
        return None


def partition(items: list[tuple[object, str, str, str]], base: Baseline) -> Partition:
    """Split ``(numeral, rule, key, example)`` items into new vs recorded backlog.

    ``key`` is occurrence-addressed, so ``base.counts.get(key)`` is 1 for a
    recorded occurrence and 0 for anything else.  The count arithmetic survives
    only to cover the case of two occurrences whose normalized context windows
    are byte-identical, which is a genuine tie rather than a free slot.
    """
    out = Partition()
    seen: dict[str, int] = {}
    for num, rule, key, example in items:
        seen[key] = seen.get(key, 0) + 1
        out.counts[key] = seen[key]
        out.examples.setdefault(key, example)
        allowed = 0 if base.legacy else base.counts.get(key, 0)
        (out.backlog if seen[key] <= allowed else out.new).append((num, rule, key))
    out.vacated = {k: n - out.counts.get(k, 0)
                   for k, n in base.counts.items() if n > out.counts.get(k, 0)}
    return out


# -- The manifest census -----------------------------------------------------
@dataclass
class Census:
    """Manifest membership now, against membership when the record was written."""

    withdrawn: list[tuple[str, dict]] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    known: bool = False

    @property
    def withdrawn_mismatched(self) -> list[tuple[str, dict]]:
        return [(cid, d) for cid, d in self.withdrawn
                if str(d.get("state", "")) == "MISMATCHED"]


def census(claim_states: dict[str, dict], base: Baseline) -> Census:
    """Which claims have left the manifest since the record was written.

    A withdrawn claim releases its numeral back into the unregistered pool.  Under
    schema 1 that was invisible in the exit code: three MISMATCHED claims deleted
    from the manifest moved three findings from ``mismatched`` to ``new`` and left
    the blocking count exactly where it was.  Recording the roster makes the
    withdrawal itself a finding, with the state the claim was last known to be in.
    """
    if not base.has_census:
        return Census(known=False)
    withdrawn = [(cid, dict(d)) for cid, d in sorted(base.claims.items())
                 if cid not in claim_states]
    added = [cid for cid in sorted(claim_states) if cid not in base.claims]
    return Census(withdrawn=withdrawn, added=added, known=True)


# -- Reporting the record ----------------------------------------------------
def _lifetime_from(ledger: list[dict]) -> dict[str, object]:
    """Counters derived from the ledger, so a later write cannot reset them.

    ``absorbed`` counts only ``update`` events.  Seeding an empty record and
    migrating schema 1 to schema 2 take in the existing backlog by definition and
    are not acts of silencing; both are still ledgered, with their own kind.
    """
    absorbed = [k for e in ledger if e.get("kind") == "update"
                for k in _as_list(e.get("absorbed"))]
    withdrawn = [w for e in ledger for w in _as_list(e.get("claims_withdrawn"))]
    return {
        "records": len(ledger),
        "absorbed": len(absorbed),
        "pruned": sum(len(_as_list(e.get("pruned"))) + int(e.get("pruned_count", 0) or 0)
                      for e in ledger),
        "claims_withdrawn": len(withdrawn),
        "mismatched_claims_withdrawn": sum(
            1 for w in withdrawn if str(w.get("state", "")) == "MISMATCHED"),
    }


def absorption_history(base: Baseline) -> list[tuple[str, str, str]]:
    """``(when, reason, key)`` for every finding this record has ever absorbed."""
    out: list[tuple[str, str, str]] = []
    for e in base.ledger:
        if e.get("kind") != "update":
            continue
        when, why = str(e.get("at", "?")), str(e.get("reason", ""))
        for key in _as_list(e.get("absorbed")):
            out.append((when, why, str(key)))
    return out


def withdrawal_history(base: Baseline) -> list[tuple[str, str, dict]]:
    """``(when, reason, claim record)`` for every claim ever withdrawn."""
    out: list[tuple[str, str, dict]] = []
    for e in base.ledger:
        when, why = str(e.get("at", "?")), str(e.get("reason", ""))
        for w in _as_list(e.get("claims_withdrawn")):
            out.append((when, why, dict(w)))
    return out


BANNER_ABSORB_LINES = 25


def banner(base: Baseline, model: str) -> list[str]:
    """Header lines describing the record and everything it has ever silenced."""
    if not base.exists:
        return [f"baseline: NONE at {base.path.name} -- every unregistered numeral "
                f"is treated as new, so this run cannot pass.",
                "          record the known backlog once with: "
                "--update-baseline --reason \"...\""]
    if base.legacy:
        return [f"baseline: {base.path.name} is schema {LEGACY_SCHEMA} (count-based).",
                "          A count cannot say which occurrence it covers, so a vacated "
                "slot silently absorbs the next",
                "          number with the same value. It grants no coverage here. "
                "Convert it with:",
                "            --migrate-baseline --reason \"...\"",
                "          Migration is lossless: the occurrences the counts covered "
                "stay recorded, the excess stays NEW."]
    r = base.recorded
    life = base.lifetime or _lifetime_from(base.ledger)
    lines = [f"baseline: {base.total} recorded occurrence(s) in {len(base.counts)} keys, "
             f"written {r.get('at', '?')} against model {r.get('production_model', '?')}",
             f"          reason: {r.get('reason', '(none recorded)')}",
             f"          ledger: {life.get('records', 0)} record(s) -- lifetime "
             f"{life.get('absorbed', 0)} finding(s) absorbed, "
             f"{life.get('pruned', 0)} pruned, "
             f"{life.get('claims_withdrawn', 0)} claim(s) withdrawn from the manifest"]
    if r.get("production_model") not in (None, model):
        lines.append(f"          WARNING: recorded against {r.get('production_model')!r}, "
                     f"production model is now {model!r} -- the backlog describes "
                     f"numbers generated by a different model and should be re-recorded.")

    # Permanent, not "since the last update". Schema 1 stored absorbed_new in the
    # latest record only, so one further routine re-record set it to 0 and the
    # absorbed keys vanished from every subsequent run.
    absorbed = absorption_history(base)
    if absorbed:
        lines.append(f"          ABSORBED (permanent -- {len(absorbed)} number(s) this "
                     f"file silenced; nothing verifies them):")
        for when, why, key in absorbed[-BANNER_ABSORB_LINES:]:
            lines.append(f"            {when[:10]}  {key}")
            lines.append(f"                      reason: {why}")
        if len(absorbed) > BANNER_ABSORB_LINES:
            lines.append(f"            ... {len(absorbed) - BANNER_ABSORB_LINES} earlier "
                         f"absorption(s); --ledger prints all")

    withdrawn = withdrawal_history(base)
    bad = [(w, y, d) for w, y, d in withdrawn if str(d.get("state", "")) == "MISMATCHED"]
    if bad:
        lines.append(f"          WITHDRAWN WHILE MISMATCHED (permanent -- {len(bad)} "
                     f"claim(s) deleted from the manifest while proven wrong):")
        for when, why, d in bad:
            lines.append(f"            {when[:10]}  {d.get('id', '?')}  "
                         f"paper {d.get('printed', '?')} vs generated {d.get('expected', '?')}")
            lines.append(f"                      reason: {why}")
    other = len(withdrawn) - len(bad)
    if other:
        lines.append(f"          {other} further claim(s) withdrawn from the manifest "
                     f"over this record's life (--ledger)")
    return lines


def render_ledger(base: Baseline) -> list[str]:
    """Every record event, in order.  ``--ledger`` prints this."""
    if not base.ledger:
        return ["ledger: empty"]
    out = [f"ledger: {len(base.ledger)} event(s) in {base.path.name}"]
    for i, e in enumerate(base.ledger, 1):
        out.append(f"  [{i}] {e.get('at', '?')}  kind={e.get('kind', '?')}  "
                   f"model={e.get('production_model', '?')}")
        out.append(f"      reason: {e.get('reason', '')}")
        out.append(f"      tex: {e.get('tex', '?')} @ {str(e.get('tex_sha256', ''))[:16]}")
        out.append(f"      entries {e.get('n_entries', '?')} in "
                   f"{e.get('n_keys', '?')} keys; mismatched at record: "
                   f"{len(_as_list(e.get('mismatched_at_record')))}")
        if e.get("seeded"):
            out.append(f"      = seeded    {len(_as_list(e.get('seeded')))} occurrence(s)")
        for key in _as_list(e.get("absorbed")):
            out.append(f"      + absorbed  {key}")
        for key in _as_list(e.get("pruned")):
            out.append(f"      - pruned    {key}")
        for w in _as_list(e.get("claims_withdrawn")):
            out.append(f"      x withdrawn claim {w.get('id', '?')} "
                       f"(last state {w.get('state', '?')})")
    return out


# -- Writing the record ------------------------------------------------------
def _display_path(p: Path) -> str:
    """Repo-relative when it is in the repo, absolute otherwise.

    A record written against a scratch copy must say so: the stored path is the
    only evidence that the entries describe some other file.
    """
    try:
        return str(p.resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(p.resolve())


def _require_reason(reason: str) -> str:
    if not reason or not reason.strip():
        raise SystemExit("refusing to update: --reason is required and is stored "
                         "in the file, so the next reader knows what was taken in.")
    return reason.strip()


def _guard_anchor_faults(base: Baseline, anchor_faults: int) -> None:
    if anchor_faults:
        raise SystemExit(
            f"refusing to update {base.path.name}: {anchor_faults} anchor fault(s) "
            f"outstanding. A claim that stopped resolving has released its numeral "
            f"back into the unregistered pool, so recording now would absorb a "
            f"number that was under manifest control a moment ago. Fix the anchors "
            f"first.")


def _write(base: Baseline, event: dict, counts: dict[str, int],
           examples: dict[str, str], claim_states: dict[str, dict],
           first_seen: dict[str, str]) -> None:
    ledger = list(base.ledger) + [event]
    payload = {
        "schema": SCHEMA,
        "_warning": WARNING,
        "recorded": event,
        "lifetime": _lifetime_from(ledger),
        "entries": {k: {"count": counts[k], "example": examples.get(k, ""),
                        "first_recorded": first_seen.get(k, str(event["at"]))}
                    for k in sorted(counts)},
        "claims": {cid: dict(d) for cid, d in sorted(claim_states.items())},
        "ledger": ledger,
    }
    base.path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n",
                         encoding="utf-8")


def _event(kind: str, reason: str, *, model: str, tex: Path, counts: dict[str, int],
           absorbed: list[str], pruned: list[str], withdrawn: list[dict],
           mismatched: list[str]) -> dict:
    return {
        "at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "kind": kind,
        "reason": reason,
        "tex": _display_path(tex),
        "tex_sha256": sha256_of(tex),
        "production_model": model,
        "n_entries": sum(counts.values()),
        "n_keys": len(counts),
        "absorbed": absorbed,
        "pruned": pruned,
        "claims_withdrawn": withdrawn,
        # State of the registered claims at record time. A later reader can tell
        # whether this record was written over a manuscript already known stale.
        "mismatched_at_record": sorted(mismatched),
    }


def record(base: Baseline, part: Partition, cen: Census, reason: str, *,
           model: str, tex: Path, claim_states: dict[str, dict],
           anchor_faults: int, mismatched: list[str]) -> dict:
    """Rewrite the record from the current run.  Returns the ledger event written.

    Refuses on an anchor fault (the unregistered pool is untrustworthy) and on a
    legacy record (migrate first).  It does **not** refuse on MISMATCHED: those
    claims are registered, never become entries here, and keep blocking whatever
    this file says -- the caller's exit code is computed after this returns, from
    the state this call leaves behind.
    """
    if base.legacy:
        raise SystemExit(
            f"refusing to update {base.path.name}: it is schema {LEGACY_SCHEMA}. "
            f"Recording over it would re-seed a count-based backlog as if it were "
            f"occurrence-addressed and take in every current numeral as if it had "
            f"always been there. Run --migrate-baseline --reason \"...\" first; that "
            f"is lossless and absorbs nothing.")
    _guard_anchor_faults(base, anchor_faults)
    reason = _require_reason(reason)

    kind = "update" if base.counts else "seed"
    absorbed = [k for _n, _r, k in part.new]
    pruned = sorted(part.vacated)
    withdrawn = [{"id": cid, **{k: v for k, v in d.items() if k != "id"}}
                 for cid, d in cen.withdrawn]

    if mismatched:
        print(f"\nNOTE: {len(mismatched)} MISMATCHED claim(s) are outstanding and are "
              f"being recorded alongside this baseline. They are registered, so they "
              f"keep blocking whatever this file says, and this run will still fail:")
        for cid in sorted(mismatched):
            print(f"  ! {cid}")
    if withdrawn:
        print(f"\nACKNOWLEDGING {len(withdrawn)} claim(s) withdrawn from the manifest. "
              f"Each is written to the ledger permanently:")
        for w in withdrawn:
            flag = "!!" if w.get("state") == "MISMATCHED" else "  "
            print(f"  {flag} {w.get('id')}  (last state {w.get('state', '?')}, "
                  f"paper {w.get('printed', '?')} vs generated {w.get('expected', '?')})")
    if absorbed:
        label = ("SEEDING the record with" if kind == "seed"
                 else "ABSORBING")
        print(f"\n{label} {len(absorbed)} finding(s) -- each is a number the paper "
              f"asserts and nothing verifies:")
        for key in absorbed:
            print(f"  + {key}")
        if kind == "update":
            print("  these keys are written to the ledger and printed by every "
                  "later run.")
    if pruned:
        print(f"\nPRUNING {len(pruned)} recorded occurrence(s) that no longer appear:")
        for key in pruned:
            print(f"  - {key}")

    event = _event(kind, reason, model=model, tex=tex, counts=part.counts,
                   absorbed=absorbed if kind == "update" else [],
                   pruned=pruned, withdrawn=withdrawn, mismatched=mismatched)
    if kind == "seed":
        event["seeded"] = absorbed
    # An occurrence that survives a rewrite keeps the date it entered the record,
    # so "how long has this number gone unverified" is answerable from the file.
    first_seen = {k: base.first_seen.get(k) or str(event["at"]) for k in part.counts}
    _write(base, event, part.counts, part.examples, claim_states, first_seen)
    print(f"\nwrote {base.path} -- {event['n_entries']} occurrence(s) in "
          f"{event['n_keys']} keys, kind={kind}, absorbed "
          f"{len(event['absorbed'])}, pruned {len(pruned)}")
    return event


@dataclass
class Migration:
    """What a schema-1 -> schema-2 conversion carried over, and what it did not."""

    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)
    carried: list[str] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)


def plan_migration(items: list[tuple[object, str, str, str]],
                   base: Baseline) -> Migration:
    """Grant a schema-2 slot to exactly the occurrences schema 1 already covered.

    Occurrences are consumed in document order and matched against the legacy
    per-key allowance -- the same rule schema 1's ``partition`` used -- so the
    verdict is unchanged by the conversion: whatever was BACKLOG stays BACKLOG,
    whatever was NEW stays NEW.  Migration is therefore not an act of absorption
    and does not need to be, which is what makes it safe to run while findings
    are outstanding.
    """
    out = Migration()
    seen: dict[str, int] = {}
    for _num, _rule, key, example in items:
        lkey = legacy_key_of(key)
        seen[lkey] = seen.get(lkey, 0) + 1
        if seen[lkey] <= base.legacy_counts.get(lkey, 0):
            out.counts[key] = out.counts.get(key, 0) + 1
            out.examples.setdefault(key, example)
            out.carried.append(key)
        else:
            out.uncovered.append(key)
    out.dropped = {k: n - min(n, seen.get(k, 0))
                   for k, n in base.legacy_counts.items()
                   if n > seen.get(k, 0)}
    return out


def migrate(base: Baseline, plan: Migration, cen: Census, reason: str, *,
            model: str, tex: Path, claim_states: dict[str, dict],
            mismatched: list[str]) -> dict:
    """Convert a schema-1 record in place.  Absorbs nothing; returns the event."""
    if not base.legacy:
        raise SystemExit(f"{base.path.name} is already schema {SCHEMA}; "
                         f"--migrate-baseline has nothing to do.")
    reason = _require_reason(reason)
    print(f"\nMIGRATING {base.path.name}: schema {LEGACY_SCHEMA} -> {SCHEMA}, "
          f"count-based -> occurrence-addressed.")
    print(f"  carried over : {len(plan.carried)} occurrence(s) the counts covered")
    print(f"  left NEW     : {len(plan.uncovered)} occurrence(s) the counts did not "
          f"cover -- unchanged, still blocking")
    print(f"  legacy slots dropped (no occurrence): "
          f"{sum(plan.dropped.values())} in {len(plan.dropped)} key(s)")
    for key in plan.uncovered:
        print(f"  ! still NEW  {key}")

    event = _event("migrate", reason, model=model, tex=tex, counts=plan.counts,
                   absorbed=[], pruned=sorted(plan.dropped),
                   withdrawn=[], mismatched=mismatched)
    event["migrated_from_schema"] = LEGACY_SCHEMA
    event["carried"] = len(plan.carried)
    event["left_new"] = len(plan.uncovered)
    first_seen = {k: str(base.recorded.get("at", event["at"])) for k in plan.counts}
    _write(base, event, plan.counts, plan.examples, claim_states, first_seen)
    print(f"\nwrote {base.path} -- {event['n_entries']} occurrence(s) in "
          f"{event['n_keys']} keys, kind=migrate, absorbed 0")
    return event
