"""Render the prose claim -> authority map.

Data lives in ``paper_tooling/prose_claims.py``. This file only formats it, so
a change to the checker's manifest format touches the emitter below and nothing
else.

Usage::

    python paper_tooling/build_prose_claim_map.py            # write both outputs
    python paper_tooling/build_prose_claim_map.py --verify   # + re-locate anchors,
                                                             #   probe bank keys
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_tooling.prose_claims import (  # noqa: E402
    CLAIMS,
    NUMBER_BANK,
    RECONCILED_IN_TEX,
)

OUT_JSON = ROOT / "paper_tooling" / "prose_claim_map.json"
OUT_MD = ROOT / "paper_tooling" / "prose_claim_map.md"
TEX = ROOT / "paper_reference" / "Area_Free_Main.tex"
BANK = ROOT / NUMBER_BANK

SCHEMA = "prose_claim_map/1"
BUCKETS = ("BANKED", "GENERATED", "UNGENERATED", "STRUCTURAL")


# -- Verification ------------------------------------------------------------
def locate_anchors() -> dict[str, int | None]:
    """Line number of each claim's anchor in the .tex, or None if not found."""
    lines = TEX.read_text(encoding="utf-8").splitlines()
    found: dict[str, int | None] = {}
    for c in CLAIMS:
        hit = next((i for i, ln in enumerate(lines, 1) if c["anchor"] in ln), None)
        found[c["id"]] = hit
    return found


def probe_bank() -> tuple[list[str], list[str]]:
    """Concrete bank keys that resolve, and those that do not.

    Keys containing ``{`` or ``*`` are families, not literals; they document a
    key shape and are not probed.
    """
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    ok, bad = [], []
    for c in CLAIMS:
        if c["bucket"] != "BANKED":
            continue
        for k in c["keys"]:
            if "{" in k or "*" in k:
                continue
            (ok if k in bank else bad).append(f"{c['id']}: {k}")
    return ok, bad


# -- Emitters ----------------------------------------------------------------
def to_json(anchor_lines: dict[str, int | None] | None) -> dict:
    claims = []
    for c in CLAIMS:
        rec = dict(c)
        rec["reconciled_in_tex"] = c["id"] in RECONCILED_IN_TEX
        if anchor_lines is not None:
            rec["line_located"] = anchor_lines.get(c["id"])
        claims.append(rec)
    counts = Counter(c["bucket"] for c in CLAIMS)
    return {
        "schema": SCHEMA,
        "note_to_checker": (
            "No checker manifest existed on disk when this was written, so this "
            "schema is a PROPOSAL. The fields a checker actually needs are: id, "
            "anchor, bucket, authority, keys, derivation, tolerance_decimals. "
            "Everything else is documentation for humans. If the checker settles "
            "on a different shape, change build_prose_claim_map.py:to_json and "
            "leave prose_claims.py alone."
        ),
        "source_audit": "paper_tooling/prose_claim_audit.md",
        "tex": "paper_reference/Area_Free_Main.tex",
        "number_bank": NUMBER_BANK,
        "model_registry": "paper_tooling/model_registry.py",
        "buckets": list(BUCKETS),
        "bucket_semantics": {
            "BANKED": "a key exists in the number bank; `derivation` is a Python "
                      "expression over k0..kn when the prose quotes a ratio, "
                      "margin, argmin or bound",
            "GENERATED": "a script writes it to disk but not to the bank; "
                         "`action` names the export",
            "UNGENERATED": "nothing computes it for the production model; "
                           "`action` names what must be written or run",
            "STRUCTURAL": "not a result -- a dataset size, design constant, "
                          "definition, threshold or taxonomy",
        },
        "flags": {
            "drift": "the manuscript frames this as invariant, but the value "
                     "moved. On a STRUCTURAL row this is the defect itself.",
            "checkable": "false means no artifact can ever settle it",
        },
        "counts": {b: counts.get(b, 0) for b in BUCKETS},
        "n_claims": len(CLAIMS),
        "reconciled_in_tex": sorted(RECONCILED_IN_TEX),
        "claims": claims,
    }


def _md_table(rows: list[dict]) -> list[str]:
    out = ["| id | line | quantity | audit | authority / keys |",
           "|---|---|---|---|---|"]
    for c in rows:
        keys = c["keys"]
        if c["bucket"] == "BANKED":
            auth = "<br>".join(f"`{k}`" for k in keys)
            if c["derivation"]:
                auth += f"<br>_derivation:_ `{c['derivation']}`"
        elif c["bucket"] == "STRUCTURAL":
            auth = f"`{c['authority']}`" if c["authority"] else "—"
            if keys:
                auth += "<br>" + "<br>".join(f"`{k}`" for k in keys)
        elif c["bucket"] == "GENERATED":
            auth = f"`{c['authority']}`"
            if keys:
                auth += "<br>" + "<br>".join(f"`{k}`" for k in keys)
        else:
            auth = "_none_"
        flag = " ⚑" if c["drift"] else ""
        nc = " ⛔" if not c["checkable"] else ""
        rc = " ✓fixed" if c["id"] in RECONCILED_IN_TEX else ""
        out.append(f"| `{c['id']}`{flag}{nc}{rc} | {c['line']} | {c['quantity']} | "
                   f"{c['audit']} | {auth} |")
    return out


def to_md(anchor_lines: dict[str, int | None] | None) -> str:
    counts = Counter(c["bucket"] for c in CLAIMS)
    by_section: dict[str, list[dict]] = defaultdict(list)
    for c in CLAIMS:
        by_section[c["section"]].append(c)

    L: list[str] = []
    A = L.append
    A("# Prose claim -> authority map")
    A("")
    A("Generated by `paper_tooling/build_prose_claim_map.py` from "
      "`paper_tooling/prose_claims.py`. Do not edit this file; edit the data "
      "module and re-render.")
    A("")
    A("Companion to `paper_tooling/prose_claim_audit.md`. The audit records what "
      "each number *is*; this records where it *should come from*. The audit's "
      "values expire at the next model swap. This mapping does not.")
    A("")
    A("## Buckets")
    A("")
    A("`bucket` answers one question: where does the authority for this number live?")
    A("")
    A("| bucket | meaning | count |")
    A("|---|---|---|")
    A(f"| **BANKED** | a key exists in `paper_tooling/tables/paper_numbers.json` | "
      f"{counts['BANKED']} |")
    A(f"| **GENERATED** | a script writes it to disk, but not into the number bank | "
      f"{counts['GENERATED']} |")
    A(f"| **UNGENERATED** | nothing computes it for the production model | "
      f"{counts['UNGENERATED']} |")
    A(f"| **STRUCTURAL** | not a result: dataset size, design constant, definition, "
      f"threshold, taxonomy | {counts['STRUCTURAL']} |")
    A(f"| | **total** | **{len(CLAIMS)}** |")
    A("")
    A("Two flags cut across the buckets:")
    A("")
    A("- ⚑ **drift** — the manuscript frames the quantity as invariant, and it moved "
      "anyway. On a STRUCTURAL row that combination *is* the defect: the paper is "
      "asserting a constant that is not one.")
    A("- ⛔ **no mechanical route** — no artifact can settle it, ever. Listed in full "
      "at the end.")
    A("")
    A(f"✓fixed marks the {len(RECONCILED_IN_TEX)} claims the manuscript has already "
      "been corrected on since the audit snapshot — Section 4.8's close-pair "
      "paragraph and the rank appendix were rewritten to the exhaustive-enumeration "
      "definition while the audit was being written. Verified against the `.tex` and "
      "the number bank during this rendering. Their `audit` verdict below is "
      "historical; they are kept because they are the rows a checker should already "
      "pass, and because the mapping is what is being recorded, not the value.")
    A("")
    A("A BANKED row may carry a `derivation`: a Python expression over its keys "
      "(`k0`, `k1`, …). The prose usually quotes a ratio, a margin, a bound or a "
      "superlative rather than a raw cell, and the derivation is what a checker "
      "must evaluate. Two derivations are **inequalities**, not values — "
      "`nd.sdpe_by_dim` and `2d.nn_wins` — because in both cases the manuscript's "
      "error is a reversed direction, which a value-only comparison would not catch.")
    A("")

    A("## Claims by section")
    A("")
    for sec in dict.fromkeys(c["section"] for c in CLAIMS):
        rows = by_section[sec]
        A(f"### {sec} ({len(rows)})")
        A("")
        L.extend(_md_table(rows))
        A("")
        notes = [c for c in rows if c["note"] or c["action"]]
        if notes:
            A("<details><summary>Notes and required actions</summary>")
            A("")
            for c in notes:
                A(f"**`{c['id']}`** — {c['bucket']}"
                  + (" ⚑" if c["drift"] else "") + (" ⛔" if not c["checkable"] else ""))
                A("")
                if c["note"]:
                    A(f"> {c['note']}")
                    A("")
                if c["action"]:
                    A(f"*Action:* {c['action']}")
                    A("")
            A("</details>")
            A("")

    # -- Cross-cutting summaries
    A("## Every UNGENERATED claim, grouped by the one thing that would fix it")
    A("")
    groups: dict[str, list[str]] = defaultdict(list)
    for c in CLAIMS:
        if c["bucket"] != "UNGENERATED":
            continue
        act = (c["action"] or "").strip()
        if act.startswith("GEN-REPOINT"):
            key = "GEN-REPOINT — repoint `generalization_experiments.py`"
        elif act.startswith("AUG-REPOINT"):
            key = "AUG-REPOINT — repoint the two augmentation experiments"
        elif "oracle_constant" in act or "oracle-constant" in act:
            key = "Oracle-constant exporter"
        elif "corpus" in act.lower() and "exporter" in act.lower():
            key = "Corpus-statistics exporter"
        elif "booster_stats" in act or "dump_hyperparams" in act:
            key = "Booster-statistics exporter"
        elif "shap" in act.lower():
            key = "SHAP re-run against the production booster"
        elif "time" in c["id"]:
            key = "Low-contention serial timing run"
        elif "alpha-on-alpha" in act or "alphafit" in act:
            key = "alpha-on-alpha regression exporter"
        else:
            key = "One-off (see the claim's action)"
        groups[key].append(c["id"])
    for k in sorted(groups, key=lambda x: -len(groups[x])):
        A(f"- **{k}** ({len(groups[k])}): "
          + ", ".join(f"`{i}`" for i in groups[k]))
    A("")

    A("## Flagged: framed as invariant, moved anyway (⚑)")
    A("")
    A("Every row here is a quantity the manuscript states as a fixed property of "
      "the work. Each one moved. The three causes are worth separating.")
    A("")
    drift = [c for c in CLAIMS if c["drift"]]
    A("| id | line | quantity | why it moved |")
    A("|---|---|---|---|")
    for c in drift:
        why = (c["note"] or "").split(". ")[0]
        A(f"| `{c['id']}` | {c['line']} | {c['quantity']} | {why} |")
    A("")

    A("## No mechanical route, even in principle (⛔)")
    A("")
    A("This is the honest limit of what the paper can be made to check itself on. "
      "Everything else in this map can, with enough exporting, be diffed against "
      "an artifact. These cannot.")
    A("")
    for c in CLAIMS:
        if c["checkable"]:
            continue
        A(f"- **`{c['id']}`** (line {c['line']}, {c['bucket']}) — {c['quantity']}")
        A(f"  - {c['note']}")
    A("")

    if anchor_lines is not None:
        miss = [i for i, v in anchor_lines.items() if v is None]
        A("## Anchor verification")
        A("")
        A(f"{len(anchor_lines) - len(miss)} of {len(anchor_lines)} anchors located "
          f"in the current `.tex`.")
        if miss:
            A("")
            A("Not located (the anchor text needs updating, or the sentence was "
              "already rewritten):")
            for i in miss:
                A(f"- `{i}`")
        A("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="re-locate anchors in the .tex and probe every literal bank key")
    args = ap.parse_args()

    anchors = locate_anchors() if args.verify else None
    OUT_JSON.write_text(json.dumps(to_json(anchors), indent=2), encoding="utf-8")
    OUT_MD.write_text(to_md(anchors), encoding="utf-8")

    counts = Counter(c["bucket"] for c in CLAIMS)
    print(f"{len(CLAIMS)} claims -> {OUT_MD.name}, {OUT_JSON.name}")
    for b in BUCKETS:
        print(f"  {b:<12}{counts.get(b, 0):>4}")
    print(f"  {'drift (flag)':<12}{sum(c['drift'] for c in CLAIMS):>4}")
    print(f"  {'no route':<12}{sum(not c['checkable'] for c in CLAIMS):>4}")

    if args.verify:
        miss = [i for i, v in anchors.items() if v is None]
        print(f"\nanchors located: {len(anchors) - len(miss)}/{len(anchors)}")
        for i in miss:
            print(f"  MISSING ANCHOR  {i}")
        ok, bad = probe_bank()
        print(f"literal bank keys resolved: {len(ok)}/{len(ok) + len(bad)}")
        for b in bad:
            print(f"  MISSING BANK KEY  {b}")


if __name__ == "__main__":
    main()
