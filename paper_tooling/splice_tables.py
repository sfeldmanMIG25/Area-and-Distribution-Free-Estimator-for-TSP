"""Splice generated table bodies into Area_Free_Main.tex.

Each results table in the manuscript has the shape

    \\begin{tabular}{...}
    \\toprule
    <header row> \\\\
    \\midrule
    <BODY>
    \\bottomrule
    \\end{tabular}

and this script replaces <BODY> with the matching fragment from
``paper_tooling/tables``. Tables are located by their ``\\label``, so the
manuscript's captions, float placement and column specs are preserved and only
the numbers move. Run it after ``build_paper_tables.py``.

Five of the spliced tables are ``longtable`` rather than ``tabular``, because
they run long and were previously being scaled by ``\\resizebox`` instead of
paginated. Their shape is

    \\begin{longtable}{...}
    \\caption{...}\\label{...}\\\\
    \\toprule <header> \\midrule \\endfirsthead
    <continuation banner> \\toprule <header> \\midrule \\endhead
    <BODY>
    \\bottomrule
    \\end{longtable}

so the header appears twice before the body and the "first \\midrule after the
first \\toprule" rule would cut into the repeated header instead of the rows.
``body_span`` keys off ``\\endhead`` for those.

``--only LABEL[,LABEL...]`` restricts the run to the named tables, so a
structural change to one table does not rewrite the bodies of the others.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper_reference" / "Area_Free_Main.tex"
FRAG = Path(__file__).resolve().parent / "tables"

# label -> fragment file. Every table is a single manuscript environment, so
# one fragment goes to one \label and nothing needs splitting. The 2D per-size
# table used to be the exception: it was printed as "part 1 of 2" / "part 2 of
# 2" across two \resizebox floats and this script cut the fragment in half to
# feed them. It is one longtable now, so the split is gone.
SIMPLE: dict[str, str] = {
    "tab:tsplib_by_size": "table_tsplib_by_size.tex",
    "tab:tsplib_nonEuc": "table_tsplib_nonEuc.tex",
    # Compact Section 4 body tables, written by build_paper_tables.write_tex_compact.
    "tab:results_nd": "table_results_nd.tex",
    "tab:results_2d": "table_results_2d.tex",
}


def body_span(text: str, label: str) -> tuple[int, int]:
    """Return the character span of the body rows of the table with ``label``."""
    m = re.search(re.escape("\\label{" + label + "}"), text)
    if m is None:
        raise KeyError(f"label {label!r} not found in {TEX.name}")
    end = text.find(r"\bottomrule", m.end())
    if end == -1:
        raise ValueError(f"no \\bottomrule after {label!r}")
    head = text.find(r"\endhead", m.end())
    if head != -1 and head < end:
        # longtable: the body is everything after the repeated-header block.
        return head + len(r"\endhead"), end
    # The body starts after the LAST \midrule that precedes \bottomrule but
    # follows the header; the header \midrule is the first one after \toprule.
    top = text.find(r"\toprule", m.end())
    if top == -1 or top > end:
        raise ValueError(f"no \\toprule after {label!r}")
    start = text.find(r"\midrule", top)
    if start == -1 or start > end:
        raise ValueError(f"no header \\midrule after {label!r}")
    return start + len(r"\midrule"), end


def splice(text: str, label: str, body: str) -> str:
    lo, hi = body_span(text, label)
    return text[:lo] + "\n" + body.strip("\n") + "\n" + text[hi:]


def main() -> int:
    ap = argparse.ArgumentParser(description="Splice generated bodies into the manuscript.")
    ap.add_argument("--only", default="",
                    help="comma-separated \\label values; default is every table")
    args = ap.parse_args()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = wanted - set(SIMPLE)
    if unknown:
        print("Unknown label(s): " + ", ".join(sorted(unknown)))
        return 2

    text = TEX.read_text(encoding="utf-8")
    original = text
    applied: list[str] = []

    for label, fname in SIMPLE.items():
        if wanted and label not in wanted:
            continue
        path = FRAG / fname
        if not path.exists():
            print(f"  SKIP {label}: {fname} missing")
            continue
        try:
            text = splice(text, label, path.read_text(encoding="utf-8"))
        except KeyError as exc:
            print(f"  SKIP {label}: {exc}")
            continue
        applied.append(label)

    if text == original:
        print("No changes.")
        return 1
    TEX.write_text(text, encoding="utf-8")
    print("Spliced: " + ", ".join(applied))
    return 0


if __name__ == "__main__":
    sys.exit(main())
