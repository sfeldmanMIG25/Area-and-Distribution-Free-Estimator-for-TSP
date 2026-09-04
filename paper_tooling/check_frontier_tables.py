"""Verify the two hand-typed frontier tables against the frontier bank.

``build_paper_tables.py --check`` covers only the six generated tables; it does
not know ``tab:frontier_tsplib`` or ``tab:frontier_nd``, which is deliberate --
those bodies must not enter its 1,910-cell gate.  That leaves them unchecked,
and a hand-typed ladder is precisely the kind of table that rots.  This module
closes that hole: it re-derives every numeric cell of both tables from
``frontier_manuscript_bank.json`` and diffs against the typeset value at the
precision the cell prints.

Exit code is non-zero on any disagreement.

Run::

    python paper_tooling/check_frontier_tables.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEX = ROOT / "paper_reference" / "Area_Free_Main.tex"
BANK = HERE / "frontier_manuscript_bank.json"

LADDER = ["0", "10", "25", "50", "100", "200", "500"]
TSPLIB_BUCKETS = ["n in [51,150]", "n in [151,400]", "n > 400"]
ND_GROUPS = ["d in {2,3}", "d in [4,10]", "d in [15,50]", "d = 100"]


def expected_tsplib(bank: dict) -> list[tuple[str, float, int]]:
    """(cell name, generated value, decimals the manuscript prints)."""
    t, cal = bank["tsplib"], bank["tsplib_calibrated_bound"]
    out: list[tuple[str, float, int]] = []
    for b in TSPLIB_BUCKETS:
        s = t[b]
        out.append((f"{b}|gart2|ms", s["gart2_ms"], 2))
        out.append((f"{b}|gart2|mape", s["gart2_mape_pct"], 2))
        for k in LADDER:
            out.append((f"{b}|k{k}|ms", s["bound_ms_by_k"][k], 2))
            out.append((f"{b}|k{k}|x", s["bound_x_gart2_by_k"][k], 2))
            out.append((f"{b}|k{k}|raw", s["bound_mape_pct_by_k"][k], 2))
            out.append((f"{b}|k{k}|cal", cal[b]["mape_pct_by_k"][k], 2))
    return out


def expected_nd(bank: dict) -> list[tuple[str, float, int]]:
    g = bank["nd"]["by_group"]
    out: list[tuple[str, float, int]] = []
    for name in ND_GROUPS:
        s = g[name]
        out.append((f"{name}|gart2|ms", s["GART_2.0_ms"], 2))
        out.append((f"{name}|gart2|mape", s["GART_2.0_MAPE_pct"], 3))
        for k in LADDER:
            out.append((f"{name}|k{k}|ms", s[f"ms_k{k}"], 2))
            out.append((f"{name}|k{k}|x", s[f"x_k{k}"], 2))
            out.append((f"{name}|k{k}|mape", s[f"MAPE_k{k}"], 3))
    return out


def _sig3(x: float) -> str:
    """Three significant figures, trailing zeros kept (author, 2026-09-03).

    Every ladder cell prints at this precision; the ``decimals`` field of the
    expected tuples is retained for provenance but no longer drives the check.
    """
    if abs(x) >= 1000:
        return f"{int(round(x)):,}".replace(",", "{,}")
    return f"{x:#.3g}".rstrip(".")


def _cells(body: str, first_col: str) -> list[str]:
    """Numeric cells of the row whose first column is ``first_col``."""
    for line in body.splitlines():
        cells = [c.strip() for c in line.split("&")]
        if cells and cells[0].replace("$", "") == first_col.replace("$", ""):
            return [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).replace(r"\\", "").strip()
                    for c in cells[1:]]
    return []


def _table_body(tex: str, label: str) -> str:
    i = tex.index(rf"\label{{{label}}}")
    j = tex.index(r"\bottomrule", i)
    return tex[i:j]


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    bad: list[str] = []
    checked = 0

    # -- TSPLIB: 3 buckets x 4 columns, GART 2.0 row spans its two MAPE cells.
    body = _table_body(tex, "tab:frontier_tsplib")
    printed: dict[str, str] = {}
    # The GART 2.0 row spans its two MAPE columns with \multicolumn, so it has
    # three &-cells per bucket where a ladder row has four.
    g = _cells(body, "GART 2.0")
    for i, b in enumerate(TSPLIB_BUCKETS):
        printed[f"{b}|gart2|ms"] = g[3 * i]
        printed[f"{b}|gart2|mape"] = re.sub(
            r"\\multicolumn\{2\}\{c\}\{([^}]*)\}", r"\1", g[3 * i + 2])
    for k in LADDER:
        row = _cells(body, f"$k={k}$")
        for i, b in enumerate(TSPLIB_BUCKETS):
            printed[f"{b}|k{k}|ms"] = row[4 * i]
            printed[f"{b}|k{k}|x"] = row[4 * i + 1]
            printed[f"{b}|k{k}|raw"] = row[4 * i + 2]
            printed[f"{b}|k{k}|cal"] = row[4 * i + 3]

    # -- ND: 4 groups x 3 columns.
    body_nd = _table_body(tex, "tab:frontier_nd")
    g = _cells(body_nd, "GART 2.0")
    for i, name in enumerate(ND_GROUPS):
        printed[f"{name}|gart2|ms"] = g[3 * i]
        printed[f"{name}|gart2|mape"] = g[3 * i + 2]
    for k in LADDER:
        row = _cells(body_nd, f"$k={k}$")
        for i, name in enumerate(ND_GROUPS):
            printed[f"{name}|k{k}|ms"] = row[3 * i]
            printed[f"{name}|k{k}|x"] = row[3 * i + 1]
            printed[f"{name}|k{k}|mape"] = row[3 * i + 2]

    for cell, value, _dp in expected_tsplib(bank) + expected_nd(bank):
        checked += 1
        got = printed.get(cell)
        want = _sig3(value)
        if got is None:
            bad.append(f"{cell}: not found in the typeset table")
        elif got != want:
            bad.append(f"{cell}: manuscript {got!r} vs generated {want!r}")

    # The manuscript states this count in prose, next to the count that
    # build_paper_tables.py --check banks for the generated tables. Bank it the
    # same way, so neither figure can go stale silently when the roster or a
    # ladder budget changes.
    numbers_path = HERE / "tables" / "paper_numbers.json"
    if numbers_path.exists():
        banked = json.loads(numbers_path.read_text(encoding="utf-8"))
        if banked.get("frontier_table_cells") != checked:
            banked["frontier_table_cells"] = checked
            numbers_path.write_text(json.dumps(banked, indent=2, sort_keys=True),
                                    encoding="utf-8")

    print("=" * 74)
    print(f"frontier tables vs frontier_manuscript_bank.json: {checked} cells")
    print("=" * 74)
    for line in bad:
        print(f"  MISMATCH {line}")
    print(f"{'FAIL' if bad else 'PASS'}: {len(bad)} discrepancies")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
