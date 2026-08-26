"""Three anaphor/count repairs found on read-back of the 2026-08-11 copy-edit.

1. "Those three figures" named six figures across three estimators.
2. "the two rival figures" named four figures across two rivals.
3. "the variant orders" sat one noun after two named variants -- the same class
   of loose anaphor F3 was raised about, so it is made explicit here.

The third also moves the anchor of ``conclusion.rank.close10_v4`` in
prose_manifest.py, which this script edits in the same pass so the two never
disagree.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(r"D:\Area-and-Distribution-Free-Estimator-for-TSP")
TEX = ROOT / "paper_reference" / "Area_Free_Main.tex"
MANIFEST = ROOT / "paper_tooling" / "prose_manifest.py"

TEX_EDITS: list[tuple[str, bytes, bytes]] = [
    ("methods-those-figures",
     rb"on both axes. Those three figures are single-seed values",
     rb"on both axes. Those figures are single-seed values"),
    ("intro-rival-figures",
     rb"and the two rival figures are single-seed readings",
     rb"and the rival figures are single-seed readings"),
    ("conclusion-rank-anaphor",
     rb"the extended-block variant on each: the variant orders 82.08\%",
     rb"the extended-block variant on each: that variant orders 82.08\%"),
]

MANIFEST_EDITS: list[tuple[str, bytes, bytes]] = [
    ("close10_v4-anchor",
     rb'        anchor=r"the variant orders {v}\% of that band against",',
     rb'        anchor=r"that variant orders {v}\% of that band against",'),
]


def apply(path: Path, edits: list[tuple[str, bytes, bytes]]) -> tuple[list[str], list[str]]:
    raw = path.read_bytes()
    ok: list[str] = []
    bad: list[str] = []
    for tag, old, new in edits:
        hits = raw.count(old)
        if hits != 1:
            bad.append(f"{tag} ({hits} matches)")
            continue
        raw = raw.replace(old, new, 1)
        ok.append(tag)
    if not bad:
        path.write_bytes(raw)
    return ok, bad


def main() -> int:
    ok1, bad1 = apply(TEX, TEX_EDITS)
    ok2, bad2 = apply(MANIFEST, MANIFEST_EDITS)
    for t in ok1 + ok2:
        print(f"  + {t}")
    for t in bad1 + bad2:
        print(f"  ! {t}")
    print(f"applied {len(ok1) + len(ok2)} / missing {len(bad1) + len(bad2)}")
    return 1 if (bad1 or bad2) else 0


if __name__ == "__main__":
    raise SystemExit(main())
