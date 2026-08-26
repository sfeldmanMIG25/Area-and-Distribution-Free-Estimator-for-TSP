"""Add the ``subsec:related`` label the withdrawal text points at.

The literature-review subsection carried no label. Two sentences added by
``patch_withdraw_unsourced.py`` send the reader there to find the three surveyed
but unscored estimators, so it needs one. Byte-safe literal replacement.
"""

from __future__ import annotations

import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

OLD = "\\subsection{Limitations of the Current Models}\n"
NEW = "\\subsection{Limitations of the Current Models} \\label{subsec:related}\n"


def main() -> int:
    text = TEX.read_text(encoding="utf-8")
    if text.count("\\label{subsec:related}") == 1:
        print("label already present; nothing to do")
        return 0
    if text.count(OLD) != 1:
        print(f"ABORTED: {text.count(OLD)} occurrences of the subsection heading")
        return 1
    TEX.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print("added \\label{subsec:related}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
