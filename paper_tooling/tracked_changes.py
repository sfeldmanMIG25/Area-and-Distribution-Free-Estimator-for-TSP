"""Read a manuscript that carries ``changes``-package markup as either of its two documents.

``\\added[opts]{new}``, ``\\deleted[opts]{old}`` and ``\\replaced[opts]{new}{old}`` mark a
proposed edit in place.  ``final_view`` is the document with every edit accepted,
``original_view`` the document with every edit rejected.  The prose-number gate reads
``final_view``, so a number inside struck-out text is never checked and a number inside
proposed text always is.
"""
from __future__ import annotations

import re

_CMD = re.compile(r"\\(added|deleted|replaced)\s*(\[[^\]]*\])?\s*\{")


def _brace_arg(text: str, open_pos: int) -> tuple[str, int]:
    """Return (content, index after the closing brace) for the group opening at open_pos."""
    assert text[open_pos] == "{"
    depth, i = 0, open_pos
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
                return text[open_pos + 1:i], i + 1
        i += 1
    raise ValueError(f"unbalanced brace group at offset {open_pos}")


def _view(text: str, accept: bool) -> str:
    out, pos = [], 0
    for m in _CMD.finditer(text):
        if m.start() < pos:          # inside an argument already consumed
            continue
        out.append(text[pos:m.start()])
        first, end = _brace_arg(text, m.end() - 1)
        kind = m.group(1)
        if kind == "replaced":
            j = end
            while j < len(text) and text[j] in " \t\n":
                j += 1
            second, end = _brace_arg(text, j)
            out.append(_view(first if accept else second, accept))
        elif kind == "added":
            out.append(_view(first, accept) if accept else "")
        else:  # deleted
            out.append("" if accept else _view(first, accept))
        pos = end
    out.append(text[pos:])
    return "".join(out)


def final_view(text: str) -> str:
    """Every proposed edit accepted."""
    return _view(text, accept=True)


def original_view(text: str) -> str:
    """Every proposed edit rejected."""
    return _view(text, accept=False)


def has_markup(text: str) -> bool:
    body = text[text.find(r"\begin{document}"):] if r"\begin{document}" in text else text
    return bool(_CMD.search(body))
