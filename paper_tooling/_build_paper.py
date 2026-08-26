"""Build the manuscript with latexmk and report exit, pages, bytes, refs.

A file watcher on this machine grabs ``Area_Free_Main.pdf`` within a second of
pdflatex writing it, so the third pass of a normal latexmk sequence dies with
"I can't write on file".  Building under a scratch jobname sidesteps the
watcher entirely; the finished PDF is copied onto the published name once, at
the end, when nothing else is mid-sequence.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "paper_reference"
SRC = "Area_Free_Main"
JOB = "AFM_build"
CMD = ["latexmk", "-pdf", f"-jobname={JOB}", "-synctex=1",
       "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
       f"{SRC}.tex"]


def main() -> int:
    proc = subprocess.run(CMD, cwd=HERE, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    log = HERE / f"{JOB}.log"
    raw = log.read_text(errors="ignore") if log.exists() else ""
    flat = raw.replace("\n", "")

    print(f"latexmk exit = {proc.returncode}")
    out = re.findall(r"Output written on .*?\((\d+) pages, (\d+) bytes\)", flat)
    if out:
        print(f"pages = {out[-1][0]}  bytes = {out[-1][1]}")
    else:
        print("no 'Output written' line in the log")
        print(proc.stdout[-3000:])

    print("undefined refs :",
          len(re.findall(r"Reference `.*?' on page .*? undefined", raw)))
    print("undefined cites:",
          len(re.findall(r"Citation `.*?' on page .*? undefined", raw)))
    for pat, name in ((r"! LaTeX Error", "LaTeX Error"),
                      (r"! Undefined control sequence", "undefined control seq"),
                      (r"Emergency stop", "emergency stop"),
                      (r"Float\(s\) lost", "floats lost")):
        n = len(re.findall(pat, raw))
        if n:
            print(f"{name}: {n}")
    print("overfull hbox :", len(re.findall(r"Overfull \\hbox", raw)))

    if proc.returncode == 0 and (HERE / f"{JOB}.pdf").exists():
        for attempt in range(6):
            try:
                shutil.copyfile(HERE / f"{JOB}.pdf", HERE / f"{SRC}.pdf")
                print(f"copied {JOB}.pdf -> {SRC}.pdf")
                # The scratch PDF has served its purpose; leaving it behind puts a
                # duplicate of the whole paper in the working tree.
                (HERE / f"{JOB}.pdf").unlink(missing_ok=True)
                break
            except OSError as exc:
                if attempt == 5:
                    print(f"could not refresh {SRC}.pdf ({exc}); {JOB}.pdf is the build")
                else:
                    time.sleep(2)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
