"""Process-level tests for the prose checker's baseline/gating layer.

Each test drives ``check_prose_numbers.py`` as a subprocess inside a disposable
sandbox -- a copy of the tooling, a copy of the manuscript, a copy of the number
bank -- and asserts a property of the *gate*, never of an internal function.
That is deliberate: every defect these cover was reachable through the command
line with no import in sight, and three of the four were invisible to anything
that only read the report.

Run against the shipped tooling::

    python paper_tooling/test_prose_gate.py

Run against some other copy of it -- the point being that each test fails
against the count-based, ``return 0`` predecessor and passes against the
occurrence-addressed one::

    python paper_tooling/test_prose_gate.py --tooling <dir containing paper_tooling/>

Sandboxes are written under ``--work`` (default: a temp directory) and kept when
``--keep`` is passed, so a failing case can be re-run by hand.

The properties, in the order the redesign ranks them
----------------------------------------------------
P1  a MISMATCHED claim blocks, and no baseline operation makes it stop.
P2  a newly unverified number is reported and cannot be absorbed by a
    coincidentally-matching vacated slot.
P3  removing a claim from the manifest is visible and attributable.
P4  ranking and ``--limit`` never page a NEW or MISMATCHED finding out.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PY = sys.executable

TOOLING_FILES = ("check_prose_numbers.py", "prose_baseline.py", "prose_manifest.py",
                 "model_registry.py")
# Copied so a sandbox resolves the same generated artifacts the real run does.
DATA_FILES = (Path("paper_tooling/tables/paper_numbers.json"),
              Path("lgbm_model_v3/gart2_final.json"))
TEX = Path("paper_reference/Area_Free_Main.tex")

# A value that occurs nowhere in the manuscript, so a sandbox edit that
# introduces it cannot collide with a real number.
FRESH = "7.77"
FRESH2 = "8.88"
SENTENCE_A = rf"The withheld-fold refit reports {FRESH}\% MAPE."
SENTENCE_B = rf"A late refit reaches {FRESH}\% MAPE on the withheld fold."

SUMMARY = re.compile(r"^(PASS|FAIL): (\d+) blocking finding", re.M)


class Sandbox:
    """One disposable repository root: tooling, manuscript, number bank."""

    def __init__(self, root: Path, tooling: Path) -> None:
        self.root = root
        (root / "paper_tooling" / "tables").mkdir(parents=True, exist_ok=True)
        (root / "paper_reference").mkdir(parents=True, exist_ok=True)
        (root / "lgbm_model_v3").mkdir(parents=True, exist_ok=True)
        for name in TOOLING_FILES:
            shutil.copy2(tooling / "paper_tooling" / name, root / "paper_tooling" / name)
        for rel in DATA_FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, root / rel)
        shutil.copy2(REPO / TEX, root / TEX)

    # -- the manuscript ----------------------------------------------------
    @property
    def tex(self) -> Path:
        return self.root / TEX

    @property
    def baseline(self) -> Path:
        return self.root / "paper_tooling" / "prose_baseline.json"

    def add_section(self, body: str, label: str = "sec:scratch") -> None:
        """Append a real prose section, so the numeral in it is scanned."""
        text = self.tex.read_text(encoding="utf-8")
        block = f"\n\\section{{Scratch}}\\label{{{label}}}\n{body}\n"
        text = text.replace("\\end{document}", block + "\\end{document}", 1)
        self.tex.write_text(text, encoding="utf-8")

    def reword(self, old: str, new: str) -> None:
        text = self.tex.read_text(encoding="utf-8")
        if old not in text:
            raise AssertionError(f"sandbox manuscript does not contain {old!r}")
        self.tex.write_text(text.replace(old, new, 1), encoding="utf-8")

    # -- the manifest ------------------------------------------------------
    def drop_claims(self, ids: list[str]) -> None:
        """Delete claims from the manifest the way a person would: they vanish."""
        path = self.root / "paper_tooling" / "prose_manifest.py"
        drop = "{" + ", ".join(repr(i) for i in ids) + "}"
        path.write_text(path.read_text(encoding="utf-8")
                        + f"\n\n_DROPPED = {drop}\n"
                          f"CLAIMS = [c for c in CLAIMS if c.id not in _DROPPED]\n",
                        encoding="utf-8")

    def keep_only_states(self, states: set[str], run_json: dict) -> None:
        keep = [c["id"] for c in run_json["claims"] if c["state"] in states]
        path = self.root / "paper_tooling" / "prose_manifest.py"
        path.write_text(path.read_text(encoding="utf-8")
                        + f"\n\n_KEEP = {set(keep)!r}\n"
                          f"CLAIMS = [c for c in CLAIMS if c.id in _KEEP]\n",
                        encoding="utf-8")

    # -- running it --------------------------------------------------------
    def run(self, *args: str, json_out: bool = False) -> tuple[int, str, dict | None]:
        out_path = self.root / "run.json"
        argv = [PY, str(self.root / "paper_tooling" / "check_prose_numbers.py"),
                "--tex", str(self.tex), "--baseline", str(self.baseline), *args]
        if json_out:
            argv += ["--json", str(out_path)]
        proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(self.root))
        blob = None
        if json_out and out_path.exists():
            blob = json.loads(out_path.read_text(encoding="utf-8"))
            out_path.unlink()
        return proc.returncode, proc.stdout + proc.stderr, blob

    def seed(self, reason: str = "test seed") -> tuple[int, str]:
        code, out, _ = self.run("--update-baseline", "--reason", reason)
        # A schema-1 predecessor writes directly; the successor needs the record
        # to exist in its own schema first. Either way the sandbox ends with a
        # record that covers exactly the current manuscript.
        if "--migrate-baseline" in out:
            self.run("--migrate-baseline", "--reason", "test migrate")
            code, out, _ = self.run("--update-baseline", "--reason", reason)
        return code, out

    def blocking(self) -> int:
        _code, out, _ = self.run()
        m = SUMMARY.search(out)
        if not m:
            raise AssertionError(f"no summary line in:\n{out[-2000:]}")
        return int(m.group(2))


# -- the tests ---------------------------------------------------------------
def t1_update_baseline_exit_code(sb: Sandbox) -> None:
    """P1: recording a baseline cannot make an outstanding MISMATCHED pass."""
    sb.seed()
    _code, _out, blob = sb.run(json_out=True)
    mismatched = [c["id"] for c in blob["claims"] if c["state"] == "MISMATCHED"]
    assert mismatched, "fixture needs at least one MISMATCHED claim"
    code, out, _ = sb.run("--update-baseline", "--reason", "routine re-record")
    assert code != 0, (
        f"--update-baseline exited {code} with {len(mismatched)} MISMATCHED "
        f"outstanding; a CI step reading the exit code is permanently green.\n"
        f"{out[-1500:]}")


def t2_vacated_slot_cannot_absorb(sb: Sandbox) -> None:
    """P2: a new sentence may not take over a slot its value happens to match."""
    sb.add_section(SENTENCE_A)
    sb.seed()
    before = sb.blocking()
    sb.reword(SENTENCE_A, SENTENCE_B)          # same value, same section, new sentence
    code, out, blob = sb.run(json_out=True)
    values = [n["value"] for n in blob["new_unregistered"]]
    assert FRESH in values, (
        f"the replacement sentence asserting {FRESH}% was absorbed by the slot the "
        f"deleted sentence vacated: it is not in NEW.\n"
        f"  blocking before {before}, after {sb_blocking_from(out)}\n"
        f"  NEW values: {values}")
    assert code != 0, "a new unverified number must block"


def t3_withdrawing_a_claim_is_visible(sb: Sandbox) -> None:
    """P3: deleting a MISMATCHED claim may not reduce the blocking count."""
    sb.seed()
    _code, _out, blob = sb.run(json_out=True)
    mismatched = [c["id"] for c in blob["claims"] if c["state"] == "MISMATCHED"][:3]
    assert len(mismatched) == 3, "fixture needs at least three MISMATCHED claims"
    before = sb.blocking()
    sb.drop_claims(mismatched)
    _code, out, _ = sb.run()
    after = sb_blocking_from(out)
    assert after > before, (
        f"dropping {len(mismatched)} MISMATCHED claims from the manifest left the "
        f"blocking count at {after} (was {before}): the deletion is exit-code "
        f"neutral.\n  dropped: {mismatched}")
    for cid in mismatched:
        assert cid in out, f"withdrawn claim {cid} is not named in the report"


def t4_absorption_is_permanent(sb: Sandbox) -> None:
    """P2/P3: an absorbed finding stays named after later routine re-records."""
    sb.seed()
    sb.add_section(rf"The recalibrated fold reports {FRESH2}\% MAPE.", "sec:scratch2")
    code, out, blob = sb.run(json_out=True)
    assert FRESH2 in [n["value"] for n in blob["new_unregistered"]], out[-1500:]
    sb.run("--update-baseline", "--reason", f"absorb the {FRESH2} finding")
    sb.run("--update-baseline", "--reason", "routine re-record, nothing changed")
    sb.run("--update-baseline", "--reason", "another routine re-record")
    _code, out, _ = sb.run()
    assert FRESH2 in out, (
        f"the absorbed {FRESH2} finding is no longer named anywhere in the report "
        f"after two further routine re-records; the only trace left is a history "
        f"entry no run prints.")


def t5_new_findings_are_not_pageable(sb: Sandbox) -> None:
    """P4: --limit/--offset may not hide a newly unverified number."""
    sb.add_section(SENTENCE_A)
    sb.seed()
    sb.reword(SENTENCE_A, SENTENCE_B)
    _code, out, _ = sb.run("--limit", "1", "--offset", "100000")
    assert FRESH in out, (
        f"the {FRESH}% finding is invisible at --limit 1 --offset 100000: it was "
        f"ranked into the pageable backlog instead of being reported as new.")


def t6_withdrawal_and_absorption_survive_re_recording(sb: Sandbox) -> None:
    """P1/P3: the proven-wrong numbers stay attributable after the cover-up."""
    sb.seed()
    _code, _out, blob = sb.run(json_out=True)
    mismatched = [(c["id"], c["printed"]) for c in blob["claims"]
                  if c["state"] == "MISMATCHED"][:3]
    sb.drop_claims([cid for cid, _v in mismatched])
    sb.run("--update-baseline", "--reason", "absorb the released numerals")
    sb.run("--update-baseline", "--reason", "routine re-record")
    _code, out, _ = sb.run()
    missing = [cid for cid, _v in mismatched if cid not in out]
    assert not missing, (
        f"after one absorb and one routine re-record, {len(missing)} withdrawn "
        f"MISMATCHED claim(s) are named nowhere in the report: {missing}. Three "
        f"proven-wrong headline numbers are now anonymous backlog rows.")


def t7_clean_update_still_passes(sb: Sandbox) -> None:
    """Guard: the exit-code fix must not make every baseline write fail."""
    _code, _out, blob = sb.run(json_out=True)
    sb.keep_only_states({"MATCHED"}, blob)
    sb.seed("seed with only MATCHED claims registered")
    code, out, _ = sb.run("--update-baseline", "--reason", "routine, nothing pending")
    assert code == 0, (
        f"--update-baseline exited {code} with nothing outstanding; the gate has "
        f"become one nobody can satisfy.\n{out[-1500:]}")


def sb_blocking_from(out: str) -> int:
    m = SUMMARY.search(out)
    if not m:
        raise AssertionError(f"no summary line in:\n{out[-2000:]}")
    return int(m.group(2))


TESTS = [
    ("P1 update-baseline exit code", t1_update_baseline_exit_code),
    ("P2 vacated slot cannot absorb", t2_vacated_slot_cannot_absorb),
    ("P3 withdrawing a claim is visible", t3_withdrawing_a_claim_is_visible),
    ("P2 absorption is permanent", t4_absorption_is_permanent),
    ("P4 new findings are not pageable", t5_new_findings_are_not_pageable),
    ("P1 withdrawal survives re-recording", t6_withdrawal_and_absorption_survive_re_recording),
    ("-- clean update still passes", t7_clean_update_still_passes),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tooling", type=Path, default=REPO, metavar="DIR",
                    help="directory containing paper_tooling/ (default: this repo)")
    ap.add_argument("--work", type=Path, default=None,
                    help="where to build sandboxes (default: a temp directory)")
    ap.add_argument("--keep", action="store_true", help="do not delete sandboxes")
    ap.add_argument("--only", default="", help="substring filter on test names")
    args = ap.parse_args()

    work = args.work or Path(tempfile.mkdtemp(prefix="prose_gate_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"tooling under test : {args.tooling}")
    print(f"sandboxes          : {work}\n")

    failed = 0
    for i, (name, fn) in enumerate(TESTS, 1):
        if args.only and args.only not in name:
            continue
        root = work / f"case{i:02d}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        sb = Sandbox(root, args.tooling)
        try:
            fn(sb)
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}\n        {str(exc).replace(chr(10), chr(10) + '        ')}")
        except Exception as exc:  # a crash is a failure, and its text is the evidence
            failed += 1
            print(f"  ERROR {name}\n        {type(exc).__name__}: {exc}")
        finally:
            if not args.keep:
                shutil.rmtree(root, ignore_errors=True)

    print(f"\n{'FAIL' if failed else 'PASS'}: {failed} of {len(TESTS)} propert(y/ies) "
          f"violated")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
