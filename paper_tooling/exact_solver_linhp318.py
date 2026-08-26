"""Concorde on ``linhp318``, the one EUC_2D instance it will not read as shipped.

The finding
-----------
``linhp318.tsp`` carries a ``FIXED_EDGES_SECTION`` forcing edge 1-214. That is
how TSPLIB encodes the *Hamiltonian path* variant of ``lin318`` as a TSP file,
and Concorde refuses it outright::

    ERROR: Not set up for fixed edges
    CCutil_gettsplib failed

So the main campaign records ``linhp318`` as unsupported rather than solved. It
is the only one of the 78 with that section.

Why a second run is worth doing
-------------------------------
Every estimator on the cost axis reads ``info["raw_coords"]`` and estimates the
*plain* TSP tour through ``lin318``'s coordinates. None of them sees the fixed
edge. So the like-for-like exact anchor for this instance is the plain TSP, not
the path variant, and it is obtained by deleting the two-line fixed-edges
section and nothing else.

This also lands on a known defect from the other side. ``frontier_manuscript_bank
.json -> labels.linhp318`` records a stored label of 41345 -- the Hamiltonian
path optimum -- against a tour optimum of 42029 on the same coordinates,
derived elsewhere in this project. If Concorde independently certifies 42029
here, that is a second, solver-issued witness for the repair.

The result is written to its own file and is never folded into the 78-instance
corpus statistics: it is a different instance from the one the corpus ships.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper_tooling"))

from exact_solver_tsplib import run_concorde_once  # noqa: E402
from quiet_box import LoadSampler, observe  # noqa: E402

SRC = ROOT / "tsplib_benchmark" / "instances" / "linhp318.tsp"
SCRATCH = Path(r"C:\Temp_TSP_Scratch") / "linhp318_plain"
OUT = ROOT / "paper_tooling" / "exact_solver_linhp318_plain_tsp.json"

STORED_LABEL = 41345          # TSPLIB's published optimum, for the PATH variant
TOUR_OPT_ON_COORDS = 42029.0  # frontier_manuscript_bank.json -> labels.linhp318


def strip_fixed_edges(text: str) -> tuple[str, list[str]]:
    """Delete only the FIXED_EDGES_SECTION and its terminating ``-1``."""
    lines = text.splitlines()
    out: list[str] = []
    removed: list[str] = []
    in_sec = False
    for ln in lines:
        s = ln.strip()
        if s.upper().startswith("FIXED_EDGES_SECTION"):
            in_sec = True
            removed.append(ln)
            continue
        if in_sec:
            removed.append(ln)
            if s == "-1":
                in_sec = False
            elif re.match(r"^-?\d+(\s+-?\d+)*$", s):
                continue
            else:               # a new section header ends it implicitly
                in_sec = False
                removed.pop()
                out.append(ln)
            continue
        out.append(ln)
    return "\n".join(out) + "\n", removed


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    text = SRC.read_text(encoding="utf-8", errors="replace")
    stripped, removed = strip_fixed_edges(text)
    if "FIXED_EDGES_SECTION" in stripped:
        raise RuntimeError("fixed-edges section survived the strip")
    variant = SCRATCH / "linhp318_plain.tsp"
    variant.write_text(stripped, encoding="ascii")

    n_src = len(text.splitlines())
    n_out = len(stripped.splitlines())
    print(f"removed {removed} ({n_src} -> {n_out} lines)")

    pre = observe(interval_s=1.0, settle_s=1.5)
    print(f"box before run: {pre.busy_pct:.1f}% busy, "
          f"foreign solvers: {pre.foreign_solver_procs or 'none'}")
    with LoadSampler(period_s=5.0) as sampler:
        res = run_concorde_once("linhp318_plain", variant, cap_s=600)
    load = sampler.summary()

    payload = {
        "_what": ("Concorde on lin318's coordinates with the FIXED_EDGES_SECTION "
                  "removed -- the plain TSP that every estimator on the cost axis "
                  "is actually scored against."),
        "_written_by": "paper_tooling/exact_solver_linhp318.py",
        "shipped_file_rejected_by_concorde": {
            "reason": "ERROR: Not set up for fixed edges / CCutil_gettsplib failed",
            "fixed_edge": "1 214",
            "meaning": "TSPLIB encodes the Hamiltonian path variant of lin318 this way",
            "only_such_instance_in_the_78": True,
        },
        "edit_applied": {"removed_lines": removed},
        "status": res.status,
        "wall_s": res.wall_s,
        "concorde_self_time_s": res.solver_self_time_s,
        "certified_optimum": res.cost,
        "bbnodes": res.extra.get("bbnodes"),
        "stored_tsplib_label_for_path_variant": STORED_LABEL,
        "project_tour_optimum_on_these_coordinates": TOUR_OPT_ON_COORDS,
        "agrees_with_project_repair": (
            res.cost is not None and abs(res.cost - TOUR_OPT_ON_COORDS) < 0.5),
        "excess_of_tour_optimum_over_stored_label_pct": (
            100.0 * (res.cost - STORED_LABEL) / STORED_LABEL
            if res.cost is not None else None),
        "box": {"before_pct": pre.busy_pct, "during_pct": load},
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "excluded_from_corpus_statistics": (
            "yes -- this is a modified instance and is reported on its own"),
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if not k.startswith("_")},
                     indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
