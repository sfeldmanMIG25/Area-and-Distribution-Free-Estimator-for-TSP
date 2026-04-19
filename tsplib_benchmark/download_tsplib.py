"""
One-time downloader for TSPLIB95 symmetric TSP instances.

The canonical Heidelberg server (http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
is the authoritative source but is frequently unreachable. This script instead
downloads the same files from the mastqe/tsplib GitHub mirror, which matches
the Heidelberg archive byte-for-byte for the TSP family.

Run once:

    python tsplib_benchmark/download_tsplib.py

Re-running is safe: files that already exist (and are non-empty) are skipped.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

from exclusions import TRIANGLE_INEQ_VIOLATORS

# Mirror base URL. Each instance is stored as {INSTANCE}.tsp at this location.
MIRROR_BASE = "https://raw.githubusercontent.com/mastqe/tsplib/master"

# Destination directory (relative to this file).
DEST = Path(__file__).resolve().parent / "instances"

# Full list of symmetric TSP instances present in TSPLIB95. Triangle-inequality
# violators (see ``exclusions.py``) are filtered out below before download.
_ALL_INSTANCES = [
    "a280", "ali535", "att48", "att532", "bayg29", "bays29", "berlin52",
    "bier127", "brazil58", "brd14051", "brg180", "burma14", "ch130", "ch150",
    "d1291", "d15112", "d1655", "d18512", "d198", "d2103", "d493", "d657",
    "dantzig42", "dsj1000", "eil101", "eil51", "eil76", "fl1400", "fl1577",
    "fl3795", "fl417", "fnl4461", "fri26", "gil262", "gr120", "gr137", "gr17",
    "gr202", "gr21", "gr229", "gr24", "gr431", "gr48", "gr666", "gr96", "hk48",
    "kroA100", "kroA150", "kroA200", "kroB100", "kroB150", "kroB200", "kroC100",
    "kroD100", "kroE100", "lin105", "lin318", "linhp318", "nrw1379", "p654",
    "pa561", "pcb1173", "pcb3038", "pcb442", "pla33810", "pla7397", "pla85900",
    "pr1002", "pr107", "pr124", "pr136", "pr144", "pr152", "pr226", "pr2392",
    "pr264", "pr299", "pr439", "pr76", "rat195", "rat575", "rat783", "rat99",
    "rd100", "rd400", "rl11849", "rl1304", "rl1323", "rl1889", "rl5915",
    "rl5934", "si1032", "si175", "si535", "st70", "swiss42", "ts225", "tsp225",
    "u1060", "u1432", "u159", "u1817", "u2152", "u2319", "u574", "u724",
    "ulysses16", "ulysses22", "usa13509", "vm1084", "vm1748",
]

# Public list with triangle-inequality violators filtered out.
INSTANCES = [n for n in _ALL_INSTANCES if n not in TRIANGLE_INEQ_VIOLATORS]


def download_one(name: str, dest_dir: Path, retries: int = 3) -> bool:
    """Download a single instance. Returns True if the file is present after."""
    target = dest_dir / f"{name}.tsp"
    if target.exists() and target.stat().st_size > 0:
        return True
    url = f"{MIRROR_BASE}/{name}.tsp"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tsplib-downloader"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError("empty response")
            target.write_bytes(data)
            return True
        except Exception as exc:
            if attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            print(f"  [FAIL] {name}: {exc}", file=sys.stderr)
            return False
    return False


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    for i, name in enumerate(INSTANCES, 1):
        status = download_one(name, DEST)
        marker = "OK" if status else "FAIL"
        size = (DEST / f"{name}.tsp").stat().st_size if status else 0
        print(f"[{i:3d}/{len(INSTANCES)}] {marker:4s} {name:12s} ({size} bytes)")
        ok += int(status)
        fail += int(not status)
    print()
    print(f"Downloaded {ok}/{len(INSTANCES)} instances to {DEST}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
