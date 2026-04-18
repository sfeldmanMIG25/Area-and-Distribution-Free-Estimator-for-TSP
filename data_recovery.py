"""Data integrity recovery script.

Classifies every (bin, json) pair in ``instances/`` into one of:

1. BOTH_OK        : header valid, JSON parses  -> nothing to do.
2. BIN_CORRUPT    : bin header invalid, JSON valid  -> delete the bin,
                    fall through to JSON at read time (already supported by
                    :func:`feature_creator_v3.load_instance_data`).
3. BOTH_CORRUPT_SEED_RECOVERABLE : bin invalid, JSON unparseable, but the
                    ``"generation_seed":`` byte pattern is still readable
                    inside the corrupt JSON. Regenerate the instance exactly
                    from that seed (verified byte-identical on known-good
                    instances), rewrite JSON + bin cleanly.
4. BOTH_CORRUPT_TRULY_LOST : no seed recoverable. Delete instance and
                    solution; these rows are pruned from the training grid.

Strict: no silent fallbacks. Every corrupt file is either repaired
deterministically or dropped with a logged reason.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Dataset_Generator import DISTRIBUTION_MAP_1D, make_unique_numba, save_instance_binary

INSTANCES = ROOT / "instances"
SOLUTIONS = ROOT / "solutions"

SEED_PATTERN = re.compile(rb'"generation_seed"\s*:\s*(\d+)')
NAME_PATTERN = re.compile(r"^N(\d+)_D(\d+)_G(\d+)_([a-z]+)_(\d+)$")


def classify_bin(bin_path: Path) -> str:
    """'OK' iff the binary passes every check load_instance_data performs.

    load_instance_data validates: 12-byte header; n in [3, 100000]; d in
    [1, 200]; dist_len <= 1024; file_size >= 12 + 4 + dist_len + n*d*4.
    Any failure short-circuits the bin and forces the JSON fallback (or an
    explicit repair in this recovery tool).
    """
    with open(bin_path, "rb") as f:
        hdr = f.read(12)
        if len(hdr) != 12:
            return "SHORT_HEADER"
        n, d, _ = struct.unpack("III", hdr)
        if not (3 <= n <= 100000 and 1 <= d <= 200):
            return f"BAD_HDR_n={n}_d={d}"
        dl_bytes = f.read(4)
        if len(dl_bytes) != 4:
            return "SHORT_DIST_LEN"
        dist_len = struct.unpack("I", dl_bytes)[0]
        if dist_len > 1024:
            return f"BAD_DIST_LEN_{dist_len}"
    fs = bin_path.stat().st_size
    need = 12 + 4 + dist_len + n * d * 4
    if fs < need:
        return f"TRUNCATED_{fs}<{need}"
    return "OK"


def json_is_valid(json_path: Path) -> bool:
    with open(json_path, "rb") as f:
        data = f.read()
    # Try utf-8 JSON decode; binary contamination => fail.
    try:
        json.loads(data.decode("utf-8"))
        return True
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False


def recover_seed(json_path: Path) -> int | None:
    with open(json_path, "rb") as f:
        data = f.read()
    m = SEED_PATTERN.search(data)
    return int(m.group(1)) if m else None


def regenerate_from_seed(name: str, seed: int) -> dict:
    m = NAME_PATTERN.match(name)
    if m is None:
        raise ValueError(f"Filename {name!r} does not match expected pattern")
    n, d, grid_size, dist_str, seq_j = m.groups()
    n = int(n); d = int(d); grid_size = int(grid_size); seq_j = int(seq_j)
    dist_letters = list(dist_str)

    coords = np.zeros((n, d), dtype=np.float32)
    base = None
    for j in range(d):
        letter = dist_letters[j]
        func = DISTRIBUTION_MAP_1D[letter]
        if letter == "k" and base is not None:
            coords[:, j] = func(n, seed + j, grid_size, base=base)
        else:
            coords[:, j] = func(n, seed + j, grid_size)
        if base is None and letter != "k":
            base = coords[:, j]
    coords = make_unique_numba(coords, float(grid_size), seed)

    return {
        "instance_name": name,
        "n_customers": n,
        "dimension": d,
        "grid_size": grid_size,
        "distribution_types": dist_letters,
        "generation_seed": seed,
        "coordinates": coords.tolist(),
    }


def main() -> None:
    bins = sorted(INSTANCES.glob("*.bin"))
    print(f"[recovery] scanning {len(bins)} binary files")

    ok = 0
    bin_corrupt_json_ok = 0
    both_corrupt_recovered = 0
    both_corrupt_lost = []

    for bp in bins:
        name = bp.stem
        jp = bp.with_suffix(".json")
        bin_status = classify_bin(bp)

        if bin_status == "OK":
            ok += 1
            continue

        # Binary is broken.
        if jp.exists() and json_is_valid(jp):
            # JSON is the authoritative source — delete the bad binary.
            bp.unlink()
            bin_corrupt_json_ok += 1
            continue

        # Both corrupt — attempt seed recovery.
        seed = recover_seed(jp) if jp.exists() else None
        if seed is None:
            both_corrupt_lost.append(name)
            continue

        regen = regenerate_from_seed(name, seed)
        if jp.exists():
            jp.unlink()
        if bp.exists():
            bp.unlink()
        # save_instance_binary writes both the .bin and the .json
        save_instance_binary(str(jp), regen)
        both_corrupt_recovered += 1

    print(f"[recovery] bin OK                    : {ok}")
    print(f"[recovery] bin corrupt, json valid   : {bin_corrupt_json_ok} (deleted bin)")
    print(f"[recovery] both corrupt, recovered   : {both_corrupt_recovered} (regen from seed)")
    print(f"[recovery] both corrupt, truly lost  : {len(both_corrupt_lost)}")

    if both_corrupt_lost:
        # Drop both instance and solution for truly-lost rows.
        lost_log = ROOT / "data_recovery_lost.txt"
        with open(lost_log, "w", encoding="utf-8") as f:
            for name in both_corrupt_lost:
                f.write(name + "\n")
                for p in (
                    INSTANCES / f"{name}.json",
                    INSTANCES / f"{name}.bin",
                    SOLUTIONS / f"{name}.sol.json",
                ):
                    if p.exists():
                        p.unlink()
        print(f"[recovery] dropped-instance log: {lost_log}")


if __name__ == "__main__":
    main()
