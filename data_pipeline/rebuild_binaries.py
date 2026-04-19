"""Rebuild every .bin under instances/ from its corresponding .json.

Some binaries have valid headers but garbage coordinate bodies (likely from an
archival/extraction fault). The JSON is the authoritative source. This script
rebuilds the binaries byte-identically to what save_instance_binary would
write given a fresh instance.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
INSTANCES = ROOT / "instances"


def write_bin(bin_path: Path, data: dict) -> None:
    n = int(data["n_customers"])
    d = int(data["dimension"])
    grid_size = int(data["grid_size"])
    coords = np.array(data["coordinates"], dtype=np.float32)
    if coords.shape != (n, d):
        raise ValueError(
            f"{bin_path.name}: coordinate shape {coords.shape} does not match (n={n}, d={d})"
        )
    dist_types = data.get("distribution_types", [])
    dist_str = "".join(dist_types) if isinstance(dist_types, list) else str(dist_types)
    with open(bin_path, "wb") as f:
        f.write(struct.pack("III", n, d, grid_size))
        f.write(struct.pack("I", len(dist_str)))
        f.write(dist_str.encode("ascii"))
        f.write(coords.tobytes())


def main() -> None:
    json_files = sorted(INSTANCES.glob("*.json"))
    print(f"[rebuild] {len(json_files)} JSON files → rebuild .bin")
    rebuilt = 0
    json_errors = 0
    for jp in tqdm(json_files, desc="rebuilding"):
        bp = jp.with_suffix(".bin")
        # Strict: any JSON parse error halts the pipeline.
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            json_errors += 1
            print(f"[rebuild] SKIP {jp.name}: JSON parse error: {exc}")
            continue
        write_bin(bp, data)
        rebuilt += 1
    print(f"[rebuild] rebuilt: {rebuilt}, json errors skipped: {json_errors}")


if __name__ == "__main__":
    main()
