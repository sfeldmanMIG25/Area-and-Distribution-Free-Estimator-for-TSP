"""
TSPLIB95 instance parser.

Parses the symmetric TSP file format described in TSPLIB95 (Reinelt, 1991) and
returns a structured representation suitable for downstream use by the GART 3.0
estimator.

Supported EDGE_WEIGHT_TYPE values
---------------------------------
EUC_2D   : 2D Euclidean, rounded to nearest integer (nint).
CEIL_2D  : 2D Euclidean, rounded up (ceil).
ATT      : Pseudo-Euclidean used by the att* instances.
GEO      : Geographical. Coordinates are latitude/longitude in DDD.MM format;
           distances are great-circle distances on a sphere of radius 6378.388.
EXPLICIT : A full distance matrix is given directly. The matrix layout is
           controlled by EDGE_WEIGHT_FORMAT. Supported layouts:
           FULL_MATRIX, LOWER_DIAG_ROW, UPPER_DIAG_ROW, LOWER_ROW, UPPER_ROW.

The distance formulas are implemented as specified in the TSPLIB95 documentation
so that computed tour lengths match the published optima in TSPLIB's solutions
file.

Returned structure
------------------
`parse_tsplib_file()` returns a dict with the following keys:

    name            : str     TSPLIB NAME field.
    n               : int     number of nodes.
    edge_weight_type: str     one of EUC_2D, CEIL_2D, ATT, GEO, EXPLICIT.
    edge_weight_format: str   EDGE_WEIGHT_FORMAT (only set for EXPLICIT).
    raw_coords      : np.ndarray or None
                              Raw coordinates exactly as given in the file. For
                              EUC_2D / CEIL_2D / ATT this is the native 2D
                              coordinate array. For GEO this is the (lat,lon)
                              array in DDD.MM form. For EXPLICIT this is None.
    distance_matrix : np.ndarray (n, n) float64
                              Symmetric distance matrix computed with the
                              TSPLIB95 rules for the given edge-weight type.
                              Entries are integers stored as float64.
    is_native_euclidean : bool
                              True if raw_coords lives in a real Euclidean space
                              whose pairwise distances match the TSPLIB edge
                              weights (up to integer rounding). For EUC_2D and
                              CEIL_2D we set this True. For ATT (pseudo-Euclidean
                              with a sqrt(10) divisor), GEO (spherical), and
                              EXPLICIT we set this False — an MDS embedding on
                              the TSPLIB distance matrix is required so that the
                              estimator's internal Euclidean MST matches the
                              distance metric the published optimum was scored
                              against.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Low-level header parsing
# ---------------------------------------------------------------------------

_HEADER_KEYS = {
    "NAME",
    "TYPE",
    "COMMENT",
    "DIMENSION",
    "EDGE_WEIGHT_TYPE",
    "EDGE_WEIGHT_FORMAT",
    "NODE_COORD_TYPE",
    "DISPLAY_DATA_TYPE",
    "CAPACITY",
}

_SECTION_KEYS = {
    "NODE_COORD_SECTION",
    "EDGE_WEIGHT_SECTION",
    "DISPLAY_DATA_SECTION",
    "DEPOT_SECTION",
    "EOF",
}


def _read_header(lines):
    """Parse the header of a TSPLIB file.

    Returns (header_dict, body_start_index, current_section). ``body_start_index``
    points at the first line *after* a section keyword was encountered. The
    section keyword itself is returned in ``current_section``.
    """
    header: Dict[str, str] = {}
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        # Section keywords end the header
        token = line.split(":")[0].strip().upper()
        if token in _SECTION_KEYS:
            return header, idx + 1, token
        # key : value pairs (with or without spaces around the colon)
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().upper()
            if key in _HEADER_KEYS:
                header[key] = value.strip()
                continue
        # Some files use "KEY value" without a colon; treat first token as key
        parts = line.split(None, 1)
        if parts[0].upper() in _HEADER_KEYS and len(parts) == 2:
            header[parts[0].upper()] = parts[1].strip()
            continue
    raise ValueError("No section keyword (NODE_COORD_SECTION / EDGE_WEIGHT_SECTION) found")


# ---------------------------------------------------------------------------
# Section readers
# ---------------------------------------------------------------------------

def _read_node_coords(lines, start_idx, n, dim=2):
    """Read n lines of node coordinates. Each line is `index x y [z ...]`."""
    coords = np.zeros((n, dim), dtype=np.float64)
    i = 0
    idx = start_idx
    while i < n and idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue
        if line.upper() in _SECTION_KEYS:
            break
        parts = line.split()
        # First token is node index (1-based); remaining are coordinates
        if len(parts) < dim + 1:
            raise ValueError(f"NODE_COORD row has too few fields: {line!r}")
        coords[i] = [float(p) for p in parts[1 : dim + 1]]
        i += 1
    if i != n:
        raise ValueError(f"Expected {n} coordinate rows, got {i}")
    return coords, idx


def _read_edge_weights(lines, start_idx, n, fmt):
    """Read EXPLICIT edge weights and assemble the full (n, n) distance matrix.

    Supported formats: FULL_MATRIX, LOWER_DIAG_ROW, UPPER_DIAG_ROW, LOWER_ROW,
    UPPER_ROW.
    """
    fmt = fmt.upper()
    # Flatten all numeric tokens until we hit another section or EOF
    tokens = []
    idx = start_idx
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue
        if line.upper() in _SECTION_KEYS:
            break
        tokens.extend(line.split())
    vals = [float(t) for t in tokens]

    D = np.zeros((n, n), dtype=np.float64)
    k = 0
    if fmt == "FULL_MATRIX":
        if len(vals) < n * n:
            raise ValueError(f"FULL_MATRIX expected {n*n} values, got {len(vals)}")
        D = np.array(vals[: n * n], dtype=np.float64).reshape(n, n)
    elif fmt == "LOWER_DIAG_ROW":
        for i in range(n):
            for j in range(i + 1):
                D[i, j] = vals[k]
                D[j, i] = vals[k]
                k += 1
    elif fmt == "UPPER_DIAG_ROW":
        for i in range(n):
            for j in range(i, n):
                D[i, j] = vals[k]
                D[j, i] = vals[k]
                k += 1
    elif fmt == "LOWER_ROW":
        for i in range(1, n):
            for j in range(i):
                D[i, j] = vals[k]
                D[j, i] = vals[k]
                k += 1
    elif fmt == "UPPER_ROW":
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[i, j] = vals[k]
                D[j, i] = vals[k]
                k += 1
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
    return D


# ---------------------------------------------------------------------------
# TSPLIB95 distance functions
# ---------------------------------------------------------------------------

def _dist_euc_2d(coords):
    """Distance matrix under TSPLIB EUC_2D rules (nint rounding)."""
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    return np.round(d).astype(np.float64)  # TSPLIB's nint = round half to nearest


def _dist_ceil_2d(coords):
    """Distance matrix under TSPLIB CEIL_2D rules (ceiling rounding)."""
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    return np.ceil(d).astype(np.float64)


def _dist_att(coords):
    """Pseudo-Euclidean distance used by the ATT instances (TSPLIB95 spec)."""
    diff = coords[:, None, :] - coords[None, :, :]
    xd = diff[..., 0]
    yd = diff[..., 1]
    rij = np.sqrt((xd * xd + yd * yd) / 10.0)
    tij = np.round(rij)
    # If tij < rij, dij = tij + 1 else dij = tij
    dij = np.where(tij < rij, tij + 1, tij)
    np.fill_diagonal(dij, 0)
    return dij.astype(np.float64)


def _geo_to_radians(coord_column):
    """Convert DDD.MM coordinates (degrees.minutes) to radians as per TSPLIB95."""
    PI = 3.141592
    deg = np.trunc(coord_column)
    minutes = coord_column - deg
    return PI * (deg + 5.0 * minutes / 3.0) / 180.0


def _dist_geo(coords):
    """Great-circle distance on Earth (radius 6378.388 km) per TSPLIB95."""
    RRR = 6378.388
    lat = _geo_to_radians(coords[:, 0])
    lon = _geo_to_radians(coords[:, 1])
    # Standard TSPLIB95 formulation
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    for i in range(n):
        # Vectorized row
        q1 = np.cos(lon[i] - lon)
        q2 = np.cos(lat[i] - lat)
        q3 = np.cos(lat[i] + lat)
        D[i] = np.floor(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
    np.fill_diagonal(D, 0)
    return D


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_NATIVE_EUCLIDEAN_TYPES = {"EUC_2D", "CEIL_2D", "EUC_3D", "CEIL_3D"}


def parse_tsplib_file(path) -> Dict:
    """Parse a TSPLIB .tsp file and return a dict with coords + distance matrix.

    See the module docstring for the returned schema.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header, body_idx, section = _read_header(lines)

    if header.get("TYPE", "TSP").upper() != "TSP" and header.get("TYPE", "").upper() != "TSP":
        # Some files omit TYPE; we only care that NODE / EDGE sections parse correctly.
        pass

    n = int(header["DIMENSION"])
    ewt = header["EDGE_WEIGHT_TYPE"].upper()
    ewf = header.get("EDGE_WEIGHT_FORMAT", "").upper()

    raw_coords: Optional[np.ndarray] = None
    dist_matrix: Optional[np.ndarray] = None

    # --- Dispatch on section ---
    if section == "NODE_COORD_SECTION":
        # Most instances land here (EUC_2D, CEIL_2D, ATT, GEO).
        dim_for_coords = 3 if "3D" in ewt else 2
        raw_coords, _ = _read_node_coords(lines, body_idx, n, dim=dim_for_coords)

        if ewt in _NATIVE_EUCLIDEAN_TYPES:
            # For native Euclidean types the benchmark uses the raw coordinates
            # directly (the estimator computes its own MST internally). Building
            # the full (n, n) distance matrix is unnecessary and impossible for
            # very large instances (e.g. pla85900 with n=85900 would need 55 GiB).
            # We skip it here; downstream code that needs it (MDS path) won't
            # reach this branch anyway.
            dist_matrix = None
        elif ewt == "ATT":
            dist_matrix = _dist_att(raw_coords)
        elif ewt == "GEO":
            dist_matrix = _dist_geo(raw_coords)
        else:
            raise NotImplementedError(
                f"EDGE_WEIGHT_TYPE {ewt} with coordinate section is not supported yet"
            )

    elif section == "EDGE_WEIGHT_SECTION":
        if ewt != "EXPLICIT":
            raise ValueError(
                f"EDGE_WEIGHT_SECTION found but EDGE_WEIGHT_TYPE={ewt} (expected EXPLICIT)"
            )
        dist_matrix = _read_edge_weights(lines, body_idx, n, ewf)

    else:
        raise ValueError(f"Unexpected section keyword: {section}")

    return {
        "name": header.get("NAME", path.stem),
        "n": n,
        "edge_weight_type": ewt,
        "edge_weight_format": ewf,
        "raw_coords": raw_coords,
        "distance_matrix": dist_matrix,
        "is_native_euclidean": ewt in _NATIVE_EUCLIDEAN_TYPES,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tsplib_parser.py <file.tsp>")
        sys.exit(1)
    info = parse_tsplib_file(sys.argv[1])
    print(f"name       : {info['name']}")
    print(f"n          : {info['n']}")
    print(f"ewt        : {info['edge_weight_type']}")
    print(f"ewf        : {info['edge_weight_format']}")
    print(f"native_euc : {info['is_native_euclidean']}")
    if info["raw_coords"] is not None:
        print(f"raw_coords shape: {info['raw_coords'].shape}")
    print(f"dist_matrix shape: {info['distance_matrix'].shape}")
    print(f"mean dist  : {info['distance_matrix'][np.triu_indices(info['n'], 1)].mean():.3f}")
