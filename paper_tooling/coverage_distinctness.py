"""Is the alpha-coverage corpus a different generator from the evaluation set?

``decontaminated_arm_protocol.md`` §1 records why this needs an executable
answer rather than a naming convention: an earlier augmentation shipped a d=2
``lattice`` family that was line-for-line the 2D benchmark's ``grid``
generator, and the docstring claiming cross-family transfer was simply false.
The rule taken from that episode is that distinctness must be checked in code.

The argument that settles it is the source code, read in the module docstring
of ``data_pipeline/coverage_gen.py``.  What follows corroborates it with two
statistics, each derived from what one specific evaluation generator actually
constructs, so that "different generator" is testable rather than asserted.

  lattice_residual -- mean distance from each coordinate to the nearest line of
      the best axis-aligned lattice of ceil(sqrt(n)) lines per axis, in units
      of the lattice spacing.  ``d2_benchmark_gen.generate_grid`` places points
      at cell centres and jitters them by U(+-0.05 spacing), so it must score
      about 0.025.  Anything not built on a lattice scores near 0.25, the mean
      of a uniform residual.  This is the statistic that would have caught the
      d=2 ``lattice`` contamination in the earlier arm.

  line_band_ratio -- transverse RMS about the principal axis, in units of
      0.02 G.  ``extend_line_noise.generate_line_noise`` puts every point in a
      Gaussian band of exactly sigma = 0.02 G about one straight line, so it
      must score about 1.  A curved or branched skeleton spreads much wider:
      an arc at the sweep floor of 0.5 rad already has a sagitta of 6% of its
      chord, roughly three times the band.

A first attempt used "max perpendicular offset over axial extent" and found
overlap.  That statistic is not a fingerprint: ``line_noise`` clips
out-of-box coordinates onto the faces instead of rejecting them, so its
instances are bent polylines rather than straight bands and the statistic
reaches 0.23 on them.  The overlap was a property of the statistic, not of the
generators, and it is recorded here rather than dropped.

The claim is NOT that the two corpora are far apart in feature space.  Coverage
was aimed at a region the evaluation set also occupies -- that is the point of
fixing a coverage gap -- and it is disclosed.  The claim is that the generating
processes are different, which is what keeps the evaluation a transfer test.

Writes ``paper_tooling/coverage_distinctness.csv`` only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COV_INST = ROOT / "alpha_coverage" / "instances"
B2_INST = ROOT / "Generalized_TSP_Analysis" / "instances"
OUT = ROOT / "paper_tooling" / "coverage_distinctness.csv"


def straightness(coords: np.ndarray) -> float:
    """Max perpendicular offset from the principal axis / extent along it."""
    c = coords - coords.mean(axis=0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    along = c @ axis
    perp = np.linalg.norm(c - np.outer(along, axis), axis=1)
    extent = float(along.max() - along.min())
    return float(perp.max() / extent) if extent > 0 else float("nan")


def lattice_residual(coords: np.ndarray) -> float:
    """Best-fit lattice residual per axis, in spacing units.

    For each axis the line count is searched rather than assumed, because
    ``generate_grid`` leaves the last column partial whenever n is not a
    perfect square, and a fixed ceil(sqrt(n)) fit then misses by enough to hide
    a real lattice: assuming the count put the benchmark's grid class at a
    median residual of 0.125, indistinguishable from uniform.  Searching the
    count recovers the ~0.025 the +-0.05-spacing jitter implies.

    ~0.03 for a jittered square lattice, ~0.25 (the mean of a uniform residual)
    for coordinates that are not quantised at all.
    """
    n, d = coords.shape
    m = max(2, math.ceil(n ** (1.0 / d)))
    best = []
    for ax in range(d):
        x = coords[:, ax]
        span = float(x.max() - x.min())
        if span <= 0:
            continue
        r = 1.0
        for k in range(max(2, m - 2), 2 * m + 3):
            u = (x - x.min()) / (span / (k - 1))
            r = min(r, float(np.mean(np.abs(u - np.round(u)))))
        best.append(r)
    return float(np.mean(best)) if best else float("nan")


def line_band_ratio(coords: np.ndarray, grid_size: float) -> float:
    """Transverse RMS about the principal axis, in units of 0.02 * G.

    ~1 for line_noise, which is that band by construction.
    """
    c = coords - coords.mean(axis=0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    perp = c - np.outer(c @ axis, axis)
    return float(np.sqrt(np.mean(np.sum(perp ** 2, axis=1))) / (0.02 * grid_size))


def scan(paths, tag: str, limit: int | None = None) -> list[dict]:
    rows = []
    for i, p in enumerate(sorted(paths)):
        if limit and i >= limit:
            break
        with open(p) as f:
            inst = json.load(f)
        if int(inst.get("dimension", 2)) != 2:
            continue
        c = np.asarray(inst["coordinates"], dtype=np.float64)
        if c.shape[0] < 5:
            continue
        g = float(inst.get("grid_size", 1000))
        rows.append({"tag": tag, "instance": p.stem, "n": c.shape[0], "grid": g,
                     "family": inst.get("coverage_family", inst.get("distribution_type")),
                     "straightness": straightness(c),
                     "lattice_residual": lattice_residual(c),
                     "line_band_ratio": line_band_ratio(c, g)})
    return rows


def main() -> None:
    rows = scan(COV_INST.glob("*.json"), "coverage")
    rows += scan(B2_INST.glob("TSP-line_noise-*.json"), "bench_line_noise")
    rows += scan(B2_INST.glob("TSP-grid-*.json"), "bench_grid")
    rows += scan(B2_INST.glob("TSP-random-*.json"), "bench_random")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print(f"[distinct] {len(df)} d=2 instances scanned -> {OUT.name}\n")
    for col in ("lattice_residual", "line_band_ratio", "straightness"):
        print(f"--- {col}")
        print(df.groupby("tag")[col]
              .agg(["count", "min", lambda s: s.quantile(0.05), "median",
                    lambda s: s.quantile(0.95), "max"])
              .set_axis(["n", "min", "p05", "median", "p95", "max"], axis=1)
              .round(4).to_string())
        print()

    cov = df[df.tag == "coverage"]
    ln = df[df.tag == "bench_line_noise"]
    gr = df[df.tag == "bench_grid"]
    print("separation (a coverage instance is inside an evaluation generator's "
          "range only if the intervals overlap):")
    if len(gr):
        sep = cov.lattice_residual.min() > gr.lattice_residual.max()
        print(f"  lattice_residual  grid max {gr.lattice_residual.max():.4f}  "
              f"coverage min {cov.lattice_residual.min():.4f}  "
              f"-> {'DISJOINT' if sep else 'OVERLAP'}")
    if len(ln):
        sep = cov.line_band_ratio.min() > ln.line_band_ratio.max()
        print(f"  line_band_ratio   line_noise max {ln.line_band_ratio.max():.4f}  "
              f"coverage min {cov.line_band_ratio.min():.4f}  "
              f"-> {'DISJOINT' if sep else 'OVERLAP'}")
        n_in = int((cov.line_band_ratio <= ln.line_band_ratio.max()).sum())
        print(f"  coverage rows inside the line_noise band range: {n_in} / {len(cov)}")
    print("\nper coverage family:")
    print(cov.groupby("family")[["lattice_residual", "line_band_ratio"]]
          .agg(["count", "min", "median", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
