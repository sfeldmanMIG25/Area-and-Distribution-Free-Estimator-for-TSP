"""Is the augmentation 'lattice' family the benchmark 'grid' generator renamed?

Two independent lines of evidence:
  (1) direct coordinate-level re-derivation: run BOTH generators at matched
      (n, d=2, G) and compare the realised point sets and the 31 model features;
  (2) parameter-cell census + how much of the headline 2D gain the grid family
      is carrying.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
for p in (ROOT, ROOT / "lgbm_model_v3"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_pipeline import d2_benchmark_gen as bg          # noqa: E402
from data_pipeline.augment_gen import gen_lattice          # noqa: E402

print("=" * 78)
print("(1) COORDINATE-LEVEL GENERATOR COMPARISON  (d=2, jitter 0.05)")
print("=" * 78)
for n, G in ((400, 1000.0), (900, 10000.0), (1000, 10000.0), (300, 1000.0)):
    rng = np.random.default_rng(7)
    a = bg.generate_grid(n, G, rng)                       # benchmark 'grid'
    b = gen_lattice(np.random.default_rng(7), n, 2, G, jitter=0.05)  # augment
    side_b = math.ceil(math.sqrt(n))
    m_a = math.ceil(n ** (1.0 / 2))
    # lattice constants each generator lands on
    sp_bench, sp_aug = G / side_b, G / m_a
    # nearest-neighbour spacing distribution is the discriminating statistic
    def nnsp(X):
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        return d.min(1)
    na, nb = nnsp(a), nnsp(b)
    print(f"n={n:5d} G={G:8.0f} | sites/side bench={side_b} aug={m_a} "
          f"| spacing bench={sp_bench:.4f} aug={sp_aug:.4f} "
          f"| NN-spacing mean bench={na.mean():.4f} aug={nb.mean():.4f} "
          f"| sd bench={na.std():.4f} aug={nb.std():.4f}")

print("\nJitter law:")
print("  benchmark d2_benchmark_gen.generate_grid:262  "
      "(rng.random(2)-0.5) * (G/side * 0.1)   ->  U(-0.05*spacing, +0.05*spacing)")
print("  augment    augment_gen.gen_lattice:409        "
      "rng.uniform(-jitter*spacing, jitter*spacing), jitter in (0.0, 0.01, 0.05)")
print("  at jitter=0.05 the two jitter laws are the SAME distribution.")
print("  cell centres: benchmark (i+0.5)*G/side ; augment (digits+0.5)*G/m ; "
      "side == m == ceil(sqrt(n)) for d=2.")

print("\n" + "=" * 78)
print("(2) PARAMETER-CELL CENSUS AND GAIN DECOMPOSITION")
print("=" * 78)
C = pd.read_csv(HERE / "v4_study_feature_cache.csv", low_memory=False)
gen = pd.read_csv(HERE / "augmentation_2d_features.csv",
                  usecols=["instance_name", "generator"])
C = C.merge(gen.rename(columns={"instance_name": "instance"}), on="instance",
            how="left")
b2 = C[C.stratum == "bench2d"]
grid = b2[b2.generator == "grid"]
print("benchmark grid cells (n, G):")
gcells = grid["instance"].str.extract(r"n(\d+)-g(\d+)").astype(int)
print(sorted(set(map(tuple, gcells.values))))

aug = C[C.stratum == "augment"]["instance"]
lat = aug[aug.str.contains("lattice")]
print(f"\naugment lattice/hexlattice instances: {len(lat)}")
lc = lat.str.extract(r"AUG_(hex)?lattice-d(\d+)-n(\d+)-g(\d+)-jitter(\w+?)-r")
lc.columns = ["hex", "d", "n", "g", "jit"]
d2 = lc[lc.d == "2"]
print("d=2 lattice cells (n, G, jitter):")
print(d2.groupby(["n", "g", "jit"]).size().to_string())

# exact (n, G) collisions with the benchmark grid cells
bench_cells = set(map(tuple, gcells.values))
aug_cells = {(int(r.n), int(r.g)) for r in d2.itertuples()}
print(f"\nbenchmark grid (n,G) cells:      {sorted(bench_cells)}")
print(f"augment d=2 lattice (n,G) cells: {sorted(aug_cells)}")
print(f"EXACT (n,G) collisions: {sorted(bench_cells & aug_cells)}")

# ---- how much of the headline 2D gain does grid carry? -------------------
PI = pd.read_csv(HERE / "support_arms_per_instance.csv")
p = PI[PI.stratum == "bench2d"].pivot_table(index="instance", columns="model",
                                            values="err_pct")
p = p.join(C.set_index("instance")["generator"], how="left")
tot = {}
for m in ("FROZEN", "A"):
    tot[m] = p[m].abs().mean()
print(f"\n2D overall MAPE  frozen={tot['FROZEN']:.4f}  A={tot['A']:.4f}  "
      f"gain={tot['FROZEN']-tot['A']:.4f}")

rows = []
for g, s in p.groupby("generator"):
    rows.append({"generator": g, "n": len(s),
                 "frozen": s.FROZEN.abs().mean(), "A": s.A.abs().mean(),
                 "gain": s.FROZEN.abs().mean() - s.A.abs().mean(),
                 "share_of_total_gain_pct":
                     (s.FROZEN.abs().mean() - s.A.abs().mean()) * len(s)
                     / len(p) / (tot["FROZEN"] - tot["A"]) * 100.0})
D = pd.DataFrame(rows).sort_values("share_of_total_gain_pct", ascending=False)
D.to_csv(HERE / "armA_verify_gain_decomposition.csv", index=False)
print("\nper-generator gain decomposition:")
print(D.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

ng = p[p.generator != "grid"]
print(f"\n2D MAPE EXCLUDING the grid family (N={len(ng)}): "
      f"frozen={ng.FROZEN.abs().mean():.4f}  A={ng.A.abs().mean():.4f}  "
      f"gain={ng.FROZEN.abs().mean()-ng.A.abs().mean():.4f}")
nge = p[~p.generator.isin(["grid", "line_noise"])]
print(f"2D MAPE EXCLUDING grid AND line_noise (N={len(nge)}): "
      f"frozen={nge.FROZEN.abs().mean():.4f}  A={nge.A.abs().mean():.4f}  "
      f"gain={nge.FROZEN.abs().mean()-nge.A.abs().mean():.4f}")
