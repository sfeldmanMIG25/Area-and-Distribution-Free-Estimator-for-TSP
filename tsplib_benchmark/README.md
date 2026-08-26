# TSPLIB95 Benchmark for GART 2.0

This module evaluates the GART 2.0 (LGBM V3) estimator against the full
TSPLIB95 symmetric TSP benchmark library (Reinelt, 1991). It tests whether
GART 2.0 — trained exclusively on synthetic instances — generalizes to
real-world and structured problem instances with known optimal solutions.

## Directory structure

```
tsplib_benchmark/
  instances/            # Downloaded .tsp files (111 instances, gitignored)
  ground_truth/
    optima.csv          # Published TSPLIB optimal tour lengths
  results/              # Timestamped benchmark CSVs (never overwritten)
  published_results/    # Literature baselines from other estimators (future)
  tsplib_parser.py      # TSPLIB95 file format parser
  classical_mds.py      # Classical MDS (Torgerson 1952) embedding
  download_tsplib.py    # One-time instance downloader
  run_all_models_tsplib.py # Benchmark runner (canonical)
```

## Quick start

```bash
# 1. Download all 111 TSPLIB instances (~50 MB)
python tsplib_benchmark/download_tsplib.py

# 2. Run the full benchmark
python tsplib_benchmark/run_all_models_tsplib.py

# 3. Cap instance size
python tsplib_benchmark/run_all_models_tsplib.py --max-n 500
```

Results are written to `results/all_models_tsplib.csv`, which the run
overwrites. Every (instance, model) pair produces a row: an instance that
cannot be scored carries a `status` naming the reason rather than being
dropped.

`run_tsplib_benchmark.py`, `run_supplemental_baselines.py` and
`backfill_kwon_daganzo.py` have been deleted. They scored Chien,
Kwon--Golden--Wasil and Daganzo from coefficients transcribed out of a
secondary source, and the primaries are paywalled with no obtainable
open-access copy. Those three estimators are no longer benchmarked anywhere,
and the superseded functions behind them in `tsp_utils_2.py` now raise.

## How instances are handled

### Native Euclidean (EUC_2D, CEIL_2D)

Raw 2D coordinates are passed directly to the estimator. The estimator computes
its own MST internally using `scipy.spatial.distance.cdist` (Euclidean metric).
This is the closest match to the synthetic training data.

### Non-Euclidean (ATT, GEO, EXPLICIT)

The TSPLIB distance function is used to build the full distance matrix, then
**classical MDS** (Torgerson 1952) embeds the distance matrix into a Euclidean
coordinate space. The embedding dimensionality is chosen automatically to retain
99.9% of the positive eigenvalue mass, capped at 100 dimensions.

**Why MDS is necessary:** The estimator internally computes a Euclidean MST from
coordinates. If the coordinates don't respect the TSPLIB distance function (e.g.,
ATT divides by sqrt(10), GEO uses great-circle distance), the internal MST will
be wrong, and the alpha prediction meaningless.

**Limitations of MDS on non-metric data:** Some EXPLICIT instances (notably
`brg180`) use distance matrices that violate the triangle inequality. Classical
MDS assumes metric distances; on non-metric data, the Gram matrix has large
negative eigenvalues and the embedding is unreliable. These instances should
be interpreted with caution — they represent a fundamentally different problem
class that no Euclidean embedding can faithfully represent.

## Key results (April 2026 benchmark)

| Subset | n | MAPE | Median |Gap| | Bias |
|--------|---|------|---------|------|
| EUC_2D (all) | 78 | 3.45% | 2.37% | +2.95% |
| EUC_2D (n <= 1000) | 49 | 3.03% | 2.17% | +2.29% |
| EUC_2D (n > 1000) | 29 | 4.15% | 2.62% | +4.06% |
| CEIL_2D | 3 | 6.02% | 4.95% | +3.26% |
| ATT (via MDS) | 2 | 32.16% | 32.16% | +29.36% |
| GEO (via MDS) | 10 | 39.14% | 6.77% | +36.64% |
| EXPLICIT (via MDS) | 17 | 4437.58% | 51.93% | +4422.93% |

### Interpretation

**EUC_2D performance is strong.** At 3.45% MAPE across 78 instances spanning
n=14 to n=18512, the model trained on synthetic data generalizes well to
real-world Euclidean instances. The positive bias suggests GART 2.0 slightly
overestimates — a conservative property useful for planning applications.
Extrapolation beyond n=1000 (the training cap) adds only ~1% MAPE.

**MDS-embedded instances are mixed.** GEO instances with low MDS dimensionality
(ulysses16: 2D, burma14: 4D) work well (<7% error). Instances requiring high
dimensionality or with non-metric distance matrices fail badly. The EXPLICIT
aggregate is dominated by `brg180` (74,331% error; a non-metric programmed
logic array instance).

**CEIL_2D works as expected.** The ceil-rounding vs nint-rounding difference is
negligible for the estimator.

## Dependencies

- numpy, scipy, pandas, tqdm (standard scientific stack)
- lightgbm, numba, joblib, scikit-learn (via the LGBM V3 estimator)
- No additional packages required

## References

- Reinelt, G. (1991). TSPLIB — A Traveling Salesman Problem Library.
  ORSA Journal on Computing, 3(4), 376–384.
- Torgerson, W. S. (1952). Multidimensional scaling: I. Theory and method.
  Psychometrika, 17(4), 401–419.
