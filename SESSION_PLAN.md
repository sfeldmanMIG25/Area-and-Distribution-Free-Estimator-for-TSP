# Session handoff — 2026-04-18

Session terminated before V4 training. Purpose of this file: capture everything
that changed on disk today, every unresolved issue, and the exact commands the
next session needs to pick up.

## TL;DR of the open state

| Item | State | Note |
|---|---|---|
| V3 codebase | **strict, no fallbacks** (except allowed Delaunay→dense MST) | breaking change from previous session; any data issue now halts the pipeline |
| V3 retrains | **done** (Linear, LGBM, Interp) against clean 90,395-row CSV | Neural V3 checkpoint still missing — not used in paper |
| Dataset repair | **partial** — 101 seed-recovered instances still have stale `optimal_cost` in solutions/ | must re-solve before V4 training |
| 2D benchmark | **done** against retrained V3 — 32,670 rows, 13 models | see `Generalized_TSP_Analysis/benchmark_results_2D_v3.csv` |
| TSPLIB benchmark | **done** — `tsplib_benchmark/results/all_models_tsplib.csv` | LGBM_V3 110/110 ok, MAPE 4.15 % |
| ND benchmark | **deferred** per user — to be run externally on separate compute | ran at 8 it/s after dim cap fix; user killed |
| LGBM V4 | **scaffolded only**, no model yet | all scripts in `lgbm_model_v4/`; follow its README |
| Paper | **edited** — Task 25/26/29 applied, training counts updated | did not recompile after last round of edits |

## What changed on disk this session

### Core library
- `tsp_utils_2.py`
  - All `try/except` stripped except the single allowed Delaunay→dense MST
    fallback in `get_mst_length`, `estimate_tsp_mst_ratio`,
    `estimate_tsp_composite`, `_calculate_gart_features`.
  - `DELAUNAY_MAX_DIM = 3` (was effectively 8) — measured crossover at d=4.
  - `CONVEX_HULL_MAX_DIM = 8` — blocks QHull from being called on d ≥ 9.
  - `estimate_tsp_kwon` now raises `ValueError` for `n > 300`
    (KWON_CALIBRATION_N_MAX). Callers must gate and record status rows.
- `feature_creator_v3.py`
  - Strict I/O (no silent file skips). `load_instance_data` raises on
    malformed binary/JSON. `process_file_worker` has no try/except.
  - Bounding-hypervolume computed in log-space with clipping at
    `log ≤ 690` to prevent float64 overflow at d = 100; exposes both
    `bounding_hypervolume` and `log_bounding_hypervolume`.
  - Delaunay dim cap = 3.
- V3 estimators (`linear_model_v3`, `interpretable_model_v3`, `nn_est_alpha_v3`,
  `lgbm_model_v3`):
  - All use Delaunay for d ∈ {2, 3} only; dense otherwise.
  - LGBM estimator now emits `log_bounding_hypervolume` and `log_node_density`
    inline for parity with the new feature schema.

### Training scripts
- `lgbm_model_v3/LGBM_Alpha_Model_V3.py`
  - Optuna import deferred; reuses committed `best_params_v3.json` when
    present (skip the search — user directive).
- `linear_model_v3/train_linear_v3.py` — unchanged logic, retrained.
- `interpretable_model_v3/train_interpretable_v3.py`
  - `LGBM_MODEL` constant changed from a stale absolute path to a
    relative path joined off `ROOT_DIR`.

### Runners
- `run_benchmark_2D_all.py`
  - Stripped silent `try/except: return None` worker. Errors now halt
    the benchmark (policy).
  - Kwon gate: worker emits `status='kwon_out_of_calibration'` row for
    `n > 300`. `calculate_metrics_and_print` filters by `status == 'ok'`
    before computing aggregates.
  - Emoji removed (Windows cp1252 can't encode ✅). Use `[OK]` text.
  - Kwon + Daganzo added to the schedule.
  - Neural V3 commented out (checkpoint missing in this repo snapshot).
- `run_benchmark_ND_final.py`
  - Same worker/strictness changes.
  - Schedule expanded to include every N-D-capable academic estimator
    (BHH, Cavdar, Chien, Vinel, Composite, MST_Ratio, Hilbert).
  - Kwon/Daganzo excluded (2D-planar only).
- `tsplib_benchmark/run_all_models_tsplib.py` — already had never-silent
  status emission; no changes.

### Data repair artifacts (at repo root)
- `Chunk_Archiver.py` — MODE restored to `'upload'` after one `'unpack'` run.
- `data_recovery.py`, `data_recovery_v2.py` — ad-hoc repair scripts.
  **v2 is the canonical one.** The results of v2:
  - 82,643 bins OK unchanged
  - 3,261 corrupt bins replaced by their valid JSON
  - 4,882 JSONs rewritten from valid bins
  - 101 both-corrupt pairs regenerated from the `generation_seed` in the
    corrupt JSON text (byte-identical to truth)
  - 24 truly lost (23 in grid → dropped to 90,395)
- `rebuild_binaries.py` — writes `.bin` from every valid `.json`. Used once.
- `data_recovery_lost_v2.txt` — list of 24 truly-lost instance names.
- `tsp_features_v3.previous.csv` — 90,418-row snapshot of the pre-repair
  feature CSV. Renamed before regeneration.

### Paper
- `paper_reference/Area_Free_Main.tex`
  - Task 25 (complexity fold): `§3` renamed from "Estimation Model Development
    and Complexity Analysis" → "Estimation Model Development". `§3.3
    Computational Complexity Analysis` deleted; a condensed `\paragraph{Complexity}`
    now lives at the end of `§3.1 Model Architecture and Training`.
  - Task 26 (MDS mechanics to appendix): `§5.1 MDS preprocessing` now carries
    only a short description with forward-ref to `Appendix app:mds`.
  - Task 29 (training data rewrite): line 288 now says "90,418 solved
    instances, stratified 70/20/10 into 59,247 train / 16,714 val /
    14,457 test; d = 100 locked to test". Appendix `app:training` mirrors
    the same numbers with the stratification rule.
- `paper_reference/plot_runtime.py` — scaffolded, not yet run against the
  fresh TSPLIB CSV.

### V4 scaffold (new folder `lgbm_model_v4/`)
- `feature_engineering.py`, `build_features_csv.py`, `feature_analysis.py`,
  `train.py`, `lgbm_estimator_v4.py`, `README.md` — see folder.

## Pending to finish V4

Full recipe is in `lgbm_model_v4/README.md`. Summary:

1. **Resolve the 101 stale-solution rows** (critical — otherwise training labels
   are contaminated).
2. Build `tsp_features_v4.csv`.
3. Run `feature_analysis.py`, review `feature_report.md`, adjust
   `selected_features.json`.
4. `train.py` (100 Optuna trials, multi-objective).
5. Add `LGBM_V4` row to the 2D / TSPLIB / ND schedules and re-run.
6. Update paper tables with V4 column + 95 % bootstrap CIs on SDPE.

## Pending to finish the paper

1. Recompile `Area_Free_Main.tex` after the above edits — not yet done this
   session.
2. Update the 2D results table with the retrained numbers (file already
   exists: `Generalized_TSP_Analysis/benchmark_results_2D_v3.csv`).
3. Update the TSPLIB results table (`tsplib_benchmark/results/all_models_tsplib.csv`).
4. Regenerate `boxplot_2d_errors.png`, `boxplot_tsplib_errors.png` against
   the new CSVs; delete the old ones (last session's CLAUDE todo).
5. Once ND is resolved (externally), regenerate `boxplot_nd_errors.png` and
   the ND tables.
6. `plot_runtime.py` → `paper_reference/plot_runtime.png`.

## Pending to finish the dataset

1. Re-solve the 101 seed-recovered instances with Concorde + LKH-3. The
   instance names aren't logged (omission); recover them by:
   - Listing names in `tsp_features_v3.previous.csv` that still exist
     on disk with fresh `.json` content (compare file `mtime` against
     the recovery run time) — the 101 regenerated files were written by
     `data_recovery_v2.py`, so their mtimes match that run.
2. Re-run `feature_creator_v3.py` → `tsp_features_v3.csv`.
3. Only then rebuild the V4 CSV.

## Decisions made this session that should survive

- **SDPE is reported first** on all metric tables going forward (precision-first
  framing per user's standing preference).
- **Kwon is capped at n = 300** (KWON_CALIBRATION_N_MAX) with an explicit
  `status='kwon_out_of_calibration'` row for out-of-range instances. Paper must
  cite this cap.
- **Delaunay is used for d ∈ {2, 3} only.** Empirical crossover at d = 4.
- **ConvexHull blocks at d = 9+.** BHH/Cavdar/Vinel/Kwon/Daganzo fall back to
  axis-aligned bounding volume in higher dimensions; this is documented in
  `tsp_utils_2.py`.
- **No silent fallbacks.** Integrity pipeline — any malformed input halts
  execution. The only whitelisted fallback is `Delaunay → dense MST` for
  degenerate point sets.
- **V4 objective is precision-first.** Multi-objective Optuna (MAPE, SDPE); at
  Pareto picking time, find the SDPE-minimum trial inside a 5.5 % MAPE budget.

## Environment

- Python 3.14.3 on Windows 11 (cp1252 default stdout — use `PYTHONUTF8=1`).
- PyTorch installed this session (`pip install torch`) — required by the
  Neural V3 estimator class, even though the checkpoint is missing.
- Optuna installed.
- scikit-learn 1.8, scipy 1.17.1, lightgbm 4.6, numba 0.65.

## Pointers

- Everything V4-specific → `lgbm_model_v4/README.md`.
- Paper source of truth → `paper_reference/Area_Free_Main.tex`.
- Training data → `tsp_features_v3.csv` (current), `tsp_features_v3.previous.csv` (pre-repair snapshot).
- Canonical benchmarks:
  - 2D: `Generalized_TSP_Analysis/benchmark_results_2D_v3.csv`
  - TSPLIB: `tsplib_benchmark/results/all_models_tsplib.csv`
  - ND: **not regenerated this session** — old values in
    `Generalized_TSP_Analysis_ND/benchmark_results_ND_final.csv` are stale,
    do not quote them in the paper.
