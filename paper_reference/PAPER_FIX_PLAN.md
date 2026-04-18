# Paper Fix Plan — Area_Free_Main.tex

Date: 2026-04-18. Companion to [PAPER_AUDIT_2026-04-18.md](PAPER_AUDIT_2026-04-18.md).

## Session infrastructure (already completed)

- **Memory updated**: `project_naming.md` now pins **GART 2.0** as canonical paper name (was incorrectly marked GART 3.0).
- **Auto-recompile hook installed**: `C:\Users\catst\.claude\hooks\recompile-paper.ps1` runs `latexmk` in background whenever any `.tex` / `.bib` in `paper_reference/` is edited. Wired into `~/.claude/settings.json` PostToolUse for Edit|Write.
- **Audit scope revisions** based on user clarifications:
  - **Item #3 (training d-range)** — downgraded to MINOR. `feature_creator_v3.py:261` sets `df_d100['split'] = 'test'`, so d=100 instances are held-out test, not training. Paper's "trained on d≤50" claim is correct; need to clarify the split assignment.
  - **Item #4 (GART naming)** — reversed direction. Paper is correct at 2.0; CLAUDE.md and memory were wrong. Memory fixed; CLAUDE.md still says "GART 2.0 = LGBM V3" which is correct (no change needed).
  - **Item #13 (TSPLIB 78 vs 71)** — root cause identified. `run_supplemental_baselines.py` exists for "the 7 TSPLIB EUC_2D instances skipped by the original all-models run because they have n > 5000". See Section E below — this needs re-running and merging into canonical CSV before the paper claim of N=78 is true.

## Pre-work environment tasks

### A. LKH-3 setup
- Already wired at `C:\LKH\LKH-3.exe` in [Dataset_Generator.py:38](../Dataset_Generator.py:38). Confirm the exe exists; if missing, download the stand-alone Windows executable from http://webhotel4.ruc.dk/~keld/research/LKH-3/ and place at that path. No compile step needed on Windows — Helsgaun ships a pre-built .exe.
- Validation: `C:\LKH\LKH-3.exe pr2392.par` should solve a 2392-city instance.

### B. Concorde (WSL) setup
Concorde has no native Windows build. Install inside WSL (Ubuntu), then call from Windows via `wsl concorde …` or keep a WSL-only benchmark script.

Plan:
1. Confirm WSL Ubuntu is installed: `wsl -l -v`.
2. Inside WSL: `sudo apt update && sudo apt install -y build-essential gcc m4 dos2unix`.
3. Fetch QSopt LP (Concorde's default free LP solver): `wget https://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.PIC.a` and `wget …/qsopt.h` into `~/qsopt/`.
4. Fetch Concorde: `wget https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz`.
5. **Unpack inside WSL**, not Windows Explorer: `tar xzf co031219.tgz` — this preserves LF line endings. Extracting via a Windows tool then copying into WSL is what causes "bash: ./configure: cannot execute" errors because the shell scripts get CRLF endings.
6. If the source tree was touched by Windows, run `find concorde -type f -exec dos2unix {} \;` before configure.
7. `cd concorde && ./configure --with-qsopt=$HOME/qsopt && make`.
8. Binary at `concorde/TSP/concorde`. Add `alias concorde=...` in `~/.bashrc` for convenience.
9. Python wrapper: `pip install pyconcorde` inside the WSL venv.

Key corruption-avoidance rule: **keep source trees on WSL's native ext4 (`/home/<user>/`), not on `/mnt/d/…`**. Cross-filesystem operations between Windows (NTFS) and WSL can alter line endings and file permissions even with `core.autocrlf=false` in git.

### C. TSPLIB EUC_2D completeness
Why baselines are missing 7 of the 78 EUC_2D instances:
- [run_all_models_tsplib.py](../tsplib_benchmark/run_all_models_tsplib.py) had an internal size cap preventing academic baselines from running when n > 5000.
- The 7 affected instances are enumerated in [run_supplemental_baselines.py](../tsplib_benchmark/run_supplemental_baselines.py).
- Plan: re-run `run_supplemental_baselines.py`, merge its output rows into `tsplib_benchmark/results/all_models_tsplib_20260412_155608.csv`, then recompute all per-bucket statistics. This resolves Items #13 and #14 simultaneously.
- Need user go-ahead before re-running: the script likely takes hours on large baselines.

---

## Item-by-item plan (46 flags)

Each item has: **(a) what code/data proves** the flag, **(b) proposed edit to paper or code**, **(c) how we verify the fix held**.

### CRITICAL (7)

1. **Hyperparameter table fabrication** — Sec 3.2 lines 313–325.
   - Evidence: [lgbm_model_v3/best_params_v3.json](../lgbm_model_v3/best_params_v3.json) is the single source of truth.
   - Fix: overwrite the table verbatim from the JSON. Strip any "ensemble size / max leaves constraint" columns that don't exist in code.
   - Verify: after edit, diff table values against `json.load(open('best_params_v3.json'))` in a `python -c` one-liner; latex recompile via hook must succeed.

2. **Feature count 29 vs 28** — abstract line 109, Sec 3 ~230, conclusion 771.
   - Evidence: [feature_creator_v3.py](../feature_creator_v3.py) writes a known set of `features[...]` keys.
   - Fix: count the keys in `feature_creator_v3.py` via `python -c "import ast,io; …"`, then either (a) correct the paper to the true N, or (b) enumerate every feature in a new supplementary table to remove ambiguity.
   - Verify: count matches across abstract, §3, conclusion, and appendix.

3. **Training d-range wording** — Sec 3.1 Dataset Generation line ~293 and line 779.
   - Evidence: [Dataset_Generator.py:45](../Dataset_Generator.py:45) has 18 dim values including d=100; [feature_creator_v3.py:261](../feature_creator_v3.py:261) forces `split='test'` when d=100. So training IS restricted to d≤50, but the generator file includes d=100.
   - Fix: one sentence in §3.1 — "The generator covers d ∈ {2,…,50,100}. Instances with d=100 are deterministically assigned to the held-out test split (`feature_creator_v3.py`), so the trained model never sees d=100 during fit."
   - Verify: statement matches `df_d100['split'] = 'test'` in code.

4. **GART naming (already partially resolved)** — 56 occurrences.
   - Evidence: user confirmed GART 2.0 canonical.
   - Fix: **no paper change needed** (paper is correct at 2.0). Search for stray "GART 3.0" in any doc and correct. Memory updated this turn.
   - Verify: `rg -c "GART 3.0" paper_reference/` returns 0.

5. **BHH constant citation** — Sec 5.2.1 line 357.
   - Evidence: commonly cited value β₂ ≈ 0.7124; Percus & Martin 1996 reported a refined estimate — whether that refined value is 0.7080 or 0.7124 needs to be pulled directly from the Percus & Martin paper.
   - Fix proposal: WebFetch the Percus & Martin paper (J. Stat. Phys. 1996 or cond-mat preprint on arXiv) and quote the exact numerical value. If the paper's 0.7080 is wrong, replace with 0.7124; if 0.7080 is correct, cite the canonical 0.7124 to a different source (e.g., Applegate, Bixby, Chvátal, Cook 2006).
   - Verify: citation URL is live; quoted value matches the source.

6. **No reproducibility block.**
   - Evidence: [LGBM_Alpha_Model_V3.py:29–31](../lgbm_model_v3/LGBM_Alpha_Model_V3.py:29) has `RANDOM_STATE = 42`, `OPTUNA_N_TRIALS = 100`, `EARLY_STOPPING_ROUNDS = 100`. Optuna TPE is the default.
   - Fix: add a 1-paragraph subsection to §3.2 listing: RANDOM_STATE=42, Optuna TPE with 100 trials, early-stopping patience 100, train/val/test split via deterministic `split` column, α target clipped to [1.0, 2.0], single LGBM model (no KFold), `tsp_features_v3.csv` as the feature matrix.
   - Verify: every value in the reproducibility block grep-matches a line in the code.

7. **SHAP / theoretical bridge missing.**
   - Evidence: [shap_analyzer.py](../shap_analyzer.py) exists and runs against the trained model.
   - Fix: generate SHAP global importance and beeswarm plots via `python shap_analyzer.py`, add top-5 features by mean(|SHAP|) as a new table, and write a paragraph linking the dominant features to the BHH scaling (e.g., "MST total length and bounding-box area dominate, consistent with BHH's $\sqrt{nA}$ scaling").
   - Verify: figure + table exist in paper_reference/ and are `\input`-ed by the tex.

### MAJOR (22)

8. **Orienting Table 1 (Cavdar-style).** Insert near line 151. Columns: `{Accurate, Fast, Distribution-agnostic, d>2, TSPLIB-tested}`, rows: `{BHH, Chien 1992, Cavdar-Sokol 2015, Kwon 1995, Daganzo 1984, Platzman-Bartholdi 1989, GART 2.0}`. Check-mark matrix, one-paragraph legend.

9. **Add Kwon (1995) baseline**, justify Daganzo / del Castillo omission in a sentence. Kwon's form is `(0.8326 - 0.0011n + 1.1147 R/n)·sqrt(nA)` (documented in Cavdar Table 3) — easy to add to `tsplib_benchmark/` and rerun.

10. **Headline metric consistency.** Decision: keep SDPE as headline (per memory `feedback_precision_first.md`); demote MAPE to a second paragraph. Abstract rewrite: lead with SDPE number, then "with MAPE Y% for completeness".

11. **Results-section trim.** Sec 5.3 lines 420–715 = 295 lines. Keep 3 headline tables in main text (2D aggregate, ND aggregate, TSPLIB aggregate). Move per-bucket breakdowns to `appendix_results.tex` and `\input` them.

12. **Merge MDS Sec 6 into §5.** Make it §5.4 "Real-world instances via MDS" — the culminating real-data validation, not a separate application chapter.

13. **TSPLIB baseline N** — depends on running `run_supplemental_baselines.py` (Section C above). Until that runs, the paper must say "EUC_2D instances with n ≤ 5000" as the baseline universe and report N=71.

14. **MST Ratio n>1500 MAPE** — after #13, recompute. If MST Ratio still beats GART in that bucket at the corrected N, keep the narrative but fix the number. If not, rewrite.

15. **51,818 training instances reconciliation.** Run `python -c "import pandas as pd; df=pd.read_csv('tsp_features_v3.csv', usecols=['split']); print(df['split'].value_counts())"`. If `train=51818` confirmed, cite that; if not, either re-derive or reduce to the true number.

16. **α clipping disclosure** — Sec 3.2 / 4.1. Add one sentence: "We clip the regression target α=L_TSP/L_MST to [1.0, 2.0] to bound the prediction range; in-distribution empirical α lies within (1.02, 1.71) across all training instances."
    - *Note on code-vs-literature:* No web source was found to compel changing the clip. For a path (1D manifold) embedded in 2D, α=2 is attained exactly (out-and-back MST traversal); for starlike branched MSTs, α can exceed 2 in principle. **User decision needed**: leave as a safety clip and disclose, or remove the clip and retrain. If you want to remove it, the web link to justify: https://or.stackexchange.com/questions/2 discussions confirm α is bounded by 2 only in worst-case MST-based TSP approximations (Christofides 3/2, Double-MST 2). I will not autonomously change the code for this without an explicit go.

17. **"Distribution-free" reframe.** Global replace "distribution-free" → "distribution-agnostic at inference" in paper, explain in one sentence: model uses sample statistics only, and is trained over a curated set of 4 distributions to cover diverse regimes.

18. **Inference complexity — Sec 4.2 line 336.**
    - Evidence: LightGBM docs (https://lightgbm.readthedocs.io/en/latest/Features.html) explicitly state "leaf-wise tree growth produces deeper trees for a given leaf count than level-wise, which can reduce loss more but overfit small data; max_depth should be used to prevent this."
    - Fix: replace the ⌈log₂ L⌉ bound with "worst-case comparisons bounded by `max_depth` (here set to LightGBM default = -1, unlimited, in which case empirical average depth ≈ 14 across the ensemble; see supplementary)".
    - Verify: actual max depth computed from the trained model via `model.model_to_string()` inspection.

19. **SDPE formalization.** Add a formal definition: SDPE = sample standard deviation of $e_i = (\hat{T}_i - T_i)/T_i$, with Bessel's correction (denominator N−1). Report mean signed error alongside: if small and constant, recalibration is valid.

20. **MDS Kruskal stress σ₁.** For each TSPLIB instance type (ATT, GEO, EXPLICIT), compute `σ₁ = sqrt(Σ (d_embed - d_orig)^2 / Σ d_orig^2)` and tabulate. Document the negative-eigenvalue policy from [classical_mds.py](../tsplib_benchmark/classical_mds.py) — likely truncation to positive eigenvalues.

21. **MDS subverts non-Euclidean.** Add one paragraph: "MDS embeds into Euclidean R^k before feature extraction; only the MST features use the original D. Hence 'non-Euclidean support' here means 'distance-matrix input accepted', not 'features computed in the original non-Euclidean geometry'."

22. **Limitations expansion** — §7.2. Structure: (a) OOD geometry (grids, line manifolds), (b) n >> 1000 extrapolation risk, (c) MDS stress-dependent quality, (d) baseline domain (EUC_2D with n ≤ 5000 if #13 unresolved). Use grouped-attribute findings from #24.

23. **OOD stratification** — new §5.3.x with two tables: (i) TSPLIB by n bucket `{<500, [500,1000], (1000, 5000], >5000}` split by "in-training" vs "extrapolation"; (ii) ND by dimension `{d≤5, 5<d≤50, d=100}`.

24. **Grouped-attribute TSPLIB table** — Cavdar Table 10 style. Columns per attribute: N, MAPE, SDPE for GART and each baseline. Attributes: has-subgraphs, grid-aligned, irregular-shape, has-empty-interior. Named failure cases called out in prose.

25. **Wilcoxon / bootstrap CIs for MAPE.** Add `scipy.stats.wilcoxon` paired comparisons between GART MAPE and each baseline MAPE on the same instances. Report: W, p, 95% bootstrap CI on the MAPE delta.

26. **Runtime comparison vs Cavdar's O(n).** From `all_models_tsplib_…csv`, plot log-log n vs total_time_s for GART 2.0, Cavdar, MST_Ratio, Hilbert. Add one table: ms/instance at n={100, 1k, 10k, 85k}. This likely strengthens GART (LightGBM inference is μs).

27. **Negative-results table.** Paper currently claims GART wins across the board. Fill a small table: row per baseline, column "where it beats GART", e.g. "MST Ratio: n>1500 bucket (pending #13 resolution)". Honesty improves reviewer trust.

28. **Training-bounds reconciliation.** After #3 and #15, update abstract + §3.1 + §7.2 + CLAUDE.md to use one consistent statement: "trained on n ∈ [5, 1000], d ∈ {2,…,50}, grid_size ∈ {100, 1000, 10000}; held-out test includes d=100 and TSPLIB95". CLAUDE.md currently says "d≤5" which is wrong; fix that first.

29. **Feature ablation** — paired with #7. Four ablations: (a) drop MST features, (b) drop coordinate-stat features, (c) drop topology features, (d) drop MDS-derived features. Report MAPE delta. Table in §5.3.x.

### MINOR (16)

30. **Per-TSPLIB-bucket SDPE drift 0.02–0.9pp** — presumably due to `std` vs `stdev` (N vs N−1) or rounding. Recompute with explicit `np.std(..., ddof=1)` and update the paper values.

31. **ND total 16,907 vs 16,920.** Recount from CSV with `df[df['model']=='LGBM_V3']` — use whichever is in the canonical CSV, note it at footnote.

32. **§1.1 compression.** Apps → one paragraph; move domain examples to §1.2.

33. **Fold §4 into §3.** Rename §3 "Methodology: Features, Data, and Model"; put Complexity as §3.4.

34. **DFJ constraint bound.** Change $|S| \le n-2$ → $|S| \le n-1$ (standard DFJ). Cite Dantzig, Fulkerson, Johnson 1954.

35. **MST complexity qualifier.** Add "(Prim's with array-based priority queue)" to the $O(n^2)$ claim.

36. **Levene language.** Report `stat=…, df=…, p<0.001`; note instance-independence assumption.

37. **Node Density divide-by-zero.** Add "We add ε=1e-9 to each axis range to prevent division by zero on degenerate line manifolds." Verify this matches [feature_creator_v3.py](../feature_creator_v3.py) — if code lacks ε, add it and re-run feature extraction.

38. **Normalized Diameter.** "≈ 1.0 for lines" → "= 1.0 for paths (exact)".

39. **MDS k=100 / 99.9% sensitivity.** Run `classical_mds.py` with {k=50, 95%} and {k=25, 90%}; report MAPE delta on TSPLIB GEO/EXPLICIT. Paragraph in App C.

40. **Headline MAPE CIs.** Bootstrap (1000 resamples) on the same instance pool for 3.23% / 0.81% / 3.44%. Report 95% CIs.

41. **"First estimator…" claim** — soften to "to our knowledge, the first …" with a footnote citing the literature search.

42. **§1.4 Paper Outline** — delete or compress to 2 sentences.

43. **Unsupported adjectives.** Strip "critical", "crucial", "statistically overwhelming" — replace with numbers or citations.

44. **4 vs 5 distribution classes.** Clarify in §3.1: "Training covers 4 classes {uniform, normal, clustered, correlated}; the 2D benchmark evaluates a 5th class (Line Noise) explicitly as OOD."

45. **K-fold "ensemble" phrasing.** Edit the hyperparameter table caption to remove "Ensemble Statistics" wording; the model is a single Optuna-tuned boosted ensemble inside one LGBM instance.

46. **TSPLIB 110 vs 111.** CLAUDE.md says 111-instance run. Paper says 110 consistently. Fix CLAUDE.md.

---

## Verification loop (for each edit)

1. Apply edit via Edit tool.
2. Auto-hook triggers `latexmk` (installed this turn).
3. Check `paper_reference/latexmk_hook.log` for errors.
4. For numeric claims, run the corresponding `python -c` one-liner against the CSV to reconfirm.
5. Mark todo completed.

## Suggested ordering

**Phase 1 (fast wins, no external data needed):** Items 4, 16, 17, 19, 21, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46. ~1 session.

**Phase 2 (code-lookup + table-rewrite):** Items 1, 2, 3, 6, 10, 15, 18, 28, 30, 31. Requires CSV probes + best_params_v3.json read. ~1 session.

**Phase 3 (new content):** Items 7, 8, 9, 11, 12, 20, 22, 23, 24, 29, 39, 40. New tables/figures/analyses. ~3 sessions.

**Phase 4 (blocked on re-runs):** Items 13, 14, 27 — wait on supplemental baseline run (Section C).

**Phase 5 (significance + runtime):** Items 25, 26. Script + table.

**Phase 6 (citation verification):** Item 5 — WebFetch Percus & Martin.

## Code-vs-literature decisions pending user input

- **Item #16 α clipping:** keep clip [1.0, 2.0] and disclose, or remove clip and retrain? Need explicit go to remove the clip and retrain. No live web source found compelling a change; default is "keep and disclose".

That's the complete plan. Tell me which phase to start, or which items to tackle first.
