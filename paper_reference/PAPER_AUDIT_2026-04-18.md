# Paper Audit — Area_Free_Main.tex vs. Cavdar 2015 (reviewer)

Date: 2026-04-18. Five parallel review agents (Cavdar-comparison, code-claim, benchmark-numbers, math-rigor, gaps).

## CRITICAL — must fix before sending to Cavdar

1. **Hyperparameter table is fabricated.** Sec 3.2 Table `tab:hyperparams` (lines 313–325) lists `num_leaves 258/300, lr 0.0180, feature_fraction 0.416, bagging 0.561/3, min_child 45, L1 0.417, L2 4.58e-6, n_estimators 4431`. Actual `lgbm_model_v3/best_params_v3.json`: `lr 0.01284, num_leaves 142, L1 8.49e-8, L2 1.43, feature_fraction 0.657, bagging 0.432/5, min_child 22`. Optuna search was `num_leaves [64,512]` (not capped at 300). **Every numeric value disagrees with committed model.**

2. **Feature count mismatch.** Paper repeats "29 features" (abstract 109, Sec 3 ~230, conclusion 771). `feature_creator_v3.py` writes 28 keys; 29 only if `grid_size` is counted, which memory and CLAUDE.md say is unused. No `FEATURE_NAMES` constant pins the count. Enumerate features or correct the count.

3. **Training d-range wrong.** Sec 3.1 Dataset Generation (~293) says `d ∈ {2..50}, 17 values` with "d=100 in appendix". Code `Dataset_Generator.py:45` has 18 values including d=100 in the training set. Line 779 "extrapolation to d=100" is false — d=100 is in training.

4. **GART 2.0 vs 3.0 naming.** 56 occurrences of "GART 2.0" in paper; MEMORY.md canonical is "GART 3.0 = LGBM V3 in code"; CLAUDE.md says 2.0. Pick one and global-replace. Cavdar will notice.

5. **BHH constant attribution error.** Sec 5.2.1 line 357 attributes β₂≈0.7080 to Percus 1996; Percus & Martin (1996) reported ≈0.7124. Either the number or the citation is wrong. Cavdar knows this constant.

6. **No reproducibility block.** Paper has no random-seed, Optuna trial count (100 in code), TPE method, train/val/test split ratios, early-stopping, or CV protocol stated. Reviewer will ask.

7. **No SHAP / theoretical bridge.** `shap_analyzer.py` exists in repo, never cited. Cavdar derives his model as sample-statistics BHH — GART has no analogous interpretability story. Add SHAP top-5 gain table + one paragraph on why MST-topology features matter.

## MAJOR — reviewer will push back

8. **Missing orienting Table 1** (Cavdar-style ✓ matrix: accurate / fast / distribution-free / d>2 / TSPLIB). Insert near line 151.

9. **Missing baselines.** Kwon et al. (1995) and Daganzo (1984a/b TSP+CVRP) and del Castillo (1998) absent. Cavdar benchmarks all of them. Add Kwon (simplest to implement) at minimum, justify Daganzo omission if kept out.

10. **Headline metric inconsistency.** Abstract leads with MAPE, precision-first framing needs SDPE. Pick one headline; demote the other.

11. **Results section bloat.** Sec 5.3 is 295 lines (~35% of paper). Move per-bucket tables to appendix; keep 3 headline tables (2D, ND, TSPLIB) in main text.

12. **Sec 6 MDS as "Application" reads as afterthought** while being an abstract headline. Merge into Sec 5 as the culminating real-data subsection.

13. **TSPLIB baseline N wrong.** Table `tab:tsplib_by_size` claims 78 EUC_2D for all models; CSV has 71 for non-GART baselines (12 in n>1500 bucket, not 19). TOTAL row uses wrong denominator for baselines. Fix per-bucket N for baselines and re-average MAPE/SDPE.

14. **MST Ratio n>1500 MAPE wrong.** Line 713: "MST Ratio beats GART 2.93% vs 4.47%". CSV gives MST_Ratio n>1500 MAPE = 3.76, not 2.93. Recompute + rewrite discussion.

15. **Training instance count 51,818 unverifiable.** Cartesian product from generator is 110,700. No filter/sampling rule in code. Either provide the filter script or recount from the actual training CSV.

16. **α clipped to [1.0, 2.0] not disclosed.** `LGBM_Alpha_Model_V3.py:58`. Material for ND where α can exceed 2. Add to methods.

17. **"Distribution-free" is overreach.** Paper trains on 4 named distributions. It's *distribution-agnostic at inference* via sample-stat features — reframe to avoid Cavdar's own stricter usage of the term.

18. **Inference complexity error.** Sec 4.2 line 336 says "⌈log₂ L⌉ = 9 comparisons". LightGBM leaf-wise trees are unbalanced; worst-case depth is L−1. State "average depth ≈ log₂ L empirically" or bound by max_depth.

19. **SDPE definition incomplete.** Sec 5 line 346 — no N vs N−1 denominator specified; no mean signed error reported alongside to support the recalibration claim. Add both.

20. **MDS stress not quantified.** Sec 6 claims "low stress" without reporting Kruskal stress σ₁ per TSPLIB instance type. 99.9% eigenvalue-variance proxy breaks when Gram matrix has negative eigenvalues; negative-eigenvalue policy undocumented.

21. **MDS subverts "non-Euclidean" framing.** Features are extracted from Euclidean embedding; only MST uses original D. State this explicitly — otherwise readers think features work on raw non-metric D.

22. **Limitations throwaway paragraph.** Lines 777–781: expand into structured §7.2 with per-spatial-class breakdown (grid/subgraphs/empty-interior/line-manifold).

23. **No OOD stratification.** TSPLIB up to n=85,900 is pure extrapolation but not split out. Add a "pure extrapolation" subsection showing GART n>1000 performance.

24. **No TSPLIB grouped-attribute analysis.** Cavdar groups by subgraphs/grid/irregular/empty-interior (Table 10). GART only buckets by size. Add grouped table citing named failure cases (ts225 grid, d657 subgraphs, etc.).

25. **Statistical tests only on SDPE.** Add paired Wilcoxon / bootstrap CIs on MAPE deltas vs baselines.

26. **Runtime comparison missing.** Cavdar's models are O(n). GART inference vs Cavdar on the same machine/instances isn't plotted. Add asymptotic scaling plot + ms/instance side-by-side.

27. **No negative-results table.** List instance types where a simpler baseline beats GART (e.g., MST Ratio n>1500 per #14). Cavdar-style honesty.

28. **Training provenance conflict.** Abstract says n ≤ 1000; CLAUDE.md says d ≤ 5; Appendix B says d ≤ 50; code uses d ≤ 100. Reconcile.

29. **No ablation.** One-paragraph ablation dropping MST vs coordinate features. Pair with SHAP (#7).

## MINOR — polish

- SDPE values in TSPLIB buckets off 0.02–0.9 pp from CSV (items C#2,4,5,6).
- ND total 16,907 vs CSV 16,920 (13-row delta in n∈[501,1000] bucket).
- §1.1 intro application front-loading (UPS/bio/sensor) — compress to one paragraph.
- §4 placement between methodology and evaluation is awkward — fold into §3.
- DFJ subtour constraint upper bound $|S|\le n-2$ should be $n-1$ or $\lfloor n/2\rfloor$.
- MST complexity claim O(n²) requires "Prim's with array priority queue" qualifier.
- "statistically overwhelming" + p≈0 — report test statistic, df, and note Levene independence assumption.
- Node Density divide-by-zero undisclosed for near-linear manifolds; add ε handling note.
- Normalized Diameter "≈ 1.0 for lines" is exactly 1.0 for a path.
- MDS k=100 cap and 99.9% threshold need a 95%/k=50 sensitivity check.
- No bootstrap CIs on headline 3.23% / 0.81% / 3.44% MAPE numbers.
- "First estimator across every metric distance type" (line 775) unsupported — soften.
- §1.4 Paper Outline paragraph redundant — delete or compress to 2 sentences.
- Unsupported adjectives: "critical" (135), "crucial" (137), "statistically overwhelming" (775) — quantify or cite.
- 4 vs 5 distribution classes: training=4, eval=5 — clarify explicitly.
- K-fold "ensemble" phrasing in hyperparameter table misleading — single Optuna-tuned LGBM, no KFold.
- TSPLIB 110 vs 111: paper consistent at 110; CLAUDE.md says 111 — update memory/CLAUDE.

## Items that check out (do not touch)

- α = optimal_cost / mst_total_length (code matches paper).
- Concorde + LKH-3 dual solver keeping shorter (matches Sec 3.1).
- Classical MDS in `tsplib_benchmark/classical_mds.py` with `variance_threshold` + `max_dim=100` (matches Sec 6 & App C).
- Delaunay-based MST for 2D.
- Optuna 100 TPE trials minimizing RMSE (matches line 307).
- 2D diverse benchmark numbers (Table `tab:2d_by_size`) — verified exact to 0.01.
- ND per-bucket MAPE/SDPE/N (Table `tab:nd_by_size`/`tab:nd_by_dim`) — match CSV exactly except for 13-row total discrepancy.
- TSPLIB GART 2.0 totals: MAPE 3.44 / MEDIAN 3.50 / SDPE 2.37 ✓ (only 0.02 SDPE drift in some buckets).
