# Citation Audit — Area_Free_Main.tex

Audit date: 2026-04-17. All 40 unique `\citep`/`\citet` keys used in the TeX file were checked against the bib entry in `references.bib` and against real-world publication records (Google Scholar / DOI resolvers / publisher sites / DBLP).

## Section 1 — Summary

- **Total unique cited keys:** 40
- **VERIFIED:** 35
- **WRONG_METADATA:** 4
- **NOT_FOUND (likely fabricated):** 1

## Section 2 — Issues

| Key | Problem | Correct metadata (if known) | Source |
|---|---|---|---|
| `vinel2015estimation` | NOT_FOUND. No paper with title "On the estimation of the optimal tour length in the Euclidean TSP" by Vinel & Silva in Optimization Letters 9:1567-1583 (2015) appears on Google Scholar, DBLP, Springer, or CrossRef. The DOI `10.1007/s11590-015-0888-5` does not resolve; neighbor DOI `s11590-015-0888-1` belongs to an unrelated humanitarian-logistics paper. Alexander Vinel's DBLP page has no Optimization Letters publication and no TSP-length paper with a coauthor named "Silva" in 2015. His actual TSP tour-length paper is the WSC 2018 one (`vinel2018probability`, already in the bib but uncited). | Likely fabricated; if the paper citing a convex-hull TSP estimator is needed, replace with `vinel2018probability` (Vinel & Silva, WSC 2018) or drop. | https://dblp.org/pid/133/2956.html, https://link.springer.com/search?query=Vinel+TSP |
| `varol2023neural` | WRONG_METADATA. Title, year, and volume wrong. Bib title: "Estimating optimal tour lengths of TSP instances by combining neural networks and domain knowledge." Real title: "Neural Network Estimators for Optimal Tour Lengths of Traveling Salesperson Problem Instances with Arbitrary Node Distributions." Authors correct (Varol, Özener, Albey). Journal (Transportation Science) and DOI (10.1287/trsc.2022.0015) are correct. Published online 2023; volume 58(1):45-66, 2024. | Title: "Neural Network Estimators for Optimal Tour Lengths of Traveling Salesperson Problem Instances with Arbitrary Node Distributions" | https://pubsonline.informs.org/doi/10.1287/trsc.2022.0015 |
| `percus1996` | WRONG_METADATA (title). Bib: "Finite-size corrections to the Euclidean traveling salesman problem." Real title: "Finite Size and Dimensional Dependence in the Euclidean Traveling Salesman Problem." Authors, journal, volume 76, issue 8, pages 1188-1191, year 1996, DOI 10.1103/PhysRevLett.76.1188 all correct. | Title: "Finite Size and Dimensional Dependence in the Euclidean Traveling Salesman Problem" | https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.76.1188 |
| `smithmiles2010` | WRONG_METADATA. No paper by Smith-Miles & van Hemert with journal/volume/pages "Computers & Operations Research, 37(5):850-862, 2010" exists. DBLP's Comp&OR vol. 37 TOC does not list such a paper. The only 2010 Smith-Miles/van Hemert TSP paper is "Understanding TSP Difficulty by Learning from Evolved Instances," LION 2010, LNCS 6073:266-280 (with Lim). A later Smith-Miles Comp&OR paper ("Towards objective measures of algorithm performance across instance space") is Smith-Miles, Baatar, Wreford & Lewis, Comp&OR 45:12-24, 2014 — but that title exactly matches our bib title while authors/venue do not. | Likely intended: Smith-Miles, Baatar, Wreford, Lewis (2014), *Computers & Operations Research*, 45:12-24. Title matches. | https://www.sciencedirect.com/science/article/abs/pii/S0305054813003389 |
| `wang2008` | PARTIAL UNCERTAINTY. Title "Mobile sink routing for data collection in wireless sensor networks," TMC 7(6):756-770, 2008, DOI 10.1109/TMC.2007.70734. Neither DBLP's TMC vol. 7 TOC, direct IEEE search, nor Google Scholar surfaces a paper at exactly that citation. The authors (Wang, Cao, La Porta, Zhang) co-authored "Sensor Relocation in Mobile Sensor Networks" (INFOCOM 2005) but not a 2008 TMC mobile-sink-routing paper with that exact title. The DOI pattern 10.1109/TMC.2007.70734 has the form of a 2007/2008 TMC entry; I could not load the IEEE page (403/418 blocking). **Recommend the author manually verify by fetching the DOI; it may still be correct, but it could not be independently confirmed.** | Uncertain — author should verify DOI resolves to the claimed paper. | https://dblp.org/db/journals/tmc/tmc7.html |

## Section 3 — Verified

aaai2023
agarwala2000fast
akiba2019optuna
bartholdi1982heuristic
bhh1959
canturk2024scalable
carlsson2024upper
cavdar2015distribution
chien1992operational
christofides1976
concorde2006
cook1999
delaunay1934
drake2003
duan2014
finmile2025
freeman1975determining
frontiers2021
halper2011mobile
heldkarp1962
helsgaun2017
holland2017ups
jolliffe2002
karlin2021slightly
karp1972
ke2017lightgbm
kou2022standard
menger1928
numba2015
orourke1998
preparata1977
prim1957
reinelt1991tsplib
scipy2020
sklearn2011

## Section 4 — Recommendations

### Citations to fix before submission

1. **`vinel2015estimation` — DROP or REPLACE.** The paper as cited does not exist. The TeX in Section 4 (baselines) presents it as "a geometric estimator that scales tour cost with the convex hull volume." If a convex-hull-based estimator baseline is needed, the paper is most likely `vinel2018probability` (already in the bib, uncited). If that paper does not actually contain a convex-hull estimator, the baseline equation `L ~ beta_d * V(ConvHull)^(1/d) * n^((d-1)/d)` should be reattributed or dropped from the baseline list.

2. **`varol2023neural` — UPDATE title in bib** to "Neural Network Estimators for Optimal Tour Lengths of Traveling Salesperson Problem Instances with Arbitrary Node Distributions."

3. **`percus1996` — UPDATE title in bib** to "Finite Size and Dimensional Dependence in the Euclidean Traveling Salesman Problem."

4. **`smithmiles2010` — RESOLVE.** The bib entry conflates two works. Either:
   - Change to Smith-Miles, K., van Hemert, J., Lim, X.Y. (2010), "Understanding TSP Difficulty by Learning from Evolved Instances," *Learning and Intelligent Optimization* (LION 2010), LNCS 6073:266-280 — if the claim is about TSP-feature characterisation of instance difficulty; or
   - Change author list to Smith-Miles, Baatar, Wreford, Lewis and year to 2014, vol 45, pp 12-24 — if the exact title "Towards objective measures of algorithm performance across instance space" is what the paper intends to reference.
   Note: this key is cited heavily in the MST-feature table (rows for MST Edge Mean/Std/Skewness/Kurtosis, MST Leaf Ratio, MST Degree). The LION 2010 paper actually defines these TSP features, so the LION citation is almost certainly the correct reference.

5. **`wang2008` — VERIFY.** The author should load the DOI to confirm it resolves to the claimed paper. If it does not, the claim in Section 1 (sensor-network data-collection motivation) is well-supported by related Wang/Cao/La Porta INFOCOM 2005 or other TMC papers; substitute as needed.

### Suggested additions (claims in the paper that lack a citation)

- Section 3.2, second paragraph introducing Prim's and Kruskal's complexity: "Classical algorithms such as Prim's [Prim 1957] and Kruskal's construct the MST in $O(n^2)$ time for complete graphs." — Kruskal's algorithm is uncited. Add Kruskal (1956), "On the shortest spanning subtree of a graph and the traveling salesman problem," *Proc. AMS* 7:48-50.
- Section 4 (baselines), discussion of the Hilbert curve construction: cite the Hilbert-curve reference (Hilbert 1891) alongside `bartholdi1982heuristic`, since the paper refers to "fractal Hilbert curve" but cites only the TSP heuristic that uses it.
- Section 2 / 3 uses the BHH constant β_d. The one-line claim "β_2 ≈ 0.7080 by Percus 1996" is correctly cited, but the constant estimate 0.7124 attributed to "the range of published estimates [percus1996, carlsson2024upper]" is not actually in either of those papers as a point estimate — consider citing Applegate, Bixby, Chvátal & Cook (2006, already in bib as `concorde2006`) or Johnson, McGeoch & Rothberg (1996) whose Monte Carlo estimate is 0.7124.
- Section 6 mentions "Christofides heuristic requires Minimum Weight Perfect Matching" — Christofides 1976 is cited, but the Edmonds blossom algorithm is not. Edmonds (1965), "Paths, trees, and flowers," *Can. J. Math.* 17:449-467, would be appropriate alongside `cook1999`.

### Citations whose support of the claim is weak

- `finmile2025` is a company whitepaper used to motivate the last-mile / fleet-sizing claim. For a peer-reviewed paper, this is acceptable as a trade-press citation but should ideally be paired with a peer-reviewed logistics-industry paper (e.g., `holland2017ups`, already cited).
- `frontiers2021` (Santa Claus Challenge) is cited for "need for fast TSP on ≥1M-node instances." It supports the scale claim but is a single-instance benchmark paper; consider supplementing with a survey if tightening.

Audit completed — see Section 1 summary.
