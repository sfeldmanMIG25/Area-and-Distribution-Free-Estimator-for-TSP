"""Rewrite Area_Free_Main.tex for the withdrawal of Daganzo, Chien, Kwon and Cavdar_region.

Exact-string replacement over ``read_text``/``write_text``. No regular
expression touches the manuscript body: a previous heredoc-based patch corrupted
``\\ref`` into a literal CR + ``ef{`` four times, so every edit here is a literal
pair asserted to occur exactly once.

Run once. It is idempotent only in the sense that a second run aborts on the
first missing anchor rather than half-applying.
"""

from __future__ import annotations

import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

EDITS: list[tuple[str, str]] = []


def edit(old: str, new: str) -> None:
    EDITS.append((old, new))


# --------------------------------------------------------------------------
# Abstract and contributions: the baseline roster drops from ten to seven.
# --------------------------------------------------------------------------
edit("the lowest of a ten-baseline roster on all four strata",
     "the lowest of a seven-baseline roster on all four strata")

edit(
    "We benchmark it against ten baselines: five classical estimators, three "
    "constant multiples of $L_{\\mathrm{MST}}$ of which one is calibrated on the "
    "training split, the GART 1.0 predecessor, and a space-filling-curve tour "
    "construction. Each classical estimator is scored twice, once on the full "
    "benchmarks with the convex hull standing in for its region measure and once "
    "on the subset where its published assumptions hold, supplied with the exact "
    "sampling region and gated to its fitted node-count range.",
    "We benchmark it against seven baselines: two classical estimators, three "
    "constant multiples of $L_{\\mathrm{MST}}$ of which one is calibrated on the "
    "training split, the GART 1.0 predecessor, and a space-filling-curve tour "
    "construction. Both classical estimators are scored on the full benchmarks "
    "and again on the i.i.d.\\ uniform subset where their published assumptions "
    "hold, BHH there receiving the exact sampling region its theorem names. "
    "Three further classical estimators are surveyed in "
    "Section~\\ref{subsec:related} and scored nowhere, because their primaries "
    "are paywalled and we decline to benchmark against constants we have not "
    "read in the original.")

edit("Against the ten baselines GART 2.0 has the lowest aggregate MAPE and SDPE",
     "Against the seven baselines GART 2.0 has the lowest aggregate MAPE and SDPE")

# --------------------------------------------------------------------------
# Section 1.2, literature review: the three works stay, their numbers go.
# --------------------------------------------------------------------------
edit(
    "Section~\\ref{sec:evaluation} evaluates each of these source models "
    "numerically, on the full benchmarks and on the subset where its published "
    "assumptions hold (Table~\\ref{tab:benchmark_models}).",
    "Of these, Section~\\ref{sec:evaluation} scores only BHH and "
    "\\citet{cavdar2015distribution} numerically, on the full benchmarks and on "
    "the subset where their published assumptions hold "
    "(Table~\\ref{tab:benchmark_models}). We survey "
    "\\citet{chien1992operational}, \\citet{daganzo1984b} and "
    "\\citet{kwon1995tsp} without benchmarking against them: all three "
    "formulations sit behind paywalls we could not obtain, and we decline to "
    "publish numbers produced by constants we have not read in their original.")

# --------------------------------------------------------------------------
# Section 4.1, metrics: the SDPE-is-blind-to-bias example was Daganzo's.
# --------------------------------------------------------------------------
edit(
    "Daganzo's strip constant carries a $+15.4$\\% offset on uniform instances "
    "with only 9.8\\% SDPE,",
    "BHH given the exact sampling region carries a $-8.65$\\% offset on uniform "
    "instances with only 7.76\\% SDPE,")

# --------------------------------------------------------------------------
# Section 4.2, the estimator declarations.
# --------------------------------------------------------------------------
edit(
    "\\citet{daganzo1984b} gives $\\hat L=0.9\\sqrt{nA}$ for a compact Euclidean "
    "zone of area $A$; this constant approximates the expected length of a "
    "strip-strategy tour, which its author reports as suboptimal, so it is biased "
    "upward relative to optimal tours. \\citet{chien1992operational} fits $\\hat "
    "L=2.1\\bar r+0.67\\sqrt{nR}$ on $n=5$--30, where $\\bar r$ is the mean "
    "node-to-depot distance and $R$ is the area of the smallest rectangle covering "
    "the nodes. \\citet{kwon1995tsp} fits $\\hat "
    "L=[0.83-0.0011(n+1)+1.11S/(n+1)]\\sqrt{nA}$ on $n=10$--80 over rectangles "
    "whose shape factor $S$, the ratio of the longer to the shorter side, lies in "
    "$[1,8]$. \\citet{cavdar2015distribution} gives",
    "\\citet{cavdar2015distribution} gives")

edit(
    "where $\\mathrm{stdev}_j$ is the standard deviation of the raw coordinates on "
    "axis $j$, and $\\bar c_j$ and $\\mathrm{cstdev}_j$ are the mean and standard "
    "deviation of the absolute distances from the nodes to the region's midpoint "
    "line on axis $j$. Appendix~\\ref{app:bench_details} records the transcription "
    "sources and the two quantities we could not verify.",
    "where $A$ is the area of the rectangle covering the nodes, "
    "$\\mathrm{stdev}_j$ is the standard deviation of the raw coordinates on axis "
    "$j$, and $\\bar c_j$ and $\\mathrm{cstdev}_j$ are the mean and standard "
    "deviation of the absolute distances from the nodes to that rectangle's "
    "midpoint line on axis $j$. Both forms are transcribed from a primary "
    "document we read: the BHH constants from the sources cited above, and the "
    "\\c{C}avdar--Sokol model together with both of its constant sets from "
    "\\citet{cavdar2014dissertation}. Appendix~\\ref{app:bench_details} records "
    "what each estimator receives, and why three further classical estimators "
    "are surveyed but not scored.")

edit(
    "\\paragraph{Region measure and matched domains.} Every synthetic generator "
    "draws coordinates inside $[0,G]^d$ for a coordinate-grid side length $G$, so "
    "the sampling region measure is $G^d$ exactly. TSPLIB defines no sampling "
    "region, so there the classical estimators receive the convex hull of the "
    "instance as a stated plug-in. On the synthetic benchmarks we report both "
    "forms as separate rows, because the region measure is the source-faithful "
    "input only where the generator is uniform on that region: a near-collinear "
    "point set has one-dimensional effective support, and handing an area-based "
    "estimator $G^2$ inflates its prediction by an amount that describes the data "
    "and not the estimator. Section~\\ref{subsec:matched_domain} therefore reports "
    "each classical estimator on the subset where its published assumptions hold, "
    "and the full-benchmark tables report the plug-in form. Neither assignment is "
    "a free parameter: the region measure is $G^d$ by construction of the "
    "generator, and the matched domains are pinned by the sources themselves at "
    "i.i.d.\\ uniform sampling for BHH and Daganzo, $n\\in[5,30]$ for Chien, and "
    "$n\\in[10,80]$ with shape factor $S\\in[1,8]$ for Kwon--Golden--Wasil, so a "
    "reader can reconstruct every subset from the published ranges without "
    "reference to our results.",
    "\\paragraph{Region measure and matched domains.} Every synthetic generator "
    "draws coordinates inside $[0,G]^d$ for a coordinate-grid side length $G$, so "
    "the sampling region measure is $G^d$ exactly. BHH is the only estimator here "
    "whose source names that measure, and on the synthetic benchmarks we report "
    "it as two rows, one given $G^d$ and one given the convex hull of the "
    "realized sample. Both are reported because the region measure is the "
    "source-faithful input only where the generator is uniform on that region: a "
    "near-collinear point set has one-dimensional effective support, and handing "
    "an area-based estimator $G^2$ inflates its prediction by an amount that "
    "describes the data and not the estimator. TSPLIB defines no sampling region "
    "at all, so there BHH receives the convex hull as a stated plug-in. "
    "\\c{C}avdar--Sokol takes no region input: its $A$ is the area of the "
    "rectangle covering the nodes, a statistic of the sample, so it has a single "
    "row on every benchmark and supplying it $G^2$ would be our construction "
    "rather than its authors'. Section~\\ref{subsec:matched_domain} reports both "
    "estimators on the subset where their published assumptions hold. Neither "
    "assignment is a free parameter: the region measure is $G^d$ by construction "
    "of the generator, and the matched domain is pinned by the source itself at "
    "i.i.d.\\ uniform sampling, so a reader can reconstruct the subset without "
    "reference to our results.")

edit("We compare GART 2.0 against ten baselines in four families:",
     "We compare GART 2.0 against seven baselines in four families:")

edit(
    "A source-to-code audit found that the repository had been feeding the "
    "classical estimators the convex hull of the realized sample in place of the "
    "sampling-region measure their sources require, and that the substitution "
    "rather than the estimators produced their reported errors. We repaired the "
    "implementations instead of dropping the comparison, and we state for every "
    "row which inputs it received.",
    "A source-to-code audit found that the repository had been feeding the "
    "classical estimators the convex hull of the realized sample in place of the "
    "quantity their sources require, and that the substitution rather than the "
    "estimators produced their reported errors. Where we could read the primary "
    "we repaired the implementation instead of dropping the comparison, and we "
    "state for every row which inputs it received. Where we could not, we "
    "withdrew the estimator: the same audit established that our Daganzo, Chien "
    "and Kwon--Golden--Wasil coefficients had been transcribed from a secondary "
    "source, so those three are surveyed in Section~\\ref{subsec:related} and "
    "scored nowhere in this paper.")

# --------------------------------------------------------------------------
# tab:benchmark_models -- three rows out.
# --------------------------------------------------------------------------
edit(
    "Daganzo strip tour & $0.9\\sqrt{nA}$ & Compact zone; estimates a strip tour "
    "& $O(nd)$ & \\citet{daganzo1984b} \\\\\n"
    "Chien & $2.1\\,\\bar r + 0.67\\sqrt{nR}$, $R$ the covering-rectangle area & "
    "$n=5$--30; corner depot & $O(nd)$ & \\citet{chien1992operational} \\\\\n"
    "Kwon--Golden--Wasil & $[0.83-0.0011(n{+}1)+1.11S/(n{+}1)]\\sqrt{nA}$ & "
    "$n=10$--80; rectangles $S\\in[1,8]$ & $O(nd)$ & \\citet{kwon1995tsp} \\\\\n"
    "\\c{C}avdar--Sokol &",
    "\\c{C}avdar--Sokol &")

edit(
    "\\c{C}avdar--Sokol & $2.791\\sqrt{n\\,\\mathrm{cstdev}_x\\mathrm{cstdev}_y}"
    "+0.2669\\sqrt{n\\,\\mathrm{stdev}_x\\mathrm{stdev}_y A/(\\bar c_x\\bar c_y)}$ "
    "& Axis-aligned rectangle & $O(nd)$ & \\citet{cavdar2015distribution} \\\\",
    "\\c{C}avdar--Sokol & $2.791\\sqrt{n\\,\\mathrm{cstdev}_x\\mathrm{cstdev}_y}"
    "+0.2669\\sqrt{n\\,\\mathrm{stdev}_x\\mathrm{stdev}_y A/(\\bar c_x\\bar c_y)}$ "
    "& Minimum-area rectangle; $n\\ge100$ correction & $O(nd)$ & "
    "\\citet{cavdar2014dissertation} \\\\")

# --------------------------------------------------------------------------
# Section 4.4, TSPLIB results.
# --------------------------------------------------------------------------
edit(
    "The classical estimators are far behind on this benchmark, and the reason is "
    "the same one that motivates the paper. \\c{C}avdar--Sokol obtains 23.18\\% "
    "MAPE, BHH 25.14\\%, Chien 30.38\\% when extrapolated past its fitted "
    "$n\\le30$, Daganzo 44.45\\%, and Kwon--Golden--Wasil 54.44\\% when "
    "extrapolated past its fitted $n\\le80$. TSPLIB instances are drawn from real "
    "geography and circuit layouts rather than from a uniform density on a known "
    "region, so every area-based estimator is applied outside the conditions it "
    "was derived under. Chien's published range excludes all 78 instances "
    "outright, since the smallest has $n=51$ against a fitted ceiling of 30. These "
    "rows measure the cost of that mismatch, not the quality of the cited methods, "
    "which is why Section~\\ref{subsec:matched_domain} evaluates each of them "
    "where its assumptions hold.",
    "Both classical estimators are far behind on this benchmark, and the reason "
    "is the same one that motivates the paper. \\c{C}avdar--Sokol obtains "
    "23.87\\% MAPE and BHH 25.14\\%. TSPLIB instances are drawn from real "
    "geography and circuit layouts rather than from a uniform density on a known "
    "region, so an area-based estimator is applied outside the conditions it was "
    "derived under: BHH receives a convex hull in place of a sampling region that "
    "does not exist, and \\c{C}avdar--Sokol is evaluated on graphs that carry no "
    "node at the corners of their covering rectangle, which every one of its "
    "training graphs did. Both overpredict, by $+13.66$\\% and $+20.07$\\% "
    "respectively. These rows measure the cost of that mismatch, not the quality "
    "of the cited methods, which is why Section~\\ref{subsec:matched_domain} "
    "evaluates both where their assumptions hold.")

# --------------------------------------------------------------------------
# Section 4.5, the matched-domain re-evaluation.
# --------------------------------------------------------------------------
edit(
    "The full-benchmark rows above apply each classical estimator far outside the "
    "conditions it was derived under, so they cannot support a claim about the "
    "methods themselves. This section evaluates each one where its published "
    "assumptions hold. The 210 instances of the \\texttt{random} generator, the "
    "Uniform member of the Isotropic class of Table~\\ref{tab:dataset_counts}, are "
    "drawn i.i.d.\\ uniformly on $[0,G]^2$, which is exactly the sampling model "
    "BHH, Daganzo, Chien, and Kwon--Golden--Wasil assume, and the region measure "
    "$A=G^2$ is known exactly. Chien and Kwon are further restricted to their "
    "fitted node counts, $n\\in[5,30]$ and $n\\in[10,80]$, leaving 50 and 80 "
    "instances. Table~\\ref{tab:classical} reports both panels.",
    "The full-benchmark rows above apply both classical estimators outside the "
    "conditions they were derived under, so they cannot support a claim about the "
    "methods themselves. This section evaluates each one where its published "
    "assumptions hold. The 210 instances of the \\texttt{random} generator, the "
    "Uniform member of the Isotropic class of Table~\\ref{tab:dataset_counts}, are "
    "drawn i.i.d.\\ uniformly on $[0,G]^2$, which is exactly the sampling model "
    "BHH assumes, and the region measure $A=G^2$ is known exactly. "
    "\\c{C}avdar--Sokol carries the same single row here that it carries above, "
    "because it consumes no region: between the two panels its instance set "
    "changes and nothing else, which is what makes the pair readable as a domain "
    "effect. Table~\\ref{tab:classical} reports both panels.")

edit(
    "Each estimator's error falls substantially on its matched domain. Three "
    "things change at once between the two panels: the region input, the instance "
    "set, and (for Chien and Kwon) the node-count gate. These figures are "
    "therefore the combined effect and not the isolated value of the region input. "
    "BHH falls from 23.80\\% MAPE on the full 2D benchmark to 8.91\\% here, "
    "\\c{C}avdar--Sokol from 25.54\\% to 10.81\\%, and Kwon--Golden--Wasil from "
    "41.72\\% extrapolated to 5.41\\% on its own range with a mean signed error of "
    "$-0.05$\\%. Kwon--Golden--Wasil is the strongest classical estimator in this "
    "study, and it is unbiased to within 0.05 percentage points where it was "
    "fitted. Its shape term is untested here: every one of these instances is "
    "square, so $S=1$ exactly, an endpoint of the $S\\in[1,8]$ design it was fitted "
    "over. Daganzo overpredicts by $+15.40$\\%, which reproduces the accuracy "
    "\\citet{delcastillo1999} records for the strip strategy on non-elongated "
    "zones and confirms that the row measures the published method and not a "
    "broken implementation. Chien remains the weakest at 18.54\\% with a "
    "$+17.90$\\% bias.",
    "Both estimators' errors fall substantially on the matched domain. BHH falls "
    "from 23.80\\% MAPE on the full 2D benchmark to 8.91\\% here. Two things "
    "change at once between its two panels, the region input and the instance "
    "set, so that figure is the combined effect and not the isolated value of the "
    "region input. What is left is a systematic underprediction, $-8.65$\\% "
    "signed against 7.76\\% SDPE, which is what an asymptotic constant evaluated "
    "at $n$ as low as 5 should do. \\c{C}avdar--Sokol falls from 18.24\\% to "
    "8.16\\%, and here only the instance set changed, so that figure is a clean "
    "domain effect. It is the strongest classical estimator in this study on both "
    "aggregate metrics, and it is also the least even: at 14.28\\% SDPE its "
    "dispersion is the widest in the panel, and its median absolute error of "
    "2.84\\% sits at under a third of its mean, so it is close on most uniform "
    "instances and badly wrong on a few.")

edit(
    "GART 2.0 beats every classical estimator on every matched domain, on both "
    "metrics, by a smaller and more credible margin than the full-benchmark tables "
    "suggest. On the 80 Kwon-domain instances it obtains 2.04\\% MAPE against "
    "Kwon's 5.41\\%; on the 50 Chien-domain instances 2.46\\% against Chien's "
    "18.54\\%; and on all 210 uniform instances 1.31\\% against BHH's 8.91\\% and "
    "the $\\alpha=1$ floor's 15.60\\%. The extended-block ablation is lower than "
    "GART 2.0 on both metrics on all three panels, 1.24\\%/1.90\\%, "
    "1.97\\%/2.45\\% and 2.19\\%/3.04\\%, and none of those three paired "
    "differences resolves: the Wilcoxon $p$ values are 0.11, 0.48 and 0.11. Sets "
    "of 50 to 210 instances reproduce the sign of the 2D ordering and cannot "
    "settle it, and none of it bears on the comparison against the classical "
    "estimators, which is unchanged. A factor of 2.7 over Kwon--Golden--Wasil on "
    "its home ground states what this model adds over the classical literature in "
    "the plane; the tenfold gaps in the full-benchmark table describe domain "
    "mismatch instead.",
    "GART 2.0 beats both classical estimators on the matched domain, on both "
    "metrics, by a smaller and more credible margin than the full-benchmark tables "
    "suggest. On the 210 uniform instances it obtains 1.31\\% MAPE against "
    "\\c{C}avdar--Sokol's 8.16\\%, BHH's 8.91\\% and the $\\alpha=1$ floor's "
    "15.60\\%. The extended-block ablation is lower than GART 2.0 on both metrics "
    "on this panel, 1.23\\% against 1.31\\% on MAPE and 1.90\\% against 2.07\\% on "
    "SDPE, and that paired difference does not resolve: the Wilcoxon $p$ value is "
    "0.09. A set of 210 instances reproduces the sign of the 2D ordering and "
    "cannot settle it, and none of it bears on the comparison against the "
    "classical estimators, which is unchanged. A factor of 6.2 over "
    "\\c{C}avdar--Sokol on i.i.d.\\ uniform draws states what this model adds over "
    "the classical literature in the plane; the wider gaps in the full-benchmark "
    "table describe domain mismatch instead.")

# --------------------------------------------------------------------------
# Section 4.7, timing.
# --------------------------------------------------------------------------
edit(
    "The five classical estimators are a convex hull plus closed-form arithmetic "
    "and cost an order of magnitude less: on the same protocol and the same 78 "
    "instances they take 0.615--0.792~ms, so GART 2.0 costs 7.73 to 9.96 times "
    "what they do.",
    "The two classical estimators are a convex hull plus closed-form arithmetic "
    "and cost an order of magnitude less: on the same protocol and the same 78 "
    "instances they take 0.645--0.880~ms, so GART 2.0 costs 6.96 to 9.49 times "
    "what they do.")

edit("and its multiple over the classical estimators widens to 16.9--21.0.",
     "and its multiple over the classical estimators widens to 13.1--20.2.")

edit(
    "Seven is the roster of this table and not the size of the timed field: "
    "fourteen estimators carry a measurement on this protocol,",
    "Seven is the roster of this table and not the size of the timed field: "
    "eleven estimators carry a measurement on this protocol,")

# --------------------------------------------------------------------------
# Section 7, conclusion.
# --------------------------------------------------------------------------
edit(
    "Against a baseline set of five classical estimators given their published "
    "inputs, three constant-ratio references, the GART 1.0 predecessor and a "
    "space-filling-curve construction,",
    "Against a baseline set of two classical estimators given their published "
    "inputs, three constant-ratio references, the GART 1.0 predecessor and a "
    "space-filling-curve construction,")

edit(
    "a factor of 2.7 over Kwon--Golden--Wasil on the planar uniform instances "
    "where that regression was fitted,",
    "a factor of 6.2 over \\c{C}avdar--Sokol on the planar i.i.d.\\ uniform "
    "instances,")

edit(
    "6.12~ms per TSPLIB EUC\\_2D instance against 0.615--0.792~ms for the five "
    "classical estimators",
    "6.12~ms per TSPLIB EUC\\_2D instance against 0.645--0.880~ms for the two "
    "classical estimators")

# --------------------------------------------------------------------------
# Appendix: per-estimator provenance.
# --------------------------------------------------------------------------
edit(
    "This appendix records what each classical estimator receives, what we "
    "repaired, and what we could not verify.",
    "This appendix records what each scored classical estimator receives, what we "
    "repaired, and why three further estimators are surveyed in the literature "
    "review and scored nowhere.")

edit(
    "\\paragraph{Daganzo.} \\citet{daganzo1984b} constructs a strip-strategy tour "
    "and states that the resulting tours are suboptimal. The compact-zone "
    "Euclidean coefficient is $0.9$, and \\citet{delcastillo1999} records the "
    "strategy as producing tours roughly 15\\% longer than the shortest for "
    "roughly uniform points in a non-elongated zone. Our measured mean signed "
    "error of $+15.4$\\% on square uniform regions reproduces that stated "
    "accuracy, so the positive bias is a property of the method and not of the "
    "reimplementation. The coefficient $0.57$ that appears in some secondary "
    "literature is the local-travel term of the capacitated VRP approximation "
    "$2\\bar rn/C+0.57\\sqrt{nA}$ in \\citet{daganzo1984a}, valid for $C>6$ and "
    "$n>4C^2$; it is not a TSP coefficient. An earlier revision of our code used "
    "$0.57$.",
    "\\paragraph{Withdrawn: Daganzo, Chien, and Kwon--Golden--Wasil.} Earlier "
    "revisions of this paper scored all three. They are withdrawn, and the "
    "reason is provenance rather than performance. Their equations reached our "
    "code through the literature review in \\citet{figliozzi2009planning}, not "
    "through the articles themselves: \\citet{daganzo1984b}, "
    "\\citet{chien1992operational} and \\citet{kwon1995tsp} are paywalled, a "
    "direct check of each DOI returned no open-access location, and we could not "
    "obtain them. Nor is the secondary record self-consistent. For Chien, "
    "\\citet{cavdar2015distribution} give a Daganzo-form coefficient of $0.69$ "
    "and \\citet{choi2021adjustment} give $0.88$, against the $0.67$ implied by "
    "the $2.1\\bar r+0.67\\sqrt{nR}$ we had adopted; for "
    "Kwon--Golden--Wasil, \\citet{cavdar2015distribution} render the bracket with "
    "a plain $n$ and four-significant-figure coefficients where our transcription "
    "carried $(n{+}1)$ and two. Choosing between those renderings requires the "
    "primaries, which is exactly what we lack, and reporting either would have "
    "repeated the defect this appendix exists to disclose. We therefore print no "
    "number for any of the three. Section~\\ref{subsec:related} still surveys "
    "them: they are part of the record this work sits in, and omitting them from "
    "the survey would misdescribe the field. Two consequences are worth stating "
    "plainly. The strongest classical estimator this paper previously reported "
    "was Kwon--Golden--Wasil on its own fitted node range, so withdrawing it "
    "removes our most demanding classical comparator; and the surviving pair, "
    "BHH and \\c{C}avdar--Sokol, are the two whose primary documents we hold.")

edit(
    "\\paragraph{Chien.} The correct reference is \\citet{chien1992operational}, "
    "\\emph{Computers \\& Operations Research} 19(6):469--478; an earlier revision "
    "of our code cited a nonexistent \\emph{Transportation Science} article. The "
    "best-fitting published model is $2.1\\bar r+0.67\\sqrt{nR}$ with reported "
    "$R^2=0.99$ and MAPE 6.9\\%, where $R$ is the area of the smallest rectangle "
    "covering the customers, a realized-sample statistic rather than a service "
    "region. It was fitted on exact optima over 16 region shapes at $n=5$--30 with "
    "the depot fixed at the origin, the lower-left corner of the rectangular area. "
    "We adopt that published depot convention rather than selecting one after "
    "seeing results: for a generator supported on $[0,G]^2$ the origin is the "
    "lower-left corner, and for instances with no service region we use the corner "
    "of the coordinate bounding box. Chien reports seven regression variants and "
    "we could not obtain the other six, so this row rests on one transcription of "
    "one variant. Published renderings of ``Chien's model'' disagree: "
    "\\citet{cavdar2015distribution} give a Daganzo-form coefficient of $0.69$ and "
    "\\citet{choi2021adjustment} give $0.88$, against the $2.1\\bar "
    "r+0.67\\sqrt{nR}$ we adopt from \\citet{figliozzi2009planning} because it is "
    "the only rendering carrying the mean-depot-distance term.\n\n",
    "")

edit(
    "\\paragraph{Kwon--Golden--Wasil.} The published fits use rectangular service "
    "regions with length/width ratio $S\\in[1,8]$, $n=10$--80, exact optima, an "
    "intercept forced through zero, and a stated depot convention "
    "\\citep{kwon1995tsp}. We evaluate the depot-at-origin variant that carries no "
    "depot term, $[0.83-0.0011(n{+}1)+1.11S/(n{+}1)]\\sqrt{nA}$, because our "
    "benchmarks define no depot; its coefficients were nonetheless fitted under a "
    "corner-depot design. The bracket's linear term drives the estimator negative "
    "near $n\\approx750$, so the equation is numerically invalid there, which is "
    "one reason we gate it to its published range. The source rendering available "
    "to us is ambiguous between $(n{-}1)$ and $(n{+}1)$ in the linear term; we "
    "implement $(n{+}1)$, and the two readings differ by under 0.3\\% in predicted "
    "length over $n\\in[10,80]$. \\citet{cavdar2015distribution} render the same "
    "model with a plain $n$ and four-significant-figure coefficients, a third "
    "reading that differs by no more, and report that it returns negative or very "
    "low values at large $n$, which is the defect the node-count gate avoids.\n\n",
    "")

edit(
    "\\paragraph{\\c{C}avdar--Sokol.} The estimator's area term is the area of the "
    "rectangular space, and $\\bar c_j$ and $\\mathrm{cstdev}_j$ are the mean and "
    "standard deviation of the absolute distances to that region's midpoint line "
    "on axis $j$, not to the sample mean \\citep{cavdar2015distribution}. The "
    "divisor $\\bar c_x\\bar c_y$ sits inside the radical; placing it outside "
    "leaves the second term dimensionless. An earlier revision of our code "
    "substituted the convex hull for the area and measured dispersion about the "
    "sample midrange; both are corrected. Two further features of the published "
    "method are applied, so every \\c{C}avdar--Sokol row in this paper is the "
    "corrected, rotated model. The first is the finite-$n$ correction "
    "$0.9325e^{0.00005298n}-0.2972e^{-0.01452n}$, fitted over "
    "$n\\in\\{100,125,\\dots,975\\}$ at $R^2=0.9867$ \\citep{cavdar2014dissertation}, "
    "which multiplies the estimate and inflates it by roughly a seventh at "
    "$n=100$. The second is the frame: the model is defined on an axis-aligned "
    "rectangular region and is not rotation invariant, so outside the "
    "matched-domain rows, where the sampling region is supplied, the "
    "implementation rotates each instance into its minimum-area enclosing "
    "rectangle by scanning convex-hull edges \\citep{freeman1975determining}, which "
    "is what the source prescribes. \\citet{choi2021adjustment} records that the "
    "uncorrected model underestimates for $n<1000$, which is the direction the "
    "correction moves it.",
    "\\paragraph{\\c{C}avdar--Sokol.} This estimator was rebuilt against a "
    "primary document we hold: \\citet{cavdar2014dissertation}, Chapter 4, the "
    "same work as \\citet{cavdar2015distribution} and by the same author. The "
    "area term is the area of the rectangle covering the nodes, and $\\bar c_j$ "
    "and $\\mathrm{cstdev}_j$ are the mean and standard deviation of the absolute "
    "distances to that rectangle's midpoint line on axis $j$, not to the sample "
    "mean. The divisor $\\bar c_x\\bar c_y$ sits inside the radical; placing it "
    "outside leaves the second term dimensionless. An earlier revision of our code "
    "substituted the convex hull for the area and measured dispersion about the "
    "sample midrange; both are corrected, and every \\c{C}avdar--Sokol row in this "
    "paper is the corrected, rotated model. Two further features of the published "
    "method are now applied. The first is the frame: the model is defined on an "
    "axis-aligned rectangle and is not rotation invariant, so the implementation "
    "rotates each instance into its minimum-area enclosing rectangle by scanning "
    "convex-hull edges \\citep{freeman1975determining}, which is what the source "
    "prescribes. The second is the finite-$n$ correction "
    "$0.9325e^{0.00005298n}-0.2972e^{-0.01452n}$, Eq.~(21) of the dissertation, "
    "fitted over $n\\in\\{100,125,\\dots,975\\}$ at $R^2=0.9867$; the raw estimate "
    "is divided by it. \\citet{choi2021adjustment} records that the uncorrected "
    "model underestimates for $n<1000$, which is the direction the correction "
    "moves it. That correction is bounded to the range it was fitted over, and "
    "the bound binds on this benchmark: 1{,}320 of the 2{,}580 2D instances have "
    "$n<100$, so for slightly over half the benchmark the ratio is evaluated at "
    "the $n=100$ endpoint rather than extrapolated below it. The authors' own "
    "figures show the ratio continuing to fall toward $0.4$ as $n$ approaches 10, "
    "but publish no fitted form there, so extrapolating would substitute our "
    "curve for theirs. We hold the upper end fixed for the same reason: above "
    "$n=975$ no correction is applied, because Eq.~(21) grows without bound "
    "($E/T=1.22$ at $n=5000$) and would contradict the training fit it exists to "
    "repair. That leaves a 1.8\\% step at the upper boundary, where the fitted "
    "ratio reads 0.982. Two properties of the source, not of our evaluation, "
    "remain: the regression targets Lin--Kernighan tour lengths rather than "
    "optima, and every training graph carried a node at each corner of its "
    "rectangle, which pins the graph area to the covering rectangle in a way our "
    "instances do not.")

edit(
    "\\paragraph{Transcription sources.} \\citet{chien1992operational}, "
    "\\citet{kwon1995tsp}, and \\citet{daganzo1984b} are paywalled with no "
    "open-access copy we could reach, and we transcribed their equations from the "
    "literature review in \\citet{figliozzi2009planning}, which renders them "
    "identically across two manuscript versions. Readers with access to the "
    "primaries should treat those as authoritative over our transcriptions. The "
    "\\c{C}avdar--Sokol form and both of its constant sets are verified directly "
    "against \\citet{cavdar2014dissertation}, which is open access.",
    "\\paragraph{Sources.} Every constant this paper scores is read from a "
    "primary document. The BHH constants come from \\citet{johnson1996asymptotic} "
    "and \\citet{percus1996finite}; the \\c{C}avdar--Sokol form and both of its "
    "constant sets are verified directly against \\citet{cavdar2014dissertation}, "
    "which is open access. No scored constant is transcribed from a secondary "
    "source, and where the only available rendering was secondary we withdrew the "
    "estimator rather than publish the number.")

edit(
    "Table~\\ref{tab:classical} reports the classical estimators twice. The upper "
    "panels apply them to the complete 2D and TSPLIB EUC\\_2D benchmarks with the "
    "convex hull of each instance standing in for the region measure, which is the "
    "only option on TSPLIB and the fairer option on the degenerate 2D classes. The "
    "lower panels restrict to the 210 i.i.d.\\ uniform 2D instances, where the "
    "sampling region $[0,G]^2$ is exact, and the last two restrict further to the "
    "published node-count ranges of Chien and Kwon--Golden--Wasil, which is where "
    "those two estimators appear. Rows carry their own $N$ because those supports "
    "differ.",
    "Table~\\ref{tab:classical} reports the two classical estimators twice. The "
    "upper panels apply them to the complete 2D and TSPLIB EUC\\_2D benchmarks, "
    "with the convex hull of each instance standing in for BHH's region measure, "
    "which is the only option on TSPLIB and the fairer option on the degenerate 2D "
    "classes. The lower panel restricts to the 210 i.i.d.\\ uniform 2D instances, "
    "where the sampling region $[0,G]^2$ is exact and BHH receives it. "
    "\\c{C}avdar--Sokol receives the same input in both, the rectangle covering "
    "the nodes, so only its instance set changes between the panels.")

# --------------------------------------------------------------------------
# tab:paired -- three rows out, the Cavdar row restated.
# --------------------------------------------------------------------------
edit(
    "2D uniform, Kwon domain & Kwon--Golden--Wasil & 80 & $-3.37$ [$-4.50$, "
    "$-2.35$] & $1.2\\times10^{-10}$ \\\\\n"
    "2D uniform, Chien domain & Chien & 50 & $-16.09$ [$-19.70$, $-12.75$] & "
    "$1.6\\times10^{-13}$ \\\\\n"
    "2D uniform & BHH (sampling region) & 210 & $-7.60$ [$-8.56$, $-6.71$] & "
    "$2.0\\times10^{-35}$ \\\\\n"
    "2D uniform & \\c{C}avdar--Sokol (sampling region) & 210 & $-9.50$ "
    "[$-10.72$, $-8.30$] & $4.0\\times10^{-36}$ \\\\\n"
    "2D uniform & Daganzo (sampling region) & 210 & $-15.46$ [$-16.56$, $-14.35$] "
    "& $1.5\\times10^{-35}$ \\\\",
    "2D uniform & BHH (sampling region) & 210 & $-7.60$ [$-8.56$, $-6.71$] & "
    "$2.0\\times10^{-35}$ \\\\\n"
    "2D uniform & \\c{C}avdar--Sokol & 210 & $-6.85$ [$-8.57$, $-5.30$] & "
    "$7.2\\times10^{-27}$ \\\\\n"
    "2D uniform & $L_{\\mathrm{MST}}$ ($\\alpha=1$) & 210 & $-14.29$ [$-15.35$, "
    "$-13.38$] & $3.3\\times10^{-36}$ \\\\")


def main() -> int:
    text = TEX.read_text(encoding="utf-8")
    failures: list[str] = []
    for i, (old, new) in enumerate(EDITS):
        count = text.count(old)
        if count != 1:
            failures.append(f"edit {i}: {count} occurrences of {old[:80]!r}...")
            continue
        text = text.replace(old, new)
    if failures:
        print("ABORTED, nothing written:")
        for f in failures:
            print("  " + f)
        return 1
    TEX.write_text(text, encoding="utf-8")
    print(f"applied {len(EDITS)} edits to {TEX}")
    for token in ("Daganzo", "Chien", "Kwon", "daganzo1984", "chien1992", "kwon1995"):
        print(f"  remaining {token!r}: {text.count(token)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
