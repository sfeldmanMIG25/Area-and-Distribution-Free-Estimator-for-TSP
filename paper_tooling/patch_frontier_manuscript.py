"""Splice the Held--Karp 1-tree frontier into ``Area_Free_Main.tex``.

Byte-level, CRLF-aware, anchored, idempotent-checked.  A heredoc has corrupted
``\\ref`` into a literal CR + ``ef{`` in this project four times, so this file
does ``read_bytes`` / ``replace`` / ``write_bytes`` and nothing else, and it
reports applied-versus-missing per patch instead of exiting silently.

No generated table body is touched.  The two tables and the figure this adds
carry labels that ``build_paper_tables.py`` does not know, so its parser resets
to ``None`` on them and none of their cells enters the 1,910-cell gate.

Run::

    python paper_tooling/patch_frontier_manuscript.py          # apply
    python paper_tooling/patch_frontier_manuscript.py --dry    # report only
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper_reference" / "Area_Free_Main.tex"
BIB = ROOT / "paper_reference" / "references.bib"

CRLF = "\r\n"


def L(*lines: str) -> str:
    """Join literal manuscript lines with the file's CRLF ending."""
    return CRLF.join(lines)


# ---------------------------------------------------------------------------
# 1.  Abstract -- scope the false superlative, then state the adverse result.
# ---------------------------------------------------------------------------
A_OLD = (
    r"The strongest baseline that uses no learned model is a calibrated ratio "
    r"$\hat\rho(d,n)$ on the multidimensional benchmark, where GART 2.0 beats "
    r"it by a factor of 2.9."
)
A_NEW = (
    r"The strongest baseline that uses no learned model, among the closed forms "
    r"and constant ratios that make up that roster, is a calibrated ratio "
    r"$\hat\rho(d,n)$ on the multidimensional benchmark, where GART 2.0 beats "
    r"it by a factor of 2.9. That scoping is load-bearing. A Held--Karp 1-tree "
    r"lower bound uses no learned model and no training corpus and returns a "
    r"certificate rather than an estimate, and on that same multidimensional "
    r"benchmark it reaches 0.125\% MAPE against GART 2.0's 0.620\% at 0.71 "
    r"times the cost, so GART 2.0 is dominated there and we report it as "
    r"dominated. On TSPLIB EUC\_2D the ordering depends on instance size: "
    r"scaled by one training-split constant the bound costs 0.90 times GART "
    r"2.0 at 2.00\% MAPE against 2.58\% on the corpus median and 0.16 times "
    r"its cost at 1.69\% against 2.01\% on the smallest size bucket, while in "
    r"the largest bucket matching GART 2.0's accuracy costs the raw bound 6.65 "
    r"times what GART 2.0 spends. What survives is an asymptotic separation "
    r"rather than a constant-factor one: measured cost exponents in $n$ are "
    r"0.975 for GART 2.0 against 2.08--2.13 for the bound, $\Theta(n\log n)$ "
    r"against $\Theta(k n^{2})$ in the plane, so the present gap is narrow and "
    r"widens without limit in $n$."
)

# ---------------------------------------------------------------------------
# 2.  Introduction, contribution subsection.
# ---------------------------------------------------------------------------
I_OLD = (
    r"Its margin over the strongest baseline that uses no learned model is a "
    r"factor of 2.9 on the multidimensional set and 1.00 percentage points on "
    r"TSPLIB EUC\_2D."
)
I_NEW = (
    r"Its margin over the strongest baseline that uses no learned model is a "
    r"factor of 2.9 on the multidimensional set and 1.00 percentage points on "
    r"TSPLIB EUC\_2D. That margin is over the strongest baseline in the "
    r"estimator roster, and the roster is not the whole comparison. "
    r"Section~\ref{sec:frontier} admits a Held--Karp 1-tree lower bound, which "
    r"needs no training corpus and certifies its own output, and the result is "
    r"adverse in two places: on the multidimensional benchmark the bound is "
    r"4.95 times more accurate than GART 2.0 at 0.71 times the cost, and on "
    r"the TSPLIB EUC\_2D corpus median the same bound scaled by a single "
    r"training-split constant is more accurate at 0.90 times the cost. GART "
    r"2.0 is therefore not on the cost/accuracy Pareto front of either "
    r"benchmark once that family is admitted. Where the ordering does favour "
    r"it is above the smallest planar sizes and, asymptotically, everywhere: "
    r"the bound costs $\Theta(k n^{2})$ against $\Theta(n\log n)$ for GART 2.0 "
    r"at $d\le3$, measured as cost exponents in $n$ of 2.08--2.13 against 0.975."
)

# ---------------------------------------------------------------------------
# 3.  Paper outline.
# ---------------------------------------------------------------------------
O_OLD = (
    r"Section~\ref{sec:application} extends GART 2.0 to non-Euclidean TSPLIB95 "
    r"instances via MDS embedding."
)
O_NEW = (
    r"Section~\ref{sec:frontier} then admits a certified Held--Karp 1-tree "
    r"lower bound as a comparator and reports where GART 2.0 is, and is not, "
    r"on the resulting cost/accuracy front. Section~\ref{sec:application} "
    r"extends GART 2.0 to non-Euclidean TSPLIB95 instances via MDS embedding."
)

# ---------------------------------------------------------------------------
# 4.  Provenance audit -- withdraw a sentence the certificate falsifies.
# ---------------------------------------------------------------------------
P_OLD = r"The stale field is the tour permutation, not the cost."
P_NEW = (
    r"An earlier revision of this paper stated here that the stale field is the "
    r"tour permutation and not the cost. That is now known to be false for part "
    r"of this set, and we withdraw it: Section~\ref{subsec:frontier_labels} "
    r"certifies 40 stored costs across the whole corpus as strictly below a "
    r"proven lower bound on the optimum of their own released coordinates, and "
    r"39 of those 40 are inside these 184 rows."
)

# ---------------------------------------------------------------------------
# 5.  Benchmark models -- the comparator paragraph and two table rows.
# ---------------------------------------------------------------------------
B_OLD = (
    r"\paragraph{Region measure and matched domains.}"
)
B_NEW = L(
    r"\paragraph{Certified lower bound.} One further row is not an estimator at "
    r"all. The Held--Karp 1-tree bound returns a proven lower bound on the "
    r"optimal tour rather than a prediction of it, at a cost fixed by an "
    r"explicit ascent budget $k$. We score it on the multidimensional benchmark "
    r"and on TSPLIB EUC\_2D, both raw and scaled by a single constant fitted "
    r"per budget on the training split, and Section~\ref{sec:frontier} reports "
    r"it in full. It is deliberately outside the seventeen-baseline counts "
    r"above, because it answers a different question than an estimator does; "
    r"it is reported anyway, because the counts are not the comparison and "
    r"admitting it changes this paper's positioning.",
    r"",
    r"\paragraph{Region measure and matched domains.}",
)

BT_OLD = L(
    r"Custom Hilbert sort & Bare $d$-dimensional ordering, per-axis normalized, "
    r"$p=16$ & Constructs a tour & $O(n\log n)$ & After "
    r"\citet{bartholdi1982heuristic} \\",
    r"\bottomrule",
)
BT_NEW = L(
    r"Custom Hilbert sort & Bare $d$-dimensional ordering, per-axis normalized, "
    r"$p=16$ & Constructs a tour & $O(n\log n)$ & After "
    r"\citet{bartholdi1982heuristic} \\",
    r"\midrule",
    r"\multicolumn{5}{@{}l}{\textit{Certified lower bound}} \\",
    r"\midrule",
    r"Held--Karp 1-tree, budget $k$ & $L_{\mathrm{1tree}}(\pi_k)-2\sum_i\pi_i$ "
    r"after $k$ ascent steps & Any metric, any $d$; certificate & "
    r"$\Theta(k n^2 d)$ & \citet{heldkarp1970,heldkarp1971} \\",
    r"Calibrated 1-tree & $c_k$ times that bound, $c_k$ fitted per budget on "
    r"the training split & As trained & $\Theta(k n^2 d)$ & This work \\",
    r"\bottomrule",
)

# ---------------------------------------------------------------------------
# 6.  Discussion -- point the cost paragraph at the new section.
# ---------------------------------------------------------------------------
D_OLD = (
    r"The classical estimators are the comparison this paper positions itself "
    r"against, so the honest statement is comparable cost among learned "
    r"estimators, twice the cost of a constant multiple of $L_{\mathrm{MST}}$, "
    r"and roughly an order of magnitude over the classical closed forms."
)
D_NEW = (
    r"The classical estimators are the comparison this paper positions itself "
    r"against, so the honest statement is comparable cost among learned "
    r"estimators, twice the cost of a constant multiple of $L_{\mathrm{MST}}$, "
    r"and roughly an order of magnitude over the classical closed forms. Every "
    r"row in that accounting is an estimator. Section~\ref{sec:frontier} runs "
    r"the same accounting against a certified lower bound and reaches the "
    r"opposite conclusion at small $n$ and on the multidimensional benchmark."
)

# ---------------------------------------------------------------------------
# 7.  The new section.
# ---------------------------------------------------------------------------
SECTION_ANCHOR = (
    r"\section{Application: Non-Euclidean Estimation via MDS Embedding} "
    r"\label{sec:application}"
)

FRONTIER = L(
    r"\section{A Certified Lower Bound as a Comparator} \label{sec:frontier}",
    r"",
    r"Every baseline of Section~\ref{subsec:bench_models} is an estimator: it "
    r"returns a number that carries no guarantee, and it is scored only by how "
    r"close that number lands. This section admits a comparator of a different "
    r"kind, the Held--Karp 1-tree lower bound, and reports what admitting it "
    r"does to this paper's positioning. Three findings follow and two of them "
    r"are adverse. GART 2.0's cost advantage is a function of instance size and "
    r"ambient dimension rather than a property of the estimator. On the "
    r"multidimensional benchmark the bound is both more accurate and cheaper, "
    r"so GART 2.0 is off the cost/accuracy Pareto front there outright. What "
    r"survives is an asymptotic separation rather than a constant-factor one.",
    r"",
    r"\subsection{The bound, the ascent, and what is certified} "
    r"\label{subsec:frontier_bound}",
    r"",
    r"Fix a node $s$. A \emph{1-tree} is a spanning tree on $V\setminus\{s\}$ "
    r"together with the two cheapest edges incident to $s$. It has $n$ edges "
    r"and total degree $2n$, and every Hamiltonian tour is a 1-tree, so the "
    r"minimum 1-tree is a relaxation of the tour problem. Introduce node "
    r"potentials $\pi\in\mathbb{R}^{n}$ and transformed costs "
    r"$c^{\pi}_{ij}=c_{ij}+\pi_i+\pi_j$. Under $c^{\pi}$ every tour costs "
    r"exactly $L_{\mathrm{TSP}}+2\sum_i\pi_i$, because every node has degree "
    r"two, so for \emph{any} real $\pi$",
    r"\begin{equation} \label{eq:hk}",
    r"    w(\pi) \;=\; L_{\mathrm{1tree}}(\pi) \;-\; 2\sum_{i\in V}\pi_i "
    r"\;\le\; L_{\mathrm{TSP}},",
    r"\end{equation}",
    r"which is the Held--Karp bound \citep{heldkarp1970,heldkarp1971}. The map "
    r"$w$ is concave and piecewise linear, a subgradient at $\pi$ is "
    r"$g_i=\deg_i(\pi)-2$, and $\max_\pi w(\pi)$ is approached by subgradient "
    r"ascent. We write $w(\pi_k)$ for the bound after $k$ ascent iterations and "
    r"treat $k$ as an explicit knob trading cost against tightness. Validity "
    r"needs neither $\pi\ge0$ nor $c^{\pi}\ge0$: the argument above is "
    r"sign-free, and the spanning tree is built on the perturbed costs directly.",
    r"",
    r"Two properties separate this row from every other row in this paper. "
    r"First, Eq.~\eqref{eq:hk} holds at every $\pi$, so $w(\pi_k)$ is a "
    r"\emph{certificate} whatever the ascent did: the printed number is a "
    r"proven lower bound on that instance's optimal tour, and a reader can "
    r"verify it from the 1-tree alone. No other row in this paper certifies "
    r"anything about its own output, GART 2.0 included. Second, the bound "
    r"consumes no training corpus, so no question of coverage, distribution "
    r"shift or leakage arises about it at all.",
    r"",
    r"The certificate is paid for with one-sidedness. Because $w(\pi_k)$ never "
    r"exceeds the optimum, its error against a reference tour is a duality gap "
    r"plus whatever that reference is itself above the optimum, and not the "
    r"two-sided spread of an estimator. Turning the bound into an estimator "
    r"costs one number, and we report that row too: the \emph{calibrated} bound "
    r"$c_k\,w(\pi_k)$, where $c_k$ is a single scalar fitted per budget on the "
    r"planar training split and never on any scored instance. That row is not a "
    r"certificate, since $c_k>1$ pushes some predictions above the optimum, and "
    r"it is not training-free. It is the like-for-like comparator against a "
    r"trained estimator, and it is the stronger of the two rows on every corpus "
    r"scored below.",
    r"",
    r"Two ascents are used and each is reported where it was run. The planar arm "
    r"uses the Volgenant--Jonker step schedule with Helsgaun's direction "
    r"smoothing \citep{volgenant1982,helsgaun2000}. The multidimensional arm "
    r"uses a Polyak step \citep{polyak1969} against a constructive upper bound, "
    r"a nearest-neighbour tour improved by 2-opt and built from coordinates "
    r"alone, with the stored label never read during the ascent. Neither ascent "
    r"is proved to attain $\max_\pi w(\pi)$, so every accuracy figure below is a "
    r"floor on what this family reaches rather than a ceiling. The two arms "
    r"disagree: on the multidimensional split the Volgenant--Jonker arm returns "
    r"the higher bound on 4.28\% of instances, and taking the per-instance "
    r"maximum of both, at the cost of running both, improves the converged "
    r"multidimensional error from 0.0663\% to 0.0632\%. Validity was checked "
    r"against exhaustively enumerated optima on small instances and against a "
    r"second exact solver, with no violation of Eq.~\eqref{eq:hk} anywhere "
    r"(\texttt{paper\_tooling/hk1tree\_polyak\_validate.py}).",
    r"",
    r"\subsection{TSPLIB95 EUC\_2D: the ordering depends on size} "
    r"\label{subsec:frontier_tsplib}",
    r"",
    r"Cost here is measured on the protocol of Table~\ref{tab:tsplib_by_size}, "
    r"one estimator per process with threads pinned and the median of 11 "
    r"repeats taken inside a single quiet window, and both arms are compared on "
    r"the 77 of the 78 EUC\_2D instances that the bound's own dense kernel "
    r"covers. Table~\ref{tab:frontier_tsplib} gives the ladder. The one "
    r"excluded instance is reported at the end of this subsection rather than "
    r"dropped.",
    r"",
    r"Read on the corpus median, the raw certified bound first matches GART "
    r"2.0's MAPE at an ascent budget of 50, where it costs 1.46 times GART 2.0 "
    r"for a margin of 0.04 percentage points, and where the paired win rate is "
    r"exactly 50.0\%. The calibrated row settles what the raw row only "
    r"balances. At a budget of 25 it costs 0.90 times GART 2.0 and reaches "
    r"2.00\% MAPE against 2.58\%: strict domination on both axes, on the same "
    r"instances and the same protocol. GART 2.0 is therefore not on the "
    r"cost/accuracy Pareto front of this corpus median once the family is "
    r"admitted, and Figure~\ref{fig:frontier} shows the geometry.",
    r"",
    r"Splitting by size shows what the median hides, and the split rather than "
    r"the median is this section's useful output. On the 23 smallest instances "
    r"the calibrated bound at that same budget costs 0.16 times GART 2.0 at "
    r"1.69\% MAPE against 2.01\%, and even the raw uncalibrated certificate "
    r"reaches 22.2\% lower error at 0.46 times the cost. In the middle bucket "
    r"nothing dominates, and the raw bound needs the top of the ladder to match "
    r"at all. In the largest bucket the raw bound matches at 6.65 times the "
    r"cost and again nothing dominates. The reversal is monotone in $n$, which "
    r"is why the output of this comparison is a rule keyed to instance size "
    r"rather than a headline.",
    r"",
    r"Two disclosures bound the cost column. The bound is far more "
    r"load-sensitive than GART 2.0, its median rising by a factor of 1.25 "
    r"between a quiet and a noisy window at the crossing budget against 1.05 "
    r"for GART 2.0, so every published repeat is from one quiet window and the "
    r"noisy repeats are retained only as the control; mixing the two would have "
    r"overstated the bound's cost and flattered GART 2.0. And the excluded "
    r"instance is real cost rather than a hidden exclusion: on "
    r"\texttt{d18512}, whose node count is above the bound's own dense kernel "
    r"switch, the bound costs 167 times GART 2.0 at the crossing budget and "
    r"2{,}078 times at the top of the ladder, where that single instance takes "
    r"493 seconds against GART 2.0's 237 milliseconds.",
    r"",
    _TSPLIB_TABLE_PLACEHOLDER := r"%%FRONTIER_TSPLIB_TABLE%%",
    r"",
    r"\subsection{The multidimensional benchmark: GART 2.0 is dominated} "
    r"\label{subsec:frontier_nd}",
    r"",
    r"On the multidimensional benchmark the result is not a trade-off. "
    r"Table~\ref{tab:frontier_nd} reports the ladder by dimension group. At an "
    r"ascent budget of 200 the bound reaches 0.125\% MAPE against GART 2.0's "
    r"0.620\%, a factor of 4.95, at 0.71 times the cost, and it wins the paired "
    r"comparison on 82.1\% of instances. It first overtakes GART 2.0 at a "
    r"budget of 100, where it costs 0.40 times as much. Those ratios are "
    r"medians over a serial single-thread timing sample; weighted by the corpus "
    r"composition instead, the cost ratio at a budget of 200 is 0.54.",
    r"",
    r"The domination is not an artefact of one dimension group. It holds at "
    r"$d\in[4,10]$, at $d\in[15,50]$ and at $d=100$, and it is widest exactly "
    r"where the estimator was never trained: at $d=100$ a budget of 500 costs "
    r"0.35 times GART 2.0 and is 38.1 times more accurate. The one group GART "
    r"2.0 holds is $d\in\{2,3\}$, where matching its accuracy costs the bound "
    r"1.27 times as much. That is the planar regime, and it is the same regime "
    r"in which GART 2.0's own cost is subquadratic.",
    r"",
    r"Three checks make the finding hard to dismiss. The ascent never reads the "
    r"stored label, so the comparison is not circular: its upper bound is a "
    r"nearest-neighbour tour improved by 2-opt from coordinates alone, and "
    r"supplying the released optimum in its place converges faster early and to "
    r"the same value. On the 3{,}070 instances whose label came from an exact "
    r"solver, where the target is a proven optimum rather than a heuristic "
    r"tour, GART 2.0 scores 1.03\% against the bound's 0.10\% at a budget of "
    r"200. And the relaxation closes exactly, returning the optimum rather than "
    r"a bound, on 37.1\% of the split. That last figure is also the mechanism: "
    r"pairwise distances concentrate as $d$ grows, so the integrality gap this "
    r"relaxation is usually described by is a planar fact rather than a general "
    r"one.",
    r"",
    r"Two limits are ours to state. Roughly one instance in seven exhausts the "
    r"largest budget we ran, so at the top of the size range the reported "
    r"accuracy is a floor rather than a converged value. And the complexity "
    r"defence of Section~\ref{subsec:frontier_complexity} is planar and does "
    r"not transfer here: GART 2.0's own MST construction dispatches to a dense "
    r"kernel from $d=4$ upward, so on this benchmark both families are "
    r"quadratic in $n$ and what separates them is a constant that points the "
    r"wrong way.",
    r"",
    _ND_TABLE_PLACEHOLDER := r"%%FRONTIER_ND_TABLE%%",
    r"",
    _FIG_PLACEHOLDER := r"%%FRONTIER_FIGURE%%",
    r"",
    r"\subsection{Complexity across the estimator families} "
    r"\label{subsec:frontier_complexity}",
    r"",
    r"Table~\ref{tab:complexity} states the dominant term of every family in "
    r"this paper as the code executes it, not as the underlying algorithm is "
    r"usually quoted, together with whether the row certifies its own output "
    r"and whether it needs a training corpus. Three entries differ from the "
    r"textbook reading. The classical closed forms are dominated by a "
    r"row-deduplication pass rather than by their arithmetic. The convex-hull "
    r"blow-up this paper cites as a motivation never occurs at runtime, because "
    r"the implementation caps hull construction at three dimensions and falls "
    r"back to a bounding box above it. And the 1-tree cannot use the Delaunay "
    r"shortcut the MST family relies on in the plane: the node potentials break "
    r"the Euclidean metric, so after the first iteration the perturbed minimum "
    r"spanning tree is not a subgraph of the triangulation. That is measured "
    r"rather than assumed. At 144 potential vectors drawn from real ascent "
    r"trajectories the Delaunay-restricted 1-tree is heavier than the exact one "
    r"at 141 of them, so the shortcut is unavailable at every iteration but the "
    r"first, and that is why the bound sits a complexity class above the MST "
    r"family rather than a constant factor above it.",
    r"",
    r"That class difference is the separation that survives everything else in "
    r"this section. In the plane GART 2.0 is $\Theta(n\log n)$ and the bound is "
    r"$\Theta(k n^{2})$, and the measured log--log cost exponents in $n$ agree: "
    r"0.975 for GART 2.0 above a thousand nodes against 2.08--2.13 for the "
    r"bound over the same window. The consequence is a ratio that grows rather "
    r"than a fixed multiple. Against the MST-ratio family GART 2.0 holds a flat "
    r"2.0 to 2.3 times the cost across four orders of magnitude in $n$, which "
    r"is the signature of a shared complexity class; against the 1-tree at a "
    r"fixed budget the ratio moves from 7.7 at a thousand nodes to 211 at "
    r"sixteen thousand. The bound's advantage at small $n$ is a statement about "
    r"constants and the estimator's advantage at large $n$ is a statement about "
    r"growth, and only the second is stable.",
    r"",
    r"The separation is planar, and we do not extend it. On the "
    r"multidimensional benchmark GART 2.0's own measured cost exponent in $n$ "
    r"is 2.02 at $d\in[4,10]$ and 1.82 at $d\in[15,50]$, against 1.79--1.89 for "
    r"the bound, because \texttt{compute\_mst} takes a dense kernel from $d=4$ "
    r"upward. Only at $d\in\{2,3\}$ is that exponent 0.56. On that benchmark "
    r"both families are quadratic, the separation is a constant, and the "
    r"constant points the wrong way.",
    r"",
    _COMPLEXITY_TABLE_PLACEHOLDER := r"%%COMPLEXITY_TABLE%%",
    r"",
    r"\subsection{What the exact-solver comparison establishes} "
    r"\label{subsec:frontier_exact}",
    r"",
    r"This paper's cost argument has always had an unbounded anchor above it, "
    r"and that anchor is a floor rather than an achievement. Over the 25 TSPLIB "
    r"instances for which a Concorde solve time is on record, the published "
    r"median is 108 seconds against GART 2.0's 6.12 milliseconds, and the "
    r"recorded range runs from 0.13 seconds to more than eleven million. Those "
    r"times are the published record and not measurements on this machine, so "
    r"they are order-of-magnitude only. The load-bearing statement is the "
    r"complexity rather than the ratio: branch-and-cut over an exponential "
    r"family of subtour constraints admits no polynomial bound, so every row of "
    r"Table~\ref{tab:complexity} is asymptotically cheaper than an exact solve "
    r"and being cheaper than one distinguishes nothing among them. The "
    r"comparison that discriminates is the one against the 1-tree, which is "
    r"polynomial, certified, and in the same cost decade as the estimators.",
    r"",
    r"\subsection{Labels the bound refutes} \label{subsec:frontier_labels}",
    r"",
    r"A certified lower bound is also an audit instrument. If $B$ is a proven "
    r"lower bound on an instance's optimum and $L$ is its stored label, then "
    r"$B>L$ by more than the quantisation slack of the metric the solver was "
    r"handed refutes that label outright, with no tour entering the test. "
    r"Applying it across all 108{,}956 labelled instances in this project, with "
    r"each solver's own integer scale factor reverse-engineered and verified "
    r"against the released labels, refutes 40 of them, and 36 of those sit in a "
    r"set this paper scores (\texttt{paper\_tooling/label\_certificate.py}). By "
    r"split they are 35 in the multidimensional test partition, one on TSPLIB, "
    r"two in training and two in validation, and none at all on the 2D "
    r"benchmark; the worst overshoots its label by 4.13\%. The overlap with "
    r"Section~\ref{subsec:provenance} is near-total, since 39 of the 40 lie "
    r"inside that section's 184 inconsistent-tour instances. The certificate "
    r"therefore found no new population. What it did was upgrade part of the "
    r"old one from ``the stored tour disagrees with the coordinates'' to ``the "
    r"stored cost is provably not the optimum'', which is a different and worse "
    r"defect, and it is the reason the sentence withdrawn in "
    r"Section~\ref{subsec:provenance} was withdrawn.",
    r"",
    r"The single TSPLIB refutation is a parsing defect rather than a solver "
    r"defect, and it is the largest single artifact this audit found. The file "
    r"\texttt{linhp318.tsp} declares a fixed-edge section that our parser does "
    r"not recognise and walks past, so the instance is read as a plain tour "
    r"problem whose parsed coordinates and distance matrix are bit-identical to "
    r"\texttt{lin318}. Its stored label, 41{,}345, is the fixed-edge "
    r"Hamiltonian-path optimum; the tour optimum on those coordinates is "
    r"42{,}029, and the 1-tree bound in TSPLIB's own integer metric, with no "
    r"slack at all, is 41{,}802. Every estimator on this paper's TSPLIB tables "
    r"is scored against a target no tour on those coordinates can reach. A scan "
    r"of all 111 files found no second unrecognised section keyword and no "
    r"second duplicated geometry.",
    r"",
    r"The blast radius of repairing any of this is measured rather than "
    r"asserted, and it is not small "
    r"(\texttt{paper\_tooling/label\_defect\_blast\_radius.py}). Rebuilding "
    r"every gated table under all repairs together moves 847 of the 1{,}910 "
    r"checked cells, the largest accuracy cell by 4.30 percentage points, while "
    r"a control rebuild that applies no repair moves zero cells, so the "
    r"rebuild harness is faithful and every reported delta is real. Taken "
    r"singly, excluding \texttt{linhp318} moves 142 cells, dropping the 35 "
    r"refuted multidimensional labels moves 272, and rescoring the 2D benchmark "
    r"in float64 rather than in the integer metric its labels were solved in "
    r"moves 265, whose corpus-wide effect is to double GART 2.0's near-zero 2D "
    r"signed bias from $-0.05$ to $-0.10$. \textbf{The tables in this paper are "
    r"the unrepaired ones.} We report the counterfactual instead of the repair "
    r"because re-solving the refuted instances is a separate exercise, and a "
    r"reader who wants the repaired figures should take them from that artifact "
    r"rather than from our tables.",
    r"",
    r"Two of these bear directly on "
    r"Section~\ref{subsec:frontier_tsplib}, and the direction favours us, which "
    r"is why they are stated rather than left out. Excluding \texttt{linhp318} "
    r"moves GART 2.0's matched-corpus MAPE from 2.58\% to 2.54\% and moves the "
    r"raw bound's crossing budget from 50 to 100, and with it the cost multiple "
    r"at the crossing from 1.46 to 2.60. The honest reading is not that the "
    r"bound is dearer than we said, but that the printed crossing budget was "
    r"never a robust statistic: the interpolated crossing moves only from 48.7 "
    r"to 53.0, and the ladder rung is a step function over a margin of 0.04 "
    r"percentage points. The domination by the calibrated row is untouched, "
    r"because its margin is 0.59 percentage points rather than 0.04.",
    r"",
    r"Finally, a coverage gap. Seven TSPLIB instances carry no bound at all "
    r"because the dense perturbed matrix does not fit in memory, five of them "
    r"inside the scored EUC\_2D set. Nothing in this section is claimed about "
    r"those instances.",
    r"",
    r"\subsection{What this changes} \label{subsec:frontier_verdict}",
    r"",
    r"Three statements this paper could have made are false, and are therefore "
    r"not made. GART 2.0 is not cheaper than a 1-tree bound in general: it is "
    r"cheaper above roughly 400 nodes in the plane and dearer below roughly "
    r"150. Its accuracy is not closer to the bound than to the closed forms in "
    r"general: that holds against the area-based estimators and fails against "
    r"the MST-ratio family, from which it is separated by 1.00 percentage "
    r"points on the full 78-instance EUC\_2D set against 1.36 points from the "
    r"converged bound on the same set. And it does not occupy a middle ground "
    r"on the multidimensional benchmark, where the bound is better on both axes "
    r"at every dimension group but the planar one.",
    r"",
    r"What replaces them is a rule keyed to the instance rather than a claim "
    r"about the estimator. Below a few hundred nodes, prefer the certified "
    r"bound: it is cheaper, it is more accurate once scaled by a single "
    r"constant, it needs no training corpus, and it tells the caller how wrong "
    r"it can be. Above that, and at every dimension where the estimator is not "
    r"competing against a planar fast path it also enjoys, the ordering "
    r"reverses, and it reverses further with every increase in $n$ because the "
    r"two methods are in different complexity classes. A practitioner choosing "
    r"between them should read Table~\ref{tab:frontier_tsplib} and "
    r"Figure~\ref{fig:frontier} at their own instance size rather than reading "
    r"an aggregate. A paper that states where its own method should not be used "
    r"is more useful than one that does not, and the measurements above are why "
    r"this one states it.",
    r"",
    SECTION_ANCHOR,
)


# ---------------------------------------------------------------------------
# Tables and figure for the new section.
# ---------------------------------------------------------------------------
TSPLIB_TABLE = L(
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\caption{TSPLIB95 EUC\_2D cost/accuracy ladder, 77 instances matched "
    r"between both arms. Time is the median per-instance time in milliseconds "
    r"on the solo protocol of Table~\ref{tab:tsplib_by_size}; $\times$ is that "
    r"time over GART 2.0's in the same bucket. Raw is the certified bound "
    r"$w(\pi_k)$; Cal. is $c_k w(\pi_k)$ with $c_k$ fitted per budget on the "
    r"planar training split. Bold marks every cell where the bound family is "
    r"both cheaper and more accurate than GART 2.0 in that bucket.}",
    r"\label{tab:frontier_tsplib}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\renewcommand{\arraystretch}{1.1}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}lrrrrrrrrrrrr@{}}",
    r"\toprule",
    r" & \multicolumn{4}{c}{$n\in[51,150]$, $N=23$} & "
    r"\multicolumn{4}{c}{$n\in[151,400]$, $N=16$} & "
    r"\multicolumn{4}{c}{$n>400$, $N=38$} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
    r"Row & ms & $\times$ & Raw & Cal. & ms & $\times$ & Raw & Cal. & ms & "
    r"$\times$ & Raw & Cal. \\",
    r"\midrule",
    r"GART 2.0 & 3.73 & 1.00 & \multicolumn{2}{c}{2.01} & 4.55 & 1.00 & "
    r"\multicolumn{2}{c}{2.49} & 18.29 & 1.00 & \multicolumn{2}{c}{2.97} \\",
    r"\midrule",
    r"$k=0$ & 0.19 & 0.05 & 11.83 & 3.12 & 0.45 & 0.10 & 11.22 & 3.71 & "
    r"24.49 & 1.34 & 9.13 & 4.02 \\",
    r"$k=10$ & 0.37 & 0.10 & 8.47 & 2.04 & 0.91 & 0.20 & 9.75 & 3.66 & "
    r"43.93 & 2.40 & 7.63 & 2.53 \\",
    r"$k=25$ & 0.59 & 0.16 & 2.91 & \textbf{1.69} & 1.59 & 0.35 & 4.74 & "
    r"3.10 & 72.52 & 3.97 & 3.48 & 1.71 \\",
    r"$k=50$ & 0.96 & 0.26 & 2.09 & \textbf{1.68} & 2.79 & 0.61 & 3.76 & "
    r"3.08 & 121.59 & 6.65 & 2.31 & 1.56 \\",
    r"$k=100$ & 1.72 & 0.46 & \textbf{1.56} & \textbf{1.20} & 5.16 & 1.13 & "
    r"2.98 & 2.40 & 222.22 & 12.15 & 1.96 & 1.32 \\",
    r"$k=200$ & 3.22 & 0.86 & \textbf{1.57} & \textbf{1.22} & 10.10 & 2.22 & "
    r"2.96 & 2.39 & 427.85 & 23.40 & 1.88 & 1.26 \\",
    r"$k=500$ & 7.88 & 2.11 & 0.97 & 0.74 & 25.65 & 5.64 & 1.97 & 1.52 & "
    r"1120.37 & 61.27 & 1.62 & 1.12 \\",
    r"\bottomrule",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
)

ND_TABLE = L(
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\caption{Multidimensional benchmark cost/accuracy ladder, all 16{,}920 "
    r"held-out instances, Polyak ascent. Time is the sample median in "
    r"milliseconds on a serial single-thread timing sample; $\times$ is that "
    r"time over GART 2.0's in the same group. Bold marks every budget that is "
    r"both cheaper and more accurate than GART 2.0 in that group, that is, "
    r"every budget at which GART 2.0 is strictly dominated.}",
    r"\label{tab:frontier_nd}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\renewcommand{\arraystretch}{1.1}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}lrrrrrrrrrrrr@{}}",
    r"\toprule",
    r" & \multicolumn{3}{c}{$d\in\{2,3\}$, $N=1{,}296$} & "
    r"\multicolumn{3}{c}{$d\in[4,10]$, $N=4{,}536$} & "
    r"\multicolumn{3}{c}{$d\in[15,50]$, $N=5{,}184$} & "
    r"\multicolumn{3}{c}{$d=100$, $N=5{,}904$} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
    r"Row & ms & $\times$ & MAPE & ms & $\times$ & MAPE & ms & $\times$ & "
    r"MAPE & ms & $\times$ & MAPE \\",
    r"\midrule",
    r"GART 2.0 & 5.87 & 1.00 & 1.457 & 3.30 & 1.00 & 0.728 & 3.59 & 1.00 & "
    r"0.343 & 3.26 & 1.00 & 0.597 \\",
    r"\midrule",
    r"$k=0$ & 2.43 & 0.24 & 12.188 & 0.13 & 0.04 & 7.447 & 0.37 & 0.07 & "
    r"3.835 & 0.23 & 0.07 & 2.138 \\",
    r"$k=10$ & 3.51 & 0.31 & 7.417 & 0.24 & 0.08 & 4.515 & 0.36 & 0.12 & "
    r"2.171 & 0.26 & 0.08 & 1.037 \\",
    r"$k=25$ & 5.18 & 0.47 & 4.780 & 0.38 & 0.13 & 3.255 & 0.55 & 0.17 & "
    r"1.467 & 0.37 & 0.12 & 0.700 \\",
    r"$k=50$ & 7.37 & 0.69 & 3.218 & 0.62 & 0.21 & 1.988 & 0.92 & 0.27 & "
    r"0.876 & 0.49 & 0.15 & \textbf{0.422} \\",
    r"$k=100$ & 12.90 & 1.27 & 1.430 & 1.21 & 0.39 & \textbf{0.709} & 1.95 & "
    r"0.56 & \textbf{0.308} & 0.76 & 0.23 & \textbf{0.158} \\",
    r"$k=200$ & 22.97 & 2.18 & 0.601 & 2.15 & 0.69 & \textbf{0.179} & 3.30 & "
    r"0.95 & \textbf{0.061} & 1.19 & 0.37 & \textbf{0.036} \\",
    r"$k=500$ & 50.80 & 4.92 & 0.429 & 4.51 & 1.38 & 0.086 & 6.05 & 1.75 & "
    r"0.019 & 1.12 & 0.35 & \textbf{0.016} \\",
    r"\bottomrule",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
)

COMPLEXITY_TABLE = L(
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\caption{Cost class of every estimator family in this paper, as the "
    r"released code executes it rather than as the algorithm is usually quoted, "
    r"with whether the row certifies its own output and whether it consumes a "
    r"training corpus. $k$ is the 1-tree ascent budget and $p$ the Hilbert "
    r"order. Derivations and the measurements behind them are in "
    r"\texttt{paper\_tooling/complexity\_bank.json}.}",
    r"\label{tab:complexity}",
    r"\setlength{\tabcolsep}{5pt}",
    r"\renewcommand{\arraystretch}{1.15}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}llll@{}}",
    r"\toprule",
    r"\textbf{Family} & \textbf{Dominant term as implemented} & "
    r"\textbf{Certifies?} & \textbf{Needs training?} \\",
    r"\midrule",
    r"Classical closed forms & $\Theta(d\,n\log n)$; a row-deduplication pass, "
    r"not the arithmetic & No & Constants fitted in the source \\",
    r"Constant multiples of $L_{\mathrm{MST}}$ & $\Theta(n\log n)$ at $d\le3$ "
    r"(Delaunay); $\Theta(n^{2}d+n^{2}\log n)$ at $d\ge4$ & Lower bound only at "
    r"$\alpha=1$ & The two calibrated rows only \\",
    r"Space-filling curve (Hilbert) & $\Theta(n d p)$ interpreted, plus an "
    r"$O(n\log n)$ sort & Yes, an upper bound: it builds a tour & No \\",
    r"Learned estimators on this feature block & Feature cost as the MST family "
    r"above; head $O(1)$ in $n$ & No & Yes \\",
    r"GART 1.0 & $\Theta(n^{2}d)$ time and $\Theta(n^{2})$ memory; a dense "
    r"matrix for two scalars & No & Yes \\",
    r"Held--Karp 1-tree, budget $k$ & $\Theta(k n^{2} d)$; no Delaunay path "
    r"exists after the first iteration & Yes, a lower bound & No \\",
    r"Calibrated 1-tree & As above & No & One scalar per budget \\",
    r"Exact solver & $\Theta(n^{2}2^{n})$ for the bitmask DP; branch-and-cut "
    r"admits no polynomial bound & Yes, exact & No \\",
    r"\bottomrule",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
)

FIGURE = L(
    r"\begin{figure}[!ht]",
    r"\centering",
    r"\includegraphics[width=\textwidth]{frontier_cost_accuracy.pdf}",
    r"\caption{Cost against accuracy, both axes logarithmic. Circles trace the "
    r"1-tree ascent ladder at increasing budget, stars mark GART 2.0, and open "
    r"rings mark the points on each bucket's Pareto front. In panel (a) the "
    r"solid line is the raw certified bound and the dashed line the same bound "
    r"scaled by one training-split constant. A star inside a ring is on the "
    r"front; a star with ladder points below and to its left is dominated. GART "
    r"2.0 is dominated in the smallest planar bucket and in every "
    r"multidimensional group except the planar one. Point set: "
    r"\texttt{paper\_tooling/frontier\_manuscript\_points.csv}.}",
    r"\label{fig:frontier}",
    r"\end{figure}",
)

# ---------------------------------------------------------------------------
# 8.  Conclusion.
# ---------------------------------------------------------------------------
C_ANCHOR = r"Three limitations bound these results."
C_NEW = L(
    r"GART 2.0's cost advantage is a function of instance size rather than a "
    r"property of the estimator, and Section~\ref{sec:frontier} states where it "
    r"fails. Admitting a Held--Karp 1-tree lower bound, which consumes no "
    r"training corpus and certifies its own output, removes GART 2.0 from the "
    r"cost/accuracy Pareto front of the multidimensional benchmark outright, "
    r"where the bound is 4.95 times more accurate at 0.71 times the cost, and "
    r"from the TSPLIB EUC\_2D corpus median once the bound is scaled by a "
    r"single training-split constant, where it is more accurate at 0.90 times "
    r"the cost. In the plane the ordering reverses with size: in the smallest "
    r"size bucket the uncalibrated certificate reaches 22.2\% lower error at "
    r"0.46 times the cost, while in the largest bucket matching GART 2.0 costs "
    r"it 6.65 times as much. What survives is asymptotic, and it is the durable "
    r"part of the claim: the estimator is $\Theta(n\log n)$ in the plane "
    r"against $\Theta(k n^{2})$ for the bound, measured as cost exponents in "
    r"$n$ of 0.975 against 2.08--2.13, so a present factor of about one and a "
    r"half widens without limit. The same bound used as an audit instrument "
    r"refutes 40 stored labels across the corpus, 36 of them in a scored set; "
    r"the tables in this paper are the unrepaired ones and "
    r"Section~\ref{subsec:frontier_labels} reports the counterfactual.",
    r"",
    r"Three limitations bound these results.",
)

# ---------------------------------------------------------------------------
# Bibliography additions.
# ---------------------------------------------------------------------------
BIB_ADD = L(
    r"",
    r"@article{heldkarp1971,",
    r"  title={The traveling-salesman problem and minimum spanning trees: "
    r"{Part II}},",
    r"  author={Held, Michael and Karp, Richard M.},",
    r"  journal={Mathematical Programming},",
    r"  volume={1},",
    r"  number={1},",
    r"  pages={6--25},",
    r"  year={1971},",
    r"  publisher={Springer},",
    r"  doi={10.1007/BF01584070}",
    r"}",
    r"",
    r"@article{volgenant1982,",
    r"  title={A branch and bound algorithm for the symmetric traveling "
    r"salesman problem based on the 1-tree relaxation},",
    r"  author={Volgenant, Ton and Jonker, Roy},",
    r"  journal={European Journal of Operational Research},",
    r"  volume={9},",
    r"  number={1},",
    r"  pages={83--89},",
    r"  year={1982},",
    r"  publisher={Elsevier},",
    r"  doi={10.1016/0377-2217(82)90015-7}",
    r"}",
    r"",
    r"@article{polyak1969,",
    r"  title={Minimization of unsmooth functionals},",
    r"  author={Polyak, Boris T.},",
    r"  journal={USSR Computational Mathematics and Mathematical Physics},",
    r"  volume={9},",
    r"  number={3},",
    r"  pages={14--29},",
    r"  year={1969},",
    r"  publisher={Elsevier},",
    r"  doi={10.1016/0041-5553(69)90061-5}",
    r"}",
    r"",
)


def _frontier_block() -> str:
    """The new section with its float placeholders filled in."""
    return (FRONTIER
            .replace(_TSPLIB_TABLE_PLACEHOLDER, TSPLIB_TABLE)
            .replace(_ND_TABLE_PLACEHOLDER, ND_TABLE)
            .replace(_FIG_PLACEHOLDER, FIGURE)
            .replace(_COMPLEXITY_TABLE_PLACEHOLDER, COMPLEXITY_TABLE))


PATCHES: list[tuple[str, str, str]] = [
    ("abstract.scope_and_frontier", A_OLD, A_NEW),
    ("intro.contribution", I_OLD, I_NEW),
    ("intro.outline", O_OLD, O_NEW),
    ("provenance.withdraw_stale_field", P_OLD, P_NEW),
    ("bench_models.certified_bound_paragraph", B_OLD, B_NEW),
    ("bench_models.table_rows", BT_OLD, BT_NEW),
    ("discussion.pointer", D_OLD, D_NEW),
    ("section.frontier", SECTION_ANCHOR, None),   # filled below
    ("conclusion.frontier", C_ANCHOR, C_NEW),
]
PATCHES[7] = ("section.frontier", SECTION_ANCHOR, _frontier_block())


def apply(dry: bool) -> int:
    raw = TEX.read_bytes()
    text = raw.decode("utf-8")
    applied, missing, ambiguous = [], [], []

    for name, old, new in PATCHES:
        count = text.count(old)
        if count == 0:
            missing.append(name)
            continue
        if count > 1:
            ambiguous.append(f"{name} ({count} matches)")
            continue
        text = text.replace(old, new, 1)
        applied.append(name)

    bib_raw = BIB.read_bytes().decode("utf-8")
    bib_needed = "@article{heldkarp1971," not in bib_raw
    if bib_needed:
        bib_raw = bib_raw.rstrip() + CRLF + BIB_ADD

    print(f"applied {len(applied)} / {len(PATCHES)}")
    for n in applied:
        print(f"  APPLIED  {n}")
    for n in missing:
        print(f"  MISSING  {n}")
    for n in ambiguous:
        print(f"  AMBIGUOUS {n}")
    print(f"  BIB entries {'appended' if bib_needed else 'already present'}")

    if missing or ambiguous:
        print("REFUSING to write: every patch must land exactly once.")
        return 1
    if dry:
        print("--dry: nothing written")
        return 0

    out = text.encode("utf-8")
    if b"\r\n" not in out or out.count(b"\n") != out.count(b"\r\n"):
        print("REFUSING to write: line endings are no longer uniformly CRLF.")
        return 1
    TEX.write_bytes(out)
    if bib_needed:
        BIB.write_bytes(bib_raw.encode("utf-8"))
    print(f"wrote {TEX} ({len(out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(apply(dry="--dry" in sys.argv))
