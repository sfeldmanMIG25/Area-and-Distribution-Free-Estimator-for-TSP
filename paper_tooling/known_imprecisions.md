# Known imprecisions in `paper_reference/Area_Free_Main.tex`

Recorded 2026-08-11, alongside the six-finding copy-edit of the same date.

These are the imprecisions that a reader could notice and that a rebuild was
nonetheless judged not worth. Each entry gives the manuscript line at the time
of recording, the text as printed, the correct value, and why the printed form
was left alone. **None of them changes a conclusion, a ranking, or a reported
metric.** That is the admission criterion for this file: anything that would
change what a reader concludes was fixed in the manuscript instead.

Line numbers are those of the 1,424-line CRLF source at commit-time on
2026-08-11. They rot; the quoted text is the durable locator.

---

## 1. The probe's dimension grid has 22 points, not 24

**Line 214**, `\subsection{Model Architecture and Training}`

> swept on log-spaced grids of 24 points, $d$ from 2 to 200 and $n$ from 5 to
> 4{,}000

**Correct.** 24 is the requested grid size and is exact for the node-count axis.
The dimension axis dedupes to **22** distinct integers after rounding
`_log_int_grid(2, 200, 24)` to integers, so "grids of 24 points" is right for one
axis and two points generous for the other.
`paper_tooling/v4_study_gart2_probe.csv` records `grid_points` = 22 for every
`swept=dimension` row and 24 for every `swept=n_customers` row; the bank agrees
(`cons_probe_gart_2_0_dimension_grid_points` = 22,
`cons_probe_gart_2_0_n_customers_grid_points` = 24).

**Why not rebuilt.** The number is a protocol constant, not a result. The claim
`methods.probe.grid_points` is registered against the node-count key and its
`note` already records the 22, so the discrepancy is documented where the gate
looks. Two grid points on one axis change no probe percentage: the constrained
model is non-increasing on 100% of sweeps on both axes at every seed, and the
unconstrained control fails on both, at either grid size.

---

## 2. "the remaining 47.6\%" leaves 100.2\%

**Line 427**, `\subsection{Is the degenerate-geometry error a coverage gap?}`

> the \texttt{grid} family carries 52.6\% of it and Line Noise the remaining
> 47.6\%

**Correct.** Both percentages are right;
`paper_tooling/armA_verify_gain_decomposition.csv`, column
`share_of_total_gain_pct`, gives `grid` = 52.611481 and `line_noise` = 47.560384.
The word **"remaining" is what is wrong**: the two sum to 100.17, because the
other eleven generators carry a net **−0.17\%** of the gain — seven of them are
negative, the largest being `boundary` at −0.75\%. The residual after `grid` is
47.388519\%, so either "Line Noise a further 47.6\%" or "the remaining 47.4\%"
would be exact.

**Why not rebuilt.** The sentence's claim is that two of thirteen generators carry
the whole gain, and a net −0.17\% spread over the other eleven is that claim
holding, not failing. The overshoot is visible only to a reader who adds the two
numbers, and adding them recovers the correct reading.

---

## 3. "all six results tables" — seven table environments print the row

**Line 244**, `\subsection{Benchmark Models}`

> It is scored on every benchmark here and is a row in all six results tables.

**Correct.** GART 2.0 (V3 features) has a row in **seven** printed `table`
environments: `tab:nd_by_dim`, `tab:nd_by_size`, `tab:2d_by_size`, its unlabelled
"part 2 of 2" continuation, `tab:genclass`, `tab:tsplib_by_size` and
`tab:tsplib_nonEuc`. (An eighth, `tab:benchmark_models`, names the row but is the
roster, not a results table.)

**Why not rebuilt.** Six is the count of logical results tables; seven is the
count of floats, because the 2D by-size table is split across two `table`
environments to fit the page. The extra environment is a typesetting artefact of
the same table, and the sentence's purpose — that this baseline is printed
everywhere and was previously left out of the enumerated set — is served by
either count.

---

## 4. The abstract enumerates fourteen baselines and then says seventeen

**Line 90**, abstract

> We compare against five classical estimators, against constant multiples of
> $L_{\mathrm{MST}}$ calibrated on the training split, and against a linear model
> and a neural network, each fitted twice \dots and against a boosted variant on
> an extended 32-feature block.

and later in the same abstract

> It has the lowest MAPE and SDPE of the seventeen baselines on one stratum of
> the four.

**Correct.** The enumeration reaches 14: five classical, four constant-ratio,
four learned refits (linear and network, each on the predecessor block and on
GART 2.0's own 31 features), and the extended-block variant. The three the
abstract never names are **GART 1.0**, the **space-filling-curve construction**,
and the **boosted predecessor on the 30-feature block**. Section 1.3 enumerates
all seventeen.

**Why not rebuilt.** The three unnamed baselines are the three weakest in the set
— GART 1.0 at 8.46\% TSPLIB MAPE and the Hilbert sort at 45.20\% against GART
2.0's 2.55\% — so naming them would widen every margin the abstract reports, not
narrow one. The abstract is already the longest element of the manuscript, and
"seventeen" is the count the reader is asked to hold, with the roster one section
away.

---

## 5. The abstract's constant-ratio descriptor covers two of the four rows

**Line 90**, abstract, same clause as item 4

> against constant multiples of $L_{\mathrm{MST}}$ calibrated on the training
> split

**Correct.** Four constant-ratio rows bound what is achievable without learning
(Section 4.1, line 242): the $\alpha=1$ floor, the asymptotic MST ratio, and the
two calibrated rows $\hat\rho(d)$ and $\hat\rho(d,n)$. Only the **last two** are
calibrated on the training split. The $\alpha=1$ floor and the asymptotic ratio
use no fitted quantity at all. The abstract goes on to name exactly two rows by
name — $\hat\rho(d,n)$ and the asymptotic ratio — so a reader cannot recover the
other two from the abstract alone.

**Why not rebuilt.** The descriptor understates the baseline set rather than
overstating it: the two rows it mis-describes as calibrated are the two that are
*harder* to beat on the strata where they lead, and the abstract already reports
that the asymptotic ratio, not the calibrated one, is the strongest non-learned
comparator on TSPLIB. Naming all four would make the abstract's first sentence
about baselines longer without changing any number in it.

---

## 6. Two superlatives print without the scope that makes them true

### 6a. "Kwon--Golden--Wasil is the strongest classical estimator in this study"

**Line 405**, `\subsection{Matched-domain comparison}`

**Correct on the matched domain, reversed on both other panels.** On the panel
this sentence sits in — 2D `random`, sampling region supplied — Kwon--Golden--Wasil
is the best of the five classical estimators at **5.40\%** MAPE against BHH 8.89,
Cavdar--Sokol 10.79, Daganzo 16.80 and Chien 18.54
(`paper_tooling/tables/table_classical_matched.csv`, panel B). On the two panel-A
rows it is the **worst** of the five: **41.71\%** on the full 2D benchmark, where
BHH is 23.82, and **54.43\%** on TSPLIB EUC\_2D, where Cavdar--Sokol is 23.20
(`table_classical_full.csv`).

**Why not rebuilt.** The sentence is inside the matched-domain subsection, two
sentences after the panel is defined, and the immediately preceding sentence
prints the 41.71-to-5.40 collapse that establishes the scope. The paragraph reads
correctly; only the sentence lifted out of it does not.

### 6b. "the $\alpha=1$ floor \dots is the weakest MST-informed estimator available"

**Line 242**, `\subsection{Benchmark Models}`, constant-ratio paragraph

**False on at least two printed rows.** On TSPLIB EUC\_2D the floor reaches
**11.35\%** MAPE and **4.22\%** SDPE, beating the calibrated $\hat\rho(d)$ row on
both (11.97\% / 5.33\%) and beating the asymptotic ratio, the fixed
$\alpha=1.136$ row and the same-feature linear model on SDPE
(4.75\%, 4.80\% and 18.36\%). On the `grid` generator of the 2D benchmark it has
the **lowest MAPE of every row in the class at 4.50\%**, ahead of Daganzo
(sampling region) at 5.67, Fixed MST scaling at 5.85, the same-feature linear
model at 6.51 and GART 2.0 itself at 7.11
(`paper_tooling/tables/table_2d_by_genclass.csv`).

**Why not rebuilt.** The sentence is a definition of the row's role in the
baseline set — it is the floor, the thing every other MST-informed estimator is
supposed to improve on — and the manuscript already discloses both exceptions
where they matter. Section 4.9 states outright that `grid` "is the only row
anywhere in this paper on which the $\alpha=1$ floor beats the shipped model" and
prints the 4.50\%. The TSPLIB reversal is visible in `tab:tsplib_by_size`, which
the same paragraph points at. Rewriting the definition to carry its own
exceptions would move a disclosure that is already made, twice, to the places a
reader checks it.
