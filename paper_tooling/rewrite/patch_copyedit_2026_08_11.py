"""Targeted copy-edit of Area_Free_Main.tex: six findings + the constraint-transfer control.

Byte-level, CRLF-safe: read_bytes / replace / write_bytes with intra-line anchors,
never a heredoc.  Every replacement is asserted to occur exactly once, and the
script reports applied-vs-missing and refuses to write if any anchor is missing
or ambiguous.

F1  Conclusion retracts "cost labels remain internally consistent", which section
    3.3 no longer supports.
F2  The 100% consistency figure is enforced, not learned; that caveat now travels
    to the abstract, 1.3, the Discussion and the Conclusion, together with the
    constraint-transfer control (paper_tooling/constraint_transfer_protocol.md).
F3  Broken anaphor: "those last three" named the wrong three baselines.
F4  The probe grid does not reach four times the largest evaluated n.
F5  Conclusion omitted that GART 2.0 also trails the extended-block variant on
    both ranking measures.
F6  A Pearson coefficient was labelled Spearman; the label is kept and the value
    corrected to the Spearman value on the same slice.
"""
from __future__ import annotations

from pathlib import Path

TEX = Path(r"D:\Area-and-Distribution-Free-Estimator-for-TSP\paper_reference\Area_Free_Main.tex")

# (tag, old, new).  Anchors are intra-line, so no CRLF appears inside any of them.
EDITS: list[tuple[str, bytes, bytes]] = [

    # ---- F2 abstract: split the false decision rule off its evidence --------
    ("F2-abstract-split",
     rb"Aggregate error therefore does not select which estimator to ship, and "
     rb"consistency does: on a ceteris-paribus sweep GART 2.0 is",
     rb"Aggregate error therefore does not select which estimator to ship. On a "
     rb"ceteris-paribus sweep GART 2.0 is"),

    # ---- F2 abstract: carry the caveat and the transfer control ------------
    ("F2-abstract-caveat",
     rb"45.8\%/83.2\% for the network. The accuracy is not free:",
     rb"45.8\%/83.2\% for the network. That 100\% is enforced inside the tree "
     rb"builder rather than learned, so it verifies that the constraint was "
     rb"applied and is not evidence of an inductive bias, and a control "
     rb"registered before the first fit refits the boosted variant with the same "
     rb"constraints and recovers 100\% on both axes at every one of seven seeds "
     rb"for at most 0.049 percentage points of MAPE: consistency is a flag the "
     rb"rival can set, so it does not select the estimator either. What selects "
     rb"GART 2.0 is TSPLIB EUC\_2D accuracy, where a matched refit of it leads "
     rb"that constrained variant 2.567\% against 2.959\% on median MAPE with "
     rb"disjoint seed bands. The accuracy is not free:"),

    # ---- F3 introduction: name the three baselines the anaphor meant -------
    ("F3-intro-anaphor",
     rb"32-feature block. Two of those last three, the refitted network and the "
     rb"extended-block variant, are the strongest comparators we have;",
     rb"32-feature block. Three of those seventeen enter the baseline set only in "
     rb"this revision: the two refits on GART 2.0's own feature vector and the "
     rb"extended-block variant. Two of the three, the refitted network and the "
     rb"extended-block variant, are the strongest comparators we have;"),

    # ---- F2 introduction: carry the caveat and the transfer control --------
    ("F2-intro-caveat",
     rb"and it is the only one of the four that holds on both axes "
     rb"(Section~\ref{subsec:model_training}). The accuracy also costs time:",
     rb"and it is the only one of the four that holds on both axes "
     rb"(Section~\ref{subsec:model_training}). Those constraints are enforced "
     rb"inside the tree builder, so GART 2.0's figure verifies that the "
     rb"constraint was applied rather than demonstrating an inductive bias, and "
     rb"the two rival figures are single-seed readings of the released "
     rb"artifacts. The property is also not exclusive to this model. A control "
     rb"registered before the first fit sets the same two constraints on the "
     rb"extended-block variant and recovers 100\% on both axes at all seven "
     rb"seeds, for a median 0.039 percentage points of 2D MAPE and at most 0.049 "
     rb"points on any stratum, and the constrained variant keeps its 2D lead "
     rb"over a matched refit of GART 2.0. Consistency therefore does not select "
     rb"which estimator to ship any more than aggregate error does; what selects "
     rb"GART 2.0 is TSPLIB EUC\_2D, where that matched refit leads the "
     rb"constrained variant 2.567\% against 2.959\% on median MAPE across seven "
     rb"seeds, with disjoint bands and a paired Wilcoxon $p=0.00021$ over 78 "
     rb"instances. The accuracy also costs time:"),

    # ---- F4 methods: the probe grid does not reach 4x the evaluated n ------
    ("F4-probe-reach",
     rb"The grids run to twice the largest dimension and four times the largest "
     rb"node count anything in this paper is evaluated at, deliberately, because "
     rb"the guarantee is wanted outside the evaluated range and not only inside "
     rb"it.",
     rb"The dimension grid runs to twice the largest dimension anything in this "
     rb"paper is evaluated at; the node-count grid runs to four times the largest "
     rb"node count in the synthetic corpora, and stops well short of the largest "
     rb"TSPLIB instance this paper scores. Both overshoot the synthetic "
     rb"evaluation range deliberately, because the guarantee is wanted outside "
     rb"that range and not only inside it."),

    # ---- F2 methods: the seed band, and the constraint set on the rival ----
    ("F2-methods-transfer",
     rb"Of the four estimators the probe covers, GART 2.0 is nonetheless the only "
     rb"one that is non-increasing on both axes. The probe is only meaningful for "
     rb"an estimator",
     rb"Of the four estimators the probe covers, GART 2.0 is nonetheless the only "
     rb"one that is non-increasing on both axes. Those three figures are "
     rb"single-seed values read off the released artifacts, and the boosted "
     rb"variant's pair is the favourable extreme of its own seed band: refit "
     rb"unconstrained at seven seeds, that variant holds a median 5.4\% of the "
     rb"dimension sweeps and 53.6\% of the size sweeps, so the two figures "
     rb"printed for it above are that band's maxima and the true contrast is "
     rb"wider than they state. The complementary control is the constraint set on "
     rb"a rival rather than removed from this model, and under a protocol fixed "
     rb"before the first fit "
     rb"(\texttt{paper\_tooling/constraint\_transfer\_protocol.md}) it transfers "
     rb"completely: the extended-block variant refitted with the same two "
     rb"non-increasing constraints returns 100\% non-increasing sweeps on both "
     rb"axes at all seven seeds, at a maximum raw violation of zero. The "
     rb"constraint costs that variant a median 0.039 percentage points of MAPE on "
     rb"the 2D benchmark, 0.049 on TSPLIB EUC\_2D and 0.046 on the screened "
     rb"non-Euclidean set, and costs it nothing on the multidimensional split, "
     rb"where the median paired difference favours the constrained fit. Nor does "
     rb"it cost the variant its accuracy: against a matched refit of GART 2.0 the "
     rb"constrained variant keeps the 2D benchmark, 2.751\% against 2.890\% on "
     rb"median MAPE with disjoint seven-seed bands, and loses TSPLIB EUC\_2D, "
     rb"2.959\% against 2.567\%, disjoint bands at a paired Wilcoxon $p=0.00021$ "
     rb"over 78 instances. Monotone consistency is therefore a flag any "
     rb"competitor on this feature block can set, not a property this model holds "
     rb"alone, and it does not by itself select which estimator to ship. The "
     rb"probe is only meaningful for an estimator"),

    # ---- F6 results: the coefficient is Spearman, so print the Spearman ----
    ("F6-spearman-value",
     rb"(Spearman 0.80, rising monotonically",
     rb"(Spearman 0.86, rising monotonically"),

    # ---- F2 discussion: the enforced caveat and the removable objection ----
    ("F2-discussion-caveat",
     rb"against 100\% on both axes for the shipped model. The second is cost:",
     rb"against 100\% on both axes for the shipped model, a figure enforced inside "
     rb"the tree builder and therefore evidence that the constraint was applied "
     rb"rather than of an inductive bias. That objection is removable, and "
     rb"Section~\ref{subsec:model_training} removes it: setting the same "
     rb"constraints on the variant restores 100\% on both axes at all seven seeds "
     rb"for at most 0.049 percentage points of MAPE on any stratum. The second is "
     rb"cost:"),

    # ---- F2 discussion: withdraw the shipping argument the control refutes -
    ("F2-discussion-argument",
     rb"A candidate that is more accurate on three strata and cannot be relied on "
     rb"to move the right way in $n$ or $d$ is the argument of this paper rather "
     rb"than a counterexample to it: aggregate MAPE does not settle which "
     rb"estimator to ship.",
     rb"A candidate that is more accurate on three strata and could not be relied "
     rb"on to move the right way in $n$ or $d$ would be the argument of this paper "
     rb"rather than a counterexample to it, and an earlier revision made that "
     rb"argument. The constraint-transfer control withdraws it: the constraint is "
     rb"portable at negligible cost, the constrained variant is non-increasing on "
     rb"every sweep of both axes at all seven seeds, and it still keeps the 2D "
     rb"benchmark against a matched refit of GART 2.0. Aggregate MAPE does not "
     rb"settle which estimator to ship, and neither does monotonicity; what "
     rb"settles it here is accuracy on TSPLIB EUC\_2D, the stratum closest to "
     rb"deployment."),

    # ---- F2 conclusion: the property is not exclusive ----------------------
    ("F2-conclusion-exclusive",
     rb"What GART 2.0 leads on, and leads on alone, is that last property: of the "
     rb"four estimators built on its feature vector or an extension of it, it is "
     rb"the only one whose predicted $\alpha$ is non-increasing in both dimension "
     rb"and node count on every sweep of the probe. The honest size of that "
     rb"advantage depends on the benchmark.",
     rb"Of the four estimators built on its feature vector or an extension of it, "
     rb"GART 2.0 as released is the only one whose predicted $\alpha$ is "
     rb"non-increasing in both dimension and node count on every sweep of the "
     rb"probe. That property is enforced inside the tree builder rather than "
     rb"learned, and it is not exclusive: a control registered before the first "
     rb"fit refits the extended-block variant with the same constraints and "
     rb"recovers every sweep on both axes at all seven seeds for at most 0.049 "
     rb"percentage points of MAPE, and the constrained variant keeps its 2D lead "
     rb"(Section~\ref{subsec:model_training}). What selects GART 2.0 is therefore "
     rb"not consistency but accuracy on TSPLIB EUC\_2D, the stratum closest to "
     rb"deployment, where a matched refit leads that constrained variant 2.567\% "
     rb"against 2.959\% on median MAPE with disjoint seven-seed bands. The honest "
     rb"size of that accuracy advantage depends on the benchmark."),

    # ---- F5 conclusion: the omitted second estimator it trails -------------
    ("F5-conclusion-rank",
     rb"It leads the two GART variants on the wider 10\% band and on global rank "
     rb"correlation and trails the production-feature network on both.",
     rb"It leads the two GART variants on the wider 10\% band and on global rank "
     rb"correlation, and trails both the production-feature network and the "
     rb"extended-block variant on each: the variant orders 82.08\% of that band "
     rb"against GART 2.0's 79.25\%, at Spearman 0.999248 against 0.999222 and "
     rb"Kendall 0.987843 against 0.985845."),

    # ---- F1 conclusion: match the body, which withdrew "internally consistent"
    ("F1-conclusion-provenance",
     rb"A corpus audit found 184 instances whose stored tour is inconsistent with "
     rb"their released coordinates; their cost labels remain internally "
     rb"consistent and removing them moves every reported metric by less than "
     rb"0.06 percentage points, but the released artifact should carry the audit.",
     rb"A corpus audit found 184 instances whose stored tour is inconsistent with "
     rb"their released coordinates. Their stored costs are not thereby shown to be "
     rb"sound: what carries them is the agreement of the two solvers on 166 of "
     rb"those rows, and the per-cell $\alpha$ comparison cannot corroborate it, "
     rb"because 17 of the 29 affected cells depart from their unaffected "
     rb"neighbours by more than 1.6 standard deviations "
     rb"(Section~\ref{subsec:provenance}). Removing them moves every reported "
     rb"metric by less than 0.06 percentage points, but the released artifact "
     rb"should carry the audit."),
]


def main() -> int:
    raw = TEX.read_bytes()
    before = len(raw)
    applied: list[str] = []
    missing: list[str] = []
    ambiguous: list[tuple[str, int]] = []

    for tag, old, new in EDITS:
        hits = raw.count(old)
        if hits == 0:
            missing.append(tag)
            continue
        if hits > 1:
            ambiguous.append((tag, hits))
            continue
        raw = raw.replace(old, new, 1)
        applied.append(tag)

    print(f"edits declared : {len(EDITS)}")
    print(f"applied        : {len(applied)}")
    for t in applied:
        print(f"  + {t}")
    print(f"missing        : {len(missing)}")
    for t in missing:
        print(f"  ! {t}")
    print(f"ambiguous      : {len(ambiguous)}")
    for t, k in ambiguous:
        print(f"  ! {t} ({k} matches)")

    if missing or ambiguous:
        print("\nNOT WRITTEN: every anchor must match exactly once.")
        return 1

    TEX.write_bytes(raw)
    print(f"\nwritten: {before} -> {len(raw)} bytes (+{len(raw) - before})")
    print(f"CRLF pairs: {raw.count(b'\\r\\n')}, bare LF: "
          f"{raw.count(b'\\n') - raw.count(b'\\r\\n')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
