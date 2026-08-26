"""Write the Lagrangian 1-tree bound into Section 1.2 and Table benchmark_models.

Byte-level read/replace/write. A bash heredoc has corrupted ``\\ref`` into a
literal CR + "ef{" four times in this repo; this script exists so that cannot
happen again. Every edit is anchored on a unique substring and asserted to
apply exactly once.
"""

from __future__ import annotations

from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

# ---------------------------------------------------------------------------
# 1. Literature review: the bound the survey never mentioned.
# ---------------------------------------------------------------------------
LIT_ANCHOR = (
    b"Section~\\ref{sec:evaluation} evaluates each of these source models "
    b"numerically, on the full benchmarks and on the subset where its published "
    b"assumptions hold (Table~\\ref{tab:benchmark_models}).\n"
)

LIT_ADDITION = (
    b"\n"
    b"A separate classical line does not estimate the tour at all: it bounds it, "
    b"and it is the strongest comparator this paper has. \\citet{heldkarp1970,heldkarp1971} "
    b"relax the Hamiltonian-cycle constraint to a \\emph{1-tree} -- a spanning tree on all "
    b"but one node, plus the two cheapest edges at that node -- and introduce node penalties "
    b"$\\pi$ under which every tour's cost shifts by the same constant $2\\sum_i\\pi_i$. "
    b"Maximising the concave piecewise-linear dual $w(\\pi)=L_{\\mathrm{1tree}}(\\pi)-2\\sum_i\\pi_i$ "
    b"yields the Held--Karp bound, and because $w(\\pi)\\le L_{\\mathrm{TSP}}$ holds at "
    b"\\emph{every} $\\pi$, an ascent stopped anywhere returns a certificate rather than a "
    b"prediction. Three properties separate it from every estimator surveyed above. It "
    b"requires no training corpus and no distributional assumption, so there is no split to "
    b"contaminate and no domain to leave. It is one-sided and proved, where an estimator is "
    b"two-sided and unguaranteed. And its cost is a knob rather than a property of the "
    b"instance: $\\Theta(k n^{2})$ for $k$ ascent iterations, so accuracy is bought "
    b"continuously instead of being fixed by a closed form. The relaxation also never invokes "
    b"the triangle inequality, so the bound is defined wherever a symmetric distance matrix "
    b"is -- including the non-Euclidean instances that Section~\\ref{sec:application} reaches "
    b"only through an embedding. The ascent is where the literature divides. "
    b"\\citet{heldkarp1971} give the subgradient $\\deg_i-2$; \\citet{volgenant1982} supply a "
    b"step schedule that consults no upper bound, to which \\citet{helsgaun2000} adds "
    b"direction smoothing; \\citet{polyak1969} instead sizes the step from the gap to a known "
    b"feasible solution, which converges faster wherever a good tour is cheap to build. We "
    b"score both schedules as first-class rows on every benchmark in this paper "
    b"(Table~\\ref{tab:benchmark_models}), and Section~\\ref{sec:frontier} reports the "
    b"consequence: on part of the domain the bound is both cheaper and more accurate than the "
    b"estimator proposed here.\n"
)

# ---------------------------------------------------------------------------
# 2. Table: name the ascent on the existing row, add the Polyak row.
# ---------------------------------------------------------------------------
TABLE_OLD = (
    b"Held--Karp 1-tree, budget $k$ & $L_{\\mathrm{1tree}}(\\pi_k)-2\\sum_i\\pi_i$ after $k$ "
    b"ascent steps & Any metric, any $d$; certificate & $\\Theta(k n^2 d)$ & "
    b"\\citet{heldkarp1970,heldkarp1971} \\\\\n"
    b"Calibrated 1-tree & $c_k$ times that bound, $c_k$ fitted per budget on the training "
    b"split & As trained & $\\Theta(k n^2 d)$ & This work \\\\\n"
)

TABLE_NEW = (
    b"Held--Karp 1-tree, V\\&J step, budget $k$ & $L_{\\mathrm{1tree}}(\\pi_k)-2\\sum_i\\pi_i$ "
    b"after $k$ ascent steps; step schedule consults no upper bound & Any symmetric "
    b"distances, any $d$; certificate & $\\Theta(k n^2 d)$ & "
    b"\\citet{heldkarp1970,heldkarp1971,volgenant1982,helsgaun2000} \\\\\n"
    b"Held--Karp 1-tree, Polyak step, budget $k$ & The same bound; step "
    b"$\\gamma(\\mathrm{UB}-w)/\\lVert g\\rVert^{2}$ against a constructive tour & Any "
    b"symmetric distances, any $d$; certificate & $\\Theta(k n^2 d)$ & "
    b"\\citet{heldkarp1970,polyak1969} \\\\\n"
    b"Calibrated 1-tree & $c_k$ times either bound, $c_k$ fitted per budget on the training "
    b"split & As trained & $\\Theta(k n^2 d)$ & This work \\\\\n"
)

# ---------------------------------------------------------------------------
# 3. The declaration paragraph above the table, so prose and table agree.
# ---------------------------------------------------------------------------
DECL_OLD = (
    b"The Held--Karp 1-tree bound returns a proven lower bound on the optimal tour rather "
    b"than a prediction of it, at a cost fixed by an explicit ascent budget $k$. We score it "
    b"on the multidimensional benchmark and on TSPLIB EUC\\_2D, both raw and scaled by a "
    b"single constant fitted per budget on the training split, and "
    b"Section~\\ref{sec:frontier} reports it in full."
)

DECL_NEW = (
    b"The Held--Karp 1-tree bound returns a proven lower bound on the optimal tour rather "
    b"than a prediction of it, at a cost fixed by an explicit ascent budget $k$. Two rows "
    b"carry it, differing only in the step rule of the subgradient ascent: the "
    b"Volgenant--Jonker schedule, which consults no upper bound, and the Polyak step, which "
    b"sizes each move from the gap to a constructively built tour. Both are scored on all "
    b"four benchmarks -- the 2D diverse set, the multidimensional set, TSPLIB EUC\\_2D and "
    b"the non-EUC\\_2D TSPLIB instances -- both raw and scaled by a single constant fitted "
    b"per budget on the training split, and Section~\\ref{sec:frontier} reports them in full."
)

EDITS = [
    ("literature review", LIT_ANCHOR, LIT_ANCHOR + LIT_ADDITION),
    ("benchmark_models table", TABLE_OLD, TABLE_NEW),
    ("declaration paragraph", DECL_OLD, DECL_NEW),
]


def _eol(pattern: bytes, crlf: bool) -> bytes:
    """Patterns above are written with LF. The manuscript is CRLF on this
    machine, so every newline in an anchor has to become CRLF or nothing
    matches -- and every newline we *insert* has to match the file's own
    convention or the next anchor stops resolving."""
    return pattern.replace(b"\n", b"\r\n") if crlf else pattern


def main() -> None:
    blob = TEX.read_bytes()
    before = len(blob)
    cr, lf = blob.count(b"\r\n"), blob.count(b"\n")
    if cr not in (0, lf):
        raise SystemExit(f"ABORT: mixed line endings ({cr} CRLF of {lf} LF)")
    crlf = cr == lf and lf > 0
    print(f"  line endings: {'CRLF' if crlf else 'LF'} ({lf} lines)")

    refs_before = blob.count(b"\\ref{")
    expected_delta = sum(new.count(b"\\ref{") - old.count(b"\\ref{")
                         for _, old, new in EDITS)

    for name, old, new in EDITS:
        old_b, new_b = _eol(old, crlf), _eol(new, crlf)
        hits = blob.count(old_b)
        if hits != 1:
            raise SystemExit(f"ABORT [{name}]: anchor found {hits} times, expected 1")
        blob = blob.replace(old_b, new_b, 1)
        print(f"  applied: {name}")

    lf2 = blob.count(b"\n")
    if blob.count(b"\r\n") != (lf2 if crlf else 0):
        raise SystemExit("ABORT: line endings went mixed; refusing to write")
    # The exact corruption this repo has hit four times: a CR lands between the
    # backslash and "ef{", turning \ref into a line break plus literal text.
    if b"\ref{" in blob.replace(b"\\ref{", b""):
        raise SystemExit("ABORT: CR+'ef{' present; \\ref was corrupted")
    refs_after = blob.count(b"\\ref{")
    if refs_after != refs_before + expected_delta:
        raise SystemExit(f"ABORT: \\ref count {refs_before} -> {refs_after}, "
                         f"expected +{expected_delta}")
    print(f"  \\ref{{}} count: {refs_before} -> {refs_after} (+{expected_delta})")
    TEX.write_bytes(blob)
    print(f"\nwrote {TEX}  {before} -> {len(blob)} bytes ({lf2 - lf} lines added)")


if __name__ == "__main__":
    main()
