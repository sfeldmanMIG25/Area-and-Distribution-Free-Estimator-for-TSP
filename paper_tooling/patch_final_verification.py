"""Final-verification fixes to Area_Free_Main.tex.

Byte-level read/replace/write. A bash heredoc has corrupted ``\\ref`` into a
carriage return plus ``ef{`` four times in this repository, so every manuscript
patch goes through this path instead.

Four edits, each an exact-match replacement that must fire exactly once:

1. ROSTER COHERENCE. Section 4.2 still described the certified bound as being
   reported "outside the ten-baseline count". The roster has been seven since
   Daganzo, Chien, Kwon--Golden--Wasil and Cavdar_region were withdrawn, and the
   abstract, the contribution paragraph, the benchmark-model section and the
   conclusion all say seven. This was the last survivor.

2. INTERNAL CONTRADICTION. The matched-domain paragraph claimed Cavdar--Sokol
   was "the strongest classical estimator in this study on both aggregate
   metrics" and then, in the same sentence, that its dispersion was the widest
   in the panel. Both cannot hold: on the 210 uniform instances it reads 8.16%
   MAPE against BHH's 8.91% but 14.28% SDPE against BHH's 7.76%. It leads on
   MAPE and on median, not on dispersion, and the sentence now says so.

   The same sentence put its median absolute error "at under a third of its
   mean". 2.84/8.16 = 0.348, which is over a third, not under it. Corrected to
   "just over".

3. VERIFICATION SCOPE. The availability appendix claimed that *every* results
   table is generated and written in mechanically, and that the pass re-derives
   all 1,431 cells. Neither half was right. tab:paired is typed by hand (its
   values are correct -- all ten rows reproduce paired_tests.csv -- but it is
   not spliced), and --check covered six of the eight tables that *are*
   generated, leaving tab:classical and tab:genclass unread. build_paper_tables
   now checks all eight, which is 1,921 cells, and check_frontier_tables banks
   its own 182. The sentence now states the covered set exactly.

4. NON-ASCII. The only two non-ASCII bytes in the manuscript were a pair of raw
   U+2014 em dashes in the feature-selection appendix, where the other 70 em
   dashes in the file are written ``---``. Normalised, leaving the source pure
   ASCII.
"""
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper_reference" / "Area_Free_Main.tex"

EDITS: list[tuple[str, bytes, bytes]] = [
    (
        "1. roster count: ten -> seven",
        b"We report it outside the ten-baseline count because it answers a "
        b"different question than an estimator does.",
        b"We report it outside the seven-baseline count because it answers a "
        b"different question than an estimator does.",
    ),
    (
        "2. Cavdar--Sokol: metric-by-metric standing, and a third that is over a third",
        b"It is the strongest classical estimator in this study on both aggregate "
        b"metrics, and it is also the least even: at 14.28\\% SDPE its dispersion "
        b"is the widest in the panel, and its median absolute error of 2.84\\% "
        b"sits at under a third of its mean, so it is close on most uniform "
        b"instances and badly wrong on a few.",
        b"It has the lower MAPE of the two classical estimators on every panel of "
        b"Table~\\ref{tab:classical}, and the lower median absolute error here, but "
        b"not the lower dispersion, and it is also the least even: at 14.28\\% SDPE "
        b"its dispersion is the widest in the panel, and its median absolute error "
        b"of 2.84\\% sits at just over a third of its mean, so it is close on most "
        b"uniform instances and badly wrong on a few.",
    ),
    (
        "3. verification scope and cell count",
        b"Every results table in this paper is generated from the scored benchmark "
        b"outputs and written into the manuscript mechanically rather than typed; a "
        b"verification pass re-derives all 1{,}431 table cells from those outputs "
        b"and reports any that disagrees with the typeset value. Scalars quoted in "
        b"the running prose, such as the timing split and the audit counts, are "
        b"transcribed by hand from the same artifacts and are not covered by that "
        b"check.",
        b"The eight per-benchmark results tables are generated from the scored "
        b"benchmark outputs and written into the manuscript mechanically rather "
        b"than typed; a verification pass re-derives all 1{,}921 table cells from "
        b"those outputs and reports any that disagrees with the typeset value, and "
        b"a second pass does the same for the 182 cells of the cost/accuracy "
        b"ladders of Section~\\ref{sec:frontier}. The remaining tables, and scalars "
        b"quoted in the running prose such as the timing split and the audit "
        b"counts, are transcribed by hand from the same artifacts and are not "
        b"covered by either check.",
    ),
    (
        "4. raw U+2014 em dashes -> ---",
        b"higher-order structures\xe2\x80\x94such as clusters, filaments, and "
        b"hubs\xe2\x80\x94without performing",
        b"higher-order structures---such as clusters, filaments, and "
        b"hubs---without performing",
    ),
]


def main() -> int:
    raw = TEX.read_bytes()
    before = len(raw)
    for name, old, new in EDITS:
        hits = raw.count(old)
        if hits != 1:
            print(f"ABORT: {name}: expected exactly 1 match, found {hits}")
            return 1
        raw = raw.replace(old, new)
        print(f"  applied: {name}")

    # Guards. The manuscript is CRLF throughout; a patch that introduced a lone
    # CR is the exact corruption this script exists to avoid.
    if raw.count(b"\r\n") != raw.count(b"\r"):
        print("ABORT: lone carriage return introduced")
        return 1
    if raw.count(b"\r\n") != raw.count(b"\n"):
        print("ABORT: lone line feed introduced")
        return 1
    if b"\ref{" in raw.replace(b"\\ref{", b""):
        print("ABORT: carriage-return corruption of a \\ref")
        return 1
    if any(b > 0x7E for b in raw):
        print("ABORT: non-ASCII byte remains")
        return 1

    TEX.write_bytes(raw)
    print(f"wrote {TEX} ({before} -> {len(raw)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
