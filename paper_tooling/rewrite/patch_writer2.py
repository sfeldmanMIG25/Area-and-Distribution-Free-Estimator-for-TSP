r"""Single-writer pass 2: Appendix Table tab:paired.

WHY THIS TOUCHES A TABLE
------------------------
The standing rule is "do not touch table bodies, they are generated and
machine-verified (1,910 cells, 0 discrepancies)".  ``tab:paired`` is neither.
``build_paper_tables.TEX_TABLES`` verifies exactly six labels --
tab:nd_by_dim, tab:nd_by_size, tab:2d_by_size, tab:tsplib_by_size,
tab:tsplib_nonEuc, tab:rank -- and tab:paired is not among them.  It is
hand-transcribed from ``paper_tooling/tables/paired_tests.csv`` and it went
stale when the production model changed.  ``--check`` reports 1,910 cells
before and after this patch, unchanged, because none of these cells is
counted there.

Leaving it was not an option.  Every one of its fifteen data rows was the
predecessor's, and two of them now contradict the corrected prose outright:

  * TSPLIB / Asymptotic MST ratio printed $-0.28$ [$-1.05$, $+0.45$], p=0.77
    while Section 4.8 states $-1.00$ [$-1.77$, $-0.22$], p=0.0049 -- the
    result on which the paper's one distribution-free TSPLIB claim turns;
  * 2D / Neural net (same features) printed $+0.34$ [$+0.27$, $+0.43$],
    p=4.7e-38 -- the network ahead and overwhelmingly significant -- while
    the true figure for that control is $-0.09$ [$-0.16$, $-0.01$], p=0.44,
    with GART 2.0 marginally ahead and the test resolving nothing.  Section
    4.5.2 now says so in the prose; the table said the opposite on the same
    page-turn.

Every replacement value below is read from paired_tests.csv, which
build_paper_tables regenerates.  Rounding follows the table's own
conventions: mean and interval to two decimals with explicit sign, p to two
significant figures, and a Wilcoxon p that underflows to 0.0 in float64
printed as $<10^{-300}$, exactly as the pre-existing rows did.

Row provenance (table, bucket_slug, model_b) -> printed row:
  tsplib_by_size total Asymptotic_MST  -0.996751 [-1.770050,-0.223230] 4.853e-3
  tsplib_by_size total MST_Only        -8.795865 [-9.959517,-7.605456] 1.274e-13
  2d_by_size     total Asymptotic_MST  -6.275166 [-6.648949,-5.954649] 1.505e-280
  2d_by_size     total Calibrated_MST_dn
                                       -2.847272 [-3.047317,-2.663130] 7.922e-152
  2d_by_size     total NN_V3           -0.086286 [-0.163584,-0.007579] 0.4396
  2d_by_size     total MST_Only       -15.010780 [-15.386513,-14.658849] 0.0
  nd_by_size     total Calibrated_MST_dn
                                       -1.194239 [-1.225589,-1.162146] 0.0
  nd_by_size     total NN_V3           -0.554851 [-0.581520,-0.529952] 0.0
  nd_by_size     total MST_Only        -9.060881 [-9.165351,-8.956158] 0.0
  classical      b_kwon   Kwon_region  -3.366382 [-4.498621,-2.345459] 1.261e-10
  classical      b_chien  Chien_region -16.085727 [-19.702255,-12.748485] 1.563e-13
  classical      b_random BHH_region   -7.572212 [-8.542393,-6.670039] 2.075e-35
  classical      b_random Cavdar_region
                                       -9.469094 [-10.693394,-8.270897] 4.147e-36
  classical      b_random Daganzo_region
                                      -15.484990 [-16.600373,-14.372534] 1.540e-35
  classical      b_random NN_V3        -0.200056 [-0.424027,+0.018954] 0.06929
"""

from __future__ import annotations

from pathlib import Path

TEX = Path(__file__).resolve().parents[2] / "paper_reference" / "Area_Free_Main.tex"


EDITS: list[tuple[str, str, str]] = [

    # Preamble: name the control the "same features" rows actually carry, and
    # point at the refitted 31-feature controls the Discussion now quotes.
    (
        "T0 app:paired preamble",
        r"Negative values favour GART 2.0. The complete set is in \texttt{paper\_tooling/tables/paired\_tests.csv}.",
        r"Negative values favour GART 2.0. The rows labelled ``same features'' are the predecessor's 30-input controls, which is what the benchmark tables carry; the paired tests against the controls refitted on GART 2.0's production 31-feature vector, quoted in Section~\ref{subsec:results_2d}, are in \texttt{paper\_tooling/controls\_31f/paired.csv}. The complete set is in \texttt{paper\_tooling/tables/paired\_tests.csv}.",
    ),

    (
        "T1 TSPLIB vs asymptotic ratio",
        r"TSPLIB EUC\_2D & Asymptotic MST ratio & 78 & $-0.28$ [$-1.05$, $+0.45$] & 0.77 \\",
        r"TSPLIB EUC\_2D & Asymptotic MST ratio & 78 & $-1.00$ [$-1.77$, $-0.22$] & 0.0049 \\",
    ),
    (
        "T2 TSPLIB vs alpha=1",
        r"TSPLIB EUC\_2D & $L_{\mathrm{MST}}$ ($\alpha=1$) & 78 & $-8.08$ [$-9.30$, $-6.78$] & $1.4\times10^{-12}$ \\",
        r"TSPLIB EUC\_2D & $L_{\mathrm{MST}}$ ($\alpha=1$) & 78 & $-8.80$ [$-9.96$, $-7.61$] & $1.3\times10^{-13}$ \\",
    ),
    (
        "T3 2D vs asymptotic ratio",
        r"2D & Asymptotic MST ratio & 2{,}580 & $-5.85$ [$-6.21$, $-5.53$] & $7.3\times10^{-248}$ \\",
        r"2D & Asymptotic MST ratio & 2{,}580 & $-6.28$ [$-6.65$, $-5.95$] & $1.5\times10^{-280}$ \\",
    ),
    (
        "T4 2D vs calibrated rho(d,n)",
        r"2D & Calibrated MST ratio $\hat\rho(d,n)$ & 2{,}580 & $-2.42$ [$-2.60$, $-2.24$] & $5.5\times10^{-118}$ \\",
        r"2D & Calibrated MST ratio $\hat\rho(d,n)$ & 2{,}580 & $-2.85$ [$-3.05$, $-2.66$] & $7.9\times10^{-152}$ \\",
    ),
    (
        "T5 2D vs neural net (same features)",
        r"2D & Neural net (same features) & 2{,}580 & $+0.34$ [$+0.27$, $+0.43$] & $4.7\times10^{-38}$ \\",
        r"2D & Neural net (same features) & 2{,}580 & $-0.09$ [$-0.16$, $-0.01$] & 0.44 \\",
    ),
    (
        "T6 2D vs alpha=1",
        r"2D & $L_{\mathrm{MST}}$ ($\alpha=1$) & 2{,}580 & $-14.58$ [$-14.96$, $-14.23$] & $<10^{-300}$ \\",
        r"2D & $L_{\mathrm{MST}}$ ($\alpha=1$) & 2{,}580 & $-15.01$ [$-15.39$, $-14.66$] & $<10^{-300}$ \\",
    ),
    (
        "T7 ND vs calibrated rho(d,n)",
        r"ND & Calibrated MST ratio $\hat\rho(d,n)$ & 16{,}920 & $-0.94$ [$-0.97$, $-0.91$] & $<10^{-300}$ \\",
        r"ND & Calibrated MST ratio $\hat\rho(d,n)$ & 16{,}920 & $-1.19$ [$-1.23$, $-1.16$] & $<10^{-300}$ \\",
    ),
    (
        "T8 ND vs neural net (same features)",
        r"ND & Neural net (same features) & 16{,}920 & $-0.30$ [$-0.32$, $-0.28$] & $7.9\times10^{-11}$ \\",
        r"ND & Neural net (same features) & 16{,}920 & $-0.55$ [$-0.58$, $-0.53$] & $<10^{-300}$ \\",
    ),
    (
        "T9 ND vs alpha=1",
        r"ND & $L_{\mathrm{MST}}$ ($\alpha=1$) & 16{,}920 & $-8.80$ [$-8.91$, $-8.70$] & $<10^{-300}$ \\",
        r"ND & $L_{\mathrm{MST}}$ ($\alpha=1$) & 16{,}920 & $-9.06$ [$-9.17$, $-8.96$] & $<10^{-300}$ \\",
    ),
    (
        "T10 Kwon domain",
        r"2D uniform, Kwon domain & Kwon--Golden--Wasil & 80 & $-3.19$ [$-4.29$, $-2.15$] & $2.0\times10^{-9}$ \\",
        r"2D uniform, Kwon domain & Kwon--Golden--Wasil & 80 & $-3.37$ [$-4.50$, $-2.35$] & $1.3\times10^{-10}$ \\",
    ),
    (
        "T11 Chien domain",
        r"2D uniform, Chien domain & Chien & 50 & $-15.14$ [$-18.72$, $-11.97$] & $3.0\times10^{-13}$ \\",
        r"2D uniform, Chien domain & Chien & 50 & $-16.09$ [$-19.70$, $-12.75$] & $1.6\times10^{-13}$ \\",
    ),
    (
        "T12 uniform vs BHH (region)",
        r"2D uniform & BHH (sampling region) & 210 & $-7.31$ [$-8.18$, $-6.45$] & $2.5\times10^{-35}$ \\",
        r"2D uniform & BHH (sampling region) & 210 & $-7.57$ [$-8.54$, $-6.67$] & $2.1\times10^{-35}$ \\",
    ),
    (
        "T13 uniform vs Cavdar-Sokol (region)",
        r"2D uniform & \c{C}avdar--Sokol (sampling region) & 210 & $-9.20$ [$-10.38$, $-8.08$] & $5.8\times10^{-36}$ \\",
        r"2D uniform & \c{C}avdar--Sokol (sampling region) & 210 & $-9.47$ [$-10.69$, $-8.27$] & $4.1\times10^{-36}$ \\",
    ),
    (
        "T14 uniform vs Daganzo (region)",
        r"2D uniform & Daganzo (sampling region) & 210 & $-15.22$ [$-16.35$, $-14.10$] & $2.0\times10^{-35}$ \\",
        r"2D uniform & Daganzo (sampling region) & 210 & $-15.48$ [$-16.60$, $-14.37$] & $1.5\times10^{-35}$ \\",
    ),
    (
        "T15 uniform vs neural net (same features)",
        r"2D uniform & Neural net (same features) & 210 & $+0.07$ [$-0.10$, $+0.26$] & 0.73 \\",
        r"2D uniform & Neural net (same features) & 210 & $-0.20$ [$-0.42$, $+0.02$] & 0.069 \\",
    ),
]


def main() -> int:
    raw = TEX.read_bytes()
    text = raw.decode("utf-8")

    applied, missing, ambiguous = [], [], []
    for label, old, new in EDITS:
        count = text.count(old)
        if count == 0:
            missing.append(label)
            continue
        if count > 1:
            ambiguous.append((label, count))
            continue
        text = text.replace(old, new, 1)
        applied.append(label)

    if missing or ambiguous:
        print("patch_writer2: ABORTED, no bytes written")
    else:
        TEX.write_bytes(text.encode("utf-8"))

    print(f"patch_writer2: applied {len(applied)} / {len(EDITS)}, "
          f"missing {len(missing)}, ambiguous {len(ambiguous)}")
    for label in applied:
        print(f"  ok        {label}")
    for label in missing:
        print(f"  MISSING   {label}")
    for label, count in ambiguous:
        print(f"  AMBIGUOUS {label} ({count} occurrences)")
    return 1 if (missing or ambiguous) else 0


if __name__ == "__main__":
    raise SystemExit(main())
