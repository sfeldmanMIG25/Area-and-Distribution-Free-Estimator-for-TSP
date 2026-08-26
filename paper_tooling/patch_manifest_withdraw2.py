"""Repair the seven broken anchors and register every result numeral the rewrite added.

Second manifest pass. ``patch_manifest_withdraw.py`` re-pointed the claims whose
generator key vanished; this one fixes the anchors whose sentence was reworded
and registers the numbers the new sentences assert, so nothing in the
matched-domain or TSPLIB paragraphs rides on the backlog.
"""

from __future__ import annotations

import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "prose_manifest.py"

EDITS: list[tuple[str, str]] = []


def edit(old: str, new: str) -> None:
    EDITS.append((old, new))


# -- abstract: the roster is seven baselines now, not ten -------------------
for hole in ('{v}\\%/{~}\\%, and {~}\\%/{~}\\%', '{~}\\%/{v}\\%, and {~}\\%/{~}\\%',
             'and {v}\\%/{~}\\%', 'and {~}\\%/{v}\\%'):
    edit(f'anchor=r"{hole}, the lowest of a ten-baseline roster"',
         f'anchor=r"{hole}, the lowest of a seven-baseline roster"')

# -- the extended-block ablation now reports one matched panel --------------
edit(
    '''    Claim(
        id="matched.v4.uniform_mape",
        anchor=r"on all three panels, {v}\\%/{~}\\%, {~}\\%/{~}\\% and {~}\\%/{~}\\%, and none",''',
    '''    Claim(
        id="matched.v4.uniform_mape",
        anchor=r"on this panel, {v}\\% against {~}\\% on MAPE",''')

edit(
    '''    Claim(
        id="matched.v4.uniform_sdpe",
        anchor=r"on all three panels, {~}\\%/{v}\\%, {~}\\%/{~}\\% and {~}\\%/{~}\\%, and none",
        no_generator=_MATCHED("LGBM_V4", "uniform (all 210 random)", "SDPE 1.8974774"),
        tol=("dp", 2),
    ),''',
    '''    Claim(
        id="matched.v4.uniform_sdpe",
        anchor=r"on MAPE and {v}\\% against {~}\\% on SDPE",
        no_generator=_MATCHED("LGBM_V4", "uniform (all 210 random)", "SDPE 1.8986759"),
        tol=("dp", 2),
    ),
    # The GART 2.0 side of that same comparison IS bankable, so it is checked
    # rather than left to the ablation's no_generator note.
    Claim(
        id="matched.v4.gart_mape_panel",
        anchor=r"on this panel, {~}\\% against {v}\\% on MAPE",
        expect="bank:classical_b_random_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.v4.gart_sdpe_panel",
        anchor=r"on MAPE and {~}\\% against {v}\\% on SDPE",
        expect="bank:classical_b_random_gart_2_0_sdpe_pct",
        tol=("dp", 2),
    ),''')

edit(
    '        no_generator=_MATCHED("LGBM_V4", "uniform (all 210 random)", "MAPE 1.2308292"),',
    '        no_generator=_MATCHED("LGBM_V4", "uniform (all 210 random)", "MAPE 1.2308292"),')

# -- the headline factor: single backslash, the .tex has ``i.i.d.\ uniform`` -
edit(
    'anchor=r"A factor of {v} over \\c{C}avdar--Sokol on i.i.d.\\\\ uniform draws",',
    'anchor=r"A factor of {v} over \\c{C}avdar--Sokol on i.i.d.\\ uniform draws",')

# -- everything the two rewritten result paragraphs now asserts -------------
NEW_CLAIMS = '''
    # -- Section 4.4 / 4.5, rewritten for the two surviving classical rows ---
    # Every numeral in those two paragraphs is registered here. They were
    # rewritten wholesale when Daganzo, Chien and Kwon--Golden--Wasil were
    # withdrawn, so none of them can ride on the recorded backlog.
    Claim(
        id="results.tsplib.cavdar_mape",
        anchor=r"\\c{C}avdar--Sokol obtains {v}\\% MAPE and BHH",
        expect="bank:classical_a_tsplib_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.bhh_mape",
        anchor=r"MAPE and BHH {v}\\%. TSPLIB instances",
        expect="bank:classical_a_tsplib_bhh_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.bhh_mspe",
        anchor=r"Both overpredict, by $+{v}$\\% and",
        expect="bank:classical_a_tsplib_bhh_mspe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="results.tsplib.cavdar_mspe",
        anchor=r"by $+{~}$\\% and $+{v}$\\% respectively",
        expect="bank:classical_a_tsplib_cavdar_sokol_mspe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.bhh.full_mape",
        anchor=r"BHH falls from {v}\\% MAPE on the full 2D benchmark",
        expect="bank:classical_a_2d_bhh_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.bhh.uniform_mape",
        anchor=r"on the full 2D benchmark to {v}\\% here",
        expect="bank:classical_b_random_bhh_sampling_region_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.bhh.uniform_mspe",
        anchor=r"a systematic underprediction, $-{v}$\\% signed against",
        expect="= -1 * {classical_b_random_bhh_sampling_region_mspe_pct}",
        tol=("dp", 2),
        note="Adverse result kept explicit: BHH's residual on its own matched "
             "domain is a bias, not noise -- the signed error exceeds the "
             "dispersion. Printed as a magnitude after a literal minus sign, "
             "hence the sign flip.",
    ),
    Claim(
        id="matched.bhh.uniform_sdpe",
        anchor=r"\\% signed against {v}\\% SDPE, which is what an asymptotic",
        expect="bank:classical_b_random_bhh_sampling_region_sdpe_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.cavdar.full_mape",
        anchor=r"\\c{C}avdar--Sokol falls from {v}\\% to",
        expect="bank:classical_a_2d_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.cavdar.uniform_mape",
        anchor=r"\\c{C}avdar--Sokol falls from {~}\\% to {v}\\%, and here only",
        expect="bank:classical_b_random_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.cavdar.uniform_sdpe",
        anchor=r"the least even: at {v}\\% SDPE its dispersion",
        expect="bank:classical_b_random_cavdar_sokol_sdpe_pct",
        tol=("dp", 2),
        note="Adverse result: the widest dispersion in the matched panel, "
             "wider than the alpha=1 floor's. Registered so a later run cannot "
             "quietly drop the qualification.",
    ),
    Claim(
        id="matched.cavdar.uniform_medape",
        anchor=r"its median absolute error of {v}\\% sits at under a third",
        expect="bank:classical_b_random_cavdar_sokol_medape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.uniform_domain.n",
        anchor=r"suggest. On the {v} uniform instances it obtains",
        expect="bank:classical_b_random_gart_2_0_n",
        tol="exact",
    ),
    Claim(
        id="matched.uniform_domain.n_restated",
        anchor=r"value is {~}. A set of {v} instances reproduces the sign",
        expect="bank:classical_b_random_gart_2_0_n",
        tol="exact",
    ),
    Claim(
        id="conclusion.cavdar_factor",
        anchor=r"a factor of {v} over \\c{C}avdar--Sokol on the planar i.i.d.",
        expect="= {classical_b_random_cavdar_sokol_mape_pct}"
               " / {classical_b_random_gart_2_0_mape_pct}",
        tol=("dp", 1),
        note="Conclusion restatement of matched.cavdar_factor.",
    ),
'''

ANCHOR_TAIL = '''    Claim(
        id="matched.cavdar_factor",'''
edit(ANCHOR_TAIL, NEW_CLAIMS.lstrip("\\n") + ANCHOR_TAIL)


def main() -> int:
    text = MANIFEST.read_text(encoding="utf-8")
    failures: list[str] = []
    for i, (old, new) in enumerate(EDITS):
        if old == new:
            continue
        count = text.count(old)
        if count != 1:
            failures.append(f"edit {i}: {count} occurrences of {old.strip()[:90]!r}")
            continue
        text = text.replace(old, new)
    if failures:
        print("ABORTED, nothing written:")
        for f in failures:
            print("  " + f)
        return 1
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"applied {len([e for e in EDITS if e[0] != e[1]])} edits to {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
