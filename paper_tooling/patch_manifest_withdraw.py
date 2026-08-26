"""Re-register the prose claims that the estimator withdrawal moved.

Byte-safe literal replacement over ``prose_manifest.py``. Every claim whose
anchor sentence was rewritten by ``patch_withdraw_unsourced.py`` is either
re-pointed at a surviving bank key or deleted with its sentence.
"""

from __future__ import annotations

import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "prose_manifest.py"

EDITS: list[tuple[str, str]] = []


def edit(old: str, new: str) -> None:
    EDITS.append((old, new))


# -- metrics: the bias-vs-dispersion example is BHH's now, not Daganzo's -----
edit(
    '''    Claim(
        id="metrics.daganzo.sdpe_uniform",
        anchor=r"offset on uniform instances with only {v}\\% SDPE",
        expect="bank:2d_random_by_size_total_daganzo_sampling_region_sdpe_pct",
        note="'uniform instances ... given their region' is the i.i.d.-uniform "
             "subset scored with the exact sampling region, i.e. the "
             "*_sampling_region_* variant. The plain Daganzo key for the same "
             "subset reads 22.4 and would be the wrong provenance.",
    ),''',
    '''    Claim(
        id="metrics.bhh_region.mspe_uniform",
        anchor=r"BHH given the exact sampling region carries a $-{v}$\\% offset",
        expect="= -1 * {2d_random_by_size_total_bhh_sampling_region_mspe_pct}",
        tol=("dp", 2),
        note="Replaces metrics.daganzo.sdpe_uniform. The example of a bias that "
             "SDPE alone cannot see used to be Daganzo's strip constant; that "
             "estimator is withdrawn (unobtainable primary), so the sentence now "
             "uses BHH on the same i.i.d.-uniform subset, where the signed error "
             "-8.65 exceeds the 7.76 dispersion and makes the same point. The "
             "prose prints the magnitude after a literal minus sign, hence the "
             "sign flip in the expression.",
    ),
    Claim(
        id="metrics.bhh_region.sdpe_uniform",
        anchor=r"offset on uniform instances with only {v}\\% SDPE",
        expect="bank:2d_random_by_size_total_bhh_sampling_region_sdpe_pct",
        tol=("dp", 2),
        note="The *_sampling_region_* variant is the right provenance: the "
             "sentence says 'given the exact sampling region'. The plain BHH key "
             "for the same subset reads 17.63.",
    ),''')

# -- matched domain: the Chien and Kwon panels no longer exist --------------
edit(
    '''    Claim(
        id="matched.kwon_domain.gart_mape",
        anchor=r"Kwon-domain instances it obtains {v}\\% MAPE against Kwon's",
        expect="bank:classical_b_kwon_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.chien_domain.gart_mape",
        anchor=r"Chien-domain instances {v}\\% against Chien's",
        expect="bank:classical_b_chien_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.uniform_domain.gart_mape",
        anchor=r"uniform instances {v}\\% against BHH's",
        expect="bank:classical_b_random_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),''',
    '''    # matched.kwon_domain.gart_mape and matched.chien_domain.gart_mape are
    # deleted, not re-pointed. They quoted GART 2.0 on the Chien and Kwon fitted
    # node ranges; both estimators are withdrawn (unobtainable primaries), the
    # sub-domain panels they anchored are gone from tab:classical, and the
    # sentence now reports the one remaining matched panel.
    Claim(
        id="matched.uniform_domain.gart_mape",
        anchor=r"uniform instances it obtains {v}\\% MAPE against",
        expect="bank:classical_b_random_gart_2_0_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.uniform_domain.cavdar_mape",
        anchor=r"against \\c{C}avdar--Sokol's {v}\\%, BHH's",
        expect="bank:classical_b_random_cavdar_sokol_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.uniform_domain.bhh_mape",
        anchor=r"\\c{C}avdar--Sokol's {~}\\%, BHH's {v}\\% and the",
        expect="bank:classical_b_random_bhh_sampling_region_mape_pct",
        tol=("dp", 2),
    ),
    Claim(
        id="matched.uniform_domain.floor_mape",
        anchor=r"and the $\\alpha=1$ floor's {v}\\%. The extended-block",
        expect="bank:classical_b_random_l_mathrm_mst_alpha_1_mape_pct",
        tol=("dp", 2),
    ),''')

# -- the headline factor over the strongest classical estimator -------------
edit(
    '''    Claim(
        id="matched.kwon_factor",
        anchor=r"A factor of {v} over Kwon--Golden--Wasil on its home ground",
        expect="= {classical_b_kwon_kwon_golden_wasil_sampling_region_mape_pct}"
               " / {classical_b_kwon_gart_2_0_mape_pct}",
        tol=("dp", 1),
        note="5.402075 / 2.035693 = 2.654 on the 80 Kwon-domain instances.",
    ),''',
    '''    Claim(
        id="matched.cavdar_factor",
        anchor=r"A factor of {v} over \\c{C}avdar--Sokol on i.i.d.\\\\ uniform draws",
        expect="= {classical_b_random_cavdar_sokol_mape_pct}"
               " / {classical_b_random_gart_2_0_mape_pct}",
        tol=("dp", 1),
        note="Replaces matched.kwon_factor. Kwon--Golden--Wasil was the strongest "
             "classical estimator this paper reported and is now withdrawn "
             "(unobtainable primary), so the headline factor is taken over the "
             "strongest surviving one: 8.162786 / 1.313575 = 6.214 on the 210 "
             "i.i.d.-uniform instances. The factor is LARGER than the 2.7 it "
             "replaces because the comparator is weaker on its own domain, not "
             "because GART 2.0 improved -- 1.313575 is unchanged.",
    ),''')

edit(
    '''    Claim(
        id="conclusion.timing.classical_lo",
        anchor=r"expensive estimator, {~}~ms per TSPLIB EUC\\_2D instance against {v}--{~}~ms",
        expect="bank:tsplib_by_size_total_daganzo_time_ms",
        tol=("dp", 3),
    ),''',
    '''    Claim(
        id="conclusion.timing.classical_lo",
        anchor=r"expensive estimator, {~}~ms per TSPLIB EUC\\_2D instance against {v}--{~}~ms",
        expect="bank:tsplib_by_size_total_bhh_time_ms",
        tol=("dp", 3),
        note="Was keyed on Daganzo, the cheapest of the five classical rows. "
             "With three of the five withdrawn, BHH is the cheapest survivor.",
    ),''')

edit(
    '''    Claim(
        id="discussion.timing.classical_lo",
        anchor=r"same {~} instances they take {v}--{~}~ms, so GART 2.0",
        expect="bank:tsplib_by_size_total_daganzo_time_ms",
        tol=("dp", 3),
    ),''',
    '''    Claim(
        id="discussion.timing.classical_lo",
        anchor=r"same {~} instances they take {v}--{~}~ms, so GART 2.0",
        expect="bank:tsplib_by_size_total_bhh_time_ms",
        tol=("dp", 3),
        note="Was keyed on Daganzo; BHH is the cheapest surviving classical row.",
    ),''')

edit(
    '''    Claim(
        id="discussion.timing.classical_ratio_hi",
        anchor=r"so GART 2.0 costs {~} to {v} times what they do",
        expect="= {tsplib_by_size_total_gart_2_0_time_ms}"
               " / {tsplib_by_size_total_daganzo_time_ms}",
        tol=("dp", 2),
    ),''',
    '''    Claim(
        id="discussion.timing.classical_ratio_hi",
        anchor=r"so GART 2.0 costs {~} to {v} times what they do",
        expect="= {tsplib_by_size_total_gart_2_0_time_ms}"
               " / {tsplib_by_size_total_bhh_time_ms}",
        tol=("dp", 2),
    ),''')

edit(
    '''    Claim(
        id="discussion.timing.classical_ratio_gt400_lo",
        anchor=r"over the classical estimators widens to {v}--{~}",
        expect="= {tsplib_by_size_gt400_gart_2_0_time_ms}"
               " / {tsplib_by_size_gt400_kwon_golden_wasil_extrapolated_time_ms}",
        tol=("dp", 1),
    ),
    Claim(
        id="discussion.timing.classical_ratio_gt400_hi",
        anchor=r"over the classical estimators widens to {~}--{v}",
        expect="= {tsplib_by_size_gt400_gart_2_0_time_ms}"
               " / {tsplib_by_size_gt400_daganzo_time_ms}",
        tol=("dp", 1),
    ),''',
    '''    Claim(
        id="discussion.timing.classical_ratio_gt400_lo",
        anchor=r"over the classical estimators widens to {v}--{~}",
        expect="= {tsplib_by_size_gt400_gart_2_0_time_ms}"
               " / {tsplib_by_size_gt400_cavdar_sokol_time_ms}",
        tol=("dp", 1),
        note="Was keyed on Kwon (extrapolated). Cavdar--Sokol is the dearest "
             "surviving classical row above n=400 and so sets the lower ratio.",
    ),
    Claim(
        id="discussion.timing.classical_ratio_gt400_hi",
        anchor=r"over the classical estimators widens to {~}--{v}",
        expect="= {tsplib_by_size_gt400_gart_2_0_time_ms}"
               " / {tsplib_by_size_gt400_bhh_time_ms}",
        tol=("dp", 1),
    ),''')

# -- the extended-block ablation now reports one panel, not three ------------
edit(
    '''    Claim(
        id="matched.v4.kwon_mape",
        anchor=r"on all three panels, {~}\\%/{~}\\%, {v}\\%/{~}\\% and {~}\\%/{~}\\%, and none",
        no_generator=_MATCHED("LGBM_V4", "Kwon_region domain (80)", "MAPE 1.9692992"),
        tol=("dp", 2),
    ),
    Claim(
        id="matched.v4.kwon_sdpe",
        anchor=r"on all three panels, {~}\\%/{~}\\%, {~}\\%/{v}\\% and {~}\\%/{~}\\%, and none",
        no_generator=_MATCHED("LGBM_V4", "Kwon_region domain (80)", "SDPE 2.4451856"),
        tol=("dp", 2),
    ),
    Claim(
        id="matched.v4.chien_mape",
        anchor=r"on all three panels, {~}\\%/{~}\\%, {~}\\%/{~}\\% and {v}\\%/{~}\\%, and none",
        no_generator=_MATCHED("LGBM_V4", "Chien_region domain (50)", "MAPE 2.1907830"),
        tol=("dp", 2),
    ),
    Claim(
        id="matched.v4.chien_sdpe",
        anchor=r"on all three panels, {~}\\%/{~}\\%, {~}\\%/{~}\\% and {~}\\%/{v}\\%, and none",
        no_generator=_MATCHED("LGBM_V4", "Chien_region domain (50)", "SDPE 3.0379803"),
        tol=("dp", 2),
    ),''',
    '''    # matched.v4.{kwon,chien}_{mape,sdpe} are deleted with their panels. The
    # sentence reported the extended-block ablation on three matched panels; the
    # Chien and Kwon panels required those two estimators' fitted node ranges and
    # both estimators are withdrawn, so one panel is left.''')

edit(
    '''    Claim(
        id="matched.v4.p_uniform",
        anchor=r"Wilcoxon $p$ values are {v}, {~} and {~}. Sets of",
        no_generator=_MATCHED("LGBM_V4", "uniform (all 210 random)",
                              "paired Wilcoxon p on |APE| difference vs GART 2.0 = 0.1086"),
        tol=("dp", 2),
    ),
    Claim(
        id="matched.v4.p_kwon",
        anchor=r"Wilcoxon $p$ values are {~}, {v} and {~}. Sets of",
        no_generator=_MATCHED("LGBM_V4", "Kwon_region domain (80)",
                              "paired Wilcoxon p on |APE| difference vs GART 2.0 = 0.4808"),
        tol=("dp", 2),
    ),
    Claim(
        id="matched.v4.p_chien",
        anchor=r"Wilcoxon $p$ values are {~}, {~} and {v}. Sets of",
        no_generator=_MATCHED("LGBM_V4", "Chien_region domain (50)",
                              "paired Wilcoxon p on |APE| difference vs GART 2.0 = 0.1117"),
        tol=("dp", 2),
    ),''',
    '''    Claim(
        id="matched.v4.p_uniform",
        anchor=r"the Wilcoxon $p$ value is {v}. A set of",
        no_generator=_MATCHED("LGBM_V4", "uniform (all 210 random)",
                              "paired Wilcoxon p on |APE| difference vs GART 2.0 = 0.0905"),
        tol=("dp", 2),
    ),
    # matched.v4.p_kwon and matched.v4.p_chien are deleted with their panels.''')


def main() -> int:
    text = MANIFEST.read_text(encoding="utf-8")
    failures: list[str] = []
    for i, (old, new) in enumerate(EDITS):
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
    print(f"applied {len(EDITS)} edits to {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
