"""Register the one numeral the copy-edit re-keyed but did not author.

"a factor of 2.9 on MAPE over a calibrated constant" is pre-existing Conclusion
prose.  It re-keys because prose_baseline addresses an occurrence by a digest of
the surrounding text, and the sentence before it changed.  Registering it against
the same derivation the Introduction uses is strictly better than absorbing it
into the baseline: it moves the number from the unverified backlog into the
checked set.
"""
from __future__ import annotations

from pathlib import Path

MANIFEST = Path(r"D:\Area-and-Distribution-Free-Estimator-for-TSP\paper_tooling\prose_manifest.py")

OLD = b'''    Claim(
        id="conclusion.ctrans.cost_max_pp",'''

NEW = b'''    Claim(
        id="conclusion.nd.factor_over_rho_dn",
        anchor=r"It is a factor of {v} on MAPE over a calibrated constant",
        expect="= {nd_by_dim_total_calibrated_mst_ratio_hat_rho_d_n_mape_pct}"
               " / {nd_by_dim_total_gart_2_0_mape_pct}",
        tol=("dp", 1),
        note="Pre-existing Conclusion prose, unchanged by the 2026-08-11 "
             "copy-edit. It re-keyed in prose_baseline because the sentence "
             "before it changed, and is registered here against the same "
             "derivation intro.nd.factor_over_rho_dn uses rather than absorbed.",
    ),
    Claim(
        id="conclusion.ctrans.cost_max_pp",'''


def main() -> int:
    raw = MANIFEST.read_bytes()
    old = OLD.replace(b"\n", b"\r\n")
    new = NEW.replace(b"\n", b"\r\n")
    hits = raw.count(old)
    print(f"anchor matches: {hits}")
    if hits != 1:
        print("NOT WRITTEN: anchor must match exactly once.")
        return 1
    before = len(raw)
    raw = raw.replace(old, new, 1)
    MANIFEST.write_bytes(raw)
    print(f"applied  : + conclusion.nd.factor_over_rho_dn")
    print(f"missing  : 0")
    print(f"written  : {before} -> {len(raw)} bytes (+{len(raw) - before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
