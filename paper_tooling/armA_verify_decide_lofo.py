"""Deciding-reviewer LOFO: settle whether the grid gain rests on the
identical-generator `lattice` family or on the different-generator
`hexlattice` family. Also re-checks the line-noise slope attribution.

Writes only paper_tooling/armA_verify_decide_lofo.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import armA_verify_common as C  # noqa: E402

KEEP = ["bench2d_mape", "b2_grid_mspe", "b2_grid_mape", "b2_line_noise_mape",
        "linenoise_slope", "nd_test_mape", "tsplib_euc2d_mape",
        "tsplib_euc2d_sdpe", "tsplib_noneuc_mape", "trees"]

ARMS = {
    "frozen_noaug": [],
    "A_full": None,
    "A_minus_lattice": ["lattice"],
    "A_minus_hexlattice": ["hexlattice"],
    "A_minus_both_lattices": ["lattice", "hexlattice"],
    "A_only_lattice": ("only", ["lattice"]),
    "A_only_hexlattice": ("only", ["hexlattice"]),
    "A_minus_collinear": ["collinear"],
    "A_minus_subspace": ["subspace"],
    "A_minus_1d": ["collinear", "subspace", "curve", "filament"],
}


def main() -> None:
    cache = C.load_cache()
    corpus, aug, cc, feats = (cache["corpus"], cache["aug"],
                              cache["cache"], cache["feats31"])
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    print("train", len(tr), "val", len(va), "aug", len(aug),
          "families", sorted(aug.family.unique()))

    rows = []
    for seed in (42, 1729):
        for name, spec in ARMS.items():
            if spec == []:
                sub = aug.iloc[0:0]
            elif spec is None:
                sub = aug
            elif isinstance(spec, tuple):
                sub = aug[aug.family.isin(spec[1])]
            else:
                sub = aug[~aug.family.isin(spec)]
            train = pd.concat([tr, sub], ignore_index=True) if len(sub) else tr
            m = C.fit(train, va, feats, seed=seed)
            mm = C.metrics(m, feats, cc)
            r = {"arm": name, "seed": seed, "n_aug": int(len(sub))}
            r.update({k: mm[k] for k in KEEP})
            rows.append(r)
            print(f"{seed} {name:24s} n_aug={len(sub):4d} "
                  f"grid_mspe={mm['b2_grid_mspe']:+.4f} "
                  f"slope={mm['linenoise_slope']:.4f} "
                  f"2d={mm['bench2d_mape']:.4f} nd={mm['nd_test_mape']:.4f} "
                  f"tsplib_sdpe={mm['tsplib_euc2d_sdpe']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(HERE / "armA_verify_decide_lofo.csv", index=False)
    print("written")


if __name__ == "__main__":
    main()
