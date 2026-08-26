"""Deciding-reviewer minimal surgery: remove only the augmentation rows whose
generator is provably the benchmark's own `generate_grid`, and see whether the
gates survive. Three seeds so the answer is not a single nuisance draw.

d=2 `gen_lattice` is `generate_grid` with the same site lattice (identical at
perfect-square n) and, at jitter=0.05, the identical jitter law.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import armA_verify_common as C  # noqa: E402

KEEP = ["bench2d_mape", "b2_grid_mspe", "b2_line_noise_mape", "linenoise_slope",
        "nd_test_mape", "nd_test_sdpe", "tsplib_euc2d_mape", "tsplib_euc2d_sdpe",
        "tsplib_noneuc_mape", "trees"]


def main() -> None:
    D = C.load_cache()
    corpus, aug, cc, feats = D["corpus"], D["aug"], D["cache"], D["feats31"]
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    nm = aug.instance_name.astype(str)

    d2lat = nm.str.startswith("AUG_lattice-d2-")
    j05 = d2lat & nm.str.contains("jitter0p05")
    print("d2 lattice rows", int(d2lat.sum()), "| of which jitter0.05", int(j05.sum()))

    variants = {
        "A_full": pd.Series(True, index=aug.index),
        "A_minus_d2lattice_j05": ~j05,
        "A_minus_d2lattice_all": ~d2lat,
    }
    rows = []
    for seed in (42, 1729, 97):
        for name, mask in variants.items():
            sub = aug[mask.to_numpy()]
            train = pd.concat([tr, sub], ignore_index=True)
            m = C.fit(train, va, feats, seed=seed)
            mm = C.metrics(m, feats, cc)
            r = {"arm": name, "seed": seed, "n_aug": int(len(sub))}
            r.update({k: mm[k] for k in KEEP})
            rows.append(r)
            print(f"{seed} {name:24s} n={len(sub):4d} grid={mm['b2_grid_mspe']:+.4f} "
                  f"slope={mm['linenoise_slope']:.4f} 2d={mm['bench2d_mape']:.4f} "
                  f"nd={mm['nd_test_mape']:.4f} tsplib_mape={mm['tsplib_euc2d_mape']:.4f} "
                  f"tsplib_sdpe={mm['tsplib_euc2d_sdpe']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(HERE / "armA_verify_decide_surgery.csv", index=False)
    print("written")


if __name__ == "__main__":
    main()
