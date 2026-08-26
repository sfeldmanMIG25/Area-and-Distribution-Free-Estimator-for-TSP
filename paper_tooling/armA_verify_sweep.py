"""Robustness sweep for arm A: dose-response, seed sensitivity, composition.

Blocks
------
dose  : nested random subsets of the 874 augmentation rows at
        {0, 2.5, 5, 10, 25, 50, 75, 100}% of dose, 8 independent permutations.
        Nesting removes the composition confound within a permutation, the 8
        permutations give the subset-draw noise band.
seed  : PAIRED refits -- for every LightGBM seed, both the no-augmentation and
        the full-augmentation model are refit. The published sweep varied only
        the augmented arm and compared it against a seed-42 frozen model, which
        cannot detect an ordering inversion caused by seed noise alone.
lofo  : leave-one-family-out and only-one-family, plus SIZE-MATCHED random
        removals. A bare LOFO confounds "this family matters" with "this many
        rows matter"; the matched control separates them.

Resumable: each run is appended to armA_verify_runs.csv, existing keys skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import armA_verify_common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "armA_verify_runs.csv"

DOSES = [0.0, 0.0057, 0.0114, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_PERM = 8
SEEDS = [42, 11, 97, 314, 1729, 5150]


# One canonical schema for every block. Appending frames with different column
# sets silently misaligns the file, so the schema is fixed up front.
SCHEMA = (["key", "block", "seed", "n_aug", "n_train", "fit_s",
           "dose", "perm", "fam_mix", "arm", "family", "mode", "fam_n"] + K.KEYS)


def done_keys() -> set[str]:
    if not OUT.exists():
        return set()
    return set(pd.read_csv(OUT, usecols=["key"])["key"].astype(str))


def append(row: dict) -> None:
    df = pd.DataFrame([{c: row.get(c) for c in SCHEMA}], columns=SCHEMA)
    df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)


def run_one(key: str, block: str, tr, va, aug_sub, feats, C, seed, extra: dict):
    t0 = time.perf_counter()
    if len(aug_sub):
        cols = [c for c in tr.columns if c in aug_sub.columns]
        train = pd.concat([tr, aug_sub[cols]], ignore_index=True, sort=False)
    else:
        train = tr
    m = K.fit(train, va, feats, seed=seed)
    mm = K.metrics(m, feats, C)
    row = {"key": key, "block": block, "seed": seed,
           "n_aug": int(len(aug_sub)), "n_train": int(len(train)),
           "fit_s": round(time.perf_counter() - t0, 1)}
    row.update(extra)
    row.update({k: mm[k] for k in K.KEYS})
    append(row)
    print(f"  {key:44s} nd={mm['nd_test_mape']:.4f} 2d={mm['bench2d_mape']:.4f} "
          f"gridMSPE={mm['b2_grid_mspe']:+.4f} lnMAPE={mm['b2_line_noise_mape']:.4f} "
          f"slope={mm['linenoise_slope']:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks", nargs="*", default=["dose", "seed", "lofo"])
    args = ap.parse_args()

    D = K.load_cache()
    corpus, aug, C, feats = D["corpus"], D["aug"], D["cache"], D["feats31"]
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    have = done_keys()
    N = len(aug)
    print(f"[data] train={len(tr)} val={len(va)} aug={N}; {len(have)} runs cached")

    # ---------------- dose ------------------------------------------------
    if "dose" in args.blocks:
        print("\n=== DOSE ===", flush=True)
        for perm in range(N_PERM):
            rng = np.random.default_rng(1000 + perm)
            order = rng.permutation(N)
            for d in DOSES:
                k = int(round(d * N))
                # dose 0 is order-invariant (no rows added); dose 1 is NOT --
                # it is the same 874 rows in a different order, which is a free
                # measurement of pure row-order nuisance noise at fixed data.
                if d == 0.0 and perm > 0:
                    continue
                key = f"dose_{d:g}_p{perm}"
                if key in have:
                    continue
                sub = aug.iloc[order[:k]]
                comp = sub["family"].value_counts().to_dict() if k else {}
                run_one(key, "dose", tr, va, sub, feats, C, 42,
                        {"dose": d, "perm": perm,
                         "fam_mix": ";".join(f"{a}={b}" for a, b in sorted(comp.items()))})

    # ---------------- seed ------------------------------------------------
    if "seed" in args.blocks:
        print("\n=== SEED (paired) ===", flush=True)
        for s in SEEDS:
            for tag, sub in (("noaug", aug.iloc[:0]), ("aug", aug)):
                key = f"seed_{tag}_{s}"
                if key in have:
                    continue
                run_one(key, "seed", tr, va, sub, feats, C, s,
                        {"arm": tag, "dose": 0.0 if tag == "noaug" else 1.0})

    # ---------------- composition ----------------------------------------
    if "lofo" in args.blocks:
        print("\n=== COMPOSITION ===", flush=True)
        fams = sorted(aug["family"].unique())
        for f in fams:
            m = aug["family"] == f
            key = f"lofo_drop_{f}"
            if key not in have:
                run_one(key, "lofo", tr, va, aug[~m], feats, C, 42,
                        {"family": f, "mode": "drop", "fam_n": int(m.sum())})
            key = f"lofo_only_{f}"
            if key not in have:
                run_one(key, "lofo", tr, va, aug[m], feats, C, 42,
                        {"family": f, "mode": "only", "fam_n": int(m.sum())})
            # size-matched random removal control
            for r in range(3):
                key = f"lofo_rand{r}_{f}"
                if key in have:
                    continue
                # zlib.crc32, not hash(): str hashing is salted per process.
                rng = np.random.default_rng(7000 + r * 97 + zlib.crc32(f.encode()) % 1000)
                drop = rng.choice(len(aug), size=int(m.sum()), replace=False)
                keep = np.setdiff1d(np.arange(len(aug)), drop)
                run_one(key, "lofo", tr, va, aug.iloc[keep], feats, C, 42,
                        {"family": f, "mode": f"rand_match{r}", "fam_n": int(m.sum())})

    print("\nwrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
