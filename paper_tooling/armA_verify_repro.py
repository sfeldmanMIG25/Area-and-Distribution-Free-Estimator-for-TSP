"""Step 0: does an INDEPENDENT re-implementation reproduce R0 and arm A exactly?

Everything downstream (dose-response, seed sweep, leave-family-out) is
meaningless unless this harness lands on the same model the study shipped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

import armA_verify_common as K

HERE = Path(__file__).resolve().parent


def main() -> None:
    t0 = time.perf_counter()
    D = K.load_cache()
    corpus, aug, C, feats = D["corpus"], D["aug"], D["cache"], D["feats31"]
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    te = corpus[corpus.split == "test"]
    print(f"[data] corpus={len(corpus)} train={len(tr)} val={len(va)} test={len(te)} "
          f"aug={len(aug)} feats={len(feats)}")
    print(f"[data] cache load {time.perf_counter()-t0:.0f}s")

    import pandas as pd
    tr_aug = pd.concat([tr, aug[[c for c in tr.columns if c in aug.columns]]],
                       ignore_index=True, sort=False)
    assert len(tr_aug) == len(tr) + len(aug)

    rep = {}
    frozen = joblib.load(K.FROZEN)
    Xte = te[feats]

    # ---- R0: no augmentation -------------------------------------------
    p = HERE / "armA_verify_models" / "R0.joblib"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        r0 = joblib.load(p)
    else:
        r0 = K.fit(tr, va, feats, seed=42)
        joblib.dump(r0, p)
    d = float(np.max(np.abs(frozen.predict(Xte, num_iteration=frozen.best_iteration)
                            - r0.predict(Xte, num_iteration=r0.best_iteration))))
    rep["R0_vs_frozen_max_abs_pred_diff"] = d
    rep["R0_trees"] = int(r0.num_trees())
    rep["frozen_trees"] = int(frozen.num_trees())
    print(f"[R0] trees {r0.num_trees()} vs frozen {frozen.num_trees()}, "
          f"max|dpred| = {d:.3e}")

    # ---- A: full dose ---------------------------------------------------
    p = HERE / "armA_verify_models" / "A_full.joblib"
    if p.exists():
        a = joblib.load(p)
    else:
        a = K.fit(tr_aug, va, feats, seed=42)
        joblib.dump(a, p)
    shipped = joblib.load(HERE / "support_arms_models" / "A.joblib")
    d = float(np.max(np.abs(shipped.predict(Xte, num_iteration=shipped.best_iteration)
                            - a.predict(Xte, num_iteration=a.best_iteration))))
    rep["A_vs_shippedA_max_abs_pred_diff"] = d
    rep["A_trees"] = int(a.num_trees())
    rep["shippedA_trees"] = int(shipped.num_trees())
    print(f"[A] trees {a.num_trees()} vs shipped {shipped.num_trees()}, "
          f"max|dpred| = {d:.3e}")

    # ---- reported numbers, recomputed -----------------------------------
    for tag, m in (("FROZEN", frozen), ("R0", r0), ("A", a)):
        mm = K.metrics(m, feats, C)
        rep[tag] = {k: mm[k] for k in K.KEYS}
        print(f"[{tag}] " + "  ".join(f"{k}={mm[k]:.4f}" for k in
              ("nd_test_mape", "bench2d_mape", "b2_grid_mspe",
               "b2_line_noise_mape", "linenoise_slope", "tsplib_euc2d_mape",
               "tsplib_euc2d_sdpe", "augment_mape")))

    json.dump(rep, open(HERE / "armA_verify_repro.json", "w"), indent=2)
    print("wrote armA_verify_repro.json")


if __name__ == "__main__":
    main()
