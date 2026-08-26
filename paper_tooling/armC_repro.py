"""Step 0 for arm C: prove this harness reproduces the frozen incumbent.

A comparison against a mis-scored incumbent is worthless, so nothing downstream
runs until all four checks below pass:

1. the frozen artifact is the sha256 the protocol names;
2. re-scoring it here reproduces grid MSPE +7.1118 / LineNoise slope 0.3621 /
   2D MAPE 2.9037 / ND MAPE 0.6201 exactly;
3. a FRESH refit of the unaugmented recipe, in the original (non-protocol) row
   order, reproduces the frozen booster's predictions bit-for-bit;
4. the de-contamination rule removes exactly 24 rows and retains exactly 850.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

import armA_verify_common as K
import armC_common as CC

HERE = Path(__file__).resolve().parent
SHA_EXPECT = "69d277441e20f04ecf2f29a10a610532d1a338d84b77a495822ed5359be64197"
TOL = 0.0


def main() -> None:
    rep: dict = {}
    ok = True

    sha = hashlib.sha256(K.FROZEN.read_bytes()).hexdigest()
    rep["frozen_sha256"] = sha
    rep["frozen_sha256_matches"] = sha == SHA_EXPECT
    ok &= rep["frozen_sha256_matches"]
    print(f"[1] frozen sha256 {sha[:16]}... matches={rep['frozen_sha256_matches']}")

    D = K.load_cache()
    corpus, aug, C, feats = D["corpus"], D["aug"], D["cache"], D["feats31"]
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]

    frozen = joblib.load(K.FROZEN)
    mm = CC.extended_metrics(frozen, feats, C)
    dev = {}
    for k, want in CC.FROZEN_REF.items():
        got = float(mm[k])
        dev[k] = {"got": got, "want": want, "abs_diff": abs(got - want)}
    worst = max(v["abs_diff"] for v in dev.values())
    rep["frozen_rescore"] = dev
    rep["frozen_rescore_worst_abs_diff"] = worst
    rep["frozen_rescore_exact"] = worst <= TOL
    ok &= rep["frozen_rescore_exact"]
    print(f"[2] frozen rescore: grid MSPE {mm['b2_grid_mspe']:+.4f}  "
          f"slope {mm['linenoise_slope']:.4f}  2D {mm['bench2d_mape']:.4f}  "
          f"ND {mm['nd_test_mape']:.4f}  worst|diff|={worst:.3e}")

    t0 = time.perf_counter()
    r0 = K.fit(tr, va, feats, seed=42)
    fit_s = time.perf_counter() - t0
    Xte = corpus[corpus.split == "test"][feats]
    d = float(np.max(np.abs(
        frozen.predict(Xte, num_iteration=frozen.best_iteration)
        - r0.predict(Xte, num_iteration=r0.best_iteration))))
    rep["R0_refit_vs_frozen_max_abs_pred_diff"] = d
    rep["R0_refit_trees"] = int(r0.num_trees())
    rep["frozen_trees"] = int(frozen.num_trees())
    rep["R0_refit_exact"] = (d == 0.0) and (r0.num_trees() == frozen.num_trees())
    ok &= rep["R0_refit_exact"]
    print(f"[3] fresh R0 refit: trees {r0.num_trees()} vs {frozen.num_trees()}, "
          f"max|dpred|={d:.3e}  exact={rep['R0_refit_exact']}")

    keep = CC.armC_aug(aug)          # raises if the counts are wrong
    rep["aug_total"] = int(len(aug))
    rep["aug_removed"] = int(len(aug) - len(keep))
    rep["aug_retained"] = int(len(keep))
    rep["removed_names"] = sorted(aug.loc[CC.d2_lattice_mask(aug).to_numpy(),
                                          "instance_name"].astype(str).tolist())
    rep["retained_by_family"] = {k: int(v) for k, v
                                 in keep.family.value_counts().sort_index().items()}
    print(f"[4] augmentation {len(aug)} -> removed {rep['aug_removed']}, "
          f"retained {rep['aug_retained']}  by family {rep['retained_by_family']}")

    # feature identity: gate 5 is structural, not timed
    rep["armC_features_equal_frozen"] = list(frozen.feature_name()) == list(feats)
    rep["n_features"] = len(feats)
    rep["single_fit_seconds"] = round(fit_s, 1)

    rep["ALL_CHECKS_PASS"] = bool(ok)
    json.dump(rep, open(HERE / "armC_repro.json", "w"), indent=2)
    print(f"\nALL_CHECKS_PASS = {ok}   (one fit took {fit_s:.0f}s)")
    if not ok:
        raise SystemExit("reproduction failed -- STOP, do not compare anything")


if __name__ == "__main__":
    main()
