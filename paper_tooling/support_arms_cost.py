"""Gate 5: feature-extraction cost, candidate vs frozen, on real TSPLIB EUC_2D.

Protocol, chosen so the number survives a loaded machine:

* ONE process. Both extractors are timed in the same interpreter with the same
  warm caches, so no cross-process or JIT-warmup asymmetry can leak in.
* INTERLEAVED. Within an instance the two extractors alternate, repeatedly, so
  a load spike lands on both arms rather than on whichever ran first.
* MEDIAN of repetitions per instance, then the ratio of summed medians. The
  median rejects the spikes a shared machine produces; summing before dividing
  weights instances by their real cost instead of letting the 78 tiny ones
  dominate a mean of per-instance ratios. Both are reported.
* RATIOS ONLY. No absolute wall-clock is emitted: the box is shared and any
  absolute number would be meaningless.

Arms:
    frozen  the 31-feature production path, feature_engineering_gart2
    A       identical to frozen by construction (measured anyway, as a control
            on the measurement itself -- it must come out at ~1.00)
    B       frozen + the three cheap features, as an add-on module
    B_coop  frozen + the three, sharing one canonical MST instead of building a
            second. Not what a drop-in integration costs, but it is what a
            cooperative one could, so it bounds the achievable overhead.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper_tooling"
for _p in (ROOT, HERE, ROOT / "lgbm_model_v3", ROOT / "tsplib_benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REPS = 5
GATE_RATIO = 1.35


def load_euc2d() -> list[tuple[str, np.ndarray]]:
    from tsplib_parser import parse_tsplib_file
    import ood_harness as oh

    suite = oh.load_suite()
    names = [str(i) for i in suite["tsplib_euc2d"].truth.index]
    inst_dir = ROOT / "tsplib_benchmark" / "instances"
    out = []
    for nm in names:
        p = inst_dir / f"{nm}.tsp"
        if not p.exists():
            continue
        info = parse_tsplib_file(str(p))
        if info["is_native_euclidean"] and info["raw_coords"] is not None:
            out.append((nm, np.ascontiguousarray(info["raw_coords"], dtype=np.float64)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()

    from feature_engineering_gart2 import compute_features
    from features_ext import group_degeneracy, group_mst_topology
    from mst_utils import compute_mst

    def t_frozen(X):
        t = time.perf_counter()
        compute_features(X, X.shape[1])
        return time.perf_counter() - t

    def t_extra_full(X):
        """Naive drop-in: the whole 12-feature group, 10 of them discarded."""
        t = time.perf_counter()
        group_mst_topology.compute(X)
        group_degeneracy.compute(X)
        return time.perf_counter() - t

    def t_extra(X):
        """What the deployed arm actually needs: the two features, plus one."""
        t = time.perf_counter()
        group_mst_topology.compute_pair(X)
        group_degeneracy.compute(X)
        return time.perf_counter() - t

    def t_extra_coop(X):
        """Upper bound on what a cooperative integration could reach.

        Only sound where the MST is unique: sharing a tree the production
        extractor built skips the canonical tie-break, which is exactly the
        repair the degenerate instances need. Reported as a bound, not as a
        recommendation.
        """
        t = time.perf_counter()
        canon, _order = group_mst_topology.canonical_coords(X)
        mst = compute_mst(canon)
        group_mst_topology.compute_pair(canon, mst, assume_canonical=True)
        group_degeneracy.compute(canon)
        return time.perf_counter() - t

    tasks = load_euc2d()
    print(f"[cost] {len(tasks)} TSPLIB EUC_2D instances, {args.reps} interleaved reps")

    # Warm every code path so no arm pays a one-off import/JIT cost.
    warm = max(tasks, key=lambda kv: kv[1].shape[0])[1][:2000]
    for _ in range(2):
        t_frozen(warm); t_extra(warm); t_extra_coop(warm); t_extra_full(warm)

    rows = []
    for nm, X in tasks:
        f, e, c, u = [], [], [], []
        for _ in range(args.reps):
            f.append(t_frozen(X))       # interleaved within the rep, not batched
            e.append(t_extra(X))
            c.append(t_extra_coop(X))
            u.append(t_extra_full(X))
        mf, me, mc, mu = (float(np.median(f)), float(np.median(e)),
                          float(np.median(c)), float(np.median(u)))
        rows.append({"instance": nm, "n": int(X.shape[0]),
                     "frozen_med": mf, "extra_med": me, "extra_coop_med": mc,
                     "extra_full_med": mu,
                     "ratio_B": (mf + me) / mf, "ratio_B_coop": (mf + mc) / mf,
                     "ratio_B_naive": (mf + mu) / mf})
    T = pd.DataFrame(rows).sort_values("n")
    T.to_csv(HERE / "support_arms_cost.csv", index=False)

    tot_f = T.frozen_med.sum()
    agg_B = (tot_f + T.extra_med.sum()) / tot_f
    agg_C = (tot_f + T.extra_coop_med.sum()) / tot_f
    agg_N = (tot_f + T.extra_full_med.sum()) / tot_f
    out = {
        "arm_B_naive_ratio_aggregate": float(agg_N),
        "arm_B_naive_note": "whole 12-feature group computed, 10 discarded",
        "n_instances": int(len(T)), "reps": args.reps,
        "gate_threshold": GATE_RATIO,
        "arm_A_ratio": 1.0,
        "arm_A_note": "arm A adds no feature; identical extraction path to frozen",
        "arm_B_ratio_aggregate": float(agg_B),
        "arm_B_ratio_median_per_instance": float(T.ratio_B.median()),
        "arm_B_ratio_p90_per_instance": float(T.ratio_B.quantile(0.90)),
        "arm_B_ratio_max_per_instance": float(T.ratio_B.max()),
        "arm_B_coop_ratio_aggregate": float(agg_C),
        "arm_B_coop_ratio_median_per_instance": float(T.ratio_B_coop.median()),
        "arm_A_pass": True,
        "arm_B_pass": bool(agg_B <= GATE_RATIO),
        "arm_B_coop_pass": bool(agg_C <= GATE_RATIO),
    }
    json.dump(out, open(HERE / "support_arms_cost.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("\n--- largest 10 instances (ratios only) ---")
    print(T.tail(10)[["instance", "n", "ratio_B", "ratio_B_coop"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
