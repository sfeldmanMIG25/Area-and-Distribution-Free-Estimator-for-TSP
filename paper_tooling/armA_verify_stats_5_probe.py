"""Does the swept-feature monotonicity probe have any power?

100% / 100% with zero violations for every arm is exactly what a LightGBM model
trained with monotone_constraints=-1 MUST produce, because the constraint is
enforced during split-finding and a sum of monotone trees is monotone. This
script asks three things the reported table cannot distinguish:

  (a) is the constraint actually present in the saved boosters, and on the right
      feature indices?
  (b) does the sweep MOVE the prediction at all, or is 'non-increasing' being
      satisfied by a flat line?
  (c) would the probe flag a genuinely non-monotone predictor? Two controls: a
      synthetic bump, and a real LightGBM refit with the constraint removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import support_arms_eval as EV  # noqa: E402
import support_arms_study as S  # noqa: E402

pd.set_option("display.width", 240)
MODELS = {"FROZEN": ROOT / "lgbm_model_v3" / "gart2_final.joblib",
          "A": HERE / "support_arms_models" / "A.joblib"}


class Bumped:
    """Frozen model plus a deliberately non-monotone bump in one feature.

    Non-monotone by construction: the bump is a raised cosine in log(feature),
    so the prediction rises then falls as the swept feature increases. Amplitude
    is set in the model's own z units.
    """

    def __init__(self, base, col: str, amp: float, lo: float, hi: float):
        self.base, self.col, self.amp = base, col, amp
        self.lo, self.hi = np.log(lo), np.log(hi)
        self.best_iteration = base.best_iteration

    def predict(self, X, num_iteration=None):
        z = self.base.predict(X, num_iteration=num_iteration)
        t = (np.log(np.clip(X[self.col].to_numpy(dtype=float), 1e-9, None))
             - self.lo) / (self.hi - self.lo)
        return z + self.amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * np.clip(t, 0, 1)))


def sweep_range(model, feats, base, col, grid):
    """How far does the prediction actually travel across the swept grid?"""
    P = np.empty((len(base), len(grid)))
    for j, g in enumerate(grid):
        X = base.copy()
        X[col] = g
        P[:, j] = model.predict(X[feats], num_iteration=model.best_iteration)
    A = EV.to_alpha(P)
    span_z = P.max(axis=1) - P.min(axis=1)
    span_a = A.max(axis=1) - A.min(axis=1)
    return {
        "grid_lo": grid[0], "grid_hi": grid[-1], "grid_pts": len(grid),
        "median_span_z": float(np.median(span_z)),
        "median_span_alpha": float(np.median(span_a)),
        "p90_span_alpha": float(np.percentile(span_a, 90)),
        "frac_rows_flat": float((span_z < 1e-9).mean()),
        "max_span_alpha": float(span_a.max()),
    }


def main():
    feats31 = S.frozen_features()

    print("=== (a) is the monotone constraint actually in the saved boosters? ===")
    for tag, path in MODELS.items():
        b = joblib.load(path)
        js = b.dump_model()
        mono = js.get("monotone_constraints")
        names = list(b.feature_name())
        if mono is None:
            # fall back to the params string LightGBM stores on the booster
            print(f"  {tag}: dump_model has no 'monotone_constraints' key")
            mono = None
        else:
            nz = {names[i]: c for i, c in enumerate(mono) if c != 0}
            print(f"  {tag}: {len(names)} features, non-zero constraints = {nz}")
        pstr = b.params if hasattr(b, "params") else {}
        print(f"        params.monotone_constraints_method="
              f"{pstr.get('monotone_constraints_method')!r}  trees={b.num_trees()}")

    # base frame, exactly as the shipped probe builds it
    corpus = S.load_corpus(feats31)
    nd = corpus[corpus.split == "test"].reset_index(drop=True)
    base = nd.sample(min(EV.PROBE_N_INSTANCES, len(nd)), random_state=EV.SEED).copy()
    print(f"\nprobe base: {len(base)} rows drawn from the ND test split")
    print(f"  observed n_customers range in base: "
          f"[{base.n_customers.min():.0f}, {base.n_customers.max():.0f}]  "
          f"probe grid: [{EV.PROBE_N_GRID[0]}, {EV.PROBE_N_GRID[-1]}]")
    print(f"  observed dimension  range in base: "
          f"[{base.dimension.min():.0f}, {base.dimension.max():.0f}]  "
          f"probe grid: [{EV.PROBE_D_GRID[0]}, {EV.PROBE_D_GRID[-1]}]")

    print("\n=== (b) does the sweep move the prediction? ===")
    rows = []
    for tag, path in MODELS.items():
        m = joblib.load(path)
        for col, grid in (("dimension", EV.PROBE_D_GRID),
                          ("n_customers", EV.PROBE_N_GRID)):
            r = sweep_range(m, feats31, base, col, grid)
            r.update({"model": tag, "swept": col})
            rows.append(r)
    SR = pd.DataFrame(rows)[["model", "swept", "grid_lo", "grid_hi", "grid_pts",
                             "median_span_z", "median_span_alpha",
                             "p90_span_alpha", "max_span_alpha", "frac_rows_flat"]]
    print(SR.to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    print("\n=== (c1) control: frozen model + deliberate non-monotone bump ===")
    frozen = joblib.load(MODELS["FROZEN"])
    print(f"{'amp(z)':>8} {'swept':>12} {'pct_nonincr_raw':>16} {'n_viol_raw':>11} "
          f"{'pct_nonincr_dep':>16} {'n_viol_dep':>11} {'max_viol_dep':>13}  gate4")
    for amp in (1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5):
        for col, grid in (("dimension", EV.PROBE_D_GRID),
                          ("n_customers", EV.PROBE_N_GRID)):
            bm = Bumped(frozen, col, amp, grid[0], grid[-1])
            r = EV.monotonicity(bm, feats31, base, col, grid)
            g4 = (r["pct_nonincr_deployed"] >= 99.0
                  and r["viol_max_deployed"] <= 1e-3)
            print(f"{amp:8.0e} {col:>12} {r['pct_nonincr_raw']:16.2f} "
                  f"{r['n_viol_raw']:11d} {r['pct_nonincr_deployed']:16.2f} "
                  f"{r['n_viol_deployed']:11d} {r['viol_max_deployed']:13.3e}  "
                  f"{'PASS' if g4 else 'FAIL'}")

    print("\n=== (c2) control: real LightGBM refit with the constraint REMOVED ===")
    import lightgbm as lgb
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    params = dict(S.V3_FROZEN)
    params.update({"objective": "regression_l2", "metric": "l2",
                   "boosting_type": "gbdt", "seed": S.SEED, "num_threads": 6,
                   "verbosity": -1, "feature_pre_filter": False})
    dtr = lgb.Dataset(tr[feats31], label=S.to_z(tr["alpha"].to_numpy()))
    dvl = lgb.Dataset(va[feats31], label=S.to_z(va["alpha"].to_numpy()), reference=dtr)
    for label, mono in (("UNCONSTRAINED", [0] * len(feats31)),
                        ("CONSTRAINED(-1 on n,d)",
                         [-1 if f in S.MONO_COLS else 0 for f in feats31])):
        p = dict(params)
        p["monotone_constraints"] = mono
        p["monotone_constraints_method"] = "basic"
        bst = lgb.train(p, dtr, num_boost_round=600, valid_sets=[dvl],
                        callbacks=[lgb.log_evaluation(0)])
        bst.best_iteration = 600
        for col, grid in (("dimension", EV.PROBE_D_GRID),
                          ("n_customers", EV.PROBE_N_GRID)):
            r = EV.monotonicity(bst, feats31, base, col, grid)
            g4 = (r["pct_nonincr_deployed"] >= 99.0
                  and r["viol_max_deployed"] <= 1e-3)
            print(f"  {label:24s} swept={col:12s} "
                  f"pct_nonincr_dep {r['pct_nonincr_deployed']:6.2f}  "
                  f"n_viol {r['n_viol_deployed']:6d}  "
                  f"max_viol {r['viol_max_deployed']:.3e}  gate4 "
                  f"{'PASS' if g4 else 'FAIL'}")


if __name__ == "__main__":
    main()
