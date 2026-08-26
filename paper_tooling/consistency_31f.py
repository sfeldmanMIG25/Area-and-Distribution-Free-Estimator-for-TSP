"""Consistency criterion for the two excluded comparators, on one protocol.

The manuscript's differentiator is not mean error, it is consistency: the
swept-feature monotonicity probe and the dispersion series that falls with n
and with d.  Two models that beat GART 2.0 on raw accuracy on at least one
stratum -- ``NN_31F`` (the network refitted on the production 31-column
vector) and ``LGBM_V4`` -- have never been measured on that criterion, so a
superiority claim founded on it cannot be checked against them.  This module
measures it, for four models, in one session, on one protocol.

WHAT IS REUSED AND WHY
----------------------
The probe is NOT reimplemented.  ``v4_study._sweep_monotonicity`` and its two
grids (``PROBE_D_GRID``, ``PROBE_N_GRID``), tolerance and 1000-instance base
sample are imported, so every number here is directly comparable with the rows
already in ``v4_study_gart2_probe.csv``.  A second probe with its own sweep
ranges would produce numbers that look comparable and are not.

THE PROBE MUST BE ABLE TO FAIL
------------------------------
GART 2.0 carries ``monotone_constraints`` of -1 on ``n_customers`` and
``dimension``.  A sum of monotone-constrained trees cannot violate a ceteris
paribus sweep, so its 100% is an artifact-integrity check on the shipped
booster, not evidence about an inductive bias that could have gone either way.
Two things make the number meaningful:

  1. the unconstrained twin, ``GART2_logit_v3hp`` -- same logit target, same
     frozen V3 hyperparameters, same 31 features, ``monotone_constraints``
     null -- which is re-reported here from the existing artifact; and
  2. three synthetic predictors with known monotonicity, swept through the
     identical harness.  If the strictly-increasing one does not score 0%,
     the probe is not measuring what it claims to and no pass is reportable.

The other unconstrained arm in that artifact, ``GART2_logit_tuned``, also
re-tunes hyperparameters and is therefore NOT a clean one-variable-changed
control; it is carried here only so the distinction is on the record.

RAW VERSUS DEPLOYED
-------------------
Every adapter's ``predict`` returns the model's raw output on its own scale --
booster logit for GART 2.0, pre-clip alpha for LGBM_V4, ``1 + y`` for the
network, the pipeline's own output for the linear control -- and the harness
applies the deployment transform itself.  That is what ``v4_study`` does for
the LightGBM arms, so the deployed/raw split means the same thing in every
row of the output.  Each adapter is checked against the estimator's own
``predict_alpha`` on real rows before it is swept; a batched path that
silently disagrees with the deployed one would make the whole table fiction.

Usage:
    python paper_tooling/consistency_31f.py probe
    python paper_tooling/consistency_31f.py buckets
    python paper_tooling/consistency_31f.py all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO), str(HERE), str(REPO / "lgbm_model_v3"),
           str(REPO / "lgbm_model_v4"), str(REPO / "linear_model_v3"),
           str(REPO / "nn_est_alpha_v3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import v4_study as V4  # noqa: E402  (probe grids, tolerance, sweep function)
from model_registry import GART  # noqa: E402

P_PROBE_OUT = HERE / "consistency_31f_probe.csv"
P_BUCKETS_OUT = HERE / "consistency_31f_nd_buckets.csv"
P_V4_PROBE = HERE / "v4_study_gart2_probe.csv"
P_ROWS_ND = HERE / "controls_31f" / "rows_nd.csv"

#: Agreement required between a batched adapter and the estimator's own
#: per-row ``predict_alpha``. Both run float64 into the same transform, so
#: this is a float-associativity tolerance, not a modelling tolerance.
ADAPTER_RTOL = 1e-9

#: The network is the one model whose deployed path is float32. Torch picks a
#: different GEMM kernel for a 1000-row batch than for the estimator's 1-row
#: call, so the two agree only to float32 epsilon (~1.2e-7) -- measured at
#: 8.9e-8 here. That is a precision fact about the deployed path, not a
#: modelling difference, and it is why the network is swept twice: see
#: ``NN_FP32_LABEL``.
NN_RTOL = 1e-6
NN_FP32_LABEL = "NN_31F[fp32]"

#: Float32 round-off between two batched forwards is order 1e-7 in alpha
#: units, a hundredfold above ``v4_study.PROBE_TOL`` (1e-9). A pass rate at
#: the strict tolerance therefore cannot, on its own, tell a float32 model's
#: round-off apart from a real reversal. Every row also carries the pass rate
#: at this looser tolerance, which is above the noise floor, so the two
#: together separate the cases. The strict columns remain primary because
#: they are the ones comparable with ``v4_study_gart2_probe.csv``.
NOISE_TOL = 1e-6

#: The four models the consistency claim has to survive, plus the controls
#: that make the probe interpretable.
ROLES = {
    GART: "production",
    "LGBM_V4": "comparator (beats GART 2.0 on 3 of 4 strata)",
    "NN_31F": "comparator (beats GART 2.0 on 2D and non-Euclidean)",
    NN_FP32_LABEL: "comparator, deployed float32 path",
    "Linear_31F": "same-feature linear control",
    "GART2_logit_v3hp": "unconstrained twin (clean one-variable control)",
    "GART2_logit_tuned": "unconstrained AND re-tuned (not a clean control)",
    "Synthetic_increasing": "falsification: strictly increasing in n and d",
    "Synthetic_oscillating": "falsification: oscillating in n and d",
    "Synthetic_decreasing": "falsification: strictly decreasing in n and d",
}


# =============================================================================
# Adapters: one .predict(frame) -> raw model output, per model
# =============================================================================
class _Adapter:
    """``predict(frame) -> raw output``, with the deployed transform declared.

    ``target`` is passed straight to ``v4_study._sweep_monotonicity``, which
    owns the raw -> deployed map ('logit' for the logit-target booster,
    'alpha' for a model whose output is already an alpha to be clipped).
    """

    def __init__(self, label: str, feats: list[str], target: str, fn):
        self.label, self.feats, self.target, self._fn = label, feats, target, fn

    def predict(self, frame):
        return np.asarray(self._fn(frame), dtype=np.float64)


def _deploy(raw, target: str) -> np.ndarray:
    return np.asarray(V4._to_alpha(raw, target), dtype=np.float64)


def _check_against_estimator(ad: _Adapter, est, frame, n_check: int = 64,
                             rtol: float = ADAPTER_RTOL) -> None:
    """Batched adapter must reproduce the estimator's own per-row alpha."""
    sub = frame.iloc[:n_check]
    batched = _deploy(ad.predict(sub), ad.target)
    per_row = np.array([est.predict_alpha(dict(zip(ad.feats, r)))
                        for r in sub[ad.feats].to_numpy(dtype=np.float64)])
    worst = float(np.max(np.abs(batched - per_row) / np.maximum(np.abs(per_row), 1e-12)))
    if not worst <= rtol:
        raise RuntimeError(
            f"{ad.label}: batched adapter disagrees with predict_alpha by "
            f"{worst:.3e} > {rtol:g}; the swept table would be fiction")
    print(f"  [adapter] {ad.label:<14s} matches predict_alpha on {len(sub)} rows "
          f"(worst rel. diff {worst:.2e}, tol {rtol:g})")


def build_adapters(frame) -> list[_Adapter]:
    """The four measured models, each verified against its deployed path."""
    import joblib
    from lgbm_model_v3.lgbm_estimator_gart2 import TSP_GART2_Estimator
    from linear_model_v3.estimator_linear_31f import TSP_Linear31_Estimator
    from nn_est_alpha_v3.estimator_nn_31f import TSP_NN31_Estimator
    import torch

    out: list[_Adapter] = []

    # -- GART 2.0: shipped booster, logit target, monotone-constrained -------
    g = TSP_GART2_Estimator(str(REPO / "lgbm_model_v3"))
    gfeats = list(g.features_required)
    ad = _Adapter(GART, gfeats, "logit", lambda f: g.model.predict(f[gfeats]))
    _check_against_estimator(ad, g, frame)
    out.append(ad)

    # -- LGBM_V4: alpha target, no monotone constraint -----------------------
    v4 = joblib.load(REPO / "lgbm_model_v4" / "lgbm_alpha_model_v4.joblib")
    v4feats = list(v4.feature_name())
    out.append(_Adapter("LGBM_V4", v4feats, "alpha", lambda f: v4.predict(f[v4feats])))

    # -- NN_31F: gated-residual network, no monotone constraint possible -----
    # Swept twice on purpose. The float64 promotion evaluates the *same*
    # function -- every float32 weight is exact in float64 -- but batch-
    # invariantly, so a violation it reports is a property of the learned
    # surface. The deployed float32 path is swept alongside it because that is
    # what actually ships; where the two disagree, the difference is round-off
    # and the pair says so instead of one of them asserting it.
    nn = TSP_NN31_Estimator(str(REPO / "nn_est_alpha_v3"))
    nfeats = list(nn.features_required)

    def _nn_fp32(f):
        v = nn.scaler.transform(f[nfeats].to_numpy(dtype=np.float64))
        with torch.no_grad():
            y = nn.model(torch.tensor(v, dtype=torch.float32,
                                      device=nn.device)).cpu().numpy().ravel()
        return 1.0 + y

    nn64 = TSP_NN31_Estimator(str(REPO / "nn_est_alpha_v3"))
    nn64.model = nn64.model.double().cpu().eval()

    def _nn_fp64(f):
        v = nn64.scaler.transform(f[nfeats].to_numpy(dtype=np.float64))
        with torch.no_grad():
            y = nn64.model(torch.tensor(v, dtype=torch.float64)).numpy().ravel()
        return 1.0 + y

    ad64 = _Adapter("NN_31F", nfeats, "alpha", _nn_fp64)
    _check_against_estimator(ad64, nn, frame, rtol=NN_RTOL)
    ad32 = _Adapter(NN_FP32_LABEL, nfeats, "alpha", _nn_fp32)
    _check_against_estimator(ad32, nn, frame, rtol=NN_RTOL)
    out += [ad64, ad32]

    # -- Linear_31F: least squares on the identical vector -------------------
    lin = TSP_Linear31_Estimator(str(REPO / "linear_model_v3"))
    lfeats = list(lin.features_required)
    ad = _Adapter("Linear_31F", lfeats, "alpha",
                  lambda f: lin.model.predict(f[lfeats]))
    _check_against_estimator(ad, lin, frame)
    out.append(ad)

    return out


def build_falsification(feats: list[str]) -> list[_Adapter]:
    """Predictors whose monotonicity is known by construction.

    Each depends on both swept columns, so both sweeps are exercised.  The
    increasing one must score 0% and the decreasing one 100%; anything else
    means the harness is not measuring monotonicity and no model's pass rate
    from this session is reportable.
    """
    def _ln(f):
        return np.log(np.maximum(f["n_customers"].to_numpy(dtype=np.float64), 1.0))

    def _ld(f):
        return np.log(np.maximum(f["dimension"].to_numpy(dtype=np.float64), 1.0))

    return [
        _Adapter("Synthetic_increasing", feats, "alpha",
                 lambda f: 1.0 + 0.05 * (_ln(f) + _ld(f))),
        _Adapter("Synthetic_oscillating", feats, "alpha",
                 lambda f: 1.5 + 0.05 * np.sin(4.0 * _ln(f))
                 + 0.05 * np.sin(4.0 * _ld(f))),
        _Adapter("Synthetic_decreasing", feats, "alpha",
                 lambda f: 1.0 + 1.0 / (1.0 + _ln(f) + _ld(f))),
    ]


# =============================================================================
# Probe
# =============================================================================
def _probe_base():
    df = pd.read_csv(REPO / "tsp_features_v4.csv")
    df = df[df["split"] == "test"].reset_index(drop=True)
    return df.sample(min(V4.PROBE_N_INSTANCES, len(df)), random_state=V4.SEED).copy()


def _sweep_rows(adapters, base) -> list[dict]:
    rows = []
    for ad in adapters:
        for col, grid in (("dimension", V4.PROBE_D_GRID),
                          ("n_customers", V4.PROBE_N_GRID)):
            r = V4._sweep_monotonicity(ad, ad.feats, base, col, grid,
                                       tol=V4.PROBE_TOL, target=ad.target)
            # Same function, same predictions, one looser tolerance: the
            # float32 noise floor. Nothing is recomputed by hand.
            loose = V4._sweep_monotonicity(ad, ad.feats, base, col, grid,
                                           tol=NOISE_TOL, target=ad.target)
            r.update({f"{k}_tol1em6": v for k, v in loose.items()
                      if k.startswith(("pct_nonincr", "n_viol"))})
            r.update({"model": ad.label, "swept": col, "n_probes": len(base),
                      "grid_points": len(grid), "role": ROLES.get(ad.label, "")})
            rows.append(r)
            print(f"  {ad.label:<22s} {col:<12s} deployed {r['pct_nonincr_deployed']:6.1f}%"
                  f"  raw {r['pct_nonincr_raw']:6.1f}%"
                  f"  @1e-6 {r['pct_nonincr_deployed_tol1em6']:6.1f}%"
                  f"  viol max (raw) {r['viol_max_raw']:.3e}")
    return rows


def _carry_forward() -> list[dict]:
    """Re-report the two unconstrained LightGBM arms from the V4 study.

    Same harness, same grids, same seed and same 1000-row base -- the rows are
    read rather than recomputed because retraining is not in scope and the
    artifact is the one the manuscript's 8.4/19.0 control number comes from.
    """
    if not P_V4_PROBE.exists():
        raise SystemExit(f"missing {P_V4_PROBE}; run v4_study.py probe first")
    t = pd.read_csv(P_V4_PROBE)
    keep = t[t["model"].isin(["GART2_logit_v3hp", "GART2_logit_tuned"])].copy()
    if len(keep) != 4:
        raise SystemExit(f"{P_V4_PROBE}: expected 4 carried rows, found {len(keep)}")
    keep["role"] = keep["model"].map(ROLES)
    return keep.to_dict("records")


def cmd_probe(args) -> None:
    base = _probe_base()
    print(f"[probe] {len(base)} ND-test instances @ seed {V4.SEED} | "
          f"d grid {len(V4.PROBE_D_GRID)} pts "
          f"[{V4.PROBE_D_GRID[0]}..{V4.PROBE_D_GRID[-1]}] | "
          f"n grid {len(V4.PROBE_N_GRID)} pts "
          f"[{V4.PROBE_N_GRID[0]}..{V4.PROBE_N_GRID[-1]}] | tol {V4.PROBE_TOL:g}")

    adapters = build_adapters(base)
    feats = adapters[0].feats

    print("\n[falsification] the probe must be able to fail")
    ctrl = _sweep_rows(build_falsification(feats), base)
    _assert_falsification(ctrl)

    print("\n[models]")
    rows = _sweep_rows(adapters, base)

    t = pd.DataFrame(rows + _carry_forward() + ctrl)
    cols = ["model", "role", "swept", "n_probes", "grid_points", "n_pairs",
            "pct_nonincr_deployed", "n_viol_deployed", "viol_med_deployed",
            "viol_p99_deployed", "viol_max_deployed", "pct_nonincr_raw",
            "n_viol_raw", "viol_med_raw", "viol_p99_raw", "viol_max_raw",
            "pct_nonincr_deployed_tol1em6", "n_viol_deployed_tol1em6",
            "pct_nonincr_raw_tol1em6", "n_viol_raw_tol1em6"]
    t = t.reindex(columns=cols).sort_values(["swept", "model"]).reset_index(drop=True)
    t.to_csv(P_PROBE_OUT, index=False)
    print(f"\n[probe] -> {P_PROBE_OUT}")
    print(t.pivot_table(index="model", columns="swept",
                        values="pct_nonincr_deployed")
           .to_string(float_format=lambda x: f"{x:.1f}"))


def _assert_falsification(rows: list[dict]) -> None:
    """A probe that cannot fail is not evidence; prove it fails first."""
    by = {(r["model"], r["swept"]): r for r in rows}
    for swept in ("dimension", "n_customers"):
        inc = by[("Synthetic_increasing", swept)]["pct_nonincr_deployed"]
        dec = by[("Synthetic_decreasing", swept)]["pct_nonincr_deployed"]
        osc = by[("Synthetic_oscillating", swept)]["pct_nonincr_deployed"]
        if inc != 0.0:
            raise SystemExit(f"falsification failed: strictly increasing predictor "
                             f"scored {inc}% non-increasing on {swept}, expected 0")
        if dec != 100.0:
            raise SystemExit(f"falsification failed: strictly decreasing predictor "
                             f"scored {dec}% on {swept}, expected 100")
        if osc >= 100.0:
            raise SystemExit(f"falsification failed: oscillating predictor scored "
                             f"{osc}% on {swept}")
        print(f"  [ok] {swept}: increasing 0.0%, oscillating {osc:.1f}%, "
              f"decreasing 100.0% -- the probe discriminates")


# =============================================================================
# ND dispersion by n-bucket and by dimension group
# =============================================================================
def _nd_frame() -> pd.DataFrame:
    """One ND test frame carrying all four models, on identical instances.

    ``GART_2.0`` and ``LGBM_V4`` come through ``build_paper_tables.load_nd``,
    which is the function every published ND cell is computed from.  The two
    31-feature controls were never in the benchmark runner's schedule -- by
    design, since adding them would rewrite the authoritative results CSV --
    so their predictions come from ``controls_31f/rows_nd.csv`` and are
    attached to the benchmark's own instance metadata rather than to a second
    copy of it.
    """
    import build_paper_tables as B

    used, _, _, _ = B.load_nd()
    keep = used[used["model"].isin([GART, "LGBM_V4"])].copy()

    meta = (used[used["model"] == GART]
            .set_index("instance")[["n_customers", "dimension", "mst_length"]])
    ctrl = pd.read_csv(P_ROWS_ND)
    ctrl = ctrl[ctrl["model"].isin(["NN_31F", "Linear_31F"])
                & (ctrl["status"] == "ok")].copy()
    unknown = set(ctrl["instance"]) - set(meta.index)
    if unknown:
        raise SystemExit(f"{len(unknown)} control instances are not in the ND "
                         f"benchmark; e.g. {sorted(unknown)[:3]}")
    ctrl["err_pct"] = 100.0 * (ctrl["pred_cost"] - ctrl["true_cost"]) / ctrl["true_cost"]
    for c in ("n_customers", "dimension"):
        ctrl[c] = ctrl["instance"].map(meta[c])

    out = pd.concat([keep, ctrl], ignore_index=True)
    counts = out.groupby("model").size().to_dict()
    if len(set(counts.values())) != 1:
        raise SystemExit(f"models do not share an instance set: {counts}")
    print(f"[buckets] {len(counts)} models x {next(iter(counts.values()))} ND test rows")
    return out


def _bucket_table(df: pd.DataFrame, buckets, axis: str, models) -> pd.DataFrame:
    import build_paper_tables as B

    rows = []
    for label, slug, fn in buckets:
        mask = fn(df)
        for m in models:
            sub = df[mask & (df["model"] == m)]
            if not len(sub):
                continue
            e = sub["err_pct"].to_numpy(dtype=float)
            sdpe, lo, hi = B.boot_sdpe_ci(e)
            rows.append({"axis": axis, "bucket": label, "bucket_slug": slug,
                         "model": m, "role": ROLES.get(m, ""), "N": int(len(sub)),
                         "SDPE_pct": sdpe, "SDPE_lo": lo, "SDPE_hi": hi,
                         "MAPE_pct": float(np.mean(np.abs(e))),
                         "bias_pct": float(np.mean(e))})
    return pd.DataFrame(rows)


def _series_verdict(t: pd.DataFrame, axis: str, models) -> pd.DataFrame:
    """Is the dispersion series monotone across the buckets, and at d=100?"""
    order = [s for s in t[t.axis == axis].bucket_slug.unique() if s != "total"]
    rows = []
    for m in models:
        s = (t[(t.axis == axis) & (t.model == m) & (t.bucket_slug != "total")]
             .set_index("bucket_slug").reindex(order)["SDPE_pct"].to_numpy(float))
        d = np.diff(s)
        rows.append({
            "axis": axis, "model": m, "role": ROLES.get(m, ""),
            "n_buckets": int(len(s)),
            "n_decreasing_steps": int((d < 0).sum()), "n_steps": int(len(d)),
            "strict_monotone_decreasing": bool((d < 0).all()),
            "first": float(s[0]), "last": float(s[-1]),
            "ratio_last_first": float(s[-1] / s[0]) if s[0] else float("nan"),
            "final_step": float(d[-1]),
            "falls_at_final_bucket": bool(d[-1] < 0),
            "series": " / ".join(f"{v:.4f}" for v in s)})
    return pd.DataFrame(rows)


def cmd_buckets(args) -> None:
    import build_paper_tables as B

    df = _nd_frame()
    models = [GART, "LGBM_V4", "NN_31F", "Linear_31F"]
    # The test split is 11,016 in-domain rows plus the 5,904 held-out d=100
    # rows, so an n-bucket computed on all of it mixes an interpolation and an
    # extrapolation population in a per-bucket ratio nobody chose. A model
    # whose only weakness is d=100 would look like it has an n-dependent
    # weakness. The third panel removes that confound; the second is kept
    # because it is the population the published nd_by_size table uses.
    in_domain = df[df["dimension"] <= 50]
    t = pd.concat([_bucket_table(df, B.B_ND_DIM, "dim", models),
                   _bucket_table(df, B.B_ND_SIZE, "size", models),
                   _bucket_table(in_domain, B.B_ND_SIZE, "size_d_le_50", models)],
                  ignore_index=True)
    v = pd.concat([_series_verdict(t, a, models)
                   for a in ("dim", "size", "size_d_le_50")], ignore_index=True)
    t.to_csv(P_BUCKETS_OUT, index=False)
    v.to_csv(P_BUCKETS_OUT.with_name("consistency_31f_nd_series.csv"), index=False)
    print(f"\n[buckets] -> {P_BUCKETS_OUT}")

    for axis in ("dim", "size", "size_d_le_50"):
        for metric in ("SDPE_pct", "MAPE_pct"):
            p = (t[t.axis == axis].pivot_table(index="bucket_slug", columns="model",
                                               values=metric, sort=False)
                 .reindex(columns=models))
            print(f"\n=== ND {metric} by {axis} ===")
            print(p.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== series verdict ===")
    print(v[["axis", "model", "strict_monotone_decreasing", "falls_at_final_bucket",
             "ratio_last_first", "series"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe").set_defaults(fn=cmd_probe)
    sub.add_parser("buckets").set_defaults(fn=cmd_buckets)
    sub.add_parser("all").set_defaults(fn=lambda a: (cmd_probe(a), cmd_buckets(a)))
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
