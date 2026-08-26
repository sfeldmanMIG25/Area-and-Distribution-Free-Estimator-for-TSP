"""Score the 31-feature same-feature controls on every benchmark stratum.

Produces one tidy row per (model, instance) for ``Linear_31F``, ``NN_31F`` and
every fitted neural seed, on the multidimensional test split, the 2D benchmark
and both TSPLIB strata, plus GART 2.0 rows used only as an equivalence gate.

ONE FEATURE BUILD PER INSTANCE, SHARED BY EVERY MODEL
-----------------------------------------------------
Each stratum extracts GART 2.0's 31-column vector once per instance and then
asks every model for an alpha on that one vector. That is not only cheaper than
running each model end to end -- it is the demonstration the task is about. The
models cannot be scored on different inputs, because there is only one input.

WHY THIS DOES NOT JUST CALL THE BENCHMARK RUNNERS
-------------------------------------------------
``run_benchmark_2D_all.py`` and ``run_benchmark_ND_final.py`` end by globbing
``benchmark_checkpoints/results_*.csv`` and rewriting the authoritative results
CSV. Adding two models to their schedules would rewrite
``benchmark_results_2D_v3.csv`` and ``benchmark_results_ND_final.csv`` -- the
inputs every published number is computed from -- as a side effect of scoring a
control. This script writes only into its own output directory.

The price of not reusing the runners is that the equivalence has to be proved
rather than assumed, so every stratum also scores GART 2.0 and checks it
against the published rows before any control number is reported.

The TSPLIB stratum is the exception: its hybrid path (classical MDS, MST from
the original distance matrix, geometry from the embedding, the
``greedy_nn_over_mst`` in-distribution guard) exists only inside
``run_all_models_tsplib``, so that runner's own ``_run_one_instance`` is called
and the feature vector is captured from inside it.

EQUIVALENCE OF THE MULTIDIMENSIONAL STRATUM
-------------------------------------------
The ND benchmark recomputes features from coordinates; this script reads them
from ``tsp_features_v4.csv``. Those agree exactly -- GART 2.0's benchmark rows
reproduce its sidecar test metrics to the last float digit (0.6201146416247372
vs 0.620114641624737 MAPE) -- and the gate re-verifies it per row rather than
trusting that.

Usage:
    python paper_tooling/score_31f_controls.py --stratum nd
    python paper_tooling/score_31f_controls.py --stratum 2d
    python paper_tooling/score_31f_controls.py --stratum tsplib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "lgbm_model_v3"), str(ROOT / "linear_model_v3"),
           str(ROOT / "nn_est_alpha_v3"), str(ROOT / "paper_tooling")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_registry import GART  # noqa: E402

OUT = ROOT / "paper_tooling" / "controls_31f"
OUT.mkdir(exist_ok=True)

P_ND = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_results_ND_final.csv"
P_2D = ROOT / "Generalized_TSP_Analysis" / "benchmark_results_2D_v3.csv"
P_2D_GT = ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv"
P_TSPLIB = ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv"
NN_DIR = ROOT / "nn_est_alpha_v3"

#: Relative agreement required between a re-scored GART 2.0 prediction and its
#: published value. Feature extraction runs in float32 and the greedy tour has
#: documented tie-breaking sensitivity, so this is a float-noise tolerance, not
#: a modelling tolerance.
GATE_RTOL = 1e-6


def build_bank() -> tuple[dict, list[str]]:
    """``name -> estimator`` for every model scored here, all sharing one vector.

    Includes one entry per fitted neural seed. The 2D margin this study exists
    to settle is 0.08 pp of SDPE and the neural seed band is an order of
    magnitude wider, so reporting the shipped seed alone would report one draw
    from a nuisance distribution as a result -- the failure mode that got the
    augmented arms rejected.
    """
    from lgbm_model_v3.lgbm_estimator_gart2 import TSP_GART2_Estimator
    from linear_model_v3.estimator_linear_31f import TSP_Linear31_Estimator
    from nn_est_alpha_v3.estimator_nn_31f import TSP_NN31_Estimator

    bank = {
        GART: TSP_GART2_Estimator(str(ROOT / "lgbm_model_v3")),
        "Linear_31F": TSP_Linear31_Estimator(str(ROOT / "linear_model_v3")),
        "NN_31F": TSP_NN31_Estimator(str(NN_DIR)),
    }
    seeds = sorted(int(p.stem.split("seed")[1])
                   for p in (NN_DIR / "nn_31f_seeds").glob("nn_31f_seed*.pt"))
    for s in seeds:
        bank[f"NN_31F_s{s}"] = TSP_NN31_Estimator(
            str(NN_DIR), checkpoint=str(NN_DIR / "nn_31f_seeds" / f"nn_31f_seed{s}.pt"))

    ref = list(bank[GART].features_required)
    for name, e in bank.items():
        if list(e.features_required) != ref:
            raise RuntimeError(f"{name} does not share the production feature vector")
    print(f"[31f] {len(bank)} models share {len(ref)} features in identical order "
          f"(neural seeds: {seeds})")
    return bank, ref


def _rows_from_features(bank, instance, feats, mst_len, true_cost) -> list[dict]:
    out = []
    for name, e in bank.items():
        a = e.predict_alpha(feats)
        out.append({"model": name, "instance": instance, "pred_cost": a * mst_len,
                    "true_cost": true_cost, "mst_length": mst_len, "alpha": a,
                    "status": "ok"})
    return out


def _status_rows(bank, instance, true_cost, status, **extra) -> list[dict]:
    return [{"model": name, "instance": instance, "pred_cost": np.nan,
             "true_cost": true_cost, "mst_length": np.nan, "alpha": np.nan,
             "status": status, **extra} for name in bank]


# =============================================================================
# Stratum 1: multidimensional test split
# =============================================================================
def score_nd() -> pd.DataFrame:
    from control_features_31f import load_splits, production_features

    bank, features = build_bank()
    if features != production_features():
        raise RuntimeError("bank feature order disagrees with the production list")
    test = load_splits(features)["test"]
    X, mst, cost, names = test["X"], test["mst"], test["cost"], test["instance"]
    print(f"[31f/nd] {len(X)} test rows")

    rows = []
    for i in tqdm(range(len(X)), desc="nd", leave=False):
        feats = dict(zip(features, X.iloc[i].to_numpy()))
        rows.extend(_rows_from_features(bank, names[i], feats, float(mst[i]), float(cost[i])))
    return _finish(pd.DataFrame(rows), P_ND, "nd", "rows_nd.csv")


# =============================================================================
# Stratum 2: 2D benchmark
# =============================================================================
def score_2d() -> pd.DataFrame:
    from tsp_utils import parse_tsp_instance
    from feature_engineering_gart2 import compute_features

    bank, _ = build_bank()
    gt = pd.read_csv(P_2D_GT)
    print(f"[31f/2d] {len(gt)} instances")

    rows = []
    for r in tqdm(gt.itertuples(index=False), total=len(gt), desc="2d", leave=False):
        # Identical to TSP_GART2_Estimator.estimate: collapse duplicate points,
        # decline below three distinct points, then the production extractor.
        coords = np.unique(np.asarray(parse_tsp_instance(Path(r.file_path)).coordinates,
                                      dtype=np.float32), axis=0)
        if coords.shape[0] < 3:
            rows.extend(_status_rows(bank, r.instance, r.true_cost, "n<3"))
            continue
        feats = compute_features(coords, int(r.dimension))
        rows.extend(_rows_from_features(bank, r.instance, feats,
                                        float(feats["mst_total_length"]), r.true_cost))
    return _finish(pd.DataFrame(rows), P_2D, "2d", "rows_2d.csv")


# =============================================================================
# Strata 3 and 4: TSPLIB, through the runner's own instance loop
# =============================================================================
def _make_capture(model_dir: str):
    """GART 2.0, subclassed so the runner hands back the vector it built.

    ``predict_alpha`` is the one point both the native and the hybrid paths
    funnel through, so overriding it captures the vector in either mode without
    duplicating a line of feature code. It must be a *subclass* and not a
    delegating wrapper: ``TSP_GART2_Estimator.estimate`` calls
    ``self.predict_alpha``, which on a wrapper binds to the inner object and
    silently bypasses the capture on every native instance -- which is to say
    on all 78 TSPLIB EUC_2D rows.
    """
    from lgbm_model_v3.lgbm_estimator_gart2 import TSP_GART2_Estimator

    class _CaptureGART2(TSP_GART2_Estimator):
        def __init__(self, d):
            super().__init__(d)
            self.current: str | None = None
            self.captured: dict[str, dict] = {}

        def predict_alpha(self, feats):
            self.captured[self.current] = dict(feats)
            return super().predict_alpha(feats)

    return _CaptureGART2(model_dir)


def score_tsplib() -> pd.DataFrame:
    import tsplib_benchmark.run_all_models_tsplib as R

    bank, _ = build_bank()
    del bank[GART]
    cap = _make_capture(str(ROOT / "lgbm_model_v3"))
    optima = R.load_optima()
    tasks = [(p, p.stem, optima[p.stem], None)
             for p in sorted((ROOT / "tsplib_benchmark" / "instances").glob("*.tsp"))
             if p.stem in optima]
    print(f"[31f/tsplib] {len(tasks)} instances")

    rows = []
    for task in tqdm(tasks, desc="tsplib", leave=False):
        cap.current = task[1]
        # _run_one_instance always appends its own Fixed_Alpha baseline row on
        # top of the requested ml_models, so the production row has to be
        # selected by name rather than by position.
        emitted = R._run_one_instance(task, {GART: cap}, {}, [GART], {})
        gart = [r for r in emitted if r["model"] == GART]
        if len(gart) != 1:
            raise SystemExit(f"[31f/tsplib] {task[1]}: expected one {GART} row, "
                             f"got {len(gart)}")
        g = gart[0]
        rows.append(g)

        meta = {k: g[k] for k in ("n", "edge_weight_type", "mode", "feature_dim")}
        feats = cap.captured.get(task[1])
        if g["status"] != "ok":
            # The greedy in-distribution guard, an MDS failure or a parse
            # failure is a property of the instance and of the shared feature
            # vector, so it applies identically to every model.
            rows.extend(_status_rows(bank, task[1], g["true_cost"], g["status"], **meta))
            continue
        if feats is None:
            # Never-silent: a scored instance with no captured vector means the
            # interception missed a code path. Emitting NaN under status "ok"
            # would look like coverage and read as a model failure.
            raise SystemExit(f"[31f/tsplib] {task[1]}: {GART} returned status 'ok' "
                             f"but no feature vector was captured")
        rows.extend({**r, **meta} for r in _rows_from_features(
            bank, task[1], feats, float(g["mst_length"]), g["true_cost"]))
    return _finish(pd.DataFrame(rows), P_TSPLIB, "tsplib", "rows_tsplib.csv")


# =============================================================================
# The equivalence gate
# =============================================================================
def _finish(df: pd.DataFrame, published: Path, tag: str, out_name: str) -> pd.DataFrame:
    _gate(df, pd.read_csv(published, low_memory=False), tag)
    df[df.model != GART].to_csv(OUT / out_name, index=False)
    print(f"[31f/{tag}] wrote {out_name} "
          f"({len(df[df.model != GART])} rows, {df.model.nunique() - 1} models)")
    return df


def _gate(fresh: pd.DataFrame, published: pd.DataFrame, tag: str) -> None:
    """Fail unless re-scored GART 2.0 reproduces its published predictions."""
    a = fresh[fresh.model == GART].set_index("instance")["pred_cost"]
    b = published[published.model == GART].set_index("instance")["pred_cost"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise SystemExit(f"[31f/{tag}] gate: no shared {GART} instances to compare")

    av, bv = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    both_nan = ~np.isfinite(av) & ~np.isfinite(bv)
    finite = np.isfinite(av) & np.isfinite(bv)
    if not (both_nan | finite).all():
        bad = common[~(both_nan | finite)].tolist()
        raise SystemExit(f"[31f/{tag}] gate: coverage disagrees with the published run "
                         f"on {len(bad)} instances, e.g. {bad[:5]}")

    rel = np.abs(av[finite] - bv[finite]) / np.abs(bv[finite])
    worst = float(rel.max()) if rel.size else 0.0
    print(f"[31f/{tag}] gate: {GART} reproduces {int(finite.sum())} published "
          f"predictions, max relative difference {worst:.3e} "
          f"({int(both_nan.sum())} declined on both sides)")
    (OUT / f"gate_{tag}.json").write_text(json.dumps({
        "stratum": tag, "n_compared": int(finite.sum()),
        "n_declined_both": int(both_nan.sum()),
        "max_relative_difference": worst, "tolerance": GATE_RTOL,
        "passed": bool(worst <= GATE_RTOL),
    }, indent=2), encoding="utf-8")
    if worst > GATE_RTOL:
        i = int(np.argmax(rel))
        raise SystemExit(f"[31f/{tag}] gate FAILED: {common[finite][i]} "
                         f"fresh={av[finite][i]!r} published={bv[finite][i]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", choices=["nd", "2d", "tsplib", "all"], default="all")
    args = ap.parse_args()
    todo = ["nd", "2d", "tsplib"] if args.stratum == "all" else [args.stratum]
    for s in todo:
        {"nd": score_nd, "2d": score_2d, "tsplib": score_tsplib}[s]()
