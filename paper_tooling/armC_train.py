"""Train and score every arm C run required by the protocol.

Runs produced
-------------
``C_s<seed>``      arm C (850 augmentation rows), protocol order, 7 named seeds
``A_s<seed>``      arm A (874 rows) under the SAME protocol order and seeds, so
                   that "removing the 24 rows helped/hurt" is a within-protocol
                   statement rather than a comparison across nuisance regimes
``R0p_s<seed>``    no augmentation, protocol order, 7 seeds -- isolates what the
                   corpus re-sort alone does, since protocol s2's fixed order is
                   itself a row-order change relative to the frozen fit
``Cperm<k>``       8 row-order permutations of the identical arm C rows at the
                   median seed (protocol s2 bullet 4)
``Cuncon``         arm C refit with the monotone constraints removed -- the
                   control gate 4 now requires beside the probe
``Cfold<k>``       5-fold refits holding out augmentation rows, for the held-out
                   augment stratum figure obligation s4.6 demands

Writes only ``armC_*`` artifacts.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import armA_verify_common as K  # noqa: E402
import armC_common as CC  # noqa: E402

MODELS = HERE / "armC_models"
PROBE_TOL = 1e-9
PROBE_N = 1000
PROBE_SEED = 42


def log_int_grid(lo: int, hi: int, k: int = 24) -> list[int]:
    return [int(v) for v in
            np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), k)).astype(int))]


D_GRID = log_int_grid(2, 200)
N_GRID = log_int_grid(5, 4000)


def monotonicity(model, feats, base: pd.DataFrame) -> dict:
    out = {}
    for col, grid in (("dimension", D_GRID), ("n_customers", N_GRID)):
        P = np.empty((len(base), len(grid)))
        for j, g in enumerate(grid):
            X = base.copy()
            X[col] = g
            P[:, j] = model.predict(X[feats], num_iteration=model.best_iteration)
        A = K.to_alpha(P)
        d = np.diff(A, axis=1)
        viol = d > PROBE_TOL
        out[f"mono_{col}_pct"] = float((~viol.any(axis=1)).mean()) * 100.0
        out[f"mono_{col}_nviol"] = int(viol.sum())
        out[f"mono_{col}_maxviol"] = float(d[viol].max()) if viol.any() else 0.0
    return out


def fit_uncon(train, val, feats, seed):
    """Same recipe with the monotone constraints removed (gate 4 control)."""
    import lightgbm as lgb
    params = dict(K.V3_FROZEN)
    params.update({"objective": "regression_l2", "metric": "None",
                   "boosting_type": "gbdt", "seed": seed,
                   "num_threads": K.NUM_THREADS, "verbosity": -1,
                   "feature_pre_filter": False})
    mst_v, cost_v = val["mst_total_length"].to_numpy(), val["optimal_cost"].to_numpy()

    def feval(preds, _ds):
        a = K.to_alpha(preds)
        return "cost_mape", float(np.mean(np.abs((a * mst_v - cost_v) / cost_v)) * 100), False

    dtr = lgb.Dataset(train[feats], label=K.to_z(train["alpha"].to_numpy()))
    dvl = lgb.Dataset(val[feats], label=K.to_z(val["alpha"].to_numpy()), reference=dtr)
    return lgb.train(params, dtr, num_boost_round=K.MAX_BOOST_ROUND,
                     valid_sets=[dvl], valid_names=["val"], feval=feval,
                     callbacks=[lgb.early_stopping(K.EARLY_STOP, verbose=False),
                                lgb.log_evaluation(0)])


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    D = K.load_cache()
    corpus, aug, C, feats = D["corpus"], D["aug"], D["cache"], D["feats31"]
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    keepC = CC.armC_aug(aug)
    probe_base = corpus[corpus.split == "test"].sample(PROBE_N,
                                                       random_state=PROBE_SEED).copy()
    frozen = joblib.load(K.FROZEN)

    specs: list[tuple[str, pd.DataFrame | None, int, int | None, bool]] = []
    for s in CC.SEEDS:
        specs.append((f"C_s{s}", keepC, s, None, True))
    for s in CC.SEEDS:
        specs.append((f"A_s{s}", aug, s, None, True))
    for s in CC.SEEDS:
        specs.append((f"R0p_s{s}", None, s, None, True))

    runs, preds_all, euc_err, b2_store = [], {}, {}, {}

    def record(tag: str, model, extra: dict) -> dict:
        mm = CC.extended_metrics(model, feats, C)
        preds_all[tag] = mm.pop("_preds_all")
        euc_err[tag] = mm.pop("_euc2d_err")
        b2_store[tag] = mm.pop("_b2")
        r = {"run": tag, **{k: mm[k] for k in CC.SCALARS}, **extra}
        r["features_equal_frozen"] = list(model.feature_name()) == list(feats)
        return r

    # frozen reference row, scored by this harness
    r = record("FROZEN", frozen, {"arm": "FROZEN", "seed": -1, "n_aug": 0,
                                  "row_perm": -1, "monotone": True})
    r.update(monotonicity(frozen, feats, probe_base))
    runs.append(r)
    print(f"[frozen] slope {r['linenoise_slope']:.4f} grid {r['b2_grid_mspe']:+.4f}",
          flush=True)

    t0 = time.perf_counter()
    for tag, sub, seed, perm, mono in specs:
        train = CC.protocol_train(tr, sub, row_perm=perm)
        m = K.fit(train, va, feats, seed=seed)
        joblib.dump(m, MODELS / f"{tag}.joblib")
        arm = tag.split("_s")[0]
        r = record(tag, m, {"arm": arm, "seed": seed,
                            "n_aug": 0 if sub is None else int(len(sub)),
                            "row_perm": -1, "monotone": mono})
        if arm == "C":
            r.update(monotonicity(m, feats, probe_base))
        runs.append(r)
        print(f"[{tag:10s}] n_aug={r['n_aug']:4d} slope={r['linenoise_slope']:.4f} "
              f"grid={r['b2_grid_mspe']:+.4f} 2d={r['bench2d_mape']:.4f} "
              f"nd={r['nd_test_mape']:.4f} euc_sdpe={r['tsplib_euc2d_sdpe']:.4f}",
              flush=True)

    R = pd.DataFrame(runs)

    # ---- median seed, defined on the gate-6 statistic -------------------
    cs = R[R.arm == "C"].sort_values("linenoise_slope")
    median_seed = int(cs.iloc[len(cs) // 2]["seed"])
    print(f"\n[median seed on linenoise_slope] {median_seed}", flush=True)

    # ---- row-order permutations at the median seed ----------------------
    for k in range(CC.N_ROW_PERMS):
        tag = f"Cperm{k}"
        train = CC.protocol_train(tr, keepC, row_perm=1000 + k)
        m = K.fit(train, va, feats, seed=median_seed)
        r = record(tag, m, {"arm": "Cperm", "seed": median_seed,
                            "n_aug": int(len(keepC)), "row_perm": k, "monotone": True})
        runs.append(r)
        print(f"[{tag:10s}] slope={r['linenoise_slope']:.4f} "
              f"grid={r['b2_grid_mspe']:+.4f} 2d={r['bench2d_mape']:.4f} "
              f"ln={r['b2_line_noise_mape']:.4f} "
              f"euc_sdpe={r['tsplib_euc2d_sdpe']:.4f}", flush=True)

    # ---- unconstrained control for gate 4 -------------------------------
    train = CC.protocol_train(tr, keepC)
    mu = fit_uncon(train, va, feats, median_seed)
    r = record("Cuncon", mu, {"arm": "Cuncon", "seed": median_seed,
                              "n_aug": int(len(keepC)), "row_perm": -1,
                              "monotone": False})
    r.update(monotonicity(mu, feats, probe_base))
    runs.append(r)
    print(f"[Cuncon    ] mono d={r['mono_dimension_pct']:.1f}% "
          f"n={r['mono_n_customers_pct']:.1f}% slope={r['linenoise_slope']:.4f}",
          flush=True)

    # ---- held-out augment refits (obligation s4.6) ----------------------
    names = np.array(sorted(keepC.instance_name.astype(str)))
    fold_of = {nm: i % 5 for i, nm in
               enumerate(np.random.default_rng(42).permutation(names))}
    keepC = keepC.assign(_fold=keepC.instance_name.astype(str).map(fold_of))
    ho_rows = []
    for k in range(5):
        sub = keepC[keepC._fold != k].drop(columns="_fold")
        held = keepC[keepC._fold == k]
        train = CC.protocol_train(tr, sub)
        m = K.fit(train, va, feats, seed=median_seed)
        g = C[(C.stratum == "augment")
              & C.instance.astype(str).isin(set(held.instance_name.astype(str)))]
        ok = K.score_frame(m, feats, g)
        ho_rows.append(ok[["instance", "err_pct"]].assign(fold=k))
        print(f"[Cfold{k}    ] held-out {len(ok)} rows "
              f"MAPE {np.mean(np.abs(ok.err_pct)):.4f}", flush=True)
    HO = pd.concat(ho_rows, ignore_index=True)
    HO.to_csv(HERE / "armC_augment_heldout.csv", index=False)

    R = pd.DataFrame(runs)
    R.to_csv(HERE / "armC_runs.csv", index=False)
    with open(HERE / "armC_preds.pkl", "wb") as fh:
        pickle.dump({"preds_all": preds_all, "euc_err": euc_err,
                     "b2": b2_store, "median_seed": median_seed}, fh, protocol=5)
    print(f"\nwrote armC_runs.csv ({len(R)} runs), armC_preds.pkl, "
          f"armC_augment_heldout.csv in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
