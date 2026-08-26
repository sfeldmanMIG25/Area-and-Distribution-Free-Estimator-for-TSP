"""Is arm A's advantage real, or one lucky seed?

Two questions the gate table cannot answer on its own:

1. Is the ND-test improvement significant? Arm A adds 874 rows to a 69,768-row
   training split and the ND test is untouched by them, so any ND movement is
   generalisation rather than recall -- but 0.02 MAPE could still be noise.
   Tested paired, per instance, on all 16,920 rows.

2. Does the result survive reseeding? The pre-registered arm is seed 42, as the
   frozen recipe specifies. Refitting at other seeds is NOT the shipped arm and
   is labelled as such; it exists only to show whether the gate margins are
   structural or a draw of the bagging RNG.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import support_arms_study as S  # noqa: E402

SEEDS = (7, 123, 2024)


def main() -> None:
    feats31 = S.frozen_features()
    corpus = S.load_corpus(feats31)
    aug = S.load_augment(feats31)
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    tr_aug = pd.concat([tr, aug[tr.columns.intersection(aug.columns)]],
                       ignore_index=True, sort=False)

    C = pd.read_csv(HERE / "v4_study_feature_cache.csv", low_memory=False)
    gen = pd.read_csv(HERE / "augmentation_2d_features.csv",
                      usecols=["instance_name", "generator"])
    topo = pd.read_csv(HERE / "support_arms_feats_bench2d.csv")
    b2 = C[C.stratum == "bench2d"].merge(
        gen, left_on="instance", right_on="instance_name", how="left")
    b2 = b2.merge(topo, on="instance_name", how="left", suffixes=("", "_t"))
    nd = C[C.stratum == "nd_test"]

    # ---- 1. paired significance of the ND gain --------------------------
    PI = pd.read_csv(HERE / "support_arms_per_instance.csv")
    nd_pi = PI[PI.stratum == "nd_test"]
    fz = nd_pi[nd_pi.model == "FROZEN"].set_index("instance")["err_pct"].abs()
    out = {}
    for cand in ("A", "B"):
        ca = nd_pi[nd_pi.model == cand].set_index("instance")["err_pct"].abs()
        idx = fz.index.intersection(ca.index)
        d = (ca.loc[idx] - fz.loc[idx]).to_numpy()
        w = stats.wilcoxon(d, zero_method="zsplit")
        t = stats.ttest_1samp(d, 0.0)
        out[f"nd_{cand}_vs_frozen"] = {
            "n": int(len(d)), "mean_diff_pp": float(d.mean()),
            "frozen_mape": float(fz.loc[idx].mean()),
            "candidate_mape": float(ca.loc[idx].mean()),
            "p_wilcoxon": float(w.pvalue), "p_ttest": float(t.pvalue),
            "pct_instances_improved": float((d < 0).mean() * 100),
        }

    # ---- 2. seed robustness for arm A -----------------------------------
    rows = []
    for seed in SEEDS:
        S.SEED = seed
        b = S.fit_arm(tr_aug, va, feats31, f"A_seed{seed}")
        S.SEED = 42

        def sc(frame):
            ok = frame[(frame.status == "ok") & frame[feats31].notna().all(axis=1)
                       & frame.true_cost.notna()].copy()
            a = np.clip(S.to_alpha(b.predict(ok[feats31], num_iteration=b.best_iteration)),
                        1.0, 2.0)
            ok["pa"] = a
            ok["err"] = (a * ok.mst_total_length.to_numpy()
                         - ok.true_cost.to_numpy()) / ok.true_cost.to_numpy() * 100
            return ok

        o_nd, o_b2 = sc(nd), sc(b2)
        ln = o_b2[(o_b2.generator == "line_noise") & (o_b2.n_customers >= 200)]
        lr = stats.linregress(np.clip(ln.true_cost / ln.mst_total_length, 1, 2), ln.pa)
        gr = o_b2[o_b2.generator == "grid"]
        rows.append({
            "seed": seed, "trees": int(b.num_trees()),
            "nd_mape": float(o_nd.err.abs().mean()),
            "nd_sdpe": float(o_nd.err.std(ddof=1)),
            "bench2d_mape": float(o_b2.err.abs().mean()),
            "grid_mspe": float(gr.err.mean()),
            "linenoise_slope": float(lr.slope),
        })
        print(f"  seed {seed}: ND {rows[-1]['nd_mape']:.4f} | 2D "
              f"{rows[-1]['bench2d_mape']:.4f} | grid MSPE "
              f"{rows[-1]['grid_mspe']:+.4f} | slope {rows[-1]['linenoise_slope']:.4f}",
              flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(HERE / "support_arms_seed_robustness.csv", index=False)
    json.dump(out, open(HERE / "support_arms_nd_significance.json", "w"), indent=2)

    print("\n=== ND paired significance (negative mean_diff favours candidate) ===")
    print(json.dumps(out, indent=2))
    print("\n=== arm A across seeds (seed 42 is the pre-registered arm) ===")
    ref = {"seed": 42, "trees": 1203, "nd_mape": 0.5965, "nd_sdpe": 0.9839,
           "bench2d_mape": 2.0533, "grid_mspe": 1.5400, "linenoise_slope": 0.8268}
    print(pd.concat([pd.DataFrame([ref]), R], ignore_index=True)
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\ngates that must hold at every seed: ND MAPE <= 0.6401, "
          "grid MSPE <= 4.0, slope >= 0.70")


if __name__ == "__main__":
    main()
