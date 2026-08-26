"""Do the two TSPLIB gates survive nuisance variation?

Gate 2 = dispersion ratio vs Asymptotic_MST, Holm p < 0.05.
Gate 3 = mean APE gain vs Calibrated_MST_dn, Holm p < 0.05.

Both are re-run for every nuisance variant of arm A: the eight row-order
permutations of the identical 874 rows, and six LightGBM seeds. If a gate flips
under a change that carries no information, it is not evidence about the model.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import armA_verify_common as K
from support_arms_eval import tsplib_significance

HERE = Path(__file__).resolve().parent
OUT = HERE / "armA_verify_tsplib_sig.csv"


def preds(model, feats, Cok) -> dict:
    a = np.clip(K.to_alpha(model.predict(Cok[feats], num_iteration=model.best_iteration)),
                *K.ALPHA_CLIP)
    return dict(zip(Cok["instance"].astype(str),
                    a * Cok["mst_total_length"].to_numpy()))


def main() -> None:
    D = K.load_cache()
    corpus, aug, C, feats = D["corpus"], D["aug"], D["cache"], D["feats31"]
    tr = corpus[corpus.split == "train"]
    va = corpus[corpus.split == "val"]
    Cok = C[C.status == "ok"]
    cols = [c for c in tr.columns if c in aug.columns]

    rows = []
    fr = joblib.load(K.FROZEN)
    rows.append({"variant": "FROZEN", "kind": "control",
                 **tsplib_significance(preds(fr, feats, Cok), "FROZEN")})
    a0 = joblib.load(HERE / "armA_verify_models" / "A_full.joblib")
    rows.append({"variant": "armA_shipped", "kind": "shipped",
                 **tsplib_significance(preds(a0, feats, Cok), "armA")})
    print(pd.DataFrame(rows)[["variant", "disp_ratio", "disp_p_holm",
                              "mean_gain", "mean_p_holm"]].to_string(index=False))

    N = len(aug)
    for perm in range(8):
        order = np.random.default_rng(1000 + perm).permutation(N)
        train = pd.concat([tr, aug.iloc[order][cols]], ignore_index=True, sort=False)
        m = K.fit(train, va, feats, seed=42)
        r = {"variant": f"roworder_p{perm}", "kind": "row_order",
             **tsplib_significance(preds(m, feats, Cok), "cand")}
        rows.append(r)
        print(f"  roworder_p{perm}: disp {r.get('disp_ratio', float('nan')):.4f} "
              f"p {r.get('disp_p_holm', float('nan')):.4g} | gain "
              f"{r.get('mean_gain', float('nan')):.4f} p "
              f"{r.get('mean_p_holm', float('nan')):.4g}", flush=True)

    for s in (11, 97, 314, 1729, 5150):
        train = pd.concat([tr, aug[cols]], ignore_index=True, sort=False)
        m = K.fit(train, va, feats, seed=s)
        r = {"variant": f"seed_{s}", "kind": "seed",
             **tsplib_significance(preds(m, feats, Cok), "cand")}
        rows.append(r)
        print(f"  seed_{s}: disp {r.get('disp_ratio', float('nan')):.4f} "
              f"p {r.get('disp_p_holm', float('nan')):.4g} | gain "
              f"{r.get('mean_gain', float('nan')):.4f} p "
              f"{r.get('mean_p_holm', float('nan')):.4g}", flush=True)

    T = pd.DataFrame(rows)
    T["G2_pass"] = (T["disp_ratio"] < 1.0) & (T["disp_p_holm"] < 0.05)
    T["G3_pass"] = (T["mean_gain"] > 0) & (T["mean_p_holm"] < 0.05)
    T.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print("\n" + T[["variant", "kind", "disp_ratio", "disp_p_holm", "disp_mde_holm",
                    "G2_pass", "mean_gain", "mean_p_holm", "mean_mde", "G3_pass",
                    "mape_model"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    v = T[T.kind.isin(("row_order", "seed"))]
    print(f"\nover {len(v)} nuisance variants: G2 passes {int(v.G2_pass.sum())}/{len(v)}, "
          f"G3 passes {int(v.G3_pass.sum())}/{len(v)}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
