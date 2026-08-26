"""Deciding-reviewer check: does gate 2 depend on which dispersion test is used?

Runs ood_harness.dispersion_verdict for FROZEN and shipped arm A, and prints
all three Holm-adjusted p columns plus normality diagnostics on the paired
signed-error vectors. No refits, no writes outside armA_verify_*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import armA_verify_common as K  # noqa: E402
import ood_harness as OH  # noqa: E402


def preds(model, feats, Cok) -> dict:
    a = np.clip(K.to_alpha(model.predict(Cok[feats], num_iteration=model.best_iteration)),
                *K.ALPHA_CLIP)
    return dict(zip(Cok["instance"].astype(str),
                    a * Cok["mst_total_length"].to_numpy()))


def main() -> None:
    D = K.load_cache()
    C, feats = D["cache"], D["feats31"]
    Cok = C[C.status == "ok"]

    models = {
        "FROZEN": joblib.load(K.FROZEN),
        "armA": joblib.load(HERE / "armA_verify_models" / "A_full.joblib"),
    }
    out = []
    for label, m in models.items():
        t = OH.dispersion_verdict({label: preds(m, feats, Cok)})
        r = t[t.model_b == "Asymptotic_MST"].iloc[0]
        out.append({
            "model": label,
            "n": int(r.n_pairs), "sd_ratio": float(r.sd_ratio),
            "r_xy": float(r.r_xy),
            "p_pm_holm": float(r.p_holm),
            "p_bf_holm": float(r.p_bf_holm),
            "p_swap_holm": float(r.p_swap_holm),
            "mde_holm": float(r.mde_sd_ratio_holm),
            "detectable_holm": bool(r.detectable_holm),
            "family": int(len(t)),
        })
    T = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))

    # normality of the quantities the parametric test assumes
    s = OH.load_suite()["tsplib_euc2d"]
    truth = s.truth
    base = s.baselines["Asymptotic_MST"].reindex(truth.index)
    eb = (base - truth) / truth * 100.0
    print("\nnormality of paired signed-error vectors (EUC_2D, N=78)")
    for label, m in models.items():
        p = pd.Series(preds(m, feats, Cok)).reindex(truth.index)
        ec = (p - truth) / truth * 100.0
        for nm, v in (("cand", ec), ("sum x+y", ec + eb), ("diff x-y", ec - eb)):
            v = v.dropna().to_numpy()
            print(f"  {label:7s} {nm:8s} skew {stats.skew(v):+.3f} "
                  f"exkurt {stats.kurtosis(v):+.3f} "
                  f"shapiro p {stats.shapiro(v).pvalue:.3e} "
                  f"jarque p {stats.jarque_bera(v).pvalue:.3e}")

    T.to_csv(HERE / "armA_verify_decide_disp.csv", index=False)


if __name__ == "__main__":
    main()
