"""Gates 2 and 3 for every arm C seed, read off the shipped OOD harness.

``ood_harness`` is used unmodified so that the multiplicity family, the Holm
correction and the MDE are the same ones the frozen model and arm A were judged
in. Gate 2 now needs three p-values (Pitman-Morgan, Brown-Forsythe, swap
permutation) plus ``detectable_holm``; gate 3 needs the mean gain against
``Calibrated_MST_dn`` with its own MDE.

Writes only armC_significance.csv.
"""
from __future__ import annotations

import pickle
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N_WORKERS = 5


def one(task: tuple[str, dict]) -> dict:
    import ood_harness as oh

    label, preds = task
    res: dict = {"run": label}

    d = oh.dispersion_verdict({label: preds}, stratum="tsplib_euc2d")
    r = d[d.model_b == "Asymptotic_MST"].iloc[0]
    res.update({
        "disp_n": int(r.n_pairs), "disp_sd_ratio": float(r.sd_ratio),
        "disp_sdpe_cand": float(r.sdpe_a), "disp_sdpe_base": float(r.sdpe_b),
        "disp_ci_low": float(r.ratio_ci_low), "disp_ci_high": float(r.ratio_ci_high),
        "disp_r_xy": float(r.r_xy),
        "disp_p_pm_holm": float(r.p_holm),
        "disp_p_bf_holm": float(r.p_bf_holm),
        "disp_p_swap_holm": float(r.p_swap_holm),
        "disp_p_pm_raw": float(r.p_pitman_morgan),
        "disp_p_swap_raw": float(r.p_swap_permutation),
        "disp_mde_holm": float(r.mde_sd_ratio_holm),
        "disp_detectable_holm": bool(r.detectable_holm),
        "disp_family": int(len(d)),
    })

    v = oh.evaluate_candidate(preds, label)
    c = v.comparisons
    c = c[(c.stratum == "tsplib_euc2d") & (c.model_b == "Calibrated_MST_dn")].iloc[0]
    s = oh.load_suite()["tsplib_euc2d"]
    pa = pd.Series({k: val for k, val in preds.items() if k in s.truth.index},
                   dtype=float)
    pb = s.baselines["Calibrated_MST_dn"]
    idx = [i for i in pa.index if i in pb.index and np.isfinite(pb[i])]
    a = oh.absolute_percent_error(pa.loc[idx], s.truth.loc[idx])
    b = oh.absolute_percent_error(pb.loc[idx], s.truth.loc[idx])
    diff = (a - b).to_numpy()
    res.update({
        "mean_n": int(c.n_pairs), "mean_gain": -float(c.mean_diff),
        "mean_p_holm": float(c.p_holm), "mean_p_wilcoxon": float(c.p_wilcoxon),
        "mean_mde": float(oh.min_detectable_difference(float(np.std(diff, ddof=1)),
                                                       len(diff))),
        "mean_mape_cand": float(c.mape_a), "mean_mape_base": float(c.mape_b),
        "mean_family": int(v.family_size),
    })
    return res


def main() -> None:
    with open(HERE / "armC_preds.pkl", "rb") as fh:
        P = pickle.load(fh)
    pa = P["preds_all"]
    want = ["FROZEN"] + [k for k in pa if k.startswith("C_s")] \
        + [k for k in pa if k.startswith("A_s")]
    tasks = [(k, pa[k]) for k in want]
    n = min(N_WORKERS, cpu_count())
    print(f"{len(tasks)} models on {n} workers", flush=True)
    with Pool(n) as pool:
        rows = []
        for r in pool.imap_unordered(one, tasks):
            rows.append(r)
            print(f"  {r['run']:10s} sd_ratio {r['disp_sd_ratio']:.4f} "
                  f"p_swap_holm {r['disp_p_swap_holm']:.4g} "
                  f"detectable {r['disp_detectable_holm']} | "
                  f"gain {r['mean_gain']:.4f} MDE {r['mean_mde']:.4f} "
                  f"p_holm {r['mean_p_holm']:.4g}", flush=True)
    S = pd.DataFrame(rows).set_index("run").loc[want].reset_index()
    S.to_csv(HERE / "armC_significance.csv", index=False)
    print("wrote armC_significance.csv")


if __name__ == "__main__":
    main()
