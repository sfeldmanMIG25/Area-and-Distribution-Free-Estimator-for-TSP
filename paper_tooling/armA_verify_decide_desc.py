"""Deciding-reviewer descriptive checks: how the ND gain and the 2D gain are
actually distributed. No refits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import armA_verify_common as K  # noqa: E402


def main() -> None:
    D = K.load_cache()
    C, feats = D["cache"], D["feats31"]
    fr = joblib.load(K.FROZEN)
    a = joblib.load(HERE / "armA_verify_models" / "A_full.joblib")

    f_nd = K.score_frame(fr, feats, C[C.stratum == "nd_test"])
    a_nd = K.score_frame(a, feats, C[C.stratum == "nd_test"])
    j = f_nd[["instance", "err_pct"]].merge(
        a_nd[["instance", "err_pct"]], on="instance", suffixes=("_f", "_a"))
    j["ape_f"] = j.err_pct_f.abs()
    j["ape_a"] = j.err_pct_a.abs()
    j["d"] = j.ape_a - j.ape_f
    print(f"ND N={len(j)}  frozen MAPE {j.ape_f.mean():.4f}  A MAPE {j.ape_a.mean():.4f}")
    print(f"  mean paired diff {j.d.mean():+.5f} pp | median {j.d.median():+.5f} pp")
    print(f"  worse: {(j.d > 0).mean()*100:.2f}%   better: {(j.d < 0).mean()*100:.2f}%")
    print(f"  Cohen dz {j.d.mean()/j.d.std(ddof=1):+.4f}")
    j["dec"] = pd.qcut(j.ape_f, 10, labels=False, duplicates="drop")
    g = j.groupby("dec").agg(n=("d", "size"), frozen_mape=("ape_f", "mean"),
                             a_mape=("ape_a", "mean"), sum_d=("d", "sum"),
                             pct_worse=("d", lambda s: (s > 0).mean() * 100))
    g["share_of_total_gain_pct"] = g.sum_d / j.d.sum() * 100
    print(g.round(4).to_string())

    b2f = K.score_frame(fr, feats, C[C.stratum == "bench2d"])
    b2a = K.score_frame(a, feats, C[C.stratum == "bench2d"])
    m = b2f[["instance", "generator", "err_pct"]].merge(
        b2a[["instance", "err_pct"]], on="instance", suffixes=("_f", "_a"))
    m["gain"] = m.err_pct_f.abs() - m.err_pct_a.abs()
    tot = m.gain.sum()
    dec = (m.groupby("generator")
             .agg(n=("gain", "size"), gain_sum=("gain", "sum"),
                  mape_f=("err_pct_f", lambda s: s.abs().mean()),
                  mape_a=("err_pct_a", lambda s: s.abs().mean()))
             .sort_values("gain_sum", ascending=False))
    dec["share_pct"] = dec.gain_sum / tot * 100
    print("\n2D gain decomposition by generator (total 2D MAPE gain "
          f"{tot/len(m):.4f} pp over N={len(m)})")
    print(dec.round(4).to_string())
    other = m[~m.generator.isin(["grid", "line_noise"])]
    print(f"\nexcluding grid+line_noise: N={len(other)} frozen MAPE "
          f"{other.err_pct_f.abs().mean():.4f} -> A {other.err_pct_a.abs().mean():.4f} "
          f"(delta {other.err_pct_a.abs().mean()-other.err_pct_f.abs().mean():+.4f})")


if __name__ == "__main__":
    main()
