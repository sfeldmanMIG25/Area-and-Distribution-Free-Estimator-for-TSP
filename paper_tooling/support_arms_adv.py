"""The two adversarial checks, plus the feature correlation, run standalone.

A1  Is the augmentation set a disguised copy of the degenerate benchmark?
    Training on degenerate instances and then reporting a win on a degenerate
    benchmark looks like leakage even when it is not, so this measures how
    close the added instances actually are to the benchmark in feature space,
    against the only honest yardstick: the benchmark's own within-family
    spacing.

A2  Does the winning arm beat the frozen model rescaled by ONE constant fitted
    on the family it is scored on? The prior audit found roughly half of MF's
    LineNoise gain was reproducible that way.
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

from support_arms_study import (  # noqa: E402
    frozen_features, load_augment, load_corpus,
)


def near_duplicate(corpus, aug, bench, feats) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    # Degenerate generators legitimately produce non-finite features (a
    # zero-width bounding box gives log_bounding_hypervolume = -inf). Those
    # columns carry no distance information here, so drop them rather than
    # imputing a value that would fabricate proximity.
    A = aug[feats].to_numpy(dtype=float)
    B = bench[feats].to_numpy(dtype=float)
    good = np.isfinite(A).all(axis=0) & np.isfinite(B).all(axis=0)
    used = [f for f, g in zip(feats, good) if g]
    dropped = [f for f, g in zip(feats, good) if not g]
    print(f"[A1] {len(used)}/{len(feats)} features usable; dropped {dropped}")

    tr = corpus[corpus.split == "train"]
    mu = tr[used].to_numpy(dtype=float).mean(axis=0)
    sd = tr[used].to_numpy(dtype=float).std(axis=0)
    sd[sd == 0] = 1.0

    Za = (aug[used].to_numpy(dtype=float) - mu) / sd
    tree_a = cKDTree(Za)
    rows = []
    for gen, g in bench.groupby("generator"):
        if len(g) < 2:
            continue
        Zb = (g[used].to_numpy(dtype=float) - mu) / sd
        d_aug, _ = tree_a.query(Zb, k=1)
        d_own, _ = cKDTree(Zb).query(Zb, k=2)
        d_own = d_own[:, 1]
        rows.append({
            "generator": gen, "n_bench": int(len(g)),
            "nn_to_augment_min": float(d_aug.min()),
            "nn_to_augment_p05": float(np.percentile(d_aug, 5)),
            "nn_to_augment_median": float(np.median(d_aug)),
            "nn_within_family_median": float(np.median(d_own)),
            "ratio_median": float(np.median(d_aug) / max(np.median(d_own), 1e-12)),
            "n_closer_to_aug_than_to_own_family": int((d_aug < d_own).sum()),
        })
    return pd.DataFrame(rows).sort_values("ratio_median")


def one_constant(PI: pd.DataFrame, cand: str, generator: str) -> dict:
    f = PI[(PI.model == "FROZEN") & (PI.stratum == "bench2d")
           & (PI.generator == generator)].set_index("instance")
    c = PI[(PI.model == cand) & (PI.stratum == "bench2d")
           & (PI.generator == generator)].set_index("instance")
    idx = f.index.intersection(c.index)
    f, c = f.loc[idx], c.loc[idx]
    true = f["true_cost"].to_numpy()
    pred = f["pred_cost"].to_numpy()

    ratio = true / pred
    grid = np.linspace(ratio.min(), ratio.max(), 8001)
    mapes = np.array([np.mean(np.abs((g * pred - true) / true)) * 100.0 for g in grid])
    k = int(np.argmin(mapes))
    best_c, recal_mape = float(grid[k]), float(mapes[k])

    frozen_mape = float(np.mean(np.abs(f["err_pct"])))
    cand_mape = float(np.mean(np.abs(c["err_pct"])))

    # Paired test: is the arm better than the recalibrated frozen model, or
    # only better than the raw one?
    ape_recal = np.abs((best_c * pred - true) / true) * 100.0
    ape_cand = np.abs(c["err_pct"].to_numpy())
    w = stats.wilcoxon(ape_cand, ape_recal, zero_method="zsplit")
    return {
        "model": cand, "generator": generator, "n": int(len(idx)),
        "frozen_mape": frozen_mape,
        "one_constant": best_c,
        "frozen_recalibrated_mape": recal_mape,
        "gain_one_constant_buys": frozen_mape - recal_mape,
        "candidate_mape": cand_mape,
        "gain_candidate_buys": frozen_mape - cand_mape,
        "candidate_minus_recalibrated": cand_mape - recal_mape,
        "frac_of_gain_explained_by_constant":
            (frozen_mape - recal_mape) / max(frozen_mape - cand_mape, 1e-12),
        "p_wilcoxon_cand_vs_recal": float(w.pvalue),
    }


def main() -> None:
    feats31 = frozen_features()
    corpus = load_corpus(feats31)
    aug = load_augment(feats31)
    PI = pd.read_csv(HERE / "support_arms_per_instance.csv")

    C = pd.read_csv(HERE / "v4_study_feature_cache.csv", low_memory=False)
    gen = pd.read_csv(HERE / "augmentation_2d_features.csv",
                      usecols=["instance_name", "generator"])
    bench = C[C.stratum == "bench2d"].merge(
        gen, left_on="instance", right_on="instance_name", how="left")
    bench = bench[bench.generator.isin(["line_noise", "grid", "clustered", "boundary"])]

    pd.set_option("display.width", 250)
    f4 = lambda x: f"{x:.4f}"  # noqa: E731

    dup = near_duplicate(corpus, aug, bench, feats31)
    dup.to_csv(HERE / "support_arms_adv_nearduplicate.csv", index=False)
    print("\n=== A1: augmentation vs benchmark distance in feature space ===")
    print("(ratio_median > 1 means the benchmark's own neighbours are closer "
          "than anything in the augmentation set)")
    print(dup.to_string(index=False, float_format=f4))

    rows = []
    for cand in ("A", "B"):
        for g in ("line_noise", "grid"):
            rows.append(one_constant(PI, cand, g))
    RC = pd.DataFrame(rows)
    RC.to_csv(HERE / "support_arms_adv_recalibration.csv", index=False)
    print("\n=== A2: one-constant recalibration control ===")
    print(RC.to_string(index=False, float_format=f4))

    sub = corpus[["greedy_nn_over_mst", "mst_topology_straightness"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    corr = {
        "n": int(len(sub)),
        "pearson": float(np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1]),
        "spearman": float(stats.spearmanr(sub.iloc[:, 0], sub.iloc[:, 1]).statistic),
    }
    for lo in (200, 500):
        s = corpus[corpus.n_customers >= lo][
            ["greedy_nn_over_mst", "mst_topology_straightness"]].dropna()
        corr[f"pearson_n_ge_{lo}"] = float(np.corrcoef(s.iloc[:, 0], s.iloc[:, 1])[0, 1])
    b = bench[bench.generator == "line_noise"].merge(
        pd.read_csv(HERE / "support_arms_feats_bench2d.csv"),
        left_on="instance", right_on="instance_name", how="left")
    corr["pearson_bench2d_line_noise"] = float(np.corrcoef(
        b["greedy_nn_over_mst"], b["mst_topology_straightness"])[0, 1])
    json.dump(corr, open(HERE / "support_arms_correlation.json", "w"), indent=2)
    print("\n=== greedy_nn_over_mst vs mst_topology_straightness ===")
    print(json.dumps(corr, indent=2))


if __name__ == "__main__":
    main()
