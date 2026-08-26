"""Monotone-consistency audit of SDPE across size and dimension.

Tests the paper's central claim -- that GART 2.0 is *consistent*, not merely
accurate on average -- by asking whether the standard deviation of SIGNED
percent error (SDPE) falls monotonically as instances get larger and as the
ambient dimension rises, and whether that holds for the baselines too.

Signed percent error:   e_i = 100 * (pred_i - true_i) / true_i
SDPE:                   std(e, ddof=1)

Everything is computed from existing benchmark output; nothing is re-run.

Outputs (paper_tooling/monotonicity_out/):
    sdpe_buckets.csv        per stratum x axis x model x bucket
    monotonicity.csv        headline monotonicity / trend table
    head_to_head.csv        candidate-vs-baseline SDPE ratios per bucket
    head_to_head_trend.csv  whether that ratio improves / degrades with bucket
    ranking.csv             consistency ranking vs MAPE ranking
    validation.csv          reproduction of the published reference values
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper_tooling", "monotonicity_out")
sys.path.insert(0, os.path.join(ROOT, "tsplib_benchmark"))
from exclusions import filter_metric_consistent  # noqa: E402

B_BOOT = 2000
B_TREND = 1000
MIN_BUCKET = 15
CANDIDATES = ("LGBM_V3", "LGBM_V4")

ND_SIZE = [(5, 10), (11, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]
TWOD_SIZE = [(5, 10), (11, 50), (51, 100), (101, 500), (501, 1000)]
TSPLIB_SIZE = [(51, 150), (151, 400), (401, 10**9)]
AUG_SIZE = [(50, 100), (101, 400), (401, 1000)]
ND_DIM = [(2, 2), (3, 5), (6, 10), (15, 25), (30, 50), (100, 100)]
AUG_DIM = [(2, 2), (3, 5), (10, 10), (20, 50)]
EXTRAPOLATION_BUCKETS = {("ND", "dim", "d=100")}


def blabel(lo: int, hi: int, prefix: str) -> str:
    if hi >= 10**9:
        return f"{prefix}>{lo - 1}"
    if lo == hi:
        return f"{prefix}={lo}"
    return f"{prefix}{lo}-{hi}"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


@dataclass
class Stratum:
    name: str
    long: pd.DataFrame  # model, instance, e, n, d, alpha
    meta: pd.DataFrame  # instance, n, d, alpha  (one row per instance)
    axes: dict[str, list[tuple[int, int]]]


def _signed(df: pd.DataFrame) -> pd.Series:
    return 100.0 * (df["pred_cost"] - df["true_cost"]) / df["true_cost"]


def _add_repaired(base: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    """Append repaired classical estimators as ``<model>[rep]`` when they differ."""
    frames = [base]
    for model, sub in rep.groupby("model"):
        b = base[base.model == model]
        if b.empty:
            frames.append(sub.assign(model=f"{model}[rep]"))
            continue
        j = b.set_index("instance")["pred_cost"].align(
            sub.set_index("instance")["pred_cost"], join="inner"
        )
        if len(j[0]) and float(np.nanmax(np.abs(j[0] - j[1]))) > 1e-9:
            frames.append(sub.assign(model=f"{model}[rep]"))
    return pd.concat(frames, ignore_index=True)


def load_nd() -> Stratum:
    d = pd.read_csv(
        os.path.join(ROOT, "Generalized_TSP_Analysis_ND", "benchmark_results_ND_final.csv"),
        low_memory=False,
    )
    r = pd.read_csv(
        os.path.join(ROOT, "Generalized_TSP_Analysis_ND", "benchmark_results_ND_repaired.csv"),
        low_memory=False,
    )
    g = pd.read_csv(
        os.path.join(
            ROOT, "Generalized_TSP_Analysis_ND", "benchmark_checkpoints", "base_ground_truth_nd.csv"
        )
    )
    d = d[d.status == "ok"]
    r = r[r.status == "ok"]
    cols = ["model", "instance", "pred_cost", "true_cost"]
    all_rows = _add_repaired(d[cols], r[cols])
    meta = g[["instance", "n_customers", "dimension", "true_cost", "mst_length"]].rename(
        columns={"n_customers": "n", "dimension": "d"}
    )
    meta["alpha"] = meta.true_cost / meta.mst_length
    meta = meta[["instance", "n", "d", "alpha"]]
    long = all_rows.merge(meta, on="instance", how="inner")
    long["e"] = _signed(long)
    return Stratum("ND", long, meta, {"size": ND_SIZE, "dim": ND_DIM})


def load_2d() -> Stratum:
    d = pd.read_csv(
        os.path.join(ROOT, "Generalized_TSP_Analysis", "benchmark_results_2D_v3.csv"),
        low_memory=False,
    )
    r = pd.read_csv(
        os.path.join(ROOT, "Generalized_TSP_Analysis", "benchmark_results_2D_repaired.csv"),
        low_memory=False,
    )
    g = pd.read_csv(
        os.path.join(
            ROOT, "Generalized_TSP_Analysis", "benchmark_checkpoints", "base_ground_truth_2d.csv"
        )
    )
    d = d[d.status == "ok"]
    r = r[r.status == "ok"] if "status" in r else r
    cols = ["model", "instance", "pred_cost", "true_cost"]
    all_rows = _add_repaired(d[cols], r[cols])
    # generator class from  TSP-<generator>-n<n>-g<G>-  (clustered adds -c<k>-r<r>)
    pat = re.compile(r"^TSP-(.+?)-n(\d+)-g(\d+)-")
    gen = g.instance.map(lambda s: (pat.match(s).group(1) if pat.match(s) else "UNMATCHED"))
    meta = g[["instance", "n_customers", "dimension", "true_alpha"]].rename(
        columns={"n_customers": "n", "dimension": "d", "true_alpha": "alpha"}
    )
    meta["generator"] = gen.values
    long = all_rows.merge(meta, on="instance", how="inner")
    long["e"] = _signed(long)
    return Stratum("2D", long, meta, {"size": TWOD_SIZE})


def load_tsplib() -> Stratum:
    p = os.path.join(ROOT, "tsplib_benchmark", "results", "all_models_tsplib.csv")
    pr = os.path.join(ROOT, "tsplib_benchmark", "results", "all_models_tsplib_repaired.csv")
    d = pd.read_csv(p)
    r = pd.read_csv(pr)
    d = d[d.edge_weight_type == "EUC_2D"]
    r = r[r.edge_weight_type == "EUC_2D"] if "edge_weight_type" in r else r
    inst = (
        d.drop_duplicates("instance")[["instance", "n", "true_cost", "mst_length"]]
        .pipe(filter_metric_consistent)
        .copy()
    )
    inst["alpha"] = inst.true_cost / inst.mst_length
    meta = inst[["instance", "n", "alpha"]].assign(d=2)
    keep = set(meta.instance)
    d = d[(d.status == "ok") & d.instance.isin(keep)]
    r = r[(r.status == "ok") & r.instance.isin(keep)] if "status" in r else r[r.instance.isin(keep)]
    cols = ["model", "instance", "pred_cost", "true_cost"]
    all_rows = _add_repaired(d[cols], r[cols])
    long = all_rows.merge(meta, on="instance", how="inner")
    long["e"] = _signed(long)
    return Stratum("TSPLIB", long, meta, {"size": TSPLIB_SIZE})


def load_aug() -> Stratum:
    f = pd.read_csv(os.path.join(ROOT, "paper_tooling", "augment_features_v3.csv"))
    b = pd.read_csv(os.path.join(ROOT, "paper_tooling", "ood_augment_baselines.csv"))
    meta = f[["instance_name", "n_customers", "dimension", "optimal_cost", "mst_total_length"]].rename(
        columns={"instance_name": "instance", "n_customers": "n", "dimension": "d"}
    )
    meta["alpha"] = meta.optimal_cost / meta.mst_total_length
    long = b.melt(id_vars="instance", var_name="model", value_name="pred_cost").dropna(
        subset=["pred_cost"]
    )
    long = long.merge(meta, on="instance", how="inner")
    long = long.rename(columns={"optimal_cost": "true_cost"})
    long["e"] = _signed(long)
    return Stratum(
        "AUG",
        long[["model", "instance", "pred_cost", "true_cost", "n", "d", "alpha", "e"]],
        meta[["instance", "n", "d", "alpha"]],
        {"size": AUG_SIZE, "dim": AUG_DIM},
    )


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def boot_sdpe_ci(e: np.ndarray, seed: int = 0) -> tuple[float, float]:
    n = e.size
    if n < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(B_BOOT)
    step = max(1, int(4e6 // max(n, 1)))
    for s in range(0, B_BOOT, step):
        k = min(step, B_BOOT - s)
        vals[s : s + k] = e[rng.integers(0, n, size=(k, n))].std(axis=1, ddof=1)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def _rowwise_spearman(mat: np.ndarray) -> np.ndarray:
    """Spearman rho of each row of ``mat`` against 0..K-1."""
    b, k = mat.shape
    order = np.argsort(mat, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    np.put_along_axis(ranks, order, np.tile(np.arange(k, dtype=float), (b, 1)), axis=1)
    x = np.arange(k, dtype=float)
    xc = x - x.mean()
    rc = ranks - ranks.mean(axis=1, keepdims=True)
    denom = np.sqrt((rc**2).sum(axis=1) * (xc**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rc @ xc) / denom


def boot_spearman_ci(groups: list[np.ndarray], seed: int = 1) -> tuple[float, float]:
    k = len(groups)
    if k < 3 or any(g.size < 5 for g in groups):
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    mat = np.empty((B_TREND, k))
    for j, g in enumerate(groups):
        n = g.size
        step = max(1, int(4e6 // max(n, 1)))
        for s in range(0, B_TREND, step):
            m = min(step, B_TREND - s)
            mat[s : s + m, j] = g[rng.integers(0, n, size=(m, n))].std(axis=1, ddof=1)
    rho = _rowwise_spearman(mat)
    return tuple(np.nanpercentile(rho, [2.5, 97.5]))


def jonckheere(groups: list[np.ndarray]) -> tuple[float, float]:
    """Tie-corrected Jonckheere-Terpstra z and one-sided p for a DECREASING trend."""
    groups = [g for g in groups if g.size > 0]
    k = len(groups)
    if k < 3:
        return (np.nan, np.nan)
    ns = np.array([g.size for g in groups], dtype=float)
    N = ns.sum()
    jt = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            u = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided").statistic
            jt += ns[i] * ns[j] - u  # count of (x_i < x_j) + 0.5 ties
    ev = (N**2 - (ns**2).sum()) / 4.0
    allx = np.concatenate(groups)
    _, tc = np.unique(allx, return_counts=True)
    t = tc.astype(float)
    v1 = (
        N * (N - 1) * (2 * N + 5)
        - (ns * (ns - 1) * (2 * ns + 5)).sum()
        - (t * (t - 1) * (2 * t + 5)).sum()
    ) / 72.0
    v2 = ((ns * (ns - 1) * (ns - 2)).sum() * (t * (t - 1) * (t - 2)).sum()) / (
        36.0 * N * (N - 1) * (N - 2)
    )
    v3 = ((ns * (ns - 1)).sum() * (t * (t - 1)).sum()) / (8.0 * N * (N - 1))
    var = v1 + v2 + v3
    if var <= 0:
        return (np.nan, np.nan)
    z = (jt - ev) / np.sqrt(var)
    return (float(z), float(stats.norm.cdf(z)))  # small JT == decreasing


def hc3_loglog(n: np.ndarray, e: np.ndarray) -> tuple[float, float, float, int]:
    """OLS of log|e| on log n with HC3 robust SE. Returns slope, se, p, n_used."""
    a = np.abs(e)
    m = (a > 0) & (n > 0) & np.isfinite(a) & np.isfinite(n)
    if m.sum() < 20:
        return (np.nan, np.nan, np.nan, int(m.sum()))
    y = np.log(a[m])
    X = np.column_stack([np.ones(m.sum()), np.log(n[m].astype(float))])
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    r = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    w = (r / np.clip(1 - h, 1e-12, None)) ** 2
    V = xtx_inv @ (X.T * w) @ X @ xtx_inv
    se = float(np.sqrt(V[1, 1]))
    z = beta[1] / se if se > 0 else np.nan
    return (float(beta[1]), se, float(2 * stats.norm.sf(abs(z))), int(m.sum()))


def loglog_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Slope of log(SDPE) on log(x) across bucket summaries."""
    m = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)[0])


# --------------------------------------------------------------------------
# per-bucket table
# --------------------------------------------------------------------------


def bucket_column(v: pd.Series, bins: list[tuple[int, int]], prefix: str) -> pd.Series:
    out = pd.Series(index=v.index, dtype=object)
    for lo, hi in bins:
        out[(v >= lo) & (v <= hi)] = blabel(lo, hi, prefix)
    return out


def build_bucket_table(strata: list[Stratum]) -> pd.DataFrame:
    rows = []
    for st in strata:
        for axis, bins in st.axes.items():
            col, prefix = ("n", "n") if axis == "size" else ("d", "d")
            labels = [blabel(lo, hi, prefix) for lo, hi in bins]
            meta = st.meta.copy()
            meta["bucket"] = bucket_column(meta[col], bins, prefix)
            tot = meta.groupby("bucket").instance.nunique().to_dict()
            adisp = meta.groupby("bucket").alpha.agg(["std", "mean"])
            gmean_x = meta.groupby("bucket")[col].apply(lambda s: float(np.exp(np.log(s).mean())))
            long = st.long.copy()
            long["bucket"] = bucket_column(long[col], bins, prefix)
            for (model, bkt), sub in long.groupby(["model", "bucket"]):
                if bkt not in labels or len(sub) < MIN_BUCKET:
                    continue
                e = sub.e.to_numpy(float)
                e = e[np.isfinite(e)]
                if e.size < MIN_BUCKET:
                    continue
                lo, hi = boot_sdpe_ci(e, seed=abs(hash((st.name, axis, model, bkt))) % 2**31)
                sd_a = float(adisp.loc[bkt, "std"])
                mu_a = float(adisp.loc[bkt, "mean"])
                cv_a = 100.0 * sd_a / mu_a if mu_a else np.nan
                sdpe = float(e.std(ddof=1))
                rows.append(
                    dict(
                        stratum=st.name,
                        axis=axis,
                        model=model,
                        bucket=bkt,
                        bucket_index=labels.index(bkt),
                        x_gmean=float(gmean_x.get(bkt, np.nan)),
                        n_instances=int(e.size),
                        n_bucket_total=int(tot.get(bkt, 0)),
                        coverage=float(e.size / tot[bkt]) if tot.get(bkt) else np.nan,
                        sdpe=sdpe,
                        sdpe_lo=lo,
                        sdpe_hi=hi,
                        mape=float(np.abs(e).mean()),
                        bias=float(e.mean()),
                        sd_alpha=sd_a,
                        cv_alpha_pct=cv_a,
                        skill_vs_sd_alpha=sdpe / sd_a if sd_a else np.nan,
                        skill_vs_cv_alpha=sdpe / cv_a if cv_a else np.nan,
                        extrapolation=(st.name, axis, bkt) in EXTRAPOLATION_BUCKETS,
                    )
                )
    return pd.DataFrame(rows).sort_values(["stratum", "axis", "model", "bucket_index"])


def build_monotonicity(strata: list[Stratum], bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in strata:
        for axis, bins in st.axes.items():
            col, prefix = ("n", "n") if axis == "size" else ("d", "d")
            labels = [blabel(lo, hi, prefix) for lo, hi in bins]
            long = st.long.copy()
            long["bucket"] = bucket_column(long[col], bins, prefix)
            for model, sub_all in long.groupby("model"):
                t = bt[(bt.stratum == st.name) & (bt.axis == axis) & (bt.model == model)]
                t = t.sort_values("bucket_index")
                # in-sample buckets only for the trend tests
                t_in = t[~t.extrapolation]
                if len(t_in) < 3:
                    continue
                seq = t_in.sdpe.to_numpy(float)
                dec = int((np.diff(seq) < 0).sum())
                pairs = len(seq) - 1
                groups, groups_abs = [], []
                for bkt in t_in.bucket:
                    e = sub_all[sub_all.bucket == bkt].e.to_numpy(float)
                    e = e[np.isfinite(e)]
                    groups.append(e)
                    groups_abs.append(np.abs(e))
                rho = float(stats.spearmanr(t_in.bucket_index, seq).statistic)
                rlo, rhi = boot_spearman_ci(
                    groups, seed=abs(hash((st.name, axis, model))) % 2**31
                )
                jz, jp = jonckheere(groups_abs)
                sub_in = sub_all[sub_all.bucket.isin(list(t_in.bucket))]
                slope, se, p, nused = hc3_loglog(
                    sub_in[col].to_numpy(float), sub_in.e.to_numpy(float)
                )
                sk = t_in.skill_vs_cv_alpha.to_numpy(float)
                rows.append(
                    dict(
                        stratum=st.name,
                        axis=axis,
                        model=model,
                        n_buckets=len(seq),
                        n_decreasing_pairs=dec,
                        n_pairs=pairs,
                        strict_monotone=bool(dec == pairs),
                        sdpe_first=seq[0],
                        sdpe_last=seq[-1],
                        sdpe_ratio_last_first=float(seq[-1] / seq[0]) if seq[0] else np.nan,
                        spearman=rho,
                        spearman_lo=rlo,
                        spearman_hi=rhi,
                        jt_z=jz,
                        jt_p_decreasing=jp,
                        loglog_slope_bucket=loglog_slope(t_in.x_gmean.to_numpy(float), seq),
                        hc3_slope_logabse=slope,
                        hc3_se=se,
                        hc3_p=p,
                        hc3_n=nused,
                        skill_spearman=(
                            float(stats.spearmanr(np.arange(sk.size), sk).statistic)
                            if sk.size >= 3 and np.all(np.isfinite(sk))
                            else np.nan
                        ),
                        skill_dec_pairs=int((np.diff(sk) < 0).sum()),
                        skill_strict_monotone=bool(
                            np.all(np.diff(sk) < 0) if np.all(np.isfinite(sk)) else False
                        ),
                        skill_first=sk[0] if sk.size else np.nan,
                        skill_last=sk[-1] if sk.size else np.nan,
                        mape_overall=float(
                            np.abs(sub_in.e.to_numpy(float)).mean()
                        ),
                        min_coverage=float(t_in.coverage.min()),
                    )
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# head to head
# --------------------------------------------------------------------------


def boot_ratio_ci(ec: np.ndarray, eb: np.ndarray, seed: int) -> tuple[float, float]:
    n = ec.size
    if n < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(B_BOOT)
    step = max(1, int(2e6 // max(n, 1)))
    for s in range(0, B_BOOT, step):
        k = min(step, B_BOOT - s)
        idx = rng.integers(0, n, size=(k, n))  # paired resample
        vals[s : s + k] = ec[idx].std(axis=1, ddof=1) / eb[idx].std(axis=1, ddof=1)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def build_head_to_head(strata: list[Stratum]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, trend = [], []
    for st in strata:
        for axis, bins in st.axes.items():
            col, prefix = ("n", "n") if axis == "size" else ("d", "d")
            labels = [blabel(lo, hi, prefix) for lo, hi in bins]
            long = st.long.copy()
            long["bucket"] = bucket_column(long[col], bins, prefix)
            wide = {m: s.set_index("instance").e for m, s in long.groupby("model")}
            bkt_of = long.drop_duplicates("instance").set_index("instance").bucket
            for cand in CANDIDATES:
                if cand not in wide:
                    continue
                for base in sorted(wide):
                    if base == cand:
                        continue
                    a, b = wide[cand].align(wide[base], join="inner")
                    if a.size < MIN_BUCKET:
                        continue
                    bk = bkt_of.reindex(a.index)
                    per = []
                    for bkt in labels:
                        m = (bk == bkt).to_numpy()
                        if m.sum() < MIN_BUCKET:
                            continue
                        ec, eb = a.to_numpy()[m], b.to_numpy()[m]
                        sc, sb = ec.std(ddof=1), eb.std(ddof=1)
                        if not np.isfinite(sb) or sb == 0:
                            continue
                        lo, hi = boot_ratio_ci(
                            ec, eb, abs(hash((st.name, axis, cand, base, bkt))) % 2**31
                        )
                        rows.append(
                            dict(
                                stratum=st.name,
                                axis=axis,
                                candidate=cand,
                                baseline=base,
                                bucket=bkt,
                                bucket_index=labels.index(bkt),
                                n_paired=int(m.sum()),
                                sdpe_candidate=float(sc),
                                sdpe_baseline=float(sb),
                                ratio=float(sc / sb),
                                ratio_lo=lo,
                                ratio_hi=hi,
                                candidate_better=bool(hi < 1.0),
                            )
                        )
                        per.append((labels.index(bkt), float(sc / sb)))
                    if len(per) >= 3:
                        idx = np.array([p[0] for p in per], float)
                        rat = np.array([p[1] for p in per], float)
                        rho = float(stats.spearmanr(idx, rat).statistic)
                        slope = float(np.polyfit(idx, np.log(rat), 1)[0])
                        verdict = (
                            "advantage grows"
                            if rho <= -0.6
                            else "advantage shrinks"
                            if rho >= 0.6
                            else "stable"
                        )
                        trend.append(
                            dict(
                                stratum=st.name,
                                axis=axis,
                                candidate=cand,
                                baseline=base,
                                n_buckets=len(per),
                                ratio_first=rat[0],
                                ratio_last=rat[-1],
                                spearman_ratio_vs_bucket=rho,
                                slope_log_ratio_per_bucket=slope,
                                verdict=verdict,
                            )
                        )
    return pd.DataFrame(rows), pd.DataFrame(trend)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

REFERENCE = {
    ("ND", "size"): [1.98, 1.35, 1.03, 0.61, 0.55, 0.55],
    ("ND", "dim"): [2.97, 2.03, 1.28, 0.85, 0.61, 0.65],
    ("2D", "size"): [5.88, 6.29, 5.91, 4.48, 3.05],
    ("TSPLIB", "size"): [3.29, 3.46, 3.42],
}


def validate(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stratum, axis), ref in REFERENCE.items():
        t = bt[
            (bt.stratum == stratum) & (bt.axis == axis) & (bt.model == "LGBM_V3")
        ].sort_values("bucket_index")
        got = t.sdpe.round(2).tolist()
        for i, r in enumerate(ref):
            g = got[i] if i < len(got) else np.nan
            rows.append(
                dict(
                    stratum=stratum,
                    axis=axis,
                    bucket=t.bucket.tolist()[i] if i < len(t) else "",
                    published=r,
                    reproduced=g,
                    match=bool(np.isclose(r, g, atol=0.005)),
                )
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def build_ranking(mono: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stratum, axis), sub in mono.groupby(["stratum", "axis"]):
        s = sub.copy()
        s["mono_frac"] = s.n_decreasing_pairs / s.n_pairs
        s = s.sort_values(
            ["mono_frac", "loglog_slope_bucket", "sdpe_last"], ascending=[False, True, True]
        )
        s["rank_consistency"] = range(1, len(s) + 1)
        s["rank_mape"] = s.mape_overall.rank(method="min").astype(int)
        rows.append(
            s[
                [
                    "stratum",
                    "axis",
                    "model",
                    "rank_consistency",
                    "rank_mape",
                    "mono_frac",
                    "n_decreasing_pairs",
                    "n_pairs",
                    "loglog_slope_bucket",
                    "hc3_slope_logabse",
                    "sdpe_last",
                    "mape_overall",
                    "skill_dec_pairs",
                    "min_coverage",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    strata = [load_nd(), load_2d(), load_tsplib(), load_aug()]
    for st in strata:
        print(
            f"[load] {st.name}: {st.long.model.nunique()} models, "
            f"{st.meta.instance.nunique()} instances, {len(st.long)} rows"
        )
    bt = build_bucket_table(strata)
    bt.to_csv(os.path.join(OUT, "sdpe_buckets.csv"), index=False)
    print(f"[out ] sdpe_buckets.csv  {len(bt)} rows")

    val = validate(bt)
    val.to_csv(os.path.join(OUT, "validation.csv"), index=False)
    bad = val[~val.match]
    print(f"[val ] {val.match.sum()}/{len(val)} published values reproduced")
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit("VALIDATION FAILED -- stopping before reporting")

    mono = build_monotonicity(strata, bt)
    mono.to_csv(os.path.join(OUT, "monotonicity.csv"), index=False)
    print(f"[out ] monotonicity.csv  {len(mono)} rows")

    h2h, trend = build_head_to_head(strata)
    h2h.to_csv(os.path.join(OUT, "head_to_head.csv"), index=False)
    trend.to_csv(os.path.join(OUT, "head_to_head_trend.csv"), index=False)
    print(f"[out ] head_to_head.csv  {len(h2h)} rows / trend {len(trend)} rows")

    rank = build_ranking(mono, bt)
    rank.to_csv(os.path.join(OUT, "ranking.csv"), index=False)
    print(f"[out ] ranking.csv  {len(rank)} rows")


if __name__ == "__main__":
    main()
