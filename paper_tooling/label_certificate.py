"""Use the Held--Karp 1-tree bound as a certificate on every released label.

WHY THIS EXISTS
---------------
``hk1tree_validate.py invariant`` already compares the bound against the stored
``true_cost`` and escalates every excess to a recheck against the stored optimal
tour recomputed in float64.  That escalation is **vacuous whenever the stored
tour is itself wrong**: a tour 291% longer than the optimum passes ``bound <=
tour`` trivially, so a broken label hides behind a broken tour.  Exactly that
happened -- ``hk1tree_validate`` reported "0 REAL violations" on a corpus that
contains 35 labels below a proven lower bound.

This module replaces the tour-based escalation with a test that consults no tour
at all.

THE TEST
--------
The released label ``L`` claims to be the optimum in an integer-quantised
metric with quantum ``q`` (edge weights are integer multiples of ``q``).  For
any tour ``T``,  ``|L_q(T) - L_f(T)| <= n q / 2``.  Writing ``T*`` for the
q-optimal tour and ``OPT_f`` for the float64 optimum,

    OPT_f  <=  L_f(T*)  <=  L_q(T*) + n q / 2  =  L + n q / 2 ,

and the 1-tree bound satisfies ``B <= OPT_f``.  Therefore

    B  >  L + n q / 2        =>   L is NOT the q-metric optimum of the released
                                  coordinates.

No stored tour enters, so a corrupt tour cannot make the test pass, and the
margin ``B - L - n q / 2`` is the amount by which the label is provably wrong.

WHERE q COMES FROM
------------------
2D benchmark
    ``q = 1``.  Coordinates are integers and the label is ``sum nint(euclidean)``
    -- verified exactly: the stored tour's nint length equals the released label
    on all 2,580 instances.
ND corpus
    ``label = solver_integer_length / scale``.  LKH was handed
    ``solvers.config.get_scale_factor(grid)``; Concorde was handed
    ``get_robust_scale_factor(grid, n)``, the same base damped by
    ``min(1, n/500)``, which is up to 100x coarser below n=500.  So
    ``q = 1 / scale``.  The routine verifies the label is an exact multiple of
    ``q`` before using it, and falls back to the coarser of the two candidate
    quanta when it is not, so an unrecognised provenance can only make the test
    more conservative.
TSPLIB
    No slack at all.  ``hk1tree_tsplib.py`` builds the bound on the instance's
    own integer distance matrix, so the bound and the published optimum live in
    one metric and ``B > L`` is already a contradiction.

    python paper_tooling/label_certificate.py
    python paper_tooling/label_certificate.py --require-trainval   # fail if the
                                                                   # train/val
                                                                   # sweep is absent
Outputs: paper_tooling/label_certificate.csv       one row per refuted label
         paper_tooling/label_certificate_flagged.csv   every bound above a label
         paper_tooling/label_certificate.json      summary + coverage ledger
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "paper_tooling"

#: k=1000 exact-ascent bound on the two scored corpora (hk1tree_validate.py).
P_INVARIANT = OUT / "hk1tree_invariant_k1000.csv"
#: k=2000 Polyak ascent over the ND test split (hk1tree_polyak_nd.py). Strictly
#: stronger than the k=1000 exact ascent on this corpus, so it is preferred when
#: present; the certificate is only ever tightened by a better bound.
P_POLYAK = OUT / "polyak_nd_sweep.csv"
#: k=2000 ladder over the 2D benchmark (hk1tree_frontier_accuracy.py).
P_FRONTIER_2D = OUT / "hk1tree_frontier_2d.csv"
#: k=1000 exact ascent over the ND train and val splits (hk1tree_trainval.py).
P_TRAINVAL = OUT / "hk1tree_trainval_k1000.csv"
#: TSPLIB, in TSPLIB's own integer metric (hk1tree_tsplib.py).
P_TSPLIB = OUT / "hk1tree_tsplib.csv"
#: Split assignment, released label and the tour-consistency verdict.
P_AUDIT = OUT / "reference_tour_audit.csv"
#: Which solver produced each ND test label.
P_PROV = OUT / "nd_label_provenance.csv"
P_ND_GT = ROOT / "Generalized_TSP_Analysis_ND" / "benchmark_checkpoints" / "base_ground_truth_nd.csv"
P_2D_GT = ROOT / "Generalized_TSP_Analysis" / "benchmark_checkpoints" / "base_ground_truth_2d.csv"

#: Coordinates are stored as float32; the bound reads them back in float64. The
#: resulting relative wobble is ~1e-7, six orders below the smallest quantum in
#: play. This tolerance exists so the test can never fire on it.
REL_EPS = 1e-6


# ---------------------------------------------------------------------------
# Label quantum
# ---------------------------------------------------------------------------
def _base_scale(grid: np.ndarray) -> np.ndarray:
    """``solvers.config.get_scale_factor``, vectorised."""
    return np.where(grid <= 100, 100.0, np.where(grid <= 1000, 10.0, 1.0))


def nd_quantum(grid: np.ndarray, n: np.ndarray, solver: np.ndarray,
               label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(q, verified)`` for ND labels.

    ``verified`` marks the rows where the released label really is an integer
    multiple of the quantum the provenance implies. Unverified rows fall back to
    the coarser candidate, which can only widen the slack.
    """
    base = _base_scale(grid)
    robust = base * np.minimum(1.0, n / 500.0)
    scale = np.where(solver == "concorde", robust, base)
    q = 1.0 / scale
    ratio = label / q
    verified = np.abs(ratio - np.rint(ratio)) < 1e-6 * np.maximum(1.0, np.abs(ratio))
    q_coarse = 1.0 / np.minimum(base, robust)
    return np.where(verified, q, q_coarse), verified


def apply(df: pd.DataFrame) -> pd.DataFrame:
    """Add slack / excess / margin / proven_wrong. Needs bound, label, n, q."""
    d = df.copy()
    d["slack"] = d["n"] * d["q"] / 2.0
    d["excess"] = d["bound"] - d["label"]
    d["excess_pct"] = 100.0 * d["excess"] / d["label"]
    d["margin"] = d["excess"] - d["slack"]
    d["proven_wrong"] = d["margin"] > REL_EPS * d["label"].abs()
    return d


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def _strongest(*series: pd.Series) -> pd.Series:
    out = None
    for s in series:
        out = s if out is None else out.combine(s, lambda a, b: np.nanmax([a, b]))
    return out


def corpus_2d() -> pd.DataFrame:
    inv = pd.read_csv(P_INVARIANT)
    inv = inv[(inv.corpus == "2D") & (inv.status == "ok")].set_index("instance")
    bound = inv["bound"]
    if P_FRONTIER_2D.exists():
        f = pd.read_csv(P_FRONTIER_2D, usecols=["instance", "k", "bound", "status"])
        f = f[f.status == "ok"].groupby("instance")["bound"].max()
        bound = _strongest(bound, f.reindex(bound.index))
    d = pd.DataFrame({"n": inv["n"], "d": inv["d"], "label": inv["true_cost"],
                      "bound": bound})
    d["q"] = 1.0
    d["q_verified"] = True
    d["corpus"] = "2D benchmark"
    d["split"] = "benchmark_2d"
    d["scored"] = True
    return apply(d.reset_index())


def corpus_nd_test() -> pd.DataFrame:
    inv = pd.read_csv(P_INVARIANT)
    inv = inv[(inv.corpus == "ND") & (inv.status == "ok")].set_index("instance")
    bound = inv["bound"]
    if P_POLYAK.exists():
        p = pd.read_csv(P_POLYAK, usecols=["instance", "k", "bound", "status"])
        p = p[p.status == "ok"].groupby("instance")["bound"].max()
        bound = _strongest(bound, p.reindex(bound.index))
    gt = pd.read_csv(P_ND_GT).set_index("instance")["grid_size"]
    solver = pd.read_csv(P_PROV).set_index("instance")["solver"]
    d = pd.DataFrame({"n": inv["n"], "d": inv["d"], "label": inv["true_cost"],
                      "bound": bound, "grid_size": gt.reindex(inv.index),
                      "solver": solver.reindex(inv.index)})
    d["q"], d["q_verified"] = nd_quantum(d.grid_size.to_numpy(float), d.n.to_numpy(float),
                                         d.solver.to_numpy(object), d.label.to_numpy(float))
    d["corpus"] = "ND"
    d["split"] = "test"
    d["scored"] = True
    return apply(d.reset_index())


def corpus_nd_trainval() -> pd.DataFrame | None:
    if not P_TRAINVAL.exists():
        return None
    t = pd.read_csv(P_TRAINVAL)
    t = t[t.status == "ok"].copy()
    solver = t["solver"] if "solver" in t.columns else pd.Series(index=t.index, dtype=object)
    # Only rows the sweep escalated carry a solver. Everything else is below its
    # label by a wide margin and cannot be refuted at any quantum, so the coarsest
    # candidate is the safe default there.
    solver = solver.fillna("concorde")
    d = pd.DataFrame({"instance": t["instance"], "n": t["n"], "d": t["d"],
                      "label": t["true_cost"], "bound": t["bound"],
                      "grid_size": t["grid_size"], "solver": solver,
                      "split": t["split"]})
    d["q"], d["q_verified"] = nd_quantum(d.grid_size.to_numpy(float), d.n.to_numpy(float),
                                         d.solver.to_numpy(object), d.label.to_numpy(float))
    d["corpus"] = "ND"
    d["scored"] = False
    return apply(d)


def corpus_tsplib() -> pd.DataFrame:
    h = pd.read_csv(P_TSPLIB)
    ok = h[(h.status == "ok") & h.optimum.notna()].copy()
    d = pd.DataFrame({"instance": ok["instance"], "n": ok["n"], "d": 2,
                      "label": ok["optimum"], "bound": ok["bound"]})
    # The bound is built on the instance's own integer matrix, so the two live in
    # one metric and there is nothing to allow for.
    d["q"] = 0.0
    d["q_verified"] = True
    d["corpus"] = "TSPLIB"
    d["split"] = "tsplib"
    d["scored"] = True
    d["edge_weight_type"] = ok["edge_weight_type"].to_numpy()
    return apply(d)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require-trainval", action="store_true",
                    help="exit non-zero when the train/val sweep is missing")
    args = ap.parse_args()

    parts = [corpus_2d(), corpus_nd_test(), corpus_tsplib()]
    tv = corpus_nd_trainval()
    if tv is None:
        msg = (f"train/val bounds absent ({P_TRAINVAL.name}); the certificate covers the "
               f"scored corpora only. Run paper_tooling/hk1tree_trainval.py.")
        print(f"!! {msg}", file=sys.stderr)
        if args.require_trainval:
            sys.exit(2)
    else:
        parts.append(tv)
    cert = pd.concat(parts, ignore_index=True)

    audit = pd.read_csv(P_AUDIT, usecols=["instance_name", "bucket"]).set_index("instance_name")
    cert["tour_audit_bucket"] = cert["instance"].map(audit["bucket"])

    flagged = cert[cert.excess > 0]
    wrong = cert[cert.proven_wrong].sort_values("excess_pct", ascending=False)
    flagged.to_csv(OUT / "label_certificate_flagged.csv", index=False)
    wrong.to_csv(OUT / "label_certificate.csv", index=False)

    def per(g: pd.DataFrame) -> dict:
        return {"N": int(len(g)), "bound_above_label": int((g.excess > 0).sum()),
                "proven_wrong": int(g.proven_wrong.sum()),
                "worst_excess_pct": float(g.excess_pct.max()),
                "q_unverified": int((~g.q_verified).sum())}

    summary = {
        "test": "bound > label + n*q/2  (TSPLIB: bound > label, one metric, no slack)",
        "rel_eps": REL_EPS,
        "by_split": {f"{c}/{s}": per(g) for (c, s), g in cert.groupby(["corpus", "split"])},
        "total_evaluated": int(len(cert)),
        "total_proven_wrong": int(cert.proven_wrong.sum()),
        "proven_wrong_in_a_scored_set": int(cert[cert.scored].proven_wrong.sum()),
        "proven_wrong_by_tour_audit_bucket":
            wrong.tour_audit_bucket.fillna("not_in_audit").value_counts().to_dict(),
        "coverage_gaps": {
            "tsplib_no_bound": sorted(pd.read_csv(P_TSPLIB).query("status != 'ok'").instance),
            "nd_trainval_covered": bool(tv is not None),
        },
    }
    (OUT / "label_certificate.json").write_text(json.dumps(summary, indent=2))

    print("=" * 82)
    print("LABEL CERTIFICATE")
    print("=" * 82)
    for k, v in summary["by_split"].items():
        print(f"  {k:<22} N={v['N']:>6}  bound>label {v['bound_above_label']:>5}  "
              f"PROVEN WRONG {v['proven_wrong']:>4}  worst excess {v['worst_excess_pct']:+.4f}%")
    print(f"\n  total proven wrong: {summary['total_proven_wrong']} "
          f"({summary['proven_wrong_in_a_scored_set']} of them in a scored evaluation set)")
    print(f"  against the reference-tour audit: {summary['proven_wrong_by_tour_audit_bucket']}")
    cols = [c for c in ("instance", "corpus", "split", "n", "d", "solver", "label", "bound",
                        "q", "slack", "excess_pct", "margin", "tour_audit_bucket")
            if c in wrong.columns]
    print("\n" + wrong[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {OUT / 'label_certificate.csv'}")


if __name__ == "__main__":
    main()
