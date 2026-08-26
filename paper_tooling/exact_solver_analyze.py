"""Reduce the measured solver campaign and place it on the shared cost axis.

Inputs
------
``exact_solver_tsplib_concorde.csv`` and ``exact_solver_tsplib_lkh.csv`` from
``paper_tooling/exact_solver_tsplib.py``, plus the already-banked estimator and
1-tree cost cells so the solver lands on the same axis rather than beside it.

Outputs
-------
``paper_tooling/exact_solver_bank.json``
    Every solver number, with its protocol and its censoring.
``paper_tooling/cost_axis_tsplib_euc2d.csv``
    One row per method on the TSPLIB EUC_2D corpus, from the cheapest closed
    form to a certified exact solve, in milliseconds and in multiples of
    GART 2.0.

Two statistical points that the reduction turns on
--------------------------------------------------
**Censoring.** Instances that hit the cap have no finishing time, only a lower
bound on one. Dropping them would bias the median down by exactly the hardest
instances, which is the opposite of what an upper anchor is for. The corpus
median is instead computed on the full corpus with censored entries ordered
above every observed time -- which is sound, because a censored time is known
to exceed the cap and the cap exceeds every certified time by construction. The
median of a right-censored sample is identified exactly whenever fewer than
half the sample is censored; the bank records whether that held.

**Two clocks, never averaged.** Concorde's number is time to a *certificate*.
LKH's is time to its own best tour, with no certificate at all. They answer
different questions, they are reported as separate rows, and the bank has no
key that combines them.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PT = ROOT / "paper_tooling"

CONCORDE_CSV = PT / "exact_solver_tsplib_concorde.csv"
LKH_CSV = PT / "exact_solver_tsplib_lkh.csv"
GART_BANK = PT / "gart2_timing_bank.json"
FRONTIER_BANK = PT / "frontier_manuscript_bank.json"
POSITION_BANK = PT / "frontier_positioning_bank.json"

OUT_BANK = PT / "exact_solver_bank.json"
OUT_AXIS = PT / "cost_axis_tsplib_euc2d.csv"

#: The buckets ``tab:tsplib_by_size`` and the solo timing pass both use.
BUCKETS = (("n in [51,150]", 51, 150), ("n in [151,400]", 151, 400),
           ("n > 400", 401, 10**9))

CENSORED = ("censored", "error")


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def censored_median(times: list[float], censored_at: list[float]) -> dict:
    """Median of a right-censored sample, and whether it is identified.

    ``times`` are observed completions; ``censored_at`` are lower bounds on the
    unobserved ones. Every censored value exceeds the cap, and the cap is at
    least every observed value, so the ordered sample is
    ``sorted(times) + [>= cap] * len(censored_at)``. The median is a real
    observation whenever it falls in the observed prefix.
    """
    n = len(times) + len(censored_at)
    if n == 0:
        return {"value_s": float("nan"), "identified": False, "N": 0}
    obs = sorted(times)
    lo_idx = (n - 1) // 2
    hi_idx = n // 2
    identified = hi_idx < len(obs)
    if identified:
        val = 0.5 * (obs[lo_idx] + obs[hi_idx]) if n % 2 == 0 else obs[lo_idx]
    else:
        val = float("nan")
    return {
        "value_s": val,
        "identified": identified,
        "N": n,
        "n_observed": len(obs),
        "n_censored": len(censored_at),
        "censoring_pct": 100.0 * len(censored_at) / n,
        "lower_bound_if_unidentified_s": (min(censored_at) if censored_at and not identified
                                          else None),
    }


#: Concorde refuses a TSPLIB file carrying a FIXED_EDGES_SECTION. That is a
#: property of the instance, not a fault in the run, and it must not be counted
#: as a failed solve or as a censored one.
UNSUPPORTED_MARK = "Not set up for fixed edges"


def load_arm(path: Path, solver: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df.solver == solver].copy()
    for c in ("wall_s", "time_to_best_s", "solver_self_time_s", "cost",
              "published_optimum", "lower_bound", "upper_bound", "gap_pct_at_stop"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n"] = df["n"].astype(int)
    # pandas infers bool for this column when every row parses as True/False;
    # the recovery step below writes strings into it, so widen it up front.
    df["matches_optimum"] = df["matches_optimum"].astype(str)
    df["status"] = df["status"].astype(str)
    if solver == "concorde":
        for i, r in df.iterrows():
            if r.status != "error":
                continue
            log = PT / "exact_solver_logs" / "concorde" / f"{r.instance}.log"
            if log.exists() and UNSUPPORTED_MARK in log.read_text(
                    encoding="utf-8", errors="replace"):
                df.at[i, "status"] = "unsupported"
    if solver == "lkh":
        _recover_lkh_hard_timeouts(df)
    return df.sort_values("n").reset_index(drop=True)


RE_TRIAL = re.compile(r"^\*\s*(\d+):\s*Cost\s*=\s*([0-9]+).*?Time\s*=\s*([0-9.]+)\s*sec",
                      re.MULTILINE)


def _recover_lkh_hard_timeouts(df: pd.DataFrame) -> None:
    """Rescue rows the runner could only mark ``error``.

    ``TIME_LIMIT`` bounds LKH's optimisation loop but is only tested at trial
    boundaries, and it does not bound candidate-set preprocessing at all. On the
    largest instances a single trial outlasts the limit, the harness's backstop
    process timeout fires first, and LKH never prints the ``Cost.min`` /
    ``Run 1:`` summary the runner parses -- so the row arrives as ``error`` even
    though the log holds a full trace of improvements.

    That is censoring, not failure, and the trace still carries the best tour
    and the time it was reached. Recovering it here keeps the hardest instances
    in the sample, which is exactly where dropping rows would flatter the
    solver.
    """
    for i, r in df.iterrows():
        if r.status != "error":
            continue
        log_path = PT / "exact_solver_logs" / "lkh" / f"{r.instance}.log"
        if not log_path.exists():
            continue
        trials = RE_TRIAL.findall(log_path.read_text(encoding="utf-8", errors="replace"))
        if not trials:
            continue
        best = min(float(c) for _t, c, _tt in trials)
        t_best = next(float(tt) for _t, c, tt in trials if float(c) == best)
        df.at[i, "status"] = "censored_hard_timeout"
        df.at[i, "cost"] = best
        df.at[i, "upper_bound"] = best
        df.at[i, "time_to_best_s"] = t_best
        opt = r.published_optimum
        df.at[i, "matches_optimum"] = str(
            bool(pd.notna(opt) and abs(best - float(opt)) < 0.5))


def cross_clock_check(df: pd.DataFrame) -> dict:
    """Compare the in-WSL wall clock against the Windows-side wall clock.

    The Concorde arm is timed twice by construction: ``date +%s.%N`` inside the
    VM around the solver, and ``time.perf_counter`` on the Windows side around
    the whole ``wsl.exe`` call. The second is the clock the estimator column was
    measured on, so agreement between them is what licenses putting the two on
    one axis.

    It also explains the cap overshoot. GNU ``timeout`` fires after 600 s of the
    VM's ``ITIMER_REAL``, and both wall clocks say roughly 660 s of real time
    elapsed: the VM was descheduled for about a tenth of the run. The nominal
    cap is therefore in solver time and the realised cap is in wall time, and
    the bank states both.
    """
    outer = []
    for s in df.notes.fillna("{}").astype(str):
        try:
            outer.append(float(json.loads(s)["outer_wall_s"]))
        except (ValueError, KeyError):
            outer.append(float("nan"))
    inner = df.wall_s.tolist()
    pairs = [(i, o) for i, o in zip(inner, outer) if o == o]
    if not pairs:
        return {"comparable": False, "reason": "no outer-clock readings"}

    # outer - inner is the wsl.exe launch, bash, mktemp and cp that sit OUTSIDE
    # the solver's own clock. It is additive, not proportional, so it must be
    # characterised in seconds; expressed as a percentage it looks alarming on a
    # one-second solve and invisible on a ten-minute one.
    overhead = [o - i for i, o in pairs]
    # Clock RATE agreement is only testable where the additive term is
    # negligible, i.e. on the long runs.
    longs = [(i, o) for i, o in pairs if i > 60.0]
    rel_long = [100.0 * abs(o - i) / i for i, o in longs]
    return {
        "comparable": True,
        "n_pairs": len(pairs),
        "launcher_overhead_s": {
            "what": ("Windows-side wall minus in-WSL wall: wsl.exe start, bash, "
                     "mktemp, cp. Excluded from the reported Concorde time by "
                     "construction, since the reported figure is the in-WSL wall."),
            "median": _median(overhead),
            "min": min(overhead),
            "max": max(overhead),
        },
        "clock_rate_agreement_on_runs_over_60s": {
            "n": len(rel_long),
            "max_relative_disagreement_pct": max(rel_long) if rel_long else None,
            "median_relative_disagreement_pct": _median(rel_long) if rel_long else None,
        },
        "verdict": ("Once the additive launcher cost is separated out, the in-VM "
                    "clock and the Windows clock agree to a fraction of a percent "
                    "on the long runs. The Concorde times are therefore on the "
                    "same wall-clock footing as the estimator column even though "
                    "the solver runs in a VM. What the VM does cost is throughput, "
                    "not timekeeping: see cap_enforcement.why_the_overshoot."),
    }


def quietness(df: pd.DataFrame) -> dict:
    """What the box was doing while this arm was measured."""
    pre: list[float] = []
    for s in df.cpu_busy_pct_before.dropna().astype(str):
        pre.extend(float(x) for x in s.split(";") if x.strip())
    during: list[float] = []
    dmax: list[float] = []
    for s in df.get("cpu_busy_pct_during", pd.Series(dtype=str)).dropna().astype(str):
        try:
            for blk in json.loads(s):
                if blk.get("n"):
                    during.append(float(blk["median"]))
                    dmax.append(float(blk["max"]))
        except (ValueError, TypeError):
            continue
    foreign = sorted({p for s in df.foreign_solver_procs.fillna("").astype(str)
                      for p in s.split(";") if p})
    return {
        "sensor": ("GetSystemTimes, the same quantity as "
                   "\\Processor(_Total)\\% Processor Time; cross-checked in "
                   "paper_tooling/quiet_box.py, max |delta| 7.6 points over four "
                   "paired samples. Win32_Processor.LoadPercentage, which the "
                   "older harness reads, returned 71-96% against the same 18-23%."),
        "before_run_pct": {
            "n": len(pre),
            "min": min(pre) if pre else None,
            "median": _median(pre),
            "max": max(pre) if pre else None,
        },
        "during_run_pct": {
            "n": len(during),
            "median_of_per_run_medians": _median(during),
            "max_observed": max(dmax) if dmax else None,
            "note": ("includes the solver under measurement; both solvers are "
                     "single-threaded, so on 20 logical cores the solver itself "
                     "accounts for about 5 points of this."),
        },
        "foreign_solver_processes_seen": foreign,
        "comparison_protocol_window": ("13-16% for the roster solo pass, 17.5-18.7% "
                                       "for the classical solo pass "
                                       "(gart2_timing_bank.json)"),
    }


def concorde_summary(df: pd.DataFrame, gart2_ms: float,
                     plain_variant: dict | None = None) -> dict:
    ok = df[df.status == "optimal"]
    cens = df[df.status.isin(CENSORED)]
    unsup = df[df.status == "unsupported"]
    scored = df[df.status != "unsupported"]
    cap = float(df.cap_s.iloc[0])

    # A censored row's lower bound is the time it actually ran, not the nominal
    # cap: GNU timeout fires at the cap but the SIGKILL lands when the process
    # next leaves an uninterruptible state, which overshot here. Using the
    # measured wall is both tighter and true.
    cens_lb = cens.wall_s.tolist()
    med = censored_median(ok.wall_s.tolist(), cens_lb)
    per_bucket = {}
    for label, lo, hi in BUCKETS:
        b = scored[(scored.n >= lo) & (scored.n <= hi)]
        bok = b[b.status == "optimal"]
        bc = b[b.status.isin(CENSORED)]
        per_bucket[label] = {
            "N": int(len(b)),
            "n_certified": int(len(bok)),
            "n_censored": int(len(bc)),
            "median_s": censored_median(bok.wall_s.tolist(), bc.wall_s.tolist()),
            "max_certified_s": float(bok.wall_s.max()) if len(bok) else None,
        }

    wrong = ok[(ok.published_optimum.notna()) &
               ((ok.cost - ok.published_optimum).abs() > 0.5)]
    censored_rows = [{
        "instance": r.instance,
        "n": int(r.n),
        "root_lp_lower_bound": None if pd.isna(r.lower_bound) else float(r.lower_bound),
        "best_tour_upper_bound": None if pd.isna(r.upper_bound) else float(r.upper_bound),
        "gap_pct_at_cap": None if pd.isna(r.gap_pct_at_stop) else float(r.gap_pct_at_stop),
        "published_optimum": None if pd.isna(r.published_optimum) else int(r.published_optimum),
        "status": r.status,
    } for r in cens.itertuples()]

    return {
        "question_answered": "wall clock to a certificate of optimality",
        "solver": "Concorde (branch-and-cut), build at /home/catst/concorde/TSP/concorde",
        "invocation": "concorde -s 99 -x -o <tour> <instance>.tsp, one process, no cut server, no boss/grunt parallelism",
        "environment": "WSL2 Ubuntu 24.04 on the host's physical cores; scratch on ext4 /tmp, not the 9p /mnt bridge",
        "cap_s": cap,
        "cap_enforcement": {
            "mechanism": ("GNU timeout -s KILL inside WSL, so the solver is "
                          "killed Linux-side; killing wsl.exe from Windows would "
                          "orphan it, which is the contamination that invalidated "
                          "an earlier timing pass in this project."),
            "nominal_cap_s_solver_clock": cap,
            "realised_kill_walls_s": sorted(round(float(x), 1) for x in cens_lb),
            "realised_cap_wall_s_median": _median(cens_lb) if cens_lb else None,
            "max_overshoot_s": (max(cens_lb) - cap) if cens_lb else 0.0,
            "why_the_overshoot": (
                "GNU timeout fires after 600 s of the VM's ITIMER_REAL. Both wall "
                "clocks -- date inside WSL and perf_counter on the Windows side -- "
                "report about 660 s of real time for those runs, so the VM was "
                "descheduled for roughly a tenth of each long solve. The nominal "
                "cap is in solver time; the realised cap is in wall time. One "
                "instance (fl1577) certified at 658.8 s wall, i.e. inside the "
                "600 s solver budget but past the nominal wall figure, which is "
                "the same effect seen from the other side."),
            "note": ("Censored entries are carried at their measured wall, not at "
                     "the nominal cap. Every one of them exceeds every certified "
                     "time, so the ordering the censored median relies on holds."),
        },
        "N_corpus": int(len(df)),
        "N_scored": int(len(scored)),
        "unsupported": {
            "n": int(len(unsup)),
            "instances": unsup.instance.tolist(),
            "reason": ("Concorde refuses a TSPLIB file with a "
                       "FIXED_EDGES_SECTION: 'ERROR: Not set up for fixed edges'. "
                       "linhp318 carries one (edge 1-214) because it is the "
                       "Hamiltonian path variant of lin318 encoded as a TSP. It "
                       "is the only such file in the 78."),
            "plain_tsp_variant": plain_variant,
        },
        "n_certified": int(len(ok)),
        "n_censored": int(len(cens)),
        "certified_pct": 100.0 * len(ok) / len(scored) if len(scored) else float("nan"),
        "median_s_corpus_censoring_aware": med,
        "median_s_certified_only": _median(ok.wall_s.tolist()),
        "min_s": float(ok.wall_s.min()) if len(ok) else None,
        "min_instance": ok.loc[ok.wall_s.idxmin(), "instance"] if len(ok) else None,
        "max_certified_s": float(ok.wall_s.max()) if len(ok) else None,
        "max_certified_instance": ok.loc[ok.wall_s.idxmax(), "instance"] if len(ok) else None,
        "largest_n_certified": int(ok.n.max()) if len(ok) else None,
        "smallest_n_censored": int(cens.n.min()) if len(cens) else None,
        "total_wall_hours_consumed": float(
            (df.wall_s * df.repeats.astype(int)).sum() / 3600.0),
        "median_s_including_linhp318_plain_variant": (
            censored_median(ok.wall_s.tolist() + [plain_variant["wall_s"]], cens_lb)
            if plain_variant and plain_variant.get("status") == "optimal" else None),
        "by_bucket": per_bucket,
        "censored_instances": censored_rows,
        "certificate_disagreements_with_published_optimum": [
            {"instance": r.instance, "concorde": float(r.cost),
             "published": int(r.published_optimum)} for r in wrong.itertuples()],
        "vs_gart2": {
            "gart2_median_ms": gart2_ms,
            "median_ratio_x": (med["value_s"] * 1000.0 / gart2_ms
                               if med["identified"] else None),
            "max_certified_ratio_x": (float(ok.wall_s.max()) * 1000.0 / gart2_ms
                                      if len(ok) else None),
        },
        "repeats": {
            "policy": "3 runs, median, for any instance whose first run finished under 10 s; a single run above that",
            "n_instances_with_repeats": int((df.repeats.astype(int) > 1).sum()),
            "n_instances_single_run": int((df.repeats.astype(int) == 1).sum()),
        },
    }


def lkh_summary(df: pd.DataFrame, gart2_ms: float) -> dict:
    got = df[df.cost.notna()]
    cap = float(df.cap_s.iloc[0])
    cens_soft = df[df.status == "censored"]
    cens_hard = df[df.status == "censored_hard_timeout"]
    cens = df[df.status.isin(("censored", "censored_hard_timeout"))]
    matched = got[got.matches_optimum.astype(str) == "True"]

    # Time-to-best is OBSERVED even on a censored run: LKH reached its best at a
    # timestamp it printed. What the cap censors is the tour's quality, not the
    # clock. So this median is uncensored and the quality caveat is carried
    # separately, in n_censored_by_time_limit and the gap fields.
    ttb = got.time_to_best_s.dropna().tolist()
    med_ttb = {
        "value_s": _median(ttb),
        "identified": bool(ttb),
        "N": len(ttb),
        "n_observed": len(ttb),
        "n_censored": 0,
        "note": ("uncensored: a run stopped at the cap still reported when its "
                 "best tour was reached. The cap censors tour quality, which is "
                 "reported in the gap fields, not this timestamp."),
    }
    med_wall = censored_median(
        df[df.status == "best_found"].wall_s.tolist(), cens.wall_s.tolist())

    per_bucket = {}
    for label, lo, hi in BUCKETS:
        b = got[(got.n >= lo) & (got.n <= hi)]
        per_bucket[label] = {
            "N": int(len(b)),
            "median_time_to_best_s": _median(b.time_to_best_s.dropna().tolist()),
            "median_total_wall_s": _median(b.wall_s.tolist()),
            "n_matching_published_optimum": int(
                (b.matches_optimum.astype(str) == "True").sum()),
        }

    gaps = got[got.published_optimum.notna()].copy()
    gaps["gap_pct"] = 100.0 * (gaps.cost - gaps.published_optimum) / gaps.published_optimum
    worst = gaps.sort_values("gap_pct", ascending=False).head(5)

    return {
        "question_answered": ("wall clock at which the run's best tour was first "
                              "reached, from LKH's own trial trace. No optimality "
                              "certificate is produced or implied."),
        "solver": "LKH-3 (C:\\LKH\\LKH-3.exe)",
        "invocation": f"RUNS=1, SEED=1, TIME_LIMIT={cap:g}, TRACE_LEVEL=1, MAX_TRIALS left at its default of DIMENSION",
        "environment": "Windows 11 native, the same environment as the estimator column",
        "cap_s": cap,
        "N": int(len(df)),
        "n_with_a_tour": int(len(got)),
        "n_censored_by_time_limit": int(len(cens)),
        "censoring_modes": {
            "time_limit": {
                "n": int(len(cens_soft)),
                "instances": cens_soft.instance.tolist(),
                "meaning": "LKH's own TIME_LIMIT stopped the optimisation loop",
            },
            "hard_process_timeout": {
                "n": int(len(cens_hard)),
                "instances": cens_hard.instance.tolist(),
                "walls_s": [round(float(x), 1) for x in cens_hard.wall_s],
                "meaning": (
                    "LKH overran its own TIME_LIMIT and the harness's backstop "
                    "process timeout (cap + 120 s) fired. TIME_LIMIT is tested at "
                    "trial boundaries and does not bound candidate-set "
                    "preprocessing, so at these sizes a single trial outlasts it. "
                    "The best tour and its timestamp were recovered from LKH's "
                    "trial trace; these rows are censored, not failed."),
            },
        },
        "n_matching_published_optimum": int(len(matched)),
        "matching_pct": 100.0 * len(matched) / len(got) if len(got) else float("nan"),
        "median_time_to_best_s": med_ttb,
        "median_total_wall_s": med_wall,
        "min_time_to_best_s": min(ttb) if ttb else None,
        "max_time_to_best_s": max(ttb) if ttb else None,
        "mean_gap_to_published_optimum_pct": float(gaps.gap_pct.mean()) if len(gaps) else None,
        "max_gap_to_published_optimum_pct": float(gaps.gap_pct.max()) if len(gaps) else None,
        "worst_gap_instances": [
            {"instance": r.instance, "n": int(r.n), "gap_pct": float(r.gap_pct)}
            for r in worst.itertuples()],
        "by_bucket": per_bucket,
        "clock_granularity_caveat": (
            "time-to-best is read from LKH's own trace, whose timestamps are "
            "printed to 0.01 s. Below roughly n = 300 the best tour is reached "
            "inside the first tick and the value floors at 0.00; median_total_wall_s "
            "is the honest figure at those sizes and is measured by this harness, "
            "not by LKH."),
        "vs_gart2": {
            "gart2_median_ms": gart2_ms,
            "median_total_wall_ratio_x": (med_wall["value_s"] * 1000.0 / gart2_ms
                                          if med_wall["identified"] else None),
        },
    }


def corroboration(conc_df: pd.DataFrame, lkh_df: pd.DataFrame,
                  plain: dict | None) -> dict:
    """Places where the two solvers disagree, and what the disagreement means.

    Two independent implementations reading the same files is a cheap and
    strong audit. Every disagreement found here is explained, and two of them
    corroborate a label defect this project had already identified by a
    different route.
    """
    def row(df, inst, cols):
        r = df[df.instance == inst]
        if r.empty:
            return {}
        out = {}
        for c in cols:
            v = r.iloc[0][c]
            if pd.isna(v):
                out[c] = None
            elif hasattr(v, "item"):          # numpy scalar -> json-serialisable
                out[c] = v.item()
            else:
                out[c] = v
        return out

    cols = ("status", "cost", "published_optimum", "wall_s")
    return {
        "_what": ("Concorde and LKH-3 read the same TSPLIB files with independent "
                  "parsers, distance functions and search strategies. The three "
                  "places they part company are all informative."),
        "linhp318": {
            "concorde_on_the_shipped_file": "refused -- FIXED_EDGES_SECTION",
            "lkh_on_the_shipped_file": row(lkh_df, "linhp318", cols),
            "concorde_on_the_same_coordinates_without_the_fixed_edge": (
                {"cost": plain.get("certified_optimum"),
                 "wall_s": plain.get("wall_s"),
                 "status": plain.get("status")} if plain else None),
            "reading": (
                "LKH honours the fixed edge and returns 41345, TSPLIB's published "
                "value, which is the HAMILTONIAN PATH optimum. Concorde on the "
                "same coordinates with the fixed edge removed certifies 42029, "
                "the TOUR optimum. Both are right about different problems. The "
                "estimators on the cost axis all read coordinates only, so 42029 "
                "is the value they should be scored against -- which is what "
                "frontier_manuscript_bank.json -> labels.linhp318 already says, "
                "reached independently. This pass is a second witness for it, "
                "issued by a solver."),
        },
        "d657": {
            "concorde": row(conc_df, "d657", cols),
            "lkh": row(lkh_df, "d657", cols),
            "published_optimum": 48912,
            "reading": (
                "Concorde CERTIFIES 48913; LKH finds 48912, matching the published "
                "value. A certificate cannot be wrong about its own metric, so the "
                "two are minimising over distance functions that differ on at "
                "least one edge. d657's coordinates are non-integer (875.1, 983.7, "
                "...) and EUC_2D distances are nint(euclidean), so a value landing "
                "on a .5 boundary rounds differently under two implementations. "
                "25 of the 78 EUC_2D instances have non-integer coordinates and "
                "this is the only one where the two solvers disagree, so it is a "
                "tie-break coincidence, not a systematic parsing difference. The "
                "discrepancy is 1 unit in 48912, or 0.002%, and it does not touch "
                "any timing on this axis."),
            "non_integer_coordinate_instances_in_the_78": 25,
            "of_which_disagree": 1,
        },
        "lin318": {
            "lkh": row(lkh_df, "lin318", cols),
            "reading": ("LKH's worst miss on the corpus: 42143 against the optimum "
                        "42029, 0.27% high, at n=318 where it is exact almost "
                        "everywhere else. One run, one seed, default trial budget "
                        "-- this is what a heuristic with no certificate looks "
                        "like when it is unlucky, and it is the reason the LKH "
                        "column cannot substitute for the Concorde one."),
        },
    }


def polyak_runs() -> list[dict]:
    """Every Polyak cost pass on disk, richest budget last."""
    out = []
    for p in sorted(PT.glob("polyak_tsplib_timing_k*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")) | {"_file": p.name})
        except ValueError:
            continue
    return sorted(out, key=lambda d: max(int(k) for k in d["polyak_ms_by_k"]))


def polyak_cells() -> list[tuple[int, float, str, int]]:
    """``(k, ms, source, N)``, preferring the pass with the most repeats at each k.

    The two passes overlap: the ``k<=100`` run has 11 repeats, the ``k<=500`` run
    has 3 and re-measures every cheaper checkpoint on the way. Where they
    overlap the 11-repeat reading is the better estimate, so it wins; the
    ``k>100`` cells can only come from the 3-repeat run and are marked as such.
    """
    best: dict[int, tuple[float, str, int, int]] = {}
    for run in polyak_runs():
        reps = int(run["protocol"]["repeats"])
        for k_str, ms in run["polyak_ms_by_k"].items():
            k = int(k_str)
            prev = best.get(k)
            if prev is None or reps > prev[3]:
                best[k] = (float(ms),
                           f"paper_tooling/{run['_file']} ({reps} repeats)",
                           int(run["N_instances"]), reps)
    return [(k, v[0], v[1], v[2]) for k, v in sorted(best.items())]


def matched_instances(fr_bank: dict) -> set[str]:
    """The 77 the 1-tree cost pass was measured on: the 78 minus its n cap."""
    excluded = set(fr_bank["tsplib"].get("excluded_instance", "").split(",")) - {""}
    df = pd.read_csv(ROOT / "tsplib_benchmark" / "results" / "all_models_tsplib.csv",
                     usecols=["instance", "edge_weight_type"]).drop_duplicates()
    names = set(df[df.edge_weight_type == "EUC_2D"].instance.astype(str))
    return names - excluded


def polyak_accuracy(matched: set[str]) -> dict[int, float]:
    """Polyak bound MAPE against the label, on the same matched subset.

    Read from the accuracy sweep (``hk1tree_polyak_tsplib.csv``), which is a
    sound source for *bounds* -- it is only that file's TIME columns that are
    unusable per-k, because it ran one ascent to k=2000 and read the bounds off
    at each checkpoint.
    """
    path = PT / "hk1tree_polyak_tsplib.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["instance", "k", "bound", "true_cost", "status"])
    df = df[(df.status == "ok") & df.instance.isin(matched)]
    df = df[df.true_cost > 0]
    df["ape"] = 100.0 * (df.bound - df.true_cost).abs() / df.true_cost
    return {int(k): float(v) for k, v in df.groupby("k").ape.mean().items()}


def polyak_vs_vj(fr_bank: dict) -> dict:
    """The two ascents' cost on the same corpus, same protocol, same box."""
    cells = polyak_cells()
    if not cells:
        return {"available": False,
                "reason": "no polyak_tsplib_timing_k*.json on disk"}
    vj = fr_bank["tsplib"]["bound_ms_by_k"]
    vj_acc = fr_bank["tsplib"]["bound_mape_pct_by_k"]
    pol_acc = polyak_accuracy(matched_instances(fr_bank))
    runs = polyak_runs()
    rows = {}
    dominated = []
    for k, ms, src, N in cells:
        v = vj.get(str(k))
        pa, va = pol_acc.get(k), vj_acc.get(str(k))
        if v and pa is not None and va is not None and ms > v and pa > va:
            dominated.append(k)
        rows[str(k)] = {
            "polyak_ms": ms,
            "vj_ms": v,
            "polyak_over_vj": (ms / v) if v else None,
            "polyak_mape_pct": pa,
            "vj_mape_pct": va,
            "N": N,
            "source": src,
        }
    return {
        "available": True,
        "_what": ("Cost of the Polyak ascent against the Volgenant-Jonker / "
                  "Helsgaun ascent on TSPLIB EUC_2D. Until this pass the axis had "
                  "no Polyak cost cell on TSPLIB at all: hk1tree_polyak_tsplib.csv "
                  "records one ascent time per instance replicated across its k "
                  "rows, because that sweep ran a single ascent to k=2000 and read "
                  "the BOUNDS off at each checkpoint. Those columns are an "
                  "accuracy artefact, not a per-k cost."),
        "why_polyak_costs_more": (
            "Polyak's step length is scaled by the gap to a feasible tour, so the "
            "ascent cannot start without one. Its k=0 cell therefore carries a "
            "nearest-neighbour tour plus 2-opt sweeps that the V&J ascent never "
            "pays for. The gap between the two arms is widest at k=0 and narrows "
            "as the ascent comes to dominate."),
        "verdict": (
            f"On TSPLIB EUC_2D the V&J/Helsgaun ascent dominates Polyak at every "
            f"budget measured: dearer AND less accurate at k = {dominated}. This is "
            "the opposite of the ND arm, where Polyak supersedes V&J by a factor of "
            "23 (frontier_manuscript_bank.json -> _sources.nd). The axis was already "
            "printing V&J on TSPLIB and Polyak on ND; that split is now measured "
            "rather than assumed, and it is the right split."),
        "dominated_at_k": dominated,
        "reproducibility": (
            "the k<=500 run at 3 repeats re-measures every checkpoint the k<=100 run "
            "took at 11 repeats, in a separate process and session. The two agree to "
            "within 1-3% at every shared k, which bounds this pass's session-to-"
            "session drift."),
        "by_k": rows,
        "protocol_notes": [r["protocol"] for r in runs],
    }


def build_axis(conc: dict, lkh: dict, gart_bank: dict, fr_bank: dict,
               pos_bank: dict) -> pd.DataFrame:
    """One row per method, cheapest first."""
    gart2_ms = gart_bank["tsplib_by_size_time_one_protocol"]["time_ms"][
        "Total (all EUC_2D)"]["GART_2.0"]["time_ms"]

    rows: list[dict] = []
    backbone = {r["label"]: r for r in pos_bank["table_total_corpus"]}

    # -- estimators, from the solo protocol cells -----------------------------
    solo_total = gart_bank["tsplib_by_size_time_one_protocol"]["time_ms"]["Total (all EUC_2D)"]
    classical = gart_bank["classical_region_estimators_solo"]["time_ms"]["Total (all EUC_2D)"]
    v4 = gart_bank["lgbm_v4_solo"]["time_ms"]["Total (all EUC_2D)"]

    for src, family in ((classical, "closed form"), (solo_total, None), (v4, None)):
        for label, cell in src.items():
            bb = backbone.get(label, {})
            rows.append({
                "label": label,
                "family": family or bb.get("family", ""),
                "kind": "estimator",
                "k": "",
                "N": cell["N"],
                "ms": cell["time_ms"],
                "accuracy_MAPE_pct": bb.get("mape"),
                "complexity": bb.get("complexity", ""),
                "ms_source": "gart2_timing_bank.json, tag serial_solo_median11_quiet_2026-08-11",
                "environment": "Windows 11 native, one process, threads pinned to 1",
                "statistic": "median over 78 instances of the per-instance median of 11 repeats",
            })

    # -- the 1-tree ladder, from the solo re-measurement -----------------------
    ts = fr_bank["tsplib"]
    for k_str, ms in ts["bound_ms_by_k"].items():
        k = int(k_str)
        rows.append({
            "label": f"HK_1Tree_{k}",
            "family": "1-tree bound",
            "kind": "bound",
            "k": k,
            "N": ts["N_matched"],
            "ms": ms,
            "accuracy_MAPE_pct": ts["bound_mape_pct_by_k"][k_str],
            "complexity": r"\Theta(k n^2)",
            "ms_source": ("frontier_manuscript_bank.json -> tsplib.bound_ms_by_k "
                          "(hk1tree_frontier_bank.json, tag "
                          "serial_solo_median11_quiet_2026-08-11_1tree)"),
            "environment": "Windows 11 native, one process, threads pinned to 1",
            "statistic": ("median over the matched 77 instances of the per-instance "
                          "median over repeats; d18512 is above the estimator's own "
                          "dense-kernel cap of 16384"),
        })

    # -- the Polyak ascent, if its cost pass has been run ----------------------
    pol_acc = polyak_accuracy(matched_instances(fr_bank))
    for k, ms, src, N in polyak_cells():
        rows.append({
            "label": f"HK_1Tree_Polyak_{k}",
            "family": "1-tree bound (Polyak ascent)",
            "kind": "bound",
            "k": k,
            "N": N,
            "ms": ms,
            "accuracy_MAPE_pct": pol_acc.get(k),
            "complexity": r"\Theta(k n^2) + \Theta(n^2) upper bound",
            "ms_source": src,
            "environment": "Windows 11 native, one process, threads pinned to 1",
            "statistic": ("median over instances of the per-instance median over "
                          "repeats; includes the constructive NN + 2-opt upper "
                          "bound the Polyak step length requires and the V&J "
                          "ascent does not"),
        })

    # -- the two solvers -------------------------------------------------------
    cm = conc["median_s_corpus_censoring_aware"]
    rows.append({
        "label": "Concorde (certified optimal)",
        "family": "exact solver",
        "kind": "exact solver",
        "k": "",
        "N": conc["N_scored"],
        "ms": cm["value_s"] * 1000.0 if cm["identified"] else float("nan"),
        "accuracy_MAPE_pct": 0.0,
        "complexity": "branch-and-cut over an exponential cut family; no polynomial bound",
        "ms_source": "paper_tooling/exact_solver_tsplib_concorde.csv (this pass)",
        "environment": "WSL2 Ubuntu 24.04 on the host's physical cores, one process",
        "statistic": (f"median over the {conc['N_scored']} readable instances, "
                      f"censoring-aware, with {conc['n_censored']} censored at the "
                      f"{conc['cap_s']:g} s cap and {conc['n_certified']} certified; "
                      f"linhp318 is excluded, Concorde will not read its fixed edges"),
    })
    lm = lkh["median_total_wall_s"]
    rows.append({
        "label": "LKH-3 (time to best, no certificate)",
        "family": "heuristic solver",
        "kind": "heuristic solver",
        "k": "",
        "N": lkh["N"],
        "ms": lm["value_s"] * 1000.0 if lm["identified"] else float("nan"),
        "accuracy_MAPE_pct": lkh["mean_gap_to_published_optimum_pct"],
        "complexity": "Lin-Kernighan-Helsgaun; no bound, no certificate",
        "ms_source": "paper_tooling/exact_solver_tsplib_lkh.csv (this pass)",
        "environment": "Windows 11 native, one process",
        "statistic": (f"median total wall over all {lkh['N']} instances, censoring-aware, "
                      f"with {lkh['n_censored_by_time_limit']} censored at the "
                      f"{lkh['cap_s']:g} s cap"),
    })

    df = pd.DataFrame(rows)
    df["x_gart2"] = df.ms / gart2_ms
    df = df.sort_values("ms", na_position="last").reset_index(drop=True)
    return df[["label", "family", "kind", "k", "N", "ms", "x_gart2",
               "accuracy_MAPE_pct", "complexity", "statistic", "environment",
               "ms_source"]]


def main() -> None:
    gart_bank = json.loads(GART_BANK.read_text(encoding="utf-8"))
    fr_bank = json.loads(FRONTIER_BANK.read_text(encoding="utf-8"))
    pos_bank = json.loads(POSITION_BANK.read_text(encoding="utf-8"))
    gart2_ms = gart_bank["tsplib_by_size_time_one_protocol"]["time_ms"][
        "Total (all EUC_2D)"]["GART_2.0"]["time_ms"]

    conc_df = load_arm(CONCORDE_CSV, "concorde")
    lkh_df = load_arm(LKH_CSV, "lkh")

    plain_path = PT / "exact_solver_linhp318_plain_tsp.json"
    plain = json.loads(plain_path.read_text(encoding="utf-8")) if plain_path.exists() else None

    conc = concorde_summary(conc_df, gart2_ms, plain)
    lkh = lkh_summary(lkh_df, gart2_ms)

    axis = build_axis(conc, lkh, gart_bank, fr_bank, pos_bank)
    axis.to_csv(OUT_AXIS, index=False)

    span_lo = min(x for x in axis.ms if x == x)
    span_hi = max(x for x in axis.ms if x == x)

    bank = {
        "_what": ("Concorde and LKH-3 wall times measured on this box over the 78 "
                  "TSPLIB EUC_2D instances, and the full cost axis they anchor."),
        "_written_by": "paper_tooling/exact_solver_analyze.py",
        "_measured_by": "paper_tooling/exact_solver_tsplib.py",
        "_sources": {
            "concorde": "paper_tooling/exact_solver_tsplib_concorde.csv",
            "lkh": "paper_tooling/exact_solver_tsplib_lkh.csv",
            "raw_solver_logs": "paper_tooling/exact_solver_logs/<solver>/<instance>.log",
            "estimator_cost_cells": ("paper_tooling/gart2_timing_bank.json -> "
                                     "tsplib_by_size_time_one_protocol, "
                                     "classical_region_estimators_solo, lgbm_v4_solo"),
            "ladder_cost_cells": ("paper_tooling/frontier_manuscript_bank.json -> "
                                  "tsplib.bound_ms_by_k"),
        },
        "_relation_to_exact_solver_anchor": (
            "frontier_manuscript_bank.json -> exact_solver_anchor is a PUBLISHED "
            "Waterloo record (Xeon 2.8 GHz and Alpha 500 MHz, 25 of the 78 "
            "instances) and says so in its own provenance field. This key is a "
            "measurement on the same box as every other number on the cost axis. "
            "The two are not interchangeable and exact_solver_anchor is left "
            "untouched: the manuscript currently prints from it."),
        "corpus": {
            "name": "TSPLIB EUC_2D",
            "N": int(len(conc_df)),
            "n_min": int(conc_df.n.min()),
            "n_max": int(conc_df.n.max()),
        },
        "protocol": {
            "tag": "solver_serial_solo_capped600_2026-08-12",
            "matches": ("gart2_timing_bank.json -> tsplib_by_size_time_one_protocol "
                        "(tag serial_solo_median11_quiet_2026-08-11) in everything "
                        "that transfers: one solver process at a time, never two in "
                        "flight, single-threaded solvers, no pool, box sampled with "
                        "the same counter immediately before and throughout every "
                        "timed run."),
            "deviations": [
                ("Concorde runs under WSL2 because there is no Windows build on this "
                 "box; LKH-3 and the estimator column are Windows native. Same "
                 "physical cores, different environment. The two solver arms are "
                 "reported separately in part for this reason and are never pooled."),
                ("Repeats are affordable only in the small-n part of the corpus: "
                 "3 runs and a median where the first run finished under 10 s, a "
                 "single run above that. Every per-instance row carries its own "
                 "repeat count. The estimator column is a median of 11 throughout."),
                ("The corpus median is censoring-aware rather than a mean, because "
                 "the hardest instances have no finishing time at all."),
            ],
            "cap_s": conc["cap_s"],
            "seeds": {"concorde": 99, "lkh": 1},
        },
        "machine_quietness": {
            "concorde_arm": quietness(conc_df),
            "lkh_arm": quietness(lkh_df),
        },
        "cross_clock_check_concorde": cross_clock_check(conc_df),
        "cross_solver_corroboration": corroboration(conc_df, lkh_df, plain),
        "concorde_certified_optimal": conc,
        "lkh3_time_to_best": lkh,
        "polyak_vs_vj_ascent_cost": polyak_vs_vj(fr_bank),
        "cost_axis": {
            "_what": ("every method on one axis, TSPLIB EUC_2D, milliseconds and "
                      "multiples of GART 2.0"),
            "file": "paper_tooling/cost_axis_tsplib_euc2d.csv",
            "gart2_reference_ms": gart2_ms,
            "cheapest_ms": span_lo,
            "dearest_ms": span_hi,
            "span_orders_of_magnitude": math.log10(span_hi / span_lo),
            "rows": json.loads(axis.to_json(orient="records")),
        },
    }
    OUT_BANK.write_text(json.dumps(bank, indent=1), encoding="utf-8")

    print(f"wrote {OUT_BANK}")
    print(f"wrote {OUT_AXIS}")
    print()
    print(axis.to_string(index=False,
                         columns=["label", "kind", "N", "ms", "x_gart2",
                                  "accuracy_MAPE_pct"]))


if __name__ == "__main__":
    main()
