"""Whole-box CPU load sensor, matching ``\\Processor(_Total)\\% Processor Time``.

Why this file exists
--------------------
``paper_tooling/hk1tree_frontier_timing.py`` reads
``Win32_Processor.LoadPercentage``. ``hk1tree_frontier_bank.json`` records that
this sensor returned 81-96% at moments when the real counter returned 18.8-21.8%
-- it is an instantaneous, heavily quantised reading and is not a box-load
measurement. Every timing pass that wants to state "the box was quiet" needs the
same sensor the published protocol quotes.

``GetSystemTimes`` is that sensor. The Windows performance counter
``\\Processor(_Total)\\% Processor Time`` is defined as
``100 * (1 - dIdle / (dKernel + dUser))`` over a sampling interval, where
``dKernel`` already includes idle. This module computes exactly that through
``ctypes``, with no subprocess and no third-party dependency (``psutil`` is not
installed in the project interpreter).

:func:`validate_against_perf_counter` cross-checks it against ``Get-Counter``
once so the number is not merely asserted to be the same thing.
"""

from __future__ import annotations

import ctypes
import subprocess
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetSystemTimes.argtypes = [
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
    ctypes.POINTER(wintypes.FILETIME),
]
_kernel32.GetSystemTimes.restype = wintypes.BOOL


def _filetime_to_int(ft: wintypes.FILETIME) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def _system_times() -> tuple[int, int, int]:
    """(idle, kernel, user) in 100 ns ticks. ``kernel`` includes ``idle``."""
    idle = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    if not _kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return _filetime_to_int(idle), _filetime_to_int(kernel), _filetime_to_int(user)


def cpu_busy_pct(interval_s: float = 1.0) -> float:
    """Whole-box busy percent over ``interval_s``, all logical cores pooled."""
    i0, k0, u0 = _system_times()
    time.sleep(interval_s)
    i1, k1, u1 = _system_times()
    d_idle = i1 - i0
    d_total = (k1 - k0) + (u1 - u0)
    if d_total <= 0:
        return float("nan")
    return 100.0 * (1.0 - d_idle / d_total)


@dataclass(frozen=True)
class QuietReading:
    """One pre-measurement observation of the box."""

    busy_pct: float
    interval_s: float
    foreign_solver_procs: tuple[str, ...]

    @property
    def is_quiet(self) -> bool:
        return self.busy_pct == self.busy_pct and not self.foreign_solver_procs


_SOLVER_PROC_NAMES = ("concorde", "LKH-3.exe", "LKH.exe", "linkern")


def foreign_solver_processes(exclude_pids: frozenset[int] = frozenset()) -> tuple[str, ...]:
    """Names of solver processes already running, Windows side and WSL side.

    A second solver in flight is the failure mode that invalidated the first
    1-tree cost pass in this project (see ``hk1tree_frontier_bank.json ->
    co_measurement_incident``). It is cheaper to assert than to detect after
    the fact.
    """
    found: list[str] = []
    try:
        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for line in out.splitlines():
            if not line.startswith('"'):
                continue
            parts = line.split('","')
            name = parts[0].lstrip('"')
            pid = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else -1
            if pid in exclude_pids:
                continue
            if name in ("LKH-3.exe", "LKH.exe", "linkern.exe", "concorde.exe"):
                found.append(f"win:{name}:{pid}")
    except Exception:
        found.append("win:tasklist-unavailable")
    try:
        out = subprocess.run(
            ["wsl", "-e", "bash", "-c", "pgrep -a -f 'concorde|LKH|linkern' || true"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # pgrep -f matches the pgrep pattern itself under bash -c; drop it.
            if "pgrep" in line:
                continue
            found.append(f"wsl:{line[:80]}")
    except Exception:
        found.append("wsl:pgrep-unavailable")
    return tuple(found)


def observe(interval_s: float = 1.0, settle_s: float = 0.5) -> QuietReading:
    """Sample the box immediately before a timed run.

    The process scan runs *first* and is then allowed to settle. Scanning costs
    a ``tasklist`` and a WSL ``pgrep``, and a WSL launch is itself a CPU spike:
    sampling the load in the second right after it measures the probe, not the
    box. An early pass of this harness read 45-58% that way against 18-23% at
    rest.
    """
    procs = foreign_solver_processes()
    time.sleep(settle_s)
    return QuietReading(
        busy_pct=cpu_busy_pct(interval_s),
        interval_s=interval_s,
        foreign_solver_procs=procs,
    )


class LoadSampler:
    """Sample whole-box load *while* a timed run is in flight.

    A one-second reading taken before a ten-minute solve is thin evidence that
    the box stayed quiet for the ten minutes that matter. This samples
    throughout, on a daemon thread that spends all its time asleep inside
    ``GetSystemTimes``, and reports the distribution.

    The sampled load *includes* the solver under measurement. Both solvers here
    are single-threaded, so on a 20 logical core box the solver's own
    contribution is about 5 percentage points; anything materially above the
    pre-run baseline plus 5 is another tenant.
    """

    def __init__(self, period_s: float = 5.0) -> None:
        self._period = period_s
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        # The wait is the sampling window AND the stop check. Sleeping on the
        # stop event rather than on ``time.sleep`` matters: a plain sleep makes
        # every short run pay the full period on teardown, which on a corpus of
        # sub-second solves is most of the wall clock.
        while True:
            try:
                i0, k0, u0 = _system_times()
                stopped = self._stop.wait(self._period)
                i1, k1, u1 = _system_times()
            except Exception:
                return
            d_idle = i1 - i0
            d_total = (k1 - k0) + (u1 - u0)
            if d_total > 0:
                self._samples.append(100.0 * (1.0 - d_idle / d_total))
            if stopped:
                return

    def __enter__(self) -> "LoadSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)

    @property
    def samples(self) -> list[float]:
        return list(self._samples)

    def summary(self) -> dict[str, float | int]:
        s = [x for x in self._samples if x == x]
        if not s:
            return {"n": 0}
        s.sort()
        return {
            "n": len(s),
            "min": s[0],
            "median": s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2]),
            "max": s[-1],
        }


def validate_against_perf_counter(samples: int = 4) -> dict[str, object]:
    """Read both sensors back to back and return the pairs.

    Run once per measurement campaign so the claim "this is the same counter the
    published protocol quotes" is evidence rather than an assertion.
    """
    pairs: list[dict[str, float]] = []
    for _ in range(samples):
        ours = cpu_busy_pct(1.0)
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\\Processor(_Total)\\% Processor Time' "
             "-SampleInterval 1 -MaxSamples 1).CounterSamples.CookedValue"],
            capture_output=True, text=True, timeout=90,
        ).stdout.strip()
        try:
            theirs = float(out)
        except ValueError:
            theirs = float("nan")
        wmi = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Measure-Object -Property "
             "LoadPercentage -Average).Average"],
            capture_output=True, text=True, timeout=90,
        ).stdout.strip()
        try:
            wmi_v = float(wmi)
        except ValueError:
            wmi_v = float("nan")
        pairs.append({
            "getsystemtimes_pct": ours,
            "perf_counter_pct": theirs,
            "win32_processor_loadpercentage_pct": wmi_v,
        })
    deltas = [
        abs(p["getsystemtimes_pct"] - p["perf_counter_pct"])
        for p in pairs
        if p["perf_counter_pct"] == p["perf_counter_pct"]
    ]
    return {
        "pairs": pairs,
        "max_abs_delta_pct": max(deltas) if deltas else float("nan"),
        "note": ("GetSystemTimes and \\Processor(_Total)\\% Processor Time are the "
                 "same quantity sampled over slightly different windows; "
                 "Win32_Processor.LoadPercentage is shown for contrast because "
                 "the project's older harness reads it."),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_against_perf_counter(), indent=1))
