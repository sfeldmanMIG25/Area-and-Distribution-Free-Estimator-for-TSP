"""k=1000 Held--Karp 1-tree bound over the ND train and val splits.

``hk1tree_validate.py invariant`` covers the two scored corpora -- the 2D
benchmark and the ND test split, 19,500 instances.  The certificate in
``label_certificate.py`` applies to every released label, so this closes the
remaining 89,352 (69,768 train + 19,584 val) and lets the certificate report a
count for the whole corpus instead of the evaluated part of it.

Per instance it records the bound, w(0) and the unique-point count.  Only when
the bound lands above the released label does it pay for the solution file, to
recover the solver (which fixes the label quantum) and the stored tour's float64
length.  Everything else is a single ascent.

Cost: about 8 CPU-hours at k=1000, so roughly 35 minutes on 16 workers.

    python paper_tooling/hk1tree_trainval.py [--workers 16] [--budget 1000]

Output: paper_tooling/hk1tree_trainval_k<budget>.csv
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from held_karp_1tree import one_tree_bound  # noqa: E402

OUT = ROOT / "paper_tooling"
AUDIT = OUT / "reference_tour_audit.csv"
INSTANCES = ROOT / "instances"
SOLUTIONS = ROOT / "solutions"

_BIN_HEADER = struct.Struct("IIII")


def load_coords(name: str, n: int, d: int, grid: int) -> np.ndarray:
    """Coordinates from the ``.bin`` sibling; a stale header raises, never falls back."""
    bin_path = INSTANCES / f"{name}.bin"
    if not bin_path.exists():
        return np.asarray(json.loads((INSTANCES / f"{name}.json").read_text())["coordinates"],
                          dtype=np.float64)
    blob = bin_path.read_bytes()
    b_n, b_d, b_grid, dist_len = _BIN_HEADER.unpack_from(blob, 0)
    offset = _BIN_HEADER.size + dist_len
    if (b_n, b_d, b_grid) != (int(n), int(d), int(grid)):
        raise ValueError(f"{name}: stale header {(b_n, b_d, b_grid)}")
    if len(blob) - offset != b_n * b_d * 4:
        raise ValueError(f"{name}: truncated coordinate body")
    arr = np.frombuffer(blob, np.float32, count=b_n * b_d, offset=offset)
    return arr.reshape(b_n, b_d).astype(np.float64)


def worker(rec: dict) -> dict:
    name = rec["instance"]
    out = {"instance": name, "n": rec["n"], "d": rec["d"], "grid_size": rec["grid_size"],
           "split": rec["split"], "true_cost": rec["stored_cost"]}
    try:
        coords = load_coords(name, rec["n"], rec["d"], rec["grid_size"])
    except Exception as exc:                                        # noqa: BLE001
        out["status"] = f"load_error:{type(exc).__name__}"
        return out
    uniq = np.unique(coords, axis=0)
    if uniq.shape[0] < 3:
        out["status"] = "degenerate_n"
        return out
    res = one_tree_bound(uniq, rec["budget"])
    out.update(bound=res.bound, w0=res.initial_bound, is_optimal=res.is_optimal,
               n_unique=int(uniq.shape[0]), status="ok")

    if res.bound > float(rec["stored_cost"]):
        sp = SOLUTIONS / f"{name}.sol.json"
        if sp.exists():
            sol = json.loads(sp.read_text())
            out["solver"] = sol.get("optimal_solver")
            tour = sol.get("optimal_tour")
            if tour is not None:
                t = np.asarray(tour, dtype=np.int64)
                nn = coords.shape[0]
                if t.size == nn and t.min() == 1 and t.max() == nn:
                    t = t - 1
                if t.size == nn and sorted(t.tolist()) == list(range(nn)):
                    seg = coords[t] - coords[np.roll(t, -1)]
                    out["ref_tour_float64"] = float(np.sqrt((seg * seg).sum(axis=1)).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget", type=int, default=1000)
    args = ap.parse_args()

    a = pd.read_csv(AUDIT, usecols=["instance_name", "split", "n", "d", "grid_size",
                                    "stored_cost"]).rename(columns={"instance_name": "instance"})
    a = a[a.split.isin(("train", "val"))]
    # Heaviest first, so the n=1000 tail is not still running when the pool drains.
    recs = a.sort_values("n", ascending=False).to_dict("records")
    for r in recs:
        r["budget"] = args.budget
    print(f"[trainval] {len(recs)} instances at k={args.budget}, {args.workers} workers",
          flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, r) for r in recs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 5000 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(recs)}  {el:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    path = OUT / f"hk1tree_trainval_k{args.budget}.csv"
    df.to_csv(path, index=False)
    ok = df[df.status == "ok"]
    print(f"\nWrote {path}  ({len(df)} rows, {time.time() - t0:.0f}s)")
    print(f"  statuses: {df.status.value_counts().to_dict()}")
    print(f"  bound above the released label: {int((ok.bound > ok.true_cost).sum())}")
    print("  -> run paper_tooling/label_certificate.py to apply the quantum-aware test")


if __name__ == "__main__":
    main()
