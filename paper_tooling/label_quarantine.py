"""Hard guard around instances whose ground truth is not recoverable.

This project has shipped two silent zero-fill bugs, so a sentinel that reads as
data is the wrong answer here.  A quarantined instance therefore carries

  * an **empty** ``true_cost`` in every released results CSV, so any arithmetic
    that ignores the status column produces NaN rather than a plausible number,
    **and**
  * ``status = "quarantined_label_unrecoverable"``, which is outside
    ``build_paper_tables.OK_STATUS`` and so is dropped by every consumer that
    filters on status, **and**
  * ``assert_no_quarantined`` below, which turns "somebody scored one anyway"
    into an exception instead of a wrong table cell.

Belt, braces, and a tripwire.  Use ``assert_no_quarantined`` at the top of any
routine that aggregates per-instance error.

Membership comes from ``paper_tooling/labels_repaired.csv``, which is built by
``paper_tooling/repair_labels.py``.  Nothing hard-codes an instance name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P_LABELS = ROOT / "paper_tooling" / "labels_repaired.csv"

QUARANTINE_STATUS = "quarantined_label_unrecoverable"
QUARANTINE_LABEL_STATUS = "quarantined_no_recoverable_truth"


class QuarantinedLabelError(RuntimeError):
    """Raised when a quarantined instance reaches a scoring path."""


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    if not P_LABELS.exists():
        raise FileNotFoundError(
            f"{P_LABELS} is missing. Run: python paper_tooling/repair_labels.py")
    return pd.read_csv(P_LABELS)


@lru_cache(maxsize=1)
def quarantined() -> frozenset[str]:
    """Instance names with no recoverable ground truth, all corpora."""
    t = _table()
    return frozenset(t.loc[t["label_status"] == QUARANTINE_LABEL_STATUS,
                           "instance"].astype(str))


@lru_cache(maxsize=1)
def repaired_labels() -> dict[str, float]:
    """``instance -> repaired label``. Quarantined instances are absent."""
    t = _table()
    t = t[t["label_repaired"].notna()]
    return dict(zip(t["instance"].astype(str), t["label_repaired"].astype(float)))


def label_status() -> dict[str, str]:
    """``instance -> label_status``, for every instance in every corpus."""
    t = _table()
    return dict(zip(t["instance"].astype(str), t["label_status"].astype(str)))


def assert_no_quarantined(
    df: pd.DataFrame,
    *,
    instance_col: str = "instance",
    true_col: str = "true_cost",
    where: str = "<unnamed>",
) -> None:
    """Raise if any quarantined instance carries a usable ground truth here.

    A quarantined row that survives into ``df`` with an empty ``true_col`` is
    fine -- it will not contribute to any mean.  A quarantined row with a
    *number* in ``true_col`` means a repair was skipped or a stale CSV was
    read, and that is the failure this project keeps having.
    """
    if instance_col not in df.columns:
        return
    hit = df[instance_col].astype(str).isin(quarantined())
    if not hit.any():
        return
    if true_col in df.columns:
        live = hit & df[true_col].notna()
        if not live.any():
            return
        names = sorted(df.loc[live, instance_col].astype(str).unique())[:10]
        raise QuarantinedLabelError(
            f"{where}: {int(live.sum())} row(s) covering {len(names)}+ quarantined "
            f"instance(s) carry a non-null {true_col!r}. Ground truth for these is "
            f"not recoverable; they must not be scored. First: {names}")
    names = sorted(df.loc[hit, instance_col].astype(str).unique())[:10]
    raise QuarantinedLabelError(
        f"{where}: {int(hit.sum())} quarantined row(s) reached a scoring path and "
        f"the frame has no {true_col!r} column to prove they were blanked. "
        f"First: {names}")


def drop_quarantined(df: pd.DataFrame, *, instance_col: str = "instance") -> pd.DataFrame:
    """Return ``df`` without quarantined instances. Explicit, never implicit."""
    if instance_col not in df.columns:
        return df
    return df[~df[instance_col].astype(str).isin(quarantined())].copy()


if __name__ == "__main__":
    t = _table()
    q = t[t["label_status"] == QUARANTINE_LABEL_STATUS]
    print(f"labels_repaired.csv: {len(t):,} instances")
    print(f"quarantined        : {len(q):,}")
    print(q.groupby(["corpus", "split"]).size().to_string())
    print("\nlabel_status census:")
    print(t.groupby(["corpus", "label_status"]).size().to_string())
