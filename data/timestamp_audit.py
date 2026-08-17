"""Timestamp provenance audit (Phase 23.1).

Verifies that persisted ms-epoch timestamps convert correctly to the
human-readable UTC dates claimed in provenance reports, and that a dataset's
claimed first/last/duration match the actual data.

This module exists specifically to catch the kind of report-prose error that
occurred in Phase 22 (the audit text said "2025-03-22" while the timestamp
actually mapped to 2026-04-09).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd


@dataclass
class TimestampCheck:
    name: str
    status: str  # PASS / FAIL
    detail: str = ""


@dataclass
class TimestampAudit:
    first_ts: Optional[int]
    last_ts: Optional[int]
    first_utc: Optional[str]
    last_utc: Optional[str]
    duration_days: float
    checks: List[TimestampCheck] = field(default_factory=list)
    overall: str = "PASS"  # PASS / FAIL

    def summary(self) -> str:
        lines = ["TIMESTAMP PROVENANCE AUDIT"]
        lines.append(f"  first_ts={self.first_ts}  -> {self.first_utc}")
        lines.append(f"  last_ts ={self.last_ts}  -> {self.last_utc}")
        lines.append(f"  duration_days={self.duration_days:.2f}")
        lines.append(f"  overall={self.overall}")
        for c in self.checks:
            lines.append(f"  [{c.status}] {c.name}: {c.detail}")
        return "\n".join(lines)


def ts_to_utc_iso(ts_ms: int) -> str:
    """Convert a millisecond epoch to an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def ts_to_utc_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def audit_timestamps(df: pd.DataFrame,
                      claimed_first_date: Optional[str] = None,
                      claimed_last_date: Optional[str] = None,
                      timeframe: str = "1h") -> TimestampAudit:
    """Verify timestamp ordering, monotonicity, and (optionally) that the
    claimed first/last human-readable dates match the actual timestamps.

    `claimed_first_date` / `claimed_last_date` are the dates a report CLAIMS
    (e.g. "2025-03-22"). If they disagree with the actual timestamp -> FAIL.
    """
    checks: List[TimestampCheck] = []
    if len(df) == 0:
        return TimestampAudit(None, None, None, None, 0.0,
                              checks=[TimestampCheck("non_empty", "FAIL", "empty dataset")],
                              overall="FAIL")
    ts = df["timestamp"].astype("int64").to_numpy()
    first_ts = int(ts[0])
    last_ts = int(ts[-1])
    first_utc = ts_to_utc_iso(first_ts)
    last_utc = ts_to_utc_iso(last_ts)
    duration_days = (last_ts - first_ts) / (1000 * 3600 * 24)

    # monotonicity
    non_mono = int(sum(1 for i in range(1, len(ts)) if ts[i] <= ts[i - 1]))
    checks.append(TimestampCheck("monotonic_strictly_increasing",
                                 "PASS" if non_mono == 0 else "FAIL",
                                 f"{non_mono} non-increasing steps"))
    # first < last
    checks.append(TimestampCheck("first_before_last",
                                 "PASS" if first_ts < last_ts else "FAIL",
                                 f"{first_ts} < {last_ts}"))
    # duration positive
    checks.append(TimestampCheck("duration_positive",
                                 "PASS" if duration_days > 0 else "FAIL",
                                 f"{duration_days:.2f} days"))
    # claimed date verification
    if claimed_first_date is not None:
        actual_first = ts_to_utc_date(first_ts)
        ok = actual_first == claimed_first_date
        checks.append(TimestampCheck("claimed_first_date_matches",
                                     "PASS" if ok else "FAIL",
                                     f"claimed={claimed_first_date} actual={actual_first}"))
    if claimed_last_date is not None:
        actual_last = ts_to_utc_date(last_ts)
        ok = actual_last == claimed_last_date
        checks.append(TimestampCheck("claimed_last_date_matches",
                                     "PASS" if ok else "FAIL",
                                     f"claimed={claimed_last_date} actual={actual_last}"))
    # timeframe spacing sanity (first diff should equal timeframe)
    from .validator import timeframe_ms
    tf = timeframe_ms(timeframe)
    first_diff = int(ts[1] - ts[0]) if len(ts) > 1 else 0
    checks.append(TimestampCheck("first_gap_matches_timeframe",
                                 "PASS" if first_diff == tf else "WARNING",
                                 f"first_gap={first_diff} tf={tf}"))

    overall = "FAIL" if any(c.status == "FAIL" for c in checks) else "PASS"
    return TimestampAudit(first_ts=first_ts, last_ts=last_ts,
                          first_utc=first_utc, last_utc=last_utc,
                          duration_days=duration_days, checks=checks,
                          overall=overall)
