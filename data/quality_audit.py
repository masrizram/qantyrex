"""Data quality audit + provenance for ingested historical datasets.

PASS/FAIL/WARNING per check. Never fabricates missing candles.
Records dataset hash + provenance for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from .validator import timeframe_ms


@dataclass
class QualityCheck:
    name: str
    status: str  # PASS / FAIL / WARNING / NOT_APPLICABLE
    detail: str = ""
    count: int = 0


@dataclass
class DataQualityReport:
    exchange: str
    symbol: str
    timeframe: str
    rows: int
    checks: List[QualityCheck] = field(default_factory=list)
    missing_ranges: List[List[int]] = field(default_factory=list)
    overall: str = "PASS"  # PASS / FAIL / WARNING
    audited_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [f"DATA QUALITY REPORT  {self.exchange} {self.symbol} {self.timeframe}  rows={self.rows}"]
        lines.append(f"Overall: {self.overall}")
        for c in self.checks:
            lines.append(f"  [{c.status}] {c.name}: {c.detail} (count={c.count})")
        if self.missing_ranges:
            lines.append(f"  missing_ranges: {len(self.missing_ranges)}")
        return "\n".join(lines)


@dataclass
class Provenance:
    exchange: str
    symbol: str
    timeframe: str
    retrieved_at: int
    dataset_hash: str
    row_count: int
    first_candle: int
    last_candle: int
    data_source: str
    configuration_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def dataset_hash(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return ""
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    present = [c for c in cols if c in df.columns]
    blob = df[present].round(8).astype(str).agg("|".join, axis=1).str.cat(sep="\n")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def audit(df: pd.DataFrame, exchange: str, symbol: str, timeframe: str,
          missing_ranges: Optional[List[List[int]]] = None) -> DataQualityReport:
    rep = DataQualityReport(exchange=exchange, symbol=symbol, timeframe=timeframe,
                            rows=int(len(df)), missing_ranges=missing_ranges or [])
    if len(df) == 0:
        rep.overall = "FAIL"
        rep.checks.append(QualityCheck("non_empty", "FAIL", "empty dataset", 1))
        return rep
    ts = df["timestamp"].to_numpy()
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    # monotonicity
    mono = int(np.sum(np.diff(ts) <= 0))
    rep.checks.append(QualityCheck("timestamp_monotonic", "PASS" if mono == 0 else "FAIL",
                                    f"{mono} non-increasing steps", mono))
    # duplicates
    dup = int(df["timestamp"].duplicated().sum())
    rep.checks.append(QualityCheck("duplicate_timestamps", "PASS" if dup == 0 else "FAIL",
                                    f"{dup} duplicates", dup))
    # OHLC consistency
    h_ge_o = int(np.sum(h < o)); rep.checks.append(QualityCheck("high_ge_open", "PASS" if h_ge_o == 0 else "FAIL", "", h_ge_o))
    h_ge_c = int(np.sum(h < c)); rep.checks.append(QualityCheck("high_ge_close", "PASS" if h_ge_c == 0 else "FAIL", "", h_ge_c))
    l_le_o = int(np.sum(l > o)); rep.checks.append(QualityCheck("low_le_open", "PASS" if l_le_o == 0 else "FAIL", "", l_le_o))
    l_le_c = int(np.sum(l > c)); rep.checks.append(QualityCheck("low_le_close", "PASS" if l_le_c == 0 else "FAIL", "", l_le_c))
    h_ge_l = int(np.sum(h < l)); rep.checks.append(QualityCheck("high_ge_low", "PASS" if h_ge_l == 0 else "FAIL", "", h_ge_l))
    # non-positive prices
    nonpos = int(np.sum((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)))
    rep.checks.append(QualityCheck("non_positive_prices", "PASS" if nonpos == 0 else "FAIL", "", nonpos))
    # invalid volume
    badvol = int(np.sum(v < 0))
    rep.checks.append(QualityCheck("invalid_volume", "PASS" if badvol == 0 else "FAIL", "", badvol))
    # NaN / inf
    nan = int(np.sum(~np.isfinite(o) | ~np.isfinite(h) | ~np.isfinite(l) | ~np.isfinite(c) | ~np.isfinite(v)))
    rep.checks.append(QualityCheck("nan_inf", "PASS" if nan == 0 else "FAIL", "", nan))
    # timeframe consistency
    tf_ms = timeframe_ms(timeframe)
    bad_tf = int(np.sum((np.diff(ts) % tf_ms) != 0))
    rep.checks.append(QualityCheck("timeframe_consistency", "PASS" if bad_tf == 0 else "WARNING",
                                    f"{bad_tf} non-multiple gaps", bad_tf))
    # missing candles (gaps)
    gaps = int(np.sum(np.diff(ts) > tf_ms))
    status = "PASS" if gaps == 0 else ("WARNING" if gaps < 10 else "FAIL")
    rep.checks.append(QualityCheck("missing_candles", status, f"{gaps} gap ranges", gaps))

    # overall: any FAIL -> FAIL, else WARNING if any WARNING, else PASS
    if any(ch.status == "FAIL" for ch in rep.checks):
        rep.overall = "FAIL"
    elif any(ch.status == "WARNING" for ch in rep.checks):
        rep.overall = "WARNING"
    else:
        rep.overall = "PASS"
    return rep


def make_provenance(df: pd.DataFrame, exchange: str, symbol: str, timeframe: str,
                    configuration_hash: str, data_source: str = "ccxt") -> Provenance:
    return Provenance(
        exchange=exchange, symbol=symbol, timeframe=timeframe,
        retrieved_at=int(time.time() * 1000), dataset_hash=dataset_hash(df),
        row_count=int(len(df)),
        first_candle=int(df["timestamp"].iloc[0]) if len(df) else 0,
        last_candle=int(df["timestamp"].iloc[-1]) if len(df) else 0,
        data_source=data_source, configuration_hash=configuration_hash,
    )
