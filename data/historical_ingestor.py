"""Historical data ingestion with pagination, retry, rate-limit handling,
resumable downloads, duplicate detection, and provenance.

Never silently discards failed pages — every fetch error is recorded.
Deterministic ordering and duplicate detection are enforced.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import pandas as pd

from ..core.exceptions import DataQualityError
from .validator import DataValidator, timeframe_ms


@dataclass
class FetchError:
    timestamp: int
    error: str


@dataclass
class IngestionReport:
    exchange: str
    symbol: str
    timeframe: str
    first_timestamp: Optional[int]
    last_timestamp: Optional[int]
    candle_count: int
    fetch_errors: List[FetchError] = field(default_factory=list)
    missing_ranges: List[List[int]] = field(default_factory=list)  # [start, end] ms
    configuration_hash: str = ""
    retrieved_at: int = field(default_factory=lambda: int(time.time() * 1000))
    resumed: bool = False
    data_source: str = "ccxt"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class IngestConfig:
    exchange: str = "gate"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    since_ms: Optional[int] = None
    until_ms: Optional[int] = None
    limit_per_page: int = 1000
    max_pages: int = 2000
    max_retries: int = 4
    retry_base_sleep: float = 0.5
    rate_limit_ms: int = 250
    cache_dir: Optional[str] = None
    page_size_hint: int = 1000


def _config_hash(cfg: IngestConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _dataset_hash(df: pd.DataFrame) -> str:
    """Stable hash of the candle payload for reproducibility."""
    if df is None or len(df) == 0:
        return ""
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    present = [c for c in cols if c in df.columns]
    blob = df[present].round(8).astype(str).agg("|".join, axis=1).str.cat(sep="\n")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class HistoricalIngestor:
    """Paginated, resumable, retry-safe OHLCV downloader.

    Uses ccxt's fetch_ohlcv with `since` + `limit` and advances the cursor by
    the last returned timestamp + 1 ms. Stops when `until_ms` is reached or the
    exchange stops returning newer rows.
    """

    def __init__(self, exchange, cfg: IngestConfig) -> None:
        self.exchange = exchange
        self.cfg = cfg
        self.cfg.configuration_hash = _config_hash(cfg)
        self._validator = DataValidator(cfg.timeframe)

    # ---------- public ----------
    def ingest(self) -> tuple[pd.DataFrame, IngestionReport]:
        report = IngestionReport(
            exchange=self.cfg.exchange, symbol=self.cfg.symbol,
            timeframe=self.cfg.timeframe, first_timestamp=None,
            last_timestamp=None, candle_count=0,
            configuration_hash=self.cfg.configuration_hash,
        )

        # resumable: load existing cache and continue from its last timestamp
        cache_path = self._cache_path()
        rows: list[list] = []
        cursor = self.cfg.since_ms
        resumed = False
        if cache_path and os.path.exists(cache_path):
            try:
                cached = pd.read_parquet(cache_path) if cache_path.endswith(".parquet") \
                    else pd.read_csv(cache_path)
                rows = cached[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
                if len(rows):
                    cursor = int(max(r[0] for r in rows)) + 1
                    resumed = True
                    report.resumed = True
            except Exception as e:
                report.fetch_errors.append(FetchError(int(time.time()*1000), f"cache_read: {e}"))

        page = 0
        while page < self.cfg.max_pages:
            if self.cfg.until_ms and cursor and cursor >= self.cfg.until_ms:
                break
            batch, err = self._fetch_page(cursor)
            if err is not None:
                report.fetch_errors.append(FetchError(int(time.time()*1000), err))
                # advance cursor to avoid an infinite loop on a persistent error
                if cursor is None:
                    break
                cursor += self.cfg.limit_per_page * timeframe_ms(self.cfg.timeframe)
                page += 1
                continue
            if not batch:
                break  # no more data
            rows.extend(batch)
            last_ts = max(r[0] for r in batch)
            cursor = last_ts + 1
            page += 1
            # rate-limit
            time.sleep(self.cfg.rate_limit_ms / 1000.0)

        # dedup + sort deterministically
        df = self._rows_to_df(rows)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(df):
            report.first_timestamp = int(df["timestamp"].iloc[0])
            report.last_timestamp = int(df["timestamp"].iloc[-1])
            report.candle_count = len(df)
            report.missing_ranges = self._find_missing_ranges(df)

        # persist cache
        if self.cfg.cache_dir and len(df):
            os.makedirs(self.cfg.cache_dir, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        return df, report

    # ---------- internals ----------
    def _fetch_page(self, since: Optional[int]) -> tuple[list[list], Optional[str]]:
        params = {"limit": self.cfg.limit_per_page}
        if since is not None:
            params["since"] = since
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                rows = self.exchange.fetch_ohlcv(self.cfg.symbol, self.cfg.timeframe, **params)
                return rows, None
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(self.cfg.retry_base_sleep * (2 ** attempt))
        return [], last_err or "unknown_fetch_error"

    def _rows_to_df(self, rows: list[list]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        for c in ["timestamp"]:
            df[c] = df[c].astype("int64")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype("float64")
        return df

    def _find_missing_ranges(self, df: pd.DataFrame) -> list[list[int]]:
        tf = timeframe_ms(self.cfg.timeframe)
        ts = df["timestamp"].to_numpy()
        gaps = []
        for i in range(1, len(ts)):
            diff = int(ts[i] - ts[i-1])
            if diff > tf:
                gaps.append([int(ts[i-1]) + tf, int(ts[i]) - tf])
        return gaps

    def _cache_path(self) -> Optional[str]:
        if not self.cfg.cache_dir:
            return None
        safe = f"{self.cfg.exchange}_{self.cfg.symbol.replace('/', '_')}_{self.cfg.timeframe}.parquet"
        return os.path.join(self.cfg.cache_dir, safe)


def dataset_hash(df: pd.DataFrame) -> str:
    return _dataset_hash(df)
