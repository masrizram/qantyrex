"""Market data provider with an exchange adapter and a synthetic generator.

The exchange adapter uses ccxt in paper/live mode. In backtest mode the
provider is fed an in-memory series (no network). All fetched candles pass
through the DataValidator before being returned to the rest of the system.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.clock import Clock
from ..core.exceptions import DataQualityError
from ..core.models import Candle
from .validator import DataValidator, timeframe_ms


class MarketDataProvider:
    """Wraps a ccxt-like exchange for OHLCV fetching with validation."""

    def __init__(self, exchange, symbol: str, timeframe: str,
                 validator: DataValidator, clock: Clock) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.validator = validator
        self.clock = clock

    def fetch_candles(self, limit: int = 500) -> List[Candle]:
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        candles = self._rows_to_candles(ohlcv)
        return self.validator.validate(candles, now_ms=self.clock.now_ms())

    def _rows_to_candles(self, rows) -> List[Candle]:
        out = []
        for r in rows:
            if len(r) < 5:
                raise DataQualityError(f"Malformed OHLCV row: {r}")
            ts, o, h, l, c, v = (list(r) + [0])[:6]
            out.append(Candle(
                symbol=self.symbol, timeframe=self.timeframe,
                timestamp=int(ts), open=float(o), high=float(h),
                low=float(l), close=float(c), volume=float(v),
                closed=True,
            ))
        return out


class SyntheticDataGenerator:
    """Geometric-Brownian-motion price generator for tests/backtests.

    NOT a model of real markets — used only to exercise the pipeline.
    """

    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h",
                 start_price: float = 100.0, annual_vol: float = 0.6,
                 annual_drift: float = 0.05, base_volume: float = 100.0,
                 seed: Optional[int] = None) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.tf_ms = timeframe_ms(timeframe)
        self.start_price = start_price
        self.tf_years = (self.tf_ms / 1000) / (365 * 24 * 3600)
        self.mu = annual_drift
        self.sigma = annual_vol
        self.base_volume = base_volume
        self.rng = np.random.default_rng(seed)

    def generate(self, n: int, start_ts: int = 0) -> List[Candle]:
        dt = self.tf_years
        # mean and std for log returns
        mu = (self.mu - 0.5 * self.sigma ** 2) * dt
        sd = self.sigma * np.sqrt(dt)
        log_ret = self.rng.normal(mu, sd, size=n)
        log_price = np.log(self.start_price) + np.cumsum(log_ret)
        closes = np.exp(log_price)
        # intrabar high/low via Brownian bridge approximation
        intrabar = self.rng.normal(0, sd, size=(n, 2))
        highs = closes * (1 + np.abs(intrabar[:, 0]))
        lows = closes * (1 - np.abs(intrabar[:, 1]))
        opens = np.empty(n)
        opens[0] = self.start_price
        opens[1:] = closes[:-1]
        # Guarantee OHLC integrity: open and close must lie within [low, high]
        highs = np.maximum.reduce([highs, opens, closes])
        lows = np.minimum.reduce([lows, opens, closes])
        vols = self.base_volume * (1 + 0.5 * np.abs(self.rng.normal(0, 1, size=n)))
        candles = []
        for i in range(n):
            candles.append(Candle(
                symbol=self.symbol, timeframe=self.timeframe,
                timestamp=start_ts + i * self.tf_ms,
                open=float(opens[i]), high=float(highs[i]),
                low=float(lows[i]), close=float(closes[i]),
                volume=float(vols[i]), closed=True,
            ))
        return candles


def make_synthetic_dataframe(n: int = 500, tf: str = "1h", seed: int = 0) -> pd.DataFrame:
    """Convenience helper returning a validated OHLCV DataFrame."""
    gen = SyntheticDataGenerator(timeframe=tf, seed=seed)
    candles = gen.generate(n)
    dv = DataValidator(timeframe=tf)
    return dv.to_dataframe(candles)
