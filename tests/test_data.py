"""Tests for the data engine: validator, cache, orderbook, feature store, synthetic data."""
import pytest
import numpy as np
import pandas as pd

from trading_bot.core.exceptions import DataQualityError
from trading_bot.core.models import Candle
from trading_bot.data.validator import DataValidator, timeframe_ms
from trading_bot.data.market_data import SyntheticDataGenerator, make_synthetic_dataframe
from trading_bot.data.cache import LRUCache
from trading_bot.data.feature_store import FeatureStore
from trading_bot.data.orderbook import OrderBookSnapshot, OrderBookLevel, snapshot_from_dict


# ---------- Validator ----------

def _c(ts, o, h, l, c, v=1.0):
    return Candle(symbol="BTC/USDT", timeframe="1h", timestamp=ts,
                  open=o, high=h, low=l, close=c, volume=v, closed=True)


def test_validator_accepts_clean_batch():
    dv = DataValidator("1h")
    candles = [_c(i * 3_600_000, 100, 101, 99, 100.5) for i in range(10)]
    out = dv.validate(candles)
    assert len(out) == 10


def test_validator_rejects_duplicate():
    dv = DataValidator("1h")
    candles = [_c(0, 100, 101, 99, 100), _c(0, 100, 101, 99, 100)]
    with pytest.raises(DataQualityError, match="Duplicate"):
        dv.validate(candles)


def test_validator_rejects_unordered():
    dv = DataValidator("1h")
    candles = [_c(3_600_000, 100, 101, 99, 100), _c(0, 100, 101, 99, 100)]
    with pytest.raises(DataQualityError, match="ascending"):
        dv.validate(candles)


def test_validator_rejects_ohlc_integrity():
    dv = DataValidator("1h")
    # high < low
    candles = [_c(0, 100, 99, 101, 100)]
    with pytest.raises(DataQualityError, match="low>high"):
        dv.validate(candles)


def test_validator_rejects_close_outside_range():
    dv = DataValidator("1h")
    candles = [_c(0, 100, 101, 99, 102)]  # close > high
    with pytest.raises(DataQualityError, match="close outside"):
        dv.validate(candles)


def test_validator_rejects_missing_candle_gap():
    dv = DataValidator("1h")
    candles = [_c(0, 100, 101, 99, 100), _c(3 * 3_600_000, 100, 101, 99, 100)]
    with pytest.raises(DataQualityError, match="Missing candle"):
        dv.validate(candles)


def test_validator_rejects_non_multiple_gap():
    dv = DataValidator("1h")
    candles = [_c(0, 100, 101, 99, 100), _c(3_600_000 + 1, 100, 101, 99, 100)]
    with pytest.raises(DataQualityError, match="not multiple"):
        dv.validate(candles)


def test_validator_rejects_stale_data():
    dv = DataValidator("1h", max_stale_seconds=10)
    candles = [_c(0, 100, 101, 99, 100)]
    with pytest.raises(DataQualityError, match="Stale"):
        dv.validate(candles, now_ms=60_000_000)  # far in the future


def test_validator_rejects_empty():
    dv = DataValidator("1h")
    with pytest.raises(DataQualityError, match="Empty"):
        dv.validate([])


def test_validator_rejects_nonfinite():
    dv = DataValidator("1h")
    candles = [_c(0, float("nan"), 101, 99, 100)]
    with pytest.raises(DataQualityError, match="non-finite"):
        dv.validate(candles)


def test_timeframe_ms_known_and_unknown():
    assert timeframe_ms("1h") == 3_600_000
    assert timeframe_ms("15m") == 900_000
    assert timeframe_ms("2h") == 7_200_000
    with pytest.raises(ValueError):
        timeframe_ms("garbage")


# ---------- Synthetic generator ----------

def test_synthetic_generator_validates():
    gen = SyntheticDataGenerator(timeframe="1h", seed=42)
    candles = gen.generate(300)
    dv = DataValidator("1h")
    out = dv.validate(candles)
    assert len(out) == 300
    # prices should be positive and finite
    closes = [c.close for c in out]
    assert all(np.isfinite(closes)) and all(c > 0 for c in closes)


def test_synthetic_generator_reproducible():
    a = SyntheticDataGenerator(seed=123).generate(50)
    b = SyntheticDataGenerator(seed=123).generate(50)
    assert [c.close for c in a] == [c.close for c in b]


def test_make_synthetic_dataframe_shape():
    df = make_synthetic_dataframe(n=250, tf="1h", seed=7)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 250
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns)


# ---------- LRU cache ----------

def test_lru_basic():
    c = LRUCache(capacity=3)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert c.get("a") == 1
    c.put("d", 4)  # evicts b (a was just accessed)
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("d") == 4


def test_lru_invalidate_and_len():
    c = LRUCache(capacity=4)
    c.put("x", 1); c.put("y", 2)
    assert len(c) == 2
    c.invalidate("x")
    assert c.get("x") is None
    assert len(c) == 1
    c.clear()
    assert len(c) == 0


def test_lru_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        LRUCache(capacity=0)


# ---------- Feature store ----------

def test_feature_store_put_get_up_to():
    fs = FeatureStore()
    df = pd.DataFrame({
        "timestamp": [0, 3600_000, 7200_000],
        "ema_fast": [1.0, 2.0, 3.0],
        "rsi": [50, 55, 60],
    })
    fs.put("BTC/USDT", "1h", df)
    assert fs.get("BTC/USDT", "1h") is not None
    sub = fs.up_to("BTC/USDT", "1h", 3600_000)
    assert len(sub) == 2  # rows with ts <= 3600_000
    latest = fs.latest("BTC/USDT", "1h")
    assert latest["rsi"] == 60


def test_feature_store_up_to_no_lookahead():
    fs = FeatureStore()
    df = pd.DataFrame({"timestamp": [0, 100], "x": [1, 2]})
    fs.put("X", "1h", df)
    sub = fs.up_to("X", "1h", 0)
    assert list(sub["timestamp"]) == [0]
    assert list(sub["x"]) == [1]


# ---------- Order book ----------

def test_orderbook_spread_and_liquidity():
    ob = OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=0,
        bids=[OrderBookLevel(100, 1.0), OrderBookLevel(99, 2.0)],
        asks=[OrderBookLevel(101, 1.0), OrderBookLevel(102, 1.5)],
    )
    assert ob.mid == 100.5
    assert ob.spread == 1.0
    assert abs(ob.spread_percent - (1.0 / 100.5 * 100)) < 1e-9
    assert ob.is_valid(max_spread_percent=2.0)
    assert not ob.is_valid(max_spread_percent=0.1)
    # liquidity within 2% of mid (98.49..102.51): all 4 levels
    liq = ob.liquidity_at_distance(depth_pct=2.0)
    assert liq == 5.5


def test_orderbook_from_dict():
    raw = {"bids": [["100", "1"]], "asks": [["101", "0.5"]]}
    ob = snapshot_from_dict("BTC/USDT", 0, raw)
    assert ob.best_bid == 100 and ob.best_ask == 101
