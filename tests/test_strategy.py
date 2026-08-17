"""Tests for indicators, trend, momentum, volatility, regime, support/resistance."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.strategy import indicators as ind
from trading_bot.strategy.trend import classify_trend, is_bullish, is_bearish
from trading_bot.strategy.momentum import MomentumConfig, add_momentum, buy_momentum, sell_momentum
from trading_bot.strategy.volatility import VolatilityConfig, add_volatility, volatility_status
from trading_bot.strategy.regime import classify_regime, action_for, RegimeConfig, RegimeState
from trading_bot.strategy.support_resistance import SRConfig, add_support_resistance, room_for_tp
from trading_bot.data.market_data import make_synthetic_dataframe


def _uptrend_df(n=300, seed=1):
    df = make_synthetic_dataframe(n=n, tf="1h", seed=seed)
    return df


# ---- indicators ----

def test_ema_no_lookahead_and_warmup():
    s = pd.Series(np.arange(1.0, 51.0))
    e = ind.ema(s, 10)
    # first 9 values NaN (min_periods=10)
    assert e.iloc[:9].isna().all()
    assert e.iloc[9] == pytest.approx(s.ewm(span=10, adjust=False, min_periods=10).mean().iloc[9])
    # EMA at t must equal alpha*x_t + (1-alpha)*EMA_{t-1}
    alpha = 2 / 11
    assert e.iloc[10] == pytest.approx(alpha * s.iloc[10] + (1 - alpha) * e.iloc[9])


def test_rsi_bounds_and_warmup():
    df = _uptrend_df()
    r = ind.rsi(df["close"], 14)
    assert r.iloc[:13].isna().all()
    assert ((r.dropna() >= 0) & (r.dropna() <= 100)).all()


def test_atr_positive_and_warmup():
    df = _uptrend_df()
    a = ind.atr(df, 14)
    assert a.iloc[:13].isna().all()
    assert (a.dropna() > 0).all()


def test_macd_components():
    df = _uptrend_df()
    m, s, h = ind.macd(df["close"], 12, 26, 9)
    assert len(m) == len(df)
    # histogram = macd - signal
    mask = m.notna() & s.notna()
    np.testing.assert_allclose(h[mask].to_numpy(), (m - s)[mask].to_numpy(), rtol=1e-9)


def test_adx_finite_and_nonnegative():
    df = _uptrend_df()
    a = ind.adx(df, 14)
    a_ = a.dropna()
    assert (a_ >= 0).all() and (a_ <= 100).all()


def test_swing_highs_lows_detects_extrema():
    # build a zigzag series
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    closes = np.linspace(100, 110, n)
    highs = closes + 1
    lows = closes - 1
    # inject a clear peak at i=10
    highs[10] = closes[10] + 5
    lows[10] = closes[10] - 5  # also a swing low candidate
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes,
                       "volume": 1.0}, index=idx)
    is_sh, is_sl = ind.swing_highs_lows(df, window=3)
    assert is_sh.iloc[10]  # peak detected
    assert is_sl.iloc[10]  # also lowest locally


# ---- trend ----

def test_trend_classifies_uptrend_as_bullish():
    # construct a strong uptrend
    n = 250
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    closes = 100 * (1.001 ** np.arange(n))
    df = pd.DataFrame({"open": closes, "high": closes * 1.005,
                       "low": closes * 0.995, "close": closes,
                       "volume": 1.0}, index=idx)
    t = classify_trend(df, ema_fast=50, ema_slow=200)
    last = t.iloc[-1]
    assert last["trend_state"].value in ("BULLISH", "STRONG_BULLISH")


def test_trend_warmup_returns_neutral():
    df = _uptrend_df(n=50)
    t = classify_trend(df, 50, 200)
    assert t.iloc[0]["trend_state"].value == "NEUTRAL"


# ---- momentum ----

def test_momentum_warmup_rejects():
    df = _uptrend_df(n=50)
    m = add_momentum(df)
    cfg = MomentumConfig()
    ok, reason = buy_momentum(m.iloc[-1], cfg)
    # not enough warmup for ADX/EMA200 -> warmup
    assert ok is False


def test_buy_momentum_requires_rsi_rising():
    row = pd.Series({
        "rsi": 45.0, "rsi_prev": 40.0, "adx": 25.0,
        "macd": 1.0, "macd_signal": 0.5,
    })
    ok, _ = buy_momentum(row, MomentumConfig())
    assert ok is True
    row2 = row.copy(); row2["rsi_prev"] = 50.0  # rsi falling
    ok, reason = buy_momentum(row2, MomentumConfig())
    assert ok is False and "rsi_not_rising" in reason


def test_sell_momentum_logic():
    row = pd.Series({
        "rsi": 55.0, "rsi_prev": 60.0, "adx": 25.0,
        "macd": -1.0, "macd_signal": 0.5,
    })
    ok, _ = sell_momentum(row, MomentumConfig())
    assert ok is True


# ---- volatility ----

def test_volatility_rejects_too_low():
    cfg = VolatilityConfig(atr_min_percent=0.2, atr_max_percent=3.0)
    row = pd.Series({"atr_percent": 0.1, "atr": 0.1, "rolling_vol": 0.001})
    label, ok = volatility_status(row, cfg)
    assert ok is False and label == "too_low"


def test_volatility_rejects_too_high():
    cfg = VolatilityConfig(atr_min_percent=0.2, atr_max_percent=3.0)
    row = pd.Series({"atr_percent": 5.0, "atr": 5.0, "rolling_vol": 0.01})
    label, ok = volatility_status(row, cfg)
    assert ok is False and label == "too_high"


def test_volatility_normal():
    cfg = VolatilityConfig(atr_min_percent=0.2, atr_max_percent=3.0)
    row = pd.Series({"atr_percent": 1.0, "atr": 1.0, "rolling_vol": 0.005})
    label, ok = volatility_status(row, cfg)
    assert ok is True and label == "normal"


# ---- regime ----

def test_regime_returns_action_mapping():
    assert action_for(RegimeState.STRONG_TREND).value == "TRADE"
    assert action_for(RegimeState.RANGE).value == "NO_TRADE"
    assert action_for(RegimeState.UNKNOWN).value == "NO_TRADE"


def test_regime_classifies_synthetic_data():
    df = _uptrend_df(n=300, seed=5)
    r = classify_regime(df)
    assert "regime_state" in r.columns
    # at least the last bar should be a valid regime (not all UNKNOWN)
    valid = [s for s in r["regime_state"] if s != RegimeState.UNKNOWN]
    assert len(valid) > 0


# ---- support / resistance ----

def test_sr_adds_columns_and_distances():
    df = _uptrend_df(n=200, seed=2)
    sr = add_support_resistance(df)
    assert "nearest_support" in sr.columns
    assert "nearest_resistance" in sr.columns
    # distances are percentages
    last = sr.iloc[-1]
    if not np.isnan(last["nearest_support"]):
        assert last["distance_to_support_pct"] >= 0  # support below price in uptrend


def test_room_for_tp_blocks_when_resistance_close():
    # BUY: entry 100, tp 105, resistance at 104 -> blocked
    ok, reason = room_for_tp(100, 105, "BUY", 104, 99, min_room_pct=0.5)
    assert ok is False
    # BUY: resistance at 110 -> ok
    ok2, _ = room_for_tp(100, 105, "BUY", 110, 99, min_room_pct=0.5)
    assert ok2 is True


def test_room_for_tp_sell_blocked_by_support():
    ok, _ = room_for_tp(100, 95, "SELL", 110, 96, min_room_pct=0.5)
    assert ok is False
    ok2, _ = room_for_tp(100, 95, "SELL", 110, 90, min_room_pct=0.5)
    assert ok2 is True
