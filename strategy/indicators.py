"""Technical indicators implemented from scratch (vectorized, no look-ahead).

Every indicator uses only information available at or before time t.
EMAs use the standard recursive form where EMA_t depends on EMA_{t-1}
and the close at t — never on future values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. The first `period-1` values are NaN."""
    if period < 1:
        raise ValueError("EMA period must be >= 1")
    s = series.astype(float)
    # Use pandas' ewm with adjustment=False so EMA_t = alpha*x_t + (1-alpha)*EMA_{t-1}
    # and the seed is the first value. This matches the standard recursive definition
    # and produces NaN only for the warm-up via min_periods.
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing). No look-ahead."""
    if period < 2:
        raise ValueError("RSI period must be >= 2")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == ewm with alpha = 1/period, adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.clip(0.0, 100.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(H-L, |H-prevC|, |L-prevC|)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    a = atr(df, period)
    return (a / df["close"]) * 100.0


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder). Returns ADX series."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = (-low).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr = true_range(df)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_.replace(0.0, np.nan)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    adx_ = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_


def rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling std of log returns, annualized to per-candle units (not annual)."""
    log_ret = np.log(series).diff()
    return log_ret.rolling(window, min_periods=window).std()


def swing_highs_lows(df: pd.DataFrame, window: int = 3):
    """Detect swing highs/lows as local extrema over `window` bars each side.

    Returns two boolean Series: is_swing_high, is_swing_low.
    A point at index i is a swing high if high[i] is strictly the max of
    [i-window, i+window]. Uses only past+present at the center; the
    confirmation lag of `window` bars is enforced at the signal layer
    (a swing is only "confirmed" `window` bars later — no look-ahead).
    """
    high = df["high"]
    low = df["low"]
    n = len(df)
    is_sh = np.zeros(n, dtype=bool)
    is_sl = np.zeros(n, dtype=bool)
    arr_h = high.to_numpy(dtype=float)
    arr_l = low.to_numpy(dtype=float)
    for i in range(window, n - window):
        seg_h = arr_h[i - window:i + window + 1]
        seg_l = arr_l[i - window:i + window + 1]
        if np.argmax(seg_h) == window and seg_h[window] > seg_h[window - 1]:
            is_sh[i] = True
        if np.argmin(seg_l) == window and seg_l[window] < seg_l[window - 1]:
            is_sl[i] = True
    return pd.Series(is_sh, index=df.index), pd.Series(is_sl, index=df.index)


def ema_slope(series_ema: pd.Series, lookback: int = 5) -> pd.Series:
    """Slope of an EMA over `lookback` bars (per-bar delta)."""
    return series_ema.diff(lookback) / lookback


def percentile_rank(series: pd.Series, value: float, window: int = 100) -> pd.Series:
    """Rolling percentile rank of `value` relative to the trailing `window`."""
    def _rank(x):
        return (x <= value).sum() / len(x) * 100.0
    return series.rolling(window, min_periods=window).apply(_rank, raw=True)
