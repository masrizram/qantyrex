"""Regime coverage analysis: identify bull/bear/range/shock/recovery periods
within a dataset and report per-regime performance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class RegimeSegment:
    regime: str
    start_ts: int
    end_ts: int
    n_bars: int
    ret_pct: float
    vol_pct: float


@dataclass
class RegimeCoverage:
    symbol: str
    segments: List[RegimeSegment] = field(default_factory=list)
    by_regime: Dict[str, int] = field(default_factory=dict)
    coverage_complete: bool = False  # at least one of each major regime present

    def summary(self) -> str:
        lines = [f"REGIME COVERAGE  {self.symbol}  complete={self.coverage_complete}"]
        for r, n in self.by_regime.items():
            lines.append(f"  {r}: {n} bars")
        return "\n".join(lines)


def classify_regime_coverage(df: pd.DataFrame, symbol: str = "?",
                              lookback: int = 50) -> RegimeCoverage:
    """Tag each bar with a coarse regime using a rolling slope + vol window.
    Regimes: bull, bear, range, high_vol, low_vol, shock, recovery.
    """
    if len(df) < lookback * 2:
        return RegimeCoverage(symbol=symbol, segments=[], by_regime={},
                              coverage_complete=False)
    out = df.copy()
    close = out["close"].astype(float)
    log_ret = np.log(close).diff()
    rolling_ret = close.pct_change(lookback)
    rolling_vol = log_ret.rolling(lookback).std() * np.sqrt(lookback)
    vol_pct = rolling_vol * 100

    # shock: absolute return in a short window far exceeds recent vol
    short_ret = close.pct_change(5).abs()
    shock = (short_ret > 3 * rolling_vol) & (rolling_vol > 0)

    def _tag(row):
        r = row.get("rolling_ret", 0.0) or 0.0
        v = row.get("vol_pct", 0.0) or 0.0
        if row.get("shock", False):
            return "shock"
        if pd.isna(r) or pd.isna(v):
            return "unknown"
        if v > np.nanpercentile(vol_pct.dropna(), 85):
            return "high_vol"
        if v < np.nanpercentile(vol_pct.dropna(), 15):
            return "low_vol"
        if r > 0.05:
            return "bull"
        if r < -0.05:
            return "bear"
        if abs(r) < 0.01:
            return "range"
        return "recovery"

    out["rolling_ret"] = rolling_ret
    out["vol_pct"] = vol_pct
    out["shock"] = shock
    out["regime"] = out.apply(_tag, axis=1)

    segments: List[RegimeSegment] = []
    if len(out) == 0:
        return RegimeCoverage(symbol=symbol, coverage_complete=False)
    cur = out.iloc[0]["regime"]
    seg_start = int(out.iloc[0]["timestamp"])
    seg_rows = [out.iloc[0]]
    for i in range(1, len(out)):
        r = out.iloc[i]["regime"]
        if r != cur:
            segments.append(_make_segment(cur, seg_rows))
            cur = r
            seg_start = int(out.iloc[i]["timestamp"])
            seg_rows = [out.iloc[i]]
        else:
            seg_rows.append(out.iloc[i])
    segments.append(_make_segment(cur, seg_rows))

    by_regime: Dict[str, int] = {}
    for s in segments:
        by_regime[s.regime] = by_regime.get(s.regime, 0) + s.n_bars

    # require at least bull/bear/range coverage for "complete"
    have = set(by_regime.keys())
    complete = {"bull", "bear", "range"} <= have
    return RegimeCoverage(symbol=symbol, segments=segments, by_regime=by_regime,
                          coverage_complete=complete)


def _make_segment(regime: str, rows: list) -> RegimeSegment:
    closes = [r["close"] for r in rows]
    ret_pct = (closes[-1] / closes[0] - 1) * 100 if closes[0] > 0 else 0.0
    vol_pct = float(np.std(np.diff(np.log(closes))) * 100) if len(closes) > 2 else 0.0
    return RegimeSegment(
        regime=regime, start_ts=int(rows[0]["timestamp"]),
        end_ts=int(rows[-1]["timestamp"]), n_bars=len(rows),
        ret_pct=float(ret_pct), vol_pct=float(vol_pct),
    )


def per_regime_performance(trades: pd.DataFrame, df: pd.DataFrame) -> Dict[str, dict]:
    """Bucket closed trades by the regime active at trade entry and report metrics."""
    if trades is None or len(trades) == 0:
        return {}
    # build a timestamp -> regime map from df (coarse)
    cov = classify_regime_coverage(df)
    seg_map = []
    for s in cov.segments:
        seg_map.append((s.start_ts, s.end_ts, s.regime))
    seg_map.sort()

    def _regime_at(ts: int) -> str:
        for st, en, rg in seg_map:
            if st <= ts <= en:
                return rg
        return "unknown"

    out: Dict[str, dict] = {}
    if "timestamp" not in trades.columns:
        return out
    for _, t in trades.iterrows():
        rg = _regime_at(int(t["timestamp"]))
        d = out.setdefault(rg, {"trades": 0, "wins": 0, "pnl": 0.0, "pf": 0.0})
        d["trades"] += 1
        d["pnl"] += float(t.get("pnl", 0))
        if t.get("pnl", 0) > 0:
            d["wins"] += 1
    for rg, d in out.items():
        pos = max(0, d["pnl"]); neg = max(0, -d["pnl"])
        d["pf"] = float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0)
        d["win_rate"] = d["wins"] / d["trades"] if d["trades"] else 0.0
    return out
