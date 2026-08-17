"""Signal scoring engine. Score is NOT win probability — it's a quality gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..core.enums import RegimeAction, RegimeState, Side, TrendState


# Weights from the spec; sum to 100.
WEIGHTS = {
    "trend": 20,
    "momentum": 15,
    "multi_timeframe": 15,
    "market_structure": 15,
    "volatility": 10,
    "liquidity": 5,
    "risk_reward": 10,
    "execution_quality": 10,
}
assert sum(WEIGHTS.values()) == 100


@dataclass
class ScoreBreakdown:
    total: float
    components: Dict[str, float]
    passes: bool


def _trend_score(trend: TrendState) -> float:
    return {
        TrendState.STRONG_BULLISH: 20.0, TrendState.BULLISH: 14.0,
        TrendState.NEUTRAL: 5.0,
        TrendState.BEARISH: 0.0, TrendState.STRONG_BEARISH: 0.0,
    }.get(trend, 0.0)


def _momentum_score(passes: bool, adx: float, adx_min: float) -> float:
    if not passes:
        return 0.0
    # scale 0..15 by how strongly ADX exceeds the minimum
    return min(15.0, 8.0 + (adx - adx_min) / max(adx_min, 1e-9) * 7.0)


def _mtf_score(trend_tf: TrendState, entry_tf_aligned: bool) -> float:
    base = {
        TrendState.STRONG_BULLISH: 15.0, TrendState.BULLISH: 11.0,
        TrendState.NEUTRAL: 5.0,
        TrendState.BEARISH: 0.0, TrendState.STRONG_BEARISH: 0.0,
    }.get(trend_tf, 0.0)
    return base if entry_tf_aligned else base * 0.5


def _structure_score(support_strength: int, resistance_strength: int,
                     room_ok: bool) -> float:
    if not room_ok:
        return 0.0
    return min(15.0, 5.0 + support_strength + resistance_strength)


def _volatility_score(label: str, ok: bool) -> float:
    if not ok:
        return 0.0
    return 10.0 if label == "normal" else 5.0


def _liquidity_score(spread_percent: float, max_spread: float,
                     liquidity_ok: bool) -> float:
    if not liquidity_ok or spread_percent > max_spread:
        return 0.0
    # tighter spread -> higher score
    return min(5.0, max(0.0, 5.0 * (1 - spread_percent / max(max_spread, 1e-9))))


def _rr_score(rr: float, min_rr: float) -> float:
    if rr < min_rr:
        return 0.0
    return min(10.0, 5.0 + (rr - min_rr) * 2.0)


def _execution_score(spread_ok: bool, latency_ms: float | None,
                     max_latency_ms: float = 2000.0) -> float:
    base = 10.0 if spread_ok else 0.0
    if latency_ms is not None:
        base = min(base, max(0.0, 10.0 * (1 - latency_ms / max_latency_ms)))
    return base


def score_signal(
    side: Side,
    trend: TrendState,
    trend_tf: TrendState,
    entry_tf_aligned: bool,
    momentum_passes: bool,
    adx: float,
    adx_min: float,
    support_strength: int,
    resistance_strength: int,
    room_ok: bool,
    volatility_label: str,
    volatility_ok: bool,
    spread_percent: float,
    max_spread: float,
    liquidity_ok: bool,
    rr: float,
    min_rr: float,
    latency_ms: float | None = None,
    min_score: float = 75.0,
) -> ScoreBreakdown:
    components = {
        "trend": _trend_score(trend) if side == Side.BUY else (20.0 - _trend_score(trend)),
        "momentum": _momentum_score(momentum_passes, adx, adx_min),
        "multi_timeframe": _mtf_score(trend_tf, entry_tf_aligned)
            if side == Side.BUY else (15.0 - _mtf_score(trend_tf, entry_tf_aligned)),
        "market_structure": _structure_score(support_strength, resistance_strength, room_ok),
        "volatility": _volatility_score(volatility_label, volatility_ok),
        "liquidity": _liquidity_score(spread_percent, max_spread, liquidity_ok),
        "risk_reward": _rr_score(rr, min_rr),
        "execution_quality": _execution_score(spread_percent <= max_spread, latency_ms),
    }
    # Clamp all components to [0, weight]
    for k, v in components.items():
        components[k] = max(0.0, min(float(WEIGHTS[k]), v))
    total = sum(components.values())
    return ScoreBreakdown(total=total, components=components, passes=total >= min_score)
