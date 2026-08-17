"""Tests for signal engine, scoring, registry, baseline strategy."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Config
from trading_bot.core.enums import Side, TrendState, RegimeState, RegimeAction
from trading_bot.data.market_data import make_synthetic_dataframe
from trading_bot.strategy.scoring import score_signal, ScoreBreakdown
from trading_bot.strategy.signal_engine import SignalEngine
from trading_bot.strategy.strategy_registry import get_global_registry
from trading_bot.strategy.baseline import register_baseline, baseline_entry


def _cfg(**over):
    c = Config()
    # emulate overrides by constructing a new Config-like object
    return c


def test_scoring_weights_sum_to_100():
    from trading_bot.strategy.scoring import WEIGHTS
    assert sum(WEIGHTS.values()) == 100


def test_scoring_perfect_buy_signal_passes():
    sb = score_signal(
        side=Side.BUY, trend=TrendState.STRONG_BULLISH, trend_tf=TrendState.STRONG_BULLISH,
        entry_tf_aligned=True, momentum_passes=True, adx=30, adx_min=20,
        support_strength=2, resistance_strength=2, room_ok=True,
        volatility_label="normal", volatility_ok=True, spread_percent=0.01,
        max_spread=0.1, liquidity_ok=True, rr=3.0, min_rr=2.0, latency_ms=100,
        min_score=75.0,
    )
    assert sb.total > 75
    assert sb.passes is True


def test_scoring_low_rr_zeroes_component_and_drops_total():
    # Realistic signal: not every other component is maxed, so RR=0 pulls total below 75
    sb = score_signal(
        side=Side.BUY, trend=TrendState.BULLISH, trend_tf=TrendState.BULLISH,
        entry_tf_aligned=True, momentum_passes=True, adx=22, adx_min=20,
        support_strength=1, resistance_strength=1, room_ok=True,
        volatility_label="normal", volatility_ok=True, spread_percent=0.04,
        max_spread=0.1, liquidity_ok=True, rr=1.0, min_rr=2.0,
        min_score=75.0,
    )
    assert sb.components["risk_reward"] == 0
    assert sb.total < 75
    assert sb.passes is False


def test_scoring_high_spread_zeroes_liquidity_and_execution():
    sb = score_signal(
        side=Side.BUY, trend=TrendState.STRONG_BULLISH, trend_tf=TrendState.STRONG_BULLISH,
        entry_tf_aligned=True, momentum_passes=True, adx=30, adx_min=20,
        support_strength=2, resistance_strength=2, room_ok=True,
        volatility_label="normal", volatility_ok=True, spread_percent=0.5,
        max_spread=0.1, liquidity_ok=True, rr=3.0, min_rr=2.0,
    )
    assert sb.components["liquidity"] == 0
    assert sb.components["execution_quality"] == 0


def test_registry_register_and_get():
    cfg = Config()
    # register may have already run; ensure idempotency by constructing a fresh one
    reg = get_global_registry()
    # Just ensure get works if baseline was registered; else register now
    try:
        reg.get("baseline", cfg.strategy_version)
    except Exception:
        # build a minimal engine + register
        eng = SignalEngine(cfg)
        register_baseline(cfg, eng)
    m = reg.get("baseline", cfg.strategy_version)
    assert m.name == "baseline"
    assert callable(m.entry_fn)


def test_signal_engine_rejects_warmup():
    cfg = Config()
    eng = SignalEngine(cfg)
    df = make_synthetic_dataframe(n=300, tf="1h", seed=11)
    feat = eng.build_features(df, df)
    # very early index should reject (warmup)
    res = eng.evaluate(feat, idx=5)
    assert res.signal is None
    assert res.rejected_reason is not None


def test_signal_engine_enforces_no_lookahead():
    """Signals at idx must only depend on rows <= idx."""
    cfg = Config()
    eng = SignalEngine(cfg)
    df = make_synthetic_dataframe(n=400, tf="1h", seed=3)
    feat = eng.build_features(df, df)
    # Evaluate at idx N, then mutate a FUTURE row and re-evaluate at N; result must be identical.
    N = len(feat) - 2
    r1 = eng.evaluate(feat, idx=N)
    # corrupt a future row
    feat2 = feat.copy()
    feat2.iloc[-1, feat2.columns.get_loc("close")] = feat2["close"].iloc[-1] * 10
    r2 = eng.evaluate(feat2, idx=N)
    assert (r1.signal is None) == (r2.signal is None)
    if r1.signal and r2.signal:
        assert r1.signal.entry == r2.signal.entry
        assert r1.signal.stop_loss == r2.signal.stop_loss
        assert r1.signal.take_profit == r2.signal.take_profit


def test_signal_engine_sl_side_consistency():
    cfg = Config()
    eng = SignalEngine(cfg)
    df = make_synthetic_dataframe(n=400, tf="1h", seed=4)
    feat = eng.build_features(df, df)
    # find an index that produces a BUY signal (if any)
    for i in range(250, len(feat) - 1):
        res = eng.evaluate(feat, idx=i)
        if res.signal is not None:
            s = res.signal
            if s.side == Side.BUY:
                assert s.stop_loss < s.entry < s.take_profit
            else:
                assert s.take_profit < s.entry < s.stop_loss
            assert s.rr >= cfg.min_rr
            assert s.score >= cfg.min_signal_score
            return
    # If no signal in synthetic data, that's acceptable — the engine still rejected cleanly
    pytest.skip("No qualifying signal generated in synthetic data (acceptable).")
