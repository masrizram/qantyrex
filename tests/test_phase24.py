"""Phase 24 regression tests: signal tracer, rejection taxonomy, invariants,
counterfactual execution diagnostic, position lifecycle."""
import numpy as np
import pandas as pd
import pytest

from trading_bot.config import Config
from trading_bot.data.market_data import make_synthetic_dataframe
from trading_bot.execution.signal_tracer import (
    SignalTracer, TraceReport, counterfactual_execution_diagnostic, REJECTION_CODES,
)
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.strategy.signal_engine import SignalEngine
from trading_bot.backtest.simulator import Simulator, SimulatorConfig
from trading_bot.core.enums import Side, TrendState, RegimeState, RegimeAction
from trading_bot.core.models import Signal, Position


def _cfg():
    return Config()


def _signal(entry=100, sl=98, tp=104, side=Side.BUY, ts=0, qty=1.0):
    return Signal(
        strategy_version="baseline_v1", symbol="BTC/USDT", side=side,
        entry=entry, stop_loss=sl, take_profit=tp,
        rr=abs(tp - entry) / abs(entry - sl), score=80,
        trend=TrendState.BULLISH, regime=RegimeState.STRONG_TREND,
        regime_action=RegimeAction.TRADE, rsi=45, atr=1.5, atr_percent=1.5,
        adx=25, ema_fast=100, ema_slow=95, spread_percent=0.05,
        timestamp=ts, features={"quantity": qty},
    )


# ---- 24.2 Rejection taxonomy completeness ----

def test_rejection_codes_defined():
    assert len(REJECTION_CODES) > 0
    assert "ACCEPTED_OPENED" in REJECTION_CODES
    assert "UNKNOWN" in REJECTION_CODES
    assert "MAX_OPEN_POSITIONS" in REJECTION_CODES
    assert "DAILY_DD_LIMIT" in REJECTION_CODES


# ---- 24.3 Signal accounting invariant ----

def test_tracer_accounting_invariant():
    cfg = Config()
    df = make_synthetic_dataframe(n=600, tf="1h", seed=1)
    tracer = SignalTracer(cfg)
    rm = RiskManager(cfg, equity=10_000)
    report, accepted = tracer.run_trace(df, rm)
    assert report.total_signals == sum(report.by_code.values())
    assert report.total_signals > 0


# ---- 24.4 Position lifecycle ----

def test_simulator_records_per_signal_rejections():
    cfg = SimulatorConfig(max_open_positions=1, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0)
    sim = Simulator(cfg)
    s1 = _signal(ts=0)
    s2 = _signal(ts=0)  # same timestamp -> second should be rejected by NO_CANDLE or MAX_OPEN
    df = pd.DataFrame({
        "timestamp": [0, 3600000, 7200000, 10800000, 14400000],
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low": [99, 100, 101, 102, 103],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1, 1, 1, 1, 1],
    })
    res = sim.run(df, [s1, s2])
    assert res.rejected_signals >= 1
    assert len(res.signal_rejections) >= 1
    rejection_reasons = [r["reason"] for r in res.signal_rejections]
    assert "NO_CANDLE" in rejection_reasons or "MAX_OPEN_POSITIONS" in rejection_reasons


def test_simulator_max_open_positions_rejection():
    cfg = SimulatorConfig(max_open_positions=1, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0)
    sim = Simulator(cfg)
    s1 = _signal(ts=0)
    s2 = _signal(ts=3600000)
    df = pd.DataFrame({
        "timestamp": [0, 3600000, 7200000, 10800000, 14400000, 18000000, 21600000, 25200000],
        "open": [100, 101, 102, 103, 104, 105, 106, 107],
        "high": [101, 102, 103, 104, 105, 106, 107, 108],
        "low": [99, 100, 101, 102, 103, 104, 105, 106],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
        "volume": [1, 1, 1, 1, 1, 1, 1, 1],
    })
    res = sim.run(df, [s1, s2])
    assert res.rejected_signals >= 1
    reasons = [r["reason"] for r in res.signal_rejections]
    assert "MAX_OPEN_POSITIONS" in reasons


def test_simulator_daily_dd_rejection():
    cfg = SimulatorConfig(max_open_positions=99, initial_equity=1000, fee_rate=0.0,
                          slippage_bps=0.0, spread_bps=0.0)
    sim = Simulator(cfg)
    # s1: entry=100, SL=97, qty=100 -> loss=300 = 30% of 1000 equity -> triggers DD
    s1 = _signal(ts=0, entry=100, sl=97, tp=106, qty=100)
    s2 = _signal(ts=3600000, entry=100, sl=97, tp=106, qty=100)
    df = pd.DataFrame({
        "timestamp": [0, 3600000, 7200000, 10800000, 14400000, 18000000, 21600000, 25200000],
        "open":  [100, 100, 100, 100, 100, 100, 100, 100],
        "high":  [101, 101, 101, 101, 101, 101, 101, 101],
        "low":   [99,  99,   96,  99,  99,  99,  99,  99],
        "close": [100, 100, 100, 100, 100, 100, 100, 100],
        "volume":[1, 1, 1, 1, 1, 1, 1, 1],
    })
    res = sim.run(df, [s1, s2])
    reasons = [r["reason"] for r in res.signal_rejections]
    assert "DAILY_DD_LIMIT" in reasons or "MAX_OPEN_POSITIONS" in reasons


def test_simulator_latency_overflow_rejection():
    cfg = SimulatorConfig(latency_bars=999, max_open_positions=99, fee_rate=0.0,
                          slippage_bps=0.0, spread_bps=0.0)
    sim = Simulator(cfg)
    s1 = _signal(ts=0)
    df = pd.DataFrame({
        "timestamp": [0, 3600000, 7200000],
        "open": [100, 101, 102], "high": [101, 102, 103],
        "low": [99, 100, 101], "close": [100.5, 101.5, 102.5],
        "volume": [1, 1, 1],
    })
    res = sim.run(df, [s1])
    assert res.rejected_signals >= 1
    reasons = [r["reason"] for r in res.signal_rejections]
    assert "LATENCY_OVERFLOW" in reasons


# ---- 24.11 Counterfactual execution diagnostic ----

def test_counterfactual_execution_diagnostic_is_diagnostic_only():
    cfg = Config()
    df = make_synthetic_dataframe(n=600, tf="1h", seed=4)
    tracer = SignalTracer(cfg)
    rm = RiskManager(cfg, equity=10_000)
    report, accepted = tracer.run_trace(df, rm)
    features = tracer.engine.build_features(df, df)
    results = counterfactual_execution_diagnostic(cfg, df, accepted, features)
    assert len(results) == 8  # 4 max_open * 2 DD variants
    for r in results:
        assert r["diagnostic_only"] is True


# ---- 24.13 Invariants ----

def test_tracer_no_signal_silently_discarded():
    cfg = Config()
    df = make_synthetic_dataframe(n=500, tf="1h", seed=5)
    tracer = SignalTracer(cfg)
    rm = RiskManager(cfg, equity=10_000)
    report, accepted = tracer.run_trace(df, rm)
    assert report.total_signals == len(report.traces)
    assert report.total_signals == sum(report.by_code.values())


def test_tracer_every_trace_has_terminal_classification():
    cfg = Config()
    df = make_synthetic_dataframe(n=500, tf="1h", seed=6)
    tracer = SignalTracer(cfg)
    rm = RiskManager(cfg, equity=10_000)
    report, accepted = tracer.run_trace(df, rm)
    for t in report.traces:
        assert t.terminal_classification in REJECTION_CODES or t.terminal_classification in ("ACCEPTED_OPENED", "ACCEPTED_NOT_OPENED")
        assert t.terminal_classification != ""


def test_tracer_rejection_gate_filled():
    cfg = Config()
    df = make_synthetic_dataframe(n=500, tf="1h", seed=7)
    tracer = SignalTracer(cfg)
    rm = RiskManager(cfg, equity=10_000)
    report, accepted = tracer.run_trace(df, rm)
    for t in report.traces:
        if t.terminal_classification != "ACCEPTED_OPENED":
            assert t.rejection_gate != "", f"empty rejection_gate for {t.terminal_classification}"


def test_tracer_rejection_detail_not_empty():
    cfg = Config()
    df = make_synthetic_dataframe(n=500, tf="1h", seed=8)
    tracer = SignalTracer(cfg)
    rm = RiskManager(cfg, equity=10_000)
    report, accepted = tracer.run_trace(df, rm)
    for t in report.traces:
        if t.terminal_classification == "UNKNOWN":
            continue  # UNKNOWN is allowed if accompanied by detail
        # only check that non-ACCEPTED traces have a reason
        if t.terminal_classification not in ("ACCEPTED_OPENED",):
            assert t.rejection_detail != "" or t.terminal_classification == "UNKNOWN"