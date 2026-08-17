"""Tests for monitoring: performance, degradation, health."""
import pytest

from trading_bot.monitoring.performance import RollingPerformance
from trading_bot.monitoring.degradation import DegradationMonitor, DegradationConfig
from trading_bot.monitoring.health import HealthMonitor, check_data_freshness


def test_rolling_performance_metrics():
    rp = RollingPerformance(window=5)
    for p in [10, -5, 20, -8, 12]:
        rp.add_trade(p, slippage=0.1, spread=0.2)
    snap = rp.snapshot()
    assert snap["n_trades"] == 5
    assert snap["win_rate"] == 3/5
    assert snap["profit_factor"] == 42 / 13
    assert snap["avg_slippage"] == 0.1


def test_rolling_performance_window_evicts():
    rp = RollingPerformance(window=3)
    for p in [10, -5, 20, -8, 12]:
        rp.add_trade(p)
    assert len(rp.pnls) == 3
    assert list(rp.pnls) == [20, -8, 12]


def test_degradation_insufficient_trades():
    dm = DegradationMonitor(DegradationConfig(oos_expectancy=0.05, oos_profit_factor=1.5))
    for _ in range(5):
        dm.add_trade(0.01)
    s = dm.evaluate()
    assert not s.degraded
    assert "insufficient" in s.reason
    assert s.risk_factor == 1.0


def test_degradation_lock_on_negative_expectancy():
    dm = DegradationMonitor(DegradationConfig(min_trades_to_evaluate=5, lock_threshold=0.0))
    for p in [-1, -2, -3, -4, -1]:
        dm.add_trade(p)
    s = dm.evaluate()
    assert s.severe and s.degraded
    assert s.risk_factor == 0.0


def test_degradation_reduce_when_below_fraction():
    dm = DegradationMonitor(DegradationConfig(
        oos_expectancy=0.10, min_trades_to_evaluate=10, reduce_threshold=0.5))
    # live expectancy ~0.03, which is < 0.5 * 0.10 = 0.05
    for _ in range(10):
        dm.add_trade(0.03)
    s = dm.evaluate()
    assert s.degraded
    assert 0 < s.risk_factor < 1.0


def test_degradation_ok_when_live_matches_oos():
    dm = DegradationMonitor(DegradationConfig(oos_expectancy=0.05, min_trades_to_evaluate=5))
    for _ in range(5):
        dm.add_trade(0.05)
    s = dm.evaluate()
    assert not s.degraded
    assert s.risk_factor == 1.0


def test_health_monitor_aggregates():
    hm = HealthMonitor()
    hm.update("exchange", True, "ok")
    hm.update("data", False, "stale 600s")
    assert not hm.all_healthy()
    assert "data" in hm.failing()
    assert "FAIL" in hm.summary()


def test_check_data_freshness():
    import time
    fresh_ts = int(time.time() * 1000) - 1000
    ok, _ = check_data_freshness(fresh_ts, max_age_seconds=600)
    assert ok is True
    stale_ts = int(time.time() * 1000) - 1_000_000
    ok, _ = check_data_freshness(stale_ts, max_age_seconds=600)
    assert ok is False
