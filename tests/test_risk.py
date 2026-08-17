"""Tests for risk engine: position sizing, drawdown, exposure, portfolio, circuit breakers."""
import numpy as np
import pytest

from trading_bot.config import Config
from trading_bot.core.enums import Side, SystemState, TrendState, RegimeState, RegimeAction
from trading_bot.core.exceptions import RiskLimitBreached
from trading_bot.core.models import Signal
from trading_bot.risk.position_sizer import SizingInputs, size_position
from trading_bot.risk.drawdown import DrawdownMonitor, DrawdownConfig
from trading_bot.risk.exposure import ExposureMonitor, ExposureConfig, PositionExposure
from trading_bot.risk.portfolio_risk import PortfolioRisk, ConsecutiveLossState
from trading_bot.risk.circuit_breaker import CircuitBreaker
from trading_bot.risk.risk_manager import RiskManager


# ---- Position sizing ----

def test_size_position_basic():
    sin = SizingInputs(equity=10_000, risk_percent=0.005, entry=100, stop_loss=98)
    r = size_position(sin, max_risk_percent=0.01)
    assert r.quantity > 0
    # risk_capital = 50, price_risk = 2 -> raw qty = 25
    assert abs(r.quantity - 25.0) < 1.0
    assert r.expected_loss <= 50 * 1.001  # within budget + tiny cost


def test_size_position_rejects_risk_above_max():
    sin = SizingInputs(equity=10_000, risk_percent=0.02, entry=100, stop_loss=98)
    r = size_position(sin, max_risk_percent=0.01)
    assert r.quantity == 0
    assert "max" in r.reason


def test_size_position_rejects_zero_risk():
    sin = SizingInputs(equity=10_000, risk_percent=0.0, entry=100, stop_loss=98)
    r = size_position(sin, max_risk_percent=0.01)
    assert r.quantity == 0
    assert "zero" in r.reason


def test_size_position_rejects_when_below_min_qty():
    sin = SizingInputs(equity=10, risk_percent=0.001, entry=100, stop_loss=98, min_qty=1.0)
    r = size_position(sin, max_risk_percent=0.01)
    assert r.quantity == 0
    assert "min_qty" in r.reason


def test_size_position_cost_aware_reduces_qty():
    # with high fees+slippage, qty must be smaller than naive raw_qty
    base = SizingInputs(equity=10_000, risk_percent=0.005, entry=100, stop_loss=98)
    costly = SizingInputs(equity=10_000, risk_percent=0.005, entry=100, stop_loss=98,
                          fee_rate=0.005, slippage_bps=10)
    r_base = size_position(base, max_risk_percent=0.01)
    r_cost = size_position(costly, max_risk_percent=0.01)
    assert r_cost.quantity < r_base.quantity
    assert r_cost.expected_loss <= 50 * 1.0001


# ---- Drawdown ----

def test_dd_basic_and_rollover():
    m = DrawdownMonitor(10_000)
    m.update(9_800)
    assert abs(m.state.daily_dd - 0.02) < 1e-9
    assert not m.daily_dd_exceeded()
    m.update(9_600)
    assert m.daily_dd_exceeded()  # 4% >= 3%


def test_dd_emergency_raises():
    m = DrawdownMonitor(10_000)
    m.update(9_400)  # 6% DD
    with pytest.raises(RiskLimitBreached, match="Emergency"):
        m.check()


def test_dd_risk_factor_tiers():
    m = DrawdownMonitor(10_000, DrawdownConfig(daily_max_dd=0.03, emergency_dd=0.05))
    m.update(10_000)
    assert m.risk_factor() == 1.0  # normal
    m.update(9_800)  # 2% -> moderate tier
    assert m.risk_factor() <= 0.7
    m.update(9_700)  # 3% -> lock tier
    assert m.risk_factor() == 0.0


# ---- Exposure ----

def test_exposure_leverage_violation():
    e = ExposureMonitor(10_000, ExposureConfig(max_gross_leverage=1.0))
    e.set_positions([PositionExposure("BTC/USDT", "BUY", 15_000)])
    v = e.violates()
    assert any("leverage" in r for r in v)


def test_exposure_correlation_cluster():
    e = ExposureMonitor(10_000, ExposureConfig(max_correlated_weight=0.6,
                                               correlation_threshold=0.7))
    e.set_positions([
        PositionExposure("BTC/USDT", "BUY", 4_000),
        PositionExposure("ETH/USDT", "BUY", 3_000),  # correlated, total 0.7
    ])
    e.set_correlation(["BTC/USDT", "ETH/USDT"], np.array([[1.0, 0.85], [0.85, 1.0]]))
    v = e.violates()
    assert any("correlated" in r for r in v)


# ---- Portfolio / consecutive losses ----

def test_consecutive_losses_lock():
    c = ConsecutiveLossState(max_consecutive_losses=4)
    for _ in range(4):
        c.on_loss()
    assert c.exceeded()
    assert c.max_streak == 4


def test_portfolio_allowed_to_trade():
    p = PortfolioRisk.default(10_000, max_consecutive_losses=4)
    assert p.allowed_to_trade(open_positions=0, max_open=1)
    for _ in range(4):
        p.register_loss()
    assert not p.allowed_to_trade(open_positions=0, max_open=1)


def test_portfolio_risk_factor_reduces_on_near_streak():
    p = PortfolioRisk.default(10_000, max_consecutive_losses=4)
    for _ in range(3):
        p.register_loss()
    assert p.risk_factor() <= 0.5


# ---- Circuit breaker ----

def test_circuit_breaker_lock_and_shutdown():
    cb = CircuitBreaker()
    cb.register("ok_check", lambda: (True, ""))
    cb.register("hard_check", lambda: (False, "api_down"), severity="shutdown")
    assert cb.desired_state() == SystemState.SHUTDOWN

    # Mutable stateful check: tripped until condition clears, then RUNNING
    cb2 = CircuitBreaker()
    state = {"tripped": True}
    def dd_check():
        return (not state["tripped"], "daily_dd" if state["tripped"] else "")
    cb2.register("dd_check", dd_check)
    assert cb2.desired_state() == SystemState.RISK_LOCK
    state["tripped"] = False  # underlying condition clears
    assert cb2.desired_state() == SystemState.RUNNING


# ---- Risk manager integration ----

def _signal(entry, sl, tp, side=Side.BUY):
    return Signal(
        strategy_version="baseline_v1", symbol="BTC/USDT", side=side,
        entry=entry, stop_loss=sl, take_profit=tp, rr=abs(tp-entry)/abs(entry-sl),
        score=80, trend=TrendState.BULLISH, regime=RegimeState.STRONG_TREND,
        regime_action=RegimeAction.TRADE, rsi=45, atr=1.5, atr_percent=1.5,
        adx=25, ema_fast=100, ema_slow=95, spread_percent=0.05,
    )


def test_risk_manager_allows_normal_trade():
    cfg = Config()
    rm = RiskManager(cfg, equity=10_000)
    d = rm.evaluate_signal(_signal(100, 98, 104), open_positions=0)
    assert d.allowed
    assert d.sizing.quantity > 0


def test_risk_manager_blocks_on_max_positions():
    cfg = Config()
    rm = RiskManager(cfg, equity=10_000)
    d = rm.evaluate_signal(_signal(100, 98, 104), open_positions=cfg.max_open_positions)
    assert not d.allowed
    assert "max_open_positions" in d.reason


def test_risk_manager_blocks_after_consecutive_losses():
    cfg = Config()
    rm = RiskManager(cfg, equity=10_000)
    for _ in range(cfg.max_consecutive_losses):
        rm.on_trade_closed(pnl=-50)
    d = rm.evaluate_signal(_signal(100, 98, 104), open_positions=0)
    assert not d.allowed
    assert "breaker" in d.reason or "consecutive" in d.reason


def test_risk_manager_emergency_dd_raises_on_update():
    cfg = Config()
    rm = RiskManager(cfg, equity=10_000)
    with pytest.raises(RiskLimitBreached):
        rm.update_equity(9_400)  # 6% DD > emergency 5%
