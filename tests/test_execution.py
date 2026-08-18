"""Tests for execution: guard, paper executor, order manager idempotency, SL/TP, reconciliation."""
import asyncio
import pytest

from trading_bot.config import Config, TradingMode
from trading_bot.core.enums import OrderType, Side, SystemState, OrderStatus
from trading_bot.core.exceptions import ExecutionError, OrderRejected, ReconciliationMismatch
from trading_bot.core.models import Order, Position, Signal, TrendState, RegimeState, RegimeAction
from trading_bot.execution.execution_guard import ExecutionGuard, AcceptanceEvidence
from trading_bot.execution.executor import PaperExecutor
from trading_bot.execution.order_manager import OrderManager
from trading_bot.execution.sl_tp import SlTpManager
from trading_bot.execution.reconciliation import reconcile, raise_on_critical, ReconciliationReport


# ---- Execution guard (fail-closed) ----

def test_guard_blocks_paper_mode_by_default():
    cfg = Config()  # default PAPER
    g = ExecutionGuard(cfg)
    ok, reason = g.can_trade_live(SystemState.RUNNING)
    assert ok is False and "PAPER" in reason


def test_guard_blocks_when_disabled():
    import os
    os.environ["TRADING_MODE"] = "LIVE"
    os.environ["LIVE_TRADING_ENABLED"] = "false"
    cfg = Config()
    g = ExecutionGuard(cfg)
    ok, reason = g.can_trade_live(SystemState.RUNNING)
    assert ok is False and "LIVE_TRADING_ENABLED" in reason
    del os.environ["TRADING_MODE"]; del os.environ["LIVE_TRADING_ENABLED"]


def test_guard_blocks_when_risk_locked():
    import os
    os.environ["TRADING_MODE"] = "LIVE"
    os.environ["LIVE_TRADING_ENABLED"] = "true"
    os.environ["API_KEY"] = "k"; os.environ["API_SECRET"] = "s"
    cfg = Config()
    g = ExecutionGuard(cfg)
    ok, reason = g.can_trade_live(SystemState.RISK_LOCK)
    assert ok is False and "RISK_LOCK" in reason
    del os.environ["TRADING_MODE"]; del os.environ["LIVE_TRADING_ENABLED"]
    del os.environ["API_KEY"]; del os.environ["API_SECRET"]


def test_guard_blocks_when_acceptance_gates_fail():
    import os
    os.environ["TRADING_MODE"] = "LIVE"
    os.environ["LIVE_TRADING_ENABLED"] = "true"
    os.environ["API_KEY"] = "k"; os.environ["API_SECRET"] = "s"
    cfg = Config()
    g = ExecutionGuard(cfg)
    ev = AcceptanceEvidence(oos_expectancy_positive=False)
    ok, reason = g.can_trade_live(SystemState.RUNNING, ev)
    assert ok is False and "acceptance_gates_failed" in reason
    del os.environ["TRADING_MODE"]; del os.environ["LIVE_TRADING_ENABLED"]
    del os.environ["API_KEY"]; del os.environ["API_SECRET"]


def test_guard_allows_when_all_conditions_met():
    import os
    os.environ["TRADING_MODE"] = "LIVE"
    os.environ["LIVE_TRADING_ENABLED"] = "true"
    os.environ["API_KEY"] = "k"; os.environ["API_SECRET"] = "s"
    cfg = Config()
    g = ExecutionGuard(cfg)
    ev = AcceptanceEvidence(
        oos_expectancy_positive=True, oos_pf_positive=True, wf_consistency_pass=True,
        mc_pass=True, sensitivity_pass=True, stress_pass=True, no_look_ahead=True,
        risk_controls_pass=True, critical_tests_pass=True,
    )
    ok, reason = g.can_trade_live(SystemState.RUNNING, ev)
    assert ok is True and reason == "live_allowed"
    del os.environ["TRADING_MODE"]; del os.environ["LIVE_TRADING_ENABLED"]
    del os.environ["API_KEY"]; del os.environ["API_SECRET"]


def test_assert_can_trade_live_raises_when_blocked():
    cfg = Config()
    g = ExecutionGuard(cfg)
    with pytest.raises(ExecutionError, match="BLOCKED"):
        g.assert_can_trade_live(SystemState.RUNNING)


# ---- Paper executor ----

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_paper_executor_fills_buy():
    ex = PaperExecutor(starting_balance=10_000)
    ex.set_market_price("BTC/USDT", 100.0)
    o = Order(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
              order_type=OrderType.MARKET, quantity=1.0)
    out = asyncio.run(ex.create_order(o))
    assert out.status == OrderStatus.FILLED
    assert out.avg_fill_price > 100  # slippage up
    bal = asyncio.run(ex.get_balance())
    assert bal["USDT"] < 10_000  # spent


def test_paper_executor_rejects_without_price():
    ex = PaperExecutor()
    o = Order(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
              order_type=OrderType.MARKET, quantity=1.0)
    out = asyncio.run(ex.create_order(o))
    assert out.status == OrderStatus.REJECTED


# ---- Order manager idempotency ----

def _signal(ts=0, side=Side.BUY):
    return Signal(
        strategy_version="v1", symbol="BTC/USDT", side=side, entry=100,
        stop_loss=98, take_profit=104, rr=2.0, score=80, trend=TrendState.BULLISH,
        regime=RegimeState.STRONG_TREND, regime_action=RegimeAction.TRADE,
        rsi=45, atr=1.5, atr_percent=1.5, adx=25, ema_fast=100, ema_slow=95,
        spread_percent=0.05, timestamp=ts,
    )


def test_order_manager_idempotent_resubmit():
    ex = PaperExecutor(); ex.set_market_price("BTC/USDT", 100.0)
    om = OrderManager(ex)
    sig = _signal()
    o1 = asyncio.run(om.submit(sig, quantity=1.0))
    o2 = asyncio.run(om.submit(sig, quantity=1.0))  # same signal_id
    assert o1.client_order_id == o2.client_order_id
    # only one order persisted
    assert len(om.all_orders()) == 1


def test_order_manager_tracks_open_and_rejected():
    ex = PaperExecutor()  # no price -> reject
    om = OrderManager(ex)
    sig = _signal()
    with pytest.raises(OrderRejected):
        asyncio.run(om.submit(sig, quantity=1.0))


# ---- SL/TP manager ----

def test_sltp_break_even_moves_stop_forward_only():
    pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                   quantity=1.0, entry_price=100, initial_stop_loss=98, stop_loss=98, take_profit=106)
    m = SlTpManager(break_even_r=1.0)
    # move up to +1.5R favorable -> should move BE
    r = m.update(pos, high=104, low=100)  # fav = 4, risk = 2 -> BE
    assert r is None  # no exit
    assert pos.stop_loss >= 100  # moved to BE


def test_sltp_trailing_never_moves_backward():
    pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                   quantity=1.0, entry_price=100, initial_stop_loss=98, stop_loss=98, take_profit=110)
    m = SlTpManager(trail_atr_mult=2.0)
    # new high 103, atr 1 -> trail = 103 - 2 = 101 (moves up)
    m.update(pos, high=103, low=100, atr=1.0)
    assert pos.stop_loss == 101
    # next bar lower high -> trail would be < current SL -> must NOT move back
    m.update(pos, high=102, low=100, atr=1.0)
    assert pos.stop_loss == 101


def test_sltp_exits_on_sl_and_tp():
    pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                   quantity=1.0, entry_price=100, initial_stop_loss=98, stop_loss=98, take_profit=106)
    m = SlTpManager()
    assert m.update(pos, high=100, low=97) is not None  # SL hit
    pos2 = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                    quantity=1.0, entry_price=100, initial_stop_loss=95, stop_loss=95, take_profit=106)
    assert m.update(pos2, high=107, low=100) is not None  # TP hit


def test_sltp_verify_flags_missing_protection():
    pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                   quantity=1.0, entry_price=100, initial_stop_loss=101, stop_loss=101, take_profit=106)
    m = SlTpManager()
    issues = m.verify_sl_tp(pos)  # SL above entry -> invalid
    assert "sl_not_below_entry" in issues


# ---- Reconciliation ----

def test_reconciliation_detects_unknown_position():
    local = []
    exch = [{"symbol": "BTC/USDT", "contracts": 1.0}]
    rep = reconcile(local, exch, [], [])
    assert len(rep.unknown_positions) == 1
    # unknown position is a critical mismatch -> fail-closed
    with pytest.raises(ReconciliationMismatch, match="unknown_positions"):
        raise_on_critical(rep)


def test_reconciliation_detects_missing_position():
    pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                   quantity=1.0, entry_price=100, initial_stop_loss=98, stop_loss=98, take_profit=106)
    rep = reconcile([pos], [], [], [])
    assert len(rep.missing_positions) == 1


def test_reconciliation_detects_incorrect_size():
    pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                   quantity=1.0, entry_price=100, initial_stop_loss=98, stop_loss=98, take_profit=106)
    exch = [{"symbol": "BTC/USDT", "contracts": 2.0}]
    rep = reconcile([pos], exch, [], [])
    assert len(rep.incorrect_size) == 1


def test_reconciliation_detects_duplicate_orders():
    local_orders = [{"client_order_id": "X"}, {"client_order_id": "X"}]
    rep = reconcile([], [], local_orders, [])
    assert len(rep.duplicate_orders) == 1


def test_raise_on_critical_raises():
    rep = ReconciliationReport(unknown_positions=[{"symbol": "X"}])
    with pytest.raises(ReconciliationMismatch):
        raise_on_critical(rep)
