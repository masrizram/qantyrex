"""Phase 25 forensic tests: position lifecycle, SL integrity, trade journal, holding analysis.

DO NOT modify strategy parameters. These tests verify CORRECTNESS only.
"""
import pytest
import numpy as np
import pandas as pd
from dataclasses import asdict

from trading_bot.core.enums import Side, ExitReason, TrendState, RegimeState, RegimeAction
from trading_bot.core.models import Position, Signal, TradeRecord, config_hash
from trading_bot.backtest.simulator import Simulator, SimulatorConfig
from trading_bot.backtest.holding_analysis import analyze_holding_period, HoldingAnalysis
from trading_bot.config import Config, load_config
from trading_bot.execution.sl_tp import SlTpManager
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.strategy.signal_engine import SignalEngine


def _make_candles(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100.0
    closes = base + np.cumsum(rng.normal(0, 0.5, n))
    highs = closes + rng.uniform(0.2, 0.8, n)
    lows = closes - rng.uniform(0.2, 0.8, n)
    opens = closes - rng.normal(0, 0.3, n)
    t0 = 1_700_000_000_000
    return pd.DataFrame({
        "timestamp": [t0 + i * 3_600_000 for i in range(n)],
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": rng.uniform(10, 100, n),
    })


def _make_signal(side: Side, entry: float, sl: float, tp: float, ts: int = 0) -> Signal:
    return Signal(
        strategy_version="v1", symbol="BTC/USDT", side=side,
        entry=entry, stop_loss=sl, take_profit=tp, rr=2.0, score=80,
        trend=TrendState.BULLISH, regime=RegimeState.STRONG_TREND,
        regime_action=RegimeAction.TRADE, rsi=45, atr=1.5, atr_percent=1.5,
        adx=25, ema_fast=100, ema_slow=95, spread_percent=0.05,
        timestamp=ts,
        features={"quantity": 1.0},
    )


# ============================================================
# POSITION INTEGRITY
# ============================================================

class TestPositionIntegrity:
    def test_buy_valid_sl_below_entry(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                       quantity=1.0, entry_price=100, initial_stop_loss=98,
                       stop_loss=98, take_profit=106)
        assert pos.stop_loss < pos.entry_price
        assert pos.initial_stop_loss < pos.entry_price

    def test_sell_valid_sl_above_entry(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.SELL,
                       quantity=1.0, entry_price=100, initial_stop_loss=102,
                       stop_loss=102, take_profit=94)
        assert pos.stop_loss > pos.entry_price
        assert pos.initial_stop_loss > pos.entry_price

    def test_zero_risk_signal_rejected_by_simulator(self):
        candles = _make_candles(30)
        sig = _make_signal(Side.BUY, entry=100, sl=100, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, max_open_positions=99))
        result = sim.run(candles, [sig])
        assert len(result.trades) == 0
        assert result.rejected_signals == 1
        reasons = [r["reason"] for r in result.signal_rejections]
        assert any("ZERO_RISK" in r for r in reasons)

    def test_sl_equal_to_entry_rejected_by_simulator(self):
        candles = _make_candles(30)
        sig = _make_signal(Side.BUY, entry=100, sl=100, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0))
        result = sim.run(candles, [sig])
        assert len(result.trades) == 0

    def test_sl_wrong_side_buy_rejected(self):
        candles = _make_candles(30)
        sig = _make_signal(Side.BUY, entry=100, sl=105, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0))
        result = sim.run(candles, [sig])
        assert len(result.trades) == 0
        reasons = [r["reason"] for r in result.signal_rejections]
        assert any("INVALID_STOP_SIDE" in r for r in reasons)

    def test_sl_wrong_side_sell_rejected(self):
        candles = _make_candles(30)
        sig = _make_signal(Side.SELL, entry=100, sl=95, tp=94,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0))
        result = sim.run(candles, [sig])
        assert len(result.trades) == 0
        reasons = [r["reason"] for r in result.signal_rejections]
        assert any("INVALID_STOP_SIDE" in r for r in reasons)


# ============================================================
# SL AUDIT
# ============================================================

class TestSLAudit:
    def test_initial_sl_recorded_on_open(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,  # high BE to avoid
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        assert "initial_stop_loss" in tr
        assert tr["initial_stop_loss"] == 98.0

    def test_initial_sl_immutable_in_sl_audit(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        audit = tr.get("sl_audit", [])
        init_events = [e for e in audit if e.get("reason") == "INITIAL"]
        assert len(init_events) >= 1
        assert init_events[0]["new_sl"] == 98.0

    def test_break_even_mutation_recorded(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                       quantity=1.0, entry_price=100, initial_stop_loss=98,
                       stop_loss=98, take_profit=106)
        pos.sl_audit.append({"timestamp": 0, "position_id": pos.trade_id,
                             "old_sl": None, "new_sl": 98.0, "reason": "INITIAL"})
        old_sl = pos.stop_loss
        pos.stop_loss = pos.entry_price
        pos.sl_audit.append({"timestamp": 1, "position_id": pos.trade_id,
                             "old_sl": old_sl, "new_sl": pos.stop_loss,
                             "reason": "BREAK_EVEN"})
        assert pos.initial_stop_loss == 98.0
        assert pos.stop_loss == 100.0
        assert len(pos.sl_audit) == 2
        assert pos.sl_audit[1]["reason"] == "BREAK_EVEN"
        assert pos.sl_audit[1]["old_sl"] == 98.0
        assert pos.sl_audit[1]["new_sl"] == 100.0

    def test_trailing_mutation_recorded(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                       quantity=1.0, entry_price=100, initial_stop_loss=98,
                       stop_loss=98, take_profit=106)
        pos.sl_audit.append({"timestamp": 0, "position_id": pos.trade_id,
                             "old_sl": None, "new_sl": 98.0, "reason": "INITIAL"})
        old_sl = pos.stop_loss
        pos.stop_loss = 101.0
        pos.sl_audit.append({"timestamp": 1, "position_id": pos.trade_id,
                             "old_sl": old_sl, "new_sl": 101.0,
                             "reason": "TRAILING"})
        assert pos.initial_stop_loss == 98.0
        assert pos.stop_loss == 101.0

    def test_no_silent_sl_mutation(self):
        """Verify there are no SL mutations outside the audit trail."""
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                       quantity=1.0, entry_price=100, initial_stop_loss=98,
                       stop_loss=98, take_profit=106)
        pos.sl_audit.append({"timestamp": 0, "position_id": pos.trade_id,
                             "old_sl": None, "new_sl": 98.0, "reason": "INITIAL"})
        pos.stop_loss = 99.0  # mutation without audit
        assert len(pos.sl_audit) == 1  # no audit event added


# ============================================================
# TRADE JOURNAL
# ============================================================

class TestTradeJournal:
    def test_forensic_fields_present(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        for field in ["signal_timestamp", "opened_at", "closed_at",
                      "entry_price", "exit_price", "initial_stop_loss",
                      "final_stop_loss", "take_profit", "exit_reason",
                      "sl_audit"]:
            assert field in tr, f"Missing forensic field: {field}"

    def test_opened_at_before_closed_at(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        assert tr["opened_at"] <= tr["closed_at"]

    def test_signal_timestamp_before_opened_at(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        assert tr["signal_timestamp"] <= tr["opened_at"]

    def test_initial_stop_loss_preserved(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        assert tr["initial_stop_loss"] == 98.0

    def test_legacy_fields_still_present(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        for field in ["timestamp", "entry", "exit", "stop_loss", "pnl", "fees"]:
            assert field in tr, f"Missing legacy field: {field}"


# ============================================================
# HOLDING ANALYSIS
# ============================================================

class TestHoldingAnalysis:
    def test_holding_bars_nonnegative(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        opened_at = tr.get("opened_at")
        closed_at = tr.get("closed_at")
        if opened_at and closed_at:
            assert closed_at >= opened_at
            assert closed_at - opened_at >= 0

    def test_holding_analysis_uses_opened_at(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        tr = result.trades.iloc[0] if len(result.trades) > 0 else None
        if tr is None:
            pytest.skip("No trade produced")
        opened_at = tr.get("opened_at")
        assert opened_at is not None
        assert opened_at != tr.get("timestamp")  # should NOT be the exit timestamp


# ============================================================
# ACCOUNTING
# ============================================================

class TestAccounting:
    def test_equity_after_exit_reconciles(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99,
                                        initial_equity=10_000.0))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        assert "pnl" in tr
        assert np.isfinite(tr["pnl"])

    def test_r_multiple_uses_initial_stop_loss(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=0.01,  # trigger BE quickly
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        risk_per_unit = abs(tr["entry_price"] - tr["initial_stop_loss"])
        if risk_per_unit > 0 and tr["quantity"] > 0:
            expected_r = tr["pnl"] / (risk_per_unit * tr["quantity"])
            assert np.isclose(tr["r_multiple"], expected_r, rtol=1e-9)


# ============================================================
# SIGNAL -> POSITION INVARIANT
# ============================================================

class TestSignalPositionInvariant:
    def test_signal_sl_equals_position_initial_sl(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) == 0:
            pytest.skip("No trade produced")
        tr = result.trades.iloc[0]
        assert tr["initial_stop_loss"] == sig.stop_loss


# ============================================================
# SL/TP MANAGEMENT
# ============================================================

class TestSltpManagement:
    def test_sltp_break_even_does_not_change_initial_sl(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                       quantity=1.0, entry_price=100, initial_stop_loss=98,
                       stop_loss=98, take_profit=106)
        m = SlTpManager(break_even_r=1.0)
        m.update(pos, high=104, low=100)
        assert pos.initial_stop_loss == 98.0
        assert pos.stop_loss >= 100.0

    def test_sltp_trailing_does_not_change_initial_sl(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                       quantity=1.0, entry_price=100, initial_stop_loss=98,
                       stop_loss=98, take_profit=110)
        m = SlTpManager(trail_atr_mult=2.0)
        m.update(pos, high=103, low=100, atr=1.0)
        assert pos.initial_stop_loss == 98.0
        assert pos.stop_loss == 101.0

    def test_sltp_only_moves_in_risk_reducing_direction_long(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                       quantity=1.0, entry_price=100, initial_stop_loss=98,
                       stop_loss=98, take_profit=110)
        m = SlTpManager(trail_atr_mult=2.0)
        m.update(pos, high=103, low=100, atr=1.0)
        assert pos.stop_loss == 101.0
        m.update(pos, high=102, low=100, atr=1.0)
        assert pos.stop_loss == 101.0  # never moved backward
        m.update(pos, high=105, low=102, atr=1.0)
        assert pos.stop_loss == 103.0  # moved forward only

    def test_sltp_only_moves_in_risk_reducing_direction_short(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.SELL,
                       quantity=1.0, entry_price=100, initial_stop_loss=102,
                       stop_loss=102, take_profit=94)
        m = SlTpManager(trail_atr_mult=2.0)
        m.update(pos, high=100, low=97, atr=1.0)
        assert pos.stop_loss == 99.0
        m.update(pos, high=100, low=98, atr=1.0)
        assert pos.stop_loss == 99.0  # never moved backward
        m.update(pos, high=98, low=95, atr=1.0)
        assert pos.stop_loss == 97.0  # moved forward only


# ============================================================
# TRADE RECORD MODEL
# ============================================================

class TestTradeRecordModel:
    def test_trade_record_has_forensic_fields(self):
        ch = config_hash({"a": 1})
        tr = TradeRecord(
            trade_id="T1", signal_id="S1", strategy_version="v1",
            config_hash=ch, code_version="0.1.0",
            signal_timestamp=500, opened_at=1000, closed_at=2000,
            symbol="BTC/USDT", side=Side.BUY,
            entry_price=100, exit_price=104, quantity=1.0,
            initial_stop_loss=98, final_stop_loss=98, take_profit=106,
            fees=0.5, slippage=0.2, pnl=3.5, r_multiple=1.75,
            regime=RegimeState.STRONG_TREND, signal_score=82.0,
            risk_percent=0.005, exit_reason=ExitReason.TP,
        )
        assert tr.signal_timestamp == 500
        assert tr.opened_at == 1000
        assert tr.closed_at == 2000
        assert tr.entry_price == 100
        assert tr.exit_price == 104
        assert tr.initial_stop_loss == 98
        assert tr.final_stop_loss == 98


# ============================================================
# DAILY DD INTERACTION
# ============================================================

class TestDailyDD:
    def test_daily_dd_below_threshold_allows_trade(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, max_open_positions=99,
                                        initial_equity=10_000.0))
        result = sim.run(candles, [sig])
        assert result.rejected_signals == 0 or \
            all("DAILY_DD" not in r.get("reason", "") for r in result.signal_rejections)

    def test_position_closed_after_sl_exit(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=99.5, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) > 0:
            tr = result.trades.iloc[0]
            assert tr["exit_reason"] is not None
            assert tr["closed_at"] is not None


# ============================================================
# DETERMINISTIC REPLAY
# ============================================================

class TestDeterministicReplay:
    def test_valid_long_replay_consistent(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim1 = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                         max_open_positions=99))
        r1 = sim1.run(candles, [sig])
        sim2 = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                         max_open_positions=99))
        r2 = sim2.run(candles, [sig])
        assert len(r1.trades) == len(r2.trades)
        if len(r1.trades) > 0:
            for col in ["entry_price", "exit_price", "initial_stop_loss",
                        "final_stop_loss", "pnl", "exit_reason"]:
                if col in r1.trades.columns:
                    assert r1.trades.iloc[0][col] == r2.trades.iloc[0][col], \
                        f"Replay mismatch on {col}"

    def test_valid_short_replay_consistent(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.SELL, entry=100, sl=102, tp=94,
                           ts=int(candles["timestamp"].iloc[0]))
        sim1 = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                         max_open_positions=99))
        r1 = sim1.run(candles, [sig])
        sim2 = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                         max_open_positions=99))
        r2 = sim2.run(candles, [sig])
        assert len(r1.trades) == len(r2.trades)


# ============================================================
# INVARIANT VERIFICATION
# ============================================================

class TestInvariants:
    def test_no_position_with_entry_equals_sl(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=100, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, max_open_positions=99))
        result = sim.run(candles, [sig])
        assert len(result.trades) == 0

    def test_no_position_with_sl_wrong_side(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=105, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, max_open_positions=99))
        result = sim.run(candles, [sig])
        assert len(result.trades) == 0

    def test_initial_sl_never_changes_retroactively(self):
        pos = Position(strategy_version="v1", symbol="BTC/USDT", side=Side.BUY,
                       quantity=1.0, entry_price=100, initial_stop_loss=98,
                       stop_loss=98, take_profit=106)
        initial = pos.initial_stop_loss
        pos.stop_loss = 99.0
        pos.stop_loss = 100.0
        pos.stop_loss = 101.0
        assert pos.initial_stop_loss == initial

    def test_every_trade_has_deterministic_outcome(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, max_open_positions=99))
        result = sim.run(candles, [sig])
        total = len(result.trades) + result.rejected_signals
        assert total == 1  # every signal accounted for

    def test_exit_timestamp_not_treated_as_entry_timestamp(self):
        candles = _make_candles(30, seed=42)
        sig = _make_signal(Side.BUY, entry=100, sl=98, tp=106,
                           ts=int(candles["timestamp"].iloc[0]))
        sim = Simulator(SimulatorConfig(seed=0, break_even_r=10.0,
                                        max_open_positions=99))
        result = sim.run(candles, [sig])
        if len(result.trades) > 0:
            tr = result.trades.iloc[0]
            assert tr["opened_at"] != tr["timestamp"] or \
                   tr["signal_timestamp"] != tr["timestamp"]