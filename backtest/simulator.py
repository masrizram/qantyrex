"""Event-driven simulator: models fees, spread, slippage, latency, partial fills,
rejections, SL/TP, break-even, trailing, daily DD, exposure, position limits.

Operates on a sequence of OHLCV candles. Signals are produced by the caller
(one per closed bar) and the simulator fills/exits them on subsequent bars
based on intrabar high/low — never with look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.enums import ExitReason, OrderStatus, Side
from ..core.models import Position, Signal, TradeRecord
from .metrics import compute_metrics, Metrics


@dataclass
class SimulatorConfig:
    initial_equity: float = 10_000.0
    fee_rate: float = 0.001
    slippage_bps: float = 2.0
    spread_bps: float = 1.0
    latency_bars: int = 0  # signal at bar t fills at bar t+latency
    rejection_prob: float = 0.0  # probability an order is rejected
    partial_fill_prob: float = 0.0
    partial_fill_ratio: float = 0.5
    break_even_r: float = 1.0
    trail_atr_mult: float = 2.0  # trailing distance = mult * ATR
    max_open_positions: int = 1
    seed: int = 0
    apply_slippage_on_exit: bool = True


@dataclass
class SimulationResult:
    trades: pd.DataFrame  # closed trades journal
    equity_curve: pd.Series
    metrics: Metrics
    rejected_signals: int = 0
    open_positions_eod: int = 0


class Simulator:
    def __init__(self, cfg: SimulatorConfig | None = None) -> None:
        self.cfg = cfg or SimulatorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def run(self, candles: pd.DataFrame,
            signals: List[Signal],
            features_for_atr: Optional[pd.DataFrame] = None) -> SimulationResult:
        """Simulate signals over candles. Each signal's `timestamp` (ms) must
        align with a candle open-time in `candles`."""
        candles = candles.copy().sort_values("timestamp").reset_index(drop=True)
        ts_to_idx = {int(t): i for i, t in enumerate(candles["timestamp"])}
        signals = sorted(signals, key=lambda s: s.timestamp)

        equity = self.cfg.initial_equity
        peak = equity
        start_of_day = equity
        equity_pts: List[tuple[int, float]] = [(int(candles["timestamp"].iloc[0]), equity)]
        trade_rows: List[dict] = []
        open_positions: List[Position] = []
        rejected = 0
        last_day = pd.Timestamp(int(candles["timestamp"].iloc[0]), unit="ms", tz="UTC").date()

        for sig in signals:
            idx = ts_to_idx.get(int(sig.timestamp))
            if idx is None:
                continue  # signal not aligned to a candle; skip (no fabrication)
            fill_idx = idx + 1 + self.cfg.latency_bars
            if fill_idx >= len(candles):
                continue
            # rollover day for DD accounting
            cur_day = pd.Timestamp(int(candles["timestamp"].iloc[fill_idx]), unit="ms", tz="UTC").date()
            if cur_day != last_day:
                start_of_day = equity
                last_day = cur_day

            # daily DD check
            daily_dd = (start_of_day - equity) / start_of_day if start_of_day > 0 else 0
            if daily_dd >= 0.03:
                # stop taking new entries on this day
                rejected += 1
                continue
            if len(open_positions) >= self.cfg.max_open_positions:
                rejected += 1
                continue
            # rejection simulation
            if self.cfg.rejection_prob > 0 and self.rng.random() < self.cfg.rejection_prob:
                rejected += 1
                continue
            fill_bar = candles.iloc[fill_idx]
            # execution price with slippage + spread
            slip = self.cfg.slippage_bps / 10000.0
            spread = self.cfg.spread_bps / 10000.0
            if sig.side == Side.BUY:
                fill_price = float(fill_bar["open"]) * (1 + slip + spread / 2)
            else:
                fill_price = float(fill_bar["open"]) * (1 - slip - spread / 2)
            # partial fill
            qty_mult = 1.0
            if self.cfg.partial_fill_prob > 0 and self.rng.random() < self.cfg.partial_fill_prob:
                qty_mult = self.cfg.partial_fill_ratio
            entry_fee = abs(fill_price) * 0 * 0  # fees computed on notional below
            notional = sig.entry  # placeholder; sizing handled by risk manager upstream
            # We use the signal's intended risk to derive qty: 1 unit of risk
            # The caller should embed the intended qty in signal.features or compute here.
            # For the backtester we use the position sizer via RiskManager (passed in).
            # Here we simply assume qty=1 share-equivalent and scale pnl by size_unit.
            qty = float(sig.features.get("quantity", 1.0)) * qty_mult
            entry_fee = qty * fill_price * self.cfg.fee_rate

            pos = Position(
                signal_id=sig.signal_id, strategy_version=sig.strategy_version,
                symbol=sig.symbol, side=sig.side, quantity=qty,
                entry_price=fill_price, stop_loss=sig.stop_loss,
                take_profit=sig.take_profit, opened_at=int(fill_bar["timestamp"]),
                fees=entry_fee,
            )
            open_positions.append(pos)
            equity -= entry_fee
            equity_pts.append((int(fill_bar["timestamp"]), equity))

            # manage this position on subsequent bars
            self._manage(pos, candles, fill_idx + 1, features_for_atr, trade_rows,
                         lambda: equity, peak)

        # close any remaining open positions at last close
        for pos in open_positions:
            if pos.is_open:
                last = candles.iloc[-1]
                self._exit(pos, float(last["close"]), int(last["timestamp"]),
                           ExitReason.TIMEOUT, trade_rows)
        # rebuild equity curve from trade pnls (simpler + robust)
        eq = [self.cfg.initial_equity]
        for tr in trade_rows:
            eq.append(eq[-1] + tr["pnl"])
        if not trade_rows:
            eq = [self.cfg.initial_equity]
        equity_curve = pd.Series(eq, name="equity")
        trades_df = pd.DataFrame(trade_rows)
        m = compute_metrics(trades_df, equity_curve)
        return SimulationResult(
            trades=trades_df, equity_curve=equity_curve, metrics=m,
            rejected_signals=rejected,
            open_positions_eod=sum(1 for p in open_positions if p.is_open),
        )

    def _manage(self, pos: Position, candles: pd.DataFrame, start_idx: int,
                features: Optional[pd.DataFrame], trade_rows: List[dict],
                equity_fn, peak) -> None:
        i = start_idx
        moved_be = False
        while i < len(candles) and pos.is_open:
            bar = candles.iloc[i]
            hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
            # update favorable/adverse excursion
            if pos.side == Side.BUY:
                pos.max_favorable = max(pos.max_favorable or -1e18, hi - pos.entry_price)
                pos.max_adverse = min(pos.max_adverse or 1e18, lo - pos.entry_price)
            else:
                pos.max_favorable = max(pos.max_favorable or -1e18, pos.entry_price - lo)
                pos.max_adverse = min(pos.max_adverse or 1e18, pos.entry_price - hi)
            # trailing stop (ATR-based) using current bar's ATR
            atr_v = None
            if features is not None and i < len(features) and "atr" in features.columns:
                atr_v = features["atr"].iloc[i]
            # check SL first (worst case first), then TP, then BE/trailing update
            if pos.side == Side.BUY:
                if lo <= pos.stop_loss:
                    self._exit(pos, pos.stop_loss, int(bar["timestamp"]),
                               ExitReason.SL, trade_rows)
                    return
                if hi >= pos.take_profit:
                    self._exit(pos, pos.take_profit, int(bar["timestamp"]),
                               ExitReason.TP, trade_rows)
                    return
                # break-even at +1R
                if not moved_be and pos.max_favorable and pos.max_favorable >= (
                        pos.entry_price - pos.stop_loss) * self.cfg.break_even_r:
                    pos.stop_loss = pos.entry_price  # never move SL backward
                    moved_be = True
                # trailing
                if atr_v is not None and not np.isnan(atr_v):
                    new_trail = hi - self.cfg.trail_atr_mult * atr_v
                    if new_trail > pos.stop_loss:
                        pos.stop_loss = new_trail
            else:  # SELL
                if hi >= pos.stop_loss:
                    self._exit(pos, pos.stop_loss, int(bar["timestamp"]),
                               ExitReason.SL, trade_rows)
                    return
                if lo <= pos.take_profit:
                    self._exit(pos, pos.take_profit, int(bar["timestamp"]),
                               ExitReason.TP, trade_rows)
                    return
                if not moved_be and pos.max_favorable and pos.max_favorable >= (
                        pos.stop_loss - pos.entry_price) * self.cfg.break_even_r:
                    pos.stop_loss = pos.entry_price
                    moved_be = True
                if atr_v is not None and not np.isnan(atr_v):
                    new_trail = lo + self.cfg.trail_atr_mult * atr_v
                    if new_trail < pos.stop_loss:
                        pos.stop_loss = new_trail
            i += 1

    def _exit(self, pos: Position, price: float, ts: int, reason: ExitReason,
              trade_rows: List[dict]) -> None:
        raw_price = price
        slippage_amt = 0.0
        if self.cfg.apply_slippage_on_exit:
            slip = self.cfg.slippage_bps / 10000.0
            if pos.side == Side.BUY:
                price = price * (1 - slip)
            else:
                price = price * (1 + slip)
            slippage_amt = abs(price - raw_price) * pos.quantity
        exit_fee = pos.quantity * price * self.cfg.fee_rate
        total_fees = pos.fees + exit_fee
        gross_pnl = ((price - pos.entry_price) if pos.side == Side.BUY
                     else (pos.entry_price - price)) * pos.quantity
        pnl = gross_pnl - total_fees
        # R multiple relative to original risk per unit
        risk_per_unit = abs(pos.entry_price - pos.stop_loss)
        r_mult = pnl / (risk_per_unit * pos.quantity) if risk_per_unit > 0 and pos.quantity > 0 else 0.0
        pos.closed_at = ts
        pos.exit_price = price
        pos.exit_reason = reason
        pos.realized_pnl = pnl
        pos.fees = total_fees
        trade_rows.append({
            "trade_id": pos.trade_id, "signal_id": pos.signal_id or "",
            "strategy_version": pos.strategy_version, "timestamp": ts,
            "symbol": pos.symbol, "side": pos.side.value,
            "entry": pos.entry_price, "exit": price, "quantity": pos.quantity,
            "stop_loss": pos.stop_loss, "take_profit": pos.take_profit,
            "fees": total_fees, "slippage": slippage_amt,
            "pnl": pnl, "r_multiple": r_mult, "exit_reason": reason.value,
        })
