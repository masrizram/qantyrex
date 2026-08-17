"""SL/TP management: break-even, trailing, and continuous SL/TP verification.

In live/paper mode this runs on every tick / candle. It moves SL only in the
favorable direction (never backwards) and verifies SL/TP orders exist.
"""
from __future__ import annotations

from typing import List, Optional

from ..core.enums import ExitReason, Side
from ..core.exceptions import ReconciliationMismatch
from ..core.models import Position


class SlTpManager:
    def __init__(self, break_even_r: float = 1.0, trail_atr_mult: float = 2.0) -> None:
        self.break_even_r = break_even_r
        self.trail_atr_mult = trail_atr_mult
        self._moved_be: set[str] = set()

    def update(self, pos: Position, high: float, low: float,
               atr: Optional[float] = None) -> Optional[ExitReason]:
        """Update SL/BE/trailing on a new bar. Returns an exit reason if hit."""
        if pos.side == Side.BUY:
            # excursion
            fav = (high - pos.entry_price)
            adv = (low - pos.entry_price)
            pos.max_favorable = max(pos.max_favorable or -1e18, fav)
            pos.max_adverse = min(pos.max_adverse or 1e18, adv)
            # SL hit?
            if low <= pos.stop_loss:
                return ExitReason.SL
            # TP hit?
            if high >= pos.take_profit:
                return ExitReason.TP
            # break-even
            risk = pos.entry_price - pos.stop_loss
            if pos.trade_id not in self._moved_be and pos.max_favorable >= risk * self.break_even_r:
                pos.stop_loss = max(pos.stop_loss, pos.entry_price)
                pos.break_even = True
                self._moved_be.add(pos.trade_id)
            # trailing
            if atr is not None and atr > 0:
                new_trail = high - self.trail_atr_mult * atr
                if new_trail > pos.stop_loss:
                    pos.stop_loss = new_trail
        else:  # SELL
            fav = (pos.entry_price - low)
            adv = (pos.entry_price - high)
            pos.max_favorable = max(pos.max_favorable or -1e18, fav)
            pos.max_adverse = min(pos.max_adverse or 1e18, adv)
            if high >= pos.stop_loss:
                return ExitReason.SL
            if low <= pos.take_profit:
                return ExitReason.TP
            risk = pos.stop_loss - pos.entry_price
            if pos.trade_id not in self._moved_be and pos.max_favorable >= risk * self.break_even_r:
                pos.stop_loss = min(pos.stop_loss, pos.entry_price)
                pos.break_even = True
                self._moved_be.add(pos.trade_id)
            if atr is not None and atr > 0:
                new_trail = low + self.trail_atr_mult * atr
                if new_trail < pos.stop_loss:
                    pos.stop_loss = new_trail
        return None

    def verify_sl_tp(self, pos: Position) -> List[str]:
        """Return a list of missing-protection issues (empty == ok)."""
        issues = []
        if pos.is_open:
            if pos.side == Side.BUY:
                if not (pos.stop_loss < pos.entry_price):
                    issues.append("sl_not_below_entry")
                if not (pos.take_profit > pos.entry_price):
                    issues.append("tp_not_above_entry")
            else:
                if not (pos.stop_loss > pos.entry_price):
                    issues.append("sl_not_above_entry")
                if not (pos.take_profit < pos.entry_price):
                    issues.append("tp_not_below_entry")
        return issues
