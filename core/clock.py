"""Clock abstraction for testable time handling (backtest vs live)."""
from __future__ import annotations

import time
from typing import Optional


class Clock:
    """Wall-clock for live/paper trading."""
    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class VirtualClock:
    """Deterministic clock for backtests. Never sleeps real time."""
    def __init__(self, start_ms: int = 0) -> None:
        self._t = start_ms

    def now_ms(self) -> int:
        return self._t

    def advance(self, ms: int) -> None:
        if ms < 0:
            raise ValueError("VirtualClock cannot move backwards (no look-ahead).")
        self._t += int(ms)

    def set(self, ms: int) -> None:
        if ms < self._t:
            raise ValueError("VirtualClock cannot move backwards (no look-ahead).")
        self._t = int(ms)

    def sleep(self, seconds: float) -> None:  # no-op in backtest
        return None
