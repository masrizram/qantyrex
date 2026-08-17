"""Deterministic state machine for the bot's lifecycle.

Only explicitly allowed transitions are accepted; everything else raises
InvalidStateTransition (fail-closed).
"""
from __future__ import annotations

from typing import Set

from .enums import SystemState
from .exceptions import InvalidStateTransition


_ALLOWED: dict[SystemState, Set[SystemState]] = {
    SystemState.STARTING: {SystemState.READY, SystemState.ERROR, SystemState.SHUTDOWN},
    SystemState.READY: {
        SystemState.RUNNING, SystemState.PAUSED, SystemState.RISK_LOCK,
        SystemState.ERROR, SystemState.SHUTDOWN,
    },
    SystemState.RUNNING: {
        SystemState.PAUSED, SystemState.RISK_LOCK, SystemState.ERROR,
        SystemState.SHUTDOWN,
    },
    SystemState.PAUSED: {
        SystemState.RUNNING, SystemState.RISK_LOCK, SystemState.ERROR,
        SystemState.SHUTDOWN,
    },
    SystemState.RISK_LOCK: {
        SystemState.PAUSED, SystemState.READY, SystemState.ERROR,
        SystemState.SHUTDOWN,
    },
    SystemState.ERROR: {SystemState.READY, SystemState.SHUTDOWN},
    SystemState.SHUTDOWN: set(),
}


class StateMachine:
    def __init__(self, initial: SystemState = SystemState.STARTING) -> None:
        self._state = initial

    @property
    def state(self) -> SystemState:
        return self._state

    def can(self, target: SystemState) -> bool:
        return target in _ALLOWED[self._state]

    def transition(self, target: SystemState) -> SystemState:
        if not self.can(target):
            raise InvalidStateTransition(
                f"Illegal transition {self._state.value} -> {target.value}")
        self._state = target
        return self._state

    def is_trading_active(self) -> bool:
        return self._state == SystemState.RUNNING
