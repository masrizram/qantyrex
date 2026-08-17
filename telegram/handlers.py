"""Telegram command handlers. Each returns a text response.

The handlers are pure functions of state so they can be unit-tested without
a running bot. They never expose secrets.
"""
from __future__ import annotations

from typing import Callable, Dict
from dataclasses import dataclass

from ..core.enums import SystemState
from .authorization import Authorization


@dataclass
class BotContext:
    """Snapshot of bot state passed to handlers."""
    state: SystemState
    balance: dict
    positions: list
    risk: dict
    stats: dict
    health: dict
    config_safe: dict


def _fmt_dict(d: dict, indent: int = 0) -> str:
    pad = "  " * indent
    out = []
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(_fmt_dict(v, indent + 1))
        else:
            out.append(f"{pad}{k}: {v}")
    return "\n".join(out)


def handle_status(ctx: BotContext, auth: Authorization, chat_id: int) -> str:
    auth.require(chat_id)
    return (f"STATE: {ctx.state.value}\n"
            f"SYMBOL: {ctx.config_safe.get('symbol')}\n"
            f"TRADING_MODE: {ctx.config_safe.get('trading_mode')}\n"
            f"HEALTH: {ctx.health}")


def handle_balance(ctx: BotContext, auth: Authorization, chat_id: int) -> str:
    auth.require(chat_id)
    return f"BALANCE:\n{_fmt_dict(ctx.balance)}"


def handle_positions(ctx: BotContext, auth: Authorization, chat_id: int) -> str:
    auth.require(chat_id)
    if not ctx.positions:
        return "POSITIONS: none"
    return "POSITIONS:\n" + _fmt_dict({"p": ctx.positions})


def handle_risk(ctx: BotContext, auth: Authorization, chat_id: int) -> str:
    auth.require(chat_id)
    return f"RISK:\n{_fmt_dict(ctx.risk)}"


def handle_stats(ctx: BotContext, auth: Authorization, chat_id: int) -> str:
    auth.require(chat_id)
    return f"STATS:\n{_fmt_dict(ctx.stats)}"


def handle_health(ctx: BotContext, auth: Authorization, chat_id: int) -> str:
    auth.require(chat_id)
    return f"HEALTH:\n{_fmt_dict(ctx.health)}"


def handle_startbot(ctx: BotContext, auth: Authorization, chat_id: int,
                    transition_fn: Callable[[SystemState], None]) -> str:
    auth.require(chat_id)
    try:
        transition_fn(SystemState.RUNNING)
        return "STARTING: transitioning to RUNNING"
    except Exception as e:
        return f"START FAILED: {e}"


def handle_stopbot(ctx: BotContext, auth: Authorization, chat_id: int,
                  transition_fn: Callable[[SystemState], None]) -> str:
    auth.require(chat_id)
    try:
        transition_fn(SystemState.SHUTDOWN)
        return "STOPPING: transitioning to SHUTDOWN"
    except Exception as e:
        return f"STOP FAILED: {e}"


def handle_pause(ctx: BotContext, auth: Authorization, chat_id: int,
                transition_fn: Callable[[SystemState], None]) -> str:
    auth.require(chat_id)
    try:
        transition_fn(SystemState.PAUSED)
        return "PAUSED"
    except Exception as e:
        return f"PAUSE FAILED: {e}"


def handle_resume(ctx: BotContext, auth: Authorization, chat_id: int,
                  transition_fn: Callable[[SystemState], None]) -> str:
    auth.require(chat_id)
    try:
        transition_fn(SystemState.RUNNING)
        return "RESUMED"
    except Exception as e:
        return f"RESUME FAILED: {e}"


def handle_closeall(ctx: BotContext, auth: Authorization, chat_id: int,
                   close_fn: Callable[[], int]) -> str:
    auth.require(chat_id)
    try:
        n = close_fn()
        return f"CLOSED {n} positions"
    except Exception as e:
        return f"CLOSEALL FAILED: {e}"


def handle_unknown(command: str, auth: Authorization, chat_id: int) -> str:
    auth.require(chat_id)
    return f"UNKNOWN COMMAND: {command}"


# Command dispatch table
COMMANDS = {
    "/status": handle_status,
    "/balance": handle_balance,
    "/positions": handle_positions,
    "/risk": handle_risk,
    "/stats": handle_stats,
    "/health": handle_health,
}
