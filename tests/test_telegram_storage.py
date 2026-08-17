"""Tests for Telegram authorization + handlers + storage (database/repository)."""
import pytest

from trading_bot.config import Config
from trading_bot.core.enums import SystemState
from trading_bot.core.exceptions import AuthorizationDenied
from trading_bot.telegram.authorization import Authorization
from trading_bot.telegram.handlers import (
    BotContext, handle_status, handle_balance, handle_unknown,
    handle_startbot, handle_stopbot, handle_closeall,
)
from trading_bot.telegram.bot import TelegramBot
from trading_bot.storage.database import init_engine, create_all, session_scope
from trading_bot.storage.repository import (
    TradeRecordORM, TradeJournalRepository, trade_record_to_orm,
)
from trading_bot.core.enums import Side, ExitReason, RegimeState
from trading_bot.core.models import TradeRecord, config_hash


# ---- Authorization ----

def test_auth_allows_whitelisted():
    a = Authorization([123, 456])
    assert a.is_authorized(123)
    assert a.is_authorized(456)
    a.require(123)  # no raise


def test_auth_denies_non_whitelisted():
    a = Authorization([123])
    assert not a.is_authorized(999)
    with pytest.raises(AuthorizationDenied, match="ACCESS DENIED"):
        a.require(999)


def test_auth_empty_list_denies_all():
    a = Authorization([])
    assert a.empty()
    with pytest.raises(AuthorizationDenied):
        a.require(1)


# ---- Handlers ----

def _ctx():
    return BotContext(
        state=SystemState.RUNNING,
        balance={"USDT": 10000.0},
        positions=[{"symbol": "BTC/USDT", "qty": 0.5}],
        risk={"risk_percent": 0.005, "daily_dd": 0.01},
        stats={"trades": 10, "win_rate": 0.6},
        health={"exchange": "ok"},
        config_safe={"symbol": "BTC/USDT", "trading_mode": "PAPER"},
    )


def test_handle_status_returns_state():
    a = Authorization([1])
    out = handle_status(_ctx(), a, 1)
    assert "RUNNING" in out and "BTC/USDT" in out


def test_handle_balance_redacts_secrets():
    a = Authorization([1])
    out = handle_balance(_ctx(), a, 1)
    assert "10000" in out


def test_handle_unknown_authorized():
    a = Authorization([1])
    out = handle_unknown("/foo", a, 1)
    assert "UNKNOWN" in out and "/foo" in out


def test_handle_denied_for_unauthorized_chat():
    a = Authorization([1])
    with pytest.raises(AuthorizationDenied):
        handle_status(_ctx(), a, 999)


def test_handle_startbot_transitions():
    a = Authorization([1])
    transitions = []
    def t(target):
        transitions.append(target)
    out = handle_startbot(_ctx(), a, 1, t)
    assert "RUNNING" in out
    assert transitions == [SystemState.RUNNING]


def test_handle_stopbot_transitions():
    a = Authorization([1])
    transitions = []
    def t(target): transitions.append(target)
    out = handle_stopbot(_ctx(), a, 1, t)
    assert "SHUTDOWN" in out
    assert transitions == [SystemState.SHUTDOWN]


def test_handle_closeall_calls_fn():
    a = Authorization([1])
    out = handle_closeall(_ctx(), a, 1, lambda: 2)
    assert "CLOSED 2" in out


# ---- Bot dispatch ----

def test_bot_disabled_without_token():
    cfg = Config()  # no token
    sm_state = {"s": SystemState.RUNNING}

    def provider(): return sm_state["s"]
    def transition(t): sm_state["s"] = t
    bot = TelegramBot(cfg, provider, transition, lambda: 0,
                     lambda: _ctx())
    assert bot.enabled is False


def test_bot_dispatch_authorized():
    cfg = Config()
    cfg = Config.__new__(Config)
    # construct a config with a whitelist
    import os
    os.environ["TELEGRAM_BOT_TOKEN"] = "fake"
    os.environ["TELEGRAM_ALLOWED_CHAT_IDS"] = "111,222"
    cfg = Config()
    sm_state = {"s": SystemState.RUNNING}
    bot = TelegramBot(cfg, lambda: sm_state["s"], lambda t: sm_state.update(s=t),
                      lambda: 0, lambda: _ctx())
    assert bot.enabled is True
    out = bot.dispatch("/status", 111)
    assert "RUNNING" in out
    # unauthorized chat -> raises (caught upstream in real bot)
    with pytest.raises(AuthorizationDenied):
        bot.dispatch("/status", 999)
    del os.environ["TELEGRAM_BOT_TOKEN"]
    del os.environ["TELEGRAM_ALLOWED_CHAT_IDS"]


def test_bot_dispatch_closeall():
    import os
    os.environ["TELEGRAM_BOT_TOKEN"] = "fake"
    os.environ["TELEGRAM_ALLOWED_CHAT_IDS"] = "1"
    cfg = Config()
    bot = TelegramBot(cfg, lambda: SystemState.RUNNING, lambda t: None,
                      lambda: 3, lambda: _ctx())
    out = bot.dispatch("/closeall", 1)
    assert "CLOSED 3" in out
    del os.environ["TELEGRAM_BOT_TOKEN"]
    del os.environ["TELEGRAM_ALLOWED_CHAT_IDS"]


# ---- Storage ----

def test_storage_trade_journal_roundtrip(tmp_path):
    url = f"sqlite:///{tmp_path/'tj.db'}"
    init_engine(url)
    create_all()
    repo = TradeJournalRepository(session_scope)
    ch = config_hash({"a": 1})
    tr = TradeRecord(
        trade_id="T1", signal_id="S1", strategy_version="v1", config_hash=ch,
        code_version="0.1.0", timestamp=1000, symbol="BTC/USDT", side=Side.BUY,
        entry=100, exit=104, quantity=1.0, stop_loss=98, take_profit=106,
        fees=0.5, slippage=0.2, pnl=3.5, r_multiple=1.75,
        regime=RegimeState.STRONG_TREND, signal_score=82.0,
        risk_percent=0.005, exit_reason=ExitReason.TP,
    )
    orm = trade_record_to_orm(tr, config_hash=ch, code_version="0.1.0")
    repo.insert_trade(orm)
    assert repo.trade_count() == 1
    all_t = repo.all_trades()
    assert all_t[0].trade_id == "T1"
    assert all_t[0].pnl == 3.5
    assert all_t[0].exit_reason == ExitReason.TP.value


def test_storage_trades_for_symbol(tmp_path):
    url = f"sqlite:///{tmp_path/'tj2.db'}"
    init_engine(url)
    create_all()
    repo = TradeJournalRepository(session_scope)
    ch = config_hash({"a": 1})
    for i in range(3):
        tr = TradeRecord(
            trade_id=f"T{i}", signal_id=f"S{i}", strategy_version="v1",
            config_hash=ch, code_version="0.1.0", timestamp=1000 + i,
            symbol="BTC/USDT", side=Side.BUY, entry=100, exit=104, quantity=1.0,
            stop_loss=98, take_profit=106, fees=0.5, slippage=0.2, pnl=3.5,
            r_multiple=1.75, regime=RegimeState.STRONG_TREND, signal_score=82.0,
            risk_percent=0.005, exit_reason=ExitReason.TP,
        )
        repo.insert_trade(trade_record_to_orm(tr, ch, "0.1.0"))
    assert len(repo.trades_for_symbol("BTC/USDT")) == 3
