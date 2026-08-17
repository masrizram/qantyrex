"""Telegram bot wrapper. Optional: if TELEGRAM_BOT_TOKEN is empty, the bot
is a no-op and the main loop continues. Never exposes secrets."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from ..config import Config
from ..core.enums import SystemState
from .authorization import Authorization
from .handlers import (
    BotContext, COMMANDS, handle_startbot, handle_stopbot, handle_pause,
    handle_resume, handle_closeall, handle_unknown,
)

log = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, cfg: Config,
                 state_provider: Callable[[], SystemState],
                 transition: Callable[[SystemState], None],
                 close_all: Callable[[], int],
                 ctx_provider: Callable[[], BotContext]) -> None:
        self.cfg = cfg
        self.auth = Authorization(cfg.telegram_allowed_chat_ids)
        self.state_provider = state_provider
        self.transition = transition
        self.close_all = close_all
        self.ctx_provider = ctx_provider
        self._app = None
        self._enabled = bool(cfg.telegram_bot_token) and not self.auth.empty()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def dispatch(self, command: str, chat_id: int) -> str:
        """Synchronous dispatch used for testing and for a simple polling loop."""
        ctx = self.ctx_provider()
        if command in COMMANDS:
            return COMMANDS[command](ctx, self.auth, chat_id)
        if command == "/startbot":
            return handle_startbot(ctx, self.auth, chat_id, self.transition)
        if command == "/stopbot":
            return handle_stopbot(ctx, self.auth, chat_id, self.transition)
        if command == "/pause":
            return handle_pause(ctx, self.auth, chat_id, self.transition)
        if command == "/resume":
            return handle_resume(ctx, self.auth, chat_id, self.transition)
        if command == "/closeall":
            return handle_closeall(ctx, self.auth, chat_id, self.close_all)
        return handle_unknown(command, self.auth, chat_id)

    async def start(self) -> None:
        if not self._enabled:
            log.info("Telegram disabled (no token or empty whitelist).")
            return
        try:
            from telegram.ext import ApplicationBuilder, CommandHandler
        except Exception as e:  # pragma: no cover
            log.warning(f"python-telegram-bot not available: {e}")
            return
        app = ApplicationBuilder().token(self.cfg.telegram_bot_token).build()

        async def _wrap(update, context, command):
            chat_id = update.effective_chat.id
            try:
                text = self.dispatch(command, chat_id)
            except Exception as e:
                text = f"ERROR: {e}"
            await context.bot.send_message(chat_id=chat_id, text=text)

        for cmd in list(COMMANDS) + ["/startbot", "/stopbot", "/pause", "/resume", "/closeall"]:
            app.add_handler(CommandHandler(cmd.lstrip("/"),
                (lambda c: (lambda u, ctx: asyncio.create_task(_wrap(u, ctx, c))))(cmd)))
        self._app = app
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        log.info("Telegram polling started.")

    async def stop(self) -> None:
        if self._app is None:
            return
        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception as e:  # pragma: no cover
            log.warning(f"Telegram stop error: {e}")
        self._app = None
