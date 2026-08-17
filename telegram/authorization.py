"""Telegram authorization: only whitelisted chat IDs can run commands."""
from __future__ import annotations

from typing import Iterable, Set

from ..core.exceptions import AuthorizationDenied


class Authorization:
    def __init__(self, allowed_chat_ids: Iterable[int]) -> None:
        self.allowed: Set[int] = set(int(x) for x in allowed_chat_ids)

    def is_authorized(self, chat_id: int) -> bool:
        return chat_id in self.allowed

    def require(self, chat_id: int) -> None:
        if not self.is_authorized(chat_id):
            raise AuthorizationDenied(
                f"ACCESS DENIED: chat_id {chat_id} not in whitelist")

    def empty(self) -> bool:
        return len(self.allowed) == 0
