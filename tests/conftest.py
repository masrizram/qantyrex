"""Shared pytest fixtures."""
import os
import sys

# Ensure workspace root is importable when running from inside trading_bot/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest


@pytest.fixture
def tmp_db_url(tmp_path):
    return f"sqlite:///{tmp_path/'test.db'}"
