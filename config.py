"""
config.py
---------
Centralized environment configuration for the bot.

All secrets and tunable values are read from environment variables
(populated from a local `.env` file via python-dotenv when running
outside of a host/container that already injects real env vars).
Nothing in this module -- or anywhere else in the project -- ever
hardcodes a token, API key, or chat ID.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Loads variables from a `.env` file in the current working directory,
# if one exists. In production (Docker, Railway, a plain VPS with
# systemd, etc.) you typically inject real environment variables
# directly, and this call is harmless -- it just finds no file and
# does nothing.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all runtime configuration the bot needs."""

    telegram_bot_token: str
    gemini_api_key: str
    gemini_model: str
    gemini_timeout_seconds: float
    admin_group_id: Optional[int]
    log_level: str


def _require_env(name: str) -> str:
    """Read a required environment variable or fail fast with a clear message.

    Failing fast at startup -- instead of the first time the value is
    actually used, deep inside a handler -- turns misconfiguration into
    an obvious, immediate error rather than a confusing runtime failure.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Copy .env.example to .env and fill in the missing value, "
            "or set it directly in your deployment environment."
        )
    return value


def _parse_optional_chat_id(raw: Optional[str]) -> Optional[int]:
    """Parse the optional ADMIN_GROUP_ID env var into an int, or None if unset."""
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ADMIN_GROUP_ID must be a numeric Telegram chat ID, got: {raw!r}"
        ) from exc


def load_settings() -> Settings:
    """Build and validate a `Settings` instance from the current environment."""
    return Settings(
        telegram_bot_token=_require_env("TELEGRAM_BOT_TOKEN"),
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        # Default chosen for current free-tier availability and low
        # latency. Override any time a newer/better free model ships --
        # no code changes required anywhere in the project.
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20")),
        # Optional: restrict processing to one specific group chat ID.
        # Leave unset to process every group the bot has been added to.
        admin_group_id=_parse_optional_chat_id(os.getenv("ADMIN_GROUP_ID")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
