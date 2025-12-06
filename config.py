"""Central configuration for the trading bot."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("bot.config")


@dataclass
class Settings:
    """Container for user-configurable bot settings."""

    TRADE_MODE: str
    SYMBOLS: List[str]
    MAX_OPEN_POSITIONS: int
    RISK_PER_TRADE_PCT: float
    DAILY_MAX_LOSS_PCT: float
    MAX_LEVERAGE: float

    ENABLE_DATA_FUSION: bool
    DF_WINDOW_MIN: int
    NEWS_POLL_INTERVAL_SEC: int
    SOCIAL_POLL_INTERVAL_SEC: int
    MACRO_POLL_INTERVAL_SEC: int

    BINANCE_API_KEY: Optional[str]
    BINANCE_API_SECRET: Optional[str]
    USE_TESTNET: bool
    REQUEST_TIMEOUT_SEC: int

    LOG_LEVEL: str
    LOG_FILE: Optional[str]

    TIMEFRAME: str
    HISTORY_BARS: int
    MIN_SIGNAL_CONFIDENCE: float
    COOLDOWN_AFTER_LOSS_MIN: int

    HARD_LOCK_LIVE: bool

    ENABLE_LLM_ADVISOR: bool
    OPENAI_API_KEY: Optional[str]
    OPENAI_BASE_URL: str
    OPENAI_MODEL: str
    OPENAI_MAX_TOKENS: int
    OPENAI_TEMPERATURE: float
    OPENAI_TIMEOUT_SEC: int


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return default
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s, using default %.3f", name, default)
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s, using default %d", name, default)
        return default


def load_settings() -> Settings:
    """Load settings from environment variables with safe defaults."""

    symbols = _parse_list(os.environ.get("SYMBOLS"), ["BTCUSDT", "ETHUSDT"])

    settings = Settings(
        TRADE_MODE=os.environ.get("TRADE_MODE", "futures_paper"),
        SYMBOLS=symbols,
        MAX_OPEN_POSITIONS=_get_env_int("MAX_OPEN_POSITIONS", 5),
        RISK_PER_TRADE_PCT=_get_env_float("RISK_PER_TRADE_PCT", 0.3),
        DAILY_MAX_LOSS_PCT=_get_env_float("DAILY_MAX_LOSS_PCT", 2.0),
        MAX_LEVERAGE=_get_env_float("MAX_LEVERAGE", 3.0),
        ENABLE_DATA_FUSION=_parse_bool(os.environ.get("ENABLE_DATA_FUSION"), True),
        DF_WINDOW_MIN=_get_env_int("DF_WINDOW_MIN", 60),
        NEWS_POLL_INTERVAL_SEC=_get_env_int("NEWS_POLL_INTERVAL_SEC", 30),
        SOCIAL_POLL_INTERVAL_SEC=_get_env_int("SOCIAL_POLL_INTERVAL_SEC", 20),
        MACRO_POLL_INTERVAL_SEC=_get_env_int("MACRO_POLL_INTERVAL_SEC", 300),
        BINANCE_API_KEY=os.environ.get("BINANCE_API_KEY"),
        BINANCE_API_SECRET=os.environ.get("BINANCE_API_SECRET"),
        USE_TESTNET=_parse_bool(os.environ.get("USE_TESTNET"), True),
        REQUEST_TIMEOUT_SEC=_get_env_int("REQUEST_TIMEOUT_SEC", 10),
        LOG_LEVEL=os.environ.get("LOG_LEVEL", "INFO"),
        LOG_FILE=os.environ.get("LOG_FILE"),
        TIMEFRAME=os.environ.get("TIMEFRAME", "1m"),
        HISTORY_BARS=_get_env_int("HISTORY_BARS", 500),
        MIN_SIGNAL_CONFIDENCE=_get_env_float("MIN_SIGNAL_CONFIDENCE", 0.55),
        COOLDOWN_AFTER_LOSS_MIN=_get_env_int("COOLDOWN_AFTER_LOSS_MIN", 15),
        HARD_LOCK_LIVE=_parse_bool(os.environ.get("HARD_LOCK_LIVE"), True),
        ENABLE_LLM_ADVISOR=_parse_bool(os.environ.get("ENABLE_LLM_ADVISOR"), False),
        OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY"),
        OPENAI_BASE_URL=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        OPENAI_MODEL=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
        OPENAI_MAX_TOKENS=_get_env_int("OPENAI_MAX_TOKENS", 256),
        OPENAI_TEMPERATURE=_get_env_float("OPENAI_TEMPERATURE", 0.3),
        OPENAI_TIMEOUT_SEC=_get_env_int("OPENAI_TIMEOUT_SEC", 10),
    )

    if settings.TRADE_MODE == "futures_live" and settings.HARD_LOCK_LIVE:
        logger.warning("HARD_LOCK_LIVE is enabled; forcing paper mode despite futures_live setting.")

    if settings.ENABLE_LLM_ADVISOR and not settings.OPENAI_API_KEY:
        logger.warning("LLM advisor enabled but OPENAI_API_KEY missing; disabling advisor.")
        settings.ENABLE_LLM_ADVISOR = False

    return settings


def is_live_trading(settings: Settings) -> bool:
    """Return True if live trading is allowed under current settings."""

    return settings.TRADE_MODE == "futures_live" and not settings.HARD_LOCK_LIVE


__all__ = ["Settings", "load_settings", "is_live_trading"]
