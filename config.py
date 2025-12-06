"""Central configuration for the trading bot with validation."""
from __future__ import annotations

import logging
from typing import List, Optional

from pydantic import BaseSettings, Field, validator

logger = logging.getLogger("bot.config")


class Settings(BaseSettings):
    """Validated settings container sourced from environment variables.

    The model enforces sensible defaults, validates numeric ranges, and parses
    comma-separated symbols. Values are read from environment variables or an
    optional ``.env`` file; type validation prevents booting the bot with
    malformed configuration.
    """

    TRADE_MODE: str = Field("futures_paper", description="futures_paper or futures_live")
    SYMBOLS: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    MAX_OPEN_POSITIONS: int = Field(5, ge=1)
    RISK_PER_TRADE_PCT: float = Field(0.3, ge=0.0, le=100.0)
    DAILY_MAX_LOSS_PCT: float = Field(2.0, ge=0.0, le=100.0)
    MAX_LEVERAGE: float = Field(3.0, ge=1.0)

    ENABLE_DATA_FUSION: bool = True
    DF_WINDOW_MIN: int = Field(60, ge=1)
    NEWS_POLL_INTERVAL_SEC: int = Field(30, ge=1)
    SOCIAL_POLL_INTERVAL_SEC: int = Field(20, ge=1)
    MACRO_POLL_INTERVAL_SEC: int = Field(300, ge=1)

    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None
    USE_TESTNET: bool = True
    REQUEST_TIMEOUT_SEC: int = Field(10, ge=1)

    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None

    TIMEFRAME: str = "1m"
    HISTORY_BARS: int = Field(500, ge=10)
    MIN_SIGNAL_CONFIDENCE: float = Field(0.55, ge=0.0, le=1.0)
    COOLDOWN_AFTER_LOSS_MIN: int = Field(15, ge=0)

    HARD_LOCK_LIVE: bool = True

    ENABLE_LLM_ADVISOR: bool = False
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = Field(256, ge=1)
    OPENAI_TEMPERATURE: float = Field(0.3, ge=0.0, le=2.0)
    OPENAI_TIMEOUT_SEC: int = Field(10, ge=1)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("TRADE_MODE", pre=True)
    def _normalize_mode(cls, raw: str) -> str:
        normalized = (raw or "futures_paper").strip().lower()
        if normalized not in {"futures_paper", "futures_live"}:
            logger.warning("Invalid TRADE_MODE '%s'; defaulting to futures_paper", raw)
            return "futures_paper"
        return normalized

    @validator("SYMBOLS", pre=True)
    def _split_symbols(cls, value):  # type: ignore[override]
        if isinstance(value, str):
            parsed = [item.strip().upper() for item in value.split(",") if item.strip()]
            return parsed or ["BTCUSDT", "ETHUSDT"]
        if isinstance(value, list) and value:
            return [str(item).strip().upper() for item in value if str(item).strip()]
        return ["BTCUSDT", "ETHUSDT"]

    @validator("MIN_SIGNAL_CONFIDENCE")
    def _clamp_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            logger.warning("MIN_SIGNAL_CONFIDENCE out of range; clamping to [0,1]")
        return max(0.0, min(1.0, value))

    @validator("ENABLE_LLM_ADVISOR", always=True)
    def _disable_llm_without_key(cls, value: bool, values):
        api_key = values.get("OPENAI_API_KEY")
        if value and not api_key:
            logger.warning("LLM advisor enabled but OPENAI_API_KEY missing; disabling advisor.")
            return False
        return value

    @validator("TRADE_MODE", always=True)
    def _respect_hard_lock(cls, value: str, values):
        hard_lock = values.get("HARD_LOCK_LIVE", True)
        if value == "futures_live" and hard_lock:
            logger.warning("HARD_LOCK_LIVE enabled; forcing paper mode despite futures_live setting.")
            return "futures_paper"
        return value


def load_settings() -> Settings:
    """Load validated settings from environment variables or .env file."""

    return Settings()


def is_live_trading(settings: Settings) -> bool:
    """Return True if live trading is allowed under current settings."""

    return settings.TRADE_MODE == "futures_live" and not settings.HARD_LOCK_LIVE


def describe_settings(settings: Settings) -> str:
    """Return a concise human-readable description of key settings."""

    return (
        f"mode={settings.TRADE_MODE} symbols={','.join(settings.SYMBOLS)} "
        f"fusion={'on' if settings.ENABLE_DATA_FUSION else 'off'} "
        f"llm={'on' if settings.ENABLE_LLM_ADVISOR else 'off'} max_pos={settings.MAX_OPEN_POSITIONS}"
    )


__all__ = ["Settings", "load_settings", "is_live_trading", "describe_settings"]
