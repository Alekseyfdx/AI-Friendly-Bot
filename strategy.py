"""Strategy engine with indicators, risk checks, and optional LLM advisor."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import aiohttp

from config import Settings
from data_feed import GlobalDataFusion, SymbolSnapshot
from exchange import BaseExchangeClient, Position

logger = logging.getLogger("bot.strategy")


def ema(values: List[float], period: int) -> List[float]:
    """Compute exponential moving average series."""

    if not values or period <= 0:
        return []
    k = 2 / (period + 1)
    emas: List[float] = [values[0]]
    for price in values[1:]:
        emas.append(price * k + emas[-1] * (1 - k))
    return emas


def rsi(values: List[float], period: int) -> List[float]:
    """Compute Relative Strength Index."""

    if len(values) < period + 1:
        return []
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis: List[float] = []
    for i in range(period, len(values) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - (100 / (1 + rs)))
    return rsis


def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
    """Compute Average True Range."""

    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return []
    trs: List[float] = []
    for i in range(1, len(highs)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    atr_values: List[float] = []
    first_atr = sum(trs[:period]) / period
    atr_values.append(first_atr)
    for tr in trs[period:]:
        prev_atr = atr_values[-1]
        atr_values.append((prev_atr * (period - 1) + tr) / period)
    return atr_values


@dataclass
class Bar:
    """Single OHLCV bar."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """Trading signal produced by strategy logic."""

    symbol: str
    action: str  # "LONG", "SHORT", "FLAT"
    confidence: float
    reason: str


class RiskEngine:
    """Risk manager enforcing limits and cooldowns."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.starting_equity: float | None = None
        self.daily_realized_pnl = 0.0
        self.current_date = datetime.utcnow().date()
        self.cooldown_until: datetime | None = None

    def _ensure_starting_equity(self, equity: float) -> None:
        if self.starting_equity is None:
            self.starting_equity = equity
            logger.info("RiskEngine starting equity set to %.2f", equity)

    def _reset_if_new_day(self) -> None:
        today = datetime.utcnow().date()
        if today != self.current_date:
            logger.info("New day detected; resetting daily PnL and cooldown.")
            self.current_date = today
            self.daily_realized_pnl = 0.0
            self.cooldown_until = None

    def can_open_new_position(self, equity: float, open_positions: List[Position]) -> bool:
        self._ensure_starting_equity(equity)
        self._reset_if_new_day()

        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            logger.debug("Cooldown active until %s", self.cooldown_until)
            return False

        if len(open_positions) >= self.settings.MAX_OPEN_POSITIONS:
            return False

        if self.starting_equity is not None:
            max_loss_equity = self.starting_equity * (1 - self.settings.DAILY_MAX_LOSS_PCT / 100)
            if equity < max_loss_equity:
                logger.warning("Daily max loss reached; blocking new positions.")
                return False
        return True

    def compute_position_size(self, equity: float, atr_value: float, price: float) -> float:
        risk_per_trade = equity * self.settings.RISK_PER_TRADE_PCT / 100
        volatility_risk = max(atr_value, price * 0.001)
        size = risk_per_trade / volatility_risk
        return max(size, 0.0)

    def register_trade_pnl(self, pnl: float, equity: float) -> None:
        self.daily_realized_pnl += pnl
        threshold = equity * (self.settings.RISK_PER_TRADE_PCT / 100)
        if pnl < -threshold:
            cooldown_minutes = self.settings.COOLDOWN_AFTER_LOSS_MIN
            self.cooldown_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
            logger.warning(
                "Activating cooldown for %d minutes due to large loss %.2f",
                cooldown_minutes,
                pnl,
            )


class LLMAdvisor:
    """Optional OpenAI-based advisor to refine signals."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def refine_signal(self, signal: Signal, snapshot: SymbolSnapshot | None) -> Signal:
        if not self.settings.ENABLE_LLM_ADVISOR or not self.settings.OPENAI_API_KEY:
            return signal

        payload = {
            "model": self.settings.OPENAI_MODEL,
            "max_tokens": self.settings.OPENAI_MAX_TOKENS,
            "temperature": self.settings.OPENAI_TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a risk-aware trading validator. "
                        "Return a JSON object with fields action (LONG/SHORT/FLAT), confidence (0-1), reason."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "symbol": signal.symbol,
                            "proposed_action": signal.action,
                            "proposed_confidence": signal.confidence,
                            "snapshot": snapshot.__dict__ if snapshot else None,
                        }
                    ),
                },
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        url = f"{self.settings.OPENAI_BASE_URL}/chat/completions"
        try:
            timeout = aiohttp.ClientTimeout(total=self.settings.OPENAI_TIMEOUT_SEC)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        logger.warning("LLM advisor HTTP %s", resp.status)
                        return signal
                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        logger.warning("LLM advisor returned non-JSON content")
                        return signal
                    action = parsed.get("action", signal.action)
                    confidence = float(parsed.get("confidence", signal.confidence))
                    reason = parsed.get("reason", signal.reason)
                    if action not in {"LONG", "SHORT", "FLAT"}:
                        return signal
                    if confidence < 0 or confidence > 1:
                        confidence = signal.confidence
                    if confidence < signal.confidence * 0.5:
                        action = "FLAT"
                    return Signal(symbol=signal.symbol, action=action, confidence=confidence, reason=reason)
        except asyncio.TimeoutError:
            logger.warning("LLM advisor timed out")
        except aiohttp.ClientError:
            logger.warning("LLM advisor HTTP error", exc_info=True)
        except Exception:
            logger.exception("Unexpected error in LLM advisor")
        return signal


def compute_base_signal(bars: List[Bar], snapshot: SymbolSnapshot | None, settings: Settings) -> Signal:
    """Pure function to compute base signal from indicators and sentiment."""

    if len(bars) < 50:
        return Signal(symbol="", action="FLAT", confidence=0.0, reason="Insufficient data")

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]

    ema_fast = ema(closes, 12)
    ema_slow = ema(closes, 26)
    rsi_vals = rsi(closes, 14)
    atr_vals = atr(highs, lows, closes, 14)

    if not ema_fast or not ema_slow or not rsi_vals or not atr_vals:
        return Signal(symbol="", action="FLAT", confidence=0.0, reason="Indicator data missing")

    bias_long = ema_fast[-1] > ema_slow[-1]
    bias_short = ema_fast[-1] < ema_slow[-1]
    latest_rsi = rsi_vals[-1]
    sentiment = snapshot.combined_sentiment if snapshot else 0.0
    confidence = 0.5
    reason_parts: list[str] = []

    if bias_long:
        confidence += 0.15
        reason_parts.append("EMA fast above slow")
    if bias_short:
        confidence += 0.15
        reason_parts.append("EMA fast below slow")
    if latest_rsi < 30:
        confidence += 0.1
        reason_parts.append("RSI oversold")
    elif latest_rsi > 70:
        confidence += 0.1
        reason_parts.append("RSI overbought")

    if sentiment != 0:
        confidence += min(abs(sentiment), 0.3)
        reason_parts.append(f"Sentiment {sentiment:.2f}")

    action = "FLAT"
    if bias_long and latest_rsi < 70 and sentiment >= 0:
        action = "LONG"
    elif bias_short and latest_rsi > 30 and sentiment <= 0:
        action = "SHORT"

    confidence = max(0.0, min(1.0, confidence))
    return Signal(symbol="", action=action, confidence=confidence, reason="; ".join(reason_parts))


class StrategyEngine:
    """Generate trading signals using indicators, sentiment, and optional LLM advisor."""

    def __init__(self, settings: Settings, risk_engine: RiskEngine) -> None:
        self.settings = settings
        self.risk_engine = risk_engine
        self.llm_advisor = LLMAdvisor(settings)

    def _bars_from_klines(self, klines: List[dict]) -> List[Bar]:
        bars: List[Bar] = []
        for k in klines:
            bars.append(
                Bar(
                    ts=datetime.utcfromtimestamp(k["open_time"] / 1000),
                    open=float(k["open"]),
                    high=float(k["high"]),
                    low=float(k["low"]),
                    close=float(k["close"]),
                    volume=float(k["volume"]),
                )
            )
        return bars

    def _latest_atr(self, bars: List[Bar]) -> float:
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        atr_values = atr(highs, lows, closes, 14)
        return atr_values[-1] if atr_values else max(closes[-1] * 0.002, 1e-6)

    async def generate_signal(self, symbol: str, exchange: BaseExchangeClient, fusion: GlobalDataFusion) -> Signal:
        klines = await exchange.get_klines(symbol, self.settings.TIMEFRAME, self.settings.HISTORY_BARS)
        bars = self._bars_from_klines(klines)
        snapshot = fusion.get_snapshot(symbol)
        base_signal = compute_base_signal(bars, snapshot, self.settings)
        base_signal.symbol = symbol

        if base_signal.confidence < self.settings.MIN_SIGNAL_CONFIDENCE:
            return Signal(symbol=symbol, action="FLAT", confidence=base_signal.confidence, reason="Low confidence")

        refined_signal = await self.llm_advisor.refine_signal(base_signal, snapshot)
        return refined_signal

    def latest_atr_for_symbol(self, symbol: str, bars: List[Bar]) -> float:
        return self._latest_atr(bars)


__all__ = [
    "Bar",
    "Signal",
    "RiskEngine",
    "StrategyEngine",
    "compute_base_signal",
    "ema",
    "rsi",
    "atr",
]
