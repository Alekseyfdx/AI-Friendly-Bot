"""Mock data fusion layer for news, social, and macro sentiment."""
from __future__ import annotations

import asyncio
import logging
import random
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

from config import Settings

logger = logging.getLogger("bot.data_feed")


class NLPEngine:
    """Heuristic sentiment extractor returning (sentiment, impact, symbols).

    Sentiment is in [-1.0, 1.0] where -1 is very negative and 1 is very positive.
    Impact is in [0.0, 1.0] and increases when important markers are present.
    Symbols are inferred from common ticker patterns like BTC, ETH, SOL, BTCUSDT, or $BTC.
    """

    def __init__(self) -> None:
        self.positive_words = {
            "bullish",
            "rally",
            "pump",
            "approve",
            "win",
            "surge",
            "support",
            "recover",
            "momentum",
            "bid",
        }
        self.negative_words = {
            "bearish",
            "dump",
            "ban",
            "hack",
            "reject",
            "liquidation",
            "fear",
            "selloff",
            "slump",
            "fraud",
        }
        self.impact_markers = {"sec", "etf", "breaking", "%", "fomc", "rate hike", "regulation", "macro"}
        self.symbol_patterns = [
            re.compile(r"\bBTC(?:USDT)?\b", re.IGNORECASE),
            re.compile(r"\bETH(?:USDT)?\b", re.IGNORECASE),
            re.compile(r"\bSOL(?:USDT)?\b", re.IGNORECASE),
            re.compile(r"\$(BTC|ETH|SOL)\b", re.IGNORECASE),
        ]

    def analyze(self, text: str) -> tuple[float, float, list[str]]:
        """Return sentiment, impact, and symbols for provided text."""

        lowered = text.lower()
        pos_hits = sum(lowered.count(word) for word in self.positive_words)
        neg_hits = sum(lowered.count(word) for word in self.negative_words)
        total_hits = pos_hits + neg_hits
        sentiment = 0.0
        if total_hits:
            sentiment = (pos_hits - neg_hits) / total_hits
        else:
            sentiment = random.uniform(-0.1, 0.1)
        sentiment = max(-1.0, min(1.0, sentiment))

        impact = 0.1
        if any(marker in lowered for marker in self.impact_markers):
            impact += 0.4
        if "%" in lowered:
            impact += 0.1
        impact = max(0.0, min(1.0, impact))

        symbols: list[str] = []
        for pattern in self.symbol_patterns:
            for match in pattern.findall(text):
                token = match if isinstance(match, str) else match[0]
                symbol = token.upper().replace("$", "")
                if not symbol.endswith("USDT"):
                    symbol = f"{symbol}USDT"
                if symbol not in symbols:
                    symbols.append(symbol)

        return sentiment, impact, symbols


dataclass_kwargs = dict(kw_only=False)


@dataclass
class FusionEvent:
    """Represents a processed information event with sentiment."""

    timestamp: datetime
    source: str
    symbols: List[str]
    raw_text: str
    sentiment: float
    impact: float
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class SymbolSnapshot:
    """Aggregated view of recent sentiment for a symbol."""

    symbol: str
    ts: datetime
    news_sentiment_5m: float
    news_sentiment_30m: float
    social_sentiment_5m: float
    high_impact_events_5m: int
    macro_risk: float
    combined_sentiment: float


class GlobalDataFusion:
    """Manage sentiment signals from multiple mocked sources."""

    def __init__(self, settings: Settings, nlp_engine: Optional[NLPEngine] = None) -> None:
        self.settings = settings
        self.nlp = nlp_engine or NLPEngine()
        self.events: Deque[FusionEvent] = deque()
        self.symbol_events: Dict[str, Deque[FusionEvent]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start background mock data loops if enabled."""

        if not self.settings.ENABLE_DATA_FUSION:
            logger.info("Data fusion disabled by settings.")
            return
        logger.info("Starting data fusion background tasks.")
        self._stop_event.clear()
        self._tasks = [
            asyncio.create_task(self._mock_news_loop(), name="mock_news"),
            asyncio.create_task(self._mock_social_loop(), name="mock_social"),
        ]

    async def stop(self) -> None:
        """Cancel running background tasks."""

        if not self._tasks:
            return
        logger.info("Stopping data fusion background tasks.")
        self._stop_event.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                logger.debug("Task %s cancelled", task.get_name())
        self._tasks.clear()

    def update_state_from_events(self, events: list[FusionEvent]) -> None:
        """Add new events and drop outdated ones."""

        window = timedelta(minutes=self.settings.DF_WINDOW_MIN)
        now = datetime.utcnow()
        for event in events:
            self.events.append(event)
            for symbol in event.symbols:
                bucket = self.symbol_events.setdefault(symbol, deque())
                bucket.append(event)

        while self.events and now - self.events[0].timestamp > window:
            old = self.events.popleft()
            for symbol in old.symbols:
                bucket = self.symbol_events.get(symbol)
                while bucket and now - bucket[0].timestamp > window:
                    bucket.popleft()
                if bucket is not None and not bucket:
                    self.symbol_events.pop(symbol, None)

    def get_snapshot(self, symbol: str) -> SymbolSnapshot:
        """Return aggregated snapshot for the given symbol."""

        events = self.symbol_events.get(symbol)
        now = datetime.utcnow()
        if events is None:
            return SymbolSnapshot(
                symbol=symbol,
                ts=now,
                news_sentiment_5m=0.0,
                news_sentiment_30m=0.0,
                social_sentiment_5m=0.0,
                high_impact_events_5m=0,
                macro_risk=0.0,
                combined_sentiment=0.0,
            )

        cutoff_5m = now - timedelta(minutes=5)
        cutoff_30m = now - timedelta(minutes=30)
        news_5m: list[float] = []
        news_30m: list[float] = []
        social_5m: list[float] = []
        high_impact = 0
        macro_impacts: list[float] = []

        for event in list(events):
            if event.timestamp < cutoff_30m:
                continue
            if event.source == "news":
                news_30m.append(event.sentiment)
                if event.timestamp >= cutoff_5m:
                    news_5m.append(event.sentiment)
            if event.source == "social" and event.timestamp >= cutoff_5m:
                social_5m.append(event.sentiment)
            if event.impact >= 0.7 and event.timestamp >= cutoff_5m:
                high_impact += 1
            if event.source == "macro" and event.timestamp >= cutoff_30m:
                macro_impacts.append(event.impact)

        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        news_sentiment_5m = _mean(news_5m)
        news_sentiment_30m = _mean(news_30m)
        social_sentiment_5m = _mean(social_5m)
        macro_risk = _mean(macro_impacts)
        combined_sentiment = (news_sentiment_30m * 0.5) + (social_sentiment_5m * 0.3) + (macro_risk * 0.2)
        combined_sentiment = max(-1.0, min(1.0, combined_sentiment))

        return SymbolSnapshot(
            symbol=symbol,
            ts=now,
            news_sentiment_5m=news_sentiment_5m,
            news_sentiment_30m=news_sentiment_30m,
            social_sentiment_5m=social_sentiment_5m,
            high_impact_events_5m=high_impact,
            macro_risk=macro_risk,
            combined_sentiment=combined_sentiment,
        )

    async def _mock_news_loop(self) -> None:
        """Generate synthetic news events periodically."""

        sentiment_options = ["surges", "drops", "steady"]
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.settings.NEWS_POLL_INTERVAL_SEC)
                symbol = random.choice(self.settings.SYMBOLS)
                mood = random.choice(sentiment_options)
                direction_text = "bullish rally" if mood == "surges" else "bearish slump" if mood == "drops" else "steady consolidation"
                text = f"{symbol} {mood} after {random.choice(['ETF rumor', 'SEC update', 'macro shift'])} with {direction_text}"
                sentiment, impact, symbols = self.nlp.analyze(text)
                symbols = symbols or [symbol]
                event = FusionEvent(
                    timestamp=datetime.utcnow(),
                    source="news",
                    symbols=symbols,
                    raw_text=text,
                    sentiment=sentiment,
                    impact=impact,
                    meta={},
                )
                self.update_state_from_events([event])
                logger.debug("News event: %s | sent=%.2f impact=%.2f", text, sentiment, impact)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in mock news loop")

    async def _mock_social_loop(self) -> None:
        """Generate synthetic social sentiment events periodically."""

        handles = ["@traderjoe", "@crypto_ai", "@onchain_news", "@defi_alerts"]
        vibes = ["bullish", "bearish", "neutral"]
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.settings.SOCIAL_POLL_INTERVAL_SEC)
                symbol = random.choice(self.settings.SYMBOLS)
                vibe = random.choice(vibes)
                blurbs = {
                    "bullish": f"community is bullish after pump on {symbol}",
                    "bearish": f"rumors of selloff for {symbol} causing fear",
                    "neutral": f"{symbol} traders waiting for clarity",
                }
                author = random.choice(handles)
                text = f"{author}: {blurbs[vibe]}"
                sentiment, impact, symbols = self.nlp.analyze(text)
                symbols = symbols or [symbol]
                event = FusionEvent(
                    timestamp=datetime.utcnow(),
                    source="social",
                    symbols=symbols,
                    raw_text=text,
                    sentiment=sentiment,
                    impact=impact * 0.6,
                    meta={"author": author},
                )
                self.update_state_from_events([event])
                logger.debug("Social event: %s | sent=%.2f impact=%.2f", text, sentiment, impact)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in mock social loop")


async def init_data_fusion(settings: Settings) -> GlobalDataFusion:
    """Initialize and start data fusion service."""

    fusion = GlobalDataFusion(settings)
    await fusion.start()
    return fusion


__all__ = [
    "FusionEvent",
    "GlobalDataFusion",
    "NLPEngine",
    "SymbolSnapshot",
    "init_data_fusion",
]
