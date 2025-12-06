"""Exchange abstraction with paper and placeholder live client."""
from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List

from config import Settings, is_live_trading

logger = logging.getLogger("bot.exchange")


class OrderSide(Enum):
    """Order direction."""

    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time-in-force placeholder (only GTC for paper)."""

    GTC = "GTC"


@dataclass
class Order:
    """Represents a created order and its lifecycle."""

    id: str
    symbol: str
    side: OrderSide
    qty: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    opened_at: datetime
    closed_at: datetime | None
    status: str
    pnl: float


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    side: OrderSide
    qty: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


class BaseExchangeClient(ABC):
    """Abstract interface for exchange operations."""

    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, limit: int) -> List[dict]:
        ...

    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        leverage: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Order:
        ...

    @abstractmethod
    async def close_position(self, symbol: str) -> None:
        ...

    @abstractmethod
    async def get_open_positions(self) -> List[Position]:
        ...

    @abstractmethod
    async def get_account_equity(self) -> float:
        ...


class PaperExchangeClient(BaseExchangeClient):
    """Simple in-memory paper trading simulator."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.starting_equity = 10_000.0
        self.realized_pnl = 0.0
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self._price_cache: Dict[str, float] = {symbol: self._initial_price(symbol) for symbol in settings.SYMBOLS}
        logger.info("Initialized paper exchange with starting equity %.2f", self.starting_equity)

    def _initial_price(self, symbol: str) -> float:
        base_prices = {"BTCUSDT": 30_000.0, "ETHUSDT": 2_000.0, "SOLUSDT": 60.0}
        return base_prices.get(symbol.upper(), 100.0)

    def _next_price(self, symbol: str) -> float:
        prev = self._price_cache.get(symbol, self._initial_price(symbol))
        drift = random.uniform(-0.002, 0.002)
        noise = random.uniform(-0.001, 0.001)
        new_price = prev * (1 + drift + noise)
        new_price = max(new_price, 0.1)
        self._price_cache[symbol] = new_price
        return new_price

    async def get_klines(self, symbol: str, interval: str, limit: int) -> List[dict]:
        now = datetime.utcnow()
        price = self._price_cache.get(symbol, self._initial_price(symbol))
        klines: List[dict] = []
        for i in range(limit):
            price = self._next_price(symbol)
            delta = random.uniform(-0.003, 0.003)
            open_price = price * (1 - delta)
            close_price = price * (1 + delta)
            high = max(open_price, close_price) * (1 + abs(delta))
            low = min(open_price, close_price) * (1 - abs(delta))
            volume = random.uniform(50, 500)
            klines.append(
                {
                    "open": round(open_price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close_price, 4),
                    "volume": round(volume, 2),
                    "open_time": int((now - timedelta(minutes=limit - i)).timestamp() * 1000),
                }
            )
        return klines

    async def get_price(self, symbol: str) -> float:
        price = self._next_price(symbol)
        return price

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        leverage: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Order:
        price = await self.get_price(symbol)
        order_id = f"paper-{len(self.orders) + 1}"
        position = self.positions.get(symbol)
        pnl = 0.0
        if position:
            logger.debug("Updating existing position for %s", symbol)
            combined_qty = position.qty + qty if side == position.side else position.qty - qty
            if combined_qty <= 0:
                await self.close_position(symbol)
            else:
                position.qty = combined_qty
                position.entry_price = (position.entry_price + price) / 2
                self.positions[symbol] = position
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=price,
                unrealized_pnl=0.0,
                leverage=leverage,
            )
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            opened_at=datetime.utcnow(),
            closed_at=None,
            status="OPEN",
            pnl=pnl,
        )
        self.orders[order_id] = order
        logger.info("Created paper order %s %s qty=%.4f at %.2f", side.value, symbol, qty, price)
        return order

    async def close_position(self, symbol: str) -> None:
        position = self.positions.get(symbol)
        if not position:
            logger.debug("No open position to close for %s", symbol)
            return
        price = await self.get_price(symbol)
        pnl = self._compute_pnl(position, price)
        self.realized_pnl += pnl
        for order in self.orders.values():
            if order.symbol == symbol and order.status == "OPEN":
                order.status = "CLOSED"
                order.closed_at = datetime.utcnow()
                order.pnl = pnl
        logger.info("Closed position %s %s at %.2f PnL=%.2f", position.side.value, symbol, price, pnl)
        del self.positions[symbol]

    def _compute_pnl(self, position: Position, price: float) -> float:
        direction = 1 if position.side == OrderSide.BUY else -1
        return (price - position.entry_price) * position.qty * direction

    async def get_open_positions(self) -> List[Position]:
        positions: List[Position] = []
        for symbol, position in list(self.positions.items()):
            price = await self.get_price(symbol)
            unrealized = self._compute_pnl(position, price)
            positions.append(
                Position(
                    symbol=symbol,
                    side=position.side,
                    qty=position.qty,
                    entry_price=position.entry_price,
                    unrealized_pnl=unrealized,
                    leverage=position.leverage,
                )
            )
        return positions

    async def get_account_equity(self) -> float:
        unrealized_total = 0.0
        for symbol, position in self.positions.items():
            price = await self.get_price(symbol)
            unrealized_total += self._compute_pnl(position, price)
        return self.starting_equity + self.realized_pnl + unrealized_total


class LiveExchangeClient(BaseExchangeClient):
    """Placeholder for future live trading integration."""

    def __init__(self, settings: Settings) -> None:
        logger.warning(
            "LiveExchangeClient is not implemented; DO NOT use for production trading."
        )
        self.settings = settings

    async def get_klines(self, symbol: str, interval: str, limit: int) -> List[dict]:
        raise NotImplementedError("Live trading not implemented")

    async def get_price(self, symbol: str) -> float:
        raise NotImplementedError("Live trading not implemented")

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        leverage: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Order:
        raise NotImplementedError("Live trading not implemented")

    async def close_position(self, symbol: str) -> None:
        raise NotImplementedError("Live trading not implemented")

    async def get_open_positions(self) -> List[Position]:
        raise NotImplementedError("Live trading not implemented")

    async def get_account_equity(self) -> float:
        raise NotImplementedError("Live trading not implemented")


def create_exchange_client(settings: Settings) -> BaseExchangeClient:
    """Factory returning a paper or live client based on settings."""

    if not is_live_trading(settings):
        return PaperExchangeClient(settings)
    logger.warning("Live trading requested; returning unimplemented LiveExchangeClient.")
    return LiveExchangeClient(settings)


__all__ = [
    "Order",
    "OrderSide",
    "Position",
    "TimeInForce",
    "BaseExchangeClient",
    "PaperExchangeClient",
    "LiveExchangeClient",
    "create_exchange_client",
]
