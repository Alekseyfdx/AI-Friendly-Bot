"""Entry point orchestrating the trading bot."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import Settings, load_settings
from data_feed import GlobalDataFusion, init_data_fusion
from exchange import BaseExchangeClient, OrderSide, create_exchange_client
from strategy import RiskEngine, StrategyEngine

logger = logging.getLogger("bot.main")


def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if settings.LOG_FILE:
        file_handler = RotatingFileHandler(settings.LOG_FILE, maxBytes=1_000_000, backupCount=3)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)


def _extract_atr(strategy: StrategyEngine, klines: list[dict]) -> float:
    bars = strategy._bars_from_klines(klines)  # Using protected helper intentionally inside orchestrator
    return strategy.latest_atr_for_symbol("", bars)


async def main_loop(
    settings: Settings,
    exchange: BaseExchangeClient,
    fusion: GlobalDataFusion,
    risk_engine: RiskEngine,
    strategy: StrategyEngine,
) -> None:
    """Main trading scan loop."""

    while True:
        try:
            equity = await exchange.get_account_equity()
            open_positions = await exchange.get_open_positions()

            for symbol in settings.SYMBOLS:
                klines = await exchange.get_klines(symbol, settings.TIMEFRAME, settings.HISTORY_BARS)
                atr_value = _extract_atr(strategy, klines)
                signal = await strategy.generate_signal(symbol, exchange, fusion)

                if signal.action == "FLAT":
                    logger.debug("Symbol %s: FLAT (%s)", symbol, signal.reason)
                    continue

                if not risk_engine.can_open_new_position(equity, open_positions):
                    logger.info("RiskEngine blocked new position for %s: %s", symbol, signal.reason)
                    continue

                price = await exchange.get_price(symbol)
                qty = risk_engine.compute_position_size(equity, atr_value, price)
                if qty <= 0:
                    logger.info("Computed size <= 0 for %s, skipping", symbol)
                    continue

                side = OrderSide.BUY if signal.action == "LONG" else OrderSide.SELL
                order = await exchange.create_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    leverage=settings.MAX_LEVERAGE,
                    stop_loss=None,
                    take_profit=None,
                )
                logger.info(
                    "Opened %s %s qty=%.4f at %.4f, conf=%.2f, reason=%s",
                    side.value,
                    symbol,
                    qty,
                    order.entry_price,
                    signal.confidence,
                    signal.reason,
                )

            await asyncio.sleep(5)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Error in main loop")
            await asyncio.sleep(2)


async def async_main() -> None:
    settings = load_settings()
    _configure_logging(settings)

    logger.info(
        "Starting bot | mode=%s | data_fusion=%s | llm_advisor=%s",
        settings.TRADE_MODE,
        settings.ENABLE_DATA_FUSION,
        settings.ENABLE_LLM_ADVISOR,
    )

    exchange = create_exchange_client(settings)
    fusion = await init_data_fusion(settings)
    risk_engine = RiskEngine(settings)
    strategy = StrategyEngine(settings, risk_engine)

    loop_task: Optional[asyncio.Task[None]] = None
    try:
        loop_task = asyncio.create_task(main_loop(settings, exchange, fusion, risk_engine, strategy))
        await loop_task
    finally:
        if loop_task:
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task
        await fusion.stop()


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
