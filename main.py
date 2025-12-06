import asyncio
import contextlib
import csv
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config import Settings, describe_settings, is_live_trading, load_settings
from data_feed import GlobalDataFusion, init_data_fusion
from exchange import BaseExchangeClient, OrderSide, Position, create_exchange_client
from strategy import Bar, RiskEngine, StrategyEngine, latest_atr_value

logger = logging.getLogger("bot.main")


def _configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if settings.LOG_FILE:
        file_handler = RotatingFileHandler(settings.LOG_FILE, maxBytes=1_000_000, backupCount=3)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)


def _write_trade_log(row: list[str]) -> None:
    path = Path("trades.csv")
    header = [
        "timestamp",
        "symbol",
        "side",
        "entry_price",
        "exit_price",
        "qty",
        "pnl",
        "equity_before",
        "equity_after",
        "reason",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


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
            opened_this_cycle = 0

            for symbol in settings.SYMBOLS:
                if opened_this_cycle >= settings.MAX_OPEN_POSITIONS:
                    logger.info("Max positions reached for this cycle; skipping remaining symbols")
                    break

                klines = await exchange.get_klines(symbol, settings.TIMEFRAME, settings.HISTORY_BARS)
                bars: list[Bar] = strategy.bars_from_klines(klines)
                snapshot = fusion.get_snapshot(symbol)
                signal = await strategy.generate_signal_from_bars(symbol, bars, snapshot)
                atr_value = latest_atr_value(bars)

                existing = [p for p in open_positions if p.symbol == symbol]
                if existing and (signal.action == "FLAT" or existing[0].side.value != signal.action):
                    pnl = await exchange.close_position(symbol)
                    risk_engine.register_trade_pnl(pnl, equity)
                    equity = await exchange.get_account_equity()
                    direction = 1 if existing[0].side == OrderSide.BUY else -1
                    exit_price = existing[0].entry_price + (pnl / max(existing[0].qty * direction, 1e-9))
                    _write_trade_log([
                        f"{asyncio.get_event_loop().time():.3f}",
                        symbol,
                        existing[0].side.value,
                        f"{existing[0].entry_price:.4f}",
                        f"{exit_price:.4f}",
                        f"{existing[0].qty:.6f}",
                        f"{pnl:.4f}",
                        f"{equity - pnl:.2f}",
                        f"{equity:.2f}",
                        signal.reason,
                    ])
                    open_positions = [p for p in open_positions if p.symbol != symbol]
                    if signal.action == "FLAT":
                        continue

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

                stop_loss = price - 1.5 * atr_value if signal.action == "LONG" else price + 1.5 * atr_value
                take_profit = price + 3.0 * atr_value if signal.action == "LONG" else price - 3.0 * atr_value

                side = OrderSide.BUY if signal.action == "LONG" else OrderSide.SELL
                order = await exchange.create_order(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    leverage=settings.MAX_LEVERAGE,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
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
                opened_this_cycle += 1
                open_positions.append(
                    Position(symbol=symbol, side=side, qty=qty, entry_price=order.entry_price, unrealized_pnl=0.0, leverage=settings.MAX_LEVERAGE)
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
        "Starting bot | %s | live=%s",
        describe_settings(settings),
        is_live_trading(settings),
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
