#!/usr/bin/env python3
import asyncio
import logging
import json
import time
import os
from collections import defaultdict
from typing import Dict, Any

from pybit.unified_trading import WebSocket, HTTP
from pybit.exceptions import InvalidRequestError
import pandas_ta as ta
import pandas as pd
import uvloop

uvloop.install()

# ---------------------- НАСТРОЙКИ ЛОГИРОВАНИЯ ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stoploss_manager.log")
    ]
)
logger = logging.getLogger("StopLossManager")

# ---------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------------
def safe_to_float(val: Any) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def load_users_from_json(json_path: str = "user_state.json") -> list:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)
    result = []
    for uid, data in all_users.items():
        if not data.get("banned") and data.get("api_key") and data.get("api_secret"):
            result.append({
                "user_id": uid,
                "api_key": data.get("api_key"),
                "api_secret": data.get("api_secret"),
                "trailing_start_pct": data.get("trailing_start_pct", {}).get("full", 5.0),
                "trailing_gap_pct": data.get("trailing_gap_pct", {}).get("full", 0.5),
            })
    return result

# ---------------------- КЛАСС МЕНЕДЖЕРА СТОП-ЛОССОВ ----------------------
class StopLossManager:
    def __init__(self, user_data: Dict[str, Any]):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        
        # Настройки трейлинга из конфига пользователя
        self.trailing_start_pct = safe_to_float(user_data.get("trailing_start_pct", 5.0))
        self.trailing_gap_pct = safe_to_float(user_data.get("trailing_gap_pct", 0.5))

        self.session = HTTP(testnet=False, api_key=self.api_key, api_secret=self.api_secret)
        
        self.ws_private = None
        self.ws_public = None
        
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.active_monitors: Dict[str, asyncio.Task] = {}
        
        self.price_cache: Dict[str, float] = {}
        self.price_tick_map: Dict[str, float] = {}

    async def start(self):
        logger.info(f"[{self.user_id}] Запуск менеджера стоп-лоссов.")
        await self.sync_positions_initial()
        
        # Запускаем WebSocket'ы параллельно
        await asyncio.gather(
            self.start_private_ws(),
            self.start_public_ws()
        )

    async def sync_positions_initial(self):
        """Первичная синхронизация позиций при старте."""
        try:
            response = await asyncio.to_thread(
                lambda: self.session.get_positions(category="linear", settleCoin="USDT")
            )
            if response.get("retCode") != 0:
                logger.error(f"[{self.user_id}] Ошибка API при первичной синхронизации: {response.get('retMsg')}")
                return

            live_positions = {
                pos["symbol"]: pos 
                for pos in response.get("result", {}).get("list", []) 
                if safe_to_float(pos.get("size", 0)) > 0
            }
            
            for symbol, pos_data in live_positions.items():
                self.update_local_position(pos_data)

            logger.info(f"[{self.user_id}] Первичная синхронизация завершена. Активных позиций: {len(self.open_positions)}")

        except Exception as e:
            logger.error(f"[{self.user_id}] Критическая ошибка при первичной синхронизации: {e}", exc_info=True)

    def update_local_position(self, pos_data: Dict[str, Any]):
        """Обновляет или создает локальное представление позиции и запускает/останавливает мониторинг."""
        symbol = pos_data.get("symbol")
        if not symbol:
            return

        size = safe_to_float(pos_data.get("size", 0))

        # Позиция закрыта
        if size == 0:
            if symbol in self.open_positions:
                logger.info(f"[{self.user_id}] Позиция {symbol} закрыта. Остановка мониторинга.")
                self.open_positions.pop(symbol, None)
                task = self.active_monitors.pop(symbol, None)
                if task:
                    task.cancel()
        # Позиция открыта или обновлена
        else:
            if symbol not in self.open_positions:
                logger.info(f"[{self.user_id}] Обнаружена новая позиция: {symbol}. Запуск мониторинга.")
            
            self.open_positions[symbol] = {
                "side": pos_data.get("side"),
                "size": size,
                "avg_price": safe_to_float(pos_data.get("avgPrice")),
                "leverage": safe_to_float(pos_data.get("leverage", 10.0)),
                "pos_idx": int(pos_data.get("positionIdx", 1 if pos_data.get("side") == "Buy" else 2)),
                "last_stop_price": safe_to_float(pos_data.get("stopLoss", 0))
            }

            if symbol not in self.active_monitors:
                self.active_monitors[symbol] = asyncio.create_task(self.monitor_position(symbol))

    async def start_private_ws(self):
        """Запускает приватный WebSocket для отслеживания изменений позиций."""
        def handle_message(msg):
            topic = msg.get("topic", "")
            if topic == "position":
                for pos_data in msg.get("data", []):
                    self.update_local_position(pos_data)

        self.ws_private = WebSocket(
            testnet=False, channel_type="private",
            api_key=self.api_key, api_secret=self.api_secret
        )
        self.ws_private.position_stream(callback=handle_message)
        logger.info(f"[{self.user_id}] Приватный WebSocket запущен.")

    async def start_public_ws(self):
        """Запускает публичный WebSocket для получения цен в реальном времени."""
        def handle_message(msg):
            data = msg.get("data", {})
            symbol = data.get("symbol")
            price = safe_to_float(data.get("lastPrice"))
            if symbol and price > 0:
                self.price_cache[symbol] = price

        # Подписываемся на все тикеры, так как основной бот может открыть любую позицию
        all_symbols = await self.get_all_linear_symbols()
        
        self.ws_public = WebSocket(testnet=False, channel_type="linear")
        # Bybit может иметь ограничение на количество символов в одной подписке, разбиваем на части
        chunk_size = 200
        for i in range(0, len(all_symbols), chunk_size):
            chunk = all_symbols[i:i + chunk_size]
            self.ws_public.ticker_stream(symbol=chunk, callback=handle_message)
            await asyncio.sleep(1) # Небольшая задержка между подписками
            
        logger.info(f"[{self.user_id}] Публичный WebSocket запущен. Подписано {len(all_symbols)} тикеров.")

    async def get_all_linear_symbols(self) -> list:
        """Получает список всех торгуемых USDT-M контрактов."""
        try:
            response = await asyncio.to_thread(
                lambda: self.session.get_instruments_info(category="linear")
            )
            return [item['symbol'] for item in response['result']['list']]
        except Exception as e:
            logger.error(f"[{self.user_id}] Не удалось получить список символов: {e}")
            return ["BTCUSDT", "ETHUSDT"] # Fallback

    async def ensure_price_tick(self, symbol: str) -> float:
        """Получает и кэширует шаг цены для символа."""
        if symbol in self.price_tick_map:
            return self.price_tick_map[symbol]
        try:
            response = await asyncio.to_thread(
                lambda: self.session.get_instruments_info(category="linear", symbol=symbol)
            )
            tick_size = safe_to_float(response['result']['list'][0]['priceFilter']['tickSize'])
            self.price_tick_map[symbol] = tick_size
            return tick_size
        except Exception as e:
            logger.warning(f"[{self.user_id}] Не удалось получить tickSize для {symbol}: {e}. Используем 1e-6.")
            return 1e-6

    async def monitor_position(self, symbol: str):
        """Основной цикл мониторинга для одной позиции."""
        while symbol in self.open_positions:
            try:
                pos = self.open_positions.get(symbol)
                last_price = self.price_cache.get(symbol)

                if not pos or not last_price:
                    await asyncio.sleep(1)
                    continue

                avg_price = pos['avg_price']
                side = pos['side']
                leverage = pos['leverage']
                
                pnl_pct = (((last_price - avg_price) / avg_price) * 100 * leverage) if side == "Buy" else \
                          (((avg_price - last_price) / avg_price) * 100 * leverage)

                if pnl_pct >= self.trailing_start_pct:
                    await self.set_trailing_stop(symbol, last_price, pos)

            except Exception as e:
                logger.error(f"[{self.user_id}] Ошибка в цикле мониторинга {symbol}: {e}", exc_info=True)
            
            await asyncio.sleep(1) # Проверка каждую секунду

    async def set_trailing_stop(self, symbol: str, last_price: float, pos: Dict[str, Any]):
        """Рассчитывает и устанавливает трейлинг-стоп."""
        try:
            side = pos['side']
            
            # Рассчитываем новую цену стопа с отступом от ТЕКУЩЕЙ цены
            if side == "Buy":
                stop_price_raw = last_price * (1 - self.trailing_gap_pct / 100)
            else: # Sell
                stop_price_raw = last_price * (1 + self.trailing_gap_pct / 100)
            
            tick_size = await self.ensure_price_tick(symbol)
            
            # Округляем до правильного шага цены
            stop_price = round(stop_price_raw / tick_size) * tick_size
            
            prev_stop = pos.get("last_stop_price", 0.0)

            # Не двигаем стоп, если он не улучшился
            if side == "Buy" and stop_price <= prev_stop:
                return
            if side == "Sell" and stop_price >= prev_stop:
                return
            
            # Не двигаем стоп, если изменение меньше шага цены
            if abs(stop_price - prev_stop) < tick_size:
                return

            await asyncio.to_thread(
                lambda: self.session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    positionIdx=pos['pos_idx'],
                    stopLoss=f"{stop_price:.8f}".rstrip('0').rstrip('.'),
                    slTriggerBy="LastPrice"
                )
            )
            
            # Обновляем локальное значение
            self.open_positions[symbol]["last_stop_price"] = stop_price
            logger.info(f"[{self.user_id}] {symbol} стоп обновлен на {stop_price:.6f}")

        except InvalidRequestError as e:
            # Код 110025: "Stop loss order price cannot be the same" - игнорируем, это не ошибка
            if e.status_code == 110025:
                # Обновляем локальное значение, чтобы избежать повторных запросов
                self.open_positions[symbol]["last_stop_price"] = stop_price
            else:
                logger.warning(f"[{self.user_id}] Ошибка API при установке стопа для {symbol}: {e}")
        except Exception as e:
            logger.error(f"[{self.user_id}] Критическая ошибка при установке стопа для {symbol}: {e}", exc_info=True)

# ---------------------- ГЛАВНАЯ ФУНКЦИЯ ЗАПУСКА ----------------------
async def main():
    users = load_users_from_json()
    if not users:
        logger.error("Нет активных пользователей в user_state.json. Завершение работы.")
        return

    tasks = []
    for user_data in users:
        manager = StopLossManager(user_data)
        tasks.append(manager.start())
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Менеджер стоп-лоссов остановлен пользователем.")