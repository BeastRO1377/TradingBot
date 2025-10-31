# main.py
import asyncio
import signal
import sys
import json
import pandas as pd
import os
from typing import List, Dict, Any

# Сначала настраиваем логирование
from config import setup_logging
setup_logging()

import logging
from aiogram import types
from aiogram.filters import Command

# Импортируем наши модули
import config
from utils import safe_to_float
from data_manager import PublicWebSocketManager
from bot_core import TradingBot
from telegram_bot import dp, router, router_admin, bot as telegram_bot, GLOBAL_BOTS

logger = logging.getLogger(__name__)

def load_users_from_json(json_path: str = config.USER_STATE_FILE) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)

    result: List[Dict[str, Any]] = []
    for uid, data in all_users.items():
        if data.get("banned", False) or not data.get("registered", True):
            continue
        
        # Собираем полный конфиг для каждого пользователя
        user_config = {
            "user_id": uid,
            "api_key": data.get("api_key"),
            "api_secret": data.get("api_secret"),
            "gemini_api_key": data.get("gemini_api_key"),
            "openai_api_key": data.get("openai_api_key"),
            "ai_provider": data.get("ai_provider", "ollama"),
            "strategy": data.get("strategy"),
            "volume": safe_to_float(data.get("volume", 0.0)),
            "max_total_volume": safe_to_float(data.get("max_total_volume", 0.0)),
            "mode": data.get("mode", "real"),
        }
        # Добавляем специфичные для AI настройки, если они есть
        user_config.update({k: v for k, v in data.items() if k.startswith('ai_') or k.startswith('ollama_')})
        result.append(user_config)
        
    return result

def load_golden_params(csv_path: str = config.GOLDEN_PARAMS_CSV) -> dict:
    default_params = {
        "Buy": {"period_iters": 4, "price_change": 1.7, "volume_change": 200, "oi_change": 1.5},
        "Sell": {"period_iters": 4, "price_change": 1.8, "volume_change": 200, "oi_change": 1.2}
    }
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            overrides = {}
            for _, row in df.iterrows():
                key = (row["symbol"], row["side"])
                overrides[key] = {k: safe_to_float(row[k]) for k in row.index if k not in ["symbol", "side"]}
            logger.info(f"Loaded {len(overrides)} symbol-specific golden params.")
            return {**default_params, **overrides}
        except Exception as e:
            logger.error(f"Error loading golden_params.csv: {e}")
    return default_params

async def run_all() -> None:
    """Главная функция, которая инициализирует и запускает всех ботов."""

    # --- НАЧАЛО БЛОКА ДЛЯ ИСПРАВЛЕНИЯ (MONKEY PATCH) ---
    from pybit._websocket_stream import _WebSocketManager
    from websocket._exceptions import WebSocketConnectionClosedException

    # 1. Создаем нашу безопасную версию функции
    def _safe_send_custom_ping(self):
        try:
            # Это оригинальная логика из библиотеки pybit
            if self.custom_ping_message:
                self.ws.send(self.custom_ping_message)
            else:
                self.ws.ping()
        except WebSocketConnectionClosedException:
            # Это ожидаемая ошибка при завершении работы.
            # Мы просто игнорируем ее, чтобы лог оставался чистым.
            pass 
        except Exception as e:
            # Другие, непредвиденные ошибки мы все же хотим видеть
            logger.error(f"An unexpected error occurred in the pybit ping thread: {e}")

    # 2. "На лету" заменяем оригинальную функцию в классе библиотеки на нашу
    _WebSocketManager._send_custom_ping = _safe_send_custom_ping
    logger.info("Applied monkey patch to pybit._WebSocketManager to suppress shutdown ping error.")
    # --- КОНЕЦ БЛОКА ДЛЯ ИСПРАВЛЕНИЯ ---
    
    """Главная функция, которая инициализирует и запускает всех ботов."""
    users = load_users_from_json()
    if not users:
        logger.critical("Нет активных пользователей в user_state.json. Запуск невозможен.")
        return

    golden_param_store = load_golden_params()
    shared_ws = PublicWebSocketManager(symbols=["BTCUSDT", "ETHUSDT"])

    for u_data in users:
        bot = TradingBot(user_data=u_data, shared_ws=shared_ws, golden_param_store=golden_param_store)
        GLOBAL_BOTS.append(bot)

    public_ws_task = asyncio.create_task(shared_ws.start())
    
    await shared_ws.backfill_history()
    await shared_ws.ready_event.wait()

    if dp:
        telegram_task = asyncio.create_task(dp.start_polling(telegram_bot, skip_updates=True))
    else:
        telegram_task = None
        logger.warning("Telegram bot is not initialized. Skipping polling.")


    for b in GLOBAL_BOTS:
        b.load_ml_models()

    bot_tasks = [asyncio.create_task(b.start()) for b in GLOBAL_BOTS]

    # Настройка Graceful Shutdown
    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    
    # Устанавливаем обработчики сигналов для корректного завершения
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    logger.info("Бот запущен. Нажмите Ctrl+C для остановки.")
    
    try:
        await stop # Эта строка будет ждать вечно, пока не сработает signal_handler
    finally:
        logger.info("Начинаем процедуру остановки...")
        
        # Отменяем все запущенные задачи
        public_ws_task.cancel()
        if telegram_task:
            telegram_task.cancel()
        for task in bot_tasks:
            task.cancel()
        
        # Собираем все задачи, чтобы дождаться их завершения
        all_tasks = bot_tasks + [public_ws_task]
        if telegram_task:
            all_tasks.append(telegram_task)
        
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Корректно останавливаем экземпляры ботов
        for bot_instance in GLOBAL_BOTS:
            await bot_instance.stop()
        
        logger.info("Все компоненты успешно остановлены.")


async def run_training():
    """Запускает процесс создания и обучения модели."""
    users = load_users_from_json()
    if not users:
        logger.error("Нет пользователей в user_state.json. Невозможно запустить тренировку.")
        return

    user_data = users
    logger.info(f"Используются данные пользователя: {user_data['user_id']}")
    
    shared_ws = PublicWebSocketManager(symbols=["BTCUSDT", "ETHUSDT"])
    bot = TradingBot(user_data=user_data, shared_ws=shared_ws, golden_param_store={})
    
    ws_task = asyncio.create_task(shared_ws.start())
    
    await bot.train_and_save_model()
    
    ws_task.cancel()

if __name__ == "__main__":
    if "--train" in sys.argv:
        try:
            asyncio.run(run_training())
        except KeyboardInterrupt:
            logger.info("Тренировка прервана пользователем.")
    else:
        try:
            asyncio.run(run_all())
        except KeyboardInterrupt:
            pass
        logger.info("Программа завершила работу.")