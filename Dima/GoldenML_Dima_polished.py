#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Бот для торговли на Bybit с использованием модели, дрейфа, супер-тренда и др.
Версия: интегрированный код с логированием исторических данных, усреднением позиций
         при достижении целевого уровня убытка и режимом quiet period.
================================================================================
"""

import asyncio
import hmac
import hashlib
import json
import logging
import os
import re
import math
import time
import random
import csv
import threading
import datetime
import pandas as pd
import numpy as np
import pandas_ta as ta
from decimal import Decimal
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.exceptions import TelegramRetryAfter, TelegramBadRequest, TelegramNetworkError

from dotenv import load_dotenv

import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scipy.stats import ks_2samp
from tabulate import tabulate

from pybit.unified_trading import HTTP, WebSocket
from pybit.exceptions import InvalidRequestError

from rich.console import Console
from rich.table import Table
from rich import box

import certifi
import ssl

import functools
from typing import Optional, Tuple

# Загружаем переменные окружения
load_dotenv("keys_Dima.env")  # ожидаются BYBIT_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID и т.д.

# ===================== Логгер и глобальные параметры =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[
        RotatingFileHandler("GoldenML_Dima.log", maxBytes=5 * 1024 * 1024, backupCount=2),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Торговые параметры
MAX_TOTAL_VOLUME = Decimal("500")  # общий лимит (USDT)
POSITION_VOLUME = Decimal("100")    # объём на сделку (USDT)
PROFIT_LEVEL = Decimal("0.008")     # 0.8% – порог закрытия
PROFIT_COEFFICIENT = Decimal("100") # коэффициент перевода в проценты

TAKE_PROFIT_ENABLED = False
TAKE_PROFIT_LEVEL = Decimal("0.005")  # 0.5% тейк‑профит

check_and_close_active = True
SUPER_TREND_TIMEFRAME = "1"  # например, "1", "15", "60"

TAKE_PROFIT_LEVEL_Buy = Decimal("1.025")
STOP_LOSS_LEVEL_Buy   = Decimal("0.95")
TAKE_PROFIT_LEVEL_Sell = Decimal("0.975")
STOP_LOSS_LEVEL_Sell   = Decimal("1.05")

TRAILING_STOP_ENABLED = True
TRAILING_GAP_PERCENT = Decimal("0.008")  # 0.8%
MIN_TRAILING_STOP = Decimal("0.0000001")

QUIET_PERIOD_ENABLED = False
MODEL_ONLY_SEMAPHORE = asyncio.Semaphore(5)
IS_SLEEPING_MODE = False  # Глобальный флаг

OPERATION_MODE = "ST_cross2"  # Режимы: drift_only, drift_top10, golden_setup, model_only, super_trend, ST_cross1, ST_cross2, ST_cross_global, ST_cross2_drift
HEDGE_MODE = True
INVERT_MODEL_LABELS = False

MODEL_FILENAME = "trading_model_final.pkl"
MIN_SAMPLES_FOR_TRAINING = 1000

# DRIFT параметры
VOLATILITY_THRESHOLD = 0.05
VOLUME_THRESHOLD = 2_000_000
TOP_N_PAIRS = 300

golden_params = {
    "Buy": {
        "period_iters": Decimal("4"),
        "price_change": Decimal("0.1"),
        "volume_change": Decimal("20000"),
        "oi_change": Decimal("20000"),
    },
    "Sell": {
        "period_iters": Decimal("4"),
        "price_change": Decimal("1.0"),
        "volume_change": Decimal("5000"),
        "oi_change": Decimal("5000"),
    },
}

# API ключи и Telegram
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("BYBIT_API_KEY / BYBIT_API_SECRET не заданы в .env!")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Настройки для HTTP-сессии (REST API)
custom_session = requests.Session()
retries = Retry(
    total=5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    backoff_factor=1
)
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retries)
custom_session.mount("http://", adapter)
custom_session.mount("https://", adapter)

session = HTTP(
    testnet=False,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
    timeout=60,
)

# Telegram и дополнительные объекты
telegram_bot = None
router = Router()
telegram_message_queue = None
send_semaphore = asyncio.Semaphore(10)
MAX_CONCURRENT_THREADS = 5
thread_semaphore = ThreadPoolExecutor(MAX_CONCURRENT_THREADS)

publish_drift_table = True
publish_model_table = True
iteration_counter = 3
MODEL_TOP_ENABLED = True

# Блокировки и глобальные состояния
state_lock = threading.Lock()
open_positions_lock = threading.Lock()
history_lock = threading.Lock()
current_symbol_index = 0

state = {}  # Например, state["total_open_volume"]
open_positions = {}  # Ключ – символ, значение – данные позиции
drift_history = defaultdict(list)
open_interest_history = defaultdict(list)
volume_history = defaultdict(list)

# SSL
ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())

current_model = None

selected_symbols = []
last_asset_selection_time = 0
ASSET_SELECTION_INTERVAL = 60 * 60

REAL_TRADES_FEATURES_CSV = "real_trades_features.csv"
MODEL_FEATURE_COLS = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]

INTERVAL = "1"

# ===================== Новые глобальные переменные для усредняющих позиций =====================
MAX_AVERAGING_VOLUME = MAX_TOTAL_VOLUME * Decimal("2")  # Общий лимит усредняющих позиций (в USDT)
averaging_total_volume = Decimal("0")  # Текущий суммарный объём усредняющих позиций
averaging_positions = {}  # Словарь для отслеживания усредняющих ордеров по символу

# Глобальная переменная для целевого уровня убытка для усреднения (например, 1% убытка)
TARGET_LOSS_FOR_AVERAGING = Decimal("16.0")

# Глобальная переменная для переключения режима мониторинга: "ws" – WebSocket, "http" – HTTP
MONITOR_MODE = "http"  # Можно изменить на "http", "ws" -- для WebSocket

# Добавить глобальную переменную
IS_RUNNING = True

# Глобальная переменная для WebSocket соединения
ws = None

### ПРАВКА (1) monitor_position(...) Закомментировать ###

# async def monitor_position():
#     """Monitor position changes using either WebSocket or HTTP"""
#     global ws
#     
#     if MONITOR_MODE == "ws":
#         # Инициализация WebSocket соединения с актуальным API
#         ws = WebSocket(
#             testnet=False,
#             channel_type="private",
#             api_key=BYBIT_API_KEY,
#             api_secret=BYBIT_API_SECRET,
#             ssl_context=ssl_context,
#         )
#         
#         # Подписываемся на обновления позиции
#         ws.position_stream(
#             callback=handle_position_update,  # см. ниже
#             symbol=SYMBOL
#         )
#         
#         # Держим соединение активным
#         while True:
#             await asyncio.sleep(1)
#             
#     else:
#         # Существующий HTTP мониторинг
#         while True:
#             try:
#                 position = await check_position()
#                 await asyncio.sleep(MONITOR_INTERVAL)
#             except Exception as e:
#                 logger.error(f"Error in HTTP position monitoring: {e}")
#                 await asyncio.sleep(1)

"""
### ПРАВКА (2) handle_position_update(...) – "Заменить" 
###   Мы используем handle_position_update вместо handle_message в WebSocket.
###   Переносим ЛОГИКУ из handle_message в handle_position_update, 
###   handle_message – закомментируем/удалим.
###

# Старая версия handle_position_update закомментируем:
"""
# def handle_position_update(message):
#     \"\"\"Callback function for position updates\"\"\"
#     try:
#         if \"data\" in message:
#             position_info = message[\"data\"][0]
#             size = float(position_info.get(\"size\", 0))
#             side = position_info.get(\"side\", \"\")
#             
#             # Обновляем глобальные переменные позиции
#             global POSITION_SIZE, POSITION_SIDE
#             POSITION_SIZE = size
#             POSITION_SIDE = side
#             
#             logger.info(f\"Position update - Size: {size}, Side: {side}\")
#             
#     except Exception as e:
#         logger.error(f\"Error processing position update: {e}\")
# """

# # Старая handle_message – закомментируем и вместо неё ЛОГИКУ перенесём в handle_position_update
# """
# def handle_message(message):
#     \"\"\"
#     Обработчик входящих сообщений от WebSocket.
#     Если по базовой позиции достигается целевой уровень убытка,
#     вызывается открытие усредняющей позиции.
#     \"\"\"
#     logger.info(f\"[WS] Получено сообщение: {message}\")
#     if \"data\" in message and isinstance(message[\"data\"], list):
#         for candle in message[\"data\"]:
#             symbol = candle.get(\"symbol\")
#             close_str = candle.get(\"close\")
#             if not symbol or not close_str:
#                 continue
#             try:
#                 current_price = Decimal(close_str)
#             except Exception as e:
#                 logger.error(f\"[WS] Ошибка преобразования цены для {symbol}: {e}\")
#                 continue
#
#             with open_positions_lock:
#                 if symbol not in open_positions:
#                     continue
#                 pos = open_positions[symbol]
#             side = pos[\"side\"]
#             entry_price = Decimal(str(pos[\"avg_price\"]))
#             if side.lower() == \"buy\":
#                 ratio = (current_price - entry_price) / entry_price
#             else:
#                 ratio = (entry_price - current_price) / entry_price
#             profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal(\"0.0001\"))
#             logger.info(f\"[WS] {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%\")
#
#             if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
#                 logger.info(
#                     f\"[WS] {symbol} достиг порога убытка ({profit_perc}% <= -{TARGET_LOSS_FOR_AVERAGING}). \"
#                     \"Открываю усредняющую позицию.\"
#                 )
#                 open_averaging_position(symbol)
#     else:
#         logger.debug(f\"[WS] Получено сообщение: {message}\")
# """

# Новая handle_position_update (совмещённая логика):
def handle_position_update(message):
    """
    Обработчик входящих сообщений от WebSocket.
    Для каждой полученной свечи проверяет:
      – Если PnL <= -TARGET_LOSS_FOR_AVERAGING, открывает усредняющую позицию.
      – Если с учётом плеча (leveraged ROI) PnL >= threshold (например, 5%),
        и трейлинг-стоп ещё не установлен, вызывает установку трейлинг-стопа.
    """
    logger.info(f"[WS] Получено сообщение: {message}")
    if "data" in message and isinstance(message["data"], list):
        for candle in message["data"]:
            symbol = candle.get("symbol")
            close_str = candle.get("close")
            if not symbol or not close_str:
                continue
            try:
                current_price = Decimal(close_str)
            except Exception as e:
                logger.error(f"[WS] Ошибка преобразования цены для {symbol}: {e}")
                continue

            with open_positions_lock:
                if symbol not in open_positions:
                    continue
                pos = open_positions[symbol]

            side = pos["side"]
            entry_price = Decimal(str(pos["avg_price"]))
            if side.lower() == "buy":
                ratio = (current_price - entry_price) / entry_price
            else:
                ratio = (entry_price - current_price) / entry_price

            profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
            logger.info(f"[WS] {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%")

            # Если убыток достигает порога для усреднения, открываем усредняющую позицию
            if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
                logger.info(
                    f"[WS] {symbol} достиг порога убытка ({profit_perc}% <= -{TARGET_LOSS_FOR_AVERAGING}). "
                    "Открываю усредняющую позицию."
                )
                open_averaging_position(symbol)

            # Дополнительно проверяем условие для установки трейлинг-стопа.
            default_leverage = Decimal("10")
            leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))
            threshold_trailing = Decimal("5.0")
            if leveraged_pnl_percent >= threshold_trailing:
                if not pos.get("trailing_stop_set", False):
                    logger.info(
                        f"[WS] {symbol}: Достигнут уровень для трейлинг-стопа "
                        f"(leveraged_pnl_percent = {leveraged_pnl_percent}%). Устанавливаю трейлинг-стоп."
                    )
                    set_trailing_stop(symbol, pos["size"], TRAILING_GAP_PERCENT, side)
    else:
        logger.debug(f"[WS] Получено сообщение: {message}")

# ===================== ФУНКЦИЯ TOGGLE QUIET PERIOD =====================
def is_silence_period() -> bool:
    if not QUIET_PERIOD_ENABLED:
        return False
    now_utc = datetime.datetime.utcnow()
    if now_utc.hour >= 22 or now_utc.hour < 1:
        return True
    return False

def toggle_quiet_period():
    """
    Переключает состояние quiet period.
    Возвращает строку "включён", если quiet period включён, или "выключен", если отключён.
    """
    global QUIET_PERIOD_ENABLED
    QUIET_PERIOD_ENABLED = not QUIET_PERIOD_ENABLED
    return "включён" if QUIET_PERIOD_ENABLED else "выключен"

def toggle_sleep_mode():
    """Переключает спящий режим."""
    global IS_SLEEPING_MODE
    IS_SLEEPING_MODE = not IS_SLEEPING_MODE
    return "включен" if IS_SLEEPING_MODE else "выключен"

### MODIFICATION 4: Добавлена функция is_sleeping() для проверки спящего режима.
def is_sleeping():
    return IS_SLEEPING_MODE


# ===================== ФУНКЦИИ ДЛЯ API BYBIT (REST) =====================
def get_symbol_info(symbol):
    try:
        resp = session.get_instruments_info(symbol=symbol, category="linear")
        if resp.get("retCode") != 0:
            logger.error(f"[get_symbol_info] {symbol}: {resp.get('retMsg')}")
            return None
        instruments = resp["result"].get("list", [])
        if not instruments:
            return None
        return instruments[0]
    except Exception as e:
        logger.exception(f"[get_symbol_info({symbol})]: {e}")
        return None

def get_historical_data_for_trading(symbol, interval="1", limit=200, from_time=None):
    try:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if from_time:
            params["from"] = from_time
        resp = session.get_kline(**params)
        if resp.get("retCode") != 0:
            logger.error(f"[TRADING_KLINE] {symbol}: {resp.get('retMsg')}")
            return pd.DataFrame()
        data = resp["result"].get("list", [])
        if not data:
            return pd.DataFrame()
        columns = ["open_time", "open", "high", "low", "close", "volume", "open_interest"]
        out = pd.DataFrame(data, columns=columns)
        out["startTime"] = pd.to_datetime(pd.to_numeric(out["open_time"], errors="coerce"), unit="ms", utc=True)
        out.rename(columns={
            "open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"
        }, inplace=True)
        out[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]] = \
            out[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]].apply(pd.to_numeric,
                                                                                                       errors="coerce")
        out.dropna(subset=["closePrice"], inplace=True)
        out.sort_values("startTime", inplace=True)
        out.reset_index(drop=True, inplace=True)
        logger.debug(
            f"[get_historical_data_for_trading] {symbol}: получено {len(out)} свечей. "
            f"Первая: {out.iloc[0].to_dict() if not out.empty else 'нет данных'}, "
            f"Последняя: {out.iloc[-1].to_dict() if not out.empty else 'нет данных'}."
        )
        return out[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]]
    except Exception as e:
        logger.exception(f"[get_historical_data_for_trading({symbol})]: {e}")
        return pd.DataFrame()

def get_usdt_pairs():
    try:
        tickers_resp = session.get_tickers(symbol=None, category="linear")
        if "result" not in tickers_resp or "list" not in tickers_resp["result"]:
            logger.error("[get_usdt_pairs] Некорректный ответ при get_tickers.")
            return []
        tickers_data = tickers_resp["result"]["list"]
        inst_resp = session.get_instruments_info(category="linear")
        if "result" not in inst_resp or "list" not in inst_resp["result"]:
            logger.error("[get_usdt_pairs] Некорректный ответ get_instruments_info.")
            return []
        instruments_data = inst_resp["result"]["list"]
        trading_status = {}
        for inst in instruments_data:
            sym = inst.get("symbol")
            stat = inst.get("status", "").upper()
            if sym:
                trading_status[sym] = (stat == "TRADING")
        usdt_pairs = []
        for tk in tickers_data:
            sym = tk.get("symbol")
            if not sym:
                continue
            if "USDT" in sym and "BTC" not in sym and "ETH" not in sym:
                if not trading_status.get(sym, False):
                    continue
                turnover24 = Decimal(str(tk.get("turnover24h", "0")))
                volume24 = Decimal(str(tk.get("volume24h", "0")))
                if turnover24 >= Decimal("2000000") and volume24 >= Decimal("2000000"):
                    usdt_pairs.append(sym)
        logger.info(f"[get_usdt_pairs] Отобраны USDT-пары: {usdt_pairs}")
        return usdt_pairs
    except Exception as e:
        logger.exception(f"[get_usdt_pairs] Ошибка: {e}")
        return []

def adjust_quantity(symbol: str, raw_qty: float) -> float:
    info = get_symbol_info(symbol)
    if not info:
        return 0.0
    lot_size = info.get("lotSizeFilter", {})
    min_qty = Decimal(str(lot_size.get("minOrderQty", "0")))
    qty_step = Decimal(str(lot_size.get("qtyStep", "1")))
    max_qty = Decimal(str(lot_size.get("maxOrderQty", "9999999")))
    last_price = get_last_close_price(symbol)
    if not last_price or last_price <= 0:
        return 0.0
    price_dec = Decimal(str(last_price))
    min_order_value = Decimal(str(info.get("minOrderValue", 5)))
    adj_qty = (Decimal(str(raw_qty)) // qty_step) * qty_step
    if adj_qty < min_qty:
        return 0.0
    if adj_qty > max_qty:
        adj_qty = max_qty
    order_value = adj_qty * price_dec
    if order_value < min_order_value:
        needed_qty = (min_order_value / price_dec).quantize(qty_step, rounding="ROUND_UP")
        if needed_qty > max_qty or needed_qty < min_qty:
            return 0.0
        adj_qty = needed_qty
    return float(adj_qty)

def get_exchange_positions():
    """
    Возвращает dict вида:
      {
        "SYMBOL": {
           "side": "Buy"/"Sell",
           "size": <float>,
           "avg_price": <float>,
           "position_volume": <float>,
           "symbol": "SYMBOL",
           "positionIdx": ...
        },
        ...
      }
    """
    try:
        resp = session.get_positions(category="linear", settleCoin="USDT")
        if resp.get("retCode") != 0:
            logger.error(f"[get_exchange_positions] {resp.get('retMsg')}")
            return {}
        positions = resp["result"].get("list", [])
        exchange_positions = {}
        for pos in positions:
            size = float(pos.get("size", 0))
            if size == 0:
                continue
            sym = pos.get("symbol")
            side = "Buy" if pos.get("side", "").lower() == "buy" else "Sell"
            entry_price = float(pos.get("avgPrice", 0))
            volume = size * entry_price
            exchange_positions[sym] = {
                "side": side,
                "size": size,
                "avg_price": entry_price,
                "position_volume": volume,
                "symbol": sym,
                "positionIdx": pos.get("positionIdx", None),
            }
        return exchange_positions
    except Exception as e:
        logger.exception(f"[get_exchange_positions] Ошибка: {e}")
        return {}

def get_last_close_price(symbol):
    try:
        resp = session.get_kline(category="linear", symbol=symbol, interval="1", limit=1)
        if not resp or resp.get("retCode") != 0:
            logger.error(f"[get_last_close_price] {symbol}: {resp.get('retMsg') if resp else 'Empty'}")
            return None
        klines = resp["result"].get("list", [])
        if not klines:
            return None
        row = klines[0]
        if isinstance(row, list) and len(row) > 4:
            return float(row[4])
        elif isinstance(row, dict):
            return float(row.get("close"))
        else:
            return None
    except Exception as e:
        logger.exception(f"[get_last_close_price({symbol})]: {e}")
        return None

def get_selected_symbols():
    global selected_symbols, last_asset_selection_time
    now = time.time()
    if now - last_asset_selection_time >= ASSET_SELECTION_INTERVAL or not selected_symbols:
        selected_symbols = get_usdt_pairs()
        last_asset_selection_time = now
        logger.info(f"Обновлён список активов: {selected_symbols}")
    return selected_symbols

def get_position_info(symbol, side):
    try:
        resp = session.get_positions(category="linear", symbol=symbol)
        if resp.get("retCode") != 0:
            logger.error(f"[get_position_info] {symbol}: {resp.get('retMsg')}")
            return None
        positions = resp["result"].get("list", [])
        for p in positions:
            if p.get("side", "").lower() == side.lower():
                return p
        return None
    except Exception as e:
        logger.exception(f"[get_position_info] {symbol},{side}: {e}")
        return None

### ПРАВКА (3) set_sl_and_tp_from_globals(...) – Закомментировать ###
"""
def set_sl_and_tp_from_globals(symbol: str, side: str, entry_price: float, size: float):
    try:
        pos_info = get_position_info(symbol, side)
        if not pos_info:
            logger.error(f\"[set_sl_and_tp_from_globals] Нет позиции {symbol}/{side}\")
            return
        pos_idx = pos_info.get(\"positionIdx\")
        if not pos_idx:
            logger.error(f\"[set_sl_and_tp_from_globals] positionIdx не найден для {symbol}/{side}\")
            return
        ep = Decimal(str(entry_price))
        if side.lower() == \"buy\":
            stop_loss_price = ep * STOP_LOSS_LEVEL_Buy
            take_profit_price = ep * TAKE_PROFIT_LEVEL_Buy
        else:
            stop_loss_price = ep * STOP_LOSS_LEVEL_Sell
            take_profit_price = ep * TAKE_PROFIT_LEVEL_Sell
        resp = session.set_trading_stop(
            category=\"linear\",
            symbol=symbol,
            side=side,
            positionIdx=pos_idx,
            qty=str(size),
            stopLoss=str(stop_loss_price),
            takeProfit=str(take_profit_price),
            timeInForce=\"GoodTillCancel\"
        )
        if resp and resp.get(\"retCode\") == 0:
            logger.info(f\"[set_sl_and_tp_from_globals] SL={stop_loss_price}, TP={take_profit_price} для {symbol}\")
        else:
            logger.error(f\"[set_sl_and_tp_from_globals] Ошибка set_trading_stop: {resp.get('retMsg')}\")
    except Exception as e:
        logger.exception(f\"[set_sl_and_tp_from_globals] Ошибка: {e}\")
"""

def place_order(symbol, side, qty,
                order_type="Market",
                time_in_force="GoodTillCancel",
                reduce_only=False,
                positionIdx=None):
    try:
        adj_qty = adjust_quantity(symbol, qty)
        if adj_qty <= 0:
            logger.error(f"[place_order] qty={qty} => adj_qty={adj_qty}, недопустимо.")
            return None
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(adj_qty),
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
        }
        if HEDGE_MODE:
            if positionIdx is None:
                positionIdx = 1 if side.lower() == "buy" else 2
            params["positionIdx"] = positionIdx
        resp = session.place_order(**params)
        if resp.get("retCode") == 0:
            logger.info(f"[place_order] OK {symbol}, side={side}, qty={adj_qty}")
            return resp
        else:
            logger.error(f"[place_order] Ошибка: {resp.get('retMsg')} (retCode={resp.get('retCode')})")
            return None
    except InvalidRequestError as e:
        logger.exception(f"[place_order] InvalidRequestError({symbol}): {e}")
        return None
    except Exception as e:
        logger.exception(f"[place_order] Ошибка({symbol}): {e}")
        return None


# ===================== ФУНКЦИИ ЛОГИРОВАНИЯ СДЕЛОК =====================
def log_model_features_for_trade(trade_id: str, symbol: str, side: str, features: dict):
    csv_filename = REAL_TRADES_FEATURES_CSV
    file_exists = os.path.isfile(csv_filename)
    row_to_write = {"trade_id": trade_id, "symbol": symbol, "side": side}
    for k, v in features.items():
        row_to_write[k] = v
    try:
        with open(csv_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row_to_write.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_to_write)
    except Exception as e:
        logger.exception(f"[log_model_features_for_trade] Ошибка записи в {csv_filename}: {e}")

def update_trade_outcome(trade_id: str, pnl: float):
    csv_filename = REAL_TRADES_FEATURES_CSV
    if not os.path.isfile(csv_filename):
        return
    try:
        df = pd.read_csv(csv_filename)
        mask = (df["trade_id"] == trade_id)
        if not mask.any():
            return
        df.loc[mask, "pnl"] = pnl
        df.loc[mask, "label"] = 1 if pnl > 0 else 0
        df.to_csv(csv_filename, index=False)
        logger.info(f"[update_trade_outcome] Запись {trade_id} обновлена: pnl={pnl}")
    except Exception as e:
        logger.exception(f"[update_trade_outcome] Ошибка обновления: {e}")

def log_trade(symbol, row, open_interest, action, result, closed_manually=False):
    try:
        filename = "trade_log.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["symbol", "timestamp", "openPrice", "highPrice", "lowPrice", "closePrice", "volume",
                                 "open_interest", "action", "result", "closed_manually"])
            if row is not None and isinstance(row, pd.Series):
                time_str = row["startTime"].strftime("%Y-%m-%d %H:%M:%S") if "startTime" in row else "N/A"
                open_str = row.get("openPrice", "N/A")
                high_str = row.get("highPrice", "N/A")
                low_str = row.get("lowPrice", "N/A")
                close_str = row.get("closePrice", "N/A")
                vol_str = row.get("volume", "N/A")
            else:
                time_str = open_str = high_str = low_str = close_str = vol_str = "N/A"
            oi_str = open_interest if open_interest is not None else "N/A"
            writer.writerow([symbol, time_str, open_str, high_str, low_str, close_str, vol_str, oi_str, action, result,
                             closed_manually])
        logger.info(f"Сделка зафиксирована: {symbol}, {action}, {result}, closed_manually={closed_manually}")
        if telegram_bot and telegram_message_queue:
            local_tz = datetime.timezone(datetime.timedelta(hours=3))
            row_time = None
            if row is not None and isinstance(row, pd.Series):
                row_time = row.get("startTime", None)
            if isinstance(row_time, pd.Timestamp):
                if not row_time.tzinfo:
                    row_time = row_time.tz_localize("UTC")
                row_time = row_time.astimezone(local_tz)
            escaped_symbol = escape_markdown(symbol)
            symbol_url = f"https://www.bybit.com/trade/usdt/{symbol}"
            symbol_link = f"[{escaped_symbol}]({symbol_url})"
            formatted_time = row_time.strftime("%Y-%m-%d %H:%M:%S") if row_time else "N/A"
            if result == "Opened":
                if action.lower() == "buy":
                    message = (f"🟩 *Открытие ЛОНГ‑позиции*\n"
                               f"*Символ:* {symbol_link}\n"
                               f"*Время:* {escape_markdown(formatted_time)}\n"
                               f"*Цена открытия:* {escape_markdown(str(row.get('openPrice', 'N/A')))}\n"
                               f"*Объём:* {escape_markdown(str(row.get('volume', 'N/A')))} coins\n"
                               f"*Тип открытия:* {escape_markdown(OPERATION_MODE)}")
                else:
                    message = (f"🟥 *Открытие ШОРТ‑позиции*\n"
                               f"*Символ:* {symbol_link}\n"
                               f"*Время:* {escape_markdown(formatted_time)}\n"
                               f"*Цена открытия:* {escape_markdown(str(row.get('openPrice', 'N/A')))}\n"
                               f"*Объём:* {escape_markdown(str(row.get('volume', 'N/A')))} coins\n"
                               f"*Тип открытия:* {escape_markdown(OPERATION_MODE)}")
            elif result == "Closed":
                message = (f"❌ *Закрытие позиции*\n"
                           f"*Символ:* {symbol_link}\n"
                           f"*Время:* {escape_markdown(formatted_time)}\n"
                           f"*Цена закрытия:* {escape_markdown(str(row.get('closePrice', 'N/A')))}\n"
                           f"*Объём:* {escape_markdown(str(row.get('volume', 'N/A')))} coins\n"
                           f"*Тип закрытия:* {'Вручную' if closed_manually else escape_markdown(OPERATION_MODE)}")
            elif result == "Trailing Stop Set":
                message = (f"🔄 *Трейлинг стоп‑лосс*\n"
                           f"*Символ:* {symbol_link}\n"
                           f"*Время:* {escape_markdown(formatted_time)}\n"
                           f"*Расстояние:* {escape_markdown(str(action))}\n"  # Здесь action передаётся как trailing_distance
                           f"*Дополнительно:* {escape_markdown(str(result))}")
            else:
                message = (f"🫡🔄 *Сделка*\n"
                           f"*Символ:* {symbol_link}\n"
                           f"*Результат:* {escape_markdown(result)}\n"
                           f"*Цена открытия:* {escape_markdown(str(row.get('closePrice', 'N/A')))}\n"
                           f"*Действие:* {escape_markdown(action)}\n")
            asyncio.run_coroutine_threadsafe(telegram_message_queue.put(message), loop)
    except Exception as e:
        logger.warning("Не удалось отправить сообщение в Телеграм (bot/queue не инициализирован).")
        logger.exception(e)


# ===================== DRIFT ЛОГИКА =====================
def monitor_feature_drift_per_symbol(symbol, new_data, ref_data, feature_cols, drift_csv="feature_drift.csv", threshold=0.5):
    """Мониторинг дрейфа признаков для символа."""
    try:
        if new_data.empty:
            logger.info(f"[DRIFT] {symbol}: new_data пуст")
            return False, 0.0, "нет данных"
            
        # Если ref_data пуст, используем первую половину new_data как референс
        if ref_data.empty:
            split_point = len(new_data) // 2
            ref_data = new_data.iloc[:split_point].copy()
            new_data = new_data.iloc[split_point:].copy()
            
        if new_data.empty or ref_data.empty:
            return False, 0.0, "недостаточно данных"

        # Расчет средних значений
        mean_new = new_data[feature_cols].mean().mean()
        mean_ref = ref_data[feature_cols].mean().mean()
        direction = "вверх" if mean_new > mean_ref else "вниз"
        
        # Расчет статистик KS
        stats = []
        for c in feature_cols:
            if c not in new_data.columns or c not in ref_data.columns:
                continue
            stat, _ = ks_2samp(new_data[c].values, ref_data[c].values)
            stats.append(stat)
            
        if not stats:
            return False, 0.0, "нет фич"
            
        anomaly_strength = float(np.mean(stats))
        is_anomaly = anomaly_strength > threshold
        
        # Сохраняем в историю
        ts_str = datetime.datetime.utcnow().isoformat()
        with history_lock:
            drift_history[symbol].append((ts_str, anomaly_strength, direction))
            if len(drift_history[symbol]) > 10:
                drift_history[symbol].pop(0)
                
        # Логируем результат
        logger.info(f"[DRIFT] {symbol}: strength={anomaly_strength:.3f}, direction={direction}, is_anomaly={is_anomaly}")
        
        return is_anomaly, anomaly_strength, direction
        
    except Exception as e:
        logger.exception(f"[DRIFT] Ошибка в monitor_feature_drift_per_symbol для {symbol}: {e}")
        return False, 0.0, "ошибка"

### ПРАВКА (4) get_top_anomalies_in_last_n(...) – Удалить
# Функцию удаляем полностью.

### ПРАВКА (5) send_drift_top_to_telegram(...) – Удалить
# Функцию удаляем полностью.

def handle_drift_top10(top_list):
    """Обработка сигналов дрифта только для режимов drift"""
    if OPERATION_MODE not in ["drift_only", "drift_top10"]:
        logger.info(f"[DRIFT_TOP10] Пропуск: текущий режим {OPERATION_MODE} не поддерживает торговлю по дрифту")
        return
        
    logger.info(f"[DRIFT_TOP10] Обработка сигналов дрифта в режиме {OPERATION_MODE}")
    for (sym, strength, direction) in top_list:
        side = "Buy" if direction == "вверх" else "Sell"
        logger.info(f"[DRIFT_TOP10] {sym}: side={side}, strength={strength:.2f}")
        handle_drift_order_with_trailing(sym, side)

def handle_drift_order_with_trailing(symbol, side):
    try:
        with open_positions_lock:
            if symbol in open_positions:
                logger.info(f"[DRIFT-TRADE] {symbol} уже открыта.")
                return
        price = get_last_close_price(symbol)
        if price is None:
            logger.error(f"[DRIFT_TOP10] Нет цены для {symbol}.")
            return
        dec_price = Decimal(str(price))
        vol = POSITION_VOLUME
        qty = vol / dec_price
        order_result = place_order(symbol=symbol, side=side, qty=float(qty), order_type="Market",
                                   positionIdx=1 if side.lower() == "buy" else 2)
        if order_result and order_result.get("retCode") == 0:
            with open_positions_lock, state_lock:
                open_positions[symbol] = {
                    "side": side.capitalize(),
                    "size": float(qty),
                    "avg_price": float(dec_price),
                    "position_volume": float(qty) * float(dec_price),
                    "symbol": symbol,
                    "trailing_stop_set": False,
                    "trade_id": f"{symbol}_{int(time.time())}"
                }
                used = state.get("total_open_volume", Decimal("0"))
                state["total_open_volume"] = used + vol
            row = get_last_row(symbol)
            log_trade(symbol, row, None, side, "Opened (DRIFT)", closed_manually=False)
            logger.info(f"[DRIFT_TOP10] Позиция {symbol} ({side}) открыта.")
            if TRAILING_STOP_ENABLED:
                set_trailing_stop(symbol, float(qty), TRAILING_GAP_PERCENT, side)
        else:
            logger.error(f"[DRIFT_TOP10] Ошибка place_order => {order_result}")
    except Exception as e:
        logger.exception(f"[DRIFT_TOP10] handle_drift_order_with_trailing({symbol}): {e}")


# ===================== GOLDEN SETUP =====================
def handle_golden_setup(symbol, df):
    try:
        current_oi = Decimal(str(df.iloc[-1]["open_interest"]))
        current_vol = Decimal(str(df.iloc[-1]["volume"]))
        current_price = Decimal(str(df.iloc[-1]["closePrice"]))
        with history_lock:
            open_interest_history[symbol].append(current_oi)
            volume_history[symbol].append(current_vol)
            sp_iters = int(golden_params["Sell"]["period_iters"])
            lp_iters = int(golden_params["Buy"]["period_iters"])
            period = max(sp_iters, lp_iters)
            if (len(open_interest_history[symbol]) < period or len(volume_history[symbol]) < period):
                logger.info(f"{symbol}: Недостаточно истории (golden_setup).")
                return None, None
            if df.shape[0] < period:
                logger.info(f"{symbol}: df.shape[0] < {period} => пропуск golden_setup.")
                return None, None
            oi_prev = open_interest_history[symbol][-period]
            vol_prev = volume_history[symbol][-period]
            price_prev = Decimal(str(df.iloc[-period]["closePrice"]))
            if price_prev == 0:
                return None, None
            price_change = ((current_price - price_prev) / price_prev) * 100
            volume_change = ((current_vol - vol_prev) / vol_prev) * 100 if vol_prev != 0 else Decimal("0")
            oi_change = ((current_oi - oi_prev) / oi_prev) * 100 if oi_prev != 0 else Decimal("0")
            logger.info(
                f"[GOLDEN_SETUP] {symbol}: p_ch={price_change:.2f}, vol_ch={volume_change:.2f}, oi_ch={oi_change:.2f}")
            action = None
            if (price_change <= -golden_params["Sell"]["price_change"] 
                and volume_change >= golden_params["Sell"]["volume_change"] 
                and oi_change >= golden_params["Sell"]["oi_change"]):
                action = "Sell"
            elif (price_change >= golden_params["Buy"]["price_change"] 
                  and volume_change >= golden_params["Buy"]["volume_change"] 
                  and oi_change >= golden_params["Buy"]["oi_change"]):
                action = "Buy"
            else:
                return None, None
        return (action, float(price_change))
    except Exception as e:
        logger.exception(f"Ошибка handle_golden_setup({symbol}): {e}")
        return None, None


# ===================== ФУНКЦИИ МОДЕЛИ =====================
def collect_historical_data(symbols, interval="1", limit=200):
    dfs = []
    for sym in symbols:
        df = get_historical_data_for_model(sym, interval, limit)
        df = prepare_features_for_model(df)
        if df.empty:
            continue
        df = make_multiclass_target_for_model(df, horizon=1, threshold=Decimal("0.0025"))
        if df.empty:
            continue
        df["symbol"] = sym
        dfs.append(df)
    if dfs:
        data = pd.concat(dfs, ignore_index=True)
        data.to_csv("historical_data_for_model_5m.csv", index=False)
        print("historical_data_for_model_5m.csv сохранён (с target).")
    else:
        print("Нет данных для сохранения.")

def get_historical_data_for_model(symbol, interval="1", limit=200, from_time=None):
    try:
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        if from_time:
            params["from"] = from_time
        resp = session.get_kline(**params)
        if resp.get("retCode") != 0:
            logger.error(f"[MODEL] get_kline({symbol}): {resp.get('retMsg')}")
            return pd.DataFrame()
        data = resp["result"].get("list", [])
        if not data:
            return pd.DataFrame()
        out_rows = []
        for row in data:
            if len(row) < 5:
                continue
            out_rows.append([row[0], row[1], row[2], row[3], row[4]])
        columns = ["open_time", "open", "high", "low", "close"]
        df = pd.DataFrame(out_rows, columns=columns)
        df["startTime"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
        df.rename(columns={"open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"},
                  inplace=True)
        df = df[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice"]]
        df.dropna(subset=["closePrice"], inplace=True)
        logger.debug(f"[get_historical_data_for_model] {symbol}: получено {len(df)} свечей.")
        return df
    except Exception as e:
        logger.exception(f"Ошибка get_historical_data_for_model({symbol}): {e}")
        return pd.DataFrame()

def prepare_features_for_model(df):
    try:
        for c in ["openPrice", "highPrice", "lowPrice", "closePrice"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["closePrice"], inplace=True)
        if df.empty:
            return df
        df["ohlc4"] = (df["openPrice"] + df["highPrice"] + df["lowPrice"] + df["closePrice"]) / 4
        macd_df = calculate_macd(df["ohlc4"])
        df["macd"] = macd_df["MACD_12_26_9"]
        df["macd_signal"] = macd_df["MACDs_12_26_9"]
        df["rsi_13"] = calculate_rsi(df["ohlc4"], periods=13)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["macd", "macd_signal", "rsi_13"], inplace=True)
        return df
    except Exception as e:
        logger.exception(f"Ошибка prepare_features_for_model: {e}")
        return pd.DataFrame()

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """
    Рассчитывает MACD (Moving Average Convergence Divergence)
    """
    exp1 = close_prices.ewm(span=fast, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({
        'MACD_12_26_9': macd,
        'MACDs_12_26_9': signal_line
    })

def calculate_rsi(close_prices, periods=13):
    """
    Рассчитывает RSI (Relative Strength Index)
    """
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def make_multiclass_target_for_model(df, horizon=1, threshold=Decimal("0.0025")):
    try:
        df = df.sort_values("startTime").reset_index(drop=True)
        future_close = df["closePrice"].shift(-horizon)
        current_close = df["closePrice"]
        df["price_change"] = (future_close - current_close) / current_close
        df.loc[df["price_change"] > float(threshold), "target"] = 2
        df.loc[df["price_change"] < -float(threshold), "target"] = 0
        df["target"] = df["target"].fillna(1)
        df.dropna(subset=["price_change"], inplace=True)
        return df
    except Exception as e:
        logger.exception(f"Ошибка make_multiclass_target_for_model: {e}")
        return df

def process_symbol_model_only_sync(symbol):
    global current_model
    if not current_model:
        current_model = load_model()
        if not current_model:
            return
    df_5m = get_historical_data_for_model(symbol, "5", limit=200)
    df_5m = prepare_features_for_model(df_5m)
    if df_5m.empty:
        return
    row = df_5m.iloc[[-1]]
    feat_cols = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
    X = row[feat_cols].values
    try:
        pred = current_model.predict(X)
        proba = current_model.predict_proba(X)
    except Exception as e:
        logger.exception(f"[MODEL_ONLY] Ошибка при прогнозировании для {symbol}: {e}")
        return
    log_model_prediction(symbol, pred[0], proba)
    if pred[0] == 2:
        open_position(symbol, "Buy", POSITION_VOLUME, reason="Model")
    elif pred[0] == 0:
        open_position(symbol, "Sell", POSITION_VOLUME, reason="Model")
    else:
        logger.info(f"[MODEL_ONLY] {symbol}: HOLD => пропуск.")

async def process_symbol_model_only(symbol):
    async with MODEL_ONLY_SEMAPHORE:
        await asyncio.to_thread(process_symbol_model_only_sync, symbol)

def log_model_prediction_for_symbol(symbol):
    global current_model
    if not current_model:
        current_model = load_model()
        if not current_model:
            logger.error("Модель не загружена!")
            return
    df = get_historical_data_for_model(symbol, interval="1", limit=200)
    df = prepare_features_for_model(df)
    if df.empty:
        logger.info(f"{symbol}: недостаточно данных для предсказания.")
        return
    row = df.iloc[[-1]]
    feat_cols = MODEL_FEATURE_COLS
    X = row[feat_cols].values
    try:
        pred = current_model.predict(X)
        proba = current_model.predict_proba(X)
    except Exception as e:
        logger.exception(f"[MODEL] Ошибка при прогнозировании для {symbol}: {e}")
        return
    log_model_prediction(symbol, pred[0], proba)

def log_model_prediction(symbol, prediction, prediction_proba):
    try:
        fname = "model_predictions_log.csv"
        file_exists = os.path.isfile(fname)
        with open(fname, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "symbol", "prediction", "prob_buy", "prob_hold", "prob_sell"])
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            p_sell = prediction_proba[0][0]
            p_hold = prediction_proba[0][1]
            p_buy = prediction_proba[0][2]
            writer.writerow([ts, symbol, prediction, p_buy, p_hold, p_sell])
        logger.info(f"Модельное предсказание для {symbol} записано.")
    except Exception as e:
        logger.exception(f"Ошибка log_model_prediction({symbol}): {e}")

def train_and_load_model(csv_path="historical_data_for_model_5m.csv"):
    try:
        if not os.path.isfile(csv_path):
            logger.warning(f"Нет файла {csv_path} => обучение невозможно.")
            return None
        df_all = pd.read_csv(csv_path)
        if df_all.empty:
            logger.warning(f"{csv_path} пуст => обучение невозможно.")
            return None
        if "startTime" in df_all.columns and not pd.api.types.is_datetime64_any_dtype(df_all["startTime"]):
            df_all["startTime"] = pd.to_datetime(df_all["startTime"], utc=True, errors="coerce")
        df_all.drop_duplicates(["symbol", "startTime"], inplace=True)
        df_all.dropna(subset=["closePrice"], inplace=True)
        dfs = []
        for sym in df_all["symbol"].unique():
            df_sym = df_all[df_all["symbol"] == sym].copy()
            df_sym.sort_values("startTime", inplace=True)
            df_sym = prepare_features_for_model(df_sym)
            if df_sym.empty:
                continue
            df_sym = make_multiclass_target_for_model(df_sym, horizon=1, threshold=Decimal("0.0025"))
            if df_sym.empty:
                continue
            dfs.append(df_sym)
        if not dfs:
            logger.warning("Нет данных для обучения модели.")
            return None
        data = pd.concat(dfs, ignore_index=True)
        data.dropna(subset=["target"], inplace=True)
        if data.empty:
            logger.warning("Нет данных (после target).")
            return None
        if len(data) < MIN_SAMPLES_FOR_TRAINING:
            logger.warning(f"Слишком мало строк: {len(data)} < {MIN_SAMPLES_FOR_TRAINING}.")
            return None
        feature_cols = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
        for c in feature_cols:
            if c not in data.columns:
                logger.warning(f"Колонка {c} не найдена => пропуск.")
                return None
        data = data.dropna(subset=feature_cols)
        if data.empty:
            logger.warning("Все NaN => нет данных.")
            return None
        X = data[feature_cols].values
        y = data["target"].astype(int).values
        if len(X) < 50:
            logger.warning(f"[retrain_model_with_real_trades] Слишком мало данных для обучения (всего {len(X)}).")
            return None
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")),
        ])
        tscv = TimeSeriesSplit(n_splits=3)
        best_acc = 0.0
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            if acc > best_acc:
                best_acc = acc
        pipeline.fit(X, y)
        logger.info(f"[retrain_model_with_real_trades] Обучение завершено, CV max_accuracy={best_acc:.4f}")
        y_pred_full = pipeline.predict(X)
        a_ = accuracy_score(y, y_pred_full)
        p_ = precision_score(y, y_pred_full, average="weighted", zero_division=0)
        r_ = recall_score(y, y_pred_full, average="weighted", zero_division=0)
        f1_ = f1_score(y, y_pred_full, average="weighted", zero_division=0)
        logger.info(
            f"[retrain_model_with_real_trades] Final train metrics: acc={a_:.4f}, prec={p_:.4f}, rec={r_:.4f}, f1={f1_:.4f}")
        joblib.dump(pipeline, MODEL_FILENAME)
        logger.info(f"[retrain_model_with_real_trades] Модель сохранена в {MODEL_FILENAME}")
        return pipeline
    except Exception as e:
        logger.exception(f"Ошибка train_and_load_model: {e}")
        return None

def load_model():
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except (ModuleNotFoundError, ImportError):
        logger.warning("Не удалось загрузить модель. Будет создана новая.")
        return train_and_load_model()

def retrain_model_with_real_trades(historical_csv="historical_data_for_model_5m.csv",
                                   real_trades_csv=REAL_TRADES_FEATURES_CSV):
    try:
        if not os.path.isfile(historical_csv):
            logger.warning(f"[retrain_model_with_real_trades] Файл {historical_csv} не найден.")
            return None
        df_hist = pd.read_csv(historical_csv)
        if df_hist.empty:
            logger.warning(f"[retrain_model_with_real_trades] {historical_csv} пуст.")
            return None
        if "startTime" in df_hist.columns and not pd.api.types.is_datetime64_any_dtype(df_hist["startTime"]):
            df_hist["startTime"] = pd.to_datetime(df_hist["startTime"], utc=True, errors="coerce")
        df_hist.drop_duplicates(["symbol", "startTime"], inplace=True)
        df_hist.dropna(subset=["closePrice"], inplace=True)
        dfs = []
        for sym in df_hist["symbol"].unique():
            df_sym = df_hist[df_hist["symbol"] == sym].copy()
            df_sym.sort_values("startTime", inplace=True)
            df_sym = prepare_features_for_model(df_sym)
            if df_sym.empty:
                continue
            df_sym = make_multiclass_target_for_model(df_sym, horizon=1, threshold=Decimal("0.0025"))
            if df_sym.empty:
                continue
            dfs.append(df_sym)
        if not dfs:
            logger.warning("Нет данных для обучения модели.")
            return None
        data = pd.concat(dfs, ignore_index=True)
        data.dropna(subset=["target"], inplace=True)
        if data.empty:
            logger.warning("Нет данных (после target).")
            return None
        if len(data) < MIN_SAMPLES_FOR_TRAINING:
            logger.warning(f"Слишком мало строк: {len(data)} < {MIN_SAMPLES_FOR_TRAINING}.")
            return None
        feature_cols = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
        for c in feature_cols:
            if c not in data.columns:
                logger.warning(f"Колонка {c} не найдена => пропуск.")
                return None
        data = data.dropna(subset=feature_cols)
        if data.empty:
            logger.warning("Все NaN => нет данных.")
            return None
        X = data[feature_cols].values
        y = data["target"].astype(int).values
        if len(X) < 50:
            logger.warning(f"[retrain_model_with_real_trades] Слишком мало данных для обучения (всего {len(X)}).")
            return None
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")),
        ])
        tscv = TimeSeriesSplit(n_splits=3)
        best_acc = 0.0
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            if acc > best_acc:
                best_acc = acc
        pipeline.fit(X, y)
        logger.info(f"[retrain_model_with_real_trades] Обучение завершено, CV max_accuracy={best_acc:.4f}")
        y_pred_full = pipeline.predict(X)
        a_ = accuracy_score(y, y_pred_full)
        p_ = precision_score(y, y_pred_full, average="weighted", zero_division=0)
        r_ = recall_score(y, y_pred_full, average="weighted", zero_division=0)
        f1_ = f1_score(y, y_pred_full, average="weighted", zero_division=0)
        logger.info(
            f"[retrain_model_with_real_trades] Final train metrics: acc={a_:.4f}, prec={p_:.4f}, rec={r_:.4f}, f1={f1_:.4f}")
        joblib.dump(pipeline, MODEL_FILENAME)
        logger.info(f"[retrain_model_with_real_trades] Модель сохранена в {MODEL_FILENAME}")
        return pipeline
    except Exception as e:
        logger.exception("[retrain_model_with_real_trades] Ошибка:")
        return None

async def maybe_retrain_model():
    global current_model
    new_model = retrain_model_with_real_trades(historical_csv="historical_data_for_model_5m.csv",
                                               real_trades_csv=REAL_TRADES_FEATURES_CSV)
    if new_model:
        current_model = new_model
        logger.info("[maybe_retrain_model] Модель успешно обновлена.")


# ===================== SUPER TREND ФУНКЦИИ =====================
def calculate_supertrend_bybit_34_2(df: pd.DataFrame, length=8, multiplier=3.0) -> pd.DataFrame:
    """
    Локальный расчёт SuperTrend (ATR-based), length=8, multiplier=3 (пример).
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        def extend_value(current_value, previous_value):
            if pd.isna(current_value) or current_value == 0:
                return previous_value
            else:
                return current_value

        for col in ["highPrice", "lowPrice", "closePrice"]:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            numeric_series = numeric_series.replace(0, np.nan)
            numeric_series = numeric_series.fillna(method='ffill')
            df[col] = numeric_series

        df.fillna(method='bfill', inplace=True)

        df["prev_close"] = df["closePrice"].shift(1)
        df["tr1"] = df["highPrice"] - df["lowPrice"]
        df["tr2"] = (df["highPrice"] - df["prev_close"]).abs()
        df["tr3"] = (df["lowPrice"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["true_range"].rolling(window=length, min_periods=1).mean()
        
        hl2 = (df["highPrice"] + df["lowPrice"]) / 2
        df["basic_ub"] = hl2 + multiplier * df["atr"]
        df["basic_lb"] = hl2 - multiplier * df["atr"]
        df["final_ub"] = df["basic_ub"].copy()
        df["final_lb"] = df["basic_lb"].copy()
        
        for i in range(1, len(df)):
            if (df.loc[df.index[i], "basic_ub"] < df.loc[df.index[i-1], "final_ub"]) \
               or (df.loc[df.index[i-1], "closePrice"] > df.loc[df.index[i-1], "final_ub"]):
                df.loc[df.index[i], "final_ub"] = df.loc[df.index[i], "basic_ub"]
            else:
                df.loc[df.index[i], "final_ub"] = df.loc[df.index[i-1], "final_ub"]
            
            if (df.loc[df.index[i], "basic_lb"] > df.loc[df.index[i-1], "final_lb"]) \
               or (df.loc[df.index[i-1], "closePrice"] < df.loc[df.index[i-1], "final_lb"]):
                df.loc[df.index[i], "final_lb"] = df.loc[df.index[i], "basic_lb"]
            else:
                df.loc[df.index[i], "final_lb"] = df.loc[df.index[i-1], "final_lb"]

            df.loc[df.index[i], "final_ub"] = extend_value(
                df.loc[df.index[i], "final_ub"],
                df.loc[df.index[i-1], "final_ub"]
            )
            df.loc[df.index[i], "final_lb"] = extend_value(
                df.loc[df.index[i], "final_lb"],
                df.loc[df.index[i-1], "final_lb"]
            )
        
        df["supertrend"] = df["final_ub"].copy()
        df.loc[df["closePrice"] > df["final_ub"], "supertrend"] = df["final_lb"]
        
        for i in range(1, len(df)):
            df.loc[df.index[i], "supertrend"] = extend_value(
                df.loc[df.index[i], "supertrend"],
                df.loc[df.index[i-1], "supertrend"]
            )

        return df

    except Exception as e:
        logger.exception(f"Ошибка в calculate_supertrend_bybit_34_2: {e}")
        return pd.DataFrame()


def calculate_supertrend_bybit_8_1(df: pd.DataFrame, length=3, multiplier=1.0) -> pd.DataFrame:
    """
    Локальный расчёт SuperTrend (ATR-based), length=3, multiplier=1 (пример).
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        def extend_value(current_value, previous_value):
            if pd.isna(current_value) or current_value == 0:
                return previous_value
            else:
                return current_value

        for col in ["highPrice", "lowPrice", "closePrice"]:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            numeric_series = numeric_series.replace(0, np.nan)
            numeric_series = numeric_series.fillna(method='ffill')
            df[col] = numeric_series

        df.fillna(method='bfill', inplace=True)

        df["prev_close"] = df["closePrice"].shift(1)
        df["tr1"] = df["highPrice"] - df["lowPrice"]
        df["tr2"] = (df["highPrice"] - df["prev_close"]).abs()
        df["tr3"] = (df["lowPrice"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["true_range"].rolling(window=length, min_periods=1).mean()
        
        hl2 = (df["highPrice"] + df["lowPrice"]) / 2
        df["basic_ub"] = hl2 + multiplier * df["atr"]
        df["basic_lb"] = hl2 - multiplier * df["atr"]
        df["final_ub"] = df["basic_ub"].copy()
        df["final_lb"] = df["basic_lb"].copy()
        
        for i in range(1, len(df)):
            if (df.loc[df.index[i], "basic_ub"] < df.loc[df.index[i-1], "final_ub"]) \
               or (df.loc[df.index[i-1], "closePrice"] > df.loc[df.index[i-1], "final_ub"]):
                df.loc[df.index[i], "final_ub"] = df.loc[df.index[i], "basic_ub"]
            else:
                df.loc[df.index[i], "final_ub"] = df.loc[df.index[i-1], "final_ub"]

            if (df.loc[df.index[i], "basic_lb"] > df.loc[df.index[i-1], "final_lb"]) \
               or (df.loc[df.index[i-1], "closePrice"] < df.loc[df.index[i-1], "final_lb"]):
                df.loc[df.index[i], "final_lb"] = df.loc[df.index[i], "basic_lb"]
            else:
                df.loc[df.index[i], "final_lb"] = df.loc[df.index[i-1], "final_lb"]

            df.loc[df.index[i], "final_ub"] = extend_value(
                df.loc[df.index[i], "final_ub"],
                df.loc[df.index[i-1], "final_ub"]
            )
            df.loc[df.index[i], "final_lb"] = extend_value(
                df.loc[df.index[i], "final_lb"],
                df.loc[df.index[i-1], "final_lb"]
            )
        
        df["supertrend"] = df["final_ub"].copy()
        df.loc[df["closePrice"] > df["final_ub"], "supertrend"] = df["final_lb"]

        for i in range(1, len(df)):
            df.loc[df.index[i], "supertrend"] = extend_value(
                df.loc[df.index[i], "supertrend"],
                df.loc[df.index[i-1], "supertrend"]
            )

        return df

    except Exception as e:
        logger.exception(f"Ошибка в calculate_supertrend_bybit_8_1: {e}")
        return pd.DataFrame()


def process_symbol_supertrend_open(symbol, interval="1", length=3, multiplier=1.0):
    df = get_historical_data_for_trading(symbol, interval=interval, limit=200)
    if df.empty or len(df) < 3:
        logger.info(f"{symbol}: Недостаточно данных для SuperTrend (нужно >=3 свеч).")
        return
    st_df = calculate_supertrend_bybit_8_1(df.copy(), length=length, multiplier=multiplier)
    if st_df.empty or len(st_df) < 3:
        logger.info(f"{symbol}: В st_df меньше 3 строк или NaN => пропуск.")
        return
    i0 = len(st_df) - 1
    i1 = i0 - 1
    i2 = i0 - 2
    o1 = st_df["openPrice"].iloc[i1]
    c1 = st_df["closePrice"].iloc[i1]
    st1 = st_df["supertrend"].iloc[i1]
    o0 = st_df["openPrice"].iloc[i0]
    c0 = st_df["closePrice"].iloc[i0]
    st0 = st_df["supertrend"].iloc[i0]
    is_buy = ((o1 < st1) and (c1 > st1) and (o0 > st0))
    is_sell = ((o1 > st1) and (c1 < st1) and (o0 < st0))
    if is_buy:
        logger.info(f"[SuperTrend_3Candles] {symbol}: Сигнал BUY (3-свеч. условие).")
        open_position(symbol, "Buy", POSITION_VOLUME, reason=f"SuperTrend_3C_{interval}")
    elif is_sell:
        logger.info(f"[SuperTrend_3Candles] {symbol}: Сигнал SELL (3-свеч. условие).")
        open_position(symbol, "Sell", POSITION_VOLUME, reason=f"SuperTrend_3C_{interval}")
    else:
        logger.info(f"[SuperTrend_3Candles] {symbol}: условие не выполнено.")

# def process_symbol_st_cross(symbol, interval="1", limit=200):
#     logger.info(f"[ST_cross] Начало обработки {symbol}")
#
#     with open_positions_lock:
#         if symbol in open_positions:
#             logger.info(f"[ST_cross] {symbol}: уже есть открытая позиция, пропускаем.")
#             return
#
#     df = get_historical_data_for_trading(symbol, interval=interval, limit=limit)
#     if df.empty or len(df) < 5:
#         logger.info(f"[ST_cross] {symbol}: Недостаточно данных, пропуск.")
#         return
#
#     df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
#     df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
#
#     if df_fast.empty or df_slow.empty:
#         logger.info(f"[ST_cross] {symbol}: Не удалось рассчитать SuperTrend.")
#         return
#
#     try:
#         last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
#         current_time = pd.Timestamp.utcnow()
#         
#         if last_candle_time < current_time - pd.Timedelta(minutes=5):
#             logger.warning(f"[ST_cross] {symbol}: Данные устарели! Пропускаем.")
#             return
#     except Exception as e:
#         logger.error(f"[ST_cross] Ошибка проверки времени для {symbol}: {e}")
#         return
#
#     df_fast, df_slow = df_fast.align(df_slow, join="inner", axis=0)
#
#     prev_fast = df_fast.iloc[-2]["supertrend"]
#     curr_fast = df_fast.iloc[-1]["supertrend"]
#     prev_slow = df_slow.iloc[-2]["supertrend"]
#     curr_slow = df_slow.iloc[-1]["supertrend"]
#
#     prev_diff = prev_fast - prev_slow
#     curr_diff = curr_fast - curr_slow
#     last_close = df_fast.iloc[-1]["closePrice"]
#     margin = 0.01
#
#     # Добавляем минимальное требование к изменению разницы, чтобы сигнал был значимым
#     min_diff_threshold = Decimal("0.0001")  # минимальное изменение
#
#     first_cross_up = (prev_diff < 0 or (prev_diff == 0 and curr_diff > min_diff_threshold)) and curr_diff > 0
#     first_cross_down = (prev_diff > 0 or (prev_diff == 0 and curr_diff < -min_diff_threshold)) and curr_diff < 0
#
#     confirmed_buy = first_cross_up and last_close >= curr_fast * (1 + margin)
#     confirmed_sell = first_cross_down and last_close <= curr_fast * (1 - margin)
#
#     logger.info(
#         f"[ST_cross] {symbol}: prev_fast={prev_fast:.6f}, prev_slow={prev_slow:.6f}, "
#         f"curr_fast={curr_fast:.6f}, curr_slow={curr_slow:.6f}, last_close={last_close:.6f}"
#     )
#
#     if confirmed_buy:
#         logger.info(f"[ST_cross] {symbol}: Подтверждён сигнал BUY")
#         open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross")
#     elif confirmed_sell:
#         logger.info(f"[ST_cross] {symbol}: Подтверждён сигнал SELL")
#         open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross")
#     else:
#         logger.info(f"[ST_cross] {symbol}: Пересечения не подтвердились, сигнал отсутствует")

def process_symbol_st_cross_global(symbol, interval="1", limit=200):
    """
    Базовый вариант ST_cross: определение сигнала по пересечению fast и slow SuperTrend,
    без дополнительных ограничений по процентному разрыву.
    """
    logger.info(f"[ST_cross_global] Начало обработки {symbol}")

    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross_global] {symbol}: уже есть открытая позиция, пропускаем.")
            return

    df = get_historical_data_for_trading(symbol, interval=interval, limit=limit)
    if df.empty or len(df) < 5:
        logger.info(f"[ST_cross_global] {symbol}: Недостаточно данных, пропуск.")
        return

    df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
    df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
    if df_fast.empty or df_slow.empty:
        logger.info(f"[ST_cross_global] {symbol}: Не удалось рассчитать SuperTrend.")
        return

    try:
        last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
        current_time = pd.Timestamp.utcnow()
        if last_candle_time < current_time - pd.Timedelta(minutes=5):
            logger.warning(f"[ST_cross_global] {symbol}: Данные устарели! Пропускаем.")
            return
    except Exception as e:
        logger.error(f"[ST_cross_global] Ошибка проверки времени для {symbol}: {e}")
        return

    df_fast, df_slow = df_fast.align(df_slow, join="inner", axis=0)
    prev_fast = df_fast.iloc[-2]["supertrend"]
    curr_fast = df_fast.iloc[-1]["supertrend"]
    prev_slow = df_slow.iloc[-2]["supertrend"]
    curr_slow = df_slow.iloc[-1]["supertrend"]

    prev_diff = prev_fast - prev_slow
    curr_diff = curr_fast - curr_slow
    last_close = df_fast.iloc[-1]["closePrice"]
    margin = 0.01

    first_cross_up = prev_diff <= 0 and curr_diff > 0
    first_cross_down = prev_diff >= 0 and curr_diff < 0

    confirmed_buy = first_cross_up and last_close >= curr_fast * (1 + margin)
    confirmed_sell = first_cross_down and last_close <= curr_fast * (1 - margin)

    logger.info(
        f"[ST_cross_global] {symbol}: prev_fast={prev_fast:.6f}, prev_slow={prev_slow:.6f}, "
        f"curr_fast={curr_fast:.6f}, curr_slow={curr_slow:.6f}, last_close={last_close:.6f}"
    )

    if confirmed_buy:
        logger.info(f"[ST_cross_global] {symbol}: Подтверждён сигнал BUY")
        open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross_global")
    elif confirmed_sell:
        logger.info(f"[ST_cross_global] {symbol}: Подтверждён сигнал SELL")
        open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross_global")
    else:
        logger.info(f"[ST_cross_global] {symbol}: Пересечения не подтвердились, сигнал отсутствует")

def process_symbol_st_cross1(symbol, interval="1", limit=200):
    """
    Вариант ST_cross1: открытие позиции по сигналу свечей, но с дополнительным ограничением по процентному
    разрыву между fast и slow SuperTrend.
      - При сигнале LONG: если текущее процентное различие (curr_diff_pct) больше +1%, позиция не открывается.
      - При сигнале SHORT: если curr_diff_pct меньше -1%, позиция не открывается.
    """
    logger.info(f"[ST_cross1] Начало обработки {symbol}")

    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross1] {symbol}: уже есть открытая позиция, пропускаем.")
            return

    df = get_historical_data_for_trading(symbol, interval=interval, limit=limit)
    if df.empty or len(df) < 5:
        logger.info(f"[ST_cross1] {symbol}: Недостаточно данных, пропуск.")
        return

    df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
    df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
    if df_fast.empty or df_slow.empty:
        logger.info(f"[ST_cross1] {symbol}: Не удалось рассчитать SuperTrend.")
        return

    try:
        last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
        current_time = pd.Timestamp.utcnow()
        if last_candle_time < current_time - pd.Timedelta(minutes=5):
            logger.warning(f"[ST_cross1] {symbol}: Данные устарели! Пропускаем.")
            return
    except Exception as e:
        logger.error(f"[ST_cross1] Ошибка проверки времени для {symbol}: {e}")
        return

    df_fast, df_slow = df_fast.align(df_slow, join="inner", axis=0)
    prev_fast = df_fast.iloc[-2]["supertrend"]
    curr_fast = df_fast.iloc[-1]["supertrend"]
    prev_slow = df_slow.iloc[-2]["supertrend"]
    curr_slow = df_slow.iloc[-1]["supertrend"]

    prev_diff = prev_fast - prev_slow
    curr_diff = curr_fast - curr_slow
    last_close = df_fast.iloc[-1]["closePrice"]

    # Процентное соотношение разрыва относительно последней цены
    curr_diff_pct = (Decimal(curr_diff) / Decimal(last_close)) * 100
    margin = 0.01

    first_cross_up = prev_diff <= 0 and curr_diff > 0
    first_cross_down = prev_diff >= 0 and curr_diff < 0

    if first_cross_up:
        # Для LONG: если curr_diff_pct больше +1% (слишком высокое положительное различие), не открываем
        if curr_diff_pct > Decimal("1"):
            logger.info(f"[ST_cross1] {symbol}: Слишком сильное положительное различие ({curr_diff_pct:.2f}%), не открываем LONG.")
            return
        confirmed_buy = last_close >= curr_fast * (1 + margin)
        if confirmed_buy:
            logger.info(f"[ST_cross1] {symbol}: Подтверждён сигнал BUY")
            open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross1")
        else:
            logger.info(f"[ST_cross1] {symbol}: Сигнал BUY не подтверждён по цене.")
    elif first_cross_down:
        # Для SHORT: если curr_diff_pct меньше -1% (слишком сильное отрицательное различие), не открываем
        if curr_diff_pct < Decimal("-1"):
            logger.info(f"[ST_cross1] {symbol}: Слишком сильное отрицательное различие ({curr_diff_pct:.2f}%), не открываем SHORT.")
            return
        confirmed_sell = last_close <= curr_fast * (1 - margin)
        if confirmed_sell:
            logger.info(f"[ST_cross1] {symbol}: Подтверждён сигнал SELL")
            open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross1")
        else:
            logger.info(f"[ST_cross1] {symbol}: Сигнал SELL не подтверждён по цене.")
    else:
        logger.info(f"[ST_cross1] {symbol}: Пересечения не обнаружены, сигнал отсутствует.")

def process_symbol_st_cross2(symbol, interval="1", limit=200):
    """
    Вариант ST_cross2: открытие позиции по изменению процентного разрыва между fast и slow SuperTrend.
      - Сигнал LONG: если в предыдущей свече процентный разрыв был ≤ –0.3% и в текущей стал ≥ +0.3%.
      - Сигнал SHORT: если в предыдущей свече процентный разрыв был ≥ +0.3% и в текущей стал ≤ –0.3%.
      Дополнительно: если для LONG текущее процентное различие больше +1%, позиция не открывается;
                   для SHORT – если меньше –1%, не открывается.
    """
    logger.info(f"[ST_cross2] Начало обработки {symbol}")

    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross2] {symbol}: уже есть открытая позиция, пропускаем.")
            return

    df = get_historical_data_for_trading(symbol, interval=interval, limit=limit)
    if df.empty or len(df) < 5:
        logger.info(f"[ST_cross2] {symbol}: Недостаточно данных, пропуск.")
        return

    df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
    df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
    if df_fast.empty or df_slow.empty:
        logger.info(f"[ST_cross2] {symbol}: Не удалось рассчитать SuperTrend.")
        return

    try:
        last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
        current_time = pd.Timestamp.utcnow()
        if last_candle_time < current_time - pd.Timedelta(minutes=5):
            logger.warning(f"[ST_cross2] {symbol}: Данные устарели! Пропускаем.")
            return
    except Exception as e:
        logger.error(f"[ST_cross2] Ошибка проверки времени для {symbol}: {e}")
        return

    df_fast, df_slow = df_fast.align(df_slow, join="inner", axis=0)
    prev_fast = df_fast.iloc[-2]["supertrend"]
    curr_fast = df_fast.iloc[-1]["supertrend"]
    prev_slow = df_slow.iloc[-2]["supertrend"]
    curr_slow = df_slow.iloc[-1]["supertrend"]

    prev_diff = prev_fast - prev_slow
    curr_diff = curr_fast - curr_slow
    last_close = df_fast.iloc[-1]["closePrice"]

    prev_diff_pct = (Decimal(prev_diff) / Decimal(last_close)) * 100
    curr_diff_pct = (Decimal(curr_diff) / Decimal(last_close)) * 100

    # Для ST_cross2 используем пороговые значения 0.3% для смены сигнала
    long_signal = (prev_diff_pct <= Decimal("-0.3") and curr_diff_pct >= Decimal("0.3"))
    short_signal = (prev_diff_pct >= Decimal("0.3") and curr_diff_pct <= Decimal("-0.3"))

    if long_signal:
        if curr_diff_pct > Decimal("1"):
            logger.info(f"[ST_cross2] {symbol}: Текущее положительное различие ({curr_diff_pct:.2f}%) слишком высокое, не открываем LONG.")
            return
        logger.info(f"[ST_cross2] {symbol}: Сигнал LONG по изменению diff_pct (prev: {prev_diff_pct:.2f}%, curr: {curr_diff_pct:.2f}%).")
        open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross2")
    elif short_signal:
        if curr_diff_pct < Decimal("-1"):
            logger.info(f"[ST_cross2] {symbol}: Текущее отрицательное различие ({curr_diff_pct:.2f}%) слишком низкое, не открываем SHORT.")
            return
        logger.info(f"[ST_cross2] {symbol}: Сигнал SHORT по изменению diff_pct (prev: {prev_diff_pct:.2f}%, curr: {curr_diff_pct:.2f}%).")
        open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross2")
    else:
        logger.info(f"[ST_cross2] {symbol}: Альтернативное условие для сигнала не выполнено.")


def process_symbol_st_cross2_drift(symbol, interval="1", limit=200):
    """
    Режим ST_cross2_drift:
      - Выполняет стандартную логику ST_cross2:
          * Сигнал LONG: если в предыдущей свече процентный разрыв между fast и slow SuperTrend был ≤ –0.3%
            и в текущей стал ≥ +0.3% (при условии, что текущее различие не превышает +1%).
          * Сигнал SHORT: если в предыдущей свече процентный разрыв был ≥ +0.3%
            и в текущей стал ≤ –0.3% (при условии, что текущее различие не ниже –1%).
      - Дополнительно, если drift-позиция ещё не открыта (drift_trade_executed == False),
        выбирается ТОП-1 drift-сигнал из глобального словаря drift_history.
        При drift-сигнале "вверх" открывается шорт (Sell), а при "вниз" – лонг (Buy) на фиксированный объём 500 USDT.
    """
    logger.info(f"[ST_cross2_drift] Начало обработки {symbol}")

    # Если позиция для данного символа уже открыта, пропускаем основную логику
    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross2_drift] {symbol}: уже есть открытая позиция, пропускаем ST_cross2 логику.")
            return

    # Основная логика ST_cross2
    df = get_historical_data_for_trading(symbol, interval=interval, limit=limit)
    if df.empty or len(df) < 5:
        logger.info(f"[ST_cross2_drift] {symbol}: Недостаточно данных, пропуск ST_cross2 логику.")
    else:
        df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
        df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
        if df_fast.empty or df_slow.empty:
            logger.info(f"[ST_cross2_drift] {symbol}: Не удалось рассчитать SuperTrend, пропуск ST_cross2 логику.")
        else:
            try:
                last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
                current_time = pd.Timestamp.utcnow()
                if last_candle_time < current_time - pd.Timedelta(minutes=5):
                    logger.warning(f"[ST_cross2_drift] {symbol}: Данные устарели! Пропускаем ST_cross2 логику.")
                else:
                    df_fast, df_slow = df_fast.align(df_slow, join="inner", axis=0)
                    prev_fast = df_fast.iloc[-2]["supertrend"]
                    curr_fast = df_fast.iloc[-1]["supertrend"]
                    prev_slow = df_slow.iloc[-2]["supertrend"]
                    curr_slow = df_slow.iloc[-1]["supertrend"]

                    prev_diff = prev_fast - prev_slow
                    curr_diff = curr_fast - curr_slow
                    last_close = df_fast.iloc[-1]["closePrice"]

                    prev_diff_pct = (Decimal(prev_diff) / Decimal(last_close)) * 100
                    curr_diff_pct = (Decimal(curr_diff) / Decimal(last_close)) * 100

                    # Определяем сигналы по изменению процентного разрыва
                    long_signal = (prev_diff_pct <= Decimal("-0.3") and curr_diff_pct >= Decimal("0.3"))
                    short_signal = (prev_diff_pct >= Decimal("0.3") and curr_diff_pct <= Decimal("-0.3"))

                    if long_signal:
                        if curr_diff_pct > Decimal("1"):
                            logger.info(f"[ST_cross2_drift] {symbol}: Текущее положительное различие ({curr_diff_pct:.2f}%) слишком высокое, не открываем LONG.")
                        else:
                            logger.info(f"[ST_cross2_drift] {symbol}: Сигнал LONG (prev: {prev_diff_pct:.2f}%, curr: {curr_diff_pct:.2f}%).")
                            open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross2_drift")
                    elif short_signal:
                        if curr_diff_pct < Decimal("-1"):
                            logger.info(f"[ST_cross2_drift] {symbol}: Текущее отрицательное различие ({curr_diff_pct:.2f}%) слишком низкое, не открываем SHORT.")
                        else:
                            logger.info(f"[ST_cross2_drift] {symbol}: Сигнал SHORT (prev: {prev_diff_pct:.2f}%, curr: {curr_diff_pct:.2f}%).")
                            open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross2_drift")
                    else:
                        logger.info(f"[ST_cross2_drift] {symbol}: Условия ST_cross2 не выполнены.")
            except Exception as e:
                logger.error(f"[ST_cross2_drift] Ошибка в обработке {symbol}: {e}")

    # Дополнительная логика drift: открытие позиции по ТОП-1 drift-сигналу на 500 USDT.
    # При drift-сигнале "вверх" открывается шорт (Sell), а при "вниз" – лонг (Buy).
    global drift_trade_executed
    if not drift_trade_executed:
        drift_signals = []
        # Собираем drift-сигналы из drift_history
        for drift_sym, recs in drift_history.items():
            if recs:
                avg_strength = sum(x[1] for x in recs) / len(recs)
                last_direction = recs[-1][2]
                drift_signals.append((drift_sym, avg_strength, last_direction))
        if drift_signals:
            drift_signals.sort(key=lambda x: x[1], reverse=True)
            top_drift = drift_signals[0]  # ТОП-1 drift-сигнал
            drift_sym, drift_avg_strength, drift_direction = top_drift
            with open_positions_lock:
                if drift_sym in open_positions:
                    logger.info(f"[ST_cross2_drift] Drift: позиция для {drift_sym} уже открыта, пропускаем drift trade.")
                else:
                    # При drift-сигнале "вверх" открываем шорт (Sell), иначе – лонг (Buy)
                    drift_side = "Sell" if drift_direction == "вверх" else "Buy"
                    logger.info(f"[ST_cross2_drift] Открываю drift позицию для {drift_sym}: {drift_side} на 500 USDT (drift: {drift_direction}).")
                    open_position(drift_sym, drift_side, 500, reason="ST_cross2_drift_drift")
                    drift_trade_executed = True
        else:
            logger.info("[ST_cross2_drift] Нет drift-сигналов для обработки.")

### ПРАВКА (7) analyze_trend(...) – «Предложить как увязать»
###   Пока просто закомментируем, сохранив как "пример".
"""
def analyze_trend(values):
    \"\"\"Анализирует направление тренда по последним свечам (пример).\"\"\"
    fast_trend = values[\"fast\"][-1] > values[\"fast\"][0]
    slow_trend = values[\"slow\"][-1] > values[\"slow\"][0]
    price_trend = values[\"close\"][-1] > values[\"close\"][0]
    
    volume_confirms = True
    if values[\"volume\"] is not None:
        volume_confirms = values[\"volume\"][-1] > values[\"volume\"].mean()
    
    if fast_trend and slow_trend and price_trend and volume_confirms:
        return \"uptrend\"
    elif not fast_trend and not slow_trend and not price_trend:
        return \"downtrend\"
    return \"sideways\"
"""


### ПРАВКА (8) calculate_cross_signal(...) – Закомментировать
"""
def calculate_cross_signal(prev_fast, curr_fast, prev_slow, curr_slow, last_close, trend):
    \"\"\"Рассчитывает торговый сигнал на основе пересечения и тренда\"\"\"
    try:
        prev_diff = prev_fast - prev_slow
        curr_diff = curr_fast - curr_slow
        min_distance = last_close * Decimal(\"0.005\")
        
        cross_up = prev_diff <= 0 and curr_diff > 0 and curr_diff > min_distance
        cross_down = prev_diff >= 0 and curr_diff < 0 and abs(curr_diff) > min_distance
        
        if cross_up and trend == \"uptrend\" and last_close > curr_fast:
            return {\"direction\": \"Buy\", \"strength\": float(curr_diff)}
        elif cross_down and trend == \"downtrend\" and last_close < curr_fast:
            return {\"direction\": \"Sell\", \"strength\": float(abs(curr_diff))}
        return None
        
    except Exception as e:
        logger.exception(f\"Ошибка в calculate_cross_signal: {e}\")
        return None
"""


def escape_markdown(text: str) -> str:
    escape_chars = r"_*\[\]()~`>#+\-={}|.,!\\"
    pattern = re.compile(r"([%s])" % re.escape(escape_chars))
    return pattern.sub(r"\\\1", text)


# ПРАВКА (9) set_take_profit(...) – Удалить (полностью):
"""
def set_take_profit(symbol, size, entry_price, side):
    ...
"""
# ==> Удалили.


# ===================== ОБНОВЛЕНИЕ open_positions (п.1, 2, 3) =====================
def update_open_positions_from_exch_positions(expos: dict):
    """
    Синхронизирует локальный словарь `open_positions` с биржевыми позициями `expos`.
    """
    with open_positions_lock, state_lock:
        to_remove = []
        for sym in list(open_positions.keys()):
            if sym not in expos:
                pos = open_positions[sym]
                side = pos["side"]
                volume = Decimal(str(pos.get("position_volume", 0)))
                trade_id = pos.get("trade_id")
                close_price = get_last_close_price(sym)
                pnl = Decimal("0")
                if close_price and volume > 0:
                    cp = Decimal(str(close_price))
                    ep = Decimal(str(pos.get("avg_price", 0)))
                    if side.lower() == "buy":
                        pnl = (cp - ep) / ep * volume
                    else:
                        pnl = (ep - cp) / ep * volume
                if trade_id:
                    update_trade_outcome(trade_id, float(pnl))
                close_side = "Sell" if side.lower() == "buy" else "Buy"
                log_trade(sym, get_last_row(sym), None, close_side, "Closed", closed_manually=True)
                state["total_open_volume"] -= volume
                if state["total_open_volume"] < Decimal("0"):
                    state["total_open_volume"] = Decimal("0")
                to_remove.append(sym)
        for sym in to_remove:
            del open_positions[sym]

        for sym, newpos in expos.items():
            if sym in open_positions:
                open_positions[sym]["side"]            = newpos["side"]
                open_positions[sym]["size"]            = newpos["size"]
                open_positions[sym]["avg_price"]       = newpos["avg_price"]
                open_positions[sym]["position_volume"] = newpos["position_volume"]
                open_positions[sym]["positionIdx"]     = newpos["positionIdx"]
            else:
                open_positions[sym] = {
                    "side": newpos["side"],
                    "size": newpos["size"],
                    "avg_price": newpos["avg_price"],
                    "position_volume": newpos["position_volume"],
                    "symbol": sym,
                    "positionIdx": newpos.get("positionIdx"),
                    "trailing_stop_set": False,
                    "trade_id": None,
                    "open_time": datetime.datetime.utcnow(),
                }

        total = sum(Decimal(str(p["position_volume"])) for p in open_positions.values())
        state["total_open_volume"] = total

def get_last_row(symbol):
    df = get_historical_data_for_trading(symbol, "1", limit=1)
    if df.empty:
        return None
    return df.iloc[-1]


# ===================== УСТАНОВКА / ПРОВЕРКА ТРЕЙЛИНГ-СТОПА =====================
def check_and_set_trailing_stop():
    if not TRAILING_STOP_ENABLED:
        return
    try:
        with open_positions_lock:
            positions_copy = dict(open_positions)
        threshold_roi = Decimal("5.0")
        default_leverage = Decimal("10")
        for sym, pos in positions_copy.items():
            if pos.get("trailing_stop_set"):
                continue
            side = pos["side"]
            entry_price = Decimal(str(pos["avg_price"]))
            current_price = get_last_close_price(sym)
            if current_price is None:
                continue
            cp = Decimal(str(current_price))
            ep = Decimal(str(entry_price))
            if side.lower() == "buy":
                ratio = (cp - ep) / ep
            else:
                ratio = (ep - cp) / ep
            leveraged_pnl_percent = ratio * default_leverage * Decimal("100")
            # MODIFICATION 1: Сохраняем profit_perc в open_positions для использования в trailing stop
            with open_positions_lock:
                if sym in open_positions:
                    open_positions[sym]['profit_perc'] = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
            if leveraged_pnl_percent >= threshold_roi:
                if not pos.get("trailing_stop_set", False):
                    logger.info(f"[HTTP Monitor] {sym}: Достигнут уровень для трейлинг-стопа (leveraged PnL = {leveraged_pnl_percent}%). Устанавливаю трейлинг-стоп.")
                    set_trailing_stop(sym, pos["size"], TRAILING_GAP_PERCENT, side)
    except Exception as e:
        logger.exception(f"Ошибка check_and_set_trailing_stop: {e}")

def set_trailing_stop(symbol, size, trailing_gap_percent, side):
    try:
        pos_info = get_position_info(symbol, side)
        if not pos_info:
            logger.error(f"[set_trailing_stop] Нет позиции {symbol}/{side}")
            return
        pos_idx = pos_info.get("positionIdx")
        if not pos_idx:
            return

        avg_price = Decimal(str(pos_info.get("avgPrice", "0")))
        if avg_price <= 0:
            return

        trailing_distance_abs = (avg_price * trailing_gap_percent).quantize(Decimal("0.0000001"))
        dynamic_min = max(avg_price * Decimal("0.0000001"), MIN_TRAILING_STOP)
        if trailing_distance_abs < dynamic_min:
            logger.info(f"[set_trailing_stop] {symbol}: trailingStop={trailing_distance_abs}< {dynamic_min}, пропуск.")
            return

        resp = session.set_trading_stop(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="TrailingStop",
            qty=str(size),
            trailingStop=str(trailing_distance_abs),
            timeInForce="GoodTillCancel",
            positionIdx=pos_idx
        )
        if resp:
            rc = resp.get("retCode")
            if rc == 0:
                with open_positions_lock:
                    if symbol in open_positions:
                        open_positions[symbol]["trailing_stop_set"] = True
                    # MODIFICATION 2: Получаем profit_perc для включения в сообщение
                    pnl_display = open_positions.get(symbol, {}).get("profit_perc", Decimal("0"))
                row = get_last_row(symbol)
                # В сообщение о трейлинг-стопе включаем PnL
                log_trade(symbol, row, None, f"{trailing_distance_abs} (PnL: {pnl_display}%)", "Trailing Stop Set", closed_manually=False)
                logger.info(f"[set_trailing_stop] OK {symbol}")
            elif rc == 34040:
                logger.info("[set_trailing_stop] not modified, retCode=34040.")
            else:
                logger.error(f"[set_trailing_stop] Ошибка: {resp.get('retMsg')}")
    except Exception as e:
        logger.exception(f"[set_trailing_stop] {symbol}: {e}")


### ПРАВКА (10) check_and_close_profitable_positions(...) – Закомментировать
"""
def check_and_close_profitable_positions():
    \"\"\"
    Не вызывается прямо сейчас.
    \"\"\"
    try:
        with open_positions_lock:
            positions_copy = dict(open_positions)
        to_close = []
        for sym, pos in positions_copy.items():
            side = pos[\"side\"]
            ep = Decimal(str(pos[\"avg_price\"]))
            current = get_last_close_price(sym)
            if current is None:
                continue
            cp = Decimal(str(current))
            if side.lower() == \"buy\":
                ratio = (cp - ep) / ep
            else:
                ratio = (ep - cp) / ep
            profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal(\"0.008\"))
            logger.info(f\"[ProfitCheck] {sym}: profit%={profit_perc}\")
            if profit_perc >= PROFIT_LEVEL:
                to_close.append(sym)
        for sym in to_close:
            with open_positions_lock:
                if sym not in open_positions:
                    continue
                pos = open_positions[sym]
                side = pos[\"side\"]
                size = pos[\"size\"]
                volume = Decimal(str(pos[\"position_volume\"]))
                trade_id = pos.get(\"trade_id\", None)
            close_side = \"Sell\" if side.lower() == \"buy\" else \"Buy\"
            posIdx = 1 if side.lower() == \"buy\" else 2
            res = place_order(sym, close_side, size, \"Market\", \"GoodTillCancel\", True, positionIdx=posIdx)
            if res and res.get(\"retCode\") == 0:
                close_price = get_last_close_price(sym)
                pnl = Decimal(\"0\")
                if close_price:
                    cp = Decimal(str(close_price))
                    ep = Decimal(str(pos[\"avg_price\"]))
                    if side.lower() == \"buy\":
                        pnl = (cp - ep) / ep * Decimal(str(pos[\"position_volume\"]))
                    else:
                        pnl = (ep - cp) / ep * Decimal(str(pos[\"position_volume\"]))
                if trade_id:
                    update_trade_outcome(trade_id, float(pnl))
                log_trade(sym, get_last_row(sym), None, close_side, \"Closed\", closed_manually=True)
                with state_lock:
                    state[\"total_open_volume\"] -= volume
                    if state[\"total_open_volume\"] < Decimal(\"0\"):
                        state[\"total_open_volume\"] = Decimal(\"0\")
                with open_positions_lock:
                    del open_positions[sym]
    except Exception as e:
        logger.exception(f\"Ошибка check_and_close_profitable_positions: {e}\")
"""

def generate_daily_pnl_report(input_csv="trade_log.csv", output_csv="daily_pnl_report.csv"):
    pnl_records = []
    try:
        if not os.path.isfile(input_csv):
            return
        with open(input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("result", "").lower() != "closed":
                    continue
                if row.get("closed_manually", "").strip().lower() == "true":
                    continue
                action = row.get("action", "").lower()
                try:
                    open_price = Decimal(row.get("openPrice", "0"))
                    close_price = Decimal(row.get("closePrice", "0"))
                    volume_usdt = Decimal(row.get("volume", "0"))
                except:
                    continue
                if open_price == 0 or close_price == 0 or volume_usdt == 0:
                    continue
                pnl = Decimal("0")
                if action == "buy":
                    pnl = (close_price - open_price) / open_price * volume_usdt
                elif action == "sell":
                    pnl = (open_price - close_price) / open_price * volume_usdt
                pnl_records.append({
                    "Дата/время": row.get("timestamp", ""),
                    "Символ": row.get("symbol", ""),
                    "Объём в USDT": str(volume_usdt),
                    "Прибыль/Убыток": f"{pnl:.2f}"
                })
    except Exception as e:
        logger.exception(f"Ошибка при генерации PnL: {e}")
        return
    try:
        if not pnl_records:
            logger.info("Нет данных для daily_pnl_report.")
            return
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["Дата/время", "Символ", "Объём в USDT", "Прибыль/Убыток"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in pnl_records:
                writer.writerow(r)
        logger.info(f"PNL отчёт сохранён в {output_csv}")
    except Exception as e:
        logger.exception(f"Ошибка записи PnL: {e}")


# ===================== ФУНКЦИИ ДЛЯ АВЕРСЕДНЯЮЩИХ ПОЗИЦИЙ =====================
def open_averaging_position(symbol):
    """
    Открывает позицию усреднения для заданного символа, если по нему уже открыта базовая позиция,
    усредняющая позиция ещё не открыта, и добавление нового объёма не превысит лимит.
    """
    try:
        with open_positions_lock:
            if symbol not in open_positions:
                logger.info(f"[Averaging] Нет базовой позиции для {symbol}, пропуск усреднения.")
                return
            if symbol in averaging_positions:
                logger.info(f"[Averaging] Усредняющая позиция для {symbol} уже открыта, пропуск.")
                return
            base_pos = open_positions[symbol]
            side = base_pos["side"]
            base_volume_usdt = Decimal(str(base_pos["position_volume"]))

            global averaging_total_volume
            if averaging_total_volume + base_volume_usdt > MAX_AVERAGING_VOLUME:
                logger.info(
                    f"[Averaging] Превышен лимит усреднения: {averaging_total_volume} + "
                    f"{base_volume_usdt} > {MAX_AVERAGING_VOLUME}")
                return

        order_result = place_order(
            symbol=symbol,
            side=side,
            qty=float(base_volume_usdt),
            order_type="Market",
            time_in_force="GoodTillCancel",
            reduce_only=False,
            positionIdx=1 if side.lower() == "buy" else 2
        )
        if order_result and order_result.get("retCode") == 0:
            with open_positions_lock:
                averaging_positions[symbol] = {
                    "side": side,
                    "volume": base_volume_usdt,
                    "opened_at": datetime.datetime.utcnow(),
                    "trade_id": f"averaging_{symbol}_{int(time.time())}"
                }
            averaging_total_volume += base_volume_usdt
            logger.info(
                f"[Averaging] Усредняющая позиция для {symbol} открыта на объём {base_volume_usdt}. "
                f"Текущий усредняющий объём: {averaging_total_volume}")
        else:
            logger.error(f"[Averaging] Ошибка открытия усредняющей позиции для {symbol}: {order_result}")
    except Exception as e:
        logger.exception(f"[Averaging] Ошибка в open_averaging_position для {symbol}: {e}")


### ПРАВКА (11) http_monitor_positions(...) – Закомментировать
"""
def http_monitor_positions():
    \"\"\"
    Мониторинг открытых позиций через HTTP-запросы.
    Для каждого символа из open_positions:
      - Получаем текущую цену
      - Считаем % прибыли/убытка
      - Если убыток <= -TARGET_LOSS_FOR_AVERAGING => усреднение.
    \"\"\"
    with open_positions_lock:
        symbols = list(open_positions.keys())
    for symbol in symbols:
        current_price = get_last_close_price(symbol)
        if current_price is None:
            logger.info(f\"[HTTP Monitor] Нет текущей цены для {symbol}\")
            continue
        with open_positions_lock:
            pos = open_positions[symbol]
        side = pos[\"side\"]
        entry_price = Decimal(str(pos[\"avg_price\"]))
        if side.lower() == \"buy\":
            ratio = (Decimal(str(current_price)) - entry_price) / entry_price
        else:
            ratio = (entry_price - Decimal(str(current_price))) / entry_price
        profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal(\"0.0001\"))
        logger.info(f\"[HTTP Monitor] {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%\")

        if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
            logger.info(
                f\"[HTTP Monitor] {symbol} достиг порога убытка ({profit_perc}% <= -{TARGET_LOSS_FOR_AVERAGING}). \"
                \"Открываю усредняющую позицию.\"
            )
            open_averaging_position(symbol)
"""
### MODIFICATION 3:
# В функцию /status добавлены блоки try/except для обработки ошибок при расчёте PnL для каждого символа.

@router.message(Command(commands=["status"]))
async def status_cmd(message: Message):
    with open_positions_lock:
        if not open_positions:
            await message.reply("Нет позиций.")
            return
        lines = []
        total_pnl_usdt = Decimal("0")
        total_invested = Decimal("0")
        # Копируем open_positions, чтобы не держать блокировку на долго
        positions_copy = open_positions.copy()
    for sym, pos in positions_copy.items():
        try:
            side_str = pos["side"]
            entry_price = Decimal(str(pos["avg_price"]))
            volume_usdt = Decimal(str(pos["position_volume"]))
            current_price = get_last_close_price(sym)
            if current_price is None:
                lines.append(f"{sym} {side_str}: нет текущей цены.")
                continue
            cp = Decimal(str(current_price))
            if side_str.lower() == "buy":
                ratio = (cp - entry_price) / entry_price
            else:
                ratio = (entry_price - cp) / entry_price
            pnl_usdt = ratio * volume_usdt
            pnl_percent = ratio * Decimal("100")
            total_pnl_usdt += pnl_usdt
            total_invested += volume_usdt
            lines.append(f"{sym} {side_str}: PNL = {pnl_usdt:.2f} USDT ({pnl_percent:.2f}%)")
        except Exception as e:
            logger.error(f"Ошибка при обработке {sym} в статусе: {e}")
            lines.append(f"{sym}: ошибка")
    lines.append("—" * 30)
    if total_invested > 0:
        total_pnl_percent = (total_pnl_usdt / total_invested) * Decimal("100")
        lines.append(f"Итоговый PnL по всем позициям: {total_pnl_usdt:.2f} USDT ({total_pnl_percent:.2f}%)")
    else:
        lines.append("Итоговый PnL: 0 (нет позиций с объёмом)")
    await message.reply("\n".join(lines))

@router.message(Command("stop"))
async def stop_command(message: Message):
    # Если нужно ограничивать команду по chat_id, убедитесь, что типы совпадают (int vs. str)
    # Например, можно сделать: if int(message.chat.id) != int(TELEGRAM_CHAT_ID): return
    global IS_RUNNING
    IS_RUNNING = False
    await message.answer("🛑 Бот останавливается...")
    logger.info("Получена команда /stop - инициируем остановку бота")
    # Попытка завершить все активные задачи (MODIFICATION 5)
    for task in asyncio.all_tasks():
        task.cancel()

@router.message(Command(commands=["menu"]))
async def main_menu_cmd(message: Message):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton("📈 Торговля"), KeyboardButton("🤖 Бот")],
            [KeyboardButton("ℹ️ Информация")]
        ],
        resize_keyboard=True  # Кнопки будут масштабироваться
    )
    await message.answer("Выберите раздел:", reply_markup=keyboard)


@router.callback_query(lambda c: c.data == "menu_trading")
async def menu_trading_cb(query: CallbackQuery):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Статус", callback_data="cmd_status")],
            [InlineKeyboardButton(text="🔄 Смена режима", callback_data="cmd_mode")],
            [InlineKeyboardButton(text="📉 Установить макс. объем", callback_data="cmd_setmaxvolume")],
            [InlineKeyboardButton(text="📊 Установить объем позиции", callback_data="cmd_setposvolume")],
            [InlineKeyboardButton(text="📉 Установить таймфрейм ST", callback_data="cmd_setsttf")],
            [InlineKeyboardButton(text="🔙 Назад", callback_data="menu_main")],
        ]
    )
    await query.message.edit_text("📈 **Торговля** – выберите действие:", parse_mode="Markdown", reply_markup=keyboard)

@router.callback_query(lambda c: c.data == "menu_bot")
async def menu_bot_cb(query: CallbackQuery):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🛑 Тихий режим ON/OFF", callback_data="cmd_togglesilence")],
            [InlineKeyboardButton(text="🔕 Статус тихого режима", callback_data="cmd_silencestatus")],
            [InlineKeyboardButton(text="😴 Усыпить бота", callback_data="cmd_sleep")],
            [InlineKeyboardButton(text="🌞 Разбудить бота", callback_data="cmd_wake")],
            [InlineKeyboardButton(text="🔙 Назад", callback_data="menu_main")],
        ]
    )
    await query.message.edit_text("🤖 **Управление ботом** – выберите действие:", parse_mode="Markdown", reply_markup=keyboard)

@router.callback_query(lambda c: c.data == "menu_info")
async def menu_info_cb(query: CallbackQuery):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔍 Получить данные по паре", callback_data="cmd_getpair")],
            [InlineKeyboardButton(text="📊 Публикация Drift-таблицы", callback_data="toggle_publish_drift")],
            [InlineKeyboardButton(text="📈 Публикация Model-таблицы", callback_data="toggle_publish_model")],
            [InlineKeyboardButton(text="📌 Model Top ON/OFF", callback_data="toggle_model_top")],
            [InlineKeyboardButton(text="🔙 Назад", callback_data="menu_main")],
        ]
    )
    await query.message.edit_text("ℹ️ **Информация** – выберите действие:", parse_mode="Markdown", reply_markup=keyboard)

@router.callback_query(lambda c: c.data == "menu_main")
async def menu_main_cb(query: CallbackQuery):
    await main_menu_cmd(query.message)

@router.callback_query(lambda c: c.data == "cmd_status")
async def process_cmd_status(query: CallbackQuery):
    await status_cmd(query.message)

@router.callback_query(lambda c: c.data == "cmd_mode")
async def process_cmd_mode(query: CallbackQuery):
    await change_or_get_mode_cmd(query.message)

@router.callback_query(lambda c: c.data == "cmd_setmaxvolume")
async def process_cmd_setmaxvolume(query: CallbackQuery):
    await query.message.answer("Введите команду вручную: `/setmaxvolume 500`", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_setposvolume")
async def process_cmd_setposvolume(query: CallbackQuery):
    await query.message.answer("Введите команду вручную: `/setposvolume 50`", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_setsttf")
async def process_cmd_setsttf(query: CallbackQuery):
    await query.message.answer("Введите команду вручную: `/setsttf 15`", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_togglesilence")
async def process_cmd_togglesilence(query: CallbackQuery):
    await toggle_silence_cmd(query.message)

@router.callback_query(lambda c: c.data == "cmd_silencestatus")
async def process_cmd_silencestatus(query: CallbackQuery):
    await silence_status_cmd(query.message)

@router.message(Command(commands=["mode"]))
async def change_or_get_mode_cmd(message: Message):
    available_modes = {
         "drift_only": "🌊 Drift Only",
         "drift_top10": "📊 Drift TOP-10",
         "golden_setup": "✨ Golden Setup",
         "super_trend": "📈 SuperTrend",
         "ST_cross1": "🔄 ST Cross1",
         "ST_cross2": "🔄 ST Cross2",
         "ST_cross_global": "🔄 ST Cross Global",
         "model_only": "🤖 Model Only"
    }
    keyboard = InlineKeyboardMarkup(
         inline_keyboard=[
              [InlineKeyboardButton(text=label, callback_data=f"set_mode_{mode}")]
              for mode, label in available_modes.items()
         ]
    )
    current_mode_label = available_modes.get(OPERATION_MODE, OPERATION_MODE)
    message_text = f"*Текущий режим*: {current_mode_label}\n\nВыберите новый режим работы:"
    await message.answer(message_text, reply_markup=keyboard, parse_mode="Markdown")

@router.message(Command(commands=["togglesilence"]))
async def toggle_silence_cmd(message: Message):
    st = toggle_quiet_period()
    await message.reply(f"Тихий режим: {st}")

@router.message(Command(commands=["silencestatus"]))
async def silence_status_cmd(message: Message):
    st = "включён" if QUIET_PERIOD_ENABLED else "выключен"
    await message.reply(f"Тихий режим: {st}")

@router.callback_query(lambda c: c.data == "cmd_sleep")
async def process_cmd_sleep(query: CallbackQuery):
    global QUIET_PERIOD_ENABLED
    QUIET_PERIOD_ENABLED = True
    await query.message.answer("😴 Бот переведен в **спящий режим**.", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_wake")
async def process_cmd_wake(query: CallbackQuery):
    global QUIET_PERIOD_ENABLED
    QUIET_PERIOD_ENABLED = False
    await query.message.answer("🌞 Бот **разбужен**, возобновлена торговля.", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_getpair")
async def process_cmd_getpair(query: CallbackQuery):
    await query.message.answer("Введите команду вручную: `/getpair BTCUSDT или BTC`", parse_mode="Markdown")

@router.message(Command(commands=["inline_menu"]))
async def inline_menu_command(message: Message):
    inline_kb = [
        [
            InlineKeyboardButton(text="Status", callback_data="cmd_status"),
            InlineKeyboardButton(text="Toggle Silence", callback_data="cmd_togglesilence"),
        ],
        [
            InlineKeyboardButton(text="Silence Status", callback_data="cmd_silencestatus"),
            InlineKeyboardButton(text="Set Max Volume", callback_data="cmd_setmaxvolume"),
        ],
        [
            InlineKeyboardButton(text="Set Pos Volume", callback_data="cmd_setposvolume"),
            InlineKeyboardButton(text="Set ST TF", callback_data="cmd_setsttf"),
        ]
    ]
    markup = InlineKeyboardMarkup(inline_keyboard=inline_kb)
    await message.answer("Выберите действие:", reply_markup=markup)

@router.callback_query(lambda c: c.data and c.data.startswith("cmd_"))
async def process_inline_commands(query: CallbackQuery):
    data = query.data
    if data == "cmd_status":
        await query.message.answer("Вызван STATUS — ваша логика здесь.")
    elif data == "cmd_togglesilence":
        await query.message.answer("Вызван TOGGLE SILENCE — ваша логика.")
    elif data == "cmd_silencestatus":
        await query.message.answer("Вызван SILENCE STATUS.")
    elif data == "cmd_setmaxvolume":
        await query.message.answer("Вызван SET MAX VOLUME. Пример: /setmaxvolume 500")
    elif data == "cmd_setposvolume":
        await query.message.answer("Вызван SET POS VOLUME. Пример: /setposvolume 50")
    elif data == "cmd_setsttf":
        await query.message.answer("Вызван SET ST TF. Пример: /setsttf 15")
    await query.answer()

# ===================== КОМАНДЫ ТЕЛЕГРАМ (sleep, wake, stop) =====================
@router.message(Command(commands=["sleep"]))
async def sleep_cmd(message: Message):
    status = toggle_sleep_mode()
    await message.reply(f"Спящий режим: {status}")

@router.message(Command(commands=["wake"]))
async def wake_cmd(message: Message):
    status = toggle_sleep_mode()
    await message.reply(f"Спящий режим: {status}")


def generate_drift_table_from_history(top_n=15) -> str:
    if not drift_history:
        return ""
    rows = []
    for sym, recs in drift_history.items():
        if not recs:
            continue
        avg_str = sum(x[1] for x in recs) / len(recs)
        last_dir = recs[-1][2]
        rows.append((sym, avg_str, last_dir))

    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top_n]

    console = Console(record=True, force_terminal=True, width=100)
    table = Table(title="Drift History", expand=True)
    table.box = box.ROUNDED

    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Anomaly", justify="right", style="magenta")
    table.add_column("Dir", justify="center")

    for (sym, strength, direction) in rows:
        arrow = "🔴" if direction == "вверх" else "🟢"
        table.add_row(sym, f"{strength:.3f}", arrow)

    console.print(table)
    result_text = console.export_text()
    return result_text

def generate_model_table_from_csv_no_time(csv_path="model_predictions_log.csv", last_n=200) -> str:
    if not os.path.isfile(csv_path):
        return ""
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return ""
    df.sort_values("timestamp", inplace=True)
    df_tail = df.tail(last_n)
    console = Console(record=True, force_terminal=True, width=100)
    table = Table(title="Model Predictions", expand=True)
    table.box = box.ROUNDED
    # table.add_column("Time", style="dim")
    table.add_column("Symbol", style="cyan")
    table.add_column("Pred", justify="center")
    table.add_column("p(Buy)", justify="right", style="bold green")
    table.add_column("p(Hold)", justify="right")
    table.add_column("p(Sell)", justify="right", style="bold red")
    for _, row in df_tail.iterrows():
        sym = str(row.get("symbol", ""))
        pred = str(row.get("prediction", "NA"))
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0
        p_buy_float = safe_float(row.get("prob_buy", 0.0))
        p_hold_float = safe_float(row.get("prob_hold", 0.0))
        p_sell_float = safe_float(row.get("prob_sell", 0.0))
        p_buy = f"{p_buy_float:.3f}"
        p_hold = f"{p_hold_float:.3f}"
        p_sell = f"{p_sell_float:.3f}"
        table.add_row(sym, pred, p_buy, p_hold, p_sell)
    console.print(table)
    return console.export_text()

async def publish_drift_and_model_tables():
    global telegram_bot, TELEGRAM_CHAT_ID
    if not telegram_bot or not TELEGRAM_CHAT_ID:
        logger.info("[publish_drift_and_model_tables] Telegram не инициализирован => пропуск.")
        return
    if publish_drift_table:
        drift_str = generate_drift_table_from_history(top_n=10)
        if drift_str.strip():
            msg = f"```\n{drift_str}\n```"
            await telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode="Markdown"
            )
        else:
            logger.info("[DRIFT] Таблица пуста => пропуск.")
    if publish_model_table:
        model_str = generate_model_table_from_csv_no_time("model_predictions_log.csv", last_n=10)
        if model_str.strip():
            msg = f"```\n{model_str}\n```"
            await telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode="Markdown"
            )
        else:
            logger.info("[MODEL] Таблица пуста => пропуск.")

async def initialize_telegram_bot():
    global telegram_bot
    try:
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            telegram_bot = Bot(token=TELEGRAM_TOKEN)
            dp = Dispatcher(storage=MemoryStorage())
            dp.include_router(router)
            logger.info("Telegram бот инициализирован. Запуск polling в отдельном task...")
            await dp.start_polling(telegram_bot)
        else:
            logger.warning("Нет TELEGRAM_TOKEN или TELEGRAM_CHAT_ID => Telegram не будет использоваться.")
    except Exception as e:
        logger.exception(f"Ошибка init telegram_bot: {e}")

async def send_initial_telegram_message():
    if telegram_bot and TELEGRAM_CHAT_ID:
        try:
            test_msg = "✅ Бот успешно запущен. Для запуска меню введите команду '/menu'"
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=test_msg)
            logger.info("Сообщение о запуске Telegram отправлено.")
        except Exception as e:
            logger.exception(f"Ошибка при отправке в Telegram: {e}")

async def telegram_message_sender():
    global telegram_bot, TELEGRAM_CHAT_ID
    while True:
        msg = await telegram_message_queue.get()
        if msg is None:
            break
        retry = 0
        max_ret = 5
        delay = 5
        while retry < max_ret:
            try:
                if telegram_bot:
                    async with send_semaphore:
                        await telegram_bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID,
                            text=msg,
                            parse_mode="MarkdownV2",
                            disable_web_page_preview=True,
                            request_timeout=120
                        )
                    logger.info(f"[Telegram] Отправлено: {msg}")
                    break
                else:
                    logger.warning("[Telegram] Бот не инициализирован.")
                    break
            except asyncio.CancelledError:
                # При отмене задачи отправка завершается корректно
                logger.info("Задача отправки сообщений была отменена.")
                break
            except TelegramRetryAfter as e:
                await asyncio.sleep(e.retry_after)
            except TelegramBadRequest as e:
                logger.error(f"BadRequest Telegram: {e}")
                break
            except (TelegramNetworkError, asyncio.TimeoutError, requests.exceptions.RequestException) as e:
                logger.error(f"NetworkError: {e}, retry={retry + 1}")
                retry += 1
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as e:
                logger.exception(f"Ошибка при отправке в Telegram: {e}")
                retry += 1
                await asyncio.sleep(delay)
                delay *= 2
        else:
            logger.error(f"Не отправлено после {max_ret} попыток: {msg}")
        telegram_message_queue.task_done()


def start_ws_monitor():
    ws_local = WebSocket(testnet=False, channel_type="linear")
    while True:
        with open_positions_lock:
            symbols = list(open_positions.keys())
        if not symbols:
            logger.info("[WS] Нет открытых позиций – сплю 10 секунд...")
            time.sleep(10)
            continue
        for symbol in symbols:
            logger.info(f"[WS] Подписываюсь на kline_stream для {symbol} (interval=1, category='linear')")
            # Заменили handle_message => handle_position_update:
            ws_local.kline_stream(interval=1, symbol=symbol, callback=handle_position_update)
        time.sleep(1)

# async def monitor_positions():
#     """
#     Мониторинг позиций через HTTP:
#      1) Периодически получаем все позиции с биржи.
#      2) Синхронизируем локальный словарь open_positions.
#      3) Для каждой позиции считаем % PnL.
#      4) Если убыток <= -TARGET_LOSS_FOR_AVERAGING => открываем усреднение.
#     """
#     while IS_RUNNING:
#         try:
#             await asyncio.sleep(5)  # Проверяем каждые 5 секунд (при желании меняйте задержку)
#
#             # 1) Получаем все актуальные позиции с биржи
#             positions = get_exchange_positions()
#             if not positions:
#                 # Ни одной открытой позиции нет
#                 continue
#
#             # 2) Обновляем локальный словарь open_positions
#             update_open_positions_from_exch_positions(positions)
#
#             # 3) Пробегаемся по позициям и считаем их текущий убыток/прибыль
#             for symbol, pos in positions.items():
#                 side = pos["side"]
#                 entry_price = Decimal(str(pos["avg_price"]))
#                 current_price = get_last_close_price(symbol)
#                 if current_price is None:
#                     logger.debug(f"[HTTP Monitor] Нет текущей цены для {symbol}")
#                     continue
#
#                 # 4) Считаем PnL (%)
#                 if side.lower() == "buy":
#                     ratio = (Decimal(str(current_price)) - entry_price) / entry_price
#                 else:
#                     ratio = (entry_price - Decimal(str(current_price))) / entry_price
#
#                 profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
#                 logger.info(
#                     f"[HTTP Monitor] {symbol}: current={current_price}, "
#                     f"entry={entry_price}, PnL={profit_perc}%"
#                 )
#
#                 # Если убыток ниже заданного порога => открываем усредняющую позицию
#                 if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
#                     logger.info(
#                         f"[HTTP Monitor] {symbol} достиг порога убытка "
#                         f"({profit_perc}% <= -{TARGET_LOSS_FOR_AVERAGING}). Открываю усредняющую позицию."
#                     )
#                     open_averaging_position(symbol)
#
#         except Exception as e:
#             logger.error(f"Ошибка в monitor_positions: {e}")
#             await asyncio.sleep(10)  # При ошибке делаем увеличенную паузу
#             continue

async def monitor_positions():
    """
    Мониторинг позиций через HTTP:
      1) Периодически получает актуальные позиции с биржи.
      2) Синхронизует локальный словарь open_positions.
      3) Для каждой позиции вычисляет % прибыли/убытка.
         - Если убыток <= -TARGET_LOSS_FOR_AVERAGING, открывает усредняющую позицию.
         - Если с учётом плеча (например, 10x) прибыль >= порога (например, 5%), 
           и трейлинг-стоп ещё не установлен, инициирует установку трейлинг‑стопа.
    """
    while IS_RUNNING:
        try:
            await asyncio.sleep(5)  # Проверяем каждые 5 секунд
            positions = get_exchange_positions()
            if not positions:
                continue
            update_open_positions_from_exch_positions(positions)
            for symbol, pos in positions.items():
                side = pos["side"]
                entry_price = Decimal(str(pos["avg_price"]))
                current_price = get_last_close_price(symbol)
                if current_price is None:
                    logger.debug(f"[HTTP Monitor] Нет текущей цены для {symbol}")
                    continue
                if side.lower() == "buy":
                    ratio = (Decimal(str(current_price)) - entry_price) / entry_price
                else:
                    ratio = (entry_price - Decimal(str(current_price))) / entry_price
                profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
                logger.info(f"[HTTP Monitor] {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%")
                # MODIFICATION 1: Обновляем значение profit_perc в open_positions для использования в trailing stop
                with open_positions_lock:
                    if symbol in open_positions:
                        open_positions[symbol]['profit_perc'] = profit_perc
                if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
                    logger.info(f"[HTTP Monitor] {symbol} достиг порога убытка ({profit_perc}% <= -{TARGET_LOSS_FOR_AVERAGING}). Открываю усредняющую позицию.")
                    open_averaging_position(symbol)
                default_leverage = Decimal("10")
                leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))
                threshold_trailing = Decimal("5.0")
                if leveraged_pnl_percent >= threshold_trailing:
                    if not pos.get("trailing_stop_set", False):
                        logger.info(f"[HTTP Monitor] {symbol}: Достигнут уровень для трейлинг-стопа (leveraged PnL = {leveraged_pnl_percent}%). Устанавливаю трейлинг-стоп.")
                        set_trailing_stop(symbol, pos["size"], TRAILING_GAP_PERCENT, side)
        except Exception as e_inner:
            logger.error(f"Ошибка в monitor_positions: {e_inner}")
            await asyncio.sleep(10)
            continue

# Моковая функция, чтобы не падал monitor_positions при вызове check_position_status(...)
async def check_position_status(symbol, pos):
    pass

# def open_position(symbol: str, side: str, volume_usdt: Decimal, reason: str):
#     if is_sleeping():
#         logger.info(f"[open_position] Бот в спящем режиме, открытие {symbol} отменено.")
#         return
#     try:
#         logger.info(f"[open_position] Попытка открытия {side} {symbol}, объем: {volume_usdt} USDT, причина: {reason}")
#         with state_lock:
#             current_total = Decimal("0")
#             with open_positions_lock:
#                 for pos in open_positions.values():
#                     current_total += Decimal(str(pos.get("position_volume", 0)))
#             if current_total + volume_usdt > MAX_TOTAL_VOLUME:
#                 logger.warning(
#                     f"[open_position] Превышен глобальный лимит: текущий объем {current_total} + "
#                     f"новый объем {volume_usdt} > MAX_TOTAL_VOLUME {MAX_TOTAL_VOLUME}"
#                 )
#                 return
#         with open_positions_lock:
#             if symbol in open_positions:
#                 logger.info(f"[open_position] Позиция для {symbol} уже открыта, пропуск.")
#                 return
#         last_price = get_last_close_price(symbol)
#         if not last_price or last_price <= 0:
#             logger.info(f"[open_position] Нет актуальной цены для {symbol}, пропуск.")
#             return
#         qty_dec = volume_usdt / Decimal(str(last_price))
#         qty_float = float(qty_dec)
#         pos_idx = 1 if side.lower() == "buy" else 2
#         trade_id = f"{symbol}_{int(time.time())}"
#         features_dict = {}
#         df_5m = get_historical_data_for_model(symbol, interval="1", limit=1)
#         df_5m = prepare_features_for_model(df_5m)
#         if not df_5m.empty:
#             row_feat = df_5m.iloc[-1]
#             for fc in MODEL_FEATURE_COLS:
#                 features_dict[fc] = row_feat.get(fc, 0)
#         log_model_features_for_trade(trade_id=trade_id, symbol=symbol, side=side, features=features_dict)
#         order_res = place_order(symbol=symbol, side=side, qty=qty_float, order_type="Market", positionIdx=pos_idx)
#         if not order_res or order_res.get("retCode") != 0:
#             logger.info(f"[open_position] Ошибка place_order для {symbol}, пропуск.")
#             return
#         with open_positions_lock:
#             open_positions[symbol] = {
#                 "side": side,
#                 "size": qty_float,
#                 "avg_price": float(last_price),
#                 "position_volume": float(volume_usdt),
#                 "symbol": symbol,
#                 "trailing_stop_set": False,
#                 "trade_id": trade_id,
#                 "open_time": datetime.datetime.utcnow()
#             }
#         with state_lock:
#             state["total_open_volume"] = current_total + volume_usdt
#         row = get_last_row(symbol)
#         log_trade(symbol, row, None, side, f"Opened ({reason})", closed_manually=False)
#         logger.info(f"[open_position] {symbol}: {side} успешно открыта, объем {volume_usdt} USDT")
#     except Exception as e:
#         logger.exception(f"[open_position] Ошибка: {e}")

def open_position(symbol: str, side: str, volume_usdt: Decimal, reason: str):
    if is_sleeping():
        logger.info(f"[open_position] Бот в спящем режиме, открытие {symbol} отменено.")
        return
    try:
        logger.info(f"[open_position] Попытка открытия {side} {symbol}, объем: {volume_usdt} USDT, причина: {reason}")
        
        # Объединяем блокировки для атомарной проверки и обновления
        with state_lock, open_positions_lock:
            current_total = sum(Decimal(str(pos.get("position_volume", 0))) for pos in open_positions.values())
            if current_total + volume_usdt > MAX_TOTAL_VOLUME:
                logger.warning(
                    f"[open_position] Превышен глобальный лимит: текущий объем {current_total} + "
                    f"новый объем {volume_usdt} > MAX_TOTAL_VOLUME {MAX_TOTAL_VOLUME}"
                )
                return
            if symbol in open_positions:
                logger.info(f"[open_position] Позиция для {symbol} уже открыта, пропуск.")
                return

        last_price = get_last_close_price(symbol)
        if not last_price or last_price <= 0:
            logger.info(f"[open_position] Нет актуальной цены для {symbol}, пропуск.")
            return

        qty_dec = volume_usdt / Decimal(str(last_price))
        qty_float = float(qty_dec)
        pos_idx = 1 if side.lower() == "buy" else 2
        trade_id = f"{symbol}_{int(time.time())}"

        # Получение характеристик для модели
        features_dict = {}
        df_5m = get_historical_data_for_model(symbol, interval="1", limit=1)
        df_5m = prepare_features_for_model(df_5m)
        if not df_5m.empty:
            row_feat = df_5m.iloc[-1]
            for fc in MODEL_FEATURE_COLS:
                features_dict[fc] = row_feat.get(fc, 0)

        log_model_features_for_trade(trade_id=trade_id, symbol=symbol, side=side, features=features_dict)

        order_res = place_order(symbol=symbol, side=side, qty=qty_float, order_type="Market", positionIdx=pos_idx)
        if not order_res or order_res.get("retCode") != 0:
            logger.info(f"[open_position] Ошибка place_order для {symbol}, пропуск.")
            return

        # После успешного выставления ордера – обновляем состояние, опять под объединённой блокировкой
        with state_lock, open_positions_lock:
            open_positions[symbol] = {
                "side": side,
                "size": qty_float,
                "avg_price": float(last_price),
                "position_volume": float(volume_usdt),
                "symbol": symbol,
                "trailing_stop_set": False,
                "trade_id": trade_id,
                "open_time": datetime.datetime.utcnow()
            }
            state["total_open_volume"] = current_total + volume_usdt

        row = get_last_row(symbol)
        log_trade(symbol, row, None, side, f"Opened ({reason})", closed_manually=False)
        logger.info(f"[open_position] {symbol}: {side} успешно открыта, объем {volume_usdt} USDT")
    except Exception as e:
        logger.exception(f"[open_position] Ошибка: {e}")


# Моковая функция для демонстрации работы функций drift
# class DriftAnalyzer(threading.Thread):
#     def __init__(self, interval=60):
#         super().__init__(daemon=True)
#         self.interval = interval
#         self.running = True
#         self.last_analysis = {}
#         self._lock = threading.Lock()
#     def run(self):
#         while self.running:
#             try:
#                 symbols = get_selected_symbols()
#                 random.shuffle(symbols)
#                 for sym in symbols:
#                     if not self.running:
#                         break
#                     feature_cols = ["openPrice", "closePrice", "highPrice", "lowPrice"]
#                     new_data = get_historical_data_for_trading(sym, "1", limit=200)
#                     if not new_data.empty:
#                         is_anomaly, strength, direction = monitor_feature_drift_per_symbol(
#                             sym,
#                             new_data,
#                             pd.DataFrame(),
#                             feature_cols,
#                             threshold=0.5
#                         )
#                         with self._lock:
#                             self.last_analysis[sym] = {
#                                 'timestamp': datetime.datetime.utcnow(),
#                                 'is_anomaly': is_anomaly,
#                                 'strength': strength,
#                                 'direction': direction
#                             }
#                 time.sleep(self.interval)
#             except Exception as e:
#                 logger.exception(f"[DriftAnalyzer] Ошибка: {e}")
#                 time.sleep(10)
#     def stop(self):
#         self.running = False
#     def get_latest_analysis(self):
#         with self._lock:
#             return dict(self.last_analysis)

class DriftAnalyzer(threading.Thread):
    def __init__(self, interval=60):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self._lock = threading.Lock()
    
    def run(self):
        while self.running:
            try:
                # Получаем список символов для анализа
                symbols = get_selected_symbols()
                for sym in symbols:
                    new_data = get_historical_data_for_trading(sym, "1", limit=200)
                    # Даже если drift является вспомогательным, мы всё равно обновляем данные
                    is_anomaly, strength, direction = monitor_feature_drift_per_symbol(
                        sym,
                        new_data,
                        pd.DataFrame(),  # Используем пустой DataFrame, если нет отдельного референсного набора
                        ["openPrice", "closePrice", "highPrice", "lowPrice"],
                        threshold=0.5
                    )
                    with self._lock:
                        drift_history[sym] = {"is_anomaly": is_anomaly, "strength": strength, "direction": direction, "timestamp": datetime.datetime.utcnow()}
                time.sleep(self.interval)
            except Exception as e:
                logger.exception(f"[DriftAnalyzer] Ошибка: {e}")
                time.sleep(self.interval)
    
    def stop(self):
        self.running = False

    def get_latest_analysis(self):
        with self._lock:
            # Возвращаем копию drift_history
            return dict(drift_history)

def get_top_anomalies_from_analysis(analysis_data, top_k=10):
    try:
        anomalies = []
        for symbol, data in analysis_data.items():
            if data['is_anomaly']:
                anomalies.append((symbol, data['strength'], data['direction']))
        anomalies.sort(key=lambda x: x[1], reverse=True)
        return anomalies[:top_k]
    except Exception as e:
        logger.exception(f"Ошибка в get_top_anomalies_from_analysis: {e}")
        return []


async def main_coroutine():
    global loop, telegram_bot, telegram_message_queue, current_model, iteration_counter, publish_drift_table, IS_RUNNING
    try:
        IS_RUNNING = True
        publish_drift_table = True
        loop = asyncio.get_running_loop()
        telegram_message_queue = asyncio.Queue()
        logger.info("=== Запуск основного цикла ===")
        with state_lock:
            state["total_open_volume"] = Decimal("0")
        telegram_sender_task = asyncio.create_task(telegram_message_sender())
        tg_task = asyncio.create_task(initialize_telegram_bot())
        await asyncio.sleep(3)
        await send_initial_telegram_message()
        current_model = load_model()
        symbols_all = get_usdt_pairs()
        collect_historical_data(symbols_all, interval="1", limit=200)
        exch_positions = get_exchange_positions()
        update_open_positions_from_exch_positions(exch_positions)
        drift_analyzer = DriftAnalyzer(interval=60)
        drift_analyzer.start()
        monitor_task = None
        if MONITOR_MODE == "ws":
            logger.info("[Main] Режим мониторинга: WebSocket")
            threading.Thread(target=start_ws_monitor, daemon=True).start()
        elif MONITOR_MODE == "http":
            logger.info("[Main] Режим мониторинга: HTTP")
            monitor_task = asyncio.create_task(monitor_positions())
        else:
            logger.warning(f"[Main] Неизвестный режим мониторинга: {MONITOR_MODE}")
        iteration_count = 0
        while IS_RUNNING:
        # Синхронизация глобального состояния после восстановления соединения:
            try:
                exch_positions = get_exchange_positions()
                update_open_positions_from_exch_positions(exch_positions)
            except Exception as e:
                logger.error(f"Ошибка синхронизации позиций после восстановления соединения: {e}")
                await asyncio.sleep(5)
                continue
        try:
                iteration_count += 1
                logger.info(f"[INNER_LOOP] iteration_count={iteration_count} — новый цикл")
                if tg_task.done():
                    exc = tg_task.exception()
                    if exc:
                        logger.exception("Telegram-бот упал с исключением:", exc)
                    else:
                        logger.error("Telegram-бот завершился без исключения")
                    logger.info("Пробуем перезапустить Telegram-бот через 10 секунд...")
                    await asyncio.sleep(10)
                    tg_task = asyncio.create_task(initialize_telegram_bot())
                symbols = get_selected_symbols()
                random.shuffle(symbols)
                logger.info(f"[TRADING] Обработка сигналов в режиме: {OPERATION_MODE}")
                if OPERATION_MODE == "model_only":
                    tasks = [process_symbol_model_only(s) for s in symbols]
                    if tasks:
                        await asyncio.gather(*tasks)
                elif OPERATION_MODE in ["drift_only", "drift_top10"]:
                    latest_analysis = drift_analyzer.get_latest_analysis()
                    top_list = get_top_anomalies_from_analysis(latest_analysis)
                    if OPERATION_MODE == "drift_top10" and top_list:
                        handle_drift_top10(top_list)
                elif OPERATION_MODE == "golden_setup":
                    tasks = []
                    for s in symbols:
                        df_5m = get_historical_data_for_trading(s, "1", limit=20)
                        if not df_5m.empty:
                            action, _ = handle_golden_setup(s, df_5m)
                            if action:
                                open_position(s, action, POSITION_VOLUME, reason="Golden")
                    if tasks:
                        await asyncio.gather(*tasks)
                elif OPERATION_MODE == "super_trend":
                    tasks = [
                        asyncio.to_thread(process_symbol_supertrend_open, s, 
                            interval=SUPER_TREND_TIMEFRAME, length=8, multiplier=3.0) 
                        for s in symbols
                    ]
                    if tasks:
                        await asyncio.gather(*tasks)
                if OPERATION_MODE in ["ST_cross_global", "ST_cross1", "ST_cross2"]:
                    tasks = []
                    for s in symbols:
                        if OPERATION_MODE == "ST_cross_global":
                            tasks.append(asyncio.to_thread(process_symbol_st_cross_global, s, interval=INTERVAL))
                        elif OPERATION_MODE == "ST_cross1":
                            tasks.append(asyncio.to_thread(process_symbol_st_cross1, s, interval=INTERVAL))
                        elif OPERATION_MODE == "ST_cross2":
                            tasks.append(asyncio.to_thread(process_symbol_st_cross2, s, interval=INTERVAL))
                        elif OPERATION_MODE == "ST_cross2_drift":
                            tasks.append(asyncio.to_thread(process_symbol_st_cross2_drift, s, interval=INTERVAL))
                    if tasks:
                        await asyncio.gather(*tasks)
                if check_and_close_active:
                    check_and_set_trailing_stop()
                if iteration_count % 5 == 0:
                    await publish_drift_and_model_tables()
                tasks_log = []
                for s in symbols:
                    tasks_log.append(asyncio.to_thread(log_model_prediction_for_symbol, s))
                if tasks_log:
                    await asyncio.gather(*tasks_log)
                if iteration_count % 20 == 0:
                    logger.info(f"[INNER_LOOP] iteration_count={iteration_count}, вызываем maybe_retrain_model()")
                    await maybe_retrain_model()
                final_expos = get_exchange_positions()
                update_open_positions_from_exch_positions(final_expos)
                await asyncio.to_thread(generate_daily_pnl_report, "trade_log.csv", "daily_pnl_report.csv")
                await asyncio.sleep(60)
        except Exception as e_inner:
            logger.exception(f"Ошибка во внутреннем цикле: {e_inner}")
            await asyncio.sleep(60)
    except Exception as e_outer:
        logger.exception(f"Ошибка во внешнем цикле: {e_outer}")
    finally:
        if not IS_RUNNING:
            logger.info("Бот остановлен командой /stop")
            if telegram_bot and TELEGRAM_CHAT_ID:
                try:
                    await telegram_bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text="✅ Бот успешно остановлен"
                    )
                except Exception as e:
                    logger.error(f"Ошибка отправки сообщения об остановке: {e}")
        else:
            logger.error("Бот остановился не по команде /stop! Возможна ошибка в работе.")
            if telegram_bot and TELEGRAM_CHAT_ID:
                try:
                    await telegram_bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text="⚠️ Внимание! Бот остановился не по команде. Проверьте логи!"
                    )
                except Exception as e:
                    logger.error(f"Ошибка отправки сообщения об остановке: {e}")
        
def main():
    try:
        asyncio.run(main_coroutine())
    except KeyboardInterrupt:
        logger.info("Остановка пользователем.")
    except Exception as e:
        logger.exception(f"Ошибка main: {e}")

if __name__ == "__main__":
    main()