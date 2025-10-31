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
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
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

OPERATION_MODE = "ST_cross"  # Режимы: drift_only, drift_top10, golden_setup, model_only, super_trend, ST_cross
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
    demo=True,
    api_key=BYBIT_API_KEY,
    api_secret=BYBIT_API_SECRET,
    timeout=30,
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

async def monitor_position():
    """Monitor position changes using either WebSocket or HTTP"""
    global ws
    
    if MONITOR_MODE == "ws":
        # Инициализация WebSocket соединения с актуальным API
        ws = WebSocket(
            testnet=TESTNET,
            channel_type="private",
            api_key=API_KEY,
            api_secret=API_SECRET,
        )
        
        # Подписываемся на обновления позиции
        ws.position_stream(
            callback=handle_position_update,
            symbol=SYMBOL
        )
        
        # Держим соединение активным
        while True:
            await asyncio.sleep(1)
            
    else:
        # Существующий HTTP мониторинг
        while True:
            try:
                position = await check_position()
                await asyncio.sleep(MONITOR_INTERVAL)
            except Exception as e:
                logger.error(f"Error in HTTP position monitoring: {e}")
                await asyncio.sleep(1)

def handle_position_update(message):
    """Callback function for position updates"""
    try:
        if "data" in message:
            position_info = message["data"][0]
            size = float(position_info.get("size", 0))
            side = position_info.get("side", "")
            
            # Обновляем глобальные переменные позиции
            global POSITION_SIZE, POSITION_SIDE
            POSITION_SIZE = size
            POSITION_SIDE = side
            
            logger.info(f"Position update - Size: {size}, Side: {side}")
            
    except Exception as e:
        logger.error(f"Error processing position update: {e}")


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

def set_sl_and_tp_from_globals(symbol: str, side: str, entry_price: float, size: float):
    try:
        pos_info = get_position_info(symbol, side)
        if not pos_info:
            logger.error(f"[set_sl_and_tp_from_globals] Нет позиции {symbol}/{side}")
            return
        pos_idx = pos_info.get("positionIdx")
        if not pos_idx:
            logger.error(f"[set_sl_and_tp_from_globals] positionIdx не найден для {symbol}/{side}")
            return
        ep = Decimal(str(entry_price))
        if side.lower() == "buy":
            stop_loss_price = ep * STOP_LOSS_LEVEL_Buy
            take_profit_price = ep * TAKE_PROFIT_LEVEL_Buy
        else:
            stop_loss_price = ep * STOP_LOSS_LEVEL_Sell
            take_profit_price = ep * TAKE_PROFIT_LEVEL_Sell
        resp = session.set_trading_stop(
            category="linear",
            symbol=symbol,
            side=side,
            positionIdx=pos_idx,
            qty=str(size),
            stopLoss=str(stop_loss_price),
            takeProfit=str(take_profit_price),
            timeInForce="GoodTillCancel"
        )
        if resp and resp.get("retCode") == 0:
            logger.info(f"[set_sl_and_tp_from_globals] SL={stop_loss_price}, TP={take_profit_price} для {symbol}")
        else:
            logger.error(f"[set_sl_and_tp_from_globals] Ошибка set_trading_stop: {resp.get('retMsg')}")
    except Exception as e:
        logger.exception(f"[set_sl_and_tp_from_globals] Ошибка: {e}")

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
                message = (f"🔄 *Трейлинг стоп-лосс*\n"
                           f"*Символ:* {symbol_link}\n"
                           f"*Время:* {escape_markdown(formatted_time)}\n"
                           f"*Расстояние:* {escape_markdown(action)}\n"
                           f"*Тип:* Трейлинг")
            else:
                message = (f"🫡🔄 *Сделка*\n"
                           f"*Символ:* {symbol_link}\n"
                           f"*Результат:* {escape_markdown(result)}\n"
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

def get_top_anomalies_in_last_n(n=3, top_k=10):
    results = []
    for sym, recs in drift_history.items():
        tail = recs[-n:]
        if not tail:
            continue
        avg_str = sum(x[1] for x in tail) / len(tail)
        last_dir = tail[-1][2]
        results.append((sym, avg_str, last_dir))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

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

def send_drift_top_to_telegram(top_list):
    if not publish_drift_table or not top_list:
        return
    table_data = []
    for (s, st, d) in top_list:
        arrow = "🟢" if d == "вверх" else "🔴"
        table_data.append([s, f"{st:.2f}", arrow])
    message = "```plaintext\n" + tabulate(table_data, headers=["Symbol", "Strength", "Dir"], tablefmt="grid") + "\n```"
    if telegram_message_queue:
        asyncio.run_coroutine_threadsafe(telegram_message_queue.put(message), loop)


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
            if (price_change <= -golden_params["Sell"]["price_change"] and volume_change >= golden_params["Sell"][
                "volume_change"] and oi_change >= golden_params["Sell"]["oi_change"]):
                action = "Sell"
            elif (price_change >= golden_params["Buy"]["price_change"] and volume_change >= golden_params["Buy"][
                "volume_change"] and oi_change >= golden_params["Buy"]["oi_change"]):
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

async def process_symbol_model_only(symbol):
    async with MODEL_ONLY_SEMAPHORE:
        await asyncio.to_thread(process_symbol_model_only_sync, symbol)

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
        return train_new_model()  # Убедитесь что у вас есть эта функция

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
    Локальный расчёт SuperTrend (ATR-based).
    Использует Kline-данные (highPrice, lowPrice, closePrice) внутри df.
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Вспомогательная функция "продления" значений
        def extend_value(current_value, previous_value):
            """Если current_value = NaN или 0, берём предыдущий."""
            if pd.isna(current_value) or current_value == 0:
                return previous_value
            else:
                return current_value

        # Приводим цены к float, нули превращаем в NaN, тянем вперёд
        for col in ["highPrice", "lowPrice", "closePrice"]:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            numeric_series = numeric_series.replace(0, np.nan)  # 0 => NaN
            numeric_series = numeric_series.fillna(method='ffill')
            df[col] = numeric_series

        # Если в первых строках ещё остались NaN, тянем назад
        df.fillna(method='bfill', inplace=True)

        # -- Классический локальный расчёт --
        df["prev_close"] = df["closePrice"].shift(1)
        df["tr1"] = df["highPrice"] - df["lowPrice"]
        df["tr2"] = (df["highPrice"] - df["prev_close"]).abs()
        df["tr3"] = (df["lowPrice"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        # ATR:
        df["atr"] = df["true_range"].rolling(window=length, min_periods=1).mean()
        
        hl2 = (df["highPrice"] + df["lowPrice"]) / 2
        df["basic_ub"] = hl2 + multiplier * df["atr"]
        df["basic_lb"] = hl2 - multiplier * df["atr"]
        df["final_ub"] = df["basic_ub"].copy()
        df["final_lb"] = df["basic_lb"].copy()
        
        for i in range(1, len(df)):
            # "Перекатываем" final_ub
            if (df.loc[df.index[i], "basic_ub"] < df.loc[df.index[i-1], "final_ub"]) \
               or (df.loc[df.index[i-1], "closePrice"] > df.loc[df.index[i-1], "final_ub"]):
                df.loc[df.index[i], "final_ub"] = df.loc[df.index[i], "basic_ub"]
            else:
                df.loc[df.index[i], "final_ub"] = df.loc[df.index[i-1], "final_ub"]
            
            # "Перекатываем" final_lb
            if (df.loc[df.index[i], "basic_lb"] > df.loc[df.index[i-1], "final_lb"]) \
               or (df.loc[df.index[i-1], "closePrice"] < df.loc[df.index[i-1], "final_lb"]):
                df.loc[df.index[i], "final_lb"] = df.loc[df.index[i], "basic_lb"]
            else:
                df.loc[df.index[i], "final_lb"] = df.loc[df.index[i-1], "final_lb"]

            # Продлеваем, если final_ub или final_lb стали 0/NaN
            df.loc[df.index[i], "final_ub"] = extend_value(
                df.loc[df.index[i], "final_ub"],
                df.loc[df.index[i-1], "final_ub"]
            )
            df.loc[df.index[i], "final_lb"] = extend_value(
                df.loc[df.index[i], "final_lb"],
                df.loc[df.index[i-1], "final_lb"]
            )
        
        # Итоговый supertrend
        df["supertrend"] = df["final_ub"].copy()
        df.loc[df["closePrice"] > df["final_ub"], "supertrend"] = df["final_lb"]

        # Продлеваем supertrend
        for i in range(1, len(df)):
            df.loc[df.index[i], "supertrend"] = extend_value(
                df.loc[df.index[i], "supertrend"],
                df.loc[df.index[i-1], "supertrend"]
            )

        return df

    except Exception as e:
        logger.exception(f"Ошибка в calculate_supertrend_bybit_8_1: {e}")
        return pd.DataFrame()


def calculate_supertrend_bybit_8_1(df: pd.DataFrame, length=3, multiplier=1.0) -> pd.DataFrame:
    """
    Локальный расчёт SuperTrend (ATR-based).
    Использует Kline-данные (highPrice, lowPrice, closePrice) внутри df.
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Вспомогательная функция "продления" значений
        def extend_value(current_value, previous_value):
            """Если current_value = NaN или 0, берём предыдущий."""
            if pd.isna(current_value) or current_value == 0:
                return previous_value
            else:
                return current_value

        # Приводим цены к float, нули превращаем в NaN, тянем вперёд
        for col in ["highPrice", "lowPrice", "closePrice"]:
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            numeric_series = numeric_series.replace(0, np.nan)  # 0 => NaN
            numeric_series = numeric_series.fillna(method='ffill')
            df[col] = numeric_series

        # Если в первых строках ещё остались NaN, тянем назад
        df.fillna(method='bfill', inplace=True)

        # -- Классический локальный расчёт --
        df["prev_close"] = df["closePrice"].shift(1)
        df["tr1"] = df["highPrice"] - df["lowPrice"]
        df["tr2"] = (df["highPrice"] - df["prev_close"]).abs()
        df["tr3"] = (df["lowPrice"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        # ATR:
        df["atr"] = df["true_range"].rolling(window=length, min_periods=1).mean()
        
        hl2 = (df["highPrice"] + df["lowPrice"]) / 2
        df["basic_ub"] = hl2 + multiplier * df["atr"]
        df["basic_lb"] = hl2 - multiplier * df["atr"]
        df["final_ub"] = df["basic_ub"].copy()
        df["final_lb"] = df["basic_lb"].copy()
        
        for i in range(1, len(df)):
            # "Перекатываем" final_ub
            if (df.loc[df.index[i], "basic_ub"] < df.loc[df.index[i-1], "final_ub"]) \
               or (df.loc[df.index[i-1], "closePrice"] > df.loc[df.index[i-1], "final_ub"]):
                df.loc[df.index[i], "final_ub"] = df.loc[df.index[i], "basic_ub"]
            else:
                df.loc[df.index[i], "final_ub"] = df.loc[df.index[i-1], "final_ub"]
            
            # "Перекатываем" final_lb
            if (df.loc[df.index[i], "basic_lb"] > df.loc[df.index[i-1], "final_lb"]) \
               or (df.loc[df.index[i-1], "closePrice"] < df.loc[df.index[i-1], "final_lb"]):
                df.loc[df.index[i], "final_lb"] = df.loc[df.index[i], "basic_lb"]
            else:
                df.loc[df.index[i], "final_lb"] = df.loc[df.index[i-1], "final_lb"]

            # Продлеваем, если final_ub или final_lb стали 0/NaN
            df.loc[df.index[i], "final_ub"] = extend_value(
                df.loc[df.index[i], "final_ub"],
                df.loc[df.index[i-1], "final_ub"]
            )
            df.loc[df.index[i], "final_lb"] = extend_value(
                df.loc[df.index[i], "final_lb"],
                df.loc[df.index[i-1], "final_lb"]
            )
        
        # Итоговый supertrend
        df["supertrend"] = df["final_ub"].copy()
        df.loc[df["closePrice"] > df["final_ub"], "supertrend"] = df["final_lb"]

        # Продлеваем supertrend
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

def process_symbol_st_cross(symbol, interval="1", limit=200):
    logger.info(f"[ST_cross] Начало обработки {symbol}")

    # Проверка, есть ли уже открытая позиция
    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross] {symbol}: уже есть открытая позиция, пропускаем.")
            return

    df = get_historical_data_for_trading(symbol, interval=interval, limit=limit)
    if df.empty or len(df) < 5:
        logger.info(f"[ST_cross] {symbol}: Недостаточно данных, пропуск.")
        return

    df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
    df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)

    if df_fast.empty or df_slow.empty:
        logger.info(f"[ST_cross] {symbol}: Не удалось рассчитать SuperTrend.")
        return

    # Проверка на задержку данных
    try:
        last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
        current_time = pd.Timestamp.utcnow()  # Используем pandas Timestamp
        
        if last_candle_time < current_time - pd.Timedelta(minutes=5):
            logger.warning(f"[ST_cross] {symbol}: Данные устарели! Пропускаем.")
            return
    except Exception as e:
        logger.error(f"[ST_cross] Ошибка проверки времени для {symbol}: {e}")
        return

    df_fast, df_slow = df_fast.align(df_slow, join="inner", axis=0)

    prev_fast = df_fast.iloc[-2]["supertrend"]
    curr_fast = df_fast.iloc[-1]["supertrend"]
    prev_slow = df_slow.iloc[-2]["supertrend"]
    curr_slow = df_slow.iloc[-1]["supertrend"]

    prev_diff = prev_fast - prev_slow
    curr_diff = curr_fast - curr_slow
    last_close = df_fast.iloc[-1]["closePrice"]
    margin = 0.01  # 1%

    # Проверка на первое пересечение
    first_cross_up = prev_diff <= 0 and curr_diff > 0
    first_cross_down = prev_diff >= 0 and curr_diff < 0

    # Условие подтверждения сигнала
    confirmed_buy = first_cross_up and last_close >= curr_fast * (1 + margin)
    confirmed_sell = first_cross_down and last_close <= curr_fast * (1 - margin)

    logger.info(
        f"[ST_cross] {symbol}: prev_fast={prev_fast:.6f}, prev_slow={prev_slow:.6f}, "
        f"curr_fast={curr_fast:.6f}, curr_slow={curr_slow:.6f}, last_close={last_close:.6f}"
    )

    if confirmed_buy:
        logger.info(f"[ST_cross] {symbol}: Подтверждён сигнал BUY")
        open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross")
    elif confirmed_sell:
        logger.info(f"[ST_cross] {symbol}: Подтверждён сигнал SELL")
        open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross")
    else:
        logger.info(f"[ST_cross] {symbol}: Пересечения не подтвердились, сигнал отсутствует")

def analyze_trend(values):
    """Анализирует направление тренда по последним свечам"""
    fast_trend = values["fast"][-1] > values["fast"][0]
    slow_trend = values["slow"][-1] > values["slow"][0]
    price_trend = values["close"][-1] > values["close"][0]
    
    # Объем как подтверждение тренда
    volume_confirms = True
    if values["volume"] is not None:
        volume_confirms = values["volume"][-1] > values["volume"].mean()
    
    if fast_trend and slow_trend and price_trend and volume_confirms:
        return "uptrend"
    elif not fast_trend and not slow_trend and not price_trend:
        return "downtrend"
    return "sideways"

def calculate_cross_signal(prev_fast, curr_fast, prev_slow, curr_slow, last_close, trend):
    """Рассчитывает торговый сигнал на основе пересечения и тренда"""
    try:
        # Расчет разницы
        prev_diff = prev_fast - prev_slow
        curr_diff = curr_fast - curr_slow
        
        # Минимальное расстояние для подтверждения (0.5%)
        min_distance = last_close * Decimal("0.005")
        
        # Проверка пересечения
        cross_up = prev_diff <= 0 and curr_diff > 0 and curr_diff > min_distance
        cross_down = prev_diff >= 0 and curr_diff < 0 and abs(curr_diff) > min_distance
        
        if cross_up and trend == "uptrend" and last_close > curr_fast:
            return {"direction": "Buy", "strength": float(curr_diff)}
        elif cross_down and trend == "downtrend" and last_close < curr_fast:
            return {"direction": "Sell", "strength": float(abs(curr_diff))}
        return None
        
    except Exception as e:
        logger.exception(f"Ошибка в calculate_cross_signal: {e}")
        return None


# ===================== ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ =====================
def escape_markdown(text: str) -> str:
    escape_chars = r"_*\[\]()~`>#+\-={}|.,!\\"
    pattern = re.compile(r"([%s])" % re.escape(escape_chars))
    return pattern.sub(r"\\\1", text)


def set_take_profit(symbol, size, entry_price, side):
    try:
        tp_level = TAKE_PROFIT_LEVEL
        if side.lower() == "buy":
            tp_price = entry_price * (Decimal("1") + tp_level)
        else:
            tp_price = entry_price * (Decimal("1") - tp_level)
        pos_info = get_position_info(symbol, side)
        if not pos_info:
            logger.error(f"[set_take_profit] Нет позиции {symbol}/{side}")
            return
        pos_idx = pos_info.get("positionIdx")
        if not pos_idx:
            return
        resp = session.set_trading_stop(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="TakeProfit",
            qty=str(size),
            takeProfit=str(tp_price),
            timeInForce="GoodTillCancel",
            positionIdx=pos_idx
        )
        if resp and resp.get("retCode") == 0:
            logger.info(f"[set_take_profit] Тейк‑профит установлен для {symbol} по цене {tp_price}")
        else:
            logger.error(f"[set_take_profit] Ошибка установки тейк‑профита для {symbol}: {resp.get('retMsg')}")
    except Exception as e:
        logger.exception(f"[set_take_profit] Ошибка для {symbol}: {e}")


# ===================== ОБНОВЛЕНИЕ open_positions (п.1, 2, 3) =====================
def update_open_positions_from_exch_positions(expos: dict):
    """
    Синхронизирует локальный словарь `open_positions` с биржевыми позициями `expos`.
    - Если символ есть в `open_positions`, но нет в `expos`, значит позиция закрылась => логируем закрытие, удаляем.
    - Если символ есть и там, и там, обновляем только основные поля (side, size, avg_price...), но сохраняем trade_id, trailing_stop_set и т.п.
    - Если символ новый (есть в `expos`, но не было в `open_positions`), добавляем.
    """
    with open_positions_lock, state_lock:
        # 1) Ищем позиции, которые "пропали" на бирже => значит закрыты
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

        # 2) Обновляем / добавляем новые
        for sym, newpos in expos.items():
            if sym in open_positions:
                # Уже была запись => обновим поля, но сохраним trade_id, trailing_stop_set
                open_positions[sym]["side"]             = newpos["side"]
                open_positions[sym]["size"]             = newpos["size"]
                open_positions[sym]["avg_price"]        = newpos["avg_price"]
                open_positions[sym]["position_volume"]  = newpos["position_volume"]
                open_positions[sym]["positionIdx"]      = newpos["positionIdx"]
            else:
                # Новая позиция => добавляем
                open_positions[sym] = {
                    "side": newpos["side"],
                    "size": newpos["size"],
                    "avg_price": newpos["avg_price"],
                    "position_volume": newpos["position_volume"],
                    "symbol": sym,
                    "positionIdx": newpos.get("positionIdx"),
                    # Кастомные поля по умолчанию:
                    "trailing_stop_set": False,
                    "trade_id": None,
                    "open_time": datetime.datetime.utcnow(),
                }

        # пересчитаем общий объём
        total = sum(Decimal(str(p["position_volume"])) for p in open_positions.values())
        state["total_open_volume"] = total


def get_last_row(symbol):
    df = get_historical_data_for_trading(symbol, "1", limit=1)
    if df.empty:
        return None
    return df.iloc[-1]


# ===================== УСТАНОВКА / ПРОВЕРКА ТРЕЙЛИНГ-СТОПА =====================
# def check_and_set_trailing_stop():
#     """Проверяет условия и устанавливает трейлинг-стоп"""
#     if not TRAILING_STOP_ENABLED:
#         return

#     with open_positions_lock:
#         positions_to_check = dict(open_positions)

#     for symbol, pos in positions_to_check.items():
#         try:
#             if pos.get("trailing_stop_set", False):
#                 continue

#             current_price = get_last_close_price(symbol)
#             if not current_price:
#                 continue

#             entry_price = Decimal(str(pos["avg_price"]))
#             side = pos["side"]
#             size = pos["size"]
            
#             # Расчет ROI с учетом плеча 10x
#             if side.lower() == "buy":
#                 roi = ((Decimal(str(current_price)) - entry_price) / entry_price) * Decimal("1000")  # 100% * 10x
#             else:
#                 roi = ((entry_price - Decimal(str(current_price))) / entry_price) * Decimal("1000")  # 100% * 10x

#             # Проверяем достижение 5% ROI
#             if roi >= Decimal("5.0"):  # 5% ROI
#                 logger.info(f"[TRAILING] {symbol} достиг {roi:.2f}% ROI - устанавливаем трейлинг-стоп")
                
#                 # Устанавливаем трейлинг-стоп
#                 try:
#                     resp = session.set_trading_stop(
#                         category="linear",
#                         symbol=symbol,
#                         side=side,
#                         trailingStop=str(TRAILING_GAP_PERCENT),  # конвертируем в проценты
#                         positionIdx=1 if side.lower() == "buy" else 2
#                     )
                    
#                     if resp and resp.get("retCode") == 0:
#                         with open_positions_lock:
#                             if symbol in open_positions:
#                                 open_positions[symbol]["trailing_stop_set"] = True
#                         logger.info(f"[TRAILING] Установлен трейлинг-стоп для {symbol} на {TRAILING_GAP_PERCENT}%")
                        
#                         # Логируем установку трейлинг-стопа
#                         row = get_last_row(symbol)
#                         log_trade(symbol, row, None, f"{TRAILING_GAP_PERCENT}%", "Trailing Stop Set")
#                     else:
#                         logger.error(f"[TRAILING] Ошибка установки трейлинг-стопа для {symbol}: {resp.get('retMsg')}")
                        
#                 except Exception as e:
#                     logger.exception(f"[TRAILING] Ошибка set_trading_stop для {symbol}: {e}")
#             else:
#                 logger.debug(f"[TRAILING] {symbol}: текущий ROI {roi:.2f}% < 5% - пропуск")

#         except Exception as e:
#             logger.exception(f"[TRAILING] Ошибка проверки {symbol}: {e}")

# def set_trailing_stop(symbol, size, trailing_gap_percent, side):
#     try:
#         pos_info = get_position_info(symbol, side)
#         if not pos_info:
#             logger.error(f"[set_trailing_stop] Нет позиции {symbol}/{side}")
#             return
#         pos_idx = pos_info.get("positionIdx")
#         if not pos_idx:
#             return

#         avg_price = Decimal(str(pos_info.get("avgPrice", "0")))
#         if avg_price <= 0:
#             return

#         trailing_distance_abs = (avg_price * trailing_gap_percent).quantize(Decimal("0.0000001"))
#         dynamic_min = max(avg_price * Decimal("0.0000001"), MIN_TRAILING_STOP)
#         if trailing_distance_abs < dynamic_min:
#             logger.info(f"[set_trailing_stop] {symbol}: trailingStop={trailing_distance_abs}< {dynamic_min}, пропуск.")
#             return

#         resp = session.set_trading_stop(
#             category="linear",
#             symbol=symbol,
#             side=side,
#             orderType="TrailingStop",
#             qty=str(size),
#             trailingStop=str(trailing_distance_abs),
#             timeInForce="GoodTillCancel",
#             positionIdx=pos_idx
#         )
#         if resp:
#             rc = resp.get("retCode")
#             if rc == 0:
#                 with open_positions_lock:
#                     if symbol in open_positions:
#                         open_positions[symbol]["trailing_stop_set"] = True
#                 row = get_last_row(symbol)
#                 log_trade(symbol, row, None, f"{trailing_distance_abs}", "Trailing Stop Set", closed_manually=False)
#                 logger.info(f"[set_trailing_stop] OK {symbol}")
#             elif rc == 34040:
#                 logger.info("[set_trailing_stop] not modified, retCode=34040.")
#             else:
#                 logger.error(f"[set_trailing_stop] Ошибка: {resp.get('retMsg')}")
#     except Exception as e:
#         logger.exception(f"[set_trailing_stop] {symbol}: {e}")

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
            if leveraged_pnl_percent >= threshold_roi:
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
                row = get_last_row(symbol)
                log_trade(symbol, row, None, f"{trailing_distance_abs}", "Trailing Stop Set", closed_manually=False)
                logger.info(f"[set_trailing_stop] OK {symbol}")
            elif rc == 34040:
                logger.info("[set_trailing_stop] not modified, retCode=34040.")
            else:
                logger.error(f"[set_trailing_stop] Ошибка: {resp.get('retMsg')}")
    except Exception as e:
        logger.exception(f"[set_trailing_stop] {symbol}: {e}")


# ===================== ПРИМЕР НЕИСПОЛЬЗУЕМЫХ ЛОГИК С ЗАКРЫТИЕМ ПО ПРОФИТУ =====================
def check_and_close_profitable_positions():
    """
    Не вызывается в коде прямо сейчас, но оставлено, как вы просили (п.5 не трогаем).
    """
    try:
        with open_positions_lock:
            positions_copy = dict(open_positions)
        to_close = []
        for sym, pos in positions_copy.items():
            side = pos["side"]
            ep = Decimal(str(pos["avg_price"]))
            current = get_last_close_price(sym)
            if current is None:
                continue
            cp = Decimal(str(current))
            if side.lower() == "buy":
                ratio = (cp - ep) / ep
            else:
                ratio = (ep - cp) / ep
            profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.008"))
            logger.info(f"[ProfitCheck] {sym}: profit%={profit_perc}")
            if profit_perc >= PROFIT_LEVEL:
                to_close.append(sym)
        for sym in to_close:
            with open_positions_lock:
                if sym not in open_positions:
                    continue
                pos = open_positions[sym]
                side = pos["side"]
                size = pos["size"]
                volume = Decimal(str(pos["position_volume"]))
                trade_id = pos.get("trade_id", None)
            close_side = "Sell" if side.lower() == "buy" else "Buy"
            posIdx = 1 if side.lower() == "buy" else 2
            res = place_order(sym, close_side, size, "Market", "GoodTillCancel", True, positionIdx=posIdx)
            if res and res.get("retCode") == 0:
                close_price = get_last_close_price(sym)
                pnl = Decimal("0")
                if close_price:
                    cp = Decimal(str(close_price))
                    ep = Decimal(str(pos["avg_price"]))
                    if side.lower() == "buy":
                        pnl = (cp - ep) / ep * Decimal(str(pos["position_volume"]))
                    else:
                        pnl = (ep - cp) / ep * Decimal(str(pos["position_volume"]))
                if trade_id:
                    update_trade_outcome(trade_id, float(pnl))
                log_trade(sym, get_last_row(sym), None, close_side, "Closed", closed_manually=True)
                with state_lock:
                    state["total_open_volume"] -= volume
                    if state["total_open_volume"] < Decimal("0"):
                        state["total_open_volume"] = Decimal("0")
                with open_positions_lock:
                    del open_positions[sym]
    except Exception as e:
        logger.exception(f"Ошибка check_and_close_profitable_positions: {e}")


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
    Объём усредняющей позиции равен объёму базовой позиции.
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
                    f"[Averaging] Превышен лимит усреднения: {averaging_total_volume} + {base_volume_usdt} > {MAX_AVERAGING_VOLUME}")
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
                f"[Averaging] Усредняющая позиция для {symbol} открыта на объём {base_volume_usdt}. Текущий усредняющий объём: {averaging_total_volume}")
        else:
            logger.error(f"[Averaging] Ошибка открытия усредняющей позиции для {symbol}: {order_result}")
    except Exception as e:
        logger.exception(f"[Averaging] Ошибка в open_averaging_position для {symbol}: {e}")


# ===================== ФУНКЦИЯ МОНИТОРИНГА ЧЕРЕЗ HTTP =====================
def http_monitor_positions():
    """
    Мониторинг открытых позиций через HTTP-запросы.
    Для каждого символа из open_positions:
      - Получаем текущую цену (HTTP)
      - Считаем % прибыли/убытка
      - Если убыток <= -TARGET_LOSS_FOR_AVERAGING => усреднение.
    """
    with open_positions_lock:
        symbols = list(open_positions.keys())
    for symbol in symbols:
        current_price = get_last_close_price(symbol)
        if current_price is None:
            logger.info(f"[HTTP Monitor] Нет текущей цены для {symbol}")
            continue
        with open_positions_lock:
            pos = open_positions[symbol]
        side = pos["side"]
        entry_price = Decimal(str(pos["avg_price"]))
        if side.lower() == "buy":
            ratio = (Decimal(str(current_price)) - entry_price) / entry_price
        else:
            ratio = (entry_price - Decimal(str(current_price))) / entry_price
        profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
        logger.info(f"[HTTP Monitor] {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%")

        if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
            logger.info(
                f"[HTTP Monitor] {symbol} достиг порога убытка ({profit_perc}% <= -{TARGET_LOSS_FOR_AVERAGING}). Открываю усредняющую позицию.")
            open_averaging_position(symbol)

async def monitor_positions():
    """Мониторинг позиций через HTTP"""
    while IS_RUNNING:
        try:
            await asyncio.sleep(5)  # Проверка каждые 5 секунд
            
            # Получаем текущие позиции
            positions = get_exchange_positions()
            if not positions:
                continue
                
            # Обновляем локальное состояние
            update_open_positions_from_exch_positions(positions)
            
            # Проверяем каждую позицию
            for symbol, pos in positions.items():
                try:
                    await check_position_status(symbol, pos)
                except Exception as pos_e:
                    logger.error(f"Ошибка проверки позиции {symbol}: {pos_e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Ошибка в monitor_positions: {e}")
            await asyncio.sleep(10)  # Увеличенная пауза при ошибке
            continue  # Продолжаем работу после ошибки


# ===================== ФУНКЦИИ TELEGRAM И КОМАНД =====================
class FSMSettings(StatesGroup):
    drift_table = State()
    model_table = State()

@router.message(Command(commands=["menu"]))
async def main_menu_cmd(message: Message):
    """Отображает главное меню с разделами."""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📈 Торговля", callback_data="menu_trading")],
            [InlineKeyboardButton(text="🤖 Бот", callback_data="menu_bot")],
            [InlineKeyboardButton(text="ℹ️ Информация", callback_data="menu_info")],
        ]
    )
    await message.answer("Выберите раздел:", reply_markup=keyboard)

@router.callback_query(lambda c: c.data == "menu_trading")
async def menu_trading_cb(query: CallbackQuery):
    """Отображает меню торговых команд."""
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
    """Отображает меню команд управления ботом."""
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
    """Отображает меню информационных команд."""
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
    """Возвращает в главное меню."""
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
    """
    Отправляет пользователю инлайн-клавиатуру (InlineKeyboard) с командами:
    /status, /togglesilence, /silencestatus, /setmaxvolume, /setposvolume, /setsttf
    """
    # Формируем кнопки (InlineKeyboardButton):
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

    # Отправляем сообщение с готовой разметкой (inline-кнопками)
    await message.answer("Выберите действие:", reply_markup=markup)


@router.callback_query(lambda c: c.data and c.data.startswith("cmd_"))
async def process_inline_commands(query: CallbackQuery):
    """
    Обрабатывает нажатия на инлайн-кнопки (callback_data).
    В зависимости от callback_data вызываем нужную логику.
    """
    data = query.data

    if data == "cmd_status":
        # Здесь можно либо вызвать вашу функцию /status,
        # либо напрямую прописать логику, например:
        # await status_cmd(...)
        # Но чаще просто отвечаем в чат:
        await query.message.answer("Вызван STATUS — ваша логика здесь.")

    elif data == "cmd_togglesilence":
        await query.message.answer("Вызван TOGGLE SILENCE — здесь ваша логика.")
        # Можно напрямую вызывать toggle_quiet_period() и отправить результат

    elif data == "cmd_silencestatus":
        await query.message.answer("Вызван SILENCE STATUS — логика /silencestatus.")

    elif data == "cmd_setmaxvolume":
        # Поскольку /setmaxvolume обычно требует параметр (например, /setmaxvolume 500),
        # тут вы можете либо:
        # 1) Попросить пользователя ввести число
        # 2) Показать ещё одну инлайн-клавиатуру со стандартными значениями
        # 3) Или просто написать, что "Введи /setmaxvolume 500"
        await query.message.answer("Вызван SET MAX VOLUME. Введите, например: /setmaxvolume 500")

    elif data == "cmd_setposvolume":
        await query.message.answer("Вызван SET POS VOLUME. Пример: /setposvolume 50")

    elif data == "cmd_setsttf":
        await query.message.answer("Вызван SET ST TF (SuperTrend TF). Пример: /setsttf 15")

    # Обязательно отвечаем на сам callback, чтобы убрать часики "thinking…"
    await query.answer()


@router.message(Command(commands=["sleep"]))
async def sleep_cmd(message: Message):
    """Команда /sleep включает спящий режим."""
    status = toggle_sleep_mode()
    await message.reply(f"Спящий режим: {status}")

@router.message(Command(commands=["wake"]))
async def wake_cmd(message: Message):
    """Команда /wake отключает спящий режим."""
    status = toggle_sleep_mode()
    await message.reply(f"Спящий режим: {status}")

def is_sleeping():
    """Функция для проверки спящего режима перед открытием позиций."""
    return IS_SLEEPING_MODE


@router.message(Command(commands=["getpair"]))
async def get_pair_data_cmd(message: Message):
    """Команда /getpair BTC или /getpair BTCUSDT для получения предсказания модели или дрифта."""
    parts = message.text.strip().split()
    if len(parts) < 2:
        await message.reply("Использование: /getpair BTC (или BTCUSDT)")
        return

    symbol = parts[1].upper().replace("USDT", "") + "USDT"

    # Проверка в таблице предсказаний модели
    try:
        df = pd.read_csv("model_predictions_log.csv")
        df = df[df["symbol"] == symbol].sort_values("timestamp", ascending=False)
        if not df.empty:
            last_pred = df.iloc[0]["prediction"]
            pred_map = {2: "покупка", 1: "холд", 0: "продажа"}
            pred_str = pred_map.get(last_pred, "нет данных")
            await message.reply(f"📊 *Результат модели для {symbol}*: {pred_str}", parse_mode="Markdown")
            return
    except Exception as e:
        logger.exception(f"[get_pair_data_cmd] Ошибка при загрузке предсказаний модели: {e}")

    # Проверка в истории дрифта
    if symbol in drift_history:
        last_entry = drift_history[symbol][-1]
        direction = "вверх (покупка)" if last_entry[2] == "вверх" else "вниз (продажа)"
        await message.reply(f"📉 *Дрифт-аналитика для {symbol}*: {direction}", parse_mode="Markdown")
        return

    await message.reply(f"⚠️ Нет данных по {symbol}.")


@router.message(Command(commands=["mode"]))
async def change_or_get_mode_cmd(message: Message):
    """Команда /mode для просмотра и изменения режима работы"""
    global OPERATION_MODE

    available_modes = {
        "drift_only": "🌊 Drift Only",
        "drift_top10": "📊 Drift TOP-10",
        "golden_setup": "✨ Golden Setup",
        "super_trend": "📈 SuperTrend",
        "ST_cross": "🔄 ST Cross",
        "model_only": "🤖 Model Only"
    }

    # Создаем клавиатуру с кнопками
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=label, callback_data=f"set_mode_{mode}")]
        for mode, label in available_modes.items()
    ])

    # Формируем сообщение о текущем режиме
    current_mode_label = available_modes.get(OPERATION_MODE, OPERATION_MODE)
    message_text = (
        f"*Текущий режим*: {current_mode_label}\n\n"
        f"Выберите новый режим работы:"
    )

    await message.answer(message_text, reply_markup=keyboard, parse_mode="Markdown")

@router.callback_query(lambda c: c.data and c.data.startswith("set_mode_"))
async def process_mode_change(callback: CallbackQuery):
    """Обработчик нажатия кнопок выбора режима"""
    global OPERATION_MODE
    
    new_mode = callback.data.replace("set_mode_", "")
    old_mode = OPERATION_MODE
    OPERATION_MODE = new_mode

    mode_emojis = {
        "drift_only": "🌊",
        "drift_top10": "📊",
        "golden_setup": "✨",
        "super_trend": "📈",
        "ST_cross": "🔄",
        "model_only": "🤖"
    }

    old_emoji = mode_emojis.get(old_mode, "⚪️")
    new_emoji = mode_emojis.get(new_mode, "⚪️")

    response_text = (
        f"Режим работы изменен:\n"
        f"{old_emoji} {old_mode} ➜ {new_emoji} {new_mode}"
    )

    await callback.message.edit_text(
        response_text,
        parse_mode="Markdown"
    )
    await callback.answer(f"Режим изменен на {new_mode}")

@router.message(Command(commands=["status"]))
async def status_cmd(message: Message):
    with open_positions_lock:
        if not open_positions:
            await message.reply("Нет позиций.")
            return

        lines = []
        total_pnl_usdt = Decimal("0")
        total_invested = Decimal("0")

        for sym, pos in open_positions.items():
            side_str = pos["side"]
            entry_price = Decimal(str(pos["avg_price"]))
            volume_usdt = Decimal(str(pos["position_volume"]))
            current_price = get_last_close_price(sym)

            if current_price is None:
                # не смогли получить цену, пропускаем или пишем "цену не удалось получить"
                lines.append(f"{sym} {side_str}: нет текущей цены.")
                continue

            cp = Decimal(str(current_price))
            # Расчёт PnL
            if side_str.lower() == "buy":
                ratio = (cp - entry_price) / entry_price
            else:  # side = Sell
                ratio = (entry_price - cp) / entry_price

            pnl_usdt = ratio * volume_usdt
            pnl_percent = ratio * Decimal("100")

            total_pnl_usdt += pnl_usdt
            total_invested += volume_usdt

            # Формируем строку для конкретной позиции
            lines.append(
                f"{sym} {side_str}: "
                f"PNL = {pnl_usdt:.2f} USDT "
                f"({pnl_percent:.2f}%)"
            )

        # Итого
        lines.append("—" * 30)
        if total_invested > 0:
            total_pnl_percent = (total_pnl_usdt / total_invested) * Decimal("100")
            lines.append(
                f"Итоговый PnL по всем позициям: "
                f"{total_pnl_usdt:.2f} USDT "
                f"({total_pnl_percent:.2f}%)"
            )
        else:
            lines.append("Итоговый PnL: 0 (нет позиций с объёмом)")

        # Отправляем всё одним сообщением
        await message.reply("\n".join(lines))

@router.message(Command(commands=["togglesilence"]))
async def toggle_silence_cmd(message: Message):
    st = toggle_quiet_period()
    await message.reply(f"Тихий период: {st}")

@router.message(Command(commands=["silencestatus"]))
async def silence_status_cmd(message: Message):
    st = "включён" if QUIET_PERIOD_ENABLED else "выключен"
    await message.reply(f"Тихий период: {st}")

@router.message(Command(commands=["setmaxvolume"]))
async def set_max_volume_cmd(message: Message):
    global MAX_TOTAL_VOLUME
    parts = message.text.strip().split()
    if len(parts) < 2:
        await message.reply("Формат: /setmaxvolume 500")
        return
    try:
        new_val = Decimal(parts[1])
        if new_val <= 0:
            raise ValueError
        MAX_TOTAL_VOLUME = new_val
        await message.reply(f"MAX_TOTAL_VOLUME => {MAX_TOTAL_VOLUME}")
    except:
        await message.reply("Некорректное значение. /setmaxvolume 500")

@router.message(Command(commands=["setposvolume"]))
async def set_position_volume_cmd(message: Message):
    global POSITION_VOLUME
    parts = message.text.strip().split()
    if len(parts) < 2:
        await message.reply("Формат: /setposvolume 50")
        return
    try:
        new_val = Decimal(parts[1])
        if new_val <= 0:
            raise ValueError
        POSITION_VOLUME = new_val
        await message.reply(f"POSITION_VOLUME => {POSITION_VOLUME}")
    except:
        await message.reply("Некорректное значение. /setposvolume 50")


@router.callback_query(lambda c: c.data and c.data.startswith("toggle_publish_"))
async def toggle_publish_cb(query: types.CallbackQuery):
    global publish_drift_table, publish_model_table
    data = query.data
    if data == "toggle_publish_drift":
        publish_drift_table = not publish_drift_table
        st = "включена" if publish_drift_table else "выключена"
        await query.answer(f"Публикация Drift‑таблицы {st}.")
    elif data == "toggle_publish_model":
        publish_model_table = not publish_model_table
        st = "включена" if publish_model_table else "выключена"
        await query.answer(f"Публикация модельной таблицы {st}.")
    else:
        await query.answer("Неизвестная команда.")

@router.callback_query(lambda c: c.data and c.data == "toggle_model_top")
async def toggle_model_top_cb(query: types.CallbackQuery):
    global MODEL_TOP_ENABLED
    MODEL_TOP_ENABLED = not MODEL_TOP_ENABLED
    st = "включена" if MODEL_TOP_ENABLED else "выключена"
    await query.answer(f"Model TOP => {st}.")


@router.message(Command(commands=["setsttf"]))
async def set_supertrend_tf_cmd(message: Message):
    global SUPER_TREND_TIMEFRAME
    parts = message.text.strip().split()
    if len(parts) < 2:
        await message.reply("Формат: /setsttf 15")
        return
    try:
        new_tf = parts[1]
        SUPER_TREND_TIMEFRAME = new_tf
        await message.reply(f"SUPER_TREND_TIMEFRAME => {SUPER_TREND_TIMEFRAME}")
    except:
        await message.reply("Некорректное значение. /setsttf 15")


# ===================== ПУБЛИКАЦИЯ DRIFT / MODEL ТАБЛИЦ =====================
def generate_drift_table_from_history(top_n=15) -> str:
    """
    Генерирует «красивую» табличку дрифта (Drift) с помощью Rich.
    Возвращает её в виде готовой текстовой строки, которую можно
    отправить в Telegram, обернув в ``` для Markdown.
    """
    if not drift_history:
        return ""

    # Формируем список (symbol, avg_strength, last_direction)
    rows = []
    for sym, recs in drift_history.items():
        if not recs:
            continue
        # Средняя «сила аномалии» по последним n записям
        avg_str = sum(x[1] for x in recs) / len(recs)
        last_dir = recs[-1][2]  # "вверх" или "вниз"
        rows.append((sym, avg_str, last_dir))

    # Сортируем по убыванию силы аномалий и берём top_n
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top_n]

    # Создаём «виртуальную консоль» Rich
    console = Console(record=True, force_terminal=True, width=100)

    # Настраиваем саму таблицу
    table = Table(title="Drift History", expand=True)
    table.box = box.ROUNDED  # или box.SIMPLE, box.HEAVY, box.ASCII

    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Anomaly", justify="right", style="magenta")
    table.add_column("Dir", justify="center")

    # Заполняем строки
    for (sym, strength, direction) in rows:
        arrow = "🟢" if direction == "вверх" else "🔴"
        table.add_row(sym, f"{strength:.3f}", arrow)

    # Рисуем таблицу в буфер
    console.print(table)
    # Получаем итоговую текстовую «рамку»
    result_text = console.export_text()
    return result_text


def generate_model_table_from_csv_no_time(csv_path="model_predictions_log.csv", last_n=200) -> str:
    """
    Считывает последние N записей из CSV-файла с логом модельных предсказаний
    и формирует «красивую» ASCII/Unicode-таблицу через Rich.
    Возвращает готовую строку для отправки в Telegram.
    """
    if not os.path.isfile(csv_path):
        return ""

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return ""

    # Упорядочиваем по времени и берём хвост
    df.sort_values("timestamp", inplace=True)
    df_tail = df.tail(last_n)

    console = Console(record=True, force_terminal=True, width=100)
    table = Table(title="Model Predictions", expand=True)
    table.box = box.ROUNDED  # Можно заменить на другой стиль рамок

    # Колонки
    #table.add_column("Time", style="dim")
    table.add_column("Symbol", style="cyan")
    table.add_column("Pred", justify="center")
    table.add_column("p(Buy)", justify="right", style="bold green")
    table.add_column("p(Hold)", justify="right")
    table.add_column("p(Sell)", justify="right", style="bold red")

    for _, row in df_tail.iterrows():
        # Достаем значения
        #timestamp = str(row.get("timestamp", ""))
        sym       = str(row.get("symbol", ""))
        pred      = str(row.get("prediction", "NA"))

        # Безопасно преобразуем к float
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        p_buy_float  = safe_float(row.get("prob_buy", 0.0))
        p_hold_float = safe_float(row.get("prob_hold", 0.0))
        p_sell_float = safe_float(row.get("prob_sell", 0.0))

        # Форматируем с .3f
        p_buy  = f"{p_buy_float:.3f}"
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


# ===================== ИНИЦИАЛИЗАЦИЯ TELEGRAM (с реальным запуском polling) =====================
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
                        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="MarkdownV2",
                                                        disable_web_page_preview=True, request_timeout=120)
                    logger.info(f"[Telegram] Отправлено: {msg}")
                    break
                else:
                    logger.warning("[Telegram] Бот не инициализирован.")
                    break
            except TelegramRetryAfter as e:
                await asyncio.sleep(e.retry_after)
            except TelegramBadRequest as e:
                logger.error(f"BadRequest Telegram: {e}")
                break
            except (TelegramNetworkError, asyncio.TimeoutError) as e:
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


# ===================== WEB SOCKET MONITORING (BYBIT V5) =====================
def handle_message(message):
    """
    Обработчик входящих сообщений от WebSocket.
    Если по базовой позиции достигается целевой уровень убытка,
    вызывается открытие усредняющей позиции.
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

            if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
                logger.info(
                    f"[WS] {symbol} достиг порога убытка ({profit_perc}% <= -{TARGET_LOSS_FOR_AVERAGING}). Открываю усредняющую позицию.")
                open_averaging_position(symbol)
    else:
        logger.debug(f"[WS] Получено сообщение: {message}")

def start_ws_monitor():
    ws = WebSocket(testnet=False, channel_type="linear")
    while True:
        with open_positions_lock:
            symbols = list(open_positions.keys())
        if not symbols:
            logger.info("[WS] Нет открытых позиций – сплю 10 секунд...")
            time.sleep(10)
            continue
        for symbol in symbols:
            logger.info(f"[WS] Подписываюсь на kline_stream для {symbol} (interval=1, category='linear')")
            ws.kline_stream(interval=1, symbol=symbol, callback=handle_message)
        time.sleep(1)


# ===================== ДОПОЛНИТЕЛЬНАЯ ФУНКЦИЯ =====================
def open_position(symbol: str, side: str, volume_usdt: Decimal, reason: str):
    if is_sleeping():
        logger.info(f"[open_position] Бот в спящем режиме, открытие {symbol} отменено.")
        return
    try:
        logger.info(f"[open_position] Попытка открытия {side} {symbol}, объем: {volume_usdt} USDT, причина: {reason}")

        # Строгая проверка глобального лимита
        with state_lock:
            current_total = Decimal("0")
            # Подсчитываем текущий объем всех открытых позиций
            with open_positions_lock:
                for pos in open_positions.values():
                    current_total += Decimal(str(pos.get("position_volume", 0)))
            
            # Проверяем, не превысит ли новая позиция лимит
            if current_total + volume_usdt > MAX_TOTAL_VOLUME:
                logger.warning(
                    f"[open_position] Превышен глобальный лимит: текущий объем {current_total} + "
                    f"новый объем {volume_usdt} > MAX_TOTAL_VOLUME {MAX_TOTAL_VOLUME}"
                )
                return

        # Проверка на существующую позицию
        with open_positions_lock:
            if symbol in open_positions:
                logger.info(f"[open_position] Позиция для {symbol} уже открыта, пропуск.")
                return

        # Получение последней цены
        last_price = get_last_close_price(symbol)
        if not last_price or last_price <= 0:
            logger.info(f"[open_position] Нет актуальной цены для {symbol}, пропуск.")
            return

        qty_dec = volume_usdt / Decimal(str(last_price))
        qty_float = float(qty_dec)
        pos_idx = 1 if side.lower() == "buy" else 2
        trade_id = f"{symbol}_{int(time.time())}"

        # Логирование признаков модели
        features_dict = {}
        df_5m = get_historical_data_for_model(symbol, interval="1", limit=1)
        df_5m = prepare_features_for_model(df_5m)
        if not df_5m.empty:
            row_feat = df_5m.iloc[-1]
            for fc in MODEL_FEATURE_COLS:
                features_dict[fc] = row_feat.get(fc, 0)

        log_model_features_for_trade(trade_id=trade_id, symbol=symbol, side=side, features=features_dict)

        # Размещение ордера
        order_res = place_order(symbol=symbol, side=side, qty=qty_float, order_type="Market", positionIdx=pos_idx)
        if not order_res or order_res.get("retCode") != 0:
            logger.info(f"[open_position] Ошибка place_order для {symbol}, пропуск.")
            return

        # Добавление позиции в список
        with open_positions_lock:
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

        # Обновление глобального лимита
        with state_lock:
            state["total_open_volume"] = current_total + volume_usdt

        row = get_last_row(symbol)
        log_trade(symbol, row, None, side, f"Opened ({reason})", closed_manually=False)

        logger.info(f"[open_position] {symbol}: {side} успешно открыта, объем {volume_usdt} USDT")

    except Exception as e:
        logger.exception(f"[open_position] Ошибка: {e}")


# ===================== OSNOVNAYA/MAIN ЛОГИКА =====================
def process_symbol(symbol):
    if is_silence_period():
        logger.info(f"[{symbol}] Quiet period => skip trades.")
        return

    try:
        # Получаем данные
        feature_cols = ["openPrice", "closePrice", "highPrice", "lowPrice"]
        new_data = get_historical_data_for_trading(symbol, "1", limit=200)
        
        if new_data.empty:
            logger.info(f"[{symbol}] Нет данных для анализа")
            return

        # Анализ дрейфа
        is_anomaly, strength, direction = monitor_feature_drift_per_symbol(
            symbol,
            new_data,
            pd.DataFrame(),  # Пустой ref_data - будет использована первая половина new_data
            feature_cols,
            drift_csv="feature_drift.csv",
            threshold=0.5
        )

        # Логируем результат анализа
        logger.info(f"[{symbol}] Drift analysis: anomaly={is_anomaly}, strength={strength:.3f}, direction={direction}")

        # Обработка разных режимов
        if OPERATION_MODE == "drift_only":
            if is_anomaly:
                side = "Buy" if direction == "вверх" else "Sell"
                open_position(symbol, side, POSITION_VOLUME, reason="Drift")
                logger.info(f"[{symbol}] Opening {side} position based on drift signal")

        elif OPERATION_MODE == "drift_top10":
            pass  # Обработка происходит в handle_drift_top10

        elif OPERATION_MODE == "golden_setup":
            df_5m = get_historical_data_for_trading(symbol, "1", limit=20)
            if df_5m.empty:
                return
            action, _ = handle_golden_setup(symbol, df_5m)
            if action:
                open_position(symbol, action, POSITION_VOLUME, reason="Golden")

        elif OPERATION_MODE == "super_trend":
            process_symbol_supertrend_open(symbol, interval=SUPER_TREND_TIMEFRAME, length=8, multiplier=3.0)

        elif OPERATION_MODE == "ST_cross":
            process_symbol_st_cross(symbol, interval=INTERVAL)

        elif OPERATION_MODE == "model_only":
            pass  # Обработка происходит через process_symbol_model_only

    except Exception as e:
        logger.exception(f"Error processing {symbol}: {e}")

def check_btc_correlation(symbol, df):
    """Проверяет корреляцию с BTC"""
    try:
        if symbol == "BTCUSDT":
            return True
            
        btc_df = get_historical_data_for_trading("BTCUSDT", interval="1", limit=100)
        if btc_df.empty or df.empty:
            return True
            
        # Выравниваем временные ряды
        df_aligned, btc_aligned = df.align(btc_df, join="inner", axis=0)
        
        # Рассчитываем корреляцию
        correlation = df_aligned["closePrice"].corr(btc_aligned["closePrice"])
        
        # Если корреляция слишком высокая, пропускаем сигнал
        if abs(correlation) > 0.95:
            logger.info(f"{symbol}: Высокая корреляция с BTC ({correlation:.2f})")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Ошибка проверки корреляции для {symbol}: {e}")
        return True

def retry_on_error(func):
    """Декоратор для повторных попыток при ошибках"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Превышено количество попыток для {func.__name__}: {e}")
                    raise
                logger.warning(f"Попытка {attempt + 1} не удалась: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
    return wrapper

@router.message(Command("stop"))
async def stop_command(message: Message):
    """Обработчик команды /stop"""
    if str(message.chat.id) != TELEGRAM_CHAT_ID:
        return
    global IS_RUNNING
    IS_RUNNING = False
    await message.answer("🛑 Бот останавливается...")
    logger.info("Получена команда /stop - инициируем остановку бота")

def sync_position_volumes():
    """Синхронизирует объемы позиций с биржей и проверяет общий объем"""
    try:
        exchange_positions = get_exchange_positions()
        total_volume = Decimal("0")
        
        with open_positions_lock:
            # Очищаем локальные позиции
            open_positions.clear()
            
            # Обновляем из биржевых данных
            for symbol, pos_data in exchange_positions.items():
                volume = Decimal(str(pos_data.get("position_volume", 0)))
                total_volume += volume
                open_positions[symbol] = pos_data
        
        with state_lock:
            state["total_open_volume"] = total_volume
            
        if total_volume > MAX_TOTAL_VOLUME:
            logger.warning(
                f"[sync_position_volumes] Внимание! Текущий объем {total_volume} "
                f"превышает MAX_TOTAL_VOLUME {MAX_TOTAL_VOLUME}"
            )
            
    except Exception as e:
        logger.exception(f"[sync_position_volumes] Ошибка: {e}")

    
# ===================== MAIN COROUTINE =====================
async def main_coroutine():
    global loop, telegram_bot, telegram_message_queue, current_model, iteration_counter, publish_drift_table, IS_RUNNING
    
    try:
        # Инициализация
        IS_RUNNING = True
        publish_drift_table = True
        
        loop = asyncio.get_running_loop()
        telegram_message_queue = asyncio.Queue()
        
        logger.info("=== Запуск основного цикла ===")
        
        with state_lock:
            state["total_open_volume"] = Decimal("0")
        
        # Запускаем корутину-отправщик сообщений в Телеграм
        telegram_sender_task = asyncio.create_task(telegram_message_sender())
        
        # Запускаем Telegram бота (aiogram) в отдельной корутине
        tg_task = asyncio.create_task(initialize_telegram_bot())
        
        await asyncio.sleep(3)  # небольшая задержка, чтобы бот успел стартануть
        await send_initial_telegram_message()
        
        # Загружаем или обучаем модель
        current_model = load_model()
        
        # Опционально можно вызвать collect_historical_data
        symbols_all = get_usdt_pairs()
        collect_historical_data(symbols_all, interval="1", limit=200)
        
        # Синхронизируем позиции с биржей
        exch_positions = get_exchange_positions()
        update_open_positions_from_exch_positions(exch_positions)
        
        # Инициализация анализатора дрифта в фоновом потоке
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
        publish_cycle = 3
        
        while IS_RUNNING:
            try:
                iteration_count += 1
                logger.info(f"[INNER_LOOP] iteration_count={iteration_count} — новый цикл")
                
                # Проверка Telegram бота
                if tg_task.done():
                    exc = tg_task.exception()
                    if exc:
                        logger.exception("Telegram-бот упал с исключением:", exc)
                    else:
                        logger.error("Telegram-бот завершился без исключения")
                    logger.info("Пробуем перезапустить Telegram-бот через 10 секунд...")
                    await asyncio.sleep(10)
                    tg_task = asyncio.create_task(initialize_telegram_bot())

                # Получаем и перемешиваем символы для этой итерации
                symbols = get_selected_symbols()
                random.shuffle(symbols)

                # 1. Основная торговая логика (всегда работает)
                logger.info(f"[TRADING] Обработка сигналов в режиме: {OPERATION_MODE}")
                
                if OPERATION_MODE == "model_only":
                    tasks = [process_symbol_model_only(s) for s in symbols]
                    if tasks:
                        await asyncio.gather(*tasks)
                        
                elif OPERATION_MODE in ["drift_only", "drift_top10"]:
                    if top_list:
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
                        
                elif OPERATION_MODE == "ST_cross":
                    tasks = [
                        asyncio.to_thread(process_symbol_st_cross, s, interval=INTERVAL) 
                        for s in symbols
                    ]
                    if tasks:
                        await asyncio.gather(*tasks)

                # 2. Проверка трейлинг-стопов (если включено)
                if check_and_close_active:
                    check_and_set_trailing_stop()

                # 3. Публикация дрифт-таблицы (если включено)
                if publish_drift_table and iteration_count % 5 == 0:
                    latest_analysis = drift_analyzer.get_latest_analysis()
                    if latest_analysis:
                        top_list = get_top_anomalies_from_analysis(latest_analysis)
                        if top_list:
                            await publish_drift_and_model_tables()
                            if OPERATION_MODE in ["drift_only", "drift_top10"]:
                                handle_drift_top10(top_list)

                # 4. Логирование предсказаний модели
                tasks_log = []
                for s in symbols:
                    tasks_log.append(asyncio.to_thread(log_model_prediction_for_symbol, s))
                if tasks_log:
                    await asyncio.gather(*tasks_log)

                # 5. Переобучение модели (если нужно)
                if iteration_count % 20 == 0:
                    logger.info(f"[INNER_LOOP] iteration_count={iteration_count}, вызываем maybe_retrain_model()")
                    await maybe_retrain_model()

                # 6. Сверка позиций с биржей
                final_expos = get_exchange_positions()
                update_open_positions_from_exch_positions(final_expos)

                # 7. Генерация отчёта
                await asyncio.to_thread(generate_daily_pnl_report, "trade_log.csv", "daily_pnl_report.csv")

                publish_cycle += 5
                
                await asyncio.sleep(60)  # Пауза между итерациями

            except Exception as e_inner:
                logger.exception(f"Ошибка во внутреннем цикле: {e_inner}")
                await asyncio.sleep(60)
                continue

    except Exception as e_outer:
        logger.exception(f"Ошибка во внешнем цикле: {e_outer}")
        
    finally:
        # Проверяем, действительно ли была команда stop
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

def get_top_anomalies_from_analysis(analysis_data, top_k=10):
    """Получает топ аномалий из данных анализатора"""
    try:
        anomalies = []
        for symbol, data in analysis_data.items():
            if data['is_anomaly']:
                anomalies.append((symbol, data['strength'], data['direction']))
        
        # Сортируем по силе аномалии
        anomalies.sort(key=lambda x: x[1], reverse=True)
        return anomalies[:top_k]
    except Exception as e:
        logger.exception(f"Ошибка в get_top_anomalies_from_analysis: {e}")
        return []

class DriftAnalyzer(threading.Thread):
    def __init__(self, interval=60):
        super().__init__(daemon=True)
        self.interval = interval
        self.running = True
        self.last_analysis = {}
        self._lock = threading.Lock()

    def run(self):
        while self.running:
            try:
                symbols = get_selected_symbols()
                random.shuffle(symbols)
                
                for sym in symbols:
                    if not self.running:
                        break
                        
                    feature_cols = ["openPrice", "closePrice", "highPrice", "lowPrice"]
                    new_data = get_historical_data_for_trading(sym, "1", limit=200)
                    
                    if not new_data.empty:
                        is_anomaly, strength, direction = monitor_feature_drift_per_symbol(
                            sym,
                            new_data,
                            pd.DataFrame(),
                            feature_cols,
                            threshold=0.5
                        )
                        
                        with self._lock:
                            self.last_analysis[sym] = {
                                'timestamp': datetime.datetime.utcnow(),
                                'is_anomaly': is_anomaly,
                                'strength': strength,
                                'direction': direction
                            }
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.exception(f"[DriftAnalyzer] Ошибка: {e}")
                time.sleep(10)

    def stop(self):
        self.running = False

    def get_latest_analysis(self):
        with self._lock:
            return dict(self.last_analysis)


if __name__ == "__main__":
    main()