#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Многопользовательский бот для торговли на Bybit с использованием модели, дрейфа, супер-тренда и т.д.
Полностью асинхронная версия.
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
from time import sleep
import random
import sys
import csv
import datetime
from datetime import timezone
import pandas as pd
import numpy as np
import pandas_ta as ta
from decimal import Decimal, ROUND_HALF_UP, DivisionByZero
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from urllib3.util.retry import Retry
import aiohttp

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ReadTimeout

from aiogram import Bot, Dispatcher, Router, types, html
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton, TextQuote
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.exceptions import TelegramRetryAfter, TelegramBadRequest, TelegramNetworkError
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from dotenv import load_dotenv

import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import RANSACRegressor

from filterpy.kalman import KalmanFilter

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
import threading
from concurrent.futures import ThreadPoolExecutor


# Загрузка переменных окружения
load_dotenv()
load_dotenv("keys_TESTNET2.env")  # Ожидаются BYBIT_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID и т.д.

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[
        RotatingFileHandler("bot.log", maxBytes=5 * 1024 * 1024, backupCount=2),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Торговые параметры
MAX_TOTAL_VOLUME = Decimal("1000")         # общий лимит (USDT)
POSITION_VOLUME = Decimal("100")            # объём на сделку (USDT)
PROFIT_LEVEL = Decimal("0.008")             # порог закрытия позиции (например, 0.8%)
PROFIT_COEFFICIENT = Decimal("100")         # коэффициент перевода в проценты

TAKE_PROFIT_ENABLED = False
TAKE_PROFIT_LEVEL = Decimal("0.005")        # порог тейк-профита

TRAILING_STOP_ENABLED = True
TRAILING_GAP_PERCENT = Decimal("0.007")     # 0.7%
MIN_TRAILING_STOP = Decimal("0.0000001")
TRAILING_GAP_PERCENT_CUSTOM = Decimal("3")

QUIET_PERIOD_ENABLED = False                # режим тихого периода
IS_SLEEPING_MODE = False                    # спящий режим
OPERATION_MODE = "ST_cross3"                # режим работы бота
HEDGE_MODE = True
INVERT_MODEL_LABELS = False

MODEL_FILENAME = "trading_model_final.pkl"
MIN_SAMPLES_FOR_TRAINING = 1000

# --- SUPERTREND --- #

HISTORICAL_DATA_LIMIT = 205
FAST_ST_LENGTH = 2
FAST_ST_MULTIPLIER = 1.0
SLOW_ST_LENGTH = 8
SLOW_ST_MULTIPLIER = 2.0
CONFIRM_ST_LENGTH = 50
CONFIRM_ST_MULTIPLIER = 3.0
STALE_DATA_THRESHOLD = pd.Timedelta(minutes=5)
STATE_TIMEOUT = pd.Timedelta(minutes=10)
MIN_DATA_LENGTH = 10
MIN_ALIGNED_BARS = 5
LOOKBACK_BARS = 5

# --- SUPERTREND --- #


ADMIN_ID = 36972091  # ваш user_id, кто имеет право останавливать бота

VOLATILITY_THRESHOLD = 0.05
VOLUME_THRESHOLD = Decimal("2000000")
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

# Настройка HTTP-сессии для запросов к Bybit
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

ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())

# Другие глобальные объекты
telegram_bot = None
router = Router()
router_admin = Router()
telegram_message_queue = None
send_semaphore = asyncio.Semaphore(10)

monitoring = None

ws_demo = None
ws_real = None
active_position_subscriptions = set()


# Глобальные блокировки и словари (заменяем threading.Lock на asyncio.Lock)
#open_positions_lock = asyncio.Lock()
history_lock = asyncio.Lock()

#open_positions = {}  # Ключ – символ, значение – данные позиции
open_interest_history = defaultdict(list)
volume_history = defaultdict(list)

executor = ThreadPoolExecutor()

# ------------------ Функции работы с CSV ------------------

async def load_users(filename="users.csv"):
    def _load():
        users_dict = {}
        try:
            if os.path.exists(filename):
                with open(filename, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            user_id = int(row["user_id"].strip())
                            user_api = row["user_api"].strip()
                            user_api_secret = row["user_api_secret"].strip()
                            mode = row.get("mode", "demo").strip().lower()
                            monitoring = row.get("monitoring", "demo_ws").strip().lower()
                            max_total_volume_str = row.get("max_total_volume", "1000").strip()
                            position_volume_str = row.get("position_volume", "100").strip()
                            users_dict[user_id] = {
                                "user_api": user_api,
                                "user_api_secret": user_api_secret,
                                "mode": mode,
                                "monitoring": monitoring,
                                "max_total_volume": max_total_volume_str,
                                "position_volume": position_volume_str
                            }
                            logger.info(f"Loaded user {user_id}, mode={mode}, monitoring={monitoring}, max_total_volume={max_total_volume_str}, position_volume={position_volume_str}")
                        except Exception as e:
                            logger.error(f"Error loading user from row={row}: {e}")
            else:
                logger.error(f"Users file {filename} not found!")
        except Exception as e:
            logger.error(f"Critical error loading users: {e}")
        return users_dict
    return await asyncio.to_thread(_load)

async def save_users(users_dict, filename="users.csv"):
    def _save():
        try:
            fieldnames = ["user_id", "user_api", "user_api_secret", "mode", "monitoring", "max_total_volume", "position_volume"]
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for uid, data in users_dict.items():
                    writer.writerow({
                        "user_id": uid,
                        "user_api": data["user_api"],
                        "user_api_secret": data["user_api_secret"],
                        "mode": data["mode"],
                        "monitoring": data["monitoring"],
                        "max_total_volume": data["max_total_volume"],
                        "position_volume": data["position_volume"]
                    })
            logger.info(f"save_users: файл {filename} успешно перезаписан.")
        except Exception as e:
            logger.error(f"save_users: ошибка сохранения {filename}: {e}")
    await asyncio.to_thread(_save)

# ------------------ Идентификация пользователей ------------------

users = {}
user_bots = {}

class RegisterStates(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_api_secret = State()
    waiting_for_mode = State()

async def add_user_to_csv(user_id: int, user_api: str, user_api_secret: str, filename="users.csv"):
    def _add():
        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["user_id", "user_api", "user_api_secret"])
            writer.writerow([user_id, user_api, user_api_secret])
    await asyncio.to_thread(_add)

async def init_user_bots():
    global users, user_bots
    bot = TradingBot
    users = await load_users()
    logger.info(f"Загружено пользователей: {list(users.keys())}")
    for uid, user_data in users.items():
        api = user_data["user_api"]
        secret = user_data["user_api_secret"]
        mode = user_data["mode"]
        monitoring = user_data["monitoring"]
        max_vol_str = user_data["max_total_volume"]
        pos_vol_str = user_data["position_volume"]
        bot_instance = TradingBot(
            user_id=uid,
            user_api=api,
            user_api_secret=secret,
            mode=mode,
            monitoring=monitoring,
            max_total_volume=max_vol_str,
            position_volume=pos_vol_str
        )

        bot_instance.loop = asyncio.get_running_loop()
        user_bots[uid] = bot_instance
        logger.info(f"Создан бот для user_id={uid} (mode={mode})")

# ------------------ Класс TradingBot ------------------

class TradingBot:
    def __init__(self, user_id: int, user_api: str, user_api_secret: str, mode: str, monitoring: str,
                 max_total_volume="1000", position_volume="100"):
        self.user_id = user_id
        self.user_api = user_api
        self.user_api_secret = user_api_secret
        self.mode = mode.lower()

        self.monitoring = monitoring.lower()
        self._init_http_session()

        self.ws_private = None
        self.ws_public = None
        self.ws_reconnect_interval = 5  # seconds
        self._ws_active = False
        self.active = True  # Добавляем инициализацию атрибута
        self.ws = None
        self.ws_ticker_stream = None
        self.open_positions = {}
        self.open_positions_lock = asyncio.Lock()
        self.active_position_subscriptions = set()
        self.MAX_TOTAL_VOLUME = Decimal(str(max_total_volume))
        self.POSITION_VOLUME = Decimal(str(position_volume))
        if self.mode == "demo":
            self.session = HTTP(
                demo=True,
                api_key=self.user_api,
                api_secret=self.user_api_secret,
                timeout=60,
            )
        else:
            self.session = HTTP(
                testnet=False,
                api_key=self.user_api,
                api_secret=self.user_api_secret,
                timeout=60,
            )
        self.last_kline_data = {}
        self.state = {"connectivity_ok": True}
        self.open_positions = {}  # Асинхронно защищаем через self.open_positions_lock
        self.drift_history = defaultdict(list)
        self.selected_symbols = []
        # self.MAX_TOTAL_VOLUME = MAX_TOTAL_VOLUME
        # self.POSITION_VOLUME = POSITION_VOLUME
        self.PROFIT_LEVEL = PROFIT_LEVEL
        self.PROFIT_COEFFICIENT = PROFIT_COEFFICIENT
        self.TAKE_PROFIT_ENABLED = TAKE_PROFIT_ENABLED
        self.TAKE_PROFIT_LEVEL = TAKE_PROFIT_LEVEL
        self.TRAILING_STOP_ENABLED = TRAILING_STOP_ENABLED
        self.TRAILING_GAP_PERCENT = TRAILING_GAP_PERCENT
        self.TRAILING_GAP_PERCENT_CUSTOM = TRAILING_GAP_PERCENT_CUSTOM
        self.MIN_TRAILING_STOP = MIN_TRAILING_STOP
        self.CUSTOM_TRAILING_STOP_ENABLED = True
        self.supertrend_custom_trailing_stop = False
        self.QUIET_PERIOD_ENABLED = QUIET_PERIOD_ENABLED
        self.IS_SLEEPING_MODE = IS_SLEEPING_MODE
        self.OPERATION_MODE = OPERATION_MODE
        self.HEDGE_MODE = HEDGE_MODE
        self.INVERT_MODEL_LABELS = INVERT_MODEL_LABELS
        self.MODEL_FILENAME = MODEL_FILENAME
        self.MIN_SAMPLES_FOR_TRAINING = MIN_SAMPLES_FOR_TRAINING
        self.VOLATILITY_THRESHOLD = VOLATILITY_THRESHOLD
        self.VOLUME_THRESHOLD = VOLUME_THRESHOLD
        self.TOP_N_PAIRS = TOP_N_PAIRS
        self.golden_params = golden_params
        self.REAL_TRADES_FEATURES_CSV = "real_trades_features.csv"
        self.MODEL_FEATURE_COLS = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
        self.INTERVAL = "1"
        self.MAX_AVERAGING_VOLUME = self.MAX_TOTAL_VOLUME * Decimal("2")
        self.averaging_total_volume = Decimal("0")
        self.averaging_positions = {}
        self.TARGET_LOSS_FOR_AVERAGING = Decimal("16.0")
        self.MONITOR_MODE = "http" # ws
        self.state_lock = asyncio.Lock()
        self.open_positions_lock = asyncio.Lock()
        self.history_lock = asyncio.Lock()
        self.current_model = None
        self.last_asset_selection_time = 0
        self.ASSET_SELECTION_INTERVAL = 60 * 60
        self.historical_data = pd.DataFrame()
        self.load_historical_data()  # синхронно – можно обернуть в to_thread если нужно
        self.pending_signal = None  # Здесь храним отложенный сигнал {'type': 'Buy'/'Sell', 'symbol': '...', 'activated': bool}
        self.last_volatility_check_hour = None
        self.volatile_pairs = set()
        self.loop = None  # Будет установлен в main()
        self.latest_mark_prices = {}
        self.latest_entry_prices = {}
        self.ws_positions = {}  # Данные, приходящие только от WebSocket
        self.recently_closed = {}  # symbol -> timestamp
        self.current_total_volume = Decimal("0")
        self.recent_signals = {}  # symbol -> timestamp
        self.awaiting_position_update = {}  # symbol: timestamp


    def load_historical_data(self):
        try:
            if os.path.exists("historical_data_for_model_5m.csv"):
                self.historical_data = pd.read_csv("historical_data_for_model_5m.csv")
                logger.info("Historical data loaded from historical_data_for_model_5m.csv")
            else:
                logger.warning("Файл historical_data_for_model_5m.csv не найден!")
                self.historical_data = pd.DataFrame()
        except Exception as e:
            logger.exception(f"Ошибка при загрузке historical_data_for_model_5m.csv: {e}")
            self.historical_data = pd.DataFrame()

    async def get_total_open_volume(self) -> Decimal:
        total_vol = Decimal("0")
        async with self.open_positions_lock:
            for sym, pos in self.open_positions.items():
                try:
                    # Предполагается, что объем позиции хранится либо в ключе "position_volume", либо в "size"
                    vol_str = pos.get("position_volume", pos.get("size", "0"))
                    vol = Decimal(str(vol_str))
                    total_vol += vol
                except Exception as e:
                    logger.exception(f"Ошибка суммирования объёма позиции для {sym}: {e}")
        return total_vol

    async def refresh_positions_from_stream(self):
        """
        Обновляет локальный словарь открытых позиций (self.open_positions)
        на основе данных, полученных из стрим‑сообщения.
        Фильтрует позиции – сохраняются только те, у которых size > 0.
        Предполагается, что метод self.get_exchange_positions() (синхронный) возвращает dict с позициями.
        """
        try:
            positions_snapshot = self.get_exchange_positions()
            if positions_snapshot and isinstance(positions_snapshot, dict):
                async with self.open_positions_lock:
                    logger.info(f"[refresh_positions_from_stream] Итоговый список открытых позиций: {list(self.open_positions.keys())}")
                    # Проходим по snapshot-позициям и сохраняем только открытые (size > 0)
                    for symbol, pos in positions_snapshot.items():
                        pos_size = pos.get("size")
                        if pos_size is not None and float(pos_size) > 0:
                            self.open_positions[symbol] = pos
                        else:
                            # Если позиция закрыта, удаляем её из локального словаря (если она там есть)
                            if symbol in self.open_positions:
                                del self.open_positions[symbol]
                        logger.info(f"[refresh_positions_from_stream] Получен snapshot открытых позиций: { {symbol: pos for symbol, pos in positions_snapshot.items() if pos.get('size') and float(pos.get('size')) > 0} }")
                    else:
                        logger.warning("refresh_positions_from_stream: получены некорректные данные от get_exchange_positions().")
        except Exception as e:
            logger.exception(f"refresh_positions_from_stream: ошибка при обновлении открытых позиций: {e}")

    async def get_last_close_price(self, symbol: str):
        params = {"category": "linear", "symbol": symbol, "interval": "1", "limit": 1}
        def _get_kline():
            return self.session.get_kline(**params)
        try:
            resp = await asyncio.to_thread(_get_kline)
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

    async def get_historical_data_for_trading_5m(self, symbol: str, interval="1", limit=200, from_time=None):
        CACHE_EXPIRY_SECONDS = 60  # время жизни кэша в секундах

        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        if from_time:
            params["from"] = from_time

        def _get_kline():
            return self.session.get_kline(**params)

        try:
            resp = await asyncio.to_thread(_get_kline)
            # Если запрос вернул ошибку
            if resp.get("retCode") != 0:
                logger.error(f"[TRADING_KLINE] {symbol}: {resp.get('retMsg')}")
                if symbol in self.last_kline_data:
                    cache_time, cached_df = self.last_kline_data[symbol]
                    if time.time() - cache_time < CACHE_EXPIRY_SECONDS:
                        logger.info(f"[TRADING_KLINE] Использую свежие кэшированные данные для {symbol}")
                        return cached_df
                    else:
                        logger.info(f"[TRADING_KLINE] Кэш для {symbol} устарел, очищаю кэш")
                        del self.last_kline_data[symbol]
                return pd.DataFrame()

            # Обработка загруженных данных
            data = resp["result"].get("list", [])
            if not data:
                logger.info(f"[TRADING_KLINE] Данных нет, возвращаю пустой DataFrame для {symbol}")
                return pd.DataFrame()

            columns = ["open_time", "open", "high", "low", "close", "volume", "open_interest"]
            df = pd.DataFrame(data, columns=columns)
            df["startTime"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
            df.rename(columns={"open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"}, inplace=True)
            df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]] = \
                df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]].apply(pd.to_numeric, errors="coerce")
            df.dropna(subset=["closePrice"], inplace=True)
            df.sort_values("startTime", inplace=True)
            df.reset_index(drop=True, inplace=True)

            processed_df = df[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]]
            # Сохраняем в кэш с меткой времени
            self.last_kline_data[symbol] = (time.time(), processed_df.copy())
            return processed_df

        except ReadTimeout as rt:
            logger.error(f"[get_historical_data_for_trading({symbol})]: Таймаут чтения: {rt}")
            if symbol in self.last_kline_data:
                cache_time, cached_df = self.last_kline_data[symbol]
                if time.time() - cache_time < CACHE_EXPIRY_SECONDS:
                    logger.info(f"[get_historical_data_for_trading({symbol})]: Использую свежие кэшированные данные")
                    return cached_df
                else:
                    logger.info(f"[get_historical_data_for_trading({symbol})]: Кэш для {symbol} устарел, очищаю кэш")
                    del self.last_kline_data[symbol]
            return pd.DataFrame()

        except Exception as e:
            logger.exception(f"[get_historical_data_for_trading({symbol})]: {e}")
            if symbol in self.last_kline_data:
                cache_time, cached_df = self.last_kline_data[symbol]
                if time.time() - cache_time < CACHE_EXPIRY_SECONDS:
                    logger.info(f"[get_historical_data_for_trading({symbol})]: Использую свежие кэшированные данные")
                    return cached_df
                else:
                    logger.info(f"[get_historical_data_for_trading({symbol})]: Кэш для {symbol} устарел, очищаю кэш")
                    del self.last_kline_data[symbol]
            return pd.DataFrame()

    async def get_historical_data_for_trading(self, symbol: str, interval="1", limit=200, from_time=None):
        CACHE_EXPIRY_SECONDS = 60  # время жизни кэша в секундах

        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        if from_time:
            params["from"] = from_time

        def _get_kline():
            return self.session.get_kline(**params)

        try:
            resp = await asyncio.to_thread(_get_kline)
            # Если запрос вернул ошибку
            if resp.get("retCode") != 0:
                logger.error(f"[TRADING_KLINE] {symbol}: {resp.get('retMsg')}")
                if symbol in self.last_kline_data:
                    cache_time, cached_df = self.last_kline_data[symbol]
                    if time.time() - cache_time < CACHE_EXPIRY_SECONDS:
                        logger.info(f"[TRADING_KLINE] Использую свежие кэшированные данные для {symbol}")
                        return cached_df
                    else:
                        logger.info(f"[TRADING_KLINE] Кэш для {symbol} устарел, очищаю кэш")
                        del self.last_kline_data[symbol]
                return pd.DataFrame()

            # Обработка загруженных данных
            data = resp["result"].get("list", [])
            if not data:
                logger.info(f"[TRADING_KLINE] Данных нет, возвращаю пустой DataFrame для {symbol}")
                return pd.DataFrame()

            columns = ["open_time", "open", "high", "low", "close", "volume", "open_interest"]
            df = pd.DataFrame(data, columns=columns)
            df["startTime"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
            df.rename(columns={"open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"}, inplace=True)
            df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]] = \
                df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]].apply(pd.to_numeric, errors="coerce")
            df.dropna(subset=["closePrice"], inplace=True)
            df.sort_values("startTime", inplace=True)
            df.reset_index(drop=True, inplace=True)

            processed_df = df[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]]
            # Сохраняем в кэш с меткой времени
            self.last_kline_data[symbol] = (time.time(), processed_df.copy())
            return processed_df

        except ReadTimeout as rt:
            logger.error(f"[get_historical_data_for_trading({symbol})]: Таймаут чтения: {rt}")
            if symbol in self.last_kline_data:
                cache_time, cached_df = self.last_kline_data[symbol]
                if time.time() - cache_time < CACHE_EXPIRY_SECONDS:
                    logger.info(f"[get_historical_data_for_trading({symbol})]: Использую свежие кэшированные данные")
                    return cached_df
                else:
                    logger.info(f"[get_historical_data_for_trading({symbol})]: Кэш для {symbol} устарел, очищаю кэш")
                    del self.last_kline_data[symbol]
            return pd.DataFrame()

        except Exception as e:
            logger.exception(f"[get_historical_data_for_trading({symbol})]: {e}")
            if symbol in self.last_kline_data:
                cache_time, cached_df = self.last_kline_data[symbol]
                if time.time() - cache_time < CACHE_EXPIRY_SECONDS:
                    logger.info(f"[get_historical_data_for_trading({symbol})]: Использую свежие кэшированные данные")
                    return cached_df
                else:
                    logger.info(f"[get_historical_data_for_trading({symbol})]: Кэш для {symbol} устарел, очищаю кэш")
                    del self.last_kline_data[symbol]
            return pd.DataFrame()

    async def prepare_features_for_model(self, df: pd.DataFrame):
        try:
            for c in ["openPrice", "highPrice", "lowPrice", "closePrice"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=["closePrice"], inplace=True)
            if df.empty:
                return df
            df["ohlc4"] = (df["openPrice"] + df["highPrice"] + df["lowPrice"] + df["closePrice"]) / 4
            macd_df = await self.calculate_macd(df["ohlc4"])
            df["macd"] = macd_df["MACD_12_26_9"]
            df["macd_signal"] = macd_df["MACDs_12_26_9"]
            df["rsi_13"] = await self.calculate_rsi(df["ohlc4"], periods=13)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["macd", "macd_signal", "rsi_13"], inplace=True)
            return df
        except Exception as e:
            logger.exception(f"Ошибка prepare_features_for_model: {e}")
            return pd.DataFrame()

    async def calculate_macd(self, close_prices, fast=12, slow=26, signal=9):
        exp1 = close_prices.ewm(span=fast, adjust=False).mean()
        exp2 = close_prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({'MACD_12_26_9': macd, 'MACDs_12_26_9': signal_line})

    async def calculate_rsi(self, close_prices, periods=13):
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    async def make_multiclass_target_for_model(self, df: pd.DataFrame, horizon=1, threshold=Decimal("0.0025")):
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

    def get_selected_symbols(self):
        now = time.time()
        if now - self.last_asset_selection_time >= self.ASSET_SELECTION_INTERVAL or not self.selected_symbols:
            self.load_historical_data()
            tickers_resp = self.session.get_tickers(symbol=None, category="linear")
            if "result" not in tickers_resp or "list" not in tickers_resp["result"]:
                logger.error("[get_selected_symbols] Некорректный ответ get_tickers.")
                self.selected_symbols = []
                return self.selected_symbols
            tickers_data = tickers_resp["result"]["list"]
            inst_resp = self.session.get_instruments_info(category="linear")
            if "result" not in inst_resp or "list" not in inst_resp["result"]:
                logger.error("[get_selected_symbols] Некорректный ответ get_instruments_info.")
                self.selected_symbols = []
                return self.selected_symbols
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
            self.selected_symbols = usdt_pairs
            self.last_asset_selection_time = now
            logger.info(f"Обновлён список активов: {self.selected_symbols}")
        return self.selected_symbols

    def _init_http_session(self):
        """Инициализация REST API соединения"""
        if self.mode == "demo":
            # Демо-режим mainnet
            self.session = HTTP(
                demo=True,
                api_key=self.user_api,
                api_secret=self.user_api_secret,
                timeout=60
            )
        elif self.mode == "real":
            # Реальный режим mainnet
            self.session = HTTP(
                testnet=False,
                api_key=self.user_api,
                api_secret=self.user_api_secret,
                timeout=60
            )
        else:
            raise ValueError(f"Неизвестный тип подключения: {self.monitoring}")

    # async def init_websocket(self):
    #     """Асинхронная инициализация WebSocket"""
    #     try:
    #         logger.info("[WebSocket] Запуск WS...")

    #         await self.update_open_positions_from_exchange()

    #         if self.monitoring == "demo_ws":
    #             self.ws = WebSocket(
    #                 testnet=False,
    #                 demo=True,
    #                 channel_type="private",
    #                 api_key=self.user_api,
    #                 api_secret=self.user_api_secret
    #             )
    #         elif self.monitoring == "real_ws":
    #             self.ws = WebSocket(
    #                 testnet=False,
    #                 channel_type="private",
    #                 api_key=self.user_api,
    #                 api_secret=self.user_api_secret
    #             )

    #         if self.mode == "demo":
    #             self.ws_ticker_stream = WebSocket(
    #                 testnet=False,
    #                 channel_type="linear"
    #             )
    #         if self.ws is not None:
    #             await self.ws.subscribe("position", callback=self.handle_position_update)
    #             await self.ws.subscribe("order", callback=self.handle_order_update)
    #         else:
    #             logger.warning(f"[WebSocket] ws = None! Не удалось подписаться на 'position' и 'order'")

    #         if self.ws_ticker_stream is not None:
    #             await self.ws_ticker_stream.subscribe("tickers", callback=self.handle_ticker_for_symbol)
    #         else:
    #             logger.warning(f"[WebSocket] ws_ticker_stream = None! Не удалось подписаться на 'tickers'")

    #         logger.info(f"WebSocket: подключено для {self.user_id}")

    #     except Exception as e:
    #         logger.error(f"WebSocket error: {str(e)}")
    #         await self.safe_reconnect()

    async def init_ticker_websocket(self):
        """Инициализация WebSocket для получения рыночных данных"""
        try:
            logger.info(f"[{self.user_id}] Инициализация WebSocket для тикеров...")

            self.ws_ticker_stream = WebSocket(
                testnet=False,
                channel_type="linear"
            )

            logger.info(f"[{self.user_id}] WebSocket для тикеров успешно подключен")

        except Exception as e:
            logger.error(f"[{self.user_id}] Ошибка инициализации WebSocket тикеров: {str(e)}")
            await self.safe_reconnect()

    async def handle_ticker_update(self, message):
        """Обработчик обновлений рыночных данных"""
        try:
            if 'data' not in message:
                return
                
            data = message['data']
            symbol = data['symbol']
            mark_price = float(data['markPrice'])
            
            # Обновляем последнюю цену в реальном времени
            self.latest_mark_prices[symbol] = mark_price
            
            # Если есть открытая позиция - проверяем условия
            async with self.open_positions_lock:
                if symbol in self.open_positions:
                    await self.check_and_manage_position(symbol)
                    
        except Exception as e:
            logger.error(f"[{self.user_id}] Ошибка обработки тикера: {str(e)}")

    async def realtime_position_monitor(self):
        """Мониторинг позиций в реальном времени через WebSocket"""
        try:
            await self.init_ticker_websocket()
            
            while self.active:
                # Поддерживаем соединение и обновляем подписки
                async with self.open_positions_lock:
                    current_symbols = set(self.open_positions.keys())
                    subscribed = self.active_position_subscriptions
                    
                    # Подписываемся на новые символы
                    for symbol in current_symbols - subscribed:
                        await self.subscribe_to_ticker(symbol)
                        
                    # Отписываемся от закрытых
                    for symbol in subscribed - current_symbols:
                        await self.unsubscribe_from_ticker(symbol)
                
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"[{self.user_id}] Ошибка мониторинга позиций: {str(e)}")
            await self.safe_reconnect()

    # Модифицируем метод check_and_manage_position

    async def check_and_manage_position_v2(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            logger.debug(f"[check_v2] Нет открытой позиции для {symbol}")
            return

        side = pos.get("side")
        entry_price = Decimal(str(pos.get("avg_price", 0)))
        current_price = self.latest_mark_prices.get(symbol)

        if current_price is None:
            logger.debug(f"[check_v2] Нет цены в WebSocket для {symbol}")
            return
        current_price = Decimal(str(current_price))

        if side.lower() == "buy":
            ratio = (current_price - entry_price) / entry_price
        else:
            ratio = (entry_price - current_price) / entry_price

        profit_perc = (ratio * self.PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
        logger.info(f"[WS Monitor] {symbol}: entry={entry_price}, current={current_price}, PnL={profit_perc}%")

        # --- Усреднение по убытку ---
        if profit_perc <= -self.TARGET_LOSS_FOR_AVERAGING:
            current_volume = Decimal(str(pos["position_volume"]))
            logger.info(f"[WS Monitor] {symbol}: Убыток {profit_perc}% достиг порога усреднения. Объём: {current_volume}")
            await self.open_averaging_position_all(symbol, current_volume)

        # --- PnL с учётом плеча ---
        default_leverage = Decimal("10")
        leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))

        threshold_trailing = Decimal("5.0")
        if self.CUSTOM_TRAILING_STOP_ENABLED:
            await self.apply_custom_trailing_stop(symbol, pos, leveraged_pnl_percent, side)
        elif self.supertrend_custom_trailing_stop:
            await self.apply_supertrend_custom_trailing_stop(symbol, pos, leveraged_pnl_percent, side)
        else:
            if leveraged_pnl_percent >= threshold_trailing and not pos.get("trailing_stop_set", False):
                logger.info(f"[WS Monitor] {symbol}: Достигнут уровень трейлинга (PnL = {leveraged_pnl_percent}%). Устанавливаем.")
                await self.set_trailing_stop(symbol, pos["size"], self.TRAILING_GAP_PERCENT, side)

    async def start_monitoring_via_ws(self):
        """
        Запускает мониторинг через WebSocket:
        1. Один раз получает открытые позиции через REST
        2. Инициализирует WS-поток для тикеров
        3. Подписывается на все тикеры с открытыми позициями
        4. Запускает фоновый мониторинг и (опционально) сверку
        """
        try:
            logger.info(f"[{self.user_id}] ▶️ Старт мониторинга через WebSocket")

            # (1) Разовая инициализация позиций через REST
            positions = await asyncio.to_thread(self.get_exchange_positions)
            await self.update_open_positions_from_exch_positions(positions)

            # (2) Инициализируем WebSocket соединение
            await self.init_ticker_websocket()

            # (3) Групповая подписка на тикеры (из open_positions)
            await self.update_tickers_subscription()

            # (4) Запускаем фоновый цикл для периодической сверки
            asyncio.create_task(self.periodic_position_sync())

        except Exception as e:
            logger.exception(f"[{self.user_id}] ❌ Ошибка в start_monitoring_via_ws: {e}")

    def safe_send_ping(self):
        try:
            if self.ws and self.ws.sock and self.ws.sock.connected:
                self.ws.send(self.custom_ping_message)
            else:
                logger.warning(f"[{self.user_id}] Ping пропущен: соединение закрыто")
        except Exception as e:
            logger.warning(f"[{self.user_id}] Ошибка safe_ping: {e}")

    # async def check_and_manage_position(self, symbol):
    #     """Проверка условий для трейлинг-стопа в реальном времени"""
    #     try:
    #         # Получаем актуальные данные позиции
    #         async with self.open_positions_lock:
    #             pos = self.open_positions.get(symbol)
    #             if not pos or pos.get('size', 0) <= 0:
    #                 return

    #             # Получаем последние цены из WebSocket
    #             mark_price = self.latest_mark_prices.get(symbol)
    #             entry_price = pos.get('avg_price')
                
    #             if None in (mark_price, entry_price):
    #                 return

    #             # Рассчитываем прибыльность
    #             side = pos['side'].lower()
    #             leverage = Decimal('10')  # Ваше плечо
                
    #             if side == 'buy':
    #                 price_diff = Decimal(mark_price) - Decimal(entry_price)
    #             else:
    #                 price_diff = Decimal(entry_price) - Decimal(mark_price)
                    
    #             pnl_ratio = price_diff / Decimal(entry_price)
    #             leveraged_pnl_percent = pnl_ratio * leverage * 100  # В процентах с учетом плеча

    #             # Обновляем информацию о прибыльности в позиции
    #             self.open_positions[symbol]['pnl'] = float(leveraged_pnl_percent)
    #             self.open_positions[symbol]['current_price'] = float(mark_price)

    #             # Условия активации кастомного трейлинг-стопа
    #             if leveraged_pnl_percent >= float(self.PROFIT_LEVEL * 100):  # Конвертируем в проценты
    #                 logger.info(f"[Trailing] {symbol} Достигнут Profit Level: {leveraged_pnl_percent:.2f}%")
                    
    #                 # Если трейлинг еще не активирован
    #                 if not pos.get('trailing_activated'):
    #                     await self.apply_custom_trailing_stop(symbol, pos, leveraged_pnl_percent, side)
                        
    #                 # Обновляем трейлинг при движении цены
    #                 await self.update_trailing_stop(symbol, mark_price, side)

    #     except Exception as e:
    #         logger.error(f"Ошибка управления позицией {symbol}: {str(e)}")


    # async def init_websocket(self):
    #     """Асинхронная инициализация WebSocket (без дублирования соединений)"""
    #     try:
    #         if self.ws and self.ws.is_connected():
    #             logger.info(f"[WebSocket] Уже подключено для {self.user_id}, повторная инициализация не требуется.")
    #             return

    #         logger.info("[WebSocket] Инициализация WebSocket...")

    #         await self.update_open_positions_from_exchange()

    #         # Инициализация приватного WebSocket
    #         if not self.ws:
    #             self.ws = WebSocket(
    #                 testnet=False,
    #                 demo=(self.monitoring == "demo_ws"),
    #                 channel_type="private",
    #                 api_key=self.user_api,
    #                 api_secret=self.user_api_secret
    #             )

    #         # Инициализация публичного стрима для тикеров (один раз!)
    #         if not self.ws_ticker_stream:
    #             self.ws_ticker_stream = WebSocket(
    #                 testnet=False,
    #                 channel_type="linear"
    #             )

    #         # Подписка на каналы (если ещё не подписан)
    #         if self.ws is not None:
    #             await self.ws.subscribe("position", callback=lambda msg: asyncio.create_task(self.handle_position_update(msg)))
    #             await self.ws.subscribe("order", callback=lambda msg: asyncio.create_task(self.handle_order_update(msg)))
    #         else:
    #             logger.error("WebSocket client is None during subscription. Skipping subscribe.")
    #         logger.info(f"[WebSocket] Подписан на 'position' и 'order' для {self.user_id}")

    #         await self.ws_ticker_stream.subscribe("tickers", callback=self.handle_ticker_for_symbol)
    #         logger.info(f"[WebSocket] Подписан на 'tickers' stream")

    #     except Exception as e:
    #         logger.error(f"[WebSocket Error] Ошибка при инициализации WebSocket: {e}")
    #         await self.safe_reconnect()

    # async def handle_position_update(self, message):
    #     logger.info(f"Получено сообщение по позиции: {message}")
    #     try:
    #         if not isinstance(message, dict):
    #             return

    #         positions_data = message.get("data", [])
    #         updated_symbols = set()

    #         for pos_info in positions_data:
    #             symbol = pos_info.get("symbol")
    #             side = pos_info.get("side", "")
    #             size_raw = pos_info.get("size", "0")

    #             try:
    #                 size = float(size_raw)
    #             except (ValueError, TypeError):
    #                 logger.warning(f"[WebSocket] Некорректный размер позиции: {size_raw}")
    #                 continue

    #             if not symbol:
    #                 continue

    #             updated_symbols.add(symbol)

    #             if size > 0 and side:
    #                 entry_price = float(pos_info.get("entryPrice", 0))
    #                 mark_price = float(pos_info.get("markPrice", 0))

    #                 # 💡 Не затираем валидный entryPrice, если приходит нулевой
    #                 if symbol in self.ws_positions:
    #                     existing_entry = self.ws_positions[symbol].get("avg_price", 0)
    #                     if entry_price == 0 and existing_entry > 0:
    #                         entry_price = existing_entry
                            
    #                 self.latest_mark_prices[symbol] = mark_price
    #                 position_value = size * mark_price

    #                 pos_data = {
    #                     "symbol": symbol,
    #                     "size": size,
    #                     "side": side,
    #                     "avg_price": entry_price,
    #                     "entryPrice": entry_price,
    #                     "markPrice": mark_price,
    #                     "position_volume": position_value,
    #                     "trailing_stop_set": False
    #                 }

    #                 self.ws_positions[symbol] = pos_data.copy()
    #                 self.open_positions[symbol] = pos_data.copy()
    #                 await self.subscribe_to_ticker(symbol)

    #                 logger.info(f"[DEBUG] Вызов check_and_manage_position для {symbol} из position_update")
    #                 if symbol not in self.latest_mark_prices:
    #                     logger.info(f"[WS] Пока нет mark_price для {symbol}, отложим check_and_manage_position.")
    #                     continue
    #                 await self.check_and_manage_position(symbol)

    #                 if symbol in self.awaiting_position_update:
    #                     logger.info(f"[WS] Получено подтверждение по {symbol}, убираем из ожидания.")
    #                     del self.awaiting_position_update[symbol]

    #             elif size == 0:
    #                 active = any(
    #                     float(p.get("size", 0)) > 0 and p.get("symbol") == symbol
    #                     for p in positions_data
    #                 )
    #                 if active:
    #                     continue

    #                 logger.info(f"[WebSocket] Закрытие позиции {symbol}")
    #                 old_pos = self.open_positions.pop(symbol, None)
    #                 self.ws_positions.pop(symbol, None)

    #                 if old_pos:
    #                     row = await self.get_last_row(symbol)
    #                     pnl = old_pos.get("profit_perc", 0)
    #                     await self.log_trade(
    #                         user_id=self.user_id,
    #                         symbol=symbol,
    #                         row=row,
    #                         side=old_pos.get("side", "N/A"),
    #                         open_interest=None,
    #                         action=f"Позиция закрыта (WebSocket, PnL: {pnl}%)",
    #                         result="closed",
    #                         closed_manually=False
    #                     )
    #                     self.recently_closed[symbol] = time.time()
    #                     await self.unsubscribe_from_ticker(symbol)

    #                 self.current_total_volume = sum(
    #                     Decimal(str(p.get("position_volume", 0)))
    #                     for p in self.open_positions.values()
    #                 )
    #                 logger.info(f"[Объём] Пересчитан общий объём: {self.current_total_volume} USDT")

    #     except Exception as e:
    #         logger.exception(f"Ошибка обработки позиции: {e}")


    # async def handle_order_update(self, message):
    #     logger.info(f"[DEBUG] handle_order_update вызван с message: {message}")
    #     try:
    #         data = message.get("data", [])
    #         for order in data:
    #             symbol = order.get("symbol")
    #             order_status = order.get("orderStatus", "")
    #             side = order.get("side", "")
    #             is_filled = order_status.upper() == "FILLED"

    #             if is_filled:
    #                 logger.info(f"Ордер по {symbol} ({side}) полностью исполнен!")

    #                 avg_price = float(order.get("avgPrice", 0))
    #                 if symbol in self.ws_positions:
    #                     self.ws_positions[symbol]["price"] = avg_price
    #                     self.ws_positions[symbol]["avg_price"] = avg_price
    #                     logger.info(f"[Order Update] Установлен entryPrice={avg_price} для {symbol} из avgPrice ордера")
    #                 else:
    #                     self.ws_positions[symbol] = {
    #                         "symbol": symbol,
    #                         "side": side,
    #                         "entryPrice": avg_price,
    #                         "avg_price": avg_price,
    #                         "size": 0,
    #                         "markPrice": 0,
    #                         "position_volume": 0,
    #                         "trailing_stop_set": False
    #                     }
    #                     logger.info(f"[Order Update] Создана новая позиция для {symbol} с entryPrice={avg_price}")

    #                 self.awaiting_position_update[symbol] = time.time()

    #     except Exception as e:
    #         logger.exception(f"Ошибка в обработке обновления ордера: {e}")

    # async def handle_ticker_for_symbol(self, message):
    #     try:
    #         data = message.get("data", {})
    #         symbol = data.get("symbol")
    #         mark_price = float(data.get("markPrice", 0))

    #         if not symbol or mark_price == 0:
    #             return

    #         if symbol in self.ws_positions:
    #             self.ws_positions[symbol]["markPrice"] = mark_price
    #             # self.latest_mark_prices[symbol] = mark_price

    #             logger.debug(f"[Ticker Update] {symbol}: markPrice обновлён до {mark_price}")
    #             await self.check_and_manage_position(symbol)

    #     except Exception as e:
    #         logger.exception(f"[Ticker Handler] Ошибка обработки tickers сообщения: {e}")


    # async def check_and_manage_position(self, symbol):
    #     try:
    #         pos = self.ws_positions.get(symbol)
    #         logger.info(f"[DEBUG] check_and_manage_position() вызвана для {symbol} — pos={pos}")

    #         if not pos or float(pos.get('size', 0)) <= 0:
    #             logger.warning(f"[DEBUG] Позиция неактивна или отсутствует для {symbol}: {pos}")
    #             return

    #         mark_price = self.latest_mark_prices.get(symbol)

    #         # Теперь извлекаем entry_price из avg_price, а не из entryPrice
    #         try:
    #             entry_price = float(pos.get("avg_price") or 0)
    #         except (TypeError, ValueError):
    #             entry_price = 0.0

    #         # Ждём появления mark_price
    #             if entry_price <= 0 or mark_price is None:
    #                 logger.warning(f"[check_and_manage_position] entry/mark price отсутствует: entry={entry_price}, mark={mark_price}")
    #                 return
    #             logger.warning(f"[check_and_manage_position] Нет цен для {symbol}. Повтор через 1с.")
    #             await asyncio.sleep(1)
    #             mark_price = self.latest_mark_prices.get(symbol)
    #             if mark_price is None or entry_price <= 0:
    #                 logger.warning(f"[check_and_manage_position] Повторно пропускаем {symbol} — mark_price={mark_price}, entry_price={entry_price}")
    #                 return

    #         self.latest_mark_prices[symbol] = mark_price
    #         self.latest_entry_prices[symbol] = entry_price

    #         if symbol in self.recently_closed and time.time() - self.recently_closed[symbol] < 30:
    #             logger.info(f"[WebSocket Limiter] Пропуск повторного открытия {symbol} (только что была закрыта)")
    #             return

    #         leverage = Decimal('10')
    #         side = pos.get("side", "").lower()
    #         if entry_price is None or mark_price is None or entry_price == 0:
    #             logger.warning(f"[check_and_manage_position] Пропускаем {symbol} — некорректные цены: entry_price={entry_price}, mark_price={mark_price}")
    #             return
    #         pnl_ratio = (mark_price - entry_price) / entry_price if side == "buy" else (entry_price - mark_price) / entry_price
    #         leveraged_pnl_percent = pnl_ratio * float(leverage) * 100

    #         if symbol in self.open_positions:
    #             self.open_positions[symbol]['profit_perc'] = leveraged_pnl_percent
    #             self.open_positions[symbol]['current_price'] = mark_price

    #         logger.info(f"[DEBUG] PnL: {symbol}, side={side}, entry={entry_price}, mark={mark_price}, PnL={leveraged_pnl_percent:.2f}%")

    #         if leveraged_pnl_percent >= 5 and not pos.get('trailing_stop_set'):
    #             logger.info(f"[Trailing] Установка трейлинг-стопа для {symbol}")
    #             try:
    #                 await self.apply_custom_trailing_stop(symbol, pos, leveraged_pnl_percent, side)
    #                 if symbol in self.open_positions:
    #                     self.open_positions[symbol]['trailing_stop_set'] = True
    #                 if symbol in self.ws_positions:
    #                     self.ws_positions[symbol]['trailing_stop_set'] = True
    #             except Exception as e:
    #                 logger.error(f"Ошибка установки трейлинг-стопа: {e}")

    #     except Exception as e:
    #         logger.exception(f"Ошибка управления позицией {symbol}: {e}")


    async def subscribe_to_ticker(self, symbol: str):
        """Подписка на обновления тикера по символу через WebSocket V5 (pybit)."""
        try:
            if not self.ws_ticker_stream:
                logger.warning("[subscribe_to_ticker] ws_ticker_stream не инициализирован")
                return

            if not isinstance(symbol, str):
                logger.error(f"[subscribe_to_ticker] symbol НЕ строка: {symbol} ({type(symbol)})")
                return

            if symbol in self.active_position_subscriptions:
                logger.debug(f"[subscribe_to_ticker] Уже подписан на {symbol}")
                return

            logger.info(f"[subscribe_to_ticker] Подписка на тикер: {symbol}")

            # Подписываемся — три позиционных аргумента: "tickers", сам symbol, и колбэк
            self.ws_ticker_stream.subscribe(
                topic="tickers",
                symbol=symbol,               # или symbol=[symbol], если хотите список
                callback=self.handle_ticker_update
            )

            # Помечаем локально, что подписка активна
            self.active_position_subscriptions.add(symbol)
            logger.info(f"[subscribe_to_ticker] Успешно подписан на канал 'tickers', символ: {symbol}")

        except Exception as e:
            logger.exception(f"[subscribe_to_ticker] Ошибка подписки на {symbol}: {e}")

    async def update_tickers_subscription(self):
        """
        Собирает список символов из self.open_positions (или откуда угодно) 
        и подписывается на канал "tickers" одним вызовом.
        
        Если уже была подписка, отписываемся от всего канала "tickers" 
        перед повторной подпиской.
        """
        if not self.ws_ticker_stream:
            logger.warning("[update_tickers_subscription] ws_ticker_stream не инициализирован")
            return

        # Собираем все символы открытых позиций в список
        async with self.open_positions_lock:
            symbols_to_listen = list(self.open_positions.keys())  # Например: ["BTCUSDT", "ETHUSDT", ...]

        # Отписываемся от канала "tickers", если были подписаны
        try:
            await self.ws_ticker_stream.unsubscribe(topic="tickers")
        except Exception as e:
            # Если отписка вернёт ошибку (например, "Not subscribed"), просто игнорируем
            logger.debug(f"[update_tickers_subscription] Ошибка при unsubscribe: {e}")

        if not symbols_to_listen:
            logger.info("[update_tickers_subscription] Нет открытых позиций => не подписываемся на 'tickers'")
            return

        try:
            logger.info(f"[update_tickers_subscription] Подписываемся на: {symbols_to_listen}")
            self.ws_ticker_stream.subscribe(
                topic="tickers",
                symbol=symbols_to_listen,            # список символов
                callback=self.handle_ticker_update   # ваш обработчик
            )
            logger.info("[update_tickers_subscription] Успешно подписались на канал tickers для символов: %s", symbols_to_listen)
        except Exception as e:
            logger.exception("[update_tickers_subscription] Ошибка при subscribe: %s", e)

    async def unsubscribe_from_ticker(self, symbol: str):
        if self.ws and symbol in self.active_position_subscriptions:
            await self.ws.unsubscribe(f"tickers.{symbol}")
            self.active_position_subscriptions.remove(symbol)
            logger.info(f"[Ticker Unsubscribe] Отписался от tickers.{symbol}")


    # async def open_averaging_position_all(self, symbol, volume_usdt: Decimal):
    #     async with self.open_positions_lock:
    #         pos = self.open_positions.get(symbol)
    #         if not pos:
    #             return
    #         side = pos['side']
    #         current_volume = Decimal(str(pos['position_volume']))
        
    #     # Открываем позицию с удвоенным объемом
    #     new_volume = current_volume * Decimal('2')
    #     if new_volume > self.MAX_AVERAGING_VOLUME:
    #         new_volume = self.MAX_AVERAGING_VOLUME
        
    #     logger.info(f"Усреднение {symbol}: {new_volume} USDT")
    #     await self.open_position(symbol, side, new_volume, reason="Averaging")

    # async def sync_ws_to_open_positions(self):
    #     """Принудительная синхронизация данных WebSocket с локальным хранилищем"""
    #     async with self.open_positions_lock:
    #         current_symbols = set(self.ws_positions.keys())

    #         for sym in list(self.open_positions.keys()):
    #             if sym not in current_symbols:
    #                 recently_closed_ts = self.recently_closed.get(sym, 0)
    #                 if time.time() - recently_closed_ts > 5:
    #                     logger.info(f"[Sync] Удаляю {sym} из open_positions — не найден в ws_positions и не недавно закрыт")
    #                     del self.open_positions[sym]
    #                 else:
    #                     logger.info(f"[Sync] Пропущено удаление {sym} — недавно закрыта позиция")

    #         for sym, ws_pos in self.ws_positions.items():
    #             try:
    #                 size = float(ws_pos.get("size", 0))
    #                 if size > 0:
    #                     self.open_positions[sym] = ws_pos.copy()
    #                     logger.debug(f"[Sync] Обновлена позиция {sym} в open_positions: size={size}")
    #             except Exception as e:
    #                 logger.warning(f"[Sync] Ошибка парсинга позиции {sym}: {e}")

    # async def unsubscribe_from_position_stream(self, symbol):
    #     try:
    #         if self.ws:
    #             await self.ws.unsubscribe(f"position.{symbol}")
    #             logger.info(f"Отписался от позиции {symbol}")
    #     except Exception as e:
    #         logger.error(f"Ошибка отписки: {e}")

    async def safe_reconnect(self):
        """Улучшенное переподключение с безопасной очисткой WebSocket"""
        max_retries = 5
        delay = 2
        for attempt in range(max_retries):
            try:
                logger.warning(f"[{self.user_id}] Попытка переподключения ({attempt+1}/{max_retries})...")

                # Закрываем ws_ticker_stream если активен
                if self.ws_ticker_stream:
                    try:
                        if hasattr(self.ws_ticker_stream, "exit"):
                            await self.ws_ticker_stream.exit()
                        elif hasattr(self.ws_ticker_stream, "close"):
                            close_fn = self.ws_ticker_stream.close
                            if asyncio.iscoroutinefunction(close_fn):
                                await close_fn()
                            else:
                                close_fn()
                        logger.info(f"[{self.user_id}] ws_ticker_stream закрыт")
                    except Exception as e:
                        logger.warning(f"[{self.user_id}] Ошибка закрытия ws_ticker_stream: {e}")
                    self.ws_ticker_stream = None
                    self.active_position_subscriptions.clear()

                # Переинициализируем соединение
                await self.init_ticker_websocket()
                await self.update_tickers_subscription()

                logger.info(f"[{self.user_id}] Успешное переподключение ✅")
                return

            except Exception as e:
                logger.error(f"[{self.user_id}] Ошибка переподключения ({attempt+1}): {e}")
                await asyncio.sleep(delay)
                delay *= 2  # экспоненциальная задержка

        logger.critical(f"[{self.user_id}] ❌ Переподключение не удалось после {max_retries} попыток")
        self.active = False

    # async def shutdown(self):
    #     """Корректное завершение работы"""
    #     self.active = False
    #     if self.ws:
    #         await self.ws.exit()
    #     logger.info(f"[{self.user_id}] WebSocket отключен")

    # async def log_positions_status(self):
    #     async with self.open_positions_lock:
    #         for symbol, pos in self.open_positions.items():
    #             pnl = pos.get('profit_perc', 0)
    #             logger.info(
    #                 f"Позиция {symbol}: {pos['side']} "
    #                 f"Объем: {pos['position_volume']} USDT "
    #                 f"PnL: {pnl}%"
    #             )

    # async def start_websocket_listener(self):
    #     try:
    #         if not self.ws:
    #             await self.init_websocket()

    #         logger.info(f"[WS] Старт слушателя WebSocket для user_id={self.user_id}")
    #         resubscribe_interval = 60  # каждые 60 сек переподписка
    #         last_resub_time = time.time()

    #         while self.active:
    #             await asyncio.sleep(1)

    #             # Проверка соединения
    #             is_connected = await self.check_connection()
    #             if not is_connected:
    #                 continue

    #             # Необязательно: пере-подписка на каналы (если нужно)
    #             # if time.time() - last_resub_time > resubscribe_interval:
    #             #     try:
    #             #         logger.info(f"[WS] Пере-подписка на каналы для user_id={self.user_id}")
    #             #         await self.ws.subscribe("position", callback=self.handle_position_update)
    #             #         await self.ws.subscribe("order", callback=self.handle_order_update)
    #             #         last_resub_time = time.time()
    #             #     except Exception as e:
    #             #         logger.warning(f"[WS] Ошибка при пере-подписке: {e}")

    #             try:
    #                 if hasattr(self.ws, "ping"):
    #                     await asyncio.to_thread(self.ws.ping)
    #             except Exception as e:
    #                 logger.debug(f"[WS Ping] Ошибка: {e}")

    #     except Exception as e:
    #         logger.error(f"[WebSocket Listener] Критическая ошибка: {e}")
    #         asyncio.create_task(self.safe_reconnect())


    # async def check_connection(self) -> bool:
    #     """Проверка активности соединения"""
    #     try:
    #         if not self.ws or not self.ws.is_connected():
    #             logger.warning("Соединение не активно")
    #             await self.safe_reconnect()
    #             return False
    #         return True
    #     except Exception as e:
    #         logger.exception("[check_connection] Ошибка проверки соединения")
    #         return False


    async def periodic_position_sync(self):
        """Периодическая синхронизация с REST API"""
        while self.active:
            try:
                positions = await asyncio.to_thread(self.get_exchange_positions)
                async with self.open_positions_lock:
                    self.open_positions = {
                        sym: pos for sym, pos in (positions or {}).items() 
                        if pos.get('size', 0) > 0
                    }
                now = time.time()
                for symbol, ts in list(self.awaiting_position_update.items()):
                    delay = now - ts
                    if delay > 30:
                        logger.warning(f"[WS Delay] ⚠️ {symbol}: Нет подтверждения от WebSocket более 30 сек. Удаляю из ожидания.")
                        del self.awaiting_position_update[symbol]
                    elif delay > 15:
                        logger.info(f"[WS Delay] ⏳ {symbol}: Нет подтверждения от WebSocket уже {delay:.1f} сек.")
                    if now - ts > 15:  # 15 секунд без подтверждения
                        logger.warning(f"[WS] Нет подтверждения позиции {symbol} от WebSocket в течение 15 сек.")
                        del self.awaiting_position_update[symbol]
                logger.info("Позиции синхронизированы через REST API")
                await asyncio.sleep(60)  # Каждую минуту
            except Exception as e:
                logger.error(f"Ошибка синхронизации позиций: {e}")
                await asyncio.sleep(10)
                
    # -------------------- Методы расчёта SuperTrend --------------------

    async def calculate_supertrend_universal(self, df: pd.DataFrame, length: int = 10, multiplier: float = 3.0, use_wilder_atr: bool = False) -> pd.DataFrame:
        ###
        # Универсальный метод расчёта SuperTrend (Bybit/TradingView).
        # Параметры:
        #  - df: DataFrame со столбцами ['openPrice', 'highPrice', 'lowPrice', 'closePrice']
        #   - length: период для ATR (по умолчанию 10)
        #   - multiplier: множитель ATR (по умолчанию 3.0)
        #   - use_wilder_atr: True/False, если True — используем RMA (сглаживание Уайлдера)
        #                     для расчёта ATR, иначе — простое скользящее среднее.
        # Возвращает:
        #   DataFrame c колонками:
        #     'final_ub', 'final_lb', 'supertrend' (и все исходные колонки df).
        ###
        try:
            if df.empty:
                return pd.DataFrame()

            def extend_value(current_value, previous_value):
                # «Протягиваем» текущее значение, если оно стало NaN или 0
                return previous_value if pd.isna(current_value) or current_value == 0 else current_value

            # 1) Приводим столбцы к числам, избавляемся от 0/NaN
            # for col in ["highPrice", "lowPrice", "closePrice"]:
            #     df[col] = (
            #         pd.to_numeric(df[col], errors="coerce")
            #         .replace(0, np.nan)
            #         .ffill()
            #     )
            # df.bfill(inplace=True)

            for col in ["highPrice", "lowPrice", "closePrice"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 2) True Range
            df["prev_close"] = df["closePrice"].shift(1)
            df["tr1"] = df["highPrice"] - df["lowPrice"]
            df["tr2"] = (df["highPrice"] - df["prev_close"]).abs()
            df["tr3"] = (df["lowPrice"] - df["prev_close"]).abs()
            df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

            # 3) Расчёт ATR
            if use_wilder_atr:
                # RMA (Wilder's smoothing) версия ATR
                df["atr"] = self.wilder_rma(df["true_range"], length)
            else:
                # Простое скользящее среднее
                df["atr"] = df["true_range"].rolling(window=length, min_periods=1).mean()

            # 4) Базовые верхняя/нижняя границы
            hl2 = (df["highPrice"] + df["lowPrice"]) / 2
            df["basic_ub"] = hl2 + multiplier * df["atr"]
            df["basic_lb"] = hl2 - multiplier * df["atr"]

            # 5) Итоговые финальные границы (final_ub/final_lb)
            df["final_ub"] = df["basic_ub"].copy()
            df["final_lb"] = df["basic_lb"].copy()

            for i in range(1, len(df)):
                prev_idx = df.index[i - 1]
                curr_idx = df.index[i]

                # --- Верхняя граница ---
                if (
                    df.loc[curr_idx, "basic_ub"] < df.loc[prev_idx, "final_ub"]
                    or df.loc[prev_idx, "closePrice"] > df.loc[prev_idx, "final_ub"]
                ):
                    df.loc[curr_idx, "final_ub"] = df.loc[curr_idx, "basic_ub"]
                else:
                    df.loc[curr_idx, "final_ub"] = df.loc[prev_idx, "final_ub"]

                # --- Нижняя граница ---
                if (
                    df.loc[curr_idx, "basic_lb"] > df.loc[prev_idx, "final_lb"]
                    or df.loc[prev_idx, "closePrice"] < df.loc[prev_idx, "final_lb"]
                ):
                    df.loc[curr_idx, "final_lb"] = df.loc[curr_idx, "basic_lb"]
                else:
                    df.loc[curr_idx, "final_lb"] = df.loc[prev_idx, "final_lb"]

                # «Протягиваем» значения
                df.loc[curr_idx, "final_ub"] = extend_value(
                    df.loc[curr_idx, "final_ub"],
                    df.loc[prev_idx, "final_ub"]
                )
                df.loc[curr_idx, "final_lb"] = extend_value(
                    df.loc[curr_idx, "final_lb"],
                    df.loc[prev_idx, "final_lb"]
                )

            # 6) Линия SuperTrend — если closePrice выше final_ub, берём final_lb, иначе final_ub
            df["supertrend"] = df["final_ub"]
            df.loc[df["closePrice"] > df["final_ub"], "supertrend"] = df["final_lb"]

            # 7) Доп. проход для устранения возможных NaN/0
            for i in range(1, len(df)):
                prev_idx = df.index[i - 1]
                curr_idx = df.index[i]
                df.loc[curr_idx, "supertrend"] = extend_value(
                    df.loc[curr_idx, "supertrend"],
                    df.loc[prev_idx, "supertrend"]
                )

            return df
        except Exception as e:
            logger.exception(f"Ошибка в calculate_supertrend_universal: {e}")
        return pd.DataFrame()

    async def calculate_supertrend_beacon(self, df: pd.DataFrame) -> pd.DataFrame:
        """Считаем SuperTrend (length=50, multiplier=3)."""
        try:
            return await self.calculate_supertrend_universal(
                df,
                length=50,
                multiplier=3.0,
                use_wilder_atr=False
            )
        except Exception as e:
            logger.exception(f"[calculate_supertrend_beacon] Ошибка: {e}")
            return pd.DataFrame()

    def wilder_rma(sel, series: pd.Series, length: int) -> pd.Series:
        """
        Реализация сглаживания по Уайлдеру (Wilder's Smoothing), часто называемого RMA.
        Подходит для расчёта ATR (согласно формуле из TradingView).
        
        1) Для первых 'length' значений берём простое среднее => это стартовое значение RMA.
        2) Далее (для i >= length) применяем рекурсивную формулу:
            RMA[i] = (RMA[i-1] * (length - 1) + series[i]) / length

        Возвращает Series со сглаженными значениями.
        """
        if series.empty:
            return series

        rma_vals = series.copy().values  # numpy массив
        # Этап 1: первая точка (после накопления length элементов) — среднее
        # Если length больше реальной длины series, берём доступную часть
        window_size = min(length, len(series))
        first_val = np.mean(rma_vals[:window_size])
        
        # Устанавливаем RMA для всех точек до window_size
        # — но корректно чтобы не «портить» начальные значения
        for i in range(window_size):
            rma_vals[i] = first_val

        # Этап 2: рекурсивная формула
        for i in range(window_size, len(series)):
            rma_vals[i] = ((rma_vals[i - 1] * (length - 1)) + rma_vals[i]) / length

        return pd.Series(rma_vals, index=series.index)

    async def check_st50_multi_tf(self, symbol: str) -> Optional[str]:
        """Проверяет направление тренда по ST50 на минутном и пятиминутном таймфреймах.
        Если оба дают одинаковый тренд, возвращает 'bullish' или 'bearish', иначе None."""
        # Получаем минутные данные и вычисляем ST50
        df_1m = await self.get_historical_data_for_trading(symbol, interval="1", limit=205)
        if df_1m.empty:
            return None
        st_df_1m = await self.calculate_supertrend_universal(df_1m.copy(), length=50, multiplier=3.0, use_wilder_atr=False)
        if st_df_1m.empty:
            return None
        last_row_1m = st_df_1m.iloc[-1]
        price_1m = last_row_1m.get("closePrice")
        st_value_1m = last_row_1m.get("supertrend")
        if price_1m is None or st_value_1m is None:
            return None
        trend_1m = "bullish" if price_1m > st_value_1m else "bearish"

        # Получаем пятиминутные данные и вычисляем ST50
        df_5m = await self.get_historical_data_for_trading(symbol, interval="5", limit=205)
        if df_5m.empty:
            return None
        st_df_5m = await self.calculate_supertrend_universal(df_5m.copy(), length=50, multiplier=3.0, use_wilder_atr=False)
        if st_df_5m.empty:
            return None
        last_row_5m = st_df_5m.iloc[-1]
        price_5m = last_row_5m.get("closePrice")
        st_value_5m = last_row_5m.get("supertrend")
        if price_5m is None or st_value_5m is None:
            return None
        trend_5m = "bullish" if price_5m > st_value_5m else "bearish"

        if trend_1m == trend_5m:
            return trend_1m
        return None


    async def calculate_supertrend_bybit_8_1(self, df: pd.DataFrame, length=3, multiplier=1.0) -> pd.DataFrame:
        try:
            if df.empty:
                return pd.DataFrame()

            def extend_value(current_value, previous_value):
                return previous_value if pd.isna(current_value) or current_value == 0 else current_value

            for col in ["highPrice", "lowPrice", "closePrice"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan).ffill()
            df.bfill(inplace=True)
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
                if (df.loc[df.index[i], "basic_ub"] < df.loc[df.index[i-1], "final_ub"]) or (df.loc[df.index[i-1], "closePrice"] > df.loc[df.index[i-1], "final_ub"]):
                    df.loc[df.index[i], "final_ub"] = df.loc[df.index[i], "basic_ub"]
                else:
                    df.loc[df.index[i], "final_ub"] = df.loc[df.index[i-1], "final_ub"]

                if (df.loc[df.index[i], "basic_lb"] > df.loc[df.index[i-1], "final_lb"]) or (df.loc[df.index[i-1], "closePrice"] < df.loc[df.index[i-1], "final_lb"]):
                    df.loc[df.index[i], "final_lb"] = df.loc[df.index[i], "basic_lb"]
                else:
                    df.loc[df.index[i], "final_lb"] = df.loc[df.index[i-1], "final_lb"]

                df.loc[df.index[i], "final_ub"] = extend_value(df.loc[df.index[i], "final_ub"], df.loc[df.index[i-1], "final_ub"])
                df.loc[df.index[i], "final_lb"] = extend_value(df.loc[df.index[i], "final_lb"], df.loc[df.index[i-1], "final_lb"])

            df["supertrend"] = df["final_ub"].copy()
            df.loc[df["closePrice"] > df["final_ub"], "supertrend"] = df["final_lb"]

            for i in range(1, len(df)):
                df.loc[df.index[i], "supertrend"] = extend_value(df.loc[df.index[i], "supertrend"], df.loc[df.index[i-1], "supertrend"])

            return df
        except Exception as e:
            logger.exception(f"Ошибка в calculate_supertrend_bybit_8_1: {e}")
            return pd.DataFrame()


    async def get_last_row(self, symbol: str):
        df = await self.get_historical_data_for_trading(symbol, interval="1", limit=1)
        if df.empty:
            return None
        return df.iloc[-1]

    async def generate_drift_table_from_history(self, drift_history: dict, top_n: int = 15) -> str:
        if not drift_history:
            return "Нет данных для дрифта."
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
        table = Table(title="Drift History", expand=True, box=box.ROUNDED)
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Anomaly", justify="right", style="magenta")
        table.add_column("Dir", justify="center")
        for sym, strength, direction in rows:
            arrow = "🔴" if direction == "вверх" else "🟢"
            table.add_row(sym, f"{strength:.3f}", arrow)
        console.print(table)
        result_text = console.export_text()
        return result_text

    async def generate_model_table_from_csv_no_time(self, user_id: int, csv_path: str = "model_predictions_log.csv", last_n: int = 200) -> str:
        def _read_csv():
            if not os.path.isfile(csv_path):
                return None
            df = pd.read_csv(csv_path, low_memory=False)
            return df
        df = await asyncio.to_thread(_read_csv)
        if df is None or df.empty:
            return "Нет данных для данного пользователя."
        df = df[df["user_id"] == user_id]
        if df.empty:
            return "Нет данных для данного пользователя."
        df.sort_values("timestamp", inplace=True)
        df_tail = df.tail(last_n)
        console = Console(record=True, force_terminal=True, width=100)
        table = Table(title="Model Predictions", expand=True, box=box.ROUNDED)
        table.add_column("Symbol", style="cyan")
        table.add_column("Pred", justify="center")
        table.add_column("p(Buy)", justify="right", style="bold green")
        table.add_column("p(Hold)", justify="right")
        table.add_column("p(Sell)", justify="right", style="bold red")
        for _, row in df_tail.iterrows():
            sym = str(row.get("symbol", ""))
            pred = str(row.get("prediction", "NA"))
            try:
                p_buy = float(row.get("prob_buy", 0.0))
                p_hold = float(row.get("prob_hold", 0.0))
                p_sell = float(row.get("prob_sell", 0.0))
            except Exception:
                p_buy = p_hold = p_sell = 0.0
            table.add_row(sym, pred, f"{p_buy:.3f}", f"{p_hold:.3f}", f"{p_sell:.3f}")
        console.print(table)
        return console.export_text()

    async def publish_drift_and_model_tables(self, trading_bot) -> None:
        if not telegram_bot:
            logger.info("[publish_drift_and_model_tables] Telegram bot не инициализирован => пропуск.")
            return
        drift_str = await self.generate_drift_table_from_history(trading_bot.drift_history, top_n=10)
        if drift_str.strip():
            msg = f"```\n{drift_str}\n```"
            await telegram_bot.send_message(
                chat_id=trading_bot.user_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN_V2
            )
        else:
            logger.info("[DRIFT] Таблица дрифта пуста => пропуск.")
        model_str = await self.generate_model_table_from_csv_no_time(trading_bot.user_id, csv_path="model_predictions_log.csv", last_n=10)
        if model_str.strip():
            msg = f"```\n{model_str}\n```"
            await telegram_bot.send_message(
                chat_id=trading_bot.user_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN_V2
            )
        else:
            logger.info("[MODEL] Таблица модели пуста => пропуск.")

    async def train_and_load_model(self, csv_path="historical_data_for_model_5m.csv"):
        def _train():
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
                    df_sym = asyncio.run(self.prepare_features_for_model(df_sym))
                    if df_sym.empty:
                        continue
                    df_sym = asyncio.run(self.make_multiclass_target_for_model(df_sym, horizon=1, threshold=Decimal("0.0025")))
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
                if len(data) < self.MIN_SAMPLES_FOR_TRAINING:
                    logger.warning(f"Слишком мало строк: {len(data)} < {self.MIN_SAMPLES_FOR_TRAINING}.")
                    return None
                feature_cols = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
                data = data.dropna(subset=feature_cols)
                if data.empty:
                    logger.warning("Все NaN => нет данных.")
                    return None
                X = data[feature_cols].values
                y = data["target"].astype(int).values
                if len(X) < 50:
                    logger.warning(f"[train_and_load_model] Слишком мало данных для обучения (всего {len(X)}).")
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
                logger.info(f"[train_and_load_model] Обучение завершено, CV max_accuracy={best_acc:.4f}")
                joblib.dump(pipeline, self.MODEL_FILENAME)
                logger.info(f"[train_and_load_model] Модель сохранена в {self.MODEL_FILENAME}")
                return pipeline
            except Exception as e:
                logger.exception(f"Ошибка train_and_load_model: {e}")
                return None
        model = await asyncio.to_thread(_train)
        return model

    def load_model(self):
        try:
            model = joblib.load(self.MODEL_FILENAME)
            return model
        except (ModuleNotFoundError, ImportError):
            logger.warning("Не удалось загрузить модель. Будет создана новая.")
            return asyncio.run(self.train_and_load_model())

    async def maybe_retrain_model(self):
        new_model = await self.train_and_load_model(csv_path="historical_data_for_model_5m.csv")
        if new_model:
            self.current_model = new_model
            logger.info(f"[maybe_retrain_model] Пользователь {self.user_id}: Модель успешно обновлена.")

    def get_usdt_pairs(self):
        try:
            tickers_resp = self.session.get_tickers(symbol=None, category="linear")
            if "result" not in tickers_resp or "list" not in tickers_resp["result"]:
                logger.error("[get_usdt_pairs] Некорректный ответ при get_tickers.")
                return []
            tickers_data = tickers_resp["result"]["list"]
            inst_resp = self.session.get_instruments_info(category="linear")
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

    async def get_historical_data_for_model(self, symbol, interval="1", limit=200, from_time=None):
        params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
        if from_time:
            params["from"] = from_time
        def _get_kline():
            return self.session.get_kline(**params)
        try:
            resp = await asyncio.to_thread(_get_kline)
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
            df.rename(columns={"open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"}, inplace=True)
            df = df[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice"]]
            df.dropna(subset=["closePrice"], inplace=True)
            logger.debug(f"[get_historical_data_for_model] {symbol}: получено {len(df)} свечей.")
            return df
        except Exception as e:
            logger.exception(f"Ошибка get_historical_data_for_model({symbol}): {e}")
            return pd.DataFrame()

    async def adjust_quantity(self, symbol: str, raw_qty: float) -> float:
        info = self.get_symbol_info(symbol)
        if not info:
            logger.warning(f"[adjust_quantity] get_symbol_info({symbol}) вернул None, qty={raw_qty}")
            return 0.0
        lot = info.get("lotSizeFilter", {})
        min_qty = Decimal(str(lot.get("minOrderQty", "0")))
        qty_step = Decimal(str(lot.get("qtyStep", "1")))
        max_qty = Decimal(str(lot.get("maxOrderQty", "9999999")))
        min_order_value = Decimal(str(info.get("minOrderValue", 5)))
        last_price = await self.get_last_close_price(symbol)
        if not last_price or last_price <= 0:
            logger.warning(f"[adjust_quantity] last_price={last_price} недопустим, qty={raw_qty}")
            return 0.0
        price_dec = Decimal(str(last_price))
        dec_qty = Decimal(str(raw_qty))
        adj_qty = (dec_qty // qty_step) * qty_step
        if adj_qty < min_qty:
            logger.info(f"[adjust_quantity] {symbol}: adj_qty={adj_qty} < min_qty={min_qty} => 0")
            return 0.0
        if adj_qty > max_qty:
            adj_qty = max_qty
        order_value = adj_qty * price_dec
        if order_value < min_order_value:
            needed_qty = (min_order_value / price_dec).quantize(qty_step, rounding="ROUND_UP")
            if needed_qty > max_qty or needed_qty < min_qty:
                logger.warning(f"[adjust_quantity] {symbol}: needed_qty не удовлетворяет условиям")
                return 0.0
            adj_qty = needed_qty
        final_qty = float(adj_qty)
        logger.info(f"[adjust_quantity] {symbol}: raw_qty={raw_qty}, price={price_dec}, adj_qty={final_qty}")
        return final_qty

    async def monitor_feature_drift_per_symbol(self, symbol, new_data, ref_data, feature_cols, drift_csv="feature_drift.csv", threshold=0.5):
        try:
            if new_data.empty:
                logger.info(f"[DRIFT] {symbol}: new_data пуст")
                return False, 0.0, "нет данных"
            if ref_data.empty:
                split_point = len(new_data) // 2
                ref_data = new_data.iloc[:split_point].copy()
                new_data = new_data.iloc[split_point:].copy()
            if new_data.empty or ref_data.empty:
                return False, 0.0, "недостаточно данных"
            mean_new = new_data[feature_cols].mean().mean()
            mean_ref = ref_data[feature_cols].mean().mean()
            direction = "вверх" if mean_new > mean_ref else "вниз"
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
            ts_str = datetime.datetime.utcnow().isoformat()
            async with self.history_lock:
                self.drift_history[symbol].append((ts_str, anomaly_strength, direction))
                if len(self.drift_history[symbol]) > 10:
                    self.drift_history[symbol].pop(0)
            logger.info(f"[DRIFT] {symbol}: strength={anomaly_strength:.3f}, direction={direction}, is_anomaly={is_anomaly}")
            return is_anomaly, anomaly_strength, direction
        except Exception as e:
            logger.exception(f"[DRIFT] Ошибка в monitor_feature_drift_per_symbol для {symbol}: {e}")
            return False, 0.0, "ошибка"

    async def check_drift_for_symbol(self, symbol: str):
        df_trading = await self.get_historical_data_for_trading(symbol, interval="1", limit=200)
        feature_cols = ["openPrice", "closePrice", "highPrice", "lowPrice"]
        result = await self.monitor_feature_drift_per_symbol(symbol, df_trading, pd.DataFrame(), feature_cols, threshold=0.5)
        is_anomaly, strength, direction = result
        if is_anomaly:
            logger.info(f"[Drift] {symbol}: аномалия обнаружена, strength={strength:.3f}, direction={direction}")
        return result

    async def check_and_set_trailing_stop(self):
        if not self.TRAILING_STOP_ENABLED:
            return
        try:
            # Читаем копию открытых позиций через асинхронную блокировку
            async with self.open_positions_lock:
                positions_copy = dict(self.open_positions)
            threshold_roi = Decimal("5.0")
            default_leverage = Decimal("10")
            for sym, pos in positions_copy.items():
                if pos.get("trailing_stop_set"):
                    continue
                side = pos["side"]
                entry_price = Decimal(str(pos["avg_price"]))
                current_price = await self.get_last_close_price(sym)
                if current_price is None:
                    continue
                cp = Decimal(str(current_price))
                if side.lower() == "buy":
                    ratio = (cp - entry_price) / entry_price
                else:
                    ratio = (entry_price - cp) / entry_price
                # Вычисляем показатель прибыльности с плечом
                leveraged_pnl_percent = ratio * default_leverage * Decimal("100")
                async with self.open_positions_lock:
                    if sym in self.open_positions:
                        self.open_positions[sym]['profit_perc'] = (ratio * self.PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
                if leveraged_pnl_percent >= threshold_roi:
                    if not pos.get("trailing_stop_set", False):
                        logger.info(f"[Trailing Stop] {sym}: Уровень достигнут (leveraged PnL = {leveraged_pnl_percent}%). Устанавливаю трейлинг-стоп.")
                        await self.set_trailing_stop(sym, pos["size"], self.TRAILING_GAP_PERCENT, side)
        except Exception as e:
            logger.exception(f"Ошибка check_and_set_trailing_stop: {e}")

    async def check_and_set_insurance_stop_loss(self):
        """Устанавливает страховой стоп-лосс, если рыночная прибыль с учетом плеча (10х) достигла 2% ROI.
        После установки флаг pos[\"insurance_stop_set\"] выставляется, чтобы не повторять установку."""
        async with self.open_positions_lock:
            for symbol, pos in self.open_positions.items():
                if pos.get("insurance_stop_set"):
                    continue
                entry_price = Decimal(str(pos.get("avg_price", "0")))
                current_price = await self.get_last_close_price(symbol)
                if not current_price or entry_price <= 0:
                    continue
                cp = Decimal(str(current_price))
                if pos["side"].lower() == "buy":
                    ratio = (cp - entry_price) / entry_price
                else:
                    ratio = (entry_price - cp) / entry_price
                leveraged_roi = ratio * Decimal("10") * Decimal("100")
                if leveraged_roi >= Decimal("3"):
                    # Устанавливаем стоп‑лосс в точку входа (на уровне цены открытия позиции)
                    stop_price = entry_price * Decimal("1.01")
                    await self.set_fixed_stop_loss(symbol, pos.get("size"), pos["side"], stop_price)                    

    async def set_stop_loss_to_fast_st(self, symbol: str, side: str):
        df = await self.get_historical_data_for_trading(symbol, interval="1", limit=205)
        if df.empty:
            logger.warning(f"[set_stop_loss_to_fast_st] {symbol}: недостаточно данных для расчёта.")
            return

        st_df = await self.calculate_supertrend_universal(df.copy(), length=8, multiplier=2.0, use_wilder_atr=False)
        if st_df.empty:
            logger.warning(f"[set_stop_loss_to_fast_st] {symbol}: не удалось рассчитать fast supertrend.")
            return

        new_fast_st = Decimal(str(st_df["supertrend"].iloc[-1]))
        logger.info(f"[set_stop_loss_to_fast_st] {symbol}: вычислено новое значение fast ST = {new_fast_st}")

        async with self.open_positions_lock:
            current_stop = self.open_positions.get(symbol, {}).get("stop_loss")
        logger.info(f"[set_stop_loss_to_fast_st] {symbol}: текущее значение стоп-лосса = {current_stop}")

        updated = False
        if current_stop is None:
            updated = True
        else:
            if side.lower() == "buy" and new_fast_st > current_stop:
                updated = True
            elif side.lower() == "sell" and new_fast_st < current_stop:
                updated = True

        if updated:
            logger.info(f"[set_stop_loss_to_fast_st] {symbol}: условие обновления выполнено, вызываю set_fixed_stop_loss с stop_price={new_fast_st}")
            await self.set_fixed_stop_loss(symbol, size=None, side=side, stop_price=new_fast_st)
            async with self.open_positions_lock:
                if symbol in self.open_positions:
                    self.open_positions[symbol]["stop_loss"] = new_fast_st
            logger.info(f"[set_stop_loss_to_fast_st] {symbol}: стоп-лосс обновлён до fast ST ({new_fast_st})")
            await self.log_trade(
                user_id=self.user_id,
                symbol=symbol,
                row=None,
                side=side,
                open_interest=None,
                action=f"Обновлён SL по fast ST: {new_fast_st}",
                result="TrailingStop",
                closed_manually=False
            )
        else:
            logger.info(f"[set_stop_loss_to_fast_st] {symbol}: новый fast ST {new_fast_st} не выгоднее текущего стоп-лосса {current_stop}")

    async def set_trailing_stop(self, symbol, size, trailing_gap_percent, side):
        try:
            pos_info = self.get_position_info(symbol, side)
            if not pos_info:
                logger.error(f"[set_trailing_stop] Нет позиции {symbol}/{side}")
                return
            pos_idx = pos_info.get("positionIdx")
            if not pos_idx:
                return
            avg_price = Decimal(str(pos_info.get("avgPrice", "0")))
            if avg_price <= 0:
                return
            trailing_distance_abs = (avg_price * Decimal(str(trailing_gap_percent))).quantize(Decimal("0.0000001"))
            dynamic_min = max(avg_price * Decimal("0.0000001"), self.MIN_TRAILING_STOP)
            if trailing_distance_abs < dynamic_min:
                logger.info(f"[set_trailing_stop] {symbol}: trailingStop={trailing_distance_abs} < {dynamic_min}, пропуск.")
                return
            # Выполняем синхронный API-вызов в потоке
            resp = await asyncio.to_thread(lambda: self.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="TrailingStop",
                qty=str(size),
                trailingStop=str(trailing_distance_abs),
                timeInForce="GoodTillCancel",
                positionIdx=pos_idx
            ))
            if resp:
                rc = resp.get("retCode")
                if rc == 0:
                    async with self.open_positions_lock:
                        if symbol in self.open_positions:
                            self.open_positions[symbol]["trailing_stop_set"] = True
                            leveraged_pnl_percent = self.open_positions.get(symbol, {}).get("profit_perc", Decimal("0"))
                            row = await self.get_last_row(symbol)
                            await self.log_trade(
                                user_id=self.user_id,
                                symbol=symbol,
                                row=None,
                                side=side,
                                open_interest=None,
                                action=f"Установлен каст.стоп, PnL={leveraged_pnl_percent}%",
                                result="TrailingStop",
                                closed_manually=False
                            )
                    logger.info(f"[set_trailing_stop] OK {symbol}")
                elif rc == 34040:
                    logger.info("[set_trailing_stop] not modified, retCode=34040.")
                else:
                    logger.error(f"[set_trailing_stop] Ошибка: {resp.get('retMsg')}")
        except Exception as e:
            logger.exception(f"[set_trailing_stop] {symbol}: {e}")


    async def set_fixed_stop_loss(self, symbol, size, side, stop_price):
        position_info = self.get_position_info(symbol, side)
        if not position_info:
            logger.error(f"[set_fixed_stop_loss] Нет позиции {symbol}/{side}")
            return
        pos_idx = position_info.get("positionIdx")
        if not pos_idx:
            return
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "positionIdx": pos_idx,
            "stopLoss": str(stop_price),
            "timeInForce": "GoodTillCancel",
        }
        try:
            resp = await asyncio.to_thread(lambda: self.session.set_trading_stop(**params))
            if resp.get("retCode") == 0:
                logger.info(f"[set_fixed_stop_loss] {symbol}: стоп-лосс выставлен на {stop_price}")
            elif resp.get("retCode") == 34040:
                logger.info(f"[set_fixed_stop_loss] {symbol}: стоп-лосс уже установлен на {stop_price} (not modified)")
            else:
                logger.error(f"[set_fixed_stop_loss] Ошибка: {resp.get('retMsg')}")
        except Exception as e:
            # Если исключение связано с 'not modified', можно обработать его здесь
            if "not modified" in str(e):
                logger.info(f"[set_fixed_stop_loss] {symbol}: стоп-лосс уже установлен (not modified)")
            else:
                logger.exception(f"[set_fixed_stop_loss] {symbol}: {e}")

    async def compute_trailing_gap_from_acceleration(self, symbol: str, interval="1", limit=10, scale=1.0, min_gap=0.1, max_gap=2.0):
        df = await self.get_historical_data_for_trading(symbol, interval=interval, limit=limit)
        if df.empty or len(df) < 5:
            return min_gap

        prices = df["closePrice"].values.astype(np.float64)

        # Первая производная — скорость
        velocity = np.gradient(prices)

        # Вторая производная — ускорение
        acceleration = np.gradient(velocity)

        acc = acceleration[-1]
        gap = abs(acc) * scale
        gap = max(min_gap, min(gap, max_gap))

        return round(gap, 4)

    # async def apply_custom_trailing_stop(self, symbol, pos, leveraged_pnl_percent, side):
    #     """
    #     Кастовый трейлинг-стоп:
    #     - включается, когда текущая (с учётом плеча) прибыль >= 5%;
    #     - стоп ставим на (leveraged_pnl - X)%, при этом не даём стопу опуститься обратно,
    #     т.е. если новая цель меньше предыдущей -- ничего не меняем.
    #     """
    #     START_CUSTOM_TRAIL = Decimal("5.0")
    #     TRAIL_OFFSET = Decimal("3.0")
    #     if leveraged_pnl_percent < START_CUSTOM_TRAIL:
    #         return
    #     desired_stop = leveraged_pnl_percent - TRAIL_OFFSET
    #     if desired_stop < Decimal("0"):
    #         desired_stop = Decimal("0")
    #     async with self.open_positions_lock:
    #         pos_in_bot = self.open_positions.get(symbol)
    #         if not pos_in_bot:
    #             return
    #         old_stop = pos_in_bot.get("custom_stop_loss_percent", Decimal("0"))
    #         if desired_stop <= old_stop:
    #             return
    #         pos_in_bot["custom_stop_loss_percent"] = desired_stop
    #     entry_price = Decimal(str(pos.get("avg_price", 0)))
    #     if entry_price <= 0:
    #         return
    #     leverage = Decimal("10")
    #     stop_ratio = desired_stop / Decimal("100") / leverage
    #     if side.lower() == "buy":
    #         stop_price = entry_price * (Decimal("1") - stop_ratio)
    #     else:
    #         stop_price = entry_price * (Decimal("1") + stop_ratio)
    #     logger.info(f"[CustomTrailingStop] {symbol}: тек. pnl={leveraged_pnl_percent}%, двигаем стоп на {desired_stop}% => цена {stop_price:.4f}")
    #     await self.set_fixed_stop_loss(symbol, pos["size"], side, stop_price)
    #     await self.log_trade(
    #         user_id=self.user_id,
    #         symbol=symbol,
    #         row=None,
    #         side=side,
    #         open_interest=None,
    #         action=f"Установлен каст.стоп, PnL={leveraged_pnl_percent}%",
    #         result="TrailingStop",
    #         closed_manually=False
    #     )
    
    # async def apply_custom_trailing_stop(self, symbol, pos, pnl_percent, side):
    #     """
    #     Устанавливает кастомный трейлинг-стоп вручную (в виде обычного stop loss), как только PnL превышает порог.
    #     """
    #     try:
    #         # Получаем текущую рыночную цену
    #         current_price = await self.get_last_close_price(symbol)
    #         if not current_price:
    #             logger.warning(f"[apply_custom_trailing_stop] Нет текущей цены для {symbol}")
    #             return

    #         entry_price = Decimal(str(pos["avg_price"]))
    #         cp = Decimal(str(current_price))
    #         gap = self.TRAILING_GAP_PERCENT_CUSTOM

    #         # Для лонга — стоп ниже текущей, для шорта — выше
    #         if side.lower() == "buy":
    #             stop_price = cp * (1 + gap)
    #         else:
    #             stop_price = cp * (1 - gap)

    #         stop_price = stop_price.quantize(Decimal("0.0001"))  # округление под биржу

    #         # Устанавливаем стоп через set_fixed_stop_loss
    #         await self.set_fixed_stop_loss(symbol, size=pos["size"], side=side, stop_price=stop_price)

    #         # Обновляем локальные данные
    #         async with self.open_positions_lock:
    #             if symbol in self.open_positions:
    #                 self.open_positions[symbol]["stop_loss"] = stop_price
    #                 self.open_positions[symbol]["trailing_stop_set"] = True

    #         logger.info(f"[Trailing] Установлен кастомный трейлинг-стоп {stop_price} для {symbol} (PnL: {pnl_percent:.2f}%)")

    #     except Exception as e:
    #         logger.exception(f"[apply_custom_trailing_stop] Ошибка установки кастомного трейлинг-стопа для {symbol}: {e}")


    async def apply_custom_trailing_stop(self, symbol, pos, leveraged_pnl_percent, side):
        """
        Кастовый трейлинг-стоп:
        - включается, когда текущая (с учётом плеча) прибыль >= 5%;
        - стоп ставим на (leveraged_pnl - X)%, при этом не даём стопу опуститься обратно,
        т.е. если новая цель меньше предыдущей -- ничего не меняем.
        """
        START_CUSTOM_TRAIL = Decimal("5.0")
        TRAIL_OFFSET = Decimal("2.0")
        if leveraged_pnl_percent < START_CUSTOM_TRAIL:
            return
        desired_stop = leveraged_pnl_percent - TRAIL_OFFSET
        if desired_stop < Decimal("0"):
            desired_stop = Decimal("0")
        async with self.open_positions_lock:
            pos_in_bot = self.open_positions.get(symbol)
            if not pos_in_bot:
                return
            old_stop = pos_in_bot.get("custom_stop_loss_percent", Decimal("0"))
            if desired_stop <= old_stop:
                return
            pos_in_bot["custom_stop_loss_percent"] = desired_stop
        entry_price = Decimal(str(pos.get("avg_price", 0)))
        if entry_price <= 0:
            return
        leverage = Decimal("10")
        stop_ratio = desired_stop / Decimal("100") / leverage
        if side.lower() == "buy":
            stop_price = entry_price * (Decimal("1") + stop_ratio)
        else:
            stop_price = entry_price * (Decimal("1") - stop_ratio)
        logger.info(f"[CustomTrailingStop] {symbol}: тек. pnl={leveraged_pnl_percent}%, двигаем стоп на {desired_stop}% => цена {stop_price:.4f}")
        await self.set_fixed_stop_loss(symbol, pos["size"], side, stop_price)
        await self.log_trade(
            user_id=self.user_id,
            symbol=symbol,
            row=None,
            side=side,
            open_interest=None,
            action=f"Установлен каст.стоп, PnL={leveraged_pnl_percent}%",
            result="TrailingStop",
            closed_manually=False
        )


    async def open_averaging_order(self, symbol: str, side: str, volume_usdt: Decimal, reason: str):
        # Получаем текущую цену
        last_price = await self.get_last_close_price(symbol)
        if not last_price or last_price <= 0:
            logger.info(f"[open_averaging_order] Нет актуальной цены для {symbol}, пропуск.")
            return
        # Вычисляем количество по объёму усреднения
        qty_dec = volume_usdt / Decimal(str(last_price))
        qty_float = float(qty_dec)
        pos_idx = 1 if side.lower() == "buy" else 2
        trade_id = f"{symbol}_averaging_{int(time.time())}"
        features_dict = {}
        df_5m = await self.get_historical_data_for_model(symbol, interval="1", limit=1)
        df_5m = await self.prepare_features_for_model(df_5m)
        if not df_5m.empty:
            row_feat = df_5m.iloc[-1]
            for fc in self.MODEL_FEATURE_COLS:
                features_dict[fc] = row_feat.get(fc, 0)
        await self.log_model_features_for_trade(trade_id=trade_id, symbol=symbol, side=side, features=features_dict)
        order_res = await self.place_order(symbol, side, qty_float)
        if not order_res or order_res.get("retCode") != 0:
            logger.info(f"[open_averaging_order] Ошибка place_order для {symbol}, пропуск.")
            return
        async with self.open_positions_lock:
            self.averaging_positions[symbol] = {
                "side": side,
                "size": qty_float,
                "avg_price": float(last_price),
                "position_volume": float(volume_usdt),
                "symbol": symbol,
                "trade_id": trade_id,
                "open_time": datetime.datetime.utcnow()
            }
        row = await self.get_last_row(symbol)
        await self.log_trade(
            user_id=self.user_id,
            symbol=symbol,
            row=row,
            side=side,
            open_interest=None,
            action=f"{side} усреднение",
            result="Averaging Opened",
            closed_manually=False
        )
        logger.info(f"[open_averaging_order] {symbol}: {side} усреднение успешно открыто, объем {volume_usdt} USDT")
            

    async def apply_superttrend_custom_trailing_stop(self, symbol, pos, leveraged_pnl_percent, side):
        # Получаем исторические данные (используем 1-минутный таймфрейм, limit=205)
        df = await self.get_historical_data_for_trading(symbol, interval="1", limit=205)
        if df.empty:
            logger.warning(f"[Custom Trailing Stop] {symbol}: недостаточно данных для вычисления ST(50,3).")
            return
        # Вычисляем SuperTrend(50,3) – используем уже существующую функцию, например, calculate_supertrend_beacon
        st50_df = await self.calculate_supertrend_beacon(df.copy())
        if st50_df.empty:
            logger.warning(f"[Custom Trailing Stop] {symbol}: не удалось вычислить ST(50,3).")
            return
        last_st50 = st50_df["supertrend"].iloc[-1]
        
        # Обновляем трейлинг-стоп в позиции на основании SuperTrend(50,3)
        current_stop = pos.get("trailing_stop", None)
        if side.lower() == "buy":
            # Для длинной позиции стоп движется вверх
            if current_stop is None or last_st50 > current_stop:
                pos["trailing_stop"] = last_st50
                logger.info(f"[Custom Trailing Stop] {symbol}: Обновлён трейлинг-стоп для LONG до {last_st50}")
        elif side.lower() == "sell":
            # Для короткой позиции стоп движется вниз
            if current_stop is None or last_st50 < current_stop:
                pos["trailing_stop"] = last_st50
                logger.info(f"[Custom Trailing Stop] {symbol}: Обновлён трейлинг-стоп для SHORT до {last_st50}")
        await self.log_trade(
            user_id=self.user_id,
            symbol=symbol,
            row=None,
            side=side,
            open_interest=None,
            action=f"Установлен каст.стоп, PnL={leveraged_pnl_percent}%",
            result="TrailingStop",
            closed_manually=False
        )

        
        # Проверяем, не пересекла ли цена обновлённый стоп, чтобы закрыть позицию
        current_price = await self.get_last_close_price(symbol)
        if current_price is None:
            return
        cp = Decimal(str(current_price))
        if side.lower() == "buy" and cp < pos["trailing_stop"]:
            logger.info(f"[Custom Trailing Stop] {symbol}: Цена {cp} ниже трейлинг-стопа {pos['trailing_stop']}, закрываю LONG.")
            await self.close_position(symbol, position_idx=pos.get("positionIdx"))
        elif side.lower() == "sell" and cp > pos["trailing_stop"]:
            logger.info(f"[Custom Trailing Stop] {symbol}: Цена {cp} выше трейлинг-стопа {pos['trailing_stop']}, закрываю SHORT.")
            await self.close_position(symbol, position_idx=pos.get("positionIdx"))


    async def open_position(self, symbol: str, side: str, volume_usdt: Decimal, reason: str):
        if not self.state.get("connectivity_ok", True):
            logger.warning(f"[open_position] Связь с биржей нестабильна! Открытие позиции для {symbol} блокируется.")
            return
        if self.IS_SLEEPING_MODE:
            logger.info(f"[open_position] Бот в спящем режиме, открытие {symbol} отменено.")
            return
        async with self.state_lock, self.open_positions_lock:
            current_total = sum(Decimal(str(pos.get("position_volume", 0))) for pos in self.open_positions.values())
            if current_total + volume_usdt > self.MAX_TOTAL_VOLUME:
                logger.warning(f"[open_position] Превышен глобальный лимит: {current_total} + {volume_usdt} > {self.MAX_TOTAL_VOLUME}")
                return
            if reason != "Averaging":
                if symbol in self.open_positions:
                    logger.info(f"[open_position] Позиция для {symbol} уже открыта, пропуск.")
                    return
            self.open_positions[symbol] = {
                "side": side,
                "size": None,
                "avg_price": None,
                "position_volume": volume_usdt,
                "symbol": symbol,
                "trailing_stop_set": False,
                "trade_id": None,
                "open_time": datetime.datetime.now(timezone.utc)
            }
            self.state["total_open_volume"] = current_total + volume_usdt
        last_price = await self.get_last_close_price(symbol)
        if not last_price or last_price <= 0:
            logger.info(f"[open_position] Нет актуальной цены для {symbol}, пропуск.")
            async with self.open_positions_lock:
                if symbol in self.open_positions:
                    del self.open_positions[symbol]
            return
        qty_dec = volume_usdt / Decimal(str(last_price))
        qty_float = float(qty_dec)
        pos_idx = 1 if side.lower() == "buy" else 2
        trade_id = f"{symbol}_{int(time.time())}"
        features_dict = {}
        df_5m = await self.get_historical_data_for_model(symbol, interval="1", limit=1)
        df_5m = await self.prepare_features_for_model(df_5m)
        if not df_5m.empty:
            row_feat = df_5m.iloc[-1]
            for fc in self.MODEL_FEATURE_COLS:
                features_dict[fc] = row_feat.get(fc, 0)
        await self.log_model_features_for_trade(trade_id=trade_id, symbol=symbol, side=side, features=features_dict)
        order_res = await self.place_order(symbol, side, qty_float)
        if not order_res or order_res.get("retCode") != 0:
            logger.info(f"[open_position] Ошибка place_order для {symbol}, пропуск.")
            async with self.open_positions_lock:
                if symbol in self.open_positions:
                    del self.open_positions[symbol]
            return
        async with self.state_lock, self.open_positions_lock:
            self.open_positions[symbol] = {
                "side": side,
                "size": qty_float,
                "avg_price": float(last_price),
                "position_volume": float(volume_usdt),
                "symbol": symbol,
                "position_Idx": pos_idx,
                "trailing_stop_set": False,
                "trade_id": trade_id,
                "open_time": datetime.datetime.now(timezone.utc)
            }
        row = await self.get_last_row(symbol)
        await self.log_trade(
            user_id=self.user_id,
            symbol=symbol,
            row=row,
            side=side,
            open_interest=None,
            action=side,
            result="Opened",
            closed_manually=False
        )
        logger.info(f"[open_position] {symbol}: {side} успешно открыта, объем {volume_usdt} USDT")

    async def place_order(self, symbol, side, qty, order_type="Market", time_in_force="GoodTillCancel", reduce_only=False, positionIdx=None):
        adj_qty = await self.adjust_quantity(symbol, qty)
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
        def _place_order():
            return self.session.place_order(**params)
        resp = await asyncio.to_thread(_place_order)
        if resp.get("retCode") == 0:
            logger.info(f"[place_order] OK {symbol}, side={side}, qty={adj_qty}")
            return resp
        else:
            logger.error(f"[place_order] Ошибка: {resp.get('retMsg')} (retCode={resp.get('retCode')})")
            return None

    def get_symbol_info(self, symbol):
        try:
            resp = self.session.get_instruments_info(symbol=symbol, category="linear")
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

    def get_open_position(self, symbol: str) -> Optional[str]:
        """
        Возвращает текущую открытую позицию по символу.
        "Buy" / "Sell" / None
        """
        pos = self.open_positions.get(symbol)
        if pos and "side" in pos:
            return pos["side"]
        return None

    async def log_model_features_for_trade(self, trade_id: str, symbol: str, side: str, features: dict):
        csv_filename = self.REAL_TRADES_FEATURES_CSV
        def _write():
            file_exists = os.path.isfile(csv_filename)
            row_to_write = {"trade_id": trade_id, "symbol": symbol, "side": side}
            row_to_write.update(features)
            try:
                with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=row_to_write.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row_to_write)
            except Exception as e:
                logger.exception(f"[log_model_features_for_trade] Ошибка записи в {csv_filename}: {e}")
        await asyncio.to_thread(_write)

    async def update_trade_outcome(self, trade_id: str, pnl: float):
        csv_filename = self.REAL_TRADES_FEATURES_CSV
        def _update():
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
        await asyncio.to_thread(_update)

    async def log_trade(self, user_id: int, symbol: str, row, side, open_interest,
                        action: str, result: str, closed_manually: bool = False,
                        csv_filename: str = "trade_log.csv"):
        row = await self.get_last_row(symbol)
        if row is None or (hasattr(row, "empty") and row.empty):
            row = {
                "startTime": datetime.datetime.utcnow(),
                "openPrice": 0,
                "highPrice": 0,
                "lowPrice": 0,
                "closePrice": 0,
                "volume": 0,
            }
        time_str = "N/A"
        open_str = "N/A"
        high_str = "N/A"
        low_str  = "N/A"
        close_str= "N/A"
        vol_str  = "N/A"
        if row is not None:
            time_val = row.get("startTime", None)
            if isinstance(time_val, datetime.datetime):
                time_str = time_val.strftime("%Y-%m-%d %H:%M:%S")
            elif time_val is not None:
                time_str = str(time_val)
            open_val = row.get("openPrice", None)
            if open_val is not None:
                open_str = str(open_val)
            high_val = row.get("highPrice", None)
            if high_val is not None:
                high_str = str(high_val)
            low_val  = row.get("lowPrice", None)
            if low_val is not None:
                low_str  = str(low_val)
            close_val= row.get("closePrice", None)
            if close_val is not None:
                close_str= str(close_val)
            vol_val  = row.get("volume", None)
            if vol_val is not None:
                vol_str  = str(vol_val)
        oi_str = str(open_interest) if open_interest is not None else "N/A"
        closed_str = "вручную" if closed_manually else "по сигналу"
        csv_user_id   = str(user_id)
        csv_symbol    = symbol
        csv_timestamp = time_str
        csv_open      = open_str
        csv_high      = high_str
        csv_low       = low_str
        csv_close     = close_str
        csv_volume    = vol_str
        csv_oi        = oi_str
        csv_action    = action
        csv_result    = result
        csv_closed    = closed_str
        def _log(loop):
            file_exists = os.path.isfile(csv_filename)
            try:
                with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "user_id", "symbol", "timestamp",
                            "openPrice", "highPrice", "lowPrice", "closePrice", "volume",
                            "open_interest", "action", "result", "closed_manually"
                        ])
                    writer.writerow([
                        csv_user_id, csv_symbol, csv_timestamp,
                        csv_open, csv_high, csv_low, csv_close, csv_volume,
                        csv_oi, csv_action, csv_result, csv_closed
                    ])
            except Exception as e:
                logger.error(f"[log_trade] Ошибка выполнения: {e}")
            
            link_url = f"https://www.bybit.com/trade/usdt/{csv_symbol}"
            s_manually = closed_str
            s_side     = side if side else ""
            s_result   = (result or "").lower()
            if s_result == "opened":
                if side.lower() == "buy":
                    msg = (f"🟩 <b>Открытие ЛОНГ-позиции</b>\n"
                        f"<b>Символ:</b> <a href=\"{link_url}\">{csv_symbol}</a>\n"
                        f"<b>Пользователь:</b> {csv_user_id}\n"
                        f"<b>Время:</b> {csv_timestamp}\n"
                        f"<b>Цена открытия:</b> {csv_open}\n"
                        f"<b>Объём:</b> {csv_volume}\n"
                        f"<b>Тип открытия:</b> ЛОНГ")
                elif side.lower() == "sell":
                    msg = (f"🟥 <b>Открытие SHORT-позиции</b>\n"
                        f"<b>Символ:</b> <a href=\"{link_url}\">{csv_symbol}</a>\n"
                        f"<b>Пользователь:</b> {csv_user_id}\n"
                        f"<b>Время:</b> {csv_timestamp}\n"
                        f"<b>Цена открытия:</b> {csv_open}\n"
                        f"<b>Объём:</b> {csv_volume}\n"
                        f"<b>Тип открытия:</b> ШОРТ")
                else:
                    msg = (f"🟩🔴 <b>Открытие позиции</b>\n"
                        f"<b>Символ:</b> <a href=\"{link_url}\">{csv_symbol}</a>\n"
                        f"<b>Пользователь:</b> {csv_user_id}\n"
                        f"<b>Время:</b> {csv_timestamp}\n"
                        f"<b>Цена открытия:</b> {csv_open}\n"
                        f"<b>Объём:</b> {csv_volume}\n"
                        f"<b>Тип открытия:</b> {s_side}")
            elif s_result == "closed":
                msg = (f"❌ <b>Закрытие позиции</b>\n"
                    f"<b>Символ:</b> <a href=\"{link_url}\">{csv_symbol}</a>\n"
                    f"<b>Пользователь:</b> {csv_user_id}\n"
                    f"<b>Время закрытия:</b> {csv_timestamp}\n"
                    f"<b>Цена закрытия:</b> {csv_close}\n"
                    f"<b>Объём:</b> {csv_volume}\n"
                    f"<b>Тип закрытия:</b> {s_manually}")
            elif s_result == "trailingstop":
                msg = (f"🔄 <b>Трейлинг-стоп</b>\n"
                    f"<b>Символ:</b> <a href=\"{link_url}\">{csv_symbol}</a>\n"
                    f"<b>Пользователь:</b> {csv_user_id}\n"
                    f"<b>Время:</b> {csv_timestamp}\n"
                    f"<b>Статус:</b> {csv_action}")
            else:
                msg = (f"🫡🔄 <b>Сделка</b>\n"
                    f"<b>Символ:</b> <a href=\"{link_url}\">{csv_symbol}</a>\n"
                    f"<b>Пользователь:</b> {csv_user_id}\n"
                    f"<b>Время:</b> {csv_timestamp}\n"
                    f"<b>Результат:</b> {csv_result}\n"
                    f"<b>Цена:</b> {csv_close}\n"
                    f"<b>Действие:</b> {csv_action}\n"
                    f"<b>Закрытие:</b> {s_manually}")

            # Планируем отправку сообщения в основном цикле
            asyncio.run_coroutine_threadsafe(telegram_bot.send_message(csv_user_id, msg, parse_mode=ParseMode.HTML), loop)
            
        await asyncio.to_thread(_log, asyncio.get_running_loop())

    async def process_symbol_model_only_async(self, symbol):
        if not self.current_model:
            self.current_model = self.load_model()
            if not self.current_model:
                return
        df_5m = await self.get_historical_data_for_model(symbol, "1", limit=200)
        df_5m = await self.prepare_features_for_model(df_5m)
        if df_5m.empty:
            return
        row = df_5m.iloc[[-1]]
        feat_cols = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
        X = row[feat_cols].values
        try:
            pred = self.current_model.predict(X)
            proba = self.current_model.predict_proba(X)
        except Exception as e:
            logger.exception(f"[MODEL_ONLY] Ошибка для {symbol}: {e}")
            return
        await self.log_model_prediction(symbol, pred[0], proba)
        if pred[0] == 2:
            await self.open_position(symbol, "Buy", self.POSITION_VOLUME, reason="Model")
        elif pred[0] == 0:
            await self.open_position(symbol, "Sell", self.POSITION_VOLUME, reason="Model")
        else:
            logger.info(f"[MODEL_ONLY] {symbol}: HOLD")

    async def log_model_prediction(self, symbol, prediction, prediction_proba):
        def _log():
            fname = "model_predictions_log.csv"
            file_exists = os.path.isfile(fname)
            try:
                with open(fname, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["timestamp", "symbol", "prediction", "prob_buy", "prob_hold", "prob_sell", "user_id"])
                    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    p_sell = prediction_proba[0][0]
                    p_hold = prediction_proba[0][1]
                    p_buy = prediction_proba[0][2]
                    writer.writerow([ts, symbol, prediction, p_buy, p_hold, p_sell, self.user_id])
                logger.info(f"[MODEL] Предсказание для {symbol} записано, user_id={self.user_id}")
            except Exception as e:
                logger.exception(f"Ошибка log_model_prediction({symbol}): {e}")
        await asyncio.to_thread(_log)

    async def main_loop(self):
        logger.info(f"Запуск основного цикла для пользователя {self.user_id}")

        # Запускаем мониторинг через WebSocket
        # asyncio.create_task(self.realtime_position_monitor())

        trading_logic = TradingLogic(self)
        iteration_count = 0
        while self.state.get("run", True) and not self.IS_SLEEPING_MODE:
            try:
                exch_positions = await asyncio.to_thread(self.get_exchange_positions)
                # WebSocket логика закомментирована для REST-мониторинга:
                # if monitoring in ("demo_ws", "real_ws"):
                #     try:
                #         await self.init_websocket()
                #         while self.active:
                # 
                #             await self.sync_ws_to_open_positions()  # Только синхронизируем при необходимости
                #             await trading_logic.execute_trading_mode()
                # 
                #             if not await self.check_connection():
                #                 continue
                #             
                #             if self.ws and self.ws.is_connected():
                #                 await self.ws.close()
                #             logger.info("Цикл остановлен")
                # 
                #     finally:
                #         await self.ws.close()
                #         logger.info("WebSocket отключён")


                if monitoring == "http":
                    await self.update_open_positions_from_exch_positions(exch_positions)
                    await self.update_tickers_subscription()

                usdt_pairs = self.get_usdt_pairs()
                if usdt_pairs:
                    self.selected_symbols = usdt_pairs
                for symbol in self.selected_symbols:
                    df_trading = await self.get_historical_data_for_trading(symbol, interval="1", limit=200)
                    feature_cols = ["openPrice", "closePrice", "highPrice", "lowPrice"]
                    is_anomaly, strength, direction = await self.monitor_feature_drift_per_symbol(symbol, df_trading, pd.DataFrame(), feature_cols, threshold=0.5)
                    if is_anomaly:
                        logger.info(f"[Drift] {symbol}: аномалия обнаружена, strength={strength:.3f}, direction={direction}")
                    await trading_logic.execute_trading_mode()
                if iteration_count % 5 == 0:
                    await self.publish_drift_and_model_tables(self)
                # await self.update_open_positions_from_exch_positions(exch_positions)
                await asyncio.sleep(60)
                iteration_count += 1
                if iteration_count % 20 == 0:
                    await self.maybe_retrain_model()
                await asyncio.sleep(10)
            except Exception as e:
                logger.exception(f"Ошибка во внутреннем цикле для пользователя {self.user_id}: {e}")
                await asyncio.sleep(5)
            await asyncio.sleep(1)



        logger.info(f"Основной цикл для пользователя {self.user_id} завершён.")

    def get_position_info(self, symbol, side):
        try:
            resp = self.session.get_positions(category="linear", symbol=symbol)
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
        
    def get_exchange_positions(self):
        try:
            resp = self.session.get_positions(category="linear", settleCoin="USDT")
            if resp.get("retCode") != 0:
                logger.error(f"[get_exchange_positions] retCode={resp.get('retCode')} => {resp.get('retMsg')}")
                self.state["connectivity_ok"] = False
                # Возвращаем None, чтобы метод update_open_positions_from_exch_positions понял, что 
                # что-то не так, и не стал затирать локальные позиции
                return None
            self.state["connectivity_ok"] = True

            positions = resp["result"].get("list", [])
            exchange_positions = {}
            for pos in positions:
                size = float(pos.get("size", 0))
                if size == 0:
                    continue
                sym = pos.get("symbol")
                side = pos.get("side", "").capitalize()
                entry_price = float(pos.get("avgPrice", 0))
                volume_usdt = size * entry_price
                exchange_positions[sym] = {
                    "side": side,
                    "size": size,
                    "avg_price": entry_price,
                    "position_volume": volume_usdt,
                    "symbol": sym,
                    "positionIdx": pos.get("positionIdx"),
                }
            return exchange_positions

        except Exception as e:
            logger.exception(f"[get_exchange_positions] Ошибка: {e}")
            self.state["connectivity_ok"] = False
            return None  # В случае исключения тоже вернём None

    async def close_position(self, symbol: str, position_idx=None):
        """
        Универсальная функция для закрытия (reduce-only) текущей позиции по символу.
        Параметры:
        - symbol: строка с названием торгового инструмента (например, 'BTCUSDT')
        - position_idx: (опционально) идентификатор позиции, если биржа требует точечного закрытия 
                        конкретной позиции (long/short). Если не нужно, можно не указывать.
        """
        # Проверяем, есть ли у нас в словаре open_positions информация по символу
        open_pos = self.open_positions.get(symbol)
        if not open_pos:
            logger.warning(f"[close_position] Нет открытой позиции в self.open_positions для {symbol}.")
            return
        
        # Определяем текущие параметры позиции: сторона, размер и т.д.
        side_str = open_pos["side"]  # 'Buy' или 'Sell'
        size = open_pos["size"]      # количество контрактов
        
        if size is None or size == 0:
            logger.warning(f"[close_position] В позиции {symbol} размер size=0, пропускаем закрытие.")
            return

        # Для закрытия нужно разместить рыночный ордер в противоположную сторону
        opposite_side = "Buy" if side_str.lower() == "sell" else "Sell"

        try:
            logger.info(f"[close_position] Закрываем позицию по {symbol}: {side_str} -> {opposite_side} (qty={size}).")

            # Размещаем ордер с reduce_only=True (чтобы не открыть «противоположную» позицию).
            await self.place_order(
                symbol=symbol,
                side=opposite_side,
                qty=size,
                reduce_only=True,
                position_idx=position_idx
            )

            # Логируем закрытие (при необходимости – в таблицу сделок)
            await self.log_trade(
                user_id=self.user_id,
                symbol=symbol,
                row=None,
                side=opposite_side,
                open_interest=None,
                action="CLOSE",
                result="Closed",
                closed_manually=False
            )

            # Убираем позицию из локального словаря, чтобы не значилась как «открытая»
            async with self.open_positions_lock:
                if symbol in self.open_positions:
                    self.open_positions.pop(symbol)

            logger.info(f"[close_position] Позиция по {symbol} успешно закрыта.")

        except Exception as e:
            logger.exception(f"[close_position] Ошибка при закрытии позиции {symbol}: {e}")

    # async def update_open_positions_from_exch_positions(self):
    #     try:
    #         # Запрашиваем позиции (пример для Bybit v5)
    #         positions_resp = self.session.get_positions(category="linear", settleCoin="USDT")
    #         if not positions_resp or positions_resp.get("retCode") != 0:
    #             logger.warning("[update_open_positions_from_exch_positions] Ошибка или пустой ответ при get_positions => пропускаем.")
    #             return

    #         positions_data = positions_resp["result"].get("list", [])
    #         if not positions_data:
    #             logger.warning("[update_open_positions_from_exch_positions] Пустой список позиций => все закрыты или ошибка, пропускаем.")
    #             # Если хотите, можно явно очистить все локальные open_positions:
    #             # async with self.open_positions_lock:
    #             #     self.open_positions.clear()
    #             # Но чаще лучше не трогать, если это не гарантированно значит «все закрыты».
    #             return

    #         # Соберём то, что пришло с биржи, в удобную структуру
    #         new_positions = {}
    #         for p in positions_data:
    #             symbol = p.get("symbol")
    #             side = p.get("side")  # 'Buy' / 'Sell'
    #             size_str = p.get("size", "0")
    #             size = float(size_str) if size_str else 0.0
    #             if not symbol or size <= 0:
    #                 # Если позиция 0 или нет symbol, считаем закрытой
    #                 continue

    #             avg_price = float(p.get("avgPrice", "0"))
    #             new_positions[symbol] = {
    #                 "symbol": symbol,
    #                 "side": side,
    #                 "size": size,
    #                 "avg_price": avg_price,
    #                 "position_volume": float(p.get("positionValue", 0.0)),  # пример, если нужно
    #                 # добавьте, что нужно
    #             }

    #         # Обновляем локальный словарь open_positions под блокировкой
    #         async with self.open_positions_lock:
    #             # Удаляем те символы, которых нет в новых позициях
    #             to_remove = [sym for sym in self.open_positions if sym not in new_positions]
    #             for sym in to_remove:
    #                 logger.info(f"[update_open_positions_from_exch_positions] Позиция {sym} закрыта на бирже — удаляем из локального списка.")
    #                 del self.open_positions[sym]

    #             # Обновляем или добавляем те, что есть (свежие) в new_positions
    #             for sym, pos_info in new_positions.items():
    #                 if sym not in self.open_positions:
    #                     self.open_positions[sym] = pos_info
    #                     logger.info(f"[update_open_positions_from_exch_positions] Добавили новую позицию {sym} => size={pos_info['size']}")
    #                 else:
    #                     # Уже есть локально — обновим поля
    #                     self.open_positions[sym].update(pos_info)
    #                     logger.info(f"[update_open_positions_from_exch_positions] Обновили позицию {sym} => size={pos_info['size']}")
    #     except Exception as e:
    #         logger.exception(f"[update_open_positions_from_exch_positions] Ошибка при обновлении позиций: {e}")

    # def update_open_positions_from_exchange(self):
    #     """Загружает открытые позиции через REST и запускает управление ими."""
    #     try:
    #         logger.info("[Init] Загружаю активные позиции через REST...")
    #         resp = self.session.get_positions(category="linear", settleCoin="USDT")
    #         if not resp or resp.get("retCode") != 0:
    #             logger.warning("[Init] Ошибка получения позиций: %s", resp.get("retMsg"))
    #             return

    #         positions_data = resp["result"].get("list", [])
    #         count = 0
    #         for pos in positions_data:
    #             size = float(pos.get("size", 0))
    #             symbol = pos.get("symbol")
    #             side = pos.get("side")
    #             if size > 0 and symbol and side:
    #                 entry_price = float(pos.get("entryPrice", 0))
    #                 mark_price = float(pos.get("markPrice", 0))

    #                 # Если entry_price отсутствует, ожидаем обновления через WebSocket
    #                 if entry_price == 0:
    #                     self.awaiting_position_update[symbol] = time.time()
    #                     logger.info(f"[Init] Позиция {symbol} без entryPrice, ждём обновления WebSocket")

    #                 pos_data = {
    #                     "symbol": symbol,
    #                     "size": size,
    #                     "side": side,
    #                     "avg_price": entry_price,
    #                     "entryPrice": entry_price,
    #                     "markPrice": mark_price,
    #                     "position_volume": size * mark_price,
    #                     "trailing_stop_set": False
    #                 }

    #                 self.ws_positions[symbol] = pos_data.copy()
    #                 async with self.open_positions_lock:
    #                     self.open_positions[symbol] = pos_data.copy()

    #                 self.check_and_manage_position(symbol)

    #                 count += 1

    #         logger.info(f"[Init] Обновлено {count} активных позиций из REST")

    #     except Exception as e:
    #         logger.exception(f"[Init] Ошибка загрузки активных позиций: {e}")

    async def update_open_positions_from_exch_positions(self, exch_positions: dict):
        if not exch_positions:
            logger.warning("[update_open_positions_from_exch_positions] Данные от биржи пустые или None => пропускаем обновление.")
            return
        async with self.open_positions_lock, self.state_lock:
            logger.info(f"[update_open_positions_from_exch_positions] BEFORE: {self.open_positions}")
            to_remove = []
            for sym in list(self.open_positions.keys()):
                if sym not in exch_positions:
                    pos = self.open_positions[sym]
                    trade_id = pos.get("trade_id")
                    close_price = await self.get_last_close_price(sym)
                    if close_price:
                        cp = Decimal(str(close_price))
                        avg_price = pos.get("avg_price")
                        if avg_price is None:
                            logger.warning(f"avg_price для {sym} не задано, пропускаем расчёт PnL.")
                            continue
                        try:
                            ep = Decimal(str(avg_price))
                        except Exception as e:
                            logger.error(f"Ошибка конвертации avg_price для {sym}: {avg_price} - {e}")
                            continue                        
                        pnl = (cp - ep) / ep * Decimal(str(pos.get("position_volume", 0))) if pos["side"].lower() == "buy" else (ep - cp) / ep * Decimal(str(pos.get("position_volume", 0)))
                        if trade_id:
                            await self.update_trade_outcome(trade_id, float(pnl))
                    to_remove.append(sym)
                    await self.log_trade(
                        user_id=self.user_id,
                        symbol=sym,
                        row=None,
                        side=pos["side"],
                        open_interest=None,
                        action="TrailingStop" if pos.get("trailing_stop_set") else "Closed",
                        result="closed",
                        closed_manually=False
                    )
        async with self.open_positions_lock:
            for sym in to_remove:
                self.open_positions.pop(sym, None)
            for sym, newpos in exch_positions.items():
                if sym in self.open_positions:
                    self.open_positions[sym].update({
                        "side": newpos["side"],
                        "size": newpos["size"],
                        "avg_price": newpos["avg_price"],
                        "position_volume": newpos["position_volume"],
                        "positionIdx": newpos["positionIdx"]
                    })
                else:
                    self.open_positions[sym] = {
                        "side": newpos["side"],
                        "size": newpos["size"],
                        "avg_price": newpos["avg_price"],
                        "position_volume": newpos["position_volume"],
                        "symbol": sym,
                        "positionIdx": newpos.get("positionIdx"),
                        "trailing_stop_set": False,
                        "trade_id": None,
                        "open_time": datetime.datetime.now(timezone.utc),
                    }
            total = sum(Decimal(str(p["position_volume"])) for p in self.open_positions.values())
            self.state["total_open_volume"] = total
            logger.info(f"[update_open_positions_from_exch_positions] AFTER: total_open_volume = {total}")


    async def update_open_positions_from_exchange(self):
        """Асинхронно загружает открытые позиции через REST и запускает управление ими."""
        try:
            logger.info("[Init] Загружаю активные позиции через REST...")
            resp = await asyncio.to_thread(lambda: self.session.get_positions(category="linear", settleCoin="USDT"))
            if not resp or resp.get("retCode") != 0:
                logger.warning("[Init] Ошибка получения позиций: %s", resp.get("retMsg"))
                return

            positions_data = resp["result"].get("list", [])
            count = 0
            for pos in positions_data:
                size = float(pos.get("size", 0))
                symbol = pos.get("symbol")
                side = pos.get("side")
                if size > 0 and symbol and side:
                    entry_price = float(pos.get("entryPrice", 0))
                    mark_price = float(pos.get("markPrice", 0))

                    # Если entry_price отсутствует, ожидаем обновления через WebSocket
                    if entry_price == 0:
                        self.awaiting_position_update[symbol] = time.time()
                        logger.info(f"[Init] Позиция {symbol} без entryPrice, ждём обновления WebSocket")

                    pos_data = {
                        "symbol": symbol,
                        "size": size,
                        "side": side,
                        "avg_price": entry_price,
                        "entryPrice": entry_price,
                        "markPrice": mark_price,
                        "position_volume": size * mark_price,
                        "trailing_stop_set": False
                    }

                    self.ws_positions[symbol] = pos_data.copy()
                    async with self.open_positions_lock:
                        self.open_positions[symbol] = pos_data.copy()

                    await self.check_and_manage_position(symbol)
                    count += 1

            logger.info(f"[Init] Обновлено {count} активных позиций из REST")

        except Exception as e:
            logger.exception(f"[Init] Ошибка загрузки активных позиций: {e}")


    def escape_markdown(self, text: str) -> str:
        escape_chars = r"_*\[\]()~`>#+\-={}|.,!\\"
        pattern = re.compile(r"([%s])" % re.escape(escape_chars))
        return pattern.sub(r"\\\1", text)

    async def send_telegram_message(self, user_id, message, parse_mode=None):
        try:
            if telegram_bot is None:
                logger.error("Telegram bot не инициализирован!")
                return
            await telegram_bot.send_message(
                chat_id=user_id,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.exception(f"Ошибка отправки сообщения в Telegram: {e}")

    async def update_open_positions(self):
        try:
            # Выполняем REST-запрос для получения открытых позиций. (Параметры запроса могут отличаться в зависимости от API Bybit.)
            resp = await asyncio.to_thread(lambda: self.session.get_positions(category="linear", settleCoin="USDT"))
            if resp.get("retCode") != 0:
                logger.error(f"[update_open_positions] Ошибка получения позиций: {resp.get('retMsg')}")
                return
            positions = resp.get("result", {}).get("list", [])
            async with self.open_positions_lock:
                self.open_positions.clear()
                for pos in positions:
                    symbol = pos.get("symbol")
                    self.open_positions[symbol] = pos
            logger.info(f"[update_open_positions] Обновлено позиций: {self.open_positions}")
        except Exception as e:
            logger.exception(f"[update_open_positions] Ошибка: {e}")


# ------------------ Класс TradingLogic ------------------

class TradingLogic:
    def __init__(self, trading_bot: TradingBot):
        self.bot = trading_bot
        self.st_cross2_state = defaultdict(dict)

    async def execute_trading_mode(self):
        mode = self.bot.OPERATION_MODE
        logger.info(f"[TradingLogic] Пользователь {self.bot.user_id}: режим {mode}")
        if mode == "drift_only":
            await self.execute_drift_only()
        elif mode == "drift_top10":
            await self.execute_drift_top10()
        elif mode == "golden_setup":
            await self.execute_golden_setup()
        elif mode == "super_trend":
            await self.execute_super_trend()
        elif mode == "ST_cross_global":
            await self.execute_st_cross_global()
        elif mode == "ST_cross1":
            await self.execute_st_cross1()
        elif mode == "ST_cross2":
            await self.execute_st_cross2()
        elif mode == "ST_cross3":
            await self.execute_st_cross3()
        elif mode == "ST_cross2_drift":
            await self.execute_st_cross2_drift()
        elif mode == "model_only":
            await self.bot.process_symbol_model_only_async(self.bot.selected_symbols)
        elif mode == "golden_regression":
            await self.execute_golden_regression()
        elif mode == "kalman_golden_regression":
            await self.execute_kalman_regression()

        else:
            logger.info(f"[TradingLogic] Режим {mode} не реализован.")

    async def execute_drift_only(self):
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            df = await self.bot.get_historical_data_for_trading(sym, interval="1", limit=200)
            feature_cols = ["openPrice", "closePrice", "highPrice", "lowPrice"]
            is_anomaly, strength, direction = await self.bot.monitor_feature_drift_per_symbol(
                sym, df, pd.DataFrame(), feature_cols, threshold=0.5
            )
            if is_anomaly:
                side = "Sell" if direction == "вверх" else "Buy"
                logger.info(f"[Drift Only] {sym}: аномалия (strength={strength:.3f}, direction={direction}). Открытие {side}.")
                await self.bot.open_position(sym, side, self.bot.POSITION_VOLUME, reason="Drift_only")

    async def execute_drift_top10(self):
        drift_signals = []
        for sym, records in self.bot.drift_history.items():
            if records:
                avg_strength = sum(r[1] for r in records) / len(records)
                last_direction = records[-1][2]
                drift_signals.append((sym, avg_strength, last_direction))
        drift_signals.sort(key=lambda x: x[1], reverse=True)
        top_signals = drift_signals[:10]
        for sym, strength, direction in top_signals:
            side = "Sell" if direction == "вверх" else "Buy"
            logger.info(f"[Drift Top10] {sym}: side={side}, strength={strength:.3f}.")
            await self.bot.open_position(sym, side, self.bot.POSITION_VOLUME, reason="Drift_top10")

    async def execute_golden_setup(self):
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            df = await self.bot.get_historical_data_for_trading(sym, interval="1", limit=20)
            if df.empty:
                continue
            action, price_change = await self.handle_golden_setup(sym, df)
            if action:
                logger.info(f"[Golden Setup] {sym}: action={action}, price_change={price_change:.2f}.")
                await self.bot.open_position(sym, action, self.bot.POSITION_VOLUME, reason="Golden_setup")

    async def handle_golden_setup(self, symbol, df):
        try:
            current_oi = Decimal(str(df.iloc[-1]["open_interest"]))
            current_vol = Decimal(str(df.iloc[-1]["volume"]))
            current_price = Decimal(str(df.iloc[-1]["closePrice"]))
            async with history_lock:
                if symbol not in open_interest_history:
                    open_interest_history[symbol] = []
                if symbol not in volume_history:
                    volume_history[symbol] = []
                open_interest_history[symbol].append(current_oi)
                volume_history[symbol].append(current_vol)
                sp_iters = int(self.bot.golden_params["Sell"]["period_iters"])
                lp_iters = int(self.bot.golden_params["Buy"]["period_iters"])
                period = max(sp_iters, lp_iters)
                if len(open_interest_history[symbol]) < period or len(volume_history[symbol]) < period:
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
                logger.info(f"[GOLDEN_SETUP] {symbol}: p_ch={price_change:.2f}, vol_ch={volume_change:.2f}, oi_ch={oi_change:.2f}")
                action = None
                if (price_change <= -self.bot.golden_params["Sell"]["price_change"] and
                    volume_change >= self.bot.golden_params["Sell"]["volume_change"] and
                    oi_change >= self.bot.golden_params["Sell"]["oi_change"]):
                    action = "Sell"
                elif (price_change >= self.bot.golden_params["Buy"]["price_change"] and
                      volume_change >= self.bot.golden_params["Buy"]["volume_change"] and
                      oi_change >= self.bot.golden_params["Buy"]["oi_change"]):
                    action = "Buy"
                else:
                    return None, None
            return (action, float(price_change))
        except Exception as e:
            logger.exception(f"Ошибка handle_golden_setup({symbol}): {e}")
            return None, None

    async def execute_super_trend(self):
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            df = await self.bot.get_historical_data_for_trading(sym, interval=self.bot.INTERVAL, limit=200)
            if df.empty or len(df) < 3:
                logger.info(f"[SuperTrend] {sym}: недостаточно данных.")
                continue
            st_df = await self.bot.calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
            if st_df.empty or len(st_df) < 3:
                logger.info(f"[SuperTrend] {sym}: ошибка расчёта SuperTrend.")
                continue
            i0 = len(st_df) - 1
            i1 = i0 - 1
            o1 = st_df["openPrice"].iloc[i1]
            c1 = st_df["closePrice"].iloc[i1]
            st1 = st_df["supertrend"].iloc[i1]
            o0 = st_df["openPrice"].iloc[i0]
            st0 = st_df["supertrend"].iloc[i0]
            is_buy = (o1 < st1 and c1 > st1 and o0 > st0)
            is_sell = (o1 > st1 and c1 < st1 and o0 < st0)
            if is_buy:
                logger.info(f"[SuperTrend] {sym}: сигнал BUY.")
                await self.bot.open_position(sym, "Buy", self.bot.POSITION_VOLUME, reason=f"SuperTrend_{self.bot.INTERVAL}")
            elif is_sell:
                logger.info(f"[SuperTrend] {sym}: сигнал SELL.")
                await self.bot.open_position(sym, "Sell", self.bot.POSITION_VOLUME, reason=f"SuperTrend_{self.bot.INTERVAL}")
            else:
                logger.info(f"[SuperTrend] {sym}: условия не выполнены.")

    async def execute_st_cross_global(self):
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            df = await self.bot.get_historical_data_for_trading(sym, interval=self.bot.INTERVAL, limit=200)
            if df.empty or len(df) < 5:
                logger.info(f"[ST_cross_global] {sym}: недостаточно данных.")
                continue
            df_fast = await self.bot.calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
            df_slow = await self.bot.calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
            if df_fast.empty or df_slow.empty:
                logger.info(f"[ST_cross_global] {sym}: не удалось рассчитать SuperTrend.")
                continue
            try:
                last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
                if last_candle_time < pd.Timestamp.utcnow() - pd.Timedelta(minutes=5):
                    logger.warning(f"[ST_cross_global] {sym}: данные устарели.")
                    continue
            except Exception as e:
                logger.error(f"[ST_cross_global] Ошибка проверки времени для {sym}: {e}")
                continue
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
            logger.info(f"[ST_cross_global] {sym}: prev_fast={prev_fast:.6f}, prev_slow={prev_slow:.6f}, curr_fast={curr_fast:.6f}, curr_slow={curr_slow:.6f}, last_close={last_close:.6f}")
            if confirmed_buy:
                logger.info(f"[ST_cross_global] {sym}: сигнал BUY подтверждён.")
                await self.bot.open_position(sym, "Buy", self.bot.POSITION_VOLUME, reason="ST_cross_global")
            elif confirmed_sell:
                logger.info(f"[ST_cross_global] {sym}: сигнал SELL подтверждён.")
                await self.bot.open_position(sym, "Sell", self.bot.POSITION_VOLUME, reason="ST_cross_global")
            else:
                logger.info(f"[ST_cross_global] {sym}: сигнал отсутствует.")

    async def execute_st_cross1(self):
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            df = await self.bot.get_historical_data_for_trading(sym, interval=self.bot.INTERVAL, limit=200)
            if df.empty or len(df) < 5:
                logger.info(f"[ST_cross1] {sym}: недостаточно данных.")
                continue
            df_fast = await self.bot.calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
            df_slow = await self.bot.calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
            if df_fast.empty or df_slow.empty:
                logger.info(f"[ST_cross1] {sym}: не удалось рассчитать SuperTrend.")
                continue
            try:
                last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
                if last_candle_time < pd.Timestamp.utcnow() - pd.Timedelta(minutes=5):
                    logger.warning(f"[ST_cross1] {sym}: данные устарели.")
                    continue
            except Exception as e:
                logger.error(f"[ST_cross1] Ошибка проверки времени для {sym}: {e}")
                continue
            df_fast, df_slow = df_fast.align(df_slow, join="inner", axis=0)
            prev_fast = df_fast.iloc[-2]["supertrend"]
            curr_fast = df_fast.iloc[-1]["supertrend"]
            prev_slow = df_slow.iloc[-2]["supertrend"]
            curr_slow = df_slow.iloc[-1]["supertrend"]
            prev_diff = prev_fast - prev_slow
            curr_diff = curr_fast - curr_slow
            last_close = df_fast.iloc[-1]["closePrice"]
            curr_diff_pct = (Decimal(curr_diff) / Decimal(last_close)) * 100
            margin = 0.01
            first_cross_up = prev_diff <= 0 and curr_diff > 0
            first_cross_down = prev_diff >= 0 and curr_diff < 0
            if first_cross_up:
                if curr_diff_pct > Decimal("2"):
                    logger.info(f"[ST_cross1] {sym}: положительное различие слишком велико, пропуск LONG.")
                    continue
                confirmed_buy = last_close >= curr_fast * (1 + margin)
                if confirmed_buy:
                    logger.info(f"[ST_cross1] {sym}: сигнал BUY подтверждён.")
                    await self.bot.open_position(sym, "Buy", self.bot.POSITION_VOLUME, reason="ST_cross1")
                else:
                    logger.info(f"[ST_cross1] {sym}: сигнал BUY не подтверждён.")
            elif first_cross_down:
                if curr_diff_pct < Decimal("-2"):
                    logger.info(f"[ST_cross1] {sym}: отрицательное различие слишком велико, пропуск SHORT.")
                    continue
                confirmed_sell = last_close <= curr_fast * (1 - margin)
                if confirmed_sell:
                    logger.info(f"[ST_cross1] {sym}: сигнал SELL подтверждён.")
                    await self.bot.open_position(sym, "Sell", self.bot.POSITION_VOLUME, reason="ST_cross1")
                else:
                    logger.info(f"[ST_cross1] {sym}: сигнал SELL не подтверждён.")
            else:
                logger.info(f"[ST_cross1] {sym}: сигнал отсутствует.")


    def apply_kalman_filter(self, prices):
        dt = 1.0
        kf = KalmanFilter(dim_x=2, dim_z=1)
        # Модель: состояние = [цена, скорость изменения цены]
        kf.F = np.array([[1, dt],
                        [0, 1]])
        # Наблюдаем только цену
        kf.H = np.array([[1, 0]])
        # Задаём уровень шума измерений
        kf.R = np.array([[0.1]])
        # Процессный шум (слабое воздействие на скорость и цену)
        kf.Q = np.array([[0.001, 0],
                        [0, 0.001]])
        # Начальное состояние: первая цена, скорость равна 0
        kf.x = np.array([prices[0], 0])
        kf.P = np.eye(2)
        filtered_prices = []
        for price in prices:
            kf.predict()
            kf.update(np.array([price]))
            filtered_prices.append(kf.x[0])
        return np.array(filtered_prices)

    async def execute_st_cross2(self):
        """
        Стратегия ST_cross2 на одном таймфрейме (5m).
        - fast ST(2,1), slow ST(8,2) дают базовый «пересекающийся» сигнал (bullish/bearish).
        - confirm ST(50,3) подтверждает сигнал (либо «по положению», либо «по пересечению цены и ST» за последние 3 свечи).
        - Добавлены логи (logger.info) для отслеживания шагов.
        """

        # Список символов для торговли
        symbols = self.bot.get_selected_symbols()
        logger.info("[ST_cross2] Старт обработки символов.")

        # Переключатели подтверждения
        CROSS_TYPE_POSITION = False   # Проверка «цена выше/ниже ST(50,3) на последней свече»
        CROSS_TYPE_CROSSING = True  # Проверка «пересечения цены и ST(50,3) за последние 3 свечи»

        for symbol in symbols:
            logger.info(f"[ST_cross2] Начало обработки {symbol} ...")

            # 1) Загружаем 5-минутные данные
            df_5m = await self.bot.get_historical_data_for_trading(symbol, interval="1", limit=205)
            if df_5m.empty or len(df_5m) < 60:
                logger.info(f"[ST_cross2] {symbol}: недостаточно данных (длина {len(df_5m)}). Пропуск.")
                continue

            # 2) Расчёт ST: fast(2,1), slow(8,2), confirm(50,3)
            st_fast = await self.bot.calculate_supertrend_universal(
                df_5m.copy(),
                length=2,
                multiplier=1.0,
                use_wilder_atr=False
            )
            st_slow = await self.bot.calculate_supertrend_universal(
                df_5m.copy(),
                length=8,
                multiplier=2.0,
                use_wilder_atr=False
            )
            st_conf = await self.bot.calculate_supertrend_universal(
                df_5m.copy(),
                length=50,
                multiplier=3.0,
                use_wilder_atr=False
            )

            # Проверяем не пустые ли результаты
            if st_fast.empty or st_slow.empty or st_conf.empty:
                logger.info(f"[ST_cross2] {symbol}: ошибка расчёта ST (empty DataFrame). Пропуск.")
                continue

            logger.debug(f"[ST_cross2] {symbol}: st_fast={len(st_fast)} баров, st_slow={len(st_slow)}, st_conf={len(st_conf)}")

            # 3) Проверяем «факт пересечения» fast vs slow (берём -2 и -1 бар)
            f_prev = st_fast["supertrend"].iloc[-2]
            f_curr = st_fast["supertrend"].iloc[-1]
            s_prev = st_slow["supertrend"].iloc[-2]
            s_curr = st_slow["supertrend"].iloc[-1]

            prev_diff = f_prev - s_prev
            curr_diff = f_curr - s_curr

            crossed_up = (prev_diff < 0) and (curr_diff > 0)
            crossed_dn = (prev_diff > 0) and (curr_diff < 0)

            # Положение цены относительно fast/slow
            c_fast = st_fast["closePrice"].iloc[-1]
            stF_val = st_fast["supertrend"].iloc[-1]
            c_slow = st_slow["closePrice"].iloc[-1]
            stS_val = st_slow["supertrend"].iloc[-1]

            price_above_both = (c_fast > stF_val) and (c_slow > stS_val)
            price_below_both = (c_fast < stF_val) and (c_slow < stS_val)

            logger.debug(
                f"[ST_cross2] {symbol}: f_prev={f_prev:.4f}, f_curr={f_curr:.4f}, "
                f"s_prev={s_prev:.4f}, s_curr={s_curr:.4f}, "
                f"prev_diff={prev_diff:.4f}, curr_diff={curr_diff:.4f}, "
                f"crossed_up={crossed_up}, crossed_dn={crossed_dn}, "
                f"price_above_both={price_above_both}, price_below_both={price_below_both}"
            )

            # 4) Основной (базовый) сигнал на пересечении fast vs slow
            if crossed_up and price_above_both:
                base_signal = "bullish"
            elif crossed_dn and price_below_both:
                base_signal = "bearish"
            else:
                base_signal = None

            if not base_signal:
                logger.info(f"[ST_cross2] {symbol}: базового сигнала нет, пропуск.")
                continue

            logger.info(f"[ST_cross2] {symbol}: базовый сигнал = {base_signal}")

            # 5) Подтверждение через ST(50,3) (3 последние свечи)
            recent_conf = st_conf.tail(3).reset_index(drop=True)
            c0, s0 = recent_conf.loc[0, "closePrice"], recent_conf.loc[0, "supertrend"]
            c1, s1 = recent_conf.loc[1, "closePrice"], recent_conf.loc[1, "supertrend"]
            c2, s2 = recent_conf.loc[2, "closePrice"], recent_conf.loc[2, "supertrend"]

            conf_signal = None
            logger.debug(f"[ST_cross2] {symbol}: CROSS_TYPE_POSITION={CROSS_TYPE_POSITION}, CROSS_TYPE_CROSSING={CROSS_TYPE_CROSSING}")

            if CROSS_TYPE_POSITION:
                # Если цена последней (текущей) 5m свечи > ST => bullish, если < ST => bearish
                if c2 > s2:
                    conf_signal = "bullish"
                elif c2 < s2:
                    conf_signal = "bearish"
                else:
                    conf_signal = None
                logger.info(f"[ST_cross2] {symbol}: подтверждение (POSITION) => conf_signal={conf_signal}")

            elif CROSS_TYPE_CROSSING:
                # Проверяем пересечение цены и ST(50,3) за последние 3 свечи
                bullish_cross = ((c0 < s0) and (c1 <= s1) and (c2 > s2))
                bearish_cross = ((c0 > s0) and (c1 >= s1) and (c2 < s2))
                if bullish_cross:
                    conf_signal = "bullish"
                elif bearish_cross:
                    conf_signal = "bearish"
                else:
                    conf_signal = None
                logger.info(f"[ST_cross2] {symbol}: подтверждение (CROSSING) => conf_signal={conf_signal}")

            # 6) Итоговый сигнал (совпадение base_signal и conf_signal)
            if base_signal == conf_signal:
                final_signal = base_signal
                logger.info(f"[ST_cross2] {symbol}: Финальный сигнал подтверждён => {final_signal}. Открываем позицию.")
            else:
                final_signal = None
                logger.info(f"[ST_cross2] {symbol}: Подтверждение не совпало, сигнал аннулирован.")
                continue

            # 7) Открываем позицию (если финальный сигнал есть)
            if final_signal == "bullish":
                try:
                    await self.bot.open_position(symbol, "Buy", self.bot.POSITION_VOLUME, reason="ST_cross2")
                    logger.info(f"[ST_cross2] {symbol}: BUY позиция открыта.")
                except Exception as e:
                    logger.error(f"[ST_cross2] {symbol}: Ошибка при открытии BUY: {e}")

            elif final_signal == "bearish":
                try:
                    await self.bot.open_position(symbol, "Sell", self.bot.POSITION_VOLUME, reason="ST_cross2")
                    logger.info(f"[ST_cross2] {symbol}: SELL позиция открыта.")
                except Exception as e:
                    logger.error(f"[ST_cross2] {symbol}: Ошибка при открытии SELL: {e}")

        logger.info("[ST_cross2] Обработка всех символов завершена.")

    # async def execute_st_cross3(self):
    #     """
    #     Стратегия ST_cross3 на одном таймфрейме (1m).
    #     - Добавлена проверка свежести данных через timestamp
    #     - Обрабатываем только события за последние 2 минуты
    #     """
    #     symbols = self.bot.get_selected_symbols()
    #     # symbols = self.get_selected_symbols()
    #     # Randomize order of trading pairs
    #     random.shuffle(symbols)
    #     logger.info("[ST_cross3] Старт обработки символов.")

    #     # Переключатели подтверждения
    #     CROSS_TYPE_POSITION = True   # Проверка «цена выше/ниже ST(50,3) на последней свече»
    #     CROSS_TYPE_CROSSING = False  # Проверка «пересечения цены и ST(50,3) за последние 3 свечи»
        
    #     # Максимальное время "свежести" данных (2 минуты)
    #     MAX_DATA_AGE_SECONDS = 120

    #     for symbol in symbols:
    #         logger.info(f"[ST_cross3] Начало обработки {symbol} ...")

    #         # 1) Загружаем минутные данные
    #         df_1m = await self.bot.get_historical_data_for_trading(symbol, interval="1", limit=205)
    #         if df_1m.empty or len(df_1m) < 60:
    #             logger.info(f"[ST_cross3] {symbol}: недостаточно данных (длина {len(df_1m)}). Пропуск.")
    #             continue

    #         # Проверяем свежесть последней свечи
    #         latest_timestamp = pd.to_datetime(df_1m["startTime"].iloc[-1])
    #         current_time = pd.Timestamp.now(tz='UTC')
    #         time_diff = (current_time - latest_timestamp).total_seconds()
            
    #         if time_diff > MAX_DATA_AGE_SECONDS:
    #             logger.info(f"[ST_cross3] {symbol}: данные устарели (возраст {time_diff:.1f} сек). Пропуск.")
    #             continue

    #         # 2) Расчёт ST: fast(2,1), slow(8,2), confirm(50,3)
    #         st_fast = await self.bot.calculate_supertrend_universal(
    #             df_1m.copy(),
    #             length=2,
    #             multiplier=2.0,
    #             use_wilder_atr=False
    #         )
    #         st_slow = await self.bot.calculate_supertrend_universal(
    #             df_1m.copy(),
    #             length=8,
    #             multiplier=1.0,
    #             use_wilder_atr=False
    #         )

    #         # Проверяем не пустые ли результаты
    #         if st_fast.empty or st_slow.empty:
    #             logger.info(f"[ST_cross3] {symbol}: ошибка расчёта ST (empty DataFrame). Пропуск.")
    #             continue

    #         # 3) Проверяем пересечение только на последних 3 свечах
    #         fast_prev3 = st_fast["supertrend"].iloc[-4]
    #         fast_prev2 = st_fast["supertrend"].iloc[-3]
    #         fast_prev1 = st_fast["supertrend"].iloc[-2]
    #         fast_curr = st_fast["supertrend"].iloc[-1]

    #         slow_prev3 = st_slow["supertrend"].iloc[-4]
    #         slow_prev2 = st_slow["supertrend"].iloc[-3]
    #         slow_prev1 = st_slow["supertrend"].iloc[-2]
    #         slow_curr = st_slow["supertrend"].iloc[-1]

    #         # Проверяем пересечение именно на последних свечах
    #         crossed_up = (fast_prev3 > slow_prev3) and (fast_prev2 >= slow_prev2) and (fast_prev1 < slow_prev1) and (fast_curr < slow_curr)
    #         if crossed_up:
    #             logger.info(f"[ST_cross3] {symbol}: Зафиксировано пересечение ВВЕРХ линий супертренда (время последней свечи: {latest_timestamp}). Анализируем рынок...")
            
    #         crossed_dn = (fast_prev3 < slow_prev3) and (fast_prev2 <= slow_prev2) and (fast_prev1 > slow_prev1) and (fast_curr > slow_curr)
    #         if crossed_dn:
    #             logger.info(f"[ST_cross3] {symbol}: Зафиксировано пересечение ВНИЗ линий супертренда (время последней свечи: {latest_timestamp}). Анализируем рынок...")
                
    #         going_up = (
    #             (df_1m["closePrice"].iloc[-4] < fast_prev3) and
    #             (df_1m["closePrice"].iloc[-3] < fast_prev2) and
    #             (df_1m["closePrice"].iloc[-2] >= fast_prev1) and
    #             (df_1m["openPrice"].iloc[-1] > fast_curr)
    #         )

    #         going_down = (
    #             (df_1m["closePrice"].iloc[-4] > fast_prev3) and
    #             (df_1m["closePrice"].iloc[-3] > fast_prev2) and
    #             (df_1m["closePrice"].iloc[-2] <= fast_prev1) and
    #             (df_1m["openPrice"].iloc[-1] < fast_curr)
    #         )

    #         # Назначаем сигнал
    #         bullish_signal = crossed_up # and going_up
    #         bearish_signal = crossed_dn # and going_down


    #         if bullish_signal:
    #             logger.info(f"[ST_cross3] {symbol}: Сформирован LONG сигнал. Открываем позицию...")
    #             try:
    #                 total_open_vol = await self.bot.get_total_open_volume()
    #                 if total_open_vol + self.bot.POSITION_VOLUME > self.bot.MAX_TOTAL_VOLUME:
    #                     logger.info(f"[open_position] Лимит общего объёма достигнут: текущий объем {total_open_vol} USDT, новый ордер {self.bot.POSITION_VOLUME} USDT, лимит {self.bot.MAX_TOTAL_VOLUME} USDT. Позиция не открывается.")
    #                     continue
    #                 else:
    #                         # Добавляем защиту от повторного открытия:
    #                     now = time.time()
    #                     if symbol in self.bot.recent_signals and now - self.bot.recent_signals[symbol] < 60:
    #                         logger.info(f"[ST_cross3] {symbol}: Пропускаем повторный сигнал — уже был недавно")
    #                         continue

    #                     async with self.bot.open_positions_lock:
    #                         if symbol in self.bot.open_positions:
    #                             logger.info(f"[ST_cross3] Позиция по {symbol} уже открыта — пропускаем.")
    #                             continue
    #                     self.bot.recent_signals[symbol] = now

    #                     await self.bot.open_position(symbol, "Buy", self.bot.POSITION_VOLUME, reason="ST_cross3")
    #                     logger.info(f"[ST_cross3] {symbol}: LONG позиция успешно открыта")
    #             except Exception as e:
    #                 logger.error(f"[ST_cross3] {symbol}: Ошибка при открытии LONG позиции: {e}")

    #         elif bearish_signal:
    #             logger.info(f"[ST_cross3] {symbol}: Сформирован SHORT сигнал. Открываем позицию...")
    #             try:
    #                 total_open_vol = await self.bot.get_total_open_volume()

    #                 if total_open_vol + self.bot.POSITION_VOLUME > self.bot.MAX_TOTAL_VOLUME:
    #                     logger.info(f"[open_position] Лимит общего объёма достигнут: текущий объем {total_open_vol} USDT, новый ордер {self.bot.POSITION_VOLUME} USDT, лимит {self.bot.MAX_TOTAL_VOLUME} USDT. Позиция не открывается.")
    #                     continue
    #                 else:
    #                         # Добавляем защиту от повторного открытия:
    #                     now = time.time()
    #                     if symbol in self.bot.recent_signals and now - self.bot.recent_signals[symbol] < 60:
    #                         logger.info(f"[ST_cross3] {symbol}: Пропускаем повторный сигнал — уже был недавно")
    #                         continue

    #                     async with self.bot.open_positions_lock:
    #                         if symbol in self.bot.open_positions:
    #                             logger.info(f"[ST_cross3] Позиция по {symbol} уже открыта — пропускаем.")
    #                             continue
    #                     self.bot.recent_signals[symbol] = now


    #                     await self.bot.open_position(symbol, "Sell", self.bot.POSITION_VOLUME, reason="ST_cross3")
    #                     logger.info(f"[ST_cross3] {symbol}: SHORT позиция успешно открыта")
    #             except Exception as e:
    #                 logger.error(f"[ST_cross3] {symbol}: Ошибка при открытии SHORT позиции: {e}")
    #         else:
    #             logger.info(f"[ST_cross3] {symbol}: Нет сигнала на вход.")

    #     logger.info("[ST_cross3] Обработка всех символов завершена.")

    async def execute_st_cross3(self):
        """
        Стратегия ST_cross3 на одном таймфрейме (1m).
        - Добавлена проверка свежести данных через timestamp
        - Обрабатываем только события за последние 2 минуты
        """
        symbols = self.bot.get_selected_symbols()
        # symbols = self.get_selected_symbols()
        # Randomize order of trading pairs
        random.shuffle(symbols)
        logger.info("[ST_cross3] Старт обработки символов.")

        # Переключатели подтверждения
        CROSS_TYPE_POSITION = False   # Проверка «цена выше/ниже ST(50,3) на последней свече»
        CROSS_TYPE_CROSSING = True  # Проверка «пересечения цены и ST(50,3) за последние 3 свечи»
        
        # Максимальное время "свежести" данных (2 минуты)
        MAX_DATA_AGE_SECONDS = 120

        for symbol in symbols:
            logger.info(f"[ST_cross3] Начало обработки {symbol} ...")

            # 1) Загружаем минутные данные
            df_1m = await self.bot.get_historical_data_for_trading(symbol, interval="1", limit=205)
            if df_1m.empty or len(df_1m) < 60:
                logger.info(f"[ST_cross3] {symbol}: недостаточно данных (длина {len(df_1m)}). Пропуск.")
                continue

            # Проверяем свежесть последней свечи
            latest_timestamp = pd.to_datetime(df_1m["startTime"].iloc[-1])
            current_time = pd.Timestamp.now(tz='UTC')
            time_diff = (current_time - latest_timestamp).total_seconds()
            
            if time_diff > MAX_DATA_AGE_SECONDS:
                logger.info(f"[ST_cross3] {symbol}: данные устарели (возраст {time_diff:.1f} сек). Пропуск.")
                continue

            # 2) Расчёт ST: fast(2,1), slow(8,2), confirm(50,3)
            st_fast = await self.bot.calculate_supertrend_universal(
                df_1m.copy(),
                length=2,
                multiplier=2.0,
                use_wilder_atr=False
            )
            st_slow = await self.bot.calculate_supertrend_universal(
                df_1m.copy(),
                length=8,
                multiplier=1.0,
                use_wilder_atr=False
            )

            # Проверяем не пустые ли результаты
            if st_fast.empty or st_slow.empty:
                logger.info(f"[ST_cross3] {symbol}: ошибка расчёта ST (empty DataFrame). Пропуск.")
                continue

            # 3) Проверяем пересечение только на последних 3 свечах
            fast_prev3 = st_fast["supertrend"].iloc[-4]
            fast_prev2 = st_fast["supertrend"].iloc[-3]
            fast_prev1 = st_fast["supertrend"].iloc[-2]
            fast_curr = st_fast["supertrend"].iloc[-1]

            slow_prev3 = st_slow["supertrend"].iloc[-4]
            slow_prev2 = st_slow["supertrend"].iloc[-3]
            slow_prev1 = st_slow["supertrend"].iloc[-2]
            slow_curr = st_slow["supertrend"].iloc[-1]

            # Проверяем пересечение именно на последних свечах
            crossed_dn = (fast_prev3 < slow_prev3) and (fast_prev2 <= slow_prev2) and (fast_prev1 > slow_prev1) and (fast_curr > slow_curr)
            if crossed_dn:
                logger.info(f"[ST_cross3] {symbol}: Зафиксировано пересечение ВНИЗ линий супертренда (время последней свечи: {latest_timestamp}). Анализируем рынок...")
            
            crossed_up = (fast_prev3 > slow_prev3) and (fast_prev2 >= slow_prev2) and (fast_prev1 < slow_prev1) and (fast_curr < slow_curr)
            if crossed_up:
                logger.info(f"[ST_cross3] {symbol}: Зафиксировано пересечение ВВЕРХ линий супертренда (время последней свечи: {latest_timestamp}). Анализируем рынок...")


            going_up = (
                (df_1m["closePrice"].iloc[-3] < fast_prev2) and
                (df_1m["closePrice"].iloc[-2] < fast_prev1) and
                (df_1m["closePrice"].iloc[-1] > fast_curr)
            )
            if going_up:
                logger.info(f"[DEBUG] {symbol} | Close[-4]: {df_1m['closePrice'].iloc[-4]}, Fast[-4]: {fast_prev3}")
                logger.info(f"[DEBUG] {symbol} | Close[-3]: {df_1m['closePrice'].iloc[-3]}, Fast[-3]: {fast_prev2}")
                logger.info(f"[DEBUG] {symbol} | Close[-2]: {df_1m['closePrice'].iloc[-2]}, Fast[-2]: {fast_prev1}")
                logger.info(f"[DEBUG] {symbol} | Close[-1]: {df_1m['closePrice'].iloc[-1]}, Fast[-1]: {fast_curr}")
                logger.info(f"[ST_cross3] {symbol}: Все признаки движения вверх. Можно открывать ШОРТ-позицию")

            going_down = (
                (df_1m["closePrice"].iloc[-3] > fast_prev2) and
                (df_1m["closePrice"].iloc[-2] > fast_prev1) and
                (df_1m["closePrice"].iloc[-1] < fast_curr)
            )
            if going_down:
                logger.info(f"[DEBUG] {symbol} | Сlose[-4]: {df_1m['closePrice'].iloc[-4]}, Fast[-4]: {fast_prev3}")
                logger.info(f"[DEBUG] {symbol} | Close[-3]: {df_1m['closePrice'].iloc[-3]}, Fast[-3]: {fast_prev2}")
                logger.info(f"[DEBUG] {symbol} | Close[-2]: {df_1m['closePrice'].iloc[-2]}, Fast[-2]: {fast_prev1}")
                logger.info(f"[DEBUG] {symbol} | Close[-1]: {df_1m['closePrice'].iloc[-1]}, Fast[-1]: {fast_curr}")

                logger.info(f"[ST_cross3] {symbol}: Все признаки движения вниз. Можно открывать ЛОНГ-позицию")

            # Назначаем сигнал
            # bullish_signal = going_up # 
            # bearish_signal = going_down # 


            if going_up:
                logger.info(f"[ST_cross3] {symbol}: Сформирован LONG сигнал. Открываем позицию...")
                try:
                    total_open_vol = await self.bot.get_total_open_volume()
                    if total_open_vol + self.bot.POSITION_VOLUME <= self.bot.MAX_TOTAL_VOLUME:
                        now = time.time()
                        if symbol not in self.bot.recent_signals or now - self.bot.recent_signals[symbol] > 60:
                            async with self.bot.open_positions_lock:
                                if symbol not in self.bot.open_positions:
                                    self.bot.recent_signals[symbol] = now
                                    await self.bot.open_position(symbol, "Buy", self.bot.POSITION_VOLUME, reason="ST_cross3")
                                    logger.info(f"[ST_cross3] {symbol}: LONG позиция успешно открыта")
                                else:
                                    logger.info(f"[ST_cross3] {symbol}: Позиция уже открыта — пропускаем.")
                    else:
                        logger.info(f"[ST_cross3] {symbol}: Превышение лимита общего объёма — пропуск LONG.")
                except Exception as e:
                    logger.error(f"[ST_cross3] {symbol}: Ошибка при открытии LONG позиции: {e}")

            if going_down:
                logger.info(f"[ST_cross3] {symbol}: Сформирован SHORT сигнал. Открываем позицию...")
                try:
                    total_open_vol = await self.bot.get_total_open_volume()
                    if total_open_vol + self.bot.POSITION_VOLUME <= self.bot.MAX_TOTAL_VOLUME:
                        now = time.time()
                        if symbol not in self.bot.recent_signals or now - self.bot.recent_signals[symbol] > 60:
                            async with self.bot.open_positions_lock:
                                if symbol not in self.bot.open_positions:
                                    self.bot.recent_signals[symbol] = now
                                    await self.bot.open_position(symbol, "Sell", self.bot.POSITION_VOLUME, reason="ST_cross3")
                                    logger.info(f"[ST_cross3] {symbol}: SHORT позиция успешно открыта")
                                else:
                                    logger.info(f"[ST_cross3] {symbol}: Позиция уже открыта — пропускаем.")
                    else:
                        logger.info(f"[ST_cross3] {symbol}: Превышение лимита общего объёма — пропуск SHORT.")
                except Exception as e:
                    logger.error(f"[ST_cross3] {symbol}: Ошибка при открытии SHORT позиции: {e}")

            # if bullish_signal:
            #     logger.info(f"[ST_cross3] {symbol}: Сформирован LONG сигнал. Открываем позицию...")
            #     try:
            #         total_open_vol = await self.bot.get_total_open_volume()
            #         if total_open_vol + self.bot.POSITION_VOLUME > self.bot.MAX_TOTAL_VOLUME:
            #             logger.info(f"[open_position] Лимит общего объёма достигнут: текущий объем {total_open_vol} USDT, новый ордер {self.bot.POSITION_VOLUME} USDT, лимит {self.bot.MAX_TOTAL_VOLUME} USDT. Позиция не открывается.")
            #             continue
            #         else:
            #                 # Добавляем защиту от повторного открытия:
            #             now = time.time()
            #             if symbol in self.bot.recent_signals and now - self.bot.recent_signals[symbol] < 60:
            #                 logger.info(f"[ST_cross3] {symbol}: Пропускаем повторный сигнал — уже был недавно")
            #                 continue

            #             async with self.bot.open_positions_lock:
            #                 if symbol in self.bot.open_positions:
            #                     logger.info(f"[ST_cross3] Позиция по {symbol} уже открыта — пропускаем.")
            #                     continue
            #             self.bot.recent_signals[symbol] = now

            #             await self.bot.open_position(symbol, "Buy", self.bot.POSITION_VOLUME, reason="ST_cross3")
            #             logger.info(f"[ST_cross3] {symbol}: LONG позиция успешно открыта")
            #     except Exception as e:
            #         logger.error(f"[ST_cross3] {symbol}: Ошибка при открытии LONG позиции: {e}")

            # elif bearish_signal:
            #     logger.info(f"[ST_cross3] {symbol}: Сформирован SHORT сигнал. Открываем позицию...")
            #     try:
            #         total_open_vol = await self.bot.get_total_open_volume()

            #         if total_open_vol + self.bot.POSITION_VOLUME > self.bot.MAX_TOTAL_VOLUME:
            #             logger.info(f"[open_position] Лимит общего объёма достигнут: текущий объем {total_open_vol} USDT, новый ордер {self.bot.POSITION_VOLUME} USDT, лимит {self.bot.MAX_TOTAL_VOLUME} USDT. Позиция не открывается.")
            #             continue
            #         else:
            #                 # Добавляем защиту от повторного открытия:
            #             now = time.time()
            #             if symbol in self.bot.recent_signals and now - self.bot.recent_signals[symbol] < 60:
            #                 logger.info(f"[ST_cross3] {symbol}: Пропускаем повторный сигнал — уже был недавно")
            #                 continue

            #             async with self.bot.open_positions_lock:
            #                 if symbol in self.bot.open_positions:
            #                     logger.info(f"[ST_cross3] Позиция по {symbol} уже открыта — пропускаем.")
            #                     continue
            #             self.bot.recent_signals[symbol] = now


            #             await self.bot.open_position(symbol, "Sell", self.bot.POSITION_VOLUME, reason="ST_cross3")
            #             logger.info(f"[ST_cross3] {symbol}: SHORT позиция успешно открыта")
            #     except Exception as e:
            #         logger.error(f"[ST_cross3] {symbol}: Ошибка при открытии SHORT позиции: {e}")
            else:
                logger.info(f"[ST_cross3] {symbol}: Нет сигнала на вход.")

        logger.info("[ST_cross3] Обработка всех символов завершена.")


    async def execute_golden_regression(self):
        """
        golden_regression c фильтрацией «ложных» пересечений:
        1) Строим медленную (200 свечей) и быструю (100 свечей) регрессию.
        2) Проверяем пересечение (смену знака разницы).
        3) Дополнительно требуем, чтобы модуль разницы был > 0.005 * closeCurr (к примеру).
        """
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            # Загружаем 200 свечей
            df = await self.bot.get_historical_data_for_trading(
                sym, interval=self.bot.INTERVAL, limit=200
            )
            if df.empty or len(df) < 200:
                logger.info(f"[golden_regression] {sym}: недостаточно данных.")
                continue

            # Вычисляем «взвешенную» цену
            df["weighted_price"] = (
                df["openPrice"] + df["highPrice"] + df["lowPrice"] + 2*df["closePrice"]
            ) / 5.0

            # Делим на «медленную» (200) и «быструю» (последние 100)
            df_slow = df
            df_fast = df.iloc[-100:]  # последние 100 строк

            # Готовим данные для регрессии:
            x_slow = np.arange(len(df_slow)).reshape(-1, 1)   # [0..199]
            y_slow = df_slow["weighted_price"].values

            x_fast = np.arange(len(df_fast)).reshape(-1, 1)   # [0..99]
            y_fast = df_fast["weighted_price"].values

            # Строим робастную (RANSAC) регрессию для медленной
            try:
                slow_model = RANSACRegressor()
                slow_model.fit(x_slow, y_slow)
                slow_slope = slow_model.estimator_.coef_[0]
                slow_intercept = slow_model.estimator_.intercept_
            except Exception as e:
                logger.error(f"[golden_regression] {sym}: ошибка slow_regression: {e}")
                continue

            # Строим робастную (RANSAC) регрессию для быстрой
            try:
                fast_model = RANSACRegressor()
                fast_model.fit(x_fast, y_fast)
                fast_slope = fast_model.estimator_.coef_[0]
                fast_intercept = fast_model.estimator_.intercept_
            except Exception as e:
                logger.error(f"[golden_regression] {sym}: ошибка fast_regression: {e}")
                continue

            # Получаем значение каждой регрессии на «предпоследней» и «последней» свече
            # для медленной: x=198 (предпоследняя), x=199 (последняя)
            slow_prev = slow_intercept + slow_slope * (len(df_slow) - 2)
            slow_curr = slow_intercept + slow_slope * (len(df_slow) - 1)

            # для быстрой: x=98 (предпоследняя), x=99 (последняя)
            fast_prev = fast_intercept + fast_slope * (len(df_fast) - 2)
            fast_curr = fast_intercept + fast_slope * (len(df_fast) - 1)

            # Разница fast - slow
            prev_diff = fast_prev - slow_prev
            curr_diff = fast_curr - slow_curr

            # Проверка «пересечения»: меняется знак
            crossed_up   = (prev_diff < 0) and (curr_diff > 0)
            crossed_down = (prev_diff > 0) and (curr_diff < 0)

            if not (crossed_up or crossed_down):
                logger.info(f"[golden_regression] {sym}: пересечения нет.")
                continue

            # Дополнительный фильтр по «глубине» пересечения:
            # возьмём текущую цену закрытия (последней свечи), чтобы смотреть в процентах
            close_curr = df["closePrice"].iloc[-1]
            diff_threshold = 0.003 * close_curr  # 0.5% от цены закрытия
            if abs(curr_diff) < diff_threshold:
                logger.info(
                    f"[golden_regression] {sym}: пересечение есть, но разница={abs(curr_diff):.4f} < threshold={diff_threshold:.4f} => пропускаем."
                )
                continue

            # Если дошли сюда — значит пересечение «настоящее» (по нашему критерию)
            if crossed_up:
                logger.info(f"[golden_regression] {sym}: НАДЁЖНОЕ пересечение вверх => LONG.")
                await self.bot.open_position(
                    sym, 
                    side="Buy", 
                    volume_usdt=self.bot.POSITION_VOLUME,
                    reason="golden_regression"
                )
            else:  # crossed_down
                logger.info(f"[golden_regression] {sym}: НАДЁЖНОЕ пересечение вниз => SHORT.")
                await self.bot.open_position(
                    sym, 
                    side="Sell", 
                    volume_usdt=self.bot.POSITION_VOLUME,
                    reason="golden_regression"
                )

    async def execute_kalman_regression(self):
        """
        Вместо 2-х S-регрессий используем 2 Калман-фильтра (200 бар vs. 100 бар),
        анализируем пересечение, как в golden_regression.
        """
        logger.info("[kalman_regression] Запуск анализа символов.")
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            # 1) Загружаем 200 свечей
            df = await self.bot.get_historical_data_for_trading(
                symbol=sym, interval=self.bot.INTERVAL, limit=200
            )
            if df.empty or len(df) < 200:
                logger.info(f"[kalman_regression] {sym}: недостаточно данных (нужно >=200). Пропуск.")
                continue

            # 2) Вычисляем взвешенную цену
            df["weighted_price"] = (
                df["openPrice"] + df["highPrice"] + df["lowPrice"] + 2 * df["closePrice"]
            ) / 5.0

            logger.debug(f"[kalman_regression] {sym}: готовим 2 Калман-фильтра (slow=200бар, fast=100бар).")

            # Целый массив для медленного фильтра
            prices = df["weighted_price"].values  # длина 200

            # Массив для быстрого (последние 100)
            fast_prices = prices[-100:]

            # Считаем медленный (slow_KF) по всем 200
            slow_estimates = self.apply_kalman_filter(prices)
            # Быстрый (fast_KF) по 100
            fast_estimates_100 = self.apply_kalman_filter(fast_prices)

            # slow: берем индексы -2, -1 => (198, 199)
            slow_prev = slow_estimates[-2]
            slow_curr = slow_estimates[-1]
            # fast: берем индексы -2, -1 => (98, 99)
            fast_prev = fast_estimates_100[-2]
            fast_curr = fast_estimates_100[-1]

            prev_diff = fast_prev - slow_prev
            curr_diff = fast_curr - slow_curr

            logger.debug(
                f"[kalman_regression] {sym}: slow_prev={slow_prev:.4f}, slow_curr={slow_curr:.4f}, "
                f"fast_prev={fast_prev:.4f}, fast_curr={fast_curr:.4f}, "
                f"prev_diff={prev_diff:.4f}, curr_diff={curr_diff:.4f}"
            )

            # Проверяем пересечение
            crossed_up = (prev_diff < 0) and (curr_diff > 0)
            crossed_down = (prev_diff > 0) and (curr_diff < 0)

            if not (crossed_up or crossed_down):
                logger.info(f"[kalman_regression] {sym}: пересечения не обнаружено. Пропуск.")
                continue

            # Фильтрация слабого пересечения: >0.5% от цены
            close_curr = df["closePrice"].iloc[-1]
            diff_threshold = 0.005 * close_curr
            if abs(curr_diff) < diff_threshold:
                logger.info(
                    f"[kalman_regression] {sym}: пересечение есть, но diff={abs(curr_diff):.4f} < "
                    f"threshold={diff_threshold:.4f} => пропуск."
                )
                continue

            # Сигнал
            if crossed_up:
                logger.info(f"[kalman_regression] {sym}: пересечение ВВЕРХ => Открываем LONG.")
                await self.bot.open_position(
                    sym,
                    side="Buy",
                    volume_usdt=self.bot.POSITION_VOLUME,
                    reason="kalman_regression"
                )
            else:  # crossed_down
                logger.info(f"[kalman_regression] {sym}: пересечение ВНИЗ => Открываем SHORT.")
                await self.bot.open_position(
                    sym,
                    side="Sell",
                    volume_usdt=self.bot.POSITION_VOLUME,
                    reason="kalman_regression"
                )

# ------------------ Telegram клавиатуры и маршрутизация ------------------

class VolumeStates(StatesGroup):
    waiting_for_max_volume = State()
    waiting_for_position_volume = State()

def get_main_menu_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📈 Торговля")],
            [KeyboardButton(text="🤖 Бот")],
            [KeyboardButton(text="ℹ️ Информация")]
        ],
        resize_keyboard=True
    )
    return keyboard

def get_trading_menu_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📊 Статус"), KeyboardButton(text="🔄 Смена режима")],
            [KeyboardButton(text="📉 Установить макс. объем"), KeyboardButton(text="📊 Установить объем позиции")],
            [KeyboardButton(text="📉 Установить таймфрейм ST")],
            [KeyboardButton(text="🔙 Назад")]
        ],
        resize_keyboard=True
    )
    return keyboard

def get_trading_mode_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="drift_only"), KeyboardButton(text="drift_top10")],
            [KeyboardButton(text="golden_setup"), KeyboardButton(text="super_trend")],
            [KeyboardButton(text="ST_cross_global"), KeyboardButton(text="ST_cross1")],
            [KeyboardButton(text="ST_cross2"), KeyboardButton(text="ST_cross2_drift")],
            [KeyboardButton(text="🔙 Назад")]
        ],
        resize_keyboard=True
    )
    return keyboard

def get_bot_menu_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛑 Тихий режим ON/OFF"), KeyboardButton(text="🔕 Статус тихого режима")],
            [KeyboardButton(text="😴 Усыпить бота"), KeyboardButton(text="🌞 Разбудить бота")],
            [KeyboardButton(text="🔙 Назад")]
        ],
        resize_keyboard=True
    )
    return keyboard

def get_info_menu_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔍 Получить данные по паре")],
            [KeyboardButton(text="📊 Публикация Drift-таблицы"), KeyboardButton(text="📈 Публикация Model-таблицы")],
            [KeyboardButton(text="📌 Model Top ON/OFF")],
            [KeyboardButton(text="🔙 Назад")]
        ],
        resize_keyboard=True
    )
    return keyboard

@router.message(Command("menu"))
async def menu_cmd(message: Message):
    await message.reply("Выберите раздел:", reply_markup=get_main_menu_keyboard())

@router_admin.message(Command("stop_admin"))
async def stop_admin_cmd(message: types.Message):
    user_id = message.from_user.id
    if user_id == ADMIN_ID:
        await message.reply("Бот завершается по команде /stop_admin...")
        logger.warning("Получена команда /stop_admin от администратора, останавливаем бота.")
        # Попробуйте сначала остановить поллинг:
        await dp.stop_polling()
        # Если нужно — завершите процесс:
        import sys
        sys.exit(0)
    else:
        await message.reply("У вас нет прав для использования этой команды.")

    
@router.message(lambda msg: msg.text == "📉 Установить макс. объем")
async def set_max_volume_step1(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.answer("Вы не зарегистрированы. Воспользуйтесь /register")
        return
    await message.answer("Введите новый максимальный объём (USDT), например: 2000")
    await state.set_state(VolumeStates.waiting_for_max_volume)

@router.message(VolumeStates.waiting_for_max_volume)
async def set_max_volume_step2(message: Message, state: FSMContext):
    user_id = message.from_user.id
    new_value_str = message.text.strip()
    try:
        val_dec = Decimal(new_value_str)
        if val_dec <= 0:
            raise ValueError("Значение должно быть > 0")
    except Exception as e:
        await message.answer(f"Некорректное число: {e}\nПопробуйте ещё раз или /cancel")
        return
    bot_instance = user_bots[user_id]
    bot_instance.MAX_TOTAL_VOLUME = val_dec
    users[user_id]["max_total_volume"] = str(val_dec)
    await save_users(users)
    await message.answer(f"Установлен новый макс. объём позиций: {val_dec} USDT", reply_markup=get_trading_menu_keyboard())
    await state.clear()

@router.message(lambda msg: msg.text == "📊 Установить объем позиции")
async def set_position_volume_step1(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.answer("Вы не зарегистрированы. Воспользуйтесь /register")
        return
    await message.answer("Введите объём лота (USDT), например 150")
    await state.set_state(VolumeStates.waiting_for_position_volume)

@router.message(VolumeStates.waiting_for_position_volume)
async def set_position_volume_step2(message: Message, state: FSMContext):
    user_id = message.from_user.id
    new_value_str = message.text.strip()
    try:
        val_dec = Decimal(new_value_str)
        if val_dec <= 0:
            raise ValueError("Значение должно быть > 0")
    except Exception as e:
        await message.answer(f"Некорректное число: {e}\nПопробуйте ещё раз или /cancel")
        return
    bot_instance = user_bots[user_id]
    bot_instance.POSITION_VOLUME = val_dec
    users[user_id]["position_volume"] = str(val_dec)
    await save_users(users)
    await message.answer(f"Теперь объём одной позиции: {val_dec} USDT", reply_markup=get_trading_menu_keyboard())
    await state.clear()

@router.message(Command("publish_tables"))
async def publish_tables_cmd(message: Message):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.reply("Пользователь не зарегистрирован.")
        return
    trading_bot = user_bots[user_id]
    await trading_bot.publish_drift_and_model_tables(trading_bot)

@router.message(Command("register"))
async def register_cmd(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id in users:
        await message.answer("Вы уже зарегистрированы в системе.")
        return
    await message.answer(
        "Введите, пожалуйста, ваш API Key.\n"
        "Внимание: Ключи НЕ должны содержать права перевода средств!"
    )
    await state.set_state(RegisterStates.waiting_for_api_key)

@router.message(RegisterStates.waiting_for_api_key)
async def process_api_key(message: Message, state: FSMContext):
    user_id = message.from_user.id
    api_key = message.text.strip()
    await state.update_data(api_key=api_key)
    await message.answer("Принято! Теперь отправьте ваш API Secret.")
    await state.set_state(RegisterStates.waiting_for_api_secret)

@router.message(RegisterStates.waiting_for_api_secret)
async def process_api_secret(message: Message, state: FSMContext):
    user_id = message.from_user.id
    api_secret = message.text.strip()
    await state.update_data(api_secret=api_secret)
    await message.answer(
        "Принято!\n"
        "Теперь выберите режим торговли: напишите 'demo' (для тестовой сети) или 'real' (для реальной биржи)."
    )
    await state.set_state(RegisterStates.waiting_for_mode)

@router.message(RegisterStates.waiting_for_mode)
async def process_mode(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_mode = message.text.strip().lower()
    if user_mode not in ("demo", "real"):
        await message.answer("Пожалуйста, введите только 'demo' или 'real'.")
        return
    data = await state.get_data()
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    def _write():
        file_exists = os.path.isfile("users.csv")
        with open("users.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["user_id", "user_api", "user_api_secret", "mode"])
            writer.writerow([user_id, api_key, api_secret, user_mode])
    await asyncio.to_thread(_write)
    users[user_id] = {
        "user_api": api_key,
        "user_api_secret": api_secret,
        "mode": user_mode,
        "max_total_volume": "1000",
        "position_volume": "100"
    }
    bot_instance = TradingBot(user_id, api_key, api_secret, user_mode)
    user_bots[user_id] = bot_instance
    asyncio.create_task(bot_instance.main_loop())
    await message.answer(
        f"Регистрация завершена!\n"
        f"Вы выбрали режим: {user_mode}.\n"
        "ПОМНИТЕ: Ключи не должны содержать права перевода средств!\n"
        "Теперь вы можете использовать /start."
    )
    await state.clear()

@router.message(lambda message: message.text == "📈 Торговля")
async def trading_menu(message: Message):
    await message.reply("📈 Трейдинг – выберите действие:", reply_markup=get_trading_menu_keyboard())

@router.message(lambda message: message.text == "🔄 Смена режима")
async def change_trading_mode(message: Message):
    await message.reply("Выберите торговый режим:", reply_markup=get_trading_mode_keyboard())

@router.message(lambda message: message.text == "🤖 Бот")
async def bot_menu(message: Message):
    await message.reply("🤖 Бот – выберите действие:", reply_markup=get_bot_menu_keyboard())

@router.message(lambda message: message.text == "ℹ️ Информация")
async def info_menu(message: Message):
    await message.reply("ℹ️ Информация – выберите действие:", reply_markup=get_info_menu_keyboard())

@router.message(lambda message: message.text == "🔙 Назад")
async def back_menu(message: Message):
    await message.reply("Главное меню:", reply_markup=get_main_menu_keyboard())

@router.message(lambda message: message.text == "📊 Статус")
async def status_cmd(message: Message):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.reply("Пользователь не зарегистрирован.")
        return
    bot_instance = user_bots[user_id]
    async with bot_instance.open_positions_lock:
        if not bot_instance.open_positions:
            await message.reply("Нет позиций.")
            return
        lines = []
        total_pnl_usdt = Decimal("0")
        total_invested = Decimal("0")
        for sym, pos in bot_instance.open_positions.items():
            side_str = pos["side"]
            entry_price = Decimal(str(pos["avg_price"]))
            volume_usdt = Decimal(str(pos["position_volume"]))
            current_price = await bot_instance.get_last_close_price(sym)
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
            lines.append(f"{sym} {side_str}: PnL = {pnl_usdt:.2f} USDT ({pnl_percent:.2f}%)")
        lines.append("—" * 30)
        if total_invested > 0:
            total_pnl_percent = (total_pnl_usdt / total_invested) * Decimal("100")
            lines.append(f"Итоговый PnL по всем позициям: {total_pnl_usdt:.2f} USDT ({total_pnl_percent:.2f}%)")
        else:
            lines.append("Итоговый PnL: 0 (нет позиций с объёмом)")
    await message.reply("\n".join(lines))

@router.message(lambda message: message.text in [
    "drift_only", "drift_top10", "golden_setup", "super_trend",
    "ST_cross_global", "ST_cross1", "ST_cross2", "ST_cross2_drift"
])
async def set_trading_mode(message: Message):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.reply("Пользователь не зарегистрирован.")
        return
    trading_bot = user_bots[user_id]
    trading_bot.OPERATION_MODE = message.text
    await message.reply(f"Торговый режим установлен: {message.text}", reply_markup=get_main_menu_keyboard())

@router.message(lambda message: message.text in ["🛑 Тихий режим ON/OFF", "🔕 Статус тихого режима", "😴 Усыпить бота", "🌞 Разбудить бота"])
async def bot_commands(message: Message):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.reply("Пользователь не зарегистрирован.")
        return
    trading_logic = TradingLogic(user_bots[user_id])
    if message.text == "🛑 Тихий режим ON/OFF":
        trading_logic.bot.QUIET_PERIOD_ENABLED = not trading_logic.bot.QUIET_PERIOD_ENABLED
        status = "включён" if trading_logic.bot.QUIET_PERIOD_ENABLED else "выключен"
        await message.reply(f"Тихий режим: {status}")
    elif message.text == "🔕 Статус тихого режима":
        status = "включён" if user_bots[user_id].QUIET_PERIOD_ENABLED else "выключен"
        await message.reply(f"Тихий режим: {status}")
    elif message.text == "😴 Усыпить бота":
        trading_logic.bot.IS_SLEEPING_MODE = True
        await message.reply("Спящий режим включён")
    elif message.text == "🌞 Разбудить бота":
        trading_logic.bot.IS_SLEEPING_MODE = False
        await message.reply("Спящий режим выключен")

@router.message(lambda message: message.text == "🔍 Получить данные по паре")
async def get_pair_info(message: Message):
    await message.reply("Введите символ пары (например, BTCUSDT):")

async def check_user_registration(user_id: int, message: Message):
    if user_id not in user_bots:
        await message.answer("❌ Вы не зарегистрированы!\nДоступные команды:\n/register - Регистрация\n/help - Помощь")
        return False
    return True

@router.message(Command("start"))
async def start_cmd(message: Message):
    user_id = message.from_user.id
    if not await check_user_registration(user_id, message):
        return
    await message.reply("Добро пожаловать! Выберите раздел:", reply_markup=get_main_menu_keyboard())

@router.message(Command("stop"))
async def stop_cmd(message: Message):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.reply("Пользователь не зарегистрирован.")
        return
    bot_instance = user_bots[user_id]
    bot_instance.state["run"] = False
    await message.reply("Бот остановлен для данного пользователя. Для возобновления работы отправьте /start.")

@router.message(Command("ping"))
async def ping_handler(message: Message):
    try:
        await message.answer("🏓 Pong!")
        logger.info(f"Ping received from {message.from_user.id}")
    except Exception as e:
        logger.error(f"Ping handler error: {e}")

@router.message()
async def log_all_messages(message: Message):
    logger.info(f"Received message from {message.from_user.id}: {message.text}")

def setup_telegram_bot():
    global telegram_bot, dp
    try:
        TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        if not TELEGRAM_TOKEN:
            logger.error("TELEGRAM_TOKEN not found in environment!")
            return
        logger.info("Initializing Telegram bot...")
        telegram_bot = Bot(token=TELEGRAM_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        dp = Dispatcher(storage=MemoryStorage())
        dp.include_router(router)
        dp.include_router(router_admin)
        logger.info("Telegram bot initialized successfully")
    except Exception as e:
        logger.exception(f"Error initializing Telegram bot: {e}")
        raise

async def start_telegram_polling():
    await dp.start_polling(telegram_bot)

async def monitor_positions_http():
    while True:
        try:
            for user_id, bot in user_bots.items():
                if bot.IS_SLEEPING_MODE:
                    continue
                positions = await asyncio.to_thread(bot.get_exchange_positions)
                await bot.update_open_positions_from_exch_positions(positions)
                for symbol, pos in positions.items():
                    side = pos.get("side")
                    entry_price = Decimal(str(pos.get("avg_price", 0)))
                    current_price = await bot.get_last_close_price(symbol)
                    if current_price is None:
                        logger.debug(f"[HTTP Monitor] Нет текущей цены для {symbol}")
                        continue
                    current_price = Decimal(str(current_price))
                    if side.lower() == "buy":
                        ratio = (current_price - entry_price) / entry_price
                    else:
                        ratio = (entry_price - current_price) / entry_price
                    profit_perc = (ratio * bot.PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
                    logger.info(f"[HTTP Monitor] User {user_id} {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%")
                    if profit_perc <= -bot.TARGET_LOSS_FOR_AVERAGING:
                        logger.info(f"[HTTP Monitor] {symbol} (User {user_id}) достиг порога убытка ({profit_perc}% <= -{bot.TARGET_LOSS_FOR_AVERAGING}). Открываю усредняющую позицию.")
                        if profit_perc <= -bot.TARGET_LOSS_FOR_AVERAGING:
                            current_volume = Decimal(str(bot.open_positions[symbol]["position_volume"]))
                            logger.info(f"[HTTP Monitor] {symbol} (User {user_id}) достиг порога убытка ({profit_perc}% <= -{bot.TARGET_LOSS_FOR_AVERAGING}). Открываю усредняющую позицию с объёмом {current_volume} USDT.")
                            await bot.open_averaging_position_all(symbol, current_volume)
                    default_leverage = Decimal("10")
                    leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))
                    threshold_trailing = Decimal("5.0")
                    if bot.CUSTOM_TRAILING_STOP_ENABLED:
                    #    await bot.check_and_set_insurance_stop_loss()
                        await bot.apply_custom_trailing_stop(symbol, pos, leveraged_pnl_percent, side)
                    #    await bot.set_stop_loss_to_fast_st(symbol, side)
                    elif bot.supertrend_custom_trailing_stop:
                        await bot.apply_supertrend_custom_trailing_stop(symbol, pos, leveraged_pnl_percent, side)
                    else:
                        if leveraged_pnl_percent >= threshold_trailing and not pos.get("trailing_stop_set", False):
                            logger.info(f"[HTTP Monitor] {symbol}: Достигнут уровень для трейлинг-стопа (leveraged PnL = {leveraged_pnl_percent}%). Устанавливаю трейлинг-стоп.")
                            await bot.set_trailing_stop(symbol, pos["size"], bot.TRAILING_GAP_PERCENT, side)
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Ошибка в monitor_positions_http: {e}")
            await asyncio.sleep(5)

async def main_coroutine():
    try:
        await init_user_bots()
        if not user_bots:
            logger.error("No users loaded! Check users.csv file")
            return
        setup_telegram_bot()
        if not telegram_bot:
            logger.error("Telegram bot not initialized!")
            return
        telegram_task = asyncio.create_task(start_telegram_polling())
        trading_tasks = [asyncio.create_task(bot.main_loop()) for bot in user_bots.values()]
        
        # Пример вызова обновления позиций при запуске:
        #monitor_http_task = asyncio.create_task(monitor_positions_http())

        monitor_tasks = []
        loop = asyncio.get_running_loop()

        for bot in user_bots.values():
            if bot.monitoring == "http":
            #    monitor_tasks.append(asyncio.create_task(bot.monitor_positions_ws()))
            #    monitor_tasks.append(asyncio.create_task(bot.start_monitoring_via_ws()))
                monitor_tasks.append(asyncio.create_task(monitor_positions_http()))
                monitor_tasks.append(asyncio.create_task(bot.init_ticker_websocket()))
            #    monitor_tasks.append(asyncio.create_task(bot.realtime_position_monitor()))
            elif bot.monitoring in ("demo_ws", "real_ws"):
                monitor_tasks.append(asyncio.create_task(bot.start_websocket_listener()))
            #    monitor_tasks.append(asyncio.create_task(bot.start_websocket_listener()))
            #    monitor_tasks.append(await bot.start_websocket_listener())
            #    await bot.init_websocket()
            #    asyncio.create_task(bot.start_websocket_listener())
#                asyncio.create_task(bot.periodic_position_status_report())

        results = await asyncio.gather(telegram_task, *monitor_tasks, *trading_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task raised an exception: {result}")
    except Exception as e:
        logger.exception(f"Main coroutine error: {e}")

def main():
    try:
        asyncio.run(main_coroutine())
    except KeyboardInterrupt:
        logger.info("Остановка пользователем.")
    except Exception as e:
        logger.exception(f"Ошибка main: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Нажато Ctrl+C. Завершение работы...")