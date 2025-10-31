#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Многопользовательский бот для торговли на Bybit с использованием модели, дрейфа, супер-тренда и т.д.
Часть 1: Импорты библиотек, глобальные переменные и идентификация пользователей.
"""

# Импорты всех необходимых библиотек
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
import sys
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
from requests.exceptions import ReadTimeout
from urllib3.util.retry import Retry

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

# Загрузка переменных окружения (например, для TELEGRAM_TOKEN)
load_dotenv()

# ----------------------------------------------------------------------
# Глобальные переменные и настройки
# ----------------------------------------------------------------------

load_dotenv("keys_TESTNET2.env")  # ожидаются BYBIT_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID и т.д.

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

# Трейлинг-стоп параметры
TRAILING_STOP_ENABLED = True
TRAILING_GAP_PERCENT = Decimal("0.007")     # 0.8%
MIN_TRAILING_STOP = Decimal("0.0000001")

# Режимы работы бота
QUIET_PERIOD_ENABLED = False                # режим тихого периода
IS_SLEEPING_MODE = False                    # спящий режим
OPERATION_MODE = "ST_cross2"                # торговые режимы (например, drift_only, drift_top10, golden_setup, model_only, super_trend, ST_cross1, ST_cross2, ST_cross_global, ST_cross2_drift)
HEDGE_MODE = True
INVERT_MODEL_LABELS = False

# Файл модели и параметры обучения
MODEL_FILENAME = "trading_model_final.pkl"
MIN_SAMPLES_FOR_TRAINING = 1000

ADMIN_ID = 36972091  # Укажите свой user_id, кто имеет право останавливать бота

# Параметры для дрейфа
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

# Настройка SSL
ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())

# Другие глобальные объекты
telegram_bot = None
router = Router()
router_admin = Router()
telegram_message_queue = None
send_semaphore = asyncio.Semaphore(10)
MAX_CONCURRENT_THREADS = 5
thread_semaphore = ThreadPoolExecutor(MAX_CONCURRENT_THREADS)
drift_trade_executed = False

open_positions_lock = threading.Lock()
open_positions = {}  # Ключ – символ, значение – данные позиции

history_lock = threading.Lock()
open_interest_history = defaultdict(list)
volume_history = defaultdict(list)

# ----------------------------------------------------------------------
# Функция загрузки пользователей
# ----------------------------------------------------------------------

def load_users(filename="users.csv"):
    """
    Загрузка списка пользователей из CSV. Теперь ожидаем колонки:
    user_id, user_api, user_api_secret, mode, max_total_volume, position_volume.
    Если каких-то колонок нет, берём значения по умолчанию.
    """
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

                        # Новые поля
                        max_total_volume_str = row.get("max_total_volume", "1000").strip()
                        position_volume_str = row.get("position_volume", "100").strip()

                        # Запоминаем всё как словарь (для удобства)
                        users_dict[user_id] = {
                            "user_api": user_api,
                            "user_api_secret": user_api_secret,
                            "mode": mode,
                            "max_total_volume": max_total_volume_str,
                            "position_volume": position_volume_str
                        }
                        logger.info(f"Loaded user {user_id}, mode={mode}, max_total_volume={max_total_volume_str}, position_volume={position_volume_str}")

                    except Exception as e:
                        logger.error(f"Error loading user from row={row}: {e}")
        else:
            logger.error(f"Users file {filename} not found!")
    except Exception as e:
        logger.error(f"Critical error loading users: {e}")

    return users_dict

def save_users(users_dict, filename="users.csv"):
    """
    Сохраняет словарь users_dict (user_id -> {ключи}) в CSV-файл.
    """
    try:
        # Определим набор колонок. Можно хранить жёстко, чтобы порядок не прыгал.
        fieldnames = [
            "user_id",
            "user_api",
            "user_api_secret",
            "mode",
            "max_total_volume",
            "position_volume"
        ]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for uid, data in users_dict.items():
                row = {
                    "user_id": uid,
                    "user_api": data["user_api"],
                    "user_api_secret": data["user_api_secret"],
                    "mode": data["mode"],
                    "max_total_volume": data["max_total_volume"],
                    "position_volume": data["position_volume"]
                }
                writer.writerow(row)
        logger.info(f"save_users: файл {filename} успешно перезаписан.")
    except Exception as e:
        logger.error(f"save_users: ошибка сохранения {filename}: {e}")

# Идентификация пользователей
users = load_users()
logger.info(f"Загружено пользователей: {list(users.keys())}")
user_bots = {}

class RegisterStates(StatesGroup):
    waiting_for_api_key = State()
    waiting_for_api_secret = State()
    waiting_for_mode = State()

def add_user_to_csv(user_id: int, user_api: str, user_api_secret: str, filename="users.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user_id", "user_api", "user_api_secret"])
        writer.writerow([user_id, user_api, user_api_secret])

def init_user_bots():
    for uid, user_data in users.items():
        # user_data - это словарь с полями: user_api, user_api_secret, mode, max_total_volume, position_volume
        api = user_data["user_api"]
        secret = user_data["user_api_secret"]
        mode = user_data["mode"]
        max_vol_str = user_data["max_total_volume"]
        pos_vol_str = user_data["position_volume"]

        bot_instance = TradingBot(
            user_id=uid,
            user_api=api,
            user_api_secret=secret,
            mode=mode,
            max_total_volume=max_vol_str,
            position_volume=pos_vol_str
        )
        user_bots[uid] = bot_instance
        logger.info(f"Создан бот для user_id={uid} (mode={mode})")

########################################################################
# Логика бота TradingBot – постоянная работа с историческими данными,
# вычисление индикаторов, формирование признаков для модели, загрузка и 
# переобучение предиктивной модели, ведение логов торговых операций.
########################################################################

class TradingBot:
    def __init__(self, user_id: int, user_api: str, user_api_secret: str, mode: str,
                 max_total_volume="1000", position_volume="100"):
        self.user_id = user_id
        self.user_api = user_api
        self.user_api_secret = user_api_secret
        self.mode = mode.lower()

        # Сохраняем новые параметры (можно в Decimal, можно в float — как удобнее)
        self.MAX_TOTAL_VOLUME = Decimal(str(max_total_volume))
        self.POSITION_VOLUME = Decimal(str(position_volume))

        if self.mode == "demo":
            # Торговля на демо-аккаунте (Testnet)
            self.session = HTTP(
                demo=True,
                api_key=self.user_api,
                api_secret=self.user_api_secret,
                timeout=60,
            )
        else:
            # Реальная торговля (Mainnet)
            self.session = HTTP(
                testnet=False,
                api_key=self.user_api,
                api_secret=self.user_api_secret,
                timeout=60,
            )

        # Добавьте инициализацию last_kline_data
        self.last_kline_data = {}

        self.state = {}
        self.state["connectivity_ok"] = True  # Бот считает связь доступной по умолчанию
        self.open_positions = {}         # открытые позиции для данного пользователя
        self.drift_history = defaultdict(list)
        self.selected_symbols = []
        
        self.MAX_TOTAL_VOLUME = MAX_TOTAL_VOLUME
        self.POSITION_VOLUME = POSITION_VOLUME
        self.PROFIT_LEVEL = PROFIT_LEVEL
        self.PROFIT_COEFFICIENT = PROFIT_COEFFICIENT

        self.TAKE_PROFIT_ENABLED = TAKE_PROFIT_ENABLED
        self.TAKE_PROFIT_LEVEL = TAKE_PROFIT_LEVEL

        self.TRAILING_STOP_ENABLED = TRAILING_STOP_ENABLED
        self.TRAILING_GAP_PERCENT = TRAILING_GAP_PERCENT
        self.MIN_TRAILING_STOP = MIN_TRAILING_STOP
        self.CUSTOM_TRAILING_STOP_ENABLED = True  # False => используем старую логику, True => кастовый трейлинг

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

        self.MONITOR_MODE = "http"

        self.state_lock = threading.Lock()
        self.open_positions_lock = threading.Lock()
        self.history_lock = threading.Lock()

        self.current_model = None

        self.last_asset_selection_time = 0
        self.ASSET_SELECTION_INTERVAL = 60 * 60

        self.historical_data = pd.DataFrame()  # переменная для хранения исторических данных
        # Загружаем исторические данные при старте
        self.load_historical_data()


    # -------------------- Методы получения данных --------------------

    async def open_averaging_position_all(self, symbol):
        try:
            with self.open_positions_lock:
                if symbol not in self.open_positions:
                    logger.info(f"[Averaging] Нет базовой позиции для {symbol}, пропуск усреднения.")
                    return
                if symbol in self.averaging_positions:
                    logger.info(f"[Averaging] Усредняющая позиция для {symbol} уже открыта, пропуск.")
                    return
                base_pos = self.open_positions[symbol]
                side = base_pos["side"]
                base_volume_usdt = Decimal(str(base_pos["position_volume"]))
                
                # Если позиция не в убытке, усреднение не требуется.
                if "profit_perc" in base_pos and base_pos["profit_perc"] >= 0:
                    logger.info(f"[Averaging] Позиция {symbol} не в убытке, усреднение не требуется.")
                    return

                if self.averaging_total_volume + base_volume_usdt > self.MAX_AVERAGING_VOLUME:
                    logger.info(
                        f"[Averaging] Превышен лимит усреднения: {self.averaging_total_volume} + "
                        f"{base_volume_usdt} > {self.MAX_AVERAGING_VOLUME}"
                    )
                    return

            # Выставляем ордер (если place_order — синхронная функция, оборачиваем через asyncio.to_thread)
            order_result = await self.place_order(
                symbol=symbol,
                side=side,
                qty=float(base_volume_usdt),
                order_type="Market",
                time_in_force="GoodTillCancel",
                reduce_only=False,
                positionIdx=1 if side.lower() == "buy" else 2
            )
            if order_result and order_result.get("retCode") == 0:
                with self.open_positions_lock:
                    self.averaging_positions[symbol] = {
                        "side": side,
                        "volume": base_volume_usdt,
                        "opened_at": datetime.datetime.utcnow(),
                        "trade_id": f"averaging_{symbol}_{int(time.time())}"
                    }
                self.averaging_total_volume += base_volume_usdt
                logger.info(
                    f"[Averaging] Усредняющая позиция для {symbol} открыта на объём {base_volume_usdt}. "
                    f"Текущий усредняющий объём: {self.averaging_total_volume}"
                )
            else:
                logger.error(f"[Averaging] Ошибка открытия усредняющей позиции для {symbol}: {order_result}")
        except Exception as e:
            logger.exception(f"[Averaging] Ошибка в open_averaging_position для {symbol}: {e}")

    async def generate_drift_table_from_history(self, drift_history: dict, top_n: int = 15) -> str:
        """
        Генерирует текстовое представление drift-таблицы для конкретного пользователя,
        используя его историю drift (drift_history).
        """
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
        """
        Генерирует текстовое представление таблицы предсказаний модели только для указанного пользователя.
        """
        if not os.path.isfile(csv_path):
            return "Файл с предсказаниями не найден."
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty:
            return "Файл с предсказаниями пуст."
        # Фильтруем записи по user_id
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
        """
        Отправляет таблицы drift и модели в Telegram для конкретного пользователя.
        Здесь trading_bot – это экземпляр TradingBot, который содержит:
        - trading_bot.drift_history
        - trading_bot.user_id (Telegram user id)
        """
        if not telegram_bot:
            logger.info("[publish_drift_and_model_tables] Telegram bot не инициализирован => пропуск.")
            return

        # Генерируем drift-таблицу для пользователя
        drift_str = self.generate_drift_table_from_history(trading_bot.drift_history, top_n=10)
        if drift_str.strip():
            msg = f"```\n{drift_str}\n```"
            await telegram_bot.send_message(
                chat_id=trading_bot.user_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN_V2
            )
        else:
            logger.info("[DRIFT] Таблица дрифта пуста => пропуск.")

        # Генерируем таблицу предсказаний модели для пользователя
        model_str = self.generate_model_table_from_csv_no_time(trading_bot.user_id, csv_path="model_predictions_log.csv", last_n=10)
        if model_str.strip():
            msg = f"```\n{model_str}\n```"
            await telegram_bot.send_message(
                chat_id=trading_bot.user_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN_V2
            )
        else:
            logger.info("[MODEL] Таблица модели пуста => пропуск.")

    # -------------------- Методы получения данных --------------------
    # async def get_historical_data_for_trading(self, symbol: str, interval="1", limit=200, from_time=None):
    #     try:
    #         params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    #         if from_time:
    #             params["from"] = from_time
    #         def _get_kline():
    #             return self.session.get_kline(**params)

    #         resp = await asyncio.to_thread(_get_kline)
    #         if resp.get("retCode") != 0:
    #             logger.error(f"[TRADING_KLINE] {symbol}: {resp.get('retMsg')}")
    #             return pd.DataFrame()
    #         data = resp["result"].get("list", [])
    #         if not data:
    #             return pd.DataFrame()
    #         columns = ["open_time", "open", "high", "low", "close", "volume", "open_interest"]
    #         df = pd.DataFrame(data, columns=columns)
    #         df["startTime"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
    #         df.rename(columns={
    #             "open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"
    #         }, inplace=True)
    #         df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]] = \
    #             df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]].apply(
    #                 pd.to_numeric, errors="coerce"
    #             )
    #         df.dropna(subset=["closePrice"], inplace=True)
    #         df.sort_values("startTime", inplace=True)
    #         df.reset_index(drop=True, inplace=True)
    #         return df[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]]
    #     except Exception as e:
    #         logger.exception(f"[get_historical_data_for_trading({symbol})]: {e}")
    #         return pd.DataFrame()


    async def get_historical_data_for_trading(self, symbol: str, interval="1", limit=200, from_time=None):
        try:
            params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
            if from_time:
                params["from"] = from_time

            def _get_kline():
                return self.session.get_kline(**params)

            resp = await asyncio.to_thread(_get_kline)
            if resp.get("retCode") != 0:
                logger.error(f"[TRADING_KLINE] {symbol}: {resp.get('retMsg')}")
                # Если ошибка – возвращаем кэш, если есть
                if symbol in self.last_kline_data:
                    logger.info(f"[TRADING_KLINE] Использую кэшированные данные для {symbol}")
                    return self.last_kline_data[symbol]
                return pd.DataFrame()

            data = resp["result"].get("list", [])
            if not data:
                if symbol in self.last_kline_data:
                    logger.info(f"[TRADING_KLINE] Данных нет, использую кэш для {symbol}")
                    return self.last_kline_data[symbol]
                return pd.DataFrame()

            columns = ["open_time", "open", "high", "low", "close", "volume", "open_interest"]
            df = pd.DataFrame(data, columns=columns)
            df["startTime"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
            df.rename(columns={
                "open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"
            }, inplace=True)
            df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]] = \
                df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]].apply(pd.to_numeric, errors="coerce")
            df.dropna(subset=["closePrice"], inplace=True)
            df.sort_values("startTime", inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Обновляем кэш
            self.last_kline_data[symbol] = df.copy()
            return df[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]]

        except ReadTimeout as rt:
            logger.error(f"[get_historical_data_for_trading({symbol})]: Таймаут чтения: {rt}")
            if symbol in self.last_kline_data:
                logger.info(f"[get_historical_data_for_trading({symbol})]: Использую кэшированные данные")
                return self.last_kline_data[symbol]
            return pd.DataFrame()
        except Exception as e:
            logger.exception(f"[get_historical_data_for_trading({symbol})]: {e}")
            if symbol in self.last_kline_data:
                logger.info(f"[get_historical_data_for_trading({symbol})]: Использую кэшированные данные")
                return self.last_kline_data[symbol]
            return pd.DataFrame()
    
    def load_historical_data(self):
        """Загружает данные из файла historical_data_for_model_5m.csv и сохраняет их в self.historical_data"""
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

    async def get_last_close_price(self, symbol: str):
        try:
            params = {"category": "linear", "symbol": symbol, "interval": "1", "limit": 1}
            def _get_kline():
                return self.session.get_kline(**params)
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
        return pd.DataFrame({
            'MACD_12_26_9': macd,
            'MACDs_12_26_9': signal_line
        })

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

    # -------------------- Методы отбора пар --------------------
    # def get_selected_symbols(self):
    #     now = time.time()
    #     if now - self.last_asset_selection_time >= self.ASSET_SELECTION_INTERVAL or not self.selected_symbols:
    #         tickers_resp = self.session.get_tickers(symbol=None, category="linear")
    #         if "result" not in tickers_resp or "list" not in tickers_resp["result"]:
    #             logger.error("[get_selected_symbols] Некорректный ответ get_tickers.")
    #             self.selected_symbols = []
    #             return self.selected_symbols
    #         tickers_data = tickers_resp["result"]["list"]
    #         inst_resp = self.session.get_instruments_info(category="linear")
    #         if "result" not in inst_resp or "list" not in inst_resp["result"]:
    #             logger.error("[get_selected_symbols] Некорректный ответ get_instruments_info.")
    #             self.selected_symbols = []
    #             return self.selected_symbols
    #         instruments_data = inst_resp["result"]["list"]
    #         trading_status = {}
    #         for inst in instruments_data:
    #             sym = inst.get("symbol")
    #             stat = inst.get("status", "").upper()
    #             if sym:
    #                 trading_status[sym] = (stat == "TRADING")
    #         usdt_pairs = []
    #         for tk in tickers_data:
    #             sym = tk.get("symbol")
    #             if not sym:
    #                 continue
    #             if "USDT" in sym and "BTC" not in sym and "ETH" not in sym:
    #                 if not trading_status.get(sym, False):
    #                     continue
    #                 turnover24 = Decimal(str(tk.get("turnover24h", "0")))
    #                 volume24 = Decimal(str(tk.get("volume24h", "0")))
    #                 if turnover24 >= Decimal("2000000") and volume24 >= Decimal("2000000"):
    #                     usdt_pairs.append(sym)
    #         self.selected_symbols = usdt_pairs
    #         self.last_asset_selection_time = now
    #         logger.info(f"Обновлён список активов: {self.selected_symbols}")
    #     return self.selected_symbols

    def get_selected_symbols(self):
        now = time.time()
        # Обновляем историю сразу, если прошло достаточно времени или список пустой
        if now - self.last_asset_selection_time >= self.ASSET_SELECTION_INTERVAL or not self.selected_symbols:
            # Загружаем исторические данные
            self.load_historical_data()
            # Далее идёт отбор пар, как и раньше:
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

    async def get_last_row(self, symbol: str):
        df = await self.get_historical_data_for_trading(symbol, interval="1", limit=1)
        if df.empty:
            return None
        return df.iloc[-1]

    # -------------------- Методы расчёта SuperTrend --------------------
    async def calculate_supertrend_bybit_34_2(self, df: pd.DataFrame, length=8, multiplier=3.0) -> pd.DataFrame:
        try:
            if df.empty:
                return pd.DataFrame()

            def extend_value(current_value, previous_value):
                return previous_value if pd.isna(current_value) or current_value == 0 else current_value

            for col in ["highPrice", "lowPrice", "closePrice"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan).fillna(method='ffill')
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
            logger.exception(f"Ошибка в calculate_supertrend_bybit_34_2: {e}")
            return pd.DataFrame()

    async def calculate_supertrend_bybit_8_1(self, df: pd.DataFrame, length=3, multiplier=1.0) -> pd.DataFrame:
        try:
            if df.empty:
                return pd.DataFrame()

            def extend_value(current_value, previous_value):
                return previous_value if pd.isna(current_value) or current_value == 0 else current_value

            for col in ["highPrice", "lowPrice", "closePrice"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan).fillna(method='ffill')
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

    # -------------------- Методы загрузки/обучения модели --------------------
    async def train_and_load_model(self, csv_path="historical_data_for_model_5m.csv"):
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
                df_sym = self.prepare_features_for_model(df_sym)
                if df_sym.empty:
                    continue
                df_sym = self.make_multiclass_target_for_model(df_sym, horizon=1, threshold=Decimal("0.0025"))
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

    def load_model(self):
        try:
            model = joblib.load(self.MODEL_FILENAME)
            return model
        except (ModuleNotFoundError, ImportError):
            logger.warning("Не удалось загрузить модель. Будет создана новая.")
            return self.train_and_load_model()

    async def maybe_retrain_model(self):
        new_model = self.train_and_load_model(csv_path="historical_data_for_model_5m.csv")
        if new_model:
            self.current_model = new_model
            logger.info(f"[maybe_retrain_model] Пользователь {self.user_id}: Модель успешно обновлена.")

    # -------------------- Методы для модели и торговли --------------------
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
        try:
            params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
            if from_time:
                params["from"] = from_time
            resp = self.session.get_kline(**params)
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
        try:
            info = self.get_symbol_info(symbol)  # Предполагаем, что этот метод синхронный
            if not info:
                logger.warning(f"[adjust_quantity] get_symbol_info({symbol}) вернул None, qty={raw_qty}")
                return 0.0
            lot = info.get("lotSizeFilter", {})
            min_qty = Decimal(str(lot.get("minOrderQty", "0")))
            qty_step = Decimal(str(lot.get("qtyStep", "1")))
            max_qty = Decimal(str(lot.get("maxOrderQty", "9999999")))
            min_order_value = Decimal(str(info.get("minOrderValue", 5)))
            last_price = await self.get_last_close_price(symbol)  # await!
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
        except Exception as e:
            logger.exception(f"[adjust_quantity] {symbol}: {e}")
            return 0.0

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
            with self.history_lock:
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
        # Здесь вторым аргументом передаём пустой DataFrame для ref_data – можно настроить, если есть исторические данные
        result = await self.monitor_feature_drift_per_symbol(symbol, df_trading, pd.DataFrame(), feature_cols, threshold=0.5)
        # Логгируем результат для данного символа
        is_anomaly, strength, direction = result
        if is_anomaly:
            logger.info(f"[Drift] {symbol}: аномалия обнаружена, strength={strength:.3f}, direction={direction}")
        return result

    # -------------------- Методы трейлинг-стопа --------------------
    async def check_and_set_trailing_stop(self):
        if not self.TRAILING_STOP_ENABLED:
            return
        try:
            with self.open_positions_lock:
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
                leveraged_pnl_percent = ratio * default_leverage * Decimal("100")
                with self.open_positions_lock:
                    if sym in self.open_positions:
                        self.open_positions[sym]['profit_perc'] = (ratio * self.PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
                if leveraged_pnl_percent >= threshold_roi:
                    if not pos.get("trailing_stop_set", False):
                        logger.info(f"[Trailing Stop] {sym}: Уровень достигнут (leveraged PnL = {leveraged_pnl_percent}%). Устанавливаю трейлинг-стоп.")
                        await self.set_trailing_stop(sym, pos["size"], self.TRAILING_GAP_PERCENT, side)
        except Exception as e:
            logger.exception(f"Ошибка check_and_set_trailing_stop: {e}")

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
            trailing_distance_abs = (avg_price * trailing_gap_percent).quantize(Decimal("0.0000001"))
            dynamic_min = max(avg_price * Decimal("0.0000001"), self.MIN_TRAILING_STOP)
            if trailing_distance_abs < dynamic_min:
                logger.info(f"[set_trailing_stop] {symbol}: trailingStop={trailing_distance_abs} < {dynamic_min}, пропуск.")
                return
            resp = self.session.set_trading_stop(
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
                    with self.open_positions_lock:
                        if symbol in self.open_positions:
                            self.open_positions[symbol]["trailing_stop_set"] = True
                        pnl_display = self.open_positions.get(symbol, {}).get("profit_perc", Decimal("0"))
                    row = await self.get_last_row(symbol)
                    await self.log_trade(self.user_id, symbol, row, None, f"{trailing_distance_abs} (PnL: {pnl_display}%)", "Trailing Stop Set", closed_manually=False)
                    logger.info(f"[set_trailing_stop] OK {symbol}")
                elif rc == 34040:
                    logger.info("[set_trailing_stop] not modified, retCode=34040.")
                else:
                    logger.error(f"[set_trailing_stop] Ошибка: {resp.get('retMsg')}")
        except Exception as e:
            logger.exception(f"[set_trailing_stop] {symbol}: {e}")

    # async def apply_custom_trailing_stop(self, symbol, pos, leveraged_pnl_percent, side):
    #     """
    #     Кастовый механизм трейлинг-стопа:
    #     1. Ждём, пока PnL >= 5% (с учётом плеча).
    #     2. Ставим (или двигаем) обычный stop-loss в зоне 2% прибыли.
    #     3. Если прибыль растёт выше - тянем стоп выше.
    #     """
    #     # Допустим, начинаем кастовый стоп, когда достигли 5% (с учётом плеча)
    #     START_CUSTOM_TRAIL = Decimal("5.0")
    #     # И стоп хотим ставить так, чтобы позиция в случае возврата выйдет с +2% (с учётом плеча)
    #     TRAIL_OFFSET = Decimal("2.0")

    #     # Если позиция ещё не достигла 5% прибыли, ничего не делаем
    #     if leveraged_pnl_percent < START_CUSTOM_TRAIL:
    #         return

    #     # Когда превысили 5%, нужно ставить стоп таким образом,
    #     # чтобы зафиксировать 2% из текущих ~X% (если X > 5).
    #     # Пример: если PnL сейчас 6%, значит стоп ставим на +4%,
    #     # если сейчас 8%, стоп на +6 и т.д.
    #     # Но в условии сказано «установить стоп при достижении 5% – на 2%»,
    #     # и «смещаем ... на величину = движению прибыли». Интерпретация:
    #     #   - как только мы пересекаем 5%, ставим стоп на 2%.
    #     #   - если PnL растёт дальше (say 6%, 7%), двигаем стоп (4%, 5%),
    #     #     то есть «стоп = (текущий PnL) - 3%».
    #     # Но ниже упрощённо: стоп = PnL - 3%, но не опускать ниже +2%!

    #     desired_stop = leveraged_pnl_percent - (START_CUSTOM_TRAIL - TRAIL_OFFSET)
    #     # Если сейчас 5%, desired_stop = 2%. Если 6%, desired_stop=3%, если 7% => 4% и т.д.

    #     # Но не опускать ниже 2%, если прибыль вдруг упала:
    #     if desired_stop < TRAIL_OFFSET:
    #         desired_stop = TRAIL_OFFSET

    #     # Теперь переводим этот desired_stop в «процент от входа» без плеча.
    #     # leveraged_pnl_percent = ratio * 100 * 10
    #     # => ratio = leveraged_pnl_percent / 100 / 10
    #     # => desired_ratio = desired_stop / 100 / 10
    #     desired_ratio = desired_stop / Decimal("100") / Decimal("10")
        
    #     # Для BUY: стоп цена = entry_price * (1 + desired_ratio)
    #     # Для SELL: стоп цена = entry_price * (1 - desired_ratio)
    #     entry_price = Decimal(str(pos.get("avg_price", 0)))
    #     if entry_price == 0:
    #         return

    #     if side.lower() == "buy":
    #         stop_price = entry_price * (Decimal("1") + desired_ratio)
    #     else:
    #         stop_price = entry_price * (Decimal("1") - desired_ratio)

    #     # Посылаем запрос на выставление (или модификацию) стоп-лосса.
    #     logger.info(f"[CustomTrailingStop] {symbol}: leveraged_pnl={leveraged_pnl_percent}%, ставим SL на ~{desired_stop}% => stop_price={stop_price:.4f}")

    #     # Вызываем отдельную функцию, которая выставит обычный StopLoss (не trailingStop).
    #     # Можно назвать set_stop_loss(...), аналогично set_trailing_stop(...).
    #     await self.set_fixed_stop_loss(symbol, pos["size"], side, stop_price)
    #     # внутри apply_custom_trailing_stop:
    #     await self.log_trade(
    #         user_id=self.user_id,
    #         symbol=symbol,
    #         row=None,      # можно не передавать, если не нужно
    #         side=side,
    #         open_interest=None,
    #         action=f"Установлен каст.стоп, PnL={leveraged_pnl_percent}%",
    #         result="TrailingStop",
    #         closed_manually=False
    #     )

    async def apply_custom_trailing_stop(self, symbol, pos, leveraged_pnl_percent, side):
        """
        Кастовый трейлинг-стоп:
        - включается, когда текущая (с учётом плеча) прибыль >= 5%;
        - стоп ставим на (leveraged_pnl - X)%, при этом не даём стопу опуститься обратно,
          т.е. если новая цель меньше предыдущей -- ничего не меняем.
        """
        START_CUSTOM_TRAIL = Decimal("5.0")    # c какой прибыли начинаем trailing
        TRAIL_OFFSET = Decimal("3.0")         # например, хотим зафиксировать (pnl - 3%) 

        # Ниже 5% — трейлинг не включается:
        if leveraged_pnl_percent < START_CUSTOM_TRAIL:
            return

        # Предположим, хотим поставить stop на (текущая прибыль - 3%).
        # Пример: если сейчас 6%, то стоп будет на 3%, если 8% — на 5%, и т.д.
        desired_stop = leveraged_pnl_percent - TRAIL_OFFSET
        if desired_stop < Decimal("0"):
            # Теоретически, если 5% - 3% = 2%, здесь все ок
            desired_stop = Decimal("0")

        # Из локального словаря pos может не сохраниться информация, поэтому
        # мы возьмём ту же запись из self.open_positions, чтобы сохранить там «максимальный стоп».
        with self.open_positions_lock:
            pos_in_bot = self.open_positions.get(symbol)
            if not pos_in_bot:
                # Например, позиция уже куда-то делась
                return

            # Предыдущий зафиксированный уровень (0, если его нет)
            old_stop = pos_in_bot.get("custom_stop_loss_percent", Decimal("0"))

            # Если новый уровень ниже старого, значит прибыль упала => не двигаем стоп
            if desired_stop <= old_stop:
                # Не меняем стоп на более низкий
                return

            # Иначе обновляем «максимальный» уровень стопа
            pos_in_bot["custom_stop_loss_percent"] = desired_stop

        # Теперь считаем, где должна быть стоп-цена для этого %:
        # leveraged_pnl_percent = (ratio * 100 * leverage)
        # => ratio = leveraged_pnl_percent / 100 / leverage
        # => stop_ratio = desired_stop / 100 / leverage
        # Для BUY: stop_price = entry_price * (1 + stop_ratio)
        # Для SELL: stop_price = entry_price * (1 - stop_ratio)

        entry_price = Decimal(str(pos.get("avg_price", 0)))
        if entry_price <= 0:
            return

        leverage = Decimal("10")  # ваша константа
        stop_ratio = desired_stop / Decimal("100") / leverage

        if side.lower() == "buy":
            stop_price = entry_price * (Decimal("1") + stop_ratio)
        else:
            stop_price = entry_price * (Decimal("1") - stop_ratio)

        logger.info(
            f"[CustomTrailingStop] {symbol}: тек. pnl={leveraged_pnl_percent}%, "
            f"двигаем стоп на {desired_stop}% => цена {stop_price:.4f}"
        )

        await self.set_fixed_stop_loss(symbol, pos["size"], side, stop_price)
        await self.log_trade(
            user_id=self.user_id,
            symbol=symbol,
            row=None,      # можно не передавать, если не нужно
            side=side,
            open_interest=None,
            action=f"Установлен каст.стоп, PnL={leveraged_pnl_percent}%",
            result="TrailingStop",
            closed_manually=False
            )


    async def set_fixed_stop_loss(self, symbol, size, side, stop_price):
        """
        Выставляем обычный стоп-лосс (StopLoss) на stop_price, без сдвига обратно.
        """
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
            def _call():
                return self.session.set_trading_stop(**params)
            resp = await asyncio.to_thread(_call)
            if resp.get("retCode") == 0:
                logger.info(f"[set_fixed_stop_loss] {symbol}: стоп-лосс выставлен на {stop_price}")
            else:
                logger.error(f"[set_fixed_stop_loss] Ошибка: {resp.get('retMsg')}")
        except Exception as e:
            logger.exception(f"[set_fixed_stop_loss] {symbol}: {e}")
            

    async def set_fixed_stop_loss(self, symbol, size, side, stop_price):
        """
        Выставляет обычный стоп-лосс (StopLoss) по заданной цене.
        Пример: для Bybit v5 (unified endpoints) это orderType="StopLoss".
        Или можно использовать set_trading_stop(...), но через stopLoss параметр.
        """
        position_info = self.get_position_info(symbol, side)
        if not position_info:
            logger.error(f"[set_fixed_stop_loss] Нет позиции {symbol}/{side}")
            return
        pos_idx = position_info.get("positionIdx")
        if not pos_idx:
            return

        # Пример вызова: set_trading_stop с stopLoss
        # (Документация: https://bybit-exchange.github.io/docs/v5/position/trade-stop)
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "positionIdx": pos_idx,
            "stopLoss": str(stop_price),  # строка
            "timeInForce": "GoodTillCancel",
        }
        try:
            def _call():
                return self.session.set_trading_stop(**params)
            resp = await asyncio.to_thread(_call)
            if resp.get("retCode") == 0:
                logger.info(f"[set_fixed_stop_loss] {symbol}: стоп-лосс выставлен на {stop_price}")
            else:
                logger.error(f"[set_fixed_stop_loss] Ошибка: {resp.get('retMsg')}")
        except Exception as e:
            logger.exception(f"[set_fixed_stop_loss] {symbol}: {e}")

    # -------------------- Методы работы с ордерами и позициями --------------------
    async def open_position(self, symbol: str, side: str, volume_usdt: Decimal, reason: str):
        if not self.state.get("connectivity_ok", True):
            logger.warning(f"[open_position] Связь с биржей нестабильна или прервана! "
                       f"Открытие позиции для {symbol} блокируется.")
            return  # Выходим без открытия

        
        if self.IS_SLEEPING_MODE:
            logger.info(f"[open_position] Бот в спящем режиме, открытие {symbol} отменено.")
            return
        try:
            logger.info(f"[open_position] Попытка открытия {side} {symbol}, объем: {volume_usdt} USDT, причина: {reason}")
            
            # Атомарно проверяем и резервируем объём
            with self.state_lock, self.open_positions_lock:
                current_total = sum(Decimal(str(pos.get("position_volume", 0))) for pos in self.open_positions.values())
                if current_total + volume_usdt > self.MAX_TOTAL_VOLUME:
                    logger.warning(f"[open_position] Превышен глобальный лимит: {current_total} + {volume_usdt} > {self.MAX_TOTAL_VOLUME}")
                    return
                if symbol in self.open_positions:
                    logger.info(f"[open_position] Позиция для {symbol} уже открыта, пропуск.")
                    return
                # Резервируем объём, создавая placeholder для новой позиции
                self.open_positions[symbol] = {
                    "side": side,
                    "size": None,
                    "avg_price": None,
                    "position_volume": volume_usdt,  # резервирование объёма
                    "symbol": symbol,
                    "trailing_stop_set": False,
                    "trade_id": None,
                    "open_time": datetime.datetime.utcnow()
                }
                self.state["total_open_volume"] = current_total + volume_usdt

            # Далее выполняем асинхронные операции
            last_price = await self.get_last_close_price(symbol)
            if not last_price or last_price <= 0:
                logger.info(f"[open_position] Нет актуальной цены для {symbol}, пропуск.")
                # Снимаем резервирование при ошибке
                with self.open_positions_lock:
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

            # Выставление ордера выполняем в отдельном потоке (так как place_order — синхронная)
            order_res = await self.place_order(symbol=symbol, side=side, qty=qty_float, order_type="Market", positionIdx=pos_idx)
            if not order_res or order_res.get("retCode") != 0:
                logger.info(f"[open_position] Ошибка place_order для {symbol}, пропуск.")
                with self.open_positions_lock:
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]
                return

            # Обновляем информацию по позиции
            with self.state_lock, self.open_positions_lock:
                self.open_positions[symbol] = {
                    "side": side,
                    "size": qty_float,
                    "avg_price": float(last_price),
                    "position_volume": float(volume_usdt),
                    "symbol": symbol,
                    "trailing_stop_set": False,
                    "trade_id": trade_id,
                    "open_time": datetime.datetime.utcnow()
                }
            row = await self.get_last_row(symbol)
            await self.log_trade(
                user_id=self.user_id,
                symbol=symbol,
                row=row,
                side=side,                   # Позиция / "Buy" или "Sell"
                open_interest=None,          # Или что-то осмысленное
                action=side,                 # Для log_trade, чтобы if action.lower() == "buy": ...
                result="Opened",             # Именно "Opened" (для if result == "Opened")
                closed_manually=False
            )
            logger.info(f"[open_position] {symbol}: {side} успешно открыта, объем {volume_usdt} USDT")
        except Exception as e:
            logger.exception(f"[open_position] Ошибка: {e}")

    async def place_order(self, symbol, side, qty, order_type="Market", time_in_force="GoodTillCancel", reduce_only=False, positionIdx=None):
        try:
            # Ожидаем вычисление скорректированного количества
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
        except InvalidRequestError as e:
            logger.exception(f"[place_order] InvalidRequestError({symbol}): {e}")
            return None
        except Exception as e:
            logger.exception(f"[place_order] Ошибка({symbol}): {e}")
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

    async def log_model_features_for_trade(self, trade_id: str, symbol: str, side: str, features: dict):
        csv_filename = self.REAL_TRADES_FEATURES_CSV
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

    async def update_trade_outcome(self, trade_id: str, pnl: float):
        csv_filename = self.REAL_TRADES_FEATURES_CSV
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

    # async def log_trade(self, user_id: int, symbol: str, row, side, open_interest,
    #                     action: str, result: str, closed_manually: bool = False,
    #                     csv_filename: str = "trade_log.csv"):
    #     try:
    #         # Запись в CSV
    #         file_exists = os.path.isfile(csv_filename)
    #         time_str = "N/A"
    #         open_str = "N/A"
    #         high_str = "N/A"
    #         low_str  = "N/A"
    #         close_str= "N/A"
    #         vol_str  = "N/A"

    #         if row is not None:
    #             if isinstance(row, dict):
    #                 time_val = row.get("startTime", None)
    #                 open_str = str(row.get("openPrice", "N/A"))
    #                 high_str = str(row.get("highPrice", "N/A"))
    #                 low_str  = str(row.get("lowPrice", "N/A"))
    #                 close_str= str(row.get("closePrice", "N/A"))
    #                 vol_str  = str(row.get("volume", "N/A"))
    #             else:
    #                 time_val = row.get("startTime", None)
    #                 open_str = str(row.get("openPrice", "N/A"))
    #                 high_str = str(row.get("highPrice", "N/A"))
    #                 low_str  = str(row.get("lowPrice", "N/A"))
    #                 close_str= str(row.get("closePrice", "N/A"))
    #                 vol_str  = str(row.get("volume", "N/A"))

    #             if isinstance(time_val, datetime.datetime):
    #                 time_str = time_val.strftime("%Y-%m-%d %H:%M:%S")
    #             else:
    #                 time_str = str(time_val)

    #         oi_str = str(open_interest) if open_interest is not None else "N/A"

    #         with open(csv_filename, "a", newline="", encoding="utf-8") as f:
    #             writer = csv.writer(f)
    #             if not file_exists:
    #                 writer.writerow([
    #                     "user_id", "symbol", "timestamp",
    #                     "openPrice", "highPrice", "lowPrice", "closePrice", "volume",
    #                     "open_interest", "action", "result", "closed_manually"
    #                 ])
    #             writer.writerow([
    #                 user_id, symbol, time_str,
    #                 open_str, high_str, low_str, close_str, vol_str,
    #                 oi_str, action, result, str(closed_manually)
    #             ])

    #         # Экранирование для MarkdownV2
    #         s_symbol  = self.escape_markdown(symbol)
    #         s_time    = self.escape_markdown(time_str)
    #         s_open    = self.escape_markdown(open_str)
    #         s_close   = self.escape_markdown(close_str)
    #         s_vol     = self.escape_markdown(vol_str)
    #         s_action  = self.escape_markdown(action)
    #         s_result  = self.escape_markdown(result)
    #         s_user    = self.escape_markdown(str(user_id))
    #         s_manually= "вручную" if closed_manually else "по сигналу"

    #         # Формирование сообщения в MarkdownV2
    #         if result == "Opened":
    #             if action.lower() == "Buy":
    #                 msg = (f"🟩 *Открытие ЛОНГ-позиции*\n"
    #                     f"*User:* {s_user}\n"
    #                     f"*Символ:* {s_symbol}\n"
    #                     f"*Время:* {s_time}\n"
    #                     f"*Цена открытия:* {s_open}\n"
    #                     f"*Объём:* {s_vol}\n"
    #                     f"*Тип открытия:* {s_action}")
    #             else:
    #                 msg = (f"🟥 *Открытие ШОРТ-позиции*\n"
    #                     f"*User:* {s_user}\n"
    #                     f"*Символ:* {s_symbol}\n"
    #                     f"*Время:* {s_time}\n"
    #                     f"*Цена открытия:* {s_open}\n"
    #                     f"*Объём:* {s_vol}\n"
    #                     f"*Тип открытия:* {s_action}")
    #         elif result == "Closed":
    #             msg = (f"❌ *Закрытие позиции*\n"
    #                 f"*User:* {s_user}\n"
    #                 f"*Символ:* {s_symbol}\n"
    #                 f"*Время:* {s_time}\n"
    #                 f"*Цена закрытия:* {s_close}\n"
    #                 f"*Объём:* {s_vol}\n"
    #                 f"*Тип закрытия:* {s_manually}")
    #         elif result == "Trailing Stop Set":
    #             msg = (f"🔄 *Трейлинг стоп-лосс*\n"
    #                 f"*User:* {s_user}\n"
    #                 f"*Символ:* {s_symbol}\n"
    #                 f"*Время:* {s_time}\n"
    #                 f"*Расстояние:* {s_action} (PnL: {s_result}%)\n"
    #                 f"*Дополнительно:* {s_result}")
    #         else:
    #             msg = (f"🫡🔄 *Сделка*\n"
    #                 f"*User:* {s_user}\n"
    #                 f"*Символ:* {s_symbol}\n"
    #                 f"*Результат:* {s_result}\n"
    #                 f"*Цена:* {s_close}\n"
    #                 f"*Действие:* {s_action}\n"
    #                 f"*Закрытие:* {s_manually}")

    #         # Отправка сообщения в Telegram (при условии, что метод send_telegram_message определён в классе)
    #         await self.send_telegram_message(user_id, msg)

    #     except Exception as e:
    #         logging.error(f"[log_trade] Ошибка выполнения: {e}")

    async def log_trade(
        self, 
        user_id: int,
        symbol: str,
        row,
        side,
        open_interest,
        action: str,
        result: str,
        closed_manually: bool = False,
        csv_filename: str = "trade_log.csv"
    ):
        """
        Записывает сделку в CSV (как раньше) и отправляет HTML-сообщение в Telegram
        с сохранением полной структуры CSV для обучения модели.
        """

        # -----------------------------
        # 1. Преобразуем данные для CSV
        # -----------------------------

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

        # Предположим, что row (если не None) содержит поля: startTime, openPrice, highPrice, lowPrice, closePrice, volume
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

        # Приводим open_interest к строке (если нужно)
        oi_str = str(open_interest) if open_interest is not None else "N/A"

        # Преобразуем bool в строку
        closed_str = str(closed_manually)

        # Это наши будущие значения для CSV
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

        # --------------------------------
        # 2. Пишем в CSV (как у вас раньше)
        # --------------------------------
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "user_id", 
                    "symbol", 
                    "timestamp",
                    "openPrice", 
                    "highPrice", 
                    "lowPrice", 
                    "closePrice", 
                    "volume",
                    "open_interest", 
                    "action", 
                    "result", 
                    "closed_manually"
                ])
            writer.writerow([
                csv_user_id, 
                csv_symbol, 
                csv_timestamp,
                csv_open, 
                csv_high, 
                csv_low, 
                csv_close, 
                csv_volume,
                csv_oi, 
                csv_action, 
                csv_result, 
                csv_closed
            ])

        # --------------------------------------------------------
        # 3. Формируем HTML-сообщение для Telegram (без лишних "\")
        # --------------------------------------------------------

        # Чтобы текст был искабельным, оставляем «Символ: MYSYMBOL».
        # Делать слово «Символ» синим: через <font color="blue">Символ
        # Сам symbol — ссылкой на https://www.bybit.com/trade/usdt/symbol
        link_url = f"https://www.bybit.com/trade/usdt/{symbol}"

        # Для простоты – мы подставим какие-то ветки if result=="Opened" / "Closed" / "TrailingStop" и т.д.
        # Можно расширять по желанию.
        s_manually = "вручную" if closed_manually else "по сигналу"
        s_side     = side if side else ""
        s_result   = (result or "").lower()

        if s_result == "opened":
            # Открытие
            if s_side.lower() == "buy":
                # Лонг
                msg = (
                    f"🟩 <b>Открытие ЛОНГ-позиции</b>\n"
                    f"<b>Символ:</b> "
                    f"<a href=\"{link_url}\">{symbol}</a>\n"
                    f"<b>Пользователь:</b> {user_id}\n"
                    f"<b>Время:</b> {time_str}\n"
                    f"<b>Цена открытия:</b> {open_str}\n"
                    f"<b>Объём:</b> {vol_str}\n"
                    f"<b>Тип открытия:</b> {s_side}"
                )
            else:
                # Шорт
                msg = (
                    f"🟥 <b>Открытие ШОРТ-позиции</b>\n"
                    f"<b>Символ:</b> "
                    f"<a href=\"{link_url}\">{symbol}</a>\n"
                    f"<b>Пользователь:</b> {user_id}\n"
                    f"<b>Время:</b> {time_str}\n"
                    f"<b>Цена открытия:</b> {open_str}\n"
                    f"<b>Объём:</b> {vol_str}\n"
                    f"<b>Тип открытия:</b> {s_side}"
                )

        elif s_result == "closed":
            # Закрытие
            msg = (
                f"❌ <b>Закрытие позиции</b>\n"
                f"<b>Символ:</b> "
                f"<a href=\"{link_url}\">{symbol}</a>\n"
                f"<b>Пользователь:</b> {user_id}\n"
                f"<b>Время закрытия:</b> {time_str}\n"
                f"<b>Цена закрытия:</b> {close_str}\n"
                f"<b>Объём:</b> {vol_str}\n"
                f"<b>Тип закрытия:</b> {s_manually}"
            )

        elif s_result == "trailingstop":
            # Когда кастомный/стандартный trailingStop
            # Пусть action содержит что-то вроде "Установлен, PNL=12.34%"
            msg = (
                f"🔄 <b>Трейлинг-стоп</b>\n"
                f"<b>Символ:</b> "
                f"<a href=\"{link_url}\">{symbol}</a>\n"
                f"<b>Пользователь:</b> {user_id}\n"
                f"<b>Время:</b> {time_str}\n"
                f"<b>Статус:</b> {action}"
            )

        else:
            # Все остальные случаи
            msg = (
                f"🫡🔄 <b>Сделка</b>\n"
                f"<b>Символ:</b> "
                f"<a href=\"{link_url}\">{symbol}</a>\n"
                f"<b>Пользователь:</b> {user_id}\n"
                f"<b>Время:</b> {time_str}\n"
                f"<b>Результат:</b> {result}\n"
                f"<b>Цена:</b> {close_str}\n"
                f"<b>Действие:</b> {action}\n"
                f"<b>Закрытие:</b> {s_manually}"
            )
        await self.send_telegram_message(user_id, msg, parse_mode=ParseMode.HTML)
    

    async def process_symbol_model_only_async(self, symbol):
        if not self.current_model:
            self.current_model = self.load_model()
            if not self.current_model:
                return
        df_5m = await self.get_historical_data_for_model(symbol, "5", limit=200)
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
        try:
            fname = "model_predictions_log.csv"
            file_exists = os.path.isfile(fname)
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

    # -------------------- Основной торговый цикл --------------------
    async def main_loop(self):
        logger.info(f"Запуск основного цикла для пользователя {self.user_id}")
        trading_logic = TradingLogic(self)
        iteration_count = 0
        while self.state.get("run", True) and not self.IS_SLEEPING_MODE:
            try:
                exch_positions = self.get_exchange_positions()
                self.update_open_positions_from_exch_positions(exch_positions)
                usdt_pairs = self.get_usdt_pairs()
                if usdt_pairs:
                    self.selected_symbols = usdt_pairs
                for symbol in self.selected_symbols:
                    df_trading = await self.get_historical_data_for_trading(symbol, interval="1", limit=200)
                    feature_cols = ["openPrice", "closePrice", "highPrice", "lowPrice"]
                    is_anomaly, strength, direction = await self.monitor_feature_drift_per_symbol(symbol, df_trading, pd.DataFrame(), feature_cols, threshold=0.5)
                    if is_anomaly:
                        logger.info(f"[Drift] {symbol}: аномалия обнаружена, strength={strength:.3f}, direction={direction}")
                # if self.OPERATION_MODE == "model_only":
                #     tasks = [asyncio.to_thread(self.process_symbol_model_only_sync, s) for s in self.selected_symbols]
                #     if tasks:
                #         await asyncio.gather(*tasks)
                
                    await trading_logic.execute_trading_mode()

                if iteration_count % 5 == 0:
                    await self.publish_drift_and_model_tables()

                if self.TRAILING_STOP_ENABLED:
                    await self.check_and_set_trailing_stop()
                iteration_count += 1
                if iteration_count % 20 == 0:
                    await self.maybe_retrain_model()
                await asyncio.sleep(60)
            except Exception as e:
                logger.exception(f"Ошибка во внутреннем цикле для пользователя {self.user_id}: {e}")
                await asyncio.sleep(10)
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
            logger.info(f"[get_exchange_positions] BYBIT response => {resp}")
            if resp.get("retCode") != 0:
                logger.error(f"[get_exchange_positions] retCode={resp.get('retCode')} => {resp.get('retMsg')}")
                        # Сбрасываем флаг «связь в порядке?»
                self.state["connectivity_ok"] = False
            
            # Дополнительно: можно вернуть пустой словарь, чтобы сигнализировать,
            # что никакие позиции мы не распознали
                return {}
        
        # Если код дошёл сюда — запрос прошёл успешно, можем восстановить флаг в True
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
            logger.info(f"[get_exchange_positions] => {exchange_positions}")
            return exchange_positions
        except Exception as e:
            logger.exception(f"[get_exchange_positions] Ошибка: {e}")
            return {}

    async def update_open_positions_from_exch_positions(self, expos: dict):
        with self.open_positions_lock, self.state_lock:
            logger.info(f"[update_open_positions_from_exch_positions] BEFORE: {self.open_positions}")
            to_remove = []
            for sym in list(self.open_positions.keys()):
                if sym not in expos:
                    pos = self.open_positions[sym]
                    trade_id = pos.get("trade_id")
                    close_price = await self.get_last_close_price(sym)  # await здесь!
                    if close_price:
                        cp = Decimal(str(close_price))
                        ep = Decimal(str(pos.get("avg_price", 0)))
                        pnl = (cp - ep) / ep * Decimal(str(pos.get("position_volume", 0))) if pos["side"].lower() == "buy" else (ep - cp) / ep * Decimal(str(pos.get("position_volume", 0)))
                        if trade_id:
                            self.update_trade_outcome(trade_id, float(pnl))
                    to_remove.append(sym)
                    await self.log_trade(
                        user_id=self.user_id,
                        symbol=sym,
                        row=None,                # или передайте свечу, если хотите
                        side=pos["side"],
                        open_interest=None,
                        action="TrailingStop closed" if pos.get("trailing_stop_set") else "Closed",
                        result="closed",        # В log_trade у вас ‘closed’ выдаёт «Закрытие позиции»
                        closed_manually=False
                    )

                    to_remove.append(sym)
            for sym in to_remove:
                del self.open_positions[sym]
            for sym, newpos in expos.items():
                if sym in self.open_positions:
                    self.open_positions[sym]["side"] = newpos["side"]
                    self.open_positions[sym]["size"] = newpos["size"]
                    self.open_positions[sym]["avg_price"] = newpos["avg_price"]
                    self.open_positions[sym]["position_volume"] = newpos["position_volume"]
                    self.open_positions[sym]["positionIdx"] = newpos["positionIdx"]
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
                        "open_time": datetime.datetime.utcnow(),
                    }
            total = sum(Decimal(str(p["position_volume"])) for p in self.open_positions.values())
            self.state["total_open_volume"] = total
            logger.info(f"[update_open_positions_from_exch_positions] AFTER: total_open_volume = {total}")


    def escape_markdown(self,text: str) -> str:
        escape_chars = r"_*\[\]()~`>#+\-={}|.,!\\"
        pattern = re.compile(r"([%s])" % re.escape(escape_chars))
        return pattern.sub(r"\\\1", text)

    async def send_telegram_message(self, user_id, message, parse_mode=None):
        """
        Отправляет сообщение через telegram_bot.send_message(...).
        Параметр parse_mode позволяет указывать "HTML", "Markdown" и т.п.
        """
        try:
            if telegram_bot is None:
                logger.error("Telegram bot не инициализирован!")
                return

            # Передаём parse_mode дальше – если None, Telegram не будет делать разметку
            await telegram_bot.send_message(
                chat_id=user_id,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.exception(f"Ошибка отправки сообщения в Telegram: {e}")

########################################################################
# Торговая логика для пользователей (TradingLogic)
# (Все торговые режимы реализованы здесь)
########################################################################

class TradingLogic:
    def __init__(self, trading_bot: TradingBot):
        self.bot = trading_bot

    async def execute_trading_mode(self):
        mode = self.bot.OPERATION_MODE
        logger.info(f"[TradingLogic] Пользователь {self.bot.user_id}: режим {mode}")
        if mode == "drift_only":
            await asyncio.to_thread(self.execute_drift_only)
        elif mode == "drift_top10":
            await asyncio.to_thread(self.execute_drift_top10)
        elif mode == "golden_setup":
            await asyncio.to_thread(self.execute_golden_setup)
        elif mode == "super_trend":
            await asyncio.to_thread(self.execute_super_trend)
        elif mode == "ST_cross_global":
            await asyncio.to_thread(self.execute_st_cross_global)
        elif mode == "ST_cross1":
            await asyncio.to_thread(self.execute_st_cross1)
        elif mode == "ST_cross2":
            await asyncio.to_thread(self.execute_st_cross2)
        elif mode == "ST_cross2_drift":
            await asyncio.to_thread(self.execute_st_cross2_drift)
        elif mode == "model_only":
            await asyncio.to_thread(self.bot.process_symbol_model_only_sync, self.bot.selected_symbols)
        else:
            logger.info(f"[TradingLogic] Режим {mode} не реализован.")

    def toggle_sleep_mode(self):
        self.bot.IS_SLEEPING_MODE = not self.bot.IS_SLEEPING_MODE
        status = "включен" if self.bot.IS_SLEEPING_MODE else "выключен"
        logger.info(f"[TradingLogic] Спящий режим для пользователя {self.bot.user_id}: {status}")
        return status

    def toggle_quiet_period(self):
        self.bot.QUIET_PERIOD_ENABLED = not self.bot.QUIET_PERIOD_ENABLED
        status = "включён" if self.bot.QUIET_PERIOD_ENABLED else "выключен"
        logger.info(f"[TradingLogic] Тихий режим для пользователя {self.bot.user_id}: {status}")
        return status

    def is_sleeping(self):
        return self.bot.IS_SLEEPING_MODE

    async def execute_drift_only(self):
        """
        Режим drift_only: для каждого символа выполняется drift-анализ.
        При обнаружении аномалии позиция открывается в противоположном направлении:
         - Если drift направлен "вверх" → открываем позицию Sell,
         - Иначе → открываем позицию Buy.
        """
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
        """
        Режим drift_top10: собираются drift-сигналы из истории,
        сортируются по средней силе и выбираются топ-10.
        Для каждого из топ-сигналов открывается позиция согласно направлению.
        """
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
        """
        Режим golden_setup: для каждого символа анализируются исторические данные за ограниченный период.
        Если условия золотой установки выполняются, возвращается действие (Buy или Sell)
        и открывается соответствующая позиция.
        """
        symbols = await self.bot.get_selected_symbols()
        for sym in symbols:
            df = await self.bot.get_historical_data_for_trading(sym, interval="1", limit=20)
            if df.empty:
                continue
            action, price_change = await self.handle_golden_setup(sym, df)
            if action:
                logger.info(f"[Golden Setup] {sym}: action={action}, price_change={price_change:.2f}.")
                await self.bot.open_position(sym, action, self.bot.POSITION_VOLUME, reason="Golden_setup")

    async def handle_golden_setup(self, symbol, df):
        """
        Анализ данных для золотой установки.
        Возвращает (action, price_change), если условия выполнены, иначе (None, None).
        """
        try:
            # Получаем текущие значения из последней свечи
            current_oi = Decimal(str(df.iloc[-1]["open_interest"]))
            current_vol = Decimal(str(df.iloc[-1]["volume"]))
            current_price = Decimal(str(df.iloc[-1]["closePrice"]))
            
            # Работа с историей — предполагается, что history_lock,
            # open_interest_history и volume_history определены глобально
            with history_lock:
                # Если истории для данного символа ещё нет — создаём
                if symbol not in open_interest_history:
                    open_interest_history[symbol] = []
                if symbol not in volume_history:
                    volume_history[symbol] = []
                
                # Добавляем текущие значения в историю
                open_interest_history[symbol].append(current_oi)
                volume_history[symbol].append(current_vol)
                
                sp_iters = int(self.golden_params["Sell"]["period_iters"])
                lp_iters = int(self.golden_params["Buy"]["period_iters"])
                period = max(sp_iters, lp_iters)
                
                # Если истории недостаточно, возвращаем отсутствие сигнала
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
                
                # Расчитываем изменения
                price_change = ((current_price - price_prev) / price_prev) * 100
                volume_change = ((current_vol - vol_prev) / vol_prev) * 100 if vol_prev != 0 else Decimal("0")
                oi_change = ((current_oi - oi_prev) / oi_prev) * 100 if oi_prev != 0 else Decimal("0")
                
                logger.info(
                    f"[GOLDEN_SETUP] {symbol}: p_ch={price_change:.2f}, vol_ch={volume_change:.2f}, oi_ch={oi_change:.2f}"
                )
                
                action = None
                if (price_change <= -self.golden_params["Sell"]["price_change"] and
                    volume_change >= self.golden_params["Sell"]["volume_change"] and
                    oi_change >= self.golden_params["Sell"]["oi_change"]):
                    action = "Sell"
                elif (price_change >= self.golden_params["Buy"]["price_change"] and
                    volume_change >= self.golden_params["Buy"]["volume_change"] and
                    oi_change >= self.golden_params["Buy"]["oi_change"]):
                    action = "Buy"
                else:
                    return None, None

            return (action, float(price_change))
        except Exception as e:
            logger.exception(f"Ошибка handle_golden_setup({symbol}): {e}")
            return None, None
    
    async def execute_super_trend(self):
        """
        Режим super_trend: для каждого символа рассчитывается SuperTrend (с параметрами length=3, multiplier=1.0).
        Если условия по свечам удовлетворены, открывается соответствующая позиция.
        """
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            df = await self.bot.get_historical_data_for_trading(sym, interval=self.bot.SUPER_TREND_TIMEFRAME, limit=200)
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
                await self.bot.open_position(sym, "Buy", self.bot.POSITION_VOLUME, reason=f"SuperTrend_{self.bot.SUPER_TREND_TIMEFRAME}")
            elif is_sell:
                logger.info(f"[SuperTrend] {sym}: сигнал SELL.")
                await self.bot.open_position(sym, "Sell", self.bot.POSITION_VOLUME, reason=f"SuperTrend_{self.bot.SUPER_TREND_TIMEFRAME}")
            else:
                logger.info(f"[SuperTrend] {sym}: условия не выполнены.")

    async def execute_st_cross_global(self):
        """
        Режим ST_cross_global: анализ пересечения fast и slow SuperTrend без дополнительных ограничений.
        Рассчитываются два варианта SuperTrend:
         - fast: calculate_supertrend_bybit_8_1 с length=3, multiplier=1.0
         - slow: calculate_supertrend_bybit_34_2 с length=8, multiplier=3.0
        Если пересечение подтверждено, открывается позиция.
        """
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
        """
        Режим ST_cross1: открытие позиций по сигналу свечей с дополнительным ограничением
        по процентному различию между fast и slow SuperTrend.
        """
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

    async def execute_st_cross2(self):
        """
        Режим ST_cross2: открытие позиций по изменению процентного разрыва между fast и slow SuperTrend.
        Если в предыдущей свече разрыв был ≤ -0.3% и в текущей ≥ +0.3% → сигнал LONG,
        если наоборот → сигнал SHORT.
        """
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            df = await self.bot.get_historical_data_for_trading(sym, interval=self.bot.INTERVAL, limit=200)
            if df.empty or len(df) < 5:
                logger.info(f"[ST_cross2] {sym}: недостаточно данных.")
                continue
            df_fast = await self.bot.calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
            df_slow = await self.bot.calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
            if df_fast.empty or df_slow.empty:
                logger.info(f"[ST_cross2] {sym}: не удалось рассчитать SuperTrend.")
                continue
            try:
                last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
                if last_candle_time < pd.Timestamp.utcnow() - pd.Timedelta(minutes=5):
                    logger.warning(f"[ST_cross2] {sym}: данные устарели.")
                    continue
            except Exception as e:
                logger.error(f"[ST_cross2] Ошибка проверки времени для {sym}: {e}")
                continue
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
            long_signal = (prev_diff_pct >= Decimal("-0.3") and curr_diff_pct <= Decimal("0.3"))
            short_signal = (prev_diff_pct <= Decimal("0.3") and curr_diff_pct >= Decimal("-0.3"))
            if long_signal:
                if curr_diff_pct > Decimal("1"):
                    logger.info(f"[ST_cross2] {sym}: положительное различие слишком велико, пропуск LONG.")
                    continue
                logger.info(f"[ST_cross2] {sym}: сигнал LONG (prev: {prev_diff_pct:.2f}%, curr: {curr_diff_pct:.2f}%).")
                await self.bot.open_position(sym, "Buy", self.bot.POSITION_VOLUME, reason="ST_cross2")
            elif short_signal:
                if curr_diff_pct < Decimal("-1"):
                    logger.info(f"[ST_cross2] {sym}: отрицательное различие слишком велико, пропуск SHORT.")
                    continue
                logger.info(f"[ST_cross2] {sym}: сигнал SHORT (prev: {prev_diff_pct:.2f}%, curr: {curr_diff_pct:.2f}%).")
                await self.bot.open_position(sym, "Sell", self.bot.POSITION_VOLUME, reason="ST_cross2")
            else:
                logger.info(f"[ST_cross2] {sym}: условия не выполнены.")

    async def execute_st_cross2_drift(self):
        """
        Режим ST_cross2_drift: сочетает логику ST_cross2 и дополнительно drift-логику.
        Если drift-сигнал отсутствует в основной логике ST_cross2, дополнительно выбирается ТОП-1 drift-сигнал
        из истории, и по нему открывается позиция.
        """
        symbols = self.bot.get_selected_symbols()
        for sym in symbols:
            self.execute_st_cross2()  # Запускаем стандартную логику ST_cross2 для символа
        # Дополнительная drift-логика:
        if not self.bot.drift_trade_executed:
            drift_signals = []
            for drift_sym, records in self.bot.drift_history.items():
                if records:
                    avg_strength = sum(r[1] for r in records) / len(records)
                    last_direction = records[-1][2]
                    drift_signals.append((drift_sym, avg_strength, last_direction))
            if drift_signals:
                drift_signals.sort(key=lambda x: x[1], reverse=True)
                top_signal = drift_signals[0]
                drift_sym, strength, direction = top_signal
                with self.bot.open_positions_lock:
                    if drift_sym in self.bot.open_positions:
                        logger.info(f"[ST_cross2_drift] Drift: позиция для {drift_sym} уже открыта, пропуск drift trade.")
                    else:
                        drift_side = "Sell" if direction == "вверх" else "Buy"
                        logger.info(f"[ST_cross2_drift] Drift: {drift_sym}: сигнал {drift_side}, strength={strength:.2f}.")
                        await self.bot.open_position(drift_sym, drift_side, Decimal("500"), reason="ST_cross2_drift_drift")
                        self.bot.drift_trade_executed = True
            else:
                logger.info("[ST_cross2_drift] Нет drift-сигналов для обработки.")

    async def execute_trading_mode(self):
        mode = self.bot.OPERATION_MODE
        logger.info(f"[TradingLogic] Пользователь {self.bot.user_id}: режим {mode}")
        if mode == "drift_only":
            await self.execute_drift_only()         # Если этот метод тоже асинхронный
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
        elif mode == "ST_cross2_drift":
            await self.execute_st_cross2_drift()
        elif mode == "model_only":
            # Если process_symbol_model_only_sync остаётся синхронной, можно запустить её в отдельном потоке:
            await self.bot.process_symbol_model_only_async
        else:
            logger.info(f"[TradingLogic] Режим {mode} не реализован.")
            
########################################################################
# TelegramBot: Интеграция с Telegram с использованием ReplyKeyboardMarkup
# Главное меню разделено на три группы: "Трейдинг", "Бот", "Информация".
# Добавлена клавиатура для смены торговых режимов.
########################################################################
class VolumeStates(StatesGroup):
    waiting_for_max_volume = State()
    waiting_for_position_volume = State()

# Главное меню – три раздела
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
        logging.warning("Получена команда /stop_admin от администратора, останавливаем бота.")

        # 1) Останавливаем диспетчер (чтобы не шли новые сообщения)
        await dp.stop()

        # 2) Ждём корректного завершения
        await dp.shutdown()

        # 3) Если хотите гарантированно «убить» все процессы Python:
        # import sys
        # sys.exit(0)

        # Или os._exit(0), но это «жёстко»:
        # os._exit(0)

    else:
        await message.reply("У вас нет прав для использования этой команды.")

@router.message(lambda msg: msg.text == "📉 Установить макс. объем")
async def set_max_volume_step1(message: Message, state: FSMContext):
    user_id = message.from_user.id
    # Проверяем, зарегистрирован ли
    if user_id not in user_bots:
        await message.answer("Вы не зарегистрированы. Воспользуйтесь /register")
        return

    await message.answer("Введите новый максимальный объём (USDT), например: 2000")
    # Переходим в состояние ожидания числа
    await state.set_state(VolumeStates.waiting_for_max_volume)


@router.message(VolumeStates.waiting_for_max_volume)
async def set_max_volume_step2(message: Message, state: FSMContext):
    user_id = message.from_user.id
    new_value_str = message.text.strip()

    # Проверим, что это число (можем использовать Decimal или float)
    try:
        val_dec = Decimal(new_value_str)
        if val_dec <= 0:
            raise ValueError("Значение должно быть > 0")
    except Exception as e:
        await message.answer(f"Некорректное число: {e}\nПопробуйте ещё раз или /cancel")
        return

    # Сохраняем в TradingBot
    bot_instance = user_bots[user_id]
    bot_instance.MAX_TOTAL_VOLUME = val_dec

    # Пишем и в глобальном словаре users, чтобы потом сделать save_users
    users[user_id]["max_total_volume"] = str(val_dec)

    # Сохраняем на диск
    save_users(users)

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
    save_users(users)

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

    # 1) Проверяем, не зарегистрирован ли уже
    if user_id in users:
        await message.answer("Вы уже зарегистрированы в системе.")
        return

    # 2) Иначе предлагаем ввести API-ключ
    await message.answer(
        "Введите, пожалуйста, ваш API Key.\n"
        "Внимание: Ключи НЕ должны содержать права перевода средств!"
    )
    # 3) Переходим в состояние ожидания ввода API Key
    await state.set_state(RegisterStates.waiting_for_api_key)

@router.message(RegisterStates.waiting_for_api_key)
async def process_api_key(message: Message, state: FSMContext):
    user_id = message.from_user.id
    api_key = message.text.strip()

    # Сохраняем во временном хранилище (FSM), чтобы не потерять
    await state.update_data(api_key=api_key)

    # Переходим к запросу API Secret
    await message.answer("Принято! Теперь отправьте ваш API Secret.")
    await state.set_state(RegisterStates.waiting_for_api_secret)

@router.message(RegisterStates.waiting_for_api_secret)
async def process_api_secret(message: Message, state: FSMContext):
    user_id = message.from_user.id
    api_secret = message.text.strip()

    # Сохраняем во FSM, но ещё не пишем в CSV
    await state.update_data(api_secret=api_secret)

    # Спрашиваем про режим
    await message.answer(
        "Принято!\n"
        "Теперь выберите режим торговли: напишите 'demo' (для тестовой сети) или 'real' (для реальной биржи)."
    )
    await state.set_state(RegisterStates.waiting_for_mode)

@router.message(RegisterStates.waiting_for_mode)
async def process_mode(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_mode = message.text.strip().lower()

    # Допускаем только 'demo' или 'real'
    if user_mode not in ("demo", "real"):
        await message.answer("Пожалуйста, введите только 'demo' или 'real'.")
        return

    # Достаём из FSM ранее сохранённые api_key и api_secret
    data = await state.get_data()
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")

    # Теперь пишем в CSV
    # (Важно: файл должен иметь колонки: user_id, user_api, user_api_secret, mode)
    file_exists = os.path.isfile("users.csv")
    with open("users.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user_id", "user_api", "user_api_secret", "mode"])
        writer.writerow([user_id, api_key, api_secret, user_mode])

    # Обновляем глобальный словарь users (user_id -> (api, secret, mode))
    users[user_id] = (api_key, api_secret, user_mode)

    # Создаём новый TradingBot
    bot_instance = TradingBot(user_id, api_key, api_secret, user_mode)
    user_bots[user_id] = bot_instance

    # Запускаем основной торговый цикл «на лету»
    asyncio.create_task(bot_instance.main_loop())

    await message.answer(
        f"Регистрация завершена!\n"
        f"Вы выбрали режим: {user_mode}.\n"
        "ПОМНИТЕ: Ключи не должны содержать права перевода средств!\n"
        "Теперь вы можете использовать /start."
    )
    # Сбрасываем состояние
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

# @router.message(lambda message: message.text == "📊 Статус")
# async def trading_status(message: Message):
#     user_id = message.from_user.id
#     if user_id not in user_bots:
#         await message.reply("Пользователь не зарегистрирован.")
#         return
#     bot_instance = user_bots[user_id]
#     with bot_instance.open_positions_lock:
#         if not bot_instance.open_positions:
#             await message.reply("Нет открытых позиций.")
#             return
#         lines = []
#         for sym, pos in bot_instance.open_positions.items():
#             try:
#                 side = pos["side"]
#                 entry_price = Decimal(str(pos["avg_price"]))
#                 volume = Decimal(str(pos["position_volume"]))
#                 current_price = await bot_instance.get_last_close_price(sym)
#                 if current_price is None:
#                     lines.append(f"{sym}: Нет цены")
#                     continue
#                 cp = Decimal(str(current_price))
#                 pnl = (cp - entry_price) / entry_price * volume if side.lower() == "buy" else (entry_price - cp) / entry_price * volume
#                 lines.append(f"{sym} {side}: PnL = {pnl:.2f} USDT")
#             except Exception as e:
#                 lines.append(f"{sym}: ошибка")
#         await message.reply("\n".join(lines))

@router.message(lambda message: message.text == "📊 Статус")
async def status_cmd(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.reply("Пользователь не зарегистрирован.")
        return

    bot_instance = user_bots[user_id]
    with bot_instance.open_positions_lock:
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
            # Ожидаем асинхронный вызов для получения текущей цены
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

            lines.append(
                f"{sym} {side_str}: "
                f"PNL = {pnl_usdt:.2f} USDT "
                f"({pnl_percent:.2f}%)"
            )

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

@router.message(lambda message: message.text in ["🛑 Тихий режим ON/OFF", "🔕 Статус тихого режима",
                                                  "😴 Усыпить бота", "🌞 Разбудить бота"])
async def bot_commands(message: Message):
    user_id = message.from_user.id
    if user_id not in user_bots:
        await message.reply("Пользователь не зарегистрирован.")
        return
    trading_logic = TradingLogic(user_bots[user_id])
    if message.text == "🛑 Тихий режим ON/OFF":
        status = trading_logic.toggle_quiet_period()
        await message.reply(f"Тихий режим: {status}")
    elif message.text == "🔕 Статус тихого режима":
        status = "включён" if user_bots[user_id].QUIET_PERIOD_ENABLED else "выключен"
        await message.reply(f"Тихий режим: {status}")
    elif message.text == "😴 Усыпить бота":
        status = trading_logic.toggle_sleep_mode()
        await message.reply(f"Спящий режим: {status}")
    elif message.text == "🌞 Разбудить бота":
        status = trading_logic.toggle_sleep_mode()
        await message.reply(f"Спящий режим: {status}")

@router.message(lambda message: message.text == "🔍 Получить данные по паре")
async def get_pair_info(message: Message):
    await message.reply("Введите символ пары (например, BTCUSDT):")

async def check_user_registration(user_id: int, message: Message):
    if user_id not in user_bots:
        await message.answer("❌ Вы не зарегистрированы!\n"
                            "Доступные команды:\n"
                            "/register - Регистрация\n"
                            "/help - Помощь")
        return False
    return True

@router.message(Command("start"))
async def start_cmd(message: Message):
    user_id = message.from_user.id
    if not await check_user_registration(user_id, message):
        return

# @router.message(Command("start"))
# async def start_cmd(message: Message):
#     user_id = message.from_user.id
#     if user_id not in user_bots:
#         await message.reply("Пользователь не зарегистрирован. Пожалуйста, пройдите регистрацию.")
#         return
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
#    await message.continue_propagation()

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

# Функции для установки режимов и изменения других параметров можно добавить аналогично


########################################################################
# Финальная часть: запуск Telegram polling и торговых циклов для каждого пользователя
########################################################################

async def monitor_positions_http():
    while True:
        try:
            # Обходим всех пользователей (ботов)
            for user_id, bot in user_bots.items():
                # Если бот в спящем режиме, пропускаем его
                if bot.IS_SLEEPING_MODE:
                    continue

                # Получаем актуальные позиции через HTTP в отдельном потоке
                positions = await asyncio.to_thread(bot.get_exchange_positions)
                # Обновляем локальное состояние позиций
                await bot.update_open_positions_from_exch_positions(positions)
                
                # Проходим по позициям и рассчитываем PnL
                for symbol, pos in positions.items():
                    side = pos.get("side")
                    entry_price = Decimal(str(pos.get("avg_price", 0)))
                    current_price = await bot.get_last_close_price(symbol)
                    if current_price is None:
                        logger.debug(f"[HTTP Monitor] Нет текущей цены для {symbol}")
                        continue
                    current_price = Decimal(str(current_price))
                    # Расчёт PnL для позиции
                    if side.lower() == "buy":
                        ratio = (current_price - entry_price) / entry_price
                    else:
                        ratio = (entry_price - current_price) / entry_price
                    profit_perc = (ratio * bot.PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
                    logger.info(f"[HTTP Monitor] User {user_id} {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%")
                    
                    # Если убыток превышает порог TARGET_LOSS_FOR_AVERAGING, открываем усредняющую позицию
                    if profit_perc <= -bot.TARGET_LOSS_FOR_AVERAGING:
                        logger.info(f"[HTTP Monitor] {symbol} (User {user_id}) достиг порога убытка ({profit_perc}% <= -{bot.TARGET_LOSS_FOR_AVERAGING}). Открываю усредняющую позицию.")
                        await bot.open_averaging_position_all(symbol)
                    
                    default_leverage = Decimal("10")
                    leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))
                    threshold_trailing = Decimal("5.0")

                    # Дополнительно: проверка условий для установки трейлинг-стопа
                    if bot.CUSTOM_TRAILING_STOP_ENABLED:
                        await bot.apply_custom_trailing_stop(symbol, pos, leveraged_pnl_percent, side)
                    
                    else:
                        if leveraged_pnl_percent >= threshold_trailing and not pos.get("trailing_stop_set", False):
                            logger.info(
                                f"[HTTP Monitor] {symbol}: Достигнут уровень для трейлинг-стопа "
                                f"(leveraged PnL = {leveraged_pnl_percent}%). Устанавливаю трейлинг-стоп."
                            )
                            await bot.set_trailing_stop(symbol, pos["size"], bot.TRAILING_GAP_PERCENT, side)
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Ошибка в monitor_positions_http: {e}")
            await asyncio.sleep(10)

# async def handle_position_update(message):
#     """
#     Асинхронный обработчик входящих сообщений от WebSocket.
#     Для каждой полученной свечи проверяет:
#       – Если PnL <= -TARGET_LOSS_FOR_AVERAGING, открывает усредняющую позицию.
#       – Если с учётом плеча (leveraged ROI) PnL >= threshold (например, 5%),
#         и трейлинг-стоп ещё не установлен, вызывает установку трейлинг‑стопа.
#     Обрабатывает данные для всех пользователей (user_bots).
#     """
#     logger.info(f"[WS] Получено сообщение: {message}")
#     if "data" in message and isinstance(message["data"], list):
#         for candle in message["data"]:
#             symbol = candle.get("symbol")
#             close_str = candle.get("close")
#             if not symbol or not close_str:
#                 continue
#             try:
#                 current_price = Decimal(close_str)
#             except Exception as e:
#                 logger.error(f"[WS] Ошибка преобразования цены для {symbol}: {e}")
#                 continue

#             # Обходим всех пользователей
#             for user_id, bot in user_bots.items():
#                 # Если бот в спящем режиме — пропускаем
#                 if bot.IS_SLEEPING_MODE:
#                     continue

#                 # Используем асинхронную блокировку, если она определена (иначе можно использовать обычный with)
#                 async with bot.open_positions_lock:
#                     if symbol not in bot.open_positions:
#                         continue
#                     pos = bot.open_positions[symbol]

#                 side = pos.get("side")
#                 entry_price = Decimal(str(pos.get("avg_price")))
#                 ratio = ((current_price - entry_price) / entry_price) if side.lower() == "buy" else ((entry_price - current_price) / entry_price)
#                 profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
#                 logger.info(f"[WS] {symbol} (User {user_id}): current={current_price}, entry={entry_price}, PnL={profit_perc}%")

#                 # Если убыток достигает порога для усреднения, открываем усредняющую позицию
#                 if profit_perc <= -bot.self.TARGET_LOSS_FOR_AVERAGING:
#                     logger.info(
#                         f"[WS] {symbol} (User {user_id}) достиг порога убытка ({profit_perc}% <= -{bot.self.TARGET_LOSS_FOR_AVERAGING}). "
#                         "Открываю усредняющую позицию."
#                     )
#                     # Предполагается, что у класса TradingBot реализована асинхронная функция open_averaging_position
#                     await bot.open_averaging_position_all(symbol)

#                 # Дополнительно: проверка условий для установки трейлинг-стопа.
#                 default_leverage = Decimal("10")
#                 leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))
#                 threshold_trailing = Decimal("5.0")
#                 if leveraged_pnl_percent >= threshold_trailing:
#                     if not pos.get("trailing_stop_set", False):
#                         logger.info(
#                             f"[WS] {symbol} (User {user_id}): Достигнут уровень для трейлинг-стопа "
#                             f"(leveraged_pnl_percent = {leveraged_pnl_percent}%). Устанавливаю трейлинг-стоп."
#                         )
#                         await bot.set_trailing_stop(symbol, pos["size"], TRAILING_GAP_PERCENT, side)
#     else:
#         logger.debug(f"[WS] Получено сообщение: {message}")

async def main_coroutine():
    try:
        init_user_bots()
        if not user_bots:
            logger.error("No users loaded! Check users.csv file")
            return

        setup_telegram_bot()
        if not telegram_bot:
            logger.error("Telegram bot not initialized!")
            return

        # Запуск Telegram polling в отдельной задаче
        telegram_task = asyncio.create_task(start_telegram_polling())
        # Запуск торговых циклов для каждого пользователя
        trading_tasks = [asyncio.create_task(bot.main_loop()) for bot in user_bots.values()]

        monitor_http_task = asyncio.create_task(monitor_positions_http())

        # Собираем задачи, чтобы если одна упадёт, остальные продолжали работу
        results = await asyncio.gather(telegram_task, monitor_http_task, *trading_tasks, return_exceptions=True)
        
        # Обработка исключений (например, логирование)
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
    main()