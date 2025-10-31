#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Бот для торговли на Bybit с использованием модели, дрейфа, супер-тренда и др.
Версия: полностью асинхронная реализация с использованием aiohttp для REST‑запросов,
         асинхронной обработкой (aiogram, asyncio) и защитой критических секций.
================================================================================
"""

import asyncio
import aiohttp
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
import datetime
from decimal import Decimal
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from pybit.unified_trading import HTTP, WebSocket
from pybit.exceptions import InvalidRequestError


from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.exceptions import TelegramRetryAfter, TelegramBadRequest, TelegramNetworkError

from dotenv import load_dotenv

import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from scipy.stats import ks_2samp
from tabulate import tabulate

from rich.console import Console
from rich.table import Table
from rich import box

import certifi
import ssl

# Загружаем переменные окружения
load_dotenv("keys_TESTNET.env")  # ожидаются BYBIT_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID и т.д.

# ===================== Логгер и глобальные переменные =====================
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[
        RotatingFileHandler("GoldenML_Dima.log", maxBytes=5*1024*1024, backupCount=2),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Глобальные константы и настройки
MAX_TOTAL_VOLUME = Decimal("500")      # USDT
POSITION_VOLUME = Decimal("100")         # USDT
PROFIT_COEFFICIENT = Decimal("100")
SUPER_TREND_TIMEFRAME = "1"
TRAILING_STOP_ENABLED = True
TRAILING_GAP_PERCENT = Decimal("0.008")
MIN_TRAILING_STOP = Decimal("0.0000001")
MODEL_FILENAME = "trading_model_final.pkl"
MIN_SAMPLES_FOR_TRAINING = 1000

# DRIFT параметры
VOLATILITY_THRESHOLD = 0.05
VOLUME_THRESHOLD = 2000000
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

# Определяем недостающие глобальные переменные
HEDGE_MODE = True
OPERATION_MODE = "ST_cross2"  # Возможные режимы: drift_only, drift_top10, golden_setup, model_only, super_trend, ST_cross1, ST_cross2, ST_cross_global, ST_cross2_drift
check_and_close_active = True

# Блокировки для синхронных операций
import threading
open_positions_lock = threading.Lock()
state_lock = threading.Lock()

# Переменные для активных символов и состояния
selected_symbols = []
last_asset_selection_time = 0
ASSET_SELECTION_INTERVAL = 3600

# Остальные глобальные переменные
REAL_TRADES_FEATURES_CSV = "real_trades_features.csv"
MODEL_FEATURE_COLS = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
INTERVAL = "1"
MAX_AVERAGING_VOLUME = MAX_TOTAL_VOLUME * Decimal("2")
averaging_total_volume = Decimal("0")
averaging_positions = {}
TARGET_LOSS_FOR_AVERAGING = Decimal("16.0")
MONITOR_MODE = "http"  # "ws" или "http"
IS_RUNNING = True
drift_running = True

# Для хранения открытых позиций и прочего
state = {}
open_positions = {}
drift_history = defaultdict(list)
open_interest_history = defaultdict(list)
volume_history = defaultdict(list)
drift_state = {"last_analysis": {}}
drift_lock = asyncio.Lock()

# API ключи и Telegram
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("BYBIT_API_KEY / BYBIT_API_SECRET не заданы в .env!")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ===================== Асинхронный HTTP-клиент для Bybit =====================
class AsyncBybitClient:
    def __init____init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        demo: bool = None,
        timeout = 60
    ):
        self.instance = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            demo=True,
            log_requests=True)
        self.timeout = timeout
        self.base_url = "https://api-demo.bybit.com"
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
    async def close(self):
        await self.session.close()
    async def get(self, path, params=None):
        url = f"{self.base_url}{path}"
        async with self.session.get(url, params=params) as response:
            return await response.json()
    async def post(self, path, json_payload=None):
        url = f"{self.base_url}{path}"
        async with self.session.post(url, json=json_payload) as response:
            return await response.json()
    async def get_kline(self, **params):
        return await self.get("/v5/market/kline", params=params)
    async def get_instruments_info(self, **params):
        return await self.get("/v5/market/instruments-info", params=params)
    async def get_tickers(self, **params):
        return await self.get("/v5/market/tickers", params=params)
    async def get_positions(self, **params):
        return await self.get("/v5/position/list", params=params)
    async def place_order(self, **params):
        return await self.post("/v5/order/create", json_payload=params)
    async def set_trading_stop(self, **params):
        return await self.post("/v5/position/trading-stop", json_payload=params)

#session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, timeout=60)

#async_client = AsyncBybitClient(os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET"), testnet=False, timeout=60)

# ===================== Остальные глобальные объекты =====================
telegram_bot = None
router = Router()
telegram_message_queue = None
send_semaphore = asyncio.Semaphore(10)
MAX_CONCURRENT_THREADS = 5
thread_semaphore = ThreadPoolExecutor(MAX_CONCURRENT_THREADS)
publish_drift_table = True
publish_model_table = True

# ===================== Определения недостающих функций =====================

def adjust_quantity(symbol: str, raw_qty: float) -> float:
    # Примерная реализация; замените по необходимости
    try:
        # Здесь предположим, что информация по инструменту хранится в словаре (stub)
        info = {"lotSizeFilter": {"minOrderQty": "0.001", "qtyStep": "0.001", "maxOrderQty": "1000"},
                "minOrderValue": 5}
        min_qty = Decimal(info["lotSizeFilter"]["minOrderQty"])
        qty_step = Decimal(info["lotSizeFilter"]["qtyStep"])
        max_qty = Decimal(info["lotSizeFilter"]["maxOrderQty"])
        min_order_value = Decimal(info["minOrderValue"])
        # Для цены используем get_last_close_price (вызываем синхронно через asyncio.run)
        last_price = asyncio.run(get_last_close_price(symbol))
        if not last_price or last_price <= 0:
            return 0.0
        price_dec = Decimal(str(last_price))
        dec_qty = Decimal(str(raw_qty))
        adj_qty = (dec_qty // qty_step) * qty_step
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
    except Exception as e:
        logger.exception(f"[adjust_quantity] {symbol}: {e}")
        return 0.0

def open_position(symbol: str, side: str, volume_usdt: Decimal, reason: str):
    # Примерная синхронная реализация открытия позиции
    if IS_SLEEPING_MODE:
        logger.info(f"[open_position] Бот в спящем режиме, открытие {symbol} отменено.")
        return
    try:
        logger.info(f"[open_position] Открываю {side} позицию для {symbol}, объем: {volume_usdt} USDT, причина: {reason}")
        # Здесь реализуйте логику открытия ордера (например, вызов place_order)
        last_price = asyncio.run(get_last_close_price(symbol))
        if not last_price or last_price <= 0:
            logger.info(f"[open_position] Нет актуальной цены для {symbol}.")
            return
        qty = volume_usdt / Decimal(str(last_price))
        pos_idx = 1 if side.lower() == "buy" else 2
        trade_id = f"{symbol}_{int(time.time())}"
        # Логируем сделку (синхронно)
        log_model_features_for_trade(trade_id, symbol, side, {})
        order_res = asyncio.run(place_order(symbol, side, float(qty), order_type="Market", time_in_force="GoodTillCancel", reduce_only=False, positionIdx=pos_idx))
        if not order_res or order_res.get("retCode") != 0:
            logger.info(f"[open_position] Ошибка ордера для {symbol}.")
            return
        with open_positions_lock, state_lock:
            open_positions[symbol] = {
                "side": side,
                "size": float(qty),
                "avg_price": float(last_price),
                "position_volume": float(volume_usdt),
                "symbol": symbol,
                "trailing_stop_set": False,
                "trade_id": trade_id,
                "open_time": datetime.datetime.utcnow()
            }
            state["total_open_volume"] = state.get("total_open_volume", Decimal("0")) + volume_usdt
        row = get_last_row(symbol)
        log_trade(symbol, row, None, side, f"Opened ({reason})", closed_manually=False)
        logger.info(f"[open_position] {symbol}: позиция {side} успешно открыта.")
    except Exception as e:
        logger.exception(f"[open_position] Ошибка для {symbol}: {e}")

def get_last_close_price(symbol):
    # Примерная синхронная реализация, вызывающая асинхронную функцию через asyncio.run
    try:
        resp = asyncio.run(async_client.get_kline(category="linear", symbol=symbol, interval="1", limit=1))
        if not resp or resp.get("retCode") != 0:
            return None
        klines = resp["result"].get("list", [])
        if not klines:
            return None
        row = klines[0]
        if isinstance(row, list) and len(row) > 4:
            return float(row[4])
        elif isinstance(row, dict):
            return float(row.get("close"))
        return None
    except Exception as e:
        logger.exception(f"[get_last_close_price] {symbol}: {e}")
        return None

def update_open_positions_from_exch_positions(expos: dict):
    # Обновляем глобальный словарь open_positions на основе полученных данных
    with open_positions_lock, state_lock:
        # Удаляем позиции, которых уже нет
        to_remove = []
        for sym in list(open_positions.keys()):
            if sym not in expos:
                to_remove.append(sym)
        for sym in to_remove:
            del open_positions[sym]
        # Обновляем или добавляем новые позиции
        for sym, newpos in expos.items():
            if sym in open_positions:
                open_positions[sym].update(newpos)
            else:
                open_positions[sym] = newpos
        total = sum(Decimal(str(p.get("position_volume", 0))) for p in open_positions.values())
        state["total_open_volume"] = total
        logger.info(f"[update_open_positions_from_exch_positions] Итоговый объем: {total}")

def get_selected_symbols():
    # Возвращает глобальный список символов; если время обновления истекло, обновляем список через get_usdt_pairs
    global selected_symbols, last_asset_selection_time
    now = time.time()
    if not selected_symbols or now - last_asset_selection_time >= ASSET_SELECTION_INTERVAL:
        selected_symbols = asyncio.run(get_usdt_pairs())
        last_asset_selection_time = now
    return selected_symbols

def monitor_feature_drift_per_symbol(symbol, new_data, ref_data, feature_cols, drift_csv="feature_drift.csv", threshold=0.5):
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
            if c in new_data.columns and c in ref_data.columns:
                stat, _ = ks_2samp(new_data[c].values, ref_data[c].values)
                stats.append(stat)
        if not stats:
            return False, 0.0, "нет фич"
        anomaly_strength = float(np.mean(stats))
        is_anomaly = anomaly_strength > threshold
        ts_str = datetime.datetime.utcnow().isoformat()
        drift_history[symbol].append((ts_str, anomaly_strength, direction))
        if len(drift_history[symbol]) > 10:
            drift_history[symbol].pop(0)
        logger.info(f"[DRIFT] {symbol}: strength={anomaly_strength:.3f}, direction={direction}, anomaly={is_anomaly}")
        return is_anomaly, anomaly_strength, direction
    except Exception as e:
        logger.exception(f"[DRIFT] Ошибка в monitor_feature_drift_per_symbol для {symbol}: {e}")
        return False, 0.0, "ошибка"

def handle_drift_top10(top_list):
    if OPERATION_MODE not in ["drift_only", "drift_top10"]:
        logger.info(f"[DRIFT_TOP10] Режим {OPERATION_MODE} не поддерживает дрейф.")
        return
    logger.info("[DRIFT_TOP10] Обработка дрейф сигналов.")
    for (sym, strength, direction) in top_list:
        side = "Buy" if direction == "вверх" else "Sell"
        logger.info(f"[DRIFT_TOP10] {sym}: side={side}, strength={strength:.2f}")
        open_position(sym, side, POSITION_VOLUME, reason="Drift")

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
            ep = entry_price
            ratio = (cp - ep) / ep if side.lower() == "buy" else (ep - cp) / ep
            leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))
            with open_positions_lock:
                if sym in open_positions:
                    open_positions[sym]['profit_perc'] = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
            if leveraged_pnl_percent >= threshold_roi and not pos.get("trailing_stop_set", False):
                logger.info(f"[TrailingStop] {sym}: устанавливаю трейлинг-стоп, leveraged PnL = {leveraged_pnl_percent}%")
                asyncio.run(set_trailing_stop(sym, open_positions[sym]["size"], TRAILING_GAP_PERCENT, side))
    except Exception as e:
        logger.exception(f"Ошибка check_and_set_trailing_stop: {e}")

# ===================== Функции работы с данными и моделью =====================
def get_last_row(symbol):
    df = asyncio.run(get_historical_data_for_trading(symbol, interval="1", limit=1))
    if df.empty:
        return None
    return df.iloc[-1]

def log_trade(symbol, row, open_interest, action, result, closed_manually=False):
    try:
        filename = "trade_log.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["symbol", "timestamp", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest", "action", "result", "closed_manually"])
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
            writer.writerow([symbol, time_str, open_str, high_str, low_str, close_str, vol_str, oi_str, action, result, closed_manually])
        logger.info(f"Сделка: {symbol}, {action}, {result}")
    except Exception as e:
        logger.exception("Ошибка log_trade:", exc_info=e)

def log_model_features_for_trade(trade_id: str, symbol: str, side: str, features: dict):
    csv_filename = REAL_TRADES_FEATURES_CSV
    file_exists = os.path.isfile(csv_filename)
    row = {"trade_id": trade_id, "symbol": symbol, "side": side}
    row.update(features)
    try:
        with open(csv_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
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
    exp1 = close_prices.ewm(span=fast, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({'MACD_12_26_9': macd, 'MACDs_12_26_9': signal_line})

def calculate_rsi(close_prices, periods=13):
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
        logger.info(f"[MODEL] Предсказание для {symbol} записано.")
    except Exception as e:
        logger.exception(f"Ошибка log_model_prediction({symbol}): {e}")

def train_and_load_model(csv_path="historical_data_for_model_5m.csv"):
    try:
        if not os.path.isfile(csv_path):
            logger.warning(f"Нет файла {csv_path} => обучение невозможно.")
            return None
        df_all = pd.read_csv(csv_path)
        if df_all.empty:
            logger.warning(f"{csv_path} пуст.")
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
            logger.warning("Нет данных после target.")
            return None
        if len(data) < MIN_SAMPLES_FOR_TRAINING:
            logger.warning(f"Слишком мало строк: {len(data)} < {MIN_SAMPLES_FOR_TRAINING}.")
            return None
        feature_cols = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
        data = data.dropna(subset=feature_cols)
        if data.empty:
            logger.warning("Нет данных после очистки.")
            return None
        X = data[feature_cols].values
        y = data["target"].astype(int).values
        if len(X) < 50:
            logger.warning(f"Слишком мало данных: {len(X)}")
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
        logger.info(f"[train_and_load_model] CV max_accuracy={best_acc:.4f}")
        joblib.dump(pipeline, MODEL_FILENAME)
        logger.info(f"[train_and_load_model] Модель сохранена в {MODEL_FILENAME}")
        return pipeline
    except Exception as e:
        logger.exception(f"Ошибка train_and_load_model: {e}")
        return None

def load_model():
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except (ModuleNotFoundError, ImportError):
        logger.warning("Не удалось загрузить модель. Будет обучена новая.")
        return train_and_load_model()

def retrain_model_with_real_trades(historical_csv="historical_data_for_model_5m.csv", real_trades_csv=REAL_TRADES_FEATURES_CSV):
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
            logger.warning("Нет данных после target.")
            return None
        if len(data) < MIN_SAMPLES_FOR_TRAINING:
            logger.warning(f"Слишком мало строк: {len(data)} < {MIN_SAMPLES_FOR_TRAINING}.")
            return None
        feature_cols = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
        data = data.dropna(subset=feature_cols)
        if data.empty:
            logger.warning("Нет данных после очистки.")
            return None
        X = data[feature_cols].values
        y = data["target"].astype(int).values
        if len(X) < 50:
            logger.warning(f"Слишком мало данных: {len(X)}")
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
        logger.info(f"[retrain_model_with_real_trades] CV max_accuracy={best_acc:.4f}")
        joblib.dump(pipeline, MODEL_FILENAME)
        logger.info(f"[retrain_model_with_real_trades] Модель сохранена в {MODEL_FILENAME}")
        return pipeline
    except Exception as e:
        logger.exception("[retrain_model_with_real_trades] Ошибка:")
        return None

async def maybe_retrain_model():
    global current_model
    new_model = retrain_model_with_real_trades(historical_csv="historical_data_for_model_5m.csv", real_trades_csv=REAL_TRADES_FEATURES_CSV)
    if new_model:
        current_model = new_model
        logger.info("[maybe_retrain_model] Модель обновлена.")

# ===================== SUPER TREND функции =====================
def calculate_supertrend_bybit_34_2(df: pd.DataFrame, length=8, multiplier=3.0) -> pd.DataFrame:
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
        logger.exception(f"Ошибка calculate_supertrend_bybit_34_2: {e}")
        return pd.DataFrame()

def calculate_supertrend_bybit_8_1(df: pd.DataFrame, length=3, multiplier=1.0) -> pd.DataFrame:
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
        logger.exception(f"Ошибка calculate_supertrend_bybit_8_1: {e}")
        return pd.DataFrame()

def process_symbol_supertrend_open(symbol, interval="1", length=3, multiplier=1.0):
    df = asyncio.run(get_historical_data_for_trading(symbol, interval=interval, limit=200))
    if df.empty or len(df) < 3:
        logger.info(f"{symbol}: недостаточно данных для SuperTrend.")
        return
    st_df = calculate_supertrend_bybit_8_1(df.copy(), length=length, multiplier=multiplier)
    if st_df.empty or len(st_df) < 3:
        logger.info(f"{symbol}: недостаточно данных в st_df.")
        return
    i0 = len(st_df) - 1
    i1 = i0 - 1
    o1 = st_df["openPrice"].iloc[i1]
    c1 = st_df["closePrice"].iloc[i1]
    st1 = st_df["supertrend"].iloc[i1]
    o0 = st_df["openPrice"].iloc[i0]
    c0 = st_df["closePrice"].iloc[i0]
    st0 = st_df["supertrend"].iloc[i0]
    is_buy = (o1 < st1) and (c1 > st1) and (o0 > st0)
    is_sell = (o1 > st1) and (c1 < st1) and (o0 < st0)
    if is_buy:
        logger.info(f"[SuperTrend] {symbol}: сигнал BUY.")
        open_position(symbol, "Buy", POSITION_VOLUME, reason=f"SuperTrend_{interval}")
    elif is_sell:
        logger.info(f"[SuperTrend] {symbol}: сигнал SELL.")
        open_position(symbol, "Sell", POSITION_VOLUME, reason=f"SuperTrend_{interval}")
    else:
        logger.info(f"[SuperTrend] {symbol}: сигнал отсутствует.")

def process_symbol_st_cross_global(symbol, interval="1", limit=200):
    logger.info(f"[ST_cross_global] Обработка {symbol}")
    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross_global] {symbol}: позиция уже открыта.")
            return
    df = asyncio.run(get_historical_data_for_trading(symbol, interval=interval, limit=limit))
    if df.empty or len(df) < 5:
        logger.info(f"[ST_cross_global] {symbol}: недостаточно данных.")
        return
    df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
    df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
    if df_fast.empty or df_slow.empty:
        logger.info(f"[ST_cross_global] {symbol}: ошибка расчёта SuperTrend.")
        return
    try:
        last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
        if last_candle_time < pd.Timestamp.utcnow() - pd.Timedelta(minutes=5):
            logger.warning(f"[ST_cross_global] {symbol}: данные устарели.")
            return
    except Exception as e:
        logger.error(f"[ST_cross_global] {symbol}: ошибка времени: {e}")
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
    logger.info(f"[ST_cross_global] {symbol}: prev_fast={prev_fast:.6f}, curr_fast={curr_fast:.6f}, last_close={last_close:.6f}")
    if confirmed_buy:
        logger.info(f"[ST_cross_global] {symbol}: сигнал BUY подтверждён.")
        open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross_global")
    elif confirmed_sell:
        logger.info(f"[ST_cross_global] {symbol}: сигнал SELL подтверждён.")
        open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross_global")
    else:
        logger.info(f"[ST_cross_global] {symbol}: сигнал отсутствует.")

def process_symbol_st_cross1(symbol, interval="1", limit=200):
    logger.info(f"[ST_cross1] Обработка {symbol}")
    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross1] {symbol}: позиция уже открыта.")
            return
    df = asyncio.run(get_historical_data_for_trading(symbol, interval=interval, limit=limit))
    if df.empty or len(df) < 5:
        logger.info(f"[ST_cross1] {symbol}: недостаточно данных.")
        return
    df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
    df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
    if df_fast.empty or df_slow.empty:
        logger.info(f"[ST_cross1] {symbol}: ошибка расчёта SuperTrend.")
        return
    try:
        last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
        if last_candle_time < pd.Timestamp.utcnow() - pd.Timedelta(minutes=5):
            logger.warning(f"[ST_cross1] {symbol}: данные устарели.")
            return
    except Exception as e:
        logger.error(f"[ST_cross1] {symbol}: ошибка времени: {e}")
        return
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
        if curr_diff_pct > Decimal("1"):
            logger.info(f"[ST_cross1] {symbol}: положительное различие слишком велико.")
            return
        confirmed_buy = last_close >= curr_fast * (1 + margin)
        if confirmed_buy:
            logger.info(f"[ST_cross1] {symbol}: сигнал BUY подтверждён.")
            open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross1")
        else:
            logger.info(f"[ST_cross1] {symbol}: сигнал BUY не подтверждён.")
    elif first_cross_down:
        if curr_diff_pct < Decimal("-1"):
            logger.info(f"[ST_cross1] {symbol}: отрицательное различие слишком велико.")
            return
        confirmed_sell = last_close <= curr_fast * (1 - margin)
        if confirmed_sell:
            logger.info(f"[ST_cross1] {symbol}: сигнал SELL подтверждён.")
            open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross1")
        else:
            logger.info(f"[ST_cross1] {symbol}: сигнал SELL не подтверждён.")
    else:
        logger.info(f"[ST_cross1] {symbol}: сигнал отсутствует.")

def process_symbol_st_cross2(symbol, interval="1", limit=200):
    logger.info(f"[ST_cross2] Обработка {symbol}")
    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross2] {symbol}: позиция уже открыта.")
            return
    df = asyncio.run(get_historical_data_for_trading(symbol, interval=interval, limit=limit))
    if df.empty or len(df) < 5:
        logger.info(f"[ST_cross2] {symbol}: недостаточно данных.")
        return
    df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
    df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
    if df_fast.empty or df_slow.empty:
        logger.info(f"[ST_cross2] {symbol}: ошибка расчёта SuperTrend.")
        return
    try:
        last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
        if last_candle_time < pd.Timestamp.utcnow() - pd.Timedelta(minutes=5):
            logger.warning(f"[ST_cross2] {symbol}: данные устарели.")
            return
    except Exception as e:
        logger.error(f"[ST_cross2] {symbol}: ошибка времени: {e}")
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
    long_signal = (prev_diff_pct <= Decimal("-0.3") and curr_diff_pct >= Decimal("0.3"))
    short_signal = (prev_diff_pct >= Decimal("0.3") and curr_diff_pct <= Decimal("-0.3"))
    if long_signal:
        if curr_diff_pct > Decimal("1"):
            logger.info(f"[ST_cross2] {symbol}: положительное различие слишком велико.")
            return
        logger.info(f"[ST_cross2] {symbol}: сигнал LONG обнаружен.")
        open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross2")
    elif short_signal:
        if curr_diff_pct < Decimal("-1"):
            logger.info(f"[ST_cross2] {symbol}: отрицательное различие слишком велико.")
            return
        logger.info(f"[ST_cross2] {symbol}: сигнал SHORT обнаружен.")
        open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross2")
    else:
        logger.info(f"[ST_cross2] {symbol}: условия не выполнены.")

def process_symbol_st_cross2_drift(symbol, interval="1", limit=200):
    logger.info(f"[ST_cross2_drift] Обработка {symbol}")
    with open_positions_lock:
        if symbol in open_positions:
            logger.info(f"[ST_cross2_drift] {symbol}: позиция уже открыта, пропуск.")
            return
    df = asyncio.run(get_historical_data_for_trading(symbol, interval=interval, limit=limit))
    if not df.empty and len(df) >= 5:
        df_fast = calculate_supertrend_bybit_8_1(df.copy(), length=3, multiplier=1.0)
        df_slow = calculate_supertrend_bybit_34_2(df.copy(), length=8, multiplier=3.0)
        if not df_fast.empty and not df_slow.empty:
            try:
                last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
                if last_candle_time >= pd.Timestamp.utcnow() - pd.Timedelta(minutes=5):
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
                    long_signal = (prev_diff_pct <= Decimal("-0.3") and curr_diff_pct >= Decimal("0.3"))
                    short_signal = (prev_diff_pct >= Decimal("0.3") and curr_diff_pct <= Decimal("-0.3"))
                    if long_signal:
                        if curr_diff_pct > Decimal("1"):
                            logger.info(f"[ST_cross2_drift] {symbol}: положительное различие слишком велико.")
                        else:
                            logger.info(f"[ST_cross2_drift] {symbol}: сигнал LONG обнаружен.")
                            open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross2_drift")
                    elif short_signal:
                        if curr_diff_pct < Decimal("-1"):
                            logger.info(f"[ST_cross2_drift] {symbol}: отрицательное различие слишком велико.")
                        else:
                            logger.info(f"[ST_cross2_drift] {symbol}: сигнал SHORT обнаружен.")
                            open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross2_drift")
                    else:
                        logger.info(f"[ST_cross2_drift] {symbol}: условия не выполнены.")
            except Exception as e:
                logger.error(f"[ST_cross2_drift] {symbol}: ошибка обработки: {e}")
    global drift_trade_executed
    if not drift_trade_executed:
        drift_signals = []
        for drift_sym, recs in drift_history.items():
            if recs:
                avg_strength = sum(x[1] for x in recs) / len(recs)
                last_direction = recs[-1][2]
                drift_signals.append((drift_sym, avg_strength, last_direction))
        if drift_signals:
            drift_signals.sort(key=lambda x: x[1], reverse=True)
            top_drift = drift_signals[0]
            drift_sym, drift_avg_strength, drift_direction = top_drift
            with open_positions_lock:
                if drift_sym in open_positions:
                    logger.info(f"[ST_cross2_drift] Drift: позиция для {drift_sym} уже открыта.")
                else:
                    drift_side = "Sell" if drift_direction == "вверх" else "Buy"
                    logger.info(f"[ST_cross2_drift] Открываю drift позицию для {drift_sym}: {drift_side} на 500 USDT.")
                    open_position(drift_sym, drift_side, Decimal("500"), reason="ST_cross2_drift_drift")
                    drift_trade_executed = True
        else:
            logger.info("[ST_cross2_drift] Нет drift-сигналов.")

# Функция place_order (асинхронная)
async def place_order(symbol, side, qty, order_type="Market", time_in_force="GoodTillCancel", reduce_only=False, positionIdx=None):
    adj_qty = await asyncio.to_thread(adjust_quantity, symbol, qty)
    if adj_qty <= 0:
        logger.error(f"[place_order] Недопустимое количество для {symbol}.")
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
    logger.info(f"[place_order] Отправка ордера: {params}")
    resp = await async_client.place_order(**params)
    if resp.get("retCode") == 0:
        logger.info(f"[place_order] Ордер выполнен: {symbol}, side={side}, qty={adj_qty}")
        return resp
    else:
        logger.error(f"[place_order] Ошибка: retCode={resp.get('retCode')} => {resp.get('retMsg')}")
        return None

# Функция get_usdt_pairs (асинхронная)
async def get_usdt_pairs():
    resp = await async_client.get_tickers(category="linear")
    if "result" not in resp or "list" not in resp["result"]:
        logger.error("[get_usdt_pairs] Некорректный ответ get_tickers.")
        return []
    tickers_data = resp["result"]["list"]
    inst_resp = await async_client.get_instruments_info(category="linear")
    if "result" not in inst_resp or "list" not in inst_resp["result"]:
        logger.error("[get_usdt_pairs] Некорректный ответ get_instruments_info.")
        return []
    instruments_data = inst_resp["result"]["list"]
    trading_status = {inst.get("symbol"): (inst.get("status", "").upper() == "TRADING") for inst in instruments_data if inst.get("symbol")}
    usdt_pairs = []
    for tk in tickers_data:
        sym = tk.get("symbol")
        if sym and "USDT" in sym and "BTC" not in sym and "ETH" not in sym:
            if not trading_status.get(sym, False):
                continue
            turnover24 = Decimal(str(tk.get("turnover24h", "0")))
            volume24 = Decimal(str(tk.get("volume24h", "0")))
            if turnover24 >= Decimal("2000000") and volume24 >= Decimal("2000000"):
                usdt_pairs.append(sym)
    logger.info(f"[get_usdt_pairs] USDT-пары: {usdt_pairs}")
    return usdt_pairs

# Функция set_trailing_stop (асинхронная)
async def set_trailing_stop(symbol, size, trailing_gap_percent, side):
    pos_info = await get_position_info(symbol, side)
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
        logger.info(f"[set_trailing_stop] {symbol}: trailing_stop={trailing_distance_abs} < {dynamic_min}, пропуск.")
        return
    params = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "TrailingStop",
        "qty": str(size),
        "trailingStop": str(trailing_distance_abs),
        "timeInForce": "GoodTillCancel",
        "positionIdx": pos_idx
    }
    resp = await async_client.set_trading_stop(**params)
    rc = resp.get("retCode")
    if rc == 0:
        with open_positions_lock:
            if symbol in open_positions:
                open_positions[symbol]["trailing_stop_set"] = True
        row = await asyncio.to_thread(get_last_row, symbol)
        await asyncio.to_thread(log_trade, symbol, row, None, f"{trailing_distance_abs}", "Trailing Stop Set", False)
        logger.info(f"[set_trailing_stop] {symbol}: трейлинг-стоп установлен.")
    elif rc == 34040:
        logger.info(f"[set_trailing_stop] {symbol}: not modified, retCode=34040.")
    else:
        logger.error(f"[set_trailing_stop] {symbol}: Ошибка: {resp.get('retMsg')}")

# Функция get_historical_data_for_trading (асинхронная)
async def get_historical_data_for_trading(symbol, interval="1", limit=200, from_time=None):
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    if from_time:
        params["from"] = from_time
    resp = await async_client.get_kline(**params)
    if resp.get("retCode") != 0:
        logger.error(f"[get_historical_data_for_trading] {symbol}: {resp.get('retMsg')}")
        return pd.DataFrame()
    data = resp["result"].get("list", [])
    if not data:
        return pd.DataFrame()
    columns = ["open_time", "open", "high", "low", "close", "volume", "open_interest"]
    df = pd.DataFrame(data, columns=columns)
    df["startTime"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
    df.rename(columns={"open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"}, inplace=True)
    df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]] = df[["openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["closePrice"], inplace=True)
    df.sort_values("startTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.debug(f"[get_historical_data_for_trading] {symbol}: {len(df)} свечей.")
    return df[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "open_interest"]]

# Функция get_exchange_positions (асинхронная)
async def get_exchange_positions():
    params = {"category": "linear", "settleCoin": "USDT"}
    resp = await async_client.get_positions(**params)
    if resp.get("retCode") != 0:
        logger.error(f"[get_exchange_positions] Ошибка: {resp.get('retMsg')}")
        return {}
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
    logger.info(f"[get_exchange_positions] Получены: {exchange_positions}")
    return exchange_positions

# Функция get_top_anomalies_from_analysis (синхронная)
def get_top_anomalies_from_analysis(analysis_data, top_k=10):
    try:
        anomalies = []
        for symbol, data in analysis_data.items():
            if data.get('is_anomaly'):
                anomalies.append((symbol, data.get('strength', 0.0), data.get('direction', '')))
        anomalies.sort(key=lambda x: x[1], reverse=True)
        return anomalies[:top_k]
    except Exception as e:
        logger.exception(f"Ошибка в get_top_anomalies_from_analysis: {e}")
        return []

# Функция handle_golden_setup (синхронная)
def handle_golden_setup(symbol, df):
    try:
        current_oi = Decimal(str(df.iloc[-1]["open_interest"]))
        current_vol = Decimal(str(df.iloc[-1]["volume"]))
        current_price = Decimal(str(df.iloc[-1]["closePrice"]))
        with state_lock:
            open_interest_history[symbol].append(current_oi)
            volume_history[symbol].append(current_vol)
            sp_iters = int(golden_params["Sell"]["period_iters"])
            lp_iters = int(golden_params["Buy"]["period_iters"])
            period = max(sp_iters, lp_iters)
            if len(open_interest_history[symbol]) < period or len(volume_history[symbol]) < period:
                logger.info(f"{symbol}: Недостаточно истории для golden_setup.")
                return None, None
            if df.shape[0] < period:
                logger.info(f"{symbol}: Недостаточно свечей для golden_setup.")
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
            if (price_change <= -golden_params["Sell"]["price_change"] and volume_change >= golden_params["Sell"]["volume_change"] and oi_change >= golden_params["Sell"]["oi_change"]):
                action = "Sell"
            elif (price_change >= golden_params["Buy"]["price_change"] and volume_change >= golden_params["Buy"]["volume_change"] and oi_change >= golden_params["Buy"]["oi_change"]):
                action = "Buy"
            else:
                return None, None
        return (action, float(price_change))
    except Exception as e:
        logger.exception(f"Ошибка handle_golden_setup для {symbol}: {e}")
        return None, None

# Функция log_model_prediction_for_symbol (синхронная)
def log_model_prediction_for_symbol(symbol):
    global current_model
    if not current_model:
        current_model = load_model()
        if not current_model:
            logger.error("Модель не загружена!")
            return
    df = asyncio.run(get_historical_data_for_model(symbol, interval="1", limit=200))
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
        logger.exception(f"[MODEL] Ошибка для {symbol}: {e}")
        return
    log_model_prediction(symbol, pred[0], proba)

def open_averaging_position(symbol):
    try:
        with open_positions_lock:
            if symbol not in open_positions:
                logger.info(f"[Averaging] Нет базовой позиции для {symbol}.")
                return
            if symbol in averaging_positions:
                logger.info(f"[Averaging] Усредняющая позиция для {symbol} уже открыта.")
                return
            base_pos = open_positions[symbol]
            side = base_pos["side"]
            base_volume_usdt = Decimal(str(base_pos["position_volume"]))
            global averaging_total_volume
            if averaging_total_volume + base_volume_usdt > MAX_AVERAGING_VOLUME:
                logger.info(f"[Averaging] Превышен лимит усреднения: {averaging_total_volume} + {base_volume_usdt} > {MAX_AVERAGING_VOLUME}")
                return
        last_price = get_last_close_price(symbol)
        if not last_price or last_price <= 0:
            logger.warning(f"[Averaging] Нет актуальной цены для {symbol}.")
            return
        qty_in_coins = base_volume_usdt / Decimal(str(last_price))
        logger.info(f"[Averaging] Открытие усреднения для {symbol} на ~{qty_in_coins} монет.")
        order_result = asyncio.run(place_order(symbol, side, float(qty_in_coins), order_type="Market", time_in_force="GoodTillCancel", reduce_only=False, positionIdx=1 if side.lower() == "buy" else 2))
        if order_result and order_result.get("retCode") == 0:
            with open_positions_lock:
                averaging_positions[symbol] = {
                    "side": side,
                    "volume": base_volume_usdt,
                    "opened_at": datetime.datetime.utcnow(),
                    "trade_id": f"averaging_{symbol}_{int(time.time())}"
                }
            averaging_total_volume += base_volume_usdt
            logger.info(f"[Averaging] Усреднение для {symbol} открыто.")
        else:
            logger.error(f"[Averaging] Ошибка открытия усреднения для {symbol}: {order_result}")
    except Exception as e:
        logger.exception(f"[Averaging] Ошибка open_averaging_position для {symbol}: {e}")

# ===================== Функция generate_daily_pnl_report =====================
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
        logger.exception(f"Ошибка генерации PnL: {e}")
        return
    try:
        if not pnl_records:
            logger.info("Нет данных для отчёта PnL.")
            return
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["Дата/время", "Символ", "Объём в USDT", "Прибыль/Убыток"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in pnl_records:
                writer.writerow(r)
        logger.info(f"Отчёт PnL сохранён в {output_csv}")
    except Exception as e:
        logger.exception(f"Ошибка записи PnL: {e}")

# Функция get_position_info (асинхронная)
async def get_position_info(symbol, side):
    params = {"category": "linear", "symbol": symbol}
    resp = await async_client.get_positions(**params)
    if resp.get("retCode") != 0:
        logger.error(f"[get_position_info] {symbol}: {resp.get('retMsg')}")
        return None
    positions = resp["result"].get("list", [])
    for p in positions:
        if p.get("side", "").lower() == side.lower():
            return p
    return None

# Функция get_historical_data_for_model (асинхронная)
async def get_historical_data_for_model(symbol, interval="1", limit=200, from_time=None):
    return await get_historical_data_for_trading(symbol, interval, limit, from_time)

# Функция collect_historical_data (асинхронная)
async def collect_historical_data(symbols, interval="1", limit=200):
    dfs = []
    for sym in symbols:
        df = await get_historical_data_for_model(sym, interval, limit)
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
        logger.info("Historical data saved.")
    else:
        logger.info("Нет данных для исторического сохранения.")

# Функция process_symbol_model_only (асинхронная)
async def process_symbol_model_only(symbol):
    async with asyncio.Semaphore(5):
        await asyncio.to_thread(process_symbol_model_only_sync, symbol)

def generate_drift_table_from_history(top_n=15) -> str:
    if not drift_history:
        return ""
    rows = []
    for sym, recs in drift_history.items():
        if not recs:
            continue
        avg_strength = sum(x[1] for x in recs) / len(recs)
        last_dir = recs[-1][2]
        rows.append((sym, avg_strength, last_dir))
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top_n]
    console = Console(record=True, force_terminal=True, width=100)
    table = Table(title="Drift History", expand=True, box=box.ROUNDED)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Anomaly", justify="right", style="magenta")
    table.add_column("Dir", justify="center")
    for (sym, strength, direction) in rows:
        arrow = "🔴" if direction == "вверх" else "🟢"
        table.add_row(sym, f"{strength:.3f}", arrow)
    console.print(table)
    return console.export_text()

def generate_model_table_from_csv_no_time(csv_path="model_predictions_log.csv", last_n=200) -> str:
    if not os.path.isfile(csv_path):
        return ""
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return ""
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
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0
        p_buy = f"{safe_float(row.get('prob_buy', 0.0)):.3f}"
        p_hold = f"{safe_float(row.get('prob_hold', 0.0)):.3f}"
        p_sell = f"{safe_float(row.get('prob_sell', 0.0)):.3f}"
        table.add_row(sym, pred, p_buy, p_hold, p_sell)
    console.print(table)
    return console.export_text()

def process_symbol_model_only_sync(symbol):
    global current_model
    if not current_model:
        current_model = load_model()
        if not current_model:
            return
    # Получаем исторические данные с интервалом "5" минут
    df_5m = asyncio.run(get_historical_data_for_model(symbol, "5", limit=200))
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
        logger.exception(f"[MODEL_ONLY] Ошибка для {symbol}: {e}")
        return
    # Логирование характеристик и предсказания модели
    log_model_features_for_trade(symbol, symbol, "model", {})
    log_model_prediction(symbol, pred[0], proba)
    if pred[0] == 2:
        open_position(symbol, "Buy", POSITION_VOLUME, reason="Model")
    elif pred[0] == 0:
        open_position(symbol, "Sell", POSITION_VOLUME, reason="Model")
    else:
        logger.info(f"[MODEL_ONLY] {symbol}: HOLD, пропуск.")

async def publish_drift_and_model_tables():
    global telegram_bot, TELEGRAM_CHAT_ID
    if not telegram_bot or not TELEGRAM_CHAT_ID:
        logger.info("[publish_drift_and_model_tables] Telegram не инициализирован.")
        return
    if publish_drift_table:
        drift_str = generate_drift_table_from_history(top_n=10)
        if drift_str.strip():
            msg = f"```\n{drift_str}\n```"
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
        else:
            logger.info("[DRIFT] Таблица пуста.")
    if publish_model_table:
        model_str = generate_model_table_from_csv_no_time("model_predictions_log.csv", last_n=10)
        if model_str.strip():
            msg = f"```\n{model_str}\n```"
            await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
        else:
            logger.info("[MODEL] Таблица пуста.")

async def initialize_telegram_bot():
    global telegram_bot, router
    try:
        TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            telegram_bot = Bot(token=TELEGRAM_TOKEN)
            dp = Dispatcher(storage=MemoryStorage())
            if router.parent_router is None:
                dp.include_router(router)
            logger.info("Telegram бот инициализирован. Запуск polling...")
            asyncio.create_task(dp.start_polling(telegram_bot))
        else:
            logger.warning("Нет TELEGRAM_TOKEN или TELEGRAM_CHAT_ID.")
    except Exception as e:
        logger.exception(f"Ошибка инициализации Telegram: {e}")

async def send_initial_telegram_message():
    if telegram_bot and os.getenv("TELEGRAM_CHAT_ID"):
        try:
            test_msg = "✅ Бот успешно запущен. Введите '/menu' для меню."
            await telegram_bot.send_message(chat_id=os.getenv("TELEGRAM_CHAT_ID"), text=test_msg)
            logger.info("Отправлено сообщение о запуске Telegram.")
        except Exception as e:
            logger.exception(f"Ошибка отправки Telegram: {e}")

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
                        await telegram_bot.send_message(chat_id=os.getenv("TELEGRAM_CHAT_ID"), text=msg, parse_mode="MarkdownV2", disable_web_page_preview=True, request_timeout=120)
                    logger.info(f"[Telegram] Отправлено: {msg}")
                    break
                else:
                    logger.warning("[Telegram] Бот не инициализирован.")
                    break
            except asyncio.CancelledError:
                logger.info("Задача отправки сообщений отменена.")
                break
            except TelegramRetryAfter as e:
                await asyncio.sleep(e.retry_after)
            except TelegramBadRequest as e:
                logger.error(f"BadRequest Telegram: {e}")
                break
            except (TelegramNetworkError, asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.error(f"NetworkError: {e}, попытка {retry+1}")
                retry += 1
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as e:
                logger.exception(f"Ошибка отправки Telegram: {e}")
                retry += 1
                await asyncio.sleep(delay)
                delay *= 2
        else:
            logger.error(f"Не отправлено после {max_ret} попыток: {msg}")
        telegram_message_queue.task_done()

async def start_ws_monitor():
    from pybit.unified_trading import WebSocket
    ws_local = WebSocket(testnet=False, channel_type="linear")
    while True:
        async with asyncio.Lock():
            symbols = list(open_positions.keys())
        if not symbols:
            logger.info("[WS] Нет открытых позиций – сплю 10 секунд...")
            await asyncio.sleep(10)
            continue
        for symbol in symbols:
            logger.info(f"[WS] Подписка на kline_stream для {symbol}")
            ws_local.kline_stream(interval=1, symbol=symbol, callback=handle_position_update)
        await asyncio.sleep(1)

def handle_position_update(message):
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
            ratio = (current_price - entry_price) / entry_price if side.lower() == "buy" else (entry_price - current_price) / entry_price
            profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
            logger.info(f"[WS] {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%")
            if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
                logger.info(f"[WS] {symbol}: порог убытка достигнут, усредняем.")
                open_averaging_position(symbol)
            default_leverage = Decimal("10")
            leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))
            threshold_trailing = Decimal("5.0")
            with open_positions_lock:
                if symbol in open_positions and leveraged_pnl_percent >= threshold_trailing and not open_positions[symbol].get("trailing_stop_set", False):
                    logger.info(f"[WS] {symbol}: трейлинг-стоп, устанавливаю стоп.")
                    asyncio.run(set_trailing_stop(symbol, open_positions[symbol]["size"], TRAILING_GAP_PERCENT, side))
    else:
        logger.debug(f"[WS] Получено: {message}")

async def monitor_positions():
    while IS_RUNNING:
        try:
            await asyncio.sleep(5)
            positions = await get_exchange_positions()
            update_open_positions_from_exch_positions(positions)
            for symbol, pos in positions.items():
                side = pos["side"]
                entry_price = Decimal(str(pos["avg_price"]))
                current_price = get_last_close_price(symbol)
                if current_price is None:
                    logger.debug(f"[HTTP Monitor] Нет цены для {symbol}")
                    continue
                ratio = (Decimal(str(current_price)) - entry_price) / entry_price if side.lower() == "buy" else (entry_price - Decimal(str(current_price))) / entry_price
                profit_perc = (ratio * PROFIT_COEFFICIENT).quantize(Decimal("0.0001"))
                logger.info(f"[HTTP Monitor] {symbol}: current={current_price}, entry={entry_price}, PnL={profit_perc}%")
                with open_positions_lock:
                    if symbol in open_positions:
                        open_positions[symbol]['profit_perc'] = profit_perc
                if profit_perc <= -TARGET_LOSS_FOR_AVERAGING:
                    logger.info(f"[HTTP Monitor] {symbol}: порог убытка достигнут, усредняем.")
                    open_averaging_position(symbol)
                default_leverage = Decimal("10")
                leveraged_pnl_percent = (ratio * default_leverage * Decimal("100")).quantize(Decimal("0.0001"))
                threshold_trailing = Decimal("5.0")
                if leveraged_pnl_percent >= threshold_trailing:
                    with open_positions_lock:
                        if symbol in open_positions and not open_positions[symbol].get("trailing_stop_set", False):
                            logger.info(f"[HTTP Monitor] {symbol}: трейлинг-стоп, устанавливаю стоп.")
                            await set_trailing_stop(symbol, open_positions[symbol]["size"], TRAILING_GAP_PERCENT, side)
        except Exception as e_inner:
            logger.error(f"Ошибка в monitor_positions: {e_inner}")
            await asyncio.sleep(10)
            continue

async def async_drift_analyzer(interval: int = 60):
    global drift_running
    while drift_running:
        try:
            symbols = await get_selected_symbols()
            random.shuffle(symbols)
            for sym in symbols:
                if not drift_running:
                    break
                feature_cols = ["openPrice", "closePrice", "highPrice", "lowPrice"]
                new_data = await get_historical_data_for_trading(sym, "1", limit=200)
                if not new_data.empty:
                    is_anomaly, strength, direction = monitor_feature_drift_per_symbol(sym, new_data, pd.DataFrame(), feature_cols, threshold=0.5)
                    async with drift_lock:
                        drift_state["last_analysis"][sym] = {
                            'timestamp': datetime.datetime.utcnow(),
                            'is_anomaly': is_anomaly,
                            'strength': strength,
                            'direction': direction
                        }
            await asyncio.sleep(interval)
        except Exception as e:
            logger.exception(f"[DriftAnalyzer] Ошибка: {e}")
            await asyncio.sleep(10)

async def get_latest_drift_analysis():
    async with drift_lock:
        return dict(drift_state["last_analysis"])

def stop_drift_analyzer():
    global drift_running
    drift_running = False

# ===================== Телеграм обработка =====================
@router.message(Command(commands=["status"]))
async def status_cmd(message: Message):
    with open_positions_lock:
        if not open_positions:
            await message.reply("Нет позиций.")
            return
        lines = []
        total_pnl_usdt = Decimal("0")
        total_invested = Decimal("0")
        positions_copy = open_positions.copy()
    for sym, pos in positions_copy.items():
        try:
            side_str = pos["side"]
            entry_price = Decimal(str(pos["avg_price"]))
            volume_usdt = Decimal(str(pos["position_volume"]))
            current_price = get_last_close_price(sym)
            if current_price is None:
                lines.append(f"{sym} {side_str}: нет цены.")
                continue
            cp = Decimal(str(current_price))
            ratio = (cp - entry_price) / entry_price if side_str.lower() == "buy" else (entry_price - cp) / entry_price
            pnl_usdt = ratio * volume_usdt
            pnl_percent = ratio * Decimal("100")
            total_pnl_usdt += pnl_usdt
            total_invested += volume_usdt
            lines.append(f"{sym} {side_str}: PNL = {pnl_usdt:.2f} USDT ({pnl_percent:.2f}%)")
        except Exception as e:
            logger.error(f"Ошибка в статусе для {sym}: {e}")
            lines.append(f"{sym}: ошибка")
    lines.append("—" * 30)
    if total_invested > 0:
        total_pnl_percent = (total_pnl_usdt / total_invested) * Decimal("100")
        lines.append(f"Итоговый PnL: {total_pnl_usdt:.2f} USDT ({total_pnl_percent:.2f}%)")
    else:
        lines.append("Итоговый PnL: 0")
    await message.reply("\n".join(lines))

@router.message(Command("stop"))
async def stop_command(message: Message):
    global IS_RUNNING
    IS_RUNNING = False
    await message.answer("🛑 Бот останавливается...")
    logger.info("Получена команда /stop")
    for task in asyncio.all_tasks():
        task.cancel()

@router.message(Command(commands=["menu"]))
async def main_menu_cmd(message: Message):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📈 Торговля"), KeyboardButton(text="🤖 Бот")],
            [KeyboardButton(text="ℹ️ Информация")]
        ],
        resize_keyboard=True
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
    await query.message.answer("Введите команду: `/setmaxvolume 500`", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_setposvolume")
async def process_cmd_setposvolume(query: CallbackQuery):
    await query.message.answer("Введите команду: `/setposvolume 50`", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_setsttf")
async def process_cmd_setsttf(query: CallbackQuery):
    await query.message.answer("Введите команду: `/setsttf 15`", parse_mode="Markdown")

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
    message_text = f"*Текущий режим*: {current_mode_label}\n\nВыберите новый режим:"
    await message.answer(message_text, reply_markup=keyboard, parse_mode="Markdown")

@router.message(Command(commands=["togglesilence"]))
async def toggle_silence_cmd(message: Message):
    st = toggle_quiet_period()
    await message.reply(f"Тихий режим: {st}")

@router.message(Command(commands=["silencestatus"]))
async def silence_status_cmd(message: Message):
    st = "включён" if os.getenv("QUIET_PERIOD_ENABLED", "False") == "True" else "выключен"
    await message.reply(f"Тихий режим: {st}")

@router.callback_query(lambda c: c.data == "cmd_sleep")
async def process_cmd_sleep(query: CallbackQuery):
    global QUIET_PERIOD_ENABLED
    QUIET_PERIOD_ENABLED = True
    await query.message.answer("😴 Бот переведен в спящий режим.", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_wake")
async def process_cmd_wake(query: CallbackQuery):
    global QUIET_PERIOD_ENABLED
    QUIET_PERIOD_ENABLED = False
    await query.message.answer("🌞 Бот разбужен, торговля возобновлена.", parse_mode="Markdown")

@router.callback_query(lambda c: c.data == "cmd_getpair")
async def process_cmd_getpair(query: CallbackQuery):
    await query.message.answer("Введите команду: `/getpair BTCUSDT или BTC`", parse_mode="Markdown")

@router.message(Command(commands=["inline_menu"]))
async def inline_menu_command(message: Message):
    inline_kb = [
        [InlineKeyboardButton(text="Status", callback_data="cmd_status"),
         InlineKeyboardButton(text="Toggle Silence", callback_data="cmd_togglesilence")],
        [InlineKeyboardButton(text="Silence Status", callback_data="cmd_silencestatus"),
         InlineKeyboardButton(text="Set Max Volume", callback_data="cmd_setmaxvolume")],
        [InlineKeyboardButton(text="Set Pos Volume", callback_data="cmd_setposvolume"),
         InlineKeyboardButton(text="Set ST TF", callback_data="cmd_setsttf")]
    ]
    markup = InlineKeyboardMarkup(inline_keyboard=inline_kb)
    await message.answer("Выберите действие:", reply_markup=markup)

@router.callback_query(lambda c: c.data and c.data.startswith("cmd_"))
async def process_inline_commands(query: CallbackQuery):
    data = query.data
    if data == "cmd_status":
        await query.message.answer("Вызван STATUS.")
    elif data == "cmd_togglesilence":
        await query.message.answer("Вызван TOGGLE SILENCE.")
    elif data == "cmd_silencestatus":
        await query.message.answer("Вызван SILENCE STATUS.")
    elif data == "cmd_setmaxvolume":
        await query.message.answer("Вызван SET MAX VOLUME. Пример: /setmaxvolume 500")
    elif data == "cmd_setposvolume":
        await query.message.answer("Вызван SET POS VOLUME. Пример: /setposvolume 50")
    elif data == "cmd_setsttf":
        await query.message.answer("Вызван SET ST TF. Пример: /setsttf 15")
    await query.answer()

@router.message(Command(commands=["sleep"]))
async def sleep_cmd(message: Message):
    status = toggle_sleep_mode()
    await message.reply(f"Спящий режим: {status}")

@router.message(Command(commands=["wake"]))
async def wake_cmd(message: Message):
    status = toggle_sleep_mode()
    await message.reply(f"Спящий режим: {status}")

def toggle_quiet_period():
    global QUIET_PERIOD_ENABLED
    QUIET_PERIOD_ENABLED = not QUIET_PERIOD_ENABLED
    return "включён" if QUIET_PERIOD_ENABLED else "выключен"

def toggle_sleep_mode():
    global IS_SLEEPING_MODE
    IS_SLEEPING_MODE = not IS_SLEEPING_MODE
    return "включен" if IS_SLEEPING_MODE else "выключен"

# ===================== Главная асинхронная функция =====================
async def main_coroutine():
    global async_client
    async_client = AsyncBybitClient(demo=True, api_key=os.getenv("BYBIT_API_KEY"), api_secret=os.getenv("BYBIT_API_SECRET"), timeout=60)
    global current_model, IS_RUNNING
    IS_RUNNING = True
    loop = asyncio.get_running_loop()
    global telegram_message_queue
    telegram_message_queue = asyncio.Queue()
    logger.info("=== Запуск основного цикла ===")
    state["total_open_volume"] = Decimal("0")
    telegram_sender_task = asyncio.create_task(telegram_message_sender())
    tg_task = asyncio.create_task(initialize_telegram_bot())
    await asyncio.sleep(3)
    await send_initial_telegram_message()
    current_model = load_model()
    symbols_all = await get_usdt_pairs()
    await collect_historical_data(symbols_all, "1", 200)
    exch_positions = await get_exchange_positions()
    update_open_positions_from_exch_positions(exch_positions)
    drift_task = asyncio.create_task(async_drift_analyzer(60))
    monitor_task = None
    if MONITOR_MODE == "ws":
        logger.info("[Main] Режим мониторинга: WebSocket")
        asyncio.create_task(start_ws_monitor())
    elif MONITOR_MODE == "http":
        logger.info("[Main] Режим мониторинга: HTTP")
        monitor_task = asyncio.create_task(monitor_positions())
    else:
        logger.warning(f"[Main] Неизвестный режим: {MONITOR_MODE}")
    iteration_count = 0
    try:
        while IS_RUNNING:
            try:
                exch_positions = await get_exchange_positions()
                update_open_positions_from_exch_positions(exch_positions)
            except Exception as e:
                logger.error(f"Ошибка синхронизации позиций: {e}")
                await asyncio.sleep(5)
                continue
            if tg_task.done():
                exc = tg_task.exception()
                if exc:
                    logger.exception("Telegram-бот упал:", exc)
                else:
                    logger.error("Telegram-бот завершился.")
                logger.info("Перезапуск Telegram-бота через 10 секунд...")
                await asyncio.sleep(10)
                tg_task = asyncio.create_task(initialize_telegram_bot())
            iteration_count += 1
            logger.info(f"[INNER_LOOP] iteration_count={iteration_count}")
            symbols = await get_selected_symbols()
            random.shuffle(symbols)
            logger.info(f"[TRADING] Режим: {OPERATION_MODE}")
            if OPERATION_MODE == "model_only":
                tasks = [process_symbol_model_only(s) for s in symbols]
                if tasks:
                    await asyncio.gather(*tasks)
            elif OPERATION_MODE in ["drift_only", "drift_top10"]:
                latest_analysis = await get_latest_drift_analysis()
                top_list = get_top_anomalies_from_analysis(latest_analysis)
                if OPERATION_MODE == "drift_top10" and top_list:
                    handle_drift_top10(top_list)
            elif OPERATION_MODE == "golden_setup":
                for s in symbols:
                    df_5m = await get_historical_data_for_trading(s, "1", limit=20)
                    if not df_5m.empty:
                        action, _ = handle_golden_setup(s, df_5m)
                        if action:
                            open_position(s, action, POSITION_VOLUME, reason="Golden")
            elif OPERATION_MODE == "super_trend":
                tasks = [asyncio.to_thread(process_symbol_supertrend_open, s, interval=SUPER_TREND_TIMEFRAME, length=8, multiplier=3.0) for s in symbols]
                if tasks:
                    await asyncio.gather(*tasks)
            elif OPERATION_MODE in ["ST_cross_global", "ST_cross1", "ST_cross2", "ST_cross2_drift"]:
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
            tasks_log = [asyncio.to_thread(log_model_prediction_for_symbol, s) for s in symbols]
            if tasks_log:
                await asyncio.gather(*tasks_log)
            if iteration_count % 20 == 0:
                logger.info(f"[INNER_LOOP] iteration_count={iteration_count}, вызываем maybe_retrain_model()")
                await maybe_retrain_model()
            final_expos = await get_exchange_positions()
            update_open_positions_from_exch_positions(final_expos)
            await asyncio.to_thread(generate_daily_pnl_report, "trade_log.csv", "daily_pnl_report.csv")
            await asyncio.sleep(60)
    except Exception as e_outer:
        logger.exception(f"Ошибка во внешнем цикле: {e_outer}")
    finally:
        stop_drift_analyzer()
        drift_task.cancel()
        if telegram_bot and os.getenv("TELEGRAM_CHAT_ID"):
            try:
                await telegram_bot.send_message(chat_id=os.getenv("TELEGRAM_CHAT_ID"), text="✅ Бот успешно остановлен")
            except Exception as e:
                logger.error(f"Ошибка отправки сообщения об остановке: {e}")
        await async_client.close()

async def main():
    await main_coroutine()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка пользователем.")
    except Exception as e:
        logger.exception(f"Ошибка main: {e}")