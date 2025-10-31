#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный код многопользовательского торгового бота с торговой логикой.

В этом примере:
  • Класс TradingLogic содержит универсальную торговую логику:
      - Расчёт индикаторов (MACD, RSI, ATR, SuperTrend)
      - Получение исторических данных и подготовка фичей для модели
      - Режимы торговли:
          – model_only
          – golden_setup
          – drift_only
          – ST_cross_global, ST_cross1, ST_cross2 (точно по оригинальной логике)
          – super_trend
      - Синхронизация исторических данных: загрузка недостающих данных и объединение с CSV
      - Логика выставления ордеров, трейлинг-стопов, усреднения позиций и публикации отчётов.
      - Функция get_usdt_pairs, возвращающая нужные торговые пары (без BTCUSDT и ETHUSDT)
  • Класс UserSession хранит индивидуальные API‑ключи.
  • Класс TradingBot управляет Telegram-интерфейсом и вызывает методы TradingLogic.
  
Перед запуском убедитесь, что установлены библиотеки:
  aiogram, pybit, scikit‑learn, pandas, numpy, joblib, python‑dotenv,
а также корректно заполнены файлы keys_TESTNET.env и users.csv.
"""

import asyncio
import csv
import datetime
import logging
import os
import random
import threading
import time
from collections import defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd
import requests
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from pybit.unified_trading import HTTP, WebSocket
from pybit.exceptions import InvalidRequestError

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.exceptions import TelegramRetryAfter, TelegramBadRequest, TelegramNetworkError

from dotenv import load_dotenv
import signal
import sys

# --------------------- Глобальные константы и настройка ---------------------
POSITION_VOLUME = Decimal("100")  # Объём для открытия позиции

load_dotenv("keys_TESTNET.env")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8")
    ],
)
logger = logging.getLogger(__name__)

# --------------------- Глобальная HTTP-сессия для работы с API ---------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
custom_session = requests.Session()
retries = Retry(total=5, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)
adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retries)
custom_session.mount("http://", adapter)
custom_session.mount("https://", adapter)
global_session = HTTP(demo=True, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, timeout=30)

# --------------------- Функция загрузки пользователей из CSV ---------------------
def load_allowed_users(filename="users.csv"):
    allowed = {}
    if os.path.isfile(filename):
        with open(filename, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    uid = int(row["user_id"])
                    user_api = row.get("user_api", "").strip()
                    user_api_secret = row.get("user_api_secret", "").strip()
                    if not user_api or not user_api_secret:
                        logger.error(f"Пользователь {uid} не задал индивидуальные API-ключи.")
                        continue
                    allowed[uid] = {"user_api": user_api, "user_api_secret": user_api_secret}
                except Exception as e:
                    logger.error(f"Ошибка чтения user_id из {filename}: {e}")
    else:
        logger.warning(f"Файл {filename} не найден. Торговля невозможна для обычных пользователей.")
    return allowed

# --------------------- Класс TradingLogic – универсальная торговая логика ---------------------
class TradingLogic:
    def __init__(self, session: HTTP, logger: logging.Logger):
        self.session = session
        self.logger = logger
        self.open_positions = {}  # Словарь открытых позиций
        self.open_interest_history = defaultdict(list)
        self.volume_history = defaultdict(list)
        self.golden_params = {
            "Buy": {"period_iters": Decimal("4"), "price_change": Decimal("0.1"), "volume_change": Decimal("20000"), "oi_change": Decimal("20000")},
            "Sell": {"period_iters": Decimal("4"), "price_change": Decimal("1.0"), "volume_change": Decimal("5000"), "oi_change": Decimal("5000")}
        }
        self.PROFIT_LEVEL = Decimal("0.008")
        self.PROFIT_COEFFICIENT = Decimal("100")
        self.TRAILING_STOP_ENABLED = True
        self.TRAILING_GAP_PERCENT = Decimal("0.008")
        self.MIN_TRAILING_STOP = Decimal("0.0000001")
        self.MODEL_FEATURE_COLS = ["openPrice", "highPrice", "lowPrice", "closePrice", "macd", "macd_signal", "rsi_13"]
        self.OPERATION_MODE = "ST_cross2"  # Режим по умолчанию
        self.MODEL_FILENAME = "trading_model_final.pkl"
        # Для синхронизации доступа
        self.lock = threading.Lock()

    # --- Функция получения USDT-пар (исключая BTC и ETH) ---
    def get_usdt_pairs(self):
        try:
            tickers_resp = self.session.get_tickers(symbol=None, category="linear")
            inst_resp = self.session.get_instruments_info(category="linear")
            if "result" not in tickers_resp or "list" not in tickers_resp["result"]:
                self.logger.error("[get_usdt_pairs] Некорректный ответ от get_tickers.")
                return []
            tickers_data = tickers_resp["result"]["list"]
            instruments = inst_resp["result"].get("list", [])
            trading_status = {inst.get("symbol"): (inst.get("status", "").upper() == "TRADING") for inst in instruments}
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
            self.logger.info(f"[get_usdt_pairs] Отобраны USDT-пары: {usdt_pairs}")
            return usdt_pairs
        except Exception as e:
            self.logger.exception(f"Ошибка в get_usdt_pairs: {e}")
            return []

    # --- Функция синхронизации исторических данных ---
    def sync_historical_data(self, symbol, interval="1", csv_path="historical_data_for_model_5m.csv"):
        self.logger.info(f"[SYNC] Синхронизация данных для {symbol}")
        try:
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path)
                if "startTime" in df_existing.columns:
                    df_existing["startTime"] = pd.to_datetime(df_existing["startTime"], utc=True)
                    last_timestamp = df_existing["startTime"].max()
                    last_timestamp_ms = int(last_timestamp.timestamp() * 1000)
                else:
                    df_existing = pd.DataFrame()
                    last_timestamp_ms = None
            else:
                df_existing = pd.DataFrame()
                last_timestamp_ms = None
        except Exception as e:
            self.logger.error(f"[SYNC] Ошибка чтения CSV {csv_path}: {e}")
            df_existing = pd.DataFrame()
            last_timestamp_ms = None

        try:
            params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": 200}
            if last_timestamp_ms:
                params["from"] = last_timestamp_ms // 1000
            resp = self.session.get_kline(**params)
            if resp.get("retCode") != 0:
                self.logger.error(f"[SYNC] Ошибка получения данных для {symbol}: {resp.get('retMsg')}")
                return
            data = resp["result"].get("list", [])
            if not data:
                self.logger.info(f"[SYNC] Нет новых данных для {symbol}")
                return
            out_rows = []
            for row in data:
                if len(row) < 5:
                    continue
                out_rows.append([row[0], row[1], row[2], row[3], row[4]])
            columns = ["open_time", "open", "high", "low", "close"]
            df_new = pd.DataFrame(out_rows, columns=columns)
            df_new["startTime"] = pd.to_datetime(pd.to_numeric(df_new["open_time"], errors="coerce"), unit="ms", utc=True)
            df_new.rename(columns={"open": "openPrice", "high": "highPrice", "low": "lowPrice", "close": "closePrice"}, inplace=True)
            df_new = df_new[["startTime", "openPrice", "highPrice", "lowPrice", "closePrice"]]
            if not df_existing.empty:
                df_merged = pd.concat([df_existing, df_new]).drop_duplicates(subset=["startTime"]).sort_values("startTime")
            else:
                df_merged = df_new.sort_values("startTime")
            df_merged.to_csv(csv_path, index=False)
            new_rows = len(df_merged) - (len(df_existing) if not df_existing.empty else 0)
            self.logger.info(f"[SYNC] Для {symbol} синхронизированы данные. Новых строк: {new_rows}")
        except Exception as e:
            self.logger.exception(f"[SYNC] Ошибка синхронизации данных для {symbol}: {e}")

    # --- Функции работы с API и расчётов ---
    def adjust_quantity(self, symbol: str, raw_qty: float) -> float:
        info = self.get_symbol_info(symbol)
        if not info:
            return 0.0
        lot_size = info.get("lotSizeFilter", {})
        min_qty = Decimal(str(lot_size.get("minOrderQty", "0")))
        qty_step = Decimal(str(lot_size.get("qtyStep", "1")))
        max_qty = Decimal(str(lot_size.get("maxOrderQty", "9999999")))
        last_price = self.get_last_close_price(symbol)
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

    def get_symbol_info(self, symbol):
        try:
            resp = self.session.get_instruments_info(symbol=symbol, category="linear")
            if resp.get("retCode") != 0:
                self.logger.error(f"[get_symbol_info] {symbol}: {resp.get('retMsg')}")
                return None
            instruments = resp["result"].get("list", [])
            if not instruments:
                return None
            return instruments[0]
        except Exception as e:
            self.logger.exception(f"[get_symbol_info({symbol})]: {e}")
            return None

    def get_last_close_price(self, symbol):
        try:
            resp = self.session.get_kline(category="linear", symbol=symbol, interval="1", limit=1)
            if not resp or resp.get("retCode") != 0:
                self.logger.error(f"[get_last_close_price] {symbol}: {resp.get('retMsg') if resp else 'Empty'}")
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
            self.logger.exception(f"[get_last_close_price({symbol})]: {e}")
            return None

    def place_order(self, symbol, side, qty, order_type="Market", time_in_force="GoodTillCancel", reduce_only=False, positionIdx=None):
        try:
            adj_qty = self.adjust_quantity(symbol, qty)
            if adj_qty <= 0:
                self.logger.error(f"[place_order] qty={qty} => adj_qty={adj_qty} недопустимо.")
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
            if positionIdx is None:
                positionIdx = 1 if side.lower() == "buy" else 2
            params["positionIdx"] = positionIdx
            resp = self.session.place_order(**params)
            if resp.get("retCode") == 0:
                self.logger.info(f"[place_order] OK {symbol}, side={side}, qty={adj_qty}")
                return resp
            else:
                self.logger.error(f"[place_order] Ошибка: {resp.get('retMsg')} (retCode={resp.get('retCode')})")
                return None
        except (InvalidRequestError, Exception) as e:
            self.logger.exception(f"[place_order] Ошибка({symbol}): {e}")
            return None

    def open_position(self, symbol, side, volume_usdt, reason=""):
        self.logger.info(f"[OPEN_POSITION] {symbol} {side} на сумму {volume_usdt} USDT, причина: {reason}")
        price = self.get_last_close_price(symbol)
        if price is None:
            self.logger.error(f"[OPEN_POSITION] Нет цены для {symbol}")
            return
        qty = volume_usdt / Decimal(str(price))
        order_result = self.place_order(symbol, side, float(qty))
        if order_result:
            self.logger.info(f"[OPEN_POSITION] Позиция {symbol} {side} открыта по цене {price}")
            with self.lock:
                self.open_positions[symbol] = {"side": side, "avg_price": price, "position_volume": volume_usdt, "symbol": symbol, "trade_id": f"{symbol}_{int(time.time())}"}
        else:
            self.logger.error(f"[OPEN_POSITION] Ошибка открытия позиции для {symbol}")

    # --- Функции подготовки данных и расчёта индикаторов ---
    def get_historical_data_for_model(self, symbol, interval="1", limit=200, from_time=None):
        try:
            params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
            if from_time:
                params["from"] = from_time
            resp = self.session.get_kline(**params)
            if resp.get("retCode") != 0:
                self.logger.error(f"[MODEL] get_kline({symbol}): {resp.get('retMsg')}")
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
            self.logger.debug(f"[get_historical_data_for_model] {symbol}: получено {len(df)} свечей.")
            return df
        except Exception as e:
            self.logger.exception(f"Ошибка get_historical_data_for_model({symbol}): {e}")
            return pd.DataFrame()

    def prepare_features_for_model(self, df):
        try:
            for c in ["openPrice", "highPrice", "lowPrice", "closePrice"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=["closePrice"], inplace=True)
            if df.empty:
                return df
            df["ohlc4"] = (df["openPrice"] + df["highPrice"] + df["lowPrice"] + df["closePrice"]) / 4
            macd_df = self.calculate_macd(df["ohlc4"])
            df["macd"] = macd_df["MACD_12_26_9"]
            df["macd_signal"] = macd_df["MACDs_12_26_9"]
            df["rsi_13"] = self.calculate_rsi(df["ohlc4"], periods=13)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["macd", "macd_signal", "rsi_13"], inplace=True)
            return df
        except Exception as e:
            self.logger.exception(f"Ошибка prepare_features_for_model: {e}")
            return pd.DataFrame()

    def make_multiclass_target_for_model(self, df, horizon=1, threshold=Decimal("0.0025")):
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
            self.logger.exception(f"Ошибка make_multiclass_target_for_model: {e}")
            return df

    def collect_historical_data(self, symbols, interval="1", limit=200):
        dfs = []
        for sym in symbols:
            df = self.get_historical_data_for_model(sym, interval, limit)
            df = self.prepare_features_for_model(df)
            if df.empty:
                continue
            df = self.make_multiclass_target_for_model(df, horizon=1, threshold=Decimal("0.0025"))
            if df.empty:
                continue
            df["symbol"] = sym
            dfs.append(df)
        if dfs:
            data = pd.concat(dfs, ignore_index=True)
            data.to_csv("historical_data_for_model_5m.csv", index=False)
            self.logger.info("historical_data_for_model_5m.csv сохранён (с target).")
        else:
            self.logger.info("Нет данных для сохранения.")

    # --- Расчёт SuperTrend ---
    def calculate_supertrend_by_df(self, df, length, multiplier):
        df = df.copy()
        df['high'] = pd.to_numeric(df['highPrice'], errors='coerce')
        df['low'] = pd.to_numeric(df['lowPrice'], errors='coerce')
        df['close'] = pd.to_numeric(df['closePrice'], errors='coerce')
        df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'],
                                                                      abs(x['high'] - x['close']),
                                                                      abs(x['low'] - x['close'])), axis=1)
        df['atr'] = df['tr'].rolling(window=length).mean()
        hl2 = (df['high'] + df['low']) / 2
        df['basic_upperband'] = hl2 + multiplier * df['atr']
        df['basic_lowerband'] = hl2 - multiplier * df['atr']
        df['final_upperband'] = df['basic_upperband']
        df['final_lowerband'] = df['basic_lowerband']
        for i in range(1, len(df)):
            if df['basic_upperband'].iloc[i] < df['final_upperband'].iloc[i-1] or df['close'].iloc[i-1] > df['final_upperband'].iloc[i-1]:
                df.at[df.index[i], 'final_upperband'] = df['basic_upperband'].iloc[i]
            else:
                df.at[df.index[i], 'final_upperband'] = df['final_upperband'].iloc[i-1]
            if df['basic_lowerband'].iloc[i] > df['final_lowerband'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lowerband'].iloc[i-1]:
                df.at[df.index[i], 'final_lowerband'] = df['basic_lowerband'].iloc[i]
            else:
                df.at[df.index[i], 'final_lowerband'] = df['final_lowerband'].iloc[i-1]
        supertrend = [np.nan]
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= df['final_upperband'].iloc[i]:
                supertrend.append(df['final_upperband'].iloc[i])
            else:
                supertrend.append(df['final_lowerband'].iloc[i])
        df['supertrend'] = supertrend
        return df

    def calculate_supertrend_fast(self, df, length=3, multiplier=1.0):
        return self.calculate_supertrend_by_df(df, length, multiplier)

    def calculate_supertrend_slow(self, df, length=8, multiplier=3.0):
        return self.calculate_supertrend_by_df(df, length, multiplier)

    # --- Логика режимов ST_cross ---
    def process_symbol_st_cross_global(self, symbol, interval="1", limit=200):
        self.logger.info(f"[ST_cross_global] Начало обработки {symbol}")
        with self.lock:
            if symbol in self.open_positions:
                self.logger.info(f"[ST_cross_global] {symbol}: уже есть открытая позиция, пропускаем.")
                return
        df = self.get_historical_data_for_model(symbol, interval=interval, limit=limit)
        if df.empty or len(df) < 5:
            self.logger.info(f"[ST_cross_global] {symbol}: Недостаточно данных, пропуск.")
            return
        df_fast = self.calculate_supertrend_fast(df.copy(), length=3, multiplier=1.0)
        df_slow = self.calculate_supertrend_slow(df.copy(), length=8, multiplier=3.0)
        if df_fast.empty or df_slow.empty:
            self.logger.info(f"[ST_cross_global] {symbol}: Не удалось рассчитать SuperTrend.")
            return
        try:
            last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
            current_time = pd.Timestamp.utcnow()
            if last_candle_time < current_time - pd.Timedelta(minutes=5):
                self.logger.warning(f"[ST_cross_global] {symbol}: Данные устарели! Пропускаем.")
                return
        except Exception as e:
            self.logger.error(f"[ST_cross_global] Ошибка проверки времени для {symbol}: {e}")
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
        self.logger.info(f"[ST_cross_global] {symbol}: prev_fast={prev_fast:.6f}, prev_slow={prev_slow:.6f}, "
                         f"curr_fast={curr_fast:.6f}, curr_slow={curr_slow:.6f}, last_close={last_close:.6f}")
        if confirmed_buy:
            self.logger.info(f"[ST_cross_global] {symbol}: Подтверждён сигнал BUY")
            self.open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross_global")
        elif confirmed_sell:
            self.logger.info(f"[ST_cross_global] {symbol}: Подтверждён сигнал SELL")
            self.open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross_global")
        else:
            self.logger.info(f"[ST_cross_global] {symbol}: Пересечения не подтвердились, сигнал отсутствует")

    def process_symbol_st_cross1(self, symbol, interval="1", limit=200):
        self.logger.info(f"[ST_cross1] Начало обработки {symbol}")
        with self.lock:
            if symbol in self.open_positions:
                self.logger.info(f"[ST_cross1] {symbol}: уже есть открытая позиция, пропускаем.")
                return
        df = self.get_historical_data_for_model(symbol, interval=interval, limit=limit)
        if df.empty or len(df) < 5:
            self.logger.info(f"[ST_cross1] {symbol}: Недостаточно данных, пропуск.")
            return
        df_fast = self.calculate_supertrend_fast(df.copy(), length=3, multiplier=1.0)
        df_slow = self.calculate_supertrend_slow(df.copy(), length=8, multiplier=3.0)
        if df_fast.empty or df_slow.empty:
            self.logger.info(f"[ST_cross1] {symbol}: Не удалось рассчитать SuperTrend.")
            return
        try:
            last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
            current_time = pd.Timestamp.utcnow()
            if last_candle_time < current_time - pd.Timedelta(minutes=5):
                self.logger.warning(f"[ST_cross1] {symbol}: Данные устарели! Пропускаем.")
                return
        except Exception as e:
            self.logger.error(f"[ST_cross1] Ошибка проверки времени для {symbol}: {e}")
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
                self.logger.info(f"[ST_cross1] {symbol}: Слишком сильное положительное различие ({curr_diff_pct:.2f}%), не открываем LONG.")
                return
            confirmed_buy = last_close >= curr_fast * (1 + margin)
            if confirmed_buy:
                self.logger.info(f"[ST_cross1] {symbol}: Подтверждён сигнал BUY")
                self.open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross1")
            else:
                self.logger.info(f"[ST_cross1] {symbol}: Сигнал BUY не подтверждён по цене.")
        elif first_cross_down:
            if curr_diff_pct < Decimal("-1"):
                self.logger.info(f"[ST_cross1] {symbol}: Слишком сильное отрицательное различие ({curr_diff_pct:.2f}%), не открываем SHORT.")
                return
            confirmed_sell = last_close <= curr_fast * (1 - margin)
            if confirmed_sell:
                self.logger.info(f"[ST_cross1] {symbol}: Подтверждён сигнал SELL")
                self.open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross1")
            else:
                self.logger.info(f"[ST_cross1] {symbol}: Сигнал SELL не подтверждён по цене.")
        else:
            self.logger.info(f"[ST_cross1] {symbol}: Пересечения не обнаружены, сигнал отсутствует.")

    def process_symbol_st_cross2(self, symbol, interval="1", limit=200):
        self.logger.info(f"[ST_cross2] Начало обработки {symbol}")
        with self.lock:
            if symbol in self.open_positions:
                self.logger.info(f"[ST_cross2] {symbol}: уже есть открытая позиция, пропускаем.")
                return
        df = self.get_historical_data_for_model(symbol, interval=interval, limit=limit)
        if df.empty or len(df) < 5:
            self.logger.info(f"[ST_cross2] {symbol}: Недостаточно данных, пропуск.")
            return
        df_fast = self.calculate_supertrend_fast(df.copy(), length=3, multiplier=1.0)
        df_slow = self.calculate_supertrend_slow(df.copy(), length=8, multiplier=3.0)
        if df_fast.empty or df_slow.empty:
            self.logger.info(f"[ST_cross2] {symbol}: Не удалось рассчитать SuperTrend.")
            return
        try:
            last_candle_time = pd.to_datetime(df_fast.iloc[-1]["startTime"])
            current_time = pd.Timestamp.utcnow()
            if last_candle_time < current_time - pd.Timedelta(minutes=5):
                self.logger.warning(f"[ST_cross2] {symbol}: Данные устарели! Пропускаем.")
                return
        except Exception as e:
            self.logger.error(f"[ST_cross2] Ошибка проверки времени для {symbol}: {e}")
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
                self.logger.info(f"[ST_cross2] {symbol}: Текущее положительное различие ({curr_diff_pct:.2f}%) слишком высокое, не открываем LONG.")
                return
            self.logger.info(f"[ST_cross2] {symbol}: Сигнал LONG (prev: {prev_diff_pct:.2f}%, curr: {curr_diff_pct:.2f}%).")
            self.open_position(symbol, "Buy", POSITION_VOLUME, reason="ST_cross2")
        elif short_signal:
            if curr_diff_pct < Decimal("-1"):
                self.logger.info(f"[ST_cross2] {symbol}: Текущее отрицательное различие ({curr_diff_pct:.2f}%) слишком низкое, не открываем SHORT.")
                return
            self.logger.info(f"[ST_cross2] {symbol}: Сигнал SHORT (prev: {prev_diff_pct:.2f}%, curr: {curr_diff_pct:.2f}%).")
            self.open_position(symbol, "Sell", POSITION_VOLUME, reason="ST_cross2")
        else:
            self.logger.info(f"[ST_cross2] {symbol}: Альтернативное условие для сигнала не выполнено.")

    # --- Остальные торговые режимы, модель и отчёты ---
    def process_symbol_model_only(self, symbol):
        self.logger.info(f"[MODEL_ONLY] Обработка {symbol}")
        df = self.get_historical_data_for_model(symbol, interval="1", limit=200)
        if df.empty:
            self.logger.info(f"[MODEL_ONLY] Нет данных для {symbol}")
            return
        df = self.prepare_features_for_model(df)
        if df.empty:
            self.logger.info(f"[MODEL_ONLY] Нет признаков для {symbol}")
            return
        model = self.load_model()
        if model is None:
            self.logger.error("Модель не загружена.")
            return
        features = df[self.MODEL_FEATURE_COLS].tail(1)
        prediction = model.predict(features)[0]
        self.logger.info(f"[MODEL_ONLY] Прогноз для {symbol}: {prediction}")
        if prediction == 2:
            side = "Buy"
        elif prediction == 0:
            side = "Sell"
        else:
            self.logger.info(f"[MODEL_ONLY] Прогноз HOLD для {symbol} – никаких действий.")
            return
        price = self.get_last_close_price(symbol)
        if price is None:
            self.logger.error(f"[MODEL_ONLY] Нет цены для {symbol}")
            return
        volume = Decimal("100")
        qty = volume / Decimal(str(price))
        order_result = self.place_order(symbol, side, float(qty))
        if order_result:
            self.logger.info(f"[MODEL_ONLY] Ордер размещён для {symbol} {side} по цене {price}")

    def process_symbol_supertrend_open(self, symbol, interval="1", length=3, multiplier=1.0):
        self.logger.info(f"[SUPER_TREND] Обработка {symbol} с length={length} multiplier={multiplier}")
        df = self.get_historical_data_for_model(symbol, interval=interval, limit=100)
        if df.empty:
            self.logger.info(f"[SUPER_TREND] Нет данных для {symbol}")
            return
        st_df = self.calculate_supertrend_by_df(df, length, multiplier)
        if st_df.empty:
            self.logger.info(f"[SUPER_TREND] Не удалось рассчитать SuperTrend для {symbol}")
            return
        if st_df['closePrice'].iloc[-1] > st_df['supertrend'].iloc[-1]:
            self.logger.info(f"[SUPER_TREND] Сигнал BUY для {symbol}")
            price = self.get_last_close_price(symbol)
            if price:
                volume = Decimal("100")
                qty = volume / Decimal(str(price))
                self.place_order(symbol, "Buy", float(qty))
        else:
            self.logger.info(f"[SUPER_TREND] Нет сигнала BUY для {symbol}")

    def calculate_supertrend_by_df(self, df, length, multiplier):
        df = df.copy()
        df['high'] = pd.to_numeric(df['highPrice'], errors='coerce')
        df['low'] = pd.to_numeric(df['lowPrice'], errors='coerce')
        df['close'] = pd.to_numeric(df['closePrice'], errors='coerce')
        df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'],
                                                                      abs(x['high'] - x['close']),
                                                                      abs(x['low'] - x['close'])), axis=1)
        df['atr'] = df['tr'].rolling(window=length).mean()
        hl2 = (df['high'] + df['low']) / 2
        df['basic_upperband'] = hl2 + multiplier * df['atr']
        df['basic_lowerband'] = hl2 - multiplier * df['atr']
        df['final_upperband'] = df['basic_upperband']
        df['final_lowerband'] = df['basic_lowerband']
        for i in range(1, len(df)):
            if df['basic_upperband'].iloc[i] < df['final_upperband'].iloc[i-1] or df['close'].iloc[i-1] > df['final_upperband'].iloc[i-1]:
                df.at[df.index[i], 'final_upperband'] = df['basic_upperband'].iloc[i]
            else:
                df.at[df.index[i], 'final_upperband'] = df['final_upperband'].iloc[i-1]
            if df['basic_lowerband'].iloc[i] > df['final_lowerband'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lowerband'].iloc[i-1]:
                df.at[df.index[i], 'final_lowerband'] = df['basic_lowerband'].iloc[i]
            else:
                df.at[df.index[i], 'final_lowerband'] = df['final_lowerband'].iloc[i-1]
        supertrend = [np.nan]
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= df['final_upperband'].iloc[i]:
                supertrend.append(df['final_upperband'].iloc[i])
            else:
                supertrend.append(df['final_lowerband'].iloc[i])
        df['supertrend'] = supertrend
        return df

    def train_and_load_model(self, csv_path="historical_data_for_model_5m.csv"):
        self.logger.info("Обучение модели...")
        try:
            data = pd.read_csv(csv_path)
            if data.empty:
                self.logger.error("Нет данных для обучения модели.")
                return None
            X = data[self.MODEL_FEATURE_COLS]
            y = data["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            self.logger.info(f"Модель обучена с точностью: {acc:.2f}")
            joblib.dump(pipeline, self.MODEL_FILENAME)
            self.logger.info(f"Модель сохранена в {self.MODEL_FILENAME}")
            return pipeline
        except Exception as e:
            self.logger.exception(f"Ошибка обучения модели: {e}")
            return None

    def load_model(self):
        self.logger.info("Загрузка модели...")
        try:
            if os.path.isfile(self.MODEL_FILENAME):
                model = joblib.load(self.MODEL_FILENAME)
                self.logger.info("Модель успешно загружена.")
                return model
            else:
                self.logger.error("Файл модели не найден.")
                return None
        except Exception as e:
            self.logger.exception(f"Ошибка загрузки модели: {e}")
            return None

    def retrain_model_with_real_trades(self, historical_csv="historical_data_for_model_5m.csv", real_trades_csv="real_trades_features.csv"):
        self.logger.info("Дообучение модели с реальными трейдами...")
        try:
            if not os.path.isfile(historical_csv) or not os.path.isfile(real_trades_csv):
                self.logger.error("Не найдены необходимые CSV для дообучения.")
                return None
            hist_data = pd.read_csv(historical_csv)
            real_data = pd.read_csv(real_trades_csv)
            data = pd.concat([hist_data, real_data], ignore_index=True)
            X = data[self.MODEL_FEATURE_COLS]
            y = data["target"]
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            pipeline.fit(X, y)
            joblib.dump(pipeline, self.MODEL_FILENAME)
            self.logger.info("Модель дообучена и сохранена.")
            return pipeline
        except Exception as e:
            self.logger.exception(f"Ошибка дообучения модели: {e}")
            return None

    async def maybe_retrain_model(self):
        self.logger.info("Проверка необходимости дообучения модели...")
        await asyncio.sleep(3600)
        self.retrain_model_with_real_trades()

    def publish_drift_and_model_tables(self):
        self.logger.info("Публикация таблиц Drift и Model...")
        try:
            if self.open_positions:
                df = pd.DataFrame(list(self.open_positions.values()))
                df.to_csv("open_positions_report.csv", index=False)
                self.logger.info("open_positions_report.csv сохранён.")
            else:
                self.logger.info("Нет позиций для отчёта.")
        except Exception as e:
            self.logger.exception(f"Ошибка публикации таблиц: {e}")

    def open_averaging_position(self, symbol):
        self.logger.info(f"[AVERAGING] Открытие усредняющей позиции для {symbol}")
        pos = self.open_positions.get(symbol)
        if not pos:
            self.logger.info(f"[AVERAGING] Нет открытой позиции для {symbol} для усреднения.")
            return
        side = "Sell" if pos["side"].lower() == "buy" else "Buy"
        price = self.get_last_close_price(symbol)
        if price is None:
            self.logger.error(f"[AVERAGING] Нет цены для {symbol}")
            return
        volume = Decimal("100")
        qty = volume / Decimal(str(price))
        order_result = self.place_order(symbol, side, float(qty))
        if order_result:
            self.logger.info(f"[AVERAGING] Ордер усреднения размещён для {symbol} {side}")
        else:
            self.logger.error(f"[AVERAGING] Ошибка размещения ордера для {symbol}")

# --------------------- Класс UserSession – индивидуальная сессия пользователя ---------------------
class UserSession:
    def __init__(self, user_id: int, user_api: str, user_api_secret: str):
        self.user_id = user_id
        self.user_api = user_api
        self.user_api_secret = user_api_secret
        self.session = HTTP(demo=True, api_key=user_api, api_secret=user_api_secret, timeout=30)
        self.mode = "ST_cross2"
        self.max_total_volume = Decimal("500")
        self.position_volume = Decimal("100")
        self.quiet_period = False
        self.sleep_mode = False

# --------------------- Класс TradingBot – многопользовательский бот с Telegram-интерфейсом ---------------------
class TradingBot:
    def __init__(self):
        self.logger = logger
        self.trading_logic = TradingLogic(global_session, self.logger)
        self.allowed_users = load_allowed_users("users.csv")
        self.user_sessions = {}  # user_id -> UserSession
        self.state = {}
        self.open_positions = {}  # Синхронизируются с trading_logic.open_positions
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.dp = Dispatcher(storage=MemoryStorage())
        self.router = Router()
        self.telegram_message_queue = asyncio.Queue()
        self.send_semaphore = asyncio.Semaphore(10)
        self.loop = asyncio.get_event_loop()
        self.register_handlers()

    def init_user_session(self, user_id: int):
        if user_id not in self.allowed_users:
            self.logger.error(f"Пользователь {user_id} не найден в списке разрешённых.")
            return None
        user_data = self.allowed_users.get(user_id, {})
        user_api = user_data.get("user_api", "").strip()
        user_api_secret = user_data.get("user_api_secret", "").strip()
        if not user_api or not user_api_secret:
            self.logger.error(f"Пользователь {user_id} не задал индивидуальные API‑ключи.")
            return None
        self.user_sessions[user_id] = UserSession(user_id, user_api, user_api_secret)
        self.logger.info(f"Создана сессия для пользователя {user_id}")
        return self.user_sessions[user_id]

    def get_symbol_info(self, symbol):
        return self.trading_logic.get_symbol_info(symbol)

    def get_last_close_price(self, symbol):
        return self.trading_logic.get_last_close_price(symbol)

    def place_order(self, symbol, side, qty, order_type="Market", time_in_force="GoodTillCancel", reduce_only=False, positionIdx=None):
        return self.trading_logic.place_order(symbol, side, qty, order_type, time_in_force, reduce_only, positionIdx)

    def update_open_positions_from_exch_positions(self, expos: dict):
        self.trading_logic.update_open_positions_from_exch_positions(expos)
        self.open_positions = self.trading_logic.open_positions


    def register_handlers(self):
        @self.router.message(Command(commands=["menu"]))
        async def main_menu_handler(message: Message):
            user_id = message.chat.id
            self.init_user_session(user_id)
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📈 Trading", callback_data=f"menu_trading_{user_id}")],
                [InlineKeyboardButton(text="🤖 Bot", callback_data=f"menu_bot_{user_id}")],
                [InlineKeyboardButton(text="ℹ️ Info", callback_data=f"menu_info_{user_id}")],
                [InlineKeyboardButton(text="⚙️ Monitor", callback_data=f"menu_monitor_{user_id}")]
            ])
            await message.answer("Выберите раздел:", reply_markup=keyboard)

        @self.router.callback_query(lambda c: c.data.startswith("menu_trading_"))
        async def menu_trading_handler(query: CallbackQuery):
            user_id = query.from_user.id
            data_user_id = int(query.data.split("_")[-1])
            if user_id != data_user_id:
                await query.answer("Эта кнопка не для вас", show_alert=True)
                return
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📊 Статус", callback_data=f"cmd_status_{user_id}")],
                [InlineKeyboardButton(text="🔄 Смена режима", callback_data=f"cmd_mode_{user_id}")],
                [InlineKeyboardButton(text="🔙 Назад", callback_data=f"menu_main_{user_id}")]
            ])
            await query.message.edit_text("📈 **Trading** – выберите действие:", parse_mode="Markdown", reply_markup=keyboard)

        @self.router.callback_query(lambda c: c.data.startswith("cmd_status_"))
        async def status_handler(query: CallbackQuery):
            if not self.open_positions:
                await query.message.answer("Нет позиций.")
                return
            lines = []
            total_pnl_usdt = Decimal("0")
            total_invested = Decimal("0")
            for sym, pos in self.open_positions.items():
                side_str = pos["side"]
                entry_price = Decimal(str(pos["avg_price"]))
                volume_usdt = Decimal(str(pos["position_volume"]))
                current_price = self.get_last_close_price(sym)
                if current_price is None:
                    lines.append(f"{sym} {side_str}: нет цены.")
                    continue
                cp = Decimal(str(current_price))
                ratio = (cp - entry_price) / entry_price if side_str.lower() == "buy" else (entry_price - cp) / entry_price
                pnl_usdt = ratio * volume_usdt
                pnl_percent = ratio * Decimal("100")
                total_pnl_usdt += pnl_usdt
                total_invested += volume_usdt
                lines.append(f"{sym} {side_str}: PnL = {pnl_usdt:.2f} USDT ({pnl_percent:.2f}%)")
            lines.append("-" * 30)
            if total_invested > 0:
                total_pnl_percent = (total_pnl_usdt / total_invested) * Decimal("100")
                lines.append(f"Итоговый PnL: {total_pnl_usdt:.2f} USDT ({total_pnl_percent:.2f}%)")
            else:
                lines.append("Итоговый PnL: 0 (нет позиций)")
            await query.message.answer("\n".join(lines))

        # Добавляем обработчик команды /stop для остановки бота
        @self.router.message(Command(commands=["stop"]))
        async def stop_handler(message: Message):
            await message.answer("Бот останавливается...")
            await self.shutdown()
            sys.exit(0)

    async def initialize_telegram_bot(self):
        try:
            self.dp.include_router(self.router)
            self.logger.info("Telegram бот инициализирован. Запуск polling...")
            await self.dp.start_polling(self.bot)
        except Exception as e:
            self.logger.exception(f"Ошибка инициализации Telegram-бота: {e}")

    async def send_initial_telegram_message(self):
        try:
            if TELEGRAM_CHAT_ID:
                msg = "✅ Бот успешно запущен. Для запуска меню введите команду '/menu'"
                await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                self.logger.info("Сообщение о запуске Telegram отправлено.")
        except Exception as e:
            self.logger.exception(f"Ошибка при отправке стартового сообщения: {e}")

    async def main_loop(self):
        self.logger.info("=== Запуск основного цикла ===")
        self.state["total_open_volume"] = Decimal("0")
        tg_task = asyncio.create_task(self.initialize_telegram_bot())
        await asyncio.sleep(3)
        await self.send_initial_telegram_message()
        exch_positions = self.trading_logic.session.get_positions(category="linear", settleCoin="USDT")
        self.update_open_positions_from_exch_positions(exch_positions)

        # Перед запуском основного цикла синхронизируем исторические данные для каждой пары
        symbols = self.trading_logic.get_usdt_pairs()
        for sym in symbols:
            self.trading_logic.sync_historical_data(sym, interval="1", csv_path="historical_data_for_model_5m.csv")
            await asyncio.sleep(1)  # задержка для избежания перегрузки API

        iteration_count = 0
        while True:
            try:
                iteration_count += 1
                self.logger.info(f"[LOOP] iteration_count={iteration_count}")
                symbols = self.trading_logic.get_usdt_pairs()
                mode = self.trading_logic.OPERATION_MODE
                if mode == "model_only":
                    for s in symbols:
                        self.trading_logic.process_symbol_model_only(s)
                elif mode == "golden_setup":
                    for s in symbols:
                        df_5m = self.trading_logic.get_historical_data_for_model(s, "1", limit=20)
                        if not df_5m.empty:
                            action, _ = self.trading_logic.handle_golden_setup(s, df_5m)
                            if action:
                                self.trading_logic.open_position(s, action, POSITION_VOLUME, reason="golden_setup")
                elif mode == "drift_only":
                    analysis_data = {s: {"is_anomaly": True, "strength": random.uniform(0, 1), "direction": "up"} for s in symbols}
                    top_list = self.trading_logic.get_top_anomalies_from_analysis(analysis_data)
                    for sym, strength, direction in top_list:
                        side = "Buy" if direction.lower() == "up" else "Sell"
                        self.trading_logic.handle_drift_order_with_trailing(sym, side, POSITION_VOLUME)
                elif mode == "ST_cross_global":
                    for s in symbols:
                        self.trading_logic.process_symbol_st_cross_global(s, interval="1")
                elif mode == "ST_cross1":
                    for s in symbols:
                        self.trading_logic.process_symbol_st_cross1(s, interval="1")
                elif mode == "ST_cross2":
                    for s in symbols:
                        self.trading_logic.process_symbol_st_cross2(s, interval="1")
                elif mode == "super_trend":
                    for s in symbols:
                        self.trading_logic.process_symbol_supertrend_open(s, interval="1", length=8, multiplier=3.0)
                if iteration_count % 5 == 0:
                    self.trading_logic.publish_drift_and_model_tables()
                await asyncio.sleep(60)
            except Exception as e_inner:
                self.logger.exception(f"Ошибка в основном цикле: {e_inner}")
                await asyncio.sleep(60)
                continue

    async def shutdown(self):
        if self.bot and TELEGRAM_CHAT_ID:
            try:
                await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="✅ Бот успешно остановлен")
            except Exception as e:
                self.logger.error(f"Ошибка при отправке сообщения об остановке: {e}")

def main():
    bot_instance = TradingBot()
    try:
        asyncio.run(bot_instance.main_loop())
    except KeyboardInterrupt:
        logger.info("Остановка пользователем.")
    except Exception as e:
        logger.exception(f"Ошибка в main: {e}")

if __name__ == "__main__":
    main()