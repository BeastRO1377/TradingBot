#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiuserBot_V12.py – обновлённый торговый бот на Bybit с расчётом индикаторов через WebSocket и оптимизированной логикой стоп-лоссов.
"""

# Импорты системные и сторонние
import asyncio
from asyncio import run_coroutine_threadsafe
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
import datetime as dt
from datetime import datetime, timedelta, timezone
import pathlib
import pandas as pd
import numpy as np
import pandas_ta as ta
from decimal import Decimal, ROUND_HALF_UP, DivisionByZero, InvalidOperation, getcontext
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from urllib3.util.retry import Retry
import aiohttp


import lightgbm as lgb
try:
    import joblib  # used in a few legacy helper functions
except ImportError:
    joblib = None

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
import websocket

import functools
from typing import Optional, Tuple, Dict
import threading
from concurrent.futures import ThreadPoolExecutor

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Заготовка класса TradingBot с будущей реализацией
class TradingBot:
    def __init__(self, user_id, api_key, api_secret, mode, monitoring):
        self.user_id = user_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.mode = mode.lower()
        self.monitoring = monitoring.lower()
        self.session = HTTP(demo=(self.mode == "demo"), api_key=api_key, api_secret=api_secret)
        self.ws_public = None
        self.ws_private = None
        self.open_positions = {}
        self.candles_data = defaultdict(list)
        self.recent_closes = {}
        self.loop = asyncio.get_event_loop()

    async def start(self):
        logger.info(f"[Bot] Запуск бота для user_id={self.user_id}, режим: {self.mode}, мониторинг: {self.monitoring}")
        await self.setup_websockets()

    async def setup_websockets(self):
        # Основные подписки на WebSocket будем реализовывать здесь
        pass

    async def handle_position_update(self, message):
        # Здесь будет логика обновления по unrealisedPnl и установка трейлинг-стопа
        pass

    async def handle_kline(self, message):
        # Здесь будет логика обработки свечей и расчёта индикаторов
        pass

    async def place_order(self, symbol, side, qty, reason=""):
        # Логика отправки ордера через WebSocket или REST в зависимости от monitoring/demo
        pass

# Пример инициализации и запуска (будет использоваться в main)
async def main():
    user_id = 123456
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    mode = "live"
    monitoring = "websocket"

    bot = TradingBot(user_id, api_key, api_secret, mode, monitoring)
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())



    async def setup_websockets(self):
        def _on_message(msg):
            asyncio.run_coroutine_threadsafe(self.route_ws_message(msg), self.loop)

        # Инициализация публичного WebSocket для свечей
        self.ws_public = WebSocket(
            testnet=False,
            channel_type="linear",
            ping_interval=20,
            ping_timeout=10,
        )
        self.ws_public.kline_stream(interval="1", symbol=["BTCUSDT", "ETHUSDT"], callback=_on_message)

        # Инициализация приватного WebSocket для позиций
        self.ws_private = WebSocket(
            testnet=False,
            demo=self.mode == "demo",
            channel_type="private",
            api_key=self.api_key,
            api_secret=self.api_secret,
            ping_interval=20,
            ping_timeout=10,
        )
        self.ws_private.position_stream(callback=_on_message)

    async def route_ws_message(self, message):
        topic = message.get("topic", "")
        if "kline." in topic:
            await self.handle_kline(message)
        elif "position" in topic:
            await self.handle_position_update(message)

    async def handle_kline(self, message):
        symbol = message["topic"].split(".")[-1]
        data = message["data"]
        if not data.get("confirm", False):
            return

        candle = {
            "open": float(data["open"]),
            "high": float(data["high"]),
            "low": float(data["low"]),
            "close": float(data["close"]),
            "volume": float(data["volume"]),
            "startTime": int(data["start"]),
        }

        # Обновляем локальные свечи
        df = self.candles_data[symbol]
        ts = pd.to_datetime(candle["startTime"], unit="ms")
        row = {
            "startTime": ts,
            "openPrice": candle["open"],
            "highPrice": candle["high"],
            "lowPrice": candle["low"],
            "closePrice": candle["close"],
            "volume": candle["volume"],
        }

        if df and not df.empty and df.iloc[-1]["startTime"] == ts:
            self.candles_data[symbol].iloc[-1] = row
        else:
            self.candles_data[symbol].append(row)
            if len(self.candles_data[symbol]) > 500:
                self.candles_data[symbol] = self.candles_data[symbol][-500:]

        logger.info(f"[{symbol}] Обновлена свеча: {row}")

        # Расчёт индикаторов и сигналов можно вставить здесь

    async def handle_position_update(self, message):
        data = message.get("data", {})
        if isinstance(data, list):
            for position in data:
                await self.process_position(position)
        else:
            await self.process_position(data)

    async def process_position(self, position):
        symbol = position["symbol"]
        size = float(position["size"])
        entry_price = float(position.get("entryPrice", 0))
        mark_price = float(position.get("markPrice", 0))
        pnl = float(position.get("unrealisedPnl", 0))
        if size == 0:
            self.open_positions.pop(symbol, None)
            logger.info(f"[{symbol}] Позиция закрыта")
            return

        self.open_positions[symbol] = position
        logger.info(f"[{symbol}] Обновлена позиция. PnL: {pnl}")

        # Расчёт условий стоп-лосса можно вставить здесь



    async def place_order(self, symbol, side, qty, reason=""):
        order_type = "Market"
        position_idx = 0  # assume One-Way mode (can be changed to 1/2 if Hedge mode needed)
        logger.info(f"[place_order] Попытка открыть ордер {side} {symbol} qty={qty} по {self.monitoring.upper()}")

        order_data = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "positionIdx": position_idx,
            "timeInForce": "Market"
        }

        if self.monitoring == "http" or self.mode == "demo":
            logger.info(f"[place_order] Отправка через REST")
            try:
                result = await asyncio.to_thread(lambda: self.session.place_order(**order_data))
                logger.info(f"[place_order][REST] Ответ: {result}")
                return result
            except Exception as e:
                logger.error(f"[place_order][REST] Ошибка: {e}")
        else:
            logger.info(f"[place_order] Отправка через WebSocket")
            try:
                if not self.ws_private:
                    raise RuntimeError("WebSocket не инициализирован")

                await self.ws_private.send_cmd("order.place", order_data)
                logger.info(f"[place_order][WS] Ордер отправлен через WebSocket")
            except Exception as e:
                logger.error(f"[place_order][WS] Ошибка отправки через WebSocket: {e}")



    async def manage_trailing_stop(self, symbol, position, pnl_pct, side):
        try:
            base_threshold = Decimal("5.0")  # базовый порог прибыли (%)
            base_step = Decimal("2.5")       # базовый шаг перемещения стопа (%)
            step_decrease = Decimal("0.5")   # уменьшение шага при выполнении условий

            # Определяем динамический шаг
            dynamic_step = base_step

            # Учитываем увеличение объема торгов по паре
            vol_hist = self.volume_history.get(symbol, [])
            if len(vol_hist) >= 2:
                last_volume = vol_hist[-1]
                prev_volume = vol_hist[-2]
                if last_volume - prev_volume >= 1000:
                    dynamic_step -= step_decrease

            # Учитываем рост цены более чем на 1% от входа
            entry_price = Decimal(position.get("entry_price", 0))
            mark_price = Decimal(position.get("mark_price", 0))
            if entry_price and mark_price:
                price_change = abs(mark_price - entry_price) / entry_price * Decimal("100")
                if price_change >= 1:
                    dynamic_step -= step_decrease

            # Гарантируем, что шаг не станет слишком маленьким
            dynamic_step = max(dynamic_step, Decimal("0.5"))

            # Если PnL выше порога — ставим/двигаем стоп
            if pnl_pct >= base_threshold:
                logger.info(f"[TrailingStop] {symbol}: PnL={pnl_pct}%, динамический шаг={dynamic_step}%")
                stop_pct = pnl_pct - dynamic_step
                stop_price = None
                if side == "Buy":
                    stop_price = entry_price * (Decimal("1") + stop_pct / Decimal("100"))
                else:
                    stop_price = entry_price * (Decimal("1") - stop_pct / Decimal("100"))

                if stop_price:
                    await self.set_fixed_stop_loss(symbol, position['size'], side, float(stop_price))
        except Exception as e:
            logger.error(f"[TrailingStop] Ошибка установки стопа для {symbol}: {e}")



    async def process_position(self, position):
        symbol = position["symbol"]
        size = float(position["size"])
        entry_price = float(position.get("entryPrice", 0))
        mark_price = float(position.get("markPrice", 0))
        pnl = float(position.get("unrealisedPnl", 0))

        if size == 0:
            self.open_positions.pop(symbol, None)
            logger.info(f"[{symbol}] Позиция закрыта")
            return

        self.open_positions[symbol] = position
        logger.info(f"[{symbol}] Обновлена позиция. PnL: {pnl:.2f} USDT")

        # Расчёт прибыли в процентах
        if entry_price > 0:
            direction = 1 if position["side"] == "Buy" else -1
            pnl_pct = direction * ((mark_price - entry_price) / entry_price) * 100

            # Начальное условие на установку трейлинг-стопа
            if pnl_pct >= 5:
                base_trailing = 2.5

                # Модификаторы для объема и цены
                volume = float(position.get("cumRealisedPnl", 0))
                open_interest = self.latest_open_interest.get(symbol, 0)
                price_change_pct = 0

                if symbol in self.latest_mark_prices and entry_price > 0:
                    price_change_pct = abs((self.latest_mark_prices[symbol] - entry_price) / entry_price * 100)

                # Модификация trailing gap
                step_reduction = 0
                if open_interest > 1000:
                    step_reduction += 0.5
                if price_change_pct > 1:
                    step_reduction += 0.5

                final_trailing = max(base_trailing - step_reduction, 0.5)
                stop_pct = pnl_pct - final_trailing

                # Вычисляем стоп-цену
                if position["side"] == "Buy":
                    stop_price = entry_price * (1 + stop_pct / 100)
                else:
                    stop_price = entry_price * (1 - stop_pct / 100)

                logger.info(f"[{symbol}] Установка трейлинг-стопа: {stop_price:.4f} при PnL={pnl_pct:.2f}% (шаг={final_trailing:.2f}%)")
                await self.set_trailing_stop(symbol, stop_price, position["side"])

    async def set_trailing_stop(self, symbol, stop_price, side):
        # REST вызов на установку стопа через set_trading_stop
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "stopLoss": str(stop_price),
            "timeInForce": "GoodTillCancel",
        }

        try:
            result = await asyncio.to_thread(lambda: self.session.set_trading_stop(**params))
            if result.get("retCode") == 0:
                logger.info(f"[set_trailing_stop] Стоп успешно установлен для {symbol} на уровне {stop_price}")
            else:
                logger.warning(f"[set_trailing_stop] Ошибка от Bybit: {result}")
        except Exception as e:
            logger.error(f"[set_trailing_stop] Ошибка установки стопа: {e}")



    async def handle_kline(self, message):
        import pandas as pd
        import numpy as np
        import pandas_ta as ta

        symbol = message["topic"].split(".")[-1]
        data = message["data"]
        if not data.get("confirm", False):
            return

        candle = {
            "open": float(data["open"]),
            "high": float(data["high"]),
            "low": float(data["low"]),
            "close": float(data["close"]),
            "volume": float(data["volume"]),
            "startTime": int(data["start"]),
        }

        ts = pd.to_datetime(candle["startTime"], unit="ms")
        row = {
            "startTime": ts,
            "openPrice": candle["open"],
            "highPrice": candle["high"],
            "lowPrice": candle["low"],
            "closePrice": candle["close"],
            "volume": candle["volume"],
        }

        df = self.candles_data[symbol]
        if df and not df.empty and df[-1]["startTime"] == ts:
            self.candles_data[symbol][-1] = row
        else:
            self.candles_data[symbol].append(row)
            if len(self.candles_data[symbol]) > 500:
                self.candles_data[symbol] = self.candles_data[symbol][-500:]

        # Преобразуем в DataFrame
        df = pd.DataFrame(self.candles_data[symbol])
        df.set_index("startTime", inplace=True)

        # Рассчёт индикаторов
        df["macd"], df["macd_signal"], df["macd_hist"] = ta.macd(df["closePrice"])
        df["rsi"] = ta.rsi(df["closePrice"], length=14)

        # Простой сигнал: RSI пересек 70 сверху или 30 снизу
        signal = None
        if df["rsi"].iloc[-2] < 70 and df["rsi"].iloc[-1] >= 70:
            signal = "Sell"
        elif df["rsi"].iloc[-2] > 30 and df["rsi"].iloc[-1] <= 30:
            signal = "Buy"

        if signal:
            logger.info(f"[{symbol}] 📣 RSI сигнал: {signal}")
            await self.send_telegram(f"📣 [{symbol}] RSI сигнал: {signal}")
            await self.place_order(symbol, signal, qty=0.01, reason="RSI_signal")

    async def send_telegram(self, message):
        # Здесь должна быть логика отправки в Telegram
        logger.info(f"[Telegram] {message}")



    async def setup_websockets(self):
        def _on_message(msg):
            asyncio.run_coroutine_threadsafe(self.route_ws_message(msg), self.loop)

        # WebSocket подключения
        self.ws_public = WebSocket(
            testnet=(self.mode == "demo"),
            channel_type="linear",
            ping_interval=20,
            ping_timeout=10,
        )
        symbols = ["BTCUSDT", "ETHUSDT"]

        self.ws_public.kline_stream(interval="1", symbol=symbols, callback=_on_message)
        self.ws_public.ticker_stream(symbol=symbols, callback=_on_message)

        self.ws_private = WebSocket(
            testnet=(self.mode == "demo"),
            channel_type="private",
            api_key=self.api_key,
            api_secret=self.api_secret,
            ping_interval=20,
            ping_timeout=10,
        )
        self.ws_private.position_stream(callback=_on_message)

    async def route_ws_message(self, message):
        topic = message.get("topic", "")
        if "kline." in topic:
            await self.handle_kline(message)
        elif "position" in topic:
            await self.handle_position_update(message)
        elif "tickers" in topic:
            await self.handle_ticker_update(message)

    async def handle_ticker_update(self, message):
        data = message.get("data", {})
        if not isinstance(data, dict):
            return

        symbol = data.get("symbol")
        if not symbol:
            return

        try:
            open_interest = float(data.get("openInterest", 0))
            self.latest_open_interest[symbol] = open_interest
            logger.debug(f"[Ticker] {symbol} openInterest обновлён: {open_interest}")
        except Exception as e:
            logger.error(f"[handle_ticker_update] Ошибка разбора openInterest: {e}")



    async def execute_golden_setup_websocket(self, symbol):
        try:
            df = pd.DataFrame(self.candles_data.get(symbol, []))
            if df.empty or len(df) < 10:
                return

            # Преобразуем в DataFrame, если это не DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            # Параметры стратегии
            params = {
                "price_change": 0.2,
                "volume_change": 3500,
                "oi_change": 0.4,
                "period_iters": 4,
            }

            # Последние свечи
            recent = df.tail(int(params["period_iters"]) + 1)
            if len(recent) < int(params["period_iters"]) + 1:
                return

            close_now = recent["closePrice"].iloc[-1]
            close_prev = recent["closePrice"].iloc[0]
            price_delta_pct = (close_now - close_prev) / close_prev * 100

            volume_now = recent["volume"].iloc[-1]
            volume_avg = recent["volume"].iloc[:-1].mean()
            volume_ratio = (volume_now / volume_avg) if volume_avg > 0 else 0

            oi_now = self.latest_open_interest.get(symbol, 0)
            oi_hist = recent["volume"].mean()  # или можно кэшировать историю OI
            oi_delta_pct = (oi_now - oi_hist) / oi_hist * 100 if oi_hist else 0

            logger.info(f"[Golden] {symbol}: ΔP={price_delta_pct:.2f}%, vol_ratio={volume_ratio:.2f}, ΔOI={oi_delta_pct:.2f}%")

            if price_delta_pct > params["price_change"] and                volume_ratio > (params["volume_change"] / 1000) and                oi_delta_pct > params["oi_change"]:

                logger.info(f"[Golden] ⚡ GOLDEN SETUP {symbol} - условия выполнены")
                await self.send_telegram(f"⚡ GOLDEN SETUP {symbol} — Цена +{price_delta_pct:.2f}%, Объём x{volume_ratio:.1f}, ΔOI {oi_delta_pct:.2f}%")
                await self.place_order(symbol, "Buy", qty=0.01, reason="GoldenSetup")

        except Exception as e:
            logger.error(f"[Golden Setup] {symbol}: ошибка выполнения golden setup: {e}")



    async def execute_golden_setup_websocket(self, symbol):
        try:
            df = pd.DataFrame(self.candles_data.get(symbol, []))
            if df.empty or len(df) < 10:
                return

            # Преобразуем в DataFrame, если это не DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            # Параметры стратегии (можно будет вынести в users.csv позже)
            buy_params = {
                "price_change": 0.2,
                "volume_change": 3500,
                "oi_change": 0.4,
                "period_iters": 4,
            }

            sell_params = {
                "price_change": -0.5,
                "volume_change": 1300,
                "oi_change": 0.8,
                "period_iters": 4,
            }

            recent = df.tail(int(buy_params["period_iters"]) + 1)
            if len(recent) < int(buy_params["period_iters"]) + 1:
                return

            close_now = recent["closePrice"].iloc[-1]
            close_prev = recent["closePrice"].iloc[0]
            price_delta_pct = (close_now - close_prev) / close_prev * 100

            volume_now = recent["volume"].iloc[-1]
            volume_avg = recent["volume"].iloc[:-1].mean()
            volume_ratio = (volume_now / volume_avg) if volume_avg > 0 else 0

            oi_now = self.latest_open_interest.get(symbol, 0)
            oi_hist = recent["volume"].mean()
            oi_delta_pct = (oi_now - oi_hist) / oi_hist * 100 if oi_hist else 0

            logger.info(f"[Golden] {symbol}: ΔP={price_delta_pct:.2f}%, vol_ratio={volume_ratio:.2f}, ΔOI={oi_delta_pct:.2f}%")

            if price_delta_pct > buy_params["price_change"] and volume_ratio > (buy_params["volume_change"] / 1000) and oi_delta_pct > buy_params["oi_change"]:

                logger.info(f"[Golden] ⚡ BUY SETUP {symbol} - условия выполнены")
                await self.send_telegram(f"⚡ GOLDEN BUY SETUP {symbol} — Цена +{price_delta_pct:.2f}%, Объём x{volume_ratio:.1f}, ΔOI {oi_delta_pct:.2f}%")
                await self.place_order(symbol, "Buy", qty=0.01, reason="GoldenSetupBuy")

            elif price_delta_pct < sell_params["price_change"] and volume_ratio > (sell_params["volume_change"] / 1000) and                  oi_delta_pct > sell_params["oi_change"]:

                logger.info(f"[Golden] ⚠️ SELL SETUP {symbol} - условия выполнены")
                await self.send_telegram(f"⚠️ GOLDEN SELL SETUP {symbol} — Цена {price_delta_pct:.2f}%, Объём x{volume_ratio:.1f}, ΔOI {oi_delta_pct:.2f}%")
                await self.place_order(symbol, "Sell", qty=0.01, reason="GoldenSetupSell")

        except Exception as e:
            logger.error(f"[Golden Setup] {symbol}: ошибка выполнения golden setup: {e}")
