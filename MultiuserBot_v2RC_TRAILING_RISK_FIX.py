#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence

from mimetypes import init
import os, sys, faulthandler

from networkx import sigma
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
faulthandler.enable(file=sys.stderr, all_threads=True)
import signal, functools, asyncio as _aio

import datetime as dt
import aiogram
from aiogram.enums import ParseMode
import google.generativeai as genai

from safetensors.numpy import save_file as save_safetensors, load_file as load_safetensors


# Ensure required imports
import asyncio
import pytz
from datetime import timedelta
from pathlib import Path
import joblib
import ccxt
from sklearn.preprocessing import StandardScaler
from requests.adapters import HTTPAdapter

import coremltools as ct
import re
import tempfile

# ADD: Import InvalidRequestError for advanced order error handling
from pybit.exceptions import InvalidRequestError

# ---------------------- IMPORTS ----------------------
import csv
import json
import logging
import time
import hmac
from sympy import N
import websockets
import hashlib
import requests
from requests.exceptions import ReadTimeout as RequestsReadTimeout, ConnectionError as RequestsConnectionError
from urllib3.exceptions import ReadTimeoutError as UrllibReadTimeoutError

# --- numpy ≥2.0 compatibility shim for pandas_ta ------------------------
import pandas as pd
# --- numpy ≥2.0 compatibility shim for pandas_ta ------------------------
import numpy as _np
if not hasattr(_np, "NaN"):   # NumPy 2.0 dropped the alias
    _np.NaN = _np.nan         # restore alias expected by pandas_ta
# -----------------------------------------------------------------------
# --- numpy ≥2.0 compatibility shim for pandas_ta ------------------------
import numpy as np
if not hasattr(np, "NaN"):   # newer numpy removed the alias
    np.NaN = np.nan         # restore alias expected by pandas_ta
# -----------------------------------------------------------------------
import pandas_ta as ta

from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timezone
from pybit.unified_trading import WebSocket, HTTP
from telegram_fsm_v12 import dp, router, router_admin
from telegram_fsm_v12 import bot as telegram_bot
from collections import deque
import pickle
import math
import random
import signal
from decimal import Decimal, InvalidOperation

from aiolimiter import AsyncLimiter
from aiogram import types
from aiogram.filters import Command
import lightgbm as lgb
from itertools import product

import telegram_fsm_v12 as tg_fsm

from lightgbm import Booster

import mlx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim

from concurrent.futures import ThreadPoolExecutor
from websockets.exceptions import ConnectionClosed
import pickle

from pathlib import Path
import httpx
import uuid
import textwrap
from openai import AsyncOpenAI
import uvloop
uvloop.install()

# ── safe loader for StandardScaler (скейлер может быть сохранён в другой версии sklearn) ──
import warnings
from sklearn.exceptions import InconsistentVersionWarning

def _safe_load_scaler(path: str = "scaler.pkl"):
    """
    Пытается загрузить pickled-StandardScaler, созданный в любой версии sklearn.
    Если pickle несовместим с текущей (≤1.5.1) – возвращает новый «пустой» scaler,
    чтобы скрипт не падал и не ловил segfault.
    """

    if not os.path.exists(path):
        return StandardScaler()          # файла нет → пустой скейлер

    with warnings.catch_warnings(record=True):
        # Превращаем предупреждение о несовместимости в исключение
        warnings.filterwarnings("error", category=InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except (InconsistentVersionWarning, Exception):
            print(f"[Scaler] {path} несовместим с текущей sklearn – создаю новый StandardScaler()")
            return StandardScaler()

# В НАЧАЛЕ ФАЙЛА, ПОСЛЕ ИМПОРТОВ

def log_for_finetune(prompt: str, pnl_pct: float, source: str):
    """Записывает данные, необходимые для дообучения модели."""
    log_file = Path("finetune_log.csv")
    is_new = not log_file.exists()
    try:
        with log_file.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            if is_new:
                writer.writerow(["timestamp", "source", "pnl_pct", "prompt"])
            writer.writerow([datetime.utcnow().isoformat(), source, pnl_pct, prompt])
    except Exception as e:
        logger.error(f"[FineTuneLog] Ошибка записи лога для дообучения: {e}")

 # Telegram‑ID(-ы) администраторов, которым доступна команда /snapshot
ADMIN_IDS = {36972091}   # ← замените на свой реальный ID

# Глобальный реестр всех экземпляров TradingBot (используется для snapshot)
GLOBAL_BOTS: list = []

# ── ML configuration ─────────────────────────────────────────────
MODEL_PATH         = "golden_model_v19.pt"   # актуальный чек-пойнт
ML_PROB_THRESHOLD  = 0.55                    # min(model-confidence) для вмешательства

logger = logging.getLogger(__name__)

BOT_DEVICE = "cpu" # MLX работает на CPU/GPU, PyTorch-специфичная логика убрана

logger.info("[ML] Using compute device: %s", BOT_DEVICE)

# Configure rotating log file: 10 MB per file, keep 5 backups
rotating_handler = RotatingFileHandler('bot.log', maxBytes=50*1024*1024, backupCount=5)
rotating_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(rotating_handler)
# Также выводить логи в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

root_logger.addHandler(console_handler)
# ── wallet state logger ───────────────────────────────────────────────
wallet_logger = logging.getLogger("wallet_state")
wallet_handler = RotatingFileHandler(
    "wallet_state.log", maxBytes=20 * 1024 * 1024, backupCount=3
)
wallet_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
wallet_logger.setLevel(logging.INFO)
wallet_logger.addHandler(wallet_handler)

SNAPSHOT_CSV_PATH = "golden_setup_snapshots.csv"
DEC_TICK = 0.000001  # float tick
SQUEEZE_REPRICE_INTERVAL = 0.2      # сек между перестановками
# фиксированный отступ лимитника для Сквиза
SQUEEZE_LIMIT_OFFSET_PCT = 0.005   # 0.5 %

SQUEEZE_COOLDOWN_SEC = 60      # 1 минута

# Where per‑symbol optimal liquidation thresholds are stored
LIQ_THRESHOLD_CSV_PATH = "liq_thresholds.csv"
# Centralised location of historical liquidation events
LIQUIDATIONS_CSV_PATH = "liquidations.csv"

TRADES_UNIFIED_CSV_PATH = "trades_unified.csv"
TRADES_UNIFIED_JSON_PATH = "trades_unified.json"

SQUEEZE_THRESHOLD_PCT = 4.0    # рост ≥ 4 % за 5 мин

# Минимум мощности сквиза (произведение % изменения цены на % изменения объёма)
DEFAULT_SQUEEZE_POWER_MIN = 8.0

AVERAGE_LOSS_TRIGGER = -160.0   # усредняем, если unrealised PnL ≤ −160%

GLOBAL_BOTS: list["TradingBot"] = []
tg_fsm.GLOBAL_BOTS = GLOBAL_BOTS

CACHE_TTL_SEC = 180

# --- dynamic-threshold & volatility coefficients (v3) ---
LARGE_TURNOVER = 100_000_000     # 100 M USDT 24h turnover
MID_TURNOVER   = 10_000_000      # 10 M USDT
VOL_COEF       = 1.2             # ≥ 1.2σ spike
VOL_WINDOW     = 60              # 12 × 5-мин свечей = 1 час
VOLUME_COEF    = 3.0             # объём ≥ 3× ср.30 мин
LIQ_CLUSTER_WINDOW_SEC  = 1800      # 30-мин кластерное окно
LIQ_CLUSTER_MIN_USDT    = 5_000.0   # игнорируем кластеры < 5 000 USDT
LIQ_PRICE_PROXIMITY_PCT = 1.0       # торгуем, если цена в ±1 % от кластера

LISTING_AGE_MIN_MINUTES = 1400    # игнорируем пары младше ~23 часов

# ── shared JSON paths for Telegram-FSM ───────────────────────────────────────
OPEN_POS_JSON   = "open_positions.json"
WALLET_JSON     = "wallet_state.json"
TRADES_JSON     = "trades_history.json"

# [NEW LOGIC] Risk Management Parameters
TRAILING_START_PNL_PCT = 5.0
TRAILING_ATR_MULTIPLIER = 2.5
PARTIAL_TAKE_PROFIT_LEVELS = {15.0: (0.3, "Take Profit 1 (30%)"), 30.0: (0.3, "Take Profit 2 (30%)")}
TIME_STOP_HOURS = 24
TIME_STOP_PNL_RANGE = (-3.0, 3.0)
ML_QUALITY_THRESHOLD = 0.6

LISTING_AGE_CACHE_TTL_SEC = 3600          # 1-hour cache
_listing_age_cache: dict[str, tuple[float, float]] = {}
_listing_sem = asyncio.Semaphore(5)       # max 5 concurrent REST calls

# --- REST throttling ---
_TRADING_STOP_SEM = asyncio.Semaphore(3)      # ≤3 одновременных запросов
_LAST_STOP: dict[str, float] = {}             # symbol -> последний поставленный стоп

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT"}

GOLDEN_WEIGHTS = {
    "price_change": 0.4,
    "volume_change": 0.2,
    "oi_change":     0.4,
}

# ───── ML snapshots ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
ML_BEST_ENTRIES_CSV_PATH = ROOT_DIR / "ml_best_entries.csv"

SQUEEZE_LIMIT_OFFSET_PCT = 0.005

# --- runtime safety thresholds (patch 2025-07-10) -----------------
ML_CONFIDENCE_FLOOR  = 0.65     # min(model-prob) for ML overrides
MAX_ACTIVE_SAME_SIDE = 1        # ≤1 averaging order in the same direction

# Global scaler placeholder; will be loaded in __main__
scaler: StandardScaler = None

def new_cid() -> str:
    """Короткий корреляционный id для цепочки логов одной сделки."""
    return uuid.uuid4().hex[:8]

def j(obj, maxlen=600):
    """Безопасный JSON для логов (усекает длинные структуры)."""
    import json as _json
    try:
        s = _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        return (s[:maxlen] + "…") if len(s) > maxlen else s
    except Exception:
        return str(obj)[:maxlen]

def log_state(bot, symbol: str) -> dict:
    """Снимок ключевых состояний бота для диагностики."""
    try:
        op = list(getattr(bot, "open_positions", {}).keys())
        pend = list(getattr(bot, "pending_orders", {}).keys())
        max_total = float(getattr(bot, "MAX_TOTAL_VOLUME", 0.0) or 0.0)
        total = 0.0
        for p in getattr(bot, "open_positions", {}).values():
            try:
                total += float(p.get("cost", 0.0))
            except Exception:
                pass
        for v in getattr(bot, "pending_orders", {}).values():
            try:
                total += float(v)
            except Exception:
                pass
        return {
            "sym": symbol,
            "mode": getattr(bot, "strategy_mode", ""),
            "open_cnt": len(op),
            "pending_cnt": len(pend),
            "has_open": symbol in op,
            "has_pending": symbol in pend,
            "max_total": max_total,
            "current_total": total,
        }
    except Exception:
        return {"sym": symbol, "state_err": True}

def _atomic_json_write(path: str, data) -> None:
    """Safely write JSON using temp-file + rename."""
    import json, os, tempfile
    dirname = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".tmp_", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _append_trades_unified(row: dict) -> None:
    """Записывает строку в trades_unified.csv, создавая файл при первом запуске."""
    file_exists = os.path.isfile(TRADES_UNIFIED_CSV_PATH)
    with open(TRADES_UNIFIED_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------------------------------

# --- helpers used by WS/REST position-refresh code ---
def _safe_avg_price(pos: dict) -> float:
    return safe_to_float(
        pos.get("avgPrice")
        or pos.get("entryPrice")
        or pos.get("avg_entry_price")
        or 0
    )

def _normalize_side(pos: dict) -> str:
    """Normalise Bybit side to "Buy" / "Sell"."""
    s = str(pos.get("side") or "").lower()
    return "Buy" if s == "buy" else "Sell"

# ---------------------- FLOAT HELPER ----------------------
def safe_to_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

# ───────── Trailing-stop defaults ─────────
DEFAULT_TRAILING_START_PCT = 5.0     # %-PnL, когда включать трейлинг
DEFAULT_TRAILING_GAP_PCT   = 2.5    # %-отступ стопа от цены

FEATURE_KEYS = [
    "price", "pct1m", "pct5m", "pct15m",
    "vol1m", "vol5m", "vol15m",
    "OI_now", "dOI1m", "dOI5m",
    "spread_pct", "sigma5m", "CVD1m", "CVD5m",
    "rsi14", "sma50", "ema20", "atr14", "bb_width",
    "supertrend", "cci20", "macd", "macd_signal",
    "avgVol30m", "avgOI30m", "deltaCVD30m",
    "GS_pct4m", "GS_vol4m", "GS_dOI4m", "GS_cvd4m",
    "GS_supertrend", "GS_cooldown",
    "SQ_pct1m", "SQ_pct5m", "SQ_vol1m", "SQ_vol5m", "SQ_dOI1m",
    "SQ_spread_pct", "SQ_sigma5m", "SQ_liq10s",
    "SQ_power", "SQ_strength",
    "SQ_cooldown",
    "LIQ_cluster_val10s", "LIQ_cluster_count10s", "LIQ_direction",
    "LIQ_pct1m", "LIQ_pct5m", "LIQ_vol1m", "LIQ_vol5m", "LIQ_dOI1m",
    "LIQ_spread_pct", "LIQ_sigma5m", "LIQ_golden_flag", "LIQ_squeeze_flag",
    "LIQ_cooldown",
    "hour_of_day", "day_of_week", "month_of_year", "adx14",
]
INPUT_DIM = len(FEATURE_KEYS)

SQUEEZE_KEYS = [
    "pct_5m", "vol_change_pct", "sigma5m",
    "d_oi", "spread_pct", "squeeze_power"
]

# ---------------------- INDICATOR FUNCTIONS ----------------------
def compute_supertrend(df: pd.DataFrame,
                       period: int = 10,
                       multiplier: float | int = 3,):
    """
    Vectorised SuperTrend indicator.

    Returns a boolean Series aligned with *df.index*:
      • True  – price is in an up‑trend
      • False – price is in a down‑trend

    The function is defensive: if there is not enough history or ATR
    cannot be computed it returns a Series of **False** values instead
    of raising, so the caller never hits ``TypeError`` again.
    """
    if len(df) < (period + 1) * 2:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)

    high  = df["highPrice"].astype("float32")
    low   = df["lowPrice"].astype("float32")
    close = df["closePrice"].astype("float32")

    atr = ta.atr(high, low, close, length=period)

    if atr.isna().all():
        return pd.Series([False] * len(df), index=df.index, dtype=bool)

    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=bool)
    in_uptrend = True

    for i in range(len(df)):
        if i == 0:
            supertrend.iat[i] = in_uptrend
            continue

        if close.iat[i] > upperband.iat[i - 1]:
            in_uptrend = True
        elif close.iat[i] < lowerband.iat[i - 1]:
            in_uptrend = False

        if in_uptrend and lowerband.iat[i] < lowerband.iat[i - 1]:
            lowerband.iat[i] = lowerband.iat[i - 1]
        if not in_uptrend and upperband.iat[i] > upperband.iat[i - 1]:
            upperband.iat[i] = upperband.iat[i - 1]

        supertrend.iat[i] = in_uptrend

    return supertrend


# ---------------------- WEBSOCKET: PUBLIC ----------------------
class PublicWebSocketManager:
    
    ENABLE_UNSUBSCRIBE = False

    __slots__ = (
        "symbols", "interval", "ws",
        "candles_data", "ticker_data", "latest_open_interest",
        "active_symbols", "_last_selection_ts", "_callback",
        "ready_event",
        "loop", "volume_history", "oi_history", "cvd_history",
        "_last_saved_time", "position_handlers", "_history_file",
        "_save_task", "latest_liquidation",
        "_liq_thresholds", "last_liq_trade_time", "funding_history",
        "bot",
    )
    def __init__(self, symbols, interval="1"):
        self.symbols = symbols
        self.interval = interval
        self.ws = None
        self.candles_data   = defaultdict(lambda: deque(maxlen=1000))
        self.ticker_data = {}
        self.latest_open_interest = {}
        self.active_symbols = set(symbols)
        self._last_selection_ts = time.time()
        self._callback = None
        self.ready_event = asyncio.Event()
        self.loop = asyncio.get_event_loop()
        self.volume_history = defaultdict(lambda: deque(maxlen=500))
        self.oi_history     = defaultdict(lambda: deque(maxlen=500))
        self.cvd_history    = defaultdict(lambda: deque(maxlen=500))
        self.funding_history = defaultdict(lambda: deque(maxlen=3))
        self._last_saved_time = {}
        self.position_handlers = []
        self.latest_liquidation = {}
        self._liq_thresholds = defaultdict(lambda: 5000.0)
        self.last_liq_trade_time = {}

        try:
            with open(LIQ_THRESHOLD_CSV_PATH, "r", newline="") as _f:
                for _row in csv.DictReader(_f):
                    self._liq_thresholds[_row["symbol"]] = float(_row["threshold"])
            logger.info("[liq_thresholds] Loaded %d symbols from %s",
                        len(self._liq_thresholds), LIQ_THRESHOLD_CSV_PATH)
        except FileNotFoundError:
            pass
        except Exception as _e:
            logger.warning("[liq_thresholds] load error: %s", _e)
        bot = TradingBot
        self.bot = None
        self._history_file = 'history.pkl'
        try:
            with open(self._history_file, 'rb') as f:
                data = pickle.load(f)
                for sym, rows in data.get('candles', {}).items():
                    self.candles_data[sym] = rows
                for sym, vol in data.get('volume_history', {}).items():
                    self.volume_history[sym] = deque(vol, maxlen=1000)
                for sym, oi in data.get('oi_history', {}).items():
                    self.oi_history[sym] = deque(oi, maxlen=1000)
                for sym, cvd in data.get('cvd_history', {}).items():
                    self.cvd_history[sym] = deque(cvd, maxlen=1000)
            logger.info(f"[History] загружена история из {self._history_file}")
        except Exception:
            pass

    async def start(self):
        while True:
            try:
                def _on_message(msg):
                    try:
                        if not self.loop.is_closed():
                            asyncio.run_coroutine_threadsafe(
                                self.route_message(msg),
                                self.loop
                            )
                    except Exception as e:
                        logger.warning(f"[PublicWS callback] loop closed, skipping message: {e}")

                # <-- ВАЖНО: сохранить колбэк здесь, вне функции
                self._callback = _on_message
                self.ws = WebSocket(
                    testnet=False,
                    channel_type="linear",
                    ping_interval=30,
                    ping_timeout=15,
                    restart_on_error=True,
                    retries=200,
                )
                self.ws.kline_stream(
                    interval=self.interval,
                    symbol=self.symbols,
                    callback=_on_message
                )
                self.ws.ticker_stream(
                    symbol=self.symbols,
                    callback=_on_message
                )
                self.ws.all_liquidation_stream(
                    symbol=self.active_symbols,
                    callback=_on_message
                )
                
                self.active_symbols = set(self.symbols)  # считаем их уже подписанными

                if not hasattr(self, "_save_task"):
                    self._save_task = asyncio.create_task(self._save_loop())

                self._callback = _on_message
                asyncio.create_task(self.manage_symbol_selection(check_interval=60))
                await asyncio.Event().wait()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[PublicWS] reconnect after error: %s", e)
                await asyncio.sleep(5)

    def get_liq_threshold(self, symbol: str, default: float = 5000.0) -> float:
        t24 = safe_to_float(self.ticker_data.get(symbol, {}).get("turnover24h", 0))
        if t24 > 0:
            return max(5_000.0, 0.001 * t24)
        return max(15_000.0, self._liq_thresholds.get(symbol, default))

    def _is_already_subscribed(self, topic_template: str, sym: str) -> bool:
        """Проверка, есть ли такой топик в current_topics."""
        return topic_template.format(symbol=sym) in self.ws.current_topics
    
    # --- helpers for WS topics / duplicate filtering ---

    def _existing_topics(self) -> set[str]:
        """Вернуть множество уже зарегистрированных топиков у pybit через callback_directory."""
        try:
            d = getattr(self.ws, "callback_directory", {})
            return set(d.keys()) if isinstance(d, dict) else set()
        except Exception:
            return set()

    def _fmt_topic(self, tpl: str, sym: str) -> str:
        return tpl.format(symbol=sym)

    def _filter_new_symbols(self, tpl: str, symbols: list[str], existing: set[str]) -> list[str]:
        """Оставить только те символы, для которых ещё нет подписки по шаблону tpl."""
        fmt = self._fmt_topic
        return [s for s in symbols if fmt(tpl, s) not in existing]



    # [REWORKED FOR STABILITY] Отказоустойчивое управление подписками
    async def manage_symbol_selection(self, min_turnover=2_000_000,
                                    min_volume=1_200_000, check_interval=3600):
        http = HTTP(testnet=False)
        is_first_run = True

        while True:
            await asyncio.sleep(10 if is_first_run else check_interval)
            try:
                resp = await asyncio.to_thread(lambda: http.get_tickers(category="linear"))
                all_tickers = {tk["symbol"]: tk for tk in resp["result"]["list"]}
                self.ticker_data.update(all_tickers)

                liquid_symbols = {
                    s for s, t in all_tickers.items()
                    if safe_to_float(t.get("turnover24h", 0)) >= min_turnover and
                       safe_to_float(t.get("volume24h", 0)) >= min_volume
                }
                open_pos_symbols = {s for bot in self.position_handlers for s in bot.open_positions.keys()}
                new_active_symbols = liquid_symbols.union(open_pos_symbols)
                
                # symbols_to_add = new_active_symbols - self.active_symbols
                # symbols_to_remove = self.active_symbols - new_active_symbols

                desired_symbols   = liquid_symbols | open_pos_symbols
                symbols_to_add    = desired_symbols - self.active_symbols
                symbols_to_remove = self.active_symbols - desired_symbols

                # 5) Динамически обновляем подписки (с фильтром дублей у pybit)
                k_tpl = f"kline.{self.interval}.{{symbol}}"
                t_tpl = "tickers.{symbol}"
                l_tpl = "liquidation.{symbol}"
                
                if symbols_to_add:
                    symbols_to_add = list(symbols_to_add)
                    existing = self._existing_topics()

                    add_k = self._filter_new_symbols(k_tpl, list(symbols_to_add), existing)
                    add_t = self._filter_new_symbols(t_tpl, list(symbols_to_add), existing)
                    add_l = self._filter_new_symbols(l_tpl, list(symbols_to_add), existing)

                    for tpl, bucket in ((k_tpl, add_k), (t_tpl, add_t), (l_tpl, add_l)):
                        for s in bucket:
                            topic = self._fmt_topic(tpl, s)
                            if topic in existing:
                                continue
                            try:
                                self.ws.subscribe(topic=tpl, symbol=[s], callback=self._callback)
                                existing.add(topic)
                            except Exception as e:
                                msg = str(e).lower()
                                if "already subscribed" in msg:
                                    existing.add(topic)
                                    logger.debug(f"[SymbolManager] dup topic {topic} ignored")
                                else:
                                    logger.error(f"[SymbolManager] subscribe {topic} failed: {e}")
                
                    # Синхронизируем своё состояние даже если были дубли
                    self.active_symbols.update(symbols_to_add)
                
                # Отписка — опциональна
                if symbols_to_remove and getattr(self, "ENABLE_UNSUBSCRIBE", False):
                    try:
                        self.ws.unsubscribe(topic=k_tpl, symbol=list(symbols_to_remove))
                    except Exception:
                        pass
                    try:
                        self.ws.unsubscribe(topic=t_tpl, symbol=list(symbols_to_remove))
                    except Exception:
                        pass
                    try:
                        self.ws.unsubscribe(topic=l_tpl, symbol=list(symbols_to_remove))
                    except Exception:
                        pass

                self.active_symbols -= symbols_to_remove
                self.symbols = list(self.active_symbols)

                if is_first_run:
                    self.ready_event.set()
                    is_first_run = False
                    logger.info(f"[SymbolManager] Начальный список из {len(self.active_symbols)} символов сформирован. Бот готов к работе.")

            except Exception as e:
                logger.error(f"[SymbolManager] Ошибка: {e}", exc_info=True)
                if is_first_run and not self.ready_event.is_set():
                    self.ready_event.set()
                    is_first_run = False
                    logger.info(f"[SymbolManager] Начальная настройка завершена. Бот готов к работе.")


    async def route_message(self, msg):
        topic = msg.get("topic", "")

        if topic.startswith("kline."):
            await self.handle_kline(msg)

        elif topic.startswith("tickers."):
            await self.handle_ticker(msg)

        elif "liquidation" in topic.lower():
            # Поддержка both: "allLiquidation.SYM" и любые варианты c 'liquidation'
            try:
                if not hasattr(self, "shared_ws") or not hasattr(self.shared_ws, "latest_liquidation"):
                    return  # никуда писать
                data = msg.get("data", [])
                if isinstance(data, dict):
                    data = [data]
                now = time.time()
                for evt in data:
                    sym = evt.get("s")
                    if not sym:
                        continue
                    side_evt = str(evt.get("S", "")).capitalize()  # "Buy"/"Sell"
                    qty = safe_to_float(evt.get("v", 0))
                    px = safe_to_float(evt.get("p", 0))
                    value = qty * px
                    if value <= 0:
                        continue
                    self.shared_ws.latest_liquidation[sym] = {
                        "ts": now,
                        "side": side_evt,
                        "value": value,
                    }
            except Exception:
                logger.exception("[route_message] failed to ingest liquidation")

    async def handle_kline(self, msg):
        raw = msg.get("data")
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            if not entry.get("confirm", False):
                continue
            symbol = msg["topic"].split(".")[-1]
            try:
                ts = pd.to_datetime(int(entry["start"]), unit="ms")
            except Exception as e:
                print(f"[handle_kline] invalid start: {e}")
                continue
            if self._last_saved_time.get(symbol) == ts:
                continue
            row = {
                "startTime": ts,
                "openPrice": safe_to_float(entry.get("open", 0)),
                "highPrice": safe_to_float(entry.get("high", 0)),
                "lowPrice": safe_to_float(entry.get("low", 0)),
                "closePrice": safe_to_float(entry.get("close", 0)),
                "volume": safe_to_float(entry.get("volume", 0)),
            }
            self.candles_data[symbol].append(row)
            self.volume_history[symbol].append(row["volume"])
            oi_val = self.latest_open_interest.get(symbol, 0.0)
            self.oi_history[symbol].append(oi_val)
            delta = row["volume"] if row["closePrice"] >= row["openPrice"] else -row["volume"]
            prev_cvd = self.cvd_history[symbol][-1] if self.cvd_history[symbol] else 0.0
            self.cvd_history[symbol].append(prev_cvd + delta)
            self._last_saved_time[symbol] = ts
            logger.debug("[handle_kline] stored candle for %s @ %s", symbol, ts)

    async def handle_ticker(self, msg):
        data = msg.get("data", {})
        entries = data if isinstance(data, list) else [data]
        for ticker in entries:
            symbol = ticker.get("symbol")
            if not symbol:
                continue
            oi = ticker.get("openInterest") or ticker.get("open_interest") or 0
            oi_val = safe_to_float(oi)
            self.latest_open_interest[symbol] = oi_val
            self.ticker_data[symbol] = ticker
            f_raw = ticker.get("fundingRate") or ticker.get("funding_rate")
            if f_raw is not None:
                f_val = safe_to_float(f_raw)
                self.funding_history[symbol].append(f_val)
            hist = self.oi_history.setdefault(symbol, deque(maxlen=500))
            if not hist or hist[-1] != oi_val:
                hist.append(oi_val)

        for bot in self.position_handlers:
            for ticker in entries:
                sym = ticker.get("symbol")
                if not sym or sym not in bot.open_positions:
                    continue
                last_price = safe_to_float(ticker.get("lastPrice", 0))
                asyncio.create_task(bot.on_ticker_update(sym, last_price))

    async def backfill_history(self):
        http = HTTP(testnet=False)
        for symbol in self.symbols:
            recent = self.candles_data.get(symbol, [])
            last_ms = int(recent[-1]['startTime'].timestamp()*1000) if recent else None
            try:
                params = {'symbol': symbol, 'interval': self.interval}
                if last_ms:
                    params['start'] = last_ms
                else:
                    params['limit'] = 500
                resp = await asyncio.to_thread(lambda: http.get_kline(**params))
                bars = resp.get('result', {}).get('list', [])
                count = 0
                for entry in bars:
                    if isinstance(entry, list):
                        try:
                            ts = pd.to_datetime(int(entry[0]), unit='ms')
                            open_p  = safe_to_float(entry[1])
                            high_p  = safe_to_float(entry[2])
                            low_p   = safe_to_float(entry[3])
                            close_p = safe_to_float(entry[4])
                            vol     = safe_to_float(entry[5])
                        except Exception:
                            print(f"[History] backfill invalid list entry for {symbol}: {entry}")
                            continue
                    else:
                        try:
                            ts = pd.to_datetime(int(entry['start']), unit='ms')
                        except Exception as e:
                            print(f"[History] backfill invalid dict entry for {symbol}: {e}")
                            continue
                        open_p  = safe_to_float(entry.get('open', 0))
                        high_p  = safe_to_float(entry.get('high', 0))
                        low_p   = safe_to_float(entry.get('low', 0))
                        close_p = safe_to_float(entry.get('close', 0))
                        vol     = safe_to_float(entry.get('volume', 0))
                    row = {
                        'startTime': ts,
                        'openPrice': open_p,
                        'highPrice': high_p,
                        'lowPrice': low_p,
                        'closePrice': close_p,
                        'volume': vol,
                    }
                    self.candles_data[symbol].append(row)
                    self.volume_history[symbol].append(vol)
                    self.oi_history[symbol].append(0.0)
                    delta = vol if close_p >= open_p else -vol
                    prev_cvd = self.cvd_history[symbol][-1] if self.cvd_history[symbol] else 0.0
                    self.cvd_history[symbol].append(prev_cvd + delta)
                    count += 1
                if count:
                    self._save_history()
                    try:
                        ticker_resp = http.get_tickers(category="linear", symbol=symbol)
                        oi_val = safe_to_float(
                            ticker_resp["result"]["list"][0].get("openInterest", 0) or
                            ticker_resp["result"]["list"][0].get("open_interest", 0)
                        )
                    except Exception:
                        oi_val = 0.0

                    need = len(self.candles_data[symbol]) - len(self.oi_history[symbol])
                    if need > 0:
                        self.oi_history[symbol].extend([oi_val] * need)
                    logger.info("[History] backfilled %d bars for %s", count, symbol)
            except Exception as e:
                print(f"[History] backfill error for {symbol}: {e}")

        self._liq_thresholds.clear()
        try:
            import pandas as _pd
            _liq = _pd.read_csv(LIQUIDATIONS_CSV_PATH)
            for _sym, _grp in _liq.groupby('symbol'):
                p5 = _grp['value_usdt'].quantile(0.05)
                self._liq_thresholds[_sym] = max(5000.0, p5 * 4)
        except Exception as _e:
            logger.warning('Threshold init failed: %s', _e)

    def get_liq_threshold(self, symbol: str, default: float = 5000.0) -> float:
        t24 = safe_to_float(self.ticker_data.get(symbol, {}).get("turnover24h", 0))
        if t24 >= LARGE_TURNOVER:
            return 0.0015 * t24
        elif t24 >= MID_TURNOVER:
            return 0.0025 * t24
        return max(8_000.0, self._liq_thresholds.get(symbol, default))

    def get_avg_volume(self, symbol: str, minutes: int = 30) -> float:
        candles_deque = self.candles_data.get(symbol, [])
        if not candles_deque:
            return 0.0
        recent = list(candles_deque)[-minutes:]
        vols = [
            safe_to_float(c.get("turnover") or c.get("volume", 0))
            for c in recent
        ]
        vols = [v for v in vols if v > 0]
        return sum(vols) / max(1, len(vols))

    def _sigma_5m(self, symbol: str, window: int = VOL_WINDOW) -> float:
        candles = list(self.candles_data.get(symbol, []))[-window:]
        if len(candles) < window:
            return 0.0
        # [BUG FIX] Явно приводим все значения к float перед вычислениями
        moves = [
            abs(float(c["closePrice"]) - float(c["openPrice"])) / float(c["openPrice"])
            for c in candles if safe_to_float(c.get("openPrice")) > 0
        ]
        return float(np.std(moves)) if moves else 0.0

    def _listing_age_minutes(self, symbol: str) -> float:
        candles = self.candles_data.get(symbol, [])
        if not candles:
            return 0.0
        first_ts = candles[0]["startTime"]
        return (dt.datetime.utcnow() - first_ts.to_pydatetime()).total_seconds() / 60.0

    def is_too_new(self, symbol: str, min_age: int | None = None) -> bool:
        min_age = min_age or getattr(self, "listing_age_min", LISTING_AGE_MIN_MINUTES)
        return self._listing_age_minutes(symbol) < min_age

    def is_volatile_spike(self, symbol: str, candle: dict) -> bool:
        sigma = self._sigma_5m(symbol)
        if sigma == 0:
            return False
        move = abs(candle["closePrice"] - candle["openPrice"]) / candle["openPrice"]
        return move >= VOL_COEF * sigma

    def funding_cool(self, symbol: str) -> bool:
        hist = self.funding_history.get(symbol, [])
        if len(hist) < 2:
            return True
        prev, curr = hist[-2], hist[-1]
        if prev > 0 and curr < 0:
            return True
        return abs(curr) <= 0.5 * abs(prev)

    def _rsi_series(self, symbol: str, period: int = 14, lookback: int = 180):
        candles = list(self.candles_data.get(symbol, []))
        if len(candles) < period + lookback:
            return pd.Series(dtype=float)
        closes = [c["closePrice"] for c in candles[-lookback:]]
        return ta.rsi(pd.Series(closes), length=period)

    def rsi_blocked(self, symbol: str, side: str,
                    overbought: float = 82.0,
                    oversold: float = 20.0,
                    lookback: int = 180) -> bool:
        rsi = self._rsi_series(symbol, period=14, lookback=lookback)
        if rsi.empty:
            return False
        if side == "Buy":
            return rsi.min() > overbought
        if side == "Sell":
            return rsi.max() < oversold
        return False

    def get_delta_oi(self, symbol: str):
        hist = self.oi_history.get(symbol, [])
        if len(hist) < 2:
            return None
        prev, last = hist[-2], hist[-1]
        if prev == 0:
            return None
        return (last - prev) / prev

    def check_liq_cooldown(self, symbol: str) -> bool:
        sigma = self._sigma_5m(symbol)
        cooldown = 900 if sigma >= 0.01 else 600
        last = self.last_liq_trade_time.get(symbol)
        if not last:
            return True
        return (dt.datetime.utcnow() - last).total_seconds() >= cooldown

    def _aggregate_last_candles(self, symbol: str, n: int = 1):
        deque_c = self.shared_ws.candles_data.get(symbol, [])
        if len(deque_c) < n:
            return None
        recent = list(deque_c)[-n:]
        return {
            "openPrice": recent[0]["openPrice"],
            "closePrice": recent[-1]["closePrice"],
            "highPrice": max(c["highPrice"] for c in recent),
            "lowPrice":  min(c["lowPrice"]  for c in recent),
            "volume":    sum(c["volume"]    for c in recent),
        }

    def _save_history(self):
        try:
            with open(self._history_file, 'wb') as f:
                pickle.dump({
                    'candles': dict(self.candles_data),
                    'volume_history': {k: list(v) for k, v in self.volume_history.items()},
                    'oi_history': {k: list(v) for k, v in self.oi_history.items()},
                    'cvd_history': {k: list(v) for k, v in self.cvd_history.items()},
                }, f)
        except Exception as e:
            logger.warning("[History] ошибка сохранения: %s", e)

    async def _save_loop(self, interval: int = 60):
        while True:
            await asyncio.sleep(interval)
            self._save_history()

    async def optimize_liq_thresholds(self,
                                      trades_csv: str = "trades_for_training.csv",
                                      min_trades: int = 30):
        try:
            df = pd.read_csv(trades_csv)
        except FileNotFoundError:
            logger.debug("[opt_liq] %s not found – skip", trades_csv)
            return
        if "source" not in df.columns or "liq_value" not in df.columns:
            logger.debug("[opt_liq] required columns missing – skip")
            return
        df = df[df["source"] == "liquidation"]
        if df.empty:
            return

        results = {}
        for sym, grp in df.groupby("symbol"):
            if len(grp) < min_trades:
                continue
            best_wr, best_thr = 0.0, None
            for q in (0.01, 0.05, 0.10, 0.20):
                thr = grp["liq_value"].quantile(q)
                test = grp[grp["liq_value"] >= thr]
                if len(test) < min_trades:
                    continue
                wr = (test["pnl_pct"] > 0).mean()
                if wr > best_wr:
                    best_wr, best_thr = wr, thr
            if best_thr:
                results[sym] = best_thr

        if results:
            try:
                with open(LIQ_THRESHOLD_CSV_PATH, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["symbol", "threshold"])
                    for sym, thr in results.items():
                        w.writerow([sym, thr])
                logger.info("[opt_liq] saved %d thresholds → %s",
                            len(results), LIQ_THRESHOLD_CSV_PATH)
            except Exception as e:
                logger.warning("[opt_liq] save error: %s", e)

            for sym, thr in results.items():
                self._liq_thresholds[sym] = thr

    async def place_order_ws(self, symbol, side, qty,
                            position_idx=1, price=None,
                            order_type="Market", cid=None):
        header = {
            "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
            "X-BAPI-RECV-WINDOW": "5000"
        }
        args = {
            "symbol": symbol,
            "side": side,
            "qty": str(qty),
            "category": "linear",
            "timeInForce": "GTC"
        }
        step    = self.qty_step_map.get(symbol, DEC_TICK)
        min_qty = self.min_qty_map.get(symbol, step)
        if float(qty) < float(min_qty):
            raise RuntimeError(f"Qty {qty} < min_qty {min_qty}")
        args["orderType"] = order_type
        args["positionIdx"] = position_idx
        if price is not None and order_type == "Limit":
            args["price"] = str(price)

        req_id = f"{symbol}-{int(time.time()*1000)}"
        req = {
            "op": "order.create",
            "req_id": req_id,
            "header": header,
            "args": [args]
        }
        t0 = time.time()
        logger.info("[ORDER][%s] send %s", cid or "-", j({"req_id": req_id, "args": args}))
        await self.ws_trade.send(json.dumps(req))
        while True:
            resp = json.loads(await self.ws_trade.recv())
            if resp.get("req_id") == req_id:
                break
        resp = json.loads(await self.ws_trade.recv())
        dt_ms = int((time.time() - t0) * 1000)
        logger.info("[ORDER][%s] recv %dms %s", cid or "-", dt_ms, j(resp))

        if resp.get("retCode", 1) != 0:
            logger.error("[ORDER][%s] failed ret=%s msg=%s", cid or "-", resp.get("retCode"), resp.get("retMsg"))
            raise RuntimeError(f"Order failed: {resp}")

        return resp.get("data", resp)

    def has_5_percent_growth(self, symbol: str, minutes: int = 20) -> bool:
        """Проверяет, был ли рост цены на 5% за последние N минут"""
        candles = list(self.candles_data.get(symbol, []))
        if len(candles) < minutes:
            return False

        old_candle = candles[-minutes]
        new_candle = candles[-1]

        old_close = safe_to_float(old_candle.get("closePrice", 0))
        new_close = safe_to_float(new_candle.get("closePrice", 0))

        if old_close <= 0:
            return False

        pct_change = (new_close - old_close) / old_close * 100.0
        return pct_change >= 3.0

def compute_pct(candles_deque, minutes: int) -> float:
    """
    Compute percentage price change over the last `minutes` intervals.
    Returns 0.0 if there are not enough candles.
    """
    data = list(candles_deque)
    if len(data) < minutes + 1:
        return 0.0
    old_close = safe_to_float(data[-minutes - 1].get("closePrice", 0))
    new_close = safe_to_float(data[-1].get("closePrice", 0))
    if old_close <= 0:
        return 0.0
    return (new_close - old_close) / old_close * 100.0

def sum_last_vol(candles_deque, minutes: int) -> float:
    """
    Sum the volumes of the last `minutes` candles.
    Returns 0.0 if there are not enough candles.
    """
    data = list(candles_deque)[-minutes:]
    total = 0.0
    for c in data:
        total += safe_to_float(c.get("volume", 0))
    return total

# ======================================================================
# == МАШИННОЕ ОБУЧЕНИЕ: MLX-ONLY
# ======================================================================

class GoldenNetMLX(mlx_nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = mlx_nn.Linear(input_size, hidden_size)
        self.bn1 = mlx_nn.BatchNorm(hidden_size)
        self.fc2 = mlx_nn.Linear(hidden_size, hidden_size)
        self.bn2 = mlx_nn.BatchNorm(hidden_size)
        self.fc3 = mlx_nn.Linear(hidden_size, 1)
        self.dropout = mlx_nn.Dropout(0.2)

    def __call__(self, x):
        x = mlx_nn.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = mlx_nn.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

    # === добавляем ===
    def state_dict_numpy(self) -> dict:
        """Собираем веса слоёв в dict numpy-массивов — для safetensors."""
        to_np = lambda t: np.array(mlx.to_numpy(t))
        sd = {}
        # Linear 1
        sd["fc1.weight"] = to_np(self.fc1.weight)
        sd["fc1.bias"]   = to_np(self.fc1.bias)
        # BN1
        sd["bn1.weight"]       = to_np(self.bn1.weight)
        sd["bn1.bias"]         = to_np(self.bn1.bias)
        sd["bn1.running_mean"] = to_np(self.bn1.running_mean)
        sd["bn1.running_var"]  = to_np(self.bn1.running_var)
        # Linear 2
        sd["fc2.weight"] = to_np(self.fc2.weight)
        sd["fc2.bias"]   = to_np(self.fc2.bias)
        # BN2
        sd["bn2.weight"]       = to_np(self.bn2.weight)
        sd["bn2.bias"]         = to_np(self.bn2.bias)
        sd["bn2.running_mean"] = to_np(self.bn2.running_mean)
        sd["bn2.running_var"]  = to_np(self.bn2.running_var)
        # Linear 3
        sd["fc3.weight"] = to_np(self.fc3.weight)
        sd["fc3.bias"]   = to_np(self.fc3.bias)
        return sd

    def load_weights(self, path: str):
        """Загружаем веса из safetensors в слои MLX."""
        tensors = load_safetensors(path)
        to_mlx = lambda a: mlx.array(a)
        # Linear 1
        self.fc1.weight = to_mlx(tensors["fc1.weight"])
        self.fc1.bias   = to_mlx(tensors["fc1.bias"])
        # BN1
        self.bn1.weight       = to_mlx(tensors["bn1.weight"])
        self.bn1.bias         = to_mlx(tensors["bn1.bias"])
        self.bn1.running_mean = to_mlx(tensors["bn1.running_mean"])
        self.bn1.running_var  = to_mlx(tensors["bn1.running_var"])
        # Linear 2
        self.fc2.weight = to_mlx(tensors["fc2.weight"])
        self.fc2.bias   = to_mlx(tensors["fc2.bias"])
        # BN2
        self.bn2.weight       = to_mlx(tensors["bn2.weight"])
        self.bn2.bias         = to_mlx(tensors["bn2.bias"])
        self.bn2.running_mean = to_mlx(tensors["bn2.running_mean"])
        self.bn2.running_var  = to_mlx(tensors["bn2.running_var"])
        # Linear 3
        self.fc3.weight = to_mlx(tensors["fc3.weight"])
        self.fc3.bias   = to_mlx(tensors["fc3.bias"])

class MLXInferencer:
    def __init__(self, model_path="golden_model_mlx.safetensors", scaler_path="scaler.pkl"):
        self.model = None
        self.scaler = None

        if Path(model_path).exists():
            try:
                self.model = GoldenNetMLX(input_size=len(FEATURE_KEYS))
                self.model.load_weights(model_path)
                self.model.eval()
                logger.info(f"[MLX] Модель из {model_path} успешно загружена.")
            except Exception as e:
                logger.error(f"[MLX] Ошибка загрузки модели {model_path}: {e}", exc_info=True)

        if Path(scaler_path).exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"[MLX] Скейлер из {scaler_path} успешно загружен.")
            except Exception as e:
                logger.error(f"[MLX] Ошибка загрузки скейлера {scaler_path}: {e}", exc_info=True)

    def infer(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([[0.0]])
        if self.scaler:
            features = self.scaler.transform(features)
        prediction = self.model(mlx.array(features))
        return np.array(prediction)
    
    
    @staticmethod
    def train_golden_model_mlx(training_data, num_epochs: int = 30, lr: float = 1e-3):
        logger.info("[MLX] Запуск обучения на MLX...")
        feats = np.asarray([d["features"] for d in training_data], dtype=np.float32)
        targ = np.asarray([d["target"] for d in training_data], dtype=np.float32)
        mask = ~(np.isnan(feats).any(1) | np.isinf(feats).any(1))
        feats, targ = feats[mask], targ[mask]
        if feats.size == 0:
            raise ValueError("train_golden_model_mlx: нет валидных сэмплов")

        scaler = StandardScaler().fit(feats)
        feats_scaled = scaler.transform(feats).astype(np.float32)
        targ = targ.reshape(-1, 1)

        model = GoldenNetMLX(input_size=feats_scaled.shape[1])
        optimizer = mlx_optim.Adam(learning_rate=lr)
        loss_fn = lambda model, x, y: mlx_nn.losses.mse_loss(model(x), y).mean()
        loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

        for epoch in range(num_epochs):
            x_train, y_train = mlx.array(feats_scaled), mlx.array(targ)
            loss, grads = loss_and_grad_fn(model, x_train, y_train)
            optimizer.update(model, grads)
            mlx.eval(model.parameters(), optimizer.state)
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1} [MLX] – Loss: {loss.item():.5f}")

        return model, scaler

def save_mlx_checkpoint(model: GoldenNetMLX, scaler: StandardScaler,
                        model_path: str = "golden_model_mlx.safetensors",
                        scaler_path: str = "scaler.pkl"):
    tensors = model.state_dict_numpy()
    save_safetensors(tensors, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("[MLX] Модель сохранена → %s; scaler → %s", model_path, scaler_path)

# ======================================================================
# == КОНЕЦ ML-БЛОКА
# ======================================================================

# === AI ensemble (primary + confirm/plutus) ==============================
# Настройки по умолчанию – можно вынести в конфиг user_state.json
AI_PRIMARY_MODEL   = "trading-llama"              # ваша дообученная локальная модель (Ollama)
AI_CONFIRM_MODEL   = "0xroyce/plutus:latest"       # подтверждающая модель
AI_TIMEOUT_PRIMARY = 8.0
AI_TIMEOUT_CONFIRM = 6.0

AI_EXEC_TH   = 0.60       # EXECUTE если >= 0.60 (после всех фильтров)
AI_WATCH_LO  = 0.35       # WATCH если в [0.35..EXEC_TH)
AI_WATCH_HI  = 0.60

NEGATIVE_CUES = {
    "негатив", "опас", "перегрет", "переоцен", "risk", "bear", "downtrend",
    "слабый спрос", "дистрибуц", "падени", "разворот вниз", "sell-off", "dump"
}

def _clamp01(x):
    try:
        x = float(x)
        if x < 0: return 0.0
        if x > 1: return 1.0
        return x
    except Exception:
        return 0.0


# async def _ask_ollama_json(self, model: str, messages: list[dict], timeout_s: float | None = None) -> dict:
#     from openai import AsyncOpenAI
#     client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
#     tmo = float(timeout_s or self.ai_timeout_sec or 4.0)

#     try:
#         resp = await asyncio.wait_for(
#             client.chat.completions.create(
#                 model=model,
#                 messages=messages,
#                 response_format={"type": "json_object"},
#                 temperature=0.2,
#                 top_p=1,
#                 max_tokens=256,
#             ),
#             timeout=tmo
#         )
#         raw = resp.choices[0].message.content
#         return safe_parse_json(raw, default={"action": "REJECT", "confidence_score": 0.0, "justification": "bad json"})
#     except asyncio.TimeoutError:
#         raise
#     except Exception as e:
#         logger.error(f"[AI_ERROR] {model}: {e}", exc_info=True)
#         return {"action": "REJECT", "confidence_score": 0.0, "justification": f"error: {e}"}
    

def safe_parse_json(text: str | None, default: Any = None) -> Any:
    """
    Пытается распарсить JSON из ответа LLM/Ollama.
    - Аккуратно обрабатывает пустую строку
    - Срезает кодовые блоки ```json ... ```
    - Если модель вернула пояснительный текст вокруг JSON, пытается вытащить первый блок {...}
    - Возвращает default при любой ошибке
    """
    if text is None:
        return default

    s = text.strip()

    # Снимаем ограждение ```json ... ```
    if s.startswith("```"):
        # убираем первую строку с ```/```json
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        # убираем завершающие ```
        s = re.sub(r"\s*```$", "", s).strip()

    # Прямая попытка
    try:
        return json.loads(s)
    except Exception:
        pass

    # Если вокруг JSON есть текст – вытащим первый блок {...}
    try:
        start = s.find("{")
        end   = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start:end+1]
            return json.loads(candidate)
    except Exception:
        pass

    # Ничего не вышло
    return default


# ---------------------- TRADING BOT ----------------------
class TradingBot:

    __slots__ = (
        "user_id", "api_key", "api_secret", "monitoring", "mode",
        "history", "_last_golden_ts", "golden_cooldown_sec", "golden_rsi_max",
        "session", "shared_ws", "ws_private", "symbols",
        "open_positions", "last_position_state", "golden_param_store",
        "market_task", "sync_task", "ws_trade", "loop", "position_idx", "model",
        "POSITION_VOLUME", "MAX_TOTAL_VOLUME", "qty_step_map", "min_qty_map", "price_tick_map",
        "failed_orders", "pending_orders", "pending_strategy_comments", "last_trailing_stop_set",
        "position_lock", "closed_positions", "pnl_task", "last_seq",
        "ws_opened_symbols", "ws_closed_symbols", "averaged_symbols", "limiter",
        "turnover24h", "selected_symbols", "last_asset_selection_time",
        "wallet_task", "last_stop_price", "_last_trailing_ts", "_recv_lock", "max_allowed_volume",
        "strategy_mode", "liq_buffers", "liq_window_sec", "liq_min_cluster_usdt",
        "trailing_start_map", "trailing_gap_map", "enable_start_stop_adapt",
        "trailing_start_pct", "trailing_gap_pct", "ml_inferencer",
        "pending_timestamps", "_cooldown_noise_until",
        "squeeze_threshold_pct", "squeeze_power_min", "averaging_enabled", "squeeze_lookback_min", "squeeze_min_score", 
        "exhaustion_enter_thr", "continuation_follow_thr", "squeeze_atr_k", "squeeze_entry_strategy",
        "warmup_done", "warmup_seconds", "_last_snapshot_ts", "reserve_orders",
        "MLX_model", "feature_scaler", "last_retrain", "training_data", "device",
        "last_squeeze_ts", "squeeze_cooldown_sec", "active_trade_entries", "listing_age_min", "_age_cache",
        "symbol_info", "trade_history_file", "active_trades", "pending_signals", "max_signal_age",
        "_oi_sigma", "_pending_clean_task", "squeeze_tuner",
        "_best_entries", "_best_entry_seen", "_best_entry_cache", "pending_ttl_sec",
        "_golden_weights", "_squeeze_weights", "_liq_weights",
        "gemini_api_key", "ml_lock", "ml_model_bundle", "_last_manage_ts", "training_data_path", "evaluated_signals_cache",
        "gemini_limiter", "_evaluated_signals_cache", "openai_api_key", "ai_stop_management_enabled", "failed_stop_attempts",
        "ml_framework", "_build_default", "ai_provider", "stop_loss_mode", "_last_logged_stop_price", "recently_closed", "_cleanup_task",
        "_last_trailing_stop_order_id", "ai_timeout_sec", "ai_sem", "ai_circuit_open_until", "_ai_silent_until",
        "ml_gate_abs_roi", "ml_gate_min_score", "ml_gate_sigma_coef", "leverage", "order_correlation",
        "STOP_WARMUP_SEC", "ATR_MULT_SL_INIT", "ATR_MULT_TP_INIT", "ATR_MULT_TRAIL", "BREAKEVEN_TRIGGER_ATR", "trailing_update_interval_sec",
        "stop_update_count", "intraday_trailing_enabled", "last_entry_ts", "entry_cooldown_sec", "_pending_close_ts", "watchlist", "watch_tasks",
        "ai_primary_model", "ai_advisor_model", "ai_base_url", "ai_primary_concurrency", "ai_secondary_concurrency", "ai_primary_sem",
        "ai_secondary_sem", "ai_primary_base_url", "ai_secondary_base_url", "ai_primary_temperature", "ai_secondary_temperature", "ai_temperature",
        "ai_max_tokens", "ollama_primary_openai", "ollama_secondary_openai", "ollama_advisor_openai", "ollama_primary_api", "ollama_advisor_api",
        "ai_advisor_base_url", "ai_advisor_sem", "sqz_entry_timeout_sec", "risk_check_timeout_sec", "_ai_inflight_signals", "pending_orders_lock",
        "current_total_volume", "squeeze_signal_ts", "squeeze_k_factor", "squeeze_signal_ttl_sec", "_last_ws_order_id",
        "_sym_locks", "_lock_owner", "_ai_rev_workers_started", "ai_rev_queue", "ai_rev_workers", "enable_ai_reversal_guard",
        "_funding_log_ts", "_funding_log_last", "funding_thr_pos", "funding_thr_neg", "last_entry_comment", "pending_cids",
        "ai_rev_pending", "ai_rev_results",
        )

    def __init__(self, user_data, shared_ws, golden_param_store):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "real")
        self.listing_age_min = int(user_data.get("listing_age_min_minutes", LISTING_AGE_MIN_MINUTES))

        self.current_total_volume = 0.0
        self._cooldown_noise_until = {}
        self.ml_inferencer = MLXInferencer(
            model_path=user_data.get("mlx_model_path", "golden_model_mlx.safetensors"),
            scaler_path=user_data.get("scaler_path", "scaler.pkl"),
        )

        self.squeeze_tuner = None
        logger.info("[MLX] squeeze_tuner disabled – using static squeeze thresholds")

        self.session = HTTP(demo=(self.mode == "demo"),
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            timeout=30)
        try:
            adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
            self.session.client.mount("https://", adapter)
        except Exception:
            pass

        self.entry_cooldown_sec = int(user_data.get("entry_cooldown_sec", 30))
        self.last_entry_ts = {}
        self._pending_close_ts = {}     # символ -> первый раз увидели size=0
        # --- timeouts (safe defaults)
        self.sqz_entry_timeout_sec = getattr(self, "sqz_entry_timeout_sec", 30)   # раньше было 6с — увеличиваем
        self.risk_check_timeout_sec = getattr(self, "risk_check_timeout_sec", 5)  # явный таймаут риск-чека

        self._last_ws_order_id = {}  # symbol -> last ws orderId

        self.shared_ws = shared_ws
        self.history = self.shared_ws.candles_data if self.shared_ws else {}

        if self.shared_ws is not None:
            self.shared_ws.position_handlers.append(self)
        self.symbols = shared_ws.symbols if shared_ws else []

        self.gemini_api_key = user_data.get("gemini_api_key")
        if not self.gemini_api_key:
                    logger.warning(f"[User {self.user_id}] Ключ Gemini API не найден! AI-оценка будет отключена.")

        self.openai_api_key = user_data.get("openai_api_key")
        if not self.openai_api_key:
            logger.warning(f"[User {self.user_id}] Ключ OpenAI API не найден! Оценка через OpenAI будет отключена.")

        self.ml_lock = asyncio.Lock()
        self.ml_model_bundle = {"model": None, "scaler": None}

        self._last_manage_ts: dict[str, float] = {}
        self.pending_ttl_sec = 15

        self._sym_locks: dict[str, asyncio.Lock] = {}
        self._lock_owner: dict[str, tuple[str, float]] = {}  # symbol -> (who, since_ts)

        self.ws_private = None
        self.open_positions = {}
        self.last_position_state: dict[str, tuple[str, float]] = {}
        self.golden_param_store = golden_param_store
        self.market_task = None
        self.ws_trade = None
        self.position_idx = user_data.get("position_idx", 1)

        self.POSITION_VOLUME = safe_to_float(user_data.get("volume", 1000))
        self.MAX_TOTAL_VOLUME = safe_to_float(user_data.get("max_total_volume", 5000))
        self.qty_step_map: dict[str, float] = {}
        self.min_qty_map: dict[str, float] = {}
        self.price_tick_map: dict[str, float] = {}
        self.failed_orders: dict[str, float] = {}
        self.pending_orders: dict[str, float] = {}
        self.pending_cids: dict[str, str] = {}
        self.pending_orders_lock = asyncio.Lock()

        self.liq_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.pending_strategy_comments: dict[str, str] = {}
        self.last_trailing_stop_set: dict[str, float] = {}
        self.position_lock = asyncio.Lock()
        self.closed_positions = {}
        self.last_seq = {}
        self.wallet_task = None
        self.last_stop_price: dict[str, float] = {}
        self._last_logged_stop_price: dict[str, float] = {}
        self._last_trailing_stop_order_id: dict[str, str] = {}
        self._last_trailing_ts: dict[str, float] = {}
        self.ws_opened_symbols = set()
        self.ws_closed_symbols = set()
        self.limiter = AsyncLimiter(max_rate=100, time_period=1)
        self.averaged_symbols: set[str] = set()
        self._recv_lock = asyncio.Lock()
        GLOBAL_BOTS.append(self)
        self.max_allowed_volume = float('10000')

        self.squeeze_threshold_pct = getattr(self, "squeeze_threshold_pct", 4.5)  # % амплитуда за окно
        self.squeeze_lookback_min  = getattr(self, "squeeze_lookback_min", 5)     # минут в окне
        self.squeeze_entry_strategy = "immediate_plus_watcher"
        self.squeeze_min_score         = getattr(self, "squeeze_min_score", 0.60) # фильтр силы сквиза (0..1)
        self.exhaustion_enter_thr      = getattr(self, "exhaustion_enter_thr", 0.60)
        self.continuation_follow_thr   = getattr(self, "continuation_follow_thr", 0.65)
        self.squeeze_atr_k             = getattr(self, "squeeze_atr_k", 0.40)     # сколько ATR за экстремумом

        # ─ сквиз ─
        self.squeeze_signal_ts = {}         # symbol -> first_ts (monotonic)
        self.squeeze_k_factor = 1.5         # динамический порог на основе ATR5m
        self.squeeze_signal_ttl_sec = 180   # «жизнь» сигнала, сек
        # ─ ликвидации ─
        self.liq_buffers = {}               # symbol -> deque[(ts, side_evt, usdt)]
        self.liq_window_sec = 8             # окно агрегации, сек
        self.liq_min_cluster_usdt = 80000.0 # минимальный кластер ликвидаций в USDT

        self.enable_start_stop_adapt = False

        self._age_cache = {}
        self.trade_history_file = Path("trades_history.json")
        self.active_trades: dict[str, dict] = {}
        self._load_trade_history()

        raw_mode = user_data.get("strategy_mode")
        raw_mode = str(raw_mode).lower()

        alias_map = {
            "golden":          "golden_only",
            "golden_only":     "golden_only",
            "squeeze":         "squeeze_only",
            "squeeze_only":    "squeeze_only",
            "liq":             "liquidation_only",
            "liquidation":     "liquidation_only",
            "liquidation_only":"liquidation_only",
            "full":            "full",
            "all":             "full",
            "golden_squeeze":  "golden_squeeze",
            "liq_squeeze":     "liq_squeeze",
        }

        self.strategy_mode = alias_map.get(raw_mode, "full")
        self.active_trade_entries = {}

        self._funding_log_ts = {}
        self._funding_log_last = {}
        self.funding_thr_pos = 0.0010  # 0.10% (подкрути по вкусу)
        self.funding_thr_neg = -0.0010

        # ── Watch-list для «наблюдаемого» сквиза ───────────────────
        self.watchlist: dict[str, float] = {}           # symbol → ts_placed
        self.watch_tasks: dict[str, asyncio.Task] = {}  # symbol → asyncio.Task


        self.trailing_start_map: dict[str, float] = user_data.get("trailing_start_pct", {})
        self.trailing_start_pct: float = self.trailing_start_map.get(
            self.strategy_mode,
            DEFAULT_TRAILING_START_PCT,
        )

        self.trailing_update_interval_sec = float(user_data.get("trailing_update_interval_sec", 0.5)) # По умолчанию - 2 раза в секунду

        self.trailing_gap_map: dict[str, float] = user_data.get("trailing_gap_pct", {})
        self.trailing_gap_pct: float = self.trailing_gap_map.get(
            self.strategy_mode,
            DEFAULT_TRAILING_GAP_PCT,
        )

        self.intraday_trailing_enabled = bool(
            user_data.get("intraday_trailing_enabled", True)  # по умолчанию включено
        )

        self.recently_closed: dict[str, float] = {}
        self._cleanup_task = asyncio.create_task(self._cleanup_recently_closed())

        self.STOP_WARMUP_SEC = int(user_data.get("stop_warmup_sec", 60))
        self.ATR_MULT_SL_INIT = float(user_data.get("atr_mult_sl_init", 2.5))
        self.ATR_MULT_TP_INIT = float(user_data.get("atr_mult_tp_init", 3.0))
        self.ATR_MULT_TRAIL   = float(user_data.get("atr_mult_trail", 1.8))
        self.BREAKEVEN_TRIGGER_ATR = float(user_data.get("breakeven_trigger_atr", 1.0))

        self.ai_stop_management_enabled = user_data.get("ai_stop_management_enabled", False)

        self.squeeze_threshold_pct = user_data.get(
            "squeeze_threshold_pct",
            SQUEEZE_THRESHOLD_PCT,
        )
        self.squeeze_power_min = safe_to_float(user_data.get("squeeze_power_min", DEFAULT_SQUEEZE_POWER_MIN))

        self.apply_user_settings()

        self.pending_timestamps = {}
        self.reserve_orders = {}
        self.order_correlation: dict[str, str] = {}

        self.last_squeeze_ts: dict[str, float] = defaultdict(float)
        self.squeeze_cooldown_sec: int = int(
            user_data.get("squeeze_cooldown_sec", SQUEEZE_COOLDOWN_SEC)
        )
        self.golden_cooldown_sec = int(user_data.get("golden_cooldown_sec", 300))
        self.golden_rsi_max      = float(user_data.get("golden_rsi_max", 80))
        self._last_golden_ts     = defaultdict(float)

        self.warmup_done     = False
        self.warmup_seconds  = int(user_data.get("warmup_seconds", 480))

        self.averaging_enabled: bool = True
        self.gemini_limiter = AsyncLimiter(max_rate=1000, time_period=60)
        self.failed_stop_attempts = {}

        self._evaluated_signals_cache: Dict[str, float] = {}
        asyncio.create_task(self._unfreeze_guard())

        self._last_snapshot_ts: dict[str, float] = {}
        self._oi_sigma: dict[str, float] = defaultdict(float)

        self.reserve_orders: dict[str, dict] = {}
        self._pending_clean_task = asyncio.create_task(self._pending_cleaner())

        self.device = None
        logger.info(f"[ML] Using compute device: {self.device}")

        self.ml_framework = "mlx"
        self.ml_inferencer = MLXInferencer(
            model_path="golden_model_mlx.safetensors",
            scaler_path="scaler.pkl"
        )

        self.ml_gate_abs_roi    = float(user_data.get("ml_gate_abs_roi", 1.5))
        self.ml_gate_min_score  = float(user_data.get("ml_gate_min_score", 0.02))
        self.ml_gate_sigma_coef = float(user_data.get("ml_gate_sigma_coef", 0.0))

        raw_ai = str(user_data.get("ai_provider", "")).lower()
        provider_map = {
            "":       "ollama",
            "ai":     "ollama",
            "ollama": "ollama",
            "openai": "openai",
            "gpt":    "openai",
        }
        self.ai_provider = "ollama"
        self.ai_base_url = "http://localhost:11434/v1"
        logger.info(f"Выбран AI-провайдер: {self.ai_provider.upper()}")
        self.stop_loss_mode = user_data.get("stop_loss_mode", "strat_loss")
        logger.info(f"Выбран режим стоп-лосса: {self.stop_loss_mode.upper()}")

        self._ai_rev_workers_started = False
        # Порог/настройки можно менять на лету: getattr(..., default) в коде ниже
        # self.ai_reversal_conf_threshold = 0.62
        # self.ai_rev_probe_min_dd_bp    = 35   # 0.35% в б.п. с плечом
        # self.ai_rev_k_atr              = 1.2
        self.enable_ai_reversal_guard  = True
        self.ai_rev_queue = None                # будет создана лениво
        self.ai_rev_workers = []                # список тасков-воркеров


        self.last_entry_comment: dict[str, str] = {}

    
        self.ai_timeout_sec = float(user_data.get("ai_timeout_sec", 8.0))
        self.ai_sem = asyncio.Semaphore(user_data.get("ai_max_concurrent", 2))
        self.ai_circuit_open_until = 0.0
        self._ai_silent_until = 0.0
        self._ai_inflight_signals: set[str] = set()   # ключи вида f"{symbol}_{side}_{source}" — сейчас в работе


        self.model = None
        self.MLX_model = None
        self.feature_scaler = None
        self.load_ml_models()
        self.last_retrain = time.time()

        self.training_data_path = Path("training_data.pkl")
        if self.training_data_path.exists():
            try:
                with open(self.training_data_path, "rb") as f:
                    self.training_data = pickle.load(f)
                logger.info(f"[ML] Загружено {len(self.training_data)} обучающих примеров из файла.")
            except Exception as e:
                logger.error(f"[ML] Ошибка загрузки training_data.pkl: {e}. Создан новый буфер.")
                self.training_data = deque(maxlen=5000)
        else:
            self.training_data = deque(maxlen=5000)
        try:
            asyncio.get_running_loop().create_task(self._retrain_loop())
            logger.info("[retrain] task scheduled")
        except RuntimeError:
            pass
        self.symbol_info: dict[str, dict] = {}

        self.pending_signals: dict[str, float] = {}
        self.max_signal_age = 30.0
        self.leverage = 10.0

        self._best_entries: dict[tuple[str, str, str], tuple[float, dict]] = {}
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._drop_stale_signals())
        except RuntimeError:
            pass

        self._oi_sigma: dict[str, float] = defaultdict(float)

        self._best_entry_seen: set[tuple] = set()
        self._best_entry_cache: dict[tuple, tuple[float, dict]] = {}

        self._golden_weights = {}
        self._squeeze_weights = {}
        self._liq_weights = {}
        asyncio.create_task(self._reload_weights_loop())
        asyncio.create_task(self.ensure_ollama_models([self.ai_primary_model, self.ai_advisor_model]))


    def _safe_parse_ai_json(text: str) -> dict:
        import json, re
        try:
            return json.loads(text)
        except Exception:
            # вырезаем возможный префикс/постфикс, ищем первую { … }
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {}



    def _get_sym_lock(self, symbol: str) -> asyncio.Lock:
        lock = self._sym_locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._sym_locks[symbol] = lock
        return lock

    async def _pending_reaper(self):
        while True:
            now = time.monotonic()
            for sym, ts in list(self.pending_timestamps.items()):
                if now - ts > self.pending_ttl_sec:
                    logger.warning("[PENDING] reap stale %s (ttl=%ss)", sym, self.pending_ttl_sec)
                    self.pending_orders.pop(sym, None)
                    self.pending_timestamps.pop(sym, None)
            await asyncio.sleep(5)


    def get_total_open_volume_fast(self) -> float:
        """
        Быстрый расчёт текущей экспозиции по открытым позициям БЕЗ REST.
        Складываем |size| * price по self.open_positions, цену берём из позиции
        (avg_price/markPrice) или из WS-тиков.
        """
        total = 0.0
        try:
            positions = getattr(self, "open_positions", {}) or {}
            ticker_data = getattr(getattr(self, "shared_ws", None), "ticker_data", {}) or {}
            for sym, pos in positions.items():
                # size / volume / qty — поддержим несколько вариантов ключей
                size = (
                    safe_to_float(pos.get("volume", 0.0))
                    if "volume" in pos else
                    safe_to_float(pos.get("size", 0.0))
                    if "size" in pos else
                    safe_to_float(pos.get("qty", 0.0))
                )
                if size <= 0:
                    continue

                price = safe_to_float(
                    pos.get("avg_price")
                    or pos.get("avgPrice")
                    or pos.get("entryPrice")
                    or pos.get("markPrice")
                    or (ticker_data.get(sym, {}) or {}).get("lastPrice")
                    or (ticker_data.get(sym, {}) or {}).get("last_price")
                    or 0.0
                )
                if price > 0:
                    total += abs(size) * price
        except Exception:
            # не ломаем горячий путь из-за любых неожиданностей
            pass
        return total


    async def get_effective_total_volume(self) -> float:
        """
        Эффективная занятость лимита:
        open_volume (по IM кошелька, а если IM=0 — по WS позициям) + pending_orders.
        Никаких REST-вызовов (не блокируемся в горячем пути).
        """
        # pending по твоей схеме: dict[symbol] -> est_cost
        pending_vol = 0.0
        try:
            pending = getattr(self, "pending_orders", {}) or {}
            pending_vol = float(sum(pending.values())) if pending else 0.0
        except Exception:
            pending_vol = 0.0

        # открытые: сначала IM из кошелька (current_total_volume),
        # если он 0/не задан — подстрахуемся объёмом, посчитанным по WS
        wallet_im = float(getattr(self, "current_total_volume", 0.0) or 0.0)
        if wallet_im > 0:
            open_vol = wallet_im
        else:
            open_vol = self.get_total_open_volume_fast()

        effective = open_vol + pending_vol

        # лёгкий информативный лог, когда есть pending
        if pending_vol > 0:
            logger.info(
                "[Risk] Effective volume: %.2f (open: %.2f + pending: %.2f)",
                effective, open_vol, pending_vol
            )
        return effective

    def _append_snapshot(row: dict) -> None:
        """Записывает строку в golden_setup_snapshots.csv, создавая файл при первом запуске."""
        file_exists = os.path.isfile(SNAPSHOT_CSV_PATH)
        with open(SNAPSHOT_CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "close_price",
                    "price_change",
                    "volume_change",
                    "oi_change",
                    "period_iters",
                    "user_id",
                    "symbol",
                    "timestamp",
                    "signal",
                    "signal_strength",
                    "cvd_change",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


    # ---------------- AI REVERSAL BROKER ----------------
    def _ensureai_rev_workers(self):
        """
        Безопасный ленивый запуск воркеров AI-реверсала.
        Приводим ai_rev_concurrency к int, создаём недостающих воркеров,
        не дублируем уже запущенные.
        """
        import asyncio, time

        # --- safe coerce to int ---
        raw = getattr(self, "ai_rev_concurrency", 2)
        try:
            # списки/кортежи -> берём первый элемент
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else 2
            # булево (True/False) не воспринимаем как «1/0»
            if isinstance(raw, bool):
                raw = 2
            # строки -> попытка парсинга первой «цифры» (напр. "2", "2, extra")
            if isinstance(raw, str):
                s = raw.strip()
                # отрезаем по пробелу/запятой/точке с запятой
                for sep in (",", ";", " "):
                    if sep in s:
                        s = s.split(sep, 1)[0].strip()
                        break
                raw = int(float(s))
            want = int(raw)
        except Exception:
            want = 2

        # границы здравого смысла
        want = max(1, min(8, want))

        # --- init storages if missing ---
        if not hasattr(self, "ai_rev_queue") or self.ai_rev_queue is None:
            self.ai_rev_queue = asyncio.Queue(maxsize=256)
        if not hasattr(self, "ai_rev_results") or self.ai_rev_results is None:
            self.ai_rev_results = {}   # symbol -> {"verdict": dict, "ts": float}
        if not hasattr(self, "ai_rev_pending") or self.ai_rev_pending is None:
            self.ai_rev_pending = {}   # symbol -> next_allowed_ts
        if not hasattr(self, "ai_rev_workers") or self.ai_rev_workers is None:
            self.ai_rev_workers = []

        # уже запущено?
        current = len(self.ai_rev_workers)
        need = want - current
        if need <= 0:
            return

        # создаём недостающих
        for _ in range(need):
            task = asyncio.create_task(self._ai_reversal_worker())
            self.ai_rev_workers.append(task)

    async def _ai_reversal_worker(self):
        while True:
            item = await self.ai_rev_queue.get()
            try:
                symbol  = item["symbol"]
                side    = item["side"]
                roi_pct = float(item["roi_pct"])
                feats   = item.get("features") or {}
                verdict = await self._ai_reversal_judge(symbol, side, feats, roi_pct)
                self.ai_rev_results[symbol] = {"verdict": verdict, "ts": time.time()}
            except Exception:
                logger.debug("[AI_REVERSAL] worker error", exc_info=True)
            finally:
                self.ai_rev_queue.task_done()
                await asyncio.sleep(0)  # отдаём управление циклу


    async def _ai_reversal_worker_adapter(self, worker_id: int = 0):
        """Адаптер с worker_id: вызывает старый воркер без аргументов."""
        try:
            await self._ai_reversal_worker()  # <-- тело твоего воркера не меняем
        except Exception:
            logger.exception("[REV_WORKER][%s] crashed", worker_id)


    async def _ai_reversal_judge(self, symbol: str, side: str, features: dict, roi_pct: float) -> dict | None:
        """
        Оценивает: разворот или откат. Возвращает {"action":"EXIT|HOLD","confidence":0..1,"reason":str}
        Использует твой _ask_ollama_json; при ошибке — безопасная эвристика.
        """
        p   = safe_to_float(features.get("price") or 0)
        rsi = safe_to_float(features.get("rsi") or features.get("rsi14") or 0)
        adx = safe_to_float(features.get("adx") or features.get("adx14") or 0)
        obv = safe_to_float(features.get("obv_d1m") or features.get("obv") or 0)
        d_oi = safe_to_float(features.get("dOI1m") or 0)
        messages = [
            {"role":"system","content":"Ты — риск-менеджер. Отвечай строго JSON."},
            {"role":"user","content":(
                "Определи, похоже ли текущее движение на РАЗВОРОТ тренда позиции, или это обычный откат.\n"
                f"Позиция: {side} {symbol}\n"
                f"ROI%(с плечом): {roi_pct:.3f}\n"
                f"price={p:.8f}, rsi={rsi:.2f}, adx={adx:.2f}, obv_d1m={obv:.2f}, dOI1m={d_oi:.4f}\n"
                'Ответь JSON ровно вида: {"action":"EXIT|HOLD","confidence":0..1,"reason":"..."}'
            )}
        ]
        try:
            ans = await self._ask_ollama_json(
                self.ai_advisor_model, messages,
                base_openai_url=getattr(self, "ollama_advisor_openai", None),
                timeout_s=5.0, num_predict=128, temperature=0.0, top_p=1.0,
            )
            if isinstance(ans, dict) and "action" in ans:
                return ans
            from json import loads
            return loads(str(ans))
        except Exception:
            logger.debug("[AI_REVERSAL] AI call failed, using heuristic", exc_info=True)
        # Фолбэк-эвристика: уверенный сдвиг импульса против позиции
        if side == "Buy":
            if rsi < 47 and adx >= 18 and obv < 0 and roi_pct < 0:
                return {"action":"EXIT","confidence":0.66,"reason":"rsi<47 & adx≥18 & obv- & roi<0"}
        else:
            if rsi > 53 and adx >= 18 and obv > 0 and roi_pct < 0:
                return {"action":"EXIT","confidence":0.66,"reason":"rsi>53 & adx≥18 & obv+ & roi<0"}
        return {"action":"HOLD","confidence":0.55,"reason":"no strong reversal signal"}
    # -------------- /AI REVERSAL BROKER -----------------



    async def ensure_ollama_models(self, models: list[str] | None = None, keep_alive: str = "30m"):
        """
        Тянем недостающие модели на их /api-хостах и прогреваем обе через /v1.
        Параметр models оставлен для совместимости (не используется).
        """
        import httpx

        async def _pull_if_needed(base_api: str, model_name: str):
            base_api = base_api.rstrip("/")
            async with httpx.AsyncClient(timeout=None) as cli:
                try:
                    tags = await cli.get(f"{base_api}/api/tags")
                    have = {m.get("name") for m in tags.json().get("models", [])}
                except Exception as e:
                    logger.warning(f"[AI_INIT] {base_api} /api/tags failed: {e}")
                    have = set()

                if model_name not in have:
                    logger.warning(f"[AI_INIT] pulling {model_name} on {base_api}")
                    await cli.post(f"{base_api}/api/pull", json={"name": model_name}, timeout=None)

        # Pull на СВОИХ /api
        try:
            await _pull_if_needed(self.ollama_primary_api, self.ai_primary_model)
        except Exception as e:
            logger.warning(f"[AI_INIT][PRIMARY] pull failed: {e}")

        try:
            await _pull_if_needed(self.ollama_advisor_api, self.ai_advisor_model)
        except Exception as e:
            logger.warning(f"[AI_INIT][ADVISOR] pull failed: {e}")

        # Warmup через /v1 (короткий JSON-ответ)
        try:
            await self._ask_ollama_json(
                self.ai_primary_model,
                [{"role": "user", "content": "ping"}],
                base_openai_url=self.ollama_primary_openai,
                num_predict=1,
                timeout_s=3.0,
            )
        except Exception as e:
            logger.warning(f"[AI_WARMUP][PRIMARY] ping failed: {e}")

        try:
            await self._ask_ollama_json(
                self.ai_advisor_model,
                [{"role": "user", "content": "ping"}],
                base_openai_url=self.ollama_advisor_openai,
                num_predict=1,
                timeout_s=3.0,
            )
        except Exception as e:
            logger.warning(f"[AI_WARMUP][ADVISOR] ping failed: {e}")


    def _record_best_entry(
        self,
        symbol: str,
        logic: str,
        side: str,
        prob: float,
        features: dict[str, float],
    ) -> None:
        t_key      = datetime.utcnow().replace(second=0, microsecond=0)
        tuple_key  = (symbol, logic, side, t_key)
        cache_key  = (symbol, logic, side)

        if tuple_key in self._best_entry_seen:
            return

        prev_prob = self._best_entry_cache.get(cache_key, (0.0,))[0]
        if prob <= prev_prob:
            return

        self._best_entry_cache[cache_key] = (prob, features)
        self._best_entry_seen.add(tuple_key)

        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "symbol":    symbol,
            "logic":     logic,
            "side":      side,
            "prob":      round(prob, 6),
            **features,
        }
        path = globals().get("ML_BEST_ENTRIES_CSV_PATH", "ml_best_entries.csv")
        is_new = not Path(path).exists()
        with open(path, "a", newline="", encoding="utf-8") as fp:
            import csv
            w = csv.DictWriter(fp, fieldnames=row.keys())
            if is_new:
                w.writeheader()
            w.writerow(row)

    async def _unfreeze_guard(self, ttl: float = 180.0):
        while True:
            await asyncio.sleep(30)
            try:
                now = time.time()
                stale_keys = []
                for key, value in self._evaluated_signals_cache.items():
                    if isinstance(value, dict) and 'time' in value:
                        timestamp = value['time']
                        if now - timestamp > ttl:
                            stale_keys.append(key)
                    elif isinstance(value, (int, float)):
                        if now - value > ttl:
                            stale_keys.append(key)

                if stale_keys:
                    logger.debug(f"[unfreeze_guard] Очистка {len(stale_keys)} устаревших ключей из кэша сигналов.")
                    for key in stale_keys:
                        self._evaluated_signals_cache.pop(key, None)

            except Exception as e:
                logger.error(f"[unfreeze_guard] Ошибка в цикле очистки кэша: {e}", exc_info=True)

    async def adaptive_entry(self, symbol, side, qty, max_entry_timeout):
        return await self.adaptive_squeeze_entry(symbol, side, qty, max_entry_timeout)

    async def adaptive_entry_ws(self, symbol, side, qty, position_idx, max_entry_timeout):
        return await self.adaptive_squeeze_entry_ws(symbol, side, qty, position_idx, max_entry_timeout)

    async def ensure_symbol_meta(self, symbol: str) -> None:
        if symbol in self.qty_step_map and symbol in self.min_qty_map:
            return
        try:
            resp = await asyncio.to_thread(
                lambda: self.session.get_instruments_info(
                    category="linear",
                    symbol=symbol
                )
            )
            info = (resp.get("result", {})
                          .get("list", [{}]))[0]
            filt = info.get("lotSizeFilter", {})
            step = safe_to_float(filt.get("qtyStep", 0.001))
            minq = safe_to_float(filt.get("minOrderQty", step))
            pfilt = info.get("priceFilter", {})
            tick_sz = safe_to_float(pfilt.get("tickSize", DEC_TICK))
            self.price_tick_map[symbol] = tick_sz
            self.qty_step_map[symbol] = step
            self.min_qty_map[symbol]  = minq
            logger.debug("[symbol_meta] %s qtyStep=%s  minQty=%s", symbol, step, minq)
        except Exception as e:
            self.qty_step_map.setdefault(symbol, 0.001)
            self.min_qty_map.setdefault(symbol, 0.001)
            logger.warning("[symbol_meta] fetch failed for %s: %s", symbol, e)

    def _build_entry_features(self, symbol: str) -> list[float]:
        candles = self.shared_ws.candles_data.get(symbol, [])
        vec = [0.0] * len(FEATURE_KEYS)
        try:
            idx = FEATURE_KEYS.index("pct5m")
            vec[idx] = compute_pct(candles, minutes=5)
        except ValueError:
            pass
        try:
            idx = FEATURE_KEYS.index("vol5m")
            vec[idx] = sum_last_vol(candles, minutes=5)
        except ValueError:
            pass
        return vec

    async def _periodic_autotune(self):
        pass

    async def _reload_weights_loop(self):
        while True:
            self._golden_weights  = self._read_weights("golden_feature_weights.csv")
            self._squeeze_weights = self._read_weights("squeeze_feature_weights.csv")
            self._liq_weights     = self._read_weights("liq_feature_weights.csv")
            await asyncio.sleep(600)

    @staticmethod
    def _read_weights(fname: str):
        import pandas as pd, numpy as np, pathlib
        p = pathlib.Path(fname)
        if not p.exists():
            if "golden" in fname:
                return {("__default__", "Buy"): np.array(
                    [0.989, 0.009, 0.002], dtype=float
                )}
            return {}
        df = pd.read_csv(p)
        return {
            (r.symbol, r.side): np.array(
                [getattr(r, c) for c in df.columns if c.startswith("w_")],
                dtype=float
            )
            for r in df.itertuples()
        }

    async def _cleanup_recently_closed(self, interval: int = 15, max_age: int = 60):
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            expired_symbols = [
                symbol for symbol, timestamp in self.recently_closed.items()
                if now - timestamp > max_age
            ]
            for symbol in expired_symbols:
                self.recently_closed.pop(symbol, None)
                logger.debug(f"[Cleanup] Символ {symbol} удален из кэша недавно закрытых.")

    def _load_trade_history(self) -> None:
        if not self.trade_history_file.exists():
            self.trade_history_file.write_text("[]", encoding="utf-8")

    async def _drop_stale_signals(self) -> None:
        while True:
            now = time.time()
            for sym, ts in list(self.pending_signals.items()):
                if now - ts > self.max_signal_age:
                    self.pending_signals.pop(sym, None)
                    logger.debug("[signals] %s expired (%.1fs)", sym, now - ts)
            await asyncio.sleep(5)

    async def _pending_cleaner(self, interval: int = 30):
        while True:
            await asyncio.sleep(interval)
            try:
                now = time.time()
                expired_symbols = [
                    symbol for symbol, timestamp in self.pending_timestamps.items()
                    if now - timestamp > 120
                ]
                if not expired_symbols:
                    continue
                async with self.position_lock:
                    for symbol in expired_symbols:
                        self.pending_orders.pop(symbol, None)
                self.pending_cids.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)
                logger.warning(f"[Pending Cleanup] Ордер для {symbol} завис и был удален из очереди.")
            except Exception as e:
                logger.error(f"[Pending Cleanup] Критическая ошибка в чистильщике: {e}", exc_info=True)

    async def _calc_qty_from_usd(self, symbol: str, usd_amount: float, price: float) -> float:
        """
        Переводим бюджет в USDT в размер позиции с соблюдением шага и minQty.
        Никаких модификаторов (фандинг и пр.) — только размер из candidate/настроек.
        """
        if price <= 0 or usd_amount <= 0:
            return 0.0

        step = float(self.qty_step_map.get(symbol, 1.0))
        min_qty = float(self.min_qty_map.get(symbol, step))
        if step <= 0:
            step = 1.0

        raw_qty = usd_amount / price

        # округление вниз к шагу, без Decimal
        import math
        qty_units = math.floor(raw_qty / step + 1e-12)
        qty = qty_units * step

        # соблюдаем minQty
        if qty < min_qty:
            qty = min_qty

        # техническое округление до нужного количества знаков (по шагу)
        step_str = f"{step:.10f}".rstrip('0').rstrip('.')
        if '.' in step_str:
            dec = len(step_str.split('.')[1])
            qty = float(f"{qty:.{dec}f}")
        else:
            qty = float(int(qty))

        return qty

    def _golden_allowed(self, symbol: str) -> bool:
        now = time.time()
        if now - self._last_golden_ts.get(symbol, 0.0) < self.golden_cooldown_sec:
            return False
        return True

    def _squeeze_allowed(self, symbol: str) -> bool:
        return time.time() - self.last_squeeze_ts.get(symbol, 0.0) >= self.squeeze_cooldown_sec

    def _tune_squeeze(self, feats: dict[str, float]) -> None:
        if not getattr(self, "squeeze_tuner", None):
            return
        vec = np.array([[feats[k] for k in SQUEEZE_KEYS]], np.float32)
        prediction = None

    async def listing_age_minutes(self, symbol: str) -> float:
        now = time.time()
        cached = _listing_age_cache.get(symbol)
        if cached and cached[0] >= self.listing_age_min:
            return cached[0]
        if cached and now - cached[1] < LISTING_AGE_CACHE_TTL_SEC:
            return cached[0]

        async with _listing_sem:
            try:
                resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: self.session.get_instruments_info(
                            category="linear", symbol=symbol
                        )
                    ),
                    timeout=8
                )
                info = resp.get("result", {}).get("list", [{}])[0]
                launch_ms = safe_to_float(info.get("launchTime", 0))
                if launch_ms <= 0:
                    raise ValueError("launchTime missing")
                age_min = (now * 1000 - launch_ms) / 1000 / 60.0
            except Exception as e:
                age_min = 999_999.0
                _listing_age_cache[symbol] = (age_min, now)
                logger.debug("[listing_age] %s REST err (suppressed, marked old): %s", symbol, e)
            else:
                _listing_age_cache[symbol] = (age_min, now)
            return age_min

    def check_liq_cooldown(self, symbol: str) -> bool:
        return self.shared_ws.check_liq_cooldown(symbol)

    def get_total_open_volume_fast(self) -> float:
        total = 0.0
        try:
            for sym, pos in (self.open_positions or {}).items():
                size = safe_to_float(pos.get("volume", 0.0))
                if size <= 0:
                    continue
                price = safe_to_float(
                    pos.get("avg_price")
                    or pos.get("markPrice")
                    or (self.shared_ws.ticker_data.get(sym, {}) or {}).get("lastPrice")
                    or 0.0
                )
                if price > 0:
                    total += abs(size) * price
        except Exception:
            pass
        return total

    def load_ml_models(self,
                    model_path: str = "golden_model_mlx.safetensors",
                    scaler_path: str = "scaler.pkl",
                    input_dim: int = None) -> None:
        self.model = None
        self.feature_scaler = None

        if Path(scaler_path).exists():
            try:
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("[MLX] Scaler загружен из %s", scaler_path)
            except Exception as e:
                logger.warning("[MLX] Не удалось загрузить scaler (%s): %s", scaler_path, e)

        if Path(model_path).exists():
            try:
                in_dim = input_dim if input_dim is not None else len(FEATURE_KEYS)
                self.model = GoldenNetMLX(input_size=in_dim)
                self.model.load_weights(model_path)
                self.model.eval()
                logger.info("[MLX] Модель загружена из %s", model_path)
            except Exception as e:
                logger.error("[MLX] Ошибка загрузки модели (%s): %s", model_path, e)
                self.model = None
        else:
            logger.info("[MLX] Файл модели %s не найден — инференс будет отдавать нули", model_path)

    async def extract_realtime_features(self, symbol: str) -> Optional[Dict[str, float]]:
        last_price = 0.0
        bid1 = 0.0
        ask1 = 0.0

        tdata = self.shared_ws.ticker_data.get(symbol) or {}
        last_price = safe_to_float(tdata.get("lastPrice", 0.0))
        bid1 = safe_to_float(tdata.get("bid1Price", 0.0))
        ask1 = safe_to_float(tdata.get("ask1Price", 0.0))

        if last_price <= 0.0:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_tickers(category="linear", symbol=symbol)
                )
                if resp and resp.get("result", {}).get("list"):
                    tdata = resp["result"]["list"][0]
                    self.shared_ws.ticker_data[symbol] = tdata
                    last_price = safe_to_float(tdata.get("lastPrice", 0.0))
                    bid1 = safe_to_float(tdata.get("bid1Price", 0.0))
                    ask1 = safe_to_float(tdata.get("ask1Price", 0.0))
            except Exception:
                pass

        if last_price <= 0.0:
            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if candles:
                last_price = safe_to_float(candles[-1].get("closePrice", 0.0))
                bid1 = last_price
                ask1 = last_price

        if last_price <= 0.0:
            logger.warning("[features] %s: нет актуальной цены, прерываем", symbol)
            return {}

        spread_pct = (ask1 - bid1) / bid1 * 100.0 if bid1 > 0 else 0.0

        candles = list(self.shared_ws.candles_data.get(symbol, []))
        oi_hist = list(self.shared_ws.oi_history.get(symbol, []))
        cvd_hist = list(self.shared_ws.cvd_history.get(symbol, []))

        pct1m  = compute_pct(self.shared_ws.candles_data.get(symbol, []), 1)
        pct5m  = compute_pct(self.shared_ws.candles_data.get(symbol, []), 5)
        pct15m = compute_pct(self.shared_ws.candles_data.get(symbol, []), 15)

        V1m  = sum_last_vol(self.shared_ws.candles_data.get(symbol, []), 1)
        V5m  = sum_last_vol(self.shared_ws.candles_data.get(symbol, []), 5)
        V15m = sum_last_vol(self.shared_ws.candles_data.get(symbol, []), 15)

        OI_now    = safe_to_float(oi_hist[-1]) if oi_hist else 0.0
        OI_prev1m = safe_to_float(oi_hist[-2]) if len(oi_hist) >= 2 else 0.0
        OI_prev5m = safe_to_float(oi_hist[-6]) if len(oi_hist) >= 6 else 0.0
        dOI1m = (OI_now - OI_prev1m) / OI_prev1m if OI_prev1m > 0 else 0.0
        dOI5m = (OI_now - OI_prev5m) / OI_prev5m if OI_prev5m > 0 else 0.0

        CVD_now    = safe_to_float(cvd_hist[-1]) if cvd_hist else 0.0
        CVD_prev1m = safe_to_float(cvd_hist[-2]) if len(cvd_hist) >= 2 else 0.0
        CVD_prev5m = safe_to_float(cvd_hist[-6]) if len(cvd_hist) >= 6 else 0.0
        CVD1m = CVD_now - CVD_prev1m
        CVD5m = CVD_now - CVD_prev5m

        try:
            sigma5m = safe_to_float(self.shared_ws._sigma_5m(symbol))
        except Exception:
            sigma5m = 0.0

        df_src = candles[-100:] if candles else []
        df = pd.DataFrame(df_src)

        for col in ("closePrice", "highPrice", "lowPrice", "volume"):
            if col not in df.columns:
                df[col] = np.nan
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df[col] = s.ffill().bfill()

        n = len(df)
        close = df["closePrice"] if n else pd.Series(dtype="float64")
        high  = df["highPrice"]  if n else pd.Series(dtype="float64")
        low   = df["lowPrice"]   if n else pd.Series(dtype="float64")

        def _safe_last(series, default=0.0):
            try:
                v = series.iloc[-1]
                return float(v) if pd.notna(v) else default
            except Exception:
                return default

        rsi14 = 50.0
        try:
            rsi14 = _safe_last(ta.rsi(close, length=14), 50.0) if n >= 15 else 50.0
        except Exception:
            rsi14 = 50.0

        sma50 = _safe_last(ta.sma(close, length=50), _safe_last(close, 0.0)) if n >= 50 else _safe_last(close, 0.0)
        try:
            ema20 = _safe_last(ta.ema(close, length=20), sma50) if n >= 20 else sma50
        except Exception:
            ema20 = sma50

        try:
            atr14 = _safe_last(ta.atr(high, low, close, length=14), 0.0) if n >= 15 else 0.0
        except Exception:
            atr14 = 0.0

        bb_width = 0.0
        try:
            if n >= 20:
                bb = ta.bbands(close, length=20)
                if bb is not None and not bb.empty:
                    bbu = _safe_last(bb.iloc[:, 0], 0.0)
                    bbl = _safe_last(bb.iloc[:, 2], 0.0)
                    bb_width = bbu - bbl
        except Exception:
            bb_width = 0.0

        try:
            st_ser = compute_supertrend(df, period=10, multiplier=3) if n > 20 else None
            supertrend_val = bool(_safe_last(st_ser, 0.0)) if st_ser is not None else False
        except Exception:
            supertrend_val = False
        supertrend_num = 1 if supertrend_val else -1

        try:
            adx14 = _safe_last(ta.adx(high, low, close, length=14)["ADX_14"], 0.0) if n >= 15 else 0.0
        except Exception:
            adx14 = 0.0
        try:
            cci20 = _safe_last(ta.cci(high, low, close, length=20), 0.0) if n >= 20 else 0.0
        except Exception:
            cci20 = 0.0

        macd_val, macd_signal = 0.0, 0.0
        try:
            if n >= 35:
                macd_df = ta.macd(close, fast=12, slow=26, signal=9)
                if macd_df is not None and not macd_df.empty and macd_df.shape[1] >= 3:
                    macd_val   = _safe_last(macd_df.iloc[:, 0], 0.0)
                    macd_signal= _safe_last(macd_df.iloc[:, 2], 0.0)
        except Exception as e:
            logger.debug("[TA] MACD failed for %s: %s", symbol, e)

        GS_pct4m = compute_pct(self.shared_ws.candles_data.get(symbol, []), 4)
        GS_vol4m = sum_last_vol(self.shared_ws.candles_data.get(symbol, []), 4)
        base_OI = safe_to_float(oi_hist[-5]) if len(oi_hist) >= 5 else (OI_now if OI_now > 0 else 1.0)
        GS_dOI4m = (OI_now - base_OI) / base_OI if base_OI > 0 else 0.0
        base_CVD = safe_to_float(cvd_hist[-5]) if len(cvd_hist) >= 5 else CVD_now
        GS_cvd4m = CVD_now - base_CVD
        GS_supertrend_flag = supertrend_num
        try:
            GS_cooldown_flag = int(not self._golden_allowed(symbol))
        except Exception:
            GS_cooldown_flag = 1

        mean_V5m = (V5m / 5.0) if V5m > 0 else 1e-8
        SQ_power = abs(pct5m) * abs((V1m - mean_V5m) / max(1e-8, mean_V5m) * 100.0)
        try:
            thr_pct = float(getattr(self, "squeeze_threshold_pct", 0.0))
            power_min = float(getattr(self, "squeeze_power_min", 0.0))
        except Exception:
            thr_pct, power_min = 0.0, 0.0
        SQ_strength = int(abs(pct5m) >= thr_pct and SQ_power >= power_min)
        try:
            recent_liq_vals = [safe_to_float(v_usdt) for (ts, s, v_usdt, price_liq) in self.liq_buffers.get(symbol, []) if time.time() - ts <= 10]
        except Exception:
            recent_liq_vals = []
        SQ_liq10s = float(np.nansum(recent_liq_vals)) if recent_liq_vals else 0.0
        try:
            SQ_cooldown_flag = int(not self._squeeze_allowed(symbol))
        except Exception:
            SQ_cooldown_flag = 1

        try:
            buf = self.liq_buffers.get(symbol, [])
            recent_all = [(ts, s, v_usdt, price_liq) for (ts, s, v_usdt, price_liq) in buf if time.time() - ts <= 10]
            same_side = [safe_to_float(v_usdt) for (ts, s, v_usdt, _price) in recent_all if recent_all and s == recent_all[-1][1]]
            LIQ_cluster_val10s = float(np.nansum(same_side)) if same_side else 0.0
            LIQ_cluster_count10s = int(len(same_side))
            LIQ_direction = 1 if (recent_all and recent_all[-1][1] == "Buy") else -1
        except Exception:
            LIQ_cluster_val10s = 0.0
            LIQ_cluster_count10s = 0
            LIQ_direction = -1

        try:
            LIQ_cooldown_flag = int(not self.check_liq_cooldown(symbol))
        except Exception:
            LIQ_cooldown_flag = 1

        try:
            now_ts = dt.datetime.now()
        except Exception:
            import datetime as _dt
            now_ts = _dt.datetime.now()
        hour_of_day = int(now_ts.hour)
        day_of_week = int(now_ts.weekday())
        month_of_year = int(now_ts.month)

        try:
            avgVol30m = safe_to_float(self.shared_ws.get_avg_volume(symbol, 30))
        except Exception:
            avgVol30m = 0.0
        tail_oi = [safe_to_float(x) for x in oi_hist[-30:]] if oi_hist else []
        avgOI30m = float(np.nanmean(tail_oi)) if tail_oi else 0.0
        deltaCVD30m = CVD_now - (safe_to_float(cvd_hist[-31]) if len(cvd_hist) >= 31 else 0.0)

        features: Dict[str, float] = {
            "price": last_price, "pct1m": pct1m, "pct5m": pct5m, "pct15m": pct15m,
            "vol1m": V1m, "vol5m": V5m, "vol15m": V15m, "OI_now": OI_now, "dOI1m": dOI1m, "dOI5m": dOI5m,
            "spread_pct": spread_pct, "sigma5m": sigma5m, "CVD1m": CVD1m, "CVD5m": CVD5m,
            "rsi14": rsi14, "sma50": sma50, "ema20": ema20, "atr14": atr14, "bb_width": bb_width,
            "supertrend": supertrend_num, "cci20": cci20, "macd": macd_val, "macd_signal": macd_signal,
            "avgVol30m": avgVol30m, "avgOI30m": avgOI30m, "deltaCVD30m": deltaCVD30m, "adx14": adx14,
            "GS_pct4m": GS_pct4m, "GS_vol4m": GS_vol4m, "GS_dOI4m": GS_dOI4m, "GS_cvd4m": GS_cvd4m,
            "GS_supertrend": GS_supertrend_flag, "GS_cooldown": GS_cooldown_flag,
            "SQ_pct1m": pct1m, "SQ_pct5m": pct5m, "SQ_vol1m": V1m, "SQ_vol5m": V5m, "SQ_dOI1m": dOI1m,
            "SQ_spread_pct": spread_pct, "SQ_sigma5m": sigma5m, "SQ_liq10s": SQ_liq10s, "SQ_cooldown": SQ_cooldown_flag,
            "SQ_power": SQ_power, "SQ_strength": SQ_strength,
            "LIQ_cluster_val10s": LIQ_cluster_val10s, "LIQ_cluster_count10s": LIQ_cluster_count10s,
            "LIQ_direction": LIQ_direction, "LIQ_pct1m": pct1m, "LIQ_pct5m": pct5m, "LIQ_vol1m": V1m,
            "LIQ_vol5m": V5m, "LIQ_dOI1m": dOI1m, "LIQ_spread_pct": spread_pct, "LIQ_sigma5m": sigma5m,
            "LIQ_golden_flag": GS_cooldown_flag, "LIQ_squeeze_flag": SQ_cooldown_flag, "LIQ_cooldown": LIQ_cooldown_flag,
            "hour_of_day": hour_of_day, "day_of_week": day_of_week, "month_of_year": month_of_year,
        }

        for k in FEATURE_KEYS:
            features.setdefault(k, 0.0)

        return features

    async def _capture_training_sample(self, symbol: str, pnl_pct: float) -> None:
        try:
            feats = await self.extract_realtime_features(symbol)
        except Exception as exc:
            logger.debug("[ML] feature extraction failed: %s", exc)
            logger.exception("[ML] _capture_training_sample failed")
            return
        self.training_data.append({
            "features": [feats[k] for k in FEATURE_KEYS],
            "target":   pnl_pct,
        })

    def compute_strength(self, close_price: float,
                        volume: float,
                        oi: float,
                        cvd: float) -> float:
        return (volume / (oi + 1e-8)) * cvd

    async def build_and_save_trainset(self,
                                    csv_path: str,
                                    scaler_path: str,
                                    symbol: str | list[str],
                                    future_horizon: int = 3,
                                    future_thresh: float = 0.005):
        try:
            await self.shared_ws.backfill_history()
        except Exception as e:
            logger.warning(f"[DatasetBuilder] backfill_history failed: {e}")

        symbols_list = symbol if isinstance(symbol, (list, tuple)) else [symbol]

        rows: list[dict] = []
        min_index = 50

        for sym in symbols_list:
            candles_orig = self.shared_ws.candles_data[sym]
            oi_orig      = self.shared_ws.oi_history[sym]
            cvd_orig     = self.shared_ws.cvd_history[sym]

            candles_maxlen = getattr(candles_orig, "maxlen", len(candles_orig))
            oi_maxlen      = getattr(oi_orig,      "maxlen", len(oi_orig))
            cvd_maxlen     = getattr(cvd_orig,     "maxlen", len(cvd_orig))

            bars_all = list(candles_orig)
            oi_all   = list(oi_orig)
            cvd_all  = list(cvd_orig)

            max_i = len(bars_all) - future_horizon
            if max_i <= min_index:
                logger.warning(f"[DatasetBuilder] Not enough bars for {sym}: "
                            f"have={len(bars_all)}, need>{min_index+future_horizon}")
                continue

            for i in range(min_index, max_i):
                self.shared_ws.candles_data[sym] = bars_all[:i + 1]
                self.shared_ws.oi_history[sym]    = oi_all[:i + 1]
                self.shared_ws.cvd_history[sym]   = cvd_all[:i + 1]

                feats = await self.extract_realtime_features(sym)

                if not feats:
                    continue

                future_price  = bars_all[i + future_horizon]["closePrice"]
                current_price = bars_all[i]["closePrice"]
                ret = future_price / current_price - 1
                feats["label"] = int(ret > future_thresh)

                rows.append(feats)

            self.shared_ws.candles_data[sym] = deque(bars_all, maxlen=candles_maxlen)
            self.shared_ws.oi_history[sym]    = deque(oi_all,   maxlen=oi_maxlen)
            self.shared_ws.cvd_history[sym]   = deque(cvd_all,  maxlen=cvd_maxlen)

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("[DatasetBuilder] No training rows generated. "
                            "Проверьте объём истории и future_horizon.")

        feature_cols = [c for c in df.columns if c != "label"]
        scaler = StandardScaler().fit(df[feature_cols])
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")

        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
        df_scaled.to_csv(csv_path, index=False)
        print(f"✅ trainset.csv saved: {csv_path}, shape={df_scaled.shape}")

    async def predict_action(self, symbol: str) -> str:
        cid = new_cid()
        try:
            features = await self.extract_realtime_features(symbol)
            if not features:
                logger.info("[DECIDE][%s] %s -> HOLD (no features)", cid, symbol)
                return "HOLD"

            if getattr(self, "strategy_mode", "") == "golden_only":
                adx = safe_to_float(features.get("adx14", 0.0))
                rsi = safe_to_float(features.get("rsi14", 0.0))
                if adx < 25.0 or rsi > 80.0:
                    logger.info("[DECIDE][%s] %s -> HOLD (golden filter adx=%.1f rsi=%.1f)",
                                cid, symbol, adx, rsi)
                    return "HOLD"

            action = "HOLD"
            score_str = "n/a"

            if getattr(self, "MLX_model", None):
                vector = [features.get(k, 0.0) for k in FEATURE_KEYS]
                x = np.array(vector, dtype=np.float32).reshape(1, -1)

                pred = self.MLX_model.predict({"input": x})
                y = pred.get("output")

                if isinstance(y, np.ndarray):
                    if y.ndim == 2 and y.shape[1] == 3:
                        idx = int(np.argmax(y[0]))
                        action = ["BUY", "SELL", "HOLD"][idx]
                        score_str = j(y[0].tolist())
                    elif y.ndim == 2 and y.shape[1] == 1:
                        score = float(y[0, 0])
                        buy_thr  = float(getattr(self, "buy_threshold", 0.6))
                        sell_thr = float(getattr(self, "sell_threshold", -0.6))
                        action = "BUY" if score >= buy_thr else "SELL" if score <= sell_thr else "HOLD"
                        score_str = f"{score:.3f}"
                    else:
                        action = "HOLD"
                else:
                    action = "HOLD"
            else:
                action = "HOLD"

            logger.info("[DECIDE][%s] %s -> %s | rsi=%.1f adx=%.1f score=%s",
                        cid, symbol, action,
                        safe_to_float(features.get("rsi14", 0.0)),
                        safe_to_float(features.get("adx14", 0.0)),
                        score_str)
            return action

        except Exception as e:
            logger.error("[MLX][%s] predict_action(%s) failed: %s", cid, symbol, e, exc_info=True)
            return "HOLD"


    async def log_trade_for_ml(self, symbol: str, entry_data: dict, exit_data: dict):
        try:
            features = await self.extract_realtime_features(symbol)

            global FEATURE_KEYS, INPUT_DIM
            if not FEATURE_KEYS:
                FEATURE_KEYS = list(features)
                INPUT_DIM = len(FEATURE_KEYS)
                logger.info("[ML] FEATURE_KEYS initialised with %d fields", INPUT_DIM)

            missing = [k for k in FEATURE_KEYS if k not in features]
            extra   = [k for k in features     if k not in FEATURE_KEYS]
            if missing:
                logger.debug("[ML] %s missing features: %s", symbol, missing)
            if extra:
                logger.debug("[ML] %s extra   features: %s", symbol, extra)

            vector = [features.get(k, 0.0) for k in FEATURE_KEYS]

            if entry_data["side"].lower() == "buy":
                pnl = (exit_data["price"] - entry_data["price"]) / entry_data["price"] * 1000.0
            else:
                pnl = (entry_data["price"] - exit_data["price"]) / entry_data["price"] * 1000.0

            label = 0 if pnl < 0 else 1 if pnl > 1 else 2

            self.training_data.append({"features": vector, "target": label})

            if time.time() - getattr(self, "last_retrain", 0) > 3600 and len(self.training_data) > 100:
                self.last_retrain = time.time()
                asyncio.create_task(self.retrain_models())

        except Exception as e:
            logger.error("[ML] Trade logging error: %s", e)

    async def retrain_models(self) -> None:
        try:
            data = getattr(self, "training_data", [])
            if not data or len(data) < 200:   # поднимите порог, когда данных станет много
                return

            # Собираем матрицу признаков и таргет
            X = np.asarray([rec["features"] for rec in data], dtype=np.float32)
            y = np.asarray([rec["target"]   for rec in data], dtype=np.float32)

            if X.ndim != 2 or X.shape[1] != len(FEATURE_KEYS):
                logger.warning("[MLX] retrain_models: неверная форма X=%s", X.shape if hasattr(X, "shape") else type(X))
                return

            # Маска валидных строк
            mask = ~(np.isnan(X).any(1) | np.isinf(X).any(1) | np.isnan(y) | np.isinf(y))
            X, y = X[mask], y[mask]
            if X.size == 0:
                return

            # Преобразуем к формату, который ожидает train_golden_model_mlx
            pairs = [{"features": X[i], "target": y[i]} for i in range(len(y))]

            loop = asyncio.get_running_loop()

            def _train_and_save():
                # ВАЖНО: train_golden_model_mlx должна вернуть (model, scaler)
                model, scaler = MLXInferencer.train_golden_model_mlx(
                    pairs, num_epochs=30, lr=1e-3, batch_size=512
                )

                # Сохраняем веса и scaler
                from safetensors.numpy import save_file as save_safetensors
                import joblib, os
                os.makedirs(".", exist_ok=True)
                save_safetensors(model.state_dict_numpy(), "golden_model_mlx.safetensors")
                joblib.dump(scaler, "scaler.pkl")
                return True

            # Обучаем в пуле потоков (чтобы не блокировать event loop)
            ok = await loop.run_in_executor(None, _train_and_save)
            if not ok:
                logger.warning("[MLX] retrain_models: train/save returned falsy")
                return

            # Горячая перезагрузка инференсера
            try:
                self.ml_inferencer = MLXInferencer(
                    model_path="golden_model_mlx.safetensors",
                    scaler_path="scaler.pkl"
                )
                logger.info("[retrain] MLX inferencer refreshed")
            except Exception as _e:
                logger.warning("[retrain] could not init MLXInferencer: %s", _e)

            # Очищаем буфер тренировок
            try:
                from collections import deque
                if isinstance(self.training_data, deque):
                    self.training_data.clear()
                else:
                    self.training_data = []
            except Exception:
                self.training_data = []

            self.last_retrain = time.time()
            logger.info("[MLX] retrain complete: samples=%d (saved weights + scaler)", len(pairs))

        except Exception as e:
            logger.error(f"[MLX] retrain_models failed: {e}", exc_info=True)

    def load_model(self):
        model_path = f"models/golden_model_{self.user_id}.pt"
        if os.path.exists(model_path):
            logger.info(f"[ML] Model loaded for user {self.user_id}")
        else:
            logger.info(f"[ML] No pretrained model for user {self.user_id}")

    def save_model(self):
        model_path = f"models/golden_model_{self.user_id}.pt"
        os.makedirs("models", exist_ok=True)
        logger.info(f"[ML] Model saved for user {self.user_id}")

    def append_training_sample(self, features: list[float], target: float):
        self.training_data.append({"features": features, "target": target})

    async def retrain_model(self):
        if len(self.training_data) < 300:
            logger.warning("[ML] Not enough data to retrain")
            return
        self.save_model()

    def predict_entry_quality(self, features: Sequence[float]) -> float:
        if not features:
            return 0.0

        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != len(FEATURE_KEYS):
            logger.warning("predict_entry_quality: expected %d features, got %d",
                        len(FEATURE_KEYS), x.shape[1])
            return 0.0

        inf = getattr(self, "ml_inferencer", None)
        if inf is None:
            return 0.0
        if getattr(inf, "scaler", None) is not None:
            try:
                x = inf.scaler.transform(x).astype(np.float32)
            except Exception as e:
                logger.warning("predict_entry_quality: scaler.transform failed: %s", e)
                return 0.0

        if getattr(inf, "model", None) is None:
            return 0.0

        # MLX model inference would go here
        return 0.0

    def extract_features(self, symbol: str) -> list[float]:
        recent_1m = self.shared_ws.candles_data.get(symbol, [])
        if len(recent_1m) < 6:
            return []

        old_close = safe_to_float(recent_1m[-6]["closePrice"])
        new_close = safe_to_float(recent_1m[-1]["closePrice"])
        pct_change = (new_close - old_close) / old_close * 100 if old_close > 0 else 0

        vol_change = safe_to_float(recent_1m[-1]["volume"]) - safe_to_float(recent_1m[-6]["volume"])
        oi_change = self.shared_ws.get_oi_change(symbol)

        cvd = self.shared_ws.get_recent_cvd_delta(symbol)
        supertrend_trend = self.shared_ws.get_trend_state(symbol)

        return [
            pct_change,
            vol_change,
            oi_change,
            cvd,
            supertrend_trend,
        ]

    async def optimize_golden_params(self):
        try:
            df = pd.read_csv("trades_for_training.csv")
        except FileNotFoundError:
            logger.warning("[optimize] trades_for_training.csv not found, skipping optimization")
            return

        if "event" not in df.columns:
            logger.warning("[optimize] 'event' column missing in trades_for_training.csv – skipping optimization")
            return

        df = df[df["event"] == "open"]
        if df.empty:
            logger.warning("[optimize] No open trades in training data, skipping optimization")
            return
        df["win"] = (df["pnl_pct"] > 0).astype(int)

        best_params = {}
        symbols = df["symbol"].unique()

        for symbol in symbols:
            for side in ["Buy", "Sell"]:
                sub = df[(df["symbol"] == symbol) & (df["side"] == side)]
                if len(sub) < 30:
                    continue
                best = {"winrate": 0, "p0": None, "v0": None, "o0": None}
                p_range = np.arange(0.1, 2.1, 0.1)
                v_range = np.arange(10, 201, 10)
                o_range = np.arange(0.1, 2.1, 0.1)
                for p0, v0, o0 in product(p_range, v_range, o_range):
                    if side == "Buy":
                        mask = (
                            (sub["price_change"] >= p0)
                            & (sub["volume_change"] >= v0)
                            & (sub["oi_change"] >= o0)
                        )
                    else:
                        mask = (
                            (sub["price_change"] <= -p0)
                            & (sub["volume_change"] >= v0)
                            & (sub["oi_change"] >= o0)
                        )
                    filt = sub[mask]
                    if len(filt) < 30:
                        continue
                    winrate = filt["win"].mean()
                    if winrate > best["winrate"]:
                        best = {"winrate": winrate, "p0": p0, "v0": v0, "o0": o0}
                if best["p0"] is not None:
                    params = {
                        "period_iters": 4,
                        "price_change": best["p0"],
                        "volume_change": best["v0"],
                        "oi_change": best["o0"],
                    }
                    self.golden_param_store[(symbol, side)] = params
                    best_params[(symbol, side)] = params
                    logger.info(f"[optimize] {symbol} {side} params: {params}, winrate={best['winrate']:.2f}")

        if best_params:
            with open("golden_params.csv", "w", newline="") as f:
                fieldnames = ["symbol", "side", "period_iters", "price_change", "volume_change", "oi_change"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for (symbol, side), params in best_params.items():
                    writer.writerow({
                        "symbol": symbol,
                        "side": side,
                        "period_iters": params["period_iters"],
                        "price_change": params["price_change"],
                        "volume_change": params["volume_change"],
                        "oi_change": params["oi_change"],
                    })

    async def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        candles = list(self.shared_ws.candles_data.get(symbol, []))[-100:]
        if len(candles) < period + 1:
            return 0.0
        try:
            df = pd.DataFrame(candles)
            df['highPrice'] = pd.to_numeric(df['highPrice'])
            df['lowPrice'] = pd.to_numeric(df['lowPrice'])
            df['closePrice'] = pd.to_numeric(df['closePrice'])
            atr_series = ta.atr(df['highPrice'], df['lowPrice'], df['closePrice'], length=period)
            return atr_series.iloc[-1] if not atr_series.empty else 0.0
        except Exception as e:
            logger.warning(f"[_calculate_atr] Ошибка расчета ATR для {symbol}: {e}")
            return 0.0

    async def _close_position_market(self, symbol: str, qty_to_close: float, reason: str):
        async with self.position_lock:
            pos = self.open_positions.get(symbol)
            if not pos:
                return

            side_to_close = "Sell" if pos['side'] == "Buy" else "Buy"

            logger.info(f"Закрытие {qty_to_close} {symbol} по рынку. Причина: {reason}")
            try:
                await self.place_unified_order(symbol, side_to_close, qty_to_close, "Market", comment=f"Close: {reason}")
            except Exception as e:
                logger.error(f"Не удалось закрыть {qty_to_close} {symbol} по причине '{reason}': {e}")


    # === FUNDING HELPERS (вставь в класс TradingBot) ===================

    def _funding__extract_rate(self, symbol: str, features: dict | None) -> float | None:
        """
        Достаём funding rate из доступных источников:
        - features['fundingRate'] / ['funding_rate'] / ['funding']
        - shared_ws.funding_history[symbol] (берём последний)
        Возвращаем float вида 0.0001 = 0.01% (bybit формát).
        """
        rate = None
        # 1) Из фич (если уже туда кладёшь)
        if isinstance(features, dict):
            for k in ("fundingRate", "funding_rate", "funding"):
                v = features.get(k)
                if v is not None:
                    try:
                        rate = float(v)
                        break
                    except (TypeError, ValueError):
                        pass

        # 2) Из WS-кэша
        if rate is None and hasattr(self, "shared_ws") and hasattr(self.shared_ws, "funding_history"):
            hist = self.shared_ws.funding_history.get(symbol)
            if hist:
                # поддержим deque,list,dict
                last = hist[-1] if hasattr(hist, "__getitem__") else None
                if isinstance(last, dict):
                    last = last.get("rate")
                try:
                    rate = float(last)
                except (TypeError, ValueError):
                    rate = None

        return rate


    def _funding__bucket(self, rate: float | None) -> dict:
        """
        Классификация по величине фандинга (bybit: доля на 8ч).
        HOT пороги можно подвинуть — базово 0.05% и 0.025%.
        """
        if rate is None:
            return {"bucket": "unknown", "hot": False, "abs": None, "sign": 0, "pct8h": None, "bps8h": None}

        abs_r = abs(rate)
        sign = 1 if rate > 0 else (-1 if rate < 0 else 0)
        pct = rate * 100.0            # 0.0001 -> 0.01%
        bps = rate * 10_000.0         # 0.0001 -> 1 bps

        if abs_r >= 0.005:           # >= 0.5%
            bucket = "hot"
            hot = True
        elif abs_r >= 0.001:        # >= 0.1%
            bucket = "warm"
            hot = False
        else:
            bucket = "cool"
            hot = False

        return {"bucket": bucket, "hot": hot, "abs": abs_r, "sign": sign, "pct8h": pct, "bps8h": bps}


    def _funding_snapshot(self, symbol: str, features: dict | None = None) -> dict:
        """
        Готовый снапшот по фандингу для логики/AI/логов.
        Структура:
        {
            "funding_rate": 0.0001,
            "funding_pct8h": 0.01,
            "funding_bps8h": 1.0,
            "funding_bucket": "hot|warm|cool|unknown",
            "funding_sign": -1|0|1,
            "funding_hot": True|False
        }
        """
        rate = self._funding__extract_rate(symbol, features)
        buck = self._funding__bucket(rate)
        return {
            "funding_rate": rate,
            "funding_pct8h": buck["pct8h"],
            "funding_bps8h": buck["bps8h"],
            "funding_bucket": buck["bucket"],
            "funding_sign": buck["sign"],
            "funding_hot": buck["hot"],
        }


    def _apply_funding_to_features(self, symbol: str, features: dict) -> dict:
        """
        Обогащаем features полями фандинга (in-place + возврат снапшота).
        Безопасно, не требует новых конфигов.
        """
        snap = self._funding_snapshot(symbol, features)
        try:
            features.update(snap)
        except Exception:
            pass
        return snap


    def _apply_funding_to_candidate(self, candidate: dict, funding_snap: dict) -> None:
        """
        Вкладываем краткую сводку по фандингу в кандидат (для логов/AI).
        Ничего «не ломаем», просто добавляем блок.
        """
        try:
            # base_metrics
            fm = {
                "funding_rate": funding_snap.get("funding_rate"),
                "funding_pct8h": funding_snap.get("funding_pct8h"),
                "funding_bucket": funding_snap.get("funding_bucket"),
                "funding_sign": funding_snap.get("funding_sign"),
                "funding_hot": funding_snap.get("funding_hot"),
            }
            # если уже есть base_metrics — добавим, иначе создадим
            if "base_metrics" in candidate and isinstance(candidate["base_metrics"], dict):
                candidate["base_metrics"].update(fm)
            else:
                candidate["base_metrics"] = fm

            # для наглядности в reason / justification (опционально)
            note = f"funding:{fm['funding_bucket']}"
            if "justification" in candidate and candidate["justification"]:
                candidate["justification"] += f" | {note}"
            else:
                candidate["justification"] = note
        except Exception:
            pass
    # === END FUNDING HELPERS ============================================


    # [FINAL VERSION] Высокочастотный менеджер позиций с умным логированием
    async def manage_open_position(self, symbol: str):
        """
        [V8 - High-Frequency Guardian with Smart Logging] Автономная, быстрая задача
        для управления одной позицией. Использует текущие настройки трейлинга и
        логирует обновления стопа, не создавая лишнего шума.
        """
        # гарантируем запуск AI-воркеров (1 раз)
        self._ensureai_rev_workers()

        logger.info(f"🛡️ [Position Guardian] Активирован для {symbol}. Интервал: {self.trailing_update_interval_sec}с.")

        if not self.intraday_trailing_enabled:
            logger.info(f"[Position Guardian] Трейлинг для {symbol} отключен в настройках.")
            return

        # Заводим счетчик обновлений стопа прямо в данных позиции
        if symbol in self.open_positions:
            self.open_positions[symbol]['stop_update_count'] = 0

        while symbol in self.open_positions:
            await asyncio.sleep(self.trailing_update_interval_sec)

            pos_data = self.open_positions.get(symbol)
            if not pos_data: break

            last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
            if not last_price > 0: continue

            entry_price = safe_to_float(pos_data.get("avg_price", 0))
            side = pos_data.get("side")
            # Леверидж со стрима бывает пустой/0 → из-за этого трейлинг вообще не стартует.
            # Дефолтим к разумному значению (то же делал в adopt_existing_position).
            leverage = safe_to_float(pos_data.get("leverage", 0))
            if leverage <= 0:
                leverage = 10.0
            if not (entry_price > 0 and side):
                continue


            current_roi = (
                ((last_price - entry_price) / entry_price) * 100 * leverage
                if side == "Buy"
                else ((entry_price - last_price) / entry_price) * 100 * leverage
            )
            pos_data['pnl'] = current_roi

                # === AI reversal: постановка задачи и чтение вердикта (не блокируем цикл) ===
            if getattr(self, "enable_ai_reversal_guard", True):
                now = time.time()
                # быстрый хук: решаем, стоит ли вообще проверять разворот (дешёвые признаки)
                probe = False
                min_dd_bp = int(getattr(self, "ai_rev_probe_min_dd_bp", 35))  # 0.35% в б.п. (с плечом)
                if (current_roi * 100.0) <= -min_dd_bp:
                    probe = True
                else:
                    f_quick = await self.extract_realtime_features(symbol)
                    atr = safe_to_float((f_quick or {}).get("atr") or (f_quick or {}).get("atr14") or 0)
                    if atr > 0:
                        adverse = (entry_price - last_price) if side == "Buy" else (last_price - entry_price)
                        if adverse >= float(getattr(self, "ai_rev_k_atr", 1.2)) * atr:
                            probe = True
                # если хотим проверить — ставим в очередь (с локальным кулдауном, чтобы не заспамить)
                due = (self.ai_rev_pending.get(symbol) or 0)
                if probe and now >= due:
                    try:
                        f = f_quick if 'f_quick' in locals() and f_quick else (await self.extract_realtime_features(symbol))
                        self.ai_rev_queue.put_nowait({
                            "symbol": symbol, "side": side,
                            "roi_pct": float(current_roi),
                            "features": f or {}
                        })
                        self.ai_rev_pending[symbol] = now + 10.0  # повторный запрос не раньше чем через 10с
                        logger.debug("[AI_REVERSAL] queued %s", symbol)
                    except asyncio.QueueFull:
                        pass
                    except Exception:
                        logger.debug("[AI_REVERSAL] queue put failed", exc_info=True)
                # читаем свежий вердикт (если уже готов)
                res = self.ai_rev_results.get(symbol)
                if res:
                    verdict = res.get("verdict") or {}
                    conf = float(verdict.get("confidence", 0.0))
                    need = float(getattr(self, "ai_reversal_conf_threshold", 0.62))
                    if verdict.get("action") == "EXIT" and conf >= need:
                        reason = verdict.get("reason", "ai_reversal")
                        size = safe_to_float(pos_data.get("size") or pos_data.get("volume") or pos_data.get("qty") or 0)
                        if size > 0:
                            logger.info("[AI_REVERSAL] %s: EXIT (conf=%.2f ≥ %.2f) — %s", symbol, conf, need, reason)
                            await self._close_position_market(symbol, size, reason)
                            break
            # === /AI reversal ===

            # Учитываем флоты — пусть минимальная пыль не мешает старту трейлинга
            if (current_roi + 1e-6) < self.trailing_start_pct:
                continue


            if side == "Buy":
                new_stop_price = last_price * (1 - self.trailing_gap_pct / 100.0)
            else:
                new_stop_price = last_price * (1 + self.trailing_gap_pct / 100.0)

            current_stop_price = self.last_stop_price.get(symbol)

            is_update_needed = False
            if current_stop_price is None:
                is_update_needed = True
            elif side == "Buy" and new_stop_price > current_stop_price:
                is_update_needed = True
            elif side == "Sell" and new_stop_price < current_stop_price:
                is_update_needed = True

            if is_update_needed:
                # [ИСПРАВЛЕНИЕ] Логика теперь находится здесь, после успешной установки стопа
                await self.set_or_amend_stop_loss(symbol, new_stop_price)
                
                # Проверяем, что стоп действительно обновился (set_or_amend_stop_loss обновляет self.last_stop_price)
                if self.last_stop_price.get(symbol) == new_stop_price or abs(self.last_stop_price.get(symbol, 0) - new_stop_price) < 1e-9:
                    update_count = pos_data.get('stop_update_count', 0) + 1
                    pos_data['stop_update_count'] = update_count
                    
                    # Логируем только первое обновление и каждое 5-е после этого
                    if update_count == 1 or update_count % 5 == 0:
                        await self.log_trade(
                            symbol=symbol, side=side, avg_price=entry_price, volume=pos_data.get('volume', 0),
                            action="trailing_update", result="success"
                        )

        logger.info(f"🛡️ [Position Guardian] Позиция {symbol} закрыта. Хранитель завершает работу.")


    # [HELPER] Исполнитель установки/изменения стопа
    async def set_or_amend_stop_loss(self, symbol: str, new_stop_price: float):
        now = time.time()
        if now - self._last_trailing_ts.get(symbol, 0.0) < 0.5:
            return
        
        pos_data = self.open_positions.get(symbol)
        if not pos_data: return

        try:
            tick = self.price_tick_map.get(symbol, 1e-6)
            # псевдокод внутри try:
            
            if not tick > 0: tick = 1e-6
            final_stop_price = round(math.floor(new_stop_price / tick) * tick, 8)

            if abs(final_stop_price - self.last_stop_price.get(symbol, 0)) < tick:
                return

            pos_idx = pos_data.get("pos_idx", 1 if pos_data.get("side") == 'Buy' else 2)
            
            async with _TRADING_STOP_SEM:
                await asyncio.to_thread(
                    lambda: self.session.set_trading_stop(
                        category="linear", symbol=symbol, positionIdx=pos_idx,
                        stopLoss=f"{final_stop_price:.8f}".rstrip('0').rstrip('.'),
                        triggerBy="LastPrice", timeInForce="GTC"
                    )
                )

            self.last_stop_price[symbol] = final_stop_price
            self._last_trailing_ts[symbol] = now
            logger.info(f"✅ [Guardian] {symbol} стоп обновлен на {final_stop_price}")

        except InvalidRequestError as e:
            if e.status_code == 34040:
                self.last_stop_price[symbol] = final_stop_price
                self._last_trailing_ts[symbol] = now
            elif e.status_code == 10001:
                self._purge_symbol_state(symbol)
            else:
                logger.warning(f"[Guardian] {symbol} ошибка API: {e}")
        except Exception as e:
            logger.error(f"[Guardian] {symbol} критическая ошибка: {e}", exc_info=True)


    async def set_trailing_stop(self, symbol: str, avg_price: float, pnl_pct: float, side: str) -> bool:
        """
        Трейлинг-стоп:
        - Исправлено деление: gap_pct / 100.0 (раньше было /1000 и стоп ставился слишком близко).
        - Не даём стопу ухудшаться.
        - Учитываем биржевой tick, логируем только успешные апдейты.
        - Кулдаун между апдейтами ~1.5с, чтобы не спамить API.
        """
        pos = self.open_positions.get(symbol)
        if not pos or safe_to_float(pos.get("volume", 0)) <= 0:
            return False

        now = time.time()
        if now - self._last_trailing_ts.get(symbol, 0.0) < 1.5:
            return False

        try:
            pos_idx = pos.get("pos_idx", 1 if side == 'Buy' else 2)
            gap_pct = float(getattr(self, "trailing_gap_pct", 2.7))  # в процентах (ROI), напр. 2.7
            last_price = float(pos.get("last_price") or pos.get("mark_price") or 0.0)
            if last_price <= 0.0:
                # Последний апдейт цены не подтянулся — ничего не делаем
                return False

            # Рассчитываем «сырой» уровень стопа от текущей цены и GAP (в процентах!)
            if side.lower() == "buy":
                raw_price = last_price * (1.0 - gap_pct / 100.0)
            else:  # sell
                raw_price = last_price * (1.0 + gap_pct / 100.0)

            # Округляем по тик-сайзу
            tick = float(self.price_tick_map.get(symbol, 1e-6))
            if tick <= 0:
                tick = 1e-6

            import math
            stop_price = math.floor(raw_price / tick) * tick
            stop_price = float(f"{stop_price:.8f}")  # приводим к «чистому» виду

            # Не ухудшаем стоп
            prev_stop = self.last_stop_price.get(symbol)
            if prev_stop is not None:
                worse_for_buy = (side.lower() == "buy" and stop_price < prev_stop)
                worse_for_sell = (side.lower() == "sell" and stop_price > prev_stop)
                if worse_for_buy or worse_for_sell:
                    return False
                # если сдвиг меньше 1 тик — не трогаем
                if abs(stop_price - prev_stop) < tick:
                    return False

            # Отправляем на биржу
            async with _TRADING_STOP_SEM:
                if self.mode == "real":
                    await self.place_set_trailing_stop_ws(
                        symbol=symbol,
                        position_idx=pos_idx,
                        stop_loss=f"{stop_price:.8f}".rstrip('0').rstrip('.'),
                    )
                else:
                    # REST-демо
                    await asyncio.to_thread(
                        lambda: self.session.set_trading_stop(
                            category="linear", symbol=symbol, positionIdx=pos_idx,
                            stopLoss=f"{stop_price:.8f}".rstrip('0').rstrip('.'),
                            triggerBy="LastPrice", timeInForce="GTC"
                        )
                    )

            self.last_stop_price[symbol] = stop_price
            self._last_trailing_ts[symbol] = now
            logger.info(f"[TRAILING_STOP] {symbol} стоп обновлён -> {stop_price:.6f}")
            return True

        except InvalidRequestError as e:
            # 34040 = Order not modified (нормально: биржа посчитала, что менять нечего)
            if getattr(e, "status_code", None) == 34040:
                self.last_stop_price[symbol] = stop_price
                self._last_trailing_ts[symbol] = now
                return True
            # 10001 = ордер/позиция не найдены (вероятно уже закрыта)
            if getattr(e, "status_code", None) == 10001:
                logger.warning(f"[TRAILING_STOP] Позиция {symbol} уже закрыта. Чищу состояние.")
                self._purge_symbol_state(symbol)
                return False
            logger.warning(f"[TRAILING_STOP] {symbol} API error: {e}")
            return False
        except Exception as e:
            logger.error(f"[TRAILING_STOP] {symbol} критическая ошибка: {e}", exc_info=True)
            return False

    def _purge_symbol_state(self, symbol: str):
        logger.debug(f"Полная очистка состояния для символа: {symbol}")
        self.open_positions.pop(symbol, None)
        self.last_trailing_stop_set.pop(symbol, None)
        self.last_stop_price.pop(symbol, None)
        self._last_logged_stop_price.pop(symbol, None)
        self._last_trailing_ts.pop(symbol, None)
        self.pending_orders.pop(symbol, None)
        self.pending_cids.pop(symbol, None)
        self.pending_timestamps.pop(symbol, None)
        self.averaged_symbols.discard(symbol)
        self.recently_closed[symbol] = time.time()
        # ── убираем из watch-листа и останавливаем наблюдателя ──
        self.watchlist.pop(symbol, None)
        task = self.watch_tasks.pop(symbol, None)
        if task:
            task.cancel()

    async def run_daily_optimization(self):
        tz = pytz.timezone("Europe/Moscow")
        while True:
            now = datetime.now(tz)
            next_run = now.replace(hour=4, minute=0, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)
            delay = (next_run - now).total_seconds()
            await asyncio.sleep(delay)
            await self.optimize_golden_params()
            try:
                if self.shared_ws and hasattr(self.shared_ws, "optimize_liq_thresholds"):
                    await self.shared_ws.optimize_liq_thresholds()
            except Exception as e:
                logger.warning("[opt_liq] optimisation error: %s", e)

    def load_user_state(self) -> dict:
        try:
            with open("user_state.json", "r", encoding="utf-8") as f:
                all_users = json.load(f)
            return all_users.get(str(self.user_id), {}) # [IMPROVEMENT] Ensure user_id is string for JSON key
        except Exception:
            return {}

    def apply_user_settings(self) -> None:
        cfg = self.load_user_state()
        
        self.ai_primary_model  = cfg.get("ai_primary_model",  "trading-llama")            # основная
        self.ai_advisor_model  = cfg.get("ai_advisor_model",  "0xroyce/plutus:latest")    # помощник/аналитик

        # --- AI concurrency & timeouts ---
        self.ai_timeout_sec = cfg.get("ai_timeout_sec", 8)

        # Модели
        self.ai_primary_model  = getattr(self, 'ai_primary_model',  None) or os.getenv("AI_PRIMARY_MODEL",  "trading-llama")
        self.ai_advisor_model  = getattr(self, 'ai_advisor_model',  None) or os.getenv("AI_ADVISOR_MODEL",  "0xroyce/plutus:latest")

        # Базы для OpenAI-совместимого /v1
        self.ollama_primary_openai = getattr(self, 'ollama_primary_openai', None) or os.getenv("OLLAMA_PRIMARY_OPENAI", "http://localhost:11434/v1")
        self.ollama_advisor_openai = getattr(self, 'ollama_advisor_openai', None) or os.getenv("OLLAMA_ADVISOR_OPENAI", "http://127.0.0.1:11435/v1")

        # Базы для нативного Ollama /api (pull/tags)
        self.ollama_primary_api = getattr(self, 'ollama_primary_api', None) or os.getenv("OLLAMA_PRIMARY_API", "http://localhost:11434")
        self.ollama_advisor_api = getattr(self, 'ollama_advisor_api', None) or os.getenv("OLLAMA_ADVISOR_API", "http://127.0.0.1:11435")

        # Back-compat (если где-то в коде еще используются старые имена)
        self.ai_primary_base_url  = self.ollama_primary_openai
        self.ai_advisor_base_url  = self.ollama_advisor_openai

        # Семафоры на всякий случай
        self.ai_primary_sem = getattr(self, 'ai_primary_sem', None) or asyncio.Semaphore(1)
        self.ai_advisor_sem = getattr(self, 'ai_advisor_sem', None) or asyncio.Semaphore(1)

        self.ai_temperature  = cfg.get("ai_temperature", 0.2)
        self.ai_max_tokens   = cfg.get("ai_max_tokens", 256)   # Ollama честно может игнорить, но не мешает

        self.squeeze_entry_strategy = cfg.get("squeeze_entry_strategy", self.squeeze_entry_strategy)
        # отдельные лимиты параллельности для primary/secondary моделей
        self.ai_primary_concurrency   = cfg.get("ai_primary_concurrency", 2)
        self.ai_secondary_concurrency = cfg.get("ai_secondary_concurrency", 1)

        self.ai_stop_management_enabled = cfg.get("ai_stop_management_enabled", False)

        self.intraday_trailing_enabled = cfg.get(
            "intraday_trailing_enabled", self.intraday_trailing_enabled
        )
        self.entry_cooldown_sec = int(cfg.get("entry_cooldown_sec", self.entry_cooldown_sec))
        if "volume" in cfg:
            self.POSITION_VOLUME = safe_to_float(cfg["volume"])
        if "max_total_volume" in cfg:
            self.MAX_TOTAL_VOLUME = safe_to_float(cfg["max_total_volume"])

        if "trailing_start_pct" in cfg and isinstance(cfg["trailing_start_pct"], dict):
            self.trailing_start_map.update(cfg["trailing_start_pct"])
        if "trailing_gap_pct" in cfg and isinstance(cfg["trailing_gap_pct"], dict):
            self.trailing_gap_map.update(cfg["trailing_gap_pct"])

        self.trailing_start_pct = self.trailing_start_map.get(
            self.strategy_mode, DEFAULT_TRAILING_START_PCT
        )
        self.trailing_gap_pct = self.trailing_gap_map.get(
            self.strategy_mode, DEFAULT_TRAILING_GAP_PCT
        )

        start_pct = self.trailing_start_map.get(self.strategy_mode)
        if start_pct is None:
            # Если для текущего режима нет, ищем "default"
            start_pct = self.trailing_start_map.get("default", DEFAULT_TRAILING_START_PCT)
        self.trailing_start_pct = start_pct

        gap_pct = self.trailing_gap_map.get(self.strategy_mode)
        if gap_pct is None:
            gap_pct = self.trailing_gap_map.get("default", DEFAULT_TRAILING_GAP_PCT)
        self.trailing_gap_pct = gap_pct

        # Логируем текущие активные значения для ясности
        logger.info(
            f"[Settings] Trailing params for mode '{self.strategy_mode}': "
            f"Start at {self.trailing_start_pct}% ROI, Gap {self.trailing_gap_pct}% ROI"
            )

        if "strategy_mode" in cfg:
            alias = {
            "golden":           "golden_only",
            "golden_only":      "golden_only",
            "golden_squeeze":   "golden_squeeze",
            "liq":              "liquidation_only",
            "liquidation_only": "liquidation_only",
            "liq_squeeze":      "liq_squeeze",
            "full":             "full",
            }
            self.strategy_mode = alias.get(str(cfg["strategy_mode"]).lower(), self.strategy_mode)

        if "squeeze_threshold_pct" in cfg:
            self.squeeze_threshold_pct = safe_to_float(cfg["squeeze_threshold_pct"])

        if "stop_loss_mode" in cfg:
            mode = str(cfg["stop_loss_mode"]).lower()
            if mode in ("scalp_loss", "strat_loss"):
                self.stop_loss_mode = mode

    async def reload_settings_loop(self, interval: int = 10):
        last = None
        while True:
            await asyncio.sleep(interval)
            try:
                cfg = self.load_user_state()
            except Exception:
                continue
            if cfg != last:
                last = cfg
                self.apply_user_settings()
                logger.info("[reload_settings] cfg updated: vol=%s  max=%s  mode=%s",
                            self.POSITION_VOLUME, self.MAX_TOTAL_VOLUME, self.strategy_mode)

    async def notify_user(self, text: str) -> None:
        try:
            await telegram_bot.send_message(self.user_id, text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.warning("[notify_user] send error: %s", e)

    async def evaluate_position_with_ml(self, symbol: str):
        try:
            features = self.extract_realtime_features(symbol)
            pred = self.ml_inferencer.infer(features.reshape(1, -1))[0]
            buy_score, sell_score, neutral = pred
            logger.info(f"[ML] {symbol} infer: Buy={buy_score:.3f} Sell={sell_score:.3f} Neutral={neutral:.3f}")

            threshold = 0.65
            if buy_score > threshold:
                await self._open_position(symbol, side="Buy")
            elif sell_score > threshold:
                await self._open_position(symbol, side="Sell")

        except Exception as e:
            logger.warning(f"[ML] {symbol} ML eval failed: {e}")


    def _clamp01(x):
        try:
            x = float(x)
            if x < 0: return 0.0
            if x > 1: return 1.0
            return x
        except Exception:
            return 0.0

    def _safe_parse_ai_json(text: str) -> dict:
        import json, re
        try:
            return json.loads(text)
        except Exception:
            # вырезаем возможный префикс/постфикс, ищем первую { … }
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {}

    # === AI / Ollama ===
    async def _ask_ollama_json(
        self,
        model: str,
        messages: list[dict],
        *,
        timeout_s: float | None = None,
        base_openai_url: str | None = None,
        num_predict: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_json: dict | None = None,
    ) -> dict:
        """
        Унифицированный запрос к Ollama (OpenAI-совместимый /v1) с ожиданием JSON-ответа.
        - Автовыбор base_url по модели (primary/advisor), если base_openai_url не задан.
        - Поддержка безопасных параметров генерации: num_predict -> max_tokens, temperature, top_p.
        - НЕ передаём keep_alive: его нет в OpenAI-совместимом API.
        """
        from openai import AsyncOpenAI

        # Выбор правильной базы /v1
        resolved_base = base_openai_url or (
            self.ollama_primary_openai if model == self.ai_primary_model else self.ollama_advisor_openai
        )

        tmo = float(timeout_s or getattr(self, "ai_timeout_sec", 4.0) or 4.0)

        gen_kwargs: dict = {}
        if num_predict is not None:
            gen_kwargs["max_tokens"] = int(num_predict)
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if extra_json:
            gen_kwargs.update(extra_json)

        client = AsyncOpenAI(base_url=resolved_base, api_key="ollama")

        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **gen_kwargs,
                ),
                timeout=tmo,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                return {"action": "REJECT", "confidence_score": 0.7, "justification": "empty response"}
            return safe_parse_json(raw, default={"action": "REJECT", "confidence_score": 0.7, "justification": "bad json"})
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error(f"[AI_ERROR] {model}: {e}", exc_info=True)
            return {"action": "REJECT", "confidence_score": 0.7, "justification": f"error: {e}"}
        

    def safe_parse_json(text: str | None, default: Any = None) -> Any:
        """
        Пытается распарсить JSON из ответа LLM/Ollama.
        - Аккуратно обрабатывает пустую строку
        - Срезает кодовые блоки ```json ... ```
        - Если модель вернула пояснительный текст вокруг JSON, пытается вытащить первый блок {...}
        - Возвращает default при любой ошибке
        """
        if text is None:
            return default

        s = text.strip()

        # Снимаем ограждение ```json ... ```
        if s.startswith("```"):
            # убираем первую строку с ```/```json
            s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
            # убираем завершающие ```
            s = re.sub(r"\s*```$", "", s).strip()

        # Прямая попытка
        try:
            return json.loads(s)
        except Exception:
            pass

        # Если вокруг JSON есть текст – вытащим первый блок {...}
        try:
            start = s.find("{")
            end   = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = s[start:end+1]
                return json.loads(candidate)
        except Exception:
            pass

        # Ничего не вышло
        return default



    def can_open_position(self, symbol: str, cost: float) -> tuple[bool, str]:
        if symbol in getattr(self, "open_positions", {}):
            return False, "already_open"
        if symbol in getattr(self, "pending_orders", {}):
            return False, "pending_exists"

        try:
            max_total = float(getattr(self, "MAX_TOTAL_VOLUME", 0.0) or 0.0)
            total = 0.0
            for p in getattr(self, "open_positions", {}).values():
                try:
                    total += float(p.get("cost", 0.0))
                except Exception:
                    pass
            for v in getattr(self, "pending_orders", {}).values():
                try:
                    total += float(v)
                except Exception:
                    pass
            if max_total > 0.0 and (total + float(cost)) > max_total:
                return False, f"max_total_exceeded({total}+{cost}>{max_total})"
        except Exception as e:
            logger.warning("[LIMITS] calc_failed: %s", e)

        return True, "ok"

    async def _open_position(self, symbol: str, side: str):
        cid = new_cid()
        st = log_state(self, symbol)
        logger.info("[EXECUTE][%s] start %s/%s | %s", cid, symbol, side, j(st))

        now = time.time()
        # Быстрые отсеки
        if now - self.last_entry_ts.get(symbol, 0) < self.entry_cooldown_sec:
            logger.info(f"[OpenSkip] {symbol} cooldown {self.entry_cooldown_sec}s")
            return
        if now - self.recently_closed.get(symbol, 0) < self.entry_cooldown_sec:
            logger.info(f"[OpenSkip] {symbol} recently closed")
            return

        ok, why = self.can_open_position(symbol, 0.0)
        if not ok:
            logger.info("[EXECUTE][%s] denied(primary): %s | %s", cid, why, j(st))
            return

        # Берём lock и повторно проверяем открытия/pendings атомарно
        async with self.position_lock:
            if symbol in self.open_positions:
                logger.debug(f"[OpenSkip] {symbol} уже открыт")
                return
            if symbol in self.pending_orders:
                logger.debug(f"[OpenSkip] {symbol} уже pending")
                return

        # Вне lock считаем цену/объём
        last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0))
        if not (last_price > 0):
            logger.warning(f"[ML Order] Не удалось получить цену для {symbol}, открытие отменено.")
            return

        qty = await self._calc_qty_from_usd(symbol, self.POSITION_VOLUME, last_price)
        if not (qty > 0):
            logger.warning(f"[ML Order] Рассчитан нулевой объем для {symbol}, открытие отменено.")
            return

        # Риск-чек до установки pending (с поддержкой cid)
        if not await self._risk_check(symbol, side, qty, last_price, cid=cid):
            logger.info("[EXECUTE][%s] denied(risk_check) %s/%s", cid, symbol, side)
            return

        order_cost = float(qty * last_price)

        # Устанавливаем pending атомарно и делаем повторный бюджетный чек
        async with self.position_lock:
            # повторная защита
            if symbol in self.open_positions or symbol in self.pending_orders:
                logger.debug(f"[OpenSkip] {symbol} гонка: уже открыт/pending")
                return

            ok2, why2 = self.can_open_position(symbol, order_cost)
            if not ok2:
                logger.info("[ORDER][%s] denied(recheck): %s", cid, why2)
                return  # ВАЖНО: выходим сразу, ничего не ставим

            # фиксируем pending
            self.pending_orders[symbol] = order_cost
            self.pending_timestamps[symbol] = time.time()
            self.pending_cids[symbol] = cid
            logger.info("[ORDER][%s] pending_set %s cost=%.2f", cid, symbol, order_cost)

        # Пытаемся отправить ордер
        pos_idx = 1 if side == "Buy" else 2
        try:
            resp = await self.place_order_ws(symbol, side, qty, position_idx=pos_idx, cid=cid)

            order_id = None
            try:
                if isinstance(resp, dict):
                    order_id = resp.get("orderId") or resp.get("order_id")
                elif isinstance(resp, list) and resp:
                    order_id = resp[0].get("orderId")
            except Exception:
                order_id = None

            if order_id:
                self.order_correlation[order_id] = cid
                self.last_entry_ts[symbol] = time.time()
                logger.info("[EXECUTE][%s] order_accepted id=%s qty=%.8f", cid, order_id, float(qty))
                logger.info(f"[ML Order] {symbol}: {side} qty={qty}")
                # ВАЖНО: pending НЕ чистим здесь — дождёмся подтверждения по приватному WS (New/Filled/Cancelled)
                return

            # если дошли сюда — ответы без order_id считаем неуспехом
            logger.warning("[EXECUTE][%s] no order_id in response → cleanup pending", cid)
            # cleanup on failure
            self.pending_orders.pop(symbol, None)
            self.pending_cids.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)
            return

        except Exception as e:
            logger.warning(f"[ML Order] Не удалось разместить ордер для {symbol}: {e}", exc_info=True)
            # обязательный cleanup при исключении
            self.pending_orders.pop(symbol, None)
            self.pending_cids.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)
            return

    async def get_total_open_volume(self) -> float:
        try:
            resp = await asyncio.to_thread(
                lambda: self.session.get_positions(
                    category="linear",
                    settleCoin="USDT"
                )
            )
            total = 0.0
            for pos in resp.get("result", {}).get("list", []):
                size = safe_to_float(pos.get("size", 0))
                price = safe_to_float(pos.get("entryPrice", 0)) or safe_to_float(
                    pos.get("markPrice", 0)
                )
                total += size * price
            return total
        except Exception as e:
            logger.warning(
                "[get_total_open_volume] fallback due to %s", e
            )
            total = 0.0
            for pos in self.open_positions.values():
                try:
                    size = safe_to_float(pos.get("size", 0))
                    price = safe_to_float(pos.get("entryPrice", 0)) or safe_to_float(
                        pos.get("markPrice", 0)
                    )
                    total += size * price
                except (ValueError, TypeError):
                    continue
            return total

    def write_open_positions_json(self) -> None:
        try:
            data: dict = {}
            if os.path.exists(OPEN_POS_JSON):
                with open(OPEN_POS_JSON, "r", encoding="utf-8") as fp:
                    try:
                        data = json.load(fp)
                    except Exception:
                        data = {}
            user_positions = {}
            for sym, pos in self.open_positions.items():
                user_positions[sym] = {
                    "side": pos.get("side"),
                    "volume": float(pos.get("volume", 0)),
                    "avg_price": float(pos.get("avg_price", 0)),
                    "pnl": float(pos.get("pnl", 0)),
                }
            data[str(self.user_id)] = user_positions
            _atomic_json_write(OPEN_POS_JSON, data)
        except Exception as e:
            logger.debug("[write_open_positions_json] error: %s", e)

    async def get_server_time(session, demo: bool) -> float:
        url = ("https://api.bybit.com" if not demo else
            "https://api-demo.bybit.com") + "/v5/market/time"
        try:
            t0 = time.time()
            resp = await asyncio.to_thread(lambda: session._request("GET", url))
            st  = resp["result"]["time"] / 1000
            rt  = (t0 + time.time()) / 2
            return st - rt
        except Exception:
            return 0.0

    async def start(self):
        logger.info(f"[User {self.user_id}] Бот запущен")
        self.open_positions.clear()
        self.liq_buffers.clear()
        self.pending_timestamps.clear()
        self.loop = asyncio.get_running_loop()
        await self.update_open_positions()

        await self.setup_private_ws()
        if self.mode == "real":
            await self.init_trade_ws()
        else:
            logger.info("[start] demo mode – trade WebSocket is disabled")
        if self.shared_ws and hasattr(self.shared_ws, "ready_event"):
            await self.shared_ws.ready_event.wait()

        await asyncio.sleep(self.warmup_seconds)
        self.warmup_done = True
        logger.info("[warmup] user %s finished (%d s)", self.user_id, self.warmup_seconds)

        asyncio.create_task(self.run_daily_optimization())

        self.market_task = asyncio.create_task(self.market_loop())
        asyncio.create_task(self.reload_settings_loop())
        self.sync_task   = asyncio.create_task(self.sync_open_positions_loop())
        if self.wallet_task is None:
            self.wallet_task = asyncio.create_task(self.wallet_loop())

        asyncio.create_task(self.cleanup_pending_loop())

    # ── TradingBot.__init__ ─────────────────────────────────────────────
    # уже должен быть:
    # self.current_total_volume = 0.0

    # ── helpers (рядом с остальными утилитами) ─────────────────────────
    def _almost_equal(self, a: float, b: float, eps: float = 1e-6) -> bool:
        return abs(a - b) <= eps

    # ── wallet_loop (замените тело метода) ─────────────────────────────
    async def wallet_loop(self):
        # локальный кэш последнего IM, чтобы не спамить логами
        last_im = self.current_total_volume
        while True:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_wallet_balance(accountType="UNIFIED")
                )
                wallet_raw = resp.get("result", {})
                wallet_list = wallet_raw.get("list", [])

                if wallet_list:
                    im = safe_to_float(wallet_list[0].get("totalInitialMargin", 0))
                    # обновляем только если реально изменилась нагрузка
                    if not self._almost_equal(im, self.current_total_volume):
                        self.current_total_volume = im
                        last_im = im
                        wallet_logger.info("[User %s] IM=%.2f", self.user_id, im)
                    else:
                        # без лишнего шума — только debug, если очень хочется видеть пульс
                        wallet_logger.debug("[User %s] IM unchanged (%.2f)", self.user_id, im)
            except Exception as e:
                wallet_logger.debug("[wallet_loop] error: %s", e)

            # ── адаптивный интервал опроса (без новых конфигов) ─────────
            has_pending = bool(self.pending_orders)
            # если есть занятость (IM > 0) — держим умеренно частый опрос
            if not self._almost_equal(self.current_total_volume, 0.0):
                sleep_s = 7.0 if has_pending else 10.0
            else:
                # пусто: при pending — чаще, без pending — редкий опрос
                sleep_s = 5.0 if has_pending else 30.0

            await asyncio.sleep(sleep_s)


    async def init_trade_ws(self):
        url = "wss://stream.bybit.com/v5/trade"
        while True:
            try:
                self.ws_trade = await websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=15,
                    open_timeout=10
                )
                expires = int((time.time() + 1) * 1000)
                msg = f"GET/realtime{expires}"
                sig = hmac.new(
                    self.api_secret.encode(),
                    msg.encode(),
                    hashlib.sha256
                ).hexdigest()

                auth_req = {
                    "op": "auth",
                    "args": [self.api_key, expires, sig]
                }
                await self.ws_trade.send(json.dumps(auth_req))
                resp = json.loads(await self.ws_trade.recv())
                if resp.get("retCode", None) not in (0, None) and not resp.get("success", False):
                    raise RuntimeError(f"WS auth failed: {resp}")
                logger.info("[init_trade_ws] Trade WS connected and authenticated")
                break
            except Exception as e:
                logger.warning(f"[init_trade_ws] connection/auth error: {e}, retrying in 5s...")
                await asyncio.sleep(5)

        def _market_done(task: asyncio.Task) -> None:
            try:
                exc = task.exception()
                if exc:
                    logger.exception("[market_loop] task for user %s crashed: %s", self.user_id, exc)
                else:
                    logger.warning("[market_loop] task for user %s finished unexpectedly", self.user_id)
            except asyncio.CancelledError:
                logger.info("[market_loop] task for user %s was cancelled", self.user_id)

        if hasattr(self, "market_task") and self.market_task is not None:
            self.market_task.add_done_callback(_market_done)

    async def setup_private_ws(self):
        while True:
            try:
                def _on_private(msg):
                    try:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[PrivateWS] Raw message: {msg}")
                        if not self.loop.is_closed():
                            asyncio.run_coroutine_threadsafe(
                                self.route_private_message(msg),
                                self.loop
                            )
                    except Exception as e:
                        logger.warning(f"[PrivateWS callback] loop closed, skipping message: {e}")

                self.ws_private = WebSocket(
                    testnet=False,
                    demo=self.mode == "demo",
                    channel_type="private",
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    ping_interval=30,
                    ping_timeout=15,
                    restart_on_error=True,
                    retries=200
                )
                self.ws_private.position_stream(callback=_on_private)
                self.ws_private.execution_stream(callback=_on_private)
                logger.info("[setup_private_ws] Подключение к private WebSocket установлено")
                break
            except Exception as e:
                logger.warning(f"[setup_private_ws] Ошибка подключения: {e}, повторная попытка через 5 секунд")
                await asyncio.sleep(5)

    async def route_private_message(self, msg):
        try:
            topic = msg.get("topic", "")
            if topic in ("position", "position.linear"):
                await self.handle_position_update(msg)
            elif topic == "execution":
                await self.handle_execution(msg)
        except Exception as e:
            logger.error(f"[route_private_message] Ошибка: {e}", exc_info=True)
            if self.ws_private:
                self.ws_private.exit()
            await self.setup_private_ws()

    # [IMPROVEMENT] Делаем handle_execution главным обработчиком закрытия
    async def handle_execution(self, msg: dict):
        """
        [V6 - Primary Closer] Вызывается при исполнении ордера.
        Обрабатывает закрытие первым и ставит флаг в recently_closed,
        чтобы предотвратить дублирующую обработку в handle_position_update.
        """
        for exec_data in msg.get("data", []):
            symbol = exec_data.get("symbol")
            if not symbol: continue

            async with self.position_lock:
                pos = self.open_positions.get(symbol)
                if not pos: continue

                exec_side = exec_data.get("side")
                # Проверяем, что это ордер на закрытие (противоположный стороне позиции)
                if exec_side and pos.get("side") and exec_side != pos.get("side"):
                    # leavesQty == 0 означает, что ордер исполнен полностью
                    if safe_to_float(exec_data.get("leavesQty", 0)) == 0:
                        exit_price = safe_to_float(exec_data.get("execPrice"))
                        pos_volume = safe_to_float(pos.get("volume", 0))
                        
                        # Расчет PnL на основе точной цены исполнения
                        pnl_usdt = self._calc_pnl(pos["side"], safe_to_float(pos["avg_price"]), exit_price, pos_volume)
                        position_value = safe_to_float(pos["avg_price"]) * pos_volume
                        pnl_pct = (pnl_usdt / position_value) * 1000 if position_value else 0.0

                        logger.info(f"[EXECUTION_CLOSE] {symbol}. PnL: {pnl_usdt:.2f} USDT ({pnl_pct:.2f}%).")

                        # Сохраняем данные для логгирования
                        self.closed_positions[symbol] = dict(pos)
                        
                        # [КЛЮЧЕВОЕ ИЗМЕНЕНИЕ] Очищаем состояние и СРАЗУ ЖЕ ставим флаг
                        self._purge_symbol_state(symbol) 
                        self.write_open_positions_json()

                        # Логируем корректное закрытие
                        asyncio.create_task(self.log_trade(
                            symbol=symbol, side=pos["side"], avg_price=exit_price, volume=pos_volume,
                            action="close", result="closed_by_execution", pnl_usdt=pnl_usdt, pnl_pct=pnl_pct,
                            comment=self.last_entry_comment.pop(symbol, None)
                        ))
                        
                        # Прерываем дальнейшую обработку для этого символа в этом сообщении
                        return

    @staticmethod
    def _calc_pnl(entry_side: str, entry_price: Any,
                  exit_price: Any, qty: Any) -> float:
        """
        [V2 - Type Safe] Безопасно рассчитывает PnL в USDT.
        Принимает на вход значения любого типа (str, float, Decimal) и
        выполняет вычисления с высокой точностью, используя Decimal.
        Возвращает результат как стандартный float.
        """
        try:
            # 1. Преобразуем все входные данные в Decimal для точных вычислений
            d_entry_price = Decimal(str(entry_price))
            d_exit_price = Decimal(str(exit_price))
            d_qty = Decimal(str(qty))

            # 2. Выполняем математику с высокой точностью
            if entry_side == "Buy":
                pnl = (d_exit_price - d_entry_price) * d_qty
            else:  # Sell
                pnl = (d_entry_price - d_exit_price) * d_qty
            
            # 3. Возвращаем результат как стандартный float для совместимости
            return float(pnl)
        
        except (InvalidOperation, TypeError, ValueError) as e:
            # Если преобразование или расчет не удались, логируем ошибку и возвращаем 0
            logger.error(f"[PNL_CALC_ERROR] Не удалось рассчитать PnL. "
                         f"Данные: entry={entry_price}, exit={exit_price}, qty={qty}. Ошибка: {e}")
            return 0.0

    def _save_trade(self, trade: dict) -> None:
        try:
            data = json.loads(self.trade_history_file.read_text(encoding="utf-8"))
            data.append(trade)
            self.trade_history_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info("[trade] %s %s qty=%.4f pnl=%.2f USDT",
                        trade["symbol"], trade["side"],
                        trade["qty"], trade["pnl"])
        except Exception:
            logger.exception("[trade] cannot write trade_history.json")

    async def adopt_existing_position(self, symbol: str, pos_data: dict):
        # ⛔ Консервация стартовой адаптации стопов.
        # По умолчанию выключено. Включить можно, установив self.enable_start_stop_adapt = True.
        if not getattr(self, "enable_start_stop_adapt", False):
            logger.info(f"[ADAPT] стартовая адаптация стопов отключена; пропускаю {symbol}")
            # помечаем как обработанную, чтобы не пытаться снова на каждом цикле
            async with self.position_lock:
                if symbol in self.open_positions:
                    self.open_positions[symbol]['adopted'] = True
            return

        logger.info(f"[ADAPT] Обнаружена существующая позиция по {symbol}. Адаптация...")

        sl_price = safe_to_float(pos_data.get("stopLoss", "0"))
        if sl_price > 0:
            self.last_stop_price[symbol] = sl_price
            logger.info(f"[ADAPT] {symbol}: Подхвачен существующий стоп-лосс на бирже: {sl_price}")
        else:
            logger.warning(f"[ADAPT] {symbol}: На бирже не установлен стоп-лосс. Устанавливаем начальный.")

            avg_price = safe_to_float(pos_data.get("avgPrice"))
            side = pos_data.get("side")
            last_price = safe_to_float(pos_data.get("markPrice", 0)) or avg_price
            if not last_price:
                logger.error(f"[ADAPT] {symbol}: Не удалось получить цену для установки начального стопа!")
                return

            leverage = safe_to_float(pos_data.get("leverage", 10.0))
            if leverage == 0:
                leverage == 10.0
            current_roi = (
                (((last_price - avg_price) / avg_price) * 100 * leverage)
                if side == "Buy" else
                (((avg_price - last_price) / avg_price) * 100 * leverage)
            )

            if await self.set_trailing_stop(symbol, avg_price, current_roi, side):
                await self.log_trade(symbol=symbol, side=side, avg_price=avg_price,
                                    volume=pos_data['size'], action="adopt_stop_set", result="success")
            else:
                logger.error(f"[ADAPT] {symbol}: Не удалось установить безопасный начальный стоп!")

        async with self.position_lock:
            if symbol in self.open_positions:
                self.open_positions[symbol]['adopted'] = True

                
    # [FINAL VERSION] Обработчик сообщений о позициях
    async def handle_position_update(self, msg: dict):
        """
        [V8 - Reliable Fallback] Обрабатывает открытие, обновление и служит
        надежным fallback-механизмом для закрытия, уважая флаг от handle_execution.
        """
        data = msg.get("data", [])
        if isinstance(data, dict): data = [data]

        async with self.position_lock:
            for p in data:
                symbol = p.get("symbol")
                if not symbol: continue

                new_size = safe_to_float(p.get("size", 0))
                prev_pos = self.open_positions.get(symbol)

                # --- Сценарий 1: Открытие новой позиции ---
                if prev_pos is None and new_size > 0:
                    side_raw = p.get("side")
                    if not side_raw: continue
                    
                    # Убираем метки "в ожидании"
                    self.pending_orders.pop(symbol, None)
                    self.pending_cids.pop(symbol, None)
                    self.pending_timestamps.pop(symbol, None)
                    self._pending_close_ts.pop(symbol, None)

                    avg_price = safe_to_float(p.get("avgPrice") or p.get("entryPrice"))
                    
                    # Получаем комментарий от AI/стратегии, если он был
                    entry_candidate = self.active_trade_entries.pop(symbol, {})
                    comment = entry_candidate.get("comment")

                    # Запомним, чтобы показать и при закрытии
                    if comment:
                        self.last_entry_comment[symbol] = comment

                    # Создаем запись о новой открытой позиции
                    self.open_positions[symbol] = {
                        "avg_price": avg_price, 
                        "side": side_raw,
                        "pos_idx": 1 if side_raw == 'Buy' else 2,
                        "volume": new_size, 
                        "leverage": safe_to_float(p.get("leverage", "1")),
                        "entry_candidate": entry_candidate,
                        "markPrice": avg_price, 
                        "pnl": 0.0,
                        "entry_features": self._build_entry_features(symbol)
                    }
                    logger.info(f"[PositionStream] NEW {side_raw} {symbol} {new_size:.3f} @ {avg_price:.6f}")
                    
                    # Логируем открытие
                    asyncio.create_task(self.log_trade(
                        symbol=symbol, side=side_raw, avg_price=avg_price,
                        volume=new_size, action="open", result="opened",
                        comment=comment
                    ))
                    
                    # Запускаем автономного "Хранителя" для этой позиции
                    asyncio.create_task(self.manage_open_position(symbol))
                    
                    self.write_open_positions_json()
                    continue

                # --- Сценарий 2: Обновление существующей позиции (усреднение) ---
                if prev_pos and new_size > 0 and abs(new_size - safe_to_float(prev_pos.get("volume", 0))) > 1e-9:
                    logger.info(f"[PositionStream] {symbol} volume updated: {prev_pos.get('volume')} -> {new_size}")
                    self.open_positions[symbol]["volume"] = new_size
                    self.open_positions[symbol]["avg_price"] = safe_to_float(p.get("avgPrice") or p.get("entryPrice"))
                    self.write_open_positions_json()
                    continue

                # --- Сценарий 3: Fallback-закрытие (если handle_execution не сработал) ---
                now = time.time()

                # Игнорируем fallback-закрытия во время прогрева
                if not getattr(self, "warmup_done", False):
                    logger.debug(f"[PositionStream] {symbol} size=0 во время прогрева — игнорируем")
                    continue

                ts = self._pending_close_ts.get(symbol, 0.0)
                if ts == 0.0:
                    self._pending_close_ts[symbol] = now
                    logger.debug(f"[PositionStream] {symbol} size=0 (кандидат на закрытие). Ждём подтверждения...")
                    continue

                if now - ts < 2.0:
                    continue

                ok = await self.confirm_position_closing(symbol)
                if not ok:
                    self._pending_close_ts.pop(symbol, None)
                    logger.debug(f"[PositionStream] {symbol} закрытие не подтвердилось по REST — игнорируем")
                    continue

                self._pending_close_ts.pop(symbol, None)
                # дальше — ваш текущий код закрытия
                if prev_pos and new_size == 0:
                    # [КЛЮЧЕВАЯ ПРОВЕРКА] Если handle_execution уже обработал закрытие, он
                    # добавил символ в recently_closed. В этом случае мы пропускаем дублирующую обработку.
                    if symbol in self.recently_closed:
                        logger.debug(f"[PositionStream] Закрытие {symbol} уже обработано handle_execution. Пропуск.")
                        continue

                    logger.info(f"[PositionStream] Fallback: {symbol} closed (size=0). Запускаем обработку.")
                    
                    # Сохраняем снимок позиции для точного логгирования
                    snapshot = dict(prev_pos)
                    self.closed_positions[symbol] = snapshot
                    
                    # Полностью очищаем все состояния, связанные с этой позицией
                    self._purge_symbol_state(symbol)
                    self.write_open_positions_json()
                    
                    # Расчет PnL. В этом fallback-сценарии цена выхода может быть неточной,
                    # но это лучше, чем ничего или расчет с ошибкой.
                    exit_price = safe_to_float(p.get("avgPrice") or snapshot.get("markPrice", snapshot.get("avg_price", 0)))
                    pos_volume = safe_to_float(snapshot.get("volume", 0))
                    entry_price = safe_to_float(snapshot.get("avg_price", 0))
                    
                    pnl_usdt = self._calc_pnl(snapshot.get("side", "Buy"), entry_price, exit_price, pos_volume)
                    pos_value = entry_price * pos_volume
                    pnl_pct = (pnl_usdt / pos_value) * 1000 if pos_value else 0.0

                    # Логируем закрытие
                    asyncio.create_task(self.log_trade(
                        symbol=symbol, side=snapshot.get("side", "Buy"), avg_price=exit_price,
                        volume=pos_volume, action="close", result="closed_by_position_stream",
                        pnl_usdt=pnl_usdt, pnl_pct=pnl_pct,
                        comment=self.last_entry_comment.pop(symbol, None)
                    ))

    async def confirm_position_closing(self, symbol: str) -> bool:
        try:
            resp = await asyncio.to_thread(
                lambda: self.session.get_positions(category="linear", settleCoin="USDT")
            )
            for pos in resp.get("result", {}).get("list", []):
                if pos.get("symbol") == symbol:
                    size = safe_to_float(pos.get("size", 0))
                    return size == 0
            return True
        except Exception as e:
            logger.warning(f"[confirm_position_closing] Ошибка проверки позиции {symbol}: {e}")
            return True

    async def update_open_positions(self):
        try:
            response = await asyncio.to_thread(
                lambda: self.session.get_positions(category="linear", settleCoin="USDT")
            )
            if response.get("retCode") != 0:
                logger.warning(f"[update_open_positions] Ошибка API: {response.get('retMsg')}")
                return

            live_positions = {pos["symbol"]: pos for pos in response.get("result", {}).get("list", []) if safe_to_float(pos.get("size", 0)) > 0}

            for symbol, pos_data in live_positions.items():
                if symbol not in self.open_positions:
                    if symbol in self.recently_closed:
                        logger.debug(f"[SYNC] Игнорируем 'воскрешение' {symbol}, т.к. он был недавно закрыт.")
                        continue

                    logger.info(f"[SYNC] Обнаружена новая активная позиция на бирже: {symbol}")

                    side = pos_data.get("side", "")
                    correct_pos_idx = 1 if side == 'Buy' else 2
                    self.open_positions[symbol] = {
                        "avg_price": safe_to_float(pos_data.get("entryPrice") or pos_data.get("avgPrice")),
                        "side":      side,
                        "pos_idx":   pos_data.get("positionIdx", correct_pos_idx),
                        "volume":    safe_to_float(pos_data.get("size", 0)),
                        "leverage":  safe_to_float(pos_data.get("leverage", "1")),
                        "markPrice": safe_to_float(pos_data.get("markPrice", 0)),
                        "open_timestamp": time.time()
                    }
                    await self.adopt_existing_position(symbol, pos_data)

            for symbol in list(self.open_positions.keys()):
                if symbol not in live_positions:
                    logger.info(f"[SYNC] Позиция {symbol} больше не активна. Полная очистка состояния.")
                    self._purge_symbol_state(symbol)

            self.write_open_positions_json()
        except Exception as e:
            logger.error(f"[update_open_positions] Критическая ошибка синхронизации: {e}", exc_info=True)

    async def stop(self) -> None:
        try:
            with open(self.training_data_path, "wb") as f:
                pickle.dump(self.training_data, f)
            logger.info(f"[ML] Сохранено {len(self.training_data)} обучающих примеров в {self.training_data_path}.")
        except Exception as e:
            logger.error(f"[ML] Ошибка сохранения обучающих данных: {e}")

        for name in ("market_task", "sync_task", "wallet_task", "_cleanup_task"):
            task = getattr(self, name, None)
            if task and not task.done():
                task.cancel()

        if getattr(self, "ws_private", None):
            self.ws_private.exit()
        if getattr(self, "ws_trade", None):
            try:
                await self.ws_trade.close()
            except Exception:
                pass
        logger.info("[User %s] Bot stopped", self.user_id)

    def _step_decimals(self, step: float) -> int:
        """
        Сколько знаков после запятой у шага, устойчиво к '1e-07'.
        Ограничиваемся максимум 7 знаками (требование).
        """
        # печатаем фиксированно, с запасом, отрезаем хвосты
        s = f"{float(step):.12f}".rstrip("0").rstrip(".")
        if "." in s:
            dec = len(s.split(".")[1])
        else:
            dec = 0
        return min(dec, 7)

    def _format_qty(self, symbol: str, qty: float) -> str:
        """
        Округляет ВНИЗ к шагу лота и форматирует строку для API
        (устойчиво к научной записи шага; до 7 знаков после запятой).
        """
        step = self.qty_step_map.get(symbol, 0.001) or 0.001
        min_qty = self.min_qty_map.get(symbol, step)
        # безопасный floor без дребезга float
        ticks = math.floor((qty / step) + 1e-12)
        q = ticks * step
        if q < min_qty and min_qty > 0:
            # поднимем до минимума вровень по шагу
            ticks_min = math.ceil((min_qty / step) - 1e-12)
            q = ticks_min * step
        decimals = self._step_decimals(step)
        s = f"{q:.{decimals}f}".rstrip("0").rstrip(".")
        return s if s else "0"

    async def reprice_pending_order(self, symbol: str, last_price: float) -> None:
        ctx = self.reserve_orders.get(symbol)
        if not ctx:
            return

        if time.time() - ctx.get("last_reprice_ts", 0) < SQUEEZE_REPRICE_INTERVAL:
            return

        offset = SQUEEZE_LIMIT_OFFSET_PCT
        tick   = float(DEC_TICK)
        old    = ctx["price"]

        if ctx["action"] == "SHORT":
            new = math.floor(last_price * (1 + offset) / tick) * tick
            if new >= last_price:
                new = last_price - tick
        else:
            new = math.ceil(last_price * (1 - offset) / tick) * tick
            if new <= last_price:
                new = last_price + tick

        if abs(new - old) <= tick:
            return

        resp = await asyncio.to_thread(
            lambda: self.session.amend_order(
                category="linear",
                symbol=symbol,
                orderId=ctx["orderId"],
                price=str(new),
            )
        )

        if resp.get("retCode") == 0:
            ctx.update({"price": new, "last_reprice_ts": time.time()})
            logger.debug("[reprice] %s amended → %.6f", symbol, new)
        else:
            logger.warning("[reprice] amend failed: %s | %s",
                        resp.get("retCode"), resp.get("retMsg"))

    def _aggregate_candles_5m(self, candles: list) -> list:
        candles = list(candles)
        if not candles:
            return []
        result = []
        full_blocks = len(candles) // 5
        for i in range(full_blocks):
            chunk = candles[i * 5:(i + 1) * 5]
            open_p = safe_to_float(chunk[0]["openPrice"])
            high_p = max(safe_to_float(c["highPrice"]) for c in chunk)
            low_p = min(safe_to_float(c["lowPrice"]) for c in chunk)
            close_p = safe_to_float(chunk[-1]["closePrice"])
            volume = sum(safe_to_float(c["volume"]) for c in chunk)
            result.append(
                {
                    "openPrice": open_p,
                    "highPrice": high_p,
                    "lowPrice": low_p,
                    "closePrice": close_p,
                    "volume": volume,
                }
            )
        return result


    def _get_1m_bars(self, symbol: str, count: int = 120) -> list:
        """
        Возвращает последние count 1m-баров по символу, пробуя несколько известных контейнеров.
        Формат бара должен быть dict с ключами open/high/low/close/volume (или короткие o/h/l/c/vol).
        """
        # 1) Пробуем словари на самом боте
        for attr in ("bars_1m", "history_1m", "history"):
            d = getattr(self, attr, None)
            if isinstance(d, dict) and symbol in d:
                seq = d[symbol]
                if isinstance(seq, list) and seq:
                    return seq[-count:]

        # 2) Пробуем внутри WS-объекта, если он есть
        ws = getattr(self, "shared_ws", None)
        if ws:
            for attr in ("bars_1m", "ohlcv_1m", "history_1m"):
                d = getattr(ws, attr, None)
                if isinstance(d, dict) and symbol in d:
                    seq = d[symbol]
                    if isinstance(seq, list) and seq:
                        return seq[-count:]

        return []


    # def _aggregate_series_5m(self, series: list, method: str = "sum") -> list:
    #     if not series:
    #         return []
    #     result = []
    #     full_blocks = len(series) // 5
    #     for i in range(full_blocks):
    #         chunk = series[i * 5:(i + 1) * 5]
    #         if method == "sum":
    #             # [BUG FIX] Преобразуем каждый элемент в float перед суммированием
    #             float_chunk = [safe_to_float(x) for x in chunk]
    #             result.append(sum(float_chunk))
    #         else:
    #             # Для 'last' также безопасно приводим к float
    #             result.append(safe_to_float(chunk[-1]))
    #     return result

    # async def handle_liquidation(self, msg):
    #     if not getattr(self, "warmup_done", False):
    #         return
    #     symbol = msg.get("data", [{}])[0].get("s")
    #     if symbol in self.open_positions:
    #         logger.info(f"Skipping liquidation trade for {symbol}: position already open")
    #         return

    #     data = msg.get("data", [])
    #     if isinstance(data, dict):
    #         data = [data]

    #     for evt in data:
    #         symbol = evt.get("s")
    #         qty    = safe_to_float(evt.get("v", 0))
    #         side_evt = evt.get("S")
    #         price  = safe_to_float(evt.get("p", 0))
    #         if not symbol or qty <= 0 or price <= 0:
    #             continue
    #         value_usdt = qty * price

    #         self.shared_ws.latest_liquidation[symbol] = {
    #             "value": value_usdt,
    #             "side":  side_evt,
    #             "ts":    time.time(),
    #         }
    #         logger.info("[liquidation] %s %s %.2f USDT  (qty=%s × price=%s)",
    #                     symbol, side_evt, value_usdt, qty, price)

    #         csv_path = LIQUIDATIONS_CSV_PATH if "LIQUIDATIONS_CSV_PATH" in globals() \
    #                                             else "liquidations.csv"
    #         new_file = not os.path.isfile(csv_path)
    #         with open(csv_path, "a", newline="", encoding="utf-8") as fp:
    #             w = csv.writer(fp)
    #             if new_file:
    #                 w.writerow(["timestamp", "symbol", "side",
    #                             "price", "quantity", "value_usdt"])
    #             w.writerow([datetime.utcnow().isoformat(),
    #                         symbol, side_evt, price, qty, value_usdt])

    #         if self.strategy_mode not in ("liquidation_only", "full", "liq_squeeze"):
    #             continue

    #         now_ts = time.time()
    #         buf = self.liq_buffers[symbol]
    #         buf.append((now_ts, side_evt, value_usdt, price))

    #         cutoff_ts = now_ts - LIQ_CLUSTER_WINDOW_SEC
    #         while buf and buf[0][0] < cutoff_ts:
    #             buf.popleft()

    #         cluster_val = sum(v for _, _, v, _ in buf)
    #         if cluster_val < LIQ_CLUSTER_MIN_USDT:
    #             continue

    #         cluster_price = sum(p * v for _, _, v, p in buf) / cluster_val
    #         if abs(price - cluster_price) / cluster_price * 100 > LIQ_PRICE_PROXIMITY_PCT:
    #             continue

    #         long_val  = sum(v for _, s, v, _ in buf if s == "Buy")
    #         short_val = sum(v for _, s, v, _ in buf if s == "Sell")
    #         if long_val == 0 and short_val == 0:
    #             continue

    #         order_side = "Buy" if short_val > long_val else "Sell"

    #         if (symbol in self.pending_orders or
    #                 not self.shared_ws.check_liq_cooldown(symbol)):
    #             continue

    #         async with self.position_lock:
    #             await self.ensure_symbol_meta(symbol)
    #             last_price = safe_to_float(
    #                 self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
    #             )
    #             if last_price <= 0:
    #                 continue

    #             usd_qty = min(self.POSITION_VOLUME, self.max_allowed_volume)
    #             step    = self.qty_step_map.get(symbol, 0.001)
    #             min_qty = self.min_qty_map.get(symbol, 0.001)
    #             original_qty = max(usd_qty / last_price, min_qty)
    #             qty_ord = min(math.floor(original_qty / step) * step, self.POSITION_VOLUME)
    #             qty_ord = math.floor(qty_ord / step) * step
    #             if qty_ord <= 0:
    #                 continue

    #             if not await self._risk_check(symbol, order_side, qty_ord, last_price):
    #                 logger.info("[liq_trade] %s skipped — exposure limit", symbol)
    #                 continue

    #         self.pending_orders[symbol] = qty_ord * last_price # [IMPROVEMENT] Use dict for pending orders
    #         self.pending_timestamps[symbol]        = time.time()
    #         self.pending_strategy_comments[symbol] = "liq-cluster"

    #         try:
    #             pos_idx = (1 if order_side == "Buy" else 2)

    #             resp = await asyncio.to_thread(lambda: self.session.place_order(
    #                 category="linear",
    #                 symbol=symbol,
    #                 side=order_side,
    #                 orderType="Market",
    #                 qty=self._format_qty(symbol, qty_ord),
    #                 positionIdx=pos_idx,
    #                 timeInForce="GTC",
    #             ))

    #             if resp.get("retCode", 0) != 0:
    #                 raise InvalidRequestError(
    #                 resp.get("retMsg", "order rejected"),
    #                 resp.get("retCode", 0),
    #                 resp.get("time", int(time.time() * 1000)),
    #                 {}
    #             )

    #             self.shared_ws.last_liq_trade_time[symbol] = dt.datetime.utcnow()
    #             logger.info(
    #                 "[liq_trade] %s %s qty=%s opened by cluster (Σ=%.0f USDT)",
    #                 order_side, symbol, qty_ord, cluster_val
    #             )
    #         except Exception as exc:
    #             logger.warning("[liq_trade] %s error: %s", symbol, exc)
    #             self.pending_orders.pop(symbol, None)
    #            self.pending_cids.pop(symbol, None) # [IMPROVEMENT] Use pop for dict
    #         finally:
    #             self.pending_signals.pop(symbol, None)


    def _aggregate_series_5m(self, source, lookback: int = 6, method: str = "sum") -> list:
        """
        Универсальный агрегатор "5 минут".

        Если source — str (символ):
            - берём сырые 1m бары (dict’ы), объединяем по 5 штук в 5m OHLCV,
            - возвращаем список dict’ов: {"open","high","low","close","volume"} (последние lookback).

        Если source — list чисел:
            - старая логика агрегирования рядов (sum/last) по блокам из 5 значений,
            - возвращаем последние lookback значений.
        """
        # Ветка 1: по символу -> OHLCV 5m
        if isinstance(source, str):
            symbol = source
            # +10 берём "с запасом", чтобы точно хватило целых блоков
            m1 = self._get_1m_bars(symbol, count=max(lookback * 5 + 10, 30))
            if not m1 or len(m1) < 5:
                return []

            # Берём только полные блоки по 5 минут
            full_blocks = len(m1) // 5
            if full_blocks == 0:
                return []
            start = len(m1) - full_blocks * 5

            out = []
            for i in range(full_blocks):
                chunk = m1[start + i*5 : start + (i+1)*5]

                # Достаём поля с безопасными синонимами
                def _g(b, *keys, default=0.0):
                    for k in keys:
                        if k in b and b[k] is not None:
                            return b[k]
                    return default

                o = safe_to_float(_g(chunk[0],  "open","o","O"))
                c = safe_to_float(_g(chunk[-1], "close","c","C"))
                h = max(safe_to_float(_g(b, "high","h","H")) for b in chunk)
                l = min(safe_to_float(_g(b, "low","l","L"))  for b in chunk)
                v = sum(safe_to_float(_g(b, "volume","vol","q","V")) for b in chunk)

                out.append({"open": o, "high": h, "low": l, "close": c, "volume": v})

            if lookback and lookback > 0:
                out = out[-lookback:]
            return out

        # Ветка 2: как раньше — агрегируем список чисел
        series = source
        if not series:
            return []
        full_blocks = len(series) // 5
        result = []
        for i in range(full_blocks):
            chunk = series[i * 5:(i + 1) * 5]
            if method == "sum":
                float_chunk = [safe_to_float(x) for x in chunk]
                result.append(sum(float_chunk))
            else:
                result.append(safe_to_float(chunk[-1]))

        # Приводим к lookback
        if lookback and lookback > 0:
            result = result[-lookback:]
        return result


    def _make_signal_key(self,
                     symbol: Any,
                     side: Optional[str] = None,
                     source: Optional[str] = None) -> str:
        if isinstance(symbol, dict):
            symbol_str = symbol.get("symbol") or str(symbol)
        else:
            symbol_str = str(symbol)

        parts = [symbol_str]
        if side:
            parts.append(side)
        if source:
            parts.append(source)

        return "_".join(str(p) for p in parts)

    async def _rescue_router(self, symbol: str, pnl_pct: float, side: str) -> None:
        rescue = self.active_trade_entries.setdefault("rescue_mode", {})
        if pnl_pct > -1.0:
            rescue.pop(symbol, None)
            return

        DCA_TRG   = -180.0
        HEDGE_TRG = -60.0
        EXIT_TRG  = -230.0

        if pnl_pct <= EXIT_TRG:
            await self._hard_stop(symbol)
            rescue.pop(symbol, None)
            return

        if pnl_pct <= HEDGE_TRG and rescue.get(symbol) != "hedge":
            await self._open_hedge(symbol, side)
            rescue[symbol] = "hedge"
            return

        if pnl_pct <= DCA_TRG and rescue.get(symbol) != "dca":
            await self._average_down(symbol, side)
            rescue[symbol] = "dca"
            return

    async def _average_down(self, symbol: str, side: str) -> None:
        pos = self.open_positions.get(symbol, {})
        base_qty = safe_to_float(pos.get("volume", 0))
        if base_qty == 0: return

        ticker = self.shared_ws.ticker_data.get(symbol, {})
        bid = safe_to_float(ticker.get("bid1Price"))
        ask = safe_to_float(ticker.get("ask1Price"))
        if not (bid > 0 and ask > 0): return

        usd_amount = safe_to_float(pos.get("avg_price", 0)) * base_qty
        add_qty = await self._calc_qty_from_usd(symbol, usd_amount, (bid + ask) / 2)
        if not add_qty > 0: return

        tick = self.price_tick_map.get(symbol, DEC_TICK)
        if side == "Buy":
            price = math.floor(bid * 0.996 / tick) * tick
            idx = 1
        else:
            price = math.ceil(ask * 1.004 / tick) * tick
            idx = 2

        try:
            await self._submit_with_risk(symbol, side, add_qty, last_price=(bid+ask)/2, order_type="Limit", price=price, comment="RESCUE-DCA")
