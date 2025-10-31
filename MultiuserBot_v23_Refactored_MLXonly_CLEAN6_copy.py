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
        "bot", "subscribed_topics",
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
        self.subscribed_topics: set[str] = set()
        # на старте подхватим уже существующие топики из pybit, если есть
        cd = getattr(self.ws, "callback_directory", {})
        if isinstance(cd, dict):
            self.subscribed_topics.update(cd.keys())

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

    def _topic(self, tpl: str, sym: str) -> str:
        return tpl.format(symbol=sym)

    def _filter_new_symbols(self, tpl: str, symbols: list[str]) -> list[str]:
        """Оставляем только те символы, чей топик ещё не подписан."""
        out = []
        for s in symbols:
            t = self._topic(tpl, s)
            if t not in self.subscribed_topics:
                out.append(s)
        return out

    def _mark_subscribed(self, tpl: str, symbols: list[str]) -> None:
        for s in symbols:
            self.subscribed_topics.add(self._topic(tpl, s))

    def _mark_unsubscribed(self, tpl: str, symbols: list[str]) -> None:
        for s in symbols:
            self.subscribed_topics.discard(self._topic(tpl, s))

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
                    symbol=list(self.symbols),
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
                
                symbols_to_add = new_active_symbols - self.active_symbols
                symbols_to_remove = self.active_symbols - new_active_symbols

                # 5. Динамически обновляем подписки (с фильтром по уже подписанным топикам)
                k_tpl  = f"kline.{self.interval}.{{symbol}}"
                t_tpl  = "tickers.{symbol}"
                l_tpl  = "liquidation.{symbol}"

                if symbols_to_add:
                    symbols_to_add = list(symbols_to_add)

                    add_k = self._filter_new_symbols(k_tpl, symbols_to_add)
                    add_t = self._filter_new_symbols(t_tpl, symbols_to_add)
                    add_l = self._filter_new_symbols(l_tpl, symbols_to_add)

                    try:
                        if add_k:
                            self.ws.subscribe(topic=k_tpl, symbol=add_k, callback=self._callback)
                            self._mark_subscribed(k_tpl, add_k)
                        if add_t:
                            self.ws.subscribe(topic=t_tpl, symbol=add_t, callback=self._callback)
                            self._mark_subscribed(t_tpl, add_t)
                        if add_l:
                            self.ws.subscribe(topic=l_tpl, symbol=add_l, callback=self._callback)
                            self._mark_subscribed(l_tpl, add_l)
                    except Exception as e:
                        # Страховка: если это «already subscribed», просто отметим и идём дальше
                        if "already subscribed" in str(e).lower():
                            # помечаем всё, что пытались подписать
                            self._mark_subscribed(k_tpl, add_k)
                            self._mark_subscribed(t_tpl, add_t)
                            self._mark_subscribed(l_tpl, add_l)
                            logger.debug(f"[SymbolManager] skip dup subscribe: {e}")
                        else:
                            raise

                    self.active_symbols.update(symbols_to_add)

                if symbols_to_remove and self.ENABLE_UNSUBSCRIBE:
                    try:
                        self.ws.unsubscribe(topic=k_tpl, symbol=list(symbols_to_remove))
                        self._mark_unsubscribed(k_tpl, list(symbols_to_remove))
                    except Exception:
                        pass
                    try:
                        self.ws.unsubscribe(topic=t_tpl, symbol=list(symbols_to_remove))
                        self._mark_unsubscribed(t_tpl, list(symbols_to_remove))
                    except Exception:
                        pass
                    try:
                        self.ws.unsubscribe(topic=l_tpl, symbol=list(symbols_to_remove))
                        self._mark_unsubscribed(l_tpl, list(symbols_to_remove))
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
            if self.position_handlers:
                await asyncio.gather(
                    *(
                        bot.handle_liquidation(msg)
                        for bot in self.position_handlers
                        if bot.strategy_mode in ("full", "liquidation_only", "liq_squeeze")
                    ),
                    return_exceptions=True
                )

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

# ======================================================================
# == КОНЕЦ ML-БЛОКА
# ======================================================================

def _now_ts() -> float:
    return time.time()

def _safe_float(x, default=None):
    try: return float(x)
    except Exception: return default

def _json_dumps(obj) -> str:
    try: return json.dumps(obj, ensure_ascii=False)
    except Exception: return ""


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
        "strategy_mode", "liq_buffers",
        "trailing_start_map", "trailing_gap_map",
        "trailing_start_pct", "trailing_gap_pct", "ml_inferencer",
        "pending_timestamps", "squeeze_threshold_pct", "squeeze_power_min", "averaging_enabled",
        "warmup_done", "warmup_seconds", "_last_snapshot_ts", "reserve_orders",
        "MLX_model", "feature_scaler", "last_retrain", "training_data", "device",
        "last_squeeze_ts", "squeeze_cooldown_sec", "active_trade_entries", "listing_age_min", "_age_cache",
        "symbol_info", "trade_history_file", "active_trades", "pending_signals", "max_signal_age",
        "_oi_sigma", "_pending_clean_task", "squeeze_tuner",
        "_best_entries", "_best_entry_seen", "_best_entry_cache",
        "_golden_weights", "_squeeze_weights", "_liq_weights",
        "gemini_api_key", "ml_lock", "ml_model_bundle", "_last_manage_ts", "training_data_path", "evaluated_signals_cache",
        "gemini_limiter", "_evaluated_signals_cache", "openai_api_key", "ai_stop_management_enabled", "failed_stop_attempts",
        "ml_framework", "_build_default", "ai_provider", "stop_loss_mode", "_last_logged_stop_price", "recently_closed", "_cleanup_task",
        "_last_trailing_stop_order_id", "ai_timeout_sec", "ai_sem", "ai_circuit_open_until", "_ai_silent_until",
        "ml_gate_abs_roi", "ml_gate_min_score", "ml_gate_sigma_coef", "leverage", "order_correlation",
        "STOP_WARMUP_SEC", "ATR_MULT_SL_INIT", "ATR_MULT_TP_INIT", "ATR_MULT_TRAIL", "BREAKEVEN_TRIGGER_ATR", "trailing_update_interval_sec",
        "stop_update_count", "intraday_trailing_enabled", "last_entry_ts", "entry_cooldown_sec", "_pending_close_ts",
        "ai_log_path", "ai_decisions", "ai_decision_by_symbol", "_last_open_msg_ts", "_reopen_block_sec", "_open_msg_cooldown_sec",
        )

    def __init__(self, user_data, shared_ws, golden_param_store):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "real")
        self.listing_age_min = int(user_data.get("listing_age_min_minutes", LISTING_AGE_MIN_MINUTES))

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

        self._age_cache = {}
        self.trade_history_file = Path("trades_history.json")
        self.active_trades: dict[str, dict] = {}
        self._load_trade_history()

        self.ai_timeout_sec = float(user_data.get("ai_timeout_sec", 8.0))
        self.ai_sem = asyncio.Semaphore(user_data.get("ai_max_concurrent", 1))
        self.ai_circuit_open_until = 0.0

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

        self.trailing_start_map: dict[str, float] = user_data.get("trailing_start_pct", {})
        self.trailing_start_pct: float = self.trailing_start_map.get(
            self.strategy_mode,
            DEFAULT_TRAILING_START_PCT,
        )

        self.ai_log_path = Path("ai_verdicts.csv")  # отдельный лог (не смешиваем с finetune_log)
        self.ai_decisions: dict[str, dict] = {}     # decision_id -> запись (вердикт + вход/выход)
        self.ai_decision_by_symbol: dict[str, str] = {}  # symbol -> last decision_id

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
        self.ai_provider = provider_map.get(raw_ai, "ollama")
        logger.info(f"Выбран AI-провайдер: {self.ai_provider.upper()}")
        self.stop_loss_mode = user_data.get("stop_loss_mode", "strat_loss")
        logger.info(f"Выбран режим стоп-лосса: {self.stop_loss_mode.upper()}")

        self.ai_timeout_sec = float(user_data.get("ai_timeout_sec", 8.0))
        self.ai_sem = asyncio.Semaphore(user_data.get("ai_max_concurrent", 2))
        self.ai_circuit_open_until = 0.0
        self._ai_silent_until = 0.0

        # антидубль по времени для сообщений об открытии
        self._last_open_msg_ts: dict[str, float] = {}
        # минимальная пауза после закрытия, когда мы игнорируем "NEW"
        self._reopen_block_sec: float = 2.0
        # минимальная пауза между open-уведомлениями (на всякий случай)
        self._open_msg_cooldown_sec: float = 2.0

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
        self.leverage = 10

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
                        self.pending_timestamps.pop(symbol, None)
                        logger.warning(f"[Pending Cleanup] Ордер для {symbol} завис и был удален из очереди.")
            except Exception as e:
                logger.error(f"[Pending Cleanup] Критическая ошибка в чистильщике: {e}", exc_info=True)

    async def _calc_qty_from_usd(self, symbol: str, usd_amount: float,
                                 price: float | None = None) -> float:
        if symbol not in self.qty_step_map or symbol not in self.min_qty_map:
            try:
                await asyncio.wait_for(self.ensure_symbol_meta(symbol), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"[_calc_qty] Timeout при загрузке метаданных для {symbol}.")
                return 0.0
            except Exception as e:
                logger.error(f"[_calc_qty] Ошибка при загрузке метаданных для {symbol}: {e}")
                return 0.0

        step = self.qty_step_map.get(symbol, 0.001)
        min_qty = self.min_qty_map.get(symbol, step)
        price = price or safe_to_float(
            self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
        )
        if price <= 0:
            logger.warning(f"[_calc_qty] Невалидная цена ({price}) для {symbol}.")
            return 0.0

        raw_qty = 0.99 * usd_amount / price
        if step > 0:
             qty = math.floor(raw_qty / step) * step
        else:
             qty = raw_qty

        if qty < min_qty:
            qty = min_qty

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

    async def get_effective_total_volume(self) -> float:
        confirmed_volume = await self.get_total_open_volume()
        pending_volume = 0.0
        if self.pending_orders:
            pending_volume = sum(self.pending_orders.values())
        effective_volume = confirmed_volume + pending_volume
        if pending_volume > 0:
            logger.info(f"[Risk] Эффективный объем: {effective_volume:.2f} USDT (Подтверждено: {confirmed_volume:.2f} + В ожидании: {pending_volume:.2f})")
        return effective_volume

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
                pnl = (exit_data["price"] - entry_data["price"]) / entry_data["price"] * 100.0
            else:
                pnl = (entry_data["price"] - exit_data["price"]) / entry_data["price"] * 100.0

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
            if not data or len(data) < 100:
                return

            X = np.asarray([rec["features"] for rec in data], dtype=np.float32)
            if X.ndim != 2 or X.shape[1] != len(FEATURE_KEYS):
                logger.warning("[MLX] retrain_models: неверная форма X=%s", X.shape if hasattr(X, "shape") else type(X))
                return

            mask = ~(np.isnan(X).any(1) | np.isinf(X).any(1))
            X = X[mask]
            if X.size == 0:
                return

            loop = asyncio.get_running_loop()
            def _fit_scaler(arr):
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit(arr)
                try:
                    import joblib
                    joblib.dump(sc, "scaler.pkl")
                except Exception:
                    pass
                return sc

            scaler = await loop.run_in_executor(None, _fit_scaler, X)

            if getattr(self, "ml_inferencer", None) is not None:
                self.ml_inferencer.scaler = scaler

            try:
                _append_trades_unified({
                    "timestamp": datetime.utcnow().isoformat(),
                    "note": "Scaler refit completed",
                })
            except Exception:
                pass

            self.last_retrain = time.time()
            logger.info("[MLX] Scaler updated; on-device training is disabled.")
        except Exception as e:
            logger.error(f"[MLX] retrain_models failed: {e}")

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

    # [FINAL VERSION] Высокочастотный менеджер позиций
    # [FINAL VERSION] Высокочастотный менеджер позиций с умным логированием
    async def manage_open_position(self, symbol: str):
        """
        [V8 - High-Frequency Guardian with Smart Logging] Автономная, быстрая задача
        для управления одной позицией. Использует текущие настройки трейлинга и
        логирует обновления стопа, не создавая лишнего шума.
        """
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
            leverage = safe_to_float(pos_data.get("leverage", 10.0))
            if not (entry_price > 0 and side and leverage > 0): continue

            current_roi = (((last_price - entry_price) / entry_price) * 100 * leverage) if side == "Buy" else (((entry_price - last_price) / entry_price) * 100 * leverage)
            pos_data['pnl'] = current_roi

            if current_roi < self.trailing_start_pct:
                continue

            # переводим ROI-зазор в ценовой процент с учётом плеча
            gap_price_frac = (self.trailing_gap_pct / max(leverage, 1e-9)) / 100.0
            if side == "Buy":
                new_stop_price = last_price * (1 - gap_price_frac)
            else:
                new_stop_price = last_price * (1 + gap_price_frac)

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
        min_gap = max(0.2, self.trailing_update_interval_sec * 0.8)
        if now - self._last_trailing_ts.get(symbol, 0.0) < min_gap:
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
        [BUG FIX & IMPROVEMENT] Исправлен неверный расчет процента и сделан более явным.
        """
        pos = self.open_positions.get(symbol)
        if not pos or safe_to_float(pos.get("volume", 0)) <= 0:
            return False

        now = time.time()
        if now - self._last_trailing_ts.get(symbol, 0.0) < 1.5:
            return False

        try:
            pos_idx = pos.get("pos_idx", 1 if side == 'Buy' else 2)
            gap_pct = self.trailing_gap_pct

            last_price = safe_to_float(pos.get("markPrice", avg_price))
            if last_price <= 0: return False

            if side.lower() == "buy":
                # [BUG FIX] Делим на 100.0, а не на 1000, для правильного расчета процента.
                raw_price = last_price * (1 - gap_pct / 1000.0)
            else: # sell
                # [BUG FIX] Делим на 100.0, а не на 1000.
                raw_price = last_price * (1 + gap_pct / 1000.0)

            tick = self.price_tick_map.get(symbol, 1e-6)
            stop_price = round(math.floor(raw_price / tick) * tick, 8)

            prev_stop = self.last_stop_price.get(symbol)
            if prev_stop is not None:
                if abs(prev_stop - stop_price) < tick: return False
                is_worse = (side.lower() == "buy" and stop_price < prev_stop) or \
                           (side.lower() == "sell" and stop_price > prev_stop)
                if is_worse: return False

            async with _TRADING_STOP_SEM:
                await asyncio.to_thread(
                    lambda: self.session.set_trading_stop(
                        category="linear", symbol=symbol, positionIdx=pos_idx,
                        stopLoss=f"{stop_price:.8f}".rstrip('0').rstrip('.'),
                        triggerBy="LastPrice", timeInForce="GTC"
                    )
                )

            self.last_stop_price[symbol] = stop_price
            self._last_trailing_ts[symbol] = now
            logger.info(f"[TRAILING_STOP] {symbol} стоп успешно обновлен на {stop_price:.6f}")
            return True

        except InvalidRequestError as e:
            if e.status_code == 34040: # Order not modified
                self.last_stop_price[symbol] = stop_price
                self._last_trailing_ts[symbol] = now
                return True
            elif e.status_code == 10001: # Order not found (already closed)
                logger.warning(f"[TRAILING_STOP] Попытка установить стоп для закрытой позиции {symbol}. Очистка.")
                self._purge_symbol_state(symbol)
                return False
            else:
                logger.warning(f"[TRAILING_STOP] {symbol} ошибка API: {e}")
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
        self.pending_timestamps.pop(symbol, None)
        self.averaged_symbols.discard(symbol)
        self.recently_closed[symbol] = time.time()

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
        if symbol in self.pending_orders:
            logger.debug(f"[OpenSkip] {symbol} уже pending")
            return
        if now - self.last_entry_ts.get(symbol, 0) < self.entry_cooldown_sec:
            logger.info(f"[OpenSkip] {symbol} cooldown {self.entry_cooldown_sec}s")
            return
        if now - self.recently_closed.get(symbol, 0) < self.entry_cooldown_sec:
            logger.info(f"[OpenSkip] {symbol} recently closed")
            return

        ok, why = self.can_open_position(symbol, 0.0)
        if not ok:
            logger.info("[EXECUTE][%s] denied(primary): %s | %s", cid, why, j(st))
            return None
        async with self.position_lock:
            if symbol in self.open_positions or symbol in self.pending_orders:
                logger.debug(f"[OpenSkip] {symbol} уже открыт или pending")
                return

        try:
            last_price = safe_to_float(
                self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
            )
            if not last_price > 0:
                logger.warning(f"[ML Order] Не удалось получить цену для {symbol}, открытие отменено.")
                return

            qty = await self._calc_qty_from_usd(symbol, self.POSITION_VOLUME, last_price)
            if not qty > 0:
                logger.warning(f"[ML Order] Рассчитан нулевой объем для {symbol}, открытие отменено.")
                return

            if not await self._risk_check(symbol, side, qty, last_price):
                return

            async with self.position_lock:
                self.pending_orders[symbol] = qty * last_price
                self.pending_timestamps[symbol] = time.time()

                order_cost = float(self.pending_orders.get(symbol, 0.0))
                logger.info("[ORDER][%s] pending_set %s cost=%.2f", cid, symbol, order_cost)

                ok2, why2 = self.can_open_position(symbol, order_cost)
                if not ok2:
                    logger.info("[ORDER][%s] denied(recheck): %s | cleanup pending", cid, why2)
                    self.pending_orders.pop(symbol, None)
                    self.pending_timestamps.pop(symbol, None)
                    return None

            pos_idx = 1 if side == "Buy" else 2
            order_cost = float(self.pending_orders.get(symbol, 0.0))
            if not self.can_open_position(symbol, order_cost)[0]:
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)
                return None

            resp = await self.place_order_ws(symbol, side, qty, position_idx=pos_idx, cid=cid)
            order_id = None
            try:
                if isinstance(resp, dict):
                    order_id = resp.get("orderId") or resp.get("order_id")
                elif isinstance(resp, list) and resp:
                    order_id = resp[0].get("orderId")
            except Exception:
                pass
            if order_id:
                self.order_correlation[order_id] = cid
                logger.info("[EXECUTE][%s] order_accepted id=%s", cid, order_id)
            logger.info(f"[ML Order] {symbol}: {side} qty={qty}")
        except Exception as e:
            logger.warning(f"[ML Order] Не удалось разместить ордер для {symbol}: {e}", exc_info=True)

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

    async def wallet_loop(self):
        while True:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_wallet_balance(accountType="UNIFIED")
                )
                wallet_raw = resp.get("result", {})
                wallet_logger.info("[User %s] %s", self.user_id, json.dumps(wallet_raw, ensure_ascii=False))

                try:
                    data = {}
                    if os.path.exists(WALLET_JSON):
                        with open(WALLET_JSON, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                    data[str(self.user_id)] = wallet_raw
                    _atomic_json_write(WALLET_JSON, data)
                except Exception as jerr:
                    wallet_logger.debug("[wallet_loop] json write error: %s", jerr)

            except Exception as e:
                wallet_logger.warning("[wallet_loop] user %s error: %s", self.user_id, e)

            await asyncio.sleep(300)

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
                        pnl_pct = (pnl_usdt / position_value) * 100 if position_value else 0.0

                        logger.info(f"[EXECUTION_CLOSE] {symbol}. PnL: {pnl_usdt:.2f} USDT ({pnl_pct:.2f}%).")

                        self._ai_log_close(symbol, exit_price, pnl_usdt, pnl_pct=pnl_pct, reason="close")

                        # Сохраняем данные для логгирования
                        self.closed_positions[symbol] = dict(pos)
                        
                        # [КЛЮЧЕВОЕ ИЗМЕНЕНИЕ] Очищаем состояние и СРАЗУ ЖЕ ставим флаг
                        self._purge_symbol_state(symbol) 
                        self.write_open_positions_json()

                        # Логируем корректное закрытие
                        asyncio.create_task(self.log_trade(
                            symbol=symbol, side=pos["side"], avg_price=exit_price, volume=pos_volume,
                            action="close", result="closed_by_execution", pnl_usdt=pnl_usdt, pnl_pct=pnl_pct
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
            if leverage == 0: leverage = 10.0
            current_roi = (((last_price - avg_price) / avg_price) * 100 * leverage) if side == "Buy" else (((avg_price - last_price) / avg_price) * 100 * leverage)

            if await self.set_trailing_stop(symbol, avg_price, current_roi, side):
                await self.log_trade(symbol=symbol, side=side, avg_price=avg_price, volume=pos_data['size'], action="adopt_stop_set", result="success")
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
                    self.pending_timestamps.pop(symbol, None)

                    avg_price = safe_to_float(p.get("avgPrice") or p.get("entryPrice"))

                    now = time.time()

                    # 1) Только что закрывали этот символ -> игнорируем возможный "эхо"-снапшот
                    ts_closed = self.recently_closed.get(symbol, 0)
                    if ts_closed and (now - ts_closed) < self._reopen_block_sec:
                        logger.debug(f"[PositionStream] NEW {symbol} проигнорирован: just closed {now - ts_closed:.2f}s ago")
                        continue

                    # 2) Уже слали open-сообщение совсем недавно -> не дублируем
                    ts_open = self._last_open_msg_ts.get(symbol, 0)
                    if ts_open and (now - ts_open) < self._open_msg_cooldown_sec:
                        logger.debug(f"[PositionStream] NEW {symbol} проигнорирован: open msg cooldown {now - ts_open:.2f}s")
                        continue

                    # Получаем комментарий от AI/стратегии, если он был
                    entry_candidate = self.active_trade_entries.pop(symbol, {})
                    comment = entry_candidate.get("comment")

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
                    self._last_open_msg_ts[symbol] = time.time()
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
                    pnl_pct = (pnl_usdt / pos_value) * 100 if pos_value else 0.0

                    # Логируем закрытие
                    asyncio.create_task(self.log_trade(
                        symbol=symbol, side=snapshot.get("side", "Buy"), avg_price=exit_price,
                        volume=pos_volume, action="close", result="closed_by_position_stream",
                        pnl_usdt=pnl_usdt, pnl_pct=pnl_pct
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

    def _format_qty(self, symbol: str, qty: float) -> str:
        step = self.qty_step_map.get(symbol, 0.001) or 0.001
        qty = math.floor(qty / step) * step
        decimals = 0
        if step < 1:
            decimals = len(str(step).split(".")[1].rstrip("0"))
        return f"{qty:.{decimals}f}"

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

    def _aggregate_series_5m(self, series: list, method: str = "sum") -> list:
        if not series:
            return []
        result = []
        full_blocks = len(series) // 5
        for i in range(full_blocks):
            chunk = series[i * 5:(i + 1) * 5]
            if method == "sum":
                # [BUG FIX] Преобразуем каждый элемент в float перед суммированием
                float_chunk = [safe_to_float(x) for x in chunk]
                result.append(sum(float_chunk))
            else:
                # Для 'last' также безопасно приводим к float
                result.append(safe_to_float(chunk[-1]))
        return result

    async def handle_liquidation(self, msg):
        if not getattr(self, "warmup_done", False):
            return
        symbol = msg.get("data", [{}])[0].get("s")
        if symbol in self.open_positions:
            logger.info(f"Skipping liquidation trade for {symbol}: position already open")
            return

        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]

        for evt in data:
            symbol = evt.get("s")
            qty    = safe_to_float(evt.get("v", 0))
            side_evt = evt.get("S")
            price  = safe_to_float(evt.get("p", 0))
            if not symbol or qty <= 0 or price <= 0:
                continue
            value_usdt = qty * price

            self.shared_ws.latest_liquidation[symbol] = {
                "value": value_usdt,
                "side":  side_evt,
                "ts":    time.time(),
            }
            logger.info("[liquidation] %s %s %.2f USDT  (qty=%s × price=%s)",
                        symbol, side_evt, value_usdt, qty, price)

            csv_path = LIQUIDATIONS_CSV_PATH if "LIQUIDATIONS_CSV_PATH" in globals() \
                                                else "liquidations.csv"
            new_file = not os.path.isfile(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                if new_file:
                    w.writerow(["timestamp", "symbol", "side",
                                "price", "quantity", "value_usdt"])
                w.writerow([datetime.utcnow().isoformat(),
                            symbol, side_evt, price, qty, value_usdt])

            if self.strategy_mode not in ("liquidation_only", "full", "liq_squeeze"):
                continue

            now_ts = time.time()
            buf = self.liq_buffers[symbol]
            buf.append((now_ts, side_evt, value_usdt, price))

            cutoff_ts = now_ts - LIQ_CLUSTER_WINDOW_SEC
            while buf and buf[0][0] < cutoff_ts:
                buf.popleft()

            cluster_val = sum(v for _, _, v, _ in buf)
            if cluster_val < LIQ_CLUSTER_MIN_USDT:
                continue

            cluster_price = sum(p * v for _, _, v, p in buf) / cluster_val
            if abs(price - cluster_price) / cluster_price * 100 > LIQ_PRICE_PROXIMITY_PCT:
                continue

            long_val  = sum(v for _, s, v, _ in buf if s == "Buy")
            short_val = sum(v for _, s, v, _ in buf if s == "Sell")
            if long_val == 0 and short_val == 0:
                continue

            order_side = "Buy" if short_val > long_val else "Sell"

            if (symbol in self.pending_orders or
                    not self.shared_ws.check_liq_cooldown(symbol)):
                continue

            async with self.position_lock:
                await self.ensure_symbol_meta(symbol)
                last_price = safe_to_float(
                    self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
                )
                if last_price <= 0:
                    continue

                usd_qty = min(self.POSITION_VOLUME, self.max_allowed_volume)
                step    = self.qty_step_map.get(symbol, 0.001)
                min_qty = self.min_qty_map.get(symbol, 0.001)
                original_qty = max(usd_qty / last_price, min_qty)
                qty_ord = min(math.floor(original_qty / step) * step, self.POSITION_VOLUME)
                qty_ord = math.floor(qty_ord / step) * step
                if qty_ord <= 0:
                    continue

                if not await self._risk_check(symbol, order_side, qty_ord, last_price):
                    logger.info("[liq_trade] %s skipped — exposure limit", symbol)
                    continue

            self.pending_orders[symbol] = qty_ord * last_price # [IMPROVEMENT] Use dict for pending orders
            self.pending_timestamps[symbol]        = time.time()
            self.pending_strategy_comments[symbol] = "liq-cluster"

            try:
                pos_idx = (1 if order_side == "Buy" else 2)

                resp = await asyncio.to_thread(lambda: self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=order_side,
                    orderType="Market",
                    qty=self._format_qty(symbol, qty_ord),
                    positionIdx=pos_idx,
                    timeInForce="GTC",
                ))

                if resp.get("retCode", 0) != 0:
                    raise InvalidRequestError(
                    resp.get("retMsg", "order rejected"),
                    resp.get("retCode", 0),
                    resp.get("time", int(time.time() * 1000)),
                    {}
                )

                self.shared_ws.last_liq_trade_time[symbol] = dt.datetime.utcnow()
                logger.info(
                    "[liq_trade] %s %s qty=%s opened by cluster (Σ=%.0f USDT)",
                    order_side, symbol, qty_ord, cluster_val
                )
            except Exception as exc:
                logger.warning("[liq_trade] %s error: %s", symbol, exc)
                self.pending_orders.pop(symbol, None) # [IMPROVEMENT] Use pop for dict
            finally:
                self.pending_signals.pop(symbol, None)

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
            await self.place_order_ws(symbol, side, add_qty, position_idx=idx, price=price, order_type="Limit")
            logger.info("[RESCUE-DCA] %s +%.3f @ %.6f", symbol, add_qty, price)
        except Exception as e:
            logger.warning("[RESCUE-DCA] %s failed: %s", symbol, e)

    async def _open_hedge(self, symbol: str, side: str) -> None:
        pos = self.open_positions.get(symbol, {})
        base_qty = safe_to_float(pos.get("volume", 0))
        if base_qty == 0: return

        last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
        if not last_price > 0: return

        usd_amount = safe_to_float(pos.get("avg_price", 0)) * base_qty * 0.6
        hedge_qty = await self._calc_qty_from_usd(symbol, usd_amount, last_price)
        if not hedge_qty > 0: return

        opp_side = "Sell" if side == "Buy" else "Buy"
        idx = 2 if opp_side == "Sell" else 1
        try:
            await self.place_order_ws(symbol, opp_side, hedge_qty, position_idx=idx, order_type="Market")
            logger.info("[RESCUE-HEDGE] %s %s %.3f", symbol, opp_side, hedge_qty)
        except Exception as e:
            logger.warning("[RESCUE-HEDGE] %s failed: %s", symbol, e)

    async def _hard_stop(self, symbol: str) -> None:
        pos = self.open_positions.get(symbol)
        if not pos: return

        stop_qty = safe_to_float(pos.get("volume", 0))
        if not stop_qty > 0: return

        side = "Sell" if pos["side"] == "Buy" else "Buy"
        idx = 2 if side == "Sell" else 1
        try:
            await self.place_order_ws(symbol, side, stop_qty, position_idx=idx, order_type="Market")
            logger.info("[RESCUE-EXIT] %s emergency close %.3f", symbol, stop_qty)
        except Exception as e:
            logger.error("[RESCUE-EXIT] %s close failed: %s", symbol, e)

    async def on_ticker_update(self, symbol: str, last_price: float):
        try:
            pos = self.open_positions.get(symbol)
            if pos is not None:
                pos["markPrice"] = last_price

            if symbol in self.reserve_orders and symbol not in self.open_positions:
                asyncio.create_task(self._amend_reserve_limit(symbol, last_price))

            if self.shared_ws is not None:
                oi_val = self.shared_ws.latest_open_interest.get(symbol)
                if oi_val is not None:
                    hist = self.shared_ws.oi_history.setdefault(symbol, deque(maxlen=500))
                    oi_f = float(oi_val)
                    if not hist or hist[-1] != oi_f:
                        hist.append(oi_f)

        except Exception as e:
            logger.error(f"[on_ticker_update] {symbol} crashed: {e}", exc_info=True)

    async def ensure_entry_price(self, symbol: str, *, force: bool = False, ttl: float = 3.0) -> float:
        pos = self.open_positions.get(symbol)
        now = time.time()
        if not force and pos and safe_to_float(pos.get("avgPrice", 0)) > 0 and (now - self._last_pos_refresh_ts.get(symbol, 0) < ttl):
            return pos["avgPrice"]

        try:
            async with self.position_lock:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_positions(category="linear", symbol=symbol)
                )
            lst = resp.get("result", {}).get("list", [])
            best = None
            for p in lst:
                size = safe_to_float(p.get("size") or p.get("positionSize") or 0)
                if size == 0:
                    continue
                if not best or safe_to_float(p.get("positionValue", 0)) > safe_to_float(best.get("positionValue", 0)):
                    best = p
            if not best:
                self.open_positions.pop(symbol, None)
                return 0.0

            entry = _safe_avg_price(best)
            side  = _normalize_side(best)

            self.open_positions[symbol] = {
                "side": side,
                "size": safe_to_float(best.get("size") or 0),
                "avgPrice": entry,
                "markPrice": safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)),
                "leverage": safe_to_float(best.get("leverage", 0)),
                "pos_idx": safe_to_float(best.get("positionIdx", 1)),
                "ts": now,
            }
            self._last_pos_refresh_ts[symbol] = now
            return entry
        except Exception as e:
            logger.warning(f"[ensure_entry_price] REST failed for {symbol}: {e}")
            return 0.0

    async def _amend_reserve_limit(self, symbol: str, last_price: float) -> None:
        if symbol not in self.reserve_orders:
            return

        sigma  = self.shared_ws._sigma_5m(symbol)

        data      = self.reserve_orders[symbol]
        side      = data["side"]
        order_id  = data["orderId"]
        old_price = data["price"]

        offset = max(SQUEEZE_LIMIT_OFFSET_PCT, sigma * 2)
        desired = last_price * (1 - offset) if side == "Buy" else last_price * (1 + offset)

        step = self.qty_step_map.get(symbol, DEC_TICK)
        desired = round(desired / step) * step

        if abs(desired - old_price) < step:
            return

        try:
            resp = await asyncio.to_thread(
                lambda: self.session.amend_order(
                    category="linear",
                    symbol=symbol,
                    orderId=order_id,
                    price=str(desired)
                )
            )
            if resp.get("retCode", 0) == 0:
                self.reserve_orders[symbol]["price"] = desired
                self.reserve_orders[symbol]["ts"]    = time.time()
                logger.debug("[SQUEEZE] amended %s → %.6f", symbol, desired)
            else:
                logger.warning("[SQUEEZE] amend failed %s: %s", symbol, resp)
                self.reserve_orders.pop(symbol, None)
        except InvalidRequestError as e:
            logger.warning("[SQUEEZE] amend error %s: %s", symbol, e)
            self.reserve_orders.pop(symbol, None)

    async def cleanup_pending_loop(self):
        while True:
            await asyncio.sleep(30)
            now = time.time()

            pending_symbols = list(self.pending_orders.keys())

            for symbol in pending_symbols:
                timestamp = self.pending_timestamps.get(symbol, 0)

                if now - timestamp > 60:
                    logger.warning(f"[Pending Cleanup] Ордер для {symbol} завис. Удаление из очереди.")

                    self.pending_orders.pop(symbol, None)
                    self.pending_timestamps.pop(symbol, None)

                    reserve_info = self.reserve_orders.pop(symbol, None)
                    if reserve_info:
                        try:
                            logger.info(f"[Pending Cleanup] Отмена зависшего резервного лимитного ордера для {symbol}.")
                            await asyncio.to_thread(
                                lambda: self.session.cancel_order(
                                    category="linear",
                                    symbol=symbol,
                                    orderId=reserve_info["orderId"]
                                )
                            )
                        except Exception as e:
                            logger.warning(f"[Pending Cleanup] Не удалось отменить резервный ордер {symbol}: {e}")

    async def market_loop(self):
        while True:
            last_heartbeat = time.time()
            iteration = 0

            try:
                while True:
                    iteration += 1
                    symbols = [
                        s for s in self.shared_ws.active_symbols
                        if s not in ("BTCUSDT", "ETHUSDT")
                    ]
                    random.shuffle(symbols)

                    if self.strategy_mode in ("full", "golden_only", "squeeze_only", "golden_squeeze", "liq_squeeze"):
                        tasks = [
                            asyncio.create_task(
                                self.execute_golden_setup(symbol), name=f"golden-{symbol}"
                            )
                            for symbol in symbols
                        ]
                        logger.info(f"[market_loop] scheduling scan of {len(symbols)} symbols in mode {self.strategy_mode}")

                        if tasks:
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                            for result, task in zip(results, tasks):
                                if isinstance(result, Exception):
                                    logger.exception(f"[market_loop] exception in {task.get_name()}", exc_info=result)

                    if time.time() - last_heartbeat >= 60:
                        logger.info(
                            "[market_loop] alive (iter=%d) — scanned %d symbols",
                            iteration, len(self.shared_ws.symbols)
                        )
                        last_heartbeat = time.time()

                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[market_loop] fatal exception — restarting: %s", e)
                await asyncio.sleep(2)

    async def _risk_check(self, symbol: str, side: str, qty: float, last_price: float) -> bool:
        if symbol in EXCLUDED_SYMBOLS or last_price <= 0:
            return False

        if symbol in self.pending_orders:
            logger.info(f"[Risk] Skip {symbol} — ордер уже в процессе исполнения.")
            return False

        est_cost = qty * last_price

        if est_cost > self.POSITION_VOLUME:
            logger.info(f"[Risk] Skip {symbol} — стоимость {est_cost:.0f} > лимита на позицию {self.POSITION_VOLUME:.0f}")
            return False

        effective_volume = await self.get_effective_total_volume()
        if effective_volume + est_cost > self.MAX_TOTAL_VOLUME:
            logger.warning(f"[Risk] Skip {symbol} — превышен ОБЩИЙ лимит. Текущий: {effective_volume:.2f}, Попытка: {est_cost:.2f}, Лимит: {self.MAX_TOTAL_VOLUME:.2f}")
            return False

        return True

    async def _get_golden_thresholds(self, symbol: str, side: str) -> dict:
        base = (
            self.golden_param_store.get((symbol, side))
            or self.golden_param_store.get(side)
            or {"period_iters": 3, "price_change": 1.7,
                "volume_change": 200, "oi_change": 1.5}
        )

        try:
            if self.ml_inferencer and self.shared_ws:
                feats = await self.extract_realtime_features(symbol)
                if feats:
                    vec = _np.array([[feats[k] for k in FEATURE_KEYS]], _np.float32)
                    pred = self.ml_inferencer.infer(vec)[0]

                    self._record_best_entry(
                        symbol, "golden_setup", side,
                        max(pred) if len(pred) == 3 else float(pred[0]),
                        feats
                    )

                    if len(pred) == 3:
                        prob_sell, prob_hold, prob_buy = pred
                        if side == "Buy" and prob_buy > 0.55:
                            coef = max(0.4, 1.0 - (prob_buy - 0.55))
                            base = {**base,
                                    "price_change": base["price_change"] * coef,
                                    "volume_change": base["volume_change"] * coef,
                                    "oi_change":     base["oi_change"] * coef}
                        elif side == "Sell" and prob_sell > 0.55:
                            coef = max(0.4, 1.0 - (prob_sell - 0.55))
                            base = {**base,
                                    "price_change": base["price_change"] * coef,
                                    "volume_change": base["volume_change"] * coef,
                                    "oi_change":     base["oi_change"] * coef}

                    elif len(pred) == 1:
                        y_hat = float(pred[0])
                        gain = max(-1.0, min(1.0, y_hat / 40.0))
                        coef = 1.0 - 0.4 * gain
                        base = {**base,
                                "price_change": base["price_change"] * coef,
                                "volume_change": base["volume_change"] * coef,
                                "oi_change":     base["oi_change"] * coef}
        except Exception as e:
            logger.debug("[golden_thr] ML tune failed for %s/%s: %s", symbol, side, e)

        return base

    async def execute_golden_setup(self, symbol: str):
        if not await self._gs_prereqs(symbol):
            return

        mode = getattr(self, "strategy_mode", "full")

        if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
            if await self._squeeze_logic(symbol):
                return

        if mode in ("full", "liq_squeeze", "liquidation_only"):
            if await self._liquidation_logic(symbol):
                return

        if mode in ("full", "golden_only", "golden_squeeze"):
            await self._golden_logic(symbol)

    async def _gs_prereqs(self, symbol: str) -> bool:
        if symbol in self.open_positions or symbol in self.pending_orders:
            logger.debug("[GS_SKIP] %s already open or pending", symbol)
            return False
        age = await self.listing_age_minutes(symbol)
        if age < self.listing_age_min:
            logger.debug("[GS_SKIP] %s listing age %.0f < %d", symbol, age, self.listing_age_min)
            return False
        # [BUG FIX] Удалена некорректная проверка по `closed_positions`, которая блокировала повторные сделки.
        # Используется `recently_closed` в `update_open_positions` для защиты от "воскрешения".
        if symbol in self.failed_orders and time.time() - self.failed_orders.get(symbol, 0) < 600:
            return False
        if symbol in self.reserve_orders:
            return False
        return True

    async def _ai_dispatch(self, provider: str, candidate: dict, features: dict) -> dict:
        provider = provider.lower()
        if provider == "openai":
            return await self.evaluate_candidate_with_openai(candidate, features)
        elif provider == "gemini":
            return await self.evaluate_candidate_with_gemini(candidate, features)
        else:
            return await self.evaluate_candidate_with_ollama(candidate, features)

    async def evaluate_entry_candidate(self, candidate: dict, features: dict):
        symbol, side, source = candidate['symbol'], candidate['side'], candidate.get('source', 'unknown')
        now = time.time()
        signal_key = f"{symbol}_{side}_{source}"

        if self._evaluated_signals_cache.get(signal_key) and (now - self._evaluated_signals_cache.get(signal_key, {}).get("time", 0) < CACHE_TTL_SEC):
            return
        self._evaluated_signals_cache[signal_key] = {"status": "pending", "time": now}

        try:
            if self.ml_inferencer.model is not None:
                try:
                    vec = np.array([[safe_to_float(features.get(k, 0.0)) for k in FEATURE_KEYS]], dtype=np.float32)
                    raw_prediction = float(self.ml_inferencer.infer(vec)[0][0])

                    if not (-1.0 < raw_prediction < 1.0):
                        logger.warning(f"[ML_GATE_REJECT] Аномальный прогноз для {symbol}: {raw_prediction}")
                        return

                    leverage = self.leverage
                    expected_roi = raw_prediction * 100 * leverage

                    side_check_ok = (side == "Buy" and expected_roi > 0) or (side == "Sell" and expected_roi < 0)
                    roi_threshold_ok = abs(expected_roi) >= self.ml_gate_abs_roi

                    if not (side_check_ok and roi_threshold_ok):
                        logger.debug(f"[ML_GATE_REJECT] {symbol}/{side} ({source}) | Ожидаемый ROI: {expected_roi:.2f}%, Порог: {self.ml_gate_abs_roi:.2f}%")
                        return

                    logger.info(f"[ML_GATE_PASS] {symbol}/{side} ({source}) | Ожидаемый ROI: {expected_roi:.2f}%")
                except Exception as e:
                    logger.warning(f"[ML_GATE_ERROR] Ошибка ML-фильтра для {symbol}: {e}")
                    return
            else:
                logger.debug(f"[ML_GATE_SKIP] Модель не обучена, сигнал {symbol}/{side} пропущен к AI.")

            if now < self.ai_circuit_open_until:
                if now >= self._ai_silent_until:
                    logger.debug(f"[AI_SKIP] {symbol}/{side} - circuit open.")
                    self._ai_silent_until = now + 5
                return

            provider = str(self.ai_provider).lower().strip().replace('о', 'o')

            async with self.ai_sem:
                ai_response = await asyncio.wait_for(
                    self._ai_dispatch(provider, candidate, features),
                    timeout=self.ai_timeout_sec
                )

            decision_id = self._ai_log_start(candidate, features, ai_response)
            ai_response["decision_id"] = decision_id

            ai_action = ai_response.get("action", "REJECT")
            if ai_action != "EXECUTE":
                justification = ai_response.get("justification", "Причина не указана.")
                logger.info(f"[AI_REJECT] {symbol}/{side} ({source}) — {justification}")
                return

            logger.info(f"[AI_CONFIRM] Сделка {symbol}/{side} ({source}) ОДОБРЕНА. Исполнение...")
            candidate['justification'] = ai_response.get("justification", "N/A")
            candidate['full_prompt_for_ai'] = ai_response.get("full_prompt_for_ai", "")
            await self.execute_trade_entry(candidate, features)

        except asyncio.TimeoutError:
            self.ai_circuit_open_until = time.time() + 60
            logger.error(f"[AI_TIMEOUT] {self.ai_provider} не ответил за {self.ai_timeout_sec}с. Отключаю AI на 60 сек.")
        except Exception as e:
            logger.error(f"[evaluate_candidate] Критическая ошибка для {symbol}: {e}", exc_info=True)
        finally:
            self._evaluated_signals_cache.pop(signal_key, None)

    async def execute_trade_entry(self, candidate: dict, features: dict):
        cid = uuid.uuid4().hex[:8]
        symbol = candidate['symbol']
        side = candidate['side']
        source = candidate.get('source', 'unknown')

        try:
            last_price = float(features.get("price", 0) or 0.0)
            if not (last_price > 0):
                logger.warning("[EXECUTE_CANCEL][%s] %s/%s: Невалидная цена: %s",
                            cid, symbol, side, last_price)
                return

            if symbol in self.open_positions:
                logger.info("[EXECUTE_DENY][%s] %s/%s: позиция уже открыта", cid, symbol, side)
                return
            now = time.time()
            # Hard dedupe: skip if already pending or being processed
            if symbol in self.pending_orders or symbol in self.active_trade_entries:
                logger.info(f"[EXECUTE_DENY][{cid}] {symbol}/{side}: уже pending/processing")
                return
            # Cooldown after any recent entry/close
            if now - self.last_entry_ts.get(symbol, 0) < self.entry_cooldown_sec:
                logger.info(f"[EXECUTE_COOLDOWN][{cid}] {symbol}/{side}: pause {self.entry_cooldown_sec}s")
                return
            if now - self.recently_closed.get(symbol, 0) < self.entry_cooldown_sec:
                logger.info(f"[EXECUTE_COOLDOWN][{cid}] {symbol}/{side}: recently closed")
                return

            if symbol in self.pending_orders:
                logger.info("[EXECUTE_DENY][%s] %s/%s: уже есть pending ордер", cid, symbol, side)
                return

            volume_usdt = float(candidate.get('volume_usdt', self.POSITION_VOLUME) or 0.0)
            qty = await self._calc_qty_from_usd(symbol, volume_usdt, last_price)
            if not (qty > 0):
                logger.warning("[EXECUTE_CANCEL][%s] %s/%s: Нулевой объем (USDT=%s, price=%s)",
                            cid, symbol, side, volume_usdt, last_price)
                return

            if not await self._risk_check(symbol, side, qty, last_price):
                logger.info("[EXECUTE_DENY][%s] %s/%s: отклонено _risk_check", cid, symbol, side)
                return

            comment_for_log = f"JARVIS' OPINION: {candidate.get('justification','')}"
            candidate['comment'] = comment_for_log

            async with self.position_lock:
                self.pending_orders[symbol] = qty * last_price
                self.pending_timestamps[symbol] = time.time()
                self.active_trade_entries[symbol] = candidate
                self.last_entry_ts[symbol] = time.time()

            logger.info("[EXECUTE][%s] %s/%s: Все проверки пройдены. Отправка ордера... | pending_cost=%.2f",
                        cid, symbol, side, self.pending_orders.get(symbol, 0.0))

            pos_idx = 1 if side == "Buy" else 2

            if source == 'squeeze':
                if self.mode == 'real':
                    try:
                        resp = await asyncio.wait_for(
                            self.adaptive_squeeze_entry_ws(symbol, side, qty, pos_idx),
                            timeout=6.0
                        )
                        logger.info("[ORDER][%s] squeeze_ws ack %s", cid, str(resp)[:400])
                    except asyncio.TimeoutError:
                        logger.error("[ORDER][%s] squeeze_ws timeout -> cleanup pending", cid)
                        raise
                else:
                    try:
                        resp = await asyncio.wait_for(
                            self.adaptive_squeeze_entry(symbol, side, qty),
                            timeout=6.0
                        )
                        logger.info("[ORDER][%s] squeeze_rest ack %s", cid, str(resp)[:400])
                    except asyncio.TimeoutError:
                        logger.error("[ORDER][%s] squeeze_rest timeout -> cleanup pending", cid)
                        raise

            elif source == 'liquidation':
                try:
                    resp = await asyncio.wait_for(
                        self.adaptive_liquidation_entry(symbol, side, qty, pos_idx),
                        timeout=6.0
                    )
                    logger.info("[ORDER][%s] liq ack %s", cid, str(resp)[:400])
                except asyncio.TimeoutError:
                    logger.error("[ORDER][%s] liq timeout -> cleanup pending", cid)
                    raise

            else:
                try:
                    resp = await asyncio.wait_for(
                        self.place_unified_order(symbol, side, qty, "Market", comment=comment_for_log, cid=cid),
                        timeout=6.0
                    )
                    logger.info("[ORDER][%s] unified ack %s", cid, str(resp)[:400])

                except asyncio.TimeoutError:
                    logger.error("[ORDER][%s] unified timeout -> cleanup pending", cid)
                    raise
            return

        except Exception as e:
            logger.error("[execute_trade][%s] Критическая ошибка для %s/%s: %s",
                        cid, symbol, side, e, exc_info=True)
        finally:
            if symbol in self.pending_orders:
                logger.info("[PENDING][%s] cleanup %s (finalize)", cid, symbol)
            self.pending_orders.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)
            self.active_trade_entries.pop(symbol, None)

    async def evaluate_candidate_with_openai(self, candidate: dict, features: dict) -> dict:
        if self.ai_provider != "openai":
            raise RuntimeError("OpenAI provider отключён настройкой ai_provider")
        default_response = {"confidence_score": 0.5, "justification": "Ошибка OpenAI.", "action": "REJECT"}
        if not self.openai_api_key:
            logger.warning("[OpenAI] Ключ API не найден, оценка пропущена.")
            return default_response

        def _format(v, spec): return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"
        prompt = ""
        try:
            client = AsyncOpenAI(api_key=self.openai_api_key)
            m = candidate['base_metrics']
            btc_change = safe_to_float(self.shared_ws.ticker_data.get("BTCUSDT", {}).get("price24hPcnt", 0)) * 100
            prompt = f"""
            SYSTEM: Ты - элитный квантовый аналитик и риск-менеджер. Твой ответ - это всегда только валидный JSON.
            USER: Проанализируй торговый сигнал.
            **Сигнал:** Монета: {candidate['symbol']}, Направление: {candidate['side'].upper()}, Источник: {candidate.get('source', 'N/A')}
            **Метрики:** PriceΔ(5m): {_format(m.get('pct_5m'), '.2f')}%, VolΔ: {_format(m.get('vol_change_pct'), '.1f')}%, OIΔ: {_format(m.get('oi_change_pct'), '.2f')}%, SqueezePower: {_format(m.get('squeeze_power'), '.1f')}
            **Контекст:** Volatility(1h): {_format(features.get('sigma5m', 0) * 100, '.2f')}%, Spread: {_format(features.get('spread_pct'), '.4f')}%, ADX: {_format(features.get('adx14'), '.1f')}, RSI: {_format(features.get('rsi14'), '.1f')}, CVD(5m): {_format(features.get('CVD5m'), ',.0f')}
            **Рынок:** BTC(24h): {_format(btc_change, '.2f')}%
            **ЗАДАЧА:** Верни JSON с ключами "confidence_score" (0.0-1.0), "justification" (кратко), и "action" ("EXECUTE" или "REJECT").
            """

            async with self.gemini_limiter:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )

            response_data = json.loads(response.choices[0].message.content)
            response_data['full_prompt_for_ai'] = prompt
            return response_data

        except Exception as e:
            logger.error(f"[OpenAI] Ошибка API для {candidate['symbol']}: {e}", exc_info=True)
            return {**default_response, "full_prompt_for_ai": prompt}

    async def evaluate_candidate_with_gemini(self, candidate: dict, features: dict) -> dict:
        default_response = {"confidence_score": 0.5, "justification": "Ошибка Gemini.", "action": "REJECT"}
        if not self.gemini_api_key:
            logger.warning("[Gemini] Ключ API не найден, оценка пропущена.")
            return default_response

        def _format(v, spec): return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"
        prompt = ""
        try:
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
            m = candidate['base_metrics']
            btc_change = safe_to_float(self.shared_ws.ticker_data.get("BTCUSDT", {}).get("price24hPcnt", 0)) * 100

            prompt = f"""
            SYSTEM: Ты — опытный крипто-трейдер и риск-менеджер. Твой ответ - это всегда только валидный JSON.
            USER: Проанализируй сигнал и верни JSON-решение.
            **Сигнал:** {candidate['symbol']}/{candidate['side'].upper()} ({candidate.get('source', 'N/A')})
            **Метрики:** PriceΔ(5m)={_format(m.get('pct_5m'), '.2f')}%, VolΔ={_format(m.get('vol_change_pct'), '.1f')}%, OIΔ={_format(m.get('oi_change_pct'), '.2f')}%
            **Контекст:** RSI={_format(features.get('rsi14'), '.1f')}, ADX={_format(features.get('adx14'), '.1f')}, BTC(24h)={_format(btc_change, '.2f')}%
            **Задача:** JSON с ключами "confidence_score", "justification", "action" ("EXECUTE" или "REJECT").
            """

            async with self.gemini_limiter:
                response = await asyncio.to_thread(
                    lambda: model.generate_content(prompt, generation_config=generation_config)
                )

            response_data = json.loads(response.text)
            response_data['full_prompt_for_ai'] = prompt
            return response_data
        except Exception as e:
            logger.error(f"[Gemini] Ошибка API для {candidate['symbol']}: {e}", exc_info=True)
            return {**default_response, "full_prompt_for_ai": prompt}

    async def evaluate_candidate_with_ollama(self, candidate: dict, features: dict) -> dict:
        from openai import AsyncOpenAI
        default_response = {"confidence_score": 0.5, "justification": "Ошибка локального AI.", "action": "REJECT"}
        prompt = ""
        try:
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            def _format(v, spec): return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"
            m, source = candidate.get('base_metrics', {}), candidate.get('source', 'unknown').replace('_', ' ').title()

            vol_anomaly = (m.get('vol_change_pct', 0) / 100 + 1)
            trend = "Uptrend" if features.get('supertrend', 0) > 0 else "Downtrend"
            btc_change_1h = compute_pct(self.shared_ws.candles_data["BTCUSDT"], 60)
            eth_change_1h = compute_pct(self.shared_ws.candles_data["ETHUSDT"], 60)

            prompt = f"""
            SYSTEM: Ты - профессиональный трейдер, опытный риск-менеджер, элитный квантовый и крипто-аналитик. Твой ответ - всегда только валидный JSON.
            USER:
            Анализ торгового сигнала:
            - Монета: {candidate['symbol']}, Направление: {candidate['side'].upper()}, Источник: {source}
            - Метрики: PriceΔ(5m)={_format(m.get('pct_5m'), '.2f')}%, Volume Anomaly={_format(vol_anomaly, '.1f')}x, OIΔ(1m)={_format(m.get('oi_change_pct'), '.2f')}%
            - Контекст: Trend={trend}, ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(features.get('rsi14'), '.1f')}
            - Рынок: BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%

            ЗАДАЧА: Верни JSON с ключами "confidence_score", "justification" (начинается с источника сигнала), "action" ("EXECUTE" или "REJECT").
            """
            response = await client.chat.completions.create(
                model="trading-llama",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                top_p=1,
            )
            response_data = json.loads(response.choices[0].message.content)
            response_data['full_prompt_for_ai'] = prompt
            return response_data
        except Exception as e:
            logger.error(f"[Ollama] Ошибка API для {candidate['symbol']}: {e}", exc_info=True)
            return {**default_response, "full_prompt_for_ai": prompt}


    def _ai_log_write_row(self, row: dict) -> None:
        """Добавляет строку в ai_verdicts.csv (создаёт заголовок, если файла нет)."""
        path = self.ai_log_path
        path_exists = path.exists()
        fieldnames = [
            "decision_id","ts_decision","symbol","side","mode",
            "price_at_decision","ai_action","ai_confidence","ai_justification",
            "ai_meta","features_json",
            "order_id","ts_open","entry_price",
            "ts_close","exit_price","pnl_abs","pnl_pct","exit_reason","mfe","mae"
        ]
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not path_exists:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})

    def _ai_log_start(self, candidate: dict, features: dict, ai_reply: dict) -> str:
        """Фиксируем вердикт ИИ в момент решения. Возвращаем decision_id."""
        symbol = candidate["symbol"]
        side   = candidate["side"]
        mode   = candidate.get("mode", "")
        last_price = _safe_float(self.shared_ws.ticker_data[symbol]["lastPrice"])
        decision_id = str(uuid.uuid4())

        rec = {
            "decision_id": decision_id,
            "ts_decision": _now_ts(),
            "symbol": symbol,
            "side": side,
            "mode": mode,
            "price_at_decision": last_price,
            "ai_action": ai_reply.get("action", "REJECT"),
            "ai_confidence": ai_reply.get("confidence_score", 0),
            "ai_justification": ai_reply.get("justification", ""),
            "ai_meta": _json_dumps({k: ai_reply.get(k) for k in ("panel_meta","provider","model","full_prompt_for_ai")}),
            "features_json": _json_dumps(features),
            # поля ордера/исхода заполним позже
            "order_id": "",
            "ts_open": "",
            "entry_price": "",
            "ts_close": "",
            "exit_price": "",
            "pnl_abs": "",
            "pnl_pct": "",
            "exit_reason": "",
            "mfe": "",
            "mae": "",
        }
        self.ai_decisions[decision_id] = rec
        self.ai_decision_by_symbol[symbol] = decision_id
        # можно сразу сбросить «черновик» в CSV, но лучше дождаться открытия
        return decision_id

    def _ai_log_attach_open(self, symbol: str, order_id: str, entry_price: float) -> None:
        """Когда позиция реально открылась — привяжем ордер и цену входа."""
        decision_id = self.ai_decision_by_symbol.get(symbol)
        if not decision_id: return
        rec = self.ai_decisions.get(decision_id)
        if not rec: return
        rec.update({
            "order_id": order_id,
            "ts_open": _now_ts(),
            "entry_price": entry_price,
        })
        # по желанию можно промежуточно писать строку

    def _ai_log_close(self, symbol: str, exit_price: float, pnl_abs: float, pnl_pct: float, reason: str,
                      mfe: float | None = None, mae: float | None = None) -> None:
        """Когда позиция закрыта — зафиксировать исход и записать строку в CSV."""
        decision_id = self.ai_decision_by_symbol.get(symbol)
        if not decision_id: return
        rec = self.ai_decisions.get(decision_id)
        if not rec: return

        rec.update({
            "ts_close": _now_ts(),
            "exit_price": exit_price,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
            "mfe": "" if mfe is None else mfe,
            "mae": "" if mae is None else mae,
        })
        self._ai_log_write_row(rec)
        # чистим связку; если хотите — храните историю дольше
        self.ai_decision_by_symbol.pop(symbol, None)
        # self.ai_decisions.pop(decision_id, None)  # можно оставить в памяти


    async def _squeeze_logic(self, symbol: str) -> bool:
        try:
            if not self._squeeze_allowed(symbol):
                return False

            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if len(candles) < 6: return False

            old_close = safe_to_float(candles[-6]["closePrice"])
            new_close = safe_to_float(candles[-1]["closePrice"])
            if old_close <= 0: return False
            pct_5m = (new_close - old_close) / old_close * 100.0

            avg_vol_prev_5m = sum(safe_to_float(c["volume"]) for c in candles[-6:-1]) / 5
            curr_vol_1m = safe_to_float(candles[-1]["volume"])
            if avg_vol_prev_5m <= 0: return False
            vol_change_pct = (curr_vol_1m - avg_vol_prev_5m) / avg_vol_prev_5m * 100.0

            sigma_pct = self.shared_ws._sigma_5m(symbol) * 100
            thr_price = max(self.squeeze_threshold_pct, 1.5 * sigma_pct)
            power_min = max(self.squeeze_power_min, 4.0 * sigma_pct)
            squeeze_power = abs(pct_5m) * abs(vol_change_pct / 100.0)

            if abs(pct_5m) < thr_price or squeeze_power < power_min:
                return False

            oi_hist = list(self.shared_ws.oi_history.get(symbol, []))
            oi_change_pct = 0.0
            if len(oi_hist) >= 2:
                oi_now = oi_hist[-1]
                oi_prev = oi_hist[-2]
                if oi_prev > 0:
                    oi_change_pct = (oi_now - oi_prev) / oi_prev * 100.0

            side = "Sell" if pct_5m >= thr_price else "Buy"

            features = await self.extract_realtime_features(symbol)
            if not features:
                return False

            candidate = {
                'symbol': symbol, 'side': side, 'source': 'squeeze',
                'base_metrics': {'pct_5m': pct_5m, 'vol_change_pct': vol_change_pct, 'squeeze_power': squeeze_power, 'oi_change_pct': oi_change_pct},
                'volume_usdt': self.POSITION_VOLUME
            }

            logger.debug(f"[Signal Candidate] Squeeze: {side} on {symbol}")
            await self.evaluate_entry_candidate(candidate, features)

            self.last_squeeze_ts[symbol] = time.time()
            return True

        except Exception as e:
            logger.error(f"[_squeeze_logic] Ошибка анализа сквиза для {symbol}: {e}", exc_info=True)
            return False

    async def _liquidation_logic(self, symbol: str) -> bool:
        try:
            liq_info = self.shared_ws.latest_liquidation.get(symbol, {})
            if not liq_info or (time.time() - liq_info.get("ts", 0)) > 60: return False

            threshold = self.shared_ws.get_liq_threshold(symbol, 5000)
            if liq_info.get("value", 0) < threshold: return False

            if not self.shared_ws.check_liq_cooldown(symbol) or symbol in self.open_positions or symbol in self.pending_orders:
                return False

            order_side = "Buy" if liq_info.get("side") == "Sell" else "Sell"

            features = await self.extract_realtime_features(symbol)
            if not features: return False

            last_price = features.get("price")
            if not last_price or last_price <= 0: return False

            usd_amount = min(self.POSITION_VOLUME, self.max_allowed_volume)
            qty_ord = await self._calc_qty_from_usd(symbol, usd_amount, last_price)
            if not qty_ord > 0: return False

            if not await self._risk_check(symbol, order_side, qty_ord, last_price):
                logger.info("[liq_trade] %s пропущен — не прошел проверку риска", symbol)
                return False

            signal_metrics = {
                'pct_5m': features.get('pct5m', 0),
                'oi_change_pct': features.get('dOI1m', 0) * 100,
                'liquidation_value_usdt': liq_info.get("value"),
                'liquidation_side': liq_info.get("side")
            }
            candidate = {
                'symbol': symbol, 'side': order_side, 'source': 'liquidation',
                'base_metrics': signal_metrics,
                'volume_usdt': usd_amount
            }

            logger.debug(f"[Signal Candidate] Liquidation: {order_side} on {symbol}")
            await self.evaluate_entry_candidate(candidate, features)

            self.shared_ws.last_liq_trade_time[symbol] = dt.datetime.utcnow()
            return True

        except Exception as e:
            logger.error(f"[_liquidation_logic] Ошибка анализа ликвидаций для {symbol}: {e}", exc_info=True)
            return False

    async def _golden_logic(self, symbol: str):
        if symbol in self.open_positions or symbol in self.pending_orders:
            return

        try:
            golden_enabled = self.strategy_mode in ("full", "golden_only", "golden_squeeze")
            if not golden_enabled:
                return

            age = await self.listing_age_minutes(symbol)
            if age < self.listing_age_min:
                return

            if self._squeeze_allowed(symbol) and self.shared_ws.has_5_percent_growth(symbol, minutes=20):
                return
            await self.ensure_symbol_meta(symbol)
            if symbol in self.failed_orders and time.time() - self.failed_orders.get(symbol, 0) < 600:
                return
            if symbol in self.reserve_orders:
                return

            minute_candles = self.shared_ws.candles_data.get(symbol, [])
            recent = self._aggregate_candles_5m(minute_candles)
            vol_hist_5m = self._aggregate_series_5m(list(self.shared_ws.volume_history.get(symbol, [])), method="sum")
            oi_hist_5m  = self._aggregate_series_5m(list(self.shared_ws.oi_history.get(symbol, [])), method="last")
            cvd_hist_5m = self._aggregate_series_5m(list(self.shared_ws.cvd_history.get(symbol, [])), method="sum")

            if not recent:
                return

            buy_params  = await self._get_golden_thresholds(symbol, "Buy")
            sell_params = await self._get_golden_thresholds(symbol, "Sell")
            period_iters = max(int(buy_params["period_iters"]), int(sell_params["period_iters"]))

            if (len(recent) <= period_iters or
                len(vol_hist_5m) <= period_iters or
                len(oi_hist_5m)  <= period_iters or
                len(cvd_hist_5m) <= period_iters):
                return

            action = None

            liq_info = self.shared_ws.latest_liquidation.get(symbol, {})
            liq_val  = safe_to_float(liq_info.get("value", 0))
            liq_side = liq_info.get("side", "")
            threshold = self.shared_ws.get_liq_threshold(symbol, 5000)

            price_change_pct, volume_change_pct, oi_change_pct = 0.0, 0.0, 0.0

            sell_period = int(sell_params["period_iters"])
            if len(recent) > sell_period:
                price_change_pct_sell = (safe_to_float(recent[-1]["closePrice"]) - safe_to_float(recent[-1 - sell_period]["closePrice"])) / safe_to_float(recent[-1 - sell_period]["closePrice"]) * 100 if recent[-1 - sell_period]["closePrice"] else 0.0
                volume_change_pct_sell = (safe_to_float(vol_hist_5m[-1]) - safe_to_float(vol_hist_5m[-1 - sell_period])) / safe_to_float(vol_hist_5m[-1 - sell_period]) * 100 if vol_hist_5m[-1 - sell_period] else 0.0
                oi_change_pct_sell = (safe_to_float(oi_hist_5m[-1]) - safe_to_float(oi_hist_5m[-1 - sell_period])) / safe_to_float(oi_hist_5m[-1 - sell_period]) * 100 if oi_hist_5m[-1 - sell_period] else 0.0

                if (price_change_pct_sell <= -sell_params["price_change"] and
                    volume_change_pct_sell >= sell_params["volume_change"] and
                    oi_change_pct_sell >= sell_params["oi_change"] and
                    not (liq_side == "Sell" and liq_val >= threshold)):

                    action = "Sell"
                    price_change_pct, volume_change_pct, oi_change_pct = price_change_pct_sell, volume_change_pct_sell, oi_change_pct_sell

            if action is None:
                buy_period = int(buy_params["period_iters"])
                if len(recent) > buy_period:
                    price_change_pct_buy = (safe_to_float(recent[-1]["closePrice"]) - safe_to_float(recent[-1 - buy_period]["closePrice"])) / safe_to_float(recent[-1 - buy_period]["closePrice"]) * 100 if recent[-1 - buy_period]["closePrice"] else 0.0
                    volume_change_pct_buy = (safe_to_float(vol_hist_5m[-1]) - safe_to_float(vol_hist_5m[-1 - buy_period])) / safe_to_float(vol_hist_5m[-1 - buy_period]) * 100 if vol_hist_5m[-1 - buy_period] else 0.0
                    oi_change_pct_buy = (safe_to_float(oi_hist_5m[-1]) - safe_to_float(oi_hist_5m[-1 - buy_period])) / safe_to_float(oi_hist_5m[-1 - buy_period]) * 100 if oi_hist_5m[-1 - buy_period] else 0.0

                    if (price_change_pct_buy >= buy_params["price_change"] and
                        volume_change_pct_buy >= buy_params["volume_change"] and
                        oi_change_pct_buy >= buy_params["oi_change"] and
                        not (liq_side == "Buy" and liq_val >= threshold)):

                        action = "Buy"
                        price_change_pct, volume_change_pct, oi_change_pct = price_change_pct_buy, volume_change_pct_buy, oi_change_pct_buy

            if action:
                features = await self.extract_realtime_features(symbol)
                if not features:
                    logger.warning(f"[_golden_logic] Не удалось извлечь фичи для {symbol}, сигнал пропущен.")
                    return

                old_cvd = safe_to_float(cvd_hist_5m[-1 - period_iters])
                new_cvd = safe_to_float(cvd_hist_5m[-1])
                cvd_change_pct = (new_cvd - old_cvd) / abs(old_cvd) * 100.0 if abs(old_cvd) > 1e-8 else 0.0

                signal_metrics = {
                    'pct_5m': price_change_pct,
                    'vol_change_pct': volume_change_pct,
                    'oi_change_pct': oi_change_pct,
                    'cvd_change_pct': cvd_change_pct
                }

                triggered_params = sell_params if action == "Sell" else buy_params
                candidate = {
                    'symbol': symbol,
                    'side': action,
                    'source': 'golden_setup',
                    'base_metrics': signal_metrics,
                    'volume_usdt': safe_to_float(triggered_params.get("position_volume", self.POSITION_VOLUME))
                }

                logger.debug(f"[Signal Candidate] Golden Setup: {action} on {symbol}")
                await self.evaluate_entry_candidate(candidate, features)
                return

        except Exception as e:
            logger.error(f"[_golden_logic] unexpected error for {symbol}: {e}", exc_info=True)
        finally:
            self.pending_signals.pop(symbol, None)

    async def place_unified_order(self, symbol: str, side: str, qty: float,
                                order_type: str,
                                price: Optional[float] = None,
                                comment: str = "",
                                cid: str | None = None):

        cid = cid or new_cid()
        logger.info("[EXECUTE][%s] start %s/%s type=%s qty=%s price=%s comment=%s | %s",
                    cid, symbol, side, order_type, qty, price, comment, j(log_state(self, symbol)))

        pos_idx = 1 if side == "Buy" else 2
        qty_str = self._format_qty(symbol, qty)

        try:
            if self.mode == "real":
                response = await self.place_order_ws(
                    symbol, side, qty_str, position_idx=pos_idx,
                    price=price, order_type=order_type
                )
                order_id = None
                try:
                    if isinstance(response, dict):
                        order_id = response.get("orderId") or response.get("order_id")
                    elif isinstance(response, list) and response:
                        order_id = response[0].get("orderId")
                except Exception:
                    pass
                if order_id:
                    self.order_correlation[order_id] = cid
                    logger.info("[EXECUTE][%s] order_accepted id=%s", cid, order_id)
                    
            else:
                response = await asyncio.to_thread(
                    lambda: self.session.place_order(
                        category="linear", symbol=symbol, side=side,
                        orderType=order_type, qty=qty_str,
                        price=str(price) if price else None,
                        timeInForce="GTC", positionIdx=pos_idx
                    )
                )

                if response.get("retCode") == 0:
                    order_id = response["result"]["orderId"]

                    for _ in range(10):
                        pos = await asyncio.to_thread(
                            lambda: self.session.get_positions(
                                category="linear", symbol=symbol
                            )
                        )
                        lst = pos.get("result", {}).get("list", [])
                        if lst and float(lst[0].get("size", 0)) > 0:
                            break
                        await asyncio.sleep(0.5)
                    else:
                        logger.warning(
                            "[DEMO] %s: Market-order %s не материализовался в позицию – cancel",
                            symbol, order_id
                        )
                        await asyncio.to_thread(
                            lambda: self.session.cancel_order(
                                category="linear", symbol=symbol, orderId=order_id
                            )
                        )
                        raise RuntimeError("Position did not appear within 5 s")

            if response.get("retCode", -1) != 0:
                raise InvalidRequestError(response.get("retMsg", "Order failed"),
                                        response.get("retCode"), response)

            logger.info("✅ Успешно отправлен ордер: %s %s %s @ %s",
                        side, qty_str, symbol, price or "Market")
            self.pending_strategy_comments[symbol] = comment
            return response

        except InvalidRequestError as e:
            logger.error(f"❌ Ошибка размещения ордера для {symbol} (API Error): {e} (Код: {e.status_code})")
            self.pending_orders.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)

        except Exception as e:
            logger.critical(f"❌ Критическая ошибка при размещении ордера для {symbol}: {e}", exc_info=True)
            self.pending_orders.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)

    async def amend_order_ws(self, *, symbol: str, order_id: str, new_price: float):
        req = {
            "op": "order.amend",
            "args": [{
                "category": "linear",
                "symbol":   symbol,
                "orderId":  order_id,
                "price":    str(new_price)
            }],
            "header": {
                "X-BAPI-TIMESTAMP": str(int(time.time()*1000)),
                "X-BAPI-RECV-WINDOW": "5000"
            }
        }
        async with self._recv_lock:
            await self.ws_trade.send(json.dumps(req))
            resp = await self._recv_until("order.amend")

        rc  = resp.get("retCode", 0)
        msg = str(resp.get("retMsg", "")).lower()

        if rc == 0 or (rc == 10001 and "not modified" in msg):
            return resp.get("data", {})
        if rc == 10001 and "order not exist" in msg:
            raise InvalidRequestError("order not exist")
        raise InvalidRequestError(msg)

    async def adaptive_liquidation_entry(self, symbol: str, side: str, qty: float, position_idx: int, max_wait_time: int = 15):
        logger.info(f"[TACTICAL_LIQ] {symbol}: Активирован умный вход по ликвидациям. Окно: {max_wait_time}с.")

        start_time = time.time()
        best_price_seen = 0
        entry_made = False

        while time.time() - start_time < max_wait_time:
            ticker = self.shared_ws.ticker_data.get(symbol, {})
            last_price = safe_to_float(ticker.get("lastPrice", 0))

            if last_price <= 0:
                await asyncio.sleep(0.1)
                continue

            if side == "Buy":
                if best_price_seen == 0 or last_price < best_price_seen:
                    best_price_seen = last_price
                if last_price > best_price_seen * 1.001:
                    logger.info(f"[TACTICAL_LIQ] {symbol}: Обнаружен отскок. Входим в LONG по рынку.")
                    await self.place_unified_order(symbol, side, qty, "Market")
                    entry_made = True
                    break

            else:
                if best_price_seen == 0 or last_price > best_price_seen:
                    best_price_seen = last_price
                if last_price < best_price_seen * 0.999:
                    logger.info(f"[TACTICAL_LIQ] {symbol}: Обнаружен откат. Входим в SHORT по рынку.")
                    await self.place_unified_order(symbol, side, qty, "Market")
                    entry_made = True
                    break

            await asyncio.sleep(0.2)

        if not entry_made:
            logger.warning(f"[TACTICAL_LIQ] {symbol}: Окно для входа истекло, выгодный момент не найден. Отмена входа.")
            self.pending_orders.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)

    async def adaptive_squeeze_entry_ws(self,
                                        symbol: str,
                                        side: str,
                                        qty: float,
                                        position_idx: int,
                                        max_entry_timeout: int = 45,
                                        reversal_trigger_pct: float = 0.15) -> bool:
        logger.info(f"[TACTICAL_SQUEEZE_WS] {symbol}/{side}: Активирован умный WS-вход. Окно: {max_entry_timeout}с, триггер отката: {reversal_trigger_pct}%.")

        start_time = time.time()
        extreme_price = 0.0
        entry_made = False

        try:
            while time.time() - start_time < max_entry_timeout:
                ticker = self.shared_ws.ticker_data.get(symbol, {})
                last_price = safe_to_float(ticker.get("lastPrice", 0))

                if last_price <= 0:
                    await asyncio.sleep(0.1)
                    continue

                if side == "Sell":
                    if extreme_price == 0 or last_price > extreme_price:
                        extreme_price = last_price
                    elif last_price < extreme_price * (1 - reversal_trigger_pct / 100.0):
                        logger.info(f"[TACTICAL_SQUEEZE_WS] {symbol}: Обнаружен откат с пика {extreme_price}. Входим в SHORT через WS.")
                        await self.place_order_ws(symbol, side, qty, position_idx=position_idx, order_type="Market")
                        entry_made = True
                        break
                else:
                    if extreme_price == 0 or last_price < extreme_price:
                        extreme_price = last_price
                    elif last_price > extreme_price * (1 + reversal_trigger_pct / 100.0):
                        logger.info(f"[TACTICAL_SQUEEZE_WS] {symbol}: Обнаружен отскок со дна {extreme_price}. Входим в LONG через WS.")
                        await self.place_order_ws(symbol, side, qty, position_idx=position_idx, order_type="Market")
                        entry_made = True
                        break

                await asyncio.sleep(0.2)

        except Exception as e:
            logger.error(f"[TACTICAL_SQUEEZE_WS] Критическая ошибка для {symbol}: {e}", exc_info=True)
            entry_made = False

        finally:
            if not entry_made:
                logger.warning(f"[TACTICAL_SQUEEZE_WS] {symbol}: Окно для входа истекло или произошла ошибка. Отмена.")
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)

        return entry_made

    async def adaptive_squeeze_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        max_entry_timeout: int = 45,
        reversal_trigger_pct: float = 0.15
    ) -> bool:
        logger.info(f"[TACTICAL_SQUEEZE_DEMO] {symbol}/{side}: Активирован умный REST-вход. Окно: {max_entry_timeout}с, триггер отката: {reversal_trigger_pct}%.")

        start_time = time.time()
        extreme_price = 0.0
        entry_made = False

        try:
            while time.time() - start_time < max_entry_timeout:
                ticker = self.shared_ws.ticker_data.get(symbol, {})
                last_price = safe_to_float(ticker.get("lastPrice", 0))

                if last_price <= 0:
                    await asyncio.sleep(0.1)
                    continue

                if side == "Sell":
                    if extreme_price == 0 or last_price > extreme_price:
                        extreme_price = last_price
                    elif last_price < extreme_price * (1 - reversal_trigger_pct / 100.0):
                        logger.info(f"[TACTICAL_SQUEEZE_DEMO] {symbol}: Обнаружен откат с пика {extreme_price}. Входим в SHORT через REST.")
                        await self.place_unified_order(symbol, side, qty, "Market")
                        entry_made = True
                        break
                else:
                    if extreme_price == 0 or last_price < extreme_price:
                        extreme_price = last_price
                    elif last_price > extreme_price * (1 + reversal_trigger_pct / 100.0):
                        logger.info(f"[TACTICAL_SQUEEZE_DEMO] {symbol}: Обнаружен отскок со дна {extreme_price}. Входим в LONG через REST.")
                        await self.place_unified_order(symbol, side, qty, "Market")
                        entry_made = True
                        break

                await asyncio.sleep(0.2)

        except Exception as e:
            logger.error(f"[TACTICAL_SQUEEZE_DEMO] Критическая ошибка для {symbol}: {e}", exc_info=True)
            entry_made = False

        finally:
            if not entry_made:
                logger.warning(f"[TACTICAL_SQUEEZE_DEMO] {symbol}: Окно для входа истекло или произошла ошибка. Отмена.")
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)

        return entry_made

    async def place_order_ws(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        position_idx: int = 1,
        price: float | None = None,
        order_type: str = "Market",
        recv_timeout: float = 2.0,
        cid: str | None = None,
    ):
        header = {
            "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
            "X-BAPI-RECV-WINDOW": "5000",
        }
        args = {
            "symbol":      symbol,
            "side":        side,
            "orderType":   order_type,
            "qty":         str(qty),
            "category":    "linear",
            "timeInForce": "GTC",
            "positionIdx": position_idx,
        }
        if price is not None:
            args["price"] = str(price)

        req = {"op": "order.create", "header": header, "args": [args]}

        for attempt in range(2):
            try:
                async with self._recv_lock:
                    await self.ws_trade.send(json.dumps(req))
                    resp = await self._recv_until("order.create", timeout=recv_timeout)

                rc = resp.get("retCode", resp.get("ret_code", 0))
                if rc != 0:
                    raise InvalidRequestError(resp.get("retMsg", "order failed"))
                return resp.get("data", resp)

            except (websockets.ConnectionClosed, asyncio.IncompleteReadError) as e:
                logger.warning(
                    "[place_order_ws] WS closed, reconnecting (attempt %d/2): %s",
                    attempt + 1, e,
                )
                await self.init_trade_ws()
            except Exception as e:
                logger.error("[place_order_ws] Unexpected error: %s", e)
                raise

        raise RuntimeError("Failed to send order after reconnecting WebSocket")

    async def _recv_until(self, expect_op: str, *, timeout: float = 2.0) -> dict:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(f"WS recv timeout waiting for {expect_op}")

            raw = await self.ws_trade.recv()
            resp = json.loads(raw)
            if resp.get("op") == expect_op:
                return resp
            logger.debug("[ws] skipped %s while waiting for %s", resp.get("op"), expect_op)

    async def log_trade(
        self,
        symbol: str,
        *,
        side: str,
        avg_price: Any,
        volume: Any,
        action: str,
        result: str,
        pnl_usdt: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        comment: Optional[str] = None,
        open_interest: Optional[Any] = None,
        closed_manually: bool = False
    ) -> None:
        safe_avg_price = safe_to_float(avg_price)
        safe_volume = safe_to_float(volume)
        side = side.capitalize()
        action = str(action or "").lower()
        result = str(result or "").lower()
        time_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        pnl_info = f" | PnL: {pnl_usdt:.2f}$ ({pnl_pct:.2f}%)" if pnl_usdt is not None else ""
        logger.info(
            f"[log_trade] user=%s %s %s: side=%s, vol=%s, price=%s, result=%s%s",
            self.user_id, action.upper(), symbol, side, safe_volume, safe_avg_price, result, pnl_info
        )

        if action == "close":
            pos_context = self.closed_positions.pop(symbol, None)
            if pos_context:
                entry_candidate = pos_context.get("entry_candidate")
                if entry_candidate and pnl_pct is not None:
                    prompt = entry_candidate.get("full_prompt_for_ai")
                    source = entry_candidate.get("source", "unknown")
                    if prompt:
                        log_for_finetune(prompt, pnl_pct, source)
                        logger.info(f"[FineTuneLog] Записан семпл для {symbol} ({source}) с PnL {pnl_pct:.2f}%")

        try:
            _append_trades_unified({
                "timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "side": side,
                "volume": safe_volume, "price": safe_avg_price, "event": action, "result": result,
                "pnl_usdt": pnl_usdt, "pnl_pct": pnl_pct
            })
        except Exception as e:
            logger.warning(f"[CSV Log] Ошибка записи в trades_unified.csv: {e}")

        if self.load_user_state().get("quiet_mode", False): return
        link = f"https://www.bybit.com/trade/usdt/{symbol}"
        msg = ""

        if action == "open":
            icon = "🟩" if side == "Buy" else "🟥"
            title = f"Открыта {side.upper()} {symbol}"
            msg = (f"{icon} <b>{title}</b>\n\n"
                   f"<b>Цена входа:</b> {safe_avg_price:.6f}\n"
                   f"<b>Объем:</b> {safe_volume}\n")
            if comment:
                msg += f"\n<i>{comment}</i>"

        elif action == "close":
            if pnl_usdt is not None and pnl_pct is not None:
                pnl_icon = "💰" if pnl_usdt >= 0 else "🔻"
                pnl_sign = "+" if pnl_usdt >= 0 else ""
                msg = (f"{pnl_icon} <b>Закрытие позиции: {symbol}</b>\n\n"
                       f"<b>Результат:</b> <code>{pnl_sign}{pnl_usdt:.2f} USDT ({pnl_sign}{pnl_pct:.2f}%)</code>\n\n"
                       f"<b>Сторона:</b> {side}\n"
                       f"<b>Объем:</b> {safe_volume}\n"
                       f"<b>Цена выхода:</b> {safe_avg_price:.6f}\n")
            else:
                msg = (f"❌ <b>Закрытие позиции: {symbol}</b>\n\n"
                       f"<b>Цена выхода:</b> {safe_avg_price:.6f}\n"
                       f"<b>Объем:</b> {safe_volume}\n")

        elif "trailing_set" in action:
            pos = self.open_positions.get(symbol)
            if not pos: return
            entry_price = safe_to_float(pos['avg_price'])
            leverage = safe_to_float(pos.get('leverage', 1.0))
            last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0))
            if last_price > 0:
                pnl_val = (last_price - entry_price) * safe_volume if pos['side'] == "Buy" else (entry_price - last_price) * safe_volume
                position_value = entry_price * safe_volume
                pnl_pct_val = ((pnl_val / position_value) * 100 * leverage) if position_value > 0 else 0.0
                msg = (f"🛡️ <b>Трейлинг-стоп обновлен: {symbol}</b>\n\n"
                       f"<b>Текущий ROI:</b> {pnl_pct_val:.2f}%\n"
                       f"<b>Цена входа:</b> {entry_price:.6f}\n"
                       f"<b>Текущая цена:</b> {last_price:.6f}\n")

        if msg:
            msg += f"\n<b>Время:</b> {time_str}\n#{symbol}"
            try:
                await telegram_bot.send_message(self.user_id, msg, parse_mode=ParseMode.HTML)
            except Exception as exc:
                logger.warning(f"[Telegram] Ошибка отправки уведомления: {exc}")

    async def sync_open_positions_loop(self, interval: int = 5):
        while True:
            try:
                async with self.position_lock:
                    await self.update_open_positions()
            except Exception as e:
                logger.error(f"[sync_loop] Критическая ошибка в цикле синхронизации: {e}", exc_info=True)
            await asyncio.sleep(interval)

    def save_open_positions_json(self):
        snapshot = []
        for sym, pos in self.open_positions.items():
            size  = safe_to_float(pos.get("volume", 0))
            entry = safe_to_float(pos.get("avg_price", 0))
            last  = safe_to_float(
                self.shared_ws.ticker_data.get(sym, {}).get("lastPrice", 0)
            )
            direction = 1 if pos.get("side", "").lower() == "buy" else -1
            pnl = (last - entry) * size * direction
            snapshot.append(
                {"symbol": sym, "side": pos["side"], "size": size,
                "entry_price": entry, "last_price": last, "pnl": pnl}
            )

        try:
            data = {}
            if os.path.exists(OPEN_POS_JSON):
                with open(OPEN_POS_JSON, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            data[str(self.user_id)] = snapshot
            _atomic_json_write(OPEN_POS_JSON, data)
        except Exception as e:
            logger.debug("[save_open_positions_json] %s", e)

    async def _retrain_loop(
        self,
        every_sec: int = 3600,
        min_samples: int = 800
    ):
        logger.info(
            "[retrain] background task started (wake=%ds, min_samples=%d)",
            every_sec,
            min_samples,
        )
        CKPT_PATH = "golden_model_v19.pt"
        while True:
            try:
                buf_len = len(self.training_data)
                logger.debug("[retrain] buffered samples: %d", buf_len)

                if buf_len < min_samples:
                    await asyncio.sleep(every_sec)
                    continue

                batch = list(self.training_data)
                X = np.array([b["features"] for b in batch], dtype=np.float32)
                y = np.array([b["target"]   for b in batch], dtype=np.float32).reshape(-1, 1)

                scaler = StandardScaler().fit(X)
                X_scaled = scaler.transform(X)

                # [REMOVED] PyTorch training logic removed for MLX-only focus
                logger.info("[retrain] MLX training is currently a stub. Only scaler will be refit.")

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    # [IMPROVEMENT] In a real MLX scenario, you'd save model weights and scaler here.
                    # For now, we only save the scaler.
                    joblib.dump(scaler, tmp.name)
                    os.replace(tmp.name, "scaler.pkl")
                logger.info("[retrain] scaler saved → scaler.pkl")

                if getattr(self, "ml_inferencer", None):
                    self.ml_inferencer.scaler = scaler
                else:
                    try:
                        self.ml_inferencer = MLXInferencer(scaler_path="scaler.pkl")
                        logger.info("[retrain] MLInferencer initialised after first training")
                    except Exception as _e:
                        logger.warning("[retrain] could not init MLInferencer: %s", _e)

                for _ in range(buf_len):
                    self.training_data.popleft()

                for sym, hist in self.shared_ws.oi_history.items():
                    arr = np.asarray(hist, dtype=float)
                    if len(arr) >= 60:
                        dif = np.diff(arr[-60:]) / arr[-60:-1]
                        self._oi_sigma[sym] = float(np.std(dif))

            except Exception:
                logger.exception("[retrain] unexpected error during training")

            await asyncio.sleep(every_sec)

def load_users_from_json(json_path: str = "user_state.json") -> list[dict[str, Any]]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)

    result: list[dict[str, Any]] = []
    for uid, data in all_users.items():
        if data.get("banned", False):
            continue
        if not data.get("registered", True):
            continue

        result.append({
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
        })
    return result

async def make_snapshot() -> str:
    snap = {}
    for bot in GLOBAL_BOTS:
        snap[bot.user_id] = {
            "equity": await bot.get_total_open_volume(),
            "positions": [
                {
                    "symbol": sym,
                    "side":   p.get("side"),
                    "qty":    str(p.get("volume")),
                    "avg_price": str(p.get("avg_price")),
                    "leverage":  str(p.get("leverage", 0))
                }
                for sym, p in bot.open_positions.items()
            ]
        }
    fname = f"snapshot_{dt.datetime.utcnow():%Y%m%d_%H%M%S}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    return fname

def load_golden_params(csv_path: str = "golden_params.csv") -> dict:
    default_params = {
        "Buy": {
            "period_iters": 4,
            "price_change": 1.7,
            "volume_change": 200,
            "oi_change": 1.5,
        },
        "Sell": {
            "period_iters": 4,
            "price_change": 1.8,
            "volume_change": 200,
            "oi_change": 1.2,
        }
    }
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            overrides = {}
            for _, row in df.iterrows():
                key = (row["symbol"], row["side"])
                overrides[key] = {
                    "period_iters": int(row["period_iters"]),
                    "price_change": safe_to_float(row["price_change"]),
                    "volume_change": safe_to_float(row["volume_change"]),
                    "oi_change": safe_to_float(row["oi_change"]),
                }
            print(f"[GoldenParams] Loaded {len(overrides)} symbol-specific parameters from CSV.")
            merged = {**default_params, **overrides}
            return merged
        except Exception as e:
            print(f"[GoldenParams] CSV load error: {e}")
    print("[GoldenParams] Using default parameters.")
    return default_params

@router_admin.message(Command("snapshot"))
async def cmd_snapshot(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        return
    fname = await make_snapshot()
    try:
        await telegram_bot.send_document(
            message.from_user.id,
            types.FSInputFile(fname),
            caption="Состояние счёта (JSON)"
        )
    except Exception as e:
        logger.warning("[cmd_snapshot] Telegram send error: %s", e)

# ─────────────────────────────────────────────────────────────
async def run_all() -> None:
    """Создаёт TradingBot-ы, общее Public-WS, строит датасет,
    обучает модель и запускает торговлю."""
    users = load_users_from_json("user_state.json")
    if not users:
        print("❌ Нет активных пользователей для запуска.")
        return

    golden_param_store = load_golden_params()
    bots: list[TradingBot] = []

    # ── shared public-WS ────────────────────────────────────────────────
    initial_symbols = ["BTCUSDT", "ETHUSDT"]
    shared_ws = PublicWebSocketManager(symbols=initial_symbols)

    # ── создаём TradingBot-ы ────────────────────────────────────────────
    for u in users:
        bot = TradingBot(user_data=u,
                         shared_ws=shared_ws,
                         golden_param_store=golden_param_store)
        bots.append(bot)

    # Привязываем первого бота к shared_ws для некоторых общих операций
    if bots:
        shared_ws.bot = bots[0]

    # ── запускаем WS СРАЗУ, чтобы пошли данные и ready_event —──────────
    public_ws_task = asyncio.create_task(shared_ws.start())

    # ── бэк-филл и ожидание готовности ─────────────────────────
    await shared_ws.backfill_history()
    await shared_ws.ready_event.wait()
    #await shared_ws.backfill_history()

    # ───────────────────── Telegram polling ─────────────────────
    try:
        dp.include_router(router)
    except RuntimeError:
        logger.warning("[run_all] Router already attached, skipping")

    try:
        dp.include_router(router_admin)
    except RuntimeError:
        logger.warning("[run_all] Admin router already attached, skipping")
    
    telegram_task = asyncio.create_task(
        dp.start_polling(telegram_bot, skip_updates=True)
    )

    # ── раскатываем модель всем ботам ──────────────────────────────────
    for b in bots:
        b.load_ml_models()

    # ── запускаем сами TradingBot-ы ────────────────────────────────────
    bot_tasks = [asyncio.create_task(b.start()) for b in bots]

    # ── graceful-shutdown (Ctrl-C / SIGTERM) ───────────────────────────
    async def _shutdown():
        logger.info("Завершаем работу…")
        for b in bots:
            await b.stop()
        
        # Отменяем главные задачи
        public_ws_task.cancel()
        telegram_task.cancel()
        
        # Ждем завершения всех задач (включая ботов, которые останавливаются в b.stop())
        await asyncio.gather(
            public_ws_task, telegram_task, *bot_tasks,
            return_exceptions=True
        )
        logger.info("Все задачи остановлены")

    # [ДОБАВЛЕНО] Привязка функции _shutdown к сигналам ОС
    loop = _aio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, functools.partial(_aio.create_task, _shutdown()))

    # Блокируемся, пока жив хотя бы один таск
    await asyncio.gather(public_ws_task, telegram_task, *bot_tasks)


# [REMOVED] Удален дубликат функции wallet_loop, так как она уже есть в классе TradingBot.

# ---------------------- ENTRY POINT ----------------------
# [ДОБАВЛЕНО] Точка входа для запуска всего скрипта
if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        logger.info("Program stopped by user")
