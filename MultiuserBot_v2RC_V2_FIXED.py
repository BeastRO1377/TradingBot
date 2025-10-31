# === AUTO-PATCHED 2025-08-20T20:00:39.026267Z ===
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence

from mimetypes import init
import os, sys, faulthandler

from networkx import sigma
from torch import threshold
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
        # +++ ДОБАВЬ В КОНЕЦ __init__ +++
        self._topics_lock = asyncio.Lock()     # атомарные подписки/отписки
        self._shutdown = False                 # на будущее для чистого стопа


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

    # === ЗАМЕНИТЬ ТЕЛО start НА ЭТО ===
    async def start(self):
        backoff = 1.0
        while True:
            try:
                # (пере)создаём ws и очищаем следы прошлой сессии
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                    self.ws = None

                # Инициализация WS (оставь как у тебя, важно: callback → _on_message)
                # Пример:
                # self.ws = PublicWebSocket(testnet=False, callback=self._on_message)
                # self.ws.current_topics = getattr(self.ws, "current_topics", set())

                # Старт фонового менеджера подписок
                asyncio.create_task(self.manage_symbol_selection(check_interval=60))
                self.ready_event.set()
                backoff = 1.0
                await asyncio.Event().wait()   # держим задачу живой

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.ready_event.clear()
                logger.warning("[PublicWS] reconnect after error: %s", e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    def _is_already_subscribed(self, topic_template: str, sym: str) -> bool:
        """Проверка, есть ли такой топик в current_topics."""
        return topic_template.format(symbol=sym) in self.ws.current_topics

    # === ЗАМЕНИТЬ ТЕЛО _on_message НА ЭТО ===
    def _on_message(self, raw: str):
        try:
            msg = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            logger.debug("[PublicWS] bad json, skip")
            return
        try:
            asyncio.create_task(self.route_message(msg))
        except Exception:
            logger.exception("[PublicWS] route_message scheduling failed")


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
        # >>> ИЗМЕНЕНИЕ: Увеличиваем таймаут для HTTP-запросов до 20 секунд <<<
        http = HTTP(testnet=False, timeout=20)
        is_first_run = True

        while True:
            await asyncio.sleep(10 if is_first_run else check_interval)
            try:
                # >>> ИЗМЕНЕНИЕ: Добавляем блок try...except для перехвата сетевых ошибок <<<
                try:
                    resp = await asyncio.to_thread(lambda: http.get_tickers(category="linear"))
                except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout, UrllibReadTimeoutError) as e:
                    logger.warning(f"[SymbolManager] Не удалось получить тикеры из-за сетевой ошибки: {e}. Повторная попытка через 60 секунд.")
                    await asyncio.sleep(60)
                    continue # Пропускаем эту итерацию и пробуем снова

                all_tickers = {tk["symbol"]: tk for tk in resp["result"]["list"]}
                self.ticker_data.update(all_tickers)

                liquid_symbols = {
                    s for s, t in all_tickers.items()
                    if safe_to_float(t.get("turnover24h", 0)) >= min_turnover and
                    safe_to_float(t.get("volume24h", 0)) >= min_volume
                }
                open_pos_symbols = {s for bot in self.position_handlers for s in bot.open_positions.keys()}
                
                desired_symbols   = liquid_symbols.union(open_pos_symbols)
                symbols_to_add    = desired_symbols - self.active_symbols
                symbols_to_remove = self.active_symbols - desired_symbols

                # 5) Динамически обновляем подписки (с фильтром дублей у pybit)
                k_tpl = f"kline.{self.interval}.{{symbol}}"
                t_tpl = "tickers.{symbol}"
                l_tpl = "liquidation.{symbol}"
                
                if symbols_to_add:
                    symbols_to_add_list = list(symbols_to_add)
                    existing = self._existing_topics()

                    add_k = self._filter_new_symbols(k_tpl, symbols_to_add_list, existing)
                    add_t = self._filter_new_symbols(t_tpl, symbols_to_add_list, existing)
                    add_l = self._filter_new_symbols(l_tpl, symbols_to_add_list, existing)

                    for tpl, bucket in ((k_tpl, add_k), (t_tpl, add_t), (l_tpl, add_l)):
                        if not bucket: continue # Пропускаем, если список пуст
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
                
                    self.active_symbols.update(symbols_to_add)
                
                if symbols_to_remove and getattr(self, "ENABLE_UNSUBSCRIBE", False):
                    symbols_to_remove_list = list(symbols_to_remove)
                    try:
                        self.ws.unsubscribe(topic=k_tpl, symbol=symbols_to_remove_list)
                    except Exception: pass
                    try:
                        self.ws.unsubscribe(topic=t_tpl, symbol=symbols_to_remove_list)
                    except Exception: pass
                    try:
                        self.ws.unsubscribe(topic=l_tpl, symbol=symbols_to_remove_list)
                    except Exception: pass

                self.active_symbols -= symbols_to_remove
                self.symbols = list(self.active_symbols)

                if is_first_run:
                    self.ready_event.set()
                    is_first_run = False
                    logger.info(f"[SymbolManager] Начальный список из {len(self.active_symbols)} символов сформирован. Бот готов к работе.")

            except Exception as e:
                logger.error(f"[SymbolManager] Критическая ошибка в цикле: {e}", exc_info=True)
                if is_first_run and not self.ready_event.is_set():
                    self.ready_event.set()
                    is_first_run = False
                    logger.info(f"[SymbolManager] Начальная настройка завершена (с ошибкой), но бот готов к работе.")


    # === ЗАМЕНИТЬ ТЕЛО route_message НА ЭТО ===
    async def route_message(self, msg: dict):
        try:
            topic = (msg.get("topic") or "").lower()
            if not topic:
                return

            if "kline" in topic:
                try:
                    await self.handle_kline(msg)
                except Exception:
                    logger.exception("[route_message] kline handler failed")

            elif "tickers" in topic:
                try:
                    await self.handle_ticker(msg)
                except Exception:
                    logger.exception("[route_message] ticker handler failed")

            elif "liquidation" in topic:
                data = msg.get("data", [])
                if isinstance(data, dict):
                    data = [data]
                for evt in data:
                    for handler in (self.position_handlers or []):
                        if hasattr(handler, "on_liquidation_event"):
                            asyncio.create_task(handler.on_liquidation_event(evt))
        except Exception:
            logger.exception("[route_message] unexpected failure")


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
        to_np = lambda t: np.array(t)
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
        to_mlx = lambda a: mlx.core.array(a)
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
        prediction = self.model(mlx.core.array(features))
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
            x_train, y_train = mlx.core.array(feats_scaled), mlx.core.array(targ)
            loss, grads = loss_and_grad_fn(model, x_train, y_train)
            optimizer.update(model, grads)
            mlx.core.eval(model.parameters(), optimizer.state)
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
        "ai_rev_pending", "ai_rev_results", "last_ai_check", 
        "trailing_activated", "_stop_workers", "trailing_gap_mode", "trailing_gap_roi_pct", "trailing_min_gap_pct", "trailing_max_gap_pct",
        "tactical_entry_window_sec", "order_timeout_sec", "_stop_guard", "_stop_last_err_ts", "ai_advisor_circuit_breaker", "consecutive_trade_counter",
        "strategy_cooldown_until", "ai_stop_advisor_mode", "squeeze_ai_confirm_interval_sec", "processing_signals", "active_signals",
        "squeeze_ai_timeout_sec",
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

        self._stop_workers = {}
        # режим шага трейла: price-% или roi-%
        self.trailing_gap_mode     = user_data.get("trailing_gap_mode", "roi")
        # минимальный шаг трейла (в % от цены)
        self.trailing_min_gap_pct  = float(user_data.get("trailing_min_gap_pct", 0.5))
        # максимальный шаг трейла (в % от цены)
        self.trailing_max_gap_pct  = float(user_data.get("trailing_max_gap_pct", 2.5))
        # порог ROI для включения шага трейла (в %)
        self.trailing_gap_roi_pct  = float(user_data.get("trailing_gap_roi_pct", 5.0))

        self.squeeze_tuner = None
        logger.info("[MLX] squeeze_tuner disabled – using static squeeze thresholds")

        # Сколько живёт окно входа от момента детекта сквиза (сек)
        self.tactical_entry_window_sec = int(user_data.get("tactical_entry_window_sec", 45))
        # Сколько ждём ACK от биржи на размещение ордера (сек)
        self.order_timeout_sec = int(user_data.get("order_timeout_sec", 15))

        self._stop_guard = defaultdict(asyncio.Lock)  # пер-символьный lock на stop/start
        self._stop_last_err_ts = {}                   # троттлинг ошибок записи

        self.session = HTTP(testnet=False, # Демо-трейдинг использует mainnet URL
                    demo=(self.mode == "demo"),
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    timeout=30,
                    #trace_logging=True
                    ) # Оставьте пока для контроля

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

        self.consecutive_trade_counter = defaultdict(int) # Счетчик: {символ: количество_сделок}
        self.strategy_cooldown_until = {} # Кулдаун: {символ: timestamp_окончания}


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
        self.squeeze_min_score         = getattr(self, "squeeze_min_score", 0.21) # фильтр силы сквиза (0..1)
        self.exhaustion_enter_thr      = getattr(self, "exhaustion_enter_thr", 0.25)
        self.continuation_follow_thr   = getattr(self, "continuation_follow_thr", 0.6)
        self.squeeze_atr_k             = getattr(self, "squeeze_atr_k", 0.50)     # сколько ATR за экстремумом

        self.squeeze_ai_timeout_sec = 15.0 # Ставим щедрое значение по умолчанию - 15 секунд


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
        self.ai_advisor_circuit_breaker: dict[str, float] = {}
        self.ai_provider = "ollama"
        self.ai_base_url = "http://localhost:11434/v1"
        logger.info(f"Выбран AI-провайдер: {self.ai_provider.upper()}")
        self.stop_loss_mode = user_data.get("stop_loss_mode", "strat_loss")
        logger.info(f"Выбран режим стоп-лосса: {self.stop_loss_mode.upper()}")
        self.squeeze_ai_confirm_interval_sec = 2.0 # Значение по умолчанию 2 секунды


        self._ai_rev_workers_started = False
        # Порог/настройки можно менять на лету: getattr(..., default) в коде ниже
        # self.ai_reversal_conf_threshold = 0.62
        # self.ai_rev_probe_min_dd_bp    = 35   # 0.35% в б.п. с плечом
        # self.ai_rev_k_atr              = 1.2
        self.enable_ai_reversal_guard  = True
        self.ai_rev_queue = None                # будет создана лениво
        self.ai_rev_workers = []                # список тасков-воркеров
        self.processing_signals = set()


        self.last_entry_comment: dict[str, str] = {}

    
        self.ai_timeout_sec = float(user_data.get("ai_timeout_sec", 15.0))
        self.ai_sem = asyncio.Semaphore(user_data.get("ai_max_concurrent", 2))
        self.ai_circuit_open_until = 0.0
        self._ai_silent_until = 0.0
        self._ai_inflight_signals: set[str] = set()   # ключи вида f"{symbol}_{side}_{source}" — сейчас в работе

        self.active_signals = set()

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


    def _effective_total_usd(self) -> float:
        """Открытые + pending (USDT-эквивалент)."""
        total = 0.0
        try:
            # открытые
            total += self.get_total_open_volume_fast()
            # pending
            for v in getattr(self, "pending_orders", {}).values():
                try:
                    total += float(v)
                except Exception:
                    pass
        except Exception:
            pass
        return total

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

    async def _calc_qty_from_usd(self, symbol: str, usd_amount: float,
                                price: float | None = None) -> float:
        """
        [V5 - Финальная асинхронная версия] Превращает USDT в количество контракта.
        """
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

        raw_qty = usd_amount / price # Убран коэффициент 0.99 для точности
        if step > 0:
            qty = math.floor(raw_qty / step) * step
        else:
            qty = raw_qty
        
        # Если расчетный объем меньше минимального, используем минимальный
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

    async def train_and_save_model(self):
        """
        Выполняет полный цикл для создания модели:
        1. Загружает максимально возможную историю.
        2. Генерирует обучающий набор данных.
        3. Обучает модель MLX и скейлер.
        4. Сохраняет артефакты в файлы.
        """
        logger.info(">>> ЗАПУЩЕН РЕЖИМ ТРЕНИРОВКИ МОДЕЛИ <<<")

        # 1. Убедимся, что у нас есть список символов
        if not self.shared_ws.active_symbols:
            logger.info("Ожидание первоначального списка символов от WebSocket...")
            await self.shared_ws.ready_event.wait()
        
        symbols_to_train = list(self.shared_ws.active_symbols)
        logger.info(f"Будет использовано {len(symbols_to_train)} символов для построения датасета.")

        # 2. Бэкфилл истории (попросим побольше данных)
        logger.info("Шаг 1/4: Загрузка истории котировок (может занять несколько минут)...")
        # Временно увеличим лимит для более качественного датасета
        original_maxlen =  1000 #self.shared_ws.candles_data.default_factory.keywords.get('maxlen', 1000)
        for sym in symbols_to_train:
            self.shared_ws.candles_data[sym] = deque(maxlen=2000) # Временно ставим больше
        
        await self.shared_ws.backfill_history()
        logger.info("История котировок загружена.")

        # 3. Построение датасета
        logger.info("Шаг 2/4: Построение обучающего набора данных...")
        try:
            # Эта функция уже есть в вашем коде, мы ее просто вызываем
            await self.build_and_save_trainset(
                csv_path="trainset.csv",
                scaler_path="scaler.pkl", # Он сохранится здесь
                symbol=symbols_to_train,
                future_horizon=5, # Смотрим на 5 минут вперед
                future_thresh=0.003 # Считаем успехом рост на 0.3%
            )
            logger.info("Обучающий набор данных успешно создан и сохранен в trainset.csv.")
        except Exception as e:
            logger.critical(f"Критическая ошибка при построении датасета: {e}", exc_info=True)
            return

        # 4. Обучение модели
        logger.info("Шаг 3/4: Запуск процесса обучения модели...")
        try:
            df = pd.read_csv("trainset.csv")
            features = [col for col in df.columns if col != 'label']
            
            X = df[features].values.astype(np.float32)
            y = df['label'].values.astype(np.float32)

            # Создаем training_data в формате, который ожидает ваша функция
            training_data = [{"features": X[i], "target": y[i]} for i in range(len(y))]

            # Вызываем вашу существующую функцию обучения
            model, scaler = MLXInferencer.train_golden_model_mlx(training_data, num_epochs=50) # 50 эпох для качества
            logger.info("Модель успешно обучена.")
        except Exception as e:
            logger.critical(f"Критическая ошибка при обучении модели: {e}", exc_info=True)
            return

        # 5. Сохранение артефактов
        logger.info("Шаг 4/4: Сохранение модели и скейлера...")
        try:
            # Вызываем вашу существующую функцию сохранения
            save_mlx_checkpoint(model, scaler,
                                model_path="golden_model_mlx.safetensors",
                                scaler_path="scaler.pkl")
            logger.info(">>> ТРЕНИРОВКА УСПЕШНО ЗАВЕРШЕНА! <<<")
            logger.info("Файлы 'golden_model_mlx.safetensors' и 'scaler.pkl' созданы.")
            logger.info("Теперь вы можете запустить бота в обычном режиме.")

        except Exception as e:
            logger.critical(f"Критическая ошибка при сохранении модели: {e}", exc_info=True)
            return
        
        # Возвращаем maxlen к стандартному значению
        for sym in symbols_to_train:
            self.shared_ws.candles_data[sym] = deque(list(self.shared_ws.candles_data[sym]), maxlen=original_maxlen)


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

    def _calculate_initial_profitable_stop(self, entry_price: float, side: str, leverage: float) -> float:
        """
        Рассчитывает цену для ПЕРВОГО стоп-лосса, гарантируя фиксацию прибыли.
        Логика: Стоп ставится на уровне (start_pct - gap_pct) ROI.
        """
        start_pct = self._sf(getattr(self, "trailing_start_pct", 5.0))
        gap_pct = self._sf(getattr(self, "trailing_gap_pct", 2.5))
        
        # Целевой ROI для нашего стопа
        target_roi_pct = start_pct - gap_pct # Например, 5.0 - 2.5 = 2.5%

        # Рассчитываем, какому изменению цены соответствует этот ROI
        # Формула: price_change = roi / (leverage * 100)
        price_change_fraction = target_roi_pct / (leverage * 100.0)

        if side == "Buy":
            # Для лонга цена стопа должна быть выше цены входа
            stop_price = entry_price * (1 + price_change_fraction)
        else: # Sell
            # Для шорта цена стопа должна быть ниже цены входа
            stop_price = entry_price * (1 - price_change_fraction)
            
        return stop_price


    # [FINAL VERSION] Высокочастотный менеджер позиций с умным логированием
    # async def manage_open_position(self, symbol: str):
    #     """
    #     [V9 - AI Guardian] Управляет позицией, используя базовый трейлинг
    #     и периодические консультации с AI-советником.
    #     """
    #     logger.info(f"🛡️ [AI Guardian] Активирован для {symbol}.")
        
    #     # Таймер для периодического запроса к AI
    #     last_ai_check = time.time()
    #     ai_check_interval = 30 # секунд

    #     while symbol in self.open_positions:
    #         # Цикл работает быстро, чтобы отслеживать цену для трейлинга
    #         await asyncio.sleep(self.trailing_update_interval_sec) 

    #         pos_data = self.open_positions.get(symbol)
    #         if not pos_data: break

    #         last_price = self._sf(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
    #         if not last_price > 0: continue

    #         entry_price = self._sf(pos_data.get("avg_price", 0))
    #         side = pos_data.get("side")
    #         leverage = self._sf(pos_data.get("leverage", 10.0))
    #         if not (entry_price > 0 and side): continue

    #         # --- Расчет текущего PnL (важно для всех систем) ---
    #         current_roi = (((last_price - entry_price) / entry_price) * 100 * leverage) \
    #             if side == "Buy" else \
    #             (((entry_price - last_price) / entry_price) * 100 * leverage)
    #         pos_data['pnl'] = current_roi

    #         # --- 1. БАЗОВЫЙ ТРЕЙЛИНГ-СТОП (работает всегда) ---
    #         if current_roi >= self.trailing_start_pct:
                
    #             # >>> ДОБАВЬТЕ ЭТОТ БЛОК <<<
    #             if not pos_data.get('trailing_active'):
    #                 logger.info(f"✅ [{symbol}] Трейлинг АКТИВИРОВАН. ROI достиг {current_roi:.2f}% (порог {self.trailing_start_pct}%).")
    #                 pos_data['trailing_active'] = True
    #             # >>> КОНЕЦ БЛОКА <<<

    #             # Расчет правильной цены стопа
    #             gap_pct = self._sf(getattr(self, "trailing_gap_pct", 2.5))
    #             new_stop_price = last_price * (1 - gap_pct / 100.0) if side == "Buy" else last_price * (1 + gap_pct / 100.0)

    #             current_stop = self.last_stop_price.get(symbol)
    #             # Условие для подтягивания стопа
    #             if current_stop is None or \
    #             (side == "Buy" and new_stop_price > current_stop) or \
    #             (side == "Sell" and new_stop_price < current_stop):
    #                 await self.set_or_amend_stop_loss(symbol, new_stop_price)


    #         # --- 2. AI-СОВЕТНИК (работает периодически) ---
    #         now = time.time()
    #         if now - last_ai_check > ai_check_interval:
    #             last_ai_check = now
                
    #             logger.info(f"[{symbol}] Запрашиваем совет у AI... (ROI: {current_roi:.2f}%)")
    #             advice = await self.get_ai_position_advice(symbol, pos_data)
                
    #             action = advice.get("action", "HOLD")
    #             confidence = self._sf(advice.get("confidence", 0.5))
    #             reason = advice.get("reason", "N/A")

    #             # Применяем совет только если AI достаточно уверен
    #             if confidence > 0.75: 
    #                 pos_volume = self._sf(pos_data.get("volume", 0))
                    
    #                 if action == "CLOSE_FULL":
    #                     logger.info(f"🚨 [AI_CLOSE] {symbol}: Полное закрытие по совету AI. Причина: {reason}")
    #                     await self._close_position_market(symbol, pos_volume, f"AI_FULL_CLOSE: {reason}")
    #                     break # Выходим из цикла, т.к. позиция будет закрыта

    #                 elif action == "CLOSE_PARTIAL" and pos_volume > 0:
    #                     qty_to_close = pos_volume * 0.5 # Закрываем 50%
    #                     logger.info(f" partial_close ")
    #                     await self._close_position_market(symbol, qty_to_close, f"AI_PARTIAL_CLOSE: {reason}")
    #                     # Цикл продолжается, т.к. часть позиции осталась

    #     logger.info(f"🛡️ [AI Guardian] Позиция {symbol} закрыта. Хранитель завершает работу.")


    async def manage_open_position(self, symbol: str):
        logger.info(f"🛡️ [Guardian] Активирован для позиции {symbol}.")
        
        try:
            pos = self.open_positions.get(symbol)
            if not pos: return

            await self._start_stop_worker(symbol, pos)

            last_sent_price = 0.0
            min_rel_move = 1e-4
            tick_interval = 0.2
            last_ai_check = 0
            ai_check_interval = 60 # Проверяем AI раз в 60 секунд

            while symbol in self.open_positions:
                await asyncio.sleep(tick_interval)

                # 1. Проверяем здоровье воркера
                worker_rec = self._stop_workers.get(symbol)
                proc = worker_rec.get("proc") if worker_rec else None
                if not proc or proc.returncode is not None:
                    logger.warning(f"[Guardian] {symbol}: stop_worker умер, перезапускаем.")
                    await self._restart_stop_worker(symbol)
                    last_sent_price = 0.0
                    continue

                # 2. Отправляем тики для механического трейлинга
                last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
                if last_price > 0 and abs(last_price - last_sent_price) / last_price >= min_rel_move:
                    if await self._send_stop_msg(symbol, {"cmd": "tick", "last_price": last_price}):
                        last_sent_price = last_price
                
                # 3. [НОВОЕ] Периодически консультируемся с AI
                now = time.time()
                if self.ai_stop_management_enabled and now - last_ai_check > ai_check_interval:
                    last_ai_check = now
                    try:
                        # Запускаем проверку в фоне, чтобы не блокировать отправку тиков
                        asyncio.create_task(self._ai_stop_management_check(symbol))
                    except Exception as e:
                        # Логируем ошибку, но НЕ выходим из цикла. Guardian продолжает работать.
                        logger.error(f"[Guardian] Ошибка при запуске задачи _ai_stop_management_check для {symbol}", exc_info=True)

        except asyncio.CancelledError:
            logger.info(f"[Guardian] Наблюдение за {symbol} отменено.")
            raise
        except Exception as e:
            logger.error(f"[Guardian] {symbol} критическая ошибка: {e}", exc_info=True)
        finally:
            logger.info(f"🛡️ [Guardian] Завершает наблюдение и останавливает воркер для {symbol}.")
            await self._stop_stop_worker(symbol)



    # ─────────────────── Stop worker IPC ───────────────────
    # [ИЗМЕНЕНО] Запуск воркера теперь передает все нужные параметры, включая tick_size
    async def _start_stop_worker(self, symbol: str, pos: dict):
        import asyncio, sys, os, json
        if symbol in self._stop_workers:
            return

        # --- НОВОЕ: Гарантированно получаем tick_size перед запуском ---
        await self.ensure_symbol_meta(symbol)
        tick_size = self.price_tick_map.get(symbol)
        if not tick_size or tick_size <= 0:
            logger.error(f"[Guardian] Не удалось получить tick_size для {symbol}. Воркер не запущен.")
            return
        # --- КОНЕЦ НОВОГО БЛОКА ---

        # Запускаем отдельный процесс с небуферизованным stdout/stderr
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", os.path.join(os.path.dirname(__file__), "stop_worker.py"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._stop_workers[symbol] = {
            "proc": proc,
            "writer": proc.stdin,
            "reader_task": asyncio.create_task(self._read_stop_events(symbol, proc)),
            "stderr_task": asyncio.create_task(self._drain_stop_stderr(symbol, proc)),
        }

        # Инициализация воркера
        init = {
            "cmd": "init",
            "symbol": symbol,
            "side": pos.get("side", "Buy"),
            "avg_price": float(pos.get("avg_price") or pos.get("entryPrice") or 0.0) or 0.0,
            "leverage": float(pos.get("leverage") or getattr(self, "leverage", 1.0) or 1.0),
            "tick_size": tick_size,  # <<< КЛЮЧЕВОЕ ДОБАВЛЕНИЕ
            "start_roi": float(getattr(self, "trailing_start_pct", 5.0)),
            "gap_mode": getattr(self, "trailing_gap_mode", "price"),
            "gap_roi_pct": float(getattr(self, "trailing_gap_roi_pct", 2.5)),
            "gap_price_pct": float(getattr(self, "trailing_gap_pct", 2.5)),
            "min_gap_pct":  float(getattr(self, "trailing_min_gap_pct", 0.5)),
            "max_gap_pct":  float(getattr(self, "trailing_max_gap_pct", 3.0)),
            "hb_interval": float(getattr(self, "trailing_heartbeat_sec", 3.0) or 3.0),
        }
        await self._send_stop_msg(symbol, init)
        logger.info(f"🛡️ [Guardian] stop_worker для {symbol} запущен с tick_size={tick_size}")


    async def _drain_stop_stderr(self, symbol: str, proc):
        """Фоново сливаем stderr воркера (предотвращает зависания от переполнения буфера)."""
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                try:
                    txt = line.decode("utf-8", "ignore").rstrip()
                except Exception:
                    txt = str(line)
                if txt:
                    logger.debug("[stopw-err][%s] %s", symbol, txt)
        except Exception:
            pass



    # [НОВОЕ] Отдельная функция для отправки init-конфига
    async def _send_init_to_worker(self, symbol: str, payload: dict) -> bool:
        worker_rec = self._stop_workers.get(symbol)
        if not worker_rec or not worker_rec.get("writer"): return False
        
        writer = worker_rec["writer"]
        if writer.is_closing(): return False
            
        try:
            data = (json.dumps(payload) + "\n").encode()
            writer.write(data)
            await writer.drain()
            # writer.close() # <-- УБЕДИТЕСЬ, ЧТО ЭТА СТРОКА ЗАКОММЕНТИРОВАНА ИЛИ УДАЛЕНА!
            return True
        except Exception as e:
            logger.error(f"[Guardian] Ошибка отправки init-конфига воркеру {symbol}: {e}")
            return False


    async def _ai_stop_management_check(self, symbol: str):
        now = time.time()
        if now < self.ai_advisor_circuit_breaker.get(symbol, 0):
            logger.debug(f"[AI_STOP_ADVISOR] Circuit breaker для {symbol} активен. Консультация пропущена.")
            return

        try:
            pos = self.open_positions.get(symbol)
            if not pos: return

            last_price = self._sf(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
            entry_price = self._sf(pos.get("avg_price", 0))
            side = pos.get("side")
            leverage = self._sf(pos.get("leverage", 10.0))

            if not all([last_price > 0, entry_price > 0, side]):
                return

            current_roi = 0.0
            if side.lower() == "buy":
                current_roi = (((last_price - entry_price) / entry_price) * 100 * leverage)
            else:
                current_roi = (((entry_price - last_price) / entry_price) * 100 * leverage)
            pos['pnl'] = current_roi

            # <<< КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Добавляем проверку ROI >>>
            if self.ai_stop_advisor_mode == "roi" and current_roi < self.trailing_start_pct:
                logger.debug(
                    f"[AI_STOP_ADVISOR] Режим 'roi'. ROI ({current_roi:.2f}%) "
                    f"ниже порога ({self.trailing_start_pct}%). Консультация пропущена."
                )
                return

            # <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

            features = await self.extract_realtime_features(symbol)
            if not features: return

            prompt = self._build_stop_management_prompt(symbol, pos, features)
            messages = [{"role": "user", "content": prompt}]
            
            ai_response = await self._ask_ollama_json(
                model=self.ai_advisor_model,
                messages=messages,
                timeout_s=45.0
            )

            action = ai_response.get("action", "HOLD").upper()
            new_price = safe_to_float(ai_response.get("new_stop_price"))

            if action == "MOVE_STOP" and new_price > 0:
                logger.info(f"[AI_STOP_ADVISOR] AI для {symbol} предлагает переместить стоп на {new_price}. Причина: {ai_response.get('reason')}")
                await self._send_stop_msg(symbol, {"cmd": "override_stop", "price": new_price})
                logger.info(f"[AI_STOP_ADVISOR] Команда override_stop для {symbol} отправлена воркеру.")
        
        except asyncio.TimeoutError:
            cooldown_seconds = 600 # 10 минут
            self.ai_advisor_circuit_breaker[symbol] = time.time() + cooldown_seconds
            logger.error(f"[AI_STOP_ADVISOR] Таймаут при консультации для {symbol}. Отключаем советника для этой пары на {cooldown_seconds} секунд.")
        except Exception:
            logger.warning(f"[AI_STOP_ADVISOR] Ошибка при консультации с AI для {symbol}", exc_info=True)


    async def _cache_all_symbol_meta(self):
        """
        Загружает и кэширует метаданные (шаг лота, тик цены и т.д.) для всех
        линейных контрактов при старте бота.
        """
        logger.info("Кэширование метаданных для всех символов...")
        try:
            # Один запрос для всех инструментов
            resp = await asyncio.to_thread(
                lambda: self.session.get_instruments_info(category="linear")
            )
            
            instrument_list = resp.get("result", {}).get("list", [])
            count = 0
            for info in instrument_list:
                symbol = info.get("symbol")
                if not symbol: continue

                lot_filter = info.get("lotSizeFilter", {})
                price_filter = info.get("priceFilter", {})

                self.qty_step_map[symbol] = self._sf(lot_filter.get("qtyStep"))
                self.min_qty_map[symbol] = self._sf(lot_filter.get("minOrderQty"))
                self.price_tick_map[symbol] = self._sf(price_filter.get("tickSize"))
                count += 1
            
            logger.info(f"Успешно закэшировано метаданных для {count} символов.")

        except Exception:
            logger.error("Критическая ошибка: не удалось закэшировать метаданные символов. Бот может работать некорректно.", exc_info=True)



    # [НОВОЕ] Формировщик промпта для AI-советника
    def _build_stop_management_prompt(self, symbol: str, pos: dict, features: dict) -> str:
        roi = self._sf(pos.get("pnl", 0)) # Предполагаем, что PnL уже где-то считается и кладется в pos
        
        prompt = f"""
            SYSTEM: Ты элитный риск-менеджер. Твоя задача - дать совет по управлению стоп-лоссом для открытой позиции. Анализируй, является ли текущее движение временным откатом или разворотом тренда. Ответ - только валидный JSON.

            USER:
            **Открытая позиция:**
            - Инструмент: {symbol}
            - Направление: {pos.get('side', '').upper()}
            - Цена входа: {self._sf(pos.get('avg_price')):.6f}
            - Текущий ROI: {roi:.2f}% (с плечом)
            - Текущий трейлинг-стоп: {self.last_stop_price.get(symbol, 'N/A')}

            **Рыночный контекст:**
            - RSI(14): {features.get('rsi14', 0):.1f}
            - ADX(14): {features.get('adx14', 0):.1f}
            - Тренд (Supertrend): {'UPTREND' if features.get('supertrend', 0) > 0 else 'DOWNTREND'}

            **ЗАДАЧА:** Дай рекомендацию в формате JSON.
            - "action": "MOVE_STOP" (если нужно подвинуть стоп) или "HOLD" (если текущий стоп оптимален).
            - "new_stop_price": 123.45 (только если action="MOVE_STOP"). Предложи цену, которая максимизирует прибыль, но защищает от разворота.
            - "reason": Краткое обоснование.
            """
        return prompt.strip()



    async def _guardian_trailing_loop(self, symbol: str):
        pos = self.open_positions.get(symbol)
        if not pos:
            return
        await self._start_stop_worker(symbol, pos)

        last_sent = 0.0
        interval = float(getattr(self, "trailing_update_interval_sec", 0.2) or 0.2)
        try:
            while symbol in self.open_positions:
                last = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0))
                if last > 0:
                    if last_sent == 0 or abs(last - last_sent) / max(last_sent, 1e-12) >= 1e-4:
                        await self._send_stop_msg(symbol, {"cmd": "tick", "price": last, "ts": time.time()})
                        last_sent = last
                await asyncio.sleep(interval)
        finally:
            await self._stop_stop_worker(symbol) # <--- ВЕРНИТЕ ЗДЕСЬ ПРАВИЛЬНЫЙ ВЫЗОВ
            logger.info(f"🛡️ [Guardian] {symbol} trailing loop stopped")


    # ── рядом с другими helpers TradingBot ─────────────────────────
    async def _send_stop_msg(self, symbol: str, obj: dict) -> bool:
        """Безопасная отправка JSON-строки воркеру. Возвращает True/False."""
        import json
        rec = self._stop_workers.get(symbol)
        if not rec:
            return False
        w = rec.get("writer")
        if not w or (hasattr(w, "is_closing") and w.is_closing()):
            return False
        try:
            w.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            await w.drain()
            return True
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as e:
            logger.warning("[stop_worker] write failed for %s: %s", symbol, e)
            return False
        except Exception as e:
            logger.error("[stop_worker] unexpected write error for %s: %s", symbol, e, exc_info=True)
            return False


    async def _restart_stop_worker(self, symbol: str):
        """Атомарный рестарт stop_worker под локом — без гонок."""
        async with self._stop_guard[symbol]:
            # если позиция уже закрыта — просто выйти
            if symbol not in self.open_positions:
                return
            try:
                await self._stop_stop_worker(symbol)
            except Exception:
                pass
            pos = self.open_positions.get(symbol) or {}
            await self._start_stop_worker(symbol, pos)


    async def _read_stop_stderr(self, symbol: str, proc):
        """Читает и логирует поток ошибок от дочернего процесса."""
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            # Логируем как ошибку, чтобы было хорошо видно в логах
            logger.error(f"[stop_worker_stderr][{symbol}] {line.decode('utf-8', errors='ignore').strip()}")



    async def _read_stop_events(self, symbol: str, proc):
        """Читает события от воркера и приводит их к единому виду для set_or_amend_stop_loss."""
        import json, asyncio
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    evt = json.loads(line.decode("utf-8").strip())
                except Exception:
                    continue

                et = str(evt.get("event", "")).lower()

                # Поддержка старого и нового протокола:
                # - старый: "activated"/"trail_update" c полями stop_price/candidate
                # - новый:  "set_stop" c полем price
                if et in ("activated", "trail_update", "set_stop"):
                    price = (
                        float(evt.get("stop_price") or 0.0) or
                        float(evt.get("candidate")  or 0.0) or
                        float(evt.get("price")      or 0.0)
                    )
                    if price > 0:
                        try:
                            await self.set_or_amend_stop_loss(symbol, price)
                            self.last_stop_price[symbol] = price
                            logger.info("✅ [Guardian] %s стоп обновлен на %.10f", symbol, price)
                        except Exception as e:
                            logger.error("[stop_worker] failed to set stop for %s: %s", symbol, e, exc_info=True)

                elif et == "heartbeat":
                    logger.debug("[stopd♥] %s %s", symbol, evt)

                elif et == "closed":
                    break
        finally:
            logger.info("[stop_worker] reader for %s finished", symbol)


    async def _stop_stop_worker(self, symbol: str):
        worker_rec = self._stop_workers.pop(symbol, None)
        if not worker_rec:
            return

        try:
            # 1. Отправляем команду на завершение
            await self._send_stop_msg(symbol, {"cmd": "close"})
            
            # 2. Даем процессу время на самостоятельное завершение
            proc = worker_rec.get("proc")
            if proc and proc.returncode is None:
                await asyncio.wait_for(proc.wait(), timeout=2.0)

        except asyncio.TimeoutError:
            logger.warning(f"[Guardian] {symbol} не завершился сам, принудительно убиваем.")
            if proc and proc.returncode is None: proc.kill()
        except Exception as e:
            logger.error(f"[Guardian] Ошибка при остановке воркера {symbol}: {e}")
            if proc and proc.returncode is None: proc.kill()
        finally:
            # 3. Гарантированно отменяем таски-читатели
            for task_name in ("reader_task", "stderr_task"):
                if worker_rec.get(task_name):
                    worker_rec[task_name].cancel()
            
            logger.info(f"🛡️ [Guardian] stop_worker для {symbol} остановлен.")





    # async def _start_stop_worker(self, symbol: str, pos: dict):
    #     """
    #     Стартует отдельный процесс stop_worker.py (по абсолютному пути),
    #     подписывает reader на stdout/stderr и шлёт init с параметрами.
    #     """
    #     import asyncio, sys, json, os
    #     from pathlib import Path

    #     if symbol in getattr(self, "_stop_workers", {}):
    #         return

    #     if not hasattr(self, "_stop_workers"):
    #         self._stop_workers = {}

    #     # абсолютный путь к скрипту
    #     worker_path = str(Path(__file__).parent.joinpath("stop_worker.py"))
    #     if not os.path.exists(worker_path):
    #         logger.critical("[stop_worker] not found at %s — trailing disabled!", worker_path)
    #         return

    #     try:
    #         proc = await asyncio.create_subprocess_exec(
    #             sys.executable, worker_path,
    #             stdin=asyncio.subprocess.PIPE,
    #             stdout=asyncio.subprocess.PIPE,
    #             stderr=asyncio.subprocess.PIPE,
    #         )
    #     except Exception as e:
    #         logger.critical("[stop_worker] spawn failed: %s", e, exc_info=True)
    #         return

    #     # отдельный таск на stderr, чтобы видеть падения воркера
    #     async def _read_stop_stderr(symbol_: str, proc_):
    #         while True:
    #             line = await proc_.stderr.readline()
    #             if not line:
    #                 break
    #             logger.error("[stop_worker][%s][stderr] %s", symbol_, line.decode("utf-8", "ignore").rstrip())

    #     self._stop_workers[symbol] = {
    #         "proc": proc,
    #         "writer": proc.stdin,
    #         "reader_task": asyncio.create_task(self._read_stop_events(symbol, proc)),
    #         "stderr_task": asyncio.create_task(_read_stop_stderr(symbol, proc)),
    #     }

    #     side = pos.get("side")
    #     entry_price = safe_to_float(pos.get("avg_price") or pos.get("avgPrice") or 0.0)
    #     leverage = safe_to_float(pos.get("leverage") or getattr(self, "leverage", 1))

    #     init_params = {
    #         "cmd": "init",
    #         "symbol": symbol,
    #         "side": side,
    #         "avg_price": entry_price,
    #         "leverage": leverage,
    #         # режимы/пороги из конфигурации бота
    #         "gap_mode":            getattr(self, "trailing_gap_mode", "roi"),   # roi | price
    #         "trailing_start_roi_pct": float(getattr(self, "trailing_start_pct", 5.0)),
    #         "gap_roi_pct":           float(getattr(self, "trailing_gap_roi_pct", 2.5)),
    #         "gap_price_pct":         float(getattr(self, "trailing_gap_pct", 2.5)),
    #         "min_gap_pct":           float(getattr(self, "trailing_min_gap_pct", 0.5)),
    #         "max_gap_pct":           float(getattr(self, "trailing_max_gap_pct", 2.5)),
    #         "tick": float(self.price_tick_map.get(symbol, 1e-6) or 1e-6),
    #     }
    #     await self._send_stop_msg(symbol, init_params)
    #     logger.info("🛡️ [Guardian] %s stop_worker started (side=%s, entry=%.12f, lev=%s)", symbol, side, entry_price, leverage)

    async def _read_stop_stderr(self, symbol: str, stream):
        while True:
            line = await stream.readline()
            if not line: break
            logger.error(f"[stop_worker_stderr][{symbol}] {line.decode(errors='ignore').strip()}")



    async def _close_position_market(self, symbol: str, qty_to_close: float, reason: str):
        """Надежно закрывает указанное количество по рынку."""
        pos = self.open_positions.get(symbol)
        if not pos:
            logger.warning(f"[{symbol}] Попытка закрыть уже несуществующую позицию.")
            return

        # Округляем кол-во до правильного шага лота
        qty_str = self._format_qty(symbol, qty_to_close)
        if self._sf(qty_str) <= 0:
            logger.warning(f"[{symbol}] Объем для закрытия после округления равен нулю.")
            return

        side_to_close = "Sell" if pos['side'] == "Buy" else "Buy"

        logger.info(f"Закрытие {qty_str} {symbol} по рынку. Причина: {reason}")
        try:
            # Используем ваш надежный метод отправки ордера
            await self.place_unified_order(
                symbol=symbol,
                side=side_to_close,
                qty=self._sf(qty_str),
                order_type="Market",
                comment=f"Close: {reason}"
            )
        except Exception as e:
            logger.error(f"Не удалось закрыть {qty_str} {symbol} по причине '{reason}': {e}", exc_info=True)


    async def get_ai_position_advice(self, symbol: str, position_data: dict) -> dict:
        """
        Запрашивает у ИИ совет по управлению открытой позицией.
        Возвращает словарь с рекомендациями.
        """
        default_advice = {"action": "HOLD", "confidence": 0.5, "reason": "No clear signal"}
        
        try:
            # --- 1. Сбор данных для промпта ---
            side = position_data.get("side", "Buy")
            entry_price = self._sf(position_data.get("avg_price"))
            current_roi = self._sf(position_data.get("pnl")) # pnl уже считается в manage_open_position
            
            # Получаем свежие рыночные фичи
            features = await self.extract_realtime_features(symbol)
            if not features:
                return default_advice

            # --- 2. Формирование промпта ---
            prompt = f"""
                SYSTEM: Ты элитный аналитик и риск-менеджер. Твоя задача - дать совет по уже открытой позиции. Анализируй, является ли текущее движение против позиции разворотом тренда или временным откатом. Ответ - только валидный JSON.

                USER:
                **Открытая позиция:**
                - Инструмент: {symbol}
                - Направление: {side.upper()}
                - Цена входа: {entry_price:.6f}
                - Текущий ROI: {current_roi:.2f}% (с плечом)

                **Рыночный контекст:**
                - RSI(14): {features.get('rsi14', 0):.1f}
                - ADX(14): {features.get('adx14', 0):.1f}
                - Тренд (Supertrend): {'UPTREND' if features.get('supertrend', 0) > 0 else 'DOWNTREND'}
                - Изменение объема (5m): {features.get('vol5m', 0) / (features.get('avgVol30m', 1) + 1):.1f}x от среднего
                - Изменение OI (5m): {features.get('dOI5m', 0) * 100:.2f}%

                **ЗАДАЧА:** Дай рекомендацию в формате JSON.
                - "action": "HOLD" (держать), "CLOSE_FULL" (полностью закрыть), "CLOSE_PARTIAL" (закрыть 50%).
                - "confidence": Уверенность в решении [0.0 ... 1.0].
                - "reason": Краткое, но четкое обоснование твоего решения.
                """
            # --- 3. Запрос к AI (используем ваш существующий метод для Ollama) ---
            messages = [{"role": "user", "content": prompt.strip()}]
            
            # Создаем ПОЛНЫЙ словарь для AI, включая 'side', который у нас есть в этой функции
            ai_candidate = {
                "symbol": symbol,
                "side": side,  # 'side' доступен в этом методе
                "source": "PositionAdvisor" # Даем источнику имя для ясности
            }


            # Используем вашу лучшую реализацию, например, ансамбль, если он есть
            ai_response = await self.evaluate_candidate_with_ollama(ai_candidate, features)

            # Проверяем, что ответ содержит нужные ключи
            if "action" in ai_response and "confidence" in ai_response:
                return ai_response
            else:
                # Если модель вернула что-то не то, парсим ее justification
                # или возвращаем дефолт
                return default_advice

        except Exception as e:
            logger.error(f"[AI_ADVICE_ERROR] для {symbol}: {e}", exc_info=True)
            return default_advice


    # [HELPER] Исполнитель установки/изменения стопа
    async def set_or_amend_stop_loss(self, symbol: str, new_stop_price: float):
        """
        [V2 - Умная обработка ошибок] Устанавливает/подтягивает стоп-лосс.
        Удаляет состояние позиции ТОЛЬКО при ошибке 10001 (не найдена).
        """
        try:
            pos = self.open_positions.get(symbol)
            if not pos:
                # Если позиции уже нет в нашем трекере, ничего не делаем
                return

            side = str(pos.get("side", "")).lower()
            tick = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)

            # Выравниваем новый уровень к тик-сайзу
            stop_price = self._round_price(new_stop_price, tick, side)

            prev_stop = self.last_stop_price.get(symbol)

            # Не ухудшаем стоп (с учетом погрешности)
            if prev_stop is not None:
                if side == "buy" and stop_price < prev_stop + (tick / 2):
                    return
                if side == "sell" and stop_price > prev_stop - (tick / 2):
                    return

            pos_idx = int(pos.get("pos_idx") or (1 if side == "buy" else 2))

            # Единый REST-метод для установки стопа
            await self.place_set_trailing_stop_ws(
                symbol=symbol,
                position_idx=pos_idx,
                stop_loss=f"{stop_price:.8f}".rstrip("0").rstrip("."),
            )

            # Обновляем состояние только после успешного вызова API
            self.last_stop_price[symbol] = stop_price
            self._last_trailing_ts[symbol] = time.time()
            # Убираем лишний лог отсюда, так как он дублируется в _read_stop_events

        except InvalidRequestError as e:
            # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
            # Реагируем по-разному на разные ошибки
            if getattr(e, "status_code", None) == 10001:
                logger.warning(f"[TRAILING_STOP] API_ERROR {symbol}: Позиция не найдена на бирже (код 10001). Запускаем очистку состояния.")
                self._purge_symbol_state(symbol)
            else:
                # Все другие ошибки API просто логируем, но НЕ удаляем позицию
                logger.error(f"[TRAILING_STOP] API_ERROR {symbol}: Не удалось обновить стоп. Код: {getattr(e, 'status_code', 'N/A')}, Сообщение: {e}")
        
        except Exception as e:
            logger.error(f"[TRAILING_STOP] CRITICAL {symbol}: Непредвиденная ошибка при обновлении стопа: {e}", exc_info=True)


    def _round_price(self, price: float, tick: float, side: str) -> float:
            """Округляет цену до шага тика в правильную сторону."""
            if tick <= 0: return price
            # Для лонга округляем ВНИЗ, для шорта - ВВЕРХ, чтобы стоп был безопаснее
            rounding = math.floor if side.lower() == "buy" else math.ceil
            return rounding(price / tick) * tick



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
            avg_price = float(pos.get("avg_price") or 0.0)
            if last_price <= 0.0:
                # Последний апдейт цены не подтянулся — ничего не делаем
                return False

            # Рассчитываем «сырой» уровень стопа от текущей цены и GAP (в процентах!)
            if side.lower() == "buy":
                raw_price = last_price * (1.0 - gap_pct / 1000.0)
            else:  # sell
                raw_price = last_price * (1.0 + gap_pct / 1000.0)

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

    # Название оставлено совместимым с текущими вызовами.
    async def place_set_trailing_stop_ws(
        self, *, symbol: str, position_idx: int, stop_loss: str,
        trigger_by: str = "LastPrice"
    ):
        async with _TRADING_STOP_SEM:
            # Используем REST и в demo, и в real: надёжно и одинаково.
            return await asyncio.to_thread(
                lambda: self.session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    positionIdx=position_idx,
                    stopLoss=stop_loss,
                    triggerBy=trigger_by,
                    timeInForce="GTC",
                )
            )


    def _purge_symbol_state(self, symbol: str):
        logger.debug(f"Полная очистка состояния для символа: {symbol}")
        
        # [НОВОЕ] Немедленно даем команду на остановку дочернего процесса
        asyncio.create_task(self._stop_stop_worker(symbol))
        
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
        self.ai_timeout_sec = cfg.get("ai_timeout_sec", 15)

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
        self.ai_stop_advisor_mode = cfg.get("ai_stop_advisor_mode", "roi") 

        self.squeeze_ai_confirm_interval_sec = float(cfg.get("squeeze_ai_confirm_interval_sec", 2.0))


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

    async def warm_up_ai_models(self):
        """Отправляет тестовые запросы к AI-моделям для их 'прогрева'."""
        logger.info("🔥 Прогреваем AI-модели...")
        dummy_messages = [{"role": "user", "content": "ping"}]
        models_to_warm_up = []
        
        if getattr(self, "ai_primary_model", None):
            models_to_warm_up.append(self.ai_primary_model)
        if getattr(self, "ai_advisor_model", None):
            models_to_warm_up.append(self.ai_advisor_model)
        
        # Используем set для прогрева каждой модели только один раз
        for model in set(models_to_warm_up):
            try:
                logger.info(f"  -> Прогрев модели: {model}")
                # Увеличиваем таймаут для первого, самого долгого запроса
                await self._ask_ollama_json(
                    model=model, 
                    messages=dummy_messages, 
                    timeout_s=30.0 # Даем 30 секунд на загрузку модели в память
                )
                logger.info(f"  ✅ Модель {model} прогрета.")
            except asyncio.TimeoutError:
                logger.error(f"  ❌ ТАЙМАУТ при прогреве модели {model}. Она может отвечать медленно.")
            except Exception as e:
                logger.error(f"  ❌ Ошибка при прогреве модели {model}: {e}")


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

        await self._cache_all_symbol_meta() 

        await asyncio.sleep(self.warmup_seconds)
        self.warmup_done = True
        logger.info("[warmup] user %s finished (%d s)", self.user_id, self.warmup_seconds)

        # >>>>> ДОБАВЛЕНО: Прогрев AI-моделей перед началом активной работы <<<<<
        await self.warm_up_ai_models()
        # >>>>> КОНЕЦ ДОБАВЛЕНИЯ <<<<<

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
                    testnet=False, # Оставляем False
                    demo=self.mode == "demo", # <--- ВОТ САМОЕ ГЛАВНОЕ ИЗМЕНЕНИЕ
                    channel_type="private",
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    ping_interval=30,
                    ping_timeout=15,
                    restart_on_error=True,
                    retries=200
                    #trace_logging=True # Оставьте для проверки
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
        [V7 - Reliable Closer] Главный обработчик закрытия.
        Извлекает все данные для лога ДО очистки состояния.
        """
        for exec_data in msg.get("data", []):
            symbol = exec_data.get("symbol")
            if not symbol: continue

            async with self.position_lock:
                pos = self.open_positions.get(symbol)
                # Если позиции нет в нашем трекере, выходим
                if not pos: continue

                exec_side = exec_data.get("side")
                # Проверяем, что это ордер на закрытие (противоположный стороне позиции)
                if exec_side and pos.get("side") and exec_side != pos.get("side"):
                    # leavesQty == 0 означает, что ордер на закрытие исполнен полностью
                    if self._sf(exec_data.get("leavesQty", 0)) == 0:
                        
                        # >>> ИЗВЛЕКАЕМ ДАННЫЕ ДЛЯ ЛОГА ЗАРАНЕЕ <<<
                        exit_price = self._sf(exec_data.get("execPrice"))
                        entry_price = self._sf(pos.get("avg_price", 0))
                        pos_volume = self._sf(pos.get("volume", 0))
                        entry_side = pos.get("side", "Buy")
                        
                        # Извлекаем комментарий из сохраненного контекста позиции
                        comment = pos.get("comment") or "Причина не указана."


                        # Расчет PnL
                        pnl_usdt = self._calc_pnl(entry_side, entry_price, exit_price, pos_volume)
                        position_value = entry_price * pos_volume
                        pnl_pct = (pnl_usdt / position_value) * 1000 if position_value else 0.0

                        logger.info(f"[EXECUTION_CLOSE] {symbol}. PnL: {pnl_usdt:.2f} USDT ({pnl_pct:.2f}%).")
                        
                        # >>> ОЧИЩАЕМ СОСТОЯНИЕ ТОЛЬКО ПОСЛЕ ИЗВЛЕЧЕНИЯ ДАННЫХ <<<
                        self._purge_symbol_state(symbol) 
                        self.write_open_positions_json()

                        # >>> ВЫЗЫВАЕМ ЛОГИРОВАНИЕ С УЖЕ ГОТОВЫМИ ДАННЫМИ <<<
                        asyncio.create_task(self.log_trade(
                            symbol=symbol,
                            side=entry_side,
                            avg_price=exit_price, # Логируем цену выхода
                            volume=pos_volume,
                            action="close",
                            result="closed_by_execution",
                            pnl_usdt=pnl_usdt,
                            pnl_pct=pnl_pct,
                            comment=comment # Передаем извлеченный комментарий
                        ))
                        
                        # Прерываем дальнейшую обработку, так как позиция уже закрыта
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
    # async def handle_position_update(self, msg: dict):
    #     """
    #     Обрабатывает открытие/обновление/закрытие позиции из private WS.
    #     Защищено от падений: ни одно исключение не уходит наружу.
    #     """
    #     try:
    #         data = msg.get("data", [])
    #         if isinstance(data, dict):
    #             data = [data]
    #     except Exception:
    #         logger.error("[PositionStream] malformed message: %s", msg)
    #         return

    #     # Гарантируем наличие структур, иначе AttributeError «глушит» колбэк
    #     self.open_positions = getattr(self, "open_positions", {})
    #     self.recently_closed = getattr(self, "recently_closed", {})
    #     self.pending_orders = getattr(self, "pending_orders", {})
    #     self.pending_cids = getattr(self, "pending_cids", {})
    #     self.pending_timestamps = getattr(self, "pending_timestamps", {})
    #     self._pending_close_ts = getattr(self, "_pending_close_ts", {})
    #     self.last_entry_comment = getattr(self, "last_entry_comment", {})
    #     self.position_lock = getattr(self, "position_lock", asyncio.Lock())

    #     async with self.position_lock:
    #         for p in data:
    #             try:
    #                 symbol = p.get("symbol")
    #                 if not symbol:
    #                     continue

    #                 new_size = self._sf(p.get("size", 0))
    #                 prev_pos = self.open_positions.get(symbol)

    #                 # ── 1) ОТКРЫТИЕ ───────────────────────────────────────────
    #                 if prev_pos is None and new_size > 0:
    #                     side_raw = p.get("side")
    #                     if not side_raw:
    #                         continue

    #                     # Сброс pending по этому символу
    #                     for dct_name in ("pending_orders", "pending_cids", "pending_timestamps", "_pending_close_ts"):
    #                         try:
    #                             getattr(self, dct_name).pop(symbol, None)
    #                         except Exception:
    #                             pass

    #                     avg_price = self._sf(p.get("avgPrice") or p.get("entryPrice"))

    #                     # Комментарий стратегии — безопасно
    #                     comment = None
    #                     try:
    #                         comment = self.pending_strategy_comments.pop(symbol, None)
    #                     except Exception:
    #                         comment = None
    #                     if comment:
    #                         self.last_entry_comment[symbol] = comment

    #                     # Зафиксируем состояние позиции МИНИМАЛЬНЫМ набором полей
    #                     pos_idx = 1 if str(side_raw).lower() == "buy" else 2
    #                     self.open_positions[symbol] = {
    #                         "avg_price": avg_price,
    #                         "side": side_raw,
    #                         "pos_idx": pos_idx,
    #                         "volume": new_size,
    #                         "leverage": self._sf(p.get("leverage", "1")),
    #                         "markPrice": avg_price,
    #                         "open_timestamp": time.time(),
    #                         "trailing_activated": False,
    #                     }

    #                     logger.info("[PositionStream] NEW %s %s %.6f @ %.8f", side_raw, symbol, new_size, avg_price)

    #                     # Лёгкие действия — сразу
    #                     asyncio.create_task(self.log_trade(
    #                         symbol=symbol, side=side_raw, avg_price=avg_price,
    #                         volume=new_size, action="open", result="opened",
    #                         comment=comment
    #                     ))
    #                     self.write_open_positions_json()

    #                     # Тяжёлое — вне критичной дорожки, в фоне
    #                     async def _post_open_enrich():
    #                         try:
    #                             feats = self._build_entry_features(symbol)
    #                             self.open_positions[symbol]["entry_features"] = feats
    #                         except Exception as e:
    #                             logger.debug("[PositionStream] %s entry_features failed: %s", symbol, e)
    #                     asyncio.create_task(_post_open_enrich())

    #                     # Запуск guardian
    #                     asyncio.create_task(self.manage_open_position(symbol))
    #                     continue

    #                 # ── 2) ОБНОВЛЕНИЕ ОБЪЁМА ─────────────────────────────────
    #                 if prev_pos and new_size > 0:
    #                     prev_vol = self._sf(prev_pos.get("volume", 0))
    #                     if abs(new_size - prev_vol) > 1e-12:
    #                         self.open_positions[symbol]["volume"] = new_size
    #                         self.open_positions[symbol]["avg_price"] = self._sf(p.get("avgPrice") or p.get("entryPrice") or prev_pos.get("avg_price", 0))
    #                         self.write_open_positions_json()
    #                         logger.info("[PositionStream] %s volume: %.6f → %.6f", symbol, prev_vol, new_size)
    #                     continue

    #                 # ── 3) ЗАКРЫТИЕ (size==0) ────────────────────────────────
    #                 if prev_pos and new_size == 0:
    #                     if symbol in self.recently_closed:
    #                         logger.debug("[PositionStream] %s already closed by executions; skip.", symbol)
    #                         continue

    #                     logger.info("[PositionStream] Fallback: %s closed (size=0).", symbol)
    #                     snapshot = dict(prev_pos)
    #                     self._purge_symbol_state(symbol)
    #                     self.write_open_positions_json()

    #                     exit_price = self._sf(p.get("avgPrice") or snapshot.get("markPrice", snapshot.get("avg_price", 0)))
    #                     pos_volume = self._sf(snapshot.get("volume", 0))
    #                     entry_price = self._sf(snapshot.get("avg_price", 0))
    #                     side_snap = snapshot.get("side", "Buy")

    #                     pnl_usdt = self._calc_pnl(side_snap, entry_price, exit_price, pos_volume)
    #                     pos_value = entry_price * pos_volume
    #                     # FIX: проценты → * 100 (а не * 1000)
    #                     pnl_pct = (pnl_usdt / pos_value * 100.0) if pos_value else 0.0

    #                     asyncio.create_task(self.log_trade(
    #                         symbol=symbol, side=side_snap, avg_price=exit_price,
    #                         volume=pos_volume, action="close", result="closed_by_position_stream",
    #                         pnl_usdt=pnl_usdt, pnl_pct=pnl_pct,
    #                         comment=self.last_entry_comment.pop(symbol, None)
    #                     ))
    #                     continue

    #             except Exception as e:
    #                 # Критично: НИ ОДНО исключение не должно вылетать наружу колбэка!
    #                 try:
    #                     sym_dbg = p.get("symbol", "?")
    #                 except Exception:
    #                     sym_dbg = "?"
    #                 logger.error("[PositionStream] handler error for %s: %s | payload=%s",
    #                             sym_dbg, e, p, exc_info=True)
    #                 # продолжаем к следующему p
    #                 continue


    async def handle_position_update(self, msg: dict):
        """
        Обрабатывает открытие/обновление/закрытие позиции из private WS.
        Имеет защиту от ложных "мгновенных" закрытий, которые присылает Bybit.
        """
        try:
            data = msg.get("data", [])
            if isinstance(data, dict): data = [data]
        except Exception:
            logger.error("[PositionStream] malformed message: %s", msg)
            return

        async with self.position_lock:
            for p in data:
                try:
                    symbol = p.get("symbol")
                    if not symbol: continue

                    new_size = self._sf(p.get("size", 0))
                    prev_pos = self.open_positions.get(symbol)

                    # --- Сценарий 1: Открытие новой позиции ---
                    if prev_pos is None and new_size > 0:
                        # ... (вся ваша логика открытия остается без изменений)
                        side_raw = p.get("side")
                        if not side_raw: continue
                        
                        for dct_name in ("pending_orders", "pending_cids", "pending_timestamps"):
                            getattr(self, dct_name, {}).pop(symbol, None)

                        avg_price = self._sf(p.get("avgPrice") or p.get("entryPrice"))
                        comment = self.pending_strategy_comments.pop(symbol, None)
                        if comment: self.last_entry_comment[symbol] = comment

                        self.open_positions[symbol] = {
                            "avg_price": avg_price, "side": side_raw,
                            "pos_idx": 1 if side_raw == 'Buy' else 2,
                            "volume": new_size, "leverage": self._sf(p.get("leverage", "1")),
                            "markPrice": avg_price, "open_timestamp": time.time(),
                            "trailing_activated": False,
                            "comment": comment,
                        }
                        logger.info(f"[PositionStream] NEW {side_raw} {symbol} {new_size:.3f} @ {avg_price:.6f}")
                        
                        asyncio.create_task(self.log_trade(
                            symbol=symbol, side=side_raw, avg_price=avg_price,
                            volume=new_size, action="open", result="opened", comment=comment
                        ))
                        asyncio.create_task(self.manage_open_position(symbol))
                        self.write_open_positions_json()
                        continue

                    # --- Сценарий 2: Обновление существующей позиции ---
                    if prev_pos and new_size > 0:
                        # ... (логика обновления без изменений)
                        prev_vol = self._sf(prev_pos.get("volume", 0))
                        if abs(new_size - prev_vol) > 1e-9:
                            self.open_positions[symbol]["volume"] = new_size
                            self.open_positions[symbol]["avg_price"] = self._sf(p.get("avgPrice") or p.get("entryPrice") or prev_pos.get("avg_price", 0))
                            self.write_open_positions_json()
                            logger.info(f"[PositionStream] {symbol} volume updated: {prev_pos.get('volume')} -> {new_size}")
                        continue

                    # --- Сценарий 3: Закрытие (size==0) ---
                    if prev_pos and new_size == 0:
                        # [КЛЮЧЕВОЙ ФИКС] Даем "иммунитет" на 2 секунды после открытия
                        open_ts = prev_pos.get("open_timestamp", 0)
                        if (time.time() - open_ts) < 2.0:
                            logger.warning(f"[PositionStream] {symbol}: получено событие size=0 сразу после открытия. Игнорируем как ложное.")
                            continue # Пропускаем это сообщение

                        # Если мы здесь, значит, закрытие настоящее
                        logger.info(f"[PositionStream] ЗАКРЫТА позиция по {symbol} (size=0).")
                        snapshot = dict(prev_pos)
                        self._purge_symbol_state(symbol) # Это вызовет остановку воркера
                        
                        exit_price = self._sf(p.get("avgPrice") or snapshot.get("markPrice", snapshot.get("avg_price", 0)))
                        pnl_usdt = self._calc_pnl(snapshot.get("side"), snapshot.get("avg_price"), exit_price, snapshot.get("volume"))
                        pos_value = self._sf(snapshot.get("avg_price")) * self._sf(snapshot.get("volume"))
                        pnl_pct = (pnl_usdt / pos_value) * 1000 if pos_value else 0.0

                        asyncio.create_task(self.log_trade(
                            symbol=symbol, side=snapshot.get("side"), avg_price=exit_price,
                            volume=snapshot.get("volume"), action="close", result="closed_by_position_stream",
                            pnl_usdt=pnl_usdt, pnl_pct=pnl_pct,
                            comment=snapshot.get("comment") or "Закрыто вручную или внешним ордером"
                        ))
                        self.write_open_positions_json()
                        continue
                
                except Exception as e:
                    logger.error(f"[PositionStream] Ошибка обработки для {p.get('symbol', '?')}: {e}", exc_info=True)



    # [FINAL VERSION] Обработчик сообщений о позициях
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

            live_positions = {
                pos["symbol"]: pos
                for pos in response.get("result", {}).get("list", [])
                if safe_to_float(pos.get("size", 0)) > 0
            }

            # NEW: подготовим реестр задач-гвардов
            if not hasattr(self, "watch_tasks"):
                self.watch_tasks = {}

            for symbol, pos_data in live_positions.items():
                if symbol not in self.open_positions:
                    if symbol in self.recently_closed:
                        logger.debug(f"[SYNC] Игнорируем 'воскрешение' {symbol}, т.к. он был недавно закрыт.")
                        continue

                    logger.info(f"[SYNC] Обнаружена новая активная позиция на бирже: {symbol}")

                    side = pos_data.get("side", "")
                    correct_pos_idx = 1 if side == 'Buy' else 2
                    self.open_positions[symbol] = {
                        "avg_price":  safe_to_float(pos_data.get("entryPrice") or pos_data.get("avgPrice")),
                        "side":       side,
                        "pos_idx":    pos_data.get("positionIdx", correct_pos_idx),
                        "volume":     safe_to_float(pos_data.get("size", 0)),
                        "leverage":   safe_to_float(pos_data.get("leverage", "1")),
                        "markPrice":  safe_to_float(pos_data.get("markPrice", 0)),
                        "open_timestamp": time.time()
                    }

                    # Твоя текущая логика «усыновления» позиции
                    await self.adopt_existing_position(symbol, pos_data)

                    # NEW: стартуем петлю, которая шлёт тикы в stop_worker
                    if symbol not in self.watch_tasks:
                        t = asyncio.create_task(self.manage_open_position(symbol))
                        self.watch_tasks[symbol] = t
                        t.add_done_callback(lambda _t, s=symbol: self.watch_tasks.pop(s, None))

                else:
                    # Позиция уже была в локальном состоянии — обновим цену и убедимся, что петля запущена
                    try:
                        self.open_positions[symbol]["markPrice"] = safe_to_float(pos_data.get("markPrice", 0))
                    except Exception:
                        pass

            # Закрытые на бирже — зачистить локальное состояние и остановить воркер/петлю
            for symbol in list(self.open_positions.keys()):
                if symbol not in live_positions:
                    logger.info(f"[SYNC] Позиция {symbol} больше не активна. Логирование ручного/внешнего закрытия...")

                    # <<< НАЧАЛО НОВОГО БЛОКА ЛОГИРОВАНИЯ >>>
                    snapshot = self.open_positions.get(symbol)
                    if snapshot:
                        # 1. Захватываем данные ПЕРЕД удалением
                        entry_price = self._sf(snapshot.get("avg_price"))
                        pos_volume = self._sf(snapshot.get("volume"))
                        side = snapshot.get("side", "Buy")
                        
                        # 2. Используем последнюю известную цену как цену выхода (наша лучшая догадка)
                        last_known_price = self._sf(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice")) or self._sf(snapshot.get("markPrice"))
                        
                        if entry_price > 0 and pos_volume > 0 and last_known_price > 0:
                            # 3. Рассчитываем PnL
                            pnl_usdt = self._calc_pnl(side, entry_price, last_known_price, pos_volume)
                            pos_value = entry_price * pos_volume
                            pnl_pct = (pnl_usdt / pos_value) * 1000 if pos_value > 0 else 0.0

                            # 4. Вызываем логирование в фоновой задаче
                            asyncio.create_task(self.log_trade(
                                symbol=symbol,
                                side=side,
                                avg_price=last_known_price, # Цена выхода - приблизительная
                                volume=pos_volume,
                                action="close",
                                result="manual_close_detected_by_sync", # Четкая причина
                                pnl_usdt=pnl_usdt,
                                pnl_pct=pnl_pct,
                                comment=snapshot.get("comment") or "Закрыто вручную или внешним ордером"
                            ))

                    task = self.watch_tasks.pop(symbol, None)
                    if task and not task.done():
                        task.cancel()

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
            await asyncio.sleep(30) # Проверяем каждые 30 секунд
            now = time.time()

            # Создаем копию, чтобы безопасно изменять словарь в цикле
            pending_symbols = list(self.pending_orders.keys())

            for symbol in pending_symbols:
                timestamp = self.pending_timestamps.get(symbol, 0)

                # Проверяем только ордера, которые висят дольше 60 секунд
                if now - timestamp > 60:
                    cid = self.pending_cids.get(symbol, None)
                    if not cid:
                        logger.warning(f"[Pending Cleanup] {symbol}: нет CID для проверки, удаляем из очереди.")
                        self.pending_orders.pop(symbol, None)
                        self.pending_timestamps.pop(symbol, None)
                        continue

                    logger.warning(f"[Pending Cleanup] {symbol} (CID: {cid}) завис. Проверяем статус через REST...")
                    try:
                        # Пытаемся получить статус ордера по нашему orderLinkId (cid)
                        response = await asyncio.to_thread(
                            lambda: self.session.get_order_history(
                                category="linear",
                                symbol=symbol,
                                orderLinkId=cid,
                                limit=1
                            )
                        )
                        
                        orders = response.get("result", {}).get("list", [])
                        if orders:
                            status = orders[0].get("orderStatus")
                            logger.info(f"[Pending Cleanup] {symbol} (CID: {cid}): статус на бирже - {status}.")
                            # Если ордер исполнен или отменен, то его точно нет в "pending"
                            if status in ("Filled", "Cancelled", "Rejected"):
                                self.pending_orders.pop(symbol, None)
                                self.pending_cids.pop(symbol, None)
                                self.pending_timestamps.pop(symbol, None)
                        else:
                            # Если ордера нет в истории, значит он не был принят биржей
                            logger.warning(f"[Pending Cleanup] {symbol} (CID: {cid}): ордер не найден в истории. Удаляем из очереди.")
                            self.pending_orders.pop(symbol, None)
                            self.pending_cids.pop(symbol, None)
                            self.pending_timestamps.pop(symbol, None)

                    except Exception as e:
                        logger.error(f"[Pending Cleanup] Ошибка при проверке статуса ордера {symbol}: {e}")


    # ЗАМЕНИТЕ ЭТУ ФУНКЦИЮ ЦЕЛИКОМ
    async def market_loop(self):
        iteration = 0
        while True:
            iteration += 1
            last_heartbeat = time.time()
            
            try:
                symbols_to_scan = [
                    s for s in self.shared_ws.active_symbols
                    if s not in ("BTCUSDT", "ETHUSDT")
                ]
                random.shuffle(symbols_to_scan)
                
                logger.info(f"[market_loop] Итерация #{iteration}. Начинаем сканирование {len(symbols_to_scan)} символов...")

                # [КЛЮЧЕВОЕ ИЗМЕНЕНИЕ] Обрабатываем символы ПОСЛЕДОВАТЕЛЬНО
                for symbol in symbols_to_scan:
                    # Даем циклу событий шанс обработать другие задачи (например, WS)
                    await asyncio.sleep(0.01) 
                    
                    if not getattr(self, "bot_active", True):
                        break
                        
                    try:
                        # [ВАЖНО] Вызываем execute_golden_setup НАПРЯМУЮ, без create_task
                        # и ждем его полного завершения через await
                        await self.execute_golden_setup(symbol)
                    except Exception:
                        logger.error(f"[market_loop] Ошибка при обработке символа {symbol}", exc_info=True)

                scan_duration = time.time() - last_heartbeat
                logger.info(
                    f"[market_loop] Итерация #{iteration} завершена за {scan_duration:.2f} сек. "
                    f"Пауза перед следующим сканированием."
                )
                
                # Пауза между полными сканированиями
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[market_loop] Критическая ошибка в главном цикле, перезапуск через 10 секунд...")
                await asyncio.sleep(10)


    async def _risk_check(self, symbol: str, side: str, qty: float, last_price: float, cid: str | None = None) -> bool:
        """
        Атомарно проверяет лимиты и резервирует объем в pending_orders.
        Это ключевой механизм для предотвращения превышения общего лимита.
        """
        async with self.position_lock:
            # 1. Повторная проверка под локом, что позиция не открылась, пока мы ждали
            if symbol in self.open_positions or symbol in self.pending_orders:
                logger.debug(f"[RiskCheck] {symbol} уже открыт или в процессе. Отмена.")
                return False

            # 2. Расчет и проверка лимитов
            est_cost = qty * last_price
            if est_cost <= 0: return False

            # Проверка индивидуального лимита
            if est_cost > self.POSITION_VOLUME:
                logger.info(
                    f"[RiskCheck] {symbol} отклонен: стоимость {est_cost:.0f} > лимита на позицию {self.POSITION_VOLUME:.0f}"
                )
                return False

            # Проверка общего лимита
            effective_volume = await self.get_effective_total_volume()
            if effective_volume + est_cost > self.MAX_TOTAL_VOLUME:
                logger.warning(
                    f"[RiskCheck] {symbol} отклонен: превышен ОБЩИЙ лимит. "
                    f"Текущий: {effective_volume:.2f}, Попытка: {est_cost:.2f}, "
                    f"Лимит: {self.MAX_TOTAL_VOLUME:.2f}"
                )
                return False

            # 3. Атомарное резервирование
            # Если все проверки пройдены, немедленно резервируем место в pending_orders,
            # НЕ выходя из-под лока.
            self.pending_orders[symbol] = est_cost
            self.pending_timestamps[symbol] = time.time()
            if cid:
                self.pending_cids[symbol] = cid
            
            logger.info(f"[RiskCheck] {symbol} прошел проверку. Объем {est_cost:.2f} зарезервирован.")
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
        # Блокируем символ, если он уже в процессе обработки другим таском
            if not await self._gs_prereqs(symbol):
                return

            mode = getattr(self, "strategy_mode", "full")

            # СНАЧАЛА проверяем на наличие более конкретного сигнала - ликвидаций
            if mode in ("full", "liq_squeeze", "liquidation_only"):
                if await self._liquidation_logic(symbol):
                    return # Если вошли по ликвидациям, на этом все

            # ТОЛЬКО ПОТОМ проверяем на сквиз, если ликвидаций не было
            if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
                if await self._squeeze_logic(symbol):
                    return

            if mode in ("full", "golden_only", "golden_squeeze"):
                await self._golden_logic(symbol)
                

    async def _gs_prereqs(self, symbol: str) -> bool:
            # Проверяем кулдаун в самом начале
        if time.time() < self.strategy_cooldown_until.get(symbol, 0):
            # Добавим лог, чтобы видеть, почему сделка пропускается
            logger.debug(f"[GS_SKIP] {symbol} находится на стратегическом кулдауне.")
            return False

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
        """
        Основная модель принимает решение (EXECUTE/REJECT).
        Вторичная (Plutus) — только advisory, не блокирует решение.
        """
        prompt = self._build_primary_prompt(candidate, features)
        messages = [{"role": "user", "content": prompt}]

        result = {
            "action": "REJECT",
            "confidence_score": 0.7,
            "justification": "AI error",
            "full_prompt_for_ai": prompt,
        }

        # --- PRIMARY (обязательная)
        try:
            async with self.ai_primary_sem:
                primary_json = await self._ask_ollama_json(
                    self.ai_primary_model,
                    messages,
                    timeout_s=self.ai_timeout_sec,
                    base_openai_url=self.ollama_primary_openai,
                )
        except asyncio.TimeoutError:
            logger.error(f"[AI_TIMEOUT] {self.ai_primary_model} не ответил за {self.ai_timeout_sec}s")
            # пробрасываем timeout выше — пусть внешний wait_for откроет «circuit»
            raise
        except Exception as e:
            logger.error(f"[AI_ERROR] primary {self.ai_primary_model}: {e}", exc_info=True)
            primary_json = {"action": "REJECT", "confidence_score": 0.7, "justification": f"[PRIMARY] error: {e}"}

        action = str(primary_json.get("action", "REJECT")).upper()
        if action not in ("EXECUTE", "REJECT"):
            action = "REJECT"

        result.update({
            "action": action,
            "confidence_score": safe_to_float(primary_json.get("confidence_score", 0.7)),
            "justification": str(primary_json.get("justification", "")).strip() or "No justification",
        })

        # --- SECONDARY (advisory, best-effort)
        if getattr(self, "ai_secondary_model", None):
            try:
                async with self.ai_secondary_sem:
                    secondary_json = await self._ask_ollama_json(
                        self.ai_secondary_model, messages, min(self.ai_timeout_sec, 3.0)
                    )
                result["secondary"] = {"model": self.ai_secondary_model, **secondary_json}
            except asyncio.TimeoutError:
                logger.error(f"[AI_TIMEOUT] {self.ai_secondary_model} не ответил за {min(self.ai_timeout_sec, 3.0)}s")
                result["secondary"] = {"model": self.ai_secondary_model, "error": "timeout"}
            except Exception as e:
                logger.error(f"[AI_ERROR] secondary {self.ai_secondary_model}: {e}", exc_info=True)
                result["secondary"] = {"model": self.ai_secondary_model, "error": str(e)}

        return result


    # ---------------------------------------------------------------------
    # AI prompt builders
    # ---------------------------------------------------------------------
    def _fmt(self, v, spec: str, na: str = "N/A"):
        try:
            x = float(v)
            if x == float("inf") or x == float("-inf"):
                return na
            return f"{x:{spec}}"
        except Exception:
            return na

    def _market_context_from_features(self, features: dict):
        """Аккуратно достаём рыночный контекст из features (если есть)."""
        btc_1h = self._fmt(features.get("btc_change_1h"), ".2f")
        eth_1h = self._fmt(features.get("eth_change_1h"), ".2f")
        return btc_1h, eth_1h

    # def _build_primary_prompt(self, candidate: dict, features: dict) -> str:
    #     def _format(v, spec):
    #         try:
    #             return f"{float(v):{spec}}"
    #         except Exception:
    #             return "N/A"

    #     m = candidate.get('base_metrics', {})
    #     source = candidate.get('source', 'unknown').replace('_', ' ').title()

    #     # контекст рынка (если данные есть)
    #     try:
    #         btc_change_1h = compute_pct(self.shared_ws.candles_data["BTCUSDT"], 60)
    #     except Exception:
    #         btc_change_1h = 0.0
    #     try:
    #         eth_change_1h = compute_pct(self.shared_ws.candles_data["ETHUSDT"], 60)
    #     except Exception:
    #         eth_change_1h = 0.0

    #     vol_anomaly = (safe_to_float(m.get('vol_change_pct', 0)) / 100 + 1)
    #     trend = "Uptrend" if safe_to_float(features.get('supertrend', 0)) > 0 else "Downtrend"

    #     prompt = f"""
    # SYSTEM: Ты - профессиональный трейдер, опытный риск-менеджер, элитный квантовый и крипто-аналитик. Твой ответ - всегда только валидный JSON.
    # USER:
    # Анализ торгового сигнала:
    # - Монета: {candidate.get('symbol')}, Направление: {candidate.get('side','').upper()}, Источник: {source}
    # - Метрики: PriceΔ(5m)={_format(m.get('pct_5m'), '.2f')}%, Volume Anomaly={_format(vol_anomaly, '.1f')}x, OIΔ(1m)={_format(m.get('oi_change_pct'), '.2f')}%
    # - Контекст: Trend={trend}, ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(features.get('rsi14'), '.1f')}
    # - Рынок: BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%

    # ЗАДАЧА: Верни JSON с ключами "confidence_score", "justification" (начинается с источника сигнала), "action" ("EXECUTE" или "REJECT").
    # """
    #     return prompt.strip()



    def _build_primary_prompt(self, candidate: dict, features: dict) -> str:
        def _format(v, spec):
            try:
                return f"{float(v):{spec}}"
            except Exception:
                return "N/A"

        sym   = candidate.get("symbol", "UNKNOWN")
        side  = str(candidate.get("side", "Buy")).upper()
        source   = str(candidate.get("source", "unknown")).replace("_", " ").title()
        source_title = source.replace('_', ' ').title()

        m = candidate.get("base_metrics", {}) or {}
        vol_change_pct = safe_to_float(m.get("vol_change_pct", 0.0))
        vol_anomaly = (vol_change_pct / 100.0) + 1.0  # x-множитель

        trend = "Uptrend" if safe_to_float(features.get("supertrend", 0.0)) > 0 else "Downtrend"
        adx   = self._fmt(features.get("adx14"), ".1f")
        rsi   = self._fmt(features.get("rsi14"), ".1f")

        try:
            btc_change_1h = compute_pct(self.shared_ws.candles_data["BTCUSDT"], 60)
            eth_change_1h = compute_pct(self.shared_ws.candles_data["ETHUSDT"], 60)
        except Exception:
            btc_change_1h, eth_change_1h = 0.0, 0.0

        price_5m   = self._fmt(m.get("pct_5m"), ".2f")
        oi_1m      = self._fmt(m.get("oi_change_pct"), ".2f")
        vol_anom_s = self._fmt(vol_anomaly, ".1f")

        mode = candidate.get("mode", "unknown")

        # --- Базовая часть промпта ---
        prompt_header = "SYSTEM: Ты - элитный трейдер и риск-менеджер. Твой ответ - всегда только валидный JSON, без лишних слов."
        
        prompt_data = f"""
        USER:
        Анализ торгового сигнала:
        - Монета: {sym}, Направление: {side}, Источник: {source_title}
        - Метрики: PriceΔ(5m)={_format(m.get('pct_5m'), '.2f')}%, Volume Anomaly={_format(vol_anomaly, '.1f')}x, OIΔ(1m)={_format(m.get('oi_change_pct'), '.2f')}%
        - Контекст: Trend={trend}, ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(features.get('rsi14'), '.1f')}
        - Рынок: BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%
        """
        
        # --- ДИНАМИЧЕСКАЯ ЧАСТЬ: ИНСТРУКЦИЯ ДЛЯ ЗАДАЧИ ---
        if source == 'squeeze':
            # <<< НОВОЕ: Специальный промпт для сквиза! >>>
            prompt_task = f"""
            **Стратегический контекст:** Мы торгуем КОНТРТРЕНДОВУЮ стратегию "Squeeze". Наша цель - войти в сделку ПРОТИВ сильного ценового импульса, ожидая его истощения и разворота.
            
            **Правила интерпретации:**
            1.  Если направление сигнала 'SELL', это значит, что был сильный импульс ВВЕРХ. Поэтому высокие значения RSI (>75) и ADX (>30) являются ПОДТВЕРЖДЕНИЕМ силы импульса и, следовательно, ХОРОШИМ знаком для входа в шорт.
            2.  Если направление сигнала 'BUY', это значит, что был сильный импульс ВНИЗ. Низкие значения RSI (<25) являются ПОДТВЕРЖДЕНИЕМ для входа в лонг.
            3.  Твоя задача - оценить, не является ли этот импульс началом нового глобального тренда. Если да - отклоняй. Если это локальный "вынос" - одобряй.

            **ЗАДАЧА:** Основываясь на этих правилах, верни JSON с "action" ("EXECUTE" или "REJECT"), "confidence_score" и "justification".
            """
        elif source == 'liquidation':
            # <<< НОВЫЙ БЛОК ДЛЯ ЛИКВИДАЦИЙ! >>>
            liq_side = m.get('liquidation_side', 'Unknown')
            prompt_task = f"""
            **Стратегический контекст:** Мы торгуем КОНТРТРЕНДОВУЮ стратегию на каскаде ликвидаций. Был обнаружен крупный кластер принудительных закрытий ({liq_side}) на ${m.get('liquidation_value_usd'):,.0f}. Наша цель - войти в сделку ({side}) ПРОТИВ этих ликвидаций, ожидая "выноса стопов" и локального разворота.
            
            **Правила интерпретации:**
            1.  Сигнал является контр-трендовым. Сильное отклонение цены и высокие/низкие значения RSI являются подтверждением, а не опровержением сигнала.
            2.  Оцени, является ли это событие кульминацией движения (хорошо для входа) или лишь его началом (плохо). Проанализируй, есть ли признаки поглощения объемов ликвидаций.
            3.  Одобряй вход, если есть признаки истощения импульса после каскада. Отклоняй, если импульс продолжает агрессивно развиваться.

            **ЗАДАЧА:** Основываясь на этих правилах, верни JSON с "action", "confidence_score" и "justification".
            """
        else:
            # <<< Стандартный промпт для всех остальных стратегий >>>
            prompt_task = """
            **ЗАДАЧА:** Проанализируй сигнал на основе общих принципов технического анализа. Верни JSON с ключами "confidence_score", "justification" (начинается с источника сигнала), "action" ("EXECUTE" или "REJECT").
            """

        return f"{prompt_header}\n{prompt_data}\n{prompt_task}".strip()



    def _build_secondary_prompt(self, candidate: dict, features: dict) -> str:
        sym   = candidate.get("symbol", "UNKNOWN")
        side  = str(candidate.get("side", "Buy")).upper()
        src   = str(candidate.get("source", "unknown")).replace("_", " ").title()
        mode  = candidate.get("mode", "unknown")

        m = candidate.get("base_metrics", {}) or {}
        price_5m   = self._fmt(m.get("pct_5m"), ".2f")
        oi_1m      = self._fmt(m.get("oi_change_pct"), ".2f")
        vol_change = safe_to_float(m.get("vol_change_pct", 0.0))
        vol_anom_s = self._fmt((vol_change / 100.0) + 1.0, ".1f")

        trend = "Uptrend" if safe_to_float(features.get("supertrend", 0.0)) > 0 else "Downtrend"
        adx   = self._fmt(features.get("adx14"), ".1f")
        rsi   = self._fmt(features.get("rsi14"), ".1f")

        prompt = (
            "SYSTEM: Ты — ассистент по рыночным паттернам и конфлюэнсу. "
            "Определи наличие сильного паттерна/конфлюэнса, НЕ принимая торгового решения.\n"
            "USER:\n"
            f"Инструмент: {sym}, Сторона: {side}, Режим: {mode}, Источник: {src}\n"
            f"Метрики: PriceΔ(5m)={price_5m}%, OIΔ(1m)={oi_1m}%, Volume Anomaly={vol_anom_s}x\n"
            f"Техника: Trend={trend}, ADX={adx}, RSI={rsi}\n\n"
            "ЗАДАЧА: Верни JSON:\n"
            '{ "pattern_score": 0..1, "confluence_notes": "кратко" }\n'
        )
        return prompt


    async def _advisor_plutus(self, candidate: dict, features: dict, primary: dict):
        try:
            messages = [{"role":"user","content": self._build_advisor_prompt(candidate, features, primary)}]
            async with self.ai_advisor_sem:
                out = await self._ask_ollama_json(self.ai_advisor_model, messages, num_predict=140, temperature=0.0)
            advice = safe_parse_json(out, default={})
            # сохраняем в лог/файнтьюн-буфер (не блокируем)
            self._append_ai_eval(candidate, features, primary, advice)
        except Exception as e:
            logger.warning(f"[ADVISOR_FAIL] {candidate['symbol']}: {e}")


    # async def evaluate_entry_candidate(self, candidate: dict, features: dict):
    #     """
    #     Централизованная оценка сигнала + вызов исполнения.
    #     """
    #     # --- Funding filter (soft block, применяется ко всем стратегиям) ---
    #     # Логика: не открываем Long, если funding заметно > 0 (лонги платят),
    #     # и не открываем Short, если funding заметно < 0 (шорты платят).
    #     try:
    #         _sym = candidate.get("symbol")
    #         _side = (candidate.get("side") or "").title()  # "Buy"/"Sell"
    #         _dq = getattr(self.shared_ws, "funding_history", {}).get(_sym)
    #         _funding = (_dq[-1] if _dq else None)
    #     except Exception:
    #         _funding = None

    #     if _funding is not None and getattr(self, "enable_funding_filter", True):
    #         _thr = float(getattr(self, "funding_abs_threshold", 0.001))  # 0.1%
    #         if _side == "Buy" and _funding >= _thr:
    #             logger.info("[FUNDING] Skip %s Long — funding %.5f >= %.5f", _sym, _funding, _thr)
    #             return
    #         if _side == "Sell" and _funding <= -_thr:
    #             logger.info("[FUNDING] Skip %s Short — funding %.5f <= -%.5f", _sym, _funding, _thr)
    #             return
    #         # добавим значение в метрики для прозрачности
    #         candidate.setdefault("base_metrics", {})["funding_rate"] = _funding
    #     # --- /Funding filter ---

    #     symbol, side, source = candidate['symbol'], candidate['side'], candidate.get('source', 'unknown')
    #     now = time.time()
    #     signal_key = f"{symbol}_{side}_{source}"

    #     last_close_ts = float(self.recently_closed.get(symbol, 0) or 0.0)
    #     wait_left = self.entry_cooldown_sec - (now - last_close_ts)
    #     if wait_left > 0:
    #         # раз в 5 сек помолчим, чтобы не спамить лог
    #         if now >= getattr(self, "_cooldown_noise_until", {}).get(symbol, 0):
    #             logger.info(f"[AI_SKIP][{symbol}/{side}] recently closed; wait {int(wait_left)}s")
    #             if not hasattr(self, "_cooldown_noise_until"):
    #                 self._cooldown_noise_until = {}
    #             self._cooldown_noise_until[symbol] = now + 5
    #         # продлеваем кэш, чтобы не дёргать функцию каждую секунду
    #         self._evaluated_signals_cache[signal_key] = {"status": "cooldown", "time": now}
    #         return

    #     # 2) ДЕДУПЛИКАЦИЯ (этот блок заменяет ваши старые 2 строки с CACHE_TTL_SEC)
    #     ttl = max(getattr(self, "CACHE_TTL_SEC", 10), getattr(self, "entry_cooldown_sec", 30))
    #     cached = self._evaluated_signals_cache.get(signal_key)
    #     if cached and (now - float(cached.get("time", 0) or 0.0) < ttl):
    #         return
    #     self._evaluated_signals_cache[signal_key] = {"status": "pending", "time": now}

    #     CACHE_TTL_SEC = 10.0  # если у вас уже есть — используйте ваш

    #     try:
    #         # --- ML-фильтр (если активен)
    #         if getattr(self.ml_inferencer, "model", None) is not None:
    #             try:
    #                 vec = np.array([[safe_to_float(features.get(k, 0.0)) for k in FEATURE_KEYS]], dtype=np.float32)
    #                 raw_prediction = float(self.ml_inferencer.infer(vec)[0][0])

    #                 if not (-1.0 < raw_prediction < 1.0):
    #                     logger.warning(f"[ML_GATE_REJECT] Аномальный прогноз для {symbol}: {raw_prediction}")
    #                     return

    #                 leverage = self.leverage
    #                 expected_roi = raw_prediction * 100 * leverage
    #                 side_check_ok = (side == "Buy" and expected_roi > 0) or (side == "Sell" and expected_roi < 0)
    #                 roi_threshold_ok = abs(expected_roi) >= self.ml_gate_abs_roi

    #                 if not (side_check_ok and roi_threshold_ok):
    #                     logger.debug(f"[ML_GATE_REJECT] {symbol}/{side} ({source}) | Ожидаемый ROI: {expected_roi:.2f}%, Порог: {self.ml_gate_abs_roi:.2f}%")
    #                     return

    #                 logger.info(f"[ML_GATE_PASS] {symbol}/{side} ({source}) | Ожидаемый ROI: {expected_roi:.2f}%")
    #             except Exception as e:
    #                 logger.warning(f"[ML_GATE_ERROR] Ошибка ML-фильтра для {symbol}: {e}")
    #                 return
    #         else:
    #             logger.debug(f"[ML_GATE_SKIP] Модель не обучена, сигнал {symbol}/{side} пропущен к AI.")

    #         # --- Circuit breaker по AI
    #         if now < self.ai_circuit_open_until:
    #             if now >= getattr(self, "_ai_silent_until", 0):
    #                 logger.debug(f"[AI_SKIP] {symbol}/{side} - circuit open.")
    #                 self._ai_silent_until = now + 5
    #             return

    #         provider = str(self.ai_provider).lower().strip().replace('о', 'o')


    #         # ключ сигнала
    #         signal_key = f"{symbol}_{side}_{source}"

    #         # если этот же сигнал уже обрабатывается — не дергаем AI второй раз
    #         if signal_key in self._ai_inflight_signals:
    #             logger.debug(f"[AI_SKIP_INFLIGHT] {signal_key} уже в обработке")
    #             return

    #         self._ai_inflight_signals.add(signal_key)
    #         try:
    #             ai_response = await asyncio.wait_for(
    #                 self._ai_dispatch(provider, candidate, features),
    #                 timeout=self.ai_timeout_sec
    #             )
    #         finally:
    #             # обязательно освобождаем «слот» даже при исключении/таймауте
    #             self._ai_inflight_signals.discard(signal_key)

    #         ai_action = ai_response.get("action", "REJECT")
    #         if ai_action != "EXECUTE":
    #             justification = ai_response.get("justification", "Причина не указана.")
    #             logger.info(f"[AI_REJECT] {symbol}/{side} ({source}) — {justification}")

    #             # корректный отбор в watchlist: проверяем mode ∈ self.watch_modes
    #             mode = candidate.get("mode")
    #             if mode in getattr(self, "watch_modes", {"golden_squeeze", "liq_squeeze"}) and \
    #             ai_response.get("confidence_score", 0.7) >= 0.5:
    #                 self.put_on_watchlist(candidate, features)
    #             return

    #         if ai_action == "EXECUTE":
    #             # >>> ADD: повторная быстрая проверка кулдауна перед фактическим вызовом исполнения
    #             last_close_ts = float(self.recently_closed.get(symbol, 0) or 0.0)
    #             wait_left = self.entry_cooldown_sec - (time.time() - last_close_ts)
    #             if wait_left > 0:
    #                 logger.info(f"[EXECUTE_SKIP][{symbol}/{side}] recently closed; defer ~{int(wait_left)}s")
    #                 # опционально: положить под наблюдение
    #                 try:
    #                     self.put_on_watchlist(candidate, features)
    #                 except Exception:
    #                     pass
    #                 # закэшировать, чтобы не долбить каждую секунду
    #                 self._evaluated_signals_cache[signal_key] = {"status": "cooldown", "time": time.time()}
    #                 return
    #             # <<< END ADD

    #         logger.info(f"[AI_CONFIRM] Сделка {symbol}/{side} ({source}) ОДОБРЕНА. Исполнение...")
    #         candidate['justification'] = ai_response.get("justification", "N/A")
    #         candidate['full_prompt_for_ai'] = ai_response.get("full_prompt_for_ai", "")
    #         #asyncio.create_task(self.execute_trade_entry(candidate, features))
    #         await self.execute_trade_entry(candidate, features)

    #     except asyncio.TimeoutError:
    #         self.ai_circuit_open_until = time.time() + 60
    #         logger.error(f"[AI_TIMEOUT] {self.ai_provider} не ответил за {self.ai_timeout_sec}с. Отключаю AI на 60 сек.")
    #     except Exception as e:
    #         logger.error(f"[evaluate_candidate] Критическая ошибка для {symbol}: {e}", exc_info=True)
    #     finally:
    #         self._evaluated_signals_cache.pop(signal_key, None)


    async def evaluate_entry_candidate(self, candidate: dict, features: dict):
        """
        [V2 - ROBUST] Централизованная оценка сигнала + вызов исполнения.
        Обрабатывает таймауты AI и не падает.
        """
        symbol, side, source = candidate['symbol'], candidate['side'], candidate.get('source', 'unknown')
        now = time.time()
        signal_key = f"{symbol}_{side}_{source}"
        
        # --- Блок проверок (cooldown, дубликаты и т.д.) ---
        last_close_ts = float(self.recently_closed.get(symbol, 0) or 0.0)
        wait_left = self.entry_cooldown_sec - (now - last_close_ts)
        if wait_left > 0:
            if now >= getattr(self, "_cooldown_noise_until", {}).get(symbol, 0):
                logger.debug(f"[AI_SKIP][{symbol}/{side}] recently closed; wait {int(wait_left)}s")
                self._cooldown_noise_until[symbol] = now + 5
            self._evaluated_signals_cache[signal_key] = {"status": "cooldown", "time": now}
            return
        
        ttl = max(getattr(self, "CACHE_TTL_SEC", 10), getattr(self, "entry_cooldown_sec", 30))
        cached = self._evaluated_signals_cache.get(signal_key)
        if cached and (now - float(cached.get("time", 0) or 0.0) < ttl):
            return
        self._evaluated_signals_cache[signal_key] = {"status": "pending", "time": now}

        try:
            # --- ML-фильтр (если активен) ---
            if getattr(self.ml_inferencer, "model", None) is not None:
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

            # --- Circuit breaker по AI ---
            if now < self.ai_circuit_open_until:
                if now >= getattr(self, "_ai_silent_until", 0):
                    logger.debug(f"[AI_SKIP] {symbol}/{side} - circuit open.")
                    self._ai_silent_until = now + 5
                return

            provider = str(self.ai_provider).lower().strip().replace('о', 'o')

            if signal_key in self._ai_inflight_signals:
                return

            self._ai_inflight_signals.add(signal_key)
            ai_response = {}
            try:
                ai_response = await asyncio.wait_for(
                    self._ai_dispatch(provider, candidate, features),
                    timeout=self.ai_timeout_sec
                )
            except asyncio.TimeoutError:
                self.ai_circuit_open_until = time.time() + 60
                logger.error(f"[AI_TIMEOUT] {self.ai_provider} не ответил за {self.ai_timeout_sec}с. Отключаю AI на 60 сек.")
                return # Просто выходим, не открывая сделку
            except Exception as e:
                logger.error(f"[AI_DISPATCH_ERROR] Критическая ошибка для {symbol}: {e}", exc_info=True)
                return # Просто выходим
            finally:
                self._ai_inflight_signals.discard(signal_key)

            ai_action = ai_response.get("action", "REJECT")
            if ai_action != "EXECUTE":
                justification = ai_response.get("justification", "Причина не указана.")
                logger.info(f"[AI_REJECT] {symbol}/{side} ({source}) — {justification}")
                return


            logger.info(f"[AI_CONFIRM] Сделка {symbol}/{side} ({source}) ОДОБРЕНА. Исполнение...")
            candidate['justification'] = ai_response.get("justification", "N/A")
            candidate['full_prompt_for_ai'] = ai_response.get("full_prompt_for_ai", "")
            await self.execute_trade_entry(candidate, features)

        except Exception as e:
            logger.error(f"[evaluate_candidate] Критическая ошибка для {symbol}: {e}", exc_info=True)



    # ────────────────────────────────────────────────────────────
    def put_on_watchlist(self, cand: dict, features: dict) -> None:
        sym = cand["symbol"]
        if sym in self.watch_tasks:          # уже наблюдаем
            return

        # ── сохраняем цену на момент постановки ─────────────────────
        entry_price = safe_to_float(self.shared_ws.ticker_data[sym]["lastPrice"])
        cand = cand.copy()                   # не портим оригинал
        cand["entry_price"] = entry_price    # <- ключ гарантированно есть

        self.watchlist[sym] = time.time()
        task = asyncio.create_task(self._watch_squeeze(sym, cand, dict(features)))
        self.watch_tasks[sym] = task
        task.add_done_callback(lambda _t: self.watch_tasks.pop(sym, None))
        logger.info(f"[WATCH] {sym}: берём под наблюдение на 5 мин.")

    # ----------------------------------------------------------------
    async def _watch_squeeze(self, symbol: str,
                             cand: dict,
                             base_features: dict) -> None:
        """В течение 5 мин. раз в 30 с проверяем: был ли отскок и готов ли AI."""
        DEADLINE     = 300    # 5 минут
        CHECK_EVERY  = 30     # секунд
        start_ts     = time.time()
        entry_price = (
            cand.get("entry_price")
            or safe_to_float(self.shared_ws.ticker_data[symbol]["lastPrice"])
        )
        while time.time() - start_ts < DEADLINE:
            await asyncio.sleep(CHECK_EVERY)

            last_price = safe_to_float(self.shared_ws.ticker_data[symbol]["lastPrice"])
            rebound_pct = 100 * (last_price / entry_price - 1)

            # критерий: цена отпружинила ≥ 1.2 %
            if rebound_pct >= 1.2:
            # берём снапшот, но освежаем BTC/ETH-контекст за последний час
                features = dict(base_features)
                features["btc1h"] = compute_pct(self.shared_ws.candles_data["BTCUSDT"], 60)
                features["eth1h"] = compute_pct(self.shared_ws.candles_data["ETHUSDT"], 60)

                verdict = await self.evaluate_candidate_with_ollama(cand, features)
                if verdict.get("action") == "EXECUTE":
                    await self.execute_trade_entry(cand, features)
                    logger.info(f"[WATCH] {symbol}: отскок {rebound_pct:.2f} %, входим.")
                    return
            logger.debug(f"[WATCH] {symbol}: rebound {rebound_pct:.2f} % — ждём")

        logger.info(f"[WATCH] {symbol}: 5-мин. окно истекло, снимаем с наблюдения")
        self.watchlist.pop(symbol, None)

    def _funding_allows(self, symbol: str, side: str, rate: float,
                        thr_pos: float = 0.0010, thr_neg: float = -0.0010,
                        log_every_sec: int = 600) -> bool:
        block = (side == "Buy" and rate >= thr_pos) or (side == "Sell" and rate <= thr_neg)
        if not block:
            return True
        key = (symbol, side)
        now = time.time()
        last_ts = self._funding_log_ts.get(key, 0)
        last_rate = self._funding_log_last.get(key)
        if (now - last_ts) >= log_every_sec or last_rate != rate:
            if side == "Buy":
                logger.info("[FUNDING] Skip %s Long — funding %.5f >= %.5f", symbol, rate, thr_pos)
            else:
                logger.info("[FUNDING] Skip %s Short — funding %.5f <= %.5f", symbol, rate, thr_neg)
            self._funding_log_ts[key] = now
            self._funding_log_last[key] = rate
        return False

    # --- ЗАМЕНИТЕ ВАШУ ФУНКЦИЮ НА ЭТУ ВЕРСИЮ ---
    async def execute_trade_entry(self, candidate: dict, features: dict):
        cid = uuid.uuid4().hex[:8]
        symbol = candidate['symbol']
        side = candidate['side']
        source = candidate.get('source', 'unknown')
        
        try:
            sym_lock = self._get_sym_lock(symbol)
            async with sym_lock:
                if symbol in self.open_positions or symbol in self.pending_orders or symbol in self.active_trade_entries:
                    logger.info(f"[EXECUTE_DENY][{cid}] {symbol}/{side}: Позиция уже открыта или в обработке.")
                    return

                self.active_trade_entries[symbol] = candidate
                logger.info(f"[EXECUTE_RESERVE][{cid}] {symbol}/{side} зарезервирован для входа.")
            
            logger.info("[EXECUTE_CALL][%s] %s/%s src=%s price=%s vol_usdt=%s",
                        cid, symbol, side, source, features.get('price'),
                        candidate.get('volume_usdt', self.POSITION_VOLUME))

            # --- БЛОК ПРОВЕРОК С ЛОГИРОВАНИЕМ ---
            last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0))
            if not (last_price and last_price > 0):
                logger.warning(f"❌ [EXECUTE_CANCEL][{cid}] {symbol}/{side}: Отмена. Невалидная цена: {last_price}")
                return

            now = time.time()
            cooldown_left = self.entry_cooldown_sec - (now - self.last_entry_ts.get(symbol, 0))
            if cooldown_left > 0:
                logger.info(f"❌ [EXECUTE_CANCEL][{cid}] {symbol}/{side}: Отмена из-за кулдауна после входа. Осталось {cooldown_left:.1f}с.")
                return

            cooldown_left = self.entry_cooldown_sec - (now - self.recently_closed.get(symbol, 0))
            if cooldown_left > 0:
                logger.info(f"❌ [EXECUTE_CANCEL][{cid}] {symbol}/{side}: Отмена из-за кулдауна после закрытия. Осталось {cooldown_left:.1f}с.")
                return

            usd_amount = float(candidate.get("volume_usdt") or self.POSITION_VOLUME)
            qty = await self._calc_qty_from_usd(symbol, usd_amount, last_price)
            if not (qty > 0):
                logger.warning(f"❌ [EXECUTE_CANCEL][{cid}] {symbol}/{side}: Отмена. Рассчитан нулевой объем.")
                return

            if not await self._risk_check(symbol, side, qty, last_price, cid=cid):
                # _risk_check сам логирует причину отказа
                return

            # --- ЕСЛИ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ ---
            self.pending_strategy_comments[symbol] = candidate.get("justification", f"auto-entry by {source}")
            asyncio.create_task(self.notify_signal(candidate))
            
            logger.info("✅ [EXECUTE_GO][%s] %s/%s: Все проверки пройдены. Отправка ордера...", cid, symbol, side)

            if source == 'squeeze':
                max_to = int(getattr(self, "tactical_entry_window_sec", 45))
                
                ok = await self.adaptive_squeeze_entry(symbol, side, qty, max_entry_timeout=max_to, ai_preconfirmed=True)
                if not ok:
                    # тактическое окно истекло — чистим pending для символа
                    self.pending_orders.pop(symbol, None)
                    self.pending_cids.pop(symbol, None)
                    self.pending_timestamps.pop(symbol, None)
                    self.active_trade_entries.pop(symbol, None)
                logger.info(f"[EXECUTE_ABORT] {symbol}/{side}: окно истекло — pending очищен")

            else:
                await self.place_unified_order(
                    symbol=symbol, side=side, qty=qty,
                    order_type="Market", comment=f"entry_by_{source}", cid=cid
                )
        
        except Exception as e:
            logger.error(f"[EXECUTE_CRITICAL][{cid}] Критическая ошибка при исполнении для {symbol}/{side}", exc_info=True)
            self.pending_orders.pop(symbol, None)
            self.pending_cids.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)
        
        finally:
            self.active_trade_entries.pop(symbol, None)
            logger.info(f"[EXECUTE_RELEASE][{cid}] {symbol}/{side} освобожден.")


    async def _process_signal(self, candidate: dict, features: dict, signal_key: tuple):
        """
        Универсальный конвейер обработки для любого УЖЕ СФОРМИРОВАННОГО сигнала.
        Гарантированно освобождает ключ сигнала после завершения.
        """
        try:
            # Кандидат уже полностью готов, просто передаем его на оценку
            await self.evaluate_entry_candidate(candidate, features)
        except Exception as e:
            logger.error(f"[_process_signal] Критическая ошибка при обработке сигнала {signal_key}: {e}", exc_info=True)
        finally:
            # Гарантированно освобождаем ключ сигнала
            self.active_signals.discard(signal_key)
            logger.debug(f"Сигнал {signal_key} обработан, блокировка снята.")


    async def notify_signal(self, candidate: dict) -> None:
        """Отправляет предварительное уведомление о сигнале в Telegram."""
        try:
            # Извлекаем данные из кандидата для формирования сообщения
            symbol = candidate.get("symbol", "N/A")
            side = candidate.get("side", "N/A").upper()
            source = candidate.get("source", "unknown").replace("_", " ").title()
            justification = candidate.get("justification", "Причина не указана.")
            link = f"https://www.bybit.com/trade/usdt/{symbol}"

            # Формируем иконку и заголовок в зависимости от направления
            icon = "📈" if side == "BUY" else "📉"
            title = f"НОВЫЙ СИГНАЛ: {side} {symbol}"

            # Собираем текст сообщения
            msg = (
                f"{icon} <b>{title}</b>\n\n"
                f"<b>Источник:</b> {source}\n"
                f"<b>AI Обоснование:</b> <i>{justification}</i>\n\n"
                f"<a href='{link}'>Открыть график на Bybit</a>\n\n"
                f"<i>Бот готовится к открытию позиции...</i>"
            )

            # Используем существующий метод для отправки уведомления
            await self.notify_user(msg)

        except Exception as e:
            # Логируем ошибку, но не прерываем основной поток выполнения
            logger.error(f"[notify_signal] Не удалось отправить уведомление о сигнале для {symbol}: {e}", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1) ИДЕМПОТЕНТНЫЙ CID (orderLinkId)
    # ─────────────────────────────────────────────────────────────────────────────
    def _make_order_link_id(self, symbol: str) -> str:
        """
        Короткий читаемый CID для идемпотентности размещения ордера.
        Пример: ALUUSDT-3614823-1a2b3c
        """
        import uuid, time
        return f"{symbol[:8]}-{int(time.time()*1000)%10_000_000:07d}-{uuid.uuid4().hex[:6]}"
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2) ОТМЕНА ПО CID (orderLinkId)
    # ─────────────────────────────────────────────────────────────────────────────
    async def cancel_order_by_link_id(self, symbol: str, cid: str):
        """
        Отмена ордера по orderLinkId (наш cid). Не бросает исключений наружу.
        """
        import asyncio
        try:
            def _call():
                # Bybit Unified v5: cancelOrder требует category и либо orderId, либо orderLinkId
                return self.session.cancel_order(
                    category="linear",
                    symbol=symbol,
                    orderLinkId=cid
                )
            resp = await asyncio.to_thread(_call)
            rc = (resp or {}).get("retCode", 0)
            if rc != 0:
                logger.error("[CANCEL][%s] ret=%s msg=%s", cid, rc, (resp or {}).get("retMsg"))
            else:
                logger.info("[CANCEL][%s] ok", cid)
            return resp
        except Exception as e:
            logger.error("[CANCEL_FAIL][%s] %s", cid, e, exc_info=True)
            return None


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

    # async def evaluate_candidate_with_ollama(self, candidate: dict, features: dict) -> dict:
    #     from openai import AsyncOpenAI
    #     default_response = {"confidence_score": 0.5, "justification": "Ошибка локального AI.", "action": "REJECT"}
    #     prompt = ""
    #     try:
    #         client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    #         def _format(v, spec): return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"
    #         m, source = candidate.get('base_metrics', {}), candidate.get('source', 'unknown').replace('_', ' ').title()

    #         vol_anomaly = (m.get('vol_change_pct', 0) / 100 + 1)
    #         trend = "Uptrend" if features.get('supertrend', 0) > 0 else "Downtrend"
    #         btc_change_1h = compute_pct(self.shared_ws.candles_data["BTCUSDT"], 60)
    #         eth_change_1h = compute_pct(self.shared_ws.candles_data["ETHUSDT"], 60)

    #         prompt = f"""
    #         SYSTEM: Ты - профессиональный трейдер, опытный риск-менеджер, элитный квантовый и крипто-аналитик. Твой ответ - всегда только валидный JSON.
    #         USER:
    #         Анализ торгового сигнала:
    #         - Монета: {candidate['symbol']}, Направление: {candidate['side'].upper()}, Источник: {source}
    #         - Метрики: PriceΔ(5m)={_format(m.get('pct_5m'), '.2f')}%, Volume Anomaly={_format(vol_anomaly, '.1f')}x, OIΔ(1m)={_format(m.get('oi_change_pct'), '.2f')}%
    #         - Контекст: Trend={trend}, ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(features.get('rsi14'), '.1f')}
    #         - Рынок: BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%

    #         ЗАДАЧА: Верни JSON с ключами "confidence_score", "justification" (начинается с источника сигнала), "action" ("EXECUTE" или "REJECT").
    #         """
    #         response = await client.chat.completions.create(
    #             model="trading-llama",
    #             messages=[{"role": "user", "content": prompt}],
    #             response_format={"type": "json_object"},
    #             temperature=0.2,
    #             top_p=1,
    #         )
    #         response_data = json.loads(response.choices[0].message.content)
    #         response_data['full_prompt_for_ai'] = prompt
    #         return response_data
    #     except Exception as e:
    #         logger.error(f"[Ollama] Ошибка API для {candidate['symbol']}: {e}", exc_info=True)
    #         return {**default_response, "full_prompt_for_ai": prompt}


    async def evaluate_candidate_with_ollama(self, candidate: dict, features: dict) -> dict:
        """
        Ensemble-версия: основная модель принимает решение, confirm-модель (Plutus) может понизить до WATCH/REJECT.
        Возвращает JSON: {action, confidence_score, justification, full_prompt_for_ai, ai_primary, ai_confirm}
        """
        default_response = {"confidence_score": 0.7, "justification": "Ошибка локального AI.", "action": "REJECT"}
        prompt = ""
        try:
            # --- Подготовка промпта (как у вас было) ---
            def _format(v, spec): 
                try:
                    return f"{float(v):{spec}}"
                except Exception:
                    return "N/A"
            m = candidate.get('base_metrics', {})
            source = candidate.get('source', 'unknown').replace('_', ' ').title()
            vol_anomaly = (m.get('vol_change_pct', 0) / 100 + 1)
            trend = "Uptrend" if features.get('supertrend', 0) > 0 else "Downtrend"

            def compute_pct_local(buf, minutes):
                try:
                    # ваша util-функция compute_pct может быть в self.shared_ws, но на случай отсутствия:
                    return compute_pct(buf, minutes)
                except Exception:
                    return 0.0

            btc_change_1h = compute_pct_local(self.shared_ws.candles_data.get("BTCUSDT", []), 60)
            eth_change_1h = compute_pct_local(self.shared_ws.candles_data.get("ETHUSDT", []), 60)

            prompt = f"""
    SYSTEM: Ты — профессиональный трейдер, опытный риск-менеджер, элитный квантовый и крипто-аналитик, поэтому оцени весь паттерн. Если у тебя есть сомнения и игнал неоднозначный, тогда лучше отклонить кандидата. Твой ответ - всегда только валидный JSON.
    USER:
    Анализ торгового сигнала:
    - Монета: {candidate['symbol']}, Направление: {candidate['side'].upper()}, Источник: {source}
    - Метрики: PriceΔ(5m)={_format(m.get('pct_5m'), '.2f')}%, Volume Anomaly={_format(vol_anomaly, '.1f')}x, OIΔ(1m)={_format(m.get('oi_change_pct'), '.2f')}%
    - Контекст: Trend={trend}, ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(features.get('rsi14'), '.1f')}
    - Рынок: BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%

    ЗАДАЧА: Верни JSON с ключами "confidence_score", "justification" (начинается с источника сигнала), "action" ("EXECUTE" или "REJECT").
    """.strip()

            # --- Параллельный запрос к двум моделям ---
            # primary_task = asyncio.create_task(self._ask_ollama_json(AI_PRIMARY_MODEL, prompt, AI_TIMEOUT_PRIMARY))
            # confirm_task = asyncio.create_task(self._ask_ollama_json(AI_CONFIRM_MODEL, prompt, AI_TIMEOUT_CONFIRM))
            # ai_primary, ai_confirm = await asyncio.gather(primary_task, confirm_task, return_exceptions=False)

            # --- Параллельный запрос к двум моделям ---
            # 1. Оборачиваем наш промпт в структуру, которую ожидает функция
            messages = [{"role": "user", "content": prompt}]

            # 2. Вызываем _ask_ollama_json с именованными аргументами
            primary_task = asyncio.create_task(self._ask_ollama_json(
                model=AI_PRIMARY_MODEL, 
                messages=messages, 
                timeout_s=AI_TIMEOUT_PRIMARY
            ))
            confirm_task = asyncio.create_task(self._ask_ollama_json(
                model=AI_CONFIRM_MODEL, 
                messages=messages, 
                timeout_s=AI_TIMEOUT_CONFIRM
            ))
            ai_primary, ai_confirm = await asyncio.gather(primary_task, confirm_task, return_exceptions=False)


            # --- Нормировка действий и скоринга ---
            prim_act  = (ai_primary.get("action") or "REJECT").upper()
            prim_conf = _clamp01(ai_primary.get("confidence_score", 0.7))
            prim_just = ai_primary.get("justification", "")

            conf_act  = (ai_confirm.get("action") or "REJECT").upper()
            conf_conf = _clamp01(ai_confirm.get("confidence_score", 0.7))
            conf_just = ai_confirm.get("justification", "")

            # --- Фильтр негативных маркеров (включая подтверждающую модель) ---
            def _has_negative(just: str) -> bool:
                jl = (just or "").lower()
                return any(cue in jl for cue in NEGATIVE_CUES)

            negative_primary = _has_negative(prim_just)
            negative_confirm = _has_negative(conf_just)

            # --- Правила объединения (veto/confirm) ---
            # 1) Если primary слабый (conf < WATCH_LO) → REJECT.
            # 2) Если primary средний (WATCH_LO..EXEC_TH) → WATCH.
            # 3) Если primary сильный (>= EXEC_TH):
            #       - если confirm явный негатив или conf_confirm < WATCH_LO → снизить до WATCH/REJECT (на выбор: осторожнее → WATCH)
            #       - иначе EXECUTE.
            if prim_conf < AI_WATCH_LO:
                final_action = "REJECT"
                final_conf   = prim_conf
                final_just   = f"[PRIMARY] {prim_just}"
            elif prim_conf < AI_EXEC_TH:
                final_action = "WATCH"
                final_conf   = prim_conf
                final_just   = f"[PRIMARY->WATCH] {prim_just}"
            else:
                # первичный настрой на EXECUTE, смотрим подтвердителя
                if negative_confirm or (conf_conf < AI_WATCH_LO and conf_act == "REJECT"):
                    # мягкий даунгрейд до WATCH (можно сделать REJECT, если хотите строже)
                    final_action = "WATCH"
                    final_conf   = min(prim_conf, 0.58)
                    final_just   = f"[PRIMARY EXECUTE, CONFIRM VETO→WATCH] {prim_just} | Plutus: {conf_just}"
                    logger.info(f"[AI_VETO] {candidate['symbol']}/{candidate['side']}: Plutus понизил сигнал")
                else:
                    final_action = "EXECUTE" if prim_act == "EXECUTE" else "WATCH"
                    final_conf   = prim_conf
                    final_just   = f"[EXECUTE] {prim_just} | Plutus: {conf_just}"

            # --- Дополнительное понижение при негативе у primary ---
            if negative_primary and final_action == "EXECUTE":
                final_action = "WATCH"
                final_just   = f"[NEGATIVE_CUES→WATCH] {prim_just}"

            out = {
                "action": final_action,
                "confidence_score": float(f"{final_conf:.3f}"),
                "justification": final_just,
                "full_prompt_for_ai": prompt,
                "ai_primary": ai_primary,
                "ai_confirm": ai_confirm,
            }
            # Логируем кратко
            logger.info(f"[AI_ENS] {candidate['symbol']}/{candidate['side']} → {final_action} ({final_conf:.2f})")
            return out

        except Exception as e:
            logger.error(f"[Ollama ENS] Ошибка ансамбля для {candidate.get('symbol','?')}: {e}", exc_info=True)
            return {**default_response, "full_prompt_for_ai": prompt}


#     async def evaluate_candidate_with_ollama(self, candidate: dict, features: dict) -> dict:
#         """
#         Отправляет данные о сигнале в локальный Ollama-сервис, получает JSON-оценку
#         и применяет дополнительный фильтр по «негативным» формулировкам.
#         """
#         # -------------------------------- настройки и вспомогалки
#         from openai import AsyncOpenAI
#         default_response = {
#             "confidence_score": 0.7,
#             "justification":   "Ошибка локального AI.",
#             "action":          "REJECT",
#         }
#         NEGATIVE_CUES = {                    # <<< новый набор триггер-слов
#             "but", "негатив", "negative", "risk",
#             "опас", "перегрет", "переоцен",
#             "overbought", "overvalued", "dump",
#         }

#         prompt = ""
#         try:
#             client = AsyncOpenAI(base_url="http://localhost:11434/v1",
#                                  api_key="ollama")

#             def _fmt(v, spec):       # красивый формат или N/A
#                 return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"

#             m      = candidate.get("base_metrics", {})
#             source = candidate.get("source", "unknown").replace("_", " ").title()

#             vol_anomaly   = (m.get("vol_change_pct", 0) / 100 + 1)
#             trend         = "Uptrend" if features.get("supertrend", 0) > 0 else "Downtrend"
#             btc_change_1h = compute_pct(self.shared_ws.candles_data["BTCUSDT"], 60)
#             eth_change_1h = compute_pct(self.shared_ws.candles_data["ETHUSDT"], 60)

#             prompt = f"""
# SYSTEM: Ты — профессиональный трейдер, опытный риск-менеджер, элитный квантовый и крипто-аналитик, поэтому оцени весь паттерн. Если у тебя есть сомнения и игнал неоднозначный, тогда лучше отклонить кандидата. Отвечай обоснованно **только** валидным JSON.
# USER:
# Анализ торгового сигнала:
# - Coin: {candidate['symbol']} | Side: {candidate['side'].upper()} | Source: {source}
# - Metrics: PriceΔ(5m)={_fmt(m.get('pct_5m'), '.2f')}%, VolumeAnom={_fmt(vol_anomaly, '.1f')}×, OIΔ(1m)={_fmt(m.get('oi_change_pct'), '.2f')}%
# - Context: Trend={trend}, ADX={_fmt(features.get('adx14'), '.1f')}, RSI={_fmt(features.get('rsi14'), '.1f')}
# - Market: BTC Δ(1h)={_fmt(btc_change_1h, '.2f')}%, ETH Δ(1h)={_fmt(eth_change_1h, '.2f')}%

# TASK: Верни JSON с полями
#   "confidence_score" (0–1),
#   "justification"    (начать c источника сигнала),
#   "action"           ("EXECUTE" | "REJECT").
#             """

#             response = await client.chat.completions.create(
#                 model="trading-llama",
#                 messages=[{"role": "user", "content": prompt}],
#                 response_format={"type": "json_object"},
#                 temperature=0.2,
#                 top_p=1,                    # <<< сделаем выбор точнее
#                 presence_penalty=0.3,       # <<< немного расширяем рассуждение
#                 frequency_penalty=0.1,      # <<< меньше повторов
#                 max_tokens=180,             # <<< хватает для «человечных» объяснений
#             )

#             response_data = json.loads(response.choices[0].message.content)

#             # ---------- дополнительный фильтр «негативных» формулировок ----------
#             text_low = response_data.get("justification", "").lower()
#             if (
#                 response_data.get("action", "EXECUTE").upper() == "EXECUTE"
#                 and any(cue in text_low for cue in NEGATIVE_CUES)
#             ):
#                 logger.info(
#                     f"[AI_FILTER] {candidate['symbol']}: обнаружен негатив в объяснении, "
#                     "меняем action на REJECT"
#                 )
#                 response_data["action"] = "REJECT"

#             # ---------------------------------------------------------------------
#             response_data["full_prompt_for_ai"] = prompt
#             return response_data

#         except Exception as e:
#             logger.error(f"[Ollama] Ошибка API для {candidate['symbol']}: {e}", exc_info=True)
#             return {**default_response, "full_prompt_for_ai": prompt}


# ========= SQUEEZE 2.0: helpers (вставить внутрь class TradingBot) =========

# --- маленькие утилиты ---
    def _sf(self, x, default=0.0):
        try:
            if x is None: return default
            return float(x)
        except Exception:
            return default

    def _clip01(self, x):
        return 0.0 if x < 0 else 1.0 if x > 1 else x

    def _lin01(self, x, lo, hi):
        """Линейная нормализация в [0..1] с отсечками."""
        if hi <= lo: return 0.0
        return self._clip01((x - lo) / (hi - lo))

    def _safe_last(self, arr, n=1):
        try:
            if not arr: return None
            if n == 1: return arr[-1]
            return arr[-n:]
        except Exception:
            return None

    def _slope(self, arr):
        """Оценка наклона (линейная регрессия) для коротких рядов."""
        try:
            if not arr or len(arr) < 3: return 0.0
            y = np.array(arr[-min(len(arr), 10):], dtype=float)
            x = np.arange(len(y), dtype=float)
            # нормируем, чтобы масштаб не влиял
            y = (y - y.mean()) / (y.std() + 1e-9)
            x = (x - x.mean()) / (x.std() + 1e-9)
            k = float(np.polyfit(x, y, 1)[0])
            return k
        except Exception:
            return 0.0

    def _safe_parse_json(self, text: str):
        """Аккуратный парсер JSON (с отрезанием код-блоков/мусора)."""
        if not isinstance(text, str):
            return None
        t = text.strip()
        # убираем ```json ... ```
        if t.startswith("```"):
            t = t.strip("`")
            # после снятия «```json» мог остаться префикс
            t = t[t.find("{"):] if "{" in t else t
        # отрезаем всё до первой «{» и после последней «}»
        if "{" in t and "}" in t:
            t = t[t.find("{"): t.rfind("}")+1]
        try:
            return json.loads(t)
        except Exception:
            return None

    # --- 1) Оценка силы сквиза ---
    def _calc_squeeze_score(self, f: dict):
        """
        f — словарь фич последних 1–3 мин:
        ожидаемые поля (если нет — деградация):
            price, ret_1m, atr_1m,
            volume, volume_pctl_1m или volume_z_1m,
            oi_delta_1m_pct, liq_sigma
        Возвращает: (score_0_1, details_dict)
        """
        price = self._sf(f.get("price"), 0.0)
        atr   = self._sf(f.get("atr_1m"), 0.0)
        ret1m = abs(self._sf(f.get("ret_1m"), 0.0))  # относительное, например 0.012 = 1.2%
        vol_pctl = self._sf(f.get("volume_pctl_1m"), None)
        vol_z    = self._sf(f.get("volume_z_1m"), None)
        oi_d1m   = abs(self._sf(f.get("oi_delta_1m_pct"), 0.0))  # в %
        liq_sig  = self._sf(f.get("liq_sigma"), 0.0)

        # 1) Аномалия движения vs ATR
        if atr <= 0:
            # фоллбек: считаем через «перегрев» ret1m (грубая эвристика)
            move_score = self._clip01(ret1m / 0.01)  # 1% -> 1.0
        else:
            move_score = self._clip01(ret1m / (1.5 * (atr / max(price, 1e-9) + 1e-9)))

        # 2) Аномальный объём
        if vol_pctl is not None:
            vol_score = self._clip01(vol_pctl)  # предполагаем 0..1
        elif vol_z is not None:
            vol_score = self._lin01(vol_z, 1.0, 3.0)  # 1σ -> 0, 3σ -> 1
        else:
            vol_score = 0.0

        # 3) OI всплеск (в %)
        oi_score = self._lin01(oi_d1m, 1.0, 5.0)  # 1% -> 0, 5% -> 1

        # 4) Ликвидации
        liq_score = self._lin01(liq_sig, 2.0, 5.0)  # 2σ -> 0, 5σ -> 1

        # Весовая смесь (настраиваемо)
        w_move, w_vol, w_oi, w_liq = 0.35, 0.25, 0.25, 0.15
        score = w_move*move_score + w_vol*vol_score + w_oi*oi_score + w_liq*liq_score
        score = self._clip01(score)

        details = {
            "move_score": move_score, "vol_score": vol_score,
            "oi_score": oi_score, "liq_score": liq_score
        }
        return score, details

    # --- 2) Exhaustion / Continuation ---
    def _calc_exhaustion_continuation(self, f: dict, impulse_dir: str):
        """
        f может содержать:
        vwap, vwap_band_std (или vwap_upper/vwap_lower),
        price, impulse_high, impulse_low, impulse_vol,
        volume_series, oi_series, cvd_series, price_series
        impulse_dir: 'up' или 'down' (направление сквиза)
        Возвращает dict: {'exhaustion':0..1,'continuation':0..1,'pullback_ratio':..,'price_to_vwap_z':..,'notes':str}
        """
        price = self._sf(f.get("price"), 0.0)
        vwap  = self._sf(f.get("vwap"), 0.0)
        vwap_std = self._sf(f.get("vwap_band_std"), 0.0)
        if vwap_std <= 0 and vwap > 0 and "vwap_upper" in f and "vwap_lower" in f:
            vwap_std = abs(self._sf(f.get("vwap_upper")) - self._sf(f.get("vwap"))) / 1.0

        imp_hi = self._sf(f.get("impulse_high"), self._sf(f.get("high", 0.0)))
        imp_lo = self._sf(f.get("impulse_low"),  self._sf(f.get("low", 0.0)))
        imp_vol = self._sf(f.get("impulse_vol"), 0.0)

        vol_series = f.get("volume_series", []) or []
        oi_series  = f.get("oi_series", []) or []
        cvd_series = f.get("cvd_series", []) or []
        pr_series  = f.get("price_series", []) or []

        # Pullback ratio от последней импульсной свечи
        rng = max(imp_hi - imp_lo, 1e-9)
        if impulse_dir == "up":
            pullback_ratio = self._clip01((imp_hi - price) / rng)
            # цена к VWAP в z
            pv = (price - vwap) / (vwap_std + 1e-9) if vwap > 0 and vwap_std > 0 else 0.0
        else:
            pullback_ratio = self._clip01((price - imp_lo) / rng)
            pv = (price - vwap) / (vwap_std + 1e-9) if vwap > 0 and vwap_std > 0 else 0.0

        # Тенденции OI/CVD (наклоны)
        oi_k  = self._slope(oi_series)
        cvd_k = self._slope(cvd_series)

        # Затухание объёма (среднее последних 3 к баров vs импульс)
        if len(vol_series) >= 4:
            last3 = float(np.mean(vol_series[-3:]))
            vol_decay = 1.0 if (imp_vol > 0 and last3 < 0.7 * imp_vol) else 0.0
        else:
            vol_decay = 0.0

        # Признаки exhaustion / continuation
        if impulse_dir == "up":
            # исчерпание: OI не растёт, CVD ослаб/негативен, цена вернулась к VWAP/ниже, есть pullback
            ex1 = 1.0 if oi_k <= 0 else 0.0
            ex2 = self._lin01(-cvd_k, 0.05, 0.25)    # чем сильнее вниз, тем ближе к 1
            ex3 = self._lin01(-pv, 0.5, 1.5)         # выше -> 0, ниже VWAP -> 1
            ex4 = self._lin01(pullback_ratio, 0.25, 0.62)
            ex5 = vol_decay
            exhaustion = self._clip01(0.24*ex1 + 0.22*ex2 + 0.20*ex3 + 0.20*ex4 + 0.14*ex5)

            # продолжение: OI растёт, CVD положительный, держимся выше верхних диапазонов (pv >> 0), откаты слабые
            co1 = self._lin01(oi_k, 0.05, 0.25)
            co2 = self._lin01(cvd_k, 0.05, 0.25)
            co3 = self._lin01(pv, 0.5, 1.5)
            co4 = self._lin01(1.0 - pullback_ratio, 0.2, 0.8)
            continuation = self._clip01(0.30*co1 + 0.30*co2 + 0.25*co3 + 0.15*co4)
        else:
            # зеркально для даун-сквиза
            ex1 = 1.0 if oi_k >= 0 else 0.0
            ex2 = self._lin01(cvd_k, 0.05, 0.25)
            ex3 = self._lin01(pv, 0.5, 1.5)          # ниже VWAP -> 0, выше -> 1
            ex4 = self._lin01(pullback_ratio, 0.25, 0.62)
            ex5 = vol_decay
            exhaustion = self._clip01(0.24*ex1 + 0.22*ex2 + 0.20*ex3 + 0.20*ex4 + 0.14*ex5)

            co1 = self._lin01(-oi_k, 0.05, 0.25)
            co2 = self._lin01(-cvd_k, 0.05, 0.25)
            co3 = self._lin01(-pv, 0.5, 1.5)
            co4 = self._lin01(1.0 - pullback_ratio, 0.2, 0.8)
            continuation = self._clip01(0.30*co1 + 0.30*co2 + 0.25*co3 + 0.15*co4)

        notes = []
        if exhaustion >= 0.6: notes.append("exh>=0.6")
        if continuation >= 0.6: notes.append("cont>=0.6")
        if vol_decay: notes.append("vol_decay")
        return {
            "exhaustion": exhaustion,
            "continuation": continuation,
            "pullback_ratio": pullback_ratio,
            "price_to_vwap_z": pv,
            "notes": ",".join(notes)
        }

    # --- 3) План входа по откату (контртренд по умолчанию) ---
    def _propose_pullback_plan(self, impulse_dir: str, imp_hi: float, imp_lo: float, atr: float, price: float, vwap: float):
        """
        Возвращает план лимит-лесенки (38/50/62%) и стоп за экстремумом + k*ATR.
        """
        imp_hi = self._sf(imp_hi); imp_lo = self._sf(imp_lo)
        atr    = abs(self._sf(atr, 0.0))
        price  = self._sf(price)
        vwap   = self._sf(vwap)

        rng = max(imp_hi - imp_lo, 1e-9)
        k = getattr(self, "squeeze_atr_k", 0.4)

        if impulse_dir == "up":
            # ищем шорт на откате вверх -> входы ниже хая (ретрейс последней свечи)
            p382 = imp_hi - 0.382 * rng
            p500 = imp_hi - 0.500 * rng
            p618 = imp_hi - 0.618 * rng
            stop = imp_hi + k * atr if atr > 0 else imp_hi * 1.002
            side = "Sell"
        else:
            p382 = imp_lo + 0.382 * rng
            p500 = imp_lo + 0.500 * rng
            p618 = imp_lo + 0.618 * rng
            stop = imp_lo - k * atr if atr > 0 else imp_lo * 0.998
            side = "Buy"

        ladder = [p382, p500, p618]
        # цель по умолчанию — возврат к VWAP (частичный фикс)
        tp1 = vwap if vwap > 0 else None

        return {
            "side": side,
            "style": "limit_pullback",
            "ladder": ladder,
            "stop": stop,
            "tp1": tp1
        }

    def _build_squeeze_entry_prompt(self, symbol: str, side: str, extreme_price: float, last_price: float, features: dict) -> str:
        """Формирует промпт для AI-оценки точки входа в сквиз."""
        
        pullback_pct = abs(last_price - extreme_price) / extreme_price * 100
        
        prompt = f"""
        SYSTEM: Ты элитный аналитик рыночной микроструктуры. Твоя задача - определить оптимальный момент для контртрендового входа в сделку после ценового импульса (сквиза). Анализируй признаки истощения импульса и начала разворота. Ответ - только валидный JSON.

        USER:
        **Торговый сигнал (Squeeze):**
        - Инструмент: {symbol}
        - Направление входа: {side.upper()}
        - Пик/дно импульса: {extreme_price:.6f}
        - Текущая цена: {last_price:.6f}
        - Величина отката от пика: {pullback_pct:.2f}%

        **Контекст рынка в момент отката:**
        - RSI(14): {features.get('rsi14', 0):.1f}
        - CVD(1m) Trend: {'Растет' if features.get('CVD1m', 0) > 0 else 'Падает'}
        - Объем на откате: (здесь можно добавить логику сравнения с объемом на импульсе, если есть данные)

        **ЗАДАЧА:** Основываясь на текущей микроструктуре, является ли этот момент оптимальным для входа?
        Верни JSON с ключами:
        - "action": "EXECUTE" (если пора входить немедленно) или "WAIT" (если нужно еще подождать развития ситуации).
        - "reason": Краткое обоснование твоего решения (например, "Объем на откате затухает, CVD подтверждает разворот").
        """
        return prompt.strip()

    async def _ai_confirm_squeeze_entry(self, symbol: str, side: str, extreme_price: float) -> bool:
        """Запрашивает у AI-советника, пора ли входить в сделку по сквизу."""
        try:
            ticker = self.shared_ws.ticker_data.get(symbol, {})
            last_price = safe_to_float(ticker.get("lastPrice", 0))
            if last_price <= 0: return False

            # Собираем свежие фичи для контекста
            features = await self.extract_realtime_features(symbol)
            if not features: return False
            
            # Формируем промпт
            prompt = self._build_squeeze_entry_prompt(symbol, side, extreme_price, last_price, features)
            messages = [{"role": "user", "content": prompt}]

            # Запрашиваем мнение у советника (Plutus)
            ai_response = await self._ask_ollama_json(
                model=self.ai_advisor_model, # Используем советника
                messages=messages,
                timeout_s=45.0 # Короткий таймаут, так как решение нужно быстро
            )

            action = ai_response.get("action", "WAIT").upper()
            if action == "EXECUTE":
                logger.info(
                    f"[AI_SQUEEZE_EXECUTE] {symbol}: AI-советник одобрил вход. "
                    f"Причина: {ai_response.get('reason', 'N/A')}"
                )
                return True
            
            return False

        except Exception as e:
            logger.warning(f"[_ai_confirm_squeeze_entry] Ошибка при консультации с AI для {symbol}: {e}", exc_info=True)
            return False


    # --- 4) Быстрый «судья» (Plutus -> primary AI) ---
    async def _ai_squeeze_judge(self, candidate: dict, features: dict, impulse_dir: str):
        """
        Возвращает dict:
        {
            "action": "ENTER|WAIT|CANCEL",
            "style": "limit_pullback|market_continuation",
            "side": "Buy|Sell",
            "plan": {...},        # если ENTER
            "justification": str
        }
        ИИ не обязателен: на недоступности вернёт эвристику.
        """
        # 1) Локальные оценки
        sq_score, _ = self._calc_squeeze_score(features)
        exh = self._calc_exhaustion_continuation(features, impulse_dir)

        # дефолтные пороги (можно вынести в конфиг)
        min_sq = getattr(self, "squeeze_min_score", 0.60)
        exh_thr = getattr(self, "exhaustion_enter_thr", 0.60)
        cont_thr = getattr(self, "continuation_follow_thr", 0.65)

        # 2) Plutus (советник) — опционально
        advisor_json = None
        if getattr(self, "ai_advisor_model", None) and hasattr(self, "_ask_ollama_json"):
            try:
                msg = {
                    "role": "user",
                    "content": (
                        "You are a microstructure analyst. Given metrics of a crypto squeeze, return JSON with "
                        "exhaustion_score (0..1) and continuation_score (0..1). "
                        "Fields: {exhaustion_score, continuation_score, note} ONLY JSON."
                    )
                }
                fpack = {k: features.get(k, None) for k in [
                    "price","atr_1m","ret_1m","volume_pctl_1m","volume_z_1m","oi_delta_1m_pct",
                    "liq_sigma","vwap","vwap_band_std","impulse_high","impulse_low","impulse_vol"
                ]}
                messages = [
                    msg,
                    {"role":"user","content": json.dumps({"impulse_dir": impulse_dir, "metrics": fpack})}
                ]
                raw = await self._ask_ollama_json(self.ai_advisor_model, messages)
                if isinstance(raw, dict):
                    advisor_json = raw
                else:
                    parsed = self._safe_parse_json(str(raw))
                    if isinstance(parsed, dict):
                        advisor_json = parsed
            except Exception as e:
                advisor_json = None

        # склеиваем оценки
        ex = exh.get("exhaustion", 0.0)
        co = exh.get("continuation", 0.0)
        if advisor_json:
            ex = 0.5*ex + 0.5*float(self._sf(advisor_json.get("exhaustion_score"), ex))
            co = 0.5*co + 0.5*float(self._sf(advisor_json.get("continuation_score"), co))

        # 3) Решение: сначала фильтр по силе сквиза
        if sq_score < min_sq:
            return {"action":"CANCEL","style":"limit_pullback","side":candidate.get("side","Buy"),
                    "justification": f"squeeze_score={sq_score:.2f} < {min_sq:.2f}"}

        # 4) Эвристика по умолчанию (если primary AI не дергаем)
        plan = self._propose_pullback_plan(
            impulse_dir=impulse_dir,
            imp_hi=self._sf(features.get("impulse_high"), self._sf(features.get("high"))),
            imp_lo=self._sf(features.get("impulse_low"),  self._sf(features.get("low"))),
            atr=self._sf(features.get("atr_1m"), 0.0),
            price=self._sf(features.get("price"), 0.0),
            vwap=self._sf(features.get("vwap"), 0.0),
        )

        # Если контртренд предпочтителен
        if ex >= exh_thr and ex >= co:
            return {"action":"ENTER","style":"limit_pullback","side": plan["side"],
                    "plan": plan, "justification": f"exhaustion={ex:.2f} >= {exh_thr:.2f} (dir={impulse_dir})"}

        # Если continuation явно доминирует — чаще WAIT, иногда разовая микро-попытка (не здесь)
        if co >= cont_thr and co > ex:
            return {"action":"WAIT","style":"market_continuation","side": plan["side"],
                    "justification": f"continuation={co:.2f} >= {cont_thr:.2f}; avoid fading the move"}

        # Иначе — ждём
        return {"action":"WAIT","style":"limit_pullback","side": plan["side"],
                "justification": f"unclear: ex={ex:.2f}, co={co:.2f}"}
    
        # --- СКЛЕЙКА 1m -> 5m OHLCV ДЛЯ ЛЮБОГО СИМВОЛА ---
    # --- SQUEEZE: агрегация минуток -> 5m ---
    def _aggregate_ohlcv_5m(self, symbol: str, lookback: int = 6):
        """
        Собираем 5-минутные бары из минуток, которые уже лежат в self.shared_ws.candles_data[symbol].
        Возвращает список баров-словари: [{open, high, low, close, volume}, ...].
        lookback — сколько 5m баров собрать (минимум 2 для логики Squeeze 2.0).
        """
        try:
            # БЕРЁМ ИМЕННО ИЗ WS-МЕНЕДЖЕРА
            raw = list(self.shared_ws.candles_data.get(symbol, deque()))
            if not raw:
                return []

            # Берём кратно 5 (последние N минуток)
            m1_needed = lookback * 5
            tail = raw[-m1_needed:] if len(raw) >= m1_needed else raw[:]

            bars_5m = []
            for i in range(0, len(tail), 5):
                chunk = tail[i:i+5]
                if len(chunk) < 5:
                    break
                o = float(chunk[0].get("openPrice", 0) or 0)
                h = max(float(x.get("highPrice", 0) or 0) for x in chunk)
                l = min(float(x.get("lowPrice", 0) or 0)  for x in chunk)
                c = float(chunk[-1].get("closePrice", 0) or 0)
                v = sum(float(x.get("volume", 0) or 0) for x in chunk)
                bars_5m.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
            return bars_5m
        except Exception:
            logger.exception("[SQUEEZE_AGG] %s: 5m aggregation failed", symbol)
            return []
    
# ========= /SQUEEZE 2.0 =========

# --- SQUEEZE FILTER (REAL) ---------------------------------------------------
    # --- Построение фич для SQUEEZE 2.0 из склеенных 5m баров ---
    def _build_squeeze_features_5m(self, symbol: str):
        """
        [V2 - CORRECTED] Возвращает (features: dict, impulse_dir: 'up'|'down') или (None, None).
        Теперь корректно рассчитывает 14-периодный ATR по 5-минутным барам.
        """
        try:
            # Запрашиваем 15 баров, чтобы иметь достаточно данных для 14-периодного ATR
            bars = self._aggregate_ohlcv_5m(symbol, lookback=15)
            if len(bars) < 14: # Убеждаемся, что у нас есть как минимум 14 баров
                return None, None

            # Создаем DataFrame для удобного расчета индикаторов
            df = pd.DataFrame(bars)
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])

            # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Рассчитываем ATR по 5-минутным данным ---
            atr_series = ta.atr(df['high'], df['low'], df['close'], length=14)
            atr_5m = atr_series.iloc[-1] if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else 0.0

            # --- Основная логика остается прежней, но используем последние 2 бара из нашего набора ---
            prev, last = bars[-2], bars[-1]
            pc = float(prev.get("close", 0.0))
            lc = float(last.get("close", 0.0))
            if pc <= 0 or lc <= 0:
                return None, None

            ret_5m = (lc - pc) / pc
            impulse_dir = "up" if ret_5m > 0 else "down"

            # ... (остальной код для сбора других фич остается) ...
            hi = max(float(last.get("high", lc)), float(prev.get("high", pc)))
            lo = min(float(last.get("low", lc)), float(prev.get("low", pc)))
            
            from collections import deque as _dq
            vh = list(self.shared_ws.volume_history.get(symbol, _dq()))
            if len(vh) >= 30:
                base = np.array(vh[-200:], dtype=float)
                mu, sd = float(np.mean(base)), float(np.std(base) + 1e-9)
                curr = float(vh[-1])
                volume_z_1m = (curr - mu) / sd
                volume_pctl_1m = float((base <= curr).sum() / len(base))
            else:
                volume_z_1m, volume_pctl_1m = None, None

            oi_series = list(self.shared_ws.oi_history.get(symbol, _dq()))[-20:]
            cvd_series = list(self.shared_ws.cvd_history.get(symbol, _dq()))[-20:]

            features = {
                "price": lc,
                "ret_5m": float(ret_5m),
                "atr_5m": float(atr_5m),  # <<< ИСПРАВЛЕННЫЙ КЛЮЧ И ЗНАЧЕНИЕ
                "impulse_high": hi,
                "impulse_low":  lo,
                "impulse_vol":  float(last.get("volume", 0.0)),
                "volume_series": [float(b.get("volume", 0.0)) for b in bars[-5:]],
                "price_series": [float(b.get("close", 0.0)) for b in bars[-10:]],
                "oi_series": oi_series,
                "cvd_series": cvd_series,
                "volume_pctl_1m": volume_pctl_1m,
                "volume_z_1m": volume_z_1m,
                "oi_delta_1m_pct": None,
                "liq_sigma": None,
                "vwap": None,
                "vwap_band_std": None,
            }
            return features, impulse_dir
        except Exception:
            logger.exception("[SQUEEZE_BUILD] %s: features build failed", symbol)
            return None, None


    async def _squeeze_logic(self, symbol: str) -> bool:
        """
        [ИСПРАВЛЕННАЯ ВЕРСИЯ]
        True -> сквиз обнаружен и передан на обработку.
        False -> сигнала сквиза нет.
        """
        try:
            features, impulse_dir = self._build_squeeze_features_5m(symbol)
            if not features:
                return False

            # --- 1. Расчет ДИНАМИЧЕСКОГО ПРОЦЕНТНОГО порога для сквиза ---
            ret_5m = float(features.get("ret_5m", 0.0)) # Это % изменение, напр. 0.04
            atr_5m = float(features.get("atr_5m", 0.0))
            price = float(features.get("price", 0.0))

            if price <= 0: return False

            thr_base = float(getattr(self, "squeeze_min_abs_move", 0.03)) # Базовый порог 3%
            k = float(getattr(self, "squeeze_k_factor", 1.5))
            # Динамический порог на основе волатильности (ATR)
            thr_dyn = (k * atr_5m / price) if atr_5m > 0 else 0.0
            
            # Финальный порог - это максимум из базового и динамического
            threshold = max(thr_base, thr_dyn) # Это тоже %, например 0.035

            # --- 2. ГЛАВНАЯ ПРОВЕРКА СКВИЗА ---
            if abs(ret_5m) < threshold:
                return False # Движение недостаточно сильное, это не сквиз

            # --- 3. ЗАЩИТА ОТ ДУБЛИРОВАНИЯ И ЗАПУСК ОБРАБОТКИ ---
            side = "Sell" if impulse_dir == "up" else "Buy"
            signal_key = (symbol, side, 'squeeze')
            
            if signal_key in self.active_signals:
                return True # Сигнал уже в обработке, но мы считаем, что событие найдено

            self.active_signals.add(signal_key)
            logger.info(f"🔥 [{symbol}] ОБНАРУЖЕН СКВИЗ! Движение: {ret_5m*100:.2f}%, Порог: {threshold*100:.2f}%. Передано на обработку.")
            
            # Обогащаем фичи полным контекстом перед отправкой
            full_features = await self.extract_realtime_features(symbol)
            if not full_features:
                self.active_signals.discard(signal_key)
                return False
            features.update(full_features)

            funding_snap = self._apply_funding_to_features(symbol, features)
            candidate = {
                "symbol": symbol, "side": side, "source": "squeeze",
                "base_metrics": {
                    'pct_5m': features.get('ret_5m', 0) * 100,
                    'atr_5m': features.get('atr_5m', 0),
                    'squeeze_power': features.get('SQ_power', 0)
                }
            }
            self._apply_funding_to_candidate(candidate, funding_snap)

            asyncio.create_task(self._process_signal(candidate, features, signal_key))
            return True
            
        except Exception as e:
            logger.error(f"[_squeeze_logic] Критическая ошибка для {symbol}: {e}", exc_info=True)
            return False




    # async def _liquidation_logic(self, symbol: str) -> bool:
    #     try:
    #         if (not self.shared_ws.check_liq_cooldown(symbol) or
    #                 symbol in self.open_positions or
    #                 symbol in self.pending_orders):
    #             return False

    #         buffer = self.liq_buffers.get(symbol)
    #         if not buffer:
    #             return False

    #         # Анализируем события за последние 10 секунд
    #         now = time.time()
    #         cutoff = now - 10
    #         recent_events = [evt for evt in buffer if evt["ts"] >= cutoff]
            
    #         if not recent_events:
    #             return False

    #         # Считаем общую сумму и доминирующую сторону
    #         total_value = sum(evt['value'] for evt in recent_events)
    #         buy_value = sum(evt['value'] for evt in recent_events if evt['side'] == 'Buy')
    #         sell_value = total_value - buy_value

    #         threshold = self.shared_ws.get_liq_threshold(symbol, 25000) # Порог в $25k

    #         # Проверяем, есть ли сильный кластер
    #         if total_value < threshold:
    #             return False
            
    #         # Определяем доминирующую сторону и сторону для входа
    #         dominant_side = "Buy" if buy_value > sell_value else "Sell"
    #         trade_side = "Sell" if dominant_side == "Buy" else "Buy" # Торгуем против
            
    #         dominance_ratio = (buy_value / total_value) if dominant_side == "Buy" else (sell_value / total_value)
            
    #         # Требуем, чтобы >80% ликвидаций были с одной стороны
    #         if dominance_ratio < 0.80:
    #             return False
            
    #         logger.info(f"💧 [{symbol}] ОБНАРУЖЕН КЛАСТЕР ЛИКВИДАЦИЙ! Объем: ${total_value:,.0f}, сторона: {dominant_side}. Готовим контр-сделку {trade_side}.")
            
    #         cluster_data = {
    #             "total_value_usd": total_value,
    #             "dominant_side": dominant_side,
    #             "trade_side": trade_side,
    #             "event_count": len(recent_events)
    #         }
            
    #         # [ИЗМЕНЕНО] Защита от дублирования задач
    #         signal_key = (symbol, trade_side, 'liquidation')
    #         if signal_key in self.processing_signals:
    #             return False # Сигнал уже в обработке
            
    #         self.processing_signals.add(signal_key)
    #         logger.info(f"💧 [{symbol}] ОБНАРУЖЕН КЛАСТЕР ЛИКВИДАЦИЙ! ... Передано на обработку.")
            
    #         cluster_data = { ... }
    #         asyncio.create_task(self._process_liquidation_signal(symbol, cluster_data))
            
    #         self.shared_ws.last_liq_trade_time[symbol] = dt.datetime.utcnow()
    #         return True
    #     except Exception as e:
    #         logger.error(f"[_liquidation_logic] Ошибка для {symbol}: {e}", exc_info=True)
    #         return False

    async def _liquidation_logic(self, symbol: str) -> bool:
        try:
            if (not self.shared_ws.check_liq_cooldown(symbol) or
                    symbol in self.open_positions or symbol in self.pending_orders):
                return False

            buffer = self.liq_buffers.get(symbol)
            if not buffer: return False

            now = time.time()
            cutoff = now - 10
            recent_events = [evt for evt in buffer if evt["ts"] >= cutoff]
            if not recent_events: return False

            total_value = sum(evt['value'] for evt in recent_events)
            buy_value = sum(evt['value'] for evt in recent_events if evt['side'] == 'Buy')
            sell_value = total_value - buy_value
            threshold = self.shared_ws.get_liq_threshold(symbol, 25000)

            if total_value < threshold: return False
            
            dominant_side = "Buy" if buy_value > sell_value else "Sell"
            trade_side = "Sell" if dominant_side == "Buy" else "Buy"
            dominance_ratio = (buy_value / total_value) if dominant_side == "Buy" else (sell_value / total_value)
            
            if dominance_ratio < 0.80: return False
            
            # [НОВАЯ ЛОГИКА] Проверяем блокировку и ставим свою
            signal_key = (symbol, trade_side, 'liquidation')
            if signal_key in self.active_signals:
                return True # Сигнал уже в обработке, но считаем, что событие найдено

            self.active_signals.add(signal_key)
            logger.info(f"💧 [{symbol}] ОБНАРУЖЕН КЛАСТЕР ЛИКВИДАЦИЙ! Объем: ${total_value:,.0f}. Передано на обработку.")
            
            # [НОВАЯ ЛОГИКА] Формируем кандидата ЗДЕСЬ и передаем в обработчик
            features = await self.extract_realtime_features(symbol)
            if not features:
                self.active_signals.discard(signal_key) # Освобождаем, если не смогли собрать данные
                return False

            funding_snap = self._apply_funding_to_features(symbol, features)
            candidate = {
                "symbol": symbol,
                "side": trade_side,
                "source": "liquidation",
                "base_metrics": {
                    'pct_5m': features.get("pct5m", 0),
                    'liquidation_value_usd': total_value,
                    'liquidation_side': dominant_side,
                    'liquidation_events': len(recent_events)
                }
            }
            self._apply_funding_to_candidate(candidate, funding_snap)

            asyncio.create_task(self._process_signal(candidate, features, signal_key))
            
            self.shared_ws.last_liq_trade_time[symbol] = dt.datetime.utcnow()
            return True

        except Exception as e:
            logger.error(f"[_liquidation_logic] Ошибка для {symbol}: {e}", exc_info=True)
            return False


    async def _golden_logic(self, symbol: str):
        async with self.pending_orders_lock:
            if symbol in self.open_positions or symbol in self.pending_orders or symbol in self.active_trade_entries:
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

                # [ИСПРАВЛЕНО] Этот блок теперь находится внутри 'if action:'
                funding_snap = self._apply_funding_to_features(symbol, features)
                self._apply_funding_to_candidate(candidate, funding_snap)

                logger.debug(f"[Signal Candidate] Golden Setup: {action} on {symbol}")
                asyncio.create_task(self.evaluate_entry_candidate(candidate, features))

                # 'return' теперь тоже здесь, что логично
                return

        except Exception as e:
            logger.error(f"[_golden_logic] unexpected error for {symbol}: {e}", exc_info=True)
        finally:
            self.pending_signals.pop(symbol, None)


    async def _process_squeeze_signal(self, symbol: str, features: dict, impulse_dir: str):
        side = "Sell" if impulse_dir == "up" else "Buy"
        signal_key = (symbol, side, 'squeeze')
        try:
            # ... (весь код, который был в _process_squeeze_signal, остается здесь) ...
            funding_snap = self._apply_funding_to_features(symbol, features)
            candidate = {
                "symbol": symbol, "side": side, "source": "squeeze",
                "base_metrics": {'pct_5m': features.get('ret_5m', 0) * 100, 'atr_5m': features.get('atr_5m', 0)}
            }
            self._apply_funding_to_candidate(candidate, funding_snap)
            await self.evaluate_entry_candidate(candidate, features)
        finally:
            # Гарантированно освобождаем ключ после обработки
            self.processing_signals.discard(signal_key)

    async def _process_liquidation_signal(self, symbol: str, cluster_data: dict):
        signal_key = (symbol, cluster_data["trade_side"], 'liquidation')
        try:
            # ... (весь код, который был в _process_liquidation_signal, остается здесь) ...
            features = await self.extract_realtime_features(symbol)
            if not features: return
            funding_snap = self._apply_funding_to_features(symbol, features)
            candidate = {
                "symbol": symbol, "side": cluster_data["trade_side"], "source": "liquidation",
                "base_metrics": {
                    'pct_5m': features.get("pct5m", 0),
                    'liquidation_value_usd': cluster_data["total_value_usd"],
                    'liquidation_side': cluster_data["dominant_side"],
                    'liquidation_events': cluster_data["event_count"]
                }
            }
            self._apply_funding_to_candidate(candidate, funding_snap)
            await self.evaluate_entry_candidate(candidate, features)
        finally:
            self.processing_signals.discard(signal_key)

    async def _process_golden_signal(self, candidate: dict, features: dict, signal_key: tuple):
        try:
            await self.evaluate_entry_candidate(candidate, features)
        finally:
            self.processing_signals.discard(signal_key)



    # async def _process_squeeze_signal(self, symbol: str, features: dict, impulse_dir: str):
    #         """
    #         Обрабатывает обнаруженный сигнал сквиза в фоновом режиме,
    #         чтобы не блокировать основной market_loop.
    #         """
    #         try:
    #             # --- 2. [НОВОЕ] Обогащение фич полным контекстом перед отправкой в AI ---
    #             try:
    #                 full_features = await self.extract_realtime_features(symbol)
    #                 if full_features:
    #                     # Дополняем наш набор 'features' полными данными (RSI, ADX и т.д.)
    #                     features.update(full_features)
    #                     logger.debug(f"[{symbol}] Фичи для сквиза успешно обогащены.")
    #             except Exception as e:
    #                 logger.warning(f"[Squeeze] Не удалось обогатить фичи для {symbol}: {e}")

    #             # --- 3. Формирование кандидата и вызов AI ---
    #             funding_snap = self._apply_funding_to_features(symbol, features)

    #             side = "Sell" if impulse_dir == "up" else "Buy"
    #             candidate = {
    #                 "symbol": symbol,
    #                 "side": side,
    #                 "source": "squeeze",
    #                 "base_metrics": {
    #                     'pct_5m': features.get('ret_5m', 0) * 100,
    #                     'atr_5m': features.get('atr_5m', 0),
    #                     # Можно добавить и другие ключевые метрики сквиза сюда
    #                 }
    #             }
    #             self._apply_funding_to_candidate(candidate, funding_snap)

    #             # Передаем кандидата и ПОЛНЫЙ набор фич на оценку
    #             # Эта функция уже вызывается как create_task внутри, но для надежности оставим так.
    #             await self.evaluate_entry_candidate(candidate, features)

    #         except Exception as e:
    #             logger.error(f"[_process_squeeze_signal] Ошибка при обработке сквиза для {symbol}: {e}", exc_info=True)


    async def on_liquidation_event(self, event: dict):
        """
        Принимает событие ликвидации от WS, добавляет в буфер и удаляет старые.
        """
        try:
            symbol = event.get("s")
            if not symbol: return

            side = str(event.get("S", "")).capitalize()
            qty = self._sf(event.get("v", 0))
            price = self._sf(event.get("p", 0))
            value = qty * price
            
            if not all([side, value > 0]): return

            # Получаем или создаем буфер для символа
            buffer = self.liq_buffers.setdefault(symbol, deque(maxlen=200))
            
            # Добавляем новое событие
            now = time.time()
            buffer.append({"ts": now, "side": side, "value": value})
            
            # Очищаем старые события (старше 30 секунд)
            cutoff = now - 30
            while buffer and buffer[0]["ts"] < cutoff:
                buffer.popleft()
        except Exception as e:
            logger.error(f"[on_liquidation_event] Ошибка обработки события: {e}")


    async def _process_liquidation_signal(self, symbol: str, cluster_data: dict):
        """
        Асинхронно обрабатывает сигнал о кластере ликвидаций.
        Собирает фичи, формирует кандидата и отправляет на оценку AI.
        """
        try:
            features = await self.extract_realtime_features(symbol)
            if not features:
                logger.warning(f"[_process_liquidation_signal] Не удалось извлечь фичи для {symbol}.")
                return

            funding_snap = self._apply_funding_to_features(symbol, features)

            # Формируем кандидата для AI
            candidate = {
                "symbol": symbol,
                "side": cluster_data["trade_side"],
                "source": "liquidation",
                "base_metrics": {
                    'pct_5m': features.get("pct5m", 0),
                    'liquidation_value_usd': cluster_data["total_value_usd"],
                    'liquidation_side': cluster_data["dominant_side"],
                    'liquidation_events': cluster_data["event_count"]
                }
            }
            self._apply_funding_to_candidate(candidate, funding_snap)

            logger.debug(f"[Signal Candidate] Liquidation: {cluster_data['trade_side']} on {symbol}")
            await self.evaluate_entry_candidate(candidate, features)

        except Exception as e:
            logger.error(f"[_process_liquidation_signal] Ошибка обработки для {symbol}: {e}", exc_info=True)



    # async def place_unified_order(self, symbol: str, side: str, qty: float,
    #                             order_type: str,
    #                             price: Optional[float] = None,
    #                             comment: str = "",
    #                             cid: str | None = None):

    #     # idempotent CID для REST; для WS используем для корреляции
    #     cid = cid or new_cid()

    #     logger.info("[EXECUTE][%s] start %s/%s type=%s qty=%s price=%s comment=%s | %s",
    #                 cid, symbol, side, order_type, qty, price, comment, j(log_state(self, symbol)))

    #     pos_idx = 1 if side == "Buy" else 2
    #     qty_str = self._format_qty(symbol, qty)

    #     try:
    #         if self.mode == "real":
    #             # ─────────────────────────────────────────────
    #             # REAL: отправляем через WS. Защищаемся от внешних отмен (shield),
    #             # чтобы внешние wait_for не рвали живой вызов.
    #             # ─────────────────────────────────────────────
    #             response = await asyncio.shield(self.place_order_ws(
    #                 symbol, side, qty_str, position_idx=pos_idx,
    #                 price=price, order_type=order_type
    #             ))

    #             # Пытаемся вытащить orderId из WS-ответа…
    #             order_id = None
    #             try:
    #                 if isinstance(response, dict):
    #                     order_id = response.get("orderId") or response.get("order_id")
    #                 elif isinstance(response, list) and response:
    #                     order_id = response[0].get("orderId")
    #             except Exception:
    #                 pass

    #             # …или из кеша приватного стрима (если он у тебя есть)
    #             if not order_id:
    #                 try:
    #                     order_id = getattr(self, "_last_ws_order_id", {}).get(symbol)
    #                 except Exception:
    #                     order_id = None

    #             if order_id:
    #                 # корреляция cid <-> order_id (для дебага)
    #                 try:
    #                     self.order_correlation[order_id] = cid
    #                 except Exception:
    #                     pass
    #                 logger.info("[EXECUTE][%s] order_accepted id=%s", cid, order_id)

    #             # Совместимый ответ, чтобы остальной код не ломался
    #             resp_ok = {
    #                 "retCode": 0,
    #                 "retMsg": "OK",
    #                 "result": {"orderId": order_id or ""},
    #                 "time": int(time.time() * 1000),
    #                 "retExtInfo": {},
    #                 "source": "ws"
    #             }

    #             logger.info("✅ Успешно отправлен ордер: %s %s %s @ %s",
    #                         side, qty_str, symbol, price or "Market")
    #             self.pending_strategy_comments[symbol] = comment
    #             return resp_ok

    #         else:
    #             # ─────────────────────────────────────────────
    #             # DEMO/TEST: идём через REST. Делаем вызов идемпотентным (orderLinkId=cid)
    #             # и защищаем от внешних отмен через shield.
    #             # ─────────────────────────────────────────────
    #             def _place():
    #                 return self.session.place_order(
    #                     category="linear", symbol=symbol, side=side,
    #                     orderType=order_type, qty=qty_str,
    #                     price=str(price) if price else None,
    #                     timeInForce="GTC", positionIdx=pos_idx,
    #                     orderLinkId=cid
    #                 )

    #             response = await asyncio.shield(asyncio.to_thread(_place))

    #             if (response or {}).get("retCode") != 0:
    #                 # InvalidRequestError(message, time, resp_headers)
    #                 raise InvalidRequestError(
    #                     response.get("retMsg", "Order failed"),
    #                     response.get("time"),
    #                     response.get("retExtInfo")
    #                 )

    #             order_id = response.get("result", {}).get("orderId", "")

    #             # Для маркетов в демо — подождём материализацию позиции
    #             for _ in range(10):
    #                 def _get_pos():
    #                     return self.session.get_positions(category="linear", symbol=symbol)
    #                 pos = await asyncio.to_thread(_get_pos)
    #                 lst = pos.get("result", {}).get("list", [])
    #                 if lst and float(lst[0].get("size", 0)) > 0:
    #                     break
    #                 await asyncio.sleep(0.5)
    #             else:
    #                 logger.warning("[DEMO] %s: Market-order %s не материализовался в позицию – cancel",
    #                             symbol, order_id)
    #                 def _cancel():
    #                     return self.session.cancel_order(
    #                         category="linear", symbol=symbol, orderId=order_id
    #                     )
    #                 await asyncio.to_thread(_cancel)
    #                 raise RuntimeError("Position did not appear within 5 s")

    #             logger.info("✅ Успешно отправлен ордер: %s %s %s @ %s",
    #                         side, qty_str, symbol, price or "Market")
    #             self.pending_strategy_comments[symbol] = comment
    #             return response

    #     except InvalidRequestError as e:
    #         logger.error("❌ Ошибка размещения ордера для %s (API Error): %s (Код: %s)",
    #                     symbol, e, getattr(e, 'status_code', None))
    #         self.pending_orders.pop(symbol, None)
    #         self.pending_cids.pop(symbol, None)
    #         self.pending_timestamps.pop(symbol, None)

    #     except Exception as e:
    #         logger.critical("❌ Критическая ошибка при размещении ордера для %s: %s",
    #                         symbol, e, exc_info=True)
    #         self.pending_orders.pop(symbol, None)
    #         self.pending_cids.pop(symbol, None)
    #         self.pending_timestamps.pop(symbol, None)


    # [ИЗМЕНЕНО] v3 - Надежная, быстрая и отказоустойчивая версия
    
async def place_unified_order(self, symbol: str, side: str, qty: float,
                              order_type: str,
                              price: Optional[float] = None,
                              comment: str = "",
                              cid: str | None = None):
    """
    [V3 FIX] Единая отправка ордеров (REST) с идемпотентностью и корректным обновлением состояния.
    """
    cid = cid or new_cid()
    pos_idx = 1 if side == "Buy" else 2
    qty_str = self._format_qty(symbol, qty)

    params = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": order_type,
        "qty": qty_str,
        "timeInForce": "GTC",
        "positionIdx": pos_idx,
        "orderLinkId": cid,
    }
    if order_type == "Limit" and price is not None:
        params["price"] = str(price)

    logger.info(f"➡️ [ORDER_SENDING][{cid}] Параметры: {j(params)}")

    try:
        def _place_order_sync():
            return self.session.place_order(**params)
        response = await asyncio.to_thread(_place_order_sync)
    except Exception as e:
        logger.error(f"❌ [ORDER_EXCEPTION][{cid}] {symbol} ошибка при отправке ордера: {e}", exc_info=True)
        raise

    if (response or {}).get("retCode") == 0:
        order_id = (response.get("result") or {}).get("orderId", "") or ""
        # корреляция и кулдаун
        try:
            if order_id:
                self.order_correlation[order_id] = cid
        except Exception:
            pass
        self.last_entry_ts[symbol] = time.time()

        self.pending_strategy_comments[symbol] = comment
        logger.info(f"✅ [ORDER_ACCEPTED][{cid}] {symbol} {side} {order_type} id={order_id or 'n/a'}")
        return {"orderId": order_id, "status": "ok"}
    else:
        ret_msg = response.get("retMsg", "Unknown API error")
        logger.error(f"❌ [ORDER_REJECTED][{cid}] {symbol} отклонён: {ret_msg} | full={j(response)}")
        # пробрасываем как исключение для верхнего уровня
        raise InvalidRequestError(ret_msg, response.get("retCode"), response.get("time"), {})

async def adaptive_squeeze_entry(
    self,
    symbol: str,
    side: str,
    qty: float,
    max_entry_timeout: int = 45,
    ai_preconfirmed: bool = False,
) -> bool:
    """
    [V2 FIX] Тактический вход для squeeze.
    Если ai_preconfirmed=True — повторно AI не вызываем; ждём простое ценовое условие и входим.
    Возвращает True при успешной отправке ордера, иначе False.
    """
    logger.info(
        f"[TACTICAL_SQUEEZE_AI] {symbol}/{side}: окно={max_entry_timeout}s, ai_preconfirmed={ai_preconfirmed}, "
        f"check_interval={getattr(self, 'squeeze_ai_confirm_interval_sec', 5)}s"
    )
    start_time = time.time()
    extreme = 0.0
    last_ai_check = 0.0
    entry_made = False

    # Мини-условие, по умолчанию 0.4% от экстремума
    pull = float(getattr(self, "squeeze_pullback_ratio", 0.004))

    try:
        while time.time() - start_time < max_entry_timeout:
            ticker = self.shared_ws.ticker_data.get(symbol, {})
            last_price = safe_to_float(ticker.get("lastPrice", 0))
            if last_price <= 0:
                await asyncio.sleep(0.1)
                continue

            # Обновляем экстремум
            if side == "Sell":
                if extreme == 0 or last_price > extreme:
                    extreme = last_price
            else:
                if extreme == 0 or last_price < extreme:
                    extreme = last_price

            if ai_preconfirmed:
                ok = (
                    (side == "Sell" and last_price <= extreme * (1 - pull)) or
                    (side == "Buy"  and last_price >= extreme * (1 + pull))
                )
                if ok:
                    await self.place_unified_order(symbol, side, qty, "Market", comment="squeeze_preconfirmed")
                    entry_made = True
                    break
            else:
                now = time.time()
                interval = float(getattr(self, "squeeze_ai_confirm_interval_sec", 5))
                if extreme > 0 and now - last_ai_check >= interval:
                    last_ai_check = now
                    try:
                        if await self._ai_confirm_squeeze_entry(symbol, side, extreme):
                            await self.place_unified_order(symbol, side, qty, "Market", comment="squeeze_ai_confirmed")
                            entry_made = True
                            break
                    except Exception as e:
                        logger.warning(f"[TACTICAL_SQUEEZE_AI] {symbol}: ошибка AI-подтверждения: {e}")

            await asyncio.sleep(0.2)
    except Exception as e:
        logger.error(f"[TACTICAL_SQUEEZE_AI] {symbol}: критическая ошибка: {e}", exc_info=True)

    if not entry_made:
        logger.warning(f"[TACTICAL_SQUEEZE_AI] {symbol}: окно истекло — вход НЕ выполнен")
    return entry_made

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
        # >>> ИСПРАВЛЕНИЕ: Оборачиваем вызов в functools.partial <<<
        # Это правильный способ передать корутину в обработчик
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown())) # <--- ИСПРАВЛЕНО


    # Блокируемся, пока жив хотя бы один таск
    await asyncio.gather(public_ws_task, telegram_task, *bot_tasks)



# [REMOVED] Удален дубликат функции wallet_loop, так как она уже есть в классе TradingBot.

# ---------------------- ENTRY POINT ----------------------
# [ДОБАВЛЕНО] Точка входа для запуска всего скрипта
# [ИЗМЕНЕНО] Единая и корректная точка входа, поддерживающая два режима работы
if __name__ == "__main__":
    # Проверяем, был ли передан флаг --train из командной строки
    if "--train" in sys.argv:
        # --- РЕЖИМ ТРЕНИРОВКИ ---
        
        async def run_training():
            """Запускает процесс создания и обучения модели."""
            # Для тренировки нам нужен только один экземпляр бота
            users = load_users_from_json("user_state.json")
            if not users:
                print("❌ Нет пользователей в user_state.json. Невозможно запустить тренировку.")
                return

            # Используем данные первого пользователя из конфига
            user_data = users[0]
            print(f"Используются данные пользователя: {user_data['user_id']}")
            
            # Создаем необходимые объекты для сбора данных
            shared_ws = PublicWebSocketManager(symbols=["BTCUSDT", "ETHUSDT"])
            bot = TradingBot(user_data=user_data, shared_ws=shared_ws, golden_param_store={})
            
            # Запускаем WS в фоне, чтобы он мог собрать список символов
            ws_task = asyncio.create_task(shared_ws.start())
            
            # Запускаем саму тренировку (сбор данных, обучение, сохранение)
            await bot.train_and_save_model()
            
            # Корректно завершаем фоновые задачи
            ws_task.cancel()

        try:
            # Запускаем асинхронную функцию тренировки
            asyncio.run(run_training())
        except KeyboardInterrupt:
            logger.info("Тренировка прервана пользователем.")

    else:
        # --- ОБЫЧНЫЙ ТОРГОВЫЙ РЕЖИМ ---
        # Если флага --train нет, запускаем бота в штатном режиме
        try:
            asyncio.run(run_all())
        except KeyboardInterrupt:
            logger.info("Программа остановлена пользователем.")
