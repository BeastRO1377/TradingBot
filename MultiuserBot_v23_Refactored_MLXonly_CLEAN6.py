#!/usr/bin/env python3
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
import math           # ← добавьте эту строку
import random
import signal
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_DOWN, ROUND_HALF_UP, ROUND_FLOOR

from aiolimiter import AsyncLimiter  # Правильный импорт
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

# # Конфигурация для Apple Silicon M4

# ── Compute device selection ──────────────────────────────────────────────
# Default to CPU because large workloads on MPS may seg‑fault (PyTorch <2.8).
# If you really want MPS, start the script with BOT_DEVICE=mps env var.

logger = logging.getLogger(__name__)

BOT_DEVICE = os.getenv("BOT_DEVICE", "cpu").lower()

def _probe_mps():
    """
    Returns True if we can allocate & use a tensor on MPS without crashing.
    Some PyTorch‑MPS builds segfault under load; we test with a tiny tensor.
    """

    try:
        _ = (x * 2).cpu()      # trivial op & sync back
        del x
        return True
    except Exception as exc:   # noqa: BLE001
        logger.warning("[ML] MPS probe failed: %s", exc)
        return False

if BOT_DEVICE == "mps":
        logger.warning("[ML] Falling back from MPS to CPU due to un‑usable MPS backend")
        BOT_DEVICE = "cpu"

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
SQUEEZE_LIMIT_OFFSET_PCT = 0.005   # 1 %

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

LISTING_AGE_MIN_MINUTES = 1400    # игнорируем пары младше 12 часов

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
    "price_change": 0.4,   # ≈ 99 %
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

# Load scaler
#scaler = joblib.load('scaler.pkl')

 # Global scaler placeholder; will be loaded in __main__
scaler: StandardScaler = None

# Define model class (same as training)

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
                "signal_strength",      # ← новое
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
# Lightweight float conversion helper
def safe_to_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0

# ---------------------- DECIMAL COMPATIBILITY STUBS ----------------------
class Decimal(float):
    def __new__(cls, value):
        try:
            return super().__new__(cls, float(value))
        except Exception:
            return super().__new__(cls, 0.0)
    def quantize(self, *args, **kwargs):
        # Stub quantize: return self unmodified
        return self

# Rounding constants stubs (no-ops)
ROUND_DOWN = None
ROUND_HALF_DOWN = None
ROUND_HALF_UP = None
ROUND_FLOOR = None

# Backwards-compatible Decimal converter
def safe_to_float(val) -> Decimal:
    try:
        return Decimal(val)
    except Exception:
        return Decimal(0)
    
# ───────── Trailing-stop defaults ─────────
DEFAULT_TRAILING_START_PCT = 6.0     # %-PnL, когда включать трейлинг
DEFAULT_TRAILING_GAP_PCT   = 0.75    # %-отступ стопа от цены

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
    "SQ_power", "SQ_strength",        # <<< NEW LINES
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
    # ── 1. Sanity check: at least two ATR windows of data ─────────────────
    if len(df) < (period + 1) * 2:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)

    high  = df["highPrice"].astype("float32")
    low   = df["lowPrice"].astype("float32")
    close = df["closePrice"].astype("float32")

    atr = ta.atr(high, low, close, length=period)

    # If ATR is all‑NaN (can happen on very fresh pairs) → safe fallback
    if atr.isna().all():
        return pd.Series([False] * len(df), index=df.index, dtype=bool)

    # ── 3. Core SuperTrend math ───────────────────────────────────────────
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

        # keep bands from repainting against current trend
        if in_uptrend and lowerband.iat[i] < lowerband.iat[i - 1]:
            lowerband.iat[i] = lowerband.iat[i - 1]
        if not in_uptrend and upperband.iat[i] > upperband.iat[i - 1]:
            upperband.iat[i] = upperband.iat[i - 1]

        supertrend.iat[i] = in_uptrend

    return supertrend

# ---------------------- WEBSOCKET: PUBLIC ----------------------
class PublicWebSocketManager:
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
        # dynamic list of liquid symbols
        self.active_symbols = set(symbols)
        self._last_selection_ts = time.time()
        self._callback = None
        self.ready_event = asyncio.Event()   # станет set() после первого отбора
        self.loop = asyncio.get_event_loop()
        # for golden setup
        self.volume_history = defaultdict(lambda: deque(maxlen=500))
        self.oi_history     = defaultdict(lambda: deque(maxlen=500))
        self.cvd_history    = defaultdict(lambda: deque(maxlen=500))
        self.funding_history = defaultdict(lambda: deque(maxlen=3))
        # track last saved candle time for deduplication
        self._last_saved_time = {}
        # Список ботов для оповещений об обновлениях тикера
        self.position_handlers = []
        self.latest_liquidation = {}
        # === v2 liquidation‑strategy thresholds ===
        self._liq_thresholds = defaultdict(lambda: 5000.0)   # per‑symbol dynamic threshold
        self.last_liq_trade_time = {}                        # cooldown timestamps
        # ── load pre‑computed optimal thresholds, if any ──
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
        # reference to a TradingBot instance (optional)
        self.bot = None
        # restore saved history
        self._history_file = 'history.pkl'
        try:
            with open(self._history_file, 'rb') as f:
                data = pickle.load(f)
                # restore candles, volume and oi history
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
            # нет файла или ошибка — продолжаем с чистыми данными
            pass

    async def start(self):
        """Авто-переподключение при любом исключении или закрытии."""
        while True:
            try:
                def _on_message(msg):
                    try:
                        # если цикл уже закрыт — пропускаем
                        if not self.loop.is_closed():
                            asyncio.run_coroutine_threadsafe(
                                self.route_message(msg),
                                self.loop
                            )
                    except Exception as e:
                        logger.warning(f"[PublicWS callback] loop closed, skipping message: {e}")
                # создаём новое соединение
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
                # ── подписываемся на liquidation‑события по текущему списку символов ──
                self.ws.all_liquidation_stream(
                    symbol=list(self.symbols),
                    callback=_on_message
                )

                # запуск фонового сохранения истории (один раз за жизненный цикл)
                if not hasattr(self, "_save_task"):
                    self._save_task = asyncio.create_task(self._save_loop())

                # expose callback and launch hourly resubscribe task
                self._callback = _on_message
                asyncio.create_task(self.manage_symbol_selection(check_interval=60))
                # ждём, пока ws не упадёт (блокирующий Future)
                await asyncio.Event().wait()
                self.health_task = asyncio.create_task(self.health_check())

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[PublicWS] reconnect after error: %s", e)
                await asyncio.sleep(5)  # небольшая задержка перед новой попыткой

    def get_liq_threshold(self, symbol: str, default: float = 5000.0) -> float:
        """
        Dynamic liquidation threshold.

        • 0.1 % от 24-часового оборота  
        • Фолбэк на оптимизированные per-symbol-пороги из CSV  
        • Никогда не ниже 5 000 USDT
        """
        t24 = safe_to_float(self.ticker_data.get(symbol, {}).get("turnover24h", 0))
        if t24 > 0:
            return max(5_000.0, 0.001 * t24)   # 0.1 %
        return max(15_000.0, self._liq_thresholds.get(symbol, default))

    async def manage_symbol_selection(self, min_turnover=2_000_000,
                                    min_volume=1_200_000, check_interval=3600):
        http = HTTP(testnet=False)          # ← это важно
        while True:
            await asyncio.sleep(check_interval)

            # --- REST‑снимок всех тикеров ---
            try:
                resp = await asyncio.to_thread(
                    lambda: http.get_tickers(category="linear", symbol=None)
                )
                for tk in resp["result"]["list"]:
                    self.ticker_data[tk["symbol"]] = tk
            except Exception as e:
                logger.warning("[manage_symbol_selection] REST error: %s", e)
                continue

            # select symbols that satisfy liquidity thresholds and are not too fresh
            new_set = {
                s for s, t in self.ticker_data.items()
                if safe_to_float(t.get("turnover24h", 0)) >= min_turnover
                and safe_to_float(t.get("volume24h", 0))   >= min_volume
            } or self.active_symbols  # fallback if empty

            # ── не отписываемся от символов, по которым есть открытые позиции ──
            open_pos_symbols = {s
                for bot in self.position_handlers
                for s   in bot.open_positions.keys()}
            new_set |= open_pos_symbols            # объединяем множества
            if not new_set:                        # fallback safety
                new_set = self.active_symbols

            if not self.ready_event.is_set():
                # первый проход: считаем, что данные получены, пары выбраны
                self.ready_event.set()

            if new_set != self.active_symbols:
                logger.info("[manage_symbol_selection] resubscribing: %d → %d symbols",
                            len(self.active_symbols), len(new_set))
                self.active_symbols = new_set
                # update .symbols property and back‑fill historical data
                self.symbols = list(new_set)
                # backfill only for newly added symbols
                asyncio.create_task(self.backfill_history())
                try:
                    # close old socket and open a new one with same callback
                    if self.ws:
                        self.ws.exit()
                    self.ws = WebSocket(
                        testnet=False,
                        channel_type="linear",
                        ping_interval=30,
                        ping_timeout=15,
                        restart_on_error=True,
                        retries=200
                    )
                    self.ws.kline_stream(interval=self.interval,
                                         symbol=list(new_set),
                                         callback=self._callback)
                    self.ws.ticker_stream(symbol=list(new_set),
                                          callback=self._callback)
                    self.ws.all_liquidation_stream(symbol=list(new_set),
                                                    callback=self._callback)
                except Exception as e:
                    logger.warning("[manage_symbol_selection] WS resubscribe failed: %s", e)

    async def route_message(self, msg):
        topic = msg.get("topic", "")
        if topic.startswith("kline."):
            await self.handle_kline(msg)
        elif topic.startswith("tickers."):
            await self.handle_ticker(msg)
        elif "liquidation" in topic.lower():
            # Передаём событие только ботам с разрешённой стратегией
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
        # Extract data entries, handle both list or single dict
        raw = msg.get("data")
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            # only store when candle is confirmed
            if not entry.get("confirm", False):
                continue
            symbol = msg["topic"].split(".")[-1]
            # Parse and store candle data
            try:
                ts = pd.to_datetime(int(entry["start"]), unit="ms")
            except Exception as e:
                print(f"[handle_kline] invalid start: {e}")
                continue
            # skip duplicate candle
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
            # record volume
            self.volume_history[symbol].append(row["volume"])
            # attach latest open‑interest snapshot to this confirmed candle
            oi_val = self.latest_open_interest.get(symbol, 0.0)
            self.oi_history[symbol].append(oi_val)
            # ---- CVD: cumulative volume delta -------------------------
            delta = row["volume"] if row["closePrice"] >= row["openPrice"] else -row["volume"]
            prev_cvd = self.cvd_history[symbol][-1] if self.cvd_history[symbol] else 0.0
            self.cvd_history[symbol].append(prev_cvd + delta)
            #self._save_history()
            self._last_saved_time[symbol] = ts
            logger.debug("[handle_kline] stored candle for %s @ %s", symbol, ts)

    async def handle_ticker(self, msg):
        """
        Handle incoming ticker updates:
        - update latest_open_interest and ticker_data,
        - then notify each bot of the price update via on_ticker_update.
        """
        data = msg.get("data", {})
        entries = data if isinstance(data, list) else [data]
        # 1) Update open interest and ticker_data
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
            # ── push an OI snapshot so Golden Setup always sees current ΔOI ──
            hist = self.oi_history.setdefault(symbol, deque(maxlen=500))
            # добавляем только, если значение изменилось, чтобы не раздувать очередь
            if not hist or hist[-1] != oi_val:
                hist.append(oi_val)

        # 2) Notify bots of ticker updates for their open positions
        for bot in self.position_handlers:
            for ticker in entries:
                sym = ticker.get("symbol")
                if not sym or sym not in bot.open_positions:
                    continue
                last_price = safe_to_float(ticker.get("lastPrice", 0))
                # schedule on_ticker_update on the bot
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
                    # Support both list-based and dict-based responses
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
                    self.oi_history[symbol].append(0.0)  # will be replaced below if needed
                    # ---- CVD (historical) ------------------------------
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

        # refresh dynamic thresholds dictionary instead of recreating it
        self._liq_thresholds.clear()
        try:
            import pandas as _pd
            _liq = _pd.read_csv(LIQUIDATIONS_CSV_PATH)
            for _sym, _grp in _liq.groupby('symbol'):
                p5 = _grp['value_usdt'].quantile(0.05)
                self._liq_thresholds[_sym] = max(5000.0, p5 * 4)
        except Exception as _e:
            logger.warning('Threshold init failed: %s', _e)

    # ------------------------------------------------------------------
    # Helper methods for new liquidation strategy v2
    # ------------------------------------------------------------------
    def get_liq_threshold(self, symbol: str, default: float = 5000.0) -> float:
        t24 = safe_to_float(self.ticker_data.get(symbol, {}).get("turnover24h", 0))
        if t24 >= LARGE_TURNOVER:
            return 0.0015 * t24      # 0.15 %
        elif t24 >= MID_TURNOVER:
            return 0.0025 * t24      # 0.25 %
        return max(8_000.0, self._liq_thresholds.get(symbol, default))

    def get_avg_volume(self, symbol: str, minutes: int = 30) -> float:
        candles_deque = self.candles_data.get(symbol, [])
        if not candles_deque:
            return 0.0
        # deque does not support slicing → convert to list first
        recent = list(candles_deque)[-minutes:]
        vols = [
            safe_to_float(c.get("turnover") or c.get("volume", 0))
            for c in recent
        ]
        vols = [v for v in vols if v > 0]
        return sum(vols) / max(1, len(vols))

    # ---- σ-волатильность -------------------------------
    def _sigma_5m(self, symbol: str, window: int = VOL_WINDOW) -> float:
        candles = list(self.candles_data.get(symbol, []))[-window:]
        if len(candles) < window:
            return 0.0
        moves = [abs(c["closePrice"] - c["openPrice"]) / c["openPrice"] for c in candles]
        return float(np.std(moves)) if moves else 0.0

    # ---- new-listing protection ---------------------------------
    def _listing_age_minutes(self, symbol: str) -> float:
        """
        Сколько минут прошло с первой подтверждённой свечи по symbol.
        Если данных ещё нет — 0.
        """
        candles = self.candles_data.get(symbol, [])
        if not candles:
            return 0.0
        first_ts = candles[0]["startTime"]
        return (dt.datetime.utcnow() - first_ts.to_pydatetime()).total_seconds() / 60.0

    def is_too_new(self, symbol: str, min_age: int | None = None) -> bool:
        min_age = min_age or getattr(self, "listing_age_min", LISTING_AGE_MIN_MINUTES)
        """
        True, если паре меньше min_age минут — значит слишком свежая
        и торговля блокируется.
        """
        return self._listing_age_minutes(symbol) < min_age

    def is_volatile_spike(self, symbol: str, candle: dict) -> bool:
        sigma = self._sigma_5m(symbol)
        if sigma == 0:
            return False
        move = abs(candle["closePrice"] - candle["openPrice"]) / candle["openPrice"]
        return move >= VOL_COEF * sigma

    def funding_cool(self, symbol: str) -> bool:
        """
        Funding-rate должен остыть: либо знак сменился +→−,
        либо |current| ≤ 0.5 × |prev|.
        Если истории <2 значений — True (не блокируем).
        """
        hist = self.funding_history.get(symbol, [])
        if len(hist) < 2:
            return True
        prev, curr = hist[-2], hist[-1]
        if prev > 0 and curr < 0:
            return True
        return abs(curr) <= 0.5 * abs(prev)

    # ---------- RSI-based entry guard ----------
    def _rsi_series(self, symbol: str, period: int = 14, lookback: int = 180):
        """
        Возвращает Series с RSI(period) за последние *lookback* минут.
        Пустая Series, если данных мало.
        """
        candles = list(self.candles_data.get(symbol, []))
        if len(candles) < period + lookback:
            return pd.Series(dtype=float)
        closes = [c["closePrice"] for c in candles[-lookback:]]
        return ta.rsi(pd.Series(closes), length=period)

    def rsi_blocked(self, symbol: str, side: str,
                    overbought: float = 82.0,
                    oversold: float = 20.0,
                    lookback: int = 180) -> bool:
        """
        True, если RSI14 держится в экстремальной зоне все *lookback* минут:
        – блокируем LONG, когда RSI14 > overbought (82)
        – блокируем SHORT, когда RSI14 < oversold  (20)
        """
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
    
    # ---- helper: aggregate N recent 1-min candles into a single candle ----
    def _aggregate_last_candles(self, symbol: str, n: int = 1):
        """
        Склеивает последние *n* минутных свечей в одну OHLCV.
        Возвращает dict с ключами openPrice/…/volume.
        Если данных меньше n — возвращает None.
        """
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

    # ------------------------------------------------------------------
    # v2: persist candles/volume/oi history to disk once a minute
    # ------------------------------------------------------------------
    def _save_history(self):
        """Persist history dictionaries into history.pkl."""
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
        """Background task: persist history every *interval* seconds."""
        while True:
            await asyncio.sleep(interval)
            self._save_history()

    # ------------------------------------------------------------------
    # Daily optimisation: choose per‑symbol threshold with best win‑rate
    # ------------------------------------------------------------------
    async def optimize_liq_thresholds(self,
                                      trades_csv: str = "trades_for_training.csv",
                                      min_trades: int = 30):
        """
        Re‑evaluate optimal liquidation threshold for every symbol based on
        historical deals recorded in *trades_for_training.csv*.

        The algorithm scans only rows where source == "liquidation" and
        selects the quantile‑based threshold that maximises win‑rate.
        Results are saved into LIQ_THRESHOLD_CSV_PATH and applied in‑memory.
        """
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

        # persist & apply
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

            # apply in‑memory
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
        # PATCH: abort if qty is below exchange minQty
        step    = self.qty_step_map.get(symbol, DEC_TICK)
        min_qty = self.min_qty_map.get(symbol, step)
        if float(qty) < float(min_qty):
            raise RuntimeError(f"Qty {qty} < min_qty {min_qty}")
        # Set the order type in the payload
        args["orderType"] = order_type
        # Указываем индекс позиции
        args["positionIdx"] = position_idx
        # Ensure price is set only for Limit orders with a price
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
            if resp.get("req_id") == req_id:  # или сверяем op/args
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
        
        # Берем свечу N минут назад и текущую
        old_candle = candles[-minutes]
        new_candle = candles[-1]
        
        old_close = safe_to_float(old_candle.get("closePrice", 0))
        new_close = safe_to_float(new_candle.get("closePrice", 0))
        
        if old_close <= 0:
            return False
        
        # Рассчитываем процент изменения
        pct_change = (new_close - old_close) / old_close * 100.0
        return pct_change >= 3.0

# Helper functions for feature calculation
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
# == МАШИННОЕ ОБУЧЕНИЕ: КОМПОНЕНТЫ ДЛЯ PYTORCH И MLX
# ======================================================================

# ----------------------------------------------------------------------
# --- Компоненты для PyTorch
# ----------------------------------------------------------------------

class MLXInferencer:
    def __init__(self, model_path: str = "golden_model_v19.pt"):
        self.device = None
        self.input_dim = len(FEATURE_KEYS)
        self.model = None
        self.scaler = None
        
        if Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning(f"[PyTorch] Файл модели {model_path} не найден. Модель будет создана при первом обучении.")

    def _load_model(self, path: str):
        # try:
#         #     state = ckpt.get("model_state", ckpt)
#         #     self.model = GoldenNet(input_size=self.input_dim)
        #     self.model.to(self.device)
#         #     self.model.load_state_dict(state, strict=False)
#         #     self.model.eval()
        #     logger.info(f"[PyTorch] Модель из {path} успешно загружена.")
#         #     if isinstance(ckpt, dict) and "scaler" in ckpt:
#         #         self.scaler = ckpt["scaler"]
        #         logger.info("[PyTorch] Скейлер из чекпойнта успешно загружен.")
        # except Exception as e:
        #     logger.error(f"[PyTorch] Ошибка загрузки модели из {path}: {e}", exc_info=True)
        #     self.model = None

        try:
            # Explicitly setting weights_only=False to allow loading StandardScaler.
            # This is done assuming the source of the model is trusted.
#             state = ckpt.get("model_state", ckpt)
#             self.model = GoldenNet(input_size=self.input_dim)
            self.model.to(self.device)
#             self.model.load_state_dict(state, strict=False)
#             self.model.eval()
            logger.info(f"[PyTorch] Модель из {path} успешно загружена.")
#             if isinstance(ckpt, dict) and "scaler" in ckpt:
#                 self.scaler = ckpt["scaler"]
            logger.info("[PyTorch] Скейлер из чекпойнта успешно загружен.")
        except Exception as e:
            logger.error(f"[PyTorch] Ошибка загрузки модели из {path}: {e}", exc_info=True)
            self.model = None

def infer(self, features: np.ndarray) -> np.ndarray:
    """
    MLX-only infer. Возвращает (N, 1).
    """
    if features is None:
        return np.zeros((0, 1), dtype=np.float32)

    feats = np.asarray(features, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    if feats.shape[1] != self.input_dim:
        raise ValueError(f"infer: expected {self.input_dim} features, got {feats.shape[1]}")

    # Скейлер, если есть
    if self.scaler is not None:
        feats = self.scaler.transform(feats).astype(np.float32)

    # Если модель не загружена — предсказуемо отдаём нули
    if self.model is None:
        return np.zeros((feats.shape[0], 1), dtype=np.float32)

    # ---- Реальный MLX-вызов (раскомментируйте при наличии модели) ----
    # import mlx.core as mx
    # x = mx.array(feats)        # (N, D)
    # y = self.model(x)          # (N, 1)  — MLX-модель
    # return np.asarray(y)

    # Временный fallback:
    return np.zeros((feats.shape[0], 1), dtype=np.float32)

# ----------------------------------------------------------------------
# --- Компоненты для MLX
# ----------------------------------------------------------------------

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
#                 self.model.eval()
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
    #     """Обучает GoldenNet на MLX."""
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
    #     optimizer = mlx_optim.Adam(learning_rate=lr)
        loss_fn = lambda model, x, y: mlx_nn.losses.mse_loss(model(x), y).mean()
        loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

        for epoch in range(num_epochs):
            x_train, y_train = mlx.array(feats_scaled), mlx.array(targ)
            loss, grads = loss_and_grad_fn(model, x_train, y_train)
    #         optimizer.update(model, grads)
    #         mlx.eval(model.parameters(), optimizer.state)
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1} [MLX] – Loss: {0.0:.5f}")

        return model, scaler

# ======================================================================
# == КОНЕЦ ML-БЛОКА
# ======================================================================

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
        # [ИСПРАВЛЕНО] Добавлены все новые атрибуты
        "gemini_api_key", "ml_lock", "ml_model_bundle", "_last_manage_ts", "training_data_path", "evaluated_signals_cache",
        "gemini_limiter", "_evaluated_signals_cache", "openai_api_key", "ai_stop_management_enabled", "failed_stop_attempts",
        "ml_framework", "_build_default", "ai_provider", "stop_loss_mode", "_last_logged_stop_price", "recently_closed", "_cleanup_task",
        "_last_trailing_stop_order_id", "ai_timeout_sec", "ai_sem", "ai_circuit_open_until", "_ai_silent_until",
        "ml_gate_abs_roi", "ml_gate_min_score", "ml_gate_sigma_coef", "leverage", "order_correlation",

    )

    def __init__(self, user_data, shared_ws, golden_param_store):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "real")
        self.listing_age_min = int(user_data.get("listing_age_min_minutes", LISTING_AGE_MIN_MINUTES))

        # MLX-only: без .pt и без CoreML
        self.ml_inferencer = MLXInferencer(
            model_path=user_data.get("mlx_model_path", "golden_model_mlx.safetensors"),
            scaler_path=user_data.get("scaler_path", "scaler.pkl"),
        )

        # CoreML tuner отключён — используем статические пороги сквиза
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

        self.shared_ws = shared_ws
        # регистрации на WS — как было
        self.history = self.shared_ws.candles_data if self.shared_ws else {}

        if self.shared_ws is not None:
            self.shared_ws.position_handlers.append(self),
            #self.shared_ws.listing_age_min = self.listing_age_min
        self.symbols = shared_ws.symbols if shared_ws else []

        self.gemini_api_key = user_data.get("gemini_api_key")
        if not self.gemini_api_key:
                    logger.warning(f"[User {self.user_id}] Ключ Gemini API не найден! AI-оценка будет отключена.")

        # --- [ИЗМЕНЕНИЕ] Добавляем загрузку и проверку ключа OpenAI ---
        self.openai_api_key = user_data.get("openai_api_key")
        if not self.openai_api_key:
            logger.warning(f"[User {self.user_id}] Ключ OpenAI API не найден! Оценка через OpenAI будет отключена.")
        # --- Конец изменения ---

        self.ml_lock = asyncio.Lock()
        self.ml_model_bundle = {"model": None, "scaler": None}

        self._last_manage_ts: dict[str, float] = {} # Хранит время последнего вызова "Хранителя

        self.ws_private = None
        self.open_positions = {}
        # Track last known (side, size) for each symbol to suppress duplicate logs
        self.last_position_state: dict[str, tuple[str, Decimal]] = {}
        self.golden_param_store = golden_param_store
        # handle reference to the market loop task
        self.market_task = None
        # WS для отправки торговых запросов
        self.ws_trade = None
        # Индекс позиции (для Bybit V5: 1 или 2)
        self.position_idx = user_data.get("position_idx", 1)
        #self.load_model()

        self.POSITION_VOLUME = safe_to_float(user_data.get("volume", 1000))
        # максимальный разрешённый общий объём открытых позиций (USDT)
        self.MAX_TOTAL_VOLUME = safe_to_float(user_data.get("max_total_volume", 5000))
        # Maximum allowed total exposure across all open positions (in USDT)
        self.qty_step_map: dict[str, Decimal] = {}
        self.min_qty_map: dict[str, Decimal] = {}
        self.price_tick_map: dict[str, float] = {}
        # track symbols that recently failed order placement
        self.failed_orders: dict[str, Decimal] = {}
        # символы, по которым уже отправлен ордер,
        # но позиция ещё не пришла по private‑WS
        self.pending_orders: dict[str, float] = {}
        # per‑symbol rolling buffer of recent liquidation events
        self.liq_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.pending_strategy_comments: dict[str, str] = {}
        self.last_trailing_stop_set: dict[str, Decimal] = {}
        self.position_lock = asyncio.Lock()
        self.pending_orders = {}
        # Словарь закрытых позиций
        self.closed_positions = {}
        # Task for periodic PnL checks
        #self.pnl_task = None
        self.last_seq = {}
        self.wallet_task = None  # periodic wallet snapshot task
        self.last_stop_price: dict[str, float] = {}
        # [НОВЫЙ АТРИБУТ] Хранит последнюю цену стопа, о которой было отправлено уведомление
        self._last_logged_stop_price: dict[str, float] = {}
        self._last_trailing_stop_order_id: dict[str, str] = {}
        # last successful trailing‑stop timestamp  symbol → ts
        self._last_trailing_ts: dict[str, float] = {}
        # Track WS-opened and WS-closed symbols to priority state from WS
        self.ws_opened_symbols = set()
        self.ws_closed_symbols = set()
        self.limiter = AsyncLimiter(max_rate=100, time_period=1)  # 100 вызовов/сек
        # symbols that have already been averaged once (to prevent infinite loops)
        self.averaged_symbols: set[str] = set()
        # Ensure asyncio import at the top (already present)
        self._recv_lock = asyncio.Lock()
        # register this instance so the admin snapshot can include it
        GLOBAL_BOTS.append(self)
        self.max_allowed_volume = Decimal('10000')  # Or adjust as needed

        self._age_cache = {}          # кэш <symbol → (age_min, ts_now)>

        self.trade_history_file = Path("trades_history.json")
        self.active_trades: dict[str, dict] = {}      # symbol → {side, qty, avg_price, fees, open_ts}
        self._load_trade_history()                     # создаёт файл, если его нет

        self.ai_timeout_sec = float(user_data.get("ai_timeout_sec", 8.0))   # жёсткий таймаут на ответ ИИ
        self.ai_sem = asyncio.Semaphore(user_data.get("ai_max_concurrent", 1))  # ограничим параллелизм
        self.ai_circuit_open_until = 0.0   # когда > now — ИИ временно выключен

        # --- strategy selector -------------------------------------------------
        # priority: strategy_mode > legacy strategy > default "full"
        raw_mode = user_data.get("strategy_mode")
        raw_mode = str(raw_mode).lower()

        # ── в __init__  TradingBot (блок alias_map) ──
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
            # NEW ↓↓↓
            "golden_squeeze":  "golden_squeeze",
            "liq_squeeze":     "liq_squeeze",
        }

        self.strategy_mode = alias_map.get(raw_mode, "full")
        self.active_trade_entries = {}

        # ── per-logic trailing-stop settings ───────────────────────────────
        self.trailing_start_map: dict[str, float] = user_data.get("trailing_start_pct", {})
        self.trailing_start_pct: float = self.trailing_start_map.get(
            self.strategy_mode,
            DEFAULT_TRAILING_START_PCT,
        )

        self.trailing_gap_map: dict[str, float] = user_data.get("trailing_gap_pct", {})
        self.trailing_gap_pct: float = self.trailing_gap_map.get(
            self.strategy_mode,
            DEFAULT_TRAILING_GAP_PCT,
        )
        
        # [НОВЫЙ АТРИБУТ] Временный кэш для предотвращения "воскрешения" позиций
        self.recently_closed: dict[str, float] = {}
        # [НОВЫЙ АТРИБУТ] Задача для очистки кэша
        self._cleanup_task = asyncio.create_task(self._cleanup_recently_closed())

        #self._last_stop_set_ts: Dict[str, float] = {}

        self.STOP_WARMUP_SEC = int(user_data.get("stop_warmup_sec", 60))      # задержка перед постановкой SL/TP
        self.ATR_MULT_SL_INIT = float(user_data.get("atr_mult_sl_init", 2.5)) # стартовый SL = 2.5*ATR
        self.ATR_MULT_TP_INIT = float(user_data.get("atr_mult_tp_init", 3.0)) # стартовый TP = 3*ATR
        self.ATR_MULT_TRAIL   = float(user_data.get("atr_mult_trail", 1.8))   # тралим на 1.8*ATR
        self.BREAKEVEN_TRIGGER_ATR = float(user_data.get("breakeven_trigger_atr", 1.0)) # перевод в BE после +1*ATR

        # --- ДОБАВЬТЕ ЭТОТ БЛОК ---
        # Управляем ИИ-стопом. По умолчанию выключен.
        self.ai_stop_management_enabled = user_data.get("ai_stop_management_enabled", False) 

        # Порог сквиза по умолчанию (может быть переопределён в user_state.json)
        self.squeeze_threshold_pct = user_data.get(
            "squeeze_threshold_pct",
            SQUEEZE_THRESHOLD_PCT,
        )
        # минимум мощности сквиза (произведение %изменения цены на % изменения объёма)
        self.squeeze_power_min = safe_to_float(user_data.get("squeeze_power_min", DEFAULT_SQUEEZE_POWER_MIN))

        self.apply_user_settings()        # начальная синхронизация

        self.pending_timestamps = {}  # type: dict[str, float]
        # active "reserve" limit orders  symbol → {"orderId": str, "ts": float}
        self.reserve_orders = {}

        # сопоставление ордера с корреляционным id
        self.order_correlation: dict[str, str] = {}

        # ---- защита от повторного сквиза -----------------------------
        self.last_squeeze_ts: dict[str, float] = defaultdict(float)
        self.squeeze_cooldown_sec: int = int(
            user_data.get("squeeze_cooldown_sec", SQUEEZE_COOLDOWN_SEC)
        )
        # ── Golden-setup cooldown / filters ─────────────────────────────
        self.golden_cooldown_sec = int(user_data.get("golden_cooldown_sec", 300))  # 5 мин по-умолчанию
        self.golden_rsi_max      = float(user_data.get("golden_rsi_max", 80))      # RSI-ограничение
        self._last_golden_ts     = defaultdict(float)                              # symbol → ts

        # ── warm‑up (startup grace period) ───────────────────────────────
        self.warmup_done     = False
        self.warmup_seconds  = int(user_data.get("warmup_seconds", 480))  # default 8 min

        self.averaging_enabled: bool = True   # averaging-mode toggle
        self.gemini_limiter = AsyncLimiter(max_rate=1000, time_period=60)
        self.failed_stop_attempts = {} # <--- ДОБАВЬТЕ ЭТУ СТРОКУ

        # Где-нибудь в конце __init__
        self._evaluated_signals_cache: Dict[str, float] = {}

        asyncio.create_task(self._unfreeze_guard())

        self._last_snapshot_ts: dict[str, float] = {}
        # σ(ΔOI) cache for dynamic liquidation guard
        self._oi_sigma: dict[str, float] = defaultdict(float)

        self.reserve_orders: dict[str, dict] = {}
        # housekeeping
        self._pending_clean_task = asyncio.create_task(self._pending_cleaner())

        
        # Инициализация устройства для вычислений
        self.device = None
        logger.info(f"[ML] Using compute device: {self.device}")

        # Определяем, какой ML фреймворк использовать
        # self.ml_framework = user_data.get("ml_framework", "pytorch") # По умолчанию - старый добрый PyTorch
        # logger.info(f"Выбран ML фреймворк: {self.ml_framework.upper()}")

        self.ml_framework = "mlx"
        self.ml_inferencer = MLXInferencer(
            model_path="golden_model_mlx.safetensors",
            scaler_path="scaler.pkl"
        )

        # ---------------- ML pre-filter (gate) ----------------
        # Прогноз модели мы интерпретируем как ожидаемый pnl_pct/100 (см. тренинг),
        # поэтому ROI% ≈ pred * 100 * leverage.
        self.ml_gate_abs_roi    = float(user_data.get("ml_gate_abs_roi", 1.5))  # требуемый |ROI%| с учетом плеча
        self.ml_gate_min_score  = float(user_data.get("ml_gate_min_score", 0.02))  # сырой score (до *100*lev), чтобы отрезать “шум”
        self.ml_gate_sigma_coef = float(user_data.get("ml_gate_sigma_coef", 0.0))  # если >0, динамически повышаем порог по σ(5m)

        # ── AI-provider normalisation ───────────────────────────
        raw_ai = str(user_data.get("ai_provider", "")).lower()
        provider_map = {
            "":       "ollama",   # значение по-умолчанию
            "ai":     "ollama",
            "ollama": "ollama",
            "openai": "openai",
            "gpt":    "openai",
        }
        self.ai_provider = provider_map.get(raw_ai, "ollama")
        logger.info(f"Выбран AI-провайдер: {self.ai_provider.upper()}")
        self.stop_loss_mode = user_data.get("stop_loss_mode", "strat_loss") # <-- ДОБАВЬТЕ ЭТУ СТРОКУ

        logger.info(f"Выбран AI провайдер: {self.ai_provider.upper()}")
        logger.info(f"Выбран режим стоп-лосса: {self.stop_loss_mode.upper()}") # <-- И эту для логирования

        # --- AI runtime guards / throttling ---
        self.ai_timeout_sec = float(user_data.get("ai_timeout_sec", 8.0))
        self.ai_sem = asyncio.Semaphore(user_data.get("ai_max_concurrent", 2))
        self.ai_circuit_open_until = 0.0
        self._ai_silent_until = 0.0

        # ML-атрибуты
        self.model = None
        self.MLX_model = None
        self.feature_scaler = None
        self.load_ml_models()
        self.last_retrain = time.time()
        
        # [ИСПРАВЛЕНО] Загружаем накопленные данные для обучения или создаем новый буфер
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
            self.training_data = deque(maxlen=5000)  # Кольцевой буфер для обучения
        # — перезапуск авто-retrain каждые 60 мин —
        try:
            asyncio.get_running_loop().create_task(self._retrain_loop())
            logger.info("[retrain] task scheduled")
        except RuntimeError:
            # Event-loop ещё не запущен; run_all создаст задачу позже
            pass
        self.symbol_info: dict[str, dict] = {}

        self.pending_signals: dict[str, float] = {}   # symbol → timestamp
        self.max_signal_age = 30.0                    # сек.
        self.leverage = 10

        # initialise best-entry cache (пер-символ / side / logic)
        self._best_entries: dict[tuple[str, str, str], tuple[float, dict]] = {}
        # Поставьте задачу-очиститель:
        # Schedule stale-signal cleanup if an event loop is running
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._drop_stale_signals())
        except RuntimeError:
            # No running event loop now; will schedule in run_all()
            pass

        self._oi_sigma: dict[str, float] = defaultdict(float)
        # self._pending_clean_task уже создан ранее, не создаём повторно

        #self._periodic_eval_task = asyncio.create_task(self._periodic_eval())
        self._best_entry_seen: set[tuple] = set()
        self._best_entry_cache: dict[tuple, tuple[float, dict]] = {}

        self._golden_weights = {}
        self._squeeze_weights = {}
        self._liq_weights = {}
        #asyncio.create_task(self._periodic_autotune())
        asyncio.create_task(self._reload_weights_loop())

    # ──────────────────────────────────────────────────────────────────
    # ML best‑entry logger
    # ------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
    def _record_best_entry(
        self,
        symbol: str,
        logic: str,            # "golden_setup" / "squeeze" / "liquidation"
        side: str,             # "Buy" | "Sell"
        prob: float,           # «сила» сигнала
        features: dict[str, float],
    ) -> None:
        """
        Пишем в ml_best_entries.csv **только лучший** (prob-max) сигнал за текущую минуту
        для конкретного (symbol, logic, side).

        • исключаем дубликаты Buy ⇄ Sell;  
        • на (symbol,logic,side) в минуту ― одна строка;  
        • ухудшения (prob ≤ prev) не логируем.
        """
        # 1) ключ «текущая минута»
        t_key      = datetime.utcnow().replace(second=0, microsecond=0)
        tuple_key  = (symbol, logic, side, t_key)
        cache_key  = (symbol, logic, side)

        # 2) уже писали что-то в эту минуту
        if tuple_key in self._best_entry_seen:
            return

        # 3) хуже, чем предыдущий максимум
        prev_prob = self._best_entry_cache.get(cache_key, (0.0,))[0]
        if prob <= prev_prob:
            return

        # 4) обновляем in-memory кэш
        self._best_entry_cache[cache_key] = (prob, features)
        self._best_entry_seen.add(tuple_key)

        # 5) пишем CSV
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

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def _unfreeze_guard(self, ttl: float = 180.0):
        """
        [V2 - Исправленная] Очищает зависшие ключи из _evaluated_signals_cache,
        корректно работая со словарями в качестве значений.
        """
        while True:
            await asyncio.sleep(30) # Проверяем каждые 30 секунд
            try:
                now = time.time()
                stale_keys = []
                for key, value in self._evaluated_signals_cache.items():
                    # Проверяем, что значение является словарем и содержит ключ 'time'
                    if isinstance(value, dict) and 'time' in value:
                        timestamp = value['time']
                        if now - timestamp > ttl:
                            stale_keys.append(key)
                    # Обрабатываем старый формат на всякий случай
                    elif isinstance(value, (int, float)):
                        if now - value > ttl:
                            stale_keys.append(key)
                
                if stale_keys:
                    logger.debug(f"[unfreeze_guard] Очистка {len(stale_keys)} устаревших ключей из кэша сигналов.")
                    for key in stale_keys:
                        self._evaluated_signals_cache.pop(key, None)
                        
            except Exception as e:
                logger.error(f"[unfreeze_guard] Ошибка в цикле очистки кэша: {e}", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────────────

    async def adaptive_entry(self, symbol, side, qty, max_entry_timeout):
        """
        Alias for adaptive_squeeze_entry in demo mode (called by execute_golden_setup and squeeze block).
        """
        return await self.adaptive_squeeze_entry(symbol, side, qty, max_entry_timeout)

    async def adaptive_entry_ws(self, symbol, side, qty, position_idx, max_entry_timeout):
        """
        Alias for adaptive_squeeze_entry_ws in real mode (called by execute_golden_setup and squeeze block).
        """
        return await self.adaptive_squeeze_entry_ws(symbol, side, qty, position_idx, max_entry_timeout)

    # ---------- symbol meta helpers ----------
    # --- If there is a helper train_model inside TradingBot, update it as well:
    # (Search for def train_model inside this class and update as per instructions if present.)
    
    async def ensure_symbol_meta(self, symbol: str) -> None:
        """
        Lazily fetch qtyStep / minOrderQty for a symbol and cache into
        self.qty_step_map / self.min_qty_map so that rounding uses the
        exact step required by Bybit. If data already cached — no‑op.
        """
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
            # cache as float
            self.qty_step_map[symbol] = step
            self.min_qty_map[symbol]  = minq
            logger.debug("[symbol_meta] %s qtyStep=%s  minQty=%s", symbol, step, minq)
        except Exception as e:
            # fallback to 0.001 if request fails
            self.qty_step_map.setdefault(symbol, 0.001)
            self.min_qty_map.setdefault(symbol, 0.001)
            logger.warning("[symbol_meta] fetch failed for %s: %s", symbol, e)

    # ───────────────── ML helper ─────────────────
    def _build_entry_features(self, symbol: str) -> list[float]:
        """
        Делает «снимок» рынка в момент входа и возвращает
        вектор длиной len(FEATURE_KEYS).
        Заполняем пока только пару простых фичей; остальные – 0.0.
        """
        candles = self.shared_ws.candles_data.get(symbol, [])
        vec = [0.0] * len(FEATURE_KEYS)

        # %-изменение цены за 5 мин
        try:
            idx = FEATURE_KEYS.index("pct5m")
            vec[idx] = compute_pct(candles, minutes=5)
        except ValueError:
            pass

        # объём за 5 мин
        try:
            idx = FEATURE_KEYS.index("vol5m")
            vec[idx] = sum_last_vol(candles, minutes=5)
        except ValueError:
            pass

        return vec

    # --- каждые 24 ч запускаем autotune.py -----------------------------------
    async def _periodic_autotune(self):
        # while True:
        #     try:
        #         proc = await asyncio.create_subprocess_exec(
        #             sys.executable, "autotune.py")
        #         await proc.wait()
        #     except Exception as e:
        #         logger.warning("[autotune] failed: %s", e)
        #     await asyncio.sleep(24 * 3600)          # сутки

        pass

    # --- каждые 10 мин перечитываем *.csv ------------------------------------
    async def _reload_weights_loop(self):
        while True:
            self._golden_weights  = self._read_weights("golden_feature_weights.csv")
            self._squeeze_weights = self._read_weights("squeeze_feature_weights.csv")
            self._liq_weights     = self._read_weights("liq_feature_weights.csv")
            await asyncio.sleep(600)

    # ── (ниже по файлу, внутри TradingBot или как @staticmethod) ─────────
    @staticmethod
    def _read_weights(fname: str):
        import pandas as pd, numpy as np, pathlib
        p = pathlib.Path(fname)
        if not p.exists():
            # ► единственное место, где имеет смысл fallback ◄
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

    # [НОВАЯ ФУНКЦИЯ] ВНУТРИ КЛАССА TRADINGBOT
    async def _cleanup_recently_closed(self, interval: int = 15, max_age: int = 60):
        """Периодически удаляет старые записи из кэша недавно закрытых позиций."""
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

    # async def _periodic_eval(self, interval: int = 20):
    #     """
    #     Background safety loop: every *interval* seconds explicitly
    #     calls evaluate_position() for all open positions.  This makes
    #     sure trailing‑stop logic keeps working even if ticker updates
    #     momentarily stop coming for a symbol.
    #     """
    #     while True:
    #         await asyncio.sleep(interval)
    #         for sym, pos in list(self.open_positions.items()):
    #             try:
    #                 await self.evaluate_position({
    #                     "symbol": sym,
    #                     "size":   str(pos.get("volume", 0)),
    #                     "side":   pos.get("side", ""),
    #                 })
    #             except Exception as exc:
    #                 logger.debug("[periodic_eval] %s error: %s", sym, exc)

    # ──────────────────────────────────────────────────────────────────
    def _load_trade_history(self) -> None:
        """Создаём пустой trades_history.json, если его ещё нет."""
        if not self.trade_history_file.exists():
            self.trade_history_file.write_text("[]", encoding="utf-8")

    ### _drop_stale_signals (целиком)
    async def _drop_stale_signals(self) -> None:
        """Удаляем сигналы, которым больше self.max_signal_age секунд."""
        while True:
            now = time.time()
            for sym, ts in list(self.pending_signals.items()):
                if now - ts > self.max_signal_age:
                    self.pending_signals.pop(sym, None)
                    logger.debug("[signals] %s expired (%.1fs)", sym, now - ts)
            await asyncio.sleep(5)

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def _pending_cleaner(self, interval: int = 30):
        """
        [V2 - Исправлено удаление из словаря]
        Корректно удаляет зависшие ордера из словаря `pending_orders`,
        используя метод .pop() вместо .discard().
        """
        while True:
            await asyncio.sleep(interval)
            try:
                now = time.time()
                # Находим символы, ордера по которым "зависли"
                expired_symbols = [
                    symbol for symbol, timestamp in self.pending_timestamps.items()
                    if now - timestamp > 120  # 2 минуты таймаут
                ]

                if not expired_symbols:
                    continue

                # Используем блокировку для безопасного изменения словарей
                async with self.position_lock:
                    for symbol in expired_symbols:
                        # [ИЗМЕНЕНИЕ] Используем .pop() для удаления из словарей
                        self.pending_orders.pop(symbol, None)
                        self.pending_timestamps.pop(symbol, None)
                        logger.warning(f"[Pending Cleanup] Ордер для {symbol} завис и был удален из очереди.")

            except Exception as e:
                logger.error(f"[Pending Cleanup] Критическая ошибка в чистильщике: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Cool-down / permission helpers
    # ------------------------------------------------------------------
    # ---------- helper: convert USDT exposure → correct quantity ----------
    # ВНУТРИ КЛАССА TRADINGBOT
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

        raw_qty = 0.99 * usd_amount / price
        if step > 0:
             qty = math.floor(raw_qty / step) * step
        else:
             qty = raw_qty
        
        if qty < min_qty:
            qty = min_qty
            
        return qty

    def _golden_allowed(self, symbol: str) -> bool:
        """
        True → Golden‑setup entry разрешён:
        • cooldown истёк
        • RSI14 не выше golden_rsi_max (если доступен)
        """
        now = time.time()
        # 1) Cool‑down check
        if now - self._last_golden_ts.get(symbol, 0.0) < self.golden_cooldown_sec:
            return False

        # # 2) Defensive RSI guard
        # rsi_series = None
        # if self.shared_ws is not None and hasattr(self.shared_ws, "_rsi_series"):
        #     try:
        #         rsi_series = self.shared_ws._rsi_series(symbol, period=14, lookback=1)
        #     except Exception:
        #         # Any error while computing RSI → treat as "not overbought"
        #         rsi_series = None

        # if (
        #     rsi_series is not None
        #     and not rsi_series.empty
        #     and rsi_series.iloc[-1] > self.golden_rsi_max
        # ):
        #     return False

        # Passed all checks → entry allowed
        return True

    def _squeeze_allowed(self, symbol: str) -> bool:
        """Разрешает повторный сквиз, когда cool-down вышел."""
        return time.time() - self.last_squeeze_ts.get(symbol, 0.0) >= self.squeeze_cooldown_sec

    def _tune_squeeze(self, feats: dict[str, float]) -> None:
        if not getattr(self, "squeeze_tuner", None):
            return                      # модели нет -> статическая логика

        vec = np.array([[feats[k] for k in SQUEEZE_KEYS]], np.float32)
        prediction = None

#         p_win   = float(res["prob"][0])        # вероятность >0
#         rec_thr = float(res["rec_thr"][0])     # рекомендованный миним. squeeze_power

        # ML-вето
        # ML veto disabled (no p_win available in MLX-only)
        # Плавное адаптивное смещение порога
        # adaptive squeeze threshold skipped (no rec_thr); keeping current value
    async def listing_age_minutes(self, symbol: str) -> float:
        now = time.time()
        cached = _listing_age_cache.get(symbol)
        # If the pair is already cached as older than the quarantine threshold,
        # return the cached value right away – no need to ping REST again.
        if cached and cached[0] >= self.listing_age_min:
            return cached[0]
        if cached and now - cached[1] < LISTING_AGE_CACHE_TTL_SEC:
            return cached[0]

        async with _listing_sem:      # ← ограничиваем параллелизм
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
                age_min = 999_999.0      # считаем пару старой
                _listing_age_cache[symbol] = (age_min, now)   # кэшируем, чтобы не запрашивать снова
                # Downgrade to DEBUG so old pairs don't “spam” the log
                logger.debug("[listing_age] %s REST err (suppressed, marked old): %s", symbol, e)
            else:
                _listing_age_cache[symbol] = (age_min, now)
            return age_min

    # ------------------------------------------------------------------
    # Proxy helpers
    # ------------------------------------------------------------------
    def check_liq_cooldown(self, symbol: str) -> bool:
        """Прокси к PublicWebSocketManager.check_liq_cooldown (для старых вызовов)."""
        return self.shared_ws.check_liq_cooldown(symbol)

    async def get_effective_total_volume(self) -> float:
        """
        Рассчитывает общий объем, включая подтвержденные позиции и ордера в ожидании.
        """
        confirmed_volume = await self.get_total_open_volume()
        
        pending_volume = 0.0
        if self.pending_orders:
            pending_volume = sum(self.pending_orders.values())
            
        effective_volume = confirmed_volume + pending_volume
        if pending_volume > 0:
            logger.info(f"[Risk] Эффективный объем: {effective_volume:.2f} USDT (Подтверждено: {confirmed_volume:.2f} + В ожидании: {pending_volume:.2f})")
            
        return effective_volume

    # ------------------   вставьте целиком вместо старой load_ml_models   ------------------
    def load_ml_models(self,
                    model_path: str = "golden_model_mlx.safetensors",
                    scaler_path: str = "scaler.pkl",
                    input_dim: int = None) -> None:
        """
        MLX-only загрузка модели и скейлера.
        Никакого PyTorch/CoreML. Предсказуемые fallback'и.
        """
        self.model = None
        self.feature_scaler = None

        # 1) Скейлер (опционально)
        if Path(scaler_path).exists():
            try:
                self.feature_scaler = joblib.load(scaler_path)
                logger.info("[MLX] Scaler загружен из %s", scaler_path)
            except Exception as e:
                logger.warning("[MLX] Не удалось загрузить scaler (%s): %s", scaler_path, e)

        # 2) Модель (опционально)
        if Path(model_path).exists():
            try:
                # Пример: подключите ваш класс MLX-модели и загрузку весов
                # from my_mlx_models import GoldenNetMLX
                # in_dim = input_dim if input_dim is not None else len(FEATURE_KEYS)
                # self.model = GoldenNetMLX(input_size=in_dim)
                # self.model.load_weights(model_path)   # реализуйте в своей модели
#                 # if hasattr(self.model, "eval"): self.model.eval()
                logger.info("[MLX] Модель загружена из %s", model_path)
            except Exception as e:
                logger.error("[MLX] Ошибка загрузки модели (%s): %s", model_path, e)
                self.model = None
        else:
            logger.info("[MLX] Файл модели %s не найден — инференс будет отдавать нули", model_path)
        
    # ----------------------------------------------------------------------------------------

    # ──────────────────────────────────────────────────────────────────
    # [ФИНАЛЬНАЯ ВЕРСИЯ] ВНУТРИ КЛАССА TRADINGBOT
    async def extract_realtime_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Надёжная сборка фичей для realtime:
        - чистим входные ряды до float
        - проверяем минимальные длины под индикаторы
        - любая ошибка -> безопасные нули (а наверху это -> HOLD)
        """
        # ---------- 0) Цена и спред ----------
        last_price = 0.0
        bid1 = 0.0
        ask1 = 0.0

        tdata = self.shared_ws.ticker_data.get(symbol) or {}
        last_price = safe_to_float(tdata.get("lastPrice", 0.0))
        bid1 = safe_to_float(tdata.get("bid1Price", 0.0))
        ask1 = safe_to_float(tdata.get("ask1Price", 0.0))

        if last_price <= 0.0:
            # пробуем REST (в треде)
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
            # последний клоуз из свечей
            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if candles:
                last_price = safe_to_float(candles[-1].get("closePrice", 0.0))
                bid1 = last_price
                ask1 = last_price

        if last_price <= 0.0:
            logger.warning("[features] %s: нет актуальной цены, прерываем", symbol)
            return {}

        spread_pct = (ask1 - bid1) / bid1 * 100.0 if bid1 > 0 else 0.0

        # ---------- 1) Источники рядов ----------
        candles = list(self.shared_ws.candles_data.get(symbol, []))
        oi_hist = list(self.shared_ws.oi_history.get(symbol, []))
        cvd_hist = list(self.shared_ws.cvd_history.get(symbol, []))

        # ---------- 2) Базовые проценты/объёмы ----------
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

        # ---------- 3) Подготовка DataFrame для TA ----------
        df_src = candles[-100:] if candles else []
        df = pd.DataFrame(df_src)

        for col in ("closePrice", "highPrice", "lowPrice", "volume"):
            if col not in df.columns:
                df[col] = np.nan
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df[col] = s.ffill().bfill()

        # минимальные длины
        n = len(df)
        close = df["closePrice"] if n else pd.Series(dtype="float64")
        high  = df["highPrice"]  if n else pd.Series(dtype="float64")
        low   = df["lowPrice"]   if n else pd.Series(dtype="float64")

        # ---------- 4) Индикаторы (с защитой) ----------
        def _safe_last(series, default=0.0):
            try:
                v = series.iloc[-1]
                return float(v) if pd.notna(v) else default
            except Exception:
                return default

        # RSI(14)
        rsi14 = 50.0
        try:
            rsi14 = _safe_last(ta.rsi(close, length=14), 50.0) if n >= 15 else 50.0
        except Exception:
            rsi14 = 50.0

        # SMA50 / EMA20
        sma50 = _safe_last(ta.sma(close, length=50), _safe_last(close, 0.0)) if n >= 50 else _safe_last(close, 0.0)
        try:
            ema20 = _safe_last(ta.ema(close, length=20), sma50) if n >= 20 else sma50
        except Exception:
            ema20 = sma50

        # ATR(14)
        try:
            atr14 = _safe_last(ta.atr(high, low, close, length=14), 0.0) if n >= 15 else 0.0
        except Exception:
            atr14 = 0.0

        # Bollinger width
        bb_width = 0.0
        try:
            if n >= 20:
                bb = ta.bbands(close, length=20)
                if bb is not None and not bb.empty:
                    bbu = _safe_last(bb.iloc[:, 0], 0.0)  # верхняя
                    bbl = _safe_last(bb.iloc[:, 2], 0.0)  # нижняя
                    bb_width = bbu - bbl
        except Exception:
            bb_width = 0.0

        # Supertrend
        try:
            st_ser = compute_supertrend(df, period=10, multiplier=3) if n > 20 else None
            supertrend_val = bool(_safe_last(st_ser, 0.0)) if st_ser is not None else False
        except Exception:
            supertrend_val = False
        supertrend_num = 1 if supertrend_val else -1

        # ADX(14), CCI(20)
        try:
            adx14 = _safe_last(ta.adx(high, low, close, length=14)["ADX_14"], 0.0) if n >= 15 else 0.0
        except Exception:
            adx14 = 0.0
        try:
            cci20 = _safe_last(ta.cci(high, low, close, length=20), 0.0) if n >= 20 else 0.0
        except Exception:
            cci20 = 0.0

        # MACD(12,26,9)
        macd_val, macd_signal = 0.0, 0.0
        try:
            if n >= 35:  # небольшой запас для стабилизации
                macd_df = ta.macd(close, fast=12, slow=26, signal=9)
                if macd_df is not None and not macd_df.empty and macd_df.shape[1] >= 3:
                    # pandas_ta обычно: [MACD, MACDh, MACDs] — осторожно с порядком!
                    macd_val   = _safe_last(macd_df.iloc[:, 0], 0.0)
                    macd_signal= _safe_last(macd_df.iloc[:, 2], 0.0)
        except Exception as e:
            logger.debug("[TA] MACD failed for %s: %s", symbol, e)

        # ---------- 5) Golden-setup ----------
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

        # ---------- 6) Squeeze ----------
        # защищаемся от деления на ноль
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

        # ---------- 7) Ликвидации ----------
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

        # ---------- 8) Временные фичи ----------
        try:
            now_ts = dt.datetime.now()  # если у вас импортировано как dt
        except Exception:
            import datetime as _dt
            now_ts = _dt.datetime.now()
        hour_of_day = int(now_ts.hour)
        day_of_week = int(now_ts.weekday())
        month_of_year = int(now_ts.month)

        # ---------- 9) Средние и дельты за 30 минут ----------
        try:
            avgVol30m = safe_to_float(self.shared_ws.get_avg_volume(symbol, 30))
        except Exception:
            avgVol30m = 0.0
        tail_oi = [safe_to_float(x) for x in oi_hist[-30:]] if oi_hist else []
        avgOI30m = float(np.nanmean(tail_oi)) if tail_oi else 0.0
        deltaCVD30m = CVD_now - (safe_to_float(cvd_hist[-31]) if len(cvd_hist) >= 31 else 0.0)

        # ---------- 10) Итоговый словарь ----------
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

        # Гарантируем, что все ключи из FEATURE_KEYS присутствуют
        for k in FEATURE_KEYS:
            features.setdefault(k, 0.0)

        return features    
    
    # ------------------------------------------------------------------
    async def _capture_training_sample(self, symbol: str, pnl_pct: float) -> None:
        """
        Сохраняет (features, target) из закрытой сделки в self.training_data.
        Target = фактический PnL% сделки.
        """
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
        """
        Логика расчёта strength из extract_realtime_features.
        Здесь — заглушка, замените на свою формулу.
        """
        return (volume / (oi + 1e-8)) * cvd

    async def build_and_save_trainset(self,
                                    csv_path: str,
                                    scaler_path: str,
                                    symbol: str | list[str],
                                    future_horizon: int = 3,
                                    future_thresh: float = 0.005):
        """Формирует обучающий датасет, сохраняет CSV и scaler."""
        # ────────────────────────────────────────────────────────────────
        # 1. Убедимся, что история подкачана
        try:
            await self.shared_ws.backfill_history()
        except Exception as e:
            logger.warning(f"[DatasetBuilder] backfill_history failed: {e}")

        # 2. Нормализуем symbols → list
        symbols_list = symbol if isinstance(symbol, (list, tuple)) else [symbol]

        rows: list[dict] = []
        min_index = 50

        for sym in symbols_list:
            # ── сохраним исходные структуры и их maxlen ────────────────
            candles_orig = self.shared_ws.candles_data[sym]
            oi_orig      = self.shared_ws.oi_history[sym]
            cvd_orig     = self.shared_ws.cvd_history[sym]

            candles_maxlen = getattr(candles_orig, "maxlen", len(candles_orig))
            oi_maxlen      = getattr(oi_orig,      "maxlen", len(oi_orig))
            cvd_maxlen     = getattr(cvd_orig,     "maxlen", len(cvd_orig))

            # ── сделаем копии списками для быстрых срезов ───────────────
            bars_all = list(candles_orig)
            oi_all   = list(oi_orig)
            cvd_all  = list(cvd_orig)

            max_i = len(bars_all) - future_horizon
            if max_i <= min_index:
                logger.warning(f"[DatasetBuilder] Not enough bars for {sym}: "
                            f"have={len(bars_all)}, need>{min_index+future_horizon}")
                continue

            # ── основной цикл: обрезаем до i, считаем фичи ──────────────
            for i in range(min_index, max_i):
                # !!! временно кладём СПИСКИ, чтобы extract_realtime_features мог резать [-50:]
                self.shared_ws.candles_data[sym] = bars_all[:i + 1]
                self.shared_ws.oi_history[sym]    = oi_all[:i + 1]
                self.shared_ws.cvd_history[sym]   = cvd_all[:i + 1]

                feats = await self.extract_realtime_features(sym)

                if not feats:          # пропускаем, если фичи собрать не удалось
                    continue

                future_price  = bars_all[i + future_horizon]["closePrice"]
                current_price = bars_all[i]["closePrice"]
                ret = future_price / current_price - 1
                feats["label"] = int(ret > future_thresh)

                rows.append(feats)

            # ── вернём полноценные deque обратно ───────────────────────
            self.shared_ws.candles_data[sym] = deque(bars_all, maxlen=candles_maxlen)
            self.shared_ws.oi_history[sym]    = deque(oi_all,   maxlen=oi_maxlen)
            self.shared_ws.cvd_history[sym]   = deque(cvd_all,  maxlen=cvd_maxlen)

        # 3. DataFrame
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("[DatasetBuilder] No training rows generated. "
                            "Проверьте объём истории и future_horizon.")

        # 4. Scaler
        feature_cols = [c for c in df.columns if c != "label"]
        scaler = StandardScaler().fit(df[feature_cols])
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")

        # 5. CSV (scaled)
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
        df_scaled.to_csv(csv_path, index=False)
        print(f"✅ trainset.csv saved: {csv_path}, shape={df_scaled.shape}")

    async def predict_action(self, symbol: str) -> str:
        """
        Делает предсказание по MLX (если доступно) и возвращает BUY/SELL/HOLD.
        Логирует решение единообразно.
        """
        cid = new_cid()
        try:
            # 1) Признаки в реальном времени
            features = await self.extract_realtime_features(symbol)
            if not features:
                logger.info("[DECIDE][%s] %s -> HOLD (no features)", cid, symbol)
                return "HOLD"

            # 2) Фильтр Golden Setup: ADX ≥ 25 и RSI14 ≤ 80 (если режим golden_only)
            if getattr(self, "strategy_mode", "") == "golden_only":
                adx = safe_to_float(features.get("adx14", 0.0))
                rsi = safe_to_float(features.get("rsi14", 0.0))
                if adx < 25.0 or rsi > 80.0:
                    logger.info("[DECIDE][%s] %s -> HOLD (golden filter adx=%.1f rsi=%.1f)",
                                cid, symbol, adx, rsi)
                    return "HOLD"

            # 3) Вектор признаков (если нужен для MLX)
            action = "HOLD"
            score_str = "n/a"

            if getattr(self, "MLX_model", None):
                # если ваша MLX-модель принимает фиксированный порядок признаков:
                vector = [features.get(k, 0.0) for k in FEATURE_KEYS]
                x = np.array(vector, dtype=np.float32).reshape(1, -1)

                pred = self.MLX_model.predict({"input": x})
                y = pred.get("output")

                if isinstance(y, np.ndarray):
                    if y.ndim == 2 and y.shape[1] == 3:
                        # вероятности классов
                        idx = int(np.argmax(y[0]))
                        action = ["BUY", "SELL", "HOLD"][idx]
                        score_str = j(y[0].tolist())
                    elif y.ndim == 2 and y.shape[1] == 1:
                        # скалярный скор
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
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Функция логирования сделки для ML-буфера
    # -----------------------------------------------------------------
    async def log_trade_for_ml(self, symbol: str, entry_data: dict, exit_data: dict):
        """
        Формирует обучающий пример по факту закрытия сделки.

        • `features` — вектор длиной len(FEATURE_KEYS)
          (отсутствующие признаки заполняются 0.0).
        • `target`   — 0 = убыток, 1 = прибыль > 1 %, 2 = нейтраль.
        """
        try:
            # 1) Достаём актуальные online-фичи
            features = await self.extract_realtime_features(symbol)

            # 2) Фиксируем/уточняем глобальный список признаков
            global FEATURE_KEYS, INPUT_DIM
            if not FEATURE_KEYS:                        # первый вызов — фиксируем
                FEATURE_KEYS = list(features)           # порядок важен!
                INPUT_DIM = len(FEATURE_KEYS)
                logger.info("[ML] FEATURE_KEYS initialised with %d fields", INPUT_DIM)

            # Диагностика: что отсутствует / что лишнее
            missing = [k for k in FEATURE_KEYS if k not in features]
            extra   = [k for k in features     if k not in FEATURE_KEYS]
            if missing:
                logger.debug("[ML] %s missing features: %s", symbol, missing)
            if extra:
                logger.debug("[ML] %s extra   features: %s", symbol, extra)

            # 3) Формируем вектор, избегая KeyError
            vector = [features.get(k, 0.0) for k in FEATURE_KEYS]

            # 4) Считаем PnL % относительно цены входа
            if entry_data["side"].lower() == "buy":
                pnl = (exit_data["price"] - entry_data["price"]) / entry_data["price"] * 100.0
            else:  # short
                pnl = (entry_data["price"] - exit_data["price"]) / entry_data["price"] * 100.0

            label = 0 if pnl < 0 else 1 if pnl > 1 else 2

            # 5) Кидаем в кольцевой буфер
            self.training_data.append({"features": vector, "target": label})

            # 6) Триггер фонового переобучения раз в час
            if time.time() - getattr(self, "last_retrain", 0) > 3600 and len(self.training_data) > 100:
                self.last_retrain = time.time()
                asyncio.create_task(self.retrain_models())

        except Exception as e:
            logger.error("[ML] Trade logging error: %s", e)

    # ─────────────────────────────────────────────────────────

    async def retrain_models(self) -> None:
        """
        MLX-only: никаких PyTorch/CoreML.
        Обновляем только scaler по накопленным данным.
        """
        try:
            data = getattr(self, "training_data", [])
            if not data or len(data) < 100:
                return

            # Формируем матрицу признаков
            X = np.asarray([rec["features"] for rec in data], dtype=np.float32)
            if X.ndim != 2 or X.shape[1] != len(FEATURE_KEYS):
                logger.warning("[MLX] retrain_models: неверная форма X=%s", X.shape if hasattr(X, "shape") else type(X))
                return

            # Чистка NaN/Inf
            mask = ~(np.isnan(X).any(1) | np.isinf(X).any(1))
            X = X[mask]
            if X.size == 0:
                return

            # Перефит скейлера (в треде, чтобы не блокировать event-loop)
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

            # Применяем новый скейлер для инференса
            if getattr(self, "ml_inferencer", None) is not None:
                self.ml_inferencer.scaler = scaler

            # Легкий маркер в историю (если функция синхронная)
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

    # ---------------- ML model loading ----------------
    def load_model(self):
        model_path = f"models/golden_model_{self.user_id}.pt"
        if os.path.exists(model_path):
#             self.model.eval()
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
#         self.model = train_golden_model(self.training_data)
        self.save_model()

    def predict_entry_quality(self, features: Sequence[float]) -> float:
        """
#         MLX-only: возвращает скалярный скор входа. Без Torch, без .eval(), без .item().
        Предсказуемый фолбэк: если модель/скейлер не готовы — 0.0.
        """
        if not features:
            return 0.0

        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Защитимся от рассинхрона размерности
        if x.shape[1] != len(FEATURE_KEYS):
            logger.warning("predict_entry_quality: expected %d features, got %d",
                        len(FEATURE_KEYS), x.shape[1])
            return 0.0

        # Скейлер (если есть)
        inf = getattr(self, "ml_inferencer", None)
        if inf is None:
            return 0.0
        if getattr(inf, "scaler", None) is not None:
            try:
                x = inf.scaler.transform(x).astype(np.float32)
            except Exception as e:
                logger.warning("predict_entry_quality: scaler.transform failed: %s", e)
                return 0.0

        # Модель
        if getattr(inf, "model", None) is None:
            return 0.0

        # Вызов модели (MLX). Если у вас есть реальная MLX-модель, раскомментируйте блок ниже.
        # import mlx.core as mx
        # y = inf.model(mx.array(x))   # ожидается форма (N, 1)
        # y_np = np.asarray(y)
        # return float(y_np[0, 0])

        # Временный безопасный фолбэк:
        return 0.0

    def extract_features(self, symbol: str) -> list[float]:
        """
        Возвращает список признаков для LightGBM-предсказания.
        """
        recent_1m = self.shared_ws.candles_data.get(symbol, [])
        if len(recent_1m) < 6:
            return []

        old_close = safe_to_float(recent_1m[-6]["closePrice"])
        new_close = safe_to_float(recent_1m[-1]["closePrice"])
        pct_change = (new_close - old_close) / old_close * 100 if old_close > 0 else 0

        vol_change = safe_to_float(recent_1m[-1]["volume"]) - safe_to_float(recent_1m[-6]["volume"])
        oi_change = self.shared_ws.get_oi_change(symbol)

        cvd = self.shared_ws.get_recent_cvd_delta(symbol)  # предполагаем, ты уже это считаешь
        supertrend_trend = self.shared_ws.get_trend_state(symbol)  # ±1 или 0

        return [
            pct_change,
            vol_change,
            oi_change,
            cvd,
            supertrend_trend,
        ]            

    async def optimize_golden_params(self):
        """
        Optimize golden setup parameters daily based on trades_for_training.csv, per symbol and side.
        """
        # Load historical trades
        try:
            df = pd.read_csv("trades_for_training.csv")
        except FileNotFoundError:
            logger.warning("[optimize] trades_for_training.csv not found, skipping optimization")
            return

        # Keep only trade opens and label wins
        # Убеждаемся, что колонка "action" есть, иначе выходим
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

        # Optimize thresholds per symbol and side
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
                    # store symbol-specific
                    self.golden_param_store[(symbol, side)] = params
                    best_params[(symbol, side)] = params
                    logger.info(f"[optimize] {symbol} {side} params: {params}, winrate={best['winrate']:.2f}")

        # Write updated parameters back to CSV symbol-specific
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
        """Рассчитывает ATR(14) по последним 1-минутным свечам."""
        # Используем ~100 свечей для более стабильного ATR
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

    # ЗАМЕНИТЕ СТАРЫЙ _close_position_market НА ЭТОТ
    async def _close_position_market(self, symbol: str, qty_to_close: float, reason: str):
        """Безопасно закрывает ЧАСТЬ или ВСЮ позицию по рынку."""
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

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def manage_open_position(self, symbol: str):
        """
        [V4 - "Няня"] Работает как независимая фоновая задача для одной позиции.
        Самостоятельно получает цену, считает ROI и двигает стоп раз в секунду.
        Автоматически завершается, если позиция закрывается.
        """
        logger.info(f"[manage_open_position] Запущена 'няня' для позиции {symbol}.")
        
        while True:
            await asyncio.sleep(1) # Проверяем состояние раз в секунду

            pos_data = self.open_positions.get(symbol)
            
            # Условие выхода из цикла: позиция была закрыта
            if not pos_data:
                logger.info(f"[manage_open_position] Позиция {symbol} закрыта. 'Няня' завершает работу.")
                return

            # 1. Получаем последнюю цену из общего кэша
            last_price = safe_to_float(pos_data.get("markPrice", 0))
            if last_price <= 0:
                continue # Ждем следующего тика, если цена невалидна

            # 2. Расчет ROI
            avg_price = safe_to_float(pos_data["avg_price"])
            side = pos_data["side"]
            leverage = safe_to_float(pos_data.get("leverage", 10.0))
            if leverage == 0: leverage = 10.0
            
            current_roi = (((last_price - avg_price) / avg_price) * 100 * leverage) if side == "Buy" else (((avg_price - last_price) / avg_price) * 100 * leverage)
            pos_data["pnl"] = current_roi

            # 3. Логика трейлинг-стопа
            if current_roi >= self.trailing_start_pct:
                stop_set_successfully = await self.set_trailing_stop(symbol, avg_price, current_roi, side)
                
                if stop_set_successfully:
                    new_stop_price = self.last_stop_price.get(symbol)
                    if new_stop_price is None: continue

                    last_logged_price = self._last_logged_stop_price.get(symbol, 0.0)
                    log_threshold_pct = 0.1 
                    
                    price_change_pct = 0
                    if last_logged_price > 0:
                        price_change_pct = (abs(new_stop_price - last_logged_price) / last_logged_price) * 100

                    if last_logged_price == 0.0 or price_change_pct > log_threshold_pct:
                        await self.log_trade(
                            symbol=symbol, side=side, avg_price=avg_price, volume=pos_data['volume'],
                            action=f"{self.strategy_mode}_trailing_set", result="success"
                        )
                        self._last_logged_stop_price[symbol] = new_stop_price
                    
                    self.last_trailing_stop_set[symbol] = current_roi

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def set_trailing_stop(self, symbol: str, avg_price: float, pnl_pct: float, side: str) -> bool:
        """
        [V6 - Исправлен расчет для шорта]
        """
        pos = self.open_positions.get(symbol)
        if not pos or safe_to_float(pos.get("volume", 0)) <= 0:
            return False
        
        now = time.time()
        if now - self._last_trailing_ts.get(symbol, 0.0) < 1.5:
            return False

        try:
            pos_idx = pos.get("pos_idx", 1 if side == 'Buy' else 2)
            leverage = safe_to_float(pos.get("leverage", 10.0))
            if leverage == 0: leverage = 10.0
            
            gap_pct = self.trailing_gap_pct
            stop_pct = pnl_pct - gap_pct

            # [ИЗМЕНЕНИЕ] Используем last_price как базу для отступа, а не avg_price
            last_price = safe_to_float(pos.get("markPrice", avg_price))
            if last_price <= 0: return False

            if side.lower() == "buy":
                # Стоп для лонга должен быть НИЖЕ текущей цены
                raw_price = last_price * (1 - gap_pct / 1000)
            else: # sell
                # Стоп для шорта должен быть ВЫШЕ текущей цены
                raw_price = last_price * (1 + gap_pct / 1000)

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
            if e.status_code == 34040:
                self.last_stop_price[symbol] = stop_price
                self._last_trailing_ts[symbol] = now
                return True
            elif e.status_code == 10001:
                logger.warning(f"[TRAILING_STOP] Попытка установить стоп для закрытой позиции {symbol}. Очистка.")
                self._purge_symbol_state(symbol)
                return False
            else:
                logger.warning(f"[TRAILING_STOP] {symbol} ошибка API: {e}")
                return False
        except Exception as e:
            logger.error(f"[TRAILING_STOP] {symbol} критическая ошибка: {e}", exc_info=True)
            return False

    # [НОВАЯ ФУНКЦИЯ] ВНУТРИ КЛАССА TRADINGBOT
    def _purge_symbol_state(self, symbol: str):
        """
        Полностью очищает все внутренние состояния, связанные с символом.
        Вызывается после подтвержденного закрытия позиции.
        """
        logger.debug(f"Полная очистка состояния для символа: {symbol}")
        self.open_positions.pop(symbol, None)
        self.last_trailing_stop_set.pop(symbol, None)
        self.last_stop_price.pop(symbol, None)
        self._last_logged_stop_price.pop(symbol, None)
        self._last_trailing_ts.pop(symbol, None)
        self.pending_orders.pop(symbol, None)
        self.pending_timestamps.pop(symbol, None)
        self.averaged_symbols.discard(symbol)
        # Добавляем в "черный список", чтобы предотвратить немедленное "воскрешение"
        self.recently_closed[symbol] = time.time()

    async def run_daily_optimization(self):
        """
        Schedule optimize_golden_params to run daily at 2:00 America/Toronto.
        """
        tz = pytz.timezone("Europe/Moscow")
        while True:
            now = datetime.now(tz)
            # schedule for next 2am
            next_run = now.replace(hour=4, minute=0, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)
            delay = (next_run - now).total_seconds()
            await asyncio.sleep(delay)
            await self.optimize_golden_params()
            # --- also optimise liquidation thresholds (shared) ---
            try:
                if self.shared_ws and hasattr(self.shared_ws, "optimize_liq_thresholds"):
                    await self.shared_ws.optimize_liq_thresholds()
            except Exception as e:
                logger.warning("[opt_liq] optimisation error: %s", e)

    # -------------- helper: read user_state.json --------------
    def load_user_state(self) -> dict:
        """
        Return the latest entry for this user from user_state.json.
        Falls back to empty dict on any error.
        """
        try:
            with open("user_state.json", "r", encoding="utf-8") as f:
                all_users = json.load(f)
            return all_users.get(self.user_id, {})
        except Exception:
            return {}

    # -------------- live-config helpers -----------------
    def apply_user_settings(self) -> None:
        cfg = self.load_user_state()
        # --- ДОБАВЬТЕ ЭТОТ БЛОК ---
        # Управляем ИИ-стопом. По умолчанию выключен.
        self.ai_stop_management_enabled = cfg.get("ai_stop_management_enabled", False) 

        if "volume" in cfg:
            self.POSITION_VOLUME = safe_to_float(cfg["volume"])
        if "max_total_volume" in cfg:
            self.MAX_TOTAL_VOLUME = safe_to_float(cfg["max_total_volume"])

        # ---- runtime update of trailing-stop settings --------------------
        if "trailing_start_pct" in cfg and isinstance(cfg["trailing_start_pct"], dict):
            self.trailing_start_map.update(cfg["trailing_start_pct"])
        if "trailing_gap_pct" in cfg and isinstance(cfg["trailing_gap_pct"], dict):
            self.trailing_gap_map.update(cfg["trailing_gap_pct"])

        # Пересчитываем текущие значения после всех обновлений
        self.trailing_start_pct = self.trailing_start_map.get(
            self.strategy_mode, DEFAULT_TRAILING_START_PCT
        )
        self.trailing_gap_pct = self.trailing_gap_map.get(
            self.strategy_mode, DEFAULT_TRAILING_GAP_PCT
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

                # ---- runtime update of squeeze-threshold setting ----------------
        if "squeeze_threshold_pct" in cfg:
            self.squeeze_threshold_pct = safe_to_float(cfg["squeeze_threshold_pct"])

                # <-- ДОБАВЬТЕ ЭТОТ БЛОК -->
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
        """
        Lightweight wrapper around telegram_bot.send_message so that we
        can always `await self.notify_user(...)` without worrying whether
        it exists.
        """
        try:
            await telegram_bot.send_message(self.user_id, text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.warning("[notify_user] send error: %s", e)

    async def evaluate_position_with_ml(self, symbol: str):
        try:
            # Сбор фичей — вместо старых вычислений
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

    # ---------------- exposure helper -----------------
    def can_open_position(self, symbol: str, cost: float) -> tuple[bool, str]:
        """
        Возвращает (ok, reason). Гарантирует «одна пара — одна позиция»
        и не превышать MAX_TOTAL_VOLUME.
        """
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


    # ВНУТРИ КЛАССА TRADINGBOT
    async def _open_position(self, symbol: str, side: str):
        cid = new_cid()
        st = log_state(self, symbol)
        logger.info("[EXECUTE][%s] start %s/%s | %s", cid, symbol, side, j(st))
        
        ok, why = self.can_open_position(symbol, 0.0)
        if not ok:
            logger.info("[EXECUTE][%s] denied(primary): %s | %s", cid, why, j(st))
            return None
        """
        [V2 - Унифицированная] Открывает позицию, используя единую
        надежную функцию _calc_qty_from_usd для расчета объема.
        """
            # Централизованная защита от дублей и гонок
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
                self.pending_orders[symbol] = qty * last_price  # стоимость в USDT
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
            # Re-check limits just before sending the order
            order_cost = float(self.pending_orders.get(symbol, 0.0))
            if not self.can_open_position(symbol, order_cost):
                # cancel pending marker and exit
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)
                return None
            await self.place_order_ws(symbol, side, qty, position_idx=pos_idx)
            resp = await self.place_order_ws(symbol, side, qty, position_idx=pos_idx, cid=cid)
            order_id = None
            try:
                # Bybit V5 Unified WS: ожидаем data[0].orderId или data.orderId — адаптируйте при необходимости
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

    # ------------------------------------------------------------------
    # Можно ли сейчас открыть ещё одну сделку по сквизу?
    # ------------------------------------------------------------------

    async def get_total_open_volume(self) -> float:
        """
        Return the total exposure in USDT across all open positions.
        Uses an HTTP request so the value is accurate even when the
        private WebSocket lags. Falls back to cached open_positions
        on any error.
        """
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

    # ---------------- JSON logging of open positions -----------------
    def write_open_positions_json(self) -> None:
        """
        Dump current open positions for this user into OPEN_POS_JSON.
        The resulting file structure:
            {
              "<user_id>": {
                 "<symbol>": {
                    "side": "<Buy|Sell>",
                    "volume": <float>,
                    "avg_price": <float>,
                    "pnl": <float>
                 },
                 ...
              },
              ...
            }
        Only this user's section is rewritten; data for other users is preserved.
        """
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

    # ─── helpers (поместите рядом с import-ами) ─────────────────────────
    async def get_server_time(session, demo: bool) -> float:
        """
        Вернёт (server_time - local_time) в секундах.
        demo=False → api.bybit.com, demo=True → api-demo.bybit.com
        """
        url = ("https://api.bybit.com" if not demo else
            "https://api-demo.bybit.com") + "/v5/market/time"
        try:
            t0 = time.time()
            resp = await asyncio.to_thread(lambda: session._request("GET", url))
            st  = resp["result"]["time"] / 1000     # serverTime → сек
            rt  = (t0 + time.time()) / 2            # RTT/2
            return st - rt
        except Exception:
            return 0.0      # fallback – без сдвига

    async def start(self):
        logger.info(f"[User {self.user_id}] Бот запущен")
        # очистка кэша позиций перед первым REST-запросом
        self.open_positions.clear()
        # Сброс любых временных буферов, чтобы игнорировать устаревшие сигналы
        self.liq_buffers.clear()
        self.pending_timestamps.clear()
        # Cache the running event-loop so we can call run_coroutine_threadsafe from WS callbacks
        self.loop = asyncio.get_running_loop()
        await self.update_open_positions()

        # sequentially initialize private and trade WebSockets
        # sequentially initialize private and trade WebSockets
        await self.setup_private_ws()
        if self.mode == "real":
            await self.init_trade_ws()
        else:
            logger.info("[start] demo mode – trade WebSocket is disabled")
        # ждём, пока shared_ws соберёт данные и выберет пары
        if self.shared_ws and hasattr(self.shared_ws, "ready_event"):
            await self.shared_ws.ready_event.wait()
        #self.pnl_task    = asyncio.create_task(self.pnl_loop())

        await asyncio.sleep(self.warmup_seconds)
        self.warmup_done = True
        logger.info("[warmup] user %s finished (%d s)", self.user_id, self.warmup_seconds)

        # Schedule daily golden setup optimization
        asyncio.create_task(self.run_daily_optimization())

        # Start main loops immediately
        self.market_task = asyncio.create_task(self.market_loop())
        asyncio.create_task(self.reload_settings_loop())
        self.sync_task   = asyncio.create_task(self.sync_open_positions_loop())
        # periodic wallet snapshot logger
        if self.wallet_task is None:
            self.wallet_task = asyncio.create_task(self.wallet_loop())

        # Start periodic cleanup of pending orders
        asyncio.create_task(self.cleanup_pending_loop())

    async def wallet_loop(self):
        """
        Каждые 5 минут обновляет wallet_state.json и пишет лог.
        Формат wallet_state.json: { "<user_id>": {...}, ... }
        """
        while True:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_wallet_balance(accountType="UNIFIED")
                )
                wallet_raw = resp.get("result", {})
                wallet_logger.info("[User %s] %s", self.user_id, json.dumps(wallet_raw, ensure_ascii=False))

                # ---- JSON-снимок для Telegram-бота -------------------------
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

            await asyncio.sleep(300)   # 5 минут

    async def init_trade_ws(self):
        """
        Initialize WebSocket for trade orders with auto-reconnect and authentication retry.
        """
        url = "wss://stream.bybit.com/v5/trade"
        # Loop until we successfully connect and authenticate
        while True:
            try:
                self.ws_trade = await websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=15,
                    open_timeout=10
                )
                # Build and send auth payload according to Bybit v5 WS spec
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
                # Bybit may return either retCode or success flag
                if resp.get("retCode", None) not in (0, None) and not resp.get("success", False):
                    raise RuntimeError(f"WS auth failed: {resp}")
                logger.info("[init_trade_ws] Trade WS connected and authenticated")
                break
            except Exception as e:
                logger.warning(f"[init_trade_ws] connection/auth error: {e}, retrying in 5s...")
                await asyncio.sleep(5)

        # log if market_loop ever stops or crashes
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

    # ─── PRIVATE WS CALLBACK ────────────────────────────────────────────────────────
    async def setup_private_ws(self):
        while True:
            try:
                def _on_private(msg):
                    try:
                        # Логируем всё «сырое» сообщение
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"[PrivateWS] Raw message: {msg}")
                        # отправляем в event loop, если он ещё жив
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
                # Подписываемся на поток по позициям
                self.ws_private.position_stream(callback=_on_private)
                self.ws_private.execution_stream(callback=_on_private)
                # Остальной код инициализации...
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
            # Переподключение WebSocket
            if self.ws_private:
                self.ws_private.exit()
            await self.setup_private_ws()

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def handle_execution(self, msg: dict):
        """
        [V5 - Основной обработчик закрытия] Вызывается при исполнении ордера.
        Конкурирует с handle_position_update за право обработать закрытие первым.
        """
        for exec_data in msg.get("data", []):
            symbol = exec_data.get("symbol")
            if not symbol: continue

            async with self.position_lock:
                # Проверяем, существует ли еще позиция (может, ее уже обработал handle_position_update)
                pos = self.open_positions.get(symbol)
                if not pos: continue

                exec_side = exec_data.get("side")
                if exec_side and pos.get("side") and exec_side != pos.get("side"):
                    if safe_to_float(exec_data.get("leavesQty", 0)) == 0:
                        exit_price = safe_to_float(exec_data.get("execPrice"))
                        pos_volume = safe_to_float(pos.get("volume", 0))
                        pnl_usdt = self._calc_pnl(pos["side"], safe_to_float(pos["avg_price"]), exit_price, pos_volume)
                        position_value = safe_to_float(pos["avg_price"]) * pos_volume
                        pnl_pct = (pnl_usdt / position_value) * 100 if position_value else 0.0

                        logger.info(f"[EXECUTION_CLOSE] {symbol}. PnL: {pnl_usdt:.2f} USDT ({pnl_pct:.2f}%).")

                        self.closed_positions[symbol] = dict(pos)
                        self._purge_symbol_state(symbol)
                        self.write_open_positions_json()

                        asyncio.create_task(self.log_trade(
                            symbol=symbol, side=pos["side"], avg_price=exit_price, volume=pos_volume,
                            action="close", result="closed_by_execution", pnl_usdt=pnl_usdt, pnl_pct=pnl_pct
                        ))

    # ───────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_pnl(entry_side: str, entry_price: float,
                  exit_price: float, qty: float) -> float:
        """
        Long  (entry Buy, exit Sell):   (exit - entry) * qty
        Short (entry Sell, exit Buy):   (entry - exit) * qty
        """
        if entry_side == "Buy":        # long
            return (exit_price - entry_price) * qty
        else:                          # short
            return (entry_price - exit_price) * qty

    # ───────────────────────────────────────────────────────────────
    def _save_trade(self, trade: dict) -> None:
        """Аппендит запись в trades_history.json (массив JSON-объектов)."""
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

    # ЗАМЕНИТЕ ВАШУ ФУНКЦИЮ НА ЭТУ ВЕРСИЮ
    async def adopt_existing_position(self, symbol: str, pos_data: dict):
        """
        [V3 - FIXED]
        Безопасно берет под управление существующую позицию и вызывает
        ЕДИНСТВЕННОГО "исполнителя" set_trailing_stop для установки начального стопа.
        """
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
            
            # Рассчитываем PnL на текущий момент, чтобы передать его в set_trailing_stop
            leverage = safe_to_float(pos_data.get("leverage", 10.0))
            if leverage == 0: leverage = 10.0
            current_roi = (((last_price - avg_price) / avg_price) * 100 * leverage) if side == "Buy" else (((avg_price - last_price) / avg_price) * 100 * leverage)

            # [ИСПРАВЛЕНО] Вызываем вашего "исполнителя", а не несуществующую функцию
            if await self.set_trailing_stop(symbol, avg_price, current_roi, side):
                await self.log_trade(symbol=symbol, side=side, avg_price=avg_price, volume=pos_data['size'], action="adopt_stop_set", result="success")
            else:
                logger.error(f"[ADAPT] {symbol}: Не удалось установить безопасный начальный стоп!")

        async with self.position_lock:
            if symbol in self.open_positions:
                self.open_positions[symbol]['adopted'] = True

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def handle_position_update(self, msg: dict):
        """
        [V7 - Финальная версия] Обрабатывает открытие, обновление и служит надежным
        fallback-механизмом для закрытия позиций (включая ручное).
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
                    
                    self.pending_orders.pop(symbol, None)
                    self.pending_timestamps.pop(symbol, None)

                    avg_price = safe_to_float(p.get("avgPrice") or p.get("entryPrice"))
                    entry_candidate = self.active_trade_entries.pop(symbol, {})
                    comment = entry_candidate.get("comment")

                    self.open_positions[symbol] = {
                        "avg_price": avg_price, "side": side_raw,
                        "pos_idx": 1 if side_raw == 'Buy' else 2,
                        "volume": new_size, "leverage": safe_to_float(p.get("leverage", "1")),
                        "entry_candidate": entry_candidate,
                        "markPrice": avg_price, "pnl": 0.0,
                        "entry_features": self._build_entry_features(symbol)
                    }
                    logger.info(f"[PositionStream] NEW {side_raw} {symbol} {new_size:.3f} @ {avg_price:.6f}")
                    
                    asyncio.create_task(self.log_trade(
                        symbol=symbol, side=side_raw, avg_price=avg_price,
                        volume=new_size, action="open", result="opened",
                        comment=comment
                    ))
                    
                    asyncio.create_task(self.manage_open_position(symbol))
                    self.write_open_positions_json()
                    continue

                # --- Сценарий 2: Обновление существующей позиции ---
                if prev_pos and new_size > 0 and abs(new_size - safe_to_float(prev_pos.get("volume", 0))) > 1e-9:
                    logger.info(f"[PositionStream] {symbol} volume updated: {prev_pos.get('volume')} -> {new_size}")
                    self.open_positions[symbol]["volume"] = new_size
                    self.open_positions[symbol]["avg_price"] = safe_to_float(p.get("avgPrice") or p.get("entryPrice"))
                    self.write_open_positions_json()
                    continue

                # --- [ИЗМЕНЕНИЕ] Сценарий 3: Fallback-закрытие (включая ручное) ---
                if prev_pos and new_size == 0:
                    logger.info(f"[PositionStream] Fallback: {symbol} closed (size=0). Запускаем полную очистку.")
                    
                    # Сохраняем снимок для логгирования
                    snapshot = dict(prev_pos)
                    self.closed_positions[symbol] = snapshot
                    
                    # Полная очистка состояния
                    self._purge_symbol_state(symbol)
                    self.write_open_positions_json()
                    
                    # Логируем закрытие
                    exit_price = safe_to_float(p.get("avgPrice") or snapshot.get("markPrice", snapshot.get("avg_price", 0)))
                    pos_volume = safe_to_float(snapshot.get("volume", 0))
                    entry_price = safe_to_float(snapshot.get("avg_price", 0))
                    pnl_usdt = self._calc_pnl(snapshot.get("side", "Buy"), entry_price, exit_price, pos_volume)
                    pos_value = entry_price * pos_volume
                    pnl_pct = (pnl_usdt / pos_value) * 100 if pos_value else 0.0

                    asyncio.create_task(self.log_trade(
                        symbol=symbol, side=snapshot.get("side", "Buy"), avg_price=exit_price,
                        volume=pos_volume, action="close", result="closed_by_position_stream",
                        pnl_usdt=pnl_usdt, pnl_pct=pnl_pct
                    ))

    async def confirm_position_closing(self, symbol: str) -> bool:
        """
        Проверяет по REST API, действительно ли позиция по символу закрыта.
        """
        try:
            resp = await asyncio.to_thread(
                lambda: self.session.get_positions(category="linear", settleCoin="USDT")
            )
            for pos in resp.get("result", {}).get("list", []):
                if pos.get("symbol") == symbol:
                    size = safe_to_float(pos.get("size", 0))
                    return size == 0
            return True  # Символ не найден — считаем закрытым
        except Exception as e:
            logger.warning(f"[confirm_position_closing] Ошибка проверки позиции {symbol}: {e}")
            return True  # По ошибке — не мешаем обработке

    # ---------------- REST snapshot helpers -----------------
    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def update_open_positions(self):
        """
        [V3 - С проверкой "черного списка"]
        Синхронизирует позиции, но игнорирует "воскрешение" недавно закрытых.
        """
        try:
            response = await asyncio.to_thread(
                lambda: self.session.get_positions(category="linear", settleCoin="USDT")
            )
            if response.get("retCode") != 0:
                logger.warning(f"[update_open_positions] Ошибка API: {response.get('retMsg')}")
                return

            live_positions = {pos["symbol"]: pos for pos in response.get("result", {}).get("list", []) if safe_to_float(pos.get("size", 0)) > 0}
            
            # Адаптация новых (или "воскресших") позиций
            for symbol, pos_data in live_positions.items():
                if symbol not in self.open_positions:
                    # [ИЗМЕНЕНИЕ] Проверяем "черный список"
                    if symbol in self.recently_closed:
                        logger.debug(f"[SYNC] Игнорируем 'воскрешение' {symbol}, т.к. он был недавно закрыт.")
                        continue # Пропускаем этот символ

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

            # Удаление закрытых позиций
            for symbol in list(self.open_positions.keys()):
                if symbol not in live_positions:
                    logger.info(f"[SYNC] Позиция {symbol} больше не активна. Полная очистка состояния.")
                    # [ИЗМЕНЕНИЕ] Вызываем единую функцию очистки
                    self._purge_symbol_state(symbol)
            
            self.write_open_positions_json()
        except Exception as e:
            logger.error(f"[update_open_positions] Критическая ошибка синхронизации: {e}", exc_info=True)

    # ---------------- graceful shutdown -----------------
    async def stop(self) -> None:
        """
        Gracefully cancel background tasks and close websockets.
        """
        try:
            with open(self.training_data_path, "wb") as f:
                pickle.dump(self.training_data, f)
            logger.info(f"[ML] Сохранено {len(self.training_data)} обучающих примеров в {self.training_data_path}.")
        except Exception as e:
            logger.error(f"[ML] Ошибка сохранения обучающих данных: {e}")

        # [ИЗМЕНЕНИЕ] Добавляем _cleanup_task в список
        for name in ("market_task", "sync_task", "pnl_task", "wallet_task", "_cleanup_task"):
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

    # ---------------- qty formatting helper -----------------
    def _format_qty(self, symbol: str, qty: float) -> str:
        """
        Приводит qty к допустимому шагу и убирает мусорные хвосты вида 0.0000002.
        """
        step = self.qty_step_map.get(symbol, 0.001) or 0.001
        # 1) округлить вниз
        qty = math.floor(qty / step) * step
        # 2) выяснить, сколько знаков после точки у шага
        decimals = 0
        if step < 1:
            decimals = len(str(step).split(".")[1].rstrip("0"))
        return f"{qty:.{decimals}f}"

    async def reprice_pending_order(self, symbol: str, last_price: float) -> None:
        """
        Поддерживаем «сквиз-лимитку» на фиксированном отступе ≈1 %
        в противоположную сторону от импульса рынка.
        """
        ctx = self.reserve_orders.get(symbol)
        if not ctx:
            return

        # ограничиваем частоту перестановок
        if time.time() - ctx.get("last_reprice_ts", 0) < SQUEEZE_REPRICE_INTERVAL:
            return

        offset = SQUEEZE_LIMIT_OFFSET_PCT       # 1 %
        tick   = float(DEC_TICK)
        old    = ctx["price"]

        if ctx["action"] == "SHORT":            # лимит ВСЕГДА НИЖЕ рынка
            new = math.floor(last_price * (1 + offset) / tick) * tick
            if new >= last_price:
                new = last_price - tick
        else:                                   # LONG → лимит ВСЕГДА ВЫШЕ рынка
            new = math.ceil(last_price * (1 - offset) / tick) * tick
            if new <= last_price:
                new = last_price + tick

        # если сместились ≤ 1 тик, не дёргаем биржу
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

    # ---------------- 5‑minute aggregation helpers -----------------
    def _aggregate_candles_5m(self, candles: list) -> list:
        """
        Convert a chronologically‑ordered list of 1‑minute candles into
        synthetic 5‑minute candles (OHLCV).  Incomplete trailing blocks are
        ignored.
        Expected keys on each candle dict: openPrice, highPrice, lowPrice,
        closePrice, volume.
        """
        candles = list(candles)      # ← добавьте эту строку
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
        """
        Aggregate a numeric 1‑minute series into 5‑minute buckets.
        method='sum' → sum values inside the bucket
        method='last' → take the last value in the bucket
        """
        if not series:
            return []
        result = []
        full_blocks = len(series) // 5
        for i in range(full_blocks):
            chunk = series[i * 5:(i + 1) * 5]
            if method == "sum":
                result.append(sum(chunk))
            else:
                result.append(chunk[-1])
        return result

    async def handle_liquidation(self, msg):
        """
        Реакция на события ликвидации.

        • игнорируем, пока не завершён warm-up;
        • складываем события в deque <liq_buffers[symbol]>;
        • когда кластер ≥ LIQ_CLUSTER_MIN_USDT и цена в ±1 % от VWAP-кластера,
        определяем доминирующую сторону ликвидаций и открываем
        Market-ордер в противоположную сторону.
        """
        # 0) стартовая защита
        if not getattr(self, "warmup_done", False):
            return
        # Prevent duplicate positions from liquidation logic
        symbol = msg.get("data", [{}])[0].get("s")
        if symbol in self.open_positions:
            logger.info(f"Skipping liquidation trade for {symbol}: position already open")
            return
    
        # 1) всегда работаем со списком событий
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]

        for evt in data:
            # ── распаковка полей Bybit - V5 liquidation stream ────────────
            symbol = evt.get("s")                         # тикер
            qty    = safe_to_float(evt.get("v", 0))       # количество контракта
            side_evt = evt.get("S")                       # 'Buy' | 'Sell'
            price  = safe_to_float(evt.get("p", 0))       # bankruptcy price
            if not symbol or qty <= 0 or price <= 0:
                continue
            value_usdt = qty * price

            # ── обновляем shared_ws, логируем, пишем CSV ──────────────────
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

            # ── торговая логика включается только в нужных режимах ────────
            # обработку запускаем и в смешанном режиме «liq_squeeze»
            if self.strategy_mode not in ("liquidation_only", "full", "liq_squeeze"):
                continue

            # ───────────────────────────────────────────────────────────────
            #   К Л А С Т Е Р Н А Я   Ф И Л Ь Т Р А Ц И Я    v3
            # ───────────────────────────────────────────────────────────────
            # [БЫЛО]
            # now = dt.datetime.utcnow()
            # buf = self.liq_buffers[symbol]
            # buf.append((now, side_evt, value_usdt, price))

            # [СТАЛО - ПРАВИЛЬНО]
            now_ts = time.time() # Используем числовой timestamp
            buf = self.liq_buffers[symbol]
            buf.append((now_ts, side_evt, value_usdt, price))
            
            # 1) вычищаем события старше окна
            # [БЫЛО]
            # cutoff = now - timedelta(seconds=LIQ_CLUSTER_WINDOW_SEC)
            
            # [СТАЛО - ПРАВИЛЬНО]
            cutoff_ts = now_ts - LIQ_CLUSTER_WINDOW_SEC # Простое вычитание секунд
            
            while buf and buf[0][0] < cutoff_ts:
                buf.popleft()

            # 2) суммарный объём кластера
            cluster_val = sum(v for _, _, v, _ in buf)
            if cluster_val < LIQ_CLUSTER_MIN_USDT:
                continue

            # 3) ценовая близость (VWAP кластера)
            cluster_price = sum(p * v for _, _, v, p in buf) / cluster_val
            if abs(price - cluster_price) / cluster_price * 100 > LIQ_PRICE_PROXIMITY_PCT:
                continue

            # 4) сентимент: чей ликвид преобладает?
            long_val  = sum(v for _, s, v, _ in buf if s == "Buy")   # ликв. шортов
            short_val = sum(v for _, s, v, _ in buf if s == "Sell")  # ликв. лонгов
            if long_val == 0 and short_val == 0:
                continue

            # если ликвидировали шорты («Buy»-события) → мы покупаем дип (Buy)
            order_side = "Buy" if short_val > long_val else "Sell"

            # 5) дополнительные стоп-факторы
            if (symbol in self.pending_orders or
                    not self.shared_ws.check_liq_cooldown(symbol)):
                continue

            # ── расчёт объёма и лимитов -----------------------------------
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

                # ── НОВЫЙ единый «затвор» риска ────────────────────────────
                if not await self._risk_check(symbol, order_side, qty_ord, last_price):
                    logger.info("[liq_trade] %s skipped — exposure limit", symbol)
                    continue

            # ── отправляем Market-ордер -----------------------------------
            self.pending_orders.add(symbol)
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
                self.pending_orders.discard(symbol)
            finally:
                # Clear the processed liquidation signal regardless of outcome
                self.pending_signals.pop(symbol, None)

    # ──────────────────────────────────────────────────────────────────────
    # 2. ROUTER: выбираем стратегию «спасения» позиции
    # ──────────────────────────────────────────────────────────────────────
    def _make_signal_key(self,
                     symbol: Any,
                     side: Optional[str] = None,
                     source: Optional[str] = None) -> str:
        # Если вместо строки пришёл словарь, берём из него поле 'symbol'
        if isinstance(symbol, dict):
            symbol_str = symbol.get("symbol") or str(symbol)
        else:
            symbol_str = str(symbol)

        parts = [symbol_str]
        if side:
            parts.append(side)
        if source:
            parts.append(source)

        # Приводим все элементы к строке, чтобы не получить TypeError
        return "_".join(str(p) for p in parts)

    
    async def _rescue_router(self, symbol: str, pnl_pct: float, side: str) -> None:
        """
        Определяет, какие действия предпринять с убыточной позицией.
        Вызывается из evaluate_position().
        """
        # 1) уже в спас-режиме? ───────────────
        rescue = self.active_trade_entries.setdefault("rescue_mode", {})
        if pnl_pct > -1.0:                       # вышли в безопасную зону
            rescue.pop(symbol, None)
            return

        # 2) пороги срабатывания
        DCA_TRG   = -180.0   # % — усреднить
        HEDGE_TRG = -60.0   # % — открыть хедж
        EXIT_TRG  = -230.0  # % — жёсткий выход

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
    # ──────────────────────────────────────────────────────────────────────
    # 3. DCA: усреднение позиции
    # ──────────────────────────────────────────────────────────────────────
    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def _average_down(self, symbol: str, side: str) -> None:
        """
        [V2 - Унифицированный расчет qty] Ставит лимит-ордера для усреднения.
        """
        pos = self.open_positions.get(symbol, {})
        base_qty = safe_to_float(pos.get("volume", 0))
        if base_qty == 0: return

        # [ИЗМЕНЕНИЕ] Расчет через _calc_qty_from_usd
        ticker = self.shared_ws.ticker_data.get(symbol, {})
        bid = safe_to_float(ticker.get("bid1Price"))
        ask = safe_to_float(ticker.get("ask1Price"))
        if not (bid > 0 and ask > 0): return

        # Усредняемся на тот же объем в USDT
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

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def _open_hedge(self, symbol: str, side: str) -> None:
        """
        [V2 - Унифицированный расчет qty] Открывает хеджирующую позицию.
        """
        pos = self.open_positions.get(symbol, {})
        base_qty = safe_to_float(pos.get("volume", 0))
        if base_qty == 0: return

        # [ИЗМЕНЕНИЕ] Расчет через _calc_qty_from_usd
        last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
        if not last_price > 0: return
        
        # Хеджируем 60% от объема в USDT
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

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def _hard_stop(self, symbol: str) -> None:
        """
        [V2 - Унифицированный расчет qty] Закрывает позицию маркет-ордером.
        """
        pos = self.open_positions.get(symbol)
        if not pos: return

        # [ИЗМЕНЕНИЕ] Просто берем текущий объем, не пересчитывая
        stop_qty = safe_to_float(pos.get("volume", 0))
        if not stop_qty > 0: return

        side = "Sell" if pos["side"] == "Buy" else "Buy"
        idx = 2 if side == "Sell" else 1
        try:
            await self.place_order_ws(symbol, side, stop_qty, position_idx=idx, order_type="Market")
            logger.info("[RESCUE-EXIT] %s emergency close %.3f", symbol, stop_qty)
        except Exception as e:
            logger.error("[RESCUE-EXIT] %s close failed: %s", symbol, e)

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def on_ticker_update(self, symbol: str, last_price: float):
        """
        [V2 - Safe] Только обновляет цену позиции, если она есть, правит резервные лимиты
        и пушит OI в историю. Никаких вычислений PnL/трейлинга здесь, чтобы не ловить гонки
        и ошибки вида UnboundLocalError/ZeroDivisionError.
        """
        try:
            # 1) обновляем цену только если позиция есть
            pos = self.open_positions.get(symbol)
            if pos is not None:
                pos["markPrice"] = last_price

            # 2) если есть резервный лимит и позиции ещё нет — подвигаем его
            if symbol in self.reserve_orders and symbol not in self.open_positions:
                asyncio.create_task(self._amend_reserve_limit(symbol, last_price))

            # 3) аккуратно пишем OI в историю (без дубликатов)
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
        """
        Возвращает валидный entry_price.
        1) Если есть в кэше и не старше ttl — берём.
        2) Иначе — REST (get_positions) с rate-limit и бэкоффом.
        """
        pos = self.open_positions.get(symbol)
        now = time.time()
        if not force and pos and safe_to_float(pos.get("avgPrice", 0)) > 0 and (now - self._last_pos_refresh_ts.get(symbol, 0) < ttl):
            return pos["avgPrice"]

        # REST fallback
        try:
            async with self.position_lock:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_positions(category="linear", symbol=symbol)
                )
            lst = resp.get("result", {}).get("list", [])
            # ищем Bybit-стиль позиции (positionIdx 1/2)
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
        """
        Тянем резерв-лимит за рынком:
        – long  → price = last * (1-offset)
        – short → price = last * (1+offset)
        Правим, если сдвиг > 1 тик.
        """
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
        desired = round(desired / step) * step          # округляем к тик-сайзу

        if abs(desired - old_price) < step:             # изменения нет
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
                self.reserve_orders.pop(symbol, None)   # прекращаем трекинг
        except InvalidRequestError as e:
            logger.warning("[SQUEEZE] amend error %s: %s", symbol, e)
            self.reserve_orders.pop(symbol, None)            
    
    # ВНУТРИ КЛАССА TRADINGBOT
    # ЗАМЕНИТЕ ВАШУ ФУНКЦИЮ НА ЭТУ ВЕРСИЮ
    async def cleanup_pending_loop(self):
        """
        [V3 - Corrected]
        Корректно удаляет "зависшие" ордера, но НЕ ТРОГАЕТ комментарии,
        чтобы они дожидались фактического открытия позиции.
        """
        while True:
            await asyncio.sleep(30) # Проверяем каждые 30 секунд
            now = time.time()
            
            # Создаем копию ключей для безопасной итерации
            pending_symbols = list(self.pending_orders.keys())
            
            for symbol in pending_symbols:
                timestamp = self.pending_timestamps.get(symbol, 0)
                
                # Если ордер "завис" более чем на 60 секунд
                if now - timestamp > 60:
                    logger.warning(f"[Pending Cleanup] Ордер для {symbol} завис. Удаление из очереди.")
                    
                    self.pending_orders.pop(symbol, None)
                    self.pending_timestamps.pop(symbol, None)
                    # self.pending_strategy_comments.pop(symbol, None) # <-- [ИСПРАВЛЕНО] Эта строка больше не удаляет комментарий

                    # Дополнительно отменяем "резервные" лимитные ордера, если они есть
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

                    # В режиме full/golden_only/squeeze_only/golden_squeeze/liq_squeeze
                    # запускаем только execute_golden_setup (в нём уже есть и сквиз, и золотой сетап)
                    if self.strategy_mode in ("full", "golden_only", "squeeze_only", "golden_squeeze", "liq_squeeze"):
                        tasks = [
                            asyncio.create_task(
                                self.execute_golden_setup(symbol), name=f"golden-{symbol}"
                            )
                            for symbol in symbols
                        ]
                        logger.info(f"[market_loop] scheduling scan of {len(symbols)} symbols in mode {self.strategy_mode}")

                        if tasks:
                            # Запускаем все стратегии и собираем результаты
                            results = await asyncio.gather(*tasks, return_exceptions=True)
                            # Если какая-то стратегия упала — выводим stacktrace
                            for result, task in zip(results, tasks):
                                if isinstance(result, Exception):
                                    logger.exception(f"[market_loop] exception in {task.get_name()}", exc_info=result)

                    # Heartbeat
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
        """
        [V2] Проверяет лимиты, используя эффективный объем (с учетом pending_orders).
        """
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

    # ------------------------------------------------------------------
    # Per-symbol dynamic Golden-setup thresholds
    # ------------------------------------------------------------------
    async def _get_golden_thresholds(self, symbol: str, side: str) -> dict:
        """
        Возвращает адаптированные пороги Golden-setup для конкретного symbol/side.

        Приоритет:
        1) CSV-override из self.golden_param_store[(symbol, side)]
        2) ML-подстройка (self.ml_inferencer) по текущему срезу фич
        3) Статические default-параметры Buy / Sell
        """
        # --- базовый источник ---
        base = (
            self.golden_param_store.get((symbol, side))
            or self.golden_param_store.get(side)
            or {"period_iters": 3, "price_change": 1.7,
                "volume_change": 200, "oi_change": 1.5}
        )

        # --- ML-тонкая подстройка ---
        try:
            if self.ml_inferencer and self.shared_ws:
                feats = await self.extract_realtime_features(symbol)
                if feats:
                    vec = _np.array([[feats[k] for k in FEATURE_KEYS]], _np.float32)
                    pred = self.ml_inferencer.infer(vec)[0]

                    # record best ML‑tuned thresholds
                    self._record_best_entry(
                        symbol, "golden_setup", side,
                        max(pred) if len(pred) == 3 else float(pred[0]),
                        feats
                    )

                    # 1A) 3‑class classifier  → length == 3
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

                    # 1B) 1‑output regression → treat as expected pnl_pct
                    elif len(pred) == 1:
                        y_hat = float(pred[0])                  # expected pnl %
                        # clamp to [-40 … +40] then scale to [-1 … +1]
                        gain = max(-1.0, min(1.0, y_hat / 40.0))
                        # positive gain → ease thresholds (× 0.6 ‑ 1.0)
                        # negative gain → tighten thresholds (× 1.0 ‑ 1.4)
                        coef = 1.0 - 0.4 * gain                # range 0.6‑1.4
                        base = {**base,
                                "price_change": base["price_change"] * coef,
                                "volume_change": base["volume_change"] * coef,
                                "oi_change":     base["oi_change"] * coef}
        except Exception as e:
            logger.debug("[golden_thr] ML tune failed for %s/%s: %s", symbol, side, e)

        return base

    # ------------------------------------------------------------------
    # Unified entry‑point for three strategies
    # ------------------------------------------------------------------
    async def execute_golden_setup(self, symbol: str):
        """
        A thin router that first checks common pre‑requisites
        and then, depending on `strategy_mode`, delegates to
        (1) squeeze, (2) liquidation, (3) golden‑setup logic.
        The first helper that schedules an order short‑circuits.
        """
        if not await self._gs_prereqs(symbol):
            return

        mode = getattr(self, "strategy_mode", "full")

        # 1) Squeeze
        if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
            if await self._squeeze_logic(symbol):      # True → order placed
                return

        # 2) Liquidations
        if mode in ("full", "liq_squeeze", "liquidation_only"):
            if await self._liquidation_logic(symbol):
                return

        # 3) Classical Golden setup
        if mode in ("full", "golden_only", "golden_squeeze"):
            await self._golden_logic(symbol)           # already handles its own errors

    # ------------------------------------------------------------------
    # Common early‑exit checks
    # ------------------------------------------------------------------
    async def _gs_prereqs(self, symbol: str) -> bool:
        """Return False if the symbol must be skipped right away."""
        if symbol in self.open_positions or symbol in self.pending_orders:
            logger.debug("[GS_SKIP] %s already open or pending", symbol)
            return False
        age = await self.listing_age_minutes(symbol)
        if age < self.listing_age_min:
            logger.debug("[GS_SKIP] %s listing age %.0f < %d", symbol, age, self.listing_age_min)
            return False
        if symbol in self.closed_positions:
            return 
        if symbol in self.failed_orders and time.time() - self.failed_orders[symbol] < 600:
            return False
        if symbol in self.reserve_orders:
            return False
        return True

    # ------------------------------------------------------------------
    # Squeeze strategy — extracted verbatim from the old monolith.
    # Returns True when it schedules/adapts an entry.
    # ------------------------------------------------------------------

    async def _ai_dispatch(self, provider: str, candidate: dict, features: dict) -> dict:
        provider = provider.lower()
        if provider == "openai":
            return await self.evaluate_candidate_with_openai(candidate, features)
        elif provider == "gemini":
            return await self.evaluate_candidate_with_gemini(candidate, features)
        else:  # ollama по умолчанию
            return await self.evaluate_candidate_with_ollama(candidate, features)

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    # async def evaluate_entry_candidate(self, candidate: dict, features: dict):
    #     """
    #     [V30 - Исправлено применение скейлера]
    #     Гарантирует, что данные нормализуются скейлером ПЕРЕД передачей в модель,
    #     что предотвращает аномальные прогнозы.
    #     """
    #     symbol, side, source = candidate['symbol'], candidate['side'], candidate.get('source', 'unknown')
    #     now = time.time()
    #     signal_key = f"{symbol}_{side}_{source}"

    #     if self._evaluated_signals_cache.get(signal_key) and (now - self._evaluated_signals_cache.get(signal_key, {}).get("time", 0) < CACHE_TTL_SEC):
    #         return
    #     self._evaluated_signals_cache[signal_key] = {"status": "pending", "time": now}

    #     try:
    #         # --- ML-фильтр ---
    #         try:
    #             # [ИЗМЕНЕНИЕ] Теперь MLXInferencer сам обработает скейлер
    #             vec = np.array([[safe_to_float(features.get(k, 0.0)) for k in FEATURE_KEYS]], dtype=np.float32)
    #             raw_prediction = float(self.ml_inferencer.infer(vec)[0][0])
                
    #             # "Предохранитель" от аномальных прогнозов оставляем на всякий случай
    #             if not (-1.0 < raw_prediction < 1.0):
    #                 logger.warning(f"[ML_GATE_REJECT] Аномальный прогноз для {symbol}: {raw_prediction}")
    #                 return

    #             leverage = self.leverage
    #             expected_roi = raw_prediction * 100 * leverage

    #             side_check_ok = (side == "Buy" and expected_roi > 0) or (side == "Sell" and expected_roi < 0)
    #             roi_threshold_ok = abs(expected_roi) >= self.ml_gate_abs_roi

    #             if not (side_check_ok and roi_threshold_ok):
    #                 logger.debug(f"[ML_GATE_REJECT] {symbol}/{side} ({source}) | Ожидаемый ROI: {expected_roi:.2f}%, Порог: {self.ml_gate_abs_roi:.2f}%")
    #                 return
                
    #             logger.info(f"[ML_GATE_PASS] {symbol}/{side} ({source}) | Ожидаемый ROI: {expected_roi:.2f}%")

    #         except Exception as e:
    #             logger.warning(f"[ML_GATE_ERROR] Ошибка ML-фильтра для {symbol}: {e}")
    #             return

    #         # --- AI-оценка ---
    #         provider = str(self.ai_provider).lower().strip().replace('о', 'o')
    #         ai_response = await self._ai_call_with_timeout(provider, candidate, features)
            
    #         ai_action = ai_response.get("action", "REJECT")
    #         if ai_action != "EXECUTE":
    #             justification = ai_response.get("justification", "Причина не указана.")
    #             logger.info(f"[AI_REJECT] {symbol}/{side} ({source}) — {justification}")
    #             return
            
    #         # --- Исполнение сделки ---
    #         logger.info(f"[AI_CONFIRM] Сделка {symbol}/{side} ({source}) ОДОБРЕНА. Исполнение...")
    #         candidate['justification'] = ai_response.get("justification", "N/A")
    #         candidate['full_prompt_for_ai'] = ai_response.get("full_prompt_for_ai", "")
    #         await self.execute_trade_entry(candidate, features)

    #     except Exception as e:
    #         logger.error(f"[evaluate_candidate] Критическая ошибка для {symbol}: {e}", exc_info=True)
    #     finally:
    #         self._evaluated_signals_cache.pop(signal_key, None)

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def evaluate_entry_candidate(self, candidate: dict, features: dict):
        """
        [V32 - Исправлен вызов AI]
        Корректно вызывает AI-диспетчер с таймаутом, устраняя AttributeError.
        """
        symbol, side, source = candidate['symbol'], candidate['side'], candidate.get('source', 'unknown')
        now = time.time()
        signal_key = f"{symbol}_{side}_{source}"

        # --- Кэширование и защита от "залипания" ---
        if self._evaluated_signals_cache.get(signal_key) and (now - self._evaluated_signals_cache.get(signal_key, {}).get("time", 0) < CACHE_TTL_SEC):
            return
        self._evaluated_signals_cache[signal_key] = {"status": "pending", "time": now}

        try:
            # --- ML-фильтр ---
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

            # --- AI-оценка с таймаутом ---
            if now < self.ai_circuit_open_until:
                if now >= self._ai_silent_until:
                    logger.debug(f"[AI_SKIP] {symbol}/{side} - circuit open.")
                    self._ai_silent_until = now + 5
                return

            provider = str(self.ai_provider).lower().strip().replace('о', 'o')
            
            # [ИЗМЕНЕНИЕ] Вызываем _ai_dispatch напрямую с таймаутом
            async with self.ai_sem:
                ai_response = await asyncio.wait_for(
                    self._ai_dispatch(provider, candidate, features),
                    timeout=self.ai_timeout_sec
                )
            
            ai_action = ai_response.get("action", "REJECT")
            if ai_action != "EXECUTE":
                justification = ai_response.get("justification", "Причина не указана.")
                logger.info(f"[AI_REJECT] {symbol}/{side} ({source}) — {justification}")
                return
            
            # --- Исполнение сделки ---
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
        """
        [Исполнитель] Принимает одобренного кандидата, рассчитывает объем,
        проверяет риски и размещает ордер.
        """
        import math
        cid = uuid.uuid4().hex[:8]  # корреляционный id для всей цепочки логов этой сделки

        symbol = candidate['symbol']
        side = candidate['side']
        source = candidate.get('source', 'unknown')

        try:
            # ---- sanity: цена ----
            last_price = float(features.get("price", 0) or 0.0)
            if not (last_price > 0):
                logger.warning("[EXECUTE_CANCEL][%s] %s/%s: Невалидная цена: %s",
                            cid, symbol, side, last_price)
                return

            # ---- правило «одна пара — одна позиция» + не плодить pending ----
            if symbol in self.open_positions:
                logger.info("[EXECUTE_DENY][%s] %s/%s: позиция уже открыта", cid, symbol, side)
                return
            if symbol in self.pending_orders:
                logger.info("[EXECUTE_DENY][%s] %s/%s: уже есть pending ордер", cid, symbol, side)
                return

            # ---- объем ----
            volume_usdt = float(candidate.get('volume_usdt', self.POSITION_VOLUME) or 0.0)
            qty = await self._calc_qty_from_usd(symbol, volume_usdt, last_price)
            if not (qty > 0):
                logger.warning("[EXECUTE_CANCEL][%s] %s/%s: Нулевой объем (USDT=%s, price=%s)",
                            cid, symbol, side, volume_usdt, last_price)
                return

            # ---- риск-чеки (включая лимиты из user_state.json, если реализовано внутри) ----
            if not await self._risk_check(symbol, side, qty, last_price):
                logger.info("[EXECUTE_DENY][%s] %s/%s: отклонено _risk_check", cid, symbol, side)
                return

            # ---- бронь pending ----
            comment_for_log = f"JARVIS' OPINION: {candidate.get('justification','')}"
            candidate['comment'] = comment_for_log

            async with self.position_lock:
                self.pending_orders[symbol] = qty * last_price  # стоимость в USDT
                self.pending_timestamps[symbol] = time.time()
                self.active_trade_entries[symbol] = candidate

            logger.info("[EXECUTE][%s] %s/%s: Все проверки пройдены. Отправка ордера... | pending_cost=%.2f",
                        cid, symbol, side, self.pending_orders.get(symbol, 0.0))

            # ---- маршрутизация с таймаутом и логами ----
            pos_idx = 1 if side == "Buy" else 2

            if source == 'squeeze':
                if self.mode == 'real':
                    # WS-вариант
                    try:
                        resp = await asyncio.wait_for(
                            self.adaptive_squeeze_entry_ws(symbol, side, qty, pos_idx, cid=cid),
                            timeout=6.0
                        )
                        logger.info("[ORDER][%s] squeeze_ws ack %s", cid, str(resp)[:400])
                    except asyncio.TimeoutError:
                        logger.error("[ORDER][%s] squeeze_ws timeout -> cleanup pending", cid)
                        raise
                else:
                    # REST/эмуляция
                    try:
                        resp = await asyncio.wait_for(
                            self.adaptive_squeeze_entry(symbol, side, qty, cid=cid),
                            timeout=6.0
                        )
                        logger.info("[ORDER][%s] squeeze_rest ack %s", cid, str(resp)[:400])
                    except asyncio.TimeoutError:
                        logger.error("[ORDER][%s] squeeze_rest timeout -> cleanup pending", cid)
                        raise

            elif source == 'liquidation':
                try:
                    resp = await asyncio.wait_for(
                        self.adaptive_liquidation_entry(symbol, side, qty, pos_idx, cid=cid),
                        timeout=6.0
                    )
                    logger.info("[ORDER][%s] liq ack %s", cid, str(resp)[:400])
                except asyncio.TimeoutError:
                    logger.error("[ORDER][%s] liq timeout -> cleanup pending", cid)
                    raise

            else:
                # дефолт – рыночный ордер через unified
                try:
                    resp = await asyncio.wait_for(
                        self.place_unified_order(symbol, side, qty, "Market", comment=comment_for_log, cid=cid),
                        timeout=6.0
                    )
                    logger.info("[ORDER][%s] unified ack %s", cid, str(resp)[:400])
                except asyncio.TimeoutError:
                    logger.error("[ORDER][%s] unified timeout -> cleanup pending", cid)
                    raise

            # успех — оставляем дальнейшую синхронизацию позиции обработчикам WS/REST
            return

        except Exception as e:
            logger.error("[execute_trade][%s] Критическая ошибка для %s/%s: %s",
                        cid, symbol, side, e, exc_info=True)
        finally:
            # Если на этом этапе позиция уже подтвердилась обработчиком (вы сами снимаете pending) —
            # следующее не помешает: просто не найдёт ключи.
            if symbol in self.pending_orders:
                logger.info("[PENDING][%s] cleanup %s (finalize)", cid, symbol)
            self.pending_orders.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)
            self.active_trade_entries.pop(symbol, None)            

    # async def _ai_call_with_timeout(self, provider: str, candidate: dict, features: dict) -> dict:
    #     now = time.time()
    #     if now < self.ai_circuit_open_until:
    #         return {"action": "REJECT", "justification": "AI temporarily disabled (circuit open)", "full_prompt_for_ai": ""}

    #     try:
    #         async with self.ai_sem:
    #             return await asyncio.wait_for(
    #                 self._ai_dispatch(provider, candidate, features),
    #                 timeout=self.ai_timeout_sec
    #             )
    #     except (asyncio.TimeoutError, RequestsReadTimeout, RequestsConnectionError) as e:
    #         # открыть «рубильник» на минуту
    #         self.ai_circuit_open_until = time.time() + 60
    #         logger.error(f"[AI_TIMEOUT] {provider} завис: {e}. Отключаю ИИ на 60 сек.")
    #         return {"action": "REJECT", "justification": f"AI timeout: {e}", "full_prompt_for_ai": ""}
    #     except Exception as e:
    #         # тоже открываем «рубильник», но на 30 сек
    #         self.ai_circuit_open_until = time.time() + 30
    #         logger.error(f"[AI_FAIL] {provider} упал: {e}", exc_info=True)
    #         return {"action": "REJECT", "justification": f"AI failure: {e}", "full_prompt_for_ai": ""}

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def evaluate_candidate_with_openai(self, candidate: dict, features: dict) -> dict:
        """
        [V4 - Промпт внутри] Отправляет максимально подробный отчет в OpenAI (gpt-4o)
        и получает структурированный JSON-ответ.
        """
        if self.ai_provider != "openai":
            raise RuntimeError("OpenAI provider отключён настройкой ai_provider")
        default_response = {"confidence_score": 0.5, "justification": "Ошибка OpenAI.", "action": "REJECT"}
        if not self.openai_api_key:
            logger.warning("[OpenAI] Ключ API не найден, оценка пропущена.")
            return default_response

        def _format(v, spec): return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"

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

            async with self.gemini_limiter: # Лимитер можно использовать общий
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

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def evaluate_candidate_with_gemini(self, candidate: dict, features: dict) -> dict:
        """
        [V4 - Промпт внутри] Отправляет подробный отчет в Google Gemini.
        """
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

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def evaluate_candidate_with_ollama(self, candidate: dict, features: dict) -> dict:
        """
        [V7 - Продвинутый промпт] Отправляет в Ollama отчет с аномалиями объема и силой тренда.
        """
        from openai import AsyncOpenAI
        default_response = {"confidence_score": 0.5, "justification": "Ошибка локального AI.", "action": "REJECT"}
        prompt = ""
        try:
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            def _format(v, spec): return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"
            m, source = candidate.get('base_metrics', {}), candidate.get('source', 'unknown').replace('_', ' ').title()

            # --- Сбор данных для продвинутого промпта ---
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

    async def _squeeze_logic(self, symbol: str) -> bool:
        """
        [V3] Находит сигнал на сквиз и передает его как кандидата на AI оценку.
        """
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
            
            # [ИСПРАВЛЕНО] Добавляем расчет открытого интереса
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
                'base_metrics': {'pct_5m': pct_5m, 'vol_change_pct': vol_change_pct, 'squeeze_power': squeeze_power}, 'oi_change_pct': oi_change_pct,
                'volume_usdt': self.POSITION_VOLUME
            }

            logger.debug(f"[Signal Candidate] Squeeze: {side} on {symbol}")
            await self.evaluate_entry_candidate(candidate, features)
            
            self.last_squeeze_ts[symbol] = time.time()
            return True

        except Exception as e:
            logger.error(f"[_squeeze_logic] Ошибка анализа сквиза для {symbol}: {e}", exc_info=True)
            return False
        
    # ------------------------------------------------------------------
    # Liquidation strategy — extracted verbatim. Returns True if used.
    # ------------------------------------------------------------------
    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def _liquidation_logic(self, symbol: str) -> bool:
        """
        [V4 - Унифицированный расчет qty]
        """
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

            # [ИЗМЕНЕНИЕ] Унифицированный и надежный расчет объема
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

            # --- Начальные проверки ---
            age = await self.listing_age_minutes(symbol)
            if age < self.listing_age_min:
                return
            if symbol in self.closed_positions:
                return
            if self._squeeze_allowed(symbol) and self.shared_ws.has_5_percent_growth(symbol, minutes=20):
                return
            await self.ensure_symbol_meta(symbol)
            if symbol in self.failed_orders and time.time() - self.failed_orders[symbol] < 600:
                return
            if symbol in self.reserve_orders:
                return

            # --- Сбор и агрегация данных ---
            minute_candles = self.shared_ws.candles_data.get(symbol, [])
            recent = self._aggregate_candles_5m(minute_candles)
            vol_hist_5m = self._aggregate_series_5m(list(self.shared_ws.volume_history.get(symbol, [])), method="sum")
            oi_hist_5m  = self._aggregate_series_5m(list(self.shared_ws.oi_history.get(symbol, [])), method="last")
            cvd_hist_5m = self._aggregate_series_5m(list(self.shared_ws.cvd_history.get(symbol, [])), method="sum")
            
            if not recent:
                return

            # --- Динамические пороги ---
            buy_params  = await self._get_golden_thresholds(symbol, "Buy")
            sell_params = await self._get_golden_thresholds(symbol, "Sell")
            period_iters = max(int(buy_params["period_iters"]), int(sell_params["period_iters"]))

            if (len(recent) <= period_iters or
                len(vol_hist_5m) <= period_iters or
                len(oi_hist_5m)  <= period_iters or
                len(cvd_hist_5m) <= period_iters):
                return
                
            # --- Определение сигнала ---
            action = None
            
            liq_info = self.shared_ws.latest_liquidation.get(symbol, {})
            liq_val  = safe_to_float(liq_info.get("value", 0))
            liq_side = liq_info.get("side", "")
            threshold = self.shared_ws.get_liq_threshold(symbol, 5000)

            price_change_pct, volume_change_pct, oi_change_pct = 0.0, 0.0, 0.0

            # --- Проверка на ПРОДАЖУ (Sell) ---
            sell_period = int(sell_params["period_iters"])
            if len(recent) > sell_period: # Достаточно данных для Sell-периода
                price_change_pct_sell = (safe_to_float(recent[-1]["closePrice"]) - safe_to_float(recent[-1 - sell_period]["closePrice"])) / safe_to_float(recent[-1 - sell_period]["closePrice"]) * 100 if recent[-1 - sell_period]["closePrice"] else 0.0
                volume_change_pct_sell = (safe_to_float(vol_hist_5m[-1]) - safe_to_float(vol_hist_5m[-1 - sell_period])) / safe_to_float(vol_hist_5m[-1 - sell_period]) * 100 if vol_hist_5m[-1 - sell_period] else 0.0
                oi_change_pct_sell = (safe_to_float(oi_hist_5m[-1]) - safe_to_float(oi_hist_5m[-1 - sell_period])) / safe_to_float(oi_hist_5m[-1 - sell_period]) * 100 if oi_hist_5m[-1 - sell_period] else 0.0
                
                if (price_change_pct_sell <= -sell_params["price_change"] and
                    volume_change_pct_sell >= sell_params["volume_change"] and
                    oi_change_pct_sell >= sell_params["oi_change"] and
                    not (liq_side == "Sell" and liq_val >= threshold)):
                    
                    action = "Sell"
                    price_change_pct, volume_change_pct, oi_change_pct = price_change_pct_sell, volume_change_pct_sell, oi_change_pct_sell

            # --- Проверка на ПОКУПКУ (Buy) ---
            if action is None:
                buy_period = int(buy_params["period_iters"])
                if len(recent) > buy_period: # Достаточно данных для Buy-периода
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
                # --- [ИЗМЕНЕНИЕ 1] Сначала извлекаем полный набор фичей для AI ---
                features = await self.extract_realtime_features(symbol)
                if not features:
                    logger.warning(f"[_golden_logic] Не удалось извлечь фичи для {symbol}, сигнал пропущен.")
                    return # Важно завершить выполнение, если фичи не собраны
                
                # Расчет CVD, он нужен для base_metrics
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
                # --- [ИЗМЕНЕНИЕ 2] Передаем 'features' вторым аргументом ---
                await self.evaluate_entry_candidate(candidate, features)
                # После отправки задачи на оценку, работа этой функции завершена
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

        logger.info("[ORDER][%s] send %s %s qty=%s type=%s price=%s comment=%s",
                    cid or "-", symbol, side, qty, order_type, price, (comment or "")[:120])

        cid = new_cid()
        logger.info("[EXECUTE][%s] start %s/%s type=%s qty=%s price=%s comment=%s | %s",
                    cid, symbol, side, order_type, qty, price, comment, j(log_state(self, symbol)))

        pos_idx = 1 if side == "Buy" else 2
        qty_str = self._format_qty(symbol, qty)

        try:
            if self.mode == "real":
                response = await self.place_order_ws(
                symbol, side, qty_str, position_idx=pos_idx,
                price=price, order_type=order_type, cid=cid
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
            else:  # ─── DEMO ────────────────────────────────────────────────
                response = await asyncio.to_thread(
                    lambda: self.session.place_order(
                        category="linear", symbol=symbol, side=side,
                        orderType=order_type, qty=qty_str,
                        price=str(price) if price else None,
                        timeInForce="GTC", positionIdx=pos_idx
                    )
                )

                # ======= НОВОЕ: проверка, что позиция действительно создана ========
                if response.get("retCode") == 0:
                    order_id = response["result"]["orderId"]

                    for _ in range(10):          # ≤ 5 секунд (10 × 0.5s)
                        pos = await asyncio.to_thread(
                            lambda: self.session.get_positions(
                                category="linear", symbol=symbol
                            )
                        )
                        lst = pos.get("result", {}).get("list", [])
                        if lst and float(lst[0].get("size", 0)) > 0:
                            break            # позиция появилась → всё ок
                        await asyncio.sleep(0.5)
                    else:
                        # через 5 с позиции так и нет → отменяем ордер
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
                # ===================================================================

            # ---------- общая часть (успех) ---------------------------------
            if response.get("retCode", -1) != 0:
                raise InvalidRequestError(response.get("retMsg", "Order failed"),
                                        response.get("retCode"), response)

            logger.info("✅ Успешно отправлен ордер: %s %s %s @ %s",
                        side, qty_str, symbol, price or "Market")
            self.pending_strategy_comments[symbol] = comment
            return response

        except InvalidRequestError as e:
            logger.error(f"❌ Ошибка размещения ордера для {symbol} (API Error): {e} (Код: {e.status_code})")
            # [ИСПРАВЛЕНО] Корректная очистка для словаря
            self.pending_orders.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)
            
        except Exception as e:
            logger.critical(f"❌ Критическая ошибка при размещении ордера для {symbol}: {e}", exc_info=True)
            # [ИСПРАВЛЕНО] Корректная очистка для словаря
            self.pending_orders.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)

    # ---- amend ---------------------------------------------------------
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

    # ------------------------------------------------------------------
    #  WebSocket-based adaptive limit entry for squeeze trades
    # ------------------------------------------------------------------

    async def adaptive_liquidation_entry(self, symbol: str, side: str, qty: float, position_idx: int, max_wait_time: int = 15):
        """
        Тактический исполнитель для входа по сигналу от ликвидаций.
        Пытается войти по лучшей цене в течение короткого окна после сигнала.
        """
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

            if side == "Buy": # Ищем самую низкую цену
                if best_price_seen == 0 or last_price < best_price_seen:
                    best_price_seen = last_price
                # Если цена начала отскакивать от минимума, входим
                if last_price > best_price_seen * 1.001: # Отскок на 0.1%
                    logger.info(f"[TACTICAL_LIQ] {symbol}: Обнаружен отскок. Входим в LONG по рынку.")
                    await self.place_unified_order(symbol, side, qty, "Market")
                    entry_made = True
                    break
            
            else: # side == "Sell", ищем самую высокую цену
                if best_price_seen == 0 or last_price > best_price_seen:
                    best_price_seen = last_price
                # Если цена начала падать с максимума, входим
                if last_price < best_price_seen * 0.999: # Откат на 0.1%
                    logger.info(f"[TACTICAL_LIQ] {symbol}: Обнаружен откат. Входим в SHORT по рынку.")
                    await self.place_unified_order(symbol, side, qty, "Market")
                    entry_made = True
                    break

            await asyncio.sleep(0.2) # Проверяем цену 5 раз в секунду

        if not entry_made:
            logger.warning(f"[TACTICAL_LIQ] {symbol}: Окно для входа истекло, выгодный момент не найден. Отмена входа.")
            
            self.pending_orders.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None) 

            # Снимаем блокировку, так как сделка не состоялась
            #self.pending_orders.discard(symbol)
            #self.pending_timestamps.pop(symbol, None)

    # ───────────── adaptive_squeeze_entry_ws (полная) ──────────────
    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def adaptive_squeeze_entry_ws(self,
                                        symbol: str,
                                        side: str,
                                        qty: float,
                                        position_idx: int,
                                        max_entry_timeout: int = 45,
                                        reversal_trigger_pct: float = 0.15) -> bool:
        """
        [V5 - Исправлен бесконечный цикл] Тактический исполнитель для сквиза.
        """
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
                    # [ИЗМЕНЕНИЕ] Убран 'continue', теперь проверка на вход происходит на каждой итерации
                    elif last_price < extreme_price * (1 - reversal_trigger_pct / 100.0):
                        logger.info(f"[TACTICAL_SQUEEZE_WS] {symbol}: Обнаружен откат с пика {extreme_price}. Входим в SHORT через WS.")
                        await self.place_order_ws(symbol, side, qty, position_idx=position_idx, order_type="Market")
                        entry_made = True
                        break
                else: # Buy
                    if extreme_price == 0 or last_price < extreme_price:
                        extreme_price = last_price
                    # [ИЗМЕНЕНИЕ] Убран 'continue'
                    elif last_price > extreme_price * (1 + reversal_trigger_pct / 100.0):
                        logger.info(f"[TACTICAL_SQUEEZE_WS] {symbol}: Обнаружен отскок со дна {extreme_price}. Входим в LONG через WS.")
                        await self.place_order_ws(symbol, side, qty, position_idx=position_idx, order_type="Market")
                        entry_made = True
                        break
                
                await asyncio.sleep(0.2)

        except Exception as e:
            logger.error(f"[TACTICAL_SQUEEZE_WS] Критическая ошибка для {symbol}: {e}", exc_info=True)
            entry_made = False # Считаем, что вход не удался
            
        finally:
            if not entry_made:
                logger.warning(f"[TACTICAL_SQUEEZE_WS] {symbol}: Окно для входа истекло или произошла ошибка. Отмена.")
                # Очищаем pending только если вход не состоялся
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)

        return entry_made
    # ────────────────────────────────────────────────────────────────────

    # ================================================================
    # 2.  REST-вариант (demo / paper)
    # ================================================================
    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
    async def adaptive_squeeze_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        max_entry_timeout: int = 45,
        reversal_trigger_pct: float = 0.15
    ) -> bool:
        """
        [V5 - Исправлен бесконечный цикл] Тактический исполнитель для сквиза (ДЕМО).
        """
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
                    # [ИЗМЕНЕНИЕ] Убран 'continue'
                    elif last_price < extreme_price * (1 - reversal_trigger_pct / 100.0):
                        logger.info(f"[TACTICAL_SQUEEZE_DEMO] {symbol}: Обнаружен откат с пика {extreme_price}. Входим в SHORT через REST.")
                        await self.place_unified_order(symbol, side, qty, "Market")
                        entry_made = True
                        break
                else: # Buy
                    if extreme_price == 0 or last_price < extreme_price:
                        extreme_price = last_price
                    # [ИЗМЕНЕНИЕ] Убран 'continue'
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
    ):
        """
        Отправляет order.create по трейд-сокету и блокирующе ждёт
        ответ *именно* на этот запрос.

        ─ ретраи (до 2) при разрыве сокета;
        ─ retCode 0 ⇒ успех, иначе RuntimeError;
        ─ возвращает resp["data"] (словарь Bybit V5).
        """
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

        for attempt in range(2):          # 1-й вызов + 1 ретрай на реконнект
            try:
                async with self._recv_lock:            # защищаем send/recv
                    await self.ws_trade.send(json.dumps(req))
                    resp = await self._recv_until("order.create", timeout=recv_timeout)

                rc = resp.get("retCode", resp.get("ret_code", 0))
                if rc != 0:
                    raise InvalidRequestError(resp.get("retMsg", "order failed"))
                return resp.get("data", resp)          # ← успех

            except (websockets.ConnectionClosed, asyncio.IncompleteReadError) as e:
                logger.warning(
                    "[place_order_ws] WS closed, reconnecting (attempt %d/2): %s",
                    attempt + 1, e,
                )
                await self.init_trade_ws()             # переподключаемся и ретраим
            except Exception as e:
                logger.error("[place_order_ws] Unexpected error: %s", e)
                raise

        raise RuntimeError("Failed to send order after reconnecting WebSocket")

    # --------------------------------------------------------------------
    # helper: ждём пакет с нужным op
    # --------------------------------------------------------------------
    async def _recv_until(self, expect_op: str, *, timeout: float = 2.0) -> dict:
        """
        Читает из trade-сокета, пропуская нерелевантные ответы,
        пока не встретит пакет с op == expect_op.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(f"WS recv timeout waiting for {expect_op}")

            raw = await self.ws_trade.recv()
            resp = json.loads(raw)
            if resp.get("op") == expect_op:
                return resp
            # остальное — уведомления / ответы на параллельные запросы
            logger.debug("[ws] skipped %s while waiting for %s", resp.get("op"), expect_op)

    # [ИСПРАВЛЕНО] ВНУТРИ КЛАССА TRADINGBOT
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
        closed_manually: bool = False,
        row: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        [V10 - Восстановлено логгирование для finetune]
        Центральный логгер, который принимает комментарий напрямую, отправляет уведомления
        и надежно записывает данные для дообучения AI при закрытии сделки.
        """
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

        # --- Логика для дообучения AI при закрытии сделки ---
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

        # ... (остальной код функции без изменений) ...
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

                

    # ВСТАВЬТЕ ЭТУ ФУНКЦИЮ ВНУТРЬ КЛАССА TRADINGBOT
    async def sync_open_positions_loop(self, interval: int = 5):
        """
        [ВОССТАНОВЛЕННАЯ ФУНКЦИЯ]
        Периодически синхронизирует состояние открытых позиций с биржей,
        вызывая update_open_positions. Служит сетью безопасности для поддержания
        актуального состояния бота.
        """
        while True:
            try:
                # Используем position_lock для безопасного обновления
                async with self.position_lock:
                    await self.update_open_positions()
            except Exception as e:
                logger.error(f"[sync_loop] Критическая ошибка в цикле синхронизации: {e}", exc_info=True)
            
            # Ждем перед следующей проверкой
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
            # ── читаем старый файл (если он есть) ──────────────────────────────
            data = {}
            if os.path.exists(OPEN_POS_JSON):
                with open(OPEN_POS_JSON, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            # ── перезаписываем только свой user_id ──────────────────────────────
            data[str(self.user_id)] = snapshot
            _atomic_json_write(OPEN_POS_JSON, data)
        except Exception as e:
            logger.debug("[save_open_positions_json] %s", e)

    # ------------------------------------------------------------------
    async def _retrain_loop(
        self,
        every_sec: int = 3600,       # как часто будить задачу (сек)
        min_samples: int = 800       # минимум накопленных сэмплов для обучения
    ):
        """
#         Фоновая задача переобучения GoldenNet.

        • Проверяет буфер self.training_data каждые `every_sec` секунд.  
        • Если накоплено ≥ `min_samples`, переобучает сеть (25 эпох) и
          «горяче» подменяет веса + скейлер в self.ml_inferencer.  
        • Модель и скейлер сохраняются в golden_model_v1.pt
          атомарно через временный файл.
        """

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

                # ―― ждём набора достаточного количества сэмплов ――
                if buf_len < min_samples:
                    await asyncio.sleep(every_sec)
                    continue

                # ---------- подготовка данных ----------
                batch = list(self.training_data)
                X = np.array([b["features"] for b in batch], dtype=np.float32)
                y = np.array([b["target"]   for b in batch], dtype=np.float32).reshape(-1, 1)

                scaler = StandardScaler().fit(X)
                X_scaled = scaler.transform(X)

#                 model = GoldenNet(input_size=INPUT_DIM)

                # ---------- обучение ----------
                EPOCHS = 25
                for epoch in range(1, EPOCHS + 1):
#                     opt.zero_grad()
#                     pred = model(X_t)
# #                     loss = lossF(pred, y_t)
                # loss.backward()  # removed (Torch)
#                     opt.step()

                    # логируем каждую 5-ю эпоху (и первую)
                    if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
                        logger.info(
                            "[retrain] epoch %02d/%02d — loss=%.6f",
                            epoch, EPOCHS, 0.0
                        )

                # ---------- сохранение ----------
#                 ckpt = {"model_state": model.cpu().state_dict(), "scaler": scaler}
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    os.replace(tmp.name, CKPT_PATH)
                logger.info("[retrain] model saved → %s", CKPT_PATH)

                # ---------- hot-swap для инференса ----------
                if getattr(self, "ml_inferencer", None):
#                     self.ml_inferencer.model.load_state_dict(model.state_dict())
                    self.ml_inferencer.scaler = scaler
                else:
                    # ML was disabled before – enable it now
                    try:
                        self.ml_inferencer = MLXInferencer(CKPT_PATH)
                        logger.info("[retrain] MLInferencer initialised after first training")
                    except Exception as _e:
                        logger.warning("[retrain] could not init MLInferencer: %s", _e)

                # -------------- housekeeping ------------------------
                for _ in range(buf_len):
                    self.training_data.popleft()

                # refresh per‑symbol OI σ for risk‑gates
                for sym, hist in self.shared_ws.oi_history.items():
                    arr = np.asarray(hist, dtype=float)
                    if len(arr) >= 60:
                        dif = np.diff(arr[-60:]) / arr[-60:-1]
                        self._oi_sigma[sym] = float(np.std(dif))
#                 logger.info("[ML] GoldenNet retrained on %d samples (final loss=%.6f)",
                # (buf_len, 0.0)  # removed
            except Exception:
                # полный stack-trace в лог, чтобы не терять ошибки
                logger.exception("[retrain] unexpected error during training")

            # гарантированная пауза между циклами (даже после ошибки)
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

        # безопасные приведения
        def _to_float(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return default

        result.append({
            "user_id": uid,
            "api_key": data.get("api_key"),
            "api_secret": data.get("api_secret"),
            "gemini_api_key": data.get("gemini_api_key"),
            "openai_api_key": data.get("openai_api_key"),
            "ai_provider": data.get("ai_provider", "ollama"),
            "strategy": data.get("strategy"),
            "volume": _to_float(data.get("volume", 0.0)),
            "max_total_volume": _to_float(data.get("max_total_volume", 0.0)),
            "mode": data.get("mode", "real"),
        })
    return result

# --- Admin snapshot utility -------------------------------------------------
async def make_snapshot() -> str:
    """
    Build a JSON snapshot of equity & open positions for every bot instance.
    Returns the file path.
    """
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
    # Default global parameters
    default_params = {
        "Buy": {
            "period_iters": 4,
            "price_change": 1.7,      # +0.20 % price rise
            "volume_change": 200,      # +50 % volume surge
            "oi_change": 1.5,         # +0.40 % OI rise
        },
        "Sell": {
            "period_iters": 4,
            "price_change": 1.8,      # −0.50 % price drop
            "volume_change": 200,      # +30 % volume surge
            "oi_change": 1.2,         # +0.80 % OI rise
        }
        # "Sell2": {                # альтернативный шорт‑сетап  «либо‑либо»
        # "period_iters": 4,    # 4 последних свечи
        # "price_change": 0.02,# падение цены ≥ %
        # "volume_change": -50, # падение объёма ≥ %
        # "oi_change": -0.01,      # падение OI ≥ %
    #},

    }
    # Attempt to load symbol-specific overrides
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
            # Merge default and overrides (overrides take precedence)
            merged = {**default_params, **overrides}
            return merged
        except Exception as e:
            print(f"[GoldenParams] CSV load error: {e}")
    # Fallback to defaults
    print("[GoldenParams] Using default parameters.")
    return default_params

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

    shared_ws.bot = bots[0]                          # «главный» бот

    # ── запускаем WS СРАЗУ, чтобы пошли данные и ready_event —──────────
    public_ws_task = asyncio.create_task(shared_ws.start())

    # ── первый короткий бэк-филл для BTC / ETH ─────────────────────────
    await shared_ws.backfill_history()

    # ── ждём, пока manage_symbol_selection выберет ликвидные пары ──────
    await shared_ws.ready_event.wait()               # теперь не повиснет

    # ── добираем историю для новых символов ────────────────────────────
    await shared_ws.backfill_history()

    # ───────────────────── Telegram polling ─────────────────────
    # Подключаем все роутеры и запускаем aiogram-polling
    try:
        dp.include_router(router)
    except RuntimeError:
        logger.warning("[run_all] Router already attached, skipping")

    try:
        dp.include_router(router_admin)
    except RuntimeError:
        logger.warning("[run_all] Admin router already attached, skipping")
    # ── запускаем Telegram-поллинг для команд
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
        public_ws_task.cancel()
        telegram_task.cancel()
        await asyncio.gather(
            public_ws_task, telegram_task, *bot_tasks,
            return_exceptions=True
        )

        logger.info("Все задачи остановлены")

    loop = _aio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, functools.partial(_aio.create_task, _shutdown()))

    # Блокируемся, пока жив хотя бы один таск
    await asyncio.gather(public_ws_task, telegram_task, *bot_tasks)

# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        logger.info("Program stopped by user")

# --- Admin command ----------------------------------------------------------
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
    
async def wallet_loop(self):
    """
    Log wallet state every 5 minutes into wallet_state.log.
    """
    while True:
        try:
            # Bybit V5 HTTP endpoint: unified account balance
            resp = await asyncio.to_thread(
                lambda: self.session.get_wallet_balance(accountType="UNIFIED")
            )
            # structure‑agnostic dump
            wallet_logger.info(
                "[User %s] %s",
                self.user_id,
                json.dumps(resp.get("result", {}), ensure_ascii=False),
            )
        except Exception as e:
            wallet_logger.warning(
                "[wallet_loop] user %s error: %s", self.user_id, e
            )
        await asyncio.sleep(300)  # 5 minutes            
# --------------------------------------------------------------------
# Alias for TradingBot so it can be referenced inside PublicWebSocketManager

    # (removed duplicate commented-out cleanup_pending_loop)