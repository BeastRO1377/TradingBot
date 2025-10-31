#!/usr/bin/env python3

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

import coremltools as ct
from lightgbm import Booster
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

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

def _norm_ai_provider(s: str) -> str:
    if not s:
        return "ollama"
    return s.strip().lower().replace("о", "o")


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
# DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ── Compute device selection ──────────────────────────────────────────────
# Default to CPU because large workloads on MPS may seg‑fault (PyTorch <2.8).
# If you really want MPS, start the script with BOT_DEVICE=mps env var.

logger = logging.getLogger(__name__)

# --- Logging level from ENV ---
import os, json as _json
_LOG_LEVEL = os.getenv("LOG_LEVEL", "").upper()
if _LOG_LEVEL:
    try:
        logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    except Exception:
        pass

# --- Debug trace helper ---
TRACE_ALL = bool(int(os.getenv("TRACE_ALL", "0") or "0"))
TRACE_SYMBOL = os.getenv("TRACE_SYMBOL", "")

def dbg_trace(sym: str, phase: str, **kw):
    try:
        if not logger.isEnabledFor(logging.INFO):
            return
        if TRACE_SYMBOL and (str(sym) != str(TRACE_SYMBOL)) and not TRACE_ALL:
            return
        def _to_safe(v):
            try:
                if isinstance(v, (int, float)):
                    return float(v)
                s = str(v)
                return s if len(s) <= 300 else s[:300] + "..."
            except Exception:
                return "<err>"
        payload = {k: _to_safe(v) for k,v in kw.items()}
        logger.debug("[TRACE] %s | %s | %s", sym, phase, _json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass

BOT_DEVICE = os.getenv("BOT_DEVICE", "cpu").lower()

def _probe_mps():
    """
    Returns True if we can allocate & use a tensor on MPS without crashing.
    Some PyTorch‑MPS builds segfault under load; we test with a tiny tensor.
    """
    import torch
    try:
        x = torch.ones(1, device="mps")
        _ = (x * 2).cpu()      # trivial op & sync back
        del x
        torch.mps.empty_cache()
        return True
    except Exception as exc:   # noqa: BLE001
        logger.warning("[ML] MPS probe failed: %s", exc)
        return False

if BOT_DEVICE == "mps":
    if not torch.backends.mps.is_available() or not _probe_mps():
        logger.warning("[ML] Falling back from MPS to CPU due to un‑usable MPS backend")
        BOT_DEVICE = "cpu"

DEVICE = torch.device(BOT_DEVICE)
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

torch.manual_seed(42)

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
class TradingModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super(TradingModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        #return out

        # MLX fallback: отсутствие скейлера/модели не блокирует вход
        try:
            self.use_score_gate = bool(getattr(self, "mlx_scaler", None))
        except Exception:
            self.use_score_gate = False
        if not self.use_score_gate:
            self.min_score = 0.0
            try:
                logger.info("[ML] scaler/model not found — CORE+AI only (no blocking by ML-score)")
            except Exception:
                pass

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
    """Return a robust average/entry price extracted from a raw position dict."""
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
                       multiplier: float | int = 3,
                       device: str | torch.device = "cpu") -> pd.Series:
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

    # ── 2. Use pandas Series (not torch tensors) for pandas_ta compatibility ──
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
        "active_symbols", "_last_selection_ts", "_callback", "_last_resubscribe_ts",
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
        self._last_resubscribe_ts = 0.0
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


    #         pass

    async def manage_symbol_selection(self, *, check_interval: int = 30, enforce_cooldown: bool = True, **kwargs):
        """
        Periodically checks desired symbols set and (re)subscribes with a cooldown.
        Accepts extra **kwargs for backward compatibility.
        """
        try:
            # Ensure cooldown state exists
            if not hasattr(self, "_last_resub"):
                self._last_resub = 0.0
            if not hasattr(self, "RESUB_COOLDOWN"):
                self.RESUB_COOLDOWN = int(os.getenv("WS_RESUB_COOLDOWN", "480"))

            import time, asyncio
            while True:
                try:
                    desired = set(self.select_symbols())
                    current = set(self.subscribed_symbols or [])
                    if desired != current:
                        now = time.time()
                        if not enforce_cooldown or (now - self._last_resub) >= self.RESUB_COOLDOWN:
                            self._last_resub = now
                            await self._resubscribe(list(desired))
                        else:
                            logger.info("[manage_symbol_selection] change muted (Δ=%d, cooldown %ds left)",
                                        len(desired - current) - len(current - desired),
                                        int(self.RESUB_COOLDOWN - (now - self._last_resub)))
                except Exception as inner_e:
                    logger.warning("[manage_symbol_selection] inner error: %s", inner_e, exc_info=True)
                await asyncio.sleep(max(5, int(check_interval)))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("[manage_symbol_selection] fatal: %s", e, exc_info=True)
