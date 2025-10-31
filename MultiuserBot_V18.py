from mimetypes import init
import os, sys, faulthandler

from networkx import sigma
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
faulthandler.enable(file=sys.stderr, all_threads=True)


import datetime as dt
import aiogram
from aiogram.enums import ParseMode

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
import asyncio
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
from typing import Any, Dict, List
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

from concurrent.futures import ThreadPoolExecutor
from websockets.exceptions import ConnectionClosed
import pickle

import uuid

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


 # Telegram‑ID(-ы) администраторов, которым доступна команда /snapshot
ADMIN_IDS = {36972091}   # ← замените на свой реальный ID

# Глобальный реестр всех экземпляров TradingBot (используется для snapshot)
GLOBAL_BOTS: list = []

# # Конфигурация для Apple Silicon M4
# DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

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
SQUEEZE_REPRICE_INTERVAL = 2      # сек между перестановками
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


LISTING_AGE_CACHE_TTL_SEC = 3600          # 1-hour cache
_listing_age_cache: dict[str, tuple[float, float]] = {}
_listing_sem = asyncio.Semaphore(5)       # max 5 concurrent REST calls


# --- REST throttling ---
_TRADING_STOP_SEM = asyncio.Semaphore(3)      # ≤3 одновременных запросов
_LAST_STOP: dict[str, float] = {}             # symbol -> последний поставленный стоп

EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT"}

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
        return out


# === NOTE: Model loading and export should occur in the __main__ block ===
# The following lines are commented out to prevent top-level use of `scaler`
# before it is loaded in __main__.
#
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# input_size = scaler.mean_.shape[0]
# model = TradingModel(input_size=input_size).to(device)
# model.load_state_dict(torch.load('trading_model.pth', map_location=device))
# model.eval()
#
# # Prediction function
# def predict_signal(raw_features: np.array) -> str:
#     # Scale features
#     features_scaled = scaler.transform([raw_features])
#     features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
#     
#     with torch.no_grad():
#         output = model(features_tensor)
#         predicted_class = torch.argmax(output, dim=1).item()
#
#     # Map prediction to signal
#     signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
#     return signal_map.get(predicted_class, 'UNKNOWN')

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


# # Reload model to CPU for export
# model_cpu = TradingModel(input_size=input_size)
# model_cpu.load_state_dict(torch.load('trading_model.pth', map_location='cpu'))
# model_cpu.eval()
#
# # Example input for tracing
# example_input = torch.rand(1, input_size)
#
# # Convert to CoreML
# traced_model = torch.jit.trace(model_cpu, example_input)
# mlmodel = ct.convert(
#     traced_model,
#     inputs=[ct.TensorType(shape=example_input.shape)]
# )
# mlmodel.save("TradingModel.mlmodel")
# print("CoreML model exported!")


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
                await self.backfill_history()
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
                             order_type="Market"):
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
        await self.ws_trade.send(json.dumps(req))
        while True:
            resp = json.loads(await self.ws_trade.recv())
            if resp.get("req_id") == req_id:  # или сверяем op/args
                break
        resp = json.loads(await self.ws_trade.recv())
        if resp["retCode"] != 0:
            raise RuntimeError(f"Order failed: {resp}")
        return resp["data"]  # contains orderId, etc.

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

class MLInferencer:
    def __init__(self, model_path: str = None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.input_dim = len(FEATURE_KEYS)
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        self.model_path = model_path or "golden_model_v18.pt"
        self.scaler = None  # Поддержка scaler (будет расширяться)
        
        if Path(self.model_path).exists():
            self._load_model(self.model_path)
        
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, 64),  # входной размер можно подстроить
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 выхода: Buy, Sell, None
            nn.Softmax(dim=1)
        )
    
    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler = checkpoint.get("scaler")

    def infer(self, features: np.ndarray):
        self.model.eval()
        if self.scaler is not None:
            features = self.scaler.transform(features)
        input_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            pred = logits.cpu().numpy()
        return pred

    def convert_to_coreml(self, export_path: str):
        dummy_input = torch.randn(1, 50)
        traced = torch.jit.trace(self.model.cpu(), dummy_input)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=dummy_input.shape)]
        )
        mlmodel.save(export_path)
        print(f"CoreML model exported: {export_path}")

class GoldenNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# def train_golden_model(training_data, num_epochs=50, batch_size=64, lr=0.001):
#     features = np.array([d['features'] for d in training_data], dtype=np.float32)
#     targets = np.array([d['target'] for d in training_data], dtype=np.float32).reshape(-1, 1)
#     # --- filter out rows that contain NaN or Inf -----------------------
#     mask = ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
#     dropped = len(features) - mask.sum()
#     if dropped:
#         logger.debug("[ML] dropped %d invalid training samples (NaN / Inf)", dropped)
#     if mask.sum() == 0:
#         raise ValueError("train_golden_model: no valid samples remain after NaN/Inf filtering")
#     features = features[mask]
#     targets  = targets[mask]
#     # ------------------------------------------------------------------
#     # 1)  Standard‑scale every feature column → zero‑mean / unit‑var
#     #     This prevents exploding activations which were producing NaNs.
#     # ------------------------------------------------------------------
#     scaler = StandardScaler().fit(features)
#     features = scaler.transform(features).astype(np.float32)

#     # persist for online inference
#     try:
#         joblib.dump(scaler, "scaler.pkl")
#         logger.info("[ML] scaler.pkl saved (%d samples)", len(features))
#     except Exception as _e:
#         logger.warning("[ML] scaler save failed: %s", _e)
#     input_size = features.shape[1]
#     model = GoldenNet(input_size=input_size).to(DEVICE)
#     # Adjust optimizer for lower LR and AdamW
#     optimizer = optim.AdamW(model.parameters(), lr=lr * 0.5)
#     criterion = nn.MSELoss()

#     dataset = torch.utils.data.TensorDataset(torch.tensor(features), torch.tensor(targets))
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     for epoch in range(num_epochs):
#         total_loss = 0
#         # for xb, yb in loader:
#         #     optimizer.zero_grad()
#         #     preds = model(xb)
#         #     loss = criterion(preds, yb)
#         for xb, yb in loader:
#             xb = xb.to(DEVICE, non_blocking=True)     # <<< NEW
#             yb = yb.to(DEVICE, non_blocking=True)     # <<< NEW

#             optimizer.zero_grad()
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             # Protect against gradient explosion
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()
#         if (epoch+1) % 10 == 0:
#             print(f"Epoch {epoch+1}, Loss: {total_loss:.5f}")
#     # attach scaler so that checkpoint includes it
#     model.scaler = scaler
#     return model

# === train_golden_model ===
def train_golden_model(training_data,
                       num_epochs: int = 50,
                       batch_size: int = 64,
                       lr: float = 1e-3):
    """
    Обучаем GoldenNet и одновременно строим StandardScaler.
    • отбрасываем NaN / Inf
    • отсекаем нереальные PnL (|pnl| > 300 %)
    • нормируем target → [-3 … +3]
    • используем SmoothL1Loss и grad-clip
    """
    # ---------- матрица признаков ----------
    feats = np.asarray([d["features"] for d in training_data], dtype=np.float32)
    targ = np.asarray([d["target"]   for d in training_data], dtype=np.float32)

    # 1) убираем строки с некорректными фичами
    m = ~(np.isnan(feats).any(1) | np.isinf(feats).any(1))
    feats, targ = feats[m], targ[m]

    # 2) обрезаем выбросы по PnL
    m = np.abs(targ) <= 300.0          # допустим до ±300 %
    feats, targ = feats[m], targ[m]
    if feats.size == 0:
        raise ValueError("train_golden_model: нет валидных сэмплов")

    # 3) StandardScaler
    scaler = StandardScaler().fit(feats)
    feats = scaler.transform(feats).astype(np.float32)

    # 4) нормализация target
    targ = np.clip(targ / 100.0, -3.0, 3.0).astype(np.float32).reshape(-1, 1)

    ds = TensorDataset(torch.tensor(feats), torch.tensor(targ))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = GoldenNet(input_size=feats.shape[1]).to(DEVICE)
    optim_ = optim.AdamW(model.parameters(), lr=lr * 0.5)
    loss_f = nn.SmoothL1Loss()

    for epoch in range(num_epochs):
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim_.zero_grad()
            out = model(xb)
            loss = loss_f(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim_.step()
            loss_sum += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} – Loss: {loss_sum:.4f}")

    model.scaler = scaler            # чтобы save_model () сохранил scaler
    return model

def train_model_for_ml(
        df: pd.DataFrame,
        scaler_path: str = "scaler.pkl",
        model_path: str = "golden_model_v18.pt",
        mlmodel_path: str = "TradingModel.mlmodel",
        num_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        export_coreml: bool = True,          # ← флаг, чтобы можно было отключить
):
    """
    Обучает GoldenNet на *df*, сохраняет веса + scaler.
    При ``export_coreml=True`` сразу же экспортирует CoreML-модель.
    Требует, чтобы в df присутствовали FEATURE_KEYS и целевой столбец
    (pnl_pct | label | target).
    """
    # ---------- целевая переменная ----------
    if "pnl_pct" in df.columns:
        y = df["pnl_pct"].values.astype("float32")
    elif "label" in df.columns:
        y = df["label"].values.astype("float32")
    else:
        y = df["target"].values.astype("float32")

    X = df[FEATURE_KEYS].values.astype("float32")

    # ---------- очистка ----------
    mask = ~(np.isnan(X).any(1) | np.isinf(X).any(1))
    X, y = X[mask], y[mask]

    # ---------- скейлер ----------
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X).astype("float32")
    joblib.dump(scaler, scaler_path)

    # ---------- данные ----------
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # ---------- модель ----------
    model = GoldenNet(input_size=X.shape[1]).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=lr)
    crit  = nn.SmoothL1Loss()

    for ep in range(num_epochs):
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            loss_sum += loss.item()
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1} – Loss: {loss_sum:.4f}")

    # ---------- сохранение PyTorch ----------
    torch.save({"model_state": model.state_dict(), "scaler": scaler}, model_path)
    logger.info("[ML] PyTorch model saved → %s", model_path)

    # ---------- CoreML-экспорт ----------
    if export_coreml:
        mdl_cpu = GoldenNet(input_size=X.shape[1])          # пустая копия
        mdl_cpu.load_state_dict(model.state_dict())         # переносим веса
        mdl_cpu.eval()

        dummy = torch.randn(1, X.shape[1])                  # правильный размер входа
        traced = torch.jit.trace(mdl_cpu, dummy)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input", shape=dummy.shape)]
        )
        mlmodel.save(mlmodel_path)
        logger.info("[ML] CoreML model exported → %s", mlmodel_path)

    return model


# === Сохраняем / загружаем модель ===
def save_model(model, filename="golden_model.pt"):
    torch.save(
        {"model_state": model.state_dict(),
         "scaler": getattr(model, "scaler", None)},
        filename
    )

def load_model(filename="golden_model.pt"):
    # Must pass input_size explicitly; load_model is only called after training, so input_size known
    # If input_size is unknown here, you must persist it in your checkpoint or pass as argument.
    # Here, as fallback, we use INPUT_DIM.
    model = GoldenNet(input_size=INPUT_DIM).to(DEVICE)
    ckpt = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt["model_state"])
    if "scaler" in ckpt and ckpt["scaler"] is not None:
        joblib.dump(ckpt["scaler"], "scaler.pkl")
    model.eval()
    return model

# ---------------------- TRADING BOT ----------------------
class TradingBot:

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
        "coreml_model", "feature_scaler", "last_retrain", "training_data", "device", "FEATURE_KEYS", 
        "last_squeeze_ts", "squeeze_cooldown_sec", "active_trade_entries", "listing_age_min", "_age_cache",
        "symbol_info", "trade_history_file", "active_trades", "symbol_info", "pending_signals", "max_signal_age",
        "_oi_sigma", "_pending_clean_task", "squeeze_tuner",
    )


    def __init__(self, user_data, shared_ws, golden_param_store):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "real")
        self.listing_age_min = int(user_data.get("listing_age_min_minutes", LISTING_AGE_MIN_MINUTES))
        self.ml_inferencer = MLInferencer("golden_model_v1.pt")
        try:
            self.squeeze_tuner = ct.models.MLModel("squeeze_tuner.mlmodel")
            logger.info("[ML] squeeze_tuner.mlmodel loaded")
        except Exception:
            self.squeeze_tuner = None
            logger.info("[ML] squeeze_tuner absent – using static squeeze thresholds")

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
        # Регистрируемся на обновления тикера для trailing-stop (если shared_ws передан)

        self.history = self.shared_ws.candles_data if self.shared_ws else {}

        if self.shared_ws is not None:
            self.shared_ws.position_handlers.append(self),
            #self.shared_ws.listing_age_min = self.listing_age_min
        self.symbols = shared_ws.symbols if shared_ws else []
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
        self.pending_orders: set[str] = set()
        # per‑symbol rolling buffer of recent liquidation events
        self.liq_buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.pending_strategy_comments: dict[str, str] = {}
        self.last_trailing_stop_set: dict[str, Decimal] = {}
        self.position_lock = asyncio.Lock()
        # Словарь закрытых позиций
        self.closed_positions = {}
        # Task for periodic PnL checks
        self.pnl_task = None
        self.last_seq = {}
        self.wallet_task = None  # periodic wallet snapshot task
        self.last_stop_price: dict[str, float] = {}
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

        self._last_snapshot_ts: dict[str, float] = {}
        # σ(ΔOI) cache for dynamic liquidation guard
        self._oi_sigma: dict[str, float] = defaultdict(float)

        self.reserve_orders: dict[str, dict] = {}
        # housekeeping
        self._pending_clean_task = asyncio.create_task(self._pending_cleaner())

        
        # Инициализация устройства для вычислений
        self.device = DEVICE
        logger.info(f"[ML] Using compute device: {self.device}")

        # ML-атрибуты
        self.model = None
        self.coreml_model = None
        self.feature_scaler = None
        self.load_ml_models()
        self.last_retrain = time.time()
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

    async def _pending_cleaner(self, interval: int = 30):
        """Remove symbols stuck in pending_orders > 120 s"""
        while True:
            now = time.time()
            for sym, ts in list(self.pending_timestamps.items()):
                if now - ts > 120:
                    self.pending_orders.discard(sym)
                    self.pending_timestamps.pop(sym, None)
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Cool-down / permission helpers
    # ------------------------------------------------------------------
    # ---------- helper: convert USDT exposure → correct quantity ----------
    def _calc_qty_from_usd(self, symbol: str, usd_amount: float,
                           price: float | None = None) -> float:
        """
        Превращает желаемое плечо в USDT (usd_amount) в количество
        контракта, округляя ВНИЗ до qtyStep и не ниже minOrderQty.
        Возвращает 0.0, если цену получить не удалось.
        """
        # Убедимся, что meta уже закеширована
        if symbol not in self.qty_step_map or symbol not in self.min_qty_map:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.ensure_symbol_meta(symbol),
                    asyncio.get_event_loop()
                ).result(timeout=5)
            except Exception:
                return 0.0

        step = self.qty_step_map.get(symbol, 0.001)
        min_qty = self.min_qty_map.get(symbol, step)
        price = price or safe_to_float(
            self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
        )
        if price <= 0:
            return 0.0

        raw_qty = 0.98 * usd_amount / price
        qty = math.floor(raw_qty / step) * step
        if qty < min_qty:          # поднять до минимума биржи
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
        """Делегируем CoreML-модели решение: брать / не брать сквиз и какой порог силы поставить."""
        if not getattr(self, "squeeze_tuner", None):
            return                      # модели нет -> статическая логика

        vec = np.array([[feats[k] for k in SQUEEZE_KEYS]], np.float32)
        res = self.squeeze_tuner.predict({"input": vec})

        p_win   = float(res["prob"][0])        # вероятность >0
        rec_thr = float(res["rec_thr"][0])     # рекомендованный миним. squeeze_power

        # ML-вето
        if p_win < 0.55:
            raise RuntimeError("ML veto")

        # Плавное адаптивное смещение порога
        self.squeeze_power_min = 0.8 * self.squeeze_power_min + 0.2 * rec_thr

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


    # ------------------   вставьте целиком вместо старой load_ml_models   ------------------
    def load_ml_models(self):
        """
        • Сначала пробуем CoreML (*.mlmodel) — исключительно для on-device-inference  
        • Затем пытаемся найти PyTorch чекпойнт в одном из стандартных имён
        и подбираем нужный класс модели:
            ─ GoldenNet  – если checkpoint={'model_state': …, 'scaler': …}
            ─ TradingEnsemble – «сырые» state_dict без обёртки
        • Если scaler присутствует в checkpoint — кладём в self.feature_scaler
        """
        # ---------- 1. CoreML -------------------------------------------------
        coreml_path = Path("TradingModel.mlmodel")
        if coreml_path.exists():
            try:
                self.coreml_model = ct.models.MLModel(coreml_path)
                logger.info("[ML] CoreML модель %s загружена", coreml_path.name)
            except Exception as e:
                logger.error("[ML] Ошибка загрузки CoreML (%s): %s", coreml_path.name, e)

        # ---------- 2. PyTorch ------------------------------------------------
        # порядок приоритета: V18 → V1 → legacy
        candidates = ("golden_model_v18.pt",
                    "golden_model_v1.pt",
                    "trading_model.pth")

        self.model = None          # default – «ничего не нашли»
        self.feature_scaler = None

        for fname in candidates:
            ckpt_path = Path(fname)
            if not ckpt_path.exists():
                continue

            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
            except Exception as e:
                logger.error("[ML] %s: чтение не удалось: %s", fname, e)
                continue

            try:
                # ── (а) Новый формат: dict с 'model_state' ─────────────────
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    self.model = GoldenNet(input_size=INPUT_DIM).to(self.device)
                    self.model.load_state_dict(ckpt["model_state"])
                    self.feature_scaler = ckpt.get("scaler")
                # ── (б) Старый raw-state_dict → TradingEnsemble ────────────
                else:
                    self.model = TradingEnsemble(
                        input_size=50, tech_size=20, fund_size=10
                    ).to(self.device)
                    self.model.load_state_dict(ckpt)

                self.model.eval()
                logger.info("[ML] PyTorch модель %s успешно загружена", fname)
                if self.feature_scaler is not None:
                    logger.info("[ML] scaler из %s подцеплен", fname)
                break                        # ← модель найдена → выходим из цикла

            except Exception as e:
                logger.error("[ML] %s: ошибка init/restore — %s", fname, e)

        else:
            # цикл завершён без break → файл не найден или все попытки неудачны
            logger.info("[ML] PyTorch модель не найдена; будет создана при первой тренировке")
# ----------------------------------------------------------------------------------------

    # ──────────────────────────────────────────────────────────────────
    async def extract_realtime_features(self, symbol: str) -> Dict[str, float]:
        """
        Собирает актуальные фичи в словарь **features**.  
        • добавлена диагностика сквиза: SQ_power, SQ_strength*  
        • заполняет отсутствующие ключи нулями, чтобы len(features)==len(FEATURE_KEYS)
        """
# ---- 0.  Делаем всё, чтобы достать цену --------------------------------
        ticker = self.shared_ws.ticker_data.get(symbol)

        if not ticker:
            # fallback-REST: быстрый тикер-снэпшот
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_tickers(category="linear", symbol=symbol)
                )
                if resp and resp.get("result", {}).get("list"):
                    ticker = resp["result"]["list"][0]
                    # положим в кэш, чтобы второй раз не ходить
                    self.shared_ws.ticker_data[symbol] = ticker
            except Exception:
                ticker = None

        if not ticker:
            # fallback-2: используем последнюю подтверждённую свечу
            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if candles:
                last_price = safe_to_float(candles[-1]["closePrice"])
                bid1 = ask1 = last_price
                spread_pct = 0.0
            else:
                logger.debug("[features] %s skipped – нет ни тикера, ни свечей", symbol)
                return None            # вызывающий код должен пропустить symbol
        else:
            last_price = safe_to_float(ticker["lastPrice"])
            bid1       = safe_to_float(ticker["bid1Price"])
            ask1       = safe_to_float(ticker["ask1Price"])
            spread_pct = (ask1 - bid1) / bid1 * 100 if bid1 > 0 else 0.0

        # 1. ───────── базовые фичи ──────────────────────────────────────
        last_price = safe_to_float(self.shared_ws.ticker_data[symbol]["lastPrice"])
        bid1       = safe_to_float(self.shared_ws.ticker_data[symbol]["bid1Price"])
        ask1       = safe_to_float(self.shared_ws.ticker_data[symbol]["ask1Price"])
        spread_pct = (ask1 - bid1) / bid1 * 100 if bid1 > 0 else 0.0

        pct1m  = compute_pct(self.shared_ws.candles_data[symbol], 1)
        pct5m  = compute_pct(self.shared_ws.candles_data[symbol], 5)
        pct15m = compute_pct(self.shared_ws.candles_data[symbol], 15)

        V1m  = sum_last_vol(self.shared_ws.candles_data[symbol], 1)
        V5m  = sum_last_vol(self.shared_ws.candles_data[symbol], 5)
        V15m = sum_last_vol(self.shared_ws.candles_data[symbol], 15)

        OI_now     = safe_to_float(self.shared_ws.latest_open_interest.get(symbol, 0))
        OI_prev1m  = self.shared_ws.oi_history[symbol][-2] if len(self.shared_ws.oi_history[symbol]) >= 2 else 0.0
        OI_prev5m  = self.shared_ws.oi_history[symbol][-6] if len(self.shared_ws.oi_history[symbol]) >= 6 else 0.0
        dOI1m      = (OI_now - OI_prev1m) / OI_prev1m if OI_prev1m > 0 else 0.0
        dOI5m      = (OI_now - OI_prev5m) / OI_prev5m if OI_prev5m > 0 else 0.0

        CVD_now    = self.shared_ws.cvd_history[symbol][-1] if self.shared_ws.cvd_history[symbol] else 0.0
        CVD_prev1m = self.shared_ws.cvd_history[symbol][-2] if len(self.shared_ws.cvd_history[symbol]) >= 2 else 0.0
        CVD_prev5m = self.shared_ws.cvd_history[symbol][-6] if len(self.shared_ws.cvd_history[symbol]) >= 6 else 0.0
        CVD1m      = CVD_now - CVD_prev1m
        CVD5m      = CVD_now - CVD_prev5m

        sigma5m = self.shared_ws._sigma_5m(symbol)

        # 2. ───────── технические индикаторы ────────────────────────────
        df = pd.DataFrame(list(self.shared_ws.candles_data[symbol])[-50:])
        rsi14 = ta.rsi(df["closePrice"], length=14).iloc[-1] if len(df) >= 14 else 50.0
        sma50 = ta.sma(df["closePrice"], length=50).iloc[-1] if len(df) >= 50 else (df["closePrice"].iloc[-1] if len(df) else 0.0)
        ema20 = ta.ema(df["closePrice"], length=20).iloc[-1] if len(df) >= 20 else sma50
        atr14 = ta.atr(df["highPrice"], df["lowPrice"], df["closePrice"], length=14).iloc[-1] if len(df) >= 14 else 0.0

        bb_width = 0.0
        if len(df) >= 20:
            bb = ta.bbands(df["closePrice"], length=20)
            bb_width = bb["BBU_20_2.0"].iloc[-1] - bb["BBL_20_2.0"].iloc[-1]

        supertrend_val  = compute_supertrend(df, period=10, multiplier=3).iloc[-1] if len(df) else False
        supertrend_num  = 1 if supertrend_val else -1
        adx14           = ta.adx(df["highPrice"], df["lowPrice"], df["closePrice"], length=14)["ADX_14"].iloc[-1] if len(df) >= 14 else 0.0
        cci20           = ta.cci(df["highPrice"], df["lowPrice"], df["closePrice"], length=20).iloc[-1] if len(df) >= 20 else 0.0
        macd_block      = ta.macd(df["closePrice"], 12, 26, 9)
        macd_val        = macd_block["MACD_12_26_9"].iloc[-1]  if not macd_block["MACD_12_26_9"].isna().all() else 0.0
        macd_signal     = macd_block["MACDs_12_26_9"].iloc[-1] if not macd_block["MACDs_12_26_9"].isna().all() else 0.0

        avgVol30m  = self.shared_ws.get_avg_volume(symbol, 30)
        oi_hist    = list(self.shared_ws.oi_history[symbol])
        avgOI30m   = sum(oi_hist[-30:]) / max(1, len(oi_hist[-30:]))
        deltaCVD30m = CVD_now - (self.shared_ws.cvd_history[symbol][-31] if len(self.shared_ws.cvd_history[symbol]) >= 31 else 0.0)

        # 3. ───────── Golden-setup блок (unchanged) ─────────────────────
        GS_pct4m  = compute_pct(self.shared_ws.candles_data[symbol], 4)
        GS_vol4m  = sum_last_vol(self.shared_ws.candles_data[symbol], 4)
        GS_dOI4m  = (OI_now - (self.shared_ws.oi_history[symbol][-5] if len(self.shared_ws.oi_history[symbol]) >= 5 else OI_now)) / \
                    max(1, (self.shared_ws.oi_history[symbol][-5] if len(self.shared_ws.oi_history[symbol]) >= 5 else 1))
        GS_cvd4m  = CVD_now - (self.shared_ws.cvd_history[symbol][-5] if len(self.shared_ws.cvd_history[symbol]) >= 5 else CVD_now)
        GS_supertrend_flag = supertrend_num
        GS_cooldown_flag   = int(not self._golden_allowed(symbol))

        # 4. ───────── Squeeze блок + *диагностика*  ─────────────────────
        SQ_pct1m, SQ_pct5m = pct1m, pct5m
        SQ_vol1m, SQ_vol5m = V1m, V5m
        SQ_dOI1m           = dOI1m
        SQ_spread_pct      = spread_pct
        SQ_sigma5m         = sigma5m

        recent_liq_vals = [v for (ts, s, v) in self.liq_buffers[symbol] if time.time() - ts <= 10]
        SQ_liq10s       = sum(recent_liq_vals)
        SQ_cooldown_flag = int(not self._squeeze_allowed(symbol))

        #  **НОВЫЕ диагностические поля**
        #  – «мощность» сквиза (то же, что использует execute_golden_setup)
        SQ_power    = abs(SQ_pct5m) * abs((SQ_vol1m - SQ_vol5m/5) / max(1e-8, SQ_vol5m/5) * 100)
        #  – булево «проходной» фильтр (1 — сигнал проходит все базовые фильтры)
        SQ_strength = int(
            abs(SQ_pct5m) >= self.squeeze_threshold_pct and
            SQ_power      >= self.squeeze_power_min
        )

        # 5. ───────── Liquidation блок (unchanged) ─────────────────────
        buf          = self.liq_buffers[symbol]
        recent_all   = [(ts, s, v) for (ts, s, v) in buf if time.time() - ts <= 10]
        same_side    = [v for (ts, s, v) in recent_all if s == recent_all[-1][1]] if recent_all else []
        LIQ_cluster_val10s   = sum(same_side)
        LIQ_cluster_count10s = len(same_side)
        LIQ_direction        = 1 if (recent_all and recent_all[-1][1] == "Buy") else -1
        LIQ_pct1m, LIQ_pct5m = pct1m, pct5m
        LIQ_vol1m, LIQ_vol5m = V1m, V5m
        LIQ_dOI1m            = dOI1m
        LIQ_spread_pct       = spread_pct
        LIQ_sigma5m          = sigma5m
        LIQ_golden_flag      = int(not self._golden_allowed(symbol))
        LIQ_squeeze_flag     = int(not self._squeeze_allowed(symbol))
        LIQ_cooldown_flag    = int(not self.check_liq_cooldown(symbol))

        # 6. ───────── Временные фичи ────────────────────────────────────
        now_ts       = dt.datetime.now()
        hour_of_day  = now_ts.hour
        day_of_week  = now_ts.weekday()
        month_of_year= now_ts.month

        # 7. ───────── собираем словарь ─────────────────────────────────
        features: Dict[str, float] = {
            # базовые
            "price": last_price,
            "pct1m": pct1m, "pct5m": pct5m, "pct15m": pct15m,
            "vol1m": V1m, "vol5m": V5m, "vol15m": V15m,
            "OI_now": OI_now, "dOI1m": dOI1m, "dOI5m": dOI5m,
            "spread_pct": spread_pct, "sigma5m": sigma5m,
            "CVD1m": CVD1m, "CVD5m": CVD5m,

            # технические
            "rsi14": rsi14, "sma50": sma50, "ema20": ema20,
            "atr14": atr14, "bb_width": bb_width,
            "supertrend": supertrend_num, "cci20": cci20,
            "macd": macd_val, "macd_signal": macd_signal,
            "avgVol30m": avgVol30m, "avgOI30m": avgOI30m,
            "deltaCVD30m": deltaCVD30m, "adx14": adx14,

            # Golden
            "GS_pct4m": GS_pct4m, "GS_vol4m": GS_vol4m, "GS_dOI4m": GS_dOI4m,
            "GS_cvd4m": GS_cvd4m, "GS_supertrend": GS_supertrend_flag,
            "GS_cooldown": GS_cooldown_flag,

            # Squeeze (расширенный набор)
            "SQ_pct1m": SQ_pct1m, "SQ_pct5m": SQ_pct5m,
            "SQ_vol1m": SQ_vol1m, "SQ_vol5m": SQ_vol5m, "SQ_dOI1m": SQ_dOI1m,
            "SQ_spread_pct": SQ_spread_pct, "SQ_sigma5m": SQ_sigma5m,
            "SQ_liq10s": SQ_liq10s, "SQ_cooldown": SQ_cooldown_flag,
            "SQ_power": SQ_power,            # ← новое
            "SQ_strength": SQ_strength,      # ← новое

            # Liquidation
            "LIQ_cluster_val10s": LIQ_cluster_val10s,
            "LIQ_cluster_count10s": LIQ_cluster_count10s,
            "LIQ_direction": LIQ_direction,
            "LIQ_pct1m": LIQ_pct1m, "LIQ_pct5m": LIQ_pct5m,
            "LIQ_vol1m": LIQ_vol1m, "LIQ_vol5m": LIQ_vol5m,
            "LIQ_dOI1m": LIQ_dOI1m, "LIQ_spread_pct": LIQ_spread_pct,
            "LIQ_sigma5m": LIQ_sigma5m,
            "LIQ_golden_flag": LIQ_golden_flag, "LIQ_squeeze_flag": LIQ_squeeze_flag,
            "LIQ_cooldown": LIQ_cooldown_flag,

            # Временные
            "hour_of_day": hour_of_day, "day_of_week": day_of_week,
            "month_of_year": month_of_year,
        }

        # 8. ───────── дополняем отсутствующие ключи нулями ──────────────
        for k in FEATURE_KEYS:
            features.setdefault(k, 0.0)

        # 9. ───────── короткий debug-лог (по желанию) ───────────────────
        logger.debug(
            "[features] %s squeeze: pct5m=%.2f%%  power=%.1f  str=%d  cool=%d",
            symbol, SQ_pct5m, SQ_power, SQ_strength, SQ_cooldown_flag
        )

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


    # def build_trainset_from_ws(self,
    #                            csv_path: str,
    #                            symbol: str,
    #                            future_horizon: int = 3,
    #                            future_thresh: float = 0.005):
    #     """
    #     Формируем trainset.csv из накопленных в shared_ws данных.
    #     """
    #     # 1) Строим DataFrame из свечей
    #     df = pd.DataFrame(self.shared_ws.candles_data[symbol])
    #     # Пусть структура элементов: {'closePrice','openPrice','highPrice','lowPrice','volume','timestamp',…}

    #     # 2) Берём историю OI и CVD из shared_ws
    #     df['OI']  = list(self.shared_ws.oi_history[symbol])[:len(df)]
    #     df['CVD'] = list(self.shared_ws.cvd_history[symbol])[:len(df)]

    #     # 3) Считаем базовые шесть признаков
    #     df['price']         = df['closePrice']
    #     df['price_change']  = df['closePrice'].pct_change().fillna(0)
    #     df['volume_change'] = df['volume'].pct_change().fillna(0)
    #     df['oi_change']     = df['OI'].pct_change().fillna(0)
    #     df['cvd_change']    = df['CVD'].diff().fillna(0)
    #     # strength: перенести сюда ту же формулу, что в торговой логике
    #     df['strength'] = df.apply(
    #         lambda r: self.compute_strength(r['closePrice'], r['volume'], r['OI'], r['CVD']),
    #         axis=1
    #     )

    #     # 4) Генерируем метку по будущей доходности
    #     df['future_return'] = df['closePrice'].shift(-future_horizon) / df['closePrice'] - 1
    #     df['label'] = (df['future_return'] > future_thresh).astype(int)

    #     # 5) Сохраняем только нужные колонки
    #     out = df[['price','price_change','volume_change','oi_change','cvd_change','strength','label']].dropna()
    #     out.to_csv(csv_path, index=False)
    #     logger.info(f"[DatasetBuilder] Сформирован {csv_path}, shape={out.shape}")

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
        try:
            # Extract flat feature dict
            features = await self.extract_realtime_features(symbol)
            
            # Фильтр для Golden Setup: только при ADX ≥25 и RSI14 ≤80
            if self.strategy_mode == "golden_only":
                if features.get("adx14", 0.0) < 25.0 or features.get("rsi14", 0.0) > 80.0:
                    return "HOLD"
            # Build a feature vector in the predefined order
            vector = [features[k] for k in FEATURE_KEYS]
            tensor = torch.tensor([vector], dtype=torch.float32).to(self.device)

            if self.coreml_model:
                # CoreML expects a single MLMultiArray named "input"
                input_arr = np.array(vector, dtype=np.float32).reshape(1, -1)
                prediction = self.coreml_model.predict({"input": input_arr})
                probs = prediction["output"]
                actions = ['BUY', 'SELL', 'HOLD']
                return actions[int(np.argmax(probs))]
            else:
                # PyTorch inference
                with torch.no_grad():
                    output = self.model(tensor)  # assumes model signature input_size = len(FEATURE_KEYS)
                    probs = output.cpu().numpy()[0]
                    actions = ['BUY', 'SELL', 'HOLD']
                    return actions[int(np.argmax(probs))]
        except Exception as e:
            logger.error(f"[ML] Prediction error: {e}")
            return "HOLD"

    # async def log_trade_for_ml(self, symbol: str, entry_data: dict, exit_data: dict):
    #     try:
    #         features = await self.extract_realtime_features(symbol)
    #         vector = [features[k] for k in FEATURE_KEYS]
    #         pnl = ((exit_data['price'] - entry_data['price']) / entry_data['price']) * 100.0 \
    #             if entry_data['side'] == "Buy" else \
    #             ((entry_data['price'] - exit_data['price']) / entry_data['price']) * 100.0

    #         record = {
    #             'features': vector,
    #             'label': 0 if pnl < 0 else 1 if pnl > 1 else 2  # 0=loss, 1=profit, 2=neutral
    #         }

    #         self.training_data.append(record)

    #         if time.time() - self.last_retrain > 3600 and len(self.training_data) > 100:
    #             asyncio.create_task(self.retrain_models())
    #     except Exception as e:
    #         logger.error(f"[ML] Trade logging error: {e}")


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

    # --- Обучение ---
    # def train_model(self, csv_path, out_model_path, epochs=20, batch_size=64):
    #     # Если trainset.csv нет или пуст — соберём его из shared_ws
    #     if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
    #         self.build_trainset_from_ws(csv_path, symbol=self.symbols[0] if self.symbols else "", future_horizon=3, future_thresh=0.005)
    #     dataset = GoldenDataset(csv_path)
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #     #model = GoldenNet().to(DEVICE)
    #     model = GoldenNet(input_size=INPUT_DIM).to(DEVICE)
    #     criterion = nn.BCEWithLogitsLoss()
        
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)

    #     for epoch in range(epochs):
    #         losses = []
    #         for X_batch, y_batch in dataloader:
    #             X_batch = X_batch.to(DEVICE, non_blocking=True)
    #             y_batch = y_batch.to(DEVICE, non_blocking=True)

    #             optimizer.zero_grad()
    #             outputs = model(X_batch).squeeze(1)  # -> shape [batch]
    #             if y_batch.ndim > 1:                 # squeeze only when needed
    #                 y_batch = y_batch.squeeze(1)

    #             loss = criterion(outputs, y_batch.float())
    #             loss.backward()
    #             optimizer.step()
    #             losses.append(loss.item())
    #         print(f"Epoch {epoch+1} - Loss: {np.mean(losses):.4f}")

    #     torch.save(model.state_dict(), out_model_path)
    #     print(f"✅ Model saved: {out_model_path}")


    # ─────────────────────────────────────────────────────────
    def train_model(self,
                    csv_path: str = "trainset.csv",
                    num_epochs: int = 30,
                    batch_size: int = 64,
                    lr: float = 1e-3,
                    model_path: str = "golden_model_v18.pt"):
        """
        Локальное пере-обучение GoldenNet на trainset.csv.
        • подхватывает scaler из scaler.pkl (если есть)
        • автоматически выбирает целевой столбец
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)

        # --- выбираем target-колонку гибко ------------------
        if "pnl_pct" in df.columns:
            y = df["pnl_pct"].values.astype("float32")
        elif "label" in df.columns:             # 0/1/2 (loss/neutral/profit)
            y = df["label"].values.astype("float32")
        else:
            y = df["target"].values.astype("float32")

        X = df[FEATURE_KEYS].values.astype("float32")

        # ---- sanity cleanup (NaN / Inf) --------------------
        mask = ~(np.isnan(X).any(1) | np.isinf(X).any(1))
        X, y = X[mask], y[mask]

        # ---- scaler ---------------------------------------
        scaler = _safe_load_scaler("scaler.pkl")
        scaler = scaler.fit(X)            # fit/update
        X = scaler.transform(X).astype("float32")
        joblib.dump(scaler, "scaler.pkl")

        ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = GoldenNet(input_size=X.shape[1]).to(self.device)
        optim_ = optim.AdamW(model.parameters(), lr=lr)
        loss_f = nn.SmoothL1Loss()

        for epoch in range(num_epochs):
            loss_sum = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim_.zero_grad()
                pred = model(xb)
                loss = loss_f(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optim_.step()
                loss_sum += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1} – Loss: {loss_sum:.4f}")

        # ---- сохраняем чек-пойнт (+ scaler) ---------------
        torch.save({"model_state": model.state_dict(),
                    "scaler": scaler},
                   model_path)
        self.model = model.eval()         # грузим в текущий бот


    async def retrain_models(self):
        try:
            if len(self.training_data) < 100:
                return

            logger.info("[ML] Starting model retraining...")

            X = [record['features'] for record in self.training_data]
            y = [record['label'] for record in self.training_data]

            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y, dtype=torch.long).to(self.device)

            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(10):
                optimizer.zero_grad()
                outputs = self.model(X_t)
                loss = criterion(outputs, y_t)
                loss.backward()
                optimizer.step()
                logger.info(f"[ML] Epoch {epoch+1}, Loss: {loss.item():.4f}")

            torch.save(self.model.state_dict(), "trading_model.pth")
            self.convert_to_coreml()

            _append_trades_unified({
                "timestamp": datetime.utcnow().isoformat(),
                "note": "Training completed",
                "loss": float(loss.item())
            })

            logger.info("[ML] Model retraining completed successfully")
            self.last_retrain = time.time()
        except Exception as e:
            logger.error(f"[ML] Retraining failed: {e}")

    def convert_to_coreml(self):
        try:
            example_input = torch.rand(1, len(FEATURE_KEYS)).to(self.device)
            traced_model = torch.jit.trace(self.model, example_input)

            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=example_input.shape)],
                outputs=[ct.TensorType(name="output")],
                compute_units=ct.ComputeUnit.ALL
            )

            mlmodel.save("TradingModel.mlmodel")
            self.coreml_model = mlmodel
            logger.info("[ML] Model converted to CoreML format")
        except Exception as e:
            logger.error(f"[ML] CoreML conversion failed: {e}")


    # ---------------- ML model loading ----------------
    def load_model(self):
        model_path = f"models/golden_model_{self.user_id}.pt"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            logger.info(f"[ML] Model loaded for user {self.user_id}")
        else:
            logger.info(f"[ML] No pretrained model for user {self.user_id}")

    def save_model(self):
        model_path = f"models/golden_model_{self.user_id}.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"[ML] Model saved for user {self.user_id}")

    def append_training_sample(self, features: list[float], target: float):
        self.training_data.append({"features": features, "target": target})

    async def retrain_model(self):
        if len(self.training_data) < 300:
            logger.warning("[ML] Not enough data to retrain")
            return
        self.model = train_golden_model(self.training_data)
        self.save_model()

    def predict_entry_quality(self, features: list[float]):
        self.model.eval()
        with torch.no_grad():
            tensor_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            pred = self.model(tensor_input)
            return pred.item()


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
        if "volume" in cfg:
            self.POSITION_VOLUME = safe_to_float(cfg["volume"])
        if "max_total_volume" in cfg:
            self.MAX_TOTAL_VOLUME = safe_to_float(cfg["max_total_volume"])

        # ---- runtime update of trailing-stop settings --------------------
        if "trailing_start_pct" in cfg:
            if isinstance(cfg["trailing_start_pct"], dict):
                self.trailing_start_map.update(cfg["trailing_start_pct"])
            else:  # backward compatibility: single number
                self.trailing_start_map[self.strategy_mode] = safe_to_float(cfg["trailing_start_pct"])

            self.trailing_start_pct = self.trailing_start_map.get(
                self.strategy_mode,
                DEFAULT_TRAILING_START_PCT,
            )

        if "trailing_gap_pct" in cfg:
            if isinstance(cfg["trailing_gap_pct"], dict):
                self.trailing_gap_map.update(cfg["trailing_gap_pct"])
            else:
                self.trailing_gap_map[self.strategy_mode] = safe_to_float(cfg["trailing_gap_pct"])

            self.trailing_gap_pct = self.trailing_gap_map.get(
                self.strategy_mode,
                DEFAULT_TRAILING_GAP_PCT,
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
            features = self.extract_features(symbol)
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

    def extract_features(self, symbol: str) -> np.ndarray:
        # Тут полная сборка твоих feature из всех источников (tickers, oi, volume, cvd, liq и пр.)
        # Временно поставим dummy-заглушку, ты сможешь сюда вставить свою полную генерацию
        return np.random.randn(50)
    
    async def _open_position(self, symbol: str, side: str):
        # Простейшее открытие через стандартную твою функцию ордеров
        try:
            qty = self.calc_qty(symbol)
            last_price = safe_to_float(
                self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
            )
            if not await self._risk_check(symbol, side, qty, last_price):
                return        # лимиты нарушены – выйти без ордера
            pos_idx = 1 if side == "Buy" else 2
            await self.place_order_ws(symbol, side, qty, pos_idx)
            logger.info(f"[ML Order] {symbol}: {side} qty={qty}")
        except Exception as e:
            logger.warning(f"[ML Order] Failed to place {symbol}: {e}")

    def calc_qty(self, symbol):
        price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 1))
        qty = self.POSITION_VOLUME / price
        step = self.qty_step_map.get(symbol, 0.001)
        return math.floor(qty / step) * step


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
        self.pnl_task    = asyncio.create_task(self.pnl_loop())

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



    # async def setup_private_ws(self):
    #     def _on_private(msg):
    #         logger.info(f"[PrivateWS] Raw message: {msg}")
    #         # Отправляем корутину в основной loop, чтобы избежать ошибки "no current event loop in thread"
    #         asyncio.run_coroutine_threadsafe(
    #             self.route_private_message(msg),
    #             self.loop
    #         )

    #     self.ws_private = WebSocket(
    #         testnet=False,
    #         demo=self.mode == "demo",
    #         channel_type="private",
    #         api_key=self.api_key,
    #         api_secret=self.api_secret,
    #         ping_interval=20,
    #         ping_timeout=10,
    #         restart_on_error=True,
    #         retries=200
    #     )
    #     # Subscribe to all position updates (unified margin).
    #     # For Bybit V5 private WS the topic is simply "position".
    #     self.ws_private.position_stream(callback=_on_private)
    #     # Инициализация локального состояния позиций: однократный REST-снапшот
    #     await self.update_open_positions()
    #     # Даем время на установку WS и получение первых событий
    #     await asyncio.sleep(1)
    #     # Subscribe to liquidation events for additional protection
    #     self.latest_liquidation = {}
        
    #     logger.info("[setup_private_ws] Подключение к private WebSocket установлено")

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
        except Exception as e:
            logger.error(f"[route_private_message] Ошибка: {e}", exc_info=True)
            # Переподключение WebSocket
            if self.ws_private:
                self.ws_private.exit()
            await self.setup_private_ws()

    # ──────────────────────────────────────────────────────────────────
    def _on_execution(self, msg: dict) -> None:
        """
        Обрабатываем каждое исполнение:
        • наращиваем/сокращаем позицию в self.active_trades
        • когда размер стал 0 → считаем PnL и пишем в trades_history.json
        """
        for row in msg["data"]:
            symbol     = row["symbol"]
            side       = row["side"]        # 'Buy' | 'Sell'
            qty        = float(row["execQty"])
            price      = float(row["execPrice"])
            fee        = abs(float(row.get("fee", 0)))
            ts         = int(row["execTime"])          # мс

            pos = self.active_trades.get(symbol)

            # ─── 1. позиция только открывается ───────────────────────
            if pos is None:
                self.active_trades[symbol] = {
                    "side"     : side,             # какая СТОРОНА открывается
                    "qty"      : qty,
                    "avg_price": price,
                    "fees"     : fee,
                    "open_ts"  : ts,
                }
                continue

            # ─── 2. добавление к открытой позиции той же стороны ─────
            if pos["side"] == side:
                new_qty   = pos["qty"] + qty
                pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * qty) / new_qty
                pos["qty"]       = new_qty
                pos["fees"]     += fee
                pos["execPrice"] = price
                continue

            # ─── 3. закрываем (частично/полностью) противоположной стороной ──
            close_qty = min(pos["qty"], qty)
            realised  = self._calc_pnl(
                entry_side=pos["side"],
                entry_price=pos["execPrice"],
                exit_price=price,
                qty=close_qty,
            )
            pos["qty"]   -= close_qty
            pos["fees"]  += fee

            # если всё закрыто → фиксация трейда
            if pos["qty"] == 0:
                trade = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol"      : symbol,
                    "side"        : pos["side"],
                    "qty"         : close_qty,
                    "entry_price" : pos["avg_price"],
                    "exit_price"  : price,
                    "pnl"         : realised,
                    "pnl_pct"     : realised / (pos["avg_price"] * close_qty) * 100,
                    "fees"        : pos["fees"],
                    "open_time"   : pos["open_ts"],
                    "close_time"  : ts,
                }
                self._save_trade(trade)
                del self.active_trades[symbol]


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


# ─── ОБРАБОТКА POSITION_STREAM ──────────────────────────────────────────────────
    async def handle_position_update(self, msg):
        """
        Обработка обновлений позиций из приватного WS:
        – логируем «сырое» сообщение,
        – фильтруем события с size>0 (открытие) или уже существующие (закрытие),
        – пропускаем дубликаты по seq,
        – при открытии сохраняем в словарь, планируем оценку, логируем + пушим нотификацию,
        – при закрытии логируем + пушим нотификацию,
        – при изменении объёма — обновляем только volume.
        """
        # 1) Нормализуем data → всегда список
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]

        # 2) Оставляем новые size>0 или уже открытые позиции
        data = [
            p for p in data
            if safe_to_float(p.get("size", 0)) > 0 or p.get("symbol") in self.open_positions
        ]
        if not data:
            return

        try:
            for position in data:
                symbol = position["symbol"]

                # 3) Пропускаем дубликаты/старые по seq
                seq = position.get("seq", 0)
                if seq <= self.last_seq.get(symbol, 0):
                    continue
                self.last_seq[symbol] = seq

                # 4) Извлекаем ключевые поля
                side_raw   = position.get("side", "")  # 'Buy' или 'Sell'
                avg_price  = (safe_to_float(position.get("avgPrice"))
                              or safe_to_float(position.get("entryPrice")))
                new_size   = safe_to_float(position.get("size", 0))
                status     = position.get("status", "").lower()  # Добавляем проверку статуса!
                open_int   = self.shared_ws.latest_open_interest.get(symbol, 0.0)
                prev       = self.open_positions.get(symbol)
                lastClose  = self._on_execution.price(symbol)

                # 5) Открытие позиции
                if prev is None and new_size > 0 and side_raw:
                    # Сохраняем в open_positions
                    self.open_positions[symbol] = {
                        "avg_price": avg_price,
                        "side":      side_raw,
                        "pos_idx":   position.get("positionIdx", 1),
                        "volume":    new_size,
                        "amount":    safe_to_float(position.get("positionValue")),
                        "stop_loss": safe_to_float(position.get("stopLoss", 0)),
                        "lastClose": safe_to_float(self._on_execution.price(symbol))   # <---- добавлено

                    }
                    # Запомним entry_data для ML
                    entry_data = {
                        "price": avg_price,
                        "side": side_raw,
                        "volume": new_size,
                        "symbol": symbol,
                        "lastClose": lastClose,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self.active_trade_entries[symbol] = entry_data

                    # всё остальное как раньше:
                    self.reserve_orders.pop(symbol, None)
                    self.open_positions[symbol]["pnl"] = safe_to_float(position.get("unrealisedPnl", 0))
                    self.write_open_positions_json()
                    self.ws_opened_symbols.add(symbol)
                    self.ws_closed_symbols.discard(symbol)

                    logger.info(f"[PositionStream] Scheduling evaluate_position for {symbol}")
                    asyncio.create_task(self.evaluate_position(position))

                    asyncio.create_task(self.log_trade(
                        symbol=symbol,
                        side=side_raw,
                        avg_price=avg_price,
                        volume=new_size,
                        open_interest=open_int,
                        action="open",
                        result="opened"
                    ))

                    comment = self.pending_strategy_comments.pop(symbol, "")
                    msg = (f"🟢 Открыта {side_raw.upper()}-позиция {symbol}: объём {new_size} @ {avg_price}")
                    if comment:
                        msg += f"\nКомментарий: <i>{comment}</i>"
                    asyncio.create_task(self.notify_user(msg))

                    self.pending_orders.discard(symbol)
                    continue

                # 6) Закрытие позиции
                if prev is not None and new_size == 0 and status == "closed":
                    # Берём entry_data, которое мы сохраняли при открытии
                    entry_data = self.active_trade_entries.get(symbol)
                    if entry_data:
                        exit_price = avg_price  # avg_price – это цена, на которой закрылась позиция
                        exit_data = {
                            "price": exit_price,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        # Вызываем логгер для ML (помечаем метку внутри этой функции)
                        await self.log_trade_for_ml(symbol, entry_data, exit_data)
                        # Удаляем из активных, чтобы не дублировать
                        del self.active_trade_entries[symbol]

                    # Логируем само закрытие и закидываем позицию в closed_positions
                    logger.info(f"[PositionStream] Закрытие позиции {symbol}, PnL={position.get('unrealisedPnl')}")
                    self.closed_positions[symbol] = {
                        **prev,
                        "closed_pnl":  position.get("unrealisedPnl"),
                        "closed_time": position.get("updatedTime")
                    }
                    self.ws_closed_symbols.add(symbol)
                    self.ws_opened_symbols.discard(symbol)

                    # Асинхронный лог по сделке
                    asyncio.create_task(self.log_trade(
                        symbol=symbol,
                        side=prev["side"],
                        avg_price=prev["avg_price"],
                        volume=prev["volume"],
                        open_interest=open_int,
                        action="close",
                        result="closed",
                        closed_manually=False
                    ))
                    # Уведомление
                    asyncio.create_task(self.notify_user(
                        f"⏹️ Закрыта {prev['side'].upper()}-позиция {symbol}: "
                        f"объём {prev['volume']} @ {prev['avg_price']}"
                    ))

                    # Удаляем позицию из активных
                    del self.open_positions[symbol]
                    self.averaged_symbols.discard(symbol)
                    self.pending_orders.discard(symbol)
                    continue

                # 7) Обновление объёма существующей позиции
                if prev and new_size > 0 and new_size != prev.get("volume"):
                    logger.info(f"[PositionStream] Обновление объёма {symbol}: "
                                f"{prev['volume']} → {new_size}")
                    self.open_positions[symbol]["volume"] = new_size
                    self.open_positions[symbol]["pnl"] = safe_to_float(position.get("unrealisedPnl", 0))
                    self.open_positions[symbol]["stop_loss"] = safe_to_float(position.get("stopLoss", 0))  # <--- обновляем stop_loss при любом изменении

                    self.write_open_positions_json()
                    asyncio.create_task(self.evaluate_position({
                        "symbol": symbol,
                        "size":   str(new_size),
                        "side":   prev["side"]
                    }))

        except Exception as e:
            logger.error(f"[handle_position_update] Ошибка обработки: {e}", exc_info=True)
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            await self.update_open_positions()

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

    # ---------------- exposure helper -----------------
    async def can_open_position(self, est_cost: float) -> bool:
        """
        Return True if opening a position costing *est_cost* USDT would
        keep total exposure ≤ MAX_TOTAL_VOLUME.
        """
        try:
            total = await self.get_total_open_volume()
        except Exception:
            total = 0.0
        return (total + est_cost) <= self.MAX_TOTAL_VOLUME

    # ---------------- REST snapshot helpers -----------------
    async def update_open_positions(self):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: self.session.get_positions(category="linear", settleCoin="USDT")
                ),
                timeout=10  # Таймаут 10 секунд
            )

            if response.get("retCode") != 0:
                logger.warning(f"[update_open_positions] Ошибка получения позиций: {response.get('retMsg')}")
                return

            # REST snapshot of active positions
            new_positions_raw = response.get("result", {}).get("list", [])
            new_positions = {
                pos["symbol"]: pos
                for pos in new_positions_raw
                if safe_to_float(pos.get("size", 0)) > 0
            }

            # Remove positions no longer active, whether closed via REST or WS
            for symbol in list(self.open_positions.keys()):
                if symbol not in new_positions:
                    # Log how the position was closed
                    if symbol in self.ws_closed_symbols:
                        logger.info(f"[update_open_positions] Позиция {symbol} закрыта через WS")
                    else:
                        logger.info(f"[update_open_positions] Позиция {symbol} закрыта через REST")
                    # Log closed position to trades_unified.csv
                    old = self.open_positions[symbol]
                    entry_price = safe_to_float(old.get("avg_price"))
                    side        = old.get("side")
                    qty         = safe_to_float(old.get("volume"))
                    exit_price  = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0))
                    pnl = ((exit_price - entry_price) / entry_price * 100.0) if side == "Buy" \
                          else ((entry_price - exit_price) / entry_price * 100.0)
                    _append_trades_unified({
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol":    symbol,
                        "side":      side,
                        "entry_price": entry_price,
                        "exit_price":  exit_price,
                        "qty":          qty,
                        "pnl":          pnl,
                    })
                    # Remove from open_positions and clear WS-closed marker
                    self.open_positions.pop(symbol, None)
                    self.ws_closed_symbols.discard(symbol)

            # Merge new REST positions, preserving WS state
            for symbol, pos in new_positions.items():
                if symbol not in self.open_positions:
                    self.open_positions[symbol] = {
                        "avg_price": safe_to_float(pos.get("entryPrice") or pos.get("avgPrice")),
                        "side":      pos.get("side", ""),
                        "pos_idx":   pos.get("positionIdx", 1),
                        "leverage":   safe_to_float(pos.get("leverage", 0)),
                        "volume":    safe_to_float(pos.get("size", 0)),
                        "amount":    safe_to_float(pos.get("positionValue", 0))
                    }

            logger.info(f"[update_open_positions] Текущие позиции: {list(self.open_positions.keys())}")

        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"[update_open_positions] Timeout/Error: {e}")


    # ---------------- graceful shutdown -----------------
    async def stop(self) -> None:
        """
        Gracefully cancel background tasks and close websockets.
        """
        for name in ("market_task", "sync_task", "pnl_task", "wallet_task"):
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
            now = dt.datetime.utcnow()
            buf = self.liq_buffers[symbol]                    # deque(maxlen=200)
            buf.append((now, side_evt, value_usdt, price))    # (ts, side, val, px)

            # 1) вычищаем события старше окна
            cutoff = now - timedelta(seconds=LIQ_CLUSTER_WINDOW_SEC)
            while buf and buf[0][0] < cutoff:
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
                    raise InvalidRequestError(resp.get("retMsg", "order rejected"))

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

    async def evaluate_position(self, position):
        async with self.limiter:
            """
            Process position for PnL and trailing stops based on latest ticker prices.
            """
            #logger.debug("EVALUATE POSITION запустился")
            symbol = position.get("symbol")
            #logger.info(f"[evaluate_position] Start for {symbol}: "
            #            f"position={position}, "
            #            f"open_positions={self.open_positions.get(symbol)}")

            # Retrieve stored position data
            if symbol not in self.open_positions:
                return     # позиция уже закрыта → тихо выходим
            # support both underscore and camelCase keys
            data = self.open_positions.get(symbol)
            avg_price = safe_to_float(data.get("avg_price") or data.get("avgPrice", 0))
            pos_idx   = data.get("pos_idx") or data.get("positionIdx") or 1
            prev_vol  = safe_to_float(data.get("volume") or data.get("size", 0))

            # For compatibility: keep data["avg_price"], data["pos_idx"], data["volume"] available
            # (normalize values in-place if not present)
            if "avg_price" not in data:
                data["avg_price"] = avg_price
            if "pos_idx" not in data:
                data["pos_idx"] = pos_idx
            if "volume" not in data:
                data["volume"] = prev_vol

            # Compute current values
            size = safe_to_float(position.get("size", 0))
            side = data["side"].lower()
            # avg_price, prev_vol already set above

            # Get latest price
            last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0))


            # Fallback: если lastPrice из тикеров нулевой, берём closePrice из последней свечи
            if last_price <= 0.0:
                recent = self.shared_ws.candles_data.get(symbol, []) if self.shared_ws else []
                if recent:
                    candle_close = safe_to_float(recent[-1]["closePrice"])
                    logger.info(f"[evaluate_position] fallback candle price for {symbol}: {candle_close}")
                    last_price = candle_close

            # Compute PnL
            pnl = (last_price - avg_price) if side == "buy" else (avg_price - last_price)
            # v12 style: use normal floats and round to 5 decimals
            try:
                pnl_pct = round((pnl / avg_price) * 1000, 7)
            except ZeroDivisionError:
                pnl_pct = 0.0
            # Conditional logging: only log if notable profit or major loss

            if pnl > 0:
                logger.info(f"[evaluate_position] {symbol}: last_price={last_price}, avg_price={avg_price}, pnl={pnl}, pnl_pct={pnl_pct}%")

            if pnl <= AVERAGE_LOSS_TRIGGER:
                logger.info(f"[evaluate_position] {symbol}: last_price={last_price}, avg_price={avg_price}, pnl={pnl}, pnl_pct={pnl_pct}%")


                # # Ensure averaging is done on the full position
                # volume_to_add = size  # Default to full position size
                # current_volume = safe_to_float(self.open_positions.get(symbol, {}).get("volume", 0))
                # if size < current_volume:
                #     volume_to_add = current_volume  # Correct averaging size to match total open volume

                # current_volume = safe_to_float(self.open_positions.get(symbol, {}).get("volume", 0))
                # if volume_to_add + current_volume > self.max_allowed_volume:
                #     logger.warning(f"[evaluate_position] Skipping {symbol}: attempted volume %.2f exceeds max %.2f",
                #                 volume_to_add + current_volume, self.max_allowed_volume)
                #     return
                # try:
                #     # use original side casing for order submission
                #     orig_side = data.get("side", "Buy")
                #     if symbol in self.averaged_symbols:
                #         logger.info(f"[evaluate_position] Пропуск усреднения: {symbol} уже усреднён")
                #         return

                #     # сразу помечаем как усреднённый, чтобы избежать гонки
                #     self.averaged_symbols.add(symbol)

                #     if self.mode == "real":
                #         await self.place_order_ws(symbol, orig_side, volume_to_add, position_idx=pos_idx)
                #     else:
                #         resp = await asyncio.to_thread(lambda: self.session.place_order(
                #             category="linear",
                #             symbol=symbol,
                #             side=orig_side,
                #             orderType="Market",
                #             qty=str(volume_to_add),
                #             timeInForce="GTC",
                #             positionIdx=pos_idx
                #         ))
                #         if resp.get("retCode", 0) != 0:
                #             raise InvalidRequestError(resp.get("retMsg", "order rejected"))
                #     logger.info(f"[evaluate_position] Averaging executed for {symbol}: added volume {volume_to_add}")
                #     # обновляем внутренний объём
                #     self.open_positions[symbol]["volume"] += volume_to_add
                # except RuntimeError as e:
                #     msg = str(e)
                #     self.averaged_symbols.discard(symbol)

                #     # Handle insufficient balance error from Bybit
                #     if "ab not enough for new order" in msg:
                #         logger.warning(f"[evaluate_position] averaging skipped for {symbol}: insufficient balance ({msg})")
                #         return
                #     # Re-raise or log other runtime errors
                #     logger.error(f"[evaluate_position] averaging failed for {symbol}: {e}", exc_info=True)
                # except Exception as e:
                #     logger.error(f"[evaluate_position] averaging failed for {symbol}: {e}", exc_info=True)

                # -------- AVERAGE
                
                # if (self.averaging_enabled
                #         and pnl_pct <= AVERAGE_LOSS_TRIGGER
                #         and symbol not in self.averaged_symbols):
                #     logger.info(
                #         "[average] %s drawdown %.2f %% ≤ %.2f %% — averaging",
                #         symbol, pnl_pct, AVERAGE_LOSS_TRIGGER
                #     )

                #     volume_to_add = size            # берём объём = текущей позиции
                #     current_volume = safe_to_float(self.open_positions[symbol]["volume"])
                #     if volume_to_add + current_volume > self.max_allowed_volume:
                #         logger.warning(
                #             "[average] skip %s: %.2f > max %.2f",
                #             symbol, volume_to_add + current_volume, self.max_allowed_volume
                #         )
                #     else:
                #         try:
                #             orig_side = data.get("side", "Buy")          # Buy / Sell
                #             self.averaged_symbols.add(symbol)            # защитимся от гонки
                #             if self.mode == "real":
                #                 await self.place_order_ws(symbol, orig_side,
                #                                         volume_to_add, position_idx=pos_idx)
                #             else:
                #                 resp = await asyncio.to_thread(lambda: self.session.place_order(
                #                     category="linear",
                #                     symbol=symbol,
                #                     side=orig_side,
                #                     orderType="Market",
                #                     qty=str(volume_to_add),
                #                     timeInForce="GTC",
                #                     positionIdx=pos_idx
                #                 ))
                #                 if resp.get("retCode", 0) != 0:
                #                     raise InvalidRequestError(resp.get("retMsg", "order rejected"))
                #             logger.info(
                #                 "[average] %s averaged: +%.3f (new total %.3f)",
                #                 symbol, volume_to_add,
                #                 current_volume + volume_to_add
                #             )
                #             self.open_positions[symbol]["volume"] += volume_to_add
                #         except Exception as e:
                #             self.averaged_symbols.discard(symbol)
                #             logger.error("[average] %s failed: %s", symbol, e, exc_info=True)

            self.open_positions[symbol]["volume"] = size
            self.pending_orders.discard(symbol)

            # ── Trailing-stop ───────────────────────────────────────────
            threshold = self.trailing_start_pct
            last_pct  = self.last_trailing_stop_set.get(symbol, 0.0)

            if pnl_pct >= threshold and pnl_pct > last_pct:
                stop_set = await self.set_trailing_stop(symbol, avg_price, pnl_pct, side)
                if stop_set:
                    open_int = self.shared_ws.latest_open_interest.get(symbol, 0.0)
                    await self.log_trade(
                        symbol=symbol,
                        side=data["side"],
                        avg_price=avg_price,
                        volume=size,
                        open_interest=open_int,
                        action="trailing_set",
                        result="set",
                        closed_manually=False
                    )
                    self.last_trailing_stop_set[symbol] = pnl_pct


    async def on_ticker_update(self, symbol, last_price):
        # --- IGNORE TICKERS, КОГДА ПОЗИЦИИ НЕТ ────────────────────────
        if symbol not in self.open_positions:      # <--- добавлен guard
            return
        # --- Reprice pending Squeeze limit orders on every ticker update ---
        # Use REST amend if available, or WS amend if no open position yet
        await self.reprice_pending_order(symbol, last_price)
        if symbol in self.reserve_orders and symbol not in self.open_positions:
            asyncio.create_task(self._amend_reserve_limit(symbol, last_price))

        # If there's an open position, update mark price, OI history, and evaluate PnL
        if symbol in self.open_positions:
            pos_data = self.open_positions[symbol]

            # 1) Update mark price
            pos_data["markPrice"] = last_price

            # 2) Keep a rolling OI history (used by Golden Setup)
            oi_val = self.shared_ws.latest_open_interest.get(symbol)
            if oi_val is not None:
                hist = self.shared_ws.oi_history.setdefault(symbol, deque(maxlen=500))
                hist.append(float(oi_val))

            # 3) Trigger position evaluation (PnL, trailing stops, averaging)
            position = {
                "symbol": symbol,
                "size": str(pos_data.get("volume", 0)),
                "side": pos_data.get("side", "")
            }
            await self.evaluate_position(position)


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
    
    async def cleanup_pending_loop(self):
        while True:
            now = time.time()
            to_remove = [s for s, ts in self.pending_timestamps.items() if now - ts > 60]
            for symbol in to_remove:
                # 1) снимаем «висящий» сигнал
                if symbol in self.pending_orders and symbol not in self.open_positions:
                    self.pending_orders.discard(symbol)
                    self.pending_strategy_comments.pop(symbol, None)
                    logger.info("[pending_cleanup] Удалён сигнал по %s — истёк таймаут", symbol)

                # 2) отменяем резерв-лимитку, если она ещё жива  (доп. логика)
                info = self.reserve_orders.pop(symbol, None)
                if info:
                    try:
                        await asyncio.to_thread(
                            lambda: self.session.cancel_order(
                                category="linear",
                                symbol=symbol,
                                orderId=info["orderId"])
                        )
                        logger.info("[pending_cleanup] Отменён резерв-лимит по %s", symbol)
                    except Exception as e:
                        logger.warning("[pending_cleanup] Не удалось отменить резерв %s: %s", symbol, e)

                # 3) и обязательно очищаем timestamp
                self.pending_timestamps.pop(symbol, None)

            await asyncio.sleep(10)

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
        Проверка лимитов и запрет хеджа.

        •  POSITION_VOLUME — максимум на один символ;
        •  MAX_TOTAL_VOLUME — общий лимит счёта;
        •  нельзя открывать противоположную сторону, если уже есть позиция;
        •  нельзя ставить сигнал, если по символу есть pending-order;
        •  усреднение разрешено только в ту же сторону и в пределах POSITION_VOLUME.
        """
        if symbol in EXCLUDED_SYMBOLS or last_price <= 0:
            return False

        # — существующая позиция —
        existing = self.open_positions.get(symbol)
        if existing:
            existing_side = (existing.get("side") or "").lower()
            existing_vol  = safe_to_float(existing.get("volume", 0))

            if existing_side and existing_side != side.lower():
                logger.info("[risk] Skip %s — open %s, tried %s", symbol, existing_side, side.lower())
                return False

            if (existing_vol + qty) * last_price > self.POSITION_VOLUME * 1.02:
                logger.info("[risk] Skip %s — averaging %.0f exceeds POSITION_VOLUME %.0f",
                            symbol, (existing_vol + qty) * last_price, self.POSITION_VOLUME)
                return False
        else:
            if qty * last_price > self.POSITION_VOLUME:
                logger.info("[risk] Skip %s — %.0f > POSITION_VOLUME %.0f",
                            symbol, qty * last_price, self.POSITION_VOLUME)
                return False

        if symbol in self.pending_orders:
            logger.info("[risk] Skip %s — pending order exists", symbol)
            return False

        if not await self.can_open_position(qty * last_price):
            logger.info("[risk] Skip %s — total exposure limit", symbol)
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
                    vec = np.array([[feats[k] for k in FEATURE_KEYS]], np.float32)
                    prob_sell, prob_hold, prob_buy = self.ml_inferencer.infer(vec)[0]

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
        except Exception as e:
            logger.debug("[golden_thr] ML tune failed for %s/%s: %s", symbol, side, e)

        return base


    async def execute_golden_setup(self, symbol: str):
        #logger.info(f"GOLDEN SETUP STARTED")
        # Prevent duplicate or hedged positions
        if symbol in self.open_positions or symbol in self.pending_orders:
            reason = "cooldown not elapsed"  # или другая динамическая причина
            logger.info(f"[GS_SKIP] {symbol} skipped golden setup: {reason}")
            return
        
        try:
            """
            Анализ и исполнение «golden setup» для symbol:
            – вычисляем Δ цены/объёма/OI,
            – при сигнале (Buy/Sell) рассчитываем qty и выставляем рыночный ордер,
            – логируем через log_trade с передачей avg_price и volume,
            – подтверждаем открытие через REST или обрабатываем ошибки.
            """
            golden_enabled = self.strategy_mode in ("full", "golden_only", "golden_squeeze")
            # Skip symbols manually closed to prevent re-entry
            # ── quick liquidity metric: bid/ask spread (always initialised) ──
            ticker     = self.shared_ws.ticker_data.get(symbol, {})
            bid_px     = safe_to_float(ticker.get("bid1Price"))
            ask_px     = safe_to_float(ticker.get("ask1Price"))
            spread_pct = (ask_px - bid_px) / bid_px * 100 if bid_px > 0 else 0.0
            age = await self.listing_age_minutes(symbol)
            if age < self.listing_age_min:
                logger.info("[listing_age] %s %.0f min < %d min – skip",
                            symbol, age, self.listing_age_min)
                return
            if symbol in self.open_positions:
                logger.info(f"Skipping golden setup for {symbol}: position already open")
                return
            if symbol in self.closed_positions:
                return
            # Пропускаем символы с ростом 5% за 20 минут
            if self._squeeze_allowed(symbol) \
            and self.shared_ws.has_5_percent_growth(symbol, minutes=20):
                logger.debug(f"[GoldenSetup & SQUEEZE] ВНИМАНИЕ {symbol}: рост ≥3% за 20 минут")
                #return

            # 1. Пропускаем недавно провалившиеся символы
            # ensure we know the correct qtyStep / minQty
            await self.ensure_symbol_meta(symbol)
            if symbol in self.failed_orders and time.time() - self.failed_orders[symbol] < 600:
                return
            # 2. Не трогаем уже открытые или ожидающие открытие
            if symbol in self.open_positions or symbol in self.pending_orders:
                return
            # 2b. Если уже есть резерв‑лимитка, ждём её исполнения/отмены
            if symbol in self.reserve_orders:
                return

            # ---- SQUEEZE STRATEGY v2 (обновлено на базе V16) --------------------
            recent_deque = self.shared_ws.candles_data.get(symbol, [])
            if len(recent_deque) < 6:          # нужно ≥ 5 завершённых минут
                volchg = Decimal("0"); oichg = Decimal("0")
                return

            recent = list(recent_deque)        # ← единственная конверсия

            ticker_data = self.shared_ws.ticker_data.get(symbol, {})
            last_price = safe_to_float(ticker_data.get("lastPrice", 0)) \
                or safe_to_float(recent[-1]["closePrice"])
            if last_price <= 0:
                logger.info(f"[SQUEEZE] {symbol} skipped: no price")
                return

            # Δ-цены за последние 5 мин
            old_close = safe_to_float(recent[-6]["closePrice"])
            new_close = safe_to_float(recent[-1]["closePrice"])
            limit_price = last_price  # используем текущую рыночную цену для расчёта qty
            if old_close <= 0:
                return
            if old_close > 0:
                pct_5m = (new_close - old_close) / old_close * 100.0

                # Δ-объёма: текущая 1-мин свеча vs сумма предыдущих 5
                prev_vol_5m = sum(safe_to_float(c["volume"]) for c in recent[-6:-1])
                curr_vol_1m = safe_to_float(recent[-1]["volume"])
                if prev_vol_5m <= 0:
                    return
                # vol_change_pct = (curr_vol_1m - prev_vol_5m) / prev_vol_5m * 100.0

                # vol_sigma = max(1.0, self.shared_ws._sigma_5m(symbol) * 100)
                # dynamic_power_min = max(self.squeeze_power_min, 6 * vol_sigma)
                # squeeze_power = abs(pct_5m) * abs(vol_change_pct)
                # d_oi = self.shared_ws.get_delta_oi(symbol)
                # if (
                #     abs(pct_5m) < self.squeeze_threshold_pct or
                #     squeeze_power < dynamic_power_min or
                #     (d_oi is not None and d_oi * math.copysign(1, pct_5m) < 0)
                # ):
                #      return


                vol_change_pct = (curr_vol_1m - prev_vol_5m) / prev_vol_5m * 100.0

                # ───────────────────────────────────────────────────────────
                # Д И А Г Н О С Т И К А  «почему пропустили сквиз»
                # ───────────────────────────────────────────────────────────
                diag_path  = "squeeze_diag.csv"
                sigma_pct  = self.shared_ws._sigma_5m(symbol) * 100
                thr_price  = max(self.squeeze_threshold_pct, 1.4 * sigma_pct)
                power_min  = max(self.squeeze_power_min,   3.3 * sigma_pct)

                squeeze_power = abs(pct_5m) * abs(vol_change_pct)
                d_oi = self.shared_ws.get_delta_oi(symbol)

                dbg = []
                # sign-aware ΔP threshold
                if pct_5m >= 0:  # upward move
                    if pct_5m < thr_price:
                        dbg.append(f"ΔP {pct_5m:.2f}%<{thr_price:.2f}")
                else:           # downward move
                    if pct_5m > -thr_price:
                        dbg.append(f"ΔP {pct_5m:.2f}%>{-thr_price:.2f}")

                # power check (always magnitude)
                if squeeze_power < power_min:
                    dbg.append(f"power {squeeze_power:.1f}<{power_min:.1f}")
                    # reject only if OI moves against price direction
                    if d_oi is not None and math.copysign(1, d_oi) != math.copysign(1, pct_5m):
                        dbg.append("dOI sign contra")

                try:
                    self._tune_squeeze({
                        "pct_5m": pct_5m,
                        "vol_change_pct": vol_change_pct,
                        "sigma5m": sigma_pct,
                        "d_oi": d_oi or 0.0,
                        "spread_pct": spread_pct,
                        "squeeze_power": squeeze_power,
                    })
                except RuntimeError:
                    logger.debug("[SQUEEZE] %s vetoed by ML-tuner", symbol)
                    return

                if dbg:
                    reasons = "|".join(dbg)
                    logger.debug("[SQZ_SKIP] %s %s", symbol, reasons)

                # --- SHORT: require both volume spike and OI growth
                if pct_5m < 0:
                    if not (vol_change_pct >= power_min and (d_oi or 0.0) >= 0.02):
                        logger.debug("[SQZ_SKIP] %s short fails vol+OI combo", symbol)
                        return

                header = [
                    "timestamp", "symbol",
                    "pct5m", "vol_change_pct",
                    "sigma_pct", "squeeze_power",
                    "delta_oi", "reasons"
                ]

                if os.path.isfile(diag_path):
                    try:
                        df = pd.read_csv(diag_path)
                    except (pd.errors.EmptyDataError, FileNotFoundError):
                        df = pd.DataFrame(columns=header)

                    # ── если колонок нет – переписываем файл с правильным заголовком ──
                    if "timestamp" not in df.columns:
                        logger.warning(
                            "[squeeze_diag] detected broken file – recreating %s", diag_path)
                        df = pd.DataFrame(columns=header)
                        df.to_csv(diag_path, index=False)
                    else:
                        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)
                        df = df[df["timestamp"] >= cutoff.isoformat()]
                        df.to_csv(diag_path, index=False)

                    # — добавление текущей диагностики —
                    new_file = not os.path.isfile(diag_path)
                    with open(diag_path, "a", newline="", encoding="utf-8") as fp:
                        w = csv.writer(fp)
                        if new_file:
                            w.writerow([                      # строка-заголовок
                                "timestamp", "symbol",
                                "pct5m", "vol_change_pct",
                                "sigma_pct", "squeeze_power",
                                "delta_oi", "reasons"
                            ])
                        w.writerow([                          # ОДНА строка-данных
                            datetime.utcnow().isoformat(timespec="seconds"),
                            symbol,
                            f"{pct_5m:.4f}",
                            f"{vol_change_pct:.4f}",
                            f"{sigma_pct:.4f}",
                            f"{squeeze_power:.2f}",
                            f"{d_oi:.6f}" if d_oi is not None else "",
                            reasons,
                        ])
                    return

                # — выбор направления сделки —
                if pct_5m >= self.squeeze_threshold_pct:
                    action, side, position_idx = "SHORT", "Sell", 2
                elif pct_5m <= -self.squeeze_threshold_pct:
                    action, side, position_idx = "LONG",  "Buy",  1
                else:
                    return

                # — динамический лимит по спреду —
                bid_price = safe_to_float(ticker_data.get("bid1Price", 0))
                ask_price = safe_to_float(ticker_data.get("ask1Price", 0))
                if bid_price > 0 and ask_price > 0:
                    avg_vol_30m = self.shared_ws.get_avg_volume(symbol, 30)
                    if avg_vol_30m >= 10_000_000:
                        spread_limit = 0.05
                    elif avg_vol_30m >= 1_000_000:
                        spread_limit = 0.10
                    elif avg_vol_30m >= 100_000:
                        spread_limit = 0.30
                    else:
                        spread_limit = 0.50

                    if spread_pct > spread_limit:
                        logger.debug(
                            "[SQUEEZE] %s spread %.3f%% > limit %.2f%% — skip",
                            symbol, spread_pct, spread_limit
                        )
                        return

                # — расчёт количества контрактов —
                async with self.position_lock:
                    usd_size = min(self.POSITION_VOLUME, self.max_allowed_volume)
                    total_expo = await self.get_total_open_volume()
                    if total_expo + usd_size > self.MAX_TOTAL_VOLUME:
                        logger.info(
                            "[SQUEEZE] skip %s: exposure %.0f + %.0f > %.0f",
                            symbol, total_expo, usd_size, self.MAX_TOTAL_VOLUME
                        )
                        return

                    await self.ensure_symbol_meta(symbol)
                    qty = self._calc_qty_from_usd(symbol, usd_size, limit_price)

                if qty <= 0:
                    logger.warning("[SQUEEZE] %s: calculated qty=%.6f is invalid", symbol, qty)
                    return

                # — резервируем комментарий и время —
                self.pending_strategy_comments[symbol] = (
                    f"Сквиз {self.squeeze_threshold_pct}%/5m ({action})"
                )
                self.pending_orders.add(symbol)
                self.pending_timestamps[symbol] = time.time()

                # — лог планирования и сброс cooldown —
                logger.info(
                    "[SQUEEZE] %s %s qty=%.6f adaptive entry scheduled",
                    symbol, action, qty
                )
                self.last_squeeze_ts[symbol] = time.time()

                # — запускаем adaptive entry в фоне —
                if self.mode == "real":
                    asyncio.create_task(self.adaptive_entry_ws(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        position_idx=position_idx,
                        max_entry_timeout=60
                    ))
                else:
                    asyncio.create_task(self.adaptive_entry(
                        symbol=symbol,
                        side=side,
                        qty=qty,
                        max_entry_timeout=60
                    ))
                return
                
            # --- strategy switches ---------------------------------
            mode = getattr(self, "strategy_mode", "full")
            golden_enabled = mode in ("golden_squeeze", "golden_only", "full")
            liq_enabled = mode in ("liq_squeeze", "liquidation_only", "full")

            # ---- LIQUIDATION INFO ------------------------------------------------
            if liq_enabled:
                liq_info = self.shared_ws.latest_liquidation.get(symbol, {})
                liq_val  = safe_to_float(liq_info.get("value", 0))
                liq_side = liq_info.get("side", "")
                liq_ts   = liq_info.get("ts", 0.0)

                liq_recent = (time.time() - liq_ts) <= 60
                threshold  = self.shared_ws.get_liq_threshold(symbol, 5000)

                logger.info(f"[DIAG] {symbol} LiqCheck: val={liq_val}, side={liq_side}, ts_age={time.time()-liq_ts:.1f}s, threshold={threshold}")

                if not liq_recent:
                    logger.info(f"[DIAG] {symbol} ликвидация не свежая (>60s) → пропуск")
                    return
                if liq_val < threshold:
                    logger.info(f"[DIAG] {symbol} ликвидация ниже порога {liq_val} < {threshold} → пропуск")
                    return
                if liq_side not in ("Buy", "Sell"):
                    logger.info(f"[DIAG] {symbol} нет валидной стороны ликвидации → пропуск")
                    return

                candles = self.shared_ws.candles_data.get(symbol, [])
                volume_now = safe_to_float(candles[-1].get("volume", 0)) if candles else 0.0
                avg_vol_30m = self.shared_ws.get_avg_volume(symbol, 30)
                VOLUME_COEF_ADJ = VOLUME_COEF * 0.8

                logger.info(f"[DIAG] {symbol} VolumeCheck: now={volume_now}, avg30m={avg_vol_30m}, threshold={VOLUME_COEF_ADJ * avg_vol_30m}")

                if volume_now < VOLUME_COEF_ADJ * avg_vol_30m:
                    logger.info(f"[DIAG] {symbol} объём ниже порога → пропуск")
                    return

                funding_ok = self.shared_ws.funding_cool(symbol)
                if not funding_ok:
                    logger.info(f"[DIAG] {symbol} funding горячий → пропуск")
                    return

                delta_oi = self.shared_ws.get_delta_oi(symbol)
                if delta_oi is None:
                    logger.info(f"[DIAG] {symbol} нет данных delta_oi → пропуск")
                    return
                thr = -0.5 * self._oi_sigma.get(symbol, 0.003)
                if delta_oi > thr:
                    logger.info(f"[DIAG] {symbol} delta_oi {delta_oi:.4f} выше порога → пропуск")
                    return

                vol_hist = list(self.shared_ws.volume_history.get(symbol, []))
                if len(vol_hist) >= 2:
                    volume_change = (vol_hist[-1] - vol_hist[-2]) / max(1e-8, vol_hist[-2]) * 100
                    logger.info(f"[DIAG] {symbol} dV={volume_change:.2f}%")
                    if volume_change <= 0:
                        logger.info(f"[DIAG] {symbol} объём не растёт → пропуск")
                        return
                else:
                    logger.info(f"[DIAG] {symbol} недостаточно исторических данных объёма → пропуск")
                    return

                cooldown_ok = self.shared_ws.check_liq_cooldown(symbol)
                if not cooldown_ok:
                    logger.info(f"[DIAG] {symbol} cooldown ещё активен → пропуск")
                    return

                logger.info(f"[DIAG] {symbol} --- ВСЕ ФИЛЬТРЫ ПРОЙДЕНЫ --- ГОТОВИМСЯ К ОРДЕРУ")

                # всё прошло — открываем сделку
                opposite = "Buy" if liq_side == "Sell" else "Sell"
                if symbol not in self.open_positions and symbol not in self.pending_orders:
                    self.pending_strategy_comments[symbol] = "От ликвидаций"

                    total_expo = await self.get_total_open_volume()
                    potential = self.POSITION_VOLUME
                    if total_expo + potential > self.MAX_TOTAL_VOLUME:
                        logger.warning("[LiqTrade] skip %s: exposure %.0f + %.0f > %.0f",
                                    symbol, total_expo, potential, self.MAX_TOTAL_VOLUME)
                        return

                    # расчёт qty
                    close_price = safe_to_float(
                        self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
                    ) or safe_to_float(
                        self.shared_ws.candles_data.get(symbol, [])[-1]["closePrice"]
                    )

                    if close_price > 0:
                        step = self.qty_step_map.get(symbol, 0.001)
                        step_str = str(step)
                        dec = len(step_str.split(".")[1].rstrip("0")) if "." in step_str else 0
                        qty_raw = safe_to_float(self.POSITION_VOLUME) / close_price
                        qty = math.floor(qty_raw / step) * step
                        qty = round(qty, dec)
                        qty_str = f"{qty:.{dec}f}"
                        pos_idx = 1 if opposite == "Buy" else 2

                        try:
                            logger.info("[LiqTrade] %s ликв %.0f USDT → %s %.3f",
                                        symbol, liq_val, opposite, qty)

                            if self.mode == "real":
                                await self.place_order_ws(symbol, opposite, qty, position_idx=pos_idx)
                            else:
                                resp = await asyncio.to_thread(lambda: self.session.place_order(
                                    category="linear",
                                    symbol=symbol,
                                    side=opposite,
                                    orderType="Market",
                                    qty=qty_str,
                                    timeInForce="GTC",
                                    positionIdx=pos_idx
                                ))
                                # Clear the processed golden setup signal
                                if resp.get("retCode", 0) != 0:
                                    raise InvalidRequestError(resp.get("retMsg", "order rejected"))

                            self.pending_strategy_comments[symbol] = "Чистый Золотой Сетап"
                            self.pending_orders.add(symbol)
                            self.pending_timestamps[symbol] = time.time()

                        except Exception as e:
                            logger.warning("[LiqTrade] order failed for %s: %s", symbol, e)
                    return  # пропускаем golden-setup если сработала ликвидация


            if not golden_enabled:
                return     # режим "только ликвидации" – классический GS пропускаем

            minute_candles = self.shared_ws.candles_data.get(symbol, [])
            recent = self._aggregate_candles_5m(minute_candles)
            # aggregated 5‑minute helpers for volume / OI / CVD
            vol_hist_5m = self._aggregate_series_5m(list(self.shared_ws.volume_history.get(symbol, [])), method="sum")
            oi_hist_5m  = self._aggregate_series_5m(list(self.shared_ws.oi_history.get(symbol, [])), method="last")
            cvd_hist_5m = self._aggregate_series_5m(list(self.shared_ws.cvd_history.get(symbol, [])), method="sum")
            if not recent:
                return

            # --- динамические пороги Golden‑setup (CSV → ML) ---
            buy_params  = await self._get_golden_thresholds(symbol, "Buy")
            sell_params = await self._get_golden_thresholds(symbol, "Sell")
            #  reserve: если понадобится альтернативный short‑сетап
            #sell2_params = await self._get_golden_thresholds(symbol, "Sell2")

            period_iters = max(
                int(buy_params["period_iters"]),
                int(sell_params["period_iters"]),
            #    int(sell2_params.get("period_iters", 0)),
            )

            if (len(recent) <= period_iters or
                len(vol_hist_5m) <= period_iters or
                len(oi_hist_5m)  <= period_iters or
                len(cvd_hist_5m) <= period_iters):
                return

            # 4. Рассчитываем Δ от прошлой точки
            old_bar = recent[-1 - period_iters]
            new_bar = recent[-1]
            close_price = safe_to_float(new_bar["closePrice"])
            old_close   = safe_to_float(old_bar["closePrice"]) if old_bar["closePrice"] else 0.0

            price_change_pct = (
                (close_price - old_close) / old_close * 100.0
                if old_close != 0 else 0.0
            )

            old_vol = safe_to_float(vol_hist_5m[-1 - period_iters])
            new_vol = safe_to_float(vol_hist_5m[-1])
            volume_change_pct = (
                (new_vol - old_vol) / old_vol * 100.0
                if old_vol != 0 else 0.0
            )

            # --- use Decimal for higher precision ---
            old_oi = Decimal(str(oi_hist_5m[-1 - period_iters]))
            new_oi = Decimal(str(oi_hist_5m[-1]))
            oi_change_pct = (
                (new_oi - old_oi) / old_oi * Decimal("100")
                if old_oi != 0 else Decimal("0")
            )

            # --- CVD % change over the same period ----------------
            old_cvd = safe_to_float(cvd_hist_5m[-1 - period_iters])
            new_cvd = safe_to_float(cvd_hist_5m[-1])
            if abs(old_cvd) > 1e-8:
                cvd_change_pct = (new_cvd - old_cvd) / abs(old_cvd) * 100.0
            else:
                cvd_change_pct = 0.0
            logger.debug("[CVD%%] %s Δ=%.0f  old=%.0f  new=%.0f  pct=%.2f%%",
                        symbol, new_cvd-old_cvd, old_cvd, new_cvd, cvd_change_pct)

            # --- composite signal strength ---
            # OI и объём — главные (0.4 + 0.4), цена — подтверждающая (0.2)
            strength_oi    = abs(float(oi_change_pct))
            strength_vol   = abs(volume_change_pct)
            strength_price = abs(price_change_pct)
            signal_strength = (
                0.5 * strength_oi +
                0.3 * strength_vol +
                0.2 * strength_price
            )

            # --- diagnostic snapshot of current Golden-setup metrics ----
            logger.info(
                "[Golden data] %s ΔP=%.2f%% ΔV=%.1f%% ΔOI=%.2f%% ΔCVD=%.1f%% "
                "strength=%.1f period=%d  |  "
                "Buy-thr: p≥%.2f v≥%.1f oi≥%.2f  •  "
                "Sell-thr: p≤%.2f v≥%.1f oi≥%.2f",
                symbol,
                price_change_pct,
                volume_change_pct,
                float(oi_change_pct),
                cvd_change_pct,
                signal_strength,
                period_iters,
                float(buy_params["price_change"]),
                float(buy_params["volume_change"]),
                float(buy_params["oi_change"]),
                -float(sell_params["price_change"]),   # drop is negative
                float(sell_params["volume_change"]),
                float(sell_params["oi_change"]),
            )

            SAME_SIDE_LIQ_THRESHOLD = 4000.0  # или другое

            # 5. Определяем сигнал
            action = None
            # основной Sell
            sp = int(sell_params["period_iters"])
            if len(recent) > sp:
                old = recent[-1 - sp]
                new = recent[-1]
                pchg = (Decimal(str(new["closePrice"])) - Decimal(str(old["closePrice"]))) / Decimal(str(old["closePrice"])) * Decimal("100") if old["closePrice"] else Decimal("0")
                volchg = (
                    (Decimal(str(vol_hist_5m[-1])) - Decimal(str(vol_hist_5m[-1 - sp])))
                    / Decimal(str(vol_hist_5m[-1 - sp]))
                    * Decimal("100")
                    if vol_hist_5m[-1 - sp]
                    else Decimal("0")
                )
                oichg = (
                    (Decimal(str(oi_hist_5m[-1])) - Decimal(str(oi_hist_5m[-1 - sp])))
                    / Decimal(str(oi_hist_5m[-1 - sp]))
                    * Decimal("100")
                    if oi_hist_5m[-1 - sp]
                    else Decimal("0")
                )

                # --- thresholds used only for diagnostic logging ---
                p_thr  = -float(sell_params["price_change"])   # required price drop (negative)
                v_thr  =  float(sell_params["volume_change"])  # required volume surge
                oi_thr =  float(sell_params["oi_change"])      # required OI increase

                logger.info("[Golden skip] %s pchg=%.2f vol=%.1f oi=%.2f  (need ≥ %.2f / %.1f / %.2f)",
                            symbol, float(pchg), float(volchg), float(oichg),
                            p_thr, v_thr, oi_thr)

                logger.info(
                    "[Golden SELL-probe] %s ΔP=%.2f%% ΔV=%.1f%% ΔOI=%.2f%% ΔCVD=%.1f%% "
                    "(need ≥ %.2f / %.1f / %.2f / CVD 18.3-200)",
                    symbol, pchgb, volb, oib, cvd_change_pct,
                    sell_params["price_change"],
                    sell_params["volume_change"],
                    sell_params["oi_change"],
                )

                # SELL
                if (
                    pchg <= -Decimal(str(sell_params["price_change"]))
                    and volchg >= Decimal(str(sell_params["volume_change"]))
                    and oichg >= Decimal(str(sell_params["oi_change"]))
                    and (-50 <= cvd_change_pct <= -18.3)
                    and not (liq_side == "Sell" and liq_val >= threshold)
                ):
                    action = "Sell"
    
            if action is None:
                lp = int(buy_params["period_iters"])
                if len(recent) > lp:
                    oldb, newb = recent[-1 - lp], recent[-1]
                    pchgb = (Decimal(str(newb["closePrice"])) - Decimal(str(oldb["closePrice"])))/Decimal(str(oldb["closePrice"])) * Decimal("100") if oldb["closePrice"] else Decimal("0")
                    volb = (
                        (Decimal(str(vol_hist_5m[-1])) - Decimal(str(vol_hist_5m[-1 - lp])))
                        / Decimal(str(vol_hist_5m[-1 - lp]))
                        * Decimal("100")
                        if vol_hist_5m[-1 - lp]
                        else Decimal("0")
                    )
                    oib = (
                        (Decimal(str(oi_hist_5m[-1])) - Decimal(str(oi_hist_5m[-1 - lp])))
                        / Decimal(str(oi_hist_5m[-1 - lp]))
                        * Decimal("100")
                        if oi_hist_5m[-1 - lp]
                        else Decimal("0")
                    )

                    logger.info(
                        "[Golden BUY-probe] %s ΔP=%.2f%% ΔV=%.1f%% ΔOI=%.2f%% ΔCVD=%.1f%% "
                        "(need ≥ %.2f / %.1f / %.2f / CVD 18.3-200)",
                        symbol, pchgb, volb, oib, cvd_change_pct,
                        buy_params["price_change"],
                        buy_params["volume_change"],
                        buy_params["oi_change"],
                    )

                    # BUY
                    if (
                        pchgb >= Decimal(str(buy_params["price_change"]))
                        and volb >= Decimal(str(buy_params["volume_change"]))
                        and oib  >= Decimal(str(buy_params["oi_change"]))
                        and (18.3 <= cvd_change_pct <= 200)
                        and not (liq_side == "Buy" and liq_val >= threshold)
                    ):
                        action = "Buy"
                        volchg = volb
                        oichg  = oib

            if action is None:
                return

            # --- ML override / veto -----------------------------------
            try:
                # 1) соберём «живые» признаки
                feats = await self.extract_realtime_features(symbol)
                if feats:
                    import numpy as _np          # локальный импорт
                    X = _np.array([[feats.get(k, 0.0) for k in FEATURE_KEYS]],
                                  dtype=_np.float32)
                    ml_probs = self.ml_inferencer.infer(X)[0]      # [Sell, Hold, Buy]
                    ml_cls   = int(ml_probs.argmax())
                    ml_map   = {0: "Sell", 1: None, 2: "Buy"}
                    ml_signal = ml_map.get(ml_cls)

                    # «Hold» – вето модели
                    if ml_signal is None:
                        logger.info("[Golden ML veto] %s: модель = HOLD – сигнал отменён", symbol)
                        return

                    # Модель предлагает иной direction → переопределяем
                    if ml_signal != action:
                        logger.info("[Golden ML override] %s: %s → %s  (p=%.2f)",
                                    symbol, action, ml_signal, ml_probs[ml_cls])
                        action = ml_signal
            except Exception as _e:
                # Ошибки ML-инференса не должны ломать стратегию
                logger.debug("[Golden ML] inference failed: %s", _e)


            # 7. Пропускаем или резервируем позицию и проверяем лимит в критической секции
            if volchg <= 0 or oichg <= 0:
                logger.info("[Golden] %s отменён: объём/OI не растут (dV=%.1f dOI=%.2f)",
                            symbol, volchg, oichg)
                return


            side_params = buy_params if action == "Buy" else sell_params
            volume_usdt = safe_to_float(side_params.get("position_volume", self.POSITION_VOLUME))
            last_price  = safe_to_float(close_price)
            if last_price == 0:
                logger.info(f"[GoldenSetup] {symbol} — пропуск, цена нулевая")
                return

            # атомарно проверяем и резервируем слот под новую позицию
            async with self.position_lock:
                if symbol in self.open_positions or symbol in self.pending_orders:
                    return
                current_total = await self.get_total_open_volume()
                if current_total + volume_usdt > self.MAX_TOTAL_VOLUME:
                    logger.info(
                        f"[GoldenSetup] {symbol}: превышен MAX_TOTAL_VOLUME "
                        f"{current_total:.2f} + {volume_usdt:.2f} > {self.MAX_TOTAL_VOLUME:.2f}"
                    )
                    self.failed_orders[symbol] = time.time()
                    return
                # резервируем место для этой позиции
                self.pending_orders.add(symbol)

            # игнорируем сигнал, если только что была крупная ликвидация в противоположную сторону
            if liq_side and liq_side != action and liq_val >= 3000.0:
                logger.info(
                    "[GoldenSetup] Пропуск %s — %s-ликвидация %.0f USDT против сигнала %s",
                    symbol, liq_side, liq_val, action
                )
                self.pending_orders.discard(symbol)
                return

            qty     = self._calc_qty_from_usd(symbol, volume_usdt, last_price)
            step    = self.qty_step_map.get(symbol, 0.001)
            min_qty = self.min_qty_map.get(symbol, step)
            if qty < min_qty or qty <= 0:
                logger.info(f"[GoldenSetup] {symbol} — qty {qty:.8f} < min_qty {min_qty:.8f}; пропуск")
                self.pending_orders.discard(symbol)
                return

            # 6. Логирование снимка
            snapshot_row = ({
                "close_price":    safe_to_float(close_price),
                "price_change":   safe_to_float(price_change_pct),
                "volume_change":  safe_to_float(volume_change_pct),
                "oi_change":      safe_to_float(oi_change_pct),
                "period_iters":   period_iters,
                "user_id":        self.user_id,
                "symbol":         symbol,
                "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "signal":         action,
                "signal_strength": signal_strength,
                "cvd_change": cvd_change_pct,
                
            })

            # логируем не чаще, чем раз в 250 с для каждого тикера
            if time.time() - self._last_snapshot_ts.get(symbol, 0) >= 250:
                _append_snapshot(snapshot_row)
                self._last_snapshot_ts[symbol] = time.time()

            pos_idx = 1 if action == "Buy" else 2

            # 9. Исполнение и логирование
            remaining_qty = qty
            step = self.qty_step_map.get(symbol, 0.001)
            min_qty = self.min_qty_map.get(symbol, step)

            self.golden_param_store.setdefault("last_snapshot", {})[symbol] = {
                "price_change": price_change_pct,
                "volume_change": volume_change_pct,
                "oi_change": float(oi_change_pct),
                "period_iters": period_iters,
                "user_id": self.user_id,
                "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "signal": action,
                "signal_strength": signal_strength,
                "cvd_change": cvd_change_pct,
                
            }

            while remaining_qty >= min_qty:
                try:
                    if self.mode == "real":
                        await self.place_order_ws(symbol, action, remaining_qty, position_idx=pos_idx)
                    else:
                        resp = await asyncio.to_thread(lambda: self.session.place_order(
                            category="linear",
                            symbol=symbol,
                            side=action,
                            orderType="Market",
                            qty=str(remaining_qty),
                            timeInForce="GTC",
                            positionIdx=pos_idx
                        ))
                        if resp.get("retCode", 0) != 0:
                            raise InvalidRequestError(resp.get("retMsg", "order rejected"))
                    # ордер успешно поставлен — выходим из цикла
                        self._last_golden_ts[symbol] = time.time()
                    break
                except (InvalidRequestError, RuntimeError) as e:
                    msg = str(e)
                    if "Qty invalid" in msg:
                        logger.warning(
                            "[execute_golden_setup] %s invalid quantity %s: %s, retrying with smaller qty",
                            symbol, remaining_qty, msg
                        )
                        remaining_qty -= step
                        remaining_qty = math.floor(remaining_qty / step) * step
                        continue
                    if "position idx not match position mode" in msg:
                        logger.warning(
                            "[execute_golden_setup] %s position idx not match position mode: %s",
                            symbol, msg
                        )
                        self.pending_orders.discard(symbol)
                        return
                    # для любых других ошибок повторно выбрасываем
                    raise
        except Exception as e:
            logger.error(f"[execute_golden_setup] unexpected error for {symbol}: {e}", exc_info=True)
            self.pending_orders.discard(symbol)
        finally:
            # ── во всех случаях удаляем сигнал ──────────────────────────
            self.pending_signals.pop(symbol, None)



    async def adaptive_squeeze_entry_ws(
        self,
        symbol: str,
        side: str,
        qty: float,
        position_idx: int,
        max_entry_timeout: int = 15
    ):
        """
        Быстрый WS-адаптивный лимитный вход в позицию для сквизов.
        """
        if symbol in self.open_positions:
            logger.info(f"Skipping squeeze for {symbol}: position already open")
            return
        try:
            # подбираем точный tickSize
            await self.ensure_symbol_meta(symbol)
            tick = safe_to_float(self.price_tick_map.get(symbol, DEC_TICK))
            offset_pct = SQUEEZE_LIMIT_OFFSET_PCT
            reprice_interval = SQUEEZE_REPRICE_INTERVAL

            order_id = None
            started_ts = asyncio.get_running_loop().time()

            while True:
                # Актуальная цена через WS
                ticker = self.shared_ws.ticker_data.get(symbol, {})
                last_price = safe_to_float(ticker.get("lastPrice", 0))
                # --- risk gate ---
                if not await self._risk_check(symbol, side, qty, last_price):
                    break     # лимиты нарушены – выходим из цикла
                best_bid = safe_to_float(ticker.get("bid1Price", last_price))
                best_ask = safe_to_float(ticker.get("ask1Price", last_price))

                if last_price == 0:
                    logger.warning(f"[adaptive_ws] Нет цены для {symbol}, ждём...")
                    await asyncio.sleep(reprice_interval)
                    continue

                # Расчет лимитной цены
                if side.lower() == "buy":
                    raw_price = best_bid * (1 - offset_pct)
                    limit_price = math.floor(raw_price / tick) * tick
                else:
                    raw_price = best_ask * (1 + offset_pct)
                    limit_price = math.ceil(raw_price / tick) * tick

                # Первый запуск → создаём ордер
                if order_id is None:
                    try:
                        resp = await self.ws_private.place_order_ws(
                            category="linear",
                            symbol=symbol,
                            side=side.capitalize(),
                            orderType="Limit",
                            qty=self._format_qty(symbol, qty),
                            price=str(limit_price),
                            timeInForce="PostOnly",
                            reduceOnly=False,
                            positionIdx=position_idx
                        )
                        order_id = resp["result"]["orderId"]
                        logger.info(f"[adaptive_ws] Created order {order_id} @ {limit_price}")

                    except Exception as e:
                        logger.warning(f"[adaptive_ws] Ошибка создания ордера: {e}")
                        await asyncio.sleep(reprice_interval)
                        continue

                else:
                    # Переставляем ордер через amend
                    try:
                        await self.ws_private.amend_order_ws(
                            category="linear",
                            symbol=symbol,
                            orderId=order_id,
                            price=str(limit_price)
                        )
                        logger.info(f"[adaptive_ws] Amended {order_id} → {limit_price}")

                    except Exception as e:
                        logger.warning(f"[adaptive_ws] Ошибка amend: {e}")

                # ждём следующей итерации или выхода
                await asyncio.sleep(reprice_interval)

                # таймаут
                if asyncio.get_running_loop().time() - started_ts > max_entry_timeout:
                    logger.warning(f"[adaptive_ws] Entry timeout for {symbol}")
                    break

                # Проверка — позиция уже открыта
                pos = self.open_positions.get(symbol)
                if pos and safe_to_float(pos.get("volume", 0)) >= qty:
                    logger.info(f"[adaptive_ws] Позиция по {symbol} уже открыта")
                    break

            # Финальная отмена ордера
            if order_id:
                try:
                    await self.ws_private.cancel_order_ws(
                        category="linear",
                        symbol=symbol,
                        orderId=order_id
                    )
                except Exception:
                    pass

        except Exception:
            # снять блокировку, чтобы новые сигналы прошли
            self.pending_orders.discard(symbol)
            self.pending_timestamps.pop(symbol, None)
            logger.exception(f"[adaptive_ws] Критическая ошибка {symbol}")


    async def adaptive_squeeze_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        max_entry_timeout: int = 30  # Увеличено до 30 секунд для большей гибкости
    ) -> bool:
        """
        Адаптивный лимитный вход в позицию для сквизов.
        Ставит лимитный ордер с подвижкой за ценой, обеспечивая корректное форматирование qty.

        :param symbol: Тикер (например, 'AEVOUSDT')
        :param side: 'Buy' или 'Sell'
        :param qty: Объём позиции
        :param max_entry_timeout: Макс. время попыток (секунд)
        :return: True, если позиция открыта, False в случае неудачи
        """
        try:
            # 1. Загружаем метаданные символа (qtyStep, minOrderQty)
            await self.ensure_symbol_meta(symbol)
            step = self.qty_step_map.get(symbol, 0.001)
            min_qty = self.min_qty_map.get(symbol, step)
            logger.debug(f"[adaptive_entry] {symbol}: qtyStep={step}, minQty={min_qty}, requested_qty={qty}")

            # 2. Форматируем количество
            qty_formatted = self._format_qty(symbol, qty)
            qty_float = float(qty_formatted)
            if qty_float < min_qty:
                logger.warning(f"[adaptive_entry] {symbol}: qty {qty_float} меньше minQty {min_qty}, пропуск")
                return False

            # 3. Параметры для лимитного ордера
            # точный tickSize из priceFilter
            await self.ensure_symbol_meta(symbol)
            tick = safe_to_float(self.price_tick_map.get(symbol, DEC_TICK))
            offset_pct = SQUEEZE_LIMIT_OFFSET_PCT  # 0.005 (0.5%) по умолчанию
            reprice_interval = SQUEEZE_REPRICE_INTERVAL  # 2 секунды
            pos_idx = 1  # Всегда linear для Bybit V5
            order_id = None
            started_ts = asyncio.get_running_loop().time()

            while True:
                # 4. Получаем актуальные рыночные данные
                ticker = self.shared_ws.ticker_data.get(symbol, {})
                last_price = safe_to_float(ticker.get("lastPrice", 0))
                # --- risk gate ---
                if not await self._risk_check(symbol, side, qty, last_price):
                    break
                best_bid = safe_to_float(ticker.get("bid1Price", last_price))
                best_ask = safe_to_float(ticker.get("ask1Price", last_price))

                if last_price == 0:
                    logger.warning(f"[adaptive_entry] {symbol}: нет цены, ждём {reprice_interval}с")
                    await asyncio.sleep(reprice_interval)
                    continue

                # Логируем рыночные данные для диагностики
                logger.debug(f"[adaptive_entry] {symbol}: lastPrice={last_price}, bid={best_bid}, ask={best_ask}")

                # 5. Рассчитываем цену лимитника
                if side.lower() == "buy":
                    raw_price = best_bid * (1 - offset_pct)
                    limit_price = math.floor(raw_price / tick) * tick
                else:
                    raw_price = best_ask * (1 + offset_pct)
                    limit_price = math.ceil(raw_price / tick) * tick

                # 6-7. Первый проход — place, далее — amend
                if order_id is None:
                    try:
                        resp = await asyncio.to_thread(
                            lambda: self.session.place_order(
                                category="linear",
                                symbol=symbol,
                                side=side.capitalize(),
                                orderType="Limit",
                                qty=qty_formatted,
                                price=str(limit_price),
                                timeInForce="PostOnly",
                                reduceOnly=False,
                                positionIdx=pos_idx,
                            )
                        )
                        order_id = resp.get("result", {}).get("orderId")
                        logger.info(f"[adaptive_entry] {symbol}: создан ордер {order_id} @ {limit_price}")
                    except Exception as e:
                        logger.warning(f"[adaptive_entry] {symbol}: ошибка place: {e}")
                        await asyncio.sleep(reprice_interval)
                        continue
                else:
                    try:
                        await asyncio.to_thread(
                            lambda: self.session.amend_order(
                                category="linear",
                                symbol=symbol,
                                orderId=order_id,
                                price=str(limit_price),
                            )
                        )
                        logger.debug(f"[adaptive_entry] {symbol}: amend {order_id} → {limit_price}")
                    except Exception as e:
                        if "order not exist" in str(e).lower():
                            order_id = None      # ордер уже исполнился/отменён
                        logger.warning(f"[adaptive_entry] {symbol}: ошибка amend: {e}")

                # 8. Проверяем статус ордера
                try:
                    order = await asyncio.to_thread(
                        lambda: self.session.get_order_history(
                            category="linear",
                            symbol=symbol,
                            orderId=order_id
                        )
                    )
                    order_info = order.get("result", {}).get("list", [{}])[0]
                    status = order_info.get("orderStatus", "").upper()

                    if status in ("FILLED", "PARTIALLY_FILLED"):
                        logger.info(f"[adaptive_entry] {symbol}: ордер {status}, позиция открыта")
                        return True

                except Exception as e:
                    logger.warning(f"[adaptive_entry] {symbol}: ошибка проверки статуса ордера: {e}")

                # 9. Проверяем, открыта ли позиция
                pos = self.open_positions.get(symbol)
                if pos and safe_to_float(pos.get("volume", 0)) >= qty_float:
                    logger.info(f"[adaptive_entry] {symbol}: позиция уже открыта (volume={pos.get('volume')})")
                    return True

                # 10. Проверка тайм-аута
                if asyncio.get_running_loop().time() - started_ts > max_entry_timeout:
                    logger.warning(f"[adaptive_entry] {symbol}: время на вход истекло ({max_entry_timeout}с)")
                    break

                # Ждём следующей итерации
                await asyncio.sleep(reprice_interval)

            # 11. Финальная отмена висящего ордера, если не исполнен
            if order_id:
                try:
                    await asyncio.to_thread(
                        lambda: self.session.cancel_order(
                            category="linear",
                            symbol=symbol,
                            orderId=order_id
                        )
                    )
                    logger.debug(f"[adaptive_entry] {symbol}: финальная отмена ордера {order_id}")
                except Exception as e:
                    logger.warning(f"[adaptive_entry] {symbol}: ошибка финальной отмены ордера: {e}")

            return False

        except Exception as e:
            # снять блокировку
            self.pending_orders.discard(symbol)
            self.pending_timestamps.pop(symbol, None)
            logger.exception(f"[adaptive_entry] {symbol}: критическая ошибка")
            return False

    async def place_order_ws(self, symbol, side, qty, position_idx=1, price=None, order_type="Market"):
        """
        Send a WS order.create on the trade socket.
        If the socket is closed, try to reconnect and retry once.
        """
        header = {
            "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
            "X-BAPI-RECV-WINDOW": "5000"
        }
        args = {
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "category": "linear",
            "timeInForce": "GTC",
            "positionIdx": position_idx
        }
        if price is not None:
            args["price"] = str(price)

        req = {
            "op": "order.create",
            "header": header,
            "args": [args]
        }

        for attempt in range(2):
            try:
                await self.ws_trade.send(json.dumps(req))
                async with self._recv_lock:
                    resp = json.loads(await self.ws_trade.recv())
                if resp.get("retCode", resp.get("ret_code", 0)) != 0:
                    raise RuntimeError(f"Order failed: {resp}")
                return resp["data"]
            except (websockets.ConnectionClosed, asyncio.IncompleteReadError) as e:
                logger.warning(f"[place_order_ws] WebSocket closed, reconnecting (attempt {attempt+1}): {e}")
                await self.init_trade_ws()
            except Exception as e:
                logger.error(f"[place_order_ws] Unexpected error: {e}")
                raise
        raise RuntimeError("Failed to send order after reconnecting WebSocket")


    async def log_trade(self, symbol: str, row=None, *, side: str, avg_price: Decimal, volume: Decimal, open_interest: Decimal, action: str,
                        result: str, closed_manually: bool = False, csv_filename: str = "trade_log_MUWS.log"):

        if row is None:
            row = {}

        if "symbol" not in row:
            logger.warning("[log_trade] called without symbol: %s", row)
            row["symbol"] = "UNKNOWN"

        logger.info(
            f"[log_trade] user={self.user_id} {action.upper()} position {symbol}: "
            f"side={side}, avg_price={avg_price}, volume={volume}, result={result}"
        )

        row = row or {}
        if isinstance(row, list) and row:
            row = row[-1]
        elif hasattr(row, "iloc") and not row.empty:
            row = row.iloc[-1].to_dict()

        time_str = row.get("startTime", dt.datetime.utcnow())
        if isinstance(time_str, dt.datetime):
            time_str = time_str.strftime("%Y-%m-%d %H:%M:%S")

        open_str = str(row.get("openPrice", "N/A"))
        high_str = str(row.get("highPrice", "N/A"))
        low_str = str(row.get("lowPrice", "N/A"))
        close_str = str(row.get("closePrice", "N/A"))
        vol_str = str(row.get("volume", "N/A"))
        oi_str = str(open_interest) if open_interest is not None else "N/A"
        closed_str = "вручную" if closed_manually else "по сигналу"

        # помечаем, какой логикой открыта/закрыта сделка
        row["logic"] = self.pending_strategy_comments.get(row["symbol"], "")
        # ── ML-логирование: сохраняем пример, когда позиция закрыта ──
        if str(result).lower() in ("closed", "exit", "close_success", "exit_ok"):
            entry_data = {
                "price": float(avg_price),
                "side":  side.capitalize()
            }
            exit_price = safe_to_float(
                self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
            )
            exit_data = {
                "price": exit_price,
                "side":  side.capitalize()
            }
            asyncio.create_task(
                self.log_trade_for_ml(symbol, entry_data, exit_data)
            )

        # Добавляем sample в ML-буфер при закрытии
        if row.get("action") == "close":
            pnl_val = row.get("pnl_pct", 0.0)
            asyncio.create_task(
                self._capture_training_sample(row["symbol"], pnl_val)
            )

        def _log_csv():
            file_exists = os.path.isfile(csv_filename)
            with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "user_id", "symbol", "timestamp", "side", "entry_price", "volume",
                        "open_interest", "action", "result", "closed_manually"
                    ])
                writer.writerow([
                    self.user_id, symbol, time_str, side.upper(),               # BUY / SELL
                    str(avg_price), str(volume), str(open_interest),
                    action, result, closed_str
                ])

        await asyncio.to_thread(_log_csv)

        try:
            trade_path = "trades_for_training.csv"
            file_exists = os.path.isfile(trade_path)
            with open(trade_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "symbol", "datetime", "avg_price", "volume", "open_interest",
                        "side", "event", "pnl_pct", "close_price",
                        "price_change", "volume_change", "oi_change",
                        "period_iters", "user_id", "signal"
                    ])

                # Получаем цену закрытия
                close_price = self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", "")

                # Получаем saved snapshot (если был)
                snapshot = self.golden_param_store.get("last_snapshot", {}).get(symbol, {})

                if close_price and avg_price:
                    try:
                        direction = 1 if side.lower() == "buy" else -1
                        pnl_pct = round((float(close_price) - float(avg_price)) / float(avg_price) * 1000 * direction, 5)
                    except ZeroDivisionError:
                        pnl_pct = ""
                else:
                    pnl_pct = ""

                writer.writerow([
                    symbol,
                    datetime.utcnow().isoformat(),
                    avg_price,
                    volume,
                    open_interest,
                    side,
                    action,
                    pnl_pct,  # PnL% можешь отдельно добавить если хочешь
                    close_price,
                    snapshot.get("price_change", ""),
                    snapshot.get("volume_change", ""),
                    snapshot.get("oi_change", ""),
                    snapshot.get("period_iters", ""),
                    snapshot.get("user_id", ""),
                    snapshot.get("signal", ""),
                ])
        except Exception as e:
            logger.warning(f"[log_trade] Ошибка при записи trades_for_training.csv: {e}")

        try:
            # Fallback: derive PnL from `row` if it contains such field
            pnl_val = 0
            if isinstance(row, dict):
                pnl_val = row.get("pnl", 0)
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "pnl": pnl_val
            }
            import json, pathlib
            hist = []
            if pathlib.Path(TRADES_JSON).exists():
                with open(TRADES_JSON, "r", encoding="utf-8") as fp:
                    hist = json.load(fp) or []
            hist.append(record)
            _atomic_json_write(TRADES_JSON, hist[-5000:])
        except Exception:
            pass

        _append_trades_unified({
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "price": avg_price,
            "event": action,
            "result": result
        })

        user_state = self.load_user_state()
        if user_state.get("quiet_mode", False):
            return

        link_url = f"https://www.bybit.com/trade/usdt/{symbol}"
        s_result = str(result or "").strip().lower()
        if s_result in ("open", "entry", "open_success", "entry_ok"):
            s_result = "opened"
        elif s_result in ("close", "exit", "close_success", "exit_ok"):
            s_result = "closed"
        elif s_result in ("trailing", "trailing_stop", "trail", "ts"):
            s_result = "trailingstop"
        s_side = str(side or "").strip().lower()
        if s_side in ("long", "buy", "b"):
            s_side = "buy"
        elif s_side in ("short", "sell", "s"):
            s_side = "sell"
        s_manually = closed_str

        if s_result == "opened":
            if s_side.lower() == "buy":
                msg = (
                    f"🟩 <b>Открытие ЛОНГ-позиции</b>\n"
                    f"<b>Символ:</b> <a href=\"{link_url}\">{symbol}</a>\n"
                    f"<b>Пользователь:</b> {self.user_id}\n"
                    f"<b>Время:</b> {time_str}\n"
                    f"<b>Цена открытия:</b> {avg_price}\n"
                    f"<b>Объём:</b> {vol_str}\n"
                    f"<b>Тип открытия:</b> ЛОНГ\n"
                    f"#{symbol}"
                )
            elif s_side.lower() == "sell":
                msg = (
                    f"🟥 <b>Открытие SHORT-позиции</b>\n"
                    f"<b>Символ:</b> <a href=\"{link_url}\">{symbol}</a>\n"
                    f"<b>Пользователь:</b> {self.user_id}\n"
                    f"<b>Время:</b> {time_str}\n"
                    f"<b>Цена открытия:</b> {avg_price}\n"
                    f"<b>Объём:</b> {vol_str}\n"
                    f"<b>Тип открытия:</b> ШОРТ\n"
                    f"#{symbol}"
                )
            else:
                msg = (
                    f"🟩🔴 <b>Открытие позиции</b>\n"
                    f"<b>Символ:</b> <a href=\"{link_url}\">{symbol}</a>\n"
                    f"<b>Пользователь:</b> {self.user_id}\n"
                    f"<b>Время:</b> {time_str}\n"
                    f"<b>Тип открытия:</b> {s_side}\n"
                    f"#{symbol}"
                )
        elif s_result == "closed":
            msg = (
                f"❌ <b>Закрытие позиции</b>\n"
                f"<b>Символ:</b> <a href=\"{link_url}\">{symbol}</a>\n"
                f"<b>Пользователь:</b> {self.user_id}\n"
                f"<b>Время закрытия:</b> {time_str}\n"
                f"<b>Цена закрытия:</b> {avg_price}\n"
                f"<b>Объём:</b> {vol_str}\n"
                f"<b>Тип закрытия:</b> {s_manually}\n"
                f"#{symbol}"
            )
        elif s_result == "trailingstop":
            # Compute actual last price, PnL, and PnL percentage
            last_price = safe_to_float(
                self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
            )
            entry_price = safe_to_float(avg_price)
            vol = safe_to_float(volume)
            # Determine direction: Buy or Sell
            direction = 1 if s_side.lower() == "buy" else -1
            # Calculate PnL and PnL percentage
            pnl_val = (last_price - entry_price) * vol * direction
            try:
                pnl_pct_val = ((last_price - entry_price) / entry_price * 1000 * direction) if entry_price else Decimal(0)
            except Exception:
                pnl_pct_val = Decimal(0)
            msg = (
                f"🔄 <b>Установлен кастомный трейлинг-стоп</b>\n"
                f"<b>Символ:</b> <a href=\"{link_url}\">{symbol}</a>\n"
                f"<b>Пользователь:</b> {self.user_id}\n"
                f"<b>Цена входа:</b> {entry_price}\n"
                f"<b>Последняя цена:</b> {last_price}\n"
                f"<b>PnL:</b> {pnl_val:.4f} USDT ({pnl_pct_val:.2f}%)\n"
                f"<b>Объём:</b> {vol}\n"
                f"<b>Комментарий:</b> {action}"
            )
        else:
            msg = (
                f"🫡🔄 <b>Сделка</b>\n"
                f"<b>Символ:</b> <a href=\"{link_url}\">{symbol}</a>\n"
                f"<b>Результат:</b> {result}\n"
                f"<b>Действие:</b> {action}\n"
                f"<b>Закрытие:</b> {s_manually}"
            )

        try:
            await telegram_bot.send_message(self.user_id, msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            print(f"[log_trade] Ошибка Telegram: {e}")


    async def sync_open_positions_loop(self, interval: int = 5):
        while True:
            try:
                async with self.position_lock:
                    await self.update_open_positions()
            except Exception as e:
                logger.error(f"[sync] Critical error: {e}", exc_info=True)
            await asyncio.sleep(interval)


    async def set_trailing_stop(
        self,
        symbol: str,
        avg_price: float,
        pnl_pct: float,
        side: str
    ):
        """
        Устанавливает/обновляет трейлинг-стоп, работая только с float-значениями.

        Логика v18-patch-TS:

        • игнорируем попытку, если позиция уже закрылась / ещё не открылась;
        • дебаунсим одинаковые запросы: ≤ 2 с — пропуск;
        • новый стоп должен «улучшать» предыдущий (Buy → больше, Sell → меньше);
        • код Bybit 34040 (“not modified / already had stop”) считаем нормой и
          не спамим логами;
        • число одновременных REST-запросов ограничено семафором
          `_TRADING_STOP_SEM` (макс 3).
        """
        # ──────────────────────────────────────────────────────────────
        # 0. early-exit – позиция ещё не подтверждена
        if symbol in self.pending_orders:
            return False

        # 1. debounce ≤ 2 с для идентичного stop_price
        now = time.time()
        if not hasattr(self, "_last_trailing_ts"):
            self._last_trailing_ts = {}
        last_ts = self._last_trailing_ts.get(symbol, 0.0)
        if now - last_ts < 2.0:
            logger.debug("[trailing_stop] %s: cool-down skip (%.2fs)",
                         symbol, now - last_ts)
            return False

        try:
            logger.info("[trailing_stop] Попытка установки стопа %s | ROI=%.5f%%",
                        symbol, pnl_pct)

            pos = self.open_positions.get(symbol)
            if not pos:
                logger.warning("[trailing_stop] Позиция %s не найдена", symbol)
                return False

            volume = safe_to_float(pos.get("volume", 0))
            if volume <= 0:
                logger.warning("[trailing_stop] %s: объём 0 – пропуск", symbol)
                return False
            pos_idx = pos["pos_idx"]

            # --- расчёт trail-gap ------------------------------------
            base_trail = self.trailing_gap_pct
            reduction  = 0.0
            oi = self.shared_ws.latest_open_interest.get(symbol, 0.0)
            if oi > 1_000:
                reduction += 0.5

            candles = self.shared_ws.candles_data.get(symbol, [])
            if candles:
                last_price = safe_to_float(candles[-1]["closePrice"])
                if abs(last_price - avg_price) / avg_price * 100 > 1:
                    reduction += 0.5
            else:
                last_price = safe_to_float(
                    self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
                )

            final_trail = max(base_trail - reduction, 0.5)
            stop_pct    = round(pnl_pct - final_trail, 2)

            if side.lower() == "buy":
                raw_price = avg_price * (1 + stop_pct / 1000)
            elif side.lower() == "sell":
                raw_price = avg_price * (1 - stop_pct / 1000)
            else:
                logger.error("[trailing_stop] Unknown side %s", side)
                return False

            tick = float(DEC_TICK) if not isinstance(DEC_TICK, (int, float)) else DEC_TICK
            stop_price = round(math.floor(raw_price / tick) * tick, 6)

            logger.info("[trailing_stop] calc: base=%.2f red=%.2f final=%.2f "
                        "pct=%.2f price=%.6f", base_trail, reduction,
                        final_trail, stop_pct, stop_price)

            prev = self.last_stop_price.get(symbol)
            # 2. skip if identical within price-tick
            if prev is not None and abs(prev - stop_price) < tick:
                logger.debug("[trailing_stop] %s unchanged (%.6f) – skip",
                             symbol, stop_price)
                return False
            # 3. don’t worsen
            if prev is not None:
                worse = (
                    (side.lower() == "buy"  and stop_price < prev) or
                    (side.lower() == "sell" and stop_price > prev)
                )
                if worse:
                    logger.info("[trailing_stop] %s: хуже предыдущего – пропуск", symbol)
                    return False
            # # Определяем способ установки стопа в зависимости от режима
            # if self.mode == "real":
            #     # Режим REAL - используем WebSocket
            #     try:
            #         header = {
            #             "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
            #             "X-BAPI-RECV-WINDOW": "5000"
            #         }
            #         args = {
            #             "category": "linear",
            #             "symbol": symbol,
            #             "positionIdx": pos_idx,
            #             "stopLoss": str(stop_price),
            #             "triggerBy": "LastPrice",
            #             "timeInForce": "GTC"
            #         }

            #         req = {
            #             "op": "position.trading-stop",
            #             "header": header,
            #             "args": [args]
            #         }

            #         await self.ws_trade.send(json.dumps(req))
            #         async with self._recv_lock:
            #             resp = json.loads(await self.ws_trade.recv())
                    
            #         if resp.get("retCode", 0) == 0:
            #             self.last_stop_price[symbol] = stop_price
            #             logger.info("[trailing_stop] WS стоп установлен: %s | %.6f", symbol, stop_price)
            #             return True
            #         else:
            #             logger.error("[trailing_stop] WS ошибка: %s", resp)
            #             return False
            #     except Exception as e:
            #         logger.error("[trailing_stop] WS ошибка: %s", str(e))
            #         return False
            # else:

            async with _TRADING_STOP_SEM:
                try:
                    resp = await asyncio.to_thread(
                        lambda: self.session.set_trading_stop(
                            category="linear",
                            symbol=symbol,
                            positionIdx=pos_idx,
                            stopLoss=f"{stop_price:.6f}",
                            triggerBy="LastPrice",
                            timeInForce="GTC",
                        )
                    )
                except InvalidRequestError as exc:
                    if "not modified" in str(exc).lower() or "already" in str(exc).lower():
                        # стоп уже такой же – тихо завершаем
                        self.last_stop_price[symbol] = stop_price
                        self._last_trailing_ts[symbol] = now
                        return False
                    logger.warning("[trailing_stop] %s InvalidRequest: %s", symbol, exc)
                    return False
                except Exception as exc:
                    logger.warning("[trailing_stop] %s unexpected error: %s", symbol, exc)
                    return False

            ret = resp.get("retCode", resp.get("ret_code", 0))
            if ret in (0, 34040):
                self.last_stop_price[symbol] = stop_price
                self._last_trailing_ts[symbol] = now
                if ret == 0:
                    logger.info("[trailing_stop] %s stop set @ %.6f", symbol, stop_price)
                else:
                    logger.debug("[trailing_stop] %s already had %.6f", symbol, stop_price)
                return True

            logger.warning("[trailing_stop] %s failed: retCode=%s, resp=%s",
                           symbol, ret, resp)
            return False

        except Exception as e:
            logger.error("[trailing_stop] Critical error: %s", e, exc_info=True)
            return False


    async def health_check(self):
        while True:
            logger.info(
                f"[HealthCheck] Open positions: {len(self.open_positions)} "
                f"Pending orders: {len(self.pending_orders)}"
            )
            await asyncio.sleep(60)


    async def pnl_loop(self):
        """
        Periodically evaluate all open positions for updated PnL.
        """
        while True:
            # For each open position, schedule evaluation
            for symbol, data in list(self.open_positions.items()):
                position = {
                    "symbol": symbol,
                    "size":   str(data.get("volume", data.get("size", 0))),
                    "side":   data.get("side", "")
                }
                # Fire-and-forget
                asyncio.create_task(self.evaluate_position(position))
            # Wait one second between checks
            await asyncio.sleep(0.5)


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
        Фоновая задача переобучения GoldenNet.

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

                model = GoldenNet(input_size=INPUT_DIM).to(DEVICE)
                opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
                lossF = torch.nn.MSELoss()

                X_t = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
                y_t = torch.tensor(y,         dtype=torch.float32, device=DEVICE)

                # ---------- обучение ----------
                EPOCHS = 25
                for epoch in range(1, EPOCHS + 1):
                    opt.zero_grad()
                    pred = model(X_t)
                    l = lossF(pred, y_t)
                    l.backward()
                    opt.step()

                    # логируем каждую 5-ю эпоху (и первую)
                    if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
                        logger.info(
                            "[retrain] epoch %02d/%02d — loss=%.6f",
                            epoch, EPOCHS, l.item()
                        )

                # ---------- сохранение ----------
                ckpt = {"model_state": model.cpu().state_dict(), "scaler": scaler}
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    torch.save(ckpt, tmp.name)
                    os.replace(tmp.name, "golden_model_v1.pt")

                # ---------- hot-swap для инференса ----------
                self.ml_inferencer.model.load_state_dict(model.state_dict())
                self.ml_inferencer.scaler = scaler

                for _ in range(len(batch)):
                   self.training_data.popleft()
                for sym, hist in self.shared_ws.oi_history.items():
                    arr = np.asarray(hist, dtype=float)
                    if len(arr) >= 60:
                        dif = np.diff(arr[-60:]) / arr[-60:-1]
                        self._oi_sigma[sym] = float(np.std(dif))
                logger.info(
                    "[ML] GoldenNet retrained on %d samples (final loss=%.6f)",
                    buf_len,
                    l.item(),
                )

            except Exception:
                # полный stack-trace в лог, чтобы не терять ошибки
                logger.exception("[retrain] unexpected error during training")

            # гарантированная пауза между циклами (даже после ошибки)
            await asyncio.sleep(every_sec)


class TradingEnsemble(nn.Module):
    """Многоуровневая нейросетевая модель для трейдинга"""
    def __init__(self, input_size: int, tech_size: int, fund_size: int):
        super().__init__()
        # LSTM для временных рядов
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        
        # Ветвь для технических индикаторов
        self.tech_nn = nn.Sequential(
            nn.Linear(tech_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Ветвь для фундаментальных данных
        self.fundamental_nn = nn.Sequential(
            nn.Linear(fund_size, 16),
            nn.ReLU()
        )
        
        # Объединяющий классификатор
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 класса: BUY, SELL, HOLD
            nn.Softmax(dim=1)
        )
    
    def forward(self, x_time_series, x_technical, x_fundamental):
        # Обработка временных рядов
        lstm_out, _ = self.lstm(x_time_series)
        time_features = lstm_out[:, -1, :]
        
        # Технические индикаторы
        tech_features = self.tech_nn(x_technical)
        
        # Фундаментальные данные
        fund_features = self.fundamental_nn(x_fundamental)
        
        # Объединение фич
        combined = torch.cat((time_features, tech_features, fund_features), dim=1)
        return self.classifier(combined)


def load_users_from_json(json_path: str = "user_state.json") -> list:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)
    result = []
    for uid, data in all_users.items():
        if not data.get("banned") and data.get("registered"):
            result.append({
                "user_id": uid,
                "api_key": data.get("api_key"),
                "api_secret": data.get("api_secret"),
                "strategy": data.get("strategy"),
                "volume": data.get("volume"),
                "max_total_volume": data.get("max_total_volume"),
                "mode": data.get("mode", "real")
            })
    return result

# --- Dataset-класс под Torch ---
class GoldenDataset(Dataset):
    """
    Загружает CSV-датасет и формирует тензоры X (признаки) и y (метки).
    """
    def __init__(self, csv_path: str):
        super().__init__()

        # Читаем CSV и заполняем NaN нулями
        self.df = pd.read_csv(csv_path).fillna(0.0)

        # Все столбцы, кроме label, считаем признаками
        self.feature_cols = [c for c in self.df.columns if c != "label"]
        if not self.feature_cols:
            raise ValueError("В CSV нет признаков (кроме 'label').")

        self.X = torch.from_numpy(
            self.df[self.feature_cols].values.astype(np.float32)
        )
        self.y = torch.from_numpy(
            self.df["label"].values.astype(np.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


# async def run_all():
#     users = load_users_from_json("user_state.json")
#     if not users:
#         print("❌ Нет активных пользователей для запуска.")
#         return
    
#     golden_param_store = load_golden_params()
#     bots = [TradingBot(user_data=u, shared_ws=None, golden_param_store=golden_param_store) for u in users]
#     #symbols = await bots[0].get_selected_symbols() if bots else []
#     initial_symbols = ["BTCUSDT", "ETHUSDT"]           # минимальный старт
#     # создаём публичный WS сразу
#     shared_ws = PublicWebSocketManager(symbols=initial_symbols)

#     # создаём бота, передавая WS
#     bot = TradingBot(user_data=users[0], shared_ws=shared_ws, golden_param_store=golden_param_store)

#     # устанавливаем обратную ссылку
#     shared_ws.bot = bot

#     bots.append(bot)
    
#     await shared_ws.backfill_history()
#     public_ws_task = asyncio.create_task(shared_ws.start())
    
#     for bot in bots:
#         bot.shared_ws = shared_ws
#         shared_ws.position_handlers.append(bot)  # register for ticker-based evaluate_position
    
#     bot_tasks = [asyncio.create_task(bot.start()) for bot in bots]
    
#     # Подготовка к завершению
#     async def shutdown():
#         logger.info("Завершение работы всех ботов...")
#         for bot in bots:
#             await bot.stop()
#         public_ws_task.cancel()
#         await asyncio.gather(*bot_tasks, public_ws_task, return_exceptions=True)
#         logger.info("Все задачи остановлены")

# async def run_all():
#     users = load_users_from_json("user_state.json")
#     if not users:
#         print("❌ Нет активных пользователей для запуска.")
#         return
    
#     golden_param_store = load_golden_params()
#     bots = []

#     initial_symbols = ["BTCUSDT", "ETHUSDT"]
#     shared_ws = PublicWebSocketManager(symbols=initial_symbols)

#     # создаём всех ботов — но уже сразу передаем WS
#     for u in users:
#         bot = TradingBot(user_data=u, shared_ws=shared_ws, golden_param_store=golden_param_store)
#         bots.append(bot)

#     # создаём обратную связь для WS
#     for bot in bots:
#         shared_ws.position_handlers.append(bot)
    
#     # тут можно сделать общую привязку shared_ws → bot
#     # если нужен доступ к какому-то одному главному боту
#     shared_ws.bot = bots[0]   # если у тебя один основной
#     # или shared_ws.bots = bots  если хочешь сделать список ботов внутри shared_ws

#     await shared_ws.backfill_history()
#         # ───────────────── ML: dataset + модель (единожды на запуск) ─────────────
#     CSV_PATH    = "trainset.csv"
#     SCALER_PATH = "scaler.pkl"
#     MODEL_PATH  = "golden_model_v18.pt"

#     main_bot = bots[0]              # первый бот отвечает за подготовку

#     if not (os.path.exists(CSV_PATH) and os.path.exists(SCALER_PATH)):
#         logger.info("[ML] Building dataset and scaler…")
#         await main_bot.build_and_save_trainset(
#             csv_path=CSV_PATH,
#             scaler_path=SCALER_PATH,
#             symbol=list(shared_ws.active_symbols),
#             future_horizon=3,
#             future_thresh=0.005
#         )

#     scaler = _safe_load_scaler(SCALER_PATH)
#     logger.info("[ML] Scaler initialised – %s",
#                 "loaded" if hasattr(scaler, "mean_") else "fresh/identity")

#     logger.info("[ML] Training model…")
#     main_bot.train_model(
#         #csv_path=trainset_path,
#         num_epochs=30,
#         batch_size=64,
#         lr=1e-3,
#     )
#     logger.info("[ML] Model saved to %s", MODEL_PATH)

#     # ── раздаём новую модель всем ботам ─────────────────────────────
#     for b in bots:
#         b.load_ml_models()
#     # ────────────────────────────────────────────────────────────────
#     public_ws_task = asyncio.create_task(shared_ws.start())
    
#     bot_tasks = [asyncio.create_task(bot.start()) for bot in bots]
    
#     # Подготовка к завершению
#     async def shutdown():
#         logger.info("Завершение работы всех ботов...")
#         for bot in bots:
#             await bot.stop()
#         public_ws_task.cancel()
#         await asyncio.gather(*bot_tasks, public_ws_task, return_exceptions=True)
#         logger.info("Все задачи остановлены")

#     # Запуск Telegram-бота
#     #dp.include_router(router)
#     #dp.include_router(router_admin)
    
#     async def run_with_shutdown():
#         try:
#             tg_task = asyncio.create_task(dp.start_polling(telegram_bot))
#             await asyncio.gather(*[public_ws_task, *bot_tasks, tg_task])
#         except asyncio.CancelledError:
#             await shutdown()
#         except Exception as e:
#             logger.error(f"Критическая ошибка: {e}", exc_info=True)
#             await shutdown()
    
#     # Запуск основного цикла
#     loop = asyncio.get_event_loop()
#     loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown()))
#     loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(shutdown()))
    
#     await run_with_shutdown()

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

    # ───────────────── ML: датасет + модель ────────────────────────────
    TRAINSET_PATH = "trainset.csv"
    SCALER_PATH   = "scaler.pkl"
    MODEL_PATH    = "golden_model_v18.pt"

    main_bot = bots[0]

    if not (os.path.exists(TRAINSET_PATH) and os.path.exists(SCALER_PATH)):
        logger.info("[ML] Building dataset and scaler…")
        await main_bot.build_and_save_trainset(
            csv_path       = TRAINSET_PATH,
            scaler_path    = SCALER_PATH,
            symbol         = list(shared_ws.active_symbols),
            future_horizon = 3,
            future_thresh  = 0.005,
        )

    scaler = _safe_load_scaler(SCALER_PATH)
    logger.info("[ML] Scaler initialised – %s",
                "loaded" if hasattr(scaler, "mean_") else "fresh/identity")

    logger.info("[ML] Training model…")
    main_bot.train_model(
        csv_path   = TRAINSET_PATH,
        num_epochs = 30,
        batch_size = 64,
        lr         = 1e-3,
        model_path = MODEL_PATH,
    )
    logger.info("[ML] Model saved to %s", MODEL_PATH)

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

    import signal, functools, asyncio as _aio
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