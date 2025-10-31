#!/usr/bin/env python3

# ----------------- НАЧАЛО ФАЙЛА MultiuserBot_READY6_REFACTORED.py -----------------

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
import math
import random
import signal
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_DOWN, ROUND_HALF_UP, ROUND_FLOOR

from aiolimiter import AsyncLimiter
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

# +++ MLX ИМПОРТЫ +++
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
MODEL_PATH_PYTORCH = "golden_model_v19.pt"
MODEL_PATH_MLX = "golden_model_mlx.safetensors"
SCALER_PATH = "scaler.pkl"
ML_GATE_MIN_SCORE = 0.015  # Минимальный ожидаемый PnL% от ML-модели для передачи сигнала на AI-оценку

# ── Compute device selection ──────────────────────────────────────────────
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

# Global scaler placeholder; will be loaded in __main__
scaler: StandardScaler = None

# ======================================================================
# == МАШИННОЕ ОБУЧЕНИЕ: КОМПОНЕНТЫ ДЛЯ PYTORCH И MLX
# ======================================================================

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

# ----------------------------------------------------------------------
# --- Общая архитектура модели (используется и в PyTorch, и в MLX)
# ----------------------------------------------------------------------
class GoldenNetBase:
    def __init__(self, input_size: int, hidden_size: int = 64):
        self.input_size = input_size
        self.hidden_size = hidden_size

# ----------------------------------------------------------------------
# --- Компоненты для PyTorch
# ----------------------------------------------------------------------
class GoldenNet(nn.Module, GoldenNetBase):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        GoldenNetBase.__init__(self, input_size, hidden_size)
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

class PyTorchInferencer:
    def __init__(self, model_path: str = MODEL_PATH_PYTORCH, scaler_path: str = SCALER_PATH):
        self.device = DEVICE
        self.input_dim = len(FEATURE_KEYS)
        self.model = None
        self.scaler = None
        
        if Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning(f"[PyTorch] Файл модели {model_path} не найден.")
        
        if Path(scaler_path).exists():
            self.scaler = _safe_load_scaler(scaler_path)
            logger.info(f"[PyTorch] Скейлер {scaler_path} загружен.")

    def _load_model(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state", ckpt)
            self.model = GoldenNet(input_size=self.input_dim)
            self.model.to(self.device)
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            logger.info(f"[PyTorch] Модель из {path} успешно загружена.")
        except Exception as e:
            logger.error(f"[PyTorch] Ошибка загрузки модели из {path}: {e}", exc_info=True)
            self.model = None

    def infer(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([[0.0]])
        self.model.eval()
        if self.scaler:
            features = self.scaler.transform(features)
        input_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor)
        return prediction.cpu().numpy()

def train_golden_model_pytorch(training_data, num_epochs: int = 30, lr: float = 1e-3):
    logger.info("[PyTorch] Запуск обучения на PyTorch...")
    feats = np.asarray([d["features"] for d in training_data], dtype=np.float32)
    targ = np.asarray([d["target"] for d in training_data], dtype=np.float32)
    mask = ~(np.isnan(feats).any(1) | np.isinf(feats).any(1))
    feats, targ = feats[mask], targ[mask]
    if feats.size == 0:
        raise ValueError("train_golden_model_pytorch: нет валидных сэмплов")

    scaler = StandardScaler().fit(feats)
    feats_scaled = scaler.transform(feats).astype(np.float32)
    targ = np.clip(targ, -3.0, 3.0).astype(np.float32).reshape(-1, 1) # PnL% уже в %

    ds = TensorDataset(torch.tensor(feats_scaled), torch.tensor(targ))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = GoldenNet(input_size=feats.shape[1]).to(DEVICE)
    optim_ = optim.AdamW(model.parameters(), lr=lr)
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
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1} [PyTorch] – Loss: {loss_sum:.5f}")

    return model, scaler

# ----------------------------------------------------------------------
# --- Компоненты для MLX
# ----------------------------------------------------------------------
class GoldenNetMLX(mlx_nn.Module, GoldenNetBase):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        GoldenNetBase.__init__(self, input_size, hidden_size)
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
    def __init__(self, model_path=MODEL_PATH_MLX, scaler_path=SCALER_PATH):
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
            self.scaler = _safe_load_scaler(scaler_path)
            logger.info(f"[MLX] Скейлер из {scaler_path} успешно загружен.")

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
    targ = np.clip(targ, -3.0, 3.0).astype(np.float32).reshape(-1, 1) # PnL% уже в %

    model = GoldenNetMLX(input_size=feats_scaled.shape[1])
    optimizer = mlx_optim.Adam(learning_rate=lr)
    
    # MSE loss для регрессии
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
                asyncio.create_task(self.manage_symbol_selection())
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


    # async def manage_symbol_selection():
    #     http = HTTP(testnet=False)          # ← это важно
    #     while True:
    #         await asyncio.sleep(check_interval)

    #         # --- REST‑снимок всех тикеров ---
    #         try:
    #             resp = await asyncio.to_thread(
    #                 lambda: http.get_tickers(category="linear", symbol=None)
    #             )
    #             for tk in resp["result"]["list"]:
    #                 self.ticker_data[tk["symbol"]] = tk
    #         except Exception as e:
    #             logger.warning("[manage_symbol_selection] REST error: %s", e)
    #             continue

    #         # select symbols that satisfy liquidity thresholds and are not too fresh
    #         new_set = {
    #             s for s, t in self.ticker_data.items()
    #             if safe_to_float(t.get("turnover24h", 0)) >= min_turnover
    #             and safe_to_float(t.get("volume24h", 0))   >= min_volume
    #         } or self.active_symbols  # fallback if empty

    #         # ── не отписываемся от символов, по которым есть открытые позиции ──
    #         open_pos_symbols = {s
    #             for bot in self.position_handlers
    #             for s   in bot.open_positions.keys()}
    #         new_set |= open_pos_symbols            # объединяем множества
    #         if not new_set:                        # fallback safety
    #             new_set = self.active_symbols

    #         if not self.ready_event.is_set():
    #             # первый проход: считаем, что данные получены, пары выбраны
    #             self.ready_event.set()

    #         if new_set != self.active_symbols:
    #             added = new_set - self.active_symbols
    #             removed = self.active_symbols - new_set
    #             now = time.time()
    #             # debounce minor churn to avoid WS reconnect storms
    #             if (len(added) + len(removed) < 3) and (now - self._last_resubscribe_ts < 600):
    #                 # skip resubscribe this time; keep previous subscriptions stable
    #                 logger.info("[manage_symbol_selection] change muted (Δ=%d, cooldown %.0fs left)",
    #                             len(added)+len(removed), max(0, 600 - (now - self._last_resubscribe_ts)))
    #             else:
    #                 logger.info("[manage_symbol_selection] resubscribing: %d → %d symbols",
    #                             len(self.active_symbols), len(new_set))
    #                 self._last_resubscribe_ts = now
    #                 self.active_symbols = new_set
    #                 # update .symbols property and back‑fill historical data
    #                 self.symbols = list(new_set)
    #                 # backfill only for newly added symbols
    #                 if added:
    #                     asyncio.create_task(self.backfill_history(symbols=list(added)))
    #             try:
    #                 # close old socket and open a new one with same callback
    #                 if self.ws:
    #                     self.ws.exit()
    #                 self.ws = WebSocket(
    #                     testnet=False,
    #                     channel_type="linear",
    #                     ping_interval=30,
    #                     ping_timeout=15,
    #                     restart_on_error=True,
    #                     retries=200
    #                 )
    #                 self.ws.kline_stream(interval=self.interval,
    #                                      symbol=list(new_set),
    #                                      callback=self._callback)
    #                 self.ws.ticker_stream(symbol=list(new_set),
    #                                       callback=self._callback)
    #                 self.ws.all_liquidation_stream(symbol=list(new_set),
    #                                                 callback=self._callback)
    #             except Exception as e:
    #                 logger.warning("[manage_symbol_selection] WS resubscribe failed: %s", e)


    async def manage_symbol_selection(self):
        desired = self._compute_desired_symbols()          # как у тебя уже сделано
        current = getattr(self, "ws_subscribed", set())
        if desired == current:
            return

        now = time.time()
        if now < getattr(self, "_next_resub_ts", 0.0):
            left = int(self._next_resub_ts - now)
            # запоминаем, что хотим применить, но позже
            self._pending_symbols = desired
            logger.info("[manage_symbol_selection] change muted (Δ=%d, cooldown %ss left)",
                        len(desired - current), left)
            return  # <<< КРИТИЧЕСКО: НИЧЕГО НЕ ДЕЛАЕМ — НЕТ reconnect/resubscribe

        # кулдаун истек — применяем изменения единожды
        self._next_resub_ts = now + self.RESUB_COOLDOWN
        self._pending_symbols = None

        if hasattr(self.shared_ws, "resubscribe_idempotent"):
            await self.shared_ws.resubscribe_idempotent(desired)  # переподписка только если реально изменилось
        else:
            await self.shared_ws.apply_subscriptions(desired)     # твой текущий метод

        self.ws_subscribed = desired
        logger.info("[manage_symbol_selection] resubscribing: %d → %d symbols", len(current), len(desired))


    async def resubscribe_idempotent(self, desired: set[str]):
        desired = set(desired)
        current = getattr(self, "subscribed", set())
        if desired == current:
            return  # ничего не делаем, сокет не трогаем
        # далее твоя логика отписки/подписки, и только затем, если нужно — reconnect
        await self.apply_subscriptions(desired)
        self.subscribed = desired


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


    async def backfill_history(self, symbols: list[str] | None = None):
        http = HTTP(testnet=False)
        symbols = symbols or self.symbols
        for symbol in symbols:
            recent = self.candles_data.get(symbol, [])
            last_ts = recent[-1]['startTime'] if recent else None
            last_ms = int(last_ts.timestamp()*1000) if last_ts is not None else None
            try:
                params = {'symbol': symbol, 'interval': self.interval}
                if last_ms is not None:
                    # avoid duplicate inclusive start; step by one interval
                    try:
                        _imap = {"1":60000,"3":180000,"5":300000,"15":900000,"30":1800000,"60":3600000,"120":7200000,"240":14400000,"720":43200000,"D":86400000}
                        step = _imap.get(str(self.interval), 60000)
                    except Exception:
                        step = 60000
                    params['start'] = last_ms + step
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
                    # skip duplicates / overlaps
                    if last_ts is not None and ts <= last_ts:
                        continue
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


# ---------------------- TRADING BOT ----------------------
class TradingBot:

    __slots__ = (
        "user_data", 

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
        "coreml_model", "feature_scaler", "last_retrain", "training_data", "device", 
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
        "ml_gate_abs_roi", "ml_gate_min_score", "ml_gate_sigma_coef", "leverage",
        "_entry_locks", "_exec_locks", "_pending_entries", "_order_timeout_sec", "_last_market_heartbeat", "_inflight_until",
        "stats", "stats_task", "RESUB_COOLDOWN", "_next_resub_ts", "_pending_symbols",
    )



    def __init__(self, user_data, shared_ws, golden_param_store):
        self.stats = defaultdict(int)
        # entry/exec guards & settings
        self.user_data = user_data or {}
        self.stats = getattr(self, 'stats', None) or defaultdict(int)
        self._entry_locks = {}
        self._exec_locks = {}
        self._pending_entries = {}
        self._order_timeout_sec = 20
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "real")
        self.listing_age_min = int(user_data.get("listing_age_min_minutes", LISTING_AGE_MIN_MINUTES))
        
        # +++ ИЗМЕНЕНИЕ: ВЫБОР ML ФРЕЙМВОРКА +++
        self.ml_framework = user_data.get("ml_framework", "mlx").lower() # По умолчанию MLX
        logger.info(f"[User {self.user_id}] Выбран ML фреймворк: {self.ml_framework.upper()}")
        if self.ml_framework == 'mlx':
            self.ml_inferencer = MLXInferencer()
        else:
            self.ml_inferencer = PyTorchInferencer()
        # +++ КОНЕЦ ИЗМЕНЕНИЯ +++
        
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
        self._inflight_until = {}  # symbol -> unix_ts до которого считаем запрос «в полёте»

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

        prov = (self.user_data.get("ai_provider") or "ollama").strip().lower()
        # Normalize common typos (e.g., Cyrillic 'о' in "оllama")
        try:
            prov_norm = prov.replace("о", "o")  # Cyrillic o -> Latin o
        except Exception:
            prov_norm = prov
        prov = prov_norm

        # починка потенциальной кириллицы: 'оllama' -> 'ollama'
        prov = prov.replace("о", "o")  # кириллическая 'о' -> латинская 'o'
        prov = prov.replace("о", "o")
        self.ai_provider = prov

        # используем OpenAI-совместимый эндпоинт, как в ваших логах

        self.RESUB_COOLDOWN = int(os.getenv("WS_RESUB_COOLDOWN", "480"))  # или твое значение
        self._next_resub_ts = 0.0
        self._pending_symbols = None

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
        self.device = DEVICE
        
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
        prov = prov.replace("о", "o")
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
        self.coreml_model = None
        self.feature_scaler = None
        # self.load_ml_models() # Загрузка будет в __init__ через выбор фреймворка
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
    def _inflight_check(self, symbol: str) -> bool:
        import time
        exp = self._inflight_until.get(symbol, 0.0)
        return exp > time.time()

    def _inflight_open(self, symbol: str, ttl: float = 3.0):
        import time
        self._inflight_until[symbol] = time.time() + ttl


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

    async def place_unified_order_guarded(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        comment: str = ""
    ):
        """
        Одна позиция на пару без «долгого pending».
        Короткий inflight-дебаунс + orderLinkId для идемпотентности.
        """
        import time, asyncio, uuid

        # 0) Уже открытая позиция — выходим
        pos = self.open_positions.get(symbol)
        if pos and safe_to_float(pos.get("volume", 0)) > 0:
            logger.info(f"[ENTRY_GUARD] already open position for {symbol}, skip")
            return None

        # 1) Короткий «inflight» (2–3 c), чтобы не отправлять 2 запроса параллельно
        if self._inflight_check(symbol):
            logger.info(f"[ENTRY_GUARD] inflight exists for {symbol}, skip")
            return None

        entry_lock = self._get_entry_lock(symbol)
        async with entry_lock:
            # перепроверка под локом
            pos = self.open_positions.get(symbol)
            if pos and safe_to_float(pos.get("volume", 0)) > 0:
                logger.info(f"[ENTRY_GUARD] already open position for {symbol}, skip")
                return None
            if self._inflight_check(symbol):
                logger.info(f"[ENTRY_GUARD] inflight exists for {symbol}, skip")
                return None

            self._inflight_open(symbol, ttl=3.0)

            # 2) Идемпотентность на стороне Bybit
            # Желательно ключ по свече, чтобы «сигнал на новой свече» был новым ID
            candle_key = int(time.time()) // 60  # замените на ваш ts последнего бара, если доступен
            order_link_id = f"bot:{self.user_id}:{symbol}:{side}:{candle_key}:{uuid.uuid4().hex[:8]}"

            try:
                logger.info(f"[ORDER_SEND] {symbol} {side} qty={qty} type={order_type} linkId={order_link_id} comment='{comment}'")

                # ВАЖНО: пробросьте orderLinkId в ваш метод/place API
                result = await asyncio.wait_for(
                    self.place_unified_order(
                        symbol, side, qty, order_type,
                        comment=comment,
                        orderLinkId=order_link_id  # ← добавьте поддержку в place_unified_order()
                    ),
                    timeout=self._order_timeout_sec
                )
                logger.info(f"[ORDER_RESP] {symbol} {side} -> {str(result)[:300]}")

            except asyncio.TimeoutError:
                logger.error(f"[ORDER_TIMEOUT] {symbol} {side} qty={qty} type={order_type} exceeded {self._order_timeout_sec}s")
                return None
            except Exception as e:
                logger.exception(f"[ORDER_ERROR] {symbol} {side} failed: {e}")
                return None
            finally:
                # inflight держим только пока выполнялся запрос к API
                self._inflight_until.pop(symbol, None)

        # 3) Фоновая валидация открытия позиции (НЕ держим никаких pending)
        async def _confirm_open():
            try:
                deadline = time.time() + 20  # короче, чем раньше
                while time.time() < deadline:
                    p = self.open_positions.get(symbol)
                    if p and safe_to_float(p.get("volume", 0)) > 0:
                        logger.info(f"[OPEN_CONFIRMED] {symbol} side={side} vol={p.get('volume')} price={p.get('avg_price')}")
                        return
                    await asyncio.sleep(1.0)
                logger.warning(f"[OPEN_WAIT_TIMEOUT] {symbol} side={side} not visible in positions after 20s; will rely on next poll/WS")
            except Exception:
                logger.exception(f"[OPEN_CHECK_ERROR] {symbol} side={side}")

        asyncio.create_task(_confirm_open())
        return result
                

    def _get_entry_lock(self, symbol: str):
        import asyncio
        lock = self._entry_locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._entry_locks[symbol] = lock
        return lock

    def _get_exec_lock(self, symbol: str):
        import asyncio
        lock = self._exec_locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._exec_locks[symbol] = lock
        return lock

    def _set_pending(self, key: str, ttl: float = 45.0):
        import time
        exp = time.time() + ttl
        self._pending_entries[key] = exp
        logger.debug(f"[PENDING_SET] {key} ttl={ttl}s until={exp}")

    def _clear_pending(self, key: str):
        if self._pending_entries.pop(key, None) is not None:
            logger.debug(f"[PENDING_CLEAR] {key}")


    def _is_pending(self, key: str) -> bool:
        import time
        exp = self._pending_entries.get(key, 0.0)
        if exp and time.time() < exp:
            return True
        if exp:
            self._pending_entries.pop(key, None)
        return False

    def set_market_heartbeat(self):
        import time
        self._last_market_heartbeat = time.time()
        return self._last_market_heartbeat

    async def market_watchdog(self, heartbeat_ref, timeout_sec: int = 180):
        import time, asyncio
        while True:
            await asyncio.sleep(10)
            try:
                last = float(heartbeat_ref() or 0)
                if last and (time.time() - last) > timeout_sec:
                    logger.error(f"[WATCHDOG] market loop stalled > {timeout_sec}s; requesting WS reconnect")
                    if hasattr(self, "ws") and hasattr(self.ws, "_force_reconnect"):
                        await self.ws._force_reconnect(reason="watchdog")
                    self._last_market_heartbeat = time.time()
            except Exception:
                logger.exception("[WATCHDOG] error")



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


    # ──────────────────────────────────────────────────────────────────
    # [ФИНАЛЬНАЯ ВЕРСИЯ] ВНУТРИ КЛАССА TRADINGBOT
    async def extract_realtime_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        [V7 - Объединенная и отказоустойчивая] Собирает все фичи, используя
        надежные методы расчета и проверки на наличие данных.
        """
        # ---- 0. Получение цены и спреда ----
        ticker = self.shared_ws.ticker_data.get(symbol)
        last_price = 0.0
        bid1 = 0.0
        ask1 = 0.0

        if ticker:
            last_price = safe_to_float(ticker.get("lastPrice", 0))
            bid1 = safe_to_float(ticker.get("bid1Price", 0))
            ask1 = safe_to_float(ticker.get("ask1Price", 0))

        if not last_price:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_tickers(category="linear", symbol=symbol)
                )
                if resp and resp.get("result", {}).get("list"):
                    ticker = resp["result"]["list"][0]
                    self.shared_ws.ticker_data[symbol] = ticker
                    last_price = safe_to_float(ticker.get("lastPrice", 0))
                    bid1 = safe_to_float(ticker.get("bid1Price", 0))
                    ask1 = safe_to_float(ticker.get("ask1Price", 0))
            except Exception:
                pass

        if not last_price:
            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if candles:
                last_price = safe_to_float(candles[-1].get("closePrice", 0))
                bid1 = last_price
                ask1 = last_price

        if not last_price or last_price <= 0:
            logger.warning(f"[features] Не удалось получить актуальную цену для {symbol}. Сбор фичей прерван.")
            return None

        spread_pct = (ask1 - bid1) / bid1 * 100 if bid1 > 0 else 0.0

        # --- 1. Базовые фичи ---
        pct1m = compute_pct(self.shared_ws.candles_data[symbol], 1)
        pct5m = compute_pct(self.shared_ws.candles_data[symbol], 5)
        pct15m = compute_pct(self.shared_ws.candles_data[symbol], 15)
        V1m = sum_last_vol(self.shared_ws.candles_data[symbol], 1)
        V5m = sum_last_vol(self.shared_ws.candles_data[symbol], 5)
        V15m = sum_last_vol(self.shared_ws.candles_data[symbol], 15)
        oi_hist = list(self.shared_ws.oi_history[symbol])
        OI_now = oi_hist[-1] if oi_hist else 0.0
        OI_prev1m = oi_hist[-2] if len(oi_hist) >= 2 else 0.0
        OI_prev5m = oi_hist[-6] if len(oi_hist) >= 6 else 0.0
        dOI1m = (OI_now - OI_prev1m) / OI_prev1m if OI_prev1m > 0 else 0.0
        dOI5m = (OI_now - OI_prev5m) / OI_prev5m if OI_prev5m > 0 else 0.0
        cvd_hist = list(self.shared_ws.cvd_history[symbol])
        CVD_now = cvd_hist[-1] if cvd_hist else 0.0
        CVD_prev1m = cvd_hist[-2] if len(cvd_hist) >= 2 else 0.0
        CVD_prev5m = cvd_hist[-6] if len(cvd_hist) >= 6 else 0.0
        CVD1m = CVD_now - CVD_prev1m
        CVD5m = CVD_now - CVD_prev5m
        sigma5m = self.shared_ws._sigma_5m(symbol)

        # --- 2. Технические индикаторы ---
        df = pd.DataFrame(list(self.shared_ws.candles_data[symbol])[-50:])
        rsi14 = ta.rsi(df["closePrice"], length=14).iloc[-1] if len(df) >= 15 else 50.0
        sma50 = ta.sma(df["closePrice"], length=50).iloc[-1] if len(df) >= 50 else (df["closePrice"].iloc[-1] if not df.empty else 0.0)
        ema20 = ta.ema(df["closePrice"], length=20).iloc[-1] if len(df) >= 20 else sma50
        atr14 = ta.atr(df["highPrice"], df["lowPrice"], df["closePrice"], length=14).iloc[-1] if len(df) >= 15 else 0.0
        bb_width = 0.0
        if len(df) >= 20:
            bb = ta.bbands(df["closePrice"], length=20)
            if bb is not None and not bb.empty:
                bb_width = bb["BBU_20_2.0"].iloc[-1] - bb["BBL_20_2.0"].iloc[-1]
        supertrend_val = compute_supertrend(df, period=10, multiplier=3).iloc[-1] if len(df) > 20 else False
        supertrend_num = 1 if supertrend_val else -1
        adx14 = ta.adx(df["highPrice"], df["lowPrice"], df["closePrice"], length=14)["ADX_14"].iloc[-1] if len(df) >= 15 else 0.0
        cci20 = ta.cci(df["highPrice"], df["lowPrice"], df["closePrice"], length=20).iloc[-1] if len(df) >= 20 else 0.0
        
        macd_val, macd_signal = 0.0, 0.0
        if len(df) >= 26:
            macd_block = ta.macd(df["closePrice"], fast=12, slow=26, signal=9)
            if macd_block is not None and not macd_block.empty and len(macd_block.columns) >= 3:
                macd_val = macd_block.iloc[-1, 0] if pd.notna(macd_block.iloc[-1, 0]) else 0.0
                macd_signal = macd_block.iloc[-1, 2] if pd.notna(macd_block.iloc[-1, 2]) else 0.0
        
        avgVol30m = self.shared_ws.get_avg_volume(symbol, 30)
        avgOI30m = sum(oi_hist[-30:]) / max(1, len(oi_hist[-30:]))
        deltaCVD30m = CVD_now - (cvd_hist[-31] if len(cvd_hist) >= 31 else 0.0)

        # --- 3. Golden-setup блок ---
        GS_pct4m = compute_pct(self.shared_ws.candles_data[symbol], 4)
        GS_vol4m = sum_last_vol(self.shared_ws.candles_data[symbol], 4)
        GS_dOI4m = (OI_now - (oi_hist[-5] if len(oi_hist) >= 5 else OI_now)) / max(1, (oi_hist[-5] if len(oi_hist) >= 5 else 1))
        GS_cvd4m = CVD_now - (cvd_hist[-5] if len(cvd_hist) >= 5 else CVD_now)
        GS_supertrend_flag = supertrend_num
        GS_cooldown_flag = int(not self._golden_allowed(symbol))

        # --- 4. Squeeze блок ---
        SQ_power = abs(pct5m) * abs((V1m - V5m / 5) / max(1e-8, V5m / 5) * 100)
        SQ_strength = int(abs(pct5m) >= self.squeeze_threshold_pct and SQ_power >= self.squeeze_power_min)
        recent_liq_vals = [v_usdt for (ts, s, v_usdt, price_liq) in self.liq_buffers[symbol] if time.time() - ts <= 10]
        SQ_liq10s = sum(recent_liq_vals)
        SQ_cooldown_flag = int(not self._squeeze_allowed(symbol))

        # --- 5. Liquidation блок ---
        buf = self.liq_buffers[symbol]
        recent_all = [(ts, s, v_usdt, price_liq) for (ts, s, v_usdt, price_liq) in buf if time.time() - ts <= 10]
        same_side = [v_usdt for (ts, s, v_usdt, _price) in recent_all if recent_all and s == recent_all[-1][1]]
        LIQ_cluster_val10s = sum(same_side)
        LIQ_cluster_count10s = len(same_side)
        LIQ_direction = 1 if (recent_all and recent_all[-1][1] == "Buy") else -1
        LIQ_cooldown_flag = int(not self.check_liq_cooldown(symbol))

        # --- 6. Временные фичи ---
        now_ts = dt.datetime.now()
        hour_of_day = now_ts.hour
        day_of_week = now_ts.weekday()
        month_of_year = now_ts.month

        # --- 7. Собираем итоговый словарь ---
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
    
    
    # ------------------------------------------------------------------
    async def _capture_training_sample(self, symbol: str, pnl_pct: float) -> None:
        """
        Сохраняет (features, target) из закрытой сделки в self.training_data.
        Target = фактический PnL% сделки.
        """
        try:
            # Получаем фичи, которые были на МОМЕНТ ВХОДА в сделку
            pos = self.closed_positions.get(symbol)
            if not pos or "entry_features" not in pos:
                logger.warning(f"[ML_Train] Не найдены фичи для входа по закрытой сделке {symbol}")
                return

            entry_features_vector = pos["entry_features"]
            
            self.training_data.append({
                "features": entry_features_vector,
                "target":   pnl_pct,
            })
            logger.info(f"[ML_Train] Добавлен обучающий пример для {symbol}, PnL={pnl_pct:.2f}%. Всего: {len(self.training_data)}")

        except Exception as exc:
            logger.debug("[ML] feature extraction failed: %s", exc)
            logger.exception("[ML] _capture_training_sample failed")
            return


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


    # +++ УПРАВЛЕНИЕ СТОП-ЛОССАМИ +++
    # ВНИМАНИЕ: Эта логика остается здесь временно.
    # Для надежной работы в реальном времени ее необходимо вынести в отдельный сервис.
    # Это будет следующим шагом нашей работы.
    async def manage_open_position(self, symbol: str):
        """
        [ВРЕМЕННАЯ РЕАЛИЗАЦИЯ] "Няня" для управления стопами.
        ЗАМЕТКА: Зависимость от основного event loop делает ее ненадежной при лагах.
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


    async def set_trailing_stop(self, symbol: str, avg_price: float, pnl_pct: float, side: str) -> bool:
        """
        [ВРЕМЕННАЯ РЕАЛИЗАЦИЯ] Устанавливает трейлинг-стоп.
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

            last_price = safe_to_float(pos.get("markPrice", avg_price))
            if last_price <= 0: return False

            if side.lower() == "buy":
                raw_price = last_price * (1 - gap_pct / 1000)
            else: # sell
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
    # +++ КОНЕЦ БЛОКА УПРАВЛЕНИЯ СТОП-ЛОССАМИ +++


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


    async def sync_open_positions_loop(self):
        """Синхронизация self.open_positions через REST get_positions (Unified V5)."""
        import asyncio
        from math import fabs
        poll_secs = getattr(self, "_positions_poll_interval", 5.0) or 5.0
        def _to_float(x, default=0.0):
            try: return float(x)
            except Exception: return default
        while getattr(self, "running", True):
            try:
                if not (hasattr(self, "session") and hasattr(self.session, "get_positions")):
                    logger.warning("[SYNC] session.get_positions не доступен; пауза")
                    await asyncio.sleep(poll_secs); continue
                resp = await asyncio.to_thread(self.session.get_positions, category="linear", settleCoin="USDT")
                if int(resp.get("retCode", 0)) != 0:
                    logger.warning(f"[SYNC] get_positions retCode={resp.get('retCode')} retMsg={resp.get('retMsg')}")
                    await asyncio.sleep(poll_secs); continue
                lst = resp.get("result", {}).get("list", []) or []
                actual = {}
                for p in lst:
                    sym = p.get("symbol"); 
                    if not sym: continue
                    side = (p.get("side") or "").title()
                    size = _to_float(p.get("size") or p.get("qty") or 0.0, 0.0)
                    if abs(size) < 1e-12: continue
                    avg_price = (_to_float(p.get("avgPrice")) or _to_float(p.get("avg_entry_price")) or _to_float(p.get("entryPrice")) or 0.0)
                    actual[sym] = {"symbol": sym, "side": side, "volume": abs(size), "avg_price": avg_price}
                prev = self.open_positions or {}
                for sym, pos in actual.items():
                    if sym not in prev:
                        logger.info(f"[SYNC] Обнаружена новая активная позиция на бирже: {sym}")
                        try:
                            if hasattr(self, "adopt_existing_position"): await self.adopt_existing_position(sym, pos)
                            elif hasattr(self, "_adopt_position"):       await self._adopt_position(sym, pos)
                        except Exception:
                            logger.exception(f"[ADAPT] ошибка адаптации позиции {sym}")
                    else:
                        prev[sym].update(pos)
                for sym in list(prev.keys()):
                    if sym not in actual:
                        logger.info(f"[SYNC] Позиция закрыта на бирже: {sym} — удаляем локально")
                        prev.pop(sym, None)
                self.open_positions = prev
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[sync_open_positions_loop] unexpected error")
            await asyncio.sleep(poll_secs)

    async def _stats_reporter(self):
        import asyncio, time
        interval = 300  # 5 минут
        while getattr(self, "running", True):
            try:
                s = dict(self.stats) if isinstance(self.stats, dict) else {}
                logger.info("[STATS] candidates=%s core_pass=%s ai_confirm=%s risk_block=%s executed=%s orders=%s",
                            s.get("candidates", 0), s.get("core_pass", 0), s.get("ai_confirm", 0),
                            s.get("risk_block", 0), s.get("executed", 0), s.get("orders", 0))
            except Exception:
                pass
            await asyncio.sleep(interval)


    async def start(self):
        try:
            self.stats_task = asyncio.create_task(self._stats_reporter())
        except Exception:
            pass
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
                        if logger.isEnabledFor(logging.INFO):
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
                        try:
                            trade = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "symbol": symbol,
                                "side": pos["side"],
                                "qty": float(pos_volume),
                                "entry_price": float(safe_to_float(pos.get("avg_price", 0))),
                                "exit_price": float(exit_price),
                                "pnl": float(pnl_usdt),
                                "pnl_pct": float(pnl_pct),
                                "reason": "closed_by_execution"
                            }
                            self._save_trade(trade)
                        except Exception:
                            logger.exception("[trade] failed to persist closed_by_execution trade")


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


    async def handle_position_update(self, msg: dict):
        """
        [V8 - Refactored for Fine-tuning] Обрабатывает открытие, обновление и закрытие.
        При закрытии инициирует сбор данных для обучения ML и дообучения AI.
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
                    
                    # Получаем кандидата, который привел к открытию, из временного хранилища
                    entry_candidate = self.active_trade_entries.pop(symbol, {})
                    comment = entry_candidate.get("comment", "N/A")
                    
                    # Собираем и сохраняем фичи на момент входа
                    entry_features_vector = self._build_entry_features(symbol)

                    self.open_positions[symbol] = {
                        "avg_price": avg_price, "side": side_raw,
                        "pos_idx": 1 if side_raw == 'Buy' else 2,
                        "volume": new_size, "leverage": safe_to_float(p.get("leverage", "1")),
                        "entry_candidate": entry_candidate, # Сохраняем всего кандидата (включая промпт)
                        "markPrice": avg_price, "pnl": 0.0,
                        "entry_features": entry_features_vector # Сохраняем вектор фичей для ML
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

                # --- Сценарий 2: Обновление существующей позиции (усреднение) ---
                if prev_pos and new_size > 0 and abs(new_size - safe_to_float(prev_pos.get("volume", 0))) > 1e-9:
                    logger.info(f"[PositionStream] {symbol} volume updated: {prev_pos.get('volume')} -> {new_size}")
                    self.open_positions[symbol]["volume"] = new_size
                    self.open_positions[symbol]["avg_price"] = safe_to_float(p.get("avgPrice") or p.get("entryPrice"))
                    self.write_open_positions_json()
                    continue

                # --- Сценарий 3: Закрытие позиции ---
                if prev_pos and new_size == 0:
                    logger.info(f"[PositionStream] CLOSE: {symbol} closed (size=0).")
                    
                    snapshot = dict(prev_pos)
                    self.closed_positions[symbol] = snapshot
                    
                    self._purge_symbol_state(symbol)
                    self.write_open_positions_json()
                    
                    exit_price = safe_to_float(p.get("avgPrice") or snapshot.get("markPrice", snapshot.get("avg_price", 0)))
                    pos_volume = safe_to_float(snapshot.get("volume", 0))
                    entry_price = safe_to_float(snapshot.get("avg_price", 0))
                    pnl_usdt = self._calc_pnl(snapshot.get("side", "Buy"), entry_price, exit_price, pos_volume)
                    pos_value = entry_price * pos_volume
                    pnl_pct = (pnl_usdt / pos_value) * 100 if pos_value else 0.0

                    # +++ ЛОГИРОВАНИЕ ДЛЯ ОБУЧЕНИЯ +++
                    # 1. Для ML-модели (фичи + PnL)
                    asyncio.create_task(self._capture_training_sample(symbol, pnl_pct))
                    
                    # 2. Для дообучения AI (промпт + PnL)
                    entry_candidate = snapshot.get("entry_candidate", {})
                    ai_prompt = entry_candidate.get("full_prompt_for_ai")
                    source = entry_candidate.get("source", "unknown")
                    if ai_prompt:
                        log_for_finetune(prompt=ai_prompt, pnl_pct=pnl_pct, source=source)
                        logger.info(f"[FineTuneLog] Записан лог для дообучения AI по сделке {symbol}")
                    # +++ КОНЕЦ БЛОКА ЛОГИРОВАНИЯ +++

                    asyncio.create_task(self.log_trade(
                        symbol=symbol, side=snapshot.get("side", "Buy"), avg_price=exit_price,
                        volume=pos_volume, action="close", result="closed_by_position_stream",
                        pnl_usdt=pnl_usdt, pnl_pct=pnl_pct
                    ))
                    try:
                        trade = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "symbol": symbol,
                            "side": snapshot.get("side", "Buy"),
                            "qty": float(pos_volume),
                            "entry_price": float(entry_price),
                            "exit_price": float(exit_price),
                            "pnl": float(pnl_usdt),
                            "pnl_pct": float(pnl_pct),
                            "reason": "closed_by_position_stream"
                        }
                        self._save_trade(trade)
                    except Exception:
                        logger.exception("[trade] failed to persist closed_by_position_stream trade")


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
                        "open_timestamp": time.time(),
                        "entry_features": self._build_entry_features(symbol) # Сохраняем фичи для адаптированных позиций
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

            # --- ПЕРЕДАЧА СИГНАЛА В НОВЫЙ ПАЙПЛАЙН ---
            features = await self.extract_realtime_features(symbol)
            if not features: return

            candidate = {
                'symbol': symbol,
                'side': order_side,
                'source': 'liquidation',
                'base_metrics': {
                    'liquidation_cluster_value_usdt': cluster_val,
                    'dominant_liquidation_side': 'longs' if short_val > long_val else 'shorts'
                },
                'volume_usdt': self.POSITION_VOLUME
            }
            await self.process_signal_candidate(candidate, features)
            self.shared_ws.last_liq_trade_time[symbol] = dt.datetime.utcnow()


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
        2) Статические default-параметры Buy / Sell
        """
        # --- базовый источник ---
        base = (
            self.golden_param_store.get((symbol, side))
            or self.golden_param_store.get(side)
            or {"period_iters": 3, "price_change": 1.7,
                "volume_change": 200, "oi_change": 1.5}
        )
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
            return False
        if symbol in self.failed_orders and time.time() - self.failed_orders[symbol] < 600:
            return False
        if symbol in self.reserve_orders:
            return False
        return True


    # +++ НОВЫЙ ЕДИНЫЙ ПАЙПЛАЙН ОБРАБОТКИ СИГНАЛОВ +++
    async def process_signal_candidate(self, candidate: dict, features: dict):
        """
        Единый конвейер для обработки любого сигнала: ML-оценка -> AI-вердикт -> Исполнение.
        """
        symbol = candidate['symbol']
        side = candidate['side']
        source = candidate.get('source', 'unknown')
        signal_key = f"{symbol}_{side}_{source}_{int(time.time()) // 60}"

        # --- Дебаунс, чтобы не спамить оценками по одному и тому же сигналу в минуту ---
        if self._evaluated_signals_cache.get(signal_key):
            return
        self._evaluated_signals_cache[signal_key] = time.time()

        logger.info(f"[PIPELINE_START] {symbol}/{side} ({source})")

        # --- 1. Оценка ML-моделью (MLX или PyTorch) ---
        try:
            vector = np.array([[safe_to_float(features.get(k, 0.0)) for k in FEATURE_KEYS]], dtype=np.float32)
            # self.ml_inferencer уже правильного типа (MLX или PyTorch)
            expected_pnl_pct = float(self.ml_inferencer.infer(vector)[0][0])
            
            if not (-5.0 < expected_pnl_pct < 5.0): # Защита от аномальных прогнозов
                logger.warning(f"[ML_GATE] {symbol} отклонен: аномальный прогноз PnL% = {expected_pnl_pct:.2f}")
                return

            if abs(expected_pnl_pct) < ML_GATE_MIN_SCORE:
                logger.info(f"[ML_GATE] {symbol} отклонен: прогноз PnL% ({expected_pnl_pct:.2f}%) ниже порога ({ML_GATE_MIN_SCORE:.2f}%)")
                return
            
            logger.info(f"[ML_GATE] {symbol} одобрен: прогноз PnL% = {expected_pnl_pct:.2f}%")
            candidate['ml_score'] = expected_pnl_pct # Добавляем оценку в кандидата

        except Exception as e:
            logger.warning(f"[ML_GATE] Ошибка ML-фильтра для {symbol}: {e}", exc_info=True)
            return # Ошибка в ML - не продолжаем

        # --- 2. Финальный вердикт от AI (Ollama) ---
        try:
            ai_response = await self._ai_call_with_timeout("ollama", candidate, features)
            ai_action = ai_response.get("action", "REJECT")

            if ai_action != "EXECUTE":
                justification = ai_response.get("justification", "Причина не указана.")
                logger.info(f"[AI_REJECT] {symbol}/{side} ({source}) — {justification}")
                return
            
            logger.info(f"[AI_CONFIRM] {symbol}/{side} ({source}) ОДОБРЕН. {ai_response.get('justification')}")
            candidate['justification'] = ai_response.get("justification", "N/A")
            candidate['full_prompt_for_ai'] = ai_response.get("full_prompt_for_ai", "")
        
        except Exception as e:
            logger.error(f"[AI_EVAL] Критическая ошибка AI для {symbol}: {e}", exc_info=True)
            return

        # --- 3. Исполнение сделки ---
        await self.execute_trade_entry(candidate, features)

    async def execute_trade_entry(self, candidate: dict, features: dict):
        """
        [Исполнитель] Принимает одобренного кандидата, рассчитывает объем,
        проверяет риски и размещает ордер.
        """
        symbol = candidate['symbol']
        side = candidate['side']

        try:
            volume_usdt = candidate.get('volume_usdt', self.POSITION_VOLUME)
            last_price = features.get("price", 0)
            if not last_price > 0:
                logger.warning(f"[EXECUTE_CANCEL] {symbol}/{side}: Невалидная цена.")
                return

            qty = await self._calc_qty_from_usd(symbol, volume_usdt, last_price)
            if not qty > 0:
                logger.warning(f"[EXECUTE_CANCEL] {symbol}/{side}: Нулевой объем.")
                return

            if not await self._risk_check(symbol, side, qty, last_price):
                return # Причина уже залогирована в _risk_check

            logger.info(f"[EXECUTE] {symbol}/{side}: Все проверки пройдены. Отправка ордера...")

            # Сохраняем кандидата (с промптом) для последующего логирования при закрытии
            async with self.position_lock:
                self.active_trade_entries[symbol] = candidate

            # Используем единый метод для размещения ордера
            await self.place_unified_order_guarded(symbol, side, qty, "Market", comment=candidate['justification'])

        except Exception as e:
            logger.error(f"[EXECUTE] Критическая ошибка для {symbol}: {e}", exc_info=True)
            self.active_trade_entries.pop(symbol, None) # Очищаем, если исполнение не удалось
            

    async def _ai_call_with_timeout(self, provider: str, candidate: dict, features: dict) -> dict:
        now = time.time()
        if now < self.ai_circuit_open_until:
            return {"action": "REJECT", "justification": "AI temporarily disabled (circuit open)", "full_prompt_for_ai": ""}

        try:
            async with self.ai_sem:
                return await asyncio.wait_for(
                    self.evaluate_candidate_with_ollama(candidate, features),
                    timeout=self.ai_timeout_sec
                )
        except (asyncio.TimeoutError, RequestsReadTimeout, RequestsConnectionError) as e:
            # открыть «рубильник» на минуту
            self.ai_circuit_open_until = time.time() + 60
            logger.error(f"[AI_TIMEOUT] {provider} завис: {e}. Отключаю ИИ на 60 сек.")
            return {"action": "REJECT", "justification": f"AI timeout: {e}", "full_prompt_for_ai": ""}
        except Exception as e:
            # тоже открываем «рубильник», но на 30 сек
            self.ai_circuit_open_until = time.time() + 30
            logger.error(f"[AI_FAIL] {provider} упал: {e}", exc_info=True)
            return {"action": "REJECT", "justification": f"AI failure: {e}", "full_prompt_for_ai": ""}

    async def evaluate_candidate_with_ollama(self, candidate: dict, features: dict) -> dict:
        """
        [V8 - Refactored] Отправляет в Ollama отчет, включая оценку от ML-модели.
        """
        from openai import AsyncOpenAI
        default_response = {"confidence_score": 0.5, "justification": "Ошибка локального AI.", "action": "REJECT"}
        prompt = ""
        try:
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            def _format(v, spec): return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"
            
            m = candidate.get('base_metrics', {})
            source = candidate.get('source', 'unknown').replace('_', ' ').title()
            ml_score = candidate.get('ml_score', 0.0)

            btc_change_1h = compute_pct(self.shared_ws.candles_data.get("BTCUSDT", deque()), 60)
            eth_change_1h = compute_pct(self.shared_ws.candles_data.get("ETHUSDT", deque()), 60)

            prompt = f"""
            SYSTEM: Ты - элитный квантовый аналитик и риск-менеджер. Твой ответ - всегда только валидный JSON.
            USER:
            Анализ торгового сигнала для принятия решения.
            - Сигнал: {candidate['symbol']}, Направление: {candidate['side'].upper()}, Источник: {source}
            - Ключевые метрики сигнала: {json.dumps(m)}
            - Оценка ML-модели: Ожидаемый PnL = {_format(ml_score, '.2f')}%
            - Контекст рынка: ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(features.get('rsi14'), '.1f')}, BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%

            ЗАДАЧА: На основе всех данных, верни JSON с ключами "confidence_score" (0.0-1.0, твоя уверенность), "justification" (краткое, но емкое обоснование), и "action" ("EXECUTE" или "REJECT").
            """
            response = await client.chat.completions.create(
                model="trading-llama",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                top_p=1,
            )
            response_data = json.loads(response.choices[0].message.content)
            response_data['full_prompt_for_ai'] = prompt # Сохраняем промпт для будущего дообучения
            return response_data
        except Exception as e:
            logger.error(f"[Ollama] Ошибка API для {candidate['symbol']}: {e}", exc_info=True)
            return {**default_response, "full_prompt_for_ai": prompt}


    async def _squeeze_logic(self, symbol: str) -> bool:
        """
        [Refactored] Находит сигнал на сквиз и передает его как кандидата в пайплайн.
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
            
            oi_hist = list(self.shared_ws.oi_history.get(symbol, []))
            oi_change_pct = 0.0
            if len(oi_hist) >= 2:
                oi_now, oi_prev = oi_hist[-1], oi_hist[-2]
                if oi_prev > 0: oi_change_pct = (oi_now - oi_prev) / oi_prev * 100.0

            side = "Sell" if pct_5m >= thr_price else "Buy"
            
            features = await self.extract_realtime_features(symbol)
            if not features: return False

            candidate = {
                'symbol': symbol, 'side': side, 'source': 'squeeze',
                'base_metrics': {
                    'price_change_5m_pct': pct_5m, 
                    'volume_change_1m_vs_5m_pct': vol_change_pct, 
                    'squeeze_power': squeeze_power,
                    'oi_change_1m_pct': oi_change_pct
                },
                'volume_usdt': self.POSITION_VOLUME
            }
            
            await self.process_signal_candidate(candidate, features)
            self.last_squeeze_ts[symbol] = time.time()
            return True

        except Exception as e:
            logger.error(f"[_squeeze_logic] Ошибка анализа сквиза для {symbol}: {e}", exc_info=True)
            return False
        
    async def _liquidation_logic(self, symbol: str) -> bool:
        """
        [Refactored] Находит сигнал по ликвидациям и передает его как кандидата в пайплайн.
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

            candidate = {
                'symbol': symbol, 'side': order_side, 'source': 'liquidation',
                'base_metrics': {
                    'liquidation_value_usdt': liq_info.get("value"),
                    'liquidation_side': liq_info.get("side")
                },
                'volume_usdt': self.POSITION_VOLUME
            }
            
            await self.process_signal_candidate(candidate, features)
            self.shared_ws.last_liq_trade_time[symbol] = dt.datetime.utcnow()
            return True

        except Exception as e:
            logger.error(f"[_liquidation_logic] Ошибка анализа ликвидаций для {symbol}: {e}", exc_info=True)
            return False

    async def _golden_logic(self, symbol: str):
        """
        [Refactored] Ищет "золотой сетап" и передает сигнал как кандидата в пайплайн.
        Использует процентные изменения цены, объема (не CVD) и открытого интереса.
        """
        if symbol in self.open_positions or symbol in self.pending_orders:
            return

        try:
            if not self.strategy_mode in ("full", "golden_only", "golden_squeeze"):
                return

            if not self._golden_allowed(symbol):
                return

            # --- Агрегация данных до 5-минутных свечей ---
            minute_candles = list(self.shared_ws.candles_data.get(symbol, []))
            vol_hist_1m = list(self.shared_ws.volume_history.get(symbol, []))
            oi_hist_1m = list(self.shared_ws.oi_history.get(symbol, []))
            
            if len(minute_candles) < 6: return

            # --- Расчет процентных изменений за 5 минут ---
            p_start = safe_to_float(minute_candles[-6]["closePrice"])
            p_end = safe_to_float(minute_candles[-1]["closePrice"])
            price_change_pct = (p_end - p_start) / p_start * 100.0 if p_start > 0 else 0.0

            vol_start = safe_to_float(vol_hist_1m[-6]) if len(vol_hist_1m) >= 6 else 0.0
            vol_end = safe_to_float(vol_hist_1m[-1]) if vol_hist_1m else 0.0
            volume_change_pct = (vol_end - vol_start) / vol_start * 100.0 if vol_start > 0 else 0.0

            oi_start = safe_to_float(oi_hist_1m[-6]) if len(oi_hist_1m) >= 6 else 0.0
            oi_end = safe_to_float(oi_hist_1m[-1]) if oi_hist_1m else 0.0
            oi_change_pct = (oi_end - oi_start) / oi_start * 100.0 if oi_start > 0 else 0.0

            # --- Получение динамических порогов ---
            buy_params = await self._get_golden_thresholds(symbol, "Buy")
            sell_params = await self._get_golden_thresholds(symbol, "Sell")

            # --- Проверка условий ---
            side = None
            if (price_change_pct >= buy_params["price_change"] and
                volume_change_pct >= buy_params["volume_change"] and
                oi_change_pct >= buy_params["oi_change"]):
                side = "Buy"
            elif (price_change_pct <= -sell_params["price_change"] and
                  volume_change_pct >= sell_params["volume_change"] and
                  oi_change_pct >= sell_params["oi_change"]):
                side = "Sell"

            if not side:
                return

            # --- Формирование кандидата и отправка в пайплайн ---
            features = await self.extract_realtime_features(symbol)
            if not features: return

            candidate = {
                "symbol": symbol,
                "side": side,
                "source": "golden_setup",
                "base_metrics": {
                    "price_change_5m_pct": price_change_pct,
                    "volume_change_5m_pct": volume_change_pct,
                    "oi_change_5m_pct": oi_change_pct,
                },
                "volume_usdt": self.POSITION_VOLUME,
            }

            await self.process_signal_candidate(candidate, features)
            self._last_golden_ts[symbol] = time.time()

        except Exception as e:
            logger.error(f"[_golden_logic] unexpected error for {symbol}: {e}", exc_info=True)
        

    async def place_unified_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        *,
        comment: str = "",
        orderLinkId: str | None = None,
        order_link_id: str | None = None,
        **kwargs,
    ):

        # Bybit Unified V5 order via pybit.HTTP (sync call in thread)
        # Idempotency by orderLinkId: treat duplicate as success
        _cli_id = (orderLinkId or order_link_id or kwargs.pop("orderLinkId", None) or kwargs.pop("order_link_id", None))
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": self._format_qty(symbol, qty),
        }

        if _cli_id:
            params["orderLinkId"] = _cli_id
        for k, v in kwargs.items():
            if v is not None and k not in params:
                params[k] = v
        try:
            logger.info("[ORDER_SEND] %s %s qty=%s type=%s linkId=%s", symbol, side, params.get("qty"), params.get("orderType"), params.get("orderLinkId"))
            if comment:
                logger.debug("[ORDER_COMMENT] %s", comment)
        except Exception:
            pass


        if not hasattr(self, "session") or not hasattr(self.session, "place_order"):
            raise RuntimeError("self.session.place_order is not available")
        _call = functools.partial(self.session.place_order, **params)
        resp = await asyncio.to_thread(_call)
        try:
            # telemetry
            try:
                self.stats["orders"] += 1
            except Exception:
                pass
            rc = int(resp.get("retCode", 0))
        except Exception:
            rc = 0
        ret_msg = str(resp.get("retMsg", ""))
        if rc == 0:
            try:
                logger.debug("[ORDER_RESP] %s retCode=0 result=%s", symbol, resp.get("result"))
            except Exception:
                pass
            return resp
        if _cli_id and (rc == 10006 or "duplicate" in ret_msg.lower()):
            logger.info("[BYBIT_IDEMPOTENT] duplicate orderLinkId=%s -> treat as success", _cli_id)
            return resp

        raise RuntimeError("Bybit place_order failed: retCode=%s, retMsg=%s, resp=%s" % (rc, ret_msg, str(resp)[:300]))



    def _make_order_link_id(self, symbol: str, side: str) -> str:
        """Generate idempotent client order ID for Bybit unified V5.
        Format: bot:<uid>:<symbol>:<Side>:<epoch_ms>:<rand4>
        """
        import time, uuid
        uid = getattr(self, "user_id", "user")
        now = int(time.time() * 1000)
        rand4 = uuid.uuid4().hex[:4]
        return f"bot:{uid}:{symbol}:{side}:{now}:{rand4}"

    async def place_order_ws(self, symbol: str, side: str, qty: float,
                             position_idx: int = 1, price: float | None = None,
                             order_type: str = "Market", orderLinkId: str | None = None) -> dict:
        """Place order via Bybit Private Trade WebSocket (real account).
        Respects exchange minQty and supports Market/Limit. Returns WS response dict.
        """
        import time, json as _json
        # sanity checks
        if not getattr(self, "ws_trade", None):
            raise RuntimeError("Trade WebSocket is not connected.")
        step    = self.qty_step_map.get(symbol, DEC_TICK)
        min_qty = self.min_qty_map.get(symbol, step)
        fqty    = float(qty)
        if fqty < float(min_qty):
            raise RuntimeError(f"Qty {fqty} < min_qty {min_qty}")
        # payload
        header = {
            "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
            "X-BAPI-RECV-WINDOW": "5000"
        }
        args = {
            "symbol": symbol,
            "side": side,
            "qty": str(self._format_qty(symbol, fqty)),
            "category": "linear",
            "timeInForce": "GTC",
            "positionIdx": int(position_idx),
            "orderType": order_type,
        }
        if orderLinkId:
            args["orderLinkId"] = orderLinkId
        if price is not None and order_type == "Limit":
            args["price"] = str(price)
        req_id = f"{symbol}-{int(time.time()*1000)}"
        req_id = f"{symbol}-{int(time.time()*1000)}"
        req = {
            "op": "order.create",
            "req_id": req_id,
            "header": header,
            "args": [args]
        }
        await self.ws_trade.send(_json.dumps(req))
        # wait for ack with same req_id
        while True:
            raw = await self.ws_trade.recv()
            try:
                resp = _json.loads(raw)
            except Exception:
                continue
            if resp.get("req_id") == req_id:
                break
        if resp.get("retCode") != 0:
            raise RuntimeError(f"WS order failed: {resp}")
        return resp

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
        Фоновая задача переобучения ML-модели (MLX или PyTorch).
        """
        logger.info(
            f"[Retrain] Background task started (framework={self.ml_framework.upper()}, wake={every_sec}s, min_samples={min_samples})"
        )

        while True:
            await asyncio.sleep(every_sec)
            try:
                buf_len = len(self.training_data)
                logger.debug(f"[Retrain] Buffered samples: {buf_len}")

                if buf_len < min_samples:
                    continue

                # --- Обучение ---
                if self.ml_framework == 'mlx':
                    model, scaler = train_golden_model_mlx(list(self.training_data))
                    model.save_weights(MODEL_PATH_MLX)
                    joblib.dump(scaler, SCALER_PATH)
                    logger.info(f"[Retrain] MLX model saved to {MODEL_PATH_MLX}, scaler to {SCALER_PATH}")
                    # Hot-swap
                    if isinstance(self.ml_inferencer, MLXInferencer):
                        self.ml_inferencer.model = model
                        self.ml_inferencer.scaler = scaler
                else: # PyTorch
                    model, scaler = train_golden_model_pytorch(list(self.training_data))
                    torch.save({"model_state": model.state_dict()}, MODEL_PATH_PYTORCH)
                    joblib.dump(scaler, SCALER_PATH)
                    logger.info(f"[Retrain] PyTorch model saved to {MODEL_PATH_PYTORCH}, scaler to {SCALER_PATH}")
                    # Hot-swap
                    if isinstance(self.ml_inferencer, PyTorchInferencer):
                        self.ml_inferencer.model = model
                        self.ml_inferencer.scaler = scaler

                # --- Очистка буфера ---
                for _ in range(buf_len):
                    self.training_data.popleft()

                logger.info(f"[Retrain] ML model retrained on {buf_len} samples.")

            except Exception:
                logger.exception("[Retrain] Unexpected error during training")


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
                "gemini_api_key": data.get("gemini_api_key"),
                "openai_api_key": data.get("openai_api_key"),
                "ai_provider": data.get("ai_provider", "ollama"),
                "ml_framework": data.get("ml_framework", "mlx"), # Добавлено поле для выбора фреймворка
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
            "price_change": 1.7,      # +1.7 % price rise
            "volume_change": 200,      # +200 % volume surge
            "oi_change": 1.5,         # +1.5 % OI rise
        },
        "Sell": {
            "period_iters": 4,
            "price_change": 1.8,      # −1.8 % price drop
            "volume_change": 200,      # +200 % volume surge
            "oi_change": 1.2,         # +1.2 % OI rise
        }
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

    shared_ws.bot = bots[0] if bots else None # «главный» бот

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

async def ai_confirm(self, symbol: str, side: str, features: dict, aux: dict | None = None):
    """
    Обёртка над evaluate_candidate_with_ollama, приводит ответ к (approve, confidence, reason).
    Не блокирует вход при сбое локального AI.
    """
    base_metrics = features.get("base_metrics") or {
        "pct_5m": features.get("pct_5m"),
        "vol_change_pct": features.get("vol_change_pct"),
        "oi_change_pct": features.get("oi_change_pct"),
        "cvd_change_pct": features.get("cvd_change_pct"),
    }
    candidate = {
        "symbol": symbol,
        "side": side,
        "source": features.get("source", "golden_setup"),
        "base_metrics": base_metrics,
    }
    try:
        r = await self.evaluate_candidate_with_ollama(candidate, features)
        if isinstance(r, dict):
            approve    = (str(r.get("action", "")).upper() == "EXECUTE")
            confidence = float(r.get("confidence_score", 0.5) or 0.5)
            reason     = str(r.get("justification", ""))[:300]
            return approve, confidence, reason
        elif isinstance(r, tuple):
            if len(r) == 3:
                return bool(r[0]), float(r[1] or 0.0), str(r[2] or "")
            elif len(r) == 2:
                return bool(r[0]), float(r[1] or 0.0), ""
            else:
                return bool(r[0]), 0.6, ""
        else:
            return bool(r), 0.6, "non_dict_response"
    except Exception as e:
        logger.exception("[AI_ERROR] %s/%s ai_confirm failed: %s", symbol, side, e)
        return True, 0.7, "ai_unavailable_bypass"

# ----------------- КОНЕЦ ФАЙЛА MultiuserBot_READY6_REFACTORED.py -----------------