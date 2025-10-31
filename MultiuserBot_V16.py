import datetime as dt
import aiogram
from aiogram.enums import ParseMode

# Ensure required imports
import asyncio
import pytz
from datetime import timedelta
from pathlib import Path

# ADD: Import InvalidRequestError for advanced order error handling
from pybit.exceptions import InvalidRequestError

# ---------------------- IMPORTS ----------------------
import os
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
from concurrent.futures import ThreadPoolExecutor
from websockets.exceptions import ConnectionClosed

import uvloop
uvloop.install()

import uuid

 # Telegram‑ID(-ы) администраторов, которым доступна команда /snapshot
ADMIN_IDS = {36972091}   # ← замените на свой реальный ID

# Глобальный реестр всех экземпляров TradingBot (используется для snapshot)
GLOBAL_BOTS: list = []

# Конфигурация для Apple Silicon M4
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

logger = logging.getLogger(__name__)
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

SQUEEZE_THRESHOLD_PCT = 5.5    # рост ≥ 3 % за 5 мин

# Минимум мощности сквиза (произведение % изменения цены на % изменения объёма)
DEFAULT_SQUEEZE_POWER_MIN = 20.0

AVERAGE_LOSS_TRIGGER = -160.0   # усредняем, если unrealised PnL ≤ −160%

GLOBAL_BOTS: list["TradingBot"] = []
tg_fsm.GLOBAL_BOTS = GLOBAL_BOTS

# --- dynamic-threshold & volatility coefficients (v3) ---
LARGE_TURNOVER = 100_000_000     # 100 M USDT 24h turnover
MID_TURNOVER   = 10_000_000      # 10 M USDT
VOL_COEF       = 1.2             # ≥ 1.2σ spike
VOL_WINDOW     = 60              # 12 × 5-мин свечей = 1 час
VOLUME_COEF    = 3.0             # объём ≥ 3× ср.30 мин

LISTING_AGE_MIN_MINUTES = 1400    # игнорируем пары младше 12 часов

# ── shared JSON paths for Telegram-FSM ───────────────────────────────────────
OPEN_POS_JSON   = "open_positions.json"
WALLET_JSON     = "wallet_state.json"
TRADES_JSON     = "trades_history.json"

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

# ---------------------- INDICATOR FUNCTIONS ----------------------
def compute_supertrend(df, period=10, multiplier=3):
    device = torch.device('mps')  # Использование Metal
    high = torch.tensor(df["highPrice"].values, device=device)
    low = torch.tensor(df["lowPrice"].values, device=device)
    close = torch.tensor(df["closePrice"].values, device=device)

    hl2 = ((high + low) / 2).to(device)
    atr = ta.atr(high, low, close, length=period)
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend = pd.Series(index=df.index, dtype=bool)

    in_uptrend = True
    for current in range(1, len(df)):
        if df["closePrice"].iloc[current] > upperband.iloc[current - 1]:
            in_uptrend = True
        elif df["closePrice"].iloc[current] < lowerband.iloc[current - 1]:
            in_uptrend = False
        supertrend.iloc[current] = in_uptrend
    return supertrend


# ---------------------- WEBSOCKET: PUBLIC ----------------------
class PublicWebSocketManager:
    __slots__ = (
        "symbols", "interval", "ws",
        "candles_data", "ticker_data", "latest_open_interest",
        "active_symbols", "_last_selection_ts", "_callback",
        "ready_event", "bot",
        "loop", "volume_history", "oi_history", "cvd_history",
        "_last_saved_time", "position_handlers", "_history_file",
        "_save_task", "latest_liquidation",
        "_liq_thresholds", "last_liq_trade_time", "funding_history",
    )
    def __init__(self, symbols, interval="1", bot=None):
        self.symbols = symbols
        self.interval = interval
        self.ws = None
        self.bot = bot
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
        self.bot = bot
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

            filtered_set = set()
            for symbol in new_set:
                age_min = await self.bot.listing_age_minutes(symbol)
                if age_min >= 60:
                    filtered_set.add(symbol)
                else:
                    logger.debug("[symbol_selection] %s skipped (listing age %d min)", symbol, age_min)

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
            # Передаём событие напрямую каждому TradingBot
            if self.position_handlers:
                await asyncio.gather(
                    *(bot.handle_liquidation(msg) for bot in self.position_handlers),
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
        deque_c = self.ws.candles_data.get(symbol, [])
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

# ---------------------- TRADING BOT ----------------------
class TradingBot:
    # ---------- symbol meta helpers ----------
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
        "session", "shared_ws", "ws_private", "symbols",
        "open_positions", "last_position_state", "golden_param_store",
        "market_task", "sync_task", "ws_trade", "loop", "position_idx", "model",
        "POSITION_VOLUME", "MAX_TOTAL_VOLUME", "qty_step_map", "min_qty_map",
        "failed_orders", "pending_orders", "pending_strategy_comments", "last_trailing_stop_set",
        "position_lock", "closed_positions", "pnl_task", "last_seq",
        "ws_opened_symbols", "ws_closed_symbols", "averaged_symbols", "limiter",
        "turnover24h", "selected_symbols", "last_asset_selection_time",
        "wallet_task", "last_stop_price", "_recv_lock", "max_allowed_volume",
        "strategy_mode", "liq_buffers",
        "trailing_start_map", "trailing_gap_map",
        "trailing_start_pct", "trailing_gap_pct",
        "pending_timestamps", "squeeze_threshold_pct", "squeeze_power_min", "averaging_enabled",
        "warmup_done", "warmup_seconds", "_last_snapshot_ts", "reserve_orders", 
        "coreml_model", "feature_scaler", "last_retrain", "training_data", "device", "FEATURE_KEYS", 
        "last_squeeze_ts", "squeeze_cooldown_sec", "active_trade_entries", "listing_age_min", "_age_cache",
        "symbol_info", "trade_history_file", "active_trades", "symbol_info", "pending_signals", "max_signal_age"
    )


    def __init__(self, user_data, shared_ws, golden_param_store):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "real")
        self.listing_age_min = int(user_data.get("listing_age_min_minutes", LISTING_AGE_MIN_MINUTES))

        self.session = HTTP(demo=(self.mode == "demo"),
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            timeout=30,
                            max_retries=3,
                            recv_window=1000)
        self.shared_ws = shared_ws
        # Регистрируемся на обновления тикера для trailing-stop (если shared_ws передан)

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
        self.load_model()
        self.POSITION_VOLUME = safe_to_float(user_data.get("volume", 1000))
        # максимальный разрешённый общий объём открытых позиций (USDT)
        self.MAX_TOTAL_VOLUME = safe_to_float(user_data.get("max_total_volume", 5000))
        # Maximum allowed total exposure across all open positions (in USDT)
        self.qty_step_map: dict[str, Decimal] = {}
        self.min_qty_map: dict[str, Decimal] = {}
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
        raw_mode = user_data.get("strategy_mode") or user_data.get("strategy") or "full"
        raw_mode = str(raw_mode).lower()

        alias_map = {
            "golden": "golden_only",
            "golden_only": "golden_only",
            "liq": "liquidation_only",
            "liquidation": "liquidation_only",
            "liquidation_only": "liquidation_only",
            "full": "full",
            "all": "full",
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

        # ── warm‑up (startup grace period) ───────────────────────────────
        self.warmup_done     = False
        self.warmup_seconds  = int(user_data.get("warmup_seconds", 480))  # default 8 min

        self.averaging_enabled: bool = True   # averaging-mode toggle

        self._last_snapshot_ts: dict[str, float] = {}

        self.reserve_orders: dict[str, dict] = {}   # symbol → {...}
        
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
        self.FEATURE_KEYS = [
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
            "SQ_spread_pct", "SQ_sigma5m", "SQ_liq10s", "SQ_cooldown",
            "LIQ_cluster_val10s", "LIQ_cluster_count10s", "LIQ_direction",
            "LIQ_pct1m", "LIQ_pct5m", "LIQ_vol1m", "LIQ_vol5m", "LIQ_dOI1m",
            "LIQ_spread_pct", "LIQ_sigma5m", "LIQ_golden_flag", "LIQ_squeeze_flag",
            "LIQ_cooldown",
            "hour_of_day", "day_of_week", "month_of_year", "adx14",
        ]
        
        self.symbol_info: dict[str, dict] = {}

        self.pending_signals: dict[str, float] = {}   # symbol → timestamp
        self.max_signal_age = 30.0                    # сек.
        # Поставьте задачу-очиститель:
        asyncio.create_task(self._drop_stale_signals())

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


    async def listing_age_minutes(self, symbol: str) -> int:
        """
        Возраст деривативного контракта в минутах.

        - Сначала смотрим `first_seen` в кеше;
        - затем пробуем REST /v5/market/instruments-info
          **во всех категориях** и берём минимальный launchTime;
        - если биржа отдаёт 0 или None – делаем длинный
          REST бэкфилл 720h, время первой свечи считаем launchTime;
        - на крайний случай – длина локальной истории.
        """
        if not hasattr(self, "symbol_info"):
            self.symbol_info = {}

        now = int(time.time() * 1000)

        # 1) кеш
        info = self.symbol_info.get(symbol, {})
        if "first_seen" in info:
            return int((now - info["first_seen"]) / 60_000)

        # 2) REST: ищем самый "старый" launchTime
        min_ts = None
        for cat in ("linear", "inverse"):
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_instruments_info(
                        category=cat, symbol=symbol
                    )
                )
                data = resp["result"]["list"]
                if data:
                    ts = int(data[0].get("launchTime") or 0)
                    if ts > 0:
                        min_ts = ts if min_ts is None else min(min_ts, ts)
            except Exception:
                logger.exception("[listing_age] REST failed for %s/%s", symbol, cat)

        # 3) Если REST помог -> кешируем и возвращаем
        if min_ts:
            info["first_seen"] = min_ts
            self.symbol_info[symbol] = info
            return int((now - min_ts) / 60_000)

        # 4) REST не дал launchTime – тянем длинный бэкфилл 720h
        try:
            klines = await asyncio.to_thread(
                lambda: self.session.get_kline(
                    category="linear", symbol=symbol, interval=60,
                    limit=720, from_time=(now // 1000) - 720 * 60 * 60
                )
            )
            if klines["result"]["list"]:
                ts = int(klines["result"]["list"][-1][0])  # самый старый бар
                info["first_seen"] = ts
                self.symbol_info[symbol] = info
                return int((now - ts) / 60_000)
        except Exception:
            logger.exception("[listing_age] long backfill failed for %s", symbol)

        # 5) Фоллбэк: длина локальной истории
        hist = self.history.get(symbol)
        if hist is not None and not hist.empty:
            ts = int(hist.index.min().timestamp() * 1000)
            return int((now - ts) / 60_000)

        # Без понятия – считаем «0 минут» (пусть отфильтруется как новый)
        return 0

    def load_ml_models(self):
        """Загрузка CoreML и PyTorch моделей"""
        # Попытка загрузки CoreML модели
        coreml_path = "TradingModel.mlmodel"
        if os.path.exists(coreml_path):
            try:
                self.coreml_model = ct.models.MLModel(coreml_path)
                logger.info("[ML] CoreML модель успешно загружена")
            except Exception as e:
                logger.error(f"[ML] Ошибка загрузки CoreML модели: {e}")
        
        # Попытка загрузки PyTorch модели (для дообучения)
        torch_path = "trading_model.pth"
        if os.path.exists(torch_path):
            try:
                # Инициализация пустой модели
                self.model = TradingEnsemble(
                    input_size=50,
                    tech_size=20,
                    fund_size=10
                ).to(self.device)
                
                # Загрузка весов
                self.model.load_state_dict(torch.load(torch_path, map_location=self.device))
                self.model.eval()
                logger.info("[ML] PyTorch модель успешно загружена")
            except Exception as e:
                logger.error(f"[ML] Ошибка загрузки PyTorch модели: {e}")
        else:
            # Инициализация новой модели, если файл не найден
            self.model = TradingEnsemble(
                input_size=50,
                tech_size=20,
                fund_size=10
            ).to(self.device)
            logger.info("[ML] Инициализирована новая PyTorch модель")

    async def extract_realtime_features(self, symbol: str) -> Dict[str, float]:
        # 1. Базовые
        last_price = safe_to_float(self.shared_ws.ticker_data[symbol]["lastPrice"])
        bid1 = safe_to_float(self.shared_ws.ticker_data[symbol]["bid1Price"])
        ask1 = safe_to_float(self.shared_ws.ticker_data[symbol]["ask1Price"])
        spread_pct = (ask1 - bid1) / bid1 * 100 if bid1 > 0 else 0.0
        # ... собрать последние свечи 1m,5m,15m из self.shared_ws.candles_data[symbol]
        pct1m = compute_pct(self.shared_ws.candles_data[symbol], minutes=1)
        pct5m = compute_pct(self.shared_ws.candles_data[symbol], minutes=5)
        pct15m = compute_pct(self.shared_ws.candles_data[symbol], minutes=15)

        V1m = sum_last_vol(self.shared_ws.candles_data[symbol], minutes=1)
        V5m = sum_last_vol(self.shared_ws.candles_data[symbol], minutes=5)
        V15m = sum_last_vol(self.shared_ws.candles_data[symbol], minutes=15)

        OI_now = safe_to_float(self.shared_ws.latest_open_interest.get(symbol, 0))
        OI_prev1m = self.shared_ws.oi_history[symbol][-2] if len(self.shared_ws.oi_history[symbol]) >= 2 else 0.0
        OI_prev5m = self.shared_ws.oi_history[symbol][-6] if len(self.shared_ws.oi_history[symbol]) >= 6 else 0.0
        dOI1m = (OI_now - OI_prev1m)/OI_prev1m if OI_prev1m>0 else 0.0
        dOI5m = (OI_now - OI_prev5m)/OI_prev5m if OI_prev5m>0 else 0.0

        CVD_now = self.shared_ws.cvd_history[symbol][-1] if self.shared_ws.cvd_history[symbol] else 0.0
        CVD_prev1m = self.shared_ws.cvd_history[symbol][-2] if len(self.shared_ws.cvd_history[symbol]) >= 2 else 0.0
        CVD_prev5m = self.shared_ws.cvd_history[symbol][-6] if len(self.shared_ws.cvd_history[symbol]) >= 6 else 0.0
        CVD1m = CVD_now - CVD_prev1m
        CVD5m = CVD_now - CVD_prev5m

        sigma5m = self.shared_ws._sigma_5m(symbol)

        # 2. Технические
        df = pd.DataFrame(self.shared_ws.candles_data[symbol][-50:])
        rsi14 = ta.rsi(df["closePrice"], length=14).iloc[-1] if len(df) >= 14 else 50.0
        sma50 = ta.sma(df["closePrice"], length=50).iloc[-1] if len(df) >= 50 else (df["closePrice"].iloc[-1] if len(df) > 0 else 0.0)
        ema20 = ta.ema(df["closePrice"], length=20).iloc[-1] if len(df) >= 20 else (df["closePrice"].iloc[-1] if len(df) > 0 else 0.0)
        atr14 = ta.atr(df["highPrice"], df["lowPrice"], df["closePrice"], length=14).iloc[-1] if len(df) >= 14 else 0.0
        # Compute Bollinger Bands only once for efficiency
        if len(df) >= 20:
            bb = ta.bbands(df["closePrice"], length=20)
            bb_width = bb["BBU_20_2.0"].iloc[-1] - bb["BBL_20_2.0"].iloc[-1]
        else:
            bb_width = 0.0
        # Supertrend as ±1/0
        if len(df) > 0:
            supertrend_val = compute_supertrend(df, period=10, multiplier=3).iloc[-1]
            supertrend_num = 1 if supertrend_val else -1
        else:
            supertrend_num = 0
        # ADX для режима рынка
        adx_series = ta.adx(df["highPrice"], df["lowPrice"], df["closePrice"], length=14)
        adx14 = adx_series["ADX_14"].iloc[-1] if not adx_series["ADX_14"].isna().all() else 0.0
        cci20 = ta.cci(df["highPrice"], df["lowPrice"], df["closePrice"], length=20).iloc[-1] if len(df) >= 20 else 0.0
        macd = ta.macd(df["closePrice"], fast=12, slow=26, signal=9)
        macd_val = macd["MACD_12_26_9"].iloc[-1] if not macd["MACD_12_26_9"].isna().all() else 0.0
        macd_signal = macd["MACDs_12_26_9"].iloc[-1] if not macd["MACDs_12_26_9"].isna().all() else 0.0
        avgVol30m = self.shared_ws.get_avg_volume(symbol, minutes=30)
        avgOI30m = (sum(self.shared_ws.oi_history[symbol][-30:]) / max(1, len(self.shared_ws.oi_history[symbol][-30:])))
        deltaCVD30m = CVD_now - (self.shared_ws.cvd_history[symbol][-31] if len(self.shared_ws.cvd_history[symbol])>=31 else 0.0)

        # 3. Golden Setup фичи
        GS_pct4m = compute_pct(self.shared_ws.candles_data[symbol], minutes=4)
        GS_vol4m = sum_last_vol(self.shared_ws.candles_data[symbol], minutes=4)
        GS_dOI4m = (OI_now - (self.shared_ws.oi_history[symbol][-5] if len(self.shared_ws.oi_history[symbol])>=5 else OI_now)) / \
                max(1, (self.shared_ws.oi_history[symbol][-5] if len(self.shared_ws.oi_history[symbol])>=5 else 1))
        GS_cvd4m = CVD_now - (self.shared_ws.cvd_history[symbol][-5] if len(self.shared_ws.cvd_history[symbol])>=5 else CVD_now)
        GS_supertrend_flag = supertrend_num
        GS_cooldown_flag = 1 if not self._golden_allowed(symbol) else 0  # своя функция, проверяющая cooldown

        # 4. Squeeze фичи
        SQ_pct1m = pct1m
        SQ_pct5m = pct5m
        SQ_vol1m = V1m
        SQ_vol5m = V5m
        SQ_dOI1m = dOI1m
        SQ_spread_pct = spread_pct
        SQ_sigma5m = sigma5m
        # кластер ликвидаций за 10 сек
        recent_liq_vals = [
            v for (ts, s, v) in self.liq_buffers[symbol]
            if time.time() - ts <= 10
        ]
        SQ_liq10s = sum(recent_liq_vals)
        SQ_cooldown_flag = 1 if not self._squeeze_allowed(symbol) else 0  # проверка cooldown

        # 5. Liquidation фичи
        # аналогично, собрать последние события за 10 сек:
        buf = self.liq_buffers[symbol]
        recent_all = [(ts, s, v) for (ts, s, v) in buf if time.time() - ts <= 10]
        same_side = [v for (ts, s, v) in recent_all if s == recent_all[-1][1]] if recent_all else []
        LIQ_cluster_val10s = sum(same_side)
        LIQ_cluster_count10s = len(same_side)
        LIQ_direction = 1 if (recent_all and recent_all[-1][1] == "Buy") else -1
        LIQ_pct1m = pct1m
        LIQ_pct5m = pct5m
        LIQ_vol1m = V1m
        LIQ_vol5m = V5m
        LIQ_dOI1m = dOI1m
        LIQ_spread_pct = spread_pct
        LIQ_sigma5m = sigma5m
        LIQ_golden_flag = 1 if not self._golden_allowed(symbol) else 0
        LIQ_squeeze_flag = 1 if not self._squeeze_allowed(symbol) else 0
        LIQ_cooldown_flag = 1 if not self.check_liq_cooldown(symbol) else 0

        # 6. Временные фичи
        now = dt.datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()
        month_of_year = now.month

        # Собираем всё в один словарь
        features = {
            # базовые (14)
            "price": last_price,
            "pct1m": pct1m, "pct5m": pct5m, "pct15m": pct15m,
            "vol1m": V1m, "vol5m": V5m, "vol15m": V15m,
            "OI_now": OI_now, "dOI1m": dOI1m, "dOI5m": dOI5m,
            "spread_pct": spread_pct, "sigma5m": sigma5m,
            "CVD1m": CVD1m, "CVD5m": CVD5m,

            # технические (11)
            "rsi14": rsi14, "sma50": sma50, "ema20": ema20,
            "atr14": atr14, "bb_width": bb_width,
            "supertrend": supertrend_num, "cci20": cci20,
            "macd": macd_val, "macd_signal": macd_signal,
            "avgVol30m": avgVol30m, "avgOI30m": avgOI30m, "deltaCVD30m": deltaCVD30m,
            "adx14": adx14,

            # Golden (6)
            "GS_pct4m": GS_pct4m, "GS_vol4m": GS_vol4m, "GS_dOI4m": GS_dOI4m,
            "GS_cvd4m": GS_cvd4m, "GS_supertrend": GS_supertrend_flag,
            "GS_cooldown": GS_cooldown_flag,

            # Squeeze (9)
            "SQ_pct1m": SQ_pct1m, "SQ_pct5m": SQ_pct5m,
            "SQ_vol1m": SQ_vol1m, "SQ_vol5m": SQ_vol5m, "SQ_dOI1m": SQ_dOI1m,
            "SQ_spread_pct": SQ_spread_pct, "SQ_sigma5m": SQ_sigma5m,
            "SQ_liq10s": SQ_liq10s, "SQ_cooldown": SQ_cooldown_flag,

            # Liquidation (13)
            "LIQ_cluster_val10s": LIQ_cluster_val10s,
            "LIQ_cluster_count10s": LIQ_cluster_count10s,
            "LIQ_direction": LIQ_direction,
            "LIQ_pct1m": LIQ_pct1m, "LIQ_pct5m": LIQ_pct5m,
            "LIQ_vol1m": LIQ_vol1m, "LIQ_vol5m": LIQ_vol5m,
            "LIQ_dOI1m": LIQ_dOI1m, "LIQ_spread_pct": LIQ_spread_pct,
            "LIQ_sigma5m": LIQ_sigma5m,
            "LIQ_golden_flag": LIQ_golden_flag, "LIQ_squeeze_flag": LIQ_squeeze_flag,
            "LIQ_cooldown": LIQ_cooldown_flag,

            # Временные (3)
            "hour_of_day": hour_of_day, "day_of_week": day_of_week, "month_of_year": month_of_year,
        }

        return features

    async def predict_action(self, symbol: str) -> str:
        try:
            # Extract flat feature dict
            features = await self.extract_realtime_features(symbol)
            
            # Фильтр для Golden Setup: только при ADX ≥25 и RSI14 ≤80
            if self.strategy_mode == "golden_only":
                if features.get("adx14", 0.0) < 25.0 or features.get("rsi14", 0.0) > 80.0:
                    return "HOLD"
            # Build a feature vector in the predefined order
            vector = [features[k] for k in self.FEATURE_KEYS]
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

    async def log_trade_for_ml(self, symbol: str, entry_data: dict, exit_data: dict):
        try:
            features = await self.extract_realtime_features(symbol)
            vector = [features[k] for k in self.FEATURE_KEYS]
            pnl = ((exit_data['price'] - entry_data['price']) / entry_data['price']) * 100.0 \
                if entry_data['side'] == "Buy" else \
                ((entry_data['price'] - exit_data['price']) / entry_data['price']) * 100.0

            record = {
                'features': vector,
                'label': 0 if pnl < 0 else 1 if pnl > 1 else 2  # 0=loss, 1=profit, 2=neutral
            }

            self.training_data.append(record)

            if time.time() - self.last_retrain > 3600 and len(self.training_data) > 100:
                asyncio.create_task(self.retrain_models())
        except Exception as e:
            logger.error(f"[ML] Trade logging error: {e}")

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
            example_input = torch.rand(1, len(self.FEATURE_KEYS)).to(self.device)
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
        """
        Load LightGBM model from model.txt if it exists.
        Sets self.model to None if the file is missing or loading fails.
        """
        model_path = "model.txt"
        if os.path.exists(model_path):
            try:
                self.model = lgb.Booster(model_file=model_path)
                #coreml_model = ct.converters.lightgbm.convert(self.model)
                # coreml_model = ct.convert(self.model, source='lightgbm')
                # coreml_model.save('model.mlmodel')
                # print(f"[User {self.user_id}] ✅ ML-модель загружена из {model_path}")
                # CoreML conversion for LightGBM is unsupported in this coremltools version
                self.coreml_model = None
                print(f"[User {self.user_id}] ✅ LightGBM модель загружена из {model_path}")
            except Exception as e:
                self.model = None
                print(f"[User {self.user_id}] ❌ Ошибка загрузки модели: {e}")
        else:
            self.model = None
            print(f"[User {self.user_id}] ⚠️ model.txt не найден — ML сигналы отключены")


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
                "golden": "golden_only",
                "golden_only": "golden_only",
                "liq": "liquidation_only",
                "liquidation_only": "liquidation_only",
                "full": "full",
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

    # ------------------------------------------------------------------
    # Можно ли сейчас открыть ещё одну сделку по сквизу?
    # ------------------------------------------------------------------
    def _squeeze_allowed(self, symbol: str) -> bool:
        """
        True, если по symbol нет открытой/ожидающей позиции
        и прошёл пауза squeeze_cooldown_sec.
        """
        if symbol in self.open_positions or symbol in self.pending_orders:
            return False
        last = self.last_squeeze_ts.get(symbol, 0.0)
        return (time.time() - last) >= self.squeeze_cooldown_sec


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
        await self.setup_private_ws()
        await self.init_trade_ws()
        # ждём, пока shared_ws соберёт данные и выберет пары
        if self.shared_ws and hasattr(self.shared_ws, "ready_event"):
            await self.shared_ws.ready_event.wait()
        self.pnl_task    = asyncio.create_task(self.pnl_loop())

        await asyncio.sleep(self.warmup_seconds)
        self.warmup_done = True
        logger.info("[warmup] user %s finished (%d s)", self.user_id, self.warmup_seconds)

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
                # ‼️ новая подписка
                self.ws_private.execution_stream(callback=self._on_execution)
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
                continue

            # ─── 3. закрываем (частично/полностью) противоположной стороной ──
            close_qty = min(pos["qty"], qty)
            realised  = self._calc_pnl(
                entry_side=pos["side"],
                entry_price=pos["avg_price"],
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
    # async def handle_position_update(self, msg):
    #     """
    #     Обработка обновлений позиций из приватного WS:
    #     – логируем «сырое» сообщение,
    #     – фильтруем события с size>0 (открытие) или уже существующие (закрытие),
    #     – пропускаем дубликаты по seq,
    #     – при открытии сохраняем в словарь, планируем оценку, логируем + пушим нотификацию,
    #     – при закрытии логируем + пушим нотификацию,
    #     – при изменении объёма — обновляем только volume.
    #     """
    #     #logger.info(f"[PositionStream] Received position update via WebSocket: {msg}")
    #     # 1) Нормализуем data
    #     data = msg.get("data", [])
    #     if isinstance(data, dict):
    #         data = [data]

    #     # 2) Оставляем только новые size>0 (открытие) и уже открытые (для закрытия)
    #     data = [
    #         p for p in data
    #         if safe_to_float(p.get("size", 0)) > 0 or p.get("symbol") in self.open_positions
    #     ]
    #     if not data:
    #         return

    #     try:
    #         for position in data:
    #             symbol = position["symbol"]

    #             # 3) Пропускаем дубликаты/старые по seq
    #             seq = position.get("seq", 0)
    #             if seq <= self.last_seq.get(symbol, 0):
    #                 continue
    #             self.last_seq[symbol] = seq

    #             # 4) Распаковываем ключевые поля
    #             side_raw   = position.get("side", "")  # 'Buy' или 'Sell', иначе ''
    #             # avgPrice иногда пустой -> fallback на entryPrice
    #             avg_price  = safe_to_float(position.get("avgPrice")) \
    #                         or safe_to_float(position.get("entryPrice"))
    #             new_size   = safe_to_float(position.get("size", 0))
    #             open_int   = self.shared_ws.latest_open_interest.get(symbol, Decimal(0))
    #             prev       = self.open_positions.get(symbol)

    #             #logger.info(
    #             #    f"[PositionStream] {symbol} update: side={side_raw or 'N/A'}, "
    #             #    f"avg_price={avg_price}, size={new_size}"
    #             #)

    #             # 5) Открытие позиции
    #             if prev is None and new_size > 0 and side_raw:
    #                 # Сохраняем в открытые
    #                 self.open_positions[symbol] = {
    #                     "avg_price": avg_price,
    #                     "side":      side_raw,
    #                     "pos_idx":   position.get("positionIdx", 1),
    #                     "volume":    new_size,
    #                     "amount":    safe_to_float(position.get("positionValue"))
    #                 }
    #                 # Mark WS-opened
    #                 self.ws_opened_symbols.add(symbol)
    #                 self.ws_closed_symbols.discard(symbol)
    #                 logger.info(f"[PositionStream] Scheduling evaluate_position for {symbol}")
    #                 asyncio.create_task(self.evaluate_position(position))

    #                 # Логируем открытие (в фоне)
    #                 asyncio.create_task(self.log_trade(
    #                     symbol,
    #                     side=side_raw,
    #                     avg_price=avg_price,
    #                     volume=new_size,
    #                     open_interest=open_int,
    #                     action="open",
    #                     result="opened"
    #                 ))

    #                 # Уведомляем пользователя (в фоне)
    #                 asyncio.create_task(self.notify_user(
    #                     f"🟢 Открыта {side_raw.upper()}-позиция {symbol}: объём {new_size} @ {avg_price}"
    #                 ))
    #                 # После подтверждения открытия освобождаем слот в pending_orders
    #                 self.pending_orders.discard(symbol)

    #             # 6) Закрытие позиции
    #             if prev is not None and new_size == 0:
    #                 logger.info(f"[PositionStream] Закрытие позиции {symbol}, PnL={position.get('unrealisedPnl')}")
    #                 # Копируем данные в closed_positions
    #                 self.closed_positions[symbol] = {
    #                     **prev,
    #                     "closed_pnl":  position.get("unrealisedPnl"),
    #                     "closed_time": position.get("updatedTime")
    #                 }
    #                 # # Удаляем из открытых
    #                 # del self.open_positions[symbol]
    #                 # Mark WS-closed
    #                 self.ws_closed_symbols.add(symbol)
    #                 self.ws_opened_symbols.discard(symbol)

    #                 # Логируем закрытие (в фоне)
    #                 asyncio.create_task(self.log_trade(
    #                     symbol,
    #                     side=prev["side"],
    #                     avg_price=prev["avg_price"],
    #                     volume=prev["volume"],
    #                     open_interest=open_int,
    #                     action="close",
    #                     result="closed",
    #                     closed_manually=False
    #                 ))
    #                 # Уведомляем пользователя (в фоне)
    #                 asyncio.create_task(self.notify_user(
    #                     f"⏹️ Закрыта {prev['side'].upper()}-позиция {symbol}: "
    #                     f"объём {prev['volume']} @ {prev['avg_price']}"
    #                 ))

    #                 # Удаляем позицию из активных, чтобы золотой сетап мог запуститься снова
    #                 del self.open_positions[symbol]
    #                 # reset averaging flag so a new position can be averaged again
    #                 self.averaged_symbols.discard(symbol)
    #                 # На всякий случай сбрасываем флаг pending_orders
    #                 self.pending_orders.discard(symbol)
    #                 continue

    #             # 7) Обновление объёма существующей позиции
    #             if prev is not None and new_size > 0 and new_size != prev.get("volume"):
    #                 logger.info(f"[PositionStream] Обновление объёма {symbol}: {prev['volume']} → {new_size}")
    #                 self.open_positions[symbol]["volume"] = new_size
    #                 # Запускаем мгновенную переоценку позиции по новому объёму
    #                 asyncio.create_task(self.evaluate_position({
    #                     "symbol": symbol,
    #                     "size":   str(new_size),
    #                     "side":   prev["side"]
    #                 }))

    #                 continue

    #     except Exception as e:
    #         logger.error(f"[handle_position_update] Ошибка обработки: {e}", exc_info=True)
    #         # Сбрасываем состояние для символа
    #         if symbol in self.open_positions:
    #             del self.open_positions[symbol]
    #         await self.update_open_positions()

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

                # 5) Открытие позиции
                if prev is None and new_size > 0 and side_raw:
                    # Сохраняем в open_positions
                    self.open_positions[symbol] = {
                        "avg_price": avg_price,
                        "side":      side_raw,
                        "pos_idx":   position.get("positionIdx", 1),
                        "volume":    new_size,
                        "amount":    safe_to_float(position.get("positionValue")),
                        "stop_loss": safe_to_float(position.get("stopLoss", 0))   # <---- добавлено

                    }
                    # Запомним entry_data для ML
                    entry_data = {
                        "price": avg_price,
                        "side": side_raw,
                        "volume": new_size,
                        "symbol": symbol,
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
                        symbol,
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
                if prev and new_size == 0 and status == "closed":
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
                        symbol,
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
        Обработка событий ликвидации. Пропускает обработку, если warm-up не завершён.
        """
        # ignore liquidation events until warm‑up completes
        if not getattr(self, "warmup_done", False):
            return

        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]

        for evt in data:
            # ----- новые поля --------------------------
            symbol = evt.get("s")                    # тикер
            qty    = safe_to_float(evt.get("v", 0))  # размер сделки
            side   = evt.get("S")                    # 'Buy' | 'Sell'
            price  = safe_to_float(evt.get("p", 0))  # bankruptcy‑price
            value_usdt = qty * price                 # объём ликвидации в USDT
            # -------------------------------------------

            self.shared_ws.latest_open_interest.setdefault(symbol, Decimal(0))
            # сохраняем сразу и объём, и сторону ликвидации
            self.shared_ws.latest_liquidation[symbol] = {
                "value": value_usdt,
                "side":  side,
                "ts":    time.time()       # epoch-время
            }

            # (если нужно хранить историю – можно убрать append в oi_history,
            #  чтобы не путать OI и ликвидации)
            logger.info("[liquidation] %s %s value=%.2f USDT  (qty=%s × price=%s)",
                        symbol, side, value_usdt, qty, price)

            # --- CSV (добавили side и price) -----------
            csv_filename = "liquidations.csv"
            file_exists  = os.path.isfile(csv_filename)
            with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "symbol", "side", "price", "quantity", "value_usdt"])
                writer.writerow([datetime.utcnow().isoformat(), symbol, side, price, qty, value_usdt])

                # --- trading trigger: cluster of same‑side liquidations ---------------
                if self.strategy_mode in ("liquidation_only", "full"):
                    now = dt.datetime.utcnow()
                    buf = self.liq_buffers[symbol]
                    buf.append((now, side, value_usdt))
                    # keep only last 150 s
                    cutoff = now - timedelta(seconds=150)
                    while buf and buf[0][0] < cutoff:
                        buf.popleft()

                    same_side_vals = [v for t, s, v in buf if s == side]
                    # adaptive multiplier: less stringent for low‑cap symbols
                    thr         = self.shared_ws.get_liq_threshold(symbol)
                    turnover24h = safe_to_float(
                        self.shared_ws.ticker_data.get(symbol, {}).get("turnover24h", 0)
                    )
                    needed_val  = (2 * thr) if turnover24h < 20_000_000 else (3 * thr)
                    if (len(same_side_vals) >= 8
                        and sum(same_side_vals) >= needed_val
                        and self.shared_ws.check_liq_cooldown(symbol)
                        and symbol not in self.open_positions
                        and symbol not in self.pending_orders):

                        order_side = "Buy" if side == "Sell" else "Sell"
                        async with self.position_lock:
                            await self.ensure_symbol_meta(symbol)
                            last_price = safe_to_float(
                                self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
                            )
                            if last_price <= 0:
                                continue  # нет цены – не рискуем

                            usd_qty = min(self.POSITION_VOLUME, self.max_allowed_volume)
                            step = self.qty_step_map.get(symbol, 0.001)
                            min_qty = self.min_qty_map.get(symbol, 0.001)
                            qty = max(usd_qty / last_price, min_qty)
                            qty = math.floor(qty / step) * step
                            if qty <= 0:
                                continue
                            # ---- total‑exposure safeguard ---------------------------------
                            current_total = await self.get_total_open_volume()
                            est_cost_usdt = qty * last_price
                            if not await self.can_open_position(est_cost_usdt):
                                logger.info(
                                    "[liq_trade] Skip %s – exposure limit reached (%.0f + %.0f > %.0f)",
                                    symbol, current_total, est_cost_usdt, self.MAX_TOTAL_VOLUME
                                )
                                continue

                        self.pending_orders.add(symbol)
                        self.pending_timestamps[symbol] = time.time()
                        self.pending_strategy_comments[symbol] = "liq‑cluster"

                        # ── Проверяем фактический размер позиции ──────────────────────────────
                        try:
                            resp = await asyncio.to_thread(lambda: self.session.place_order(
                                category="linear",
                                symbol=symbol,
                                side=order_side,           # противоположная сторона к кластеру
                                orderType="Market",
                                qty=self._format_qty(symbol, qty),
                                positionIdx=1 if order_side == "Buy" else 2,
                                timeInForce="IOC",
                                reduceOnly=False,
                            ))
                            if resp.get("retCode", 0) != 0:
                                raise InvalidRequestError(resp.get("retMsg", "order rejected"))

                            logger.info("[liq_trade] %s %s qty=%s opened by cluster (%s)",
                                        order_side, symbol, qty, side)
                        except Exception as e:
                            logger.warning("[liq_trade] %s error: %s", symbol, e)
                            self.pending_orders.discard(symbol)

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
            data = self.open_positions.get(symbol)
            # если у нас нет данных по avg_price или avgPrice – выходим
            if not data:
                logger.info(f"[evaluate_position] No open position data for {symbol}, skipping")
                return
            # support both underscore and camelCase keys
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
                    # open_int = self.shared_ws.latest_open_interest.get(symbol, 0.0)
                    # await self.log_trade(
                    #     symbol,
                    #     side=data["side"],
                    #     avg_price=avg_price,
                    #     volume=size,
                    #     open_interest=open_int,
                    #     action="trailing_set",
                    #     result="set",
                    #     closed_manually=False
                    # )
                    self.last_trailing_stop_set[symbol] = pnl_pct


    async def on_ticker_update(self, symbol, last_price):
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
                    symbols_shuffled = [
                        s for s in self.shared_ws.active_symbols
                        if s not in ("BTCUSDT", "ETHUSDT", "NEIROCTOUSDT")
                    ]
                    random.shuffle(symbols_shuffled)
                    tasks = []
                    for symbol in symbols_shuffled:
                        task = asyncio.create_task(
                            self.execute_golden_setup(symbol),
                            name=f"execute_golden_setup-{symbol}"
                        )
                        tasks.append(task)
                    
                    # Ожидаем завершения всех задач для текущей итерации
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    if time.time() - last_heartbeat >= 60:
                        logger.info(
                            "[market_loop] alive (iter=%d) — scanned %d symbols",
                            iteration, len(self.shared_ws.symbols)
                        )
                        last_heartbeat = time.time()
                    await asyncio.sleep(0.5)  # Уменьшаем задержку
            except asyncio.CancelledError:
                raise
            except Exception as fatal:
                logger.exception("[market_loop] fatal exception — restarting: %s", fatal)
                await asyncio.sleep(2)  # Уменьшаем время ожидания перед перезапуском

    async def execute_golden_setup(self, symbol: str):
        #logger.info(f"GOLDEN SETUP STARTED")
        try:
            """
            Анализ и исполнение «golden setup» для symbol:
            – вычисляем Δ цены/объёма/OI,
            – при сигнале (Buy/Sell) рассчитываем qty и выставляем рыночный ордер,
            – логируем через log_trade с передачей avg_price и volume,
            – подтверждаем открытие через REST или обрабатываем ошибки.
            """
            golden_enabled = self.strategy_mode in ("full", "golden_only")
            # Skip symbols manually closed to prevent re-entry
            age = await self.listing_age_minutes(symbol)
            if age < self.listing_age_min:
                logger.info("[listing_age] %s %.0f min < %d min – skip",
                            symbol, age, self.listing_age_min)
                return

            if symbol in self.closed_positions:
                return
            # Пропускаем символы с ростом 5% за 20 минут
            if self._squeeze_allowed(symbol) \
            and self.shared_ws.has_5_percent_growth(symbol, minutes=20):
                logger.debug(f"[GoldenSetup] Пропуск {symbol}: рост ≥3% за 20 минут")
                return

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

            # # ---- SQUEEZE STRATEGY (общая для всех режимов) ----------------
            # recent_1m = self.shared_ws.candles_data.get(symbol, [])
            # if len(recent_1m) >= 6:  # 5 завершённых баров (≈5 мин)
            #     # Получаем актуальную цену из тикера или последней свечи
            #     ticker_data = self.shared_ws.ticker_data.get(symbol, {})
            #     last_price = safe_to_float(ticker_data.get("lastPrice", 0))
                
            #     # Fallback на цену закрытия последней свечи
            #     if last_price <= 0 and recent_1m:
            #         last_price = safe_to_float(recent_1m[-1]["closePrice"])
                    
            #     # Если цена всё ещё не определена - пропускаем
            #     if last_price <= 0:
            #         return

            #     old_close_s = safe_to_float(recent_1m[-6]["closePrice"])
            #     new_close_s = safe_to_float(recent_1m[-1]["closePrice"])
            #     if old_close_s > 0:
            #         pct_5m = (new_close_s - old_close_s) / old_close_s * 100

            #         # Рассчитываем абсолютное относительное изменение объёма
            #         old_vol = safe_to_float(recent_1m[-6]["volume"])
            #         new_vol = safe_to_float(recent_1m[-1]["volume"])
            #         vol_change_pct = abs((new_vol - old_vol) / old_vol * 100) if old_vol > 0 else 0

            #         # Мощность сквиза: |ΔP| × |ΔV|
            #         squeeze_power = abs(pct_5m) * vol_change_pct

            #         # Дефолтный уровень для лимитного ордера: текущая цена закрытия
            #         default_price = new_close_s

            #         action = None
            #         limit_price = None
            #         side = None
            #         position_idx = None

            #         # SHORT: рост цены ≥ squeeze_threshold_pct (например, 5.5%)
            #         if pct_5m >= self.squeeze_threshold_pct:
            #             action = "SHORT"
            #             side = "Sell"
            #             position_idx = 2
            #             # Для SHORT ордер выше текущей цены, max +5%
            #             deviation_pct = 1 + SQUEEZE_LIMIT_OFFSET_PCT       # +1 %
            #             limit_price = default_price * deviation_pct
            #             logger.info("[SQUEEZE] %s ΔP=+%.2f%% за 5 мин — открываем SHORT", symbol, pct_5m)


            #         # LONG: падение цены ≤ -squeeze_threshold_pct (например, -5.5%)
            #         elif pct_5m <= -self.squeeze_threshold_pct:
            #             action = "LONG"
            #             side = "Buy"
            #             position_idx = 1
            #             # Для LONG ордер ниже текущей цены, max -5%
            #             deviation_pct = 1 - SQUEEZE_LIMIT_OFFSET_PCT       # +1 %
            #             limit_price = default_price * deviation_pct
            #             logger.info("[SQUEEZE] %s ΔP=%.2f%% за 5 мин — открываем LONG", symbol, pct_5m)

            #         if action:
            #             # # Округляем цену до шага tick
            #             # tick = float(DEC_TICK)
            #             # limit_price = round(math.floor(limit_price / tick) * tick, 6)

            #             # Добавить после расчета:
            #             tick = float(DEC_TICK)
            #             limit_price = round(math.floor(limit_price / tick) * tick, 6)

            #             # Гарантировать правильное расположение цены
            #             if action == "SHORT" and limit_price <= last_price:
            #                 limit_price = math.floor((last_price * (1 + SQUEEZE_LIMIT_OFFSET_PCT) + tick) / tick) * tick
            #             elif action == "LONG" and limit_price >= last_price:
            #                 limit_price = math.floor((last_price * (1 - SQUEEZE_LIMIT_OFFSET_PCT) - tick) / tick) * tick

            #             logger.info(
            #                 f"[SQUEEZE] {symbol} {action} limit order: "
            #                 f"qty={qty:.6f} price={limit_price:.6f} "
            #                 f"({SQUEEZE_LIMIT_OFFSET_PCT*100}% offset)"
            #             )

            #             # Лимит позиций / экспозиции
            #             async with self.position_lock:
            #                 usd_size = min(self.POSITION_VOLUME, self.max_allowed_volume)
            #                 total_expo = await self.get_total_open_volume()
            #                 if total_expo + usd_size > self.MAX_TOTAL_VOLUME:
            #                     logger.info("[SQUEEZE] skip %s: exposure %.0f + %.0f > %.0f",
            #                                 symbol, total_expo, usd_size, self.MAX_TOTAL_VOLUME)
            #                     return

            #                 await self.ensure_symbol_meta(symbol)
            #                 step = self.qty_step_map.get(symbol, 0.001)
            #                 min_qty = self.min_qty_map.get(symbol, step)
            #                 qty = max(math.floor((usd_size / limit_price) / step) * step, min_qty)
            #             if qty <= 0:
            #                 logger.warning("[SQUEEZE] %s: calculated qty=%.6f is invalid", symbol, qty)
            #                 return

            #             # Помечаем комментарий и резервируем ордер ДО отправки
            #             self.pending_strategy_comments[symbol] = f"Сквиз {self.squeeze_threshold_pct}%/5m ({action})"
            #             self.pending_orders.add(symbol)
            #             self.pending_timestamps[symbol] = time.time()

            #             # --- RSI entry protection ---
            #             if self.shared_ws.rsi_blocked(symbol, side):
            #                 logger.info("[RSI_guard] %s %s blocked: persistent RSI14 extreme", symbol, side)
            #                 return

            #             try:
            #                 if self.mode == "real":
            #                     await self.place_order_ws(
            #                         symbol=symbol,
            #                         side=side,
            #                         qty=qty,
            #                         position_idx=position_idx,
            #                         price=limit_price,
            #                         order_type="Limit"
            #                     )
            #                 else:
            #                     resp = await asyncio.to_thread(lambda: self.session.place_order(
            #                         category="linear",
            #                         symbol=symbol,
            #                         side=side,
            #                         orderType="Limit",
            #                         qty=self._format_qty(symbol, qty),
            #                         price=str(limit_price),
            #                         timeInForce="GTC",
            #                         positionIdx=position_idx
            #                     ))
            #                     if resp.get("retCode", 0) != 0:
            #                         raise InvalidRequestError(resp.get("retMsg", "order rejected"))
            #                     oid = resp.get("result", {}).get("orderId")
            #                     if oid:
            #                         self.reserve_orders[symbol] = {
            #                             "orderId":      oid,
            #                             "ts":           time.time(),
            #                             "action":       action,          # LONG | SHORT
            #                             "squeeze_power": squeeze_power,
            #                             "price":        limit_price,
            #                             "qty":          qty
            #                         }
            #                 logger.info("[SQUEEZE] %s %s qty=%.6f limit_price=%.6f opened", symbol, action, qty, limit_price)
            #             except Exception as e:
            #                 logger.warning("[SQUEEZE] order failed for %s: %s", symbol, e)
            #                 self.pending_orders.discard(symbol)
            #             return  # не продолжаем к другим стратегиям

            # ---- SQUEEZE STRATEGY (диагностический патч) ----------------
            recent_1m = self.shared_ws.candles_data.get(symbol, [])
            if len(recent_1m) >= 2:
                ticker_data = self.shared_ws.ticker_data.get(symbol, {})
                last_price = safe_to_float(ticker_data.get("lastPrice", 0))

                if last_price <= 0 and recent_1m:
                    last_price = safe_to_float(recent_1m[-1]["closePrice"])

                if last_price <= 0:
                    logger.info(f"[SQUEEZE] {symbol} пропущен: last_price <= 0")
                    return

                old_close_s = safe_to_float(recent_1m[-2]["closePrice"])
                new_close_s = safe_to_float(recent_1m[-1]["closePrice"])

                if old_close_s > 0:
                    pct_5m = (new_close_s - old_close_s) / old_close_s * 100
                    old_vol = safe_to_float(recent_1m[-2]["volume"])
                    new_vol = safe_to_float(recent_1m[-1]["volume"])
                    vol_change_pct = abs((new_vol - old_vol) / old_vol * 100) if old_vol > 0 else 0

                    squeeze_power = abs(pct_5m) * vol_change_pct

                    prev_vol_5m = sum_last_vol(self.shared_ws.candles_data[symbol], minutes=5)
                    curr_vol_1m = sum_last_vol(self.shared_ws.candles_data[symbol], minutes=1)
                    vol_change_pct = (curr_vol_1m - prev_vol_5m) / prev_vol_5m * 100.0 if prev_vol_5m > 0 else 0.0
                    squeeze_power = abs(pct_5m) * abs(vol_change_pct)

                    logger.info(f"[SQUEEZE_DIAG] {symbol} ΔP={pct_5m:.2f}%, ΔV={vol_change_pct:.2f}%, power={squeeze_power:.2f}")

                    if squeeze_power < self.squeeze_power_min:
                        logger.info(f"[SQUEEZE_DIAG] {symbol} пропуск: power {squeeze_power:.2f} < min {self.squeeze_power_min}")
                        return

                    action = None
                    side = None
                    position_idx = None
                    tick = float(DEC_TICK)
                    offset = SQUEEZE_LIMIT_OFFSET_PCT

                    if pct_5m >= self.squeeze_threshold_pct:
                        limit_price = last_price * (1 - offset)
                        limit_price = math.floor(limit_price / tick) * tick
                        if limit_price >= last_price:
                            limit_price = last_price - tick
                        action, side, position_idx = "SHORT", "Sell", 2

                    elif pct_5m <= -self.squeeze_threshold_pct:
                        limit_price = last_price * (1 + offset)
                        limit_price = math.ceil(limit_price / tick) * tick
                        if limit_price <= last_price:
                            limit_price = last_price + tick
                        action, side, position_idx = "LONG", "Buy", 1

                    if not action:          # защита от ложного срабатывания
                        return

                    # ---------- общий расчёт объёма -------------
                    async with self.position_lock:
                        usd_size = min(self.POSITION_VOLUME, self.max_allowed_volume)
                        total_expo = await self.get_total_open_volume()
                        if total_expo + usd_size > self.MAX_TOTAL_VOLUME:
                            logger.info("[SQUEEZE_DIAG] %s превышен exposure %.0f + %.0f > %.0f",
                                        symbol, total_expo, usd_size, self.MAX_TOTAL_VOLUME)
                            return

                        await self.ensure_symbol_meta(symbol)
                        step     = self.qty_step_map.get(symbol, 0.001)
                        min_qty  = self.min_qty_map.get(symbol, step)
                        qty      = max(math.floor((usd_size / last_price) / step) * step, min_qty)
                        logger.info(f"[SQUEEZE_DIAG] {symbol} qty={qty:.6f} (step={step})")

                    if not action:
                        return

                    bid_price = safe_to_float(ticker_data.get("bid1Price", 0))
                    ask_price = safe_to_float(ticker_data.get("ask1Price", 0))
                    if bid_price > 0 and ask_price > 0:
                        spread_pct = (ask_price - bid_price) / bid_price * 100
                        logger.info(f"[SQUEEZE_DIAG] {symbol} spread={spread_pct:.3f}%")
                        if spread_pct > 0.1:
                            logger.info(f"[SQUEEZE_DIAG] {symbol} пропуск: spread {spread_pct:.3f}% > 0.1%")
                            return

                    if qty <= 0:
                        logger.info(f"[SQUEEZE_DIAG] {symbol} qty <= 0")
                        return

                    #if self.shared_ws.rsi_blocked(symbol, side):
                    #    logger.info(f"[SQUEEZE_DIAG] {symbol} блокировка RSI")
                    #    return

                    logger.info(f"[SQUEEZE_DIAG] {symbol} прошёл все фильтры, готов к adaptive entry")

                    # Помечаем комментарий и резервируем ордер ДО отправки
                    self.pending_strategy_comments[symbol] = f"Сквиз {self.squeeze_threshold_pct}%/5m ({action})"
                    self.pending_orders.add(symbol)
                    self.pending_timestamps[symbol] = time.time()

                    try:
                        if self.mode == "real":
                            await self.adaptive_squeeze_entry_ws(
                                symbol=symbol,
                                side=side,
                                qty=qty,
                                position_idx=position_idx,
                                max_entry_timeout=15
                            )

                            # await self.place_order_ws(
                            #     symbol=symbol,
                            #     side=side,
                            #     qty=qty,
                            #     position_idx=position_idx,
                            #     price=limit_price,
                            #     order_type="Limit"
                            # )
                        else:
                            # Demo / Backtest / Тестовый режим — безопасный REST-адаптивный лимитник
                            await self.adaptive_squeeze_entry(
                                symbol=symbol,
                                side=side,
                                qty=qty,
                                max_entry_timeout=15
                            )

                        logger.info("[SQUEEZE] %s %s qty=%.6f adaptive entry started", 
                                    symbol, action, qty)
                        self.last_squeeze_ts[symbol] = time.time()

                    except Exception as e:
                        logger.warning("[SQUEEZE] order failed for %s: %s", symbol, e)
                        self.pending_orders.discard(symbol)

                    return
                        #         resp = await asyncio.to_thread(lambda: self.session.place_order(
                        #             category="linear",
                        #             symbol=symbol,
                        #             side=side,
                        #             orderType="Limit",
                        #             qty=self._format_qty(symbol, qty),
                        #             price=str(limit_price),
                        #             timeInForce="GTC",
                        #             positionIdx=position_idx
                        #         ))
                                
                        #         if resp.get("retCode", 0) != 0:
                        #             raise InvalidRequestError(resp.get("retMsg", "order rejected"))
                                    
                        #         if symbol in self.reserve_orders:
                        #             return
                        #         oid = resp.get("result", {}).get("orderId")
                        #         if oid:
                        #             self.reserve_orders[symbol] = {
                        #                 "orderId":      oid,
                        #                 "side":         side,
                        #                 "ts":           time.time(),
                        #                 "action":       action,
                        #                 "squeeze_power": squeeze_power,
                        #                 "price":        limit_price,
                        #                 "qty":          qty,
                        #                 "last_reprice_ts": 0  # Для отслеживания времени последнего обновления цены
                        #             }
                                    
                        #     logger.info("[SQUEEZE] %s %s qty=%.6f limit_price=%.6f opened", 
                        #                 symbol, action, qty, limit_price)
                        #     # отмечаем время, чтобы включился кулдаун
                        #     self.last_squeeze_ts[symbol] = time.time()
                        # except Exception as e:
                        #     logger.warning("[SQUEEZE] order failed for %s: %s", symbol, e)
                        #     self.pending_orders.discard(symbol)
                        # return

                # 3. Берём историю свечей/объёма/OI
                # # ---- LIQUIDATION INFO ------------------------------------------------
                # liq_data = self.shared_ws.latest_liquidation.get(symbol, {})
                # liq_val  = safe_to_float(liq_data.get("value", 0))
                # liq_side = liq_data.get("side")          # 'Buy' | 'Sell' | None

                # # --- strategy switches ---------------------------------
                # mode = getattr(self, "strategy_mode", "full")
                # golden_enabled = mode in ("golden_only", "full")
                # liq_enabled    = mode in ("liquidation_only", "full")


                # # 4‑A. Торговля от крупной ликвидации (v2: динамический порог + фильтры)
                # # TradingBot не имеет собственного метода get_liq_threshold ─ берём из shared_ws
                # threshold = self.shared_ws.get_liq_threshold(symbol, 5000)
                # avg_vol_30m = self.shared_ws.get_avg_volume(symbol, 30)
                # delta_oi = self.shared_ws.get_delta_oi(symbol)
                # cooldown_ok = self.shared_ws.check_liq_cooldown(symbol)

                # candles = self.shared_ws.candles_data.get(symbol, [])
                # candle_ok = False
                # if candles:
                #     candle_ok = self.shared_ws.is_volatile_spike(symbol, candles[-1])
                # # --- текущий объём (последняя 5‑мин свеча) ---
                # if candles:
                #     volume_now = safe_to_float(candles[-1].get("volume")
                #                             or candles[-1].get("turnover", 0))
                # else:
                #     # fallback на историю объёмов, если свечи ещё не прогреты
                #     vol_hist = self.shared_ws.volume_history.get(symbol, [])
                #     volume_now = safe_to_float(vol_hist[-1]) if vol_hist else 0.0

                # funding_ok = (
                #     self.shared_ws.funding_cool(symbol)
                #     if self.shared_ws else True
                # )

                # # ---- 3-баровое среднее ΔV / ΔOI ------------------------
                # vol_hist = list(self.shared_ws.volume_history.get(symbol, []))
                # oi_hist  = list(self.shared_ws.oi_history.get(symbol, []))

                # # Initialize mean_dvol and mean_doi with default values
                # mean_dvol = 0.0
                # mean_doi = 0.0

                # # быстрые deltas «сейчас – 1 мин» (для fallback)
                # volume_change = 0.0
                # oi_change     = 0.0
                # if len(vol_hist) >= 2:
                #     volume_change = (vol_hist[-1] - vol_hist[-2]) / max(1e-8, vol_hist[-2]) * 100
            
                # if len(oi_hist) >= 2:
                #     oi_change = (oi_hist[-1] - oi_hist[-2]) / max(1e-8, oi_hist[-2]) * 100

                # # Always assign fallback mean values even if both hist lists are short
                # mean_dvol = volume_change
                # mean_doi  = oi_change

                # # ---- liquidation analytics filter ----
                # liq_info = self.shared_ws.latest_liquidation.get(symbol, {}) if self.shared_ws else {}
                # liq_val  = liq_info.get("value", 0.0)
                # liq_side = liq_info.get("side", "")
                # liq_ts   = liq_info.get("ts", 0.0)

                # # ликвидация считается «свежей», если была ≤ 60 сек назад
                # liq_recent = (time.time() - liq_ts) <= 60 and liq_val >= threshold

                # passed_filters = (
                #     liq_enabled
                #     and liq_val >= threshold
                #     and candle_ok
                #     and mean_dvol > 0
                #     and mean_doi  > 0
                #     and delta_oi is not None and delta_oi <= -0.003
                #     and volume_now >= VOLUME_COEF * avg_vol_30m
                #     and funding_ok
                #     and cooldown_ok
                #     and liq_side in ("Buy", "Sell")
                # )
                # if passed_filters:
                #     opposite = "Buy" if liq_side == "Sell" else "Sell"
                #     if symbol not in self.open_positions and symbol not in self.pending_orders:
                #         self.pending_strategy_comments[symbol] = "От ликвидаций"
                #         # ---- лимит общего экспозиционного объёма -------------------
                #         total_expo = await self.get_total_open_volume()
                #         potential  = self.POSITION_VOLUME    # в USDT, т.к. qty ещё не посчитан
                #         if total_expo + potential > self.MAX_TOTAL_VOLUME:
                #             logger.warning("[LiqTrade] skip %s: exposure %.0f + %.0f > %.0f",
                #                         symbol, total_expo, potential, self.MAX_TOTAL_VOLUME)
                #             return
                        
                #         # Берём последнюю цену
                #         close_price = safe_to_float(
                #             self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
                #         ) or safe_to_float(
                #             self.shared_ws.candles_data.get(symbol, [])[-1]["closePrice"]
                #         )
                #         if close_price > 0:
                #             step = self.qty_step_map.get(symbol, 0.001)
                #             # вычисляем количество знаков после запятой у qtyStep
                #             step_str = str(step)
                #             dec = len(step_str.split(".")[1].rstrip("0")) if "." in step_str else 0
                #             qty_raw = safe_to_float(self.POSITION_VOLUME) / close_price
                #             qty = math.floor(qty_raw / step) * step                 # округление к шагу
                #             qty = round(qty, dec)                                   # убираем «166.10000002»
                #             qty_str = f"{qty:.{dec}f}"
                #             pos_idx = 1 if opposite == "Buy" else 2                 # индекс позиции
                #             try:
                #                 logger.info("[LiqTrade] %s крупная %s-ликв %.0f USDT → %s %.3f",
                #                             symbol, liq_side, liq_val, opposite, qty)
                                
                #                 if mean_dvol <= 0 or mean_doi <= 0:
                #                     logger.info("[Golden] %s отменён: ΔV/ΔOI стали отрицательны "
                #                                 "(dV=%.2f  dOI=%.3f)", symbol, mean_dvol, mean_doi)
                #                     return

                #                 if self.mode == "real":
                #                     if self.mode == "real":
                #                         await self.place_order_ws(symbol, opposite, qty,
                #                                                 position_idx=pos_idx)
                #                 else:
                #                     resp = await asyncio.to_thread(lambda: self.session.place_order(
                #                         category="linear",
                #                         symbol=symbol,
                #                         side=opposite,
                #                         orderType="Market",
                #                         qty=qty_str,
                #                         timeInForce="GTC",
                #                         positionIdx=pos_idx
                #                     ))
                #                     if resp.get("retCode", 0) != 0:
                #                         raise InvalidRequestError(resp.get("retMsg", "order rejected"))
                #                 # --- помечаем ордер ДО отправки, чтобы WS успел его прочитать ---
                #                 self.pending_strategy_comments[symbol] = "Чистый Золотой Сетап"
                #                 self.pending_orders.add(symbol)
                #                 self.pending_timestamps[symbol] = time.time()
                #             except Exception as e:
                #                 logger.warning("[LiqTrade] order failed for %s: %s", symbol, e)
                #     return  # после торговли от ликвидации — пропускаем golden‑setup дальше

                # --- strategy switches ---------------------------------
                mode = getattr(self, "strategy_mode", "full")
                golden_enabled = mode in ("golden_only", "full")
                liq_enabled = mode in ("liquidation_only", "full")

                # ---- LIQUIDATION INFO ------------------------------------------------
                if liq_enabled:
                    liq_info = self.shared_ws.latest_liquidation.get(symbol, {})
                    liq_val = safe_to_float(liq_info.get("value", 0))
                    liq_side = liq_info.get("side", "")
                    liq_ts = liq_info.get("ts", 0.0)

                    liq_recent = (time.time() - liq_ts) <= 60
                    threshold = self.shared_ws.get_liq_threshold(symbol, 5000)

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
                    if delta_oi > -0.0015:
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

            buy_params  = self.golden_param_store.get((symbol, "Buy"),  self.golden_param_store.get("Buy"))
            sell_params = self.golden_param_store.get((symbol, "Sell"), self.golden_param_store.get("Sell"))
            #sell2_params= self.golden_param_store.get((symbol, "Sell2"),self.golden_param_store.get("Sell2", {}))

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
            
            # logger.info(
            #     "[GoldenSetup] %s | ΔP=%.3f%%  ΔV=%.1f%%  ΔOI=%.2f%%  ΔCVD=%.1f%%  Strength=%.2f  iters=%d",
            #     symbol,
            #     price_change_pct,
            #     volume_change_pct,
            #     float(oi_change_pct),
            #     cvd_change_pct,
            #     signal_strength,
            #     period_iters
            # )

            # if cvd_change_pct >= 50:
            #     logger.info(f"{symbol} has CVD = {cvd_change_pct}")

            # if cvd_change_pct <= -100:
            #     logger.info(f"{symbol} has CVD = {cvd_change_pct}")

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

                # SELL
                if (
                    pchg <= -Decimal(str(sell_params["price_change"]))
                    and volchg >= Decimal(str(sell_params["volume_change"]))
                    and oichg >= Decimal(str(sell_params["oi_change"]))
                    and (-50 <= cvd_change_pct <= -18.3)
                    and not (liq_side == "Sell" and liq_val >= threshold)
                ):
                    action = "Sell"
    
            # # альтернативный Sell2
            # if action is None and sell2_params:
            #     sp2 = int(sell2_params["period_iters"])
            #     if len(recent) > sp2:
            #         old2, new2 = recent[-1 - sp2], recent[-1]
            #         pchg2 = (Decimal(str(new2["closePrice"])) - Decimal(str(old2["closePrice"]))) / Decimal(str(old2["closePrice"])) * Decimal("100") if old2["closePrice"] else Decimal("0")
            #         volchg2 = (Decimal(str(self.shared_ws.volume_history[symbol][-1 - sp2])) - Decimal(str(self.shared_ws.volume_history[symbol][-1]))) / Decimal(str(self.shared_ws.volume_history[symbol][-1 - sp2])) * Decimal("100") if self.shared_ws.volume_history[symbol][-1 - sp2] else Decimal("0")
            #         oichg2 = (Decimal(str(self.shared_ws.oi_history[symbol][-1 - sp2])) - Decimal(str(self.shared_ws.oi_history[symbol][-1]))) / Decimal(str(self.shared_ws.oi_history[symbol][-1 - sp2])) * Decimal("100") if self.shared_ws.oi_history[symbol][-1 - sp2] else Decimal("0")
            #         if (pchg2 <= Decimal(str(sell2_params["price_change"])) * -1 and
            #             volchg2 <= Decimal(str(sell2_params["volume_change"])) and
            #             oichg2 <= Decimal(str(sell2_params["oi_change"]))):
            #             action = "Sell"
            # Buy
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

                    # BUY
                    if (
                        pchgb >= Decimal(str(buy_params["price_change"]))
                        and volb >= Decimal(str(buy_params["volume_change"]))
                        and oib  >= Decimal(str(buy_params["oi_change"]))
                        and (18.3 <= cvd_change_pct <= 200)
                        and not (liq_side == "Buy" and liq_val >= threshold)
                    ):
                        action = "Buy"

            if action is None:
                return

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

            step    = self.qty_step_map.get(symbol, 0.001)
            min_qty = self.min_qty_map.get(symbol, step)
            qty_raw = volume_usdt / last_price
            # round down to the nearest step
            factor = math.floor(qty_raw / step)
            qty = factor * step
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

        try:
            tick = float(DEC_TICK)
            offset_pct = SQUEEZE_LIMIT_OFFSET_PCT
            reprice_interval = SQUEEZE_REPRICE_INTERVAL

            order_id = None
            started_ts = asyncio.get_running_loop().time()

            while True:
                # Актуальная цена через WS
                ticker = self.shared_ws.ticker_data.get(symbol, {})
                last_price = safe_to_float(ticker.get("lastPrice", 0))
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
            logger.exception(f"[adaptive_ws] Критическая ошибка {symbol}")


    async def adaptive_squeeze_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        max_entry_timeout: int = 15
    ):
        """
        Адаптивный лимитный вход в позицию для сквизов.
        Ставит лимитник с подвижкой за ценой.

        :param symbol: тикер
        :param side: 'Buy' или 'Sell'
        :param qty: объём
        :param max_entry_timeout: макс. время попыток (секунд)
        """
        try:
            # вычисляем параметры
            tick = float(DEC_TICK) if isinstance(DEC_TICK, (int, float)) else float(DEC_TICK)
            offset_pct = SQUEEZE_LIMIT_OFFSET_PCT
            reprice_interval = SQUEEZE_REPRICE_INTERVAL

            order_id = None
            pos_idx = 1  # всегда linear
            started_ts = asyncio.get_running_loop().time()

            while True:
                # получаем актуальную цену через WS (best bid / ask)
                ticker = self.shared_ws.ticker_data.get(symbol, {})
                last_price = safe_to_float(ticker.get("lastPrice", 0))
                best_bid = safe_to_float(ticker.get("bid1Price", last_price))
                best_ask = safe_to_float(ticker.get("ask1Price", last_price))

                if last_price == 0:
                    logger.warning(f"[adaptive_entry] Нет цены для {symbol}, пропуск...")
                    await asyncio.sleep(reprice_interval)
                    continue

                # рассчитываем цену лимитника
                if side.lower() == "buy":
                    raw_price = best_bid * (1 - offset_pct)
                    limit_price = math.floor(raw_price / tick) * tick
                else:
                    raw_price = best_ask * (1 + offset_pct)
                    limit_price = math.ceil(raw_price / tick) * tick

                # отменяем предыдущий ордер если есть
                if order_id:
                    try:
                        await asyncio.to_thread(
                            lambda: self.session.cancel_order(
                                category="linear",
                                symbol=symbol,
                                orderId=order_id
                            )
                        )
                        logger.debug(f"[adaptive_entry] Cancelled previous order {order_id}")
                    except Exception:
                        pass

                # выставляем новый лимитный ордер
                try:
                    resp = await asyncio.to_thread(
                        lambda: self.session.place_order(
                            category="linear",
                            symbol=symbol,
                            side=side.capitalize(),
                            orderType="Limit",
                            qty=str(qty),
                            price=str(limit_price),
                            timeInForce="PostOnly",
                            reduceOnly=False,
                            positionIdx=pos_idx
                        )
                    )
                    order_id = resp.get("result", {}).get("orderId")
                    logger.info(f"[adaptive_entry] {symbol} лимитник {side} {qty} @ {limit_price}")

                except Exception as e:
                    logger.warning(f"[adaptive_entry] Ошибка постановки ордера: {e}")
                    await asyncio.sleep(reprice_interval)
                    continue

                # ждём исполнения или перезаказ
                await asyncio.sleep(reprice_interval)

                # проверяем исполнение
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
                        logger.info(f"[adaptive_entry] {symbol} исполнен {status}")
                        break  # успешный вход — выходим

                except Exception:
                    pass

                # таймаут: выходим из цикла
                if asyncio.get_running_loop().time() - started_ts > max_entry_timeout:
                    logger.warning(f"[adaptive_entry] Время на вход истекло для {symbol}")
                    break

            # финальная отмена висящего ордера, если не заполнилось
            if order_id:
                try:
                    await asyncio.to_thread(
                        lambda: self.session.cancel_order(
                            category="linear",
                            symbol=symbol,
                            orderId=order_id
                        )
                    )
                except Exception:
                    pass

        except Exception:
            logger.exception(f"[adaptive_entry] Критическая ошибка при входе в {symbol}")


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
        s_result = (result or "").lower()
        s_side = side or ""
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


    # ──────────────────────────────────────────────────────────────────
    # async def set_trailing_stop(
    #     self,
    #     symbol: str,
    #     avg_price: float,
    #     pnl_pct: float,
    #     side: str,
    # ):
    #     """
    #     Ставит / подтягивает трейлинг-стоп.

    #     ▸ Включается при достижении прибыли ≥ trailing_start_pct[mode].
    #     ▸ Стоп держится в gap_pct % от текущей цены.
    #     ▸ Никогда не «ухудшается» (не отодвигается дальше от цены).
    #     """
    #     try:
    #         # -------- параметры из пользовательского JSON -------------
    #         mode          = getattr(self, "mode", "full")          # full | golden_only | liquidation_only
    #         #start_pct     = self.trailing_start_pct[mode]          # 5.0
    #         #gap_pct       = self.trailing_gap_pct[mode]            # 2.7
    #             # если это словарь – берём по ключу; иначе используем само значение
    #         if isinstance(self.trailing_start_pct, dict):
    #             start_pct = self.trailing_start_pct.get(mode,                     # full / golden_only …
    #                                                     next(iter(self.trailing_start_pct.values())))
    #         else:
    #             start_pct = float(self.trailing_start_pct)      # уже число

    #         if isinstance(self.trailing_gap_pct, dict):
    #             gap_pct = self.trailing_gap_pct.get(mode,
    #                                                 next(iter(self.trailing_gap_pct.values())))
    #         else:
    #             gap_pct = float(self.trailing_gap_pct)
                
    #         tick          = float(DEC_TICK) if isinstance(DEC_TICK, (int, float)) else float(DEC_TICK)

    #         if pnl_pct < start_pct:
    #             return                                            # ещё рано

    #         data = self.open_positions.get(symbol)
    #         if not data:
    #             return
    #         volume  = safe_to_float(data.get("volume", 0))
    #         pos_idx = data["pos_idx"]
    #         if volume <= 0:
    #             return

    #         # ------------- текущая цена (last) ------------------------
    #         last_price = safe_to_float(
    #             self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
    #         )
    #         if last_price == 0:
    #             return

    #         # ------------- рассчитываем стоп --------------------------
    #         if side.lower() == "buy":          # long
    #             raw_price = last_price * (1 - gap_pct / 1000)
    #             new_stop  = math.floor(raw_price / tick) * tick
    #         else:                              # short
    #             raw_price = last_price * (1 + gap_pct / 1000)
    #             new_stop  = math.ceil(raw_price / tick) * tick

    #         prev_stop = self.last_stop_price.get(symbol)
    #         if prev_stop is not None:
    #             better = (side.lower() == "buy"  and new_stop > prev_stop) or \
    #                      (side.lower() == "sell" and new_stop < prev_stop)
    #             if not better:
    #                 return                      # новый уровень хуже или равен — пропускаем

    #         # -------- отправляем set_trading_stop (до 3 ретраев) -------
    #         for attempt in range(1, 4):
    #             try:
    #                 async with self.limiter:
    #                     resp = await asyncio.to_thread(
    #                         lambda: self.session.set_trading_stop(
    #                             category="linear",
    #                             symbol=symbol,
    #                             positionIdx=pos_idx,
    #                             stopLoss=str(new_stop),
    #                             triggerBy="LastPrice",
    #                             timeInForce="GTC",
    #                         )
    #                     )
    #                 break
    #             except (RequestsReadTimeout, RequestsConnectionError, UrllibReadTimeoutError):
    #                 if attempt == 3:
    #                     raise
    #                 await asyncio.sleep(2 ** attempt)

    #         if resp.get("retCode", resp.get("ret_code", 0)) in (0, 34040):
    #             self.last_stop_price[symbol] = new_stop
    #             logger.info("[trailing_stop] %s set @ %.6f (gap %.2f%%)",
    #                         symbol, new_stop, gap_pct)
    #             # ► логируем сделку
    #             await self.log_trade(
    #                 symbol=symbol,
    #                 side=side,
    #                 avg_price=avg_price,
    #                 volume=volume,
    #                 open_interest=self.shared_ws.latest_open_interest.get(symbol, 0.0),
    #                 action="stoploss",
    #                 result="trailingstop",
    #             )
    #         else:
    #             logger.error("[trailing_stop] %s error: %s", symbol, resp)

    #     except Exception:
    #         logger.exception("[trailing_stop] unexpected error for %s", symbol)

    async def set_trailing_stop(
            self,
            symbol: str,
            avg_price: float,
            pnl_pct: float,
            side: str,
        ):
            """
            Ставит / подтягивает трейлинг-стоп.

            ▸ Включается при достижении прибыли ≥ trailing_start_pct[mode].
            ▸ Стоп держится в gap_pct % от текущей цены.
            ▸ Никогда не «ухудшается» (не отодвигается дальше от цены).
            """
            try:
                mode = getattr(self, "mode", "full")
                if isinstance(self.trailing_start_pct, dict):
                    start_pct = self.trailing_start_pct.get(mode, next(iter(self.trailing_start_pct.values())))
                else:
                    start_pct = float(self.trailing_start_pct)

                if isinstance(self.trailing_gap_pct, dict):
                    gap_pct = self.trailing_gap_pct.get(mode, next(iter(self.trailing_gap_pct.values())))
                else:
                    gap_pct = float(self.trailing_gap_pct)

                tick = float(DEC_TICK) if isinstance(DEC_TICK, (int, float)) else float(DEC_TICK)

                if pnl_pct < start_pct:
                    return

                data = self.open_positions.get(symbol)
                if not data:
                    return
                volume = safe_to_float(data.get("volume", 0))
                pos_idx = data["pos_idx"]
                if volume <= 0:
                    return

                last_price = safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0))
                if last_price == 0:
                    return

                if side.lower() == "buy":
                    raw_price = last_price * (1 - gap_pct / 1000)
                    new_stop = math.floor(raw_price / tick) * tick
                else:
                    raw_price = last_price * (1 + gap_pct / 1000)
                    new_stop = math.ceil(raw_price / tick) * tick

                # Достаем актуальный стоп из open_positions
                prev_stop = data.get("stop_loss")

                if prev_stop is not None:
                    better = (side.lower() == "buy" and new_stop > prev_stop) or \
                            (side.lower() == "sell" and new_stop < prev_stop)
                    if not better:
                        return  # уже стоит лучше или такой же стоп

                for attempt in range(1, 4):
                    try:
                        async with self.limiter:
                            resp = await asyncio.to_thread(
                                lambda: self.session.set_trading_stop(
                                    category="linear",
                                    symbol=symbol,
                                    positionIdx=pos_idx,
                                    stopLoss=str(new_stop),
                                    triggerBy="LastPrice",
                                    timeInForce="GTC",
                                )
                            )
                        break
                    except (RequestsReadTimeout, RequestsConnectionError, UrllibReadTimeoutError):
                        if attempt == 3:
                            raise
                        await asyncio.sleep(2 ** attempt)

                if resp.get("retCode", resp.get("ret_code", 0)) in (0, 34040):
                    self.last_stop_price[symbol] = new_stop
                    logger.info("[trailing_stop] %s set @ %.6f (gap %.2f%%)",
                                symbol, new_stop, gap_pct)
                    await self.log_trade(
                        symbol=symbol,
                        side=side,
                        avg_price=avg_price,
                        volume=volume,
                        open_interest=self.shared_ws.latest_open_interest.get(symbol, 0.0),
                        action="stoploss",
                        result="trailingstop",
                    )
                else:
                    logger.error("[trailing_stop] %s error: %s", symbol, resp)

            except Exception:
                logger.exception("[trailing_stop] unexpected error for %s", symbol)

    # async def stop(self):
    #     logger.info(f"[User {self.user_id}] Остановка бота")
    #     tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    #     for task in tasks:
    #         task.cancel()
    #     await asyncio.gather(*tasks, return_exceptions=True)
    #     if self.ws_private:
    #         self.ws_private.exit()
    #     logger.info(f"[User {self.user_id}] Полная остановка")


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
            # ── читаем старый файл (если он есть) ────────────────────────────────
            data = {}
            if os.path.exists(OPEN_POS_JSON):
                with open(OPEN_POS_JSON, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            # ── перезаписываем только свой user_id ──────────────────────────────
            data[str(self.user_id)] = snapshot
            _atomic_json_write(OPEN_POS_JSON, data)
        except Exception as e:
            logger.debug("[save_open_positions_json] %s", e)

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
            "period_iters": 3,
            "price_change": 1.7,      # +0.20 % price rise
            "volume_change": 200,      # +50 % volume surge
            "oi_change": 1.5,         # +0.40 % OI rise
        },
        "Sell": {
            "period_iters": 3,
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

async def run_all():
    users = load_users_from_json("user_state.json")
    if not users:
        print("❌ Нет активных пользователей для запуска.")
        return
    
    golden_param_store = load_golden_params()
    bots = []

    initial_symbols = ["BTCUSDT", "ETHUSDT"]
    shared_ws = PublicWebSocketManager(symbols=initial_symbols)

    # создаём всех ботов — но уже сразу передаем WS
    for u in users:
        bot = TradingBot(user_data=u, shared_ws=shared_ws, golden_param_store=golden_param_store)
        bots.append(bot)

    # создаём обратную связь для WS
    for bot in bots:
        shared_ws.position_handlers.append(bot)
    
    # тут можно сделать общую привязку shared_ws → bot
    # если нужен доступ к какому-то одному главному боту
    shared_ws.bot = bots[0]   # если у тебя один основной
    # или shared_ws.bots = bots  если хочешь сделать список ботов внутри shared_ws

    await shared_ws.backfill_history()
    public_ws_task = asyncio.create_task(shared_ws.start())
    
    bot_tasks = [asyncio.create_task(bot.start()) for bot in bots]
    
    # Подготовка к завершению
    async def shutdown():
        logger.info("Завершение работы всех ботов...")
        for bot in bots:
            await bot.stop()
        public_ws_task.cancel()
        await asyncio.gather(*bot_tasks, public_ws_task, return_exceptions=True)
        logger.info("Все задачи остановлены")

    # Запуск Telegram-бота
    #dp.include_router(router)
    #dp.include_router(router_admin)
    
    async def run_with_shutdown():
        try:
            tg_task = asyncio.create_task(dp.start_polling(telegram_bot))
            await asyncio.gather(*[public_ws_task, *bot_tasks, tg_task])
        except asyncio.CancelledError:
            await shutdown()
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}", exc_info=True)
            await shutdown()
    
    # Запуск основного цикла
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(shutdown()))
    loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(shutdown()))
    
    await run_with_shutdown()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("bot.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger.info("Логирование настроено: файл bot.log и консоль")
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        logger.info("\nПрограмма остановлена пользователем")
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