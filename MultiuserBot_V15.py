import datetime as dt
from aiogram.enums import ParseMode

# Ensure required imports
import asyncio
import pytz
from datetime import timedelta

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
import websockets
import hashlib
import requests
import pandas as pd
import pandas_ta as ta

from logging.handlers import RotatingFileHandler
from typing import List
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
import numpy as np




 # Telegram‑ID(-ы) администраторов, которым доступна команда /snapshot
ADMIN_IDS = {36972091}   # ← замените на свой реальный ID

# Глобальный реестр всех экземпляров TradingBot (используется для snapshot)
GLOBAL_BOTS: list = []

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

# Where per‑symbol optimal liquidation thresholds are stored
LIQ_THRESHOLD_CSV_PATH = "liq_thresholds.csv"
# Centralised location of historical liquidation events
LIQUIDATIONS_CSV_PATH = "liquidations.csv"

SQUEEZE_THRESHOLD_PCT = 5.5    # рост ≥ 3 % за 5 мин

# --- dynamic-threshold & volatility coefficients (v3) ---
LARGE_TURNOVER = 100_000_000     # 100 M USDT 24h turnover
MID_TURNOVER   = 10_000_000      # 10 M USDT
VOL_COEF       = 1.2             # ≥ 1.2σ spike
VOL_WINDOW     = 60              # 12 × 5-мин свечей = 1 час
VOLUME_COEF    = 3.0             # объём ≥ 3× ср.30 мин

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
    
def safe_to_float(val) -> float:       # ← переименовать или оставить
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
    hl2 = (df["highPrice"] + df["lowPrice"]) / 2
    atr = ta.atr(df["highPrice"], df["lowPrice"], df["closePrice"], length=period)
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
        "ready_event",
        "loop", "volume_history", "oi_history", "cvd_history",
        "_last_saved_time", "position_handlers", "_history_file",
        "_save_task", "latest_liquidation",
        "_liq_thresholds", "last_liq_trade_time", "funding_history",
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

            # select symbols that satisfy liquidity thresholds
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
            # ── push an OI snapshot so Golden Setup always sees current ΔOI ──
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



    # async def place_order_ws(self, symbol, side, qty, position_idx=1, price=None, order_type="Market"):
    #     header = {
    #         "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
    #         "X-BAPI-RECV-WINDOW": "5000"
    #     }
    #     args = {
    #         "symbol": symbol,
    #         "side": side,
    #         "orderType": order_type,
    #         "qty": str(qty),
    #         "category": "linear",
    #         "timeInForce": "GoodTillCancel"
    #     }
    #     # Указываем индекс позиции
    #     args["positionIdx"] = position_idx
    #     if price:
    #         args["price"] = str(price)

    #     req = {
    #         "op": "order.create",
    #         "header": header,
    #         "args": [ args ]
    #     }
    #     await self.ws_trade.send(json.dumps(req))
    #     resp = json.loads(await self.ws_trade.recv())
    #     if resp["retCode"] != 0:
    #         raise RuntimeError(f"Order failed: {resp}")
    #     return resp["data"]  # contains orderId, etc.

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
        "pending_timestamps", "squeeze_threshold_pct",
    )

    def __init__(self, user_data, shared_ws, golden_param_store):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.monitoring = user_data.get("monitoring", "http")
        self.mode = user_data.get("mode", "live")
        self.session = HTTP(demo=(self.mode == "demo"),
                            api_key=self.api_key,
                            api_secret=self.api_secret,
                            timeout=30)
        self.shared_ws = shared_ws
        # Регистрируемся на обновления тикера для trailing-stop (если shared_ws передан)
        if self.shared_ws is not None:
            self.shared_ws.position_handlers.append(self)
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
        self.mode = user_data.get("mode", "live")
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

        self.apply_user_settings()        # начальная синхронизация

        self.pending_timestamps = {}  # type: dict[str, float]


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
                print(f"[User {self.user_id}] ✅ ML-модель загружена из {model_path}")
            except Exception as e:
                self.model = None
                print(f"[User {self.user_id}] ❌ Ошибка загрузки модели: {e}")
        else:
            self.model = None
            print(f"[User {self.user_id}] ⚠️ model.txt не найден — ML сигналы отключены")

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

    
    async def start(self):
        logger.info(f"[User {self.user_id}] Бот запущен")
        # очистка кэша позиций перед первым REST-запросом
        self.open_positions.clear()
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

        await asyncio.sleep(480)  # ← Вставлено: 8 минут на «разогрев»

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
                open_int   = self.shared_ws.latest_open_interest.get(symbol, 0.0)
                prev       = self.open_positions.get(symbol)

                # 5) Открытие позиции
                if prev is None and new_size > 0 and side_raw:
                    self.open_positions[symbol] = {
                        "avg_price": avg_price,
                        "side":      side_raw,
                        "pos_idx":   position.get("positionIdx", 1),
                        "volume":    new_size,
                        "amount":    safe_to_float(position.get("positionValue"))
                    }
                    # Mark WS‑opened
                    self.ws_opened_symbols.add(symbol)
                    self.ws_closed_symbols.discard(symbol)

                    logger.info(f"[PositionStream] Scheduling evaluate_position for {symbol}")
                    asyncio.create_task(self.evaluate_position(position))

                    # Логируем открытие
                    asyncio.create_task(self.log_trade(
                        symbol,
                        side=side_raw,
                        avg_price=avg_price,
                        volume=new_size,
                        open_interest=open_int,
                        action="open",
                        result="opened"
                    ))

                    # Уведомление
                    comment = self.pending_strategy_comments.pop(symbol, "")
                    msg = (f"🟢 Открыта {side_raw.upper()}-позиция {symbol}: "
                        f"объём {new_size} @ {avg_price}")
                    if comment:
                        msg += f"\nКомментарий: <i>{comment}</i>"
                    asyncio.create_task(self.notify_user(msg))

                    # подтверждение открытия → снимаем из pending
                    self.pending_orders.discard(symbol)
                    continue

                # 6) Закрытие позиции
                if prev is not None and new_size == 0:
                    self.pending_strategy_comments.pop(symbol, None)
                    logger.info(f"[PositionStream] Закрытие позиции {symbol}, "
                                f"PnL={position.get('unrealisedPnl')}")
                    self.closed_positions[symbol] = {
                        **prev,
                        "closed_pnl":  position.get("unrealisedPnl"),
                        "closed_time": position.get("updatedTime")
                    }
                    self.ws_closed_symbols.add(symbol)
                    self.ws_opened_symbols.discard(symbol)

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
                    asyncio.create_task(self.notify_user(
                        f"⏹️ Закрыта {prev['side'].upper()}‑позиция {symbol}: "
                        f"объём {prev['volume']} @ {prev['avg_price']}"
                    ))

                    # Удаляем из активных, сбрасываем флаги
                    del self.open_positions[symbol]
                    self.averaged_symbols.discard(symbol)
                    self.pending_orders.discard(symbol)
                    continue

                # 7) Обновление объёма существующей позиции
                if prev is not None and new_size > 0 and new_size != prev.get("volume"):
                    logger.info(f"[PositionStream] Обновление объёма {symbol}: "
                                f"{prev['volume']} → {new_size}")
                    self.open_positions[symbol]["volume"] = new_size
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
    async def update_open_positions(self) -> None:
        """
        Refresh open_positions with a one‑shot REST snapshot from Bybit.
        """
        try:
            resp = await asyncio.to_thread(
                lambda: self.session.get_positions(category="linear", settleCoin="USDT")
            )
            self.open_positions.clear()
            for pos in resp.get("result", {}).get("list", []):
                size = safe_to_float(pos.get("size", 0))
                if size <= 0:
                    continue
                symbol = pos["symbol"]
                self.open_positions[symbol] = {
                    "avg_price": safe_to_float(pos.get("avgPrice") or pos.get("entryPrice")),
                    "side": pos.get("side", ""),
                    "pos_idx": pos.get("positionIdx", 1),
                    "volume": size,
                    "amount": safe_to_float(pos.get("positionValue")),
                    "leverage": safe_to_float(pos.get("leverage", 0)),
                }
            logger.info("[update_open_positions] user %s: %d open positions loaded",
                        self.user_id, len(self.open_positions))
        except Exception as e:
            logger.warning("[update_open_positions] user %s REST error: %s",
                           self.user_id, e)

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


    async def handle_liquidation(self, msg):
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
                    # keep only last 150 s
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
                        try:
                            resp = await asyncio.to_thread(lambda: self.session.place_order(
                                category="linear",
                                symbol=symbol,
                                side=order_side,
                                orderType="Market",
                                qty=str(qty),
                                positionIdx=self.position_idx,
                                timeInForce="GoodTillCancel"
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
                recent = self.shared_ws.candles_data.get(symbol, [])
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
            if pnl_pct > 0:
                logger.info(f"[evaluate_position] {symbol}: last_price={last_price}, avg_price={avg_price}, pnl={pnl}, pnl_pct={pnl_pct}%")

            # if pnl > 0:
            #     logger.info(f"[evaluate_position] {symbol}: last_price={last_price}, avg_price={avg_price}, pnl={pnl}, pnl_pct={pnl_pct}%")

            #     # Ensure averaging is done on the full position
            #     volume_to_add = size  # Default to full position size
            #     current_volume = safe_to_float(self.open_positions.get(symbol, {}).get("volume", 0))
            #     if size < current_volume:
            #         volume_to_add = current_volume  # Correct averaging size to match total open volume

            #     current_volume = safe_to_float(self.open_positions.get(symbol, {}).get("volume", 0))
            #     if volume_to_add + current_volume > self.max_allowed_volume:
            #         logger.warning(f"[evaluate_position] Skipping {symbol}: attempted volume %.2f exceeds max %.2f",
            #                     volume_to_add + current_volume, self.max_allowed_volume)
            #         return
            #     try:
            #         # use original side casing for order submission
            #         orig_side = data.get("side", "Buy")
            #         if symbol in self.averaged_symbols:
            #             logger.info(f"[evaluate_position] Пропуск усреднения: {symbol} уже усреднён")
            #             return

            #         # сразу помечаем как усреднённый, чтобы избежать гонки
            #         self.averaged_symbols.add(symbol)

            #         if self.mode == "real":
            #             await self.place_order_ws(symbol, orig_side, volume_to_add, position_idx=pos_idx)
            #         else:
            #             resp = await asyncio.to_thread(lambda: self.session.place_order(
            #                 category="linear",
            #                 symbol=symbol,
            #                 side=orig_side,
            #                 orderType="Market",
            #                 qty=str(volume_to_add),
            #                 timeInForce="GoodTillCancel",
            #                 positionIdx=pos_idx
            #             ))
            #             if resp.get("retCode", 0) != 0:
            #                 raise InvalidRequestError(resp.get("retMsg", "order rejected"))
            #         logger.info(f"[evaluate_position] Averaging executed for {symbol}: added volume {volume_to_add}")
            #         # обновляем внутренний объём
            #         self.open_positions[symbol]["volume"] += volume_to_add
            #     except RuntimeError as e:
            #         msg = str(e)
            #         self.averaged_symbols.discard(symbol)

            #         # Handle insufficient balance error from Bybit
            #         if "ab not enough for new order" in msg:
            #             logger.warning(f"[evaluate_position] averaging skipped for {symbol}: insufficient balance ({msg})")
            #             return
            #         # Re-raise or log other runtime errors
            #         logger.error(f"[evaluate_position] averaging failed for {symbol}: {e}", exc_info=True)
            #     except Exception as e:
            #         logger.error(f"[evaluate_position] averaging failed for {symbol}: {e}", exc_info=True)

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
        if symbol in self.open_positions:
            pos_data = self.open_positions[symbol]
            # update stored mark price
            pos_data['markPrice'] = last_price
            # ── keep a rolling history of Open Interest so Golden Setup sees ΔOI ──
            oi_val = self.shared_ws.latest_open_interest.get(symbol)
            if oi_val is not None:
                hist = self.shared_ws.oi_history.setdefault(symbol, deque(maxlen=500))
                hist.append(float(oi_val))
            # build a minimal position dict for evaluation
            position = {
                "symbol": symbol,
                "size": str(pos_data.get("volume", 0)),
                "side": pos_data.get("side", "")
            }
            await self.evaluate_position(position)
    
    async def cleanup_pending_loop(self):
        while True:
            now = time.time()
            to_remove = [s for s, ts in self.pending_timestamps.items() if now - ts > 60]
            for symbol in to_remove:
                if symbol in self.pending_orders and symbol not in self.open_positions:
                    self.pending_orders.discard(symbol)
                    self.pending_strategy_comments.pop(symbol, None)
                    logger.info("[pending_cleanup] Удалён сигнал по %s — истёк таймаут", symbol)
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
            # Skip symbols manually closed to prevent re-entry
            if symbol in self.closed_positions:
                return
            # 1. Пропускаем недавно провалившиеся символы
            # ensure we know the correct qtyStep / minQty
            await self.ensure_symbol_meta(symbol)
            if symbol in self.failed_orders and time.time() - self.failed_orders[symbol] < 600:
                return
            # 2. Не трогаем уже открытые или ожидающие открытие
            if symbol in self.open_positions or symbol in self.pending_orders:
                return

            # ---- SQUEEZE STRATEGY (общая для всех режимов) ----------------
            recent_1m = self.shared_ws.candles_data.get(symbol, [])
            if len(recent_1m) >= 6:                     # 5 завершённых баров (≈5 мин)
                old_close_s = safe_to_float(recent_1m[-6]["closePrice"])
                new_close_s = safe_to_float(recent_1m[-1]["closePrice"])
                if old_close_s > 0:
                    pct_5m = (new_close_s - old_close_s) / old_close_s * 100
                    if pct_5m >= self.squeeze_threshold_pct:                   # порог 3 %
                        logger.info("[SQUEEZE] %s ΔP=%.2f%% за 5 мин — открываем SHORT",
                                    symbol, pct_5m)

                        # лимит позиций / экспозиции
                        usd_size = min(self.POSITION_VOLUME, self.max_allowed_volume)
                        total_expo = await self.get_total_open_volume()
                        remaining_cap = max(0.0, self.MAX_TOTAL_VOLUME - total_expo)
                        usd_size = min(self.POSITION_VOLUME, remaining_cap)
                        if usd_size <= 0:
                            logger.info("[SQUEEZE] skip %s: нет свободного лимита (expo %.0f / %.0f)",
                                        symbol, total_expo, self.MAX_TOTAL_VOLUME)
                            return
                        if total_expo + usd_size > self.MAX_TOTAL_VOLUME:
                            logger.info("[SQUEEZE] skip %s: exposure %.0f + %.0f > %.0f",
                                        symbol, total_expo, usd_size, self.MAX_TOTAL_VOLUME)
                            return

                        await self.ensure_symbol_meta(symbol)
                        step    = self.qty_step_map.get(symbol, 0.001)
                        min_qty = self.min_qty_map.get(symbol, step)
                        qty     = max(math.floor((usd_size / new_close_s) / step) * step, min_qty)
                        if qty <= 0:
                            return

                        # помечаем комментарий ДО отправки
                        self.pending_strategy_comments[symbol] = "Сквиз 3%/5m"
                        self.pending_orders.add(symbol)
                        self.pending_timestamps[symbol] = time.time()

                        try:
                            if self.mode == "real":
                                await self.place_order_ws(symbol, "Sell", qty, position_idx=2)
                            else:
                                resp = await asyncio.to_thread(lambda: self.session.place_order(
                                    category="linear",
                                    symbol=symbol,
                                    side="Sell",
                                    orderType="Market",
                                    qty=str(qty),
                                    timeInForce="GoodTillCancel",
                                    positionIdx=2
                                ))
                                if resp.get("retCode", 0) != 0:
                                    raise InvalidRequestError(resp.get("retMsg", "order rejected"))
                            logger.info("[SQUEEZE] %s SHORT qty=%.6f opened", symbol, qty)
                        except Exception as e:
                            logger.warning("[SQUEEZE] order failed for %s: %s", symbol, e)
                            self.pending_orders.discard(symbol)
                        return  # не продолжаем к другим стратегиям


            # 3. Берём историю свечей/объёма/OI
            # ---- LIQUIDATION INFO ------------------------------------------------
            liq_data = self.shared_ws.latest_liquidation.get(symbol, {})
            liq_val  = safe_to_float(liq_data.get("value", 0))
            liq_side = liq_data.get("side")          # 'Buy' | 'Sell' | None

            # --- strategy switches ---------------------------------
            mode = getattr(self, "strategy_mode", "full")
            golden_enabled = mode in ("golden_only", "full")
            liq_enabled    = mode in ("liquidation_only", "full")


            # 4‑A. Торговля от крупной ликвидации (v2: динамический порог + фильтры)
            # TradingBot не имеет собственного метода get_liq_threshold ─ берём из shared_ws
            threshold = self.shared_ws.get_liq_threshold(symbol, 5000)
            avg_vol_30m = self.shared_ws.get_avg_volume(symbol, 30)
            delta_oi = self.shared_ws.get_delta_oi(symbol)
            cooldown_ok = self.shared_ws.check_liq_cooldown(symbol)

            candles = self.shared_ws.candles_data.get(symbol, [])
            candle_ok = False
            if candles:
                candle_ok = self.shared_ws.is_volatile_spike(symbol, candles[-1])
            # --- текущий объём (последняя 5‑мин свеча) ---
            if candles:
                volume_now = safe_to_float(candles[-1].get("volume")
                                           or candles[-1].get("turnover", 0))
            else:
                # fallback на историю объёмов, если свечи ещё не прогреты
                vol_hist = self.shared_ws.volume_history.get(symbol, [])
                volume_now = safe_to_float(vol_hist[-1]) if vol_hist else 0.0

            funding_ok = (
                self.shared_ws.funding_cool(symbol)
                if self.shared_ws else True
            )

            # ---- 3-баровое среднее ΔV / ΔOI ------------------------
            vol_hist = list(self.shared_ws.volume_history.get(symbol, []))
            oi_hist  = list(self.shared_ws.oi_history.get(symbol, []))

            # Initialize mean_dvol and mean_doi with default values
            mean_dvol = 0.0
            mean_doi = 0.0

            # быстрые deltas «сейчас – 1 мин» (для fallback)
            volume_change = 0.0
            oi_change     = 0.0
            if len(vol_hist) >= 2:
                volume_change = (vol_hist[-1] - vol_hist[-2]) / max(1e-8, vol_hist[-2]) * 100
           
            if len(oi_hist) >= 2:
                oi_change = (oi_hist[-1] - oi_hist[-2]) / max(1e-8, oi_hist[-2]) * 100

            # Always assign fallback mean values even if both hist lists are short
            mean_dvol = volume_change
            mean_doi  = oi_change

            # ---- liquidation analytics filter ----
            liq_info = self.shared_ws.latest_liquidation.get(symbol, {}) if self.shared_ws else {}
            liq_val  = liq_info.get("value", 0.0)
            liq_side = liq_info.get("side", "")
            liq_ts   = liq_info.get("ts", 0.0)

            # ликвидация считается «свежей», если была ≤ 60 сек назад
            liq_recent = (time.time() - liq_ts) <= 60 and liq_val >= threshold

            passed_filters = (
                liq_enabled
                and liq_val >= threshold
                and candle_ok
                and mean_dvol > 0
                and mean_doi  > 0
                and delta_oi is not None and delta_oi <= -0.003
                and volume_now >= VOLUME_COEF * avg_vol_30m
                and funding_ok
                and cooldown_ok
                and liq_side in ("Buy", "Sell")
            )
            if passed_filters:
                            opposite = "Buy" if liq_side == "Sell" else "Sell"
                            if symbol not in self.open_positions and symbol not in self.pending_orders:
                                self.pending_strategy_comments[symbol] = "От ликвидаций"
                                # ---- лимит общего экспозиционного объёма -------------------
                                total_expo = await self.get_total_open_volume()
                                potential  = self.POSITION_VOLUME    # в USDT, т.к. qty ещё не посчитан
                                if total_expo + potential > self.MAX_TOTAL_VOLUME:
                                    logger.warning("[LiqTrade] skip %s: exposure %.0f + %.0f > %.0f",
                                                symbol, total_expo, potential, self.MAX_TOTAL_VOLUME)
                                    return
                                
                                # Берём последнюю цену
                                close_price = safe_to_float(
                                    self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
                                ) or safe_to_float(
                                    self.shared_ws.candles_data.get(symbol, [])[-1]["closePrice"]
                                )
                                if close_price > 0:
                                    step = self.qty_step_map.get(symbol, 0.001)
                                    # вычисляем количество знаков после запятой у qtyStep
                                    step_str = str(step)
                                    dec = len(step_str.split(".")[1].rstrip("0")) if "." in step_str else 0
                                    qty_raw = safe_to_float(self.POSITION_VOLUME) / close_price
                                    qty = math.floor(qty_raw / step) * step                 # округление к шагу
                                    qty = round(qty, dec)                                   # убираем «166.10000002»
                                    qty_str = f"{qty:.{dec}f}"
                                    pos_idx = 1 if opposite == "Buy" else 2                 # индекс позиции
                                    try:
                                        logger.info("[LiqTrade] %s крупная %s-ликв %.0f USDT → %s %.3f",
                                                    symbol, liq_side, liq_val, opposite, qty)
                                        
                                        if mean_dvol <= 0 or mean_doi <= 0:
                                            logger.info("[Golden] %s отменён: ΔV/ΔOI стали отрицательны "
                                                        "(dV=%.2f  dOI=%.3f)", symbol, mean_dvol, mean_doi)
                                            return

                                        if self.mode == "real":
                                            if self.mode == "real":
                                                await self.place_order_ws(symbol, opposite, qty,
                                                                        position_idx=pos_idx)
                                        else:
                                            resp = await asyncio.to_thread(lambda: self.session.place_order(
                                                category="linear",
                                                symbol=symbol,
                                                side=opposite,
                                                orderType="Market",
                                                qty=qty_str,
                                                timeInForce="GoodTillCancel",
                                                positionIdx=pos_idx
                                            ))
                                            if resp.get("retCode", 0) != 0:
                                                raise InvalidRequestError(resp.get("retMsg", "order rejected"))
                                        # --- помечаем ордер ДО отправки, чтобы WS успел его прочитать ---
                                        self.pending_strategy_comments[symbol] = "Чистый Золотой Сетап"
                                        self.pending_orders.add(symbol)
                                        self.pending_timestamps[symbol] = time.time()
                                    except Exception as e:
                                        logger.warning("[LiqTrade] order failed for %s: %s", symbol, e)
                            return  # после торговли от ликвидации — пропускаем golden‑setup дальше

            if not golden_enabled:
                return     # режим "только ликвидации" – классический GS пропускаем

            recent = self.shared_ws.candles_data.get(symbol, [])
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
                len(self.shared_ws.volume_history[symbol]) <= period_iters or
                len(self.shared_ws.oi_history[symbol])     <= period_iters or
                len(self.shared_ws.cvd_history[symbol])    <= period_iters):
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

            old_vol = safe_to_float(self.shared_ws.volume_history[symbol][-1 - period_iters])
            new_vol = safe_to_float(self.shared_ws.volume_history[symbol][-1])
            volume_change_pct = (
                (new_vol - old_vol) / old_vol * 100.0
                if old_vol != 0 else 0.0
            )

            # --- use Decimal for higher precision ---
            old_oi = Decimal(str(self.shared_ws.oi_history[symbol][-1 - period_iters]))
            new_oi = Decimal(str(self.shared_ws.oi_history[symbol][-1]))
            oi_change_pct = (
                (new_oi - old_oi) / old_oi * Decimal("100")
                if old_oi != 0 else Decimal("0")
            )

            # --- CVD % change over the same period ----------------
            old_cvd = safe_to_float(self.shared_ws.cvd_history[symbol][-1 - period_iters])
            new_cvd = safe_to_float(self.shared_ws.cvd_history[symbol][-1])
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
                    (Decimal(str(self.shared_ws.volume_history[symbol][-1]))
                    - Decimal(str(self.shared_ws.volume_history[symbol][-1 - sp])))
                    / Decimal(str(self.shared_ws.volume_history[symbol][-1 - sp]))
                    * Decimal("100")
                    if self.shared_ws.volume_history[symbol][-1 - sp]
                    else Decimal("0")
                )
                oichg = (
                    (Decimal(str(self.shared_ws.oi_history[symbol][-1]))
                    - Decimal(str(self.shared_ws.oi_history[symbol][-1 - sp])))
                    / Decimal(str(self.shared_ws.oi_history[symbol][-1 - sp]))
                    * Decimal("100")
                    if self.shared_ws.oi_history[symbol][-1 - sp]
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
                    pchgb = (Decimal(str(newb["closePrice"])) - Decimal(str(oldb["closePrice"]))) / Decimal(str(oldb["closePrice"])) * Decimal("100") if oldb["closePrice"] else Decimal("0")
                    volb = (
                        (Decimal(str(self.shared_ws.volume_history[symbol][-1]))
                        - Decimal(str(self.shared_ws.volume_history[symbol][-1 - lp])))
                        / Decimal(str(self.shared_ws.volume_history[symbol][-1 - lp]))
                        * Decimal("100")
                        if self.shared_ws.volume_history[symbol][-1 - lp]
                        else Decimal("0")
                    )
                    oib = (
                        (Decimal(str(self.shared_ws.oi_history[symbol][-1]))
                        - Decimal(str(self.shared_ws.oi_history[symbol][-1 - lp])))
                        / Decimal(str(self.shared_ws.oi_history[symbol][-1 - lp]))
                        * Decimal("100")
                        if self.shared_ws.oi_history[symbol][-1 - lp]
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
            _append_snapshot({
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
                            timeInForce="GoodTillCancel",
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
            "timeInForce": "GoodTillCancel",
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


    async def set_trailing_stop(
        self,
        symbol: str,
        avg_price: float,
        pnl_pct: float,
        side: str
    ):
        """
        Устанавливает/обновляет трейлинг‑стоп, работая только с float‑значениями.
        """
        try:
            logger.info(
                "[trailing_stop] Попытка установки стопа для %s | ROI=%.5f%%",
                symbol, pnl_pct
            )

            data = self.open_positions.get(symbol)
            if not data:
                logger.warning("[trailing_stop] Позиция %s не найдена", symbol)
                return

            volume = safe_to_float(data.get("volume", 0))
            if volume <= 0:
                logger.warning(
                    "[trailing_stop] Пропуск установки стопа для %s: нулевая позиция",
                    symbol
                )
                return
            pos_idx = data["pos_idx"]

            base_trail = self.trailing_gap_pct
            reduction = 0.0
            oi = self.shared_ws.latest_open_interest.get(symbol, 0.0)

            if oi > 1000:
                reduction += 0.5

            recent = self.shared_ws.candles_data.get(symbol, [])
            if recent:
                last_price = safe_to_float(recent[-1]["closePrice"])
                if abs(last_price - avg_price) / avg_price * 100 > 1:
                    reduction += 0.5
            else:
                # fallback на тикер, если свечей нет
                last_price = safe_to_float(
                    self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
                )

            final_trail = max(base_trail - reduction, 0.5)
            stop_pct = round(pnl_pct - final_trail, 2)

            if symbol not in self.open_positions:
                logger.warning("[trailing_stop] Позиция %s уже закрыта", symbol)
                return

            if side.lower() == "buy":
                raw_price = avg_price * (1 + stop_pct / 1000)
            elif side.lower() == "sell":
                raw_price = avg_price * (1 - stop_pct / 1000)
            else:
                logger.error("[trailing_stop] Unknown side %s", side)
                return

            tick = float(DEC_TICK) if not isinstance(DEC_TICK, (int, float)) else DEC_TICK
            stop_price = round(math.floor(raw_price / tick) * tick, 6)

            logger.info(
                "[trailing_stop] вычислено: base_trail=%.2f  reduction=%.2f  "
                "final_trail=%.2f  stop_pct=%.2f  stop_price=%.6f",
                base_trail, reduction, final_trail, stop_pct, stop_price
            )

            # Проверка: если стоп уже такой — выходим
            prev = self.last_stop_price.get(symbol)
            if prev is not None and abs(prev - stop_price) < 1e-8:
                logger.debug("[trailing_stop] %s stop unchanged (%.8f) — skipping", symbol, stop_price)
                return False

            # Проверка: не ухудшаем
            if prev is not None:
                if (side.lower() == "buy" and stop_price < prev) or (side.lower() == "sell" and stop_price > prev):
                    logger.info("[trailing_stop] Новый стоп хуже предыдущего — не обновляем")
                    return

            # ── Отправка запроса на биржу ──
            try:
                logger.info("[trailing_stop] Отправка запроса на установку стопа: %.6f", stop_price)
                resp = await asyncio.to_thread(
                    lambda: self.session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        positionIdx=pos_idx,
                        stopLoss=str(stop_price),
                        triggerBy="LastPrice",
                        timeInForce="GoodTillCancel"
                    )
                )
                ret = resp.get("retCode", resp.get("ret_code", 0))
                if ret in (0, 34040):
                    self.last_stop_price[symbol] = stop_price
                    if ret == 34040:
                        logger.info("[trailing_stop] %s already had stop %.8f — treated as OK", symbol, stop_price)
                    else:
                        logger.info("[trailing_stop] Стоп установлен: %s | stopPrice=%.6f | pct=%.2f%%",
                                    symbol, stop_price, stop_pct)
                    await self.log_trade(
                        symbol=symbol,
                        side=side,
                        avg_price=avg_price,
                        volume=data["volume"],
                        open_interest=oi,
                        action="stoploss",
                        result="trailingstop"
                    )
                    return True
                else:
                    logger.error(
                        "[trailing_stop] Ошибка при установке стопа для %s: retCode=%s, resp=%s",
                        symbol, ret, resp
                    )

            except InvalidRequestError as e:
                msg = str(e)
                if "not modified" in msg or "zero position" in msg:
                    # Only log error, do not notify user.
                    return
                logger.error("[trailing_stop] Ошибка при установке стопа для %s: %s", symbol, msg, exc_info=True)

            except Exception as e:
                logger.error("[trailing_stop] Непредвиденная ошибка при установке стопа для %s: %s",
                            symbol, e, exc_info=True)

        except Exception as e:
            logger.error("[trailing_stop] Critical error: %s", e, exc_info=True)

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


    async def stop(self):
        logger.info(f"[User {self.user_id}] Остановка бота")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.ws_private:
            self.ws_private.exit()
        logger.info(f"[User {self.user_id}] Полная остановка")


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
            "period_iters": 2,
            "price_change": 0.8,      # +0.20 % price rise
            "volume_change": 150,      # +50 % volume surge
            "oi_change": 0.5,         # +0.40 % OI rise
        },
        "Sell": {
            "period_iters": 2,
            "price_change": 1,      # −0.50 % price drop
            "volume_change": 100,      # +30 % volume surge
            "oi_change": 1,         # +0.80 % OI rise
        }
        # "Sell2": {                # альтернативный шорт‑сетап  «либо‑либо»
        # "period_iters": 4,    # 4 последних свечи
        # "price_change": 0.02,# падение цены ≥ 0.075 %
        # "volume_change": -50, # падение объёма ≥ 80 %
        # "oi_change": -0.01,      # падение OI ≥ 1 %
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


async def run_all():
    users = load_users_from_json("user_state.json")
    if not users:
        print("❌ Нет активных пользователей для запуска.")
        return
    
    golden_param_store = load_golden_params()
    bots = [TradingBot(user_data=u, shared_ws=None, golden_param_store=golden_param_store) for u in users]
    #symbols = await bots[0].get_selected_symbols() if bots else []
    initial_symbols = ["BTCUSDT", "ETHUSDT"]           # минимальный старт
    shared_ws = PublicWebSocketManager(symbols=initial_symbols)
    
    await shared_ws.backfill_history()
    public_ws_task = asyncio.create_task(shared_ws.start())
    
    for bot in bots:
        bot.shared_ws = shared_ws
        shared_ws.position_handlers.append(bot)  # register for ticker-based evaluate_position
    
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
    Log wallet state every 5 minutes into wallet_state.log.
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
        await asyncio.sleep(300)  # 5 minutes            
# --------------------------------------------------------------------
# Alias for TradingBot so it can be referenced inside PublicWebSocketManager

    # async def cleanup_pending_loop(self):
    #     while True:
    #         now = time.time()
    #         to_remove = [s for s, ts in self.pending_timestamps.items() if now - ts > 60]
    #         for symbol in to_remove:
    #             if symbol in self.pending_orders and symbol not in self.open_positions:
    #                 self.pending_orders.discard(symbol)
    #                 self.pending_strategy_comments.pop(symbol, None)
    #                 logger.info("[pending_cleanup] Удалён сигнал по %s — истёк таймаут", symbol)
    #             self.pending_timestamps.pop(symbol, None)
    #         await asyncio.sleep(10)