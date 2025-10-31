#!/usr/bin/env python3

# ----------------- НАЧАЛО ФАЙЛА MultiuserBot_FINAL_COMPLETE.py -----------------

# ---------------------- 1. ИМПОРТЫ ----------------------
import os
import sys
import faulthandler
import signal
import functools
import asyncio
import datetime as dt
import json
import logging
import time
import hmac
import hashlib
import csv
import pickle
import math
import random
import warnings
import tempfile
from pathlib import Path
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

# --- Библиотеки для работы с биржами и API ---
import websockets
from pybit.unified_trading import WebSocket, HTTP
from pybit.exceptions import InvalidRequestError

# --- Библиотеки для анализа данных и ML ---
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import InconsistentVersionWarning

# --- ML Фреймворк (только MLX) ---
import mlx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim

# --- Компоненты Telegram-бота ---
import aiogram
from aiogram import types
from aiogram.enums import ParseMode
from aiogram.filters import Command
from telegram_fsm_v12 import dp, router, router_admin, bot as telegram_bot

# --- Асинхронные утилиты ---
import uvloop
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI

# --- Установка асинхронного цикла ---
uvloop.install()

# --- Совместимость numpy >= 2.0 для pandas_ta ---
if not hasattr(np, "NaN"):
    np.NaN = np.nan

# ---------------------- 2. НАСТРОЙКА ЛОГГЕРА И ГЛОБАЛЬНЫХ КОНСТАНТ ----------------------

# --- Оптимизация производительности ---
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
faulthandler.enable(file=sys.stderr, all_threads=True)

# --- Настройка логгера ---
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Ротируемый файл для основных логов
rotating_handler = logging.FileHandler('bot.log', mode='a')
rotating_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
root_logger.addHandler(rotating_handler)

# Вывод в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
root_logger.addHandler(console_handler)

# --- Глобальные переменные и константы ---
ADMIN_IDS = {36972091}
GLOBAL_BOTS: list = []

# --- Пути к файлам ---
MODEL_PATH_MLX = "golden_model_mlx.safetensors"
SCALER_PATH = "scaler.pkl"
LIQUIDATIONS_CSV_PATH = "liquidations.csv"
OPEN_POS_JSON = "open_positions.json"
WALLET_JSON = "wallet_state.json"

# --- Параметры стратегий и ML ---
ML_GATE_MIN_SCORE = 0.015
SQUEEZE_THRESHOLD_PCT = 4.0
DEFAULT_SQUEEZE_POWER_MIN = 8.0
LISTING_AGE_MIN_MINUTES = 1400
EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT"}


# ---------------------- 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------------

def safe_to_float(val: Any) -> float:
    """Безопасное преобразование в float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def _safe_load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    """Безопасно загружает скейлер, созданный в любой версии sklearn."""
    if not os.path.exists(path):
        return StandardScaler()
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("error", category=InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except (InconsistentVersionWarning, Exception):
            logger.warning(f"[Scaler] {path} несовместим. Создаю новый StandardScaler().")
            return StandardScaler()

def log_for_finetune(prompt: str, pnl_pct: float, source: str):
    """Записывает данные для дообучения AI модели."""
    log_file = Path("finetune_log.csv")
    is_new = not log_file.exists()
    try:
        with log_file.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            if is_new:
                writer.writerow(["timestamp", "source", "pnl_pct", "prompt"])
            writer.writerow([dt.datetime.utcnow().isoformat(), source, pnl_pct, prompt])
    except Exception as e:
        logger.error(f"[FineTuneLog] Ошибка записи лога: {e}")

def _atomic_json_write(path: str, data: Any):
    """Атомарная запись JSON через временный файл."""
    dirname = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirname, prefix=".tmp_", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def compute_pct(candles_deque: deque, minutes: int) -> float:
    """Рассчитывает процентное изменение цены."""
    data = list(candles_deque)
    if len(data) < minutes + 1: return 0.0
    old_close = safe_to_float(data[-minutes - 1].get("closePrice", 0))
    new_close = safe_to_float(data[-1].get("closePrice", 0))
    if old_close <= 0: return 0.0
    return (new_close - old_close) / old_close * 100.0

def sum_last_vol(candles_deque: deque, minutes: int) -> float:
    """Суммирует объем за последние N свечей."""
    data = list(candles_deque)[-minutes:]
    return sum(safe_to_float(c.get("volume", 0)) for c in data)

def load_users_from_json(json_path: str = "user_state.json") -> list:
    """Загружает конфигурации активных пользователей."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)
    result = []
    for uid, data in all_users.items():
        if not data.get("banned") and data.get("api_key") and data.get("api_secret"):
            result.append({
                "user_id": uid,
                "api_key": data.get("api_key"),
                "api_secret": data.get("api_secret"),
                "openai_api_key": data.get("openai_api_key"),
                "ai_provider": data.get("ai_provider", "ollama"),
                "strategy_mode": data.get("strategy_mode", "full"),
                "volume": data.get("volume", 1000),
                "max_total_volume": data.get("max_total_volume", 5000),
                "mode": data.get("mode", "real")
            })
    return result

def load_golden_params(csv_path: str = "golden_params.csv") -> dict:
    """Загружает параметры для стратегии Golden Setup."""
    default_params = {
        "Buy": {"price_change": 1.7, "volume_change": 200, "oi_change": 1.5},
        "Sell": {"price_change": 1.8, "volume_change": 200, "oi_change": 1.2}
    }
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            overrides = {}
            for _, row in df.iterrows():
                key = (row["symbol"], row["side"])
                overrides[key] = {k: safe_to_float(row[k]) for k in default_params["Buy"].keys()}
            logger.info(f"[GoldenParams] Загружено {len(overrides)} пользовательских параметров.")
            return {**default_params, **overrides}
        except Exception as e:
            logger.error(f"[GoldenParams] Ошибка загрузки CSV: {e}")
    return default_params

async def get_and_filter_symbols(http_session: HTTP, min_turnover_24h=5_000_000) -> set:
    """Получает все тикеры с биржи и фильтрует их по обороту."""
    try:
        logger.info("[SymbolSelection] Получение списка ликвидных символов...")
        response = await asyncio.to_thread(lambda: http_session.get_tickers(category="linear"))
        if response.get("retCode") != 0:
            logger.error(f"Ошибка API при получении тикеров: {response.get('retMsg')}")
            return {"BTCUSDT", "ETHUSDT"}

        all_tickers = response.get("result", {}).get("list", [])
        desired_symbols = {
            t['symbol'] for t in all_tickers
            if safe_to_float(t.get("turnover24h")) > min_turnover_24h
            and 'USDT' in t['symbol'] and t['symbol'] not in EXCLUDED_SYMBOLS
        }
        desired_symbols.update({"BTCUSDT", "ETHUSDT"})
        logger.info(f"Найдено {len(desired_symbols)} ликвидных символов.")
        return desired_symbols
    except Exception as e:
        logger.error(f"Критическая ошибка при получении тикеров: {e}", exc_info=True)
        return {"BTCUSDT", "ETHUSDT"}

# ---------------------- 4. БЛОК ML-КОМПОНЕНТОВ (MLX ONLY) ----------------------

FEATURE_KEYS = [
    "price", "pct1m", "pct5m", "pct15m", "vol1m", "vol5m", "vol15m", "OI_now", "dOI1m", "dOI5m",
    "spread_pct", "sigma5m", "CVD1m", "CVD5m", "rsi14", "sma50", "ema20", "atr14", "bb_width",
    "supertrend", "cci20", "macd", "macd_signal", "avgVol30m", "avgOI30m", "deltaCVD30m", "adx14",
    "hour_of_day", "day_of_week", "month_of_year"
]
INPUT_DIM = len(FEATURE_KEYS)

class GoldenNetMLX(mlx_nn.Module):
    """Архитектура модели на MLX."""
    def __init__(self, input_size: int = INPUT_DIM, hidden_size: int = 64):
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
        return self.fc3(x)

class MLXInferencer:
    """Класс для инференса (предсказаний) с помощью MLX модели."""
    def __init__(self, model_path=MODEL_PATH_MLX, scaler_path=SCALER_PATH):
        self.model = None
        self.scaler = _safe_load_scaler(scaler_path)
        if Path(model_path).exists():
            try:
                self.model = GoldenNetMLX()
                self.model.load_weights(model_path)
                self.model.eval()
                logger.info(f"[MLX] Модель из {model_path} успешно загружена.")
            except Exception as e:
                logger.error(f"[MLX] Ошибка загрузки модели: {e}", exc_info=True)

    def infer(self, features: np.ndarray) -> np.ndarray:
        if not self.model: return np.array([[0.0]])
        if self.scaler:
            features = self.scaler.transform(features)
        return np.array(self.model(mlx.array(features)))

def train_golden_model_mlx(training_data: list, num_epochs: int = 25) -> (GoldenNetMLX, StandardScaler):
    """Функция обучения MLX модели."""
    logger.info("[MLX] Запуск обучения...")
    feats = np.asarray([d["features"] for d in training_data], dtype=np.float32)
    targ = np.asarray([d["target"] for d in training_data], dtype=np.float32)
    mask = ~np.isnan(feats).any(1) & ~np.isinf(feats).any(1)
    feats, targ = feats[mask], targ[mask]
    if feats.size == 0: raise ValueError("Нет валидных сэмплов для обучения.")

    scaler = StandardScaler().fit(feats)
    feats_scaled = scaler.transform(feats)
    targ = np.clip(targ, -3.0, 3.0).reshape(-1, 1)

    model = GoldenNetMLX()
    optimizer = mlx_optim.Adam(learning_rate=1e-3)
    loss_and_grad_fn = mlx_nn.value_and_grad(model, lambda m, x, y: mlx_nn.losses.mse_loss(m(x), y).mean())

    for epoch in range(num_epochs):
        loss, grads = loss_and_grad_fn(model, mlx.array(feats_scaled), mlx.array(targ))
        optimizer.update(model, grads)
        mlx.eval(model.parameters(), optimizer.state)
        if (epoch + 1) % 5 == 0: logger.info(f"Epoch {epoch+1} [MLX] – Loss: {loss.item():.5f}")

    return model, scaler

# ---------------------- 5. КЛАССЫ БОТА ----------------------

class PublicWebSocketManager:
    __slots__ = (
        "symbols", "interval", "ws", "candles_data", "ticker_data", "latest_open_interest",
        "loop", "volume_history", "oi_history", "cvd_history", "_last_saved_time", 
        "position_handlers", "_history_file", "_save_task", "latest_liquidation", 
        "last_liq_trade_time", "http_session", "ready_event"
    )

    def __init__(self, symbols, interval="1"):
        self.symbols = symbols
        self.interval = interval
        self.ws = None
        self.candles_data = defaultdict(lambda: deque(maxlen=1000))
        self.ticker_data = {}
        self.latest_open_interest = {}
        self.ready_event = asyncio.Event()
        self.loop = asyncio.get_running_loop()
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        self.oi_history = defaultdict(lambda: deque(maxlen=1000))
        self.cvd_history = defaultdict(lambda: deque(maxlen=1000))
        self._last_saved_time = {}
        self.position_handlers = []
        self.latest_liquidation = {}
        self.last_liq_trade_time = {}
        self.http_session = HTTP(testnet=False)
        self._history_file = 'history.pkl'
        self._load_local_history()

    def _load_local_history(self):
        try:
            with open(self._history_file, 'rb') as f:
                data = pickle.load(f)
                for sym, rows in data.get('candles', {}).items(): self.candles_data[sym] = deque(rows, maxlen=1000)
                for sym, vol in data.get('volume_history', {}).items(): self.volume_history[sym] = deque(vol, maxlen=1000)
                for sym, oi in data.get('oi_history', {}).items(): self.oi_history[sym] = deque(oi, maxlen=1000)
                for sym, cvd in data.get('cvd_history', {}).items(): self.cvd_history[sym] = deque(cvd, maxlen=1000)
            logger.info(f"[History] Загружена история из {self._history_file}")
        except FileNotFoundError:
            logger.info(f"[History] Файл {self._history_file} не найден.")
        except Exception as e:
            logger.error(f"[History] Ошибка загрузки истории: {e}")

    async def start(self):
        self._save_task = asyncio.create_task(self._save_loop())
        # Запускаем задачу динамического обновления символов
        asyncio.create_task(self.manage_symbol_selection_loop())
        
        while True:
            try:
                await self._connect_and_subscribe()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                logger.info("Public WS task cancelled.")
                break
            except Exception as e:
                logger.error(f"[PublicWS] Соединение разорвано: {e}. Переподключение через 5 секунд...")
                if self.ws: self.ws.exit()
                await asyncio.sleep(5)

    async def _connect_and_subscribe(self):
        logger.info(f"[PublicWS] Подключение к WebSocket с {len(self.symbols)} символами...")
        def _on_message(msg):
            if not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.route_message(msg), self.loop)

        self.ws = WebSocket(testnet=False, channel_type="linear")
        
        symbols_list = list(self.symbols)
        chunk_size = 200 
        for i in range(0, len(symbols_list), chunk_size):
            chunk = symbols_list[i:i + chunk_size]
            self.ws.kline_stream(interval=self.interval, symbol=chunk, callback=_on_message)
            self.ws.ticker_stream(symbol=chunk, callback=_on_message)
            self.ws.liquidation_stream(symbol=chunk, callback=_on_message)
            await asyncio.sleep(1)
        
        logger.info("[PublicWS] Соединение и подписки установлены.")
        if not self.ready_event.is_set():
            self.ready_event.set()

    async def manage_symbol_selection_loop(self, interval_sec=3600):
        """Цикл, который раз в час обновляет список символов."""
        await self.ready_event.wait() # Ждем, пока WS установит первое соединение
        while True:
            try:
                logger.info("[SymbolSelection] Начало обновления списка активных символов...")
                new_symbols = await get_and_filter_symbols(self.http_session)
                
                open_pos_symbols = {s for bot in self.position_handlers for s in bot.open_positions.keys()}
                new_symbols.update(open_pos_symbols)

                current_symbols = set(self.symbols)
                if new_symbols and new_symbols != current_symbols:
                    added = new_symbols - current_symbols
                    removed = current_symbols - new_symbols
                    
                    logger.info(f"[SymbolSelection] Обновление подписок: +{len(added)} новых, -{len(removed)} старых.")
                    
                    if added:
                        logger.info(f"Добавляем: {list(added)}")
                        self.ws.subscribe([f"kline.1.{s}" for s in added])
                        self.ws.subscribe([f"tickers.{s}" for s in added])
                        self.ws.subscribe([f"publicTrade.{s}" for s in added]) # liquidation stream
                        await self.backfill_history(list(added))
                    
                    if removed:
                        logger.info(f"Удаляем: {list(removed)}")
                        self.ws.unsubscribe([f"kline.1.{s}" for s in removed])
                        self.ws.unsubscribe([f"tickers.{s}" for s in removed])
                        self.ws.unsubscribe([f"publicTrade.{s}" for s in removed])

                    self.symbols = list(new_symbols)
                else:
                    logger.info("[SymbolSelection] Список символов не изменился.")

            except Exception as e:
                logger.error(f"[SymbolSelection] Ошибка при обновлении списка символов: {e}", exc_info=True)
            
            await asyncio.sleep(interval_sec)

    async def route_message(self, msg):
        topic = msg.get("topic", "")
        if topic.startswith("kline."):
            await self.handle_kline(msg)
        elif topic.startswith("tickers."):
            await self.handle_ticker(msg)
        elif "liquidation" in topic:
            for bot in self.position_handlers:
                if bot.strategy_mode in ("full", "liquidation_only", "liq_squeeze"):
                    asyncio.create_task(bot.handle_liquidation(msg))

    async def handle_kline(self, msg):
        raw = msg.get("data")
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            if not entry.get("confirm", False): continue
            symbol = msg["topic"].split(".")[-1]
            try:
                ts = pd.to_datetime(int(entry["start"]), unit="ms")
            except (ValueError, TypeError):
                continue
            if self._last_saved_time.get(symbol) == ts: continue
            
            row = {
                "startTime": ts, "openPrice": safe_to_float(entry.get("open", 0)),
                "highPrice": safe_to_float(entry.get("high", 0)), "lowPrice": safe_to_float(entry.get("low", 0)),
                "closePrice": safe_to_float(entry.get("close", 0)), "volume": safe_to_float(entry.get("volume", 0)),
            }
            self.candles_data[symbol].append(row)
            self.volume_history[symbol].append(row["volume"])
            oi_val = self.latest_open_interest.get(symbol, 0.0)
            self.oi_history[symbol].append(oi_val)
            delta = row["volume"] if row["closePrice"] >= row["openPrice"] else -row["volume"]
            prev_cvd = self.cvd_history[symbol][-1] if self.cvd_history[symbol] else 0.0
            self.cvd_history[symbol].append(prev_cvd + delta)
            self._last_saved_time[symbol] = ts

    async def handle_ticker(self, msg):
        data = msg.get("data", {})
        entries = data if isinstance(data, list) else [data]
        for ticker in entries:
            symbol = ticker.get("symbol")
            if not symbol: continue
            oi_val = safe_to_float(ticker.get("openInterest", 0))
            self.latest_open_interest[symbol] = oi_val
            self.ticker_data[symbol] = ticker
            
            hist = self.oi_history.setdefault(symbol, deque(maxlen=1000))
            if not hist or hist[-1] != oi_val:
                hist.append(oi_val)
    
    async def backfill_history(self, symbols: list[str]):
        logger.info(f"Начинаем подгрузку истории для {len(symbols)} новых символов...")
        for symbol in symbols:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.http_session.get_kline(category="linear", symbol=symbol, interval=self.interval, limit=1000)
                )
                if resp.get("retCode") == 0:
                    bars = resp.get("result", {}).get("list", [])
                    for entry in reversed(bars):
                        ts = pd.to_datetime(int(entry[0]), unit="ms")
                        row = {
                            "startTime": ts, "openPrice": safe_to_float(entry[1]), "highPrice": safe_to_float(entry[2]),
                            "lowPrice": safe_to_float(entry[3]), "closePrice": safe_to_float(entry[4]),
                            "volume": safe_to_float(entry[5]),
                        }
                        self.candles_data[symbol].appendleft(row)
                        self.volume_history[symbol].appendleft(row["volume"])
                        self.oi_history[symbol].appendleft(0.0)
                        delta = row["volume"] if row["closePrice"] >= row["openPrice"] else -row["volume"]
                        prev_cvd = self.cvd_history[symbol][0] if self.cvd_history[symbol] else 0.0
                        self.cvd_history[symbol].appendleft(prev_cvd - delta)
            except Exception as e:
                logger.error(f"[History] Ошибка подгрузки для {symbol}: {e}")
        logger.info("Подгрузка истории для новых символов завершена.")

    async def _save_loop(self, interval: int = 60):
        while True:
            await asyncio.sleep(interval)
            self._save_history()

    def _save_history(self):
        try:
            with open(self._history_file, 'wb') as f:
                pickle.dump({
                    'candles': {k: list(v) for k, v in self.candles_data.items()},
                    'volume_history': {k: list(v) for k, v in self.volume_history.items()},
                    'oi_history': {k: list(v) for k, v in self.oi_history.items()},
                    'cvd_history': {k: list(v) for k, v in self.cvd_history.items()},
                }, f)
        except Exception as e:
            logger.warning(f"[History] Ошибка сохранения: {e}")
            
    def _sigma_5m(self, symbol: str, window: int = 60) -> float:
        candles = list(self.candles_data.get(symbol, []))[-window:]
        if len(candles) < window: return 0.0
        moves = [abs(c["closePrice"] - c["openPrice"]) / c["openPrice"] for c in candles if c["openPrice"] > 0]
        return float(np.std(moves)) if moves else 0.0

class TradingBot:
    __slots__ = (
        "user_id", "api_key", "api_secret", "mode", "session", "shared_ws", "ws_private", "ws_trade", "loop",
        "open_positions", "closed_positions", "position_lock", "active_trade_entries",
        "POSITION_VOLUME", "MAX_TOTAL_VOLUME", "leverage",
        "qty_step_map", "min_qty_map", "price_tick_map",
        "pending_orders", "pending_timestamps", "_inflight_until",
        "strategy_mode", "golden_param_store",
        "market_task", "sync_task", "wallet_task", "_cleanup_task",
        "ml_inferencer", "training_data", "training_data_path",
        "ai_provider", "openai_api_key", "ai_sem", "ai_timeout_sec", "ai_circuit_open_until",
        "warmup_done", "warmup_seconds", "listing_age_min",
        "last_squeeze_ts", "squeeze_cooldown_sec", "squeeze_threshold_pct", "squeeze_power_min",
        "_last_golden_ts", "golden_cooldown_sec",
        "recently_closed", "_evaluated_signals_cache",
    )

    def __init__(self, user_data, shared_ws, golden_param_store):
        # --- Основные данные пользователя и API ---
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.mode = user_data.get("mode", "real")
        self.session = HTTP(demo=(self.mode == "demo"), api_key=self.api_key, api_secret=self.api_secret, timeout=30)
        
        # --- WebSocket и общие данные ---
        self.shared_ws = shared_ws
        self.ws_private = None
        self.ws_trade = None
        self.loop = None
        
        # --- Управление позициями и ордерами ---
        self.open_positions: Dict[str, Dict] = {}
        self.closed_positions: Dict[str, Dict] = {}
        self.position_lock = asyncio.Lock()
        self.active_trade_entries: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, float] = {}
        self.pending_timestamps: Dict[str, float] = {}
        self._inflight_until: Dict[str, float] = {}
        self.recently_closed: Dict[str, float] = {}
        
        # --- Параметры торговли и риск-менеджмент ---
        self.POSITION_VOLUME = safe_to_float(user_data.get("volume", 1000))
        self.MAX_TOTAL_VOLUME = safe_to_float(user_data.get("max_total_volume", 5000))
        self.leverage = 10
        self.qty_step_map: Dict[str, float] = {}
        self.min_qty_map: Dict[str, float] = {}
        self.price_tick_map: Dict[str, float] = {}
        
        # --- Стратегии ---
        self.strategy_mode = user_data.get("strategy_mode", "full")
        self.golden_param_store = golden_param_store
        self.squeeze_threshold_pct = user_data.get("squeeze_threshold_pct", SQUEEZE_THRESHOLD_PCT)
        self.squeeze_power_min = user_data.get("squeeze_power_min", DEFAULT_SQUEEZE_POWER_MIN)
        self.squeeze_cooldown_sec = 60
        self.last_squeeze_ts: Dict[str, float] = defaultdict(float)
        self.golden_cooldown_sec = 300
        self._last_golden_ts: Dict[str, float] = defaultdict(float)

        # --- ML и AI ---
        self.ml_inferencer = MLXInferencer()
        self.ai_provider = user_data.get("ai_provider", "ollama")
        self.openai_api_key = user_data.get("openai_api_key")
        self.ai_sem = asyncio.Semaphore(2)
        self.ai_timeout_sec = 8.0
        self.ai_circuit_open_until = 0.0
        
        # --- Обучение ---
        self.training_data_path = Path(f"training_data_{self.user_id}.pkl")
        self.training_data: deque = self._load_training_data()
        
        # --- Задачи и прочее ---
        self.market_task = None
        self.sync_task = None
        self.wallet_task = None
        self._cleanup_task = asyncio.create_task(self._cleanup_recently_closed())
        self.warmup_seconds = 480
        self.warmup_done = False
        self.listing_age_min = LISTING_AGE_MIN_MINUTES
        self._evaluated_signals_cache: Dict[str, float] = {}

        GLOBAL_BOTS.append(self)
        if self.shared_ws:
            self.shared_ws.position_handlers.append(self)

    def _load_training_data(self) -> deque:
        """Загружает или создает буфер для обучающих данных."""
        if self.training_data_path.exists():
            try:
                with open(self.training_data_path, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"[{self.user_id}] Загружено {len(data)} обучающих примеров.")
                return data
            except Exception as e:
                logger.error(f"[{self.user_id}] Ошибка загрузки training_data: {e}.")
        return deque(maxlen=5000)

    async def start(self):
        self.loop = asyncio.get_running_loop()
        logger.info(f"[{self.user_id}] Бот запущен в режиме {self.mode.upper()}.")
        
        await self.update_open_positions()
        await self.setup_private_ws()
        if self.mode == "real":
            await self.init_trade_ws()

        logger.info(f"[{self.user_id}] Ожидание завершения warm-up периода ({self.warmup_seconds} сек)...")
        await asyncio.sleep(self.warmup_seconds)
        self.warmup_done = True
        logger.info(f"[{self.user_id}] Warm-up завершен. Запуск основного цикла.")

        self.market_task = asyncio.create_task(self.market_loop())
        self.sync_task = asyncio.create_task(self.sync_open_positions_loop())
        self.wallet_task = asyncio.create_task(self.wallet_loop())
        asyncio.create_task(self._retrain_loop())

    async def stop(self):
        """Корректная остановка всех задач бота."""
        logger.info(f"[{self.user_id}] Остановка бота...")
        try:
            with open(self.training_data_path, "wb") as f:
                pickle.dump(self.training_data, f)
            logger.info(f"[{self.user_id}] Сохранено {len(self.training_data)} обучающих примеров.")
        except Exception as e:
            logger.error(f"[{self.user_id}] Ошибка сохранения обучающих данных: {e}")

        for task_name in ("market_task", "sync_task", "wallet_task", "_cleanup_task"):
            task = getattr(self, task_name, None)
            if task:
                task.cancel()
        
        if self.ws_private: self.ws_private.exit()
        if self.ws_trade: await self.ws_trade.close()
        logger.info(f"[{self.user_id}] Бот остановлен.")

    async def market_loop(self):
        """Основной цикл поиска и оценки торговых сигналов."""
        while True:
            try:
                symbols_to_scan = [s for s in self.shared_ws.symbols if s not in EXCLUDED_SYMBOLS]
                random.shuffle(symbols_to_scan)
                
                tasks = [self.evaluate_strategies_for_symbol(s) for s in symbols_to_scan]
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[MarketLoop] Критическая ошибка: {e}", exc_info=True)
            
            await asyncio.sleep(1) # Пауза между полными циклами сканирования

    async def evaluate_strategies_for_symbol(self, symbol: str):
        """Запускает все активные стратегии для одного символа."""
        if not self.warmup_done or symbol in self.open_positions or symbol in self.pending_orders:
            return

        mode = self.strategy_mode
        # 1. Сквиз
        if mode in ("full", "squeeze_only", "golden_squeeze", "liq_squeeze"):
            await self._squeeze_logic(symbol)
        # 2. Ликвидации
        if mode in ("full", "liquidation_only", "liq_squeeze"):
            await self._liquidation_logic(symbol)
        # 3. Золотой сетап
        if mode in ("full", "golden_only", "golden_squeeze"):
            await self._golden_logic(symbol)

    async def process_signal_candidate(self, candidate: dict, features: dict):
        """Единый конвейер для обработки любого сигнала: ML-оценка -> AI-вердикт -> Исполнение."""
        symbol, side, source = candidate['symbol'], candidate['side'], candidate.get('source', 'unknown')
        signal_key = f"{symbol}_{side}_{source}_{int(time.time()) // 60}"

        if self._evaluated_signals_cache.get(signal_key): return
        self._evaluated_signals_cache[signal_key] = time.time()
        logger.info(f"[PIPELINE_START] {symbol}/{side} ({source})")

        # --- 1. Оценка ML-моделью ---
        try:
            vector = np.array([[safe_to_float(features.get(k, 0.0)) for k in FEATURE_KEYS]], dtype=np.float32)
            expected_pnl_pct = float(self.ml_inferencer.infer(vector)[0][0])
            
            if not (-5.0 < expected_pnl_pct < 5.0):
                logger.warning(f"[ML_GATE] {symbol} отклонен: аномальный прогноз PnL% = {expected_pnl_pct:.2f}")
                return

            if abs(expected_pnl_pct) < ML_GATE_MIN_SCORE:
                logger.info(f"[ML_GATE] {symbol} отклонен: прогноз PnL% ({expected_pnl_pct:.2f}%) ниже порога ({ML_GATE_MIN_SCORE:.2f}%)")
                return
            
            logger.info(f"[ML_GATE] {symbol} одобрен: прогноз PnL% = {expected_pnl_pct:.2f}%")
            candidate['ml_score'] = expected_pnl_pct
        except Exception as e:
            logger.warning(f"[ML_GATE] Ошибка ML-фильтра для {symbol}: {e}", exc_info=True)
            return

        # --- 2. Финальный вердикт от AI ---
        try:
            ai_response = await self._ai_call_with_timeout(self.ai_provider, candidate, features)
            if ai_response.get("action", "REJECT") != "EXECUTE":
                logger.info(f"[AI_REJECT] {symbol}/{side} ({source}) — {ai_response.get('justification', 'N/A')}")
                return
            
            logger.info(f"[AI_CONFIRM] {symbol}/{side} ({source}) ОДОБРЕН. {ai_response.get('justification')}")
            candidate.update(ai_response)
        except Exception as e:
            logger.error(f"[AI_EVAL] Критическая ошибка AI для {symbol}: {e}", exc_info=True)
            return

        # --- 3. Исполнение сделки ---
        await self.execute_trade_entry(candidate, features)

    async def execute_trade_entry(self, candidate: dict, features: dict):
        """Принимает одобренного кандидата, рассчитывает объем, проверяет риски и размещает ордер."""
        symbol, side = candidate['symbol'], candidate['side']
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
                return

            logger.info(f"[EXECUTE] {symbol}/{side}: Все проверки пройдены. Отправка ордера...")
            
            async with self.position_lock:
                self.active_trade_entries[symbol] = candidate

            if self.mode == 'real':
                pos_idx = 1 if side == "Buy" else 2
                order_link_id = self._make_order_link_id(symbol, side)
                await self.place_order_ws(symbol, side, qty, position_idx=pos_idx, orderLinkId=order_link_id)
            else: # demo
                await self.place_unified_order_guarded(symbol, side, qty, "Market", comment=candidate['justification'])

        except Exception as e:
            logger.error(f"[EXECUTE] Критическая ошибка для {symbol}: {e}", exc_info=True)
            self.active_trade_entries.pop(symbol, None)

    async def _ai_call_with_timeout(self, provider: str, candidate: dict, features: dict) -> dict:
        now = time.time()
        if now < self.ai_circuit_open_until:
            return {"action": "REJECT", "justification": "AI временно отключен (предохранитель)."}

        try:
            async with self.ai_sem:
                return await asyncio.wait_for(
                    self.evaluate_candidate_with_ollama(candidate, features),
                    timeout=self.ai_timeout_sec
                )
        except (asyncio.TimeoutError, RequestsReadTimeout, RequestsConnectionError) as e:
            self.ai_circuit_open_until = time.time() + 60
            logger.error(f"[AI_TIMEOUT] {provider} завис: {e}. Отключаю ИИ на 60 сек.")
            return {"action": "REJECT", "justification": f"AI timeout: {e}"}
        except Exception as e:
            self.ai_circuit_open_until = time.time() + 30
            logger.error(f"[AI_FAIL] {provider} упал: {e}", exc_info=True)
            return {"action": "REJECT", "justification": f"AI failure: {e}"}

    async def evaluate_candidate_with_ollama(self, candidate: dict, features: dict) -> dict:
        """Отправляет отчет в Ollama, включая оценку от ML-модели."""
        default_response = {"action": "REJECT", "justification": "Ошибка локального AI."}
        prompt = ""
        try:
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            def _format(v, spec): return f"{v:{spec}}" if isinstance(v, (int, float)) else "N/A"
            
            m, source, ml_score = candidate.get('base_metrics', {}), candidate.get('source', 'unknown'), candidate.get('ml_score', 0.0)
            btc_change_1h = compute_pct(self.shared_ws.candles_data.get("BTCUSDT", deque()), 60)
            eth_change_1h = compute_pct(self.shared_ws.candles_data.get("ETHUSDT", deque()), 60)

            prompt = f"""
            SYSTEM: Ты - элитный квантовый аналитик и риск-менеджер. Твой ответ - всегда только валидный JSON.
            USER:
            Анализ торгового сигнала для принятия решения.
            - Сигнал: {candidate['symbol']}, Направление: {candidate['side'].upper()}, Источник: {source.replace('_', ' ').title()}
            - Ключевые метрики сигнала: {json.dumps(m)}
            - Оценка ML-модели: Ожидаемый PnL = {_format(ml_score, '.2f')}%
            - Контекст рынка: ADX={_format(features.get('adx14'), '.1f')}, RSI={_format(features.get('rsi14'), '.1f')}, BTC Δ(1h)={_format(btc_change_1h, '.2f')}%, ETH Δ(1h)={_format(eth_change_1h, '.2f')}%

            ЗАДАЧА: На основе всех данных, верни JSON с ключами "action" ("EXECUTE" или "REJECT"), "justification" (краткое, но емкое обоснование) и "confidence_score" (0.0-1.0, твоя уверенность).
            """
            response = await client.chat.completions.create(
                model="trading-llama", messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.2
            )
            response_data = json.loads(response.choices[0].message.content)
            response_data['full_prompt_for_ai'] = prompt
            return response_data
        except Exception as e:
            logger.error(f"[Ollama] Ошибка API для {candidate['symbol']}: {e}", exc_info=True)
            return {**default_response, "full_prompt_for_ai": prompt}


    async def _squeeze_logic(self, symbol: str):
            """Ищет сигнал на сквиз и передает его как кандидата в пайплайн."""
            try:
                if not self._squeeze_allowed(symbol): return

                candles = list(self.shared_ws.candles_data.get(symbol, []))
                if len(candles) < 6: return

                old_close = safe_to_float(candles[-6]["closePrice"])
                new_close = safe_to_float(candles[-1]["closePrice"])
                if old_close <= 0: return
                pct_5m = (new_close - old_close) / old_close * 100.0
                
                avg_vol_prev_5m = sum(safe_to_float(c["volume"]) for c in candles[-6:-1]) / 5
                curr_vol_1m = safe_to_float(candles[-1]["volume"])
                if avg_vol_prev_5m <= 0: return
                vol_change_pct = (curr_vol_1m - avg_vol_prev_5m) / avg_vol_prev_5m * 100.0
                
                sigma_pct = self.shared_ws._sigma_5m(symbol) * 100
                thr_price = max(self.squeeze_threshold_pct, 1.5 * sigma_pct)
                power_min = max(self.squeeze_power_min, 4.0 * sigma_pct)
                squeeze_power = abs(pct_5m) * abs(vol_change_pct / 100.0)

                if abs(pct_5m) < thr_price or squeeze_power < power_min:
                    return
                
                oi_hist = list(self.shared_ws.oi_history.get(symbol, []))
                oi_change_pct = 0.0
                if len(oi_hist) >= 2:
                    oi_now, oi_prev = oi_hist[-1], oi_hist[-2]
                    if oi_prev > 0: oi_change_pct = (oi_now - oi_prev) / oi_prev * 100.0

                side = "Sell" if pct_5m >= thr_price else "Buy"
                
                features = await self.extract_realtime_features(symbol)
                if not features: return

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

            except Exception as e:
                logger.error(f"[_squeeze_logic] Ошибка для {symbol}: {e}", exc_info=True)
        
    async def _liquidation_logic(self, symbol: str):
        """Ищет сигнал по ликвидациям и передает его как кандидата в пайплайн."""
        try:
            liq_info = self.shared_ws.latest_liquidation.get(symbol, {})
            if not liq_info or (time.time() - liq_info.get("ts", 0)) > 60: return
            
            # Используем более простой порог для ликвидаций
            threshold = 10000 # $10k
            if liq_info.get("value", 0) < threshold: return

            if symbol in self.open_positions or symbol in self.pending_orders: return

            order_side = "Buy" if liq_info.get("side") == "Sell" else "Sell"
            
            features = await self.extract_realtime_features(symbol)
            if not features: return

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

        except Exception as e:
            logger.error(f"[_liquidation_logic] Ошибка для {symbol}: {e}", exc_info=True)

    async def _golden_logic(self, symbol: str):
        """Ищет "золотой сетап" и передает сигнал как кандидата в пайплайн."""
        try:
            if not self._golden_allowed(symbol): return

            minute_candles = list(self.shared_ws.candles_data.get(symbol, []))
            vol_hist_1m = list(self.shared_ws.volume_history.get(symbol, []))
            oi_hist_1m = list(self.shared_ws.oi_history.get(symbol, []))
            
            if len(minute_candles) < 6: return

            p_start = safe_to_float(minute_candles[-6]["closePrice"])
            p_end = safe_to_float(minute_candles[-1]["closePrice"])
            price_change_pct = (p_end - p_start) / p_start * 100.0 if p_start > 0 else 0.0

            vol_start = safe_to_float(vol_hist_1m[-6]) if len(vol_hist_1m) >= 6 else 0.0
            vol_end = safe_to_float(vol_hist_1m[-1]) if vol_hist_1m else 0.0
            volume_change_pct = (vol_end - vol_start) / vol_start * 100.0 if vol_start > 0 else 0.0

            oi_start = safe_to_float(oi_hist_1m[-6]) if len(oi_hist_1m) >= 6 else 0.0
            oi_end = safe_to_float(oi_hist_1m[-1]) if oi_hist_1m else 0.0
            oi_change_pct = (oi_end - oi_start) / oi_start * 100.0 if oi_start > 0 else 0.0

            buy_params = self.golden_param_store.get("Buy")
            sell_params = self.golden_param_store.get("Sell")

            side = None
            if (price_change_pct >= buy_params["price_change"] and
                volume_change_pct >= buy_params["volume_change"] and
                oi_change_pct >= buy_params["oi_change"]):
                side = "Buy"
            elif (price_change_pct <= -sell_params["price_change"] and
                  volume_change_pct >= sell_params["volume_change"] and
                  oi_change_pct >= sell_params["oi_change"]):
                side = "Sell"

            if not side: return

            features = await self.extract_realtime_features(symbol)
            if not features: return

            candidate = {
                "symbol": symbol, "side": side, "source": "golden_setup",
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
            logger.error(f"[_golden_logic] Ошибка для {symbol}: {e}", exc_info=True)




# ---------------------- 6. ОСНОВНАЯ ЛОГИКА ЗАПУСКА ----------------------

async def run_all() -> None:
    """Создаёт и запускает все компоненты системы."""
    users = load_users_from_json()
    if not users:
        logger.error("Нет активных пользователей в user_state.json. Завершение работы.")
        return

    # 1. Получаем первоначальный список символов
    http_session = HTTP(testnet=False)
    initial_symbols = await get_and_filter_symbols(http_session)
    if not initial_symbols:
        logger.error("Не удалось получить список символов. Завершение работы.")
        return

    # 2. Создаем WS менеджер с первоначальным списком
    shared_ws = PublicWebSocketManager(symbols=list(initial_symbols))
    
    # 3. Создаем ботов
    golden_param_store = load_golden_params()
    for user_data in users:
        GLOBAL_BOTS.append(TradingBot(user_data, shared_ws, golden_param_store))

    # 4. Запускаем все компоненты
    public_ws_task = asyncio.create_task(shared_ws.start())
    
    await shared_ws.ready_event.wait()
    logger.info("Public WebSocket готов. Начинаем подгрузку истории...")
    await shared_ws.backfill_history(list(initial_symbols))
    logger.info("Подгрузка истории завершена.")

    try:
        dp.include_router(router)
        dp.include_router(router_admin)
    except RuntimeError:
        pass
    telegram_task = asyncio.create_task(dp.start_polling(telegram_bot, skip_updates=True))

    bot_tasks = [asyncio.create_task(b.start()) for b in GLOBAL_BOTS]
    
    async def shutdown_handler():
        logger.info("Получен сигнал завершения. Остановка всех сервисов...")
        for task in bot_tasks + [public_ws_task, telegram_task]:
            task.cancel()
        for bot in GLOBAL_BOTS:
            await bot.stop()
        await asyncio.gather(*bot_tasks, public_ws_task, telegram_task, return_exceptions=True)
        logger.info("Все сервисы успешно остановлены.")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown_handler()))

    logger.info("Все боты запущены. Для остановки нажмите Ctrl+C.")
    await asyncio.gather(public_ws_task, telegram_task, *bot_tasks)


# ---------------------- 7. ТОЧКА ВХОДА И ОБРАБОТЧИКИ TELEGRAM ----------------------

@router_admin.message(Command("snapshot"))
async def cmd_snapshot(message: types.Message):
    """Обработчик команды /snapshot для администраторов."""
    if message.from_user.id not in ADMIN_IDS:
        return
    try:
        fname = await make_snapshot()
        await telegram_bot.send_document(
            message.from_user.id,
            types.FSInputFile(fname),
            caption="Снимок состояния счетов"
        )
    except Exception as e:
        logger.warning(f"[cmd_snapshot] Ошибка отправки документа: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        logger.info("Программа остановлена пользователем.")

# ----------------- КОНЕЦ ФАЙЛА MultiuserBot_FINAL_COMPLETE.py -----------------


