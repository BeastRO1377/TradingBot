import datetime as dt
from aiogram.enums import ParseMode

# Ensure required imports
import asyncio

# ADD: Import InvalidRequestError for advanced order error handling
from pybit.exceptions import InvalidRequestError

# ---------------------- IMPORTS ----------------------
import os
import csv
import json
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import time
import hmac
import websockets
import hashlib
import pandas as pd
import requests
import numpy as np
import lightgbm as lgb
from typing import List
from collections import defaultdict
from datetime import datetime, timezone
import pandas_ta as ta
from pybit.unified_trading import WebSocket, HTTP
from telegram_fsm_v12 import dp, router, router_admin
from telegram_fsm_v12 import bot as telegram_bot
from collections import deque
import pickle
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_DOWN, ROUND_HALF_UP, ROUND_FLOOR
import math           # ← добавьте эту строку
import random
import signal
from aiolimiter import AsyncLimiter  # Правильный импорт


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
SNAPSHOT_CSV_PATH = "golden_setup_snapshots.csv"
DEC_TICK = Decimal("0.000001")      # 1e‑6

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
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ---------------------- DECIMAL HELPER ----------------------
def safe_to_decimal(val) -> Decimal:
    """
    Robust conversion of Bybit numeric strings to Decimal.

    Returns Decimal(0) for "", None or unparsable values so that the
    open-interest handler never crashes on bad data.
    """
    try:
        if val in ("", None):
            return Decimal(0)
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(0)

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
    def __init__(self, symbols, interval="1"):
        self.symbols = symbols
        self.interval = interval
        self.ws = None
        self.candles_data   = defaultdict(lambda: deque(maxlen=1000))
        self.ticker_data = {}
        self.latest_open_interest = {}
        self.loop = asyncio.get_event_loop()
        # for golden setup
        self.volume_history = defaultdict(lambda: deque(maxlen=500))
        self.oi_history     = defaultdict(lambda: deque(maxlen=500))
        # track last saved candle time for deduplication
        self._last_saved_time = {}
        # Список ботов для оповещений об обновлениях тикера
        self.position_handlers = []
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
            print(f"[History] загружена история из {self._history_file}")
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
                # ── подпишемся на все liquidation-события ────────────────────────
                self.ws.all_liquidation_stream(
                    self.symbols,
                    callback=_on_message
                )
                # ждём, пока ws не упадёт (блокирующий Future)
                await asyncio.Event().wait()
                self.health_task = asyncio.create_task(self.health_check())

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[PublicWS] reconnect after error: %s", e)
                await asyncio.sleep(5)  # небольшая задержка перед новой попыткой


    async def route_message(self, msg):
        topic = msg.get("topic", "")
        if topic.startswith("kline."):
            await self.handle_kline(msg)
        elif topic.startswith("tickers."):
            await self.handle_ticker(msg)
        elif topic.startswith("liquidation."):
            await self.handle_liquidation(msg)
            for bot in self.position_handlers:
                asyncio.create_task(bot.handle_liquidation(msg))


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
                "openPrice": Decimal(str(entry.get("open", 0))),
                "highPrice": Decimal(str(entry.get("high", 0))),
                "lowPrice": Decimal(str(entry.get("low", 0))),
                "closePrice": Decimal(str(entry.get("close", 0))),
                "volume": Decimal(str(entry.get("volume", 0))),
            }
            self.candles_data[symbol].append(row)
            if len(self.candles_data[symbol]) > 500:
                self.candles_data[symbol] = self.candles_data[symbol][-500:]
            # record volume
            self.volume_history[symbol].append(row["volume"])
            # attach latest open‑interest snapshot to this confirmed candle
            oi_val = self.latest_open_interest.get(symbol, Decimal(0))
            self.oi_history[symbol].append(oi_val)
            self._save_history()
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
            oi_val = safe_to_decimal(oi).quantize(DEC_TICK)
            self.latest_open_interest[symbol] = oi_val
            self.ticker_data[symbol] = ticker

        # 2) Notify bots of ticker updates for their open positions
        for bot in self.position_handlers:
            for ticker in entries:
                sym = ticker.get("symbol")
                if not sym or sym not in bot.open_positions:
                    continue
                last_price = safe_to_decimal(ticker.get("lastPrice", 0)).quantize(DEC_TICK)
                # schedule on_ticker_update on the bot
                asyncio.create_task(bot.on_ticker_update(sym, last_price))



    async def backfill_history(self):
        """Backfill missing candles using public HTTP endpoint."""
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
                            open_p  = safe_to_decimal(entry[1])
                            high_p  = safe_to_decimal(entry[2])
                            low_p   = safe_to_decimal(entry[3])
                            close_p = safe_to_decimal(entry[4])
                            vol     = safe_to_decimal(entry[5])
                        except Exception:
                            print(f"[History] backfill invalid list entry for {symbol}: {entry}")
                            continue
                    else:
                        try:
                            ts = pd.to_datetime(int(entry['start']), unit='ms')
                        except Exception as e:
                            print(f"[History] backfill invalid dict entry for {symbol}: {e}")
                            continue
                        open_p  = safe_to_decimal(entry.get('open', 0))
                        high_p  = safe_to_decimal(entry.get('high', 0))
                        low_p   = safe_to_decimal(entry.get('low', 0))
                        close_p = safe_to_decimal(entry.get('close', 0))
                        vol     = safe_to_decimal(entry.get('volume', 0))
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
                    count += 1
                if count:
                    # Trim candle history to last 500 bars without slicing deque
                    self._save_history()
                    while len(self.candles_data[symbol]) > 500:
                        self.candles_data[symbol].popleft()
                    # В backfill_history, после получения ticker_resp:
                    try:
                        ticker_resp = http.get_tickers(category="linear", symbol=symbol)
                        # Используем безопасное извлечение
                        oi_val = safe_to_decimal(
                            ticker_resp["result"]["list"][0].get("openInterest", 0) or
                            ticker_resp["result"]["list"][0].get("open_interest", 0)
                        ).quantize(DEC_TICK)
                    except Exception:
                        oi_val = Decimal(0)

                    # если длина истории OI меньше, чем количества свечей,
                    # дозаполняем одинаковым значением (этого достаточно,
                    # чтобы пройти проверку len(...) в execute_golden_setup)
                    need = len(self.candles_data[symbol]) - len(self.oi_history[symbol])
                    if need > 0:
                        self.oi_history[symbol].extend([oi_val] * need)
                logger.info("[History] backfilled %d bars for %s", count, symbol)
            except Exception as e:
                print(f"[History] backfill error for {symbol}: {e}")


    def _save_history(self):
        try:
            with open(self._history_file, 'wb') as f:
                pickle.dump({
                    'candles': dict(self.candles_data),
                    'volume_history': {k: list(v) for k, v in self.volume_history.items()},
                    'oi_history': {k: list(v) for k, v in self.oi_history.items()},
                }, f)
        except Exception as e:
            print(f"[History] ошибка сохранения: {e}")




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
        # Lock to serialize WS send/recv calls
        self.ws_trade_lock = asyncio.Lock()
        # Индекс позиции (для Bybit V5: 1 или 2)
        self.position_idx = user_data.get("position_idx", 1)
        self.load_model()
        self.POSITION_VOLUME = safe_to_decimal(user_data.get("volume", 1000))
        # максимальный разрешённый общий объём открытых позиций (USDT)
        self.MAX_TOTAL_VOLUME = safe_to_decimal(user_data.get("max_total_volume", 5000))
        # Maximum allowed total exposure across all open positions (in USDT)
        self.qty_step_map: dict[str, Decimal] = {}
        self.min_qty_map: dict[str, Decimal] = {}
        # track symbols that recently failed order placement
        self.failed_orders: dict[str, Decimal] = {}
        # символы, по которым уже отправлен ордер,
        # но позиция ещё не пришла по private‑WS
        self.pending_orders: set[str] = set()
        self.last_trailing_stop_set: dict[str, Decimal] = {}
        self.position_lock = asyncio.Lock()
        # Словарь закрытых позиций
        self.closed_positions = {}
        # Task for periodic PnL checks
        self.pnl_task = None
        self.last_seq = {}
        # Track WS-opened and WS-closed symbols to priority state from WS
        self.ws_opened_symbols = set()
        self.ws_closed_symbols = set()
        self.limiter = AsyncLimiter(max_rate=5, time_period=1)  # 5 вызовов/сек


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

    async def get_total_open_volume(self) -> Decimal:
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
            total = Decimal(0)
            for pos in resp.get("result", {}).get("list", []):
                size = safe_to_decimal(pos.get("size", 0))
                price = safe_to_decimal(pos.get("entryPrice", 0)) or safe_to_decimal(
                    pos.get("markPrice", 0)
                )
                total += size * price
            return total
        except Exception as e:
            logger.warning(
                "[get_total_open_volume] fallback due to %s", e
            )
            total = Decimal(0)
            for pos in self.open_positions.values():
                try:
                    size = safe_to_decimal(pos.get("size", 0))
                    price = safe_to_decimal(pos.get("entryPrice", 0)) or safe_to_decimal(
                        pos.get("markPrice", 0)
                    )
                    total += size * price
                except (ValueError, TypeError):
                    continue
            return total

    async def symbols_refresh_loop(self):
        """
        Periodically refresh selected symbols list every hour
        and update shared_ws subscription for market_loop.
        """
        while True:
            await asyncio.sleep(3600)  # 1 час
            try:
                new_symbols = await self.get_selected_symbols()
                if set(new_symbols) != set(self.symbols):
                    self.symbols = new_symbols
                    if self.shared_ws:
                        self.shared_ws.symbols = new_symbols
                    logger.info(f"[symbols_refresh_loop] Updated symbols: {new_symbols}")
            except Exception as e:
                logger.warning(f"[symbols_refresh_loop] error updating symbols: {e}")

    async def start(self):
        logger.info(f"[User {self.user_id}] Бот запущен")
        # очистка кэша позиций перед первым REST-запросом
        self.open_positions.clear()
        # Cache the running event-loop so we can call run_coroutine_threadsafe from WS callbacks
        self.loop = asyncio.get_running_loop()
        await self.update_open_positions()

        self.symbols = await self.get_selected_symbols()
        # передаём список в shared_ws для market_loop
        if self.shared_ws:
            self.shared_ws.symbols = self.symbols
        # запускаем периодическое обновление списка пар
        self.symbols_refresh_task = asyncio.create_task(self.symbols_refresh_loop())


        # sequentially initialize private and trade WebSockets
        await self.setup_private_ws()
        await self.init_trade_ws()
        # Start main loops immediately
        self.market_task = asyncio.create_task(self.market_loop())
        self.sync_task   = asyncio.create_task(self.sync_open_positions_loop())
        self.pnl_task    = asyncio.create_task(self.pnl_loop())
        self.symbols_refresh_task = asyncio.create_task(self.symbols_refresh_loop())

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
                        logger.info(f"[PrivateWS] Raw message: {msg}")
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
        #logger.info(f"[PositionStream] Received position update via WebSocket: {msg}")
        # 1) Нормализуем data
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]

        # 2) Оставляем только новые size>0 (открытие) и уже открытые (для закрытия)
        data = [
            p for p in data
            if safe_to_decimal(p.get("size", 0)) > 0 or p.get("symbol") in self.open_positions
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

                # 4) Распаковываем ключевые поля
                side_raw   = position.get("side", "")  # 'Buy' или 'Sell', иначе ''
                # avgPrice иногда пустой -> fallback на entryPrice
                avg_price  = safe_to_decimal(position.get("avgPrice")) \
                            or safe_to_decimal(position.get("entryPrice"))
                new_size   = safe_to_decimal(position.get("size", 0))
                open_int   = self.shared_ws.latest_open_interest.get(symbol, Decimal(0))
                prev       = self.open_positions.get(symbol)

                #logger.info(
                #    f"[PositionStream] {symbol} update: side={side_raw or 'N/A'}, "
                #    f"avg_price={avg_price}, size={new_size}"
                #)

                # 5) Открытие позиции
                if prev is None and new_size > 0 and side_raw:
                    # Сохраняем в открытые
                    self.open_positions[symbol] = {
                        "avg_price": avg_price,
                        "side":      side_raw,
                        "pos_idx":   position.get("positionIdx", 1),
                        "volume":    new_size,
                        "amount":    safe_to_decimal(position.get("positionValue"))
                    }
                    # Mark WS-opened
                    self.ws_opened_symbols.add(symbol)
                    self.ws_closed_symbols.discard(symbol)
                    logger.info(f"[PositionStream] Scheduling evaluate_position for {symbol}")
                    asyncio.create_task(self.evaluate_position(position))

                    # Логируем открытие (в фоне)
                    asyncio.create_task(self.log_trade(
                        symbol,
                        side=side_raw,
                        avg_price=avg_price,
                        volume=new_size,
                        open_interest=open_int,
                        action="open",
                        result="opened"
                    ))

                    # Уведомляем пользователя (в фоне)
                    asyncio.create_task(self.notify_user(
                        f"🟢 Открыта {side_raw.upper()}-позиция {symbol}: объём {new_size} @ {avg_price}"
                    ))
                    # После подтверждения открытия освобождаем слот в pending_orders
                    self.pending_orders.discard(symbol)

                # 6) Закрытие позиции
                if prev is not None and new_size == 0:
                    logger.info(f"[PositionStream] Закрытие позиции {symbol}, PnL={position.get('unrealisedPnl')}")
                    # Копируем данные в closed_positions
                    self.closed_positions[symbol] = {
                        **prev,
                        "closed_pnl":  position.get("unrealisedPnl"),
                        "closed_time": position.get("updatedTime")
                    }
                    # # Удаляем из открытых
                    # del self.open_positions[symbol]
                    # Mark WS-closed
                    self.ws_closed_symbols.add(symbol)
                    self.ws_opened_symbols.discard(symbol)

                    # Логируем закрытие (в фоне)
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
                    # Уведомляем пользователя (в фоне)
                    asyncio.create_task(self.notify_user(
                        f"⏹️ Закрыта {prev['side'].upper()}-позиция {symbol}: "
                        f"объём {prev['volume']} @ {prev['avg_price']}"
                    ))

                    # Удаляем позицию из активных, чтобы золотой сетап мог запуститься снова
                    del self.open_positions[symbol]
                    # На всякий случай сбрасываем флаг pending_orders
                    self.pending_orders.discard(symbol)
                    continue

                # 7) Обновление объёма существующей позиции
                if prev is not None and new_size > 0 and new_size != prev.get("volume"):
                    logger.info(f"[PositionStream] Обновление объёма {symbol}: {prev['volume']} → {new_size}")
                    self.open_positions[symbol]["volume"] = new_size
                    # Запускаем мгновенную переоценку позиции по новому объёму
                    asyncio.create_task(self.evaluate_position({
                        "symbol": symbol,
                        "size":   str(new_size),
                        "side":   prev["side"]
                    }))

                    continue

        except Exception as e:
            logger.error(f"[handle_position_update] Ошибка обработки: {e}", exc_info=True)
            # Сбрасываем состояние для символа
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            await self.update_open_positions()


    # async def evaluate_position(self, position):
    #     symbol = position.get("symbol")
    #     size = safe_to_decimal(position.get("size", 0))
    #     side = position.get("side", "Buy")
    #     entry_price = safe_to_decimal(position.get("entryPrice", 0))
    #     previous = self.last_position_state.get(symbol)
    #     current = (side, size)
    #     if previous == current:
    #         return  # состояние не изменилось — пропускаем
    #     self.last_position_state[symbol] = current
    #     mark_price = safe_to_decimal(position.get("markPrice", 0))
    #     pnl = safe_to_decimal(position.get("unrealisedPnl", 0))

    #     if size == 0:
    #         self.open_positions.pop(symbol, None)
    #         print(f"[User {self.user_id}] {symbol} позиция закрыта")
    #         row = self.shared_ws.candles_data.get(symbol, [])[-1] if self.shared_ws.candles_data.get(symbol) else {}
    #         await self.log_trade(symbol, row=row, side=side, open_interest=self.shared_ws.latest_open_interest.get(symbol),
    #              action="close", result="closed", closed_manually=False)
    #         return

        
    #     if symbol not in self.open_positions:
    #         row = self.shared_ws.candles_data.get(symbol, [])[-1] if self.shared_ws.candles_data.get(symbol) else {}
    #         await self.log_trade(symbol, row=row, side=side, open_interest=self.shared_ws.latest_open_interest.get(symbol),
    #              action="open", result="opened")
    #     self.open_positions[symbol] = position
    #     print(f"[User {self.user_id}] {symbol} позиция обновлена, PnL: {pnl:.2f} USDT")
    #     # позиция обновлена — без логирования, чтобы не засорять лог

        
    #     # Проверка лимита общего объёма
    #     user_state = self.load_user_state()
    #     max_volume = safe_to_decimal(user_state.get("max_total_volume", 10000))
    #     current_total = sum(safe_to_decimal(p.get("size", 0)) for p in self.open_positions.values())
    #     if symbol not in self.open_positions and (current_total + size) > max_volume:
    #         print(f"[User {self.user_id}] ⚠️ Превышен лимит общего объёма: {current_total + size} > {max_volume}")
    #         await self.notify_user(f"⚠️ Позиция по {symbol} не открыта — превышен лимит общего объёма ({max_volume} USDT)")
    #         return

    #     if entry_price > 0:
    #         direction = 1 if side == "Buy" else -1
    #         pnl_pct = direction * ((mark_price - entry_price) / entry_price) * 100
    #         if pnl_pct >= 5:
    #             await self.set_trailing_stop(symbol, entry_price, pnl_pct, side)

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
                    size = safe_to_decimal(pos.get("size", 0))
                    return size == 0
            return True  # Символ не найден — считаем закрытым
        except Exception as e:
            logger.warning(f"[confirm_position_closing] Ошибка проверки позиции {symbol}: {e}")
            return True  # По ошибке — не мешаем обработке

    async def handle_liquidation(self, msg):
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]
        for evt in data:
            symbol = evt.get("symbol")
            qty = safe_to_decimal(evt.get("qty", 0))
            self.latest_open_interest.setdefault(symbol, Decimal(0))
            # сохраняем последнюю ликвидацию в shared_ws
            self.oi_history[symbol].append(qty)      # или отдельный deque, если хотите
            logger.info(f"[PublicWS liquidation] {symbol} qty={qty}")

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
            avg_price = safe_to_decimal(data.get("avg_price") or data.get("avgPrice", 0))
            pos_idx   = data.get("pos_idx") or data.get("positionIdx") or 1
            prev_vol  = safe_to_decimal(data.get("volume") or data.get("size", 0))

            # For compatibility: keep data["avg_price"], data["pos_idx"], data["volume"] available
            # (normalize values in-place if not present)
            if "avg_price" not in data:
                data["avg_price"] = avg_price
            if "pos_idx" not in data:
                data["pos_idx"] = pos_idx
            if "volume" not in data:
                data["volume"] = prev_vol

            # Compute current values
            size = safe_to_decimal(position.get("size", 0)).quantize(DEC_TICK)
            side = data["side"].lower()
            # avg_price, prev_vol already set above

            # Get latest price
            last_price = safe_to_decimal(
                self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
            ).quantize(DEC_TICK)

            # Fallback: если lastPrice из тикеров нулевой, берём closePrice из последней свечи
            if last_price <= Decimal("0"):
                recent = self.shared_ws.candles_data.get(symbol, [])
                if recent:
                    candle_close = safe_to_decimal(recent[-1]["closePrice"]).quantize(DEC_TICK)
                    logger.info(f"[evaluate_position] fallback candle price for {symbol}: {candle_close}")
                    last_price = candle_close

            # Compute PnL
            pnl = (last_price - avg_price) if side == "buy" else (avg_price - last_price)
            pnl_pct = (pnl / avg_price * Decimal("1000")).quantize(Decimal("0.00001"))
            #logger.info(f"[evaluate_position] {symbol}: "
            #            f"last_price={last_price}, avg_price={avg_price}, "
            #            f"pnl={pnl}, pnl_pct={pnl_pct}%")

            if pnl > 0:
                logger.info(f"[evaluate_position] {symbol}: last_price={last_price}, avg_price={avg_price}, pnl={pnl}, pnl_pct={pnl_pct}%")

            # Averaging: if loss exceeds 16% unlevered (with 160x leverage → pnl_pct ≤ -160)
            if pnl_pct <= Decimal("-160.0"):
                logger.info(f"[evaluate_position] {symbol}: last_price={last_price}, avg_price={avg_price}, pnl={pnl}, pnl_pct={pnl_pct}%")
                # volume_to_add = size
                # try:
                #     # use original side casing for order submission
                #     orig_side = data.get("side", "Buy")
                #     if self.mode == "real":
                #         await self.place_order_ws(symbol, orig_side, volume_to_add, position_idx=pos_idx)
                #     else:
                #         resp = await asyncio.to_thread(lambda: self.session.place_order(
                #             category="linear",
                #             symbol=symbol,
                #             side=orig_side,
                #             orderType="Market",
                #             qty=str(volume_to_add),
                #             timeInForce="GoodTillCancel",
                #             positionIdx=pos_idx
                #         ))
                #         if resp.get("retCode", 0) != 0:
                #             raise InvalidRequestError(resp.get("retMsg", "order rejected"))
                #     logger.info(f"[evaluate_position] Averaging executed for {symbol}: added volume {volume_to_add}")
                #     # обновляем внутренний объём
                #     self.open_positions[symbol]["volume"] += volume_to_add
                # except RuntimeError as e:
                #     msg = str(e)
                #     # Handle insufficient balance error from Bybit
                #     if "ab not enough for new order" in msg:
                #         logger.warning(f"[evaluate_position] averaging skipped for {symbol}: insufficient balance ({msg})")
                #         return
                #     # Re-raise or log other runtime errors
                #     logger.error(f"[evaluate_position] averaging failed for {symbol}: {e}", exc_info=True)
                # except Exception as e:
                #     logger.error(f"[evaluate_position] averaging failed for {symbol}: {e}", exc_info=True)

            # Update volume
            self.open_positions[symbol]["volume"] = size
            self.pending_orders.discard(symbol)

            # Trailing stop logic: trigger once when PnL ≥ threshold
            threshold = Decimal("5")
            last_pct = self.last_trailing_stop_set.get(symbol, Decimal("0"))
            if pnl_pct >= threshold and pnl_pct > last_pct:
                await self.set_trailing_stop(symbol, avg_price, pnl_pct, side)
                self.last_trailing_stop_set[symbol] = pnl_pct

    async def on_ticker_update(self, symbol, last_price):
        if symbol in self.open_positions:
            pos_data = self.open_positions[symbol]
            # update stored mark price
            pos_data['markPrice'] = last_price
            # build a minimal position dict for evaluation
            position = {
                "symbol": symbol,
                "size": str(pos_data.get("volume", 0)),
                "side": pos_data.get("side", "")
            }
            await self.evaluate_position(position)

    async def market_loop(self):
        while True:
            last_heartbeat = time.time()
            iteration = 0
            try:
                while True:
                    iteration += 1
                    symbols_shuffled = list(self.shared_ws.symbols)
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
            # 1. Пропускаем недавно провалившиеся символы
            if symbol in self.failed_orders and time.time() - self.failed_orders[symbol] < 600:
                return
            # 2. Не трогаем уже открытые или ожидающие открытие
            if symbol in self.open_positions or symbol in self.pending_orders:
                return

            # 3. Берём историю свечей/объёма/OI
            recent = self.shared_ws.candles_data.get(symbol, [])
            if not recent:
                return

            buy_params  = self.golden_param_store.get((symbol, "Buy"),  self.golden_param_store.get("Buy"))
            sell_params = self.golden_param_store.get((symbol, "Sell"), self.golden_param_store.get("Sell"))
            sell2_params= self.golden_param_store.get((symbol, "Sell2"),self.golden_param_store.get("Sell2", {}))

            period_iters = max(
                int(buy_params["period_iters"]),
                int(sell_params["period_iters"]),
                int(sell2_params.get("period_iters", 0)),
            )

            if (len(recent) <= period_iters or
                len(self.shared_ws.volume_history[symbol]) <= period_iters or
                len(self.shared_ws.oi_history[symbol]) <= period_iters):
                return

            # 4. Рассчитываем Δ от прошлой точки
            old_bar = recent[-1 - period_iters]
            new_bar = recent[-1]
            close_price = Decimal(str(new_bar["closePrice"]))
            old_close   = Decimal(str(old_bar["closePrice"])) if old_bar["closePrice"] else Decimal("0")

            price_change_pct = (
                (close_price - old_close) / old_close * Decimal("100")
                if old_close != 0 else Decimal("0")
            )

            old_vol = Decimal(str(self.shared_ws.volume_history[symbol][-1 - period_iters]))
            new_vol = Decimal(str(self.shared_ws.volume_history[symbol][-1]))
            volume_change_pct = (
                (new_vol - old_vol) / old_vol * Decimal("100")
                if old_vol != 0 else Decimal("0")
            )

            old_oi  = Decimal(str(self.shared_ws.oi_history[symbol][-1 - period_iters]))
            new_oi  = Decimal(str(self.shared_ws.oi_history[symbol][-1]))
            oi_change_pct = (
                (new_oi - old_oi) / old_oi * Decimal("100")
                if old_oi != 0 else Decimal("0")
            )

            #logger.info(
            #    "[GoldenSetup] %s | ΔP=%.3f%%  ΔV=%.1f%%  ΔOI=%.2f%%  iters=%d",
            #    symbol, price_change_pct, volume_change_pct, oi_change_pct, period_iters
            #)

            # 5. Определяем сигнал
            action = None
            # основной Sell
            sp = int(sell_params["period_iters"])
            if len(recent) > sp:
                old = recent[-1 - sp]
                new = recent[-1]
                pchg = (Decimal(str(new["closePrice"])) - Decimal(str(old["closePrice"]))) / Decimal(str(old["closePrice"])) * Decimal("100") if old["closePrice"] else Decimal("0")
                volchg = (Decimal(str(self.shared_ws.volume_history[symbol][-1 - sp])) - Decimal(str(self.shared_ws.volume_history[symbol][-1]))) / Decimal(str(self.shared_ws.volume_history[symbol][-1 - sp])) * Decimal("100") if self.shared_ws.volume_history[symbol][-1 - sp] else Decimal("0")
                oichg = (Decimal(str(self.shared_ws.oi_history[symbol][-1 - sp])) - Decimal(str(self.shared_ws.oi_history[symbol][-1]))) / Decimal(str(self.shared_ws.oi_history[symbol][-1 - sp])) * Decimal("100") if self.shared_ws.oi_history[symbol][-1 - sp] else Decimal("0")
                if (pchg <= Decimal(str(sell_params["price_change"])) * -1 and
                    volchg >= Decimal(str(sell_params["volume_change"])) and
                    oichg >= Decimal(str(sell_params["oi_change"]))):
                    action = "Sell"
            # альтернативный Sell2
            if action is None and sell2_params:
                sp2 = int(sell2_params["period_iters"])
                if len(recent) > sp2:
                    old2, new2 = recent[-1 - sp2], recent[-1]
                    pchg2 = (Decimal(str(new2["closePrice"])) - Decimal(str(old2["closePrice"]))) / Decimal(str(old2["closePrice"])) * Decimal("100") if old2["closePrice"] else Decimal("0")
                    volchg2 = (Decimal(str(self.shared_ws.volume_history[symbol][-1 - sp2])) - Decimal(str(self.shared_ws.volume_history[symbol][-1]))) / Decimal(str(self.shared_ws.volume_history[symbol][-1 - sp2])) * Decimal("100") if self.shared_ws.volume_history[symbol][-1 - sp2] else Decimal("0")
                    oichg2 = (Decimal(str(self.shared_ws.oi_history[symbol][-1 - sp2])) - Decimal(str(self.shared_ws.oi_history[symbol][-1]))) / Decimal(str(self.shared_ws.oi_history[symbol][-1 - sp2])) * Decimal("100") if self.shared_ws.oi_history[symbol][-1 - sp2] else Decimal("0")
                    if (pchg2 <= Decimal(str(sell2_params["price_change"])) * -1 and
                        volchg2 <= Decimal(str(sell2_params["volume_change"])) and
                        oichg2 <= Decimal(str(sell2_params["oi_change"]))):
                        action = "Sell"
            # Buy
            if action is None:
                lp = int(buy_params["period_iters"])
                if len(recent) > lp:
                    oldb, newb = recent[-1 - lp], recent[-1]
                    pchgb = (Decimal(str(newb["closePrice"])) - Decimal(str(oldb["closePrice"]))) / Decimal(str(oldb["closePrice"])) * Decimal("100") if oldb["closePrice"] else Decimal("0")
                    volb = (Decimal(str(self.shared_ws.volume_history[symbol][-1 - lp])) - Decimal(str(self.shared_ws.volume_history[symbol][-1]))) / Decimal(str(self.shared_ws.volume_history[symbol][-1 - lp])) * Decimal("100") if self.shared_ws.volume_history[symbol][-1 - lp] else Decimal("0")
                    oib = (Decimal(str(self.shared_ws.oi_history[symbol][-1 - lp])) - Decimal(str(self.shared_ws.oi_history[symbol][-1]))) / Decimal(str(self.shared_ws.oi_history[symbol][-1 - lp])) * Decimal("100") if self.shared_ws.oi_history[symbol][-1 - lp] else Decimal("0")
                    if (pchgb >= Decimal(str(buy_params["price_change"])) and
                        volb >= Decimal(str(buy_params["volume_change"])) and
                        oib >= Decimal(str(buy_params["oi_change"]))):
                        action = "Buy"

            if action is None:
                return

            # 7. Пропускаем или резервируем позицию и проверяем лимит в критической секции
            side_params = buy_params if action == "Buy" else sell_params
            volume_usdt = safe_to_decimal(side_params.get("position_volume", self.POSITION_VOLUME))
            last_price  = safe_to_decimal(close_price)
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

            qty_raw = volume_usdt / last_price
            step    = self.qty_step_map.get(symbol, Decimal("0.001"))
            min_qty = self.min_qty_map.get(symbol, step)
            factor   = (qty_raw / step).to_integral_value(rounding=ROUND_FLOOR)  # Decimal
            qty      = factor * step                                           # Decimal
            
            decimals = max(0, -step.as_tuple().exponent)
            qty      = qty.quantize(step, rounding=ROUND_DOWN)
            # Ensure qty is rounded down to an integral multiple of step
            qty = qty.to_integral_value(rounding=ROUND_FLOOR)
            if qty < min_qty or qty <= 0:
                logger.info(f"[GoldenSetup] {symbol} — qty {qty:.8f} < min_qty {min_qty:.8f}; пропуск")
                # Убираем резерв, если не прошли проверку количества
                self.pending_orders.discard(symbol)
                return

            # 6. Логирование снимка
            _append_snapshot({
                "close_price":    safe_to_decimal(str(close_price)),
                "price_change":   safe_to_decimal(str(price_change_pct)),
                "volume_change":  safe_to_decimal(str(volume_change_pct)),
                "oi_change":      safe_to_decimal(str(oi_change_pct)),
                "period_iters":   period_iters,
                "user_id":        self.user_id,
                "symbol":         symbol,
                "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })

            pos_idx = 1 if action == "Buy" else 2

            # 9. Исполнение и логирование
            remaining_qty = qty
            step = self.qty_step_map.get(symbol, Decimal("0.001"))
            min_qty = self.min_qty_map.get(symbol, step)

            while remaining_qty >= min_qty:
                # Failsafe: re-check total volume limit before placing any orders
                cur_total = await self.get_total_open_volume()
                # volume_usdt already reserved; here we calculate actual USDT exposure of this tranche
                tranche_usdt = remaining_qty * last_price
                if cur_total + tranche_usdt > self.MAX_TOTAL_VOLUME:
                    logger.info(f"[execute_golden_setup] {symbol}: total volume limit exceeded, aborting placement")
                    # release reservation and skip
                    self.pending_orders.discard(symbol)
                    return
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
                        remaining_qty = remaining_qty.quantize(step, rounding=ROUND_FLOOR)
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
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"[execute_golden_setup] trade WS closed for {symbol}: {e}, reconnecting")
            # снимаем резерв, чтобы не застрять
            self.pending_orders.discard(symbol)
            # закрываем старый trade WS, если он есть
            if self.ws_trade:
                await self.ws_trade.close()
            # повторно инициализируем trade WS
            await self.init_trade_ws()
            return
        except Exception as e:
            logger.error(f"[execute_golden_setup] unexpected error for {symbol}: {e}", exc_info=True)
            self.pending_orders.discard(symbol)


    async def place_order_ws(self, symbol, side, qty, position_idx=1, price=None, order_type="Market"):
        """
        Send a WS order.create on the trade socket.
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

        # send & wait for response under lock to avoid concurrent recv
        async with self.ws_trade_lock:
            await self.ws_trade.send(json.dumps(req))
            resp = json.loads(await self.ws_trade.recv())
        if resp.get("retCode", resp.get("ret_code", 0)) != 0:
            raise RuntimeError(f"Order failed: {resp}")
        return resp["data"]


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
            last_price = safe_to_decimal(
                self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice", 0)
            )
            entry_price = safe_to_decimal(avg_price)
            vol = safe_to_decimal(volume)
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


    # --- внутри класса TradingBot ---

    # async def sync_open_positions_loop(self, interval: int = 30):
    #     while True:
    #         try:
    #             await self.update_open_positions()
    #             # Принудительно удаляем устаревшие позиции
    #             active_symbols = set(pos["symbol"] for pos in self.open_positions.values())
    #             for symbol in list(self.open_positions.keys()):
    #                 if symbol not in active_symbols:
    #                     logger.info(f"[sync] Позиция {symbol} удалена из open_positions")
    #                     self.open_positions.pop(symbol, None)
    #         except Exception as e:
    #             logger.warning("[sync_open_positions] Ошибка синхронизации: %s", e)
    #         await asyncio.sleep(interval)

    async def sync_open_positions_loop(self, interval: int = 5):
        while True:
            try:
                async with self.position_lock:
                    await self.update_open_positions()
                    current_symbols = set(self.open_positions.keys())
                    for symbol in list(self.open_positions.keys()):
                        if symbol not in current_symbols:
                            logger.info(f"[sync] Удалена позиция: {symbol}")
                            del self.open_positions[symbol]
            except Exception as e:
                logger.error(f"[sync] Critical error: {e}", exc_info=True)
            await asyncio.sleep(interval)

    async def set_trailing_stop(self, symbol, avg_price: Decimal, pnl_pct: Decimal, side: str):
        try:
            logger.info(f"[trailing_stop] Попытка установки стопа для {symbol} | ROI={pnl_pct}%")
            data = self.open_positions.get(symbol)
            if not data:
                logger.warning(f"[trailing_stop] Позиция {symbol} не найдена")
                return
            # Skip trailing-stop for positions of zero size
            volume = safe_to_decimal(data.get("volume", 0))
            if volume <= 0:
                logger.warning(f"[trailing_stop] Пропуск установки стопа для {symbol}: нулевая позиция")
                return
            pos_idx = data["pos_idx"]

            # базовый трейл и редукции
            base_trail = Decimal("2.5")
            reduction = Decimal("0")
            oi = self.shared_ws.latest_open_interest.get(symbol, Decimal(0))
            if oi > Decimal("1000"):
                reduction += Decimal("0.5")
            recent = self.shared_ws.candles_data.get(symbol, [])
            if recent:
                last_close = safe_to_decimal(recent[-1]["closePrice"])
                if abs(last_close - avg_price) / avg_price * Decimal("100") > Decimal("1"):
                    reduction += Decimal("0.5")

            final_trail = max(base_trail - reduction, Decimal("0.5"))
            stop_pct = (pnl_pct - final_trail).quantize(Decimal("0.01"))

            # повторная проверка позиции
            if symbol not in self.open_positions:
                logger.warning(f"[trailing_stop] Позиция {symbol} уже закрыта")
                return

            # вычисляем цену стопа
            if side.lower() == "buy":
                stop_price = (avg_price * (Decimal("1") + stop_pct / Decimal("1000"))).quantize(DEC_TICK, rounding=ROUND_HALF_UP)
            elif side.lower() == "sell":
                # Для шорт-позиций стоп-лосс должен быть выше цены входа
                stop_price = (avg_price * (Decimal("1") - stop_pct / Decimal("1000"))).quantize(DEC_TICK, rounding=ROUND_HALF_UP)
            else:
                logger.error(f"[trailing_stop] Unknown side {side}")
                return

            logger.info(
                f"[trailing_stop] вычислено: base_trail={base_trail}, reduction={reduction}, "
                f"final_trail={final_trail}, stop_pct={stop_pct}, stop_price={stop_price}"
            )

            try:
                resp = await asyncio.to_thread(lambda: self.session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    positionIdx=pos_idx,
                    stopLoss=str(stop_price),
                    triggerBy="LastPrice",
                    timeInForce="GoodTillCancel"
                ))
                logger.info(f"[trailing_stop] Стоп установлен: {symbol} | stopPrice={stop_price} | pct={stop_pct}%")
                await self.log_trade(
                    symbol=symbol,
                    side=side,
                    avg_price=avg_price,
                    volume=data["volume"],
                    open_interest=oi,
                    action="stoploss",
                    result="trailingstop"
                )
            except InvalidRequestError as e:
                msg = str(e)
                # ignore both "not modified" and "zero position" errors
                if "not modified" in msg or "zero position" in msg:
                    logger.info(f"[trailing_stop] Пропуск установки стопа для {symbol}: {msg}")
                    return
                else:
                    logger.error(f"[trailing_stop] Ошибка при установке стопа для {symbol}: {msg}", exc_info=True)
                    await self.log_trade(
                        symbol=symbol,
                        side=side,
                        avg_price=avg_price,
                        volume=data["volume"],
                        open_interest=oi,
                        action="stoploss",
                        result="rejected"
                    )
            except Exception as e:
                logger.error(f"[trailing_stop] Непредвиденная ошибка при установке стопа для {symbol}: {e}", exc_info=True)
                await self.log_trade(
                    symbol=symbol,
                    side=side,
                    avg_price=avg_price,
                    volume=data["volume"],
                    open_interest=oi,
                    action="stoploss",
                    result="rejected"
                )

        except Exception as e:
            logger.error(f"[trailing_stop] Critical error: {e}", exc_info=True)
            await self.notify_user(f"🚨 Ошибка трейлинг-стопа {symbol}: {e}")



    async def get_selected_symbols(self):
        """
        Берёт список ликвидных USDT‑пар. Если REST‑запрос к Bybit
        рвётся (Connection reset и т. п.), пытаемся до трёх раз с
        экспоненциальной задержкой. При полном провале возвращаем
        последний удачный список либо несколько «дефолтных» пар.
        """
        now = time.time()
        self.turnover24h = {}
        fallback = getattr(self, "selected_symbols", ["BTCUSDT", "ETHUSDT"])

        async def safe_fetch():
            for attempt in range(3):
                try:
                    return await asyncio.to_thread(
                        lambda: (
                            self.session.get_tickers(symbol=None, category="linear", settleCoin="USDT"),
                            self.session.get_instruments_info(category="linear"),
                        )
                    )
                except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
                    logger.warning("[get_selected_symbols] REST-ошибка (%d/3): %s", attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)  # 1 s, 2 s, 4 s
            raise RuntimeError("Bybit REST not reachable after 3 attempts")

        try:
            tickers_resp, inst_resp = await safe_fetch()
        except Exception as e:
            logger.error("[get_selected_symbols] %s — использую fallback‑список %s", e, fallback)
            return fallback

        # Проверяем структуры ответов
        if not isinstance(tickers_resp, dict) or "result" not in tickers_resp or "list" not in tickers_resp["result"]:
            logger.error("[get_selected_symbols] Некорректный ответ get_tickers.")
            return fallback

        if not isinstance(inst_resp, dict) or "result" not in inst_resp or "list" not in inst_resp["result"]:
            logger.error("[get_selected_symbols] Некорректный ответ get_instruments_info.")
            return fallback

        # Обновляем список раз в час либо если он ещё не был создан
        if now - getattr(self, "last_asset_selection_time", 0) >= 3600 or not getattr(self, "selected_symbols", []):
            tickers_data = tickers_resp["result"]["list"]
            instruments_data = inst_resp["result"]["list"]

            trading_status = {
                inst.get("symbol"): inst.get("status", "").upper() == "TRADING"
                for inst in instruments_data if inst.get("symbol")
            }

            usdt_pairs = []
            for tk in tickers_data:
                sym = tk.get("symbol")
                if not sym:
                    continue
                if "USDT" in sym and "BTC" not in sym and "ETH" not in sym:
                    if not trading_status.get(sym, False):
                        continue
                    turnover24 = safe_to_decimal(tk.get("turnover24h", "0"))
                    volume24 = safe_to_decimal(tk.get("volume24h", "0"))
                    # Debug log for specific symbols
                    if sym in ("XRPUSDT", "CYBERUSDT"):
                        logger.info(
                            f"[get_selected_symbols] Debug {sym}: trading_status={trading_status.get(sym)}, "
                            f"turnover24h={turnover24}, volume24h={volume24}"
                        )
                    if turnover24 >= 2_000_000 and volume24 >= 1_500_000:
                        self.turnover24h[sym] = turnover24
                        usdt_pairs.append(sym)

            self.selected_symbols = usdt_pairs

            # ── кэшируем шаг лота и минимальный объём ────────────────────
            self.qty_step_map = {
                inst["symbol"]: safe_to_decimal(
                    inst.get("lotSizeFilter", {}).get("qtyStep", "0.001")
                )
                for inst in instruments_data
                if inst.get("symbol")
            }
            self.min_qty_map = {
                inst["symbol"]: safe_to_decimal(
                    inst.get("lotSizeFilter", {}).get("minOrderQty", "0.001")
                )
                for inst in instruments_data
                if inst.get("symbol")
            }
            
            self.last_asset_selection_time = now
            logger.info("[get_selected_symbols] Пары выбраны (%d): %s", len(self.selected_symbols), self.selected_symbols)

        return getattr(self, "selected_symbols", fallback)


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
                if safe_to_decimal(pos.get("size", 0)) > 0
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
                        "avg_price": safe_to_decimal(pos.get("entryPrice") or pos.get("avgPrice")),
                        "side":      pos.get("side", ""),
                        "pos_idx":   pos.get("positionIdx", 1),
                        "volume":    safe_to_decimal(pos.get("size", 0)),
                        "amount":    safe_to_decimal(pos.get("positionValue", 0))
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


def load_golden_params(csv_path: str = "golden_params.csv") -> dict:
    from decimal import Decimal
    # Default global parameters
    default_params = {
        "Buy": {
            "period_iters": 4,
            "price_change": 0.2,      # +0.20 % price rise
            "volume_change": 50,      # +50 % volume surge
            "oi_change": 0.04,         # +0.40 % OI rise
        },
        "Sell": {
            "period_iters": 4,
            "price_change": 0.5,      # −0.50 % price drop
            "volume_change": 30,      # +30 % volume surge
            "oi_change": 0.08,         # +0.80 % OI rise
        },
        "Sell2": {                # альтернативный шорт‑сетап  «либо‑либо»
        "period_iters": 4,    # 4 последних свечи
        "price_change": 0.02,# падение цены ≥ 0.075 %
        "volume_change": -50, # падение объёма ≥ 80 %
        "oi_change": -0.01,      # падение OI ≥ 1 %
    },

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
                    "price_change": safe_to_decimal(row["price_change"]),
                    "volume_change": safe_to_decimal(row["volume_change"]),
                    "oi_change": safe_to_decimal(row["oi_change"]),
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

#     symbols = await bots[0].get_selected_symbols() if bots else []
#     shared_ws = PublicWebSocketManager(symbols=symbols)
#     # backfill historical candles
#     await shared_ws.backfill_history()
#     public_ws_task = asyncio.create_task(shared_ws.start())

#     # Подключаем shared_ws ко всем ботам
#     for bot in bots:
#         bot.shared_ws = shared_ws

#     bot_tasks = [asyncio.create_task(bot.start()) for bot in bots]
#     dp.include_router(router)
#     dp.include_router(router_admin)
#     # ── patch: restore logging handlers that aiogram wipes out ──
#     def _restore_logging_handlers() -> None:
#         """
#         Aiogram re‑configures the root logger inside `start_polling`
#         and removes our file/stream handlers.  Re‑add them once the
#         event‑loop has handed control back.
#         """
#         root = logging.getLogger()
#         if root.handlers:
#             # handlers already present – nothing to do
#             return

#         fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

#         fh = logging.FileHandler("bot.log", encoding="utf-8")
#         fh.setFormatter(fmt)

#         sh = logging.StreamHandler()
#         sh.setFormatter(fmt)

#         root.addHandler(fh)
#         root.addHandler(sh)
#         root.setLevel(logging.INFO)
#         root.info("[logging-patch] handlers restored")

#     # schedule the restoration right after aiogram finishes its own setup
#     asyncio.get_running_loop().call_later(0, _restore_logging_handlers)
#     tg_task = asyncio.create_task(dp.start_polling(telegram_bot))
#     await asyncio.gather(public_ws_task, *bot_tasks, tg_task)

async def run_all():
    users = load_users_from_json("user_state.json")
    if not users:
        print("❌ Нет активных пользователей для запуска.")
        return
    
    golden_param_store = load_golden_params()
    bots = [TradingBot(user_data=u, shared_ws=None, golden_param_store=golden_param_store) for u in users]
    symbols = await bots[0].get_selected_symbols() if bots else []
    shared_ws = PublicWebSocketManager(symbols=symbols)
    
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
    dp.include_router(router)
    dp.include_router(router_admin)
    
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
    #print("Логирование настроено: файл bot.log и консоль")
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")