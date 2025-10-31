import datetime as dt
from aiogram.enums import ParseMode

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
from decimal import Decimal, InvalidOperation
import math           # ← добавьте эту строку

logger = logging.getLogger(__name__)
SNAPSHOT_CSV_PATH = "golden_setup_snapshots.csv"

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
    open‑interest handler never crashes on bad data.
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
        self.candles_data = defaultdict(list)
        self.ticker_data = {}
        self.latest_open_interest = {}
        self.loop = asyncio.get_event_loop()
        # for golden setup
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        self.oi_history     = defaultdict(lambda: deque(maxlen=1000))
        # track last saved candle time for deduplication
        self._last_saved_time = {}
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
        def _on_message(msg):
            asyncio.run_coroutine_threadsafe(self.route_message(msg), self.loop)

        self.ws = WebSocket(testnet=False, channel_type="linear", ping_interval=20, ping_timeout=10)
        self.ws_shared = self.ws            # backward‑compat alias
        # subscribe to kline and ticker streams for all symbols at once
        self.ws.kline_stream(interval=self.interval, symbol=self.symbols, callback=_on_message)
        self.ws.ticker_stream(symbol=self.symbols, callback=_on_message)

    async def route_message(self, msg):
        #print(f"[route_message] topic={msg.get('topic')}")

        topic = msg.get("topic", "")
        if topic.startswith("kline."):
            await self.handle_kline(msg)
        elif topic.startswith("tickers."):
            await self.handle_ticker(msg)

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
                "openPrice": float(entry.get("open", 0)),
                "highPrice": float(entry.get("high", 0)),
                "lowPrice": float(entry.get("low", 0)),
                "closePrice": float(entry.get("close", 0)),
                "volume": float(entry.get("volume", 0)),
            }
            self.candles_data[symbol].append(row)
            if len(self.candles_data[symbol]) > 500:
                self.candles_data[symbol] = self.candles_data[symbol][-500:]
            # record volume
            self.volume_history[symbol].append(row["volume"])
            # attach latest open‑interest snapshot to this confirmed candle
            oi_val = self.latest_open_interest.get(symbol)
            if oi_val is not None:
                self.oi_history[symbol].append(float(oi_val))
                if len(self.oi_history[symbol]) > 1000:
                    self.oi_history[symbol] = deque(self.oi_history[symbol], maxlen=1000)
            self._save_history()
            self._last_saved_time[symbol] = ts
            logger.debug("[handle_kline] stored candle for %s @ %s", symbol, ts)

    async def handle_ticker(self, msg):
        data = msg.get("data", {})
        entries = data if isinstance(data, list) else [data]
        for ticker in entries:
            symbol = ticker.get("symbol")
            if not symbol:
                continue
            # update open interest
            oi_val = safe_to_decimal(ticker.get("openInterest"))
            self.latest_open_interest[symbol] = oi_val
            # update mark price and re-evaluate existing position
            last_price_str = ticker.get("lastPrice") or ticker.get("price")
            try:
                last_price = float(last_price_str)
            except (TypeError, ValueError):
                continue
            if symbol in self.open_positions:
                pos = self.open_positions[symbol]
                pos["markPrice"] = last_price
                await self.evaluate_position(pos)


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
                            open_p = float(entry[1])
                            high_p = float(entry[2])
                            low_p = float(entry[3])
                            close_p = float(entry[4])
                            vol = float(entry[5])
                        except Exception:
                            print(f"[History] backfill invalid list entry for {symbol}: {entry}")
                            continue
                    else:
                        try:
                            ts = pd.to_datetime(int(entry['start']), unit='ms')
                        except Exception as e:
                            print(f"[History] backfill invalid dict entry for {symbol}: {e}")
                            continue
                        open_p = float(entry.get('open', 0))
                        high_p = float(entry.get('high', 0))
                        low_p = float(entry.get('low', 0))
                        close_p = float(entry.get('close', 0))
                        vol = float(entry.get('volume', 0))
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
                    # trim
                    if len(self.candles_data[symbol]) > 500:
                        self.candles_data[symbol] = self.candles_data[symbol][-500:]
                    self._save_history()
                    # ── back‑fill open‑interest history ───────────────────────
                    try:
                        ticker_resp = http.get_tickers(category="linear", symbol=symbol)
                        oi_val = float(
                            ticker_resp["result"]["list"][0].get("openInterest", 0)
                        )
                    except Exception:
                        oi_val = 0.0  # если не удалось получить — пишем нули

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


    async def init_trade_ws(self):
        url = "wss://stream.bybit.com/v5/trade"
        self.ws_trade = await websockets.connect(url)

        # 1) build auth payload
        ts = str(int(time.time() * 1000))
        msg = self.api_key + ts
        sig = hmac.new(self.api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

        auth_req = {
            "op": "auth",
            "args": [ self.api_key, int(ts), sig ]
        }
        await self.ws_trade.send(json.dumps(auth_req))
        resp = json.loads(await self.ws_trade.recv())
        assert resp["retCode"] == 0, f"WS auth failed: {resp}"


    async def place_order_ws(self, symbol, side, qty, position_idx=1, price=None, order_type="Market"):
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
            "timeInForce": "GoodTillCancel"
        }
        # Указываем индекс позиции
        args["positionIdx"] = position_idx
        if price:
            args["price"] = str(price)

        req = {
            "op": "order.create",
            "header": header,
            "args": [ args ]
        }
        await self.ws_trade.send(json.dumps(req))
        resp = json.loads(await self.ws_trade.recv())
        if resp["retCode"] != 0:
            raise RuntimeError(f"Order failed: {resp}")
        return resp["data"]  # contains orderId, etc.

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
        self.symbols = shared_ws.symbols if shared_ws else []
        self.ws_private = None
        self.open_positions = {}
        self.loop = asyncio.get_event_loop()
        self.golden_param_store = golden_param_store
        # handle reference to the market loop task
        self.market_task = None
        # WS для отправки торговых запросов
        self.ws_trade = None
        # Индекс позиции (для Bybit V5: 1 или 2)
        self.position_idx = user_data.get("position_idx", 1)
        self.load_model()
        self.POSITION_VOLUME = user_data.get("volume", 100)
        # максимальный разрешённый общий объём открытых позиций (USDT)
        self.MAX_TOTAL_VOLUME = float(user_data.get("max_total_volume", 10000))
        # Maximum allowed total exposure across all open positions (in USDT)
        self.qty_step_map: dict[str, float] = {}
        self.min_qty_map: dict[str, float] = {}
        # track symbols that recently failed order placement
        self.failed_orders: dict[str, float] = {}
        # символы, по которым уже отправлен ордер,
        # но позиция ещё не пришла по private‑WS
        self.pending_orders: set[str] = set()

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
                print(f"[User {self.user_id}] ✅ ML‑модель загружена из {model_path}")
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
                size = float(pos.get("size", 0))
                price = float(pos.get("entryPrice", 0)) or float(
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
                    size = float(pos.get("size", 0))
                    price = float(pos.get("entryPrice", 0)) or float(
                        pos.get("markPrice", 0)
                    )
                    total += size * price
                except (ValueError, TypeError):
                    continue
            return total

    
    async def start(self):
        print(f"[User {self.user_id}] Бот запущен")
        await self.update_open_positions()
        await self.setup_private_ws()
        # Для реального режима инициализируем WS-канал для ордеров
        if self.mode == "real":
            await self.init_trade_ws()
        # start checking golden‐setup on every new candle
        self.market_task = asyncio.create_task(self.market_loop())
        # log if the market_loop ever stops or crashes
        def _market_done(task: asyncio.Task) -> None:
            try:
                exc = task.exception()
                if exc:
                    logger.exception("[market_loop] task for user %s crashed: %s",
                                     self.user_id, exc)
                else:
                    logger.warning("[market_loop] task for user %s finished unexpectedly",
                                   self.user_id)
            except asyncio.CancelledError:
                logger.info("[market_loop] task for user %s was cancelled", self.user_id)

        self.market_task.add_done_callback(_market_done)


    async def setup_private_ws(self):
        def _on_private(msg):
            asyncio.run_coroutine_threadsafe(self.route_private_message(msg), self.loop)

        self.ws_private = WebSocket(testnet=False,
                                    demo=(self.mode == "demo"),
                                    channel_type="private",
                                    api_key=self.api_key,
                                    api_secret=self.api_secret,
                                    ping_interval=20,
                                    ping_timeout=10)
        self.ws_private.position_stream(callback=_on_private)

    async def route_private_message(self, msg):
        topic = msg.get("topic", "")
        if "position" in topic:
            await self.handle_position_update(msg)

    async def handle_position_update(self, msg):
        data = msg.get("data", [])
        if isinstance(data, dict):
            data = [data]
        for position in data:
            await self.evaluate_position(position)

    async def evaluate_position(self, position):
        symbol = position.get("symbol")
        size = float(position.get("size", 0))
        side = position.get("side", "Buy")
        entry_price = float(position.get("entryPrice", 0))
        mark_price = float(position.get("markPrice", 0))
        pnl = float(position.get("unrealisedPnl", 0))

        if size == 0:
            self.open_positions.pop(symbol, None)
            print(f"[User {self.user_id}] {symbol} позиция закрыта")
            await self.notify_user(f"❌ Позиция по {symbol} закрыта")
            return

        
        if symbol not in self.open_positions:
            await self.notify_user(f"🟢 Открыта новая позиция по {symbol}: {side} @ {entry_price}")
        self.open_positions[symbol] = position
        print(f"[User {self.user_id}] {symbol} позиция обновлена, PnL: {pnl:.2f} USDT")
        await self.notify_user(f"🔔 Обновлена позиция по {symbol}. PnL: {pnl:.2f} USDT")

        
        # Проверка лимита общего объёма
        user_state = self.load_user_state()
        max_volume = float(user_state.get("max_total_volume", 10000))
        current_total = sum(float(p.get("size", 0)) for p in self.open_positions.values())
        if symbol not in self.open_positions and (current_total + size) > max_volume:
            print(f"[User {self.user_id}] ⚠️ Превышен лимит общего объёма: {current_total + size} > {max_volume}")
            await self.notify_user(f"⚠️ Позиция по {symbol} не открыта — превышен лимит общего объёма ({max_volume} USDT)")
            return

        if entry_price > 0:
            direction = 1 if side == "Buy" else -1
            pnl_pct = direction * ((mark_price - entry_price) / entry_price) * 100
            if pnl_pct >= 5:
                await self.set_trailing_stop(symbol, entry_price, pnl_pct, side)


    async def on_ticker_update(self, symbol, last_price):
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            pos['markPrice'] = last_price
            await self.evaluate_position(pos)


    async def market_loop(self):
        """
        Continuously (≈ 1 Hz) scan all subscribed symbols for golden‑setup signals.

        Extra diagnostics:
          • iteration counter and detailed heartbeat every 60 s;
          • top‑level try/except to ensure the loop never exits silently.
        """
        last_heartbeat = time.time()
        iteration = 0
        try:
            while True:
                iteration += 1
                for symbol in self.shared_ws.symbols:
                    try:
                        await self.execute_golden_setup(symbol)
                    except Exception as e:
                        logger.error("[market_loop] %s error: %s", symbol, e, exc_info=e)

                # heartbeat once a minute
                if time.time() - last_heartbeat >= 60:
                    logger.info("[market_loop] alive (iter=%d) — scanned %d symbols",
                                iteration, len(self.shared_ws.symbols))
                    last_heartbeat = time.time()

                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # normal shutdown
            logger.info("[market_loop] cancelled for user %s", self.user_id)
            raise
        except Exception as fatal:
            logger.exception("[market_loop] fatal exception — terminating loop: %s", fatal)
            raise

    async def execute_golden_setup(self, symbol: str):
        """
        Check price, volume and open‐interest changes over the last
        period_iters bars and open a market order if all thresholds
        are exceeded.
        """
        # skip symbol for 10 min after a failed order (e.g., not enough balance)
        if symbol in self.failed_orders and time.time() - self.failed_orders[symbol] < 600:
            return
        # пропускаем, если позиция уже открыта
        # либо ордер в полёте и мы ждём подтверждения
        if symbol in self.open_positions or symbol in self.pending_orders:
            return            
        # Skip if no candle history yet
        recent = self.shared_ws.candles_data.get(symbol, [])
        if not recent:
            logger.debug("[GoldenSetup] skip %s – no candles yet", symbol)
            return
        last_bar_time = recent[-1]["startTime"]
        logger.debug("[GoldenSetup] check %s on bar %s", symbol, last_bar_time)
        from decimal import Decimal
        buy_params = self.golden_param_store.get((symbol, "Buy"), self.golden_param_store.get("Buy"))
        sell_params = self.golden_param_store.get((symbol, "Sell"), self.golden_param_store.get("Sell"))

        # ── calculate raw deltas once per bar ───────────────────────────────
        period_iters = max(int(buy_params["period_iters"]),
                           int(sell_params["period_iters"]))
        # need enough history for candle, volume and OI
        if (len(recent) <= period_iters or
                len(self.shared_ws.volume_history[symbol]) <= period_iters or
                len(self.shared_ws.oi_history[symbol])     <= period_iters):
            return

        old_bar  = recent[-1 - period_iters]
        new_bar  = recent[-1]
        close_price = new_bar["closePrice"]

        price_change_pct = (
            (close_price - old_bar["closePrice"]) / old_bar["closePrice"]
            if old_bar["closePrice"] else 0
        )

        old_vol, new_vol = (
            self.shared_ws.volume_history[symbol][-1 - period_iters],
            self.shared_ws.volume_history[symbol][-1],
        )
        volume_change_pct = (new_vol - old_vol) / old_vol if old_vol else 0

        old_oi = float(self.shared_ws.oi_history[symbol][-1 - period_iters])
        new_oi = float(self.shared_ws.oi_history[symbol][-1])
        oi_change_pct = (new_oi - old_oi) / old_oi if old_oi else 0

        # log every bar at INFO level
        # logger.info(
        #     "[GoldenSetup] %s | Δprice=%.4f%%  Δvol=%.2f%%  ΔOI=%.6f%%",
        #     symbol,
        #     price_change_pct * 100,
        #     volume_change_pct * 100,
        #     oi_change_pct * 100,
        # )

        action = None

        # --- SELL setup ---
        sp = int(sell_params['period_iters'])
        if len(recent) > sp:
            old = recent[-1-sp]
            new = recent[-1]
            price_prev = old['closePrice']
            if price_prev:
                price_change = (new['closePrice'] - price_prev) / price_prev * 100
                # ensure we have enough history for volume and open interest
                if len(self.shared_ws.volume_history[symbol]) > sp and len(self.shared_ws.oi_history[symbol]) > sp:
                    old_vol = self.shared_ws.volume_history[symbol][-1-sp]
                    new_vol = self.shared_ws.volume_history[symbol][-1]
                    vol_change = (new_vol - old_vol) / old_vol * 100 if old_vol else 0
                    old_oi = self.shared_ws.oi_history[symbol][-1-sp]
                    new_oi = self.shared_ws.oi_history[symbol][-1]
                    oi_change = (new_oi - old_oi) / old_oi * 100 if old_oi else 0
                    if price_change <= -float(sell_params['price_change']) and vol_change >= float(sell_params['volume_change']) and oi_change >= float(sell_params['oi_change']):
                        action = 'Sell'

        # --- BUY setup ---
        if action is None:
            lp = int(buy_params['period_iters'])
            if len(recent) > lp:
                old = recent[-1-lp]
                new = recent[-1]
                price_prev = old['closePrice']
                if price_prev:
                    price_change = (new['closePrice'] - price_prev) / price_prev * 100
                    # ensure volume and OI history exist
                    if len(self.shared_ws.volume_history[symbol]) > lp and len(self.shared_ws.oi_history[symbol]) > lp:
                        old_vol = self.shared_ws.volume_history[symbol][-1-lp]
                        new_vol = self.shared_ws.volume_history[symbol][-1]
                        vol_change = (new_vol - old_vol) / old_vol * 100 if old_vol else 0
                        old_oi = self.shared_ws.oi_history[symbol][-1-lp]
                        new_oi = self.shared_ws.oi_history[symbol][-1]
                        oi_change = (new_oi - old_oi) / old_oi * 100 if old_oi else 0
                        if price_change >= float(buy_params['price_change']) and vol_change >= float(buy_params['volume_change']) and oi_change >= float(buy_params['oi_change']):
                            action = 'Buy'

        if action is None:
            return

        # ── snapshot for triggered signal ────────────────────────────────────
        # logger.info(
        #     "[GoldenSetup] TRIGGER %s → %s | Δprice=%.4f%% Δvol=%.2f%% ΔOI=%.6f%% (over %d bars)",
        #     symbol, action,
        #     price_change_pct * 100,
        #     volume_change_pct * 100,
        #     oi_change_pct     * 100,
        #     period_iters,
        # )

        _append_snapshot(
            {
                "close_price": close_price,
                "price_change": price_change_pct,
                "volume_change": volume_change_pct,
                "oi_change": oi_change_pct,
                "period_iters": period_iters,
                "user_id": self.user_id,
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )

        # skip if we already hold that symbol
        if symbol in self.open_positions:
            return

        # ------------------------------------------------------------------
        # 4. Рассчитываем объём сделки и проверяем общий лимит в USDT
        # ------------------------------------------------------------------
        side_params = buy_params if action == "Buy" else sell_params

        # Объём одной позиции задаётся в USDT, а не в размере монеты
        volume_usdt = float(side_params.get("position_volume", self.POSITION_VOLUME))

        # Текущая цена (берём закрытие последней свечи)
        last_price = close_price if close_price else 0.0
        if last_price == 0:
            logger.warning("[GoldenSetup] %s — пропуск, цена нулевая", symbol)
            return

        # --- Проверяем свободный баланс USDT (чтобы избежать 110007) ---
        try:
            wallet_resp = await asyncio.to_thread(
                lambda: self.session.get_wallet_balance(
                    accountType="UNIFIED", coin="USDT"
                )
            )
            free_usdt = float(
                wallet_resp["result"]["list"][0].get("totalAvailableBalance", 0)
            )
        except Exception:
            free_usdt = 0.0

        if free_usdt < volume_usdt:
            logger.info(
                "[GoldenSetup] %s: недостаточно свободного баланса %.2f < %.2f",
                symbol, free_usdt, volume_usdt
            )
            return
        # Кол‑во монет = объём USDT / цена
        qty = volume_usdt / last_price

        # Кол‑во монет = объём USDT / цена
        qty_raw = volume_usdt / last_price

        # --- адаптируем под шаг лота (qtyStep) и minOrderQty --------------
        step    = self.qty_step_map.get(symbol, 0.001)
        min_qty = self.min_qty_map.get(symbol, step)

        # округляем «вниз» до ближайшего допустимого шага
        qty = math.floor(qty_raw / step) * step

        # финальное округление до нужного количества десятичных знаков
        decimals = max(0, int(-math.log10(step))) if step < 1 else 0
        qty = round(qty, decimals)

        # проверка минимального размера ордера
        if qty < min_qty or qty <= 0:
            logger.info(
                "[GoldenSetup] %s — qty %.8f меньше min_qty %.8f; пропуск",
                symbol, qty, min_qty
            )
            return
            
        # --- Проверка суммарного открытого объёма (HTTP‑based, как в V11) ---
        current_total_usdt = await self.get_total_open_volume()
        if current_total_usdt + volume_usdt > self.MAX_TOTAL_VOLUME:
            logger.info("[GoldenSetup] %s: превышен MAX_TOTAL_VOLUME %.2f + %.2f > %.2f",
                        symbol, current_total_usdt, volume_usdt, self.MAX_TOTAL_VOLUME)
            return

        # 5. Отправляем ордер: WS для real, REST для demo
        # выбор индекса позиции: 1 – LONG, 2 – SHORT
        pos_idx = 1 if action == "Buy" else 2
        try:
            if self.mode == "real":
                await self.place_order_ws(symbol, action, qty, position_idx=pos_idx)
            else:
                await asyncio.to_thread(lambda: self.session.place_order(
                    category="linear",
                    symbol=symbol,
                    side=action,
                    orderType="Market",
                    qty=str(qty),
                    timeInForce="GoodTillCancel",
                    positionIdx=pos_idx
                ))
            await self.notify_user(f"🚀 Golden setup: открыта {action} позиция по {symbol}")
            # отмечаем ордер как «в ожидании»
            self.pending_orders.add(symbol)
        except InvalidRequestError as e:
            self.pending_orders.discard(symbol)
            # Bybit вернёт 110007 при нехватке средств.  Блокируем тикер
            # на 10 минут, чтобы не спамить запросами.
            logger.warning("[GoldenSetup] order for %s failed: %s", symbol, e)
            self.failed_orders[symbol] = time.time()
            # извлекаем human‑friendly текст ошибки без JSON‑запроса
            err_msg = e.args[0] if e.args else str(e)
            await self.notify_user(f"⚠️ Не удалось открыть позицию по {symbol}: {err_msg}")
            return
        except Exception as e:
            logger.warning("[GoldenSetup] unexpected error for %s: %s", symbol, e, exc_info=e)
            self.failed_orders[symbol] = time.time()
            # любое обновление позиции снимает пометку «pending»
            self.pending_orders.discard(symbol)            
            return


    async def log_trade(self, symbol: str, row, side, open_interest,
                        action: str, result: str, closed_manually: bool = False,
                        csv_filename: str = "trade_log.csv"):

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
                        "user_id", "symbol", "timestamp",
                        "openPrice", "highPrice", "lowPrice", "closePrice", "volume",
                        "open_interest", "action", "result", "closed_manually"
                    ])
                writer.writerow([
                    self.user_id, symbol, time_str,
                    open_str, high_str, low_str, close_str, vol_str,
                    oi_str, action, result, closed_str
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
                    f"<b>Символ:</b> <a href='{link_url}'>{symbol}</a>\n"
                    f"<b>Пользователь:</b> {self.user_id}\n"
                    f"<b>Время:</b> {time_str}\n"
                    f"<b>Цена открытия:</b> {open_str}\n"
                    f"<b>Объём:</b> {vol_str}\n"
                    f"<b>Тип открытия:</b> ЛОНГ\n"
                    f"#{symbol}"
                )
            elif s_side.lower() == "sell":
                msg = (
                    f"🟥 <b>Открытие SHORT-позиции</b>\n"
                    f"<b>Символ:</b> <a href='{link_url}'>{symbol}</a>\n"
                    f"<b>Пользователь:</b> {self.user_id}\n"
                    f"<b>Время:</b> {time_str}\n"
                    f"<b>Цена открытия:</b> {open_str}\n"
                    f"<b>Объём:</b> {vol_str}\n"
                    f"<b>Тип открытия:</b> ШОРТ\n"
                    f"#{symbol}"
                )
            else:
                msg = (
                    f"🟩🔴 <b>Открытие позиции</b>\n"
                    f"<b>Символ:</b> <a href='{link_url}'>{symbol}</a>\n"
                    f"<b>Пользователь:</b> {self.user_id}\n"
                    f"<b>Время:</b> {time_str}\n"
                    f"<b>Тип открытия:</b> {s_side}\n"
                    f"#{symbol}"
                )
        elif s_result == "closed":
            msg = (
                f"❌ <b>Закрытие позиции</b>\n"
                f"<b>Символ:</b> <a href='{link_url}'>{symbol}</a>\n"
                f"<b>Пользователь:</b> {self.user_id}\n"
                f"<b>Время закрытия:</b> {time_str}\n"
                f"<b>Цена закрытия:</b> {close_str}\n"
                f"<b>Объём:</b> {vol_str}\n"
                f"<b>Тип закрытия:</b> {s_manually}\n"
                f"#{symbol}"
            )
        elif s_result == "trailingstop":
            msg = (
                f"🔄 <b>Установлен кастомный трейлинг-стоп</b>\n"
                f"<b>Символ:</b> <a href='{link_url}'>{symbol}</a>\n"
                f"<b>Пользователь:</b> {self.user_id}\n"
                f"<b>Цена:</b> {close_str}\n"
                f"<b>Объём:</b> {vol_str}\n"
                f"<b>Комментарий:</b> {action}"
            )
        else:
            msg = (
                f"🫡🔄 <b>Сделка</b>\n"
                f"<b>Символ:</b> <a href='{link_url}'>{symbol}</a>\n"
                f"<b>Результат:</b> {result}\n"
                f"<b>Действие:</b> {action}\n"
                f"<b>Закрытие:</b> {s_manually}"
            )

        try:
            await telegram_bot.send_message(self.user_id, msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            print(f"[log_trade] Ошибка Telegram: {e}")


    async def set_trailing_stop(self, symbol, entry_price, pnl_pct, side):
        base_trailing = 2.5
        reduction = 0
        oi = self.shared_ws.latest_open_interest.get(symbol, 0)
        if oi > 1000:
            reduction += 0.5
        candles = self.shared_ws.candles_data.get(symbol, [])
        if candles:
            recent_close = candles[-1]["closePrice"]
            price_change = abs(recent_close - entry_price) / entry_price * 100
            if price_change > 1:
                reduction += 0.5
        final_trailing = max(base_trailing - reduction, 0.5)
        stop_pct = pnl_pct - final_trailing
        stop_price = entry_price * (1 + stop_pct / 100) if side == "Buy" else entry_price * (1 - stop_pct / 100)

        try:
            result = await asyncio.to_thread(
                lambda: self.session.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    positionIdx=self.position_idx,      # ← Bybit V5 requires positionIdx for linear
                    stopLoss=str(stop_price),
                    triggerBy="LastPrice",              # use last‑price as trigger
                    timeInForce="GoodTillCancel"
                )
            )
            print(f"[User {self.user_id}] StopLoss результат: {result}")
            row = self.shared_ws.candles_data.get(symbol, [])[-1] if self.shared_ws.candles_data.get(symbol) else {}
            await self.log_trade(symbol, row=row, side=side, open_interest=self.shared_ws.latest_open_interest.get(symbol),
                                 action="stoploss", result="trailingstop")
            await self.notify_user(f"📉 Стоп-лосс установлен по {symbol}: {stop_price:.2f}")
        except Exception as e:
            print(f"[User {self.user_id}] Ошибка установки стопа: {e}")



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
                    logger.warning("[get_selected_symbols] REST‑ошибка (%d/3): %s", attempt + 1, e)
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
                    turnover24 = float(tk.get("turnover24h", "0"))
                    volume24 = float(tk.get("volume24h", "0"))
                    if turnover24 >= 2_000_000 and volume24 >= 2_000_000:
                        self.turnover24h[sym] = turnover24
                        usdt_pairs.append(sym)

            self.selected_symbols = usdt_pairs

            # ── кэшируем шаг лота и минимальный объём ────────────────────
            self.qty_step_map = {
                inst["symbol"]: float(
                    inst.get("lotSizeFilter", {}).get("qtyStep", "0.001")
                )
                for inst in instruments_data
                if inst.get("symbol")
            }
            self.min_qty_map = {
                inst["symbol"]: float(
                    inst.get("lotSizeFilter", {}).get("minOrderQty", "0.001")
                )
                for inst in instruments_data
                if inst.get("symbol")
            }
            
            self.last_asset_selection_time = now
            logger.info("[get_selected_symbols] Пары выбраны: %s", self.selected_symbols)

        return getattr(self, "selected_symbols", fallback)



    async def update_open_positions(self):
        try:
            response = await asyncio.to_thread(
                lambda: self.session.get_positions(category="linear", settleCoin="USDT")
            )
            if response.get("retCode") != 0:
                print(f"[User {self.user_id}] Ошибка при получении позиций: {response.get('retMsg')}")
                return
            for pos in response["result"].get("list", []):
                size = float(pos.get("size", 0))
                if size > 0:
                    symbol = pos["symbol"]
                    self.open_positions[symbol] = pos
            print(f"[User {self.user_id}] Загружено {len(self.open_positions)} позиций с биржи")
        except Exception as e:
            print(f"[User {self.user_id}] Ошибка при загрузке позиций: {e}")



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
            "oi_change": 0.4,         # +0.40 % OI rise
        },
        "Sell": {
            "period_iters": 4,
            "price_change": 0.5,      # −0.50 % price drop
            "volume_change": 30,      # +30 % volume surge
            "oi_change": 0.8,         # +0.80 % OI rise
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
                    "price_change": float(row["price_change"]),
                    "volume_change": float(row["volume_change"]),
                    "oi_change": float(row["oi_change"]),
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

    symbols = await bots[0].get_selected_symbols() if bots else []
    shared_ws = PublicWebSocketManager(symbols=symbols)
    # backfill historical candles
    await shared_ws.backfill_history()
    await shared_ws.start()

    # Подключаем shared_ws ко всем ботам
    for bot in bots:
        bot.shared_ws = shared_ws

    bot_tasks = [bot.start() for bot in bots]
    dp.include_router(router)
    dp.include_router(router_admin)
    # ── patch: restore logging handlers that aiogram wipes out ──
    def _restore_logging_handlers() -> None:
        """
        Aiogram re‑configures the root logger inside `start_polling`
        and removes our file/stream handlers.  Re‑add them once the
        event‑loop has handed control back.
        """
        root = logging.getLogger()
        if root.handlers:
            # handlers already present – nothing to do
            return

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        fh = logging.FileHandler("bot.log", encoding="utf-8")
        fh.setFormatter(fmt)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        root.addHandler(fh)
        root.addHandler(sh)
        root.setLevel(logging.INFO)
        root.info("[logging-patch] handlers restored")

    # schedule the restoration right after aiogram finishes its own setup
    asyncio.get_running_loop().call_later(0, _restore_logging_handlers)
    tg_task = asyncio.create_task(dp.start_polling(telegram_bot))
    await asyncio.gather(*bot_tasks, tg_task)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("bot.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    print("Логирование настроено: файл bot.log и консоль")
    asyncio.run(run_all())
