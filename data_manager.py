# data_manager.py
import asyncio
import logging
import time
import pickle
import random 
import csv
import os
import tempfile
import datetime as dt
from collections import defaultdict, deque
import collections
import pandas as pd
import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan

from ta_compat import ta
from pybit.unified_trading import WebSocket, HTTP
import requests
from urllib3.exceptions import ReadTimeoutError as UrllibReadTimeoutError

import config
import utils
from websocket_monitor import get_monitor

logger = logging.getLogger(__name__)

LARGE_TURNOVER = 300_000_000
MID_TURNOVER   = 10_000_000
VOL_WINDOW     = 60
VOL_COEF       = 1.2
LISTING_AGE_MIN_MINUTES = 1400

# ... (функция compute_supertrend без изменений) ...
def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float | int = 3):
    if len(df) < (period + 1) * 2:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    high = df["highPrice"].astype("float32")
    low = df["lowPrice"].astype("float32")
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


class PublicWebSocketManager:
    def __init__(self, symbols, interval="1"):
        self.symbols = symbols
        self.interval = interval
        self.ws = None
        self.candles_data = defaultdict(lambda: deque(maxlen=3000))
        self.ticker_data = {}
        self.latest_open_interest = {}
        self.active_symbols = set()
        self.ready_event = asyncio.Event()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        self.volume_history = defaultdict(lambda: deque(maxlen=1000))
        self.oi_history = defaultdict(lambda: deque(maxlen=1000))
        self.cvd_history = defaultdict(lambda: deque(maxlen=1000))
        self.funding_history = defaultdict(lambda: deque(maxlen=3))
        self.depth_history = defaultdict(lambda: deque(maxlen=300))
        self.imbalance_history = defaultdict(lambda: deque(maxlen=300))
        self.overnight_ranges = defaultdict(lambda: deque(maxlen=120))
        self._last_saved_time = {}
        self.position_handlers = []
        self._liq_thresholds = defaultdict(lambda: 5000.0)
        self.last_liq_trade_time = {}
        self._history_file = config.HISTORY_FILE
        self._load_history_from_disk()
        self._load_liq_thresholds()
        self.monitor = get_monitor()
        self.connection_name = "public_websocket"
        self.watchlist = set()

        self._candles_refresh_locks: dict[str, asyncio.Lock] = {}
        
        self.orderbooks = defaultdict(lambda: {'bids': {}, 'asks': {}})
        self.trade_history = defaultdict(lambda: deque(maxlen=200))

    def on_config_reload(self):
        new_history_file = config.HISTORY_FILE
        history_changed = new_history_file != self._history_file
        self._history_file = new_history_file
        if history_changed:
            try:
                self._load_history_from_disk()
            except Exception as e:
                logger.error("Ошибка повторной загрузки истории после обновления config: %s", e, exc_info=True)
        self._load_liq_thresholds()

    async def start(self):
        if not hasattr(self, "_save_task") or self._save_task.done():
            self._save_task = asyncio.create_task(self._save_loop())
            self._watchlist_task = asyncio.create_task(self._update_watchlist_loop())
        
        await self.ready_event.wait()
        await self._connect_websocket()

    async def _connect_websocket(self):
        if self.ws:
            logger.info("Перезапуск Public WebSocket с обновленным списком символов...")
            self.ws.exit()
            await asyncio.sleep(5)

        try:
            def _on_message(msg):
                try:
                    self.monitor.record_message(self.connection_name)
                    if not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(self.route_message(msg), self.loop)
                except Exception as e:
                    logger.warning(f"[PublicWS callback] loop closed, skipping message: {e}")
            
            # --- НАЧАЛО ИЗМЕНЕНИЙ: СТАБИЛИЗАЦИЯ СОЕДИНЕНИЯ ---
            # Увеличиваем таймауты, чтобы дать сети больше времени на ответ
            self.ws = WebSocket(
                testnet=False, channel_type="linear",
                ping_interval=30, # Увеличиваем интервал пингов
                ping_timeout=20,  # Увеличиваем время ожидания ответа
                restart_on_error=True, retries=200,
            )
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            
            if not self.watchlist:
                logger.warning("Watchlist пуст! Подписка только на BTC и ETH.")
                self.watchlist.update(["BTCUSDT", "ETHUSDT"])

            symbols_list = list(self.watchlist)
            logger.info(f"Подписка на данные для {len(symbols_list)} символов из watchlist...")

            self.ws.kline_stream(interval=self.interval, symbol=symbols_list, callback=_on_message)
            self.ws.ticker_stream(symbol=symbols_list, callback=_on_message)
            self.ws.orderbook_stream(depth=50, symbol=symbols_list, callback=_on_message)
            self.ws.trade_stream(symbol=symbols_list, callback=_on_message)
            self.ws.all_liquidation_stream(symbol=symbols_list, callback=_on_message)
            
            self.monitor.register_connection(self.connection_name)
            logger.info("Public WebSocket успешно запущен в фоновом режиме.")

        except asyncio.CancelledError:
            logger.info("Public WS task cancelled.")
            if self.ws: self.ws.exit()
        except Exception as e:
            logger.error(f"Критическая ошибка при запуске Public WS: {e}", exc_info=True)
            if self.ws: self.ws.exit()

    async def _update_watchlist_now(self):
        try:
            await self._initialize_symbol_list()
            
            all_tickers = list(self.ticker_data.values())
            MIN_TURNOVER = 20_000_000
            
            liquid_tickers = {
                t['symbol'] for t in all_tickers 
                if utils.safe_to_float(t.get("turnover24h")) > MIN_TURNOVER
            }
            
            liquid_tickers.add("BTCUSDT")
            liquid_tickers.add("ETHUSDT")

            # --- НАЧАЛО ИЗМЕНЕНИЙ: ИСПРАВЛЕНИЕ ЛОГИКИ ПЕРЕЗАПУСКА ---
            # Сравниваем не сами множества, а их симметрическую разность.
            # Перезапускаем, только если разница непустая.
            if self.watchlist.symmetric_difference(liquid_tickers):
                logger.info(f"Watchlist изменился! Старый: {len(self.watchlist)}, Новый: {len(liquid_tickers)}. Перезапускаем WebSocket.")
                self.watchlist = liquid_tickers
                self.active_symbols = self.watchlist
                if self.ready_event.is_set():
                    asyncio.create_task(self._connect_websocket())
            else:
                 logger.info(f"Watchlist не изменился ({len(self.watchlist)} монет). Перезапуск WS не требуется.")
            # --- КОНЕЦ ИЗМЕНЕНИЙ ---

            if not self.ready_event.is_set():
                self.watchlist = liquid_tickers
                self.active_symbols = self.watchlist
                self.ready_event.set()
        except Exception as e:
            logger.error(f"Ошибка при формировании Watchlist: {e}", exc_info=True)
    
    # ... (остальной код файла без изменений) ...
    async def route_message(self, msg):
        topic = msg.get("topic", "")
        if topic.startswith("kline."):
            await self.handle_kline(msg)
        elif topic.startswith("tickers."):
            await self.handle_ticker(msg)
        elif topic.startswith("orderbook."):
            await self.handle_orderbook(msg)
        elif topic.startswith("publicTrade."):
            await self.handle_trade(msg)
        elif topic.startswith("liquidations."):
            for evt in msg.get("data", []):
                for handler in self.position_handlers:
                    if hasattr(handler, "on_liquidation_event"):
                        asyncio.create_task(handler.on_liquidation_event(evt))

    async def handle_orderbook(self, msg):
        try:
            symbol = msg["topic"].split(".")[-1]
            data = msg.get('data', {})
            self.orderbooks[symbol]['bids'] = {float(price): float(qty) for price, qty in data.get('b', [])}
            self.orderbooks[symbol]['asks'] = {float(price): float(qty) for price, qty in data.get('a', [])}
            bids = self.orderbooks[symbol]["bids"]
            asks = self.orderbooks[symbol]["asks"]

            best_bid = max(bids) if bids else 0.0
            best_ask = min(asks) if asks else 0.0

            depth_span = 10
            bid_depth = sum(qty for _, qty in sorted(bids.items(), reverse=True)[:depth_span])
            ask_depth = sum(qty for _, qty in sorted(asks.items())[:depth_span])
            total_depth = bid_depth + ask_depth
            depth_ratio = (total_depth / max(total_depth, 1e-6))
            book_imbalance = (bid_depth - ask_depth) / max(total_depth, 1e-6)
            self.depth_history[symbol].append({"ts": time.time(), "bid_depth": bid_depth, "ask_depth": ask_depth})
            self.imbalance_history[symbol].append({"ts": time.time(), "imbalance": book_imbalance})

            for bot in list(self.position_handlers):
                if best_bid and best_bid > 0:
                    bot.best_bid_map[symbol] = best_bid
                if best_ask and best_ask > 0:
                    bot.best_ask_map[symbol] = best_ask

                if hasattr(bot, "update_orderbook_metrics"):
                    asyncio.create_task(
                        bot.update_orderbook_metrics(
                            symbol,
                            {
                                "bid_depth": bid_depth,
                                "ask_depth": ask_depth,
                                "depth_ratio": depth_ratio,
                                "orderbook_imbalance": book_imbalance,
                            },
                        )
                    )
        except Exception as e:
            logger.debug(f"Ошибка обработки стакана: {e}")

    async def handle_trade(self, msg):
        """Обрабатывает и сохраняет данные ленты сделок."""
        try:
            symbol = msg["topic"].split(".")[-1]
            trades = msg.get('data', [])
            trade_deque = self.trade_history[symbol]
            for trade in trades:
                # --- НАЧАЛО ИЗМЕНЕНИЯ: Добавляем объем и сторону ---
                trade_deque.append({
                    't': int(trade['T']), 
                    'p': float(trade['p']),
                    'q': float(trade['v']), # Объем сделки
                    'S': trade['S']         # Сторона (Buy или Sell)
                })
                # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        except Exception as e:
            logger.debug(f"Ошибка обработки ленты сделок: {e}")


    async def _initialize_symbol_list(self):
        http = HTTP(testnet=False, timeout=20)
        try:
            tickers_resp = await asyncio.to_thread(lambda: http.get_tickers(category="linear"))
            self.ticker_data.update({tk["symbol"]: tk for tk in tickers_resp["result"]["list"]})
        except Exception as e:
            logger.error(f"Критическая ошибка при получении тикеров: {e}", exc_info=True)

    async def _update_watchlist_loop(self):
        await self._update_watchlist_now()
        while True:
            await asyncio.sleep(60 * 15)
            await self._update_watchlist_now()
            
    def _load_history_from_disk(self):
        try:
            if self._history_file.exists() and self._history_file.stat().st_size > 0:
                with open(self._history_file, 'rb') as f:
                    data = pickle.load(f)
                    default_maxlen = self.candles_data.default_factory().maxlen
                    for sym, rows in data.get('candles', {}).items():
                        self.candles_data[sym] = deque(rows, maxlen=default_maxlen)
                logger.info(f"Загружена история из {self._history_file}")
        except FileNotFoundError:
            logger.info(f"Файл истории {self._history_file} не найден.")
        except Exception as e:
            logger.error(f"Ошибка загрузки истории: {e}", exc_info=True)
            try:
                backup_path = self._history_file.with_suffix(self._history_file.suffix + ".corrupt")
                self._history_file.replace(backup_path)
                logger.warning(f"Поврежденный файл истории перемещен в {backup_path}")
            except Exception as backup_err:
                logger.warning(f"Не удалось переместить поврежденный файл истории: {backup_err}")


    def _load_liq_thresholds(self):
        try:
            self._liq_thresholds.clear()
            if config.LIQ_THRESHOLD_CSV_PATH.exists():
                with open(config.LIQ_THRESHOLD_CSV_PATH, "r", newline="") as f:
                    for row in csv.DictReader(f):
                        self._liq_thresholds[row["symbol"]] = float(row["threshold"])
                logger.info(f"Загружено {len(self._liq_thresholds)} порогов ликвидаций.")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Ошибка загрузки порогов ликвидаций: {e}")

    def stop(self):
        logger.info("Остановка Public WebSocket Manager...")
        if self.ws:
            self.ws.exit()
        if hasattr(self, "_save_task") and not self._save_task.done():
            self._save_task.cancel()
        logger.info("Public WebSocket Manager остановлен.")

    async def handle_kline(self, msg):
        for entry in msg.get("data", []):
            if not entry.get("confirm", False): continue
            symbol = msg["topic"].split(".")[-1]
            try:
                ts = pd.to_datetime(int(entry["start"]), unit="ms")
            except (ValueError, TypeError):
                continue
            if self._last_saved_time.get(symbol) == ts: continue
            
            row = {
                "startTime": ts, "openPrice": utils.safe_to_float(entry.get("open")),
                "highPrice": utils.safe_to_float(entry.get("high")), "lowPrice": utils.safe_to_float(entry.get("low")),
                "closePrice": utils.safe_to_float(entry.get("close")), "volume": utils.safe_to_float(entry.get("volume")),
            }
            self.candles_data[symbol].append(row)
            self.volume_history[symbol].append(row["volume"])
            oi_val = self.latest_open_interest.get(symbol, 0.0)
            self.oi_history[symbol].append(oi_val)
            delta = row["volume"] if row["closePrice"] >= row["openPrice"] else -row["volume"]
            prev_cvd = self.cvd_history[symbol][-1] if self.cvd_history[symbol] else 0.0
            self.cvd_history[symbol].append(prev_cvd + delta)
            self._last_saved_time[symbol] = ts

            # --- НАЧАЛО ИЗМЕНЕНИЯ: Отладочный лог ---
            # Эта строка покажет нам, что свечи обрабатываются.
            # Мы логгируем только раз в 100 свечей, чтобы не забивать лог.
            if len(self.candles_data[symbol]) % 100 == 0:
                logger.debug(f"[History] Накоплено {len(self.candles_data[symbol])} свечей для {symbol}.")
            # --- КОНЕЦ ИЗМЕНЕНИЯ ---


            for bot in self.position_handlers:
                asyncio.create_task(bot.run_low_frequency_strategies(symbol))

    async def handle_ticker(self, msg):
        """
        Универсальный обработчик тикеров с WS:
        - обновляет локальные карты best_bid/best_ask у ВСЕХ position_handlers;
        - пушит last_price в on_ticker_update(bot) ровно ОДИН раз;
        - поддерживает разные ключи от Bybit Unified (bid1Price/bestBidPrice/bidPrice и ask1Price/...).
        """
        data = msg.get("data", {})
        entries = data if isinstance(data, list) else [data]

        for ticker in entries:
            symbol = ticker.get("symbol")
            if not symbol:
                continue

            # --- базовые кэши тикера/ОI/фандинга ---
            self.ticker_data[symbol] = ticker

            oi_val = utils.safe_to_float(ticker.get("openInterest", 0))
            self.latest_open_interest[symbol] = oi_val

            if (f_raw := ticker.get("fundingRate")) is not None:
                self.funding_history[symbol].append(utils.safe_to_float(f_raw))

            hist = self.oi_history.setdefault(symbol, collections.deque(maxlen=1000))
            if not hist or hist[-1] != oi_val:
                hist.append(oi_val)

            # --- извлекаем bid/ask аккуратно из того, что пришло ---
            t = entries[0] if isinstance(data, list) else data
            best_bid = utils.safe_to_float(
                t.get("bid1Price") or t.get("bestBidPrice") or t.get("bidPrice") or 0.0
            )
            best_ask = utils.safe_to_float(
                t.get("ask1Price") or t.get("bestAskPrice") or t.get("askPrice") or 0.0
            )
            last_price = utils.safe_to_float(ticker.get("lastPrice") or 0.0)

            # --- раздаём всем хэндлерам позиции ---
            for bot in self.position_handlers:
                # гарантируем наличие карт (устраняем AttributeError в bot_core)
                if not hasattr(bot, "best_bid_map"):
                    bot.best_bid_map = {}
                if not hasattr(bot, "best_ask_map"):
                    bot.best_ask_map = {}

                if best_bid > 0.0:
                    bot.best_bid_map[symbol] = best_bid
                if best_ask > 0.0:
                    bot.best_ask_map[symbol] = best_ask

                if last_price > 0.0:
                    # Важно: без kwargs; сигнатура on_ticker_update(self, symbol, last_price)
                    asyncio.create_task(bot.on_ticker_update(symbol, last_price))
            symbol = data.get("s") or data.get("symbol")
            price = float(data.get("lastPrice") or data.get("price") or 0.0)

            if symbol and price > 0:
                bot = self._find_bot_for_symbol(symbol)
                if bot:
                    try:
                        await bot._handle_realtime_price_tick(symbol, price)
                    except Exception as e:
                        logger.warning(f"[WS][TRAIL] Ошибка обновления трейлинга: {e}")
            

    def _find_bot_for_symbol(self, symbol: str):
        for bot in getattr(self, "bots", []):
            if symbol in bot.open_positions:
                return bot
        return None

    async def backfill_history(self):
        """
        [ВЕРСИЯ 4.0] Надежный циклический бэкфилл.
        Загружает ВСЕ недостающие свечи с момента последнего сохранения.
        """
        http = HTTP(testnet=False, timeout=20)
        symbols_to_backfill = list(self.watchlist)
        
        for symbol in symbols_to_backfill:
            try:
                all_new_candles = []
                # Определяем, с какого момента нам нужны данные
                last_known_ts_ms = 0
                if self.candles_data[symbol]:
                    last_known_ts_ms = int(self.candles_data[symbol][-1]['startTime'].timestamp() * 1000)

                # Загружаем данные в цикле, пока API отдает свечи
                while True:
                    limit = 1000
                    params = {'symbol': symbol, 'interval': self.interval, 'limit': limit}
                    # Запрашиваем данные ДО самой старой из уже имеющихся свечей
                    # Если данных нет, end_ts = 0, и API вернет самые свежие
                    end_ts = 0
                    if all_new_candles:
                        end_ts = all_new_candles[0]['startTime'] # startTime уже в timestamp
                    elif last_known_ts_ms > 0:
                        # Этот блок не будет работать, так как мы запрашиваем в прошлое, а не в будущее
                        # end_ts = last_known_ts_ms
                        pass

                    # Bybit API v5 kline `end` параметр не поддерживается для запросов в прошлое.
                    # Вместо этого мы делаем запросы без `start` и `end`, получаем самые свежие,
                    # и если они старее наших, значит мы заполнили пробел.
                    # Правильная логика - запрашивать в цикле и останавливаться, когда timestamp'ы пересекутся.
                    
                    current_chunk_ts = int(time.time() * 1000)
                    
                    local_all_candles = list(self.candles_data[symbol])

                    while True:
                        # Если локальных данных нет, просто загружаем последние 1000 и выходим
                        if not local_all_candles:
                             params['limit'] = 1000
                        else:
                            # Запрашиваем свечи до самой старой из тех, что у нас есть
                            current_chunk_ts = int(local_all_candles[0]['startTime'].timestamp() * 1000)
                            params['end'] = current_chunk_ts
                            params['limit'] = 1000

                        resp = await asyncio.to_thread(lambda: http.get_kline(**params))
                        bars = resp.get('result', {}).get('list', [])
                        if not bars:
                            break # Если API ничего не вернул, значит, данных больше нет

                        new_count = 0
                        for entry in reversed(bars): # Bybit отдает от новых к старым
                            ts = pd.to_datetime(int(entry[0]), unit='ms')

                            # Проверяем, не достигли ли мы уже имеющихся данных
                            if local_all_candles and ts >= local_all_candles[0]['startTime']:
                                continue # Пропускаем дубликаты

                            row = {
                                'startTime': ts, 'openPrice': utils.safe_to_float(entry[1]),
                                'highPrice': utils.safe_to_float(entry[2]), 'lowPrice': utils.safe_to_float(entry[3]),
                                'closePrice': utils.safe_to_float(entry[4]), 'volume': utils.safe_to_float(entry[5]),
                            }
                            self.candles_data[symbol].appendleft(row)
                            new_count += 1

                        if new_count == 0 or not local_all_candles: # Если не добавили новых свечей или это был первый запрос
                            break

                        # Обновляем список локальных свечей для следующей итерации
                        local_all_candles = list(self.candles_data[symbol])
                        await asyncio.sleep(0.5) # Чтобы не превышать лимиты API

                total_candles = len(self.candles_data[symbol])
                if total_candles > 0:
                    logger.info(f"[History] Бэкфилл для {symbol} завершен. Всего свечей: {total_candles}.")
            except Exception as e:
                logger.error(f"[History] backfill error for {symbol}: {e}", exc_info=True)
        
        self._save_history()


    def _save_history(self):
        tmp_path = None
        try:
            data_to_save = {'candles': {k: list(v) for k, v in self.candles_data.items()}}
            history_dir = self._history_file.parent
            history_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(history_dir)) as tmp_file:
                pickle.dump(data_to_save, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_path = tmp_file.name
            os.replace(tmp_path, self._history_file)
        except Exception as e:
            logger.warning(f"Ошибка сохранения истории: {e}", exc_info=True)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


    async def _save_loop(self, interval: int = 60):
        while True:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(self._save_history)
            except Exception as e:
                logger.error(f"Критическая ошибка в цикле сохранения истории: {e}", exc_info=True)

    async def _refresh_symbol_candles(self, symbol: str, limit: int = 120) -> None:
        logger.debug(f"[History] Refreshing last {limit} candles for {symbol} via HTTP kline")

        def _fetch():
            client = HTTP(testnet=False, timeout=20)
            return client.get_kline(symbol=symbol, interval=self.interval, limit=limit)

        resp = await asyncio.to_thread(_fetch)
        bars = resp.get("result", {}).get("list", [])
        if not bars:
            logger.warning(f"[History] refresh returned no data for {symbol}")
            return

        new_rows = []
        for entry in bars:
            try:
                ts = pd.to_datetime(int(entry[0]), unit="ms")
            except Exception:
                continue
            new_rows.append({
                "startTime": ts,
                "openPrice": utils.safe_to_float(entry[1]),
                "highPrice": utils.safe_to_float(entry[2]),
                "lowPrice": utils.safe_to_float(entry[3]),
                "closePrice": utils.safe_to_float(entry[4]),
                "volume": utils.safe_to_float(entry[5]),
            })

        dq = self.candles_data[symbol]
        maxlen = dq.maxlen if isinstance(dq, deque) else 3000

        combined = {row.get("startTime"): row for row in dq}
        for row in new_rows:
            combined[row["startTime"]] = row

        sorted_rows = sorted(combined.values(), key=lambda r: r.get("startTime"))
        self.candles_data[symbol] = deque(sorted_rows[-maxlen:], maxlen=maxlen)
        logger.info(f"[History] Refreshed candles for {symbol}: now {len(self.candles_data[symbol])} rows")

    async def ensure_recent_candles(self, symbol: str, lookback: int = 120, max_age_sec: float = 180.0) -> None:
        lock = self._candles_refresh_locks.setdefault(symbol, asyncio.Lock())
        async with lock:
            candles = list(self.candles_data.get(symbol, []))
            now_utc = dt.datetime.utcnow()
            need_refresh = False

            if len(candles) < lookback:
                need_refresh = True
            elif candles:
                last_ts = candles[-1].get("startTime")
                if last_ts is not None:
                    ts = pd.to_datetime(last_ts)
                    if pd.isna(ts):
                        need_refresh = True
                    else:
                        if ts.tzinfo is not None:
                            ts = ts.tz_convert(None)
                        age = (now_utc - ts.to_pydatetime()).total_seconds()
                        if age > max_age_sec:
                            need_refresh = True

            if not need_refresh:
                return

            await self._refresh_symbol_candles(symbol, limit=max(lookback, 120))

    def get_liq_threshold(self, symbol: str, default: float = 5000.0) -> float:
        t24 = utils.safe_to_float(self.ticker_data.get(symbol, {}).get("turnover24h", 0))
        if t24 >= LARGE_TURNOVER: return 0.0015 * t24
        if t24 >= MID_TURNOVER: return 0.0025 * t24
        return max(8_000.0, self._liq_thresholds.get(symbol, default))

    def _sigma_5m(self, symbol: str, window: int = VOL_WINDOW) -> float:
        candles = list(self.candles_data.get(symbol, []))[-window:]
        if len(candles) < window: return 0.0
        moves = [
            abs(c["closePrice"] - c["openPrice"]) / c["openPrice"]
            for c in candles if utils.safe_to_float(c.get("openPrice")) > 0
        ]
        return float(np.std(moves)) if moves else 0.0

    def has_5_percent_growth(self, symbol: str, minutes: int = 20) -> bool:
        candles = list(self.candles_data.get(symbol, []))
        if len(candles) < minutes: return False
        old_close = utils.safe_to_float(candles[-minutes].get("closePrice", 0))
        new_close = utils.safe_to_float(candles[-1].get("closePrice", 0))
        if old_close <= 0: return False
        return (new_close - old_close) / old_close * 100.0 >= 3.0

    def check_liq_cooldown(self, symbol: str) -> bool:
        sigma = self._sigma_5m(symbol)
        cooldown = 900 if sigma >= 0.01 else 600
        last = self.last_liq_trade_time.get(symbol)
        if not last: return True
        return (time.time() - last) >= cooldown

    def get_avg_volume(self, symbol: str, minutes: int) -> float:
        vol_history = self.volume_history.get(symbol)
        if not vol_history or len(vol_history) < minutes:
            return 0.0
        recent_vols = list(vol_history)[-minutes:]
        total_volume = sum(utils.safe_to_float(v) for v in recent_vols)
        return total_volume / len(recent_vols) if recent_vols else 0.0
