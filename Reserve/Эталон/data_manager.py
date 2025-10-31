# data_manager.py
import asyncio
import logging
import time
import pickle
import random 
import csv
import os
import datetime as dt
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# --- НАЧАЛО ФИКСА СОВМЕСТИМОСТИ NUMPY 2.0 ---
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# --- КОНЕЦ ФИКСА СОВМЕСТИМОСТИ ---

import pandas_ta as ta
from pybit.unified_trading import WebSocket, HTTP
import requests
from urllib3.exceptions import ReadTimeoutError as UrllibReadTimeoutError

import config
import utils
from websocket_monitor import get_monitor

logger = logging.getLogger(__name__)

# --- Константы, необходимые для работы класса ---
LARGE_TURNOVER = 100_000_000
MID_TURNOVER   = 10_000_000
VOL_WINDOW     = 60
VOL_COEF       = 1.2
LISTING_AGE_MIN_MINUTES = 1400
# -------------------------------------------------

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
        self.candles_data = defaultdict(lambda: deque(maxlen=2000))
        self.ticker_data = {}
        self.latest_open_interest = {}
        self.active_symbols = set(symbols)
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
        self._last_saved_time = {}
        self.position_handlers = []
        self.latest_liquidation = {}
        self._liq_thresholds = defaultdict(lambda: 5000.0)
        self.last_liq_trade_time = {}
        self._history_file = config.HISTORY_FILE
        self._load_history_from_disk()
        self._load_liq_thresholds()
        self.monitor = get_monitor()
        self.connection_name = "public_websocket"

    def _load_history_from_disk(self):
        try:
            if self._history_file.exists() and self._history_file.stat().st_size > 0:
                with open(self._history_file, 'rb') as f:
                    data = pickle.load(f)
                    for sym, rows in data.get('candles', {}).items():
                        self.candles_data[sym] = rows
                logger.info(f"Загружена история из {self._history_file}")
        except FileNotFoundError:
            logger.info(f"Файл истории {self._history_file} не найден.")
        except Exception as e:
            logger.error(f"Ошибка загрузки истории: {e}")

    def _load_liq_thresholds(self):
        try:
            if config.LIQ_THRESHOLD_CSV_PATH.exists():
                with open(config.LIQ_THRESHOLD_CSV_PATH, "r", newline="") as f:
                    for row in csv.DictReader(f):
                        self._liq_thresholds[row["symbol"]] = float(row["threshold"])
                logger.info(f"Загружено {len(self._liq_thresholds)} порогов ликвидаций.")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Ошибка загрузки порогов ликвидаций: {e}")

    async def start(self):
        # Запускаем фоновые задачи только один раз
        if not hasattr(self, "_save_task") or self._save_task.done():
            self._save_task = asyncio.create_task(self._save_loop())
        
        # Определяем список символов один раз при запуске
        await self._initialize_symbol_list()

        try:
            def _on_message(msg):
                try:
                    # Мониторинг активности
                    self.monitor.record_message(self.connection_name)
                    if not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self.route_message(msg),
                            self.loop
                        )
                except Exception as e:
                    logger.warning(f"[PublicWS callback] loop closed, skipping message: {e}")

            # РЕКОМЕНДАЦИЯ: Измените параметры ping согласно документации Bybit V5
            self.ws = WebSocket(
                testnet=False,
                channel_type="linear",
                ping_interval=20,  # Рекомендуемый интервал - 20 секунд
                ping_timeout=10,   # Таймаут ответа
                restart_on_error=True,
                retries=200,
            )
            
            logger.info(f"WebSocket подключается с {len(self.symbols)} символами.")
            
            # Подписка на стримы остается без изменений
            self.ws.kline_stream(
                interval=self.interval,
                symbol=self.symbols,
                callback=_on_message
            )
            self.ws.ticker_stream(
                symbol=self.symbols,
                callback=_on_message
            )
            self.ws.all_liquidation_stream(
                symbol=self.symbols,
                callback=_on_message
            )
            
            self.monitor.register_connection(self.connection_name)
            
            # УБРАН БЛОКИРУЮЩИЙ ВЫЗОВ:
            # await asyncio.Event().wait() <--- ЭТО БЫЛО НЕПРАВИЛЬНО
            
            logger.info("Public WebSocket запущен в фоновом режиме.")

        except asyncio.CancelledError:
            logger.info("Public WS task cancelled.")
            if self.ws: self.ws.exit()
        except Exception as e:
            logger.error(f"Критическая ошибка при запуске Public WS: {e}", exc_info=True)
            if self.ws: self.ws.exit()

    def stop(self):
        """Добавлен метод для корректной остановки."""
        logger.info("Остановка Public WebSocket Manager...")
        if self.ws:
            self.ws.exit()
        if hasattr(self, "_save_task") and not self._save_task.done():
            self._save_task.cancel()
        logger.info("Public WebSocket Manager остановлен.")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---


    async def _initialize_symbol_list(self):
        http = HTTP(testnet=False, timeout=20)
        try:
            instruments_resp = await asyncio.to_thread(lambda: http.get_instruments_info(category="linear"))
            all_instruments = {i["symbol"]: i for i in instruments_resp["result"]["list"]}
            
            min_age_minutes = 1440
            now_ms = time.time() * 1000
            
            tradable_and_mature_symbols = {
                s for s, info in all_instruments.items() 
                if info.get("status") == "Trading" and 
                   (now_ms - utils.safe_to_float(info.get("launchTime", 0))) / 60000 > min_age_minutes
            }

            tickers_resp = await asyncio.to_thread(lambda: http.get_tickers(category="linear"))
            all_tickers = tickers_resp["result"]["list"]
            self.ticker_data.update({tk["symbol"]: tk for tk in all_tickers})

            TOP_N_SYMBOLS = 100
            sorted_tickers = sorted(
                [t for t in all_tickers if t["symbol"] in tradable_and_mature_symbols],
                key=lambda x: utils.safe_to_float(x.get("turnover24h", 0)),
                reverse=True
            )
            top_symbols = {t["symbol"] for t in sorted_tickers[:TOP_N_SYMBOLS]}
            
            open_pos_symbols = {s for bot in self.position_handlers for s in bot.open_positions.keys()}
            desired_symbols = top_symbols.union(open_pos_symbols)
            desired_symbols.add("BTCUSDT")
            desired_symbols.add("ETHUSDT")

            self.active_symbols = desired_symbols
            self.symbols = list(desired_symbols)
            
            logger.info(f"Начальный список из {len(self.active_symbols)} символов сформирован.")
        except Exception as e:
            logger.error(f"Критическая ошибка при формировании списка символов: {e}", exc_info=True)
            self.active_symbols = {"BTCUSDT", "ETHUSDT"}
            self.symbols = ["BTCUSDT", "ETHUSDT"]
        
        self.ready_event.set()

    async def route_message(self, msg):
        topic = msg.get("topic", "")
        if topic.startswith("kline."):
            await self.handle_kline(msg)
        elif topic.startswith("tickers."):
            await self.handle_ticker(msg)
        elif "liquidation" in topic.lower():
            for evt in msg.get("data", []):
                for handler in self.position_handlers:
                    if hasattr(handler, "on_liquidation_event"):
                        asyncio.create_task(handler.on_liquidation_event(evt))

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

            for bot in self.position_handlers:
                asyncio.create_task(bot.run_low_frequency_strategies(symbol))

    async def handle_ticker(self, msg):
        data = msg.get("data", {})
        entries = data if isinstance(data, list) else [data]
        
        for ticker in entries:
            symbol = ticker.get("symbol")
            if not symbol: continue
            
            self.ticker_data[symbol] = ticker
            oi_val = utils.safe_to_float(ticker.get("openInterest", 0))
            self.latest_open_interest[symbol] = oi_val
            
            if (f_raw := ticker.get("fundingRate")) is not None:
                self.funding_history[symbol].append(utils.safe_to_float(f_raw))
            
            hist = self.oi_history.setdefault(symbol, deque(maxlen=1000))
            if not hist or hist[-1] != oi_val:
                hist.append(oi_val)

            for bot in self.position_handlers:
                last_price = utils.safe_to_float(ticker.get("lastPrice"))
                if last_price > 0:
                    asyncio.create_task(bot.on_ticker_update(symbol, last_price))

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
                    ts = pd.to_datetime(int(entry[0]), unit='ms')
                    row = {
                        'startTime': ts,
                        'openPrice': utils.safe_to_float(entry[1]),
                        'highPrice': utils.safe_to_float(entry[2]),
                        'lowPrice': utils.safe_to_float(entry[3]),
                        'closePrice': utils.safe_to_float(entry[4]),
                        'volume': utils.safe_to_float(entry[5]),
                    }
                    self.candles_data[symbol].append(row)
                    self.volume_history[symbol].append(row['volume'])
                    self.oi_history[symbol].append(0.0)
                    delta = row['volume'] if row['closePrice'] >= row['openPrice'] else -row['volume']
                    prev_cvd = self.cvd_history[symbol][-1] if self.cvd_history[symbol] else 0.0
                    self.cvd_history[symbol].append(prev_cvd + delta)
                    count += 1
                if count:
                    self._save_history()
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
                    'cvd_history': {k: list(v) for k, v in self.cvd_history.items()},
                }, f)
        except Exception as e:
            logger.warning(f"Ошибка сохранения истории: {e}")

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    async def _save_loop(self, interval: int = 60):
        """Асинхронный цикл сохранения, который вызывает синхронную функцию неблокирующим способом."""
        while True:
            await asyncio.sleep(interval)
            try:
                # Выполняем блокирующую операцию в отдельном потоке
                await asyncio.to_thread(self._save_history)
            except Exception as e:
                logger.error(f"Критическая ошибка в цикле сохранения истории: {e}", exc_info=True)
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---


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


    # def _log_liquidations_to_csv(self):
    #     """
    #     [ИСПРАВЛЕННАЯ ВЕРСИЯ] Собирает данные о ликвидациях из буферов
    #     ВСЕХ зарегистрированных ботов и записывает их в единый CSV-файл.
    #     """
    #     try:
    #         file_path = config.LIQUIDATIONS_CSV_PATH
    #         headers = ["timestamp", "user_id", "symbol", "side", "price", "value_usd"]
    #         file_exists = file_path.is_file() and file_path.stat().st_size > 0
            
    #         all_events_to_log = []
    #         now_iso = dt.datetime.utcnow().isoformat()

    #         # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Собираем данные из всех ботов ---
    #         for bot in self.position_handlers:
    #             # Проверяем, есть ли у бота буфер и не пустой ли он
    #             if not hasattr(bot, 'liq_buffers') or not bot.liq_buffers:
    #                 continue

    #             # Проходим по буферам конкретного бота
    #             for symbol, buffer in bot.liq_buffers.items():
    #                 for evt in buffer:
    #                     all_events_to_log.append({
    #                         "timestamp": now_iso,
    #                         "user_id": bot.user_id, # Добавляем user_id для ясности
    #                         "symbol": symbol,
    #                         "side": evt.get("side"),
    #                         "price": evt.get("price"),
    #                         "value_usd": evt.get("value")
    #                     })
                
    #             # Важно: Очищаем буфер у бота, чьи данные мы собрали
    #             bot.liq_buffers.clear()
            
    #         if not all_events_to_log:
    #             return # Если не было ликвидаций, выходим

    #         # Записываем все собранные события в файл
    #         with open(file_path, "a", newline="", encoding="utf-8") as f:
    #             writer = csv.DictWriter(f, fieldnames=headers)
    #             if not file_exists:
    #                 writer.writeheader()
    #             writer.writerows(all_events_to_log)
            
    #         logger.debug(f"Успешно записано {len(all_events_to_log)} событий ликвидации в лог.")

    #     except Exception as e:
    #         logger.warning(f"Ошибка при записи лога ликвидаций: {e}", exc_info=True)
