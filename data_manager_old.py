# data_manager.py
import asyncio
import logging
import time
import pickle
import random 
import csv
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# --- НАЧАЛО ФИКСА СОВМЕСТИМОСТИ NUMPY 2.0 ---
# Эта проверка должна быть ПЕРЕД импортом pandas_ta
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # Восстанавливаем псевдоним NaN, который ожидает pandas-ta
# --- КОНЕЦ ФИКСА СОВМЕСТИМОСТИ ---

import pandas_ta as ta
from pybit.unified_trading import WebSocket, HTTP
import requests
from urllib3.exceptions import ReadTimeoutError as UrllibReadTimeoutError

import config
import utils

logger = logging.getLogger(__name__)

# Фикс совместимости для NumPy 2.0+
if not hasattr(np, "NaN"):
    np.NaN = np.nan

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
        self.is_first_connection = True
        self.ready_event = asyncio.Event()
        self.loop = asyncio.get_event_loop()
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

    def _load_history_from_disk(self):
        try:
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
            with open(config.LIQ_THRESHOLD_CSV_PATH, "r", newline="") as f:
                for row in csv.DictReader(f):
                    self._liq_thresholds[row["symbol"]] = float(row["threshold"])
            logger.info(f"Загружено {len(self._liq_thresholds)} порогов ликвидаций.")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Ошибка загрузки порогов ликвидаций: {e}")


    async def start(self):
        """Главный цикл, который поддерживает WebSocket-соединение живым и перезапускает его при необходимости."""
        if not hasattr(self, "_symbol_manager_task") or self._symbol_manager_task.done():
            self._symbol_manager_task = asyncio.create_task(self.manage_symbol_selection())
        if not hasattr(self, "_save_task") or self._save_task.done():
            self._save_task = asyncio.create_task(self._save_loop())

        while True:
            try:
                def _on_message(msg):
                    try:
                        if not self.loop.is_closed():
                            asyncio.run_coroutine_threadsafe(self.route_message(msg), self.loop)
                    except Exception as e:
                        logger.warning(f"PublicWS callback error: {e}")

                self.ws = WebSocket(
                    testnet=False,
                    channel_type="linear",
                    ping_interval=30,
                    ping_timeout=15,  # Это соотношение (30 > 15) верное и не вызывает ошибок
                    restart_on_error=False,
                )

                active_symbols_list = list(self.active_symbols)

                self.ws.kline_stream(
                    interval=self.interval,
                    symbol=active_symbols_list,
                    callback=_on_message
                )

                self.ws.ticker_stream(
                    symbol=active_symbols_list,
                    callback=_on_message
                )

                # --- ИСПРАВЛЕНИЕ ОШИБКИ ---
                # Метод ТРЕБУЕТ аргумент 'symbol'. Передаем ему наш список.
                self.ws.all_liquidation_stream(
                    symbol=active_symbols_list,
                    callback=_on_message
                )
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---


                logger.info(f"WebSocket подключается с {len(active_symbols_list) * 3} индивидуальными подписками.")

                while self.ws.is_connected():
                    await asyncio.sleep(1)

                logger.warning("WebSocket соединение разорвано (обнаружено в цикле проверки)")

            except asyncio.CancelledError:
                logger.info("Public WS task cancelled.")
                if self.ws: self.ws.exit()
                if hasattr(self, "_symbol_manager_task"): self._symbol_manager_task.cancel()
                if hasattr(self, "_save_task"): self._save_task.cancel()
                break
            except Exception as e:
                logger.warning(f"Ошибка в цикле PublicWS start: {e}")

            logger.info("Переподключение через 5 секунд...")
            if self.ws:
                self.ws.exit()
            await asyncio.sleep(5)




    def _existing_topics(self) -> set[str]:
        try:
            return set(getattr(self.ws, "callback_directory", {}).keys())
        except Exception:
            return set()

    def _filter_new_symbols(self, tpl: str, symbols: list[str], existing: set[str]) -> list[str]:
        return [s for s in symbols if tpl.format(symbol=s) not in existing]

    async def manage_symbol_selection(self, check_interval=3600):
        http = HTTP(testnet=False, timeout=20)
        is_first_run = True
        while True:
            try:
                await asyncio.sleep(15 if is_first_run else check_interval)

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
                all_tickers = {tk["symbol"]: tk for tk in tickers_resp["result"]["list"]}
                self.ticker_data.update(all_tickers)

                # --- ОГРАНИЧЕНИЕ КОЛИЧЕСТВА СИМВОЛОВ ---
                
                sorted_tickers = sorted(
                    [t for t in all_tickers.values() if t["symbol"] in tradable_and_mature_symbols],
                    key=lambda x: utils.safe_to_float(x.get("turnover24h", 0)),
                    reverse=True
                )

                TOP_N_SYMBOLS = 70 # Ограничиваемся 70 самыми ликвидными инструментами
                top_symbols = {t["symbol"] for t in sorted_tickers[:TOP_N_SYMBOLS]}
                
                # --- КОНЕЦ БЛОКА ОГРАНИЧЕНИЯ ---

                open_pos_symbols = {s for bot in self.position_handlers for s in bot.open_positions.keys()}
                
                desired_symbols = top_symbols.union(open_pos_symbols)
                desired_symbols.add("BTCUSDT")
                desired_symbols.add("ETHUSDT")

                if desired_symbols != self.active_symbols:
                    logger.info(f"Список активных символов изменился. Старый: {len(self.active_symbols)}, Новый: {len(desired_symbols)}.")
                    
                    symbols_list = list(desired_symbols)
                    random.shuffle(symbols_list)
                    
                    self.active_symbols = set(symbols_list)
                    self.symbols = symbols_list

                    if self.ws and self.ws.is_connected():
                        logger.info("Перезапускаем WebSocket для применения нового списка символов...")
                        self.ws.exit()
                
                if is_first_run:
                    self.ready_event.set()
                    is_first_run = False
                    logger.info(f"Начальный список из {len(self.active_symbols)} символов сформирован. Бот готов.")

            except Exception as e:
                logger.error(f"Критическая ошибка в SymbolManager: {e}", exc_info=True)
                if is_first_run and not self.ready_event.is_set():
                    self.ready_event.set()



    async def route_message(self, msg):
        topic = msg.get("topic", "")
        if topic.startswith("kline."):
            await self.handle_kline(msg)
        elif topic.startswith("tickers."):
            await self.handle_ticker(msg)
        elif topic.startswith("liquidation"): # Теперь тема будет просто 'liquidation'
            # Данные от all_liquidation_stream приходят как единичный объект
            liq_event = msg.get("data")
            if liq_event:
                for handler in self.position_handlers:
                    if hasattr(handler, "on_liquidation_event"):
                        asyncio.create_task(handler.on_liquidation_event(liq_event))



    async def handle_kline(self, msg):
        symbol = msg["topic"].split(".")[-1]
        # Фильтруем только активные символы
        if symbol not in self.active_symbols:
            return

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
            # Фильтруем только активные символы
            if not symbol or symbol not in self.active_symbols:
                continue
                    
            self.ticker_data[symbol] = ticker

            oi_val = utils.safe_to_float(ticker.get("openInterest", 0))
            self.latest_open_interest[symbol] = oi_val
            
            if (f_raw := ticker.get("fundingRate")) is not None:
                self.funding_history[symbol].append(utils.safe_to_float(f_raw))
            
            hist = self.oi_history.setdefault(symbol, deque(maxlen=1000))
            if not hist or hist[-1] != oi_val:
                hist.append(oi_val)

            for bot in self.position_handlers:
                for ticker in entries:
                    symbol = ticker.get("symbol")
                    last_price = utils.safe_to_float(ticker.get("lastPrice"))
                    if symbol and last_price > 0:
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
                    if isinstance(entry, list):
                        try:
                            ts = pd.to_datetime(int(entry[0]), unit='ms')
                            open_p  = utils.safe_to_float(entry[1])
                            high_p  = utils.safe_to_float(entry[2])
                            low_p   = utils.safe_to_float(entry[3])
                            close_p = utils.safe_to_float(entry[4])
                            vol     = utils.safe_to_float(entry[5])
                        except Exception:
                            print(f"[History] backfill invalid list entry for {symbol}: {entry}")
                            continue
                    else:
                        try:
                            ts = pd.to_datetime(int(entry['start']), unit='ms')
                        except Exception as e:
                            print(f"[History] backfill invalid dict entry for {symbol}: {e}")
                            continue
                        open_p  = utils.safe_to_float(entry.get('open', 0))
                        high_p  = utils.safe_to_float(entry.get('high', 0))
                        low_p   = utils.safe_to_float(entry.get('low', 0))
                        close_p = utils.safe_to_float(entry.get('close', 0))
                        vol     = utils.safe_to_float(entry.get('volume', 0))
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
                    self.oi_history[symbol].append(0.0)
                    delta = vol if close_p >= open_p else -vol
                    prev_cvd = self.cvd_history[symbol][-1] if self.cvd_history[symbol] else 0.0
                    self.cvd_history[symbol].append(prev_cvd + delta)
                    count += 1
                if count:
                    self._save_history()
                    try:
                        ticker_resp = http.get_tickers(category="linear", symbol=symbol)
                        oi_val = utils.safe_to_float(
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


    async def _save_loop(self, interval: int = 60):
        while True:
            await asyncio.sleep(interval)
            self._save_history()


    def get_liq_threshold(self, symbol: str, default: float = 5000.0) -> float:
        t24 = utils.safe_to_float(self.ticker_data.get(symbol, {}).get("turnover24h", 0))
        if t24 >= 100_000_000: return 0.0015 * t24
        if t24 >= 10_000_000: return 0.0025 * t24
        return max(8_000.0, self._liq_thresholds.get(symbol, default))

    def _sigma_5m(self, symbol: str, window: int = 60) -> float:
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