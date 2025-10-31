import asyncio
import logging
import json
import time
import os
import sys
import multiprocessing as mp
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, Optional, Sequence
import datetime as dt
import pandas as pd
import numpy as np
import math
import random
import uuid

from pybit.unified_trading import HTTP, WebSocket
from pybit.exceptions import InvalidRequestError
from requests.adapters import HTTPAdapter
from aiogram.enums import ParseMode
import pandas_ta as ta

# Импортируем наши модули
import config
import utils
import strategies
import ai_ml
from telegram_bot import bot as telegram_bot # <-- ИСПРАВЛЕННАЯ СТРОКА
from signal_worker import start_worker_process
from data_manager import compute_supertrend
from utils import async_retry
from decimal import Decimal

logger = logging.getLogger(__name__)

# Глобальные переменные, которые теперь будут инкапсулированы в классе
_listing_age_cache: dict[str, tuple[float, float]] = {}
_listing_sem = asyncio.Semaphore(5)

class TradingBot:
    def __init__(self, user_data: Dict[str, Any], shared_ws, golden_param_store: Dict):
        # --- Основные данные пользователя ---
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.user_data = user_data
        self.mode = user_data.get("mode", "real")

        # --- Подключение к компонентам ---
        self.shared_ws = shared_ws
        self.shared_ws.position_handlers.append(self)
        self.loop = asyncio.get_running_loop()

        # --- Состояние бота ---
        self.open_positions: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, float] = {}
        self.pending_timestamps: Dict[str, float] = {}
        self.pending_cids: Dict[str, str] = {}
        self.recently_closed: Dict[str, float] = {}
        self.active_signals = set()
        self.strategy_cooldown_until: Dict[str, float] = {}
        self.last_entry_ts: Dict[str, float] = {}
        self.failed_orders: Dict[str, float] = {}
        self.reserve_orders: Dict[str, Dict] = {}
        self.closed_positions: Dict[str, Dict] = {}
        self.last_entry_comment: Dict[str, str] = {}
        self.pending_strategy_comments: Dict[str, str] = {}
        
        self.pending_open_exec: Dict[str, Dict[str, Any]] = {}   # symbol -> {price, side, ts}
        self.momentum_cooldown_until = defaultdict(float)        # (symbol, side) -> ts (уже пригодится охотнику)

        # --- Состояние позиций и стопов ---
        self.last_stop_price: Dict[str, float] = {}
        self.watch_tasks: Dict[str, asyncio.Task] = {}
        self.user_state = getattr(self, "user_state", {})

        self.trailing_activated: Dict[str, bool] = {} # Флаг, что трейлинг для позиции активен


        # --- Конфигурация торговли ---
        self.POSITION_VOLUME = utils.safe_to_float(user_data.get("volume", 1000))
        self.MAX_TOTAL_VOLUME = utils.safe_to_float(user_data.get("max_total_volume", 5000))
        self.leverage = utils.safe_to_float(user_data.get("leverage", 10.0))
        self.listing_age_min = int(user_data.get("listing_age_min_minutes", config.LISTING_AGE_MIN_MINUTES))
        
        # --- Метаданные символов ---
        self.qty_step_map: Dict[str, float] = {}
        self.min_qty_map: Dict[str, float] = {}
        self.price_tick_map: Dict[str, float] = {}

        # --- ML и AI ---
        self.ml_inferencer: Optional[ai_ml.MLXInferencer] = None
        self.training_data = deque(maxlen=5000)
        self.ai_circuit_open_until = 0.0
        self._ai_inflight_signals = set()
        self.apply_user_settings()

        self.momentum_cooldown_until = defaultdict(float)   # key: (symbol, side) -> ts

        # --- HTTP сессия ---
        self.session = HTTP(
            testnet=False, demo=(self.mode == "demo"),
            api_key=self.api_key, api_secret=self.api_secret, timeout=30
        )
        try:
            adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
            self.session.client.mount("https://", adapter)
        except Exception: pass

        # --- Воркеры и очереди ---
        # self.signal_task_queue: Optional[mp.Queue] = None
        # self.order_command_queue: Optional[mp.Queue] = None
        # self.signal_worker_process: Optional[mp.Process] = None
        
        # --- Прочее ---
        self.position_lock = asyncio.Lock()
        self.pending_orders_lock = asyncio.Lock()
        self.liq_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.golden_param_store = golden_param_store
        self._last_golden_ts = defaultdict(float)
        self.last_squeeze_ts = defaultdict(float)
        self.current_total_volume = 0.0
        self.time_offset = 0


    def apply_user_settings(self):
        """Применяет настройки из user_state.json, связанные с AI и стратегиями."""
        cfg = self.user_data
        self.strategy_mode = cfg.get("strategy_mode", "full")
        self.ai_primary_model = cfg.get("ai_primary_model", config.AI_PRIMARY_MODEL)
        self.ai_advisor_model = cfg.get("ai_advisor_model", config.AI_ADVISOR_MODEL)
        self.ollama_primary_openai = cfg.get("ollama_primary_openai", config.OLLAMA_PRIMARY_OPENAI)
        self.ollama_advisor_openai = cfg.get("ollama_advisor_openai", config.OLLAMA_ADVISOR_OPENAI)
        self.ai_timeout_sec = float(cfg.get("ai_timeout_sec", 15.0))
        self.entry_cooldown_sec = int(cfg.get("entry_cooldown_sec", 30))
        self.tactical_entry_window_sec = int(cfg.get("tactical_entry_window_sec", 45))
        self.squeeze_ai_confirm_interval_sec = float(cfg.get("squeeze_ai_confirm_interval_sec", 2.0))
        logger.info(f"Настройки для пользователя {self.user_id} применены. Режим: {self.strategy_mode}")

    async def _sync_server_time(self):
            """
            [ФИНАЛЬНАЯ ВЕРСИЯ] Корректно синхронизирует время с сервером Bybit.
            """
            try:
                logger.info("Синхронизация времени с сервером Bybit...")
                
                # --- ИСПРАВЛЕНИЕ НАЧИНАЕТСЯ ЗДЕСЬ ---
                # 1. Вызываем метод API
                response = await asyncio.to_thread(self.session.get_server_time)
                
                # 2. Извлекаем время в миллисекундах из ответа
                server_time_ms = int(response.get("time", 0))

                if server_time_ms == 0:
                    logger.error("Не удалось получить время сервера из ответа API.")
                    return

                # --- ОСТАЛЬНАЯ ЛОГИКА ОСТАЕТСЯ ПРЕЖНЕЙ ---
                server_time_s = server_time_ms / 1000.0
                local_time_s = time.time()
                self.time_offset = server_time_s - local_time_s
                
                # pybit автоматически использует этот offset, если он установлен
                self.session.time_offset = self.time_offset * 1000
                
                logger.info(f"Синхронизация времени завершена. Смещение: {self.time_offset:.3f} секунд.")
            except Exception as e:
                logger.error(f"Не удалось синхронизировать время с сервером: {e}", exc_info=True)


    async def start(self):        
        await self._sync_server_time()
        logger.info(f"[User {self.user_id}] Бот запущен")


        # self.signal_task_queue = mp.Queue()
        # self.order_command_queue = mp.Queue()
        # self.signal_worker_process = mp.Process(
        #     target=start_worker_process,
        #     args=(self.signal_task_queue, self.order_command_queue, self.user_data),
        #     daemon=True
        # )
        # self.signal_worker_process.start()
        # logger.info(f"[SignalWorker] Процесс-аналитик для user {self.user_id} запущен с PID: {self.signal_worker_process.pid}")
        
        asyncio.create_task(self.sync_open_positions_loop())
        asyncio.create_task(self.wallet_loop())
        asyncio.create_task(self._cleanup_recently_closed())
        # asyncio.create_task(self._order_queue_listener())

        asyncio.create_task(self.reload_settings_loop())
        
        await self.update_open_positions()
        await self.setup_private_ws()
        await self._cache_all_symbol_meta()
        
        logger.info(f"Бот для пользователя {self.user_id} полностью готов к работе.")

    async def stop(self):
        logger.info(f"Остановка бота для пользователя {self.user_id}...")
        if self.signal_worker_process and self.signal_worker_process.is_alive():
            self.signal_worker_process.terminate()
            self.signal_worker_process.join()
        
        for symbol in list(self._stop_workers.keys()):
            await self._stop_stop_worker(symbol)
            
        logger.info(f"Бот для пользователя {self.user_id} остановлен.")

    # --- Основные циклы и обработчики ---

    async def run_high_frequency_strategies(self, symbol: str):
        """
        Высокочастотный триггер от тикера. Вызывает диспетчер быстрых стратегий.
        Эта функция вызывается из data_manager при каждом обновлении тикера.
        """
        # Передаем управление диспетчеру в модуле стратегий
        await strategies.high_frequency_dispatcher(self, symbol)

    async def run_low_frequency_strategies(self, symbol: str):
        """
        Низкочастотный триггер от закрытия свечи. Вызывает диспетчер медленных стратегий.
        Эта функция вызывается из data_manager при закрытии каждой минутной свечи.
        """
        # Передаем управление диспетчеру в модуле стратегий
        await strategies.low_frequency_dispatcher(self, symbol)

    async def on_liquidation_event(self, event: dict):
        """
        Обрабатывает событие ликвидации от Public WS и сохраняет его в буфер.
        Эта функция вызывается из data_manager для каждого события ликвидации.
        """
        symbol = event.get("symbol")
        if not symbol:
            return

        # Рассчитываем стоимость ликвидации в USDT для удобства анализа
        price = utils.safe_to_float(event.get("price"))
        size = utils.safe_to_float(event.get("size"))
        value_usd = price * size

        # Игнорируем события с нулевой стоимостью
        if value_usd <= 0:
            return

        # Сохраняем ключевую информацию о событии в соответствующий буфер
        # Буфер - это deque с ограниченной длиной, старые события автоматически удаляются
        self.liq_buffers[symbol].append({
            "ts": time.time(),          # Время получения события
            "side": event.get("side"),  # 'Buy' (ликвидация лонга) или 'Sell' (ликвидация шорта)
            "price": price,             # Цена ликвидации
            "value": value_usd,         # Объем ликвидации в USDT
        })


    async def _entry_guard(self, symbol: str, side: str, candidate: dict | None = None, features: dict | None = None) -> tuple[bool, str]:
        """
        Единый входной фильтр («охотник-сторож»), чтобы не входить:
        - на пампе/дампе против нашей стороны,
        - без отката от локального экстремума,
        - при раздутом спреде.
        Возвращает (ok, reason).
        """
        cfg = getattr(config, "ENTRY_GUARD", {})
        now = time.time()

        # 0) локальный кулдаун по символу/стороне после блокировки импульсом
        cd_key = (symbol, side)
        if now < self.momentum_cooldown_until.get(cd_key, 0.0):
            left = int(self.momentum_cooldown_until[cd_key] - now)
            return False, f"cooldown {left}s"

        # 1) актуальные фичи (если не передали)
        if not features:
            features = await self.extract_realtime_features(symbol)
        if not features:
            # Нет данных — не мешаем входу (или, если хотите, наоборот блокируйте)
            return True, "no_features"

        # Извлекаем, что нужно
        pct1m   = float(features.get("pct1m", 0.0))
        pct5m   = float(features.get("pct5m", 0.0))
        spread  = float(features.get("spread_pct", 0.0))

        dOI1m = float(features.get("dOI1m", features.get("dOI_1m", 0.0)))
        dOI5m = float(features.get("dOI5m", features.get("dOI_5m", 0.0)))
        CVD1m = float(features.get("CVD1m", features.get("CVD_1m", 0.0)))
        CVD5m = float(features.get("CVD5m", features.get("CVD_5m", 0.0)))

        # 1.1) Спред-гард
        if spread > cfg.get("MAX_SPREAD_PCT", 0.25):
            return False, f"spread {spread:.2f}% > {cfg.get('MAX_SPREAD_PCT', 0.25):.2f}%"

        # 2) Анти-чейз по импульсу ПРОТИВ нашей стороны
        pump1 = cfg.get("PUMP_BLOCK_1M_PCT", 1.2)
        pump5 = cfg.get("PUMP_BLOCK_5M_PCT", 3.0)
        dump1 = cfg.get("DUMP_BLOCK_1M_PCT", 1.2)
        dump5 = cfg.get("DUMP_BLOCK_5M_PCT", 3.0)

        req_cvd = bool(cfg.get("REQUIRE_CVD_ALIGNMENT", True))
        req_oi  = bool(cfg.get("REQUIRE_OI_ALIGNMENT", True))

        def aligned_up():
            ok_cvd = (CVD1m > 0 or CVD5m > 0) if req_cvd else True
            ok_oi  = (dOI1m > 0 or dOI5m > 0) if req_oi  else True
            return ok_cvd and ok_oi

        def aligned_down():
            ok_cvd = (CVD1m < 0 or CVD5m < 0) if req_cvd else True
            ok_oi  = (dOI1m < 0 or dOI5m < 0) if req_oi  else True
            return ok_cvd and ok_oi

        if side == "Sell":
            # Памп вверх против шорта?
            if (pct1m > pump1 or pct5m > pump5) and aligned_up():
                self.momentum_cooldown_until[cd_key] = now + cfg.get("MOMENTUM_COOLDOWN_SEC", 90)
                return False, f"anti-chase: pump {pct1m:.2f}/{pct5m:.2f}%"
        else:  # side == "Buy"
            # Дамп вниз против лонга?
            if (pct1m < -dump1 or pct5m < -dump5) and aligned_down():
                self.momentum_cooldown_until[cd_key] = now + cfg.get("MOMENTUM_COOLDOWN_SEC", 90)
                return False, f"anti-chase: dump {pct1m:.2f}/{pct5m:.2f}%"

        # 3) Минимальный откат от локального экстремума
        min_retrace = cfg.get("MIN_RETRACE_FROM_EXTREME_PCT", 0.4)

        # last price
        last_price = 0.0
        tdata = self.shared_ws.ticker_data.get(symbol) or {}
        last_price = utils.safe_to_float(tdata.get("lastPrice", 0.0))
        if last_price <= 0.0:
            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if candles:
                last_price = utils.safe_to_float(candles[-1].get("closePrice", 0.0))

        extreme_price = None
        if candidate and "extreme_price" in candidate:
            extreme_price = utils.safe_to_float(candidate.get("extreme_price"))

        if not extreme_price:
            last5 = list(self.shared_ws.candles_data.get(symbol, []))[-5:]
            if last5:
                if side == "Sell":
                    extreme_price = max(utils.safe_to_float(c.get("highPrice")) for c in last5)
                else:
                    extreme_price = min(utils.safe_to_float(c.get("lowPrice")) for c in last5)

        if last_price and extreme_price:
            if side == "Sell":
                retrace = (extreme_price - last_price) / extreme_price * 100.0
                if retrace < min_retrace:
                    return False, f"need retrace {min_retrace:.2f}%, got {retrace:.2f}%"
            else:
                retrace = (last_price - extreme_price) / extreme_price * 100.0
                if retrace < min_retrace:
                    return False, f"need retrace {min_retrace:.2f}%, got {retrace:.2f}%"

        return True, "ok"


    async def reload_settings_loop(self, interval: int = 15):
        """
        Периодически проверяет user_state.json на изменения и применяет их "на лету".
        """
        # Сохраняем начальную конфигурацию для сравнения
        last_known_config = self.user_data.copy()

        while True:
            await asyncio.sleep(interval)
            try:
                with open(config.USER_STATE_FILE, 'r', encoding="utf-8") as f:
                    all_configs = json.load(f)
                
                new_config = all_configs.get(str(self.user_id))

                # Если конфиг изменился, применяем новые настройки
                if new_config and new_config != last_known_config:
                    logger.info(f"Обнаружены новые настройки для пользователя {self.user_id}. Применяю...")
                    
                    # 1. Обновляем основной словарь с данными
                    self.user_data = new_config
                    # 2. Вызываем метод, который раскидает эти данные по переменным класса
                    self.apply_user_settings()
                    # 3. Обновляем "слепок" конфига для следующей проверки
                    last_known_config = new_config.copy()
                    
                    logger.info("Настройки успешно применены онлайн.")

            except FileNotFoundError:
                # Это нормально, если файл еще не создан
                pass
            except Exception as e:
                logger.warning(f"Ошибка при онлайн-перезагрузке настроек: {e}")



    async def sync_open_positions_loop(self, interval: int = 30):
        """Периодически синхронизирует состояние позиций с биржей."""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.update_open_positions()
            except Exception as e:
                logger.error(f"Ошибка в цикле синхронизации позиций: {e}", exc_info=True)

    async def wallet_loop(self):
        wallet_logger = logging.getLogger("wallet_state")
        while True:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_wallet_balance(accountType="UNIFIED")
                )
                wallet_list = resp.get("result", {}).get("list", [])
                if wallet_list:
                    im = utils.safe_to_float(wallet_list[0].get("totalInitialMargin", 0))
                    if abs(im - self.current_total_volume) > 1e-6:
                        self.current_total_volume = im
                        wallet_logger.info(f"[User {self.user_id}] IM={im:.2f}")
            except Exception as e:
                wallet_logger.debug(f"[wallet_loop] error: {e}")
            
            sleep_s = 10.0 if self.current_total_volume > 0 else 30.0
            await asyncio.sleep(sleep_s)

    # async def _order_queue_listener(self):
    #     logger.info(f"[OrderListener] User {self.user_id} запущен.")
    #     while True:
    #         try:
    #             if not self.order_command_queue.empty():
    #                 order_data = self.order_command_queue.get_nowait()
    #                 if order_data and order_data.get("action") == "EXECUTE_ORDER":
    #                     logger.info(f"[OrderListener] Получен приказ на исполнение: {order_data}")
    #                     candidate = {
    #                         "symbol": order_data["symbol"], "side": order_data["side"],
    #                         "source": order_data["source"], "volume_usdt": order_data.get("volume_usdt"),
    #                         "justification": order_data.get("justification", "")
    #                     }
    #                     asyncio.create_task(self.execute_trade_entry(candidate, {}))
    #         except Exception as e:
    #             logger.error(f"Критическая ошибка в OrderListener: {e}", exc_info=True)
    #         await asyncio.sleep(0.1)

    # --- Взаимодействие с биржей и состояние ---

    async def execute_trade_entry(self, candidate: dict, features: dict):
        """
        [ФИНАЛЬНАЯ ВЕРСИЯ] Атомарное исполнение входа с надежной блокировкой и
        корректным учетом pending-ордеров.
        """
        symbol = candidate.get("symbol")
        side = candidate.get("side")
        source_comment = candidate.get("justification", candidate.get("source", "N/A"))

        # Используем единый лок на все операции с ордерами, чтобы избежать гонок
        async with self.pending_orders_lock:
            # --- Финальные проверки под локом ---
            if symbol in self.open_positions or symbol in self.pending_orders:
                logger.warning(f"[EXECUTE_SKIP] Позиция по {symbol} уже существует или в процессе открытия. Вход отменен.")
                return

            volume_to_open = self.POSITION_VOLUME
            effective_total_vol = await self.get_effective_total_volume() # <--- ИСПОЛЬЗУЕМ НОВЫЙ МЕТОД
            
            if effective_total_vol + volume_to_open > self.MAX_TOTAL_VOLUME:
                logger.warning(f"[EXECUTE_REJECT] Превышен лимит общего объема. Текущий: {effective_total_vol:.2f}, Попытка: {volume_to_open:.2f}, Лимит: {self.MAX_TOTAL_VOLUME:.2f}")
                return

            # --- Резервируем место ---
            self.pending_orders[symbol] = volume_to_open
            self.pending_timestamps[symbol] = time.time()
        
        # --- Расчет и отправка ордера (уже вне лока) ---
        try:
            qty = await self._calc_qty_from_usd(symbol, volume_to_open)
            if qty <= 0:
                raise ValueError("Рассчитан нулевой или отрицательный объем.")

            logger.info(f"🚀 [EXECUTION] Исполнение входа: {symbol} {side}, Qty: {qty:.4f}")
            await self.place_unified_order(
                symbol=symbol, side=side, qty=qty, 
                order_type="Market", comment=source_comment
            )
            self.last_entry_ts[symbol] = time.time()

        except Exception as e:
            logger.error(f"[EXECUTE_CRITICAL] Критическая ошибка при исполнении входа для {symbol}: {e}", exc_info=True)
            # Снимаем резерв при любой ошибке
            ok, reason = await self._entry_guard(symbol, side, candidate, features)
            if not ok:
                logger.info(f"[ENTRY_GUARD][{symbol}] {side}: вход заблокирован — {reason}")
                return
            async with self.pending_orders_lock:
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)

    def _get_trailing_params(self) -> tuple[float, float]:
        """
        Централизованно получает параметры трейлинга (start_roi, gap_roi)
        из настроек пользователя для текущего режима стратегии.
        Возвращает значения по умолчанию, если ничего не найдено.
        """
        # Значения по умолчанию, если ничего не будет найдено
        default_start = 5.0
        default_gap = 2.5 # Ваш желаемый отступ
        
        # Получаем текущие настройки пользователя
        user_settings = self.user_data or {}
        mode = user_settings.get("strategy_mode", "full")

        # Извлекаем карты порогов и отступов для разных режимов
        start_map = user_settings.get("trailing_start_pct", {})
        gap_map = user_settings.get("trailing_gap_pct", {})

        # Получаем значение для текущего режима или глобальное значение, или дефолт
        start_roi = start_map.get(mode, start_map.get("full", default_start))
        gap_roi = gap_map.get(mode, gap_map.get("full", default_gap))

        return float(start_roi), float(gap_roi)

    def _resolve_avg_price(self, symbol: str, pos: dict) -> float:
        """
        Гарантированно возвращает avg_price (>0) для позиции, при необходимости
        используя буфер "раннего исполнения" (pending_open_exec).
        """
        # Сначала пробуем стандартные поля
        avg = utils.safe_to_float(pos.get("avg_price") or pos.get("entry_price"))
        if avg > 0:
            return avg
        
        # Если цена 0, возможно, позиция только что открылась и мы поймали execution раньше
        pend_exec = self.pending_open_exec.get(symbol)
        if pend_exec and pend_exec.get("side") == pos.get("side"):
            avg_from_pend = utils.safe_to_float(pend_exec.get("price"))
            if avg_from_pend > 0:
                # Сразу "усыновляем" эту цену, чтобы не ждать
                pos["avg_price"] = avg_from_pend
                return avg_from_pend
                
        return 0.0


    async def place_unified_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs):
            cid = kwargs.get("cid") or utils.new_cid()
            pos_idx = 1 if side == "Buy" else 2
            comment = kwargs.get("comment", "")
            if comment:
                self.pending_strategy_comments[symbol] = comment
            
            params = {
                "category":"linear", "symbol":symbol, "side":side, "orderType":order_type,
                "qty": f"{qty:.12f}".rstrip("0").rstrip("."), "timeInForce":"GTC",
                "positionIdx": pos_idx, "orderLinkId": cid
            }
            if order_type == "Limit" and (price := kwargs.get("price")) is not None:
                params["price"] = str(price)

            logger.info(f"➡️ [ORDER_SENDING][{cid}] {symbol} {side} {order_type} qty={params['qty']}")
            try:
                # Библиотека pybit сама выбросит исключение InvalidRequestError при retCode != 0
                resp = await asyncio.to_thread(self.session.place_order, **params)
                
                # Если исключения не было, значит все успешно
                order_id = resp.get("result", {}).get("orderId", "")
                logger.info(f"✅ [ORDER_ACCEPTED][{cid}] {symbol} id={order_id or 'n/a'}")
                return resp

            except InvalidRequestError as e:
                # --- НОВЫЙ НАДЕЖНЫЙ ОБРАБОТЧИК ОШИБОК ---
                error_text = str(e)
                
                # Ищем код ошибки прямо в тексте сообщения
                if "(ErrCode: 110100)" in error_text:
                    logger.warning(f"❌ [ORDER_REJECTED][{cid}] {symbol} не торгуется (Pre-Market на демо). Блокирую на 24 часа.")
                    # Добавляем в список "неудачных" ордеров с очень длинным кулдауном
                    self.failed_orders[symbol] = time.time() + 86400 
                
                # Логируем полную ошибку, но уже без трассировки, т.к. мы ее обработали
                logger.error(f"💥 [ORDER_API_FAIL][{cid}] {symbol}: {error_text}")
                raise  # Передаем исключение дальше, чтобы execute_trade_entry знал об ошибке

            except Exception as e:
                logger.error(f"💥 [ORDER_CRITICAL_FAIL][{cid}] {symbol}: {e}", exc_info=True)


    @async_retry(max_retries=5, delay=3)
    async def update_open_positions(self):
        """
        [ФИНАЛЬНАЯ ВЕРСИЯ] Синхронизирует позиции, корректно "усыновляя" их с реальной ценой.
        """
        response = await asyncio.to_thread(lambda: self.session.get_positions(category="linear", settleCoin="USDT"))
        if response.get("retCode") != 0:
            raise ConnectionError(f"API Error: {response.get('retMsg')}")
        live_positions = {p["symbol"]: p for p in response.get("result", {}).get("list", []) if utils.safe_to_float(p.get("size", 0)) > 0}
        async with self.position_lock:
            for symbol, pos_data in live_positions.items():
                if symbol not in self.open_positions:
                    logger.info(f"[SYNC] Обнаружена существующая позиция: {symbol}. Адаптация...")
                    side = pos_data.get("side", "")
                    self.open_positions[symbol] = {
                        "avg_price": utils.safe_to_float(pos_data.get("avgPrice")), "side": side,
                        "pos_idx": 1 if side == 'Buy' else 2, "volume": utils.safe_to_float(pos_data.get("size")),
                        "leverage": utils.safe_to_float(pos_data.get("leverage", "1")), "comment": "Adopted on startup"
                    }
                    if symbol not in self.watch_tasks:
                        task = asyncio.create_task(self.manage_open_position(symbol))
                        self.watch_tasks[symbol] = task
            for symbol in list(self.open_positions.keys()):
                if symbol not in live_positions:
                    logger.info(f"[SYNC] Позиция {symbol} больше не активна. Очистка.")
                    if task := self.watch_tasks.pop(symbol, None): task.cancel()
                    self._purge_symbol_state(symbol)
        utils._atomic_json_write(config.OPEN_POS_JSON, self.open_positions)


    async def handle_execution(self, msg: dict):
        for exec_data in msg.get("data", []):
            symbol = exec_data.get("symbol")
            if not symbol: continue

            async with self.position_lock:
                pos = self.open_positions.get(symbol)
                if not pos:
                    # Если позиции еще нет, запоминаем "раннее" исполнение
                    if exec_data.get("execPrice"):
                        self.pending_open_exec[symbol] = {
                            "price": utils.safe_to_float(exec_data.get("execPrice")),
                            "side": exec_data.get("side"), "ts": time.time()
                        }
                    continue

                # --- Сценарий 1: ПЕРВОЕ исполнение для новой позиции ---
                if pos.get("is_opening") and exec_data.get("side") == pos.get("side"):
                    exec_price = utils.safe_to_float(exec_data.get("execPrice"))
                    pos["avg_price"] = exec_price
                    pos["comment"] = self.pending_strategy_comments.pop(symbol, "N/A")
                    pos.pop("is_opening")

                    logger.info(f"[EXECUTION_OPEN] {pos['side']} {symbol} {pos['volume']:.3f} @ {exec_price:.6f}")
                    
                    await self.log_trade(
                        symbol=symbol, side=pos['side'], avg_price=exec_price,
                        volume=pos['volume'], action="open", result="opened",
                        comment=pos['comment']
                    )

                    # --- ЗАПУСКАЕМ ХРАНИТЕЛЯ ---
                    if symbol not in self.watch_tasks:
                        task = asyncio.create_task(self.manage_open_position(symbol))
                        self.watch_tasks[symbol] = task

                # --- Сценарий 2: Закрытие позиции ---
                elif exec_data.get("side") != pos.get("side"):
                    if utils.safe_to_float(exec_data.get("leavesQty", 0)) == 0:
                        entry_price = self._resolve_avg_price(symbol, pos)
                        exit_price = utils.safe_to_float(exec_data.get("execPrice"))
                        pos_volume = utils.safe_to_float(pos.get("volume", 0))
                        
                        pnl_usdt = utils.calc_pnl(pos.get("side", "Buy"), entry_price, exit_price, pos_volume)
                        position_value = entry_price * pos_volume
                        pnl_pct = (pnl_usdt / position_value) * 100 if position_value > 0 else 0.0

                        logger.info(f"[EXECUTION_CLOSE] {symbol}. PnL: {pnl_usdt:.2f} USDT ({pnl_pct:.3f}%).")
                        
                        await self.log_trade(
                            symbol=symbol, side=pos['side'], avg_price=exit_price, volume=pos_volume,
                            action="close", result="closed_by_execution", pnl_usdt=pnl_usdt,
                            pnl_pct=pnl_pct, comment=pos.get('comment')
                        )
                        
                        self._purge_symbol_state(symbol)



    async def handle_position_update(self, msg: dict):
        async with self.position_lock:
            for p in msg.get("data", []):
                symbol = p.get("symbol")
                if not symbol: continue

                new_size = utils.safe_to_float(p.get("size", 0))
                is_new_pos = symbol not in self.open_positions and new_size > 0

                # --- Открытие новой позиции ---
                if is_new_pos:
                    side = p.get("side")
                    if not side: continue

                    self.pending_orders.pop(symbol, None)

                    self.open_positions[symbol] = {
                        "avg_price": 0.0, "side": side,
                        "volume": new_size, "leverage": utils.safe_to_float(p.get("leverage")),
                        "comment": None, "is_opening": True
                    }
                    logger.info(f"[PositionStream] NEW_PRELIMINARY {side} {symbol} {new_size:.3f}")

                    # Если "раннее" исполнение уже пришло, "усыновляем" его
                    pend = self.pending_open_exec.pop(symbol, None)
                    if pend and pend.get("side") == side:
                        pos = self.open_positions[symbol]
                        pos["avg_price"] = pend["price"]
                        pos.pop("is_opening")
                        pos["comment"] = self.pending_strategy_comments.pop(symbol, "N/A")
                        
                        logger.info(f"[EXECUTION_OPEN][adopted] {side} {symbol} {pos['volume']:.3f} @ {pos['avg_price']:.6f}")
                        
                        await self.log_trade(
                            symbol=symbol, side=side, avg_price=pos["avg_price"],
                            volume=pos["volume"], action="open", result="opened(adopted)",
                            comment=pos["comment"]
                        )
                        
                        # --- ЗАПУСКАЕМ ХРАНИТЕЛЯ ---
                        if symbol not in self.watch_tasks:
                            task = asyncio.create_task(self.manage_open_position(symbol))
                            self.watch_tasks[symbol] = task

                # --- Закрытие (size=0) ---
                elif symbol in self.open_positions and new_size == 0:
                    logger.debug(f"[PositionStream] {symbol} size=0. Закрытие будет обработано execution handler.")



    async def setup_private_ws(self):
        while True:
            try:
                def _on_private(msg):
                    try:
                        if not self.loop.is_closed():
                            asyncio.run_coroutine_threadsafe(self.route_private_message(msg), self.loop)
                    except Exception as e:
                        logger.warning(f"PrivateWS callback error: {e}")

                self.ws_private = WebSocket(
                    testnet=False, demo=self.mode == "demo", channel_type="private",
                    api_key=self.api_key, api_secret=self.api_secret,
                    ping_interval=30, ping_timeout=15, restart_on_error=True, retries=200
                )
                self.ws_private.position_stream(callback=_on_private)
                self.ws_private.execution_stream(callback=_on_private)
                logger.info(f"Private WebSocket для user {self.user_id} подключен.")
                break
            except Exception as e:
                logger.warning(f"Ошибка подключения Private WS: {e}, повтор через 5с.")
                await asyncio.sleep(5)

    async def route_private_message(self, msg):
        topic = (msg.get("topic") or "").lower()
        if "position" in topic:
            await self.handle_position_update(msg)
        elif "execution" in topic:
            await self.handle_execution(msg)


    # ======================================================================
    # 4. Оркестрация стратегий и AI (Продолжение)
    # ======================================================================

    async def _process_signal(self, candidate: dict, features: dict, signal_key: tuple):
            """
            [ВЕРСИЯ С PLUTUS] Обрабатывает сигналы, используя AI-советника
            (plutus) в качестве основного принимающего решения.
            """
            symbol = candidate.get("symbol")
            source = candidate.get("source", "")
            
            try:
                # Предварительная программная проверка на перегретость для Golden Setup
                if 'golden_setup' in source:
                    side = candidate.get("side")
                    pct_30m = utils.compute_pct(self.shared_ws.candles_data.get(symbol, []), 30)
                    features['pct_30m'] = pct_30m
                    REJECTION_THRESHOLD = 7.0
                    is_overheated = (side == "Buy" and pct_30m > REJECTION_THRESHOLD) or \
                                    (side == "Sell" and pct_30m < -REJECTION_THRESHOLD)
                    if is_overheated:
                        reason = f"Отклонено кодом: вход в {side} после движения {pct_30m:.2f}% за 30 мин."
                        logger.info(f"🔥 [{symbol}] СИГНАЛ ОТКЛОНЕН (ПЕРЕГРЕТОСТЬ). {reason}")
                        self.active_signals.discard(signal_key)
                        return

                # Формируем промпт. Он должен быть адаптирован под новую модель.
                prompt = ai_ml.build_primary_prompt(candidate, features, self.shared_ws)
                
                logger.debug(f"Сигнал {signal_key} передан AI-аналитику (модель: {self.ai_advisor_model})...")
                
                # --- ГЛАВНОЕ ИЗМЕНЕНИЕ: ВЫЗЫВАЕМ ДРУГУЮ МОДЕЛЬ ---
                ai_response = await ai_ml.ask_ollama_json(
                    self.ai_advisor_model,          # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
                    [{"role": "user", "content": prompt}],
                    self.ai_timeout_sec,
                    self.ollama_advisor_openai      # <--- И ИЗМЕНЕНИЕ ЗДЕСЬ
                )
                # ----------------------------------------------------
                
                action = ai_response.get("action", "REJECT").upper()

                if action == "EXECUTE":
                    logger.info(f"✅ [{symbol}] Сигнал ОДОБРЕН AI ({self.ai_advisor_model}). Причина: {ai_response.get('justification')}")
                    candidate['justification'] = ai_response.get('justification')
                    await self.execute_trade_entry(candidate, features)
                
                # Мы убираем WATCH, так как plutus будет давать финальное решение
                # elif action == "WATCH":
                #     asyncio.create_task(self._hunt_entry_point(candidate, features, signal_key))

                else: # REJECT
                    logger.info(f"❌ [{symbol}] Сигнал ОТКЛОНЕН AI ({self.ai_advisor_model}). Причина: {ai_response.get('justification')}")
            
            except Exception as e:
                logger.error(f"Ошибка в _process_signal для {signal_key}: {e}", exc_info=True)
            finally:
                self.active_signals.discard(signal_key)
                if symbol:
                    self.strategy_cooldown_until[symbol] = time.time() + 60




    async def _hunt_entry_point(self, candidate: dict, features: dict, signal_key: tuple):
            """
            [НОВАЯ ВЕРСИЯ] Запускает "тактическое наблюдение" за перспективным,
            но еще не созревшим сигналом.
            """
            symbol = candidate["symbol"]
            side = candidate["side"]
            try:
                logger.info(f"🎯 [{symbol}] AI одобрил сигнал для наблюдения. Активирован 'Охотник'...")

                timeout = self.tactical_entry_window_sec # Например, 90 секунд из конфига
                start_time = time.time()
                
                # Определяем цену экстремума на момент обнаружения сигнала
                last_5_candles = list(self.shared_ws.candles_data.get(symbol, []))[-5:]
                if not last_5_candles: return

                if side == "Sell":
                    extreme_price = max(utils.safe_to_float(c.get('highPrice')) for c in last_5_candles)
                else:
                    extreme_price = min(utils.safe_to_float(c.get('lowPrice')) for c in last_5_candles)

                while time.time() - start_time < timeout:
                    # Проверяем, не открыли ли мы уже позицию по этому символу
                    if symbol in self.open_positions or symbol in self.pending_orders:
                        logger.info(f"[{symbol}] Позиция уже открыта. 'Охотник' завершает работу.")
                        return

                    # Запрашиваем подтверждение у тактического советника
                    if await self._ai_confirm_entry(symbol, side, extreme_price):
                        logger.info(f"✅ [HUNT SUCCESS] {symbol}/{side}: AI-советник дал команду на вход! Исполняем.")
                        await self.execute_trade_entry(candidate, features)
                        return # Успех, выходим из цикла и функции
                    
                    # Ждем перед следующей проверкой
                    await asyncio.sleep(self.squeeze_ai_confirm_interval_sec)

                logger.warning(f"⏳ [HUNT TIMEOUT] {symbol}/{side}: Окно для входа ({timeout}s) истекло. Сигнал отменен.")

            except Exception as e:
                logger.error(f"💥 [HUNT FAIL] {symbol}/{side}: Критическая ошибка в 'Охотнике': {e}", exc_info=True)
            finally:
                # Важно: убираем сигнал из активных только после завершения охоты
                self.active_signals.discard(signal_key)


    # async def _ai_confirm_entry(self, symbol: str, side: str, extreme_price: float) -> bool:
    #     """
    #     Запрашивает у AI-советника, является ли ТЕКУЩИЙ момент оптимальным для входа.
    #     """
    #     try:
    #         last_price = utils.safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
    #         if last_price <= 0: return False

    #         features = await self.extract_realtime_features(symbol)
    #         if not features: return False
            
    #         # Используем специализированный промпт для тактического анализа
    #         prompt = ai_ml.build_squeeze_entry_prompt(symbol, side, extreme_price, last_price, features)
            
    #         ai_response = await ai_ml.ask_ollama_json(
    #             self.ai_advisor_model, # Используем второго, быстрого советника
    #             [{"role": "user", "content": prompt}],
    #             timeout_s=15.0, # Короткий таймаут для быстрого решения
    #             base_url=self.ollama_advisor_openai
    #         )

    #         if ai_response.get("action", "WAIT").upper() == "EXECUTE":
    #             logger.debug(f"[{symbol}] Тактический AI-советник одобрил вход.")
    #             return True
    #         return False
    #     except Exception as e:
    #         logger.warning(f"[_ai_confirm_entry] Ошибка консультации с AI для {symbol}: {e}")
    #         return False


    async def _ai_confirm_entry(self, symbol: str, side: str, extreme_price: float) -> bool:
            """
            [ЖЕЛЕЗОБЕТОННАЯ ВЕРСИЯ] Выполняет всю логику подтверждения входа в Python,
            полностью исключая ошибки интерпретации AI.
            """
            try:
                # --- Шаг 1: Сбор свежих данных ---
                last_price = utils.safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
                if last_price <= 0: return False

                features = await self.extract_realtime_features(symbol)
                if not features: return False
                
                rsi_val = utils.safe_to_float(features.get('rsi14'))
                cvd_1m = features.get('CVD1m', 0)
                vol_anomaly = features.get('volume_anomaly', 0.0) # Предполагаем, что extract_realtime_features считает это поле

                # --- Шаг 2: Выполнение строгих правил в Python ---
                base_condition_met = False
                if side.upper() == 'SELL' and rsi_val > 75:
                    base_condition_met = True
                elif side.upper() == 'BUY' and rsi_val < 25:
                    base_condition_met = True

                # Если базовое условие не выполнено, немедленно выходим
                if not base_condition_met:
                    logger.debug(f"[{symbol}] Hunt check: WAIT. Reason: Base RSI condition not met (RSI: {rsi_val:.1f})")
                    return False

                # Если базовое условие выполнено, ищем хотя бы одно подтверждение
                cvd_confirms = (side.upper() == 'SELL' and cvd_1m < 0) or \
                            (side.upper() == 'BUY' and cvd_1m > 0)
                volume_confirms = vol_anomaly > 2.5

                if cvd_confirms or volume_confirms:
                    logger.info(f"[{symbol}] Hunt check: EXECUTE! Reason: Base RSI OK, confirmation found (CVD: {cvd_confirms}, Vol: {volume_confirms})")
                    return True
                else:
                    logger.debug(f"[{symbol}] Hunt check: WAIT. Reason: Base RSI OK, but no confirmation yet.")
                    return False

            except Exception as e:
                logger.warning(f"[_ai_confirm_entry] Ошибка в программной проверке для {symbol}: {e}")
                return False



    # async def _hunt_squeeze_entry_point(self, candidate: dict, features: dict, signal_key: tuple):
    #     """
    #     Запускает "охоту" за точкой входа для сквиза: сначала стратегическое одобрение,
    #     затем тактический поиск с помощью AI-советника.
    #     """
    #     symbol = candidate["symbol"]
    #     side = candidate["side"]
    #     try:
    #         logger.debug(f"[{symbol}] Сигнал Squeeze передан на стратегическую оценку AI.")
            
    #         # Фаза 1: Стратегическое одобрение от основного AI
    #         prompt = ai_ml.build_primary_prompt(candidate, features, self.shared_ws)
    #         messages = [{"role": "user", "content": prompt}]
    #         ai_response = await ai_ml.ask_ollama_json(
    #             self.ai_primary_model, messages, self.ai_timeout_sec, self.ollama_primary_openai
    #         )
            
    #         if ai_response.get("action") != "EXECUTE":
    #             logger.info(f"[AI_REJECT] {symbol}/{side} (squeeze) — {ai_response.get('justification', 'N/A')}")
    #             return

    #         candidate['justification'] = ai_response.get('justification', 'Одобрено основным AI')
    #         logger.info(f"[AI_CONFIRM] {symbol}/{side} (squeeze) ОДОБРЕНА. Начинаем тактическую охоту...")

    #         # Фаза 2: Тактическая охота за точкой входа
    #         timeout = self.tactical_entry_window_sec
    #         start_time = time.time()
            
    #         # Определяем цену экстремума на момент обнаружения сигнала
    #         last_5_candles = list(self.shared_ws.candles_data.get(symbol, []))[-5:]
    #         if not last_5_candles: return

    #         if side == "Sell": # Если входим в шорт, значит был рост, ищем максимум
    #             extreme_price = max(utils.safe_to_float(c.get('highPrice')) for c in last_5_candles)
    #         else: # Если входим в лонг, значит было падение, ищем минимум
    #             extreme_price = min(utils.safe_to_float(c.get('lowPrice')) for c in last_5_candles)

    #         while time.time() - start_time < timeout:
    #             await asyncio.sleep(0.01) # Даем шанс event loop обработать сеть
    #             if await self._ai_confirm_squeeze_entry(symbol, side, extreme_price):
    #                 logger.info(f"✅ [AI_EXECUTE] {symbol}/{side}: AI-советник дал команду на вход! Исполняем.")
    #                 await self.execute_trade_entry(candidate, features)
    #                 return
    #             await asyncio.sleep(self.squeeze_ai_confirm_interval_sec)

    #         logger.warning(f"[HUNT_TIMEOUT] {symbol}/{side}: Окно входа ({timeout}s) истекло. Вход отменен.")
    #     except Exception as e:
    #         logger.error(f"[HUNT_FAIL] {symbol}/{side}: Критическая ошибка: {e}", exc_info=True)
    #     finally:
    #         self.active_signals.discard(signal_key)

    # async def _ai_confirm_squeeze_entry(self, symbol: str, side: str, extreme_price: float) -> bool:
    #     """
    #     Запрашивает у AI-советника, является ли текущий момент оптимальным для входа в сквиз.
    #     """
    #     try:
    #         last_price = utils.safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
    #         if last_price <= 0: return False

    #         features = await self.extract_realtime_features(symbol)
    #         if not features: return False
            
    #         prompt = ai_ml.build_squeeze_entry_prompt(symbol, side, extreme_price, last_price, features)
    #         messages = [{"role": "user", "content": prompt}]

    #         ai_response = await ai_ml.ask_ollama_json(
    #             self.ai_advisor_model, messages, 45.0, self.ollama_advisor_openai
    #         )

    #         if ai_response.get("action", "WAIT").upper() == "EXECUTE":
    #             logger.info(f"[AI_SQUEEZE_EXECUTE] {symbol}: Советник одобрил вход. Причина: {ai_response.get('reason', 'N/A')}")
    #             return True
    #         return False
    #     except Exception as e:
    #         logger.warning(f"[_ai_confirm_squeeze_entry] Ошибка консультации с AI для {symbol}: {e}")
    #         return False



    async def manage_open_position(self, symbol: str):
        """
        [ФИНАЛЬНАЯ ВЕРСИЯ] Единый "Хранитель" позиции.
        1. Устанавливает защитный стоп.
        2. Запускает AI-советника (если нужно).
        """
        logger.info(f"🛡️ [Guardian] Активирован для позиции {symbol}.")
        try:
            pos = self.open_positions.get(symbol)
            if not pos: return

            # Шаг 1: Устанавливаем первоначальный защитный стоп
            if not pos.get("initial_stop_set"):
                await self._set_initial_stop_loss(symbol, pos)

            # Шаг 2: Основной цикл консультаций с AI
            # Примечание: AI-советник будет работать всегда, а не только после активации трейлинга
            AI_ADVISOR_INTERVAL_SEC = 120 
            while symbol in self.open_positions:
                await asyncio.sleep(AI_ADVISOR_INTERVAL_SEC)
                await self._ai_advise_on_stop(symbol)

        except asyncio.CancelledError:
            logger.info(f"[Guardian] Наблюдение за {symbol} отменено.")
        except Exception as e:
            logger.error(f"[Guardian] {symbol} критическая ошибка: {e}", exc_info=True)
        finally:
            logger.info(f"🛡️ [Guardian] Завершает наблюдение за {symbol}.")
            # Воркер больше не нужен, ничего не останавливаем




    async def _calculate_fibonacci_stop_price(self, symbol: str, side: str) -> Optional[float]:
            """
            [ИСПРАВЛЕННАЯ ВЕРСИЯ] Корректно рассчитывает уровень стоп-лосса
            на основе Фибоначчи для контртрендовых входов.
            """
            try:
                LOOKBACK_MINUTES = 30
                FIB_LEVEL = 0.618

                candles = list(self.shared_ws.candles_data.get(symbol, []))
                if len(candles) < LOOKBACK_MINUTES:
                    logger.warning(f"[{symbol}] Недостаточно свечей для Фибоначчи.")
                    return None

                recent_candles = candles[-LOOKBACK_MINUTES:]
                
                highest_high = max(utils.safe_to_float(c.get("highPrice")) for c in recent_candles)
                lowest_low = min(utils.safe_to_float(c.get("lowPrice")) for c in recent_candles)
                price_range = highest_high - lowest_low

                if price_range == 0: return None

                # --- НОВАЯ ЛОГИКА ---
                if side.lower() == "buy":
                    # Мы вошли в лонг ПОСЛЕ ПАДЕНИЯ. Нам нужно найти уровень сопротивления выше дна,
                    # чтобы цена его пробила, а наш стоп был ниже дна.
                    # Поэтому мы считаем от ДНА (lowest_low).
                    fib_level = lowest_low + (price_range * FIB_LEVEL)
                    return fib_level # Возвращаем сам уровень, а буфер добавится позже
                else: # side.lower() == "sell"
                    # Мы вошли в шорт ПОСЛЕ РОСТА. Нам нужно найти уровень поддержки ниже пика.
                    # Считаем от ВЕРШИНЫ (highest_high).
                    fib_level = highest_high - (price_range * FIB_LEVEL)
                    return fib_level
                
            except Exception as e:
                logger.error(f"[{symbol}] Ошибка при расчете уровня Фибоначчи: {e}")
                return None



    async def _ai_advise_on_stop(self, symbol: str):
        """
        Запрашивает совет у AI по управлению стоп-лоссом для открытой позиции.
        """
        try:
            pos = self.open_positions.get(symbol)
            if not pos:
                return

            features = await self.extract_realtime_features(symbol)
            if not features:
                logger.warning(f"[{symbol}] Не удалось получить фичи для AI-советника.")
                return

            # Дополняем данные о позиции последним известным стопом
            pos['last_stop_price'] = self.last_stop_price.get(symbol)

            prompt = ai_ml.build_stop_management_prompt(symbol, pos, features)
            messages = [{"role": "user", "content": prompt}]

            logger.info(f"🤖 [{symbol}] Запрос совета у AI-риск-менеджера...")
            
            ai_response = await ai_ml.ask_ollama_json(
                self.ai_advisor_model, 
                messages, 
                timeout_s=45.0, # Даем советнику чуть больше времени
                base_url=self.ollama_advisor_openai
            )

            action = ai_response.get("action", "").upper()
            
            if action == "MOVE_STOP":
                new_price = utils.safe_to_float(ai_response.get("new_stop_price"))
                reason = ai_response.get("reason", "N/A")
                if new_price > 0:
                    logger.info(f"🤖✅ [{symbol}] AI-советник РЕКОМЕНДОВАЛ переместить стоп на {new_price:.6f}. Причина: {reason}")
                    await self.set_or_amend_stop_loss(new_price, symbol=symbol)
            else:
                logger.info(f"🤖 HOLD [{symbol}] AI-советник рекомендует держать текущий стоп.")

        except Exception as e:
            logger.error(f"[{symbol}] Ошибка в работе AI-советника: {e}", exc_info=True)


    async def _set_initial_stop_loss(self, symbol: str, pos: dict):
        """
        [ФИНАЛЬНАЯ ВЕРСИЯ V5] Устанавливает защитный стоп согласно стратегии:
        1. Приоритет: уровень Фибоначчи + буфер 0.5%.
        2. Фоллбэк (если Фибо не рассчитан): стоп по макс. риску.
        """
        try:
            avg_price = self._resolve_avg_price(symbol, pos)
            if avg_price <= 0: return

            side = str(pos.get("side", "")).lower()
            stop_price = None

            # --- ШАГ 1: Пытаемся рассчитать стоп на основе Фибоначчи ---
            fib_level_price = await self._calculate_fibonacci_stop_price(symbol, side)
            
            if fib_level_price:
                logger.info(f"[{symbol}] Рассчитан базовый уровень Фибоначчи: {fib_level_price:.6f}")
                # Применяем ваш буфер 0.5%, отодвигая стоп ДАЛЬШЕ от уровня
                if side == "buy":
                    # Для лонга стоп должен быть НИЖЕ уровня поддержки
                    stop_price = fib_level_price * 0.995 
                else: # side == "sell"
                    # Для шорта стоп должен быть ВЫШЕ уровня сопротивления
                    stop_price = fib_level_price * 1.005
            
            # --- ШАГ 2: Если Фибоначчи не сработал, используем фоллбэк ---
            if not stop_price:
                logger.warning(f"[{symbol}] Не удалось рассчитать Фибо-стоп, используется фоллбэк по макс. риску.")
                max_stop_pct = utils.safe_to_float(self.user_data.get("max_safety_stop_pct", 2.5)) / 100.0
                if side == "buy":
                    stop_price = avg_price * (1 - max_stop_pct)
                else:
                    stop_price = avg_price * (1 + max_stop_pct)

            # --- ШАГ 3: Финальная проверка на "здравый смысл" ---
            # Стоп для лонга не может быть выше цены входа, и наоборот.
            # Если это произошло (из-за аномалий рынка), принудительно используем фоллбэк.
            if (side == "buy" and stop_price >= avg_price) or \
            (side == "sell" and stop_price <= avg_price):
                logger.error(f"[{symbol}] КРИТИЧЕСКАЯ ЛОГИЧЕСКАЯ ОШИБКА! Рассчитанный стоп ({stop_price:.6f}) находится по неверную сторону от цены входа ({avg_price:.6f}). Принудительно используется фоллбэк.")
                max_stop_pct = utils.safe_to_float(self.user_data.get("max_safety_stop_pct", 2.5)) / 100.0
                if side == "buy":
                    stop_price = avg_price * (1 - max_stop_pct)
                else:
                    stop_price = avg_price * (1 + max_stop_pct)
            
            logger.info(f"🛡️ [{symbol}] Установка первоначального стопа. Цена входа: {avg_price:.6f}, Финальный стоп: {stop_price:.6f}")
            await self.set_or_amend_stop_loss(stop_price, symbol=symbol)

        except Exception as e:
            logger.error(f"[{symbol}] Критическая ошибка при установке первоначального стоп-лосса: {e}", exc_info=True)



    # Здесь начинается следующий метод, например, _start_stop_worker


    # # ЗАМЕНИТЕ ВЕСЬ МЕТОД _START_STOP_WORKER НА ЭТОТ
    # async def _start_stop_worker(self, symbol: str, pos: dict):
    #     # 0) Не запускаем дубликаты
    #     if symbol in self._stop_workers:
    #         logger.warning(f"Попытка запустить дубликат воркера для {symbol}. Пропускаем.")
    #         return

    #     # 1) Получаем метаданные (шаг цены), если их нет
    #     await self.ensure_symbol_meta(symbol)
    #     tick_size = float(self.price_tick_map.get(symbol, 0.0))
    #     if tick_size <= 0:
    #         logger.error(f"[STOP][{symbol}] Не удалось получить tick_size. Воркер не запущен.")
    #         return

    #     # 2) Гарантированно получаем корректную среднюю цену входа
    #     avg_price = self._resolve_avg_price(symbol, pos)
    #     if avg_price <= 0:
    #         # Ждём до 3 секунд, если цена ещё не пришла
    #         for _ in range(30):
    #             await asyncio.sleep(0.1)
    #             avg_price = self._resolve_avg_price(symbol, pos)
    #             if avg_price > 0:
    #                 break
        
    #     if avg_price <= 0:
    #         logger.error(f"[STOP][{symbol}] КРИТИЧЕСКАЯ ОШИБКА: не удалось получить avg_price > 0. Воркер не запущен.")
    #         return

    #     # 3) Получаем параметры трейлинга (старт и отступ) из настроек
    #     start_roi_pct, gap_roi_pct = self._get_trailing_params()

    #     # 4) Получаем последнюю известную цену для мгновенной проверки
    #     initial_price = 0.0
    #     if self.shared_ws and hasattr(self.shared_ws, "ticker_data"):
    #         ticker = self.shared_ws.ticker_data.get(symbol, {})
    #         initial_price = utils.safe_to_float(ticker.get("lastPrice"))
    #     # Фолбэк: если тикера нет, берем цену закрытия последней свечи
    #     if initial_price <= 0 and self.shared_ws and hasattr(self.shared_ws, "candles_data"):
    #         candles = list(self.shared_ws.candles_data.get(symbol, []))
    #         if candles:
    #             initial_price = utils.safe_to_float(candles[-1].get("closePrice"))

    #     # 5) Собираем финальные параметры для запуска воркера
    #     side_norm = "Buy" if str(pos.get("side", "")).lower() == "buy" else "Sell"
    #     init_params = {
    #         "symbol": symbol,
    #         "side": side_norm,
    #         "avg_price": avg_price,
    #         "leverage": float(pos.get("leverage") or self.leverage),
    #         "tick_size": tick_size,
    #         "start_roi": start_roi_pct,
    #         "gap_mode": "roi", # Явно работаем в режиме ROI
    #         "gap_roi_pct": gap_roi_pct,
    #         "hb_interval": 15.0,
    #         "initial_price": initial_price, # Передаем цену для мгновенного "кика"
    #     }

    #     # 6) Запускаем воркер как отдельный процесс
    #     import os, sys, json
    #     worker_script_path = os.path.join(os.path.dirname(__file__), "stop_worker.py")
    #     proc = await asyncio.create_subprocess_exec(
    #         sys.executable, "-u", worker_script_path, json.dumps(init_params, ensure_ascii=False),
    #         stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    #     )

    #     self._stop_workers[symbol] = {
    #         "proc": proc,
    #         "reader_task": asyncio.create_task(self._read_stop_events(symbol, proc)),
    #         "stderr_task": asyncio.create_task(self._read_stop_stderr(symbol, proc.stderr)),
    #     }

    #     logger.info(
    #         "🛡️ [Guardian] stop_worker для %s запущен (PID %s): avg=%.6f, start_roi=%.2f%%, gap_roi=%.2f%%, init_price=%.6f",
    #         symbol, proc.pid, avg_price, start_roi_pct, gap_roi_pct, initial_price
    #     )



    # async def _stop_stop_worker(self, symbol: str):
    #     worker_rec = self._stop_workers.pop(symbol, None)
    #     if not worker_rec: return

    #     proc = worker_rec.get("proc")
    #     if proc and proc.returncode is None:
    #         try:
    #             # Сначала закрываем stdin, чтобы воркер понял, что данных больше не будет
    #             if proc.stdin and not proc.stdin.is_closing():
    #                 proc.stdin.close()

    #             # Мягко ждем завершения процесса
    #             await asyncio.wait_for(proc.wait(), timeout=2.0)
    #         except asyncio.TimeoutError:
    #             logger.warning(f"Stop_worker для {symbol} не завершился штатно, убиваем принудительно.")
    #             proc.kill()
    #         except Exception as e:
    #             logger.error(f"Ошибка при остановке воркера {symbol}: {e}")

    #     # Отменяем таски-читатели
    #     for task_name in ("reader_task", "stderr_task"):
    #         if task := worker_rec.get(task_name):
    #             if not task.done():
    #                 task.cancel()
        
    #     logger.info(f"🛡️ [Guardian] stop_worker для {symbol} остановлен.")



    # async def _read_stop_events(self, symbol: str, proc):
    #     """
    #     Асинхронно читает stdout из воркера с "умным" логированием.
    #     """
    #     logger.info(f"[{symbol}] Запущен listener для команд от stop_worker.")
    #     try:
    #         while proc.returncode is None:
    #             line_bytes = await proc.stdout.readline()
    #             if not line_bytes:
    #                 if symbol in self.open_positions:
    #                     logger.warning(f"[{symbol}] Канал связи с stop_worker закрыт, но позиция еще открыта!")
    #                 break

    #             line_str = line_bytes.decode('utf-8').strip()
    #             if not line_str: continue

    #             try:
    #                 evt = json.loads(line_str)
    #                 event_type = str(evt.get("event", "")).lower()

    #                 if event_type == "hb":
    #                     logger.debug(f"📬 HB [{symbol}]: {line_str}")
                    
    #                 elif event_type == "activated":
    #                     roi = evt.get('roi_pct', 0.0)
    #                     price = evt.get('price', 0.0)
    #                     logger.info(f"✅ [{symbol}] Трейлинг-стоп АКТИВИРОВАН! (ROI: {roi:.2f}%, Цена: {price:.6f})")
                        
    #                     if event := self.trailing_activated_events.get(symbol):
    #                         event.set()


    #                 elif event_type == "stop_update":
    #                     price = utils.safe_to_float(evt.get("stop"))
    #                     reason = evt.get("reason", "trail")
    #                     logger.info(f"📬 [{symbol}] Получена команда от воркера на установку/обновление стопа: {price:.6f} (Причина: {reason})")
    #                     if price > 0:
    #                         if symbol not in self.open_positions:
    #                             logger.warning(f"[{symbol}] Получен стоп от воркера, но позиция уже закрыта. Игнорируем.")
    #                             continue
    #                         # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ВЫЗОВА ---
    #                         await self.set_or_amend_stop_loss(price, symbol=symbol)

                    
    #                 else:
    #                     logger.debug(f"📬 EVT [{symbol}]: {line_str}")

    #             except json.JSONDecodeError:
    #                 logger.error(f"[{symbol}] Не удалось декодировать JSON от stop_worker: {line_str}")
    #             except Exception as e:
    #                 logger.error(f"[{symbol}] Ошибка при обработке команды от воркера: {e}", exc_info=True)

    #     except asyncio.CancelledError:
    #         logger.debug(f"[{symbol}] Listener для stop_worker отменен.")
    #     except Exception as e:
    #         logger.error(f"[{symbol}] Критическая ошибка в listener'е stop_worker: {e}", exc_info=True)
    #     finally:
    #         logger.info(f"[{symbol}] Listener для stop_worker завершил работу (PID: {proc.pid}).")


    # async def on_ticker_update(self, symbol: str, last_price: float):
    #     """
    #     [ФИНАЛЬНАЯ ВЕРСИЯ] Единый обработчик тикера.
    #     1. Выполняет логику трейлинга.
    #     2. Запускает высокочастотные стратегии.
    #     """
    #     pos = self.open_positions.get(symbol)
    #     if pos:
    #         pos["markPrice"] = last_price

    #         # --- ВСЯ ЛОГИКА ТРЕЙЛИНГА ТЕПЕРЬ ЗДЕСЬ ---
    #         start_roi_pct, gap_roi_pct = self._get_trailing_params()
            
    #         # Расчет текущего ROI
    #         avg_price = self._resolve_avg_price(symbol, pos)
    #         if avg_price > 0:
    #             side = pos.get("side", "Buy")
    #             leverage = utils.safe_to_float(pos.get("leverage", 10.0))
                
    #             if side == "Buy":
    #                 pnl = (last_price / avg_price) - 1.0
    #             else:
    #                 pnl = (avg_price / last_price) - 1.0
    #             current_roi = pnl * 100.0 * leverage

    #             # Проверка активации трейлинга
    #             if not self.trailing_activated.get(symbol) and current_roi >= start_roi_pct:
    #                 self.trailing_activated[symbol] = True
    #                 logger.info(f"✅ [{symbol}] ТРЕЙЛИНГ АКТИВИРОВАН! ROI: {current_roi:.2f}%")

    #             # Если трейлинг активен, рассчитываем и двигаем стоп
    #             if self.trailing_activated.get(symbol):
    #                 target_roi = current_roi - gap_roi_pct
    #                 denom = 1.0 + (target_roi / (100.0 * leverage))
                    
    #                 new_stop_price = 0.0
    #                 if side == "Buy":
    #                     new_stop_price = avg_price * denom
    #                 elif denom > 1e-9:
    #                     new_stop_price = avg_price / denom
                    
    #                 if new_stop_price > 0:
    #                     # Вызываем set_or_amend_stop_loss, который уже содержит проверку _is_better
    #                     await self.set_or_amend_stop_loss(new_stop_price, symbol=symbol)

    #     # Запускаем быстрые стратегии для поиска НОВЫХ входов
    #     await strategies.high_frequency_dispatcher(self, symbol)




    # async def _read_stop_events(self, symbol: str, proc):
    #     import json, asyncio
    #     side_cached = (self.open_positions.get(symbol, {}).get("side") or "").lower()
    #     try:
    #         while True:
    #             line = await proc.stdout.readline()
    #             if not line:
    #                 break
    #             s = line.decode("utf-8", "ignore").strip()
    #             if not s:
    #                 continue
    #             try:
    #                 evt = json.loads(s)
    #             except Exception:
    #                 logger.debug("[STOP][%s] junk stdout: %s", symbol, s)
    #                 continue

    #             et = evt.get("event")
    #             if et == "init_ok":
    #                 logger.info("[STOP][%s] воркер инициализирован: %s", symbol, evt)
    #             elif et == "hb":
    #                 # не шумим
    #                 pass
    #             elif et == "activated":
    #                 logger.info("[STOP][%s] трейлинг АКТИВИРОВАН: ROI=%.2f%% price=%.8f",
    #                             symbol, float(evt.get("roi_pct") or 0.0), float(evt.get("price") or 0.0))
    #             elif et == "stop_update":
    #                 new_stop = float(evt.get("stop") or 0.0)
    #                 if new_stop > 0:
    #                     pos = self.open_positions.get(symbol) or {}
    #                     side = (pos.get("side") or side_cached or "").lower()  # "buy"/"sell"
    #                     try:
    #                         await self.set_or_amend_stop_loss(symbol, side, new_stop, reason="trail")
    #                     except Exception as e:
    #                         logger.exception("[STOP][%s] ошибка установки стопа %.8f: %s", symbol, new_stop, e)
    #             elif et in ("closed_by_parent", "closed", "init_error"):
    #                 logger.warning("[STOP][%s] завершение воркера: %s", symbol, et)
    #                 break
    #     finally:
    #         # уборка
    #         w = getattr(self, "_stop_workers", None)
    #         if isinstance(w, dict):
    #             w.pop(symbol, None)


    # async def _read_stop_stderr(self, symbol: str, stream):
    #     try:
    #         while True:
    #             line = await stream.readline()
    #             if not line:
    #                 break
    #             logger.error("[STOP][%s][worker] %s", symbol, line.decode("utf-8", "ignore").rstrip())
    #     finally:
    #         pass


    # async def _read_stop_stderr(self, symbol: str, proc):
    #     while True:
    #         try:
    #             line = await proc.stderr.readline()
    #             if not line: break
    #             logger.error(f"[stop_worker_stderr][{symbol}] {line.decode('utf-8', errors='ignore').strip()}")
    #         except Exception:
    #             break

    # async def _send_stop_msg(self, symbol: str, obj: dict) -> bool:
    #     # Этот метод нужен только для отправки команды 'close', поэтому он остается,
    #     # но нам нужно получить writer другим способом
    #     rec = self._stop_workers.get(symbol)
    #     if not rec or not (proc := rec.get("proc")) or proc.stdin.is_closing():
    #         return False
    #     try:
    #         w = proc.stdin
    #         w.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    #         await w.drain()
    #         return True
    #     except (BrokenPipeError, ConnectionResetError, RuntimeError):
    #         return False
    #     except Exception as e:
    #         logger.error(f"Неожиданная ошибка записи в stop_worker для {symbol}: {e}", exc_info=True)
    #         return False


    # async def _read_stop_stdout(self, symbol: str, proc):
    #     import json, asyncio
    #     try:
    #         while True:
    #             line = await proc.stdout.readline()
    #             if not line:
    #                 break
    #             try:
    #                 evt = json.loads(line.decode("utf-8", "ignore").strip())
    #             except Exception:
    #                 continue

    #             et = evt.get("event")
    #             if et == "init_ok":
    #                 logger.info("[STOP][%s] воркер инициализирован: %s", symbol, evt)
    #             elif et == "hb":
    #                 # можно логировать реже, чтобы не шуметь
    #                 pass
    #             elif et == "activated":
    #                 logger.info("[STOP][%s] трейлинг АКТИВИРОВАН: roi=%.2f%% price=%.8f",
    #                             symbol, float(evt.get("roi_pct", 0)), float(evt.get("price", 0)))
    #             elif et == "stop_update":
    #                 new_stop = float(evt.get("stop") or 0)
    #                 if new_stop > 0:
    #                     # достанем сторону позиции
    #                     pos = self.open_positions.get(symbol) or {}
    #                     side = (pos.get("side") or "").lower()  # "buy"/"sell"
    #                     try:
    #                         await self.set_or_amend_stop_loss(symbol, side, new_stop, reason="trail")
    #                     except Exception as e:
    #                         logger.exception("[STOP][%s] ошибка установки стопа %.8f: %s", symbol, new_stop, e)
    #             elif et in ("closed_by_parent", "closed"):
    #                 logger.info("[STOP][%s] воркер завершён (%s)", symbol, et)
    #                 break
    #     finally:
    #         # очистка
    #         if symbol in getattr(self, "stop_procs", {}):
    #             self.stop_procs.pop(symbol, None)


    # async def _read_stop_stderr(self, symbol: str, proc):
    #     # транслируем ошибки воркера в наш лог
    #     try:
    #         while True:
    #             line = await proc.stderr.readline()
    #             if not line:
    #                 break
    #             logger.error("[STOP][%s][worker] %s", symbol, line.decode("utf-8", "ignore").rstrip())
    #     finally:
    #         pass


    # async def _ensure_stop_worker(self, symbol: str, *, why: str = "unknown"):
    #     """
    #     Стартует воркер трейлинга только когда у позиции есть avg_price > 0.
    #     Если avg ещё не проставлен – подождёт до ~3 сек или возьмёт из pending_open_exec.
    #     """
    #     pos = self.open_positions.get(symbol)
    #     if not pos:
    #         return

    #     if symbol in getattr(self, "stop_procs", {}):
    #         return  # уже запущен

    #     # 1) получить реальную цену входа
    #     avg = float(pos.get("avg_price") or pos.get("entry_price") or 0.0)
    #     if avg <= 0:
    #         # попробуем быстро подождать и/или усыновить "раннюю" execution
    #         for _ in range(30):  # ~3 сек
    #             pend = getattr(self, "pending_open_exec", {}).get(symbol)
    #             if pend and pend.get("side") == pos.get("side"):
    #                 avg = float(pend.get("price") or 0.0)
    #                 if avg > 0:
    #                     pos["avg_price"] = avg
    #                     break
    #             await asyncio.sleep(0.1)
    #             avg = float(pos.get("avg_price") or pos.get("entry_price") or 0.0)

    #         if avg <= 0:
    #             logger.warning("[STOP][%s] %s: не стартуем воркер — avg_price=0", symbol, why)
    #             return

    #     # 2) собрать init_params (ROI-режим по умолчанию!)
    #     tick_size = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)
    #     start_roi_pct = float(self.user_state.get("trailing_start_pct", 5.0))
    #     gap_roi_pct   = float(self.user_state.get("trailing_gap_pct", 1.0))  # трактуем как ROI-отступ

    #     init_params = {
    #         "symbol": symbol,
    #         "side": pos.get("side"),                       # "Buy" / "Sell"
    #         "avg_price": avg,                              # ОБЯЗАТЕЛЬНО > 0
    #         "leverage": float(pos.get("leverage") or self.leverage),
    #         "tick_size": tick_size,
    #         "start_roi": start_roi_pct,
    #         "gap_mode": "roi",                             # критично
    #         "gap_roi_pct": gap_roi_pct,
    #         "hb_interval": 15.0,
    #     }

    #     # 3) запустить процесс-воркер
    #     import json, asyncio, sys, os
    #     worker_script_path = os.path.join(os.path.dirname(__file__), "stop_worker.py")
    #     args = [sys.executable, worker_script_path, json.dumps(init_params, ensure_ascii=False)]
    #     proc = await asyncio.create_subprocess_exec(
    #         *args,
    #         stdout=asyncio.subprocess.PIPE,
    #         stderr=asyncio.subprocess.PIPE,
    #     )
    #     self.stop_procs = getattr(self, "stop_procs", {})
    #     self.stop_procs[symbol] = proc
    #     logger.info("[STOP][%s] старт воркера (%s), avg=%.8f", symbol, why, avg)

    #     # 4) навесить читателей stdout/stderr
    #     asyncio.create_task(self._read_stop_stdout(symbol, proc))
    #     asyncio.create_task(self._read_stop_stderr(symbol, proc))


    async def set_or_amend_stop_loss(self, new_stop_price: float, *, symbol: str):
        """
        [ФИНАЛЬНАЯ ВЕРСИЯ] Просто устанавливает или изменяет стоп-лосс.
        Не содержит логики "улучшения", только исполнение.
        """
        pos = self.open_positions.get(symbol)
        if not pos:
            logger.warning(f"[{symbol}] Попытка установить стоп, но позиция не найдена.")
            return

        try:
            side = str(pos.get("side", "")).lower()
            tick = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)
            
            if side == "buy":
                stop_price = math.floor(new_stop_price / tick) * tick
            else:
                stop_price = math.ceil(new_stop_price / tick) * tick
            
            # --- ГЛАВНОЕ ИСПРАВЛЕНИЕ: ПРОВЕРКА ПРОТИВ СПАМА ---
            # Сравниваем новую округленную цену с последней, которую мы успешно сохранили.
            # Если они идентичны, нет нужды отправлять ордер снова.
            last_known_stop = self.last_stop_price.get(symbol)
            if last_known_stop is not None and abs(stop_price - last_known_stop) < 1e-9:
                logger.debug(f"[{symbol}] Новый стоп {stop_price} совпадает с текущим. Отправка пропущена.")
                return 
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---


            pos_idx = 1 if side == "buy" else 2
            
            logger.info(f"⚙️ [{symbol}] Отправка команды на установку/обновление стопа: {stop_price:.6f}")
            response = await asyncio.to_thread(
                lambda: self.session.set_trading_stop(
                    category="linear", symbol=symbol, positionIdx=pos_idx,
                    stopLoss=f"{stop_price:.8f}".rstrip("0").rstrip("."),
                )
            )
            
            if response.get("retCode") == 0:
                logger.info(f"✅ [{symbol}] API подтвердил установку/обновление стопа на {stop_price:.6f}")
                self.last_stop_price[symbol] = stop_price
                pos['initial_stop_set'] = True 
            elif response.get("retCode") == 34040 or "not modified" in response.get("retMsg", "").lower():
                logger.info(f"[{symbol}] Стоп уже установлен на этом или лучшем уровне (API: not modified).")
                self.last_stop_price[symbol] = stop_price # Все равно обновляем наше состояние
                pos['initial_stop_set'] = True
            else:
                logger.error(f"❌ [{symbol}] API отклонил установку стопа: {response.get('retMsg')} (Код: {response.get('retCode')})")

        except InvalidRequestError as e:
            if "34040" in str(e) or "not modified" in str(e).lower():
                logger.info(f"[{symbol}] Стоп уже установлен на этом уровне, API отклонил изменение (not modified).")
                pos['initial_stop_set'] = True # Считаем, что стоп установлен успешно
            elif getattr(e, "status_code", None) == 10001 or "position does not exist" in str(e):
                logger.warning(f"[{symbol}] Ошибка API (10001): Позиция не найдена. Очистка состояния.")
                if pos: pos['initial_stop_set'] = True 
                self._purge_symbol_state(symbol)
            else:
                logger.error(f"[{symbol}] Ошибка API (InvalidRequestError): {e}")
        except Exception as e:
            logger.error(f"[{symbol}] Критическая ошибка в set_or_amend_stop_loss: {e}", exc_info=True)


    # def _purge_symbol_state(self, symbol: str):
    #     """
    #     [ФИНАЛЬНАЯ ВЕРСИЯ] Атомарно и немедленно останавливает все активности,
    #     связанные с символом, предотвращая гонку состояний.
    #     """
    #     logger.debug(f"Полная очистка состояния для символа: {symbol}")

    #     # 1. Немедленно отменяем guardian'а, если он есть. Это остановит цикл в manage_open_position.
    #     if task := self.watch_tasks.pop(symbol, None):
    #         if not task.done():
    #             task.cancel()
    #             logger.debug(f"[{symbol}] Guardian task отменен.")

    #     # 2. Немедленно отправляем команду на остановку воркера.
    #     #    Мы не ждем его завершения здесь, чтобы не блокировать основной поток.
    #     #    Его ресурсы будут освобождены ОС.
    #     worker_rec = self._stop_workers.pop(symbol, None)
    #     if worker_rec:
    #         proc = worker_rec.get("proc")
    #         if proc and proc.returncode is None:
    #             try:
    #                 proc.kill()
    #                 logger.debug(f"[{symbol}] Процесс stop_worker (PID: {proc.pid}) принудительно завершен.")
    #             except ProcessLookupError:
    #                 pass # Процесс уже мог завершиться сам
    #         # Отменяем таски-читатели, чтобы они не висели в памяти
    #         for task_name in ("reader_task", "stderr_task"):
    #             if task := worker_rec.get(task_name):
    #                 task.cancel()
        
    #     # 3. Очищаем все локальные состояния ПОСЛЕ остановки активностей.
    #     self.open_positions.pop(symbol, None)
    #     self.last_stop_price.pop(symbol, None)
    #     self.pending_orders.pop(symbol, None)
    #     self.pending_cids.pop(symbol, None)
    #     self.pending_timestamps.pop(symbol, None)
    #     self.recently_closed[symbol] = time.time()
    #     self.trailing_activated.pop(symbol, None) 


    def _purge_symbol_state(self, symbol: str):
            """
            [ФИНАЛЬНАЯ ВЕРСИЯ] Атомарно и немедленно останавливает все активности,
            связанные с символом.
            """
            logger.debug(f"Полная очистка состояния для символа: {symbol}")

            # 1. Немедленно отменяем guardian'а, если он есть.
            if task := self.watch_tasks.pop(symbol, None):
                if not task.done():
                    task.cancel()
                    logger.debug(f"[{symbol}] Guardian task отменен.")

            # <-- ИСПРАВЛЕНИЕ: Блок, связанный с _stop_workers, полностью удален. -->
            
            # 2. Очищаем все локальные состояния.
            self.open_positions.pop(symbol, None)
            self.last_stop_price.pop(symbol, None)
            self.pending_orders.pop(symbol, None)
            self.pending_cids.pop(symbol, None)
            self.pending_timestamps.pop(symbol, None)
            self.recently_closed[symbol] = time.time()
            self.trailing_activated.pop(symbol, None)


    async def _cleanup_recently_closed(self, interval: int = 15, max_age: int = 60):
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            expired = [s for s, ts in self.recently_closed.items() if now - ts > max_age]
            for s in expired:
                self.recently_closed.pop(s, None)

    async def on_ticker_update(self, symbol: str, last_price: float):
        pos = self.open_positions.get(symbol)
        if pos:
            pos["markPrice"] = last_price
            start_roi_pct, gap_roi_pct = self._get_trailing_params()
            
            avg_price = self._resolve_avg_price(symbol, pos)
            if avg_price > 0:
                side = pos.get("side", "Buy")
                leverage = utils.safe_to_float(pos.get("leverage", 10.0))
                
                pnl = ((last_price / avg_price) - 1.0) if side == "Buy" else ((avg_price / last_price) - 1.0)
                current_roi = pnl * 100.0 * leverage

                if not self.trailing_activated.get(symbol) and current_roi >= start_roi_pct:
                    self.trailing_activated[symbol] = True
                    logger.info(f"✅ [{symbol}] ТРЕЙЛИНГ АКТИВИРОВАН! ROI: {current_roi:.2f}%")

                if self.trailing_activated.get(symbol):
                    target_roi = current_roi - gap_roi_pct
                    denom = 1.0 + (target_roi / (100.0 * leverage))
                    
                    new_stop_price = (avg_price * denom) if side == "Buy" else (avg_price / denom if denom > 1e-9 else 0.0)
                    
                    if new_stop_price > 0:
                        # --- ПРОВЕРКА НА УЛУЧШЕНИЕ ТЕПЕРЬ ЗДЕСЬ ---
                        prev_stop = self.last_stop_price.get(symbol)
                        is_better = prev_stop is None or \
                                    (side == "Buy" and new_stop_price > prev_stop) or \
                                    (side == "Sell" and new_stop_price < prev_stop)

                        if is_better:
                            await self.set_or_amend_stop_loss(new_stop_price, symbol=symbol)
                        # --- КОНЕЦ ПРОВЕРКИ ---

        await strategies.high_frequency_dispatcher(self, symbol)


    async def get_total_open_volume(self) -> float:
        total = 0.0
        for pos in self.open_positions.values():
            size = utils.safe_to_float(pos.get("volume", 0))
            price = utils.safe_to_float(pos.get("markPrice", 0)) or utils.safe_to_float(pos.get("avg_price", 0))
            total += size * price
        return total

    async def get_effective_total_volume(self) -> float:
        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        open_vol = await self.get_total_open_volume()
        # -------------------------
        pending_vol = sum(self.pending_orders.values())
        return open_vol + pending_vol


    @async_retry(max_retries=5, delay=3) # <--- ДОБАВЬТЕ ДЕКОРАТОР
    async def _cache_all_symbol_meta(self):
        logger.info("Кэширование метаданных для всех символов...")
        try:
            resp = await asyncio.to_thread(
                lambda: self.session.get_instruments_info(category="linear")
            )
            instrument_list = resp.get("result", {}).get("list", [])
            for info in instrument_list:
                if symbol := info.get("symbol"):
                    self.qty_step_map[symbol] = utils.safe_to_float(info["lotSizeFilter"]["qtyStep"])
                    self.min_qty_map[symbol] = utils.safe_to_float(info["lotSizeFilter"]["minOrderQty"])
                    self.price_tick_map[symbol] = utils.safe_to_float(info["priceFilter"]["tickSize"])
            logger.info(f"Успешно закэшировано метаданных для {len(self.qty_step_map)} символов.")
        except Exception:
            logger.error("Критическая ошибка: не удалось закэшировать метаданные символов.", exc_info=True)


    # ======================================================================
    # 8. Сбор данных и Feature Engineering
    # ======================================================================

    def load_ml_models(self):
        """Загружает ML-модель и скейлер при старте бота."""
        self.ml_inferencer = ai_ml.MLXInferencer()

    async def extract_realtime_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Собирает и рассчитывает все необходимые признаки (фичи) для ML-модели и AI."""
        last_price = 0.0
        bid1 = 0.0
        ask1 = 0.0

        tdata = self.shared_ws.ticker_data.get(symbol) or {}
        last_price = utils.safe_to_float(tdata.get("lastPrice", 0.0))
        bid1 = utils.safe_to_float(tdata.get("bid1Price", 0.0))
        ask1 = utils.safe_to_float(tdata.get("ask1Price", 0.0))

        if last_price <= 0.0:
            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if candles:
                last_price = utils.safe_to_float(candles[-1].get("closePrice", 0.0))
                bid1 = last_price
                ask1 = last_price

        if last_price <= 0.0:
            logger.warning(f"[features] {symbol}: нет актуальной цены, прерываем")
            return {}

        spread_pct = (ask1 - bid1) / bid1 * 100.0 if bid1 > 0 else 0.0

        candles = list(self.shared_ws.candles_data.get(symbol, []))
        oi_hist = list(self.shared_ws.oi_history.get(symbol, []))
        cvd_hist = list(self.shared_ws.cvd_history.get(symbol, []))

        pct1m  = utils.compute_pct(self.shared_ws.candles_data.get(symbol, []), 1)
        pct5m  = utils.compute_pct(self.shared_ws.candles_data.get(symbol, []), 5)
        pct15m = utils.compute_pct(self.shared_ws.candles_data.get(symbol, []), 15)
        pct30m = utils.compute_pct(self.shared_ws.candles_data.get(symbol, []), 30)


        V1m  = utils.sum_last_vol(self.shared_ws.candles_data.get(symbol, []), 1)
        V5m  = utils.sum_last_vol(self.shared_ws.candles_data.get(symbol, []), 5)
        V15m = utils.sum_last_vol(self.shared_ws.candles_data.get(symbol, []), 15)

        OI_now    = utils.safe_to_float(oi_hist[-1]) if oi_hist else 0.0
        OI_prev1m = utils.safe_to_float(oi_hist[-2]) if len(oi_hist) >= 2 else 0.0
        OI_prev5m = utils.safe_to_float(oi_hist[-6]) if len(oi_hist) >= 6 else 0.0
        dOI1m = (OI_now - OI_prev1m) / OI_prev1m if OI_prev1m > 0 else 0.0
        dOI5m = (OI_now - OI_prev5m) / OI_prev5m if OI_prev5m > 0 else 0.0

        CVD_now    = utils.safe_to_float(cvd_hist[-1]) if cvd_hist else 0.0
        CVD_prev1m = utils.safe_to_float(cvd_hist[-2]) if len(cvd_hist) >= 2 else 0.0
        CVD_prev5m = utils.safe_to_float(cvd_hist[-6]) if len(cvd_hist) >= 6 else 0.0
        CVD1m = CVD_now - CVD_prev1m
        CVD5m = CVD_now - CVD_prev5m

        sigma5m = self.shared_ws._sigma_5m(symbol)

        df = pd.DataFrame(candles[-100:])
        if not df.empty:
            for col in ("closePrice", "highPrice", "lowPrice", "volume"):
                if col not in df.columns: df[col] = np.nan
                s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                df[col] = s.ffill().bfill()
        
        n = len(df)
        close = df["closePrice"] if n else pd.Series(dtype="float64")
        high  = df["highPrice"]  if n else pd.Series(dtype="float64")
        low   = df["lowPrice"]   if n else pd.Series(dtype="float64")

        def _safe_last(series, default=0.0):
            try:
                v = series.iloc[-1]
                return float(v) if pd.notna(v) else default
            except (IndexError, TypeError):
                return default

        rsi14 = _safe_last(ta.rsi(close, length=14), 50.0) if n >= 15 else 50.0
        sma50 = _safe_last(ta.sma(close, length=50), _safe_last(close, 0.0)) if n >= 50 else _safe_last(close, 0.0)
        ema20 = _safe_last(ta.ema(close, length=20), sma50) if n >= 20 else sma50
        atr14 = _safe_last(ta.atr(high, low, close, length=14), 0.0) if n >= 15 else 0.0
        
        bb_width = 0.0
        if n >= 20:
            bb = ta.bbands(close, length=20)
            if bb is not None and not bb.empty:
                bb_width = _safe_last(bb.iloc[:, 0], 0.0) - _safe_last(bb.iloc[:, 2], 0.0)

        st_ser = compute_supertrend(df, period=10, multiplier=3) if n > 20 else None
        supertrend_num = 1 if _safe_last(st_ser, False) else -1

        adx14 = _safe_last(ta.adx(high, low, close, length=14)["ADX_14"], 0.0) if n >= 15 else 0.0
        cci20 = _safe_last(ta.cci(high, low, close, length=20), 0.0) if n >= 20 else 0.0

        macd_val, macd_signal = 0.0, 0.0
        if n >= 35:
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty and macd_df.shape[1] >= 3:
                macd_val = _safe_last(macd_df.iloc[:, 0], 0.0)
                macd_signal = _safe_last(macd_df.iloc[:, 2], 0.0)

        GS_pct4m = utils.compute_pct(self.shared_ws.candles_data.get(symbol, []), 4)
        GS_vol4m = utils.sum_last_vol(self.shared_ws.candles_data.get(symbol, []), 4)
        base_OI = utils.safe_to_float(oi_hist[-5]) if len(oi_hist) >= 5 else (OI_now or 1.0)
        GS_dOI4m = (OI_now - base_OI) / base_OI if base_OI > 0 else 0.0
        base_CVD = utils.safe_to_float(cvd_hist[-5]) if len(cvd_hist) >= 5 else CVD_now
        GS_cvd4m = CVD_now - base_CVD
        GS_supertrend_flag = supertrend_num
        GS_cooldown_flag = int(not self._golden_allowed(symbol))

        mean_V5m = (V5m / 5.0) if V5m > 0 else 1e-8
        SQ_power = abs(pct5m) * abs((V1m - mean_V5m) / max(1e-8, mean_V5m) * 100.0)
        SQ_strength = int(abs(pct5m) >= config.SQUEEZE_THRESHOLD_PCT and SQ_power >= config.DEFAULT_SQUEEZE_POWER_MIN)
        
        recent_liq_vals = [utils.safe_to_float(evt['value']) for evt in self.liq_buffers.get(symbol, []) if time.time() - evt['ts'] <= 10]
        SQ_liq10s = float(np.nansum(recent_liq_vals)) if recent_liq_vals else 0.0
        SQ_cooldown_flag = int(not self._squeeze_allowed(symbol))

        buf = self.liq_buffers.get(symbol, [])
        recent_all = [evt for evt in buf if time.time() - evt['ts'] <= 10]
        same_side = [utils.safe_to_float(evt['value']) for evt in recent_all if recent_all and evt['side'] == recent_all[-1]['side']]
        LIQ_cluster_val10s = float(np.nansum(same_side)) if same_side else 0.0
        LIQ_cluster_count10s = int(len(same_side))
        LIQ_direction = 1 if (recent_all and recent_all[-1]['side'] == "Buy") else -1
        LIQ_cooldown_flag = int(not self.shared_ws.check_liq_cooldown(symbol))

        now_ts = dt.datetime.now()
        hour_of_day = int(now_ts.hour)
        day_of_week = int(now_ts.weekday())
        month_of_year = int(now_ts.month)

        avgVol30m = self.shared_ws.get_avg_volume(symbol, 30)
        tail_oi = [utils.safe_to_float(x) for x in oi_hist[-30:]] if oi_hist else []
        avgOI30m = float(np.nanmean(tail_oi)) if tail_oi else 0.0
        deltaCVD30m = CVD_now - (utils.safe_to_float(cvd_hist[-31]) if len(cvd_hist) >= 31 else 0.0)

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
            "LIQ_direction": LIQ_direction, "LIQ_pct1m": pct1m, "LIQ_pct5m": pct5m, "pct30m": pct30m, "LIQ_vol1m": V1m,
            "LIQ_vol5m": V5m, "LIQ_dOI1m": dOI1m, "LIQ_spread_pct": spread_pct, "LIQ_sigma5m": sigma5m,
            "LIQ_golden_flag": GS_cooldown_flag, "LIQ_squeeze_flag": SQ_cooldown_flag, "LIQ_cooldown": LIQ_cooldown_flag,
            "hour_of_day": hour_of_day, "day_of_week": day_of_week, "month_of_year": month_of_year,
        }

        for k in config.FEATURE_KEYS:
            features.setdefault(k, 0.0)

        return features

    async def _get_golden_thresholds(self, symbol: str, side: str) -> dict:
        """Получает пороговые значения для стратегии Golden Setup, возможно, корректируя их с помощью ML."""
        base = (
            self.golden_param_store.get((symbol, side))
            or self.golden_param_store.get(side)
            or {"period_iters": 3, "price_change": 1.7,
                "volume_change": 200, "oi_change": 1.5}
        )
        return base # ML-тюнинг пока отключен для упрощения

    def _aggregate_candles_5m(self, candles: any) -> list:
        """Агрегирует минутные свечи в пятиминутные."""
        # --- ИСПРАВЛЕНИЕ: Явно преобразуем в list ---
        candle_list = list(candles)
        if not candle_list:
            return []

            
        result = []
        full_blocks = len(candle_list) // 5
        for i in range(full_blocks):
            chunk = candle_list[i * 5:(i + 1) * 5]
            result.append({
                "openPrice": utils.safe_to_float(chunk[0]["openPrice"]),
                "highPrice": max(utils.safe_to_float(c["highPrice"]) for c in chunk),
                "lowPrice": min(utils.safe_to_float(c["lowPrice"]) for c in chunk),
                "closePrice": utils.safe_to_float(chunk[-1]["closePrice"]),
                "volume": sum(utils.safe_to_float(c["volume"]) for c in chunk),
            })
        return result


    def _aggregate_series_5m(self, source, lookback: int = 6, method: str = "sum") -> list:
        """Универсальный агрегатор временных рядов в 5-минутный интервал."""
        if not source: return []
        full_blocks = len(source) // 5
        result = []
        for i in range(full_blocks):
            chunk = source[i * 5:(i + 1) * 5]
            if method == "sum":
                result.append(sum(utils.safe_to_float(x) for x in chunk))
            else:
                result.append(utils.safe_to_float(chunk[-1]))
        return result[-lookback:]


    def _aggregate_ohlcv_5m(self, minute_candles: list, lookback: int = 15):
        """
        Собираем 5-минутные бары из минуток.
        Возвращает список баров-словарей: [{'open', 'high', 'low', 'close', 'volume'}, ...].
        """
        try:
            if not minute_candles:
                return []

            # Берем кратно 5 (последние N минуток)
            m1_needed = lookback * 5
            tail = minute_candles[-m1_needed:] if len(minute_candles) >= m1_needed else minute_candles[:]

            bars_5m = []
            for i in range(0, len(tail), 5):
                chunk = tail[i:i+5]
                if len(chunk) < 5:
                    break # Обрабатываем только полные 5-минутные блоки
                
                o = utils.safe_to_float(chunk[0].get("openPrice", 0))
                h = max(utils.safe_to_float(x.get("highPrice", 0)) for x in chunk)
                l = min(utils.safe_to_float(x.get("lowPrice", 0)) for x in chunk)
                c = utils.safe_to_float(chunk[-1].get("closePrice", 0))
                v = sum(utils.safe_to_float(x.get("volume", 0)) for x in chunk)
                
                bars_5m.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
                
            return bars_5m
        except Exception as e:
            logger.error(f"Ошибка при агрегации 5m OHLCV: {e}", exc_info=True)
            return []

    # async def _send_to_stop_worker(self, symbol: str, command: dict):
    #     """Безопасно отправляет JSON-команду в stdin воркера."""
    #     worker_rec = self._stop_workers.get(symbol)
    #     if not worker_rec:
    #         return

    #     proc = worker_rec.get("proc")
    #     if proc and proc.returncode is None:
    #         try:
    #             stdin_writer = proc.stdin
    #             if not stdin_writer.is_closing():
    #                 stdin_writer.write((json.dumps(command) + '\n').encode('utf-8'))
    #                 await stdin_writer.drain()
    #         except (BrokenPipeError, ConnectionResetError):
    #             logger.warning(f"[{symbol}] Канал связи с stop_worker разорван (BrokenPipe).")
    #         except Exception as e:
    #             logger.error(f"[{symbol}] Ошибка при отправке команды в stop_worker: {e}")


    def _build_squeeze_features_5m(self, symbol: str):
        """
        [V3 - CORRECTED] Возвращает (features: dict, impulse_dir: 'up'|'down') или (None, None).
        Корректно работает с именами столбцов и рассчитывает ATR.
        """
        try:
            bars = self._aggregate_ohlcv_5m(list(self.shared_ws.candles_data.get(symbol, []))[-75:])
            if len(bars) < 15:
                return None, None

            df = pd.DataFrame(bars)
            
            # --- ИСПРАВЛЕНИЕ: Используем правильные имена столбцов из _aggregate_ohlcv_5m ---
            df['highPrice'] = pd.to_numeric(df['high'])
            df['lowPrice'] = pd.to_numeric(df['low'])
            df['closePrice'] = pd.to_numeric(df['close'])

            atr_series = ta.atr(df['highPrice'], df['lowPrice'], df['closePrice'], length=14)
            atr_5m = atr_series.iloc[-1] if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else 0.0

            prev, last = bars[-2], bars[-1]
            pc = utils.safe_to_float(prev.get("close", 0.0))
            lc = utils.safe_to_float(last.get("close", 0.0))
            if pc <= 0 or lc <= 0:
                return None, None

            ret_5m = (lc - pc) / pc
            impulse_dir = "up" if ret_5m > 0 else "down"

            features = {
                "price": lc, "ret_5m": float(ret_5m), "atr_5m": float(atr_5m),
                "impulse_high": max(float(last.get("high", lc)), float(prev.get("high", pc))),
                "impulse_low":  min(float(last.get("low", lc)), float(prev.get("low", pc))),
            }
            return features, impulse_dir
        except Exception:
            logger.exception(f"[SQUEEZE_BUILD] {symbol}: features build failed")
            return None, None

    def _golden_allowed(self, symbol: str) -> bool:
        """
        Проверяет, истек ли кулдаун для стратегии Golden Setup на данном символе.
        """
        # Устанавливаем кулдаун в 5 минут (300 секунд)
        cooldown_period_sec = 300
        last_signal_time = self._last_golden_ts.get(symbol, 0)
        return (time.time() - last_signal_time) > cooldown_period_sec

    def _squeeze_allowed(self, symbol: str) -> bool:
        """
        Проверяет, истек ли кулдаун для стратегии Squeeze на данном символе.
        """
        # Устанавливаем кулдаун в 10 минут (600 секунд) для сквизов, т.к. они могут быть затяжными
        cooldown_period_sec = 600
        last_signal_time = self.last_squeeze_ts.get(symbol, 0)
        return (time.time() - last_signal_time) > cooldown_period_sec


    # --- Утилиты и вспомогательные методы ---
    
    async def notify_user(self, text: str):
        """Отправляет сообщение пользователю в Telegram."""
        if not telegram_bot: return
        try:
            await telegram_bot.send_message(self.user_id, text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.warning(f"Не удалось отправить сообщение пользователю {self.user_id}: {e}")

    async def log_trade(self, **kwargs):
        """Централизованно логирует все торговые события (открытие, закрытие) в CSV и Telegram."""
        symbol = kwargs.get("symbol")
        side = str(kwargs.get("side", "")).capitalize()
        action = str(kwargs.get("action", "")).lower()
        result = str(kwargs.get("result", "")).lower()
        avg_price = utils.safe_to_float(kwargs.get("avg_price"))
        volume = utils.safe_to_float(kwargs.get("volume"))
        pnl_usdt = kwargs.get("pnl_usdt")
        pnl_pct = kwargs.get("pnl_pct")
        comment = kwargs.get("comment")
        time_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Логирование в файл
        pnl_info = f" | PnL: {pnl_usdt:.2f}$ ({pnl_pct:.2f}%)" if pnl_usdt is not None else ""
        logger.info(f"[LOG_TRADE] user={self.user_id} {action.upper()} {symbol}: side={side}, vol={volume}, price={avg_price}, result={result}{pnl_info}")

        # Запись в CSV для анализа
        try:
            base_row = {
                "timestamp": dt.datetime.utcnow().isoformat(), "symbol": symbol, "side": side,
                "event": action, "result": result, "volume_trade": volume, "price_trade": avg_price,
                "pnl_usdt": pnl_usdt, "pnl_pct": pnl_pct
            }
            # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ---
            extended_metrics = await self._dataset_metrics(symbol)
            base_row.update(extended_metrics)
            
            utils._append_trades_unified(base_row)
        except Exception as e:
            logger.warning(f"Ошибка записи в trades_unified.csv: {e}")

        # Отправка уведомления в Telegram
        link = f"https://www.bybit.com/trade/usdt/{symbol}"
        msg = ""
        if action == "open":
            icon = "🟩" if side == "Buy" else "🟥"
            msg = (f"{icon} <b>Открыта {side.upper()} {symbol}</b>\n\n"
                f"<b>Цена входа:</b> {avg_price:.6f}\n"
                f"<b>Объем:</b> {volume}\n")
            if comment: msg += f"\n<i>AI: {comment}</i>"
        elif action == "close" and pnl_usdt is not None:
            pnl_icon = "💰" if pnl_usdt >= 0 else "🔻"
            pnl_sign = "+" if pnl_usdt >= 0 else ""
            msg = (f"{pnl_icon} <b>Закрытие {symbol}</b>\n\n"
                f"<b>Результат:</b> <code>{pnl_sign}{pnl_usdt:.2f} USDT ({pnl_sign}{pnl_pct:.3f}%)</code>\n"
                f"<b>Цена выхода:</b> {avg_price:.6f}\n")

        if msg:
            msg += f"\n<a href='{link}'>График</a> | {time_str} | #{symbol}"
            await self.notify_user(msg)
            

    async def _calc_qty_from_usd(self, symbol: str, usd_amount: float, price: float | None = None) -> float:
            """
            [ФИНАЛЬНАЯ ВЕРСИЯ] Надежно рассчитывает количество актива,
            используя Decimal для финансовой точности и строго соблюдая правила биржи.
            """
            await self.ensure_symbol_meta(symbol)
            
            step_str = str(self.qty_step_map.get(symbol, "0.001"))
            min_qty_str = str(self.min_qty_map.get(symbol, step_str))
            
            p = price or utils.safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
            if not p > 0:
                logger.error(f"[{symbol}] Не удалось получить актуальную цену для расчета количества.")
                return 0.0

            try:
                # --- ШАГ 1: Преобразуем все в Decimal для точных вычислений ---
                d_usd_amount = Decimal(str(usd_amount))
                d_price = Decimal(str(p))
                d_step = Decimal(step_str)
                d_min_qty = Decimal(min_qty_str)

                if d_price == 0: return 0.0

                # --- ШАГ 2: Рассчитываем "сырое" количество ---
                raw_qty = d_usd_amount / d_price

                # --- ШАГ 3: Округляем количество ВНИЗ до ближайшего шага (qtyStep) ---
                # Это самая важная часть, которая предотвращает ошибки "Qty invalid".
                ticks = (raw_qty / d_step).quantize(Decimal('1'), rounding='ROUND_DOWN')
                final_qty = ticks * d_step

                # --- ШАГ 4: Проверяем на минимальный объем ---
                if final_qty < d_min_qty:
                    logger.warning(f"[{symbol}] Расчетный объем {final_qty} меньше минимального {d_min_qty}. Для ордера будет использован минимальный объем.")
                    final_qty = d_min_qty
                
                return float(final_qty)

            except Exception as e:
                logger.error(f"[{symbol}] Критическая ошибка при расчете количества: {e}", exc_info=True)
                return 0.0



    @async_retry(max_retries=5, delay=3) # <--- ДОБАВЬТЕ ДЕКОРАТОР
    async def ensure_symbol_meta(self, symbol: str):
        """Получает и кэширует метаданные инструмента (шаг цены, лота) с биржи."""
        if symbol in self.qty_step_map: return
        try:
            resp = await asyncio.to_thread(
                lambda: self.session.get_instruments_info(category="linear", symbol=symbol)
            )
            info = resp["result"]["list"][0]
            self.qty_step_map[symbol] = utils.safe_to_float(info["lotSizeFilter"]["qtyStep"])
            self.min_qty_map[symbol] = utils.safe_to_float(info["lotSizeFilter"]["minOrderQty"])
            self.price_tick_map[symbol] = utils.safe_to_float(info["priceFilter"]["tickSize"])
        except Exception as e:
            logger.warning(f"Не удалось получить метаданные для {symbol}: {e}")
            self.qty_step_map.setdefault(symbol, 0.001)
            self.min_qty_map.setdefault(symbol, 0.001)
            self.price_tick_map.setdefault(symbol, 0.0001)
        
    # async def _cache_all_symbol_meta(self):
    #     """Кэширует метаданные для всех доступных символов при старте."""
    #     logger.info("Кэширование метаданных для всех символов...")
    #     try:
    #         resp = await asyncio.to_thread(
    #             lambda: self.session.get_instruments_info(category="linear")
    #         )
    #         instrument_list = resp.get("result", {}).get("list", [])
    #         for info in instrument_list:
    #             if symbol := info.get("symbol"):
    #                 self.qty_step_map[symbol] = utils.safe_to_float(info["lotSizeFilter"]["qtyStep"])
    #                 self.min_qty_map[symbol] = utils.safe_to_float(info["lotSizeFilter"]["minOrderQty"])
    #                 self.price_tick_map[symbol] = utils.safe_to_float(info["priceFilter"]["tickSize"])
    #         logger.info(f"Успешно закэшировано метаданных для {len(self.qty_step_map)} символов.")
    #     except Exception:
    #         logger.error("Критическая ошибка: не удалось закэшировать метаданные символов.", exc_info=True)

    async def listing_age_minutes(self, symbol: str) -> float:
        """
        [ФИНАЛЬНАЯ ВЕРСИЯ С КЭШИРОВАНИЕМ] Определяет "возраст" торговой пары.
        Делает сетевой запрос только один раз для каждого символа.
        """
        now = time.time()
        
        # Шаг 1: Проверяем, есть ли уже результат в кэше
        # _listing_age_cache - это глобальный словарь в начале файла
        cached_data = _listing_age_cache.get(symbol)
        
        # Если данные есть и они не старше часа, возвращаем их немедленно
        if cached_data and (now - cached_data[1] < 3600):
            return cached_data[0]

        # Шаг 2: Если в кэше нет, делаем ОДИН сетевой запрос под семафором
        # (семафор ограничивает количество одновременных запросов, чтобы не забанили)
        async with _listing_sem:
            try:
                # Делаем медленный сетевой вызов
                resp = await asyncio.to_thread(
                    lambda: self.session.get_instruments_info(category="linear", symbol=symbol)
                )
                info = resp["result"]["list"][0]
                launch_ms = utils.safe_to_float(info.get("launchTime", 0))
                
                if launch_ms <= 0:
                    raise ValueError("launchTime missing or invalid")
                
                # Рассчитываем возраст в минутах
                age_min = (now * 1000 - launch_ms) / 60000.0

            except Exception as e:
                # В случае любой ошибки считаем монету "очень старой", чтобы не блокировать ее зря
                logger.warning(f"Не удалось определить возраст для {symbol}: {e}. Считаем ее 'старой'.")
                age_min = 999_999.0
            
            # Шаг 3: Сохраняем результат в кэш
            _listing_age_cache[symbol] = (age_min, now)
            return age_min

        
    def _apply_funding_to_features(self, symbol: str, features: dict) -> dict:
        """Обогащает словарь признаков данными о ставке финансирования."""
        snap = self._funding_snapshot(symbol, features)
        features.update(snap)
        return snap

    def _apply_funding_to_candidate(self, candidate: dict, funding_snap: dict) -> None:
        """Добавляет информацию о финансировании в кандидата на сделку."""
        fm = {
            "funding_rate": funding_snap.get("funding_rate"),
            "funding_bucket": funding_snap.get("funding_bucket"),
        }
        if "base_metrics" in candidate:
            candidate["base_metrics"].update(fm)
        else:
            candidate["base_metrics"] = fm

    def _funding_snapshot(self, symbol: str, features: dict | None = None) -> dict:
        """Создает сводку по текущей ставке финансирования."""
        rate = None
        if features:
            rate = features.get("fundingRate")
        if rate is None:
            hist = self.shared_ws.funding_history.get(symbol)
            if hist: rate = hist[-1]
        
        rate = utils.safe_to_float(rate)
        abs_r = abs(rate)
        
        if abs_r >= 0.005: bucket = "hot"
        elif abs_r >= 0.001: bucket = "warm"
        else: bucket = "cool"
        
        return {"funding_rate": rate, "funding_bucket": bucket}

    async def _dataset_metrics(self, symbol: str) -> dict:
        """
        [ИСПРАВЛЕННАЯ ВЕРСИЯ] Собирает расширенные метрики для записи в CSV,
        включая RSI, ADX и аномалию объема.
        """
        # Вызываем основную функцию для получения всех возможных признаков
        features = await self.extract_realtime_features(symbol)
        if not features:
            # Если фичи по какой-то причине не собрались, возвращаем пустой словарь
            return {}

        # --- Расчет аномалии объема (как в ai_ml.py) ---
        vol_1m = features.get('vol1m', 0)
        avg_vol_30m = features.get('avgVol30m', 1)
        vol_anomaly = vol_1m / avg_vol_30m if avg_vol_30m > 0 else 1.0

        # Собираем финальный словарь для записи в CSV
        metrics = {
            "price": features.get("price", 0.0),
            "open_interest": features.get("OI_now", 0.0),
            "volume_1m": vol_1m,
            "rsi14": features.get("rsi14", 0.0),
            "adx14": features.get("adx14", 0.0),
            "volume_anomaly": vol_anomaly
        }
        return metrics
