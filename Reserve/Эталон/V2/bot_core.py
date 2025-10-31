# bot_core.py
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
from telegram_bot import bot as telegram_bot
from signal_worker import start_worker_process
from data_manager import compute_supertrend
from utils import async_retry
from decimal import Decimal
from websocket_monitor import get_monitor

logger = logging.getLogger(__name__)

_listing_age_cache: dict[str, tuple[float, float]] = {}
_listing_sem = asyncio.Semaphore(5)

class TradingBot:
    def __init__(self, user_data: Dict[str, Any], shared_ws, golden_param_store: Dict):
        self.user_id = user_data["user_id"]
        self.api_key = user_data["api_key"]
        self.api_secret = user_data["api_secret"]
        self.user_data = user_data
        self.mode = user_data.get("mode", "real")
        self.shared_ws = shared_ws
        self.shared_ws.position_handlers.append(self)
        self.loop = asyncio.get_running_loop()
        self.monitor = get_monitor()
        self.private_ws_name = f"private_websocket_user_{self.user_id}"
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
        self.pending_open_exec: Dict[str, Dict[str, Any]] = {}
        self.momentum_cooldown_until = defaultdict(float)
        self.last_stop_price: Dict[str, float] = {}
        self.watch_tasks: Dict[str, asyncio.Task] = {}
        self.user_state = getattr(self, "user_state", {})
        self.trailing_activated: Dict[str, bool] = {}
        self.trailing_activation_ts: Dict[str, float] = {} 
        self.take_profit_price: Dict[str, float] = {}
        self.position_mode = 0
        self.flea_positions_count = 0
        self.flea_cooldown_until: Dict[str, float] = {}
        self.POSITION_VOLUME = utils.safe_to_float(user_data.get("volume", 1000))
        self.MAX_TOTAL_VOLUME = utils.safe_to_float(user_data.get("max_total_volume", 5000))
        self.leverage = utils.safe_to_float(user_data.get("leverage", 10.0))
        self.listing_age_min = int(user_data.get("listing_age_min_minutes", config.LISTING_AGE_MIN_MINUTES))
        self.qty_step_map: Dict[str, float] = {}
        self.min_qty_map: Dict[str, float] = {}
        self.price_tick_map: Dict[str, float] = {}
        self.ml_inferencer: Optional[ai_ml.MLXInferencer] = None
        self.training_data = deque(maxlen=5000)
        self.ai_circuit_open_until = 0.0
        self._ai_inflight_signals = set()
        self.apply_user_settings()
        self.momentum_cooldown_until = defaultdict(float)
        self.session = HTTP(
            testnet=False, demo=(self.mode == "demo"),
            api_key=self.api_key, api_secret=self.api_secret, timeout=30
        )
        try:
            adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
            self.session.client.mount("https://", adapter)
        except Exception: pass
        self.position_lock = asyncio.Lock()
        self.pending_orders_lock = asyncio.Lock()
        self.liq_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.golden_param_store = golden_param_store
        self._last_golden_ts = defaultdict(float)
        self.last_squeeze_ts = defaultdict(float)
        self.current_total_volume = 0.0
        self.time_offset = 0
        self.trade_counters = defaultdict(int)
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        # Ограничиваем количество одновременных тяжелых вычислений. 
        # 4 - хорошее значение для 4-8 ядерного процессора.
        self.feature_extraction_sem = asyncio.Semaphore(4)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        self.last_hf_check_ts = defaultdict(float) # Для троттлинга высокочастотных стратегий


    def apply_user_settings(self):
        cfg = self.user_data
        self.strategy_mode = cfg.get("strategy_mode", "full")
        self.ai_primary_model = cfg.get("ai_primary_model", config.AI_PRIMARY_MODEL)
        self.ai_advisor_model = cfg.get("ai_advisor_model", config.AI_ADVISOR_MODEL)
        self.ollama_primary_openai = cfg.get("ollama_primary_openai", config.OLLAMA_PRIMARY_OPENAI)
        self.ollama_advisor_openai = cfg.get("ollama_advisor_openai", config.OLLAMA_ADVISOR_OPENAI)
        self.ai_timeout_sec = float(cfg.get("ai_timeout_sec", 15.0))
        self.entry_cooldown_sec = int(cfg.get("entry_cooldown_sec", 30))
        self.tactical_entry_window_sec = int(cfg.get("tactical_entry_window_sec", 300))
        self.squeeze_ai_confirm_interval_sec = float(cfg.get("squeeze_ai_confirm_interval_sec", 2.0))
        new_pos_vol = utils.safe_to_float(cfg.get("volume", self.POSITION_VOLUME))
        if new_pos_vol != self.POSITION_VOLUME:
            self.POSITION_VOLUME = new_pos_vol
            logger.info(f"Объем позиции обновлен на: {self.POSITION_VOLUME}")
        new_max_vol = utils.safe_to_float(cfg.get("max_total_volume", self.MAX_TOTAL_VOLUME))
        if new_max_vol != self.MAX_TOTAL_VOLUME:
            self.MAX_TOTAL_VOLUME = new_max_vol
            logger.info(f"Максимальный общий объем обновлен на: {self.MAX_TOTAL_VOLUME}")
        logger.info(f"Настройки для пользователя {self.user_id} применены. Режим: {self.strategy_mode}")

    async def _sync_server_time(self):
        try:
            logger.info("Синхронизация времени с сервером Bybit...")
            response = await asyncio.to_thread(self.session.get_server_time)
            time_nano_str = response.get("result", {}).get("timeNano", "0")
            server_time_ms = int(time_nano_str) // 1_000_000
            if server_time_ms == 0:
                logger.error("Не удалось получить время сервера из ответа API.")
                return
            server_time_s = server_time_ms / 1000.0
            local_time_s = time.time()
            self.time_offset = server_time_s - local_time_s
            self.session.time_offset = self.time_offset * 1000
            logger.info(f"Синхронизация времени завершена. Смещение: {self.time_offset:.3f} секунд.")
        except Exception as e:
            logger.error(f"Не удалось синхронизировать время с сервером: {e}", exc_info=True)

    async def start(self):        
        await self._sync_server_time()
        self._load_trade_counters_from_history()
        logger.info(f"[User {self.user_id}] Бот запущен")
        
        async def time_sync_loop():
            while True:
                await asyncio.sleep(3600)
                await self._sync_server_time()
        
        asyncio.create_task(time_sync_loop())
        await self._sync_position_mode()
        asyncio.create_task(self.sync_open_positions_loop())
        asyncio.create_task(self.wallet_loop())
        asyncio.create_task(self._cleanup_recently_closed())
        asyncio.create_task(self.reload_settings_loop())
        await self.update_open_positions()
        await self.setup_private_ws()
        await self._cache_all_symbol_meta()
        
        # [ИЗМЕНЕНИЕ] Запускаем проактивный сканер для Golden Setup
        asyncio.create_task(self._golden_setup_screener_loop())
        
        logger.info(f"Бот для пользователя {self.user_id} полностью готов к работе.")

    async def _golden_setup_screener_loop(self):
        """
        [НОВАЯ ФУНКЦИЯ] Раз в 60 секунд проходит по "горячему списку"
        в СЛУЧАЙНОМ порядке и ищет Golden Setup.
        """
        await self.shared_ws.ready_event.wait()
        logger.info(f"Проактивный сканер Golden Setup запущен для пользователя {self.user_id}.")
        while True:
            await asyncio.sleep(60)
            
            try:
                watchlist_set = self.shared_ws.watchlist
                if not watchlist_set:
                    continue

                mode = self.strategy_mode
                if mode not in ("full", "golden_only", "golden_squeeze"):
                    continue

                # Преобразуем set в list и перемешиваем для справедливости
                watchlist_list = list(watchlist_set)
                random.shuffle(watchlist_list)
                
                logger.debug(f"Начинаю сканирование {len(watchlist_list)} монет в случайном порядке...")

                # Асинхронно проверяем все монеты из перемешанного списка
                tasks = [strategies.golden_strategy(self, symbol) for symbol in watchlist_list]
                await asyncio.gather(*tasks)

            except Exception as e:
                logger.error(f"Ошибка в цикле сканера Golden Setup: {e}", exc_info=True)


    # async def stop(self):
    #     logger.info(f"Остановка бота для пользователя {self.user_id}...")
    #     for symbol in list(self.watch_tasks.keys()):
    #         task = self.watch_tasks.pop(symbol, None)
    #         if task and not task.done():
    #             task.cancel()
    #     logger.info(f"Бот для пользователя {self.user_id} остановлен.")

    async def _sync_position_mode(self):
        try:
            logger.info("Синхронизация режима позиций с биржей...")
            resp = await asyncio.to_thread(
                lambda: self.session.get_positions(category="linear", symbol="BTCUSDT")
            )
            mode = resp.get("result", {}).get("list", [{}])[0].get("positionIdx", 0)
            self.position_mode = int(mode)
            if self.position_mode == 0:
                logger.info("Режим позиций определен: One-Way Mode (positionIdx=0).")
            else:
                logger.info("Режим позиций определен: Hedge Mode (positionIdx=1 для Buy, 2 для Sell).")
        except Exception as e:
            logger.error(f"Не удалось определить режим позиций: {e}. Будет использоваться режим по умолчанию One-Way (positionIdx=0).")
            self.position_mode = 0

    async def run_high_frequency_strategies(self, symbol: str):
        await strategies.high_frequency_dispatcher(self, symbol)

    async def run_low_frequency_strategies(self, symbol: str):
        await strategies.low_frequency_dispatcher(self, symbol)

    async def on_liquidation_event(self, event: dict):
        symbol = event.get("symbol")
        if not symbol: return
        price = utils.safe_to_float(event.get("price"))
        size = utils.safe_to_float(event.get("size"))
        value_usd = price * size
        if value_usd <= 0: return
        self.liq_buffers[symbol].append({
            "ts": time.time(), "side": event.get("side"),
            "price": price, "value": value_usd,
        })

    def _load_trade_counters_from_history(self):
        try:
            if not config.TRADES_UNIFIED_CSV_PATH.exists():
                logger.info("Файл истории сделок не найден. Счетчики начинаются с нуля.")
                return
            df = pd.read_csv(config.TRADES_UNIFIED_CSV_PATH)
            open_trades = df[df['event'] == 'open'].copy()
            open_trades['strategy_key'] = open_trades['source'].apply(
                lambda x: 'squeeze' if 'squeeze' in str(x) else ('golden_setup' if 'golden' in str(x) else 'other')
            )
            counts = open_trades['strategy_key'].value_counts().to_dict()
            self.trade_counters['squeeze'] = counts.get('squeeze', 0)
            self.trade_counters['golden_setup'] = counts.get('golden_setup', 0)
            logger.info(f"Счетчики сделок восстановлены из истории: {dict(self.trade_counters)}")
        except Exception as e:
            logger.error(f"Не удалось восстановить счетчики сделок из истории: {e}", exc_info=True)

    def _should_allow_trade(self, source: str) -> tuple[bool, str]:
        squeeze_count = self.trade_counters.get('squeeze', 0)
        golden_count = self.trade_counters.get('golden_setup', 0)
        total_trades = squeeze_count + golden_count
        if total_trades < 10:
            return True, f"Начальный период ({total_trades}/10)."
        current_signal_key = 'squeeze' if 'squeeze' in source.lower() else 'golden_setup'
        TARGET_SQUEEZE_RATIO = 0.7
        TARGET_GOLDEN_RATIO = 0.3
        LEEWAY_FACTOR = 0.2 
        if current_signal_key == 'squeeze':
            golden_ratio = golden_count / total_trades
            if golden_ratio < (TARGET_GOLDEN_RATIO * LEEWAY_FACTOR):
                return True, f"Разрешаем сквиз, так как доля Golden Setup ({golden_ratio:.0%}) критически мала."
            squeeze_ratio = squeeze_count / total_trades
            if squeeze_ratio >= TARGET_SQUEEZE_RATIO:
                return False, f"Доля сквизов ({squeeze_ratio:.0%}) >= цели ({TARGET_SQUEEZE_RATIO:.0%})."
        elif current_signal_key == 'golden_setup':
            squeeze_ratio = squeeze_count / total_trades
            if squeeze_ratio < (TARGET_SQUEEZE_RATIO * LEEWAY_FACTOR):
                return True, f"Разрешаем Golden, так как доля Squeeze ({squeeze_ratio:.0%}) критически мала."
            golden_ratio = golden_count / total_trades
            if golden_ratio >= TARGET_GOLDEN_RATIO:
                return False, f"Доля Golden Setup ({golden_ratio:.0%}) >= цели ({TARGET_GOLDEN_RATIO:.0%})."
        return True, "Пропорция в допустимых пределах."

    async def _entry_guard(self, symbol: str, side: str, candidate: dict | None = None, features: dict | None = None) -> tuple[bool, str]:
        cfg = self.user_data.get("entry_guard_settings", config.ENTRY_GUARD)
        now = time.time()
        cd_key = (symbol, side)
        if now < self.momentum_cooldown_until.get(cd_key, 0.0):
            left = int(self.momentum_cooldown_until[cd_key] - now)
            return False, f"cooldown {left}s"
        if not features:
            features = await self.extract_realtime_features(symbol)
        if not features:
            return True, "no_features"
        pct1m = float(features.get("pct1m", 0.0))
        pct5m = float(features.get("pct5m", 0.0))
        spread = float(features.get("spread_pct", 0.0))
        dOI1m = float(features.get("dOI1m", 0.0))
        dOI5m = float(features.get("dOI5m", 0.0))
        CVD1m = float(features.get("CVD1m", 0.0))
        CVD5m = float(features.get("CVD5m", 0.0))
        if spread > cfg.get("MAX_SPREAD_PCT", 0.25):
            return False, f"spread {spread:.2f}% > {cfg['MAX_SPREAD_PCT']:.2f}%"
        pump1 = cfg.get("PUMP_BLOCK_1M_PCT", 1.2)
        pump5 = cfg.get("PUMP_BLOCK_5M_PCT", 3.0)
        dump1 = cfg.get("DUMP_BLOCK_1M_PCT", 1.2)
        dump5 = cfg.get("DUMP_BLOCK_5M_PCT", 3.0)
        req_cvd = bool(cfg.get("REQUIRE_CVD_ALIGNMENT", True))
        req_oi = bool(cfg.get("REQUIRE_OI_ALIGNMENT", True))
        def aligned_up():
            ok_cvd = (CVD1m > 0 or CVD5m > 0) if req_cvd else True
            ok_oi = (dOI1m > 0 or dOI5m > 0) if req_oi else True
            return ok_cvd and ok_oi
        def aligned_down():
            ok_cvd = (CVD1m < 0 or CVD5m < 0) if req_cvd else True
            ok_oi = (dOI1m < 0 or dOI5m < 0) if req_oi else True
            return ok_cvd and ok_oi
        if side == "Sell":
            if (pct1m > pump1 or pct5m > pump5) and aligned_up():
                self.momentum_cooldown_until[cd_key] = now + cfg.get("MOMENTUM_COOLDOWN_SEC", 90)
                return False, "anti-chase-pump"
        else:
            if (pct1m < -dump1 or pct5m < -dump5) and aligned_down():
                self.momentum_cooldown_until[cd_key] = now + cfg.get("MOMENTUM_COOLDOWN_SEC", 90)
                return False, "anti-chase-dump"
        # --- НАЧАЛО ИЗМЕНЕНИЙ (ФИНАЛЬНАЯ ВЕРСИЯ ФИЛЬТРА) ---
        
        # Теперь мы не требуем откат в 0.4%, а только проверяем,
        # что цена не продолжает агрессивно двигаться против нас.
        min_retrace = 0.0 # Было cfg.get("MIN_RETRACE_FROM_EXTREME_PCT", 0.4)
        
        last_price = features.get("price")
        extreme_price = None
        last5 = list(self.shared_ws.candles_data.get(symbol, []))[-5:]
        if last5:
            if side == "Sell":
                extreme_price = max(utils.safe_to_float(c.get("highPrice")) for c in last5)
            else:
                extreme_price = min(utils.safe_to_float(c.get("lowPrice")) for c in last5)
        
        if last_price and extreme_price and extreme_price > 0:
            if side == "Sell":
                retrace = (extreme_price - last_price) / extreme_price * 100.0
                if retrace < min_retrace:
                    # Эта блокировка сработает, только если цена УСТАНОВИЛА НОВЫЙ МАКСИМУМ (retrace < 0)
                    return False, f"retrace low ({retrace:.2f}% < {min_retrace:.2f}%)"
            else: # side == "Buy"
                retrace = (last_price - extreme_price) / extreme_price * 100.0
                if retrace < min_retrace:
                    # Эта блокировка сработает, только если цена УСТАНОВИЛА НОВЫЙ МИНИМУМ (retrace < 0)
                    return False, f"retrace low ({retrace:.2f}% < {min_retrace:.2f}%)"
                    
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            
        return True, "ok"


    async def reload_settings_loop(self, interval: int = 15):
        last_known_config = self.user_data.copy()
        while True:
            await asyncio.sleep(interval)
            try:
                with open(config.USER_STATE_FILE, 'r', encoding="utf-8") as f:
                    all_configs = json.load(f)
                new_config = all_configs.get(str(self.user_id))
                if new_config and new_config != last_known_config:
                    logger.info(f"Обнаружены новые настройки для пользователя {self.user_id}. Применяю...")
                    self.user_data = new_config
                    self.apply_user_settings()
                    last_known_config = new_config.copy()
                    logger.info("Настройки успешно применены онлайн.")
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Ошибка при онлайн-перезагрузке настроек: {e}")

    async def sync_open_positions_loop(self, interval: int = 30):
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

    async def execute_trade_entry(self, candidate: dict, features: dict):
        symbol = candidate.get("symbol")
        side = candidate.get("side")
        source = candidate.get("source", "N/A")
        source_comment = candidate.get("justification", candidate.get("source", "N/A"))
        async with self.pending_orders_lock:
            if symbol in self.open_positions or symbol in self.pending_orders:
                logger.warning(f"[EXECUTE_SKIP] Позиция по {symbol} уже существует или в процессе открытия. Вход отменен.")
                return
            volume_to_open = self.POSITION_VOLUME
            effective_total_vol = await self.get_effective_total_volume()
            if effective_total_vol + volume_to_open > self.MAX_TOTAL_VOLUME:
                logger.warning(f"[EXECUTE_REJECT] Превышен лимит общего объема. Текущий: {effective_total_vol:.2f}, Попытка: {volume_to_open:.2f}, Лимит: {self.MAX_TOTAL_VOLUME:.2f}")
                return
            self.pending_orders[symbol] = volume_to_open
            self.pending_timestamps[symbol] = time.time()
        try:
            qty = await self._calc_qty_from_usd(symbol, volume_to_open)
            if qty <= 0:
                raise ValueError("Рассчитан нулевой или отрицательный объем.")
            logger.info(f"🚀 [EXECUTION] Исполнение входа: {symbol} {side}, Qty: {qty:.4f}")
            await self.place_unified_order(
                symbol=symbol, side=side, qty=qty, 
                order_type="Market", comment=source_comment
            )
            self.pending_strategy_comments[symbol] = source
            self.last_entry_ts[symbol] = time.time()
        except Exception as e:
            logger.error(f"[EXECUTE_CRITICAL] Критическая ошибка при исполнении входа для {symbol}: {e}", exc_info=True)
            async with self.pending_orders_lock:
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)

    async def execute_flea_trade(self, candidate: dict):
        symbol = candidate.get("symbol")
        side = candidate.get("side")
        source = candidate.get("source", "flea_scalp")
        async with self.pending_orders_lock:
            if symbol in self.open_positions or symbol in self.pending_orders:
                return
            cfg = self.user_data.get("flea_settings", config.FLEA_STRATEGY)
            volume_to_open = cfg.get("POSITION_USDT", 100.0)
            self.pending_orders[symbol] = volume_to_open
            self.pending_timestamps[symbol] = time.time()
            self.pending_strategy_comments[symbol] = source
        try:
            qty = await self._calc_qty_from_usd(symbol, volume_to_open)
            if qty <= 0: raise ValueError("Рассчитан нулевой объем.")
            logger.info(f"🦟🚀 [FLEA_EXECUTION] Этап 1: Открытие {symbol} {side}, Qty: {qty:.4f}")
            response = await self.place_unified_order(
                symbol=symbol, side=side, qty=qty, order_type="Market", comment="Flea Scalp Entry"
            )
            order_id = response.get("result", {}).get("orderId")
            if not order_id: raise ValueError("Не удалось получить ID ордера.")
            await asyncio.sleep(0.5) 
            tp_price = candidate.get('take_profit_price')
            sl_price = candidate.get('stop_loss_price')
            if tp_price or sl_price:
                logger.info(f"🦟⚙️ [FLEA_EXECUTION] Этап 2: Установка TP/SL для {symbol}")
                await self.set_or_amend_stop_loss(
                    symbol=symbol, new_stop_price=sl_price, take_profit_price=tp_price
                )
            else:
                logger.info(f"🦟 [{symbol}] Сделка открыта без немедленной установки TP/SL.")
        except Exception as e:
            logger.error(f"🦟💥 [FLEA_CRITICAL] Критическая ошибка при исполнении входа для {symbol}: {e}", exc_info=True)
        finally:
            async with self.pending_orders_lock:
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)

    def _get_trailing_params(self) -> tuple[float, float]:
        default_start = 5.0
        default_gap = 2.5
        user_settings = self.user_data or {}
        mode = user_settings.get("strategy_mode", "full")
        start_map = user_settings.get("trailing_start_pct", {})
        gap_map = user_settings.get("trailing_gap_pct", {})
        start_roi = start_map.get(mode, start_map.get("full", default_start))
        gap_roi = gap_map.get(mode, gap_map.get("full", default_gap))
        return float(start_roi), float(gap_roi)

    def _resolve_avg_price(self, symbol: str, pos: dict) -> float:
        avg = utils.safe_to_float(pos.get("avg_price") or pos.get("entry_price"))
        if avg > 0:
            return avg
        pend_exec = self.pending_open_exec.get(symbol)
        if pend_exec and pend_exec.get("side") == pos.get("side"):
            avg_from_pend = utils.safe_to_float(pend_exec.get("price"))
            if avg_from_pend > 0:
                pos["avg_price"] = avg_from_pend
                return avg_from_pend
        return 0.0

    async def place_unified_order(self, symbol: str, side: str, qty: float, order_type: str, **kwargs):
        cid = kwargs.get("cid") or utils.new_cid()
        comment = kwargs.get("comment", "")
        if comment:
            self.pending_strategy_comments[symbol] = comment
        if self.mode == "demo":
            pos_idx = 0
        elif self.position_mode == 0:
            pos_idx = 0
        else:
            pos_idx = 1 if side == "Buy" else 2
        params = {
            "category":"linear", "symbol":symbol, "side":side, "orderType":order_type,
            "qty": f"{qty:.12f}".rstrip("0").rstrip("."), "timeInForce":"GTC",
            "positionIdx": pos_idx, "orderLinkId": cid
        }
        if order_type == "Limit" and (price := kwargs.get("price")) is not None:
            params["price"] = str(price)
        logger.info(f"➡️ [ORDER_SENDING][{cid}] {utils.j(params)}")
        try:
            resp = await asyncio.to_thread(self.session.place_order, **params)
            order_id = resp.get("result", {}).get("orderId", "")
            logger.info(f"✅ [ORDER_ACCEPTED][{cid}] {symbol} id={order_id or 'n/a'}")
            return resp
        except InvalidRequestError as e:
            error_text = str(e)
            if "(ErrCode: 110100)" in error_text:
                logger.warning(f"❌ [ORDER_REJECTED][{cid}] {symbol} не торгуется. Блокирую на 24 часа.")
                self.failed_orders[symbol] = time.time() + 86400 
            else:
                logger.error(f"💥 [ORDER_API_FAIL][{cid}] {symbol}: {error_text}")
            raise
        except Exception as e:
            logger.error(f"💥 [ORDER_CRITICAL_FAIL][{cid}] {symbol}: {e}", exc_info=True)
            raise

    @async_retry(max_retries=5, delay=3)
    async def update_open_positions(self):
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
                if pos and pos.get("is_opening") and exec_data.get("side") == pos.get("side"):
                    exec_price = utils.safe_to_float(exec_data.get("execPrice"))
                    if exec_price > 0:
                        pos["avg_price"] = exec_price
                        pos["source"] = self.pending_strategy_comments.pop(symbol, "unknown")
                        pos["comment"] = pos["source"]
                        pos.pop("is_opening")
                        logger.info(f"[EXECUTION_OPEN] {pos['side']} {symbol} {pos['volume']:.3f} @ {exec_price:.6f}")
                        await self.log_trade(
                            symbol=symbol, side=pos['side'], avg_price=exec_price,
                            volume=pos['volume'], action="open", result="opened",
                            comment=pos['comment'], source=pos['source']
                        )
                        if symbol not in self.watch_tasks:
                            source = pos.get("source", "")
                            if 'flea' in source:
                                task = asyncio.create_task(self._guardian_flea_strategy(symbol))
                            else:
                                task = asyncio.create_task(self._guardian_standard_strategy(symbol))
                            self.watch_tasks[symbol] = task
                    continue
                if pos and exec_data.get("side") != pos.get("side"):
                    if utils.safe_to_float(exec_data.get("leavesQty", 0)) == 0:
                        entry_price = self._resolve_avg_price(symbol, pos)
                        if entry_price <= 0:
                            logger.warning(f"[{symbol}] Не удалось определить цену входа для закрытой позиции. PnL не будет рассчитан.")
                            self._purge_symbol_state(symbol)
                            continue
                        exit_price = utils.safe_to_float(exec_data.get("execPrice"))
                        pos_volume = utils.safe_to_float(pos.get("volume", 0))
                        pnl_usdt = utils.calc_pnl(pos.get("side", "Buy"), entry_price, exit_price, pos_volume)
                        leverage = utils.safe_to_float(pos.get("leverage", 1.0)) or 1.0
                        position_margin = (entry_price * pos_volume) / leverage
                        pnl_pct = (pnl_usdt / position_margin) * 100 if position_margin > 0 else 0.0
                        logger.info(f"[EXECUTION_CLOSE] {symbol}. PnL: {pnl_usdt:.2f} USDT ({pnl_pct:.3f}%).")
                        await self.log_trade(
                            symbol=symbol, side=pos['side'], avg_price=exit_price, volume=pos_volume,
                            action="close", result="closed_by_execution", pnl_usdt=pnl_usdt,
                            pnl_pct=pnl_pct, comment=pos.get('comment'), source=pos.get("source", "unknown")
                        )
                        self._purge_symbol_state(symbol)
                elif not pos and exec_data.get("execPrice"):
                    self.pending_open_exec[symbol] = {
                        "price": utils.safe_to_float(exec_data.get("execPrice")),
                        "side": exec_data.get("side"), "ts": time.time()
                    }

    async def handle_position_update(self, msg: dict):
        async with self.position_lock:
            for p in msg.get("data", []):
                symbol = p.get("symbol")
                if not symbol: continue
                new_size = utils.safe_to_float(p.get("size", 0))
                is_new_pos = symbol not in self.open_positions and new_size > 0
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
                            comment=pos["comment"], source=pos.get("source", "unknown")
                        )
                        if symbol not in self.watch_tasks:
                            task = asyncio.create_task(self.manage_open_position(symbol))
                            self.watch_tasks[symbol] = task
                elif symbol in self.open_positions and new_size == 0:
                    logger.debug(f"[PositionStream] {symbol} size=0. Закрытие будет обработано execution handler.")

    # --- НАЧАЛО ИЗМЕНЕНИЙ ---
    async def setup_private_ws(self):
        try:
            def _on_private(msg):
                try:
                    self.monitor.record_message(self.private_ws_name)
                    if not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(self.route_private_message(msg), self.loop)
                except Exception as e:
                    if "'NoneType' object has no attribute 'sock'" not in str(e):
                        logger.warning(f"PrivateWS callback error: {e}")
            
            # РЕКОМЕНДАЦИЯ: Измените параметры ping
            ping_interval = 20  # Рекомендуемый интервал - 20 секунд
            ping_timeout = 10   # Таймаут ответа
            
            self.ws_private = WebSocket(
                testnet=False, demo=self.mode == "demo", channel_type="private",
                api_key=self.api_key, api_secret=self.api_secret,
                ping_interval=ping_interval, ping_timeout=ping_timeout, 
                restart_on_error=True,
                retries=200
            )
            self.ws_private.position_stream(callback=_on_private)
            self.ws_private.execution_stream(callback=_on_private)
            
            self.monitor.register_connection(self.private_ws_name)
            mode_info = " (демо-режим)" if self.mode == "demo" else ""
            logger.info(f"Private WebSocket для user {self.user_id} запущен{mode_info} в фоновом режиме.")
            
            # УБРАН БЛОКИРУЮЩИЙ ВЫЗОВ:
            # await asyncio.Event().wait() <--- ЭТО БЫЛО НЕПРАВИЛЬНО

        except asyncio.CancelledError:
            logger.info(f"Private WS task для user {self.user_id} отменен.")
            if self.ws_private: self.ws_private.exit()
        except Exception as e:
            logger.error(f"Критическая ошибка при запуске Private WS для user {self.user_id}: {e}", exc_info=True)
            if self.ws_private: self.ws_private.exit()

    async def stop(self):
        """Переименованный и доработанный метод stop для полноты."""
        logger.info(f"Остановка бота для пользователя {self.user_id}...")
        if hasattr(self, 'ws_private') and self.ws_private:
            self.ws_private.exit()
            
        for symbol in list(self.watch_tasks.keys()):
            task = self.watch_tasks.pop(symbol, None)
            if task and not task.done():
                task.cancel()
        logger.info(f"Бот для пользователя {self.user_id} остановлен.")
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---


    async def route_private_message(self, msg):
        topic = (msg.get("topic") or "").lower()
        if "position" in topic:
            await self.handle_position_update(msg)
        elif "execution" in topic:
            await self.handle_execution(msg)

    async def _initiate_hunt(self, candidate: dict, features: dict, signal_key: tuple):
        source = candidate.get("source", "")
        symbol = candidate.get("symbol")
        if 'squeeze' in source or 'liquidation' in source:
            logger.info(f"🎯 [{symbol}] Активирована тактическая группа 'Сталкер' для поиска разворота.")
            asyncio.create_task(self._hunt_squeeze_reversal(candidate, features, signal_key))
        elif 'golden_setup' in source:
            logger.info(f"🎯 [{symbol}] Активирована тактическая группа 'Прорыватель' для подтверждения пробоя.")
            asyncio.create_task(self._hunt_golden_breakout(candidate, features, signal_key))
        else:
            logger.warning(f"[{symbol}] Для источника '{source}' не найден 'Охотник'. Исполнение сразу.")
            await self.execute_trade_entry(candidate, features)
            self.active_signals.discard(signal_key)

# bot_core.py

    async def _process_signal(self, candidate: dict, features: dict, signal_key: tuple):
        """
        [ФИНАЛЬНАЯ ВЕРСИЯ] Обрабатывает входящий торговый сигнал, используя
        многоступенчатый конвейер с "экспресс-линией" для сквиз-сигналов.
        """
        symbol = candidate.get("symbol")
        source = candidate.get("source", "")
        original_side = candidate.get("side")
        try:
            # --- Этап 1: Проверка пропорций (для всех сигналов) ---
            should_proceed, reason_prop = self._should_allow_trade(source)
            if not should_proceed:
                logger.info(f"⚖️ [{symbol}] Сигнал '{source}' пропущен для соблюдения пропорции. Причина: {reason_prop}")
                self.active_signals.discard(signal_key)
                return

            # --- Этап 2: Разделение потоков (Главный логический форк) ---

            # Сквиз и Ликвидации идут по "Экспресс-линии" напрямую к AI, минуя "Сторожа"
            if 'squeeze' in source or 'liquidation' in source:
                logger.debug(f"[{symbol}] Сигнал '{source}' использует экспресс-линию, минуя 'Сторожа'.")
            
            # Все остальные сигналы (например, golden_setup) идут по "Стандартному пути"
            else:
                ok, reason_guard = await self._entry_guard(symbol, original_side, candidate, features)
                if not ok:
                    # --- Здесь работает "Оппортунист" для заблокированных стандартных сигналов ---
                    use_trend_chance = self.user_data.get("USE_TREND_CHANCE", False)
                    if reason_guard in ("anti-chase-pump", "anti-chase-dump") and use_trend_chance:
                        pct_30m = utils.compute_pct(self.shared_ws.candles_data.get(symbol, []), 30)
                        OVERHEAT_THRESHOLD_PCT = 5.0
                        if abs(pct_30m) > OVERHEAT_THRESHOLD_PCT:
                            logger.warning(f"🔥 [{symbol}] 'Оппортунист' ОТКЛОНИЛ вход. Рынок 'перегрет'. Движение за 30м: {pct_30m:.2f}% (Порог: {OVERHEAT_THRESHOLD_PCT}%)")
                            self.active_signals.discard(signal_key)
                            return
                        
                        adx_val = features.get('adx14', 0.0)
                        min_adx_threshold = self.user_data.get("OPPORTUNIST_MIN_ADX", 25.0)
                        if adx_val < min_adx_threshold:
                            logger.warning(f"📉 [{symbol}] 'Оппортунист' ОТКЛОНИЛ вход. Сила тренда недостаточна. ADX: {adx_val:.1f} (Порог: >{min_adx_threshold})")
                            self.active_signals.discard(signal_key)
                            return
                        
                        new_side = "Buy" if reason_guard == "anti-chase-pump" else "Sell"
                        candidate["side"] = new_side
                        candidate["source"] = f"trend_chance_from_{source}"
                        logger.warning(f"💡 [{symbol}] 'Оппортунист' перехватил сигнал! Контртренд {original_side} заблокирован, но тренд сильный (ADX: {adx_val:.1f}). Входим в {new_side}!")
                        await self.execute_trade_entry(candidate, features)
                        self.active_signals.discard(signal_key)
                        return
                    else:
                        logger.info(f"🛡️ [{symbol}] Сигнал заблокирован 'Сторожем': {reason_guard}")
                        self.active_signals.discard(signal_key)
                        return

            # --- Этап 3: Финальные проверки перед AI (для сигналов, прошедших свои пути) ---
            if 'golden_setup' in source:
                side = candidate.get("side")
                pct_30m = utils.compute_pct(self.shared_ws.candles_data.get(symbol, []), 30)
                features['pct_30m'] = pct_30m
                REJECTION_THRESHOLD = 7.0
                is_overheated = (side == "Buy" and pct_30m > REJECTION_THRESHOLD) or \
                                (side == "Sell" and pct_30m < -REJECTION_THRESHOLD)
                if is_overheated:
                    reason_gs = f"Отклонено кодом: вход в {side} после движения {pct_30m:.2f}% за 30 мин."
                    logger.info(f"🔥 [{symbol}] GOLDEN СИГНАЛ ОТКЛОНЕН (ПЕРЕГРЕТОСТЬ). {reason_gs}")
                    self.active_signals.discard(signal_key)
                    return

            # --- Этап 4: AI-Аналитик (Финальное одобрение) ---
            prompt = ai_ml.build_primary_prompt(candidate, features, self.shared_ws)
            logger.debug(f"Сигнал {signal_key} передан AI-аналитику...")
            ai_response = await ai_ml.ask_ollama_json(
                self.ai_advisor_model,
                [{"role": "user", "content": prompt}],
                self.ai_timeout_sec,
                self.ollama_advisor_openai
            )
            
            # --- Этап 5: Отправка в тактическую группу или отказ ---
            action = ai_response.get("action", "REJECT").upper()
            if action == "EXECUTE":
                logger.info(f"✅ [{symbol}] Сигнал ОДОБРЕН AI ({self.ai_advisor_model}). Причина: {ai_response.get('justification')}. Передано тактической группе.")
                candidate['justification'] = ai_response.get('justification')
                await self._initiate_hunt(candidate, features, signal_key)
            else:
                logger.info(f"❌ [{symbol}] Сигнал ОТКЛОНЕН AI ({self.ai_advisor_model}). Причина: {ai_response.get('justification')}")
                self.active_signals.discard(signal_key)

        except Exception as e:
            logger.error(f"Критическая ошибка в _process_signal для {signal_key}: {e}", exc_info=True)
            self.active_signals.discard(signal_key)
        finally:
            if symbol:
                self.strategy_cooldown_until[symbol] = time.time() + 60

# bot_core.py

    async def _hunt_squeeze_reversal(self, candidate: dict, features: dict, signal_key: tuple):
        """
        [НОВАЯ ВЕРСИЯ] "Сталкер" как "Охотник за Пиками".
        Он отслеживает самый экстремум импульса и входит на первом откате от него.
        """
        symbol = candidate["symbol"]
        side = candidate["side"]
        initial_rsi = features.get('rsi14', 50.0)
        
        # Запоминаем цену из первоначального сигнала как стартовый экстремум
        extreme_price = features.get('price', 0.0)
        if extreme_price == 0:
            logger.error(f"💥 [СТАЛКЕР] {symbol}/{side}: Не удалось получить начальную цену. Охота отменена.")
            self.active_signals.discard(signal_key)
            return

        start_time = time.time()
        logger.info(f"🏹 [{symbol}] 'Охотник за Пиками' активирован. Начальный экстремум: {extreme_price:.6f}")

        while time.time() - start_time < self.tactical_entry_window_sec:
            try:
                current_features = await self.extract_realtime_features(symbol)
                if not current_features:
                    await asyncio.sleep(self.squeeze_ai_confirm_interval_sec)
                    continue

                # --- ЛОГИКА ОХОТЫ ЗА ПИКОМ ---
                current_price = current_features.get('price', 0.0)
                
                # Если цена продолжает идти против нас, мы обновляем экстремум - это ХОРОШО!
                if side == 'Sell' and current_price > extreme_price:
                    logger.debug(f"🏹 [{symbol}] Новый пик обнаружен: {current_price:.6f}")
                    extreme_price = current_price
                elif side == 'Buy' and current_price < extreme_price:
                    logger.debug(f"🏹 [{symbol}] Новое дно обнаружено: {current_price:.6f}")
                    extreme_price = current_price

                # Передаем в скоринг всегда самый последний, самый "выгодный" экстремум
                score, reasons = self._calculate_squeeze_reversal_score(
                    side, initial_rsi, extreme_price, current_features
                )
                
                ENTRY_SCORE_THRESHOLD = 60 # Порог остается 60
                if score >= ENTRY_SCORE_THRESHOLD:
                    reasons_str = ', '.join(reasons)
                    logger.info(f"✅ [ОХОТНИК ЗА ПИКАМИ] {symbol}/{side}: Цель захвачена! Счет: {score}. Причины: {reasons_str}")
                    
                    # Исполняем сделку напрямую, без повторной проверки "Сторожем"
                    await self.execute_trade_entry(candidate, current_features)
                    return # Успешно выходим
                    
                await asyncio.sleep(self.squeeze_ai_confirm_interval_sec)
                
            except Exception as e:
                logger.error(f"💥 [ОХОТНИК ЗА ПИКАМИ] {symbol}/{side}: Критическая ошибка в цикле: {e}", exc_info=True)
                break
                
        logger.warning(f"⏳ [ОХОТНИК ЗА ПИКАМИ] {symbol}/{side}: Окно входа истекло, откат от пика не подтвержден.")
        self.active_signals.discard(signal_key)

    async def _hunt_golden_breakout(self, candidate: dict, features: dict, signal_key: tuple):
        symbol = candidate["symbol"]
        side = candidate["side"]
        reference_price = features.get("price", 0.0)
        BREAKOUT_CONFIRM_PCT = 0.3
        start_time = time.time()
        while time.time() - start_time < self.tactical_entry_window_sec:
            try:
                current_features = await self.extract_realtime_features(symbol)
                if not current_features:
                    await asyncio.sleep(2)
                    continue
                last_price = current_features.get("price")
                cvd_1m = current_features.get("CVD1m", 0)
                vol_anomaly = current_features.get("volume_anomaly", 1.0)
                price_change_pct = ((last_price - reference_price) / reference_price) * 100.0
                price_confirmed = (side == "Buy" and price_change_pct >= BREAKOUT_CONFIRM_PCT) or \
                                  (side == "Sell" and price_change_pct <= -BREAKOUT_CONFIRM_PCT)
                flow_confirmed = (side == "Buy" and cvd_1m > 0) or \
                                 (side == "Sell" and cvd_1m < 0)
                volume_confirmed = vol_anomaly > 1.5
                if price_confirmed and flow_confirmed and volume_confirmed:
                    logger.info(f"✅ [ПРОРЫВАТЕЛЬ] {symbol}/{side}: Пробой подтвержден! ΔЦена: {price_change_pct:.2f}%, CVD: {cvd_1m:,.0f}, Vol x{vol_anomaly:.1f}")
                    ok, reason = await self._entry_guard(symbol, side, features=current_features)
                    if not ok:
                        logger.warning(f"❌ [ПРОРЫВАТЕЛЬ] Вход в последний момент заблокирован 'Сторожем': {reason}")
                        break
                    await self.execute_trade_entry(candidate, current_features)
                    return
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.error(f"💥 [ПРОРЫВАТЕЛЬ] {symbol}/{side}: Критическая ошибка в цикле: {e}", exc_info=True)
                break
        logger.warning(f"⏳ [ПРОРЫВАТЕЛЬ] {symbol}/{side}: Окно входа истекло, пробой не подтвержден.")
        self.active_signals.discard(signal_key)

    def _calculate_squeeze_reversal_score(self, side: str, initial_rsi: float, peak_price: float, current_features: dict) -> tuple[int, list]:
        score = 0
        reasons = []
        current_price = current_features.get('price', 0.0)
        current_rsi = current_features.get('rsi14', 50.0)
        cvd_1m = current_features.get('CVD1m', 0.0)
        vol_1m = current_features.get('vol1m', 0.0)
        avg_vol_30m = current_features.get('avgVol30m', 1.0)
        funding_rate = current_features.get('funding_rate', 0.0)
        if side == 'Sell' and initial_rsi > 70 and current_rsi < 85:
            score += 35
            reasons.append(f"RSI cross < 85 ({current_rsi:.1f})")
        elif side == 'Buy' and initial_rsi < 15 and current_rsi > 20:
            score += 35
            reasons.append(f"RSI cross > 20 ({current_rsi:.1f})")
        if peak_price > 0:
            pullback_pct = abs(current_price - peak_price) / peak_price * 100
            if pullback_pct > 0.15:
                score += 25
                reasons.append(f"Pullback {pullback_pct:.2f}%")
        if (side == 'Sell' and cvd_1m < 0) or (side == 'Buy' and cvd_1m > 0):
            score += 20
            reasons.append(f"CVD Confirm ({cvd_1m:,.0f})")
        if avg_vol_30m > 0 and vol_1m < avg_vol_30m:
            score += 15
            reasons.append(f"Volume Exhaustion (vol {vol_1m:,.0f} < avg {avg_vol_30m:,.0f})")
        HOT_FUNDING_THRESHOLD = 0.04
        if (side == 'Sell' and funding_rate >= HOT_FUNDING_THRESHOLD) or \
           (side == 'Buy' and funding_rate <= -HOT_FUNDING_THRESHOLD):
            score += 10
            reasons.append(f"Hot Funding ({funding_rate*100:.4f}%)")
        return score, reasons

    async def execute_priority_trade(self, candidate: dict):
        """
        [НОВЫЙ МЕТОД] "Инсайдер". Исполняет элитные сигналы немедленно,
        в обход стандартных проверок.
        """
        symbol = candidate.get("symbol")
        side = candidate.get("side")
        source = candidate.get("source", "N/A")
        
        logger.warning(f"⚡️ [{symbol}] 'ИНСАЙДЕР' АКТИВИРОВАН! Сверхсильный сигнал '{source}'. Немедленное исполнение.")
        
        async with self.pending_orders_lock:
            if symbol in self.open_positions or symbol in self.pending_orders:
                logger.warning(f"[INSIDER_SKIP] Позиция по {symbol} уже существует. Вход отменен.")
                return

            volume_to_open = self.POSITION_VOLUME
            effective_total_vol = await self.get_effective_total_volume()
            if effective_total_vol + volume_to_open > self.MAX_TOTAL_VOLUME:
                logger.warning(f"[INSIDER_REJECT] Превышен лимит общего объема.")
                return
            
            self.pending_orders[symbol] = volume_to_open
            self.pending_timestamps[symbol] = time.time()

        try:
            qty = await self._calc_qty_from_usd(symbol, volume_to_open)
            if qty <= 0:
                raise ValueError("Рассчитан нулевой объем.")
            
            await self.place_unified_order(
                symbol=symbol, side=side, qty=qty, 
                order_type="Market", comment=f"Insider Signal: {source}"
            )
            self.pending_strategy_comments[symbol] = source
            self.last_entry_ts[symbol] = time.time()

        except Exception as e:
            logger.error(f"[INSIDER_CRITICAL] Критическая ошибка при исполнении приоритетного входа для {symbol}: {e}", exc_info=True)
            async with self.pending_orders_lock:
                self.pending_orders.pop(symbol, None)
                self.pending_timestamps.pop(symbol, None)



    async def manage_open_position(self, symbol: str):
        logger.info(f"🛡️ [Guardian] Активирован для позиции {symbol}.")
        try:
            pos = self.open_positions.get(symbol)
            if not pos: return
            source = pos.get("source", "")
            if 'flea_scalp' in source:
                cfg = self.user_data.get("flea_settings", config.FLEA_STRATEGY)
                max_hold_sec = cfg.get("MAX_HOLD_MINUTES", 10) * 60
                logger.info(f"🦟 [{symbol}] 'Хранитель' активирован для 'Блохи'. Таймер на {max_hold_sec / 60:.1f} мин.")
                await asyncio.sleep(max_hold_sec)
                if symbol in self.open_positions:
                    logger.warning(f"⏰ [{symbol}] 'Блоха' превысила лимит удержания. Принудительное закрытие.")
                    await self.close_position(symbol, reason="Flea time limit exceeded")
                return
            if not pos.get("initial_stop_set"):
                await self._set_initial_stop_loss(symbol, pos)
            is_squeeze = 'squeeze' in source.lower()
            if is_squeeze:
                HOPE_TIMEOUT_SEC = 10 * 60
                start_time = time.time()
                while time.time() - start_time < HOPE_TIMEOUT_SEC:
                    current_pos = self.open_positions.get(symbol)
                    if not current_pos: return
                    last_price = utils.safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
                    avg_price = self._resolve_avg_price(symbol, current_pos)
                    if last_price > 0 and avg_price > 0:
                        side = current_pos.get("side", "Buy")
                        pnl = ((last_price / avg_price) - 1.0) if side == "Buy" else ((avg_price / last_price) - 1.0)
                        if pnl >= 0:
                            logger.info(f"[{symbol}] Позиция вышла в плюс. 'Таймер надежды' отключен.")
                            break
                    await asyncio.sleep(30)
                else:
                    logger.warning(f"[{symbol}] 'Таймер надежды' истек, позиция в убытке. Принудительное закрытие.")
                    await self.close_position(symbol, reason="Hope timer expired")
                    return
            while symbol in self.open_positions:
                await asyncio.sleep(300)
        except asyncio.CancelledError:
            logger.info(f"[Guardian] Наблюдение за {symbol} отменено.")
        except Exception as e:
            logger.error(f"[Guardian] {symbol} критическая ошибка: {e}", exc_info=True)
        finally:
            logger.info(f"🛡️ [Guardian] Завершает наблюдение за {symbol}.")

    async def _watch_flea_position(self, symbol: str):
        cfg = self.user_data.get("flea_settings", config.FLEA_STRATEGY)
        max_hold_sec = cfg.get("MAX_HOLD_MINUTES", 10) * 60
        logger.info(f"🦟 [{symbol}] 'Смотритель Блохи' активирован. Таймер на {max_hold_sec / 60:.1f} мин.")
        await asyncio.sleep(max_hold_sec)
        if symbol in self.open_positions:
            logger.warning(f"⏰ [{symbol}] 'Блоха' превысила лимит удержания. Принудительное закрытие.")
            await self.close_position(symbol, reason="Flea time limit exceeded")

    async def close_position(self, symbol: str, reason: str = "Forced close"):
        pos = self.open_positions.get(symbol)
        if not pos:
            logger.warning(f"[{symbol}] Попытка закрыть позицию, но она не найдена.")
            return
        try:
            side = pos.get("side")
            close_side = "Sell" if side == "Buy" else "Buy"
            qty = pos.get("volume")
            logger.info(f"🚨 [{symbol}] Отправка рыночного ордера на принудительное закрытие. Side: {close_side}, Qty: {qty}")
            await self.place_unified_order(
                symbol=symbol, side=close_side, qty=qty,
                order_type="Market", comment=reason
            )
        except Exception as e:
            logger.error(f"[{symbol}] Ошибка при принудительном закрытии позиции: {e}", exc_info=True)

    async def _ai_advise_on_position(self, symbol: str):
        try:
            pos = self.open_positions.get(symbol)
            if not pos: return
            features = await self.extract_realtime_features(symbol)
            if not features:
                logger.warning(f"[{symbol}] Не удалось получить фичи для AI-советника.")
                return
            pos['last_stop_price'] = self.last_stop_price.get(symbol)
            pos['current_roi'] = 0.0
            avg_price = self._resolve_avg_price(symbol, pos)
            last_price = features.get("price", avg_price)
            if avg_price > 0:
                side = pos.get("side", "Buy")
                leverage = utils.safe_to_float(pos.get("leverage", 10.0))
                pnl = ((last_price / avg_price) - 1.0) if side == "Buy" else ((avg_price / last_price) - 1.0)
                pos['current_roi'] = pnl * 100.0 * leverage
            prompt = ai_ml.build_position_management_prompt(symbol, pos, features)
            logger.info(f"🤖 [{symbol}] Запрос комплексного совета у AI-менеджера...")
            ai_response = await ai_ml.ask_ollama_json(
                self.ai_advisor_model, 
                [{"role": "user", "content": prompt}], 
                timeout_s=45.0,
                base_url=self.ollama_advisor_openai
            )
            action = ai_response.get("action", "").upper()
            reason = ai_response.get("reason", "N/A")
            if action == "ADJUST_SL":
                new_price = utils.safe_to_float(ai_response.get("new_stop_price"))
                if new_price > 0:
                    logger.info(f"🤖✅ [{symbol}] AI РЕКОМЕНДОВАЛ скорректировать стоп на {new_price:.6f}. Причина: {reason}")
                    await self.set_or_amend_stop_loss(new_price, symbol=symbol)
            if action == "SET_DYNAMIC_TP_AND_TRAIL":
                tp_price = utils.safe_to_float(ai_response.get("take_profit_price"))
                new_gap_pct = utils.safe_to_float(ai_response.get("new_trailing_gap_pct"))
                if tp_price > 0 and new_gap_pct > 0:
                    logger.info(f"🤖💡 [{symbol}] AI обновил цели: TP={tp_price:.6f}, Trailing Gap={new_gap_pct:.2f}%. Причина: {reason}")
                    pos['dynamic_gap_pct'] = new_gap_pct
                    pos['target_tp_price'] = tp_price
                    await self.set_or_amend_stop_loss(0, symbol=symbol, take_profit_price=tp_price)
            elif action == "SET_BREAKEVEN_TP":
                tp_price = utils.safe_to_float(ai_response.get("take_profit_price"))
                if tp_price > 0:
                    logger.info(f"🤖✅ [{symbol}] AI РЕКОМЕНДОВАЛ установить защитный TP в {tp_price:.6f}. Причина: {reason}")
                    await self.set_or_amend_stop_loss(0, symbol=symbol, take_profit_price=tp_price)
            else:
                logger.info(f"🤖 HOLD [{symbol}] AI-менеджер рекомендует не вносить изменений. Причина: {reason}")
        except Exception as e:
            logger.error(f"[{symbol}] Ошибка в работе AI-менеджера: {e}", exc_info=True)

    async def _calculate_fibonacci_stop_price(self, symbol: str, side: str) -> Optional[float]:
        try:
            LOOKBACK_MINUTES = 180
            FIB_LEVEL = 0.618
            candles = list(self.shared_ws.candles_data.get(symbol, []))
            if len(candles) < LOOKBACK_MINUTES:
                return None
            recent_candles = candles[-LOOKBACK_MINUTES:]
            highest_high = max(utils.safe_to_float(c.get("highPrice")) for c in recent_candles)
            lowest_low = min(utils.safe_to_float(c.get("lowPrice")) for c in recent_candles)
            price_range = highest_high - lowest_low
            if price_range == 0: return None
            if side.lower() == "buy":
                fib_level = lowest_low + (price_range * FIB_LEVEL)
                return fib_level
            else:
                fib_level = highest_high - (price_range * FIB_LEVEL)
                return fib_level
        except Exception as e:
            logger.error(f"[{symbol}] Ошибка при расчете уровня Фибоначчи: {e}")
            return None

    async def _ai_advise_on_stop(self, symbol: str):
        try:
            pos = self.open_positions.get(symbol)
            if not pos:
                return
            features = await self.extract_realtime_features(symbol)
            if not features:
                logger.warning(f"[{symbol}] Не удалось получить фичи для AI-советника.")
                return
            pos['last_stop_price'] = self.last_stop_price.get(symbol)
            prompt = ai_ml.build_position_management_prompt(symbol, pos, features)
            messages = [{"role": "user", "content": prompt}]
            logger.info(f"🤖 [{symbol}] Запрос совета у AI-риск-менеджера...")
            ai_response = await ai_ml.ask_ollama_json(
                self.ai_advisor_model, 
                messages, 
                timeout_s=45.0,
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
        try:
            avg_price = self._resolve_avg_price(symbol, pos)
            if avg_price <= 0: return
            side = str(pos.get("side", "")).lower()
            stop_price = None
            source_comment = pos.get("comment", "") 
            is_squeeze = 'squeeze' in source_comment.lower()
            if is_squeeze:
                candles = list(self.shared_ws.candles_data.get(symbol, []))
                if candles:
                    impulse_candle = candles[-1]
                    high_price = utils.safe_to_float(impulse_candle.get("highPrice"))
                    low_price = utils.safe_to_float(impulse_candle.get("lowPrice"))
                    candle_range = high_price - low_price
                    STOP_RANGE_MULTIPLIER = 3.0
                    if candle_range > 0:
                        if side == "buy":
                            stop_price = avg_price - (candle_range * STOP_RANGE_MULTIPLIER)
                        else:
                            stop_price = avg_price + (candle_range * STOP_RANGE_MULTIPLIER)
                        logger.info(f"[{symbol}] Расчет стопа по сквиз-свече: High={high_price:.6f}, Low={low_price:.6f}, Range={candle_range:.6f}. Финальный стоп: {stop_price:.6f}")
            if not stop_price:
                if is_squeeze:
                    logger.warning(f"[{symbol}] Не удалось рассчитать стоп по сквиз-свече, используется фоллбэк.")
                max_stop_pct = utils.safe_to_float(self.user_data.get("max_safety_stop_pct", 2.5)) / 100.0
                if side == "buy":
                    stop_price = avg_price * (1 - max_stop_pct)
                else:
                    stop_price = avg_price * (1 + max_stop_pct)
            if (side == "buy" and stop_price >= avg_price) or \
               (side == "sell" and stop_price <= avg_price):
                logger.error(f"[{symbol}] КРИТИЧЕСКАЯ ЛОГИЧЕСКАЯ ОШИБКА! ...")
            logger.info(f"🛡️ [{symbol}] Установка первоначального стопа. Цена входа: {avg_price:.6f}, Финальный стоп: {stop_price:.6f}")
            await self.set_or_amend_stop_loss(stop_price, symbol=symbol)
        except Exception as e:
            logger.error(f"[{symbol}] Критическая ошибка при установке первоначального стопа: {e}", exc_info=True)

    async def set_or_amend_stop_loss(self, new_stop_price: float, *, symbol: str, take_profit_price: Optional[float] = None):
        pos = self.open_positions.get(symbol)
        if not pos:
            logger.debug(f"[{symbol}] Попытка установить SL/TP для уже отсутствующей позиции. Пропуск.")
            return
        try:
            side = str(pos.get("side", "")).lower()
            tick = float(self.price_tick_map.get(symbol, 1e-6) or 1e-6)
            if self.mode == "demo":
                pos_idx = 0
            elif self.position_mode == 0:
                pos_idx = 0
            else:
                pos_idx = 1 if side == "buy" else 2
            params = {"category": "linear", "symbol": symbol, "positionIdx": pos_idx}
            if new_stop_price and new_stop_price > 0:
                if side == "buy":
                    stop_price = math.floor(new_stop_price / tick) * tick
                else:
                    stop_price = math.ceil(new_stop_price / tick) * tick
                last_known_stop = self.last_stop_price.get(symbol)
                if last_known_stop is None or abs(stop_price - last_known_stop) > 1e-9:
                    params["stopLoss"] = f"{stop_price:.8f}".rstrip("0").rstrip(".")
            if take_profit_price and take_profit_price > 0:
                if side == "buy":
                    tp_price = math.ceil(take_profit_price / tick) * tick
                else:
                    tp_price = math.floor(take_profit_price / tick) * tick
                params["takeProfit"] = f"{tp_price:.8f}".rstrip("0").rstrip(".")
            if "stopLoss" not in params and "takeProfit" not in params:
                return
            logger.info(f"⚙️ [{symbol}] Отправка команды SL/TP: {params}")
            response = await asyncio.to_thread(lambda: self.session.set_trading_stop(**params))
            if response.get("retCode") == 0:
                log_msg = f"✅ [{symbol}] API подтвердил:"
                if "stopLoss" in params:
                    self.last_stop_price[symbol] = float(params["stopLoss"])
                    log_msg += f" SL={params['stopLoss']}"
                if "takeProfit" in params:
                    pos["tp_set"] = True
                    log_msg += f" TP={params['takeProfit']}"
                logger.info(log_msg)
        except InvalidRequestError as e:
            err_str = str(e).lower()
            if "not modified" in err_str or "34040" in err_str:
                logger.info(f"[{symbol}] SL/TP уже установлен на этом уровне.")
            elif "position does not exist" in err_str or "can not set tp/sl/ts for zero position" in err_str:
                logger.warning(f"[{symbol}] Ошибка API: Позиция не найдена. Очистка состояния.")
                if symbol in self.open_positions: self._purge_symbol_state(symbol)
            elif "should lower than" in err_str or "should be higher than" in err_str:
                logger.warning(f"[{symbol}] API отклонил стоп из-за рыночного движения. Это нормально.")
            else:
                logger.error(f"[{symbol}] Необработанная ошибка API: {e}")
        except Exception as e:
            logger.error(f"[{symbol}] Критическая ошибка в set_or_amend_stop_loss: {e}", exc_info=True)

    def _purge_symbol_state(self, symbol: str):
        logger.debug(f"Полная очистка состояния для символа: {symbol}")
        if task := self.watch_tasks.pop(symbol, None):
            if not task.done():
                task.cancel()
                logger.debug(f"[{symbol}] Guardian task отменен.")
        self.open_positions.pop(symbol, None)
        self.last_stop_price.pop(symbol, None)
        self.pending_orders.pop(symbol, None)
        self.recently_closed[symbol] = time.time()
        self.trailing_activated.pop(symbol, None)
        self.trailing_activation_ts.pop(symbol, None)
        self.take_profit_price.pop(symbol, None)

    async def _cleanup_recently_closed(self, interval: int = 15, max_age: int = 60):
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            expired = [s for s, ts in self.recently_closed.items() if now - ts > max_age]
            for s in expired:
                self.recently_closed.pop(s, None)

    async def on_ticker_update(self, symbol: str, last_price: float):
        # --- НАЧАЛО ИЗМЕНЕНИЙ (ТРОТТЛИНГ) ---
        now = time.time()
        # Запускаем ВЧ-стратегии для одного символа не чаще, чем раз в 2 секунды
        if now - self.last_hf_check_ts[symbol] < 2.0:
            return
        self.last_hf_check_ts[symbol] = now
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

        pos = self.open_positions.get(symbol)
        if not pos:
            await strategies.high_frequency_dispatcher(self, symbol)
            return
        source = pos.get("source", "")
        if 'flea_scalp' in source:
            return
        if pos.get("is_closing") or pos.get("is_updating_sl"):
            return
        avg_price = self._resolve_avg_price(symbol, pos)
        if avg_price <= 0: return
        side = pos.get("side", "Buy")
        leverage = utils.safe_to_float(pos.get("leverage", 10.0)) or 10.0
        pnl = ((last_price / avg_price) - 1.0) if side == "Buy" else ((avg_price / last_price) - 1.0)
        current_roi = pnl * 100.0 * leverage
        start_roi_pct, gap_roi_pct = self._get_trailing_params()
        if not pos.get("trailing_activated") and current_roi >= start_roi_pct:
            pos["trailing_activated"] = True
            pos["trailing_activation_ts"] = time.time()
            logger.info(f"✅ [{symbol}] ТРЕЙЛИНГ АКТИВИРОВАН! ROI: {current_roi:.2f}%.")
            features = await self.extract_realtime_features(symbol)
            atr15m = features.get("atr15m", 0.0)
            if atr15m > 0:
                ATR_MULTIPLIER = 2.5
                tp_price = last_price + (atr15m * ATR_MULTIPLIER) if side == "Buy" else last_price - (atr15m * ATR_MULTIPLIER)
                logger.info(f"🎯 [{symbol}] Рассчитан предиктивный TP на {tp_price:.6f}")
                await self.set_or_amend_stop_loss(0, symbol=symbol, take_profit_price=tp_price)
        if pos.get("trailing_activated"):
            if not pos.get("breakeven_stop_set"):
                BREAKEVEN_BUFFER_ROI = 1.0 
                if current_roi >= start_roi_pct + BREAKEVEN_BUFFER_ROI:
                    tick = float(self.price_tick_map.get(symbol, 1e-6))
                    breakeven_price = avg_price - tick if side == "Sell" else avg_price + tick
                    logger.info(f"🛡️ [{symbol}] ROI ({current_roi:.2f}%) достиг порога Б/У. Перемещаем стоп на {breakeven_price:.6f}")
                    try:
                        pos["is_updating_sl"] = True
                        await self.set_or_amend_stop_loss(breakeven_price, symbol=symbol)
                        pos["breakeven_stop_set"] = True
                    finally:
                        pos["is_updating_sl"] = False
                return
            if pos.get("breakeven_stop_set"):
                target_roi = current_roi - gap_roi_pct
                denom = 1.0 + (target_roi / (100.0 * leverage))
                new_stop_price = (avg_price * denom) if side == "Buy" else (avg_price / denom if denom > 1e-9 else 0.0)
                if new_stop_price > 0:
                    if (side == "Buy" and new_stop_price >= last_price) or \
                       (side == "Sell" and new_stop_price <= last_price):
                        new_stop_price = 0
                if new_stop_price > 0:
                    prev_stop = self.last_stop_price.get(symbol)
                    is_better = prev_stop is None or \
                                (side == "Buy" and new_stop_price > prev_stop) or \
                                (side == "Sell" and new_stop_price < prev_stop)
                    tick_size = float(self.price_tick_map.get(symbol, 1e-8))
                    is_change_significant = prev_stop is None or abs(new_stop_price - prev_stop) > (tick_size * 5)
                    if is_better and is_change_significant:
                        try:
                            pos["is_updating_sl"] = True
                            await self.set_or_amend_stop_loss(new_stop_price, symbol=symbol)
                        finally:
                            pos["is_updating_sl"] = False

    async def get_total_open_volume(self) -> float:
        total = 0.0
        for pos in self.open_positions.values():
            size = utils.safe_to_float(pos.get("volume", 0))
            price = utils.safe_to_float(pos.get("markPrice", 0)) or utils.safe_to_float(pos.get("avg_price", 0))
            total += size * price
        return total

    async def get_effective_total_volume(self) -> float:
        open_vol = await self.get_total_open_volume()
        pending_vol = sum(self.pending_orders.values())
        return open_vol + pending_vol

    @async_retry(max_retries=5, delay=3)
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

    def load_ml_models(self):
        self.ml_inferencer = ai_ml.MLXInferencer()

    def _extract_realtime_features_sync(self, symbol: str) -> Optional[Dict[str, float]]:
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---

        # ПЕРЕМЕЩАЕМ ОПРЕДЕЛЕНИЕ ФУНКЦИИ В САМОЕ НАЧАЛО МЕТОДА
        def _safe_last(series, default):
            if series is None or not isinstance(series, pd.Series) or series.empty:
                return default
            try:
                v = series.iloc[-1]
                return v if pd.notna(v) else default
            except IndexError:
                return default

        # --- КОНЕЦ ИЗМЕНЕНИЙ ---

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
                s = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
                df[col] = s.ffill().bfill()
        n = len(df)
        close = df["closePrice"] if n else pd.Series(dtype="float64")
        high  = df["highPrice"]  if n else pd.Series(dtype="float64")
        low   = df["lowPrice"]   if n else pd.Series(dtype="float64")

        # Вычисляем индикаторы, необходимые для flea_strategy
        flea_cfg = self.user_data.get("flea_settings", config.FLEA_STRATEGY)
        fast_ema_len = flea_cfg.get("FAST_EMA_PERIOD", 5)
        slow_ema_len = flea_cfg.get("SLOW_EMA_PERIOD", 10)
        trend_ema_len = flea_cfg.get("TREND_EMA_PERIOD", 200)
        
        fast_ema_val = _safe_last(ta.ema(close, length=fast_ema_len), 0.0) if n >= fast_ema_len else 0.0
        slow_ema_val = _safe_last(ta.ema(close, length=slow_ema_len), 0.0) if n >= slow_ema_len else 0.0
        trend_ema_val = _safe_last(ta.ema(close, length=trend_ema_len), 0.0) if n >= trend_ema_len else 0.0
        
        # Добавляем предыдущие значения EMA для проверки пересечения
        fast_ema_prev = _safe_last(ta.ema(close, length=fast_ema_len).shift(1), 0.0) if n > fast_ema_len else 0.0
        slow_ema_prev = _safe_last(ta.ema(close, length=slow_ema_len).shift(1), 0.0) if n > slow_ema_len else 0.0

        # Вычисляем данные для golden_strategy
        avg_volume_prev_4m = 0
        if len(candles) >= 5:
            avg_volume_prev_4m = np.mean([utils.safe_to_float(c.get("volume", 0)) for c in candles[-5:-1]])


        def _safe_last(series, default):
            if series is None or not isinstance(series, pd.Series) or series.empty:
                return default
            try:
                v = series.iloc[-1]
                return v if pd.notna(v) else default
            except IndexError:
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
        candles_15m = self._aggregate_candles_15m(candles)
        df_15m = pd.DataFrame(candles_15m)
        atr15m = 0.0
        if len(df_15m) >= 15:
            high15 = pd.to_numeric(df_15m["highPrice"])
            low15 = pd.to_numeric(df_15m["lowPrice"])
            close15 = pd.to_numeric(df_15m["closePrice"])
            atr_series_15 = ta.atr(high15, low15, close15, length=14)
            if atr_series_15 is not None and not atr_series_15.empty:
                atr15m = _safe_last(atr_series_15, 0.0)
        candles_1h = self._aggregate_candles_60m(candles)
        trend_h1 = 0
        if len(candles_1h) > 2:
            if candles_1h[-1]['closePrice'] > candles_1h[-2]['closePrice']:
                trend_h1 = 1
            elif candles_1h[-1]['closePrice'] < candles_1h[-2]['closePrice']:
                trend_h1 = -1
        funding_snap = self._funding_snapshot(symbol, tdata) 
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
            "hour_of_day": hour_of_day, "day_of_week": day_of_week, "month_of_year": month_of_year, "atr15m": atr15m,
            "trend_h1": trend_h1, "funding_rate": funding_snap.get("funding_rate", 0.0), "fast_ema": fast_ema_val,
            "slow_ema": slow_ema_val, "trend_ema": trend_ema_val, "fast_ema_prev": fast_ema_prev, "slow_ema_prev": slow_ema_prev,
            "avg_volume_prev_4m": avg_volume_prev_4m,

        }
        for k in config.FEATURE_KEYS:
            features.setdefault(k, 0.0)
        return features


    async def extract_realtime_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        АСИНХРОННАЯ обертка для неблокирующего вызова с ограничением параллелизма.
        """
        # --- НАЧАЛО ИЗМЕНЕНИЙ ---
        async with self.feature_extraction_sem:
            # Теперь только 4 задачи смогут одновременно выполнять код ниже.
            # Остальные будут ждать, не блокируя event loop.
            return await asyncio.to_thread(self._extract_realtime_features_sync, symbol)
        # --- КОНЕЦ ИЗМЕНЕНИЙ ---


    async def _get_golden_thresholds(self, symbol: str, side: str) -> dict:
        base = (
            self.golden_param_store.get((symbol, side))
            or self.golden_param_store.get(side)
            or {"period_iters": 3, "price_change": 1.7,
                "volume_change": 200, "oi_change": 1.5}
        )
        return base

    def _aggregate_candles_5m(self, candles: any) -> list:
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
        try:
            if not minute_candles:
                return []
            m1_needed = lookback * 5
            tail = minute_candles[-m1_needed:] if len(minute_candles) >= m1_needed else minute_candles[:]
            bars_5m = []
            for i in range(0, len(tail), 5):
                chunk = tail[i:i+5]
                if len(chunk) < 5:
                    break
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

    def _aggregate_candles_15m(self, candles: list) -> list:
        if not candles: return []
        result = []
        num_candles = (len(candles) // 15) * 15
        candle_list = candles[-num_candles:]
        for i in range(0, len(candle_list), 15):
            chunk = candle_list[i : i + 15]
            if len(chunk) < 15: continue
            result.append({
                "openPrice": utils.safe_to_float(chunk[0]["openPrice"]),
                "highPrice": max(utils.safe_to_float(c["highPrice"]) for c in chunk),
                "lowPrice": min(utils.safe_to_float(c["lowPrice"]) for c in chunk),
                "closePrice": utils.safe_to_float(chunk[-1]["closePrice"]),
                "volume": sum(utils.safe_to_float(c["volume"]) for c in chunk),
            })
        return result

    def _aggregate_candles_60m(self, candles: list) -> list:
        if not candles: return []
        result = []
        num_candles = (len(candles) // 60) * 60
        candle_list = candles[-num_candles:]
        for i in range(0, len(candle_list), 60):
            chunk = candle_list[i : i + 60]
            if len(chunk) < 60: continue
            result.append({
                "openPrice": utils.safe_to_float(chunk[0]["openPrice"]),
                "highPrice": max(utils.safe_to_float(c["highPrice"]) for c in chunk),
                "lowPrice": min(utils.safe_to_float(c["lowPrice"]) for c in chunk),
                "closePrice": utils.safe_to_float(chunk[-1]["closePrice"]),
                "volume": sum(utils.safe_to_float(c["volume"]) for c in chunk),
            })
        return result

    def _build_squeeze_features_5m(self, symbol: str):
        try:
            bars = self._aggregate_ohlcv_5m(list(self.shared_ws.candles_data.get(symbol, []))[-75:])
            if len(bars) < 15:
                return None, None
            df = pd.DataFrame(bars)
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
        cooldown_period_sec = 300
        last_signal_time = self._last_golden_ts.get(symbol, 0)
        return (time.time() - last_signal_time) > cooldown_period_sec

    def _squeeze_allowed(self, symbol: str) -> bool:
        cooldown_period_sec = 600
        last_signal_time = self.last_squeeze_ts.get(symbol, 0)
        return (time.time() - last_signal_time) > cooldown_period_sec

    async def notify_user(self, text: str):
        if not telegram_bot: return
        try:
            await telegram_bot.send_message(self.user_id, text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.warning(f"Не удалось отправить сообщение пользователю {self.user_id}: {e}")

    async def log_trade(self, **kwargs):
        symbol = kwargs.get("symbol")
        side = str(kwargs.get("side", "")).capitalize()
        action = str(kwargs.get("action", "")).lower()
        source = str(kwargs.get("source", "unknown"))
        result = str(kwargs.get("result", "")).lower()
        avg_price = utils.safe_to_float(kwargs.get("avg_price"))
        volume = utils.safe_to_float(kwargs.get("volume"))
        pnl_usdt = kwargs.get("pnl_usdt")
        pnl_pct = kwargs.get("pnl_pct")
        comment = kwargs.get("comment")
        time_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        pnl_info = f" | PnL: {pnl_usdt:.2f}$ ({pnl_pct:.2f}%)" if pnl_usdt is not None else ""
        logger.info(f"[LOG_TRADE] user={self.user_id} {action.upper()} {symbol}: side={side}, vol={volume}, price={avg_price}, result={result}{pnl_info}")
        try:
            base_row = {
                "timestamp": dt.datetime.utcnow().isoformat(), "symbol": symbol, "side": side,
                "event": action, "result": result, "volume_trade": volume, "price_trade": avg_price,
                "pnl_usdt": pnl_usdt, "pnl_pct": pnl_pct, "source": source
            }
            extended_metrics = await self._dataset_metrics(symbol)
            base_row.update(extended_metrics)
            utils._append_trades_unified(base_row)
            if 'flea_scalp' in source:
                if action == "open":
                    self.flea_positions_count += 1
                elif action == "close":
                    self.flea_positions_count = max(0, self.flea_positions_count - 1)
                logger.info(f"🦟 Счетчик позиций 'Блохи': {self.flea_positions_count}")
            elif action == "open":
                strategy_key = 'squeeze' if 'squeeze' in source.lower() else ('golden_setup' if 'golden' in source.lower() else None)
                if strategy_key:
                    self.trade_counters[strategy_key] += 1
                    logger.info(f"Счетчики основных стратегий обновлены: {dict(self.trade_counters)}")
            if action == "open":
                strategy_key = None
                if 'squeeze' in source.lower():
                    strategy_key = 'squeeze'
                elif 'golden' in source.lower():
                    strategy_key = 'golden_setup'
                if strategy_key:
                    self.trade_counters[strategy_key] += 1
                    logger.info(f"Счетчики сделок обновлены: {dict(self.trade_counters)}")
        except Exception as e:
            logger.warning(f"Ошибка записи в trades_unified.csv: {e}")
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
        await self.ensure_symbol_meta(symbol)
        step_str = str(self.qty_step_map.get(symbol, "0.001"))
        min_qty_str = str(self.min_qty_map.get(symbol, step_str))
        p = price or utils.safe_to_float(self.shared_ws.ticker_data.get(symbol, {}).get("lastPrice"))
        if not p > 0:
            logger.error(f"[{symbol}] Не удалось получить актуальную цену для расчета количества.")
            return 0.0
        try:
            d_usd_amount = Decimal(str(usd_amount))
            d_price = Decimal(str(p))
            d_step = Decimal(step_str)
            d_min_qty = Decimal(min_qty_str)
            if d_price == 0: return 0.0
            raw_qty = d_usd_amount / d_price
            ticks = (raw_qty / d_step).quantize(Decimal('1'), rounding='ROUND_DOWN')
            final_qty = ticks * d_step
            if final_qty < d_min_qty:
                logger.warning(f"[{symbol}] Расчетный объем {final_qty} меньше минимального {d_min_qty}. Для ордера будет использован минимальный объем.")
                final_qty = d_min_qty
            return float(final_qty)
        except Exception as e:
            logger.error(f"[{symbol}] Критическая ошибка при расчете количества: {e}", exc_info=True)
            return 0.0

    @async_retry(max_retries=5, delay=3)
    async def ensure_symbol_meta(self, symbol: str):
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
        
    async def listing_age_minutes(self, symbol: str) -> float:
        now = time.time()
        cached_data = _listing_age_cache.get(symbol)
        if cached_data and (now - cached_data[1] < 3600):
            return cached_data[0]
        async with _listing_sem:
            try:
                resp = await asyncio.to_thread(
                    lambda: self.session.get_instruments_info(category="linear", symbol=symbol)
                )
                info = resp["result"]["list"][0]
                launch_ms = utils.safe_to_float(info.get("launchTime", 0))
                if launch_ms <= 0:
                    raise ValueError("launchTime missing or invalid")
                age_min = (now * 1000 - launch_ms) / 60000.0
            except Exception as e:
                logger.warning(f"Не удалось определить возраст для {symbol}: {e}. Считаем ее 'старой'.")
                age_min = 999_999.0
            _listing_age_cache[symbol] = (age_min, now)
            return age_min
        
    def _apply_funding_to_features(self, symbol: str, features: dict) -> dict:
        snap = self._funding_snapshot(symbol, features)
        features.update(snap)
        return snap

    def _apply_funding_to_candidate(self, candidate: dict, funding_snap: dict) -> None:
        fm = {
            "funding_rate": funding_snap.get("funding_rate"),
            "funding_bucket": funding_snap.get("funding_bucket"),
        }
        if "base_metrics" in candidate:
            candidate["base_metrics"].update(fm)
        else:
            candidate["base_metrics"] = fm

    def _funding_snapshot(self, symbol: str, features: dict | None = None) -> dict:
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
        features = await self.extract_realtime_features(symbol)
        if not features:
            return {}
        vol_1m = features.get('vol1m', 0)
        avg_vol_30m = features.get('avgVol30m', 1)
        vol_anomaly = vol_1m / avg_vol_30m if avg_vol_30m > 0 else 1.0
        metrics = {
            "price": features.get("price", 0.0),
            "open_interest": features.get("OI_now", 0.0),
            "volume_1m": vol_1m,
            "rsi14": features.get("rsi14", 0.0),
            "adx14": features.get("adx14", 0.0),
            "volume_anomaly": vol_anomaly
        }
        return metrics
